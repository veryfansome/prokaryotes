"""Compaction lifecycle: pre_tail filter on working files, CAS swap, parent marking, relabel-target write.

The compactor carries `working_file_windows` forward from the **live Redis snapshot at CAS time** (not the
deep-copy `snapshot` taken at compaction start), filtered so that windows whose originating file-tool `call_id`
lives in the pre-tail `TurnExecution.items` are dropped. The keep buckets are recency-tail-minted windows
(call_id in recency-tail TurnExecutions), post-snapshot-turn-minted windows (call_id in a TurnExecution that may
not have been written yet at CAS time, but never in pre_tail), and carryforward windows (call_id nowhere in
current TurnExecutions). This filter is race-safe by construction: `pre_tail_turns` is loaded in
`_prepare_compaction` well before CAS, and `finalize_turn` writes the Redis conversation before the
TurnExecution — so a post-snapshot turn's call_id can be invisible to the CAS-time TurnExecution read while its
window is already visible in `current.working_file_windows`. The pre_tail bucket is well-known, so the
"NOT in pre_tail" check is unaffected.

After commit, the parent is marked `is_compacted=true` (its `messages_json` is retained — the DAG-scoped
guardrail requires walking the parent chain after compaction), and `compaction_status:{pending_snapshot_uuid}`
is written to Redis with the child's `snapshot_uuid` as the relabel target.

The summarization step is invoked via `compact_fn`; the input is built by `HarnessBase._summarize_and_compact`,
which projects `pre_tail_conv` with `working_file_windows=[]` and `ancestor_summaries=[]` so live file bodies and
prior summaries don't fossilize into the new summary.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from redis.exceptions import WatchError

from prokaryotes.context_v1.conversation_sync import ConversationSyncer
from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    TurnExecution,
    WorkingFileWindow,
    compute_boundary_hash,
    compute_tail_hash,
)
from prokaryotes.search_v1.conversations import (
    COMPACTION_STATE_COMMITTED,
    COMPACTION_STATE_PENDING,
)
from prokaryotes.utils_v1.llm_utils import COMPACTION_RECENCY_TAIL

logger = logging.getLogger(__name__)

_COMPACTION_SEARCH_WRITE_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2)
_NO_RELABEL_SENTINEL = ""


@dataclass
class _CompactionPrep:
    """Internal handoff between snapshot and CAS swap."""

    pre_tail_messages: list[ConversationMessage]
    recency_tail_messages: list[ConversationMessage]
    tail_offset: int
    pre_tail_turns: dict[str, TurnExecution]
    recency_tail_turns: dict[str, TurnExecution]


class ConversationCompactor(ConversationSyncer):
    """Compaction lifecycle. Inherits the three-tier load + apply machinery so the CAS check can re-read the active
    Conversation and verify the prefix hasn't changed since the snapshot."""

    async def _compact_conversation(
        self,
        *,
        compact_fn: Callable[[Conversation, _CompactionPrep], Coroutine[Any, Any, str]],
        conversation_uuid: str,
        lock_key: str,
        snapshot: Conversation,
    ) -> None:
        """Snapshot → summarize → CAS swap (with pre_tail working-file filter) → mark parent compacted.

        `compact_fn(snapshot, prep)` produces the summary text. The prep object carries the pre-tail/recency-tail
        split and loaded TurnExecutions; the summarization input projection happens in `compact_fn`.
        """
        try:
            prep = await self._prepare_compaction(snapshot)
            if not prep.pre_tail_messages:
                logger.info(
                    "Skipping compaction for snapshot %s: nothing in pre-tail",
                    snapshot.snapshot_uuid,
                )
                return

            summary = await compact_fn(snapshot, prep)
            if not summary:
                logger.warning(
                    "Compaction produced empty summary for snapshot %s",
                    snapshot.snapshot_uuid,
                )
                return

            compaction_attempt_uuid = str(uuid.uuid4())
            child_snapshot_uuid = str(uuid.uuid4())

            boundary_messages = await self._boundary_messages_for_conversation(snapshot)
            boundary_hash = compute_boundary_hash(boundary_messages)
            tail_hash = compute_tail_hash(boundary_messages, snapshot.bot_author_id)
            boundary_message_count = len(boundary_messages)

            swapped = await self._cas_swap_child(
                conversation_uuid=conversation_uuid,
                snapshot=snapshot,
                prep=prep,
                summary=summary,
                child_snapshot_uuid=child_snapshot_uuid,
                compaction_attempt_uuid=compaction_attempt_uuid,
            )

            if swapped is None:
                await self._write_compaction_status(snapshot.snapshot_uuid, _NO_RELABEL_SENTINEL)
                return

            # `refresh="wait_for"`: marking the child committed makes it findable by `find_latest_active_child`. A
            # post-compaction `_rebuild_from_chain` (cold Redis) recovers the child's working-file state through that
            # search, so the committed transition must be searchable before this background task returns. This runs
            # off the request path, so the refresh wait costs no interactive latency.
            await _retry_compaction_search_write(
                f"mark child snapshot {swapped.snapshot_uuid} committed",
                lambda: self.search_client.update_conversation(
                    swapped.snapshot_uuid,
                    compaction_state=COMPACTION_STATE_COMMITTED,
                    compaction_attempt_uuid=compaction_attempt_uuid,
                    refresh="wait_for",
                ),
            )
            await _retry_compaction_search_write(
                f"mark parent snapshot {snapshot.snapshot_uuid} compacted",
                lambda: self.search_client.update_conversation(
                    snapshot.snapshot_uuid,
                    compaction_attempt_uuid=compaction_attempt_uuid,
                    is_compacted=True,
                    summary=summary,
                    boundary_hash=boundary_hash,
                    boundary_message_count=boundary_message_count,
                    tail_hash=tail_hash,
                ),
            )
            await self._write_compaction_status(snapshot.snapshot_uuid, swapped.snapshot_uuid)
        except Exception:
            logger.exception("compact_conversation failed for snapshot %s", snapshot.snapshot_uuid)
            await self._write_compaction_status(snapshot.snapshot_uuid, _NO_RELABEL_SENTINEL)
        finally:
            await self.redis_client.delete(lock_key)

    async def _cas_swap_child(
        self,
        *,
        conversation_uuid: str,
        snapshot: Conversation,
        prep: _CompactionPrep,
        summary: str,
        child_snapshot_uuid: str,
        compaction_attempt_uuid: str,
    ) -> Conversation | None:
        redis_key = f"conversation:{conversation_uuid}"
        async with self.redis_client.pipeline() as pipe:
            while True:
                try:
                    await pipe.watch(redis_key)
                    current_data = await pipe.get(redis_key)
                    if not current_data:
                        logger.info("Skipping compaction swap: Redis conversation missing")
                        return None

                    current = Conversation.model_validate_json(current_data)
                    if current.snapshot_uuid != snapshot.snapshot_uuid:
                        logger.info(
                            "Skipping compaction swap for %s: active snapshot is now %s",
                            snapshot.snapshot_uuid,
                            current.snapshot_uuid,
                        )
                        return None
                    if (
                        current.raw_message_start_index != snapshot.raw_message_start_index
                        or current.ancestor_summaries != snapshot.ancestor_summaries
                        or not _messages_match_prefix(current.messages, snapshot.messages)
                    ):
                        logger.info(
                            "Skipping compaction swap for %s: active prefix changed",
                            snapshot.snapshot_uuid,
                        )
                        return None

                    post_snapshot_messages = current.messages[len(snapshot.messages) :]
                    pre_tail_call_ids = _file_tool_call_ids_in(prep.pre_tail_turns)
                    carried_windows = _carry_forward_windows(current.working_file_windows, pre_tail_call_ids)
                    swapped = Conversation(
                        conversation_uuid=conversation_uuid,
                        snapshot_uuid=child_snapshot_uuid,
                        parent_snapshot_uuid=snapshot.snapshot_uuid,
                        bot_author_id=snapshot.bot_author_id,
                        ancestor_summaries=snapshot.ancestor_summaries + [summary],
                        raw_message_start_index=snapshot.raw_message_start_index + prep.tail_offset,
                        messages=prep.recency_tail_messages + post_snapshot_messages,
                        working_file_windows=carried_windows,
                    )

                    async def put_child(
                        conv: Conversation = swapped,
                    ) -> None:
                        await self.search_client.put_conversation(
                            conv,
                            compaction_attempt_uuid=compaction_attempt_uuid,
                            compaction_state=COMPACTION_STATE_PENDING,
                        )

                    await _retry_compaction_search_write(
                        f"persist pending child snapshot {child_snapshot_uuid}",
                        put_child,
                    )

                    pipe.multi()
                    pipe.set(
                        redis_key,
                        swapped.model_dump_json(),
                        ex=self.conversation_cache_ex,
                    )
                    await pipe.execute()
                    return swapped
                except WatchError:
                    logger.info("WatchError on compaction swap; retrying")
                    await pipe.reset()
                    continue

    async def _prepare_compaction(self, snapshot: Conversation) -> _CompactionPrep:
        """Build the pre-tail/recency-tail split and load TurnExecutions for both windows. Pure read step.

        `pre_tail_turns` is loaded here, well before the CAS swap. The CAS swap's pre_tail-call_id working-file
        filter reads from this set only — it does **not** re-query post-snapshot TurnExecutions at CAS time, which
        would race against `finalize_turn`'s write order (Redis conversation before TurnExecution).
        """
        recency_tail, tail_offset = _recency_tail_messages(
            snapshot.messages,
            snapshot.bot_author_id,
            COMPACTION_RECENCY_TAIL,
        )
        if not recency_tail:
            return _CompactionPrep(
                pre_tail_messages=[],
                recency_tail_messages=[],
                tail_offset=0,
                pre_tail_turns={},
                recency_tail_turns={},
            )

        tail_start = len(snapshot.messages) - len(recency_tail)
        pre_tail_messages = snapshot.messages[:tail_start]

        pre_tail_bot_ids = [
            m.source_id for m in pre_tail_messages if not m.deleted and m.author_id == snapshot.bot_author_id
        ]
        recency_tail_bot_ids = [
            m.source_id for m in recency_tail if not m.deleted and m.author_id == snapshot.bot_author_id
        ]
        pre_tail_turns = await self.search_client.get_turn_executions(snapshot.conversation_uuid, pre_tail_bot_ids)
        recency_tail_turns = await self.search_client.get_turn_executions(
            snapshot.conversation_uuid, recency_tail_bot_ids
        )

        return _CompactionPrep(
            pre_tail_messages=pre_tail_messages,
            recency_tail_messages=recency_tail,
            tail_offset=tail_offset,
            pre_tail_turns=pre_tail_turns,
            recency_tail_turns=recency_tail_turns,
        )

    async def _write_compaction_status(
        self,
        pending_snapshot_uuid: str,
        value: str,
    ) -> None:
        """Write the relabel target (or sentinel) for `/compaction-status` polling.

        TTL matches `conversation_cache_ex` so a long-idle client returning to the conversation still finds the
        relabel target.
        """
        try:
            await self.redis_client.set(
                f"compaction_status:{pending_snapshot_uuid}",
                value,
                ex=self.conversation_cache_ex,
            )
        except Exception:
            logger.warning(
                "Failed to write compaction_status for %s",
                pending_snapshot_uuid,
                exc_info=True,
            )


def _carry_forward_windows(
    windows: list[WorkingFileWindow],
    pre_tail_call_ids: set[str],
) -> list[WorkingFileWindow]:
    """The CAS-time working-file carry-forward filter. Keep a window iff at least one of its `origin_call_ids`
    escapes the pre-tail span — `set(origin_call_ids) - pre_tail_call_ids` is non-empty.

    A surviving origin is a recency-tail / post-snapshot call (present, but never in pre_tail) or a compacted
    ancestor call (present nowhere in current turns). A window is dropped only when *every* origin is a pre-tail
    call. `origin_call_ids` generalizes the old `window_id`-only check now that consolidation/fold can merge
    several calls into one window with a synthetic `wfw-*` id. Extracted from `_cas_swap_child` so the predicate
    is importable and unit-testable rather than mirrored.
    """
    return [w for w in windows if set(w.origin_call_ids) - pre_tail_call_ids]


def _file_tool_call_ids_in(turns: dict[str, TurnExecution]) -> set[str]:
    """Collect file-tool `function_call.call_id`s across the given `TurnExecution`s.

    Used by the CAS-time working-file filter: a window whose `window_id` lives in this set originated in one of
    the supplied turns. With `pre_tail_turns` as input, the result is the universe of pre-compaction file-tool
    calls; windows in that universe are dropped during carry-forward.
    """
    call_ids: set[str] = set()
    for turn in turns.values():
        for item in turn.items:
            if item.type != "function_call":
                continue
            if item.name != "file_tool":
                continue
            call_id = item.call_id or item.id
            if call_id is not None:
                call_ids.add(call_id)
    return call_ids


def _messages_match_prefix(
    current: list[ConversationMessage],
    snapshot: list[ConversationMessage],
) -> bool:
    """The Conversation prefix check at the CAS swap step. Messages are append-only with `deleted`/`edited` flags,
    so prefix equality is structural."""
    if len(current) < len(snapshot):
        return False
    return current[: len(snapshot)] == snapshot


def _recency_tail_messages(
    messages: list[ConversationMessage],
    bot_author_id: str,
    tail_count: int,
) -> tuple[list[ConversationMessage], int]:
    """Split off the last `tail_count` non-deleted messages, advancing forward to skip leading bot messages so the
    tail leads with non-bot content.

    Returns `(recency_tail, tail_offset)` where `tail_offset` is the count of non-deleted messages in the pre-tail
    (used to bump child's `raw_message_start_index`).
    """
    non_deleted = [(idx, m) for idx, m in enumerate(messages) if not m.deleted]
    if not non_deleted:
        return [], 0
    first_tail_pos = max(0, len(non_deleted) - tail_count)
    while first_tail_pos < len(non_deleted) and non_deleted[first_tail_pos][1].author_id == bot_author_id:
        first_tail_pos += 1
    if first_tail_pos >= len(non_deleted):
        return [], 0
    tail_start_index = non_deleted[first_tail_pos][0]
    return messages[tail_start_index:], first_tail_pos


async def _retry_compaction_search_write(
    operation_name: str,
    action: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    total_attempts = len(_COMPACTION_SEARCH_WRITE_RETRY_DELAYS_SECONDS) + 1
    for attempt_idx in range(total_attempts):
        try:
            await action()
            return
        except Exception:
            if attempt_idx == total_attempts - 1:
                raise
            delay = _COMPACTION_SEARCH_WRITE_RETRY_DELAYS_SECONDS[attempt_idx]
            logger.warning(
                "Compaction %s failed on attempt %d/%d; retrying in %.2fs",
                operation_name,
                attempt_idx + 1,
                total_attempts,
                delay,
            )
            await asyncio.sleep(delay)
