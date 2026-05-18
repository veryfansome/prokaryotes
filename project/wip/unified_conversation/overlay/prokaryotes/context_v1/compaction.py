"""Compaction lifecycle: lift step, CAS swap, parent marking, relabel-target write.

Replaces `PartitionCompactor`. The lift step computes `lifted_turn_items` and
`lifted_anchor_source_id` for the child snapshot before summarization, then
swaps the child into Redis atomically. After commit, the parent is marked
`is_compacted=true` (its `messages_json` is retained — see design doc for the
DAG-scoped guardrail requirement), and `compaction_status:{pending_snapshot_uuid}`
is written to Redis with the child's `snapshot_uuid` as the relabel target.

The recency-tail / live-window stripping is invoked via `compact_fn`; callers
(WebHarness) build the summarization input.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from redis.exceptions import WatchError

from prokaryotes.context_v1.conversation_sync import (
    ConversationSyncer,
    _active_paths_in_turns,
)
from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    TurnExecution,
    TurnItem,
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
class _LiftPlan:
    lifted_turn_items: list[TurnItem]
    lifted_anchor_source_id: str | None


@dataclass
class _CompactionPrep:
    """Internal handoff between snapshot and CAS swap."""

    pre_tail_messages: list[ConversationMessage]
    recency_tail_messages: list[ConversationMessage]
    tail_offset: int
    pre_tail_turns: dict[str, TurnExecution]
    recency_tail_turns: dict[str, TurnExecution]
    lift_plan: _LiftPlan


class ConversationCompactor(ConversationSyncer):
    """Compaction lifecycle. Inherits the three-tier load + apply machinery so
    the CAS check can re-read the active Conversation and verify the prefix
    hasn't changed since the snapshot."""

    async def _compact_conversation(
        self,
        *,
        compact_fn: Callable[[Conversation, _CompactionPrep], Coroutine[Any, Any, str]],
        conversation_uuid: str,
        lock_key: str,
        snapshot: Conversation,
    ) -> None:
        """Snapshot → lift plan → summarize → CAS swap → mark parent compacted.

        `compact_fn(snapshot, prep)` produces the summary text. The prep object
        carries the pre-tail/recency-tail split and loaded TurnExecutions so the
        summarizer can strip live-window bodies before projection without
        re-fetching anything.
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
            boundary_user_count = sum(1 for msg in boundary_messages if msg.author_id != snapshot.bot_author_id)

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

            await _retry_compaction_search_write(
                f"mark child snapshot {swapped.snapshot_uuid} committed",
                lambda: self.search_client.update_conversation(
                    swapped.snapshot_uuid,
                    compaction_state=COMPACTION_STATE_COMMITTED,
                    compaction_attempt_uuid=compaction_attempt_uuid,
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
                    boundary_user_count=boundary_user_count,
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
                    swapped = Conversation(
                        conversation_uuid=conversation_uuid,
                        snapshot_uuid=child_snapshot_uuid,
                        parent_snapshot_uuid=snapshot.snapshot_uuid,
                        bot_author_id=snapshot.bot_author_id,
                        ancestor_summaries=snapshot.ancestor_summaries + [summary],
                        lifted_turn_items=prep.lift_plan.lifted_turn_items,
                        lifted_anchor_source_id=prep.lift_plan.lifted_anchor_source_id,
                        raw_message_start_index=snapshot.raw_message_start_index + prep.tail_offset,
                        messages=prep.recency_tail_messages + post_snapshot_messages,
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
                        f"persist child snapshot {swapped.snapshot_uuid}",
                        put_child,
                    )

                    pipe.multi()
                    pipe.set(
                        redis_key,
                        swapped.model_dump_json(),
                        ex=self.conversation_cache_ex,
                    )
                    await pipe.execute()
                    logger.info(
                        "Compaction complete: new snapshot %s (parent %s)",
                        swapped.snapshot_uuid,
                        snapshot.snapshot_uuid,
                    )
                    return swapped
                except WatchError:
                    logger.info("Compaction Redis swap contended; retrying")
                    await pipe.reset()

    async def _prepare_compaction(self, snapshot: Conversation) -> _CompactionPrep:
        """Build the pre-tail/recency-tail split, load TurnExecutions for both
        windows, and compute the lift plan. Pure read step — no writes."""
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
                lift_plan=_LiftPlan(lifted_turn_items=[], lifted_anchor_source_id=None),
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

        lift_plan = _compute_lift_plan(
            pre_tail_messages=pre_tail_messages,
            recency_tail_messages=recency_tail,
            pre_tail_turns=pre_tail_turns,
            recency_tail_turns=recency_tail_turns,
            parent_lifted=snapshot.lifted_turn_items,
            bot_author_id=snapshot.bot_author_id,
        )

        return _CompactionPrep(
            pre_tail_messages=pre_tail_messages,
            recency_tail_messages=recency_tail,
            tail_offset=tail_offset,
            pre_tail_turns=pre_tail_turns,
            recency_tail_turns=recency_tail_turns,
            lift_plan=lift_plan,
        )

    async def _write_compaction_status(
        self,
        pending_snapshot_uuid: str,
        value: str,
    ) -> None:
        """Write the relabel target (or sentinel) for `/compaction-status` polling.

        TTL matches `conversation_cache_ex` so a long-idle client returning to
        the conversation still finds the relabel target.
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


def _compute_lift_plan(
    *,
    pre_tail_messages: list[ConversationMessage],
    recency_tail_messages: list[ConversationMessage],
    pre_tail_turns: dict[str, TurnExecution],
    recency_tail_turns: dict[str, TurnExecution],
    parent_lifted: list[TurnItem],
    bot_author_id: str,
) -> _LiftPlan:
    """Determine active paths from the new raw window's TurnExecutions, lift
    pre-tail TurnItems + parent's already-lifted items whose path is active,
    and choose the anchor bot. Invariant: anchor=None iff lifted_turn_items==[]."""
    active_paths = _active_paths_in_turns(recency_tail_turns)
    if not active_paths:
        return _LiftPlan(lifted_turn_items=[], lifted_anchor_source_id=None)

    candidates: list[TurnItem] = []
    for msg in pre_tail_messages:
        if msg.deleted or msg.author_id != bot_author_id:
            continue
        turn = pre_tail_turns.get(msg.source_id)
        if turn is not None:
            candidates.extend(turn.items)
    candidates.extend(parent_lifted)

    by_call_id: dict[str, TurnItem] = {}
    for item in candidates:
        if item.type == "function_call":
            cid = item.call_id or item.id
            if cid is not None:
                by_call_id[cid] = item

    fresh_paths = _paths_freshly_read_in_window(recency_tail_turns)

    lifted: list[TurnItem] = []
    for item in candidates:
        if item.type != "function_call_output":
            continue
        ann = item.prokaryotes_annotations or {}
        if ann.get("file_tool.status") != "live":
            continue
        path = ann.get("file_tool.path")
        if path not in active_paths:
            continue
        if path in fresh_paths:
            # Superseded by a fresh read in the new raw window — let that pair represent it.
            continue
        cid = item.call_id or item.id
        if cid is None:
            continue
        function_call_item = by_call_id.get(cid)
        if function_call_item is None:
            continue
        lifted.append(function_call_item)
        lifted.append(item)

    anchor: str | None = None
    for msg in recency_tail_messages:
        if msg.deleted or msg.author_id != bot_author_id:
            continue
        turn = recency_tail_turns.get(msg.source_id)
        if turn is None:
            continue
        if any(
            (item.prokaryotes_annotations or {}).get("file_tool.path")
            and (item.prokaryotes_annotations or {}).get("file_tool.status") != "stale"
            for item in turn.items
        ):
            anchor = msg.source_id
            break

    if not lifted or anchor is None:
        return _LiftPlan(lifted_turn_items=[], lifted_anchor_source_id=None)
    return _LiftPlan(lifted_turn_items=lifted, lifted_anchor_source_id=anchor)


def _messages_match_prefix(
    current: list[ConversationMessage],
    snapshot: list[ConversationMessage],
) -> bool:
    """The Conversation prefix check at the CAS swap step. Messages are
    append-only with `deleted`/`edited` flags, so prefix equality is structural —
    no `mod_live_windows` carve-out is needed (live-window mutation lives in
    `TurnExecution` items, which the CAS step does not touch)."""
    if len(current) < len(snapshot):
        return False
    return current[: len(snapshot)] == snapshot


def _paths_freshly_read_in_window(
    turns: dict[str, TurnExecution],
) -> set[str]:
    """Paths with a live `function_call_output` in the recency tail. A prior
    lifted pair for the same path is superseded by such a fresh read; we let
    the new pair carry forward instead of duplicating."""
    fresh: set[str] = set()
    for turn in turns.values():
        for item in turn.items:
            if item.type != "function_call_output":
                continue
            ann = item.prokaryotes_annotations or {}
            if ann.get("file_tool.status") != "live":
                continue
            path = ann.get("file_tool.path")
            if path:
                fresh.add(path)
    return fresh


def _recency_tail_messages(
    messages: list[ConversationMessage],
    bot_author_id: str,
    tail_count: int,
) -> tuple[list[ConversationMessage], int]:
    """Split off the last `tail_count` non-deleted messages, advancing forward
    to skip leading bot messages so the tail leads with non-bot content.

    Returns `(recency_tail, tail_offset)` where `tail_offset` is the count of
    non-deleted messages in the pre-tail (used to bump child's
    `raw_message_start_index`).
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
                exc_info=True,
            )
            await asyncio.sleep(delay)
