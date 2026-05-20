"""`HarnessBase` — Redis + Search + compaction lifecycle.

`sync_conversation` returns a `SyncResult` carrying the snapshot plus any `source_id` assignments and resync info.
`stream_and_finalize` is the central orchestrator: it emits the handshake (first event), drives the LLM stream,
commits the final assistant message to the `Conversation` and tool items to the `TurnExecution`, and emits
`bot_message` (last persistence-relevant event) before yielding back any compaction signal.

Compaction wiring is shared, not web-specific: `_build_compact_fn` builds the summarization closure and
`_maybe_compact` acquires the per-conversation lock and fires the background CAS swap. The web and Slack
harnesses trigger compaction through the identical pair of helpers.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis

from prokaryotes.context_v1 import (
    ConversationCompactor,
    SyncResult,
    get_redis_client,
)
from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    ProjectedItem,
    TurnExecution,
    TurnItem,
)
from prokaryotes.conversation_v1.project import project_for_llm
from prokaryotes.graph_v1 import GraphClient
from prokaryotes.search_v1 import SearchClient
from prokaryotes.utils_v1.llm_utils import COMPACTION_LOCK_TTL_SECONDS
from prokaryotes.utils_v1.logging_utils import log_async_task_exception

logger = logging.getLogger(__name__)

_SUMMARIZATION_PROMPT = (
    "Summarize the conversation above as a structured briefing for future continuation."
    " Preserve key decisions, facts, code produced, and tool call outcomes."
    " Use markdown sections. Be concise."
)


@dataclass
class _StreamFinalizationContext:
    """Handoff between stream_turn callbacks and `stream_and_finalize`'s commit step."""

    final_assistant_text: list[str]
    committed_turn_items: list[TurnItem]


class HarnessBase(ConversationCompactor):
    """Redis + Search + compaction lifecycle. No transport.

    Concrete harnesses (`WebHarness`, `SlackHarness`) supply `llm_client` and `default_model`; `_build_compact_fn`
    and `_summarize_and_compact` read them off `self`.
    """

    def __init__(self):
        self.background_tasks: set[asyncio.Task] = set()
        self._conversation_cache_ex = int(os.getenv("CONVERSATION_CACHE_EXPIRY_SECONDS", 60 * 60 * 24 * 7))
        self.graph_client = GraphClient()
        self._redis_client: Redis | None = None
        self._search_client = SearchClient()

    def background_and_forget(self, coro: Coroutine):
        bg_task = asyncio.create_task(coro)
        self.background_tasks.add(bg_task)
        bg_task.add_done_callback(log_async_task_exception)
        bg_task.add_done_callback(self.background_tasks.discard)

    @property
    def conversation_cache_ex(self) -> int:
        return self._conversation_cache_ex

    async def drain_background_tasks(self):
        if self.background_tasks:
            _done, pending = await asyncio.wait(self.background_tasks, timeout=30.0)
            if pending:
                logger.warning("Exiting with %d tasks pending", len(pending))

    def ensure_runtime_clients(self):
        """Idempotently create Redis / Search clients. Subclasses needing them before `on_start` (e.g. `WebBase`
        mounting Redis-backed sessions) call this in their own setup."""
        # self.graph_client.init_client()
        if self._redis_client is None:
            self._redis_client = get_redis_client()
        if self._search_client.es is None:
            self._search_client.init_client()

    async def finalize_turn(
        self,
        *,
        conversation: Conversation,
        bot_message_source_id: str,
        bot_message_content: str,
        turn_items: list[TurnItem],
        triggering_source_id: str,
    ) -> None:
        """Append the bot's final ConversationMessage, persist the TurnExecution, and add the bot message to the
        DAG-scoped assistant index so the next POST's guardrail recognizes it. Called by `stream_and_finalize`.

        `triggering_source_id` is the user message this reply answers; it is stamped onto the bot
        `ConversationMessage` as `reply_to_source_id` so `project_for_llm`'s two-pass walk can keep the turn pair
        intact on later turns.
        """
        conversation.messages.append(
            ConversationMessage(
                source_id=bot_message_source_id,
                author_id=conversation.bot_author_id,
                content=bot_message_content,
                reply_to_source_id=triggering_source_id,
            )
        )
        redis_key = f"conversation:{conversation.conversation_uuid}"
        await self.redis_client.set(
            redis_key,
            conversation.model_dump_json(),
            ex=self.conversation_cache_ex,
        )
        await self.search_client.put_conversation(conversation)
        if turn_items:
            await self.search_client.put_turn_execution(
                TurnExecution(
                    conversation_uuid=conversation.conversation_uuid,
                    bot_message_source_id=bot_message_source_id,
                    items=turn_items,
                    completed=True,
                )
            )
        await self.refresh_assistant_index_with(
            conversation.conversation_uuid,
            bot_message_source_id,
            bot_message_content,
        )

    async def on_start(self):
        self.ensure_runtime_clients()

    async def on_stop(self):
        await self.drain_background_tasks()
        close_tasks = []
        # close_tasks.append(self.graph_client.close())
        if self._search_client.es is not None:
            close_tasks.append(self._search_client.close())
        if self._redis_client is not None:
            close_tasks.append(self._redis_client.aclose())
        if close_tasks:
            await asyncio.gather(*close_tasks)

    @property
    def redis_client(self) -> Redis:
        if self._redis_client is None:
            raise RuntimeError("Redis client has not been initialized")
        return self._redis_client

    @property
    def search_client(self) -> SearchClient:
        return self._search_client

    async def stream_and_finalize(
        self,
        *,
        sync_result: SyncResult,
        bot_message_source_id_provider: Callable[[Conversation], str],
        pending_compaction: list[bool] | None = None,
        response_generator_factory: Callable[[_StreamFinalizationContext], AsyncGenerator[str, Any]] | None = None,
    ) -> AsyncGenerator[str, Any]:
        """Drive the stream, emit handshake first / bot_message last, finalize commit.

        - The **handshake** is always the first event, before any text/tool/progress. It carries the snapshot the
          bot will reply into (a fresh branch `snapshot_uuid` on divergence, the existing one otherwise), the
          server-assigned `source_id` map, and on resync, `unacknowledged_bot_messages`.
        - On `resync`, the stream closes after the handshake without invoking the LLM, and
          `response_generator_factory` is unused; it is required for every other path.
        - Otherwise, the LLM stream runs; intermediate `text_delta`/`tool_call`/`progress_message`/`context_pct`
          events flow through unchanged. The final assistant text is collected via the stream context object, and
          tool items are appended as they commit.
        - **`bot_message`** is emitted *exactly once*, after the final assistant text has been committed to the
          `Conversation`. If the turn fails before commit (LLM error, tool crash, stream abort), no `bot_message`
          is emitted.
        - When `pending_compaction` is set, `_maybe_compact` is invoked after commit; the `compaction_pending`
          event is emitted only if it actually scheduled a compaction.
        """
        yield json.dumps(_handshake_payload(sync_result)) + "\n"

        if sync_result.resync:
            # Non-reconcile path: close the stream and let the client retry.
            return

        if response_generator_factory is None:
            raise ValueError("response_generator_factory is required for the non-resync path")

        conversation = sync_result.conversation
        ctx = _StreamFinalizationContext(final_assistant_text=[], committed_turn_items=[])

        # The factory wires `on_committed_turn_item` / `on_final_assistant_message` into the LLM client; we own the
        # buffers here.
        async for event_str in response_generator_factory(ctx):
            if not event_str:
                continue
            yield event_str

        final_text = "".join(ctx.final_assistant_text)
        if not final_text:
            # No commitable bot message — do NOT emit bot_message; the client must not create an assistant node.
            return

        bot_source_id = bot_message_source_id_provider(conversation)
        await self.finalize_turn(
            conversation=conversation,
            bot_message_source_id=bot_source_id,
            bot_message_content=final_text,
            turn_items=ctx.committed_turn_items,
            triggering_source_id=_triggering_source_id(sync_result),
        )
        yield json.dumps({"bot_message": {"source_id": bot_source_id}}) + "\n"

        if pending_compaction is not None and pending_compaction[0]:
            if await self._maybe_compact(conversation, True):
                yield json.dumps({"compaction_pending": True}) + "\n"

    def _build_compact_fn(self) -> Callable[[Conversation, Any], Coroutine[Any, Any, str]]:
        """Build the compaction summary closure passed to `_compact_conversation`.

        It depends only on `self.llm_client` and `self.default_model`, both supplied by the concrete harness, so
        the web and Slack harnesses construct the identical callable.
        """

        async def compact(snapshot: Conversation, prep: Any) -> str:
            return await self._summarize_and_compact(model=self.default_model, snapshot=snapshot, prep=prep)

        return compact

    async def _maybe_compact(self, conversation: Conversation, pending: bool) -> bool:
        """Schedule background compaction when `pending` and the per-conversation lock can be acquired.

        Returns `True` when a compaction was scheduled (lock acquired, background task fired). The web harness
        gates its `compaction_pending` NDJSON event on the return value; Slack ignores it — Slack clients have no
        compaction indicator.
        """
        if not pending:
            return False
        lock_key = f"compaction_lock:{conversation.conversation_uuid}"
        acquired = await self.redis_client.set(lock_key, "1", ex=COMPACTION_LOCK_TTL_SECONDS, nx=True)
        if not acquired:
            return False
        # Deepcopy before scheduling: the background task runs arbitrarily later and the caller may mutate
        # `conversation` after the trigger, so `_compact_conversation`'s CAS swap must see the trigger-time
        # pre-image.
        snapshot = copy.deepcopy(conversation)
        self.background_and_forget(
            self._compact_conversation(
                compact_fn=self._build_compact_fn(),
                conversation_uuid=conversation.conversation_uuid,
                lock_key=lock_key,
                snapshot=snapshot,
            )
        )
        return True

    async def _summarize_and_compact(self, *, model: str, snapshot: Conversation, prep: Any) -> str:
        """Build the summarization input and make a non-streaming LLM call for the summary.

        `ancestor_summaries=[]` and `working_file_windows=[]` keep the compactor's input free of prior summaries
        and live file bodies — the projection then carries no leading `<compacted_summary>` or `<working_files>`
        block into the summarizer.
        """
        pre_tail_conv = snapshot.model_copy(
            update={
                "messages": prep.pre_tail_messages,
                "ancestor_summaries": [],
                "working_file_windows": [],
            }
        )
        items_for_summary = project_for_llm(pre_tail_conv, historical_turns=prep.pre_tail_turns)
        items_for_summary.append(ProjectedItem(type="message", role="user", content=_SUMMARIZATION_PROMPT))
        return await self.llm_client.complete(
            items=items_for_summary,
            instruction=None,
            model=model,
            reasoning_effort=None,
        )


def _handshake_payload(sync_result: SyncResult) -> dict:
    payload: dict[str, Any] = {
        "snapshot_uuid": sync_result.conversation.snapshot_uuid,
        "source_id_assignments": [
            {"client_index": a.client_index, "source_id": a.source_id} for a in sync_result.source_id_assignments
        ],
    }
    if sync_result.unacknowledged_bot_messages:
        payload["unacknowledged_bot_messages"] = [
            {
                "source_id": m.source_id,
                "content": m.content,
                "parent_source_id": m.parent_source_id,
            }
            for m in sync_result.unacknowledged_bot_messages
        ]
    return payload


def _triggering_source_id(sync_result: SyncResult) -> str:
    """The `source_id` of the user message that triggered this turn — the last in the submitted batch.

    A turn can carry several user messages; the trigger is the highest-`client_index` assignment. Falls back to
    the latest non-bot `source_id` already on the snapshot when sync emitted no assignment at all (every incoming
    message already had a server-assigned `source_id`).
    """
    assignments = sync_result.source_id_assignments
    if assignments:
        return max(assignments, key=lambda a: a.client_index).source_id
    conversation = sync_result.conversation
    return max(
        m.source_id for m in conversation.messages if not m.deleted and m.author_id != conversation.bot_author_id
    )
