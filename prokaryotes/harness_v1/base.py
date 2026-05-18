"""`HarnessBase` — Redis + Search + compaction lifecycle.

`sync_conversation` returns a `SyncResult` carrying the snapshot plus any `source_id` assignments and resync info.
`stream_and_finalize` is the central orchestrator: it emits the handshake (first event), drives the LLM stream,
commits the final assistant message to the `Conversation` and tool items to the `TurnExecution`, and emits
`bot_message` (last persistence-relevant event) before yielding back any compaction signal.
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
    TurnExecution,
    TurnItem,
)
from prokaryotes.graph_v1 import GraphClient
from prokaryotes.search_v1 import SearchClient
from prokaryotes.utils_v1.llm_utils import COMPACTION_LOCK_TTL_SECONDS
from prokaryotes.utils_v1.logging_utils import log_async_task_exception

logger = logging.getLogger(__name__)


@dataclass
class _StreamFinalizationContext:
    """Handoff between stream_turn callbacks and `stream_and_finalize`'s commit step."""

    final_assistant_text: list[str]
    committed_turn_items: list[TurnItem]


class HarnessBase(ConversationCompactor):
    """Redis + Search + compaction lifecycle. No transport."""

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
    ) -> None:
        """Append the bot's final ConversationMessage, persist the TurnExecution, and add the bot message to the
        DAG-scoped assistant index so the next POST's guardrail recognizes it. Called by `stream_and_finalize`."""
        conversation.messages.append(
            ConversationMessage(
                source_id=bot_message_source_id,
                author_id=conversation.bot_author_id,
                content=bot_message_content,
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
        response_generator_factory: Callable[[_StreamFinalizationContext], AsyncGenerator[str, Any]],
        compact_fn=None,
        pending_compaction: list[bool] | None = None,
    ) -> AsyncGenerator[str, Any]:
        """Drive the stream, emit handshake first / bot_message last, finalize commit.

        - The **handshake** is always the first event, before any text/tool/progress. It carries the snapshot the
          bot will reply into (a fresh branch `snapshot_uuid` on divergence, the existing one otherwise), the
          server-assigned `source_id` map, and on resync, `unacknowledged_bot_messages`.
        - On `resync`, the stream closes after the handshake without invoking the LLM.
        - Otherwise, the LLM stream runs; intermediate `text_delta`/`tool_call`/`progress_message`/`context_pct`
          events flow through unchanged. The final assistant text is collected via the stream context object, and
          tool items are appended as they commit.
        - **`bot_message`** is emitted *exactly once*, after the final assistant text has been committed to the
          `Conversation`. If the turn fails before commit (LLM error, tool crash, stream abort), no `bot_message`
          is emitted.
        """
        yield json.dumps(_handshake_payload(sync_result)) + "\n"

        if sync_result.resync:
            # Non-reconcile path: close the stream and let the client retry.
            return

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
        )
        yield json.dumps({"bot_message": {"source_id": bot_source_id}}) + "\n"

        should_compact = pending_compaction is not None and pending_compaction[0] and compact_fn is not None
        if should_compact:
            lock_key = f"compaction_lock:{conversation.conversation_uuid}"
            acquired = await self.redis_client.set(
                lock_key,
                "1",
                ex=COMPACTION_LOCK_TTL_SECONDS,
                nx=True,
            )
            if acquired:
                yield json.dumps({"compaction_pending": True}) + "\n"
                snapshot = copy.deepcopy(conversation)
                self.background_and_forget(
                    self._compact_conversation(
                        compact_fn=compact_fn,
                        conversation_uuid=conversation.conversation_uuid,
                        lock_key=lock_key,
                        snapshot=snapshot,
                    )
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
