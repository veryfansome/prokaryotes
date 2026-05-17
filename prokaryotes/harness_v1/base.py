import asyncio
import copy
import json
import logging
import os
from collections.abc import AsyncGenerator, Coroutine
from typing import Any

from redis.asyncio import Redis

from prokaryotes.api_v1.models import ContextPartition
from prokaryotes.context_v1 import PartitionCompactor, get_redis_client
from prokaryotes.graph_v1 import GraphClient
from prokaryotes.search_v1 import SearchClient
from prokaryotes.utils_v1.llm_utils import COMPACTION_LOCK_TTL_SECONDS
from prokaryotes.utils_v1.logging_utils import log_async_task_exception

logger = logging.getLogger(__name__)


class HarnessBase(PartitionCompactor):
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
            _done_tasks, pending_tasks = await asyncio.wait(self.background_tasks, timeout=30.0)
            if pending_tasks:
                logger.warning("Exiting with %d tasks pending", len(pending_tasks))

    def ensure_runtime_clients(self):
        """Idempotently create Redis / Search clients for subclasses that need them pre-startup."""
        # self.graph_client.init_client()
        if self._redis_client is None:
            self._redis_client = get_redis_client()
        if self._search_client.es is None:
            self._search_client.init_client()

    async def finalize(self, context_partition: ContextPartition):
        context_partition.pop_system_message()  # The leading system message is re-injected on each request.
        redis_key = f"context_partition:{context_partition.conversation_uuid}"
        await self.redis_client.set(
            redis_key,
            context_partition.model_dump_json(),
            ex=self.conversation_cache_ex,
        )
        await self.search_client.put_partition(context_partition)

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
        context_partition: ContextPartition,
        conversation_uuid: str,
        response_generator: AsyncGenerator[str, Any],
        compact_fn=None,
        pending_compaction: list[bool] | None = None,
    ) -> AsyncGenerator[str, Any]:
        yield json.dumps({"partition_uuid": context_partition.partition_uuid}) + "\n"

        async for str_to_yield in response_generator:
            if not str_to_yield:
                logger.warning("Received empty %r to yield", str_to_yield)
                continue
            yield str_to_yield

        should_compact = pending_compaction is not None and pending_compaction[0] and compact_fn is not None
        if should_compact:
            lock_key = f"compaction_lock:{conversation_uuid}"
            acquired = await self.redis_client.set(
                lock_key,
                "1",
                ex=COMPACTION_LOCK_TTL_SECONDS,
                nx=True,
            )
            if acquired:
                await self.finalize(context_partition)
                yield json.dumps({"compaction_pending": True}) + "\n"
                snapshot = copy.deepcopy(context_partition)
                self.background_and_forget(
                    self._compact_partition(
                        compact_fn=compact_fn,
                        conversation_uuid=conversation_uuid,
                        lock_key=lock_key,
                        snapshot=snapshot,
                    )
                )
                return

        self.background_and_forget(self.finalize(context_partition=context_partition))
