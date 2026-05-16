import asyncio
import copy
import json
import logging
import os
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from asyncpg import Pool
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis
from starlette.middleware import Middleware
from starsessions import SessionMiddleware
from starsessions.stores.redis import RedisStore

from prokaryotes.api_v1.models import ContextPartition
from prokaryotes.graph_v1 import GraphClient
from prokaryotes.search_v1 import SearchClient
from prokaryotes.utils_v1.llm_utils import COMPACTION_LOCK_TTL_SECONDS
from prokaryotes.utils_v1.logging_utils import log_async_task_exception
from prokaryotes.web_v1.auth import (
    AuthHandler,
    hash_password,
    verify_password,
)
from prokaryotes.web_v1.compaction import PartitionCompactor
from prokaryotes.web_v1.partition_sync import (
    PartitionSyncer,
    _partition_can_follow_client,
)
from prokaryotes.web_v1.stores import get_postgres_pool, get_redis_client

logger = logging.getLogger(__name__)


class WebBase(AuthHandler, PartitionCompactor):
    def __init__(self, static_dir: str):
        self.app: FastAPI | None = None
        self.background_tasks: set[asyncio.Task] = set()
        self._conversation_cache_ex = int(os.getenv("CONVERSATION_CACHE_EXPIRY_SECONDS", 60 * 60 * 24 * 7))
        self.graph_client = GraphClient()
        self._postgres_pool: Pool | None = None
        self._redis_client: Redis | None = None
        self._search_client = SearchClient()
        self.static_dir = Path(static_dir)
        self._html_dir = self.static_dir.parent / "html"

    def background_and_forget(self, coro: Coroutine):
        bg_task = asyncio.create_task(coro)
        self.background_tasks.add(bg_task)
        bg_task.add_done_callback(log_async_task_exception)
        bg_task.add_done_callback(self.background_tasks.discard)

    @property
    def conversation_cache_ex(self) -> int:
        return self._conversation_cache_ex

    async def finalize(self, context_partition: ContextPartition):
        context_partition.pop_system_message()  # The leading system message is re-injected on each request.
        redis_key = f"context_partition:{context_partition.conversation_uuid}"
        await self.redis_client.set(
            redis_key,
            context_partition.model_dump_json(),
            ex=self.conversation_cache_ex,
        )
        await self.search_client.put_partition(context_partition)

    @staticmethod
    async def get_health():
        return {"status": "ok"}

    @property
    def html_dir(self) -> Path:
        return self._html_dir

    def init(self):
        # self.graph_client.init_client()
        self._redis_client = get_redis_client()
        self._search_client.init_client()

        self.app = FastAPI(
            lifespan=self.lifespan,
            middleware=[
                Middleware(
                    SessionMiddleware,
                    store=RedisStore(connection=self._redis_client, prefix="session:"),
                    cookie_name="prokaryotes_session",
                    cookie_https_only=False,
                    lifetime=(60 * 60 * 24 * 7),
                    rolling=True,
                )
            ],
        )
        self.app.add_api_route("/", self.get_root, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/compaction-status", self.get_compaction_status, methods=["GET"])
        self.app.add_api_route("/conversation", self.get_conversation, methods=["GET"])
        self.app.add_api_route("/health", self.get_health, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/login", self.get_login, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/login", self.post_login, methods=["POST"])
        self.app.add_api_route("/logout", self.get_logout, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/register", self.get_register, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/register", self.post_register, methods=["POST"])
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info("Entering setup")
        self._postgres_pool = await get_postgres_pool()
        await self.on_start()
        yield
        logger.info("Entering teardown")
        if self.background_tasks:
            done_task, pending_tasks = await asyncio.wait(self.background_tasks, timeout=30.0)
            if pending_tasks:
                logger.warning(f"Exiting with {len(pending_tasks)} tasks pending")
        await asyncio.gather(
            self.on_stop(),
            self.postgres_pool.close(),
            self.redis_client.aclose(),
        )

    async def on_start(self):
        pass

    async def on_stop(self):
        # await self.graph_client.close()
        await self._search_client.close()

    @property
    def postgres_pool(self) -> Pool:
        if self._postgres_pool is None:
            raise RuntimeError("Postgres pool has not been initialized")
        return self._postgres_pool

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
                logger.warning(f"Received empty '{str_to_yield}' to yield")
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


__all__ = [
    "AuthHandler",
    "PartitionCompactor",
    "PartitionSyncer",
    "WebBase",
    "_partition_can_follow_client",
    "get_postgres_pool",
    "get_redis_client",
    "hash_password",
    "verify_password",
]
