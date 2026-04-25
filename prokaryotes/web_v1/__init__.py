import asyncio
import copy
import json
import logging
import os
import uuid
from collections.abc import (
    AsyncGenerator,
    Coroutine,
)
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import asyncpg
import bcrypt
import httpx
from asyncpg import Pool
from fastapi import (
    FastAPI,
    Form,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
)
from redis.asyncio import Redis
from redis.exceptions import WatchError
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starsessions import (
    SessionMiddleware,
    load_session,
)
from starsessions.stores.redis import RedisStore

from prokaryotes.api_v1.models import (
    ChatConversation,
    ContextPartition,
    ContextPartitionItem,
    ConversationMatchesPartitionError,
    compute_boundary_hash,
    compute_tail_hash,
    conversation_message_items,
)
from prokaryotes.graph_v1 import GraphClient
from prokaryotes.search_v1 import SearchClient
from prokaryotes.search_v1.context_partitions import (
    items_from_doc,
    partition_from_doc,
)
from prokaryotes.utils_v1 import http_utils
from prokaryotes.utils_v1.llm_utils import (
    COMPACTION_LOCK_TTL_SECONDS,
    COMPACTION_RECENCY_TAIL,
)
from prokaryotes.utils_v1.logging_utils import log_async_task_exception

logger = logging.getLogger(__name__)


def _message_count_before_item_index(items: list[ContextPartitionItem], item_index: int) -> int:
    return len(conversation_message_items(items[:item_index]))


def _partition_can_follow_client(partition: ContextPartition, client_partition_uuid: str | None) -> bool:
    if client_partition_uuid is None:
        return True
    return (
        partition.partition_uuid == client_partition_uuid
        or partition.parent_partition_uuid == client_partition_uuid
    )


def _recency_tail_items(
        items: list[ContextPartitionItem],
        message_tail_count: int,
) -> tuple[list[ContextPartitionItem], int]:
    message_indexes = [
        idx
        for idx, item in enumerate(items)
        if item.type == "message" and item.role in {"user", "assistant"}
    ]
    if not message_indexes:
        return [], 0
    first_tail_message_pos = max(0, len(message_indexes) - message_tail_count)
    while (first_tail_message_pos < len(message_indexes)
           and items[message_indexes[first_tail_message_pos]].role != "user"):
        first_tail_message_pos += 1
    if first_tail_message_pos >= len(message_indexes):
        return [], 0
    first_tail_item_index = message_indexes[first_tail_message_pos]
    return items[first_tail_item_index:], _message_count_before_item_index(items, first_tail_item_index)


class WebBase:
    def __init__(self, static_dir: str):
        self.app: FastAPI | None = None
        self.background_tasks: set[asyncio.Task] = set()
        self.conversation_cache_ex = int(os.getenv("CONVERSATION_CACHE_EXPIRY_SECONDS", 60 * 60 * 24 * 7))
        self.graph_client = GraphClient()
        self.postgres_pool: Pool | None = None
        self.redis_client: Redis | None = None
        self.search_client = SearchClient()
        self.static_dir = Path(static_dir)

    async def _boundary_message_items_for_doc(
            self,
            doc: dict,
            memo: dict[str, list[ContextPartitionItem]] | None = None,
    ) -> list[ContextPartitionItem]:
        memo = memo or {}
        partition_uuid = doc.get("partition_uuid")
        if partition_uuid in memo:
            return memo[partition_uuid]

        prefix: list[ContextPartitionItem] = []
        raw_start = doc.get("raw_message_start_index") or 0
        parent_uuid = doc.get("parent_partition_uuid")
        if parent_uuid and raw_start > 0:
            parent_doc = await self.search_client.get_partition(parent_uuid)
            if parent_doc:
                parent_messages = await self._boundary_message_items_for_doc(parent_doc, memo)
                prefix = parent_messages[:raw_start]

        items = items_from_doc(doc)

        messages = prefix + conversation_message_items(items)
        if partition_uuid:
            memo[partition_uuid] = messages
        return messages

    async def _boundary_message_items_for_partition(
            self,
            partition: ContextPartition,
    ) -> list[ContextPartitionItem]:
        prefix: list[ContextPartitionItem] = []
        if partition.parent_partition_uuid and partition.raw_message_start_index > 0:
            parent_doc = await self.search_client.get_partition(partition.parent_partition_uuid)
            if parent_doc:
                parent_messages = await self._boundary_message_items_for_doc(parent_doc)
                prefix = parent_messages[:partition.raw_message_start_index]
        return prefix + conversation_message_items(partition.items)

    async def _cache_and_persist_partition(self, context_partition: ContextPartition) -> None:
        await self.redis_client.set(
            f"context_partition:{context_partition.conversation_uuid}",
            context_partition.model_dump_json(),
            ex=self.conversation_cache_ex,
        )
        await self.search_client.put_partition(context_partition)

    async def _compact_partition(
            self,
            compact_fn,
            conversation_uuid: str,
            lock_key: str,
            snapshot: ContextPartition,
    ) -> None:
        swapped_partition: ContextPartition | None = None
        try:
            summary = await compact_fn(snapshot)
            if not summary:
                logger.warning("Compaction produced empty summary for partition %s", snapshot.partition_uuid)
                return

            boundary_items = await self._boundary_message_items_for_partition(snapshot)
            boundary_hash = compute_boundary_hash(boundary_items)
            tail_hash = compute_tail_hash(boundary_items)
            boundary_message_count = len(boundary_items)
            boundary_user_count = sum(1 for item in boundary_items if item.role == "user")
            await self.search_client.update_partition(
                snapshot.partition_uuid,
                is_compacted=True,
                summary=summary,
                boundary_hash=boundary_hash,
                boundary_message_count=boundary_message_count,
                boundary_user_count=boundary_user_count,
                tail_hash=tail_hash,
            )

            redis_key = f"context_partition:{conversation_uuid}"
            async with self.redis_client.pipeline() as pipe:
                while True:
                    try:
                        await pipe.watch(redis_key)
                        current_data = await pipe.get(redis_key)
                        if not current_data:
                            logger.info("Skipping compaction swap because Redis partition is missing")
                            return

                        current_partition = ContextPartition.model_validate_json(current_data)
                        if current_partition.partition_uuid != snapshot.partition_uuid:
                            logger.info(
                                "Skipping compaction swap for %s because active partition is now %s",
                                snapshot.partition_uuid,
                                current_partition.partition_uuid,
                            )
                            return
                        if (
                            current_partition.raw_message_start_index != snapshot.raw_message_start_index
                            or current_partition.ancestor_summaries != snapshot.ancestor_summaries
                            or current_partition.items[:len(snapshot.items)] != snapshot.items
                        ):
                            logger.info(
                                "Skipping compaction swap for %s because the active prefix changed",
                                snapshot.partition_uuid,
                            )
                            return

                        recency_tail, tail_offset = _recency_tail_items(
                            snapshot.items,
                            COMPACTION_RECENCY_TAIL,
                        )
                        post_snapshot_items = current_partition.items[len(snapshot.items):]
                        swapped_partition = ContextPartition(
                            conversation_uuid=conversation_uuid,
                            parent_partition_uuid=snapshot.partition_uuid,
                            ancestor_summaries=snapshot.ancestor_summaries + [summary],
                            raw_message_start_index=snapshot.raw_message_start_index + tail_offset,
                            items=recency_tail + post_snapshot_items,
                        )

                        pipe.multi()
                        pipe.set(
                            redis_key,
                            swapped_partition.model_dump_json(),
                            ex=self.conversation_cache_ex,
                        )
                        await pipe.execute()
                        logger.info(
                            "Compaction complete: new partition %s (parent %s)",
                            swapped_partition.partition_uuid,
                            snapshot.partition_uuid,
                        )
                        break
                    except WatchError:
                        logger.info("Compaction Redis swap contended, retrying")
                        await pipe.reset()

            if swapped_partition is not None:
                await self.search_client.put_partition(swapped_partition)
        except Exception:
            logger.exception("compact_partition failed for partition %s", snapshot.partition_uuid)
        finally:
            await self.redis_client.delete(lock_key)

    async def _load_exact_partition(
            self,
            conversation: ChatConversation,
    ) -> tuple[ContextPartition | None, dict | None]:
        """Fetch the partition doc for conversation.partition_uuid.

        Returns (partition, None) when the doc exists and is not compacted.
        Returns (None, doc) when the doc is compacted so the caller can pass it as head_doc
        to _rebuild_from_chain, avoiding a redundant fetch in _walk_partition_chain.
        Returns (None, None) when the doc is missing or fails to parse.
        """
        doc = await self.search_client.get_partition(conversation.partition_uuid)
        if not doc:
            return None, None
        if doc.get("is_compacted"):
            return None, doc
        try:
            return partition_from_doc(conversation.conversation_uuid, doc), None
        except Exception as exc:
            logger.warning("Failed to rebuild exact ES partition %s: %s", conversation.partition_uuid, exc)
            return None, None

    async def _rebuild_from_chain(
            self,
            conversation: ChatConversation,
            head_doc: dict | None = None,
    ) -> ContextPartition:
        """Rebuild a ContextPartition by walking the ES ancestor chain.

        head_doc is forwarded to _walk_partition_chain to skip a redundant fetch when
        the caller already holds the doc for conversation.partition_uuid.
        """
        if not conversation.partition_uuid:
            logger.info("Starting new context partition from request payload")
            return conversation.to_context_partition()

        chain = await self._walk_partition_chain(
            conversation.conversation_uuid,
            conversation.partition_uuid,
            head_doc=head_doc,
        )
        compacted_ancestors = [
            doc for doc in reversed(chain)
            if doc.get("is_compacted")
        ]

        matched_ancestor: dict | None = None
        for doc in compacted_ancestors:
            boundary_count = doc.get("boundary_message_count")
            boundary_hash = doc.get("boundary_hash")
            if boundary_count is None or not boundary_hash:
                continue
            if boundary_count > len(conversation.messages):
                continue
            incoming_hash = compute_boundary_hash(conversation.messages[:boundary_count])
            if incoming_hash == boundary_hash:
                matched_ancestor = doc

        if matched_ancestor is None:
            logger.info("No compacted ancestor validated; starting from raw request payload")
            return conversation.to_context_partition()

        ancestor_summaries = []
        for doc in compacted_ancestors:
            summary = doc.get("summary")
            if summary:
                ancestor_summaries.append(summary)
            if doc["partition_uuid"] == matched_ancestor["partition_uuid"]:
                break

        raw_message_start_index = matched_ancestor.get("boundary_message_count") or 0
        return ContextPartition(
            conversation_uuid=conversation.conversation_uuid,
            parent_partition_uuid=matched_ancestor["partition_uuid"],
            ancestor_summaries=ancestor_summaries,
            raw_message_start_index=raw_message_start_index,
            items=[
                message.to_context_partition_item()
                for message in conversation.messages[raw_message_start_index:]
            ],
        )

    def _try_sync_partition(
            self,
            context_partition: ContextPartition,
            conversation: ChatConversation,
            source: str,
    ) -> ContextPartition | None:
        try:
            context_partition.sync_from_conversation(conversation)
            logger.info("Synced context partition from %s", source)
            return context_partition
        except ConversationMatchesPartitionError:
            logger.info("Context partition from %s already matches request", source)
            return context_partition
        except Exception as exc:
            logger.info("Context partition from %s could not sync: %s", source, exc)
            return None

    async def _walk_partition_chain(
            self,
            conversation_uuid: str,
            partition_uuid: str,
            head_doc: dict | None = None,
    ) -> list[dict]:
        """Walk the ancestor chain from partition_uuid back to the root, returning docs oldest-last.

        head_doc, if provided, is used as the first doc instead of fetching it — avoids a
        redundant ES round-trip when the caller already has the doc for partition_uuid.
        """
        chain = []
        seen: set[str] = set()
        current_uuid = partition_uuid
        while current_uuid and current_uuid not in seen:
            seen.add(current_uuid)
            if head_doc is not None and head_doc.get("partition_uuid") == current_uuid:
                doc = head_doc
                head_doc = None
            else:
                doc = await self.search_client.get_partition(current_uuid)
            if not doc:
                break
            if doc.get("conversation_uuid") and doc["conversation_uuid"] != conversation_uuid:
                break
            chain.append(doc)
            current_uuid = doc.get("parent_partition_uuid")
        return chain

    def background_and_forget(self, coro: Coroutine):
        bg_task = asyncio.create_task(coro)
        self.background_tasks.add(bg_task)
        bg_task.add_done_callback(log_async_task_exception)
        bg_task.add_done_callback(self.background_tasks.discard)

    async def finalize(
            self,
            context_partition: ContextPartition,
    ):
        context_partition.pop_system_message()  # The leading system message is re-injected on each request.
        redis_key = f"context_partition:{context_partition.conversation_uuid}"
        await self.redis_client.set(
            redis_key,
            context_partition.model_dump_json(),
            ex=self.conversation_cache_ex,
        )
        await self.search_client.put_partition(context_partition)

    async def get_common_auth_css(self):
        return FileResponse(self.static_dir / "common-auth.css")

    async def get_common_auth_js(self):
        return FileResponse(self.static_dir / "common-auth.js")

    async def get_common_css(self):
        return FileResponse(self.static_dir / "common.css")

    @staticmethod
    async def get_conversation(request: Request):
        await load_session(request)
        session = request.session
        if not session:
            raise HTTPException(status_code=400, detail="Session expired")
        return {"conversation_uuid": uuid.uuid4()}

    @staticmethod
    async def get_health():
        return {"status": "ok"}

    async def get_login(self, request: Request):
        await load_session(request)
        session = request.session
        if session:
            return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        return FileResponse(self.static_dir / "login.html")

    async def get_logo(self):
        logo_png_path = self.static_dir / "logo.png"
        if logo_png_path.exists():
            return FileResponse(
                media_type="image/png",
                path=logo_png_path,
            )
        raise HTTPException(status_code=404, detail="Not found")

    @staticmethod
    async def get_logout(request: Request):
        await load_session(request)
        request.session.clear()
        info = urlencode({"info": "Logged out."})
        return RedirectResponse(url=f"/login?{info}", status_code=status.HTTP_303_SEE_OTHER)

    async def get_register(self, request: Request):
        await load_session(request)
        session = request.session
        if session:
            return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        return FileResponse(self.static_dir / "register.html")

    async def get_root(self, request: Request):
        await load_session(request)
        session = request.session
        if not session:
            return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        return FileResponse(self.static_dir / "ui.html")

    async def get_ui_js(self):
        return FileResponse(self.static_dir / "ui.js")

    def init(self):
        """Synchronous setup steps"""
        self.graph_client.init_client()
        self.redis_client = get_redis_client()
        self.search_client.init_client()

        self.app = FastAPI(
            lifespan=self.lifespan,
            middleware=[
                Middleware(
                    SessionMiddleware,
                    store=RedisStore(connection=self.redis_client, prefix="session:"),
                    cookie_name="prokaryotes_session",
                    cookie_https_only=False,
                    lifetime=(60 * 60 * 24 * 7),
                    rolling=True,
                )
            ],
        )
        self.app.add_api_route("/", self.get_root, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/common-auth.css", self.get_common_auth_css, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/common-auth.js", self.get_common_auth_js, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/common.css", self.get_common_css, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/conversation", self.get_conversation, methods=["GET"])
        self.app.add_api_route("/health", self.get_health, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/login", self.get_login, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/login", self.post_login, methods=["POST"])
        self.app.add_api_route("/logo.png", self.get_logo, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/logout", self.get_logout, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/register", self.get_register, methods=["GET"], include_in_schema=False)
        self.app.add_api_route("/register", self.post_register, methods=["POST"])
        self.app.add_api_route("/ui.js", self.get_ui_js, methods=["GET"], include_in_schema=False)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Asynchronous setup/teardown steps"""
        logger.info("Entering setup")
        http_utils.httpx_client = httpx.AsyncClient()
        self.postgres_pool = await get_postgres_pool()
        await self.on_start()
        yield
        logger.info("Entering teardown")
        if self.background_tasks:
            done_task, pending_tasks = await asyncio.wait(self.background_tasks, timeout=30.0)
            if pending_tasks:
                logger.warning(f"Exiting with {len(pending_tasks)} tasks pending")
        await asyncio.gather(
            http_utils.httpx_client.aclose(),
            self.on_stop(),
            self.postgres_pool.close(),
            self.redis_client.close(),
        )

    async def on_start(self):
        """Asynchronous setup steps"""
        pass

    async def on_stop(self):
        """Asynchronous teardown steps"""
        await self.graph_client.close()
        await self.search_client.close()

    async def post_login(
            self,
            request: Request,
            email: str = Form(...),
            password: str = Form(...),
    ):
        error = urlencode({"error": f"Not able to login using {email}"})
        if email and password:
            load_session_task = asyncio.create_task(load_session(request))
            load_session_task.add_done_callback(log_async_task_exception)
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT full_name, password_hash, user_id FROM chat_user WHERE email = $1
                    """,
                    email
                )
                if row and await run_in_threadpool(verify_password, password, row["password_hash"]):
                    await load_session_task
                    request.session.clear()
                    request.session.update({
                        "full_name": row["full_name"],
                        "user_id": row["user_id"],
                    })
                    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        return RedirectResponse(f"/login?{error}", status_code=status.HTTP_303_SEE_OTHER)

    async def post_register(
            self,
            request: Request,
            confirm_password: str = Form(...),
            email: str = Form(...),
            full_name: str = Form(...),
            password: str = Form(...),
    ):
        error = urlencode({"error": f"Not able to register using {email}"})
        if confirm_password and email and full_name and password:
            if password != confirm_password:
                error = urlencode({"error": "'Password' and 'Confirm password' must match"})
            else:
                load_session_task = asyncio.create_task(load_session(request))
                load_session_task.add_done_callback(log_async_task_exception)
                async with self.postgres_pool.acquire() as conn:
                    password_hash = await conn.fetchval(
                        """
                        SELECT password_hash FROM chat_user WHERE email = $1
                        """,
                        email
                    )
                    if not password_hash:
                        password_hash = await run_in_threadpool(hash_password, password)
                        user_id = await conn.fetchval(
                            """
                            INSERT INTO chat_user (email, full_name, password_hash)
                            VALUES ($1, $2, $3)
                            RETURNING user_id
                            """,
                            email, full_name, password_hash,
                        )
                        if user_id:
                            await load_session_task
                            request.session.clear()
                            request.session.update({
                                "full_name": full_name,
                                "user_id": user_id,
                            })
                            return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        return RedirectResponse(f"/register?{error}", status_code=status.HTTP_303_SEE_OTHER)

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

        should_compact = (
            pending_compaction is not None
            and pending_compaction[0]
            and compact_fn is not None
        )
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

        self.background_and_forget(
            self.finalize(
                context_partition=context_partition,
            )
        )

    async def sync_context_partition(self, conversation: ChatConversation) -> ContextPartition:
        cached_partition_data = await self.redis_client.get(f"context_partition:{conversation.conversation_uuid}")
        if cached_partition_data:
            context_partition = ContextPartition.model_validate_json(cached_partition_data)
            if _partition_can_follow_client(context_partition, conversation.partition_uuid):
                synced = self._try_sync_partition(context_partition, conversation, source="Redis")
                if synced is not None:
                    return synced
            else:
                logger.info(
                    "Cached partition %s does not match client partition %s; falling back to ES",
                    context_partition.partition_uuid,
                    conversation.partition_uuid,
                )

        head_doc: dict | None = None
        if conversation.partition_uuid:
            exact_partition, head_doc = await self._load_exact_partition(conversation)
            if exact_partition is not None:
                synced = self._try_sync_partition(exact_partition, conversation, source="ES")
                if synced is not None:
                    await self._cache_and_persist_partition(synced)
                    return synced

        rebuilt = await self._rebuild_from_chain(conversation, head_doc=head_doc)
        await self._cache_and_persist_partition(rebuilt)
        return rebuilt


async def get_postgres_pool():
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "prokaryotes")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    ssl_mode = os.getenv("POSTGRES_SSL_MODE", "disable")
    if host and password and user:
        return await asyncpg.create_pool(
            dsn=f"postgresql://{user}:{password}@{host}:{port}/{db}",
            ssl=ssl_mode,
            min_size=int(os.getenv("POSTGRES_POOL_MIN_SIZE", "1")),
            max_size=int(os.getenv("POSTGRES_POOL_MAX_SIZE", "3")),
        )
    raise RuntimeError("Unable to initialize postgres pool")


def get_redis_client() -> Redis:
    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT", "6379")
    db = os.getenv("REDIS_DB", "0")
    if host:
        return Redis.from_url(
            f"redis://{host}:{port}/{db}",
            decode_responses=False
        )
    raise RuntimeError("Unable to initialize Redis client")


def hash_password(plain_text_password: str) -> str:
    return bcrypt.hashpw(
        plain_text_password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')


def verify_password(plain_text_password: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(
        plain_text_password.encode('utf-8'),
        stored_hash.encode('utf-8')
    )
