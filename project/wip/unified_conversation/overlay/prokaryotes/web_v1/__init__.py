import asyncio
import logging
import pathlib
from contextlib import asynccontextmanager
from pathlib import Path

import prokaryotes

# Fall through to upstream for unchanged sibling modules (auth.py).
_HERE = pathlib.Path(__file__).resolve().parent
for _parent_path in prokaryotes.__path__:
    _candidate = pathlib.Path(_parent_path).resolve() / "web_v1"
    if _candidate != _HERE and _candidate.is_dir() and str(_candidate) not in __path__:
        __path__.append(str(_candidate))

from asyncpg import Pool  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402
from starsessions import SessionMiddleware  # noqa: E402
from starsessions.stores.redis import RedisStore  # noqa: E402

from prokaryotes.context_v1 import (  # noqa: E402
    ConversationCompactor,
    ConversationSyncer,
    _conversation_can_follow_client,
    get_redis_client,
)
from prokaryotes.harness_v1.base import HarnessBase  # noqa: E402
from prokaryotes.utils_v1.db_utils import get_postgres_pool  # noqa: E402
from prokaryotes.web_v1.auth import AuthHandler, hash_password, verify_password  # noqa: E402
from prokaryotes.web_v1.compaction import CompactionStatusHandler  # noqa: E402

logger = logging.getLogger(__name__)


class WebBase(HarnessBase, AuthHandler, CompactionStatusHandler):
    def __init__(self, static_dir: str):
        super().__init__()
        self.app: FastAPI | None = None
        self._postgres_pool: Pool | None = None
        self.static_dir = Path(static_dir)
        self._html_dir = self.static_dir.parent / "html"

    @staticmethod
    async def get_health():
        return {"status": "ok"}

    @property
    def html_dir(self) -> Path:
        return self._html_dir

    def init(self):
        self.ensure_runtime_clients()
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
        try:
            yield
        finally:
            logger.info("Entering teardown")
            await asyncio.gather(self.on_stop(), self.postgres_pool.close())

    @property
    def postgres_pool(self) -> Pool:
        if self._postgres_pool is None:
            raise RuntimeError("Postgres pool has not been initialized")
        return self._postgres_pool


__all__ = [
    "AuthHandler",
    "ConversationCompactor",
    "ConversationSyncer",
    "WebBase",
    "_conversation_can_follow_client",
    "get_postgres_pool",
    "get_redis_client",
    "hash_password",
    "verify_password",
]
