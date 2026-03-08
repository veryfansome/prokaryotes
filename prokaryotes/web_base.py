import asyncio
import logging
import os
from abc import abstractmethod
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
from redis.asyncio import Redis
from starlette.middleware import Middleware
from starsessions import SessionMiddleware, load_session
from starsessions.stores.redis import RedisStore
from typing import Coroutine

from prokaryotes.utils import log_async_task_exception

logger = logging.getLogger(__name__)

class WebBase:
    def __init__(self, static_dir: str):
        self.background_tasks: set[asyncio.Task] = set()
        self.redis_client = get_redis_client()
        self.static_dir = Path(static_dir)

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
        self.app.add_api_route("/", self.root, methods=["GET"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/logo.png", self.logo, methods=["GET"])

    def background_and_forget(self, coro: Coroutine):
        bg_task = asyncio.create_task(coro)
        self.background_tasks.add(bg_task)
        bg_task.add_done_callback(log_async_task_exception)
        bg_task.add_done_callback(self.background_tasks.discard)

    @classmethod
    async def health(cls):
        """Health check."""
        return {"status": "ok"}

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info("Entering lifespan")
        await self.on_start()
        yield
        if self.background_tasks:
            done_task, pending_tasks = await asyncio.wait(self.background_tasks, timeout=30.0)
            if pending_tasks:
                logger.warning(f"Exiting with {len(pending_tasks)} tasks pending")
        await asyncio.gather(
            self.redis_client.close(),
            self.on_stop(),
        )
        logger.info("Exiting lifespan")

    async def logo(self):
        logo_png_path = self.static_dir / "logo.png"
        if logo_png_path.exists():
            return FileResponse(
                media_type="image/png",
                path=logo_png_path,
            )
        raise HTTPException(status_code=404, detail="Not found")

    @abstractmethod
    async def on_start(self):
        pass

    @abstractmethod
    async def on_stop(self):
        pass

    async def root(self):
        """Serve the chat UI."""
        ui_html_path = self.static_dir / "ui.html"
        with open(ui_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)

def get_redis_client() -> Redis:
    redis_host = os.getenv('REDIS_HOST')
    if redis_host:
        return Redis.from_url(
            f"redis://{redis_host}:{os.getenv('REDIS_PORT', '6379')}/0",
            decode_responses=False
        )
    raise RuntimeError("Unable to initialize Redis client")
