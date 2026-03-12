import asyncio
import asyncpg
import bcrypt
import logging
import os
from abc import abstractmethod
from asyncpg import Pool
from contextlib import asynccontextmanager
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
from pathlib import Path
from redis.asyncio import Redis
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starsessions import (
    SessionMiddleware,
    load_session,
)
from starsessions.stores.redis import RedisStore
from typing import Coroutine
from urllib.parse import urlencode

from prokaryotes.utils import log_async_task_exception

logger = logging.getLogger(__name__)

class WebBase:
    def __init__(self, static_dir: str):
        self.app: FastAPI | None = None
        self.background_tasks: set[asyncio.Task] = set()
        self.postgres_pool: Pool | None = None
        self.redis_client: Redis | None = None
        self.static_dir = Path(static_dir)

    def background_and_forget(self, coro: Coroutine):
        bg_task = asyncio.create_task(coro)
        self.background_tasks.add(bg_task)
        bg_task.add_done_callback(log_async_task_exception)
        bg_task.add_done_callback(self.background_tasks.discard)

    @classmethod
    async def get_health(cls):
        return {"status": "ok"}

    async def get_common_auth_css(self):
        return FileResponse(self.static_dir / "common-auth.css")

    async def get_common_auth_js(self):
        return FileResponse(self.static_dir / "common-auth.js")

    async def get_common_css(self):
        return FileResponse(self.static_dir / "common.css")

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

    @classmethod
    async def get_logout(cls, request: Request):
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
        self.redis_client = get_redis_client()
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
        self.postgres_pool = await get_postgres_pool()
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
            self.redis_client.close(),
        )

    @abstractmethod
    async def on_start(self):
        """Asynchronous setup steps"""
        pass

    @abstractmethod
    async def on_stop(self):
        """Asynchronous teardown steps"""
        pass

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
    db = os.getenv("REDIS_HOST", "0")
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
