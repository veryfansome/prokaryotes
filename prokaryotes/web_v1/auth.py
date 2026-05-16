import asyncio
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlencode

import bcrypt
from asyncpg import Pool
from fastapi import (
    Form,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
)
from starlette.concurrency import run_in_threadpool
from starsessions import load_session

from prokaryotes.utils_v1.logging_utils import log_async_task_exception


class AuthHandler(ABC):
    """Authentication routes (login, register, logout, root) and the session-gated GET handlers."""

    @staticmethod
    async def get_conversation(request: Request):
        await load_session(request)
        session = request.session
        if not session:
            raise HTTPException(status_code=400, detail="Session expired")
        return {"conversation_uuid": uuid.uuid4()}

    async def get_login(self, request: Request):
        await load_session(request)
        session = request.session
        if session:
            return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        return FileResponse(self.html_dir / "login.html")

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
        return FileResponse(self.html_dir / "register.html")

    async def get_root(self, request: Request):
        await load_session(request)
        session = request.session
        if not session:
            return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        return FileResponse(self.html_dir / "ui.html")

    @property
    @abstractmethod
    def html_dir(self) -> Path:
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
                    email,
                )
                if row and await run_in_threadpool(verify_password, password, row["password_hash"]):
                    await load_session_task
                    request.session.clear()
                    request.session.update(
                        {
                            "full_name": row["full_name"],
                            "user_id": row["user_id"],
                        }
                    )
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
                        email,
                    )
                    if not password_hash:
                        password_hash = await run_in_threadpool(hash_password, password)
                        user_id = await conn.fetchval(
                            """
                            INSERT INTO chat_user (email, full_name, password_hash)
                            VALUES ($1, $2, $3)
                            RETURNING user_id
                            """,
                            email,
                            full_name,
                            password_hash,
                        )
                        if user_id:
                            await load_session_task
                            request.session.clear()
                            request.session.update(
                                {
                                    "full_name": full_name,
                                    "user_id": user_id,
                                }
                            )
                            return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        return RedirectResponse(f"/register?{error}", status_code=status.HTTP_303_SEE_OTHER)

    @property
    @abstractmethod
    def postgres_pool(self) -> Pool:
        pass


def hash_password(plain_text_password: str) -> str:
    return bcrypt.hashpw(plain_text_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_text_password: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(plain_text_password.encode("utf-8"), stored_hash.encode("utf-8"))
