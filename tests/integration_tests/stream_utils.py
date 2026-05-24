"""Helpers for collecting NDJSON streams and scoping background-task waits."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from contextlib import asynccontextmanager
from typing import Any

import httpx

from prokaryotes.utils_v1.logging_utils import log_async_task_exception


async def collect_stream(response: httpx.Response) -> list[dict]:
    events = []
    async for line in response.aiter_lines():
        if line.strip():
            events.append(json.loads(line))
    return events


@asynccontextmanager
async def request_scope(harness):
    """Track background tasks spawned during a request and surface their failures.

    The harness fixture is session-scoped, so harness.background_tasks is shared across tests. We temporarily wrap
    background_and_forget so this scope records the exact tasks spawned by the current request, even if they finish
    and remove themselves from harness.background_tasks before the context exits.
    """
    spawned_tasks: list[asyncio.Task[Any]] = []
    original_background_and_forget = harness.background_and_forget

    def tracking_background_and_forget(coro: Coroutine[Any, Any, Any]) -> None:
        bg_task = asyncio.create_task(coro)
        spawned_tasks.append(bg_task)
        harness.background_tasks.add(bg_task)
        bg_task.add_done_callback(log_async_task_exception)
        bg_task.add_done_callback(harness.background_tasks.discard)

    harness.background_and_forget = tracking_background_and_forget
    try:
        yield
    finally:
        harness.background_and_forget = original_background_and_forget
        if spawned_tasks:
            results = await asyncio.gather(*spawned_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
