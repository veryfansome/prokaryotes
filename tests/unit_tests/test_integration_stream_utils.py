import asyncio

import pytest

from tests.integration_tests.stream_utils import request_scope


class DummyHarness:
    def __init__(self):
        self.background_tasks: set[asyncio.Task] = set()

    def background_and_forget(self, coro):
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)


@pytest.mark.asyncio
async def test_request_scope_re_raises_completed_background_task_failures():
    harness = DummyHarness()

    async def boom():
        await asyncio.sleep(0)
        raise RuntimeError("background task failed")

    with pytest.raises(RuntimeError, match="background task failed"):
        async with request_scope(harness):
            harness.background_and_forget(boom())
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_request_scope_awaits_spawned_background_tasks_before_exit():
    harness = DummyHarness()
    finished = False

    async def work():
        nonlocal finished
        await asyncio.sleep(0.01)
        finished = True

    async with request_scope(harness):
        harness.background_and_forget(work())

    assert finished is True
    assert harness.background_tasks == set()
