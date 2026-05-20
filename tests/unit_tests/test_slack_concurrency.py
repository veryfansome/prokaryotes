"""Per-thread turn serialization and cancellation-safe finalize.

`_conversation_turn_lock` serializes turns for one `conversation_uuid` while leaving different conversations
fully concurrent. `_sweep_turn_locks` reclaims only cold entries. `_await_finalize_through_cancel` keeps the
per-thread lock held until a finalize task actually completes, even when the per-turn timeout cancels the
caller — and surfaces a finalize exception ahead of the cancellation.
"""

from __future__ import annotations

import asyncio

import pytest

import prokaryotes.harness_v1.slack as slack_harness
import prokaryotes.slack_v1 as slack_v1
from prokaryotes.harness_v1.slack import SlackHarness
from prokaryotes.slack_v1 import _TURN_LOCK_IDLE_SECONDS, _TURN_LOCK_SWEEP_SECONDS, SlackBase
from tests.unit_tests._slack_fakes import FakeRedis


class _Harness(SlackBase):
    """Minimal `SlackBase` for lock / finalize-helper tests."""

    def __init__(self) -> None:
        super().__init__(app_token="xapp", bot_token="xoxb")
        self._redis_client = FakeRedis()

    async def handle_event(self, *, event: dict) -> None:
        pass


class _TimeoutHarness(SlackHarness):
    """`SlackHarness` whose `_locked_turn` is replaced by a controllable coroutine, so the per-turn timeout
    wrapping the whole locked region can be exercised."""

    def __init__(self) -> None:
        # Skip SlackHarness.__init__ — it builds a real LLM client.
        SlackBase.__init__(self, app_token="xapp", bot_token="xoxb")
        self._redis_client = FakeRedis()
        self.team_id = "T_TEAM"
        self._bot_token = "xoxb"
        self.locked_turn_impl = None
        self.locked_turn_started = asyncio.Event()

    async def _locked_turn(self, **kwargs) -> None:
        self.locked_turn_started.set()
        await self.locked_turn_impl()


@pytest.fixture
def harness() -> _Harness:
    return _Harness()


# -----------------------------------------------------------------------------
# _conversation_turn_lock — serialization
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_conversation_turns_run_strictly_in_order(harness: _Harness):
    """Two turns for the same `conversation_uuid` run serially — the second only starts after the first
    releases the lock."""
    order: list[str] = []

    async def turn(label: str) -> None:
        async with harness._conversation_turn_lock("conv-A"):
            order.append(f"{label}-start")
            await asyncio.sleep(0.02)
            order.append(f"{label}-end")

    await asyncio.gather(turn("first"), turn("second"))

    # No interleaving: one turn fully completes before the next begins.
    assert order in (
        ["first-start", "first-end", "second-start", "second-end"],
        ["second-start", "second-end", "first-start", "first-end"],
    )


@pytest.mark.asyncio
async def test_different_conversations_run_concurrently(harness: _Harness):
    """Turns for different `conversation_uuid`s do not serialize against each other."""
    both_inside = asyncio.Event()
    inside_count = [0]

    async def turn(conversation_uuid: str) -> None:
        async with harness._conversation_turn_lock(conversation_uuid):
            inside_count[0] += 1
            if inside_count[0] == 2:
                both_inside.set()
            await asyncio.wait_for(both_inside.wait(), timeout=1.0)

    await asyncio.gather(turn("conv-A"), turn("conv-B"))

    # Both turns were inside their locks simultaneously — proof they ran concurrently.
    assert both_inside.is_set()


@pytest.mark.asyncio
async def test_lock_entry_reused_across_back_to_back_turns(harness: _Harness):
    """The `_LockEntry` is never popped on release: two back-to-back turns reuse the same entry, and
    `last_used_monotonic` is refreshed on the second use."""
    async with harness._conversation_turn_lock("conv-A"):
        pass
    entry = harness._turn_locks["conv-A"]
    first_used = entry.last_used_monotonic

    async with harness._conversation_turn_lock("conv-A"):
        pass

    # Same object, refreshed timestamp.
    assert harness._turn_locks["conv-A"] is entry
    assert entry.last_used_monotonic >= first_used


# -----------------------------------------------------------------------------
# _sweep_turn_locks
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_reclaims_only_cold_unlocked_entries(harness: _Harness, monkeypatch: pytest.MonkeyPatch):
    """The sweep removes an entry that is unlocked AND idle past the threshold, but leaves a freshly-used
    entry and a still-locked entry alone."""
    clock = [1000.0]
    monkeypatch.setattr(slack_v1.time, "monotonic", lambda: clock[0])

    # `cold` was touched long ago; `fresh` is touched now; `held` is locked by a running turn.
    async with harness._conversation_turn_lock("cold"):
        pass
    cold_entry = harness._turn_locks["cold"]
    cold_entry.last_used_monotonic = clock[0] - _TURN_LOCK_IDLE_SECONDS - 1

    async with harness._conversation_turn_lock("fresh"):
        pass

    held_entry = slack_v1._LockEntry()
    await held_entry.lock.acquire()
    held_entry.last_used_monotonic = clock[0] - _TURN_LOCK_IDLE_SECONDS - 1
    harness._turn_locks["held"] = held_entry

    # Advance the clock past the sweep throttle window so the sweep actually runs.
    clock[0] += _TURN_LOCK_SWEEP_SECONDS + 1
    harness._last_lock_sweep_monotonic = clock[0] - _TURN_LOCK_SWEEP_SECONDS - 1
    harness._sweep_turn_locks()

    assert "cold" not in harness._turn_locks  # reclaimed
    assert "fresh" in harness._turn_locks  # recently used — kept
    assert "held" in harness._turn_locks  # still locked — kept
    held_entry.lock.release()


@pytest.mark.asyncio
async def test_sweep_is_throttled(harness: _Harness, monkeypatch: pytest.MonkeyPatch):
    """`_sweep_turn_locks` scans at most once per `_TURN_LOCK_SWEEP_SECONDS`; a call inside the window is a
    no-op even when a cold entry exists, and the entry is reclaimed on the first call past the window."""
    clock = [1000.0]
    monkeypatch.setattr(slack_v1.time, "monotonic", lambda: clock[0])

    cold = slack_v1._LockEntry()
    cold.last_used_monotonic = clock[0] - _TURN_LOCK_IDLE_SECONDS - 1
    harness._turn_locks["cold"] = cold
    harness._last_lock_sweep_monotonic = clock[0]

    # Second call inside the throttle window — no-op.
    clock[0] += _TURN_LOCK_SWEEP_SECONDS - 1
    harness._sweep_turn_locks()
    assert "cold" in harness._turn_locks

    # First call past the window — reclaims it.
    clock[0] += 2
    harness._sweep_turn_locks()
    assert "cold" not in harness._turn_locks


# -----------------------------------------------------------------------------
# _await_finalize_through_cancel
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_through_cancel_completes_then_raises_cancelled(harness: _Harness):
    """When the caller is cancelled mid-finalize, the finalize task still runs to completion and only then
    does `CancelledError` propagate — the lock is held for the finalize's real lifetime."""
    finalize_done = asyncio.Event()
    resume = asyncio.Event()

    async def finalize() -> None:
        await resume.wait()  # controlled await point
        finalize_done.set()

    async def caller() -> None:
        await harness._await_finalize_through_cancel(finalize())

    caller_task = asyncio.create_task(caller())
    await asyncio.sleep(0)  # let the finalize task start and hit `resume.wait()`

    caller_task.cancel()
    await asyncio.sleep(0)  # cancellation lands on the outer await

    assert not finalize_done.is_set()  # finalize is NOT abandoned
    resume.set()  # let the finalize finish

    with pytest.raises(asyncio.CancelledError):
        await caller_task
    assert finalize_done.is_set()  # finalize ran to completion


@pytest.mark.asyncio
async def test_finalize_through_cancel_surfaces_exception_over_cancellation(harness: _Harness):
    """When the finalize task raises AND the caller is cancelled, the finalize exception is surfaced — not
    `CancelledError` — so a durable-write failure is not hidden behind a generic cancellation."""
    resume = asyncio.Event()

    class _StorageError(RuntimeError):
        pass

    async def finalize() -> None:
        await resume.wait()
        raise _StorageError("ES put_conversation failed")

    async def caller() -> None:
        await harness._await_finalize_through_cancel(finalize())

    caller_task = asyncio.create_task(caller())
    await asyncio.sleep(0)

    caller_task.cancel()
    await asyncio.sleep(0)
    resume.set()

    with pytest.raises(_StorageError, match="ES put_conversation failed"):
        await caller_task


@pytest.mark.asyncio
async def test_finalize_through_cancel_exception_only(harness: _Harness):
    """When the finalize raises but the caller is NOT cancelled, the finalize exception propagates normally
    and no `CancelledError` is raised — the helper is transparent in the happy-error path."""

    class _StorageError(RuntimeError):
        pass

    async def finalize() -> None:
        raise _StorageError("boom")

    with pytest.raises(_StorageError, match="boom"):
        await harness._await_finalize_through_cancel(finalize())


@pytest.mark.asyncio
async def test_finalize_through_cancel_returns_result_on_happy_path(harness: _Harness):
    """With no cancellation and no exception, the helper returns the coroutine's result transparently."""

    async def finalize() -> str:
        return "committed"

    result = await harness._await_finalize_through_cancel(finalize())
    assert result == "committed"


# -----------------------------------------------------------------------------
# handle_event — the per-turn timeout wraps the whole locked region
# -----------------------------------------------------------------------------


def _mention(ts: str = "100.0", channel: str = "C1") -> dict:
    return {"type": "app_mention", "channel": channel, "ts": ts, "user": "U_ALICE", "text": "<@U_BOT> hi"}


@pytest.mark.asyncio
async def test_handle_event_timeout_cancels_whole_locked_region(monkeypatch: pytest.MonkeyPatch):
    """A `_locked_turn` that stalls past `SLACK_TURN_TIMEOUT_SECONDS` (e.g. a hung `sync_slack_thread`) is
    cancelled. The timeout wraps the entire locked region, not just `_run_turn`."""
    monkeypatch.setattr(slack_harness, "SLACK_TURN_TIMEOUT_SECONDS", 0.05)
    harness = _TimeoutHarness()

    cancelled = asyncio.Event()

    async def stalled() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    harness.locked_turn_impl = stalled

    with pytest.raises(asyncio.TimeoutError):
        await harness.handle_event(event=_mention())

    assert harness.locked_turn_started.is_set()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_handle_event_timeout_releases_lock_for_next_mention(monkeypatch: pytest.MonkeyPatch):
    """After a stalled turn times out, the per-thread lock releases cleanly so the next same-thread mention
    proceeds — a hung turn does not wedge the thread."""
    monkeypatch.setattr(slack_harness, "SLACK_TURN_TIMEOUT_SECONDS", 0.05)
    harness = _TimeoutHarness()

    async def stalled() -> None:
        await asyncio.sleep(10)

    harness.locked_turn_impl = stalled
    with pytest.raises(asyncio.TimeoutError):
        await harness.handle_event(event=_mention())

    # Second mention on the same thread — must not block on a wedged lock.
    second_ran = asyncio.Event()

    async def quick() -> None:
        second_ran.set()

    harness.locked_turn_impl = quick
    harness.locked_turn_started.clear()
    await asyncio.wait_for(harness.handle_event(event=_mention(ts="200.0")), timeout=1.0)

    assert second_ran.is_set()
