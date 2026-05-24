"""`SlackBase.on_start` / `on_stop` lifecycle and `scripts.slack.run` signal handling.

`on_start` resolves workspace identity via `auth.test` (+ a fallback `users.info` for `app_id`) and opens the
Socket Mode connection; `on_stop` disconnects the socket, drains in-flight `handle_event` tasks, and only then
closes the shared Slack HTTP client. `scripts.slack.run` installs SIGTERM/SIGINT handlers so `docker compose
stop` reaches the `on_stop` drain instead of terminating the process before the `finally` block.
"""

from __future__ import annotations

import asyncio
import os
import signal

import pytest

import prokaryotes.slack_v1 as slack_v1
import scripts.slack as slack_script
from prokaryotes.slack_v1 import SlackBase
from tests.unit_tests._slack_fakes import FakeRedis, FakeSearchClient, FakeSlackClient, FakeSocketModeClient


class _TestHarness(SlackBase):
    """`SlackBase` subclass wired to in-memory fakes for lifecycle tests."""

    def __init__(self, *, auth_ok: bool = True, app_id: str | None = "A_APP") -> None:
        super().__init__(app_token="xapp-test", bot_token="xoxb-test")
        self.slack_client = FakeSlackClient(auth_ok=auth_ok, app_id=app_id)
        self._redis_client = FakeRedis()
        self._search_client = FakeSearchClient()
        self.drain_observed_client_open: bool | None = None

    def ensure_runtime_clients(self) -> None:
        # Fakes are injected in __init__; the real client construction is skipped.
        pass

    async def handle_event(self, *, event: dict) -> None:
        pass


@pytest.fixture(autouse=True)
def _fake_socket(monkeypatch: pytest.MonkeyPatch):
    """Replace `build_socket_mode_client` so `on_start` opens a fake socket."""
    sockets: list[FakeSocketModeClient] = []

    def _build(app_token: str) -> FakeSocketModeClient:
        sock = FakeSocketModeClient()
        sockets.append(sock)
        return sock

    monkeypatch.setattr(slack_v1, "build_socket_mode_client", _build)
    return sockets


# -----------------------------------------------------------------------------
# on_start
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_start_resolves_workspace_identity_and_connects():
    harness = _TestHarness()

    await harness.on_start()

    assert harness.team_id == "T_TEAM"
    assert harness.bot_user_id == "U_BOT"
    assert harness.bot_id == "B_BOT"
    assert harness.team_name == "Acme"
    assert harness.app_id == "A_APP"
    assert harness.slack_client.resolve_app_id_calls == 1
    assert harness.socket is not None
    assert "connect" in harness.socket.calls
    assert harness._listener in harness.socket.socket_mode_request_listeners

    await harness.on_stop()


@pytest.mark.asyncio
async def test_on_start_raises_when_auth_test_fails():
    """`auth.test` returning `ok: false` makes `on_start` raise — the process is expected to exit."""
    harness = _TestHarness(auth_ok=False)

    with pytest.raises(RuntimeError, match="auth.test failed"):
        await harness.on_start()

    assert harness.socket is None


# -----------------------------------------------------------------------------
# on_stop
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_stop_disconnects_and_closes_socket():
    harness = _TestHarness()
    await harness.on_start()

    await harness.on_stop()

    assert harness.socket.calls == ["connect", "disconnect", "close"]
    assert harness.slack_client.closed is True


@pytest.mark.asyncio
async def test_on_stop_safe_when_on_start_failed_mid_way():
    """`on_stop` is safe to call when `on_start` never opened the socket (auth failure path)."""
    harness = _TestHarness(auth_ok=False)
    with pytest.raises(RuntimeError):
        await harness.on_start()

    # Should not raise even though `self.socket` is None.
    await harness.on_stop()

    assert harness.slack_client.closed is True


@pytest.mark.asyncio
async def test_on_stop_drains_in_flight_tasks_before_closing_slack_client():
    """A `handle_event` task that started just before shutdown still has the shared Slack HTTP client open
    while it runs — `on_stop` drains background tasks before `slack_client.close()`."""
    harness = _TestHarness()
    await harness.on_start()

    client_open_during_task = asyncio.Event()

    async def slow_turn() -> None:
        await asyncio.sleep(0.05)
        # The shared Slack client must still be usable here — the drain has not finished.
        if not harness.slack_client.closed:
            client_open_during_task.set()

    harness.background_and_forget(slow_turn())
    await harness.on_stop()

    assert client_open_during_task.is_set()
    assert harness.slack_client.closed is True


# -----------------------------------------------------------------------------
# scripts.slack.run — signal handling
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sig",
    [
        pytest.param(signal.SIGTERM, id="sigterm_docker_stop"),
        pytest.param(signal.SIGINT, id="sigint_ctrl_c"),
    ],
)
async def test_run_installs_signal_handlers_and_runs_on_stop(monkeypatch: pytest.MonkeyPatch, sig):
    """SIGTERM (docker compose stop) and SIGINT (Ctrl-C) both set the stop event so `on_stop` runs."""
    os.environ.pop("SLACK_HARNESS_IMPL", None)

    started = asyncio.Event()
    stopped = asyncio.Event()

    class _ScriptHarness:
        def __init__(self, *, impl: str, app_token: str, bot_token: str) -> None:
            self.impl = impl

        async def on_start(self) -> None:
            started.set()

        async def on_stop(self) -> None:
            stopped.set()

    monkeypatch.setattr(slack_script, "SlackHarness", _ScriptHarness)

    run_task = asyncio.create_task(slack_script.run(app_token="xapp-test", bot_token="xoxb-test"))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    signal.raise_signal(sig)

    await asyncio.wait_for(run_task, timeout=1.0)
    assert stopped.is_set()
