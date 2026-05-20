"""`SlackBase._dispatch_event` and `_should_handle` — the inbound trigger gate.

`_dispatch_event` acks, dedupes, drops removal events and team mismatches, then defers to `_should_handle`.
`_should_handle` encodes every trigger rule: `app_mention` and `message.im` are handled; channel/mpim chatter,
bot self-mentions, hidden events, malformed events, and subtyped messages are dropped.
"""

from __future__ import annotations

import logging

import pytest

from prokaryotes.slack_v1 import SlackBase
from tests.unit_tests._slack_fakes import FakeRedis, envelope


class _RecordingHarness(SlackBase):
    """`SlackBase` subclass that records `handle_event` calls and the listener's ack calls."""

    def __init__(self) -> None:
        super().__init__(app_token="xapp-test", bot_token="xoxb-test")
        self.handled: list[dict] = []
        self.team_id = "T_TEAM"
        self.bot_user_id = "U_BOT"
        self.bot_id = "B_BOT"
        self.app_id = "A_APP"

    async def handle_event(self, *, event: dict) -> None:
        self.handled.append(event)


class _FakeSocketClient:
    """Captures `send_socket_mode_response` calls so the listener's ack-first ordering can be asserted."""

    def __init__(self) -> None:
        self.responses: list[object] = []

    async def send_socket_mode_response(self, response) -> None:
        self.responses.append(response)


class _FakeRequest:
    def __init__(self, envelope_dict: dict, envelope_id: str = "env-1") -> None:
        self.envelope_id = envelope_id
        self._envelope = envelope_dict

    def to_dict(self) -> dict:
        return self._envelope


@pytest.fixture
def harness() -> _RecordingHarness:
    h = _RecordingHarness()
    h._redis_client = FakeRedis()
    return h


# -----------------------------------------------------------------------------
# _dispatch_event
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_listener_acks_before_dispatch(harness: _RecordingHarness):
    """The listener acks via `send_socket_mode_response` before `_dispatch_event` runs the trigger gate."""
    client = _FakeSocketClient()
    event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "user": "U_ALICE", "text": "<@U_BOT> hi"}
    request = _FakeRequest(envelope(event))

    await harness._listener(client, request)
    await harness.drain_background_tasks()

    assert len(client.responses) == 1
    assert client.responses[0].envelope_id == "env-1"
    assert len(harness.handled) == 1


@pytest.mark.asyncio
async def test_app_mention_envelope_invokes_handle_event(harness: _RecordingHarness):
    event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "user": "U_ALICE", "text": "<@U_BOT> hi"}

    await harness._dispatch_event(envelope=envelope(event))
    await harness.drain_background_tasks()

    assert harness.handled == [event]


@pytest.mark.asyncio
async def test_non_events_api_envelope_ignored(harness: _RecordingHarness):
    await harness._dispatch_event(envelope={"type": "hello"})
    await harness.drain_background_tasks()

    assert harness.handled == []


@pytest.mark.parametrize("removal_type", ["tokens_revoked", "app_uninstalled"])
@pytest.mark.asyncio
async def test_removal_events_logged_and_dropped(
    harness: _RecordingHarness, removal_type: str, caplog: pytest.LogCaptureFixture
):
    """`tokens_revoked` / `app_uninstalled` are logged at error level; `handle_event` is not invoked and the
    harness keeps running."""
    with caplog.at_level(logging.ERROR, logger="prokaryotes.slack_v1"):
        await harness._dispatch_event(envelope=envelope({"type": removal_type}))
    await harness.drain_background_tasks()

    assert harness.handled == []
    assert any(r.levelno == logging.ERROR for r in caplog.records)


@pytest.mark.asyncio
async def test_duplicate_event_id_dropped(harness: _RecordingHarness):
    event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "user": "U_ALICE", "text": "<@U_BOT> hi"}
    env = envelope(event, event_id="Ev_DUP")

    await harness._dispatch_event(envelope=env)
    await harness._dispatch_event(envelope=env)
    await harness.drain_background_tasks()

    assert len(harness.handled) == 1


@pytest.mark.asyncio
async def test_team_id_mismatch_dropped_with_warning(harness: _RecordingHarness, caplog: pytest.LogCaptureFixture):
    """An envelope `team_id` that does not match `self.team_id` means a mis-configured token — drop with a
    warning."""
    event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "user": "U_ALICE", "text": "<@U_BOT> hi"}

    with caplog.at_level(logging.WARNING, logger="prokaryotes.slack_v1"):
        await harness._dispatch_event(envelope=envelope(event, team_id="T_OTHER"))
    await harness.drain_background_tasks()

    assert harness.handled == []
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# -----------------------------------------------------------------------------
# _should_handle — accept rules
# -----------------------------------------------------------------------------


def test_should_handle_accepts_top_level_app_mention(harness: _RecordingHarness):
    assert harness._should_handle(
        {"type": "app_mention", "channel": "C1", "ts": "1.0", "user": "U_ALICE", "text": "<@U_BOT> hi"}
    )


def test_should_handle_accepts_threaded_app_mention(harness: _RecordingHarness):
    assert harness._should_handle(
        {
            "type": "app_mention",
            "channel": "C1",
            "ts": "2.0",
            "thread_ts": "1.0",
            "user": "U_ALICE",
            "text": "<@U_BOT> follow-up",
        }
    )


def test_should_handle_accepts_dm_message(harness: _RecordingHarness):
    assert harness._should_handle(
        {"type": "message", "channel": "D1", "ts": "1.0", "user": "U_ALICE", "text": "hi", "channel_type": "im"}
    )


# -----------------------------------------------------------------------------
# _should_handle — reject rules
# -----------------------------------------------------------------------------


def test_should_handle_rejects_channel_thread_reply_without_mention(harness: _RecordingHarness):
    """A threaded channel `message` event without an `@`-mention is ordinary chatter — dropped."""
    assert not harness._should_handle(
        {
            "type": "message",
            "channel": "C1",
            "ts": "2.0",
            "thread_ts": "1.0",
            "user": "U_ALICE",
            "text": "more talk",
            "channel_type": "channel",
        }
    )


def test_should_handle_rejects_mpim_message_without_mention(harness: _RecordingHarness):
    assert not harness._should_handle(
        {
            "type": "message",
            "channel": "G1",
            "ts": "1.0",
            "user": "U_ALICE",
            "text": "group chatter",
            "channel_type": "mpim",
        }
    )


def test_should_handle_rejects_subtyped_message(harness: _RecordingHarness):
    """A `message` event with a subtype (e.g. `channel_join`) is not addressable content."""
    assert not harness._should_handle(
        {
            "type": "message",
            "subtype": "channel_join",
            "channel": "C1",
            "ts": "1.0",
            "user": "U_ALICE",
            "text": "joined",
        }
    )


def test_should_handle_rejects_hidden_event(harness: _RecordingHarness):
    """`hidden=true` events (e.g. `message_replied`) carry no real content and drop before the trigger
    branches."""
    assert not harness._should_handle(
        {
            "type": "message",
            "hidden": True,
            "channel": "C1",
            "ts": "1.0",
            "user": "U_ALICE",
            "text": "x",
            "channel_type": "im",
        }
    )


@pytest.mark.parametrize("missing", ["channel", "ts", "user", "text"])
def test_should_handle_rejects_missing_required_field(harness: _RecordingHarness, missing: str):
    """Events missing `channel`, `ts`, `user`, or `text` are dropped at the gate so `handle_event` can rely on
    them."""
    event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "user": "U_ALICE", "text": "<@U_BOT> hi"}
    del event[missing]
    assert not harness._should_handle(event)


def test_should_handle_accepts_empty_text(harness: _RecordingHarness):
    """`text` is required to be present but may be the empty string — only a missing `text` key is dropped."""
    assert harness._should_handle({"type": "app_mention", "channel": "C1", "ts": "1.0", "user": "U_ALICE", "text": ""})


def test_should_handle_rejects_app_mention_with_document_subtype(harness: _RecordingHarness):
    """An `app_mention` with `subtype=document_mention` (Canvas mention) has a different field shape — dropped."""
    assert not harness._should_handle(
        {
            "type": "app_mention",
            "subtype": "document_mention",
            "channel": "C1",
            "ts": "1.0",
            "user": "U_ALICE",
            "text": "<@U_BOT> canvas",
        }
    )


# -----------------------------------------------------------------------------
# _should_handle — bot self-guard
# -----------------------------------------------------------------------------


def test_self_guard_drops_app_mention_by_bot_user_id(harness: _RecordingHarness):
    """`event["user"] == bot_user_id` — the bot mentioning itself — drops before the `app_mention` accept."""
    assert not harness._should_handle(
        {"type": "app_mention", "channel": "C1", "ts": "1.0", "user": "U_BOT", "text": "<@U_BOT> echo"}
    )


def test_self_guard_drops_app_mention_by_bot_id(harness: _RecordingHarness):
    """A `bot_message`-subtype `app_mention` source whose `bot_id` matches `self.bot_id` drops."""
    assert not harness._should_handle(
        {
            "type": "app_mention",
            "channel": "C1",
            "ts": "1.0",
            "user": "U_ALICE",
            "text": "<@U_BOT> x",
            "bot_id": "B_BOT",
        }
    )


def test_self_guard_drops_app_mention_by_bot_profile_app_id(harness: _RecordingHarness):
    """An `app_mention` carrying `bot_profile.app_id == self.app_id` drops via the fallback identity check."""
    assert not harness._should_handle(
        {
            "type": "app_mention",
            "channel": "C1",
            "ts": "1.0",
            "user": "U_ALICE",
            "text": "<@U_BOT> x",
            "bot_profile": {"app_id": "A_APP"},
        }
    )


def test_foreign_bot_app_mention_dropped(harness: _RecordingHarness):
    """A foreign bot's `app_mention` (`bot_message` subtype, different `bot_id`/`app_id`) is dropped by the
    `subtype == "bot_message"` guard ahead of the `app_mention` accept branch."""
    assert not harness._should_handle(
        {
            "type": "app_mention",
            "subtype": "bot_message",
            "channel": "C1",
            "ts": "1.0",
            "user": "U_ALICE",
            "text": "<@U_BOT> from another bot",
            "bot_id": "B_FOREIGN",
            "bot_profile": {"app_id": "A_FOREIGN"},
        }
    )


def test_none_guard_does_not_drop_normal_human_event():
    """Regression: with `app_id` and `bot_id` unresolved (`None`), a normal human event with no `bot_profile`
    and no `bot_id` is still accepted — the guards short-circuit on the `self.<field>` truthiness check before
    comparing `None == None`."""
    harness = _RecordingHarness()
    harness._redis_client = FakeRedis()
    harness.app_id = None
    harness.bot_id = None

    assert harness._should_handle(
        {"type": "app_mention", "channel": "C1", "ts": "1.0", "user": "U_ALICE", "text": "<@U_BOT> hi"}
    )
