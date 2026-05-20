"""`event_id` idempotency for inbound Socket Mode delivery.

Slack redelivers an envelope when the ack times out. `_claim_event_id` is a Redis `SET NX` claim: the first
delivery wins the claim and runs `handle_event`; every redelivery of the same `event_id` loses the claim and is
dropped.
"""

from __future__ import annotations

import asyncio

import pytest

from prokaryotes.slack_v1 import SlackBase
from tests.unit_tests._slack_fakes import FakeRedis, envelope


class _RecordingHarness(SlackBase):
    """`SlackBase` subclass that records every `handle_event` call instead of running a turn."""

    def __init__(self) -> None:
        super().__init__(app_token="xapp-test", bot_token="xoxb-test")
        self.handled: list[dict] = []
        self.team_id = "T_TEAM"
        self.bot_user_id = "U_BOT"
        self.bot_id = "B_BOT"
        self.app_id = "A_APP"

    async def handle_event(self, *, event: dict) -> None:
        self.handled.append(event)


def _mention(ts: str = "100.0") -> dict:
    return {"type": "app_mention", "channel": "C1", "ts": ts, "user": "U_ALICE", "text": "<@U_BOT> hi"}


@pytest.fixture
def harness() -> _RecordingHarness:
    h = _RecordingHarness()
    h._redis_client = FakeRedis()
    return h


@pytest.mark.asyncio
async def test_redelivered_envelope_dropped_on_second_delivery(harness: _RecordingHarness):
    """The same `event_id` delivered twice runs `handle_event` exactly once."""
    env = envelope(_mention(), event_id="Ev_DUP")

    await harness._dispatch_event(envelope=env)
    await harness._dispatch_event(envelope=env)
    await harness.drain_background_tasks()

    assert len(harness.handled) == 1


@pytest.mark.asyncio
async def test_distinct_event_ids_both_handled(harness: _RecordingHarness):
    """Two distinct `event_id`s each win their own claim and both run."""
    await harness._dispatch_event(envelope=envelope(_mention("100.0"), event_id="Ev_A"))
    await harness._dispatch_event(envelope=envelope(_mention("200.0"), event_id="Ev_B"))
    await harness.drain_background_tasks()

    assert len(harness.handled) == 2


@pytest.mark.asyncio
async def test_event_id_claim_race_only_one_winner(harness: _RecordingHarness):
    """Two concurrent dispatches of the same `event_id` — only one `handle_event` runs."""
    env = envelope(_mention(), event_id="Ev_RACE")

    await asyncio.gather(
        harness._dispatch_event(envelope=env),
        harness._dispatch_event(envelope=env),
    )
    await harness.drain_background_tasks()

    assert len(harness.handled) == 1


@pytest.mark.asyncio
async def test_claim_event_id_returns_true_then_false(harness: _RecordingHarness):
    """`_claim_event_id` is the underlying `SET NX` primitive: first call claims, second is rejected."""
    assert await harness._claim_event_id("Ev_X") is True
    assert await harness._claim_event_id("Ev_X") is False


@pytest.mark.asyncio
async def test_claim_event_id_uses_namespaced_key_with_ttl(harness: _RecordingHarness):
    """The claim key is namespaced and carries a TTL so stale claims expire."""
    await harness._claim_event_id("Ev_Y")

    key, _value, ex, nx = harness._redis_client.set_calls[-1]
    assert key == "slack_event_seen:Ev_Y"
    assert nx is True
    assert ex == 600
