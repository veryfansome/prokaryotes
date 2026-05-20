"""`SlackClient` shared rate limiter.

The limiter spaces calls per `(method, channel_id)` bucket at ~1 req/sec, runs different buckets concurrently,
and on a 429 records a workspace-global per-method `Retry-After` floor. Timing is exercised against a patched
clock and a recording `asyncio.sleep` so the suite never waits out real seconds.
"""

from __future__ import annotations

from urllib.parse import parse_qs

import httpx
import pytest

import prokaryotes.slack_v1.client as client_mod
from prokaryotes.slack_v1.client import _BUCKET_MIN_INTERVAL_SECONDS, SlackClient

BOT_TOKEN = "xoxb-test"


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch):
    """A patched `time.monotonic`. Tests advance `clock[0]` themselves; `asyncio.sleep` is replaced with a
    recorder that also advances the clock by the requested delay so reservations resolve deterministically."""
    clock = [1000.0]
    sleeps: list[float] = []

    monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock[0])

    async def _record_sleep(delay: float) -> None:
        sleeps.append(delay)
        if delay > 0:
            clock[0] += delay

    monkeypatch.setattr(client_mod.asyncio, "sleep", _record_sleep)
    return clock, sleeps


def _ok_handler(record: list[dict] | None = None):
    """A MockTransport handler that returns `ok: true` for every call, optionally recording form bodies."""

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append({k: v[0] for k, v in parse_qs(request.content.decode()).items()})
        return httpx.Response(200, json={"ok": True})

    return handler


def _make_client(handler) -> SlackClient:
    client = SlackClient()
    client._http = httpx.AsyncClient(
        base_url="https://slack.com/api/",
        transport=httpx.MockTransport(handler),
    )
    return client


@pytest.mark.asyncio
async def test_same_bucket_calls_are_spaced_one_per_second(fake_clock):
    """Two `chat.update` calls into the same `(method, channel)` bucket are serialized ~1 req/sec — the second
    call sleeps the bucket interval before issuing."""
    clock, sleeps = fake_clock
    client = _make_client(_ok_handler())

    await client.chat_update(bot_token=BOT_TOKEN, channel="C_A", ts="1.0", text="one")
    await client.chat_update(bot_token=BOT_TOKEN, channel="C_A", ts="2.0", text="two")

    # The second same-bucket call waited out the bucket spacing floor.
    assert any(abs(s - _BUCKET_MIN_INTERVAL_SECONDS) < 1e-6 for s in sleeps)
    await client.close()


@pytest.mark.asyncio
async def test_different_channels_do_not_serialize(fake_clock):
    """Two `chat.update` calls in different channels land in different buckets and do not wait on each
    other — neither sleeps."""
    clock, sleeps = fake_clock
    client = _make_client(_ok_handler())

    await client.chat_update(bot_token=BOT_TOKEN, channel="C_A", ts="1.0", text="a")
    await client.chat_update(bot_token=BOT_TOKEN, channel="C_B", ts="1.0", text="b")

    # Distinct buckets — neither call had to wait.
    assert all(s == 0 for s in sleeps) or sleeps == []
    await client.close()


@pytest.mark.asyncio
async def test_concurrent_same_bucket_calls_serialize_via_reservation(fake_clock):
    """Two concurrent callers into the same bucket reserve sequential slots — the limiter's lock keeps them
    from both reading a stale `next_allowed` and bursting Slack."""
    import asyncio

    clock, sleeps = fake_clock
    client = _make_client(_ok_handler())

    await asyncio.gather(
        client.chat_update(bot_token=BOT_TOKEN, channel="C_A", ts="1.0", text="a"),
        client.chat_update(bot_token=BOT_TOKEN, channel="C_A", ts="2.0", text="b"),
    )

    # One of the two reserved a slot a full bucket interval out.
    assert any(abs(s - _BUCKET_MIN_INTERVAL_SECONDS) < 1e-6 for s in sleeps)
    await client.close()


@pytest.mark.asyncio
async def test_429_retry_after_pauses_method_workspace_wide(fake_clock):
    """A 429 with `Retry-After: 3` on `chat.postMessage` records a workspace-global floor: the call retries
    after the indicated delay and the floor is stored against the method."""
    clock, sleeps = fake_clock

    state = {"served_429": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if not state["served_429"]:
            state["served_429"] = True
            return httpx.Response(429, headers={"Retry-After": "3"}, json={"ok": False})
        return httpx.Response(200, json={"ok": True})

    client = _make_client(handler)
    await client.chat_post_message(bot_token=BOT_TOKEN, channel="C_A", text="hi")

    # The method-global floor was recorded.
    assert "chat.postMessage" in client._method_retry_after_until
    # The retry waited at least the Retry-After window.
    assert any(s >= 3 for s in sleeps)
    await client.close()


@pytest.mark.asyncio
async def test_429_floor_is_per_method_not_blanket(fake_clock):
    """A 429 on `chat.postMessage` pauses `chat.postMessage` workspace-wide but does not touch the
    `chat.update` bucket — different methods keep independent floors."""
    clock, sleeps = fake_clock
    state = {"post_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("chat.postMessage"):
            state["post_calls"] += 1
            if state["post_calls"] == 1:
                # First postMessage call → 429 with a Retry-After floor.
                return httpx.Response(429, headers={"Retry-After": "5"}, json={"ok": False})
        return httpx.Response(200, json={"ok": True})

    client = _make_client(handler)

    await client.chat_post_message(bot_token=BOT_TOKEN, channel="C_A", text="hi")
    sleeps_after_post = list(sleeps)

    # chat.update on the same channel is unaffected by the postMessage floor.
    await client.chat_update(bot_token=BOT_TOKEN, channel="C_A", ts="1.0", text="x")

    # Only postMessage carries a floor; chat.update has none.
    assert "chat.postMessage" in client._method_retry_after_until
    assert "chat.update" not in client._method_retry_after_until
    # The big >=5s wait belongs to the postMessage retry, not the chat.update call.
    assert any(s >= 5 for s in sleeps_after_post)
    await client.close()


@pytest.mark.asyncio
async def test_method_keyed_calls_bucket_with_none_channel(fake_clock):
    """Non-channel-scoped methods (`users.info`) bucket by `(method, None)` and still space themselves."""
    clock, sleeps = fake_clock
    client = _make_client(_ok_handler())

    await client.users_info(bot_token=BOT_TOKEN, user="U_A")
    await client.users_info(bot_token=BOT_TOKEN, user="U_B")

    assert ("users.info", None) in client._bucket_next_allowed
    assert any(abs(s - _BUCKET_MIN_INTERVAL_SECONDS) < 1e-6 for s in sleeps)
    await client.close()
