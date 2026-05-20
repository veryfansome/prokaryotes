"""`SlackClient.conversations_replies` cursor pagination + `paginate_until_ts` cap + `inclusive` flag.

Drives the real `SlackClient` against an in-memory `httpx.MockTransport` so cursor following, the pagination
cap, and the `inclusive=true` form-field can be asserted without a network.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx
import pytest

from prokaryotes.slack_v1.client import SlackClient

CHANNEL = "C_CHAN"
THREAD_TS = "100.000000"
BOT_TOKEN = "xoxb-test"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch):
    """The shared limiter spaces same-bucket calls ~1s apart; replace its `asyncio.sleep` with a no-op so
    cursor-pagination tests do not wait out real seconds."""
    import prokaryotes.slack_v1.client as client_mod

    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(client_mod.asyncio, "sleep", _instant)


def _make_client(handler) -> SlackClient:
    """Build a `SlackClient` whose underlying `httpx.AsyncClient` is backed by `handler`."""
    client = SlackClient()
    client._http = httpx.AsyncClient(
        base_url="https://slack.com/api/",
        transport=httpx.MockTransport(handler),
    )
    return client


def _replies_pages(pages: list[list[dict]]):
    """A MockTransport handler serving `conversations.replies` pages with cursor pagination.

    Page i is served while `cursor == str(i)`; the response carries `next_cursor = str(i+1)` until the last
    page. Records every request's parsed form body on `handler.requests`.
    """
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        requests.append(form)
        cursor = int(form.get("cursor", "0"))
        body: dict = {"ok": True, "messages": pages[cursor]}
        if cursor + 1 < len(pages):
            body["response_metadata"] = {"next_cursor": str(cursor + 1)}
        return httpx.Response(200, json=body)

    handler.requests = requests
    return handler


@pytest.mark.asyncio
async def test_cold_fetch_pages_until_has_more_false():
    """A 1500-reply thread cold-fetched (oldest=None) pages until there is no `next_cursor`."""
    pages = [[{"ts": f"{1000 + p * 200 + i}.000000"} for i in range(200)] for p in range(7)]
    pages.append([{"ts": f"{2400 + i}.000000"} for i in range(100)])  # 7*200 + 100 = 1500
    handler = _replies_pages(pages)
    client = _make_client(handler)

    messages = await client.conversations_replies(bot_token=BOT_TOKEN, channel=CHANNEL, ts=THREAD_TS)

    assert len(messages) == 1500
    # 8 pages → 8 requests; the first has no cursor.
    assert len(handler.requests) == 8
    assert "cursor" not in handler.requests[0]
    await client.close()


@pytest.mark.asyncio
async def test_paginate_until_ts_cap_stops_at_first_crossing_page():
    """`paginate_until_ts` stops pagination at the first page whose tail crosses the effective cutoff."""
    pages = [
        [{"ts": "100.000000"}, {"ts": "200.000000"}],
        [{"ts": "300.000000"}, {"ts": "400.000000"}],  # tail 400 >= cap 350 → stop here
        [{"ts": "500.000000"}, {"ts": "600.000000"}],  # never fetched
    ]
    handler = _replies_pages(pages)
    client = _make_client(handler)

    messages = await client.conversations_replies(
        bot_token=BOT_TOKEN, channel=CHANNEL, ts=THREAD_TS, paginate_until_ts="350.000000"
    )

    # Only the first two pages were fetched — the cap stopped pagination.
    assert [m["ts"] for m in messages] == ["100.000000", "200.000000", "300.000000", "400.000000"]
    assert len(handler.requests) == 2
    await client.close()


@pytest.mark.asyncio
async def test_post_compaction_fetch_starts_at_raw_window_boundary():
    """A post-compaction thread of 3000 historical + 20 raw-window replies: with `oldest` set to the
    raw-window boundary the fake returns only the ~20 raw-window messages including the boundary."""
    raw_window = [{"ts": f"{5000 + i}.000000"} for i in range(20)]
    handler = _replies_pages([raw_window])
    client = _make_client(handler)

    messages = await client.conversations_replies(
        bot_token=BOT_TOKEN,
        channel=CHANNEL,
        ts=THREAD_TS,
        oldest="5000.000000",
        inclusive=True,
    )

    assert len(messages) == 20
    assert messages[0]["ts"] == "5000.000000"  # boundary included
    # The request carried the bounded oldest and inclusive flag.
    form = handler.requests[0]
    assert form["oldest"] == "5000.000000"
    assert form["inclusive"] == "true"
    await client.close()


@pytest.mark.asyncio
async def test_inclusive_flag_sent_whenever_oldest_supplied():
    """Regression: omitting `inclusive` drops the boundary message. The client sends `inclusive=true`
    whenever the caller passes `inclusive=True` alongside `oldest`."""
    handler = _replies_pages([[{"ts": "500.000000"}]])
    client = _make_client(handler)

    await client.conversations_replies(
        bot_token=BOT_TOKEN, channel=CHANNEL, ts=THREAD_TS, oldest="500.000000", inclusive=True
    )
    assert handler.requests[0]["inclusive"] == "true"

    # Without inclusive, the flag is absent (Slack then excludes the boundary).
    handler2 = _replies_pages([[{"ts": "500.000000"}]])
    client2 = _make_client(handler2)
    await client2.conversations_replies(
        bot_token=BOT_TOKEN, channel=CHANNEL, ts=THREAD_TS, oldest="500.000000", inclusive=False
    )
    assert "inclusive" not in handler2.requests[0]

    await client.close()
    await client2.close()


@pytest.mark.asyncio
async def test_include_all_metadata_flag_sent_when_requested():
    """`include_all_metadata=True` is forwarded so `conversations.replies` returns the `metadata` field the
    orphan pre-pass needs."""
    handler = _replies_pages([[{"ts": "100.000000"}]])
    client = _make_client(handler)

    await client.conversations_replies(bot_token=BOT_TOKEN, channel=CHANNEL, ts=THREAD_TS, include_all_metadata=True)
    assert handler.requests[0]["include_all_metadata"] == "true"
    # The bot token is threaded onto every request.
    assert handler.requests[0]["token"] == BOT_TOKEN
    await client.close()


@pytest.mark.asyncio
async def test_messages_returned_in_fetch_order_across_pages():
    """Pages are flattened in fetch order — the returned list is the concatenation of every page."""
    pages = [
        [{"ts": "1.000000"}, {"ts": "2.000000"}],
        [{"ts": "3.000000"}],
    ]
    client = _make_client(_replies_pages(pages))

    messages = await client.conversations_replies(bot_token=BOT_TOKEN, channel=CHANNEL, ts=THREAD_TS)
    assert [m["ts"] for m in messages] == ["1.000000", "2.000000", "3.000000"]
    await client.close()


def test_replies_pages_handler_serves_json():
    """Sanity: the page handler builds well-formed Slack JSON bodies."""
    handler = _replies_pages([[{"ts": "1.0"}]])
    response = handler(httpx.Request("POST", "https://slack.com/api/conversations.replies", content=b""))
    assert json.loads(response.content)["ok"] is True
