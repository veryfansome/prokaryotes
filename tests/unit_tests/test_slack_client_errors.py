"""`SlackClient._call` raises `SlackApiError` on Slack's HTTP-200 `ok=false` application errors.

Slack reports method failures (`channel_not_found`, `message_not_found`, `not_in_channel`, `thread_not_found`,
…) as a 200 response carrying `{"ok": false, "error": ...}`, not an HTTP error status. `_call` must surface
those as errors rather than handing back a body that callers misread as empty data (`conversations.replies`) or
a missing `ts` (`chat.postMessage`). Driven against an in-memory `httpx.MockTransport` — no network.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx
import pytest

from prokaryotes.slack_v1.client import SlackApiError, SlackClient

BOT_TOKEN = "xoxb-test"
CHANNEL = "C_CHAN"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch):
    """The shared limiter spaces same-bucket calls ~1s apart; replace its `asyncio.sleep` with a no-op."""
    import prokaryotes.slack_v1.client as client_mod

    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr(client_mod.asyncio, "sleep", _instant)


def _client(payload: dict, *, status: int = 200) -> SlackClient:
    """A `SlackClient` whose transport always replies with `payload` at `status`."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=json.dumps(payload), headers={"Content-Type": "application/json"})

    client = SlackClient()
    client._http = httpx.AsyncClient(base_url="https://slack.com/api/", transport=httpx.MockTransport(handler))
    return client


@pytest.mark.asyncio
async def test_ok_false_raises_slack_api_error_with_method_and_error():
    client = _client({"ok": False, "error": "channel_not_found"})
    try:
        with pytest.raises(SlackApiError) as exc:
            await client.chat_post_message(bot_token=BOT_TOKEN, channel=CHANNEL, text="hi")
        assert exc.value.method == "chat.postMessage"
        assert exc.value.error == "channel_not_found"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ok_false_on_conversations_replies_raises_not_empty():
    """A `conversations.replies` failure must raise — not look like an empty thread, which would make reconcile
    false-`delete` the entire stored conversation."""
    client = _client({"ok": False, "error": "thread_not_found"})
    try:
        with pytest.raises(SlackApiError):
            await client.conversations_replies(bot_token=BOT_TOKEN, channel=CHANNEL, ts="100.000000")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ok_true_response_is_returned_unchanged():
    client = _client({"ok": True, "messages": [{"ts": "1.000000", "text": "hi"}]})
    try:
        messages = await client.conversations_replies(bot_token=BOT_TOKEN, channel=CHANNEL, ts="100.000000")
        assert messages == [{"ts": "1.000000", "text": "hi"}]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_metadata_and_blocks_are_json_encoded_in_form_body():
    """Slack requires `metadata` and `blocks` to be JSON-encoded strings inside the form body. Passing a raw dict
    or list to httpx's `data=` form encoder produces a Python `repr` (single quotes, list-flattened), which Slack
    rejects as `invalid_metadata_format` / `invalid_blocks`. This test captures the on-the-wire form and asserts
    that `json.loads(form["metadata"])` round-trips and that `blocks` is a single JSON-encoded array field.
    """
    captured: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(parse_qs(request.content.decode(), keep_blank_values=True))
        return httpx.Response(
            200,
            content=json.dumps({"ok": True, "ts": "1.0"}),
            headers={"Content-Type": "application/json"},
        )

    client = SlackClient()
    client._http = httpx.AsyncClient(base_url="https://slack.com/api/", transport=httpx.MockTransport(handler))
    try:
        metadata = {"event_type": "prokaryotes_in_flight", "event_payload": {"turn_id": "abc"}}
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "hi"}}]
        await client.chat_post_message(
            bot_token=BOT_TOKEN,
            channel=CHANNEL,
            text="hi",
            metadata=metadata,
            blocks=blocks,
        )
    finally:
        await client.close()

    # parse_qs returns lists per key; both fields must be a single value (not list-flattened) and JSON-decodable.
    assert captured["metadata"] == [json.dumps(metadata)]
    assert json.loads(captured["metadata"][0]) == metadata
    assert captured["blocks"] == [json.dumps(blocks)]
    assert json.loads(captured["blocks"][0]) == blocks


@pytest.mark.asyncio
async def test_resolve_app_id_degrades_to_none_on_users_info_failure():
    """`app_id` is only a fallback own-bot identity check — a `users.info` failure must degrade to `None` so
    startup is not blocked, not raise."""
    client = _client({"ok": False, "error": "user_not_found"})
    try:
        assert await client.resolve_app_id(bot_token=BOT_TOKEN, bot_user_id="U_BOT") is None
    finally:
        await client.close()
