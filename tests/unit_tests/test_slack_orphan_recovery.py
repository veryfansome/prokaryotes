"""`_strip_in_flight_orphans` — the placeholder-then-crash recovery pre-pass.

A bot post carrying `metadata.event_type=prokaryotes_in_flight` is an orphan when its `ts` is NOT a stored
`source_id` (crash before finalize) and is preserved when its `ts` IS stored (crash after finalize, before the
metadata clear) — in the preserved case, the stale marker is `chat.update`-cleared best-effort so the next
replay does not see it. `chat.delete` and the stale-marker `chat.update` failures are logged and tolerated.
"""

from __future__ import annotations

import pytest

from prokaryotes.conversation_v1.models import Conversation, ConversationMessage
from prokaryotes.slack_v1.replay import _strip_in_flight_orphans
from tests.unit_tests._slack_fakes import FakeSlackThreadClient

BOT_USER = "U_BOT"
CONV_UUID = "c-orphan"
CHANNEL = "C_CHAN"


def _in_flight(ts: str, text: str = "_…working_") -> dict:
    return {
        "ts": ts,
        "bot_id": "B_BOT",
        "text": text,
        "metadata": {"event_type": "prokaryotes_in_flight", "event_payload": {"turn_id": "t1"}},
    }


def _stored(*source_ids: str) -> Conversation:
    return Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id=sid, author_id="U_ALICE", content="x") for sid in source_ids],
    )


@pytest.mark.asyncio
async def test_crash_before_finalize_orphan_is_deleted_and_dropped():
    """A bot in-flight message whose `ts` is NOT in stored is an orphan from a crashed turn: `chat.delete` is
    called and it is dropped from the returned thread."""
    thread = [
        {"ts": "100.000000", "user": "U_ALICE", "text": "hi"},
        _in_flight("101.000000"),
    ]
    slack = FakeSlackThreadClient()
    stored = _stored("100.000000")  # 101 not stored — orphan

    kept = await _strip_in_flight_orphans(thread, channel_id=CHANNEL, slack_client=slack, stored=stored)

    assert slack.chat_delete_calls == ["101.000000"]
    assert [m["ts"] for m in kept] == ["100.000000"]


@pytest.mark.asyncio
async def test_crash_after_finalize_message_is_kept_and_marker_cleared():
    """A bot in-flight message whose `ts` IS in stored was finalized — the post-finalize metadata clear just
    didn't run. It is kept (not `chat.delete`d) and the stale marker is best-effort `chat.update`-cleared with
    `metadata={}` so the next replay no longer sees the marker."""
    thread = [
        {"ts": "100.000000", "user": "U_ALICE", "text": "hi"},
        _in_flight("101.000000", text="real bot reply"),
    ]
    slack = FakeSlackThreadClient()
    stored = _stored("100.000000", "101.000000")  # 101 IS stored — finalized

    kept = await _strip_in_flight_orphans(thread, channel_id=CHANNEL, slack_client=slack, stored=stored)

    assert slack.chat_delete_calls == []
    assert [m["ts"] for m in kept] == ["100.000000", "101.000000"]
    # Stale marker cleared: chat.update on the finalized in-flight message with empty metadata, text preserved.
    assert slack.chat_update_calls == [{"ts": "101.000000", "text": "real bot reply", "metadata": {}}]


@pytest.mark.asyncio
async def test_message_without_metadata_is_treated_as_regular_post():
    """A bot message with no `prokaryotes_in_flight` metadata is a regular post — no special handling, kept
    as-is even when its `ts` is not in stored."""
    thread = [
        {"ts": "100.000000", "user": "U_ALICE", "text": "hi"},
        {"ts": "101.000000", "bot_id": "B_BOT", "text": "regular bot reply"},
    ]
    slack = FakeSlackThreadClient()
    stored = _stored("100.000000")

    kept = await _strip_in_flight_orphans(thread, channel_id=CHANNEL, slack_client=slack, stored=stored)

    assert slack.chat_delete_calls == []
    assert [m["ts"] for m in kept] == ["100.000000", "101.000000"]


@pytest.mark.asyncio
async def test_message_with_unrelated_metadata_event_type_is_kept():
    """A message carrying metadata with a different `event_type` is not an in-flight marker — kept untouched."""
    thread = [
        {
            "ts": "101.000000",
            "bot_id": "B_BOT",
            "text": "other",
            "metadata": {"event_type": "something_else", "event_payload": {}},
        },
    ]
    slack = FakeSlackThreadClient()
    stored = _stored("100.000000")

    kept = await _strip_in_flight_orphans(thread, channel_id=CHANNEL, slack_client=slack, stored=stored)

    assert slack.chat_delete_calls == []
    assert [m["ts"] for m in kept] == ["101.000000"]


@pytest.mark.asyncio
async def test_chat_delete_failure_is_logged_and_tolerated(caplog: pytest.LogCaptureFixture):
    """A `chat.delete` failure (e.g. permission error) is logged and tolerated — the orphan stays in Slack and
    is re-checked next turn. The pre-pass does not raise; the orphan is still dropped from the returned list."""
    thread = [
        {"ts": "100.000000", "user": "U_ALICE", "text": "hi"},
        _in_flight("101.000000"),
    ]
    slack = FakeSlackThreadClient()
    slack.chat_delete_error = RuntimeError("missing_scope")
    stored = _stored("100.000000")

    with caplog.at_level("WARNING"):
        kept = await _strip_in_flight_orphans(thread, channel_id=CHANNEL, slack_client=slack, stored=stored)

    assert slack.chat_delete_calls == ["101.000000"]  # attempted
    assert [m["ts"] for m in kept] == ["100.000000"]  # orphan dropped despite the failure
    assert any("chat.delete" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_stale_marker_clear_failure_is_logged_and_message_still_kept(caplog: pytest.LogCaptureFixture):
    """A `chat.update` failure on the stale-marker clear is logged and tolerated — the finalized message is
    still kept in the returned thread, and the next turn will retry the clear."""
    thread = [
        {"ts": "100.000000", "user": "U_ALICE", "text": "hi"},
        _in_flight("101.000000", text="real bot reply"),
    ]
    slack = FakeSlackThreadClient()
    slack.chat_update_error = RuntimeError("rate_limited")
    stored = _stored("100.000000", "101.000000")

    with caplog.at_level("WARNING"):
        kept = await _strip_in_flight_orphans(thread, channel_id=CHANNEL, slack_client=slack, stored=stored)

    assert slack.chat_delete_calls == []
    assert [m["ts"] for m in kept] == ["100.000000", "101.000000"]
    assert any("stale in-flight metadata" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_multiple_orphans_and_one_finalized_handled_independently():
    """A thread with two crash-before orphans and one finalized in-flight message: both orphans are deleted,
    the finalized one is kept and its stale marker is cleared."""
    thread = [
        {"ts": "100.000000", "user": "U_ALICE", "text": "hi"},
        _in_flight("101.000000"),  # orphan
        _in_flight("102.000000", text="finalized reply"),  # stored — kept
        _in_flight("103.000000"),  # orphan
    ]
    slack = FakeSlackThreadClient()
    stored = _stored("100.000000", "102.000000")

    kept = await _strip_in_flight_orphans(thread, channel_id=CHANNEL, slack_client=slack, stored=stored)

    assert sorted(slack.chat_delete_calls) == ["101.000000", "103.000000"]
    assert [m["ts"] for m in kept] == ["100.000000", "102.000000"]
    assert slack.chat_update_calls == [{"ts": "102.000000", "text": "finalized reply", "metadata": {}}]
