"""`SlackStreamer` — NDJSON consumption and Slack reply formatting.

Covers placeholder posting and in-flight metadata, `text_delta` buffering / flush cadence, soft-limit
paragraph splitting, the `<@user>` first-post prefix rule, tool-call / progress status-line rendering, the
`finish()` / `fail()` contracts, the empty-output `EMPTY_REPLY_NOTICE`, and `fail()` placeholder recovery.
"""

from __future__ import annotations

import json

import pytest

import prokaryotes.slack_v1.streaming as streaming_mod
from prokaryotes.slack_v1.streaming import SlackStreamer
from tests.unit_tests._slack_fakes import FakeSlackThreadClient

CHANNEL = "C_CHAN"
CONV_UUID = "c-stream"
THREAD_TS = "100.000000"
TURN_ID = "turn-1"


def _streamer(slack: FakeSlackThreadClient, *, reply_to_user_id: str | None = None) -> SlackStreamer:
    return SlackStreamer(
        channel_id=CHANNEL,
        conversation_uuid=CONV_UUID,
        slack_client=slack,
        thread_ts=THREAD_TS,
        turn_id=TURN_ID,
        reply_to_user_id=reply_to_user_id,
    )


def _delta(text: str) -> str:
    return json.dumps({"text_delta": text})


# -----------------------------------------------------------------------------
# placeholder + in-flight metadata
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_post_carries_in_flight_metadata():
    """Placeholder, continuation, and status-update posts all carry `prokaryotes_in_flight` metadata with the
    turn_id and conversation_uuid."""
    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000"]
    streamer = _streamer(slack)
    await streamer.post_placeholder()

    meta = slack.chat_post_calls[0]["metadata"]
    assert meta["event_type"] == "prokaryotes_in_flight"
    assert meta["event_payload"]["turn_id"] == TURN_ID
    assert meta["event_payload"]["conversation_uuid"] == CONV_UUID

    await streamer.consume(_delta("hello"))
    await streamer.finish()
    # Every chat.update during the turn also carried the metadata.
    assert all(c["metadata"]["event_type"] == "prokaryotes_in_flight" for c in slack.chat_update_calls)


# -----------------------------------------------------------------------------
# buffering and flush cadence
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_delta_buffers_and_flushes_on_1hz_cadence(monkeypatch: pytest.MonkeyPatch):
    """`text_delta` chunks buffer until the ~1 Hz flush hint fires; the flush writes the accumulated text to
    the placeholder post."""
    clock = [1000.0]
    monkeypatch.setattr(streaming_mod.time, "monotonic", lambda: clock[0])

    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000"]
    streamer = _streamer(slack)
    await streamer.post_placeholder()
    updates_after_placeholder = len(slack.chat_update_calls)

    # A small chunk arriving within the 1 Hz window does not force a flush.
    await streamer.consume(_delta("part one "))
    assert len(slack.chat_update_calls) == updates_after_placeholder

    # Advance past the flush interval — the next chunk flushes the whole buffer.
    clock[0] += SlackStreamer.FLUSH_INTERVAL_SECONDS + 0.1
    await streamer.consume(_delta("part two"))
    assert len(slack.chat_update_calls) > updates_after_placeholder
    assert slack.chat_update_calls[-1]["text"] == "part one part two"


@pytest.mark.asyncio
async def test_large_buffer_flushes_on_flush_chars(monkeypatch: pytest.MonkeyPatch):
    """A buffer crossing `FLUSH_CHARS` flushes immediately even inside the 1 Hz window."""
    clock = [1000.0]
    monkeypatch.setattr(streaming_mod.time, "monotonic", lambda: clock[0])

    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000"]
    streamer = _streamer(slack)
    await streamer.post_placeholder()
    before = len(slack.chat_update_calls)

    await streamer.consume(_delta("x" * (SlackStreamer.FLUSH_CHARS + 1)))
    assert len(slack.chat_update_calls) > before


# -----------------------------------------------------------------------------
# soft-limit splitting + <@user> prefix
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soft_limit_split_only_first_post_carries_prefix():
    """A reply past `SLACK_MESSAGE_SOFT_LIMIT` splits at a paragraph boundary; only the first message carries
    the `<@user>` prefix and all posts carry the same in-flight metadata."""
    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000", "201.000000"]
    streamer = _streamer(slack, reply_to_user_id="U_ALICE")
    await streamer.post_placeholder()

    para_a = "A" * 3000
    para_b = "B" * 3000
    await streamer.consume(_delta(f"{para_a}\n\n{para_b}"))
    posted = await streamer.finish()

    assert len(posted) == 2
    # First post prefixed, continuation not.
    assert posted[0].content.startswith("<@U_ALICE> ")
    assert not posted[1].content.startswith("<@U_ALICE> ")
    # Same metadata on every post.
    assert {c["metadata"]["event_type"] for c in slack.chat_post_calls} == {"prokaryotes_in_flight"}


@pytest.mark.asyncio
async def test_reply_to_user_id_set_prefixes_placeholder_and_final():
    """`reply_to_user_id` set → both the placeholder and the final message start with `<@user> `."""
    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000"]
    streamer = _streamer(slack, reply_to_user_id="U_ALICE")
    await streamer.post_placeholder()

    assert slack.chat_post_calls[0]["text"].startswith("<@U_ALICE> ")

    await streamer.consume(_delta("the answer"))
    posted = await streamer.finish()
    assert posted[0].content == "<@U_ALICE> the answer"


@pytest.mark.asyncio
async def test_dm_no_prefix_on_placeholder_or_final():
    """`reply_to_user_id=None` (DM) → no prefix on the placeholder or the final message."""
    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000"]
    streamer = _streamer(slack, reply_to_user_id=None)
    await streamer.post_placeholder()
    assert not slack.chat_post_calls[0]["text"].startswith("<@")

    await streamer.consume(_delta("dm answer"))
    posted = await streamer.finish()
    assert posted[0].content == "dm answer"


# -----------------------------------------------------------------------------
# tool-call / progress rendering
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_and_progress_render_then_clear_at_finish():
    """`tool_call` / `progress_message` render an ephemeral status line during the stream; `finish` strips it
    so the stored content is clean."""
    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000"]
    streamer = _streamer(slack)
    await streamer.post_placeholder()
    await streamer.consume(_delta("working on it"))

    await streamer.consume(json.dumps({"tool_call": {"name": "think"}}))
    # The status line is shown on the most recent post.
    assert "running think" in slack.chat_update_calls[-1]["text"]

    await streamer.consume(json.dumps({"progress_message": {"message": "almost done"}}))
    assert "almost done" in slack.chat_update_calls[-1]["text"]

    posted = await streamer.finish()
    # Status line is gone from the stored content.
    assert posted[0].content == "working on it"
    assert "running think" not in posted[0].content
    assert "almost done" not in posted[0].content


@pytest.mark.asyncio
async def test_context_pct_and_unknown_events_are_ignored():
    """`context_pct` and unknown event types are ignored — they do not error or post."""
    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000"]
    streamer = _streamer(slack)
    await streamer.post_placeholder()
    before = len(slack.chat_update_calls)

    await streamer.consume(json.dumps({"context_pct": 42}))
    await streamer.consume(json.dumps({"some_future_event": {"x": 1}}))

    assert len(slack.chat_update_calls) == before  # no posts triggered


# -----------------------------------------------------------------------------
# finish() contract
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finish_returns_one_posted_message_per_slack_post():
    """`finish()` returns one `PostedMessage` per Slack post with `source_id` = the post's `ts` and `content`
    = the exact text left in it — a multi-post reply round-trips losslessly."""
    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000", "201.000000"]
    streamer = _streamer(slack)
    await streamer.post_placeholder()
    await streamer.consume(_delta("A" * 3000 + "\n\n" + "B" * 3000))
    posted = await streamer.finish()

    assert [p.source_id for p in posted] == ["200.000000", "201.000000"]
    assert all(p.content for p in posted)


@pytest.mark.asyncio
async def test_finish_translates_markdown_bold_to_mrkdwn():
    """`finish()`'s stored content carries Slack `mrkdwn` formatting — `**bold**` becomes `*bold*`."""
    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000"]
    streamer = _streamer(slack)
    await streamer.post_placeholder()
    await streamer.consume(_delta("this is **bold** text"))
    posted = await streamer.finish()

    assert posted[0].content == "this is *bold* text"


@pytest.mark.asyncio
async def test_empty_output_rewrites_placeholder_to_empty_reply_notice():
    """Empty model output after a placeholder posted: `finish()` rewrites the placeholder in place to
    `EMPTY_REPLY_NOTICE` and returns it as a single `PostedMessage` — never an empty list."""
    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000"]
    streamer = _streamer(slack)
    await streamer.post_placeholder()
    # No text_delta consumed.
    posted = await streamer.finish()

    assert len(posted) == 1
    assert posted[0].source_id == "200.000000"
    assert posted[0].content == SlackStreamer.EMPTY_REPLY_NOTICE


# -----------------------------------------------------------------------------
# fail() contract
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_rewrites_placeholder_and_deletes_continuations():
    """`fail()` rewrites the placeholder to `FAILURE_NOTICE`, `chat.delete`s any continuation posts, and
    returns the notice as a single `PostedMessage`."""
    slack = FakeSlackThreadClient()
    slack.next_post_ts = ["200.000000", "201.000000"]
    streamer = _streamer(slack)
    await streamer.post_placeholder()
    # Force a continuation post by overflowing the soft limit.
    await streamer.consume(_delta("A" * 3000 + "\n\n" + "B" * 3000))

    notice = await streamer.fail()

    assert notice is not None
    assert notice.source_id == "200.000000"
    assert notice.content == SlackStreamer.FAILURE_NOTICE
    # The continuation post was deleted.
    assert slack.chat_delete_calls == ["201.000000"]


@pytest.mark.asyncio
async def test_fail_returns_none_when_no_placeholder_reached_slack():
    """`fail()` with no placeholder ever posted and no recoverable in-flight post returns `None` — the caller
    skips `_finalize_slack_turn` entirely."""
    slack = FakeSlackThreadClient(thread=[])  # nothing in the thread to recover
    streamer = _streamer(slack)

    notice = await streamer.fail()
    assert notice is None


@pytest.mark.asyncio
async def test_fail_recovers_placeholder_via_conversations_replies():
    """`fail()` recovers a Slack-accepted but streamer-unrecorded placeholder by scanning
    `conversations.replies` for this turn's `prokaryotes_in_flight` metadata."""
    # The placeholder post exists in the thread carrying this turn's metadata, but the streamer never recorded
    # its ts (post_placeholder cancelled mid-call after Slack accepted the post).
    slack = FakeSlackThreadClient(
        thread=[
            {"ts": "100.000000", "user": "U_ALICE", "text": "hi"},
            {
                "ts": "200.000000",
                "bot_id": "B_BOT",
                "text": "_…working_",
                "metadata": {"event_type": "prokaryotes_in_flight", "event_payload": {"turn_id": TURN_ID}},
            },
        ]
    )
    streamer = _streamer(slack)
    # _posts is empty — simulate the unrecorded-placeholder state.

    notice = await streamer.fail()

    assert notice is not None
    assert notice.source_id == "200.000000"
    assert notice.content == SlackStreamer.FAILURE_NOTICE


@pytest.mark.asyncio
async def test_clear_in_flight_metadata_drops_metadata_and_tolerates_failure(caplog: pytest.LogCaptureFixture):
    """`clear_in_flight_metadata` `chat.update`s each post with `metadata={}`; a failure is logged and
    swallowed."""
    from prokaryotes.slack_v1.streaming import PostedMessage

    slack = FakeSlackThreadClient()
    streamer = _streamer(slack)
    await streamer.clear_in_flight_metadata([PostedMessage(source_id="200.000000", content="reply")])

    assert slack.chat_update_calls[-1]["metadata"] == {}

    # A failing chat.update is logged, not raised.
    class _FailingClient(FakeSlackThreadClient):
        async def chat_update(self, **kwargs):
            raise RuntimeError("rate limited")

    failing = _FailingClient()
    streamer2 = _streamer(failing)
    with caplog.at_level("WARNING"):
        await streamer2.clear_in_flight_metadata([PostedMessage(source_id="200.000000", content="reply")])
    assert any("in-flight metadata" in rec.message for rec in caplog.records)
