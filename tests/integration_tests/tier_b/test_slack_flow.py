"""Tier B end-to-end suite for the Slack harness.

One `SlackHarness` bound to a synthetic workspace runs against real Redis + Elasticsearch with a fake
`SocketModeClient`, a fake Slack Web API (`FakeSlackThreadClient`), and a fake LLM client. Every scenario the
design doc's "Integration tests (Tier B)" section lists is exercised here: the channel-mention happy path, the
non-mention drop, same-thread continuation off the Redis fast path, the DM and mpim flows, ack-before-work,
cold-Redis rebuild, edit / delete reconcile, the two crash-recovery windows, the compaction trigger and
post-compaction replay, compaction interruption, same-thread serialization with post-race future-turn fidelity,
concurrent threads, the foreign bot, and `app_uninstalled`.

The harness reconciles a *real* Slack thread on each turn, so a test that wants a multi-turn conversation must
thread the bot's replies back into the fake thread itself — `_thread_bot_replies` does that, mirroring what
Slack would do when the bot's `chat.postMessage` lands in the thread.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from prokaryotes.conversation_v1.models import Conversation
from prokaryotes.harness_v1.slack import slack_conversation_uuid
from prokaryotes.slack_v1.streaming import SlackStreamer
from tests.integration_tests.tier_b._slack_tier_b import (
    APP_ID,
    BOT_ID,
    BOT_USER_ID,
    TEAM_ID,
    LLMRound,
    LLMScript,
)
from tests.unit_tests._slack_fakes import FakeSlackThreadClient

pytestmark = pytest.mark.integration


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------


def _mention(*, channel: str, ts: str, user: str = "U_ALICE", text: str = "<@U_BOT> hello", **extra) -> dict:
    """A top-level (or threaded, when `thread_ts` is passed in `extra`) channel `app_mention` event."""
    event = {"type": "app_mention", "channel": channel, "ts": ts, "user": user, "text": text}
    event.update(extra)
    return event


def _dm(*, channel: str, ts: str, user: str = "U_ALICE", text: str = "hello", **extra) -> dict:
    """A 1:1 DM `message` event (`channel_type: im`) — no `@`-mention needed."""
    event = {"type": "message", "channel": channel, "ts": ts, "user": user, "text": text, "channel_type": "im"}
    event.update(extra)
    return event


def _human_msg(*, ts: str, user: str, text: str) -> dict:
    """A human-authored Slack thread message as `conversations.replies` would return it."""
    return {"ts": ts, "user": user, "text": text}


def _script(text: str, *, input_tokens: int = 1000) -> LLMScript:
    """A one-round fake-LLM script that streams `text` as the assistant reply."""
    return LLMScript(rounds=[LLMRound(text_deltas=[text], stop_reason="end_turn", input_tokens=input_tokens)])


def _bot_post(post: dict, *, thread_ts: str) -> dict:
    """Render one recorded `chat.postMessage` call as the Slack thread message it became.

    The harness stores bot posts verbatim and reconciles the next turn against `conversations.replies`, so the
    threaded-back message must carry the exact `text` and the `bot_id` Slack attributes to a bot post.
    """
    return {
        "ts": post["ts"],
        "bot_id": BOT_ID,
        "text": post["text"],
        "thread_ts": thread_ts,
        "metadata": post.get("metadata"),
    }


def _thread_bot_replies(thread_client: FakeSlackThreadClient, *, thread_ts: str) -> list[dict]:
    """Append every post this turn made (placeholder rewritten to its final text) into `thread_client.thread`.

    Slack threads the bot's `chat.postMessage` results into the thread; the harness's next-turn reconcile reads
    them back via `conversations.replies`. The final text of each post is the last `chat.update` applied to that
    `ts` (the streamer rewrites the placeholder in place), so the threaded-back message uses that text.
    """
    final_text: dict[str, str] = {}
    final_meta: dict[str, dict | None] = {}
    for call in thread_client.chat_post_calls:
        final_text[call["ts"]] = call["text"]
        final_meta[call["ts"]] = call["metadata"]
    for call in thread_client.chat_update_calls:
        final_text[call["ts"]] = call["text"]
        final_meta[call["ts"]] = call["metadata"]
    posted = [
        {"ts": ts, "bot_id": BOT_ID, "text": final_text[ts], "thread_ts": thread_ts, "metadata": final_meta[ts]}
        for ts in (c["ts"] for c in thread_client.chat_post_calls)
        if ts not in set(thread_client.chat_delete_calls)
    ]
    thread_client.thread.extend(posted)
    thread_client.thread.sort(key=lambda m: m["ts"])
    return posted


async def _conv(harness, conversation_uuid: str) -> Conversation:
    """Load the active stored `Conversation` for `conversation_uuid` straight off the Redis fast path."""
    cached = await harness.redis_client.get(f"conversation:{conversation_uuid}")
    assert cached is not None, f"no cached conversation for {conversation_uuid}"
    return Conversation.model_validate_json(cached)


# -----------------------------------------------------------------------------
# First channel mention / non-mention drop
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_first_channel_mention_creates_conversation(slack_harness, thread_client):
    """A first channel `app_mention` creates a conversation, posts a placeholder prefixed with the trigger
    user's `<@…>` mention and carrying `prokaryotes_in_flight` metadata, finalizes the assistant message, and
    best-effort clears the metadata."""
    channel, ts = "C_FIRST", "100.000100"
    slack_harness.llm_client.set_script(_script("Hi there!"))
    thread_client.thread = [_human_msg(ts=ts, user="U_ALICE", text="<@U_BOT> hello")]

    await slack_harness.deliver(_mention(channel=channel, ts=ts), thread_client=thread_client)

    # The placeholder carried the <@user> prefix and in-flight metadata.
    placeholder = thread_client.chat_post_calls[0]
    assert placeholder["text"].startswith("<@U_ALICE> ")
    assert placeholder["metadata"]["event_type"] == "prokaryotes_in_flight"
    assert placeholder["thread_ts"] == ts

    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, ts)
    conv = await _conv(slack_harness, conversation_uuid)
    assert [m.author_id for m in conv.sorted_messages()] == ["U_ALICE", BOT_USER_ID]
    bot_msg = conv.message_by_source_id(placeholder["ts"])
    assert bot_msg is not None
    assert bot_msg.content.startswith("<@U_ALICE> ")
    assert "Hi there!" in bot_msg.content
    assert bot_msg.reply_to_source_id == ts

    # Best-effort metadata clear ran — the post's final chat.update carried metadata={}.
    clears = [c for c in thread_client.chat_update_calls if c["ts"] == placeholder["ts"] and c["metadata"] == {}]
    assert clears, "expected a metadata-clearing chat.update on the finalized post"

    # Persisted in Elasticsearch too.
    doc = await slack_harness.search_client.get_conversation(conv.snapshot_uuid)
    assert doc is not None and doc["is_compacted"] is False


@pytest.mark.asyncio(loop_scope="session")
async def test_non_mention_thread_reply_is_dropped(slack_harness, thread_client):
    """A plain (non-`@`) thread reply by another human never reaches `handle_event` — the bot does not post."""
    channel = "C_DROP"
    reply = {
        "type": "message",
        "channel": channel,
        "ts": "200.000200",
        "thread_ts": "200.000100",
        "user": "U_BOB",
        "text": "just chatting, no mention",
        "channel_type": "channel",
    }

    await slack_harness.deliver_via_socket(reply, thread_client=thread_client)

    assert thread_client.chat_post_calls == []


# -----------------------------------------------------------------------------
# Second mention — continuation off the Redis fast path
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_second_mention_continues_conversation(slack_harness, thread_client):
    """A second `@`-mention in the same thread continues the conversation off the Redis fast path; the reply is
    prefixed with the second trigger user's mention and prior chatter is visible to the model."""
    channel, root_ts = "C_CONT", "300.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)

    # Turn 1 — Alice's mention.
    slack_harness.llm_client.set_script(_script("first answer"))
    thread_client.thread = [_human_msg(ts=root_ts, user="U_ALICE", text="<@U_BOT> first question")]
    await slack_harness.deliver(_mention(channel=channel, ts=root_ts), thread_client=thread_client)
    _thread_bot_replies(thread_client, thread_ts=root_ts)

    # In-thread chatter that did NOT mention the bot — still visible as context, never triggered a turn.
    thread_client.thread.append(_human_msg(ts="300.000200", user="U_BOB", text="some side chatter"))
    thread_client.thread.sort(key=lambda m: m["ts"])

    # Turn 2 — Bob's mention in the same thread.
    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("second answer"))
    mention_2 = _mention(channel=channel, ts="300.000300", thread_ts=root_ts, user="U_BOB", text="<@U_BOB> more?")
    await slack_harness.deliver(mention_2, thread_client=thread_client)

    # The second turn's placeholder is prefixed with Bob (the second trigger user).
    second_placeholder = thread_client.chat_post_calls[-1]
    assert second_placeholder["text"].startswith("<@U_BOB> ")

    conv = await _conv(slack_harness, conversation_uuid)
    contents = [m.content for m in conv.sorted_messages()]
    # first mention, first bot reply, chatter, second mention, second bot reply.
    assert contents[0] == "<@U_BOT> first question"
    assert "first answer" in contents[1]
    assert contents[2] == "some side chatter"
    assert contents[3] == "<@U_BOB> more?"
    assert "second answer" in contents[4]

    # The second turn's projection saw the prior exchange — the chatter message was in the projected input.
    second_call = slack_harness.llm_client.stream_turn_calls[-1]
    rendered = " ".join(item.content or "" for item in second_call["items"])
    assert "some side chatter" in rendered
    assert "first answer" in rendered


# -----------------------------------------------------------------------------
# DM flow
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_dm_flow_no_mention_required(slack_harness, thread_client):
    """A top-level DM message creates a conversation and replies as a thread on it with no `<@user>` prefix; a
    subsequent in-thread DM reply continues it without needing an `@`-mention."""
    channel, root_ts = "D_DM1", "400.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)

    slack_harness.llm_client.set_script(_script("dm answer one"))
    thread_client.thread = [_human_msg(ts=root_ts, user="U_ALICE", text="hi bot")]
    await slack_harness.deliver(_dm(channel=channel, ts=root_ts), thread_client=thread_client)

    # No <@user> prefix in a DM.
    placeholder = thread_client.chat_post_calls[0]
    assert not placeholder["text"].startswith("<@")
    assert placeholder["thread_ts"] == root_ts
    _thread_bot_replies(thread_client, thread_ts=root_ts)

    # In-thread DM reply — continues without an @-mention.
    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("dm answer two"))
    reply = _dm(channel=channel, ts="400.000300", thread_ts=root_ts, text="follow up")
    await slack_harness.deliver(reply, thread_client=thread_client)

    conv = await _conv(slack_harness, conversation_uuid)
    contents = [m.content for m in conv.sorted_messages()]
    assert contents[0] == "hi bot"
    assert "dm answer one" in contents[1]
    assert contents[2] == "follow up"
    assert "dm answer two" in contents[3]


# -----------------------------------------------------------------------------
# mpim flow
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_mpim_chatter_dropped_mention_handled(slack_harness, thread_client):
    """mpim top-level chatter without an `@`-mention is dropped; an mpim `@`-mention creates a conversation and
    replies in a thread; non-mention thread replies are dropped; a follow-up `@`-mention continues it."""
    channel, root_ts = "G_MPIM", "500.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)

    # Non-mention mpim chatter — dropped at the gate.
    chatter = {
        "type": "message",
        "channel": channel,
        "ts": "500.000050",
        "user": "U_ALICE",
        "text": "group chatter",
        "channel_type": "mpim",
    }
    await slack_harness.deliver_via_socket(chatter, thread_client=thread_client)
    assert thread_client.chat_post_calls == []

    # mpim @-mention — creates the conversation.
    slack_harness.llm_client.set_script(_script("mpim answer one"))
    thread_client.thread = [_human_msg(ts=root_ts, user="U_ALICE", text="<@U_BOT> question")]
    await slack_harness.deliver(_mention(channel=channel, ts=root_ts, channel_type="mpim"), thread_client=thread_client)
    assert thread_client.chat_post_calls[0]["text"].startswith("<@U_ALICE> ")
    _thread_bot_replies(thread_client, thread_ts=root_ts)

    # Non-mention thread reply — dropped.
    thread_reply = {
        "type": "message",
        "channel": channel,
        "ts": "500.000300",
        "thread_ts": root_ts,
        "user": "U_BOB",
        "text": "no mention here",
        "channel_type": "mpim",
    }
    posts_before = len(thread_client.chat_post_calls)
    await slack_harness.deliver_via_socket(thread_reply, thread_client=thread_client)
    assert len(thread_client.chat_post_calls) == posts_before

    # Follow-up @-mention — continues the conversation.
    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("mpim answer two"))
    mention_2 = _mention(
        channel=channel, ts="500.000400", thread_ts=root_ts, user="U_BOB", text="<@U_BOB> again", channel_type="mpim"
    )
    await slack_harness.deliver(mention_2, thread_client=thread_client)

    conv = await _conv(slack_harness, conversation_uuid)
    contents = [m.content for m in conv.sorted_messages()]
    # The non-mention thread reply is gate-dropped (no turn fires) but it still lives in the Slack thread, so the
    # next turn's reconcile picks it up — that's how Slack delivers ambient channel context to the bot.
    assert "mpim answer one" in contents[1]
    assert contents[2] == "no mention here"
    assert contents[3] == "<@U_BOB> again"
    assert "mpim answer two" in contents[4]


# -----------------------------------------------------------------------------
# Ack before work
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_envelope_ack_before_llm_work(slack_harness, thread_client):
    """The Socket Mode envelope is acked before any LLM work begins — asserted via the fake socket's response
    log captured at the moment the LLM stream is first touched."""
    channel, ts = "C_ACK", "600.000100"
    slack_harness.llm_client.set_script(_script("ack answer"))
    thread_client.thread = [_human_msg(ts=ts, user="U_ALICE", text="<@U_BOT> hi")]

    acked_before_llm: list[bool] = []
    original_stream_turn = slack_harness.llm_client.stream_turn

    def _stream_turn_spy(*args, **kwargs):
        # By the time the LLM stream is constructed the listener has long since acked.
        acked_before_llm.append(len(socket_responses) > 0)
        return original_stream_turn(*args, **kwargs)

    slack_harness.llm_client.stream_turn = _stream_turn_spy

    class _Socket:
        def __init__(self) -> None:
            self.responses: list[object] = []

        async def send_socket_mode_response(self, response) -> None:
            self.responses.append(response)

    socket = _Socket()
    socket_responses = socket.responses

    # Fresh event_id per run — `slack_event_seen:*` has a 10-minute TTL and isn't touched by the per-session
    # state cleanup (those keys are not test-owned). A fixed event_id would otherwise dedupe across test runs.
    event_id = f"Ev_ACK_{uuid4().hex[:8]}"

    class _Request:
        envelope_id = "env-ack-1"

        @staticmethod
        def to_dict() -> dict:
            from tests.unit_tests._slack_fakes import envelope

            return envelope(_mention(channel=channel, ts=ts), event_id=event_id)

    slack_harness._thread_client = thread_client
    await slack_harness._listener(socket, _Request())
    await slack_harness.drain_background_tasks()
    slack_harness.llm_client.stream_turn = original_stream_turn

    assert len(socket.responses) == 1
    assert socket.responses[0].envelope_id == "env-ack-1"
    assert acked_before_llm == [True]


# -----------------------------------------------------------------------------
# Cold-Redis rebuild
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_cold_redis_rebuilds_from_elasticsearch(slack_harness, thread_client):
    """After the Redis key is evicted, a fresh `@`-mention rebuilds the active snapshot from Elasticsearch via
    `find_latest_active_snapshot_uuid` and reconciles cleanly."""
    channel, root_ts = "C_COLD", "700.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)

    slack_harness.llm_client.set_script(_script("cold answer one"))
    thread_client.thread = [_human_msg(ts=root_ts, user="U_ALICE", text="<@U_BOT> first")]
    await slack_harness.deliver(_mention(channel=channel, ts=root_ts), thread_client=thread_client)
    _thread_bot_replies(thread_client, thread_ts=root_ts)
    snapshot_before = (await _conv(slack_harness, conversation_uuid)).snapshot_uuid

    # `put_conversation` uses `refresh=False`, so the new snapshot isn't searchable yet. Force a refresh so the
    # next turn's `find_latest_active_snapshot_uuid` can locate it after the Redis evict.
    await slack_harness.search_client.es.indices.refresh(index="conversations")

    # Evict the Redis fast path — the next turn must rebuild from ES.
    await slack_harness.redis_client.delete(f"conversation:{conversation_uuid}")
    assert await slack_harness.redis_client.get(f"conversation:{conversation_uuid}") is None

    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("cold answer two"))
    mention_2 = _mention(channel=channel, ts="700.000300", thread_ts=root_ts, text="<@U_BOT> second")
    await slack_harness.deliver(mention_2, thread_client=thread_client)

    conv = await _conv(slack_harness, conversation_uuid)
    assert conv.snapshot_uuid == snapshot_before  # same snapshot — reattached, not branched
    contents = [m.content for m in conv.sorted_messages()]
    assert "cold answer one" in contents[1]
    assert contents[2] == "<@U_BOT> second"
    assert "cold answer two" in contents[3]


@pytest.mark.asyncio(loop_scope="session")
async def test_cold_redis_no_snapshot_starts_fresh(slack_harness, thread_client):
    """A mention in a thread with no stored snapshot at all starts a fresh `Conversation`."""
    channel, ts = "C_FRESH", "750.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, ts)
    assert await slack_harness.redis_client.get(f"conversation:{conversation_uuid}") is None

    slack_harness.llm_client.set_script(_script("fresh answer"))
    thread_client.thread = [_human_msg(ts=ts, user="U_ALICE", text="<@U_BOT> brand new")]
    await slack_harness.deliver(_mention(channel=channel, ts=ts), thread_client=thread_client)

    conv = await _conv(slack_harness, conversation_uuid)
    assert conv.bot_author_id == BOT_USER_ID
    assert [m.author_id for m in conv.sorted_messages()] == ["U_ALICE", BOT_USER_ID]


# -----------------------------------------------------------------------------
# Edit / delete reconcile
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_edited_message_reconciles_in_place(slack_harness, thread_client):
    """An earlier thread message edited between turns reconciles as an in-place `edit` on the next `@`-mention —
    no branch."""
    channel, root_ts = "C_EDIT", "800.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)

    slack_harness.llm_client.set_script(_script("edit answer one"))
    thread_client.thread = [_human_msg(ts=root_ts, user="U_ALICE", text="<@U_BOT> original text")]
    await slack_harness.deliver(_mention(channel=channel, ts=root_ts), thread_client=thread_client)
    _thread_bot_replies(thread_client, thread_ts=root_ts)
    snapshot_before = (await _conv(slack_harness, conversation_uuid)).snapshot_uuid

    # Alice edits her original message; the next replies fetch returns the new text at the same `ts`.
    for m in thread_client.thread:
        if m["ts"] == root_ts:
            m["text"] = "<@U_BOT> EDITED text"

    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("edit answer two"))
    mention_2 = _mention(channel=channel, ts="800.000300", thread_ts=root_ts, text="<@U_BOT> next")
    await slack_harness.deliver(mention_2, thread_client=thread_client)

    conv = await _conv(slack_harness, conversation_uuid)
    assert conv.snapshot_uuid == snapshot_before  # in-place edit, no branch
    assert conv.message_by_source_id(root_ts).content == "<@U_BOT> EDITED text"


@pytest.mark.asyncio(loop_scope="session")
async def test_deleted_message_tombstoned(slack_harness, thread_client):
    """A thread message that has been reconciled into stored and is later deleted from Slack is tombstoned in
    place on the next `@`-mention — no branch.

    `sync_slack_thread`'s inner filter drops messages past the trigger ts that aren't already in stored, so
    Bob's message must first land in stored via a turn whose trigger is later than Bob's ts before the
    delete-and-replay can exercise the tombstone path.
    """
    channel, root_ts = "C_DEL", "850.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)

    # Turn 1 — Alice's mention at the root seeds the conversation.
    slack_harness.llm_client.set_script(_script("del answer one"))
    thread_client.thread = [_human_msg(ts=root_ts, user="U_ALICE", text="<@U_BOT> keep me")]
    await slack_harness.deliver(_mention(channel=channel, ts=root_ts), thread_client=thread_client)
    _thread_bot_replies(thread_client, thread_ts=root_ts)

    # Bob posts a non-mention thread reply — no turn fires, but Slack writes it to the thread.
    thread_client.thread.append(_human_msg(ts="850.000200", user="U_BOB", text="delete me later"))
    thread_client.thread.sort(key=lambda m: m["ts"])

    # Turn 2 — Alice mentions again. Reconcile picks up Bob's message (its ts is now < trigger ts).
    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("del answer two"))
    mention_2 = _mention(channel=channel, ts="850.000400", thread_ts=root_ts, text="<@U_BOT> next")
    await slack_harness.deliver(mention_2, thread_client=thread_client)
    _thread_bot_replies(thread_client, thread_ts=root_ts)
    conv = await _conv(slack_harness, conversation_uuid)
    assert conv.message_by_source_id("850.000200") is not None  # Bob is now in stored.

    # Bob's message is deleted from Slack — the next replies fetch no longer returns it.
    thread_client.thread = [m for m in thread_client.thread if m["ts"] != "850.000200"]

    # Turn 3 — another mention. Reconcile sees Bob missing and tombstones the stored entry.
    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("del answer three"))
    mention_3 = _mention(channel=channel, ts="850.000600", thread_ts=root_ts, text="<@U_BOT> after delete")
    await slack_harness.deliver(mention_3, thread_client=thread_client)

    conv = await _conv(slack_harness, conversation_uuid)
    deleted = conv.message_by_source_id("850.000200")
    assert deleted is not None and deleted.deleted is True


# -----------------------------------------------------------------------------
# Crash recovery
# -----------------------------------------------------------------------------


class _SimulatedCrash(BaseException):
    """A `BaseException` (not `Exception`) so it escapes `_run_turn`'s `except (Exception, CancelledError)` —
    simulating a process death after the placeholder posted but before `_finalize_slack_turn`."""


@pytest.mark.asyncio(loop_scope="session")
async def test_crash_before_finalize_orphan_is_stripped(slack_harness, thread_client, monkeypatch):
    """A turn crashes after `post_placeholder()` but before `_finalize_slack_turn`. On the next `@`-mention the
    orphan pre-pass `chat.delete`s the placeholder (its `ts` is not in stored) and reconciles cleanly — no
    orphan bot `ConversationMessage` lands in storage."""
    channel, root_ts = "C_CRASH1", "900.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)
    thread_client.thread = [_human_msg(ts=root_ts, user="U_ALICE", text="<@U_BOT> trigger crash")]
    slack_harness.llm_client.set_script(_script("never persisted"))

    # finish() raising a BaseException leaves the placeholder posted but escapes _run_turn's except clause, so
    # _finalize_slack_turn never runs.
    async def _crash(self):
        raise _SimulatedCrash("simulated process death before finalize")

    monkeypatch.setattr(SlackStreamer, "finish", _crash)
    with pytest.raises(_SimulatedCrash):
        await slack_harness.deliver(_mention(channel=channel, ts=root_ts), thread_client=thread_client)
    monkeypatch.undo()

    # The placeholder post is an orphan: posted to Slack, never finalized into storage. `sync_slack_thread`
    # *did* cache the user's mention before the crash, so the Redis key holds a Conversation that contains
    # only the user message — no bot ConversationMessage for the placeholder.
    crashed_conv = await _conv(slack_harness, conversation_uuid)
    assert [m.author_id for m in crashed_conv.sorted_messages() if not m.deleted] == ["U_ALICE"]
    orphan = _thread_bot_replies(thread_client, thread_ts=root_ts)
    assert len(orphan) == 1

    # Next mention — the orphan pre-pass deletes the placeholder.
    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("recovered answer"))
    next_thread = FakeSlackThreadClient(
        thread=[
            _human_msg(ts=root_ts, user="U_ALICE", text="<@U_BOT> trigger crash"),
            orphan[0],
            _human_msg(ts="900.000300", user="U_ALICE", text="<@U_BOT> try again"),
        ]
    )
    mention_2 = _mention(channel=channel, ts="900.000300", thread_ts=root_ts, text="<@U_BOT> try again")
    await slack_harness.deliver(mention_2, thread_client=next_thread)

    assert orphan[0]["ts"] in next_thread.chat_delete_calls
    conv = await _conv(slack_harness, conversation_uuid)
    # No orphan bot message — only the two human mentions and the one real reply.
    assert conv.message_by_source_id(orphan[0]["ts"]) is None
    contents = [m.content for m in conv.sorted_messages()]
    assert "recovered answer" in contents[-1]


@pytest.mark.asyncio(loop_scope="session")
async def test_crash_after_finalize_preserves_reply(slack_harness, thread_client, monkeypatch):
    """A turn persists the bot reply but crashes before `clear_in_flight_metadata`. On the next `@`-mention the
    orphan pre-pass sees the `ts` IS in stored, preserves the message, and the valid reply survives."""
    channel, root_ts = "C_CRASH2", "950.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)
    thread_client.thread = [_human_msg(ts=root_ts, user="U_ALICE", text="<@U_BOT> first")]
    slack_harness.llm_client.set_script(_script("durable reply"))

    # Skip the post-finalize metadata clear entirely — the in-flight metadata stays on the persisted post.
    async def _skip_clear(self, posted):
        return None

    monkeypatch.setattr(SlackStreamer, "clear_in_flight_metadata", _skip_clear)
    await slack_harness.deliver(_mention(channel=channel, ts=root_ts), thread_client=thread_client)
    monkeypatch.undo()

    bot_ts = thread_client.chat_post_calls[0]["ts"]
    posted = _thread_bot_replies(thread_client, thread_ts=root_ts)
    # The persisted post still carries in-flight metadata (the clear was skipped).
    assert posted[0]["metadata"]["event_type"] == "prokaryotes_in_flight"
    conv = await _conv(slack_harness, conversation_uuid)
    assert conv.message_by_source_id(bot_ts) is not None  # finalized — in storage

    # Next mention — the orphan pre-pass keeps the message (ts is in stored) and does not delete it.
    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("second reply"))
    mention_2 = _mention(channel=channel, ts="950.000300", thread_ts=root_ts, text="<@U_BOT> second")
    await slack_harness.deliver(mention_2, thread_client=thread_client)

    assert bot_ts not in thread_client.chat_delete_calls  # never deleted — it was finalized
    conv = await _conv(slack_harness, conversation_uuid)
    assert conv.message_by_source_id(bot_ts) is not None
    assert "durable reply" in conv.message_by_source_id(bot_ts).content


# -----------------------------------------------------------------------------
# Compaction
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_compaction_trigger_commits_child_snapshot(slack_harness, thread_client):
    """Pushing `on_usage` past the threshold triggers the background compaction swap; the committed child
    snapshot becomes the active head and the next `@`-mention uses it."""
    channel, root_ts = "C_COMPACT", "1000.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)

    # Two ordinary turns to build a multi-message raw window the compactor has something to summarize.
    slack_harness.llm_client.set_script(_script("compact answer one"))
    thread_client.thread = [_human_msg(ts=root_ts, user="U_ALICE", text="<@U_BOT> turn one")]
    await slack_harness.deliver(_mention(channel=channel, ts=root_ts), thread_client=thread_client)
    _thread_bot_replies(thread_client, thread_ts=root_ts)

    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("compact answer two"))
    thread_client.thread.append(_human_msg(ts="1000.000300", user="U_ALICE", text="<@U_BOT> turn two"))
    thread_client.thread.sort(key=lambda m: m["ts"])
    await slack_harness.deliver(
        _mention(channel=channel, ts="1000.000300", thread_ts=root_ts, text="<@U_BOT> turn two"),
        thread_client=thread_client,
    )
    _thread_bot_replies(thread_client, thread_ts=root_ts)
    parent_snapshot = (await _conv(slack_harness, conversation_uuid)).snapshot_uuid

    # Third turn — a huge `input_tokens` trips the COMPACTION_TOKEN_THRESHOLD_PCT=1 threshold.
    slack_harness.llm_client.reset()
    over_threshold = LLMScript(rounds=[LLMRound(text_deltas=["over threshold"], input_tokens=10**9)])
    slack_harness.llm_client.set_script(over_threshold)
    slack_harness.llm_client._script.summary_text = "COMPACTED SUMMARY"
    thread_client.thread.append(_human_msg(ts="1000.000500", user="U_ALICE", text="<@U_BOT> turn three"))
    thread_client.thread.sort(key=lambda m: m["ts"])
    await slack_harness.deliver(
        _mention(channel=channel, ts="1000.000500", thread_ts=root_ts, text="<@U_BOT> turn three"),
        thread_client=thread_client,
    )

    # The background compaction task was scheduled — drain it and wait for the swap to commit.
    await slack_harness.drain_background_tasks()
    for _ in range(50):
        docs = await slack_harness.search_client.find_all_conversation_docs(conversation_uuid)
        child = [d for d in docs if d.get("parent_snapshot_uuid") == parent_snapshot]
        if child:
            break
        await asyncio.sleep(0.1)
    head = await slack_harness.search_client.find_latest_active_snapshot_uuid(conversation_uuid)
    assert head is not None
    head_doc = await slack_harness.search_client.get_conversation(head)
    # The active head is the compacted child — it carries the ancestor summary.
    assert head_doc["parent_snapshot_uuid"] == parent_snapshot
    assert head_doc["ancestor_summaries"]


@pytest.mark.asyncio(loop_scope="session")
async def test_post_compaction_replay_bounded_fetch(slack_harness, thread_client):
    """After a thread is compacted, a new `@`-mention's bounded `oldest` + `inclusive=True` fetch returns only
    the raw suffix, reconcile classifies `append`, no historical messages re-inflate, and `ancestor_summaries`
    are preserved. A parallel run without `inclusive=True` drops the boundary message."""
    channel, root_ts = "C_PC", "1100.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)

    # Stage a compacted parent + a committed child snapshot directly in ES so the next turn reconciles against
    # the bounded raw window.
    boundary_ts = "1100.000400"
    child = Conversation(
        conversation_uuid=conversation_uuid,
        bot_author_id=BOT_USER_ID,
        ancestor_summaries=["earlier history was compacted into this summary"],
        messages=[],
    )
    from prokaryotes.conversation_v1.models import ConversationMessage

    child.messages = [
        ConversationMessage(source_id=boundary_ts, author_id="U_ALICE", content="<@U_BOT> boundary message"),
    ]
    # `refresh="wait_for"` so the next turn's `find_latest_active_snapshot_uuid` search sees the staged child.
    await slack_harness.search_client.put_conversation(child, refresh="wait_for")

    slack_harness.llm_client.set_script(_script("post-compaction answer"))
    thread_client.thread = [
        _human_msg(ts=boundary_ts, user="U_ALICE", text="<@U_BOT> boundary message"),
        _human_msg(ts="1100.000600", user="U_ALICE", text="<@U_BOT> after compaction"),
    ]
    await slack_harness.deliver(
        _mention(channel=channel, ts="1100.000600", thread_ts=root_ts, text="<@U_BOT> after compaction"),
        thread_client=thread_client,
    )

    # The replies fetch was bounded — oldest = boundary ts, inclusive so the boundary lands in the result.
    replies_call = thread_client.replies_calls[0]
    assert replies_call["oldest"] == boundary_ts
    assert replies_call["inclusive"] is True

    conv = await _conv(slack_harness, conversation_uuid)
    assert conv.ancestor_summaries == ["earlier history was compacted into this summary"]
    # Storage stays bounded — only the raw window, not the compacted prefix.
    assert conv.message_by_source_id(boundary_ts) is not None
    assert "post-compaction answer" in conv.sorted_messages()[-1].content

    # Parallel regression: the same fetch WITHOUT inclusive drops the boundary message exactly at `oldest`.
    excludes_boundary = await thread_client.conversations_replies(
        channel=channel, ts=root_ts, oldest=boundary_ts, inclusive=False
    )
    assert all(m["ts"] != boundary_ts for m in excludes_boundary), "non-inclusive fetch must drop the boundary"
    includes_boundary = await thread_client.conversations_replies(
        channel=channel, ts=root_ts, oldest=boundary_ts, inclusive=True
    )
    assert any(m["ts"] == boundary_ts for m in includes_boundary)


@pytest.mark.asyncio(loop_scope="session")
async def test_compaction_interruption_recovers_parent(slack_harness, thread_client):
    """A crash between the Redis CAS swap and the child-committed update leaves a `pending` child in ES. On the
    next `@`-mention `find_latest_active_snapshot_uuid` returns the parent — the pending child is filtered out —
    and the harness re-compacts cleanly while the pending doc remains as an inert artifact."""
    channel, root_ts = "C_CI", "1200.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)

    parent = Conversation(
        conversation_uuid=conversation_uuid,
        bot_author_id=BOT_USER_ID,
        messages=[],
    )
    from prokaryotes.conversation_v1.models import ConversationMessage

    parent.messages = [
        ConversationMessage(source_id="1200.000100", author_id="U_ALICE", content="<@U_BOT> parent message"),
    ]
    await slack_harness.search_client.put_conversation(parent)
    # An abandoned pending child — the compactor crashed before marking it committed.
    pending_child = Conversation(
        conversation_uuid=conversation_uuid,
        parent_snapshot_uuid=parent.snapshot_uuid,
        bot_author_id=BOT_USER_ID,
        messages=list(parent.messages),
    )
    await slack_harness.search_client.put_conversation(pending_child, compaction_state="pending", refresh="wait_for")

    head = await slack_harness.search_client.find_latest_active_snapshot_uuid(conversation_uuid)
    # The pending child is filtered out — the committed parent is the head.
    assert head == parent.snapshot_uuid

    # Evict Redis so the next turn rebuilds from ES via that query, then run a clean turn.
    await slack_harness.redis_client.delete(f"conversation:{conversation_uuid}")
    slack_harness.llm_client.set_script(_script("recompacted answer"))
    thread_client.thread = [
        _human_msg(ts="1200.000100", user="U_ALICE", text="<@U_BOT> parent message"),
        _human_msg(ts="1200.000700", user="U_ALICE", text="<@U_BOT> continue"),
    ]
    await slack_harness.deliver(
        _mention(channel=channel, ts="1200.000700", thread_ts=root_ts, text="<@U_BOT> continue"),
        thread_client=thread_client,
    )

    conv = await _conv(slack_harness, conversation_uuid)
    # Reattached to the parent, not the abandoned pending child.
    assert conv.snapshot_uuid == parent.snapshot_uuid
    assert "recompacted answer" in conv.sorted_messages()[-1].content


# -----------------------------------------------------------------------------
# Same-thread serialization + post-race future-turn fidelity
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_same_thread_serialization_and_future_turn_fidelity(slack_harness):
    """Two back-to-back same-thread `@`-mentions are answered serially; the second turn's reconcile includes the
    first turn's committed reply (ts greater than the second mention's `triggering_ts`, exercising the
    `max(triggering_ts, latest_stored_source_id)` cap and the in-stored exemption). A third mention then
    projects the two prior turn pairs intact — `[user(A), assistant(botA), user(B), assistant(botB), user(C)]`,
    not the merged form."""
    channel, root_ts = "C_RACE", "10.000000"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)

    a_ts, b_ts = "10.000000", "11.000000"
    # Pre-pin the bot reply ts so the storage-order assertion is deterministic: botA@12, botB@13.
    thread_a = FakeSlackThreadClient(thread=[_human_msg(ts=a_ts, user="U_ALICE", text="<@U_BOT> A question")])
    thread_a.next_post_ts = ["12.000000"]
    thread_b = FakeSlackThreadClient(
        thread=[
            _human_msg(ts=a_ts, user="U_ALICE", text="<@U_BOT> A question"),
            _human_msg(ts=b_ts, user="U_BOB", text="<@U_BOB> B question"),
        ]
    )
    thread_b.next_post_ts = ["13.000000"]

    slack_harness.llm_client.set_script(_script("A bot reply"))

    async def turn_a():
        await slack_harness.deliver(_mention(channel=channel, ts=a_ts), thread_client=thread_a)

    async def turn_b():
        # Wait until A holds the lock, then deliver B so it queues behind A.
        await asyncio.sleep(0.01)
        slack_harness.llm_client.reset()
        slack_harness.llm_client.set_script(_script("B bot reply"))
        # Thread B's fetch must see botA@12 (committed by turn A before B's locked region runs).
        thread_b.thread.append(_bot_post({"ts": "12.000000", "text": "<@U_ALICE> A bot reply"}, thread_ts=root_ts))
        thread_b.thread.sort(key=lambda m: m["ts"])
        await slack_harness.deliver(
            _mention(channel=channel, ts=b_ts, thread_ts=root_ts, user="U_BOB", text="<@U_BOB> B question"),
            thread_client=thread_b,
        )

    await asyncio.gather(turn_a(), turn_b())

    conv = await _conv(slack_harness, conversation_uuid)
    sorted_msgs = conv.sorted_messages()
    # Storage is source-id-sorted: A_mention@10, B_mention@11, A_bot@12, B_bot@13.
    assert [m.source_id for m in sorted_msgs] == [a_ts, b_ts, "12.000000", "13.000000"]
    assert sorted_msgs[2].reply_to_source_id == a_ts  # botA replied to A
    assert sorted_msgs[3].reply_to_source_id == b_ts  # botB replied to B

    # B's turn projection: terminates in B's user message, with assistant(A) before it.
    b_call = slack_harness.llm_client.stream_turn_calls[-1]
    b_items = b_call["items"]
    assert b_items[-1].role == "user"
    assert "B question" in (b_items[-1].content or "")
    a_assistant_idx = next(
        i for i, it in enumerate(b_items) if it.role == "assistant" and "A bot reply" in (it.content or "")
    )
    assert a_assistant_idx < len(b_items) - 1

    # Third mention C — projects the two prior turn pairs intact.
    c_ts = "14.000000"
    thread_c = FakeSlackThreadClient(
        thread=[
            _human_msg(ts=a_ts, user="U_ALICE", text="<@U_BOT> A question"),
            _human_msg(ts=b_ts, user="U_BOB", text="<@U_BOB> B question"),
            _bot_post({"ts": "12.000000", "text": "<@U_ALICE> A bot reply"}, thread_ts=root_ts),
            _bot_post({"ts": "13.000000", "text": "<@U_BOB> B bot reply"}, thread_ts=root_ts),
            _human_msg(ts=c_ts, user="U_ALICE", text="<@U_BOT> C question"),
        ]
    )
    slack_harness.llm_client.reset()
    slack_harness.llm_client.set_script(_script("C bot reply"))
    await slack_harness.deliver(
        _mention(channel=channel, ts=c_ts, thread_ts=root_ts, text="<@U_BOT> C question"),
        thread_client=thread_c,
    )

    c_call = slack_harness.llm_client.stream_turn_calls[-1]
    roles = [it.role for it in c_call["items"] if it.type == "message"]
    contents = [it.content or "" for it in c_call["items"] if it.type == "message"]
    # The two turn pairs survive intact — not merged into one user/assistant block.
    assert roles[-5:] == ["user", "assistant", "user", "assistant", "user"]
    assert "A question" in contents[-5] and "A bot reply" in contents[-4]
    assert "B question" in contents[-3] and "B bot reply" in contents[-2]
    assert "C question" in contents[-1]


# -----------------------------------------------------------------------------
# Concurrent threads
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_threads_run_independently(slack_harness):
    """Mentions in two different threads run concurrently and each commits its own conversation."""
    ch1, ts1 = "C_MULTI_A", "20.000100"
    ch2, ts2 = "C_MULTI_B", "21.000100"
    uuid1 = slack_conversation_uuid(TEAM_ID, ch1, ts1)
    uuid2 = slack_conversation_uuid(TEAM_ID, ch2, ts2)

    thread1 = FakeSlackThreadClient(thread=[_human_msg(ts=ts1, user="U_ALICE", text="<@U_BOT> thread one")])
    thread2 = FakeSlackThreadClient(thread=[_human_msg(ts=ts2, user="U_BOB", text="<@U_BOT> thread two")])
    slack_harness.llm_client.set_script(_script("shared answer"))

    await asyncio.gather(
        slack_harness.deliver(_mention(channel=ch1, ts=ts1), thread_client=thread1),
        slack_harness.deliver(_mention(channel=ch2, ts=ts2, user="U_BOB"), thread_client=thread2),
    )

    conv1 = await _conv(slack_harness, uuid1)
    conv2 = await _conv(slack_harness, uuid2)
    assert conv1.conversation_uuid != conv2.conversation_uuid
    assert conv1.sorted_messages()[0].content == "<@U_BOT> thread one"
    assert conv2.sorted_messages()[0].content == "<@U_BOT> thread two"


# -----------------------------------------------------------------------------
# Foreign bot / app_uninstalled
# -----------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_foreign_bot_in_thread_normalized(slack_harness, thread_client):
    """A foreign bot integration posting `bot_message`-subtype into the thread (with `bot_id`, no `user`) is
    normalized to `bot:{bot_id}` and does not crash replay; `users.info` is never called for it."""
    channel, root_ts = "C_FOREIGN", "1600.000100"
    conversation_uuid = slack_conversation_uuid(TEAM_ID, channel, root_ts)
    slack_harness.llm_client.set_script(_script("answer with foreign bot present"))
    # The foreign-bot post must precede the trigger ts — `sync_slack_thread`'s inner filter drops thread
    # messages past the trigger that are not already in stored. The trigger here IS the root, so we use a
    # second mention as the trigger and place both the root and the foreign-bot post before it.
    foreign_ts = "1600.000150"
    trigger_ts = "1600.000300"
    thread_client.thread = [
        _human_msg(ts=root_ts, user="U_ALICE", text="<@U_BOT> hello"),
        {"ts": foreign_ts, "bot_id": "B_FOREIGN", "subtype": "bot_message", "text": "foreign integration post"},
    ]

    await slack_harness.deliver(
        _mention(channel=channel, ts=trigger_ts, thread_ts=root_ts, text="<@U_BOT> note the foreign bot"),
        thread_client=thread_client,
    )

    conv = await _conv(slack_harness, conversation_uuid)
    foreign = conv.message_by_source_id(foreign_ts)
    assert foreign is not None
    assert foreign.author_id == "bot:B_FOREIGN"
    # users.info was never called for the foreign-bot author ID.
    assert "B_FOREIGN" not in thread_client.users_info_calls
    assert "bot:B_FOREIGN" not in thread_client.users_info_calls


@pytest.mark.asyncio(loop_scope="session")
async def test_app_uninstalled_logged_harness_keeps_running(slack_harness, thread_client, caplog):
    """An `app_uninstalled` event over the socket is logged at error level; the harness keeps running and the
    next `@`-mention still completes a turn."""
    import logging

    with caplog.at_level(logging.ERROR, logger="prokaryotes.slack_v1"):
        await slack_harness.deliver_via_socket({"type": "app_uninstalled"}, thread_client=thread_client)
    assert any(r.levelno == logging.ERROR for r in caplog.records)

    # The harness is still alive — a normal mention still runs.
    channel, ts = "C_ALIVE", "1700.000100"
    slack_harness.llm_client.set_script(_script("still alive"))
    next_thread = FakeSlackThreadClient(thread=[_human_msg(ts=ts, user="U_ALICE", text="<@U_BOT> still up?")])
    await slack_harness.deliver(_mention(channel=channel, ts=ts), thread_client=next_thread)

    conv = await _conv(slack_harness, slack_conversation_uuid(TEAM_ID, channel, ts))
    assert "still alive" in conv.sorted_messages()[-1].content


# Keep the module-level imports referenced even when a subset of tests is selected.
_ = (uuid4, APP_ID)
