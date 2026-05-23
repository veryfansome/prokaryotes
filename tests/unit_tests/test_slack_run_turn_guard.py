"""The tombstoned-trigger guard in `SlackHarness._run_turn`.

If the triggering mention is deleted (or absent) by the time the turn projects, the two-pass walk has no live
trigger to place last — the projection ends on a prior assistant message. `_run_turn` must drop the turn before
posting a placeholder or calling the LLM rather than answering stale thread context.
"""

from __future__ import annotations

import logging

import pytest

from prokaryotes.conversation_v1.models import Conversation, ConversationMessage
from prokaryotes.harness_v1.slack import SlackHarness
from tests.unit_tests._slack_fakes import FakeRedis, FakeSearchClient

BOT_USER = "U_BOT"
CONV_UUID = "c-run-turn-guard"
CHANNEL = "C_CHAN"
THREAD_TS = "100.000000"


class _UnreachableLLM:
    """`stream_turn` raises — the guard must return before any LLM call is made."""

    def stream_turn(self, **kwargs):
        raise AssertionError("stream_turn must not be called when the trigger is tombstoned")


class _RecordingSlack:
    """Records `chat_post_message` — the guard must return before a placeholder is posted."""

    def __init__(self) -> None:
        self.posts = 0

    async def chat_post_message(self, **kwargs) -> dict:
        self.posts += 1
        return {"ts": "999.000000"}


class _GuardHarness(SlackHarness):
    """`SlackHarness` with fakes injected, bypassing `__init__`'s LLM-client construction."""

    def __init__(self) -> None:
        self._redis_client = FakeRedis()
        self._search_client = FakeSearchClient()
        self._conversation_cache_ex = 60 * 60 * 24 * 7
        self.bot_user_id = BOT_USER
        self.team_id = "T_TEAM"
        self.llm_client = _UnreachableLLM()
        self.default_model = "fake-model"


async def _run(harness: _GuardHarness, conversation: Conversation, trigger_ts: str) -> _RecordingSlack:
    slack = _RecordingSlack()
    await harness._run_turn(
        channel_id=CHANNEL,
        conversation=conversation,
        event={"ts": trigger_ts, "user": "U_BOB", "channel_type": "channel"},
        prelude=None,
        slack_client=slack,
        thread_ts=THREAD_TS,
    )
    return slack


@pytest.mark.asyncio
async def test_run_turn_drops_turn_when_trigger_is_tombstoned(caplog):
    """A trigger reconciled as `deleted=True` → `_run_turn` logs an error and returns; no LLM call, no
    placeholder post."""
    harness = _GuardHarness()
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="10", author_id="U_ALICE", content="earlier"),
            ConversationMessage(source_id="11", author_id=BOT_USER, content="botA", reply_to_source_id="10"),
            ConversationMessage(source_id="12", author_id="U_BOB", content="@bot hi", deleted=True),
        ],
    )
    with caplog.at_level(logging.ERROR):
        slack = await _run(harness, conv, trigger_ts="12")

    assert slack.posts == 0
    assert "missing or tombstoned" in caplog.text


@pytest.mark.asyncio
async def test_run_turn_drops_turn_when_trigger_absent(caplog):
    """A trigger `source_id` not present in the snapshot at all → same drop (no LLM call, no placeholder)."""
    harness = _GuardHarness()
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id="10", author_id="U_ALICE", content="earlier")],
    )
    with caplog.at_level(logging.ERROR):
        slack = await _run(harness, conv, trigger_ts="999")

    assert slack.posts == 0
    assert "missing or tombstoned" in caplog.text
