"""Unit tests for `distinct_human_author_ids`.

Covers the author-shape edge cases the helper is documented to exclude: the bot's own ID, foreign-bot IDs,
the `"unknown"` sentinel, and tombstoned messages. Stickiness across deletion/compaction is a property of the
*call site* (Site A in `_locked_turn`) — see `test_slack_continuation.py`.
"""

from __future__ import annotations

from prokaryotes.conversation_v1.models import Conversation, ConversationMessage
from prokaryotes.slack_v1.replay import distinct_human_author_ids

BOT_USER = "U_BOT"


def _make(messages: list[ConversationMessage]) -> Conversation:
    return Conversation(conversation_uuid="c-human-set", bot_author_id=BOT_USER, messages=messages)


def test_returns_distinct_human_authors():
    conv = _make(
        [
            ConversationMessage(source_id="1", author_id="U_ALICE", content="hi"),
            ConversationMessage(source_id="2", author_id="U_BOB", content="hi back"),
            ConversationMessage(source_id="3", author_id="U_ALICE", content="more"),
        ]
    )
    assert distinct_human_author_ids(conv) == {"U_ALICE", "U_BOB"}


def test_excludes_bot_user():
    conv = _make(
        [
            ConversationMessage(source_id="1", author_id="U_ALICE", content="hi"),
            ConversationMessage(source_id="2", author_id=BOT_USER, content="bot reply"),
        ]
    )
    assert distinct_human_author_ids(conv) == {"U_ALICE"}


def test_excludes_foreign_bot_prefix():
    conv = _make(
        [
            ConversationMessage(source_id="1", author_id="U_ALICE", content="hi"),
            ConversationMessage(source_id="2", author_id="bot:B_FOREIGN", content="from another bot"),
        ]
    )
    assert distinct_human_author_ids(conv) == {"U_ALICE"}


def test_excludes_unknown_sentinel():
    conv = _make(
        [
            ConversationMessage(source_id="1", author_id="U_ALICE", content="hi"),
            ConversationMessage(source_id="2", author_id="unknown", content="anonymous"),
        ]
    )
    assert distinct_human_author_ids(conv) == {"U_ALICE"}


def test_excludes_tombstoned_messages():
    conv = _make(
        [
            ConversationMessage(source_id="1", author_id="U_ALICE", content="hi"),
            ConversationMessage(source_id="2", author_id="U_BOB", content="gone", deleted=True),
        ]
    )
    assert distinct_human_author_ids(conv) == {"U_ALICE"}


def test_empty_conversation_returns_empty_set():
    conv = _make([])
    assert distinct_human_author_ids(conv) == set()


def test_only_bot_messages_returns_empty_set():
    conv = _make(
        [
            ConversationMessage(source_id="1", author_id=BOT_USER, content="bot a"),
            ConversationMessage(source_id="2", author_id=BOT_USER, content="bot b"),
        ]
    )
    assert distinct_human_author_ids(conv) == set()
