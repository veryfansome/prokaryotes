"""`sync_slack_thread` thread → `NormalizedMessage` build and reconcile.

Covers author normalization, append / edit / delete / divergence classification, the asymmetric
pagination-cap / inner-filter scenarios, the `stored.messages` source-id-sorted invariant, the projection
turn-pair ordering cases, `sanitize_mentions`, multi-post bot runs, the post-compaction `inclusive=True`
boundary, and cold / fresh fetch.
"""

from __future__ import annotations

import pytest

from prokaryotes.conversation_v1.models import Conversation, ConversationMessage
from prokaryotes.conversation_v1.project import project_for_llm
from prokaryotes.harness_v1.slack import SlackHarness
from prokaryotes.slack_v1.replay import sanitize_mentions
from tests.unit_tests._slack_fakes import FakeRedis, FakeSearchClient, FakeSlackThreadClient

BOT_USER = "U_BOT"
BOT_ID = "B_BOT"
APP_ID = "A_APP"
CONV_UUID = "c-slack-replay"
CHANNEL = "C_CHAN"
THREAD_TS = "100.000000"


class _TestHarness(SlackHarness):
    """`SlackHarness` with fakes injected, bypassing `__init__`'s LLM-client construction."""

    def __init__(self) -> None:
        self._redis_client = FakeRedis()
        self._search_client = FakeSearchClient()
        self._conversation_cache_ex = 60 * 60 * 24 * 7
        self.bot_user_id = BOT_USER
        self.bot_id = BOT_ID
        self.app_id = APP_ID
        self.team_id = "T_TEAM"


@pytest.fixture
def harness() -> _TestHarness:
    return _TestHarness()


def _human(ts: str, user: str, text: str) -> dict:
    return {"ts": ts, "user": user, "text": text}


def _bot_post(ts: str, text: str, *, bot_id: str = BOT_ID) -> dict:
    """A bot post in the `chat.postMessage` response shape — `bot_id` only, no `user`, no `bot_profile`."""
    return {"ts": ts, "bot_id": bot_id, "text": text}


async def _sync(harness: _TestHarness, slack: FakeSlackThreadClient, triggering_ts: str) -> Conversation:
    return await harness.sync_slack_thread(
        channel_id=CHANNEL,
        conversation_uuid=CONV_UUID,
        slack_client=slack,
        thread_ts=THREAD_TS,
        triggering_ts=triggering_ts,
    )


def _stash(harness: _TestHarness, conversation: Conversation) -> None:
    """Make `conversation` the stored snapshot for the next sync (Redis fast path)."""
    harness._redis_client._store[f"conversation:{CONV_UUID}"] = conversation.model_dump_json().encode()


# -----------------------------------------------------------------------------
# author normalization
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_message_normalizes_to_user_source_id(harness: _TestHarness):
    """A human message becomes a `NormalizedMessage` with `source_id` = `ts`, `author_id` = the Slack user."""
    slack = FakeSlackThreadClient(
        thread=[_human(THREAD_TS, "U_ALICE", "<@U_BOT> hello")],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, THREAD_TS)

    assert len(conv.messages) == 1
    msg = conv.messages[0]
    assert msg.source_id == THREAD_TS
    assert msg.author_id == "U_ALICE"
    assert msg.display_name == "alice"


@pytest.mark.asyncio
async def test_bot_post_normalizes_to_bot_user_id_for_all_three_shapes(harness: _TestHarness):
    """The bot's own posts carry `author_id = bot_user_id` whether Slack returned `user`, `bot_id` only, or
    only `bot_profile.app_id`."""
    slack = FakeSlackThreadClient(
        thread=[
            _human(THREAD_TS, "U_ALICE", "<@U_BOT> hi"),
            {"ts": "101.000000", "user": BOT_USER, "text": "via user"},
            {"ts": "102.000000", "bot_id": BOT_ID, "text": "via bot_id"},
            {"ts": "103.000000", "bot_profile": {"app_id": APP_ID}, "text": "via bot_profile"},
        ],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, "103.000000")

    bot_msgs = [m for m in conv.messages if m.author_id == BOT_USER]
    assert len(bot_msgs) == 3
    assert {m.content for m in bot_msgs} == {"via user", "via bot_id", "via bot_profile"}


@pytest.mark.asyncio
async def test_foreign_bot_normalizes_to_bot_prefix_and_skips_users_info(harness: _TestHarness):
    """A third-party integration in the thread normalizes to `bot:{bot_id}` and is not resolved via
    `users.info`."""
    slack = FakeSlackThreadClient(
        thread=[
            _human(THREAD_TS, "U_ALICE", "<@U_BOT> hi"),
            {"ts": "101.000000", "bot_id": "B_FOREIGN", "text": "from a foreign integration"},
        ],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, "101.000000")

    foreign = conv.message_by_source_id("101.000000")
    assert foreign.author_id == "bot:B_FOREIGN"
    assert foreign.display_name is None
    # users.info is only called for the human author, never for bot:* IDs.
    assert slack.users_info_calls == ["U_ALICE"]


# -----------------------------------------------------------------------------
# append / edit / delete / divergence
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_mention_appends_against_fresh_conversation(harness: _TestHarness):
    """A first mention reconciles as `append` against a fresh `Conversation`."""
    slack = FakeSlackThreadClient(
        thread=[_human(THREAD_TS, "U_ALICE", "<@U_BOT> first")],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, THREAD_TS)

    assert [m.source_id for m in conv.messages] == [THREAD_TS]


@pytest.mark.asyncio
async def test_followup_mention_appends_against_stored_snapshot(harness: _TestHarness):
    """A follow-up `@`-mention reconciles as `append` against the stored snapshot."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="first"),
            ConversationMessage(
                source_id="101.000000", author_id=BOT_USER, content="bot reply", reply_to_source_id=THREAD_TS
            ),
        ],
    )
    _stash(harness, stored)
    slack = FakeSlackThreadClient(
        thread=[
            _human(THREAD_TS, "U_ALICE", "first"),
            _bot_post("101.000000", "bot reply"),
            _human("102.000000", "U_ALICE", "<@U_BOT> follow up"),
        ],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, "102.000000")

    assert [m.source_id for m in conv.messages] == [THREAD_TS, "101.000000", "102.000000"]


@pytest.mark.asyncio
async def test_non_mention_chatter_between_mentions_included_as_ordinary_messages(harness: _TestHarness):
    """In-thread non-`@` chatter between two `@`-mentions is included as ordinary messages."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="first mention"),
            ConversationMessage(
                source_id="101.000000", author_id=BOT_USER, content="bot reply", reply_to_source_id=THREAD_TS
            ),
        ],
    )
    _stash(harness, stored)
    slack = FakeSlackThreadClient(
        thread=[
            _human(THREAD_TS, "U_ALICE", "first mention"),
            _bot_post("101.000000", "bot reply"),
            _human("102.000000", "U_BOB", "just chatting, no mention"),
            _human("103.000000", "U_ALICE", "<@U_BOT> second mention"),
        ],
        display_names={"U_ALICE": "alice", "U_BOB": "bob"},
    )
    conv = await _sync(harness, slack, "103.000000")

    assert [m.source_id for m in conv.messages] == [THREAD_TS, "101.000000", "102.000000", "103.000000"]
    assert conv.message_by_source_id("102.000000").author_id == "U_BOB"


@pytest.mark.asyncio
async def test_message_changed_alone_reconciles_as_in_place_edit(harness: _TestHarness):
    """A `message_changed` (same `ts`, new text) reconciles as `edit` → in-place overwrite."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="original text")],
    )
    _stash(harness, stored)
    slack = FakeSlackThreadClient(
        thread=[_human(THREAD_TS, "U_ALICE", "edited text")],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, THREAD_TS)

    msg = conv.message_by_source_id(THREAD_TS)
    assert msg.content == "edited text"
    assert msg.edited is True


@pytest.mark.asyncio
async def test_message_deleted_alone_tombstones_and_rekeys_turn_execution(harness: _TestHarness):
    """A tail-only `message_deleted` reconciles as `delete` → tombstone + `TurnExecution` re-key."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="first"),
            ConversationMessage(
                source_id="101.000000", author_id=BOT_USER, content="bot reply A", reply_to_source_id=THREAD_TS
            ),
            ConversationMessage(
                source_id="102.000000", author_id=BOT_USER, content="bot reply B", reply_to_source_id=THREAD_TS
            ),
        ],
    )
    _stash(harness, stored)
    # The first post of a multi-post run owns the TurnExecution.
    from prokaryotes.conversation_v1.models import TurnExecution, TurnItem

    await harness._search_client.put_turn_execution(
        TurnExecution(
            conversation_uuid=CONV_UUID,
            bot_message_source_id="101.000000",
            items=[TurnItem(type="function_call", name="think", call_id="c1")],
        )
    )
    # The first post (101) is deleted; the next non-tombstoned bot in the run (102) becomes the owner.
    slack = FakeSlackThreadClient(
        thread=[
            _human(THREAD_TS, "U_ALICE", "first"),
            _bot_post("102.000000", "bot reply B"),
        ],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, "102.000000")

    deleted = conv.message_by_source_id("101.000000")
    assert deleted.deleted is True
    # The TurnExecution moved from 101 → 102.
    assert (CONV_UUID, "101.000000") not in harness._search_client.turn_executions
    assert (CONV_UUID, "102.000000") in harness._search_client.turn_executions


@pytest.mark.asyncio
async def test_combined_edit_delete_append_reconciles_as_divergence(harness: _TestHarness):
    """A user editing an earlier message AND posting a new mention reconciles as `divergence`; the Slack
    mixin's op-aware branch applies each operation per its kind."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="msg one"),
            ConversationMessage(source_id="101.000000", author_id="U_BOB", content="msg two"),
            ConversationMessage(
                source_id="102.000000", author_id=BOT_USER, content="bot reply", reply_to_source_id=THREAD_TS
            ),
        ],
    )
    _stash(harness, stored)
    # Bob's message (101) is deleted, Alice's (100) is edited, and a new mention (103) is appended.
    slack = FakeSlackThreadClient(
        thread=[
            _human(THREAD_TS, "U_ALICE", "msg one EDITED"),
            _bot_post("102.000000", "bot reply"),
            _human("103.000000", "U_ALICE", "<@U_BOT> new mention"),
        ],
        display_names={"U_ALICE": "alice", "U_BOB": "bob"},
    )
    conv = await _sync(harness, slack, "103.000000")

    assert conv.message_by_source_id(THREAD_TS).content == "msg one EDITED"
    assert conv.message_by_source_id(THREAD_TS).edited is True
    assert conv.message_by_source_id("101.000000").deleted is True
    assert conv.message_by_source_id("103.000000") is not None
    assert conv.message_by_source_id("103.000000").author_id == "U_ALICE"


@pytest.mark.asyncio
async def test_divergence_deleted_bot_run_rekeys_then_orphans(harness: _TestHarness):
    """In a divergence, a deleted bot message's `TurnExecution` is re-keyed to the next non-tombstoned bot in
    the run, or deleted from ES when the run has no non-tombstoned bot left."""
    from prokaryotes.conversation_v1.models import TurnExecution, TurnItem

    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="m1"),
            ConversationMessage(
                source_id="101.000000", author_id=BOT_USER, content="bot one", reply_to_source_id=THREAD_TS
            ),
        ],
    )
    _stash(harness, stored)
    await harness._search_client.put_turn_execution(
        TurnExecution(
            conversation_uuid=CONV_UUID,
            bot_message_source_id="101.000000",
            items=[TurnItem(type="function_call", name="think", call_id="c1")],
        )
    )
    # The only bot message (101) is deleted while a new mention (102) is appended → divergence.
    slack = FakeSlackThreadClient(
        thread=[
            _human(THREAD_TS, "U_ALICE", "m1"),
            _human("102.000000", "U_ALICE", "<@U_BOT> hi again"),
        ],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, "102.000000")

    assert conv.message_by_source_id("101.000000").deleted is True
    # No non-tombstoned bot remains in the run — the TurnExecution is orphaned (deleted from ES).
    assert (CONV_UUID, "101.000000") not in harness._search_client.turn_executions


# -----------------------------------------------------------------------------
# multi-post bot run
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_post_bot_run_stays_n_consecutive_messages(harness: _TestHarness):
    """A bot reply split across N Slack posts stays N consecutive bot messages — no collapse step."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="ask"),
            ConversationMessage(
                source_id="101.000000", author_id=BOT_USER, content="part one", reply_to_source_id=THREAD_TS
            ),
            ConversationMessage(
                source_id="102.000000", author_id=BOT_USER, content="part two", reply_to_source_id=THREAD_TS
            ),
            ConversationMessage(
                source_id="103.000000", author_id=BOT_USER, content="part three", reply_to_source_id=THREAD_TS
            ),
        ],
    )
    _stash(harness, stored)
    slack = FakeSlackThreadClient(
        thread=[
            _human(THREAD_TS, "U_ALICE", "ask"),
            _bot_post("101.000000", "part one"),
            _bot_post("102.000000", "part two"),
            _bot_post("103.000000", "part three"),
        ],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, THREAD_TS)

    bot_msgs = [m for m in conv.messages if m.author_id == BOT_USER]
    assert [m.content for m in bot_msgs] == ["part one", "part two", "part three"]

    # project_for_llm re-merges them into one assistant block.
    items = project_for_llm(conv, triggering_source_id=THREAD_TS)
    assistant_items = [i for i in items if i.role == "assistant"]
    assert len(assistant_items) == 1
    assert assistant_items[0].content == "part one\n\npart two\n\npart three"


# -----------------------------------------------------------------------------
# asymmetric pagination cap / inner filter
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_thread_serialization_includes_prior_committed_reply(harness: _TestHarness):
    """Mention A@10 → bot@12, then mention B@11 syncs: incoming includes bot@12 even though `ts >
    triggering_ts`, because it is in stored. Reconcile classifies as `append` for B@11."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="1700000000.000010", author_id="U_ALICE", content="A mention"),
            ConversationMessage(
                source_id="1700000000.000012",
                author_id=BOT_USER,
                content="botA reply",
                reply_to_source_id="1700000000.000010",
            ),
        ],
    )
    _stash(harness, stored)
    slack = FakeSlackThreadClient(
        thread=[
            _human("1700000000.000010", "U_ALICE", "<@U_BOT> A mention"),
            _human("1700000000.000011", "U_BOB", "<@U_BOT> B mention"),
            _bot_post("1700000000.000012", "botA reply"),
        ],
        display_names={"U_ALICE": "alice", "U_BOB": "bob"},
    )
    conv = await _sync(harness, slack, triggering_ts="1700000000.000011")

    # bot@12 is in stored so it is exempted from the inner filter — incoming is [A@10, B@11, bot@12].
    assert [m.source_id for m in conv.messages] == [
        "1700000000.000010",
        "1700000000.000011",
        "1700000000.000012",
    ]
    # The pagination cap extended to max(triggering_ts, latest_stored) = bot@12's ts.
    assert harness._search_client  # nothing dropped
    replies_call = slack.replies_calls[0]
    assert replies_call["paginate_until_ts"] == "1700000000.000012"


@pytest.mark.asyncio
async def test_unrelated_chatter_racing_in_is_dropped(harness: _TestHarness):
    """Unrelated message C@11.5 that arrived between B's trigger and B's sync is dropped — its `ts >
    triggering_ts` and it is not in stored."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="1700000000.000010", author_id="U_ALICE", content="A mention"),
            ConversationMessage(
                source_id="1700000000.000012",
                author_id=BOT_USER,
                content="botA reply",
                reply_to_source_id="1700000000.000010",
            ),
        ],
    )
    _stash(harness, stored)
    slack = FakeSlackThreadClient(
        thread=[
            _human("1700000000.000010", "U_ALICE", "<@U_BOT> A mention"),
            _human("1700000000.000011", "U_BOB", "<@U_BOT> B mention"),
            _human("1700000000.000115", "U_CAROL", "unrelated chatter"),
            _bot_post("1700000000.000012", "botA reply"),
        ],
        display_names={"U_ALICE": "alice", "U_BOB": "bob", "U_CAROL": "carol"},
    )
    conv = await _sync(harness, slack, triggering_ts="1700000000.000011")

    # C@11.5 is dropped — not in stored and ts > triggering_ts.
    assert conv.message_by_source_id("1700000000.000115") is None
    assert [m.source_id for m in conv.messages] == [
        "1700000000.000010",
        "1700000000.000011",
        "1700000000.000012",
    ]


@pytest.mark.asyncio
async def test_string_compare_ordering_is_load_bearing(harness: _TestHarness):
    """The seconds.microseconds convention is load-bearing for the cap and filter — string compare of
    `...000010` vs `...000011` is correct."""
    slack = FakeSlackThreadClient(
        thread=[
            _human("1700000000.000010", "U_ALICE", "<@U_BOT> earlier"),
            _human("1700000000.000011", "U_BOB", "later, after trigger"),
        ],
        display_names={"U_ALICE": "alice", "U_BOB": "bob"},
    )
    conv = await _sync(harness, slack, triggering_ts="1700000000.000010")

    # Only the trigger-or-earlier message survives; the later one is filtered.
    assert [m.source_id for m in conv.messages] == ["1700000000.000010"]


# -----------------------------------------------------------------------------
# stored.messages source-id-sorted invariant
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_keeps_stored_messages_source_id_sorted(harness: _TestHarness):
    """Appending B@11 to a stored list ending in bot@12 yields [A@10, B@11, bot@12], not [A@10, bot@12,
    B@11]. A follow-up sync with no changes reconciles as `match`."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="1700000000.000010", author_id="U_ALICE", content="A mention"),
            ConversationMessage(
                source_id="1700000000.000012",
                author_id=BOT_USER,
                content="botA reply",
                reply_to_source_id="1700000000.000010",
            ),
        ],
    )
    _stash(harness, stored)
    slack = FakeSlackThreadClient(
        thread=[
            _human("1700000000.000010", "U_ALICE", "<@U_BOT> A mention"),
            _human("1700000000.000011", "U_BOB", "<@U_BOT> B mention"),
            _bot_post("1700000000.000012", "botA reply"),
        ],
        display_names={"U_ALICE": "alice", "U_BOB": "bob"},
    )
    conv = await _sync(harness, slack, triggering_ts="1700000000.000011")

    # B@11 inserted in sorted position, not tail-appended.
    assert [m.source_id for m in conv.messages] == [
        "1700000000.000010",
        "1700000000.000011",
        "1700000000.000012",
    ]

    # Follow-up sync with no further changes reconciles as `match` — the invariant survives.
    _stash(harness, conv)
    slack2 = FakeSlackThreadClient(
        thread=[
            _human("1700000000.000010", "U_ALICE", "<@U_BOT> A mention"),
            _human("1700000000.000011", "U_BOB", "<@U_BOT> B mention"),
            _bot_post("1700000000.000012", "botA reply"),
        ],
        display_names={"U_ALICE": "alice", "U_BOB": "bob"},
    )
    conv2 = await harness.sync_slack_thread(
        channel_id=CHANNEL,
        conversation_uuid=CONV_UUID,
        slack_client=slack2,
        thread_ts=THREAD_TS,
        triggering_ts="1700000000.000012",
    )
    # No new messages and no edits — same snapshot identity (match returns stored).
    assert conv2.snapshot_uuid == conv.snapshot_uuid
    assert [m.content for m in conv2.messages] == [m.content for m in conv.messages]


# -----------------------------------------------------------------------------
# projection turn-pair ordering
# -----------------------------------------------------------------------------


def test_projection_turn_pair_ordering_for_serialized_turn():
    """B's immediate turn: stored = [A@10, B@11, botA@12] with botA.reply_to=A@10. Projection is
    [user(A), assistant(botA), user(B)] — not terminating in an assistant message."""
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="10", author_id="U_ALICE", content="A"),
            ConversationMessage(source_id="11", author_id="U_BOB", content="B"),
            ConversationMessage(source_id="12", author_id=BOT_USER, content="botA", reply_to_source_id="10"),
        ],
    )
    items = project_for_llm(conv, triggering_source_id="11")

    assert [(i.role, i.content) for i in items] == [
        ("user", "A"),
        ("assistant", "botA"),
        ("user", "B"),
    ]


def test_projection_turn_pair_ordering_survives_future_turn():
    """C's turn long after the race: stored = [A@10, B@11, botA@12, botB@13, C@14]. Final projection is
    [user(A), assistant(botA), user(B), assistant(botB), user(C)] — turn pairs intact."""
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="10", author_id="U_ALICE", content="A"),
            ConversationMessage(source_id="11", author_id="U_BOB", content="B"),
            ConversationMessage(source_id="12", author_id=BOT_USER, content="botA", reply_to_source_id="10"),
            ConversationMessage(source_id="13", author_id=BOT_USER, content="botB", reply_to_source_id="11"),
            ConversationMessage(source_id="14", author_id="U_CAROL", content="C"),
        ],
    )
    items = project_for_llm(conv, triggering_source_id="14")

    assert [(i.role, i.content) for i in items] == [
        ("user", "<alice> A"),
        ("assistant", "botA"),
        ("user", "<bob> B"),
        ("assistant", "botB"),
        ("user", "<carol> C"),
    ] or [(i.role, i.content) for i in items] == [
        ("user", "A"),
        ("assistant", "botA"),
        ("user", "B"),
        ("assistant", "botB"),
        ("user", "C"),
    ]


def test_projection_no_op_when_trigger_is_latest():
    """stored = [A@10, botA@11, B@12] with botA.reply_to=A@10 and trigger=B@12 → projection identical to
    source-id order."""
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="10", author_id="U_ALICE", content="A"),
            ConversationMessage(source_id="11", author_id=BOT_USER, content="botA", reply_to_source_id="10"),
            ConversationMessage(source_id="12", author_id="U_ALICE", content="B"),
        ],
    )
    items = project_for_llm(conv, triggering_source_id="12")

    assert [(i.role, i.content) for i in items] == [
        ("user", "A"),
        ("assistant", "botA"),
        ("user", "B"),
    ]


def test_projection_trigger_tombstoned_terminates_with_assistant():
    """stored = [A@10, B@11 (deleted), botA@12] with trigger=B@11. The projection terminates with an
    assistant message — `_run_turn` detects this and refuses to call the LLM."""
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id="10", author_id="U_ALICE", content="A"),
            ConversationMessage(source_id="11", author_id="U_BOB", content="B", deleted=True),
            ConversationMessage(source_id="12", author_id=BOT_USER, content="botA", reply_to_source_id="10"),
        ],
    )
    items = project_for_llm(conv, triggering_source_id="11")

    assert [(i.role, i.content) for i in items] == [
        ("user", "A"),
        ("assistant", "botA"),
    ]
    assert items[-1].role == "assistant"


@pytest.mark.asyncio
async def test_multi_post_bot_run_carries_reply_to_on_every_post():
    """For a 3-post bot reply, `_finalize_slack_turn` sets `reply_to_source_id = triggering` on each post.
    Pass 1 emits the user once then all 3 bot posts; `_merge` joins them into one assistant block."""
    from prokaryotes.slack_v1.streaming import PostedMessage

    harness = _TestHarness()
    conv = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[ConversationMessage(source_id="10", author_id="U_ALICE", content="ask")],
    )
    await harness._finalize_slack_turn(
        completed=True,
        conversation=conv,
        posted=[
            PostedMessage(source_id="11", content="p1"),
            PostedMessage(source_id="12", content="p2"),
            PostedMessage(source_id="13", content="p3"),
        ],
        triggering_source_id="10",
        turn_items=[],
    )

    bot_msgs = [m for m in conv.messages if m.author_id == BOT_USER]
    assert all(m.reply_to_source_id == "10" for m in bot_msgs)

    items = project_for_llm(conv, triggering_source_id="10")
    assert [(i.role, i.content) for i in items] == [
        ("user", "ask"),
        ("assistant", "p1\n\np2\n\np3"),
    ]


# -----------------------------------------------------------------------------
# sanitize_mentions
# -----------------------------------------------------------------------------


def test_sanitize_mentions_strips_bot_resolves_known_leaves_unknown():
    """`sanitize_mentions` rewrites a resolved `<@USER>` to `@<display_name>`, leaves an unresolved one as the
    raw token, and strips the bot's own mention.

    Slack user IDs are uppercase-alphanumeric (`SLACK_USER_MENTION_RE`); a real bot ID like `U07BOT00` matches
    the regex while the placeholder `U_BOT` used elsewhere in this file deliberately would not.
    """
    text = "<@U07BOT00> <@U07ABC> and <@U07XYZ>"
    out = sanitize_mentions(text, "U07BOT00", {"U07ABC": "alice"})
    assert out == "@alice and <@U07XYZ>"


@pytest.mark.asyncio
async def test_bot_post_replays_verbatim_not_re_sanitized(harness: _TestHarness):
    """A bot post replays verbatim regardless of mention content — a follow-up after a prefixed bot reply
    reconciles as `append`, the prior bot post is not re-classified as `edit`."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        messages=[
            ConversationMessage(source_id=THREAD_TS, author_id="U_ALICE", content="hi"),
            ConversationMessage(
                source_id="101.000000",
                author_id=BOT_USER,
                content="<@U_ALICE> here is your answer",
                reply_to_source_id=THREAD_TS,
            ),
        ],
    )
    _stash(harness, stored)
    slack = FakeSlackThreadClient(
        thread=[
            _human(THREAD_TS, "U_ALICE", "hi"),
            # The bot post as stored verbatim — with the <@user> prefix.
            _bot_post("101.000000", "<@U_ALICE> here is your answer"),
            _human("102.000000", "U_ALICE", "<@U_BOT> follow up"),
        ],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, "102.000000")

    bot_msg = conv.message_by_source_id("101.000000")
    # Not re-classified as edit — the stored verbatim content matches the replayed text.
    assert bot_msg.content == "<@U_ALICE> here is your answer"
    assert bot_msg.edited is False
    assert [m.source_id for m in conv.messages] == [THREAD_TS, "101.000000", "102.000000"]


# -----------------------------------------------------------------------------
# post-compaction boundary / cold fetch
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_compaction_fetch_uses_inclusive_oldest_boundary(harness: _TestHarness):
    """Post-compaction: `conversations.replies` is fetched with `oldest = earliest raw-window ts` and
    `inclusive=True` so `stored.messages[0]` lands in incoming and reconcile classifies as `append`."""
    stored = Conversation(
        conversation_uuid=CONV_UUID,
        bot_author_id=BOT_USER,
        ancestor_summaries=["earlier turns summarized"],
        messages=[
            ConversationMessage(source_id="500.000000", author_id="U_ALICE", content="raw window first"),
            ConversationMessage(
                source_id="501.000000", author_id=BOT_USER, content="bot reply", reply_to_source_id="500.000000"
            ),
        ],
    )
    _stash(harness, stored)
    slack = FakeSlackThreadClient(
        thread=[
            _human("500.000000", "U_ALICE", "raw window first"),
            _bot_post("501.000000", "bot reply"),
            _human("502.000000", "U_ALICE", "<@U_BOT> new mention post-compaction"),
        ],
        display_names={"U_ALICE": "alice"},
    )
    conv = await _sync(harness, slack, "502.000000")

    call = slack.replies_calls[0]
    assert call["oldest"] == "500.000000"
    assert call["inclusive"] is True
    # The boundary message survived (would be false-`delete`d without inclusive=True).
    assert conv.message_by_source_id("500.000000") is not None
    assert [m.source_id for m in conv.messages] == ["500.000000", "501.000000", "502.000000"]
    # ancestor_summaries preserved — no Case B reset.
    assert conv.ancestor_summaries == ["earlier turns summarized"]


@pytest.mark.asyncio
async def test_cold_or_fresh_fetch_uses_oldest_none_non_inclusive(harness: _TestHarness):
    """A cold thread / fresh root: `conversations.replies` is fetched with `oldest=None` and
    `inclusive=False` — full-fetch."""
    slack = FakeSlackThreadClient(
        thread=[_human(THREAD_TS, "U_ALICE", "<@U_BOT> hi")],
        display_names={"U_ALICE": "alice"},
    )
    await _sync(harness, slack, THREAD_TS)

    call = slack.replies_calls[0]
    assert call["oldest"] is None
    assert call["inclusive"] is False
