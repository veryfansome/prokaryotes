"""Two-pass turn-pair projection — the load-bearing `project_for_llm` change for the Slack harness.

Covers the scenarios from the design doc's "Storage order != LLM input order" section: a bot reply must stay
adjacent to the user message it answered even when storage order interleaves them, and that association must
survive into every later reader of the snapshot.
"""

from __future__ import annotations

from prokaryotes.conversation_v1.models import Conversation, ConversationMessage, TurnExecution, TurnItem
from prokaryotes.conversation_v1.project import project_for_llm

BOT = "__bot__"


def _conv(*messages: ConversationMessage) -> Conversation:
    return Conversation(conversation_uuid="c", bot_author_id=BOT, messages=list(messages))


def _user(source_id: str, content: str, author_id: str = "user", **kw) -> ConversationMessage:
    return ConversationMessage(source_id=source_id, author_id=author_id, content=content, **kw)


def _bot(source_id: str, content: str, reply_to: str | None) -> ConversationMessage:
    return ConversationMessage(source_id=source_id, author_id=BOT, content=content, reply_to_source_id=reply_to)


def _roles_contents(items) -> list[tuple[str, str]]:
    return [(i.role, i.content) for i in items if i.type == "message"]


def test_serialized_turn_pairs_trigger_lands_last():
    """`[A, B, botA]` projecting B's turn → `user(A) -> assistant(botA) -> user(B)`.

    The unreshuffled source-id walk would terminate in an assistant message, which both providers misbehave on.
    """
    conv = _conv(_user("10", "A"), _user("11", "B"), _bot("12", "botA", reply_to="10"))
    projected = project_for_llm(conv, triggering_source_id="11")
    assert _roles_contents(projected) == [("user", "A"), ("assistant", "botA"), ("user", "B")]


def test_turn_pairs_survive_into_a_future_turn():
    """`[A, B, botA, botB, C]` → the two original turn pairs stay intact for C's turn.

    Without the durable `reply_to_source_id`, source-id-order projection collapses this to
    `user("A\\n\\nB") -> assistant("botA\\n\\nbotB") -> user(C)` and the turn structure is lost forever.
    """
    conv = _conv(
        _user("10", "A"),
        _user("11", "B"),
        _bot("12", "botA", reply_to="10"),
        _bot("13", "botB", reply_to="11"),
        _user("14", "C"),
    )
    projected = project_for_llm(conv, triggering_source_id="14")
    assert _roles_contents(projected) == [
        ("user", "A"),
        ("assistant", "botA"),
        ("user", "B"),
        ("assistant", "botB"),
        ("user", "C"),
    ]


def test_noop_when_trigger_is_already_latest():
    """Steady state `[A, botA, B]` projects identically to source-id order."""
    conv = _conv(_user("10", "A"), _bot("11", "botA", reply_to="10"), _user("12", "B"))
    projected = project_for_llm(conv, triggering_source_id="12")
    assert _roles_contents(projected) == [("user", "A"), ("assistant", "botA"), ("user", "B")]


def test_tombstoned_trigger_has_no_pass_two_entry():
    """A deleted trigger `B` is emitted by neither pass; the projection ends on the assistant message so
    `_run_turn` can detect the missing trigger before calling the LLM."""
    conv = _conv(
        _user("10", "A"),
        _user("11", "B", deleted=True),
        _bot("12", "botA", reply_to="10"),
    )
    projected = project_for_llm(conv, triggering_source_id="11")
    assert _roles_contents(projected) == [("user", "A"), ("assistant", "botA")]


def test_multi_post_bot_run_emits_user_once():
    """A reply split across N posts carries the same `reply_to_source_id` on each; the user is pulled forward
    once and the N bot posts merge into one assistant block."""
    conv = _conv(
        _user("10", "A"),
        _bot("11", "part one", reply_to="10"),
        _bot("12", "part two", reply_to="10"),
        _bot("13", "part three", reply_to="10"),
    )
    projected = project_for_llm(conv, triggering_source_id="10")
    assert _roles_contents(projected) == [("user", "A"), ("assistant", "part one\n\npart two\n\npart three")]


def test_tool_items_interleave_before_the_bot_message():
    """Pass 1 emits a bot's `TurnExecution` items between the pulled-forward user and the bot message."""
    conv = _conv(_user("10", "A"), _bot("12", "botA", reply_to="10"))
    turn = TurnExecution(
        conversation_uuid="c",
        bot_message_source_id="12",
        items=[
            TurnItem(type="function_call", call_id="x", name="think", arguments="{}"),
            TurnItem(type="function_call_output", call_id="x", output="thought"),
        ],
    )
    projected = project_for_llm(conv, historical_turns={"12": turn}, triggering_source_id="10")
    kinds = [(i.type, i.role) for i in projected]
    assert kinds == [
        ("message", "user"),
        ("function_call", None),
        ("function_call_output", None),
        ("message", "assistant"),
    ]


def test_foreign_bot_message_projects_as_user_role():
    """A foreign bot (`author_id` != `bot_author_id`) is skipped by Pass 1 and emitted as a user-role item by
    Pass 2 — so it lands after the turn pairs, like any other unclaimed message."""
    foreign = ConversationMessage(source_id="11", author_id="bot:B123", content="foreign post")
    conv = _conv(_user("10", "A"), foreign, _bot("12", "botA", reply_to="10"))
    projected = project_for_llm(conv, triggering_source_id="10")
    assert _roles_contents(projected) == [("user", "A"), ("assistant", "botA"), ("user", "foreign post")]


def test_strips_addressee_mention_from_bot_reply():
    """A Slack-stored bot reply carries the addressee's `<@USERID> ` mention as the leading prefix; projection
    strips it so the LLM sees the bare body. Without this strip the LLM mimics the prefix into its own outputs
    and the streamer prepends a second one on the wire."""
    conv = _conv(
        _user("10", "question", author_id="U_ALICE"),
        _bot("11", "<@U_ALICE> answer", reply_to="10"),
    )
    projected = project_for_llm(conv, triggering_source_id="10")
    assert _roles_contents(projected) == [("user", "question"), ("assistant", "answer")]


def test_strips_addressee_mention_per_message_in_multi_user_thread():
    """Two users mention the bot in series; each stored bot reply carries the addressee it was answering. The
    strip uses each bot message's own `reply_to_source_id` so the two different prefixes come off independently."""
    conv = _conv(
        _user("10", "q1", author_id="U_ALICE"),
        _user("11", "q2", author_id="U_BOB"),
        _bot("12", "<@U_ALICE> a1", reply_to="10"),
        _bot("13", "<@U_BOB> a2", reply_to="11"),
    )
    projected = project_for_llm(conv, triggering_source_id="11")
    assistant_contents = [item.content for item in projected if item.role == "assistant"]
    assert assistant_contents == ["a1", "a2"]


def test_continuation_posts_pass_through_unstripped():
    """Continuation posts of a multi-post bot reply share `reply_to_source_id` with the first post but do not
    carry the `<@USER>` prefix (only the first post does). The `startswith` guard makes the strip a no-op for
    continuations, and their content reaches the LLM intact."""
    conv = _conv(
        _user("10", "go", author_id="U_ALICE"),
        _bot("11", "<@U_ALICE> part one", reply_to="10"),
        _bot("12", "part two", reply_to="10"),
        _bot("13", "part three", reply_to="10"),
    )
    projected = project_for_llm(conv, triggering_source_id="10")
    assert _roles_contents(projected) == [("user", "go"), ("assistant", "part one\n\npart two\n\npart three")]


def test_strip_no_ops_when_reply_to_source_id_is_unset():
    """Legacy bot messages (and tests built with the shared `bot_msg` builder) don't carry
    `reply_to_source_id`; with no addressee to resolve, content passes through verbatim."""
    legacy = ConversationMessage(source_id="11", author_id=BOT, content="<@U_ALICE> hello")
    conv = _conv(_user("10", "hi", author_id="U_ALICE"), legacy)
    projected = project_for_llm(conv, triggering_source_id="10")
    assert _roles_contents(projected) == [("user", "hi"), ("assistant", "<@U_ALICE> hello")]


def test_strip_no_ops_for_dm_or_non_slack_bot_reply():
    """In Slack DMs the streamer passes `reply_to_user_id=None` (channel_type=='im'), so the `<@USER> ` prefix is
    never added; non-Slack harnesses use opaque session/user IDs as `author_id` and likewise never produce that
    literal as a prefix. With `reply_to_source_id` set to a non-bot author but the content lacking the exact
    leading mention, `startswith` makes the strip a no-op and the body reaches the LLM verbatim — the contract the
    projection layer's surface-agnostic strip relies on for correctness outside Slack channels."""
    conv = _conv(
        _user("10", "hi", author_id="U_ALICE"),
        _bot("11", "plain reply without a leading mention", reply_to="10"),
    )
    projected = project_for_llm(conv, triggering_source_id="10")
    assert _roles_contents(projected) == [("user", "hi"), ("assistant", "plain reply without a leading mention")]
