"""Search-tier ES persistence: helpers + ConversationSearcher methods.

Tests the pure helpers (`_extract_message_content`, `conversation_from_doc`,
`messages_from_doc`, `turn_execution_from_doc`, `_default_boundary_fields`)
plus the searcher's ES verbs through a fake ES client.
"""

from __future__ import annotations

from typing import Any

import pytest

from prokaryotes.conversation_v1.models import (
    TurnExecution,
    compute_boundary_hash,
    compute_tail_hash,
)
from prokaryotes.search_v1.conversations import (
    COMPACTION_STATE_COMMITTED,
    COMPACTION_STATE_PENDING,
    CONVERSATIONS_INDEX,
    TURN_EXECUTIONS_INDEX,
    ConversationSearcher,
    _default_boundary_fields,
    _extract_message_content,
    conversation_from_doc,
    messages_from_doc,
    turn_execution_from_doc,
)
from tests.unit_tests._builders import (
    BOT_ID,
    bot_msg,
    function_call,
    function_call_output,
    msg,
)

# -----------------------------------------------------------------------------
# Pure helpers
# -----------------------------------------------------------------------------


def test_extract_message_content_joins_messages():
    messages = [msg("1", "alpha"), bot_msg("2", "beta"), msg("3", "gamma")]
    assert _extract_message_content(messages) == "alpha beta gamma"


def test_extract_message_content_skips_deleted_messages():
    messages = [msg("1", "alpha"), msg("2", "DELETED", deleted=True), msg("3", "gamma")]
    assert _extract_message_content(messages) == "alpha gamma"


def test_conversation_from_doc_returns_none_for_missing_messages_json():
    doc = {
        "snapshot_uuid": "s1",
        "conversation_uuid": "c1",
        "bot_author_id": BOT_ID,
    }
    assert conversation_from_doc("c1", doc) is None


def test_conversation_from_doc_logs_and_returns_none_for_wrong_conversation():
    doc = {
        "snapshot_uuid": "s1",
        "conversation_uuid": "OTHER",
        "bot_author_id": BOT_ID,
        "messages_json": '{"messages": []}',
    }
    assert conversation_from_doc("c1", doc) is None


def test_conversation_from_doc_returns_conversation():
    doc = {
        "snapshot_uuid": "s1",
        "conversation_uuid": "c1",
        "bot_author_id": BOT_ID,
        "parent_snapshot_uuid": "p1",
        "ancestor_summaries": ["S0"],
        "lifted_turn_items_json": '{"items": []}',
        "lifted_anchor_source_id": "1.000000",
        "messages_json": '{"messages": [{"source_id": "1", "author_id": "u", "content": "hi"}]}',
        "raw_message_start_index": 2,
    }
    conv = conversation_from_doc("c1", doc)
    assert conv is not None
    assert conv.snapshot_uuid == "s1"
    assert conv.parent_snapshot_uuid == "p1"
    assert conv.ancestor_summaries == ["S0"]
    assert conv.lifted_anchor_source_id == "1.000000"
    assert conv.raw_message_start_index == 2
    assert len(conv.messages) == 1
    assert conv.messages[0].source_id == "1"


def test_conversation_from_doc_skips_conversation_check_when_key_absent():
    """Legacy docs may lack `conversation_uuid` — treat as belonging to the
    queried conversation rather than rejecting."""
    doc = {
        "snapshot_uuid": "s1",
        # No conversation_uuid key.
        "bot_author_id": BOT_ID,
        "messages_json": '{"messages": []}',
    }
    conv = conversation_from_doc("c1", doc)
    assert conv is not None
    assert conv.conversation_uuid == "c1"


def test_messages_from_doc_returns_empty_when_messages_json_absent():
    assert messages_from_doc({}) == []


def test_turn_execution_from_doc_returns_none_when_items_json_absent():
    assert turn_execution_from_doc({}) is None


def test_turn_execution_from_doc_round_trips():
    doc = {
        "bot_message_source_id": "2",
        "conversation_uuid": "c1",
        "items_json": '{"items": [{"type": "function_call", "call_id": "c", "name": "t", "arguments": "{}"}]}',
        "completed": True,
    }
    te = turn_execution_from_doc(doc)
    assert te is not None
    assert te.bot_message_source_id == "2"
    assert len(te.items) == 1
    assert te.items[0].type == "function_call"


def test_default_boundary_fields_per_snapshot_only():
    """Per-snapshot boundary — no parent walk. Counts non-deleted messages
    in the snapshot itself plus its raw_message_start_index baseline."""
    from tests.unit_tests._builders import conversation

    conv = conversation(
        msg("3", "U2"),
        bot_msg("4", "A2"),
        raw_message_start_index=2,
        snapshot_uuid="s1",
    )

    fields = _default_boundary_fields(conv)

    assert fields["raw_message_start_index"] == 2
    assert fields["boundary_message_count"] == 2 + 2  # parent baseline + own
    assert fields["boundary_user_count"] == 1  # one user msg
    assert fields["boundary_hash"] == compute_boundary_hash(conv.messages)
    assert fields["tail_hash"] == compute_tail_hash(conv.messages, BOT_ID)


# -----------------------------------------------------------------------------
# ConversationSearcher — fake ES coverage
# -----------------------------------------------------------------------------


class _FakeES:
    """Minimal AsyncElasticsearch fake: only the verbs ConversationSearcher uses."""

    def __init__(self) -> None:
        # index → {doc_id: source}
        self._docs: dict[str, dict[str, dict[str, Any]]] = {
            CONVERSATIONS_INDEX: {},
            TURN_EXECUTIONS_INDEX: {},
        }
        self.update_calls: list[tuple[str, str, dict]] = []

    async def get(self, *, index: str, id: str) -> dict[str, Any]:
        if id not in self._docs[index]:
            raise KeyError(id)
        return {"_source": self._docs[index][id]}

    async def index(self, *, index: str, id: str, document: dict[str, Any]) -> None:
        self._docs[index][id] = dict(document)

    async def delete(self, *, index: str, id: str) -> None:
        self._docs[index].pop(id, None)

    async def update(self, *, index: str, id: str, doc: dict[str, Any]) -> None:
        self.update_calls.append((index, id, doc))
        if id in self._docs[index]:
            self._docs[index][id].update(doc)
        else:
            self._docs[index][id] = dict(doc)

    async def search(self, *, index: str, query: dict, size: int = 10, sort=None) -> dict:
        hits = []
        for did, src in self._docs[index].items():
            if self._matches(src, query):
                hits.append({"_id": did, "_source": src})
        if sort:
            key = list(sort[0].keys())[0]
            direction = sort[0][key]
            reverse = direction == "desc"
            hits.sort(key=lambda h: h["_source"].get(key, ""), reverse=reverse)
        return {"hits": {"hits": hits[:size]}}

    def _matches(self, src: dict, query: dict) -> bool:
        # Very small subset of ES query DSL: bool/must with term + nested bool clauses.
        bool_q = query.get("bool", {})
        for clause in bool_q.get("must", []):
            if not self._evaluate_clause(src, clause):
                return False
        return True

    def _evaluate_clause(self, src: dict, clause: dict) -> bool:
        if "term" in clause:
            field, val = next(iter(clause["term"].items()))
            return src.get(field) == val
        if "terms" in clause:
            field, vals = next(iter(clause["terms"].items()))
            return src.get(field) in vals
        if "exists" in clause:
            return clause["exists"]["field"] in src
        if "multi_match" in clause:
            mm = clause["multi_match"]
            q = mm["query"]
            for field in mm["fields"]:
                if q in (src.get(field) or ""):
                    return True
            return False
        if "bool" in clause:
            inner = clause["bool"]
            if "should" in inner:
                min_match = inner.get("minimum_should_match", 0)
                hits = sum(1 for c in inner["should"] if self._evaluate_clause(src, c))
                return hits >= min_match
            if "must_not" in inner:
                return not any(self._evaluate_clause(src, c) for c in inner["must_not"])
            if "must" in inner:
                return all(self._evaluate_clause(src, c) for c in inner["must"])
        return True


class _FakeSearcher(ConversationSearcher):
    def __init__(self):
        self._es = _FakeES()

    @property
    def es(self) -> _FakeES:  # type: ignore[override]
        return self._es


@pytest.mark.asyncio
async def test_put_conversation_indexes_document_with_boundary_fields():
    searcher = _FakeSearcher()
    from tests.unit_tests._builders import conversation

    conv = conversation(msg("1", "U1"), bot_msg("2", "A1"), snapshot_uuid="s1")

    await searcher.put_conversation(conv)

    doc = searcher.es._docs[CONVERSATIONS_INDEX]["s1"]
    assert doc["snapshot_uuid"] == "s1"
    assert doc["boundary_hash"] == compute_boundary_hash(conv.messages)
    assert doc["tail_hash"] == compute_tail_hash(conv.messages, BOT_ID)
    assert doc["lifted_turn_items_json"]  # non-empty JSON
    assert doc["messages_json"]
    assert doc["compaction_state"] == COMPACTION_STATE_COMMITTED


@pytest.mark.asyncio
async def test_put_conversation_accepts_pending_compaction_metadata():
    searcher = _FakeSearcher()
    from tests.unit_tests._builders import conversation

    conv = conversation(msg("1", "U1"), snapshot_uuid="s2")

    await searcher.put_conversation(
        conv, compaction_attempt_uuid="attempt-1", compaction_state=COMPACTION_STATE_PENDING
    )

    doc = searcher.es._docs[CONVERSATIONS_INDEX]["s2"]
    assert doc["compaction_state"] == COMPACTION_STATE_PENDING
    assert doc["compaction_attempt_uuid"] == "attempt-1"


@pytest.mark.asyncio
async def test_search_conversations_returns_sources():
    searcher = _FakeSearcher()
    from tests.unit_tests._builders import conversation

    conv = conversation(msg("1", "hello world"), snapshot_uuid="s1")
    await searcher.put_conversation(conv)

    results = await searcher.search_conversations("c-1", "hello")

    assert len(results) == 1
    assert results[0]["snapshot_uuid"] == "s1"


@pytest.mark.asyncio
async def test_update_conversation_sets_dt_modified():
    searcher = _FakeSearcher()
    from tests.unit_tests._builders import conversation

    await searcher.put_conversation(conversation(msg("1", "U1"), snapshot_uuid="s1"))

    await searcher.update_conversation("s1", is_compacted=True, summary="S")

    call = next(c for c in searcher.es.update_calls if c[1] == "s1")
    _, _, doc = call
    assert doc["is_compacted"] is True
    assert doc["summary"] == "S"
    assert "dt_modified" in doc


@pytest.mark.asyncio
async def test_find_conversation_by_tail_hash_filters_to_compacted_docs():
    searcher = _FakeSearcher()
    from tests.unit_tests._builders import conversation

    conv = conversation(msg("1", "U1"), bot_msg("2", "A1"), snapshot_uuid="s1")
    await searcher.put_conversation(conv)
    # Mark it compacted manually so the filter clauses match.
    searcher.es._docs[CONVERSATIONS_INDEX]["s1"]["is_compacted"] = True

    th = compute_tail_hash(conv.messages, BOT_ID)
    result = await searcher.find_conversation_by_tail_hash("c-1", th)

    assert result is not None
    assert result["snapshot_uuid"] == "s1"


@pytest.mark.asyncio
async def test_find_all_conversation_docs_returns_committed_dag():
    """Returns committed-or-legacy conversation docs (pending excluded)."""
    searcher = _FakeSearcher()
    from tests.unit_tests._builders import conversation

    await searcher.put_conversation(conversation(msg("1", "U1"), snapshot_uuid="s1"))
    await searcher.put_conversation(conversation(msg("2", "U2"), snapshot_uuid="s2"))
    # Pending should be excluded.
    await searcher.put_conversation(
        conversation(msg("3", "U3"), snapshot_uuid="s3"),
        compaction_state=COMPACTION_STATE_PENDING,
    )

    docs = await searcher.find_all_conversation_docs("c-1")

    snapshot_uuids = {d["snapshot_uuid"] for d in docs}
    assert snapshot_uuids == {"s1", "s2"}


@pytest.mark.asyncio
async def test_rekey_turn_execution_moves_doc():
    searcher = _FakeSearcher()
    te = TurnExecution(
        conversation_uuid="c-1",
        bot_message_source_id="old-id",
        items=[function_call("c1", "tool"), function_call_output("c1", "out")],
        completed=True,
    )
    await searcher.put_turn_execution(te)

    await searcher.rekey_turn_execution("old-id", "new-id")

    assert "old-id" not in searcher.es._docs[TURN_EXECUTIONS_INDEX]
    assert "new-id" in searcher.es._docs[TURN_EXECUTIONS_INDEX]
    moved = searcher.es._docs[TURN_EXECUTIONS_INDEX]["new-id"]
    assert moved["bot_message_source_id"] == "new-id"
    # Items survive.
    revived = turn_execution_from_doc(moved)
    assert revived is not None
    assert len(revived.items) == 2


@pytest.mark.asyncio
async def test_rekey_turn_execution_silent_no_op_on_missing_old_id():
    searcher = _FakeSearcher()

    # Should not raise.
    await searcher.rekey_turn_execution("missing-old-id", "new-id")

    assert "new-id" not in searcher.es._docs[TURN_EXECUTIONS_INDEX]
