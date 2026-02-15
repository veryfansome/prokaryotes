import json
from unittest.mock import AsyncMock

import pytest
from elasticsearch import AsyncElasticsearch

from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
)
from prokaryotes.search_v1.context_partitions import (
    ContextPartitionSearcher,
    _extract_message_content,
    items_from_doc,
    partition_from_doc,
)
from tests.context_partition_utils import make_doc


class FakeContextPartitionSearcher(ContextPartitionSearcher):
    def __init__(self, es: AsyncElasticsearch):
        self._es = es

    @property
    def es(self) -> AsyncElasticsearch:
        return self._es


def make_partition():
    return ContextPartition(
        conversation_uuid="conv-1",
        raw_message_start_index=2,
        ancestor_summaries=["Earlier summary"],
        items=[
            ContextPartitionItem(role="user", content="Hello"),
            ContextPartitionItem(role="assistant", content="Hi there"),
        ],
    )


def make_searcher(es=None):
    return FakeContextPartitionSearcher(es or AsyncMock(spec=AsyncElasticsearch))


def test_extract_message_content_joins_user_and_assistant_messages():
    items = [
        ContextPartitionItem(role="user", content="What is Python?"),
        ContextPartitionItem(role="assistant", content="A programming language."),
        ContextPartitionItem(type="function_call", name="lookup", arguments="{}"),
    ]

    result = _extract_message_content(items)

    assert "What is Python?" in result
    assert "A programming language." in result
    assert "lookup" not in result


@pytest.mark.asyncio
async def test_find_partition_by_tail_hash_filters_to_compacted_conversation_docs():
    source = {"partition_uuid": "p1", "tail_hash": "abc123", "is_compacted": True}
    es = AsyncMock(spec=AsyncElasticsearch)
    es.search = AsyncMock(return_value={"hits": {"hits": [{"_source": source}]}})
    searcher = make_searcher(es)

    result = await searcher.find_partition_by_tail_hash("conv-1", "abc123")

    assert result == source
    must = es.search.call_args.kwargs["query"]["bool"]["must"]
    assert {"term": {"conversation_uuid": "conv-1"}} in must
    assert {"term": {"tail_hash": "abc123"}} in must
    assert {"term": {"is_compacted": True}} in must


def test_items_from_doc_returns_empty_list_when_items_json_absent():
    assert items_from_doc({}) == []
    assert items_from_doc({"partition_uuid": "p1"}) == []


def test_partition_from_doc_returns_none_for_missing_items_json():
    doc = {"partition_uuid": "p1", "conversation_uuid": "abc"}
    assert partition_from_doc("abc", doc) is None


def test_partition_from_doc_returns_none_for_wrong_conversation():
    source = ContextPartition(conversation_uuid="abc", partition_uuid="p1", items=[])
    doc = make_doc(source)
    doc["conversation_uuid"] = "other"

    assert partition_from_doc("abc", doc) is None


def test_partition_from_doc_returns_partition():
    source = ContextPartition(
        conversation_uuid="abc",
        partition_uuid="p1",
        ancestor_summaries=["Summary"],
        raw_message_start_index=2,
        items=[ContextPartitionItem(role="user", content="Hi")],
    )
    doc = make_doc(source)
    doc["conversation_uuid"] = "abc"

    result = partition_from_doc("abc", doc)

    assert result is not None
    assert result.partition_uuid == "p1"
    assert result.ancestor_summaries == ["Summary"]
    assert result.raw_message_start_index == 2
    assert result.items == [ContextPartitionItem(role="user", content="Hi")]


def test_partition_from_doc_skips_conversation_check_when_key_absent():
    # When conversation_uuid is absent from the doc, the guard evaluates falsy
    # and the mismatch check is skipped entirely — the function proceeds to parse.
    source = ContextPartition(
        conversation_uuid="abc",
        partition_uuid="p1",
        items=[ContextPartitionItem(role="user", content="Hi")],
    )
    doc = make_doc(source)
    del doc["conversation_uuid"]

    result = partition_from_doc("different-conversation", doc)

    assert result is not None
    assert result.partition_uuid == "p1"


@pytest.mark.asyncio
async def test_put_partition_indexes_document_with_boundary_fields():
    es = AsyncMock(spec=AsyncElasticsearch)
    es.index = AsyncMock()
    searcher = make_searcher(es)
    partition = make_partition()

    await searcher.put_partition(partition)

    es.index.assert_called_once()
    call_kwargs = es.index.call_args.kwargs
    doc = call_kwargs["document"]
    assert call_kwargs["index"] == "context-partitions"
    assert call_kwargs["id"] == partition.partition_uuid
    assert doc["conversation_uuid"] == "conv-1"
    assert doc["ancestor_summaries"] == ["Earlier summary"]
    assert doc["raw_message_start_index"] == 2
    assert doc["boundary_message_count"] == 4
    assert doc["boundary_hash"]
    assert doc["tail_hash"]
    assert json.loads(doc["items_json"])


@pytest.mark.asyncio
async def test_search_partitions_returns_sources():
    es = AsyncMock(spec=AsyncElasticsearch)
    es.search = AsyncMock(return_value={"hits": {"hits": [
        {"_source": {"partition_uuid": "p1"}},
        {"_source": {"partition_uuid": "p2"}},
    ]}})
    searcher = make_searcher(es)

    results = await searcher.search_partitions("conv-1", "Python")

    assert [result["partition_uuid"] for result in results] == ["p1", "p2"]


@pytest.mark.asyncio
async def test_update_partition_sets_dt_modified():
    es = AsyncMock(spec=AsyncElasticsearch)
    es.update = AsyncMock()
    searcher = make_searcher(es)

    await searcher.update_partition("partition-1", is_compacted=True, summary="Summary")

    doc = es.update.call_args.kwargs["doc"]
    assert doc["is_compacted"] is True
    assert doc["summary"] == "Summary"
    assert "dt_modified" in doc
