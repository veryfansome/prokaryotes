import json

import pytest

from prokaryotes.api_v1.models import (
    ChatConversation,
    ContextPartition,
    ContextPartitionItem,
    compute_boundary_hash,
    compute_tail_hash,
)
from prokaryotes.web_v1 import WebBase
from tests.context_partition_utils import (
    FakeRedis,
    FakeSearchClient,
    make_chat_messages,
    make_doc,
    make_message_items,
    make_web_base,
)


class TrackingWebBase(WebBase):
    def background_and_forget(self, coro):
        self.background_tasks.add(coro)


def make_compacted_doc(
        boundary_items: list[ContextPartitionItem],
        partition: ContextPartition,
        summary: str,
        **overrides,
) -> dict:
    return make_doc(
        partition,
        is_compacted=True,
        summary=summary,
        boundary_hash=compute_boundary_hash(boundary_items),
        boundary_message_count=len(boundary_items),
        boundary_user_count=sum(1 for item in boundary_items if item.role == "user"),
        tail_hash=compute_tail_hash(boundary_items),
        **overrides,
    )


def make_tracking_web_base(redis_data: dict | None = None, search_client=None) -> TrackingWebBase:
    wb = object.__new__(TrackingWebBase)
    wb.background_tasks = set()
    wb.conversation_cache_ex = 3600
    wb.redis_client = FakeRedis(redis_data)
    wb.search_client = search_client or FakeSearchClient()
    return wb


@pytest.mark.asyncio
async def test_rebuild_from_chain_assembles_two_generation_summaries_in_order():
    p0_items = make_message_items(("user", "U1"), ("assistant", "A1"))
    p1_boundary_items = make_message_items(
        ("user", "U1"),
        ("assistant", "A1"),
        ("user", "U2"),
        ("assistant", "A2"),
    )
    p0 = ContextPartition(conversation_uuid="conv", partition_uuid="p0", items=p0_items)
    p1 = ContextPartition(
        conversation_uuid="conv",
        partition_uuid="p1",
        parent_partition_uuid="p0",
        ancestor_summaries=["S0"],
        raw_message_start_index=2,
        items=make_message_items(("user", "U2"), ("assistant", "A2")),
    )
    search = FakeSearchClient([
        make_compacted_doc(p0_items, p0, summary="S0"),
        make_compacted_doc(p1_boundary_items, p1, summary="S1"),
    ])
    wb = make_web_base(search_client=search)
    conversation = ChatConversation(
        conversation_uuid="conv",
        partition_uuid="p1",
        messages=make_chat_messages(
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U2"),
            ("assistant", "A2"),
            ("user", "U3"),
        ),
    )

    partition = await wb._rebuild_from_chain(conversation)

    assert partition.parent_partition_uuid == "p1"
    assert partition.ancestor_summaries == ["S0", "S1"]
    assert partition.raw_message_start_index == 4
    assert partition.items == [ContextPartitionItem(role="user", content="U3")]


@pytest.mark.asyncio
async def test_rebuild_from_chain_does_not_inject_stale_summary_without_valid_boundary():
    p0_items = make_message_items(("user", "U1"), ("assistant", "A1"))
    p0 = ContextPartition(conversation_uuid="conv", partition_uuid="p0", items=p0_items)
    wb = make_web_base(search_client=FakeSearchClient([
        make_compacted_doc(p0_items, p0, summary="Stale summary"),
    ]))
    conversation = ChatConversation(
        conversation_uuid="conv",
        partition_uuid="p0",
        messages=make_chat_messages(("user", "Edited U1"), ("assistant", "A1")),
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.parent_partition_uuid is None
    assert partition.ancestor_summaries == []
    assert partition.items == make_message_items(("user", "Edited U1"), ("assistant", "A1"))


@pytest.mark.asyncio
async def test_rebuild_from_chain_includes_only_summaries_up_to_matched_ancestor():
    p0_items = make_message_items(("user", "U1"), ("assistant", "A1"))
    p1_boundary_items = make_message_items(
        ("user", "U1"),
        ("assistant", "A1"),
        ("user", "U2"),
        ("assistant", "A2"),
    )
    p0 = ContextPartition(conversation_uuid="conv", partition_uuid="p0", items=p0_items)
    p1 = ContextPartition(
        conversation_uuid="conv",
        partition_uuid="p1",
        parent_partition_uuid="p0",
        ancestor_summaries=["S0"],
        raw_message_start_index=2,
        items=make_message_items(("user", "U2"), ("assistant", "A2")),
    )
    search = FakeSearchClient([
        make_compacted_doc(p0_items, p0, summary="S0"),
        make_compacted_doc(p1_boundary_items, p1, summary="S1"),
    ])
    wb = make_web_base(search_client=search)
    conversation = ChatConversation(
        conversation_uuid="conv",
        partition_uuid="p1",
        messages=make_chat_messages(
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U3-new"),
        ),
    )

    partition = await wb._rebuild_from_chain(conversation)

    assert partition.parent_partition_uuid == "p0"
    assert partition.ancestor_summaries == ["S0"]
    assert partition.raw_message_start_index == 2
    assert partition.items == [ContextPartitionItem(role="user", content="U3-new")]


@pytest.mark.asyncio
async def test_rebuild_from_chain_uses_deepest_valid_compacted_ancestor():
    p0_items = make_message_items(("user", "U1"), ("assistant", "A1"))
    p0 = ContextPartition(conversation_uuid="conv", partition_uuid="p0", items=p0_items)
    p1 = ContextPartition(
        conversation_uuid="conv",
        partition_uuid="p1",
        parent_partition_uuid="p0",
        ancestor_summaries=["Summary P0"],
        raw_message_start_index=2,
        items=make_message_items(("user", "U2")),
    )
    search = FakeSearchClient([
        make_compacted_doc(p0_items, p0, summary="Summary P0"),
        make_doc(p1),
    ])
    wb = make_web_base(search_client=search)
    conversation = ChatConversation(
        conversation_uuid="conv",
        partition_uuid="p1",
        messages=make_chat_messages(
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U2-new"),
        ),
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.parent_partition_uuid == "p0"
    assert partition.ancestor_summaries == ["Summary P0"]
    assert partition.raw_message_start_index == 2
    assert partition.items == [ContextPartitionItem(role="user", content="U2-new")]


@pytest.mark.asyncio
async def test_stream_and_finalize_emits_partition_uuid_and_compaction_pending():
    wb = make_tracking_web_base()
    partition = ContextPartition(
        conversation_uuid="abc",
        items=[ContextPartitionItem(role="user", content="Hi")],
    )

    async def fake_compact(snapshot):
        return "Summary"

    async def fake_generator():
        yield json.dumps({"text_delta": "Hello"}) + "\n"

    events = []
    async for chunk in wb.stream_and_finalize(
        context_partition=partition,
        conversation_uuid="abc",
        response_generator=fake_generator(),
        pending_compaction=[True],
        compact_fn=fake_compact,
    ):
        events.append(json.loads(chunk.strip()))

    try:
        assert events[0] == {"partition_uuid": partition.partition_uuid}
        assert {"compaction_pending": True} in events
        assert len(wb.background_tasks) == 1
    finally:
        for coro in wb.background_tasks:
            coro.close()


@pytest.mark.asyncio
async def test_stream_and_finalize_skips_duplicate_compaction_when_lock_held():
    wb = make_tracking_web_base(redis_data={"compaction_lock:abc": "1"})
    partition = ContextPartition(
        conversation_uuid="abc",
        items=[ContextPartitionItem(role="user", content="Hi")],
    )

    async def fake_compact(snapshot):
        return "Summary"

    async def fake_generator():
        yield json.dumps({"text_delta": "Hello"}) + "\n"

    events = []
    async for chunk in wb.stream_and_finalize(
        context_partition=partition,
        conversation_uuid="abc",
        response_generator=fake_generator(),
        pending_compaction=[True],
        compact_fn=fake_compact,
    ):
        events.append(json.loads(chunk.strip()))

    assert len(wb.background_tasks) == 1
    [finalize_coro] = list(wb.background_tasks)
    try:
        assert {"compaction_pending": True} not in events
        assert finalize_coro.cr_code.co_name == "finalize"
        assert wb.redis_client.sets == []
    finally:
        finalize_coro.close()


@pytest.mark.asyncio
async def test_sync_context_partition_accepts_compacted_child_of_client_partition():
    compacted_child = ContextPartition(
        conversation_uuid="abc",
        partition_uuid="new-active",
        parent_partition_uuid="client-old",
        ancestor_summaries=["Summary of U1/A1."],
        raw_message_start_index=2,
        items=make_message_items(("user", "U2"), ("assistant", "A2")),
    )
    wb = make_web_base({"context_partition:abc": compacted_child.model_dump_json()})
    conversation = ChatConversation(
        conversation_uuid="abc",
        partition_uuid="client-old",
        messages=make_chat_messages(
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U2"),
            ("assistant", "A2"),
            ("user", "U3"),
        ),
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.partition_uuid == "new-active"
    assert partition.parent_partition_uuid == "client-old"
    assert partition.ancestor_summaries == ["Summary of U1/A1."]
    assert partition.raw_message_start_index == 2
    assert partition.items[-1] == ContextPartitionItem(role="user", content="U3")


@pytest.mark.asyncio
async def test_sync_context_partition_edit_within_raw_window_preserves_ancestor_summaries():
    cached = ContextPartition(
        conversation_uuid="abc",
        partition_uuid="p1",
        ancestor_summaries=["Summary P0"],
        raw_message_start_index=2,
        items=make_message_items(
            ("user", "U2"),
            ("assistant", "A2"),
            ("user", "U3"),
            ("assistant", "A3"),
        ),
    )
    wb = make_web_base({"context_partition:abc": cached.model_dump_json()})
    conversation = ChatConversation(
        conversation_uuid="abc",
        partition_uuid="p1",
        messages=make_chat_messages(
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U2"),
            ("assistant", "A2"),
            ("user", "U3-edited"),
        ),
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.ancestor_summaries == ["Summary P0"]
    assert partition.raw_message_start_index == 2
    assert partition.items == make_message_items(
        ("user", "U2"),
        ("assistant", "A2"),
        ("user", "U3-edited"),
    )


@pytest.mark.asyncio
async def test_sync_context_partition_retry_before_raw_window_start_falls_back_to_fresh_partition():
    cached = ContextPartition(
        conversation_uuid="abc",
        partition_uuid="p1",
        ancestor_summaries=["Summary P0"],
        raw_message_start_index=4,
        items=make_message_items(
            ("user", "U3"),
            ("assistant", "A3"),
            ("user", "U4"),
            ("assistant", "A4"),
        ),
    )
    wb = make_web_base({"context_partition:abc": cached.model_dump_json()})
    conversation = ChatConversation(
        conversation_uuid="abc",
        partition_uuid="p1",
        messages=make_chat_messages(
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U2-edited"),
        ),
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.ancestor_summaries == []
    assert partition.raw_message_start_index == 0
    assert partition.items == make_message_items(
        ("user", "U1"),
        ("assistant", "A1"),
        ("user", "U2-edited"),
    )


def test_sync_context_window_with_raw_message_start_index():
    context_partition = ContextPartition(
        conversation_uuid="test",
        ancestor_summaries=["Earlier summary"],
        raw_message_start_index=2,
        items=make_message_items(("user", "U2"), ("assistant", "A2")),
    )

    context_partition.sync_from_conversation(ChatConversation(
        conversation_uuid="abc",
        messages=make_chat_messages(
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U2"),
            ("assistant", "A2"),
            ("user", "U3"),
        ),
    ))

    assert context_partition.items == make_message_items(
        ("user", "U2"),
        ("assistant", "A2"),
        ("user", "U3"),
    )


@pytest.mark.asyncio
async def test_walk_partition_chain_stops_on_conversation_uuid_mismatch():
    p0 = ContextPartition(
        conversation_uuid="conv-b",
        partition_uuid="p0",
        items=make_message_items(("user", "U0")),
    )
    p1 = ContextPartition(
        conversation_uuid="conv-a",
        partition_uuid="p1",
        parent_partition_uuid="p0",
        items=make_message_items(("user", "U1")),
    )
    wb = make_web_base(search_client=FakeSearchClient([
        make_doc(p0),
        make_doc(p1),
    ]))

    chain = await wb._walk_partition_chain("conv-a", "p1")

    assert [doc["partition_uuid"] for doc in chain] == ["p1"]


@pytest.mark.asyncio
async def test_walk_partition_chain_stops_on_cycle():
    p2 = ContextPartition(
        conversation_uuid="conv",
        partition_uuid="p2",
        parent_partition_uuid="p2",
        items=make_message_items(("user", "U2")),
    )
    wb = make_web_base(search_client=FakeSearchClient([make_doc(p2)]))

    chain = await wb._walk_partition_chain("conv", "p2")

    assert [doc["partition_uuid"] for doc in chain] == ["p2"]


@pytest.mark.asyncio
async def test_walk_partition_chain_stops_when_intermediate_partition_missing():
    p0 = ContextPartition(
        conversation_uuid="conv",
        partition_uuid="p0",
        items=make_message_items(("user", "U0")),
    )
    p2 = ContextPartition(
        conversation_uuid="conv",
        partition_uuid="p2",
        parent_partition_uuid="p1",
        items=make_message_items(("user", "U2")),
    )
    wb = make_web_base(search_client=FakeSearchClient([
        make_doc(p0),
        make_doc(p2),
    ]))

    chain = await wb._walk_partition_chain("conv", "p2")

    assert [doc["partition_uuid"] for doc in chain] == ["p2"]
