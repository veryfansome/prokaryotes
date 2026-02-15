import pytest

from prokaryotes.api_v1.models import (
    ChatConversation,
    ChatMessage,
    ContextPartition,
    ContextPartitionItem,
)
from prokaryotes.web_v1 import (
    _message_count_before_item_index,
    _partition_can_follow_client,
    _recency_tail_items,
    get_postgres_pool,
    get_redis_client,
    hash_password,
    verify_password,
)
from tests.context_partition_utils import (
    FakeSearchClient,
    make_doc,
    make_web_base,
)


@pytest.mark.asyncio
async def test_finalize_strips_system_message_and_persists_to_redis_and_es():
    wb = make_web_base()
    partition = ContextPartition(
        conversation_uuid="abc",
        items=[
            ContextPartitionItem(role="system", content="Be brief"),
            ContextPartitionItem(role="user", content="Hi"),
            ContextPartitionItem(role="assistant", content="Hello"),
        ],
    )

    await wb.finalize(partition)

    cached = ContextPartition.model_validate_json(wb.redis_client._data["context_partition:abc"])
    assert cached.partition_uuid == partition.partition_uuid
    assert all(item.role != "system" for item in cached.items)
    assert wb.search_client.puts[0].partition_uuid == partition.partition_uuid


@pytest.mark.asyncio
async def test_get_postgres_pool_raises_without_env(monkeypatch):
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    with pytest.raises(RuntimeError, match="Unable to initialize postgres pool"):
        await get_postgres_pool()


def test_get_redis_client_raises_without_env(monkeypatch):
    monkeypatch.delenv("REDIS_HOST", raising=False)
    with pytest.raises(RuntimeError, match="Unable to initialize Redis client"):
        get_redis_client()


# hash_password / verify_password

def test_hash_password_produces_verifiable_hash():
    hashed = hash_password("secret")
    assert hashed != "secret"
    assert verify_password("secret", hashed) is True


# _message_count_before_item_index

def test_message_count_before_item_index_empty_slice():
    items = [ContextPartitionItem(role="user", content="U1")]
    assert _message_count_before_item_index(items, 0) == 0


def test_message_count_before_item_index_skips_non_messages():
    items = [
        ContextPartitionItem(role="user", content="U1"),
        ContextPartitionItem(type="function_call", name="tool", arguments="{}"),
        ContextPartitionItem(type="function_call_output", output="result"),
        ContextPartitionItem(role="assistant", content="A1"),
        ContextPartitionItem(role="user", content="U2"),
    ]
    # Before index 4 (U2): U1 and A1 are messages, tool call/output are not
    assert _message_count_before_item_index(items, 4) == 2


# _partition_can_follow_client

def test_partition_can_follow_client_returns_false_when_no_match():
    partition = ContextPartition(
        conversation_uuid="abc", partition_uuid="p2", parent_partition_uuid="p1", items=[]
    )
    assert _partition_can_follow_client(partition, "p3") is False


def test_partition_can_follow_client_when_no_client_uuid():
    partition = ContextPartition(conversation_uuid="abc", items=[])
    assert _partition_can_follow_client(partition, None) is True


def test_partition_can_follow_client_when_parent_uuid_matches():
    partition = ContextPartition(
        conversation_uuid="abc", partition_uuid="p2", parent_partition_uuid="p1", items=[]
    )
    assert _partition_can_follow_client(partition, "p1") is True


def test_partition_can_follow_client_when_uuid_matches():
    partition = ContextPartition(conversation_uuid="abc", partition_uuid="p1", items=[])
    assert _partition_can_follow_client(partition, "p1") is True


# _recency_tail_items

def test_recency_tail_items_advances_past_assistant_leading_boundary():
    items = [
        ContextPartitionItem(role="user", content="U1"),
        ContextPartitionItem(role="assistant", content="A1 preamble"),
        ContextPartitionItem(type="function_call", name="lookup", arguments="{}"),
        ContextPartitionItem(type="function_call_output", output="result"),
        ContextPartitionItem(role="assistant", content="A1 cont"),
        ContextPartitionItem(role="user", content="U2"),
        ContextPartitionItem(role="assistant", content="A2"),
        ContextPartitionItem(role="user", content="U3"),
        ContextPartitionItem(role="assistant", content="A3"),
        ContextPartitionItem(role="user", content="U4"),
    ]
    # 8 message items; with tail_count=6 the first candidate is assistant "A1 cont"
    # at item index 4 — must advance to user "U2" at item index 5.
    tail, offset = _recency_tail_items(items, 6)

    assert tail[0] == ContextPartitionItem(role="user", content="U2")
    # items before the tail: U1, A1 preamble, A1 cont = 3 message items
    assert offset == 3


def test_recency_tail_items_returns_empty_when_no_user_message_in_tail():
    items = [
        ContextPartitionItem(role="user", content="U1"),
        ContextPartitionItem(role="assistant", content="A1"),
        ContextPartitionItem(role="assistant", content="A2"),
    ]
    # tail_count=2 → candidate starts at A1, no user found → empty tail
    tail, offset = _recency_tail_items(items, 2)

    assert tail == []
    assert offset == 0


@pytest.mark.asyncio
async def test_sync_context_partition_skips_mismatched_redis_and_uses_exact_es():
    cached = ContextPartition(
        conversation_uuid="abc",
        partition_uuid="hot",
        items=[ContextPartitionItem(role="user", content="Wrong branch")],
    )
    exact = ContextPartition(
        conversation_uuid="abc",
        partition_uuid="client",
        items=[ContextPartitionItem(role="user", content="Hi")],
    )
    wb = make_web_base(
        redis_data={"context_partition:abc": cached.model_dump_json()},
        search_client=FakeSearchClient([make_doc(exact)]),
    )
    conversation = ChatConversation(
        conversation_uuid="abc",
        partition_uuid="client",
        messages=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ],
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.partition_uuid == "client"
    assert partition.items == [
        ContextPartitionItem(role="user", content="Hi"),
        ContextPartitionItem(role="assistant", content="Hello"),
    ]


@pytest.mark.asyncio
async def test_sync_context_partition_uses_redis_cache_when_no_client_uuid():
    cached = ContextPartition(
        conversation_uuid="abc",
        partition_uuid="p1",
        items=[ContextPartitionItem(role="user", content="Hi")],
    )
    wb = make_web_base(
        redis_data={"context_partition:abc": cached.model_dump_json()},
        search_client=FakeSearchClient(),
    )
    conversation = ChatConversation(
        conversation_uuid="abc",
        partition_uuid=None,
        messages=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ],
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.partition_uuid == "p1"
    assert partition.items[-1] == ContextPartitionItem(role="assistant", content="Hello")
    assert wb.search_client.puts == []


@pytest.mark.asyncio
async def test_sync_context_partition_uses_redis_cache_when_uuid_matches():
    cached = ContextPartition(
        conversation_uuid="abc",
        partition_uuid="p1",
        items=[ContextPartitionItem(role="user", content="Hi")],
    )
    wb = make_web_base(
        redis_data={"context_partition:abc": cached.model_dump_json()},
        search_client=FakeSearchClient(),
    )
    conversation = ChatConversation(
        conversation_uuid="abc",
        partition_uuid="p1",
        messages=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ],
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.partition_uuid == "p1"
    assert partition.items == [
        ContextPartitionItem(role="user", content="Hi"),
        ContextPartitionItem(role="assistant", content="Hello"),
    ]
    # ES was not consulted
    assert wb.search_client.puts == []


def test_verify_password_returns_false_for_wrong_password():
    hashed = hash_password("secret")
    assert verify_password("wrong", hashed) is False
