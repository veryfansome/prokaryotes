import pytest

from prokaryotes.api_v1.models import (
    ChatConversation,
    ChatMessage,
    ContextPartition,
    ContextPartitionItem,
)
from prokaryotes.web_v1 import WebBase


class FakeRedis:
    def __init__(self, data: dict = None):
        self._data: dict = data or {}
        self.sets: list[tuple] = []

    async def get(self, key: str):
        value = self._data.get(key)
        if value is None:
            return None
        return value if isinstance(value, bytes) else value.encode()

    async def set(self, key: str, value, ex=None):
        self.sets.append((key, value, ex))


def make_web_base(redis_data: dict = None) -> WebBase:
    wb = object.__new__(WebBase)
    wb.redis_client = FakeRedis(redis_data)
    wb.conversation_cache_ex = 3600
    return wb


@pytest.mark.asyncio
async def test_finalize_strips_system_message_and_caches():
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

    key, value, ex = wb.redis_client.sets[0]
    assert key == "context_partition:abc"
    assert ex == 3600
    cached = ContextPartition.model_validate_json(value)
    assert all(item.role != "system" for item in cached.items)
    assert cached.items == [
        ContextPartitionItem(role="user", content="Hi"),
        ContextPartitionItem(role="assistant", content="Hello"),
    ]


@pytest.mark.asyncio
async def test_finalize_no_system_message():
    wb = make_web_base()
    partition = ContextPartition(
        conversation_uuid="abc",
        items=[
            ContextPartitionItem(role="user", content="Hi"),
            ContextPartitionItem(role="assistant", content="Hello"),
        ],
    )

    await wb.finalize(partition)

    _, value, _ = wb.redis_client.sets[0]
    cached = ContextPartition.model_validate_json(value)
    assert len(cached.items) == 2


@pytest.mark.asyncio
async def test_sync_context_partition_cache_miss():
    wb = make_web_base()
    conversation = ChatConversation(
        conversation_uuid="abc",
        messages=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ],
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.conversation_uuid == "abc"
    assert partition.items == [
        ContextPartitionItem(role="user", content="Hi"),
        ContextPartitionItem(role="assistant", content="Hello"),
    ]


@pytest.mark.asyncio
async def test_sync_context_partition_cache_hit():
    cached_partition = ContextPartition(
        conversation_uuid="abc",
        items=[
            ContextPartitionItem(role="user", content="Hi"),
            ContextPartitionItem(role="assistant", content="Hello"),
            ContextPartitionItem(role="user", content="Tell me a joke"),
        ],
    )
    wb = make_web_base({"context_partition:abc": cached_partition.model_dump_json()})
    conversation = ChatConversation(
        conversation_uuid="abc",
        messages=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
            ChatMessage(role="user", content="Tell me a joke"),
            ChatMessage(role="assistant", content="Why did the chicken..."),
        ],
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.items == [
        ContextPartitionItem(role="user", content="Hi"),
        ContextPartitionItem(role="assistant", content="Hello"),
        ContextPartitionItem(role="user", content="Tell me a joke"),
        ContextPartitionItem(role="assistant", content="Why did the chicken..."),
    ]
