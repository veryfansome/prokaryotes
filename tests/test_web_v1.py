import json

import pytest

from prokaryotes.api_v1.models import (
    ChatConversation,
    ChatMessage,
    ContextPartition,
    ContextPartitionItem,
    compute_boundary_hash,
    compute_tail_hash,
    conversation_message_items,
)
from prokaryotes.web_v1 import (
    WebBase,
    _message_count_before_item_index,
    _partition_can_follow_client,
    _recency_tail_items,
    get_postgres_pool,
    get_redis_client,
    hash_password,
    verify_password,
)


class FakePipeline:
    def __init__(self, redis):
        self.commands = []
        self.redis = redis
        self.watched_key: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def watch(self, key: str):
        self.watched_key = key

    async def get(self, key):
        return await self.redis.get(key)

    def multi(self):
        self.commands = []

    def set(self, key, value, ex=None):
        self.commands.append((key, value, ex))

    async def execute(self):
        for key, value, ex in self.commands:
            await self.redis.set(key, value, ex=ex)

    async def reset(self):
        self.commands = []


class FakeRedis:
    def __init__(self, data: dict = None):
        self._data: dict = {}
        for key, value in (data or {}).items():
            self._data[key] = value.encode() if isinstance(value, str) else value
        self.sets: list[tuple] = []
        self.deletes: list[tuple] = []

    async def get(self, key: str):
        return self._data.get(key)

    async def set(self, key: str, value, ex=None, nx=False):
        stored_value = value.encode() if isinstance(value, str) else value
        if nx and key in self._data:
            return False
        self._data[key] = stored_value
        self.sets.append((key, value, ex, nx))
        return True

    async def delete(self, *keys):
        self.deletes.append(keys)
        for key in keys:
            self._data.pop(key, None)

    async def exists(self, key: str) -> int:
        return 1 if key in self._data else 0

    def pipeline(self):
        return FakePipeline(self)


class FakeSearchClient:
    def __init__(self, docs=None):
        self.docs = {doc["partition_uuid"]: dict(doc) for doc in (docs or [])}
        self.puts = []
        self.updates = []

    async def get_partition(self, partition_uuid: str) -> dict | None:
        return self.docs.get(partition_uuid)

    async def put_partition(self, partition: ContextPartition) -> None:
        message_items = conversation_message_items(partition.items)
        doc = {
            "partition_uuid": partition.partition_uuid,
            "conversation_uuid": partition.conversation_uuid,
            "parent_partition_uuid": partition.parent_partition_uuid,
            "ancestor_summaries": partition.ancestor_summaries,
            "raw_message_start_index": partition.raw_message_start_index,
            "is_compacted": False,
            "summary": None,
            "items_json": partition.model_dump_json(include={"items"}),
            "boundary_message_count": partition.raw_message_start_index + len(message_items),
            "boundary_user_count": sum(1 for item in message_items if item.role == "user"),
            "boundary_hash": compute_boundary_hash(message_items),
            "tail_hash": compute_tail_hash(message_items),
        }
        self.docs[partition.partition_uuid] = doc
        self.puts.append(partition)

    async def update_partition(self, partition_uuid: str, **fields) -> None:
        self.updates.append((partition_uuid, fields))
        self.docs.setdefault(partition_uuid, {"partition_uuid": partition_uuid}).update(fields)

    async def find_partition_by_tail_hash(self, conversation_uuid: str, tail_hash: str) -> dict | None:
        for doc in self.docs.values():
            if (
                doc.get("conversation_uuid") == conversation_uuid
                and doc.get("tail_hash") == tail_hash
                and doc.get("is_compacted")
            ):
                return doc
        return None

    async def search_partitions(self, conversation_uuid: str, query: str) -> list[dict]:
        return []


def make_doc(partition: ContextPartition, **overrides):
    message_items = conversation_message_items(partition.items)
    doc = {
        "partition_uuid": partition.partition_uuid,
        "conversation_uuid": partition.conversation_uuid,
        "parent_partition_uuid": partition.parent_partition_uuid,
        "ancestor_summaries": partition.ancestor_summaries,
        "raw_message_start_index": partition.raw_message_start_index,
        "is_compacted": False,
        "summary": None,
        "items_json": partition.model_dump_json(include={"items"}),
        "boundary_message_count": partition.raw_message_start_index + len(message_items),
        "boundary_user_count": sum(1 for item in message_items if item.role == "user"),
        "boundary_hash": compute_boundary_hash(message_items),
        "tail_hash": compute_tail_hash(message_items),
    }
    doc.update(overrides)
    return doc


def make_web_base(redis_data: dict = None, search_client=None) -> WebBase:
    wb = object.__new__(WebBase)
    wb.redis_client = FakeRedis(redis_data)
    wb.search_client = search_client or FakeSearchClient()
    wb.conversation_cache_ex = 3600
    wb.background_tasks = set()
    return wb


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
async def test_sync_context_partition_accepts_compacted_child_of_client_partition():
    compacted_child = ContextPartition(
        conversation_uuid="abc",
        partition_uuid="new-active",
        parent_partition_uuid="client-old",
        ancestor_summaries=["Summary of U1/A1."],
        raw_message_start_index=2,
        items=[
            ContextPartitionItem(role="user", content="U2"),
            ContextPartitionItem(role="assistant", content="A2"),
        ],
    )
    wb = make_web_base({"context_partition:abc": compacted_child.model_dump_json()})
    conversation = ChatConversation(
        conversation_uuid="abc",
        partition_uuid="client-old",
        messages=[
            ChatMessage(role="user", content="U1"),
            ChatMessage(role="assistant", content="A1"),
            ChatMessage(role="user", content="U2"),
            ChatMessage(role="assistant", content="A2"),
            ChatMessage(role="user", content="U3"),
        ],
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.partition_uuid == "new-active"
    assert partition.parent_partition_uuid == "client-old"
    assert partition.ancestor_summaries == ["Summary of U1/A1."]
    assert partition.raw_message_start_index == 2
    assert partition.items[-1] == ContextPartitionItem(role="user", content="U3")


@pytest.mark.asyncio
async def test_rebuild_from_chain_uses_deepest_valid_compacted_ancestor():
    p0_items = [
        ContextPartitionItem(role="user", content="U1"),
        ContextPartitionItem(role="assistant", content="A1"),
    ]
    p0 = ContextPartition(conversation_uuid="conv", partition_uuid="p0", items=p0_items)
    p0_doc = make_doc(
        p0,
        is_compacted=True,
        summary="Summary P0",
        boundary_hash=compute_boundary_hash(p0_items),
        boundary_message_count=2,
    )
    p1 = ContextPartition(
        conversation_uuid="conv",
        partition_uuid="p1",
        parent_partition_uuid="p0",
        ancestor_summaries=["Summary P0"],
        raw_message_start_index=2,
        items=[ContextPartitionItem(role="user", content="U2")],
    )
    wb = make_web_base(search_client=FakeSearchClient([p0_doc, make_doc(p1)]))
    conversation = ChatConversation(
        conversation_uuid="conv",
        partition_uuid="p1",
        messages=[
            ChatMessage(role="user", content="U1"),
            ChatMessage(role="assistant", content="A1"),
            ChatMessage(role="user", content="U2-new"),
        ],
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.parent_partition_uuid == "p0"
    assert partition.ancestor_summaries == ["Summary P0"]
    assert partition.raw_message_start_index == 2
    assert partition.items == [ContextPartitionItem(role="user", content="U2-new")]


@pytest.mark.asyncio
async def test_rebuild_from_chain_does_not_inject_stale_summary_without_valid_boundary():
    p0_items = [
        ContextPartitionItem(role="user", content="U1"),
        ContextPartitionItem(role="assistant", content="A1"),
    ]
    p0 = ContextPartition(conversation_uuid="conv", partition_uuid="p0", items=p0_items)
    p0_doc = make_doc(
        p0,
        is_compacted=True,
        summary="Stale summary",
        boundary_hash=compute_boundary_hash(p0_items),
        boundary_message_count=2,
    )
    wb = make_web_base(search_client=FakeSearchClient([p0_doc]))
    conversation = ChatConversation(
        conversation_uuid="conv",
        partition_uuid="p0",
        messages=[
            ChatMessage(role="user", content="Edited U1"),
            ChatMessage(role="assistant", content="A1"),
        ],
    )

    partition = await wb.sync_context_partition(conversation)

    assert partition.parent_partition_uuid is None
    assert partition.ancestor_summaries == []
    assert partition.items == [
        ContextPartitionItem(role="user", content="Edited U1"),
        ContextPartitionItem(role="assistant", content="A1"),
    ]


@pytest.mark.asyncio
async def test_stream_and_finalize_emits_partition_uuid_and_compaction_pending():
    class TrackingWebBase(WebBase):
        def background_and_forget(self, coro):
            self.background_tasks.add(coro)

    wb = object.__new__(TrackingWebBase)
    wb.redis_client = FakeRedis()
    wb.search_client = FakeSearchClient()
    wb.conversation_cache_ex = 3600
    wb.background_tasks = set()
    partition = ContextPartition(
        conversation_uuid="abc",
        items=[ContextPartitionItem(role="user", content="Hi")],
    )

    async def fake_generator():
        yield json.dumps({"text_delta": "Hello"}) + "\n"

    async def fake_compact(snapshot):
        return "Summary"

    events = []
    async for chunk in wb.stream_and_finalize(
        context_partition=partition,
        conversation_uuid="abc",
        response_generator=fake_generator(),
        pending_compaction=[True],
        compact_fn=fake_compact,
    ):
        events.append(json.loads(chunk.strip()))

    assert events[0] == {"partition_uuid": partition.partition_uuid}
    assert {"compaction_pending": True} in events
    assert len(wb.background_tasks) == 1
    for coro in wb.background_tasks:
        coro.close()


@pytest.mark.asyncio
async def test_compact_partition_retries_redis_swap_on_watch_contention():
    from redis.exceptions import WatchError

    snapshot = ContextPartition(
        conversation_uuid="conv",
        partition_uuid="snap",
        items=[
            ContextPartitionItem(role="user", content="U1"),
            ContextPartitionItem(role="assistant", content="A1"),
        ],
    )
    lock_key = "compaction_lock:conv"

    execute_calls = []

    class ContendedPipeline(FakePipeline):
        async def execute(self):
            execute_calls.append(len(execute_calls) + 1)
            if len(execute_calls) == 1:
                raise WatchError()
            await super().execute()

    class ContendedRedis(FakeRedis):
        def pipeline(self):
            return ContendedPipeline(self)

    redis = ContendedRedis({"context_partition:conv": snapshot.model_dump_json()})
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(s):
        return "Summary"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    assert len(execute_calls) == 2
    assert await redis.exists(lock_key) == 0

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    assert cached.parent_partition_uuid == "snap"
    assert cached.ancestor_summaries == ["Summary"]
    assert search.puts[-1].parent_partition_uuid == "snap"


# _partition_can_follow_client

def test_partition_can_follow_client_when_no_client_uuid():
    partition = ContextPartition(conversation_uuid="abc", items=[])
    assert _partition_can_follow_client(partition, None) is True


def test_partition_can_follow_client_when_uuid_matches():
    partition = ContextPartition(conversation_uuid="abc", partition_uuid="p1", items=[])
    assert _partition_can_follow_client(partition, "p1") is True


def test_partition_can_follow_client_when_parent_uuid_matches():
    partition = ContextPartition(
        conversation_uuid="abc", partition_uuid="p2", parent_partition_uuid="p1", items=[]
    )
    assert _partition_can_follow_client(partition, "p1") is True


def test_partition_can_follow_client_returns_false_when_no_match():
    partition = ContextPartition(
        conversation_uuid="abc", partition_uuid="p2", parent_partition_uuid="p1", items=[]
    )
    assert _partition_can_follow_client(partition, "p3") is False


# _message_count_before_item_index

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


def test_message_count_before_item_index_empty_slice():
    items = [ContextPartitionItem(role="user", content="U1")]
    assert _message_count_before_item_index(items, 0) == 0


# hash_password / verify_password

def test_hash_password_produces_verifiable_hash():
    hashed = hash_password("secret")
    assert hashed != "secret"
    assert verify_password("secret", hashed) is True


def test_verify_password_returns_false_for_wrong_password():
    hashed = hash_password("secret")
    assert verify_password("wrong", hashed) is False


# get_redis_client

def test_get_redis_client_raises_without_env(monkeypatch):
    monkeypatch.delenv("REDIS_HOST", raising=False)
    with pytest.raises(RuntimeError, match="Unable to initialize Redis client"):
        get_redis_client()


@pytest.mark.asyncio
async def test_get_postgres_pool_raises_without_env(monkeypatch):
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    with pytest.raises(RuntimeError, match="Unable to initialize postgres pool"):
        await get_postgres_pool()


# Redis cache hit path in sync_context_partition

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
