import pytest
from redis.exceptions import WatchError

from prokaryotes.api_v1.models import (
    ContextPartition,
    compute_boundary_hash,
)
from prokaryotes.utils_v1.llm_utils import COMPACTION_RECENCY_TAIL
from prokaryotes.web_v1 import _recency_tail_items
from tests.context_partition_utils import (
    FakePipeline,
    FakeRedis,
    FakeSearchClient,
    make_doc,
    make_message_items,
    make_web_base,
)


def make_snapshot(
        ancestor_summaries: list[str] | None = None,
        conversation_uuid: str = "conv",
        items=None,
        parent_partition_uuid: str | None = None,
        partition_uuid: str = "snap",
        raw_message_start_index: int = 0,
) -> ContextPartition:
    return ContextPartition(
        conversation_uuid=conversation_uuid,
        partition_uuid=partition_uuid,
        parent_partition_uuid=parent_partition_uuid,
        ancestor_summaries=ancestor_summaries or [],
        raw_message_start_index=raw_message_start_index,
        items=items or make_message_items(("user", "U1"), ("assistant", "A1")),
    )


@pytest.mark.asyncio
async def test_boundary_hash_stored_on_es_covers_full_parent_prefix():
    p0 = make_snapshot(
        partition_uuid="p0",
        items=make_message_items(("user", "U1"), ("assistant", "A1")),
    )
    snapshot = make_snapshot(
        partition_uuid="p1",
        parent_partition_uuid="p0",
        raw_message_start_index=2,
        items=make_message_items(("user", "U2"), ("assistant", "A2")),
    )
    expected_boundary_items = make_message_items(
        ("user", "U1"),
        ("assistant", "A1"),
        ("user", "U2"),
        ("assistant", "A2"),
    )
    redis = FakeRedis({"context_partition:conv": snapshot.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(p0), make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(current_snapshot):
        return "S1"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    update_fields = search.updates[0][1]
    assert update_fields["boundary_hash"] == compute_boundary_hash(expected_boundary_items)
    assert update_fields["boundary_message_count"] == 4


@pytest.mark.asyncio
async def test_compact_partition_accumulates_ancestor_summaries_across_generations():
    snapshot = make_snapshot(
        partition_uuid="p1",
        ancestor_summaries=["S0"],
        items=make_message_items(("user", "U2"), ("assistant", "A2")),
    )
    redis = FakeRedis({"context_partition:conv": snapshot.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(current_snapshot):
        return "S1"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    assert cached.ancestor_summaries == ["S0", "S1"]
    assert cached.parent_partition_uuid == "p1"


@pytest.mark.asyncio
async def test_compact_partition_advances_raw_message_start_index_by_tail_offset():
    snapshot = make_snapshot(
        items=make_message_items(
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U2"),
            ("assistant", "A2"),
            ("user", "U3"),
            ("assistant", "A3"),
            ("user", "U4"),
            ("assistant", "A4"),
        ),
    )
    expected_tail, tail_offset = _recency_tail_items(snapshot.items, COMPACTION_RECENCY_TAIL)
    redis = FakeRedis({"context_partition:conv": snapshot.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(current_snapshot):
        return "Summary"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    assert tail_offset == 2
    assert cached.raw_message_start_index == tail_offset
    assert cached.items == expected_tail


@pytest.mark.asyncio
async def test_compact_partition_carries_forward_post_snapshot_messages():
    snapshot = make_snapshot()
    current_partition = make_snapshot(
        items=make_message_items(
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U2"),
        ),
    )
    redis = FakeRedis({"context_partition:conv": current_partition.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(current_snapshot):
        return "Summary"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    assert cached.items == make_message_items(
        ("user", "U1"),
        ("assistant", "A1"),
        ("user", "U2"),
    )
    assert cached.parent_partition_uuid == "snap"


@pytest.mark.asyncio
async def test_compact_partition_releases_lock_when_compact_fn_raises():
    snapshot = make_snapshot()
    redis = FakeRedis({"context_partition:conv": snapshot.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(current_snapshot):
        raise RuntimeError("LLM unavailable")

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    assert cached.partition_uuid == "snap"
    assert cached.items == snapshot.items
    assert search.updates == []
    assert search.puts == []
    assert await redis.exists(lock_key) == 0


@pytest.mark.asyncio
async def test_compact_partition_retries_redis_swap_on_watch_contention():
    snapshot = make_snapshot()
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
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(current_snapshot):
        return "Summary"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    assert len(execute_calls) == 2
    assert cached.parent_partition_uuid == "snap"
    assert cached.ancestor_summaries == ["Summary"]
    assert search.puts[-1].parent_partition_uuid == "snap"
    assert await redis.exists(lock_key) == 0


@pytest.mark.asyncio
async def test_compact_partition_returns_early_for_empty_summary():
    snapshot = make_snapshot()
    redis = FakeRedis({"context_partition:conv": snapshot.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(current_snapshot):
        return ""

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    assert cached.partition_uuid == "snap"
    assert cached.items == snapshot.items
    assert search.updates == []
    assert search.puts == []
    assert await redis.exists(lock_key) == 0


@pytest.mark.asyncio
async def test_compact_partition_skips_swap_when_active_prefix_changed():
    snapshot = make_snapshot()
    current_partition = make_snapshot(
        raw_message_start_index=2,
        ancestor_summaries=["S0"],
        items=make_message_items(("user", "U2"), ("assistant", "A2")),
    )
    redis = FakeRedis({"context_partition:conv": current_partition.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(current_snapshot):
        return "Summary"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    assert cached.partition_uuid == current_partition.partition_uuid
    assert cached.raw_message_start_index == current_partition.raw_message_start_index
    assert cached.ancestor_summaries == current_partition.ancestor_summaries
    assert cached.items == current_partition.items
    assert search.puts == []


@pytest.mark.asyncio
async def test_compact_partition_skips_swap_when_active_uuid_changed():
    snapshot = make_snapshot()
    current_partition = make_snapshot(
        partition_uuid="new-branch",
        items=make_message_items(
            ("user", "U1"),
            ("assistant", "A1"),
            ("user", "U2"),
        ),
    )
    redis = FakeRedis({"context_partition:conv": current_partition.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(current_snapshot):
        return "Summary"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    assert cached.partition_uuid == "new-branch"
    assert cached.items == current_partition.items
    assert search.puts == []


@pytest.mark.asyncio
async def test_compact_partition_skips_swap_when_redis_partition_missing():
    snapshot = make_snapshot()
    redis = FakeRedis()
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(search_client=search)
    wb.redis_client = redis

    async def compact_fn(current_snapshot):
        return "Summary"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    assert not any(key == "context_partition:conv" for key, *_ in redis.sets)
    assert search.puts == []
    assert await redis.exists(lock_key) == 0
