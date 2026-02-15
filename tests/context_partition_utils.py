from prokaryotes.api_v1.models import (
    ChatMessage,
    ContextPartition,
    ContextPartitionItem,
    compute_boundary_hash,
    compute_tail_hash,
    conversation_message_items,
)
from prokaryotes.web_v1 import WebBase


class FakePipeline:
    def __init__(self, redis):
        self.commands = []
        self.redis = redis
        self.watched_key: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self):
        for key, value, ex in self.commands:
            await self.redis.set(key, value, ex=ex)

    async def get(self, key):
        return await self.redis.get(key)

    def multi(self):
        self.commands = []

    async def reset(self):
        self.commands = []

    def set(self, key, value, ex=None):
        self.commands.append((key, value, ex))

    async def watch(self, key: str):
        self.watched_key = key


class FakeRedis:
    def __init__(self, data: dict | None = None):
        self._data: dict = {}
        for key, value in (data or {}).items():
            self._data[key] = value.encode() if isinstance(value, str) else value
        self.deletes: list[tuple] = []
        self.sets: list[tuple] = []

    async def delete(self, *keys):
        self.deletes.append(keys)
        for key in keys:
            self._data.pop(key, None)

    async def exists(self, key: str) -> int:
        return 1 if key in self._data else 0

    async def get(self, key: str):
        return self._data.get(key)

    def pipeline(self):
        return FakePipeline(self)

    async def set(self, key: str, value, ex=None, nx=False):
        stored_value = value.encode() if isinstance(value, str) else value
        if nx and key in self._data:
            return False
        self._data[key] = stored_value
        self.sets.append((key, value, ex, nx))
        return True


class FakeSearchClient:
    def __init__(self, docs=None):
        self.docs = {doc["partition_uuid"]: dict(doc) for doc in (docs or [])}
        self.puts = []
        self.updates = []

    async def find_partition_by_tail_hash(self, conversation_uuid: str, tail_hash: str) -> dict | None:
        for doc in self.docs.values():
            if (
                doc.get("conversation_uuid") == conversation_uuid
                and doc.get("tail_hash") == tail_hash
                and doc.get("is_compacted")
            ):
                return doc
        return None

    async def get_partition(self, partition_uuid: str) -> dict | None:
        return self.docs.get(partition_uuid)

    async def put_partition(self, partition: ContextPartition) -> None:
        doc = make_doc(partition)
        self.docs[partition.partition_uuid] = doc
        self.puts.append(partition)

    async def search_partitions(self, conversation_uuid: str, query: str) -> list[dict]:
        return []

    async def update_partition(self, partition_uuid: str, **fields) -> None:
        self.updates.append((partition_uuid, fields))
        self.docs.setdefault(partition_uuid, {"partition_uuid": partition_uuid}).update(fields)


def make_chat_messages(*role_contents: tuple[str, str]) -> list[ChatMessage]:
    return [
        ChatMessage(role=role, content=content)
        for role, content in role_contents
    ]


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


def make_message_items(*role_contents: tuple[str, str]) -> list[ContextPartitionItem]:
    return [
        ContextPartitionItem(role=role, content=content)
        for role, content in role_contents
    ]


def make_web_base(redis_data: dict | None = None, search_client=None) -> WebBase:
    wb = object.__new__(WebBase)
    wb.background_tasks = set()
    wb.conversation_cache_ex = 3600
    wb.redis_client = FakeRedis(redis_data)
    wb.search_client = search_client or FakeSearchClient()
    return wb
