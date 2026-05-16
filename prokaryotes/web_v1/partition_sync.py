import logging
from abc import ABC, abstractmethod

from redis.asyncio import Redis

from prokaryotes.api_v1.models import (
    ChatConversation,
    ContextPartition,
    ContextPartitionItem,
    ConversationMatchesPartitionError,
    compute_boundary_hash,
    conversation_message_items,
)
from prokaryotes.search_v1 import SearchClient
from prokaryotes.search_v1.context_partitions import (
    items_from_doc,
    partition_from_doc,
)

logger = logging.getLogger(__name__)


class PartitionSyncer(ABC):
    """Three-tier partition reconciliation: Redis fast path, exact ES load, ancestor-chain rebuild."""

    async def _boundary_message_items_for_doc(
        self,
        doc: dict,
        memo: dict[str, list[ContextPartitionItem]] | None = None,
    ) -> list[ContextPartitionItem]:
        memo = memo or {}
        partition_uuid = doc.get("partition_uuid")
        if partition_uuid in memo:
            return memo[partition_uuid]

        prefix: list[ContextPartitionItem] = []
        raw_start = doc.get("raw_message_start_index") or 0
        parent_uuid = doc.get("parent_partition_uuid")
        if parent_uuid and raw_start > 0:
            parent_doc = await self.search_client.get_partition(parent_uuid)
            if parent_doc:
                parent_messages = await self._boundary_message_items_for_doc(parent_doc, memo)
                prefix = parent_messages[:raw_start]

        items = items_from_doc(doc)

        messages = prefix + conversation_message_items(items)
        if partition_uuid:
            memo[partition_uuid] = messages
        return messages

    async def _boundary_message_items_for_partition(
        self,
        partition: ContextPartition,
    ) -> list[ContextPartitionItem]:
        prefix: list[ContextPartitionItem] = []
        if partition.parent_partition_uuid and partition.raw_message_start_index > 0:
            parent_doc = await self.search_client.get_partition(partition.parent_partition_uuid)
            if parent_doc:
                parent_messages = await self._boundary_message_items_for_doc(parent_doc)
                prefix = parent_messages[: partition.raw_message_start_index]
        return prefix + conversation_message_items(partition.items)

    async def _cache_and_persist_partition(self, context_partition: ContextPartition) -> None:
        await self.redis_client.set(
            f"context_partition:{context_partition.conversation_uuid}",
            context_partition.model_dump_json(),
            ex=self.conversation_cache_ex,
        )
        await self.search_client.put_partition(context_partition)

    async def _load_exact_partition(
        self,
        conversation_uuid: str,
        partition_uuid: str,
    ) -> tuple[ContextPartition | None, dict | None]:
        """Fetch the partition doc for `partition_uuid`.

        Returns (partition, None) when the doc exists and is not compacted.
        Returns (None, doc) when the doc is compacted so the caller can pass it as head_doc
        to _rebuild_from_chain, avoiding a redundant fetch in _walk_partition_chain.
        Returns (None, None) when the doc is missing or fails to parse.
        """
        doc = await self.search_client.get_partition(partition_uuid)
        if not doc:
            return None, None
        if doc.get("is_compacted"):
            return None, doc
        try:
            return partition_from_doc(conversation_uuid, doc), None
        except Exception as exc:
            logger.warning("Failed to rebuild exact ES partition %s: %s", partition_uuid, exc)
            return None, None

    async def _rebuild_from_chain(
        self,
        conversation: ChatConversation,
        head_doc: dict | None = None,
    ) -> ContextPartition:
        """Rebuild a ContextPartition by walking the ES ancestor chain.

        head_doc is forwarded to _walk_partition_chain to skip a redundant fetch when
        the caller already holds the doc for conversation.partition_uuid.
        """
        if not conversation.partition_uuid:
            logger.info("Starting new context partition from request payload")
            return conversation.to_context_partition()

        chain = await self._walk_partition_chain(
            conversation.conversation_uuid,
            conversation.partition_uuid,
            head_doc=head_doc,
        )
        compacted_ancestors = [doc for doc in reversed(chain) if doc.get("is_compacted")]

        matched_ancestor: dict | None = None
        for doc in compacted_ancestors:
            boundary_count = doc.get("boundary_message_count")
            boundary_hash = doc.get("boundary_hash")
            if boundary_count is None or not boundary_hash:
                continue
            if boundary_count > len(conversation.messages):
                continue
            incoming_hash = compute_boundary_hash(conversation.messages[:boundary_count])
            if incoming_hash == boundary_hash:
                matched_ancestor = doc

        if matched_ancestor is None:
            logger.info("No compacted ancestor validated; starting from raw request payload")
            return conversation.to_context_partition()

        ancestor_summaries = []
        for doc in compacted_ancestors:
            summary = doc.get("summary")
            if summary:
                ancestor_summaries.append(summary)
            if doc["partition_uuid"] == matched_ancestor["partition_uuid"]:
                break

        raw_message_start_index = matched_ancestor.get("boundary_message_count") or 0
        return ContextPartition(
            conversation_uuid=conversation.conversation_uuid,
            parent_partition_uuid=matched_ancestor["partition_uuid"],
            ancestor_summaries=ancestor_summaries,
            raw_message_start_index=raw_message_start_index,
            items=[message.to_context_partition_item() for message in conversation.messages[raw_message_start_index:]],
        )

    def _try_sync_partition(
        self,
        context_partition: ContextPartition,
        conversation: ChatConversation,
        source: str,
    ) -> ContextPartition | None:
        try:
            context_partition.sync_from_conversation(conversation)
            logger.info("Synced context partition from %s", source)
            return context_partition
        except ConversationMatchesPartitionError:
            logger.info("Context partition from %s already matches request", source)
            return context_partition
        except Exception as exc:
            logger.info("Context partition from %s could not sync: %s", source, exc)
            return None

    async def _walk_partition_chain(
        self,
        conversation_uuid: str,
        partition_uuid: str,
        head_doc: dict | None = None,
    ) -> list[dict]:
        """Walk the ancestor chain from partition_uuid back to the root, returning docs oldest-last.

        head_doc, if provided, is used as the first doc instead of fetching it — avoids a
        redundant ES round-trip when the caller already has the doc for partition_uuid.
        """
        chain = []
        seen: set[str] = set()
        current_uuid = partition_uuid
        while current_uuid and current_uuid not in seen:
            seen.add(current_uuid)
            if head_doc is not None and head_doc.get("partition_uuid") == current_uuid:
                doc = head_doc
                head_doc = None
            else:
                doc = await self.search_client.get_partition(current_uuid)
            if not doc:
                break
            if doc.get("conversation_uuid") and doc["conversation_uuid"] != conversation_uuid:
                break
            chain.append(doc)
            current_uuid = doc.get("parent_partition_uuid")
        return chain

    @property
    @abstractmethod
    def conversation_cache_ex(self) -> int:
        pass

    @property
    @abstractmethod
    def redis_client(self) -> Redis:
        pass

    @property
    @abstractmethod
    def search_client(self) -> SearchClient:
        pass

    async def sync_context_partition(self, conversation: ChatConversation) -> ContextPartition:
        cached_partition_data = await self.redis_client.get(f"context_partition:{conversation.conversation_uuid}")
        if cached_partition_data:
            context_partition = ContextPartition.model_validate_json(cached_partition_data)
            if _partition_can_follow_client(context_partition, conversation.partition_uuid):
                synced = self._try_sync_partition(context_partition, conversation, source="Redis")
                if synced is not None:
                    return synced
            else:
                logger.info(
                    "Cached partition %s does not match client partition %s; falling back to ES",
                    context_partition.partition_uuid,
                    conversation.partition_uuid,
                )

        head_doc: dict | None = None
        if conversation.partition_uuid:
            exact_partition, head_doc = await self._load_exact_partition(
                conversation.conversation_uuid,
                conversation.partition_uuid,
            )
            if exact_partition is not None:
                synced = self._try_sync_partition(exact_partition, conversation, source="ES")
                if synced is not None:
                    await self._cache_and_persist_partition(synced)
                    return synced

        rebuilt = await self._rebuild_from_chain(conversation, head_doc=head_doc)
        await self._cache_and_persist_partition(rebuilt)
        return rebuilt


def _partition_can_follow_client(partition: ContextPartition, client_partition_uuid: str | None) -> bool:
    if client_partition_uuid is None:
        return True
    return partition.partition_uuid == client_partition_uuid or partition.parent_partition_uuid == client_partition_uuid
