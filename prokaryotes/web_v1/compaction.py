from abc import ABC, abstractmethod

from fastapi import HTTPException, Request
from redis.asyncio import Redis
from starsessions import load_session

from prokaryotes.api_v1.models import CompactionStatusResponse, ContextPartition


class CompactionStatusHandler(ABC):
    """HTTP handler for the browser compaction-status polling endpoint."""

    async def get_compaction_status(
        self,
        request: Request,
        conversation_uuid: str,
        pending_partition_uuid: str,
    ) -> CompactionStatusResponse:
        await load_session(request)
        if not request.session:
            raise HTTPException(status_code=400, detail="Session expired")
        lock_exists = await self.redis_client.exists(f"compaction_lock:{conversation_uuid}")
        if lock_exists:
            return CompactionStatusResponse(done=False)
        # Lock is acquired in `stream_and_finalize` *before* the `compaction_pending`
        # event the UI polled in response to, so by the time we observe "no lock" the
        # compaction task has finished. Either the Redis swap committed to a direct
        # child of the pending UUID or the swap was skipped (empty summary, prefix
        # divergence, live->stale tombstone). Both must surface as done so the UI stops
        # polling — only the direct-child swap case carries a `partition_uuid` for
        # `relabelPartitionUuid` to consume.
        cached = await self.redis_client.get(f"context_partition:{conversation_uuid}")
        if cached:
            partition = ContextPartition.model_validate_json(cached)
            if (
                partition.partition_uuid != pending_partition_uuid
                and partition.parent_partition_uuid == pending_partition_uuid
            ):
                return CompactionStatusResponse(done=True, partition_uuid=partition.partition_uuid)
        return CompactionStatusResponse(done=True)

    @property
    @abstractmethod
    def redis_client(self) -> Redis:
        pass
