"""HTTP handler for the browser compaction-status polling endpoint.

Reads only from Redis. The compactor writes `compaction_status:{pending_snapshot_uuid}`
at CAS-commit step with either the child `snapshot_uuid` (relabel target) or
an empty-string sentinel (no relabel target — lock released without a commit).

Returning `done=true` without `snapshot_uuid` is correct for both the empty-
sentinel case and the long-idle-past-TTL case; the client clears its indicator
without relabeling either way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fastapi import HTTPException, Request
from redis.asyncio import Redis
from starsessions import load_session

from prokaryotes.api_v1.models import CompactionStatusResponse


class CompactionStatusHandler(ABC):
    async def get_compaction_status(
        self,
        request: Request,
        conversation_uuid: str,
        pending_snapshot_uuid: str,
    ) -> CompactionStatusResponse:
        await load_session(request)
        if not request.session:
            raise HTTPException(status_code=400, detail="Session expired")
        # If the compaction lock is still held, the swap hasn't run yet.
        lock_exists = await self.redis_client.exists(f"compaction_lock:{conversation_uuid}")
        if lock_exists:
            return CompactionStatusResponse(done=False)
        # Lock released — compactor either committed a child or skipped (empty summary,
        # prefix divergence, live→stale tombstone). The compactor writes the relabel target
        # (or "" sentinel) to `compaction_status:{pending_snapshot_uuid}` at the commit step.
        relabel_target = await self.redis_client.get(f"compaction_status:{pending_snapshot_uuid}")
        if relabel_target:
            target_str = relabel_target.decode("utf-8") if isinstance(relabel_target, bytes) else relabel_target
            if target_str:
                return CompactionStatusResponse(done=True, snapshot_uuid=target_str)
        return CompactionStatusResponse(done=True)

    @property
    @abstractmethod
    def redis_client(self) -> Redis: ...
