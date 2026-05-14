import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import HTTPException, Request
from redis.exceptions import WatchError
from starsessions import load_session

from prokaryotes.api_v1.models import (
    ContextPartition,
    compute_boundary_hash,
    compute_tail_hash,
)
from prokaryotes.search_v1.context_partitions import (
    COMPACTION_STATE_COMMITTED,
    COMPACTION_STATE_PENDING,
)
from prokaryotes.tools_v1.file_tool.live_windows import (
    items_equal_mod_live_windows,
    lift_active_live_windows,
    recency_tail_items,
)
from prokaryotes.utils_v1.llm_utils import COMPACTION_RECENCY_TAIL
from prokaryotes.web_v1.partition_sync import PartitionSyncer

logger = logging.getLogger(__name__)


class PartitionCompactor(PartitionSyncer):
    """Compaction lifecycle: summary generation, CAS swap into a child partition, and the /compaction-status endpoint.

    Inherits from `PartitionSyncer` to reuse the three abstract `@property` contracts
    (`conversation_cache_ex`, `redis_client`, `search_client`) and `_boundary_message_items_for_partition`,
    which `_compact_partition` calls when computing the parent's boundary fields.
    """

    async def _compact_partition(
        self,
        compact_fn,
        conversation_uuid: str,
        lock_key: str,
        snapshot: ContextPartition,
    ) -> None:
        swapped_partition: ContextPartition | None = None
        try:
            summary = await compact_fn(snapshot)
            if not summary:
                logger.warning("Compaction produced empty summary for partition %s", snapshot.partition_uuid)
                return
            compaction_attempt_uuid = str(uuid.uuid4())
            child_partition_uuid = str(uuid.uuid4())

            # Boundary fields are derived from `snapshot` (deterministic) so we can compute
            # them up-front, but we defer writing `is_compacted=True` to ES until the Redis
            # CAS swap actually succeeds. If the swap is skipped (live -> stale tombstone,
            # concurrent edit, lost lock, etc.) and ES already says the parent is compacted,
            # `_rebuild_from_chain` would later treat the parent's stored summary as
            # authoritative and silently drop the un-swapped tail, including any tracked-file
            # state that hadn't been lifted yet.
            boundary_items = await self._boundary_message_items_for_partition(snapshot)
            boundary_hash = compute_boundary_hash(boundary_items)
            tail_hash = compute_tail_hash(boundary_items)
            boundary_message_count = len(boundary_items)
            boundary_user_count = sum(1 for item in boundary_items if item.role == "user")

            redis_key = f"context_partition:{conversation_uuid}"
            async with self.redis_client.pipeline() as pipe:
                while True:
                    try:
                        await pipe.watch(redis_key)
                        current_data = await pipe.get(redis_key)
                        if not current_data:
                            logger.info("Skipping compaction swap because Redis partition is missing")
                            return

                        current_partition = ContextPartition.model_validate_json(current_data)
                        if current_partition.partition_uuid != snapshot.partition_uuid:
                            logger.info(
                                "Skipping compaction swap for %s because active partition is now %s",
                                snapshot.partition_uuid,
                                current_partition.partition_uuid,
                            )
                            return
                        if (
                            current_partition.raw_message_start_index != snapshot.raw_message_start_index
                            or current_partition.ancestor_summaries != snapshot.ancestor_summaries
                            or not items_equal_mod_live_windows(
                                current_partition.items[: len(snapshot.items)],
                                snapshot.items,
                            )
                        ):
                            logger.info(
                                "Skipping compaction swap for %s because the active prefix changed",
                                snapshot.partition_uuid,
                            )
                            return

                        post_snapshot_items = current_partition.items[len(snapshot.items) :]
                        recency_tail, tail_offset = recency_tail_items(
                            snapshot.items,
                            COMPACTION_RECENCY_TAIL,
                        )
                        pre_tail = snapshot.items[: len(snapshot.items) - len(recency_tail)]
                        augmented_tail = lift_active_live_windows(
                            pre_tail,
                            recency_tail + post_snapshot_items,
                        )
                        swapped_partition = ContextPartition(
                            conversation_uuid=conversation_uuid,
                            partition_uuid=child_partition_uuid,
                            parent_partition_uuid=snapshot.partition_uuid,
                            ancestor_summaries=snapshot.ancestor_summaries + [summary],
                            raw_message_start_index=snapshot.raw_message_start_index + tail_offset,
                            items=augmented_tail,
                        )

                        async def put_child_partition(
                            partition: ContextPartition = swapped_partition,
                        ) -> None:
                            await self.search_client.put_partition(
                                partition,
                                compaction_attempt_uuid=compaction_attempt_uuid,
                                compaction_state=COMPACTION_STATE_PENDING,
                            )

                        await _retry_compaction_search_write(
                            f"persist child partition {swapped_partition.partition_uuid}",
                            put_child_partition,
                        )

                        pipe.multi()
                        pipe.set(
                            redis_key,
                            swapped_partition.model_dump_json(),
                            ex=self.conversation_cache_ex,
                        )
                        await pipe.execute()
                        logger.info(
                            "Compaction complete: new partition %s (parent %s)",
                            swapped_partition.partition_uuid,
                            snapshot.partition_uuid,
                        )
                        break
                    except WatchError:
                        logger.info("Compaction Redis swap contended, retrying")
                        await pipe.reset()

            if swapped_partition is not None:
                # The child is persisted as `pending` before the Redis CAS so the cache
                # never points at a partition Elasticsearch doesn't know about. After the
                # CAS commits we first promote the child to `committed`, then mark the
                # parent compacted. That way discovery paths can ignore staged children
                # while exact UUID loads still work for an already-relabeled client.
                await _retry_compaction_search_write(
                    f"mark child partition {swapped_partition.partition_uuid} committed",
                    lambda: self.search_client.update_partition(
                        swapped_partition.partition_uuid,
                        compaction_state=COMPACTION_STATE_COMMITTED,
                        compaction_attempt_uuid=compaction_attempt_uuid,
                    ),
                )
                await _retry_compaction_search_write(
                    f"mark parent partition {snapshot.partition_uuid} compacted",
                    lambda: self.search_client.update_partition(
                        snapshot.partition_uuid,
                        compaction_attempt_uuid=compaction_attempt_uuid,
                        is_compacted=True,
                        summary=summary,
                        boundary_hash=boundary_hash,
                        boundary_message_count=boundary_message_count,
                        boundary_user_count=boundary_user_count,
                        tail_hash=tail_hash,
                    ),
                )
        except Exception:
            logger.exception("compact_partition failed for partition %s", snapshot.partition_uuid)
        finally:
            await self.redis_client.delete(lock_key)

    async def get_compaction_status(
        self,
        request: Request,
        conversation_uuid: str,
        pending_partition_uuid: str,
    ):
        await load_session(request)
        if not request.session:
            raise HTTPException(status_code=400, detail="Session expired")
        lock_exists = await self.redis_client.exists(f"compaction_lock:{conversation_uuid}")
        if lock_exists:
            return {"done": False}
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
            response = {"done": True}
            if (
                partition.partition_uuid != pending_partition_uuid
                and partition.parent_partition_uuid == pending_partition_uuid
            ):
                response["partition_uuid"] = partition.partition_uuid
            return response
        return {"done": True}


_COMPACTION_SEARCH_WRITE_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2)


async def _retry_compaction_search_write(
    operation_name: str,
    action: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    total_attempts = len(_COMPACTION_SEARCH_WRITE_RETRY_DELAYS_SECONDS) + 1
    for attempt_idx in range(total_attempts):
        try:
            await action()
            return
        except Exception:
            if attempt_idx == total_attempts - 1:
                raise
            delay = _COMPACTION_SEARCH_WRITE_RETRY_DELAYS_SECONDS[attempt_idx]
            logger.warning(
                "Compaction %s failed on attempt %d/%d; retrying in %.2fs",
                operation_name,
                attempt_idx + 1,
                total_attempts,
                delay,
                exc_info=True,
            )
            await asyncio.sleep(delay)
