"""HTTP /compaction-status handler: response-shape per Redis state."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from prokaryotes.web_v1.compaction import CompactionStatusHandler
from tests.unit_tests._fakes import FakeRedis


class StubHandler(CompactionStatusHandler):
    """Concrete handler wired to a FakeRedis for assertion."""

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis

    @property
    def redis_client(self) -> FakeRedis:  # type: ignore[override]
        return self._redis


def _make_request_with_session() -> MagicMock:
    """Fake `Request` for the handler. `load_session` is mocked so the test doesn't need starsessions
    middleware to populate state."""
    req = MagicMock()
    req.session = {"user_id": "u1"}
    return req


CONVO = "conv-1"
PENDING = "pending-snap-1"


@pytest.fixture(autouse=True)
def _stub_load_session(monkeypatch):
    monkeypatch.setattr("prokaryotes.web_v1.compaction.load_session", AsyncMock())


@pytest.mark.asyncio
async def test_lock_present_returns_pending():
    redis = FakeRedis()
    await redis.set(f"compaction_lock:{CONVO}", "1")
    handler = StubHandler(redis)

    response = await handler.get_compaction_status(_make_request_with_session(), CONVO, PENDING)

    assert response.done is False
    assert response.snapshot_uuid is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(None, id="no_key_or_long_idle_past_ttl"),
        pytest.param("", id="empty_sentinel"),
    ],
)
async def test_done_without_snapshot_uuid(seed):
    """Lock released and no relabel target — same response shape whether the key was never written, was
    written as the empty sentinel, or has TTL-elapsed away."""
    redis = FakeRedis()
    if seed is not None:
        await redis.set(f"compaction_status:{PENDING}", seed)
    handler = StubHandler(redis)

    response = await handler.get_compaction_status(_make_request_with_session(), CONVO, PENDING)

    assert response.done is True
    assert response.snapshot_uuid is None


@pytest.mark.asyncio
async def test_relabel_target_returns_done_with_snapshot_uuid():
    """compaction_status holds the committed child's snapshot_uuid — relabel target."""
    redis = FakeRedis()
    await redis.set(f"compaction_status:{PENDING}", "child-snap-1")
    handler = StubHandler(redis)

    response = await handler.get_compaction_status(_make_request_with_session(), CONVO, PENDING)

    assert response.done is True
    assert response.snapshot_uuid == "child-snap-1"


@pytest.mark.asyncio
async def test_missing_session_raises_400():
    redis = FakeRedis()
    handler = StubHandler(redis)
    req = MagicMock()
    req.session = {}

    with pytest.raises(HTTPException) as excinfo:
        await handler.get_compaction_status(req, CONVO, PENDING)

    assert excinfo.value.status_code == 400
