from unittest.mock import AsyncMock, patch

import pytest

from prokaryotes.api_v1.models import ContextPartition
from tests.context_partition_utils import make_web_base


class MockRequest:
    def __init__(self, session=None):
        self.session = session if session is not None else {"user": "test"}


@pytest.mark.asyncio
async def test_get_compaction_status_lock_present():
    wb = make_web_base(redis_data={"compaction_lock:conv-1": "1"})
    with patch("prokaryotes.web_v1.load_session", new_callable=AsyncMock):
        result = await wb.get_compaction_status(MockRequest(), "conv-1", "old-uuid")
    assert result == {"done": False}


@pytest.mark.asyncio
async def test_get_compaction_status_partition_evicted():
    wb = make_web_base()
    with patch("prokaryotes.web_v1.load_session", new_callable=AsyncMock):
        result = await wb.get_compaction_status(MockRequest(), "conv-1", "old-uuid")
    assert result == {"done": True}


@pytest.mark.asyncio
async def test_get_compaction_status_partition_changed():
    partition = ContextPartition(
        conversation_uuid="conv-1",
        partition_uuid="new-uuid",
        items=[],
    )
    wb = make_web_base(redis_data={"context_partition:conv-1": partition.model_dump_json()})
    with patch("prokaryotes.web_v1.load_session", new_callable=AsyncMock):
        result = await wb.get_compaction_status(MockRequest(), "conv-1", "old-uuid")
    assert result == {"done": True}


@pytest.mark.asyncio
async def test_get_compaction_status_partition_unchanged():
    partition = ContextPartition(
        conversation_uuid="conv-1",
        partition_uuid="old-uuid",
        items=[],
    )
    wb = make_web_base(redis_data={"context_partition:conv-1": partition.model_dump_json()})
    with patch("prokaryotes.web_v1.load_session", new_callable=AsyncMock):
        result = await wb.get_compaction_status(MockRequest(), "conv-1", "old-uuid")
    assert result == {"done": False}
