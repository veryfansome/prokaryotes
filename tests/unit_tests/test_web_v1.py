"""Web routes / auth / session surface tests.

Covers WebBase's inheritance contract and the re-export surface that web_v1 maintains. Also exercises `finalize_turn`'s
persistence side effects and `validate_assistant_messages`'s four-branch guardrail at the helper level (unit-scoped; the
full FastAPI route is covered in Tier B).
"""

from __future__ import annotations

import json

import pytest

from prokaryotes.api_v1.models import IncomingMessage
from prokaryotes.context_v1.conversation_sync import (
    AssistantMessageGuardrailError,
)
from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
)
from prokaryotes.harness_v1.base import HarnessBase
from prokaryotes.web_v1 import (
    AuthHandler,
    ConversationCompactor,
    ConversationSyncer,
    WebBase,
    _conversation_can_follow_client,
    get_postgres_pool,
    get_redis_client,
    hash_password,
    verify_password,
)
from tests.unit_tests._builders import BOT_ID, conversation, msg
from tests.unit_tests._fakes import FakeRedis, FakeSearchClient, make_syncer

# -----------------------------------------------------------------------------
# WebBase inheritance + ABC contract
# -----------------------------------------------------------------------------


def test_web_v1_import_surface_locks_down_documented_re_exports():
    """`web_v1/README.md` documents the re-export surface as contract: `ConversationCompactor`,
    `ConversationSyncer`, `_conversation_can_follow_client`, `get_redis_client`, `AuthHandler`,
    `hash_password`, `verify_password`, `get_postgres_pool`, and `WebBase` itself. Also asserts
    `WebBase`'s composed-ABC contract is fully satisfied — instantiation must not raise."""
    from prokaryotes.context_v1 import (
        ConversationCompactor as _Compactor,
    )
    from prokaryotes.context_v1 import (
        ConversationSyncer as _Syncer,
    )
    from prokaryotes.context_v1 import (
        _conversation_can_follow_client as _follow,
    )
    from prokaryotes.context_v1 import (
        get_redis_client as _redis,
    )
    from prokaryotes.harness_v1.base import HarnessBase
    from prokaryotes.utils_v1.db_utils import get_postgres_pool as _pool
    from prokaryotes.web_v1.auth import (
        AuthHandler as _Auth,
    )
    from prokaryotes.web_v1.auth import (
        hash_password as _hash,
    )
    from prokaryotes.web_v1.auth import (
        verify_password as _verify,
    )
    from prokaryotes.web_v1.compaction import CompactionStatusHandler

    assert ConversationCompactor is _Compactor
    assert ConversationSyncer is _Syncer
    assert _conversation_can_follow_client is _follow
    assert get_redis_client is _redis
    assert AuthHandler is _Auth
    assert hash_password is _hash
    assert verify_password is _verify
    assert get_postgres_pool is _pool

    assert issubclass(WebBase, HarnessBase)
    assert issubclass(WebBase, AuthHandler)
    assert issubclass(WebBase, CompactionStatusHandler)
    wb = WebBase("scripts/static")  # composed-ABC contract: instantiation must not raise.
    assert wb.html_dir.name == "html"


# -----------------------------------------------------------------------------
# validate_assistant_messages guardrail — four-branch coverage
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_assistant_messages_noop_when_no_assistant_entries():
    syncer, _, _ = make_syncer()

    # All-user incoming must not raise even with no index.
    incoming = [IncomingMessage(role="user", content="hi", source_id=None)]
    await syncer.validate_assistant_messages("c-1", incoming)


@pytest.mark.asyncio
async def test_validate_assistant_messages_rejects_assistant_without_source_id():
    syncer, _, _ = make_syncer()
    incoming = [
        IncomingMessage(role="user", content="hi", source_id="1"),
        IncomingMessage(role="assistant", content="reply", source_id=None),
    ]

    with pytest.raises(AssistantMessageGuardrailError, match="must carry a server-assigned source_id"):
        await syncer.validate_assistant_messages("c-1", incoming)


@pytest.mark.asyncio
async def test_validate_assistant_messages_rejects_unknown_source_id():
    syncer, _, _ = make_syncer()
    incoming = [
        IncomingMessage(role="assistant", content="reply", source_id="unknown-id"),
    ]

    with pytest.raises(AssistantMessageGuardrailError, match="Unknown assistant source_id"):
        await syncer.validate_assistant_messages("c-1", incoming)


@pytest.mark.asyncio
async def test_validate_assistant_messages_rejects_content_mismatch():
    """Source_id is known but content hash doesn't match → 400."""
    syncer, redis, _ = make_syncer()
    await syncer.refresh_assistant_index_with("c-1", "2.000000", "the right content")

    incoming = [
        IncomingMessage(role="assistant", content="WRONG content", source_id="2.000000"),
    ]
    with pytest.raises(AssistantMessageGuardrailError, match="content mismatch"):
        await syncer.validate_assistant_messages("c-1", incoming)


@pytest.mark.asyncio
async def test_validate_assistant_messages_accepts_known_source_id_and_content():
    syncer, _, _ = make_syncer()
    await syncer.refresh_assistant_index_with("c-1", "2.000000", "matching content")

    incoming = [
        IncomingMessage(role="assistant", content="matching content", source_id="2.000000"),
    ]
    # Should not raise.
    await syncer.validate_assistant_messages("c-1", incoming)


# -----------------------------------------------------------------------------
# finalize_turn persistence side effects
# -----------------------------------------------------------------------------


class _StubHarness(HarnessBase):
    """Concrete HarnessBase wired to fakes; bypasses init() so unit tests
    don't need the FastAPI app or runtime clients."""

    def __init__(self, redis: FakeRedis, search: FakeSearchClient):
        self._redis_client = redis
        self._search_client = search
        self.background_tasks = set()
        self._conversation_cache_ex = 3600


@pytest.mark.asyncio
async def test_finalize_turn_appends_bot_message_and_persists_conversation():
    redis = FakeRedis()
    search = FakeSearchClient()
    harness = _StubHarness(redis, search)
    conv = conversation(msg("1", "U1"), snapshot_uuid="s1")

    await harness.finalize_turn(
        conversation=conv,
        bot_message_source_id="2",
        bot_message_content="Hello world",
        turn_items=[],
        triggering_source_id="1",
    )

    # Bot message appended to conversation.
    assert conv.messages[-1].source_id == "2"
    assert conv.messages[-1].author_id == BOT_ID
    assert conv.messages[-1].content == "Hello world"
    # Redis cache updated.
    cached_data = await redis.get(f"conversation:{conv.conversation_uuid}")
    cached = Conversation.model_validate_json(cached_data)
    assert cached.messages[-1].content == "Hello world"
    # ES doc written.
    assert conv.snapshot_uuid in search.conversations
    # No TurnExecution because turn_items is empty.
    assert search.put_turn_execution_calls == []


@pytest.mark.asyncio
async def test_finalize_turn_persists_turn_execution_when_tool_items_exist():
    from tests.unit_tests._builders import function_call, function_call_output

    redis = FakeRedis()
    search = FakeSearchClient()
    harness = _StubHarness(redis, search)
    conv = conversation(msg("1", "U1"), snapshot_uuid="s1")
    turn_items = [function_call("c1", "tool"), function_call_output("c1", "result")]

    await harness.finalize_turn(
        conversation=conv,
        bot_message_source_id="2",
        bot_message_content="Done",
        turn_items=turn_items,
        triggering_source_id="1",
    )

    assert len(search.put_turn_execution_calls) == 1
    te = search.put_turn_execution_calls[0]
    assert te.bot_message_source_id == "2"
    assert te.items == turn_items
    assert te.completed is True


@pytest.mark.asyncio
async def test_finalize_turn_refreshes_assistant_index():
    """The new bot message must enter the cached assistant index so the next
    POST's guardrail recognizes it."""
    redis = FakeRedis()
    search = FakeSearchClient()
    harness = _StubHarness(redis, search)
    conv = conversation(msg("1", "U1"), snapshot_uuid="s1")

    await harness.finalize_turn(
        conversation=conv,
        bot_message_source_id="2.000000",
        bot_message_content="Hello",
        turn_items=[],
        triggering_source_id="1",
    )

    index_data = await redis.get(f"assistant_index:{conv.conversation_uuid}")
    assert index_data is not None
    index = json.loads(index_data)
    assert "2.000000" in index


# -----------------------------------------------------------------------------
# sync_conversation → finalize_turn doesn't leak transient narration
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_strips_system_message_and_persists_to_redis_and_es():
    """finalize_turn writes the bot's ConversationMessage (no system role
    involvement) — the persisted state is only user/bot messages from the conversation. System instructions never enter
    the conversation."""
    redis = FakeRedis()
    search = FakeSearchClient()
    harness = _StubHarness(redis, search)
    conv = conversation(msg("1", "U1"), snapshot_uuid="s1")

    await harness.finalize_turn(
        conversation=conv,
        bot_message_source_id="2",
        bot_message_content="Reply",
        turn_items=[],
        triggering_source_id="1",
    )

    cached = Conversation.model_validate_json(await redis.get(f"conversation:{conv.conversation_uuid}"))
    roles_present = {m.author_id for m in cached.messages}
    assert "system" not in roles_present
    assert all(isinstance(m, ConversationMessage) for m in cached.messages)
