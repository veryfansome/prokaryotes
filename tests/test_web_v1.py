from datetime import (
    UTC,
    datetime,
)
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.models_v1 import (
    ChatMessage,
    FactDoc,
)
from prokaryotes.web_v1 import ProkaryoteV1


@pytest.mark.parametrize("list1, list2, divergence_idx, is_mismatch, list1_is_longer", [
    (
        # Cached conversation matches payload
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        None,
        False,
        False,
    ),
    (
        # Payload retries previous prompt
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        3,
        False,
        True,
    ),
    (
        # Payload modifies previous prompt
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Nice to meet you"),
        ],
        2,
        True,
        True,
    ),
    (
        # Cached conversation is behind payload
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
            ChatMessage(role="user", content="Do one more"),
        ],
        3,
        False,
        False,
    ),
    (
        # Cached conversation is behind and payload has diverged
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Nice to meet you"),
            ChatMessage(role="assitant", content="Nice to meet you too - what's on your mind?"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        2,
        True,
        False,
    ),
])
def test_find_context_divergence(
        list1: list[ChatMessage],
        list2: list[ChatMessage],
        divergence_idx: int,
        is_mismatch: bool,
        list1_is_longer: bool,
):
    _divergence_idx, _is_mismatch, _list1_is_longer = ProkaryoteV1.find_context_divergence(list1, list2)
    assert _divergence_idx == divergence_idx
    assert _is_mismatch == is_mismatch
    assert _list1_is_longer == list1_is_longer


def test_find_named_entities_in_facts_matches_entities_per_fact():
    fact_1 = FactDoc(
        about=["user:1"],
        created_at=datetime(2026, 4, 7, tzinfo=UTC),
        text="The user reads Nathan Hale’s Hazardous Tales with their daughter.",
    )
    fact_2 = FactDoc(
        about=["user:1"],
        created_at=datetime(2026, 4, 7, tzinfo=UTC),
        text="They also discuss parent-child reading every weekend.",
    )

    matched = ProkaryoteV1.find_named_entities_in_facts(
        facts=[
            fact_1,
            fact_2,
        ],
        named_entities=[
            "Nathan Hale's Hazardous Tales",
            "parent–child reading",
            "The Hobbit",
        ],
    )

    assert len(matched) == 2
    assert matched[0] == (fact_1, ["Nathan Hale's Hazardous Tales"])
    assert matched[1] == (fact_2, ["parent–child reading"])


@pytest.mark.asyncio
async def test_get_named_entities_embs_normalizes_and_dedupes(monkeypatch: pytest.MonkeyPatch):
    class DummyNamedEntityObserver:
        async def get_named_entities(self) -> list[str]:
            return [
                "U.S.",
                "US",
                " Nathan   Hale’s Hazardous   Tales ",
                "Nathan Hale's Hazardous Tales",
                " ",
            ]

    emb_mock = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    monkeypatch.setattr("prokaryotes.web_v1.get_document_embs", emb_mock)

    named_entities, named_entity_embs = await ProkaryoteV1.get_named_entities_embs(DummyNamedEntityObserver())

    assert named_entities == ["U.S.", "US", "Nathan Hale's Hazardous Tales"]
    assert named_entity_embs == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    emb_mock.assert_awaited_once_with(("U.S.", "US", "Nathan Hale's Hazardous Tales"))


@pytest.mark.asyncio
async def test_get_similar_topic_pairs_filters_to_extremely_similar_matches():
    web = ProkaryoteV1.__new__(ProkaryoteV1)
    web.search_client = AsyncMock()
    web.search_client.search_topics = AsyncMock(return_value=[  # type: ignore[method-assign]
        "Nathan Hale's Hazardous Tales",
        "Nathan Hale Hazardous Tales",
    ])

    similar_topic_pairs = await web.get_similar_topic_pairs(
        topics=["Nathan Hale's Hazardous Tales"],
        topic_embs=[[0.1, 0.2]],
        min_score=0.95,
    )

    assert similar_topic_pairs == [("Nathan Hale Hazardous Tales", "Nathan Hale's Hazardous Tales")]
    web.search_client.search_topics.assert_awaited_once()
    kwargs = web.search_client.search_topics.await_args.kwargs
    assert kwargs["match"] == "Nathan Hale's Hazardous Tales"
    assert kwargs["match_emb"] == [0.1, 0.2]
    assert kwargs["min_lexical_score"] == 0.95


@pytest.mark.asyncio
async def test_get_topic_embs_excludes_named_entities_before_embedding(monkeypatch: pytest.MonkeyPatch):
    class DummyTopicObserver:
        async def get_topics(self) -> list[str]:
            return [
                "Nathan Hale’s Hazardous Tales",
                "STEM comics",
                "US history",
            ]

    emb_mock = AsyncMock(return_value=[[0.2, 0.2], [0.3, 0.3]])
    monkeypatch.setattr("prokaryotes.web_v1.get_document_embs", emb_mock)

    topics, topic_embs = await ProkaryoteV1.get_topic_embs(
        DummyTopicObserver(),
        excluded_topics={
            "Nathan Hale's Hazardous Tales",
            "US",
        },
    )

    assert topics == ["STEM comics", "US history"]
    assert topic_embs == [[0.2, 0.2], [0.3, 0.3]]
    emb_mock.assert_awaited_once_with(("STEM comics", "US history"))


@pytest.mark.asyncio
async def test_get_topic_embs_normalizes_and_dedupes(monkeypatch: pytest.MonkeyPatch):
    class DummyTopicObserver:
        async def get_topics(self) -> list[str]:
            return [
                "Nathan Hale’s Hazardous Tales",
                "Nathan Hale's Hazardous Tales",
                "  STEM   comics  ",
                " ",
            ]

    emb_mock = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr("prokaryotes.web_v1.get_document_embs", emb_mock)

    topics, topic_embs = await ProkaryoteV1.get_topic_embs(DummyTopicObserver())

    assert topics == ["Nathan Hale's Hazardous Tales", "STEM comics"]
    assert topic_embs == [[0.1, 0.2], [0.3, 0.4]]
    emb_mock.assert_awaited_once_with(("Nathan Hale's Hazardous Tales", "STEM comics"))


@pytest.mark.parametrize("cached_context_window, payload_messages, context_window_to_stream", [
    # Payload progresses the conversation
    (
        [
            ChatMessage(role="user", content="Hi"),
            FunctionCallOutput(
                call_id="123",
                output="... user preferences ...",
                type="function_call_output"
            ),
            ChatMessage(role="assitant", content="How can I help you"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            FunctionCallOutput(
                call_id="123",
                output="... user preferences ...",
                type="function_call_output"
            ),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
    ),
    # Payload retries previous prompt
    (
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            FunctionCallOutput(
                call_id="123",
                output="... joke candidates ...",
                type="function_call_output"
            ),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
        ],
        # Retains function output
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            FunctionCallOutput(
                call_id="123",
                output="... joke candidates ...",
                type="function_call_output"
            ),
        ],
    ),
    # Payload modifies previous prompt
    (
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Got any good jokes?"),
            FunctionCallOutput(
                call_id="123",
                output="... joke candidates ...",
                type="function_call_output"
            ),
            ChatMessage(role="assitant", content="Why don't scientists like atoms? Because they make up everything!"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Nice to meet you"),
        ],
        # Drops function output
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
            ChatMessage(role="user", content="Nice to meet you"),
        ],
    ),
])
def test_sync_context_window(cached_context_window, payload_messages, context_window_to_stream):
    assert ProkaryoteV1.sync_context_window(cached_context_window, payload_messages) == context_window_to_stream


@pytest.mark.parametrize("cached_context_window, payload_messages", [
    (
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
        ],
        [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assitant", content="How can I help you"),
        ],
    ),
])
def test_sync_context_window_exception(cached_context_window, payload_messages):
    with pytest.raises(HTTPException, match="Payload does not alter conversation state"):
        ProkaryoteV1.sync_context_window(cached_context_window, payload_messages)
