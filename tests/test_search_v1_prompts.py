from datetime import (
    UTC,
    datetime,
)
from unittest.mock import AsyncMock

import pytest

from prokaryotes.models_v1 import ChatMessage
from prokaryotes.search_v1.prompts import PromptSearcher


class DummyPromptSearcher(PromptSearcher):
    def __init__(self, es: AsyncMock):
        self._es = es

    @property
    def es(self) -> AsyncMock:
        return self._es


@pytest.fixture
def es_mock() -> AsyncMock:
    es = AsyncMock()
    es.search = AsyncMock()
    es.index = AsyncMock()
    return es


@pytest.mark.asyncio
async def test_get_last_prompt_returns_latest(es_mock: AsyncMock):
    searcher = DummyPromptSearcher(es_mock)
    created_at = datetime(2026, 4, 10, tzinfo=UTC)
    es_mock.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "prompt-123",
                    "_source": {
                        "created_at": created_at,
                        "labels": ["conversation:abc", "user:1"],
                        "messages": [
                            {"role": "user", "content": "What should we read next?", "type": "message"},
                        ],
                        "named_entities": ["Donner Dinner Party"],
                        "summary": "The user wants to know what to read next.",
                        "summary_emb": [0.0, 0.1, 0.2],
                        "topics": ["historical comics", "reading"],
                    },
                },
            ]
        }
    }

    prompt = await searcher.get_last_prompt("abc")

    assert prompt is not None
    assert prompt.doc_id == "prompt-123"
    assert prompt.topics == ["historical comics", "reading"]
    assert prompt.named_entities == ["Donner Dinner Party"]
    kwargs = es_mock.search.await_args.kwargs
    assert kwargs["query"]["bool"]["filter"][0]["term"]["labels"] == "conversation:abc"
    assert kwargs["size"] == 1


@pytest.mark.asyncio
async def test_index_prompt_persists_topics_and_named_entities(es_mock: AsyncMock):
    searcher = DummyPromptSearcher(es_mock)
    result = await searcher.index_prompt(
        labels=["conversation:xyz", "user:1"],
        messages=[ChatMessage(role="user", content="we read one dead spy")],
        named_entities=["Donner Dinner Party"],
        prompt_uuid="prompt-xyz",
        summary="The user read One Dead Spy.",
        summary_emb=[0.0, 0.1, 0.2],
        topics=["historical comics", "reading"],
    )

    assert result is not None
    assert result.topics == ["historical comics", "reading"]
    assert result.named_entities == ["Donner Dinner Party"]
    kwargs = es_mock.index.await_args.kwargs
    assert kwargs["index"] == "prompts"
    assert kwargs["id"] == "prompt-xyz"
    assert kwargs["document"]["topics"] == ["historical comics", "reading"]
    assert kwargs["document"]["named_entities"] == ["Donner Dinner Party"]
