from unittest.mock import AsyncMock

import pytest

from prokaryotes.search_v1.named_entities import NamedEntitySearcher
from prokaryotes.utils_v1.text_utils import text_to_md5


class DummyNamedEntitySearcher(NamedEntitySearcher):
    def __init__(self, es: AsyncMock):
        self._es = es

    @property
    def es(self) -> AsyncMock:
        return self._es


@pytest.mark.asyncio
async def test_index_named_entities_normalizes_and_dedupes(monkeypatch: pytest.MonkeyPatch, es_mock: AsyncMock):
    searcher = DummyNamedEntitySearcher(es_mock)
    captured_actions = []

    async def _fake_async_bulk(_es, actions, raise_on_error=False):
        assert _es is es_mock
        assert raise_on_error is False
        captured_actions.extend(actions)
        return len(actions), []

    monkeypatch.setattr("prokaryotes.search_v1.named_entities.helpers.async_bulk", _fake_async_bulk)

    await searcher.index_named_entities(
        named_entities=[
            "Nathan Hale’s Hazardous Tales",
            "Nathan Hale's Hazardous Tales",
            "  New   York Times  ",
            "",
        ],
        named_entity_embs=[
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
        ],
    )

    assert len(captured_actions) == 2
    assert captured_actions[0]["_id"] == text_to_md5("Nathan Hale's Hazardous Tales")
    assert captured_actions[0]["_source"]["name"] == "Nathan Hale's Hazardous Tales"
    assert captured_actions[0]["_source"]["emb"] == [0.1, 0.2]
    assert captured_actions[1]["_id"] == text_to_md5("New York Times")
    assert captured_actions[1]["_source"]["name"] == "New York Times"
    assert captured_actions[1]["_source"]["emb"] == [0.5, 0.6]


@pytest.mark.asyncio
async def test_search_named_entities_excludes_seed_entities_from_lexical_and_knn(es_mock: AsyncMock):
    searcher = DummyNamedEntitySearcher(es_mock)
    es_mock.search.return_value = {"hits": {"hits": []}}

    await searcher.search_named_entities(
        match="reading",
        match_emb=[0.1, 0.2],
        excluded_entities=["Seed Entity", "Seed Entity", ""],
        min_score=0.0,
    )

    kwargs = es_mock.search.await_args.kwargs
    assert kwargs["query"]["bool"]["must_not"][0]["terms"]["name.keyword"] == ["Seed Entity"]
    assert kwargs["knn"]["filter"]["bool"]["must_not"][0]["terms"]["name.keyword"] == ["Seed Entity"]


@pytest.mark.asyncio
async def test_search_named_entities_preserves_query_and_dedupes_exact_results(es_mock: AsyncMock):
    searcher = DummyNamedEntitySearcher(es_mock)
    es_mock.search.return_value = {
        "hits": {
            "hits": [
                {"_score": 3.0, "_source": {"name": "Nathan Hale’s Hazardous Tales"}},
                {"_score": 2.0, "_source": {"name": "Nathan Hale’s Hazardous Tales"}},
            ]
        }
    }

    named_entities = await searcher.search_named_entities(
        match="Nathan Hale’s Hazardous Tales",
        match_emb=None,
        min_score=0.5,
    )

    assert named_entities == ["Nathan Hale’s Hazardous Tales"]
    kwargs = es_mock.search.await_args.kwargs
    assert kwargs["index"] == "named-entities"
    assert kwargs["min_score"] == 0.5
    assert kwargs["query"]["bool"]["should"][1]["term"]["name.keyword"]["value"] == "Nathan Hale’s Hazardous Tales"
