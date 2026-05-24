"""`TopicSearcher` indexing + search against an in-memory fake Elasticsearch.

The fake intercepts the `helpers.async_bulk` import inside `prokaryotes.search_v1.topics` so indexing writes
land in `FakeES._docs`. `FakeES.search` honors `query.bool.must_not` (excluded topics), `query.bool.filter`
(`is_named_entity`), and `knn.filter.bool.*` the same way real ES would, returning the matching docs as
`{"hits": {"hits": [...]}}`. The tests assert the externally visible contracts: indexing dedupes by normalized
identity, search filters and dedupes its results, and an empty match short-circuits.
"""

from __future__ import annotations

import pytest

from prokaryotes.search_v1.topics import TopicSearcher


class FakeES:
    """Minimal in-memory ES stand-in. `_docs` is keyed by the `_id` `index_topics` derives via `text_to_md5`.

    `search` independently applies the bool-query filter (the lexical path) and the `knn.filter.bool` filter
    (the vector path), returning a doc if it would be a hit on *either* path. This mirrors how real ES
    combines the two paths: a doc that passes only one side's filter still shows up if that side scored it.
    Critically, this means a regression that drops `excluded_topics` or `is_named_entity` from the KNN-side
    filter while leaving the bool-side intact will produce extra hits and break the assertions below.
    """

    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}
        self.search_calls: list[dict] = []

    def index(self, doc_id: str, source: dict) -> None:
        """Idempotent create — second write with the same id is a no-op (matches ES `_op_type: "create"`)."""
        self._docs.setdefault(doc_id, source)

    async def search(self, **kwargs) -> dict:
        self.search_calls.append(kwargs)
        lex_excluded, lex_ne = _filter_constraints(kwargs["query"]["bool"])
        knn_clause = kwargs.get("knn")
        if knn_clause is None:
            knn_excluded, knn_ne, knn_active = set(), None, False
        else:
            knn_excluded, knn_ne = _filter_constraints(knn_clause.get("filter", {}).get("bool", {}))
            knn_active = True
        hits = []
        for doc in self._docs.values():
            passes_lex = doc["name"] not in lex_excluded and (lex_ne is None or doc["is_named_entity"] == lex_ne)
            passes_knn = knn_active and (
                doc["name"] not in knn_excluded and (knn_ne is None or doc["is_named_entity"] == knn_ne)
            )
            if passes_lex or passes_knn:
                hits.append({"_score": 1.0, "_source": dict(doc)})
        return {"hits": {"hits": hits}}


class FakeTopicSearcher(TopicSearcher):
    def __init__(self, es: FakeES) -> None:
        self._es = es

    @property
    def es(self) -> FakeES:  # type: ignore[override]
        return self._es


def _filter_constraints(bool_clause: dict) -> tuple[set[str], bool | None]:
    """Parse a `{"must_not": [...], "filter": [...]}` block into (excluded_names, is_named_entity)."""
    excluded: set[str] = set()
    for clause in bool_clause.get("must_not", []):
        excluded.update(clause.get("terms", {}).get("name.keyword", []))
    ne: bool | None = None
    for clause in bool_clause.get("filter", []):
        candidate = clause.get("term", {}).get("is_named_entity")
        if candidate is not None:
            ne = candidate
    return excluded, ne


@pytest.fixture
def fake_es(monkeypatch: pytest.MonkeyPatch) -> FakeES:
    """Returns a FakeES with `helpers.async_bulk` patched to write into it."""
    es = FakeES()

    async def _fake_async_bulk(target_es, actions, raise_on_error=False):
        assert target_es is es
        assert raise_on_error is False
        for action in actions:
            assert action["_op_type"] == "create"
            assert action["_index"] == "topics"
            es.index(action["_id"], action["_source"])
        return len(actions), []

    monkeypatch.setattr("prokaryotes.search_v1.topics.helpers.async_bulk", _fake_async_bulk)
    return es


@pytest.mark.asyncio
async def test_index_then_search_round_trips_topic_name(fake_es: FakeES):
    searcher = FakeTopicSearcher(fake_es)
    await searcher.index_topics(topics=["STEM comics"], topic_embs=[[0.1, 0.2]])

    topics = await searcher.search_topics(match="STEM comics", match_emb=[0.1, 0.2], min_lexical_score=0.0)

    assert topics == ["STEM comics"]


@pytest.mark.asyncio
async def test_index_normalizes_curly_quotes_and_dedupes_extra_whitespace(fake_es: FakeES):
    """Smart quotes and whitespace collapse to the same identity, so an indexed pair lands as one document."""
    searcher = FakeTopicSearcher(fake_es)
    await searcher.index_topics(
        topics=[
            "Nathan Hale’s Hazardous Tales",
            "Nathan Hale's Hazardous Tales",
            "  STEM   comics  ",
            "",
        ],
        topic_embs=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
    )

    topics = await searcher.search_topics(match="anything", match_emb=[0.0, 0.0], min_lexical_score=0.0)

    assert sorted(topics) == ["Nathan Hale's Hazardous Tales", "STEM comics"]


@pytest.mark.asyncio
async def test_search_with_empty_match_short_circuits(fake_es: FakeES):
    searcher = FakeTopicSearcher(fake_es)
    await searcher.index_topics(topics=["STEM comics"], topic_embs=[[0.1, 0.2]])

    topics = await searcher.search_topics(match="", match_emb=[0.1, 0.2])

    assert topics == []
    assert fake_es.search_calls == []  # ES is never contacted for an empty match


@pytest.mark.asyncio
async def test_search_excludes_seed_topics(fake_es: FakeES):
    searcher = FakeTopicSearcher(fake_es)
    await searcher.index_topics(
        topics=["Seed Topic", "Other Topic"],
        topic_embs=[[0.1, 0.2], [0.3, 0.4]],
    )

    topics = await searcher.search_topics(
        match="reading",
        match_emb=[0.1, 0.2],
        excluded_topics=["Seed Topic", "Seed Topic", ""],
        min_lexical_score=0.0,
    )

    assert "Seed Topic" not in topics
    assert "Other Topic" in topics


@pytest.mark.asyncio
async def test_search_filters_by_named_entity_flag(fake_es: FakeES):
    searcher = FakeTopicSearcher(fake_es)
    await searcher.index_topics(topics=["New York Times"], topic_embs=[[0.1, 0.2]], is_named_entity=True)
    await searcher.index_topics(topics=["reading"], topic_embs=[[0.3, 0.4]], is_named_entity=False)

    nes = await searcher.search_topics(
        match="anything",
        match_emb=[0.0, 0.0],
        is_named_entity=True,
        min_lexical_score=0.0,
    )
    concepts = await searcher.search_topics(
        match="anything",
        match_emb=[0.0, 0.0],
        is_named_entity=False,
        min_lexical_score=0.0,
    )

    assert nes == ["New York Times"]
    assert concepts == ["reading"]


@pytest.mark.asyncio
async def test_search_threads_match_text_and_min_score_into_es_query(fake_es: FakeES):
    """The `match` argument flows into both an exact-match `term` on `name.keyword` and a fuzzy `match`
    on `name`; `min_lexical_score` flows into ES `min_score`. These wires connect the public inputs to the
    ES query shape — without them, production lexical search silently no-ops. The fake's filter modeling
    can't catch this because every passing doc scores 1.0; this test pins the contract directly.
    """
    searcher = FakeTopicSearcher(fake_es)
    await searcher.search_topics(match="Nathan Hale", match_emb=None, min_lexical_score=0.5)

    [kwargs] = fake_es.search_calls
    should_clauses = kwargs["query"]["bool"]["should"]
    assert any(c.get("term", {}).get("name.keyword", {}).get("value") == "Nathan Hale" for c in should_clauses)
    assert any(c.get("match", {}).get("name", {}).get("query") == "Nathan Hale" for c in should_clauses)
    assert kwargs["min_score"] == 0.5


@pytest.mark.asyncio
async def test_search_dedupes_repeated_hits_on_same_name(fake_es: FakeES):
    """When ES returns the same name in multiple hits (e.g. lexical + knn paths both score it), the response
    is deduped down to a single entry, preserving the first occurrence."""
    searcher = FakeTopicSearcher(fake_es)
    # Pre-populate the fake directly with two distinct docs sharing the same `name`.
    fake_es.index("id-a", {"name": "Nathan Hale's Hazardous Tales", "emb": [0.1, 0.2], "is_named_entity": True})
    fake_es.index("id-b", {"name": "Nathan Hale's Hazardous Tales", "emb": [0.3, 0.4], "is_named_entity": True})

    topics = await searcher.search_topics(
        match="Nathan Hale's Hazardous Tales",
        match_emb=None,
        min_lexical_score=0.0,
    )

    assert topics == ["Nathan Hale's Hazardous Tales"]
