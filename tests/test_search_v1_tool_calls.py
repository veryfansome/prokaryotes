from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from prokaryotes.models_v1 import ToolCallDoc
from prokaryotes.search_v1.tool_calls import ToolCallSearcher
from prokaryotes.utils_v1.text_utils import text_to_md5


class DummyToolCallSearcher(ToolCallSearcher):
    def __init__(self, es: AsyncMock):
        self._es = es

    @property
    def es(self) -> AsyncMock:
        return self._es


def _created_at(offset_minutes: int = 0) -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=offset_minutes)


def _source(
    *,
    created_at: datetime | None = None,
    labels: list[str] | None = None,
    output: str = "output",
    output_hash: str | None = None,
    reason: str = "reason",
    search_keywords: list[str] | None = None,
    tool_arguments: str = "ls -la",
    tool_arguments_hash: str | None = None,
    tool_name: str = "run_shell_command",
) -> dict:
    labels = labels or ["conversation:c1"]
    search_keywords = search_keywords or ["file.py"]
    output_hash = output_hash or text_to_md5(output)
    tool_arguments_hash = tool_arguments_hash or text_to_md5(tool_arguments)
    return {
        "created_at": created_at or _created_at(),
        "labels": labels,
        "output": output,
        "output_hash": output_hash,
        "reason": reason,
        "search_keywords": search_keywords,
        "tool_arguments": tool_arguments,
        "tool_arguments_hash": tool_arguments_hash,
        "tool_name": tool_name,
    }


def _hit(
    *,
    doc_id: str,
    score: float,
    source: dict,
    matched_queries: list[str] | None = None,
) -> dict:
    hit = {
        "_id": doc_id,
        "_score": score,
        "_source": source,
    }
    if matched_queries is not None:
        hit["matched_queries"] = matched_queries
    return hit


def _hits_response(hits: list[dict]) -> dict:
    return {"hits": {"hits": hits}}


@pytest.fixture
def es_mock() -> AsyncMock:
    es = AsyncMock()
    es.get = AsyncMock()
    es.index = AsyncMock()
    es.mget = AsyncMock(return_value={"docs": []})
    es.search = AsyncMock()
    return es


@pytest.mark.asyncio
async def test_index_tool_call_persists_hashes_and_document(es_mock: AsyncMock):
    searcher = DummyToolCallSearcher(es_mock)
    doc = await searcher.index_tool_call(
        call_id="call-1",
        labels=["conversation:c1"],
        output="hello",
        prompt_summary="user asked to inspect file",
        prompt_summary_emb=[0.1, 0.2],
        reason="inspect file contents",
        reason_emb=[0.3, 0.4],
        search_keywords=["file.py"],
        tool_arguments="sed -n '1,20p' file.py",
        tool_name="run_shell_command",
        topics=["python", "review"],
    )

    assert doc is not None
    assert doc.output_hash == text_to_md5("hello")
    assert doc.tool_arguments_hash == text_to_md5("sed -n '1,20p' file.py")

    es_mock.index.assert_awaited_once()
    kwargs = es_mock.index.await_args.kwargs
    document = kwargs["document"]
    assert kwargs["index"] == "tool-calls"
    assert kwargs["id"] == "call-1"
    assert document["output_hash"] == doc.output_hash
    assert document["tool_arguments_hash"] == doc.tool_arguments_hash
    assert document["prompt_summary"] == "user asked to inspect file"
    assert document["prompt_summary_emb"] == [0.1, 0.2]
    assert document["reason_emb"] == [0.3, 0.4]
    assert document["topics"] == ["python", "review"]


@pytest.mark.asyncio
async def test_get_tool_call_returns_doc(es_mock: AsyncMock):
    searcher = DummyToolCallSearcher(es_mock)
    source = _source()
    es_mock.get.return_value = {"_id": "call-2", "_source": source}

    doc = await searcher.get_tool_call("call-2")

    assert doc is not None
    assert isinstance(doc, ToolCallDoc)
    assert doc.doc_id == "call-2"
    assert doc.tool_arguments_hash == source["tool_arguments_hash"]


@pytest.mark.asyncio
async def test_search_tool_call_by_arguments_hash_builds_query(es_mock: AsyncMock):
    searcher = DummyToolCallSearcher(es_mock)
    es_mock.search.return_value = _hits_response([])

    await searcher.search_tool_call_by_arguments_hash(
        tool_name="run_shell_command",
        tool_arguments_hash="hash-1",
        excluded_ids=["call-x"],
        labels_and=["conversation:c1"],
        labels_or=["user:1", "user:2"],
        not_labels_and=["archived"],
        not_labels_or=["blocked"],
        size=1,
    )

    es_mock.search.assert_awaited_once()
    kwargs = es_mock.search.await_args.kwargs
    assert kwargs["index"] == "tool-calls"
    assert kwargs["size"] == 1
    assert kwargs["sort"] == [{"created_at": {"order": "desc"}}]
    bool_query = kwargs["query"]["bool"]
    assert {"term": {"tool_name": "run_shell_command"}} in bool_query["filter"]
    assert {"term": {"tool_arguments_hash": "hash-1"}} in bool_query["filter"]
    assert {"term": {"labels": "conversation:c1"}} in bool_query["filter"]
    assert {"terms": {"labels": ["user:1", "user:2"]}} in bool_query["filter"]
    assert {"ids": {"values": ["call-x"]}} in bool_query["must_not"]
    assert {"term": {"labels": "archived"}} in bool_query["must_not"]
    assert {"terms": {"labels": ["blocked"]}} in bool_query["must_not"]


@pytest.mark.asyncio
async def test_search_tool_call_builds_lexical_and_dual_knn_queries(es_mock: AsyncMock):
    searcher = DummyToolCallSearcher(es_mock)
    searcher.search_topics = AsyncMock(return_value=["topic-a"])  # type: ignore[method-assign]
    es_mock.search.side_effect = [
        _hits_response([]),  # lexical
        _hits_response([]),  # prompt_summary knn
        _hits_response([]),  # reason knn
    ]

    results = await searcher.search_tool_call(
        excluded_ids=["already-in-context"],
        match="Take a look at foo.py.",
        match_emb=[0.1, 0.2],
        min_initial_score=1.0,
    )

    assert results == []
    assert es_mock.search.await_count == 3

    lexical_kwargs = es_mock.search.await_args_list[0].kwargs
    assert lexical_kwargs["min_score"] == 1.0
    should = lexical_kwargs["query"]["bool"]["should"]
    assert should[0]["match"]["prompt_summary"]["boost"] == 0.6
    assert should[1]["match"]["reason"]["boost"] == 0.8
    assert should[2]["terms"]["boost"] == 5.0
    assert should[3]["terms"]["boost"] == 0.4
    assert should[2]["terms"]["search_keywords"] == ["Take", "a", "look", "at", "foo.py"]
    assert should[3]["terms"]["topics"] == ["topic-a"]

    prompt_knn_kwargs = es_mock.search.await_args_list[1].kwargs
    reason_knn_kwargs = es_mock.search.await_args_list[2].kwargs
    assert prompt_knn_kwargs["knn"]["field"] == "prompt_summary_emb"
    assert reason_knn_kwargs["knn"]["field"] == "reason_emb"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "score_a, score_b, created_a, created_b, expected_id",
    [
        (4.0, 2.0, _created_at(0), _created_at(10), "call-a"),
        (3.0, 3.0, _created_at(0), _created_at(10), "call-b"),
    ],
)
async def test_search_tool_call_signature_dedupe_keeps_better_or_newer(
    es_mock: AsyncMock,
    score_a: float,
    score_b: float,
    created_a: datetime,
    created_b: datetime,
    expected_id: str,
):
    searcher = DummyToolCallSearcher(es_mock)
    searcher.search_tool_call_by_arguments_hash = AsyncMock(return_value=[])  # type: ignore[method-assign]
    es_mock.search.return_value = _hits_response([
        _hit(
            doc_id="call-a",
            score=score_a,
            source=_source(
                created_at=created_a,
                tool_arguments="same command",
                tool_arguments_hash="hash-same",
                output="result a",
                output_hash="out-a",
            ),
        ),
        _hit(
            doc_id="call-b",
            score=score_b,
            source=_source(
                created_at=created_b,
                tool_arguments="same command",
                tool_arguments_hash="hash-same",
                output="result b",
                output_hash="out-b",
            ),
        ),
    ])

    results = await searcher.search_tool_call(match="inspect file", match_emb=None)

    assert len(results) == 1
    assert results[0].doc_id == expected_id


@pytest.mark.asyncio
async def test_search_tool_call_refreshes_latest_by_arguments_hash_with_size_one(es_mock: AsyncMock):
    searcher = DummyToolCallSearcher(es_mock)
    latest_doc = ToolCallDoc(
        doc_id="call-latest",
        **_source(
            created_at=_created_at(30),
            tool_arguments="cat file.py",
            tool_arguments_hash="hash-cat",
            output="latest output",
            output_hash="out-latest",
        ),
    )
    refresh_mock = AsyncMock(return_value=[latest_doc])
    searcher.search_tool_call_by_arguments_hash = refresh_mock  # type: ignore[method-assign]
    es_mock.search.return_value = _hits_response([
        _hit(
            doc_id="call-old",
            score=3.0,
            source=_source(
                created_at=_created_at(0),
                tool_arguments="cat file.py",
                tool_arguments_hash="hash-cat",
                output="old output",
                output_hash="out-old",
            ),
        ),
    ])

    results = await searcher.search_tool_call(match="cat file.py")

    assert [doc.doc_id for doc in results] == ["call-latest"]
    refresh_mock.assert_awaited_once()
    kwargs = refresh_mock.await_args.kwargs
    assert kwargs["size"] == 1
    assert kwargs["excluded_ids"] is None
    assert kwargs["tool_name"] == "run_shell_command"
    assert kwargs["tool_arguments_hash"] == "hash-cat"


@pytest.mark.asyncio
async def test_search_tool_call_skips_signature_if_latest_call_is_excluded(es_mock: AsyncMock):
    searcher = DummyToolCallSearcher(es_mock)
    es_mock.mget.return_value = {"docs": []}
    latest_doc = ToolCallDoc(
        doc_id="call-latest",
        **_source(
            tool_arguments="cat file.py",
            tool_arguments_hash="hash-cat",
            output="latest output",
            output_hash="out-latest",
        ),
    )
    searcher.search_tool_call_by_arguments_hash = AsyncMock(return_value=[latest_doc])  # type: ignore[method-assign]
    es_mock.search.return_value = _hits_response([
        _hit(
            doc_id="call-old",
            score=3.0,
            source=_source(
                tool_arguments="cat file.py",
                tool_arguments_hash="hash-cat",
                output="old output",
                output_hash="out-old",
            ),
        ),
    ])

    results = await searcher.search_tool_call(
        excluded_ids=["call-latest"],
        match="cat file.py",
    )

    assert results == []


@pytest.mark.asyncio
async def test_search_tool_call_drops_candidate_matching_excluded_argument_hash(es_mock: AsyncMock):
    searcher = DummyToolCallSearcher(es_mock)
    searcher.search_tool_call_by_arguments_hash = AsyncMock(return_value=[])  # type: ignore[method-assign]
    es_mock.mget.return_value = {
        "docs": [
            {
                "found": True,
                "_source": _source(
                    tool_arguments="cat file.py",
                    tool_arguments_hash="hash-cat",
                    output="excluded output",
                    output_hash="excluded-output-hash",
                ),
            }
        ]
    }
    es_mock.search.return_value = _hits_response([
        _hit(
            doc_id="candidate",
            score=3.0,
            source=_source(
                tool_arguments="cat file.py",
                tool_arguments_hash="hash-cat",
                output="candidate output",
                output_hash="candidate-output-hash",
            ),
        ),
    ])

    results = await searcher.search_tool_call(
        excluded_ids=["excluded-call"],
        match="cat file.py",
    )

    assert results == []


@pytest.mark.asyncio
async def test_search_tool_call_drops_candidate_matching_excluded_output_hash(es_mock: AsyncMock):
    searcher = DummyToolCallSearcher(es_mock)
    searcher.search_tool_call_by_arguments_hash = AsyncMock(return_value=[])  # type: ignore[method-assign]
    es_mock.mget.return_value = {
        "docs": [
            {
                "found": True,
                "_source": _source(
                    tool_arguments="cmd excluded",
                    tool_arguments_hash="hash-excluded",
                    output="excluded output",
                    output_hash="same-output-hash",
                ),
            }
        ]
    }
    es_mock.search.return_value = _hits_response([
        _hit(
            doc_id="candidate",
            score=3.0,
            source=_source(
                tool_arguments="cmd candidate",
                tool_arguments_hash="hash-candidate",
                output="candidate output",
                output_hash="same-output-hash",
            ),
        ),
    ])

    results = await searcher.search_tool_call(
        excluded_ids=["excluded-call"],
        match="show output",
    )

    assert results == []


@pytest.mark.asyncio
async def test_search_tool_call_drops_candidate_by_excluded_output_similarity(
    es_mock: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
):
    searcher = DummyToolCallSearcher(es_mock)
    searcher.search_tool_call_by_arguments_hash = AsyncMock(return_value=[])  # type: ignore[method-assign]
    monkeypatch.setattr("prokaryotes.search_v1.tool_calls.str_similarity_batch", lambda _a, _b: [0.95])
    es_mock.mget.return_value = {
        "docs": [
            {
                "found": True,
                "_source": _source(
                    tool_arguments="cmd excluded",
                    tool_arguments_hash="hash-excluded",
                    output="excluded output",
                    output_hash="excluded-output-hash",
                ),
            }
        ]
    }
    es_mock.search.return_value = _hits_response([
        _hit(
            doc_id="candidate",
            score=3.0,
            source=_source(
                tool_arguments="cmd candidate",
                tool_arguments_hash="hash-candidate",
                output="candidate output",
                output_hash="candidate-output-hash",
            ),
        ),
    ])

    results = await searcher.search_tool_call(
        excluded_ids=["excluded-call"],
        match="show output",
        min_output_similarity_score=0.9,
    )

    assert results == []


@pytest.mark.asyncio
async def test_search_tool_call_applies_min_final_score_and_limit(es_mock: AsyncMock):
    searcher = DummyToolCallSearcher(es_mock)
    searcher.search_tool_call_by_arguments_hash = AsyncMock(return_value=[])  # type: ignore[method-assign]
    es_mock.search.return_value = _hits_response([
        _hit(
            doc_id="top",
            score=4.5,
            source=_source(
                created_at=_created_at(0),
                tool_arguments="cmd top",
                tool_arguments_hash="hash-top",
                output="alpha output 111",
                output_hash="hash-out-top",
            ),
        ),
        _hit(
            doc_id="middle",
            score=3.0,
            source=_source(
                created_at=_created_at(1),
                tool_arguments="cmd middle",
                tool_arguments_hash="hash-middle",
                output="beta result 222",
                output_hash="hash-out-middle",
            ),
        ),
        _hit(
            doc_id="low",
            score=1.0,
            source=_source(
                created_at=_created_at(2),
                tool_arguments="cmd low",
                tool_arguments_hash="hash-low",
                output="gamma text 333",
                output_hash="hash-out-low",
            ),
        ),
    ])

    results = await searcher.search_tool_call(
        match="inspect",
        min_final_score=0.5,
        limit=1,
    )

    assert [doc.doc_id for doc in results] == ["top"]
