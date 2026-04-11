import ast
from unittest.mock import AsyncMock

import pytest

from prokaryotes.models_v1 import ChatMessage
from prokaryotes.observer_v1.named_entity_observer import NamedEntityObserver
from prokaryotes.observer_v1.topic_observer import TopicClassifyingObserver


@pytest.mark.asyncio
async def test_named_entity_observer_allows_up_to_ten_knn_examples_without_seeds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "prokaryotes.observer_v1.named_entity_observer.get_query_embs",
        AsyncMock(return_value=[[0.1, 0.2]]),
    )
    search_client = AsyncMock()
    search_client.search_named_entities = AsyncMock(return_value=[
        "KNN Entity 1",
        "KNN Entity 2",
        "KNN Entity 3",
        "KNN Entity 4",
        "KNN Entity 5",
        "KNN Entity 6",
        "KNN Entity 7",
        "KNN Entity 8",
        "KNN Entity 9",
        "KNN Entity 10",
        "KNN Entity 11",
    ])

    observer = NamedEntityObserver(
        llm_client=AsyncMock(),
        search_client=search_client,
    )
    message = await observer.developer_message([ChatMessage(role="user", content="we read one dead spy")])

    example_line = next((line for line in message.splitlines() if line.startswith("- For example:")), "")
    assert example_line
    examples = ast.literal_eval(example_line.split(": ", 1)[1])
    assert examples == [
        "KNN Entity 1",
        "KNN Entity 2",
        "KNN Entity 3",
        "KNN Entity 4",
        "KNN Entity 5",
        "KNN Entity 6",
        "KNN Entity 7",
        "KNN Entity 8",
        "KNN Entity 9",
        "KNN Entity 10",
    ]


@pytest.mark.asyncio
async def test_named_entity_observer_prioritizes_seed_entities_then_knn(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "prokaryotes.observer_v1.named_entity_observer.get_query_embs",
        AsyncMock(return_value=[[0.1, 0.2]]),
    )
    search_client = AsyncMock()
    search_client.search_named_entities = AsyncMock(return_value=[
        "KNN Entity 1",
        "KNN Entity 2",
        "KNN Entity 3",
        "KNN Entity 4",
        "KNN Entity 5",
    ])

    observer = NamedEntityObserver(
        llm_client=AsyncMock(),
        search_client=search_client,
        seed_entities=["Seed Entity", "Prior Entity"],
    )
    message = await observer.developer_message([ChatMessage(role="user", content="we read one dead spy")])

    example_line = next((line for line in message.splitlines() if line.startswith("- For example:")), "")
    assert example_line
    examples = ast.literal_eval(example_line.split(": ", 1)[1])
    assert examples == [
        "Seed Entity",
        "Prior Entity",
        "KNN Entity 1",
        "KNN Entity 2",
        "KNN Entity 3",
        "KNN Entity 4",
        "KNN Entity 5",
    ]
    kwargs = search_client.search_named_entities.await_args.kwargs
    assert kwargs["excluded_entities"] == ["Seed Entity", "Prior Entity"]


@pytest.mark.asyncio
async def test_topic_observer_allows_up_to_ten_knn_examples_without_seeds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "prokaryotes.observer_v1.topic_observer.get_query_embs",
        AsyncMock(return_value=[[0.1, 0.2]]),
    )
    search_client = AsyncMock()
    search_client.search_topics = AsyncMock(return_value=[
        "KNN Topic 1",
        "KNN Topic 2",
        "KNN Topic 3",
        "KNN Topic 4",
        "KNN Topic 5",
        "KNN Topic 6",
        "KNN Topic 7",
        "KNN Topic 8",
        "KNN Topic 9",
        "KNN Topic 10",
        "KNN Topic 11",
    ])

    observer = TopicClassifyingObserver(
        llm_client=AsyncMock(),
        search_client=search_client,
    )
    message = await observer.developer_message([ChatMessage(role="user", content="we read one dead spy")])

    example_line = next((line for line in message.splitlines() if line.startswith("- For example:")), "")
    assert example_line
    examples = ast.literal_eval(example_line.split(": ", 1)[1])
    assert examples == [
        "KNN Topic 1",
        "KNN Topic 2",
        "KNN Topic 3",
        "KNN Topic 4",
        "KNN Topic 5",
        "KNN Topic 6",
        "KNN Topic 7",
        "KNN Topic 8",
        "KNN Topic 9",
        "KNN Topic 10",
    ]


@pytest.mark.asyncio
async def test_topic_observer_prioritizes_seed_topics_then_knn(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "prokaryotes.observer_v1.topic_observer.get_query_embs",
        AsyncMock(return_value=[[0.1, 0.2]]),
    )
    search_client = AsyncMock()
    search_client.search_topics = AsyncMock(return_value=[
        "KNN Topic 1",
        "KNN Topic 2",
        "KNN Topic 3",
        "KNN Topic 4",
        "KNN Topic 5",
    ])

    observer = TopicClassifyingObserver(
        llm_client=AsyncMock(),
        search_client=search_client,
        seed_topics=["Seed Topic", "Prior Topic"],
    )
    message = await observer.developer_message([ChatMessage(role="user", content="we read one dead spy")])

    example_line = next((line for line in message.splitlines() if line.startswith("- For example:")), "")
    assert example_line
    examples = ast.literal_eval(example_line.split(": ", 1)[1])
    assert examples == [
        "Seed Topic",
        "Prior Topic",
        "KNN Topic 1",
        "KNN Topic 2",
        "KNN Topic 3",
        "KNN Topic 4",
        "KNN Topic 5",
    ]
    kwargs = search_client.search_topics.await_args.kwargs
    assert kwargs["excluded_topics"] == ["Seed Topic", "Prior Topic"]
