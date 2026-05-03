import json
import re
from unittest.mock import AsyncMock

import pytest

from prokaryotes.tools_v1.think import ThinkTool


def make_args(goal="my goal", context="my context", perspectives=None):
    return json.dumps({
        "goal": goal,
        "context": context,
        "perspectives": perspectives if perspectives is not None else ["trade-offs", "risks"],
    })


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.complete = AsyncMock(return_value="ok")
    return client


@pytest.mark.asyncio
async def test_call_omits_perspectives_block_when_empty(mock_client):
    tool = ThinkTool(mock_client, model="test-model")
    await tool.call(make_args(perspectives=[]), call_id="c1")

    partition = mock_client.complete.call_args[0][0]
    assert "<perspectives-" not in partition.items[1].content
    assert "perspectives" not in partition.items[0].content


@pytest.mark.asyncio
async def test_call_wraps_goal_context_and_perspectives_in_uuid_delimiters(mock_client):
    tool = ThinkTool(mock_client, model="test-model")
    await tool.call(make_args(goal="my goal", context="my context", perspectives=["trade-offs", "risks"]), call_id="c1")

    partition = mock_client.complete.call_args[0][0]
    user_content = partition.items[1].content
    system_content = partition.items[0].content

    goal_match = re.search(r"<goal-([0-9a-f]+)>(.*?)</goal-\1>", user_content, re.DOTALL)
    context_match = re.search(r"<context-([0-9a-f]+)>(.*?)</context-\1>", user_content, re.DOTALL)
    persp_match = re.search(r"<perspectives-([0-9a-f]+)>(.*?)</perspectives-\1>", user_content, re.DOTALL)

    assert goal_match and "my goal" in goal_match.group(2)
    assert context_match and "my context" in context_match.group(2)
    assert persp_match and "trade-offs" in persp_match.group(2) and "risks" in persp_match.group(2)
    assert "perspectives" in system_content


@pytest.mark.parametrize("kwarg,env_var,expected", [
    ("high", "medium", "high"),
    (None, "medium", "medium"),
    (None, None, "low"),
])
def test_reasoning_effort_precedence(monkeypatch, kwarg, env_var, expected):
    if env_var is not None:
        monkeypatch.setenv("THINK_TOOL_REASONING_EFFORT", env_var)
    else:
        monkeypatch.delenv("THINK_TOOL_REASONING_EFFORT", raising=False)
    tool = ThinkTool(AsyncMock(), model="test-model", reasoning_effort=kwarg)
    assert tool.reasoning_effort == expected
