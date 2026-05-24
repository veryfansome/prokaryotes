"""Tests for `EvalHarness.count_turns` + `run_task` against the new model.

`count_turns` operates on `list[TurnItem]` + a `had_final_assistant` boolean. `run_task` consumes a `ScriptRunResult`
and produces two artifact files (`conversation.json`, `turn_execution.json`) instead of the legacy
`context_partition.json`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    TurnExecution,
    TurnItem,
)
from prokaryotes.eval_v1.models import EvalTask
from prokaryotes.harness_v1.eval import EvalHarness
from prokaryotes.harness_v1.script import ScriptRunResult


def _fc(call_id: str, name: str = "shell_command") -> TurnItem:
    return TurnItem(type="function_call", call_id=call_id, name=name, arguments="{}")


def _fco(call_id: str, output: str = "ok") -> TurnItem:
    return TurnItem(type="function_call_output", call_id=call_id, output=output)


@pytest.mark.parametrize(
    "items, had_final, expected",
    [
        pytest.param([], True, 1, id="zero_tool_calls_one_final_message"),
        pytest.param([_fc("c1"), _fco("c1")], True, 2, id="one_round_then_final"),
        pytest.param([_fc("c1"), _fco("c1"), _fc("c2"), _fco("c2")], True, 3, id="two_rounds_then_final"),
        pytest.param([_fc("c1"), _fco("c1"), _fc("c2"), _fco("c2")], False, 2, id="max_rounds_hit_no_final"),
        pytest.param([_fc("c1"), _fc("c2"), _fco("c1"), _fco("c2")], True, 2, id="parallel_tool_calls_in_one_round"),
        pytest.param([_fc("c1")], False, 1, id="only_function_call_no_output"),
        pytest.param(
            [_fc("c1", name="think"), _fco("c1"), _fc("c2", name="shell_command"), _fco("c2")],
            True,
            3,
            id="think_call_counts_like_any_tool",
        ),
    ],
)
def test_count_turns(items, had_final, expected):
    """`count_turns` counts function_call → function_call_output rounds, plus one trailing call if a final
    assistant message was produced. Parallel tool calls in the same round count once."""
    assert EvalHarness.count_turns(items, had_final_assistant=had_final) == expected


@dataclass
class FakeScriptRun:
    """Per-test recipe for what FakeScriptHarness.run() should return."""

    items: list[TurnItem] = field(default_factory=list)
    final_assistant_text: str = "ok"


class FakeScriptHarness:
    """Replaces ScriptHarness for EvalHarness.run_task tests. Records its
    invocation kwargs and returns the scripted ScriptRunResult."""

    __test__ = False

    def __init__(self, *_args, **_kwargs):
        self.calls: list[dict] = []
        self.return_value: FakeScriptRun | None = FakeScriptRun()
        self.closed = False

    async def run(self, *, task, cwd, max_tool_call_rounds, on_usage, verbose):
        self.calls.append({"task": task, "cwd": cwd, "max_tool_call_rounds": max_tool_call_rounds})
        if self.return_value is None:
            return None
        conv = Conversation(
            conversation_uuid="fake-conv",
            bot_author_id="__bot__",
            messages=[ConversationMessage(source_id="0.000000", author_id="__user__", content=task)],
        )
        if self.return_value.final_assistant_text:
            conv.messages.append(
                ConversationMessage(
                    source_id="1.000000",
                    author_id="__bot__",
                    content=self.return_value.final_assistant_text,
                )
            )
        te: TurnExecution | None = None
        if self.return_value.items:
            te = TurnExecution(
                conversation_uuid="fake-conv",
                bot_message_source_id="1.000000",
                items=self.return_value.items,
                completed=True,
            )
        return ScriptRunResult(
            conversation=conv,
            final_assistant_text=self.return_value.final_assistant_text,
            turn_execution=te,
        )

    async def close(self):
        self.closed = True


def _eval_task(
    *,
    check_command: str = "true",
    check_files: dict[str, str] | None = None,
    prompt: str = "do the thing",
    setup_command: str | None = None,
    setup_files: dict[str, str] | None = None,
    task_id: str = "t1",
    tier: int = 1,
    timeout_seconds: int = 30,
) -> EvalTask:
    return EvalTask(
        check_command=check_command,
        check_files=check_files or {},
        description="test task",
        id=task_id,
        prompt=prompt,
        setup_command=setup_command,
        setup_files=setup_files or {},
        tier=tier,
        timeout_seconds=timeout_seconds,
    )


@pytest.fixture
def fake_harness(monkeypatch) -> FakeScriptHarness:
    """Patch make_script_harness to return a FakeScriptHarness."""
    fake = FakeScriptHarness()
    monkeypatch.setattr(EvalHarness, "make_script_harness", lambda self: fake)
    return fake


class TestRunTask:
    @pytest.mark.asyncio
    async def test_setup_files_written_before_agent(self, tmp_path, fake_harness):
        """Setup files must exist in the workspace before the agent's `run()` is called."""
        observed: list[bool] = []

        async def recording_run(*, task, cwd, max_tool_call_rounds, on_usage, verbose):
            observed.append((Path(cwd) / "setup.txt").exists())
            fake_harness.calls.append({"task": task, "cwd": cwd})
            return None

        fake_harness.run = recording_run
        task = _eval_task(setup_files={"setup.txt": "content"})

        await EvalHarness().run_task(task, tmp_path / "task")

        assert observed == [True]

    @pytest.mark.asyncio
    async def test_setup_command_runs_before_agent(self, tmp_path, fake_harness):
        observed: list[bool] = []

        async def recording_run(*, task, cwd, max_tool_call_rounds, on_usage, verbose):
            observed.append((Path(cwd) / "marker.txt").exists())
            fake_harness.calls.append({"task": task, "cwd": cwd})
            return None

        fake_harness.run = recording_run
        task = _eval_task(setup_command="touch marker.txt")

        await EvalHarness().run_task(task, tmp_path / "task")

        assert observed == [True]

    @pytest.mark.asyncio
    async def test_agent_receives_prompt_and_cwd(self, tmp_path, fake_harness):
        task = _eval_task(prompt="custom prompt")

        await EvalHarness().run_task(task, tmp_path / "task")

        assert fake_harness.calls[0]["task"] == "custom prompt"
        assert fake_harness.calls[0]["cwd"] == str(tmp_path / "task")

    @pytest.mark.asyncio
    async def test_agent_exception_recorded_as_error(self, tmp_path, fake_harness):
        async def raising_run(*, task, cwd, max_tool_call_rounds, on_usage, verbose):
            raise RuntimeError("boom")

        fake_harness.run = raising_run
        task = _eval_task()

        result = await EvalHarness().run_task(task, tmp_path / "task")

        assert result.error is not None
        assert "RuntimeError" in result.error
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_timeout_recorded_as_error(self, tmp_path, fake_harness):
        async def slow_run(*, task, cwd, max_tool_call_rounds, on_usage, verbose):
            await asyncio.sleep(10)
            return None

        fake_harness.run = slow_run
        task = _eval_task(timeout_seconds=1)

        result = await EvalHarness().run_task(task, tmp_path / "task")

        assert result.error is not None
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_check_files_invisible_to_agent_but_present_at_check_time(self, tmp_path, fake_harness):
        observed_during_run: list[bool] = []

        async def recording_run(*, task, cwd, max_tool_call_rounds, on_usage, verbose):
            observed_during_run.append((Path(cwd) / "check.py").exists())
            return ScriptRunResult(
                conversation=Conversation(conversation_uuid="c", bot_author_id="__bot__"),
                final_assistant_text="done",
            )

        fake_harness.run = recording_run
        task = _eval_task(check_files={"check.py": "print('hi')"}, check_command="ls check.py")

        result = await EvalHarness().run_task(task, tmp_path / "task")

        # During the agent's run, check.py was NOT present.
        assert observed_during_run == [False]
        # By check time, it was.
        assert "check.py" in result.check_output

    @pytest.mark.asyncio
    async def test_check_command_runs_in_workspace(self, tmp_path, fake_harness):
        workspace = tmp_path / "task"
        task = _eval_task(check_command="pwd")

        result = await EvalHarness().run_task(task, workspace)

        assert str(workspace) in result.check_output

    @pytest.mark.asyncio
    async def test_check_pass_recorded(self, tmp_path, fake_harness):
        task = _eval_task(check_command="true")

        result = await EvalHarness().run_task(task, tmp_path / "task")

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_fail_recorded(self, tmp_path, fake_harness):
        task = _eval_task(check_command="false")

        result = await EvalHarness().run_task(task, tmp_path / "task")

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_conversation_and_turn_execution_artifacts_written_to_workspace(self, tmp_path, fake_harness):
        """When the run produces tool calls, BOTH conversation.json AND
        turn_execution.json are written to the workspace."""
        fake_harness.return_value = FakeScriptRun(items=[_fc("c1"), _fco("c1")])
        workspace = tmp_path / "task"
        task = _eval_task()

        await EvalHarness().run_task(task, workspace)

        assert (workspace / "conversation.json").exists()
        assert (workspace / "turn_execution.json").exists()

    @pytest.mark.asyncio
    async def test_turn_execution_artifact_absent_when_run_was_pure_text(self, tmp_path, fake_harness):
        fake_harness.return_value = FakeScriptRun(items=[])  # no tools
        workspace = tmp_path / "task"
        task = _eval_task()

        await EvalHarness().run_task(task, workspace)

        assert (workspace / "conversation.json").exists()
        assert not (workspace / "turn_execution.json").exists()

    @pytest.mark.asyncio
    async def test_artifacts_not_written_when_agent_returns_none(self, tmp_path, fake_harness):
        fake_harness.return_value = None
        workspace = tmp_path / "task"
        task = _eval_task()

        await EvalHarness().run_task(task, workspace)

        assert not (workspace / "conversation.json").exists()
        assert not (workspace / "turn_execution.json").exists()

    @pytest.mark.asyncio
    async def test_tool_call_count_from_turn_execution(self, tmp_path, fake_harness):
        fake_harness.return_value = FakeScriptRun(items=[_fc("c1"), _fco("c1"), _fc("c2", name="think"), _fco("c2")])
        task = _eval_task()

        result = await EvalHarness().run_task(task, tmp_path / "task")

        assert result.tool_call_count == 2

    @pytest.mark.asyncio
    async def test_tool_call_count_zero_when_no_turn_execution(self, tmp_path, fake_harness):
        fake_harness.return_value = FakeScriptRun(items=[])
        task = _eval_task()

        result = await EvalHarness().run_task(task, tmp_path / "task")

        assert result.tool_call_count == 0

    @pytest.mark.asyncio
    async def test_think_call_count(self, tmp_path, fake_harness):
        fake_harness.return_value = FakeScriptRun(
            items=[_fc("c1", name="think"), _fco("c1"), _fc("c2", name="shell_command"), _fco("c2")]
        )
        task = _eval_task()

        result = await EvalHarness().run_task(task, tmp_path / "task")

        assert result.think_call_count == 1
