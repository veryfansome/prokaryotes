import asyncio

import pytest

from prokaryotes.api_v1.models import ContextPartition, ContextPartitionItem
from prokaryotes.eval_v1.harness import EvalHarness
from prokaryotes.eval_v1.models import EvalTask


class FakeScriptHarness:
    """Stub that optionally runs a side effect instead of calling a real LLM."""

    def __init__(self, side_effect=None, partition: ContextPartition | None = None):
        self.calls: list[dict] = []
        self._side_effect = side_effect  # callable(cwd) invoked inside run(), or None
        self._partition = partition

    async def run(self, task: str, cwd: str = None, max_tool_call_rounds=None, on_usage=None, verbose=True):
        self.calls.append({"task": task, "cwd": cwd})
        if self._side_effect:
            self._side_effect(cwd)
        return self._partition

    async def close(self):
        pass


def make_harness(fake: FakeScriptHarness) -> EvalHarness:
    harness = EvalHarness()
    harness.make_script_harness = lambda: fake
    return harness


def make_partition(*specs: str) -> ContextPartition:
    return ContextPartition(conversation_uuid="test", items=make_partition_items(*specs))

def make_partition_items(*specs) -> list[ContextPartitionItem]:
    """Build a list of ContextPartitionItems from (role_or_type, ...) shorthand."""
    result = []
    for i, spec in enumerate(specs):
        if spec == "user":
            result.append(ContextPartitionItem(role="user", content="hi"))
        elif spec == "assistant":
            result.append(ContextPartitionItem(role="assistant", content="ok"))
        elif spec == "function_call":
            result.append(ContextPartitionItem(type="function_call", name="tool", arguments="{}", call_id=f"c{i}"))
        elif spec == "function_call_output":
            result.append(ContextPartitionItem(type="function_call_output", call_id=f"c{i}", output="ok"))
    return result


def simple_task(**kwargs) -> EvalTask:
    defaults = dict(id="test_task", tier=1, description="test", prompt="do something", check_command="true")
    return EvalTask(**{**defaults, **kwargs})


@pytest.mark.asyncio
async def test_agent_exception_recorded_as_error(tmp_path):
    class BrokenHarness:
        async def run(self, **kwargs):
            raise RuntimeError("boom")

        async def close(self):
            pass

    harness = EvalHarness()
    harness.make_script_harness = lambda: BrokenHarness()
    result = await harness.run_task(simple_task(), tmp_path)

    assert result.passed is False
    assert result.error is not None
    assert "RuntimeError" in result.error


@pytest.mark.asyncio
async def test_agent_receives_prompt_and_cwd(tmp_path):
    fake = FakeScriptHarness()
    task = simple_task(prompt="list the files")
    await make_harness(fake).run_task(task, tmp_path)

    assert fake.calls[0]["task"] == "list the files"
    assert fake.calls[0]["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_check_command_runs_in_workspace(tmp_path):
    def side_effect(cwd):
        (tmp_path / "result.txt").write_text("42")

    task = simple_task(check_command='grep -qx "42" result.txt')
    result = await make_harness(FakeScriptHarness(side_effect=side_effect)).run_task(task, tmp_path)

    assert result.passed is True


@pytest.mark.asyncio
async def test_check_fail_recorded(tmp_path):
    result = await make_harness(FakeScriptHarness()).run_task(simple_task(check_command="false"), tmp_path)

    assert result.passed is False
    assert result.error is None


@pytest.mark.asyncio
async def test_check_pass_recorded(tmp_path):
    result = await make_harness(FakeScriptHarness()).run_task(simple_task(check_command="true"), tmp_path)

    assert result.passed is True
    assert result.error is None
    assert result.task_id == "test_task"
    assert result.tier == 1


@pytest.mark.asyncio
async def test_context_partition_not_written_when_none(tmp_path):
    await make_harness(FakeScriptHarness()).run_task(simple_task(), tmp_path)

    assert not (tmp_path / "context_partition.json").exists()


@pytest.mark.asyncio
async def test_context_partition_written_to_workspace(tmp_path):
    partition = make_partition("function_call", "function_call_output")
    fake = FakeScriptHarness(partition=partition)
    await make_harness(fake).run_task(simple_task(), tmp_path)

    written = ContextPartition.model_validate_json((tmp_path / "context_partition.json").read_text())
    assert len(written.items) == 2


def test_count_turns_no_items():
    assert EvalHarness.count_turns([]) == 0


def test_count_turns_text_only():
    assert EvalHarness.count_turns(make_partition_items("user", "assistant")) == 1


def test_count_turns_text_then_tool():
    # turn 1: assistant + function_call; turn 2: assistant
    assert EvalHarness.count_turns(
        make_partition_items(
            "user",
            "assistant",
            "function_call",
            "function_call_output",
            "assistant",
        )
    ) == 2


def test_count_turns_tool_only_turns():
    # turn 1: function_call+function_call; turn 2: function_call; turn 3: assistant
    assert EvalHarness.count_turns(
        make_partition_items(
            "user",
            "function_call",
            "function_call",
            "function_call_output",
            "function_call_output",
            "function_call",
            "function_call_output",
            "assistant",
        )
    ) == 3


@pytest.mark.asyncio
async def test_setup_command_runs_before_agent(tmp_path):
    seen_files: list[bool] = []

    def side_effect(cwd):
        seen_files.append((tmp_path / "created.txt").exists())

    task = simple_task(setup_command=f"touch {tmp_path}/created.txt")
    await make_harness(FakeScriptHarness(side_effect=side_effect)).run_task(task, tmp_path)

    assert seen_files == [True], "setup_command must run before the agent runs"


@pytest.mark.asyncio
async def test_setup_files_written_before_agent(tmp_path):
    seen_files: list[bool] = []

    def side_effect(cwd):
        seen_files.append((tmp_path / "data.txt").exists())

    task = simple_task(setup_files={"data.txt": "hello"})
    await make_harness(FakeScriptHarness(side_effect=side_effect)).run_task(task, tmp_path)

    assert seen_files == [True], "setup_files must be written before the agent runs"
    assert (tmp_path / "data.txt").read_text() == "hello"


@pytest.mark.asyncio
async def test_think_call_count(tmp_path):
    partition = ContextPartition(conversation_uuid="test", items=[
        ContextPartitionItem(role="user", content="go"),
        ContextPartitionItem(type="function_call", name="shell_command", arguments="{}", call_id="c1"),
        ContextPartitionItem(type="function_call", name="think", arguments="{}", call_id="c2"),
        ContextPartitionItem(type="function_call_output", call_id="c1", output="ok"),
        ContextPartitionItem(type="function_call_output", call_id="c2", output="ok"),
    ])
    result = await make_harness(FakeScriptHarness(partition=partition)).run_task(simple_task(), tmp_path)

    assert result.tool_call_count == 2
    assert result.think_call_count == 1


@pytest.mark.asyncio
async def test_timeout_recorded_as_error(tmp_path):
    class SlowHarness:
        async def run(self, **kwargs):
            await asyncio.sleep(9999)

        async def close(self):
            pass

    harness = EvalHarness()
    harness.make_script_harness = lambda: SlowHarness()
    result = await harness.run_task(simple_task(check_command="true", timeout_seconds=1), tmp_path)

    assert result.passed is False
    assert result.error is not None
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_tool_call_count_from_partition(tmp_path):
    partition = make_partition("function_call", "function_call_output", "function_call", "function_call_output")
    result = await make_harness(FakeScriptHarness(partition=partition)).run_task(simple_task(), tmp_path)

    assert result.tool_call_count == 2


@pytest.mark.asyncio
async def test_tool_call_count_zero_when_no_partition(tmp_path):
    result = await make_harness(FakeScriptHarness()).run_task(simple_task(), tmp_path)

    assert result.tool_call_count == 0
    assert result.think_call_count == 0
    assert result.turn_count == 0
