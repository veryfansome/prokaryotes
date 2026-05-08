import pytest

from prokaryotes.tools_v1.shell_command import ShellCommandTool


@pytest.mark.asyncio
async def test_call_happy_path():
    tool = ShellCommandTool()
    result = await tool.call('{"command": "echo hello", "reason": "test"}', "call_1")

    assert result.call_id == "call_1"
    assert result.type == "function_call_output"
    assert "Exit code: 0" in result.output
    assert "# STDOUT" in result.output
    assert "hello" in result.output
    assert "# STDERR" in result.output


@pytest.mark.asyncio
async def test_call_invalid_json_returns_error():
    tool = ShellCommandTool()
    result = await tool.call("not json", "call_4")

    assert "An error occurred" in result.output


@pytest.mark.asyncio
async def test_call_nonzero_exit_and_stderr():
    tool = ShellCommandTool()
    result = await tool.call('{"command": "ls /nonexistent_path_xyz", "reason": "test"}', "call_2")

    assert "Exit code: 1" in result.output or "Exit code: 2" in result.output
    assert "# STDERR" in result.output
    stderr_section = result.output.split("# STDERR")[1]
    assert len(stderr_section.strip()) > 0


@pytest.mark.asyncio
async def test_call_truncates_long_output():
    tool = ShellCommandTool()
    line_count = tool.max_output_lines + 50
    result = await tool.call(
        f'{{"command": "seq 1 {line_count}", "reason": "test"}}', "call_3"
    )

    assert f"Truncated after {tool.max_output_lines} lines" in result.output
    stdout_section = result.output.split("# STDOUT")[1].split("# STDERR")[0]
    assert str(line_count) not in stdout_section
