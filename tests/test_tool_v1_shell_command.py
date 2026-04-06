import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from prokaryotes.tools_v1.shell_command import ShellCommandCallback


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, self._stderr


@pytest.fixture
def search_client_mock() -> SimpleNamespace:
    return SimpleNamespace(index_tool_call=AsyncMock())


@pytest.mark.asyncio
async def test_tool_param_schema_has_required_fields(search_client_mock: SimpleNamespace):
    callback = ShellCommandCallback(search_client=search_client_mock)
    tool_param = callback.tool_param

    assert tool_param["name"] == "run_shell_command"
    assert tool_param["strict"] is True
    assert tool_param["parameters"]["required"] == ["command", "reason"]
    assert set(tool_param["parameters"]["properties"]) == {"command", "reason"}


@pytest.mark.asyncio
async def test_extract_keywords_keeps_existing_paths_and_filters_ambiguous_tokens(monkeypatch: pytest.MonkeyPatch):
    existing = {
        "prokaryotes/tools_v1/shell_command.py",
        "prokaryotes/web_v1.py",
    }

    async def fake_exists(path: str) -> bool:
        return path in existing

    monkeypatch.setattr("prokaryotes.tools_v1.shell_command.aiofiles.os.path.exists", fake_exists)
    command = (
        "sed -n '1,200p' prokaryotes/tools_v1/shell_command.py "
        "--path=prokaryotes/web_v1.py . / https://example.com"
    )

    keywords = await ShellCommandCallback.extract_keywords(command)

    assert set(keywords) == {
        "prokaryotes/tools_v1/shell_command.py",
        "prokaryotes/web_v1.py",
    }


@pytest.mark.asyncio
async def test_index_happy_path_forwards_normalized_fields(
    search_client_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    callback = ShellCommandCallback(search_client=search_client_mock)
    callback.extract_keywords = AsyncMock(return_value=["prokaryotes/web_v1.py"])  # type: ignore[method-assign]
    monkeypatch.setattr(
        "prokaryotes.tools_v1.shell_command.get_document_embs",
        AsyncMock(return_value=[[0.5, 0.6]]),
    )
    indexed_doc = object()
    search_client_mock.index_tool_call.return_value = indexed_doc

    result = await callback.index(
        call_id="call-1",
        arguments=json.dumps({
            "command": "sed -n '1,200p' prokaryotes/web_v1.py",
            "reason": "Inspect web flow",
        }),
        labels=["conversation:c1"],
        output="Exit code: 0",
        prompt_summary="The user asked you to inspect web_v1.",
        prompt_summary_emb=[0.1, 0.2],
        topics=["python"],
    )

    assert result is indexed_doc
    search_client_mock.index_tool_call.assert_awaited_once_with(
        call_id="call-1",
        labels=["conversation:c1"],
        output="Exit code: 0",
        prompt_summary="The user asked you to inspect web_v1.",
        prompt_summary_emb=[0.1, 0.2],
        reason="Inspect web flow",
        reason_emb=[0.5, 0.6],
        search_keywords=["prokaryotes/web_v1.py"],
        tool_arguments="sed -n '1,200p' prokaryotes/web_v1.py",
        tool_name="run_shell_command",
        topics=["python"],
    )


@pytest.mark.asyncio
async def test_index_returns_none_on_invalid_arguments(search_client_mock: SimpleNamespace):
    callback = ShellCommandCallback(search_client=search_client_mock)

    result = await callback.index(
        call_id="call-2",
        arguments="{not-json",
        labels=["conversation:c1"],
        output="ignored",
        prompt_summary="ignored",
        prompt_summary_emb=[0.1, 0.2],
        topics=["python"],
    )

    assert result is None
    search_client_mock.index_tool_call.assert_not_called()


@pytest.mark.asyncio
async def test_call_success_formats_output(
    search_client_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    callback = ShellCommandCallback(search_client=search_client_mock)
    fake_process = FakeProcess(stdout=b"line1\nline2", stderr=b"", returncode=0)
    create_subprocess = AsyncMock(return_value=fake_process)
    monkeypatch.setattr("prokaryotes.tools_v1.shell_command.asyncio.create_subprocess_shell", create_subprocess)

    result = await callback.call(
        arguments=json.dumps({"command": "echo hi", "reason": "demo"}),
        call_id="call-3",
    )

    assert result["call_id"] == "call-3"
    assert "Exit code: 0" in result["output"]
    assert "# STDOUT" in result["output"]
    assert "line1" in result["output"]
    assert "line2" in result["output"]
    assert "# STDERR" in result["output"]


@pytest.mark.asyncio
async def test_call_truncates_stdout_when_line_limit_exceeded(
    search_client_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    callback = ShellCommandCallback(search_client=search_client_mock)
    callback.max_output_lines = 2
    fake_process = FakeProcess(stdout=b"a\nb\nc\nd", stderr=b"", returncode=0)
    monkeypatch.setattr(
        "prokaryotes.tools_v1.shell_command.asyncio.create_subprocess_shell",
        AsyncMock(return_value=fake_process),
    )

    result = await callback.call(
        arguments=json.dumps({"command": "echo hi", "reason": "demo"}),
        call_id="call-4",
    )

    assert "\na\n" in result["output"]
    assert "\nb\n" in result["output"]
    assert "\nc\n" not in result["output"]
    assert "--- Truncated after 2 lines ---" in result["output"]


@pytest.mark.asyncio
async def test_call_includes_stderr_section(
    search_client_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    callback = ShellCommandCallback(search_client=search_client_mock)
    fake_process = FakeProcess(stdout=b"ok", stderr=b"permission denied", returncode=1)
    monkeypatch.setattr(
        "prokaryotes.tools_v1.shell_command.asyncio.create_subprocess_shell",
        AsyncMock(return_value=fake_process),
    )

    result = await callback.call(
        arguments=json.dumps({"command": "cat /root/secret", "reason": "demo"}),
        call_id="call-5",
    )

    assert "Exit code: 1" in result["output"]
    assert "# STDERR" in result["output"]
    assert "permission denied" in result["output"]


@pytest.mark.asyncio
async def test_call_replaces_invalid_utf8_bytes(
    search_client_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    callback = ShellCommandCallback(search_client=search_client_mock)
    fake_process = FakeProcess(stdout=b"\xff\xfehello", stderr=b"", returncode=0)
    monkeypatch.setattr(
        "prokaryotes.tools_v1.shell_command.asyncio.create_subprocess_shell",
        AsyncMock(return_value=fake_process),
    )

    result = await callback.call(
        arguments=json.dumps({"command": "echo hi", "reason": "demo"}),
        call_id="call-6",
    )

    assert "Exit code: 0" in result["output"]
    assert "hello" in result["output"]
    assert "\ufffd" in result["output"]


@pytest.mark.asyncio
async def test_call_returns_error_output_when_subprocess_raises(
    search_client_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    callback = ShellCommandCallback(search_client=search_client_mock)

    async def raise_error(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("prokaryotes.tools_v1.shell_command.asyncio.create_subprocess_shell", raise_error)

    result = await callback.call(
        arguments=json.dumps({"command": "echo hi", "reason": "demo"}),
        call_id="call-7",
    )

    assert result["call_id"] == "call-7"
    assert "An error occurred:" in result["output"]
    assert "RuntimeError: boom" in result["output"]


@pytest.mark.asyncio
async def test_call_invokes_subprocess_with_expected_contract(
    search_client_mock: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    callback = ShellCommandCallback(search_client=search_client_mock)
    fake_process = FakeProcess(stdout=b"", stderr=b"", returncode=0)
    create_subprocess = AsyncMock(return_value=fake_process)
    monkeypatch.setattr("prokaryotes.tools_v1.shell_command.asyncio.create_subprocess_shell", create_subprocess)

    await callback.call(
        arguments=json.dumps({"command": "printf test", "reason": "demo"}),
        call_id="call-8",
    )

    create_subprocess.assert_awaited_once_with(
        "printf test",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
