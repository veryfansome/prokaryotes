import json
from unittest.mock import AsyncMock

import pytest
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.models_v1 import ChatMessage
from prokaryotes.tools_v1.read_file import ReadFileCallback


class _DummyClient:
    pass


def _make_callback() -> ReadFileCallback:
    callback = ReadFileCallback(_DummyClient(), _DummyClient())
    callback.index = AsyncMock(return_value=None)
    return callback


async def _run_call(callback: ReadFileCallback, arguments: dict | str) -> FunctionCallOutput:
    payload = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return await callback.call(
        context_snapshot=[ChatMessage(role="user", content="read this file")],
        arguments=payload,
        call_id="test-call-id",
    )


@pytest.mark.asyncio
async def test_reads_text_file_contents(tmp_path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text("line1\nline2\n")

    callback = _make_callback()
    result = await _run_call(callback, {"path": str(file_path)})

    assert result["type"] == "function_call_output"
    assert result["call_id"] == "test-call-id"
    assert result["output"] == "line1\nline2\n"


@pytest.mark.asyncio
async def test_nonexistent_path_returns_error_text_not_exception(tmp_path):
    callback = _make_callback()
    missing = tmp_path / "missing.txt"

    result = await _run_call(callback, {"path": str(missing)})

    assert "No such file or directory" in result["output"]


@pytest.mark.asyncio
async def test_rejects_binary_file(tmp_path):
    file_path = tmp_path / "binary.bin"
    file_path.write_bytes(b"\x00abc")

    callback = _make_callback()
    result = await _run_call(callback, {"path": str(file_path)})

    assert "File looks binary" in result["output"]


@pytest.mark.asyncio
async def test_rejects_files_at_or_over_size_limit(tmp_path):
    file_path = tmp_path / "large.txt"
    file_path.write_bytes(b"a" * 20_000)

    callback = _make_callback()
    result = await _run_call(callback, {"path": str(file_path)})

    assert "File too large." in result["output"]
    assert "exceeds size limit of 20000" in result["output"]


def test_tool_param_contract():
    callback = _make_callback()
    tool_param = callback.tool_param

    assert tool_param["name"] == "read_file"
    assert tool_param["parameters"]["additionalProperties"] is False
    assert tool_param["parameters"]["required"] == ["path"]


@pytest.mark.asyncio
async def test_malformed_json_does_not_crash_call():
    callback = _make_callback()
    result = await _run_call(callback, "{bad")

    assert "Expecting property name enclosed in double quotes" in result["output"]
    callback.index.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    {"path": 123},
])
async def test_invalid_path_does_not_crash_call(arguments):
    callback = _make_callback()
    result = await _run_call(callback, arguments)

    assert "Invalid path" in result["output"]
    callback.index.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    {},
    {"path": "   "},
])
async def test_missing_or_blank_path_does_not_crash_call(arguments):
    callback = _make_callback()
    result = await _run_call(callback, arguments)

    assert "Missing or empty path" in result["output"]
    callback.index.assert_not_called()
