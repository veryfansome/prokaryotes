import json

import pytest
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.tools_v1.scan_directory import ScanDirectoryCallback


class _DummyClient:
    pass


def _make_callback() -> ScanDirectoryCallback:
    callback = ScanDirectoryCallback(_DummyClient(), _DummyClient())
    return callback


async def _run_call(callback: ScanDirectoryCallback, arguments: dict | str) -> FunctionCallOutput:
    payload = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return await callback.call(
        arguments=payload,
        call_id="test-call-id",
    )


def _line_names(output: str) -> list[str]:
    lines = [
        line for line in output.splitlines()
        if line.strip() and line[0] in {"-", "b", "c", "d", "l", "p", "s"}
    ]
    return [line.split()[-1] for line in lines]


@pytest.mark.asyncio
async def test_scans_directory_and_sorts_names_case_insensitive(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "A.txt").write_text("a")

    callback = _make_callback()
    result = await _run_call(
        callback,
        {"path": str(tmp_path), "inclusion_filters": []},
    )

    assert result["type"] == "function_call_output"
    assert result["call_id"] == "test-call-id"
    assert _line_names(result["output"]) == ["A.txt", "b.txt"]


@pytest.mark.asyncio
async def test_inclusion_filters_apply_substring_match(tmp_path):
    (tmp_path / "notes.md").write_text("notes")
    (tmp_path / "report.txt").write_text("report")
    (tmp_path / "image.png").write_text("image")

    callback = _make_callback()
    result = await _run_call(
        callback,
        {"path": str(tmp_path), "inclusion_filters": [".txt", "note"]},
    )

    assert _line_names(result["output"]) == ["notes.md", "report.txt"]


@pytest.mark.asyncio
async def test_omitted_inclusion_filters_defaults_to_all(tmp_path):
    (tmp_path / "one.txt").write_text("one")
    (tmp_path / "two.txt").write_text("two")

    callback = _make_callback()
    result = await _run_call(callback, {"path": str(tmp_path)})

    assert sorted(_line_names(result["output"])) == ["one.txt", "two.txt"]


@pytest.mark.asyncio
async def test_nonexistent_path_returns_error_text_not_exception(tmp_path):
    callback = _make_callback()
    missing = tmp_path / "missing"

    result = await _run_call(
        callback,
        {"path": str(missing), "inclusion_filters": []},
    )

    assert "No such file or directory" in result["output"]


@pytest.mark.asyncio
async def test_entry_limit_raises_tool_error_message(tmp_path):
    for idx in range(101):
        (tmp_path / f"file_{idx:03}.txt").write_text("x")

    callback = _make_callback()
    result = await _run_call(
        callback,
        {"path": str(tmp_path), "inclusion_filters": []},
    )

    assert "Too many files, 100 entry limit reached. Try inclusion_filters." in result["output"]


def test_tool_param_contract():
    callback = _make_callback()
    tool_param = callback.tool_param

    assert tool_param["name"] == "scan_directory"
    assert tool_param["parameters"]["additionalProperties"] is False
    assert tool_param["parameters"]["required"] == ["inclusion_filters", "path"]


@pytest.mark.asyncio
async def test_malformed_json_does_not_crash_call():
    callback = _make_callback()
    result = await _run_call(callback, "{bad")
    assert "Expecting property name enclosed in double quotes" in result["output"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    {"inclusion_filters": []},
    {"path": "   ", "inclusion_filters": []},
    {"path": 123, "inclusion_filters": []},
])
async def test_missing_or_blank_or_invalid_path_does_not_crash_call(arguments):
    callback = _make_callback()
    result = await _run_call(callback, arguments)
    assert "Missing, empty, or invalid path" in result["output"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    {"path": "abc", "inclusion_filters": "[]"},
])
async def test_invalid_inclusion_filters_does_not_crash_call(arguments):
    callback = _make_callback()
    result = await _run_call(callback, arguments)
    assert "Invalid arguments: inclusion_filters should be a list[str]" in result["output"]
