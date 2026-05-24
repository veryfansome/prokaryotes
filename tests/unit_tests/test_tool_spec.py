"""`ToolSpec` → provider wire-format converters.

Anthropic and OpenAI accept similar but not identical JSON Schema dialects. `to_anthropic_tool_param` strips
keys Anthropic's schema validator rejects on integer fields (notably `minimum`), while
`to_openai_function_tool_param` keeps them — OpenAI's validator accepts the standard JSON Schema vocabulary.
"""

from __future__ import annotations

from prokaryotes.tools_v1.file_tool import FileTool


def test_anthropic_tool_param_strips_integer_minimum_from_file_tool_schema(tmp_path):
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    anthropic_param = tool.tool_spec.to_anthropic_tool_param()

    start_line = anthropic_param["input_schema"]["properties"]["start_line"]
    end_line = anthropic_param["input_schema"]["properties"]["end_line"]
    assert start_line["type"] == ["integer", "null"]
    assert end_line["type"] == ["integer", "null"]
    assert "minimum" not in start_line
    assert "minimum" not in end_line


def test_openai_tool_param_keeps_integer_minimum_in_file_tool_schema(tmp_path):
    tool = FileTool(working_file_provider=lambda: [], workspace_root=tmp_path)

    openai_param = tool.tool_spec.to_openai_function_tool_param()

    start_line = openai_param["parameters"]["properties"]["start_line"]
    end_line = openai_param["parameters"]["properties"]["end_line"]
    assert start_line["minimum"] == 1
    assert end_line["minimum"] == 1
