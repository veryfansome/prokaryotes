"""Regression tests for internal item annotations in the shared model contract."""

import json

from prokaryotes.api_v1.models import ContextPartition, ContextPartitionItem


def test_context_partition_item_has_annotations_field_default_none():
    item = ContextPartitionItem(role="user", content="hi")

    assert item.prokaryotes_annotations is None


def test_context_partition_item_no_text_preamble_field():
    item = ContextPartitionItem(role="user", content="hi")

    assert "text_preamble" not in ContextPartitionItem.model_fields
    assert not hasattr(item, "text_preamble") or item.text_preamble is None


def test_to_openai_input_excludes_prokaryotes_annotations():
    partition = ContextPartition(
        conversation_uuid="c",
        items=[
            ContextPartitionItem(
                call_id="x",
                output="hello",
                type="function_call_output",
                prokaryotes_annotations={
                    "file_tool.path": "/tmp/x",
                    "file_tool.status": "live",
                },
            ),
        ],
    )

    result = partition.to_openai_input()

    assert len(result) == 1
    assert "prokaryotes_annotations" not in result[0]
    assert result[0]["output"] == "hello"


def test_to_anthropic_messages_excludes_prokaryotes_annotations():
    partition = ContextPartition(
        conversation_uuid="c",
        items=[
            ContextPartitionItem(role="user", content="please inspect this"),
            ContextPartitionItem(
                call_id="x",
                id="x",
                name="file_tool",
                arguments='{"action":"read","path":"/tmp/x"}',
                type="function_call",
                status="completed",
            ),
            ContextPartitionItem(
                call_id="x",
                output="FILE path=/tmp/x revision=abc status=live lines=1-1 line_count=1\n1 | hi",
                type="function_call_output",
                prokaryotes_annotations={
                    "file_tool.path": "/tmp/x",
                    "file_tool.revision": "abc",
                    "file_tool.status": "live",
                    "file_tool.view_start_line": "1",
                    "file_tool.view_end_line": "1",
                },
            ),
        ],
    )

    _system, messages = partition.to_anthropic_messages()

    assert "prokaryotes_annotations" not in json.dumps(messages)
    assert "FILE path=/tmp/x" in json.dumps(messages)


def test_to_openai_input_renames_system_role_to_developer():
    partition = ContextPartition(
        conversation_uuid="c",
        items=[ContextPartitionItem(role="system", content="dev message")],
    )

    result = partition.to_openai_input()

    assert result[0]["role"] == "developer"


def test_to_anthropic_messages_does_not_synthesize_text_block_for_function_call():
    partition = ContextPartition(
        conversation_uuid="c",
        items=[
            ContextPartitionItem(role="user", content="please run a tool"),
            ContextPartitionItem(
                call_id="x",
                id="x",
                name="some_tool",
                arguments="{}",
                type="function_call",
                status="completed",
            ),
            ContextPartitionItem(
                call_id="x",
                output="result",
                type="function_call_output",
            ),
        ],
    )

    _system, messages = partition.to_anthropic_messages()

    assistant = next(m for m in messages if m["role"] == "assistant")
    assert len(assistant["content"]) == 1
    assert assistant["content"][0]["type"] == "tool_use"


def test_annotations_round_trip_through_model_dump_json():
    item = ContextPartitionItem(
        call_id="x",
        output="hello",
        type="function_call_output",
        prokaryotes_annotations={
            "file_tool.path": "/tmp/x",
            "file_tool.revision": "abc",
            "file_tool.status": "live",
        },
    )

    serialized = item.model_dump_json()
    rebuilt = ContextPartitionItem.model_validate_json(serialized)

    assert rebuilt.prokaryotes_annotations == item.prokaryotes_annotations
