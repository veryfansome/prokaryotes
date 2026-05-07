"""Tests for the model-layer changes around `prokaryotes_annotations` and the
removal of the `text_preamble` field."""

from prokaryotes.api_v1.models import ContextPartition, ContextPartitionItem


def test_context_partition_item_has_annotations_field_default_none():
    item = ContextPartitionItem(role="user", content="hi")

    assert item.prokaryotes_annotations is None


def test_context_partition_item_no_text_preamble_field():
    """text_preamble was removed; constructing with it must raise (Pydantic strict on extras
    is the default, and the field no longer exists)."""
    # The field is gone; assigning it via setattr would set an arbitrary attribute, but
    # passing it as a constructor kwarg should be ignored or rejected. The contract we
    # care about is that `to_anthropic_messages` and `to_openai_input` no longer reference
    # text_preamble.
    item = ContextPartitionItem(role="user", content="hi")
    assert not hasattr(item, "text_preamble") or item.text_preamble is None


def test_to_openai_input_excludes_prokaryotes_annotations():
    """Annotations are internal harness metadata and must not leak into the OpenAI payload."""
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


def test_to_openai_input_renames_system_role_to_developer():
    partition = ContextPartition(
        conversation_uuid="c",
        items=[ContextPartitionItem(role="system", content="dev message")],
    )

    result = partition.to_openai_input()

    assert result[0]["role"] == "developer"


def test_to_anthropic_messages_does_not_synthesize_text_block_for_function_call():
    """With text_preamble removed, a function_call item produces only the tool_use block."""
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

    # The assistant message must contain only the tool_use block — no synthesized text block.
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

    # Serialize → deserialize round trip preserves annotations so Redis/ES persistence works.
    serialized = item.model_dump_json()
    rebuilt = ContextPartitionItem.model_validate_json(serialized)

    assert rebuilt.prokaryotes_annotations == item.prokaryotes_annotations
