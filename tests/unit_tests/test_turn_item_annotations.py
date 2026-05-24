"""TurnItem annotations + provider wire-format conversion invariants.

`prokaryotes_annotations` survives Python-side round-tripping but is structurally excluded from the provider wire format
(`ProjectedItem` has no such field, so projection drops it before wire conversion). The wire-format converters in
`anthropic_v1` and `openai_v1` further enforce role conventions (`system → developer` for OpenAI) and block-shape rules
(no synthetic empty text prefix before tool_use; correct grouping into role-bucketed messages).
"""

from __future__ import annotations

from prokaryotes.anthropic_v1 import _items_to_anthropic_messages
from prokaryotes.conversation_v1.models import ProjectedItem, TurnItem
from prokaryotes.conversation_v1.project import _turn_items_to_projected
from prokaryotes.openai_v1 import _items_to_openai_input


def test_projection_strips_annotations_from_projected_items():
    """`_turn_items_to_projected` is the boundary at which `prokaryotes_annotations`
    is structurally dropped — ProjectedItem has no such field."""
    items = [
        TurnItem(
            type="function_call",
            call_id="c1",
            name="t",
            arguments="{}",
            prokaryotes_annotations={"path": "/tmp/x"},
        ),
        TurnItem(
            type="function_call_output",
            call_id="c1",
            output="ok",
            prokaryotes_annotations={"kind": "live_window"},
        ),
    ]
    projected = _turn_items_to_projected(items)

    for entry in projected:
        assert not hasattr(entry, "prokaryotes_annotations")


def test_openai_wire_excludes_prokaryotes_annotations():
    """ProjectedItem has no annotations field; the wire converter must not
    invent one even if a malformed item slipped through."""
    items = [
        ProjectedItem(type="function_call", call_id="c1", name="t", arguments="{}"),
        ProjectedItem(type="function_call_output", call_id="c1", output="ok"),
    ]
    wire = _items_to_openai_input(items, instruction=None)

    for entry in wire:
        assert "prokaryotes_annotations" not in entry


def test_anthropic_wire_excludes_prokaryotes_annotations():
    items = [
        ProjectedItem(type="message", role="user", content="Hi"),
        ProjectedItem(type="function_call", call_id="c1", name="t", arguments="{}"),
        ProjectedItem(type="function_call_output", call_id="c1", output="ok"),
    ]
    messages = _items_to_anthropic_messages(items)

    for msg in messages:
        for block in msg["content"]:
            assert "prokaryotes_annotations" not in block


def test_to_openai_input_renames_system_role_to_developer():
    items = [ProjectedItem(type="message", role="system", content="Be brief")]

    wire = _items_to_openai_input(items, instruction=None)

    assert wire == [{"role": "developer", "content": "Be brief", "type": "message"}]


def test_to_anthropic_messages_does_not_synthesize_text_block_for_function_call():
    """An assistant function_call with no preceding text must not get an empty
    `{"type": "text", "text": ""}` block prepended on the Anthropic wire."""
    items = [
        ProjectedItem(type="message", role="user", content="x"),
        ProjectedItem(type="function_call", call_id="c", name="t", arguments="{}"),
        ProjectedItem(type="function_call_output", call_id="c", output="y"),
    ]

    messages = _items_to_anthropic_messages(items)

    # Find the assistant message group; its content must be a single tool_use block, NOT preceded by an empty text
    # block.
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == [
        {"type": "tool_use", "id": "c", "name": "t", "input": {}},
    ]


def test_anthropic_full_conversion():
    """Representative end-to-end: system items drop from messages (instruction goes
    via the `system` param), user/assistant text wrap in text blocks, function_call items become tool_use under an
    assistant message, function_call_output items become tool_result under a user message, consecutive same-role items
    coalesce."""
    items = [
        ProjectedItem(type="message", role="system", content="should be dropped"),
        ProjectedItem(type="message", role="user", content="Hi"),
        ProjectedItem(type="message", role="user", content="continued"),
        ProjectedItem(type="message", role="assistant", content="Ok, "),
        ProjectedItem(type="function_call", call_id="c1", name="t", arguments='{"a":1}'),
        ProjectedItem(type="function_call_output", call_id="c1", output="result"),
        ProjectedItem(type="message", role="assistant", content="Done."),
    ]

    messages = _items_to_anthropic_messages(items)

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hi"},
                {"type": "text", "text": "continued"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Ok, "},
                {"type": "tool_use", "id": "c1", "name": "t", "input": {"a": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "result"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Done."},
            ],
        },
    ]


def test_openai_full_conversion():
    """Representative end-to-end for OpenAI: instruction emits as a leading
    developer message; system role on projected items rewrites to developer; function_call / function_call_output emit
    with type + call_id metadata."""
    items = [
        ProjectedItem(type="message", role="system", content="background"),
        ProjectedItem(type="message", role="user", content="Hi"),
        ProjectedItem(type="message", role="assistant", content="Ok"),
        ProjectedItem(type="function_call", call_id="c1", name="t", arguments='{"a":1}'),
        ProjectedItem(type="function_call_output", call_id="c1", output="result"),
    ]

    wire = _items_to_openai_input(items, instruction="Be brief")

    assert wire == [
        {"role": "developer", "content": "Be brief", "type": "message"},
        {"role": "developer", "content": "background", "type": "message"},
        {"role": "user", "content": "Hi", "type": "message"},
        {"role": "assistant", "content": "Ok", "type": "message"},
        {"type": "function_call", "call_id": "c1", "name": "t", "arguments": '{"a":1}'},
        {"type": "function_call_output", "call_id": "c1", "output": "result"},
    ]
