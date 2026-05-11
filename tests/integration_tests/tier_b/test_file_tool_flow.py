"""Tier B file-tool regression coverage: refresh across write/reconcile and compaction."""
from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from prokaryotes.api_v1.models import ContextPartition, ContextPartitionItem
from tests.integration_tests.fakes import LLMRound, LLMScript, ToolCallSpec
from tests.integration_tests.stream_utils import collect_stream, request_scope

pytestmark = pytest.mark.integration


def _hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _user_message(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant_message(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _read_args(path: Path, start_line: int) -> str:
    return json.dumps({
        "action": "read_lines",
        "path": str(path),
        "expected_revision": None,
        "start_line": start_line,
        "end_line": None,
        "new_text": None,
    })


def _replace_args(path: Path, expected_revision: str, start_line: int, end_line: int, new_text: str) -> str:
    return json.dumps({
        "action": "replace_lines",
        "path": str(path),
        "expected_revision": expected_revision,
        "start_line": start_line,
        "end_line": end_line,
        "new_text": new_text,
    })


def _create_args(path: Path, new_text: str) -> str:
    return json.dumps({
        "action": "create_file",
        "path": str(path),
        "expected_revision": None,
        "start_line": None,
        "end_line": None,
        "new_text": new_text,
    })


def _live_windows_for_path(partition: ContextPartition, path: Path | str) -> list[ContextPartitionItem]:
    resolved = path if isinstance(path, str) else str(path.resolve())
    return [
        item for item in partition.items
        if item.type == "function_call_output"
        and (item.prokaryotes_annotations or {}).get("file_tool.path") == resolved
        and (item.prokaryotes_annotations or {}).get("file_tool.status") == "live"
    ]


def _tracked_outputs_for_path(partition: ContextPartition, path: Path | str) -> list[ContextPartitionItem]:
    resolved = path if isinstance(path, str) else str(path.resolve())
    return [
        item for item in partition.items
        if item.type == "function_call_output"
        and (item.prokaryotes_annotations or {}).get("file_tool.path") == resolved
    ]


def _provider_tool_outputs(provider: str, partition: ContextPartition) -> dict[str, str]:
    if provider == "openai":
        return {
            item["call_id"]: item["output"]
            for item in partition.to_openai_input()
            if item["type"] == "function_call_output" and "call_id" in item and "output" in item
        }

    _system, messages = partition.to_anthropic_messages()
    outputs: dict[str, str] = {}
    for message in messages:
        for block in message["content"]:
            if block["type"] == "tool_result":
                outputs[block["tool_use_id"]] = block["content"]
    return outputs


async def _wait_for_compaction(
    client,
    conversation_uuid: str,
    pending_partition_uuid: str,
    *,
    attempts: int = 30,
    delay: float = 0.1,
) -> None:
    for _ in range(attempts):
        response = await client.get(
            "/compaction-status",
            params={
                "conversation_uuid": conversation_uuid,
                "pending_partition_uuid": pending_partition_uuid,
            },
        )
        if response.json().get("done"):
            return
        await asyncio.sleep(delay)
    raise AssertionError("compaction did not complete within timeout")


async def _run_turn(
    web_harness,
    authed_client,
    conversation_uuid: str,
    *,
    messages: list[dict],
    partition_uuid: str | None,
    rounds: list[LLMRound],
    summary_text: str = "STUB SUMMARY",
) -> tuple[str, str, list[dict]]:
    web_harness.llm_client.set_script(
        LLMScript(
            rounds=rounds,
            summary_text=summary_text,
        )
    )
    payload: dict = {"conversation_uuid": conversation_uuid, "messages": messages}
    if partition_uuid is not None:
        payload["partition_uuid"] = partition_uuid
    async with request_scope(web_harness):
        async with authed_client.stream("POST", "/chat", json=payload) as response:
            events = await collect_stream(response)
    new_partition_uuid = events[0]["partition_uuid"]
    assistant_text = "".join(event["text_delta"] for event in events if "text_delta" in event)
    return new_partition_uuid, assistant_text, events


def _assert_serialized_live_windows(
    provider: str,
    partition: ContextPartition,
    path: Path,
    *,
    expected_revision: str,
    expected_substrings: list[str],
    forbidden_substrings: list[str],
) -> None:
    live_windows = _live_windows_for_path(partition, path)
    assert len(live_windows) == 2
    assert all(
        (item.prokaryotes_annotations or {}).get("file_tool.revision") == expected_revision
        for item in live_windows
    )

    serialized_outputs = _provider_tool_outputs(provider, partition)
    live_outputs = [serialized_outputs[item.call_id] for item in live_windows if item.call_id in serialized_outputs]
    assert len(live_outputs) == 2
    joined = "\n".join(live_outputs)
    for expected in expected_substrings:
        assert expected in joined
    for forbidden in forbidden_substrings:
        assert forbidden not in joined


def _summary_request_text(provider: str, llm_client) -> str:
    call = llm_client.summary_create_calls[-1]
    if provider == "openai":
        return json.dumps(call["input"], ensure_ascii=False)
    return json.dumps({
        "system": call.get("system"),
        "messages": call["messages"],
    }, ensure_ascii=False)


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_windows_refresh_across_write_and_reconcile(web_harness, authed_client, request):
    provider = request.node.callspec.params["web_harness"]
    conversation_uuid = str(uuid4())
    initial = "one\ntwo\nthree\nfour\n"
    after_write = "one\nTWO\nTHREE_X\nfour\n"
    after_external_edit = "one\nTWO\nTHREE_Y\nfour\nfive\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text(initial, encoding="utf-8")

        messages = [_user_message("Inspect the file in two windows.")]
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=None,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the top."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_read_args(target, 1),
                            call_id="call-read-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the tail."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_read_args(target, 3),
                            call_id="call-read-2",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["I inspected it."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        cached = ContextPartition.model_validate_json(
            await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
        )
        _assert_serialized_live_windows(
            provider,
            cached,
            target,
            expected_revision=_hash(initial),
            expected_substrings=["1 | one", "3 | three"],
            forbidden_substrings=["2 | TWO", "3 | THREE_X", "3 | THREE_Y"],
        )

        messages.append(_user_message("Update the middle lines."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Editing the file."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_replace_args(target, _hash(initial), 2, 3, "TWO\nTHREE_X\n"),
                            call_id="call-write-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["It is updated."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        cached = ContextPartition.model_validate_json(
            await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
        )
        _assert_serialized_live_windows(
            provider,
            cached,
            target,
            expected_revision=_hash(after_write),
            expected_substrings=["2 | TWO", "3 | THREE_X"],
            forbidden_substrings=["2 | two", "3 | three", "3 | THREE_Y"],
        )
        write_record = next(
            item for item in cached.items
            if item.call_id == "call-write-1" and item.type == "function_call_output"
        )
        assert write_record.prokaryotes_annotations == {"file_tool.path": str(target.resolve())}

        target.write_text(after_external_edit, encoding="utf-8")

        messages.append(_user_message("Double-check the current file state."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=["Checked."], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append(_assistant_message(assistant_text))

        captured = web_harness.llm_client.stream_context_partitions[-1]
        _assert_serialized_live_windows(
            provider,
            captured,
            target,
            expected_revision=_hash(after_external_edit),
            expected_substrings=["2 | TWO", "3 | THREE_Y", "5 | five"],
            forbidden_substrings=["3 | THREE_X", "3 | three"],
        )

        cached = ContextPartition.model_validate_json(
            await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
        )
        _assert_serialized_live_windows(
            provider,
            cached,
            target,
            expected_revision=_hash(after_external_edit),
            expected_substrings=["2 | TWO", "3 | THREE_Y", "5 | five"],
            forbidden_substrings=["3 | THREE_X", "3 | three"],
        )


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_live_windows_survive_compaction(web_harness, authed_client, request):
    provider = request.node.callspec.params["web_harness"]
    conversation_uuid = str(uuid4())
    initial = "one\ntwo\nthree\nfour\n"
    after_write = "one\nTWO\nTHREE_X\nfour\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text(initial, encoding="utf-8")

        messages = [_user_message("Inspect the file in two windows.")]
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=None,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the top."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_read_args(target, 1),
                            call_id="call-read-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the tail."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_read_args(target, 3),
                            call_id="call-read-2",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["I inspected it."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        messages.append(_user_message("Acknowledge the inspection."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=["Acknowledged."], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append(_assistant_message(assistant_text))

        messages.append(_user_message("Now update the middle lines."))
        pending_partition_uuid, assistant_text, events = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Editing the file."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_replace_args(target, _hash(initial), 2, 3, "TWO\nTHREE_X\n"),
                            call_id="call-write-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(
                    text_deltas=["It is updated and compacted."],
                    stop_reason="end_turn",
                    input_tokens=5000,
                ),
            ],
            summary_text="LIVE WINDOW SUMMARY",
        )
        messages.append(_assistant_message(assistant_text))
        event_types = [key for event in events for key in event.keys()]
        assert "compaction_pending" in event_types
        assert event_types[-1] == "compaction_pending"

        await _wait_for_compaction(authed_client, conversation_uuid, pending_partition_uuid)

        cached = ContextPartition.model_validate_json(
            await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
        )
        assert cached.partition_uuid != pending_partition_uuid
        assert cached.parent_partition_uuid == pending_partition_uuid
        assert cached.ancestor_summaries == ["LIVE WINDOW SUMMARY"]
        assert cached.raw_message_start_index > 0
        _assert_serialized_live_windows(
            provider,
            cached,
            target,
            expected_revision=_hash(after_write),
            expected_substrings=["2 | TWO", "3 | THREE_X"],
            forbidden_substrings=["2 | two", "3 | three"],
        )

        messages.append(_user_message("Continue from the compacted state."))
        active_partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=cached.partition_uuid,
            rounds=[LLMRound(text_deltas=["Continuing."], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append(_assistant_message(assistant_text))
        assert active_partition_uuid == cached.partition_uuid

        captured = web_harness.llm_client.stream_context_partitions[-1]
        _assert_serialized_live_windows(
            provider,
            captured,
            target,
            expected_revision=_hash(after_write),
            expected_substrings=["2 | TWO", "3 | THREE_X"],
            forbidden_substrings=["2 | two", "3 | three"],
        )


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_create_file_persists_created_record_across_turns(web_harness, authed_client, request):
    provider = request.node.callspec.params["web_harness"]
    conversation_uuid = str(uuid4())

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "created.txt"

        messages = [_user_message("Create the file.")]
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=None,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Creating it."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_create_args(target, "alpha\nbeta\n"),
                            call_id="call-create-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Created."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
        cached = ContextPartition.model_validate_json(
            await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
        )
        created_item = next(
            item for item in cached.items
            if item.call_id == "call-create-1" and item.type == "function_call_output"
        )
        assert created_item.output.startswith("CREATED ")
        assert created_item.prokaryotes_annotations == {"file_tool.path": str(target.resolve())}

        messages.append(_user_message("Continue from there."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=["Continuing."], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append(_assistant_message(assistant_text))

        captured = web_harness.llm_client.stream_context_partitions[-1]
        created_item = next(
            item for item in captured.items
            if item.call_id == "call-create-1" and item.type == "function_call_output"
        )
        assert created_item.output.startswith("CREATED ")
        assert created_item.prokaryotes_annotations == {"file_tool.path": str(target.resolve())}
        assert _provider_tool_outputs(provider, captured)["call-create-1"].startswith("CREATED ")


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_already_exists_window_normalizes_on_next_turn(web_harness, authed_client, request):
    provider = request.node.callspec.params["web_harness"]
    conversation_uuid = str(uuid4())
    initial = "alpha\nbeta\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "exists.txt"
        target.write_text(initial, encoding="utf-8")

        messages = [_user_message("Create the file if needed.")]
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=None,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Trying to create it."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_create_args(target, "ignored\n"),
                            call_id="call-exists-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["I checked."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        cached = ContextPartition.model_validate_json(
            await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
        )
        already_exists = next(
            item for item in cached.items
            if item.call_id == "call-exists-1" and item.type == "function_call_output"
        )
        assert already_exists.output.startswith("ALREADY_EXISTS ")
        assert already_exists.prokaryotes_annotations["file_tool.status"] == "live"
        assert already_exists.prokaryotes_annotations["file_tool.revision"] == _hash(initial)

        messages.append(_user_message("Continue."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=["Done."], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append(_assistant_message(assistant_text))

        captured = web_harness.llm_client.stream_context_partitions[-1]
        normalized = next(
            item for item in captured.items
            if item.call_id == "call-exists-1" and item.type == "function_call_output"
        )
        assert normalized.output.startswith("FILE ")
        assert "ALREADY_EXISTS " not in normalized.output
        assert "1 | alpha" in normalized.output
        serialized = _provider_tool_outputs(provider, captured)["call-exists-1"]
        assert serialized.startswith("FILE ")
        assert "ALREADY_EXISTS " not in serialized


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_conflict_window_normalizes_on_next_turn(web_harness, authed_client, request):
    provider = request.node.callspec.params["web_harness"]
    conversation_uuid = str(uuid4())
    initial = "alpha\nbeta\n"
    after_external_edit = "ALPHA\nbeta\ngamma\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text(initial, encoding="utf-8")

        messages = [_user_message("Read the file.")]
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=None,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading it."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_read_args(target, 1),
                            call_id="call-read-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Read."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        target.write_text(after_external_edit, encoding="utf-8")

        messages.append(_user_message("Apply the original edit."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Trying the edit."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_replace_args(target, _hash(initial), 1, 1, "alpha-updated\n"),
                            call_id="call-conflict-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Handled."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        cached = ContextPartition.model_validate_json(
            await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
        )
        conflict = next(
            item for item in cached.items
            if item.call_id == "call-conflict-1" and item.type == "function_call_output"
        )
        assert conflict.output.startswith("CONFLICT ")
        assert conflict.prokaryotes_annotations["file_tool.status"] == "live"
        assert conflict.prokaryotes_annotations["file_tool.revision"] == _hash(after_external_edit)

        messages.append(_user_message("Continue."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=["Done."], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append(_assistant_message(assistant_text))

        captured = web_harness.llm_client.stream_context_partitions[-1]
        tracked_outputs = _tracked_outputs_for_path(captured, target)
        assert len(tracked_outputs) == 2
        assert all(
            (item.prokaryotes_annotations or {}).get("file_tool.revision") == _hash(after_external_edit)
            for item in tracked_outputs
        )
        assert all(not item.output.startswith("CONFLICT ") for item in tracked_outputs)
        serialized = _provider_tool_outputs(provider, captured)
        assert serialized["call-conflict-1"].startswith("FILE ")
        assert "CONFLICT " not in serialized["call-conflict-1"]
        assert "3 | gamma" in "\n".join(serialized[item.call_id] for item in tracked_outputs)


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_range_error_window_normalizes_on_next_turn(web_harness, authed_client, request):
    provider = request.node.callspec.params["web_harness"]
    conversation_uuid = str(uuid4())
    initial = "a\nb\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text(initial, encoding="utf-8")

        messages = [_user_message("Read the file.")]
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=None,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading it."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_read_args(target, 1),
                            call_id="call-read-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Read."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        messages.append(_user_message("Replace the nonexistent range."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Trying the range."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_replace_args(target, _hash(initial), 5, 9, "X\n"),
                            call_id="call-range-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Handled."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        cached = ContextPartition.model_validate_json(
            await web_harness.redis_client.get(f"context_partition:{conversation_uuid}")
        )
        range_error = next(
            item for item in cached.items
            if item.call_id == "call-range-1" and item.type == "function_call_output"
        )
        assert range_error.output.startswith("RANGE_ERROR ")
        assert range_error.prokaryotes_annotations["file_tool.status"] == "live"
        assert range_error.prokaryotes_annotations["file_tool.revision"] == _hash(initial)

        messages.append(_user_message("Continue."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=["Done."], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append(_assistant_message(assistant_text))

        captured = web_harness.llm_client.stream_context_partitions[-1]
        normalized = next(
            item for item in captured.items
            if item.call_id == "call-range-1" and item.type == "function_call_output"
        )
        assert normalized.output.startswith("FILE ")
        assert "RANGE_ERROR " not in normalized.output
        serialized = _provider_tool_outputs(provider, captured)["call-range-1"]
        assert serialized.startswith("FILE ")
        assert "RANGE_ERROR " not in serialized


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_missing_file_tombstones_on_next_turn(web_harness, authed_client, request):
    provider = request.node.callspec.params["web_harness"]
    conversation_uuid = str(uuid4())

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text("here today\n", encoding="utf-8")

        messages = [_user_message("Read the file.")]
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=None,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading it."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_read_args(target, 1),
                            call_id="call-read-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Read."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        target.unlink()

        messages.append(_user_message("Continue."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=["Done."], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append(_assistant_message(assistant_text))

        captured = web_harness.llm_client.stream_context_partitions[-1]
        tombstone = next(
            item for item in _tracked_outputs_for_path(captured, target)
            if item.call_id == "call-read-1"
        )
        assert tombstone.prokaryotes_annotations["file_tool.status"] == "stale"
        assert "no longer accessible" in tombstone.output
        assert "FileNotFoundError" in tombstone.output
        serialized = _provider_tool_outputs(provider, captured)["call-read-1"]
        assert "status=stale" in serialized
        assert "FileNotFoundError" in serialized


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_symlink_escape_tombstones_on_next_turn(web_harness, authed_client, request):
    provider = request.node.callspec.params["web_harness"]
    conversation_uuid = str(uuid4())

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        workspace = Path(temp_dir)
        target = workspace / "tracked.txt"
        target.write_text("inside\n", encoding="utf-8")
        tracked_path = str(target.resolve())
        with TemporaryDirectory(prefix=".file-tool-outside-") as outside_dir:
            outside = Path(outside_dir) / "outside.txt"
            outside.write_text("top secret\n", encoding="utf-8")

            messages = [_user_message("Read the file.")]
            partition_uuid, assistant_text, _ = await _run_turn(
                web_harness,
                authed_client,
                conversation_uuid,
                messages=messages,
                partition_uuid=None,
                rounds=[
                    LLMRound(
                        stop_reason="tool_use",
                        text_deltas=["Reading it."],
                        tool_calls=[
                            ToolCallSpec(
                                arguments=_read_args(target, 1),
                                call_id="call-read-1",
                                name="file_tool",
                            ),
                        ],
                        input_tokens=500,
                    ),
                    LLMRound(text_deltas=["Read."], stop_reason="end_turn", input_tokens=500),
                ],
            )
            messages.append(_assistant_message(assistant_text))

            target.unlink()
            target.symlink_to(outside)

            messages.append(_user_message("Continue."))
            partition_uuid, assistant_text, _ = await _run_turn(
                web_harness,
                authed_client,
                conversation_uuid,
                messages=messages,
                partition_uuid=partition_uuid,
                rounds=[LLMRound(text_deltas=["Done."], stop_reason="end_turn", input_tokens=500)],
            )
            messages.append(_assistant_message(assistant_text))

            captured = web_harness.llm_client.stream_context_partitions[-1]
            tombstone = next(
                item for item in _tracked_outputs_for_path(captured, tracked_path)
                if item.call_id == "call-read-1"
            )
            assert tombstone.prokaryotes_annotations["file_tool.status"] == "stale"
            assert "no longer accessible" in tombstone.output
            assert "ValueError" in tombstone.output
            assert "outside" not in tombstone.output
            assert "top secret" not in tombstone.output
            serialized = _provider_tool_outputs(provider, captured)["call-read-1"]
            assert "status=stale" in serialized
            assert "ValueError" in serialized
            assert "outside" not in serialized
            assert "top secret" not in serialized


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_compaction_summary_input_strips_live_window_bodies(web_harness, authed_client, request):
    provider = request.node.callspec.params["web_harness"]
    conversation_uuid = str(uuid4())
    initial = "one\ntwo\nthree\nfour\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text(initial, encoding="utf-8")

        messages = [_user_message("Inspect the file in two windows.")]
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=None,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the top."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_read_args(target, 1),
                            call_id="call-read-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the tail."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_read_args(target, 3),
                            call_id="call-read-2",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["I inspected it."], stop_reason="end_turn", input_tokens=500),
            ],
        )
        messages.append(_assistant_message(assistant_text))

        messages.append(_user_message("Acknowledge the inspection."))
        partition_uuid, assistant_text, _ = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[LLMRound(text_deltas=["Acknowledged."], stop_reason="end_turn", input_tokens=500)],
        )
        messages.append(_assistant_message(assistant_text))

        messages.append(_user_message("Now update the middle lines."))
        pending_partition_uuid, assistant_text, events = await _run_turn(
            web_harness,
            authed_client,
            conversation_uuid,
            messages=messages,
            partition_uuid=partition_uuid,
            rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Editing the file."],
                    tool_calls=[
                        ToolCallSpec(
                            arguments=_replace_args(target, _hash(initial), 2, 3, "TWO\nTHREE_X\n"),
                            call_id="call-write-1",
                            name="file_tool",
                        ),
                    ],
                    input_tokens=500,
                ),
                LLMRound(
                    text_deltas=["It is updated and compacted."],
                    stop_reason="end_turn",
                    input_tokens=5000,
                ),
            ],
            summary_text="LIVE WINDOW SUMMARY",
        )
        messages.append(_assistant_message(assistant_text))
        event_types = [key for event in events for key in event.keys()]
        assert "compaction_pending" in event_types
        assert event_types[-1] == "compaction_pending"

        await _wait_for_compaction(authed_client, conversation_uuid, pending_partition_uuid)

        request_text = _summary_request_text(provider, web_harness.llm_client)
        placeholder = (
            f"[Live tracked file: {target.resolve()} — current contents are tracked via the "
            "live-window mechanism on subsequent turns, not summarized here.]"
        )
        assert placeholder in request_text
        assert "Current view (" not in request_text
        assert f"FILE path={target.resolve()} revision=" not in request_text
        assert "status=live" not in request_text
        assert "EDITED path=" in request_text
        assert "Added (lines 2-3):" in request_text
        assert "2 | TWO" in request_text
        assert "3 | THREE_X" in request_text
        # Adjacent post-edit lines now appear inside the frozen edit record's Context blocks.
        # The live-window-stripped invariant is enforced by the placeholder / Current view /
        # FILE path= / status=live assertions above.
        assert "Context before (lines 1-1):" in request_text
        assert "1 | one" in request_text
        assert "Context after (lines 4-4):" in request_text
        assert "4 | four" in request_text
