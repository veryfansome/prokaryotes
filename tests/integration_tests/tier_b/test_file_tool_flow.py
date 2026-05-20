"""Tier B file-tool flow tests.

Live-window refresh across in-tool writes and external edits, persistence of diagnostic windows (CONFLICT /
ALREADY_EXISTS / RANGE_ERROR / missing / symlink-escape) normalizing on the next turn, survival across compaction,
and stripping of live-window bodies from the summarization input.

The web harness uses `Path.cwd()` (the repo root) as the workspace, so all tracked files are placed under a
`.file-tool-it-*` temp dir relative to cwd.

Tests run against real Postgres / Redis / Elasticsearch via `docker compose`.
"""

from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from prokaryotes.conversation_v1.models import Conversation, TurnExecution
from tests.integration_tests.tier_b._helpers import (
    apply_assignments,
    echo_assistant,
    event_types,
    post_chat,
    post_chat_and_advance,
    user_message,
)
from tests.unit_tests._llm_fakes import LLMRound, LLMScript, ToolCallSpec

pytestmark = pytest.mark.integration


def _hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


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


def _resolved(path: Path | str) -> str:
    return path if isinstance(path, str) else str(path.resolve())


def _live_windows_for_path(items, path: Path | str) -> list:
    """Filter `TurnItem`s (which carry `prokaryotes_annotations`) to live windows for the path. Use on
    `TurnExecution.items` or `Conversation.lifted_turn_items` — NOT on `ProjectedItem` lists (which strip
    annotations)."""
    resolved = _resolved(path)
    return [
        item for item in items
        if getattr(item, "type", None) == "function_call_output"
        and (getattr(item, "prokaryotes_annotations", None) or {}).get("file_tool.path") == resolved
        and (getattr(item, "prokaryotes_annotations", None) or {}).get("file_tool.status") == "live"
    ]


def _outputs_by_call_id(items, call_ids: set[str]) -> dict[str, str]:
    """Map call_id → output text for the given call_ids in `items`. Works for both `TurnItem` and `ProjectedItem`
    (both expose `type`, `call_id`, `output`)."""
    return {
        item.call_id: (item.output or "")
        for item in items
        if getattr(item, "type", None) == "function_call_output"
        and getattr(item, "call_id", None) in call_ids
    }


def _last_stream_items(web_harness):
    """Projected items the harness handed the LLM on the most recent stream_turn. Annotations are stripped during
    projection — use output text to verify post-reconcile state, not `prokaryotes_annotations`."""
    return web_harness.llm_client.stream_turn_calls[-1]["items"]


def _last_complete_items(web_harness):
    """Projected items the harness handed the LLM on the most recent compaction summary `complete()` call."""
    return web_harness.llm_client.complete_calls[-1]["items"]


async def _get_turn_execution(web_harness, conversation_uuid: str, bot_source_id: str) -> TurnExecution:
    """Fetch the TurnExecution for a bot reply that involved tool calls. Raises if missing — the test would be
    exercising the wrong code path otherwise."""
    turn = await web_harness.search_client.get_turn_execution(conversation_uuid, bot_source_id)
    assert turn is not None, f"no TurnExecution found for {conversation_uuid}:{bot_source_id}"
    return turn


async def _refresh_turn_executions(web_harness) -> None:
    """Force an Elasticsearch refresh so freshly-indexed TurnExecutions are visible to the next turn's batch
    `get_turn_executions` search. ES's default 1-second refresh interval is faster than test turn cadence; without
    an explicit refresh, projection on turn N+1 can miss turn N's tool items."""
    await web_harness.search_client.es.indices.refresh(index="turn-executions")


async def _advance_turn(web_harness, authed_client, conversation_uuid, messages, *, snapshot_uuid=None):
    """`post_chat_and_advance` plus a TurnExecutions refresh so the next turn projects this turn's persisted tool
    items."""
    record = await post_chat_and_advance(
        web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
    )
    await _refresh_turn_executions(web_harness)
    return record


async def _wait_for_compaction(
    client,
    conversation_uuid: str,
    pending_snapshot_uuid: str,
    *,
    attempts: int = 30,
    delay: float = 0.1,
) -> str | None:
    for _ in range(attempts):
        response = await client.get(
            "/compaction-status",
            params={
                "conversation_uuid": conversation_uuid,
                "pending_snapshot_uuid": pending_snapshot_uuid,
            },
        )
        body = response.json()
        if body.get("done"):
            return body.get("snapshot_uuid")
        await asyncio.sleep(delay)
    raise AssertionError("compaction did not complete within timeout")


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_windows_refresh_across_write_and_reconcile(web_harness, authed_client, request):
    """Two `read_lines` windows held open across an in-tool write and a later external edit; both windows refresh to
    the current revision."""
    conversation_uuid = str(uuid4())
    initial = "one\ntwo\nthree\nfour\n"
    after_write = "one\nTWO\nTHREE_X\nfour\n"
    after_external_edit = "one\nTWO\nTHREE_Y\nfour\nfive\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text(initial, encoding="utf-8")

        # Turn 1: two reads (top + tail), no writes.
        messages = [user_message("Inspect the file in two windows.")]
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the top."],
                    tool_calls=[ToolCallSpec(arguments=_read_args(target, 1), call_id="call-read-1", name="file_tool")],
                    input_tokens=500,
                ),
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the tail."],
                    tool_calls=[ToolCallSpec(arguments=_read_args(target, 3), call_id="call-read-2", name="file_tool")],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["I inspected it."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(web_harness, authed_client, conversation_uuid, messages)
        snapshot_uuid = record.snapshot_uuid
        turn1 = await _get_turn_execution(web_harness, conversation_uuid, record.bot_message_source_id)
        live_after_turn1 = _live_windows_for_path(turn1.items, target)
        assert len(live_after_turn1) == 2
        assert all(
            (item.prokaryotes_annotations or {}).get("file_tool.revision") == _hash(initial)
            for item in live_after_turn1
        )
        joined = "\n".join(item.output for item in live_after_turn1)
        assert "1 | one" in joined
        assert "3 | three" in joined

        # Turn 2: in-tool write. Live windows should reflect the new revision.
        messages.append(user_message("Update the middle lines."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Editing the file."],
                    tool_calls=[ToolCallSpec(
                        arguments=_replace_args(target, _hash(initial), 2, 3, "TWO\nTHREE_X\n"),
                        call_id="call-write-1",
                        name="file_tool",
                    )],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["It is updated."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        snapshot_uuid = record.snapshot_uuid

        # Turn 3: ack. Stream input shows turn-1's live windows refreshed to the post-write revision — historical
        # TurnExecutions are immutable in ES, but reconcile mutates them in memory before projection.
        messages.append(user_message("Confirm the update landed."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Confirmed."], stop_reason="end_turn", input_tokens=500)])
        )
        record = await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        snapshot_uuid = record.snapshot_uuid
        read_outputs_t3 = _outputs_by_call_id(
            _last_stream_items(web_harness), {"call-read-1", "call-read-2"}
        )
        assert len(read_outputs_t3) == 2
        joined_t3 = "\n".join(read_outputs_t3.values())
        assert f"revision={_hash(after_write)}" in joined_t3
        assert "status=live" in joined_t3
        assert "2 | TWO" in joined_t3
        assert "3 | THREE_X" in joined_t3
        assert "3 | three" not in joined_t3

        # Turn 4: external edit between turns. Live windows refresh again.
        target.write_text(after_external_edit, encoding="utf-8")
        messages.append(user_message("Double-check the current file state."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Checked."], stop_reason="end_turn", input_tokens=500)])
        )
        await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        read_outputs_t4 = _outputs_by_call_id(
            _last_stream_items(web_harness), {"call-read-1", "call-read-2"}
        )
        assert len(read_outputs_t4) == 2
        joined_t4 = "\n".join(read_outputs_t4.values())
        assert f"revision={_hash(after_external_edit)}" in joined_t4
        assert "status=live" in joined_t4
        assert "2 | TWO" in joined_t4
        assert "3 | THREE_Y" in joined_t4
        assert "5 | five" in joined_t4
        assert "3 | THREE_X" not in joined_t4


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_live_windows_survive_compaction(web_harness, authed_client, request):
    """Open two live windows, drive compaction, and verify the lifted live windows reappear in the post-compaction
    projection at the current revision."""
    conversation_uuid = str(uuid4())
    initial = "one\ntwo\nthree\nfour\n"
    after_write = "one\nTWO\nTHREE_X\nfour\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text(initial, encoding="utf-8")

        # Turn 1: open two read windows.
        messages = [user_message("Inspect the file in two windows.")]
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the top."],
                    tool_calls=[ToolCallSpec(arguments=_read_args(target, 1), call_id="call-read-1", name="file_tool")],
                    input_tokens=500,
                ),
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the tail."],
                    tool_calls=[ToolCallSpec(arguments=_read_args(target, 3), call_id="call-read-2", name="file_tool")],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["I inspected it."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(web_harness, authed_client, conversation_uuid, messages)
        snapshot_uuid = record.snapshot_uuid

        # Turn 2: plain ack to extend the conversation past the recency tail.
        messages.append(user_message("Acknowledge the inspection."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Acknowledged."], stop_reason="end_turn", input_tokens=500)])
        )
        record = await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        snapshot_uuid = record.snapshot_uuid

        # Turn 3: write + finalize triggers compaction (input_tokens=5000).
        messages.append(user_message("Now update the middle lines."))
        web_harness.llm_client.set_script(
            LLMScript(
                rounds=[
                    LLMRound(
                        stop_reason="tool_use",
                        text_deltas=["Editing the file."],
                        tool_calls=[ToolCallSpec(
                            arguments=_replace_args(target, _hash(initial), 2, 3, "TWO\nTHREE_X\n"),
                            call_id="call-write-1",
                            name="file_tool",
                        )],
                        input_tokens=500,
                    ),
                    LLMRound(text_deltas=["It is updated and compacted."], stop_reason="end_turn", input_tokens=5000),
                ],
                summary_text="LIVE WINDOW SUMMARY",
            )
        )
        record = await post_chat(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        pending_snapshot_uuid = record.snapshot_uuid
        types = event_types(record.events)
        assert "compaction_pending" in types
        assert types[-1] == "compaction_pending"
        apply_assignments(messages, record.source_id_assignments)
        echo_assistant(messages, record)
        post_compaction_uuid = await _wait_for_compaction(
            authed_client, conversation_uuid, pending_snapshot_uuid
        )

        cached = Conversation.model_validate_json(
            await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
        )
        assert cached.snapshot_uuid == post_compaction_uuid
        assert cached.parent_snapshot_uuid == pending_snapshot_uuid
        assert cached.ancestor_summaries == ["LIVE WINDOW SUMMARY"]
        assert cached.raw_message_start_index > 0
        # The compactor lifts both open live windows onto the child snapshot. Lift preserves the items at their
        # commit-time annotations; reconcile on the next turn refreshes them to the current revision.
        lifted_live = _live_windows_for_path(cached.lifted_turn_items, target)
        assert len(lifted_live) == 2

        # Turn 4: continue from compacted state. The lifted live windows are included in the projection AFTER
        # reconcile refreshes them to the current revision.
        messages.append(user_message("Continue from the compacted state."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Continuing."], stop_reason="end_turn", input_tokens=500)])
        )
        await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=post_compaction_uuid
        )
        read_outputs_t4 = _outputs_by_call_id(
            _last_stream_items(web_harness), {"call-read-1", "call-read-2"}
        )
        assert len(read_outputs_t4) == 2
        joined_t4 = "\n".join(read_outputs_t4.values())
        assert f"revision={_hash(after_write)}" in joined_t4
        assert "status=live" in joined_t4
        assert "2 | TWO" in joined_t4
        assert "3 | THREE_X" in joined_t4


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_create_file_persists_created_record_across_turns(web_harness, authed_client, request):
    """`create_file` returns a CREATED record; the next turn still sees it as a persistent function_call_output
    entry tagged with the file path."""
    conversation_uuid = str(uuid4())

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "created.txt"

        # Turn 1: create the file.
        messages = [user_message("Create the file.")]
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Creating it."],
                    tool_calls=[ToolCallSpec(
                        arguments=_create_args(target, "alpha\nbeta\n"),
                        call_id="call-create-1",
                        name="file_tool",
                    )],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Created."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(web_harness, authed_client, conversation_uuid, messages)
        snapshot_uuid = record.snapshot_uuid
        assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"

        turn1 = await _get_turn_execution(web_harness, conversation_uuid, record.bot_message_source_id)
        created_item = next(
            item for item in turn1.items
            if item.call_id == "call-create-1" and item.type == "function_call_output"
        )
        assert created_item.output.startswith("CREATED ")
        assert (created_item.prokaryotes_annotations or {}).get("file_tool.path") == _resolved(target)

        # Turn 2: continue. The created record stays attached to the turn and its output is preserved in the
        # projection (frozen historical edit record; not a live window).
        messages.append(user_message("Continue from there."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Continuing."], stop_reason="end_turn", input_tokens=500)])
        )
        await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        outputs_t2 = _outputs_by_call_id(_last_stream_items(web_harness), {"call-create-1"})
        assert outputs_t2["call-create-1"].startswith("CREATED ")
        assert _resolved(target) in outputs_t2["call-create-1"]


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_already_exists_window_normalizes_on_next_turn(web_harness, authed_client, request):
    """`create_file` against a path that exists returns an ALREADY_EXISTS live window; on the next turn it's
    reconciled into a plain FILE view."""
    conversation_uuid = str(uuid4())
    initial = "alpha\nbeta\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "exists.txt"
        target.write_text(initial, encoding="utf-8")

        messages = [user_message("Create the file if needed.")]
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Trying to create it."],
                    tool_calls=[ToolCallSpec(
                        arguments=_create_args(target, "ignored\n"),
                        call_id="call-exists-1",
                        name="file_tool",
                    )],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["I checked."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(web_harness, authed_client, conversation_uuid, messages)
        snapshot_uuid = record.snapshot_uuid

        turn1 = await _get_turn_execution(web_harness, conversation_uuid, record.bot_message_source_id)
        already_exists = next(
            item for item in turn1.items
            if item.call_id == "call-exists-1" and item.type == "function_call_output"
        )
        assert already_exists.output.startswith("ALREADY_EXISTS ")
        ann = already_exists.prokaryotes_annotations or {}
        assert ann.get("file_tool.status") == "live"
        assert ann.get("file_tool.revision") == _hash(initial)

        messages.append(user_message("Continue."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Done."], stop_reason="end_turn", input_tokens=500)])
        )
        await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        outputs = _outputs_by_call_id(_last_stream_items(web_harness), {"call-exists-1"})
        normalized = outputs["call-exists-1"]
        assert normalized.startswith("FILE ")
        assert "ALREADY_EXISTS " not in normalized
        assert "1 | alpha" in normalized


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_conflict_window_normalizes_on_next_turn(web_harness, authed_client, request):
    """An expected-revision mismatch produces a CONFLICT live window; the next turn reconciles it into a plain FILE
    view at the current revision."""
    conversation_uuid = str(uuid4())
    initial = "alpha\nbeta\n"
    after_external_edit = "ALPHA\nbeta\ngamma\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text(initial, encoding="utf-8")

        # Turn 1: read to seed the revision.
        messages = [user_message("Read the file.")]
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading it."],
                    tool_calls=[ToolCallSpec(arguments=_read_args(target, 1), call_id="call-read-1", name="file_tool")],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Read."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(web_harness, authed_client, conversation_uuid, messages)
        snapshot_uuid = record.snapshot_uuid

        # External edit between turns invalidates the stale revision.
        target.write_text(after_external_edit, encoding="utf-8")

        # Turn 2: write against the old revision → CONFLICT.
        messages.append(user_message("Apply the original edit."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Trying the edit."],
                    tool_calls=[ToolCallSpec(
                        arguments=_replace_args(target, _hash(initial), 1, 1, "alpha-updated\n"),
                        call_id="call-conflict-1",
                        name="file_tool",
                    )],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Handled."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        snapshot_uuid = record.snapshot_uuid

        turn2 = await _get_turn_execution(web_harness, conversation_uuid, record.bot_message_source_id)
        conflict = next(
            item for item in turn2.items
            if item.call_id == "call-conflict-1" and item.type == "function_call_output"
        )
        assert conflict.output.startswith("CONFLICT ")
        ann = conflict.prokaryotes_annotations or {}
        assert ann.get("file_tool.status") == "live"
        assert ann.get("file_tool.revision") == _hash(after_external_edit)

        # Turn 3: continue. CONFLICT entry normalizes to plain FILE in the projection.
        messages.append(user_message("Continue."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Done."], stop_reason="end_turn", input_tokens=500)])
        )
        await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        outputs = _outputs_by_call_id(
            _last_stream_items(web_harness), {"call-read-1", "call-conflict-1"}
        )
        assert len(outputs) == 2
        joined = "\n".join(outputs.values())
        assert f"revision={_hash(after_external_edit)}" in joined
        assert "status=live" in joined
        assert "CONFLICT " not in joined
        assert "3 | gamma" in joined


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_range_error_window_normalizes_on_next_turn(web_harness, authed_client, request):
    """An out-of-range write produces a RANGE_ERROR live window; on the next turn it normalizes to a plain FILE
    view."""
    conversation_uuid = str(uuid4())
    initial = "a\nb\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text(initial, encoding="utf-8")

        messages = [user_message("Read the file.")]
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading it."],
                    tool_calls=[ToolCallSpec(arguments=_read_args(target, 1), call_id="call-read-1", name="file_tool")],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Read."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(web_harness, authed_client, conversation_uuid, messages)
        snapshot_uuid = record.snapshot_uuid

        messages.append(user_message("Replace the nonexistent range."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Trying the range."],
                    tool_calls=[ToolCallSpec(
                        arguments=_replace_args(target, _hash(initial), 5, 9, "X\n"),
                        call_id="call-range-1",
                        name="file_tool",
                    )],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Handled."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        snapshot_uuid = record.snapshot_uuid

        turn2 = await _get_turn_execution(web_harness, conversation_uuid, record.bot_message_source_id)
        range_error = next(
            item for item in turn2.items
            if item.call_id == "call-range-1" and item.type == "function_call_output"
        )
        assert range_error.output.startswith("RANGE_ERROR ")
        ann = range_error.prokaryotes_annotations or {}
        assert ann.get("file_tool.status") == "live"
        assert ann.get("file_tool.revision") == _hash(initial)

        messages.append(user_message("Continue."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Done."], stop_reason="end_turn", input_tokens=500)])
        )
        await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        outputs = _outputs_by_call_id(_last_stream_items(web_harness), {"call-range-1"})
        normalized = outputs["call-range-1"]
        assert normalized.startswith("FILE ")
        assert "RANGE_ERROR " not in normalized


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_missing_file_tombstones_on_next_turn(web_harness, authed_client, request):
    """If a tracked file is removed between turns, reconcile tombstones the earlier live window with a stale-status
    diagnostic mentioning FileNotFoundError."""
    conversation_uuid = str(uuid4())

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text("here today\n", encoding="utf-8")

        messages = [user_message("Read the file.")]
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading it."],
                    tool_calls=[ToolCallSpec(arguments=_read_args(target, 1), call_id="call-read-1", name="file_tool")],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["Read."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(web_harness, authed_client, conversation_uuid, messages)
        snapshot_uuid = record.snapshot_uuid

        target.unlink()

        messages.append(user_message("Continue."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Done."], stop_reason="end_turn", input_tokens=500)])
        )
        await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        outputs = _outputs_by_call_id(_last_stream_items(web_harness), {"call-read-1"})
        tombstone = outputs["call-read-1"]
        assert "status=stale" in tombstone
        assert "no longer accessible" in tombstone
        assert "FileNotFoundError" in tombstone


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_file_tool_symlink_escape_tombstones_on_next_turn(web_harness, authed_client, request):
    """If a tracked path is replaced with a symlink pointing outside the workspace, reconcile tombstones the live
    window with a stale-status diagnostic mentioning ValueError, and the off-tree contents stay out of the rendered
    output."""
    conversation_uuid = str(uuid4())

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        workspace = Path(temp_dir)
        target = workspace / "tracked.txt"
        target.write_text("inside\n", encoding="utf-8")
        with TemporaryDirectory(prefix=".file-tool-outside-") as outside_dir:
            outside = Path(outside_dir) / "outside.txt"
            outside.write_text("top secret\n", encoding="utf-8")

            messages = [user_message("Read the file.")]
            web_harness.llm_client.set_script(
                LLMScript(rounds=[
                    LLMRound(
                        stop_reason="tool_use",
                        text_deltas=["Reading it."],
                        tool_calls=[ToolCallSpec(
                            arguments=_read_args(target, 1), call_id="call-read-1", name="file_tool"
                        )],
                        input_tokens=500,
                    ),
                    LLMRound(text_deltas=["Read."], stop_reason="end_turn", input_tokens=500),
                ])
            )
            record = await _advance_turn(web_harness, authed_client, conversation_uuid, messages)
            snapshot_uuid = record.snapshot_uuid

            target.unlink()
            target.symlink_to(outside)

            messages.append(user_message("Continue."))
            web_harness.llm_client.set_script(
                LLMScript(rounds=[LLMRound(text_deltas=["Done."], stop_reason="end_turn", input_tokens=500)])
            )
            await _advance_turn(
                web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
            )
            outputs = _outputs_by_call_id(_last_stream_items(web_harness), {"call-read-1"})
            tombstone = outputs["call-read-1"]
            assert "status=stale" in tombstone
            assert "no longer accessible" in tombstone
            assert "ValueError" in tombstone
            assert "top secret" not in tombstone


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_compaction_summary_input_strips_live_window_bodies(web_harness, authed_client, request):
    """The compaction summarizer must not feed live-window bodies to the LLM: the rendered input replaces them with
    a placeholder and keeps only the frozen edit records' context-before/after blocks."""
    conversation_uuid = str(uuid4())
    initial = "one\ntwo\nthree\nfour\n"

    with TemporaryDirectory(dir=Path.cwd(), prefix=".file-tool-it-") as temp_dir:
        target = Path(temp_dir) / "tracked.txt"
        target.write_text(initial, encoding="utf-8")

        # Turn 1: open two live windows.
        messages = [user_message("Inspect the file in two windows.")]
        web_harness.llm_client.set_script(
            LLMScript(rounds=[
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the top."],
                    tool_calls=[ToolCallSpec(arguments=_read_args(target, 1), call_id="call-read-1", name="file_tool")],
                    input_tokens=500,
                ),
                LLMRound(
                    stop_reason="tool_use",
                    text_deltas=["Reading the tail."],
                    tool_calls=[ToolCallSpec(arguments=_read_args(target, 3), call_id="call-read-2", name="file_tool")],
                    input_tokens=500,
                ),
                LLMRound(text_deltas=["I inspected it."], stop_reason="end_turn", input_tokens=500),
            ])
        )
        record = await _advance_turn(web_harness, authed_client, conversation_uuid, messages)
        snapshot_uuid = record.snapshot_uuid

        # Turn 2: ack to extend.
        messages.append(user_message("Acknowledge the inspection."))
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Acknowledged."], stop_reason="end_turn", input_tokens=500)])
        )
        record = await _advance_turn(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        snapshot_uuid = record.snapshot_uuid

        # Turn 3: write + trigger compaction.
        messages.append(user_message("Now update the middle lines."))
        web_harness.llm_client.set_script(
            LLMScript(
                rounds=[
                    LLMRound(
                        stop_reason="tool_use",
                        text_deltas=["Editing the file."],
                        tool_calls=[ToolCallSpec(
                            arguments=_replace_args(target, _hash(initial), 2, 3, "TWO\nTHREE_X\n"),
                            call_id="call-write-1",
                            name="file_tool",
                        )],
                        input_tokens=500,
                    ),
                    LLMRound(text_deltas=["It is updated and compacted."], stop_reason="end_turn", input_tokens=5000),
                ],
                summary_text="LIVE WINDOW SUMMARY",
            )
        )
        record = await post_chat(
            web_harness, authed_client, conversation_uuid, messages, snapshot_uuid=snapshot_uuid
        )
        pending_snapshot_uuid = record.snapshot_uuid
        types = event_types(record.events)
        assert "compaction_pending" in types
        assert types[-1] == "compaction_pending"
        apply_assignments(messages, record.source_id_assignments)
        echo_assistant(messages, record)
        await _wait_for_compaction(authed_client, conversation_uuid, pending_snapshot_uuid)

        # Inspect the summarization input: live windows in the pre-tail bot turn are replaced by a placeholder so
        # the summarizer doesn't burn tokens on contents that subsequent turns track via the live-window mechanism.
        # The write itself (turn 3) lives in the recency tail with COMPACTION_RECENCY_TAIL=2 and is not part of this
        # input.
        summary_items = _last_complete_items(web_harness)
        rendered = "\n".join(item.output or item.content or "" for item in summary_items)
        placeholder = (
            f"[Live tracked file: {_resolved(target)} — current contents are tracked via the "
            "live-window mechanism on subsequent turns, not summarized here.]"
        )
        assert placeholder in rendered
        # Live-window bodies must not leak in: no rendered view headers, no status=live tags, no inline
        # line-numbered content for the tracked path.
        assert "Current view (" not in rendered
        assert f"FILE path={_resolved(target)} revision=" not in rendered
        assert "status=live" not in rendered
        assert "1 | one" not in rendered
        assert "3 | three" not in rendered
