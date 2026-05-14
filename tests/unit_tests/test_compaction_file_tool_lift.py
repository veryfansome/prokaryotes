from types import SimpleNamespace

import pytest

import prokaryotes.web_v1 as web_v1_module
from prokaryotes.api_v1.models import ContextPartition, ContextPartitionItem
from prokaryotes.utils_v1.llm_utils import COMPACTION_RECENCY_TAIL
from prokaryotes.web_v1 import (
    _is_live_window,
    _items_equal_mod_live_windows,
    _lift_active_live_windows,
    _strip_live_window_bodies,
)
from tests.unit_tests.context_partition_utils import (
    FakeSearchClient,
    make_doc,
    make_message_items,
    make_web_base,
)


def _live_window(
        call_id: str,
        path: str,
        view_start: int = 1,
        view_end: int = 5,
        requested_end_line: int | None = None,
) -> ContextPartitionItem:
    annotations = {
        "file_tool.path": path,
        "file_tool.revision": "rev1",
        "file_tool.status": "live",
        "file_tool.view_start_line": str(view_start),
        "file_tool.view_end_line": str(view_end),
    }
    if requested_end_line is not None:
        annotations["file_tool.requested_end_line"] = str(requested_end_line)
    return ContextPartitionItem(
        call_id=call_id,
        output=f"FILE path={path} revision=rev1 status=live lines={view_start}-{view_end} line_count={view_end}",
        type="function_call_output",
        prokaryotes_annotations=annotations,
    )


def _function_call(call_id: str, path: str, action: str = "read_lines") -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        id=call_id,
        name="file_tool",
        arguments=f'{{"action":"{action}","path":"{path}"}}',
        type="function_call",
        status="completed",
    )


def _edit_record(call_id: str, path: str) -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        output=f"EDITED path={path} action=replace_lines",
        type="function_call_output",
        prokaryotes_annotations={"file_tool.path": path},
    )


def _stale_window(call_id: str, path: str) -> ContextPartitionItem:
    item = _live_window(call_id, path)
    item.prokaryotes_annotations["file_tool.status"] = "stale"
    item.output = f"FILE path={path} status=stale [no longer accessible: FileNotFoundError]"
    return item


def test_lift_skips_when_no_active_paths_in_tail():
    pre_tail = [
        _function_call("c1", "/app/foo.py"),
        _live_window("c1", "/app/foo.py"),
    ]
    recency_tail = make_message_items(("user", "next request"), ("assistant", "ok"))

    augmented = _lift_active_live_windows(pre_tail, recency_tail)

    assert augmented == recency_tail


def test_lift_inserts_pair_before_first_annotated_tail_item():
    pre_tail = [
        _function_call("c1", "/app/foo.py"),
        _live_window("c1", "/app/foo.py"),
    ]
    user_msg, assistant_msg = make_message_items(("user", "edit foo"), ("assistant", "on it"))
    edit_call = _function_call("c2", "/app/foo.py", action="replace_lines")
    edit_output = _edit_record("c2", "/app/foo.py")
    recency_tail = [user_msg, assistant_msg, edit_call, edit_output]

    augmented = _lift_active_live_windows(pre_tail, recency_tail)

    assert augmented == [user_msg, assistant_msg] + pre_tail + [edit_call, edit_output]


def test_lift_inserts_before_start_of_multitool_tail_round():
    pre_tail = [
        _function_call("c1", "/app/foo.py"),
        _live_window("c1", "/app/foo.py"),
    ]
    user_msg, = make_message_items(("user", "edit foo and inspect shell output"))
    shell_call = ContextPartitionItem(
        call_id="shell",
        id="shell",
        name="shell_command",
        arguments="{}",
        type="function_call",
        status="completed",
    )
    edit_call = _function_call("c2", "/app/foo.py", action="replace_lines")
    shell_output = ContextPartitionItem(
        call_id="shell",
        output="shell output",
        type="function_call_output",
    )
    edit_output = _edit_record("c2", "/app/foo.py")
    recency_tail = [user_msg, shell_call, edit_call, shell_output, edit_output]

    augmented = _lift_active_live_windows(pre_tail, recency_tail)

    assert augmented == [user_msg] + pre_tail + [shell_call, edit_call, shell_output, edit_output]
    _system, messages = ContextPartition(conversation_uuid="conv", items=augmented).to_anthropic_messages()
    assistant_tool_messages = [message for message in messages if message["role"] == "assistant"]
    assert [block["id"] for block in assistant_tool_messages[0]["content"]] == ["c1"]
    assert [block["id"] for block in assistant_tool_messages[1]["content"]] == ["shell", "c2"]


def test_lift_handles_multiple_paths_and_preserves_chronological_order():
    foo_call = _function_call("c1", "/app/foo.py")
    foo_window = _live_window("c1", "/app/foo.py")
    bar_call = _function_call("c2", "/app/bar.py")
    bar_window = _live_window("c2", "/app/bar.py")
    pre_tail = [foo_call, foo_window, bar_call, bar_window]

    user_msg, = make_message_items(("user", "more work"))
    foo_edit_call = _function_call("c3", "/app/foo.py", action="replace_lines")
    foo_edit_output = _edit_record("c3", "/app/foo.py")
    bar_edit_call = _function_call("c4", "/app/bar.py", action="replace_lines")
    bar_edit_output = _edit_record("c4", "/app/bar.py")
    recency_tail = [user_msg, foo_edit_call, foo_edit_output, bar_edit_call, bar_edit_output]

    augmented = _lift_active_live_windows(pre_tail, recency_tail)

    assert augmented == [
        user_msg,
        foo_call, foo_window,
        bar_call, bar_window,
        foo_edit_call, foo_edit_output,
        bar_edit_call, bar_edit_output,
    ]


def test_lift_ignores_pre_tail_live_windows_for_inactive_paths():
    foo_call = _function_call("c1", "/app/foo.py")
    foo_window = _live_window("c1", "/app/foo.py")
    bar_call = _function_call("c2", "/app/bar.py")
    bar_window = _live_window("c2", "/app/bar.py")
    pre_tail = [foo_call, foo_window, bar_call, bar_window]

    user_msg, = make_message_items(("user", "tweak foo"))
    foo_edit_call = _function_call("c3", "/app/foo.py", action="replace_lines")
    foo_edit_output = _edit_record("c3", "/app/foo.py")
    recency_tail = [user_msg, foo_edit_call, foo_edit_output]

    augmented = _lift_active_live_windows(pre_tail, recency_tail)

    assert bar_call not in augmented
    assert bar_window not in augmented
    assert foo_call in augmented
    assert foo_window in augmented


def test_lift_skips_pre_tail_edit_records_even_when_path_is_active():
    pre_tail = [
        _function_call("c1", "/app/foo.py", action="replace_lines"),
        _edit_record("c1", "/app/foo.py"),
    ]
    user_msg, = make_message_items(("user", "next"))
    edit_call = _function_call("c2", "/app/foo.py", action="replace_lines")
    edit_output = _edit_record("c2", "/app/foo.py")
    recency_tail = [user_msg, edit_call, edit_output]

    augmented = _lift_active_live_windows(pre_tail, recency_tail)

    assert augmented == recency_tail


def test_lift_skips_stale_live_windows():
    stale_call = _function_call("c1", "/app/gone.py")
    stale_window = _live_window("c1", "/app/gone.py")
    stale_window.prokaryotes_annotations["file_tool.status"] = "stale"
    stale_window.output = "FILE path=/app/gone.py status=stale [no longer accessible: FileNotFoundError]"
    pre_tail = [stale_call, stale_window]

    user_msg, = make_message_items(("user", "..."))
    edit_call = _function_call("c2", "/app/gone.py", action="replace_lines")
    edit_output = _edit_record("c2", "/app/gone.py")
    recency_tail = [user_msg, edit_call, edit_output]

    augmented = _lift_active_live_windows(pre_tail, recency_tail)

    assert stale_call not in augmented
    assert stale_window not in augmented


def test_lift_does_not_treat_tail_tombstone_as_active_path():
    pre_tail = [
        _function_call("c1", "/app/gone.py"),
        _live_window("c1", "/app/gone.py"),
    ]
    user_msg, = make_message_items(("user", "..."))
    tail_tombstone = _stale_window("c2", "/app/gone.py")
    recency_tail = [user_msg, _function_call("c2", "/app/gone.py"), tail_tombstone]

    augmented = _lift_active_live_windows(pre_tail, recency_tail)

    assert augmented == recency_tail


def test_lift_skips_orphan_function_call_output_without_matching_function_call():
    orphan_window = _live_window("orphan-id", "/app/foo.py")
    pre_tail = [orphan_window]

    user_msg, = make_message_items(("user", "..."))
    edit_call = _function_call("c2", "/app/foo.py", action="replace_lines")
    edit_output = _edit_record("c2", "/app/foo.py")
    recency_tail = [user_msg, edit_call, edit_output]

    augmented = _lift_active_live_windows(pre_tail, recency_tail)

    assert orphan_window not in augmented


def _make_snapshot_with_live_window(
        partition_uuid: str,
        path: str,
        message_count: int,
) -> ContextPartition:
    items: list[ContextPartitionItem] = [
        _function_call("c1", path),
        _live_window("c1", path),
    ]
    for i in range(message_count):
        items.extend(make_message_items(
            ("user", f"U{i}"),
            ("assistant", f"A{i}"),
        ))
    items.append(_function_call("c2", path, action="replace_lines"))
    items.append(_edit_record("c2", path))
    return ContextPartition(
        conversation_uuid="conv",
        partition_uuid=partition_uuid,
        items=items,
    )


def _make_snapshot_with_inactive_live_window(
        partition_uuid: str,
        path: str,
        message_count: int,
) -> ContextPartition:
    items: list[ContextPartitionItem] = [
        _function_call("c1", path),
        _live_window("c1", path),
    ]
    for i in range(message_count):
        items.extend(make_message_items(
            ("user", f"U{i}"),
            ("assistant", f"A{i}"),
        ))
    return ContextPartition(
        conversation_uuid="conv",
        partition_uuid=partition_uuid,
        items=items,
    )


@pytest.mark.asyncio
async def test_compact_partition_lifts_active_pretail_live_windows_into_new_tail():
    snapshot = _make_snapshot_with_live_window(
        partition_uuid="snap",
        path="/app/foo.py",
        message_count=COMPACTION_RECENCY_TAIL + 2,
    )
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(
        redis_data={"context_partition:conv": snapshot.model_dump_json()},
        search_client=search,
    )
    lock_key = "compaction_lock:conv"
    await wb.redis_client.set(lock_key, "1")

    async def compact_fn(_):
        return "S1"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await wb.redis_client.get("context_partition:conv"))
    cached_call_ids = [item.call_id for item in cached.items]
    assert "c1" in cached_call_ids
    edit_call_idx = cached_call_ids.index("c2")
    lifted_call_idx = cached_call_ids.index("c1")
    assert lifted_call_idx < edit_call_idx
    assert cached.items[0].type == "message"
    assert cached.items[0].role == "user"
    assert search.docs["snap"]["is_compacted"] is True
    assert search.docs["snap"]["summary"] == "S1"


@pytest.mark.asyncio
async def test_compact_partition_lifts_windows_for_paths_active_only_after_snapshot():
    path = "/app/foo.py"
    snapshot = _make_snapshot_with_inactive_live_window(
        partition_uuid="snap",
        path=path,
        message_count=COMPACTION_RECENCY_TAIL + 2,
    )
    current = ContextPartition.model_validate_json(snapshot.model_dump_json())
    current.items.extend([
        _function_call("c2", path, action="replace_lines"),
        _edit_record("c2", path),
    ])

    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(
        redis_data={"context_partition:conv": current.model_dump_json()},
        search_client=search,
    )
    lock_key = "compaction_lock:conv"
    await wb.redis_client.set(lock_key, "1")

    async def compact_fn(_):
        return "S1"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await wb.redis_client.get("context_partition:conv"))
    cached_call_ids = [item.call_id for item in cached.items]
    assert "c1" in cached_call_ids
    assert cached_call_ids.index("c1") < cached_call_ids.index("c2")
    assert cached.items[0].type == "message"
    assert cached.items[0].role == "user"


def test_items_equal_mod_live_windows_handles_empty_lists():
    assert _items_equal_mod_live_windows([], []) is True


def test_items_equal_mod_live_windows_returns_false_on_length_mismatch():
    item = _live_window("c1", "/app/foo.py")
    assert _items_equal_mod_live_windows([item], []) is False


def test_items_equal_mod_live_windows_treats_refreshed_window_as_equal():
    a = _live_window("c1", "/app/foo.py", view_start=1, view_end=5)
    b = _live_window("c1", "/app/foo.py", view_start=1, view_end=8)
    b.prokaryotes_annotations["file_tool.revision"] = "rev2"
    b.output = "FILE path=/app/foo.py revision=rev2 status=live lines=1-8 line_count=8"

    assert _items_equal_mod_live_windows([a], [b]) is True


def test_items_equal_mod_live_windows_detects_tombstone_transition():
    live = _live_window("c1", "/app/gone.py")
    stale = _live_window("c1", "/app/gone.py")
    stale.prokaryotes_annotations["file_tool.status"] = "stale"
    stale.output = "FILE path=/app/gone.py status=stale [no longer accessible: FileNotFoundError]"

    assert _items_equal_mod_live_windows([live], [stale]) is False


def test_items_equal_mod_live_windows_detects_path_change_within_live_window():
    a = _live_window("c1", "/app/foo.py")
    b = _live_window("c1", "/app/bar.py")

    assert _items_equal_mod_live_windows([a], [b]) is False


def test_items_equal_mod_live_windows_detects_requested_end_line_change():
    a = _live_window("c1", "/app/foo.py", view_start=1, view_end=3, requested_end_line=3)
    b = _live_window("c1", "/app/foo.py", view_start=1, view_end=3, requested_end_line=5)

    assert _items_equal_mod_live_windows([a], [b]) is False


def test_items_equal_mod_live_windows_falls_back_to_full_equality_for_non_live_items():
    msg_a = ContextPartitionItem(role="user", content="hi")
    msg_b = ContextPartitionItem(role="user", content="hi")
    msg_c = ContextPartitionItem(role="user", content="bye")
    edit_a = _edit_record("c1", "/app/foo.py")
    edit_b = _edit_record("c1", "/app/foo.py")

    assert _items_equal_mod_live_windows([msg_a, edit_a], [msg_b, edit_b]) is True
    assert _items_equal_mod_live_windows([msg_a], [msg_c]) is False


def test_items_equal_mod_live_windows_one_side_live_other_side_not_equal():
    live = _live_window("c1", "/app/foo.py")
    edit = _edit_record("c1", "/app/foo.py")

    assert _items_equal_mod_live_windows([live], [edit]) is False


@pytest.mark.asyncio
async def test_compact_partition_swap_proceeds_after_concurrent_live_window_refresh():
    snapshot = _make_snapshot_with_live_window(
        partition_uuid="snap",
        path="/app/foo.py",
        message_count=COMPACTION_RECENCY_TAIL + 2,
    )

    current = ContextPartition.model_validate_json(snapshot.model_dump_json())
    refreshed_count = 0
    for item in current.items:
        if _is_live_window(item):
            item.prokaryotes_annotations["file_tool.revision"] = "rev2"
            item.prokaryotes_annotations["file_tool.view_end_line"] = "9"
            item.output = "FILE path=/app/foo.py revision=rev2 status=live lines=1-9 line_count=9"
            refreshed_count += 1
    assert refreshed_count == 1

    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(
        redis_data={"context_partition:conv": current.model_dump_json()},
        search_client=search,
    )
    lock_key = "compaction_lock:conv"
    await wb.redis_client.set(lock_key, "1")

    async def compact_fn(_):
        return "S1"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await wb.redis_client.get("context_partition:conv"))
    assert cached.partition_uuid != "snap"
    assert cached.parent_partition_uuid == "snap"
    assert cached.ancestor_summaries == ["S1"]


def _conflict_live_window(call_id: str, path: str) -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        output=(
            f"CONFLICT path={path} expected_revision=oldrev current_revision=newrev\n"
            "The file changed since the revision returned by read_lines. Use the current view before retrying.\n"
            "Current view (lines 1-3 of 3):\n"
            "1 | actual\n"
            "2 | file\n"
            "3 | content"
        ),
        type="function_call_output",
        prokaryotes_annotations={
            "file_tool.path": path,
            "file_tool.revision": "newrev",
            "file_tool.status": "live",
            "file_tool.view_start_line": "1",
            "file_tool.view_end_line": "3",
        },
    )


def _already_exists_live_window(call_id: str, path: str) -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        output=(
            f"ALREADY_EXISTS path={path} current_revision=rev1\n"
            "The file already exists. Read or edit the existing file instead.\n"
            "Current view (lines 1-2 of 2):\n"
            "1 | first\n"
            "2 | second"
        ),
        type="function_call_output",
        prokaryotes_annotations={
            "file_tool.path": path,
            "file_tool.revision": "rev1",
            "file_tool.status": "live",
            "file_tool.view_start_line": "1",
            "file_tool.view_end_line": "2",
        },
    )


def _range_error_live_window(call_id: str, path: str) -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        output=(
            f"RANGE_ERROR path={path} action=replace_lines current_revision=rev1\n"
            "Requested line range is out of bounds for line_count=2.\n"
            "Current view (lines 1-2 of 2):\n"
            "1 | first\n"
            "2 | second"
        ),
        type="function_call_output",
        prokaryotes_annotations={
            "file_tool.path": path,
            "file_tool.revision": "rev1",
            "file_tool.status": "live",
            "file_tool.view_start_line": "1",
            "file_tool.view_end_line": "2",
        },
    )


def _range_truncated_live_window(call_id: str, path: str) -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        output=(
            f"RANGE_TRUNCATED path={path} requested_lines=1-255 returned_lines=1-200 line_count=255\n"
            "Your requested span exceeded the 200-line per-call cap."
            " The window below covers lines 1-200."
            " Call `read_lines` with `start_line=201` to page through the remaining 55 lines.\n"
            "Current view (lines 1-200 of 255):\n"
            "1 | first\n"
            "2 | second"
        ),
        type="function_call_output",
        prokaryotes_annotations={
            "file_tool.path": path,
            "file_tool.revision": "rev1",
            "file_tool.status": "live",
            "file_tool.view_start_line": "1",
            "file_tool.view_end_line": "200",
            "file_tool.requested_end_line": "200",
        },
    )


def _empty_conflict_live_window(call_id: str, path: str) -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        output=(
            f"CONFLICT path={path} expected_revision=oldrev current_revision=emptyrev\n"
            "The file changed since the revision returned by read_lines. Use the current view before retrying.\n"
            "Current view: empty file (line_count=0)"
        ),
        type="function_call_output",
        prokaryotes_annotations={
            "file_tool.path": path,
            "file_tool.revision": "emptyrev",
            "file_tool.status": "live",
            "file_tool.view_start_line": "1",
            "file_tool.view_end_line": "0",
        },
    )


def test_strip_live_window_bodies_replaces_ordinary_live_window_wholesale():
    partition = ContextPartition(
        conversation_uuid="conv",
        items=[_function_call("c1", "/app/foo.py"), _live_window("c1", "/app/foo.py")],
    )

    stripped = _strip_live_window_bodies(partition)

    placeholder = stripped.items[1].output
    assert placeholder.startswith("[Live tracked file: /app/foo.py")
    assert "1 | " not in placeholder
    assert "FILE path=/app/foo.py revision=" not in placeholder


def test_strip_live_window_bodies_preserves_conflict_diagnostic_header():
    partition = ContextPartition(
        conversation_uuid="conv",
        items=[
            _function_call("c1", "/app/foo.py", action="replace_lines"),
            _conflict_live_window("c1", "/app/foo.py"),
        ],
    )

    stripped = _strip_live_window_bodies(partition)
    output = stripped.items[1].output

    assert output.startswith(
        "CONFLICT path=/app/foo.py expected_revision=oldrev current_revision=newrev"
    )
    assert "The file changed since the revision returned by read_lines." in output
    assert "[Live tracked file: /app/foo.py" in output
    assert "1 | actual" not in output
    assert "Current view" not in output


def test_strip_live_window_bodies_preserves_already_exists_diagnostic_header():
    partition = ContextPartition(
        conversation_uuid="conv",
        items=[
            _function_call("c1", "/app/foo.py", action="create_file"),
            _already_exists_live_window("c1", "/app/foo.py"),
        ],
    )

    stripped = _strip_live_window_bodies(partition)
    output = stripped.items[1].output

    assert output.startswith("ALREADY_EXISTS path=/app/foo.py current_revision=rev1")
    assert "The file already exists. Read or edit the existing file instead." in output
    assert "[Live tracked file: /app/foo.py" in output
    assert "1 | first" not in output
    assert "Current view" not in output


def test_strip_live_window_bodies_preserves_range_error_diagnostic_header():
    partition = ContextPartition(
        conversation_uuid="conv",
        items=[
            _function_call("c1", "/app/foo.py", action="replace_lines"),
            _range_error_live_window("c1", "/app/foo.py"),
        ],
    )

    stripped = _strip_live_window_bodies(partition)
    output = stripped.items[1].output

    assert output.startswith(
        "RANGE_ERROR path=/app/foo.py action=replace_lines current_revision=rev1"
    )
    assert "Requested line range is out of bounds" in output
    assert "[Live tracked file: /app/foo.py" in output
    assert "1 | first" not in output


def test_strip_live_window_bodies_preserves_range_truncated_diagnostic_header():
    partition = ContextPartition(
        conversation_uuid="conv",
        items=[
            _function_call("c1", "/app/foo.py", action="read_lines"),
            _range_truncated_live_window("c1", "/app/foo.py"),
        ],
    )

    stripped = _strip_live_window_bodies(partition)
    output = stripped.items[1].output

    assert output.startswith(
        "RANGE_TRUNCATED path=/app/foo.py requested_lines=1-255 returned_lines=1-200 line_count=255"
    )
    assert "Your requested span exceeded the 200-line per-call cap." in output
    assert "Call `read_lines` with `start_line=201`" in output
    assert "[Live tracked file: /app/foo.py" in output
    assert "1 | first" not in output
    assert "Current view" not in output


def test_strip_live_window_bodies_handles_empty_current_view_marker():
    partition = ContextPartition(
        conversation_uuid="conv",
        items=[
            _function_call("c1", "/app/empty.py", action="replace_lines"),
            _empty_conflict_live_window("c1", "/app/empty.py"),
        ],
    )

    stripped = _strip_live_window_bodies(partition)
    output = stripped.items[1].output

    assert output.startswith(
        "CONFLICT path=/app/empty.py expected_revision=oldrev current_revision=emptyrev"
    )
    assert "The file changed since the revision returned by read_lines." in output
    assert "[Live tracked file: /app/empty.py" in output
    assert "Current view" not in output
    assert "line_count=0" not in output


def test_strip_live_window_bodies_preserves_edit_records_and_tombstones_and_messages():
    edit = _edit_record("c1", "/app/foo.py")
    edit_call = _function_call("c1", "/app/foo.py", action="replace_lines")
    stale = _stale_window("c2", "/app/gone.py")
    stale_call = _function_call("c2", "/app/gone.py")
    msg = ContextPartitionItem(role="user", content="please edit foo")
    partition = ContextPartition(
        conversation_uuid="conv",
        items=[msg, edit_call, edit, stale_call, stale],
    )

    stripped = _strip_live_window_bodies(partition)

    assert stripped.items[0].content == "please edit foo"
    assert stripped.items[2].output == edit.output
    assert stripped.items[2].prokaryotes_annotations == edit.prokaryotes_annotations
    assert stripped.items[4].output == stale.output
    assert stripped.items[4].prokaryotes_annotations["file_tool.status"] == "stale"


def test_strip_live_window_bodies_does_not_mutate_input_partition():
    live = _live_window("c1", "/app/foo.py")
    original_output = live.output
    original_revision = live.prokaryotes_annotations["file_tool.revision"]
    partition = ContextPartition(
        conversation_uuid="conv",
        items=[_function_call("c1", "/app/foo.py"), live],
    )

    _strip_live_window_bodies(partition)

    assert partition.items[1].output == original_output
    assert partition.items[1].prokaryotes_annotations["file_tool.revision"] == original_revision


def test_strip_live_window_bodies_strips_inactive_paths_too():
    inactive_path = "/app/inactive.py"
    active_path = "/app/active.py"
    partition = ContextPartition(
        conversation_uuid="conv",
        items=[
            _function_call("c1", inactive_path),
            _live_window("c1", inactive_path),
            _function_call("c2", active_path),
            _live_window("c2", active_path),
            _function_call("c3", active_path, action="replace_lines"),
            _edit_record("c3", active_path),
        ],
    )

    stripped = _strip_live_window_bodies(partition)

    assert stripped.items[1].output.startswith(f"[Live tracked file: {inactive_path}")
    assert stripped.items[3].output.startswith(f"[Live tracked file: {active_path}")


@pytest.mark.asyncio
async def test_compact_partition_skips_swap_when_live_window_was_tombstoned():
    snapshot = _make_snapshot_with_live_window(
        partition_uuid="snap",
        path="/app/gone.py",
        message_count=COMPACTION_RECENCY_TAIL + 2,
    )

    current = ContextPartition.model_validate_json(snapshot.model_dump_json())
    for item in current.items:
        if _is_live_window(item):
            item.prokaryotes_annotations["file_tool.status"] = "stale"
            item.output = "FILE path=/app/gone.py status=stale [no longer accessible: FileNotFoundError]"

    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base(
        redis_data={"context_partition:conv": current.model_dump_json()},
        search_client=search,
    )
    lock_key = "compaction_lock:conv"
    await wb.redis_client.set(lock_key, "1")

    async def compact_fn(_):
        return "S1"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await wb.redis_client.get("context_partition:conv"))
    assert cached.partition_uuid == "snap"
    assert cached.parent_partition_uuid is None
    assert cached.ancestor_summaries == []
    assert search.docs["snap"]["is_compacted"] is False
    assert search.docs["snap"]["summary"] is None


@pytest.fixture
def stub_load_session(monkeypatch):
    async def noop(_request):
        return None

    monkeypatch.setattr(web_v1_module, "load_session", noop)


@pytest.mark.asyncio
async def test_get_compaction_status_returns_new_partition_uuid_when_swap_committed(stub_load_session):
    wb = make_web_base(redis_data={"context_partition:conv": ContextPartition(
        conversation_uuid="conv",
        partition_uuid="child",
        parent_partition_uuid="parent",
        items=[],
    ).model_dump_json()})
    request = SimpleNamespace(session={"user_id": "u1"})

    response = await wb.get_compaction_status(
        request=request,
        conversation_uuid="conv",
        pending_partition_uuid="parent",
    )

    assert response == {"done": True, "partition_uuid": "child"}


@pytest.mark.asyncio
async def test_get_compaction_status_reports_in_progress_while_lock_held(stub_load_session):
    wb = make_web_base(redis_data={"compaction_lock:conv": "1"})
    request = SimpleNamespace(session={"user_id": "u1"})

    response = await wb.get_compaction_status(
        request=request,
        conversation_uuid="conv",
        pending_partition_uuid="parent",
    )

    assert response == {"done": False}


@pytest.mark.asyncio
async def test_get_compaction_status_omits_partition_uuid_for_unrelated_changed_uuid(stub_load_session):
    wb = make_web_base(redis_data={"context_partition:conv": ContextPartition(
        conversation_uuid="conv",
        partition_uuid="unrelated",
        parent_partition_uuid="some-other-parent",
        items=[],
    ).model_dump_json()})
    request = SimpleNamespace(session={"user_id": "u1"})

    response = await wb.get_compaction_status(
        request=request,
        conversation_uuid="conv",
        pending_partition_uuid="parent",
    )

    assert response == {"done": True}
    assert "partition_uuid" not in response


@pytest.mark.asyncio
async def test_get_compaction_status_reports_done_when_swap_was_skipped(stub_load_session):
    wb = make_web_base(redis_data={"context_partition:conv": ContextPartition(
        conversation_uuid="conv",
        partition_uuid="parent",
        items=[],
    ).model_dump_json()})
    request = SimpleNamespace(session={"user_id": "u1"})

    response = await wb.get_compaction_status(
        request=request,
        conversation_uuid="conv",
        pending_partition_uuid="parent",
    )

    assert response == {"done": True}
    assert "partition_uuid" not in response
