"""Compaction lift tests: pre-tail live windows for paths active in the recency tail are
lifted into the new tail before the first annotated tail item, with their matching
function_call items kept paired by call_id.

The fakes for Redis/SearchClient/WebBase are inlined here rather than imported from
`tests.context_partition_utils` so the overlay test module remains self-contained under
PYTHONPATH=overlay:. (overlay/tests has no __init__.py at this stage)."""

import pytest

from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    compute_boundary_hash,
    compute_tail_hash,
    conversation_message_items,
)
from prokaryotes.utils_v1.llm_utils import COMPACTION_RECENCY_TAIL
from prokaryotes.web_v1 import (
    WebBase,
    _is_live_window,
    _items_equal_mod_live_windows,
    _lift_active_live_windows,
    _strip_live_window_bodies,
)

# --- Inlined fakes ---

class FakePipeline:
    def __init__(self, redis):
        self.commands = []
        self.redis = redis
        self.watched_key: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self):
        for key, value, ex in self.commands:
            await self.redis.set(key, value, ex=ex)

    async def get(self, key):
        return await self.redis.get(key)

    def multi(self):
        self.commands = []

    async def reset(self):
        self.commands = []

    def set(self, key, value, ex=None):
        self.commands.append((key, value, ex))

    async def watch(self, key: str):
        self.watched_key = key


class FakeRedis:
    def __init__(self, data: dict | None = None):
        self._data: dict = {}
        for key, value in (data or {}).items():
            self._data[key] = value.encode() if isinstance(value, str) else value

    async def delete(self, *keys):
        for key in keys:
            self._data.pop(key, None)

    async def exists(self, key: str) -> int:
        return 1 if key in self._data else 0

    async def get(self, key: str):
        return self._data.get(key)

    def pipeline(self):
        return FakePipeline(self)

    async def set(self, key: str, value, ex=None, nx=False):
        stored_value = value.encode() if isinstance(value, str) else value
        if nx and key in self._data:
            return False
        self._data[key] = stored_value
        return True


class FakeSearchClient:
    def __init__(self, docs=None):
        self.docs = {doc["partition_uuid"]: dict(doc) for doc in (docs or [])}

    async def get_partition(self, partition_uuid: str) -> dict | None:
        return self.docs.get(partition_uuid)

    async def put_partition(self, partition: ContextPartition) -> None:
        self.docs[partition.partition_uuid] = make_doc(partition)

    async def update_partition(self, partition_uuid: str, **fields) -> None:
        self.docs.setdefault(partition_uuid, {"partition_uuid": partition_uuid}).update(fields)


def make_doc(partition: ContextPartition, **overrides):
    message_items = conversation_message_items(partition.items)
    doc = {
        "partition_uuid": partition.partition_uuid,
        "conversation_uuid": partition.conversation_uuid,
        "parent_partition_uuid": partition.parent_partition_uuid,
        "ancestor_summaries": partition.ancestor_summaries,
        "raw_message_start_index": partition.raw_message_start_index,
        "is_compacted": False,
        "summary": None,
        "items_json": partition.model_dump_json(include={"items"}),
        "boundary_message_count": partition.raw_message_start_index + len(message_items),
        "boundary_user_count": sum(1 for item in message_items if item.role == "user"),
        "boundary_hash": compute_boundary_hash(message_items),
        "tail_hash": compute_tail_hash(message_items),
    }
    doc.update(overrides)
    return doc


def make_message_items(*role_contents: tuple[str, str]) -> list[ContextPartitionItem]:
    return [
        ContextPartitionItem(role=role, content=content)
        for role, content in role_contents
    ]


def make_web_base() -> WebBase:
    wb = object.__new__(WebBase)
    wb.background_tasks = set()
    wb.conversation_cache_ex = 3600
    return wb


# --- Test helpers for FileTool annotations ---

def _live_window(call_id: str, path: str, view_start: int = 1, view_end: int = 5) -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        output=f"FILE path={path} revision=rev1 status=live lines={view_start}-{view_end} line_count={view_end}",
        type="function_call_output",
        prokaryotes_annotations={
            "file_tool.path": path,
            "file_tool.revision": "rev1",
            "file_tool.status": "live",
            "file_tool.view_start_line": str(view_start),
            "file_tool.view_end_line": str(view_end),
        },
    )


def _function_call(call_id: str, path: str, action: str = "read") -> ContextPartitionItem:
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


# --- Pure helper tests (`_lift_active_live_windows`) ---

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
    """Only live windows are lifted; pre-tail edit records are summarized like everything else."""
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
    """Tombstoned items (status=stale) are not lifted — they don't represent current file state."""
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


def test_lift_skips_orphan_function_call_output_without_matching_function_call():
    """Defensive: an output without a matching call in pre_tail is skipped to keep the
    (function_call, function_call_output) pairing intact."""
    orphan_window = _live_window("orphan-id", "/app/foo.py")
    pre_tail = [orphan_window]

    user_msg, = make_message_items(("user", "..."))
    edit_call = _function_call("c2", "/app/foo.py", action="replace_lines")
    edit_output = _edit_record("c2", "/app/foo.py")
    recency_tail = [user_msg, edit_call, edit_output]

    augmented = _lift_active_live_windows(pre_tail, recency_tail)

    assert orphan_window not in augmented


# --- Integration test through `_compact_partition` ---

def _make_snapshot_with_live_window(
        partition_uuid: str,
        path: str,
        message_count: int,
) -> ContextPartition:
    """Build a snapshot with a (function_call, live-window) pair followed by enough
    (user, assistant) pairs that the recency tail boundary sits well past the live window,
    plus a tail-active edit pair to mark the path as active."""
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


@pytest.mark.asyncio
async def test_compact_partition_lifts_active_pretail_live_windows_into_new_tail():
    snapshot = _make_snapshot_with_live_window(
        partition_uuid="snap",
        path="/app/foo.py",
        message_count=COMPACTION_RECENCY_TAIL + 2,
    )
    redis = FakeRedis({"context_partition:conv": snapshot.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base()
    wb.search_client = search
    wb.redis_client = redis

    async def compact_fn(_):
        return "S1"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    cached_call_ids = [item.call_id for item in cached.items]
    assert "c1" in cached_call_ids, "lifted function_call missing from new tail"
    edit_call_idx = cached_call_ids.index("c2")
    lifted_call_idx = cached_call_ids.index("c1")
    assert lifted_call_idx < edit_call_idx, "lifted pair must precede the first annotated tail item"
    # Tail must still begin with a user-role message (Anthropic API requirement).
    assert cached.items[0].type == "message"
    assert cached.items[0].role == "user"


# --- Modulo-live-window prefix-check tests (`_items_equal_mod_live_windows`) ---

def test_items_equal_mod_live_windows_handles_empty_lists():
    assert _items_equal_mod_live_windows([], []) is True


def test_items_equal_mod_live_windows_returns_false_on_length_mismatch():
    item = _live_window("c1", "/app/foo.py")
    assert _items_equal_mod_live_windows([item], []) is False


def test_items_equal_mod_live_windows_treats_refreshed_window_as_equal():
    """Same window identity (call_id, path, status, view_start_line), different
    revision/view_end_line/output — the only mutations `_refresh_live_windows`
    performs. Must compare equal so a concurrent reconciliation doesn't falsify
    the compaction prefix check."""
    a = _live_window("c1", "/app/foo.py", view_start=1, view_end=5)
    b = _live_window("c1", "/app/foo.py", view_start=1, view_end=8)
    b.prokaryotes_annotations["file_tool.revision"] = "rev2"
    b.output = "FILE path=/app/foo.py revision=rev2 status=live lines=1-8 line_count=8"

    assert _items_equal_mod_live_windows([a], [b]) is True


def test_items_equal_mod_live_windows_detects_tombstone_transition():
    """Live → stale is a substantive change the model can observe (tombstone vs
    rendered view), so it must surface as inequality even though the call_id and
    path are unchanged."""
    live = _live_window("c1", "/app/gone.py")
    stale = _live_window("c1", "/app/gone.py")
    stale.prokaryotes_annotations["file_tool.status"] = "stale"
    stale.output = "FILE path=/app/gone.py status=stale [no longer accessible: FileNotFoundError]"

    assert _items_equal_mod_live_windows([live], [stale]) is False


def test_items_equal_mod_live_windows_detects_path_change_within_live_window():
    """Different paths under the same call_id is impossible in normal operation
    but proves the helper isn't ignoring identity, only the mutation-fields."""
    a = _live_window("c1", "/app/foo.py")
    b = _live_window("c1", "/app/bar.py")

    assert _items_equal_mod_live_windows([a], [b]) is False


def test_items_equal_mod_live_windows_falls_back_to_full_equality_for_non_live_items():
    """Edit records, function_calls, message items: still compared with full
    Pydantic equality so any change to them causes the swap to skip."""
    msg_a = ContextPartitionItem(role="user", content="hi")
    msg_b = ContextPartitionItem(role="user", content="hi")
    msg_c = ContextPartitionItem(role="user", content="bye")
    edit_a = _edit_record("c1", "/app/foo.py")
    edit_b = _edit_record("c1", "/app/foo.py")

    assert _items_equal_mod_live_windows([msg_a, edit_a], [msg_b, edit_b]) is True
    assert _items_equal_mod_live_windows([msg_a], [msg_c]) is False


def test_items_equal_mod_live_windows_one_side_live_other_side_not_inequal():
    live = _live_window("c1", "/app/foo.py")
    edit = _edit_record("c1", "/app/foo.py")

    assert _items_equal_mod_live_windows([live], [edit]) is False


@pytest.mark.asyncio
async def test_compact_partition_swap_proceeds_after_concurrent_live_window_refresh():
    """B4 regression: a concurrent request that ran `reconcile_tracked_files` (or
    a `file_tool` write) between the snapshot and the compaction wake-up will have
    mutated the live window's output, revision, and view_end_line in place. Today
    that falsifies the strict prefix equality check and silently skips the swap.
    With the modulo-live-window comparison the swap proceeds; the lifted windows
    in the new partition may briefly carry pre-refresh state but the next request's
    `reconcile_tracked_files` repairs them."""
    snapshot = _make_snapshot_with_live_window(
        partition_uuid="snap",
        path="/app/foo.py",
        message_count=COMPACTION_RECENCY_TAIL + 2,
    )

    # Build a "current" partition that matches snapshot except the live window has
    # been refreshed to a new revision in place — the exact mutation a concurrent
    # request would produce.
    current = ContextPartition.model_validate_json(snapshot.model_dump_json())
    refreshed_count = 0
    for item in current.items:
        if _is_live_window(item):
            item.prokaryotes_annotations["file_tool.revision"] = "rev2"
            item.prokaryotes_annotations["file_tool.view_end_line"] = "9"
            item.output = "FILE path=/app/foo.py revision=rev2 status=live lines=1-9 line_count=9"
            refreshed_count += 1
    assert refreshed_count == 1, "test setup expected one live window in the snapshot"

    redis = FakeRedis({"context_partition:conv": current.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base()
    wb.search_client = search
    wb.redis_client = redis

    async def compact_fn(_):
        return "S1"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    # Swap must have happened: a fresh partition with a new partition_uuid and the
    # snapshot recorded as parent.
    assert cached.partition_uuid != "snap"
    assert cached.parent_partition_uuid == "snap"
    assert cached.ancestor_summaries == ["S1"]


# --- Live-window body stripping for summarization input (`_strip_live_window_bodies`) ---


def _conflict_live_window(call_id: str, path: str) -> ContextPartitionItem:
    return ContextPartitionItem(
        call_id=call_id,
        output=(
            f"CONFLICT path={path} expected_revision=oldrev current_revision=newrev\n"
            "The file changed since the revision you read. Re-read before retrying.\n"
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


def _stale_window(call_id: str, path: str) -> ContextPartitionItem:
    item = _live_window(call_id, path)
    item.prokaryotes_annotations["file_tool.status"] = "stale"
    item.output = f"FILE path={path} status=stale [no longer accessible: FileNotFoundError]"
    return item


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
    """CONFLICT live windows keep their header lines (which describe the model's
    failed attempt) and have only the embedded `Current view` body replaced."""
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
    assert "The file changed since the revision you read." in output
    assert "[Live tracked file: /app/foo.py" in output
    # The volatile body must be gone.
    assert "1 | actual" not in output
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


def test_strip_live_window_bodies_preserves_edit_records_and_tombstones_and_messages():
    """Only `status=live` items are stripped. Edit records (no status annotation),
    tombstones (`status=stale`), and message items must be preserved verbatim so the
    summarizer still sees the historical record of file activity."""
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
    """The summarization input must be a deep copy — `_compact_partition` continues to
    use the original snapshot for the lift after the summarizer call returns."""
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
    """The invariant is broad: every live window has its current contents stripped,
    not just those whose path is active in the recency tail. Once content lands in a
    summary, no future reconciliation can repair it, so stripping must apply to every
    live window in the summarization input regardless of activity classification."""
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
    """Counterpart to the refresh test above: if the prefix mutation was a
    tombstone (status live → stale), `_items_equal_mod_live_windows` surfaces it
    as a real divergence and the swap is skipped — preserving the existing
    safety behavior for substantive prefix changes."""
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

    redis = FakeRedis({"context_partition:conv": current.model_dump_json()})
    lock_key = "compaction_lock:conv"
    await redis.set(lock_key, "1")
    search = FakeSearchClient([make_doc(snapshot)])
    wb = make_web_base()
    wb.search_client = search
    wb.redis_client = redis

    async def compact_fn(_):
        return "S1"

    await wb._compact_partition(
        snapshot=snapshot,
        conversation_uuid="conv",
        compact_fn=compact_fn,
        lock_key=lock_key,
    )

    # The Redis partition is whatever the concurrent writer last left there,
    # unchanged by this skipped compaction.
    cached = ContextPartition.model_validate_json(await redis.get("context_partition:conv"))
    assert cached.partition_uuid == "snap"
    assert cached.parent_partition_uuid is None
    assert cached.ancestor_summaries == []
