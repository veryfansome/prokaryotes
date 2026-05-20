"""Two-set origin filter at branch divergence and the helper that builds the call_id sets."""

from __future__ import annotations

from prokaryotes.context_v1.conversation_sync import (
    _active_paths_in_turns,
    _file_tool_call_ids_in,
    _filter_windows_by_active_path_and_origin,
    _filter_windows_by_origin,
)
from prokaryotes.conversation_v1.models import (
    TurnExecution,
    TurnItem,
    WorkingFileWindow,
)


def _file_tool_call(call_id: str) -> TurnItem:
    return TurnItem(type="function_call", call_id=call_id, name="file_tool", arguments="{}")


def _non_file_call(call_id: str, name: str = "think") -> TurnItem:
    return TurnItem(type="function_call", call_id=call_id, name=name, arguments="{}")


def _output(call_id: str, path: str | None = None) -> TurnItem:
    annotations: dict[str, str] | None = None
    if path is not None:
        annotations = {"file_tool.persistence": "working_file", "file_tool.path": path}
    return TurnItem(
        type="function_call_output",
        call_id=call_id,
        output="...",
        prokaryotes_annotations=annotations,
    )


def _turn(bot_id: str, *items: TurnItem) -> TurnExecution:
    return TurnExecution(conversation_uuid="c-1", bot_message_source_id=bot_id, items=list(items))


def _window(window_id: str, path: str = "/abs/a.py") -> WorkingFileWindow:
    return WorkingFileWindow(
        window_id=window_id,
        path=path,
        status="live",
        revision="r1",
        rendered_output=f"FILE path={path} revision=r1",
        view_start_line=1,
        view_end_line=40,
        requested_end_line=40,
        source_kind="read_lines",
    )


class TestFileToolCallIdsIn:
    def test_collects_only_file_tool_function_calls(self):
        turns = {
            "b1": _turn("b1", _file_tool_call("c-a"), _output("c-a"), _non_file_call("c-b"), _output("c-b")),
        }
        assert _file_tool_call_ids_in(turns) == {"c-a"}

    def test_skips_outputs_and_non_file_calls(self):
        turns = {"b1": _turn("b1", _non_file_call("c-x"), _output("c-x"))}
        assert _file_tool_call_ids_in(turns) == set()

    def test_empty_turns_returns_empty_set(self):
        assert _file_tool_call_ids_in({}) == set()


class TestFilterWindowsByOrigin:
    def test_kept_call_id_window_survives(self):
        result = _filter_windows_by_origin(
            [_window("c-kept")],
            kept_call_ids={"c-kept"},
            source_call_ids={"c-kept", "c-discarded"},
        )
        assert [w.window_id for w in result] == ["c-kept"]

    def test_carryforward_window_survives(self):
        """call_id appears in neither set → carryforward bucket → keep."""
        result = _filter_windows_by_origin(
            [_window("c-carry")],
            kept_call_ids={"c-kept"},
            source_call_ids={"c-other"},
        )
        assert [w.window_id for w in result] == ["c-carry"]

    def test_discarded_sibling_window_drops(self):
        """call_id in source but not in kept → discarded-sibling origin → drop."""
        result = _filter_windows_by_origin(
            [_window("c-sibling")],
            kept_call_ids={"c-kept"},
            source_call_ids={"c-kept", "c-sibling"},
        )
        assert result == []

    def test_mixed_input(self):
        windows = [
            _window("c-kept", path="/abs/a.py"),
            _window("c-sibling", path="/abs/a.py"),
            _window("c-carry", path="/abs/b.py"),
        ]
        result = _filter_windows_by_origin(
            windows,
            kept_call_ids={"c-kept"},
            source_call_ids={"c-kept", "c-sibling"},
        )
        assert [w.window_id for w in result] == ["c-kept", "c-carry"]


class TestActivePathsInTurns:
    def test_collects_paths_from_file_tool_outputs(self):
        turns = {
            "b1": _turn(
                "b1",
                _file_tool_call("c-a"),
                _output("c-a", path="/abs/a.py"),
                _file_tool_call("c-b"),
                _output("c-b", path="/abs/b.py"),
            ),
        }
        assert _active_paths_in_turns(turns) == {"/abs/a.py", "/abs/b.py"}

    def test_collects_path_from_edit_output_without_minting_new_window(self):
        """An EDITED record annotates `file_tool.path` even though it doesn't mint a new window. Active-path
        derivation must see this path so carryforward windows for it survive Case A divergence."""
        turns = {"b1": _turn("b1", _file_tool_call("c-edit"), _output("c-edit", path="/abs/edited.py"))}
        assert _active_paths_in_turns(turns) == {"/abs/edited.py"}

    def test_skips_outputs_without_path_annotation(self):
        """Non-file-tool outputs or error outputs have no path annotation; they don't contribute."""
        turns = {"b1": _turn("b1", _file_tool_call("c-x"), _output("c-x", path=None))}
        assert _active_paths_in_turns(turns) == set()

    def test_empty_turns_returns_empty(self):
        assert _active_paths_in_turns({}) == set()


class TestFilterWindowsByActivePathAndOrigin:
    def test_carryforward_path_not_active_is_dropped(self):
        """A carryforward window (call_id in neither set) for a path no longer touched in shared prefix drops."""
        result = _filter_windows_by_active_path_and_origin(
            [_window("c-carry", path="/abs/stale.py")],
            active_paths={"/abs/active.py"},
            kept_call_ids={"c-kept"},
            source_call_ids={"c-kept"},
        )
        assert result == []

    def test_carryforward_path_active_is_kept(self):
        """A carryforward window for a path the shared prefix still touches survives — origin says carryforward,
        active-path gate passes."""
        result = _filter_windows_by_active_path_and_origin(
            [_window("c-carry", path="/abs/active.py")],
            active_paths={"/abs/active.py"},
            kept_call_ids={"c-kept"},
            source_call_ids={"c-kept"},
        )
        assert [w.window_id for w in result] == ["c-carry"]

    def test_kept_origin_active_path_keeps(self):
        result = _filter_windows_by_active_path_and_origin(
            [_window("c-kept", path="/abs/a.py")],
            active_paths={"/abs/a.py"},
            kept_call_ids={"c-kept"},
            source_call_ids={"c-kept"},
        )
        assert [w.window_id for w in result] == ["c-kept"]

    def test_discarded_sibling_origin_drops_even_on_active_path(self):
        """A sibling-origin window whose path is also active in the shared prefix still drops (origin gate)."""
        result = _filter_windows_by_active_path_and_origin(
            [_window("c-sibling", path="/abs/a.py")],
            active_paths={"/abs/a.py"},
            kept_call_ids={"c-kept"},
            source_call_ids={"c-kept", "c-sibling"},
        )
        assert result == []
