"""Core model contracts for `WorkingFileWindow`, `Conversation.working_files_block()`, and `coverage_eligible`.

Pins the `WorkingFileWindow` contract: `line_count` and `origin_call_ids` are required, `requested_end_line` is a
required `int` (no `None`), and `range_truncated` is no longer a valid `source_kind` — the value can no longer be
constructed, so an enum-rejection test stands in for the old `test_live_range_truncated_eligible`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prokaryotes.conversation_v1.models import (
    Conversation,
    WorkingFileWindow,
    coverage_eligible,
)


def _window(
    *,
    window_id: str = "call-1",
    path: str = "/abs/a.py",
    status: str = "live",
    source_kind: str = "read_lines",
    view_start_line: int = 1,
    view_end_line: int = 40,
    requested_end_line: int = 40,
    line_count: int = 40,
    origin_call_ids: list[str] | None = None,
    rendered_output: str = "FILE path=/abs/a.py revision=r1 status=live\n",
    revision: str | None = "r1",
) -> WorkingFileWindow:
    return WorkingFileWindow(
        window_id=window_id,
        path=path,
        status=status,
        source_kind=source_kind,
        view_start_line=view_start_line,
        view_end_line=view_end_line,
        requested_end_line=requested_end_line,
        line_count=line_count,
        origin_call_ids=origin_call_ids or [window_id],
        rendered_output=rendered_output,
        revision=revision,
    )


def _conv(*windows: WorkingFileWindow) -> Conversation:
    return Conversation(
        conversation_uuid="c-1",
        bot_author_id="b",
        working_file_windows=list(windows),
    )


class TestCoverageEligible:
    def test_live_read_lines_eligible(self):
        assert coverage_eligible(_window(source_kind="read_lines"))

    def test_diagnostic_source_kinds_not_eligible(self):
        for source_kind in ("already_exists", "conflict", "range_error"):
            assert not coverage_eligible(_window(source_kind=source_kind)), source_kind

    def test_tombstone_not_eligible(self):
        assert not coverage_eligible(_window(status="stale", source_kind="tombstone"))

    def test_stale_status_not_eligible_even_with_read_lines_kind(self):
        assert not coverage_eligible(_window(status="stale", source_kind="read_lines"))


class TestSourceKindEnum:
    def test_range_truncated_is_no_longer_a_valid_source_kind(self):
        # Dropped from `WorkingFileSourceKind`: RANGE_TRUNCATED is a response shape only; its window is read_lines.
        with pytest.raises(ValidationError):
            _window(source_kind="range_truncated")


class TestWorkingFilesBlock:
    def test_none_when_no_windows(self):
        assert _conv().working_files_block() is None

    def test_emits_xml_delimited_block_with_trust_attr(self):
        block = _conv(_window()).working_files_block()
        assert block is not None
        assert block.startswith('<working_files trust="file-content">\n')
        assert block.rstrip().endswith("</working_files>")

    def test_renders_one_section_per_window(self):
        block = _conv(
            _window(window_id="c-1", path="/abs/a.py", view_start_line=1, view_end_line=40),
            _window(window_id="c-2", path="/abs/b.py", view_start_line=100, view_end_line=140),
        ).working_files_block()
        assert block is not None
        assert "## Window: /abs/a.py lines 1-40" in block
        assert "## Window: /abs/b.py lines 100-140" in block

    def test_closing_tag_in_rendered_output_is_escaped(self):
        block = _conv(_window(rendered_output="pre </working_files> post")).working_files_block()
        assert block is not None
        assert "<\\/working_files>" in block
        # Only the structural closing tag survives unescaped
        assert block.count("</working_files>") == 1
        assert block.rstrip().endswith("</working_files>")
