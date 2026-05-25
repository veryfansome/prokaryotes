"""`WebHarness._build_instruction_parts` — instruction-string assembly.

After the context-loader overlay, the signature takes `historical_turns` and `workspace_root` and injects a
"Local context files detected" section between `# Tool usage` and `# User context` when any candidate paths
yield context-file matches. Tests cover the original surface (no ancestor summaries leak into instructions;
tail structure stable) plus the discovery-section ordering and presence/absence rules.
"""

from __future__ import annotations

from pathlib import Path

from prokaryotes.api_v1.models import ToolParameters, ToolSpec
from prokaryotes.conversation_v1.models import WorkingFileWindow
from prokaryotes.harness_v1.web import WebHarness
from tests.unit_tests._builders import conversation, msg


class _StubTool:
    """Minimal FunctionToolCallback for instruction-parts assembly. `_build_instruction_parts` only reads
    `system_message_parts` from each callback."""

    name = "stub_tool"
    system_message_parts = ["## stub_tool", "- stub usage"]

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description="stub", parameters=ToolParameters(properties={}))

    async def call(self, arguments: str, call_id: str):
        return None


def _make_harness() -> WebHarness:
    """WebHarness without invoking __init__/init (network, redis, etc. not needed for instruction-parts
    assembly)."""
    return object.__new__(WebHarness)


def _session() -> dict:
    return {"full_name": "Test User", "user_id": "u-1"}


def _live_window(path: Path) -> WorkingFileWindow:
    return WorkingFileWindow(
        window_id="w1",
        path=str(path),
        status="live",
        revision=None,
        rendered_output="",
        view_start_line=1,
        view_end_line=1,
        requested_end_line=1,
        line_count=1,
        origin_call_ids=["w1"],
        source_kind="read_lines",
    )


class TestInstructionDropsSummaryBlock:
    def test_no_summary_when_ancestor_summaries_empty(self, tmp_path: Path):
        harness = _make_harness()
        conv = conversation(msg("1", "hi"))
        parts = harness._build_instruction_parts(
            conversation=conv,
            historical_turns={},
            latitude=None,
            longitude=None,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
            workspace_root=tmp_path,
        )
        joined = "\n".join(parts)
        assert "<compacted_summary" not in joined
        assert "# Compacted conversation summary" not in joined

    def test_no_summary_when_ancestor_summaries_non_empty(self, tmp_path: Path):
        harness = _make_harness()
        conv = conversation(
            msg("1", "hi"),
            ancestor_summaries=["should-stay-out-of-instruction"],
        )
        parts = harness._build_instruction_parts(
            conversation=conv,
            historical_turns={},
            latitude=None,
            longitude=None,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
            workspace_root=tmp_path,
        )
        joined = "\n".join(parts)
        assert "should-stay-out-of-instruction" not in joined
        assert "<compacted_summary" not in joined
        assert "# Compacted conversation summary" not in joined


class TestInstructionTailStructure:
    def test_user_context_block_present_and_nothing_appended_after_user_lines(self, tmp_path: Path):
        harness = _make_harness()
        conv = conversation(
            msg("1", "hi"),
            ancestor_summaries=["should-not-appear"],
        )
        parts = harness._build_instruction_parts(
            conversation=conv,
            historical_turns={},
            latitude=37.7749,
            longitude=-122.4194,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
            workspace_root=tmp_path,
        )

        assert "# User context" in parts

        idx = parts.index("# User context")
        tail = parts[idx:]
        assert tail[0] == "# User context"
        assert tail[1] == ""
        assert tail[2].startswith("- User's name is Test User")
        assert tail[3].startswith("- User's location is at")
        assert len(tail) == 4  # no more lines appended after the location

    def test_no_location_line_when_lat_long_missing(self, tmp_path: Path):
        harness = _make_harness()
        conv = conversation(msg("1", "hi"))
        parts = harness._build_instruction_parts(
            conversation=conv,
            historical_turns={},
            latitude=None,
            longitude=None,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
            workspace_root=tmp_path,
        )
        idx = parts.index("# User context")
        tail = parts[idx:]
        assert tail[-1].startswith("- User's name is Test User")


class TestDiscoverySection:
    def test_no_section_when_no_candidates_match(self, tmp_path: Path):
        harness = _make_harness()
        conv = conversation(msg("1", "hi"))
        parts = harness._build_instruction_parts(
            conversation=conv,
            historical_turns={},
            latitude=None,
            longitude=None,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
            workspace_root=tmp_path,
        )
        joined = "\n".join(parts)
        assert "# Local context files detected" not in joined

    def test_section_emitted_when_live_window_neighbors_a_readme(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "README.md").write_text("local guidance")
        target = tmp_path / "src" / "main.py"
        target.write_text("x")

        harness = _make_harness()
        conv = conversation(msg("1", "hi"), working_file_windows=[_live_window(target)])
        parts = harness._build_instruction_parts(
            conversation=conv,
            historical_turns={},
            latitude=None,
            longitude=None,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
            workspace_root=tmp_path,
        )
        joined = "\n".join(parts)
        assert "# Local context files detected" in joined
        assert "src/README.md" in joined

    def test_section_sits_between_tool_usage_and_user_context(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "README.md").write_text("guide")
        target = tmp_path / "src" / "main.py"
        target.write_text("x")

        harness = _make_harness()
        conv = conversation(msg("1", "hi"), working_file_windows=[_live_window(target)])
        parts = harness._build_instruction_parts(
            conversation=conv,
            historical_turns={},
            latitude=None,
            longitude=None,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
            workspace_root=tmp_path,
        )
        tool_idx = parts.index("# Tool usage")
        discovery_idx = parts.index("# Local context files detected")
        user_idx = parts.index("# User context")
        assert tool_idx < discovery_idx < user_idx
