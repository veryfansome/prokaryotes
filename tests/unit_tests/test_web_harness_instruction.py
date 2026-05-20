"""`WebHarness._build_instruction_parts` — the instruction string no longer carries the ancestor-summary block.

The summary now lives in projection as a leading user-role block; the instruction parts retain only trusted
content.
"""

from __future__ import annotations

from prokaryotes.api_v1.models import ToolParameters, ToolSpec
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


class TestInstructionDropsSummaryBlock:
    def test_no_summary_when_ancestor_summaries_empty(self):
        harness = _make_harness()
        conv = conversation(msg("1", "hi"))
        parts = harness._build_instruction_parts(
            conversation=conv,
            latitude=None,
            longitude=None,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
        )
        joined = "\n".join(parts)
        assert "<compacted_summary" not in joined
        assert "# Compacted conversation summary" not in joined

    def test_no_summary_when_ancestor_summaries_non_empty(self):
        harness = _make_harness()
        conv = conversation(
            msg("1", "hi"),
            ancestor_summaries=["should-stay-out-of-instruction"],
        )
        parts = harness._build_instruction_parts(
            conversation=conv,
            latitude=None,
            longitude=None,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
        )
        joined = "\n".join(parts)
        assert "should-stay-out-of-instruction" not in joined
        assert "<compacted_summary" not in joined
        assert "# Compacted conversation summary" not in joined


class TestInstructionTailStructure:
    def test_user_context_block_present_and_nothing_appended_after_user_lines(self):
        harness = _make_harness()
        conv = conversation(
            msg("1", "hi"),
            ancestor_summaries=["should-not-appear"],
        )
        parts = harness._build_instruction_parts(
            conversation=conv,
            latitude=37.7749,
            longitude=-122.4194,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
        )

        assert "# User context" in parts

        idx = parts.index("# User context")
        tail = parts[idx:]
        assert tail[0] == "# User context"
        assert tail[1] == ""
        assert tail[2].startswith("- User's name is Test User")
        assert tail[3].startswith("- User's location is at")
        assert len(tail) == 4  # no more lines appended after the location

    def test_no_location_line_when_lat_long_missing(self):
        harness = _make_harness()
        conv = conversation(msg("1", "hi"))
        parts = harness._build_instruction_parts(
            conversation=conv,
            latitude=None,
            longitude=None,
            session=_session(),
            time_zone=None,
            tool_callbacks={_StubTool().name: _StubTool()},
        )
        idx = parts.index("# User context")
        tail = parts[idx:]
        assert tail[-1].startswith("- User's name is Test User")
