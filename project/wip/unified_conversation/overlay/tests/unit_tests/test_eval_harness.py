"""Tests for `EvalHarness.count_turns` rewritten against the new model.

`count_turns` was previously `partition.items`-shaped; under the unified model
it operates on `list[TurnItem]` (function_call / function_call_output only)
plus a `had_final_assistant` boolean for the terminal LLM call.
"""

from __future__ import annotations

from prokaryotes.conversation_v1.models import TurnItem
from prokaryotes.harness_v1.eval import EvalHarness


def _fc(call_id: str, name: str = "shell_command") -> TurnItem:
    return TurnItem(type="function_call", call_id=call_id, name=name, arguments="{}")


def _fco(call_id: str, output: str = "ok") -> TurnItem:
    return TurnItem(type="function_call_output", call_id=call_id, output=output)


class TestCountTurns:
    def test_zero_tool_calls_one_final_message(self):
        """Pure text completion — one LLM call, no tools."""
        assert EvalHarness.count_turns([], had_final_assistant=True) == 1

    def test_one_round_then_final(self):
        """Bot called a tool once, then produced the final answer — two LLM calls."""
        items = [_fc("c1"), _fco("c1")]
        assert EvalHarness.count_turns(items, had_final_assistant=True) == 2

    def test_two_rounds_then_final(self):
        items = [_fc("c1"), _fco("c1"), _fc("c2"), _fco("c2")]
        assert EvalHarness.count_turns(items, had_final_assistant=True) == 3

    def test_max_rounds_hit_no_final(self):
        """Tool-call loop exhausted max rounds without producing a final message."""
        items = [_fc("c1"), _fco("c1"), _fc("c2"), _fco("c2")]
        assert EvalHarness.count_turns(items, had_final_assistant=False) == 2

    def test_parallel_tool_calls_in_one_round(self):
        """Multiple function_calls before any function_call_output count as one round."""
        items = [_fc("c1"), _fc("c2"), _fco("c1"), _fco("c2")]
        assert EvalHarness.count_turns(items, had_final_assistant=True) == 2

    def test_only_function_call_no_output(self):
        """Edge case: tool call without its output yet — still counts as one round."""
        items = [_fc("c1")]
        assert EvalHarness.count_turns(items, had_final_assistant=False) == 1

    def test_think_call_counts_like_any_tool(self):
        """Think tool calls are regular function_calls — they participate in rounds."""
        items = [_fc("c1", name="think"), _fco("c1"), _fc("c2", name="shell_command"), _fco("c2")]
        assert EvalHarness.count_turns(items, had_final_assistant=True) == 3
