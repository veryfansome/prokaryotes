"""WebHarness._summarize_and_compact — summarization input invariants.

Ancestor summaries and working-file windows are NOT re-fed into the summarization LLM call. The compactor's
`pre_tail_conv` carries `ancestor_summaries=[]` and `working_file_windows=[]` so summaries-of-summaries and live
file bodies cannot fossilize into the new summary. The summarization input is the pre-tail messages projected,
with an explicit summarization prompt as the final user message, and `instruction=None`.
"""

from __future__ import annotations

import pytest

from prokaryotes.context_v1.compaction import _CompactionPrep
from prokaryotes.conversation_v1.models import ProjectedItem, TurnItem
from prokaryotes.harness_v1.base import _SUMMARIZATION_PROMPT
from prokaryotes.harness_v1.web import WebHarness
from tests.unit_tests._builders import bot_msg, msg, turn
from tests.unit_tests._llm_fakes import FakeAnthropicClient, LLMRound, LLMScript


def _make_harness() -> tuple[WebHarness, FakeAnthropicClient]:
    """WebHarness with a recording fake LLM. Bypasses init() — only `_summarize_and_compact` and llm_client
    wiring are needed."""
    fake = FakeAnthropicClient()
    fake.set_script(LLMScript(rounds=[LLMRound()]))
    harness = object.__new__(WebHarness)
    harness.llm_client = fake
    return harness, fake


def _function_call(call_id: str, name: str) -> TurnItem:
    return TurnItem(type="function_call", call_id=call_id, name=name, arguments="{}")


def _make_prep_with_pre_tail_turns(pre_tail_turns: dict) -> _CompactionPrep:
    """Build a _CompactionPrep with the given pre_tail_turns. pre_tail_messages are a bot message per bot_id in
    pre_tail_turns, plus a leading user msg."""
    pre_tail_messages = [msg("1.000000", "U-pre")]
    for bid in pre_tail_turns:
        pre_tail_messages.append(bot_msg(bid, "A-pre"))
    return _CompactionPrep(
        pre_tail_messages=pre_tail_messages,
        recency_tail_messages=[msg("9.000000", "U-tail")],
        tail_offset=len(pre_tail_messages),
        pre_tail_turns=pre_tail_turns,
        recency_tail_turns={},
    )


@pytest.mark.asyncio
async def test_summarize_omits_working_file_windows_from_pre_tail_conv():
    """`pre_tail_conv` is constructed with `working_file_windows=[]`, so the projected items never carry a
    `<working_files>` block — replacing the older strip_live_window_bodies mechanism."""
    from prokaryotes.conversation_v1.models import WorkingFileWindow
    from tests.unit_tests._builders import conversation

    harness, fake = _make_harness()
    prep = _make_prep_with_pre_tail_turns({})
    snapshot = conversation(
        *prep.pre_tail_messages,
        *prep.recency_tail_messages,
        snapshot_uuid="s1",
        working_file_windows=[
            WorkingFileWindow(
                window_id="c1",
                path="/tmp/x",
                status="live",
                revision="r1",
                rendered_output="FILE path=/tmp/x revision=r1 status=live\nLIVE-WINDOW-BODY-MARKER",
                view_start_line=1,
                view_end_line=3,
                requested_end_line=3,
                source_kind="read_lines",
            )
        ],
    )

    await harness._summarize_and_compact(model="claude-opus-4-7", snapshot=snapshot, prep=prep)

    items = fake.complete_calls[0]["items"]
    for it in items:
        haystack = (it.content or "") + (it.output or "")
        assert "LIVE-WINDOW-BODY-MARKER" not in haystack
        assert "<working_files" not in haystack


@pytest.mark.asyncio
async def test_summarize_uses_only_committed_turn_items():
    """The summarization input is built from `pre_tail_turns` items (committed function_call /
    function_call_output entries). Transient narration that never reaches `on_committed_turn_item` is
    structurally absent."""
    from tests.unit_tests._builders import conversation

    harness, fake = _make_harness()
    pre_tail_turns = {
        "2.000000": turn(
            "2.000000",
            _function_call("c1", "file_tool"),
            TurnItem(call_id="c1", type="function_call_output", output="committed-result"),
        ),
    }
    prep = _make_prep_with_pre_tail_turns(pre_tail_turns)
    snapshot = conversation(*prep.pre_tail_messages, *prep.recency_tail_messages, snapshot_uuid="s1")

    await harness._summarize_and_compact(model="claude-opus-4-7", snapshot=snapshot, prep=prep)

    items = fake.complete_calls[0]["items"]
    # The committed function_call_output text reaches the projection.
    has_committed = any("committed-result" in (it.output or "") for it in items)
    assert has_committed
    # Every function_call has a matching output — no orphans, no narration.
    function_call_ids = [it.call_id for it in items if it.type == "function_call"]
    function_call_output_ids = [it.call_id for it in items if it.type == "function_call_output"]
    assert function_call_ids == function_call_output_ids


@pytest.mark.asyncio
async def test_summarize_appends_summarization_prompt_as_user_message():
    """The final item in the summarization input is the summarization prompt as a `user` message."""
    from tests.unit_tests._builders import conversation

    harness, fake = _make_harness()
    prep = _make_prep_with_pre_tail_turns({})
    snapshot = conversation(*prep.pre_tail_messages, *prep.recency_tail_messages, snapshot_uuid="s1")

    await harness._summarize_and_compact(model="claude-opus-4-7", snapshot=snapshot, prep=prep)

    items = fake.complete_calls[0]["items"]
    last = items[-1]
    assert last == ProjectedItem(type="message", role="user", content=_SUMMARIZATION_PROMPT)


@pytest.mark.asyncio
async def test_summarize_passes_no_instruction():
    """The summarization call passes `instruction=None`. Guards against accidentally re-feeding ancestor
    summaries via the system/developer slot."""
    from tests.unit_tests._builders import conversation

    harness, fake = _make_harness()
    prep = _make_prep_with_pre_tail_turns({})
    snapshot = conversation(
        *prep.pre_tail_messages,
        *prep.recency_tail_messages,
        snapshot_uuid="s1",
        ancestor_summaries=["earlier summary that must not enter the input"],
    )

    await harness._summarize_and_compact(model="claude-opus-4-7", snapshot=snapshot, prep=prep)

    assert fake.complete_calls[0]["instruction"] is None


@pytest.mark.asyncio
async def test_summarize_does_not_inject_ancestor_summaries_into_items():
    """The snapshot's `ancestor_summaries` carry forward to the child as storage state; they must NOT be
    projected into the summarization input."""
    from tests.unit_tests._builders import conversation

    harness, fake = _make_harness()
    prep = _make_prep_with_pre_tail_turns({})
    snapshot = conversation(
        *prep.pre_tail_messages,
        *prep.recency_tail_messages,
        snapshot_uuid="s1",
        ancestor_summaries=["ancestor-summary-marker-XYZ"],
    )

    await harness._summarize_and_compact(model="claude-opus-4-7", snapshot=snapshot, prep=prep)

    items = fake.complete_calls[0]["items"]
    for it in items:
        haystack = (it.content or "") + (it.output or "")
        assert "ancestor-summary-marker-XYZ" not in haystack


@pytest.mark.asyncio
async def test_summarize_omits_compacted_summary_block_delimiters():
    """`project_for_llm` now emits a `<compacted_summary>` block whenever `ancestor_summaries` is non-empty.
    The compactor must blank that field on its summarization-input snapshot, so no opening or closing
    delimiter should reach the LLM."""
    from tests.unit_tests._builders import conversation

    harness, fake = _make_harness()
    prep = _make_prep_with_pre_tail_turns({})
    snapshot = conversation(
        *prep.pre_tail_messages,
        *prep.recency_tail_messages,
        snapshot_uuid="s1",
        ancestor_summaries=["earlier summary should not enter the input"],
    )

    await harness._summarize_and_compact(model="claude-opus-4-7", snapshot=snapshot, prep=prep)

    items = fake.complete_calls[0]["items"]
    for it in items:
        haystack = (it.content or "") + (it.output or "")
        assert "<compacted_summary" not in haystack
        assert "</compacted_summary>" not in haystack


@pytest.mark.asyncio
async def test_summarize_opts_out_under_multi_generation_chain():
    """Multi-generation compaction leaves multiple entries on `ancestor_summaries`. The opt-out must blank
    all of them, not just the head."""
    from tests.unit_tests._builders import conversation

    harness, fake = _make_harness()
    prep = _make_prep_with_pre_tail_turns({})
    snapshot = conversation(
        *prep.pre_tail_messages,
        *prep.recency_tail_messages,
        snapshot_uuid="s1",
        ancestor_summaries=["gen-1-marker", "gen-2-marker", "gen-3-marker"],
    )

    await harness._summarize_and_compact(model="claude-opus-4-7", snapshot=snapshot, prep=prep)

    items = fake.complete_calls[0]["items"]
    for it in items:
        haystack = (it.content or "") + (it.output or "")
        assert "gen-1-marker" not in haystack
        assert "gen-2-marker" not in haystack
        assert "gen-3-marker" not in haystack
        assert "<compacted_summary" not in haystack
