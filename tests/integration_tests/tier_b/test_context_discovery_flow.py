"""Tier B coverage for the context-discovery prompt section.

End-to-end across the real Redis/Postgres/Elasticsearch stack, parametrized over both LLM providers.

Two flows are validated:

1. **User-mention path.** A slash-bearing path mention in the user message surfaces in the next turn's
   instruction string, with all matched context files listed.
2. **Annotation path.** A prior turn's `file_tool.create_file` call persists a `file_tool.path` annotation
   on its `function_call_output` *without* minting a `WorkingFileWindow`. The *next* turn's instruction
   must contain the discovery section, sourced from that annotation alone — exercises the historical-turn
   fetch in `_dispatch_turn` and the annotation arm of `collect_candidate_paths` in isolation, with no
   live-window fallback that could mask a broken annotation path.

CWD during pytest is the repo root, so the workspace_root the harness picks up via `Path.cwd()` contains
real `README.md` files (`README.md`, `prokaryotes/README.md`, etc.) discovery can find. For the annotation
test we additionally seed a `CLAUDE.md` under a temp dir inside cwd, so the seeded file is uniquely
discoverable from the just-created scratch file's parent dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from prokaryotes.conversation_v1.models import Conversation
from tests.integration_tests.tier_b._helpers import (
    apply_assignments,
    echo_assistant,
    post_chat,
    user_message,
)
from tests.unit_tests._llm_fakes import LLMRound, LLMScript, ToolCallSpec

pytestmark = pytest.mark.integration


def _create_args(path: str, new_text: str) -> str:
    """Match the `create_file` arg shape FileTool expects. `create_file` stamps a `file_tool.path`
    annotation but does NOT mint a `WorkingFileWindow`, so it's the right action to exercise the
    annotation-only arm of `collect_candidate_paths`."""
    return json.dumps(
        {
            "action": "create_file",
            "path": path,
            "expected_revision": None,
            "start_line": None,
            "end_line": None,
            "new_text": new_text,
        }
    )


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_user_path_mention_surfaces_discovery_section_in_next_turn(web_harness, authed_client):
    """User message mentions `prokaryotes/README.md`; the recorded instruction for this turn must contain
    the discovery section and reference the mentioned file."""
    web_harness.llm_client.set_script(
        LLMScript(rounds=[LLMRound(text_deltas=["ok"], stop_reason="end_turn", input_tokens=500)])
    )
    conversation_uuid = str(uuid4())

    await post_chat(
        web_harness,
        authed_client,
        conversation_uuid,
        [user_message("Take a look at prokaryotes/README.md")],
    )

    assert web_harness.llm_client.stream_turn_calls, "fake LLM did not see a stream_turn call"
    instruction = web_harness.llm_client.stream_turn_calls[-1]["instruction"] or ""
    assert "# Local context files detected" in instruction
    assert "prokaryotes/README.md" in instruction


@pytest.mark.parametrize(
    "web_harness, authed_client",
    [("anthropic", "anthropic"), ("openai", "openai")],
    indirect=True,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_historical_file_tool_annotation_surfaces_discovery_section(web_harness, authed_client):
    """Turn 1 issues `file_tool.create_file` against a fresh path under a temp dir seeded with its own
    `CLAUDE.md`. `create_file` stamps the `file_tool.path` annotation but does NOT mint a
    `WorkingFileWindow`, so Turn 2's discovery section can only come from the annotation arm — proves
    `_dispatch_turn` fetches historical turns and that the annotation arm of `collect_candidate_paths`
    is wired through end-to-end, with no live-window fallback that could mask a broken annotation path."""
    conversation_uuid = str(uuid4())

    with TemporaryDirectory(dir=Path.cwd(), prefix=".discovery-it-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        seeded_context_file = temp_dir / "CLAUDE.md"
        seeded_context_file.write_text("temp-dir context for the annotation-arm test\n")

        scratch_path = temp_dir / f"scratch-{uuid4().hex[:8]}.txt"

        # Turn 1: create_file stamps the annotation; the response is a CREATED edit record (no window).
        web_harness.llm_client.set_script(
            LLMScript(
                rounds=[
                    LLMRound(
                        stop_reason="tool_use",
                        text_deltas=["Creating."],
                        tool_calls=[
                            ToolCallSpec(
                                arguments=_create_args(str(scratch_path), "hello\n"),
                                call_id="call-create-1",
                                name="file_tool",
                            )
                        ],
                        input_tokens=500,
                    ),
                    LLMRound(text_deltas=["Created."], stop_reason="end_turn", input_tokens=500),
                ]
            )
        )
        messages: list[dict] = [user_message("Write something to scratch.")]
        record_1 = await post_chat(web_harness, authed_client, conversation_uuid, messages)
        apply_assignments(messages, record_1.source_id_assignments)
        echo_assistant(messages, record_1)

        # Guard the test's premise: discovery for Turn 2 must rely on the annotation alone, so confirm
        # no `WorkingFileWindow` exists for the created path. If FileTool ever starts minting a window on
        # `create_file`, this assertion fires and signals that the test no longer isolates the annotation
        # arm — pick a different action or another history-only file-tool output shape.
        cached_after_turn1 = Conversation.model_validate_json(
            await web_harness.redis_client.get(f"conversation:{conversation_uuid}")
        )
        assert not any(w.path == str(scratch_path) for w in cached_after_turn1.working_file_windows), (
            "create_file unexpectedly minted a WorkingFileWindow; this test no longer isolates the annotation arm"
        )

        # Turn 2: plain reply. The instruction's discovery section can only come from the annotation arm.
        web_harness.llm_client.set_script(
            LLMScript(rounds=[LLMRound(text_deltas=["Thanks."], stop_reason="end_turn", input_tokens=500)])
        )
        messages.append(user_message("Anything notable?"))
        await post_chat(
            web_harness,
            authed_client,
            conversation_uuid,
            messages,
            snapshot_uuid=record_1.snapshot_uuid,
        )

        instruction = web_harness.llm_client.stream_turn_calls[-1]["instruction"] or ""
        assert "# Local context files detected" in instruction, (
            "discovery section missing from turn-2 instruction; annotation arm not wired through"
        )
        assert str(seeded_context_file) in instruction, (
            "seeded CLAUDE.md under the temp dir was not surfaced via the annotation arm"
        )
