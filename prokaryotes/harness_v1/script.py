"""ScriptHarness — non-interactive one-shot task runner.

Synthesizes a single-turn `Conversation` (one user message), projects it, drives `stream_turn` with tool
callbacks, and returns a `ScriptRunResult` carrying the final `Conversation`, the produced `TurnExecution` (if any
tool calls happened), and the final assistant text.

Used by `EvalHarness` for fixtures and by `scripts/cli.py` for ad-hoc runs. Script runs are entirely ephemeral —
no Redis/ES persistence.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from prokaryotes.api_v1.models import FunctionToolCallback
from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    ProjectedItem,
    TurnExecution,
    TurnItem,
)
from prokaryotes.harness_v1 import build_llm_client
from prokaryotes.tools_v1.shell_command import ShellCommandTool
from prokaryotes.tools_v1.think import ThinkTool
from prokaryotes.utils_v1 import system_message_utils

logger = logging.getLogger(__name__)

# Fixed source_ids for the synthesized one-shot conversation. Not timestamps: the script flow never reconciles
# with a stored snapshot, and projection sorts by source_id, so "0" < "1" is the only invariant that matters.
_USER_MESSAGE_SOURCE_ID = "0.000000"
_BOT_MESSAGE_SOURCE_ID = "1.000000"
_SCRIPT_BOT_AUTHOR_ID = "__bot__"
_SCRIPT_USER_AUTHOR_ID = "__script_user__"


@dataclass
class ScriptRunResult:
    """Output of `ScriptHarness.run`.

    Callers (e.g. `EvalHarness`) can count tool calls off `turn_execution.items`, count think calls by filtering on
    `name == "think"`, and persist both the conversation and turn execution as separate JSON artifacts.

    `final_assistant_text` is empty if the turn aborted before producing a final assistant message
    (max-tool-call-rounds hit, stream error, etc.) — in that case `turn_execution.completed` is `False` too.
    """

    conversation: Conversation
    final_assistant_text: str = ""
    turn_execution: TurnExecution | None = None


class ScriptHarness:
    """Provider-agnostic non-interactive harness.

    No persistence layer — the conversation lives in memory and is returned via `ScriptRunResult`.
    """

    def __init__(
        self,
        impl: str,
        model: str,
        reasoning_effort: str | None = None,
        think_reasoning_effort: str | None = None,
    ):
        self.impl = impl
        self.llm_client, _instruction_role = build_llm_client(impl)
        self.llm_client.init_client()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.think_reasoning_effort = think_reasoning_effort

    async def close(self) -> None:
        await self.llm_client.close()

    async def run(
        self,
        task: str,
        cwd: str | None = None,
        max_tool_call_rounds: int | None = None,
        on_usage: Callable[[int, int], None] | None = None,
        verbose: bool = True,
    ) -> ScriptRunResult:
        if cwd:
            os.chdir(cwd)

        shell_command_tool = ShellCommandTool()
        think_tool = ThinkTool(
            self.llm_client,
            self.model,
            reasoning_effort=self.think_reasoning_effort,
        )
        tool_callbacks: dict[str, FunctionToolCallback] = {
            shell_command_tool.name: shell_command_tool,
            think_tool.name: think_tool,
        }

        instruction = self._build_instruction(tool_callbacks)

        conversation = Conversation(
            conversation_uuid=str(uuid.uuid4()),
            bot_author_id=_SCRIPT_BOT_AUTHOR_ID,
            messages=[
                ConversationMessage(
                    source_id=_USER_MESSAGE_SOURCE_ID,
                    author_id=_SCRIPT_USER_AUTHOR_ID,
                    content=task,
                ),
            ],
        )
        projected_items = [ProjectedItem(type="message", role="user", content=task)]

        committed_turn_items: list[TurnItem] = []
        final_assistant_text_parts: list[str] = []

        def _on_committed_turn_item(item: TurnItem) -> None:
            committed_turn_items.append(item)

        def _on_final_assistant_message(text: str) -> None:
            final_assistant_text_parts.append(text)

        async for chunk in self.llm_client.stream_turn(
            items=projected_items,
            instruction=instruction,
            model=self.model,
            max_tool_call_rounds=max_tool_call_rounds,
            on_committed_turn_item=_on_committed_turn_item,
            on_final_assistant_message=_on_final_assistant_message,
            on_usage=on_usage,
            reasoning_effort=self.reasoning_effort,
            stream_ndjson=False,
            tool_callbacks=tool_callbacks,
        ):
            if verbose:
                print(chunk, end="", flush=True)

        if verbose:
            print()

        final_text = "".join(final_assistant_text_parts)
        if final_text:
            conversation.messages.append(
                ConversationMessage(
                    source_id=_BOT_MESSAGE_SOURCE_ID,
                    author_id=_SCRIPT_BOT_AUTHOR_ID,
                    content=final_text,
                )
            )

        turn_execution: TurnExecution | None = None
        if committed_turn_items:
            turn_execution = TurnExecution(
                conversation_uuid=conversation.conversation_uuid,
                bot_message_source_id=_BOT_MESSAGE_SOURCE_ID,
                items=committed_turn_items,
                completed=bool(final_text),
            )

        return ScriptRunResult(
            conversation=conversation,
            final_assistant_text=final_text,
            turn_execution=turn_execution,
        )

    @staticmethod
    def _build_instruction(tool_callbacks: dict[str, FunctionToolCallback]) -> str:
        system_parts = [
            *system_message_utils.get_core_instruction_parts(interactive=False, summaries=False),
            "",
            *system_message_utils.get_non_interactive_execution_mode_parts(),
            "",
            "# Tool usage",
        ]
        for name in sorted(tool_callbacks):
            system_parts.append("")
            system_parts.extend(tool_callbacks[name].system_message_parts)
        system_parts.append("")
        system_parts.extend(system_message_utils.get_runtime_context_parts())
        return "\n".join(system_parts)
