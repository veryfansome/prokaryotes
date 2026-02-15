import logging
import os
import uuid
from collections.abc import Callable

from prokaryotes.anthropic_v1 import LLMClient
from prokaryotes.api_v1.models import (
    ContextPartition,
    ContextPartitionItem,
    FunctionToolCallback,
)
from prokaryotes.tools_v1.shell_command import ShellCommandTool
from prokaryotes.tools_v1.think import ThinkTool
from prokaryotes.utils_v1 import system_message_utils

logger = logging.getLogger(__name__)


class ScriptHarness:
    def __init__(self, model: str, reasoning_effort: str = None):
        self.llm_client = LLMClient()
        self.llm_client.init_client()
        self.model = model
        self.reasoning_effort = reasoning_effort

    async def close(self):
        await self.llm_client.close()

    async def run(
        self,
        task: str,
        cwd: str = None,
        max_tool_call_rounds: int = None,
        on_usage: Callable[[int, int], None] | None = None,
        verbose: bool = True,
    ) -> ContextPartition:
        if cwd:
            os.chdir(cwd)

        shell_command_tool = ShellCommandTool()
        think_tool = ThinkTool()
        tool_callbacks: dict[str, FunctionToolCallback] = {
            shell_command_tool.name: shell_command_tool,
            think_tool.name: think_tool,
        }

        system_parts = [
            *system_message_utils.get_core_instruction_parts(interactive=False, summaries=False),
            "",
            *system_message_utils.get_non_interactive_execution_mode_parts(),
            "",
            "# Tool usage",
        ]
        for name in sorted(tool_callbacks):
            system_parts.extend(tool_callbacks[name].system_message_parts)
        system_parts.append("")
        system_parts.extend(system_message_utils.get_script_harness_runtime_context_parts())

        context_partition = ContextPartition(
            conversation_uuid=str(uuid.uuid4()),
            items=[
                ContextPartitionItem(role="system", content="\n".join(system_parts)),
                ContextPartitionItem(role="user", content=task),
            ],
        )

        async for chunk in self.llm_client.stream_response(
            context_partition=context_partition,
            model=self.model,
            max_tool_call_rounds=max_tool_call_rounds,
            on_usage=on_usage,
            reasoning_effort=self.reasoning_effort,
            tool_callbacks=tool_callbacks,
        ):
            if verbose:
                print(chunk, end="", flush=True)

        if verbose:
            print()

        return context_partition
