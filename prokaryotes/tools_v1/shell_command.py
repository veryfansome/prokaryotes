import asyncio
import json
import logging
import traceback

from prokaryotes.api_v1.models import (
    ContextPartitionItem,
    FunctionToolCallback,
    ToolParameters,
    ToolSpec,
)

logger = logging.getLogger(__name__)


class ShellCommandTool(FunctionToolCallback):
    """Tool to let the model run shell commands"""

    max_output_lines = 400

    async def call(self, arguments: str, call_id: str) -> ContextPartitionItem | None:
        error = ""
        output = ""
        try:
            command = json.loads(arguments)["command"]
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            stdout_lines = stdout.decode(errors="replace").split("\n")
            stdout_lines_len = len(stdout_lines)
            if stdout_lines_len > self.max_output_lines:
                stdout_lines = (
                    stdout_lines[:self.max_output_lines]
                    + [f"--- Truncated after {self.max_output_lines} lines ---"]
                )
            output = "\n".join([
                f"Exit code: {process.returncode}",
                "# STDOUT",
                *stdout_lines,
                "# STDERR",
                stderr.decode(errors="replace"),
            ])
        except Exception:
            error = traceback.format_exc()
        if error:
            if output:
                output += "\n\n"
            output += f"An error occurred:\n{error}"
        logger.info(f"{self.__class__.__name__}[{call_id}]:\n{output}")
        return ContextPartitionItem(
            call_id=call_id,
            output=output,
            type="function_call_output",
        )

    @property
    def name(self) -> str:
        return "shell_command"

    @property
    def system_message_parts(self) -> list[str]:
        lines = [
            f"## Using the `{self.name}` tool",
            (
                "- Don't chain multiple commands with '&&' or ';' unless the intended task can't be accomplished"
                " without doing so. Whenever possible, use a single, focused `command`, with a distinct `reason`"
                " per tool call."
            ),
            "- When reading files, default to previewing the first 200 lines, e.g. `sed -n '1,200p' <path>`.",
            f"- Command output is truncated after {self.max_output_lines} lines so plan around that.",
        ]
        return lines

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Use the tool to run arbitrary shell commands.",
            parameters=ToolParameters(
                properties={
                    "command": {
                        "type": "string",
                        "description": "A command string to pass to asyncio.create_subprocess_shell()."
                    },
                    "reason": {
                        "type": "string",
                        "description": "A concise reason for the command.",
                    },
                },
                required=["command", "reason"],
            ),
        )
