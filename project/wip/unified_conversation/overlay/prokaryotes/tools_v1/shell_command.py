"""ShellCommandTool — retyped to return `TurnItem`."""

from __future__ import annotations

import asyncio
import logging

from prokaryotes.api_v1.models import FunctionToolCallback, ToolParameters, ToolSpec
from prokaryotes.conversation_v1.models import TurnItem

logger = logging.getLogger(__name__)


class ShellCommandTool(FunctionToolCallback):
    max_output_lines = 400

    async def call(self, arguments: str, call_id: str) -> TurnItem | None:
        import json

        try:
            payload = json.loads(arguments)
        except Exception as exc:
            return TurnItem(call_id=call_id, output=f"ERROR Invalid arguments: {exc}", type="function_call_output")
        command = payload.get("command")
        if not command:
            return TurnItem(call_id=call_id, output="ERROR command is required", type="function_call_output")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
        except Exception as exc:
            logger.exception("ShellCommandTool[%s] failed", call_id)
            return TurnItem(call_id=call_id, output=f"ERROR {type(exc).__name__}: {exc}", type="function_call_output")
        text = stdout.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) > self.max_output_lines:
            tail_marker = f"\n[... output truncated; showed last {self.max_output_lines} of {len(lines)} lines ...]\n"
            text = tail_marker + "\n".join(lines[-self.max_output_lines :])
        return TurnItem(
            call_id=call_id,
            output=f"exit_code={proc.returncode}\n{text}",
            type="function_call_output",
        )

    @property
    def name(self) -> str:
        return "shell_command"

    @property
    def system_message_parts(self) -> list[str]:
        return [
            f"## Using the `{self.name}` tool",
            "",
            f"- Use `{self.name}` to run arbitrary shell commands when no specialized tool fits.",
            f"- Output is captured (stdout+stderr merged) and truncated to the last {self.max_output_lines} lines.",
        ]

    @property
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Run an arbitrary shell command and return its combined stdout+stderr output.",
            parameters=ToolParameters(
                properties={"command": {"type": "string", "description": "The shell command to execute."}},
                required=["command"],
            ),
        )
