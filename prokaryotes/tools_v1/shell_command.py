import asyncio
import json
import logging
import traceback

from openai.types.responses import FunctionToolParam
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import (
    FunctionCallOutputIndexer,
    FunctionToolCallback,
)
from prokaryotes.models_v1 import ToolCallDoc
from prokaryotes.search_v1 import SearchClient

logger = logging.getLogger(__name__)


class ShellCommandCallback(FunctionCallOutputIndexer, FunctionToolCallback):
    def __init__(self, search_client: SearchClient):
        self.search_client = search_client

    @property
    def tool_param(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name="run_shell_command",
            description="Run an arbitrary shell command.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "A command string to pass to asyncio.create_subprocess_shell()",
                    },
                },
                "additionalProperties": False,
                "required": ["command"],
            },
            strict=True,
        )

    async def index(
            self,
            call_id: str,
            arguments: str,
            labels: list[str],
            output: str,
            prompt_summary: str,
            prompt_summary_emb: list[float],
    ) -> ToolCallDoc | None:
        return await self.search_client.index_tool_call(
            call_id=call_id,
            dedupe_strategy="similar",
            labels=labels,
            output=output,
            prompt_summary=prompt_summary,
            prompt_summary_emb=prompt_summary_emb,
            tool_arguments=arguments,
            tool_name=self.tool_param["name"]
        )

    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
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
            output = "\n".join([
                "# STDOUT",
                stdout.decode(),
                "# STDERR",
                stderr.decode(),
            ])
        except Exception:
            error = traceback.format_exc()
        if error:
            if output:
                output += "\n\n"
            output += f"An error occurred:\n{error}"
        logger.info(f"\n{output}")
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=output
        )
