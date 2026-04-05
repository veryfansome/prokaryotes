import asyncio
import json
import logging
import os
import shlex
import traceback

import aiofiles.os
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
        self.max_output_lines = 200
        self.search_client = search_client

    @property
    def tool_param(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name="run_shell_command",
            description=f"Run an arbitrary shell command. Truncates output after {self.max_output_lines} lines.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "A command string to pass to asyncio.create_subprocess_shell()",
                    },
                    # TODO: reason / intent / why
                },
                "additionalProperties": False,
                "required": ["command"],
            },
            strict=True,
        )

    @classmethod
    async def extract_keywords(cls, command: str) -> list[str]:
        keywords = set()
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        for token in lexer:
            if token.startswith("-") and "=" in token:
                _, _, candidate = token.partition("=")
            else:
                candidate = token

            # Look for paths, doesn't have to be fully qualified since working directory is fixed
            if (candidate not in {".", "/"}  # These are too ambiguous
                    and "://" not in candidate  # Exclude URLs for now
                    and ("/" in candidate or "." in candidate)):
                candidate_norm = os.path.expanduser(os.path.expandvars(candidate))
                if await aiofiles.os.path.exists(candidate_norm):
                    keywords.add(candidate)
        return list(keywords)

    async def index(
            self,
            call_id: str,
            arguments: str,
            labels: list[str],
            output: str,
            prompt_summary: str,
            prompt_summary_emb: list[float],
            topics: list[str],
    ) -> ToolCallDoc | None:
        try:
            command = json.loads(arguments)["command"]
            return await self.search_client.index_tool_call(
                call_id=call_id,
                dedupe_strategy="similar",
                labels=labels,
                output=output,
                prompt_summary=prompt_summary,
                prompt_summary_emb=prompt_summary_emb,
                search_keywords=(await self.extract_keywords(command)),
                tool_arguments=arguments,
                tool_name=self.tool_param["name"],
                topics=topics,
            )
        except Exception:
            logger.exception(f"Failed to index ToolCallDoc(doc_id={call_id})")

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
            stdout_lines = stdout.decode(errors="replace").split("\n")
            stdout_lines_len = len(stdout_lines)
            if stdout_lines_len > 200:
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
        logger.info(f"{call_id}:\n{output}")
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=output
        )
