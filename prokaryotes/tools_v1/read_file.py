import aiofiles.os
import asyncio
import json
import logging
from openai.types.responses import FunctionToolParam
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import LLMClient
from prokaryotes.models_v1 import ChatMessage
from prokaryotes.search_v1 import SearchClient
from prokaryotes.tools_v1.base_path_tool import PathToolCallback
from prokaryotes.utils_v1.logging_utils import log_async_task_exception

logger = logging.getLogger(__name__)

class ReadFileCallback(PathToolCallback):
    def __init__(self, llm_client: LLMClient, search_client: SearchClient):
        super().__init__(llm_client, search_client)

    @property
    def tool_param(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name="read_file",
            description="Read a local file's contents.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to read.",
                    },
                },
                "additionalProperties": False,
                "required": ["path"],
            },
            strict=True,
        )

    async def call(self, context_snapshot: list[ChatMessage], arguments: str, call_id: str) -> FunctionCallOutput:
        contents = ""
        error = ""
        file_size_limit = 20_000  # TODO: For larger files, better to use the file API
        try:
            arguments: dict[str, str] = json.loads(arguments)
            if "path" in arguments and arguments["path"].strip():
                async with aiofiles.open(arguments["path"], mode="rb") as f:
                    stat_info = await aiofiles.os.stat(f.fileno())
                    if stat_info.st_size >= file_size_limit:
                        raise Exception(
                            "File too large."
                            f" {stat_info.st_size} bytes exceeds size limit of {file_size_limit}"
                        )
                    first_kb = await f.read(1024)
                    if b'\x00' in first_kb:
                        raise Exception(f"File looks binary")
                    await f.seek(0)  # Go back to start
                    contents = (await f.read()).decode("utf-8", errors="replace")
            else:
                raise Exception(f"Missing or empty path in {arguments}")
        except Exception as e:
            logger.exception(f"Failed to read file {arguments}")
            error = str(e)
        output = contents
        if error:
            output = f"{output}\n\n{error}"
        indexing_task = asyncio.create_task(self.index(context_snapshot, output, arguments["path"]))
        indexing_task.add_done_callback(log_async_task_exception)
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=output,
        )
