import aiofiles.os
import json
import logging
from openai.types.responses import FunctionToolParam
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback, LLMClient
from prokaryotes.search_v1 import SearchClient
from prokaryotes.tools_v1.base_path_tool import PathBasedFunctionCallOutputIndexer

logger = logging.getLogger(__name__)


class ReadFileCallback(PathBasedFunctionCallOutputIndexer, FunctionToolCallback):
    def __init__(self, llm_client: LLMClient, search_client: SearchClient, model: str = "gpt-5.1"):
        self._llm_client = llm_client
        self._model = model
        self._search_client = search_client
        self._tool_param = FunctionToolParam(
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

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    @property
    def model(self) -> str:
        return self._model

    @property
    def search_client(self) -> SearchClient:
        return self._search_client

    @property
    def tool_param(self) -> FunctionToolParam:
        return self._tool_param

    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        contents = ""
        error = ""
        file_size_limit = 20_000  # TODO: For larger files, better to use the file API
        try:
            arguments: dict[str, str] = json.loads(arguments)
            path = self.path_from_arguments(arguments)
            if path:
                async with aiofiles.open(path, mode="rb") as f:
                    stat_info = await aiofiles.os.stat(f.fileno())
                    if stat_info.st_size >= file_size_limit:
                        raise Exception(
                            "File too large."
                            f" {stat_info.st_size} bytes exceeds size limit of {file_size_limit}"
                        )
                    first_kb = await f.read(1024)
                    if b'\x00' in first_kb:
                        raise Exception("File looks binary")
                    await f.seek(0)  # Go back to start
                    contents = (await f.read()).decode("utf-8", errors="replace")
            else:
                raise Exception(f"Missing, empty, or invalid path in {arguments}")
        except Exception as e:
            logger.exception(f"Failed to read file {arguments}")
            error = str(e)
        output = contents
        if error:
            output = f"{output}\n\n{error}"
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=output,
        )
