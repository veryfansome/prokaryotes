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
from prokaryotes.utils_v1.os_utils import (
    format_st_mtime,
    format_st_size,
    gid_to_name,
    st_mode_to_symbolic_mode,
    uid_to_name,
)

logger = logging.getLogger(__name__)

class ScanDirectoryCallback(PathToolCallback):
    def __init__(self, llm_client: LLMClient, search_client: SearchClient):
        super().__init__(llm_client, search_client)

    @property
    def tool_param(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name="scan_directory",
            description="Scan a local directory's contents.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to scan.",
                    },
                    "inclusion_filters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "A flat list of filter strings."
                            " Entries that don't include any specified substrings will be excluded."
                            " Leave empty unless directed otherwise."
                        ),
                    },
                },
                "additionalProperties": False,
                "required": ["inclusion_filters", "path"],
            },
            strict=True,
        )

    async def call(self, context_snapshot: list[ChatMessage], arguments: str, call_id: str) -> FunctionCallOutput:
        contents = []
        entry_cnt_limit = 100
        error = ""
        rows = []
        try:
            arguments: dict[str, str] = json.loads(arguments)
            inclusion_filters = arguments.get("inclusion_filters", [])
            entry_cnt = 0
            if "path" in arguments and arguments["path"].strip():
                with await aiofiles.os.scandir(arguments["path"]) as entries:
                    for entry in entries:
                        if inclusion_filters and not any(f in entry.name for f in inclusion_filters):
                            continue
                        if entry_cnt >= entry_cnt_limit:
                            raise Exception(
                                f"Too many files, {entry_cnt_limit} entry limit reached."
                                " Try inclusion_filters."
                            )
                        stat_info = await aiofiles.os.stat(entry.path)
                        rows.append((
                            st_mode_to_symbolic_mode(stat_info.st_mode),
                            str(stat_info.st_nlink),
                            uid_to_name(stat_info.st_uid),
                            gid_to_name(stat_info.st_gid),
                            format_st_size(stat_info.st_size),
                            format_st_mtime(stat_info.st_mtime),
                            entry.name,
                        ))
                        entry_cnt += 1
            else:
                raise Exception(f"Missing or empty path in {arguments}")
        except Exception as e:
            logger.exception(f"Failed to scan directory {arguments}")
            error = str(e)

        # Emulate ls
        if rows:
            max_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
            rows.sort(key=lambda x: x[-1].lower())  # Sort by name
            for row in rows:
                contents.append(
                    f"{row[0]:<{max_widths[0]}} "  # Mode
                    f"{row[1]:>{max_widths[1]}} "  # Links
                    f"{row[2]:<{max_widths[2]}} "  # User
                    f"{row[3]:<{max_widths[3]}} "  # Group
                    f"{row[4]:>{max_widths[4]}} "  # Size
                    f"{row[5]:>{max_widths[5]}} "  # Mtime
                    f"{row[6]}"                    # Name
                )
        output = "\n".join(contents)
        if error:
            output = f"{output}\n\n{error}"
        indexing_task = asyncio.create_task(self.index(context_snapshot, output, arguments["path"]))
        indexing_task.add_done_callback(log_async_task_exception)
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=output,
        )
