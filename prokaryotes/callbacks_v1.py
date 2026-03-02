import aiofiles.os
import json
import logging
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback
from prokaryotes.search_v1 import PersonContext, SearchClient

logger = logging.getLogger(__name__)

class ListDirectoryCallback(FunctionToolCallback):
    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        contents = []
        entry_cnt_limit = 100
        error = None
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
                        file_metadata = {"name": entry.name}
                        if entry.is_dir():
                            file_metadata["type"] = "dir"
                        elif entry.is_symlink():
                            file_metadata["type"] = "link"
                        else:
                            file_metadata["type"] = "file"
                        stat_info = await aiofiles.os.stat(entry.path)
                        file_metadata["gid"] = stat_info.st_gid
                        file_metadata["size"] = stat_info.st_size
                        file_metadata["uid"] = stat_info.st_uid
                        contents.append(file_metadata)
                        entry_cnt += 1
            else:
                raise Exception(f"Missing or empty path in {arguments}")
        except Exception as e:
            logger.exception(f"Failed to scan directory {arguments}")
            error = str(e)
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=json.dumps({
                "contents": contents,
                "error": error,
            })
        )

class ReadFileCallback(FunctionToolCallback):
    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        contents = ""
        error = None
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
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=json.dumps({
                "contents": contents,
                "error": error,
            })
        )

class SaveUserFactsFunctionToolCallback(FunctionToolCallback):
    def __init__(self, user_context: PersonContext, search_client: SearchClient):
        self.user_context = user_context
        self.search_client = search_client

    async def call(self, arguments: str, call_id: str) -> None:
        try:
            arguments: dict[str, list[str]] = json.loads(arguments)
            if "facts" in arguments and arguments["facts"]:
                normalized_candidates = []
                for candidate in arguments["facts"]:
                    candidate = " ".join(candidate.strip(" .!?\r\n").split())
                    if candidate:
                        normalized_candidates.append(candidate)
                # Exact dedupe
                existing_fact_texts = {fact.text.casefold() for fact in self.user_context.facts}
                candidates_after_exact_dedupe = [
                    candidate for candidate in normalized_candidates
                    if candidate.casefold() not in existing_fact_texts
                ]
                # TODO: Additional dedupe via semantic similarity or another pass with an LLM (offline?)
                await self.search_client.index_facts([f"user_{self.user_context.user_id}"], candidates_after_exact_dedupe)
            else:
                logging.warning(f"Missing or empty facts in {arguments} (user {self.user_context.user_id})")
        except Exception:
            logging.exception(f"Failed to save user {self.user_context.user_id} facts")
        return None  # No continuation

class SearchEmailFunctionToolCallback(FunctionToolCallback):
    def __init__(self, search_client: SearchClient):
        self.search_client = search_client

    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output='{"messages": []}'
        )
