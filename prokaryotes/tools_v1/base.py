import anyio
import logging
from abc import ABC

from prokaryotes.llm_v1 import FunctionToolCallback
from prokaryotes.models_v1 import ChatMessage
from prokaryotes.search_v1 import SearchClient
from prokaryotes.utils_v1.context_utils import find_last_user_message
from prokaryotes.utils_v1.text_utils import normalize_text_for_search_and_embed

logger = logging.getLogger(__name__)

class PathToolCallback(ABC, FunctionToolCallback):
    def __init__(self, search_client: SearchClient):
        self.search_client = search_client

    async def index(self, messages: list[ChatMessage], output: str, path: str):
        # Look for most recent user message
        last_user_message = find_last_user_message(messages)
        if not last_user_message or not last_user_message.content.strip():
            return

        trigger_text, trigger_emb = await normalize_text_for_search_and_embed(last_user_message.content)
        logger.debug(f"Trigger text for {self.tool_param['name']}: {trigger_text}")

        path = await anyio.Path(path).resolve()
        labels = [
            f"tool:{self.tool_param['name']}",
            f"path:{path}",
        ]
        results = await self.search_client.search_tool_call_by_labels(labels)
        if not results:
            await self.search_client.index_tool_call(
                labels=labels,
                output=output,
                trigger_text=trigger_text,
                trigger_emb=trigger_emb,
            )
        else:
            if len(results) > 1:
                logger.warning(f"Multiple ToolCallDoc results found for {labels}: {results}")
            doc = results[0]
            await self.search_client.update_tool_call(doc.doc_id, output, trigger_text, trigger_emb)
