import json
import logging
from abc import ABC, abstractmethod

import anyio
from openai.types.responses import (
    FunctionToolParam,
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseTextConfigParam,
)
from starlette.concurrency import run_in_threadpool

from prokaryotes.llm_v1 import FunctionCallOutputIndexer, LLMClient
from prokaryotes.models_v1 import (
    ChatMessage,
    ToolCallDoc,
)
from prokaryotes.search_v1 import SearchClient
from prokaryotes.utils_v1.text_utils import (
    get_document_embs,
    normalize_text_for_search,
)

logger = logging.getLogger(__name__)


class PathBasedFunctionCallOutputIndexer(FunctionCallOutputIndexer, ABC):
    """
    A base FunctionToolCallback class that provides a common index method for tools that take a filesystem path as
    their primary argument.
    """
    @property
    @abstractmethod
    def llm_client(self) -> LLMClient:
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        pass

    @property
    @abstractmethod
    def tool_param(self) -> FunctionToolParam:
        pass

    @property
    @abstractmethod
    def search_client(self) -> SearchClient:
        pass

    def additional_labels(self, arguments: dict[str, str]) -> list[str]:
        return []

    async def get_anchoring_decision(
            self,
            prompt_messages: list[ChatMessage],
            last_user_message: ChatMessage,
            path: str,
    ):
        gatekeeper_developer_message = (
            "---\n"
            "## Instructions\n"
            f"The `{self.tool_param['name']}` tool, was just called with `{path}` as the `path` argument."
            
            f" Decide if the text and embedding of \"{last_user_message.content}\", should be saved as an \"anchor\""
            f" for recalling the output of `{self.tool_param['name']}` called on `{path}`."
            
            f" Choose *Yes* if the user names, describes the contents of, or refers to some aspects of `{path}`."

            " Choose *No* if the user's language is vague and can refer something else in a different context."
        )
        logger.debug(f"{self.__class__.__name__} gatekeeper developer message:\n{gatekeeper_developer_message}")
        prompt_messages.append(ChatMessage(role="developer", content=gatekeeper_developer_message))
        response_text = ""
        async for chunk in self.llm_client.stream_response(
                prompt_messages, self.model,
                reasoning_effort="none",
                text=ResponseTextConfigParam(
                    format=ResponseFormatTextJSONSchemaConfigParam(
                        name="storage_gatekeeper",
                        type="json_schema",
                        schema={
                            "type": "object",
                            "properties": {
                                "anchoring_decision": {
                                    "type": "string",
                                    "description": (
                                            f"Yes, if \"{last_user_message.content}\" should be saved as recall anchor"
                                            f" for the output of `{self.tool_param['name']}` on `{path}`, else No."
                                    ),
                                    "enum": ["Yes", "No"],
                                },
                            },
                            "additionalProperties": False,
                            "required": ["anchoring_decision"],
                        },
                        strict=True,
                    ),
                    verbosity="low",
                ),
                tool_choice="none",
                tool_params=[self.tool_param],
        ):
            response_text += chunk

        try:
            anchoring_decision = json.loads(response_text)["anchoring_decision"]
            logger.info(
                f"Anchoring decision in {self.__class__.__name__}.index for {path}: {anchoring_decision}"
            )
            return anchoring_decision
        except Exception:
            logger.warning(
                f"Aborting {self.__class__.__name__}.index for {path}, unparsable gatekeeper decision: {response_text}"
            )

    async def index(
            self,
            prompt_messages: list[ChatMessage],
            arguments: str,
            output: str,
    ) -> ToolCallDoc | None:
        try:
            arguments = json.loads(arguments)
            path = self.path_from_arguments(arguments)
            if not path:
                raise Exception(f"Missing, empty, or invalid path in {arguments}")

            last_user_message = next((msg for msg in reversed(prompt_messages) if msg.role == "user"), None)
            if not last_user_message or not last_user_message.content.strip():
                logger.warning(
                    f"Did not find a user message when {self.__class__.__name__}.index"
                    f" was called for {path}, {prompt_messages}"
                )
                return

            anchor_text = await run_in_threadpool(normalize_text_for_search, last_user_message.content)
            logger.debug(f"Anchor text for {self.tool_param['name']}: {anchor_text}")

            path = str(await anyio.Path(path).resolve())
            labels = [
                f"tool:{self.tool_param['name']}",
                f"path:{path}",
            ]
            labels.extend(self.additional_labels(arguments))
            results = await self.search_client.search_tool_call_by_labels(labels)
            if not results:
                doc = await self.search_client.index_tool_call(
                    labels=labels,
                    output=output,
                    anchor_text=path,  # Always allow search by path
                )
                anchoring_decision = await self.get_anchoring_decision(prompt_messages, last_user_message, path)
                if anchoring_decision == "Yes":
                    await self.search_client.update_tool_call(
                        tool_call=doc,
                        anchor_emb=(await get_document_embs([anchor_text]))[0],
                        anchor_text=anchor_text,
                    )
            else:
                if len(results) > 1:
                    logger.warning(f"Multiple ToolCallDoc results found for {labels}: {results}")
                doc = results[0]
                output = output if output != doc.output else None  # Touch timestamp on None, if output did not change
                if not any(anchor_text == anchor.text for anchor in doc.anchors):  # Skip if duplicate anchor
                    anchoring_decision = await self.get_anchoring_decision(prompt_messages, last_user_message, path)
                    if anchoring_decision == "Yes":
                        await self.search_client.update_tool_call(
                            tool_call=doc,
                            anchor_emb=(await get_document_embs([anchor_text]))[0],
                            anchor_text=anchor_text,
                            output=output,
                        )
                        return doc
                await self.search_client.update_tool_call(
                    tool_call=doc,
                    output=output,
                )
            return doc
        except Exception:
            logger.exception(f"Failed to index {self.tool_param['name']} output with {arguments}")

    @classmethod
    def path_from_arguments(cls, arguments: dict[str, str]) -> str | None:
        path = ""
        unsafe_path = arguments.get("path", "")
        if isinstance(unsafe_path, str):
            path = unsafe_path.strip()
        return path if path else None
