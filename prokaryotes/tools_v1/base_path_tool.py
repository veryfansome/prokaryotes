import anyio
import json
import logging
from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseTextConfigParam,
)
from starlette.concurrency import run_in_threadpool

from prokaryotes.llm_v1 import FunctionToolCallback, LLMClient
from prokaryotes.models_v1 import (
    ChatMessage,
    TextEmbeddingPrompt,
    TextEmbeddingRequest,
)
from prokaryotes.search_v1 import SearchClient
from prokaryotes.utils_v1.text_utils import (
    get_text_embeddings,
    normalize_text_for_search,
)

logger = logging.getLogger(__name__)

class PathToolCallback(FunctionToolCallback):
    """
    A base FunctionToolCallback class that provides a common index method for tools that take a filesystem path as
    their primary argument.
    """
    def __init__(self, llm_client: LLMClient, search_client: SearchClient, model: str = "gpt-5.1"):
        self.llm_client = llm_client
        self.model = model
        self.search_client = search_client

    async def get_anchoring_decision(
            self,
            context_snapshot: list[ChatMessage],
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
        context_snapshot.append(ChatMessage(role="developer", content=gatekeeper_developer_message))
        response_text = ""
        async for chunk in self.llm_client.stream_response(
                context_snapshot, self.model,
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
                                            # f"Yes, if \"{last_user_message.content}\" should be saved as an example"
                                            # f" of when to use `{self.tool_param['name']}` on `{path}`, else No."
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
            context_snapshot: list[ChatMessage],
            output: str,
            path: str,
    ):
        last_user_message = next((msg for msg in reversed(context_snapshot) if msg.role == "user"), None)
        if not last_user_message or not last_user_message.content.strip():
            logger.warning(
                f"Did not find a user message when {self.__class__.__name__}.index"
                f" was called for {path}, {context_snapshot}"
            )
            return

        anchor_text = await run_in_threadpool(normalize_text_for_search, last_user_message.content)
        logger.debug(f"Anchor text for {self.tool_param['name']}: {anchor_text}")

        path = str(await anyio.Path(path).resolve())
        labels = [
            f"tool:{self.tool_param['name']}",
            f"path:{path}",
        ]
        results = await self.search_client.search_tool_call_by_labels(labels)
        if not results:
            doc = await self.search_client.index_tool_call(
                labels=labels,
                output=output,
                anchor_text=path,  # Always allow search by path
            )
            anchoring_decision = await self.get_anchoring_decision(context_snapshot, last_user_message, path)
            if anchoring_decision == "Yes":
                await self.search_client.update_tool_call(
                    doc.doc_id,
                    anchor_emb=(await get_anchor_embedding(anchor_text)),
                    anchor_text=anchor_text,
                )
        else:
            if len(results) > 1:
                logger.warning(f"Multiple ToolCallDoc results found for {labels}: {results}")
            doc = results[0]
            output = output if output != doc.output else None  # Touch timestamp on None, if output did not change
            if not any(anchor_text == anchor.text for anchor in doc.anchors):  # Skip if duplicate anchor
                anchoring_decision = await self.get_anchoring_decision(context_snapshot, last_user_message, path)
                if anchoring_decision == "Yes":
                    await self.search_client.update_tool_call(
                        doc_id=doc.doc_id,
                        anchor_emb=(await get_anchor_embedding(anchor_text)),
                        anchor_text=anchor_text,
                        output=output,
                    )
                    return
            await self.search_client.update_tool_call(
                doc_id=doc.doc_id,
                output=output,
            )

async def get_anchor_embedding(anchor_text: str) -> list[float]:
    return (await get_text_embeddings(TextEmbeddingRequest(
        prompt=TextEmbeddingPrompt.DOCUMENT,
        texts=[anchor_text],
        truncate_to=256,
    ))).embs[0]
