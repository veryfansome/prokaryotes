import asyncio
import json
import logging
from openai.types.responses import FunctionToolParam
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback
from prokaryotes.search_v1 import SearchClient
from prokaryotes.utils_v1.text_utils import get_query_embs

logger = logging.getLogger(__name__)


class RecallResponsesCallback(FunctionToolCallback):
    def __init__(self, search_client: SearchClient, user_id: int):
        self.search_client = search_client
        self.user_id = user_id

    @property
    def tool_param(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name="recall_responses",
            description="Search memory systems for past responses to the user.",
            parameters={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "A flat list of key phrases and/or short sentences for retrieving relevant past responsees"
                            " using text and vector search."
                        ),
                    },
                },
                "additionalProperties": False,
                "required": ["queries"],
            },
            strict=True,
        )

    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        # TODO: Also look up related responses by topic
        error = ""
        response_texts: set[str] = set()
        try:
            arguments: dict[str, str] = json.loads(arguments)
            queries = arguments.get("queries", [])
            assert isinstance(queries, list) and all(isinstance(query, str) for query in queries), (
                f"Invalid `queries`: expected list[str], got {arguments}"
            )
            query_embs = await get_query_embs(queries)
            search_tasks = []
            for idx, query in enumerate(queries):
                search_tasks.append(asyncio.create_task(
                    self.search_client.search_responses(
                        labels_and=[f"user:{self.user_id}"],
                        match=query,
                        match_emb=query_embs[idx],
                        min_score=0.75,
                    )
                ))
            for docs in await asyncio.gather(*search_tasks):
                response_texts.update(doc.text for doc in docs)
        except Exception as e:
            logger.exception("Failed to retrieve past responses")
            error = str(e)

        output = "\n".join(f"- {text}" for text in response_texts)
        if error:
            output = f"{output}\n\n{error}"
        logger.info(f"\n{output}")
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=output
        )
