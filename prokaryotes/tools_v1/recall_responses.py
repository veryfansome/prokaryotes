import asyncio
import json
import logging
import traceback

from openai.types.responses import FunctionToolParam
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback
from prokaryotes.models_v1 import ResponseDoc
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
            description=(
                "Search saved assistant responses from the user's past interactions."
                " Use this tool to restore context from earlier conversations or to reference what was said before."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Topic words or key phrases that should appear in the saved responses."
                            " Don't use temporal words like \"earlier\"."
                        ),
                    },
                },
                "additionalProperties": False,
                "required": ["queries"],
            },
            strict=True,
        )

    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        error = ""
        response_texts: set[str] = set()
        responses: list[ResponseDoc] = []
        topics: set[str] = set()
        try:
            arguments: dict[str, str] = json.loads(arguments)
            queries = arguments.get("queries", [])
            assert isinstance(queries, list) and all(isinstance(query, str) for query in queries), (
                f"Invalid `queries`: expected list[str], got {arguments}"
            )
            query_embs = await get_query_embs(tuple(queries))

            topic_search_tasks = []
            for idx, query in enumerate(queries):
                topic_search_tasks.append(asyncio.create_task(
                    self.search_client.search_topics(
                        query, query_embs[idx],
                        min_lexical_score=0.75,
                    )
                ))
            for topic_results in await asyncio.gather(*topic_search_tasks):
                topics.update(topic_results)

            response_search_tasks = []
            for idx, query in enumerate(queries):
                response_search_tasks.append(asyncio.create_task(
                    self.search_client.search_responses(
                        about_or=list(topics),
                        labels_and=[f"user:{self.user_id}"],
                        match=query,
                        match_emb=query_embs[idx],
                        min_score=0.75,
                    )
                ))
            for docs in await asyncio.gather(*response_search_tasks):
                for doc in  docs:
                    if doc.text in response_texts:
                        continue
                    else:
                        response_texts.add(doc.text)
                        responses.append(doc)
        except Exception:
            error = traceback.format_exc().strip()

        output = "Did not find any relevant past responses."
        if responses:
            output_parts = ["# Responses"]
            for response in responses:
                output_parts.append(f"## From {response.created_at}")
                output_parts.append(response.text)
            output = "\n".join(output_parts)
        if error:
            output = f"{output}\n\nAn error occurred:\n{error}"
        logger.info(f"\n{output}")
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=output
        )
