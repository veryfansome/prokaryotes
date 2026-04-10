import json
import logging

from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseTextConfigParam,
)
from starlette.concurrency import run_in_threadpool

from prokaryotes.llm_v1 import LLMClient
from prokaryotes.models_v1 import ChatMessage
from prokaryotes.observer_v1.base import Observer
from prokaryotes.search_v1 import SearchClient
from prokaryotes.utils_v1.text_utils import (
    get_query_embs,
    normalize_text_for_search,
)

logger = logging.getLogger(__name__)


class NamedEntityObserver(Observer):
    def __init__(
            self,
            llm_client: LLMClient,
            search_client: SearchClient,
            seed_entities: list[str] | None = None,
            **kwargs
    ):
        super().__init__(llm_client, **kwargs)
        self.search_client = search_client
        self.seed_entities = seed_entities or []

    async def developer_message(self, messages: list[ChatMessage]) -> str | None:
        message_parts = [
            "---",
            "## Instructions",
            "You are a named entity extraction workflow component. Analyze the most recently received user message.",
            "- Generate a `named_entities` list of unique objects explicitly mentioned by name.",
            (
                "- Use other messages from the conversation for context but focus only on the most recent user"
                " message for the `named_entities` list."
            ),
            (
                "- Include only proper nouns that can be grounded as specific people, organizations, works,"
                " locations, events, products, etc."
            ),
            (
                "- Do not include generic terms that describe a role or relationship (e.g. \"boss\", \"father\","
                " \"the user's friend\", \"the doctor\") in `named_entities`."
            ),
            "- Expand any name acronyms from the message into the fully spelled out proper nouns they refer to.",
            (
                "- Replace pronouns with canonical proper names, whenever the object they refer to is clear and"
                " explicitly named elsewhere in the conversation."
            ),
        ]
        max_example_entities = 10
        example_entities = [named_entity for named_entity in self.seed_entities if named_entity][:max_example_entities]
        last_user_message = next((msg for msg in reversed(messages) if msg.role == "user"), None)
        if last_user_message and len(example_entities) < max_example_entities:
            search_text = await run_in_threadpool(normalize_text_for_search, last_user_message.content)
            search_emb = (await get_query_embs((search_text,)))[0]
            similar_entities = await self.search_client.search_named_entities(
                search_text,
                search_emb,
                excluded_entities=self.seed_entities,
                keyword_match_boost=1.0,
                knn_boost=3.0,
                lexical_match_boost=1.0,
                min_score=0.0,  # Rely on semantic similarity, rather than lexical similarity
            )
            knn_example_limit = max_example_entities - len(example_entities)
            example_entities.extend(similar_entities[:knn_example_limit])
        if example_entities:
            message_parts.append(f"- For example: {example_entities[:max_example_entities]}")
        return "\n".join(message_parts)

    async def get_named_entities(self) -> list[str]:
        try:
            if self.bg_task:
                await self.bg_task
            if self.response_text:
                data = json.loads(self.response_text)
                named_entities = data["named_entities"]
                assert isinstance(named_entities, list) and all(isinstance(word, str) for word in named_entities), (
                    f"Invalid `named_entities`: expected list[str], got {self.response_text}"
                )
                return named_entities
        except Exception:
            logger.exception(f"Failed to get named entities from '{self.response_text}'")
        return []

    def text_param(self) -> ResponseTextConfigParam:
        return ResponseTextConfigParam(
            format=ResponseFormatTextJSONSchemaConfigParam(
                name="named_entities",
                type="json_schema",
                schema={
                    "type": "object",
                    "properties": {
                        "named_entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A flat list of named entities mentioned in the most recent user message.",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["named_entities"],
                },
                strict=True,
            ),
            verbosity="low",
        )
