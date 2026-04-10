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


class TopicClassifyingObserver(Observer):
    def __init__(
            self,
            llm_client: LLMClient,
            search_client: SearchClient,
            seed_topics: list[str] | None = None,
            **kwargs
    ):
        super().__init__(llm_client, **kwargs)
        self.search_client = search_client
        self.seed_topics = seed_topics or []

    async def developer_message(self, messages: list[ChatMessage]) -> str | None:
        message_parts = [
            "---",
            "## Instructions",
            "You are a topic extraction workflow component. Analyze the most recently received message.",
            "- Generate a `topic_words` list of words or phrases that best convey the topics of the user's message.",
            (
                "- Use other messages from the conversation for context but focus only on the most recent user"
                " message for the `topic_words` list."
            ),
            (
                "- Do not include named entities (e.g., specific people, organizations, products, locations,"
                " events, or works) in `topic_words`."
            ),
        ]
        max_example_topics = 10
        example_topics = [topic for topic in self.seed_topics if topic][:max_example_topics]
        last_user_message = next((msg for msg in reversed(messages) if msg.role == "user"), None)
        if last_user_message and len(example_topics) < max_example_topics:
            search_text = await run_in_threadpool(normalize_text_for_search, last_user_message.content)
            search_emb = (await get_query_embs((search_text,)))[0]
            similar_topics = await self.search_client.search_topics(
                search_text,
                search_emb,
                excluded_topics=self.seed_topics,
                keyword_match_boost=1.0,
                knn_boost=3.0,
                lexical_match_boost=1.0,
                min_lexical_score=0.0,  # Rely on semantic similarity, rather than lexical similarity
            )
            knn_example_limit = max_example_topics - len(example_topics)
            example_topics.extend(similar_topics[:knn_example_limit])
        if example_topics:
            message_parts.append(f"- For example: {example_topics[:max_example_topics]}")
        return "\n".join(message_parts)

    async def get_topics(self) -> list[str]:
        try:
            if self.bg_task:
                await self.bg_task
            if self.response_text:
                data = json.loads(self.response_text)
                topic_words = data["topic_words"]
                assert isinstance(topic_words, list) and all(isinstance(word, str) for word in topic_words), (
                    f"Invalid `topic_words`: expected list[str], got {self.response_text}"
                )
                return topic_words
        except Exception:
            logger.exception(f"Failed to get topic words from '{self.response_text}'")
        return []

    def text_param(self) -> ResponseTextConfigParam:
        return ResponseTextConfigParam(
            format=ResponseFormatTextJSONSchemaConfigParam(
                name="topic_words",
                type="json_schema",
                schema={
                    "type": "object",
                    "properties": {
                        "topic_words": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A flat list of topic words or phrases.",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["topic_words"],
                },
                strict=True,
            ),
            verbosity="low",
        )
