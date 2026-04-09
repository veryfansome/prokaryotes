import json
import logging

from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseTextConfigParam,
)

from prokaryotes.llm_v1 import LLMClient
from prokaryotes.models_v1 import ChatMessage
from prokaryotes.observer_v1.base import Observer

logger = logging.getLogger(__name__)


class TopicClassifyingObserver(Observer):
    def __init__(self, llm_client: LLMClient, **kwargs):
        super().__init__(llm_client, **kwargs)

    def developer_message(self, messages: list[ChatMessage]) -> str | None:
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
