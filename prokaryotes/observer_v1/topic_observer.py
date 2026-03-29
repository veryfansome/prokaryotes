import json
import logging
from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseTextConfigParam,
)

from prokaryotes.llm_v1 import LLMClient
from prokaryotes.observer_v1.base import Observer

logger = logging.getLogger(__name__)


class TopicClassifyingObserver(Observer):
    def __init__(self, llm_client: LLMClient, **kwargs):
        super().__init__(llm_client, **kwargs)

    def developer_message(self) -> str | None:
        message_parts = [
            "---",
            "## Instructions",
            (
                "Consider the most recently received message."
                " Pick 1-3 words (or phrases) that best conveys the current topic."
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
                            "description": "A flat list of atomic topic words or phrases.",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["topic_words"],
                },
                strict=True,
            ),
            verbosity="low",
        )
