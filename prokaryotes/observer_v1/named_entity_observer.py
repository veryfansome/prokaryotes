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


class NamedEntityObserver(Observer):
    def __init__(self, llm_client: LLMClient, **kwargs):
        super().__init__(llm_client, **kwargs)

    def developer_message(self, messages: list[ChatMessage]) -> str | None:
        message_parts = [
            "---",
            "## Instructions",
            "Analyze the most recently received user message.",
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
                " \"the user's friend\", \"the doctor\")."
            ),
            "- Expand any name acronyms from the message into the fully spelled out proper nouns they refer to.",
            (
                "- Replace pronouns with canonical proper names, whenever the object they refer to is clear and"
                " explicitly named elsewhere in the conversation."
            ),
        ]
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
