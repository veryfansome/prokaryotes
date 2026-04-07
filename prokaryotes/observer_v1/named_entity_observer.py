import json
import logging

from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseTextConfigParam,
)

from prokaryotes.llm_v1 import LLMClient
from prokaryotes.observer_v1.base import Observer

logger = logging.getLogger(__name__)


class NamedEntityObserver(Observer):
    def __init__(self, llm_client: LLMClient, **kwargs):
        super().__init__(llm_client, **kwargs)

    def developer_message(self) -> str | None:
        message_parts = [
            "---",
            "## Instructions",
            (
                "Analyze the most recently received message."
                " - Generate a `named_entities` list that includes all proper names in the message."
                " - Expand any acronyms and pronouns into the fully spelled out proper names they refer to."
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
                            "description": "A flat list of proper names.",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["named_entities"],
                },
                strict=True,
            ),
            verbosity="low",
        )
