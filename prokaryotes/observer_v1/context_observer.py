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

class ContextFilteringObserver(Observer):
    def __init__(self, messages: list[ChatMessage], llm_client: LLMClient, **kwargs):
        super().__init__(llm_client, **kwargs)
        self.messages = messages
        self.messages_len = len(messages)

    def developer_message(self) -> str | None:
        message_parts = [
            "---",
            "## Instructions",
            (
                "Analyze the most recent message and the previous messages it depends on."
                
                " Select the *smallest* possible subset of messages required to to understand the most recent message."
                
                f" Messages are indexed from 0 (oldest) to {self.messages_len - 1} (most recent)."
                
                " It is okay to return an `index_positions` array with only a single index position or even an empty"
                " one. If removing a message does not create ambiguity about the meaning or intent of the most recent"
                " message, exclude it from the `index_positions` array."

            ),
        ]
        return "\n".join(message_parts)

    async def get_filtered_context(self) -> list[ChatMessage]:
        valid_values = list(range(self.messages_len))
        try:
            if self.bg_task:
                await self.bg_task
                data = json.loads(self.response_text)
                selected_positions = [idx for idx in sorted(list(set(data["index_positions"]))) if idx in valid_values]
                if valid_values[-1] not in selected_positions:
                    selected_positions.append(valid_values[-1])  # Append last if excluded
                return [self.messages[idx] for idx in selected_positions]
        except Exception:
            logger.exception(f"Failed to get index positions from '{self.response_text}'")
        return self.messages[-2:]  # Fallback to last 2

    def text_param(self) -> ResponseTextConfigParam:
        return ResponseTextConfigParam(
            format=ResponseFormatTextJSONSchemaConfigParam(
                name="message_filter",
                type="json_schema",
                schema={
                    "type": "object",
                    "properties": {
                        "index_positions": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "description": (
                                "The index positions of every message in the provided conversation that is required"
                                " to understand the most recent message."
                            ),
                        },
                    },
                    "additionalProperties": False,
                    "required": ["index_positions"],
                },
                strict=True,
            ),
            verbosity="low",
        )
