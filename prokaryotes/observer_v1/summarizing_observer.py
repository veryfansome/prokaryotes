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
from prokaryotes.utils_v1.text_utils import normalize_text_for_search

logger = logging.getLogger(__name__)


class MessageSummarizingObserver(Observer):
    def __init__(self, llm_client: LLMClient, **kwargs):
        super().__init__(llm_client, **kwargs)

    async def developer_message(self, messages: list[ChatMessage]) -> str | None:
        message_parts = [
            "---",
            "## Instructions",
            "- Summarize the substance of the most recent user message in as few words as possible.",
            "- Start your summary with \"The user ...\" and complete the sentence in past tense.",
            (
                "- When referencing the assistant, always address it as \"you\" (second person). e.g. \"The user"
                " wanted you to explain Newton's third law.\""
            ),
            "- Focus only on the most recent message.",
            "- Aim for less than 8 words but you can use more if important intent would be lost otherwise.",
            "- Don't use more than 20 words.",
        ]
        return "\n".join(message_parts)

    async def get_summary(self) -> str:
        try:
            if self.bg_task:
                await self.bg_task
            if self.response_text:
                data = json.loads(self.response_text)
                summary = data["summary"]
                assert isinstance(summary, str), (
                    f"Invalid `summary`: expected str, got {self.response_text}"
                )
                return summary
        except Exception:
            logger.exception(f"Failed to get summary from '{self.response_text}'")
        return await run_in_threadpool(normalize_text_for_search, self.observed_messages[-1].content)

    def text_param(self) -> ResponseTextConfigParam:
        return ResponseTextConfigParam(
            format=ResponseFormatTextJSONSchemaConfigParam(
                name="message_summary",
                type="json_schema",
                schema={
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": (
                                "A concise summary of the most recent user message, capturing the user's intent."
                            ),
                        },
                    },
                    "additionalProperties": False,
                    "required": ["summary"],
                },
                strict=True,
            ),
            verbosity="low",
        )
