import logging
from datetime import datetime, timedelta
from openai.types.responses import FunctionToolParam
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback
from prokaryotes.models_v1 import ChatMessage
from prokaryotes.search_v1 import SearchClient

logger = logging.getLogger(__name__)

class SearchEmailCallback(FunctionToolCallback):
    def __init__(self, search_client: SearchClient):
        self.search_client = search_client

    @property
    def tool_param(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name="search_email",
            description="Search the user's email using a criteria.",
            parameters={
                "type": "object",
                "properties": {
                    "search_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "A flat list of IMAP search tokens based on RFC 3501 (IMAP4rev1) and RFC 4731 (ESEARCH)"
                            " for the Python imapclient library."
                            f' Example: ["FROM", "John Smith", "SINCE", "{(datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")}"]'
                        ),
                    },
                },
                "additionalProperties": False,
                "required": ["search_criteria"],
            },
            strict=True,
        )

    async def call(self, messages: list[ChatMessage], arguments: str, call_id: str) -> FunctionCallOutput:
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output='{"messages": []}'
        )
