import json
import logging
from openai.types.responses import FunctionToolParam
from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback
from prokaryotes.models_v1 import ChatMessage

logger = logging.getLogger(__name__)


class ChatHistoryCallback(FunctionToolCallback):
    def __init__(self, messages: list[ChatMessage]):
        self.messages = messages
        self.valid_roles = ["all", "assistant", "user"]

    @property
    def tool_param(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name="get_chat_history",
            description="Get a full or filtered view of the current conversation.",
            parameters={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": (
                            "Use \"all\" if all messages should be returned."
                            " Use \"assistant\" or \"user\" to filter on a specific role."
                        ),
                        "enum": self.valid_roles,
                    },
                },
                "additionalProperties": False,
                "required": ["role"],
            },
            strict=True,
        )

    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        history = self.messages
        role_filter = None
        try:
            arguments: dict[str, str] = json.loads(arguments)
            unsafe_role = str(arguments.get("role", "all")).lower()
            if unsafe_role in self.valid_roles and unsafe_role != "all":
                role_filter = unsafe_role
        except Exception:
            logger.warning(f"Failed to get role from {arguments}")
        # TODO: Check context limit with tiktoken
        if role_filter:
            history = [msg for msg in self.messages if msg.role == role_filter]
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=json.dumps([msg.model_dump() for msg in history], indent=2),
        )
