import asyncio
import json
import logging
from abc import ABC
from openai.types.responses import (
    FunctionToolParam as OpenAIFunctionToolParam,
    ResponseFormatTextJSONSchemaConfigParam as OpenAIResponseFormatTextJSONSchemaParam,
    ResponseTextConfigParam as OpenAIResponseTextConfigParam,
)

from prokaryotes.callbacks_v1 import SaveUserFactsFunctionToolCallback
from prokaryotes.llm_v1 import (
    FunctionToolCallback,
    LLMClient,
)
from prokaryotes.models_v1 import (
    ChatMessage,
    FactDoc,
    ChatCompletionContext,
)
from prokaryotes.search_v1 import (
    PersonContext,
    SearchClient,
)
from prokaryotes.utils import (
    developer_message_parts,
    log_async_task_exception,
)

logger = logging.getLogger(__name__)

class Observer(ABC):
    def __init__(self, llm_client: LLMClient, model: str = "gpt-5.1"):
        self.bg_task = None
        self.llm_client = llm_client
        self.model = model
        self.response_text = ""

    def developer_message(self) -> str | None:
        pass

    async def observe(self, messages: list[ChatMessage]):
        context_window = []
        developer_message = self.developer_message()
        if developer_message:
            logger.debug(f"{self.__class__.__name__} developer message:\n{developer_message}")
            context_window.append(ChatMessage(role="developer", content=developer_message))
        # TODO: Roll long contexts off but in a way that can be recalled
        context_window.extend(messages)

        async for chunk in self.llm_client.stream_response(
                context_window, self.model,
                reasoning_effort=self.reasoning_effort(),
                text=self.text_param(),
                tool_callbacks=self.tool_callbacks(),
                tool_params=self.tool_params(),
        ):
            self.response_text += chunk
        logger.info(f"{self.__class__.__name__} response text: {self.response_text}")

    def observe_in_background(self, messages: list[ChatMessage]):
        self.bg_task = asyncio.create_task(self.observe(messages))
        self.bg_task.add_done_callback(log_async_task_exception)

    def reasoning_effort(self) -> str:
        return "none"

    def text_param(self) -> OpenAIResponseTextConfigParam:
        return OpenAIResponseTextConfigParam(verbosity="low")

    def tool_callbacks(self) -> dict[str, FunctionToolCallback]:
        return {}

    def tool_params(self) -> list[OpenAIFunctionToolParam]:
        return []

class TopicClassifyingObserver(Observer):
    def __init__(self, llm_client: LLMClient, **kwargs):
        super().__init__(llm_client, **kwargs)

    def developer_message(self) -> str | None:
        message_parts = [
            "---",
            "## Assistant instructions",
            (
                "Consider the most recently received message."
                " Pick 1-3 words (or phrases) that best conveys the current topic."
            ),
        ]
        return "\n".join(message_parts)

    async def get_topics(self) -> list[str]:
        try:
            await self.bg_task
            data = json.loads(self.response_text)
            return data["topic_words"]
        except Exception:
            logger.exception(f"Failed to get topic words from '{self.response_text}'")
            return []

    def text_param(self) -> OpenAIResponseTextConfigParam:
        return OpenAIResponseTextConfigParam(
            format=OpenAIResponseFormatTextJSONSchemaParam(
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

class UserFactsSavingObserver(Observer):
    def __init__(
            self,
            completion_context: ChatCompletionContext,
            user_context: PersonContext,
            llm_client: LLMClient,
            search_client: SearchClient,
            **kwargs
    ):
        super().__init__(llm_client, **kwargs)
        self.callback = SaveUserFactsFunctionToolCallback(user_context, search_client)
        self.completion_context = completion_context
        self.user_context = user_context

    def developer_message(self) -> str | None:
        message_parts = developer_message_parts(self.completion_context, self.user_context)
        message_parts.append("---")
        message_parts.append("## Assistant instructions")
        if self.user_context.facts:
            message_parts.append(
                "Consider what is already known about the user in the \"User info\" section above."
                " If the most recent message reveals new facts, call the `save_user_facts` function tool"
                " to add these new fact to the \"User info\" section."
            )
        else:
            message_parts.append(
                "Consider the most recently received message. If the user has revealed information about themselves,"
                " call the `save_user_facts` function tool to add this information to the \"User info\" section."
            )
        return "\n".join(message_parts)

    async def get_saved_facts(self) -> list[FactDoc]:
        try:
            await self.bg_task
            return self.callback.saved_facts
        except Exception:
            logger.exception(f"Failed to get saved facts")
            return []

    def reasoning_effort(self) -> str:
        # Supported values are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`.
        fact_cnt = len(self.user_context.facts)
        if fact_cnt <= 10:
            return "none"
        elif fact_cnt <= 20:
            return "low"
        elif fact_cnt <= 30:
            return "medium"
        else:
            return "high"

    def text_param(self) -> OpenAIResponseTextConfigParam:
        return OpenAIResponseTextConfigParam(format=OpenAIResponseFormatTextJSONSchemaParam(
            name="tool_called",
            type="json_schema",
            schema={
                "type": "object",
                "properties": {
                    "tool_called": {
                        "type": "string",
                        "description": "True, if the `save_user_facts` function tool was called, else False.",
                        "enum": ["True", "False"],
                    },
                },
                "additionalProperties": False,
                "required": ["tool_called"],
            },
            strict=True,
        ))

    def tool_callbacks(self) -> dict[str, FunctionToolCallback]:
        return {"save_user_facts": self.callback}

    def tool_params(self) -> list[OpenAIFunctionToolParam]:
        return [
            OpenAIFunctionToolParam(
                type="function",
                name="save_user_facts",
                description=(
                    "Add new facts to the user's \"User info\" section."
                    " Call this function whenever the user mentions private information that can't be easily looked up."
                    
                    " This includes knowledge about the user or their personal life, including:"
                    " family, friends, colleagues, past events, opinions and preferences,"
                    " hobbies, goals, projects, and more."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "facts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "A flat list of atomic facts, stating information about the user in the"
                                " simplest language possible."
                            ),
                        },
                    },
                    "additionalProperties": False,
                    "required": ["facts"],
                },
                strict=True,
            )
        ]

# TODO: Flesh out question saving
class UserQuestionsSavingObserver(Observer):
    def __init__(
            self,
            completion_context: ChatCompletionContext,
            user_context: PersonContext,
            llm_client: LLMClient,
            search_client: SearchClient,
            **kwargs
    ):
        super().__init__(llm_client, **kwargs)
        self.completion_context = completion_context
        self.search_client = search_client
        self.user_context = user_context


    def developer_message(self) -> str | None:
        pass

    def reasoning_effort(self) -> str:
        return "none"

    def text_param(self) -> OpenAIResponseTextConfigParam:
        return OpenAIResponseTextConfigParam(verbosity="low")

    def tool_callbacks(self) -> dict[str, FunctionToolCallback]:
        return {}

    def tool_params(self) -> list[OpenAIFunctionToolParam]:
        return []
