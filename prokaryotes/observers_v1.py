import logging
from abc import ABC, abstractmethod
from openai.types.responses import FunctionToolParam

from prokaryotes.callbacks_v1 import SaveUserFactsFunctionToolCallback
from prokaryotes.graph_v1 import GraphClient
from prokaryotes.llm_v1 import FunctionToolCallback, LLMClient
from prokaryotes.models_v1 import ChatMessage
from prokaryotes.search_v1 import PersonContext, SearchClient

logger = logging.getLogger(__name__)

class Observer(ABC):
    def __init__(self, llm_client: LLMClient, model: str = "gpt-5.1"):
        self.llm_client = llm_client
        self.model = model

    @abstractmethod
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

        async for _ in self.llm_client.stream_response(
                context_window, self.model,
                reasoning_effort=self.reasoning_effort(),
                tool_callbacks=self.tool_callbacks(),
                tool_params=self.tool_params(),
        ):
            pass  # Drop streamed data

    @abstractmethod
    def reasoning_effort(self) -> str:
        return "none"

    @abstractmethod
    def tool_callbacks(self) -> dict[str, FunctionToolCallback]:
        return {}

    @abstractmethod
    def tool_params(self) -> list[FunctionToolParam]:
        return []

class UserFactsSavingObserver(Observer):
    def __init__(self, user_context: PersonContext, llm_client: LLMClient, search_client: SearchClient, **kwargs):
        super().__init__(llm_client, **kwargs)
        self.search_client = search_client
        self.user_context = user_context

    def developer_message(self) -> str | None:
        message_parts = [
            "## User info",
        ]
        if self.user_context.facts:
            for fact_doc in self.user_context.facts:
                message_parts.append(f"- {fact_doc.text}")
        else:
            message_parts.append("Nothing is known about this user.")

        message_parts.append("---")
        message_parts.append("## Assistant instructions")
        if self.user_context.facts:
            message_parts.append(
                "Consider what is already known about the user in the \"User info\" section above."
                " If the most recent message reveals *NEW* facts, call the `save_user_facts` function tool"
                " to add them to the \"User info\" section."
                " The `facts` parameter of the `save_user_facts` function tool should contain *NEW* information only."
                " Do *NOT* duplicate anything that is already in the \"User info\" section."
            )
        else:
            message_parts.append(
                "Consider the most recently received message. If the user has revealed information about themselves,"
                " call the `save_user_facts` function tool to add this information to the \"User info\" section."
            )
        return "\n".join(message_parts)

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

    def tool_callbacks(self) -> dict[str, FunctionToolCallback]:
        return {
            "save_user_facts": SaveUserFactsFunctionToolCallback(self.user_context, self.search_client)
        }

    def tool_params(self) -> list[FunctionToolParam]:
        return [
            FunctionToolParam(
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
    def __init__(self, user_context: PersonContext, llm_client: LLMClient, search_client: SearchClient, **kwargs):
        super().__init__(llm_client, **kwargs)
        self.search_client = search_client
        self.user_context = user_context

    def developer_message(self) -> str | None:
        pass

    def reasoning_effort(self) -> str:
        return "none"

    def tool_callbacks(self) -> dict[str, FunctionToolCallback]:
        return {}

    def tool_params(self) -> list[FunctionToolParam]:
        return []

def get_observers(
        user_context: PersonContext,
        graph_client: GraphClient,
        llm_client: LLMClient,
        search_client: SearchClient,
) -> list[Observer]:
    return [
        UserFactsSavingObserver(user_context, llm_client, search_client),
    ]
