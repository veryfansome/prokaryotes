import logging
from openai.types.responses import FunctionToolParam

from prokaryotes.callbacks_v1 import SaveUserContextFunctionToolCallback
from prokaryotes.graph_v1 import GraphClient
from prokaryotes.llm_v1 import FunctionToolCallback, LLMClient
from prokaryotes.models_v1 import ChatMessage
from prokaryotes.search_v1 import PersonDoc, SearchClient
from prokaryotes.tool_params_v1 import save_user_context_tool_param

logger = logging.getLogger(__name__)

class Observer:
    def __init__(
            self,
            llm_client: LLMClient,
            tool_callbacks: dict[str, FunctionToolCallback],
            tool_params: list[FunctionToolParam],
            model: str = "gpt-5.1",
            # TODO: Optional reasoning level
    ):
        self.llm_client = llm_client
        self.model = model
        self.tool_callbacks = tool_callbacks
        self.tool_params = tool_params

    @classmethod
    def adjust_developer_message(cls, developer_message: str) -> str:
        return developer_message

    async def observe(self, developer_message: str, messages: list[ChatMessage]):
        adjusted_developer_message = self.adjust_developer_message(developer_message)
        logger.info(f"{self.__class__.__name__} developer message:\n{adjusted_developer_message}")

        context_window = [ChatMessage(role="developer", content=adjusted_developer_message)]
        # TODO: Roll long contexts off but in a way that can be recalled
        context_window.extend(messages)
        async for _ in self.llm_client.stream_response(
            context_window, self.model,
            tool_callbacks=self.tool_callbacks,
            tool_params=self.tool_params,
        ):
            pass  # Drop streamed data

class UserContextSavingObserver(Observer):
    def __init__(
            self,
            person_doc: PersonDoc,
            llm_client: LLMClient,
            search_client: SearchClient,
            **kwargs
    ):
        super().__init__(
            llm_client,
            {
                save_user_context_tool_param["name"]: SaveUserContextFunctionToolCallback(person_doc, search_client)
            },
            [
                save_user_context_tool_param,
            ],
            **kwargs
        )

    @classmethod
    def adjust_developer_message(cls, developer_message: str) -> str:
        adjusted_message_parts = [
            developer_message,
            "---",
            # TODO: This doesn't seem to work...
            (
                "When using the save_user_context function tool, dedupe against the recalled user info above."
                " Don't save the same information more than once."
            ),
        ]
        return "\n".join(adjusted_message_parts)

def get_observers(
        person_doc: PersonDoc,
        graph_client: GraphClient,
        llm_client: LLMClient,
        search_client: SearchClient,
) -> list[Observer]:
    return [
        UserContextSavingObserver(person_doc, llm_client, search_client),
    ]
