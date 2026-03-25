import logging
from openai.types.responses import (
    FunctionToolParam,
    ResponseTextConfigParam,
)

from prokaryotes.llm_v1 import (
    FunctionToolCallback,
    LLMClient,
)
from prokaryotes.models_v1 import (
    PersonContext,
    PromptContext,
)
from prokaryotes.observer_v1.base import Observer
from prokaryotes.search_v1 import SearchClient

logger = logging.getLogger(__name__)

# TODO: Flesh out this stub
class QuestionSavingObserver(Observer):
    def __init__(
            self,
            prompt_context: PromptContext,
            user_context: PersonContext,
            llm_client: LLMClient,
            search_client: SearchClient,
            **kwargs
    ):
        super().__init__(llm_client, **kwargs)
        self.prompt_context = prompt_context
        self.search_client = search_client
        self.user_context = user_context


    def developer_message(self) -> str | None:
        pass

    def reasoning_effort(self) -> str:
        return "none"

    def text_param(self) -> ResponseTextConfigParam:
        return ResponseTextConfigParam(verbosity="low")

    def tool_callbacks(self) -> dict[str, FunctionToolCallback]:
        return {}

    def tool_params(self) -> list[FunctionToolParam]:
        return []
