import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from prokaryotes.callbacks_v1 import SearchEmailFunctionToolCallback
from prokaryotes.graph_v1 import GraphClient
from prokaryotes.llm_v1 import get_llm_client
from prokaryotes.models_v1 import (
    ChatMessage,
    ChatRequest,
    RequestContext,
)
from prokaryotes.observers_v1 import get_observers
from prokaryotes.search_v1 import (
    PersonContext,
    SearchClient,
)
from prokaryotes.tool_params_v1 import (
    search_email_tool_param,
    web_search_tool_param,
)
from prokaryotes.utils import (
    developer_message_parts,
    log_async_task_exception,
    prep_chat_message_text_for_search,
)
from prokaryotes.web_base import ProkaryotesBase

logger = logging.getLogger(__name__)

class ProkaryoteV1(ProkaryotesBase):
    def __init__(self, static_dir: str):
        self.graph_client = GraphClient()
        self.llm_client = get_llm_client()
        self.search_client = SearchClient()

        self.background_tasks: set[asyncio.Task] = set()
        self.static_dir = static_dir

        self.tools_callbacks = {
            search_email_tool_param["name"]: SearchEmailFunctionToolCallback(self.search_client),
        }
        self.tools_params = [
            search_email_tool_param,
            web_search_tool_param,
        ]

        self.app = FastAPI(lifespan=self.lifespan)
        self.app.add_api_route("/", self.root, methods=["GET"])
        self.app.add_api_route("/logo.png", self.logo, methods=["GET"])
        self.app.add_api_route("/chat", self.chat, methods=["POST"])
        self.app.add_api_route("/health", self.health, methods=["GET"])

    async def chat(
            self,
            request: ChatRequest,
            latitude: float = Query(None),
            longitude: float = Query(None),
            model: str = Query("gpt-5.1"),
            reasoning_effort: str = Query(None),
            time_zone: str = Query(None),
    ):
        """Chat completion."""
        if len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="At least one message is required")
        request_context = RequestContext.new(latitude=latitude, longitude=longitude, time_zone=time_zone)

        search_query_text = await run_in_threadpool(
            prep_chat_message_text_for_search, " ".join(m.content for m in request.messages if m.role == "user"),
        )
        logger.info(f"Search query text:\n{search_query_text}")
        # TODO: Drop stop word and do a blind text search against facts and questions

        # TODO: Actually implement user_id
        user_context = await self.search_client.get_user_context(1)

        for observer in get_observers(
            request_context,
            user_context,
            self.graph_client,
            self.llm_client,
            self.search_client,
        ):
            observe_task = asyncio.create_task(observer.observe(request.messages))
            self.background_tasks.add(observe_task)
            observe_task.add_done_callback(log_async_task_exception)
            observe_task.add_done_callback(self.background_tasks.discard)

        context_window = [ChatMessage(role="developer", content=self.developer_message(request_context, user_context))]
        # TODO: Roll long contexts off but in a way that can be recalled
        context_window.extend(request.messages)
        return StreamingResponse(
            self.llm_client.stream_response(
                context_window, model,
                reasoning_effort=reasoning_effort,
                tool_callbacks=self.tools_callbacks,
                tool_params=self.tools_params,
            ),
            media_type="text/event-stream",
        )

    @classmethod
    def developer_message(cls, request_context: RequestContext, user_context: PersonContext):
        message_parts = developer_message_parts(request_context, user_context)
        message_parts.append("---")
        message_parts.append("## Assistant instructions")
        # TODO: Maybe this should be guided by user preferences
        message_parts.append("- Use short messages. One sentences is best, unless the user explicitly requests more.")
        message_parts.append("- Don't offer platitudes or untruths.")
        message_parts.append("- Don't project confidence when you are uncertain.")

        message = "\n".join(message_parts)
        logger.info(f"Foreground developer message:\n{message}")
        return message

    def get_static_dir(self) -> str:
        return self.static_dir

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info("Entering lifespan")
        yield
        if self.background_tasks:
            done_task, pending_tasks = await asyncio.wait(self.background_tasks, timeout=30.0)
            if pending_tasks:
                logger.warning(f"Exiting with {len(pending_tasks)} tasks pending")
        await self.graph_client.close()
        await self.search_client.close()
        logger.info("Exiting lifespan")
