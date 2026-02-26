import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)
from fastapi.responses import StreamingResponse
from zoneinfo import ZoneInfo

from prokaryotes.callbacks_v1 import SearchEmailFunctionToolCallback
from prokaryotes.graph_v1 import GraphClient
from prokaryotes.llm_v1 import get_llm_client
from prokaryotes.models_v1 import (
    ChatMessage,
    ChatRequest,
)
from prokaryotes.observers_v1 import get_observers
from prokaryotes.search_v1 import (
    PersonDoc,
    SearchClient,
)
from prokaryotes.tool_params_v1 import (
    search_email_tool_param,
    web_search_tool_param,
)
from prokaryotes.utils import log_async_task_exception
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

        person_doc = await self.search_client.get_or_create_user_person_doc(str(0))  # TODO: Implement user_id

        # Common developer message
        developer_message = self.developer_message(
            person_doc,
            latitude=latitude,
            longitude=longitude,
            time_zone=time_zone,
        )

        for observer in get_observers(
            person_doc,
            self.graph_client,
            self.llm_client,
            self.search_client,
        ):
            observe_task = asyncio.create_task(observer.observe(developer_message, request.messages))
            self.background_tasks.add(observe_task)
            observe_task.add_done_callback(log_async_task_exception)
            observe_task.add_done_callback(self.background_tasks.discard)

        # Foreground only developer message
        foreground_developer_message_parts = [
            developer_message,
            "---",
            "## Assistant behavior",
            # TODO: Maybe this should be guided by user preferences
            "- Send short messages (1-2 sentences), unless more is specifically requested.",
        ]
        if not person_doc.facts:
            foreground_developer_message_parts.append("- *suggestion*: Ask for the user's name.")
        foreground_developer_message = "\n".join(foreground_developer_message_parts)
        logger.info(f"Foreground developer message:\n{foreground_developer_message}")

        foreground_context_window = [ChatMessage(role="developer", content=foreground_developer_message)]
        # TODO: Roll long contexts off but in a way that can be recalled
        foreground_context_window.extend(request.messages)
        return StreamingResponse(
            self.llm_client.stream_response(
                foreground_context_window, model,
                reasoning_effort=reasoning_effort,
                tool_callbacks=self.tools_callbacks,
                tool_params=self.tools_params,
            ),
            media_type="text/event-stream",
        )

    @classmethod
    def developer_message(
            cls,
            person_doc: PersonDoc,
            latitude: float = None,
            longitude: float = None,
            time_zone: str = None,
    ):
        time_zone = ZoneInfo("UTC" if not time_zone else time_zone)
        message_parts = [f"Current time: {datetime.now(tz=time_zone).strftime('%Y-%m-%d %H:%M')} {time_zone}"]
        if latitude and longitude:
            message_parts.append(f"User location: {latitude:.4f}, {longitude:.4f}")

        message_parts.append("---")
        message_parts.append("## Recalled user info")
        if not person_doc.facts:
            message_parts.append(f"- Nothing is known about this user.")
        else:
            # TODO: Ranking needed when fact lists grow long
            for fact_doc in person_doc.facts:
                message_parts.append(f"- {fact_doc.created_at.astimezone(time_zone).strftime('%Y-%m-%d %H:%M')}: {fact_doc.text}")
        return "\n".join(message_parts)

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
