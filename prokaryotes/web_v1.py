import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from openai.types.responses import FunctionToolParam, WebSearchToolParam
from zoneinfo import ZoneInfo

from prokaryotes.callbacks_v1 import SearchEmailFunctionToolCallback
from prokaryotes.llm_v1 import get_llm_client
from prokaryotes.models_v1 import ChatMessage, ChatRequest
from prokaryotes.observers_v1 import Observer
from prokaryotes.utils import log_async_task_exception
from prokaryotes.web_base import ProkaryotesBase

logger = logging.getLogger(__name__)

class ProkaryoteV1(ProkaryotesBase):

    tools_callbacks = {
        "search_email": SearchEmailFunctionToolCallback(),
    }

    tools_params = [
        FunctionToolParam(
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
        ),
        WebSearchToolParam(
            type="web_search",
            filters={
                "allowed_domains": [
                    "en.wikipedia.org"
                ]
            }
        )
    ]

    @classmethod
    def developer_message(
            cls,
            latitude: float = None,
            longitude: float = None,
            time_zone: str = None,
    ):
        time_zone = ZoneInfo("UTC" if not time_zone else time_zone)
        message_parts = [f"Current time: {datetime.now(tz=time_zone).strftime("%Y-%m-%d %H:%M")} {time_zone}"]
        if latitude and longitude:
            message_parts.append(f"User location: {latitude:.4f}, {longitude:.4f}")
        logger.debug(f"Developer message parts: {message_parts}")
        return "\n".join(message_parts)

    def __init__(self, ui_filename: str):
        self._ui_filename = ui_filename
        self.background_tasks: set[asyncio.Task] = set()
        self.llm_client = get_llm_client()
        self.observers: list[Observer] = [
                Observer(
                llm_client=self.llm_client,
                tool_callbacks={},
                tool_params=[
                    FunctionToolParam(
                        type="function",
                        name="save_user_context",
                        description=(
                            "Save information about the user. This includes anything about the user or their personal life"
                            ", including: family, friends, colleagues, past events, opinions and preferences, hobbies, goals"
                            ", projects, and more."
                            " Call this function whenever the user volunteers specific information that cannot be found elsewhere."
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "context_summary": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "A flat list of atomic, independent facts about the user. Use simple language"
                                        " that clearly articulates what information to save."
                                    ),
                                },
                            },
                            "additionalProperties": False,
                            "required": ["context_summary"],
                        },
                        strict=True,
                    ),
                ],
            ),
        ]

        self.app = FastAPI(lifespan=self.lifespan)
        self.app.add_api_route("/", self.root, methods=["GET"])
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

        for observer in self.observers:
            observe_task = asyncio.create_task(observer.observe(request.messages))
            self.background_tasks.add(observe_task)
            observe_task.add_done_callback(log_async_task_exception)
            observe_task.add_done_callback(self.background_tasks.discard)

        context_window = [
            ChatMessage(
                role="developer",
                content=self.developer_message(
                    latitude=latitude,
                    longitude=longitude,
                    time_zone=time_zone,
                )
            )
        ]
        # TODO: Recall unanswered questions for the user, if any, from Neo4j
        # TODO: roll long contexts off but in a way that can be recalled
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

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info("Entering lifespan")
        yield
        if self.background_tasks:
            done_task, pending_tasks = await asyncio.wait(self.background_tasks, timeout=30.0)
            if pending_tasks:
                logger.warning(f"Exiting with {len(pending_tasks)} tasks pending")
        logger.info("Exiting lifespan")

    def ui_filename(self) -> str:
        return self._ui_filename
