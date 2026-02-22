import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from openai.types.responses import FunctionToolParam, WebSearchToolParam
from zoneinfo import ZoneInfo

from prokaryotes.llm_v1 import get_llm
from prokaryotes.models_v1 import ChatMessage, ChatRequest
from prokaryotes.web_base import ProkaryotesBase

logger = logging.getLogger(__name__)

class ProkaryoteV1(ProkaryotesBase):

    tools_spec = [
        # TODO: Function that triggers when the user mentions non-public information or opinions
        FunctionToolParam(
            type="function",
            name="get_horoscope",
            description="Get today's horoscope for an astrological sign.",
            parameters={
                "type": "object",
                "properties": {
                    "sign": {
                        "type": "string",
                        "description": "An astrological sign like Taurus or Aquarius",
                    },
                },
                "additionalProperties": False,
                "required": ["sign"],
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
        self.llm = get_llm()

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
        # TODO: Recall unanswered questions for the user, if any, from Neo4j
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
        # TODO: roll long contexts off but in a way that can be recalled
        context_window.extend(request.messages)
        return StreamingResponse(
            self.llm.stream_response(
                context_window, model,
                reasoning_effort=reasoning_effort,
                tool_spec=self.tools_spec,
            ),
            media_type="text/event-stream",
        )

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info("Entering lifespan")
        yield
        logger.info("Exiting lifespan")

    def ui_filename(self) -> str:
        return self._ui_filename
