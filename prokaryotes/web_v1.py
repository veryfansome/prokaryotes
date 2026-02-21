import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from prokaryotes.base import ProkaryotesBase
from prokaryotes.llm_v1 import get_llm
from prokaryotes.models_v1 import ChatRequest

logger = logging.getLogger(__name__)

class ProkaryoteV1(ProkaryotesBase):
    def __init__(self):
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
            time_zone: str = Query(None),
    ):
        """Chat completion."""
        if len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="At least one message is required")
        return StreamingResponse(
            self.llm.stream_response(
                request.messages,
                latitude=latitude,
                longitude=longitude,
                time_zone=time_zone,
            ),
            media_type="text/event-stream",
        )

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info("Entering lifespan")
        yield
        logger.info("Exiting lifespan")

    @classmethod
    def ui_filename(cls) -> str:
        return "ui_v1.html"
