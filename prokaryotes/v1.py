import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from prokaryotes.base import ProkaryotesBase
from prokaryotes.llm_v1 import LLM, get_llm
from prokaryotes.loop_v1 import AgentLoop
from prokaryotes.models_v1 import ChatRequest

logger = logging.getLogger(__name__)

class ProkaryoteV1(ProkaryotesBase):
    def __init__(self, llm: LLM):
        self.agent_loop = AgentLoop(llm)
        self.llm = llm

        self.app = FastAPI(lifespan=self.lifespan)
        self.app.add_api_route("/", self.root, methods=["GET"])
        self.app.add_api_route("/chat", self.chat, methods=["POST"])
        self.app.add_api_route("/health", self.health, methods=["GET"])

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        agent_loop_task = asyncio.create_task(self.agent_loop.run())
        yield
        agent_loop_task.cancel()

    async def chat(self, request: ChatRequest):
        """Chat completion."""
        if len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="At least one message is required")
        return StreamingResponse(self.llm.stream_chat_completion_response(request.messages), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    from prokaryotes.utils import setup_logging

    load_dotenv()
    setup_logging()

    v1 = ProkaryoteV1(llm=get_llm())
    uvicorn.run(v1.app)
