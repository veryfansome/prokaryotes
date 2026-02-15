import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from prokaryotes.loop.v1 import Loop

logger = logging.getLogger(__name__)

async_openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
agent_loop = Loop(async_openai)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

async def stream_openai_response(messages: list[ChatMessage]):
    response = await async_openai.chat.completions.create(
        model="gpt-4o",
        messages=[m.model_dump() for m in messages],
        stream=True,
    )
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content

def validate_chat_request(request: ChatRequest):
    """Validate chat request."""
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    agent_loop_task = asyncio.create_task(agent_loop.run())
    yield
    agent_loop_task.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("prokaryotes", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat completion."""
    validate_chat_request(request)
    return StreamingResponse(stream_openai_response(request.messages), media_type="text/event-stream")

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    import logging.config
    import uvicorn

    logging.config.dictConfig({
        "version": 1,
        "formatters": {
            "default": {"format": "%(levelname)s:     %(asctime)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "handlers": ["console"],
            },
        }
    })
    uvicorn.run(app)
