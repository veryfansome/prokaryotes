import os
import platform
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

@dataclass(frozen=True)
class ExecutionContext:
    cwd: str = os.getcwd()
    platform_short: str = platform.platform(terse=True)
    python_version: str = platform.python_version()

class FactDoc(BaseModel):
    about: list[str]
    created_at: datetime
    doc_id: str | None = Field(default=None, exclude=True)
    importance: int = 1
    invalid_after: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    text: str

class QuestionDoc(BaseModel):
    about: list[str]
    created_at: datetime
    doc_id: str | None = Field(default=None, exclude=True)
    importance: int = 1
    invalid_after: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    text: str
    to: list[str]  # Maybe for_?

class PersonContext(BaseModel):
    facts: list[FactDoc] = Field(default_factory=list)
    questions: list[QuestionDoc] = Field(default_factory=list)
    user_id: int | None = None

@dataclass(frozen=True)
class RequestContext:
    execution_context: ExecutionContext = ExecutionContext()
    latitude: float = None
    longitude: float = None
    received_at: datetime = datetime.now(timezone.utc)
    time_zone: ZoneInfo = None

    @classmethod
    def new(cls, latitude: float, longitude: float, time_zone: str) -> "RequestContext":
        return cls(latitude=latitude, longitude=longitude, time_zone=ZoneInfo("UTC" if not time_zone else time_zone))

class TextEmbeddingPrompt(Enum):
    DOCUMENT = "document"
    QUERY = "query"

class TextEmbeddingRequest(BaseModel):
    prompt: TextEmbeddingPrompt
    texts: list[str]
