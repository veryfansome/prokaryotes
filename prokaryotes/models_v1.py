from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from pydantic import (
    BaseModel,
    Field,
)
from zoneinfo import ZoneInfo

@dataclass(frozen=True)
class ChatCompletionContext:
    cwd: str = os.getcwd()
    latitude: float = None
    longitude: float = None
    platform_short: str = platform.platform(terse=True)
    python_version: str = platform.python_version()
    received_at: datetime = datetime.now(timezone.utc)
    time_zone: ZoneInfo = None

    @classmethod
    def new(cls, latitude: float, longitude: float, time_zone: str) -> ChatCompletionContext:
        return cls(latitude=latitude, longitude=longitude, time_zone=ZoneInfo("UTC" if not time_zone else time_zone))

class ChatCompletionDoc(BaseModel):
    about: list[str]
    created_at: datetime
    doc_id: str | None = Field(default=None, exclude=True)
    error: str | None = Field(default=None)
    importance: int = 1
    labels: list[str] = Field(default_factory=list)
    messages: list[ChatMessage]

class ChatCompletionPayload(BaseModel):
    messages: list[ChatMessage]

class ChatMessage(BaseModel):
    role: str
    content: str

class FactDoc(BaseModel):
    about: list[str]
    created_at: datetime
    doc_id: str | None = Field(default=None, exclude=True)
    importance: int = 1
    invalid_after: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    text: str

class PersonContext(BaseModel):
    facts: list[FactDoc] = Field(default_factory=list)
    name: str | None = None
    questions: list[QuestionDoc] = Field(default_factory=list)
    user_id: int | None = None

class QuestionDoc(BaseModel):
    about: list[str]
    created_at: datetime
    doc_id: str | None = Field(default=None, exclude=True)
    importance: int = 1
    invalid_after: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    text: str
    to: list[str]  # Maybe for_?

class TextEmbeddingPrompt(Enum):
    DOCUMENT = "document"
    QUERY = "query"

class TextEmbeddingRequest(BaseModel):
    batch_size: int = 1
    prompt: TextEmbeddingPrompt
    texts: list[str]
    truncate_to: int | None = None

class TextEmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
