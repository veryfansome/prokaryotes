from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    UTC,
    datetime,
)
from enum import Enum
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    Field,
)

from prokaryotes.utils_v1.os_utils import (
    get_cwd,
    get_platform,
    get_process_uid,
    get_python_version,
    uid_to_name,
)


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
    user_id: int | None = None


class PromptPayload(BaseModel):
    conversation_uuid: str
    messages: list[ChatMessage]


@dataclass(frozen=True)
class PromptContext:
    cwd: str
    latitude: float
    longitude: float
    platform_short: str
    python_version: str
    received_at: datetime
    time_zone: ZoneInfo
    unix_usr: tuple[int, str]

    @classmethod
    def new(cls, latitude: float, longitude: float, time_zone: str) -> PromptContext:
        return cls(
            cwd=get_cwd(),
            latitude=latitude,
            longitude=longitude,
            platform_short=get_platform(),
            python_version=get_python_version(),
            received_at=datetime.now(UTC),
            time_zone=ZoneInfo("UTC" if not time_zone else time_zone),
            unix_usr=uid_to_name(get_process_uid()),
        )


class PromptDoc(BaseModel):
    about: list[str]
    created_at: datetime
    doc_id: str | None = Field(default=None, exclude=True)
    labels: list[str] = Field(default_factory=list)
    messages: list[ChatMessage]


class ResponseDoc(BaseModel):
    about: list[str]
    created_at: datetime
    doc_id: str | None = Field(default=None, exclude=True)
    labels: list[str] = Field(default_factory=list)
    text: str


class TextEmbeddingPrompt(Enum):
    DOCUMENT = "document"
    QUERY = "query"


class TextEmbeddingRequest(BaseModel):
    batch_size: int = 1
    prompt: TextEmbeddingPrompt
    texts: list[str]
    truncate_to: int | None = None


class TextEmbeddingResponse(BaseModel):
    embs: list[list[float]]


class ToolCallAnchor(BaseModel):
    text: str


class ToolCallDoc(BaseModel):
    anchors: list[ToolCallAnchor]
    doc_id: str | None = Field(default=None, exclude=True)
    label_cnt: int
    labels: list[str] = Field(default_factory=list)
    output: str
    updated_at: datetime

    def add_anchor(self, anchor_text: str):
        self.anchors.append(ToolCallAnchor(text=anchor_text))
