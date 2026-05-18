from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal, Protocol

from anthropic.types.tool_param import ToolParam as AnthropicToolParam
from openai.types.responses import FunctionToolParam as OpenAIFunctionToolParam
from pydantic import BaseModel, Field

from prokaryotes.conversation_v1.models import (
    ProjectedItem,
    TurnItem,
)


class IncomingMessage(BaseModel):
    """Web wire format. Slack consumes Slack events directly and never builds these.

    `source_id` is omitted on newly-authored messages (typed fresh, edit, regenerate); the syncer assigns one on
    first encounter. `author_id` and `display_name` are server-derived from session info and are not on the wire —
    see `NormalizedMessage`.
    """

    role: Literal["user", "assistant"]
    content: str
    source_id: str | None = None


class IncomingConversation(BaseModel):
    """Wire payload for `POST /chat`."""

    conversation_uuid: str
    snapshot_uuid: str | None = None
    messages: list[IncomingMessage] = Field(default_factory=list)


class CompactionStatusResponse(BaseModel):
    """Response body for the `/compaction-status` polling endpoint.

    `snapshot_uuid` is populated only when the swap committed a *direct child* of the polled-for
    `pending_snapshot_uuid` — that's the relabel target. In all other "done" outcomes (lock released without a
    commit, snapshot evicted, parent mismatch) the field is omitted; the client clears its indicator without
    relabeling.
    """

    done: bool
    snapshot_uuid: str | None = None


class FunctionToolCallback(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def system_message_parts(self) -> list[str]: ...

    @property
    def tool_spec(self) -> ToolSpec: ...

    async def call(self, arguments: str, call_id: str) -> TurnItem | None: ...


class LLMClient(Protocol):
    async def complete(
        self,
        items: list[ProjectedItem],
        instruction: str | None,
        model: str,
        reasoning_effort: str | None = None,
    ) -> str: ...

    async def stream_turn(
        self,
        items: list[ProjectedItem],
        instruction: str | None,
        model: str,
        max_tool_call_rounds: int | None = None,
        on_committed_turn_item: Callable[[TurnItem], None] | None = None,
        on_final_assistant_message: Callable[[str], None] | None = None,
        on_usage: Callable[[int, int], None] | None = None,
        reasoning_effort: str | None = None,
        stream_ndjson: bool = False,
        tool_callbacks: dict[str, FunctionToolCallback] | None = None,
    ) -> AsyncGenerator[str, Any]: ...


class ToolParameters(BaseModel):
    additionalProperties: bool = False
    properties: dict[str, object]
    required: list[str] = Field(default_factory=list)
    type: Literal["object"] = "object"


def _anthropic_input_schema(schema: object) -> object:
    """Return an Anthropic-compatible copy of a JSON schema fragment.

    Anthropic's custom tool schema accepts a narrower keyword set than OpenAI's. Integer properties currently reject
    `minimum`, so strip that keyword wherever the schema allows integers while leaving the original intact for
    providers that accept it.
    """
    if isinstance(schema, dict):
        sanitized = {key: _anthropic_input_schema(value) for key, value in schema.items()}
        schema_type = sanitized.get("type")
        allows_integer = schema_type == "integer" or (isinstance(schema_type, list) and "integer" in schema_type)
        if allows_integer:
            sanitized.pop("minimum", None)
        return sanitized
    if isinstance(schema, list):
        return [_anthropic_input_schema(item) for item in schema]
    return schema


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: ToolParameters
    strict: bool = True

    def to_anthropic_tool_param(self) -> AnthropicToolParam:
        return AnthropicToolParam(
            description=self.description,
            input_schema=_anthropic_input_schema(self.parameters.model_dump()),
            name=self.name,
            strict=self.strict,
            type="custom",
        )

    def to_openai_function_tool_param(self) -> OpenAIFunctionToolParam:
        return OpenAIFunctionToolParam(
            description=self.description,
            name=self.name,
            parameters=self.parameters.model_dump(),
            strict=self.strict,
            type="function",
        )
