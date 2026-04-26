from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable
from enum import Enum
from typing import (
    Literal,
    Protocol,
)

from anthropic.types.tool_param import ToolParam as AnthropicToolParam
from openai.types.responses import FunctionToolParam as OpenAIFunctionToolParam
from pydantic import (
    BaseModel,
    Field,
)


class ChatConversation(BaseModel):
    conversation_uuid: str
    partition_uuid: str | None = None
    messages: list[ChatMessage]

    def to_context_partition(self) -> ContextPartition:
        return ContextPartition(
            conversation_uuid=self.conversation_uuid,
            items=[message.to_context_partition_item() for message in self.messages]
        )


class ChatMessage(BaseModel):
    content: str
    role: str

    def to_context_partition_item(self):
        return ContextPartitionItem(content=self.content, role=self.role)


class ContextPartition(BaseModel):
    """A provider-agnostic conversation history as a list of `ContextPartitionItem` objects."""

    conversation_uuid: str
    partition_uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_partition_uuid: str | None = None
    ancestor_summaries: list[str] = Field(default_factory=list)
    raw_message_start_index: int = 0
    items: list[ContextPartitionItem]

    def append(self, item: ContextPartitionItem):
        self.items.append(item)

    def extend(self, items: Iterable[ContextPartitionItem]):
        self.items.extend(items)

    @staticmethod
    def find_context_divergence(
            context_items: list[ContextPartitionItem],
            conversation_items: list[ContextPartitionItem],
    ) -> tuple[int | None, bool, bool]:
        context_items_len = len(context_items)
        conversation_items_len = len(conversation_items)
        for idx, (context_item, conversation_item) in enumerate(zip(context_items, conversation_items, strict=False)):
            if context_item != conversation_item:
                return idx, True, context_items_len > conversation_items_len
        if context_items_len == conversation_items_len:
            return None, False, False
        else:
            return min(context_items_len, conversation_items_len), False, context_items_len > conversation_items_len

    def message_items_for_sync(self) -> tuple[list[ContextPartitionItem], list[int]]:
        partition_messages = []
        partition_message_indexes = []
        for idx, item in enumerate(self.items):
            if item.type == "message" and item.role in {"user", "assistant"}:
                partition_messages.append(item)
                partition_message_indexes.append(idx)
        return partition_messages, partition_message_indexes

    def pop_system_message(self):
        if self.items and self.items[0].role in {"developer", "system"}:
            self.items.pop(0)

    def sync_from_conversation(self, conversation: ChatConversation):
        if len(conversation.messages) < self.raw_message_start_index:
            raise ConversationOutsideRawWindowError(
                "Conversation ends before this partition's raw message window"
            )

        partition_messages, partition_message_indexes = self.message_items_for_sync()
        conversation_items = [
            message.to_context_partition_item()
            for message in conversation.messages[self.raw_message_start_index:]
        ]
        divergence_idx, is_mismatch, is_longer = self.find_context_divergence(
            partition_messages,
            conversation_items,
        )
        if divergence_idx is None:
            raise ConversationMatchesPartitionError("Conversation does not alter partition state")

        if self.raw_message_start_index > 0 and divergence_idx == 0 and is_mismatch:
            raise ConversationOutsideRawWindowError(
                "Conversation diverged at the compacted/raw boundary"
            )

        if not partition_message_indexes:
            self.items.extend(conversation_items)
            return

        truncate_at = (
            partition_message_indexes[divergence_idx]
            if divergence_idx < len(partition_message_indexes)
            else partition_message_indexes[-1] + 1
        )
        self.items = self.items[:truncate_at]
        if is_longer and not is_mismatch:
            return
        self.items.extend(conversation_items[divergence_idx:])

    def to_anthropic_messages(self):
        """Convert ContextPartition to Anthropic (system, messages) format."""
        system_parts: list[str] = list(self.ancestor_summaries)
        messages: list[dict] = []
        current_role: str | None = None
        current_content: list[dict] = []

        def flush():
            nonlocal current_role, current_content
            if current_role and current_content:
                messages.append({"role": current_role, "content": current_content})
            current_role, current_content = None, []

        for item in self.items:
            if item.type == "message":
                if item.role == "system":
                    flush()
                    if item.content:
                        system_parts.append(item.content)
                    continue
                if item.role not in {"user", "assistant"}:
                    raise ValueError(f"Unsupported role: {item.role!r}")
                role, block = item.role, {"type": "text", "text": item.content or ""}
            elif item.type == "function_call":
                call_id = item.call_id or item.id
                if call_id is None:
                    raise ValueError("Function call items require a call_id or id")
                if item.name is None:
                    raise ValueError("Function call items require a name")
                if item.text_preamble:
                    # Intermediate text streamed before this tool call must appear as a
                    # text block in the same assistant turn as the tool_use block.
                    if current_role != "assistant":
                        flush()
                        current_role = "assistant"
                    current_content.append({"type": "text", "text": item.text_preamble})
                role, block = "assistant", {
                    "type": "tool_use",
                    "id": call_id,
                    "name": item.name,
                    "input": json.loads(item.arguments or "{}"),
                }
            elif item.type == "function_call_output":
                call_id = item.call_id or item.id
                if call_id is None:
                    raise ValueError("Function call output items require a call_id or id")
                role, block = "user", {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": item.output or "",
                }
            else:
                raise ValueError(f"Unsupported item type: {item.type!r}")

            if current_role != role:
                flush()
                current_role = role
            current_content.append(block)

        flush()
        return "\n\n".join(system_parts) or None, messages

    def to_openai_input(self) -> list[dict]:
        """Build the OpenAI Responses API input list.

        When a function_call item carries a text_preamble, a preceding text message
        dict is injected so the API receives the text output before the function call —
        mirroring the original model turn without storing a standalone assistant item.
        """
        result = []
        for item in self.items:
            if item.type == "function_call" and item.text_preamble:
                result.append({"role": "assistant", "content": item.text_preamble, "type": "message"})
            result.append(item.model_dump(exclude_none=True, exclude={"text_preamble"}))
        return result


class ContextPartitionItem(BaseModel):
    """An all-in-one class that can represent a message, function call, or function call output."""

    arguments: str | None = None
    """Corresponds with:
       - openai.types.responses.ResponseFunctionToolCall
    """

    call_id: str | None = None
    """Corresponds with:
       - openai.types.responses.ResponseFunctionToolCall
       - openai.types.responses.response_input_param.FunctionCallOutput
    """

    content: str | None = None
    """Corresponds with:
       - openai.types.responses.response_input_param.Message
    """

    id: str | None = None
    """Corresponds with:
       - openai.types.responses.ResponseFunctionToolCall
       - openai.types.responses.response_input_param.FunctionCallOutput
    """

    name: str | None = None
    """Corresponds with:
       - openai.types.responses.ResponseFunctionToolCall
    """

    output: str | None = None
    """Corresponds with:
       - openai.types.responses.response_input_param.FunctionCallOutput
    """

    role: str | None = None
    """Corresponds with:
       - openai.types.responses.response_input_param.Message
    """

    text_preamble: str | None = None
    """Intermediate text streamed before a tool_use stop; stored on the first function_call
    item of that round so to_anthropic_messages() can prepend it as a text block in the
    same assistant turn, and to_openai_input() can reconstruct a preceding text message dict.
    Excluded from OpenAI input via to_openai_input(); included in Redis/ES serialization."""

    type: Literal["function_call", "function_call_output", "message"] = "message"
    """Corresponds with:
       - openai.types.responses.ResponseFunctionToolCall
       - openai.types.responses.response_input_param.FunctionCallOutput
       - openai.types.responses.response_input_param.Message
    """

    status: Literal["in_progress", "completed", "incomplete"] | None = None
    """Corresponds with:
       - openai.types.responses.ResponseFunctionToolCall
       - openai.types.responses.response_input_param.FunctionCallOutput
       - openai.types.responses.response_input_param.Message
    """


def _message_hash_payload(items: Iterable[ChatMessage | ContextPartitionItem]) -> list[dict[str, str]]:
    payload = []
    for item in items:
        item_type = getattr(item, "type", "message")
        if item_type == "message" and item.role in {"user", "assistant"}:
            payload.append({"role": item.role or "", "content": item.content or ""})
    return payload


def compute_boundary_hash(items: Iterable[ChatMessage | ContextPartitionItem]) -> str:
    payload = _message_hash_payload(items)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_tail_hash(items: Iterable[ChatMessage | ContextPartitionItem], n: int = 5) -> str:
    tail_user_messages = [
        item.content or ""
        for item in items
        if getattr(item, "type", "message") == "message" and item.role == "user" and item.content
    ][-n:]
    encoded = json.dumps(tail_user_messages, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def conversation_message_items(items: Iterable[ContextPartitionItem]) -> list[ContextPartitionItem]:
    return [
        item
        for item in items
        if item.type == "message" and item.role in {"user", "assistant"}
    ]


class ConversationMatchesPartitionError(Exception):
    """Raised when the incoming conversation exactly matches a partition's raw span."""


class ConversationOutsideRawWindowError(Exception):
    """Raised when the incoming conversation cannot be reconciled with a compacted raw span."""


class FunctionToolCallback(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def system_message_parts(self) -> list[str]: ...

    @property
    def tool_spec(self) -> ToolSpec: ...

    async def call(self, arguments: str, call_id: str) -> ContextPartitionItem | None: ...


class TextEmbeddingPrompt(Enum):
    DOCUMENT = "document"
    QUERY = "query"


class TextEmbeddingRequest(BaseModel):
    batch_size: int = 1
    prompt: TextEmbeddingPrompt
    texts: tuple[str, ...]
    truncate_to: int | None = None


class TextEmbeddingResponse(BaseModel):
    embs: list[list[float]]


class ToolParameters(BaseModel):
    additionalProperties: bool = False
    properties: dict[str, object]
    required: list[str] = Field(default_factory=list)
    type: Literal["object"] = "object"


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: ToolParameters
    strict: bool = True

    def to_anthropic_tool_param(self) -> AnthropicToolParam:
        return AnthropicToolParam(
            description=self.description,
            input_schema=self.parameters.model_dump(),
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
