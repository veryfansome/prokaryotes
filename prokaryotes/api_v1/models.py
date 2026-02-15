from __future__ import annotations

import json
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
    items: list[ContextPartitionItem]

    def append(self, item: ContextPartitionItem):
        self.items.append(item)

    def extend(self, items: Iterable[ContextPartitionItem]):
        self.items.extend(items)

    @classmethod
    def find_context_divergence(
            cls,
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

    def pop_system_message(self):
        if self.items and self.items[0].role in {"developer", "system"}:
            self.items.pop(0)

    def sync_from_conversation(self, conversation: ChatConversation):
        partition_messages = []
        partition_message_indexes = []
        for idx, item in enumerate(self.items):
            if item.type == "message":
                partition_messages.append(item)
                partition_message_indexes.append(idx)
        divergence_idx, is_mismatch, is_longer = self.find_context_divergence(
            partition_messages,
            [message.to_context_partition_item() for message in conversation.messages],
        )
        if divergence_idx is None:
            raise Exception("Conversation does not alter partition state")
        elif is_longer:
            divergence_idx = partition_message_indexes[divergence_idx]
            if is_mismatch:
                self.items = self.items[:divergence_idx]
                self.items.append(conversation.messages[-1].to_context_partition_item())
            else:
                self.items = self.items[:divergence_idx]
        else:
            self.items = self.items[:partition_message_indexes[-1] + 1]  # Truncate to last good
            self.items.extend(message.to_context_partition_item() for message in conversation.messages[divergence_idx:])

    def to_anthropic_messages(self):
        """Convert ContextPartition to Anthropic (system, messages) format."""
        system_parts: list[str] = []
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


class ContextPartitionItem(BaseModel):
    """An all-in-one class that can represent a message, function call, or function call output."""

    arguments: str | None = None
    """Corresponds with:
       - openai.types.responses.ResponseFunctionToolCall
    """

    call_id: str | None = None
    """Corresponds with:"""
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
