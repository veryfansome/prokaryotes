"""Shared builders for overlay unit tests."""

from __future__ import annotations

from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    NormalizedMessage,
    TurnExecution,
    TurnItem,
    WorkingFileWindow,
)

BOT_ID = "__bot__"


def conversation(
    *messages: ConversationMessage,
    bot_author_id: str = BOT_ID,
    conversation_uuid: str = "c-1",
    snapshot_uuid: str = "s-1",
    parent_snapshot_uuid: str | None = None,
    ancestor_summaries: list[str] | None = None,
    raw_message_start_index: int = 0,
    working_file_windows: list[WorkingFileWindow] | None = None,
) -> Conversation:
    return Conversation(
        conversation_uuid=conversation_uuid,
        snapshot_uuid=snapshot_uuid,
        parent_snapshot_uuid=parent_snapshot_uuid,
        bot_author_id=bot_author_id,
        ancestor_summaries=ancestor_summaries or [],
        raw_message_start_index=raw_message_start_index,
        messages=list(messages),
        working_file_windows=working_file_windows or [],
    )


def msg(
    source_id: str,
    content: str,
    author_id: str = "u-alice",
    display_name: str | None = None,
    deleted: bool = False,
) -> ConversationMessage:
    return ConversationMessage(
        source_id=source_id,
        author_id=author_id,
        content=content,
        display_name=display_name,
        deleted=deleted,
    )


def bot_msg(source_id: str, content: str) -> ConversationMessage:
    return ConversationMessage(source_id=source_id, author_id=BOT_ID, content=content)


def normalized(
    source_id: str,
    content: str,
    author_id: str = "u-alice",
    display_name: str | None = None,
) -> NormalizedMessage:
    return NormalizedMessage(
        source_id=source_id,
        author_id=author_id,
        content=content,
        display_name=display_name,
    )


def normalized_bot(source_id: str, content: str) -> NormalizedMessage:
    return NormalizedMessage(source_id=source_id, author_id=BOT_ID, content=content)


def turn(
    bot_message_source_id: str,
    *items: TurnItem,
    conversation_uuid: str = "c-1",
    completed: bool = True,
) -> TurnExecution:
    return TurnExecution(
        conversation_uuid=conversation_uuid,
        bot_message_source_id=bot_message_source_id,
        items=list(items),
        completed=completed,
    )


def function_call(call_id: str, name: str, arguments: str = "{}") -> TurnItem:
    return TurnItem(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def function_call_output(call_id: str, output: str) -> TurnItem:
    return TurnItem(type="function_call_output", call_id=call_id, output=output)
