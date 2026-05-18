"""Projection from storage to LLM input.

Single place where role assignment, multi-author prefixing, lifted-window
placement, and consecutive same-role text merging happen. Everything else in
the system is role-agnostic.
"""

from __future__ import annotations

from prokaryotes.conversation_v1.models import (
    Conversation,
    ProjectedItem,
    TurnExecution,
    TurnItem,
)


def project_for_llm(
    conversation: Conversation,
    historical_turns: dict[str, TurnExecution] | None = None,
) -> list[ProjectedItem]:
    """Build the LLM input for a turn.

    Used at turn start (the harness's main LLM call) and by the compactor for
    summarization. The instruction message (system/developer) is injected by
    the caller at position 0 *after* projection — it isn't part of conversation
    storage.

    `historical_turns` maps `bot_message_source_id` → `TurnExecution` for any
    bot message whose tool-call records should be interleaved. Pass `{}` (or
    omit) when projecting for summarization — tool internals are not part of
    the summarization input.

    Lifted items (carried across compaction) emit immediately before the anchor
    bot turn's tool round, preserving today's lift placement (after any leading
    user prefix, adjacent to the first relevant file activity).
    """
    historical_turns = historical_turns or {}
    distinct_human_authors = {
        msg.author_id
        for msg in conversation.messages
        if not msg.deleted and msg.author_id != conversation.bot_author_id
    }
    needs_prefix = len(distinct_human_authors) > 1

    result: list[ProjectedItem] = []
    for msg in conversation.sorted_messages():
        if msg.deleted:
            continue
        is_bot = msg.author_id == conversation.bot_author_id

        if is_bot:
            if msg.source_id == conversation.lifted_anchor_source_id:
                result.extend(_turn_items_to_projected(conversation.lifted_turn_items))
            turn = historical_turns.get(msg.source_id)
            if turn is not None:
                result.extend(_turn_items_to_projected(turn.items))
            result.append(ProjectedItem(type="message", role="assistant", content=msg.content))
        else:
            content = msg.content
            if needs_prefix and msg.display_name:
                content = f"<{msg.display_name}> {content}"
            result.append(ProjectedItem(type="message", role="user", content=content))

    return _merge_consecutive_same_role(result)


def current_turn_items(
    conversation: Conversation,
    historical_turns: dict[str, TurnExecution],
    active_turn: TurnExecution,
) -> list[TurnItem]:
    """Flat list of `TurnItem`s visible to tools during the current turn.

    Ordering:
    1. `conversation.lifted_turn_items` (lift carried across compaction)
    2. Each `historical_turns[bot_message.source_id].items`, walking bot
       messages in `source_id` order
    3. `active_turn.items`

    Tools that need to see durable state lifted across compaction must read
    from this view rather than from a single source.
    """
    result: list[TurnItem] = list(conversation.lifted_turn_items)
    for msg in conversation.sorted_messages():
        if msg.deleted or msg.author_id != conversation.bot_author_id:
            continue
        turn = historical_turns.get(msg.source_id)
        if turn is not None:
            result.extend(turn.items)
    result.extend(active_turn.items)
    return result


def _merge_consecutive_same_role(items: list[ProjectedItem]) -> list[ProjectedItem]:
    """Join consecutive `type='message'` items with the same role, joined by `\\n\\n`.
    Function-call items break the merge run.

    This is what makes the OpenAI Responses API happy — it rejects non-
    alternating user/assistant message sequences. Anthropic's role grouping
    handles it separately but benefits from the same normalization.
    """
    merged: list[ProjectedItem] = []
    for item in items:
        if item.type == "message" and merged and merged[-1].type == "message" and merged[-1].role == item.role:
            prev = merged[-1]
            prev_content = prev.content or ""
            new_content = item.content or ""
            if prev_content and new_content:
                joined = f"{prev_content}\n\n{new_content}"
            else:
                joined = prev_content or new_content
            merged[-1] = ProjectedItem(
                type="message",
                role=prev.role,
                content=joined,
            )
        else:
            merged.append(item)
    return merged


def _turn_items_to_projected(items: list[TurnItem]) -> list[ProjectedItem]:
    projected: list[ProjectedItem] = []
    for item in items:
        if item.type == "function_call":
            projected.append(
                ProjectedItem(
                    type="function_call",
                    arguments=item.arguments,
                    call_id=item.call_id or item.id,
                    name=item.name,
                )
            )
        elif item.type == "function_call_output":
            projected.append(
                ProjectedItem(
                    type="function_call_output",
                    call_id=item.call_id or item.id,
                    output=item.output,
                )
            )
    return projected
