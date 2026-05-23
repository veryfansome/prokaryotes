"""Projection from storage to LLM input.

Single place where role assignment, multi-author prefixing, leading-block emission, working-file-output filtering,
turn-pair ordering, and consecutive same-role text merging happen. Everything else in the system is role-agnostic.
"""

from __future__ import annotations

from prokaryotes.conversation_v1.models import (
    Conversation,
    ConversationMessage,
    ProjectedItem,
    TurnExecution,
    TurnItem,
)

_FILE_TOOL_PERSISTENCE_ANNOTATION = "file_tool.persistence"
_FILE_TOOL_PERSISTENCE_WORKING_FILE = "working_file"


def project_for_llm(
    conversation: Conversation,
    historical_turns: dict[str, TurnExecution] | None = None,
    leading_context_blocks: list[str] | None = None,
    triggering_source_id: str | None = None,
) -> list[ProjectedItem]:
    """Build the LLM input for a turn.

    Used at turn start (the harness's main LLM call) and by the compactor for summarization. The instruction
    message (system/developer) is injected by the caller at position 0 *after* projection — it isn't part of
    conversation storage.

    `historical_turns` maps `bot_message_source_id` → `TurnExecution` for any bot message whose tool-call records
    should be interleaved. Pass `{}` (or omit) when projecting for summarization — tool internals are not part of
    the summarization input.

    `leading_context_blocks` is a list of pre-delimited block strings (e.g. a Slack `<channel_prelude>`) that the
    caller wants positioned between the ancestor-summary block and the first stored message. The parameter is
    `list[str]` (not `list[ProjectedItem]`) so callers cannot inject assistant/tool items at the head and break
    the merge invariant.

    `triggering_source_id` is informational/defensive — the two-pass turn-pair walk (driven by each bot message's
    durable `reply_to_source_id`) is what orders the projection, so the trigger always lands last as the
    highest-source-id unemitted user message. Callers (`_run_turn`) pass it so they can assert the projection
    terminates with the expected user message and detect a tombstoned trigger before the LLM call; the walk itself
    does not consult it.

    Working files project as a leading `<working_files>` block (via `Conversation.working_files_block()`)
    immediately after the ancestor-summary block and any caller-supplied leading blocks, ahead of the conversation
    walk. Historical `function_call_output` items annotated `file_tool.persistence="working_file"` are dropped
    from the projection — their paired `function_call` is dropped in the same pass by `call_id`. The
    durable file context for cross-turn reasoning lives in `working_file_windows`, not in the transcript.
    """
    historical_turns = historical_turns or {}
    leading_context_blocks = leading_context_blocks or []
    distinct_human_authors = {
        msg.author_id
        for msg in conversation.messages
        if not msg.deleted and msg.author_id != conversation.bot_author_id
    }
    needs_prefix = len(distinct_human_authors) > 1

    result: list[ProjectedItem] = []

    summary_block = conversation.ancestor_summary_block()
    if summary_block is not None:
        result.append(ProjectedItem(type="message", role="user", content=summary_block))
    for block in leading_context_blocks:
        result.append(ProjectedItem(type="message", role="user", content=block))
    working_files_block = conversation.working_files_block()
    if working_files_block is not None:
        result.append(ProjectedItem(type="message", role="user", content=working_files_block))

    result.extend(_project_messages(conversation, historical_turns, needs_prefix))

    return _merge_consecutive_same_role(result)


def _project_messages(
    conversation: Conversation,
    historical_turns: dict[str, TurnExecution],
    needs_prefix: bool,
) -> list[ProjectedItem]:
    """Project the stored messages as turn pairs.

    The two-pass walk keeps each bot reply adjacent to the user message it answered, even when storage order
    interleaves them — the Slack same-thread serialization case, where mention B can be stored before the bot
    reply to an earlier mention A. Without it, source-id-order projection collapses `[A, B, botA, botB]` into one
    `user` run and one `assistant` run, permanently losing which reply answered which mention.

    A harness-bot message whose `reply_to_source_id` is unset falls back to source-id-ordered emission: all
    preceding non-emitted non-bot messages are pulled forward before the bot itself emits, producing the
    interleaved user/assistant sequence in source-id order.

    When a Slack-style mention prefix (`<@addressee_id> `) leads the stored bot content, the projection strips it
    so the LLM sees the bare reply body. The addressee is resolved via the bot message's own
    `reply_to_source_id`, so multi-user threads strip per-message; continuation posts of a multi-post reply don't
    carry the prefix and pass through untouched. Without the strip the LLM mimics the prefix into its own
    outputs and the streamer prepends a second one on the wire, producing double-mention replies.
    """
    bot_author_id = conversation.bot_author_id
    sorted_msgs = conversation.sorted_messages()
    by_source_id = {msg.source_id: msg for msg in conversation.messages}
    result: list[ProjectedItem] = []
    emitted: set[str] = set()

    def emit_user(msg: ConversationMessage) -> None:
        content = msg.content
        if needs_prefix and msg.display_name:
            content = f"<{msg.display_name}> {content}"
        result.append(ProjectedItem(type="message", role="user", content=content))
        emitted.add(msg.source_id)

    # Pass 1 — harness-bot messages in source-id order. Each pulls its `reply_to_source_id` user forward (when not
    # already emitted) so the turn pair stays adjacent. A bot with `reply_to_source_id=None` falls back to pulling
    # every preceding non-emitted non-bot message in source-id order.
    for msg in sorted_msgs:
        if msg.deleted or msg.author_id != bot_author_id:
            continue
        reply_to = msg.reply_to_source_id
        addressee_id: str | None = None
        if reply_to is not None:
            target = by_source_id.get(reply_to)
            if target is not None and target.author_id != bot_author_id:
                addressee_id = target.author_id
                if reply_to not in emitted and not target.deleted:
                    emit_user(target)
        else:
            for prev in sorted_msgs:
                if prev.source_id >= msg.source_id:
                    break
                if prev.deleted or prev.source_id in emitted or prev.author_id == bot_author_id:
                    continue
                emit_user(prev)
        turn = historical_turns.get(msg.source_id)
        if turn is not None:
            result.extend(_turn_items_to_projected(_filter_working_file_outputs(turn.items)))
        content = _strip_addressee_mention(msg.content, addressee_id)
        result.append(ProjectedItem(type="message", role="assistant", content=content))
        emitted.add(msg.source_id)

    # Pass 2 — every non-deleted, non-harness-bot message not claimed in Pass 1, in source-id order. Foreign-bot
    # messages emit here as user-role items. The trigger, being the highest-source-id unemitted user message,
    # naturally lands last.
    for msg in sorted_msgs:
        if msg.deleted or msg.source_id in emitted or msg.author_id == bot_author_id:
            continue
        emit_user(msg)

    return result


def _filter_working_file_outputs(items: list[TurnItem]) -> list[TurnItem]:
    """Drop historical `function_call_output`s annotated `file_tool.persistence="working_file"` and the paired
    `function_call`s that produced them. Their durable relevance lives in `working_file_windows`; re-projecting
    them on later turns would duplicate the working-files block content and re-feed transcript-shaped file state
    the new model treats as historical only.
    """
    dropped_call_ids: set[str] = set()
    for item in items:
        if item.type != "function_call_output":
            continue
        ann = item.prokaryotes_annotations or {}
        if ann.get(_FILE_TOOL_PERSISTENCE_ANNOTATION) != _FILE_TOOL_PERSISTENCE_WORKING_FILE:
            continue
        call_id = item.call_id or item.id
        if call_id is not None:
            dropped_call_ids.add(call_id)
    if not dropped_call_ids:
        return items
    kept: list[TurnItem] = []
    for item in items:
        call_id = item.call_id or item.id
        if call_id is not None and call_id in dropped_call_ids:
            continue
        kept.append(item)
    return kept


def _merge_consecutive_same_role(items: list[ProjectedItem]) -> list[ProjectedItem]:
    """Join consecutive same-role `type='message'` items with `\\n\\n`;
    function-call items break the merge run.

    Required by the OpenAI Responses API, which rejects non-alternating user/assistant message sequences. Anthropic
    groups roles separately but benefits from the same normalization.
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


def _strip_addressee_mention(content: str, addressee_id: str | None) -> str:
    """Strip a leading `<@addressee_id> ` mention prefix from a stored Slack bot message.

    The Slack streamer prepends `<@USER> ` to the first post of each reply so the addressee is notified, and
    storage keeps the prefix verbatim so reconcile matches the as-posted text. For LLM projection we want the
    bare body — otherwise the model sees its own prior outputs prefixed with `<@USER>` and starts mimicking the
    pattern, and the streamer prepends a second prefix on the wire, producing double-mention replies.

    Continuation posts of a multi-post reply don't carry the prefix, so the `startswith` guard makes the strip a
    no-op there. DMs and non-Slack harnesses don't add the prefix at all and also pass through untouched.
    """
    if addressee_id is None:
        return content
    prefix = f"<@{addressee_id}> "
    if content.startswith(prefix):
        return content[len(prefix) :]
    return content


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
