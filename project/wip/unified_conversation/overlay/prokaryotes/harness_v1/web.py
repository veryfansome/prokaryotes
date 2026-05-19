"""WebHarness — chat-over-HTTP harness wired to the unified conversation model.

The flow per POST /chat:
1. `sync_conversation` returns a `SyncResult` carrying the snapshot to use (a
   fresh branch on divergence; the existing one otherwise), any server-assigned
   source_ids, and on stream-loss recovery, `unacknowledged_bot_messages`.
2. If resync: `stream_and_finalize` emits the handshake and closes — no LLM call.
3. Otherwise: load `TurnExecution`s for the raw window's bot messages, refresh
   tracked files against the unified view, build tools/instruction/projection,
   then stream. Tool items commit via `on_committed_turn_item`; final assistant
   text via `on_final_assistant_message`. Bot message gets a fresh server-assigned
   `source_id` and is committed last, before `bot_message` is emitted.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starsessions import load_session

from prokaryotes.api_v1.models import FunctionToolCallback, IncomingConversation
from prokaryotes.context_v1.conversation_sync import AssistantMessageGuardrailError
from prokaryotes.conversation_v1.models import (
    Conversation,
    TurnExecution,
)
from prokaryotes.conversation_v1.project import current_turn_items, project_for_llm
from prokaryotes.harness_v1 import build_llm_client
from prokaryotes.harness_v1.base import _StreamFinalizationContext
from prokaryotes.tools_v1.file_tool import FileTool, reconcile_tracked_files
from prokaryotes.tools_v1.file_tool.live_windows import strip_live_window_bodies
from prokaryotes.tools_v1.shell_command import ShellCommandTool
from prokaryotes.tools_v1.think import ThinkTool
from prokaryotes.utils_v1 import system_message_utils
from prokaryotes.utils_v1.llm_utils import (
    ANTHROPIC_DEFAULT_MODEL,
    COMPACTION_TOKEN_THRESHOLD_PCT,
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
    OPENAI_DEFAULT_MODEL,
)
from prokaryotes.web_v1 import WebBase

logger = logging.getLogger(__name__)

_BOT_AUTHOR_ID = "__bot__"
_SUMMARIZATION_PROMPT = (
    "Summarize the conversation above as a structured briefing for future continuation."
    " Preserve key decisions, facts, code produced, and tool call outcomes."
    " Use markdown sections. Be concise."
)


class WebHarness(WebBase):
    def __init__(self, impl: str, static_dir: str):
        super().__init__(static_dir)
        self.impl = impl
        self.llm_client, self.instruction_role = build_llm_client(impl)
        self.default_model = ANTHROPIC_DEFAULT_MODEL if impl == "anthropic" else OPENAI_DEFAULT_MODEL

    def init(self):
        super().init()
        self.llm_client.init_client()
        self.app.add_api_route("/chat", self.post_chat, methods=["POST"])

    async def on_stop(self):
        await super().on_stop()
        await self.llm_client.close()

    async def post_chat(
        self,
        incoming: IncomingConversation,
        request: Request,
        latitude: float = Query(None),
        longitude: float = Query(None),
        model: str | None = Query(None),
        reasoning_effort: str = Query(None),
        time_zone: str = Query(None),
    ):
        await load_session(request)
        session = request.session
        if not session:
            raise HTTPException(status_code=400, detail="Session expired")
        if len(incoming.messages) == 0:
            raise HTTPException(status_code=400, detail="At least one message is required")
        model = model or self.default_model

        # DAG-scoped guardrail: reject POSTs that fabricate or rewrite bot
        # messages. Runs before `sync_conversation` so `_partially_normalize`
        # can't quietly accept hostile `role="assistant"` entries by mapping
        # them to `bot_author_id`.
        try:
            await self.validate_assistant_messages(incoming.conversation_uuid, incoming.messages)
        except AssistantMessageGuardrailError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        sync_result = await self.sync_conversation(
            conversation_uuid=incoming.conversation_uuid,
            snapshot_uuid=incoming.snapshot_uuid,
            bot_author_id=_BOT_AUTHOR_ID,
            incoming=incoming.messages,
            session_user_id=str(session["user_id"]),
            session_display_name=session.get("full_name"),
        )

        return StreamingResponse(
            self._dispatch_turn(
                sync_result=sync_result,
                session=session,
                latitude=latitude,
                longitude=longitude,
                model=model,
                reasoning_effort=reasoning_effort,
                time_zone=time_zone,
            ),
            media_type="text/event-stream",
        )

    async def _dispatch_turn(
        self,
        *,
        sync_result,
        session: dict,
        latitude: float | None,
        longitude: float | None,
        model: str,
        reasoning_effort: str | None,
        time_zone: str | None,
    ):
        """Wire `sync_result` through `stream_and_finalize`. On resync, the
        latter emits just the handshake and returns — no further setup needed."""
        if sync_result.resync:
            async for chunk in self.stream_and_finalize(
                sync_result=sync_result,
                bot_message_source_id_provider=_assign_bot_source_id,
                response_generator_factory=lambda _ctx: _empty_stream(),
            ):
                yield chunk
            return

        conversation = sync_result.conversation
        # Load TurnExecutions for the raw window's bot messages so projection can
        # interleave tool items, and so the file-tool view sees them.
        bot_source_ids = [
            m.source_id for m in conversation.messages if not m.deleted and m.author_id == conversation.bot_author_id
        ]
        historical_turns = await self.search_client.get_turn_executions(conversation.conversation_uuid, bot_source_ids)

        # Active turn (mutable — populated as tool callbacks commit).
        active_turn = TurnExecution(
            conversation_uuid=conversation.conversation_uuid,
            # The bot's source_id is assigned at finalize time; we use a placeholder
            # here that gets ignored when the active_turn is later persisted under
            # the real bot source_id.
            bot_message_source_id="__pending__",
            items=[],
        )

        # Build the FileTool against the unified view; refresh tracked files first.
        def view_provider():
            return current_turn_items(conversation, historical_turns, active_turn)

        workspace_root = Path.cwd()
        await reconcile_tracked_files(view_provider(), workspace_root=workspace_root)

        file_tool = FileTool(view_provider, workspace_root=workspace_root)
        shell_command_tool = ShellCommandTool()
        think_tool = ThinkTool(self.llm_client, model)
        tool_callbacks: dict[str, FunctionToolCallback] = {
            file_tool.name: file_tool,
            shell_command_tool.name: shell_command_tool,
            think_tool.name: think_tool,
        }

        instruction = "\n".join(
            self._build_instruction_parts(
                conversation=conversation,
                latitude=latitude,
                longitude=longitude,
                session=session,
                time_zone=time_zone,
                tool_callbacks=tool_callbacks,
            )
        )
        projected_items = project_for_llm(conversation, historical_turns=historical_turns)

        pending_compaction = [False]

        def on_usage(input_tokens: int, output_tokens: int) -> None:
            context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            context_pct = int(input_tokens / context_window * 100)
            if context_pct >= COMPACTION_TOKEN_THRESHOLD_PCT:
                pending_compaction[0] = True

        async def compact(snapshot: Conversation, prep) -> str:
            return await self._summarize_and_compact(model=model, snapshot=snapshot, prep=prep)

        def response_generator_factory(ctx: _StreamFinalizationContext):
            def on_committed_turn_item(item):
                ctx.committed_turn_items.append(item)
                # Keep active_turn in sync so the file-tool view sees in-flight items.
                active_turn.items.append(item)

            def on_final_assistant_message(text):
                ctx.final_assistant_text.append(text)

            return self.llm_client.stream_turn(
                items=projected_items,
                instruction=instruction,
                model=model,
                on_committed_turn_item=on_committed_turn_item,
                on_final_assistant_message=on_final_assistant_message,
                on_usage=on_usage,
                reasoning_effort=reasoning_effort,
                stream_ndjson=True,
                tool_callbacks=tool_callbacks,
            )

        async for chunk in self.stream_and_finalize(
            sync_result=sync_result,
            bot_message_source_id_provider=_assign_bot_source_id,
            response_generator_factory=response_generator_factory,
            compact_fn=compact,
            pending_compaction=pending_compaction,
        ):
            yield chunk

    def _build_instruction_parts(
        self,
        *,
        conversation: Conversation,
        latitude: float | None,
        longitude: float | None,
        session: dict,
        time_zone: str | None,
        tool_callbacks: dict[str, FunctionToolCallback],
    ) -> list[str]:
        parts: list[str] = []
        parts.extend(system_message_utils.get_core_instruction_parts(summaries=bool(conversation.ancestor_summaries)))
        parts.append("")
        parts.extend(system_message_utils.get_runtime_context_parts(time_zone))
        parts.append("")
        parts.append("# Tool usage")
        for name in sorted(tool_callbacks.keys()):
            parts.append("")
            parts.extend(tool_callbacks[name].system_message_parts)
        parts.append("")
        parts.extend(system_message_utils.get_personality_parts())
        parts.append("")
        parts.append("# User context")
        parts.append("")
        parts.append(f"- User's name is {session['full_name']}")
        if latitude and longitude:
            parts.append(f"- User's location is at  *lat: {latitude:.4f}, long: {longitude:.4f}*")
        # The instruction message is the place where ancestor summaries enter the LLM
        # call — projection's same-role merge leaves them in the conversation messages
        # for Anthropic's `system` placement and OpenAI's `developer` message uniformly.
        summary_block = conversation.ancestor_summary_block()
        if summary_block:
            parts.append("")
            parts.append(summary_block)
        return parts

    async def _summarize_and_compact(
        self,
        *,
        model: str,
        snapshot: Conversation,
        prep,
    ) -> str:
        """Build the summarization input, stripping live-window bodies from
        pre-tail TurnExecutions so current file contents don't fossilize into
        ancestor summaries. Then make a non-streaming LLM call for the summary.
        """
        # Strip live windows from pre-tail TurnExecutions used by the summarization input.
        stripped_pre_tail_turns: dict[str, TurnExecution] = {}
        for bot_id, turn in prep.pre_tail_turns.items():
            stripped_pre_tail_turns[bot_id] = TurnExecution(
                conversation_uuid=turn.conversation_uuid,
                bot_message_source_id=turn.bot_message_source_id,
                items=strip_live_window_bodies(turn.items),
                completed=turn.completed,
            )

        # Build a partial Conversation containing only the pre-tail messages — that's
        # what gets summarized.
        pre_tail_conv = snapshot.model_copy(
            update={
                "messages": prep.pre_tail_messages,
                "lifted_turn_items": [],  # lift state is for the child; not part of summary
                "lifted_anchor_source_id": None,
            }
        )
        items_for_summary = project_for_llm(pre_tail_conv, historical_turns=stripped_pre_tail_turns)
        # Append the explicit summarization request as a final user message.
        from prokaryotes.conversation_v1.models import ProjectedItem

        items_for_summary.append(ProjectedItem(type="message", role="user", content=_SUMMARIZATION_PROMPT))
        return await self.llm_client.complete(
            items=items_for_summary,
            instruction=None,
            model=model,
            reasoning_effort=None,
        )


def _assign_bot_source_id(conversation: Conversation) -> str:
    """Assign a fresh monotonic `source_id` for the bot's final message.

    Same `seconds.microseconds` format as the syncer's user-message assignment,
    bumped to be strictly greater than the latest source_id in the conversation.
    """
    last_id = max(
        (m.source_id for m in conversation.messages if not m.deleted),
        default=None,
    )
    candidate = _format_now()
    if last_id is not None and candidate <= last_id:
        candidate = _bump(last_id)
    return candidate


def _bump(source_id: str) -> str:
    seconds_str, _, micros_str = source_id.partition(".")
    try:
        seconds = int(seconds_str)
        micros = int(micros_str or "0")
    except ValueError:
        return _format_now()
    micros += 1
    if micros >= 1_000_000:
        seconds += 1
        micros = 0
    return f"{seconds}.{micros:06d}"


def _format_now() -> str:
    ts = time.time()
    seconds = int(ts)
    micros = int((ts - seconds) * 1_000_000)
    return f"{seconds}.{micros:06d}"


async def _empty_stream():
    """No-op generator used on the resync path. The handshake has already been
    emitted by `stream_and_finalize`; nothing else flows."""
    if False:  # pragma: no cover
        yield ""
    return
