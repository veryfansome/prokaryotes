"""WebHarness — chat-over-HTTP harness with first-class working-file state.

`FileTool` and `ThinkTool` are wired to `conversation.working_file_windows` via callable providers. At turn
start, `reconcile_working_files` refreshes every live window against on-disk state before any FileTool call.
Compaction summarization and triggering are inherited from `HarnessBase` (`_build_compact_fn` / `_maybe_compact`).
"""

from __future__ import annotations

import logging
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
from prokaryotes.conversation_v1.project import project_for_llm
from prokaryotes.conversation_v1.source_id import bump_source_id, format_source_id_now
from prokaryotes.harness_v1 import build_llm_client
from prokaryotes.harness_v1.base import _StreamFinalizationContext
from prokaryotes.tools_v1.file_tool import FileTool
from prokaryotes.tools_v1.file_tool.live_windows import reconcile_working_files
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


class WebHarness(WebBase):
    """Provider-agnostic chat-over-HTTP harness.

    `impl` selects the LLM client and the instruction-message role:
    - `"anthropic"` → `AnthropicClient`, `instruction_role="system"`.
    - `"openai"` → `OpenAIClient`, `instruction_role="developer"`.
    """

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
        if sync_result.resync:
            async for chunk in self.stream_and_finalize(
                sync_result=sync_result,
                bot_message_source_id_provider=_assign_bot_source_id,
            ):
                yield chunk
            return

        conversation = sync_result.conversation
        bot_source_ids = [
            m.source_id for m in conversation.messages if not m.deleted and m.author_id == conversation.bot_author_id
        ]
        historical_turns = await self.search_client.get_turn_executions(conversation.conversation_uuid, bot_source_ids)

        active_turn = TurnExecution(
            conversation_uuid=conversation.conversation_uuid,
            bot_message_source_id="__pending__",
            items=[],
        )

        workspace_root = Path.cwd()
        # Turn-start reconcile: every live window's content reflects current on-disk revision before any FileTool
        # call runs in this turn. The REDUNDANT_READ coverage check inside FileTool then runs against post-reconcile
        # state.
        await reconcile_working_files(
            conversation.working_file_windows,
            workspace_root=workspace_root,
            max_file_bytes=FileTool.max_file_bytes,
            max_lines=FileTool.max_lines,
        )

        def working_file_provider():
            return conversation.working_file_windows

        file_tool = FileTool(working_file_provider, workspace_root=workspace_root)
        shell_command_tool = ShellCommandTool()
        think_tool = ThinkTool(
            self.llm_client,
            model,
            working_file_provider=working_file_provider,
            workspace_root=workspace_root,
        )
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

        def response_generator_factory(ctx: _StreamFinalizationContext):
            def on_committed_turn_item(item):
                ctx.committed_turn_items.append(item)
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
        return parts


def _assign_bot_source_id(conversation: Conversation) -> str:
    last_id = max(
        (m.source_id for m in conversation.messages if not m.deleted),
        default=None,
    )
    candidate = format_source_id_now()
    if last_id is not None and candidate <= last_id:
        candidate = bump_source_id(last_id)
    return candidate
