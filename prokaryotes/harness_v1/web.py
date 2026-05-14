import logging
from pathlib import Path

from fastapi import (
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import StreamingResponse
from starsessions import load_session

from prokaryotes.api_v1.models import (
    ChatConversation,
    ContextPartition,
    ContextPartitionItem,
    FunctionToolCallback,
)
from prokaryotes.harness_v1 import build_llm_client
from prokaryotes.tools_v1.file_tool import FileTool, reconcile_tracked_files
from prokaryotes.tools_v1.file_tool.live_windows import strip_live_window_bodies
from prokaryotes.tools_v1.shell_command import ShellCommandTool
from prokaryotes.tools_v1.think import ThinkTool
from prokaryotes.utils_v1 import system_message_utils
from prokaryotes.utils_v1.llm_utils import (
    ANTHROPIC_DEFAULT_MODEL,
    COMPACTION_SUMMARY_MAX_TOKENS,
    COMPACTION_TOKEN_THRESHOLD_PCT,
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
    OPENAI_DEFAULT_MODEL,
)
from prokaryotes.web_v1 import WebBase

logger = logging.getLogger(__name__)

_SUMMARIZATION_PROMPT = (
    "Summarize the conversation above as a structured briefing for future continuation. "
    "Preserve key decisions, facts, code produced, and tool call outcomes. "
    "Use markdown sections. Be concise."
)


class WebHarness(WebBase):
    """Provider-agnostic chat-over-HTTP harness.

    `impl` selects the LLM client and instruction-message role:
    - `"anthropic"` → `AnthropicClient`, `instruction_role="system"`. Ancestor summaries
      are auto-appended to the system prompt by `ContextPartition.to_anthropic_messages()`.
    - `"openai"` → `OpenAIClient`, `instruction_role="developer"`. Ancestor summaries must
      be appended to the developer message manually, since `to_openai_input()` does not
      inject them.

    The `_summarize_and_compact()` body differs by provider SDK; everything else
    (auth, partition sync, tool dispatch, streaming, compaction trigger) is shared.
    """

    def __init__(self, impl: str, static_dir: str):
        super().__init__(static_dir)
        self.impl = impl
        self.llm_client, self.instruction_role = build_llm_client(impl)
        self.default_model = ANTHROPIC_DEFAULT_MODEL if impl == "anthropic" else OPENAI_DEFAULT_MODEL

    def _build_instruction_parts(
        self,
        *,
        context_partition: ContextPartition,
        latitude: float | None,
        longitude: float | None,
        session: dict,
        time_zone: str | None,
        tool_callbacks: dict[str, FunctionToolCallback],
    ) -> list[str]:
        parts: list[str] = []
        parts.extend(
            system_message_utils.get_core_instruction_parts(summaries=bool(context_partition.ancestor_summaries))
        )
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
        # OpenAI's to_openai_input() does not auto-inject ancestor summaries, so we append
        # them here. Anthropic's to_anthropic_messages() handles it inside the stream loop.
        if self.impl == "openai":
            summary_block = context_partition.ancestor_summary_block()
            if summary_block:
                parts.append("")
                parts.append(summary_block)
        return parts

    async def _summarize_and_compact(
        self,
        *,
        model: str,
        snapshot: ContextPartition,
    ) -> str:
        # Strip live-window file bodies from the summarization input so current file
        # contents do not fossilize into ancestor summaries.
        summary_input = strip_live_window_bodies(snapshot)
        if self.impl == "anthropic":
            return await self._summarize_anthropic(model=model, summary_input=summary_input)
        return await self._summarize_openai(model=model, summary_input=summary_input)

    async def _summarize_anthropic(
        self,
        *,
        model: str,
        summary_input: ContextPartition,
    ) -> str:
        system_str, messages = summary_input.to_anthropic_messages()
        summarization_messages = messages + [
            {
                "role": "user",
                "content": _SUMMARIZATION_PROMPT,
            }
        ]
        create_params: dict = {
            "model": model,
            "max_tokens": COMPACTION_SUMMARY_MAX_TOKENS,
            "messages": summarization_messages,
        }
        if system_str:
            create_params["system"] = system_str
        response = await self.llm_client.async_anthropic.messages.create(**create_params)
        return response.content[0].text

    async def _summarize_openai(
        self,
        *,
        model: str,
        summary_input: ContextPartition,
    ) -> str:
        items: list[dict] = []
        summary_block = summary_input.ancestor_summary_block()
        if summary_block:
            items.append(
                {
                    "role": "developer",
                    "content": summary_block,
                    "type": "message",
                }
            )
        items.extend(summary_input.to_openai_input())
        items.append(
            {
                "role": "user",
                "content": _SUMMARIZATION_PROMPT,
                "type": "message",
            }
        )
        response = await self.llm_client.async_openai.responses.create(
            model=model,
            max_output_tokens=COMPACTION_SUMMARY_MAX_TOKENS,
            input=items,
            stream=False,
        )
        return response.output_text

    def init(self):
        super().init()
        self.llm_client.init_client()
        self.app.add_api_route("/chat", self.post_chat, methods=["POST"])

    async def on_stop(self):
        await super().on_stop()
        await self.llm_client.close()

    async def post_chat(
        self,
        conversation: ChatConversation,
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
        if len(conversation.messages) == 0:
            raise HTTPException(status_code=400, detail="At least one message is required")
        model = model or self.default_model
        context_partition = await self.sync_context_partition(conversation)
        workspace_root = Path.cwd()
        await reconcile_tracked_files(context_partition, workspace_root=workspace_root)

        file_tool = FileTool(context_partition, workspace_root=workspace_root)
        shell_command_tool = ShellCommandTool()
        think_tool = ThinkTool(self.llm_client, model)
        tool_callbacks: dict[str, FunctionToolCallback] = {
            file_tool.name: file_tool,
            shell_command_tool.name: shell_command_tool,
            think_tool.name: think_tool,
        }

        instruction_parts = self._build_instruction_parts(
            context_partition=context_partition,
            latitude=latitude,
            longitude=longitude,
            session=session,
            time_zone=time_zone,
            tool_callbacks=tool_callbacks,
        )
        instruction_message = ContextPartitionItem(role=self.instruction_role, content="\n".join(instruction_parts))
        context_partition.items.insert(0, instruction_message)

        # List so on_usage can mutate the flag in place; a plain bool passed to
        # stream_and_finalize would be an immutable copy unreachable by nonlocal.
        pending_compaction = [False]

        def on_usage(input_tokens: int, output_tokens: int) -> None:
            context_window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
            context_pct = int(input_tokens / context_window * 100)
            if context_pct >= COMPACTION_TOKEN_THRESHOLD_PCT:
                pending_compaction[0] = True

        async def compact(snapshot: ContextPartition) -> str:
            return await self._summarize_and_compact(snapshot=snapshot, model=model)

        return StreamingResponse(
            self.stream_and_finalize(
                context_partition=context_partition,
                conversation_uuid=conversation.conversation_uuid,
                response_generator=self.llm_client.stream_turn(
                    context_partition=context_partition,
                    model=model,
                    on_usage=on_usage,
                    reasoning_effort=reasoning_effort,
                    stream_ndjson=True,
                    tool_callbacks=tool_callbacks,
                ),
                pending_compaction=pending_compaction,
                compact_fn=compact,
            ),
            media_type="text/event-stream",
        )
