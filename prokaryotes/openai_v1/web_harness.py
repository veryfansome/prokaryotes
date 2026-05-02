import logging

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
from prokaryotes.openai_v1 import OpenAIClient
from prokaryotes.tools_v1.shell_command import ShellCommandTool
from prokaryotes.tools_v1.think import ThinkTool
from prokaryotes.utils_v1 import system_message_utils
from prokaryotes.utils_v1.llm_utils import (
    COMPACTION_SUMMARY_MAX_TOKENS,
    COMPACTION_TOKEN_THRESHOLD_PCT,
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
    OPENAI_DEFAULT_MODEL,
)
from prokaryotes.web_v1 import WebBase

logger = logging.getLogger(__name__)


class WebHarness(WebBase):
    def __init__(self, static_dir: str):
        super().__init__(static_dir)
        self.llm_client = OpenAIClient()

    async def _summarize_and_compact(
            self,
            model: str,
            snapshot: ContextPartition,
    ) -> str:
        items = []
        summary_block = snapshot.ancestor_summary_block()
        if summary_block:
            items.append({
                "role": "developer",
                "content": summary_block,
                "type": "message",
            })
        items.extend(snapshot.to_openai_input())
        items.append({
            "role": "user",
            "content": (
                "Summarize the conversation above as a structured briefing for future continuation. "
                "Preserve key decisions, facts, code produced, and tool call outcomes. "
                "Use markdown sections. Be concise."
            ),
            "type": "message",
        })
        response = await self.llm_client.async_openai.responses.create(
            model=model,
            max_output_tokens=COMPACTION_SUMMARY_MAX_TOKENS,
            input=items,
            stream=False,
        )
        return response.output_text

    def init(self):
        """Synchronous setup steps"""
        super().init()
        self.llm_client.init_client()
        self.app.add_api_route("/chat", self.post_chat, methods=["POST"])

    async def on_stop(self):
        """Asynchronous teardown steps"""
        await super().on_stop()
        await self.llm_client.close()

    async def post_chat(
            self,
            conversation: ChatConversation,
            request: Request,
            latitude: float = Query(None),
            longitude: float = Query(None),
            model: str = Query(OPENAI_DEFAULT_MODEL),
            reasoning_effort: str = Query(None),
            time_zone: str = Query(None),
    ):
        """Chat completion."""
        await load_session(request)
        session = request.session
        if not session:
            raise HTTPException(status_code=400, detail="Session expired")
        if len(conversation.messages) == 0:
            raise HTTPException(status_code=400, detail="At least one message is required")
        context_partition = await self.sync_context_partition(conversation)

        shell_command_tool = ShellCommandTool()
        think_tool = ThinkTool(self.llm_client)
        tool_callbacks: dict[str, FunctionToolCallback] = {
            shell_command_tool.name: shell_command_tool,
            think_tool.name: think_tool,
        }

        developer_message_parts = []
        developer_message_parts.extend(system_message_utils.get_core_instruction_parts(
            summaries=bool(context_partition.ancestor_summaries)
        ))
        developer_message_parts.append("")
        developer_message_parts.extend(system_message_utils.get_personality_parts())
        developer_message_parts.append("")
        developer_message_parts.append("# Tool usage")
        for name in sorted(tool_callbacks.keys()):
            developer_message_parts.extend(tool_callbacks[name].system_message_parts)
        developer_message_parts.append("")
        developer_message_parts.extend(system_message_utils.get_web_harness_runtime_context_parts(time_zone))
        developer_message_parts.append("")
        developer_message_parts.append("# User context")
        developer_message_parts.append(f"- User's name is {session['full_name']}")
        if latitude and longitude:
            developer_message_parts.append(f"- User's location is at  *lat: {latitude:.4f}, long: {longitude:.4f}*")
        summary_block = context_partition.ancestor_summary_block()
        if summary_block:
            developer_message_parts.append("")
            developer_message_parts.append(summary_block)

        developer_message = ContextPartitionItem(
            role="developer", content="\n".join(developer_message_parts)
        )
        context_partition.items.insert(0, developer_message)

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
