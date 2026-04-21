import logging

from fastapi import (
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import StreamingResponse
from starsessions import load_session

from prokaryotes.anthropic_v1 import LLMClient
from prokaryotes.api_v1.models import (
    ChatConversation,
    ContextPartition,
    ContextPartitionItem,
    FunctionToolCallback,
)
from prokaryotes.tools_v1.shell_command import ShellCommandTool
from prokaryotes.tools_v1.think import ThinkTool
from prokaryotes.utils_v1 import system_message_utils
from prokaryotes.utils_v1.llm_utils import (
    ANTHROPIC_DEFAULT_MODEL,
    COMPACTION_SUMMARY_MAX_TOKENS,
    COMPACTION_TOKEN_THRESHOLD_PCT,
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
)
from prokaryotes.web_v1 import WebBase

logger = logging.getLogger(__name__)


class WebHarness(WebBase):
    def __init__(self, static_dir: str):
        super().__init__(static_dir)
        self.llm_client = LLMClient()

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
            request: Request,
            conversation: ChatConversation,
            latitude: float = Query(None),
            longitude: float = Query(None),
            model: str = Query(ANTHROPIC_DEFAULT_MODEL),
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
        think_tool = ThinkTool()
        tool_callbacks: dict[str, FunctionToolCallback] = {
            shell_command_tool.name: shell_command_tool,
            think_tool.name: think_tool,
        }

        system_message_parts = []
        system_message_parts.append("# Tool usage")
        for name in sorted(tool_callbacks.keys()):
            system_message_parts.extend(tool_callbacks[name].system_message_parts)
        system_message_parts.extend(system_message_utils.get_web_harness_runtime_context_parts(time_zone))
        system_message_parts.extend(system_message_utils.get_personality_parts())
        system_message_parts.append("# User context")
        system_message_parts.append(f"- User's name is {session['full_name']}")
        if latitude and longitude:
            system_message_parts.append(f"- User's location is at  *lat: {latitude:.4f}, long: {longitude:.4f}*")

        system_message = ContextPartitionItem(
            role="system", content="\n".join(system_message_parts)
        )
        context_partition.items.insert(0, system_message)

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
                response_generator=self.llm_client.stream_response(
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

    async def _summarize_and_compact(
            self,
            snapshot: ContextPartition,
            model: str,
    ) -> str:
        system_str, messages = snapshot.to_anthropic_messages()
        summarization_messages = messages + [{
            "role": "user",
            "content": (
                "Summarize the conversation above as a structured briefing for future continuation. "
                "Preserve key decisions, facts, code produced, and tool call outcomes. "
                "Use markdown sections. Be concise."
            ),
        }]
        create_params: dict = {
            "model": model,
            "max_tokens": COMPACTION_SUMMARY_MAX_TOKENS,
            "messages": summarization_messages,
        }
        if system_str:
            create_params["system"] = system_str
        response = await self.llm_client.async_anthropic.messages.create(**create_params)
        return response.content[0].text
