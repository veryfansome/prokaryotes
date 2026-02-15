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
    ContextPartitionItem,
    FunctionToolCallback,
)
from prokaryotes.openai_v1 import LLMClient
from prokaryotes.tools_v1.shell_command import ShellCommandTool
from prokaryotes.tools_v1.think import ThinkTool
from prokaryotes.utils_v1 import system_message_utils
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
            model: str = Query("gpt-5.4"),
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

        developer_message_parts = []
        developer_message_parts.append("# Tool usage")
        for name in sorted(tool_callbacks.keys()):
            developer_message_parts.extend(tool_callbacks[name].system_message_parts)
        developer_message_parts.extend(system_message_utils.get_web_harness_runtime_context_parts(time_zone))
        developer_message_parts.extend(system_message_utils.get_personality_parts())
        developer_message_parts.append("# User context")
        developer_message_parts.append(f"- User's name is {session['full_name']}")
        if latitude and longitude:
            developer_message_parts.append(f"- User's location is at  *lat: {latitude:.4f}, long: {longitude:.4f}*")

        developer_message = ContextPartitionItem(
            role="developer", content="\n".join(developer_message_parts)
        )
        context_partition.items.insert(0, developer_message)
        logger.info(f"Web-harness developer message:\n{developer_message.content}")

        def on_usage(input_tokens: int, output_tokens: int) -> None:
            # TODO: trigger compaction when input_tokens exceeds threshold
            pass

        return StreamingResponse(
            self.stream_and_finalize(
                context_partition=context_partition,
                response_generator=self.llm_client.stream_response(
                    context_partition=context_partition,
                    model=model,
                    on_usage=on_usage,
                    reasoning_effort=reasoning_effort,
                    stream_ndjson=True,
                    tool_callbacks=tool_callbacks,
                )
            ),
            media_type="text/event-stream",
        )
