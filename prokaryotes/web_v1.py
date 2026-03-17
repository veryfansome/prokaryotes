import asyncio
import json
import logging
import uuid
from fastapi import (
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_input_param import FunctionCallOutput
from starsessions import load_session
from typing import (
    Any,
    AsyncGenerator,
)

from prokaryotes.graph_v1 import GraphClient
from prokaryotes.llm_v1 import (
    FunctionToolCallback,
    get_llm_client,
)
from prokaryotes.models_v1 import (
    ChatMessage,
    PromptPayload,
    PromptContext,
    ToolCallDoc,
)
from prokaryotes.observer_v1.context_observer import ContextFilteringObserver
from prokaryotes.observer_v1.topic_observer import TopicClassifyingObserver
from prokaryotes.observer_v1.user_fact_observer import UserFactsSavingObserver
from prokaryotes.search_v1 import (
    PersonContext,
    SearchClient,
)
from prokaryotes.tools_v1.builtins import web_search_tool_param
from prokaryotes.tools_v1.read_file import ReadFileCallback
from prokaryotes.tools_v1.scan_directory import ScanDirectoryCallback
from prokaryotes.tools_v1.search_email import SearchEmailCallback
from prokaryotes.utils_v1.context_utils import developer_message_parts
from prokaryotes.utils_v1.text_utils import normalize_text_for_search_and_embed
from prokaryotes.web_base_v1 import WebBase

logger = logging.getLogger(__name__)

class ProkaryoteV1(WebBase):
    def __init__(self, static_dir: str):
        super().__init__(static_dir)
        self.graph_client = GraphClient()
        self.llm_client = get_llm_client()
        self.search_client = SearchClient()

        tools: list[FunctionToolCallback] = [
            ReadFileCallback(self.llm_client, self.search_client),
            ScanDirectoryCallback(self.llm_client, self.search_client),
            SearchEmailCallback(self.search_client),
        ]
        self.tools_callbacks = {t.tool_param["name"]: t for t in tools}
        self.tools_params = [t.tool_param for t in tools]
        self.tools_params.append(web_search_tool_param)

    @classmethod
    def developer_message(
            cls,
            prompt_context: PromptContext,
            user_context: PersonContext,
            tool_calls: list[ToolCallDoc],
    ):
        message_parts = developer_message_parts(prompt_context, user_context)

        if tool_calls:
            # TODO: Maybe there should be some heuristic here about what to prioritize
            message_parts.append("---")
            message_parts.append("## Recalled tool outputs")
            for tool_call_doc in tool_calls:
                updated_at = tool_call_doc.updated_at.astimezone(prompt_context.time_zone).strftime('%Y-%m-%d %H:%M')
                message_parts.append(
                    f"\n### {', '.join(tool_call_doc.labels)}\n"
                    f"Updated at: {updated_at}\n"
                    f"<pre><code>{tool_call_doc.output}</code></pre>"
                )

        message_parts.append("---")
        message_parts.append("## Assistant instructions")
        # TODO: Maybe this should be guided by user preferences
        message_parts.append("- Use short messages. One sentences is best, unless the user explicitly requests more.")
        message_parts.append("- Don't offer platitudes or untruths.")
        message_parts.append("- Don't project confidence when you are uncertain.")

        message = "\n".join(message_parts)
        logger.info(f"Foreground developer message:\n{message}")
        return message

    async def finalize(
            self,
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            conversation_uuid: str,
            error: str,
            generated_response: str,
            prompt_uuid: str,
            tool_calls: list[ToolCallDoc],
            topic_observer: TopicClassifyingObserver,
            user_context: PersonContext,
            user_fact_observer: UserFactsSavingObserver,
            user_id: int,
    ):
        common_labels = [
            f"conversation:{conversation_uuid}",
            f"user:{user_id}",
        ]
        topics = await topic_observer.get_topics()
        prompt, response = await asyncio.gather(
            self.search_client.index_prompt(
                about=topics,
                prompt_uuid=prompt_uuid,
                labels=common_labels,
                messages=[
                    # Drop developer message, FunctionCallOutput, and ResponseFunctionToolCall
                    msg for msg in context_window
                    if isinstance(msg, (ChatMessage,)) and msg.role in {"assistant", "user"}
                ],
            ),
            self.search_client.index_response(
                about=topics,
                prompt_uuid=prompt_uuid,
                labels=common_labels,
                generated_response=generated_response,
                error=error,
            ),
        )

        # TODO: Evaluate recalled tool outs and facts?

        tasks = []

        if prompt and topics:
            tasks.append(asyncio.create_task(self.graph_client.create_topic_to_prompt_edge(prompt, topics)))
        if response and topics:
            tasks.append(asyncio.create_task(self.graph_client.create_topic_to_response_edge(response, topics)))

        saved_facts = await user_fact_observer.get_saved_facts()
        if prompt and saved_facts:
            tasks.append(asyncio.create_task(self.graph_client.create_fact_to_prompt_edge(prompt, saved_facts)))

        await asyncio.gather(*tasks)

    @classmethod
    async def get_conversation(cls, request: Request):
        await load_session(request)
        session = request.session
        if not session:
            raise HTTPException(status_code=400, detail="Session expired")
        return {"conversation_uuid": uuid.uuid4()}

    def init(self):
        """Synchronous setup steps"""
        super().init()
        self.graph_client.init_client()
        self.search_client.init_client()
        self.app.add_api_route("/chat", self.post_chat, methods=["POST"])
        self.app.add_api_route("/conversation", self.get_conversation, methods=["GET"])

    async def on_start(self):
        """Asynchronous setup steps"""
        pass

    async def on_stop(self):
        """Asynchronous teardown steps"""
        await self.graph_client.close()
        await self.search_client.close()

    async def post_chat(
            self,
            request: Request,
            payload: PromptPayload,
            latitude: float = Query(None),
            longitude: float = Query(None),
            model: str = Query("gpt-5.1"),
            reasoning_effort: str = Query(None),
            time_zone: str = Query(None),
    ):
        """Chat completion."""
        await load_session(request)
        session = request.session
        if not session:
            raise HTTPException(status_code=400, detail="Session expired")

        if len(payload.messages) == 0:
            raise HTTPException(status_code=400, detail="At least one message is required")

        preprocessing_tasks = []

        context_observer: ContextFilteringObserver | None = None
        if len(payload.messages) > 4:  # After third user message
            context_observer = ContextFilteringObserver(payload.messages, self.llm_client)
            context_observer.observe_in_background(payload.messages)
            preprocessing_tasks.append(context_observer.bg_task)

        topic_observer = TopicClassifyingObserver(self.llm_client)
        topic_observer.observe_in_background(payload.messages)

        prompt_context = PromptContext.new(latitude=latitude, longitude=longitude, time_zone=time_zone)

        # Payload arrives        >
        #                        *   Here, in this function, we search for the most recent PromptDoc labeled with
        #                            the current conversation UUID, that has a non-empty summary field. If something is
        #                            found, diff the messages to figure out which messages are not covered, then.
        #                            truncate everything above the summary.
        #                        <   Assistant returns new prompt UUID, responds, then indexes the prompt and response.
        #                        *   A controller watches for new prompts, checks the context length after truncation
        #                            and decides to update a summary field on the PromptDoc.
        # Pext payload arrives   >

        # TODO: Maybe a running log of what actions have been taken in each conversation would be helpful

        if preprocessing_tasks:
            await asyncio.gather(*preprocessing_tasks)

        # TODO: Use this filtered view as the context window but create a tool that allows access to the full history
        filtered_user_messages = (
            # First two user messages before context observer takes over
            " ".join([m.content for m in payload.messages if m.role == "user"][-2:]) if context_observer is None
            else " ".join([
                m.content for m in await context_observer.get_filtered_context()
                if m.role == "user"
            ])
        )
        search_text, search_emb = await normalize_text_for_search_and_embed(filtered_user_messages)
        logger.info(f"Search text: {search_text}")

        tool_calls, user_context = await asyncio.gather(
            self.search_client.search_tool_call_by_anchor(
                search_text, search_emb,
                top_k=3,
            ),
            self.search_client.get_user_context(
                session["full_name"], session["user_id"],
                match=search_text,
                match_emb=search_emb,
            ),
        )

        user_fact_observer = UserFactsSavingObserver(prompt_context, user_context, self.llm_client, self.search_client)
        user_fact_observer.observe_in_background(payload.messages)

        context_window = [
            ChatMessage(
                role="developer",
                content=self.developer_message(prompt_context, user_context, tool_calls)
            )
        ]
        context_window.extend(payload.messages)
        return StreamingResponse(
            self.stream_and_finalize(
                context_window=context_window,
                conversation_uuid=payload.conversation_uuid,
                response_generator=self.llm_client.stream_response(
                    context_window, model,
                    reasoning_effort=reasoning_effort,
                    stream_ndjson=True,
                    tool_callbacks=self.tools_callbacks,
                    tool_params=self.tools_params,
                ),
                tool_calls=tool_calls,
                topic_observer=topic_observer,
                user_context=user_context,
                user_fact_observer=user_fact_observer,
                user_id=user_context.user_id,
            ),
            media_type="text/event-stream",
        )

    async def stream_and_finalize(
            self,
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            conversation_uuid: str,
            response_generator: AsyncGenerator[str, Any],
            tool_calls: list[ToolCallDoc],
            topic_observer: TopicClassifyingObserver,
            user_context: PersonContext,
            user_fact_observer: UserFactsSavingObserver,
            user_id: int,
    ) -> AsyncGenerator[str, Any]:
        prompt_uuid = str(uuid.uuid4())
        yield json.dumps({"prompt_uuid": prompt_uuid}) + "\n"

        error = None
        generated_response = ""
        try:
            async for str_to_yield in response_generator:
                if not str_to_yield:
                    logger.warning(f"Received empty '{str_to_yield}' to yield")
                    continue
                # Should be NDJSON
                if str_to_yield.startswith('{"text_delta":'):
                    generated_response += json.loads(str_to_yield)["text_delta"]
                yield str_to_yield
        except Exception as e:
            error = str(e)
            raise
        finally:
            self.background_and_forget(self.finalize(
                context_window=context_window,
                conversation_uuid=conversation_uuid,
                error=error,
                generated_response=generated_response,
                prompt_uuid=prompt_uuid,
                tool_calls=tool_calls,
                topic_observer=topic_observer,
                user_context=user_context,
                user_fact_observer=user_fact_observer,
                user_id=user_id,
            ))
