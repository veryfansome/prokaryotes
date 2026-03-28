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
    FunctionCallOutputIndexer,
    FunctionToolCallback,
    get_llm_client,
)
from prokaryotes.models_v1 import (
    ChatMessage,
    FactDoc,
    PersonContext,
    PromptPayload,
    PromptContext,
    ToolCallDoc,
)
from prokaryotes.observer_v1.summarizing_observer import MessageSummarizingObserver
from prokaryotes.observer_v1.topic_observer import TopicClassifyingObserver
from prokaryotes.observer_v1.fact_observer import FactSavingObserver
from prokaryotes.search_v1 import SearchClient
from prokaryotes.tools_v1.builtins import web_search_tool_param
from prokaryotes.tools_v1.read_file import ReadFileCallback
from prokaryotes.tools_v1.scan_directory import ScanDirectoryCallback
from prokaryotes.tools_v1.search_email import SearchEmailCallback
from prokaryotes.utils_v1.context_utils import developer_message_parts
from prokaryotes.utils_v1.text_utils import get_query_embedding
from prokaryotes.web_base_v1 import WebBase

logger = logging.getLogger(__name__)


class ProkaryoteV1(WebBase):
    def __init__(self, static_dir: str):
        super().__init__(static_dir)
        self.graph_client = GraphClient()
        self.llm_client = get_llm_client()
        self.search_client = SearchClient()

        # TODO: Tool for intentional recall
        tools: list[FunctionToolCallback] = [
            ReadFileCallback(self.llm_client, self.search_client),
            ScanDirectoryCallback(self.llm_client, self.search_client),
            SearchEmailCallback(self.search_client),
        ]
        self.tool_callbacks = {t.tool_param["name"]: t for t in tools}
        self.tool_params = [t.tool_param for t in tools]
        self.tool_params.append(web_search_tool_param)

    @classmethod
    def developer_message(
            cls,
            prompt_context: PromptContext,
            recalled_user_context: PersonContext,
            recalled_facts: list[FactDoc],
            recalled_tool_calls: list[ToolCallDoc],
    ):
        message_parts = developer_message_parts(prompt_context, recalled_user_context, recalled_facts)

        if recalled_tool_calls:
            # TODO: Maybe there should be some heuristic here about what to prioritize
            message_parts.append("---")
            message_parts.append("## Recalled tool outputs")
            for tool_call_doc in recalled_tool_calls:
                updated_at = tool_call_doc.updated_at.astimezone(prompt_context.time_zone).strftime('%Y-%m-%d %H:%M')
                message_parts.append(
                    f"\n### {', '.join(tool_call_doc.labels)}\n"
                    f"Updated at: {updated_at}\n"
                    f"<pre><code>{tool_call_doc.output}</code></pre>"
                )

        message_parts.append("---")
        message_parts.append("## Instructions")
        message_parts.append("- Use short messages. One sentences is best, unless the user explicitly requests more.")

        message = "\n".join(message_parts)
        logger.info(f"Foreground developer message:\n{message}")
        return message

    async def finalize(
            self,
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            conversation_uuid: str,
            error: str,
            fact_observer: FactSavingObserver,
            generated_response: str,
            prompt_uuid: str,
            recalled_tool_calls: list[ToolCallDoc],
            recalled_user_context: PersonContext,
            topic_observer: TopicClassifyingObserver,
    ):
        prompt_messages = [
            # Drop developer message, FunctionCallOutput, and ResponseFunctionToolCall
            msg for msg in context_window
            if isinstance(msg, ChatMessage) and msg.role in {"assistant", "user"}
        ]
        common_labels = [
            f"conversation:{conversation_uuid}",
            f"user:{recalled_user_context.user_id}",
        ]
        topics = await topic_observer.get_topics()
        prompt, response = await asyncio.gather(
            self.search_client.index_prompt(
                about=topics,
                prompt_uuid=prompt_uuid,
                labels=common_labels,
                messages=prompt_messages,
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

        func_call_resp = {}
        func_call_outputs = {}
        for obj in context_window:
            if isinstance(obj, ResponseFunctionToolCall):
                func_call_resp[obj.call_id] = obj
            elif isinstance(obj, dict) and "call_id" in obj and "output" in obj:
                func_call_outputs[obj["call_id"]] = obj["output"]
        for call_id, output in func_call_outputs.items():
            call_resp = func_call_resp[call_id]
            if call_resp.name in self.tool_callbacks:  # In case of adhoc callbacks like the chat history callback
                tool_callback = self.tool_callbacks[call_resp.name]
                if isinstance(tool_callback, FunctionCallOutputIndexer):
                    tool_call = await tool_callback.index(prompt_messages.copy(), call_resp.arguments, output)
                    tasks.append(asyncio.create_task(
                        self.graph_client.create_tool_call_to_prompt_edge(prompt, tool_call)
                    ))

        if prompt and topics:
            tasks.append(asyncio.create_task(
                self.graph_client.create_topic_to_prompt_edge(prompt, topics)
            ))
        if response and topics:
            tasks.append(asyncio.create_task(
                self.graph_client.create_topic_to_response_edge(response, topics)
            ))

        saved_facts = await fact_observer.get_saved_facts()
        if prompt and saved_facts:
            tasks.append(asyncio.create_task(
                self.graph_client.create_fact_to_prompt_edge(prompt, saved_facts)
            ))

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

        summary_observer = MessageSummarizingObserver(self.llm_client)
        preprocessing_tasks.append(asyncio.create_task(summary_observer.observe(payload.messages.copy())))

        if preprocessing_tasks:
            await asyncio.gather(*preprocessing_tasks)

        topic_observer = TopicClassifyingObserver(self.llm_client)
        topic_observer.observe_in_background(payload.messages.copy())

        search_text = await summary_observer.get_summary()
        search_emb = await get_query_embedding(search_text)

        # TODO: Recall tool call outputs that are linked to previous messages in the context_window
        (
            recalled_user_context,
            recalled_facts,
            recalled_tool_calls,
        ) = await asyncio.gather(
            self.search_client.get_user_context(
                session["full_name"], session["user_id"],
                match=search_text,
                match_emb=search_emb,
                min_facts_score=1.5,
            ),
            self.search_client.search_facts(
                match=search_text,
                match_emb=search_emb,
                min_score=1.5,
                not_about=f"user:{session['user_id']}",
            ),
            self.search_client.search_tool_call_by_anchor(
                search_text, search_emb,
                top_k=3,
            ),
        )

        prompt_context = PromptContext.new(latitude=latitude, longitude=longitude, time_zone=time_zone)

        fact_observer = FactSavingObserver(
            prompt_context,
            recalled_user_context,
            recalled_facts,
            self.llm_client,
            self.search_client,
        )
        fact_observer.observe_in_background(payload.messages.copy())

        context_window = payload.messages.copy()
        context_window.insert(0, ChatMessage(
            role="developer",
            content=self.developer_message(
                prompt_context,
                recalled_user_context,
                recalled_facts,
                recalled_tool_calls,
            ),
        ))

        return StreamingResponse(
            self.stream_and_finalize(
                context_window=context_window,
                conversation_uuid=payload.conversation_uuid,
                fact_observer=fact_observer,
                recalled_tool_calls=recalled_tool_calls,
                recalled_user_context=recalled_user_context,
                response_generator=self.llm_client.stream_response(
                    context_window, model,
                    reasoning_effort=reasoning_effort,
                    stream_ndjson=True,
                    tool_callbacks=self.tool_callbacks,
                    tool_params=self.tool_params,
                ),
                topic_observer=topic_observer,
            ),
            media_type="text/event-stream",
        )

    async def stream_and_finalize(
            self,
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            conversation_uuid: str,
            fact_observer: FactSavingObserver,
            recalled_tool_calls: list[ToolCallDoc],
            recalled_user_context: PersonContext,
            response_generator: AsyncGenerator[str, Any],
            topic_observer: TopicClassifyingObserver,
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
                fact_observer=fact_observer,
                generated_response=generated_response,
                prompt_uuid=prompt_uuid,
                recalled_tool_calls=recalled_tool_calls,
                recalled_user_context=recalled_user_context,
                topic_observer=topic_observer,
            ))
