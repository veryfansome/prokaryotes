import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import (
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_input_param import FunctionCallOutput
from starlette.concurrency import run_in_threadpool
from starsessions import load_session

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
    PromptContext,
    PromptPayload,
    ToolCallDoc,
)
from prokaryotes.observer_v1.fact_observer import FactSavingObserver
from prokaryotes.observer_v1.summarizing_observer import MessageSummarizingObserver
from prokaryotes.observer_v1.topic_observer import TopicClassifyingObserver
from prokaryotes.search_v1 import SearchClient
from prokaryotes.tools_v1.builtins import web_search_tool_param
from prokaryotes.tools_v1.read_file import ReadFileCallback
from prokaryotes.tools_v1.recall_responses import RecallResponsesCallback
from prokaryotes.tools_v1.scan_directory import ScanDirectoryCallback
from prokaryotes.tools_v1.search_email import SearchEmailCallback
from prokaryotes.utils_v1.context_utils import developer_message_parts
from prokaryotes.utils_v1.text_utils import (
    get_document_embs,
    get_query_embs,
    normalize_text_for_search,
)
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

    @classmethod
    def developer_message(
            cls,
            prompt_context: PromptContext,
            recalled_facts: list[FactDoc],
            recalled_tool_calls: list[ToolCallDoc],
            recalled_user_context: PersonContext,
    ):
        message_parts = developer_message_parts(
            prompt_context,
            recalled_facts,
            recalled_user_context,
        )

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
            fact_observer: FactSavingObserver,
            prompt_uuid: str,
            recalled_facts: list[FactDoc],
            recalled_tool_calls: list[ToolCallDoc],
            recalled_user_context: PersonContext,
            topic_observer: TopicClassifyingObserver,
    ):
        generated_response_idx = len(context_window) - 1
        generated_response = context_window[generated_response_idx].content
        prompt_messages = [
            # Drop developer message, FunctionCallOutput, and ResponseFunctionToolCall
            msg for idx, msg in enumerate(context_window)
            if isinstance(msg, ChatMessage) and msg.role in {"assistant", "user"} and idx != generated_response_idx
        ]
        common_labels = [
            f"conversation:{conversation_uuid}",
            f"user:{recalled_user_context.user_id}",
        ]

        (
            generated_response_emb,
            (
                topics,
                topic_embs,
            ),
        ) = await asyncio.gather(
            self.get_response_embs(generated_response),
            self.get_topic_embs(topic_observer),
        )
        prompt_doc, response_doc, _ = await asyncio.gather(
            self.search_client.index_prompt(
                about=topics,
                prompt_uuid=prompt_uuid,
                labels=common_labels,
                messages=prompt_messages,
            ),
            self.search_client.index_response(
                prompt_uuid=prompt_uuid,
                about=topics,
                labels=common_labels,
                response_text=generated_response,
                response_emb=generated_response_emb,
            ),
            self.search_client.index_topics(topics, topic_embs),
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
                        self.graph_client.create_tool_call_to_prompt_edge(prompt_doc, tool_call)
                    ))

        if prompt_doc and topics:
            tasks.append(asyncio.create_task(
                self.graph_client.create_topic_to_prompt_edge(prompt_doc, topics)
            ))
        if response_doc and topics:
            tasks.append(asyncio.create_task(
                self.graph_client.create_topic_to_response_edge(response_doc, topics)
            ))

        saved_facts = await fact_observer.get_saved_facts()
        if prompt_doc and saved_facts:
            tasks.append(asyncio.create_task(
                self.graph_client.create_fact_to_prompt_edge(prompt_doc, saved_facts)
            ))

        await asyncio.gather(*tasks)

    @classmethod
    async def get_conversation(cls, request: Request):
        await load_session(request)
        session = request.session
        if not session:
            raise HTTPException(status_code=400, detail="Session expired")
        return {"conversation_uuid": uuid.uuid4()}

    @classmethod
    async def get_response_embs(cls, generated_response: str) -> list[float]:
        generated_response_normalized = await run_in_threadpool(normalize_text_for_search, generated_response)
        return (await get_document_embs([generated_response_normalized]))[0]

    @classmethod
    async def get_topic_embs(cls, topic_observer: TopicClassifyingObserver) -> tuple[list[str], list[list[float]]]:
        topics = await topic_observer.get_topics()
        return topics, await get_document_embs(topics)

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

        # Generating summaries for recall before streaming responses slows things down. Sometimes, the juice might not
        # be worth the squeeze. This crude heuristic is used to guess if we should skip summarizing.
        search_text = payload.messages[-1].content
        if len(search_text.split()) > 10:

            summary_observer = MessageSummarizingObserver(self.llm_client)
            await summary_observer.observe(payload.messages.copy())

            # TODO: Index search_text as a FactDoc if no fact tool calls?
            search_text = await summary_observer.get_summary()

        summary_emb = (await get_query_embs([search_text]))[0]
        # TODO: Recall tool call outputs that are linked to previous messages in the context_window
        (
            recalled_facts,
            recalled_tool_calls,
            recalled_user_context,
        ) = await asyncio.gather(
            self.search_client.search_facts(
                match=search_text,
                match_emb=summary_emb,
                min_score=0.75,
                not_about=f"user:{session['user_id']}",
            ),
            self.search_client.search_tool_call_by_anchor(
                search_text, summary_emb,
                min_score=0.75,
                top_k=3,
            ),
            self.search_client.get_user_context(
                session["full_name"], session["user_id"],
                match=search_text,
                match_emb=summary_emb,
                min_facts_score=0.75,
            ),
        )

        prompt_context = PromptContext.new(latitude=latitude, longitude=longitude, time_zone=time_zone)

        # Observers 
        fact_observer = FactSavingObserver(
            self.llm_client,
            self.search_client,
            prompt_context,
            recalled_facts,
            recalled_user_context,
        )
        fact_observer.observe_in_background(payload.messages.copy())
        topic_observer = TopicClassifyingObserver(self.llm_client)
        topic_observer.observe_in_background(payload.messages.copy())

        # Tools 
        response_recall_tool = RecallResponsesCallback(self.search_client, session["user_id"])
        tool_callbacks = (self.tool_callbacks | {
            response_recall_tool.tool_param["name"]: response_recall_tool
        })
        tool_params = self.tool_params + [
            response_recall_tool.tool_param,
            web_search_tool_param,
        ]

        # Stream response 
        context_window = payload.messages.copy()
        context_window.insert(0, ChatMessage(
            role="developer",
            content=self.developer_message(
                prompt_context,
                recalled_facts,
                recalled_tool_calls,
                recalled_user_context,
            ),
        ))
        return StreamingResponse(
            self.stream_and_finalize(
                context_window=context_window,
                conversation_uuid=payload.conversation_uuid,
                fact_observer=fact_observer,
                recalled_facts=recalled_facts,
                recalled_tool_calls=recalled_tool_calls,
                recalled_user_context=recalled_user_context,
                response_generator=self.llm_client.stream_response(
                    context_window, model,
                    #log_events=True,
                    reasoning_effort=reasoning_effort,
                    stream_ndjson=True,
                    tool_callbacks=tool_callbacks,
                    tool_params=tool_params,
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
            recalled_facts: list[FactDoc],
            recalled_tool_calls: list[ToolCallDoc],
            recalled_user_context: PersonContext,
            response_generator: AsyncGenerator[str, Any],
            topic_observer: TopicClassifyingObserver,
    ) -> AsyncGenerator[str, Any]:
        prompt_uuid = str(uuid.uuid4())
        yield json.dumps({"prompt_uuid": prompt_uuid}) + "\n"

        async for str_to_yield in response_generator:
            if not str_to_yield:
                logger.warning(f"Received empty '{str_to_yield}' to yield")
                continue
            yield str_to_yield

        self.background_and_forget(self.finalize(
            context_window=context_window,
            conversation_uuid=conversation_uuid,
            fact_observer=fact_observer,
            prompt_uuid=prompt_uuid,
            recalled_facts=recalled_facts,
            recalled_tool_calls=recalled_tool_calls,
            recalled_user_context=recalled_user_context,
            topic_observer=topic_observer,
        ))
