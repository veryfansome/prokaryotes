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
from pydantic import TypeAdapter
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
    ContextWindowItem,
    FactDoc,
    PersonContext,
    PromptContext,
    PromptPayload,
    ToolCallDoc,
)
from prokaryotes.observer_v1.fact_observer import FactSavingObserver
from prokaryotes.observer_v1.named_entity_observer import NamedEntityObserver
from prokaryotes.observer_v1.summarizing_observer import MessageSummarizingObserver
from prokaryotes.observer_v1.topic_observer import TopicClassifyingObserver
from prokaryotes.search_v1 import SearchClient
from prokaryotes.tools_v1.builtins import web_search_tool_param
from prokaryotes.tools_v1.recall_responses import RecallResponsesCallback
from prokaryotes.tools_v1.shell_command import ShellCommandCallback
from prokaryotes.utils_v1.context_utils import developer_message_parts
from prokaryotes.utils_v1.text_utils import (
    get_document_embs,
    get_query_embs,
    normalize_text_for_identity,
    normalize_text_for_search,
)
from prokaryotes.web_base_v1 import WebBase

logger = logging.getLogger(__name__)


class ProkaryoteV1(WebBase):
    def __init__(self, static_dir: str):
        super().__init__(static_dir)
        self.conversation_adapter = TypeAdapter(list[ContextWindowItem])
        self.conversation_cache_ex = 60 * 60 * 24 * 7
        self.graph_client = GraphClient()
        self.llm_client = get_llm_client()
        self.search_client = SearchClient()

        tools: list[FunctionToolCallback] = [
            ShellCommandCallback(self.search_client),
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
                created_at = tool_call_doc.created_at.astimezone(prompt_context.time_zone).strftime('%Y-%m-%d %H:%M')
                message_parts.append(
                    f"\n### {tool_call_doc.tool_name}\n"
                    f"Timestamp: {created_at}\n"
                    f"Arguments: {tool_call_doc.tool_arguments}\n"
                    f"<pre><code>{tool_call_doc.output}</code></pre>"
                )

        message_parts.append("---")
        message_parts.append("## Instructions")
        message_parts.append("- Use short messages. One sentences is best, unless the user explicitly requests more.")
        message_parts.append(
            "- When using the `run_shell_command` tool, don't chain multiple commands with '&&' or ';' unless"
            " the intended task can't be accomplished without chaining commands together. Whenver possible, use a"
            " single, focused `command`, with a distinct `reason` per tool call."
        )
        message_parts.append(
            "- When reading files, default to previewing the first 200 lines, e.g. `sed -n '1,200p' <path>`."
        )

        message = "\n".join(message_parts)
        logger.info(f"Foreground developer message:\n{message}")
        return message

    async def finalize(
            self,
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            conversation_uuid: str,
            fact_observer: FactSavingObserver,
            named_entity_observer: NamedEntityObserver,
            prompt_uuid: str,
            recalled_facts: list[FactDoc],
            recalled_tool_calls: list[ToolCallDoc],
            recalled_user_context: PersonContext,
            summary_observer: MessageSummarizingObserver,
            topic_observer: TopicClassifyingObserver,
    ):
        await self.redis_client.set(
            f"conversation:{conversation_uuid}", self.conversation_adapter.dump_json(context_window),
            ex=self.conversation_cache_ex,
        )

        func_call_resp = {}
        func_call_outputs = {}
        generated_response_idx = len(context_window) - 1
        generated_response = context_window[generated_response_idx].content
        last_user_message_idx = 0
        prompt_messages = []
        for idx, obj in enumerate(context_window):
            # Drop the developer message, any FunctionCallOutput or ResponseFunctionToolCall, and the generated
            # response, which will be index separately.
            if isinstance(obj, ChatMessage) and obj.role in {"assistant", "user"} and idx != generated_response_idx:
                if obj.role == "user" and idx > last_user_message_idx:
                    last_user_message_idx = idx
                prompt_messages.append(obj)
            elif isinstance(obj, ResponseFunctionToolCall):
                func_call_resp[obj.call_id] = (idx, obj)
            elif isinstance(obj, dict) and "call_id" in obj and "output" in obj:
                func_call_outputs[obj["call_id"]] = obj["output"]

        common_labels = [
            f"conversation:{conversation_uuid}",
            f"user:{recalled_user_context.user_id}",
        ]

        (
            generated_response_emb,
            (
                named_entities,
                named_entity_embs,
            ),
            (
                summary,
                summary_embs,
            ),
        ) = await asyncio.gather(
            self.get_generated_response_emb(generated_response),
            self.get_named_entities_embs(named_entity_observer),
            self.get_summary_emb(summary_observer),
        )
        topics, topic_embs = await self.get_topic_embs(
            topic_observer,
            excluded_topics=set(named_entities),
        )
        prompt_doc, _, _, _ = await asyncio.gather(
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
            self.search_client.index_named_entities(named_entities, named_entity_embs),
        )
        similar_topic_pairs = await self.get_similar_topic_pairs(topics, topic_embs)

        # TODO: Evaluate recalled tool outs and facts?

        tasks = []

        for call_id, output in func_call_outputs.items():
            call_resp_idx, call_resp = func_call_resp[call_id]
            if call_resp.name in self.tool_callbacks:
                tool_callback = self.tool_callbacks[call_resp.name]
                if isinstance(tool_callback, FunctionCallOutputIndexer):
                    if call_resp_idx > last_user_message_idx:
                        tool_call = await tool_callback.index(
                            call_id=call_id,
                            arguments=call_resp.arguments,
                            labels=common_labels,
                            output=output,
                            prompt_summary=summary,
                            prompt_summary_emb=summary_embs,
                            topics=topics,
                        )
                        if tool_call and prompt_doc:
                            tasks.append(asyncio.create_task(
                                self.graph_client.create_tool_call_to_prompt_edge(prompt_doc, tool_call)
                            ))
                    else:
                        tool_call = await self.search_client.get_tool_call(call_id)
                        if tool_call and prompt_doc:
                            tasks.append(asyncio.create_task(
                                self.graph_client.create_tool_call_context_to_prompt_edge(prompt_doc, tool_call)
                            ))

        if prompt_doc and named_entities:
            tasks.append(asyncio.create_task(
                self.graph_client.create_named_entity_to_prompt_edge(prompt_doc, named_entities)
            ))
        if prompt_doc and topics:
            tasks.append(asyncio.create_task(
                self.graph_client.create_topic_to_prompt_edge(prompt_doc, topics)
            ))

        saved_facts = await fact_observer.get_saved_facts()
        if prompt_doc and saved_facts:
            tasks.append(asyncio.create_task(
                self.graph_client.create_fact_to_prompt_edge(prompt_doc, saved_facts)
            ))
        fact_entity_matches = self.find_named_entities_in_facts(saved_facts, named_entities)
        for fact, fact_named_entities in fact_entity_matches:
            tasks.append(asyncio.create_task(
                self.graph_client.create_named_entity_to_fact_edge([fact], fact_named_entities)
            ))
        if similar_topic_pairs:
            tasks.append(asyncio.create_task(
                self.graph_client.create_similar_topic_edges(similar_topic_pairs)
            ))

        await asyncio.gather(*tasks)

    @classmethod
    def find_context_divergence(
            cls,
            list1: list[ChatMessage],
            list2: list[ChatMessage],
    ) -> tuple[int | None, bool, bool]:
        list1_len = len(list1)
        list2_len = len(list2)
        for idx, (item1, item2) in enumerate(zip(list1, list2, strict=False)):
            if item1 != item2:
                return idx, True, list1_len > list2_len
        if list1_len == list2_len:
            return None, False, False
        else:
            return min(list1_len, list2_len), False, list1_len > list2_len

    @classmethod
    def find_named_entities_in_facts(
            cls,
            facts: list[FactDoc],
            named_entities: list[str],
    ) -> list[tuple[FactDoc, list[str]]]:
        if not facts or not named_entities:
            return []

        normalized_entities = []
        seen_entities = set()
        for named_entity in named_entities:
            normalized_entity = normalize_text_for_identity(named_entity).lower()
            if not normalized_entity or normalized_entity in seen_entities:
                continue
            seen_entities.add(normalized_entity)
            normalized_entities.append((named_entity, normalized_entity))

        matched_pairs = []
        for fact in facts:
            fact_text = normalize_text_for_identity(fact.text).lower()
            matched_entities = [
                named_entity
                for named_entity, normalized_entity in normalized_entities
                if normalized_entity in fact_text
            ]
            if matched_entities:
                matched_pairs.append((fact, matched_entities))
        return matched_pairs

    @classmethod
    async def get_conversation(cls, request: Request):
        await load_session(request)
        session = request.session
        if not session:
            raise HTTPException(status_code=400, detail="Session expired")
        return {"conversation_uuid": uuid.uuid4()}

    @classmethod
    async def get_named_entities_embs(
            cls,
            named_entity_observer: NamedEntityObserver,
    ) -> tuple[list[str], list[list[float]]]:
        raw_named_entities = await named_entity_observer.get_named_entities()
        named_entities = []
        seen_named_entities = set()
        for named_entity in raw_named_entities:
            named_entity = normalize_text_for_identity(named_entity)
            if not named_entity or named_entity in seen_named_entities:
                continue
            seen_named_entities.add(named_entity)
            named_entities.append(named_entity)
        if not named_entities:
            return [], []
        return named_entities, await get_document_embs(tuple(named_entities))

    @classmethod
    async def get_generated_response_emb(cls, generated_response: str) -> list[float]:
        generated_response_normalized = await run_in_threadpool(normalize_text_for_search, generated_response)
        return (await get_document_embs((generated_response_normalized,)))[0]

    async def get_similar_topic_pairs(
            self,
            topics: list[str],
            topic_embs: list[list[float]],
            min_score: float = 0.95,
    ) -> list[tuple[str, str]]:
        if not topics:
            return []
        search_tasks = []
        for idx, topic in enumerate(topics):
            topic_emb = topic_embs[idx] if idx < len(topic_embs) else []
            search_tasks.append(asyncio.create_task(
                self.search_client.search_topics(
                    match=topic,
                    match_emb=topic_emb,
                    min_lexical_score=min_score,
                )
            ))
        similar_topics_by_topic = await asyncio.gather(*search_tasks)
        similar_topic_pairs = set()
        for topic, similar_topics in zip(topics, similar_topics_by_topic, strict=False):
            normalized_topic = normalize_text_for_identity(topic)
            normalized_topic_lower = normalized_topic.lower()
            for similar_topic in similar_topics:
                normalized_similar_topic = normalize_text_for_identity(similar_topic)
                if not normalized_similar_topic:
                    continue
                normalized_similar_topic_lower = normalized_similar_topic.lower()
                if normalized_topic_lower == normalized_similar_topic_lower:
                    continue
                similar_topic_pairs.add(tuple(sorted((normalized_topic, normalized_similar_topic))))
        return sorted(similar_topic_pairs)

    @classmethod
    async def get_summary_emb(cls, summary_observer: MessageSummarizingObserver,) -> tuple[str, list[float]]:
        summary_text = await summary_observer.get_summary()
        return summary_text, (await get_document_embs((summary_text,)))[0]

    @classmethod
    async def get_topic_embs(
            cls,
            topic_observer: TopicClassifyingObserver,
            excluded_topics: set[str] | None = None,
    ) -> tuple[list[str], list[list[float]]]:
        raw_topics = await topic_observer.get_topics()
        excluded_topics = excluded_topics or set()
        topics = []
        seen_topics = set()
        for topic in raw_topics:
            topic = normalize_text_for_identity(topic)
            if not topic or topic in seen_topics or topic in excluded_topics:
                continue
            seen_topics.add(topic)
            topics.append(topic)
        if not topics:
            return [], []
        return topics, await get_document_embs(tuple(topics))

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

        prompt_context = PromptContext.new(latitude=latitude, longitude=longitude, time_zone=time_zone)

        # Context window

        cached_conversation = await self.redis_client.get(f"conversation:{payload.conversation_uuid}")
        if cached_conversation:
            logger.info("Using context_window from cache")
            context_window = self.conversation_adapter.validate_json(cached_conversation)
            if isinstance(context_window[0], ChatMessage) and context_window[0].role == "developer":
                context_window.pop(0)  # Drop old developer message
            context_window = self.sync_context_window(context_window, payload.messages)
        else:
            logger.info("Starting new context_window from request payload")
            context_window = payload.messages.copy()

        # Pre-recall observers 

        named_entity_observer = NamedEntityObserver(self.llm_client, self.search_client)
        named_entity_observer.observe_in_background(payload.messages.copy())
        summary_observer = MessageSummarizingObserver(self.llm_client)
        summary_observer.observe_in_background(payload.messages.copy())
        topic_observer = TopicClassifyingObserver(self.llm_client, self.search_client)
        topic_observer.observe_in_background(payload.messages.copy())

        # Recall

        context_recall_text = await run_in_threadpool(normalize_text_for_search, payload.messages[-1].content)
        tool_recall_text = context_recall_text
        # If context_recall_text is long, use the generated summary
        if len(context_recall_text.split()) > 10:
            context_recall_text = await summary_observer.get_summary()
        if context_recall_text == tool_recall_text:
            context_recall_emb = tool_recall_emb = (await get_query_embs((context_recall_text,)))[0]
        else:
            context_recall_emb, tool_recall_emb = await get_query_embs((context_recall_text, tool_recall_text))
        logger.info(
            f"Recall text:\n"
            f"- Context: {context_recall_text}\n"
            f"- Tool-call: {tool_recall_text}"
        )
        (
            recalled_facts,
            recalled_tool_calls,
            recalled_user_context,
        ) = await asyncio.gather(
            self.search_client.search_facts(
                match=context_recall_text,
                match_emb=context_recall_emb,
                min_score=1.0,
                not_about=f"user:{session['user_id']}",  # TODO: Exclude other users as well
            ),
            # TODO: It might be a good idea to have a LLM review and filter recalled results
            self.search_client.search_tool_call(
                # Exclude if already in the context window
                excluded_ids=[obj.call_id for obj in context_window if isinstance(obj, ResponseFunctionToolCall)],
                match=tool_recall_text,
                match_emb=tool_recall_emb,
                min_lexical_score=1.0,
                min_final_score=0.75,
            ),
            self.search_client.get_user_context(
                session["full_name"], session["user_id"],
                match=context_recall_text,
                match_emb=context_recall_emb,
                min_facts_score=1.0,
            ),
        )

        # Post-recall observers 

        fact_observer = FactSavingObserver(
            self.llm_client,
            self.search_client,
            prompt_context,
            recalled_facts,
            recalled_user_context,
        )
        fact_observer.observe_in_background(payload.messages.copy())

        # Tools 

        response_recall_tool = RecallResponsesCallback(self.search_client, session["user_id"])
        tool_callbacks = (self.tool_callbacks | {
            response_recall_tool.tool_param["name"]: response_recall_tool
        })
        tool_params = self.tool_params + [
            response_recall_tool.tool_param,
            web_search_tool_param,
        ]

        # Developer message

        developer_message = ChatMessage(
            role="developer",
            content=self.developer_message(
                prompt_context,
                recalled_facts,
                recalled_tool_calls,
                recalled_user_context,
            )
        )
        context_window.insert(0, developer_message)

        # Stream response 

        return StreamingResponse(
            self.stream_and_finalize(
                context_window=context_window,
                conversation_uuid=payload.conversation_uuid,
                fact_observer=fact_observer,
                named_entity_observer=named_entity_observer,
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
                summary_observer=summary_observer,
                topic_observer=topic_observer,
            ),
            media_type="text/event-stream",
        )

    async def stream_and_finalize(
            self,
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            conversation_uuid: str,
            fact_observer: FactSavingObserver,
            named_entity_observer: NamedEntityObserver,
            recalled_facts: list[FactDoc],
            recalled_tool_calls: list[ToolCallDoc],
            recalled_user_context: PersonContext,
            response_generator: AsyncGenerator[str, Any],
            summary_observer: MessageSummarizingObserver,
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
            named_entity_observer=named_entity_observer,
            prompt_uuid=prompt_uuid,
            recalled_facts=recalled_facts,
            recalled_tool_calls=recalled_tool_calls,
            recalled_user_context=recalled_user_context,
            summary_observer=summary_observer,
            topic_observer=topic_observer,
        ))

    @classmethod
    def sync_context_window(
            cls,
            context_window: list[ChatMessage | FunctionCallOutput | ResponseFunctionToolCall],
            payload_messages: list[ChatMessage],
    ):
        context_window_messages = []
        context_window_message_indexes = []
        for idx, item in enumerate(context_window):
            if isinstance(item, ChatMessage):  # Exclude any FunctionCallOutput or ResponseFunctionToolCall
                context_window_messages.append(item)
                context_window_message_indexes.append(idx)
        divergence_idx, is_mismatch, cached_context_is_longer = cls.find_context_divergence(
            context_window_messages,
            payload_messages,
        )
        if divergence_idx is None:
            raise HTTPException(status_code=400, detail="Payload does not alter conversation state")
        elif cached_context_is_longer:
            divergence_idx = context_window_message_indexes[divergence_idx]
            if is_mismatch:
                logger.info("Cached state does not match payload, earlier message changed")
                context_window = context_window[:divergence_idx]
                context_window.append(payload_messages[-1])
            else:
                logger.info("Cached state does not match payload, retrying earlier message")
                context_window = context_window[:divergence_idx]
        else:
            logger.info("Request payload is ahead of cached state, patching new messages from payload")
            context_window = context_window[:context_window_message_indexes[-1] + 1]  # Truncate to last good
            context_window.extend(payload_messages[divergence_idx:])
        return context_window
