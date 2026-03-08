import logging
from fastapi import (
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from starsessions import load_session
from typing import (
    Any,
    AsyncGenerator,
)

from prokaryotes.callbacks_v1 import (
    ListDirectoryCallback,
    ReadFileCallback,
    SearchEmailFunctionToolCallback,
)
from prokaryotes.graph_v1 import GraphClient
from prokaryotes.llm_v1 import get_llm_client
from prokaryotes.models_v1 import (
    ChatMessage,
    ChatCompletionPayload,
    ChatCompletionContext,
    TextEmbeddingPrompt,
    TextEmbeddingRequest,
)
from prokaryotes.observers_v1 import (
    TopicClassifyingObserver,
    UserFactsSavingObserver,
)
from prokaryotes.search_v1 import (
    PersonContext,
    SearchClient,
)
from prokaryotes.tool_params_v1 import (
    list_directory_tool_param,
    read_file_tool_param,
    search_email_tool_param,
    web_search_tool_param,
)
from prokaryotes.utils import (
    developer_message_parts,
    get_text_embeddings,
    prep_chat_message_text_for_search,
)
from prokaryotes.web_base import WebBase

logger = logging.getLogger(__name__)

class ProkaryoteV1(WebBase):
    def __init__(self, static_dir: str):
        super().__init__(static_dir)
        self.graph_client = GraphClient()
        self.llm_client = get_llm_client()
        self.search_client = SearchClient()
        self.tools_callbacks = {
            list_directory_tool_param["name"]: ListDirectoryCallback(),
            read_file_tool_param["name"]: ReadFileCallback(),
            search_email_tool_param["name"]: SearchEmailFunctionToolCallback(self.search_client),
        }
        self.tools_params = [
            list_directory_tool_param,
            read_file_tool_param,
            search_email_tool_param,
            web_search_tool_param,
        ]

    async def chat(
            self,
            request: Request,
            payload: ChatCompletionPayload,
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

        topic_observer = TopicClassifyingObserver(self.llm_client)
        topic_observer.observe_in_background(payload.messages)

        completion_context = ChatCompletionContext.new(latitude=latitude, longitude=longitude, time_zone=time_zone)

        search_query_text = await run_in_threadpool(
            prep_chat_message_text_for_search, " ".join(m.content for m in payload.messages if m.role == "user"),
        )
        logger.info(f"Search query text: {search_query_text}")
        emb_resp = None
        if search_query_text:
            emb_resp = await get_text_embeddings(TextEmbeddingRequest(
                prompt=TextEmbeddingPrompt.QUERY,
                texts=[search_query_text],  # TODO: Concatenating all message might not be ideal
                truncate_to=256,
            ))
        user_context = await self.search_client.get_user_context(
            session["full_name"], session["user_id"],
            match=(search_query_text if search_query_text else None),
            match_emb=(emb_resp.embeddings[0] if emb_resp else None),
        )

        user_fact_observer = UserFactsSavingObserver(completion_context, user_context, self.llm_client, self.search_client)
        user_fact_observer.observe_in_background(payload.messages)

        context_window = [ChatMessage(role="developer", content=self.developer_message(completion_context, user_context))]
        # TODO: Roll long contexts off but in a way that can be recalled
        context_window.extend(payload.messages)
        return StreamingResponse(
            self.stream_and_finalize(
                user_context.user_id,
                context_window,
                self.llm_client.stream_response(
                    context_window, model,
                    reasoning_effort=reasoning_effort,
                    tool_callbacks=self.tools_callbacks,
                    tool_params=self.tools_params,
                ),
                topic_observer,
                user_fact_observer,
            ),
            media_type="text/event-stream",
        )

    @classmethod
    def developer_message(cls, completion_context: ChatCompletionContext, user_context: PersonContext):
        message_parts = developer_message_parts(completion_context, user_context)
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
            user_id: int,
            messages: list[ChatMessage],
            topic_observer: TopicClassifyingObserver,
            user_fact_observer: UserFactsSavingObserver,
            error: str = None
    ):
        topics = await topic_observer.get_topics()
        completion = await self.search_client.index_chat_completion(
            topics, [f"user_{user_id}"], messages,
            error=error,
        )

        if completion and topics:
            await self.graph_client.create_topic_to_completion_edge(completion, topics)

        saved_facts = await user_fact_observer.get_saved_facts()
        if completion and saved_facts:
            await self.graph_client.create_fact_to_completion_edge(completion, saved_facts)

    def init(self):
        """Synchronous setup steps"""
        super().init()
        self.graph_client.init_client()
        self.search_client.init_client()
        self.app.add_api_route("/chat", self.chat, methods=["POST"])

    async def on_start(self):
        """Asynchronous setup steps"""
        pass

    async def on_stop(self):
        """Asynchronous teardown steps"""
        await self.graph_client.close()
        await self.search_client.close()

    async def stream_and_finalize(
            self,
            user_id: int,
            messages: list[ChatMessage],
            response_generator: AsyncGenerator[str, Any],
            topic_observer: TopicClassifyingObserver,
            user_fact_observer: UserFactsSavingObserver,
    ) -> AsyncGenerator[str, Any]:
        error = None
        messages_to_index = [msg for msg in messages if msg.role != "developer"]
        response_text = ""
        try:
            async for chunk in response_generator:
                response_text += chunk
                yield chunk
        except Exception as e:
            error = str(e)
            raise
        finally:
            messages_to_index.append(ChatMessage(role="assistant", content=response_text))
            self.background_and_forget(self.finalize(
                user_id,
                messages_to_index,
                topic_observer,
                user_fact_observer,
                error=error
            ))
