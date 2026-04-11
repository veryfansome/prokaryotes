import logging
import os

from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
)

from prokaryotes.models_v1 import (
    FactDoc,
    PromptDoc,
    ToolCallDoc,
)
from prokaryotes.utils_v1.text_utils import normalize_text_for_identity

logger = logging.getLogger(__name__)


class GraphClient:
    def __init__(self):
        self.driver: AsyncDriver | None = None

    async def close(self):
        await self.driver.close()

    async def create_fact_to_prompt_edge(self, prompt: PromptDoc, facts: list[FactDoc]):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (p:Prompt {doc_id: input_struct.prompt_id})
        WITH input_struct, p
        MERGE (f:Fact {doc_id: input_struct.fact_id})
        SET f.text = input_struct.fact_text
        WITH p, f
        MERGE (f)-[:LEARNED_FROM]->(p)
        """
        async def _edge(tx):
            input_structs = [{
                'prompt_id': prompt.doc_id,
                'fact_id': fact.doc_id,
                'fact_text': fact.text,
            } for fact in facts]
            await tx.run(cypher, input_structs=input_structs)
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    async def create_named_entity_to_fact_edge(self, facts: list[FactDoc], named_entities: list[str]):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (f:Fact {doc_id: input_struct.fact_id})
        SET f.text = input_struct.fact_text
        WITH input_struct, f
        MERGE (e:NamedEntity {text: input_struct.named_entity})
        WITH e, f
        MERGE (e)-[:ENTITY_OF]->(f)
        """
        async def _edge(tx):
            deduped_named_entities = self.dedupe_identity_texts(named_entities)
            if not facts or not deduped_named_entities:
                return
            input_structs = [{
                'fact_id': fact.doc_id,
                'fact_text': fact.text,
                'named_entity': named_entity,
            } for fact in facts for named_entity in deduped_named_entities]
            await tx.run(cypher, input_structs=input_structs)
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    async def create_named_entity_to_prompt_edge(self, prompt: PromptDoc, named_entities: list[str]):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (p:Prompt {doc_id: input_struct.prompt_id})
        WITH input_struct, p
        MERGE (e:NamedEntity {text: input_struct.named_entity})
        WITH p, e
        MERGE (e)-[:ENTITY_OF]->(p)
        """
        async def _edge(tx):
            deduped_named_entities = self.dedupe_identity_texts(named_entities)
            if not deduped_named_entities:
                return
            input_structs = [{
                'prompt_id': prompt.doc_id,
                'named_entity': named_entity,
            } for named_entity in deduped_named_entities]
            await tx.run(cypher, input_structs=input_structs)
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    async def create_prompt_node(self, prompt: PromptDoc):
        cypher = """
        MERGE (p:Prompt {doc_id: $input_struct.doc_id})
        SET p.labels = $input_struct.labels
        SET p.summary = $input_struct.summary
        """
        async def _node(tx):
            await tx.run(cypher, input_struct={
                'doc_id': prompt.doc_id,
                'labels': prompt.labels,
                'summary': prompt.summary,
            })
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_node)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    async def create_similar_topic_edges(self, topic_pairs: list[tuple[str, str]]):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (t1:Topic {text: input_struct.topic_1})
        MERGE (t2:Topic {text: input_struct.topic_2})
        WITH t1, t2
        WHERE t1.text <> t2.text
        MERGE (t1)-[:SIMILAR_TO]->(t2)
        """
        async def _edge(tx):
            seen_pairs = set()
            input_structs = []
            for topic_1, topic_2 in topic_pairs:
                topic_1 = normalize_text_for_identity(topic_1)
                topic_2 = normalize_text_for_identity(topic_2)
                if not topic_1 or not topic_2 or topic_1 == topic_2:
                    continue
                topic_1, topic_2 = sorted((topic_1, topic_2))
                pair = (topic_1, topic_2)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                input_structs.append({
                    'topic_1': topic_1,
                    'topic_2': topic_2,
                })
            if not input_structs:
                return
            await tx.run(cypher, input_structs=input_structs)
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    async def create_tool_call_context_to_prompt_edge(self, prompt: PromptDoc, tool_call: ToolCallDoc):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (p:Prompt {doc_id: input_struct.prompt_id})
        WITH input_struct, p
        MERGE (t:ToolCall {doc_id: input_struct.tool_call_id})
        WITH p, t
        MERGE (t)-[:CONTEXT_FOR]->(p)
        """
        async def _edge(tx):
            await tx.run(cypher, input_structs=[{
                'prompt_id': prompt.doc_id,
                'tool_call_id': tool_call.doc_id,
            }])
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    async def create_tool_call_to_prompt_edge(self, prompt: PromptDoc, tool_call: ToolCallDoc):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (p:Prompt {doc_id: input_struct.prompt_id})
        WITH input_struct, p
        MERGE (t:ToolCall {doc_id: input_struct.tool_call_id})
        SET t.labels = input_struct.tool_call_labels
        SET t.search_keywords = input_struct.tool_call_search_keywords
        SET t.tool_arguments = input_struct.tool_arguments
        SET t.tool_name = input_struct.tool_name
        WITH p, t
        MERGE (t)-[:CALLED_FOR]->(p)
        """
        async def _edge(tx):
            await tx.run(cypher, input_structs=[{
                'prompt_id': prompt.doc_id,
                'tool_arguments': tool_call.tool_arguments,
                'tool_call_id': tool_call.doc_id,
                'tool_call_labels': sorted(tool_call.labels),
                'tool_call_search_keywords': tool_call.search_keywords,
                'tool_name': tool_call.tool_name,
            }])
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    async def create_topic_to_prompt_edge(self, prompt: PromptDoc, topics: list[str]):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (p:Prompt {doc_id: input_struct.prompt_id})
        WITH input_struct, p
        MERGE (t:Topic {text: input_struct.topic})
        WITH p, t
        MERGE (t)-[:TOPIC_OF]->(p)
        """
        async def _edge(tx):
            deduped_topics = self.dedupe_identity_texts(topics)
            if not deduped_topics:
                return
            input_structs = [{
                'prompt_id': prompt.doc_id,
                'topic': topic,
            } for topic in deduped_topics]
            await tx.run(cypher, input_structs=input_structs)
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    @classmethod
    def dedupe_identity_texts(cls, values: list[str]) -> list[str]:
        normalized_values = []
        seen_values = set()
        for value in values:
            value = normalize_text_for_identity(value)
            if not value or value in seen_values:
                continue
            seen_values.add(value)
            normalized_values.append(value)
        return normalized_values

    def init_client(self):
        self.driver = get_neo4j_driver()


def get_neo4j_driver() -> AsyncDriver:
    auth = os.environ.get("NEO4J_AUTH")
    uri = os.environ.get("NEO4J_URI")
    if auth and uri:
        auth = auth.split("/")
        if len(auth) == 2:
            return AsyncGraphDatabase.driver(uri, auth=(auth[0], auth[1]))
    raise RuntimeError("Unable to initialize Neo4j driver")
