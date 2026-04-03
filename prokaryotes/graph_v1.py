import logging
import os

from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
)

from prokaryotes.models_v1 import (
    FactDoc,
    PromptDoc,
    ResponseDoc,
    ToolCallDoc,
)

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

    async def create_tool_call_to_prompt_edge(self, prompt: PromptDoc, tool_call: ToolCallDoc):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (p:Prompt {doc_id: input_struct.prompt_id})
        WITH input_struct, p
        MERGE (t:ToolCall {doc_id: input_struct.tool_call_id})
        SET t.labels = input_struct.tool_call_labels
        SET t.tool_arguments = input_struct.tool_arguments
        SET t.tool_name = input_struct.tool_name
        WITH p, t
        MERGE (t)-[:CALLED_FOR]->(p)
        """
        async def _edge(tx):
            await tx.run(cypher, input_structs=[{
                'prompt_id': prompt.doc_id,
                'tool_call_id': tool_call.doc_id,
                'tool_call_labels': sorted(tool_call.labels),
                'tool_arguments': tool_call.tool_arguments,
                'tool_name': tool_call.tool_name,
            }])
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    async def create_tool_call_to_response_edge(self, response: ResponseDoc, tool_call: ToolCallDoc):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (r:Response {doc_id: input_struct.response_id})
        WITH input_struct, r
        MERGE (t:ToolCall {doc_id: input_struct.tool_call_id})
        SET t.labels = input_struct.tool_call_labels
        SET t.tool_arguments = input_struct.tool_arguments
        SET t.tool_name = input_struct.tool_name
        WITH r, t
        MERGE (t)-[:CONTEXT_FOR]->(r)
        """
        async def _edge(tx):
            await tx.run(cypher, input_structs=[{
                'response_id': response.doc_id,
                'tool_call_id': tool_call.doc_id,
                'tool_call_labels': sorted(tool_call.labels),
                'tool_arguments': tool_call.tool_arguments,
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
            input_structs = [{
                'prompt_id': prompt.doc_id,
                'topic': topic,
            } for topic in topics]
            await tx.run(cypher, input_structs=input_structs)
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    async def create_topic_to_response_edge(self, response: ResponseDoc, topics: list[str]):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (r:Response {doc_id: input_struct.response_id})
        WITH input_struct, r
        MERGE (t:Topic {text: input_struct.topic})
        WITH r, t
        MERGE (t)-[:TOPIC_OF]->(r)
        """
        async def _edge(tx):
            input_structs = [{
                'response_id': response.doc_id,
                'topic': topic,
            } for topic in topics]
            await tx.run(cypher, input_structs=input_structs)
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

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
