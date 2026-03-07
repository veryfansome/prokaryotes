import logging
import os
from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
)

from prokaryotes.models_v1 import (
    ChatCompletionDoc,
    FactDoc,
)

logger = logging.getLogger(__name__)

def get_neo4j_driver() -> AsyncDriver:
    neo4j_auth = os.environ.get("NEO4J_AUTH")
    neo4j_uri = os.environ.get("NEO4J_URI")
    if neo4j_auth and neo4j_uri:
        neo4j_auth = neo4j_auth.split("/")
        if len(neo4j_auth) == 2:
            return AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_auth[0], neo4j_auth[1]))
    raise RuntimeError("Unable to initialize Neo4j driver")

class GraphClient:
    def __init__(self, driver: AsyncDriver = get_neo4j_driver()):
        self.driver = driver

    async def close(self):
        await self.driver.close()

    async def create_fact_to_completion_edge(self, completion: ChatCompletionDoc, facts: list[FactDoc]):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (c:ChatCompletion {doc_id: input_struct.completion_id})
        WITH input_struct, c
        MERGE (f:FactDoc {doc_id: input_struct.fact_id})
        SET f.text = input_struct.fact_text
        WITH c, f
        MERGE (f)-[:LEARNED_FROM]->(c)
        """
        async def _edge(tx):
            input_structs = [{
                'completion_id': completion.doc_id,
                'fact_id': fact.doc_id,
                'fact_text': fact.text,
            } for fact in facts]
            await tx.run(cypher, input_structs=input_structs)
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")

    async def create_topic_to_completion_edge(self, completion: ChatCompletionDoc, topics: list[str]):
        cypher = """
        UNWIND $input_structs AS input_struct
        MERGE (c:ChatCompletion {doc_id: input_struct.completion_id})
        WITH input_struct, c
        MERGE (t:Topic {text: input_struct.topic})
        WITH c, t
        MERGE (t)-[:TOPIC_OF]->(c)
        """
        async def _edge(tx):
            input_structs = [{
                'completion_id': completion.doc_id,
                'topic': topic,
            } for topic in topics]
            await tx.run(cypher, input_structs=input_structs)
        try:
            async with self.driver.session() as session:
                return await session.execute_write(_edge)
        except Exception:
            logger.exception(f"Failed to execute: {cypher}")
