import logging
import os

from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
)

from prokaryotes.utils_v1.text_utils import normalize_text_for_identity

logger = logging.getLogger(__name__)


class GraphClient:
    def __init__(self):
        self.driver: AsyncDriver | None = None

    async def close(self):
        await self.driver.close()

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
