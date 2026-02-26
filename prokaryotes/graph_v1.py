import logging
import os
from neo4j import AsyncDriver, AsyncGraphDatabase

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
