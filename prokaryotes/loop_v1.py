import asyncio
import logging

from prokaryotes.llm_v1 import LLM

logger = logging.getLogger(__name__)

class AgentLoop:
    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(self):
        logger.info("Starting loop")
        try:
            while True:
                logger.info("tick")
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("Stopping loop")
