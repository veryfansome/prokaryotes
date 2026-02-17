import asyncio
import logging

from prokaryotes.imap_v1 import IMAPClientFactory, get_imap_client_factory
from prokaryotes.llm_v1 import LLM, get_llm

logger = logging.getLogger(__name__)

class AgentLoop:
    def __init__(self):
        self.imap_client_factory: IMAPClientFactory = get_imap_client_factory()
        self.llm: LLM = get_llm()

    async def run(self):
        logger.info("Starting loop")
        while True:
            try:
                logger.info("tick")
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                logger.info("Stopping loop")
                break
