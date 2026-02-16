import asyncio
import logging

from prokaryotes.imap_v1 import EmailReader, get_email_reader
from prokaryotes.llm_v1 import LLM, get_llm

logger = logging.getLogger(__name__)

class AgentLoop:
    def __init__(self):
        self.email_reader: EmailReader = get_email_reader()
        self.llm: LLM = get_llm()

    async def run(self):
        logger.info("Starting loop")
        try:
            while True:
                logger.info("tick")
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("Stopping loop")
