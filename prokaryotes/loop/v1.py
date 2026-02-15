import asyncio
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class Loop:
    def __init__(self, async_openai: AsyncOpenAI):
        self.async_openai = async_openai

    async def run(self):
        logger.info("Starting loop")
        try:
            while True:
                logger.info("tick")
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("Stopping loop")
