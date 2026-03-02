import asyncio
import logging
import numpy as np
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

from prokaryotes.models_v1 import (
    TextEmbeddingPrompt,
    TextEmbeddingRequest,
)
from prokaryotes.web_base import WebBase

logger = logging.getLogger(__name__)

class EmbeddingV1(WebBase):
    def __init__(self, model: str):
        self.encoder: SentenceTransformer | None = None
        self.max_encoding_threads: int = max(2, os.cpu_count() - 2)
        self.model = model

        self.app = FastAPI(lifespan=self.lifespan)
        self.app.add_api_route("/emb", self.emb, methods=["POST"])

    async def emb(self, payload: TextEmbeddingRequest):
        embs = []
        tasks = []
        texts_len = len(payload.texts)
        partition_len = max(1, texts_len // self.max_encoding_threads)
        for idx in range(0, texts_len, partition_len):
            stop_idx = idx + partition_len
            tasks.append(asyncio.create_task(run_in_threadpool(
                self.encoder.encode, payload.texts[idx:stop_idx],
                normalize_embeddings=True,
                prompt_name=payload.prompt.value,
                show_progress_bar=False,
            )))
        for batch_embs in await asyncio.gather(*tasks):
            if payload.truncate_to:
                # Apply Matryoshka Truncation
                trunc = batch_embs[:, :payload.truncate_to]
                # Re-normalize truncated vectors
                norms = np.linalg.norm(trunc, axis=1, keepdims=True)
                embs.extend((trunc / np.where(norms > 0, norms, 1.0)).tolist())
            else:
                embs.extend(batch_embs.tolist())
        return {"embeddings": embs}

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info("Initializing embedding model")
        self.encoder = SentenceTransformer(self.model)
        await self.emb(TextEmbeddingRequest(texts=["Hello, world!"], prompt=TextEmbeddingPrompt.DOCUMENT))
        yield
