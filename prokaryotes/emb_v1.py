import logging
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

from prokaryotes.models_v1 import (
    TextEmbeddingPrompt,
    TextEmbeddingRequest,
    TextEmbeddingResponse,
)

logger = logging.getLogger(__name__)


class EmbeddingV1:
    def __init__(self, emb_model_name: str):
        self.emb_model: SentenceTransformer | None = None
        self.emb_model_name = emb_model_name

        self.app = FastAPI(lifespan=self.lifespan)
        self.app.add_api_route("/embs", self.embs, methods=["POST"])

    def embs(self, req: TextEmbeddingRequest):
        if not req.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        if self.emb_model is None:
            raise HTTPException(status_code=503, detail="Not available")
        embs = self.emb_model.encode(
            req.texts,
            batch_size=req.batch_size,
            normalize_embeddings=True,
            prompt_name="query" if req.prompt == TextEmbeddingPrompt.QUERY else None,
            show_progress_bar=False,
        )
        if req.truncate_to:
            # Apply Matryoshka Truncation
            trunc = embs[:, :req.truncate_to]
            # Re-normalize truncated vectors
            norms = np.linalg.norm(trunc, axis=1, keepdims=True)
            return TextEmbeddingResponse(embs=(trunc / np.where(norms > 0, norms, 1.0)).tolist())
        else:
            return TextEmbeddingResponse(embs=embs.tolist())

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        logger.info("Initializing models")
        await run_in_threadpool(self.load_emb_model)
        await run_in_threadpool(self.embs, TextEmbeddingRequest(
            prompt=TextEmbeddingPrompt.QUERY,
            texts=["Hello, world!"],
        ))
        yield

    def load_emb_model(self):
        self.emb_model = SentenceTransformer(self.emb_model_name)
