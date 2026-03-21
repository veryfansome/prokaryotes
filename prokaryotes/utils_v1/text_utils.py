import mistune
import os
from bs4 import BeautifulSoup
from starlette.concurrency import run_in_threadpool

from prokaryotes.models_v1 import (
    TextEmbeddingPrompt,
    TextEmbeddingRequest,
    TextEmbeddingResponse,
)
from prokaryotes.utils_v1 import http_utils

async def get_text_embeddings(req: TextEmbeddingRequest, timeout: float = 10.0) -> TextEmbeddingResponse:
    resp = await http_utils.httpx_client.post(
        os.getenv("EMBEDDINGS_URL"),
        json=req.model_dump(mode="json"),
        timeout=timeout
    )
    resp.raise_for_status()
    return TextEmbeddingResponse.model_validate(resp.json())

def normalize_text_for_search(text: str) -> str:
    text = mistune.html(text.lower())
    soup = BeautifulSoup(text, "lxml")
    for code_tag in soup.select('pre code'):
        code_tag.decompose()
    return soup.get_text().strip()

async def normalize_text_for_search_and_embed(text: str) -> tuple[str, list[float]]:
    normalized_text = await run_in_threadpool(normalize_text_for_search, text)
    emb = (await get_text_embeddings(TextEmbeddingRequest(
        prompt=TextEmbeddingPrompt.QUERY,
        texts=[normalized_text],
        truncate_to=256,
    ))).embs[0]
    return normalized_text, emb
