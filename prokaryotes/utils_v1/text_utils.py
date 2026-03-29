import mistune
import os
from bs4 import BeautifulSoup

from prokaryotes.models_v1 import (
    TextEmbeddingPrompt,
    TextEmbeddingRequest,
    TextEmbeddingResponse,
)
from prokaryotes.utils_v1 import http_utils


async def get_document_embs(doc_texts: list[str], batch_size: int = 1) -> list[list[float]]:
    return (await get_text_embs(TextEmbeddingRequest(
        batch_size=batch_size,
        prompt=TextEmbeddingPrompt.DOCUMENT,
        texts=doc_texts,
        truncate_to=256,
    ))).embs


async def get_query_embs(qry_texts: list[str], batch_size: int = 1) -> list[list[float]]:
    return (await get_text_embs(TextEmbeddingRequest(
        batch_size=batch_size,
        prompt=TextEmbeddingPrompt.QUERY,
        texts=qry_texts,
        truncate_to=256,
    ))).embs


async def get_text_embs(req: TextEmbeddingRequest, timeout: float = 10.0) -> TextEmbeddingResponse:
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
