import difflib
import hashlib
import os

import mistune
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


def str_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def str_similarity_batch(a: str, b_list: list[str]) -> list[float]:
    results = []
    for b in b_list:
        results.append(str_similarity(a, b))
    return results


def strip_punctuation(token: str) -> str:
    return token.strip(",.;:()[]{}'\"")


def text_to_md5(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode("utf-8")).hexdigest()


def text_to_md5_batch(texts: list[str]) -> list[str]:
    return [text_to_md5(text) for text in texts]
