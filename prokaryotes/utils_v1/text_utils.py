import difflib
import hashlib
import os
import unicodedata
from functools import lru_cache

import mistune
from async_lru import alru_cache
from bs4 import BeautifulSoup

from prokaryotes.api_v1.models import (
    TextEmbeddingPrompt,
    TextEmbeddingRequest,
    TextEmbeddingResponse,
)
from prokaryotes.utils_v1 import http_utils

_IDENTITY_PUNCT_TRANSLATION = str.maketrans({
    "ʼ": "'",
    "‘": "'",
    "’": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "′": "'",
    "-": "-",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    "\u00A0": " ",  # NO-BREAK SPACE
    "\u00AD": None,  # SOFT HYPHEN
    "\u2007": " ",  # FIGURE SPACE
    "\u200B": None,  # ZERO WIDTH SPACE
    "\u200C": None,  # ZERO WIDTH NON-JOINER
    "\u200D": None,  # ZERO WIDTH JOINER
    "\u202F": " ",  # NARROW NO-BREAK SPACE
    "\u2060": None,  # WORD JOINER
    "\uFEFF": None,  # ZERO WIDTH NO-BREAK SPACE (BOM)
})


@alru_cache(maxsize=128, ttl=60)
async def get_document_embs(doc_texts: tuple[str, ...], batch_size: int = 1) -> list[list[float]]:
    return (await get_text_embs(TextEmbeddingRequest(
        batch_size=batch_size,
        prompt=TextEmbeddingPrompt.DOCUMENT,
        texts=doc_texts,
        truncate_to=256,
    ))).embs


@alru_cache(maxsize=128, ttl=60)
async def get_query_embs(qry_texts: tuple[str, ...], batch_size: int = 1) -> list[list[float]]:
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


@lru_cache(maxsize=128)
def normalize_text_for_identity(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_IDENTITY_PUNCT_TRANSLATION)
    return " ".join(text.split()).strip()


@lru_cache(maxsize=128)
def normalize_text_for_search(text: str) -> str:
    text = mistune.html(text.lower())
    soup = BeautifulSoup(text, "lxml")
    for code_tag in soup.select('pre code'):
        code_tag.decompose()
    return soup.get_text().strip()


@lru_cache(maxsize=128)
def str_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def str_similarity_batch(a: str, b_list: list[str]) -> list[float]:
    results = []
    for b in b_list:
        results.append(str_similarity(a, b))
    return results


@lru_cache(maxsize=128)
def strip_punctuation(token: str) -> str:
    return token.strip(",.!?;:()[]{}'\"")


@lru_cache(maxsize=128)
def text_to_md5(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode("utf-8")).hexdigest()


def text_to_md5_batch(texts: list[str]) -> list[str]:
    return [text_to_md5(text) for text in texts]
