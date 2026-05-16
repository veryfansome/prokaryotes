import hashlib
import unicodedata
from functools import lru_cache

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
    " ": " ",  # NO-BREAK SPACE
    "­": None,  # SOFT HYPHEN
    " ": " ",  # FIGURE SPACE
    "​": None,  # ZERO WIDTH SPACE
    "‌": None,  # ZERO WIDTH NON-JOINER
    "‍": None,  # ZERO WIDTH JOINER
    " ": " ",  # NARROW NO-BREAK SPACE
    "⁠": None,  # WORD JOINER
    "﻿": None,  # ZERO WIDTH NO-BREAK SPACE (BOM)
})


@lru_cache(maxsize=128)
def normalize_text_for_identity(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_IDENTITY_PUNCT_TRANSLATION)
    return " ".join(text.split()).strip()


@lru_cache(maxsize=128)
def text_to_md5(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode("utf-8")).hexdigest()
