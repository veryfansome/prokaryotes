import pathlib

import prokaryotes

# Allow unchanged sibling modules (eval.py, script.py, web.py — though web.py is
# overlay-overridden here for the new wire protocol) to fall through to upstream.
_HERE = pathlib.Path(__file__).resolve().parent
for _parent_path in prokaryotes.__path__:
    _candidate = pathlib.Path(_parent_path).resolve() / "harness_v1"
    if _candidate != _HERE and _candidate.is_dir() and str(_candidate) not in __path__:
        __path__.append(str(_candidate))

from prokaryotes.api_v1.models import LLMClient  # noqa: E402
from prokaryotes.harness_v1.base import HarnessBase  # noqa: E402


def build_llm_client(impl: str) -> tuple[LLMClient, str]:
    """Return `(uninitialized LLMClient, instruction_role)` for the given provider impl."""
    if impl == "anthropic":
        from prokaryotes.anthropic_v1 import AnthropicClient

        return AnthropicClient(), "system"
    if impl == "openai":
        from prokaryotes.openai_v1 import OpenAIClient

        return OpenAIClient(), "developer"
    raise ValueError(f"Unsupported impl: {impl!r}")


__all__ = ["HarnessBase", "build_llm_client"]
