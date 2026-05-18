from prokaryotes.api_v1.models import LLMClient
from prokaryotes.harness_v1.base import HarnessBase


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
