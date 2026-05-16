from prokaryotes.api_v1.models import LLMClient


def build_llm_client(impl: str) -> tuple[LLMClient, str]:
    """Return (uninitialized LLMClient, instruction_role) for the given provider impl.

    Callers decide when to call `client.init_client()`. `ScriptHarness` does it eagerly
    in `__init__`; `WebHarness` defers to its synchronous `init()` setup phase to match
    the FastAPI lifecycle.
    """
    if impl == "anthropic":
        from prokaryotes.anthropic_v1 import AnthropicClient
        return AnthropicClient(), "system"
    if impl == "openai":
        from prokaryotes.openai_v1 import OpenAIClient
        return OpenAIClient(), "developer"
    raise ValueError(f"Unsupported impl: {impl!r}")
