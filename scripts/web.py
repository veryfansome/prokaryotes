import os

from dotenv import load_dotenv

from prokaryotes.utils_v1.logging_utils import setup_logging

load_dotenv()
setup_logging()

harness_impl = os.getenv("WEB_HARNESS_IMPL", "anthropic")
match harness_impl:
    case "anthropic":
        from prokaryotes.anthropic_v1.web_harness import WebHarness
    case "openai":
        from prokaryotes.openai_v1.web_harness import WebHarness
    case _:
        raise NotImplementedError(f"Unsupported WEB_HARNESS_IMPL value '{harness_impl}'")

harness = WebHarness("scripts/static")
harness.init()
