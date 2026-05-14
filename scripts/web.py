import os

from dotenv import load_dotenv

from prokaryotes.harness_v1.web import WebHarness
from prokaryotes.utils_v1.logging_utils import setup_logging

load_dotenv()
setup_logging()

harness = WebHarness(
    impl=os.getenv("WEB_HARNESS_IMPL", "anthropic"),
    static_dir="scripts/static",
)
harness.init()
