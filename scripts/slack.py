import asyncio
import logging
import os
import signal
import sys

from dotenv import load_dotenv

from prokaryotes.harness_v1.slack import SlackHarness
from prokaryotes.utils_v1.logging_utils import setup_logging

logger = logging.getLogger(__name__)


async def run(app_token: str, bot_token: str) -> None:
    harness = SlackHarness(
        impl=os.getenv("SLACK_HARNESS_IMPL", "anthropic"),
        app_token=app_token,
        bot_token=bot_token,
    )
    await harness.on_start()
    # SIGTERM (docker compose stop / deploy) and SIGINT (Ctrl-C) set the event so the finally block runs
    # on_stop() — socket disconnect + in-flight turn drain. Without the handlers, SIGTERM terminates the process
    # before finally.
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        # Socket Mode runs in background tasks; block the foreground until a signal.
        await stop.wait()
    finally:
        await harness.on_stop()


if __name__ == "__main__":
    load_dotenv()
    setup_logging()
    # A single shared `slack-<workspace>` template in docker-compose.yml is copied per workspace and given that
    # workspace's tokens. An operator running the compose stack without populating Slack tokens (web-only setup)
    # would otherwise see this service crash-loop on a `KeyError`. Exit 0 with a clear log so the service stays
    # stopped instead.
    app_token = os.environ.get("SLACK_APP_TOKEN")
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    if not app_token or not bot_token:
        missing = [name for name, val in (("SLACK_APP_TOKEN", app_token), ("SLACK_BOT_TOKEN", bot_token)) if not val]
        logger.info("Slack harness not starting — missing env vars: %s", ", ".join(missing))
        sys.exit(0)
    asyncio.run(run(app_token, bot_token))
