import asyncio
import concurrent.futures
import logging.config
import mistune
import os
from bs4 import BeautifulSoup

from prokaryotes.models_v1 import (
    PersonContext,
    RequestContext,
)

logger = logging.getLogger(__name__)

def developer_message_parts(request_context: RequestContext, user_context: PersonContext) -> list[str]:
    now = request_context.received_at.astimezone(tz=request_context.time_zone).strftime('%Y-%m-%d %H:%M')
    message_parts = [
        "## Execution context",
        f"Time: {now} {request_context.time_zone}",
        (
            f"Environment: Python-{request_context.execution_context.python_version}"
            f" / {request_context.execution_context.platform_short}"
        ),
        f"Working directory: {request_context.execution_context.cwd}",
        "---",
        "## User info",
    ]
    if request_context.latitude and request_context.longitude:
        message_parts.append(
            f"- {now}: The user is at"
            f" *lat: {request_context.latitude:.4f}, long: {request_context.longitude:.4f}*"
        )
    if user_context.facts:
        # TODO: Optional trimming if fact lists grow long
        for fact_doc in user_context.facts:
            message_parts.append(
                f"- {fact_doc.created_at.astimezone(request_context.time_zone).strftime('%Y-%m-%d %H:%M')}"
                f": {fact_doc.text}"
            )

    message_parts.append("---")
    message_parts.append("## Assistant info")
    message_parts.append(f"- {now}: The assistant is a Python app")
    return message_parts

def log_async_task_exception(task: asyncio.Task):
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Exception in task")

def log_future_exception(future: concurrent.futures.Future):
    exception = future.exception()
    if exception:
        logger.exception("Exception in thread", exc_info=exception)

def prep_chat_message_text_for_search(text: str) -> str:
    text = mistune.html(text.lower())
    soup = BeautifulSoup(text, "lxml")
    for code_tag in soup.select('pre code'):
        code_tag.decompose()
    return soup.get_text()

def setup_logging(
        httpcore_level: str = os.getenv("HTTPCORE_LOG_LEVEL", "INFO"),
        httpx_level: str = os.getenv("HTTPX_LOG_LEVEL", "INFO"),
        imapclient_level: str = os.getenv("IMAPCLIENT_LOG_LEVEL", "INFO"),
        openai_level: str = os.getenv("OPENAI_LOG_LEVEL", "INFO"),
        root_level: str = os.getenv("LOG_LEVEL", "INFO"),
):
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(levelname)s:     %(asctime)s - %(name)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "root": {
                "level": root_level,
                "handlers": ["console"],
            },
            "httpcore": {
                "level": httpcore_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "httpx": {
                "level": httpx_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "imapclient": {
                "level": imapclient_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "openai": {
                "level": openai_level,
                "handlers": ["console"],
                "propagate": False,
            },
        }
    })
