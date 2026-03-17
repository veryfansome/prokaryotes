import asyncio
import concurrent.futures
import logging.config
import os

logger = logging.getLogger(__name__)

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

def setup_logging(
        httpcore_level: str = None,
        httpx_level: str = None,
        imapclient_level: str = None,
        openai_level: str = None,
        prokaryotes_search_v1_level: str = None,
        prokaryotes_tools_v1_level: str = None,
        root_level: str = None,
):
    if not httpcore_level:
        httpcore_level = os.getenv("HTTPCORE_LOG_LEVEL", "INFO")
    if not httpx_level:
        httpx_level = os.getenv("HTTPX_LOG_LEVEL", "INFO")
    if not imapclient_level:
        imapclient_level = os.getenv("IMAPCLIENT_LOG_LEVEL", "INFO")
    if not openai_level:
        openai_level = os.getenv("OPENAI_LOG_LEVEL", "INFO")
    if not prokaryotes_search_v1_level:
        prokaryotes_search_v1_level = os.getenv("PROKARYOTES_SEARCH_V1_LOG_LEVEL", "DEBUG")
    if not prokaryotes_tools_v1_level:
        prokaryotes_tools_v1_level = os.getenv("PROKARYOTES_TOOLS_V1_LOG_LEVEL", "DEBUG")
    if not root_level:
        root_level = os.getenv("LOG_LEVEL", "INFO")

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
            "prokaryotes.search_v1": {
                "level": prokaryotes_search_v1_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "prokaryotes.tools_v1": {
                "level": prokaryotes_tools_v1_level,
                "handlers": ["console"],
                "propagate": False,
            },
        }
    })
