import asyncio
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


def setup_logging():
    httpcore_level = os.getenv("HTTPCORE_LOG_LEVEL", "INFO")
    httpx_level = os.getenv("HTTPX_LOG_LEVEL", "INFO")
    openai_level = os.getenv("OPENAI_LOG_LEVEL", "INFO")
    prokaryotes_search_v1_level = os.getenv("PROKARYOTES_SEARCH_V1_LOG_LEVEL", "DEBUG")
    prokaryotes_tools_v1_level = os.getenv("PROKARYOTES_TOOLS_V1_LOG_LEVEL", "DEBUG")
    root_level = os.getenv("LOG_LEVEL", "INFO")

    logging.config.dictConfig(
        {
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
            },
        }
    )
