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

def setup_logging():
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
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "handlers": ["console"],
            },
            "httpcore": {
                "level": os.getenv("HTTPCORE_LOG_LEVEL", "INFO"),
                "handlers": ["console"],
                "propagate": False,
            },
            "imapclient": {
                "level": os.getenv("IMAPCLIENT_LOG_LEVEL", "INFO"),
                "handlers": ["console"],
                "propagate": False,
            },
            "openai": {
                "level": os.getenv("OPENAI_LOG_LEVEL", "INFO"),
                "handlers": ["console"],
                "propagate": False,
            },
        }
    })
