import logging.config
import os

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
