"""
Logging configuration — JSON-structured logs for production,
human-readable for development.
"""

import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    """Configure root logger with appropriate handlers and formatters."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    if settings.DEBUG:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s"
        datefmt = "%H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt=datefmt)
    else:
        # Structured format parseable by log aggregators (Datadog, Loki, etc.)
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        formatter = logging.Formatter(fmt)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "PIL", "paddleocr", "ppocr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
