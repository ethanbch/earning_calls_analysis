from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .utils import ensure_dir


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "event"):
            record.event = "-"
        return True


def configure_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger("koyfin_fast")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(event)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(ContextFilter())

    file_handler = RotatingFileHandler(
        log_dir / "koyfin_fast.log",
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.addFilter(ContextFilter())

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
