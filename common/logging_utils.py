"""Logging helpers for TimesFM fine-tuning pipelines."""
from __future__ import annotations

import logging
import logging.handlers
import pathlib
from typing import Optional

_LOGGERS = {}


def setup_logging(name: str, log_dir: pathlib.Path, level: str = "INFO") -> logging.Logger:
    """Create or retrieve a structured logger.

    Parameters
    ----------
    name: str
        Logger name.
    log_dir: pathlib.Path
        Directory where rotating log files will be written.
    level: str
        Logging level (INFO, DEBUG, etc.).
    """

    global _LOGGERS

    if name in _LOGGERS:
        return _LOGGERS[name]

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"{name}.log", maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str, log_dir: Optional[pathlib.Path] = None, level: str = "INFO") -> logging.Logger:
    """Convenience wrapper that defaults to a tmp log directory when omitted."""

    if log_dir is None:
        log_dir = pathlib.Path("logs")
    return setup_logging(name, log_dir, level=level)


__all__ = ["setup_logging", "get_logger"]
