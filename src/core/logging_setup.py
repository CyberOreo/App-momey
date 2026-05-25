"""Structured logging via loguru with JSON support and file rotation."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_logs: bool = False,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    logger.remove()

    fmt_console = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    fmt_json = "{message}"

    if json_logs:
        logger.add(
            sys.stdout,
            level=level,
            serialize=True,
        )
    else:
        logger.add(sys.stdout, level=level, format=fmt_console, colorize=True)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            rotation=rotation,
            retention=retention,
            serialize=True,
            enqueue=True,          # thread-safe async-friendly writes
            backtrace=True,
            diagnose=False,        # disable in prod (leaks local vars)
        )

    logger.info("Logging initialised", level=level, file=log_file)


def get_logger(name: str):
    return logger.bind(module=name)
