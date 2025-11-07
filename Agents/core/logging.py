"""Centralised logging utilities for the Agentic multi-agent system."""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional


def configure_logging(
    *,
    level: int | str | None = None,
    fmt: Optional[str] = None,
    extra_handlers: Optional[Iterable[logging.Handler]] = None,
) -> None:
    """Configure application-wide logging in a production-friendly manner.

    This function keeps defaults conservative and uses environment variables
    to allow runtime overrides without code changes.
    """

    log_level = _resolve_level(level)
    log_format = fmt or os.getenv(
        "AGENTIC_LOG_FORMAT",
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    logging.basicConfig(level=log_level, format=log_format)

    if extra_handlers:
        root_logger = logging.getLogger()
        for handler in extra_handlers:
            root_logger.addHandler(handler)


def _resolve_level(level: int | str | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return logging.getLevelName(level.upper())  # type: ignore[return-value]

    env_level = os.getenv("AGENTIC_LOG_LEVEL")
    if env_level:
        return logging.getLevelName(env_level.upper())  # type: ignore[return-value]

    return logging.INFO


__all__ = ["configure_logging"]


