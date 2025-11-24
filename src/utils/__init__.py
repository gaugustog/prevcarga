"""Utility functions and helpers.

This module provides common utilities used across the PrevCarga system,
including structured logging and configuration helpers.

Example:
    ```python
    from src.utils import get_logger, LogContext

    logger = get_logger(__name__)
    logger.info("Starting process")

    with LogContext(logger, request_id="abc123"):
        logger.info("Processing request")
    ```
"""

from src.utils.logger import (
    LogContext,
    configure_module_logger,
    get_logger,
    log_context,
    setup_logging,
)

__all__ = [
    "LogContext",
    "configure_module_logger",
    "get_logger",
    "log_context",
    "setup_logging",
]
