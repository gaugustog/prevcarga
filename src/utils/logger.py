"""Structured logging utility module.

This module provides a centralized logging configuration for the PrevCarga system,
supporting both human-readable console output and JSON-formatted file logs.

Example:
    ```python
    from src.utils.logger import get_logger, LogContext

    logger = get_logger(__name__)
    logger.info("Application started")

    # With context
    with LogContext(logger, user_id="123", request_id="abc"):
        logger.info("Processing request")
    ```
"""

import logging
import logging.config
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import yaml


_logging_initialized = False


def setup_logging(
    config_path: str = "config/logging.yaml",
    log_level: Optional[str] = None,
) -> None:
    """Set up logging configuration from YAML file.

    Args:
        config_path: Path to logging configuration YAML file.
        log_level: Optional log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Note:
        The logs directory is automatically created if it doesn't exist.
        If the config file is not found, falls back to basic console logging.
    """
    global _logging_initialized

    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    config_file = Path(config_path)

    if config_file.exists():
        with config_file.open() as f:
            config = yaml.safe_load(f)

        # Apply log level override if provided
        if log_level:
            level_upper = log_level.upper()
            if "loggers" in config:
                for logger_config in config["loggers"].values():
                    logger_config["level"] = level_upper
            if "root" in config:
                config["root"]["level"] = level_upper

        logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration
        level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.warning("Logging config file not found: %s. Using basic config.", config_path)

    _logging_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    This function ensures logging is configured before returning a logger.
    It supports lazy initialization and respects the LOG_LEVEL environment variable.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.

    Example:
        ```python
        from src.utils.logger import get_logger

        logger = get_logger(__name__)
        logger.info("Processing started")
        logger.debug("Debug info: %s", data)
        logger.error("Error occurred", exc_info=True)
        ```
    """
    global _logging_initialized

    if not _logging_initialized:
        log_level = os.getenv("LOG_LEVEL")
        setup_logging(log_level=log_level)

    return logging.getLogger(name)


class LogContext:
    """Context manager for adding structured context to logs.

    This allows adding extra fields to all log messages within a block,
    useful for tracking request IDs, user IDs, or other correlation data.

    Example:
        ```python
        logger = get_logger(__name__)

        with LogContext(logger, request_id="abc123", user_id="user456"):
            logger.info("Processing request")  # Will include request_id and user_id
            logger.info("Request complete")
        ```
    """

    def __init__(self, logger: logging.Logger, **context: Any) -> None:
        """Initialize log context.

        Args:
            logger: Logger instance to add context to.
            **context: Key-value pairs to add to log records.
        """
        self.logger = logger
        self.context = context
        self._old_factory: Optional[Any] = None

    def __enter__(self) -> logging.Logger:
        """Enter context and add context fields to logger."""
        old_factory = logging.getLogRecordFactory()

        def record_factory(
            *args: Any,
            **kwargs: Any,
        ) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        self._old_factory = old_factory
        logging.setLogRecordFactory(record_factory)
        return self.logger

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit context and restore original factory."""
        if self._old_factory is not None:
            logging.setLogRecordFactory(self._old_factory)


@contextmanager
def log_context(logger: logging.Logger, **context: Any) -> Generator[logging.Logger, None, None]:
    """Context manager for adding structured context to logs.

    This is a functional alternative to the LogContext class.

    Args:
        logger: Logger instance to add context to.
        **context: Key-value pairs to add to log records.

    Yields:
        The logger with context applied.

    Example:
        ```python
        logger = get_logger(__name__)

        with log_context(logger, request_id="abc123"):
            logger.info("Processing request")
        ```
    """
    with LogContext(logger, **context) as ctx_logger:
        yield ctx_logger


def configure_module_logger(
    module_name: str,
    level: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger for a specific module.

    This is useful for setting up loggers with specific levels
    different from the default configuration.

    Args:
        module_name: Name of the module (typically __name__).
        level: Log level for this module (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logger instance.
    """
    logger = get_logger(module_name)

    if level:
        log_level = getattr(logging, level.upper(), logging.DEBUG)
        logger.setLevel(log_level)

    return logger
