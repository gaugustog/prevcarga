"""Structured logger for PrevCarga workflow orchestration.

This module provides an enhanced structured logging system designed for
workflow orchestration, with support for JSON output, correlation IDs,
performance tracking, and log aggregation.

Key Components:
- StructuredLogger: Enhanced logger with structured output
- WorkflowLogContext: Context manager for workflow-scoped logging
- LogAggregator: Aggregates and summarizes log entries
- PerformanceLogger: Tracks and logs performance metrics

Example:
    ```python
    from src.orchestration.structured_logger import (
        StructuredLogger,
        WorkflowLogContext,
    )

    # Create structured logger
    logger = StructuredLogger.create("training_workflow")

    # Log with structured fields
    logger.info(
        "Training started",
        area="SECO",
        horizon=1,
        n_samples=1000,
    )

    # Use workflow context
    with WorkflowLogContext(workflow_id="wf-123", stage="training"):
        logger.info("Processing batch", batch_id=1)
    ```
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator

from src.utils.logger import get_logger


class LogLevel(Enum):
    """Log levels for structured logging.

    Attributes:
        DEBUG: Debug level logging.
        INFO: Info level logging.
        WARNING: Warning level logging.
        ERROR: Error level logging.
        CRITICAL: Critical level logging.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Output format for structured logs.

    Attributes:
        TEXT: Human-readable text format.
        JSON: Machine-readable JSON format.
        JSONL: JSON Lines format (one JSON object per line).
    """

    TEXT = "text"
    JSON = "json"
    JSONL = "jsonl"


@dataclass
class LogEntry:
    """Structured log entry.

    Attributes:
        timestamp: When the log entry was created.
        level: Log level.
        message: Log message.
        logger_name: Name of the logger.
        correlation_id: Correlation ID for tracking.
        workflow_id: Associated workflow ID.
        stage: Current workflow stage.
        extra: Additional structured fields.
        exception: Exception information if present.
        duration_ms: Duration in milliseconds for timed operations.
    """

    timestamp: datetime
    level: str
    message: str
    logger_name: str
    correlation_id: str | None = None
    workflow_id: str | None = None
    stage: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    exception: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "logger_name": self.logger_name,
        }

        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.workflow_id:
            result["workflow_id"] = self.workflow_id
        if self.stage:
            result["stage"] = self.stage
        if self.extra:
            result["extra"] = self.extra
        if self.exception:
            result["exception"] = self.exception
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms

        return result

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict())

    def to_text(self) -> str:
        """Convert to human-readable text.

        Returns:
            Text representation.
        """
        parts = [
            self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"[{self.level}]",
            f"[{self.logger_name}]",
        ]

        if self.workflow_id:
            parts.append(f"[wf:{self.workflow_id[:8]}]")

        if self.stage:
            parts.append(f"[{self.stage}]")

        parts.append(self.message)

        if self.duration_ms is not None:
            parts.append(f"({self.duration_ms:.2f}ms)")

        if self.extra:
            extra_str = " ".join(f"{k}={v}" for k, v in self.extra.items())
            parts.append(f"| {extra_str}")

        if self.exception:
            parts.append(f"\n{self.exception}")

        return " ".join(parts)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log output."""

    def __init__(
        self,
        fmt: LogFormat = LogFormat.TEXT,
        include_timestamp: bool = True,
    ) -> None:
        """Initialize formatter.

        Args:
            fmt: Output format (text, json, jsonl).
            include_timestamp: Whether to include timestamp.
        """
        super().__init__()
        self.fmt = fmt
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        """Format log record.

        Args:
            record: Log record to format.

        Returns:
            Formatted string.
        """
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created),
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            correlation_id=getattr(record, "correlation_id", None),
            workflow_id=getattr(record, "workflow_id", None),
            stage=getattr(record, "stage", None),
            extra=getattr(record, "structured_extra", {}),
            exception=self.formatException(record.exc_info) if record.exc_info else None,
            duration_ms=getattr(record, "duration_ms", None),
        )

        if self.fmt == LogFormat.JSON:
            return entry.to_json()
        elif self.fmt == LogFormat.JSONL:
            return entry.to_json()
        else:
            return entry.to_text()


class StructuredLogger:
    """Enhanced logger with structured output support.

    Provides structured logging with support for:
    - JSON and text output formats
    - Correlation IDs for request tracing
    - Workflow context tracking
    - Performance timing
    - Log aggregation

    Example:
        >>> logger = StructuredLogger.create("my_workflow")
        >>> logger.info("Task started", task_id="123", area="SECO")
        >>> with logger.timed("process_data"):
        ...     # do work
        ...     pass
    """

    # Thread-local context storage
    _context: dict[str, Any] = {}

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        fmt: LogFormat = LogFormat.TEXT,
        correlation_id: str | None = None,
    ) -> None:
        """Initialize structured logger.

        Args:
            name: Logger name.
            level: Log level.
            fmt: Output format.
            correlation_id: Optional correlation ID for tracing.
        """
        self._name = name
        self._level = level
        self._fmt = fmt
        self._correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self._logger = get_logger(name)
        self._entries: list[LogEntry] = []
        self._timers: dict[str, float] = {}

        # Configure handler with structured formatter
        self._configure_handler()

    def _configure_handler(self) -> None:
        """Configure log handler with structured formatter."""
        # Check if we already have a structured handler
        for handler in self._logger.handlers:
            if isinstance(handler.formatter, StructuredFormatter):
                return

        # Add console handler with structured formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter(fmt=self._fmt))
        handler.setLevel(getattr(logging, self._level.value))

        # Only add if no handlers exist to avoid duplicates
        if not self._logger.handlers:
            self._logger.addHandler(handler)

    @classmethod
    def create(
        cls,
        name: str,
        level: str | LogLevel = LogLevel.INFO,
        fmt: str | LogFormat = LogFormat.TEXT,
    ) -> "StructuredLogger":
        """Factory method to create a structured logger.

        Args:
            name: Logger name.
            level: Log level (string or LogLevel enum).
            fmt: Output format (string or LogFormat enum).

        Returns:
            Configured StructuredLogger instance.
        """
        if isinstance(level, str):
            level = LogLevel(level.upper())
        if isinstance(fmt, str):
            fmt = LogFormat(fmt.lower())

        return cls(name=name, level=level, fmt=fmt)

    @property
    def correlation_id(self) -> str:
        """Get current correlation ID."""
        return self._correlation_id

    @property
    def name(self) -> str:
        """Get logger name."""
        return self._name

    def _log(
        self,
        level: int,
        message: str,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        """Internal logging method.

        Args:
            level: Logging level.
            message: Log message.
            exc_info: Whether to include exception info.
            **kwargs: Additional structured fields.
        """
        # Merge context with kwargs
        extra = {**self._context, **kwargs}

        # Create enhanced record
        record_extra = {
            "correlation_id": self._correlation_id,
            "workflow_id": extra.pop("workflow_id", self._context.get("workflow_id")),
            "stage": extra.pop("stage", self._context.get("stage")),
            "duration_ms": extra.pop("duration_ms", None),
            "structured_extra": extra,
        }

        self._logger.log(
            level,
            message,
            exc_info=exc_info,
            extra=record_extra,
        )

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            message: Log message.
            **kwargs: Additional structured fields.
        """
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message.

        Args:
            message: Log message.
            **kwargs: Additional structured fields.
        """
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            message: Log message.
            **kwargs: Additional structured fields.
        """
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message.

        Args:
            message: Log message.
            exc_info: Whether to include exception info.
            **kwargs: Additional structured fields.
        """
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log critical message.

        Args:
            message: Log message.
            exc_info: Whether to include exception info.
            **kwargs: Additional structured fields.
        """
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback.

        Args:
            message: Log message.
            **kwargs: Additional structured fields.
        """
        self._log(logging.ERROR, message, exc_info=True, **kwargs)

    @contextmanager
    def timed(
        self,
        operation: str,
        log_level: LogLevel = LogLevel.INFO,
        **kwargs: Any,
    ) -> Generator[None, None, None]:
        """Context manager for timing operations.

        Args:
            operation: Name of the operation being timed.
            log_level: Level to log at.
            **kwargs: Additional structured fields.

        Yields:
            None
        """
        start_time = time.perf_counter()
        self._log(
            getattr(logging, log_level.value),
            f"{operation} started",
            **kwargs,
        )

        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log(
                getattr(logging, log_level.value),
                f"{operation} completed",
                duration_ms=duration_ms,
                **kwargs,
            )

    def start_timer(self, name: str) -> None:
        """Start a named timer.

        Args:
            name: Timer name.
        """
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return duration.

        Args:
            name: Timer name.

        Returns:
            Duration in milliseconds.

        Raises:
            KeyError: If timer was not started.
        """
        if name not in self._timers:
            raise KeyError(f"Timer '{name}' was not started")

        duration_ms = (time.perf_counter() - self._timers[name]) * 1000
        del self._timers[name]
        return duration_ms

    def set_context(self, **kwargs: Any) -> None:
        """Set persistent context fields.

        Args:
            **kwargs: Context fields to set.
        """
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context fields."""
        self._context.clear()

    @contextmanager
    def context(self, **kwargs: Any) -> Generator["StructuredLogger", None, None]:
        """Context manager for temporary context.

        Args:
            **kwargs: Context fields.

        Yields:
            This logger with context applied.
        """
        old_context = self._context.copy()
        self._context.update(kwargs)
        try:
            yield self
        finally:
            self._context = old_context


class WorkflowLogContext:
    """Context manager for workflow-scoped logging.

    Automatically adds workflow_id, stage, and other context
    to all log messages within the context.

    Example:
        >>> logger = StructuredLogger.create("workflow")
        >>> with WorkflowLogContext(logger, workflow_id="wf-123", stage="training"):
        ...     logger.info("Processing batch", batch=1)
    """

    def __init__(
        self,
        logger: StructuredLogger,
        workflow_id: str | None = None,
        stage: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize workflow log context.

        Args:
            logger: StructuredLogger instance.
            workflow_id: Workflow identifier.
            stage: Current workflow stage.
            **kwargs: Additional context fields.
        """
        self._logger = logger
        self._workflow_id = workflow_id or str(uuid.uuid4())[:8]
        self._stage = stage
        self._extra = kwargs
        self._old_context: dict[str, Any] = {}

    def __enter__(self) -> StructuredLogger:
        """Enter context."""
        self._old_context = self._logger._context.copy()
        self._logger.set_context(
            workflow_id=self._workflow_id,
            stage=self._stage,
            **self._extra,
        )
        return self._logger

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context."""
        self._logger._context = self._old_context


@dataclass
class AggregatedStats:
    """Aggregated statistics from log entries.

    Attributes:
        total_entries: Total number of log entries.
        entries_by_level: Count by log level.
        avg_duration_ms: Average duration for timed operations.
        total_duration_ms: Total duration for timed operations.
        error_count: Number of errors.
        unique_operations: Unique operation names.
    """

    total_entries: int = 0
    entries_by_level: dict[str, int] = field(default_factory=dict)
    avg_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    error_count: int = 0
    unique_operations: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "total_entries": self.total_entries,
            "entries_by_level": self.entries_by_level,
            "avg_duration_ms": self.avg_duration_ms,
            "total_duration_ms": self.total_duration_ms,
            "error_count": self.error_count,
            "unique_operations": list(self.unique_operations),
        }


class LogAggregator:
    """Aggregates and summarizes log entries.

    Useful for generating summary statistics from workflow logs.

    Example:
        >>> aggregator = LogAggregator()
        >>> aggregator.add_entry(entry1)
        >>> aggregator.add_entry(entry2)
        >>> stats = aggregator.get_stats()
        >>> print(stats.total_entries)
    """

    def __init__(self) -> None:
        """Initialize log aggregator."""
        self._entries: list[LogEntry] = []
        self._stats = AggregatedStats()

    def add_entry(self, entry: LogEntry) -> None:
        """Add a log entry.

        Args:
            entry: Log entry to add.
        """
        self._entries.append(entry)
        self._update_stats(entry)

    def _update_stats(self, entry: LogEntry) -> None:
        """Update aggregated statistics.

        Args:
            entry: Log entry to process.
        """
        self._stats.total_entries += 1

        # Count by level
        level = entry.level
        self._stats.entries_by_level[level] = (
            self._stats.entries_by_level.get(level, 0) + 1
        )

        # Track errors
        if level in ("ERROR", "CRITICAL"):
            self._stats.error_count += 1

        # Track durations
        if entry.duration_ms is not None:
            self._stats.total_duration_ms += entry.duration_ms

        # Track operations
        if "operation" in entry.extra:
            self._stats.unique_operations.add(entry.extra["operation"])

    def get_stats(self) -> AggregatedStats:
        """Get aggregated statistics.

        Returns:
            AggregatedStats instance.
        """
        # Calculate average duration
        timed_entries = [e for e in self._entries if e.duration_ms is not None]
        if timed_entries:
            self._stats.avg_duration_ms = (
                self._stats.total_duration_ms / len(timed_entries)
            )

        return self._stats

    def get_entries(
        self,
        level: str | None = None,
        workflow_id: str | None = None,
    ) -> list[LogEntry]:
        """Get filtered log entries.

        Args:
            level: Filter by log level.
            workflow_id: Filter by workflow ID.

        Returns:
            Filtered list of log entries.
        """
        entries = self._entries

        if level:
            entries = [e for e in entries if e.level == level]

        if workflow_id:
            entries = [e for e in entries if e.workflow_id == workflow_id]

        return entries

    def get_errors(self) -> list[LogEntry]:
        """Get all error entries.

        Returns:
            List of error log entries.
        """
        return [e for e in self._entries if e.level in ("ERROR", "CRITICAL")]

    def clear(self) -> None:
        """Clear all entries and reset stats."""
        self._entries.clear()
        self._stats = AggregatedStats()


class PerformanceLogger:
    """Logger specialized for performance tracking.

    Tracks execution times, memory usage, and other performance metrics.

    Example:
        >>> perf_logger = PerformanceLogger("training")
        >>> with perf_logger.track("model_fitting"):
        ...     model.fit(X, y)
        >>> perf_logger.log_summary()
    """

    def __init__(
        self,
        name: str,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Initialize performance logger.

        Args:
            name: Name for the performance logger.
            logger: Optional StructuredLogger to use.
        """
        self._name = name
        self._logger = logger or StructuredLogger.create(f"perf.{name}")
        self._timings: dict[str, list[float]] = {}
        self._counters: dict[str, int] = {}
        self._started_at = datetime.now()

    @contextmanager
    def track(
        self,
        operation: str,
        log: bool = True,
    ) -> Generator[None, None, None]:
        """Track execution time of an operation.

        Args:
            operation: Name of the operation.
            log: Whether to log the timing.

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            if operation not in self._timings:
                self._timings[operation] = []
            self._timings[operation].append(duration_ms)

            if log:
                self._logger.debug(
                    f"Operation '{operation}' completed",
                    operation=operation,
                    duration_ms=duration_ms,
                )

    def increment(self, counter: str, amount: int = 1) -> None:
        """Increment a counter.

        Args:
            counter: Counter name.
            amount: Amount to increment by.
        """
        self._counters[counter] = self._counters.get(counter, 0) + amount

    def record_timing(self, operation: str, duration_ms: float) -> None:
        """Record a timing directly.

        Args:
            operation: Operation name.
            duration_ms: Duration in milliseconds.
        """
        if operation not in self._timings:
            self._timings[operation] = []
        self._timings[operation].append(duration_ms)

    def get_timing_stats(self, operation: str) -> dict[str, float]:
        """Get timing statistics for an operation.

        Args:
            operation: Operation name.

        Returns:
            Dictionary with min, max, mean, total, count.
        """
        if operation not in self._timings:
            return {}

        timings = self._timings[operation]
        return {
            "min_ms": min(timings),
            "max_ms": max(timings),
            "mean_ms": sum(timings) / len(timings),
            "total_ms": sum(timings),
            "count": len(timings),
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get timing statistics for all operations.

        Returns:
            Dictionary mapping operation names to stats.
        """
        return {op: self.get_timing_stats(op) for op in self._timings}

    def get_counters(self) -> dict[str, int]:
        """Get all counter values.

        Returns:
            Dictionary of counter values.
        """
        return self._counters.copy()

    def log_summary(self) -> None:
        """Log a summary of performance metrics."""
        total_duration = (datetime.now() - self._started_at).total_seconds() * 1000

        self._logger.info(
            "Performance summary",
            total_duration_ms=total_duration,
            operations=len(self._timings),
            counters=self._counters,
        )

        for operation, timings in self._timings.items():
            stats = self.get_timing_stats(operation)
            self._logger.info(
                f"  {operation}",
                count=stats["count"],
                total_ms=stats["total_ms"],
                mean_ms=stats["mean_ms"],
            )

    def reset(self) -> None:
        """Reset all timings and counters."""
        self._timings.clear()
        self._counters.clear()
        self._started_at = datetime.now()


def create_workflow_logger(
    workflow_name: str,
    workflow_id: str | None = None,
    level: LogLevel = LogLevel.INFO,
    fmt: LogFormat = LogFormat.TEXT,
) -> StructuredLogger:
    """Factory function to create a workflow logger.

    Args:
        workflow_name: Name of the workflow.
        workflow_id: Optional workflow ID.
        level: Log level.
        fmt: Output format.

    Returns:
        Configured StructuredLogger.
    """
    logger = StructuredLogger.create(
        name=f"workflow.{workflow_name}",
        level=level,
        fmt=fmt,
    )

    if workflow_id:
        logger.set_context(workflow_id=workflow_id)

    return logger
