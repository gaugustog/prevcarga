"""Tests for structured logger.

This module tests the StructuredLogger and related components.
"""

import json
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.structured_logger import (
    AggregatedStats,
    LogAggregator,
    LogEntry,
    LogFormat,
    LogLevel,
    PerformanceLogger,
    StructuredFormatter,
    StructuredLogger,
    WorkflowLogContext,
    create_workflow_logger,
)


# ==================== LogEntry Tests ====================


class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_log_entry_creation(self) -> None:
        """Test basic LogEntry creation."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            level="INFO",
            message="Test message",
            logger_name="test.logger",
        )

        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.logger_name == "test.logger"
        assert entry.correlation_id is None

    def test_log_entry_with_context(self) -> None:
        """Test LogEntry with context fields."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            level="INFO",
            message="Test message",
            logger_name="test.logger",
            correlation_id="abc123",
            workflow_id="wf-456",
            stage="training",
            extra={"area": "SECO", "horizon": 1},
        )

        assert entry.correlation_id == "abc123"
        assert entry.workflow_id == "wf-456"
        assert entry.stage == "training"
        assert entry.extra["area"] == "SECO"

    def test_log_entry_to_dict(self) -> None:
        """Test LogEntry serialization to dict."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            level="WARNING",
            message="Warning message",
            logger_name="test",
            duration_ms=123.45,
        )

        data = entry.to_dict()

        assert data["level"] == "WARNING"
        assert data["message"] == "Warning message"
        assert data["duration_ms"] == 123.45
        assert "timestamp" in data

    def test_log_entry_to_json(self) -> None:
        """Test LogEntry serialization to JSON."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            level="ERROR",
            message="Error message",
            logger_name="test",
        )

        json_str = entry.to_json()
        data = json.loads(json_str)

        assert data["level"] == "ERROR"
        assert data["message"] == "Error message"

    def test_log_entry_to_text(self) -> None:
        """Test LogEntry serialization to text."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            level="INFO",
            message="Processing started",
            logger_name="workflow",
            workflow_id="wf-123",
            stage="loading",
        )

        text = entry.to_text()

        assert "INFO" in text
        assert "Processing started" in text
        assert "wf:wf-123" in text
        assert "loading" in text


# ==================== LogLevel Tests ====================


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_all_levels_exist(self) -> None:
        """Test all expected levels exist."""
        expected = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level_name in expected:
            assert hasattr(LogLevel, level_name)

    def test_level_values(self) -> None:
        """Test level value strings."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"


# ==================== LogFormat Tests ====================


class TestLogFormat:
    """Tests for LogFormat enum."""

    def test_all_formats_exist(self) -> None:
        """Test all expected formats exist."""
        expected = ["TEXT", "JSON", "JSONL"]
        for fmt_name in expected:
            assert hasattr(LogFormat, fmt_name)


# ==================== StructuredFormatter Tests ====================


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_formatter_text_output(self) -> None:
        """Test text format output."""
        import logging

        formatter = StructuredFormatter(fmt=LogFormat.TEXT)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "Test message" in output

    def test_formatter_json_output(self) -> None:
        """Test JSON format output."""
        import logging

        formatter = StructuredFormatter(fmt=LogFormat.JSON)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"


# ==================== StructuredLogger Tests ====================


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_logger_creation(self) -> None:
        """Test logger creation."""
        logger = StructuredLogger(
            name="test_logger",
            level=LogLevel.INFO,
            fmt=LogFormat.TEXT,
        )

        assert logger.name == "test_logger"
        assert logger.correlation_id is not None

    def test_logger_create_factory(self) -> None:
        """Test factory creation method."""
        logger = StructuredLogger.create("test", level="DEBUG", fmt="json")

        assert logger.name == "test"

    def test_logger_create_factory_with_enums(self) -> None:
        """Test factory creation with enums."""
        logger = StructuredLogger.create(
            "test",
            level=LogLevel.WARNING,
            fmt=LogFormat.TEXT,
        )

        assert logger.name == "test"

    def test_logger_log_methods(self) -> None:
        """Test various log methods."""
        logger = StructuredLogger.create("test")

        # These should not raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_logger_with_structured_fields(self) -> None:
        """Test logging with structured fields."""
        logger = StructuredLogger.create("test")

        # Should not raise
        logger.info("Processing", area="SECO", horizon=1, count=100)

    def test_logger_exception(self) -> None:
        """Test exception logging."""
        logger = StructuredLogger.create("test")

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("An error occurred")

    def test_logger_timed_context(self) -> None:
        """Test timed context manager."""
        logger = StructuredLogger.create("test")

        with logger.timed("test_operation"):
            time.sleep(0.01)

        # Operation should complete without error

    def test_logger_start_stop_timer(self) -> None:
        """Test manual timer start/stop."""
        logger = StructuredLogger.create("test")

        logger.start_timer("my_timer")
        time.sleep(0.01)
        duration = logger.stop_timer("my_timer")

        assert duration >= 10  # At least 10ms

    def test_logger_stop_timer_not_started(self) -> None:
        """Test stopping timer that wasn't started."""
        logger = StructuredLogger.create("test")

        with pytest.raises(KeyError):
            logger.stop_timer("nonexistent_timer")

    def test_logger_set_context(self) -> None:
        """Test setting persistent context."""
        logger = StructuredLogger.create("test")

        logger.set_context(user_id="user123", request_id="req456")
        logger.info("With context")

        logger.clear_context()
        logger.info("Without context")

    def test_logger_context_manager(self) -> None:
        """Test context manager for temporary context."""
        logger = StructuredLogger.create("test")

        with logger.context(temp_field="temporary"):
            logger.info("Inside context")

        logger.info("Outside context")


# ==================== WorkflowLogContext Tests ====================


class TestWorkflowLogContext:
    """Tests for WorkflowLogContext."""

    def test_workflow_context_basic(self) -> None:
        """Test basic workflow context."""
        logger = StructuredLogger.create("test")

        with WorkflowLogContext(logger, workflow_id="wf-123", stage="training"):
            logger.info("Inside workflow context")

    def test_workflow_context_auto_id(self) -> None:
        """Test automatic workflow ID generation."""
        logger = StructuredLogger.create("test")

        with WorkflowLogContext(logger, stage="loading") as log:
            # Should have auto-generated workflow_id
            assert "workflow_id" in log._context

    def test_workflow_context_restores_state(self) -> None:
        """Test that context is restored after exit."""
        logger = StructuredLogger.create("test")
        logger.set_context(existing_field="value")

        with WorkflowLogContext(logger, workflow_id="wf-456"):
            pass

        assert logger._context.get("existing_field") == "value"


# ==================== AggregatedStats Tests ====================


class TestAggregatedStats:
    """Tests for AggregatedStats dataclass."""

    def test_stats_creation(self) -> None:
        """Test stats creation."""
        stats = AggregatedStats()

        assert stats.total_entries == 0
        assert stats.error_count == 0

    def test_stats_to_dict(self) -> None:
        """Test stats serialization."""
        stats = AggregatedStats(
            total_entries=100,
            entries_by_level={"INFO": 80, "ERROR": 20},
            error_count=20,
        )

        data = stats.to_dict()

        assert data["total_entries"] == 100
        assert data["error_count"] == 20


# ==================== LogAggregator Tests ====================


class TestLogAggregator:
    """Tests for LogAggregator."""

    def test_aggregator_creation(self) -> None:
        """Test aggregator creation."""
        aggregator = LogAggregator()
        stats = aggregator.get_stats()

        assert stats.total_entries == 0

    def test_aggregator_add_entry(self) -> None:
        """Test adding entries."""
        aggregator = LogAggregator()

        entry = LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            message="Test",
            logger_name="test",
        )
        aggregator.add_entry(entry)

        stats = aggregator.get_stats()
        assert stats.total_entries == 1
        assert stats.entries_by_level.get("INFO") == 1

    def test_aggregator_count_errors(self) -> None:
        """Test error counting."""
        aggregator = LogAggregator()

        for level in ["INFO", "ERROR", "ERROR", "CRITICAL"]:
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                message="Test",
                logger_name="test",
            )
            aggregator.add_entry(entry)

        stats = aggregator.get_stats()
        assert stats.error_count == 3

    def test_aggregator_track_durations(self) -> None:
        """Test duration tracking."""
        aggregator = LogAggregator()

        for duration in [100.0, 200.0, 300.0]:
            entry = LogEntry(
                timestamp=datetime.now(),
                level="INFO",
                message="Timed op",
                logger_name="test",
                duration_ms=duration,
            )
            aggregator.add_entry(entry)

        stats = aggregator.get_stats()
        assert stats.total_duration_ms == 600.0
        assert stats.avg_duration_ms == 200.0

    def test_aggregator_get_entries_filtered(self) -> None:
        """Test filtering entries."""
        aggregator = LogAggregator()

        for level in ["INFO", "ERROR", "INFO"]:
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                message="Test",
                logger_name="test",
            )
            aggregator.add_entry(entry)

        info_entries = aggregator.get_entries(level="INFO")
        error_entries = aggregator.get_entries(level="ERROR")

        assert len(info_entries) == 2
        assert len(error_entries) == 1

    def test_aggregator_get_errors(self) -> None:
        """Test getting error entries."""
        aggregator = LogAggregator()

        for level in ["INFO", "ERROR", "WARNING", "CRITICAL"]:
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                message="Test",
                logger_name="test",
            )
            aggregator.add_entry(entry)

        errors = aggregator.get_errors()
        assert len(errors) == 2

    def test_aggregator_clear(self) -> None:
        """Test clearing aggregator."""
        aggregator = LogAggregator()

        entry = LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            message="Test",
            logger_name="test",
        )
        aggregator.add_entry(entry)
        aggregator.clear()

        stats = aggregator.get_stats()
        assert stats.total_entries == 0


# ==================== PerformanceLogger Tests ====================


class TestPerformanceLogger:
    """Tests for PerformanceLogger."""

    def test_perf_logger_creation(self) -> None:
        """Test performance logger creation."""
        perf = PerformanceLogger("test")

        assert perf._name == "test"

    def test_perf_logger_track_context(self) -> None:
        """Test tracking operations."""
        perf = PerformanceLogger("test")

        with perf.track("test_operation"):
            time.sleep(0.01)

        stats = perf.get_timing_stats("test_operation")
        assert stats["count"] == 1
        assert stats["total_ms"] >= 10

    def test_perf_logger_track_multiple(self) -> None:
        """Test tracking multiple operations."""
        perf = PerformanceLogger("test")

        for _ in range(3):
            with perf.track("operation_a", log=False):
                time.sleep(0.005)

        stats = perf.get_timing_stats("operation_a")
        assert stats["count"] == 3

    def test_perf_logger_increment_counter(self) -> None:
        """Test counter increment."""
        perf = PerformanceLogger("test")

        perf.increment("batches_processed")
        perf.increment("batches_processed")
        perf.increment("items", amount=10)

        counters = perf.get_counters()
        assert counters["batches_processed"] == 2
        assert counters["items"] == 10

    def test_perf_logger_record_timing(self) -> None:
        """Test direct timing recording."""
        perf = PerformanceLogger("test")

        perf.record_timing("external_call", 150.0)
        perf.record_timing("external_call", 200.0)

        stats = perf.get_timing_stats("external_call")
        assert stats["count"] == 2
        assert stats["mean_ms"] == 175.0

    def test_perf_logger_get_all_stats(self) -> None:
        """Test getting all operation stats."""
        perf = PerformanceLogger("test")

        perf.record_timing("op_a", 100.0)
        perf.record_timing("op_b", 200.0)

        all_stats = perf.get_all_stats()
        assert "op_a" in all_stats
        assert "op_b" in all_stats

    def test_perf_logger_log_summary(self) -> None:
        """Test summary logging."""
        perf = PerformanceLogger("test")

        perf.record_timing("operation", 100.0)
        perf.increment("counter")

        # Should not raise
        perf.log_summary()

    def test_perf_logger_reset(self) -> None:
        """Test reset."""
        perf = PerformanceLogger("test")

        perf.record_timing("op", 100.0)
        perf.increment("counter")
        perf.reset()

        assert len(perf._timings) == 0
        assert len(perf._counters) == 0

    def test_perf_logger_timing_stats_empty(self) -> None:
        """Test timing stats for nonexistent operation."""
        perf = PerformanceLogger("test")

        stats = perf.get_timing_stats("nonexistent")
        assert stats == {}


# ==================== Factory Function Tests ====================


class TestCreateWorkflowLogger:
    """Tests for create_workflow_logger factory function."""

    def test_create_basic(self) -> None:
        """Test basic workflow logger creation."""
        logger = create_workflow_logger("training")

        assert "training" in logger.name

    def test_create_with_workflow_id(self) -> None:
        """Test creation with workflow ID."""
        logger = create_workflow_logger(
            "prediction",
            workflow_id="wf-123",
        )

        assert logger._context.get("workflow_id") == "wf-123"

    def test_create_with_level_and_format(self) -> None:
        """Test creation with custom level and format."""
        logger = create_workflow_logger(
            "backtesting",
            level=LogLevel.DEBUG,
            fmt=LogFormat.JSON,
        )

        assert logger._fmt == LogFormat.JSON


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for structured logging."""

    def test_full_workflow_logging(self) -> None:
        """Test complete workflow logging scenario."""
        logger = create_workflow_logger("test_workflow", workflow_id="wf-001")

        with logger.timed("full_workflow"):
            logger.info("Starting workflow")

            with logger.context(stage="loading"):
                logger.info("Loading data", files=3)

            with logger.context(stage="processing"):
                logger.info("Processing complete", items=100)

            logger.info("Workflow done")

    def test_logger_with_aggregator(self) -> None:
        """Test logger with log aggregator."""
        logger = StructuredLogger.create("test")
        aggregator = LogAggregator()

        # Simulate logging and aggregation
        for i in range(5):
            entry = LogEntry(
                timestamp=datetime.now(),
                level="INFO" if i % 2 == 0 else "ERROR",
                message=f"Message {i}",
                logger_name="test",
                duration_ms=float(i * 10),
            )
            aggregator.add_entry(entry)

        stats = aggregator.get_stats()
        assert stats.total_entries == 5
        assert stats.error_count == 2

    def test_performance_tracking_with_logging(self) -> None:
        """Test performance tracking integrated with logging."""
        perf = PerformanceLogger("integration_test")

        for i in range(3):
            with perf.track("batch_process", log=False):
                time.sleep(0.005)
            perf.increment("batches")

        counters = perf.get_counters()
        stats = perf.get_all_stats()

        assert counters["batches"] == 3
        assert "batch_process" in stats
        assert stats["batch_process"]["count"] == 3
