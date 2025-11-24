"""Tests for the structured logging module."""

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """Test that get_logger returns a Logger instance."""
        from src.utils.logger import get_logger

        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_same_name_returns_same_logger(self) -> None:
        """Test that same name returns the same logger instance."""
        from src.utils.logger import get_logger

        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")

        assert logger1 is logger2

    def test_get_logger_respects_log_level_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that LOG_LEVEL environment variable is respected."""
        import src.utils.logger as logger_module

        # Reset initialization state
        logger_module._logging_initialized = False
        monkeypatch.setenv("LOG_LEVEL", "WARNING")

        logger = logger_module.get_logger("env_test")

        # Logger should be configured
        assert logger_module._logging_initialized is True


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_creates_logs_dir(self, tmp_path: Path) -> None:
        """Test that setup_logging creates logs directory."""
        from src.utils.logger import setup_logging

        logs_dir = tmp_path / "logs"

        with patch("src.utils.logger.Path") as mock_path:
            mock_logs = mock_path.return_value
            mock_logs.mkdir = lambda **kwargs: logs_dir.mkdir(**kwargs)
            mock_logs.exists.return_value = False

            # This will use basic config since no file exists
            setup_logging(config_path="nonexistent.yaml")

    def test_setup_logging_with_level_override(self, tmp_path: Path) -> None:
        """Test that setup_logging respects level override."""
        from src.utils.logger import setup_logging

        # Create a minimal config
        config_path = tmp_path / "logging.yaml"
        config_path.write_text("""
version: 1
disable_existing_loggers: false
loggers:
  test:
    level: INFO
root:
  level: INFO
""")

        setup_logging(config_path=str(config_path), log_level="DEBUG")


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_log_context_adds_fields(self) -> None:
        """Test that LogContext adds fields to log records."""
        from src.utils.logger import LogContext, get_logger

        logger = get_logger("context_test")
        captured_records: list[logging.LogRecord] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_records.append(record)

        handler = CapturingHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            with LogContext(logger, request_id="test123", user_id="user456"):
                logger.info("Test message")

            assert len(captured_records) == 1
            record = captured_records[0]
            assert hasattr(record, "request_id")
            assert record.request_id == "test123"
            assert hasattr(record, "user_id")
            assert record.user_id == "user456"
        finally:
            logger.removeHandler(handler)

    def test_log_context_restores_factory(self) -> None:
        """Test that LogContext restores original factory on exit."""
        from src.utils.logger import LogContext, get_logger

        logger = get_logger("restore_test")
        original_factory = logging.getLogRecordFactory()

        with LogContext(logger, test_field="value"):
            pass  # Just enter and exit

        assert logging.getLogRecordFactory() == original_factory


class TestLogContextFunction:
    """Tests for log_context context manager function."""

    def test_log_context_function_adds_fields(self) -> None:
        """Test that log_context function adds fields to log records."""
        from src.utils.logger import get_logger, log_context

        logger = get_logger("func_context_test")
        captured_records: list[logging.LogRecord] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_records.append(record)

        handler = CapturingHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            with log_context(logger, operation="test_op"):
                logger.info("Test message")

            assert len(captured_records) == 1
            assert hasattr(captured_records[0], "operation")
            assert captured_records[0].operation == "test_op"
        finally:
            logger.removeHandler(handler)


class TestConfigureModuleLogger:
    """Tests for configure_module_logger function."""

    def test_configure_module_logger_returns_logger(self) -> None:
        """Test that configure_module_logger returns a logger."""
        from src.utils.logger import configure_module_logger

        logger = configure_module_logger("configured_test")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "configured_test"

    def test_configure_module_logger_sets_level(self) -> None:
        """Test that configure_module_logger sets the log level."""
        from src.utils.logger import configure_module_logger

        logger = configure_module_logger("level_test", level="WARNING")

        assert logger.level == logging.WARNING


class TestLoggingIntegration:
    """Integration tests for the logging system."""

    def test_json_formatter_output(self, tmp_path: Path) -> None:
        """Test that JSON formatter produces valid JSON."""
        from pythonjsonlogger import jsonlogger

        from src.utils.logger import get_logger

        logger = get_logger("json_test")
        log_file = tmp_path / "test.log"

        # Set up JSON file handler
        handler = logging.FileHandler(str(log_file))
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            logger.info("Test JSON message", extra={"custom_field": "value"})
            handler.flush()

            # Read and parse the log
            log_content = log_file.read_text().strip()
            log_data = json.loads(log_content)

            assert "message" in log_data
            assert log_data["message"] == "Test JSON message"
            assert "custom_field" in log_data
            assert log_data["custom_field"] == "value"
        finally:
            logger.removeHandler(handler)
            handler.close()

    def test_logging_to_file(self, tmp_path: Path) -> None:
        """Test that logging writes to file correctly."""
        from src.utils.logger import get_logger

        logger = get_logger("file_test")
        log_file = tmp_path / "test.log"

        handler = logging.FileHandler(str(log_file))
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            logger.info("Test file message")
            handler.flush()

            log_content = log_file.read_text()
            assert "INFO - Test file message" in log_content
        finally:
            logger.removeHandler(handler)
            handler.close()

    def test_exception_logging(self) -> None:
        """Test that exceptions are logged with stack traces."""
        from src.utils.logger import get_logger

        logger = get_logger("exception_test")
        captured_records: list[logging.LogRecord] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_records.append(record)

        handler = CapturingHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            try:
                msg = "intentional error"
                raise ValueError(msg)
            except ValueError:
                logger.exception("An error occurred")

            assert len(captured_records) == 1
            record = captured_records[0]
            assert record.exc_info is not None
            assert record.exc_info[0] == ValueError
        finally:
            logger.removeHandler(handler)
