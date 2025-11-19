# PC-004-00: Structured Logging Framework

**Ticket ID:** PC-004-00  
**Epic:** [Epic-00: Project Foundation & Setup](../epics/Epic-00.md)  
**User Story:** US-00.4  
**Story Points:** 3  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a structured logging system with JSON formatting to enable effective debugging, monitoring, and observability across all system components.

**As a** developer  
**I want** a structured logging system  
**So that** I can debug issues and monitor system behavior effectively

---

## ✅ Acceptance Criteria

- [ ] Logging configuration created using Python's `logging` module
- [ ] JSON structured logging format implemented
- [ ] Log levels configurable via environment variable or config file
- [ ] Logger includes: timestamp, level, module, function, message, context
- [ ] Console handler configured for development
- [ ] File handler configured with rotation
- [ ] Logging utility module created in `src/utils/`
- [ ] Example usage documented

---

## 🔧 Implementation Tasks

### 1. Create Logging Configuration File
- [ ] Create `config/logging.yaml` with:
  - Formatters (JSON and standard console)
  - Handlers (console and rotating file)
  - Loggers (root and prevcarga)
  - Log levels configuration
- [ ] Configure log file rotation (10MB per file, 5 backups)
- [ ] Set appropriate log levels for each environment

### 2. Create Logging Utility Module
- [ ] Create `src/utils/logger.py`
- [ ] Implement `setup_logging()` function to load YAML config
- [ ] Implement `get_logger(name: str)` factory function
- [ ] Add support for environment variable `LOG_LEVEL` override
- [ ] Add context manager for temporary log level changes
- [ ] Ensure logs directory is created if it doesn't exist

### 3. Implement JSON Formatter
- [ ] Configure `python-json-logger` formatter
- [ ] Include fields: timestamp, name, level, module, function, message
- [ ] Support additional context fields (user_id, request_id, etc.)
- [ ] Format exceptions with stack traces

### 4. Configure Handlers
- [ ] Console handler for development:
  - Output to stdout
  - Human-readable format
  - Configurable log level (default: DEBUG)
- [ ] File handler for production:
  - JSON format
  - Log rotation (size-based)
  - Configurable log level (default: INFO)
  - Location: `logs/prevcarga.log`

### 5. Create Usage Examples and Documentation
- [ ] Create `examples/logging_example.py` with:
  - Basic logging usage
  - Logging with context
  - Exception logging
  - Different log levels
- [ ] Document logging in README.md
- [ ] Add docstrings to all logging functions
- [ ] Create troubleshooting guide for logging issues

### 6. Integrate with Existing Modules
- [ ] Add logging to `src/storage/s3_client.py`
- [ ] Add logging to `src/storage/config.py`
- [ ] Ensure all modules use `get_logger(__name__)`
- [ ] Test logging output in all integrated modules

---

## 📄 Configuration Files

### config/logging.yaml

```yaml
version: 1
disable_existing_loggers: false

formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "%(asctime)s %(name)s %(levelname)s %(module)s %(funcName)s %(message)s"
  
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: logs/prevcarga.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    encoding: utf8

loggers:
  prevcarga:
    level: DEBUG
    handlers: [console, file]
    propagate: false
  
  src:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

---

## 💻 Code Structure

### src/utils/logger.py

```python
"""Structured logging utility module."""
import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(config_path: str = "config/logging.yaml", log_level: Optional[str] = None) -> None:
    """
    Set up logging configuration from YAML file.
    
    Args:
        config_path: Path to logging configuration YAML file
        log_level: Optional log level override (from environment variable)
    """
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Load configuration
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Apply log level override if provided
        if log_level:
            if "loggers" in config and "prevcarga" in config["loggers"]:
                config["loggers"]["prevcarga"]["level"] = log_level.upper()
        
        logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.warning(f"Logging config file not found: {config_path}. Using basic config.")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    # Setup logging on first call
    if not logging.getLogger().handlers:
        log_level = os.getenv("LOG_LEVEL")
        setup_logging(log_level=log_level)
    
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, logger: logging.Logger, **context: dict) -> None:
        """
        Initialize log context.
        
        Args:
            logger: Logger instance
            **context: Key-value pairs to add to log context
        """
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self) -> logging.Logger:
        """Enter context and add context fields to logger."""
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        self.old_factory = old_factory
        logging.setLogRecordFactory(record_factory)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore original factory."""
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)
```

### examples/logging_example.py

```python
"""Example usage of the logging framework."""
from src.utils.logger import get_logger, LogContext

# Get logger for this module
logger = get_logger(__name__)


def basic_logging_example() -> None:
    """Basic logging at different levels."""
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")


def logging_with_context_example() -> None:
    """Logging with additional context."""
    with LogContext(logger, user_id="user123", request_id="req456"):
        logger.info("Processing user request")
        logger.info("Request completed successfully")


def exception_logging_example() -> None:
    """Logging exceptions with stack traces."""
    try:
        result = 1 / 0
    except ZeroDivisionError:
        logger.exception("An error occurred during calculation")


def structured_logging_example() -> None:
    """Logging with structured data."""
    logger.info(
        "Model training completed",
        extra={
            "model_name": "lgbm",
            "accuracy": 0.95,
            "training_time": 120.5,
        }
    )


if __name__ == "__main__":
    basic_logging_example()
    logging_with_context_example()
    exception_logging_example()
    structured_logging_example()
```

---

## 🧪 Testing & Validation

### Validation Commands

```bash
# Create logs directory
mkdir -p logs

# Run logging example
uv run python examples/logging_example.py

# Check console output (should be human-readable)
# Check log file (should be JSON formatted)
cat logs/prevcarga.log

# Test log level override
LOG_LEVEL=DEBUG uv run python examples/logging_example.py

# Verify JSON format
cat logs/prevcarga.log | python -m json.tool

# Test log rotation (create large log file)
for i in {1..100000}; do echo "Test log line $i" >> logs/prevcarga.log; done
uv run python examples/logging_example.py
ls -lh logs/  # Should see rotated files
```

### Unit Tests

```python
# tests/utils/test_logger.py
def test_get_logger():
    """Test logger initialization."""
    from src.utils.logger import get_logger
    logger = get_logger("test")
    assert logger.name == "test"

def test_logging_to_file(tmp_path):
    """Test logging writes to file."""
    # Test implementation
    pass

def test_json_format():
    """Test JSON log format is valid."""
    # Test implementation
    pass
```

### Success Criteria
- [ ] Console logs are human-readable
- [ ] File logs are valid JSON
- [ ] Log rotation works correctly
- [ ] Exception stack traces are captured
- [ ] Context fields appear in logs
- [ ] Log level override works
- [ ] All tests pass

---

## 📝 Technical Notes

- Use `logger = get_logger(__name__)` in all modules for proper logger hierarchy
- Never log sensitive information (passwords, API keys, PII)
- Use appropriate log levels:
  - DEBUG: Detailed diagnostic information
  - INFO: General informational messages
  - WARNING: Warning messages for potentially harmful situations
  - ERROR: Error messages for serious problems
  - CRITICAL: Critical messages for very serious problems
- JSON logs are easier to parse for log aggregation tools (CloudWatch, ELK)
- Log rotation prevents disk space issues
- Consider async logging for high-throughput scenarios (future)

---

## 🔗 Dependencies

**Depends On:**
- PC-001-00: Repository Structure Setup
- PC-002-00: Python Environment with uv

**Blocks:**
- All future tickets (logging should be used everywhere)

**Integrates With:**
- PC-003-00: S3 Storage Configuration (add logging to S3 operations)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] All implementation tasks completed
- [ ] Configuration files created
- [ ] Logging utility module complete
- [ ] Examples documented and working
- [ ] Integration with S3 client complete
- [ ] Unit tests pass
- [ ] Documentation updated in README.md
- [ ] Code reviewed and approved
- [ ] Ready for use in all future modules

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next Ticket:** [PC-005-00: Local Test Scripts](PC-005-00-local-test-scripts.md)
