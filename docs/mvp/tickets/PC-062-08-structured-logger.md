# PC-062-08: StructuredLogger

**Epic:** [EPIC-08: Orchestrator](../epics/EPIC-08-orchestrator.md)
**Task Reference:** T-08.6
**Priority:** Medium
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `StructuredLogger` R6 class for structured JSON logging with workflow context tracking. This component provides consistent logging across all PrevCarga workflows with support for multiple output handlers.

---

## Acceptance Criteria

- [ ] Logger module created in `R/orchestrator/logger.R`
- [ ] `StructuredLogger` R6 class with JSON output
- [ ] Context tracking for workflow metadata
- [ ] Multiple log levels (DEBUG, INFO, WARN, ERROR)
- [ ] Support for file, console, and custom handlers
- [ ] Log rotation and archival support

---

## Technical Specification

### File Location
```
R/orchestrator/logger.R
```

### StructuredLogger R6 Class

```r
#' @title StructuredLogger
#' @description Structured JSON logging with context tracking
#'
#' Provides structured logging capabilities with context propagation
#' for tracing operations across workflow components.
#'
#' @export
StructuredLogger <- R6::R6Class(
  "StructuredLogger",
  private = list(
    context = list(),
    handlers = list(),
    level = NULL,
    levels = c("DEBUG" = 10, "INFO" = 20, "WARN" = 30, "ERROR" = 40),
    format = NULL,
    session_id = NULL
  ),
  public = list(
    #' @description Initialize logger
    #' @param level Minimum log level ("DEBUG", "INFO", "WARN", "ERROR")
    #' @param format Output format ("json" or "text")
    initialize = function(level = "INFO", format = "json") {
      checkmate::assert_choice(level, names(private$levels))
      checkmate::assert_choice(format, c("json", "text"))

      private$level <- level
      private$format <- format
      private$session_id <- self$generate_session_id()

      # Add default console handler
      self$add_handler("console", self$console_handler)
    },

    #' @description Generate unique session ID
    #' @return Character session ID
    generate_session_id = function() {
      sprintf("%s_%s",
              format(Sys.time(), "%Y%m%d%H%M%S"),
              substr(uuid::UUIDgenerate(), 1, 8))
    },

    #' @description Add context key-value pair
    #' @param key Context key
    #' @param value Context value
    #' @return Invisible self (for chaining)
    with_context = function(key, value) {
      private$context[[key]] <- value
      invisible(self)
    },

    #' @description Remove context key
    #' @param key Context key to remove
    #' @return Invisible self
    remove_context = function(key) {
      private$context[[key]] <- NULL
      invisible(self)
    },

    #' @description Clear all context
    #' @return Invisible self
    clear_context = function() {
      private$context <- list()
      invisible(self)
    },

    #' @description Get current context
    #' @return List of context values
    get_context = function() {
      private$context
    },

    #' @description Log message with level
    #' @param level Log level
    #' @param message Log message
    #' @param ... Additional named fields
    log = function(level, message, ...) {
      # Check if level should be logged
      if (private$levels[[level]] < private$levels[[private$level]]) {
        return(invisible(NULL))
      }

      # Build log entry
      entry <- list(
        timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3%z"),
        level = level,
        message = message,
        session_id = private$session_id,
        context = private$context
      )

      # Add extra fields
      extra <- list(...)
      if (length(extra) > 0) {
        entry$extra <- extra
      }

      # Emit to all handlers
      self$emit(entry)

      invisible(entry)
    },

    #' @description Log DEBUG message
    #' @param message Log message
    #' @param ... Additional fields
    debug = function(message, ...) {
      self$log("DEBUG", message, ...)
    },

    #' @description Log INFO message
    #' @param message Log message
    #' @param ... Additional fields
    info = function(message, ...) {
      self$log("INFO", message, ...)
    },

    #' @description Log WARN message
    #' @param message Log message
    #' @param ... Additional fields
    warn = function(message, ...) {
      self$log("WARN", message, ...)
    },

    #' @description Log ERROR message
    #' @param message Log message
    #' @param ... Additional fields
    error = function(message, ...) {
      self$log("ERROR", message, ...)
    },

    #' @description Emit log entry to handlers
    #' @param entry Log entry list
    emit = function(entry) {
      # Format entry
      formatted <- if (private$format == "json") {
        jsonlite::toJSON(entry, auto_unbox = TRUE, null = "null")
      } else {
        self$format_text(entry)
      }

      # Send to all handlers
      for (handler_name in names(private$handlers)) {
        tryCatch({
          private$handlers[[handler_name]](formatted, entry)
        }, error = function(e) {
          warning(sprintf("Logger handler '%s' failed: %s",
                          handler_name, conditionMessage(e)))
        })
      }
    },

    #' @description Format entry as text
    #' @param entry Log entry
    #' @return Formatted string
    format_text = function(entry) {
      context_str <- if (length(entry$context) > 0) {
        paste(sprintf("%s=%s", names(entry$context), entry$context),
              collapse = " ")
      } else {
        ""
      }

      extra_str <- if (!is.null(entry$extra) && length(entry$extra) > 0) {
        paste(sprintf("%s=%s", names(entry$extra), entry$extra),
              collapse = " ")
      } else {
        ""
      }

      sprintf("[%s] %s - %s %s %s",
              entry$timestamp,
              entry$level,
              entry$message,
              context_str,
              extra_str)
    },

    #' @description Add output handler
    #' @param name Handler name
    #' @param handler Handler function(formatted, entry)
    add_handler = function(name, handler) {
      checkmate::assert_function(handler)
      private$handlers[[name]] <- handler
      invisible(self)
    },

    #' @description Remove output handler
    #' @param name Handler name
    remove_handler = function(name) {
      private$handlers[[name]] <- NULL
      invisible(self)
    },

    #' @description Console output handler
    #' @param formatted Formatted string
    #' @param entry Raw entry
    console_handler = function(formatted, entry) {
      if (entry$level == "ERROR") {
        message(formatted)
      } else {
        cat(formatted, "\n")
      }
    },

    #' @description Create file handler
    #' @param file_path Path to log file
    #' @param append Append to existing file
    #' @return Handler function
    create_file_handler = function(file_path, append = TRUE) {
      # Ensure directory exists
      dir.create(dirname(file_path), recursive = TRUE, showWarnings = FALSE)

      function(formatted, entry) {
        cat(formatted, "\n", file = file_path, append = append)
      }
    },

    #' @description Create rotating file handler
    #' @param file_path Base path for log files
    #' @param max_size_mb Maximum file size in MB before rotation
    #' @param max_files Maximum number of rotated files to keep
    #' @return Handler function
    create_rotating_handler = function(file_path, max_size_mb = 10, max_files = 5) {
      # Ensure directory exists
      dir.create(dirname(file_path), recursive = TRUE, showWarnings = FALSE)

      function(formatted, entry) {
        # Check if rotation needed
        if (file.exists(file_path)) {
          size_mb <- file.info(file_path)$size / (1024 * 1024)
          if (size_mb >= max_size_mb) {
            # Rotate files
            for (i in (max_files - 1):1) {
              old_file <- sprintf("%s.%d", file_path, i)
              new_file <- sprintf("%s.%d", file_path, i + 1)
              if (file.exists(old_file)) {
                file.rename(old_file, new_file)
              }
            }
            file.rename(file_path, sprintf("%s.1", file_path))

            # Remove excess files
            for (i in max_files:100) {
              excess_file <- sprintf("%s.%d", file_path, i)
              if (file.exists(excess_file)) {
                file.remove(excess_file)
              }
            }
          }
        }

        # Write to file
        cat(formatted, "\n", file = file_path, append = TRUE)
      }
    },

    #' @description Set log level
    #' @param level New log level
    set_level = function(level) {
      checkmate::assert_choice(level, names(private$levels))
      private$level <- level
      invisible(self)
    },

    #' @description Get current log level
    #' @return Log level string
    get_level = function() {
      private$level
    },

    #' @description Get session ID
    #' @return Session ID string
    get_session_id = function() {
      private$session_id
    },

    #' @description Log with timing
    #' @param message Message describing operation
    #' @param expr Expression to time
    #' @return Result of expression
    timed = function(message, expr) {
      start_time <- Sys.time()
      self$info(sprintf("Starting: %s", message))

      result <- tryCatch({
        force(expr)
      }, error = function(e) {
        elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        self$error(sprintf("Failed: %s", message),
                   elapsed_secs = elapsed,
                   error = conditionMessage(e))
        stop(e)
      })

      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      self$info(sprintf("Completed: %s", message), elapsed_secs = elapsed)

      result
    },

    #' @description Print logger info
    print = function() {
      cat("StructuredLogger\n")
      cat(sprintf("  Level: %s\n", private$level))
      cat(sprintf("  Format: %s\n", private$format))
      cat(sprintf("  Session: %s\n", private$session_id))
      cat(sprintf("  Handlers: %s\n", paste(names(private$handlers), collapse = ", ")))
      cat(sprintf("  Context: %d keys\n", length(private$context)))
      invisible(self)
    }
  )
)
```

### Convenience Functions

```r
#' Create logger from config
#'
#' @param config ConfigManager instance
#' @return StructuredLogger instance
#' @export
create_logger <- function(config) {
  level <- config$get_with_default("logging.level", "INFO")
  format <- config$get_with_default("logging.format", "json")
  file_path <- config$get("logging.file")

  logger <- StructuredLogger$new(level = level, format = format)

  if (!is.null(file_path)) {
    logger$add_handler("file", logger$create_rotating_handler(file_path))
  }

  logger
}


#' Global logger instance
#' @noRd
.global_logger <- NULL


#' Get global logger
#'
#' @return StructuredLogger instance
#' @export
get_logger <- function() {
  if (is.null(.global_logger)) {
    .global_logger <<- StructuredLogger$new()
  }
  .global_logger
}


#' Initialize global logger
#'
#' @param config ConfigManager instance
#' @export
init_logger <- function(config) {
  .global_logger <<- create_logger(config)
}


#' Log shorthand functions
#' @rdname logging
#' @export
log_debug <- function(message, ...) get_logger()$debug(message, ...)

#' @rdname logging
#' @export
log_info <- function(message, ...) get_logger()$info(message, ...)

#' @rdname logging
#' @export
log_warn <- function(message, ...) get_logger()$warn(message, ...)

#' @rdname logging
#' @export
log_error <- function(message, ...) get_logger()$error(message, ...)
```

### Usage Example

```r
# Initialize logger
logger <- StructuredLogger$new(level = "DEBUG", format = "json")

# Add file handler
logger$add_handler("file", logger$create_rotating_handler(
  file_path = "logs/prevcarga.log",
  max_size_mb = 10,
  max_files = 5
))

# Add workflow context
logger$with_context("workflow", "train")
logger$with_context("version", "v1.0.0")

# Log messages
logger$info("Starting training workflow", areas = 5, models = 3)
# {"timestamp":"2025-01-17T10:30:45.123+0000","level":"INFO",
#  "message":"Starting training workflow","session_id":"20250117103045_abc12345",
#  "context":{"workflow":"train","version":"v1.0.0"},
#  "extra":{"areas":5,"models":3}}

logger$debug("Processing area", area = "RJ")

logger$warn("Slow training detected", area = "MG", elapsed_secs = 300)

logger$error("Training failed", area = "SP", error = "Out of memory")

# Timed execution
result <- logger$timed("Training model", {
  train_model(data)
})
# Logs start and completion with elapsed time

# Update context
logger$with_context("area", "RJ")
logger$info("Area processing started")

logger$remove_context("area")
logger$info("Workflow complete")

# Clear context for new workflow
logger$clear_context()

# Using global logger
init_logger(config)
log_info("Application started")
log_warn("Cache miss", key = "model_artifact")
```

### JSON Log Output Example

```json
{"timestamp":"2025-01-17T10:30:45.123+0000","level":"INFO","message":"Starting training workflow","session_id":"20250117103045_abc12345","context":{"workflow":"train","version":"v1.0.0"},"extra":{"areas":5,"models":3}}
{"timestamp":"2025-01-17T10:30:46.234+0000","level":"DEBUG","message":"Processing area","session_id":"20250117103045_abc12345","context":{"workflow":"train","version":"v1.0.0"},"extra":{"area":"RJ"}}
{"timestamp":"2025-01-17T10:35:47.456+0000","level":"WARN","message":"Slow training detected","session_id":"20250117103045_abc12345","context":{"workflow":"train","version":"v1.0.0"},"extra":{"area":"MG","elapsed_secs":300}}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | Logger created |
| TC-002 | Initialize with custom level | Level set correctly |
| TC-003 | with_context() adds key | Context updated |
| TC-004 | remove_context() removes key | Key removed |
| TC-005 | clear_context() clears all | Context empty |
| TC-006 | debug() at INFO level | Not logged |
| TC-007 | info() at INFO level | Logged |
| TC-008 | warn() at INFO level | Logged |
| TC-009 | error() at INFO level | Logged |
| TC-010 | JSON format output | Valid JSON |
| TC-011 | Text format output | Formatted string |
| TC-012 | File handler writes | File created |
| TC-013 | Rotating handler rotates | Files rotated |
| TC-014 | timed() measures time | Elapsed logged |
| TC-015 | Multiple handlers | All called |

---

## Dependencies

None (foundation module)

---

## Definition of Done

- [ ] StructuredLogger R6 class implemented
- [ ] JSON and text format support
- [ ] Context tracking working
- [ ] File and rotating handlers
- [ ] Log level filtering
- [ ] Timed execution helper
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- JSON format recommended for production (parseable by ELK/Splunk)
- Text format useful for development/debugging
- Session ID enables tracing across log aggregators
- Consider adding CloudWatch/S3 handlers for AWS integration
- Log rotation prevents disk space issues in long-running processes
