# PC-064-09: CLI Entry Point

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.1
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Create the main CLI entry point that handles command routing, argument parsing, and dispatches to appropriate command handlers. This is the foundation for all CLI commands in PrevCarga.

---

## Acceptance Criteria

- [ ] CLI module created in `R/cli/main.R`
- [ ] Main entry point function `main()`
- [ ] Command dispatcher with handler registry
- [ ] Help and version commands
- [ ] Error handling with user-friendly messages
- [ ] Support for both argument mode and interactive mode

---

## Technical Specification

### File Location
```
R/cli/main.R
```

### Main Entry Point

```r
#' @title PrevCarga CLI Entry Point
#' @description Main entry point for the PrevCarga command-line interface
#'
#' Handles command routing to appropriate handlers and supports
#' both argument mode (for pipelines) and interactive mode.
#'
#' @param args Command line arguments (defaults to commandArgs(trailingOnly = TRUE))
#' @export
main <- function(args = commandArgs(trailingOnly = TRUE)) {
  # Set up error handling
  tryCatch({
    if (length(args) == 0) {
      # No arguments - show help
      print_help()
      return(invisible(0))
    }

    command <- tolower(args[1])

    # Handle special commands
    if (command %in% c("-h", "--help", "help")) {
      print_help()
      return(invisible(0))
    }

    if (command %in% c("-v", "--version", "version")) {
      print_version()
      return(invisible(0))
    }

    if (command == "interactive") {
      # Launch interactive shell
      shell <- PrevCargaShell$new()
      shell$run()
      return(invisible(0))
    }

    # Execute command
    execute_command(command, args[-1])

  }, error = function(e) {
    cli_error(conditionMessage(e))
    return(invisible(1))
  })

  invisible(0)
}


#' Execute a CLI command
#'
#' @param command Command name
#' @param args Command arguments
#' @noRd
execute_command <- function(command, args) {
  handler <- get_command_handler(command)

  if (is.null(handler)) {
    stop(sprintf(
      "Unknown command: '%s'. Run 'prevcarga help' for available commands.",
      command
    ))
  }

  handler(args)
}
```

### Command Registry

```r
#' Command handler registry
#' @noRd
.command_handlers <- new.env(parent = emptyenv())


#' Register a command handler
#'
#' @param name Command name
#' @param handler Handler function
#' @param description Brief description
#' @param usage Usage string
#' @noRd
register_command <- function(name, handler, description, usage = NULL) {
  checkmate::assert_string(name)
  checkmate::assert_function(handler)
  checkmate::assert_string(description)

  .command_handlers[[name]] <- list(
    handler = handler,
    description = description,
    usage = usage %||% sprintf("prevcarga %s [options]", name)
  )
}


#' Get command handler
#'
#' @param name Command name
#' @return Handler function or NULL
#' @noRd
get_command_handler <- function(name) {
  cmd <- .command_handlers[[name]]
  if (is.null(cmd)) return(NULL)
  cmd$handler
}


#' Get all registered commands
#'
#' @return Named list of command info
#' @noRd
get_all_commands <- function() {
  commands <- ls(.command_handlers)
  result <- list()

  for (cmd in commands) {
    result[[cmd]] <- .command_handlers[[cmd]]
  }

  result
}


#' Check if command exists
#'
#' @param name Command name
#' @return Logical
#' @noRd
has_command <- function(name) {
  exists(name, envir = .command_handlers)
}
```

### Help and Version

```r
#' Print CLI help
#' @noRd
print_help <- function() {
  cat("\n")
  cli_header("PrevCarga - Electric Load Forecasting System")
  cat("\n")

  cat("Usage: prevcarga <command> [options]\n\n")

  cat("Commands:\n")

  commands <- get_all_commands()
  max_len <- max(nchar(names(commands)))

  for (name in sort(names(commands))) {
    cmd <- commands[[name]]
    padding <- strrep(" ", max_len - nchar(name) + 2)
    cat(sprintf("  %s%s%s\n", name, padding, cmd$description))
  }

  cat("\n")
  cat("Options:\n")
  cat("  -h, --help      Show this help message\n")
  cat("  -v, --version   Show version information\n")
  cat("\n")
  cat("Run 'prevcarga <command> --help' for command-specific options.\n")
  cat("\n")
}


#' Print version information
#' @noRd
print_version <- function() {
  version <- utils::packageVersion("prevcargaons")
  cat(sprintf("PrevCarga version %s\n", version))
  cat(sprintf("R version %s\n", R.version.string))
}


#' Print CLI header with styling
#' @noRd
cli_header <- function(text) {
  width <- nchar(text) + 4
  border <- strrep("=", width)

  cat(border, "\n")
  cat("  ", text, "  \n", sep = "")
  cat(border, "\n")
}


#' Print CLI error message
#' @noRd
cli_error <- function(message) {
  cat(sprintf("\n[ERROR] %s\n\n", message), file = stderr())
}


#' Print CLI warning message
#' @noRd
cli_warning <- function(message) {
  cat(sprintf("[WARNING] %s\n", message), file = stderr())
}


#' Print CLI success message
#' @noRd
cli_success <- function(message) {
  cat(sprintf("[OK] %s\n", message))
}


#' Print CLI info message
#' @noRd
cli_info <- function(message) {
  cat(sprintf("[INFO] %s\n", message))
}
```

### Command Registration on Package Load

```r
#' Initialize CLI commands
#'
#' Called on package load to register all command handlers
#' @noRd
init_cli_commands <- function() {
  # Core workflow commands
  register_command("train", cmd_train,
                   "Train models for specified areas")
  register_command("predict", cmd_predict,
                   "Generate predictions")
  register_command("backtest", cmd_backtest,
                   "Run backtesting simulation")

  # Feature commands
  register_command("gen-features", cmd_gen_features,
                   "Generate features for training")
  register_command("eval-features", cmd_eval_features,
                   "Evaluate feature importance")

  # Evaluation commands
  register_command("eval-model", cmd_eval_model,
                   "Evaluate model performance")
  register_command("eval-combination", cmd_eval_combination,
                   "Evaluate forecast combination")
  register_command("combine", cmd_combine,
                   "Combine forecasts from multiple models")

  # Report commands
  register_command("report", cmd_report,
                   "Generate interactive HTML reports")

  # Interactive mode
  register_command("interactive", function(args) {
    shell <- PrevCargaShell$new()
    shell$run()
  }, "Launch interactive shell mode")
}


# Initialize on package load
.onLoad <- function(libname, pkgname) {
  init_cli_commands()
}
```

### Global Options Parser

```r
#' Create base option parser with common options
#'
#' @param usage Usage string
#' @return optparse::OptionParser
#' @noRd
create_base_parser <- function(usage) {
  parser <- optparse::OptionParser(usage = usage)

  # Add common options
  parser <- optparse::add_option(
    parser, c("-c", "--config"),
    type = "character",
    default = "config/config.yaml",
    help = "Path to configuration YAML file [default: %default]"
  )

  parser <- optparse::add_option(
    parser, c("-v", "--verbose"),
    action = "store_true",
    default = FALSE,
    help = "Enable verbose output"
  )

  parser <- optparse::add_option(
    parser, c("-q", "--quiet"),
    action = "store_true",
    default = FALSE,
    help = "Suppress non-error output"
  )

  parser
}


#' Parse common options and load config
#'
#' @param opts Parsed options
#' @return ConfigManager instance
#' @noRd
load_config_from_opts <- function(opts) {
  config <- ConfigManager$new()
  config$load(opts$config)

  # Set log level based on verbosity
  if (opts$verbose) {
    config$set_runtime("logging.level", "DEBUG")
  } else if (opts$quiet) {
    config$set_runtime("logging.level", "ERROR")
  }

  config
}
```

### Usage Example

```r
# From command line:
# Rscript -e "prevcargaons::main()" train --config config.yaml --areas RJ,SP

# From R console:
main(c("train", "--config", "config.yaml", "--areas", "RJ,SP"))

# Show help:
main(c("--help"))

# Show version:
main(c("--version"))

# Launch interactive mode:
main(c("interactive"))

# Get command help:
main(c("train", "--help"))
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | main() with no args | Shows help |
| TC-002 | main() with --help | Shows help |
| TC-003 | main() with --version | Shows version |
| TC-004 | main() with unknown command | Error message |
| TC-005 | register_command() | Command registered |
| TC-006 | get_command_handler() existing | Returns handler |
| TC-007 | get_command_handler() missing | Returns NULL |
| TC-008 | execute_command() valid | Handler called |
| TC-009 | execute_command() invalid | Error thrown |
| TC-010 | create_base_parser() | Parser created |

---

## Dependencies

- PC-057-08: ConfigManager

---

## Definition of Done

- [ ] main() entry point implemented
- [ ] Command registry working
- [ ] Help and version commands
- [ ] Base parser with common options
- [ ] Error handling with user-friendly messages
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Uses `optparse` package for argument parsing
- Commands are registered on package load
- Error messages should be user-friendly, not stack traces
- Consider adding shell completion support in future
