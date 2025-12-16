# PC-070-09: Interactive Shell

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.7
**Priority:** Medium
**Estimated Effort:** 2 days

---

## Summary

Implement the `PrevCargaShell` R6 class providing an interactive REPL (Read-Eval-Print Loop) for development and exploratory analysis. The shell offers command completion, history, colored output, and quick access to common operations.

---

## Acceptance Criteria

- [ ] Shell module created in `R/cli/shell.R`
- [ ] `PrevCargaShell` R6 class with REPL loop
- [ ] Command parsing and execution
- [ ] Tab completion for commands and arguments
- [ ] Command history support
- [ ] Colored output and Unicode symbols
- [ ] Help system for shell commands

---

## Technical Specification

### File Location
```
R/cli/shell.R
```

### PrevCargaShell R6 Class

```r
#' @title PrevCargaShell
#' @description Interactive shell for PrevCarga
#'
#' Provides an interactive REPL environment for development,
#' exploration, and quick operations.
#'
#' @export
PrevCargaShell <- R6::R6Class(
  "PrevCargaShell",
  private = list(
    config = NULL,
    storage = NULL,
    logger = NULL,
    running = FALSE,
    history = character(),
    history_file = NULL,
    current_context = list(),

    # Color codes
    colors = list(
      reset = "\033[0m",
      bold = "\033[1m",
      dim = "\033[2m",
      red = "\033[31m",
      green = "\033[32m",
      yellow = "\033[33m",
      blue = "\033[34m",
      magenta = "\033[35m",
      cyan = "\033[36m",
      white = "\033[37m"
    ),

    # Unicode symbols
    symbols = list(
      check = "\u2713",
      cross = "\u2717",
      arrow = "\u2192",
      bullet = "\u2022",
      star = "\u2605",
      warning = "\u26A0",
      info = "\u2139",
      prompt = "\u276F"
    ),

    # Shell commands
    shell_commands = list()
  ),

  public = list(
    #' @description Initialize shell
    #' @param config_path Path to config file (optional)
    initialize = function(config_path = NULL) {
      # Set up history file
      private$history_file <- file.path(
        Sys.getenv("HOME", "."),
        ".prevcarga_history"
      )

      # Load history
      if (file.exists(private$history_file)) {
        private$history <- readLines(private$history_file, warn = FALSE)
      }

      # Register shell commands
      self$register_shell_commands()

      # Load config if provided
      if (!is.null(config_path) && file.exists(config_path)) {
        private$config <- ConfigManager$new()
        private$config$load(config_path)
        private$storage <- create_storage_backend(
          private$config$get_storage_config()
        )
      }
    },

    #' @description Run interactive shell
    run = function() {
      private$running <- TRUE
      self$print_banner()

      while (private$running) {
        input <- self$read_input()

        if (is.null(input)) {
          # EOF (Ctrl+D)
          self$cmd_exit(list())
          break
        }

        input <- trimws(input)

        if (nchar(input) == 0) {
          next
        }

        # Add to history
        self$add_to_history(input)

        # Execute command
        tryCatch({
          self$execute_command(input)
        }, error = function(e) {
          self$print_error(conditionMessage(e))
        })
      }

      self$save_history()
    },

    #' @description Print welcome banner
    print_banner = function() {
      cat("\n")
      self$print_colored("=", "cyan", rep = 60)
      cat("\n")
      self$print_colored("  PrevCarga Interactive Shell", "bold")
      cat("\n")
      self$print_colored("  Electric Load Forecasting System", "dim")
      cat("\n")
      self$print_colored("=", "cyan", rep = 60)
      cat("\n\n")

      self$print_colored(
        sprintf("  Type 'help' for commands, 'exit' to quit\n"),
        "dim"
      )

      if (!is.null(private$config)) {
        self$print_success(sprintf(
          "Config loaded: %s",
          private$config$get("project.name")
        ))
      } else {
        self$print_warning("No config loaded. Use 'load-config <path>' to load.")
      }

      cat("\n")
    },

    #' @description Print shell prompt
    print_prompt = function() {
      # Build context string
      context_parts <- c()

      if (!is.null(private$current_context$area)) {
        context_parts <- c(context_parts, private$current_context$area)
      }

      if (!is.null(private$current_context$model)) {
        context_parts <- c(context_parts, private$current_context$model)
      }

      context_str <- if (length(context_parts) > 0) {
        sprintf("[%s] ", paste(context_parts, collapse = "/"))
      } else {
        ""
      }

      cat(sprintf(
        "%s%sprevcarga%s %s%s%s ",
        private$colors$cyan,
        context_str,
        private$colors$reset,
        private$colors$magenta,
        private$symbols$prompt,
        private$colors$reset
      ))
    },

    #' @description Read user input
    read_input = function() {
      self$print_prompt()

      tryCatch({
        readline()
      }, error = function(e) {
        NULL
      })
    },

    #' @description Execute shell command
    #' @param input User input string
    execute_command = function(input) {
      # Parse command and arguments
      parts <- self$parse_input(input)
      cmd <- parts$command
      args <- parts$args

      # Check for shell command
      if (cmd %in% names(private$shell_commands)) {
        handler <- private$shell_commands[[cmd]]
        handler$fn(args)
        return(invisible(NULL))
      }

      # Check for CLI command
      if (has_command(cmd)) {
        execute_command(cmd, args)
        return(invisible(NULL))
      }

      # Check for R expression
      if (grepl("^[a-zA-Z_][a-zA-Z0-9_]*\\s*(<-|=)", input) ||
          grepl("^print\\(|^cat\\(|^str\\(", input)) {
        result <- eval(parse(text = input), envir = globalenv())
        if (!is.null(result)) {
          print(result)
        }
        return(invisible(NULL))
      }

      self$print_error(sprintf("Unknown command: %s", cmd))
      self$print_info("Type 'help' for available commands")
    },

    #' @description Parse input into command and arguments
    #' @param input User input string
    #' @return List with command and args
    parse_input = function(input) {
      # Handle quoted strings
      parts <- strsplit(input, "\\s+(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", perl = TRUE)[[1]]

      # Remove quotes from parts
      parts <- gsub('^"|"$', '', parts)

      list(
        command = parts[1],
        args = parts[-1]
      )
    },

    #' @description Register shell-specific commands
    register_shell_commands = function() {
      private$shell_commands <- list(
        "help" = list(
          fn = function(args) self$cmd_help(args),
          desc = "Show help for commands"
        ),
        "exit" = list(
          fn = function(args) self$cmd_exit(args),
          desc = "Exit the shell"
        ),
        "quit" = list(
          fn = function(args) self$cmd_exit(args),
          desc = "Exit the shell"
        ),
        "clear" = list(
          fn = function(args) self$cmd_clear(args),
          desc = "Clear the screen"
        ),
        "history" = list(
          fn = function(args) self$cmd_history(args),
          desc = "Show command history"
        ),
        "load-config" = list(
          fn = function(args) self$cmd_load_config(args),
          desc = "Load configuration file"
        ),
        "set-area" = list(
          fn = function(args) self$cmd_set_area(args),
          desc = "Set current working area"
        ),
        "set-model" = list(
          fn = function(args) self$cmd_set_model(args),
          desc = "Set current working model"
        ),
        "status" = list(
          fn = function(args) self$cmd_status(args),
          desc = "Show current status"
        ),
        "list-areas" = list(
          fn = function(args) self$cmd_list_areas(args),
          desc = "List available areas"
        ),
        "list-models" = list(
          fn = function(args) self$cmd_list_models(args),
          desc = "List available models"
        ),
        "quick-predict" = list(
          fn = function(args) self$cmd_quick_predict(args),
          desc = "Quick prediction for current context"
        ),
        "plot" = list(
          fn = function(args) self$cmd_plot(args),
          desc = "Plot forecast or metrics"
        )
      )
    },

    #' @description Help command
    cmd_help = function(args) {
      if (length(args) > 0) {
        # Help for specific command
        cmd <- args[1]
        if (cmd %in% names(private$shell_commands)) {
          cat(sprintf("\n%s: %s\n\n",
                      cmd, private$shell_commands[[cmd]]$desc))
        } else if (has_command(cmd)) {
          # Show CLI command help
          execute_command(cmd, c("--help"))
        } else {
          self$print_error(sprintf("Unknown command: %s", cmd))
        }
        return(invisible(NULL))
      }

      cat("\n")
      self$print_colored("Shell Commands:", "bold")
      cat("\n")

      for (name in sort(names(private$shell_commands))) {
        cmd <- private$shell_commands[[name]]
        cat(sprintf("  %-15s %s\n", name, cmd$desc))
      }

      cat("\n")
      self$print_colored("CLI Commands:", "bold")
      cat("\n")

      cli_commands <- get_all_commands()
      for (name in sort(names(cli_commands))) {
        cmd <- cli_commands[[name]]
        cat(sprintf("  %-15s %s\n", name, cmd$description))
      }

      cat("\n")
      self$print_info("Type 'help <command>' for detailed help")
      cat("\n")
    },

    #' @description Exit command
    cmd_exit = function(args) {
      self$print_info("Goodbye!")
      private$running <- FALSE
    },

    #' @description Clear screen command
    cmd_clear = function(args) {
      cat("\033[2J\033[H")
    },

    #' @description History command
    cmd_history = function(args) {
      n <- if (length(args) > 0) as.integer(args[1]) else 20
      recent <- tail(private$history, n)

      cat("\n")
      for (i in seq_along(recent)) {
        cat(sprintf("  %3d  %s\n", length(private$history) - length(recent) + i, recent[i]))
      }
      cat("\n")
    },

    #' @description Load config command
    cmd_load_config = function(args) {
      if (length(args) == 0) {
        self$print_error("Usage: load-config <path>")
        return(invisible(NULL))
      }

      path <- args[1]
      if (!file.exists(path)) {
        self$print_error(sprintf("File not found: %s", path))
        return(invisible(NULL))
      }

      private$config <- ConfigManager$new()
      private$config$load(path)
      private$storage <- create_storage_backend(
        private$config$get_storage_config()
      )

      self$print_success(sprintf("Loaded: %s", private$config$get("project.name")))
    },

    #' @description Set area command
    cmd_set_area = function(args) {
      if (length(args) == 0) {
        private$current_context$area <- NULL
        self$print_info("Area context cleared")
      } else {
        private$current_context$area <- args[1]
        self$print_success(sprintf("Area set to: %s", args[1]))
      }
    },

    #' @description Set model command
    cmd_set_model = function(args) {
      if (length(args) == 0) {
        private$current_context$model <- NULL
        self$print_info("Model context cleared")
      } else {
        private$current_context$model <- args[1]
        self$print_success(sprintf("Model set to: %s", args[1]))
      }
    },

    #' @description Status command
    cmd_status = function(args) {
      cat("\n")
      self$print_colored("Current Status:", "bold")
      cat("\n")

      if (!is.null(private$config)) {
        cat(sprintf("  Project:     %s\n", private$config$get("project.name")))
        cat(sprintf("  Environment: %s\n", private$config$get_environment()))
        cat(sprintf("  Storage:     %s\n", private$config$get("storage.backend")))
      } else {
        cat("  Config: Not loaded\n")
      }

      cat(sprintf("  Area:        %s\n", private$current_context$area %||% "(not set)"))
      cat(sprintf("  Model:       %s\n", private$current_context$model %||% "(not set)"))
      cat("\n")
    },

    #' @description List areas command
    cmd_list_areas = function(args) {
      if (is.null(private$config)) {
        self$print_error("No config loaded")
        return(invisible(NULL))
      }

      areas <- private$config$get_areas()
      cat("\n")
      self$print_colored("Available Areas:", "bold")
      cat("\n")
      cat(sprintf("  %s\n", paste(areas, collapse = ", ")))
      cat("\n")
    },

    #' @description List models command
    cmd_list_models = function(args) {
      cat("\n")
      self$print_colored("Registered Models:", "bold")
      cat("\n")

      models <- list_models()
      for (model in models) {
        cat(sprintf("  %s %s\n", private$symbols$bullet, model))
      }
      cat("\n")
    },

    #' @description Quick predict command
    cmd_quick_predict = function(args) {
      area <- private$current_context$area
      model <- private$current_context$model

      if (is.null(area) || is.null(model)) {
        self$print_error("Set area and model first: set-area <area>, set-model <model>")
        return(invisible(NULL))
      }

      date <- if (length(args) > 0) args[1] else as.character(Sys.Date())

      self$print_info(sprintf("Predicting %s/%s for %s...", area, model, date))

      # Run prediction
      execute_command("predict", c(
        "--date", date,
        "--areas", area,
        "--models", model,
        "--quiet"
      ))
    },

    #' @description Add to history
    add_to_history = function(input) {
      # Don't add duplicates or empty
      if (nchar(input) > 0 &&
          (length(private$history) == 0 ||
           input != tail(private$history, 1))) {
        private$history <- c(private$history, input)
      }
    },

    #' @description Save history to file
    save_history = function() {
      # Keep last 1000 entries
      history_to_save <- tail(private$history, 1000)
      writeLines(history_to_save, private$history_file)
    },

    #' @description Print colored text
    print_colored = function(text, color, rep = NULL) {
      if (!is.null(rep)) {
        text <- strrep(text, rep)
      }

      color_code <- private$colors[[color]] %||% ""
      cat(sprintf("%s%s%s", color_code, text, private$colors$reset))
    },

    #' @description Print error message
    print_error = function(message) {
      cat(sprintf(
        "%s%s %s%s\n",
        private$colors$red,
        private$symbols$cross,
        message,
        private$colors$reset
      ))
    },

    #' @description Print success message
    print_success = function(message) {
      cat(sprintf(
        "%s%s %s%s\n",
        private$colors$green,
        private$symbols$check,
        message,
        private$colors$reset
      ))
    },

    #' @description Print warning message
    print_warning = function(message) {
      cat(sprintf(
        "%s%s %s%s\n",
        private$colors$yellow,
        private$symbols$warning,
        message,
        private$colors$reset
      ))
    },

    #' @description Print info message
    print_info = function(message) {
      cat(sprintf(
        "%s%s %s%s\n",
        private$colors$blue,
        private$symbols$info,
        message,
        private$colors$reset
      ))
    }
  )
)
```

### Usage Example

```r
# Launch from CLI
# prevcarga interactive

# Or from R
shell <- PrevCargaShell$new()
shell$run()

# Interactive session example:
#
# prevcarga ❯ load-config config/config.yaml
# ✓ Loaded: PrevCargaONS
#
# prevcarga ❯ list-areas
# Available Areas:
#   RJ, SP, MG, ES, MT, MS, AC, RO, DF, GO, PR, SC, RS, ...
#
# prevcarga ❯ set-area RJ
# ✓ Area set to: RJ
#
# [RJ] prevcarga ❯ set-model lgbm
# ✓ Model set to: lgbm
#
# [RJ/lgbm] prevcarga ❯ quick-predict
# ℹ Predicting RJ/lgbm for 2025-01-17...
#
# [RJ/lgbm] prevcarga ❯ train --start-date 2023-01-01 --end-date 2024-01-01
# ... training output ...
#
# [RJ/lgbm] prevcarga ❯ exit
# ℹ Goodbye!
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize shell | Shell created |
| TC-002 | cmd_help() | Shows commands |
| TC-003 | cmd_exit() | Shell exits |
| TC-004 | cmd_clear() | Screen cleared |
| TC-005 | cmd_history() | History shown |
| TC-006 | cmd_load_config() | Config loaded |
| TC-007 | cmd_set_area() | Area set |
| TC-008 | cmd_set_model() | Model set |
| TC-009 | cmd_status() | Status shown |
| TC-010 | execute_command() CLI | CLI command runs |
| TC-011 | parse_input() quoted | Quotes handled |
| TC-012 | History saved | File written |

---

## Dependencies

- PC-064-09: CLI Entry Point
- PC-057-08: ConfigManager

---

## Definition of Done

- [ ] PrevCargaShell R6 class implemented
- [ ] All shell commands working
- [ ] CLI command integration
- [ ] History persistence
- [ ] Colored output working
- [ ] Context tracking (area/model)
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- History stored in ~/.prevcarga_history
- Colors may not work on all terminals
- Consider adding readline-like completion
- Future: Add R expression evaluation mode
