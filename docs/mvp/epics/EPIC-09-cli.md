# EPIC-09: CLI

**Duration:** 2 weeks
**Dependencies:** EPIC-08
**Reference:** [MVP Plan R - Phase 9](../mvp-plan-r.md#phase-9-cli-2-weeks)

---

## Objective

Create the command-line interface with both argument mode (for pipelines) and interactive mode (for development), providing all necessary commands for training, prediction, backtesting, and evaluation.

---

## Scope

This epic covers:
- CLI entry point with `optparse`
- Commands: train, predict, backtest, forecast
- Commands: gen-features, eval-features
- Commands: eval-model, eval-combination, combine
- Interactive shell mode
- Progress bars, colors, and display utilities

**Out of Scope:**
- Plugin implementations
- Web interface

---

## Tasks

### T-09.1: Create CLI Entry Point
- [ ] Create `R/cli/main.R`
- [ ] Implement main entry point:
  ```r
  #' @export
  main = function(args = commandArgs(trailingOnly = TRUE)) {
    if (length(args) == 0 || args[1] == "interactive") {
      # Launch interactive shell
      shell <- PrevCargaShell$new()
      shell$run()
    } else {
      # Parse and execute command
      execute_command(args)
    }
  }

  execute_command = function(args) {
    command <- args[1]
    command_args <- args[-1]

    handler <- get_command_handler(command)
    if (is.null(handler)) {
      stop(sprintf("Unknown command: %s", command))
    }

    handler(command_args)
  }
  ```

### T-09.2: Create Train Command
- [ ] Create `R/cli/commands/train.R`
- [ ] Implement train command handler:
  ```r
  cmd_train = function(args) {
    parser <- optparse::OptionParser(
      usage = "prevcarga train [options]"
    )
    parser <- optparse::add_option(parser, "--config", type = "character",
                                   help = "Path to config YAML")
    parser <- optparse::add_option(parser, "--storage-backend",
                                   type = "character", default = "local")
    parser <- optparse::add_option(parser, "--areas", type = "character",
                                   help = "Comma-separated areas")
    parser <- optparse::add_option(parser, "--models", type = "character",
                                   help = "Comma-separated models")
    parser <- optparse::add_option(parser, "--start-date", type = "character")
    parser <- optparse::add_option(parser, "--end-date", type = "character")
    parser <- optparse::add_option(parser, "--version", type = "character")
    parser <- optparse::add_option(parser, "--parallel", type = "integer",
                                   default = 4)

    opts <- optparse::parse_args(parser, args)

    # Validate options
    validate_train_options(opts)

    # Execute training
    config <- ConfigManager$new()$load(opts$config)
    storage <- StorageFactory$new()$from_config(config)

    workflow <- TrainWorkflow$new(config, storage)
    workflow$run(
      areas = strsplit(opts$areas, ",")[[1]],
      models = strsplit(opts$models, ",")[[1]],
      start_date = opts$`start-date`,
      end_date = opts$`end-date`,
      version = opts$version
    )
  }
  ```

### T-09.3: Create Predict Command
- [ ] Create `R/cli/commands/predict.R`
- [ ] Implement predict command handler with options:
  - `--date`: Target date
  - `--mode`: batch | intraday
  - `--areas`: Comma-separated areas (or "all")
  - `--models`: Comma-separated models (or "all")
  - `--reconcile`: Enable reconciliation
  - `--output`: Output path

### T-09.4: Create Backtest Command
- [ ] Create `R/cli/commands/backtest.R`
- [ ] Implement backtest command handler with options:
  - `--start-date`, `--end-date`: Backtest period
  - `--retrain-intervals`: Comma-separated intervals
  - `--walk-forward`: Enable walk-forward validation
  - `--models`: Models to backtest
  - `--report`: Output report path

### T-09.5: Create Feature Commands
- [ ] Create `R/cli/commands/features.R`
- [ ] Implement `gen-features` command:
  - Generate features for specified model/areas
  - Save to storage
- [ ] Implement `eval-features` command:
  - Evaluate feature importance
  - Methods: shap, correlation, stability

### T-09.6: Create Evaluation Commands
- [ ] Create `R/cli/commands/evaluate.R`
- [ ] Implement `eval-model` command
- [ ] Implement `eval-combination` command
- [ ] Implement `combine` command

### T-09.7: Create Interactive Shell
- [ ] Create `R/cli/shell.R`
- [ ] Implement `PrevCargaShell` R6 class (from MVP plan):
  ```r
  PrevCargaShell <- R6::R6Class(
    "PrevCargaShell",
    private = list(
      config = NULL,
      storage = NULL,
      running = FALSE,
      colors = list(
        reset = "\033[0m",
        bold = "\033[1m",
        red = "\033[31m",
        green = "\033[32m",
        yellow = "\033[33m",
        cyan = "\033[36m"
      ),
      symbols = list(
        check = "\u2713",
        cross = "\u2717",
        arrow = "\u2192",
        bullet = "\u2022"
      )
    ),
    public = list(
      run = function() {
        private$running <- TRUE
        self$print_banner()
        while (private$running) {
          self$print_prompt()
          input <- readline()
          self$execute_command(input)
        }
      },

      print_banner = function() { ... },
      print_prompt = function() { ... },
      execute_command = function(input) { ... }
    )
  )
  ```

### T-09.8: Create Display Utilities
- [ ] Create `R/cli/display.R`
- [ ] Implement progress bar:
  ```r
  progress_bar = function(current, total, width = 40) {
    pct <- current / total
    filled <- floor(pct * width)
    bar <- paste0(
      "[",
      strrep("\u2588", filled),
      strrep("\u2591", width - filled),
      "]"
    )
    sprintf("%s %3.0f%%", bar, pct * 100)
  }
  ```
- [ ] Implement sparkline:
  ```r
  sparkline = function(values) {
    chars <- c("\u2581", "\u2582", "\u2583", "\u2584",
               "\u2585", "\u2586", "\u2587", "\u2588")
    # Scale and map to characters
  }
  ```

### T-09.9: Create Input Validators
- [ ] Create `R/cli/validators.R`
- [ ] Implement validators for:
  - Date formats
  - Area codes
  - Model names
  - File paths

### T-09.10: Create Shell Entry Script
- [ ] Create `inst/shell/prevcarga`
- [ ] Make executable shell script:
  ```bash
  #!/usr/bin/env Rscript
  library(prevcargaons)
  prevcargaons::main()
  ```

### T-09.11: Write Tests
- [ ] Create `tests/testthat/test-cli.R`
- [ ] Test command parsing
- [ ] Test validators
- [ ] Test shell commands
- [ ] E2E tests for main workflows

### T-09.12: Create Report Command
- [ ] Create `R/cli/commands/report.R`
- [ ] Implement report command handler:
  ```r
  cmd_report = function(args) {
    parser <- optparse::OptionParser(usage = "prevcarga report <type> [options]")
    parser <- optparse::add_option(parser, "--output", "-o", type = "character",
                                   help = "Output HTML path")
    parser <- optparse::add_option(parser, "--areas", type = "character",
                                   help = "Comma-separated areas")
    parser <- optparse::add_option(parser, "--date", type = "character",
                                   help = "Target date")
    parser <- optparse::add_option(parser, "--period", type = "character",
                                   help = "Date range (start:end)")
    parser <- optparse::add_option(parser, "--models", type = "character",
                                   help = "Comma-separated models")
    parser <- optparse::add_option(parser, "--backtest-id", type = "character",
                                   help = "Backtest identifier")
    parser <- optparse::add_option(parser, "--area", type = "character",
                                   help = "Single area for dashboard")
    parser <- optparse::add_option(parser, "--model", type = "character",
                                   help = "Single model for drift report")
    parser <- optparse::add_option(parser, "--baseline-date", type = "character",
                                   help = "Baseline date for drift comparison")
    parser <- optparse::add_option(parser, "--open", action = "store_true",
                                   help = "Open report in browser after generation")

    opts <- optparse::parse_args(parser, args)
    report_type <- args[1]

    # Validate report type
    valid_types <- c("forecast", "backtest", "compare", "dashboard", "drift")
    if (!report_type %in% valid_types) {
      stop(sprintf("Invalid report type: %s. Valid: %s",
                   report_type, paste(valid_types, collapse = ", ")))
    }

    # Load data based on report type
    data <- load_report_data(report_type, opts)

    # Generate report
    reporter <- HighcharterReporter$new()
    reporter$generate(report_type, data, opts$output)

    cli::cli_alert_success("Report generated: {opts$output}")

    if (opts$open) {
      browseURL(opts$output)
    }
  }

  load_report_data = function(report_type, opts) {
    switch(report_type,
      "forecast" = load_forecast_report_data(opts),
      "backtest" = load_backtest_report_data(opts),
      "compare" = load_comparison_report_data(opts),
      "dashboard" = load_dashboard_report_data(opts),
      "drift" = load_drift_report_data(opts)
    )
  }
  ```
- [ ] Implement data loaders for each report type
- [ ] Add report subcommands to main parser

---

## Acceptance Criteria

- [ ] `prevcarga train` command works with all options
- [ ] `prevcarga predict` supports batch and intraday modes
- [ ] `prevcarga backtest` evaluates retraining intervals
- [ ] `prevcarga gen-features` generates features
- [ ] `prevcarga report` generates interactive HTML reports
- [ ] `prevcarga interactive` launches shell mode
- [ ] Progress bars and colors display correctly
- [ ] Input validation prevents invalid parameters
- [ ] Help texts are clear and complete
- [ ] Tests pass with ≥80% coverage

---

## Definition of Done

- [ ] All tasks completed
- [ ] Code reviewed and merged
- [ ] Tests passing (≥80% coverage)
- [ ] Documentation complete with roxygen2
- [ ] E2E tests for critical paths

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `R/cli/main.R` | Create | CLI entry point |
| `R/cli/commands/train.R` | Create | Train command |
| `R/cli/commands/predict.R` | Create | Predict command |
| `R/cli/commands/backtest.R` | Create | Backtest command |
| `R/cli/commands/features.R` | Create | Feature commands |
| `R/cli/commands/evaluate.R` | Create | Evaluation commands |
| `R/cli/commands/report.R` | Create | Report generation command |
| `R/cli/shell.R` | Create | Interactive shell |
| `R/cli/display.R` | Create | Display utilities |
| `R/cli/validators.R` | Create | Input validators |
| `inst/shell/prevcarga` | Create | Shell entry script |
| `tests/testthat/test-cli.R` | Create | CLI tests |

---

## CLI Commands Reference

```bash
# Training
prevcarga train --config config.yaml --areas RJ,SP --models lgbm --version 1.0.0

# Prediction
prevcarga predict --date 2025-01-17 --mode batch --areas all --reconcile

# Backtesting
prevcarga backtest --start-date 2024-01-01 --end-date 2024-12-31 --walk-forward

# Features
prevcarga gen-features --model lgbm --areas RJ --start-date 2024-01-01

# Evaluation
prevcarga eval-model --model-path models/lgbm/latest --test-data test.parquet

# Report Generation (interactive HTML with highcharter)
prevcarga report forecast --date 2025-01-17 --areas RJ,SP -o forecast.html
prevcarga report backtest --backtest-id bt_2024Q4 -o backtest.html
prevcarga report compare --models lgbm,rf,hw --period 2024-10-01:2024-12-31 -o comparison.html
prevcarga report dashboard --area RJ --period 2024-01-01:2024-12-31 -o dashboard_RJ.html
prevcarga report drift --model lgbm --baseline-date 2024-06-01 -o drift.html --open

# Interactive
prevcarga interactive
```
