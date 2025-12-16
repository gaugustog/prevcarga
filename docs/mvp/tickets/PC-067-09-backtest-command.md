# PC-067-09: Backtest Command

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.4
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `backtest` CLI command for running walk-forward backtesting simulations. This command evaluates model performance over historical periods and compares different retraining intervals.

---

## Acceptance Criteria

- [ ] Backtest command module created in `R/cli/commands/backtest.R`
- [ ] Date range specification for backtest period
- [ ] Retraining interval configuration
- [ ] Walk-forward validation toggle
- [ ] Model and area selection
- [ ] Report generation option
- [ ] Progress output during simulation

---

## Technical Specification

### File Location
```
R/cli/commands/backtest.R
```

### Backtest Command Handler

```r
#' Backtest Command Handler
#'
#' Handles the 'prevcarga backtest' command for running backtesting simulations.
#'
#' @param args Command line arguments
#' @noRd
cmd_backtest <- function(args) {
  # Create parser
  parser <- create_backtest_parser()

  # Parse arguments
  opts <- optparse::parse_args(parser, args, positional_arguments = FALSE)

  # Validate options
  validate_backtest_options(opts)

  # Execute backtesting
  execute_backtest(opts)
}


#' Create Backtest Command Parser
#' @noRd
create_backtest_parser <- function() {
  parser <- optparse::OptionParser(
    usage = "prevcarga backtest [options]",
    description = "Run walk-forward backtesting to evaluate model performance."
  )

  # Configuration
  parser <- optparse::add_option(
    parser, c("-c", "--config"),
    type = "character",
    default = "config/config.yaml",
    help = "Path to configuration YAML file [default: %default]"
  )

  # Date range
  parser <- optparse::add_option(
    parser, "--start-date",
    type = "character",
    help = "Backtest start date (YYYY-MM-DD) [required]"
  )

  parser <- optparse::add_option(
    parser, "--end-date",
    type = "character",
    help = "Backtest end date (YYYY-MM-DD) [required]"
  )

  # Training window
  parser <- optparse::add_option(
    parser, "--train-window",
    type = "integer",
    default = 365,
    help = "Training window size in days [default: %default]"
  )

  # Retraining intervals
  parser <- optparse::add_option(
    parser, "--retrain-intervals",
    type = "character",
    default = "1,3,7,10,15",
    help = "Comma-separated retraining intervals in days [default: %default]"
  )

  # Walk-forward
  parser <- optparse::add_option(
    parser, "--walk-forward",
    action = "store_true",
    default = TRUE,
    help = "Enable walk-forward validation [default: TRUE]"
  )

  parser <- optparse::add_option(
    parser, "--no-walk-forward",
    action = "store_true",
    default = FALSE,
    help = "Disable walk-forward validation (fixed training window)"
  )

  # Areas
  parser <- optparse::add_option(
    parser, c("-a", "--areas"),
    type = "character",
    default = "all",
    help = "Comma-separated area codes or 'all' [default: %default]"
  )

  # Models
  parser <- optparse::add_option(
    parser, c("-m", "--models"),
    type = "character",
    default = "all",
    help = "Comma-separated model names or 'all' [default: %default]"
  )

  # Horizons
  parser <- optparse::add_option(
    parser, "--horizons",
    type = "character",
    default = NULL,
    help = "Comma-separated horizons to evaluate [default: from config]"
  )

  # Metrics
  parser <- optparse::add_option(
    parser, "--metrics",
    type = "character",
    default = "mape,mae,rmse",
    help = "Comma-separated metrics to calculate [default: %default]"
  )

  # Backtest ID
  parser <- optparse::add_option(
    parser, "--backtest-id",
    type = "character",
    default = NULL,
    help = "Unique backtest identifier [default: auto-generated]"
  )

  # Output
  parser <- optparse::add_option(
    parser, c("-o", "--output-dir"),
    type = "character",
    default = NULL,
    help = "Output directory for results [default: backtests/<backtest-id>]"
  )

  # Report
  parser <- optparse::add_option(
    parser, "--report",
    type = "character",
    default = NULL,
    help = "Generate HTML report at specified path"
  )

  parser <- optparse::add_option(
    parser, "--open-report",
    action = "store_true",
    default = FALSE,
    help = "Open report in browser after generation"
  )

  # Parallel
  parser <- optparse::add_option(
    parser, c("-j", "--parallel"),
    type = "integer",
    default = 4,
    help = "Number of parallel workers [default: %default]"
  )

  # Verbosity
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
    help = "Suppress progress output"
  )

  # Dry run
  parser <- optparse::add_option(
    parser, "--dry-run",
    action = "store_true",
    default = FALSE,
    help = "Validate options and show simulation plan"
  )

  parser
}


#' Validate Backtest Options
#' @noRd
validate_backtest_options <- function(opts) {
  # Required: start-date
  if (is.null(opts$`start-date`)) {
    stop("--start-date is required")
  }

  # Required: end-date
  if (is.null(opts$`end-date`)) {
    stop("--end-date is required")
  }

  # Validate date formats
  start_date <- tryCatch(
    as.Date(opts$`start-date`),
    error = function(e) stop("Invalid --start-date format. Use YYYY-MM-DD")
  )

  end_date <- tryCatch(
    as.Date(opts$`end-date`),
    error = function(e) stop("Invalid --end-date format. Use YYYY-MM-DD")
  )

  # Validate date range
  if (start_date >= end_date) {
    stop("--start-date must be before --end-date")
  }

  # Validate train window
  if (opts$`train-window` < 30) {
    stop("--train-window must be at least 30 days")
  }

  # Validate retraining intervals
  intervals <- as.integer(strsplit(opts$`retrain-intervals`, ",")[[1]])
  if (any(is.na(intervals)) || any(intervals < 1)) {
    stop("--retrain-intervals must be positive integers")
  }

  # Validate config file
  if (!file.exists(opts$config)) {
    stop(sprintf("Configuration file not found: %s", opts$config))
  }

  # Validate metrics
  valid_metrics <- c("mape", "mae", "rmse", "mbe", "smape")
  metrics <- tolower(strsplit(opts$metrics, ",")[[1]])
  invalid <- setdiff(metrics, valid_metrics)
  if (length(invalid) > 0) {
    stop(sprintf("Invalid metrics: %s. Valid: %s",
                 paste(invalid, collapse = ", "),
                 paste(valid_metrics, collapse = ", ")))
  }

  invisible(TRUE)
}


#' Execute Backtesting
#' @noRd
execute_backtest <- function(opts) {
  # Load configuration
  cli_info(sprintf("Loading configuration: %s", opts$config))
  config <- ConfigManager$new()
  config$load(opts$config)

  # Generate backtest ID if not provided
  backtest_id <- opts$`backtest-id` %||%
    sprintf("bt_%s", format(Sys.time(), "%Y%m%d_%H%M%S"))
  cli_info(sprintf("Backtest ID: %s", backtest_id))

  # Parse dates
  start_date <- as.Date(opts$`start-date`)
  end_date <- as.Date(opts$`end-date`)
  cli_info(sprintf("Period: %s to %s", start_date, end_date))

  # Parse retraining intervals
  retrain_intervals <- as.integer(strsplit(opts$`retrain-intervals`, ",")[[1]])
  cli_info(sprintf("Retraining intervals: %s days",
                   paste(retrain_intervals, collapse = ", ")))

  # Walk-forward mode
  walk_forward <- !opts$`no-walk-forward`
  cli_info(sprintf("Walk-forward: %s", if (walk_forward) "Yes" else "No"))

  # Resolve areas
  areas <- resolve_areas(opts$areas, config)
  cli_info(sprintf("Areas: %s", paste(areas, collapse = ", ")))

  # Resolve models
  models <- resolve_models(opts$models, config)
  cli_info(sprintf("Models: %s", paste(models, collapse = ", ")))

  # Resolve horizons
  horizons <- if (!is.null(opts$horizons)) {
    as.integer(strsplit(opts$horizons, ",")[[1]])
  } else {
    config$get("prediction.horizons") %||% 0:8
  }

  # Parse metrics
  metrics <- tolower(strsplit(opts$metrics, ",")[[1]])

  # Create storage and logger
  storage <- create_storage_from_opts(opts, config)
  logger <- create_logger_from_opts(opts, config)

  # Calculate simulation statistics
  n_days <- as.integer(end_date - start_date) + 1
  n_intervals <- length(retrain_intervals)
  n_simulations <- n_days * n_intervals * length(areas) * length(models)

  cli_info(sprintf("Simulation days: %d", n_days))
  cli_info(sprintf("Total simulations: %d", n_simulations))

  # Dry run check
  if (opts$`dry-run`) {
    cat("\n")
    cli_header("Backtest Simulation Plan")
    cat(sprintf("\n  Days to simulate: %d\n", n_days))
    cat(sprintf("  Retraining intervals: %d\n", n_intervals))
    cat(sprintf("  Areas: %d\n", length(areas)))
    cat(sprintf("  Models: %d\n", length(models)))
    cat(sprintf("  Total simulations: %d\n", n_simulations))
    cat(sprintf("  Estimated time: %.1f hours\n",
                n_simulations * 0.5 / 60))  # ~30s per simulation
    cat("\n")
    cli_success("Dry run complete. Configuration is valid.")
    return(invisible(NULL))
  }

  # Create and run workflow
  cli_header("Starting Backtest Simulation")

  workflow <- BacktestWorkflow$new(config, storage, logger)

  result <- workflow$run(
    start_date = start_date,
    end_date = end_date,
    models = models,
    areas = areas,
    retrain_intervals = retrain_intervals,
    walk_forward = walk_forward,
    train_window = opts$`train-window`,
    horizons = horizons,
    metrics = metrics,
    backtest_id = backtest_id
  )

  # Save results
  output_dir <- save_backtest_results(result, opts, backtest_id)

  # Generate report if requested
  if (!is.null(opts$report)) {
    generate_backtest_report(result, opts$report, opts$`open-report`)
  }

  # Print results
  print_backtest_results(result, output_dir, opts)

  invisible(result)
}


#' Save backtest results
#' @noRd
save_backtest_results <- function(result, opts, backtest_id) {
  output_dir <- opts$`output-dir` %||% sprintf("backtests/%s", backtest_id)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Save results
  saveRDS(result, file.path(output_dir, "backtest_result.rds"))

  # Save summary as CSV
  data.table::fwrite(
    result$get_summary(),
    file.path(output_dir, "summary.csv")
  )

  # Save metrics by interval
  data.table::fwrite(
    result$get_metrics_by_interval(),
    file.path(output_dir, "metrics_by_interval.csv")
  )

  # Save recommendation
  writeLines(
    result$get_recommendation_text(),
    file.path(output_dir, "recommendation.txt")
  )

  output_dir
}


#' Generate backtest HTML report
#' @noRd
generate_backtest_report <- function(result, report_path, open_report) {
  cli_info(sprintf("Generating report: %s", report_path))

  reporter <- HighcharterReporter$new()
  reporter$generate("backtest", result$as_report_data(), report_path)

  cli_success(sprintf("Report generated: %s", report_path))

  if (open_report) {
    browseURL(report_path)
  }
}


#' Print backtest results
#' @noRd
print_backtest_results <- function(result, output_dir, opts) {
  if (opts$quiet) return(invisible(NULL))

  cat("\n")
  cli_header("Backtest Complete")
  cat("\n")

  summary <- result$summary()

  cat(sprintf("  Backtest ID:    %s\n", summary$backtest_id))
  cat(sprintf("  Period:         %s to %s\n", summary$start_date, summary$end_date))
  cat(sprintf("  Days simulated: %d\n", summary$n_days))
  cat(sprintf("  Walk-forward:   %s\n", if (summary$walk_forward) "Yes" else "No"))
  cat(sprintf("  Elapsed:        %.2f minutes\n", result$elapsed_time / 60))
  cat(sprintf("  Output:         %s\n", output_dir))

  cat("\n")
  cat("Results by Retraining Interval:\n")
  cat("\n")

  metrics_by_interval <- result$get_metrics_by_interval()
  print(metrics_by_interval[, .(interval, mean_mape, std_mape, mean_mae)])

  cat("\n")
  cat("Recommendation:\n")
  cat(sprintf("  %s\n", result$get_recommendation_text()))

  cat("\n")
}
```

### Usage Examples

```bash
# Basic backtest for Q4 2024
prevcarga backtest \
  --start-date 2024-10-01 \
  --end-date 2024-12-31

# With specific retraining intervals
prevcarga backtest \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --retrain-intervals 1,7,14,30

# Specific areas and models
prevcarga backtest \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --areas RJ,SP \
  --models lgbm,rf

# With report generation
prevcarga backtest \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --report backtest_2024Q4.html \
  --open-report

# Custom backtest ID and output
prevcarga backtest \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --backtest-id bt_2024Q4_lgbm \
  --output-dir results/backtest_q4

# Dry run to see simulation plan
prevcarga backtest \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --dry-run

# Disable walk-forward (fixed training window)
prevcarga backtest \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --no-walk-forward
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | cmd_backtest() valid args | Backtest runs |
| TC-002 | Missing --start-date | Error message |
| TC-003 | Missing --end-date | Error message |
| TC-004 | Invalid date format | Error message |
| TC-005 | start-date >= end-date | Error message |
| TC-006 | Invalid retraining intervals | Error message |
| TC-007 | --walk-forward flag | Walk-forward enabled |
| TC-008 | --no-walk-forward flag | Walk-forward disabled |
| TC-009 | --report option | Report generated |
| TC-010 | --dry-run | Shows plan, no execution |
| TC-011 | Custom --backtest-id | Uses provided ID |
| TC-012 | Results saved correctly | All files present |

---

## Dependencies

- PC-064-09: CLI Entry Point
- PC-060-08: BacktestWorkflow
- PC-057-08: ConfigManager
- PC-054-07: HighcharterReporter (for reports)

---

## Definition of Done

- [ ] Backtest command handler implemented
- [ ] All options parsed correctly
- [ ] Walk-forward validation working
- [ ] Multiple retraining intervals supported
- [ ] Results saved to output directory
- [ ] Report generation working
- [ ] Recommendation output
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Default retraining intervals: 1, 3, 7, 10, 15 days
- Walk-forward is enabled by default
- Dry run shows estimated time based on simulation count
- Backtest ID auto-generated if not provided
- Consider adding checkpoint/resume for long backtests
