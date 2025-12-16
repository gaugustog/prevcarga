# PC-066-09: Predict Command

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.3
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `predict` CLI command for generating forecasts. This command supports both batch mode (full D+0 to D+8 forecasts) and intraday mode (partial day forecasts with BLF anchor completion).

---

## Acceptance Criteria

- [ ] Predict command module created in `R/cli/commands/predict.R`
- [ ] Support for batch and intraday modes
- [ ] Area and model selection
- [ ] Reconciliation option
- [ ] Combination option
- [ ] Output path configuration
- [ ] Horizon specification

---

## Technical Specification

### File Location
```
R/cli/commands/predict.R
```

### Predict Command Handler

```r
#' Predict Command Handler
#'
#' Handles the 'prevcarga predict' command for generating forecasts.
#'
#' @param args Command line arguments
#' @noRd
cmd_predict <- function(args) {
  # Create parser
  parser <- create_predict_parser()

  # Parse arguments
  opts <- optparse::parse_args(parser, args, positional_arguments = FALSE)

  # Validate options
  validate_predict_options(opts)

  # Execute prediction
  execute_predict(opts)
}


#' Create Predict Command Parser
#' @noRd
create_predict_parser <- function() {
  parser <- optparse::OptionParser(
    usage = "prevcarga predict [options]",
    description = "Generate load forecasts for specified date and areas."
  )

  # Configuration
  parser <- optparse::add_option(
    parser, c("-c", "--config"),
    type = "character",
    default = "config/config.yaml",
    help = "Path to configuration YAML file [default: %default]"
  )

  # Target date
  parser <- optparse::add_option(
    parser, c("-d", "--date"),
    type = "character",
    default = NULL,
    help = "Target date for prediction (YYYY-MM-DD) [default: today]"
  )

  # Mode
  parser <- optparse::add_option(
    parser, "--mode",
    type = "character",
    default = "batch",
    help = "Prediction mode: batch or intraday [default: %default]"
  )

  # Current hour (for intraday)
  parser <- optparse::add_option(
    parser, "--current-hour",
    type = "integer",
    default = NULL,
    help = "Current hour for intraday mode (0-23) [default: system time]"
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

  # Model version
  parser <- optparse::add_option(
    parser, "--model-version",
    type = "character",
    default = "latest",
    help = "Model version to use [default: %default]"
  )

  # Horizons
  parser <- optparse::add_option(
    parser, "--horizons",
    type = "character",
    default = NULL,
    help = "Comma-separated horizons (0-8) [default: from config]"
  )

  # Reconciliation
  parser <- optparse::add_option(
    parser, "--reconcile",
    action = "store_true",
    default = FALSE,
    help = "Enable hierarchical reconciliation"
  )

  parser <- optparse::add_option(
    parser, "--no-reconcile",
    action = "store_true",
    default = FALSE,
    help = "Disable hierarchical reconciliation"
  )

  # Combination
  parser <- optparse::add_option(
    parser, "--combine",
    action = "store_true",
    default = FALSE,
    help = "Enable forecast combination"
  )

  parser <- optparse::add_option(
    parser, "--no-combine",
    action = "store_true",
    default = FALSE,
    help = "Disable forecast combination"
  )

  # Output
  parser <- optparse::add_option(
    parser, c("-o", "--output"),
    type = "character",
    default = NULL,
    help = "Output path for predictions [default: auto-generated]"
  )

  parser <- optparse::add_option(
    parser, "--format",
    type = "character",
    default = "parquet",
    help = "Output format: parquet, csv, rds [default: %default]"
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
    help = "Validate options without executing prediction"
  )

  parser
}


#' Validate Predict Options
#' @noRd
validate_predict_options <- function(opts) {
  # Validate mode
  if (!opts$mode %in% c("batch", "intraday")) {
    stop("--mode must be 'batch' or 'intraday'")
  }

  # Validate date if provided
  if (!is.null(opts$date)) {
    tryCatch(
      as.Date(opts$date),
      error = function(e) stop("Invalid --date format. Use YYYY-MM-DD")
    )
  }

  # Validate current-hour for intraday
  if (opts$mode == "intraday" && !is.null(opts$`current-hour`)) {
    if (opts$`current-hour` < 0 || opts$`current-hour` > 23) {
      stop("--current-hour must be between 0 and 23")
    }
  }

  # Validate config file
  if (!file.exists(opts$config)) {
    stop(sprintf("Configuration file not found: %s", opts$config))
  }

  # Validate output format
  if (!opts$format %in% c("parquet", "csv", "rds")) {
    stop("--format must be 'parquet', 'csv', or 'rds'")
  }

  # Validate horizons if provided
  if (!is.null(opts$horizons)) {
    horizons <- as.integer(strsplit(opts$horizons, ",")[[1]])
    if (any(is.na(horizons)) || any(horizons < 0) || any(horizons > 8)) {
      stop("--horizons must be comma-separated integers between 0 and 8")
    }
  }

  invisible(TRUE)
}


#' Execute Prediction
#' @noRd
execute_predict <- function(opts) {
  # Load configuration
  cli_info(sprintf("Loading configuration: %s", opts$config))
  config <- ConfigManager$new()
  config$load(opts$config)

  # Resolve target date
  target_date <- if (is.null(opts$date)) {
    Sys.Date()
  } else {
    as.Date(opts$date)
  }
  cli_info(sprintf("Target date: %s", target_date))

  # Resolve mode
  mode <- opts$mode
  cli_info(sprintf("Mode: %s", mode))

  # Resolve current hour for intraday
  current_hour <- if (mode == "intraday") {
    opts$`current-hour` %||% as.integer(format(Sys.time(), "%H"))
  } else {
    NULL
  }

  if (mode == "intraday") {
    cli_info(sprintf("Current hour: %d", current_hour))
  }

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
  cli_info(sprintf("Horizons: %s", paste(horizons, collapse = ", ")))

  # Resolve reconciliation
  reconcile <- if (opts$`no-reconcile`) {
    FALSE
  } else if (opts$reconcile) {
    TRUE
  } else {
    config$get_with_default("prediction.reconciliation", TRUE)
  }

  # Resolve combination
  combine <- if (opts$`no-combine`) {
    FALSE
  } else if (opts$combine) {
    TRUE
  } else {
    config$get_with_default("prediction.combination", TRUE)
  }

  # Create storage backend
  storage <- create_storage_from_opts(opts, config)

  # Create logger
  logger <- create_logger_from_opts(opts, config)

  # Dry run check
  if (opts$`dry-run`) {
    cli_success("Dry run complete. Configuration is valid.")
    return(invisible(NULL))
  }

  # Create and run workflow
  cli_header(sprintf("Starting %s Prediction", tools::toTitleCase(mode)))

  workflow <- PredictWorkflow$new(config, storage, logger)

  result <- workflow$run(
    date = target_date,
    mode = mode,
    areas = areas,
    models = models,
    model_version = opts$`model-version`,
    horizons = horizons,
    reconcile = reconcile,
    combine = combine,
    current_hour = current_hour
  )

  # Save output
  output_path <- save_predictions(result, opts, target_date, mode)

  # Print results
  print_predict_results(result, output_path, opts)

  invisible(result)
}


#' Save prediction results
#' @noRd
save_predictions <- function(result, opts, date, mode) {
  # Determine output path
  output_path <- if (!is.null(opts$output)) {
    opts$output
  } else {
    sprintf("predictions/%s/%s/predictions.%s",
            mode, date, opts$format)
  }

  # Ensure directory exists
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  # Save based on format
  predictions <- result$get_predictions()

  switch(opts$format,
    "parquet" = arrow::write_parquet(predictions, output_path),
    "csv" = data.table::fwrite(predictions, output_path),
    "rds" = saveRDS(predictions, output_path)
  )

  output_path
}


#' Print prediction results
#' @noRd
print_predict_results <- function(result, output_path, opts) {
  if (opts$quiet) return(invisible(NULL))

  cat("\n")
  cli_header("Prediction Complete")
  cat("\n")

  summary <- result$summary()

  cat(sprintf("  Mode:         %s\n", summary$mode))
  cat(sprintf("  Target date:  %s\n", summary$date))
  cat(sprintf("  Areas:        %d\n", summary$n_areas))
  cat(sprintf("  Models:       %d\n", summary$n_models))
  cat(sprintf("  Horizons:     %s\n", paste(summary$horizons, collapse = ", ")))
  cat(sprintf("  Reconciled:   %s\n", if (summary$reconciled) "Yes" else "No"))
  cat(sprintf("  Combined:     %s\n", if (summary$combined) "Yes" else "No"))
  cat(sprintf("  Elapsed:      %.2f seconds\n", result$elapsed_time))
  cat(sprintf("  Output:       %s\n", output_path))

  cat("\n")
}
```

### Usage Examples

```bash
# Basic batch prediction for today
prevcarga predict

# Batch prediction for specific date
prevcarga predict --date 2025-01-17

# Intraday prediction
prevcarga predict --mode intraday --current-hour 10

# Specific areas and models
prevcarga predict \
  --date 2025-01-17 \
  --areas RJ,SP,MG \
  --models lgbm

# With reconciliation and combination
prevcarga predict \
  --date 2025-01-17 \
  --reconcile \
  --combine

# Disable reconciliation
prevcarga predict --date 2025-01-17 --no-reconcile

# Specific horizons
prevcarga predict --date 2025-01-17 --horizons 0,1,2

# Custom output path and format
prevcarga predict \
  --date 2025-01-17 \
  --output output/forecast_20250117.csv \
  --format csv

# Use specific model version
prevcarga predict --date 2025-01-17 --model-version v1.0.0

# Dry run
prevcarga predict --date 2025-01-17 --dry-run
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | cmd_predict() batch mode | Predictions generated |
| TC-002 | cmd_predict() intraday mode | Partial predictions |
| TC-003 | Invalid mode | Error message |
| TC-004 | Invalid date format | Error message |
| TC-005 | Invalid current-hour | Error message |
| TC-006 | --areas all | All areas predicted |
| TC-007 | --reconcile flag | Reconciliation enabled |
| TC-008 | --no-reconcile flag | Reconciliation disabled |
| TC-009 | --combine flag | Combination enabled |
| TC-010 | Custom output path | Saved to path |
| TC-011 | --format csv | CSV output |
| TC-012 | --dry-run | No execution |

---

## Dependencies

- PC-064-09: CLI Entry Point
- PC-059-08: PredictWorkflow
- PC-057-08: ConfigManager

---

## Definition of Done

- [ ] Predict command handler implemented
- [ ] Batch and intraday modes working
- [ ] All options parsed correctly
- [ ] Reconciliation toggle working
- [ ] Combination toggle working
- [ ] Multiple output formats supported
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Default date is today if not specified
- Intraday mode uses current system time for hour if not specified
- Output path is auto-generated if not provided
- Reconciliation/combination defaults come from config
