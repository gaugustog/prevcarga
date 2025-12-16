# PC-065-09: Train Command

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.2
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `train` CLI command for training models across specified areas. This command orchestrates the TrainWorkflow with user-specified parameters from command-line arguments.

---

## Acceptance Criteria

- [ ] Train command module created in `R/cli/commands/train.R`
- [ ] Full option parsing with optparse
- [ ] Support for area and model selection
- [ ] Date range specification
- [ ] Version tagging for artifacts
- [ ] Parallel worker configuration
- [ ] Progress output during training

---

## Technical Specification

### File Location
```
R/cli/commands/train.R
```

### Train Command Handler

```r
#' Train Command Handler
#'
#' Handles the 'prevcarga train' command for model training.
#'
#' @param args Command line arguments
#' @noRd
cmd_train <- function(args) {
  # Create parser
  parser <- create_train_parser()

  # Parse arguments
  opts <- optparse::parse_args(parser, args, positional_arguments = FALSE)

  # Validate options

  validate_train_options(opts)

  # Execute training
  execute_train(opts)
}


#' Create Train Command Parser
#' @noRd
create_train_parser <- function() {
  parser <- optparse::OptionParser(
    usage = "prevcarga train [options]",
    description = "Train forecasting models for specified areas and models."
  )

  # Configuration
  parser <- optparse::add_option(
    parser, c("-c", "--config"),
    type = "character",
    default = "config/config.yaml",
    help = "Path to configuration YAML file [default: %default]"
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

  # Date range
  parser <- optparse::add_option(
    parser, "--start-date",
    type = "character",
    help = "Training data start date (YYYY-MM-DD) [required]"
  )

  parser <- optparse::add_option(
    parser, "--end-date",
    type = "character",
    help = "Training data end date (YYYY-MM-DD) [required]"
  )

  # Version
  parser <- optparse::add_option(
    parser, c("-V", "--version"),
    type = "character",
    default = NULL,
    help = "Model version string [default: auto-generated timestamp]"
  )

  # Parallel
  parser <- optparse::add_option(
    parser, c("-j", "--parallel"),
    type = "integer",
    default = 4,
    help = "Number of parallel workers [default: %default]"
  )

  # Storage
  parser <- optparse::add_option(
    parser, "--storage-backend",
    type = "character",
    default = NULL,
    help = "Storage backend (local, s3) [default: from config]"
  )

  # Output
  parser <- optparse::add_option(
    parser, c("-o", "--output-dir"),
    type = "character",
    default = NULL,
    help = "Output directory for artifacts [default: from config]"
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
    help = "Validate options without executing training"
  )

  # Resume
  parser <- optparse::add_option(
    parser, "--resume",
    type = "character",
    default = NULL,
    help = "Resume from checkpoint (version string)"
  )

  parser
}


#' Validate Train Options
#' @noRd
validate_train_options <- function(opts) {
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

  # Validate config file exists
  if (!file.exists(opts$config)) {
    stop(sprintf("Configuration file not found: %s", opts$config))
  }

  # Validate parallel workers
  if (opts$parallel < 1 || opts$parallel > 32) {
    stop("--parallel must be between 1 and 32")
  }

  invisible(TRUE)
}


#' Execute Training
#' @noRd
execute_train <- function(opts) {
  # Load configuration
  cli_info(sprintf("Loading configuration: %s", opts$config))
  config <- ConfigManager$new()
  config$load(opts$config)

  # Override parallel workers if specified
  if (!is.null(opts$parallel)) {
    config$set_runtime("training.parallel.n_jobs", opts$parallel)
  }

  # Resolve areas
  areas <- resolve_areas(opts$areas, config)
  cli_info(sprintf("Training areas: %s", paste(areas, collapse = ", ")))

  # Resolve models
  models <- resolve_models(opts$models, config)
  cli_info(sprintf("Training models: %s", paste(models, collapse = ", ")))

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
  cli_header("Starting Training Workflow")

  workflow <- TrainWorkflow$new(config, storage, logger)

  result <- workflow$run(
    areas = areas,
    models = models,
    start_date = opts$`start-date`,
    end_date = opts$`end-date`,
    version = opts$version
  )

  # Print results
  print_train_results(result, opts)

  invisible(result)
}


#' Resolve area codes from option
#' @noRd
resolve_areas <- function(areas_opt, config) {
  if (tolower(areas_opt) == "all") {
    return(config$get_areas())
  }

  areas <- trimws(strsplit(areas_opt, ",")[[1]])
  valid_areas <- config$get_areas()

  invalid <- setdiff(areas, valid_areas)
  if (length(invalid) > 0) {
    stop(sprintf(
      "Invalid area codes: %s. Valid areas: %s",
      paste(invalid, collapse = ", "),
      paste(valid_areas, collapse = ", ")
    ))
  }

  areas
}


#' Resolve model names from option
#' @noRd
resolve_models <- function(models_opt, config) {
  if (tolower(models_opt) == "all") {
    # Get all enabled models from config
    model_config <- config$get("models.plugins")
    enabled <- names(Filter(function(m) isTRUE(m$enabled), model_config))
    return(enabled)
  }

  models <- trimws(strsplit(models_opt, ",")[[1]])

  # Validate models exist
  for (model in models) {
    if (!has_model(model)) {
      stop(sprintf("Unknown model: %s", model))
    }
  }

  models
}


#' Create storage backend from options
#' @noRd
create_storage_from_opts <- function(opts, config) {
  storage_config <- config$get_storage_config()

  # Override backend if specified
  if (!is.null(opts$`storage-backend`)) {
    storage_config$backend <- opts$`storage-backend`
  }

  # Override output dir if specified
  if (!is.null(opts$`output-dir`)) {
    storage_config$local_path <- opts$`output-dir`
  }

  create_storage_backend(storage_config)
}


#' Create logger from options
#' @noRd
create_logger_from_opts <- function(opts, config) {
  level <- if (opts$verbose) "DEBUG" else if (opts$quiet) "ERROR" else "INFO"
  format <- if (opts$quiet) "text" else "json"

  logger <- StructuredLogger$new(level = level, format = format)
  logger$with_context("command", "train")

  logger
}


#' Print training results
#' @noRd
print_train_results <- function(result, opts) {
  if (opts$quiet) return(invisible(NULL))

  cat("\n")
  cli_header("Training Complete")
  cat("\n")

  summary <- result$summary()

  cat(sprintf("  Version:      %s\n", result$version))
  cat(sprintf("  Elapsed:      %.2f seconds\n", result$elapsed_time))
  cat(sprintf("  Total tasks:  %d\n", summary$total))
  cat(sprintf("  Successful:   %d\n", summary$success))
  cat(sprintf("  Failed:       %d\n", summary$failed))
  cat(sprintf("  Success rate: %.1f%%\n", summary$success_rate * 100))

  if (summary$failed > 0) {
    cat("\nFailed tasks:\n")
    failed <- result$get_failed()
    for (f in failed) {
      cat(sprintf("  - %s/%s: %s\n", f$area, f$model, f$error))
    }
  }

  cat("\n")
}
```

### Usage Examples

```bash
# Basic training with required options
prevcarga train --start-date 2023-01-01 --end-date 2024-01-01

# Train specific areas and models
prevcarga train \
  --areas RJ,SP,MG \
  --models lgbm,rf \
  --start-date 2023-01-01 \
  --end-date 2024-01-01

# With custom version and parallel workers
prevcarga train \
  --areas all \
  --models lgbm \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --version v1.0.0 \
  --parallel 8

# Using custom config
prevcarga train \
  --config /path/to/config.yaml \
  --areas RJ \
  --start-date 2023-01-01 \
  --end-date 2024-01-01

# Dry run to validate options
prevcarga train \
  --areas RJ,SP \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --dry-run

# Verbose output
prevcarga train \
  --areas RJ \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --verbose
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | cmd_train() with valid args | Training executes |
| TC-002 | Missing --start-date | Error message |
| TC-003 | Missing --end-date | Error message |
| TC-004 | Invalid date format | Error message |
| TC-005 | start-date >= end-date | Error message |
| TC-006 | Invalid area code | Error message |
| TC-007 | Invalid model name | Error message |
| TC-008 | --areas all | All areas selected |
| TC-009 | --models all | All enabled models |
| TC-010 | --dry-run | No execution |
| TC-011 | --verbose | Debug output |
| TC-012 | --quiet | Minimal output |

---

## Dependencies

- PC-064-09: CLI Entry Point
- PC-058-08: TrainWorkflow
- PC-057-08: ConfigManager

---

## Definition of Done

- [ ] Train command handler implemented
- [ ] All options parsed correctly
- [ ] Input validation complete
- [ ] Dry run mode working
- [ ] Progress output during training
- [ ] Results summary printed
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Default version uses timestamp format: YYYYMMDD_HHMMSS
- Areas and models can be comma-separated or "all"
- Parallel workers capped at 32 for safety
- Consider adding --resume for checkpoint recovery
