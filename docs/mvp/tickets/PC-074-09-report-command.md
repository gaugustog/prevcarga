# PC-074-09: Report Command

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.12
**Priority:** Medium
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `report` CLI command for generating interactive HTML reports using highcharter. This command supports multiple report types: forecast, backtest, compare, dashboard, and drift analysis.

---

## Acceptance Criteria

- [ ] Report command module created in `R/cli/commands/report.R`
- [ ] Support for 5 report types
- [ ] Data loading for each report type
- [ ] Integration with HighcharterReporter
- [ ] Browser auto-open option
- [ ] Custom output path support

---

## Technical Specification

### File Location
```
R/cli/commands/report.R
```

### Report Command Handler

```r
#' Report Command Handler
#'
#' Handles the 'prevcarga report' command for generating interactive HTML reports.
#'
#' @param args Command line arguments
#' @noRd
cmd_report <- function(args) {
  parser <- create_report_parser()

  # Report type is positional first argument
  if (length(args) == 0) {
    optparse::print_help(parser)
    return(invisible(NULL))
  }

  report_type <- args[1]
  remaining_args <- args[-1]

  # Validate report type
  valid_types <- c("forecast", "backtest", "compare", "dashboard", "drift")

  if (!report_type %in% valid_types) {
    stop(sprintf(
      "Invalid report type: '%s'. Valid types: %s",
      report_type, paste(valid_types, collapse = ", ")
    ))
  }

  opts <- optparse::parse_args(parser, remaining_args, positional_arguments = FALSE)
  opts$report_type <- report_type

  validate_report_options(opts)
  execute_report(opts)
}


#' Create Report Command Parser
#' @noRd
create_report_parser <- function() {
  parser <- optparse::OptionParser(
    usage = "prevcarga report <type> [options]

Report Types:
  forecast   - Forecast vs actual analysis
  backtest   - Backtest results summary
  compare    - Model comparison report
  dashboard  - Area-specific dashboard
  drift      - Performance drift analysis",
    description = "Generate interactive HTML reports with highcharter visualizations."
  )

  # Configuration
  parser <- optparse::add_option(
    parser, c("-c", "--config"),
    type = "character",
    default = "config/config.yaml",
    help = "Path to configuration file [default: %default]"
  )

  # Output
  parser <- optparse::add_option(
    parser, c("-o", "--output"),
    type = "character",
    default = NULL,
    help = "Output HTML file path [required]"
  )

  # Common options
  parser <- optparse::add_option(
    parser, "--date",
    type = "character",
    default = NULL,
    help = "Target date (for forecast report)"
  )

  parser <- optparse::add_option(
    parser, "--period",
    type = "character",
    default = NULL,
    help = "Date range as 'start:end' (for comparison/dashboard)"
  )

  parser <- optparse::add_option(
    parser, "--areas",
    type = "character",
    default = NULL,
    help = "Comma-separated area codes (for forecast)"
  )

  parser <- optparse::add_option(
    parser, "--area",
    type = "character",
    default = NULL,
    help = "Single area code (for dashboard)"
  )

  parser <- optparse::add_option(
    parser, "--models",
    type = "character",
    default = NULL,
    help = "Comma-separated model names (for comparison)"
  )

  parser <- optparse::add_option(
    parser, "--model",
    type = "character",
    default = NULL,
    help = "Single model name (for drift)"
  )

  parser <- optparse::add_option(
    parser, "--backtest-id",
    type = "character",
    default = NULL,
    help = "Backtest identifier (for backtest report)"
  )

  parser <- optparse::add_option(
    parser, "--baseline-date",
    type = "character",
    default = NULL,
    help = "Baseline date for drift comparison"
  )

  # Report options
  parser <- optparse::add_option(
    parser, "--title",
    type = "character",
    default = NULL,
    help = "Custom report title"
  )

  parser <- optparse::add_option(
    parser, "--theme",
    type = "character",
    default = "prevcarga",
    help = "Report theme [default: %default]"
  )

  parser <- optparse::add_option(
    parser, "--self-contained",
    action = "store_true",
    default = TRUE,
    help = "Create self-contained HTML [default: TRUE]"
  )

  # Actions
  parser <- optparse::add_option(
    parser, "--open",
    action = "store_true",
    default = FALSE,
    help = "Open report in browser after generation"
  )

  # Verbosity
  parser <- optparse::add_option(
    parser, c("-v", "--verbose"),
    action = "store_true",
    default = FALSE,
    help = "Enable verbose output"
  )

  parser
}


#' Validate Report Options
#' @noRd
validate_report_options <- function(opts) {
  # Output is required
  if (is.null(opts$output)) {
    stop("--output is required")
  }

  # Type-specific validation
  switch(opts$report_type,
    "forecast" = validate_forecast_report_opts(opts),
    "backtest" = validate_backtest_report_opts(opts),
    "compare" = validate_compare_report_opts(opts),
    "dashboard" = validate_dashboard_report_opts(opts),
    "drift" = validate_drift_report_opts(opts)
  )

  invisible(TRUE)
}


#' Validate forecast report options
#' @noRd
validate_forecast_report_opts <- function(opts) {
  if (is.null(opts$date)) {
    stop("--date is required for forecast report")
  }
  validate_date(opts$date, "date")
}


#' Validate backtest report options
#' @noRd
validate_backtest_report_opts <- function(opts) {
  if (is.null(opts$`backtest-id`)) {
    stop("--backtest-id is required for backtest report")
  }
}


#' Validate comparison report options
#' @noRd
validate_compare_report_opts <- function(opts) {
  if (is.null(opts$models)) {
    stop("--models is required for comparison report")
  }

  if (is.null(opts$period)) {
    stop("--period is required for comparison report")
  }

  # Validate period format
  parts <- strsplit(opts$period, ":")[[1]]
  if (length(parts) != 2) {
    stop("--period must be in format 'start:end' (e.g., 2024-01-01:2024-12-31)")
  }
}


#' Validate dashboard report options
#' @noRd
validate_dashboard_report_opts <- function(opts) {
  if (is.null(opts$area)) {
    stop("--area is required for dashboard report")
  }

  if (is.null(opts$period)) {
    stop("--period is required for dashboard report")
  }
}


#' Validate drift report options
#' @noRd
validate_drift_report_opts <- function(opts) {
  if (is.null(opts$model)) {
    stop("--model is required for drift report")
  }
}


#' Execute Report Generation
#' @noRd
execute_report <- function(opts) {
  # Load configuration
  cli_info(sprintf("Loading configuration: %s", opts$config))
  config <- ConfigManager$new()
  config$load(opts$config)

  # Load report data
  cli_info(sprintf("Loading data for %s report...", opts$report_type))
  data <- load_report_data(opts$report_type, opts, config)

  # Generate report
  cli_info("Generating report...")

  reporter <- HighcharterReporter$new()

  report_config <- list(
    title = opts$title,
    theme = opts$theme,
    self_contained = opts$`self-contained`
  )

  reporter$generate(
    report_type = opts$report_type,
    data = data,
    output_path = opts$output,
    config = report_config
  )

  cli_success(sprintf("Report generated: %s", opts$output))

  # Open in browser if requested
  if (opts$open) {
    cli_info("Opening in browser...")
    browseURL(opts$output)
  }

  invisible(opts$output)
}


#' Load Report Data
#'
#' @param report_type Report type
#' @param opts Parsed options
#' @param config ConfigManager instance
#' @return Report data list
#' @noRd
load_report_data <- function(report_type, opts, config) {
  storage <- create_storage_backend(config$get_storage_config())

  switch(report_type,
    "forecast" = load_forecast_report_data(opts, config, storage),
    "backtest" = load_backtest_report_data(opts, config, storage),
    "compare" = load_comparison_report_data(opts, config, storage),
    "dashboard" = load_dashboard_report_data(opts, config, storage),
    "drift" = load_drift_report_data(opts, config, storage)
  )
}


#' Load Forecast Report Data
#' @noRd
load_forecast_report_data <- function(opts, config, storage) {
  date <- as.Date(opts$date)
  areas <- if (!is.null(opts$areas)) {
    strsplit(opts$areas, ",")[[1]]
  } else {
    config$get_areas()
  }

  # Load predictions
  predictions <- load_predictions(date, areas, storage)

  # Load actuals
  actuals <- load_actuals(date, areas, storage)

  # Calculate errors
  errors <- calculate_errors(predictions, actuals)

  # Calculate metrics
  calculator <- MetricsCalculator$new()
  metrics <- calculator$calculate_all(actuals$actual, predictions$predicted)

  list(
    date = date,
    areas = areas,
    predictions = predictions,
    actuals = actuals,
    errors = errors,
    metrics = metrics,
    metrics_by_hour = calculate_metrics_by_hour(errors, calculator),
    metrics_by_area = calculate_metrics_by_area(errors, calculator)
  )
}


#' Load Backtest Report Data
#' @noRd
load_backtest_report_data <- function(opts, config, storage) {
  backtest_id <- opts$`backtest-id`

  # Load backtest results
  result_path <- sprintf("backtests/%s/backtest_result.rds", backtest_id)
  result <- storage$load_rds(result_path)

  list(
    backtest_id = backtest_id,
    summary = result$summary(),
    metrics_by_interval = result$get_metrics_by_interval(),
    metrics_by_date = result$get_metrics_by_date(),
    recommendation = result$get_recommendation(),
    config = result$config
  )
}


#' Load Comparison Report Data
#' @noRd
load_comparison_report_data <- function(opts, config, storage) {
  models <- strsplit(opts$models, ",")[[1]]
  period <- strsplit(opts$period, ":")[[1]]
  start_date <- as.Date(period[1])
  end_date <- as.Date(period[2])
  areas <- if (!is.null(opts$areas)) {
    strsplit(opts$areas, ",")[[1]]
  } else {
    config$get_areas()
  }

  # Load predictions for each model
  model_data <- list()
  for (model in models) {
    preds <- load_model_predictions_period(model, start_date, end_date, areas, storage)
    model_data[[model]] <- preds
  }

  # Load actuals
  actuals <- load_actuals_period(start_date, end_date, areas, storage)

  # Calculate metrics per model
  calculator <- MetricsCalculator$new()
  model_metrics <- lapply(models, function(model) {
    calculator$calculate_all(actuals$actual, model_data[[model]]$predicted)
  })
  names(model_metrics) <- models

  # Statistical comparison
  comparator <- ModelComparator$new()
  dm_tests <- comparator$compare_all(model_data, actuals)

  list(
    models = models,
    period = list(start = start_date, end = end_date),
    areas = areas,
    model_metrics = model_metrics,
    dm_tests = dm_tests,
    ranking = comparator$rank_models(model_metrics)
  )
}


#' Load Dashboard Report Data
#' @noRd
load_dashboard_report_data <- function(opts, config, storage) {
  area <- opts$area
  period <- strsplit(opts$period, ":")[[1]]
  start_date <- as.Date(period[1])
  end_date <- as.Date(period[2])

  # Load historical data
  loader <- DataLoader$new(storage)
  data <- loader$load_carga(
    areas = area,
    start_date = start_date,
    end_date = end_date
  )

  # Calculate load profile
  load_profile <- calculate_load_profile(data)

  # Seasonal patterns
  seasonal <- calculate_seasonal_patterns(data)

  # Recent predictions and errors
  recent_preds <- load_recent_predictions(area, storage)
  recent_metrics <- calculate_recent_metrics(recent_preds)

  list(
    area = area,
    period = list(start = start_date, end = end_date),
    historical = data,
    load_profile = load_profile,
    seasonal = seasonal,
    recent_predictions = recent_preds,
    recent_metrics = recent_metrics
  )
}


#' Load Drift Report Data
#' @noRd
load_drift_report_data <- function(opts, config, storage) {
  model <- opts$model
  baseline_date <- if (!is.null(opts$`baseline-date`)) {
    as.Date(opts$`baseline-date`)
  } else {
    Sys.Date() - 90  # Default: 90 days ago
  }

  # Load drift detector results
  detector <- DriftDetector$new()
  drift_analysis <- detector$analyze(
    model = model,
    baseline_date = baseline_date,
    storage = storage
  )

  list(
    model = model,
    baseline_date = baseline_date,
    current_date = Sys.Date(),
    rolling_metrics = drift_analysis$rolling_metrics,
    baseline_metrics = drift_analysis$baseline,
    current_metrics = drift_analysis$current,
    drift_score = drift_analysis$drift_score,
    alerts = drift_analysis$alerts,
    threshold_violations = drift_analysis$violations
  )
}
```

### Usage Examples

```bash
# Forecast report
prevcarga report forecast \
  --date 2025-01-17 \
  --areas RJ,SP,MG \
  --output reports/forecast_20250117.html

# Backtest report
prevcarga report backtest \
  --backtest-id bt_2024Q4 \
  --output reports/backtest_2024Q4.html \
  --open

# Model comparison report
prevcarga report compare \
  --models lgbm,rf,hw \
  --period 2024-10-01:2024-12-31 \
  --output reports/comparison_2024Q4.html

# Area dashboard
prevcarga report dashboard \
  --area RJ \
  --period 2024-01-01:2024-12-31 \
  --output reports/dashboard_RJ_2024.html

# Drift analysis report
prevcarga report drift \
  --model lgbm \
  --baseline-date 2024-06-01 \
  --output reports/drift_lgbm.html \
  --open

# With custom title and theme
prevcarga report forecast \
  --date 2025-01-17 \
  --title "Daily Forecast Report - January 17, 2025" \
  --theme dark \
  --output reports/forecast_custom.html
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | forecast report | HTML generated |
| TC-002 | backtest report | HTML generated |
| TC-003 | compare report | HTML generated |
| TC-004 | dashboard report | HTML generated |
| TC-005 | drift report | HTML generated |
| TC-006 | Missing --output | Error message |
| TC-007 | Invalid report type | Error message |
| TC-008 | --open flag | Browser opens |
| TC-009 | Custom title | Title in report |
| TC-010 | Missing required option | Specific error |

---

## Dependencies

- PC-064-09: CLI Entry Point
- PC-054-07: HighcharterReporter
- PC-055-07: Highcharter Templates
- PC-047-07: MetricsCalculator
- PC-050-07: DriftDetector
- PC-051-07: ModelComparator

---

## Definition of Done

- [ ] Report command handler implemented
- [ ] All 5 report types working
- [ ] Data loading functions for each type
- [ ] Integration with HighcharterReporter
- [ ] Browser auto-open working
- [ ] Type-specific validation
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Reports are self-contained HTML by default
- Custom themes can be defined in config
- Consider adding PDF export option in future
- Data loading may be slow for large periods
