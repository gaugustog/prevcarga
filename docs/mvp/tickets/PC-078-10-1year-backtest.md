# PC-078-10: 1-Year Backtest Execution

**Epic:** [EPIC-10: Testing & Validation](../epics/EPIC-10-testing-validation.md)
**Task Reference:** T-10.3
**Priority:** High
**Estimated Effort:** 3 days

---

## Summary

Execute a comprehensive 1-year backtest for all registered models, testing multiple retraining intervals and generating detailed metrics by model, area, horizon, and patamar (peak/off-peak period).

---

## Acceptance Criteria

- [ ] Backtest script created in `tests/validation/backtest_2024.R`
- [ ] Full year (2024) backtest executed
- [ ] Retraining intervals tested: 1, 3, 7, 10, 15 days
- [ ] Metrics broken down by model, area, horizon, patamar
- [ ] Completion within 8 hours
- [ ] Backtest report generated

---

## Technical Specification

### File Location
```
tests/validation/backtest_2024.R
tests/validation/backtest_analysis.R
```

### Full Year Backtest Configuration

```r
#' @title YearBacktest
#' @description Comprehensive 1-year backtest execution
#' @export
YearBacktest <- R6::R6Class(
  "YearBacktest",
  private = list(
    config = NULL,
    storage = NULL,
    logger = NULL,
    output_dir = NULL,
    results = NULL
  ),

  public = list(
    #' @description Initialize year backtest
    #' @param config ConfigManager instance
    #' @param output_dir Output directory
    initialize = function(config, output_dir = "backtests/2024") {
      private$config <- config
      private$storage <- create_storage_backend(config$get_storage_config())
      private$logger <- StructuredLogger$new(level = "INFO")
      private$output_dir <- output_dir

      dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    },

    #' @description Run full year backtest
    #' @param year Year to backtest
    #' @param models Models to test (NULL = all)
    #' @param areas Areas to test (NULL = all)
    #' @param retrain_intervals Retraining intervals to evaluate
    #' @return BacktestResults
    run = function(year = 2024,
                   models = NULL,
                   areas = NULL,
                   retrain_intervals = c(1, 3, 7, 10, 15)) {
      start_time <- Sys.time()

      # Resolve parameters
      models <- models %||% list_models()
      areas <- areas %||% private$config$get_areas()

      start_date <- as.Date(sprintf("%d-01-01", year))
      end_date <- as.Date(sprintf("%d-12-31", year))

      private$logger$with_context("backtest", "year_2024")
      private$logger$info("Starting 1-year backtest",
                          models = length(models),
                          areas = length(areas),
                          intervals = length(retrain_intervals))

      # Print configuration
      cat("\n")
      cat("=" |> rep(60) |> paste(collapse = ""))
      cat("\n")
      cat("  PrevCarga 1-Year Backtest\n")
      cat("=" |> rep(60) |> paste(collapse = ""))
      cat("\n\n")

      cat(sprintf("Year:       %d\n", year))
      cat(sprintf("Period:     %s to %s (%d days)\n",
                  start_date, end_date,
                  as.integer(end_date - start_date) + 1))
      cat(sprintf("Models:     %s\n", paste(models, collapse = ", ")))
      cat(sprintf("Areas:      %d\n", length(areas)))
      cat(sprintf("Intervals:  %s days\n", paste(retrain_intervals, collapse = ", ")))
      cat("\n")

      # Create workflow
      workflow <- BacktestWorkflow$new(
        config = private$config,
        storage = private$storage,
        logger = private$logger
      )

      # Run backtest
      result <- workflow$run(
        start_date = start_date,
        end_date = end_date,
        models = models,
        areas = areas,
        retrain_intervals = retrain_intervals,
        walk_forward = TRUE,
        train_window = 365,  # 1 year training window
        horizons = 0:8,
        metrics = c("mape", "mae", "rmse", "mbe"),
        backtest_id = sprintf("bt_%d_full", year)
      )

      private$results <- result

      # Calculate additional breakdowns
      self$calculate_breakdowns()

      # Generate report
      self$generate_report()

      # Log completion
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "hours"))
      private$logger$info("Backtest complete",
                          elapsed_hours = elapsed,
                          status = "success")

      cat(sprintf("\nBacktest completed in %.2f hours\n", elapsed))

      result
    },

    #' @description Calculate additional metric breakdowns
    calculate_breakdowns = function() {
      if (is.null(private$results)) {
        stop("No results to analyze. Run backtest first.")
      }

      predictions <- private$results$get_all_predictions()
      actuals <- private$results$get_all_actuals()
      calculator <- MetricsCalculator$new()

      # By model
      metrics_by_model <- self$calculate_metrics_by(
        predictions, actuals, "model", calculator
      )

      # By area
      metrics_by_area <- self$calculate_metrics_by(
        predictions, actuals, "area_code", calculator
      )

      # By horizon
      metrics_by_horizon <- self$calculate_metrics_by(
        predictions, actuals, "horizon", calculator
      )

      # By patamar
      predictions[, patamar := classify_patamar(datetime)]
      metrics_by_patamar <- self$calculate_metrics_by(
        predictions, actuals, "patamar", calculator
      )

      # By model and horizon
      metrics_model_horizon <- predictions[, .(
        mape = calculator$mape(.SD$actual, .SD$prediction),
        mae = calculator$mae(.SD$actual, .SD$prediction),
        rmse = calculator$rmse(.SD$actual, .SD$prediction)
      ), by = .(model, horizon)]

      # By area and patamar
      metrics_area_patamar <- predictions[, .(
        mape = calculator$mape(.SD$actual, .SD$prediction),
        mae = calculator$mae(.SD$actual, .SD$prediction)
      ), by = .(area_code, patamar)]

      # Store breakdowns
      private$results$breakdowns <- list(
        by_model = metrics_by_model,
        by_area = metrics_by_area,
        by_horizon = metrics_by_horizon,
        by_patamar = metrics_by_patamar,
        model_horizon = metrics_model_horizon,
        area_patamar = metrics_area_patamar
      )
    },

    #' @description Calculate metrics by grouping variable
    calculate_metrics_by = function(predictions, actuals, group_var, calculator) {
      predictions[, .(
        mape = calculator$mape(.SD$actual, .SD$prediction),
        mae = calculator$mae(.SD$actual, .SD$prediction),
        rmse = calculator$rmse(.SD$actual, .SD$prediction),
        mbe = calculator$mbe(.SD$actual, .SD$prediction),
        n = .N
      ), by = group_var]
    },

    #' @description Generate comprehensive report
    generate_report = function() {
      report_path <- file.path(
        private$output_dir,
        "backtest_2024_report.html"
      )

      # Render report
      rmarkdown::render(
        input = system.file(
          "validation/backtest_report.Rmd",
          package = "prevcargaons"
        ),
        output_file = report_path,
        params = list(
          results = private$results,
          breakdowns = private$results$breakdowns,
          config = private$config$as_list()
        )
      )

      cat(sprintf("Report generated: %s\n", report_path))

      # Save detailed results
      self$save_results()

      report_path
    },

    #' @description Save detailed results
    save_results = function() {
      # Summary CSV
      summary_path <- file.path(private$output_dir, "summary.csv")
      data.table::fwrite(private$results$get_summary(), summary_path)

      # Metrics by interval
      interval_path <- file.path(private$output_dir, "metrics_by_interval.csv")
      data.table::fwrite(
        private$results$get_metrics_by_interval(),
        interval_path
      )

      # Breakdowns
      for (name in names(private$results$breakdowns)) {
        breakdown_path <- file.path(
          private$output_dir,
          sprintf("metrics_%s.csv", name)
        )
        data.table::fwrite(private$results$breakdowns[[name]], breakdown_path)
      }

      # Full results as RDS
      rds_path <- file.path(private$output_dir, "backtest_results.rds")
      saveRDS(private$results, rds_path)

      cat(sprintf("Results saved to: %s\n", private$output_dir))
    },

    #' @description Get results
    get_results = function() {
      private$results
    },

    #' @description Print summary
    print = function() {
      if (is.null(private$results)) {
        cat("YearBacktest: No results yet. Run backtest first.\n")
        return(invisible(self))
      }

      cat("\n1-Year Backtest Summary\n")
      cat("=======================\n\n")

      # Overall metrics
      summary <- private$results$summary()
      cat(sprintf("Days simulated: %d\n", summary$n_days))
      cat(sprintf("Total predictions: %d\n", summary$n_predictions))
      cat("\n")

      # By interval
      cat("Results by Retraining Interval:\n")
      print(private$results$get_metrics_by_interval())
      cat("\n")

      # Recommendation
      cat("Recommendation:\n")
      cat(sprintf("  %s\n", private$results$get_recommendation_text()))
      cat("\n")

      invisible(self)
    }
  )
)
```

### Patamar Classification

```r
#' Classify datetime into patamar (peak/off-peak period)
#'
#' Based on Brazilian electricity market periods:
#' - Ponta (Peak): 18:00-21:00 weekdays
#' - Fora Ponta (Off-peak): all other hours
#'
#' @param datetime POSIXct datetime
#' @return Character patamar classification
#' @export
classify_patamar <- function(datetime) {
  hour <- as.integer(format(datetime, "%H"))
  weekday <- as.integer(format(datetime, "%u"))  # 1=Monday, 7=Sunday

  # Peak: 18:00-21:00 on weekdays (Mon-Fri)
  is_peak <- weekday <= 5 & hour >= 18 & hour < 21

  ifelse(is_peak, "ponta", "fora_ponta")
}


#' Get patamar periods for a day
#'
#' @param date Date
#' @return data.table with hour and patamar
#' @export
get_patamar_schedule <- function(date) {
  weekday <- as.integer(format(date, "%u"))

  data.table::data.table(
    hour = 0:23,
    patamar = ifelse(
      weekday <= 5 & 0:23 >= 18 & 0:23 < 21,
      "ponta",
      "fora_ponta"
    )
  )
}
```

### Backtest Runner Script

```bash
#!/usr/bin/env Rscript
# tests/validation/backtest_2024.R
#
# Execute full 2024 backtest
#
# Usage: Rscript backtest_2024.R [--config config.yaml] [--output output_dir]

library(prevcargaons)

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)

config_path <- "config/config.yaml"
output_dir <- "backtests/2024"

if ("--config" %in% args) {
  idx <- which(args == "--config")
  config_path <- args[idx + 1]
}

if ("--output" %in% args) {
  idx <- which(args == "--output")
  output_dir <- args[idx + 1]
}

# Load configuration
cat("Loading configuration...\n")
config <- ConfigManager$new()
config$load(config_path)

# Initialize backtest
backtest <- YearBacktest$new(
  config = config,
  output_dir = output_dir
)

# Run full 2024 backtest
cat("Starting 1-year backtest...\n")
start_time <- Sys.time()

results <- backtest$run(
  year = 2024,
  retrain_intervals = c(1, 3, 7, 10, 15)
)

elapsed <- difftime(Sys.time(), start_time, units = "hours")

# Print summary
print(backtest)

# Check if completed within time limit
if (elapsed > 8) {
  cat(sprintf("\nWARNING: Backtest took %.2f hours (target: <8 hours)\n", elapsed))
} else {
  cat(sprintf("\n✓ Backtest completed in %.2f hours (within 8-hour target)\n", elapsed))
}
```

### Usage Example

```r
# Load configuration
config <- load_config("config/config.yaml")

# Create backtest
backtest <- YearBacktest$new(config, output_dir = "backtests/2024")

# Run full year
results <- backtest$run(
  year = 2024,
  retrain_intervals = c(1, 3, 7, 10, 15)
)

# Output:
# ============================================================
#   PrevCarga 1-Year Backtest
# ============================================================
#
# Year:       2024
# Period:     2024-01-01 to 2024-12-31 (366 days)
# Models:     lgbm, rf, hw
# Areas:      21
# Intervals:  1, 3, 7, 10, 15 days
#
# ... progress output ...
#
# Backtest completed in 6.5 hours
#
# 1-Year Backtest Summary
# =======================
#
# Days simulated: 366
# Total predictions: 2,894,544
#
# Results by Retraining Interval:
#    interval mean_mape std_mape mean_mae
# 1:        1      3.21     0.45   142.3
# 2:        3      3.28     0.48   145.1
# 3:        7      3.45     0.52   151.2
# 4:       10      3.62     0.58   158.4
# 5:       15      3.89     0.65   167.8
#
# Recommendation:
#   Optimal retraining interval: 1-3 days based on MAPE
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Run full 2024 backtest | Completes successfully |
| TC-002 | Complete in <8 hours | Time target met |
| TC-003 | All intervals tested | 5 intervals processed |
| TC-004 | Metrics by model | Per-model breakdown |
| TC-005 | Metrics by area | Per-area breakdown |
| TC-006 | Metrics by horizon | D+0 to D+8 breakdown |
| TC-007 | Metrics by patamar | Peak/off-peak breakdown |
| TC-008 | Report generated | HTML report exists |
| TC-009 | Results saved | All CSV files exist |
| TC-010 | Recommendation provided | Interval suggested |

---

## Dependencies

- PC-060-08: BacktestWorkflow
- PC-047-07: MetricsCalculator
- PC-048-07: Metrics by Period
- PC-049-07: Metrics by Horizon

---

## Definition of Done

- [ ] YearBacktest class implemented
- [ ] Full 2024 backtest executable
- [ ] All retraining intervals tested
- [ ] Complete metric breakdowns
- [ ] Patamar classification working
- [ ] Completion within 8 hours
- [ ] Report generated
- [ ] Results saved
- [ ] roxygen2 documentation complete
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Backtest is computationally intensive (~6-8 hours)
- Consider running overnight or on dedicated compute
- Results saved incrementally for recovery
- Patamar follows Brazilian electricity market rules
