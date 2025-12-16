# PC-050-07: DriftDetector

**Epic:** [EPIC-07: Evaluation Layer](../epics/EPIC-07-evaluation-layer.md)
**Task Reference:** T-07.4
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `DriftDetector` R6 class for detecting model performance degradation over time, triggering retraining alerts when metrics exceed configurable thresholds.

---

## Acceptance Criteria

- [ ] Drift detection module created in `R/evaluation/drift.R`
- [ ] `DriftDetector` R6 class with baseline comparison
- [ ] Configurable thresholds for MAPE, MAE, RMSE
- [ ] Rolling window metrics calculation
- [ ] Drift detection with detailed reporting
- [ ] Support for gradual and sudden drift

---

## Technical Specification

### File Location
```
R/evaluation/drift.R
```

### DriftDetector R6 Class

```r
#' @title DriftDetector
#' @description Detect model performance degradation over time
#'
#' Monitors forecast metrics and triggers alerts when performance
#' degrades beyond configurable thresholds relative to a baseline.
#'
#' @export
DriftDetector <- R6::R6Class(
  "DriftDetector",
  private = list(
    config = NULL,
    baseline_metrics = NULL,
    history = NULL,
    alerts = NULL
  ),
  public = list(
    #' @description Initialize drift detector
    #' @param window Rolling window size (days)
    #' @param threshold_mape MAPE ratio threshold for drift
    #' @param threshold_mae MAE ratio threshold for drift
    #' @param threshold_rmse RMSE ratio threshold for drift
    #' @param config Additional configuration
    initialize = function(window = 30,
                          threshold_mape = 1.2,
                          threshold_mae = 1.15,
                          threshold_rmse = 1.25,
                          config = list()) {
      private$config <- list(
        window = window,
        threshold_mape = threshold_mape,
        threshold_mae = threshold_mae,
        threshold_rmse = threshold_rmse,
        min_observations = config$min_observations %||% 100,
        alert_cooldown_days = config$alert_cooldown_days %||% 7,
        detect_sudden = config$detect_sudden %||% TRUE,
        sudden_threshold_multiplier = config$sudden_threshold_multiplier %||% 1.5
      )

      private$history <- data.table::data.table(
        date = as.Date(character()),
        mape = numeric(),
        mae = numeric(),
        rmse = numeric(),
        n_obs = integer()
      )

      private$alerts <- data.table::data.table(
        timestamp = as.POSIXct(character()),
        type = character(),
        metric = character(),
        baseline_value = numeric(),
        current_value = numeric(),
        ratio = numeric(),
        threshold = numeric()
      )
    },

    #' @description Set baseline metrics for comparison
    #' @param metrics List or data.table with baseline metrics
    #' @return Invisible self
    set_baseline = function(metrics) {
      if (is.data.table(metrics)) {
        private$baseline_metrics <- list(
          mape = metrics$mape[1],
          mae = metrics$mae[1],
          rmse = metrics$rmse[1],
          set_date = Sys.Date()
        )
      } else {
        checkmate::assert_list(metrics)
        private$baseline_metrics <- c(metrics, list(set_date = Sys.Date()))
      }

      message(sprintf(
        "Baseline set: MAPE=%.2f%%, MAE=%.2f, RMSE=%.2f",
        private$baseline_metrics$mape,
        private$baseline_metrics$mae,
        private$baseline_metrics$rmse
      ))

      invisible(self)
    },

    #' @description Calculate baseline from historical data
    #' @param dt data.table with actual/predicted and dates
    #' @param baseline_period Number of days for baseline
    #' @return Invisible self
    calculate_baseline = function(dt, baseline_period = 30) {
      checkmate::assert_data_table(dt)

      # Use most recent baseline_period days
      max_date <- max(as.Date(dt$DataHora))
      min_date <- max_date - baseline_period

      baseline_data <- dt[as.Date(DataHora) >= min_date &
                          as.Date(DataHora) <= max_date]

      if (nrow(baseline_data) < private$config$min_observations) {
        warning(sprintf(
          "Only %d observations in baseline period, need %d",
          nrow(baseline_data), private$config$min_observations
        ))
      }

      baseline_metrics <- calculate_all_metrics(
        baseline_data$actual,
        baseline_data$predicted
      )

      self$set_baseline(baseline_metrics)
    },

    #' @description Detect drift in current metrics
    #' @param current_metrics List or data.table with current metrics
    #' @return List with drift detection results
    detect = function(current_metrics) {
      if (is.null(private$baseline_metrics)) {
        stop("Baseline not set. Call set_baseline() or calculate_baseline() first.")
      }

      if (is.data.table(current_metrics)) {
        current <- list(
          mape = current_metrics$mape[1],
          mae = current_metrics$mae[1],
          rmse = current_metrics$rmse[1]
        )
      } else {
        current <- current_metrics
      }

      # Calculate ratios
      mape_ratio <- current$mape / private$baseline_metrics$mape
      mae_ratio <- current$mae / private$baseline_metrics$mae
      rmse_ratio <- current$rmse / private$baseline_metrics$rmse

      # Check thresholds
      drift_detected <- FALSE
      reasons <- character()
      metrics_exceeded <- list()

      if (mape_ratio > private$config$threshold_mape) {
        drift_detected <- TRUE
        reasons <- c(reasons, sprintf(
          "MAPE ratio %.2f exceeds threshold %.2f",
          mape_ratio, private$config$threshold_mape
        ))
        metrics_exceeded$mape <- TRUE
      }

      if (mae_ratio > private$config$threshold_mae) {
        drift_detected <- TRUE
        reasons <- c(reasons, sprintf(
          "MAE ratio %.2f exceeds threshold %.2f",
          mae_ratio, private$config$threshold_mae
        ))
        metrics_exceeded$mae <- TRUE
      }

      if (rmse_ratio > private$config$threshold_rmse) {
        drift_detected <- TRUE
        reasons <- c(reasons, sprintf(
          "RMSE ratio %.2f exceeds threshold %.2f",
          rmse_ratio, private$config$threshold_rmse
        ))
        metrics_exceeded$rmse <- TRUE
      }

      result <- list(
        drift_detected = drift_detected,
        drift_type = if (drift_detected) "gradual" else "none",
        reasons = reasons,
        metrics_exceeded = names(metrics_exceeded),
        ratios = list(
          mape = mape_ratio,
          mae = mae_ratio,
          rmse = rmse_ratio
        ),
        baseline = private$baseline_metrics,
        current = current,
        thresholds = list(
          mape = private$config$threshold_mape,
          mae = private$config$threshold_mae,
          rmse = private$config$threshold_rmse
        ),
        timestamp = Sys.time()
      )

      # Record alert if drift detected
      if (drift_detected) {
        self$record_alert(result)
      }

      result
    },

    #' @description Detect drift from raw data
    #' @param dt data.table with actual/predicted
    #' @return List with drift detection results
    detect_from_data = function(dt) {
      current_metrics <- calculate_all_metrics(dt$actual, dt$predicted)
      self$detect(current_metrics)
    },

    #' @description Update history with daily metrics
    #' @param date Date
    #' @param metrics Metrics for the day
    update_history = function(date, metrics) {
      new_row <- data.table::data.table(
        date = as.Date(date),
        mape = metrics$mape,
        mae = metrics$mae,
        rmse = metrics$rmse,
        n_obs = metrics$n_observations %||% NA_integer_
      )

      private$history <- data.table::rbindlist(
        list(private$history, new_row)
      )

      # Detect sudden drift
      if (private$config$detect_sudden && nrow(private$history) > 1) {
        self$detect_sudden_drift()
      }

      invisible(self)
    },

    #' @description Detect sudden (day-over-day) drift
    #' @return List with sudden drift results or NULL
    detect_sudden_drift = function() {
      if (nrow(private$history) < 2) {
        return(NULL)
      }

      last_two <- tail(private$history, 2)
      prev <- last_two[1]
      curr <- last_two[2]

      # Check for sudden spike
      mape_change <- curr$mape / prev$mape
      threshold <- private$config$sudden_threshold_multiplier

      if (mape_change > threshold) {
        result <- list(
          drift_detected = TRUE,
          drift_type = "sudden",
          reasons = sprintf(
            "Sudden MAPE increase: %.2f%% -> %.2f%% (%.1fx)",
            prev$mape, curr$mape, mape_change
          ),
          previous_date = prev$date,
          current_date = curr$date
        )

        self$record_alert(result)
        return(result)
      }

      NULL
    },

    #' @description Record drift alert
    #' @param result Detection result
    record_alert = function(result) {
      # Check cooldown
      if (nrow(private$alerts) > 0) {
        last_alert <- max(private$alerts$timestamp)
        days_since <- as.numeric(difftime(Sys.time(), last_alert, units = "days"))

        if (days_since < private$config$alert_cooldown_days) {
          return(invisible(self))
        }
      }

      # Record new alert
      for (metric in result$metrics_exceeded %||% "mape") {
        new_alert <- data.table::data.table(
          timestamp = Sys.time(),
          type = result$drift_type,
          metric = metric,
          baseline_value = private$baseline_metrics[[metric]] %||% NA_real_,
          current_value = result$current[[metric]] %||% NA_real_,
          ratio = result$ratios[[metric]] %||% NA_real_,
          threshold = result$thresholds[[metric]] %||% NA_real_
        )

        private$alerts <- data.table::rbindlist(
          list(private$alerts, new_alert)
        )
      }

      invisible(self)
    },

    #' @description Get rolling window metrics
    #' @param window_days Window size (uses config if NULL)
    #' @return data.table with rolling metrics
    get_rolling_metrics = function(window_days = NULL) {
      window <- window_days %||% private$config$window

      if (nrow(private$history) < window) {
        warning(sprintf(
          "Only %d days of history, need %d for full window",
          nrow(private$history), window
        ))
      }

      # Calculate rolling means
      history <- data.table::copy(private$history)
      data.table::setorder(history, date)

      history[, `:=`(
        mape_rolling = data.table::frollmean(mape, n = window, align = "right"),
        mae_rolling = data.table::frollmean(mae, n = window, align = "right"),
        rmse_rolling = data.table::frollmean(rmse, n = window, align = "right")
      )]

      history
    },

    #' @description Get alerts history
    #' @return data.table with alerts
    get_alerts = function() {
      data.table::copy(private$alerts)
    },

    #' @description Get history
    #' @return data.table with daily metrics
    get_history = function() {
      data.table::copy(private$history)
    },

    #' @description Get baseline metrics
    #' @return List with baseline
    get_baseline = function() {
      private$baseline_metrics
    },

    #' @description Get configuration
    #' @return Configuration list
    get_config = function() {
      private$config
    },

    #' @description Reset detector
    reset = function() {
      private$baseline_metrics <- NULL
      private$history <- data.table::data.table(
        date = as.Date(character()),
        mape = numeric(),
        mae = numeric(),
        rmse = numeric(),
        n_obs = integer()
      )
      private$alerts <- data.table::data.table(
        timestamp = as.POSIXct(character()),
        type = character(),
        metric = character(),
        baseline_value = numeric(),
        current_value = numeric(),
        ratio = numeric(),
        threshold = numeric()
      )
      invisible(self)
    },

    #' @description Print summary
    print = function() {
      cat("DriftDetector\n")
      cat(sprintf("  Window: %d days\n", private$config$window))
      cat(sprintf("  Thresholds: MAPE=%.2f, MAE=%.2f, RMSE=%.2f\n",
                  private$config$threshold_mape,
                  private$config$threshold_mae,
                  private$config$threshold_rmse))

      if (!is.null(private$baseline_metrics)) {
        cat(sprintf("  Baseline MAPE: %.2f%%\n", private$baseline_metrics$mape))
      } else {
        cat("  Baseline: Not set\n")
      }

      cat(sprintf("  History: %d days\n", nrow(private$history)))
      cat(sprintf("  Alerts: %d\n", nrow(private$alerts)))

      invisible(self)
    }
  )
)
```

### Convenience Functions

```r
#' Create a drift detector with common settings
#'
#' @param preset Preset configuration: "strict", "moderate", "relaxed"
#' @param ... Additional parameters
#' @return DriftDetector instance
#' @export
create_drift_detector <- function(preset = "moderate", ...) {
  presets <- list(
    strict = list(
      threshold_mape = 1.1,
      threshold_mae = 1.1,
      threshold_rmse = 1.15
    ),
    moderate = list(
      threshold_mape = 1.2,
      threshold_mae = 1.15,
      threshold_rmse = 1.25
    ),
    relaxed = list(
      threshold_mape = 1.5,
      threshold_mae = 1.3,
      threshold_rmse = 1.4
    )
  )

  if (!preset %in% names(presets)) {
    stop(sprintf("Unknown preset: %s", preset))
  }

  config <- presets[[preset]]

  DriftDetector$new(
    threshold_mape = config$threshold_mape,
    threshold_mae = config$threshold_mae,
    threshold_rmse = config$threshold_rmse,
    ...
  )
}
```

### Usage Example

```r
# Create detector
detector <- DriftDetector$new(
  window = 30,
  threshold_mape = 1.2  # 20% increase triggers drift
)

# Set baseline from training period
baseline_metrics <- list(
  mape = 5.0,
  mae = 50,
  rmse = 65
)
detector$set_baseline(baseline_metrics)

# Or calculate from data
detector$calculate_baseline(historical_data, baseline_period = 30)

# Detect drift in current metrics
current_metrics <- list(
  mape = 6.5,  # 30% increase
  mae = 55,
  rmse = 70
)

result <- detector$detect(current_metrics)
# $drift_detected: TRUE
# $reasons: "MAPE ratio 1.30 exceeds threshold 1.20"
# $ratios$mape: 1.3

# Track daily metrics
detector$update_history(Sys.Date(), current_metrics)

# Get rolling metrics
rolling <- detector$get_rolling_metrics()

# Get alerts
alerts <- detector$get_alerts()

# Use preset
detector <- create_drift_detector("strict")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | Default thresholds |
| TC-002 | set_baseline() | Baseline stored |
| TC-003 | calculate_baseline() | Calculated from data |
| TC-004 | detect() no drift | drift_detected=FALSE |
| TC-005 | detect() MAPE drift | drift_detected=TRUE |
| TC-006 | detect() multiple metrics | Multiple reasons |
| TC-007 | detect_sudden_drift() | Sudden spike detected |
| TC-008 | update_history() | History updated |
| TC-009 | get_rolling_metrics() | Rolling averages |
| TC-010 | Alert cooldown | No duplicate alerts |
| TC-011 | create_drift_detector() presets | Preset applied |

---

## Dependencies

- PC-047-07: Metrics Calculator

---

## Definition of Done

- [ ] DriftDetector R6 class implemented
- [ ] Gradual and sudden drift detection
- [ ] Rolling window metrics
- [ ] Alert recording with cooldown
- [ ] Preset configurations
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Drift indicates need for model retraining
- Sudden drift may indicate data quality issues
- Alert cooldown prevents notification spam
- Consider adding trend detection (improving vs degrading)
