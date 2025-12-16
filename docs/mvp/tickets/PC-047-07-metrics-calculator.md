# PC-047-07: Metrics Calculator

**Epic:** [EPIC-07: Evaluation Layer](../epics/EPIC-07-evaluation-layer.md)
**Task Reference:** T-07.1
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Implement the core metrics calculation functions for forecast evaluation, including MAPE, MAE, RMSE, percentiles, and maximum deviations.

---

## Acceptance Criteria

- [ ] Metrics module created in `R/evaluation/metrics.R`
- [ ] Core metric functions: MAPE, MAE, RMSE
- [ ] Percentile calculations for error distribution
- [ ] Maximum deviation (absolute and relative)
- [ ] Aggregate function for all metrics
- [ ] Handle NA values gracefully

---

## Technical Specification

### File Location
```
R/evaluation/metrics.R
```

### Core Metric Functions

```r
#' Calculate Mean Absolute Percentage Error (MAPE)
#'
#' @param actual Numeric vector of actual values
#' @param predicted Numeric vector of predicted values
#' @param na.rm Remove NA values (default TRUE)
#' @return MAPE as percentage
#' @export
#' @examples
#' calculate_mape(c(100, 200, 300), c(110, 190, 310))
calculate_mape <- function(actual, predicted, na.rm = TRUE) {
  checkmate::assert_numeric(actual)
  checkmate::assert_numeric(predicted)
  checkmate::assert_true(length(actual) == length(predicted))

  # Avoid division by zero
  valid <- actual != 0 & !is.na(actual) & !is.na(predicted)

  if (sum(valid) == 0) {
    return(NA_real_)
  }

  mean(abs((actual[valid] - predicted[valid]) / actual[valid]), na.rm = na.rm) * 100
}


#' Calculate Mean Absolute Error (MAE)
#'
#' @param actual Numeric vector of actual values
#' @param predicted Numeric vector of predicted values
#' @param na.rm Remove NA values (default TRUE)
#' @return MAE in original units
#' @export
calculate_mae <- function(actual, predicted, na.rm = TRUE) {
  checkmate::assert_numeric(actual)
  checkmate::assert_numeric(predicted)
  checkmate::assert_true(length(actual) == length(predicted))

  mean(abs(actual - predicted), na.rm = na.rm)
}


#' Calculate Root Mean Square Error (RMSE)
#'
#' @param actual Numeric vector of actual values
#' @param predicted Numeric vector of predicted values
#' @param na.rm Remove NA values (default TRUE)
#' @return RMSE in original units
#' @export
calculate_rmse <- function(actual, predicted, na.rm = TRUE) {
  checkmate::assert_numeric(actual)
  checkmate::assert_numeric(predicted)
  checkmate::assert_true(length(actual) == length(predicted))

  sqrt(mean((actual - predicted)^2, na.rm = na.rm))
}


#' Calculate Mean Bias Error (MBE)
#'
#' @param actual Numeric vector of actual values
#' @param predicted Numeric vector of predicted values
#' @param na.rm Remove NA values (default TRUE)
#' @return MBE (positive = over-prediction, negative = under-prediction)
#' @export
calculate_mbe <- function(actual, predicted, na.rm = TRUE) {
  checkmate::assert_numeric(actual)
  checkmate::assert_numeric(predicted)

  mean(predicted - actual, na.rm = na.rm)
}


#' Calculate error percentiles
#'
#' @param errors Numeric vector of errors (actual - predicted)
#' @param probs Probabilities for quantiles
#' @param absolute Use absolute errors (default TRUE)
#' @return Named numeric vector of percentiles
#' @export
calculate_percentiles <- function(errors,
                                   probs = c(0.05, 0.25, 0.5, 0.75, 0.95),
                                   absolute = TRUE) {
  checkmate::assert_numeric(errors)
  checkmate::assert_numeric(probs)

  if (absolute) {
    errors <- abs(errors)
  }

  quantile(errors, probs = probs, na.rm = TRUE)
}


#' Calculate maximum absolute deviation
#'
#' @param actual Numeric vector of actual values
#' @param predicted Numeric vector of predicted values
#' @return Maximum absolute error
#' @export
calculate_max_abs_deviation <- function(actual, predicted) {
  checkmate::assert_numeric(actual)
  checkmate::assert_numeric(predicted)

  max(abs(actual - predicted), na.rm = TRUE)
}


#' Calculate maximum relative deviation
#'
#' @param actual Numeric vector of actual values
#' @param predicted Numeric vector of predicted values
#' @return Maximum relative error as percentage
#' @export
calculate_max_rel_deviation <- function(actual, predicted) {
  checkmate::assert_numeric(actual)
  checkmate::assert_numeric(predicted)

  valid <- actual != 0 & !is.na(actual) & !is.na(predicted)

  if (sum(valid) == 0) {
    return(NA_real_)
  }

  max(abs((actual[valid] - predicted[valid]) / actual[valid]), na.rm = TRUE) * 100
}


#' Calculate symmetric MAPE (sMAPE)
#'
#' @param actual Numeric vector of actual values
#' @param predicted Numeric vector of predicted values
#' @param na.rm Remove NA values
#' @return sMAPE as percentage
#' @export
calculate_smape <- function(actual, predicted, na.rm = TRUE) {
  checkmate::assert_numeric(actual)
  checkmate::assert_numeric(predicted)

  denominator <- abs(actual) + abs(predicted)
  valid <- denominator != 0 & !is.na(actual) & !is.na(predicted)

  if (sum(valid) == 0) {
    return(NA_real_)
  }

  mean(2 * abs(actual[valid] - predicted[valid]) / denominator[valid],
       na.rm = na.rm) * 100
}
```

### Aggregate Metrics Function

```r
#' Calculate all forecast metrics
#'
#' @param actual Numeric vector of actual values
#' @param predicted Numeric vector of predicted values
#' @param percentile_probs Probabilities for percentile calculation
#' @return List with all metrics
#' @export
#' @examples
#' metrics <- calculate_all_metrics(actual, predicted)
#' metrics$mape
calculate_all_metrics <- function(actual,
                                   predicted,
                                   percentile_probs = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
  checkmate::assert_numeric(actual)
  checkmate::assert_numeric(predicted)
  checkmate::assert_true(length(actual) == length(predicted))

  errors <- actual - predicted

  list(
    mape = calculate_mape(actual, predicted),
    mae = calculate_mae(actual, predicted),
    rmse = calculate_rmse(actual, predicted),
    mbe = calculate_mbe(actual, predicted),
    smape = calculate_smape(actual, predicted),
    percentiles = calculate_percentiles(errors, probs = percentile_probs),
    max_abs_deviation = calculate_max_abs_deviation(actual, predicted),
    max_rel_deviation = calculate_max_rel_deviation(actual, predicted),
    n_observations = sum(!is.na(actual) & !is.na(predicted)),
    correlation = cor(actual, predicted, use = "complete.obs")
  )
}


#' Calculate metrics and return as data.table row
#'
#' @param actual Numeric vector of actual values
#' @param predicted Numeric vector of predicted values
#' @param group_vars Optional named list of grouping variables to include
#' @return data.table with one row of metrics
#' @export
calculate_metrics_dt <- function(actual, predicted, group_vars = NULL) {
  metrics <- calculate_all_metrics(actual, predicted)

  dt <- data.table::data.table(
    mape = metrics$mape,
    mae = metrics$mae,
    rmse = metrics$rmse,
    mbe = metrics$mbe,
    smape = metrics$smape,
    p05 = metrics$percentiles["5%"],
    p25 = metrics$percentiles["25%"],
    p50 = metrics$percentiles["50%"],
    p75 = metrics$percentiles["75%"],
    p95 = metrics$percentiles["95%"],
    max_abs_dev = metrics$max_abs_deviation,
    max_rel_dev = metrics$max_rel_deviation,
    n_obs = metrics$n_observations,
    correlation = metrics$correlation
  )

  if (!is.null(group_vars)) {
    for (name in names(group_vars)) {
      dt[[name]] <- group_vars[[name]]
    }
    data.table::setcolorder(dt, c(names(group_vars), setdiff(names(dt), names(group_vars))))
  }

  dt
}
```

### MetricsCalculator R6 Class

```r
#' @title MetricsCalculator
#' @description R6 class for calculating forecast evaluation metrics
#' @export
MetricsCalculator <- R6::R6Class(
  "MetricsCalculator",
  private = list(
    config = NULL,
    results_cache = NULL
  ),
  public = list(
    #' @description Initialize calculator
    #' @param config Configuration list
    initialize = function(config = list()) {
      private$config <- private$apply_default_config(config)
      private$results_cache <- list()
    },

    #' @description Calculate metrics for forecast data
    #' @param dt data.table with actual and predicted columns
    #' @param actual_col Name of actual column
    #' @param predicted_col Name of predicted column
    #' @param group_by Optional grouping columns
    #' @return data.table with metrics
    calculate = function(dt,
                         actual_col = "actual",
                         predicted_col = "predicted",
                         group_by = NULL) {
      checkmate::assert_data_table(dt)
      checkmate::assert_string(actual_col)
      checkmate::assert_string(predicted_col)

      if (!actual_col %in% names(dt)) {
        stop(sprintf("Column '%s' not found", actual_col))
      }
      if (!predicted_col %in% names(dt)) {
        stop(sprintf("Column '%s' not found", predicted_col))
      }

      if (is.null(group_by)) {
        # Single group calculation
        metrics <- calculate_metrics_dt(
          dt[[actual_col]],
          dt[[predicted_col]]
        )
      } else {
        # Grouped calculation
        metrics <- dt[,
          calculate_metrics_dt(
            get(actual_col),
            get(predicted_col)
          ),
          by = group_by
        ]
      }

      private$results_cache$last_result <- metrics
      metrics
    },

    #' @description Get summary statistics
    #' @return data.table with summary
    summary = function() {
      if (is.null(private$results_cache$last_result)) {
        stop("No metrics calculated yet. Call calculate() first.")
      }

      private$results_cache$last_result
    },

    #' @description Compare metrics between groups
    #' @param metrics1 First metrics result
    #' @param metrics2 Second metrics result
    #' @return data.table with comparison
    compare = function(metrics1, metrics2) {
      comparison <- data.table::data.table(
        metric = c("mape", "mae", "rmse"),
        value1 = c(metrics1$mape, metrics1$mae, metrics1$rmse),
        value2 = c(metrics2$mape, metrics2$mae, metrics2$rmse)
      )
      comparison[, diff := value2 - value1]
      comparison[, pct_change := (value2 - value1) / value1 * 100]

      comparison
    },

    #' @description Get configuration
    get_config = function() {
      private$config
    },

    #' @description Print summary
    print = function() {
      cat("MetricsCalculator\n")
      if (!is.null(private$results_cache$last_result)) {
        cat(sprintf("  Last result: %d rows\n",
                    nrow(private$results_cache$last_result)))
      }
      invisible(self)
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        percentile_probs = c(0.05, 0.25, 0.5, 0.75, 0.95),
        handle_zeros = "exclude"  # or "replace"
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    }
  )
)
```

### Usage Example

```r
# Direct function usage
actual <- c(100, 200, 300, 400, 500)
predicted <- c(105, 195, 310, 390, 510)

mape <- calculate_mape(actual, predicted)
# 3.2

metrics <- calculate_all_metrics(actual, predicted)
# $mape: 3.2
# $mae: 10
# $rmse: 10.95
# ...

# R6 class usage
calculator <- MetricsCalculator$new()

dt <- data.table::data.table(
  area_code = rep(c("RJ", "SP"), each = 100),
  actual = rnorm(200, mean = 1000, sd = 100),
  predicted = rnorm(200, mean = 1000, sd = 110)
)

# Calculate by area
metrics_by_area <- calculator$calculate(dt, group_by = "area_code")
#    area_code     mape      mae     rmse ...
# 1:        RJ 10.23456 102.3456 125.4567 ...
# 2:        SP  9.87654  98.7654 118.9876 ...

# Calculate overall
metrics_overall <- calculator$calculate(dt)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | calculate_mape() basic | Correct percentage |
| TC-002 | calculate_mape() with zeros | Handles division by zero |
| TC-003 | calculate_mae() basic | Correct absolute error |
| TC-004 | calculate_rmse() basic | Correct RMSE |
| TC-005 | calculate_mbe() positive bias | Positive value |
| TC-006 | calculate_mbe() negative bias | Negative value |
| TC-007 | calculate_percentiles() default | 5 percentiles |
| TC-008 | calculate_max_abs_deviation() | Maximum error |
| TC-009 | calculate_max_rel_deviation() | Percentage |
| TC-010 | calculate_all_metrics() | All metrics in list |
| TC-011 | calculate_metrics_dt() | data.table row |
| TC-012 | MetricsCalculator$calculate() | Grouped metrics |
| TC-013 | Handle NA values | No errors |

---

## Dependencies

None (foundation module)

---

## Definition of Done

- [ ] All metric functions implemented
- [ ] MetricsCalculator R6 class implemented
- [ ] NA handling tested
- [ ] Zero handling for MAPE tested
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- MAPE can be undefined when actual = 0, handle gracefully
- Consider weighted variants in future (WMAPE)
- Percentiles use absolute errors by default
- MBE indicates systematic bias direction
