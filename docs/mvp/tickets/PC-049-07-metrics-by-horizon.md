# PC-049-07: Metrics by Horizon

**Epic:** [EPIC-07: Evaluation Layer](../epics/EPIC-07-evaluation-layer.md)
**Task Reference:** T-07.3
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Implement metrics calculation by forecast horizon (D+0 through D+8), enabling analysis of forecast accuracy degradation as the prediction horizon increases.

---

## Acceptance Criteria

- [ ] Horizon classification function implemented
- [ ] Metrics by horizon calculation
- [ ] Support for D+0 to D+8 horizons
- [ ] Hourly breakdown within horizons
- [ ] Integration with MetricsCalculator

---

## Technical Specification

### File Location
```
R/evaluation/metrics_horizon.R
```

### Horizon Classification

```r
#' Calculate forecast horizon from reference and target dates
#'
#' @param reference_datetime Reference datetime (when forecast was made)
#' @param target_datetime Target datetime (what was being forecasted)
#' @return Integer horizon (0 = same day, 1 = next day, etc.)
#' @export
calculate_horizon <- function(reference_datetime, target_datetime) {
  checkmate::assert_posixct(reference_datetime)
  checkmate::assert_posixct(target_datetime)

  ref_date <- as.Date(reference_datetime)
  target_date <- as.Date(target_datetime)

  as.integer(target_date - ref_date)
}


#' Classify horizon from data.table
#'
#' @param dt data.table with reference_datetime and target_datetime columns
#' @param reference_col Name of reference datetime column
#' @param target_col Name of target datetime column
#' @return Integer vector of horizons
#' @export
classify_horizon <- function(dt,
                              reference_col = "reference_datetime",
                              target_col = "DataHora") {
  checkmate::assert_data_table(dt)

  calculate_horizon(dt[[reference_col]], dt[[target_col]])
}


#' Add horizon column to data.table
#'
#' @param dt data.table with datetime columns
#' @param reference_col Reference datetime column name
#' @param target_col Target datetime column name
#' @return data.table with horizon column added (modifies in place)
#' @export
add_horizon_column <- function(dt,
                                reference_col = "reference_datetime",
                                target_col = "DataHora") {
  dt[, horizon := calculate_horizon(
    get(reference_col),
    get(target_col)
  )]

  # Add horizon label

  dt[, horizon_label := sprintf("D+%d", horizon)]

  invisible(dt)
}
```

### Metrics by Horizon Calculator

```r
#' Calculate metrics by forecast horizon
#'
#' @param dt data.table with horizon, actual, predicted columns
#' @param horizons Optional vector of horizons to include
#' @return data.table with metrics by horizon
#' @export
calculate_metrics_by_horizon <- function(dt, horizons = 0:8) {
  checkmate::assert_data_table(dt)
  checkmate::assert_true("horizon" %in% names(dt))
  checkmate::assert_true("actual" %in% names(dt))
  checkmate::assert_true("predicted" %in% names(dt))

  # Filter to specified horizons
  dt_filtered <- dt[horizon %in% horizons]

  # Calculate metrics by horizon
  metrics <- dt_filtered[, {
    m <- calculate_all_metrics(actual, predicted)
    data.table::data.table(
      mape = m$mape,
      mae = m$mae,
      rmse = m$rmse,
      mbe = m$mbe,
      max_abs_dev = m$max_abs_deviation,
      n_obs = m$n_observations
    )
  }, by = horizon]

  # Order by horizon
  data.table::setorder(metrics, horizon)

  # Add horizon label
  metrics[, horizon_label := sprintf("D+%d", horizon)]

  metrics
}


#' Calculate metrics by horizon and hour
#'
#' @param dt data.table with horizon, actual, predicted, DataHora
#' @param horizons Optional vector of horizons to include
#' @return data.table with metrics by horizon and hour
#' @export
calculate_metrics_by_horizon_hour <- function(dt, horizons = 0:8) {
  checkmate::assert_data_table(dt)
  checkmate::assert_true("DataHora" %in% names(dt))

  # Extract hour
  dt <- data.table::copy(dt)
  dt[, hour := data.table::hour(DataHora)]

  # Filter to specified horizons
  dt_filtered <- dt[horizon %in% horizons]

  # Calculate metrics by horizon and hour
  metrics <- dt_filtered[, {
    m <- calculate_all_metrics(actual, predicted)
    data.table::data.table(
      mape = m$mape,
      mae = m$mae,
      rmse = m$rmse,
      n_obs = m$n_observations
    )
  }, by = .(horizon, hour)]

  # Order
  data.table::setorder(metrics, horizon, hour)

  metrics
}


#' Calculate metrics by horizon and area
#'
#' @param dt data.table with horizon, actual, predicted, area_code
#' @param horizons Optional vector of horizons to include
#' @return data.table with metrics by horizon and area
#' @export
calculate_metrics_by_horizon_area <- function(dt, horizons = 0:8) {
  checkmate::assert_data_table(dt)
  checkmate::assert_true("area_code" %in% names(dt))

  # Filter to specified horizons
  dt_filtered <- dt[horizon %in% horizons]

  # Calculate metrics by horizon and area
  metrics <- dt_filtered[, {
    m <- calculate_all_metrics(actual, predicted)
    data.table::data.table(
      mape = m$mape,
      mae = m$mae,
      rmse = m$rmse,
      n_obs = m$n_observations
    )
  }, by = .(area_code, horizon)]

  # Order
  data.table::setorder(metrics, area_code, horizon)

  metrics
}
```

### Horizon Analysis Functions

```r
#' Analyze forecast degradation by horizon
#'
#' @param metrics_by_horizon Output from calculate_metrics_by_horizon
#' @return data.table with degradation analysis
#' @export
analyze_horizon_degradation <- function(metrics_by_horizon) {
  checkmate::assert_data_table(metrics_by_horizon)

  # Get baseline (D+0 or first available)
  baseline <- metrics_by_horizon[1]

  # Calculate degradation relative to baseline
  analysis <- data.table::copy(metrics_by_horizon)
  analysis[, mape_degradation := mape / baseline$mape]
  analysis[, mae_degradation := mae / baseline$mae]
  analysis[, rmse_degradation := rmse / baseline$rmse]

  # Calculate average degradation per day
  if (nrow(analysis) > 1) {
    mape_slope <- coef(lm(mape ~ horizon, data = analysis))[2]
    analysis[, avg_daily_mape_increase := mape_slope]
  }

  analysis
}


#' Find horizon with acceptable accuracy
#'
#' @param metrics_by_horizon Output from calculate_metrics_by_horizon
#' @param max_mape Maximum acceptable MAPE
#' @return Maximum horizon with acceptable accuracy
#' @export
find_acceptable_horizon <- function(metrics_by_horizon, max_mape = 5.0) {
  acceptable <- metrics_by_horizon[mape <= max_mape]

  if (nrow(acceptable) == 0) {
    warning("No horizon meets MAPE threshold")
    return(NA_integer_)
  }

  max(acceptable$horizon)
}


#' Summarize metrics across all horizons
#'
#' @param metrics_by_horizon Output from calculate_metrics_by_horizon
#' @return List with summary statistics
#' @export
summarize_horizons <- function(metrics_by_horizon) {
  list(
    n_horizons = nrow(metrics_by_horizon),
    min_mape = min(metrics_by_horizon$mape),
    max_mape = max(metrics_by_horizon$mape),
    avg_mape = mean(metrics_by_horizon$mape),
    best_horizon = metrics_by_horizon[which.min(mape), horizon],
    worst_horizon = metrics_by_horizon[which.max(mape), horizon],
    mape_range = max(metrics_by_horizon$mape) - min(metrics_by_horizon$mape),
    total_observations = sum(metrics_by_horizon$n_obs)
  )
}
```

### HorizonAnalyzer R6 Class

```r
#' @title HorizonAnalyzer
#' @description R6 class for analyzing metrics by forecast horizon
#' @export
HorizonAnalyzer <- R6::R6Class(
  "HorizonAnalyzer",
  private = list(
    horizons = NULL,
    metrics_cache = NULL
  ),
  public = list(
    #' @description Initialize analyzer
    #' @param horizons Vector of horizons to analyze
    initialize = function(horizons = 0:8) {
      private$horizons <- horizons
      private$metrics_cache <- list()
    },

    #' @description Analyze forecast data by horizon
    #' @param dt data.table with forecast data
    #' @return data.table with metrics by horizon
    analyze = function(dt) {
      if (!"horizon" %in% names(dt)) {
        stop("Data must have 'horizon' column. Use add_horizon_column() first.")
      }

      metrics <- calculate_metrics_by_horizon(dt, private$horizons)
      private$metrics_cache$last <- metrics

      metrics
    },

    #' @description Get degradation analysis
    #' @return data.table with degradation metrics
    get_degradation = function() {
      if (is.null(private$metrics_cache$last)) {
        stop("No analysis performed yet. Call analyze() first.")
      }

      analyze_horizon_degradation(private$metrics_cache$last)
    },

    #' @description Get summary
    #' @return List with summary statistics
    summary = function() {
      if (is.null(private$metrics_cache$last)) {
        stop("No analysis performed yet. Call analyze() first.")
      }

      summarize_horizons(private$metrics_cache$last)
    },

    #' @description Plot metrics by horizon (returns data for plotting)
    #' @return data.table formatted for plotting
    get_plot_data = function() {
      if (is.null(private$metrics_cache$last)) {
        stop("No analysis performed yet. Call analyze() first.")
      }

      metrics <- private$metrics_cache$last

      # Melt for long format
      data.table::melt(
        metrics,
        id.vars = c("horizon", "horizon_label"),
        measure.vars = c("mape", "mae", "rmse"),
        variable.name = "metric",
        value.name = "value"
      )
    },

    #' @description Print summary
    print = function() {
      cat("HorizonAnalyzer\n")
      cat(sprintf("  Horizons: D+%d to D+%d\n",
                  min(private$horizons),
                  max(private$horizons)))
      if (!is.null(private$metrics_cache$last)) {
        s <- summarize_horizons(private$metrics_cache$last)
        cat(sprintf("  MAPE range: %.2f%% - %.2f%%\n",
                    s$min_mape, s$max_mape))
      }
      invisible(self)
    }
  )
)
```

### Usage Example

```r
# Create test data with horizons
dt <- data.table::data.table(
  reference_datetime = rep(as.POSIXct("2024-01-15 00:00:00"), 432),
  DataHora = rep(seq(
    as.POSIXct("2024-01-15 00:00:00"),
    as.POSIXct("2024-01-23 23:00:00"),
    by = "hour"
  ), each = 1),
  actual = rnorm(432, mean = 1000, sd = 100),
  predicted = rnorm(432, mean = 1000, sd = 120)
)

# Add horizon column
add_horizon_column(dt)

# Calculate metrics by horizon
metrics <- calculate_metrics_by_horizon(dt)
#    horizon     mape      mae     rmse   mbe max_abs_dev n_obs horizon_label
# 1:       0 10.23456 102.3456 125.4567  5.67      312.45    48           D+0
# 2:       1 11.34567 113.4567 138.5678  8.12      345.67    48           D+1
# ...

# Analyze degradation
degradation <- analyze_horizon_degradation(metrics)

# Find acceptable horizon
max_horizon <- find_acceptable_horizon(metrics, max_mape = 12.0)
# 5

# Use R6 analyzer
analyzer <- HorizonAnalyzer$new(horizons = 0:5)
analyzer$analyze(dt)
analyzer$summary()
# $n_horizons: 6
# $min_mape: 10.23
# $max_mape: 15.67
# $best_horizon: 0
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | calculate_horizon() same day | 0 |
| TC-002 | calculate_horizon() next day | 1 |
| TC-003 | calculate_horizon() week ahead | 7 |
| TC-004 | add_horizon_column() | Column added |
| TC-005 | calculate_metrics_by_horizon() | Metrics per horizon |
| TC-006 | calculate_metrics_by_horizon_hour() | By horizon and hour |
| TC-007 | analyze_horizon_degradation() | Degradation ratios |
| TC-008 | find_acceptable_horizon() | Correct horizon |
| TC-009 | HorizonAnalyzer$analyze() | Analysis complete |
| TC-010 | summarize_horizons() | Summary statistics |

---

## Dependencies

- PC-047-07: Metrics Calculator

---

## Definition of Done

- [ ] Horizon classification implemented
- [ ] Metrics by horizon working
- [ ] Degradation analysis implemented
- [ ] HorizonAnalyzer R6 class implemented
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- D+0 is same-day forecast (intraday updates)
- Typical horizons are D+0 through D+8 for weekly planning
- Forecast accuracy typically degrades with horizon
- Consider adding confidence intervals by horizon
