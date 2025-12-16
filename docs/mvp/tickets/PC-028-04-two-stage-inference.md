# PC-028-04: Two-Stage Inference Support

**Epic:** [EPIC-04: Model Layer - Hierarchical Support](../epics/EPIC-04-model-layer-part2.md)
**Task Reference:** T-04.5
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `TwoStageInferenceWorkflow` for D+0 → D+1 pattern, supporting ratio propagation for day completion and tracking of verified vs forecasted periods.

---

## Acceptance Criteria

- [ ] Two-stage inference module added to `R/models/inference.R`
- [ ] `TwoStageInferenceWorkflow` R6 class for D+0/D+1 pattern
- [ ] Support for ratio propagation to complete partial days
- [ ] Verified vs forecasted period tracking
- [ ] Seamless transition from D+0 (intraday) to D+1 (next day)
- [ ] Integration with InferenceWorkflow base class

---

## Technical Specification

### File Location
```
R/models/inference.R (extension)
```

### TwoStageInferenceWorkflow R6 Class

```r
#' @title TwoStageInferenceWorkflow
#' @description Two-stage inference for D+0 → D+1 transition
#'
#' Handles the transition from intraday (D+0) to day-ahead (D+1) forecasting:
#' - D+0: Complete remaining hours using ratio propagation
#' - D+1+: Use full day forecasts
#'
#' Tracks which periods are verified (actual) vs forecasted.
#'
#' @export
TwoStageInferenceWorkflow <- R6::R6Class(
  "TwoStageInferenceWorkflow",
  inherit = InferenceWorkflow,
  private = list(
    d0_model_artifact = NULL,  # Model for D+0 (intraday)
    d1_model_artifact = NULL,  # Model for D+1+ (day-ahead)
    ratio_calculator = NULL,   # For day completion
    current_hour = NULL
  ),
  public = list(
    #' @description Initialize two-stage workflow
    #' @param d0_model_artifact Model for D+0 intraday completion
    #' @param d1_model_artifact Model for D+1 day-ahead forecasts
    #' @param feature_builder Feature builder
    #' @param ratio_calculator Function for ratio propagation
    #' @param config Workflow configuration
    initialize = function(d0_model_artifact = NULL,
                          d1_model_artifact,
                          feature_builder = NULL,
                          ratio_calculator = NULL,
                          config = list()) {
      checkmate::assert_class(d1_model_artifact, "ModelArtifact")
      if (!is.null(d0_model_artifact)) {
        checkmate::assert_class(d0_model_artifact, "ModelArtifact")
      }

      # Initialize parent with D+1 model as primary
      super$initialize(
        model_artifact = d1_model_artifact,
        feature_builder = feature_builder,
        config = config
      )

      private$d0_model_artifact <- d0_model_artifact
      private$d1_model_artifact <- d1_model_artifact
      private$ratio_calculator <- ratio_calculator %||% self$default_ratio_calculator
    },

    #' @description Run two-stage inference
    #' @param anchor_data Anchor data with recent actuals
    #' @param target_date Target forecast date
    #' @param current_datetime Current datetime (for D+0 calculation)
    #' @param horizons Horizons to forecast (0:8)
    #' @param ... Additional arguments
    #' @return TwoStageInferenceResult with combined forecasts
    run = function(anchor_data,
                   target_date,
                   current_datetime = Sys.time(),
                   horizons = 0:8,
                   ...) {
      checkmate::assert_data_table(anchor_data)

      start_time <- Sys.time()
      current_date <- as.Date(current_datetime)
      private$current_hour <- lubridate::hour(current_datetime)

      results <- list()
      period_tracking <- list()

      for (h in horizons) {
        forecast_date <- target_date + h

        if (forecast_date == current_date && h == 0) {
          # D+0: Complete today using ratio propagation
          result <- self$run_d0_inference(
            anchor_data = anchor_data,
            target_date = forecast_date,
            current_hour = private$current_hour,
            ...
          )

          results[[paste0("D+", h)]] <- result$predictions
          period_tracking[[paste0("D+", h)]] <- result$period_info

        } else {
          # D+1+: Full day forecast
          result <- self$run_d1_inference(
            anchor_data = anchor_data,
            target_date = forecast_date,
            horizon = h,
            ...
          )

          results[[paste0("D+", h)]] <- result$predictions
          period_tracking[[paste0("D+", h)]] <- list(
            verified_hours = 0,
            forecasted_hours = 24,
            method = "day_ahead"
          )
        }
      }

      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      TwoStageInferenceResult$new(
        predictions = results,
        period_tracking = period_tracking,
        target_date = target_date,
        current_datetime = current_datetime,
        horizons = horizons,
        d0_model = if (!is.null(private$d0_model_artifact)) {
          private$d0_model_artifact$model_name
        } else NULL,
        d1_model = private$d1_model_artifact$model_name,
        inference_time = elapsed
      )
    },

    #' @description Run D+0 (intraday) inference
    #' @param anchor_data Anchor data
    #' @param target_date Target date (today)
    #' @param current_hour Current hour
    #' @param ... Additional arguments
    #' @return List with predictions and period info
    run_d0_inference = function(anchor_data, target_date, current_hour, ...) {
      message(sprintf("Running D+0 inference (current hour: %d)", current_hour))

      # Get verified actuals for completed hours
      today_actuals <- anchor_data[
        as.Date(DataHora) == target_date &
        lubridate::hour(DataHora) < current_hour
      ]

      verified_load <- sum(today_actuals$CargaGlobal, na.rm = TRUE)
      verified_hours <- nrow(today_actuals)
      remaining_hours <- 24 - current_hour

      # Method 1: Ratio propagation (if we have D+0 model)
      if (!is.null(private$d0_model_artifact)) {
        # Build features for remaining hours
        features <- self$build_features(anchor_data, target_date, ...)

        # Predict remaining hours
        d0_model <- private$d0_model_artifact$get_model()
        remaining_predictions <- d0_model$predict(
          features[lubridate::hour(DataHora) >= current_hour],
          ...
        )

        # Combine verified + forecasted
        predictions <- c(
          today_actuals$CargaGlobal,
          remaining_predictions
        )

      } else {
        # Method 2: Ratio propagation only
        predictions <- private$ratio_calculator(
          verified_load = verified_load,
          verified_hours = verified_hours,
          target_date = target_date,
          anchor_data = anchor_data
        )
      }

      list(
        predictions = predictions,
        period_info = list(
          verified_hours = verified_hours,
          forecasted_hours = remaining_hours,
          verified_load = verified_load,
          method = if (!is.null(private$d0_model_artifact)) "model" else "ratio"
        )
      )
    },

    #' @description Run D+1+ (day-ahead) inference
    #' @param anchor_data Anchor data
    #' @param target_date Target date
    #' @param horizon Forecast horizon
    #' @param ... Additional arguments
    #' @return List with predictions
    run_d1_inference = function(anchor_data, target_date, horizon, ...) {
      message(sprintf("Running D+%d inference for %s", horizon, target_date))

      # Build features
      features <- self$build_features(anchor_data, target_date, ...)

      # Predict
      d1_model <- private$d1_model_artifact$get_model()
      predictions <- d1_model$predict(features, horizon = horizon, ...)

      list(predictions = predictions)
    },

    #' @description Default ratio propagation calculator
    #' @param verified_load Verified load so far
    #' @param verified_hours Hours with verified data
    #' @param target_date Target date
    #' @param anchor_data Historical data for ratio calculation
    #' @return Vector of 24 hourly predictions
    default_ratio_calculator = function(verified_load, verified_hours,
                                        target_date, anchor_data) {
      if (verified_hours == 0) {
        # No verified data - use historical average
        historical_avg <- anchor_data[
          lubridate::wday(DataHora) == lubridate::wday(target_date),
          .(avg = mean(CargaGlobal, na.rm = TRUE)),
          by = lubridate::hour(DataHora)
        ]
        return(historical_avg$avg)
      }

      # Calculate completion ratio from historical patterns
      # Ratio = (24-hour total) / (partial-hour total) for similar days
      similar_days <- anchor_data[
        lubridate::wday(DataHora) == lubridate::wday(target_date)
      ]

      daily_totals <- similar_days[, .(
        partial = sum(CargaGlobal[lubridate::hour(DataHora) < verified_hours]),
        full = sum(CargaGlobal)
      ), by = as.Date(DataHora)]

      avg_ratio <- mean(daily_totals$full / daily_totals$partial, na.rm = TRUE)

      # Estimate full day load
      estimated_daily <- verified_load * avg_ratio

      # Distribute remaining load by historical pattern
      hourly_pattern <- similar_days[, .(
        share = mean(CargaGlobal, na.rm = TRUE)
      ), by = lubridate::hour(DataHora)]
      hourly_pattern[, share := share / sum(share)]

      # Build predictions: verified hours + estimated remaining
      predictions <- rep(NA_real_, 24)
      predictions[1:verified_hours] <- anchor_data[
        as.Date(DataHora) == target_date &
        lubridate::hour(DataHora) < verified_hours
      ]$CargaGlobal

      remaining_load <- estimated_daily - verified_load
      remaining_pattern <- hourly_pattern[hour >= verified_hours]$share
      remaining_pattern <- remaining_pattern / sum(remaining_pattern)

      predictions[(verified_hours + 1):24] <- remaining_load * remaining_pattern

      predictions
    },

    #' @description Set custom ratio calculator
    #' @param calculator Function for ratio calculation
    #' @return Invisible self
    set_ratio_calculator = function(calculator) {
      checkmate::assert_function(calculator)
      private$ratio_calculator <- calculator
      invisible(self)
    }
  )
)
```

### TwoStageInferenceResult Class

```r
#' @title TwoStageInferenceResult
#' @description Result container for two-stage inference
#' @export
TwoStageInferenceResult <- R6::R6Class(
  "TwoStageInferenceResult",
  public = list(
    predictions = NULL,
    period_tracking = NULL,
    target_date = NULL,
    current_datetime = NULL,
    horizons = NULL,
    d0_model = NULL,
    d1_model = NULL,
    inference_time = NULL,
    created_at = NULL,

    #' @description Initialize result
    initialize = function(predictions,
                          period_tracking,
                          target_date,
                          current_datetime,
                          horizons,
                          d0_model,
                          d1_model,
                          inference_time) {
      self$predictions <- predictions
      self$period_tracking <- period_tracking
      self$target_date <- target_date
      self$current_datetime <- current_datetime
      self$horizons <- horizons
      self$d0_model <- d0_model
      self$d1_model <- d1_model
      self$inference_time <- inference_time
      self$created_at <- Sys.time()
    },

    #' @description Get predictions for specific horizon
    #' @param horizon Horizon (0-8)
    #' @return Vector of predictions
    get_horizon = function(horizon) {
      self$predictions[[paste0("D+", horizon)]]
    },

    #' @description Get all predictions as data.table
    #' @return data.table with all forecasts
    as_data_table = function() {
      rows <- list()

      for (h in self$horizons) {
        key <- paste0("D+", h)
        preds <- self$predictions[[key]]
        tracking <- self$period_tracking[[key]]

        forecast_date <- self$target_date + h
        hours <- seq_along(preds) - 1

        rows[[key]] <- data.table::data.table(
          target_date = forecast_date,
          horizon = h,
          hour = hours,
          prediction = preds,
          is_verified = hours < tracking$verified_hours,
          method = tracking$method
        )
      }

      data.table::rbindlist(rows)
    },

    #' @description Get period tracking summary
    #' @return data.table with tracking info
    get_tracking_summary = function() {
      data.table::rbindlist(lapply(names(self$period_tracking), function(key) {
        tracking <- self$period_tracking[[key]]
        data.table::data.table(
          horizon = key,
          verified_hours = tracking$verified_hours,
          forecasted_hours = tracking$forecasted_hours,
          method = tracking$method
        )
      }))
    },

    #' @description Print result
    print = function() {
      cat(sprintf("TwoStageInferenceResult\n"))
      cat(sprintf("  Target date: %s\n", self$target_date))
      cat(sprintf("  Current time: %s\n", self$current_datetime))
      cat(sprintf("  Horizons: D+%d to D+%d\n",
                  min(self$horizons), max(self$horizons)))
      cat(sprintf("  D+0 model: %s\n", self$d0_model %||% "ratio propagation"))
      cat(sprintf("  D+1 model: %s\n", self$d1_model))
      cat(sprintf("  Inference time: %.3f sec\n", self$inference_time))

      # Print tracking summary
      cat("\nPeriod tracking:\n")
      print(self$get_tracking_summary())

      invisible(self)
    }
  )
)
```

### Usage Example

```r
# Load model artifacts
d0_artifact <- ModelArtifact$new()
d0_artifact$load("models/lgbm_d0/latest")

d1_artifact <- ModelArtifact$new()
d1_artifact$load("models/lgbm_d1/latest")

# Create two-stage workflow
workflow <- TwoStageInferenceWorkflow$new(
  d0_model_artifact = d0_artifact,  # Optional: for D+0
  d1_model_artifact = d1_artifact,
  feature_builder = build_features_fn
)

# Run inference at 14:00
result <- workflow$run(
  anchor_data = recent_data,
  target_date = as.Date("2025-01-18"),
  current_datetime = as.POSIXct("2025-01-18 14:00:00"),
  horizons = 0:8
)

# View results
print(result)
# TwoStageInferenceResult
#   Target date: 2025-01-18
#   Current time: 2025-01-18 14:00:00
#   Horizons: D+0 to D+8
#   D+0 model: lgbm_d0
#   D+1 model: lgbm_d1
#   Inference time: 0.876 sec
#
# Period tracking:
#    horizon verified_hours forecasted_hours method
# 1:    D+0             14               10  model
# 2:    D+1              0               24  day_ahead
# ...

# Get D+0 predictions (14 verified + 10 forecasted)
d0_preds <- result$get_horizon(0)

# Export as data.table
forecasts_dt <- result$as_data_table()
#    target_date horizon hour prediction is_verified method
# 1:  2025-01-18       0    0      45000        TRUE  model
# 2:  2025-01-18       0    1      44500        TRUE  model
# ...
# 15: 2025-01-18       0   14      46200       FALSE  model
# ...

# Check what's verified vs forecasted
result$get_tracking_summary()
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with both models | Both stored |
| TC-002 | Initialize D+1 only | Ratio propagation for D+0 |
| TC-003 | run() at start of day | All D+0 forecasted |
| TC-004 | run() at midday | D+0 split verified/forecast |
| TC-005 | run() near end of day | D+0 mostly verified |
| TC-006 | run_d0_inference() with model | Model predictions |
| TC-007 | run_d0_inference() ratio only | Ratio propagation |
| TC-008 | run_d1_inference() | Full day forecast |
| TC-009 | default_ratio_calculator() | Pattern-based completion |
| TC-010 | get_horizon() specific | Correct predictions |
| TC-011 | as_data_table() | All horizons formatted |
| TC-012 | get_tracking_summary() | Period breakdown |
| TC-013 | Custom ratio calculator | Calculator used |

---

## Dependencies

- PC-027-04: InferenceWorkflow (parent class)
- PC-019-03: ModelArtifact

---

## Definition of Done

- [ ] TwoStageInferenceWorkflow R6 class implemented
- [ ] D+0 model-based completion
- [ ] Ratio propagation fallback
- [ ] Period tracking (verified vs forecasted)
- [ ] TwoStageInferenceResult class
- [ ] Custom ratio calculator support
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- D+0 completion is critical for operational forecasting
- Ratio propagation uses historical daily patterns
- Verified hours increase as day progresses
- Consider timezone handling for multi-region systems
- Future: add confidence intervals that shrink with more verified data
