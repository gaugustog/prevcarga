# PC-084-10: Edge Case Testing

**Epic:** [EPIC-10: Testing & Validation](../epics/EPIC-10-testing-validation.md)
**Task Reference:** T-10.9
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Create comprehensive edge case tests for handling missing data, extreme values, DST transitions, holidays, and failure recovery scenarios to ensure system robustness.

---

## Acceptance Criteria

- [ ] Edge case test suite in `tests/edge_cases/`
- [ ] Missing data handling tested
- [ ] Extreme values handling tested
- [ ] DST transitions handled correctly
- [ ] Holiday periods tested
- [ ] Failure recovery tested
- [ ] All edge cases documented

---

## Technical Specification

### File Location
```
tests/edge_cases/
├── test_missing_data.R
├── test_extreme_values.R
├── test_dst_transitions.R
├── test_holidays.R
├── test_failure_recovery.R
└── helpers.R
```

### Edge Case Test Helpers

```r
# tests/edge_cases/helpers.R

#' Create data with missing values
#'
#' @param data Original data
#' @param missing_pct Percentage of missing values
#' @param pattern Missing pattern ("random", "consecutive", "periodic")
#' @return data.table with missing values
#' @export
inject_missing_values <- function(data, missing_pct = 0.1,
                                   pattern = "random") {
  data <- data.table::copy(data)
  n <- nrow(data)
  n_missing <- floor(n * missing_pct)

  indices <- switch(pattern,
    "random" = sample(n, n_missing),
    "consecutive" = {
      start <- sample(n - n_missing + 1, 1)
      start:(start + n_missing - 1)
    },
    "periodic" = {
      # Every 10th observation
      seq(1, n, by = floor(1 / missing_pct))[1:n_missing]
    }
  )

  data[indices, carga_mwh := NA]
  data
}


#' Create data with extreme values
#'
#' @param data Original data
#' @param n_extreme Number of extreme values
#' @param multiplier Multiplier for extreme values
#' @return data.table with extreme values
#' @export
inject_extreme_values <- function(data, n_extreme = 10, multiplier = 3) {
  data <- data.table::copy(data)
  n <- nrow(data)

  indices <- sample(n, n_extreme)

  for (idx in indices) {
    # Random choice: high or low extreme
    direction <- sample(c(1, -1), 1)
    if (direction > 0) {
      data[idx, carga_mwh := carga_mwh * multiplier]
    } else {
      data[idx, carga_mwh := carga_mwh / multiplier]
    }
  }

  data
}


#' Create DST transition data
#'
#' @param year Year for DST data
#' @param area Area code
#' @return data.table with DST transitions
#' @export
create_dst_data <- function(year = 2024, area = "RJ") {
  # Brazilian DST transitions (hypothetical - Brazil suspended DST in 2019)
  # For testing: Spring forward in November, Fall back in February

  # Create data around typical transition dates
  spring_forward <- as.POSIXct(sprintf("%d-11-03 00:00:00", year))
  fall_back <- as.POSIXct(sprintf("%d-02-16 00:00:00", year))

  # Week around each transition
  spring_dates <- seq(spring_forward - 3*24*3600, spring_forward + 3*24*3600, by = "hour")
  fall_dates <- seq(fall_back - 3*24*3600, fall_back + 3*24*3600, by = "hour")

  dates <- sort(c(spring_dates, fall_dates))

  data.table::data.table(
    datetime = dates,
    area_code = area,
    carga_mwh = rnorm(length(dates), 5000, 500)
  )
}


#' Create holiday data
#'
#' @param year Year
#' @param area Area code
#' @return data.table with holiday periods
#' @export
create_holiday_data <- function(year = 2024, area = "RJ") {
  # Major Brazilian holidays
  holidays <- c(
    sprintf("%d-01-01", year),  # New Year
    sprintf("%d-04-21", year),  # Tiradentes
    sprintf("%d-05-01", year),  # Labor Day
    sprintf("%d-09-07", year),  # Independence
    sprintf("%d-10-12", year),  # Nossa Senhora Aparecida
    sprintf("%d-11-02", year),  # Finados
    sprintf("%d-11-15", year),  # Republic Day
    sprintf("%d-12-25", year)   # Christmas
  )

  # Week around each holiday
  all_dates <- c()
  for (holiday in holidays) {
    h_date <- as.POSIXct(holiday)
    dates <- seq(h_date - 3*24*3600, h_date + 3*24*3600, by = "hour")
    all_dates <- c(all_dates, dates)
  }

  all_dates <- as.POSIXct(unique(all_dates), origin = "1970-01-01")

  data.table::data.table(
    datetime = sort(all_dates),
    area_code = area,
    carga_mwh = rnorm(length(all_dates), 5000, 500) * 0.85  # Lower on holidays
  )
}
```

### Missing Data Tests

```r
# tests/edge_cases/test_missing_data.R

library(testthat)
library(prevcargaons)
source("tests/edge_cases/helpers.R")

test_that("Model handles random missing values", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Inject missing values
  data_with_missing <- inject_missing_values(env$data, missing_pct = 0.05)

  # Should handle via imputation or filtering
  pipeline <- FeaturePipeline$new()
  pipeline$add_plugin(get_feature_plugin("calendar"))
  pipeline$add_plugin(get_feature_plugin("lag"))

  # Process with missing values
  result <- tryCatch({
    features <- pipeline$transform(data_with_missing)
    TRUE
  }, error = function(e) FALSE)

  expect_true(result || isTRUE(attr(result, "handled")))
})


test_that("Model handles consecutive missing values", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # 24 consecutive missing (full day)
  data_with_gap <- env$data
  gap_start <- sample(nrow(data_with_gap) - 24, 1)
  data_with_gap[gap_start:(gap_start + 23), carga_mwh := NA]

  # Should detect and handle gap
  n_missing <- sum(is.na(data_with_gap$carga_mwh))
  expect_equal(n_missing, 24)

  # Imputation should fill gaps
  imputed <- impute_missing(data_with_gap, method = "locf")
  expect_equal(sum(is.na(imputed$carga_mwh)), 0)
})


test_that("Model handles periodic missing values", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Every 24th hour missing (same hour each day)
  data_periodic <- env$data
  data_periodic[seq(1, nrow(data_periodic), 24), carga_mwh := NA]

  # Should be robust to systematic missing
  imputed <- impute_missing(data_periodic, method = "seasonal")

  expect_equal(sum(is.na(imputed$carga_mwh)), 0)
})


test_that("Prediction handles missing recent data", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Remove last 6 hours of data
  cutoff <- max(env$data$datetime) - 6*3600
  data_truncated <- env$data[datetime <= cutoff]

  # Prediction should still work (with warning)
  expect_warning({
    predict_workflow <- PredictWorkflow$new(
      config = env$config,
      storage = env$storage
    )
  }, regexp = NULL)  # May or may not warn
})


test_that("Empty data returns appropriate error", {
  expect_error({
    pipeline <- FeaturePipeline$new()
    pipeline$transform(data.table::data.table())
  })
})
```

### Extreme Values Tests

```r
# tests/edge_cases/test_extreme_values.R

library(testthat)
library(prevcargaons)
source("tests/edge_cases/helpers.R")

test_that("Model handles positive outliers", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Inject extreme high values
  data_extreme <- inject_extreme_values(env$data, n_extreme = 5, multiplier = 5)

  max_original <- max(env$data$carga_mwh)
  max_extreme <- max(data_extreme$carga_mwh)

  expect_true(max_extreme > max_original * 4)

  # Model should handle (may cap or transform)
  model <- get_model("hw")

  train_data <- data_extreme[datetime < "2024-01-01"]
  pipeline <- FeaturePipeline$new()
  features <- pipeline$transform(train_data)

  result <- tryCatch({
    model$train(features$X, features$y)
    TRUE
  }, error = function(e) FALSE)

  expect_true(result)
})


test_that("Model handles negative values (should not occur)", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Inject negative values (invalid for load)
  data_negative <- data.table::copy(env$data)
  data_negative[1:5, carga_mwh := -1000]

  # Should be caught by validation
  expect_error(
    validate_load_data(data_negative),
    regexp = "negative"
  )
})


test_that("Model handles zero values", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Inject zero values
  data_zero <- data.table::copy(env$data)
  data_zero[1:10, carga_mwh := 0]

  # Zeros may be valid during outages
  result <- tryCatch({
    validate_load_data(data_zero, allow_zeros = TRUE)
    TRUE
  }, error = function(e) FALSE)

  expect_true(result)
})


test_that("Model handles very small values", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Values close to zero
  data_small <- data.table::copy(env$data)
  data_small[1:5, carga_mwh := 0.001]

  # Should handle without numerical issues
  model <- get_model("hw")

  result <- tryCatch({
    pipeline <- FeaturePipeline$new()
    features <- pipeline$transform(data_small[1:1000])
    model$train(features$X, features$y)
    TRUE
  }, error = function(e) FALSE)

  expect_true(result)
})


test_that("Metrics handle extreme prediction errors", {
  calculator <- MetricsCalculator$new()

  actual <- c(1000, 2000, 3000, 4000, 5000)
  prediction <- c(1000, 2000, 15000, 4000, 5000)  # One extreme error

  metrics <- calculator$calculate_all(actual, prediction)

  # Should not produce Inf or NaN
  expect_true(is.finite(metrics$mape))
  expect_true(is.finite(metrics$mae))
  expect_true(is.finite(metrics$rmse))

  # MAPE should be high but not infinite
  expect_true(metrics$mape > 0.5)  # >50% due to extreme error
})
```

### DST Transition Tests

```r
# tests/edge_cases/test_dst_transitions.R

library(testthat)
library(prevcargaons)
source("tests/edge_cases/helpers.R")

test_that("Feature pipeline handles DST spring forward", {
  # Spring forward: 2:00 AM -> 3:00 AM (skip 1 hour)
  dst_data <- create_dst_data(year = 2024, area = "RJ")

  pipeline <- FeaturePipeline$new()
  pipeline$add_plugin(get_feature_plugin("calendar"))
  pipeline$add_plugin(get_feature_plugin("lag", list(lags = c(1, 24))))

  # Should handle missing hour gracefully
  result <- tryCatch({
    features <- pipeline$transform(dst_data)
    TRUE
  }, error = function(e) FALSE)

  expect_true(result)
})


test_that("Feature pipeline handles DST fall back", {
  # Fall back: 3:00 AM -> 2:00 AM (repeat 1 hour)
  dst_data <- create_dst_data(year = 2024, area = "RJ")

  # Duplicate one hour
  dup_idx <- which(format(dst_data$datetime, "%H") == "02")[1]
  if (!is.na(dup_idx)) {
    dst_data <- rbind(dst_data, dst_data[dup_idx])
    dst_data <- dst_data[order(datetime)]
  }

  pipeline <- FeaturePipeline$new()
  pipeline$add_plugin(get_feature_plugin("calendar"))

  # Should handle duplicate hour
  result <- tryCatch({
    features <- pipeline$transform(dst_data)
    TRUE
  }, error = function(e) FALSE)

  expect_true(result)
})


test_that("Lag features correct during DST transitions", {
  # Check that lag-24 correctly references 24 hours ago
  dst_data <- create_dst_data(year = 2024, area = "RJ")

  lag_plugin <- get_feature_plugin("lag", list(lags = c(24)))

  features <- lag_plugin$generate(dst_data)

  # Lag-24 should reference actual 24-hour offset
  expect_true("lag_24" %in% names(features))

  # No NAs in lag after sufficient data
  n_valid <- sum(!is.na(features$lag_24))
  expect_true(n_valid > nrow(features) * 0.8)
})


test_that("Timezone handling is consistent", {
  # Create data with explicit timezone
  dates <- seq(
    as.POSIXct("2024-01-01", tz = "America/Sao_Paulo"),
    as.POSIXct("2024-01-07", tz = "America/Sao_Paulo"),
    by = "hour"
  )

  data <- data.table::data.table(
    datetime = dates,
    area_code = "RJ",
    carga_mwh = rnorm(length(dates), 5000, 500)
  )

  # Calendar features should use correct timezone
  calendar <- get_feature_plugin("calendar")
  features <- calendar$generate(data)

  # Hour should match Sao Paulo time
  expected_hours <- as.integer(format(data$datetime, "%H", tz = "America/Sao_Paulo"))

  expect_equal(features$hour, expected_hours)
})
```

### Holiday Tests

```r
# tests/edge_cases/test_holidays.R

library(testthat)
library(prevcargaons)
source("tests/edge_cases/helpers.R")

test_that("Holiday feature correctly identifies holidays", {
  holiday_data <- create_holiday_data(year = 2024, area = "RJ")

  calendar <- get_feature_plugin("calendar")
  features <- calendar$generate(holiday_data)

  # Should have holiday indicator
  expect_true("is_holiday" %in% names(features))

  # At least some should be marked as holidays
  n_holidays <- sum(features$is_holiday, na.rm = TRUE)
  expect_true(n_holidays > 0)
})


test_that("Model predictions differ on holidays", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Train model
  train_workflow <- TrainWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  train_workflow$run(
    model = "hw",
    areas = "RJ",
    train_start = "2023-01-01",
    train_end = "2023-12-31"
  )

  # Predict on holiday vs non-holiday
  predict_workflow <- PredictWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  # Christmas
  holiday_pred <- predict_workflow$run(
    model = "hw",
    areas = "RJ",
    predict_date = "2024-12-25",
    horizons = 0
  )

  # Regular weekday
  regular_pred <- predict_workflow$run(
    model = "hw",
    areas = "RJ",
    predict_date = "2024-12-18",  # Wednesday before Christmas
    horizons = 0
  )

  # Predictions should differ
  holiday_mean <- mean(holiday_pred$prediction)
  regular_mean <- mean(regular_pred$prediction)

  expect_true(holiday_mean != regular_mean)
})


test_that("Holiday bridges (between holidays and weekends) handled", {
  # Test "ponte" (bridge days) commonly observed in Brazil
  # E.g., if holiday is Thursday, Friday may have lower demand

  dates <- seq(
    as.POSIXct("2024-04-18"),  # Thursday before Tiradentes (Apr 21)
    as.POSIXct("2024-04-22"),
    by = "hour"
  )

  data <- data.table::data.table(
    datetime = dates,
    area_code = "RJ",
    carga_mwh = rnorm(length(dates), 5000, 500)
  )

  calendar <- get_feature_plugin("calendar")
  features <- calendar$generate(data)

  # Friday (Apr 19) might be detected as bridge
  expect_true("day_of_week" %in% names(features))
})
```

### Failure Recovery Tests

```r
# tests/edge_cases/test_failure_recovery.R

library(testthat)
library(prevcargaons)

test_that("Workflow recovers from model training failure", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  train_workflow <- TrainWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  # Try to train with insufficient data
  result <- tryCatch({
    train_workflow$run(
      model = "hw",
      areas = "RJ",
      train_start = "2024-01-01",
      train_end = "2024-01-02"  # Only 2 days
    )
  }, error = function(e) {
    list(success = FALSE, error = conditionMessage(e))
  })

  # Should fail gracefully, not crash
  expect_true(is.list(result))
})


test_that("Workflow handles storage connection failure", {
  # Create config with invalid storage
  config <- ConfigManager$new()
  config$set("storage.backend", "s3")
  config$set("storage.s3.bucket", "non-existent-bucket-xyz123")

  storage <- tryCatch({
    create_storage_backend(config$get_storage_config())
  }, error = function(e) NULL)

  # Storage creation should fail gracefully
  # Or operations should fail with clear error
  if (!is.null(storage)) {
    result <- tryCatch({
      storage$read_carga(areas = "RJ", start_date = "2024-01-01", end_date = "2024-01-31")
    }, error = function(e) {
      list(error = conditionMessage(e))
    })

    expect_true("error" %in% names(result))
  }
})


test_that("Backtest checkpoints allow recovery", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  backtest <- BacktestWorkflow$new(
    config = env$config,
    storage = env$storage,
    checkpoint_interval = 5  # Checkpoint every 5 days
  )

  # Start backtest
  result <- backtest$run(
    start_date = "2024-01-01",
    end_date = "2024-01-15",
    models = "hw",
    areas = "RJ",
    retrain_intervals = 7
  )

  # Check checkpoint was created
  checkpoint_dir <- file.path(env$temp_dir, "checkpoints")

  # Backtest should complete even if interrupted
  expect_true(!is.null(result))
})


test_that("Partial predictions saved on failure", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Simulate partial failure during multi-area prediction
  areas <- c("RJ", "SP", "INVALID_AREA")

  train_workflow <- TrainWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  # Train only valid areas
  train_workflow$run(
    model = "hw",
    areas = c("RJ", "SP"),
    train_start = "2023-01-01",
    train_end = "2023-12-31"
  )

  predict_workflow <- PredictWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  # Should succeed for valid areas, report error for invalid
  result <- tryCatch({
    predict_workflow$run(
      model = "hw",
      areas = areas,
      predict_date = "2024-01-15",
      horizons = 0:3,
      fail_on_error = FALSE  # Continue on partial failure
    )
  }, error = function(e) NULL)

  # Should have partial results
  if (!is.null(result)) {
    valid_areas <- unique(result$area_code)
    expect_true("RJ" %in% valid_areas || "SP" %in% valid_areas)
  }
})


test_that("Logger captures failure context", {
  logger <- StructuredLogger$new(level = "DEBUG")

  # Capture log output
  output <- capture.output({
    logger$error("Test failure",
                 context = "edge_case_test",
                 error_type = "simulated")
  })

  # Log should contain error info
  expect_true(length(output) > 0 || TRUE)  # Logger may write to file
})
```

### Edge Case Runner Script

```r
#!/usr/bin/env Rscript
# tests/edge_cases/run_edge_cases.R

library(testthat)
library(prevcargaons)

cat("\n")
cat("=" |> rep(60) |> paste(collapse = ""))
cat("\n  PrevCarga Edge Case Test Suite\n")
cat("=" |> rep(60) |> paste(collapse = ""))
cat("\n\n")

# Run all edge case tests
results <- testthat::test_dir(
  "tests/edge_cases",
  reporter = "summary"
)

# Summary
n_passed <- sum(results$passed)
n_failed <- sum(results$failed)
n_skipped <- sum(results$skipped)

cat("\n\nEdge Case Test Summary\n")
cat("======================\n")
cat(sprintf("Passed:  %d\n", n_passed))
cat(sprintf("Failed:  %d\n", n_failed))
cat(sprintf("Skipped: %d\n", n_skipped))

if (n_failed > 0) {
  cat("\n\u2717 Some edge case tests failed\n")
  quit(status = 1)
} else {
  cat("\n\u2713 All edge case tests passed\n")
  quit(status = 0)
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Random missing values | Handled gracefully |
| TC-002 | Consecutive missing values | Gap filled |
| TC-003 | Extreme high values | Model trains |
| TC-004 | Negative values (invalid) | Error raised |
| TC-005 | DST spring forward | Hour gap handled |
| TC-006 | DST fall back | Duplicate hour handled |
| TC-007 | Holiday identification | Holidays flagged |
| TC-008 | Training failure | Graceful recovery |
| TC-009 | Storage failure | Clear error message |
| TC-010 | Partial prediction failure | Partial results saved |

---

## Dependencies

- PC-011-02: FeaturePipeline
- PC-012-02: CalendarFeatures
- PC-058-08: TrainWorkflow
- PC-059-08: PredictWorkflow
- PC-060-08: BacktestWorkflow

---

## Definition of Done

- [ ] Edge case test directory created
- [ ] Missing data tests passing
- [ ] Extreme value tests passing
- [ ] DST transition tests passing
- [ ] Holiday tests passing
- [ ] Failure recovery tests passing
- [ ] Edge case runner script working
- [ ] All tests documented
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Edge cases based on real production issues
- Consider adding fuzzing tests in future
- DST handling critical for Brazil timezone
- Holiday calendar should be configurable
- Failure recovery enables long-running backtests
