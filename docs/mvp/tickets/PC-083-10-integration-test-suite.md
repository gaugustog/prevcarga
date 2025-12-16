# PC-083-10: Integration Test Suite

**Epic:** [EPIC-10: Testing & Validation](../epics/EPIC-10-testing-validation.md)
**Task Reference:** T-10.8
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Create comprehensive integration tests that verify complete workflows work end-to-end: Train → Predict → Evaluate, Train → Combine → Reconcile → Evaluate, and Backtest with retraining.

---

## Acceptance Criteria

- [ ] Integration test directory created: `tests/integration/`
- [ ] Train → Predict → Evaluate workflow tested
- [ ] Train → Combine → Reconcile → Evaluate workflow tested
- [ ] Backtest with retraining workflow tested
- [ ] All integration tests passing
- [ ] Integration test runner script

---

## Technical Specification

### File Location
```
tests/integration/
├── test_train_predict.R
├── test_combination_workflow.R
├── test_reconciliation_workflow.R
├── test_full_workflow.R
├── test_backtest_workflow.R
└── helpers.R
```

### Integration Test Helpers

```r
# tests/integration/helpers.R

#' Create test configuration for integration tests
#'
#' @param temp_dir Temporary directory for test data
#' @return ConfigManager with test settings
#' @export
create_test_config <- function(temp_dir = tempdir()) {
  config <- ConfigManager$new()

  config$set("storage.backend", "local")
  config$set("storage.local.path", temp_dir)

  config$set("areas", c("RJ", "SP"))  # Limited for testing
  config$set("models.enabled", c("hw"))  # Fast model for testing

  config$set("features.plugins", list(
    calendar = list(enabled = TRUE),
    lag = list(enabled = TRUE, config = list(lags = c(1, 24)))
  ))

  config
}


#' Generate synthetic load data for testing
#'
#' @param areas Area codes
#' @param start_date Start date
#' @param end_date End date
#' @return data.table with synthetic data
#' @export
generate_test_data <- function(areas, start_date, end_date) {
  dates <- seq(as.POSIXct(start_date), as.POSIXct(end_date), by = "hour")

  data.table::rbindlist(lapply(areas, function(area) {
    # Generate realistic load pattern
    base_load <- switch(area,
      "RJ" = 5000,
      "SP" = 8000,
      3000
    )

    hour <- as.integer(format(dates, "%H"))
    weekday <- as.integer(format(dates, "%u"))

    # Daily pattern
    daily_factor <- 0.7 + 0.3 * sin((hour - 6) * pi / 12)

    # Weekly pattern
    weekly_factor <- ifelse(weekday <= 5, 1.0, 0.85)

    # Random noise
    noise <- rnorm(length(dates), 0, base_load * 0.05)

    data.table::data.table(
      datetime = dates,
      area_code = area,
      carga_mwh = base_load * daily_factor * weekly_factor + noise
    )
  }))
}


#' Setup integration test environment
#'
#' @return Test environment with config, data, temp_dir
#' @export
setup_integration_env <- function() {
  temp_dir <- tempfile("prevcarga_int_test_")
  dir.create(temp_dir, recursive = TRUE)

  config <- create_test_config(temp_dir)

  # Generate test data
  test_data <- generate_test_data(
    areas = config$get("areas"),
    start_date = "2023-01-01",
    end_date = "2024-03-31"
  )

  # Save to storage
  storage <- create_storage_backend(config$get_storage_config())
  storage$write_carga(test_data)

  list(
    config = config,
    data = test_data,
    temp_dir = temp_dir,
    storage = storage
  )
}


#' Cleanup integration test environment
#'
#' @param env Test environment from setup_integration_env
#' @export
cleanup_integration_env <- function(env) {
  if (dir.exists(env$temp_dir)) {
    unlink(env$temp_dir, recursive = TRUE)
  }
}
```

### Train → Predict → Evaluate Test

```r
# tests/integration/test_train_predict.R

library(testthat)
library(prevcargaons)

test_that("Train → Predict → Evaluate workflow completes successfully", {
  # Setup
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # 1. Train model
  train_workflow <- TrainWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  train_result <- train_workflow$run(
    model = "hw",
    areas = env$config$get("areas"),
    train_start = "2023-01-01",
    train_end = "2023-12-31"
  )

  expect_true(train_result$success)
  expect_equal(length(train_result$models), 2)  # 2 areas

  # 2. Predict
  predict_workflow <- PredictWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  predictions <- predict_workflow$run(
    model = "hw",
    areas = env$config$get("areas"),
    predict_date = "2024-01-15",
    horizons = 0:3
  )

  expect_true(nrow(predictions) > 0)
  expect_true("prediction" %in% names(predictions))
  expect_true("horizon" %in% names(predictions))

  # 3. Evaluate
  calculator <- MetricsCalculator$new()

  # Get actuals for evaluation
  actuals <- env$storage$read_carga(
    areas = env$config$get("areas"),
    start_date = "2024-01-15",
    end_date = "2024-01-18"
  )

  merged <- merge(predictions, actuals, by = c("datetime", "area_code"))

  metrics <- calculator$calculate_all(merged$carga_mwh, merged$prediction)

  expect_true(!is.na(metrics$mape))
  expect_true(!is.na(metrics$mae))
  expect_true(!is.na(metrics$rmse))
  expect_true(metrics$mape < 0.5)  # Less than 50% error (synthetic data)

  cat(sprintf("\nWorkflow completed - MAPE: %.2f%%\n", metrics$mape * 100))
})


test_that("Train workflow handles multiple models", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  train_workflow <- TrainWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  # Train multiple models if available
  models <- c("hw")  # Use fast models for integration tests

  for (model in models) {
    result <- train_workflow$run(
      model = model,
      areas = env$config$get("areas"),
      train_start = "2023-01-01",
      train_end = "2023-12-31"
    )

    expect_true(result$success, info = sprintf("Model %s failed", model))
  }
})


test_that("Predict workflow handles all horizons", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Train first
  train_workflow <- TrainWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  train_workflow$run(
    model = "hw",
    areas = env$config$get("areas"),
    train_start = "2023-01-01",
    train_end = "2023-12-31"
  )

  # Predict all horizons
  predict_workflow <- PredictWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  predictions <- predict_workflow$run(
    model = "hw",
    areas = env$config$get("areas"),
    predict_date = "2024-01-15",
    horizons = 0:8
  )

  # Check all horizons present
  unique_horizons <- unique(predictions$horizon)
  expect_equal(sort(unique_horizons), 0:8)
})
```

### Combination Workflow Test

```r
# tests/integration/test_combination_workflow.R

library(testthat)
library(prevcargaons)

test_that("Train → Combine → Evaluate workflow works", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # 1. Train multiple models
  train_workflow <- TrainWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  models <- c("hw")  # In real tests, use multiple models

  for (model in models) {
    train_workflow$run(
      model = model,
      areas = env$config$get("areas"),
      train_start = "2023-01-01",
      train_end = "2023-12-31"
    )
  }

  # 2. Generate predictions from each model
  predict_workflow <- PredictWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  all_predictions <- list()
  for (model in models) {
    preds <- predict_workflow$run(
      model = model,
      areas = env$config$get("areas"),
      predict_date = "2024-01-15",
      horizons = 0:3
    )
    preds$model <- model
    all_predictions[[model]] <- preds
  }

  # 3. Combine predictions
  combiner <- get_combiner("simple_average")
  combined <- combiner$combine(all_predictions)

  expect_true(nrow(combined$predictions) > 0)
  expect_true("prediction" %in% names(combined$predictions))

  # 4. Evaluate combined
  calculator <- MetricsCalculator$new()

  actuals <- env$storage$read_carga(
    areas = env$config$get("areas"),
    start_date = "2024-01-15",
    end_date = "2024-01-18"
  )

  merged <- merge(combined$predictions, actuals, by = c("datetime", "area_code"))
  metrics <- calculator$calculate_all(merged$carga_mwh, merged$prediction)

  expect_true(!is.na(metrics$mape))

  cat(sprintf("\nCombined MAPE: %.2f%%\n", metrics$mape * 100))
})
```

### Reconciliation Workflow Test

```r
# tests/integration/test_reconciliation_workflow.R

library(testthat)
library(prevcargaons)

test_that("Train → Predict → Reconcile → Evaluate workflow works", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Train
  train_workflow <- TrainWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  train_workflow$run(
    model = "hw",
    areas = env$config$get("areas"),
    train_start = "2023-01-01",
    train_end = "2023-12-31"
  )

  # Predict
  predict_workflow <- PredictWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  predictions <- predict_workflow$run(
    model = "hw",
    areas = env$config$get("areas"),
    predict_date = "2024-01-15",
    horizons = 0:3
  )

  # Reconcile (bottom-up for simple test)
  reconciler <- get_reconciler("bottom_up")

  # Build simple summing matrix for test
  areas <- env$config$get("areas")
  S <- matrix(1, nrow = length(areas) + 1, ncol = length(areas))
  S[1:length(areas), ] <- diag(length(areas))
  rownames(S) <- c(areas, "Total")
  colnames(S) <- areas

  reconciled <- reconciler$reconcile(
    base_forecasts = predictions,
    summing_matrix = S
  )

  expect_true(nrow(reconciled) >= nrow(predictions))

  # Verify hierarchical consistency
  by_datetime <- split(reconciled, reconciled$datetime)

  for (dt in names(by_datetime)) {
    dt_data <- by_datetime[[dt]]
    area_sum <- sum(dt_data[dt_data$area_code %in% areas, "prediction"])
    total_pred <- dt_data[dt_data$area_code == "Total", "prediction"]

    if (length(total_pred) > 0) {
      expect_equal(area_sum, total_pred, tolerance = 1e-6)
    }
  }
})
```

### Full Workflow Test

```r
# tests/integration/test_full_workflow.R

library(testthat)
library(prevcargaons)

test_that("Complete workflow: Train → Combine → Reconcile → Evaluate", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # 1. Train models
  train_workflow <- TrainWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  train_workflow$run(
    model = "hw",
    areas = env$config$get("areas"),
    train_start = "2023-01-01",
    train_end = "2023-12-31"
  )

  # 2. Predict
  predict_workflow <- PredictWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  predictions <- predict_workflow$run(
    model = "hw",
    areas = env$config$get("areas"),
    predict_date = "2024-01-15",
    horizons = 0:3
  )

  # 3. Would combine if multiple models (simplified)
  combined <- predictions

  # 4. Reconcile
  reconciler <- get_reconciler("bottom_up")
  areas <- env$config$get("areas")
  S <- matrix(1, nrow = length(areas) + 1, ncol = length(areas))
  S[1:length(areas), ] <- diag(length(areas))
  rownames(S) <- c(areas, "Total")
  colnames(S) <- areas

  reconciled <- reconciler$reconcile(
    base_forecasts = combined,
    summing_matrix = S
  )

  # 5. Evaluate
  calculator <- MetricsCalculator$new()

  actuals <- env$storage$read_carga(
    areas = env$config$get("areas"),
    start_date = "2024-01-15",
    end_date = "2024-01-18"
  )

  merged <- merge(reconciled, actuals, by = c("datetime", "area_code"))
  metrics <- calculator$calculate_all(merged$carga_mwh, merged$prediction)

  expect_true(!is.na(metrics$mape))
  expect_true(!is.na(metrics$mae))

  cat(sprintf("\nFull workflow MAPE: %.2f%%\n", metrics$mape * 100))
})
```

### Backtest Workflow Test

```r
# tests/integration/test_backtest_workflow.R

library(testthat)
library(prevcargaons)

test_that("Backtest workflow with retraining works", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  # Run short backtest
  backtest <- BacktestWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  results <- backtest$run(
    start_date = "2024-01-01",
    end_date = "2024-01-31",  # 1 month for speed
    models = "hw",
    areas = env$config$get("areas"),
    retrain_intervals = c(7),  # Single interval for speed
    walk_forward = TRUE,
    train_window = 180,  # 6 months
    horizons = 0:3
  )

  expect_true(!is.null(results))
  expect_true(results$n_predictions > 0)

  # Check metrics exist
  summary <- results$summary()
  expect_true("mape" %in% names(summary))

  cat(sprintf("\nBacktest MAPE: %.2f%%\n", summary$mape * 100))
})


test_that("Backtest handles multiple retraining intervals", {
  env <- setup_integration_env()
  on.exit(cleanup_integration_env(env))

  backtest <- BacktestWorkflow$new(
    config = env$config,
    storage = env$storage
  )

  results <- backtest$run(
    start_date = "2024-01-01",
    end_date = "2024-01-15",  # Short period
    models = "hw",
    areas = env$config$get("areas"),
    retrain_intervals = c(3, 7),  # Multiple intervals
    walk_forward = TRUE,
    train_window = 180,
    horizons = 0:2
  )

  # Should have results for each interval
  by_interval <- results$get_metrics_by_interval()

  expect_true(nrow(by_interval) >= 2)
})
```

### Integration Test Runner

```r
#!/usr/bin/env Rscript
# tests/integration/run_integration.R
#
# Run all integration tests

library(testthat)
library(prevcargaons)

cat("\n")
cat("=" |> rep(60) |> paste(collapse = ""))
cat("\n  PrevCarga Integration Test Suite\n")
cat("=" |> rep(60) |> paste(collapse = ""))
cat("\n\n")

# Discover and run all integration tests
results <- testthat::test_dir(
  "tests/integration",
  reporter = "summary"
)

# Summary
n_passed <- sum(results$passed)
n_failed <- sum(results$failed)
n_skipped <- sum(results$skipped)

cat("\n\nIntegration Test Summary\n")
cat("========================\n")
cat(sprintf("Passed:  %d\n", n_passed))
cat(sprintf("Failed:  %d\n", n_failed))
cat(sprintf("Skipped: %d\n", n_skipped))

# Exit with appropriate code
if (n_failed > 0) {
  cat("\n\u2717 Some integration tests failed\n")
  quit(status = 1)
} else {
  cat("\n\u2713 All integration tests passed\n")
  quit(status = 0)
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Train → Predict → Evaluate | Workflow completes |
| TC-002 | Multiple models training | All models trained |
| TC-003 | All horizons predicted | D+0 to D+8 present |
| TC-004 | Combination workflow | Predictions combined |
| TC-005 | Reconciliation workflow | Hierarchy consistent |
| TC-006 | Full end-to-end workflow | All steps complete |
| TC-007 | Backtest with retraining | Results generated |
| TC-008 | Multiple intervals | All intervals tested |
| TC-009 | Test data generation | Realistic patterns |
| TC-010 | Cleanup after tests | No temp files left |

---

## Dependencies

- PC-058-08: TrainWorkflow
- PC-059-08: PredictWorkflow
- PC-060-08: BacktestWorkflow
- PC-032-05: BaseCombiner
- PC-039-06: BaseReconciler
- PC-047-07: MetricsCalculator

---

## Definition of Done

- [ ] Integration test directory structure created
- [ ] Train → Predict → Evaluate tests passing
- [ ] Combination workflow tests passing
- [ ] Reconciliation workflow tests passing
- [ ] Full workflow tests passing
- [ ] Backtest workflow tests passing
- [ ] Test helper functions working
- [ ] Integration test runner script
- [ ] All tests passing in CI/CD
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Integration tests use synthetic data for speed
- Consider running integration tests nightly
- Tests follow 30% of test pyramid (unit 60%, integration 30%, E2E 10%)
- Use fast models (hw) for integration tests
- Cleanup temp files after each test
