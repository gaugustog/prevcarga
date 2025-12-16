# PC-056-07: Evaluation Infrastructure Tests

**Epic:** [EPIC-07: Evaluation Layer](../epics/EPIC-07-evaluation-layer.md)
**Task Reference:** T-07.8
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Create comprehensive test suite for the evaluation infrastructure, covering metrics calculation, period/horizon breakdown, drift detection, model comparison, and report generation.

---

## Acceptance Criteria

- [ ] Test files created for each evaluation component
- [ ] Metrics calculation tests with known values
- [ ] Patamar classification tests
- [ ] Horizon metrics tests
- [ ] DriftDetector tests
- [ ] ModelComparator tests
- [ ] Reporter tests
- [ ] ≥80% code coverage

---

## Technical Specification

### Test Files

```
tests/testthat/
├── test-evaluation-metrics.R      # Core metrics tests
├── test-evaluation-period.R       # Patamar classification tests
├── test-evaluation-horizon.R      # Horizon metrics tests
├── test-evaluation-drift.R        # DriftDetector tests
├── test-evaluation-comparator.R   # ModelComparator tests
├── test-evaluation-reporter.R     # Reporter tests
└── helper-evaluation.R            # Test fixtures
```

### Test Fixtures

```r
# tests/testthat/helper-evaluation.R

#' Create test forecast data
#' @param n Number of observations
#' @param error_sd Standard deviation of errors
#' @return data.table with actual and predicted
create_test_forecast_data <- function(n = 100, error_sd = 50) {
  set.seed(42)

  actual <- 1000 + rnorm(n, sd = 100)
  predicted <- actual + rnorm(n, sd = error_sd)

  data.table::data.table(
    DataHora = seq(Sys.time(), by = "hour", length.out = n),
    actual = actual,
    predicted = predicted
  )
}


#' Create test data with known metrics
#' @return List with data and expected metrics
create_known_metrics_data <- function() {
  # Simple case: actual = c(100, 200, 300), predicted = c(110, 200, 330)
  # MAPE = mean(abs(c(-10, 0, -30) / c(100, 200, 300))) * 100
  #      = mean(c(0.10, 0, 0.10)) * 100 = 6.67%
  # MAE = mean(abs(c(-10, 0, -30))) = 40/3 = 13.33
  # RMSE = sqrt(mean(c(100, 0, 900))) = sqrt(1000/3) = 18.26

  list(
    data = data.table::data.table(
      actual = c(100, 200, 300),
      predicted = c(110, 200, 330)
    ),
    expected = list(
      mape = 6.666667,
      mae = 13.333333,
      rmse = 18.257419,
      mbe = 13.333333  # Average: (10 + 0 + 30) / 3
    )
  )
}


#' Create test data for patamar classification
#' @return data.table with times in different patamares
create_patamar_test_data <- function() {
  dates <- seq(
    as.POSIXct("2024-01-15 00:00:00"),
    as.POSIXct("2024-01-15 23:30:00"),
    by = "30 min"
  )

  data.table::data.table(
    DataHora = dates,
    actual = rnorm(48, 1000, 50),
    predicted = rnorm(48, 1000, 60)
  )
}


#' Create test data for horizon analysis
#' @param horizons Vector of horizons
#' @return data.table with horizon column
create_horizon_test_data <- function(horizons = 0:8) {
  n_per_horizon <- 24

  dts <- lapply(horizons, function(h) {
    data.table::data.table(
      reference_datetime = as.POSIXct("2024-01-15 00:00:00"),
      DataHora = seq(
        as.POSIXct("2024-01-15 00:00:00") + h * 86400,
        by = "hour",
        length.out = n_per_horizon
      ),
      horizon = h,
      actual = rnorm(n_per_horizon, 1000 + h * 10, 50),
      predicted = rnorm(n_per_horizon, 1000 + h * 10, 50 + h * 5)
    )
  })

  data.table::rbindlist(dts)
}


#' Create test model results for comparison
#' @param n Number of observations
#' @return List with model predictions
create_model_results <- function(n = 100) {
  set.seed(42)
  actual <- rnorm(n, 1000, 100)

  list(
    actual = actual,
    model_results = list(
      model_a = list(
        model_name = "Model A",
        predictions = actual + rnorm(n, 0, 30)
      ),
      model_b = list(
        model_name = "Model B",
        predictions = actual + rnorm(n, 0, 50)
      ),
      model_c = list(
        model_name = "Model C",
        predictions = actual + rnorm(n, 0, 70)
      )
    )
  )
}
```

### Metrics Tests

```r
# tests/testthat/test-evaluation-metrics.R

describe("calculate_mape", {
  it("calculates MAPE correctly for known values", {
    test_data <- create_known_metrics_data()

    result <- calculate_mape(
      test_data$data$actual,
      test_data$data$predicted
    )

    expect_equal(result, test_data$expected$mape, tolerance = 0.0001)
  })

  it("handles zero values in actual", {
    actual <- c(100, 0, 200)
    predicted <- c(110, 10, 210)

    # Should exclude zero values
    result <- calculate_mape(actual, predicted)

    expect_true(!is.na(result))
  })

  it("returns NA for all-zero actual", {
    actual <- c(0, 0, 0)
    predicted <- c(10, 20, 30)

    result <- calculate_mape(actual, predicted)

    expect_true(is.na(result))
  })

  it("handles NA values", {
    actual <- c(100, NA, 200)
    predicted <- c(110, 190, 210)

    result <- calculate_mape(actual, predicted, na.rm = TRUE)

    expect_true(!is.na(result))
  })
})


describe("calculate_mae", {
  it("calculates MAE correctly", {
    test_data <- create_known_metrics_data()

    result <- calculate_mae(
      test_data$data$actual,
      test_data$data$predicted
    )

    expect_equal(result, test_data$expected$mae, tolerance = 0.0001)
  })

  it("returns 0 for perfect predictions", {
    actual <- c(100, 200, 300)
    predicted <- c(100, 200, 300)

    result <- calculate_mae(actual, predicted)

    expect_equal(result, 0)
  })
})


describe("calculate_rmse", {
  it("calculates RMSE correctly", {
    test_data <- create_known_metrics_data()

    result <- calculate_rmse(
      test_data$data$actual,
      test_data$data$predicted
    )

    expect_equal(result, test_data$expected$rmse, tolerance = 0.0001)
  })

  it("is always >= MAE", {
    data <- create_test_forecast_data()

    mae <- calculate_mae(data$actual, data$predicted)
    rmse <- calculate_rmse(data$actual, data$predicted)

    expect_true(rmse >= mae)
  })
})


describe("calculate_all_metrics", {
  it("returns all expected metrics", {
    data <- create_test_forecast_data()

    result <- calculate_all_metrics(data$actual, data$predicted)

    expect_true("mape" %in% names(result))
    expect_true("mae" %in% names(result))
    expect_true("rmse" %in% names(result))
    expect_true("mbe" %in% names(result))
    expect_true("percentiles" %in% names(result))
  })

  it("includes correct number of percentiles", {
    data <- create_test_forecast_data()

    result <- calculate_all_metrics(data$actual, data$predicted)

    expect_length(result$percentiles, 5)
  })
})


describe("MetricsCalculator", {
  it("initializes correctly", {
    calc <- MetricsCalculator$new()
    expect_s3_class(calc, "MetricsCalculator")
  })

  it("calculates grouped metrics", {
    calc <- MetricsCalculator$new()

    dt <- data.table::data.table(
      area_code = rep(c("RJ", "SP"), each = 50),
      actual = rnorm(100, 1000, 50),
      predicted = rnorm(100, 1000, 60)
    )

    result <- calc$calculate(dt, group_by = "area_code")

    expect_equal(nrow(result), 2)
    expect_true("RJ" %in% result$area_code)
    expect_true("SP" %in% result$area_code)
  })
})
```

### Patamar Tests

```r
# tests/testthat/test-evaluation-period.R

describe("classify_patamar", {
  it("classifies peak hours correctly", {
    # 18:00 should be peak for SECO
    peak_time <- as.POSIXct("2024-01-15 18:00:00")

    result <- classify_patamar(peak_time, "RJ")

    expect_equal(result, "ponta")
  })

  it("classifies off-peak hours correctly", {
    # 10:00 should be off-peak
    offpeak_time <- as.POSIXct("2024-01-15 10:00:00")

    result <- classify_patamar(offpeak_time, "RJ")

    expect_equal(result, "fora_ponta")
  })

  it("handles different subsystems", {
    time <- as.POSIXct("2024-01-15 19:00:00")

    # Same time might be different patamar in different regions
    result_seco <- classify_patamar(time, "RJ")  # SECO
    result_s <- classify_patamar(time, "PR")     # S

    # Both should return valid patamar
    expect_true(result_seco %in% c("ponta", "fora_ponta", "intermediario"))
    expect_true(result_s %in% c("ponta", "fora_ponta", "intermediario"))
  })
})


describe("calculate_metrics_by_patamar", {
  it("returns metrics for each patamar", {
    dt <- create_patamar_test_data()

    result <- calculate_metrics_by_patamar(dt, area_code = "RJ")

    expect_true("patamar" %in% names(result))
    expect_true("mape" %in% names(result))
    expect_true(nrow(result) >= 2)  # At least ponta and fora_ponta
  })
})
```

### Drift Tests

```r
# tests/testthat/test-evaluation-drift.R

describe("DriftDetector", {
  it("initializes with default thresholds", {
    detector <- DriftDetector$new()

    config <- detector$get_config()

    expect_equal(config$threshold_mape, 1.2)
    expect_equal(config$window, 30)
  })

  it("sets baseline correctly", {
    detector <- DriftDetector$new()

    baseline <- list(mape = 5.0, mae = 50, rmse = 65)
    detector$set_baseline(baseline)

    expect_equal(detector$get_baseline()$mape, 5.0)
  })

  it("detects drift when MAPE exceeds threshold", {
    detector <- DriftDetector$new(threshold_mape = 1.2)
    detector$set_baseline(list(mape = 5.0, mae = 50, rmse = 65))

    # 30% increase exceeds 20% threshold
    current <- list(mape = 6.5, mae = 55, rmse = 70)

    result <- detector$detect(current)

    expect_true(result$drift_detected)
    expect_true("mape" %in% result$metrics_exceeded)
  })

  it("does not detect drift below threshold", {
    detector <- DriftDetector$new(threshold_mape = 1.2)
    detector$set_baseline(list(mape = 5.0, mae = 50, rmse = 65))

    # 10% increase below 20% threshold
    current <- list(mape = 5.5, mae = 52, rmse = 68)

    result <- detector$detect(current)

    expect_false(result$drift_detected)
  })

  it("calculates correct ratios", {
    detector <- DriftDetector$new()
    detector$set_baseline(list(mape = 5.0, mae = 50, rmse = 65))

    current <- list(mape = 6.0, mae = 60, rmse = 78)

    result <- detector$detect(current)

    expect_equal(result$ratios$mape, 1.2)
    expect_equal(result$ratios$mae, 1.2)
    expect_equal(result$ratios$rmse, 1.2)
  })
})
```

### Comparator Tests

```r
# tests/testthat/test-evaluation-comparator.R

describe("ModelComparator", {
  it("compares multiple models", {
    test_data <- create_model_results()
    comparator <- ModelComparator$new()

    result <- comparator$compare(test_data$model_results, test_data$actual)

    expect_equal(nrow(result), 3)
    expect_true("model_a" %in% result$model)
  })

  it("ranks models by metric", {
    test_data <- create_model_results()
    comparator <- ModelComparator$new()
    comparator$compare(test_data$model_results, test_data$actual)

    ranked <- comparator$rank_models("mape")

    expect_equal(ranked$rank[1], 1)
    expect_true(ranked$best[1])
  })

  it("performs Diebold-Mariano test", {
    test_data <- create_model_results()
    comparator <- ModelComparator$new()
    comparator$compare(test_data$model_results, test_data$actual)

    result <- comparator$diebold_mariano_test("model_a", "model_c")

    expect_true("dm_statistic" %in% names(result))
    expect_true("p_value" %in% names(result))
    expect_true("significant" %in% names(result))
  })
})
```

### Reporter Tests

```r
# tests/testthat/test-evaluation-reporter.R

describe("HTMLReporter", {
  it("initializes with defaults", {
    reporter <- HTMLReporter$new()

    expect_s3_class(reporter, "HTMLReporter")
  })

  it("lists available templates", {
    reporter <- HTMLReporter$new()

    # Skip if templates not installed
    skip_if(length(reporter$list_templates()) == 0)

    templates <- reporter$list_templates()
    expect_true(length(templates) > 0)
  })
})


describe("CSVReporter", {
  it("exports to CSV", {
    reporter <- CSVReporter$new()

    results <- list(
      metrics = data.table::data.table(
        area = c("RJ", "SP"),
        mape = c(3.5, 4.0)
      )
    )

    tmp_file <- tempfile(fileext = ".csv")
    reporter$generate(results, tmp_file)

    expect_true(file.exists(tmp_file))

    # Clean up
    unlink(tmp_file)
  })

  it("round-trips data correctly", {
    reporter <- CSVReporter$new()

    original <- data.table::data.table(
      model = c("A", "B"),
      mape = c(3.5, 4.0)
    )

    tmp_file <- tempfile(fileext = ".csv")
    reporter$write_csv(original, tmp_file)

    read_back <- reporter$read_csv(tmp_file)

    expect_equal(read_back$model, original$model)
    expect_equal(read_back$mape, original$mape, tolerance = 0.0001)

    unlink(tmp_file)
  })
})
```

---

## Test Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| metrics.R | 90% |
| metrics_period.R | 85% |
| metrics_horizon.R | 85% |
| drift.R | 85% |
| comparator.R | 85% |
| reporter.R | 80% |
| highcharter_reporter.R | 75% |
| **Overall** | **≥80%** |

---

## Dependencies

- PC-047-07: Metrics Calculator
- PC-048-07: Metrics by Period
- PC-049-07: Metrics by Horizon
- PC-050-07: DriftDetector
- PC-051-07: ModelComparator
- PC-052-07: HTMLReporter
- PC-053-07: CSVReporter
- PC-054-07: HighcharterReporter

---

## Definition of Done

- [ ] All test files created
- [ ] Test fixtures implemented
- [ ] Metrics tests passing
- [ ] Patamar tests passing
- [ ] Horizon tests passing
- [ ] Drift detection tests passing
- [ ] Model comparison tests passing
- [ ] Reporter tests passing
- [ ] Coverage ≥80%
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use testthat 3.0+ describe/it syntax
- Known-value tests verify mathematical correctness
- Reporter tests may need to skip if templates not installed
- Consider adding integration tests for full evaluation pipeline
