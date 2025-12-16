# EPIC-07: Evaluation Layer

**Duration:** 2 weeks
**Dependencies:** EPIC-05, EPIC-06
**Reference:** [MVP Plan R - Phase 7](../mvp-plan-r.md#phase-7-evaluation-layer-2-weeks)

---

## Objective

Create the evaluation infrastructure for calculating metrics, detecting drift, comparing models, and generating reports.

---

## Scope

This epic covers:
- Metrics calculation (MAPE, MAE, RMSE, percentiles)
- Metrics by time period (patamar: peak/off-peak)
- Drift detection framework
- Model comparison framework
- Report generation (HTML, CSV)

**Out of Scope:**
- Model implementations
- Combination strategies
- Reconciliation algorithms

---

## Tasks

### T-07.1: Create Metrics Calculator
- [ ] Create `R/evaluation/metrics.R`
- [ ] Implement core metrics functions:
  ```r
  calculate_mape = function(actual, predicted) {
    mean(abs((actual - predicted) / actual), na.rm = TRUE) * 100
  }

  calculate_mae = function(actual, predicted) {
    mean(abs(actual - predicted), na.rm = TRUE)
  }

  calculate_rmse = function(actual, predicted) {
    sqrt(mean((actual - predicted)^2, na.rm = TRUE))
  }

  calculate_percentiles = function(errors, probs = c(0.05, 0.25, 0.5, 0.75, 0.95)) {
    quantile(abs(errors), probs = probs, na.rm = TRUE)
  }

  calculate_all_metrics = function(actual, predicted) {
    errors <- actual - predicted
    list(
      mape = calculate_mape(actual, predicted),
      mae = calculate_mae(actual, predicted),
      rmse = calculate_rmse(actual, predicted),
      percentiles = calculate_percentiles(errors),
      max_deviation_abs = max(abs(errors), na.rm = TRUE),
      max_deviation_rel = max(abs((errors) / actual), na.rm = TRUE) * 100
    )
  }
  ```

### T-07.2: Create Metrics by Period Calculator
- [ ] Add support for patamar (peak/off-peak) breakdown:
  ```r
  calculate_metrics_by_patamar = function(dt, patamar_config) {
    # dt must have: DataHora, actual, predicted, area_code

    # Classify each row into patamar
    dt[, patamar := classify_patamar(DataHora, area_code, patamar_config)]

    # Calculate metrics per patamar
    dt[, calculate_all_metrics(actual, predicted), by = patamar]
  }

  classify_patamar = function(datetime, area_code, config) {
    # Determine peak/off-peak based on config
  }
  ```

### T-07.3: Create Metrics by Horizon Calculator
- [ ] Add support for horizon breakdown (D+0 to D+8):
  ```r
  calculate_metrics_by_horizon = function(dt) {
    # dt must have: horizon, actual, predicted
    dt[, calculate_all_metrics(actual, predicted), by = horizon]
  }
  ```

### T-07.4: Create DriftDetector
- [ ] Create `R/evaluation/drift.R`
- [ ] Implement `DriftDetector` R6 class:
  ```r
  DriftDetector <- R6::R6Class(
    "DriftDetector",
    private = list(
      window = NULL,
      threshold_mape = NULL,
      threshold_mae = NULL,
      baseline_metrics = NULL
    ),
    public = list(
      initialize = function(window = 30, threshold_mape = 1.2,
                           threshold_mae = 1.15) {
        private$window <- window
        private$threshold_mape <- threshold_mape
        private$threshold_mae <- threshold_mae
      },

      set_baseline = function(metrics) {
        private$baseline_metrics <- metrics
      },

      detect = function(current_metrics) {
        drift_detected <- FALSE
        reasons <- character()

        if (current_metrics$mape / private$baseline_metrics$mape >
            private$threshold_mape) {
          drift_detected <- TRUE
          reasons <- c(reasons, "MAPE exceeded threshold")
        }

        if (current_metrics$mae / private$baseline_metrics$mae >
            private$threshold_mae) {
          drift_detected <- TRUE
          reasons <- c(reasons, "MAE exceeded threshold")
        }

        list(
          drift_detected = drift_detected,
          reasons = reasons,
          mape_ratio = current_metrics$mape / private$baseline_metrics$mape,
          mae_ratio = current_metrics$mae / private$baseline_metrics$mae
        )
      }
    )
  )
  ```

### T-07.5: Create ModelComparator
- [ ] Create `R/evaluation/comparator.R`
- [ ] Implement `ModelComparator` R6 class:
  ```r
  ModelComparator <- R6::R6Class(
    "ModelComparator",
    public = list(
      compare = function(model_results, test_data) {
        # Compare multiple models on same test set
        comparison <- lapply(model_results, function(result) {
          metrics <- calculate_all_metrics(
            test_data$actual,
            result$predictions
          )
          list(
            model = result$model_name,
            metrics = metrics
          )
        })

        # Create comparison matrix
        self$build_comparison_matrix(comparison)
      },

      build_comparison_matrix = function(comparison) {
        # Create data.table with models as rows, metrics as columns
      },

      rank_models = function(comparison, metric = "mape") {
        # Rank models by specified metric
      },

      statistical_test = function(model_a_errors, model_b_errors) {
        # Diebold-Mariano test for forecast comparison
      }
    )
  )
  ```

### T-07.6: Create HTMLReporter
- [ ] Create `R/evaluation/reporter.R`
- [ ] Implement `HTMLReporter` R6 class:
  ```r
  HTMLReporter <- R6::R6Class(
    "HTMLReporter",
    public = list(
      generate = function(evaluation_results, output_path) {
        # Generate HTML report using rmarkdown
        rmarkdown::render(
          input = system.file("templates/report.Rmd", package = "prevcargaons"),
          output_file = output_path,
          params = list(results = evaluation_results)
        )
      }
    )
  )
  ```
- [ ] Create report template in `inst/templates/report.Rmd`

### T-07.7: Create CSVReporter
- [ ] Implement `CSVReporter` R6 class:
  ```r
  CSVReporter <- R6::R6Class(
    "CSVReporter",
    public = list(
      generate = function(evaluation_results, output_path) {
        # Export metrics to CSV
        data.table::fwrite(evaluation_results$metrics, output_path)
      }
    )
  )
  ```

### T-07.8: Write Tests
- [ ] Create `tests/testthat/test-evaluation.R`
- [ ] Test metric calculations with known values
- [ ] Test patamar classification
- [ ] Test drift detection
- [ ] Test model comparison

### T-07.9: Create HighcharterReporter
- [ ] Add `highcharter`, `rmarkdown`, `htmlwidgets` to dependencies
- [ ] Create `R/evaluation/highcharter_reporter.R`
- [ ] Implement `HighcharterReporter` R6 class:
  ```r
  HighcharterReporter <- R6::R6Class(
    "HighcharterReporter",
    public = list(
      generate = function(report_type, data, output_path, config = list()) {
        template <- self$get_template(report_type)
        rmarkdown::render(
          input = template,
          output_file = output_path,
          params = list(data = data, config = config)
        )
      },

      get_template = function(type) {
        system.file(
          sprintf("templates/reports/%s.Rmd", type),
          package = "prevcarga"
        )
      },

      available_reports = function() {
        c("forecast", "backtest", "model_comparison",
          "area_dashboard", "drift")
      }
    )
  )
  ```

### T-07.10: Create Highcharter Report Templates
- [ ] Create `inst/templates/reports/` directory
- [ ] Create `inst/templates/reports/_common.R` - Shared highcharter helpers
- [ ] Create `inst/templates/reports/forecast_report.Rmd`:
  ```yaml
  ---
  title: "Forecast Report - `r params$date`"
  output:
    html_document:
      self_contained: true
      theme: flatly
  params:
    data: NULL
    config: NULL
  ---
  ```
  - Forecast vs Actual time series chart
  - Error distribution histogram
  - Metrics by hour column chart
  - Confidence interval visualization
- [ ] Create `inst/templates/reports/backtest_report.Rmd`:
  - Metrics by horizon line chart
  - Retraining interval comparison
  - Rolling MAPE sparkline
  - Best model ranking table
- [ ] Create `inst/templates/reports/model_comparison.Rmd`:
  - Side-by-side MAPE comparison
  - Radar chart for multi-metric comparison
  - Statistical significance table
  - Model ranking by area
- [ ] Create `inst/templates/reports/area_dashboard.Rmd`:
  - Load profile time series
  - Seasonal decomposition plots
  - Hour-of-day heatmap
  - Day-of-week patterns
- [ ] Create `inst/templates/reports/drift_report.Rmd`:
  - Rolling metrics time series
  - Threshold violation markers
  - Baseline vs current comparison
  - Drift alert summary

---

## Acceptance Criteria

- [ ] All metrics calculate correctly (verified with known test cases)
- [ ] Patamar breakdown works with configuration
- [ ] Horizon breakdown groups by D+0 to D+8
- [ ] DriftDetector flags degradation above threshold
- [ ] ModelComparator ranks models by metrics
- [ ] HTMLReporter generates readable reports
- [ ] CSVReporter exports valid CSV
- [ ] HighcharterReporter generates interactive HTML reports
- [ ] All 5 report templates render without errors
- [ ] Tests pass with ≥80% coverage

---

## Definition of Done

- [ ] All tasks completed
- [ ] Code reviewed and merged
- [ ] Tests passing (≥80% coverage)
- [ ] Documentation complete with roxygen2
- [ ] Report template verified

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `R/evaluation/metrics.R` | Create | Metric calculation functions |
| `R/evaluation/drift.R` | Create | DriftDetector |
| `R/evaluation/comparator.R` | Create | ModelComparator |
| `R/evaluation/reporter.R` | Create | HTMLReporter, CSVReporter |
| `R/evaluation/highcharter_reporter.R` | Create | HighcharterReporter |
| `inst/templates/report.Rmd` | Create | HTML report template |
| `inst/templates/reports/_common.R` | Create | Shared highcharter helpers |
| `inst/templates/reports/forecast_report.Rmd` | Create | Forecast analysis report |
| `inst/templates/reports/backtest_report.Rmd` | Create | Backtest results report |
| `inst/templates/reports/model_comparison.Rmd` | Create | Model comparison report |
| `inst/templates/reports/area_dashboard.Rmd` | Create | Area deep-dive dashboard |
| `inst/templates/reports/drift_report.Rmd` | Create | Drift analysis report |
| `tests/testthat/test-evaluation.R` | Create | Evaluation tests |

---

## Metrics Reference

| Metric | Formula | Description |
|--------|---------|-------------|
| MAPE | `mean(abs((a-p)/a)) * 100` | Mean Absolute Percentage Error |
| MAE | `mean(abs(a-p))` | Mean Absolute Error |
| RMSE | `sqrt(mean((a-p)^2))` | Root Mean Square Error |
| P5-P95 | `quantile(abs(e), probs)` | Error distribution percentiles |
| Max Dev Abs | `max(abs(e))` | Maximum absolute deviation |
| Max Dev Rel | `max(abs(e/a)) * 100` | Maximum relative deviation |

## Patamar Configuration Reference

```yaml
patamares:
  SECO:
    Jan: {ponta_inicio: "17:30", ponta_fim: "20:30"}
    Fev: {ponta_inicio: "17:30", ponta_fim: "20:30"}
    # ... other months
```
