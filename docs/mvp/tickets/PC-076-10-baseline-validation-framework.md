# PC-076-10: Baseline Validation Framework

**Epic:** [EPIC-10: Testing & Validation](../epics/EPIC-10-testing-validation.md)
**Task Reference:** T-10.1
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Create a framework for validating new model implementations against baseline (legacy) results. This ensures that contributed model plugins produce results within acceptable tolerance of the original R implementations.

---

## Acceptance Criteria

- [ ] Validation module created in `tests/validation/baseline.R`
- [ ] Baseline comparison functions with configurable tolerance
- [ ] Support for multiple comparison metrics
- [ ] Detailed difference reporting
- [ ] Automatic pass/fail determination
- [ ] Baseline data loading utilities

---

## Technical Specification

### File Location
```
tests/validation/baseline.R
tests/validation/helpers.R
tests/validation/data/      # Baseline data files
```

### Baseline Validation Framework

```r
#' @title BaselineValidator
#' @description Framework for validating model results against legacy baseline
#'
#' @export
BaselineValidator <- R6::R6Class(
  "BaselineValidator",
  private = list(
    baseline_path = NULL,
    tolerance = NULL,
    metrics = NULL,
    results = list()
  ),

  public = list(
    #' @description Initialize validator
    #' @param baseline_path Path to baseline data directory
    #' @param tolerance Default tolerance for comparisons (relative)
    #' @param metrics Metrics to compare
    initialize = function(baseline_path,
                          tolerance = 0.05,
                          metrics = c("mape", "mae", "rmse")) {
      checkmate::assert_directory_exists(baseline_path)
      checkmate::assert_number(tolerance, lower = 0, upper = 1)
      checkmate::assert_character(metrics, min.len = 1)

      private$baseline_path <- baseline_path
      private$tolerance <- tolerance
      private$metrics <- metrics
    },

    #' @description Load baseline results for a model
    #' @param model_name Model name
    #' @param area Area code (optional, NULL for all)
    #' @return data.table with baseline results
    load_baseline = function(model_name, area = NULL) {
      pattern <- if (is.null(area)) {
        sprintf("%s_*.parquet", model_name)
      } else {
        sprintf("%s_%s.parquet", model_name, area)
      }

      files <- list.files(
        private$baseline_path,
        pattern = pattern,
        full.names = TRUE
      )

      if (length(files) == 0) {
        stop(sprintf("No baseline files found for model '%s'", model_name))
      }

      # Load and combine
      data.table::rbindlist(lapply(files, arrow::read_parquet))
    },

    #' @description Validate model results against baseline
    #' @param model_results Model results data.table
    #' @param model_name Model name
    #' @param area Area code (optional)
    #' @return ValidationResult object
    validate = function(model_results, model_name, area = NULL) {
      # Load baseline
      baseline <- self$load_baseline(model_name, area)

      # Merge on datetime and area
      merged <- merge(
        model_results,
        baseline,
        by = c("datetime", "area_code"),
        suffixes = c("_new", "_baseline")
      )

      if (nrow(merged) == 0) {
        stop("No matching records between results and baseline")
      }

      # Compare each metric
      comparisons <- list()

      for (metric in private$metrics) {
        col_new <- paste0(metric, "_new")
        col_baseline <- paste0(metric, "_baseline")

        if (col_new %in% names(merged) && col_baseline %in% names(merged)) {
          comparison <- self$compare_metric(
            merged[[col_new]],
            merged[[col_baseline]],
            metric
          )
          comparisons[[metric]] <- comparison
        }
      }

      # Calculate overall pass/fail
      passed <- all(sapply(comparisons, function(c) c$passed))

      # Build result
      result <- ValidationResult$new(
        model_name = model_name,
        area = area,
        passed = passed,
        comparisons = comparisons,
        tolerance = private$tolerance,
        n_records = nrow(merged),
        details = merged
      )

      # Store result
      key <- paste(model_name, area %||% "all", sep = "_")
      private$results[[key]] <- result

      result
    },

    #' @description Compare a single metric
    #' @param new_values New model values
    #' @param baseline_values Baseline values
    #' @param metric_name Metric name
    #' @return Comparison result list
    compare_metric = function(new_values, baseline_values, metric_name) {
      # Calculate differences
      abs_diff <- abs(new_values - baseline_values)
      rel_diff <- abs_diff / abs(baseline_values + 1e-10)

      # Statistics
      mean_abs_diff <- mean(abs_diff, na.rm = TRUE)
      max_abs_diff <- max(abs_diff, na.rm = TRUE)
      mean_rel_diff <- mean(rel_diff, na.rm = TRUE)
      max_rel_diff <- max(rel_diff, na.rm = TRUE)

      # Pass/fail based on relative difference
      passed <- mean_rel_diff <= private$tolerance

      list(
        metric = metric_name,
        passed = passed,
        mean_abs_diff = mean_abs_diff,
        max_abs_diff = max_abs_diff,
        mean_rel_diff = mean_rel_diff,
        max_rel_diff = max_rel_diff,
        within_tolerance = sum(rel_diff <= private$tolerance) / length(rel_diff),
        n_values = length(new_values)
      )
    },

    #' @description Validate predictions specifically
    #' @param predictions New predictions
    #' @param baseline_predictions Baseline predictions
    #' @param tolerance Tolerance for comparison
    #' @return Comparison result
    validate_predictions = function(predictions, baseline_predictions,
                                     tolerance = NULL) {
      tolerance <- tolerance %||% private$tolerance

      # Merge on common keys
      merged <- merge(
        predictions,
        baseline_predictions,
        by = c("datetime", "area_code", "horizon"),
        suffixes = c("_new", "_baseline")
      )

      # Compare predicted values
      pred_diff <- abs(merged$prediction_new - merged$prediction_baseline)
      rel_diff <- pred_diff / abs(merged$prediction_baseline + 1e-10)

      list(
        passed = mean(rel_diff) <= tolerance,
        mean_abs_diff = mean(pred_diff),
        max_abs_diff = max(pred_diff),
        mean_rel_diff = mean(rel_diff),
        max_rel_diff = max(rel_diff),
        within_tolerance = mean(rel_diff <= tolerance),
        n_predictions = nrow(merged)
      )
    },

    #' @description Get all validation results
    #' @return List of ValidationResult objects
    get_results = function() {
      private$results
    },

    #' @description Generate summary report
    #' @return data.table summary
    summary = function() {
      if (length(private$results) == 0) {
        return(data.table::data.table())
      }

      data.table::rbindlist(lapply(private$results, function(r) {
        data.table::data.table(
          model = r$model_name,
          area = r$area %||% "all",
          passed = r$passed,
          n_records = r$n_records,
          mape_diff = r$comparisons$mape$mean_rel_diff,
          mae_diff = r$comparisons$mae$mean_rel_diff
        )
      }))
    },

    #' @description Print summary
    print = function() {
      cat("BaselineValidator\n")
      cat(sprintf("  Baseline path: %s\n", private$baseline_path))
      cat(sprintf("  Tolerance: %.1f%%\n", private$tolerance * 100))
      cat(sprintf("  Metrics: %s\n", paste(private$metrics, collapse = ", ")))
      cat(sprintf("  Validations run: %d\n", length(private$results)))

      if (length(private$results) > 0) {
        passed <- sum(sapply(private$results, function(r) r$passed))
        cat(sprintf("  Passed: %d/%d\n", passed, length(private$results)))
      }

      invisible(self)
    }
  )
)
```

### ValidationResult Class

```r
#' @title ValidationResult
#' @description Container for validation result
#' @export
ValidationResult <- R6::R6Class(
  "ValidationResult",
  public = list(
    model_name = NULL,
    area = NULL,
    passed = NULL,
    comparisons = NULL,
    tolerance = NULL,
    n_records = NULL,
    details = NULL,
    timestamp = NULL,

    #' @description Initialize result
    initialize = function(model_name, area, passed, comparisons,
                          tolerance, n_records, details) {
      self$model_name <- model_name
      self$area <- area
      self$passed <- passed
      self$comparisons <- comparisons
      self$tolerance <- tolerance
      self$n_records <- n_records
      self$details <- details
      self$timestamp <- Sys.time()
    },

    #' @description Get comparison for specific metric
    #' @param metric Metric name
    get_comparison = function(metric) {
      self$comparisons[[metric]]
    },

    #' @description Get records exceeding tolerance
    #' @param metric Metric to check
    get_failures = function(metric = "mape") {
      col_new <- paste0(metric, "_new")
      col_baseline <- paste0(metric, "_baseline")

      rel_diff <- abs(self$details[[col_new]] - self$details[[col_baseline]]) /
        abs(self$details[[col_baseline]] + 1e-10)

      self$details[rel_diff > self$tolerance]
    },

    #' @description Export to list
    as_list = function() {
      list(
        model_name = self$model_name,
        area = self$area,
        passed = self$passed,
        tolerance = self$tolerance,
        n_records = self$n_records,
        comparisons = self$comparisons,
        timestamp = self$timestamp
      )
    },

    #' @description Print result
    print = function() {
      status <- if (self$passed) "\u2713 PASSED" else "\u2717 FAILED"
      cat(sprintf("ValidationResult: %s\n", status))
      cat(sprintf("  Model: %s\n", self$model_name))
      cat(sprintf("  Area: %s\n", self$area %||% "all"))
      cat(sprintf("  Records: %d\n", self$n_records))
      cat(sprintf("  Tolerance: %.1f%%\n", self$tolerance * 100))

      cat("\nMetric Comparisons:\n")
      for (name in names(self$comparisons)) {
        comp <- self$comparisons[[name]]
        status <- if (comp$passed) "\u2713" else "\u2717"
        cat(sprintf("  %s %s: mean_diff=%.4f, max_diff=%.4f (%.1f%% within tolerance)\n",
                    status, name,
                    comp$mean_rel_diff, comp$max_rel_diff,
                    comp$within_tolerance * 100))
      }

      invisible(self)
    }
  )
)
```

### Helper Functions

```r
#' Compare model results against baseline
#'
#' Convenience function for quick validation.
#'
#' @param model_results Model results data.table
#' @param baseline_path Path to baseline data
#' @param model_name Model name
#' @param tolerance Relative tolerance
#' @return ValidationResult
#' @export
validate_against_baseline <- function(model_results, baseline_path, model_name,
                                       tolerance = 0.05) {
  validator <- BaselineValidator$new(
    baseline_path = baseline_path,
    tolerance = tolerance
  )

  validator$validate(model_results, model_name)
}


#' Load baseline data for comparison
#'
#' @param baseline_dir Baseline data directory
#' @param model Model name
#' @param area Area code (optional)
#' @return data.table
#' @export
load_baseline_data <- function(baseline_dir, model, area = NULL) {
  validator <- BaselineValidator$new(baseline_dir)
  validator$load_baseline(model, area)
}


#' Check if model passes baseline validation
#'
#' @param model_results Model results
#' @param baseline_path Baseline path
#' @param model_name Model name
#' @param tolerance Tolerance
#' @return Logical
#' @export
passes_baseline <- function(model_results, baseline_path, model_name,
                            tolerance = 0.05) {
  result <- validate_against_baseline(
    model_results, baseline_path, model_name, tolerance
  )
  result$passed
}
```

### Baseline Data Schema

```r
# Expected baseline data schema
# baseline_lgbm_RJ.parquet:
#   - datetime: POSIXct
#   - area_code: character
#   - horizon: integer (0-8)
#   - prediction: numeric
#   - actual: numeric
#   - mape: numeric
#   - mae: numeric
#   - rmse: numeric

# Create baseline from legacy model
create_baseline_data <- function(legacy_results, output_dir, model_name) {
  for (area in unique(legacy_results$area_code)) {
    area_data <- legacy_results[area_code == area]

    output_path <- file.path(
      output_dir,
      sprintf("%s_%s.parquet", model_name, area)
    )

    arrow::write_parquet(area_data, output_path)
  }
}
```

### Usage Example

```r
# Initialize validator
validator <- BaselineValidator$new(
  baseline_path = "tests/validation/data/baseline",
  tolerance = 0.05,  # 5% relative tolerance
  metrics = c("mape", "mae", "rmse")
)

# Run new model
model <- get_model("lgbm")
model$train(train_data$X, train_data$y)
predictions <- model$predict(test_data$X)

# Calculate metrics
calculator <- MetricsCalculator$new()
model_results <- data.table(
  datetime = test_data$datetime,
  area_code = test_data$area_code,
  prediction = predictions,
  actual = test_data$y,
  mape = calculator$mape(test_data$y, predictions),
  mae = calculator$mae(test_data$y, predictions),
  rmse = calculator$rmse(test_data$y, predictions)
)

# Validate
result <- validator$validate(model_results, "lgbm", area = "RJ")
print(result)
# ValidationResult: ✓ PASSED
#   Model: lgbm
#   Area: RJ
#   Records: 8760
#   Tolerance: 5.0%
#
# Metric Comparisons:
#   ✓ mape: mean_diff=0.0234, max_diff=0.0456 (98.5% within tolerance)
#   ✓ mae: mean_diff=0.0189, max_diff=0.0378 (99.1% within tolerance)
#   ✓ rmse: mean_diff=0.0267, max_diff=0.0512 (97.8% within tolerance)

# Get summary
summary <- validator$summary()
print(summary)

# Get failure details
failures <- result$get_failures("mape")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize validator | Validator created |
| TC-002 | Load baseline data | Data loaded |
| TC-003 | Validate within tolerance | PASSED |
| TC-004 | Validate outside tolerance | FAILED |
| TC-005 | Compare predictions | Difference calculated |
| TC-006 | Get failure details | Failures returned |
| TC-007 | Summary generation | Summary table |
| TC-008 | Missing baseline file | Error thrown |
| TC-009 | Empty merge | Error thrown |
| TC-010 | Multiple areas | All validated |

---

## Dependencies

- PC-047-07: MetricsCalculator

---

## Definition of Done

- [ ] BaselineValidator R6 class implemented
- [ ] ValidationResult container implemented
- [ ] Baseline data loading working
- [ ] Metric comparison with tolerance
- [ ] Pass/fail determination
- [ ] Failure detail extraction
- [ ] Summary report generation
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Default tolerance is 5% relative difference
- Baseline data stored in Parquet format
- Supports validation by area or all areas
- Consider adding statistical significance tests
