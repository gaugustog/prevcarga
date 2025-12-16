# PC-077-10: Model Results Reproduction

**Epic:** [EPIC-10: Testing & Validation](../epics/EPIC-10-testing-validation.md)
**Task Reference:** T-10.2
**Priority:** High
**Estimated Effort:** 3 days

---

## Summary

Reproduce existing R model results using the new PrevCarga infrastructure to validate that contributed model plugins produce equivalent outputs within acceptable tolerance (±5% MAPE).

---

## Acceptance Criteria

- [ ] Reproduction script created in `tests/validation/reproduce_models.R`
- [ ] All registered models validated against baseline
- [ ] MAPE within ±5% of legacy results
- [ ] Discrepancies documented
- [ ] Reproduction report generated

---

## Technical Specification

### File Location
```
tests/validation/reproduce_models.R
tests/validation/model_reproduction_report.Rmd
```

### Model Reproduction Script

```r
#' @title ModelReproduction
#' @description Reproduces legacy model results for validation
#' @export
ModelReproduction <- R6::R6Class(
  "ModelReproduction",
  private = list(
    config = NULL,
    storage = NULL,
    baseline_path = NULL,
    output_path = NULL,
    results = list(),
    tolerance = 0.05
  ),

  public = list(
    #' @description Initialize reproduction
    #' @param config ConfigManager instance
    #' @param baseline_path Path to baseline data
    #' @param output_path Output directory for results
    initialize = function(config, baseline_path, output_path) {
      private$config <- config
      private$baseline_path <- baseline_path
      private$output_path <- output_path
      private$storage <- create_storage_backend(config$get_storage_config())

      dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
    },

    #' @description Run reproduction for all registered models
    #' @param areas Area codes to validate
    #' @param start_date Start date for validation period
    #' @param end_date End date for validation period
    #' @return ReproductionReport
    run_all = function(areas = NULL, start_date, end_date) {
      models <- list_models()
      areas <- areas %||% private$config$get_areas()

      cat(sprintf("\n=== Model Reproduction Validation ===\n"))
      cat(sprintf("Models: %s\n", paste(models, collapse = ", ")))
      cat(sprintf("Areas: %d\n", length(areas)))
      cat(sprintf("Period: %s to %s\n", start_date, end_date))
      cat("\n")

      for (model_name in models) {
        cat(sprintf("Validating model: %s\n", model_name))

        tryCatch({
          result <- self$reproduce_model(
            model_name = model_name,
            areas = areas,
            start_date = start_date,
            end_date = end_date
          )
          private$results[[model_name]] <- result

          status <- if (result$passed) "\u2713 PASSED" else "\u2717 FAILED"
          cat(sprintf("  %s (MAPE diff: %.2f%%)\n",
                      status, result$overall_mape_diff * 100))

        }, error = function(e) {
          cat(sprintf("  \u2717 ERROR: %s\n", conditionMessage(e)))
          private$results[[model_name]] <- list(
            model = model_name,
            passed = FALSE,
            error = conditionMessage(e)
          )
        })
      }

      self$generate_report()
    },

    #' @description Reproduce single model
    #' @param model_name Model name
    #' @param areas Area codes
    #' @param start_date Start date
    #' @param end_date End date
    #' @return Model reproduction result
    reproduce_model = function(model_name, areas, start_date, end_date) {
      # Load baseline
      validator <- BaselineValidator$new(
        baseline_path = private$baseline_path,
        tolerance = private$tolerance
      )

      # Load training data
      loader <- DataLoader$new(private$storage)
      train_end <- as.Date(start_date) - 1
      train_start <- train_end - 365  # 1 year training

      area_results <- list()

      for (area in areas) {
        cat(sprintf("    Area %s: ", area))

        # Load and prepare data
        train_data <- loader$load_carga(
          areas = area,
          start_date = train_start,
          end_date = train_end
        )

        test_data <- loader$load_carga(
          areas = area,
          start_date = start_date,
          end_date = end_date
        )

        # Get model with config
        model_config <- private$config$get_model_config(model_name)
        model <- get_model(model_name, model_config$config %||% list())

        # Build feature pipeline
        pipeline <- self$build_pipeline(model_name)

        # Transform data
        train_features <- pipeline$transform(train_data)
        test_features <- pipeline$transform(test_data)

        # Train model
        model$train(train_features$X, train_features$y)

        # Predict
        predictions <- model$predict(test_features$X)

        # Calculate metrics
        calculator <- MetricsCalculator$new()
        metrics <- calculator$calculate_all(test_features$y, predictions)

        # Build results for comparison
        model_results <- data.table::data.table(
          datetime = test_data$datetime,
          area_code = area,
          prediction = predictions,
          actual = test_features$y,
          mape = metrics$mape,
          mae = metrics$mae,
          rmse = metrics$rmse
        )

        # Validate against baseline
        validation <- validator$validate(model_results, model_name, area)
        area_results[[area]] <- validation

        status <- if (validation$passed) "\u2713" else "\u2717"
        cat(sprintf("%s (MAPE diff: %.2f%%)\n",
                    status,
                    validation$comparisons$mape$mean_rel_diff * 100))
      }

      # Aggregate results
      all_passed <- all(sapply(area_results, function(r) r$passed))
      overall_mape_diff <- mean(sapply(area_results, function(r) {
        r$comparisons$mape$mean_rel_diff
      }))

      list(
        model = model_name,
        passed = all_passed,
        overall_mape_diff = overall_mape_diff,
        area_results = area_results,
        n_areas_passed = sum(sapply(area_results, function(r) r$passed)),
        n_areas_total = length(area_results)
      )
    },

    #' @description Build feature pipeline for model
    #' @param model_name Model name
    build_pipeline = function(model_name) {
      pipeline <- FeaturePipeline$new()

      # Get enabled plugins from config
      feature_config <- private$config$get("features.plugins")

      for (plugin_name in names(feature_config)) {
        plugin_cfg <- feature_config[[plugin_name]]
        if (isTRUE(plugin_cfg$enabled)) {
          plugin <- get_feature_plugin(plugin_name, plugin_cfg$config %||% list())
          pipeline$add_plugin(plugin)
        }
      }

      pipeline
    },

    #' @description Generate reproduction report
    generate_report = function() {
      report_path <- file.path(
        private$output_path,
        sprintf("reproduction_report_%s.html",
                format(Sys.time(), "%Y%m%d_%H%M%S"))
      )

      # Render R Markdown report
      rmarkdown::render(
        input = system.file(
          "validation/model_reproduction_report.Rmd",
          package = "prevcargaons"
        ),
        output_file = report_path,
        params = list(
          results = private$results,
          tolerance = private$tolerance,
          timestamp = Sys.time()
        )
      )

      cat(sprintf("\nReport generated: %s\n", report_path))

      report_path
    },

    #' @description Get summary of results
    #' @return data.table summary
    summary = function() {
      data.table::rbindlist(lapply(names(private$results), function(model) {
        r <- private$results[[model]]

        if (!is.null(r$error)) {
          return(data.table::data.table(
            model = model,
            passed = FALSE,
            mape_diff = NA_real_,
            areas_passed = NA_integer_,
            areas_total = NA_integer_,
            error = r$error
          ))
        }

        data.table::data.table(
          model = model,
          passed = r$passed,
          mape_diff = r$overall_mape_diff,
          areas_passed = r$n_areas_passed,
          areas_total = r$n_areas_total,
          error = NA_character_
        )
      }))
    },

    #' @description Get detailed results
    get_results = function() {
      private$results
    },

    #' @description Print summary
    print = function() {
      cat("\nModel Reproduction Summary\n")
      cat("==========================\n\n")

      summary_dt <- self$summary()
      print(summary_dt)

      n_passed <- sum(summary_dt$passed, na.rm = TRUE)
      n_total <- nrow(summary_dt)

      cat(sprintf("\nOverall: %d/%d models passed validation\n",
                  n_passed, n_total))

      invisible(self)
    }
  )
)
```

### Reproduction Runner Script

```r
#!/usr/bin/env Rscript
# tests/validation/reproduce_models.R
#
# Run model reproduction validation
#
# Usage: Rscript reproduce_models.R [--config config.yaml]

library(prevcargaons)

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)
config_path <- "config/config.yaml"

if ("--config" %in% args) {
  idx <- which(args == "--config")
  config_path <- args[idx + 1]
}

# Load configuration
config <- ConfigManager$new()
config$load(config_path)

# Initialize reproduction
reproduction <- ModelReproduction$new(
  config = config,
  baseline_path = "tests/validation/data/baseline",
  output_path = "reports/validation"
)

# Run for validation period
results <- reproduction$run_all(
  areas = config$get_areas(),
  start_date = "2024-01-01",
  end_date = "2024-03-31"  # Q1 2024 for quick validation
)

# Print summary
print(reproduction)

# Exit with appropriate code
summary_dt <- reproduction$summary()
if (all(summary_dt$passed, na.rm = TRUE)) {
  cat("\n\u2713 All models passed validation!\n")
  quit(status = 0)
} else {
  cat("\n\u2717 Some models failed validation\n")
  quit(status = 1)
}
```

### Discrepancy Documentation Template

```r
#' Document model discrepancy
#'
#' @param model_name Model name
#' @param area Area code
#' @param expected Expected metric value
#' @param actual Actual metric value
#' @param reason Explanation for discrepancy
#' @export
document_discrepancy <- function(model_name, area, expected, actual, reason) {
  discrepancy <- list(
    model = model_name,
    area = area,
    expected = expected,
    actual = actual,
    difference = abs(actual - expected) / expected,
    reason = reason,
    timestamp = Sys.time(),
    acceptable = FALSE  # Must be manually reviewed
  )

  # Append to discrepancy log
  log_path <- "tests/validation/discrepancies.yaml"

  existing <- if (file.exists(log_path)) {
    yaml::read_yaml(log_path)
  } else {
    list()
  }

  existing[[length(existing) + 1]] <- discrepancy
  yaml::write_yaml(existing, log_path)

  discrepancy
}
```

### Usage Example

```r
# Initialize
config <- load_config("config/config.yaml")

reproduction <- ModelReproduction$new(
  config = config,
  baseline_path = "tests/validation/data/baseline",
  output_path = "reports/validation"
)

# Run all models
reproduction$run_all(
  start_date = "2024-01-01",
  end_date = "2024-03-31"
)

# Output:
# === Model Reproduction Validation ===
# Models: lgbm, rf, hw
# Areas: 21
# Period: 2024-01-01 to 2024-03-31
#
# Validating model: lgbm
#     Area RJ: ✓ (MAPE diff: 2.34%)
#     Area SP: ✓ (MAPE diff: 1.89%)
#     ...
#   ✓ PASSED (MAPE diff: 2.15%)
#
# Validating model: rf
#     Area RJ: ✓ (MAPE diff: 3.12%)
#     ...
#   ✓ PASSED (MAPE diff: 2.87%)
#
# Model Reproduction Summary
# ==========================
#
#    model passed mape_diff areas_passed areas_total
# 1:  lgbm   TRUE    0.0215           21          21
# 2:    rf   TRUE    0.0287           21          21
# 3:    hw   TRUE    0.0342           21          21
#
# Overall: 3/3 models passed validation
# ✓ All models passed validation!
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Reproduce lgbm model | Within 5% MAPE |
| TC-002 | Reproduce rf model | Within 5% MAPE |
| TC-003 | Reproduce hw model | Within 5% MAPE |
| TC-004 | All areas validated | All areas pass |
| TC-005 | Report generated | HTML report exists |
| TC-006 | Summary accurate | Matches results |
| TC-007 | Discrepancy documented | Logged correctly |
| TC-008 | Missing baseline | Error handled |
| TC-009 | Model error | Error captured |
| TC-010 | Different tolerance | Applied correctly |

---

## Dependencies

- PC-076-10: Baseline Validation Framework
- PC-017-03: ModelRegistry
- PC-011-02: FeaturePipeline
- PC-047-07: MetricsCalculator

---

## Definition of Done

- [ ] ModelReproduction class implemented
- [ ] All registered models tested
- [ ] Results within ±5% MAPE tolerance
- [ ] Discrepancies documented
- [ ] HTML report generated
- [ ] Runner script working
- [ ] roxygen2 documentation complete
- [ ] Validation passing
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Baseline data must be prepared before running
- Run on representative validation period (e.g., Q1 2024)
- Document any legitimate discrepancies
- Consider model version differences in baseline
