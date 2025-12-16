# PC-034-05: CombinationWorkflow

**Epic:** [EPIC-05: Combination Infrastructure](../epics/EPIC-05-combination-layer.md)
**Task Reference:** T-05.3
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `CombinationWorkflow` R6 class that orchestrates the complete forecast combination process, including gathering model forecasts, fitting weights, applying combination, and optional bias correction.

---

## Acceptance Criteria

- [ ] Workflow module created in `R/combination/workflow.R`
- [ ] `CombinationWorkflow` R6 class with full orchestration
- [ ] Support for multiple combiner strategies
- [ ] Optional validation data for weight fitting
- [ ] Integration with BiasCorrector
- [ ] Result object with combination metadata

---

## Technical Specification

### File Location
```
R/combination/workflow.R
```

### CombinationWorkflow R6 Class

```r
#' @title CombinationWorkflow
#' @description Orchestrates forecast combination process
#'
#' Manages the complete combination pipeline:
#' 1. Gather forecasts from multiple models
#' 2. Optionally fit combination weights
#' 3. Apply combination strategy
#' 4. Optionally apply bias correction
#' 5. Return combined forecast with metadata
#'
#' @export
CombinationWorkflow <- R6::R6Class(
  "CombinationWorkflow",
  private = list(
    combiner = NULL,
    bias_corrector = NULL,
    model_forecasts = NULL,
    config = NULL
  ),
  public = list(
    #' @description Initialize workflow
    #' @param combiner BaseCombiner instance or name
    #' @param bias_corrector Optional BiasCorrector instance
    #' @param config Workflow configuration
    initialize = function(combiner,
                          bias_corrector = NULL,
                          config = list()) {
      # Accept combiner name or instance
      if (is.character(combiner)) {
        private$combiner <- get_combiner(combiner, config = config$combiner_config %||% list())
      } else {
        checkmate::assert_class(combiner, "BaseCombiner")
        private$combiner <- combiner
      }

      if (!is.null(bias_corrector)) {
        checkmate::assert_class(bias_corrector, "BiasCorrector")
      }
      private$bias_corrector <- bias_corrector

      private$config <- private$apply_default_config(config)
      private$model_forecasts <- list()
    },

    #' @description Run combination workflow
    #' @param model_results Named list of model results (each with $predictions)
    #' @param validation_data Optional data.table with actuals for weight fitting
    #' @param ... Additional arguments
    #' @return CombinationResult with combined forecast
    run = function(model_results, validation_data = NULL, ...) {
      checkmate::assert_list(model_results, min.len = 1, names = "named")

      start_time <- Sys.time()

      # Step 1: Extract forecasts from model results
      message("Extracting forecasts from model results...")
      forecasts <- self$extract_forecasts(model_results)

      # Step 2: Validate forecasts
      private$combiner$validate_forecasts(forecasts)
      private$model_forecasts <- forecasts

      # Step 3: Fit weights if validation data provided
      if (!is.null(validation_data) && private$config$fit_weights) {
        message("Fitting combination weights...")
        actuals <- self$extract_actuals(validation_data)
        private$combiner$fit_weights(forecasts, actuals, ...)
      }

      # Step 4: Combine forecasts
      message(sprintf("Combining %d forecasts using %s...",
                      length(forecasts), class(private$combiner)[1]))
      combined <- private$combiner$combine(forecasts, ...)

      # Step 5: Apply bias correction if configured
      if (!is.null(private$bias_corrector) && private$config$apply_bias_correction) {
        message("Applying bias correction...")
        combined <- private$bias_corrector$correct(combined)
      }

      # Step 6: Build result
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      CombinationResult$new(
        predictions = combined,
        weights = private$combiner$get_weights(),
        model_names = names(forecasts),
        combiner_name = class(private$combiner)[1],
        bias_corrected = !is.null(private$bias_corrector) &&
                         private$config$apply_bias_correction,
        elapsed_time = elapsed,
        config = private$config
      )
    },

    #' @description Run combination with cross-validation for weight optimization
    #' @param model_results Named list of model results
    #' @param actuals Full actuals vector
    #' @param folds Number of CV folds
    #' @param ... Additional arguments
    #' @return CombinationResult with CV-optimized weights
    run_with_cv = function(model_results, actuals, folds = 5, ...) {
      forecasts <- self$extract_forecasts(model_results)
      n <- length(actuals)

      message(sprintf("Running %d-fold CV for weight optimization...", folds))

      fold_weights <- list()
      fold_metrics <- list()

      fold_size <- floor(n / folds)

      for (i in seq_len(folds)) {
        # Define fold indices (time series CV - expanding window)
        train_end <- fold_size * i
        val_start <- train_end + 1
        val_end <- min(train_end + fold_size, n)

        if (val_start > n) break

        # Subset forecasts and actuals
        train_forecasts <- lapply(forecasts, function(f) f[1:train_end])
        train_actuals <- actuals[1:train_end]

        val_forecasts <- lapply(forecasts, function(f) f[val_start:val_end])
        val_actuals <- actuals[val_start:val_end]

        # Fit weights on training fold
        fold_combiner <- private$combiner$clone()
        fold_combiner$fit_weights(train_forecasts, train_actuals, ...)

        # Evaluate on validation fold
        val_combined <- fold_combiner$combine(val_forecasts)
        fold_mape <- mean(abs((val_actuals - val_combined) / val_actuals)) * 100

        fold_weights[[i]] <- fold_combiner$get_weights()
        fold_metrics[[i]] <- list(fold = i, mape = fold_mape)

        message(sprintf("  Fold %d MAPE: %.2f%%", i, fold_mape))
      }

      # Average weights across folds
      avg_weights <- Reduce(`+`, fold_weights) / length(fold_weights)
      private$combiner$set_weights(avg_weights)

      # Final combination
      combined <- private$combiner$combine(forecasts)

      CombinationResult$new(
        predictions = combined,
        weights = avg_weights,
        model_names = names(forecasts),
        combiner_name = class(private$combiner)[1],
        cv_metrics = fold_metrics,
        config = private$config
      )
    },

    #' @description Extract forecasts from model results
    #' @param model_results Named list
    #' @return Named list of forecast vectors
    extract_forecasts = function(model_results) {
      lapply(model_results, function(result) {
        if (is.list(result) && "predictions" %in% names(result)) {
          result$predictions
        } else if (is.numeric(result)) {
          result
        } else {
          stop("Model result must have 'predictions' field or be numeric")
        }
      })
    },

    #' @description Extract actuals from validation data
    #' @param validation_data data.table with actuals
    #' @param target_col Target column name
    #' @return Numeric vector
    extract_actuals = function(validation_data, target_col = NULL) {
      target_col <- target_col %||% private$config$target_col

      if (is.data.table(validation_data) || is.data.frame(validation_data)) {
        if (!target_col %in% names(validation_data)) {
          stop(sprintf("Target column '%s' not found in validation data", target_col))
        }
        validation_data[[target_col]]
      } else if (is.numeric(validation_data)) {
        validation_data
      } else {
        stop("validation_data must be data.table, data.frame, or numeric vector")
      }
    },

    #' @description Get the combiner
    #' @return BaseCombiner instance
    get_combiner = function() {
      private$combiner
    },

    #' @description Set a different combiner
    #' @param combiner BaseCombiner instance or name
    #' @return Invisible self
    set_combiner = function(combiner) {
      if (is.character(combiner)) {
        private$combiner <- get_combiner(combiner)
      } else {
        checkmate::assert_class(combiner, "BaseCombiner")
        private$combiner <- combiner
      }
      invisible(self)
    },

    #' @description Get stored model forecasts
    #' @return Named list
    get_model_forecasts = function() {
      private$model_forecasts
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        fit_weights = TRUE,
        apply_bias_correction = TRUE,
        target_col = "CargaGlobal",
        lookback_days = 30
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

### CombinationResult Class

```r
#' @title CombinationResult
#' @description Container for combination results with metadata
#' @export
CombinationResult <- R6::R6Class(
  "CombinationResult",
  public = list(
    predictions = NULL,
    weights = NULL,
    model_names = NULL,
    combiner_name = NULL,
    bias_corrected = FALSE,
    elapsed_time = NULL,
    cv_metrics = NULL,
    config = NULL,
    created_at = NULL,

    #' @description Initialize result
    initialize = function(predictions,
                          weights = NULL,
                          model_names = NULL,
                          combiner_name = NULL,
                          bias_corrected = FALSE,
                          elapsed_time = NULL,
                          cv_metrics = NULL,
                          config = list()) {
      self$predictions <- predictions
      self$weights <- weights
      self$model_names <- model_names
      self$combiner_name <- combiner_name
      self$bias_corrected <- bias_corrected
      self$elapsed_time <- elapsed_time
      self$cv_metrics <- cv_metrics
      self$config <- config
      self$created_at <- Sys.time()
    },

    #' @description Get predictions as data.table
    #' @param datetime Optional datetime vector
    #' @return data.table
    as_data_table = function(datetime = NULL) {
      dt <- data.table::data.table(
        prediction = self$predictions
      )

      if (!is.null(datetime)) {
        dt[, datetime := datetime]
        data.table::setcolorder(dt, c("datetime", "prediction"))
      }

      dt
    },

    #' @description Get weight summary
    #' @return data.table with model weights
    get_weight_summary = function() {
      if (is.null(self$weights)) {
        return(data.table::data.table())
      }

      data.table::data.table(
        model = names(self$weights),
        weight = unname(self$weights),
        weight_pct = unname(self$weights) * 100
      )
    },

    #' @description Calculate metrics against actuals
    #' @param actuals Actual values
    #' @return List of metrics
    evaluate = function(actuals) {
      if (length(actuals) != length(self$predictions)) {
        stop("Length of actuals must match predictions")
      }

      residuals <- actuals - self$predictions

      list(
        mape = mean(abs(residuals / actuals), na.rm = TRUE) * 100,
        rmse = sqrt(mean(residuals^2, na.rm = TRUE)),
        mae = mean(abs(residuals), na.rm = TRUE),
        bias = mean(residuals, na.rm = TRUE),
        n = length(actuals)
      )
    },

    #' @description Print result summary
    print = function() {
      cat("CombinationResult\n")
      cat(sprintf("  Combiner: %s\n", self$combiner_name))
      cat(sprintf("  Models: %s\n", paste(self$model_names, collapse = ", ")))
      cat(sprintf("  Bias corrected: %s\n", self$bias_corrected))
      cat(sprintf("  Predictions: %d values\n", length(self$predictions)))
      cat(sprintf("  Elapsed time: %.3f sec\n", self$elapsed_time %||% 0))

      if (!is.null(self$weights)) {
        cat("\nWeights:\n")
        for (i in seq_along(self$weights)) {
          cat(sprintf("  %s: %.4f (%.1f%%)\n",
                      names(self$weights)[i],
                      self$weights[i],
                      self$weights[i] * 100))
        }
      }

      invisible(self)
    }
  )
)
```

### Convenience Functions

```r
#' Create and run a combination workflow
#'
#' @param model_results Named list of model results
#' @param combiner Combiner name or instance
#' @param validation_data Optional validation data
#' @param ... Additional arguments
#' @return CombinationResult
#' @export
combine_forecasts <- function(model_results, combiner = "simple_average",
                              validation_data = NULL, ...) {
  workflow <- CombinationWorkflow$new(combiner = combiner)
  workflow$run(model_results, validation_data, ...)
}
```

### Usage Example

```r
# Model results from different models
model_results <- list(
  lgbm = list(predictions = lgbm_forecast, metrics = lgbm_metrics),
  rf = list(predictions = rf_forecast, metrics = rf_metrics),
  hw = list(predictions = hw_forecast, metrics = hw_metrics)
)

# Create workflow with weighted average combiner
workflow <- CombinationWorkflow$new(
  combiner = "weighted_average",
  config = list(fit_weights = TRUE)
)

# Run with validation data for weight fitting
result <- workflow$run(
  model_results = model_results,
  validation_data = validation_df
)

print(result)
# CombinationResult
#   Combiner: WeightedAverageCombiner
#   Models: lgbm, rf, hw
#   Bias corrected: FALSE
#   Predictions: 168 values
#   Elapsed time: 0.234 sec
#
# Weights:
#   lgbm: 0.4523 (45.2%)
#   rf: 0.3211 (32.1%)
#   hw: 0.2266 (22.7%)

# Evaluate result
metrics <- result$evaluate(actual_values)
# $mape: 2.34
# $rmse: 1234.5

# CV-based weight optimization
result_cv <- workflow$run_with_cv(
  model_results = model_results,
  actuals = actual_values,
  folds = 5
)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with combiner name | Combiner looked up |
| TC-002 | Initialize with combiner instance | Instance stored |
| TC-003 | run() basic | Returns CombinationResult |
| TC-004 | run() with validation data | Weights fitted |
| TC-005 | run() with bias correction | Correction applied |
| TC-006 | run_with_cv() | CV weights averaged |
| TC-007 | extract_forecasts() | Forecasts extracted |
| TC-008 | extract_actuals() from data.table | Vector returned |
| TC-009 | CombinationResult as_data_table() | Correct format |
| TC-010 | CombinationResult evaluate() | Metrics computed |
| TC-011 | combine_forecasts() convenience | Works correctly |
| TC-012 | set_combiner() | Combiner changed |

---

## Dependencies

- PC-032-05: BaseCombiner
- PC-033-05: CombinerRegistry
- PC-035-05: BiasCorrector (optional)

---

## Definition of Done

- [ ] CombinationWorkflow R6 class implemented
- [ ] CombinationResult class implemented
- [ ] CV-based weight optimization
- [ ] Bias correction integration
- [ ] Convenience functions
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Workflow orchestrates the full combination pipeline
- CV for weights uses expanding window (time series appropriate)
- Bias correction is applied after combination
- Consider adding ensemble selection (pick best subset)
- Future: add model weighting by recent performance
