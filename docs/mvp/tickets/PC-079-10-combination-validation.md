# PC-079-10: Combination Validation

**Epic:** [EPIC-10: Testing & Validation](../epics/EPIC-10-testing-validation.md)
**Task Reference:** T-10.4
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Validate forecast combination strategies, verifying that combined predictions outperform individual models on average, weight optimization converges properly, and bias correction is effective.

---

## Acceptance Criteria

- [ ] Combination validation script in `tests/validation/combination.R`
- [ ] All registered combiners tested
- [ ] Combined predictions outperform individuals (on average)
- [ ] Weight optimization convergence verified
- [ ] Bias correction effectiveness validated
- [ ] Validation report generated

---

## Technical Specification

### File Location
```
tests/validation/combination.R
```

### Combination Validation Framework

```r
#' @title CombinationValidator
#' @description Validates forecast combination strategies
#' @export
CombinationValidator <- R6::R6Class(
  "CombinationValidator",
  private = list(
    config = NULL,
    storage = NULL,
    model_predictions = NULL,
    actuals = NULL,
    results = list()
  ),

  public = list(
    #' @description Initialize validator
    #' @param config ConfigManager instance
    initialize = function(config) {
      private$config <- config
      private$storage <- create_storage_backend(config$get_storage_config())
    },

    #' @description Load predictions from models
    #' @param models Model names
    #' @param start_date Start date
    #' @param end_date End date
    #' @param areas Area codes
    load_predictions = function(models, start_date, end_date, areas = NULL) {
      areas <- areas %||% private$config$get_areas()

      private$model_predictions <- list()

      for (model in models) {
        cat(sprintf("Loading predictions for %s...\n", model))

        preds <- load_model_predictions(
          model = model,
          start_date = start_date,
          end_date = end_date,
          areas = areas,
          storage = private$storage
        )

        private$model_predictions[[model]] <- preds
      }

      # Load actuals
      private$actuals <- load_actuals(
        start_date = start_date,
        end_date = end_date,
        areas = areas,
        storage = private$storage
      )

      cat(sprintf("Loaded %d predictions from %d models\n",
                  nrow(preds), length(models)))
    },

    #' @description Validate all registered combiners
    #' @return List of validation results
    validate_all_combiners = function() {
      combiners <- list_combiners()

      cat("\n=== Combination Strategy Validation ===\n\n")

      for (combiner_name in combiners) {
        cat(sprintf("Testing combiner: %s\n", combiner_name))

        tryCatch({
          result <- self$validate_combiner(combiner_name)
          private$results[[combiner_name]] <- result

          status <- if (result$outperforms_individuals) "\u2713" else "\u2717"
          cat(sprintf("  %s MAPE: %.2f%% (vs best individual: %.2f%%)\n",
                      status,
                      result$combined_mape * 100,
                      result$best_individual_mape * 100))

        }, error = function(e) {
          cat(sprintf("  \u2717 ERROR: %s\n", conditionMessage(e)))
          private$results[[combiner_name]] <- list(
            combiner = combiner_name,
            error = conditionMessage(e)
          )
        })
      }

      private$results
    },

    #' @description Validate single combiner
    #' @param combiner_name Combiner name
    #' @return Validation result
    validate_combiner = function(combiner_name) {
      combiner <- get_combiner(combiner_name)
      calculator <- MetricsCalculator$new()

      # Calculate individual model metrics first
      individual_metrics <- list()

      for (model in names(private$model_predictions)) {
        preds <- private$model_predictions[[model]]
        merged <- merge(preds, private$actuals, by = c("datetime", "area_code"))

        metrics <- calculator$calculate_all(merged$actual, merged$prediction)
        individual_metrics[[model]] <- metrics
      }

      # Get best individual
      best_mape <- min(sapply(individual_metrics, function(m) m$mape))
      best_model <- names(which.min(sapply(individual_metrics, function(m) m$mape)))

      # Combine forecasts
      combined <- combiner$combine(private$model_predictions)

      # Calculate combined metrics
      merged_combined <- merge(
        combined$predictions,
        private$actuals,
        by = c("datetime", "area_code")
      )

      combined_metrics <- calculator$calculate_all(
        merged_combined$actual,
        merged_combined$prediction
      )

      # Check if combined outperforms
      outperforms <- combined_metrics$mape < best_mape
      improvement <- (best_mape - combined_metrics$mape) / best_mape

      list(
        combiner = combiner_name,
        combined_mape = combined_metrics$mape,
        combined_mae = combined_metrics$mae,
        combined_rmse = combined_metrics$rmse,
        best_individual_model = best_model,
        best_individual_mape = best_mape,
        outperforms_individuals = outperforms,
        improvement = improvement,
        weights = combined$weights,
        individual_metrics = individual_metrics
      )
    },

    #' @description Validate weight optimization convergence
    #' @param combiner_name Combiner name (must support optimization)
    #' @param n_runs Number of optimization runs
    #' @return Convergence analysis
    validate_weight_convergence = function(combiner_name, n_runs = 10) {
      cat(sprintf("\nValidating weight convergence for %s...\n", combiner_name))

      combiner <- get_combiner(combiner_name)

      if (!("optimize_weights" %in% names(combiner))) {
        return(list(
          combiner = combiner_name,
          supports_optimization = FALSE
        ))
      }

      # Run optimization multiple times
      weight_runs <- list()

      for (i in seq_len(n_runs)) {
        combined <- combiner$combine(private$model_predictions)
        weight_runs[[i]] <- combined$weights
      }

      # Analyze convergence
      weight_matrix <- do.call(rbind, weight_runs)
      weight_means <- colMeans(weight_matrix)
      weight_sds <- apply(weight_matrix, 2, sd)
      cv <- weight_sds / weight_means

      # Check convergence (CV < 0.1 is good)
      converged <- all(cv < 0.1)

      list(
        combiner = combiner_name,
        supports_optimization = TRUE,
        converged = converged,
        n_runs = n_runs,
        weight_means = weight_means,
        weight_sds = weight_sds,
        coefficient_of_variation = cv,
        all_runs = weight_matrix
      )
    },

    #' @description Validate bias correction effectiveness
    #' @return Bias correction analysis
    validate_bias_correction = function() {
      cat("\nValidating bias correction...\n")

      # Get combiner with bias correction
      combiner <- get_combiner("inverse_mape")  # Usually supports bias correction

      # Combine without bias correction
      combined_no_bias <- combiner$combine(
        private$model_predictions,
        bias_correction = FALSE
      )

      # Combine with bias correction
      combined_with_bias <- combiner$combine(
        private$model_predictions,
        bias_correction = TRUE
      )

      calculator <- MetricsCalculator$new()

      # Calculate MBE (Mean Bias Error) for both
      merged_no_bias <- merge(
        combined_no_bias$predictions,
        private$actuals,
        by = c("datetime", "area_code")
      )

      merged_with_bias <- merge(
        combined_with_bias$predictions,
        private$actuals,
        by = c("datetime", "area_code")
      )

      mbe_no_bias <- calculator$mbe(
        merged_no_bias$actual,
        merged_no_bias$prediction
      )

      mbe_with_bias <- calculator$mbe(
        merged_with_bias$actual,
        merged_with_bias$prediction
      )

      # Bias should be reduced
      bias_reduced <- abs(mbe_with_bias) < abs(mbe_no_bias)
      bias_reduction <- (abs(mbe_no_bias) - abs(mbe_with_bias)) / abs(mbe_no_bias)

      list(
        mbe_without_correction = mbe_no_bias,
        mbe_with_correction = mbe_with_bias,
        bias_reduced = bias_reduced,
        bias_reduction_pct = bias_reduction * 100,
        effective = bias_reduced && bias_reduction > 0.1  # >10% improvement
      )
    },

    #' @description Run full validation
    #' @return Complete validation report
    run_full_validation = function() {
      cat("\n" |> rep(60) |> paste(collapse = "="))
      cat("\n  Forecast Combination Validation\n")
      cat("=" |> rep(60) |> paste(collapse = ""))
      cat("\n\n")

      # Validate all combiners
      combiner_results <- self$validate_all_combiners()

      # Weight convergence for optimizing combiners
      convergence_results <- list()
      for (combiner_name in c("optimal", "constrained_optimal")) {
        if (combiner_name %in% list_combiners()) {
          conv <- self$validate_weight_convergence(combiner_name)
          convergence_results[[combiner_name]] <- conv
        }
      }

      # Bias correction
      bias_results <- self$validate_bias_correction()

      list(
        combiner_validation = combiner_results,
        weight_convergence = convergence_results,
        bias_correction = bias_results,
        timestamp = Sys.time()
      )
    },

    #' @description Generate summary
    #' @return data.table summary
    summary = function() {
      data.table::rbindlist(lapply(names(private$results), function(name) {
        r <- private$results[[name]]

        if (!is.null(r$error)) {
          return(data.table::data.table(
            combiner = name,
            mape = NA_real_,
            outperforms = NA,
            improvement = NA_real_,
            error = r$error
          ))
        }

        data.table::data.table(
          combiner = name,
          mape = r$combined_mape,
          outperforms = r$outperforms_individuals,
          improvement = r$improvement,
          error = NA_character_
        )
      }))
    },

    #' @description Print results
    print = function() {
      cat("\nCombination Validation Summary\n")
      cat("==============================\n\n")

      summary_dt <- self$summary()
      print(summary_dt)

      n_outperform <- sum(summary_dt$outperforms, na.rm = TRUE)
      n_total <- sum(!is.na(summary_dt$outperforms))

      cat(sprintf("\n%d/%d combiners outperform best individual model\n",
                  n_outperform, n_total))

      invisible(self)
    }
  )
)
```

### Validation Runner Script

```r
#!/usr/bin/env Rscript
# tests/validation/combination.R

library(prevcargaons)

# Load config
config <- load_config("config/config.yaml")

# Initialize validator
validator <- CombinationValidator$new(config)

# Load predictions
validator$load_predictions(
  models = c("lgbm", "rf", "hw"),
  start_date = "2024-01-01",
  end_date = "2024-03-31",
  areas = config$get_areas()
)

# Run full validation
results <- validator$run_full_validation()

# Print summary
print(validator)

# Check results
summary_dt <- validator$summary()

# Pass criteria:
# 1. At least one combiner outperforms individuals
# 2. Weight optimization converges
# 3. Bias correction is effective

pass_combiners <- any(summary_dt$outperforms, na.rm = TRUE)
pass_convergence <- all(sapply(results$weight_convergence, function(r) {
  r$supports_optimization == FALSE || r$converged
}))
pass_bias <- results$bias_correction$effective

if (pass_combiners && pass_convergence && pass_bias) {
  cat("\n\u2713 Combination validation PASSED\n")
  quit(status = 0)
} else {
  cat("\n\u2717 Combination validation FAILED\n")
  if (!pass_combiners) cat("  - No combiner outperforms individuals\n")
  if (!pass_convergence) cat("  - Weight optimization not converging\n")
  if (!pass_bias) cat("  - Bias correction not effective\n")
  quit(status = 1)
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | simple_average combiner | Metrics calculated |
| TC-002 | inverse_mape combiner | Metrics calculated |
| TC-003 | optimal combiner | Weights optimized |
| TC-004 | Combined outperforms | At least one combiner |
| TC-005 | Weight convergence | CV < 10% |
| TC-006 | Bias correction reduces MBE | >10% reduction |
| TC-007 | Invalid combiner | Error handled |
| TC-008 | Empty predictions | Error handled |
| TC-009 | Single model | Degrades gracefully |
| TC-010 | Summary generation | Table produced |

---

## Dependencies

- PC-032-05: BaseCombiner
- PC-033-05: CombinerRegistry
- PC-035-05: Bias Correction
- PC-036-05: Weight Optimizer
- PC-047-07: MetricsCalculator

---

## Definition of Done

- [ ] CombinationValidator class implemented
- [ ] All registered combiners tested
- [ ] Outperformance verified
- [ ] Weight convergence validated
- [ ] Bias correction effectiveness confirmed
- [ ] Summary report generated
- [ ] roxygen2 documentation complete
- [ ] Validation passing
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Combined forecasts should outperform on average
- Weight convergence is critical for reproducibility
- Bias correction may not always improve MAPE
- Consider model diversity in combination
