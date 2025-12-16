# PC-051-07: ModelComparator

**Epic:** [EPIC-07: Evaluation Layer](../epics/EPIC-07-evaluation-layer.md)
**Task Reference:** T-07.5
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `ModelComparator` R6 class for comparing multiple forecasting models on the same test set, including statistical significance tests for forecast comparison.

---

## Acceptance Criteria

- [ ] Comparator module created in `R/evaluation/comparator.R`
- [ ] `ModelComparator` R6 class with comparison methods
- [ ] Side-by-side metrics comparison
- [ ] Model ranking by metric
- [ ] Diebold-Mariano statistical test
- [ ] Comparison visualization data

---

## Technical Specification

### File Location
```
R/evaluation/comparator.R
```

### ModelComparator R6 Class

```r
#' @title ModelComparator
#' @description Compare multiple forecasting models on the same test set
#'
#' Provides tools for systematic model comparison including metrics
#' calculation, ranking, and statistical significance testing.
#'
#' @export
ModelComparator <- R6::R6Class(
  "ModelComparator",
  private = list(
    comparison_results = NULL,
    test_results = NULL
  ),
  public = list(
    #' @description Initialize comparator
    initialize = function() {
      private$comparison_results <- list()
      private$test_results <- list()
    },

    #' @description Compare multiple models
    #' @param model_results Named list of model results
    #'   Each element should have: model_name, predictions
    #' @param actual Actual values vector
    #' @return data.table with comparison
    compare = function(model_results, actual) {
      checkmate::assert_list(model_results, names = "named")
      checkmate::assert_numeric(actual)

      comparisons <- lapply(names(model_results), function(model_name) {
        result <- model_results[[model_name]]
        predictions <- result$predictions %||% result

        # Calculate metrics
        metrics <- calculate_all_metrics(actual, predictions)

        data.table::data.table(
          model = model_name,
          mape = metrics$mape,
          mae = metrics$mae,
          rmse = metrics$rmse,
          mbe = metrics$mbe,
          smape = metrics$smape,
          max_abs_dev = metrics$max_abs_deviation,
          correlation = metrics$correlation,
          n_obs = metrics$n_observations
        )
      })

      comparison_dt <- data.table::rbindlist(comparisons)

      # Add ranks
      comparison_dt[, mape_rank := rank(mape)]
      comparison_dt[, mae_rank := rank(mae)]
      comparison_dt[, rmse_rank := rank(rmse)]

      private$comparison_results$last <- list(
        comparison = comparison_dt,
        model_results = model_results,
        actual = actual,
        timestamp = Sys.time()
      )

      comparison_dt
    },

    #' @description Build comparison matrix for heatmap
    #' @param metrics Metrics to include
    #' @return Matrix with models as rows, metrics as columns
    build_comparison_matrix = function(metrics = c("mape", "mae", "rmse")) {
      if (is.null(private$comparison_results$last)) {
        stop("No comparison performed yet. Call compare() first.")
      }

      comparison <- private$comparison_results$last$comparison

      mat <- as.matrix(comparison[, ..metrics])
      rownames(mat) <- comparison$model

      mat
    },

    #' @description Rank models by specified metric
    #' @param metric Metric to rank by (mape, mae, rmse)
    #' @param descending If TRUE, higher is better
    #' @return data.table with ranked models
    rank_models = function(metric = "mape", descending = FALSE) {
      if (is.null(private$comparison_results$last)) {
        stop("No comparison performed yet. Call compare() first.")
      }

      comparison <- data.table::copy(private$comparison_results$last$comparison)

      if (descending) {
        data.table::setorderv(comparison, metric, order = -1)
      } else {
        data.table::setorderv(comparison, metric, order = 1)
      }

      comparison[, rank := seq_len(.N)]
      comparison[, best := rank == 1]

      comparison
    },

    #' @description Get best model by metric
    #' @param metric Metric to use
    #' @return Model name
    get_best_model = function(metric = "mape") {
      ranked <- self$rank_models(metric)
      ranked[rank == 1, model]
    },

    #' @description Perform Diebold-Mariano test
    #' @param model_a First model name
    #' @param model_b Second model name
    #' @param h Forecast horizon (for variance correction)
    #' @return List with test results
    diebold_mariano_test = function(model_a, model_b, h = 1) {
      if (is.null(private$comparison_results$last)) {
        stop("No comparison performed yet. Call compare() first.")
      }

      results <- private$comparison_results$last
      actual <- results$actual

      # Get predictions
      pred_a <- results$model_results[[model_a]]$predictions %||%
                results$model_results[[model_a]]
      pred_b <- results$model_results[[model_b]]$predictions %||%
                results$model_results[[model_b]]

      # Calculate errors
      errors_a <- actual - pred_a
      errors_b <- actual - pred_b

      # Loss differential (squared errors)
      d <- errors_a^2 - errors_b^2

      # DM statistic
      n <- length(d)
      d_bar <- mean(d, na.rm = TRUE)

      # Calculate variance with Newey-West adjustment
      gamma_0 <- var(d, na.rm = TRUE)

      # Autocorrelation adjustment for h > 1
      if (h > 1) {
        gamma_sum <- 0
        for (k in 1:(h - 1)) {
          gamma_k <- sum(d[1:(n-k)] * d[(k+1):n], na.rm = TRUE) / n
          gamma_sum <- gamma_sum + 2 * gamma_k
        }
        var_d <- (gamma_0 + gamma_sum) / n
      } else {
        var_d <- gamma_0 / n
      }

      dm_stat <- d_bar / sqrt(var_d)
      p_value <- 2 * (1 - pnorm(abs(dm_stat)))

      result <- list(
        model_a = model_a,
        model_b = model_b,
        dm_statistic = dm_stat,
        p_value = p_value,
        mean_loss_diff = d_bar,
        significant = p_value < 0.05,
        better_model = if (d_bar < 0) model_a else model_b,
        interpretation = if (p_value < 0.05) {
          sprintf("%s is significantly better than %s (p=%.4f)",
                  if (d_bar < 0) model_a else model_b,
                  if (d_bar < 0) model_b else model_a,
                  p_value)
        } else {
          sprintf("No significant difference between %s and %s (p=%.4f)",
                  model_a, model_b, p_value)
        }
      )

      private$test_results[[paste(model_a, model_b, sep = "_vs_")]] <- result

      result
    },

    #' @description Perform all pairwise comparisons
    #' @param h Forecast horizon
    #' @return data.table with all pairwise test results
    all_pairwise_tests = function(h = 1) {
      if (is.null(private$comparison_results$last)) {
        stop("No comparison performed yet. Call compare() first.")
      }

      models <- names(private$comparison_results$last$model_results)
      n_models <- length(models)

      if (n_models < 2) {
        stop("Need at least 2 models for pairwise comparison")
      }

      results <- list()

      for (i in 1:(n_models - 1)) {
        for (j in (i + 1):n_models) {
          test_result <- self$diebold_mariano_test(models[i], models[j], h)

          results[[length(results) + 1]] <- data.table::data.table(
            model_a = test_result$model_a,
            model_b = test_result$model_b,
            dm_statistic = test_result$dm_statistic,
            p_value = test_result$p_value,
            significant = test_result$significant,
            better_model = test_result$better_model
          )
        }
      }

      data.table::rbindlist(results)
    },

    #' @description Compare models by area
    #' @param model_predictions Named list: model -> (area -> predictions)
    #' @param actual_by_area Named list: area -> actual values
    #' @return data.table with metrics by model and area
    compare_by_area = function(model_predictions, actual_by_area) {
      checkmate::assert_list(model_predictions, names = "named")
      checkmate::assert_list(actual_by_area, names = "named")

      results <- list()

      for (model_name in names(model_predictions)) {
        model_preds <- model_predictions[[model_name]]

        for (area in names(actual_by_area)) {
          if (area %in% names(model_preds)) {
            metrics <- calculate_all_metrics(
              actual_by_area[[area]],
              model_preds[[area]]
            )

            results[[length(results) + 1]] <- data.table::data.table(
              model = model_name,
              area_code = area,
              mape = metrics$mape,
              mae = metrics$mae,
              rmse = metrics$rmse
            )
          }
        }
      }

      data.table::rbindlist(results)
    },

    #' @description Get relative improvement
    #' @param baseline_model Baseline model name
    #' @return data.table with improvement percentages
    relative_improvement = function(baseline_model) {
      if (is.null(private$comparison_results$last)) {
        stop("No comparison performed yet. Call compare() first.")
      }

      comparison <- private$comparison_results$last$comparison
      baseline <- comparison[model == baseline_model]

      if (nrow(baseline) == 0) {
        stop(sprintf("Baseline model '%s' not found", baseline_model))
      }

      improvement <- data.table::copy(comparison)
      improvement[, mape_improvement := (baseline$mape - mape) / baseline$mape * 100]
      improvement[, mae_improvement := (baseline$mae - mae) / baseline$mae * 100]
      improvement[, rmse_improvement := (baseline$rmse - rmse) / baseline$rmse * 100]

      improvement
    },

    #' @description Get summary
    #' @return List with comparison summary
    summary = function() {
      if (is.null(private$comparison_results$last)) {
        stop("No comparison performed yet. Call compare() first.")
      }

      comparison <- private$comparison_results$last$comparison

      list(
        n_models = nrow(comparison),
        best_mape = comparison[which.min(mape), .(model, mape)],
        best_mae = comparison[which.min(mae), .(model, mae)],
        best_rmse = comparison[which.min(rmse), .(model, rmse)],
        mape_range = c(min(comparison$mape), max(comparison$mape)),
        all_metrics = comparison
      )
    },

    #' @description Get comparison results
    #' @return List with last comparison
    get_results = function() {
      private$comparison_results$last
    },

    #' @description Print summary
    print = function() {
      cat("ModelComparator\n")

      if (!is.null(private$comparison_results$last)) {
        s <- self$summary()
        cat(sprintf("  Models compared: %d\n", s$n_models))
        cat(sprintf("  Best MAPE: %s (%.2f%%)\n",
                    s$best_mape$model, s$best_mape$mape))
        cat(sprintf("  Best MAE: %s (%.2f)\n",
                    s$best_mae$model, s$best_mae$mae))
      } else {
        cat("  No comparison performed yet\n")
      }

      invisible(self)
    }
  )
)
```

### Convenience Functions

```r
#' Compare multiple models
#'
#' @param model_results Named list of predictions
#' @param actual Actual values
#' @return data.table with comparison
#' @export
compare_models <- function(model_results, actual) {
  comparator <- ModelComparator$new()
  comparator$compare(model_results, actual)
}


#' Perform Diebold-Mariano test between two models
#'
#' @param pred_a Predictions from model A
#' @param pred_b Predictions from model B
#' @param actual Actual values
#' @param h Forecast horizon
#' @return List with test results
#' @export
dm_test <- function(pred_a, pred_b, actual, h = 1) {
  comparator <- ModelComparator$new()
  comparator$compare(list(model_a = pred_a, model_b = pred_b), actual)
  comparator$diebold_mariano_test("model_a", "model_b", h)
}
```

### Usage Example

```r
# Create comparator
comparator <- ModelComparator$new()

# Prepare model predictions
actual <- rnorm(100, mean = 1000, sd = 100)

model_results <- list(
  lgbm = list(
    model_name = "LightGBM",
    predictions = actual + rnorm(100, sd = 30)
  ),
  rf = list(
    model_name = "Random Forest",
    predictions = actual + rnorm(100, sd = 35)
  ),
  hw = list(
    model_name = "Holt-Winters",
    predictions = actual + rnorm(100, sd = 40)
  ),
  naive = list(
    model_name = "Naive",
    predictions = actual + rnorm(100, sd = 50)
  )
)

# Compare all models
comparison <- comparator$compare(model_results, actual)
#     model     mape      mae     rmse   mbe smape max_abs_dev correlation n_obs mape_rank mae_rank rmse_rank
# 1:   lgbm 2.876543 28.76543 35.12345 -1.23 2.845      98.765      0.9876   100         1        1         1
# 2:     rf 3.456789 34.56789 42.34567  2.34 3.423     112.456      0.9765   100         2        2         2
# 3:     hw 4.123456 41.23456 50.45678 -3.45 4.089     125.678      0.9654   100         3        3         3
# 4:  naive 5.678901 56.78901 68.90123  5.67 5.612     156.789      0.9432   100         4        4         4

# Rank by MAPE
ranked <- comparator$rank_models("mape")

# Get best model
best <- comparator$get_best_model("mape")
# "lgbm"

# Statistical test
dm_result <- comparator$diebold_mariano_test("lgbm", "rf")
# $significant: TRUE
# $better_model: "lgbm"
# $interpretation: "lgbm is significantly better than rf (p=0.0234)"

# All pairwise tests
all_tests <- comparator$all_pairwise_tests()

# Relative improvement vs baseline
improvement <- comparator$relative_improvement("naive")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | compare() multiple models | Comparison table |
| TC-002 | compare() single model | Works |
| TC-003 | build_comparison_matrix() | Matrix format |
| TC-004 | rank_models() ascending | Ranked by metric |
| TC-005 | rank_models() descending | Reversed order |
| TC-006 | get_best_model() | Best model name |
| TC-007 | diebold_mariano_test() significant | p < 0.05 |
| TC-008 | diebold_mariano_test() not significant | p >= 0.05 |
| TC-009 | all_pairwise_tests() | All pairs tested |
| TC-010 | compare_by_area() | Metrics by area |
| TC-011 | relative_improvement() | Improvement percentages |

---

## Dependencies

- PC-047-07: Metrics Calculator

---

## Definition of Done

- [ ] ModelComparator R6 class implemented
- [ ] Diebold-Mariano test implemented
- [ ] Model ranking working
- [ ] Comparison matrix generation
- [ ] Area-level comparison
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Diebold-Mariano test assumes squared error loss
- Consider adding other loss functions (absolute)
- Pairwise comparisons grow O(n²) with models
- May want to add bootstrap confidence intervals
