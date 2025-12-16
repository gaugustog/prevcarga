# PC-038-05: Combination Infrastructure Tests

**Epic:** [EPIC-05: Combination Infrastructure](../epics/EPIC-05-combination-layer.md)
**Task Reference:** T-05.7
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Create comprehensive test suite for all EPIC-05 combination infrastructure components, including BaseCombiner, CombinerRegistry, CombinationWorkflow, BiasCorrector, WeightOptimizer, and configuration handling.

---

## Acceptance Criteria

- [ ] Test file created: `tests/testthat/test-combination-base.R`
- [ ] Test file created: `tests/testthat/test-combination-workflow.R`
- [ ] Test file created: `tests/testthat/test-combination-bias.R`
- [ ] Test file created: `tests/testthat/test-combination-optimizer.R`
- [ ] Mock combiners for testing
- [ ] Integration tests for complete workflows
- [ ] ≥80% code coverage for EPIC-05 modules

---

## Technical Specification

### Test Files

```
tests/testthat/
├── test-combination-base.R      # BaseCombiner, CombinerRegistry
├── test-combination-workflow.R  # CombinationWorkflow, CombinationResult
├── test-combination-bias.R      # BiasCorrector implementations
├── test-combination-optimizer.R # WeightOptimizer implementations
├── test-combination-config.R    # Configuration validation
└── helper-mock-combiners.R      # Mock combiner classes for testing
```

### Mock Combiners Helper

```r
# tests/testthat/helper-mock-combiners.R

#' Mock combiner for testing
MockCombiner <- R6::R6Class(
  "MockCombiner",
  inherit = BaseCombiner,
  private = list(
    mock_weights = NULL
  ),
  public = list(
    combine = function(forecasts, weights = NULL, ...) {
      self$validate_forecasts(forecasts)

      # Simple average for mock
      mat <- do.call(cbind, forecasts)
      rowMeans(mat, na.rm = TRUE)
    },

    fit_weights = function(forecasts, actuals, ...) {
      n_models <- length(forecasts)
      private$mock_weights <- rep(1 / n_models, n_models)
      names(private$mock_weights) <- names(forecasts)
      private$weights <- private$mock_weights
      private$model_names <- names(forecasts)
      private$is_fitted <- TRUE
      invisible(self)
    },

    get_weights = function() {
      private$mock_weights
    }
  )
)


#' Mock weighted combiner
MockWeightedCombiner <- R6::R6Class(
  "MockWeightedCombiner",
  inherit = BaseCombiner,
  public = list(
    combine = function(forecasts, weights = NULL, ...) {
      self$validate_forecasts(forecasts)

      w <- weights %||% self$get_weights()
      if (is.null(w)) {
        w <- rep(1 / length(forecasts), length(forecasts))
      }

      # Weighted sum
      result <- rep(0, length(forecasts[[1]]))
      for (i in seq_along(forecasts)) {
        result <- result + w[i] * forecasts[[i]]
      }
      result
    },

    fit_weights = function(forecasts, actuals, ...) {
      # Inverse MSE weights
      mses <- sapply(forecasts, function(f) {
        mean((actuals - f)^2, na.rm = TRUE)
      })
      w <- 1 / (mses + 1e-8)
      w <- w / sum(w)

      private$weights <- w
      private$model_names <- names(forecasts)
      private$is_fitted <- TRUE
      invisible(self)
    }
  )
)


#' Generate test forecasts
create_test_forecasts <- function(n = 100, n_models = 3, seed = 42) {
  set.seed(seed)

  # True signal
  true_signal <- 100 + cumsum(rnorm(n, sd = 2))

  # Create forecasts with different biases/errors
  forecasts <- lapply(seq_len(n_models), function(i) {
    bias <- (i - 2) * 5  # Different biases
    noise_sd <- 5 + i    # Different noise levels
    true_signal + bias + rnorm(n, sd = noise_sd)
  })

  names(forecasts) <- sprintf("model_%d", seq_len(n_models))

  list(
    forecasts = forecasts,
    actuals = true_signal
  )
}
```

### BaseCombiner and Registry Tests

```r
# tests/testthat/test-combination-base.R

describe("BaseCombiner", {
  it("initializes with default config", {
    combiner <- MockCombiner$new(name = "test")

    expect_equal(combiner$name, "test")
    expect_true(combiner$config$auto_normalize)
    expect_false(combiner$is_fitted())
  })

  it("validates forecasts correctly", {
    combiner <- MockCombiner$new()

    # Valid forecasts
    forecasts <- list(a = 1:10, b = 1:10)
    expect_true(combiner$validate_forecasts(forecasts))

    # Invalid: different lengths
    bad_forecasts <- list(a = 1:10, b = 1:5)
    expect_error(combiner$validate_forecasts(bad_forecasts), "same length")

    # Invalid: unnamed
    expect_error(combiner$validate_forecasts(list(1:10)), "named")
  })

  it("validates and normalizes weights", {
    combiner <- MockCombiner$new()

    # Weights that sum to 1
    w1 <- combiner$validate_weights(c(0.5, 0.3, 0.2))
    expect_equal(sum(w1), 1)

    # Weights that need normalization
    w2 <- combiner$validate_weights(c(1, 1, 1))
    expect_equal(sum(w2), 1, tolerance = 1e-6)

    # Negative weights
    expect_warning(
      combiner$validate_weights(c(-0.1, 0.6, 0.5)),
      "set to 0"
    )
  })

  it("sets weights manually", {
    combiner <- MockCombiner$new()

    combiner$set_weights(c(a = 0.6, b = 0.4))

    expect_true(combiner$is_fitted())
    weights <- combiner$get_weights()
    expect_equal(weights["a"], 0.6)
    expect_equal(weights["b"], 0.4)
  })
})


describe("CombinerRegistry", {
  # Fresh registry for each test
  registry <- NULL

  beforeEach({
    registry <<- CombinerRegistry$new()
  })

  it("registers combiners", {
    registry$register("mock", MockCombiner, description = "Test combiner")

    expect_true(registry$has("mock"))
    expect_equal(registry$size(), 1)
  })

  it("gets combiner instances", {
    registry$register("mock", MockCombiner)

    combiner <- registry$get("mock")

    expect_s3_class(combiner, "MockCombiner")
    expect_s3_class(combiner, "BaseCombiner")
  })

  it("fails for unregistered combiners", {
    expect_error(
      registry$get("unknown"),
      "not registered"
    )
  })

  it("lists all combiners", {
    registry$register("a", MockCombiner)
    registry$register("b", MockWeightedCombiner)

    expect_setequal(registry$list_combiners(), c("a", "b"))
  })

  it("provides metadata", {
    registry$register(
      "mock",
      MockCombiner,
      description = "A mock",
      tags = c("test", "simple")
    )

    meta <- registry$get_metadata("mock")

    expect_equal(meta$description, "A mock")
    expect_equal(meta$tags, c("test", "simple"))
  })

  it("finds by tag", {
    registry$register("a", MockCombiner, tags = c("simple"))
    registry$register("b", MockWeightedCombiner, tags = c("weighted"))

    expect_equal(registry$find_by_tag("simple"), "a")
    expect_equal(registry$find_by_tag("weighted"), "b")
  })

  it("unregisters combiners", {
    registry$register("mock", MockCombiner)
    registry$unregister("mock")

    expect_false(registry$has("mock"))
  })
})


describe("Global registry functions", {
  it("get_combiner() works", {
    # Register first
    register_combiner("test_mock", MockCombiner)

    combiner <- get_combiner("test_mock")

    expect_s3_class(combiner, "BaseCombiner")
  })

  it("list_combiners() returns names", {
    names <- list_combiners()

    expect_type(names, "character")
  })

  it("has_combiner() checks existence", {
    expect_type(has_combiner("nonexistent"), "logical")
  })
})
```

### CombinationWorkflow Tests

```r
# tests/testthat/test-combination-workflow.R

describe("CombinationWorkflow", {
  test_data <- create_test_forecasts(n = 100, n_models = 3)

  # Register mock combiner for tests
  beforeAll({
    register_combiner("mock_test", MockCombiner)
    register_combiner("mock_weighted_test", MockWeightedCombiner)
  })

  it("initializes with combiner name", {
    workflow <- CombinationWorkflow$new(combiner = "mock_test")

    expect_s3_class(workflow$get_combiner(), "BaseCombiner")
  })

  it("initializes with combiner instance", {
    combiner <- MockCombiner$new()
    workflow <- CombinationWorkflow$new(combiner = combiner)

    expect_identical(workflow$get_combiner(), combiner)
  })

  it("runs basic combination", {
    workflow <- CombinationWorkflow$new(combiner = "mock_test")

    model_results <- lapply(test_data$forecasts, function(f) {
      list(predictions = f)
    })

    result <- workflow$run(model_results)

    expect_s3_class(result, "CombinationResult")
    expect_length(result$predictions, 100)
  })

  it("fits weights with validation data", {
    workflow <- CombinationWorkflow$new(
      combiner = "mock_weighted_test",
      config = list(fit_weights = TRUE, target_col = "actual")
    )

    model_results <- lapply(test_data$forecasts, function(f) {
      list(predictions = f)
    })

    validation_data <- data.table::data.table(actual = test_data$actuals)

    result <- workflow$run(model_results, validation_data)

    expect_not_null(result$weights)
    expect_length(result$weights, 3)
  })

  it("runs CV-based optimization", {
    workflow <- CombinationWorkflow$new(combiner = "mock_weighted_test")

    model_results <- lapply(test_data$forecasts, function(f) {
      list(predictions = f)
    })

    result <- workflow$run_with_cv(
      model_results,
      actuals = test_data$actuals,
      folds = 3
    )

    expect_not_null(result$weights)
    expect_not_null(result$cv_metrics)
  })

  it("extracts forecasts correctly", {
    workflow <- CombinationWorkflow$new(combiner = "mock_test")

    # From list with predictions field
    results1 <- list(a = list(predictions = 1:10))
    forecasts1 <- workflow$extract_forecasts(results1)
    expect_equal(forecasts1$a, 1:10)

    # From numeric directly
    results2 <- list(a = 1:10)
    forecasts2 <- workflow$extract_forecasts(results2)
    expect_equal(forecasts2$a, 1:10)
  })
})


describe("CombinationResult", {
  it("creates from predictions", {
    result <- CombinationResult$new(
      predictions = 1:10,
      weights = c(a = 0.5, b = 0.5),
      model_names = c("a", "b"),
      combiner_name = "test",
      elapsed_time = 0.1
    )

    expect_length(result$predictions, 10)
    expect_equal(result$combiner_name, "test")
  })

  it("converts to data.table", {
    result <- CombinationResult$new(
      predictions = c(100, 110, 105),
      weights = c(a = 0.6, b = 0.4)
    )

    dt <- result$as_data_table()

    expect_true(is.data.table(dt))
    expect_equal(nrow(dt), 3)
    expect_true("prediction" %in% names(dt))
  })

  it("provides weight summary", {
    result <- CombinationResult$new(
      predictions = 1:10,
      weights = c(lgbm = 0.6, rf = 0.4)
    )

    summary <- result$get_weight_summary()

    expect_true(is.data.table(summary))
    expect_equal(summary[model == "lgbm"]$weight, 0.6)
  })

  it("evaluates against actuals", {
    result <- CombinationResult$new(
      predictions = c(100, 110, 105)
    )

    metrics <- result$evaluate(c(101, 109, 106))

    expect_true("mape" %in% names(metrics))
    expect_true("rmse" %in% names(metrics))
    expect_gt(metrics$mape, 0)
  })
})
```

### BiasCorrector Tests

```r
# tests/testthat/test-combination-bias.R

describe("SimpleBiasCorrector", {
  test_data <- create_test_forecasts(n = 100)

  it("fits additive correction", {
    corrector <- SimpleBiasCorrector$new(config = list(
      correction_type = "additive"
    ))

    # Add known bias
    biased_predictions <- test_data$actuals + 10

    corrector$fit(biased_predictions, test_data$actuals)

    expect_true(corrector$is_fitted())
    factors <- corrector$get_correction_factors()
    expect_equal(factors$type, "additive")
    expect_equal(factors$value, -10, tolerance = 1)
  })

  it("corrects additive bias", {
    corrector <- SimpleBiasCorrector$new(config = list(
      correction_type = "additive"
    ))

    biased <- test_data$actuals + 10
    corrector$fit(biased, test_data$actuals)

    corrected <- corrector$correct(biased)

    # Should be close to actuals now
    expect_equal(mean(corrected), mean(test_data$actuals), tolerance = 1)
  })

  it("fits multiplicative correction", {
    corrector <- SimpleBiasCorrector$new(config = list(
      correction_type = "multiplicative"
    ))

    # Add known multiplicative bias (10% over)
    biased_predictions <- test_data$actuals * 1.1

    corrector$fit(biased_predictions, test_data$actuals)

    factors <- corrector$get_correction_factors()
    expect_equal(factors$type, "multiplicative")
    expect_equal(factors$value, 1/1.1, tolerance = 0.02)
  })

  it("calculates bias metrics", {
    corrector <- SimpleBiasCorrector$new()

    biased <- test_data$actuals + 5
    metrics <- corrector$calculate_bias_metrics(biased, test_data$actuals)

    expect_equal(metrics$mean_bias, -5, tolerance = 0.1)
  })
})


describe("ConditionalBiasCorrector", {
  it("fits by grouping columns", {
    corrector <- ConditionalBiasCorrector$new(
      grouping_cols = "hour",
      config = list(correction_type = "additive")
    )

    # Create data with hour-varying bias
    n <- 240  # 10 days
    features <- data.table::data.table(
      hour = rep(0:23, 10)
    )

    actuals <- 100 + rnorm(n, sd = 5)
    # Add hour-dependent bias
    predictions <- actuals + features$hour * 0.5

    corrector$fit(predictions, actuals, features)

    expect_true(corrector$is_fitted())
    factors <- corrector$get_correction_factors()
    expect_equal(factors$grouping_cols, "hour")
  })

  it("applies conditional correction", {
    corrector <- ConditionalBiasCorrector$new(
      grouping_cols = "hour"
    )

    n <- 240
    features <- data.table::data.table(hour = rep(0:23, 10))
    actuals <- 100 + rnorm(n, sd = 5)
    predictions <- actuals + features$hour * 0.5

    corrector$fit(predictions, actuals, features)
    corrected <- corrector$correct(predictions, features)

    # Bias should be reduced
    original_bias <- mean(abs(predictions - actuals))
    corrected_bias <- mean(abs(corrected - actuals))

    expect_lt(corrected_bias, original_bias)
  })
})


describe("RegressionBiasCorrector", {
  it("fits regression model", {
    corrector <- RegressionBiasCorrector$new()

    n <- 200
    features <- data.table::data.table(
      hour = rep(0:23, ceiling(n/24))[1:n],
      temp = rnorm(n, mean = 25, sd = 5)
    )

    actuals <- 100 + features$hour * 2 + features$temp * 0.5 + rnorm(n, sd = 3)
    predictions <- actuals + rnorm(n, sd = 10)  # Noisy predictions

    corrector$fit(predictions, actuals, features)

    expect_true(corrector$is_fitted())
    meta <- corrector$get_fit_metadata()
    expect_gt(meta$r_squared, 0.5)
  })
})
```

### WeightOptimizer Tests

```r
# tests/testthat/test-combination-optimizer.R

describe("StandardWeightOptimizer", {
  test_data <- create_test_forecasts(n = 100, n_models = 3)

  it("optimizes with inverse_mse", {
    optimizer <- StandardWeightOptimizer$new()

    weights <- optimizer$optimize(
      test_data$forecasts,
      test_data$actuals,
      method = "inverse_mse"
    )

    expect_length(weights, 3)
    expect_equal(sum(weights), 1, tolerance = 1e-6)
    expect_true(all(weights >= 0))
  })

  it("optimizes with ols", {
    optimizer <- StandardWeightOptimizer$new()

    weights <- optimizer$optimize(
      test_data$forecasts,
      test_data$actuals,
      method = "ols"
    )

    expect_length(weights, 3)
    # OLS weights may not sum to 1 without constraints
  })

  it("returns equal weights", {
    optimizer <- StandardWeightOptimizer$new()

    weights <- optimizer$optimize(
      test_data$forecasts,
      test_data$actuals,
      method = "equal"
    )

    expect_equal(weights, rep(1/3, 3), tolerance = 1e-6)
  })

  it("selects best model", {
    optimizer <- StandardWeightOptimizer$new()

    weights <- optimizer$optimize(
      test_data$forecasts,
      test_data$actuals,
      method = "best_model"
    )

    expect_equal(sum(weights), 1)
    expect_equal(sum(weights == 1), 1)  # Exactly one model selected
  })

  it("validates weights", {
    optimizer <- StandardWeightOptimizer$new()

    validated <- optimizer$validate_weights(c(0.5, 0.3, 0.2))
    expect_equal(sum(validated), 1)

    # Negative weights
    validated_neg <- optimizer$validate_weights(c(-0.1, 0.6, 0.5))
    expect_true(all(validated_neg >= 0))
  })
})


describe("ConstrainedWeightOptimizer", {
  test_data <- create_test_forecasts(n = 100)

  it("optimizes with constraints", {
    optimizer <- ConstrainedWeightOptimizer$new()

    weights <- optimizer$optimize(
      test_data$forecasts,
      test_data$actuals,
      method = "constrained_ols"
    )

    expect_equal(sum(weights), 1, tolerance = 1e-4)
    expect_true(all(weights >= 0))
  })

  it("projects to simplex", {
    optimizer <- ConstrainedWeightOptimizer$new()

    # Vector outside simplex
    v <- c(0.5, 0.5, 0.5)
    projected <- optimizer$project_to_simplex(v)

    expect_equal(sum(projected), 1, tolerance = 1e-6)
    expect_true(all(projected >= 0))
  })
})


describe("compute_optimal_weights", {
  test_data <- create_test_forecasts()

  it("works as convenience function", {
    weights <- compute_optimal_weights(
      test_data$forecasts,
      test_data$actuals,
      method = "inverse_mse"
    )

    expect_length(weights, 3)
    expect_named(weights)
  })
})


describe("compare_weight_methods", {
  test_data <- create_test_forecasts()

  it("compares multiple methods", {
    comparison <- compare_weight_methods(
      test_data$forecasts,
      test_data$actuals,
      methods = c("inverse_mse", "equal")
    )

    expect_true(is.data.table(comparison))
    expect_equal(nrow(comparison), 2)
    expect_true("mape" %in% names(comparison))
  })
})
```

### Configuration Tests

```r
# tests/testthat/test-combination-config.R

describe("validate_combination_config", {
  it("validates valid config", {
    config <- list(
      strategy = "weighted_average",
      models = list(
        list(name = "lgbm", enabled = TRUE, weight = 0.6),
        list(name = "rf", enabled = TRUE, weight = 0.4)
      )
    )

    expect_true(validate_combination_config(config))
  })

  it("rejects invalid strategy", {
    config <- list(strategy = "unknown_strategy")

    expect_error(validate_combination_config(config))
  })

  it("rejects model without name", {
    config <- list(
      models = list(
        list(enabled = TRUE)  # Missing name
      )
    )

    expect_error(validate_combination_config(config), "name")
  })
})


describe("apply_combination_defaults", {
  it("applies all defaults to empty config", {
    config <- apply_combination_defaults(list())

    expect_equal(config$strategy, "weighted_average")
    expect_true(config$weight_optimization$enabled)
    expect_false(config$bias_correction$enabled)
  })

  it("preserves existing values", {
    config <- apply_combination_defaults(list(
      strategy = "simple_average"
    ))

    expect_equal(config$strategy, "simple_average")
  })
})


describe("get_enabled_models", {
  it("filters disabled models", {
    config <- list(models = list(
      list(name = "a", enabled = TRUE),
      list(name = "b", enabled = FALSE),
      list(name = "c")  # Default enabled
    ))

    enabled <- get_enabled_models(config)

    expect_setequal(enabled, c("a", "c"))
  })
})


describe("get_horizon_config", {
  it("merges horizon-specific config", {
    config <- list(
      strategy = "weighted_average",
      horizons = list(
        `D+0` = list(strategy = "simple_average")
      )
    )

    d0_config <- get_horizon_config(config, "D+0")

    expect_equal(d0_config$strategy, "simple_average")
  })

  it("returns base config for unknown horizon", {
    config <- list(strategy = "weighted_average")

    d5_config <- get_horizon_config(config, "D+5")

    expect_equal(d5_config$strategy, "weighted_average")
  })
})
```

---

## Test Cases Summary

| Module | Test File | Test Count | Coverage Target |
|--------|-----------|------------|-----------------|
| BaseCombiner | test-combination-base.R | 12 | ≥90% |
| CombinerRegistry | test-combination-base.R | 10 | ≥90% |
| CombinationWorkflow | test-combination-workflow.R | 10 | ≥85% |
| CombinationResult | test-combination-workflow.R | 5 | ≥85% |
| SimpleBiasCorrector | test-combination-bias.R | 5 | ≥85% |
| ConditionalBiasCorrector | test-combination-bias.R | 3 | ≥80% |
| RegressionBiasCorrector | test-combination-bias.R | 2 | ≥80% |
| StandardWeightOptimizer | test-combination-optimizer.R | 6 | ≥85% |
| ConstrainedWeightOptimizer | test-combination-optimizer.R | 3 | ≥80% |
| Configuration | test-combination-config.R | 8 | ≥90% |

---

## Dependencies

- PC-032-05: BaseCombiner
- PC-033-05: CombinerRegistry
- PC-034-05: CombinationWorkflow
- PC-035-05: BiasCorrector
- PC-036-05: WeightOptimizer
- PC-037-05: Configuration

---

## Definition of Done

- [ ] All test files created
- [ ] Mock combiners implemented
- [ ] BaseCombiner tests passing
- [ ] Registry tests passing
- [ ] Workflow tests passing
- [ ] BiasCorrector tests passing
- [ ] WeightOptimizer tests passing
- [ ] Configuration tests passing
- [ ] ≥80% code coverage for EPIC-05
- [ ] CI pipeline passing
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use `testthat::describe()` and `it()` for BDD-style tests
- Mock combiners avoid external dependencies
- Test data generation ensures reproducibility with seeds
- Integration tests verify complete combination workflows
- Consider adding property-based tests for weight validation
