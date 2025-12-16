# PC-031-04: Hierarchical Model Tests

**Epic:** [EPIC-04: Model Layer - Hierarchical Support](../epics/EPIC-04-model-layer-part2.md)
**Task Reference:** T-04.8
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Create comprehensive test suite for all EPIC-04 hierarchical model infrastructure components, including HierarchicalModel, MultiModelManager, ParallelTrainer, InferenceWorkflow, and related classes.

---

## Acceptance Criteria

- [ ] Test file created: `tests/testthat/test-models-hierarchical.R`
- [ ] Test file created: `tests/testthat/test-models-parallel.R`
- [ ] Test file created: `tests/testthat/test-models-inference.R`
- [ ] Mock models for testing
- [ ] Integration tests for complete workflows
- [ ] Edge case coverage
- [ ] ≥80% code coverage for EPIC-04 modules

---

## Technical Specification

### Test Files

```
tests/testthat/
├── test-models-hierarchical.R   # HierarchicalModel, MultiModelManager
├── test-models-parallel.R       # ParallelTrainer
├── test-models-inference.R      # InferenceWorkflow, TwoStageInference, BatchInference
├── test-models-feature-selection.R  # FeatureSelector
└── helper-mock-models.R         # Mock model classes for testing
```

### Mock Models Helper

```r
# tests/testthat/helper-mock-models.R

#' Mock BaseModel for testing
MockModel <- R6::R6Class(
  "MockModel",
  inherit = BaseModel,
  private = list(
    coefficients = NULL,
    train_time = NULL
  ),
  public = list(
    initialize = function(name = "mock_model", config = list()) {
      super$initialize(name = name, config = config)
    },

    train = function(X, y, ...) {
      # Simple linear model for testing
      private$coefficients <- colMeans(X, na.rm = TRUE)
      private$train_time <- Sys.time()
      self$is_trained <- TRUE
      invisible(self)
    },

    predict = function(X, horizon = NULL, ...) {
      if (!self$is_trained) {
        stop("Model not trained")
      }
      # Return mean prediction
      rep(mean(private$coefficients), nrow(X))
    },

    save = function(path, version = NULL) {
      saveRDS(list(
        coefficients = private$coefficients,
        train_time = private$train_time
      ), file.path(path, "model.rds"))
      invisible(path)
    },

    load = function(path) {
      data <- readRDS(file.path(path, "model.rds"))
      private$coefficients <- data$coefficients
      private$train_time <- data$train_time
      self$is_trained <- TRUE
      invisible(self)
    },

    get_feature_importance = function() {
      if (is.null(private$coefficients)) return(NULL)
      data.frame(
        feature = names(private$coefficients),
        importance = abs(private$coefficients) / sum(abs(private$coefficients))
      )
    }
  )
)


#' Mock HierarchicalModel for testing
MockHierarchicalModel <- R6::R6Class(
  "MockHierarchicalModel",
  inherit = HierarchicalModel,
  public = list(
    train_dm = function(X, y, ...) {
      private$dm_model <- lm(y ~ ., data = as.data.frame(X))
      invisible(self)
    },

    train_profiles = function(dm_residuals, X = NULL, y_profiles = NULL, ...) {
      for (i in seq_len(private$n_profiles)) {
        private$profile_models[[i]] <- if (!is.null(y_profiles)) {
          mean(y_profiles[, i], na.rm = TRUE)
        } else {
          1.0
        }
      }
      invisible(self)
    },

    predict_dm = function(X, ...) {
      predict(private$dm_model, newdata = as.data.frame(X))
    },

    predict_profiles = function(dm_forecast, X = NULL, ...) {
      profile_vals <- sapply(private$profile_models, identity)
      matrix(profile_vals, nrow = length(dm_forecast),
             ncol = private$n_profiles, byrow = TRUE)
    }
  )
)


#' Create test data for hierarchical models
create_test_data <- function(n_days = 100, n_profiles = 48) {
  # Create hourly data
  n_obs <- n_days * 24

  data.table::data.table(
    DataHora = seq(
      from = as.POSIXct("2024-01-01 00:00:00"),
      by = "hour",
      length.out = n_obs
    ),
    CargaGlobal = 40000 + rnorm(n_obs, sd = 5000) +
      5000 * sin(seq_len(n_obs) * 2 * pi / 24),
    temp = 25 + rnorm(n_obs, sd = 5),
    hour = rep(0:23, n_days),
    weekday = rep(rep(1:7, each = 24), ceiling(n_days / 7))[1:n_obs]
  )
}


#' Create feature matrix for testing
create_test_features <- function(n = 1000, n_features = 20) {
  features <- data.table::data.table(
    matrix(rnorm(n * n_features), nrow = n, ncol = n_features)
  )
  names(features) <- sprintf("feature_%02d", seq_len(n_features))

  # Add some structured features
  features[, hour := rep(0:23, length.out = n)]
  features[, weekday := rep(1:7, length.out = n)]
  features[, lag_24 := rnorm(n)]
  features[, lag_48 := rnorm(n)]

  features
}
```

### HierarchicalModel Tests

```r
# tests/testthat/test-models-hierarchical.R

describe("HierarchicalModel", {
  test_data <- create_test_data()
  dm_targets <- calculate_daily_dm(test_data)
  profile_targets <- create_profile_targets(test_data)
  features <- create_test_features(n = nrow(dm_targets))

  it("initializes with default settings", {
    model <- MockHierarchicalModel$new(
      name = "test_hier",
      horizons = 0:8
    )

    expect_equal(model$get_n_profiles(), 48)
    expect_equal(model$get_composition_method(), "multiplicative")
    expect_false(model$is_trained)
  })

  it("initializes with additive composition", {
    model <- MockHierarchicalModel$new(
      name = "test_hier",
      composition_method = "additive"
    )

    expect_equal(model$get_composition_method(), "additive")
  })

  it("trains DM and profile models", {
    model <- MockHierarchicalModel$new(name = "test_hier")

    model$train(
      X = features,
      y_dm = dm_targets$daily_mean,
      y_profiles = profile_targets
    )

    expect_true(model$is_trained)
    expect_not_null(model$get_dm_model())
    expect_length(model$get_profile_models(), 48)
  })

  it("predicts with composition", {
    model <- MockHierarchicalModel$new(name = "test_hier")
    model$train(X = features, y_dm = dm_targets$daily_mean,
                y_profiles = profile_targets)

    predictions <- model$predict(features)

    expect_true(is.matrix(predictions))
    expect_equal(ncol(predictions), 48)
    expect_equal(nrow(predictions), nrow(features))
  })

  it("composes multiplicatively", {
    model <- MockHierarchicalModel$new(
      name = "test_hier",
      composition_method = "multiplicative"
    )

    dm <- c(100, 200)
    profiles <- matrix(c(0.5, 0.5, 1.5, 1.5), nrow = 2, ncol = 2)

    result <- model$compose(dm, profiles)

    expect_equal(result[1, 1], 50)   # 100 * 0.5
    expect_equal(result[2, 2], 300)  # 200 * 1.5
  })

  it("composes additively", {
    model <- MockHierarchicalModel$new(
      name = "test_hier",
      composition_method = "additive"
    )

    dm <- c(100, 200)
    profiles <- matrix(c(10, 10, 20, 20), nrow = 2, ncol = 2)

    result <- model$compose(dm, profiles)

    expect_equal(result[1, 1], 110)  # 100 + 10
    expect_equal(result[2, 2], 220)  # 200 + 20
  })

  it("saves and loads correctly", {
    model <- MockHierarchicalModel$new(name = "test_hier")
    model$train(X = features, y_dm = dm_targets$daily_mean,
                y_profiles = profile_targets)

    temp_dir <- tempdir()
    model_path <- file.path(temp_dir, "test_model")

    model$save(model_path)

    # Load into new instance
    loaded_model <- MockHierarchicalModel$new(name = "loaded")
    loaded_model$load(model_path)

    expect_true(loaded_model$is_trained)
    expect_equal(loaded_model$get_n_profiles(), 48)

    # Clean up
    unlink(model_path, recursive = TRUE)
  })

  it("validates composition dimensions", {
    model <- MockHierarchicalModel$new(name = "test_hier")

    dm <- c(100, 200, 300)
    profiles <- matrix(1, nrow = 2, ncol = 48)  # Wrong row count

    expect_error(
      model$compose(dm, profiles),
      "must match"
    )
  })
})


describe("MultiModelManager", {
  features <- create_test_features(n = 500)
  targets <- data.table::data.table(
    y_h001 = rnorm(500, mean = 100),
    y_h002 = rnorm(500, mean = 100),
    y_h003 = rnorm(500, mean = 100)
  )

  it("initializes with model factory", {
    manager <- MultiModelManager$new(
      name = "test_manager",
      model_factory = function() MockModel$new()
    )

    expect_equal(manager$n_models(), 0)
    expect_equal(manager$list_targets(), character(0))
  })

  it("trains multiple models", {
    manager <- MultiModelManager$new(
      name = "test_manager",
      model_factory = function() MockModel$new()
    )

    manager$train_multi(
      X = features,
      y = targets,
      target_cols = c("y_h001", "y_h002", "y_h003")
    )

    expect_true(manager$is_trained)
    expect_equal(manager$n_models(), 3)
    expect_setequal(manager$list_targets(), c("y_h001", "y_h002", "y_h003"))
  })

  it("predicts for multiple targets", {
    manager <- MultiModelManager$new(
      name = "test_manager",
      model_factory = function() MockModel$new()
    )
    manager$train_multi(X = features, y = targets,
                        target_cols = c("y_h001", "y_h002", "y_h003"))

    predictions <- manager$predict_multi(features)

    expect_true(is.data.table(predictions))
    expect_setequal(names(predictions), c("y_h001", "y_h002", "y_h003"))
  })

  it("predicts for subset of targets", {
    manager <- MultiModelManager$new(
      name = "test_manager",
      model_factory = function() MockModel$new()
    )
    manager$train_multi(X = features, y = targets,
                        target_cols = c("y_h001", "y_h002", "y_h003"))

    predictions <- manager$predict_multi(features, target_cols = c("y_h001"))

    expect_setequal(names(predictions), c("y_h001"))
  })

  it("applies per-target feature selection", {
    manager <- MultiModelManager$new(
      name = "test_manager",
      model_factory = function() MockModel$new()
    )

    feature_selections <- list(
      y_h001 = c("feature_01", "feature_02", "hour"),
      y_h002 = c("feature_01", "lag_24"),
      y_h003 = c("feature_01", "feature_02", "lag_24", "lag_48")
    )

    manager$train_multi(
      X = features,
      y = targets,
      target_cols = names(targets),
      feature_selections = feature_selections
    )

    expect_equal(manager$get_feature_selection("y_h001"),
                 c("feature_01", "feature_02", "hour"))
    expect_equal(manager$get_feature_selection("y_h002"),
                 c("feature_01", "lag_24"))
  })

  it("gets specific model", {
    manager <- MultiModelManager$new(
      name = "test_manager",
      model_factory = function() MockModel$new()
    )
    manager$train_multi(X = features, y = targets,
                        target_cols = c("y_h001", "y_h002"))

    model <- manager$get_model("y_h001")

    expect_s3_class(model, "MockModel")
    expect_true(model$is_trained)
  })

  it("saves and loads correctly", {
    manager <- MultiModelManager$new(
      name = "test_manager",
      model_factory = function() MockModel$new()
    )
    manager$train_multi(X = features, y = targets,
                        target_cols = c("y_h001", "y_h002"))

    temp_dir <- file.path(tempdir(), "manager_test")

    manager$save(temp_dir)

    loaded <- MultiModelManager$new(name = "loaded")
    loaded$load(temp_dir)

    expect_equal(loaded$n_models(), 2)
    expect_setequal(loaded$list_targets(), c("y_h001", "y_h002"))

    unlink(temp_dir, recursive = TRUE)
  })
})


describe("create_horizon_model_manager", {
  it("creates manager with 216 targets", {
    manager <- create_horizon_model_manager(
      name = "test_horizon",
      model_type = "mock"  # Would need mock registration
    )

    expect_equal(manager$config$n_targets, 216)
    expect_length(manager$config$target_names, 216)
    expect_equal(manager$config$target_names[1], "y_h001")
    expect_equal(manager$config$target_names[216], "y_h216")
  })
})
```

### ParallelTrainer Tests

```r
# tests/testthat/test-models-parallel.R

describe("ParallelTrainer", {
  features <- create_test_features(n = 200)
  y <- rnorm(200)

  it("initializes with default workers", {
    trainer <- ParallelTrainer$new()

    expect_gt(trainer$get_n_workers(), 0)
    expect_false(trainer$is_setup())
  })

  it("initializes with custom workers", {
    trainer <- ParallelTrainer$new(n_workers = 2)

    expect_equal(trainer$get_n_workers(), 2)
  })

  it("sets up parallel workers", {
    trainer <- ParallelTrainer$new(n_workers = 2)
    trainer$setup()

    expect_true(trainer$is_setup())

    trainer$shutdown()
    expect_false(trainer$is_setup())
  })

  it("trains tasks in parallel", {
    trainer <- ParallelTrainer$new(n_workers = 2)

    tasks <- list(
      list(name = "model_1", model = MockModel$new(), X = features, y = y),
      list(name = "model_2", model = MockModel$new(), X = features, y = y),
      list(name = "model_3", model = MockModel$new(), X = features, y = y)
    )

    results <- trainer$train_parallel(tasks)

    expect_length(results, 3)
    expect_true(all(sapply(results, function(r) r$status == "success")))

    trainer$shutdown()
  })

  it("handles task failures gracefully", {
    trainer <- ParallelTrainer$new(n_workers = 2)

    # Create a task that will fail
    failing_model <- MockModel$new()
    failing_model$train <- function(X, y, ...) {
      stop("Intentional failure")
    }

    tasks <- list(
      list(name = "good", model = MockModel$new(), X = features, y = y),
      list(name = "bad", model = failing_model, X = features, y = y)
    )

    results <- trainer$train_parallel(tasks)

    expect_equal(results[[1]]$status, "success")
    expect_equal(results[[2]]$status, "error")
    expect_true(grepl("Intentional failure", results[[2]]$error))

    trainer$shutdown()
  })

  it("trains with shared data", {
    trainer <- ParallelTrainer$new(n_workers = 2)

    y_list <- list(y, y + 10, y + 20)

    results <- trainer$train_shared_data(
      model_factory = function() MockModel$new(),
      X = features,
      y_list = y_list,
      names = c("m1", "m2", "m3")
    )

    expect_length(results, 3)
    expect_true(all(sapply(results, function(r) r$status == "success")))

    trainer$shutdown()
  })

  it("predicts in parallel", {
    trainer <- ParallelTrainer$new(n_workers = 2)

    # Train models first
    models <- lapply(1:3, function(i) {
      m <- MockModel$new()
      m$train(features, y)
      m
    })

    predictions <- trainer$predict_parallel(models, features)

    expect_length(predictions, 3)
    expect_true(all(sapply(predictions, length) == nrow(features)))

    trainer$shutdown()
  })
})
```

### InferenceWorkflow Tests

```r
# tests/testthat/test-models-inference.R

describe("InferenceWorkflow", {
  features <- create_test_features(n = 100)
  y <- rnorm(100)

  # Create and train a model
  model <- MockModel$new(name = "test_model")
  model$train(features, y)

  # Create artifact
  temp_dir <- tempdir()
  model_path <- file.path(temp_dir, "test_artifact")
  dir.create(model_path, showWarnings = FALSE)

  artifact <- ModelArtifact$new(
    model = model,
    model_name = "test_model",
    version = ModelVersion$new(),
    feature_names = names(features)
  )

  it("initializes with artifact", {
    workflow <- InferenceWorkflow$new(
      model_artifact = artifact
    )

    expect_not_null(workflow$get_model_artifact())
  })

  it("runs basic inference", {
    workflow <- InferenceWorkflow$new(model_artifact = artifact)

    result <- workflow$run(
      anchor_data = features,
      target_date = as.Date("2025-01-18")
    )

    expect_s3_class(result, "InferenceResult")
    expect_equal(result$target_date, as.Date("2025-01-18"))
    expect_length(result$predictions, nrow(features))
  })

  it("builds features with custom builder", {
    feature_builder <- function(data, target_date, ...) {
      data[, custom_feature := 1]
      data
    }

    workflow <- InferenceWorkflow$new(
      model_artifact = artifact,
      feature_builder = feature_builder
    )

    built <- workflow$build_features(features, as.Date("2025-01-18"))

    expect_true("custom_feature" %in% names(built))
  })

  it("caches results when enabled", {
    workflow <- InferenceWorkflow$new(
      model_artifact = artifact,
      config = list(cache_enabled = TRUE)
    )

    result1 <- workflow$run(features, as.Date("2025-01-18"))
    result2 <- workflow$run(features, as.Date("2025-01-18"))

    # Second should be faster (cached)
    expect_lt(result2$inference_time, result1$inference_time * 0.1)
  })

  it("clears cache", {
    workflow <- InferenceWorkflow$new(
      model_artifact = artifact,
      config = list(cache_enabled = TRUE)
    )

    workflow$run(features, as.Date("2025-01-18"))
    workflow$clear_cache()

    # Should run again (not cached)
    result <- workflow$run(features, as.Date("2025-01-18"))
    expect_gt(result$inference_time, 0)
  })
})


describe("InferenceResult", {
  it("creates data.table from vector predictions", {
    result <- InferenceResult$new(
      predictions = c(100, 200, 300),
      target_date = as.Date("2025-01-18"),
      horizon = 1,
      model_name = "test",
      model_version = "v1.0.0",
      inference_time = 0.5
    )

    dt <- result$as_data_table()

    expect_true(is.data.table(dt))
    expect_equal(nrow(dt), 3)
  })

  it("calculates summary statistics", {
    result <- InferenceResult$new(
      predictions = c(100, 200, 300, NA),
      target_date = as.Date("2025-01-18"),
      inference_time = 0.5
    )

    summary <- result$summary()

    expect_equal(summary$n_predictions, 4)
    expect_equal(summary$mean, 200)
    expect_equal(summary$n_na, 1)
  })
})


describe("TwoStageInferenceWorkflow", {
  # Setup would go here with proper artifacts
  # This is a simplified test structure

  it("identifies D+0 vs D+1 correctly", {
    # Test that current date uses D+0 logic
    # Future dates use D+1 logic
  })

  it("tracks verified vs forecasted periods", {
    # Test that period tracking is correct
  })

  it("applies ratio propagation for D+0", {
    # Test ratio calculation
  })
})


describe("BatchInferenceRunner", {
  it("runs batch for multiple areas", {
    # Test multi-area batch
  })

  it("aggregates results correctly", {
    # Test aggregation
  })

  it("handles partial failures", {
    # Test error handling
  })
})
```

### Feature Selection Tests

```r
# tests/testthat/test-models-feature-selection.R

describe("FeatureSelector", {
  features <- create_test_features(n = 500, n_features = 30)
  y <- rnorm(500)

  it("initializes with default config", {
    selector <- FeatureSelector$new()

    expect_equal(length(selector$get_all_selections()), 0)
  })

  it("selects by importance", {
    selector <- FeatureSelector$new()

    # Note: This would need a mock model that returns importance
    selected <- selector$select(
      X = features,
      y = y,
      target = "y_h024",
      method = "correlation"  # Use correlation since mock doesn't have importance
    )

    expect_gt(length(selected), 0)
    expect_true(is.character(selected))
  })

  it("selects horizon-aware features", {
    selector <- FeatureSelector$new()

    # For horizon 24, lag_24 should be valid, but not lag_1
    selected <- selector$select(
      X = features,
      y = y,
      target = "y_h024",
      method = "horizon_aware"
    )

    expect_true("lag_24" %in% selected || !"lag_24" %in% names(features))
  })

  it("stores and retrieves selections", {
    selector <- FeatureSelector$new()

    selector$set_manual_selection("target_1", c("feature_01", "feature_02"))

    retrieved <- selector$get_selection("target_1")

    expect_equal(retrieved, c("feature_01", "feature_02"))
  })

  it("saves and loads selections", {
    selector <- FeatureSelector$new()
    selector$set_manual_selection("t1", c("f1", "f2"))
    selector$set_manual_selection("t2", c("f3", "f4"))

    temp_file <- tempfile(fileext = ".rds")
    selector$save(temp_file)

    loaded <- FeatureSelector$new()
    loaded$load(temp_file)

    expect_equal(loaded$get_selection("t1"), c("f1", "f2"))
    expect_equal(loaded$get_selection("t2"), c("f3", "f4"))

    unlink(temp_file)
  })

  it("generates summary", {
    selector <- FeatureSelector$new()
    selector$set_manual_selection("t1", c("f1", "f2", "f3"))
    selector$set_manual_selection("t2", c("f1", "f2"))

    summary <- selector$summary()

    expect_true(is.data.table(summary))
    expect_equal(nrow(summary), 2)
    expect_equal(summary[target == "t1"]$n_features, 3)
  })
})


describe("HorizonFeatureConfig", {
  config <- HorizonFeatureConfig$new()
  all_features <- c("hour", "weekday", "lag_1", "lag_24", "lag_48",
                    "rolling_24", "temp_forecast")

  it("recommends appropriate features for short horizon", {
    features <- config$get_recommended_features(horizon = 1, all_features)

    expect_true("lag_1" %in% features)
    expect_true("hour" %in% features)
  })

  it("excludes short lags for long horizons", {
    features <- config$get_recommended_features(horizon = 48, all_features)

    expect_false("lag_1" %in% features)
    expect_false("lag_24" %in% features)
    expect_true("lag_48" %in% features)
  })

  it("creates feature matrix", {
    matrix_dt <- config$create_feature_matrix(
      horizons = c(1, 24, 48),
      all_features = all_features
    )

    expect_true(is.data.table(matrix_dt))
    expect_true("horizon" %in% names(matrix_dt))
    expect_true("selected" %in% names(matrix_dt))
  })
})
```

---

## Test Cases Summary

| Module | Test File | Test Count | Coverage Target |
|--------|-----------|------------|-----------------|
| HierarchicalModel | test-models-hierarchical.R | 15 | ≥85% |
| MultiModelManager | test-models-hierarchical.R | 12 | ≥85% |
| ParallelTrainer | test-models-parallel.R | 8 | ≥80% |
| InferenceWorkflow | test-models-inference.R | 10 | ≥85% |
| TwoStageInference | test-models-inference.R | 5 | ≥80% |
| BatchInferenceRunner | test-models-inference.R | 5 | ≥80% |
| FeatureSelector | test-models-feature-selection.R | 10 | ≥85% |

---

## Dependencies

- PC-024-04: HierarchicalModel
- PC-025-04: MultiModelManager
- PC-026-04: ParallelTrainer
- PC-027-04: InferenceWorkflow
- PC-028-04: TwoStageInference
- PC-029-04: BatchInferenceRunner
- PC-030-04: FeatureSelector

---

## Definition of Done

- [ ] All test files created
- [ ] Mock models implemented
- [ ] HierarchicalModel tests passing
- [ ] MultiModelManager tests passing
- [ ] ParallelTrainer tests passing
- [ ] InferenceWorkflow tests passing
- [ ] FeatureSelector tests passing
- [ ] ≥80% code coverage for EPIC-04
- [ ] CI pipeline passing
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use `testthat::describe()` and `it()` for BDD-style tests
- Mock models avoid external dependencies
- Parallel tests may need `skip_on_cran()` for CI stability
- Consider test fixtures for reusable test data
- Integration tests should cover complete workflows
