# PC-023-03: Model Infrastructure Tests

**Epic:** [EPIC-03: Model Layer - Infrastructure](../epics/EPIC-03-model-layer-part1.md)
**Task Reference:** T-03.8
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Create comprehensive test suite for the Model Layer Infrastructure (EPIC-03), covering BaseModel, registry, versioning, artifacts, trainer, configuration, and storage utilities with ≥80% coverage target.

---

## Acceptance Criteria

- [ ] Test file created: `tests/testthat/test-models-base.R`
- [ ] Test file created: `tests/testthat/test-models-registry.R`
- [ ] Test file created: `tests/testthat/test-models-versioning.R`
- [ ] Test file created: `tests/testthat/test-models-artifact.R`
- [ ] Test file created: `tests/testthat/test-models-trainer.R`
- [ ] Test file created: `tests/testthat/test-models-config.R`
- [ ] Test file created: `tests/testthat/test-models-storage.R`
- [ ] Mock models for isolated testing
- [ ] Integration tests for full training workflow
- [ ] Overall coverage ≥80% for Model module

---

## Technical Specification

### Test Directory Structure
```
tests/
└── testthat/
   ├── test-models-base.R
   ├── test-models-registry.R
   ├── test-models-versioning.R
   ├── test-models-artifact.R
   ├── test-models-trainer.R
   ├── test-models-config.R
   ├── test-models-storage.R
   └── helper-model-mocks.R
```

### Mock Models

```r
# tests/testthat/helper-model-mocks.R

#' Mock simple model for testing
MockSimpleModel <- R6::R6Class(
  "MockSimpleModel",
  inherit = BaseModel,
  private = list(
    fitted_mean = NULL
  ),
  public = list(
    initialize = function(name = "mock_simple", horizons = 0:8, config = list()) {
      super$initialize(name, horizons, config)
    },

    train = function(X, y, ...) {
      self$validate_features(X)
      private$fitted_mean <- mean(y, na.rm = TRUE)
      private$mark_trained()
      invisible(self)
    },

    predict = function(X, horizon = NULL, ...) {
      private$require_trained()
      rep(private$fitted_mean, nrow(X))
    },

    save = function(path, version) {
      saveRDS(
        list(mean = private$fitted_mean, config = self$config),
        file.path(path, "model.rds")
      )
      invisible(self)
    },

    load = function(path) {
      data <- readRDS(file.path(path, "model.rds"))
      private$fitted_mean <- data$mean
      self$config <- data$config
      private$mark_trained()
      invisible(self)
    }
  )
)


#' Mock multi-horizon model for testing
MockMultiHorizonModel <- R6::R6Class(
  "MockMultiHorizonModel",
  inherit = BaseMultiHorizonModel,
  public = list(
    train = function(X, y, ...) {
      self$validate_features(X)

      for (h in self$horizons) {
        # Store a mock model (just a list with horizon-specific offset)
        self$set_horizon_model(h, list(
          offset = h,
          mean = mean(y, na.rm = TRUE)
        ))
      }

      private$mark_trained()
      invisible(self)
    },

    predict = function(X, horizon = NULL, ...) {
      private$require_trained()

      horizons_to_predict <- if (is.null(horizon)) self$horizons else horizon

      predictions <- sapply(horizons_to_predict, function(h) {
        model <- self$get_horizon_model(h)
        rep(model$mean + model$offset, nrow(X))
      })

      if (length(horizons_to_predict) == 1) {
        as.vector(predictions)
      } else {
        predictions
      }
    },

    save = function(path, version) {
      for (h in self$horizons) {
        saveRDS(
          self$get_horizon_model(h),
          file.path(path, sprintf("model_h%d.rds", h))
        )
      }
      invisible(self)
    },

    load = function(path) {
      for (h in self$horizons) {
        model <- readRDS(file.path(path, sprintf("model_h%d.rds", h)))
        self$set_horizon_model(h, model)
      }
      private$mark_trained()
      invisible(self)
    },

    get_feature_importance = function() {
      data.table::data.table(
        feature = c("feature1", "feature2"),
        importance = c(0.6, 0.4)
      )
    }
  )
)


#' Mock failing model for testing error handling
MockFailingModel <- R6::R6Class(
  "MockFailingModel",
  inherit = BaseModel,
  public = list(
    train = function(X, y, ...) {
      stop("Training failed intentionally")
    },
    predict = function(X, horizon = NULL, ...) {
      stop("Prediction failed intentionally")
    },
    save = function(path, version) {
      stop("Save failed intentionally")
    },
    load = function(path) {
      stop("Load failed intentionally")
    }
  )
)


#' Create sample training data
create_model_test_data <- function(n = 100, seed = 42) {
  set.seed(seed)

  data.table::data.table(
    DataHora = seq(
      as.POSIXct("2024-01-01 00:00:00"),
      by = "hour",
      length.out = n
    ),
    feature1 = rnorm(n, 10, 2),
    feature2 = rnorm(n, 5, 1),
    CargaGlobal = 5000 + rnorm(n, 0, 500),
    area_code = "RJ"
  )
}


#' Create mock storage backend for testing
MockModelStorage <- R6::R6Class(
  "MockModelStorage",
  inherit = StorageBackend,
  private = list(
    files = NULL,
    directories = NULL
  ),
  public = list(
    initialize = function() {
      private$files <- list()
      private$directories <- list()
    },

    exists = function(path) {
      path %in% names(private$files) ||
        path %in% private$directories
    },

    write_rds = function(obj, path) {
      private$files[[path]] <- obj
      dir <- dirname(path)
      if (!dir %in% private$directories) {
        private$directories <- c(private$directories, dir)
      }
    },

    read_rds = function(path) {
      if (!path %in% names(private$files)) {
        stop(sprintf("File not found: %s", path))
      }
      private$files[[path]]
    },

    write_yaml = function(obj, path) {
      private$files[[path]] <- obj
    },

    read_yaml = function(path) {
      private$files[[path]]
    },

    write_json = function(obj, path) {
      private$files[[path]] <- obj
    },

    read_json = function(path) {
      private$files[[path]]
    },

    write_text = function(text, path) {
      private$files[[path]] <- text
    },

    read_text = function(path) {
      private$files[[path]]
    },

    list_directories = function(path) {
      all_dirs <- unique(private$directories)
      sub_dirs <- all_dirs[startsWith(all_dirs, paste0(path, "/"))]
      # Get immediate children only
      children <- sub("^[^/]+/([^/]+).*", "\\1",
                      sub(paste0("^", path, "/"), "", sub_dirs))
      unique(children)
    },

    delete_directory = function(path) {
      # Remove files in directory
      private$files <- private$files[!startsWith(names(private$files), path)]
      # Remove directory
      private$directories <- private$directories[private$directories != path]
    }
  )
)
```

### Base Model Tests

```r
# tests/testthat/test-models-base.R

describe("BaseModel", {
  describe("initialization", {
    it("can be instantiated", {
      model <- BaseModel$new()
      expect_s3_class(model, "BaseModel")
    })

    it("uses class name as default name", {
      model <- BaseModel$new()
      expect_equal(model$name, "BaseModel")
    })

    it("accepts custom name and horizons", {
      model <- BaseModel$new(name = "custom", horizons = 0:4)
      expect_equal(model$name, "custom")
      expect_equal(model$horizons, 0:4)
    })

    it("validates horizons range", {
      expect_error(
        BaseModel$new(horizons = 0:10),
        "upper"
      )
    })
  })

  describe("abstract methods", {
    it("train() throws error", {
      model <- BaseModel$new()
      expect_error(model$train(data.table::data.table(a = 1), 1), "must be implemented")
    })

    it("predict() throws error", {
      model <- BaseModel$new()
      expect_error(model$predict(data.table::data.table(a = 1)), "must be implemented")
    })

    it("save() throws error", {
      model <- BaseModel$new()
      expect_error(model$save("path", "v1"), "must be implemented")
    })

    it("load() throws error", {
      model <- BaseModel$new()
      expect_error(model$load("path"), "must be implemented")
    })
  })

  describe("get_feature_importance", {
    it("returns NULL by default", {
      model <- BaseModel$new()
      expect_null(model$get_feature_importance())
    })
  })

  describe("validate_features", {
    it("accepts data.table", {
      model <- BaseModel$new()
      dt <- data.table::data.table(a = 1:10)
      expect_true(model$validate_features(dt))
    })

    it("rejects empty data.table", {
      model <- BaseModel$new()
      expect_error(model$validate_features(data.table::data.table()))
    })
  })

  describe("supports_horizon", {
    it("returns TRUE for supported horizons", {
      model <- BaseModel$new(horizons = 0:4)
      expect_true(model$supports_horizon(2))
    })

    it("returns FALSE for unsupported horizons", {
      model <- BaseModel$new(horizons = 0:4)
      expect_false(model$supports_horizon(7))
    })
  })
})

describe("MockSimpleModel", {
  it("trains and predicts", {
    model <- MockSimpleModel$new()
    data <- create_model_test_data(50)

    X <- data[, .(feature1, feature2)]
    y <- data$CargaGlobal

    model$train(X, y)

    expect_true(model$is_trained)
    expect_not_null(model$training_date)

    predictions <- model$predict(X)
    expect_length(predictions, 50)
  })

  it("requires training before prediction", {
    model <- MockSimpleModel$new()
    X <- data.table::data.table(feature1 = 1:10, feature2 = 1:10)

    expect_error(model$predict(X), "must be trained")
  })
})

describe("MockMultiHorizonModel", {
  it("stores models per horizon", {
    model <- MockMultiHorizonModel$new(horizons = 0:2)
    data <- create_model_test_data(50)

    X <- data[, .(feature1, feature2)]
    y <- data$CargaGlobal

    model$train(X, y)

    expect_true(model$all_horizons_trained())
    expect_not_null(model$get_horizon_model(0))
    expect_not_null(model$get_horizon_model(1))
    expect_not_null(model$get_horizon_model(2))
  })

  it("returns feature importance", {
    model <- MockMultiHorizonModel$new()
    data <- create_model_test_data(50)
    model$train(data[, .(feature1, feature2)], data$CargaGlobal)

    importance <- model$get_feature_importance()
    expect_s3_class(importance, "data.table")
    expect_true("feature" %in% names(importance))
  })
})
```

### Versioning Tests

```r
# tests/testthat/test-models-versioning.R

describe("ModelVersion", {
  describe("initialization", {
    it("creates with defaults", {
      v <- ModelVersion$new()
      expect_equal(v$major, 0)
      expect_equal(v$minor, 0)
      expect_equal(v$patch, 0)
      expect_not_null(v$timestamp)
    })

    it("parses version string", {
      v <- ModelVersion$new(version_string = "v1.2.3")
      expect_equal(v$major, 1)
      expect_equal(v$minor, 2)
      expect_equal(v$patch, 3)
    })

    it("parses version with timestamp", {
      v <- ModelVersion$new(version_string = "v1.2.3_20250117_143052")
      expect_equal(v$major, 1)
      expect_not_null(v$timestamp)
    })

    it("rejects invalid version string", {
      expect_error(
        ModelVersion$new(version_string = "invalid"),
        "Invalid version"
      )
    })
  })

  describe("to_string", {
    it("includes timestamp by default", {
      v <- ModelVersion$new(major = 1, minor = 2, patch = 3)
      expect_match(v$to_string(), "^v1\\.2\\.3_\\d{8}_\\d{6}$")
    })

    it("short() excludes timestamp", {
      v <- ModelVersion$new(major = 1, minor = 2, patch = 3)
      expect_equal(v$short(), "v1.2.3")
    })
  })

  describe("bumping", {
    it("bump_major resets minor and patch", {
      v <- ModelVersion$new(major = 1, minor = 2, patch = 3)
      v2 <- v$bump_major()
      expect_equal(v2$major, 2)
      expect_equal(v2$minor, 0)
      expect_equal(v2$patch, 0)
    })

    it("bump_minor resets patch", {
      v <- ModelVersion$new(major = 1, minor = 2, patch = 3)
      v2 <- v$bump_minor()
      expect_equal(v2$minor, 3)
      expect_equal(v2$patch, 0)
    })

    it("bump_patch increments only patch", {
      v <- ModelVersion$new(major = 1, minor = 2, patch = 3)
      v2 <- v$bump_patch()
      expect_equal(v2$patch, 4)
    })
  })

  describe("comparison", {
    it("compare returns -1 for less than", {
      v1 <- ModelVersion$new(major = 1)
      v2 <- ModelVersion$new(major = 2)
      expect_equal(v1$compare(v2), -1)
    })

    it("compare returns 1 for greater than", {
      v1 <- ModelVersion$new(major = 2)
      v2 <- ModelVersion$new(major = 1)
      expect_equal(v1$compare(v2), 1)
    })

    it("compare returns 0 for equal", {
      v1 <- ModelVersion$new(major = 1, minor = 2, patch = 3)
      v2 <- ModelVersion$new(major = 1, minor = 2, patch = 3)
      expect_equal(v1$compare(v2), 0)
    })

    it("compares with string", {
      v1 <- ModelVersion$new(major = 1, minor = 2)
      expect_equal(v1$compare("v1.1.0"), 1)
    })
  })
})

describe("version utilities", {
  it("get_latest_version finds newest", {
    versions <- c("v1.0.0", "v2.1.0", "v1.5.0")
    expect_equal(get_latest_version(versions), "v2.1.0")
  })

  it("sort_versions orders correctly", {
    versions <- c("v2.0.0", "v1.0.0", "v1.5.0")
    sorted <- sort_versions(versions)
    expect_equal(sorted, c("v1.0.0", "v1.5.0", "v2.0.0"))
  })
})
```

### Trainer Tests

```r
# tests/testthat/test-models-trainer.R

describe("UniversalTrainer", {
  describe("initialization", {
    it("accepts model only", {
      model <- MockSimpleModel$new()
      trainer <- UniversalTrainer$new(model = model)
      expect_s3_class(trainer, "UniversalTrainer")
    })

    it("accepts model and pipeline", {
      model <- MockSimpleModel$new()
      pipeline <- FeaturePipeline$new()
      trainer <- UniversalTrainer$new(
        model = model,
        feature_pipeline = pipeline
      )
      expect_s3_class(trainer, "UniversalTrainer")
    })
  })

  describe("training", {
    it("trains model and returns artifact", {
      model <- MockSimpleModel$new()
      trainer <- UniversalTrainer$new(model = model)
      data <- create_model_test_data(100)

      artifact <- trainer$train(data, target_col = "CargaGlobal")

      expect_s3_class(artifact, "ModelArtifact")
      expect_true(model$is_trained)
    })

    it("calculates metrics", {
      model <- MockSimpleModel$new()
      trainer <- UniversalTrainer$new(model = model)
      data <- create_model_test_data(100)

      trainer$train(data, target_col = "CargaGlobal")
      metrics <- trainer$get_metrics()

      expect_true("train" %in% names(metrics))
      expect_true("validation" %in% names(metrics))
      expect_true("mape" %in% names(metrics$train))
    })
  })

  describe("cross-validation", {
    it("performs k-fold CV", {
      model <- MockSimpleModel$new()
      trainer <- UniversalTrainer$new(model = model)
      data <- create_model_test_data(200)

      cv_results <- trainer$cross_validate(
        data,
        target_col = "CargaGlobal",
        folds = 3
      )

      expect_equal(length(cv_results$folds), 3)
      expect_true("aggregated" %in% names(cv_results))
      expect_true("mean_mape" %in% names(cv_results$aggregated))
    })
  })

  describe("callbacks", {
    it("fires callbacks", {
      model <- MockSimpleModel$new()
      trainer <- UniversalTrainer$new(model = model)
      data <- create_model_test_data(100)

      callback_fired <- FALSE
      trainer$add_callback("on_train_end", function(args) {
        callback_fired <<- TRUE
      })

      trainer$train(data, target_col = "CargaGlobal")

      expect_true(callback_fired)
    })
  })
})
```

---

## Test Coverage Targets

| Module | Target Coverage |
|--------|-----------------|
| `R/models/base.R` | ≥90% |
| `R/models/registry.R` | ≥90% |
| `R/models/versioning.R` | ≥95% |
| `R/models/artifact.R` | ≥85% |
| `R/models/trainer.R` | ≥80% |
| `R/models/config.R` | ≥85% |
| `R/models/storage.R` | ≥90% |
| **Overall Model Module** | **≥80%** |

---

## Definition of Done

- [ ] All test files created
- [ ] Mock models implemented
- [ ] Mock storage implemented
- [ ] All tests passing
- [ ] Coverage ≥80% verified with covr
- [ ] Integration tests included
- [ ] Edge cases covered
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use `testthat::describe()` and `it()` for BDD-style tests
- Mock models isolate tests from actual implementations
- Mock storage avoids file system dependencies
- Test both happy path and error cases
- Run coverage with: `covr::package_coverage()`
