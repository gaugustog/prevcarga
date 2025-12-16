# PC-016-03: BaseModel Abstract Class

**Epic:** [EPIC-03: Model Layer - Infrastructure](../epics/EPIC-03-model-layer-part1.md)
**Task Reference:** T-03.1
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Create the `BaseModel` R6 abstract class that defines the contract for all model plugins. This is the foundation of the model plugin architecture for training and prediction.

---

## Acceptance Criteria

- [ ] Base class created in `R/models/base.R`
- [ ] `BaseModel` R6 class with abstract methods
- [ ] `train()` method enforced (throws if not implemented)
- [ ] `predict()` method enforced
- [ ] `save()` and `load()` methods enforced
- [ ] `get_feature_importance()` optional method with default
- [ ] Support for multiple forecast horizons (D+0 to D+8)
- [ ] Configuration support via `config` field
- [ ] roxygen2 documentation for model developers

---

## Technical Specification

### File Location
```
R/models/base.R
```

### Class Definition

```r
#' @title BaseModel
#' @description Abstract base class for forecasting model plugins
#'
#' All model plugins must inherit from this class and implement
#' the required methods: `train()`, `predict()`, `save()`, and `load()`.
#'
#' @details
#' Model plugins implement specific forecasting algorithms. They receive
#' feature matrices and target vectors, train internal models, and produce
#' forecasts for one or more horizons (D+0 through D+8).
#'
#' @section Required Methods:
#' \describe{
#'   \item{train(X, y, ...)}{Train model on features X and target y}
#'   \item{predict(X, ...)}{Generate predictions for new features}
#'   \item{save(path, version)}{Save model artifacts to storage}
#'   \item{load(path)}{Load model artifacts from storage}
#' }
#'
#' @section Optional Methods:
#' \describe{
#'   \item{get_feature_importance()}{Return feature importance scores}
#'   \item{get_params()}{Return model hyperparameters}
#'   \item{set_params(params)}{Update model hyperparameters}
#' }
#'
#' @export
BaseModel <- R6::R6Class(
  "BaseModel",
  public = list(
    #' @field name Model name identifier
    name = NULL,

    #' @field horizons Forecast horizons (default: 0:8 for D+0 to D+8)
    horizons = NULL,

    #' @field config Model configuration list
    config = NULL,

    #' @field is_trained Whether the model has been trained
    is_trained = FALSE,

    #' @field training_date Date when model was last trained
    training_date = NULL,

    #' @description Initialize model
    #' @param name Model name (optional, defaults to class name)
    #' @param horizons Integer vector of forecast horizons
    #' @param config Configuration list
    initialize = function(name = NULL, horizons = 0:8, config = list()) {
      if (is.null(name)) {
        self$name <- class(self)[1]
      } else {
        self$name <- name
      }

      checkmate::assert_integerish(horizons, lower = 0, upper = 8)
      self$horizons <- as.integer(horizons)

      checkmate::assert_list(config)
      self$config <- config
    },

    #' @description Train model on data
    #' @param X Feature matrix or data.table
    #' @param y Target vector
    #' @param ... Additional training arguments
    #' @return Invisible self (for chaining)
    train = function(X, y, ...) {
      stop(sprintf(
        "train() must be implemented by subclass '%s'",
        class(self)[1]
      ))
    },

    #' @description Generate predictions
    #' @param X Feature matrix or data.table for prediction
    #' @param horizon Specific horizon to predict (optional)
    #' @param ... Additional prediction arguments
    #' @return Numeric vector or matrix of predictions
    predict = function(X, horizon = NULL, ...) {
      stop(sprintf(
        "predict() must be implemented by subclass '%s'",
        class(self)[1]
      ))
    },

    #' @description Save model to storage
    #' @param path Base path for model artifacts
    #' @param version ModelVersion instance or version string
    #' @return Invisible self
    save = function(path, version) {
      stop(sprintf(
        "save() must be implemented by subclass '%s'",
        class(self)[1]
      ))
    },

    #' @description Load model from storage
    #' @param path Path to model artifacts
    #' @return Invisible self
    load = function(path) {
      stop(sprintf(
        "load() must be implemented by subclass '%s'",
        class(self)[1]
      ))
    },

    #' @description Get feature importance scores
    #' @return data.table with feature importance or NULL if not supported
    get_feature_importance = function() {
      # Default: not supported
      NULL
    },

    #' @description Get model hyperparameters
    #' @return Named list of parameters
    get_params = function() {
      self$config
    },

    #' @description Update model hyperparameters
    #' @param params Named list of parameters to update
    #' @return Invisible self
    set_params = function(params) {
      checkmate::assert_list(params, names = "named")

      for (key in names(params)) {
        self$config[[key]] <- params[[key]]
      }

      invisible(self)
    },

    #' @description Validate input features
    #' @param X Feature matrix or data.table
    #' @return TRUE if valid, throws error otherwise
    validate_features = function(X) {
      if (is.data.frame(X) || data.table::is.data.table(X)) {
        checkmate::assert_data_frame(X, min.rows = 1)
      } else if (is.matrix(X)) {
        checkmate::assert_matrix(X, min.rows = 1)
      } else {
        stop("X must be a data.frame, data.table, or matrix")
      }
      invisible(TRUE)
    },

    #' @description Check if model supports a specific horizon
    #' @param horizon Horizon to check
    #' @return Logical
    supports_horizon = function(horizon) {
      horizon %in% self$horizons
    },

    #' @description Get model info as list
    #' @return List with model metadata
    info = function() {
      list(
        name = self$name,
        class = class(self)[1],
        horizons = self$horizons,
        is_trained = self$is_trained,
        training_date = self$training_date,
        config = self$config,
        supports_importance = !is.null(self$get_feature_importance())
      )
    },

    #' @description Print model summary
    print = function() {
      cat(sprintf("<%s>\n", class(self)[1]))
      cat(sprintf("  Name: %s\n", self$name))
      cat(sprintf("  Horizons: D+%s\n", paste(self$horizons, collapse = ", D+")))
      cat(sprintf("  Trained: %s\n", self$is_trained))

      if (!is.null(self$training_date)) {
        cat(sprintf("  Training Date: %s\n", self$training_date))
      }

      if (length(self$config) > 0) {
        cat("  Config:\n")
        for (key in names(self$config)) {
          value <- self$config[[key]]
          if (length(value) > 3) {
            value <- paste0("[", paste(head(value, 3), collapse = ", "), ", ...]")
          }
          cat(sprintf("    %s: %s\n", key, value))
        }
      }

      invisible(self)
    }
  ),

  private = list(
    #' Store fitted model object(s)
    fitted_model = NULL,

    #' Store feature names used during training
    feature_names = NULL,

    #' Check that model is trained before prediction
    require_trained = function() {
      if (!self$is_trained) {
        stop(sprintf(
          "Model '%s' must be trained before prediction. Call train() first.",
          self$name
        ))
      }
    },

    #' Mark model as trained
    mark_trained = function() {
      self$is_trained <- TRUE
      self$training_date <- Sys.time()
    }
  )
)
```

### Multi-Horizon Model Base

```r
#' @title BaseMultiHorizonModel
#' @description Base class for models that train separate models per horizon
#'
#' Some algorithms (like gradient boosting) train one model per forecast
#' horizon. This base class provides infrastructure for managing multiple
#' internal models.
#'
#' @export
BaseMultiHorizonModel <- R6::R6Class(
  "BaseMultiHorizonModel",
  inherit = BaseModel,
  private = list(
    # List of fitted models, keyed by horizon
    horizon_models = NULL
  ),
  public = list(
    initialize = function(name = NULL, horizons = 0:8, config = list()) {
      super$initialize(name, horizons, config)
      private$horizon_models <- list()
    },

    #' @description Get model for specific horizon
    #' @param horizon Forecast horizon
    #' @return Fitted model for that horizon or NULL
    get_horizon_model = function(horizon) {
      key <- as.character(horizon)
      private$horizon_models[[key]]
    },

    #' @description Set model for specific horizon
    #' @param horizon Forecast horizon
    #' @param model Fitted model object
    set_horizon_model = function(horizon, model) {
      key <- as.character(horizon)
      private$horizon_models[[key]] <- model
      invisible(self)
    },

    #' @description Check if all horizons have trained models
    #' @return Logical
    all_horizons_trained = function() {
      all(sapply(self$horizons, function(h) {
        !is.null(self$get_horizon_model(h))
      }))
    }
  )
)
```

### Recursive Model Base

```r
#' @title BaseRecursiveModel
#' @description Base class for models using recursive forecasting
#'
#' Time series models (like ARIMA, Holt-Winters) typically produce
#' forecasts recursively from a single fitted model.
#'
#' @export
BaseRecursiveModel <- R6::R6Class(
  "BaseRecursiveModel",
  inherit = BaseModel,
  public = list(
    #' @description Forecast multiple horizons from single model
    #' @param X Feature data (may be unused for pure time series)
    #' @param max_horizon Maximum horizon to forecast
    #' @return Matrix with rows for observations, columns for horizons
    forecast = function(X = NULL, max_horizon = 8) {
      private$require_trained()

      # Subclass implements actual forecasting
      stop("forecast() must be implemented by subclass")
    }
  )
)
```

### Usage Example

```r
# Define a custom model
LightGBMModel <- R6::R6Class(
  "LightGBMModel",
  inherit = BaseMultiHorizonModel,
  public = list(
    train = function(X, y, ...) {
      self$validate_features(X)

      for (horizon in self$horizons) {
        # Train model for each horizon
        model <- lightgbm::lgb.train(
          params = self$config,
          data = lgb.Dataset(as.matrix(X), label = y),
          ...
        )
        self$set_horizon_model(horizon, model)
      }

      private$mark_trained()
      invisible(self)
    },

    predict = function(X, horizon = NULL, ...) {
      private$require_trained()
      self$validate_features(X)

      horizons_to_predict <- if (is.null(horizon)) self$horizons else horizon

      predictions <- sapply(horizons_to_predict, function(h) {
        model <- self$get_horizon_model(h)
        stats::predict(model, as.matrix(X))
      })

      predictions
    },

    save = function(path, version) {
      # Save each horizon model
      for (h in self$horizons) {
        model <- self$get_horizon_model(h)
        lightgbm::lgb.save(model, file.path(path, sprintf("model_h%d.txt", h)))
      }
      invisible(self)
    },

    load = function(path) {
      for (h in self$horizons) {
        model_path <- file.path(path, sprintf("model_h%d.txt", h))
        model <- lightgbm::lgb.load(model_path)
        self$set_horizon_model(h, model)
      }
      private$mark_trained()
      invisible(self)
    },

    get_feature_importance = function() {
      if (!self$is_trained) return(NULL)

      # Aggregate importance across horizons
      importance_list <- lapply(self$horizons, function(h) {
        model <- self$get_horizon_model(h)
        imp <- lightgbm::lgb.importance(model)
        imp$horizon <- h
        imp
      })

      data.table::rbindlist(importance_list)
    }
  )
)

# Register the model
.model_registry$register("lightgbm", LightGBMModel)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Instantiate BaseModel directly | Works (with abstract methods) |
| TC-002 | Call train() on base class | Error: must be implemented |
| TC-003 | Call predict() on base class | Error: must be implemented |
| TC-004 | Call save() on base class | Error: must be implemented |
| TC-005 | Call load() on base class | Error: must be implemented |
| TC-006 | get_feature_importance() default | Returns NULL |
| TC-007 | validate_features() data.table | TRUE |
| TC-008 | validate_features() invalid | Error |
| TC-009 | supports_horizon() in range | TRUE |
| TC-010 | supports_horizon() out of range | FALSE |
| TC-011 | set_params() updates config | Config updated |
| TC-012 | BaseMultiHorizonModel horizon models | Store/retrieve works |
| TC-013 | info() returns metadata | Correct structure |
| TC-014 | print() shows summary | Formatted output |

---

## Definition of Done

- [ ] BaseModel class implemented
- [ ] BaseMultiHorizonModel class implemented
- [ ] BaseRecursiveModel class implemented
- [ ] Abstract method enforcement working
- [ ] Private helpers for trained state
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The base class enforces the model contract via error-throwing default implementations
- Multi-horizon models (like LightGBM) train separate models per horizon
- Recursive models (like Holt-Winters) produce multi-step forecasts from single model
- The `horizons` field defaults to 0:8 for D+0 through D+8 forecasts
- Consider adding `clone_model()` method for ensemble techniques
