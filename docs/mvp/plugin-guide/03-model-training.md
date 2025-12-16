# Plugin Guide: Model Training

> **Note:** This documentation will evolve into the project's CONTRIBUTING guide.

## Complexity Tiers

Models range from simple statistical methods to complex multi-stage ML pipelines:

| Tier | Characteristics | Examples |
|------|----------------|----------|
| **Tier 1: Simple** | Single train/predict, no special components | HW, SARIMA, basic regression |
| **Tier 2: Staged Pipeline** | Multi-stage: Data→Features→Train→Inference | Models with feature preprocessing |
| **Tier 3: Custom Loss** | Custom objective/evaluation functions | Asymmetric loss, weighted MAPE |
| **Tier 4: Multi-Model** | N models per area/horizon | 216/432 models per subsystem |
| **Tier 5: Dynamic Features** | Per-horizon feature selection | Lead-hour specific features |

## Migration Decision Tree

```
Does model have multi-stage pipeline?
├─ YES: Create separate stage functions (stage_01_*, stage_02_*, etc.)
│       Implement artifact passing between stages
│       Handle train vs inference mode in each stage
└─ NO: Single train() method sufficient

Does model use custom loss/objective?
├─ YES: Extract as private method in R6 class
│       Ensure closure captures required state
└─ NO: Use standard package loss

Does model have N models per horizon?
├─ YES: Create MultiModelManager pattern
│       Implement parallel training (future package)
│       Per-model feature selection logic
│       Model registry with {target: model} structure
└─ NO: Single model instance

Does inference differ from training?
├─ YES: Separate train() and predict() data flows
│       Store normalization params for inference
│       Handle anchor data for BLF-style corrections
└─ NO: Direct predict() from trained model
```

## BaseModel Interface

```r
#' @title BaseModel
#' @description Abstract base class for forecasting models
BaseModel <- R6::R6Class(
  "BaseModel",
  public = list(
    name = NULL,
    horizons = NULL,
    config = NULL,
    is_trained = FALSE,

    initialize = function(name = NULL, horizons = 0:8, config = list()) {
      self$name <- name
      self$horizons <- horizons
      self$config <- config
    },

    train = function(X, y, ...) {
      stop("train() must be implemented by subclass")
    },

    predict = function(X, ...) {
      stop("predict() must be implemented by subclass")
    },

    save = function(path, version) {
      stop("save() must be implemented by subclass")
    },

    load = function(path) {
      stop("load() must be implemented by subclass")
    }
  )
)
```

## Tier 1: Simple Model

```r
HoltWintersModel <- R6::R6Class(
  "HoltWintersModel",
  inherit = BaseModel,
  private = list(
    model = NULL
  ),
  public = list(
    train = function(X, y, ...) {
      ts_data <- ts(y, frequency = 24)
      private$model <- HoltWinters(ts_data)
      self$is_trained <- TRUE
      invisible(self)
    },

    predict = function(n.ahead = 24, ...) {
      stopifnot(self$is_trained)
      forecast::forecast(private$model, h = n.ahead)$mean
    },

    save = function(path, version) {
      saveRDS(list(model = private$model),
              file.path(path, paste0(self$name, "_", version, ".rds")))
    }
  )
)
```

## Tier 3: Custom Loss Function

```r
LightGBMAsymmetricModel <- R6::R6Class(
  "LightGBMAsymmetricModel",
  inherit = BaseModel,
  private = list(
    model = NULL,

    # Custom asymmetric objective (penalize underestimation)
    custom_objective = function(preds, dtrain) {
      labels <- lightgbm::get_field(dtrain, "label")
      res <- preds - labels

      # Higher penalty for underestimation
      grad <- ifelse(res < 0, 2 * res, res)
      hess <- ifelse(res < 0, 2, 1)

      list(grad = grad, hess = hess)
    },

    # Custom weighted MAPE evaluation
    custom_eval = function(preds, dtrain) {
      labels <- lightgbm::get_field(dtrain, "label")
      mape <- mean(abs((preds - labels) / pmax(abs(labels), 1e-6))) * 100

      list(name = "weighted_mape", value = mape, higher_better = FALSE)
    }
  ),
  public = list(
    train = function(X, y, weights = NULL, ...) {
      lgb_data <- lightgbm::lgb.Dataset(
        data = as.matrix(X),
        label = y,
        weight = weights
      )

      params <- list(
        objective = private$custom_objective,
        metric = "mape",
        num_leaves = self$config$num_leaves %||% 31,
        learning_rate = self$config$learning_rate %||% 0.05
      )

      private$model <- lightgbm::lgb.train(
        params = params,
        data = lgb_data,
        nrounds = self$config$num_iterations %||% 1000,
        eval = private$custom_eval
      )

      self$is_trained <- TRUE
      invisible(self)
    }
  )
)
```

## Tier 4: Multi-Model Pattern

For models that train N separate models per horizon (e.g., 216 models for 9 days × 24 hours):

```r
MultiModelManager <- R6::R6Class(
  "MultiModelManager",
  inherit = BaseModel,
  private = list(
    models = list(),           # {target_col: model}
    feature_selections = list() # {target_col: features}
  ),
  public = list(
    train_multi = function(X, y, target_cols, ...) {
      # Parallel training with future
      future::plan(future::multisession, workers = self$config$n_jobs %||% 4)

      results <- future.apply::future_lapply(seq_along(target_cols), function(i) {
        target_col <- target_cols[i]

        # Train individual model
        model <- randomForest::randomForest(
          x = as.matrix(X),
          y = y[, i],
          ntree = self$config$ntree %||% 100
        )

        list(model = model, target = target_col)
      }, future.seed = TRUE)

      # Store results
      for (result in results) {
        private$models[[result$target]] <- result$model
      }

      self$is_trained <- TRUE
      invisible(self)
    },

    predict_multi = function(X, target_cols = NULL, ...) {
      if (is.null(target_cols)) target_cols <- names(private$models)

      predictions <- matrix(NA, nrow = nrow(X), ncol = length(target_cols))
      colnames(predictions) <- target_cols

      for (i in seq_along(target_cols)) {
        target <- target_cols[i]
        model <- private$models[[target]]
        predictions[, i] <- predict(model, as.matrix(X))
      }

      predictions
    }
  )
)
```

## Tier 5: Dynamic Feature Selection

For models where feature selection depends on the target horizon:

```r
DynamicFeatureModel <- R6::R6Class(
  "DynamicFeatureModel",
  inherit = BaseModel,
  private = list(
    models = list(),
    feature_selections = list(),

    select_features_for_horizon = function(lead_hour, all_features) {
      day_ahead <- ceiling(lead_hour / 24)

      # Origin calendar features
      origin <- grep("^origin_", all_features, value = TRUE)

      # Temperature up to target day
      temp <- grep(paste0("^(tmax|tmin)_[1-", day_ahead, "]$"),
                   all_features, value = TRUE)

      # Target day calendar
      dow <- grep(paste0("^day", day_ahead, "_"), all_features, value = TRUE)

      # This lead hour's features
      hour <- grep(paste0("^h", sprintf("%03d", lead_hour), "_"),
                   all_features, value = TRUE)

      unique(c(origin, temp, dow, hour))
    }
  ),
  public = list(
    train_multi = function(X, y, target_cols, ...) {
      for (i in seq_along(target_cols)) {
        target_col <- target_cols[i]
        lead_hour <- as.integer(gsub("y_h", "", target_col))

        # Dynamic feature selection for this horizon
        selected <- private$select_features_for_horizon(lead_hour, colnames(X))
        private$feature_selections[[target_col]] <- selected

        # Train with selected features only
        X_sub <- X[, selected, drop = FALSE]
        private$models[[target_col]] <- randomForest::randomForest(
          x = as.matrix(X_sub),
          y = y[, i]
        )
      }

      self$is_trained <- TRUE
      invisible(self)
    }
  )
)
```

## Artifact Serialization

```r
# Save all model state
save = function(path, version) {
  artifact <- list(
    model = private$model,
    models = private$models,
    feature_selections = private$feature_selections,
    normalization_params = private$normalization_params,
    metadata = list(
      name = self$name,
      horizons = self$horizons,
      trained_at = Sys.time(),
      config = self$config
    )
  )
  saveRDS(artifact, file.path(path, paste0(self$name, "_", version, ".rds")))
}

# Load model state
load = function(path) {
  artifact <- readRDS(path)
  private$model <- artifact$model
  private$models <- artifact$models
  private$feature_selections <- artifact$feature_selections
  private$normalization_params <- artifact$normalization_params
  self$is_trained <- TRUE
  invisible(self)
}
```

## Reference Files

| Pattern | Reference File | Key Lines |
|---------|----------------|-----------|
| Custom objective | `docs/legacy/lgbm/R/03_training.R` | 211-235 |
| Multi-model training | `docs/legacy/randon-forest/3_treinamento_teste_modelo.R` | 8-129 |
| Per-horizon features | `docs/legacy/randon-forest/3_treinamento_teste_modelo.R` | 31-84 |

## Step-by-Step Migration Process

1. **Identify Complexity Tier** - Review existing code structure
2. **Extract Core Components** - Training function, custom objectives
3. **Create R6 Class** - Inherit from BaseModel
4. **Handle Serialization** - Save/load all required state
5. **Register Plugin** - Add to ModelRegistry
6. **Write Tests** - Unit tests, integration tests

## Next Steps

- [Model Inference](04-model-inference.md) - Inference and prediction
- [Combination Strategies](05-combination.md) - Combining multiple model forecasts
- [Reconciliation](06-reconciliation.md) - Hierarchical reconciliation for SIN
