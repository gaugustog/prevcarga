# PC-020-03: UniversalTrainer

**Epic:** [EPIC-03: Model Layer - Infrastructure](../epics/EPIC-03-model-layer-part1.md)
**Task Reference:** T-03.5
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Implement the `UniversalTrainer` R6 class that provides a unified training workflow for all model types, handling feature transformation, train/validation splitting, metrics calculation, and artifact creation.

---

## Acceptance Criteria

- [ ] Trainer module created in `R/models/trainer.R`
- [ ] `UniversalTrainer` R6 class with training workflow
- [ ] Integration with FeaturePipeline for transformation
- [ ] Train/validation/test data splitting
- [ ] Cross-validation support
- [ ] Metrics calculation during training
- [ ] Automatic artifact creation after training
- [ ] Support for callbacks/hooks

---

## Technical Specification

### File Location
```
R/models/trainer.R
```

### UniversalTrainer Class

```r
#' @title UniversalTrainer
#' @description Unified training workflow for all model types
#'
#' Orchestrates the complete model training workflow including
#' feature transformation, data splitting, training, validation,
#' and artifact creation.
#'
#' @export
UniversalTrainer <- R6::R6Class(
  "UniversalTrainer",
  private = list(
    model = NULL,
    feature_pipeline = NULL,
    config = NULL,
    storage = NULL,
    metrics = NULL,
    callbacks = NULL
  ),
  public = list(
    #' @description Initialize trainer
    #' @param model BaseModel instance
    #' @param feature_pipeline FeaturePipeline instance (optional)
    #' @param config Training configuration
    #' @param storage StorageBackend for artifact saving (optional)
    initialize = function(model,
                          feature_pipeline = NULL,
                          config = list(),
                          storage = NULL) {
      checkmate::assert_class(model, "BaseModel")

      if (!is.null(feature_pipeline)) {
        checkmate::assert_class(feature_pipeline, "FeaturePipeline")
      }

      private$model <- model
      private$feature_pipeline <- feature_pipeline
      private$config <- private$apply_default_config(config)
      private$storage <- storage
      private$metrics <- list()
      private$callbacks <- list()
    },

    #' @description Train model on data
    #' @param data Training data.table
    #' @param target_col Target column name
    #' @param ... Additional arguments passed to model$train()
    #' @return ModelArtifact with trained model
    train = function(data, target_col = "CargaGlobal", ...) {
      checkmate::assert_data_table(data)
      checkmate::assert_choice(target_col, names(data))

      private$fire_callback("on_train_start", list(data = data))

      # Step 1: Apply feature pipeline if present
      features_data <- if (!is.null(private$feature_pipeline)) {
        message("Applying feature pipeline...")
        private$feature_pipeline$fit_transform(data)
      } else {
        data
      }

      # Step 2: Split train/validation
      message("Splitting train/validation data...")
      splits <- private$split_data(features_data)

      # Step 3: Prepare X and y
      feature_names <- setdiff(
        names(splits$train),
        c(target_col, private$config$exclude_cols)
      )

      X_train <- splits$train[, ..feature_names]
      y_train <- splits$train[[target_col]]

      X_val <- splits$val[, ..feature_names]
      y_val <- splits$val[[target_col]]

      # Step 4: Train model
      message(sprintf("Training model: %s", private$model$name))
      private$fire_callback("on_epoch_start", list(epoch = 1))

      private$model$train(X_train, y_train, ...)

      private$fire_callback("on_epoch_end", list(epoch = 1))

      # Step 5: Calculate validation metrics
      message("Calculating validation metrics...")
      val_predictions <- private$model$predict(X_val)
      train_predictions <- private$model$predict(X_train)

      private$metrics <- list(
        train = private$calculate_metrics(y_train, train_predictions),
        validation = private$calculate_metrics(y_val, val_predictions)
      )

      # Log metrics
      message(sprintf(
        "  Train MAPE: %.2f%%, Val MAPE: %.2f%%",
        private$metrics$train$mape,
        private$metrics$validation$mape
      ))

      # Step 6: Create artifact
      artifact <- private$create_artifact(feature_names)

      private$fire_callback("on_train_end", list(
        artifact = artifact,
        metrics = private$metrics
      ))

      artifact
    },

    #' @description Perform cross-validation
    #' @param data Training data.table
    #' @param target_col Target column name
    #' @param folds Number of folds or list of fold indices
    #' @param ... Additional arguments passed to model$train()
    #' @return List with fold metrics and aggregated results
    cross_validate = function(data, target_col = "CargaGlobal",
                              folds = 5, ...) {
      checkmate::assert_data_table(data)

      # Create fold indices
      if (is.numeric(folds) && length(folds) == 1) {
        fold_indices <- private$create_time_series_folds(
          nrow(data),
          n_folds = folds
        )
      } else {
        fold_indices <- folds
      }

      message(sprintf("Running %d-fold cross-validation...", length(fold_indices)))

      fold_results <- list()

      for (i in seq_along(fold_indices)) {
        message(sprintf("\n--- Fold %d/%d ---", i, length(fold_indices)))

        # Get fold data
        train_idx <- fold_indices[[i]]$train
        val_idx <- fold_indices[[i]]$val

        fold_data <- data[train_idx]

        # Apply feature pipeline per fold
        if (!is.null(private$feature_pipeline)) {
          # Clone pipeline for each fold
          fold_pipeline <- private$feature_pipeline$clone_pipeline()
          fold_features <- fold_pipeline$fit_transform(fold_data)
          val_features <- fold_pipeline$transform(data[val_idx])
        } else {
          fold_features <- fold_data
          val_features <- data[val_idx]
        }

        # Prepare X, y
        feature_names <- setdiff(
          names(fold_features),
          c(target_col, private$config$exclude_cols)
        )

        X_train <- fold_features[, ..feature_names]
        y_train <- fold_features[[target_col]]
        X_val <- val_features[, ..feature_names]
        y_val <- val_features[[target_col]]

        # Clone model for fold
        fold_model <- private$model$clone(deep = TRUE)
        fold_model$train(X_train, y_train, ...)

        # Validate
        val_preds <- fold_model$predict(X_val)
        fold_metrics <- private$calculate_metrics(y_val, val_preds)

        fold_results[[i]] <- list(
          fold = i,
          metrics = fold_metrics,
          n_train = length(train_idx),
          n_val = length(val_idx)
        )

        message(sprintf("  Fold %d MAPE: %.2f%%", i, fold_metrics$mape))
      }

      # Aggregate results
      aggregated <- private$aggregate_cv_results(fold_results)

      message(sprintf(
        "\nCV Mean MAPE: %.2f%% (+/- %.2f%%)",
        aggregated$mean_mape,
        aggregated$std_mape
      ))

      list(
        folds = fold_results,
        aggregated = aggregated
      )
    },

    #' @description Get training metrics
    #' @return List with train and validation metrics
    get_metrics = function() {
      private$metrics
    },

    #' @description Register callback
    #' @param event Event name
    #' @param callback Callback function
    #' @return Invisible self
    add_callback = function(event, callback) {
      checkmate::assert_string(event)
      checkmate::assert_function(callback)

      if (is.null(private$callbacks[[event]])) {
        private$callbacks[[event]] <- list()
      }

      private$callbacks[[event]] <- c(private$callbacks[[event]], list(callback))
      invisible(self)
    },

    #' @description Get the trained model
    #' @return BaseModel instance
    get_model = function() {
      private$model
    }
  ),

  private = list(
    #' Apply default configuration
    apply_default_config = function(config) {
      defaults <- list(
        validation_split = 0.15,
        test_split = 0.0,
        shuffle = FALSE,  # Time series: don't shuffle
        seed = 42,
        exclude_cols = c("DataHora", "area_code"),
        metrics = c("mape", "rmse", "mae", "r2")
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    },

    #' Split data into train/validation/test
    split_data = function(data) {
      n <- nrow(data)
      val_size <- floor(n * private$config$validation_split)
      test_size <- floor(n * private$config$test_split)
      train_size <- n - val_size - test_size

      # Time series: use end of data for validation
      list(
        train = data[1:train_size],
        val = data[(train_size + 1):(train_size + val_size)],
        test = if (test_size > 0) {
          data[(train_size + val_size + 1):n]
        } else NULL
      )
    },

    #' Create time series cross-validation folds
    create_time_series_folds = function(n, n_folds = 5) {
      fold_size <- floor(n / (n_folds + 1))

      lapply(seq_len(n_folds), function(i) {
        train_end <- fold_size * i
        val_start <- train_end + 1
        val_end <- min(train_end + fold_size, n)

        list(
          train = 1:train_end,
          val = val_start:val_end
        )
      })
    },

    #' Calculate metrics
    calculate_metrics = function(actual, predicted) {
      residuals <- actual - predicted

      list(
        mape = mean(abs(residuals / actual), na.rm = TRUE) * 100,
        rmse = sqrt(mean(residuals^2, na.rm = TRUE)),
        mae = mean(abs(residuals), na.rm = TRUE),
        r2 = 1 - sum(residuals^2) / sum((actual - mean(actual))^2),
        n = length(actual)
      )
    },

    #' Aggregate cross-validation results
    aggregate_cv_results = function(fold_results) {
      mapes <- sapply(fold_results, function(f) f$metrics$mape)
      rmses <- sapply(fold_results, function(f) f$metrics$rmse)
      maes <- sapply(fold_results, function(f) f$metrics$mae)

      list(
        mean_mape = mean(mapes),
        std_mape = sd(mapes),
        mean_rmse = mean(rmses),
        std_rmse = sd(rmses),
        mean_mae = mean(maes),
        std_mae = sd(maes),
        n_folds = length(fold_results)
      )
    },

    #' Create model artifact
    create_artifact = function(feature_names) {
      version <- ModelVersion$new()

      artifact <- ModelArtifact$new(
        model = private$model,
        model_name = private$model$name,
        version = version,
        feature_names = feature_names,
        config = private$model$config
      )

      artifact$set_metrics(list(
        train_mape = private$metrics$train$mape,
        train_rmse = private$metrics$train$rmse,
        val_mape = private$metrics$validation$mape,
        val_rmse = private$metrics$validation$rmse
      ))

      # Save if storage is configured
      if (!is.null(private$storage)) {
        artifact$save(private$storage)
      }

      artifact
    },

    #' Fire callback
    fire_callback = function(event, args = list()) {
      if (!is.null(private$callbacks[[event]])) {
        for (callback in private$callbacks[[event]]) {
          callback(args)
        }
      }
    }
  )
)
```

### Convenience Functions

```r
#' Create a trainer for a model
#'
#' @param model Model name or BaseModel instance
#' @param feature_pipeline FeaturePipeline (optional)
#' @param config Training configuration
#' @param storage StorageBackend (optional)
#' @return UniversalTrainer instance
#' @export
create_trainer <- function(model, feature_pipeline = NULL,
                           config = list(), storage = NULL) {
  if (is.character(model)) {
    model <- get_model(model, config = config$model_config %||% list())
  }

  UniversalTrainer$new(
    model = model,
    feature_pipeline = feature_pipeline,
    config = config,
    storage = storage
  )
}
```

### Usage Example

```r
# Create model and pipeline
model <- get_model("lightgbm", horizons = 0:8)
pipeline <- feature_pipeline("load_features")$
  add(get_feature_plugin("temporal"))$
  add(get_feature_plugin("lags"))

# Create trainer
trainer <- UniversalTrainer$new(
  model = model,
  feature_pipeline = pipeline,
  config = list(validation_split = 0.2),
  storage = LocalStorageBackend$new("./data")
)

# Add logging callback
trainer$add_callback("on_train_end", function(args) {
  message(sprintf(
    "Training complete. MAPE: %.2f%%",
    args$metrics$validation$mape
  ))
})

# Train
artifact <- trainer$train(train_data, target_col = "CargaGlobal")

# Access metrics
trainer$get_metrics()

# Cross-validation
cv_results <- trainer$cross_validate(
  train_data,
  target_col = "CargaGlobal",
  folds = 5
)
# CV Mean MAPE: 3.45% (+/- 0.32%)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with model only | Works |
| TC-002 | Initialize with pipeline | Pipeline attached |
| TC-003 | train() basic workflow | Returns artifact |
| TC-004 | train() with pipeline | Features transformed |
| TC-005 | train() calculates metrics | Train/val metrics set |
| TC-006 | train() saves artifact | Artifact in storage |
| TC-007 | cross_validate() 5-fold | 5 fold results |
| TC-008 | cross_validate() aggregates | Mean/std computed |
| TC-009 | get_metrics() after train | Returns metrics |
| TC-010 | add_callback() | Callback fires |
| TC-011 | Time series split | Val at end of data |
| TC-012 | create_trainer() by name | Model looked up |

---

## Definition of Done

- [ ] UniversalTrainer class implemented
- [ ] Feature pipeline integration
- [ ] Train/validation splitting
- [ ] Cross-validation support
- [ ] Metrics calculation
- [ ] Callback system
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Time series data uses expanding window CV (no shuffle)
- Consider adding early stopping support
- Callbacks enable logging, checkpointing, and monitoring
- Future: add distributed training support for large datasets
- The trainer is model-agnostic; works with any BaseModel subclass
