# PC-025-04: MultiModelManager Pattern

**Epic:** [EPIC-04: Model Layer - Hierarchical Support](../epics/EPIC-04-model-layer-part2.md)
**Task Reference:** T-04.2
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `MultiModelManager` R6 class that manages multiple models for multi-target forecasting scenarios, such as one model per horizon (216 models for 9 days × 24 hours) or one model per area.

---

## Acceptance Criteria

- [ ] Multi-model manager module created in `R/models/multi_model.R`
- [ ] `MultiModelManager` R6 class with model storage
- [ ] Support for training multiple models on different targets
- [ ] Support for per-target feature selection
- [ ] Prediction across all or selected targets
- [ ] Model retrieval by target name
- [ ] Integration with parallel training infrastructure

---

## Technical Specification

### File Location
```
R/models/multi_model.R
```

### MultiModelManager R6 Class

```r
#' @title MultiModelManager
#' @description Manages multiple models for multi-target forecasting
#'
#' Supports patterns like:
#' - One model per forecast horizon (216 models for D+0 to D+8 × 24 hours)
#' - One model per area (4-5 models for SIN subsystems)
#' - One model per target column
#'
#' @export
MultiModelManager <- R6::R6Class(
  "MultiModelManager",
  inherit = BaseModel,
  private = list(
    models = NULL,            # Named list: {target: model}
    feature_selections = NULL, # Named list: {target: feature_names}
    model_factory = NULL,     # Function to create new models
    target_metadata = NULL    # data.table with target info
  ),
  public = list(
    #' @description Initialize multi-model manager
    #' @param name Manager name
    #' @param model_factory Function that creates a new model instance
    #' @param config Additional configuration
    initialize = function(name,
                          model_factory = NULL,
                          config = list()) {
      checkmate::assert_string(name, min.chars = 1)
      checkmate::assert_function(model_factory, null.ok = TRUE)

      super$initialize(name = name, config = config)

      private$models <- list()
      private$feature_selections <- list()
      private$model_factory <- model_factory
      private$target_metadata <- data.table::data.table(
        target = character(),
        n_features = integer(),
        trained_at = as.POSIXct(character()),
        train_samples = integer()
      )
    },

    #' @description Train models for multiple targets
    #' @param X Feature matrix (data.table)
    #' @param y Target matrix or data.table with target columns
    #' @param target_cols Character vector of target column names
    #' @param feature_selections Named list of feature names per target (optional)
    #' @param ... Additional arguments passed to model$train()
    #' @return Invisible self
    train_multi = function(X, y, target_cols,
                           feature_selections = NULL, ...) {
      checkmate::assert_data_table(X)
      checkmate::assert_character(target_cols, min.len = 1)

      # Validate y contains target columns
      if (is.data.table(y) || is.data.frame(y)) {
        y_dt <- as.data.table(y)
        for (col in target_cols) {
          if (!col %in% names(y_dt)) {
            stop(sprintf("Target column '%s' not found in y", col))
          }
        }
      } else if (is.matrix(y)) {
        if (ncol(y) != length(target_cols)) {
          stop("Number of y columns must match target_cols length")
        }
        y_dt <- as.data.table(y)
        names(y_dt) <- target_cols
      } else {
        stop("y must be a data.table, data.frame, or matrix")
      }

      message(sprintf("Training %d models for targets...", length(target_cols)))

      for (i in seq_along(target_cols)) {
        target <- target_cols[i]
        message(sprintf("  [%d/%d] Training model for: %s",
                        i, length(target_cols), target))

        # Get feature selection for this target
        if (!is.null(feature_selections) && target %in% names(feature_selections)) {
          target_features <- feature_selections[[target]]
          private$feature_selections[[target]] <- target_features
        } else {
          target_features <- names(X)
          private$feature_selections[[target]] <- target_features
        }

        # Create model instance
        model <- if (!is.null(private$model_factory)) {
          private$model_factory()
        } else {
          stop("model_factory must be set to create models")
        }

        # Train model
        X_subset <- X[, ..target_features]
        y_target <- y_dt[[target]]

        model$train(X_subset, y_target, ...)

        # Store model
        private$models[[target]] <- model

        # Update metadata
        private$target_metadata <- rbind(
          private$target_metadata[target != target],
          data.table::data.table(
            target = target,
            n_features = length(target_features),
            trained_at = Sys.time(),
            train_samples = nrow(X)
          )
        )
      }

      self$is_trained <- TRUE
      message(sprintf("Completed training %d models", length(target_cols)))

      invisible(self)
    },

    #' @description Standard train interface (delegates to train_multi)
    #' @param X Feature matrix
    #' @param y Target vector or matrix
    #' @param ... Additional arguments
    #' @return Invisible self
    train = function(X, y, ...) {
      if (is.null(ncol(y)) || ncol(y) == 1) {
        # Single target - use default name
        target_cols <- "target"
        y_dt <- data.table::data.table(target = as.vector(y))
      } else {
        # Multiple targets
        target_cols <- if (!is.null(colnames(y))) {
          colnames(y)
        } else {
          sprintf("target_%03d", seq_len(ncol(y)))
        }
        y_dt <- as.data.table(y)
        names(y_dt) <- target_cols
      }

      self$train_multi(X, y_dt, target_cols, ...)
    },

    #' @description Predict for multiple targets
    #' @param X Feature matrix (data.table)
    #' @param target_cols Target columns to predict (NULL = all)
    #' @param ... Additional arguments passed to model$predict()
    #' @return data.table with predictions for each target
    predict_multi = function(X, target_cols = NULL, ...) {
      checkmate::assert_data_table(X)

      if (!self$is_trained) {
        stop("Models must be trained before prediction")
      }

      # Default to all targets
      if (is.null(target_cols)) {
        target_cols <- names(private$models)
      }

      results <- data.table::data.table(.rows = nrow(X))

      for (target in target_cols) {
        if (!target %in% names(private$models)) {
          warning(sprintf("No model for target '%s', skipping", target))
          next
        }

        model <- private$models[[target]]
        features <- private$feature_selections[[target]]

        # Validate features exist
        missing_features <- setdiff(features, names(X))
        if (length(missing_features) > 0) {
          stop(sprintf(
            "Missing features for target '%s': %s",
            target, paste(missing_features, collapse = ", ")
          ))
        }

        X_subset <- X[, ..features]
        predictions <- model$predict(X_subset, ...)

        results[, (target) := predictions]
      }

      results
    },

    #' @description Standard predict interface
    #' @param X Feature matrix
    #' @param horizon Not used (for interface compatibility)
    #' @param ... Additional arguments
    #' @return Matrix of predictions
    predict = function(X, horizon = NULL, ...) {
      result <- self$predict_multi(as.data.table(X), ...)
      as.matrix(result)
    },

    #' @description Get model for a specific target
    #' @param target Target name
    #' @return Model instance or NULL
    get_model = function(target) {
      private$models[[target]]
    },

    #' @description Set model for a specific target
    #' @param target Target name
    #' @param model Model instance
    #' @return Invisible self
    set_model = function(target, model) {
      checkmate::assert_string(target)
      checkmate::assert_class(model, "BaseModel")
      private$models[[target]] <- model
      invisible(self)
    },

    #' @description List all target names
    #' @return Character vector
    list_targets = function() {
      names(private$models)
    },

    #' @description Get number of models
    #' @return Integer
    n_models = function() {
      length(private$models)
    },

    #' @description Get feature selection for a target
    #' @param target Target name
    #' @return Character vector of feature names
    get_feature_selection = function(target) {
      private$feature_selections[[target]]
    },

    #' @description Set feature selection for a target
    #' @param target Target name
    #' @param features Character vector of feature names
    #' @return Invisible self
    set_feature_selection = function(target, features) {
      checkmate::assert_string(target)
      checkmate::assert_character(features)
      private$feature_selections[[target]] <- features
      invisible(self)
    },

    #' @description Get target metadata
    #' @return data.table with target information
    get_target_metadata = function() {
      data.table::copy(private$target_metadata)
    },

    #' @description Remove model for a target
    #' @param target Target name
    #' @return Invisible self
    remove_model = function(target) {
      private$models[[target]] <- NULL
      private$feature_selections[[target]] <- NULL
      private$target_metadata <- private$target_metadata[target != target]
      invisible(self)
    },

    #' @description Save multi-model manager
    #' @param path Base path for saving
    #' @param version ModelVersion object (optional)
    #' @return Invisible path
    save = function(path, version = NULL) {
      checkmate::assert_string(path)

      if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
      }

      # Save models directory
      models_dir <- file.path(path, "models")
      dir.create(models_dir, showWarnings = FALSE)

      for (target in names(private$models)) {
        model_path <- file.path(models_dir, paste0(target, ".rds"))
        saveRDS(private$models[[target]], model_path)
      }

      # Save feature selections
      saveRDS(private$feature_selections,
              file.path(path, "feature_selections.rds"))

      # Save metadata
      metadata <- list(
        name = self$name,
        horizons = self$horizons,
        config = self$config,
        is_trained = self$is_trained,
        targets = names(private$models),
        n_models = length(private$models),
        version = if (!is.null(version)) version$to_string() else NULL,
        saved_at = Sys.time()
      )
      yaml::write_yaml(metadata, file.path(path, "metadata.yaml"))

      # Save target metadata
      data.table::fwrite(private$target_metadata,
                         file.path(path, "target_metadata.csv"))

      message(sprintf("Saved %d models to: %s", length(private$models), path))
      invisible(path)
    },

    #' @description Load multi-model manager
    #' @param path Path to load from
    #' @return Invisible self
    load = function(path) {
      checkmate::assert_directory_exists(path)

      # Load metadata first
      metadata_path <- file.path(path, "metadata.yaml")
      if (file.exists(metadata_path)) {
        metadata <- yaml::read_yaml(metadata_path)
        self$name <- metadata$name
        self$horizons <- metadata$horizons
        self$config <- metadata$config
        self$is_trained <- metadata$is_trained
      }

      # Load models
      models_dir <- file.path(path, "models")
      if (dir.exists(models_dir)) {
        model_files <- list.files(models_dir, pattern = "\\.rds$",
                                  full.names = TRUE)
        for (model_file in model_files) {
          target <- tools::file_path_sans_ext(basename(model_file))
          private$models[[target]] <- readRDS(model_file)
        }
      }

      # Load feature selections
      fs_path <- file.path(path, "feature_selections.rds")
      if (file.exists(fs_path)) {
        private$feature_selections <- readRDS(fs_path)
      }

      # Load target metadata
      tm_path <- file.path(path, "target_metadata.csv")
      if (file.exists(tm_path)) {
        private$target_metadata <- data.table::fread(tm_path)
      }

      message(sprintf("Loaded %d models from: %s",
                      length(private$models), path))
      invisible(self)
    }
  )
)
```

### Factory Functions

```r
#' Create a multi-model manager for horizon-based forecasting
#'
#' @param name Manager name
#' @param model_type Type of model to create (e.g., "lightgbm")
#' @param horizons Vector of horizons (e.g., 0:8 for D+0 to D+8)
#' @param hours_per_day Hours per day (default 24)
#' @param model_config Configuration for individual models
#' @return MultiModelManager instance
#' @export
create_horizon_model_manager <- function(name,
                                         model_type,
                                         horizons = 0:8,
                                         hours_per_day = 24,
                                         model_config = list()) {
  # Create target names: y_h001, y_h002, ..., y_h216
  n_targets <- length(horizons) * hours_per_day
  targets <- sprintf("y_h%03d", seq_len(n_targets))

  factory <- function() {
    get_model(model_type, config = model_config)
  }

  manager <- MultiModelManager$new(
    name = name,
    model_factory = factory,
    config = list(
      model_type = model_type,
      horizons = horizons,
      hours_per_day = hours_per_day,
      n_targets = n_targets,
      target_names = targets
    )
  )

  manager
}


#' Create a multi-model manager for area-based forecasting
#'
#' @param name Manager name
#' @param model_type Type of model to create
#' @param areas Character vector of area codes
#' @param model_config Configuration for individual models
#' @return MultiModelManager instance
#' @export
create_area_model_manager <- function(name,
                                      model_type,
                                      areas = c("SECO", "S", "NE", "N"),
                                      model_config = list()) {
  factory <- function() {
    get_model(model_type, config = model_config)
  }

  MultiModelManager$new(
    name = name,
    model_factory = factory,
    config = list(
      model_type = model_type,
      areas = areas,
      n_targets = length(areas)
    )
  )
}
```

### Usage Example

```r
# Create manager with LightGBM factory
manager <- MultiModelManager$new(
  name = "lgbm_multi",
  model_factory = function() {
    get_model("lightgbm", config = list(num_leaves = 31))
  }
)

# Define targets and features
targets <- c("y_h001", "y_h002", "y_h003")
feature_selections <- list(
  y_h001 = c("temp", "hour", "weekday", "lag_24"),
  y_h002 = c("temp", "hour", "weekday", "lag_24", "lag_48"),
  y_h003 = c("temp", "hour", "weekday", "lag_24", "lag_48", "lag_72")
)

# Train models
manager$train_multi(
  X = features,
  y = target_data,
  target_cols = targets,
  feature_selections = feature_selections
)

# Predict
predictions <- manager$predict_multi(new_features)
# Returns data.table with columns: y_h001, y_h002, y_h003

# Get specific model
model_h001 <- manager$get_model("y_h001")

# Save/load
manager$save("models/lgbm_multi/v1.0.0")
manager$load("models/lgbm_multi/v1.0.0")

# Horizon-based manager (216 models)
horizon_manager <- create_horizon_model_manager(
  name = "lgbm_horizon",
  model_type = "lightgbm",
  horizons = 0:8,
  hours_per_day = 24
)
# Creates targets: y_h001 to y_h216
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with factory | Factory stored |
| TC-002 | train_multi() basic | All models trained |
| TC-003 | train_multi() with feature selection | Per-target features |
| TC-004 | predict_multi() all targets | All predictions |
| TC-005 | predict_multi() subset | Selected predictions |
| TC-006 | get_model() existing | Returns model |
| TC-007 | get_model() missing | Returns NULL |
| TC-008 | list_targets() | Returns target names |
| TC-009 | n_models() | Correct count |
| TC-010 | save() creates structure | models/, metadata.yaml |
| TC-011 | load() restores state | All models restored |
| TC-012 | create_horizon_model_manager() | 216 targets |
| TC-013 | create_area_model_manager() | Area targets |
| TC-014 | remove_model() | Target removed |

---

## Dependencies

- PC-016-03: BaseModel (parent class)
- PC-017-03: ModelRegistry (for get_model)

---

## Definition of Done

- [ ] MultiModelManager R6 class implemented
- [ ] train_multi() with feature selections
- [ ] predict_multi() with target filtering
- [ ] Model storage and retrieval
- [ ] save()/load() methods working
- [ ] Factory functions implemented
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The 216-model pattern is for 9 days × 24 hours of forecasts
- Feature selection per target enables horizon-appropriate lag features
- Consider memory optimization for large model counts
- Future: add model pruning for unused targets
- Integration with ParallelTrainer (PC-026-04) for efficient training
