# PC-027-04: Inference Workflow Infrastructure

**Epic:** [EPIC-04: Model Layer - Hierarchical Support](../epics/EPIC-04-model-layer-part2.md)
**Task Reference:** T-04.4
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `InferenceWorkflow` R6 class that provides a structured workflow for model inference, including feature building, normalization, prediction, and denormalization.

---

## Acceptance Criteria

- [ ] Inference workflow module created in `R/models/inference.R`
- [ ] `InferenceWorkflow` R6 class with complete inference pipeline
- [ ] Feature building integration
- [ ] Normalization/denormalization support
- [ ] Inference result structure with metadata
- [ ] Support for different model types
- [ ] Caching for repeated inferences

---

## Technical Specification

### File Location
```
R/models/inference.R
```

### InferenceWorkflow R6 Class

```r
#' @title InferenceWorkflow
#' @description Structured workflow for model inference
#'
#' Orchestrates the complete inference pipeline:
#' 1. Build inference features from anchor data
#' 2. Apply normalization (if configured)
#' 3. Run model prediction
#' 4. Denormalize results
#' 5. Format and validate output
#'
#' @export
InferenceWorkflow <- R6::R6Class(
  "InferenceWorkflow",
  private = list(
    model_artifact = NULL,
    feature_builder = NULL,
    normalizer = NULL,
    config = NULL,
    cache = NULL,
    cache_enabled = FALSE
  ),
  public = list(
    #' @description Initialize inference workflow
    #' @param model_artifact ModelArtifact with trained model
    #' @param feature_builder Feature builder (FeaturePipeline or function)
    #' @param normalizer Normalizer for input/output transformation (optional)
    #' @param config Workflow configuration
    initialize = function(model_artifact,
                          feature_builder = NULL,
                          normalizer = NULL,
                          config = list()) {
      checkmate::assert_class(model_artifact, "ModelArtifact")

      if (!is.null(feature_builder)) {
        # Accept FeaturePipeline or function
        if (!inherits(feature_builder, "FeaturePipeline") &&
            !is.function(feature_builder)) {
          stop("feature_builder must be a FeaturePipeline or function")
        }
      }

      private$model_artifact <- model_artifact
      private$feature_builder <- feature_builder
      private$normalizer <- normalizer
      private$config <- private$apply_default_config(config)
      private$cache <- new.env(hash = TRUE)
      private$cache_enabled <- config$cache_enabled %||% FALSE
    },

    #' @description Run inference workflow
    #' @param anchor_data Anchor data for feature building
    #' @param target_date Target date for forecast
    #' @param horizon Forecast horizon (optional, inferred from model)
    #' @param ... Additional arguments
    #' @return InferenceResult with predictions and metadata
    run = function(anchor_data, target_date, horizon = NULL, ...) {
      checkmate::assert_data_table(anchor_data)

      start_time <- Sys.time()

      # Check cache
      cache_key <- private$make_cache_key(target_date, horizon)
      if (private$cache_enabled && exists(cache_key, envir = private$cache)) {
        message("Returning cached inference result")
        return(get(cache_key, envir = private$cache))
      }

      # Step 1: Build features
      features <- self$build_features(anchor_data, target_date, ...)

      # Step 2: Apply normalization
      if (!is.null(private$normalizer)) {
        features <- self$apply_normalization(features)
      }

      # Step 3: Validate feature alignment
      self$validate_features(features)

      # Step 4: Predict
      model <- private$model_artifact$get_model()
      predictions <- model$predict(features, horizon = horizon, ...)

      # Step 5: Denormalize
      if (!is.null(private$normalizer)) {
        predictions <- self$denormalize(predictions)
      }

      # Step 6: Create result
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      result <- InferenceResult$new(
        predictions = predictions,
        target_date = target_date,
        horizon = horizon,
        model_name = private$model_artifact$model_name,
        model_version = private$model_artifact$version$to_string(),
        feature_names = private$model_artifact$feature_names,
        inference_time = elapsed,
        anchor_rows = nrow(anchor_data),
        config = private$config
      )

      # Cache result
      if (private$cache_enabled) {
        assign(cache_key, result, envir = private$cache)
      }

      result
    },

    #' @description Build features from anchor data
    #' @param anchor_data Anchor data
    #' @param target_date Target date
    #' @param ... Additional arguments
    #' @return Feature data.table
    build_features = function(anchor_data, target_date, ...) {
      if (is.null(private$feature_builder)) {
        # Return anchor_data as-is if no builder configured
        return(anchor_data)
      }

      if (inherits(private$feature_builder, "FeaturePipeline")) {
        private$feature_builder$transform(anchor_data, target_date = target_date)
      } else {
        # Function-based builder
        private$feature_builder(anchor_data, target_date = target_date, ...)
      }
    },

    #' @description Apply normalization to features
    #' @param features Feature data.table
    #' @return Normalized features
    apply_normalization = function(features) {
      if (is.null(private$normalizer)) {
        return(features)
      }

      if (inherits(private$normalizer, "Normalizer")) {
        private$normalizer$transform(features)
      } else if (is.function(private$normalizer)) {
        private$normalizer(features, inverse = FALSE)
      } else {
        features
      }
    },

    #' @description Denormalize predictions
    #' @param predictions Prediction values
    #' @return Denormalized predictions
    denormalize = function(predictions) {
      if (is.null(private$normalizer)) {
        return(predictions)
      }

      if (inherits(private$normalizer, "Normalizer")) {
        private$normalizer$inverse_transform(predictions)
      } else if (is.function(private$normalizer)) {
        private$normalizer(predictions, inverse = TRUE)
      } else {
        predictions
      }
    },

    #' @description Validate that features match model expectations
    #' @param features Feature data.table
    #' @return TRUE if valid, error otherwise
    validate_features = function(features) {
      expected_features <- private$model_artifact$feature_names

      if (is.null(expected_features)) {
        return(TRUE)
      }

      actual_features <- names(features)

      missing <- setdiff(expected_features, actual_features)
      if (length(missing) > 0) {
        stop(sprintf(
          "Missing features for inference: %s",
          paste(missing, collapse = ", ")
        ))
      }

      # Reorder to match expected
      if (private$config$strict_feature_order) {
        features <- features[, ..expected_features]
      }

      invisible(TRUE)
    },

    #' @description Clear inference cache
    #' @return Invisible self
    clear_cache = function() {
      rm(list = ls(envir = private$cache), envir = private$cache)
      invisible(self)
    },

    #' @description Enable/disable caching
    #' @param enabled Logical
    #' @return Invisible self
    set_cache_enabled = function(enabled) {
      checkmate::assert_logical(enabled, len = 1)
      private$cache_enabled <- enabled
      invisible(self)
    },

    #' @description Get model artifact
    #' @return ModelArtifact
    get_model_artifact = function() {
      private$model_artifact
    },

    #' @description Get configuration
    #' @return List
    get_config = function() {
      private$config
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        strict_feature_order = TRUE,
        cache_enabled = FALSE,
        validate_predictions = TRUE,
        prediction_bounds = NULL  # list(lower = 0, upper = Inf)
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    },

    make_cache_key = function(target_date, horizon) {
      digest::digest(list(
        target_date = as.character(target_date),
        horizon = horizon,
        model_version = private$model_artifact$version$to_string()
      ))
    }
  )
)
```

### InferenceResult Class

```r
#' @title InferenceResult
#' @description Container for inference results with metadata
#' @export
InferenceResult <- R6::R6Class(
  "InferenceResult",
  public = list(
    predictions = NULL,
    target_date = NULL,
    horizon = NULL,
    model_name = NULL,
    model_version = NULL,
    feature_names = NULL,
    inference_time = NULL,
    anchor_rows = NULL,
    config = NULL,
    created_at = NULL,

    #' @description Initialize inference result
    initialize = function(predictions,
                          target_date,
                          horizon = NULL,
                          model_name = NULL,
                          model_version = NULL,
                          feature_names = NULL,
                          inference_time = NULL,
                          anchor_rows = NULL,
                          config = list()) {
      self$predictions <- predictions
      self$target_date <- target_date
      self$horizon <- horizon
      self$model_name <- model_name
      self$model_version <- model_version
      self$feature_names <- feature_names
      self$inference_time <- inference_time
      self$anchor_rows <- anchor_rows
      self$config <- config
      self$created_at <- Sys.time()
    },

    #' @description Get predictions as data.table
    #' @return data.table with predictions
    as_data_table = function() {
      if (is.vector(self$predictions)) {
        data.table::data.table(
          target_date = self$target_date,
          prediction = self$predictions
        )
      } else if (is.matrix(self$predictions)) {
        dt <- as.data.table(self$predictions)
        dt[, target_date := self$target_date]
        data.table::setcolorder(dt, c("target_date", names(dt)[-ncol(dt)]))
        dt
      } else {
        as.data.table(self$predictions)
      }
    },

    #' @description Get summary statistics
    #' @return List with statistics
    summary = function() {
      preds <- as.vector(self$predictions)

      list(
        n_predictions = length(preds),
        mean = mean(preds, na.rm = TRUE),
        sd = sd(preds, na.rm = TRUE),
        min = min(preds, na.rm = TRUE),
        max = max(preds, na.rm = TRUE),
        n_na = sum(is.na(preds)),
        inference_time_sec = self$inference_time
      )
    },

    #' @description Print result
    print = function() {
      cat(sprintf("InferenceResult for %s\n", self$target_date))
      cat(sprintf("  Model: %s (v%s)\n", self$model_name, self$model_version))
      cat(sprintf("  Horizon: %s\n", self$horizon %||% "all"))
      cat(sprintf("  Predictions: %d values\n", length(as.vector(self$predictions))))
      cat(sprintf("  Inference time: %.3f sec\n", self$inference_time))
      invisible(self)
    }
  )
)
```

### Normalizer Classes

```r
#' @title Normalizer
#' @description Base normalizer class
#' @export
Normalizer <- R6::R6Class(
  "Normalizer",
  public = list(
    transform = function(data) {
      stop("transform() must be implemented by subclass")
    },

    inverse_transform = function(data) {
      stop("inverse_transform() must be implemented by subclass")
    },

    fit = function(data) {
      stop("fit() must be implemented by subclass")
    }
  )
)


#' @title MinMaxNormalizer
#' @description Min-max normalization to [0, 1] range
#' @export
MinMaxNormalizer <- R6::R6Class(
  "MinMaxNormalizer",
  inherit = Normalizer,
  private = list(
    min_vals = NULL,
    max_vals = NULL,
    feature_cols = NULL
  ),
  public = list(
    #' @description Fit normalizer to data
    fit = function(data, feature_cols = NULL) {
      checkmate::assert_data_table(data)

      private$feature_cols <- feature_cols %||% names(data)[sapply(data, is.numeric)]

      private$min_vals <- sapply(private$feature_cols, function(col) {
        min(data[[col]], na.rm = TRUE)
      })

      private$max_vals <- sapply(private$feature_cols, function(col) {
        max(data[[col]], na.rm = TRUE)
      })

      invisible(self)
    },

    #' @description Transform data to [0, 1] range
    transform = function(data) {
      result <- data.table::copy(data)

      for (col in private$feature_cols) {
        range_val <- private$max_vals[col] - private$min_vals[col]
        if (range_val > 0) {
          result[, (col) := (get(col) - private$min_vals[col]) / range_val]
        }
      }

      result
    },

    #' @description Inverse transform from [0, 1] to original range
    inverse_transform = function(data) {
      if (is.numeric(data) && is.null(dim(data))) {
        # Single vector - assume it's the target
        col <- private$feature_cols[1]
        range_val <- private$max_vals[col] - private$min_vals[col]
        return(data * range_val + private$min_vals[col])
      }

      result <- if (is.data.table(data)) data.table::copy(data) else as.data.table(data)

      for (col in intersect(private$feature_cols, names(result))) {
        range_val <- private$max_vals[col] - private$min_vals[col]
        result[, (col) := get(col) * range_val + private$min_vals[col]]
      }

      result
    }
  )
)
```

### Usage Example

```r
# Load trained model artifact
artifact <- ModelArtifact$new()
artifact$load("models/lgbm_rj/v1.0.0")

# Create feature builder (function-based)
build_features <- function(anchor_data, target_date, ...) {
  # Add temporal features
  anchor_data[, `:=`(
    hour = lubridate::hour(DataHora),
    weekday = lubridate::wday(DataHora),
    month = lubridate::month(DataHora)
  )]

  # Add lag features
  anchor_data[, lag_24 := shift(CargaGlobal, 24)]
  anchor_data[, lag_168 := shift(CargaGlobal, 168)]

  anchor_data
}

# Create workflow
workflow <- InferenceWorkflow$new(
  model_artifact = artifact,
  feature_builder = build_features,
  normalizer = NULL,
  config = list(cache_enabled = TRUE)
)

# Run inference
result <- workflow$run(
  anchor_data = recent_data,
  target_date = as.Date("2025-01-18"),
  horizon = 1
)

# Access results
print(result)
# InferenceResult for 2025-01-18
#   Model: lgbm_rj (v1.0.0)
#   Horizon: 1
#   Predictions: 24 values
#   Inference time: 0.234 sec

result$summary()
# $mean: 45230.5
# $sd: 12340.2
# ...

predictions_dt <- result$as_data_table()
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with artifact only | Works |
| TC-002 | Initialize with feature builder | Builder stored |
| TC-003 | run() basic workflow | Returns InferenceResult |
| TC-004 | run() with feature builder | Features built |
| TC-005 | run() with normalizer | Normalization applied |
| TC-006 | validate_features() missing | Error thrown |
| TC-007 | validate_features() extra | Passes |
| TC-008 | Cache hit | Returns cached result |
| TC-009 | clear_cache() | Cache emptied |
| TC-010 | InferenceResult as_data_table() | Correct format |
| TC-011 | InferenceResult summary() | Statistics computed |
| TC-012 | MinMaxNormalizer fit/transform | Normalized to [0,1] |
| TC-013 | MinMaxNormalizer inverse | Original scale restored |

---

## Dependencies

- PC-019-03: ModelArtifact
- PC-011-02: FeaturePipeline (optional)

---

## Definition of Done

- [ ] InferenceWorkflow R6 class implemented
- [ ] Feature building integration
- [ ] Normalization/denormalization
- [ ] InferenceResult class
- [ ] Caching support
- [ ] Normalizer classes
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Caching is useful for repeated inferences at same date
- Normalization is optional but important for some model types
- Feature validation prevents silent errors from missing features
- Consider adding prediction bounds validation
- Future: async inference support for real-time applications
