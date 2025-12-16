# PC-059-08: PredictWorkflow

**Epic:** [EPIC-08: Orchestrator](../epics/EPIC-08-orchestrator.md)
**Task Reference:** T-08.3
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Implement the `PredictWorkflow` R6 class for orchestrating batch and intraday prediction. This workflow coordinates model loading, feature engineering, prediction, combination, and reconciliation.

---

## Acceptance Criteria

- [ ] Workflow module created in `R/orchestrator/workflows/predict.R`
- [ ] `PredictWorkflow` R6 class with full orchestration
- [ ] Batch prediction mode (D+0 to D+8)
- [ ] Intraday prediction mode with anchor
- [ ] Model combination integration
- [ ] Hierarchical reconciliation integration
- [ ] Prediction result storage

---

## Technical Specification

### File Location
```
R/orchestrator/workflows/predict.R
```

### PredictWorkflow R6 Class

```r
#' @title PredictWorkflow
#' @description Orchestrates batch and intraday prediction
#'
#' Coordinates the complete prediction pipeline:
#' 1. Load latest model artifacts
#' 2. Load input data up to reference date
#' 3. Generate predictions for each area/model
#' 4. Combine predictions (optional)
#' 5. Reconcile hierarchically (optional)
#' 6. Save results
#'
#' @export
PredictWorkflow <- R6::R6Class(
  "PredictWorkflow",
  private = list(
    config = NULL,
    storage = NULL,
    logger = NULL,
    executor = NULL
  ),
  public = list(
    #' @description Initialize prediction workflow
    #' @param config ConfigManager instance
    #' @param storage StorageBackend instance
    #' @param logger StructuredLogger instance (optional)
    initialize = function(config, storage, logger = NULL) {
      checkmate::assert_class(config, "ConfigManager")

      private$config <- config
      private$storage <- storage
      private$logger <- logger %||% StructuredLogger$new()

      n_jobs <- config$get_with_default("training.parallel.n_jobs", 4)
      private$executor <- ParallelExecutor$new(n_workers = n_jobs)
    },

    #' @description Run prediction workflow
    #' @param date Reference date for prediction
    #' @param mode Prediction mode: "batch" or "intraday"
    #' @param areas Character vector of area codes (NULL = all)
    #' @param models Character vector of model names (NULL = all enabled)
    #' @param reconcile Apply hierarchical reconciliation
    #' @param combine Apply model combination
    #' @param ... Additional arguments
    #' @return PredictResult with predictions
    run = function(date,
                   mode = "batch",
                   areas = NULL,
                   models = NULL,
                   reconcile = TRUE,
                   combine = TRUE,
                   ...) {
      start_time <- Sys.time()

      private$logger$with_context("workflow", "predict")
      private$logger$with_context("date", as.character(date))
      private$logger$with_context("mode", mode)

      # Default areas and models from config
      areas <- areas %||% private$config$get_areas()
      models <- models %||% self$get_enabled_models()

      private$logger$info("Starting prediction workflow",
                          mode = mode,
                          areas = length(areas),
                          models = length(models))

      # Run appropriate mode
      if (mode == "batch") {
        result <- self$run_batch(date, areas, models, ...)
      } else if (mode == "intraday") {
        current_hour <- list(...)$current_hour %||% as.integer(format(Sys.time(), "%H"))
        result <- self$run_intraday(date, current_hour, areas, models, ...)
      } else {
        stop(sprintf("Unknown prediction mode: %s", mode))
      }

      # Combine predictions
      if (combine && length(models) > 1) {
        private$logger$info("Combining model predictions...")
        result <- self$combine_predictions(result)
      }

      # Reconcile hierarchy
      if (reconcile) {
        private$logger$info("Reconciling hierarchically...")
        result <- self$reconcile_predictions(result)
      }

      # Save results
      private$logger$info("Saving predictions...")
      self$save_predictions(result, date, mode)

      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      private$logger$info("Prediction workflow completed",
                          elapsed_secs = elapsed)

      PredictResult$new(
        predictions = result$predictions,
        combined = result$combined,
        reconciled = result$reconciled,
        date = date,
        mode = mode,
        elapsed_time = elapsed,
        metadata = list(
          areas = areas,
          models = models,
          reconciled = reconcile,
          combined = combine
        )
      )
    },

    #' @description Run batch prediction (D+0 to D+8)
    #' @param date Reference date
    #' @param areas Area codes
    #' @param models Model names
    #' @param ... Additional arguments
    #' @return List with predictions
    run_batch = function(date, areas, models, ...) {
      horizons <- private$config$get_with_default("prediction.horizons", 0:8)

      # Load input data
      data <- self$load_input_data(date, areas, max(horizons))

      # Generate predictions for each area/model
      predictions <- private$executor$map(
        areas,
        function(area) {
          area_preds <- list()

          for (model_name in models) {
            preds <- self$predict_single(area, model_name, data, date, horizons)
            area_preds[[model_name]] <- preds
          }

          list(area = area, predictions = area_preds)
        }
      )

      # Restructure results
      self$structure_predictions(predictions, date, horizons)
    },

    #' @description Run intraday prediction
    #' @param date Reference date
    #' @param current_hour Current hour
    #' @param areas Area codes
    #' @param models Model names
    #' @param ... Additional arguments
    #' @return List with predictions
    run_intraday = function(date, current_hour, areas, models, ...) {
      # For intraday:
      # - Hours 0 to current_hour use actuals (anchor)
      # - Hours current_hour+1 to 23 are predicted
      # - Next day D+1 onwards are full predictions

      private$logger$info("Running intraday prediction",
                          current_hour = current_hour)

      # Load input data including today's actuals
      data <- self$load_input_data(date, areas, lookahead = 8)

      # Get today's actuals up to current hour
      actuals <- self$get_intraday_actuals(data, date, current_hour, areas)

      # Generate predictions
      predictions <- private$executor$map(
        areas,
        function(area) {
          area_preds <- list()

          for (model_name in models) {
            # Predict remaining hours
            preds <- self$predict_intraday_single(
              area, model_name, data, date, current_hour
            )
            area_preds[[model_name]] <- preds
          }

          list(
            area = area,
            actuals = actuals[[area]],
            predictions = area_preds
          )
        }
      )

      # Combine actuals with predictions
      self$merge_intraday_predictions(predictions, date, current_hour)
    },

    #' @description Load input data for prediction
    #' @param date Reference date
    #' @param areas Area codes
    #' @param lookahead Days of history needed
    #' @return data.table with input data
    load_input_data = function(date, areas, lookahead = 8) {
      # Need enough history for lag features
      max_lag <- private$config$get_with_default("features.plugins.lags.config.max_lag", 168)
      lookback <- max(30, ceiling(max_lag / 24))

      start_date <- as.Date(date) - lookback
      end_date <- as.Date(date)

      loader <- DataLoader$new(private$storage)
      loader$load_carga(areas, start_date, end_date)
    },

    #' @description Predict for single area/model
    #' @param area Area code
    #' @param model_name Model name
    #' @param data Input data
    #' @param date Reference date
    #' @param horizons Forecast horizons
    #' @return data.table with predictions
    predict_single = function(area, model_name, data, date, horizons) {
      # Load model artifact
      artifact <- self$load_model_artifact(area, model_name)

      # Get area data
      area_data <- data[area_code == area]

      # Apply feature pipeline
      features <- artifact$feature_pipeline$transform(area_data)

      # Generate predictions for each horizon
      predictions <- lapply(horizons, function(h) {
        target_date <- as.Date(date) + h

        # Create future features
        future_features <- self$create_future_features(
          features, area_data, target_date, h
        )

        # Predict
        preds <- artifact$model$predict(future_features)

        data.table::data.table(
          DataHora = seq(
            as.POSIXct(paste(target_date, "00:00:00")),
            as.POSIXct(paste(target_date, "23:00:00")),
            by = "hour"
          ),
          area_code = area,
          model = model_name,
          horizon = h,
          prediction = preds
        )
      })

      data.table::rbindlist(predictions)
    },

    #' @description Load model artifact
    #' @param area Area code
    #' @param model_name Model name
    #' @param version Version (NULL = latest)
    #' @return ModelArtifact instance
    load_model_artifact = function(area, model_name, version = NULL) {
      if (is.null(version)) {
        version <- self$get_latest_version(area, model_name)
      }

      path <- sprintf("models/%s/%s/%s/model.rds", model_name, area, version)

      artifact <- ModelArtifact$new()
      artifact$load(private$storage, path)

      artifact
    },

    #' @description Get latest model version
    #' @param area Area code
    #' @param model_name Model name
    #' @return Version string
    get_latest_version = function(area, model_name) {
      path <- sprintf("models/%s/%s/", model_name, area)
      versions <- private$storage$list(path)

      if (length(versions) == 0) {
        stop(sprintf("No model versions found for %s/%s", model_name, area))
      }

      # Sort by version (assuming timestamp format)
      sort(versions, decreasing = TRUE)[1]
    },

    #' @description Create future features for prediction
    #' @param features Current features
    #' @param data Historical data
    #' @param target_date Target prediction date
    #' @param horizon Forecast horizon
    #' @return Feature matrix for prediction
    create_future_features = function(features, data, target_date, horizon) {
      # This would use the feature pipeline to create features
      # for future time points
      # Implementation depends on feature plugin capabilities
      features
    },

    #' @description Combine predictions from multiple models
    #' @param result Prediction result
    #' @return Result with combined predictions
    combine_predictions = function(result) {
      # Get combiner
      combiner_name <- private$config$get_with_default("prediction.combiner", "inverse_mse")
      combiner <- get_combiner(combiner_name)

      # Combine for each area
      combined <- combiner$combine(result$predictions)

      result$combined <- combined
      result
    },

    #' @description Reconcile predictions hierarchically
    #' @param result Prediction result
    #' @return Result with reconciled predictions
    reconcile_predictions = function(result) {
      # Get reconciler
      reconciler_name <- private$config$get_with_default("prediction.reconciler", "bottom_up")
      reconciler <- get_reconciler(reconciler_name)

      # Create hierarchy
      hierarchy <- create_sin_hierarchy()

      # Reconcile
      reconciled <- reconciler$reconcile(
        result$combined %||% result$predictions,
        hierarchy
      )

      result$reconciled <- reconciled
      result
    },

    #' @description Save predictions
    #' @param result Prediction result
    #' @param date Reference date
    #' @param mode Prediction mode
    save_predictions = function(result, date, mode) {
      output_path <- sprintf(
        "predictions/%s/%s/predictions.parquet",
        format(as.Date(date), "%Y/%m/%d"),
        mode
      )

      # Convert to data.table and save
      predictions_dt <- self$predictions_to_dt(result)

      private$storage$write_parquet(predictions_dt, output_path)

      private$logger$info("Predictions saved", path = output_path)
    },

    #' @description Convert predictions to data.table
    #' @param result Prediction result
    #' @return data.table
    predictions_to_dt = function(result) {
      # Flatten predictions structure to data.table
      result$reconciled %||% result$combined %||% result$predictions
    },

    #' @description Get enabled models from config
    #' @return Character vector of model names
    get_enabled_models = function() {
      models_config <- private$config$get("models.plugins")
      enabled <- names(Filter(function(m) isTRUE(m$enabled), models_config))
      enabled
    },

    #' @description Structure predictions from parallel results
    structure_predictions = function(raw_results, date, horizons) {
      # Convert parallel results to structured format
      list(predictions = raw_results)
    },

    #' @description Get intraday actuals
    get_intraday_actuals = function(data, date, current_hour, areas) {
      # Filter today's data up to current hour
      today_data <- data[as.Date(DataHora) == as.Date(date) &
                         data.table::hour(DataHora) <= current_hour]

      # Split by area
      split(today_data, today_data$area_code)
    },

    #' @description Predict intraday for single area
    predict_intraday_single = function(area, model_name, data, date, current_hour) {
      # Similar to predict_single but only for remaining hours
      NULL
    },

    #' @description Merge intraday predictions
    merge_intraday_predictions = function(predictions, date, current_hour) {
      # Combine actuals (hours 0 to current) with forecasts (current+1 to 23)
      list(predictions = predictions)
    },

    #' @description Print summary
    print = function() {
      cat("PredictWorkflow\n")
      cat(sprintf("  Modes: batch, intraday\n"))
      cat(sprintf("  Reconciliation: %s\n",
                  private$config$get_with_default("prediction.reconciliation", TRUE)))
      invisible(self)
    }
  )
)
```

### PredictResult Class

```r
#' @title PredictResult
#' @description Container for prediction workflow results
#' @export
PredictResult <- R6::R6Class(
  "PredictResult",
  public = list(
    predictions = NULL,
    combined = NULL,
    reconciled = NULL,
    date = NULL,
    mode = NULL,
    elapsed_time = NULL,
    metadata = NULL,

    #' @description Initialize result
    initialize = function(predictions, combined, reconciled,
                          date, mode, elapsed_time, metadata) {
      self$predictions <- predictions
      self$combined <- combined
      self$reconciled <- reconciled
      self$date <- date
      self$mode <- mode
      self$elapsed_time <- elapsed_time
      self$metadata <- metadata
    },

    #' @description Get final predictions
    #' @return Reconciled > Combined > Raw predictions
    get_final = function() {
      self$reconciled %||% self$combined %||% self$predictions
    },

    #' @description Get predictions for area
    #' @param area Area code
    #' @return Predictions for area
    get_area = function(area) {
      final <- self$get_final()
      if (is.data.table(final)) {
        final[area_code == area]
      } else {
        final[[area]]
      }
    },

    #' @description Print result
    print = function() {
      cat("PredictResult\n")
      cat(sprintf("  Date: %s\n", self$date))
      cat(sprintf("  Mode: %s\n", self$mode))
      cat(sprintf("  Elapsed: %.2f sec\n", self$elapsed_time))
      cat(sprintf("  Areas: %d\n", length(self$metadata$areas)))
      cat(sprintf("  Models: %s\n", paste(self$metadata$models, collapse = ", ")))
      cat(sprintf("  Reconciled: %s\n", self$metadata$reconciled))
      invisible(self)
    }
  )
)
```

### Usage Example

```r
# Initialize
config <- load_config("config/config.yaml")
storage <- create_storage_backend(config$get_storage_config())

workflow <- PredictWorkflow$new(config, storage)

# Batch prediction
result <- workflow$run(
  date = "2024-01-15",
  mode = "batch",
  areas = c("RJ", "SP"),
  reconcile = TRUE,
  combine = TRUE
)

# Intraday prediction
result <- workflow$run(
  date = "2024-01-15",
  mode = "intraday",
  current_hour = 14,
  reconcile = TRUE
)

# Get final predictions
predictions <- result$get_final()

# Get specific area
rj_preds <- result$get_area("RJ")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | run() batch mode | Predictions generated |
| TC-002 | run() intraday mode | Predictions with anchor |
| TC-003 | load_input_data() | Data loaded |
| TC-004 | predict_single() | Area predictions |
| TC-005 | load_model_artifact() | Artifact loaded |
| TC-006 | get_latest_version() | Version returned |
| TC-007 | combine_predictions() | Combined output |
| TC-008 | reconcile_predictions() | Reconciled output |
| TC-009 | save_predictions() | Saved to storage |
| TC-010 | get_enabled_models() | Enabled list |

---

## Dependencies

- PC-057-08: ConfigManager
- PC-061-08: ParallelExecutor
- PC-034-05: CombinationWorkflow
- PC-042-06: ReconciliationWorkflow

---

## Definition of Done

- [ ] PredictWorkflow R6 class implemented
- [ ] PredictResult container implemented
- [ ] Batch mode working
- [ ] Intraday mode working
- [ ] Combination integration
- [ ] Reconciliation integration
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Batch mode generates D+0 through D+8 forecasts
- Intraday mode uses actuals as anchor
- Reconciliation ensures hierarchical coherence
- Consider caching model artifacts in memory
