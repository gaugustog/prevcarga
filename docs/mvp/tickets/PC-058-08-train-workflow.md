# PC-058-08: TrainWorkflow

**Epic:** [EPIC-08: Orchestrator](../epics/EPIC-08-orchestrator.md)
**Task Reference:** T-08.2
**Priority:** High
**Estimated Effort:** 2.5 days

---

## Summary

Implement the `TrainWorkflow` R6 class for orchestrating multi-area, multi-model training. This workflow coordinates data loading, feature engineering, parallel training, and model artifact storage.

---

## Acceptance Criteria

- [ ] Workflow module created in `R/orchestrator/workflows/train.R`
- [ ] `TrainWorkflow` R6 class with full orchestration
- [ ] Parallel training across areas using ParallelExecutor
- [ ] Feature pipeline integration
- [ ] Model artifact saving with versioning
- [ ] Training report generation

---

## Technical Specification

### File Location
```
R/orchestrator/workflows/train.R
```

### TrainWorkflow R6 Class

```r
#' @title TrainWorkflow
#' @description Orchestrates multi-area, multi-model training
#'
#' Coordinates the complete training pipeline:
#' 1. Load historical data for specified areas
#' 2. Build feature pipelines for each model
#' 3. Train models in parallel across areas
#' 4. Save model artifacts with versioning
#' 5. Generate training report
#'
#' @export
TrainWorkflow <- R6::R6Class(
  "TrainWorkflow",
  private = list(
    config = NULL,
    storage = NULL,
    logger = NULL,
    executor = NULL,
    results = NULL
  ),
  public = list(
    #' @description Initialize training workflow
    #' @param config ConfigManager instance
    #' @param storage StorageBackend instance
    #' @param logger StructuredLogger instance (optional)
    initialize = function(config, storage, logger = NULL) {
      checkmate::assert_class(config, "ConfigManager")

      private$config <- config
      private$storage <- storage
      private$logger <- logger %||% StructuredLogger$new()

      # Initialize parallel executor
      n_jobs <- config$get_with_default("training.parallel.n_jobs", 4)
      private$executor <- ParallelExecutor$new(n_workers = n_jobs)

      private$results <- list()
    },

    #' @description Run training workflow
    #' @param areas Character vector of area codes
    #' @param models Character vector of model names
    #' @param start_date Training data start date
    #' @param end_date Training data end date
    #' @param version Model version string
    #' @param ... Additional arguments
    #' @return TrainResult with training outcomes
    run = function(areas,
                   models,
                   start_date,
                   end_date,
                   version = NULL,
                   ...) {
      start_time <- Sys.time()

      # Default version based on timestamp
      version <- version %||% format(Sys.time(), "%Y%m%d_%H%M%S")

      private$logger$with_context("workflow", "train")
      private$logger$with_context("version", version)
      private$logger$info("Starting training workflow",
                          areas = length(areas),
                          models = length(models))

      # Validate inputs
      self$validate_inputs(areas, models, start_date, end_date)

      # Step 1: Load historical data
      private$logger$info("Loading historical data...")
      data <- self$load_data(areas, start_date, end_date)

      # Step 2: Train all models for all areas
      private$logger$info("Starting parallel training...")
      results <- self$train_all(data, areas, models, version)

      # Step 3: Generate report
      private$logger$info("Generating training report...")
      report <- self$generate_report(results, version)

      # Build result
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      private$logger$info("Training workflow completed",
                          elapsed_secs = elapsed,
                          success_count = sum(sapply(results, function(r) r$status == "success")))

      TrainResult$new(
        results = results,
        report = report,
        version = version,
        elapsed_time = elapsed,
        config = private$config$as_list()
      )
    },

    #' @description Validate workflow inputs
    #' @param areas Area codes
    #' @param models Model names
    #' @param start_date Start date
    #' @param end_date End date
    validate_inputs = function(areas, models, start_date, end_date) {
      checkmate::assert_character(areas, min.len = 1)
      checkmate::assert_character(models, min.len = 1)
      checkmate::assert_date(as.Date(start_date))
      checkmate::assert_date(as.Date(end_date))

      if (as.Date(start_date) >= as.Date(end_date)) {
        stop("start_date must be before end_date")
      }

      # Validate areas exist in config
      valid_areas <- private$config$get_areas()
      invalid <- setdiff(areas, valid_areas)
      if (length(invalid) > 0) {
        stop(sprintf("Unknown areas: %s", paste(invalid, collapse = ", ")))
      }

      # Validate models are registered
      for (model in models) {
        if (!has_model(model)) {
          stop(sprintf("Model '%s' not registered", model))
        }
      }

      invisible(TRUE)
    },

    #' @description Load historical data for training
    #' @param areas Area codes
    #' @param start_date Start date
    #' @param end_date End date
    #' @return data.table with training data
    load_data = function(areas, start_date, end_date) {
      loader <- DataLoader$new(private$storage)

      data <- loader$load_carga(
        areas = areas,
        start_date = start_date,
        end_date = end_date
      )

      private$logger$info("Data loaded",
                          n_rows = nrow(data),
                          n_areas = length(unique(data$area_code)))

      data
    },

    #' @description Train all models for all areas
    #' @param data Training data
    #' @param areas Area codes
    #' @param models Model names
    #' @param version Version string
    #' @return List of training results
    train_all = function(data, areas, models, version) {
      # Create training tasks
      tasks <- expand.grid(
        area = areas,
        model = models,
        stringsAsFactors = FALSE
      )

      # Run in parallel
      results <- private$executor$map(
        seq_len(nrow(tasks)),
        function(i) {
          area <- tasks$area[i]
          model <- tasks$model[i]
          area_data <- data[area_code == area]

          tryCatch({
            self$train_single(area, area_data, model, version)
          }, error = function(e) {
            list(
              area = area,
              model = model,
              status = "failed",
              error = conditionMessage(e)
            )
          })
        }
      )

      # Convert to named list
      names(results) <- paste(tasks$area, tasks$model, sep = "_")
      results
    },

    #' @description Train single model for single area
    #' @param area Area code
    #' @param data Area training data
    #' @param model_name Model name
    #' @param version Version string
    #' @return Training result
    train_single = function(area, data, model_name, version) {
      start_time <- Sys.time()

      private$logger$debug("Training model",
                           area = area,
                           model = model_name)

      # Get model configuration
      model_config <- private$config$get_model_config(model_name)

      # Create model instance
      model <- get_model(model_name, config = model_config$config %||% list())

      # Build feature pipeline
      pipeline <- self$build_feature_pipeline(model_name)

      # Transform data
      features <- pipeline$transform(data)

      # Train model
      model$train(features$X, features$y)

      # Create and save artifact
      artifact <- ModelArtifact$new()
      artifact$model <- model
      artifact$version <- version
      artifact$area <- area
      artifact$trained_at <- Sys.time()
      artifact$feature_pipeline <- pipeline
      artifact$metrics <- model$get_training_metrics()

      artifact_path <- self$get_artifact_path(area, model_name, version)
      artifact$save(private$storage, artifact_path)

      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      list(
        area = area,
        model = model_name,
        version = version,
        status = "success",
        artifact_path = artifact_path,
        elapsed_secs = elapsed,
        metrics = artifact$metrics
      )
    },

    #' @description Build feature pipeline for model
    #' @param model_name Model name
    #' @return FeaturePipeline instance
    build_feature_pipeline = function(model_name) {
      # Get feature configuration
      feature_config <- private$config$get("features.plugins")

      # Build pipeline from enabled plugins
      pipeline <- FeaturePipeline$new()

      for (plugin_name in names(feature_config)) {
        plugin_cfg <- feature_config[[plugin_name]]

        if (isTRUE(plugin_cfg$enabled)) {
          plugin <- get_feature_plugin(plugin_name, plugin_cfg$config %||% list())
          pipeline$add_plugin(plugin)
        }
      }

      pipeline
    },

    #' @description Get artifact storage path
    #' @param area Area code
    #' @param model_name Model name
    #' @param version Version string
    #' @return Storage path
    get_artifact_path = function(area, model_name, version) {
      sprintf("models/%s/%s/%s/model.rds", model_name, area, version)
    },

    #' @description Generate training report
    #' @param results Training results
    #' @param version Version string
    #' @return Training report
    generate_report = function(results, version) {
      # Summarize results
      summary_dt <- data.table::rbindlist(lapply(results, function(r) {
        data.table::data.table(
          area = r$area,
          model = r$model,
          status = r$status,
          elapsed_secs = r$elapsed_secs %||% NA_real_,
          error = r$error %||% NA_character_
        )
      }))

      # Calculate statistics
      n_total <- nrow(summary_dt)
      n_success <- sum(summary_dt$status == "success")
      n_failed <- sum(summary_dt$status == "failed")

      list(
        version = version,
        timestamp = Sys.time(),
        summary = list(
          total = n_total,
          success = n_success,
          failed = n_failed,
          success_rate = n_success / n_total
        ),
        details = summary_dt,
        failed_tasks = summary_dt[status == "failed"]
      )
    },

    #' @description Get last results
    #' @return Last training results
    get_results = function() {
      private$results
    },

    #' @description Print summary
    print = function() {
      cat("TrainWorkflow\n")
      cat(sprintf("  Parallel workers: %d\n",
                  private$config$get_with_default("training.parallel.n_jobs", 4)))
      invisible(self)
    }
  )
)
```

### TrainResult Class

```r
#' @title TrainResult
#' @description Container for training workflow results
#' @export
TrainResult <- R6::R6Class(
  "TrainResult",
  public = list(
    results = NULL,
    report = NULL,
    version = NULL,
    elapsed_time = NULL,
    config = NULL,

    #' @description Initialize result
    initialize = function(results, report, version, elapsed_time, config) {
      self$results <- results
      self$report <- report
      self$version <- version
      self$elapsed_time <- elapsed_time
      self$config <- config
    },

    #' @description Get successful results
    #' @return List of successful training results
    get_successful = function() {
      Filter(function(r) r$status == "success", self$results)
    },

    #' @description Get failed results
    #' @return List of failed training results
    get_failed = function() {
      Filter(function(r) r$status == "failed", self$results)
    },

    #' @description Get summary statistics
    #' @return Summary list
    summary = function() {
      self$report$summary
    },

    #' @description Print result
    print = function() {
      cat("TrainResult\n")
      cat(sprintf("  Version: %s\n", self$version))
      cat(sprintf("  Elapsed: %.2f sec\n", self$elapsed_time))
      cat(sprintf("  Success: %d/%d\n",
                  self$report$summary$success,
                  self$report$summary$total))

      if (self$report$summary$failed > 0) {
        cat("\nFailed tasks:\n")
        print(self$report$failed_tasks)
      }

      invisible(self)
    }
  )
)
```

### Usage Example

```r
# Initialize components
config <- load_config("config/config.yaml")
storage <- create_storage_backend(config$get_storage_config())
logger <- StructuredLogger$new()

# Create workflow
workflow <- TrainWorkflow$new(config, storage, logger)

# Run training
result <- workflow$run(
  areas = c("RJ", "SP", "MG"),
  models = c("lgbm", "rf"),
  start_date = "2023-01-01",
  end_date = "2024-01-01",
  version = "v1.0.0"
)

print(result)
# TrainResult
#   Version: v1.0.0
#   Elapsed: 145.32 sec
#   Success: 6/6

# Get successful trainings
successful <- result$get_successful()

# Get training summary
summary <- result$summary()
# $total: 6
# $success: 6
# $failed: 0
# $success_rate: 1.0
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with config | Workflow created |
| TC-002 | validate_inputs() valid | No error |
| TC-003 | validate_inputs() invalid area | Error thrown |
| TC-004 | validate_inputs() invalid model | Error thrown |
| TC-005 | run() full workflow | TrainResult returned |
| TC-006 | train_single() success | Model trained |
| TC-007 | train_single() failure | Error captured |
| TC-008 | build_feature_pipeline() | Pipeline created |
| TC-009 | generate_report() | Report generated |
| TC-010 | Parallel execution | All tasks complete |

---

## Dependencies

- PC-057-08: ConfigManager
- PC-061-08: ParallelExecutor
- PC-016-03: BaseModel
- PC-011-02: FeaturePipeline

---

## Definition of Done

- [ ] TrainWorkflow R6 class implemented
- [ ] TrainResult container implemented
- [ ] Parallel training working
- [ ] Feature pipeline integration
- [ ] Artifact saving with versioning
- [ ] Report generation
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Parallel execution uses ParallelExecutor
- Version can be auto-generated or provided
- Failed tasks don't stop entire workflow
- Consider adding checkpointing for long runs
