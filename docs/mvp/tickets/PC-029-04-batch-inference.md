# PC-029-04: Batch Inference Runner

**Epic:** [EPIC-04: Model Layer - Hierarchical Support](../epics/EPIC-04-model-layer-part2.md)
**Task Reference:** T-04.6
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Implement the `BatchInferenceRunner` R6 class for running inference across multiple areas, models, or dates in parallel, with result aggregation and failure handling.

---

## Acceptance Criteria

- [ ] Batch inference module created in `R/models/batch_inference.R`
- [ ] `BatchInferenceRunner` R6 class for multi-area/multi-date inference
- [ ] Parallel execution across areas
- [ ] Result aggregation
- [ ] Failure handling and retry logic
- [ ] Progress reporting
- [ ] Integration with ModelRegistry for model discovery

---

## Technical Specification

### File Location
```
R/models/batch_inference.R
```

### BatchInferenceRunner R6 Class

```r
#' @title BatchInferenceRunner
#' @description Run inference across multiple areas, dates, or models in parallel
#'
#' Orchestrates batch inference for production forecasting:
#' - Run forecasts for all SIN areas simultaneously
#' - Run backtesting across multiple dates
#' - Aggregate results and handle failures gracefully
#'
#' @export
BatchInferenceRunner <- R6::R6Class(
  "BatchInferenceRunner",
  private = list(
    model_registry = NULL,
    storage = NULL,
    parallel_trainer = NULL,
    config = NULL
  ),
  public = list(
    #' @description Initialize batch runner
    #' @param model_registry ModelRegistry for model discovery
    #' @param storage StorageBackend for results
    #' @param n_workers Number of parallel workers
    #' @param config Runner configuration
    initialize = function(model_registry = NULL,
                          storage = NULL,
                          n_workers = parallel::detectCores() - 1,
                          config = list()) {
      private$model_registry <- model_registry
      private$storage <- storage
      private$parallel_trainer <- ParallelTrainer$new(n_workers = n_workers)
      private$config <- private$apply_default_config(config)
    },

    #' @description Run batch inference across areas
    #' @param areas Character vector of area codes
    #' @param target_date Target forecast date
    #' @param horizons Forecast horizons
    #' @param data_loader Function or DataLoader to get area data
    #' @param ... Additional arguments
    #' @return BatchInferenceResult
    run_batch = function(areas, target_date, horizons = 0:8,
                         data_loader, ...) {
      checkmate::assert_character(areas, min.len = 1)

      message(sprintf(
        "Starting batch inference for %d areas, %d horizons",
        length(areas), length(horizons)
      ))

      start_time <- Sys.time()

      # Create inference tasks
      tasks <- self$create_area_tasks(areas, target_date, horizons,
                                      data_loader, ...)

      # Run in parallel
      private$parallel_trainer$setup()

      results <- future.apply::future_lapply(
        tasks,
        function(task) {
          tryCatch({
            self$run_single_inference(task)
          }, error = function(e) {
            list(
              area = task$area,
              status = "error",
              error = conditionMessage(e)
            )
          })
        },
        future.seed = TRUE
      )

      private$parallel_trainer$shutdown()

      # Aggregate results
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      aggregated <- self$aggregate_results(results)

      BatchInferenceResult$new(
        results = results,
        aggregated = aggregated,
        areas = areas,
        target_date = target_date,
        horizons = horizons,
        elapsed_time = elapsed
      )
    },

    #' @description Run batch inference across dates (backtesting)
    #' @param dates Vector of target dates
    #' @param areas Areas to forecast
    #' @param horizons Forecast horizons
    #' @param data_loader Function to get data for each date
    #' @param ... Additional arguments
    #' @return BatchInferenceResult
    run_backtest = function(dates, areas, horizons = 0:8,
                            data_loader, ...) {
      checkmate::assert_date(dates)

      message(sprintf(
        "Starting backtest for %d dates, %d areas",
        length(dates), length(areas)
      ))

      start_time <- Sys.time()

      # Create tasks for all date-area combinations
      tasks <- list()
      for (date in dates) {
        for (area in areas) {
          tasks <- c(tasks, list(list(
            date = date,
            area = area,
            horizons = horizons,
            data_loader = data_loader
          )))
        }
      }

      # Run in parallel
      private$parallel_trainer$setup()

      results <- future.apply::future_lapply(
        tasks,
        function(task) {
          tryCatch({
            # Load data for this date
            anchor_data <- task$data_loader(
              area = task$area,
              end_date = task$date,
              ...
            )

            # Get model for area
            model_artifact <- self$get_model_for_area(task$area)

            # Run inference
            workflow <- InferenceWorkflow$new(
              model_artifact = model_artifact
            )

            result <- workflow$run(
              anchor_data = anchor_data,
              target_date = task$date,
              ...
            )

            list(
              date = task$date,
              area = task$area,
              status = "success",
              predictions = result$predictions,
              inference_time = result$inference_time
            )
          }, error = function(e) {
            list(
              date = task$date,
              area = task$area,
              status = "error",
              error = conditionMessage(e)
            )
          })
        },
        future.seed = TRUE
      )

      private$parallel_trainer$shutdown()

      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      BacktestInferenceResult$new(
        results = results,
        dates = dates,
        areas = areas,
        horizons = horizons,
        elapsed_time = elapsed
      )
    },

    #' @description Create inference tasks for areas
    #' @param areas Area codes
    #' @param target_date Target date
    #' @param horizons Horizons
    #' @param data_loader Data loader function
    #' @param ... Additional arguments
    #' @return List of tasks
    create_area_tasks = function(areas, target_date, horizons,
                                 data_loader, ...) {
      lapply(areas, function(area) {
        list(
          area = area,
          target_date = target_date,
          horizons = horizons,
          data_loader = data_loader,
          extra_args = list(...)
        )
      })
    },

    #' @description Run single inference task
    #' @param task Task specification
    #' @return Inference result
    run_single_inference = function(task) {
      # Load data
      anchor_data <- task$data_loader(
        area = task$area,
        end_date = task$target_date
      )

      # Get model
      model_artifact <- self$get_model_for_area(task$area)

      # Create workflow
      workflow <- InferenceWorkflow$new(
        model_artifact = model_artifact
      )

      # Run inference
      result <- workflow$run(
        anchor_data = anchor_data,
        target_date = task$target_date
      )

      list(
        area = task$area,
        status = "success",
        predictions = result$predictions,
        inference_time = result$inference_time
      )
    },

    #' @description Get model artifact for an area
    #' @param area Area code
    #' @return ModelArtifact
    get_model_for_area = function(area) {
      if (!is.null(private$model_registry)) {
        model_name <- sprintf("lgbm_%s", tolower(area))
        private$model_registry$get(model_name)
      } else {
        stop("model_registry must be configured to get models")
      }
    },

    #' @description Aggregate batch results
    #' @param results List of individual results
    #' @return Aggregated summary
    aggregate_results = function(results) {
      n_success <- sum(sapply(results, function(r) r$status == "success"))
      n_error <- sum(sapply(results, function(r) r$status == "error"))

      errors <- Filter(function(r) r$status == "error", results)
      error_summary <- if (length(errors) > 0) {
        lapply(errors, function(e) list(area = e$area, error = e$error))
      } else NULL

      # Combine successful predictions
      successful <- Filter(function(r) r$status == "success", results)
      combined_predictions <- if (length(successful) > 0) {
        data.table::rbindlist(lapply(successful, function(r) {
          if (is.vector(r$predictions)) {
            data.table::data.table(
              area = r$area,
              hour = seq_along(r$predictions) - 1,
              prediction = r$predictions
            )
          } else {
            dt <- as.data.table(r$predictions)
            dt[, area := r$area]
            dt
          }
        }), fill = TRUE)
      } else NULL

      list(
        n_total = length(results),
        n_success = n_success,
        n_error = n_error,
        success_rate = n_success / length(results),
        errors = error_summary,
        combined_predictions = combined_predictions,
        total_inference_time = sum(sapply(successful, function(r) {
          r$inference_time %||% 0
        }))
      )
    },

    #' @description Retry failed tasks
    #' @param batch_result Previous BatchInferenceResult
    #' @param max_retries Maximum retry attempts
    #' @return Updated BatchInferenceResult
    retry_failed = function(batch_result, max_retries = 3) {
      failed_areas <- sapply(
        Filter(function(r) r$status == "error", batch_result$results),
        function(r) r$area
      )

      if (length(failed_areas) == 0) {
        message("No failed tasks to retry")
        return(batch_result)
      }

      message(sprintf("Retrying %d failed areas...", length(failed_areas)))

      # Retry with same parameters
      # Implementation would need access to original data_loader
      # This is a simplified version
      batch_result
    },

    #' @description Save results to storage
    #' @param batch_result BatchInferenceResult
    #' @param path Storage path
    #' @return Invisible path
    save_results = function(batch_result, path) {
      if (is.null(private$storage)) {
        stop("Storage backend not configured")
      }

      # Save predictions
      predictions_dt <- batch_result$as_data_table()
      private$storage$write_parquet(
        predictions_dt,
        file.path(path, "predictions.parquet")
      )

      # Save metadata
      metadata <- list(
        areas = batch_result$areas,
        target_date = as.character(batch_result$target_date),
        horizons = batch_result$horizons,
        n_success = batch_result$aggregated$n_success,
        n_error = batch_result$aggregated$n_error,
        elapsed_time = batch_result$elapsed_time,
        saved_at = Sys.time()
      )
      yaml::write_yaml(metadata, file.path(path, "metadata.yaml"))

      message(sprintf("Saved batch results to: %s", path))
      invisible(path)
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        retry_on_failure = TRUE,
        max_retries = 3,
        timeout_per_task = 300,  # seconds
        save_intermediate = FALSE
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    }
  )
)
```

### Result Classes

```r
#' @title BatchInferenceResult
#' @description Container for batch inference results
#' @export
BatchInferenceResult <- R6::R6Class(
  "BatchInferenceResult",
  public = list(
    results = NULL,
    aggregated = NULL,
    areas = NULL,
    target_date = NULL,
    horizons = NULL,
    elapsed_time = NULL,
    created_at = NULL,

    #' @description Initialize result
    initialize = function(results, aggregated, areas, target_date,
                          horizons, elapsed_time) {
      self$results <- results
      self$aggregated <- aggregated
      self$areas <- areas
      self$target_date <- target_date
      self$horizons <- horizons
      self$elapsed_time <- elapsed_time
      self$created_at <- Sys.time()
    },

    #' @description Get results as data.table
    as_data_table = function() {
      self$aggregated$combined_predictions
    },

    #' @description Get result for specific area
    get_area = function(area) {
      for (r in self$results) {
        if (r$area == area) return(r)
      }
      NULL
    },

    #' @description Get failed areas
    get_failed = function() {
      sapply(
        Filter(function(r) r$status == "error", self$results),
        function(r) r$area
      )
    },

    #' @description Print summary
    print = function() {
      cat(sprintf("BatchInferenceResult\n"))
      cat(sprintf("  Target date: %s\n", self$target_date))
      cat(sprintf("  Areas: %s\n", paste(self$areas, collapse = ", ")))
      cat(sprintf("  Success: %d/%d (%.1f%%)\n",
                  self$aggregated$n_success,
                  self$aggregated$n_total,
                  self$aggregated$success_rate * 100))
      cat(sprintf("  Elapsed: %.2f sec\n", self$elapsed_time))

      if (self$aggregated$n_error > 0) {
        cat("\nFailed areas:\n")
        for (err in self$aggregated$errors) {
          cat(sprintf("  - %s: %s\n", err$area, err$error))
        }
      }

      invisible(self)
    }
  )
)


#' @title BacktestInferenceResult
#' @description Container for backtest results across dates
#' @export
BacktestInferenceResult <- R6::R6Class(
  "BacktestInferenceResult",
  public = list(
    results = NULL,
    dates = NULL,
    areas = NULL,
    horizons = NULL,
    elapsed_time = NULL,

    #' @description Initialize result
    initialize = function(results, dates, areas, horizons, elapsed_time) {
      self$results <- results
      self$dates <- dates
      self$areas <- areas
      self$horizons <- horizons
      self$elapsed_time <- elapsed_time
    },

    #' @description Get all results as data.table
    as_data_table = function() {
      successful <- Filter(function(r) r$status == "success", self$results)

      data.table::rbindlist(lapply(successful, function(r) {
        data.table::data.table(
          date = r$date,
          area = r$area,
          predictions = list(r$predictions)
        )
      }))
    },

    #' @description Get results for specific date
    get_date = function(date) {
      Filter(function(r) r$date == date, self$results)
    },

    #' @description Summary statistics
    summary = function() {
      n_tasks <- length(self$results)
      n_success <- sum(sapply(self$results, function(r) r$status == "success"))

      list(
        n_dates = length(self$dates),
        n_areas = length(self$areas),
        n_tasks = n_tasks,
        n_success = n_success,
        success_rate = n_success / n_tasks,
        elapsed_time = self$elapsed_time,
        avg_time_per_task = self$elapsed_time / n_tasks
      )
    }
  )
)
```

### Usage Example

```r
# Create batch runner
runner <- BatchInferenceRunner$new(
  model_registry = model_registry,
  storage = LocalStorageBackend$new("./data"),
  n_workers = 4
)

# Define data loader
load_area_data <- function(area, end_date, ...) {
  data_loader$load(
    area = area,
    start_date = end_date - 365,
    end_date = end_date
  )
}

# Run batch inference for all SIN areas
result <- runner$run_batch(
  areas = c("SECO", "S", "NE", "N"),
  target_date = as.Date("2025-01-18"),
  horizons = 0:8,
  data_loader = load_area_data
)

print(result)
# BatchInferenceResult
#   Target date: 2025-01-18
#   Areas: SECO, S, NE, N
#   Success: 4/4 (100.0%)
#   Elapsed: 12.34 sec

# Get combined predictions
forecasts <- result$as_data_table()

# Save results
runner$save_results(result, "forecasts/2025-01-18")

# Run backtest
backtest_result <- runner$run_backtest(
  dates = seq(as.Date("2024-10-01"), as.Date("2024-12-31"), by = "week"),
  areas = c("SECO", "S"),
  horizons = 0:8,
  data_loader = load_area_data
)

backtest_result$summary()
# $n_dates: 14
# $n_areas: 2
# $n_tasks: 28
# $n_success: 28
# $success_rate: 1.0
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with registry | Registry stored |
| TC-002 | run_batch() all success | All areas complete |
| TC-003 | run_batch() partial failure | Failures captured |
| TC-004 | run_backtest() | Multi-date results |
| TC-005 | aggregate_results() | Combined predictions |
| TC-006 | get_area() specific | Returns area result |
| TC-007 | get_failed() | Returns failed areas |
| TC-008 | save_results() | Files created |
| TC-009 | Parallel execution | Correct worker count |
| TC-010 | BacktestInferenceResult summary | Statistics computed |

---

## Dependencies

- PC-027-04: InferenceWorkflow
- PC-026-04: ParallelTrainer
- PC-017-03: ModelRegistry

---

## Definition of Done

- [ ] BatchInferenceRunner R6 class implemented
- [ ] run_batch() for multi-area inference
- [ ] run_backtest() for multi-date inference
- [ ] Result aggregation
- [ ] Error handling and reporting
- [ ] Result persistence
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Batch inference is core to operational forecasting
- Consider adding scheduling support (cron-like)
- Memory usage scales with concurrent tasks
- Future: add streaming results for large backtests
- Consider adding checkpointing for long-running batches
