# PC-026-04: Parallel Training Infrastructure

**Epic:** [EPIC-04: Model Layer - Hierarchical Support](../epics/EPIC-04-model-layer-part2.md)
**Task Reference:** T-04.3
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `ParallelTrainer` R6 class that enables parallel training of multiple models using the `future` and `future.apply` packages, critical for training 48+ profile models or 216+ horizon models efficiently.

---

## Acceptance Criteria

- [ ] Parallel training module created in `R/models/parallel.R`
- [ ] `ParallelTrainer` R6 class with worker management
- [ ] Support for `future::multisession` and `future::multicore`
- [ ] Task distribution across workers
- [ ] Error handling per task
- [ ] Progress reporting
- [ ] Clean shutdown of workers

---

## Technical Specification

### File Location
```
R/models/parallel.R
```

### ParallelTrainer R6 Class

```r
#' @title ParallelTrainer
#' @description Parallel training infrastructure using future package
#'
#' Distributes model training tasks across multiple workers for
#' efficient training of large model ensembles (48 profile models,
#' 216 horizon models, etc.).
#'
#' @export
ParallelTrainer <- R6::R6Class(
  "ParallelTrainer",
  private = list(
    n_workers = NULL,
    plan_type = NULL,
    is_initialized = FALSE,
    progress_callback = NULL
  ),
  public = list(
    #' @description Initialize parallel trainer
    #' @param n_workers Number of parallel workers
    #' @param plan_type Future plan type ("multisession" or "multicore")
    initialize = function(n_workers = parallel::detectCores() - 1,
                          plan_type = "multisession") {
      checkmate::assert_integerish(n_workers, lower = 1, upper = 128)
      checkmate::assert_choice(plan_type, c("multisession", "multicore"))

      private$n_workers <- as.integer(n_workers)
      private$plan_type <- plan_type
      private$is_initialized <- FALSE
    },

    #' @description Set up parallel workers
    #' @return Invisible self
    setup = function() {
      if (private$is_initialized) {
        return(invisible(self))
      }

      message(sprintf("Setting up %d parallel workers (%s)...",
                      private$n_workers, private$plan_type))

      if (private$plan_type == "multisession") {
        future::plan(future::multisession, workers = private$n_workers)
      } else {
        future::plan(future::multicore, workers = private$n_workers)
      }

      private$is_initialized <- TRUE
      invisible(self)
    },

    #' @description Train multiple models in parallel
    #' @param tasks List of training tasks, each with: model, X, y, and optional config
    #' @param ... Additional arguments passed to model$train()
    #' @return List of results with status and model/error per task
    train_parallel = function(tasks, ...) {
      checkmate::assert_list(tasks, min.len = 1)

      # Ensure workers are set up
      self$setup()

      n_tasks <- length(tasks)
      message(sprintf("Starting parallel training of %d models...", n_tasks))

      start_time <- Sys.time()

      # Validate tasks
      for (i in seq_along(tasks)) {
        if (!all(c("model", "X", "y") %in% names(tasks[[i]]))) {
          stop(sprintf("Task %d must contain 'model', 'X', and 'y'", i))
        }
      }

      # Execute in parallel
      results <- future.apply::future_lapply(
        seq_along(tasks),
        function(i) {
          task <- tasks[[i]]
          task_name <- task$name %||% sprintf("task_%d", i)

          tryCatch({
            # Train model
            task$model$train(task$X, task$y, ...)

            list(
              task_id = i,
              name = task_name,
              status = "success",
              model = task$model,
              error = NULL
            )
          }, error = function(e) {
            list(
              task_id = i,
              name = task_name,
              status = "error",
              model = NULL,
              error = conditionMessage(e)
            )
          })
        },
        future.seed = TRUE,
        future.scheduling = 1.0  # Dynamic scheduling
      )

      # Summarize results
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      n_success <- sum(sapply(results, function(r) r$status == "success"))
      n_error <- sum(sapply(results, function(r) r$status == "error"))

      message(sprintf(
        "Parallel training complete: %d/%d successful, %d errors (%.1f sec)",
        n_success, n_tasks, n_error, elapsed
      ))

      # Report errors
      if (n_error > 0) {
        for (r in results) {
          if (r$status == "error") {
            warning(sprintf("Task '%s' failed: %s", r$name, r$error))
          }
        }
      }

      # Fire progress callback if set
      if (!is.null(private$progress_callback)) {
        private$progress_callback(list(
          total = n_tasks,
          success = n_success,
          error = n_error,
          elapsed = elapsed
        ))
      }

      results
    },

    #' @description Train models with shared data (memory efficient)
    #' @param model_factory Function that creates a new model instance
    #' @param X Shared feature matrix
    #' @param y_list List of target vectors, one per model
    #' @param names Optional names for each model
    #' @param ... Additional arguments passed to model$train()
    #' @return List of trained models
    train_shared_data = function(model_factory, X, y_list, names = NULL, ...) {
      checkmate::assert_function(model_factory)
      checkmate::assert_list(y_list, min.len = 1)

      n_models <- length(y_list)
      if (is.null(names)) {
        names <- sprintf("model_%03d", seq_len(n_models))
      }

      self$setup()

      message(sprintf("Training %d models with shared features...", n_models))

      start_time <- Sys.time()

      # Export X to workers once
      results <- future.apply::future_lapply(
        seq_len(n_models),
        function(i) {
          tryCatch({
            model <- model_factory()
            model$train(X, y_list[[i]], ...)

            list(
              task_id = i,
              name = names[i],
              status = "success",
              model = model,
              error = NULL
            )
          }, error = function(e) {
            list(
              task_id = i,
              name = names[i],
              status = "error",
              model = NULL,
              error = conditionMessage(e)
            )
          })
        },
        future.seed = TRUE
      )

      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      n_success <- sum(sapply(results, function(r) r$status == "success"))

      message(sprintf(
        "Training complete: %d/%d successful (%.1f sec, %.2f sec/model)",
        n_success, n_models, elapsed, elapsed / n_models
      ))

      results
    },

    #' @description Predict in parallel across multiple models
    #' @param models List of trained models
    #' @param X Feature matrix for prediction
    #' @param ... Additional arguments passed to model$predict()
    #' @return List of predictions
    predict_parallel = function(models, X, ...) {
      checkmate::assert_list(models, min.len = 1)

      self$setup()

      future.apply::future_lapply(
        models,
        function(model) {
          tryCatch({
            model$predict(X, ...)
          }, error = function(e) {
            warning(sprintf("Prediction failed: %s", conditionMessage(e)))
            NULL
          })
        },
        future.seed = NULL  # Predictions don't need seeds
      )
    },

    #' @description Set progress callback
    #' @param callback Function to call with progress updates
    #' @return Invisible self
    set_progress_callback = function(callback) {
      checkmate::assert_function(callback)
      private$progress_callback <- callback
      invisible(self)
    },

    #' @description Shut down parallel workers
    #' @return Invisible self
    shutdown = function() {
      if (private$is_initialized) {
        message("Shutting down parallel workers...")
        future::plan(future::sequential)
        private$is_initialized <- FALSE
      }
      invisible(self)
    },

    #' @description Get number of workers
    #' @return Integer
    get_n_workers = function() {
      private$n_workers
    },

    #' @description Check if initialized
    #' @return Logical
    is_setup = function() {
      private$is_initialized
    }
  )
)
```

### Integration with MultiModelManager

```r
#' Train MultiModelManager in parallel
#'
#' @param manager MultiModelManager instance
#' @param X Feature matrix
#' @param y Target data (matrix or data.table)
#' @param target_cols Target column names
#' @param n_workers Number of parallel workers
#' @param ... Additional arguments
#' @return MultiModelManager with trained models
#' @export
train_manager_parallel <- function(manager, X, y, target_cols,
                                   n_workers = parallel::detectCores() - 1,
                                   ...) {
  checkmate::assert_class(manager, "MultiModelManager")

  # Get model factory from manager config
  model_factory <- manager$.__enclos_env__$private$model_factory
  if (is.null(model_factory)) {
    stop("Manager must have a model_factory to train in parallel")
  }

  # Prepare target data
  y_dt <- as.data.table(y)
  y_list <- lapply(target_cols, function(col) y_dt[[col]])
  names(y_list) <- target_cols

  # Create parallel trainer
  trainer <- ParallelTrainer$new(n_workers = n_workers)

  # Train
  results <- trainer$train_shared_data(
    model_factory = model_factory,
    X = X,
    y_list = y_list,
    names = target_cols,
    ...
  )

  # Store results in manager
  for (result in results) {
    if (result$status == "success") {
      manager$set_model(result$name, result$model)
    }
  }

  # Cleanup
  trainer$shutdown()

  manager$is_trained <- TRUE
  manager
}
```

### Progress Tracking

```r
#' Create a progress bar callback for parallel training
#'
#' @param total Total number of tasks
#' @return Callback function
#' @export
create_progress_callback <- function(total) {
  pb <- NULL

  function(status) {
    if (is.null(pb)) {
      pb <<- progress::progress_bar$new(
        format = "  Training [:bar] :current/:total (:percent) ETA: :eta",
        total = total,
        clear = FALSE
      )
    }

    # Update progress
    completed <- status$success + status$error
    for (i in seq_len(completed - pb$.__enclos_env__$private$current)) {
      pb$tick()
    }
  }
}


#' Parallel trainer with progress bar
#'
#' @param tasks Training tasks
#' @param n_workers Number of workers
#' @param ... Additional arguments
#' @return List of results
#' @export
train_parallel_with_progress <- function(tasks, n_workers = NULL, ...) {
  n_workers <- n_workers %||% (parallel::detectCores() - 1)

  trainer <- ParallelTrainer$new(n_workers = n_workers)
  trainer$set_progress_callback(create_progress_callback(length(tasks)))

  results <- trainer$train_parallel(tasks, ...)
  trainer$shutdown()

  results
}
```

### Usage Example

```r
# Basic parallel training
trainer <- ParallelTrainer$new(n_workers = 4)
trainer$setup()

# Create training tasks
tasks <- lapply(1:48, function(i) {
  list(
    name = sprintf("profile_%02d", i),
    model = get_model("lightgbm"),
    X = features,
    y = profile_targets[, i]
  )
})

# Train in parallel
results <- trainer$train_parallel(tasks)

# Extract successful models
models <- lapply(
  Filter(function(r) r$status == "success", results),
  function(r) r$model
)

trainer$shutdown()

# Shared data training (more memory efficient)
trainer <- ParallelTrainer$new(n_workers = 8)

y_list <- lapply(1:216, function(i) target_matrix[, i])

results <- trainer$train_shared_data(
  model_factory = function() get_model("lightgbm"),
  X = features,
  y_list = y_list,
  names = sprintf("y_h%03d", 1:216)
)

trainer$shutdown()

# With MultiModelManager
manager <- create_horizon_model_manager("lgbm_horizon", "lightgbm")
manager <- train_manager_parallel(manager, features, targets, n_workers = 8)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | detectCores() - 1 workers |
| TC-002 | Initialize with custom workers | Workers set correctly |
| TC-003 | setup() creates workers | is_setup() returns TRUE |
| TC-004 | train_parallel() success | All tasks complete |
| TC-005 | train_parallel() with errors | Errors captured per task |
| TC-006 | train_parallel() timing | elapsed reported |
| TC-007 | train_shared_data() | Models trained with shared X |
| TC-008 | predict_parallel() | Predictions for all models |
| TC-009 | shutdown() | Sequential plan restored |
| TC-010 | Progress callback | Callback invoked |
| TC-011 | train_manager_parallel() | Manager populated |
| TC-012 | Concurrent access safety | No race conditions |

---

## Dependencies

- PC-016-03: BaseModel
- PC-025-04: MultiModelManager (optional integration)

**External:**
- `future` package
- `future.apply` package
- `progress` package (optional, for progress bars)

---

## Definition of Done

- [ ] ParallelTrainer R6 class implemented
- [ ] train_parallel() distributes tasks
- [ ] train_shared_data() memory efficient
- [ ] Error handling per task
- [ ] Progress reporting
- [ ] shutdown() cleans up workers
- [ ] Integration with MultiModelManager
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use `multisession` for Windows compatibility, `multicore` for Linux/Mac
- Memory usage scales with n_workers × data size
- Consider chunking for very large model counts (1000+)
- Future: add distributed training with `future.batchtools`
- Ensure reproducibility with `future.seed = TRUE`
- Worker startup has overhead; batch small tasks together
