# PC-061-08: ParallelExecutor

**Epic:** [EPIC-08: Orchestrator](../epics/EPIC-08-orchestrator.md)
**Task Reference:** T-08.5
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `ParallelExecutor` R6 class for distributing work across multiple workers using the `future` and `future.apply` packages. This component abstracts parallel execution patterns for training, prediction, and backtesting workflows.

---

## Acceptance Criteria

- [ ] Parallel module created in `R/orchestrator/parallel.R`
- [ ] `ParallelExecutor` R6 class with worker management
- [ ] Support for `multisession` and `multicore` backends
- [ ] Progress reporting capability
- [ ] Graceful shutdown and error handling
- [ ] Resource cleanup on completion

---

## Technical Specification

### File Location
```
R/orchestrator/parallel.R
```

### ParallelExecutor R6 Class

```r
#' @title ParallelExecutor
#' @description Parallel execution manager using future package
#'
#' Provides abstraction over parallel execution backends with
#' progress reporting and error handling capabilities.
#'
#' @export
ParallelExecutor <- R6::R6Class(
  "ParallelExecutor",
  private = list(
    n_workers = NULL,
    backend = NULL,
    initialized = FALSE,
    original_plan = NULL
  ),
  public = list(
    #' @description Initialize parallel executor
    #' @param n_workers Number of parallel workers
    #' @param backend Backend type: "multisession" or "multicore"
    initialize = function(n_workers = 4, backend = "multisession") {
      checkmate::assert_int(n_workers, lower = 1)
      checkmate::assert_choice(backend, c("multisession", "multicore", "sequential"))

      private$n_workers <- n_workers
      private$backend <- backend
    },

    #' @description Start parallel workers
    #' @return Invisible self
    start = function() {
      if (private$initialized) {
        return(invisible(self))
      }

      # Save current plan for restoration
      private$original_plan <- future::plan()

      # Set up parallel backend
      if (private$backend == "multisession") {
        future::plan(future::multisession, workers = private$n_workers)
      } else if (private$backend == "multicore") {
        # multicore not available on Windows
        if (.Platform$OS.type == "windows") {
          warning("multicore not available on Windows, using multisession")
          future::plan(future::multisession, workers = private$n_workers)
        } else {
          future::plan(future::multicore, workers = private$n_workers)
        }
      } else {
        future::plan(future::sequential)
      }

      private$initialized <- TRUE
      invisible(self)
    },

    #' @description Stop parallel workers
    #' @return Invisible self
    stop = function() {
      if (!private$initialized) {
        return(invisible(self))
      }

      # Restore original plan
      if (!is.null(private$original_plan)) {
        future::plan(private$original_plan)
      } else {
        future::plan(future::sequential)
      }

      private$initialized <- FALSE
      invisible(self)
    },

    #' @description Map function over items in parallel
    #' @param items Vector or list of items to process
    #' @param fn Function to apply to each item
    #' @param ... Additional arguments passed to fn
    #' @return List of results
    map = function(items, fn, ...) {
      if (!private$initialized) {
        self$start()
      }

      tryCatch({
        future.apply::future_lapply(
          items, fn, ...,
          future.seed = TRUE,
          future.scheduling = 1.0
        )
      }, error = function(e) {
        self$stop()
        stop(sprintf("Parallel execution failed: %s", conditionMessage(e)))
      })
    },

    #' @description Map with progress reporting
    #' @param items Items to process
    #' @param fn Function to apply
    #' @param progress_callback Function(completed, total) for progress
    #' @param ... Additional arguments passed to fn
    #' @return List of results
    map_with_progress = function(items, fn, progress_callback = NULL, ...) {
      if (!private$initialized) {
        self$start()
      }

      n_items <- length(items)

      if (is.null(progress_callback)) {
        progress_callback <- function(completed, total) {
          message(sprintf("Progress: %d/%d (%.1f%%)",
                          completed, total, 100 * completed / total))
        }
      }

      # Use progressr for progress reporting
      if (requireNamespace("progressr", quietly = TRUE)) {
        progressr::with_progress({
          p <- progressr::progressor(steps = n_items)

          future.apply::future_lapply(items, function(item) {
            result <- fn(item, ...)
            p()
            result
          }, future.seed = TRUE)
        })
      } else {
        # Fallback without progressr
        results <- list()
        for (i in seq_along(items)) {
          results[[i]] <- fn(items[[i]], ...)
          progress_callback(i, n_items)
        }
        results
      }
    },

    #' @description Execute tasks in parallel batches
    #' @param tasks List of task specifications
    #' @param batch_size Number of tasks per batch
    #' @param fn Function to apply to each task
    #' @return List of results
    batch_map = function(tasks, batch_size, fn, ...) {
      n_tasks <- length(tasks)
      n_batches <- ceiling(n_tasks / batch_size)

      results <- list()

      for (batch_idx in seq_len(n_batches)) {
        start_idx <- (batch_idx - 1) * batch_size + 1
        end_idx <- min(batch_idx * batch_size, n_tasks)

        batch_tasks <- tasks[start_idx:end_idx]
        batch_results <- self$map(batch_tasks, fn, ...)

        results <- c(results, batch_results)
      }

      results
    },

    #' @description Execute with timeout
    #' @param items Items to process
    #' @param fn Function to apply
    #' @param timeout_secs Timeout in seconds per item
    #' @param ... Additional arguments
    #' @return List of results with timeout handling
    map_with_timeout = function(items, fn, timeout_secs = 300, ...) {
      if (!private$initialized) {
        self$start()
      }

      future.apply::future_lapply(
        items,
        function(item) {
          tryCatch({
            R.utils::withTimeout(
              fn(item, ...),
              timeout = timeout_secs,
              onTimeout = "error"
            )
          }, TimeoutException = function(e) {
            list(
              status = "timeout",
              error = sprintf("Task timed out after %d seconds", timeout_secs)
            )
          }, error = function(e) {
            list(
              status = "error",
              error = conditionMessage(e)
            )
          })
        },
        future.seed = TRUE
      )
    },

    #' @description Get number of workers
    #' @return Integer number of workers
    get_n_workers = function() {
      private$n_workers
    },

    #' @description Check if initialized
    #' @return Logical
    is_initialized = function() {
      private$initialized
    },

    #' @description Get backend type
    #' @return Character backend name
    get_backend = function() {
      private$backend
    },

    #' @description Print executor info
    print = function() {
      cat("ParallelExecutor\n")
      cat(sprintf("  Workers: %d\n", private$n_workers))
      cat(sprintf("  Backend: %s\n", private$backend))
      cat(sprintf("  Status: %s\n", if (private$initialized) "running" else "stopped"))
      invisible(self)
    }
  )
)
```

### Convenience Functions

```r
#' Create parallel executor from config
#'
#' @param config ConfigManager instance
#' @return ParallelExecutor instance
#' @export
create_parallel_executor <- function(config) {
  n_workers <- config$get_with_default("training.parallel.n_jobs", 4)
  backend <- config$get_with_default("training.parallel.backend", "multisession")

  ParallelExecutor$new(n_workers = n_workers, backend = backend)
}


#' Execute function in parallel
#'
#' Convenience function for one-off parallel execution
#'
#' @param items Items to process
#' @param fn Function to apply
#' @param n_workers Number of workers
#' @param ... Additional arguments
#' @return List of results
#' @export
parallel_map <- function(items, fn, n_workers = 4, ...) {
  executor <- ParallelExecutor$new(n_workers = n_workers)
  on.exit(executor$stop())

  executor$start()
  executor$map(items, fn, ...)
}


#' Detect optimal number of workers
#'
#' @param max_workers Maximum allowed workers
#' @return Integer number of workers
#' @export
detect_n_workers <- function(max_workers = 8) {
  n_cores <- parallel::detectCores(logical = FALSE)
  min(max_workers, max(1, n_cores - 1))
}
```

### Usage Example

```r
# Initialize executor
executor <- ParallelExecutor$new(n_workers = 4, backend = "multisession")

# Start workers
executor$start()

# Simple parallel map
results <- executor$map(1:10, function(x) {
  Sys.sleep(0.5)  # Simulate work
  x^2
})
# Returns: list(1, 4, 9, 16, 25, 36, 49, 64, 81, 100)

# With progress reporting
results <- executor$map_with_progress(
  items = 1:20,
  fn = function(x) {
    Sys.sleep(0.3)
    x * 2
  },
  progress_callback = function(done, total) {
    cat(sprintf("\r%d/%d complete", done, total))
  }
)

# Batch processing
tasks <- list(
  list(area = "RJ", model = "lgbm"),
  list(area = "SP", model = "lgbm"),
  list(area = "RJ", model = "rf"),
  list(area = "SP", model = "rf")
)

results <- executor$batch_map(
  tasks = tasks,
  batch_size = 2,
  fn = function(task) {
    # Train model for area
    train_model(task$area, task$model)
  }
)

# With timeout
results <- executor$map_with_timeout(
  items = areas,
  fn = train_area,
  timeout_secs = 600  # 10 minute timeout
)

# Stop workers
executor$stop()

# Using convenience function
results <- parallel_map(1:100, function(x) x^2, n_workers = 4)

# Auto-detect workers
n_workers <- detect_n_workers(max_workers = 8)
# Returns optimal number based on CPU cores
```

### Integration with Workflows

```r
# TrainWorkflow usage
TrainWorkflow <- R6::R6Class(
  "TrainWorkflow",
  public = list(
    train_all = function(data, areas, models, version) {
      # Create task grid
      tasks <- expand.grid(
        area = areas,
        model = models,
        stringsAsFactors = FALSE
      )

      # Execute in parallel
      private$executor$map(
        seq_len(nrow(tasks)),
        function(i) {
          area <- tasks$area[i]
          model <- tasks$model[i]
          self$train_single(area, data[area_code == area], model, version)
        }
      )
    }
  )
)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | Executor created |
| TC-002 | Initialize with custom workers | Workers set correctly |
| TC-003 | start() creates workers | Status = running |
| TC-004 | stop() terminates workers | Status = stopped |
| TC-005 | map() simple function | Results returned |
| TC-006 | map() with errors | Errors propagated |
| TC-007 | map_with_progress() | Progress reported |
| TC-008 | batch_map() processing | Batches processed |
| TC-009 | map_with_timeout() | Timeout handled |
| TC-010 | detect_n_workers() | Reasonable count |
| TC-011 | multicore on Windows | Falls back to multisession |
| TC-012 | Resource cleanup | Plan restored on stop |

---

## Dependencies

None (foundation module, uses future package)

---

## Definition of Done

- [ ] ParallelExecutor R6 class implemented
- [ ] Support for multisession and multicore backends
- [ ] Progress reporting with progressr integration
- [ ] Timeout handling for long-running tasks
- [ ] Batch processing for large task sets
- [ ] Resource cleanup on stop
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Uses `future` package for cross-platform parallelism
- `multicore` not available on Windows, falls back to `multisession`
- Consider adding task queue for dynamic load balancing
- Progress reporting requires `progressr` package (optional)
- Always clean up workers with `stop()` or `on.exit()`
