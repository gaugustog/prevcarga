# EPIC-08: Orchestrator

**Duration:** 3 weeks
**Dependencies:** EPIC-07
**Reference:** [MVP Plan R - Phase 8](../mvp-plan-r.md#phase-8-orchestrator-3-weeks)

---

## Objective

Create the workflow orchestration layer that coordinates training, prediction, backtesting, and all system components.

---

## Scope

This epic covers:
- `ConfigManager` for YAML loading and validation
- `TrainWorkflow` for model training orchestration
- `PredictWorkflow` for batch and intraday prediction
- `BacktestWorkflow` for backtesting with retraining evaluation
- `ParallelExecutor` using `future` package
- Structured logging integration

**Out of Scope:**
- CLI commands (EPIC-09)
- Plugin implementations

---

## Tasks

### T-08.1: Create ConfigManager
- [ ] Create `R/orchestrator/config.R`
- [ ] Implement `ConfigManager` R6 class:
  ```r
  ConfigManager <- R6::R6Class(
    "ConfigManager",
    private = list(
      config = NULL,
      config_path = NULL
    ),
    public = list(
      load = function(path) {
        private$config_path <- path
        private$config <- yaml::read_yaml(path)
        self$validate()
        invisible(self)
      },

      validate = function() {
        # Validate required sections
        required <- c("project", "storage", "regions", "models")
        missing <- setdiff(required, names(private$config))
        if (length(missing) > 0) {
          stop(sprintf("Missing config sections: %s", paste(missing, collapse = ", ")))
        }
        invisible(TRUE)
      },

      get = function(key = NULL) {
        if (is.null(key)) return(private$config)
        # Support nested keys: "storage.backend"
        keys <- strsplit(key, "\\.")[[1]]
        result <- private$config
        for (k in keys) {
          result <- result[[k]]
        }
        result
      },

      get_with_default = function(key, default) {
        tryCatch(self$get(key), error = function(e) default)
      }
    )
  )
  ```

### T-08.2: Create TrainWorkflow
- [ ] Create `R/orchestrator/workflows/train.R`
- [ ] Implement `TrainWorkflow` R6 class:
  ```r
  TrainWorkflow <- R6::R6Class(
    "TrainWorkflow",
    private = list(
      config = NULL,
      storage = NULL,
      logger = NULL
    ),
    public = list(
      initialize = function(config, storage, logger = NULL) {
        private$config <- config
        private$storage <- storage
        private$logger <- logger
      },

      run = function(areas, models, start_date, end_date, version, ...) {
        # 1. Load historical data
        loader <- DataLoader$new(private$storage)
        data <- loader$load_carga(areas, start_date, end_date)

        # 2. Parallel training across areas
        n_jobs <- private$config$get("training.parallel.n_jobs") %||% 4
        future::plan(future::multisession, workers = n_jobs)

        results <- future.apply::future_lapply(areas, function(area) {
          area_data <- data[area_code == area]
          self$train_area(area, area_data, models, version)
        }, future.seed = TRUE)

        # 3. Generate training report
        self$generate_report(results)

        results
      },

      train_area = function(area, data, models, version) {
        results <- list()
        for (model_name in models) {
          result <- self$train_single_model(area, data, model_name, version)
          results[[model_name]] <- result
        }
        results
      },

      train_single_model = function(area, data, model_name, version) {
        # Get model from registry
        model <- .model_registry$get(model_name, private$config$get(
          sprintf("models.plugins.%s.config", model_name)
        ))

        # Get feature pipeline
        pipeline <- self$build_feature_pipeline(model_name)

        # Apply features
        features <- pipeline$transform(data)

        # Train
        model$train(features$X, features$y)

        # Save artifact
        artifact <- ModelArtifact$new()
        artifact$model <- model
        artifact$version <- version
        artifact$save(private$storage, get_model_path(model_name, version))

        list(model = model_name, area = area, status = "success")
      },

      build_feature_pipeline = function(model_name) {
        # Build pipeline from config
      },

      generate_report = function(results) {
        # Generate training summary
      }
    )
  )
  ```

### T-08.3: Create PredictWorkflow
- [ ] Create `R/orchestrator/workflows/predict.R`
- [ ] Implement `PredictWorkflow` R6 class:
  ```r
  PredictWorkflow <- R6::R6Class(
    "PredictWorkflow",
    private = list(
      config = NULL,
      storage = NULL
    ),
    public = list(
      run = function(date, mode = "batch", areas = NULL, models = NULL,
                     reconcile = TRUE, combine = TRUE, ...) {
        # 1. Load latest models
        # 2. Load data up to D+0 (or current hour for intraday)
        # 3. Generate predictions for each area/model
        # 4. Combine predictions (if enabled)
        # 5. Reconcile hierarchically (if enabled)
        # 6. Save results
      },

      run_batch = function(date, areas, models, ...) {
        # Full D+0 to D+8 prediction
      },

      run_intraday = function(date, current_hour, areas, models, ...) {
        # Intraday prediction with anchor completion
      }
    )
  )
  ```

### T-08.4: Create BacktestWorkflow
- [ ] Create `R/orchestrator/workflows/backtest.R`
- [ ] Implement `BacktestWorkflow` R6 class:
  ```r
  BacktestWorkflow <- R6::R6Class(
    "BacktestWorkflow",
    public = list(
      run = function(start_date, end_date, models, areas,
                     retrain_intervals = c(1, 3, 7, 10, 15),
                     walk_forward = TRUE, ...) {
        # 1. For each day in period
        # 2. For each retraining interval
        # 3. Simulate training/prediction
        # 4. Calculate metrics
        # 5. Detect drift
        # 6. Generate comparative report
      },

      simulate_day = function(date, models, areas, ...) {
        # Simulate prediction for a single day
      },

      evaluate_retraining_interval = function(interval, results) {
        # Evaluate performance for retraining interval
      },

      recommend_interval = function(results) {
        # Recommend optimal retraining interval
      }
    )
  )
  ```

### T-08.5: Create ParallelExecutor
- [ ] Create `R/orchestrator/parallel.R`
- [ ] Implement `ParallelExecutor` R6 class:
  ```r
  ParallelExecutor <- R6::R6Class(
    "ParallelExecutor",
    private = list(
      n_workers = NULL,
      initialized = FALSE
    ),
    public = list(
      initialize = function(n_workers = 4) {
        private$n_workers <- n_workers
      },

      start = function() {
        future::plan(future::multisession, workers = private$n_workers)
        private$initialized <- TRUE
      },

      stop = function() {
        future::plan(future::sequential)
        private$initialized <- FALSE
      },

      map = function(items, fn, ...) {
        if (!private$initialized) self$start()
        future.apply::future_lapply(items, fn, ..., future.seed = TRUE)
      },

      map_with_progress = function(items, fn, progress_callback = NULL, ...) {
        # Map with progress reporting
      }
    )
  )
  ```

### T-08.6: Enhance StructuredLogger
- [ ] Extend `R/orchestrator/logger.R` from EPIC-00
- [ ] Add workflow context tracking:
  ```r
  StructuredLogger <- R6::R6Class(
    "StructuredLogger",
    private = list(
      context = list(),
      handlers = list()
    ),
    public = list(
      with_context = function(key, value) {
        private$context[[key]] <- value
        invisible(self)
      },

      log = function(level, message, ...) {
        entry <- list(
          timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
          level = level,
          message = message,
          context = private$context,
          extra = list(...)
        )
        self$emit(entry)
      },

      info = function(message, ...) self$log("INFO", message, ...),
      warn = function(message, ...) self$log("WARN", message, ...),
      error = function(message, ...) self$log("ERROR", message, ...),
      debug = function(message, ...) self$log("DEBUG", message, ...),

      emit = function(entry) {
        json <- jsonlite::toJSON(entry, auto_unbox = TRUE)
        for (handler in private$handlers) {
          handler(json)
        }
      }
    )
  )
  ```

### T-08.7: Write Tests
- [ ] Create `tests/testthat/test-orchestrator.R`
- [ ] Test ConfigManager loading and validation
- [ ] Test TrainWorkflow with mock components
- [ ] Test PredictWorkflow
- [ ] Test BacktestWorkflow
- [ ] Test ParallelExecutor

---

## Acceptance Criteria

- [ ] `ConfigManager` loads and validates YAML configuration
- [ ] `TrainWorkflow` orchestrates multi-area, multi-model training
- [ ] `PredictWorkflow` supports batch and intraday modes
- [ ] `BacktestWorkflow` evaluates retraining intervals
- [ ] `ParallelExecutor` distributes work across workers
- [ ] Structured logging captures workflow context
- [ ] Tests pass with ≥80% coverage

---

## Definition of Done

- [ ] All tasks completed
- [ ] Code reviewed and merged
- [ ] Tests passing (≥80% coverage)
- [ ] Documentation complete with roxygen2
- [ ] Integration tested with real workflows

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `R/orchestrator/config.R` | Create | ConfigManager |
| `R/orchestrator/workflows/train.R` | Create | TrainWorkflow |
| `R/orchestrator/workflows/predict.R` | Create | PredictWorkflow |
| `R/orchestrator/workflows/backtest.R` | Create | BacktestWorkflow |
| `R/orchestrator/parallel.R` | Create | ParallelExecutor |
| `R/orchestrator/logger.R` | Modify | Enhance StructuredLogger |
| `tests/testthat/test-orchestrator.R` | Create | Orchestrator tests |
