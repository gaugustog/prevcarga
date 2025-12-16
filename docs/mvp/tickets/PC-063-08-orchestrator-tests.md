# PC-063-08: Orchestrator Tests

**Epic:** [EPIC-08: Orchestrator](../epics/EPIC-08-orchestrator.md)
**Task Reference:** T-08.7
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Create comprehensive test suite for the orchestrator module covering ConfigManager, all workflow classes, ParallelExecutor, and StructuredLogger with mocked dependencies.

---

## Acceptance Criteria

- [ ] Test file created at `tests/testthat/test-orchestrator.R`
- [ ] ConfigManager tests with valid/invalid configs
- [ ] TrainWorkflow tests with mocked components
- [ ] PredictWorkflow tests for batch and intraday modes
- [ ] BacktestWorkflow tests with walk-forward simulation
- [ ] ParallelExecutor tests with parallel execution
- [ ] StructuredLogger tests for all levels and handlers
- [ ] Test coverage ≥80% for orchestrator module

---

## Technical Specification

### File Location
```
tests/testthat/test-orchestrator.R
```

### Test Configuration Fixtures

```r
# Helper: Create test config YAML
create_test_config <- function(tmp_dir) {
  config_content <- '
project:
  name: "PrevCarga-Test"
  version: "1.0.0"

storage:
  backend: "local"
  local_cache: true
  cache_dir: ".test_cache"

regions:
  national: "SIN"
  areas:
    - RJ
    - SP
    - MG

models:
  default: "lgbm"
  plugins:
    lgbm:
      enabled: true
      config:
        num_leaves: 31
        learning_rate: 0.05
    rf:
      enabled: true
      config:
        num.trees: 100

features:
  plugins:
    calendar:
      enabled: true
    lags:
      enabled: true
      config:
        lags: [1, 2, 24]

training:
  parallel:
    enabled: true
    n_jobs: 2

prediction:
  horizons: [0, 1, 2]
  reconciliation: true

evaluation:
  metrics: [mape, mae, rmse]

logging:
  level: "DEBUG"
  format: "json"
'
  config_path <- file.path(tmp_dir, "config.yaml")
  writeLines(config_content, config_path)
  config_path
}


# Helper: Create mock training data
create_mock_data <- function(areas = c("RJ", "SP"), days = 30) {
  dates <- seq(Sys.Date() - days, Sys.Date() - 1, by = "day")
  hours <- 0:23

  data.table::rbindlist(lapply(areas, function(area) {
    data.table::data.table(
      datetime = as.POSIXct(
        paste(rep(dates, each = 24), sprintf("%02d:00:00", hours)),
        tz = "America/Sao_Paulo"
      ),
      area_code = area,
      carga_mwmed = rnorm(length(dates) * 24, mean = 5000, sd = 500),
      temperature = rnorm(length(dates) * 24, mean = 25, sd = 5)
    )
  }))
}


# Helper: Create mock model
create_mock_model <- function() {
  model <- list(
    trained = FALSE,
    train = function(X, y) {
      self$trained <- TRUE
      invisible(self)
    },
    predict = function(X) {
      rep(5000, nrow(X))
    },
    get_training_metrics = function() {
      list(train_rmse = 100)
    }
  )
  class(model) <- c("MockModel", "BaseModel")
  model
}
```

### ConfigManager Tests

```r
describe("ConfigManager", {
  tmp_dir <- NULL
  config_path <- NULL

  setup({
    tmp_dir <<- tempdir()
    config_path <<- create_test_config(tmp_dir)
  })

  teardown({
    unlink(config_path)
  })

  it("loads valid YAML configuration", {
    config <- ConfigManager$new()
    config$load(config_path)

    expect_true(config$has("project.name"))
    expect_equal(config$get("project.name"), "PrevCarga-Test")
  })

  it("throws error for missing file", {
    config <- ConfigManager$new()
    expect_error(config$load("/nonexistent/config.yaml"))
  })

  it("validates required sections", {
    invalid_config <- file.path(tmp_dir, "invalid.yaml")
    writeLines("project:\n  name: test", invalid_config)

    config <- ConfigManager$new()
    expect_error(
      config$load(invalid_config),
      "Missing required configuration sections"
    )
  })

  it("supports nested key access", {
    config <- ConfigManager$new()
    config$load(config_path)

    expect_equal(config$get("models.default"), "lgbm")
    expect_equal(config$get("models.plugins.lgbm.config.num_leaves"), 31)
  })

  it("returns default for missing keys", {
    config <- ConfigManager$new()
    config$load(config_path)

    expect_null(config$get("nonexistent.key"))
    expect_equal(config$get_with_default("nonexistent.key", "default"), "default")
  })

  it("returns areas list", {
    config <- ConfigManager$new()
    config$load(config_path)

    areas <- config$get_areas()
    expect_equal(areas, c("RJ", "SP", "MG"))
  })

  it("interpolates environment variables", {
    # Set env var
    withr::with_envvar(c("TEST_BUCKET" = "my-bucket"), {
      env_config <- file.path(tmp_dir, "env_config.yaml")
      writeLines('
project:
  name: test
storage:
  backend: s3
  bucket: "${TEST_BUCKET}"
regions:
  areas: [RJ]
models:
  default: lgbm
', env_config)

      config <- ConfigManager$new()
      config$load(env_config)
      expect_equal(config$get("storage.bucket"), "my-bucket")
    })
  })

  it("applies environment overrides", {
    env_config <- file.path(tmp_dir, "env_override.yaml")
    writeLines('
project:
  name: test
storage:
  backend: local
  cache_dir: ".cache"
regions:
  areas: [RJ]
models:
  default: lgbm
environments:
  development:
    storage:
      cache_dir: ".dev_cache"
', env_config)

    config <- ConfigManager$new(environment = "development")
    config$load(env_config)
    expect_equal(config$get("storage.cache_dir"), ".dev_cache")
  })
})
```

### TrainWorkflow Tests

```r
describe("TrainWorkflow", {
  tmp_dir <- NULL
  config <- NULL
  mock_storage <- NULL

  setup({
    tmp_dir <<- tempdir()
    config_path <- create_test_config(tmp_dir)

    config <<- ConfigManager$new()
    config$load(config_path)

    # Create mock storage
    mock_storage <<- list(
      save = function(data, path) invisible(TRUE),
      load = function(path) NULL,
      exists = function(path) FALSE
    )
    class(mock_storage) <- "LocalStorageBackend"
  })

  it("initializes with config and storage", {
    workflow <- TrainWorkflow$new(config, mock_storage)
    expect_s3_class(workflow, "TrainWorkflow")
  })

  it("validates input areas", {
    workflow <- TrainWorkflow$new(config, mock_storage)

    expect_error(
      workflow$validate_inputs(c("INVALID"), c("lgbm"), "2024-01-01", "2024-06-01"),
      "Unknown areas"
    )
  })

  it("validates date range", {
    workflow <- TrainWorkflow$new(config, mock_storage)

    expect_error(
      workflow$validate_inputs(c("RJ"), c("lgbm"), "2024-06-01", "2024-01-01"),
      "start_date must be before end_date"
    )
  })

  it("builds feature pipeline from config", {
    workflow <- TrainWorkflow$new(config, mock_storage)

    # Mock feature registry
    with_mock(
      get_feature_plugin = function(name, cfg) {
        list(transform = function(data) data)
      },
      {
        pipeline <- workflow$build_feature_pipeline("lgbm")
        expect_s3_class(pipeline, "FeaturePipeline")
      }
    )
  })

  it("generates artifact path correctly", {
    workflow <- TrainWorkflow$new(config, mock_storage)

    path <- workflow$get_artifact_path("RJ", "lgbm", "v1.0.0")
    expect_equal(path, "models/lgbm/RJ/v1.0.0/model.rds")
  })

  it("generates training report", {
    workflow <- TrainWorkflow$new(config, mock_storage)

    results <- list(
      RJ_lgbm = list(area = "RJ", model = "lgbm", status = "success", elapsed_secs = 10),
      SP_lgbm = list(area = "SP", model = "lgbm", status = "failed", error = "test error")
    )

    report <- workflow$generate_report(results, "v1.0.0")

    expect_equal(report$summary$total, 2)
    expect_equal(report$summary$success, 1)
    expect_equal(report$summary$failed, 1)
    expect_equal(report$summary$success_rate, 0.5)
  })
})
```

### PredictWorkflow Tests

```r
describe("PredictWorkflow", {
  config <- NULL
  mock_storage <- NULL

  setup({
    tmp_dir <- tempdir()
    config_path <- create_test_config(tmp_dir)

    config <<- ConfigManager$new()
    config$load(config_path)

    mock_storage <<- list(
      save = function(data, path) invisible(TRUE),
      load = function(path) NULL
    )
    class(mock_storage) <- "LocalStorageBackend"
  })

  it("initializes correctly", {
    workflow <- PredictWorkflow$new(config, mock_storage)
    expect_s3_class(workflow, "PredictWorkflow")
  })

  it("validates prediction mode", {
    workflow <- PredictWorkflow$new(config, mock_storage)

    expect_error(
      workflow$validate_inputs("2024-01-01", mode = "invalid"),
      "Invalid mode"
    )
  })

  it("calculates horizons for batch mode", {
    workflow <- PredictWorkflow$new(config, mock_storage)

    horizons <- workflow$get_horizons("batch")
    expect_equal(horizons, 0:2)  # From config
  })

  it("calculates horizons for intraday mode", {
    workflow <- PredictWorkflow$new(config, mock_storage)

    horizons <- workflow$get_horizons("intraday", current_hour = 10)
    expect_true(all(horizons >= 0))
  })

  it("generates prediction output path", {
    workflow <- PredictWorkflow$new(config, mock_storage)

    path <- workflow$get_output_path("2024-01-17", "batch")
    expect_match(path, "predictions/batch/2024-01-17")
  })
})
```

### BacktestWorkflow Tests

```r
describe("BacktestWorkflow", {
  config <- NULL

  setup({
    tmp_dir <- tempdir()
    config_path <- create_test_config(tmp_dir)

    config <<- ConfigManager$new()
    config$load(config_path)
  })

  it("initializes with config", {
    workflow <- BacktestWorkflow$new(config)
    expect_s3_class(workflow, "BacktestWorkflow")
  })

  it("generates backtest dates", {
    workflow <- BacktestWorkflow$new(config)

    dates <- workflow$generate_dates("2024-01-01", "2024-01-10")
    expect_equal(length(dates), 10)
  })

  it("calculates retraining schedule", {
    workflow <- BacktestWorkflow$new(config)

    schedule <- workflow$get_retraining_schedule(
      dates = seq(as.Date("2024-01-01"), as.Date("2024-01-15"), by = "day"),
      interval = 7
    )

    expect_true(as.Date("2024-01-01") %in% schedule)
    expect_true(as.Date("2024-01-08") %in% schedule)
  })

  it("aggregates results by interval", {
    workflow <- BacktestWorkflow$new(config)

    results <- list(
      list(date = "2024-01-01", interval = 7, mape = 3.5),
      list(date = "2024-01-02", interval = 7, mape = 3.8),
      list(date = "2024-01-01", interval = 15, mape = 4.2),
      list(date = "2024-01-02", interval = 15, mape = 4.5)
    )

    aggregated <- workflow$aggregate_by_interval(results)

    expect_equal(length(aggregated), 2)  # 2 intervals
    expect_true("7" %in% names(aggregated))
    expect_true("15" %in% names(aggregated))
  })

  it("recommends optimal interval", {
    workflow <- BacktestWorkflow$new(config)

    results_by_interval <- list(
      "7" = list(mean_mape = 3.5, std_mape = 0.3),
      "15" = list(mean_mape = 4.2, std_mape = 0.5)
    )

    recommendation <- workflow$recommend_interval(results_by_interval)
    expect_equal(recommendation$interval, 7)
  })
})
```

### ParallelExecutor Tests

```r
describe("ParallelExecutor", {
  it("initializes with default workers", {
    executor <- ParallelExecutor$new()
    expect_equal(executor$get_n_workers(), 4)
    expect_false(executor$is_initialized())
  })

  it("initializes with custom workers", {
    executor <- ParallelExecutor$new(n_workers = 2)
    expect_equal(executor$get_n_workers(), 2)
  })

  it("starts and stops workers", {
    executor <- ParallelExecutor$new(n_workers = 2)

    executor$start()
    expect_true(executor$is_initialized())

    executor$stop()
    expect_false(executor$is_initialized())
  })

  it("maps function over items", {
    executor <- ParallelExecutor$new(n_workers = 2)
    on.exit(executor$stop())

    results <- executor$map(1:4, function(x) x^2)

    expect_equal(results, list(1, 4, 9, 16))
  })

  it("handles errors in mapped function", {
    executor <- ParallelExecutor$new(n_workers = 2)
    on.exit(executor$stop())

    expect_error(
      executor$map(1:4, function(x) {
        if (x == 2) stop("test error")
        x^2
      }),
      "Parallel execution failed"
    )
  })

  it("batch maps large task sets", {
    executor <- ParallelExecutor$new(n_workers = 2)
    on.exit(executor$stop())

    results <- executor$batch_map(
      tasks = as.list(1:10),
      batch_size = 3,
      fn = function(x) x * 2
    )

    expect_equal(length(results), 10)
    expect_equal(results[[5]], 10)
  })

  it("falls back to multisession on Windows for multicore", {
    skip_on_os("windows")  # Skip this specific test on Windows

    executor <- ParallelExecutor$new(n_workers = 2, backend = "multicore")
    executor$start()
    on.exit(executor$stop())

    expect_true(executor$is_initialized())
  })
})
```

### StructuredLogger Tests

```r
describe("StructuredLogger", {
  it("initializes with default settings", {
    logger <- StructuredLogger$new()

    expect_equal(logger$get_level(), "INFO")
    expect_true(nchar(logger$get_session_id()) > 0)
  })

  it("sets and removes context", {
    logger <- StructuredLogger$new()

    logger$with_context("workflow", "train")
    expect_equal(logger$get_context()$workflow, "train")

    logger$remove_context("workflow")
    expect_null(logger$get_context()$workflow)
  })

  it("clears all context", {
    logger <- StructuredLogger$new()

    logger$with_context("a", 1)
    logger$with_context("b", 2)
    logger$clear_context()

    expect_equal(length(logger$get_context()), 0)
  })

  it("respects log levels", {
    logger <- StructuredLogger$new(level = "WARN")

    # Capture output
    output <- capture.output({
      logger$debug("debug message")
      logger$info("info message")
      logger$warn("warn message")
    })

    expect_false(any(grepl("debug message", output)))
    expect_false(any(grepl("info message", output)))
    expect_true(any(grepl("warn message", output)))
  })

  it("outputs valid JSON", {
    logger <- StructuredLogger$new(format = "json")

    output <- capture.output({
      logger$info("test message", key = "value")
    })

    # Parse JSON
    parsed <- jsonlite::fromJSON(output[1])
    expect_equal(parsed$message, "test message")
    expect_equal(parsed$level, "INFO")
    expect_equal(parsed$extra$key, "value")
  })

  it("outputs formatted text", {
    logger <- StructuredLogger$new(format = "text")
    logger$with_context("ctx", "val")

    output <- capture.output({
      logger$info("test message")
    })

    expect_match(output[1], "INFO")
    expect_match(output[1], "test message")
    expect_match(output[1], "ctx=val")
  })

  it("creates file handler", {
    logger <- StructuredLogger$new()
    tmp_file <- tempfile(fileext = ".log")
    on.exit(unlink(tmp_file))

    handler <- logger$create_file_handler(tmp_file)
    logger$add_handler("file", handler)

    logger$info("file log message")

    expect_true(file.exists(tmp_file))
    content <- readLines(tmp_file)
    expect_match(content[1], "file log message")
  })

  it("times execution", {
    logger <- StructuredLogger$new()

    output <- capture.output({
      result <- logger$timed("test operation", {
        Sys.sleep(0.1)
        42
      })
    })

    expect_equal(result, 42)
    expect_match(paste(output, collapse = "\n"), "Starting: test operation")
    expect_match(paste(output, collapse = "\n"), "Completed: test operation")
    expect_match(paste(output, collapse = "\n"), "elapsed_secs")
  })

  it("logs errors in timed execution", {
    logger <- StructuredLogger$new()

    expect_error({
      logger$timed("failing operation", {
        stop("test error")
      })
    }, "test error")
  })
})
```

### Integration Tests

```r
describe("Orchestrator Integration", {
  it("workflows integrate with ConfigManager", {
    tmp_dir <- tempdir()
    config_path <- create_test_config(tmp_dir)

    config <- ConfigManager$new()
    config$load(config_path)

    logger <- StructuredLogger$new(level = "ERROR")  # Suppress output

    mock_storage <- list(
      save = function(data, path) invisible(TRUE)
    )
    class(mock_storage) <- "LocalStorageBackend"

    # Create workflows
    train_workflow <- TrainWorkflow$new(config, mock_storage, logger)
    predict_workflow <- PredictWorkflow$new(config, mock_storage, logger)
    backtest_workflow <- BacktestWorkflow$new(config, logger)

    expect_s3_class(train_workflow, "TrainWorkflow")
    expect_s3_class(predict_workflow, "PredictWorkflow")
    expect_s3_class(backtest_workflow, "BacktestWorkflow")
  })

  it("parallel executor works with workflows", {
    executor <- ParallelExecutor$new(n_workers = 2)
    on.exit(executor$stop())

    # Simulate parallel training tasks
    tasks <- list(
      list(area = "RJ", model = "lgbm"),
      list(area = "SP", model = "lgbm")
    )

    results <- executor$map(tasks, function(task) {
      list(
        area = task$area,
        model = task$model,
        status = "success"
      )
    })

    expect_equal(length(results), 2)
    expect_equal(results[[1]]$status, "success")
  })
})
```

---

## Test Cases Summary

| Component | Test Count | Coverage Target |
|-----------|------------|-----------------|
| ConfigManager | 10 | 90% |
| TrainWorkflow | 8 | 85% |
| PredictWorkflow | 6 | 80% |
| BacktestWorkflow | 6 | 80% |
| ParallelExecutor | 8 | 85% |
| StructuredLogger | 12 | 90% |
| Integration | 3 | N/A |
| **Total** | **53** | **≥80%** |

---

## Dependencies

- PC-057-08: ConfigManager
- PC-058-08: TrainWorkflow
- PC-059-08: PredictWorkflow
- PC-060-08: BacktestWorkflow
- PC-061-08: ParallelExecutor
- PC-062-08: StructuredLogger

---

## Definition of Done

- [ ] All test files created
- [ ] ConfigManager tests passing
- [ ] TrainWorkflow tests passing
- [ ] PredictWorkflow tests passing
- [ ] BacktestWorkflow tests passing
- [ ] ParallelExecutor tests passing
- [ ] StructuredLogger tests passing
- [ ] Integration tests passing
- [ ] Coverage ≥80% for orchestrator module
- [ ] CI pipeline passing
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use `testthat` describe/it style for BDD approach
- Mock external dependencies (storage, models) using local mocks
- Use `withr` for temporary environment changes
- Tests should be fast (<30 seconds total)
- Consider adding property-based tests with `hedgehog` for edge cases
