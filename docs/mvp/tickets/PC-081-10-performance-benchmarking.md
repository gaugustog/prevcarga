# PC-081-10: Performance Benchmarking

**Epic:** [EPIC-10: Testing & Validation](../epics/EPIC-10-testing-validation.md)
**Task Reference:** T-10.6
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Measure and verify that the system meets performance targets for training, prediction, and backtesting operations. Profile critical code paths and identify optimization opportunities.

---

## Acceptance Criteria

- [ ] Performance benchmarking script in `tests/validation/benchmark.R`
- [ ] Training time target: <30 minutes (single area + single model)
- [ ] Intraday prediction target: <5 minutes (full prediction cycle)
- [ ] Backtest target: <8 hours (1-year backtest)
- [ ] Critical code paths profiled
- [ ] Performance report generated

---

## Technical Specification

### File Location
```
tests/validation/benchmark.R
tests/validation/profiling.R
```

### Performance Benchmark Framework

```r
#' @title PerformanceBenchmark
#' @description Measures and validates system performance
#' @export
PerformanceBenchmark <- R6::R6Class(
  "PerformanceBenchmark",
  private = list(
    config = NULL,
    storage = NULL,
    logger = NULL,
    results = list(),
    targets = list(
      training_single_area_model = 30 * 60,   # 30 minutes in seconds
      intraday_prediction = 5 * 60,            # 5 minutes in seconds
      backtest_1_year = 8 * 60 * 60            # 8 hours in seconds
    )
  ),

  public = list(
    #' @description Initialize benchmark
    #' @param config ConfigManager instance
    initialize = function(config) {
      private$config <- config
      private$storage <- create_storage_backend(config$get_storage_config())
      private$logger <- StructuredLogger$new(level = "INFO")
    },

    #' @description Benchmark single area training
    #' @param model_name Model name
    #' @param area Area code
    #' @param train_start Training start date
    #' @param train_end Training end date
    #' @param n_runs Number of benchmark runs
    #' @return Benchmark result
    benchmark_training = function(model_name, area, train_start, train_end,
                                   n_runs = 3) {
      cat(sprintf("\nBenchmarking training: %s on %s\n", model_name, area))
      target <- private$targets$training_single_area_model

      loader <- DataLoader$new(private$storage)
      model_config <- private$config$get_model_config(model_name)

      times <- numeric(n_runs)

      for (i in seq_len(n_runs)) {
        cat(sprintf("  Run %d/%d: ", i, n_runs))

        # Load data
        train_data <- loader$load_carga(
          areas = area,
          start_date = train_start,
          end_date = train_end
        )

        # Build pipeline
        pipeline <- FeaturePipeline$new()
        for (plugin_name in private$config$get("features.enabled")) {
          plugin <- get_feature_plugin(plugin_name)
          pipeline$add_plugin(plugin)
        }

        # Measure training time
        start_time <- Sys.time()

        features <- pipeline$transform(train_data)
        model <- get_model(model_name, model_config$config %||% list())
        model$train(features$X, features$y)

        end_time <- Sys.time()
        elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
        times[i] <- elapsed

        cat(sprintf("%.2f seconds\n", elapsed))
      }

      result <- list(
        operation = "training",
        model = model_name,
        area = area,
        target_seconds = target,
        mean_seconds = mean(times),
        sd_seconds = sd(times),
        min_seconds = min(times),
        max_seconds = max(times),
        all_runs = times,
        passed = mean(times) < target
      )

      private$results$training <- result
      result
    },

    #' @description Benchmark intraday prediction
    #' @param model_name Model name
    #' @param areas Area codes
    #' @param predict_date Prediction date
    #' @param n_runs Number of benchmark runs
    #' @return Benchmark result
    benchmark_intraday_prediction = function(model_name, areas, predict_date,
                                              n_runs = 3) {
      cat(sprintf("\nBenchmarking intraday prediction: %s on %d areas\n",
                  model_name, length(areas)))
      target <- private$targets$intraday_prediction

      times <- numeric(n_runs)

      for (i in seq_len(n_runs)) {
        cat(sprintf("  Run %d/%d: ", i, n_runs))

        start_time <- Sys.time()

        # Full prediction cycle
        workflow <- PredictWorkflow$new(
          config = private$config,
          storage = private$storage
        )

        predictions <- workflow$run(
          model = model_name,
          areas = areas,
          predict_date = predict_date,
          horizons = 0:8
        )

        end_time <- Sys.time()
        elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
        times[i] <- elapsed

        cat(sprintf("%.2f seconds\n", elapsed))
      }

      result <- list(
        operation = "intraday_prediction",
        model = model_name,
        n_areas = length(areas),
        target_seconds = target,
        mean_seconds = mean(times),
        sd_seconds = sd(times),
        min_seconds = min(times),
        max_seconds = max(times),
        all_runs = times,
        passed = mean(times) < target
      )

      private$results$intraday_prediction <- result
      result
    },

    #' @description Benchmark 1-year backtest
    #' @param model_name Model name
    #' @param areas Area codes
    #' @param year Year to backtest
    #' @param retrain_interval Retraining interval
    #' @return Benchmark result
    benchmark_backtest = function(model_name, areas, year = 2024,
                                   retrain_interval = 7) {
      cat(sprintf("\nBenchmarking 1-year backtest: %s on %d areas\n",
                  model_name, length(areas)))
      target <- private$targets$backtest_1_year

      start_time <- Sys.time()

      # Run backtest
      backtest <- BacktestWorkflow$new(
        config = private$config,
        storage = private$storage
      )

      results <- backtest$run(
        start_date = sprintf("%d-01-01", year),
        end_date = sprintf("%d-12-31", year),
        models = model_name,
        areas = areas,
        retrain_intervals = retrain_interval,
        walk_forward = TRUE,
        train_window = 365
      )

      end_time <- Sys.time()
      elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

      result <- list(
        operation = "backtest_1_year",
        model = model_name,
        n_areas = length(areas),
        year = year,
        retrain_interval = retrain_interval,
        target_seconds = target,
        elapsed_seconds = elapsed,
        elapsed_hours = elapsed / 3600,
        passed = elapsed < target
      )

      private$results$backtest <- result
      result
    },

    #' @description Profile critical code paths
    #' @param code_block Code to profile
    #' @param label Label for profile
    #' @return Profiling results
    profile_code = function(code_block, label = "profile") {
      cat(sprintf("\nProfiling: %s\n", label))

      # Use Rprof for profiling
      profile_path <- tempfile(pattern = "rprof_", fileext = ".out")

      Rprof(profile_path, line.profiling = TRUE)
      result <- tryCatch({
        code_block()
      }, finally = {
        Rprof(NULL)
      })

      # Parse profiling results
      if (file.exists(profile_path) && file.info(profile_path)$size > 0) {
        prof_summary <- summaryRprof(profile_path)

        # Top time consumers
        top_by_self <- head(prof_summary$by.self, 10)
        top_by_total <- head(prof_summary$by.total, 10)

        list(
          label = label,
          total_time = sum(prof_summary$by.self$self.time),
          top_by_self = top_by_self,
          top_by_total = top_by_total,
          result = result
        )
      } else {
        list(
          label = label,
          total_time = 0,
          message = "No profiling data collected",
          result = result
        )
      }
    },

    #' @description Run all benchmarks
    #' @param sample_area Sample area for training benchmark
    #' @return Complete benchmark report
    run_all = function(sample_area = "RJ") {
      cat("\n")
      cat("=" |> rep(60) |> paste(collapse = ""))
      cat("\n  Performance Benchmark Suite\n")
      cat("=" |> rep(60) |> paste(collapse = ""))
      cat("\n")

      # Get configuration
      areas <- private$config$get_areas()
      models <- list_models()
      sample_model <- models[1]

      # Training benchmark
      train_result <- self$benchmark_training(
        model_name = sample_model,
        area = sample_area,
        train_start = "2023-01-01",
        train_end = "2023-12-31",
        n_runs = 3
      )

      # Intraday prediction benchmark
      predict_result <- self$benchmark_intraday_prediction(
        model_name = sample_model,
        areas = areas,
        predict_date = "2024-01-15",
        n_runs = 3
      )

      # Note: Full backtest benchmark is typically run separately due to duration
      cat("\n[Note: Full 1-year backtest benchmark should be run separately]\n")

      # Generate report
      self$generate_report()

      list(
        training = train_result,
        intraday_prediction = predict_result,
        timestamp = Sys.time()
      )
    },

    #' @description Generate benchmark report
    generate_report = function() {
      report_path <- sprintf(
        "reports/benchmark_%s.html",
        format(Sys.time(), "%Y%m%d_%H%M%S")
      )

      cat(sprintf("\nGenerating report: %s\n", report_path))

      # Create summary table
      summary_dt <- self$summary()
      print(summary_dt)

      report_path
    },

    #' @description Get summary of benchmark results
    #' @return data.table summary
    summary = function() {
      results_list <- list()

      for (name in names(private$results)) {
        r <- private$results[[name]]

        if (!is.null(r$mean_seconds)) {
          results_list[[name]] <- data.table::data.table(
            operation = r$operation,
            target_sec = r$target_seconds,
            mean_sec = r$mean_seconds,
            passed = r$passed
          )
        } else if (!is.null(r$elapsed_seconds)) {
          results_list[[name]] <- data.table::data.table(
            operation = r$operation,
            target_sec = r$target_seconds,
            mean_sec = r$elapsed_seconds,
            passed = r$passed
          )
        }
      }

      if (length(results_list) > 0) {
        data.table::rbindlist(results_list)
      } else {
        data.table::data.table()
      }
    },

    #' @description Print summary
    print = function() {
      cat("\nPerformance Benchmark Summary\n")
      cat("=============================\n\n")

      summary_dt <- self$summary()

      if (nrow(summary_dt) > 0) {
        print(summary_dt)

        n_passed <- sum(summary_dt$passed)
        n_total <- nrow(summary_dt)

        cat(sprintf("\n%d/%d benchmarks passed\n", n_passed, n_total))
      } else {
        cat("No benchmark results yet.\n")
      }

      invisible(self)
    }
  )
)
```

### Profiler Helper Functions

```r
#' Profile a function call
#'
#' @param fn Function to profile
#' @param ... Arguments to function
#' @param n_times Number of profiling runs
#' @return Profiling summary
#' @export
profile_function <- function(fn, ..., n_times = 5) {
  times <- numeric(n_times)
  memory <- numeric(n_times)

  for (i in seq_len(n_times)) {
    gc()  # Clean up before run
    mem_before <- gc()[2, 2]  # Used Mb

    start <- Sys.time()
    result <- fn(...)
    end <- Sys.time()

    mem_after <- gc()[2, 2]

    times[i] <- as.numeric(difftime(end, start, units = "secs"))
    memory[i] <- mem_after - mem_before
  }

  list(
    mean_time = mean(times),
    sd_time = sd(times),
    mean_memory_mb = mean(memory),
    sd_memory_mb = sd(memory),
    all_times = times,
    all_memory = memory
  )
}


#' Identify performance bottlenecks
#'
#' @param code_expr Code expression to profile
#' @return Bottleneck analysis
#' @export
identify_bottlenecks <- function(code_expr) {
  profile_path <- tempfile(pattern = "bottleneck_", fileext = ".out")

  # Profile with memory
  Rprof(profile_path, memory.profiling = TRUE, line.profiling = TRUE)
  result <- tryCatch(
    eval(code_expr),
    finally = Rprof(NULL)
  )

  if (file.exists(profile_path) && file.info(profile_path)$size > 0) {
    prof <- summaryRprof(profile_path, memory = "both")

    list(
      total_time = prof$sampling.time,
      by_self = head(prof$by.self, 15),
      by_total = head(prof$by.total, 15),
      memory = prof$memory,
      result = result
    )
  } else {
    list(message = "Profiling failed or code executed too quickly")
  }
}
```

### Benchmark Runner Script

```bash
#!/usr/bin/env Rscript
# tests/validation/benchmark.R
#
# Run performance benchmarks

library(prevcargaons)

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)

run_full_backtest <- "--full-backtest" %in% args
sample_area <- "RJ"

if ("--area" %in% args) {
  idx <- which(args == "--area")
  sample_area <- args[idx + 1]
}

# Load config
cat("Loading configuration...\n")
config <- load_config("config/config.yaml")

# Initialize benchmark
benchmark <- PerformanceBenchmark$new(config)

# Run benchmarks
results <- benchmark$run_all(sample_area = sample_area)

# Print summary
print(benchmark)

# Check targets
summary_dt <- benchmark$summary()

if (all(summary_dt$passed)) {
  cat("\n\u2713 All performance targets met!\n")
  quit(status = 0)
} else {
  cat("\n\u2717 Some performance targets not met\n")
  failed <- summary_dt[passed == FALSE, operation]
  for (op in failed) {
    cat(sprintf("  - %s\n", op))
  }
  quit(status = 1)
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Training <30 minutes | Target met |
| TC-002 | Intraday prediction <5 minutes | Target met |
| TC-003 | 1-year backtest <8 hours | Target met |
| TC-004 | Profile training code | Bottlenecks identified |
| TC-005 | Profile prediction code | Bottlenecks identified |
| TC-006 | Multiple runs consistent | SD < 10% of mean |
| TC-007 | Memory profiling | Peak memory recorded |
| TC-008 | Report generated | HTML report exists |
| TC-009 | Summary accurate | Matches results |
| TC-010 | Failed target detection | Correctly flagged |

---

## Dependencies

- PC-058-08: TrainWorkflow
- PC-059-08: PredictWorkflow
- PC-060-08: BacktestWorkflow
- PC-078-10: 1-Year Backtest Execution

---

## Definition of Done

- [ ] PerformanceBenchmark class implemented
- [ ] Training benchmark working
- [ ] Prediction benchmark working
- [ ] Backtest benchmark working
- [ ] Profiling utilities working
- [ ] All performance targets documented
- [ ] Benchmark report generated
- [ ] roxygen2 documentation complete
- [ ] Validation passing
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Run benchmarks on representative hardware
- Consider cloud compute for full backtest
- Profile results help identify optimization targets
- Memory profiling important for large datasets
