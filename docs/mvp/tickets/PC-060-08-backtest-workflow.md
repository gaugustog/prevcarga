# PC-060-08: BacktestWorkflow

**Epic:** [EPIC-08: Orchestrator](../epics/EPIC-08-orchestrator.md)
**Task Reference:** T-08.4
**Priority:** High
**Estimated Effort:** 2.5 days

---

## Summary

Implement the `BacktestWorkflow` R6 class for conducting backtests with walk-forward validation and retraining interval evaluation. This workflow simulates production prediction scenarios on historical data.

---

## Acceptance Criteria

- [ ] Workflow module created in `R/orchestrator/workflows/backtest.R`
- [ ] `BacktestWorkflow` R6 class with full orchestration
- [ ] Walk-forward validation support
- [ ] Retraining interval comparison
- [ ] Drift detection integration
- [ ] Optimal retraining recommendation
- [ ] Comprehensive backtest report

---

## Technical Specification

### File Location
```
R/orchestrator/workflows/backtest.R
```

### BacktestWorkflow R6 Class

```r
#' @title BacktestWorkflow
#' @description Orchestrates backtesting with retraining evaluation
#'
#' Conducts walk-forward backtesting to:
#' 1. Simulate production prediction on historical data
#' 2. Evaluate different retraining intervals
#' 3. Detect model drift
#' 4. Recommend optimal retraining frequency
#'
#' @export
BacktestWorkflow <- R6::R6Class(
  "BacktestWorkflow",
  private = list(
    config = NULL,
    storage = NULL,
    logger = NULL,
    executor = NULL,
    drift_detector = NULL
  ),
  public = list(
    #' @description Initialize backtest workflow
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

      # Initialize drift detector
      private$drift_detector <- DriftDetector$new(
        threshold_mape = config$get_with_default("evaluation.drift_detection.threshold_mape", 1.2)
      )
    },

    #' @description Run backtest workflow
    #' @param start_date Backtest start date
    #' @param end_date Backtest end date
    #' @param models Character vector of model names
    #' @param areas Character vector of area codes
    #' @param retrain_intervals Retraining intervals to evaluate (in days)
    #' @param walk_forward Use walk-forward validation
    #' @param initial_train_days Days of initial training data
    #' @param ... Additional arguments
    #' @return BacktestResult with evaluation metrics
    run = function(start_date,
                   end_date,
                   models,
                   areas = NULL,
                   retrain_intervals = c(1, 3, 7, 14, 30),
                   walk_forward = TRUE,
                   initial_train_days = 365,
                   ...) {
      start_time <- Sys.time()

      private$logger$with_context("workflow", "backtest")
      private$logger$info("Starting backtest workflow",
                          start_date = start_date,
                          end_date = end_date)

      areas <- areas %||% private$config$get_areas()

      # Validate inputs
      self$validate_inputs(start_date, end_date, initial_train_days)

      # Load all historical data
      private$logger$info("Loading historical data...")
      all_data <- self$load_all_data(start_date, end_date, areas, initial_train_days)

      # Run backtest for each retraining interval
      results_by_interval <- list()

      for (interval in retrain_intervals) {
        private$logger$info(sprintf("Evaluating retraining interval: %d days", interval))

        result <- self$run_interval(
          all_data = all_data,
          start_date = start_date,
          end_date = end_date,
          models = models,
          areas = areas,
          retrain_interval = interval,
          walk_forward = walk_forward,
          initial_train_days = initial_train_days
        )

        results_by_interval[[as.character(interval)]] <- result
      }

      # Compare intervals and generate recommendation
      private$logger$info("Analyzing results and generating recommendation...")
      comparison <- self$compare_intervals(results_by_interval)
      recommendation <- self$recommend_interval(comparison)

      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      private$logger$info("Backtest workflow completed",
                          elapsed_secs = elapsed,
                          recommended_interval = recommendation$interval)

      BacktestResult$new(
        results_by_interval = results_by_interval,
        comparison = comparison,
        recommendation = recommendation,
        start_date = start_date,
        end_date = end_date,
        models = models,
        areas = areas,
        elapsed_time = elapsed
      )
    },

    #' @description Validate backtest inputs
    validate_inputs = function(start_date, end_date, initial_train_days) {
      start <- as.Date(start_date)
      end <- as.Date(end_date)

      if (start >= end) {
        stop("start_date must be before end_date")
      }

      if (as.integer(end - start) < 30) {
        warning("Backtest period is less than 30 days - results may not be reliable")
      }

      invisible(TRUE)
    },

    #' @description Load all historical data for backtest
    #' @param start_date Backtest start
    #' @param end_date Backtest end
    #' @param areas Area codes
    #' @param initial_train_days Initial training period
    #' @return data.table with all data
    load_all_data = function(start_date, end_date, areas, initial_train_days) {
      # Need data from initial_train_days before start_date
      data_start <- as.Date(start_date) - initial_train_days

      loader <- DataLoader$new(private$storage)
      loader$load_carga(areas, data_start, end_date)
    },

    #' @description Run backtest for single retraining interval
    #' @param all_data All historical data
    #' @param start_date Backtest start
    #' @param end_date Backtest end
    #' @param models Model names
    #' @param areas Area codes
    #' @param retrain_interval Days between retraining
    #' @param walk_forward Use walk-forward
    #' @param initial_train_days Initial training days
    #' @return List with interval results
    run_interval = function(all_data,
                            start_date,
                            end_date,
                            models,
                            areas,
                            retrain_interval,
                            walk_forward,
                            initial_train_days) {
      start <- as.Date(start_date)
      end <- as.Date(end_date)

      # Generate simulation dates
      sim_dates <- seq(start, end, by = "day")

      # Track when to retrain
      last_train_date <- start - initial_train_days
      next_retrain <- start

      # Store results
      all_predictions <- list()
      all_actuals <- list()
      training_events <- list()

      for (current_date in sim_dates) {
        current_date <- as.Date(current_date, origin = "1970-01-01")

        # Check if retraining needed
        if (current_date >= next_retrain) {
          private$logger$debug("Retraining on", date = current_date)

          # Train models with data up to current_date
          train_end <- current_date - 1
          train_start <- if (walk_forward) {
            max(last_train_date, current_date - initial_train_days)
          } else {
            as.Date(start_date) - initial_train_days
          }

          train_data <- all_data[as.Date(DataHora) >= train_start &
                                 as.Date(DataHora) <= train_end]

          # Train all models for all areas
          models_trained <- self$train_models(train_data, models, areas)

          training_events[[as.character(current_date)]] <- list(
            date = current_date,
            train_start = train_start,
            train_end = train_end,
            n_rows = nrow(train_data)
          )

          last_train_date <- train_end
          next_retrain <- current_date + retrain_interval
        }

        # Generate predictions for current_date
        pred_data <- all_data[as.Date(DataHora) < current_date]
        actual_data <- all_data[as.Date(DataHora) == current_date]

        predictions <- self$predict_day(
          pred_data,
          actual_data,
          current_date,
          models,
          areas,
          models_trained
        )

        all_predictions[[as.character(current_date)]] <- predictions
        all_actuals[[as.character(current_date)]] <- actual_data
      }

      # Calculate metrics
      metrics <- self$calculate_backtest_metrics(all_predictions, all_actuals)

      list(
        retrain_interval = retrain_interval,
        predictions = all_predictions,
        actuals = all_actuals,
        training_events = training_events,
        metrics = metrics,
        n_retrains = length(training_events)
      )
    },

    #' @description Train models for backtest simulation
    #' @param train_data Training data
    #' @param models Model names
    #' @param areas Area codes
    #' @return List of trained models
    train_models = function(train_data, models, areas) {
      trained <- list()

      for (area in areas) {
        area_data <- train_data[area_code == area]
        trained[[area]] <- list()

        for (model_name in models) {
          # Get model config
          model_config <- private$config$get_model_config(model_name)

          # Create and train model
          model <- get_model(model_name, config = model_config$config %||% list())

          # Build features
          pipeline <- self$build_feature_pipeline(model_name)
          features <- pipeline$transform(area_data)

          # Train
          model$train(features$X, features$y)

          trained[[area]][[model_name]] <- list(
            model = model,
            pipeline = pipeline
          )
        }
      }

      trained
    },

    #' @description Predict for single day in backtest
    predict_day = function(pred_data, actual_data, date, models, areas, trained_models) {
      predictions <- list()

      for (area in areas) {
        predictions[[area]] <- list()
        area_pred_data <- pred_data[area_code == area]

        for (model_name in models) {
          trained <- trained_models[[area]][[model_name]]

          # Transform features
          features <- trained$pipeline$transform(area_pred_data)

          # Predict
          preds <- trained$model$predict(features$X)

          predictions[[area]][[model_name]] <- preds
        }
      }

      predictions
    },

    #' @description Build feature pipeline
    build_feature_pipeline = function(model_name) {
      feature_config <- private$config$get("features.plugins")
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

    #' @description Calculate backtest metrics
    #' @param all_predictions All predictions
    #' @param all_actuals All actuals
    #' @return data.table with metrics
    calculate_backtest_metrics = function(all_predictions, all_actuals) {
      # Calculate MAPE, MAE, RMSE for entire backtest period
      metrics_list <- list()

      for (date in names(all_predictions)) {
        predictions <- all_predictions[[date]]
        actuals <- all_actuals[[date]]

        for (area in names(predictions)) {
          for (model in names(predictions[[area]])) {
            pred <- predictions[[area]][[model]]
            actual <- actuals[area_code == area, carga]

            if (length(pred) == length(actual) && length(actual) > 0) {
              m <- calculate_all_metrics(actual, pred)

              metrics_list[[length(metrics_list) + 1]] <- data.table::data.table(
                date = as.Date(date),
                area_code = area,
                model = model,
                mape = m$mape,
                mae = m$mae,
                rmse = m$rmse
              )
            }
          }
        }
      }

      data.table::rbindlist(metrics_list)
    },

    #' @description Compare retraining intervals
    #' @param results_by_interval Results for each interval
    #' @return data.table with comparison
    compare_intervals = function(results_by_interval) {
      comparison <- lapply(names(results_by_interval), function(interval) {
        result <- results_by_interval[[interval]]
        metrics <- result$metrics

        data.table::data.table(
          retrain_interval = as.integer(interval),
          n_retrains = result$n_retrains,
          mean_mape = mean(metrics$mape, na.rm = TRUE),
          median_mape = median(metrics$mape, na.rm = TRUE),
          p95_mape = quantile(metrics$mape, 0.95, na.rm = TRUE),
          mean_mae = mean(metrics$mae, na.rm = TRUE),
          mean_rmse = mean(metrics$rmse, na.rm = TRUE)
        )
      })

      data.table::rbindlist(comparison)
    },

    #' @description Recommend optimal retraining interval
    #' @param comparison Interval comparison table
    #' @return List with recommendation
    recommend_interval = function(comparison) {
      # Find interval with best trade-off:
      # - Lower MAPE is better
      # - Fewer retrains is better (cost consideration)

      # Normalize metrics
      comparison <- data.table::copy(comparison)
      comparison[, mape_norm := (mean_mape - min(mean_mape)) /
                               (max(mean_mape) - min(mean_mape) + 1e-6)]
      comparison[, retrain_norm := (n_retrains - min(n_retrains)) /
                                   (max(n_retrains) - min(n_retrains) + 1e-6)]

      # Combined score (lower is better)
      # Weight MAPE more than retraining cost
      comparison[, score := 0.7 * mape_norm + 0.3 * retrain_norm]

      # Find best
      best <- comparison[which.min(score)]

      list(
        interval = best$retrain_interval,
        expected_mape = best$mean_mape,
        expected_retrains = best$n_retrains,
        score = best$score,
        rationale = sprintf(
          "Interval of %d days provides MAPE of %.2f%% with %d retrains",
          best$retrain_interval,
          best$mean_mape,
          best$n_retrains
        )
      )
    },

    #' @description Print summary
    print = function() {
      cat("BacktestWorkflow\n")
      cat(sprintf("  Default intervals: 1, 3, 7, 14, 30 days\n"))
      invisible(self)
    }
  )
)
```

### BacktestResult Class

```r
#' @title BacktestResult
#' @description Container for backtest workflow results
#' @export
BacktestResult <- R6::R6Class(
  "BacktestResult",
  public = list(
    results_by_interval = NULL,
    comparison = NULL,
    recommendation = NULL,
    start_date = NULL,
    end_date = NULL,
    models = NULL,
    areas = NULL,
    elapsed_time = NULL,

    initialize = function(results_by_interval, comparison, recommendation,
                          start_date, end_date, models, areas, elapsed_time) {
      self$results_by_interval <- results_by_interval
      self$comparison <- comparison
      self$recommendation <- recommendation
      self$start_date <- start_date
      self$end_date <- end_date
      self$models <- models
      self$areas <- areas
      self$elapsed_time <- elapsed_time
    },

    #' @description Get metrics for interval
    get_interval_metrics = function(interval) {
      self$results_by_interval[[as.character(interval)]]$metrics
    },

    #' @description Get recommended interval
    get_recommendation = function() {
      self$recommendation
    },

    #' @description Print result
    print = function() {
      cat("BacktestResult\n")
      cat(sprintf("  Period: %s to %s\n", self$start_date, self$end_date))
      cat(sprintf("  Models: %s\n", paste(self$models, collapse = ", ")))
      cat(sprintf("  Areas: %d\n", length(self$areas)))
      cat(sprintf("  Elapsed: %.2f sec\n", self$elapsed_time))
      cat("\nRecommendation:\n")
      cat(sprintf("  %s\n", self$recommendation$rationale))

      cat("\nInterval Comparison:\n")
      print(self$comparison)

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

workflow <- BacktestWorkflow$new(config, storage)

# Run backtest
result <- workflow$run(
  start_date = "2024-01-01",
  end_date = "2024-03-31",
  models = c("lgbm", "rf"),
  areas = c("RJ", "SP"),
  retrain_intervals = c(1, 7, 14, 30),
  walk_forward = TRUE
)

print(result)
# BacktestResult
#   Period: 2024-01-01 to 2024-03-31
#   Models: lgbm, rf
#   Areas: 2
#   Elapsed: 1234.56 sec
#
# Recommendation:
#   Interval of 7 days provides MAPE of 3.45% with 13 retrains
#
# Interval Comparison:
#    retrain_interval n_retrains mean_mape median_mape
# 1:                1         90      3.20        3.15
# 2:                7         13      3.45        3.40
# 3:               14          7      3.85        3.78
# 4:               30          3      4.50        4.35

# Get recommendation
rec <- result$get_recommendation()
# $interval: 7
# $expected_mape: 3.45
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | run() basic backtest | BacktestResult |
| TC-002 | run_interval() | Interval results |
| TC-003 | train_models() | Models trained |
| TC-004 | predict_day() | Day predictions |
| TC-005 | calculate_backtest_metrics() | Metrics table |
| TC-006 | compare_intervals() | Comparison table |
| TC-007 | recommend_interval() | Recommendation |
| TC-008 | Walk-forward vs expanding | Different results |
| TC-009 | Short period warning | Warning issued |
| TC-010 | Multiple intervals | All evaluated |

---

## Dependencies

- PC-057-08: ConfigManager
- PC-061-08: ParallelExecutor
- PC-050-07: DriftDetector
- PC-047-07: MetricsCalculator

---

## Definition of Done

- [ ] BacktestWorkflow R6 class implemented
- [ ] BacktestResult container implemented
- [ ] Walk-forward validation working
- [ ] Retraining interval comparison
- [ ] Recommendation algorithm
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Walk-forward uses expanding window by default
- Retraining is computationally expensive
- Consider adding early stopping if MAPE degrades
- May add parallelization across intervals
