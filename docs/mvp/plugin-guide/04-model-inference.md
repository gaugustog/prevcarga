# Plugin Guide: Model Inference

> **Note:** This documentation will evolve into the project's CONTRIBUTING guide.

## Inference vs Training

Inference often differs from training in important ways:

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Data availability** | Full historical data | Only data up to forecast origin |
| **Lag features** | Can use actual future values | Must use forecasted/imputed values |
| **Normalization** | Fit parameters from data | Apply saved parameters |
| **Feature construction** | Batch processing | Point-in-time processing |

## Key Considerations

### 1. Feature Construction Must Match Training

The exact same feature engineering must be applied during inference as during training:

```r
InferenceFeatureBuilder <- R6::R6Class(
  "InferenceFeatureBuilder",
  private = list(
    training_feature_spec = NULL,
    normalization_params = NULL
  ),
  public = list(
    initialize = function(training_artifact) {
      private$training_feature_spec <- training_artifact$feature_spec
      private$normalization_params <- training_artifact$normalization_params
    },

    build_features = function(anchor_data, forecast_horizon, ...) {
      dt <- copy(anchor_data)

      # Apply same transformations as training
      for (transform in private$training_feature_spec$transforms) {
        dt <- self$apply_transform(dt, transform)
      }

      # Apply saved normalization parameters
      for (col in names(private$normalization_params)) {
        params <- private$normalization_params[[col]]
        dt[, (col) := (get(col) - params$min) / (params$max - params$min)]
      }

      dt
    }
  )
)
```

### 2. Handling Lag Features at Inference

During inference, future values for lag features may not exist:

```r
build_inference_lags = function(dt, target_hour) {
  dt <- copy(dt)

  # For inference, we only have data up to the anchor point
  anchor_time <- max(dt$DataHora)

  # Lag 24h: Use actual data if available
  dt[, lag_24h := shift(CargaGlobal, 24)]

  # For missing future lags, use BLF (Base Load Forecast) ratios
  # or same-hour-same-day-of-week from previous week
  dt[is.na(lag_24h), lag_24h := get_blf_estimate(DataHora)]

  dt
}
```

### 3. Normalization Parameter Persistence

Always save and load normalization parameters with the model:

```r
# During training
normalization_params <- list()
for (col in numeric_columns) {
  min_val <- min(train_data[[col]], na.rm = TRUE)
  max_val <- max(train_data[[col]], na.rm = TRUE)
  normalization_params[[col]] <- list(min = min_val, max = max_val)

  train_data[, (col) := (get(col) - min_val) / (max_val - min_val)]
}

# Save with model
saveRDS(list(
  model = model,
  normalization_params = normalization_params
), model_path)

# During inference - apply saved parameters
for (col in names(normalization_params)) {
  params <- normalization_params[[col]]
  inference_data[, (col) := (get(col) - params$min) / (params$max - params$min)]
}
```

### 4. Denormalization of Predictions

If the target was normalized, denormalize predictions:

```r
denormalize_predictions = function(predictions, target_params) {
  predictions * (target_params$max - target_params$min) + target_params$min
}
```

## Inference Workflow

```r
PredictWorkflow <- R6::R6Class(
  "PredictWorkflow",
  public = list(
    run = function(model_artifact, anchor_data, forecast_horizons, ...) {
      # 1. Load model and parameters
      model <- model_artifact$model
      feature_spec <- model_artifact$feature_spec
      norm_params <- model_artifact$normalization_params

      # 2. Build inference features (matching training exactly)
      feature_builder <- InferenceFeatureBuilder$new(model_artifact)
      inference_features <- feature_builder$build_features(anchor_data)

      # 3. Generate predictions
      predictions <- model$predict(inference_features)

      # 4. Denormalize if needed
      if (!is.null(norm_params$target)) {
        predictions <- denormalize_predictions(predictions, norm_params$target)
      }

      # 5. Post-processing (BLF ratios, LOESS smoothing)
      predictions <- self$post_process(predictions, anchor_data)

      predictions
    },

    post_process = function(predictions, anchor_data) {
      # Apply BLF correction if configured
      # Apply LOESS smoothing if configured
      predictions
    }
  )
)
```

## Multi-Model Inference

For models with N models per horizon:

```r
predict_multi = function(X, target_cols = NULL, ...) {
  if (is.null(target_cols)) target_cols <- names(private$models)

  predictions <- matrix(NA, nrow = nrow(X), ncol = length(target_cols))
  colnames(predictions) <- target_cols

  for (i in seq_along(target_cols)) {
    target <- target_cols[i]

    # Get model for this target
    model <- private$models[[target]]

    # Get feature selection for this target
    features <- private$feature_selections[[target]]

    # Subset to selected features
    X_sub <- X[, features, drop = FALSE]

    # Predict
    predictions[, i] <- predict(model, as.matrix(X_sub))
  }

  predictions
}
```

## BLF (Base Load Forecast) Strategy

For intraday forecasting, use BLF ratios to correct predictions:

```r
apply_blf_correction = function(predictions, anchor_load, blf_ratios) {
  # Calculate correction factor from anchor point
  anchor_predicted <- get_anchor_prediction()
  anchor_actual <- anchor_load

  correction_factor <- anchor_actual / anchor_predicted

  # Apply to future predictions with decay
  corrected <- predictions
  for (h in seq_along(predictions)) {
    decay <- exp(-h / 24)  # Decay over 24 hours
    corrected[h] <- predictions[h] * (1 + (correction_factor - 1) * decay)
  }

  corrected
}
```

## Two-Stage Inference Workflow

For D+1 forecasting that requires today's completed data as anchor:

```
Stage A: Generate D+0 forecast (using yesterday as anchor)
    ↓
Stage B: Complete today with ratios (verified + D+0 forecast)
    ↓
Stage C: Generate D+1 forecast (using completed today as anchor)
```

### Implementation Pattern

```r
TwoStageInferenceWorkflow <- R6::R6Class(
  "TwoStageInferenceWorkflow",
  private = list(
    model = NULL,
    feature_builder = NULL
  ),
  public = list(
    run = function(anchor_data, target_date, current_hour, ...) {
      # Stage A: Generate D+0 forecast
      d0_forecast <- self$generate_d0_forecast(anchor_data, target_date)

      # Stage B: Complete the day with ratios
      completed_anchor <- self$complete_day_with_ratios(
        anchor_data,
        d0_forecast,
        current_hour
      )

      # Stage C: Generate D+1 forecast with completed anchor
      d1_forecast <- self$generate_d1_forecast(completed_anchor, target_date + 1)

      list(
        d0 = d0_forecast,
        d1 = d1_forecast,
        anchor = completed_anchor
      )
    },

    generate_d0_forecast = function(anchor_data, target_date) {
      # Build features using yesterday as complete anchor
      features <- private$feature_builder$build_features(
        anchor_data,
        target_date = target_date
      )
      private$model$predict(features)
    },

    generate_d1_forecast = function(completed_anchor, target_date) {
      # Build features using completed today as anchor
      features <- private$feature_builder$build_features(
        completed_anchor,
        target_date = target_date
      )
      private$model$predict(features)
    }
  )
)
```

### Day Completion with Ratio Propagation

Complete partial day by combining verified data with forecast using ratio propagation:

```r
complete_day_with_ratios = function(verified_data, d0_forecast, current_hour) {
  target_date <- as.Date(max(verified_data$DataHora))

  # Split into verified (past) and forecasted (future) periods
  verified_hours <- verified_data[as.Date(DataHora) == target_date]

  # Calculate ratio from last verified period
  last_verified <- tail(verified_hours, 1)
  last_forecast_at_verified <- d0_forecast[hour == hour(last_verified$DataHora)]

  ratio <- last_verified$CargaGlobal / last_forecast_at_verified$predicted

  # Apply ratio to remaining forecast periods with decay
  remaining_hours <- 24 - current_hour
  completed <- copy(d0_forecast)

  for (h in seq_len(remaining_hours)) {
    target_hour <- current_hour + h
    decay_factor <- exp(-h / 12)  # Decay over 12 hours
    adjusted_ratio <- 1 + (ratio - 1) * decay_factor

    completed[hour == target_hour,
              CargaGlobal := predicted * adjusted_ratio]
  }

  # Combine verified with adjusted forecast
  rbind(
    verified_hours[, .(DataHora, CargaGlobal, source = "verified")],
    completed[hour > current_hour,
              .(DataHora, CargaGlobal, source = "forecasted")]
  )
}
```

### Handling Verified vs Forecasted Periods

Track data provenance for downstream reconciliation:

```r
InferenceResult <- R6::R6Class(
  "InferenceResult",
  public = list(
    predictions = NULL,
    metadata = NULL,

    initialize = function(predictions, anchor_datetime, current_hour) {
      self$predictions <- predictions
      self$metadata <- list(
        anchor_datetime = anchor_datetime,
        current_hour = current_hour,
        verified_through = anchor_datetime + lubridate::hours(current_hour),
        forecast_start = anchor_datetime + lubridate::hours(current_hour + 1)
      )
    },

    get_verified_period = function() {
      self$predictions[source == "verified"]
    },

    get_forecasted_period = function() {
      self$predictions[source == "forecasted"]
    }
  )
)
```

## Batch Inference for Multiple Areas

Process multiple areas efficiently:

```r
BatchInferenceRunner <- R6::R6Class(
  "BatchInferenceRunner",
  private = list(
    model_registry = NULL,
    storage = NULL
  ),
  public = list(
    run_batch = function(areas, target_date, config, ...) {
      # Parallel execution across areas
      future::plan(future::multisession, workers = config$n_jobs %||% 4)

      results <- future.apply::future_lapply(areas, function(area) {
        tryCatch({
          # Load area-specific model
          model <- private$model_registry$get(area)

          # Load anchor data for area
          anchor <- private$storage$read_parquet(
            file.path("load", paste0("area_code=", area))
          )

          # Run inference workflow
          workflow <- TwoStageInferenceWorkflow$new(model = model)
          result <- workflow$run(anchor, target_date)

          list(area = area, status = "success", result = result)
        }, error = function(e) {
          list(area = area, status = "error", message = conditionMessage(e))
        })
      }, future.seed = TRUE)

      # Aggregate results
      self$aggregate_results(results)
    },

    aggregate_results = function(results) {
      successes <- Filter(function(r) r$status == "success", results)
      failures <- Filter(function(r) r$status == "error", results)

      if (length(failures) > 0) {
        warning(sprintf("%d areas failed: %s",
                        length(failures),
                        paste(sapply(failures, `[[`, "area"), collapse = ", ")))
      }

      list(
        predictions = rbindlist(lapply(successes, function(r) {
          r$result$d1[, area_code := r$area]
        })),
        failures = failures
      )
    }
  )
)
```

## Model Loading

```r
load_model_for_inference = function(area, model_name, version = "latest",
                                     storage) {
  if (version == "latest") {
    model_path <- storage$read_symlink(
      file.path("models", model_name, "latest")
    )
  } else {
    model_path <- file.path("models", model_name, version, "model.rds")
  }

  artifact <- storage$read_rds(model_path)

  list(
    model = artifact$model,
    feature_spec = artifact$feature_spec,
    normalization_params = artifact$normalization_params,
    metadata = artifact$metadata
  )
}
```

## Reference Files

| Pattern | Reference File | Key Lines |
|---------|----------------|-----------|
| Inference lag features | `docs/legacy/lgbm/R/04_inference_extractor.R` | 765-980 |
| Normalization | `docs/legacy/lgbm/R/04_inference_extractor.R` | 1220-1235 |
| BLF correction | `docs/legacy/lgbm/R/04_inference_extractor.R` | 100-200 |
| Two-stage workflow | `docs/legacy/lgbm/R/05a_inference_workflow.R` | 1-200 |
| Day completion with ratios | `docs/legacy/lgbm/R/05a_inference_workflow.R` | 250-350 |
| Batch inference | `docs/legacy/lgbm/R/05a_inference_workflow.R` | 400-500 |

## Checklist for Inference Implementation

- [ ] Feature construction matches training exactly
- [ ] Normalization parameters loaded from artifact
- [ ] Lag features handle missing future values
- [ ] Predictions denormalized if target was normalized
- [ ] Post-processing applied (BLF, LOESS) if needed
- [ ] Model versioning respected
- [ ] Error handling for missing data
- [ ] Two-stage workflow for D+1 forecasting (if applicable)
- [ ] Ratio propagation for day completion
- [ ] Verified vs forecasted period tracking
- [ ] Batch inference parallelization

## Complete Example

```r
# Load model artifact
artifact <- load_model_for_inference(
  area = "RJ",
  model_name = "lightgbm",
  version = "latest",
  storage = backend
)

# Load anchor data (most recent available)
anchor_data <- loader$load_carga(
  areas = "RJ",
  start_date = Sys.Date() - 30,
  end_date = Sys.Date()
)

# Generate forecast
workflow <- PredictWorkflow$new()
forecast <- workflow$run(
  model_artifact = artifact,
  anchor_data = anchor_data,
  forecast_horizons = 0:8
)

# Save results
storage$write_parquet(
  forecast,
  file.path("results/predictions", format(Sys.time(), "%Y%m%d_%H%M%S"), "RJ.parquet")
)
```

## Next Steps

- [Combination Strategies](05-combination.md) - Combining multiple model forecasts
- [Reconciliation](06-reconciliation.md) - Hierarchical reconciliation for SIN
