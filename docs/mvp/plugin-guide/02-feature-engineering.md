# Plugin Guide: Feature Engineering

> **Note:** This documentation will evolve into the project's CONTRIBUTING guide.

## BaseFeaturePlugin Interface

```r
#' @title BaseFeaturePlugin
#' @description Abstract base class for feature plugins
BaseFeaturePlugin <- R6::R6Class(
  "BaseFeaturePlugin",
  public = list(
    name = NULL,
    config = NULL,

    initialize = function(name = NULL, config = list()) {
      self$name <- name
      self$config <- config
    },

    transform = function(dt, ...) {
      stop("transform() must be implemented by subclass")
    },

    get_feature_names = function() {
      stop("get_feature_names() must be implemented by subclass")
    }
  )
)
```

## Feature Types

### 1. Temporal Features
Hour, day, month encoded as cyclical (sin/cos) or categorical.

```r
TemporalFeaturePlugin <- R6::R6Class(
  "TemporalFeaturePlugin",
  inherit = BaseFeaturePlugin,
  public = list(
    transform = function(dt, ...) {
      dt <- copy(dt)

      # Cyclical hour encoding
      dt[, hour_sin := sin(2 * pi * hour(DataHora) / 24)]
      dt[, hour_cos := cos(2 * pi * hour(DataHora) / 24)]

      # Cyclical day of week
      dt[, dow_sin := sin(2 * pi * wday(DataHora) / 7)]
      dt[, dow_cos := cos(2 * pi * wday(DataHora) / 7)]

      dt
    },

    get_feature_names = function() {
      c("hour_sin", "hour_cos", "dow_sin", "dow_cos")
    }
  )
)
```

### 2. Calendar Features
Holidays, weekends, special days (Christmas, New Year).

```r
CalendarFeaturePlugin <- R6::R6Class(
  "CalendarFeaturePlugin",
  inherit = BaseFeaturePlugin,
  public = list(
    transform = function(dt, holidays_dt, ...) {
      dt <- copy(dt)
      dt[, Data := as.Date(DataHora)]

      # Merge holidays
      dt <- merge(dt, holidays_dt[, .(Data, is_holiday = TRUE)],
                  by = "Data", all.x = TRUE)
      dt[is.na(is_holiday), is_holiday := FALSE]

      # Weekend flag
      dt[, is_weekend := wday(DataHora) %in% c(1, 7)]

      dt
    }
  )
)
```

### 3. Lag Features
Previous values (D-1, D-7, same hour lags).

```r
LagFeaturePlugin <- R6::R6Class(
  "LagFeaturePlugin",
  inherit = BaseFeaturePlugin,
  public = list(
    transform = function(dt, ...) {
      dt <- copy(dt)
      setorder(dt, DataHora)

      # Same hour previous day
      dt[, lag_24h := shift(CargaGlobal, 24)]

      # Same hour previous week
      dt[, lag_168h := shift(CargaGlobal, 168)]

      # Rolling statistics
      dt[, rolling_24h_mean := frollmean(CargaGlobal, 24)]

      dt
    }
  )
)
```

### 4. Transformation Features
LOESS smoothing, wavelets, differencing.

```r
WaveletFeaturePlugin <- R6::R6Class(
  "WaveletFeaturePlugin",
  inherit = BaseFeaturePlugin,
  public = list(
    transform = function(dt, ...) {
      dt <- copy(dt)
      window <- self$config$window %||% 32
      filter <- self$config$filter %||% "haar"

      # Apply Haar wavelet decomposition
      wav <- wavelets::modwt(dt$CargaGlobal, filter = filter, n.levels = 4)
      dt[, wav_d1 := wav@W$W1]
      dt[, wav_d2 := wav@W$W2]

      dt
    }
  )
)
```

### 5. Dummy Variables
One-hot encoding for categorical variables.

```r
DummyFeaturePlugin <- R6::R6Class(
  "DummyFeaturePlugin",
  inherit = BaseFeaturePlugin,
  public = list(
    transform = function(dt, ...) {
      dt <- copy(dt)

      # Day of week dummies
      for (d in 1:7) {
        col_name <- paste0("dow_", d)
        dt[, (col_name) := as.integer(wday(DataHora) == d)]
      }

      # Month dummies
      for (m in 1:12) {
        col_name <- paste0("month_", m)
        dt[, (col_name) := as.integer(month(DataHora) == m)]
      }

      dt
    }
  )
)
```

### 6. Normalization
Min-max normalization with stored parameters.

```r
NormalizationPlugin <- R6::R6Class(
  "NormalizationPlugin",
  inherit = BaseFeaturePlugin,
  private = list(
    params = NULL  # Store min/max for inference
  ),
  public = list(
    fit_transform = function(dt, columns, ...) {
      dt <- copy(dt)
      private$params <- list()

      for (col in columns) {
        min_val <- min(dt[[col]], na.rm = TRUE)
        max_val <- max(dt[[col]], na.rm = TRUE)
        private$params[[col]] <- list(min = min_val, max = max_val)

        dt[, (col) := (get(col) - min_val) / (max_val - min_val)]
      }

      dt
    },

    transform = function(dt, columns, ...) {
      dt <- copy(dt)

      for (col in columns) {
        params <- private$params[[col]]
        dt[, (col) := (get(col) - params$min) / (params$max - params$min)]
      }

      dt
    },

    get_params = function() {
      private$params
    },

    set_params = function(params) {
      private$params <- params
    }
  )
)
```

## Feature Pipeline

Compose multiple feature plugins into a pipeline:

```r
FeaturePipeline <- R6::R6Class(
  "FeaturePipeline",
  private = list(
    plugins = list()
  ),
  public = list(
    add = function(plugin) {
      private$plugins <- c(private$plugins, list(plugin))
      invisible(self)
    },

    transform = function(dt, ...) {
      for (plugin in private$plugins) {
        dt <- plugin$transform(dt, ...)
      }
      dt
    },

    get_all_feature_names = function() {
      unlist(lapply(private$plugins, function(p) p$get_feature_names()))
    }
  )
)

# Usage
pipeline <- FeaturePipeline$new()
pipeline$add(TemporalFeaturePlugin$new())
pipeline$add(LagFeaturePlugin$new())
pipeline$add(CalendarFeaturePlugin$new())

features <- pipeline$transform(raw_data, holidays_dt = holidays)
```

## Reference Files for Complex Patterns

| Pattern | Reference File | Key Lines |
|---------|----------------|-----------|
| Wavelet features | `docs/legacy/lgbm/R/02_feature_engineering.R` | 550-579 |
| Calendar/holiday features | `docs/legacy/lgbm/R/02_feature_engineering.R` | 34-166 |
| Lag features (training) | `docs/legacy/lgbm/R/02_feature_engineering.R` | 200-350 |
| Lag features (inference) | `docs/legacy/lgbm/R/04_inference_extractor.R` | 765-980 |

## Next Steps

- [Model Training](03-model-training.md) - Creating model plugins
- [Model Inference](04-model-inference.md) - Inference and prediction
- [Combination Strategies](05-combination.md) - Combining multiple model forecasts
- [Reconciliation](06-reconciliation.md) - Hierarchical reconciliation for SIN
