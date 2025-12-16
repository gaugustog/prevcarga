# Plugin Guide: Combination Strategies

> **Note:** This documentation will evolve into the project's CONTRIBUTING guide.

## Overview

Combination strategies aggregate forecasts from multiple models to produce more robust predictions. This is particularly important when different models excel at different horizons or conditions.

## BaseCombiner Interface

```r
#' @title BaseCombiner
#' @description Abstract base class for forecast combination strategies
BaseCombiner <- R6::R6Class(
  "BaseCombiner",
  public = list(
    name = NULL,
    config = NULL,

    initialize = function(name = NULL, config = list()) {
      self$name <- name
      self$config <- config
    },

    combine = function(forecasts, weights = NULL, ...) {
      stop("combine() must be implemented by subclass")
    },

    fit_weights = function(forecasts, actuals, ...) {
      stop("fit_weights() must be implemented by subclass")
    },

    get_weights = function() {
      stop("get_weights() must be implemented by subclass")
    }
  )
)
```

## Combination Strategies

### 1. Simple Average

Equal-weight combination of all models:

```r
SimpleAverageCombiner <- R6::R6Class(
  "SimpleAverageCombiner",
  inherit = BaseCombiner,
  public = list(
    combine = function(forecasts, ...) {
      # forecasts: list of data.tables with predictions
      n_models <- length(forecasts)

      # Stack all forecasts
      combined <- rbindlist(forecasts, idcol = "model_id")

      # Average by timestamp
      combined[, .(
        prediction = mean(prediction),
        n_models = .N
      ), by = .(DataHora, area_code)]
    },

    fit_weights = function(forecasts, actuals, ...) {
      # Simple average has equal weights
      n_models <- length(forecasts)
      rep(1 / n_models, n_models)
    }
  )
)
```

### 2. Weighted Average

Fixed weights per model:

```r
WeightedAverageCombiner <- R6::R6Class(
  "WeightedAverageCombiner",
  inherit = BaseCombiner,
  private = list(
    weights = NULL
  ),
  public = list(
    initialize = function(weights = NULL, ...) {
      super$initialize(...)
      private$weights <- weights
    },

    combine = function(forecasts, weights = NULL, ...) {
      w <- weights %||% private$weights
      stopifnot(length(w) == length(forecasts))

      # Normalize weights
      w <- w / sum(w)

      # Weighted combination
      n <- nrow(forecasts[[1]])
      result <- data.table(
        DataHora = forecasts[[1]]$DataHora,
        area_code = forecasts[[1]]$area_code,
        prediction = 0
      )

      for (i in seq_along(forecasts)) {
        result[, prediction := prediction + w[i] * forecasts[[i]]$prediction]
      }

      result
    },

    fit_weights = function(forecasts, actuals, method = "inverse_mse", ...) {
      # Calculate MSE for each model
      mse <- sapply(forecasts, function(f) {
        merged <- merge(f, actuals, by = c("DataHora", "area_code"))
        mean((merged$prediction - merged$CargaGlobal)^2, na.rm = TRUE)
      })

      # Inverse MSE weights
      private$weights <- (1 / mse) / sum(1 / mse)
      private$weights
    },

    get_weights = function() {
      private$weights
    }
  )
)
```

### 3. Optimal Combination (OLS-based)

Estimate optimal weights via constrained regression:

```r
OptimalCombiner <- R6::R6Class(
  "OptimalCombiner",
  inherit = BaseCombiner,
  private = list(
    weights = NULL,
    intercept = 0
  ),
  public = list(
    fit_weights = function(forecasts, actuals, constrained = TRUE, ...) {
      # Build matrix of predictions
      n <- nrow(forecasts[[1]])
      k <- length(forecasts)

      X <- matrix(NA, nrow = n, ncol = k)
      for (i in seq_len(k)) {
        X[, i] <- forecasts[[i]]$prediction
      }

      y <- actuals$CargaGlobal

      if (constrained) {
        # Constrained: weights sum to 1, no intercept
        # Use quadratic programming
        Dmat <- t(X) %*% X
        dvec <- t(X) %*% y
        Amat <- cbind(rep(1, k), diag(k))  # Sum to 1, non-negative
        bvec <- c(1, rep(0, k))

        sol <- quadprog::solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
        private$weights <- sol$solution
      } else {
        # Unconstrained OLS
        fit <- lm(y ~ X)
        private$intercept <- coef(fit)[1]
        private$weights <- coef(fit)[-1]
      }

      private$weights
    },

    combine = function(forecasts, ...) {
      n <- nrow(forecasts[[1]])
      result <- data.table(
        DataHora = forecasts[[1]]$DataHora,
        area_code = forecasts[[1]]$area_code,
        prediction = private$intercept
      )

      for (i in seq_along(forecasts)) {
        result[, prediction := prediction + private$weights[i] * forecasts[[i]]$prediction]
      }

      result
    }
  )
)
```

### 4. Horizon-Specific Combination

Different weights per forecast horizon:

```r
HorizonCombiner <- R6::R6Class(
  "HorizonCombiner",
  inherit = BaseCombiner,
  private = list(
    weights_by_horizon = list()  # {horizon: weights}
  ),
  public = list(
    fit_weights = function(forecasts, actuals, horizons = 0:8, ...) {
      for (h in horizons) {
        # Filter to specific horizon
        horizon_forecasts <- lapply(forecasts, function(f) {
          f[horizon == h]
        })
        horizon_actuals <- actuals[horizon == h]

        # Fit weights for this horizon
        combiner <- OptimalCombiner$new()
        private$weights_by_horizon[[as.character(h)]] <- combiner$fit_weights(
          horizon_forecasts,
          horizon_actuals
        )
      }

      private$weights_by_horizon
    },

    combine = function(forecasts, ...) {
      results <- list()

      for (h in names(private$weights_by_horizon)) {
        weights <- private$weights_by_horizon[[h]]

        # Filter to horizon
        horizon_forecasts <- lapply(forecasts, function(f) {
          f[horizon == as.integer(h)]
        })

        # Combine with horizon-specific weights
        combined <- data.table(
          DataHora = horizon_forecasts[[1]]$DataHora,
          area_code = horizon_forecasts[[1]]$area_code,
          horizon = as.integer(h),
          prediction = 0
        )

        for (i in seq_along(horizon_forecasts)) {
          combined[, prediction := prediction + weights[i] * horizon_forecasts[[i]]$prediction]
        }

        results[[h]] <- combined
      }

      rbindlist(results)
    }
  )
)
```

### 5. Model Selection (Winner-Takes-All)

Select best model per area/horizon based on recent performance:

```r
ModelSelectorCombiner <- R6::R6Class(
  "ModelSelectorCombiner",
  inherit = BaseCombiner,
  private = list(
    selected_models = list()  # {area_horizon: model_idx}
  ),
  public = list(
    fit_weights = function(forecasts, actuals, lookback_days = 30, ...) {
      # Get unique area-horizon combinations
      areas <- unique(forecasts[[1]]$area_code)
      horizons <- unique(forecasts[[1]]$horizon)

      for (area in areas) {
        for (h in horizons) {
          key <- paste(area, h, sep = "_")

          # Calculate recent MAPE for each model
          mape <- sapply(seq_along(forecasts), function(i) {
            f <- forecasts[[i]][area_code == area & horizon == h]
            a <- actuals[area_code == area & horizon == h]
            merged <- merge(f, a, by = "DataHora")

            # Only recent data
            recent <- merged[DataHora >= max(DataHora) - lookback_days]
            mean(abs((recent$prediction - recent$CargaGlobal) / recent$CargaGlobal)) * 100
          })

          # Select best model
          private$selected_models[[key]] <- which.min(mape)
        }
      }

      private$selected_models
    },

    combine = function(forecasts, ...) {
      results <- list()

      for (key in names(private$selected_models)) {
        parts <- strsplit(key, "_")[[1]]
        area <- parts[1]
        horizon <- as.integer(parts[2])
        model_idx <- private$selected_models[[key]]

        # Select from best model
        selected <- forecasts[[model_idx]][area_code == area & horizon == horizon]
        results[[key]] <- selected
      }

      rbindlist(results)
    }
  )
)
```

## Combiner Registry

Register and retrieve combination strategies:

```r
CombinerRegistry <- R6::R6Class(
  "CombinerRegistry",
  private = list(
    combiners = list()
  ),
  public = list(
    register = function(name, combiner_class) {
      private$combiners[[name]] <- combiner_class
      invisible(self)
    },

    get = function(name, ...) {
      if (!name %in% names(private$combiners)) {
        stop(sprintf("Combiner '%s' not registered", name))
      }
      private$combiners[[name]]$new(...)
    },

    list_combiners = function() {
      names(private$combiners)
    }
  )
)

# Default registry
default_combiner_registry <- function() {
  registry <- CombinerRegistry$new()
  registry$register("simple_average", SimpleAverageCombiner)
  registry$register("weighted_average", WeightedAverageCombiner)
  registry$register("optimal", OptimalCombiner)
  registry$register("horizon", HorizonCombiner)
  registry$register("model_selection", ModelSelectorCombiner)
  registry
}
```

## Combination Workflow

```r
CombinationWorkflow <- R6::R6Class(
  "CombinationWorkflow",
  private = list(
    combiner = NULL,
    model_forecasts = list()
  ),
  public = list(
    initialize = function(combiner, ...) {
      private$combiner <- combiner
    },

    run = function(model_results, validation_data = NULL, ...) {
      # Extract forecasts from model results
      forecasts <- lapply(model_results, function(r) r$predictions)

      # Fit weights if validation data provided
      if (!is.null(validation_data)) {
        private$combiner$fit_weights(forecasts, validation_data)
      }

      # Combine forecasts
      combined <- private$combiner$combine(forecasts)

      list(
        predictions = combined,
        weights = private$combiner$get_weights(),
        metadata = list(
          n_models = length(forecasts),
          combiner = class(private$combiner)[1]
        )
      )
    }
  )
)
```

## Configuration

```yaml
combination:
  strategy: weighted_average  # simple_average, weighted_average, optimal, horizon, model_selection

  models:
    - name: lightgbm
      weight: 0.6
    - name: random_forest
      weight: 0.3
    - name: holt_winters
      weight: 0.1

  fit_weights:
    enabled: true
    method: inverse_mse  # inverse_mse, optimal, per_horizon
    lookback_days: 30

  horizon_specific:
    enabled: false
    horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]
```

## Reference Files

| Pattern | Reference File | Key Lines |
|---------|----------------|-----------|
| Weighted combination | `docs/legacy/randon-forest/4_previsao_multipla.R` | TBD |
| Model selection | `docs/legacy/lgbm/R/06_model_selection.R` | TBD |

## Checklist for Combination Implementation

- [ ] Implement BaseCombiner interface
- [ ] Weight normalization (sum to 1)
- [ ] Handle missing forecasts gracefully
- [ ] Validation data for weight fitting
- [ ] Per-horizon weights (if needed)
- [ ] Serialize weights with model artifacts
- [ ] Register combiner in registry

## Next Steps

- [Reconciliation](06-reconciliation.md) - Hierarchical reconciliation for SIN
