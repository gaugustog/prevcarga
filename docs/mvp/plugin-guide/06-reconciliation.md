# Plugin Guide: Reconciliation Strategies

> **Note:** This documentation will evolve into the project's CONTRIBUTING guide.

## Overview

Reconciliation ensures hierarchical consistency: area forecasts must sum to subsystem totals, which must sum to national (SIN) total. This is critical for the Brazilian interconnected grid (SIN).

## Hierarchy Structure

```
SIN (National) - area_code=SIN, reconciliation only
├── SECO (Subsystem) - reconciliation only
│   ├── RJ, SP, MG, ES, MT, MS, AC, RO, DF, GO (Areas - models run here)
│   └── PESE (Losses - model OR by_difference)
├── S (Subsystem) - reconciliation only
│   ├── PR, SC, RS (Areas)
│   └── PES (Losses)
├── NE (Subsystem) - reconciliation only
│   ├── ALPE, PBRN, BASE, CE, PI, BAOE (Areas)
│   └── PENE (Losses)
└── N (Subsystem) - reconciliation only
    ├── AM, PA, MA, TO, RR, AP (Areas)
    └── PEN (Losses)
```

**Key Principles:**
- Models run only for the 21 areas (17 regular + 4 loss areas)
- Subsystems and SIN are computed via reconciliation, never directly forecasted
- Loss areas can be forecasted by models OR calculated as: `Loss = Subsystem - Σ(other_areas)`

## BaseReconciler Interface

```r
#' @title BaseReconciler
#' @description Abstract base class for hierarchical reconciliation
BaseReconciler <- R6::R6Class(
  "BaseReconciler",
  public = list(
    name = NULL,
    config = NULL,

    initialize = function(name = NULL, config = list()) {
      self$name <- name
      self$config <- config
    },

    reconcile = function(forecasts, hierarchy, ...) {
      stop("reconcile() must be implemented by subclass")
    },

    build_summing_matrix = function(hierarchy) {
      stop("build_summing_matrix() must be implemented by subclass")
    },

    get_coherent_forecasts = function() {
      stop("get_coherent_forecasts() must be implemented by subclass")
    }
  )
)
```

## Summing Matrix Construction

The summing matrix S defines hierarchical relationships:

```r
HierarchyBuilder <- R6::R6Class(
  "HierarchyBuilder",
  private = list(
    hierarchy_config = NULL
  ),
  public = list(
    initialize = function(config) {
      private$hierarchy_config <- config
    },

    build_summing_matrix = function() {
      subsystems <- private$hierarchy_config$subsystems
      loss_areas <- private$hierarchy_config$loss_areas

      # Get all bottom-level series (areas including losses)
      all_areas <- unlist(subsystems)
      loss_names <- names(loss_areas)

      # Include losses if they're modeled
      bottom_series <- c(all_areas, loss_names)
      n_bottom <- length(bottom_series)

      # Build S matrix
      # Rows: all series (SIN, subsystems, areas)
      # Cols: bottom-level series only
      n_total <- 1 + length(subsystems) + n_bottom  # SIN + subsystems + bottom
      S <- matrix(0, nrow = n_total, ncol = n_bottom)

      # Row indices
      row_names <- c("SIN", names(subsystems), bottom_series)
      rownames(S) <- row_names
      colnames(S) <- bottom_series

      # Fill S matrix
      # SIN = sum of all bottom series
      S["SIN", ] <- 1

      # Each subsystem = sum of its areas + loss
      for (sub in names(subsystems)) {
        areas <- subsystems[[sub]]
        loss <- loss_areas[[sub]]$loss_area

        S[sub, areas] <- 1
        if (loss %in% bottom_series) {
          S[sub, loss] <- 1
        }
      }

      # Bottom series = themselves (identity)
      for (i in seq_along(bottom_series)) {
        S[bottom_series[i], bottom_series[i]] <- 1
      }

      S
    },

    get_bottom_series = function() {
      c(unlist(private$hierarchy_config$subsystems),
        names(private$hierarchy_config$loss_areas))
    },

    get_aggregation_levels = function() {
      list(
        national = "SIN",
        subsystems = names(private$hierarchy_config$subsystems),
        areas = self$get_bottom_series()
      )
    }
  )
)
```

## Reconciliation Strategies

### 1. Bottom-Up Reconciliation

Sum bottom-level forecasts to get aggregates:

```r
BottomUpReconciler <- R6::R6Class(
  "BottomUpReconciler",
  inherit = BaseReconciler,
  private = list(
    summing_matrix = NULL
  ),
  public = list(
    reconcile = function(forecasts, hierarchy_builder, ...) {
      S <- hierarchy_builder$build_summing_matrix()
      private$summing_matrix <- S

      # Extract bottom-level forecasts
      bottom_series <- hierarchy_builder$get_bottom_series()
      n_periods <- nrow(forecasts[[bottom_series[1]]])

      # Build forecast matrix (bottom series x periods)
      Y_bottom <- matrix(NA, nrow = length(bottom_series), ncol = n_periods)
      rownames(Y_bottom) <- bottom_series

      for (series in bottom_series) {
        Y_bottom[series, ] <- forecasts[[series]]$prediction
      }

      # Reconcile: Y_reconciled = S %*% Y_bottom
      Y_reconciled <- S %*% Y_bottom

      # Convert back to data.table format
      result <- list()
      for (series in rownames(Y_reconciled)) {
        result[[series]] <- data.table(
          DataHora = forecasts[[bottom_series[1]]]$DataHora,
          area_code = series,
          prediction = Y_reconciled[series, ]
        )
      }

      result
    }
  )
)
```

### 2. Top-Down Reconciliation

Distribute top-level forecast proportionally:

```r
TopDownReconciler <- R6::R6Class(
  "TopDownReconciler",
  inherit = BaseReconciler,
  private = list(
    proportions = NULL
  ),
  public = list(
    fit_proportions = function(historical_data, hierarchy_builder) {
      # Calculate historical proportions
      bottom_series <- hierarchy_builder$get_bottom_series()

      # Get historical averages
      totals <- sapply(bottom_series, function(s) {
        mean(historical_data[[s]]$CargaGlobal, na.rm = TRUE)
      })

      private$proportions <- totals / sum(totals)
      private$proportions
    },

    reconcile = function(forecasts, hierarchy_builder, ...) {
      S <- hierarchy_builder$build_summing_matrix()
      bottom_series <- hierarchy_builder$get_bottom_series()

      # Get top-level (SIN) forecast
      sin_forecast <- forecasts[["SIN"]]$prediction

      # Distribute to bottom series using proportions
      result <- list()

      # SIN stays as is
      result[["SIN"]] <- forecasts[["SIN"]]

      # Bottom series get proportional allocation
      for (i in seq_along(bottom_series)) {
        series <- bottom_series[i]
        result[[series]] <- data.table(
          DataHora = forecasts[["SIN"]]$DataHora,
          area_code = series,
          prediction = sin_forecast * private$proportions[i]
        )
      }

      # Subsystems = sum of their components
      levels <- hierarchy_builder$get_aggregation_levels()
      for (sub in levels$subsystems) {
        sub_areas <- c(hierarchy_builder$hierarchy_config$subsystems[[sub]],
                       hierarchy_builder$hierarchy_config$loss_areas[[sub]])
        sub_pred <- rowSums(sapply(sub_areas, function(a) result[[a]]$prediction))

        result[[sub]] <- data.table(
          DataHora = forecasts[["SIN"]]$DataHora,
          area_code = sub,
          prediction = sub_pred
        )
      }

      result
    }
  )
)
```

### 3. Optimal Reconciliation (MinT-like)

Minimize trace of reconciled forecast error covariance:

```r
OptimalReconciler <- R6::R6Class(
  "OptimalReconciler",
  inherit = BaseReconciler,
  private = list(
    W = NULL,  # Covariance matrix
    G = NULL   # Reconciliation matrix
  ),
  public = list(
    fit = function(residuals_data, hierarchy_builder) {
      S <- hierarchy_builder$build_summing_matrix()
      n <- nrow(S)
      m <- ncol(S)

      # Estimate W from forecast residuals (simplified: use diagonal)
      # In practice, use shrinkage estimator
      all_series <- rownames(S)
      variances <- sapply(all_series, function(s) {
        var(residuals_data[[s]], na.rm = TRUE)
      })
      private$W <- diag(variances)

      # MinT reconciliation matrix:
      # G = (S'W^{-1}S)^{-1} S'W^{-1}
      W_inv <- solve(private$W)
      private$G <- solve(t(S) %*% W_inv %*% S) %*% t(S) %*% W_inv

      invisible(self)
    },

    reconcile = function(forecasts, hierarchy_builder, ...) {
      S <- hierarchy_builder$build_summing_matrix()
      all_series <- rownames(S)
      n_periods <- nrow(forecasts[[all_series[1]]])

      # Build base forecast matrix
      Y_base <- matrix(NA, nrow = length(all_series), ncol = n_periods)
      rownames(Y_base) <- all_series

      for (series in all_series) {
        Y_base[series, ] <- forecasts[[series]]$prediction
      }

      # Reconcile: Y_tilde = S %*% G %*% Y_base
      Y_reconciled <- S %*% private$G %*% Y_base

      # Convert back to data.table format
      result <- list()
      for (series in all_series) {
        result[[series]] <- data.table(
          DataHora = forecasts[[all_series[1]]]$DataHora,
          area_code = series,
          prediction = Y_reconciled[series, ]
        )
      }

      result
    }
  )
)
```

### 4. Loss Area by Difference

Calculate loss areas as residual:

```r
LossCalculator <- R6::R6Class(
  "LossCalculator",
  public = list(
    calculate_loss = function(subsystem_forecast, area_forecasts,
                              loss_area_code, ...) {
      # Loss = Subsystem - Σ(other_areas)
      area_sum <- Reduce(`+`, lapply(area_forecasts, `[[`, "prediction"))

      data.table(
        DataHora = subsystem_forecast$DataHora,
        area_code = loss_area_code,
        prediction = subsystem_forecast$prediction - area_sum,
        source = "by_difference"
      )
    }
  )
)

# Usage in reconciliation workflow
calculate_losses_by_difference = function(reconciled_forecasts, config) {
  loss_config <- config$loss_areas

  for (loss_name in names(loss_config)) {
    if (loss_config[[loss_name]]$method == "by_difference") {
      subsystem <- loss_config[[loss_name]]$subsystem
      areas <- config$subsystems[[subsystem]]

      # Get forecasts
      sub_forecast <- reconciled_forecasts[[subsystem]]
      area_forecasts <- lapply(areas, function(a) reconciled_forecasts[[a]])

      # Calculate loss
      calculator <- LossCalculator$new()
      reconciled_forecasts[[loss_name]] <- calculator$calculate_loss(
        sub_forecast,
        area_forecasts,
        loss_name
      )
    }
  }

  reconciled_forecasts
}
```

## Reconciler Registry

```r
ReconcilerRegistry <- R6::R6Class(
  "ReconcilerRegistry",
  private = list(
    reconcilers = list()
  ),
  public = list(
    register = function(name, reconciler_class) {
      private$reconcilers[[name]] <- reconciler_class
      invisible(self)
    },

    get = function(name, ...) {
      if (!name %in% names(private$reconcilers)) {
        stop(sprintf("Reconciler '%s' not registered", name))
      }
      private$reconcilers[[name]]$new(...)
    },

    list_reconcilers = function() {
      names(private$reconcilers)
    }
  )
)

# Default registry
default_reconciler_registry <- function() {
  registry <- ReconcilerRegistry$new()
  registry$register("bottom_up", BottomUpReconciler)
  registry$register("top_down", TopDownReconciler)
  registry$register("optimal", OptimalReconciler)
  registry
}
```

## Reconciliation Workflow

```r
ReconciliationWorkflow <- R6::R6Class(
  "ReconciliationWorkflow",
  private = list(
    reconciler = NULL,
    hierarchy_builder = NULL,
    config = NULL
  ),
  public = list(
    initialize = function(reconciler, hierarchy_config, ...) {
      private$reconciler <- reconciler
      private$hierarchy_builder <- HierarchyBuilder$new(hierarchy_config)
      private$config <- hierarchy_config
    },

    run = function(forecasts, ...) {
      # Step 1: Reconcile hierarchy
      reconciled <- private$reconciler$reconcile(
        forecasts,
        private$hierarchy_builder
      )

      # Step 2: Calculate losses by difference (if configured)
      reconciled <- calculate_losses_by_difference(reconciled, private$config)

      # Step 3: Validate coherence
      self$validate_coherence(reconciled)

      list(
        predictions = reconciled,
        summing_matrix = private$hierarchy_builder$build_summing_matrix(),
        metadata = list(
          reconciler = class(private$reconciler)[1],
          timestamp = Sys.time()
        )
      )
    },

    validate_coherence = function(forecasts) {
      S <- private$hierarchy_builder$build_summing_matrix()
      levels <- private$hierarchy_builder$get_aggregation_levels()

      # Check: SIN = sum of subsystems
      sin_pred <- forecasts[["SIN"]]$prediction
      sub_sum <- rowSums(sapply(levels$subsystems, function(s) {
        forecasts[[s]]$prediction
      }))

      if (!all(abs(sin_pred - sub_sum) < 1e-6)) {
        warning("SIN != sum of subsystems after reconciliation")
      }

      # Check: each subsystem = sum of its areas + loss
      for (sub in levels$subsystems) {
        areas <- private$config$subsystems[[sub]]
        loss <- private$config$loss_areas[[sub]]$loss_area

        sub_pred <- forecasts[[sub]]$prediction
        area_sum <- rowSums(sapply(c(areas, loss), function(a) {
          forecasts[[a]]$prediction
        }))

        if (!all(abs(sub_pred - area_sum) < 1e-6)) {
          warning(sprintf("%s != sum of areas + loss after reconciliation", sub))
        }
      }

      invisible(TRUE)
    }
  )
)
```

## Configuration

```yaml
reconciliation:
  strategy: bottom_up  # bottom_up, top_down, optimal

  hierarchy:
    national: SIN

    subsystems:
      SECO: [RJ, SP, MG, ES, MT, MS, AC, RO, DF, GO]
      S: [PR, SC, RS]
      NE: [ALPE, PBRN, BASE, CE, PI, BAOE]
      N: [AM, PA, MA, TO, RR, AP]

    loss_areas:
      PESE: {subsystem: SECO, method: by_difference}
      PES: {subsystem: S, method: by_difference}
      PENE: {subsystem: NE, method: model}  # Has model
      PEN: {subsystem: N, method: by_difference}

  optimal:
    covariance_method: shrinkage  # diagonal, sample, shrinkage
    shrinkage_target: diagonal
```

## Complete Example

```r
# Load configuration
config <- yaml::read_yaml("config/reconciliation.yaml")

# Build hierarchy
hierarchy_builder <- HierarchyBuilder$new(config$hierarchy)

# Get area forecasts (from models)
area_forecasts <- lapply(hierarchy_builder$get_bottom_series(), function(area) {
  load_forecast(area, target_date)
})
names(area_forecasts) <- hierarchy_builder$get_bottom_series()

# Create reconciler
reconciler <- default_reconciler_registry()$get(config$strategy)

# Run reconciliation workflow
workflow <- ReconciliationWorkflow$new(reconciler, config$hierarchy)
result <- workflow$run(area_forecasts)

# Save reconciled forecasts
for (series in names(result$predictions)) {
  storage$write_parquet(
    result$predictions[[series]],
    file.path("results/reconciled", target_date, paste0(series, ".parquet"))
  )
}
```

## Reference Files

| Pattern | Reference File | Key Lines |
|---------|----------------|-----------|
| Summing matrix | `docs/legacy/reconciliation/summing_matrix.R` | TBD |
| MinT reconciliation | `docs/legacy/reconciliation/mint.R` | TBD |
| Bottom-up | `docs/legacy/reconciliation/bottom_up.R` | TBD |

## Checklist for Reconciliation Implementation

- [ ] Implement BaseReconciler interface
- [ ] Build summing matrix correctly
- [ ] Handle loss areas (model vs by_difference)
- [ ] Validate hierarchical coherence after reconciliation
- [ ] Support multiple strategies (bottom_up, top_down, optimal)
- [ ] Covariance estimation for optimal reconciliation
- [ ] Register reconciler in registry
- [ ] Integration with storage backend

## Next Steps

Return to [Overview](00-overview.md) for the complete plugin architecture.
