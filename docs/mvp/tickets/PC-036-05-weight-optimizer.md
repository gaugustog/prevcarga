# PC-036-05: Weight Optimizer Framework

**Epic:** [EPIC-05: Combination Infrastructure](../epics/EPIC-05-combination-layer.md)
**Task Reference:** T-05.5
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Create the `WeightOptimizer` R6 class framework for computing optimal combination weights using various methods (inverse error, OLS regression, constrained optimization).

---

## Acceptance Criteria

- [ ] Weight optimizer module created in `R/combination/optimizer.R`
- [ ] `WeightOptimizer` abstract R6 class
- [ ] Multiple optimization methods: inverse_mse, ols, constrained
- [ ] Weight validation and normalization
- [ ] Support for weight constraints
- [ ] Integration with CombinationWorkflow

---

## Technical Specification

### File Location
```
R/combination/optimizer.R
```

### WeightOptimizer Abstract Class

```r
#' @title WeightOptimizer
#' @description Abstract base class for combination weight optimization
#'
#' Computes optimal weights for combining multiple forecasts:
#' - Inverse MSE: weights inversely proportional to error
#' - OLS: ordinary least squares regression
#' - Constrained: optimization with bounds
#'
#' @export
WeightOptimizer <- R6::R6Class(
  "WeightOptimizer",
  private = list(
    optimal_weights = NULL,
    optimization_result = NULL,
    is_optimized = FALSE
  ),
  public = list(
    #' @field config Optimizer configuration
    config = NULL,

    #' @description Initialize optimizer
    #' @param config Configuration list
    initialize = function(config = list()) {
      self$config <- private$apply_default_config(config)
    },

    #' @description Optimize combination weights
    #' @param forecasts Named list of forecast vectors
    #' @param actuals Actual values
    #' @param method Optimization method
    #' @param ... Additional arguments
    #' @return Named numeric vector of weights
    optimize = function(forecasts, actuals, method = "inverse_mse", ...) {
      stop("optimize() must be implemented by subclass")
    },

    #' @description Get optimal weights
    #' @return Named numeric vector
    get_optimal_weights = function() {
      if (!private$is_optimized) {
        stop("Weights not yet optimized. Call optimize() first.")
      }
      private$optimal_weights
    },

    #' @description Get full optimization result
    #' @return List with optimization details
    get_optimization_result = function() {
      private$optimization_result
    },

    #' @description Validate and normalize weights
    #' @param weights Numeric vector
    #' @return Validated weights
    validate_weights = function(weights) {
      checkmate::assert_numeric(weights, any.missing = FALSE)

      # Apply bounds
      weights <- pmax(weights, self$config$min_weight)
      weights <- pmin(weights, self$config$max_weight)

      # Ensure non-negative
      if (any(weights < 0)) {
        if (self$config$allow_negative) {
          # Allow for short positions in advanced strategies
        } else {
          weights[weights < 0] <- 0
          warning("Negative weights set to 0")
        }
      }

      # Normalize to sum to 1
      if (abs(sum(weights) - 1) > 1e-6) {
        if (self$config$normalize) {
          weights <- weights / sum(weights)
        } else {
          warning(sprintf("Weights sum to %.4f, not 1", sum(weights)))
        }
      }

      weights
    },

    #' @description Check if optimization is done
    #' @return Logical
    is_optimized = function() {
      private$is_optimized
    },

    #' @description Print summary
    print = function() {
      cat(sprintf("<%s>\n", class(self)[1]))
      cat(sprintf("  Optimized: %s\n", private$is_optimized))

      if (private$is_optimized) {
        cat("  Optimal weights:\n")
        for (i in seq_along(private$optimal_weights)) {
          cat(sprintf("    %s: %.4f\n",
                      names(private$optimal_weights)[i],
                      private$optimal_weights[i]))
        }
      }

      invisible(self)
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        min_weight = 0.0,
        max_weight = 1.0,
        normalize = TRUE,
        allow_negative = FALSE,
        regularization = 0.0
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    }
  )
)
```

### Standard Weight Optimizer Implementation

```r
#' @title StandardWeightOptimizer
#' @description Standard weight optimization with multiple methods
#' @export
StandardWeightOptimizer <- R6::R6Class(
  "StandardWeightOptimizer",
  inherit = WeightOptimizer,
  public = list(
    #' @description Optimize weights
    optimize = function(forecasts, actuals, method = "inverse_mse", ...) {
      checkmate::assert_list(forecasts, names = "named", min.len = 1)
      checkmate::assert_numeric(actuals)
      checkmate::assert_choice(method, c("inverse_mse", "inverse_mae",
                                         "ols", "equal", "best_model"))

      model_names <- names(forecasts)
      n_models <- length(forecasts)

      weights <- switch(method,
        inverse_mse = self$optimize_inverse_mse(forecasts, actuals),
        inverse_mae = self$optimize_inverse_mae(forecasts, actuals),
        ols = self$optimize_ols(forecasts, actuals, ...),
        equal = self$optimize_equal(n_models),
        best_model = self$optimize_best_model(forecasts, actuals),
        stop(sprintf("Unknown method: %s", method))
      )

      names(weights) <- model_names
      weights <- self$validate_weights(weights)

      private$optimal_weights <- weights
      private$is_optimized <- TRUE

      private$optimization_result <- list(
        method = method,
        weights = weights,
        n_models = n_models,
        optimized_at = Sys.time()
      )

      weights
    },

    #' @description Inverse MSE weighting
    optimize_inverse_mse = function(forecasts, actuals) {
      mses <- sapply(forecasts, function(f) {
        mean((actuals - f)^2, na.rm = TRUE)
      })

      # Inverse weighting with small constant for stability
      epsilon <- 1e-8
      inv_mses <- 1 / (mses + epsilon)
      inv_mses / sum(inv_mses)
    },

    #' @description Inverse MAE weighting
    optimize_inverse_mae = function(forecasts, actuals) {
      maes <- sapply(forecasts, function(f) {
        mean(abs(actuals - f), na.rm = TRUE)
      })

      epsilon <- 1e-8
      inv_maes <- 1 / (maes + epsilon)
      inv_maes / sum(inv_maes)
    },

    #' @description OLS regression weights
    optimize_ols = function(forecasts, actuals, intercept = FALSE, ...) {
      # Build design matrix
      X <- do.call(cbind, forecasts)

      if (intercept) {
        X <- cbind(1, X)
      }

      # OLS: minimize ||y - Xw||^2
      # Solution: w = (X'X)^-1 X'y

      # Add regularization for stability
      lambda <- self$config$regularization
      XtX <- t(X) %*% X + lambda * diag(ncol(X))
      Xty <- t(X) %*% actuals

      weights <- as.vector(solve(XtX) %*% Xty)

      if (intercept) {
        # Store intercept separately
        private$optimization_result$intercept <- weights[1]
        weights <- weights[-1]
      }

      weights
    },

    #' @description Equal weights
    optimize_equal = function(n_models) {
      rep(1 / n_models, n_models)
    },

    #' @description Best single model (winner takes all)
    optimize_best_model = function(forecasts, actuals) {
      mapes <- sapply(forecasts, function(f) {
        mean(abs((actuals - f) / actuals), na.rm = TRUE) * 100
      })

      weights <- rep(0, length(forecasts))
      weights[which.min(mapes)] <- 1

      weights
    }
  )
)
```

### Constrained Weight Optimizer

```r
#' @title ConstrainedWeightOptimizer
#' @description Weight optimization with constraints
#' @export
ConstrainedWeightOptimizer <- R6::R6Class(
  "ConstrainedWeightOptimizer",
  inherit = WeightOptimizer,
  public = list(
    #' @description Optimize with constraints
    optimize = function(forecasts, actuals, method = "constrained_ols", ...) {
      checkmate::assert_list(forecasts, names = "named")

      weights <- switch(method,
        constrained_ols = self$optimize_constrained_ols(forecasts, actuals, ...),
        quadprog = self$optimize_quadprog(forecasts, actuals, ...),
        stop(sprintf("Unknown method: %s", method))
      )

      names(weights) <- names(forecasts)
      private$optimal_weights <- weights
      private$is_optimized <- TRUE

      weights
    },

    #' @description Constrained OLS (weights sum to 1, non-negative)
    optimize_constrained_ols = function(forecasts, actuals,
                                        max_iter = 100, tol = 1e-6, ...) {
      # Build design matrix
      X <- do.call(cbind, forecasts)
      n_models <- ncol(X)

      # Initialize with equal weights
      weights <- rep(1 / n_models, n_models)

      for (iter in seq_len(max_iter)) {
        # Gradient descent step
        residuals <- actuals - X %*% weights
        gradient <- -2 * t(X) %*% residuals / length(actuals)

        # Update weights
        step_size <- 0.01
        weights <- weights - step_size * gradient

        # Project to simplex (sum to 1, non-negative)
        weights <- self$project_to_simplex(weights)

        # Check convergence
        if (max(abs(gradient)) < tol) {
          break
        }
      }

      private$optimization_result$iterations <- iter
      weights
    },

    #' @description Quadratic programming optimization
    optimize_quadprog = function(forecasts, actuals, ...) {
      if (!requireNamespace("quadprog", quietly = TRUE)) {
        stop("Package 'quadprog' required for this method")
      }

      X <- do.call(cbind, forecasts)
      n_models <- ncol(X)

      # Quadratic program: min 0.5 * w'Dw - d'w
      # subject to: A'w >= b0

      # D = X'X
      Dmat <- t(X) %*% X + 1e-8 * diag(n_models)  # Small ridge for stability

      # d = X'y
      dvec <- t(X) %*% actuals

      # Constraints:
      # 1. Sum to 1: sum(w) = 1
      # 2. Non-negative: w >= 0
      Amat <- cbind(
        rep(1, n_models),  # Equality constraint
        diag(n_models)     # Non-negativity
      )
      bvec <- c(1, rep(0, n_models))
      meq <- 1  # First constraint is equality

      result <- quadprog::solve.QP(Dmat, dvec, Amat, bvec, meq = meq)

      result$solution
    },

    #' @description Project vector to probability simplex
    project_to_simplex = function(v) {
      # Michelot (1986) algorithm
      n <- length(v)
      u <- sort(v, decreasing = TRUE)
      cssv <- cumsum(u)
      rho <- max(which(u * (1:n) > (cssv - 1)))
      theta <- (cssv[rho] - 1) / rho
      pmax(v - theta, 0)
    }
  )
)
```

### Convenience Functions

```r
#' Compute optimal combination weights
#'
#' @param forecasts Named list of forecast vectors
#' @param actuals Actual values
#' @param method Optimization method
#' @param ... Additional arguments
#' @return Named numeric vector of weights
#' @export
#' @examples
#' weights <- compute_optimal_weights(
#'   forecasts = list(lgbm = lgbm_pred, rf = rf_pred),
#'   actuals = actual_values,
#'   method = "inverse_mse"
#' )
compute_optimal_weights <- function(forecasts, actuals,
                                    method = "inverse_mse", ...) {
  if (method %in% c("constrained_ols", "quadprog")) {
    optimizer <- ConstrainedWeightOptimizer$new()
  } else {
    optimizer <- StandardWeightOptimizer$new()
  }

  optimizer$optimize(forecasts, actuals, method = method, ...)
}


#' Compare weight optimization methods
#'
#' @param forecasts Named list of forecasts
#' @param actuals Actual values
#' @param methods Methods to compare
#' @return data.table with method comparison
#' @export
compare_weight_methods <- function(forecasts, actuals,
                                   methods = c("inverse_mse", "ols", "equal")) {
  results <- lapply(methods, function(method) {
    weights <- tryCatch(
      compute_optimal_weights(forecasts, actuals, method),
      error = function(e) NULL
    )

    if (is.null(weights)) {
      return(NULL)
    }

    # Compute combined forecast
    combined <- Reduce(`+`, Map(`*`, forecasts, weights))
    mape <- mean(abs((actuals - combined) / actuals)) * 100

    data.table::data.table(
      method = method,
      mape = mape,
      weights = list(weights)
    )
  })

  data.table::rbindlist(Filter(Negate(is.null), results))
}
```

### Usage Example

```r
# Standard optimization
optimizer <- StandardWeightOptimizer$new()

forecasts <- list(
  lgbm = lgbm_predictions,
  rf = rf_predictions,
  hw = hw_predictions
)

# Inverse MSE weighting
weights <- optimizer$optimize(forecasts, actuals, method = "inverse_mse")
# lgbm: 0.45, rf: 0.35, hw: 0.20

# OLS weights
weights_ols <- optimizer$optimize(forecasts, actuals, method = "ols")


# Constrained optimization
constrained_opt <- ConstrainedWeightOptimizer$new()
weights_constrained <- constrained_opt$optimize(
  forecasts, actuals,
  method = "constrained_ols"
)


# Compare methods
comparison <- compare_weight_methods(
  forecasts, actuals,
  methods = c("inverse_mse", "ols", "equal", "best_model")
)
#      method  mape                        weights
# 1: inverse_mse  2.34  0.4500, 0.3500, 0.2000
# 2:         ols  2.28  0.5123, 0.2877, 0.2000
# 3:       equal  2.56  0.3333, 0.3333, 0.3333
# 4:  best_model  2.45  1.0000, 0.0000, 0.0000
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | WeightOptimizer abstract | Error on optimize() |
| TC-002 | optimize_inverse_mse() | Inverse proportional |
| TC-003 | optimize_inverse_mae() | Inverse proportional |
| TC-004 | optimize_ols() | Regression weights |
| TC-005 | optimize_equal() | Equal weights |
| TC-006 | optimize_best_model() | Winner takes all |
| TC-007 | validate_weights() | Normalized to 1 |
| TC-008 | validate_weights() negative | Set to 0 |
| TC-009 | optimize_constrained_ols() | Constrained weights |
| TC-010 | project_to_simplex() | Valid simplex |
| TC-011 | compute_optimal_weights() | Convenience works |
| TC-012 | compare_weight_methods() | Comparison table |

---

## Dependencies

- PC-032-05: BaseCombiner (integration)
- PC-034-05: CombinationWorkflow (integration)

**Optional:**
- `quadprog` package for constrained optimization

---

## Definition of Done

- [ ] WeightOptimizer abstract class implemented
- [ ] StandardWeightOptimizer with 5 methods
- [ ] ConstrainedWeightOptimizer
- [ ] Simplex projection algorithm
- [ ] Convenience functions
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Inverse MSE is a simple, robust default
- OLS can produce negative weights (need constraints)
- Constrained optimization ensures valid weights
- Consider adding time-varying weights in future
- Regularization helps with collinear forecasts
