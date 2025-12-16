# PC-032-05: BaseCombiner Abstract Class

**Epic:** [EPIC-05: Combination Infrastructure](../epics/EPIC-05-combination-layer.md)
**Task Reference:** T-05.1
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Create the `BaseCombiner` abstract R6 class that defines the interface for all forecast combination strategies. This is the foundation for the plugin-based combiner system.

---

## Acceptance Criteria

- [ ] Combiner base module created in `R/combination/base.R`
- [ ] `BaseCombiner` R6 abstract class with core interface
- [ ] Abstract methods: `combine()`, `fit_weights()`, `get_weights()`
- [ ] Forecast validation utilities
- [ ] Support for named model forecasts
- [ ] Integration hooks for bias correction

---

## Technical Specification

### File Location
```
R/combination/base.R
```

### BaseCombiner R6 Class

```r
#' @title BaseCombiner
#' @description Abstract base class for forecast combination strategies
#'
#' All combiner plugins must inherit from this class and implement:
#' - combine(): Combine multiple forecasts into one
#' - fit_weights(): Learn optimal weights from historical data
#' - get_weights(): Return current combination weights
#'
#' @export
BaseCombiner <- R6::R6Class(

  "BaseCombiner",
  private = list(
    weights = NULL,
    model_names = NULL,
    is_fitted = FALSE,
    fit_metadata = NULL
  ),
  public = list(
    #' @field name Combiner name
    name = NULL,

    #' @field config Combiner configuration
    config = NULL,

    #' @description Initialize combiner
    #' @param name Combiner name
    #' @param config Configuration list
    initialize = function(name = NULL, config = list()) {
      self$name <- name %||% class(self)[1]
      self$config <- private$apply_default_config(config)
    },

    #' @description Combine multiple forecasts
    #' @param forecasts Named list of forecast vectors/matrices
    #' @param weights Optional weight vector (uses fitted weights if NULL)
    #' @param ... Additional arguments
    #' @return Combined forecast
    combine = function(forecasts, weights = NULL, ...) {
      stop("combine() must be implemented by subclass")
    },

    #' @description Fit combination weights from historical data
    #' @param forecasts Named list of historical forecasts
    #' @param actuals Actual values corresponding to forecasts
    #' @param ... Additional arguments
    #' @return Invisible self
    fit_weights = function(forecasts, actuals, ...) {
      stop("fit_weights() must be implemented by subclass")
    },

    #' @description Get current combination weights
    #' @return Named numeric vector of weights
    get_weights = function() {
      if (is.null(private$weights)) {
        return(NULL)
      }
      weights <- private$weights
      names(weights) <- private$model_names
      weights
    },

    #' @description Set combination weights manually
    #' @param weights Named numeric vector of weights
    #' @return Invisible self
    set_weights = function(weights) {
      checkmate::assert_numeric(weights, lower = 0)

      # Validate weights sum to 1 (or normalize)
      weights <- self$validate_weights(weights)

      private$weights <- weights
      private$model_names <- names(weights)
      private$is_fitted <- TRUE

      invisible(self)
    },

    #' @description Validate and normalize weights
    #' @param weights Numeric vector of weights
    #' @return Normalized weights
    validate_weights = function(weights) {
      checkmate::assert_numeric(weights, lower = 0, any.missing = FALSE)

      weight_sum <- sum(weights)
      if (abs(weight_sum - 1) > 1e-6) {
        if (self$config$auto_normalize) {
          message("Normalizing weights to sum to 1")
          weights <- weights / weight_sum
        } else {
          warning(sprintf("Weights sum to %.4f, not 1", weight_sum))
        }
      }

      weights
    },

    #' @description Validate forecast structure
    #' @param forecasts List of forecasts
    #' @return TRUE if valid, error otherwise
    validate_forecasts = function(forecasts) {
      checkmate::assert_list(forecasts, min.len = 1, names = "named")

      # Check all forecasts have same length
      lengths <- sapply(forecasts, function(f) {
        if (is.matrix(f)) nrow(f) else length(f)
      })

      if (length(unique(lengths)) > 1) {
        stop(sprintf(
          "All forecasts must have same length. Got: %s",
          paste(lengths, collapse = ", ")
        ))
      }

      # Check for NA values
      na_counts <- sapply(forecasts, function(f) sum(is.na(f)))
      if (any(na_counts > 0)) {
        warning(sprintf(
          "Forecasts contain NA values: %s",
          paste(names(forecasts)[na_counts > 0], collapse = ", ")
        ))
      }

      invisible(TRUE)
    },

    #' @description Check if combiner has been fitted
    #' @return Logical
    is_fitted = function() {
      private$is_fitted
    },

    #' @description Get model names
    #' @return Character vector
    get_model_names = function() {
      private$model_names
    },

    #' @description Get fit metadata
    #' @return List with fit information
    get_fit_metadata = function() {
      private$fit_metadata
    },

    #' @description Print combiner summary
    print = function() {
      cat(sprintf("<%s>\n", class(self)[1]))
      cat(sprintf("  Name: %s\n", self$name))
      cat(sprintf("  Fitted: %s\n", private$is_fitted))

      if (private$is_fitted && !is.null(private$weights)) {
        cat("  Weights:\n")
        for (i in seq_along(private$weights)) {
          cat(sprintf("    %s: %.4f\n",
                      private$model_names[i], private$weights[i]))
        }
      }

      invisible(self)
    },

    #' @description Clone combiner
    #' @param deep Deep clone
    #' @return New combiner instance
    clone = function(deep = FALSE) {
      new_combiner <- super$clone(deep = deep)
      new_combiner
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        auto_normalize = TRUE,
        min_weight = 0.0,
        max_weight = 1.0,
        handle_na = "exclude"  # "exclude", "zero", "error"
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    },

    set_fit_metadata = function(metadata) {
      private$fit_metadata <- c(
        metadata,
        list(fitted_at = Sys.time())
      )
    }
  )
)
```

### Helper Functions

```r
#' Align forecasts to common index
#'
#' @param forecasts Named list of forecasts
#' @param method Alignment method ("intersection", "union")
#' @return Aligned forecasts
#' @export
align_forecasts <- function(forecasts, method = "intersection") {

  checkmate::assert_list(forecasts, names = "named")
  checkmate::assert_choice(method, c("intersection", "union"))


  # Get indices from each forecast
  indices <- lapply(forecasts, function(f) {
    if (is.data.table(f) && "DataHora" %in% names(f)) {
      f$DataHora
    } else if (!is.null(names(f))) {
      names(f)
    } else {
      seq_along(f)
    }
  })

  # Find common indices
  if (method == "intersection") {
    common_idx <- Reduce(intersect, indices)
  } else {
    common_idx <- Reduce(union, indices)
  }

  # Align all forecasts
  lapply(forecasts, function(f) {
    if (is.data.table(f)) {
      f[DataHora %in% common_idx]
    } else {
      f[names(f) %in% common_idx]
    }
  })
}


#' Convert forecasts to matrix format
#'
#' @param forecasts Named list of forecast vectors
#' @return Matrix with models as columns
#' @export
forecasts_to_matrix <- function(forecasts) {
  checkmate::assert_list(forecasts, types = "numeric", names = "named")

  # Stack forecasts as columns
  mat <- do.call(cbind, forecasts)
  colnames(mat) <- names(forecasts)

  mat
}


#' Convert matrix to forecast list
#'
#' @param mat Matrix with models as columns
#' @return Named list of forecast vectors
#' @export
matrix_to_forecasts <- function(mat) {
  checkmate::assert_matrix(mat)

  forecasts <- lapply(seq_len(ncol(mat)), function(i) mat[, i])
  names(forecasts) <- colnames(mat)

  forecasts
}


#' Calculate combination metrics
#'
#' @param combined Combined forecast
#' @param actuals Actual values
#' @param individual_forecasts Optional list of individual forecasts
#' @return List with combination metrics
#' @export
calculate_combination_metrics <- function(combined, actuals,
                                          individual_forecasts = NULL) {
  residuals <- actuals - combined

  metrics <- list(
    combined = list(
      mape = mean(abs(residuals / actuals), na.rm = TRUE) * 100,
      rmse = sqrt(mean(residuals^2, na.rm = TRUE)),
      mae = mean(abs(residuals), na.rm = TRUE),
      bias = mean(residuals, na.rm = TRUE)
    )
  )

  # Compare to individual models if provided
  if (!is.null(individual_forecasts)) {
    metrics$individual <- lapply(individual_forecasts, function(f) {
      res <- actuals - f
      list(
        mape = mean(abs(res / actuals), na.rm = TRUE) * 100,
        rmse = sqrt(mean(res^2, na.rm = TRUE)),
        mae = mean(abs(res), na.rm = TRUE)
      )
    })

    # Calculate improvement
    best_individual_mape <- min(sapply(metrics$individual, `[[`, "mape"))
    metrics$improvement_pct <- (best_individual_mape - metrics$combined$mape) /
                               best_individual_mape * 100
  }

  metrics
}
```

### Usage Example

```r
# Example subclass implementation
SimpleAverageCombiner <- R6::R6Class(
  "SimpleAverageCombiner",
  inherit = BaseCombiner,
  public = list(
    combine = function(forecasts, weights = NULL, ...) {
      self$validate_forecasts(forecasts)

      # Simple average - equal weights
      mat <- forecasts_to_matrix(forecasts)
      rowMeans(mat, na.rm = TRUE)
    },

    fit_weights = function(forecasts, actuals, ...) {
      # Equal weights for simple average
      n_models <- length(forecasts)
      private$weights <- rep(1 / n_models, n_models)
      private$model_names <- names(forecasts)
      private$is_fitted <- TRUE
      invisible(self)
    }
  )
)

# Usage
combiner <- SimpleAverageCombiner$new(name = "simple_avg")

forecasts <- list(
  lgbm = c(100, 110, 105),
  rf = c(102, 108, 107),
  hw = c(98, 112, 104)
)

# Combine
combined <- combiner$combine(forecasts)
# [1] 100 110 105.33

# Fit weights (optional for simple average)
combiner$fit_weights(forecasts, actuals = c(101, 109, 106))
combiner$get_weights()
# lgbm   rf   hw
# 0.33 0.33 0.33
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | Name and config set |
| TC-002 | combine() abstract | Error thrown |
| TC-003 | fit_weights() abstract | Error thrown |
| TC-004 | validate_forecasts() valid | TRUE |
| TC-005 | validate_forecasts() length mismatch | Error |
| TC-006 | validate_forecasts() with NAs | Warning |
| TC-007 | set_weights() valid | Weights stored |
| TC-008 | set_weights() unnormalized | Auto-normalize |
| TC-009 | get_weights() | Named vector |
| TC-010 | is_fitted() before fit | FALSE |
| TC-011 | is_fitted() after fit | TRUE |
| TC-012 | forecasts_to_matrix() | Correct matrix |
| TC-013 | align_forecasts() | Common index |
| TC-014 | calculate_combination_metrics() | Metrics computed |

---

## Dependencies

None (foundation class)

---

## Definition of Done

- [ ] BaseCombiner R6 class implemented
- [ ] Abstract methods defined
- [ ] Validation utilities working
- [ ] Helper functions implemented
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- All combiners must inherit from BaseCombiner
- Weight normalization is automatic by default
- Consider adding support for time-varying weights
- Future: add ensemble diversity metrics
- The combine() method should handle both vectors and matrices
