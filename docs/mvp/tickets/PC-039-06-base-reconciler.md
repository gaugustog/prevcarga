# PC-039-06: BaseReconciler Abstract Class

**Epic:** [EPIC-06: Reconciliation Infrastructure](../epics/EPIC-06-reconciliation-layer.md)
**Task Reference:** T-06.1
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Create the `BaseReconciler` abstract R6 class that defines the interface for all hierarchical reconciliation strategies. This is the foundation for ensuring forecasts sum correctly across the SIN hierarchy.

---

## Acceptance Criteria

- [ ] Reconciler base module created in `R/reconciliation/base.R`
- [ ] `BaseReconciler` R6 abstract class with core interface
- [ ] Abstract methods: `reconcile()`, `build_reconciliation_matrix()`, `get_coherent_forecasts()`
- [ ] Hierarchy validation utilities
- [ ] Support for different reconciliation approaches (bottom-up, top-down, optimal)

---

## Technical Specification

### File Location
```
R/reconciliation/base.R
```

### BaseReconciler R6 Class

```r
#' @title BaseReconciler
#' @description Abstract base class for hierarchical reconciliation strategies
#'
#' Hierarchical reconciliation ensures forecasts are coherent:
#' - SIN = SECO + S + NE + N
#' - Each subsystem = sum of its areas + losses
#'
#' All reconciler plugins must inherit from this class.
#'
#' @export
BaseReconciler <- R6::R6Class(
  "BaseReconciler",
  private = list(
    summing_matrix = NULL,
    reconciliation_matrix = NULL,
    coherent_forecasts = NULL,
    is_fitted = FALSE
  ),
  public = list(
    #' @field name Reconciler name
    name = NULL,

    #' @field config Reconciler configuration
    config = NULL,

    #' @description Initialize reconciler
    #' @param name Reconciler name
    #' @param config Configuration list
    initialize = function(name = NULL, config = list()) {
      self$name <- name %||% class(self)[1]
      self$config <- private$apply_default_config(config)
    },

    #' @description Reconcile forecasts to be hierarchically coherent
    #' @param forecasts Named list of forecast vectors (one per series)
    #' @param hierarchy HierarchyBuilder instance
    #' @param ... Additional arguments
    #' @return Named list of reconciled forecasts
    reconcile = function(forecasts, hierarchy, ...) {
      stop("reconcile() must be implemented by subclass")
    },

    #' @description Build the reconciliation matrix G
    #' @param hierarchy HierarchyBuilder instance
    #' @param ... Additional arguments (e.g., covariance matrix)
    #' @return Reconciliation matrix G
    build_reconciliation_matrix = function(hierarchy, ...) {
      stop("build_reconciliation_matrix() must be implemented by subclass")
    },

    #' @description Get coherent forecasts after reconciliation
    #' @return Named list of reconciled forecasts
    get_coherent_forecasts = function() {
      if (is.null(private$coherent_forecasts)) {
        stop("No reconciled forecasts available. Call reconcile() first.")
      }
      private$coherent_forecasts
    },

    #' @description Get the summing matrix S
    #' @return Summing matrix
    get_summing_matrix = function() {
      private$summing_matrix
    },

    #' @description Set the summing matrix S
    #' @param S Summing matrix
    #' @return Invisible self
    set_summing_matrix = function(S) {
      checkmate::assert_matrix(S)
      private$summing_matrix <- S
      invisible(self)
    },

    #' @description Get the reconciliation matrix G
    #' @return Reconciliation matrix
    get_reconciliation_matrix = function() {
      private$reconciliation_matrix
    },

    #' @description Validate forecast hierarchy structure

    #' @param forecasts Named list of forecasts
    #' @param hierarchy HierarchyBuilder instance
    #' @return TRUE if valid, error otherwise
    validate_hierarchy = function(forecasts, hierarchy) {
      checkmate::assert_list(forecasts, names = "named")

      # Get required series from hierarchy
      required_series <- hierarchy$get_all_series()

      # Check all required series are present
      missing <- setdiff(required_series, names(forecasts))
      if (length(missing) > 0) {
        stop(sprintf(
          "Missing forecasts for series: %s",
          paste(missing, collapse = ", ")
        ))
      }

      # Check all forecasts have same length
      lengths <- sapply(forecasts, length)
      if (length(unique(lengths)) > 1) {
        stop("All forecasts must have the same length")
      }

      invisible(TRUE)
    },

    #' @description Apply reconciliation formula: ỹ = S * G * ŷ
    #' @param base_forecasts Matrix of base forecasts (n_time × n_series)
    #' @param S Summing matrix
    #' @param G Reconciliation matrix
    #' @return Matrix of reconciled forecasts
    apply_reconciliation = function(base_forecasts, S, G) {
      checkmate::assert_matrix(base_forecasts)
      checkmate::assert_matrix(S)
      checkmate::assert_matrix(G)

      # Reconciled = S * G * base_forecasts'
      # Result is (n_series × n_time), transpose to (n_time × n_series)
      reconciled <- S %*% G %*% t(base_forecasts)
      t(reconciled)
    },

    #' @description Check if reconciler has been fitted
    #' @return Logical
    is_fitted = function() {
      private$is_fitted
    },

    #' @description Print reconciler summary
    print = function() {
      cat(sprintf("<%s>\n", class(self)[1]))
      cat(sprintf("  Name: %s\n", self$name))
      cat(sprintf("  Fitted: %s\n", private$is_fitted))

      if (!is.null(private$summing_matrix)) {
        cat(sprintf("  Summing matrix: %d × %d\n",
                    nrow(private$summing_matrix),
                    ncol(private$summing_matrix)))
      }

      invisible(self)
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        method = "ols",  # ols, wls, mint, bottom_up, top_down
        nonnegative = TRUE,
        residuals_type = "in_sample"
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    },

    set_coherent_forecasts = function(forecasts) {
      private$coherent_forecasts <- forecasts
      private$is_fitted <- TRUE
    }
  )
)
```

### Helper Functions

```r
#' Convert forecast list to matrix
#'
#' @param forecasts Named list of forecast vectors
#' @param series_order Optional vector specifying column order
#' @return Matrix with series as columns
#' @export
forecasts_to_matrix <- function(forecasts, series_order = NULL) {
  checkmate::assert_list(forecasts, names = "named")

  if (!is.null(series_order)) {
    forecasts <- forecasts[series_order]
  }

  do.call(cbind, forecasts)
}


#' Convert forecast matrix to list
#'
#' @param mat Matrix with series as columns
#' @param series_names Names for the series
#' @return Named list of forecast vectors
#' @export
matrix_to_forecasts <- function(mat, series_names = NULL) {
  checkmate::assert_matrix(mat)

  if (is.null(series_names)) {
    series_names <- colnames(mat)
  }

  forecasts <- lapply(seq_len(ncol(mat)), function(i) mat[, i])
  names(forecasts) <- series_names

  forecasts
}


#' Calculate forecast coherence error
#'
#' @param forecasts Named list of forecasts
#' @param hierarchy HierarchyBuilder instance
#' @return data.table with coherence errors by constraint
#' @export
calculate_coherence_error <- function(forecasts, hierarchy) {
  S <- hierarchy$build_summing_matrix()
  bottom_series <- hierarchy$get_bottom_series()

  # Get bottom-level forecasts
  bottom_forecasts <- forecasts[bottom_series]
  bottom_mat <- forecasts_to_matrix(bottom_forecasts)

  # Calculate implied aggregates
  implied <- t(S %*% t(bottom_mat))
  colnames(implied) <- hierarchy$get_all_series()

  # Compare with actual forecasts
  errors <- list()
  for (series in hierarchy$get_all_series()) {
    if (series %in% names(forecasts)) {
      error <- forecasts[[series]] - implied[, series]
      errors[[series]] <- list(
        series = series,
        mean_error = mean(error, na.rm = TRUE),
        max_abs_error = max(abs(error), na.rm = TRUE),
        is_coherent = max(abs(error), na.rm = TRUE) < 1e-6
      )
    }
  }

  data.table::rbindlist(errors)
}


#' Check if forecasts are hierarchically coherent
#'
#' @param forecasts Named list of forecasts
#' @param hierarchy HierarchyBuilder instance
#' @param tolerance Tolerance for numerical comparison
#' @return Logical
#' @export
is_coherent <- function(forecasts, hierarchy, tolerance = 1e-6) {
  errors <- calculate_coherence_error(forecasts, hierarchy)
  all(errors$max_abs_error < tolerance)
}
```

### Usage Example

```r
# Example subclass implementation (Bottom-Up reconciler)
BottomUpReconciler <- R6::R6Class(
  "BottomUpReconciler",
  inherit = BaseReconciler,
  public = list(
    reconcile = function(forecasts, hierarchy, ...) {
      self$validate_hierarchy(forecasts, hierarchy)

      # Get summing matrix
      S <- hierarchy$build_summing_matrix()
      self$set_summing_matrix(S)

      # Get bottom-level forecasts
      bottom_series <- hierarchy$get_bottom_series()
      bottom_forecasts <- forecasts[bottom_series]
      bottom_mat <- forecasts_to_matrix(bottom_forecasts)

      # Bottom-up: aggregates are sums of bottom level
      reconciled_mat <- t(S %*% t(bottom_mat))
      colnames(reconciled_mat) <- hierarchy$get_all_series()

      # Convert back to list
      reconciled <- matrix_to_forecasts(reconciled_mat)

      private$set_coherent_forecasts(reconciled)
      reconciled
    },

    build_reconciliation_matrix = function(hierarchy, ...) {
      # For bottom-up, G selects bottom-level series
      S <- hierarchy$build_summing_matrix()
      n_bottom <- ncol(S)
      n_total <- nrow(S)

      # G is (n_bottom × n_total) that selects bottom rows
      G <- matrix(0, nrow = n_bottom, ncol = n_total)
      bottom_indices <- (n_total - n_bottom + 1):n_total
      for (i in seq_len(n_bottom)) {
        G[i, bottom_indices[i]] <- 1
      }

      private$reconciliation_matrix <- G
      G
    }
  )
)

# Usage
reconciler <- BottomUpReconciler$new(name = "bottom_up")

forecasts <- list(
  SIN = sin_forecast,
  SECO = seco_forecast,
  S = s_forecast,
  NE = ne_forecast,
  N = n_forecast,
  RJ = rj_forecast,
  SP = sp_forecast,
  # ... all other areas
)

reconciled <- reconciler$reconcile(forecasts, hierarchy)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | Name and config set |
| TC-002 | reconcile() abstract | Error thrown |
| TC-003 | build_reconciliation_matrix() abstract | Error thrown |
| TC-004 | validate_hierarchy() valid | TRUE |
| TC-005 | validate_hierarchy() missing series | Error |
| TC-006 | validate_hierarchy() length mismatch | Error |
| TC-007 | apply_reconciliation() formula | Correct calculation |
| TC-008 | get_coherent_forecasts() before reconcile | Error |
| TC-009 | get_coherent_forecasts() after reconcile | Returns forecasts |
| TC-010 | forecasts_to_matrix() | Correct matrix |
| TC-011 | matrix_to_forecasts() | Correct list |
| TC-012 | calculate_coherence_error() | Errors computed |
| TC-013 | is_coherent() true | TRUE |
| TC-014 | is_coherent() false | FALSE |

---

## Dependencies

None (foundation class)

---

## Definition of Done

- [ ] BaseReconciler R6 class implemented
- [ ] Abstract methods defined
- [ ] Hierarchy validation working
- [ ] Helper functions implemented
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The summing matrix S encodes the hierarchy structure
- Reconciliation formula: ỹ = S * G * ŷ where G is method-specific
- Bottom-up is simplest: aggregates = sum of bottom level
- MinT/OLS reconcilers will use covariance estimation
- Consider adding support for temporal reconciliation in future
