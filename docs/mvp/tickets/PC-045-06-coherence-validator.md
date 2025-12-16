# PC-045-06: Coherence Validator

**Epic:** [EPIC-06: Reconciliation Infrastructure](../epics/EPIC-06-reconciliation-layer.md)
**Task Reference:** T-06.7
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Implement the `CoherenceValidator` R6 class for validating hierarchical coherence of forecasts. Ensures that aggregation constraints are satisfied: SIN = Σ(subsystems) and each subsystem = Σ(areas) + loss.

---

## Acceptance Criteria

- [ ] Coherence module created in `R/reconciliation/coherence.R`
- [ ] `CoherenceValidator` R6 class with validation methods
- [ ] National-level coherence check
- [ ] Subsystem-level coherence check
- [ ] Tolerance-based comparison
- [ ] Detailed validation reports

---

## Technical Specification

### File Location
```
R/reconciliation/coherence.R
```

### CoherenceValidator R6 Class

```r
#' @title CoherenceValidator
#' @description Validate hierarchical coherence of forecasts
#'
#' Checks aggregation constraints:
#' - SIN = SECO + S + NE + N
#' - SECO = Σ(SECO_areas) + PESE
#' - S = Σ(S_areas) + PES
#' - NE = Σ(NE_areas) + PENE
#' - N = Σ(N_areas) + PEN
#'
#' @export
CoherenceValidator <- R6::R6Class(
  "CoherenceValidator",
  private = list(
    tolerance = NULL,
    strict = NULL,
    hierarchy = NULL
  ),
  public = list(
    #' @description Initialize validator
    #' @param hierarchy HierarchyBuilder instance
    #' @param tolerance Numerical tolerance for comparison
    #' @param strict If TRUE, throw error; if FALSE, warn
    initialize = function(hierarchy,
                          tolerance = 1e-6,
                          strict = FALSE) {
      checkmate::assert_class(hierarchy, "HierarchyBuilder")
      checkmate::assert_number(tolerance, lower = 0)
      checkmate::assert_flag(strict)

      private$hierarchy <- hierarchy
      private$tolerance <- tolerance
      private$strict <- strict
    },

    #' @description Validate all hierarchical constraints
    #' @param forecasts Named list of forecast vectors
    #' @return data.table with validation results
    validate = function(forecasts) {
      checkmate::assert_list(forecasts, names = "named")

      results <- list()

      # Validate national level
      national_result <- self$validate_national(forecasts)
      results <- c(results, list(national_result))

      # Validate each subsystem
      config <- private$hierarchy$get_config()
      for (subsystem in names(config$subsystems)) {
        subsystem_result <- self$validate_subsystem(forecasts, subsystem)
        results <- c(results, list(subsystem_result))
      }

      result_dt <- data.table::rbindlist(results)

      # Report violations
      self$report_violations(result_dt)

      result_dt
    },

    #' @description Validate national-level coherence
    #' @param forecasts Named list of forecast vectors
    #' @return data.table with validation result
    validate_national = function(forecasts) {
      config <- private$hierarchy$get_config()

      national <- config$national
      subsystems <- names(config$subsystems)

      # Get forecasts
      national_forecast <- forecasts[[national]]
      subsystem_sum <- Reduce(`+`, forecasts[subsystems])

      # Calculate error
      error <- national_forecast - subsystem_sum

      data.table::data.table(
        constraint = national,
        constraint_type = "national",
        expected_sum_of = paste(subsystems, collapse = " + "),
        n_time_points = length(error),
        mean_error = mean(error, na.rm = TRUE),
        max_abs_error = max(abs(error), na.rm = TRUE),
        min_error = min(error, na.rm = TRUE),
        max_error = max(error, na.rm = TRUE),
        is_coherent = max(abs(error), na.rm = TRUE) < private$tolerance
      )
    },

    #' @description Validate subsystem-level coherence
    #' @param forecasts Named list of forecast vectors
    #' @param subsystem Subsystem code
    #' @return data.table with validation result
    validate_subsystem = function(forecasts, subsystem) {
      children <- private$hierarchy$get_children(subsystem)

      # Get forecasts
      subsystem_forecast <- forecasts[[subsystem]]
      children_sum <- Reduce(`+`, forecasts[children])

      # Calculate error
      error <- subsystem_forecast - children_sum

      data.table::data.table(
        constraint = subsystem,
        constraint_type = "subsystem",
        expected_sum_of = paste(children, collapse = " + "),
        n_time_points = length(error),
        mean_error = mean(error, na.rm = TRUE),
        max_abs_error = max(abs(error), na.rm = TRUE),
        min_error = min(error, na.rm = TRUE),
        max_error = max(error, na.rm = TRUE),
        is_coherent = max(abs(error), na.rm = TRUE) < private$tolerance
      )
    },

    #' @description Report constraint violations
    #' @param result_dt Validation results data.table
    report_violations = function(result_dt) {
      violations <- result_dt[is_coherent == FALSE]

      if (nrow(violations) > 0) {
        msg <- sprintf(
          "Coherence violations detected:\n%s",
          paste(
            sprintf(
              "  - %s: max_abs_error = %.6f",
              violations$constraint,
              violations$max_abs_error
            ),
            collapse = "\n"
          )
        )

        if (private$strict) {
          stop(msg)
        } else {
          warning(msg)
        }
      } else {
        message("All hierarchical constraints satisfied")
      }
    },

    #' @description Check if forecasts are coherent
    #' @param forecasts Named list of forecast vectors
    #' @return Logical
    is_coherent = function(forecasts) {
      results <- self$validate(forecasts)
      all(results$is_coherent)
    },

    #' @description Get coherence error for each time point
    #' @param forecasts Named list of forecast vectors
    #' @return data.table with errors by time point
    get_error_timeseries = function(forecasts) {
      config <- private$hierarchy$get_config()
      n_time <- length(forecasts[[1]])

      errors <- list()

      # National error
      national <- config$national
      subsystems <- names(config$subsystems)
      national_forecast <- forecasts[[national]]
      subsystem_sum <- Reduce(`+`, forecasts[subsystems])

      errors[[national]] <- data.table::data.table(
        time_index = seq_len(n_time),
        constraint = national,
        error = national_forecast - subsystem_sum
      )

      # Subsystem errors
      for (subsystem in subsystems) {
        children <- private$hierarchy$get_children(subsystem)
        subsystem_forecast <- forecasts[[subsystem]]
        children_sum <- Reduce(`+`, forecasts[children])

        errors[[subsystem]] <- data.table::data.table(
          time_index = seq_len(n_time),
          constraint = subsystem,
          error = subsystem_forecast - children_sum
        )
      }

      data.table::rbindlist(errors)
    },

    #' @description Get tolerance
    #' @return Tolerance value
    get_tolerance = function() {
      private$tolerance
    },

    #' @description Set tolerance
    #' @param tolerance New tolerance value
    set_tolerance = function(tolerance) {
      checkmate::assert_number(tolerance, lower = 0)
      private$tolerance <- tolerance
      invisible(self)
    },

    #' @description Print summary
    print = function() {
      cat("CoherenceValidator\n")
      cat(sprintf("  Tolerance: %.2e\n", private$tolerance))
      cat(sprintf("  Strict mode: %s\n", private$strict))
      cat(sprintf("  Hierarchy: %s\n", private$hierarchy$get_config()$national))
      invisible(self)
    }
  )
)
```

### Convenience Functions

```r
#' Validate hierarchical coherence
#'
#' @param forecasts Named list of forecast vectors
#' @param hierarchy HierarchyBuilder instance
#' @param tolerance Tolerance for comparison
#' @return data.table with validation results
#' @export
validate_hierarchical_coherence <- function(forecasts,
                                            hierarchy,
                                            tolerance = 1e-6) {
  validator <- CoherenceValidator$new(
    hierarchy = hierarchy,
    tolerance = tolerance,
    strict = FALSE
  )
  validator$validate(forecasts)
}


#' Check if forecasts are hierarchically coherent
#'
#' @param forecasts Named list of forecast vectors
#' @param hierarchy HierarchyBuilder instance
#' @param tolerance Tolerance for comparison
#' @return Logical
#' @export
is_hierarchically_coherent <- function(forecasts,
                                       hierarchy,
                                       tolerance = 1e-6) {
  validator <- CoherenceValidator$new(
    hierarchy = hierarchy,
    tolerance = tolerance,
    strict = FALSE
  )
  validator$is_coherent(forecasts)
}


#' Assert hierarchical coherence
#'
#' Throws error if forecasts are not coherent.
#'
#' @param forecasts Named list of forecast vectors
#' @param hierarchy HierarchyBuilder instance
#' @param tolerance Tolerance for comparison
#' @export
assert_hierarchically_coherent <- function(forecasts,
                                           hierarchy,
                                           tolerance = 1e-6) {
  validator <- CoherenceValidator$new(
    hierarchy = hierarchy,
    tolerance = tolerance,
    strict = TRUE
  )
  validator$validate(forecasts)
  invisible(TRUE)
}


#' Fix coherence by adjusting aggregates
#'
#' Forces coherence by recalculating aggregates from bottom level.
#'
#' @param forecasts Named list of forecast vectors
#' @param hierarchy HierarchyBuilder instance
#' @return Named list with coherent forecasts
#' @export
force_coherence <- function(forecasts, hierarchy) {
  S <- hierarchy$build_summing_matrix()
  bottom_series <- hierarchy$get_bottom_series()
  all_series <- hierarchy$get_all_series()

  # Get bottom-level forecasts
  bottom_forecasts <- forecasts[bottom_series]
  bottom_mat <- do.call(cbind, bottom_forecasts)

  # Calculate all levels using summing matrix
  reconciled_mat <- t(S %*% t(bottom_mat))
  colnames(reconciled_mat) <- all_series

  # Convert back to list
  reconciled <- lapply(seq_len(ncol(reconciled_mat)), function(i) {
    reconciled_mat[, i]
  })
  names(reconciled) <- all_series

  reconciled
}
```

### Usage Example

```r
# Create hierarchy and validator
hierarchy <- create_sin_hierarchy()
validator <- CoherenceValidator$new(
  hierarchy = hierarchy,
  tolerance = 1e-6,
  strict = FALSE
)

# Forecasts that may not be coherent
forecasts <- list(
  SIN = sin_forecast,
  SECO = seco_forecast,
  S = s_forecast,
  NE = ne_forecast,
  N = n_forecast,
  RJ = rj_forecast,
  SP = sp_forecast,
  # ... all areas and loss areas
)

# Validate
results <- validator$validate(forecasts)
print(results)
#    constraint constraint_type     expected_sum_of n_time_points   mean_error max_abs_error is_coherent
# 1:        SIN        national SECO + S + NE + N           168  0.000000e+00   0.000000e+00        TRUE
# 2:       SECO       subsystem   RJ + SP + ... + PESE        168 -5.234567e-02   1.234500e+01       FALSE
# 3:          S       subsystem     PR + SC + RS + PES        168  0.000000e+00   0.000000e+00        TRUE
# 4:         NE       subsystem ALPE + ... + PENE             168  0.000000e+00   0.000000e+00        TRUE
# 5:          N       subsystem   AM + ... + PEN               168  0.000000e+00   0.000000e+00        TRUE

# Check coherence
is_ok <- validator$is_coherent(forecasts)
# FALSE

# Get error time series for analysis
error_ts <- validator$get_error_timeseries(forecasts)

# Convenience functions
is_hierarchically_coherent(forecasts, hierarchy)
# FALSE

# Force coherence (recalculate aggregates)
coherent_forecasts <- force_coherence(forecasts, hierarchy)
is_hierarchically_coherent(coherent_forecasts, hierarchy)
# TRUE

# Assert coherence (throws error if not coherent)
tryCatch(
  assert_hierarchically_coherent(forecasts, hierarchy),
  error = function(e) message("Not coherent: ", e$message)
)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with hierarchy | Validator created |
| TC-002 | validate() coherent | All is_coherent TRUE |
| TC-003 | validate() incoherent national | is_coherent FALSE for SIN |
| TC-004 | validate() incoherent subsystem | is_coherent FALSE for subsystem |
| TC-005 | validate() strict mode | Error thrown |
| TC-006 | validate() non-strict mode | Warning issued |
| TC-007 | is_coherent() TRUE | Returns TRUE |
| TC-008 | is_coherent() FALSE | Returns FALSE |
| TC-009 | get_error_timeseries() | Error by time point |
| TC-010 | force_coherence() | Output is coherent |
| TC-011 | assert_hierarchically_coherent() coherent | No error |
| TC-012 | assert_hierarchically_coherent() incoherent | Error thrown |
| TC-013 | Tolerance respected | Small errors pass |

---

## Dependencies

- PC-041-06: HierarchyBuilder

---

## Definition of Done

- [ ] CoherenceValidator R6 class implemented
- [ ] National and subsystem validation
- [ ] Strict and non-strict modes
- [ ] Convenience functions
- [ ] force_coherence utility
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Coherence violations indicate either:
  1. Reconciliation wasn't applied
  2. Numerical precision issues
  3. Incorrect forecast aggregation
- force_coherence is a "quick fix" but loses information
- Consider adding visualization of coherence errors
- Error timeseries useful for debugging patterns
