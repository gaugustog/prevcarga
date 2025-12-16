# PC-042-06: ReconciliationWorkflow

**Epic:** [EPIC-06: Reconciliation Infrastructure](../epics/EPIC-06-reconciliation-layer.md)
**Task Reference:** T-06.4
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `ReconciliationWorkflow` R6 class that orchestrates the complete hierarchical reconciliation process, including loss calculation and coherence validation.

---

## Acceptance Criteria

- [ ] Workflow module created in `R/reconciliation/workflow.R`
- [ ] `ReconciliationWorkflow` R6 class with full orchestration
- [ ] Integration with reconcilers and hierarchy builder
- [ ] Loss calculation by difference
- [ ] Coherence validation after reconciliation
- [ ] ReconciliationResult with metadata

---

## Technical Specification

### File Location
```
R/reconciliation/workflow.R
```

### ReconciliationWorkflow R6 Class

```r
#' @title ReconciliationWorkflow
#' @description Orchestrates hierarchical reconciliation process
#'
#' Manages the complete reconciliation pipeline:
#' 1. Validate input forecasts
#' 2. Apply reconciliation method
#' 3. Calculate loss areas (if by_difference)
#' 4. Validate coherence
#' 5. Return reconciled forecasts
#'
#' @export
ReconciliationWorkflow <- R6::R6Class(
  "ReconciliationWorkflow",
  private = list(
    reconciler = NULL,
    hierarchy_builder = NULL,
    loss_calculator = NULL,
    config = NULL
  ),
  public = list(
    #' @description Initialize workflow
    #' @param reconciler BaseReconciler instance or name
    #' @param hierarchy_config Hierarchy configuration or HierarchyBuilder
    #' @param loss_calculator LossCalculator instance (optional)
    #' @param config Workflow configuration
    initialize = function(reconciler,
                          hierarchy_config = NULL,
                          loss_calculator = NULL,
                          config = list()) {
      # Accept reconciler name or instance
      if (is.character(reconciler)) {
        private$reconciler <- get_reconciler(reconciler, config = config$reconciler_config %||% list())
      } else {
        checkmate::assert_class(reconciler, "BaseReconciler")
        private$reconciler <- reconciler
      }

      # Accept hierarchy config or builder
      if (inherits(hierarchy_config, "HierarchyBuilder")) {
        private$hierarchy_builder <- hierarchy_config
      } else {
        private$hierarchy_builder <- HierarchyBuilder$new(config = hierarchy_config)
      }

      # Optional loss calculator
      if (!is.null(loss_calculator)) {
        checkmate::assert_class(loss_calculator, "LossCalculator")
      }
      private$loss_calculator <- loss_calculator %||% LossCalculator$new()

      private$config <- private$apply_default_config(config)
    },

    #' @description Run reconciliation workflow
    #' @param forecasts Named list of forecast vectors (one per series)
    #' @param ... Additional arguments passed to reconciler
    #' @return ReconciliationResult with reconciled forecasts
    run = function(forecasts, ...) {
      checkmate::assert_list(forecasts, names = "named")

      start_time <- Sys.time()

      # Step 1: Validate input forecasts
      message("Validating input forecasts...")
      self$validate_input(forecasts)

      # Step 2: Calculate loss areas if needed (before reconciliation)
      if (private$config$calculate_losses_first) {
        message("Calculating loss areas by difference...")
        forecasts <- self$calculate_losses(forecasts)
      }

      # Step 3: Apply reconciliation
      message(sprintf("Applying %s reconciliation...",
                      class(private$reconciler)[1]))
      reconciled <- private$reconciler$reconcile(
        forecasts,
        private$hierarchy_builder,
        ...
      )

      # Step 4: Calculate loss areas if needed (after reconciliation)
      if (!private$config$calculate_losses_first && private$config$calculate_losses) {
        message("Calculating loss areas by difference...")
        reconciled <- self$calculate_losses(reconciled)
      }

      # Step 5: Validate coherence
      if (private$config$validate_coherence) {
        message("Validating hierarchical coherence...")
        self$validate_coherence(reconciled)
      }

      # Build result
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      ReconciliationResult$new(
        predictions = reconciled,
        summing_matrix = private$hierarchy_builder$build_summing_matrix(),
        hierarchy = private$hierarchy_builder,
        reconciler_name = class(private$reconciler)[1],
        elapsed_time = elapsed,
        config = private$config
      )
    },

    #' @description Validate input forecasts
    #' @param forecasts Named list of forecasts
    #' @return TRUE if valid, error otherwise
    validate_input = function(forecasts) {
      # Get required series (excluding loss areas if computed by difference)
      all_series <- private$hierarchy_builder$get_all_series()
      loss_areas <- names(private$hierarchy_builder$get_loss_area_config())

      by_diff_losses <- character(0)
      for (la in loss_areas) {
        la_config <- private$hierarchy_builder$get_loss_area_config(la)
        if (la_config$method == "by_difference") {
          by_diff_losses <- c(by_diff_losses, la)
        }
      }

      # Required = all series except by_difference loss areas
      required <- setdiff(all_series, by_diff_losses)

      missing <- setdiff(required, names(forecasts))
      if (length(missing) > 0) {
        stop(sprintf(
          "Missing required forecasts: %s",
          paste(missing, collapse = ", ")
        ))
      }

      # Check forecast lengths
      lengths <- sapply(forecasts, length)
      if (length(unique(lengths)) > 1) {
        stop("All forecasts must have the same length")
      }

      invisible(TRUE)
    },

    #' @description Calculate loss areas by difference
    #' @param forecasts Named list of forecasts
    #' @return Updated forecasts with loss areas
    calculate_losses = function(forecasts) {
      loss_config <- private$hierarchy_builder$get_loss_area_config()

      for (la in names(loss_config)) {
        la_info <- loss_config[[la]]

        if (la_info$method == "by_difference") {
          loss_forecast <- private$loss_calculator$calculate_loss(
            subsystem = la_info$subsystem,
            loss_area_code = la,
            forecasts = forecasts,
            hierarchy = private$hierarchy_builder
          )

          forecasts[[la]] <- loss_forecast
          message(sprintf("  Calculated %s by difference", la))
        }
      }

      forecasts
    },

    #' @description Validate hierarchical coherence
    #' @param forecasts Named list of forecasts
    #' @return TRUE if coherent, error otherwise
    validate_coherence = function(forecasts) {
      tolerance <- private$config$coherence_tolerance

      # Check national = sum of subsystems
      national <- private$hierarchy_builder$get_config()$national
      subsystems <- names(private$hierarchy_builder$get_config()$subsystems)

      national_forecast <- forecasts[[national]]
      subsystem_sum <- Reduce(`+`, forecasts[subsystems])

      max_diff <- max(abs(national_forecast - subsystem_sum))
      if (max_diff > tolerance) {
        if (private$config$strict_coherence) {
          stop(sprintf(
            "Coherence violation: %s != sum(subsystems), max diff = %.4f",
            national, max_diff
          ))
        } else {
          warning(sprintf(
            "Coherence violation: %s != sum(subsystems), max diff = %.4f",
            national, max_diff
          ))
        }
      }

      # Check each subsystem = sum of its areas
      for (subsystem in subsystems) {
        children <- private$hierarchy_builder$get_children(subsystem)
        subsystem_forecast <- forecasts[[subsystem]]
        children_sum <- Reduce(`+`, forecasts[children])

        max_diff <- max(abs(subsystem_forecast - children_sum))
        if (max_diff > tolerance) {
          if (private$config$strict_coherence) {
            stop(sprintf(
              "Coherence violation: %s != sum(areas), max diff = %.4f",
              subsystem, max_diff
            ))
          } else {
            warning(sprintf(
              "Coherence violation: %s != sum(areas), max diff = %.4f",
              subsystem, max_diff
            ))
          }
        }
      }

      message("  Hierarchical coherence validated")
      invisible(TRUE)
    },

    #' @description Get the reconciler
    #' @return BaseReconciler instance
    get_reconciler = function() {
      private$reconciler
    },

    #' @description Get the hierarchy builder
    #' @return HierarchyBuilder instance
    get_hierarchy_builder = function() {
      private$hierarchy_builder
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        calculate_losses = TRUE,
        calculate_losses_first = TRUE,  # Before or after reconciliation
        validate_coherence = TRUE,
        strict_coherence = FALSE,
        coherence_tolerance = 1e-6
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

### ReconciliationResult Class

```r
#' @title ReconciliationResult
#' @description Container for reconciliation results with metadata
#' @export
ReconciliationResult <- R6::R6Class(
  "ReconciliationResult",
  public = list(
    predictions = NULL,
    summing_matrix = NULL,
    hierarchy = NULL,
    reconciler_name = NULL,
    elapsed_time = NULL,
    config = NULL,
    created_at = NULL,

    #' @description Initialize result
    initialize = function(predictions,
                          summing_matrix,
                          hierarchy,
                          reconciler_name,
                          elapsed_time,
                          config) {
      self$predictions <- predictions
      self$summing_matrix <- summing_matrix
      self$hierarchy <- hierarchy
      self$reconciler_name <- reconciler_name
      self$elapsed_time <- elapsed_time
      self$config <- config
      self$created_at <- Sys.time()
    },

    #' @description Get predictions as data.table
    #' @param datetime Optional datetime vector
    #' @return data.table in long format
    as_data_table = function(datetime = NULL) {
      rows <- lapply(names(self$predictions), function(series) {
        dt <- data.table::data.table(
          series = series,
          prediction = self$predictions[[series]]
        )
        if (!is.null(datetime)) {
          dt[, datetime := datetime]
        }
        dt
      })

      data.table::rbindlist(rows)
    },

    #' @description Get predictions for specific level
    #' @param level Level name: "national", "subsystems", "areas", "loss_areas"
    #' @return Named list of forecasts
    get_level = function(level) {
      levels <- self$hierarchy$get_aggregation_levels()

      if (!level %in% names(levels)) {
        stop(sprintf("Unknown level: %s", level))
      }

      series <- levels[[level]]
      self$predictions[series]
    },

    #' @description Get predictions for specific subsystem and its areas
    #' @param subsystem Subsystem code
    #' @return Named list with subsystem and area forecasts
    get_subsystem = function(subsystem) {
      children <- self$hierarchy$get_children(subsystem)

      forecasts <- list()
      forecasts[[subsystem]] <- self$predictions[[subsystem]]

      for (child in children) {
        forecasts[[child]] <- self$predictions[[child]]
      }

      forecasts
    },

    #' @description Verify coherence
    #' @return data.table with coherence check results
    check_coherence = function() {
      calculate_coherence_error(self$predictions, self$hierarchy)
    },

    #' @description Print result summary
    print = function() {
      cat("ReconciliationResult\n")
      cat(sprintf("  Reconciler: %s\n", self$reconciler_name))
      cat(sprintf("  Series: %d\n", length(self$predictions)))
      cat(sprintf("  Time points: %d\n", length(self$predictions[[1]])))
      cat(sprintf("  Elapsed time: %.3f sec\n", self$elapsed_time))

      # Show level breakdown
      levels <- self$hierarchy$get_aggregation_levels()
      cat("\nLevel breakdown:\n")
      for (level in names(levels)) {
        cat(sprintf("  %s: %d series\n", level, length(levels[[level]])))
      }

      invisible(self)
    }
  )
)
```

### Convenience Function

```r
#' Reconcile forecasts using specified method
#'
#' @param forecasts Named list of forecasts
#' @param method Reconciliation method name
#' @param hierarchy_config Hierarchy configuration
#' @param ... Additional arguments
#' @return ReconciliationResult
#' @export
reconcile_forecasts <- function(forecasts,
                                method = "bottom_up",
                                hierarchy_config = NULL,
                                ...) {
  workflow <- ReconciliationWorkflow$new(
    reconciler = method,
    hierarchy_config = hierarchy_config
  )

  workflow$run(forecasts, ...)
}
```

### Usage Example

```r
# Create workflow
workflow <- ReconciliationWorkflow$new(
  reconciler = "bottom_up",
  hierarchy_config = NULL,  # Uses default SIN
  config = list(
    calculate_losses = TRUE,
    validate_coherence = TRUE
  )
)

# Prepare forecasts (excluding loss areas - computed by difference)
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

# Run reconciliation
result <- workflow$run(forecasts)

print(result)
# ReconciliationResult
#   Reconciler: BottomUpReconciler
#   Series: 31
#   Time points: 168
#   Elapsed time: 0.456 sec
#
# Level breakdown:
#   national: 1 series
#   subsystems: 4 series
#   areas: 23 series
#   loss_areas: 4 series

# Get reconciled predictions
reconciled_dt <- result$as_data_table(datetime = forecast_datetimes)

# Get specific subsystem
seco_data <- result$get_subsystem("SECO")

# Check coherence
coherence <- result$check_coherence()
#    series mean_error max_abs_error is_coherent
# 1:    SIN    0.0000        0.0000        TRUE
# 2:   SECO    0.0000        0.0000        TRUE
# ...

# Convenience function
result <- reconcile_forecasts(forecasts, method = "bottom_up")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with reconciler name | Reconciler looked up |
| TC-002 | Initialize with reconciler instance | Instance stored |
| TC-003 | run() basic workflow | Returns ReconciliationResult |
| TC-004 | validate_input() valid | TRUE |
| TC-005 | validate_input() missing series | Error |
| TC-006 | calculate_losses() | Loss areas computed |
| TC-007 | validate_coherence() coherent | TRUE |
| TC-008 | validate_coherence() incoherent | Warning/Error |
| TC-009 | ReconciliationResult as_data_table() | Long format |
| TC-010 | ReconciliationResult get_level() | Level forecasts |
| TC-011 | ReconciliationResult get_subsystem() | Subsystem + areas |
| TC-012 | reconcile_forecasts() convenience | Works correctly |

---

## Dependencies

- PC-039-06: BaseReconciler
- PC-040-06: ReconcilerRegistry
- PC-041-06: HierarchyBuilder
- PC-043-06: LossCalculator

---

## Definition of Done

- [ ] ReconciliationWorkflow R6 class implemented
- [ ] ReconciliationResult class implemented
- [ ] Loss calculation integration
- [ ] Coherence validation
- [ ] Convenience function
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Loss areas can be calculated before or after reconciliation
- Strict coherence mode throws errors, non-strict warns
- Consider adding residual diagnostics in future
- May want to add reconciliation matrices to result for analysis
