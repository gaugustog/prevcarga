# PC-043-06: LossCalculator

**Epic:** [EPIC-06: Reconciliation Infrastructure](../epics/EPIC-06-reconciliation-layer.md)
**Task Reference:** T-06.5
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Implement the `LossCalculator` R6 class for computing loss area forecasts by difference. Loss areas (PESE, PES, PENE, PEN) represent technical losses in each subsystem and can be calculated as: Loss = Subsystem - Σ(areas).

---

## Acceptance Criteria

- [ ] Loss calculation module created in `R/reconciliation/loss_calculation.R`
- [ ] `LossCalculator` R6 class with calculation methods
- [ ] Support for by_difference calculation
- [ ] Support for model-based loss forecasts (PENE)
- [ ] Negative loss handling (warning/clamp)
- [ ] Integration with HierarchyBuilder

---

## Technical Specification

### File Location
```
R/reconciliation/loss_calculation.R
```

### LossCalculator R6 Class

```r
#' @title LossCalculator
#' @description Calculate loss area forecasts by difference or model
#'
#' Loss areas represent technical losses in each subsystem:
#' - PESE: SECO losses
#' - PES: S losses
#' - PENE: NE losses (typically modeled)
#' - PEN: N losses
#'
#' For by_difference method: Loss = Subsystem - Σ(areas)
#'
#' @export
LossCalculator <- R6::R6Class(
  "LossCalculator",
  private = list(
    config = NULL,
    min_loss_threshold = NULL,
    clamp_negative = NULL
  ),
  public = list(
    #' @description Initialize loss calculator
    #' @param config Configuration list
    initialize = function(config = list()) {
      private$config <- private$apply_default_config(config)
      private$min_loss_threshold <- private$config$min_loss_threshold
      private$clamp_negative <- private$config$clamp_negative
    },

    #' @description Calculate loss for a subsystem
    #' @param subsystem Subsystem code (SECO, S, NE, N)
    #' @param loss_area_code Loss area code (PESE, PES, PENE, PEN)
    #' @param forecasts Named list of forecasts (subsystem + areas)
    #' @param hierarchy HierarchyBuilder instance
    #' @return Numeric vector of loss forecasts
    calculate_loss = function(subsystem, loss_area_code, forecasts, hierarchy) {
      checkmate::assert_string(subsystem)
      checkmate::assert_string(loss_area_code)
      checkmate::assert_list(forecasts, names = "named")

      # Get loss area configuration
      loss_config <- hierarchy$get_loss_area_config(loss_area_code)

      if (is.null(loss_config)) {
        stop(sprintf("Unknown loss area: %s", loss_area_code))
      }

      if (loss_config$subsystem != subsystem) {
        stop(sprintf(
          "Loss area %s belongs to %s, not %s",
          loss_area_code, loss_config$subsystem, subsystem
        ))
      }

      if (loss_config$method == "by_difference") {
        self$calculate_by_difference(subsystem, forecasts, hierarchy)
      } else if (loss_config$method == "model") {
        self$get_model_forecast(loss_area_code, forecasts)
      } else {
        stop(sprintf("Unknown loss method: %s", loss_config$method))
      }
    },

    #' @description Calculate loss by difference
    #' @param subsystem Subsystem code
    #' @param forecasts Named list of forecasts
    #' @param hierarchy HierarchyBuilder instance
    #' @return Numeric vector of loss forecasts
    calculate_by_difference = function(subsystem, forecasts, hierarchy) {
      # Get children of subsystem (areas, excluding loss area)
      children <- hierarchy$get_children(subsystem)
      loss_areas <- names(hierarchy$get_loss_area_config())

      # Filter out loss areas - we're calculating, not using
      areas <- setdiff(children, loss_areas)

      # Validate subsystem forecast exists
      if (!subsystem %in% names(forecasts)) {
        stop(sprintf("Missing subsystem forecast: %s", subsystem))
      }

      # Validate all area forecasts exist
      missing_areas <- setdiff(areas, names(forecasts))
      if (length(missing_areas) > 0) {
        stop(sprintf(
          "Missing area forecasts for %s: %s",
          subsystem, paste(missing_areas, collapse = ", ")
        ))
      }

      # Calculate: Loss = Subsystem - Σ(areas)
      subsystem_forecast <- forecasts[[subsystem]]
      area_sum <- Reduce(`+`, forecasts[areas])

      loss_forecast <- subsystem_forecast - area_sum

      # Handle negative losses
      if (any(loss_forecast < 0, na.rm = TRUE)) {
        n_negative <- sum(loss_forecast < 0, na.rm = TRUE)
        min_value <- min(loss_forecast, na.rm = TRUE)

        if (private$clamp_negative) {
          warning(sprintf(
            "Clamping %d negative loss values for %s (min: %.2f)",
            n_negative, subsystem, min_value
          ))
          loss_forecast <- pmax(loss_forecast, private$min_loss_threshold)
        } else {
          warning(sprintf(
            "%d negative loss values for %s (min: %.2f)",
            n_negative, subsystem, min_value
          ))
        }
      }

      loss_forecast
    },

    #' @description Get model-based loss forecast
    #' @param loss_area_code Loss area code
    #' @param forecasts Named list of forecasts
    #' @return Numeric vector of loss forecasts
    get_model_forecast = function(loss_area_code, forecasts) {
      if (!loss_area_code %in% names(forecasts)) {
        stop(sprintf(
          "Model-based loss area %s not in forecasts. Must be provided.",
          loss_area_code
        ))
      }

      forecasts[[loss_area_code]]
    },

    #' @description Calculate all loss areas for all subsystems
    #' @param forecasts Named list of forecasts
    #' @param hierarchy HierarchyBuilder instance
    #' @return Named list with loss area forecasts added
    calculate_all_losses = function(forecasts, hierarchy) {
      loss_config <- hierarchy$get_loss_area_config()

      for (loss_area in names(loss_config)) {
        la_info <- loss_config[[loss_area]]

        if (la_info$method == "by_difference") {
          # Only calculate if not already provided
          if (!loss_area %in% names(forecasts)) {
            forecasts[[loss_area]] <- self$calculate_loss(
              subsystem = la_info$subsystem,
              loss_area_code = loss_area,
              forecasts = forecasts,
              hierarchy = hierarchy
            )
          }
        }
      }

      forecasts
    },

    #' @description Validate loss area forecasts
    #' @param forecasts Named list of forecasts
    #' @param hierarchy HierarchyBuilder instance
    #' @return data.table with validation results
    validate_losses = function(forecasts, hierarchy) {
      loss_config <- hierarchy$get_loss_area_config()

      results <- lapply(names(loss_config), function(loss_area) {
        la_info <- loss_config[[loss_area]]
        subsystem <- la_info$subsystem

        # Recalculate by difference
        expected <- self$calculate_by_difference(subsystem, forecasts, hierarchy)
        actual <- forecasts[[loss_area]]

        diff <- actual - expected
        max_diff <- max(abs(diff), na.rm = TRUE)

        data.table::data.table(
          loss_area = loss_area,
          subsystem = subsystem,
          method = la_info$method,
          mean_actual = mean(actual, na.rm = TRUE),
          mean_expected = mean(expected, na.rm = TRUE),
          max_abs_diff = max_diff,
          is_consistent = max_diff < 1e-6 || la_info$method == "model"
        )
      })

      data.table::rbindlist(results)
    },

    #' @description Get configuration
    #' @return Configuration list
    get_config = function() {
      private$config
    },

    #' @description Print summary
    print = function() {
      cat("LossCalculator\n")
      cat(sprintf("  Clamp negative: %s\n", private$clamp_negative))
      cat(sprintf("  Min threshold: %.2f\n", private$min_loss_threshold))
      invisible(self)
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        clamp_negative = TRUE,
        min_loss_threshold = 0,
        warn_on_negative = TRUE
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

### Factory Function

```r
#' Create a loss calculator
#'
#' @param config Optional configuration
#' @return LossCalculator instance
#' @export
create_loss_calculator <- function(config = list()) {
  LossCalculator$new(config = config)
}
```

### Usage Example

```r
# Create calculator
calculator <- LossCalculator$new()

# Create hierarchy
hierarchy <- create_sin_hierarchy()

# Forecasts (without loss areas)
forecasts <- list(
  SIN = sin_forecast,
  SECO = seco_forecast,
  S = s_forecast,
  NE = ne_forecast,
  N = n_forecast,
  RJ = rj_forecast,
  SP = sp_forecast,
  MG = mg_forecast,
  # ... all other areas
)

# Calculate single loss area
pese_forecast <- calculator$calculate_loss(
  subsystem = "SECO",
  loss_area_code = "PESE",
  forecasts = forecasts,
  hierarchy = hierarchy
)

# Calculate all loss areas at once
forecasts_with_losses <- calculator$calculate_all_losses(forecasts, hierarchy)
# Now includes PESE, PES, PEN (PENE must be provided as it uses "model" method)

# Validate loss calculations
validation <- calculator$validate_losses(forecasts_with_losses, hierarchy)
#    loss_area subsystem        method mean_actual mean_expected max_abs_diff is_consistent
# 1:      PESE      SECO by_difference      1234.5        1234.5       0.0000          TRUE
# 2:       PES         S by_difference       456.7         456.7       0.0000          TRUE
# 3:      PENE        NE         model       789.0         752.3      45.2000          TRUE
# 4:       PEN         N by_difference       234.5         234.5       0.0000          TRUE
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | Default config applied |
| TC-002 | calculate_loss() by_difference | Correct difference |
| TC-003 | calculate_loss() model method | Returns provided forecast |
| TC-004 | calculate_loss() missing subsystem | Error thrown |
| TC-005 | calculate_loss() missing areas | Error thrown |
| TC-006 | calculate_by_difference() negative | Warning + clamp |
| TC-007 | calculate_by_difference() clamp=FALSE | Warning only |
| TC-008 | calculate_all_losses() | All by_diff computed |
| TC-009 | validate_losses() consistent | is_consistent TRUE |
| TC-010 | validate_losses() model method | Always TRUE |
| TC-011 | get_model_forecast() missing | Error thrown |

---

## Dependencies

- PC-041-06: HierarchyBuilder

---

## Definition of Done

- [ ] LossCalculator R6 class implemented
- [ ] by_difference calculation working
- [ ] model method support
- [ ] Negative loss handling
- [ ] Validation utilities
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Loss areas represent grid transmission losses
- PENE typically uses model-based forecasts (distinct pattern)
- Negative losses indicate measurement/forecast issues
- Consider adding loss percentage validation in future
- May add temporal loss patterns (higher at peak hours)
