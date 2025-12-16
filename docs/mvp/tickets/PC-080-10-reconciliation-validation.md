# PC-080-10: Reconciliation Validation

**Epic:** [EPIC-10: Testing & Validation](../epics/EPIC-10-testing-validation.md)
**Task Reference:** T-10.5
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Validate that reconciliation strategies maintain hierarchical consistency across the Brazilian power grid structure (SIN → Subsystems → Areas), verify loss calculations, and ensure all registered reconcilers function correctly.

---

## Acceptance Criteria

- [ ] Reconciliation validation script in `tests/validation/reconciliation.R`
- [ ] All registered reconcilers tested
- [ ] Hierarchical consistency verified (SIN = Σ Subsystems)
- [ ] Subsystem aggregation validated (Subsystem = Σ Areas + Loss)
- [ ] Loss calculations verified
- [ ] Reconciliation report generated

---

## Technical Specification

### File Location
```
tests/validation/reconciliation.R
```

### Reconciliation Validation Framework

```r
#' @title ReconciliationValidator
#' @description Validates reconciliation strategies for hierarchical consistency
#' @export
ReconciliationValidator <- R6::R6Class(
  "ReconciliationValidator",
  private = list(
    config = NULL,
    storage = NULL,
    hierarchy = NULL,
    predictions = NULL,
    results = list()
  ),

  public = list(
    #' @description Initialize validator
    #' @param config ConfigManager instance
    initialize = function(config) {
      private$config <- config
      private$storage <- create_storage_backend(config$get_storage_config())
      private$hierarchy <- self$load_hierarchy()
    },

    #' @description Load SIN hierarchy structure
    #' @return Hierarchy definition
    load_hierarchy = function() {
      # Brazilian SIN hierarchy
      list(
        subsystems = list(
          SECO = c("RJ", "SP", "MG", "ES", "GO", "MS", "DF", "MT", "AC", "RO"),
          S = c("PR", "SC", "RS"),
          NE = c("BA", "SE", "AL", "PE", "PB", "RN", "CE", "PI", "MA"),
          N = c("PA", "AP", "AM", "TO", "RR")
        ),
        loss_factors = list(
          SECO = 0.025,
          S = 0.022,
          NE = 0.028,
          N = 0.030
        )
      )
    },

    #' @description Load predictions for reconciliation testing
    #' @param model Model name
    #' @param start_date Start date
    #' @param end_date End date
    load_predictions = function(model, start_date, end_date) {
      loader <- DataLoader$new(private$storage)

      all_areas <- unlist(private$hierarchy$subsystems)

      private$predictions <- loader$load_predictions(
        model = model,
        areas = all_areas,
        start_date = start_date,
        end_date = end_date
      )

      cat(sprintf("Loaded %d predictions for %d areas\n",
                  nrow(private$predictions), length(all_areas)))
    },

    #' @description Validate all registered reconcilers
    #' @return List of validation results
    validate_all_reconcilers = function() {
      reconcilers <- list_reconcilers()

      cat("\n=== Reconciliation Validation ===\n\n")

      for (reconciler_name in reconcilers) {
        cat(sprintf("Testing reconciler: %s\n", reconciler_name))

        tryCatch({
          result <- self$validate_reconciler(reconciler_name)
          private$results[[reconciler_name]] <- result

          status <- if (result$hierarchical_consistent) "\u2713" else "\u2717"
          cat(sprintf("  %s Hierarchical consistency: %s\n",
                      status,
                      if (result$hierarchical_consistent) "PASSED" else "FAILED"))

          cat(sprintf("    Max SIN discrepancy: %.4f%%\n",
                      result$max_sin_discrepancy * 100))

        }, error = function(e) {
          cat(sprintf("  \u2717 ERROR: %s\n", conditionMessage(e)))
          private$results[[reconciler_name]] <- list(
            reconciler = reconciler_name,
            error = conditionMessage(e)
          )
        })
      }

      private$results
    },

    #' @description Validate single reconciler
    #' @param reconciler_name Reconciler name
    #' @return Validation result
    validate_reconciler = function(reconciler_name) {
      reconciler <- get_reconciler(reconciler_name)

      # Build summing matrix
      S <- self$build_summing_matrix()

      # Apply reconciliation
      reconciled <- reconciler$reconcile(
        base_forecasts = private$predictions,
        summing_matrix = S,
        hierarchy = private$hierarchy
      )

      # Validate hierarchical consistency
      consistency <- self$check_hierarchical_consistency(reconciled)

      # Validate loss calculations
      loss_validation <- self$validate_loss_calculations(reconciled)

      list(
        reconciler = reconciler_name,
        hierarchical_consistent = consistency$consistent,
        max_sin_discrepancy = consistency$max_sin_discrepancy,
        mean_sin_discrepancy = consistency$mean_sin_discrepancy,
        subsystem_consistency = consistency$subsystem_details,
        loss_validation = loss_validation,
        n_observations = nrow(reconciled)
      )
    },

    #' @description Build summing matrix for hierarchy
    #' @return Summing matrix S
    build_summing_matrix = function() {
      areas <- unlist(private$hierarchy$subsystems)
      subsystems <- names(private$hierarchy$subsystems)

      n_bottom <- length(areas)
      n_subsystems <- length(subsystems)
      n_total <- n_bottom + n_subsystems + 1  # areas + subsystems + SIN

      S <- matrix(0, nrow = n_total, ncol = n_bottom)

      # Bottom level (areas): identity
      S[1:n_bottom, ] <- diag(n_bottom)

      # Subsystem level
      row_idx <- n_bottom + 1
      for (sub in subsystems) {
        sub_areas <- private$hierarchy$subsystems[[sub]]
        col_indices <- which(areas %in% sub_areas)
        S[row_idx, col_indices] <- 1
        row_idx <- row_idx + 1
      }

      # SIN level (top)
      S[n_total, ] <- 1

      rownames(S) <- c(areas, subsystems, "SIN")
      colnames(S) <- areas

      S
    },

    #' @description Check hierarchical consistency
    #' @param reconciled Reconciled predictions
    #' @return Consistency check results
    check_hierarchical_consistency = function(reconciled) {
      # Group by datetime and calculate aggregates
      by_datetime <- split(reconciled, reconciled$datetime)

      sin_discrepancies <- c()
      subsystem_details <- list()

      for (dt in names(by_datetime)) {
        dt_data <- by_datetime[[dt]]

        # Calculate SIN as sum of subsystems
        sin_calculated <- 0
        subsystem_sums <- list()

        for (sub in names(private$hierarchy$subsystems)) {
          sub_areas <- private$hierarchy$subsystems[[sub]]
          sub_data <- dt_data[dt_data$area_code %in% sub_areas, ]

          area_sum <- sum(sub_data$prediction, na.rm = TRUE)
          loss <- area_sum * private$hierarchy$loss_factors[[sub]]
          subsystem_total <- area_sum + loss

          subsystem_sums[[sub]] <- list(
            area_sum = area_sum,
            loss = loss,
            total = subsystem_total
          )

          sin_calculated <- sin_calculated + subsystem_total
        }

        # Get SIN prediction if available
        sin_pred <- dt_data[dt_data$area_code == "SIN", "prediction"]
        if (length(sin_pred) > 0 && !is.na(sin_pred)) {
          discrepancy <- abs(sin_pred - sin_calculated) / sin_calculated
          sin_discrepancies <- c(sin_discrepancies, discrepancy)
        }

        subsystem_details[[dt]] <- subsystem_sums
      }

      # Tolerance for floating point
      tolerance <- 1e-6

      list(
        consistent = all(sin_discrepancies < tolerance, na.rm = TRUE),
        max_sin_discrepancy = max(sin_discrepancies, na.rm = TRUE),
        mean_sin_discrepancy = mean(sin_discrepancies, na.rm = TRUE),
        subsystem_details = subsystem_details,
        n_checked = length(sin_discrepancies)
      )
    },

    #' @description Validate loss calculations
    #' @param reconciled Reconciled predictions
    #' @return Loss validation results
    validate_loss_calculations = function(reconciled) {
      by_datetime <- split(reconciled, reconciled$datetime)

      loss_errors <- list()

      for (sub in names(private$hierarchy$subsystems)) {
        expected_factor <- private$hierarchy$loss_factors[[sub]]
        sub_areas <- private$hierarchy$subsystems[[sub]]

        actual_losses <- c()
        expected_losses <- c()

        for (dt in names(by_datetime)) {
          dt_data <- by_datetime[[dt]]
          sub_data <- dt_data[dt_data$area_code %in% sub_areas, ]

          area_sum <- sum(sub_data$prediction, na.rm = TRUE)

          # Get subsystem prediction
          sub_pred <- dt_data[dt_data$area_code == sub, "prediction"]

          if (length(sub_pred) > 0 && !is.na(sub_pred)) {
            implied_loss <- sub_pred - area_sum
            expected_loss <- area_sum * expected_factor

            actual_losses <- c(actual_losses, implied_loss)
            expected_losses <- c(expected_losses, expected_loss)
          }
        }

        if (length(actual_losses) > 0) {
          loss_errors[[sub]] <- list(
            mean_actual = mean(actual_losses),
            mean_expected = mean(expected_losses),
            mean_diff = mean(abs(actual_losses - expected_losses)),
            correlation = cor(actual_losses, expected_losses)
          )
        }
      }

      loss_errors
    },

    #' @description Run full validation
    #' @return Complete validation report
    run_full_validation = function() {
      cat("\n")
      cat("=" |> rep(60) |> paste(collapse = ""))
      cat("\n  Reconciliation Validation Report\n")
      cat("=" |> rep(60) |> paste(collapse = ""))
      cat("\n\n")

      # Validate all reconcilers
      reconciler_results <- self$validate_all_reconcilers()

      # Summary
      n_passed <- sum(sapply(reconciler_results, function(r) {
        isTRUE(r$hierarchical_consistent)
      }))

      cat(sprintf("\n%d/%d reconcilers maintain hierarchical consistency\n",
                  n_passed, length(reconciler_results)))

      list(
        reconciler_validation = reconciler_results,
        hierarchy = private$hierarchy,
        timestamp = Sys.time()
      )
    },

    #' @description Generate summary table
    #' @return data.table summary
    summary = function() {
      data.table::rbindlist(lapply(names(private$results), function(name) {
        r <- private$results[[name]]

        if (!is.null(r$error)) {
          return(data.table::data.table(
            reconciler = name,
            consistent = NA,
            max_discrepancy = NA_real_,
            error = r$error
          ))
        }

        data.table::data.table(
          reconciler = name,
          consistent = r$hierarchical_consistent,
          max_discrepancy = r$max_sin_discrepancy,
          error = NA_character_
        )
      }))
    },

    #' @description Print summary
    print = function() {
      cat("\nReconciliation Validation Summary\n")
      cat("=================================\n\n")

      summary_dt <- self$summary()
      print(summary_dt)

      n_passed <- sum(summary_dt$consistent, na.rm = TRUE)
      n_total <- sum(!is.na(summary_dt$consistent))

      cat(sprintf("\n%d/%d reconcilers passed validation\n",
                  n_passed, n_total))

      invisible(self)
    }
  )
)
```

### Validation Runner Script

```r
#!/usr/bin/env Rscript
# tests/validation/reconciliation.R

library(prevcargaons)

# Load config
config <- load_config("config/config.yaml")

# Initialize validator
validator <- ReconciliationValidator$new(config)

# Load predictions
validator$load_predictions(
  model = "lgbm",
  start_date = "2024-01-01",
  end_date = "2024-03-31"
)

# Run full validation
results <- validator$run_full_validation()

# Print summary
print(validator)

# Check results
summary_dt <- validator$summary()

# Pass criteria:
# 1. All reconcilers maintain hierarchical consistency
# 2. SIN = Σ Subsystems (within tolerance)
# 3. Subsystem = Σ Areas + Loss

pass_all <- all(summary_dt$consistent, na.rm = TRUE)

if (pass_all) {
  cat("\n\u2713 Reconciliation validation PASSED\n")
  quit(status = 0)
} else {
  cat("\n\u2717 Reconciliation validation FAILED\n")
  cat("  Some reconcilers do not maintain hierarchical consistency\n")
  quit(status = 1)
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | SIN = Σ Subsystems | 100% consistency |
| TC-002 | Subsystem = Σ Areas + Loss | Correct aggregation |
| TC-003 | bottom_up reconciler | Passes consistency |
| TC-004 | ols reconciler | Passes consistency |
| TC-005 | wls reconciler | Passes consistency |
| TC-006 | mint reconciler | Passes consistency |
| TC-007 | Loss factor applied | Correct percentages |
| TC-008 | Summing matrix correct | Proper structure |
| TC-009 | All areas included | No missing areas |
| TC-010 | Summary generated | Table produced |

---

## Dependencies

- PC-039-06: BaseReconciler
- PC-040-06: ReconcilerRegistry
- PC-041-06: BottomUp Reconciler
- PC-042-06: OLS Reconciler
- PC-043-06: MinT Reconciler

---

## Definition of Done

- [ ] ReconciliationValidator class implemented
- [ ] All registered reconcilers tested
- [ ] Hierarchical consistency verified
- [ ] Loss calculations validated
- [ ] Summing matrix generation working
- [ ] Summary report generated
- [ ] roxygen2 documentation complete
- [ ] Validation passing
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Brazilian SIN hierarchy: SIN → Subsystems (SECO, S, NE, N) → Areas
- Loss factors are subsystem-specific (2.2% - 3.0%)
- Hierarchical consistency must be 100%
- Consider adding temporal aggregation tests
