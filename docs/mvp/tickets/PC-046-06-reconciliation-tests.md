# PC-046-06: Reconciliation Infrastructure Tests

**Epic:** [EPIC-06: Reconciliation Infrastructure](../epics/EPIC-06-reconciliation-layer.md)
**Task Reference:** T-06.8
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Create comprehensive test suite for the reconciliation infrastructure, covering all components: BaseReconciler, ReconcilerRegistry, HierarchyBuilder, ReconciliationWorkflow, LossCalculator, and CoherenceValidator.

---

## Acceptance Criteria

- [ ] Test file created at `tests/testthat/test-reconciliation.R`
- [ ] BaseReconciler contract tests
- [ ] ReconcilerRegistry CRUD tests
- [ ] HierarchyBuilder summing matrix tests
- [ ] ReconciliationWorkflow integration tests
- [ ] LossCalculator tests
- [ ] CoherenceValidator tests
- [ ] End-to-end workflow tests
- [ ] ≥80% code coverage

---

## Technical Specification

### Test Files

```
tests/testthat/
├── test-reconciliation-base.R        # BaseReconciler tests
├── test-reconciliation-registry.R    # ReconcilerRegistry tests
├── test-reconciliation-hierarchy.R   # HierarchyBuilder tests
├── test-reconciliation-workflow.R    # ReconciliationWorkflow tests
├── test-reconciliation-loss.R        # LossCalculator tests
├── test-reconciliation-coherence.R   # CoherenceValidator tests
└── test-reconciliation-integration.R # End-to-end tests
```

### Test Fixtures

```r
# tests/testthat/helper-reconciliation.R

#' Create test forecasts for reconciliation
#' @param n_time Number of time points
#' @param coherent If TRUE, forecasts sum correctly
#' @return Named list of forecasts
create_test_forecasts <- function(n_time = 168, coherent = TRUE) {
  set.seed(42)

  # Bottom-level forecasts (areas)
  areas <- c(
    "RJ", "SP", "MG", "ES", "MT", "MS", "AC", "RO", "DF", "GO",
    "PR", "SC", "RS",
    "ALPE", "PBRN", "BASE", "CE", "PI", "BAOE",
    "AM", "PA", "MA", "TO", "RR", "AP"
  )

  forecasts <- list()

  # Generate random area forecasts
  for (area in areas) {
    forecasts[[area]] <- 100 + rnorm(n_time, mean = 50, sd = 10)
  }

  # Loss areas
  forecasts[["PESE"]] <- rnorm(n_time, mean = 20, sd = 2)
  forecasts[["PES"]] <- rnorm(n_time, mean = 10, sd = 1)
  forecasts[["PENE"]] <- rnorm(n_time, mean = 15, sd = 1.5)
  forecasts[["PEN"]] <- rnorm(n_time, mean = 12, sd = 1)

  if (coherent) {
    # Calculate subsystems as sum of areas + loss
    seco_areas <- c("RJ", "SP", "MG", "ES", "MT", "MS", "AC", "RO", "DF", "GO")
    s_areas <- c("PR", "SC", "RS")
    ne_areas <- c("ALPE", "PBRN", "BASE", "CE", "PI", "BAOE")
    n_areas <- c("AM", "PA", "MA", "TO", "RR", "AP")

    forecasts[["SECO"]] <- Reduce(`+`, forecasts[seco_areas]) + forecasts[["PESE"]]
    forecasts[["S"]] <- Reduce(`+`, forecasts[s_areas]) + forecasts[["PES"]]
    forecasts[["NE"]] <- Reduce(`+`, forecasts[ne_areas]) + forecasts[["PENE"]]
    forecasts[["N"]] <- Reduce(`+`, forecasts[n_areas]) + forecasts[["PEN"]]

    # Calculate national as sum of subsystems
    forecasts[["SIN"]] <- forecasts[["SECO"]] + forecasts[["S"]] +
                          forecasts[["NE"]] + forecasts[["N"]]
  } else {
    # Add noise to aggregates (incoherent)
    forecasts[["SECO"]] <- 1000 + rnorm(n_time, mean = 500, sd = 50)
    forecasts[["S"]] <- 500 + rnorm(n_time, mean = 200, sd = 20)
    forecasts[["NE"]] <- 600 + rnorm(n_time, mean = 300, sd = 30)
    forecasts[["N"]] <- 400 + rnorm(n_time, mean = 200, sd = 20)
    forecasts[["SIN"]] <- 2500 + rnorm(n_time, mean = 1000, sd = 100)
  }

  forecasts
}


#' Create a simple test reconciler
#' @return BottomUpReconciler instance
create_test_reconciler <- function() {
  BottomUpReconciler$new(name = "test_bottom_up")
}


#' Create test hierarchy
#' @param custom If TRUE, uses custom config
#' @return HierarchyBuilder instance
create_test_hierarchy <- function(custom = FALSE) {
  if (custom) {
    config <- list(
      national = "TOTAL",
      subsystems = list(
        A = c("A1", "A2"),
        B = c("B1", "B2")
      ),
      loss_areas = list(
        LA = list(subsystem = "A", method = "by_difference")
      )
    )
  } else {
    config <- NULL  # Use default SIN
  }

  HierarchyBuilder$new(config = config)
}
```

### BaseReconciler Tests

```r
# tests/testthat/test-reconciliation-base.R

describe("BaseReconciler", {
  it("initializes with defaults", {
    reconciler <- BaseReconciler$new()

    expect_s3_class(reconciler, "BaseReconciler")
    expect_equal(reconciler$name, "BaseReconciler")
    expect_type(reconciler$config, "list")
  })

  it("initializes with custom name and config", {
    reconciler <- BaseReconciler$new(
      name = "custom",
      config = list(method = "ols")
    )

    expect_equal(reconciler$name, "custom")
    expect_equal(reconciler$config$method, "ols")
  })

  it("reconcile() throws error (abstract)", {
    reconciler <- BaseReconciler$new()
    hierarchy <- create_test_hierarchy()
    forecasts <- create_test_forecasts()

    expect_error(
      reconciler$reconcile(forecasts, hierarchy),
      "must be implemented"
    )
  })

  it("build_reconciliation_matrix() throws error (abstract)", {
    reconciler <- BaseReconciler$new()
    hierarchy <- create_test_hierarchy()

    expect_error(
      reconciler$build_reconciliation_matrix(hierarchy),
      "must be implemented"
    )
  })

  it("validate_hierarchy() validates correctly", {
    reconciler <- BaseReconciler$new()
    hierarchy <- create_test_hierarchy()
    forecasts <- create_test_forecasts()

    expect_true(reconciler$validate_hierarchy(forecasts, hierarchy))
  })

  it("validate_hierarchy() errors on missing series", {
    reconciler <- BaseReconciler$new()
    hierarchy <- create_test_hierarchy()
    forecasts <- list(SIN = 1:10)  # Missing other series

    expect_error(
      reconciler$validate_hierarchy(forecasts, hierarchy),
      "Missing forecasts"
    )
  })

  it("validate_hierarchy() errors on length mismatch", {
    reconciler <- BaseReconciler$new()
    hierarchy <- create_test_hierarchy()
    forecasts <- create_test_forecasts()
    forecasts$SIN <- 1:10  # Different length

    expect_error(
      reconciler$validate_hierarchy(forecasts, hierarchy),
      "same length"
    )
  })

  it("apply_reconciliation() computes correctly", {
    reconciler <- BaseReconciler$new()
    base <- matrix(1:6, nrow = 2, ncol = 3)
    S <- matrix(c(1, 1, 1, 0, 0, 1, 0, 0, 1), nrow = 3, ncol = 3, byrow = TRUE)
    G <- diag(3)

    result <- reconciler$apply_reconciliation(base, S, G)

    expect_equal(dim(result), c(2, 3))
  })
})
```

### HierarchyBuilder Tests

```r
# tests/testthat/test-reconciliation-hierarchy.R

describe("HierarchyBuilder", {
  it("initializes with default SIN config", {
    hierarchy <- HierarchyBuilder$new()

    expect_s3_class(hierarchy, "HierarchyBuilder")
    expect_equal(hierarchy$get_config()$national, "SIN")
  })

  it("initializes with custom config", {
    hierarchy <- create_test_hierarchy(custom = TRUE)

    expect_equal(hierarchy$get_config()$national, "TOTAL")
  })

  it("build_summing_matrix() has correct dimensions", {
    hierarchy <- HierarchyBuilder$new()
    S <- hierarchy$build_summing_matrix()

    all_series <- hierarchy$get_all_series()
    bottom_series <- hierarchy$get_bottom_series()

    expect_equal(nrow(S), length(all_series))
    expect_equal(ncol(S), length(bottom_series))
  })

  it("build_summing_matrix() has correct row/col names", {
    hierarchy <- HierarchyBuilder$new()
    S <- hierarchy$build_summing_matrix()

    expect_equal(rownames(S), hierarchy$get_all_series())
    expect_equal(colnames(S), hierarchy$get_bottom_series())
  })

  it("get_all_series() returns all 31 series", {
    hierarchy <- HierarchyBuilder$new()
    all_series <- hierarchy$get_all_series()

    expect_equal(length(all_series), 31)
    expect_true("SIN" %in% all_series)
    expect_true("SECO" %in% all_series)
    expect_true("RJ" %in% all_series)
    expect_true("PESE" %in% all_series)
  })

  it("get_bottom_series() returns 27 series", {
    hierarchy <- HierarchyBuilder$new()
    bottom <- hierarchy$get_bottom_series()

    expect_equal(length(bottom), 27)
    expect_false("SIN" %in% bottom)
    expect_false("SECO" %in% bottom)
  })

  it("get_children() returns correct children", {
    hierarchy <- HierarchyBuilder$new()

    expect_equal(
      sort(hierarchy$get_children("SIN")),
      sort(c("SECO", "S", "NE", "N"))
    )

    seco_children <- hierarchy$get_children("SECO")
    expect_true("RJ" %in% seco_children)
    expect_true("PESE" %in% seco_children)
  })

  it("get_children() returns empty for bottom level", {
    hierarchy <- HierarchyBuilder$new()

    expect_length(hierarchy$get_children("RJ"), 0)
    expect_length(hierarchy$get_children("PESE"), 0)
  })

  it("get_parent() returns correct parent", {
    hierarchy <- HierarchyBuilder$new()

    expect_null(hierarchy$get_parent("SIN"))
    expect_equal(hierarchy$get_parent("SECO"), "SIN")
    expect_equal(hierarchy$get_parent("RJ"), "SECO")
    expect_equal(hierarchy$get_parent("PESE"), "SECO")
  })

  it("validate() passes for valid config", {
    hierarchy <- HierarchyBuilder$new()
    expect_true(hierarchy$validate())
  })

  it("validate() errors on duplicate areas", {
    bad_config <- list(
      national = "SIN",
      subsystems = list(
        A = c("X", "Y"),
        B = c("Y", "Z")  # Y is duplicate
      ),
      loss_areas = list()
    )

    expect_error(
      HierarchyBuilder$new(config = bad_config)$validate(),
      "Duplicate"
    )
  })

  it("get_aggregation_levels() returns all levels", {
    hierarchy <- HierarchyBuilder$new()
    levels <- hierarchy$get_aggregation_levels()

    expect_equal(levels$national, "SIN")
    expect_length(levels$subsystems, 4)
    expect_length(levels$areas, 23)
    expect_length(levels$loss_areas, 4)
  })
})
```

### LossCalculator Tests

```r
# tests/testthat/test-reconciliation-loss.R

describe("LossCalculator", {
  it("initializes with defaults", {
    calculator <- LossCalculator$new()
    expect_s3_class(calculator, "LossCalculator")
  })

  it("calculate_by_difference() computes correctly", {
    calculator <- LossCalculator$new()
    hierarchy <- HierarchyBuilder$new()

    # Create simple forecasts
    forecasts <- list(
      SECO = rep(1000, 10),
      RJ = rep(100, 10),
      SP = rep(200, 10),
      MG = rep(150, 10),
      ES = rep(80, 10),
      MT = rep(70, 10),
      MS = rep(60, 10),
      AC = rep(40, 10),
      RO = rep(50, 10),
      DF = rep(30, 10),
      GO = rep(20, 10)
    )
    # Sum = 800, so loss should be 200

    loss <- calculator$calculate_by_difference("SECO", forecasts, hierarchy)
    expect_equal(loss, rep(200, 10))
  })

  it("calculate_loss() by_difference", {
    calculator <- LossCalculator$new()
    hierarchy <- HierarchyBuilder$new()
    forecasts <- create_test_forecasts(n_time = 10, coherent = TRUE)

    loss <- calculator$calculate_loss(
      subsystem = "SECO",
      loss_area_code = "PESE",
      forecasts = forecasts,
      hierarchy = hierarchy
    )

    expect_length(loss, 10)
    expect_type(loss, "double")
  })

  it("calculate_loss() model method requires forecast", {
    calculator <- LossCalculator$new()
    hierarchy <- HierarchyBuilder$new()
    forecasts <- create_test_forecasts()
    forecasts$PENE <- NULL  # Remove model-based loss

    expect_error(
      calculator$calculate_loss("NE", "PENE", forecasts, hierarchy),
      "not in forecasts"
    )
  })

  it("warns on negative loss", {
    calculator <- LossCalculator$new(config = list(clamp_negative = FALSE))
    hierarchy <- HierarchyBuilder$new()

    # Create forecasts where areas > subsystem (negative loss)
    forecasts <- create_test_forecasts(n_time = 10)
    forecasts$SECO <- rep(100, 10)  # Very low
    forecasts$RJ <- rep(1000, 10)   # Very high

    expect_warning(
      calculator$calculate_by_difference("SECO", forecasts, hierarchy),
      "negative"
    )
  })

  it("clamps negative loss when configured", {
    calculator <- LossCalculator$new(config = list(clamp_negative = TRUE))
    hierarchy <- HierarchyBuilder$new()

    forecasts <- create_test_forecasts(n_time = 10)
    forecasts$SECO <- rep(100, 10)
    forecasts$RJ <- rep(1000, 10)

    loss <- suppressWarnings(
      calculator$calculate_by_difference("SECO", forecasts, hierarchy)
    )

    expect_true(all(loss >= 0))
  })
})
```

### CoherenceValidator Tests

```r
# tests/testthat/test-reconciliation-coherence.R

describe("CoherenceValidator", {
  it("initializes correctly", {
    hierarchy <- HierarchyBuilder$new()
    validator <- CoherenceValidator$new(hierarchy)

    expect_s3_class(validator, "CoherenceValidator")
  })

  it("validate() passes for coherent forecasts", {
    hierarchy <- HierarchyBuilder$new()
    validator <- CoherenceValidator$new(hierarchy)
    forecasts <- create_test_forecasts(coherent = TRUE)

    results <- validator$validate(forecasts)

    expect_true(all(results$is_coherent))
  })

  it("validate() fails for incoherent forecasts", {
    hierarchy <- HierarchyBuilder$new()
    validator <- CoherenceValidator$new(hierarchy, strict = FALSE)
    forecasts <- create_test_forecasts(coherent = FALSE)

    expect_warning(
      results <- validator$validate(forecasts),
      "violations"
    )

    expect_false(all(results$is_coherent))
  })

  it("validate() throws error in strict mode", {
    hierarchy <- HierarchyBuilder$new()
    validator <- CoherenceValidator$new(hierarchy, strict = TRUE)
    forecasts <- create_test_forecasts(coherent = FALSE)

    expect_error(
      validator$validate(forecasts),
      "violations"
    )
  })

  it("is_coherent() returns correct boolean", {
    hierarchy <- HierarchyBuilder$new()
    validator <- CoherenceValidator$new(hierarchy)

    coherent <- create_test_forecasts(coherent = TRUE)
    incoherent <- create_test_forecasts(coherent = FALSE)

    expect_true(validator$is_coherent(coherent))
    expect_false(suppressWarnings(validator$is_coherent(incoherent)))
  })

  it("respects tolerance", {
    hierarchy <- HierarchyBuilder$new()
    forecasts <- create_test_forecasts(coherent = TRUE)

    # Add tiny error
    forecasts$SIN <- forecasts$SIN + 1e-8

    # Should pass with default tolerance
    validator1 <- CoherenceValidator$new(hierarchy, tolerance = 1e-6)
    expect_true(validator1$is_coherent(forecasts))

    # Should fail with strict tolerance
    validator2 <- CoherenceValidator$new(hierarchy, tolerance = 1e-10)
    expect_false(suppressWarnings(validator2$is_coherent(forecasts)))
  })
})
```

### Integration Tests

```r
# tests/testthat/test-reconciliation-integration.R

describe("ReconciliationWorkflow Integration", {
  it("completes full workflow", {
    # Create workflow
    workflow <- ReconciliationWorkflow$new(
      reconciler = "bottom_up",
      hierarchy_config = NULL,
      config = list(
        calculate_losses = TRUE,
        validate_coherence = TRUE
      )
    )

    # Create forecasts without loss areas
    forecasts <- create_test_forecasts(coherent = TRUE)
    forecasts$PESE <- NULL
    forecasts$PES <- NULL
    forecasts$PENE <- NULL  # This one uses model, keep it
    forecasts$PEN <- NULL

    # Need to add PENE back since it's model-based
    forecasts$PENE <- rnorm(168, mean = 15, sd = 1.5)

    # Run workflow
    result <- workflow$run(forecasts)

    expect_s3_class(result, "ReconciliationResult")
    expect_equal(length(result$predictions), 31)
  })

  it("produces coherent output", {
    workflow <- ReconciliationWorkflow$new(
      reconciler = "bottom_up",
      config = list(validate_coherence = TRUE)
    )

    forecasts <- create_test_forecasts(coherent = FALSE)
    forecasts$PENE <- rnorm(168, mean = 15, sd = 1.5)

    result <- workflow$run(forecasts)
    hierarchy <- workflow$get_hierarchy_builder()

    # Check coherence
    expect_true(is_hierarchically_coherent(result$predictions, hierarchy))
  })

  it("reconcile_forecasts() convenience works", {
    forecasts <- create_test_forecasts(coherent = TRUE)
    forecasts$PENE <- rnorm(168, mean = 15, sd = 1.5)

    result <- reconcile_forecasts(forecasts, method = "bottom_up")

    expect_s3_class(result, "ReconciliationResult")
  })
})
```

---

## Test Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| base.R | 90% |
| registry.R | 90% |
| hierarchy.R | 90% |
| workflow.R | 85% |
| loss_calculation.R | 85% |
| coherence.R | 90% |
| **Overall** | **≥80%** |

---

## Dependencies

- PC-039-06: BaseReconciler
- PC-040-06: ReconcilerRegistry
- PC-041-06: HierarchyBuilder
- PC-042-06: ReconciliationWorkflow
- PC-043-06: LossCalculator
- PC-045-06: CoherenceValidator

---

## Definition of Done

- [ ] All test files created
- [ ] Test fixtures and helpers implemented
- [ ] BaseReconciler tests passing
- [ ] ReconcilerRegistry tests passing
- [ ] HierarchyBuilder tests passing
- [ ] ReconciliationWorkflow tests passing
- [ ] LossCalculator tests passing
- [ ] CoherenceValidator tests passing
- [ ] Integration tests passing
- [ ] Coverage ≥80%
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use describe/it style (testthat 3.0+)
- Test fixtures create reproducible test data
- Integration tests cover end-to-end workflows
- Consider adding property-based tests for matrix operations
- May add performance benchmarks for large hierarchies
