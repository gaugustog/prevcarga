# EPIC-06: Reconciliation Infrastructure

**Duration:** 2 weeks
**Dependencies:** EPIC-04
**Reference:** [MVP Plan R - Phase 6](../mvp-plan-r.md#phase-6-reconciliation-layer-2-weeks)

---

## Objective

Create the hierarchical reconciliation infrastructure ensuring forecasts sum correctly across the SIN hierarchy. **This epic creates the infrastructure** - reconciler implementations are contributed via the plugin system.

---

## Scope

This epic covers:
- `BaseReconciler` R6 abstract class
- `ReconcilerRegistry` for strategy management
- `HierarchyBuilder` for summing matrix construction
- `ReconciliationWorkflow` for orchestrating reconciliation
- Loss area calculation (by difference)

**Out of Scope:**
- Actual reconciler implementations (MinT, OLS, WLS, etc.)
- Covariance estimation algorithms
- These are contributed separately via the plugin system

---

## Tasks

### T-06.1: Create BaseReconciler Abstract Class
- [ ] Create `R/reconciliation/base.R`
- [ ] Implement `BaseReconciler` R6 class:
  ```r
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
      },

      validate_hierarchy = function(forecasts, hierarchy) {
        # Verify all required series present
        checkmate::assert_list(forecasts)
        invisible(TRUE)
      }
    )
  )
  ```

### T-06.2: Create ReconcilerRegistry
- [ ] Create `R/reconciliation/registry.R`
- [ ] Implement `ReconcilerRegistry` R6 class:
  ```r
  ReconcilerRegistry <- R6::R6Class(
    "ReconcilerRegistry",
    private = list(
      reconcilers = list()
    ),
    public = list(
      register = function(name, reconciler_class) {
        stopifnot(inherits(reconciler_class$new(), "BaseReconciler"))
        private$reconcilers[[name]] <- reconciler_class
        invisible(self)
      },

      get = function(name, config = list()) {
        if (!name %in% names(private$reconcilers)) {
          stop(sprintf("Reconciler '%s' not registered", name))
        }
        private$reconcilers[[name]]$new(name = name, config = config)
      },

      list_reconcilers = function() {
        names(private$reconcilers)
      }
    )
  )
  ```
- [ ] Create global registry instance `.reconciler_registry`

### T-06.3: Create HierarchyBuilder
- [ ] Create `R/reconciliation/hierarchy.R`
- [ ] Implement `HierarchyBuilder` R6 class:
  ```r
  HierarchyBuilder <- R6::R6Class(
    "HierarchyBuilder",
    private = list(
      config = NULL
    ),
    public = list(
      initialize = function(config) {
        private$config <- config
      },

      build_summing_matrix = function() {
        # Build S matrix from hierarchy config
        # Rows: all series (SIN, subsystems, areas)
        # Cols: bottom-level series only
      },

      get_bottom_series = function() {
        # Return area codes including loss areas
      },

      get_aggregation_levels = function() {
        list(
          national = "SIN",
          subsystems = c("SECO", "S", "NE", "N"),
          areas = self$get_bottom_series()
        )
      },

      validate = function() {
        # Validate hierarchy structure
      }
    )
  )
  ```

### T-06.4: Create ReconciliationWorkflow
- [ ] Create `R/reconciliation/workflow.R`
- [ ] Implement `ReconciliationWorkflow` R6 class:
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
        # 1. Reconcile hierarchy
        reconciled <- private$reconciler$reconcile(
          forecasts,
          private$hierarchy_builder
        )

        # 2. Calculate losses by difference (if configured)
        reconciled <- self$calculate_losses(reconciled)

        # 3. Validate coherence
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

      calculate_losses = function(forecasts) { ... },
      validate_coherence = function(forecasts) { ... }
    )
  )
  ```

### T-06.5: Create Loss Calculator
- [ ] Create `R/reconciliation/loss_calculation.R`
- [ ] Implement `LossCalculator` R6 class:
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
  ```

### T-06.6: Create Hierarchy Configuration Schema
- [ ] Create `R/reconciliation/config.R`
- [ ] Define YAML schema:
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
        PENE: {subsystem: NE, method: model}
        PEN: {subsystem: N, method: by_difference}
  ```
- [ ] Implement config validation

### T-06.7: Create Coherence Validator
- [ ] Add validation functions:
  ```r
  validate_hierarchical_coherence = function(forecasts, hierarchy) {
    # Check: SIN = Σ(subsystems)
    # Check: each subsystem = Σ(areas) + loss
  }
  ```

### T-06.8: Write Tests
- [ ] Create `tests/testthat/test-reconciliation.R`
- [ ] Test BaseReconciler contract
- [ ] Test HierarchyBuilder summing matrix
- [ ] Test ReconciliationWorkflow
- [ ] Test LossCalculator
- [ ] Test coherence validation

---

## Acceptance Criteria

- [ ] `BaseReconciler` enforces `reconcile()` implementation
- [ ] `ReconcilerRegistry` registers and creates reconciler instances
- [ ] `HierarchyBuilder` constructs correct summing matrix
- [ ] `ReconciliationWorkflow` orchestrates reconciliation
- [ ] `LossCalculator` computes losses by difference correctly
- [ ] Coherence validation catches inconsistencies
- [ ] Configuration supports all hierarchy parameters
- [ ] Tests pass with ≥80% coverage

---

## Definition of Done

- [ ] All tasks completed
- [ ] Code reviewed and merged
- [ ] Tests passing (≥80% coverage)
- [ ] Documentation complete with roxygen2
- [ ] Example reconciler template documented in plugin guide

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `R/reconciliation/base.R` | Create | BaseReconciler abstract class |
| `R/reconciliation/registry.R` | Create | ReconcilerRegistry |
| `R/reconciliation/hierarchy.R` | Create | HierarchyBuilder |
| `R/reconciliation/workflow.R` | Create | ReconciliationWorkflow |
| `R/reconciliation/loss_calculation.R` | Create | LossCalculator |
| `R/reconciliation/config.R` | Create | Configuration schema |
| `tests/testthat/test-reconciliation.R` | Create | Reconciliation tests |

---

## Hierarchy Reference

```
SIN (National) - area_code=SIN, RECONCILIATION ONLY
├── SECO (Subsystem) - RECONCILIATION ONLY
│   ├── RJ, SP, MG, ES, MT, MS, AC, RO, DF, GO (Areas - models run)
│   └── PESE (Loss Area - model OR by_difference)
├── S (Subsystem) - RECONCILIATION ONLY
│   ├── PR, SC, RS (Areas - models run)
│   └── PES (Loss Area)
├── NE (Subsystem) - RECONCILIATION ONLY
│   ├── ALPE, PBRN, BASE, CE, PI, BAOE (Areas - models run)
│   └── PENE (Loss Area)
└── N (Subsystem) - RECONCILIATION ONLY
    ├── AM, PA, MA, TO, RR, AP (Areas - models run)
    └── PEN (Loss Area)
```

See [Plugin Guide: Reconciliation](../plugin-guide/06-reconciliation.md) for detailed documentation.
