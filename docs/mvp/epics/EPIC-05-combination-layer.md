# EPIC-05: Combination Infrastructure

**Duration:** 3 weeks
**Dependencies:** EPIC-04
**Reference:** [MVP Plan R - Phase 5](../mvp-plan-r.md#phase-5-combination-layer-3-weeks)

---

## Objective

Create the infrastructure for combining predictions from multiple models, including base class, registry, and combination workflow. **This epic does NOT implement combination strategies** - it creates the structure to receive contributed combiners.

---

## Scope

This epic covers:
- `BaseCombiner` R6 abstract class
- `CombinerRegistry` for strategy management
- `CombinationWorkflow` for orchestrating combination
- Bias correction infrastructure
- Weight optimization framework structure

**Out of Scope:**
- Actual combiner implementations (voting, stacking, Markov, etc.)
- Specific weight optimization algorithms
- These are contributed separately via the plugin system

---

## Tasks

### T-05.1: Create BaseCombiner Abstract Class
- [ ] Create `R/combination/base.R`
- [ ] Implement `BaseCombiner` R6 class:
  ```r
  BaseCombiner <- R6::R6Class(
    "BaseCombiner",
    public = list(
      name = NULL,
      config = NULL,

      initialize = function(name = NULL, config = list()) {
        self$name <- name
        self$config <- config
      },

      combine = function(forecasts, weights = NULL, ...) {
        stop("combine() must be implemented by subclass")
      },

      fit_weights = function(forecasts, actuals, ...) {
        stop("fit_weights() must be implemented by subclass")
      },

      get_weights = function() {
        stop("get_weights() must be implemented by subclass")
      },

      validate_forecasts = function(forecasts) {
        # Ensure all forecasts have same structure
        checkmate::assert_list(forecasts, min.len = 1)
        invisible(TRUE)
      }
    )
  )
  ```

### T-05.2: Create CombinerRegistry
- [ ] Create `R/combination/registry.R`
- [ ] Implement `CombinerRegistry` R6 class:
  ```r
  CombinerRegistry <- R6::R6Class(
    "CombinerRegistry",
    private = list(
      combiners = list()
    ),
    public = list(
      register = function(name, combiner_class) {
        stopifnot(inherits(combiner_class$new(), "BaseCombiner"))
        private$combiners[[name]] <- combiner_class
        invisible(self)
      },

      get = function(name, config = list()) {
        if (!name %in% names(private$combiners)) {
          stop(sprintf("Combiner '%s' not registered", name))
        }
        private$combiners[[name]]$new(name = name, config = config)
      },

      list_combiners = function() {
        names(private$combiners)
      },

      has = function(name) {
        name %in% names(private$combiners)
      }
    )
  )
  ```
- [ ] Create global registry instance `.combiner_registry`

### T-05.3: Create CombinationWorkflow
- [ ] Create `R/combination/workflow.R`
- [ ] Implement `CombinationWorkflow` R6 class:
  ```r
  CombinationWorkflow <- R6::R6Class(
    "CombinationWorkflow",
    private = list(
      combiner = NULL,
      model_forecasts = list()
    ),
    public = list(
      initialize = function(combiner, ...) {
        private$combiner <- combiner
      },

      run = function(model_results, validation_data = NULL, ...) {
        # Extract forecasts from model results
        forecasts <- lapply(model_results, function(r) r$predictions)

        # Fit weights if validation data provided
        if (!is.null(validation_data)) {
          private$combiner$fit_weights(forecasts, validation_data)
        }

        # Combine forecasts
        combined <- private$combiner$combine(forecasts)

        list(
          predictions = combined,
          weights = private$combiner$get_weights(),
          metadata = list(
            n_models = length(forecasts),
            combiner = class(private$combiner)[1]
          )
        )
      }
    )
  )
  ```

### T-05.4: Create Bias Correction Infrastructure
- [ ] Create `R/combination/bias_correction.R`
- [ ] Implement `BiasCorrector` R6 class structure:
  ```r
  BiasCorrector <- R6::R6Class(
    "BiasCorrector",
    private = list(
      correction_model = NULL
    ),
    public = list(
      fit = function(predictions, actuals, features = NULL) {
        stop("fit() must be implemented by subclass")
      },

      correct = function(predictions, features = NULL) {
        stop("correct() must be implemented by subclass")
      },

      get_correction_factors = function() {
        stop("get_correction_factors() must be implemented by subclass")
      }
    )
  )
  ```

### T-05.5: Create Weight Optimizer Framework
- [ ] Create `R/combination/optimizer.R`
- [ ] Implement `WeightOptimizer` R6 class structure:
  ```r
  WeightOptimizer <- R6::R6Class(
    "WeightOptimizer",
    public = list(
      optimize = function(forecasts, actuals, method = "inverse_mse", ...) {
        stop("optimize() must be implemented by subclass")
      },

      get_optimal_weights = function() {
        stop("get_optimal_weights() must be implemented by subclass")
      },

      validate_weights = function(weights) {
        # Ensure weights sum to 1 and are non-negative
        checkmate::assert_numeric(weights, lower = 0, upper = 1)
        if (abs(sum(weights) - 1) > 1e-6) {
          warning("Weights do not sum to 1, normalizing")
          weights <- weights / sum(weights)
        }
        weights
      }
    )
  )
  ```

### T-05.6: Create Combination Configuration Schema
- [ ] Create `R/combination/config.R`
- [ ] Define YAML schema:
  ```yaml
  combination:
    strategy: weighted_average
    models:
      - name: model_a
        weight: 0.6
      - name: model_b
        weight: 0.4
    fit_weights:
      enabled: true
      method: inverse_mse
      lookback_days: 30
  ```

### T-05.7: Write Tests
- [ ] Create `tests/testthat/test-combination.R`
- [ ] Test BaseCombiner contract enforcement
- [ ] Test CombinerRegistry operations
- [ ] Test CombinationWorkflow
- [ ] Test weight validation
- [ ] Create mock combiner for testing

---

## Acceptance Criteria

- [ ] `BaseCombiner` enforces `combine()`, `fit_weights()`, `get_weights()`
- [ ] `CombinerRegistry` registers and creates combiner instances
- [ ] `CombinationWorkflow` orchestrates combination correctly
- [ ] `BiasCorrector` structure defined for plugin extension
- [ ] `WeightOptimizer` structure defined for plugin extension
- [ ] Configuration schema supports all combination parameters
- [ ] Tests pass with ≥80% coverage

---

## Definition of Done

- [ ] All tasks completed
- [ ] Code reviewed and merged
- [ ] Tests passing (≥80% coverage)
- [ ] Documentation complete with roxygen2
- [ ] Example combiner template documented in plugin guide

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `R/combination/base.R` | Create | BaseCombiner abstract class |
| `R/combination/registry.R` | Create | CombinerRegistry |
| `R/combination/workflow.R` | Create | CombinationWorkflow |
| `R/combination/bias_correction.R` | Create | BiasCorrector structure |
| `R/combination/optimizer.R` | Create | WeightOptimizer structure |
| `R/combination/config.R` | Create | Configuration schema |
| `tests/testthat/test-combination.R` | Create | Combination tests |

---

## Combiner Plugin Development Reference

See [Plugin Guide: Combination Strategies](../plugin-guide/05-combination.md) for detailed documentation on implementing:
- Simple average combiner
- Weighted average combiner
- Optimal (OLS) combiner
- Horizon-specific combiner
- Model selection combiner
