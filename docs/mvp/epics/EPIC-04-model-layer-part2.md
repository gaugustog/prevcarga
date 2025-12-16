# EPIC-04: Model Layer - Hierarchical Support

**Duration:** 3 weeks
**Dependencies:** EPIC-03
**Reference:** [MVP Plan R - Phase 4](../mvp-plan-r.md#phase-4-model-layer---part-2-3-weeks)

---

## Objective

Extend the model infrastructure to support hierarchical forecasting patterns (DM→Profile), multi-model architectures, and advanced training workflows. **This epic creates infrastructure support**, not model implementations.

---

## Scope

This epic covers:
- `HierarchicalModel` base class extension
- Multi-model manager pattern
- Profile model infrastructure (48 models pattern)
- Parallel training support with `future`
- Inference workflow infrastructure

**Out of Scope:**
- Actual hierarchical model implementations (RegDin+SVM, HW, etc.)
- Specific ARIMA/SVM/HW algorithms
- These are contributed separately via the plugin system

---

## Tasks

### T-04.1: Create HierarchicalModel Base Class
- [ ] Create `R/models/hierarchical.R`
- [ ] Implement `HierarchicalModel` R6 class:
  ```r
  HierarchicalModel <- R6::R6Class(
    "HierarchicalModel",
    inherit = BaseModel,
    public = list(
      dm_model = NULL,      # Demand mean model
      profile_models = NULL, # Profile models (list)

      train_dm = function(X, y, ...) {
        stop("train_dm() must be implemented by subclass")
      },

      train_profiles = function(dm_residuals, ...) {
        stop("train_profiles() must be implemented by subclass")
      },

      predict_dm = function(X, ...) {
        stop("predict_dm() must be implemented by subclass")
      },

      predict_profiles = function(dm_forecast, ...) {
        stop("predict_profiles() must be implemented by subclass")
      },

      compose = function(dm_forecast, profile_forecast) {
        # Combine DM and profile forecasts
        dm_forecast * profile_forecast
      }
    )
  )
  ```

### T-04.2: Create MultiModelManager Pattern
- [ ] Create `R/models/multi_model.R`
- [ ] Implement `MultiModelManager` R6 class:
  ```r
  MultiModelManager <- R6::R6Class(
    "MultiModelManager",
    inherit = BaseModel,
    private = list(
      models = list(),           # {target: model}
      feature_selections = list() # {target: features}
    ),
    public = list(
      train_multi = function(X, y, target_cols, ...) {
        stop("train_multi() must be implemented by subclass")
      },

      predict_multi = function(X, target_cols = NULL, ...) {
        stop("predict_multi() must be implemented by subclass")
      },

      get_model = function(target) {
        private$models[[target]]
      },

      list_targets = function() {
        names(private$models)
      }
    )
  )
  ```

### T-04.3: Create Parallel Training Infrastructure
- [ ] Create `R/models/parallel.R`
- [ ] Implement `ParallelTrainer` R6 class:
  ```r
  ParallelTrainer <- R6::R6Class(
    "ParallelTrainer",
    private = list(
      n_workers = NULL
    ),
    public = list(
      initialize = function(n_workers = 4) {
        private$n_workers <- n_workers
      },

      train_parallel = function(tasks, ...) {
        # tasks: list of {model, X, y} objects
        future::plan(future::multisession, workers = private$n_workers)

        results <- future.apply::future_lapply(tasks, function(task) {
          tryCatch({
            task$model$train(task$X, task$y, ...)
            list(status = "success", model = task$model)
          }, error = function(e) {
            list(status = "error", message = conditionMessage(e))
          })
        }, future.seed = TRUE)

        results
      },

      shutdown = function() {
        future::plan(future::sequential)
      }
    )
  )
  ```

### T-04.4: Create Inference Workflow Infrastructure
- [ ] Create `R/models/inference.R`
- [ ] Implement `InferenceWorkflow` R6 class structure:
  ```r
  InferenceWorkflow <- R6::R6Class(
    "InferenceWorkflow",
    private = list(
      model_artifact = NULL,
      feature_builder = NULL
    ),
    public = list(
      initialize = function(model_artifact, feature_builder) {
        private$model_artifact <- model_artifact
        private$feature_builder <- feature_builder
      },

      run = function(anchor_data, target_date, ...) {
        # 1. Build inference features
        features <- private$feature_builder$build_features(
          anchor_data, target_date = target_date
        )

        # 2. Apply normalization
        features <- self$apply_normalization(features)

        # 3. Predict
        predictions <- private$model_artifact$model$predict(features)

        # 4. Denormalize
        predictions <- self$denormalize(predictions)

        predictions
      },

      apply_normalization = function(features) { ... },
      denormalize = function(predictions) { ... }
    )
  )
  ```

### T-04.5: Create Two-Stage Inference Support
- [ ] Add `TwoStageInferenceWorkflow` for D+0 → D+1 pattern
- [ ] Support ratio propagation for day completion
- [ ] Support verified vs forecasted period tracking
- [ ] See [Plugin Guide: Model Inference](../plugin-guide/04-model-inference.md) for patterns

### T-04.6: Create Batch Inference Runner
- [ ] Create `R/models/batch_inference.R`
- [ ] Implement `BatchInferenceRunner` R6 class:
  ```r
  BatchInferenceRunner <- R6::R6Class(
    "BatchInferenceRunner",
    private = list(
      model_registry = NULL,
      storage = NULL
    ),
    public = list(
      run_batch = function(areas, target_date, config, ...) {
        # Parallel execution across areas
      },

      aggregate_results = function(results) {
        # Combine results, handle failures
      }
    )
  )
  ```

### T-04.7: Create Dynamic Feature Selection Support
- [ ] Add infrastructure for per-horizon feature selection
- [ ] Support for storing/loading feature selections per target
- [ ] Pattern for horizon-aware models (216/432 models)

### T-04.8: Write Tests
- [ ] Create `tests/testthat/test-models-hierarchical.R`
- [ ] Test HierarchicalModel contract
- [ ] Test MultiModelManager operations
- [ ] Test ParallelTrainer with mock models
- [ ] Test InferenceWorkflow

---

## Acceptance Criteria

- [ ] `HierarchicalModel` supports DM→Profile pattern
- [ ] `MultiModelManager` handles N models per area/horizon
- [ ] `ParallelTrainer` distributes training across workers
- [ ] `InferenceWorkflow` applies normalization correctly
- [ ] Two-stage inference pattern is supported
- [ ] Batch inference runs for multiple areas
- [ ] Tests pass with ≥80% coverage

---

## Definition of Done

- [ ] All tasks completed
- [ ] Code reviewed and merged
- [ ] Tests passing (≥80% coverage)
- [ ] Documentation complete with roxygen2
- [ ] Integration with EPIC-03 verified

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `R/models/hierarchical.R` | Create | HierarchicalModel base class |
| `R/models/multi_model.R` | Create | MultiModelManager pattern |
| `R/models/parallel.R` | Create | ParallelTrainer |
| `R/models/inference.R` | Create | InferenceWorkflow |
| `R/models/batch_inference.R` | Create | BatchInferenceRunner |
| `tests/testthat/test-models-hierarchical.R` | Create | Hierarchical model tests |

---

## Hierarchical Model Patterns Reference

### DM→Profile Pattern (48 models)
```
Demand Mean (DM) Model
    │
    └──► Profile Models (48 half-hour periods)
             │
             └──► Final Forecast = DM × Profile
```

### Multi-Model Pattern (216 models)
```
Target Columns: y_h001, y_h002, ..., y_h216
    │
    └──► One model per target (9 days × 24 hours)
             │
             └──► Dynamic feature selection per horizon
```

See [Plugin Guide: Model Training](../plugin-guide/03-model-training.md) for complexity tiers.
