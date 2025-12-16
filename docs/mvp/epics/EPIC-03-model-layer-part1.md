# EPIC-03: Model Layer - Infrastructure

**Duration:** 4 weeks
**Dependencies:** EPIC-02
**Reference:** [MVP Plan R - Phase 3](../mvp-plan-r.md#phase-3-model-layer---part-1-4-weeks)

---

## Objective

Create the model plugin infrastructure including base class, registry, versioning system, and trainer abstraction. **This epic does NOT implement model plugins** - it creates the structure to receive contributed models.

---

## Scope

This epic covers:
- `BaseModel` R6 abstract class
- `ModelRegistry` for plugin management
- Semantic versioning system
- Model artifact serialization/deserialization
- `UniversalTrainer` workflow structure

**Out of Scope:**
- Actual model implementations (LightGBM, Random Forest, Holt-Winters, etc.)
- Custom loss functions (plugin responsibility)
- These are contributed separately via the plugin system

---

## Tasks

### T-03.1: Create BaseModel Abstract Class
- [ ] Create `R/models/base.R`
- [ ] Implement `BaseModel` R6 class:
  ```r
  BaseModel <- R6::R6Class(
    "BaseModel",
    public = list(
      name = NULL,
      horizons = NULL,
      config = NULL,
      is_trained = FALSE,

      initialize = function(name = NULL, horizons = 0:8, config = list()) {
        self$name <- name
        self$horizons <- horizons
        self$config <- config
      },

      train = function(X, y, ...) {
        stop("train() must be implemented by subclass")
      },

      predict = function(X, ...) {
        stop("predict() must be implemented by subclass")
      },

      save = function(path, version) {
        stop("save() must be implemented by subclass")
      },

      load = function(path) {
        stop("load() must be implemented by subclass")
      },

      get_feature_importance = function() {
        # Optional: return NULL if not supported
        NULL
      }
    )
  )
  ```

### T-03.2: Create ModelRegistry
- [ ] Create `R/models/registry.R`
- [ ] Implement `ModelRegistry` R6 class:
  ```r
  ModelRegistry <- R6::R6Class(
    "ModelRegistry",
    private = list(
      models = list()
    ),
    public = list(
      register = function(name, model_class) {
        stopifnot(inherits(model_class$new(), "BaseModel"))
        private$models[[name]] <- model_class
        invisible(self)
      },

      get = function(name, config = list()) {
        if (!name %in% names(private$models)) {
          stop(sprintf("Model '%s' not registered", name))
        }
        private$models[[name]]$new(name = name, config = config)
      },

      list_models = function() {
        names(private$models)
      },

      has = function(name) {
        name %in% names(private$models)
      }
    )
  )
  ```
- [ ] Create global registry instance `.model_registry`
- [ ] Export registration function for model contributors

### T-03.3: Implement Semantic Versioning
- [ ] Create `R/models/versioning.R`
- [ ] Implement `ModelVersion` R6 class:
  ```r
  ModelVersion <- R6::R6Class(
    "ModelVersion",
    public = list(
      major = 0,
      minor = 0,
      patch = 0,
      timestamp = NULL,

      initialize = function(version_string = NULL) { ... },

      to_string = function() {
        sprintf("v%d.%d.%d_%s", self$major, self$minor, self$patch,
                format(self$timestamp, "%Y%m%d_%H%M%S"))
      },

      bump_major = function() { ... },
      bump_minor = function() { ... },
      bump_patch = function() { ... },

      compare = function(other) { ... }
    )
  )
  ```
- [ ] Support version parsing from string
- [ ] Support version comparison

### T-03.4: Create Model Artifact Structure
- [ ] Create `R/models/artifact.R`
- [ ] Implement `ModelArtifact` R6 class:
  ```r
  ModelArtifact <- R6::R6Class(
    "ModelArtifact",
    public = list(
      model = NULL,
      metadata = NULL,
      normalization_params = NULL,
      feature_names = NULL,
      version = NULL,

      save = function(storage, path) {
        # Save model.rds
        storage$write_rds(self$model, file.path(path, "model.rds"))

        # Save metadata.yaml
        yaml::write_yaml(self$metadata, file.path(path, "metadata.yaml"))

        # Save normalization.rds if present
        if (!is.null(self$normalization_params)) {
          storage$write_rds(self$normalization_params,
                           file.path(path, "normalization.rds"))
        }

        # Save feature_names.json
        jsonlite::write_json(self$feature_names,
                            file.path(path, "feature_names.json"))
      },

      load = function(storage, path) { ... }
    )
  )
  ```

### T-03.5: Create UniversalTrainer Structure
- [ ] Create `R/models/trainer.R`
- [ ] Implement `UniversalTrainer` R6 class:
  ```r
  UniversalTrainer <- R6::R6Class(
    "UniversalTrainer",
    private = list(
      model = NULL,
      feature_pipeline = NULL,
      config = NULL
    ),
    public = list(
      initialize = function(model, feature_pipeline, config) {
        private$model <- model
        private$feature_pipeline <- feature_pipeline
        private$config <- config
      },

      train = function(data, ...) {
        # 1. Apply feature pipeline
        features <- private$feature_pipeline$transform(data)

        # 2. Split X, y
        # 3. Call model$train()
        # 4. Calculate validation metrics
        # 5. Create artifact
      },

      cross_validate = function(data, folds = 5, ...) { ... },

      get_metrics = function() { ... }
    )
  )
  ```

### T-03.6: Create Model Configuration Schema
- [ ] Create `R/models/config.R`
- [ ] Define YAML schema for model configuration:
  ```yaml
  models:
    default_hyperparameters:
      seed: 42
      validation_split: 0.15
      test_split: 0.15

    plugins:
      - name: model_name
        horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]
        config:
          param1: value1
  ```
- [ ] Implement config validation

### T-03.7: Create Model Storage Utilities
- [ ] Create `R/models/storage.R`
- [ ] Functions for model path management:
  ```r
  get_model_path = function(model_name, version, base_path = "models") {
    file.path(base_path, model_name, version)
  }

  get_latest_version = function(model_name, storage) {
    # Read symlink or find latest by timestamp
  }

  create_latest_symlink = function(model_name, version, storage) {
    # Create/update 'latest' symlink
  }
  ```

### T-03.8: Write Tests
- [ ] Create `tests/testthat/test-models-base.R`
- [ ] Test BaseModel contract enforcement
- [ ] Test ModelRegistry operations
- [ ] Test semantic versioning
- [ ] Test ModelArtifact save/load
- [ ] Create mock model for testing

---

## Acceptance Criteria

- [ ] `BaseModel` enforces `train()`, `predict()`, `save()`, `load()` implementation
- [ ] `ModelRegistry` registers and creates model instances
- [ ] Semantic versioning works with timestamps
- [ ] `ModelArtifact` saves/loads all model components
- [ ] `UniversalTrainer` structure is defined
- [ ] Configuration schema supports all hyperparameters
- [ ] Tests pass with ≥80% coverage

---

## Definition of Done

- [ ] All tasks completed
- [ ] Code reviewed and merged
- [ ] Tests passing (≥80% coverage)
- [ ] Documentation complete with roxygen2
- [ ] Example model template documented in plugin guide

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `R/models/base.R` | Create | BaseModel abstract class |
| `R/models/registry.R` | Create | ModelRegistry |
| `R/models/versioning.R` | Create | Semantic versioning |
| `R/models/artifact.R` | Create | ModelArtifact |
| `R/models/trainer.R` | Create | UniversalTrainer |
| `R/models/config.R` | Create | Configuration schema |
| `R/models/storage.R` | Create | Model storage utilities |
| `tests/testthat/test-models-base.R` | Create | Model infrastructure tests |

---

## Model Plugin Development Reference

Contributors can implement model plugins by:

1. **Inheriting from BaseModel:**
   ```r
   MyModel <- R6::R6Class(
     "MyModel",
     inherit = BaseModel,
     private = list(fitted_model = NULL),
     public = list(
       train = function(X, y, ...) {
         # Training implementation
         self$is_trained <- TRUE
         invisible(self)
       },
       predict = function(X, ...) {
         # Prediction implementation
       },
       save = function(path, version) {
         # Save implementation
       },
       load = function(path) {
         # Load implementation
         invisible(self)
       }
     )
   )
   ```

2. **Registering the model:**
   ```r
   .model_registry$register("my_model", MyModel)
   ```

See [Plugin Guide: Model Training](../plugin-guide/03-model-training.md) for detailed documentation.
