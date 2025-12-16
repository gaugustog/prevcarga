# EPIC-02: Feature Engineering Infrastructure

**Duration:** 3 weeks
**Dependencies:** EPIC-01
**Reference:** [MVP Plan R - Phase 2](../mvp-plan-r.md#phase-2-feature-engineering-3-weeks)

---

## Objective

Create the plugin infrastructure for feature engineering, including the base class, registry pattern, and pipeline composer. **This epic does NOT implement feature plugins** - it creates the structure to receive contributed plugins.

---

## Scope

This epic covers:
- `BaseFeaturePlugin` R6 abstract class
- `FeaturePluginRegistry` for plugin management
- `FeaturePipeline` for composing multiple plugins
- Feature evaluation framework structure
- Plugin configuration system

**Out of Scope:**
- Actual feature plugin implementations (temporal, calendar, lag, wavelet, etc.)
- Model-specific feature transformations
- These are contributed separately via the plugin system

---

## Tasks

### T-02.1: Create BaseFeaturePlugin Abstract Class
- [ ] Create `R/features/base.R`
- [ ] Implement `BaseFeaturePlugin` R6 class:
  ```r
  BaseFeaturePlugin <- R6::R6Class(
    "BaseFeaturePlugin",
    public = list(
      name = NULL,
      config = NULL,

      initialize = function(name = NULL, config = list()) {
        self$name <- name
        self$config <- config
      },

      transform = function(dt, ...) {
        stop("transform() must be implemented by subclass")
      },

      get_feature_names = function() {
        stop("get_feature_names() must be implemented by subclass")
      },

      validate_input = function(dt) {
        # Default validation - override in subclass
        checkmate::assert_data_table(dt)
        invisible(TRUE)
      }
    )
  )
  ```

### T-02.2: Create FeaturePluginRegistry
- [ ] Create `R/features/registry.R`
- [ ] Implement `FeaturePluginRegistry` R6 class:
  ```r
  FeaturePluginRegistry <- R6::R6Class(
    "FeaturePluginRegistry",
    private = list(
      plugins = list()
    ),
    public = list(
      register = function(name, plugin_class) {
        stopifnot(inherits(plugin_class$new(), "BaseFeaturePlugin"))
        private$plugins[[name]] <- plugin_class
        invisible(self)
      },

      get = function(name, config = list()) {
        if (!name %in% names(private$plugins)) {
          stop(sprintf("Plugin '%s' not registered", name))
        }
        private$plugins[[name]]$new(name = name, config = config)
      },

      list_plugins = function() {
        names(private$plugins)
      },

      has = function(name) {
        name %in% names(private$plugins)
      }
    )
  )
  ```
- [ ] Create global registry instance `.feature_registry`
- [ ] Export registration function for plugin contributors

### T-02.3: Create FeaturePipeline
- [ ] Create `R/features/pipeline.R`
- [ ] Implement `FeaturePipeline` R6 class:
  ```r
  FeaturePipeline <- R6::R6Class(
    "FeaturePipeline",
    private = list(
      plugins = list(),
      execution_order = NULL
    ),
    public = list(
      add = function(plugin, position = NULL) {
        # Add plugin to pipeline
        invisible(self)
      },

      remove = function(name) {
        # Remove plugin by name
        invisible(self)
      },

      transform = function(dt, ...) {
        # Execute all plugins in order
        for (plugin in private$plugins) {
          dt <- plugin$transform(dt, ...)
        }
        dt
      },

      get_all_feature_names = function() {
        unlist(lapply(private$plugins, function(p) p$get_feature_names()))
      },

      from_config = function(config, registry) {
        # Build pipeline from YAML config
        invisible(self)
      }
    )
  )
  ```

### T-02.4: Create Feature Configuration Schema
- [ ] Create `R/features/config.R`
- [ ] Define YAML schema for feature configuration:
  ```yaml
  features:
    plugins:
      - name: plugin_name
        config:
          param1: value1
          param2: value2
  ```
- [ ] Implement config validation
- [ ] Support plugin ordering

### T-02.5: Create Feature Evaluation Framework Structure
- [ ] Create `R/features/evaluation.R`
- [ ] Implement `FeatureEvaluator` R6 class structure:
  ```r
  FeatureEvaluator <- R6::R6Class(
    "FeatureEvaluator",
    public = list(
      evaluate_importance = function(features, target, method = "correlation") {
        # Placeholder for importance calculation
        stop("evaluate_importance() not implemented")
      },

      evaluate_stability = function(features, folds) {
        # Placeholder for stability analysis
        stop("evaluate_stability() not implemented")
      },

      get_report = function() {
        # Generate evaluation report
        stop("get_report() not implemented")
      }
    )
  )
  ```
- [ ] Define interface for SHAP, correlation, stability methods

### T-02.6: Create Common Feature Utilities
- [ ] Create `R/features/common.R`
- [ ] Utility functions for plugin developers:
  - [ ] `cyclical_encode(values, period)` - sin/cos encoding
  - [ ] `create_lags(dt, column, lags)` - lag feature helper
  - [ ] `create_rolling_stats(dt, column, windows)` - rolling statistics
  - [ ] `one_hot_encode(dt, column)` - dummy variables
- [ ] These are helpers, not full plugins

### T-02.7: Write Tests
- [ ] Create `tests/testthat/test-features-base.R`
- [ ] Test BaseFeaturePlugin contract enforcement
- [ ] Test FeaturePluginRegistry operations
- [ ] Test FeaturePipeline composition
- [ ] Test configuration parsing
- [ ] Create mock plugin for testing

---

## Acceptance Criteria

- [ ] `BaseFeaturePlugin` enforces `transform()` and `get_feature_names()` implementation
- [ ] `FeaturePluginRegistry` registers and creates plugin instances
- [ ] `FeaturePipeline` chains multiple plugins correctly
- [ ] Pipeline can be built from YAML configuration
- [ ] Feature evaluation structure is defined (not implemented)
- [ ] Common utilities available for plugin developers
- [ ] Tests pass with ≥80% coverage

---

## Definition of Done

- [ ] All tasks completed
- [ ] Code reviewed and merged
- [ ] Tests passing (≥80% coverage)
- [ ] Documentation complete with roxygen2
- [ ] Example plugin template documented in plugin guide

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `R/features/base.R` | Create | BaseFeaturePlugin abstract class |
| `R/features/registry.R` | Create | FeaturePluginRegistry |
| `R/features/pipeline.R` | Create | FeaturePipeline composer |
| `R/features/config.R` | Create | Configuration schema |
| `R/features/evaluation.R` | Create | FeatureEvaluator structure |
| `R/features/common.R` | Create | Utility functions |
| `tests/testthat/test-features-base.R` | Create | Feature infrastructure tests |

---

## Plugin Development Reference

Contributors can implement feature plugins by:

1. **Inheriting from BaseFeaturePlugin:**
   ```r
   MyFeaturePlugin <- R6::R6Class(
     "MyFeaturePlugin",
     inherit = BaseFeaturePlugin,
     public = list(
       transform = function(dt, ...) {
         # Implementation
         dt
       },
       get_feature_names = function() {
         c("my_feature_1", "my_feature_2")
       }
     )
   )
   ```

2. **Registering the plugin:**
   ```r
   .feature_registry$register("my_plugin", MyFeaturePlugin)
   ```

3. **Configuring in YAML:**
   ```yaml
   features:
     plugins:
       - name: my_plugin
         config:
           param: value
   ```

See [Plugin Guide: Feature Engineering](../plugin-guide/02-feature-engineering.md) for detailed documentation.
