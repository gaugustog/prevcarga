# Plugin Guide: Overview

> **Note:** This documentation will evolve into the project's CONTRIBUTING guide.

## Plugin Architecture

PrevCargaONS uses an R6-based plugin architecture that enables modular, extensible forecasting components. The system follows the **Registry Pattern** where plugins are registered at startup and created on-demand.

## Core Plugin Types

| Plugin Type | Base Class | Purpose |
|-------------|-----------|---------|
| **Feature Plugin** | `BaseFeaturePlugin` | Transform raw data into features |
| **Model Plugin** | `BaseModel` | Train and predict load forecasts |
| **Combiner Plugin** | `BaseCombiner` | Combine multiple model predictions |
| **Reconciler Plugin** | `BaseReconciler` | Hierarchical reconciliation |

## Plugin Lifecycle

```
1. REGISTER    → Plugin class registered with registry at package load
2. CONFIGURE   → Instance created with configuration from YAML
3. INITIALIZE  → Setup internal state (connections, parameters)
4. EXECUTE     → transform() / train() / predict() / combine()
5. SERIALIZE   → Save state for later use (models, normalization params)
```

## Registry Pattern

Each plugin type has a global registry that manages plugin discovery and instantiation:

```r
# Feature Plugin Registry
FeaturePluginRegistry <- R6::R6Class(
  "FeaturePluginRegistry",
  private = list(
    plugins = list()
  ),
  public = list(
    register = function(name, plugin_class) {
      private$plugins[[name]] <- plugin_class
      invisible(self)
    },

    create = function(name, config = list()) {
      plugin_class <- private$plugins[[name]]
      if (is.null(plugin_class)) {
        stop(sprintf("Plugin '%s' not found", name))
      }
      plugin_class$new(config = config)
    },

    list_plugins = function() {
      names(private$plugins)
    }
  )
)

# Global registry instance
.feature_registry <- FeaturePluginRegistry$new()
```

## R6 Class Inheritance

All plugins inherit from a base class that defines the interface:

```r
BasePlugin <- R6::R6Class(
  "BasePlugin",
  public = list(
    name = NULL,
    config = NULL,

    initialize = function(name = NULL, config = list()) {
      self$name <- name
      self$config <- config
    }
  )
)
```

## Configuration Integration

Plugins receive configuration from YAML files:

```yaml
# config.yaml
features:
  plugins:
    - name: lag_features
      config:
        lags: [1, 7, 14, 28]
        columns: [CargaGlobal]

    - name: calendar_features
      config:
        include_holidays: true
        include_dst: true

models:
  plugins:
    - name: lightgbm
      config:
        num_leaves: 31
        learning_rate: 0.05
        num_iterations: 1000
```

## Creating a New Plugin

### Step 1: Define the Plugin Class

```r
MyFeaturePlugin <- R6::R6Class(
  "MyFeaturePlugin",
  inherit = BaseFeaturePlugin,

  public = list(
    transform = function(dt, ...) {
      # Implementation here
      dt
    },

    get_feature_names = function() {
      # Return vector of feature names created
      c("my_feature_1", "my_feature_2")
    }
  )
)
```

### Step 2: Register the Plugin

```r
# In your plugin file (e.g., R/features/plugins/my_plugin.R)
.feature_registry$register("my_feature_plugin", MyFeaturePlugin)
```

### Step 3: Configure in YAML

```yaml
features:
  plugins:
    - name: my_feature_plugin
      config:
        param1: value1
```

## Directory Structure

```
R/
├── features/
│   ├── base.R              # BaseFeaturePlugin
│   ├── registry.R          # FeaturePluginRegistry
│   └── plugins/            # Plugin implementations
├── models/
│   ├── base.R              # BaseModel
│   ├── registry.R          # ModelRegistry
│   └── implementations/    # Model implementations
├── combination/
│   ├── base.R              # BaseCombiner
│   └── implementations/    # Combiner implementations
└── reconciliation/
    ├── base.R              # BaseReconciler
    └── implementations/    # Reconciler implementations
```

## Next Steps

- [Data Importing](01-data-importing.md) - Storage backends and data loading
- [Feature Engineering](02-feature-engineering.md) - Creating feature plugins
- [Model Training](03-model-training.md) - Creating model plugins
- [Model Inference](04-model-inference.md) - Inference and prediction
- [Combination Strategies](05-combination.md) - Combining multiple model forecasts
- [Reconciliation](06-reconciliation.md) - Hierarchical reconciliation for SIN
