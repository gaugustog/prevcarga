# PC-021-03: Model Configuration Schema

**Epic:** [EPIC-03: Model Layer - Infrastructure](../epics/EPIC-03-model-layer-part1.md)
**Task Reference:** T-03.6
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Define and implement the YAML configuration schema for model training, including hyperparameters, validation settings, and per-model configurations.

---

## Acceptance Criteria

- [ ] Configuration module created in `R/models/config.R`
- [ ] YAML schema defined for model configuration
- [ ] `validate_model_config()` validates configuration structure
- [ ] `apply_model_config_defaults()` fills missing values
- [ ] Support for model-specific hyperparameters
- [ ] Support for horizon-specific configurations
- [ ] Environment variable substitution support

---

## Technical Specification

### File Location
```
R/models/config.R
```

### Configuration Schema

```yaml
# models.yaml - Model training configuration schema
models:
  # Default training settings
  defaults:
    seed: 42
    validation_split: 0.15
    test_split: 0.0
    target_col: "CargaGlobal"
    exclude_cols:
      - DataHora
      - area_code

  # Default hyperparameters by model type
  hyperparameters:
    lightgbm:
      objective: "regression"
      metric: "mape"
      num_leaves: 31
      learning_rate: 0.05
      feature_fraction: 0.9
      bagging_fraction: 0.8
      bagging_freq: 5
      num_iterations: 1000
      early_stopping_rounds: 50

    random_forest:
      num_trees: 500
      max_depth: 20
      min_samples_leaf: 5
      n_jobs: -1

    holt_winters:
      seasonal: "additive"
      seasonal_periods: 24
      damped: false

  # Model definitions
  plugins:
    - name: lgbm_base
      type: lightgbm
      horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]
      enabled: true
      config:
        num_leaves: 31
        learning_rate: 0.05

    - name: lgbm_deep
      type: lightgbm
      horizons: [0, 1, 2]
      enabled: true
      config:
        num_leaves: 63
        learning_rate: 0.03

    - name: rf_ensemble
      type: random_forest
      horizons: [0, 1, 2, 3]
      enabled: false
      config:
        num_trees: 1000

  # Training schedule
  training:
    retrain_frequency: "weekly"
    retrain_day: "sunday"
    retrain_hour: 2

  # Artifact storage
  artifacts:
    base_path: "models"
    keep_versions: 5
    compress: false
```

### Configuration Validator

```r
#' @title Model Configuration Schema
#' @description Constants and validation for model configuration

#' Required fields in model plugin config
MODEL_CONFIG_REQUIRED <- c("name", "type")

#' Valid model types
MODEL_TYPES <- c("lightgbm", "random_forest", "xgboost",
                 "holt_winters", "arima", "prophet", "custom")

#' Default hyperparameters by model type
DEFAULT_HYPERPARAMETERS <- list(
  lightgbm = list(
    objective = "regression",
    metric = "mape",
    num_leaves = 31,
    learning_rate = 0.05,
    num_iterations = 1000
  ),
  random_forest = list(
    num_trees = 500,
    max_depth = 20
  ),
  holt_winters = list(
    seasonal = "additive",
    seasonal_periods = 24
  )
)


#' Validate model configuration
#'
#' @param config Configuration list (parsed YAML)
#' @return TRUE if valid, throws error otherwise
#' @export
validate_model_config <- function(config) {
  checkmate::assert_list(config)

  # Validate defaults if present
  if ("defaults" %in% names(config)) {
    validate_defaults_config(config$defaults)
  }

  # Validate hyperparameters if present
  if ("hyperparameters" %in% names(config)) {
    validate_hyperparameters_config(config$hyperparameters)
  }

  # Validate plugins
  if ("plugins" %in% names(config)) {
    for (i in seq_along(config$plugins)) {
      validate_model_plugin_config(config$plugins[[i]], index = i)
    }
  }

  # Validate training schedule if present
  if ("training" %in% names(config)) {
    validate_training_config(config$training)
  }

  invisible(TRUE)
}


#' Validate defaults section
#' @noRd
validate_defaults_config <- function(defaults) {
  checkmate::assert_list(defaults)

  if ("seed" %in% names(defaults)) {
    checkmate::assert_integerish(defaults$seed, len = 1)
  }

  if ("validation_split" %in% names(defaults)) {
    checkmate::assert_number(defaults$validation_split, lower = 0, upper = 1)
  }

  if ("test_split" %in% names(defaults)) {
    checkmate::assert_number(defaults$test_split, lower = 0, upper = 1)
  }

  invisible(TRUE)
}


#' Validate hyperparameters section
#' @noRd
validate_hyperparameters_config <- function(hyperparams) {
  checkmate::assert_list(hyperparams, names = "named")

  for (model_type in names(hyperparams)) {
    checkmate::assert_choice(model_type, MODEL_TYPES)
    checkmate::assert_list(hyperparams[[model_type]])
  }

  invisible(TRUE)
}


#' Validate single model plugin configuration
#' @noRd
validate_model_plugin_config <- function(plugin_conf, index = NULL) {
  idx_msg <- if (!is.null(index)) sprintf(" at index %d", index) else ""

  if (!is.list(plugin_conf)) {
    stop(sprintf("Model plugin config%s must be a list", idx_msg))
  }

  # Check required fields
  for (field in MODEL_CONFIG_REQUIRED) {
    if (!field %in% names(plugin_conf)) {
      stop(sprintf("Model plugin%s missing required '%s' field", idx_msg, field))
    }
  }

  # Validate type
  checkmate::assert_choice(plugin_conf$type, MODEL_TYPES)

  # Validate horizons if present
  if ("horizons" %in% names(plugin_conf)) {
    checkmate::assert_integerish(plugin_conf$horizons, lower = 0, upper = 8)
  }

  # Validate enabled if present
  if ("enabled" %in% names(plugin_conf)) {
    checkmate::assert_logical(plugin_conf$enabled, len = 1)
  }

  invisible(TRUE)
}


#' Validate training schedule configuration
#' @noRd
validate_training_config <- function(training) {
  checkmate::assert_list(training)

  if ("retrain_frequency" %in% names(training)) {
    checkmate::assert_choice(
      training$retrain_frequency,
      c("daily", "weekly", "monthly", "manual")
    )
  }

  if ("retrain_day" %in% names(training)) {
    checkmate::assert_choice(
      tolower(training$retrain_day),
      c("sunday", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday")
    )
  }

  if ("retrain_hour" %in% names(training)) {
    checkmate::assert_integerish(training$retrain_hour, lower = 0, upper = 23)
  }

  invisible(TRUE)
}
```

### Configuration Defaults and Merging

```r
#' Apply default values to model configuration
#'
#' @param config Configuration list
#' @return Configuration with defaults applied
#' @export
apply_model_config_defaults <- function(config) {
  result <- config

  # Apply top-level defaults
  if (!"defaults" %in% names(result)) {
    result$defaults <- list()
  }

  default_defaults <- list(
    seed = 42,
    validation_split = 0.15,
    test_split = 0.0,
    target_col = "CargaGlobal",
    exclude_cols = c("DataHora", "area_code")
  )

  for (key in names(default_defaults)) {
    if (!key %in% names(result$defaults)) {
      result$defaults[[key]] <- default_defaults[[key]]
    }
  }

  # Apply plugin-level defaults
  if ("plugins" %in% names(result)) {
    result$plugins <- lapply(result$plugins, function(plugin) {
      plugin$enabled <- plugin$enabled %||% TRUE
      plugin$horizons <- plugin$horizons %||% 0:8

      # Merge with type-specific defaults
      if (plugin$type %in% names(DEFAULT_HYPERPARAMETERS)) {
        type_defaults <- DEFAULT_HYPERPARAMETERS[[plugin$type]]
        plugin$config <- merge_lists(type_defaults, plugin$config %||% list())
      }

      plugin
    })
  }

  # Apply artifacts defaults
  if (!"artifacts" %in% names(result)) {
    result$artifacts <- list()
  }

  artifact_defaults <- list(
    base_path = "models",
    keep_versions = 5,
    compress = FALSE
  )

  for (key in names(artifact_defaults)) {
    if (!key %in% names(result$artifacts)) {
      result$artifacts[[key]] <- artifact_defaults[[key]]
    }
  }

  result
}


#' Merge two lists (second takes precedence)
#' @noRd
merge_lists <- function(base, override) {
  result <- base

  for (key in names(override)) {
    if (is.list(override[[key]]) && is.list(result[[key]])) {
      result[[key]] <- merge_lists(result[[key]], override[[key]])
    } else {
      result[[key]] <- override[[key]]
    }
  }

  result
}


#' Get configuration for a specific model
#'
#' @param config Full configuration
#' @param model_name Name of the model
#' @return Model-specific configuration
#' @export
get_model_config <- function(config, model_name) {
  if (!"plugins" %in% names(config)) {
    return(NULL)
  }

  for (plugin in config$plugins) {
    if (plugin$name == model_name) {
      return(plugin)
    }
  }

  NULL
}


#' Load and process model configuration from YAML
#'
#' @param path Path to YAML file
#' @param overrides Optional override configuration
#' @return Processed configuration
#' @export
load_model_config <- function(path, overrides = NULL) {
  checkmate::assert_file_exists(path)

  config <- yaml::read_yaml(path)

  # Extract models section if present
  if ("models" %in% names(config)) {
    config <- config$models
  }

  # Apply overrides if provided
  if (!is.null(overrides)) {
    config <- merge_lists(config, overrides)
  }

  # Apply defaults
  config <- apply_model_config_defaults(config)

  # Resolve environment variables
  config <- resolve_config_env_vars(config)

  # Validate
  validate_model_config(config)

  config
}


#' Resolve environment variable references
#'
#' @param config Configuration list
#' @return Configuration with env vars resolved
#' @noRd
resolve_config_env_vars <- function(config) {
  resolve_value <- function(value) {
    if (is.character(value) && length(value) == 1) {
      # Handle $ENV{VARIABLE} references
      while (grepl("\\$ENV\\{([^}]+)\\}", value)) {
        env_var <- sub(".*\\$ENV\\{([^}]+)\\}.*", "\\1", value)
        replacement <- Sys.getenv(env_var, unset = "")
        value <- sub("\\$ENV\\{[^}]+\\}", replacement, value)
      }
    } else if (is.list(value)) {
      value <- lapply(value, resolve_value)
    }
    value
  }

  resolve_value(config)
}
```

### Usage Example

```r
# Load configuration
config <- load_model_config("models.yaml")

# Get enabled models
enabled_models <- Filter(
  function(p) p$enabled,
  config$plugins
)

# Get specific model config
lgbm_config <- get_model_config(config, "lgbm_base")
# $name: "lgbm_base"
# $type: "lightgbm"
# $horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# $config:
#   $num_leaves: 31
#   $learning_rate: 0.05

# Create model from config
model <- get_model(
  lgbm_config$type,
  horizons = lgbm_config$horizons,
  config = lgbm_config$config
)

# Override at runtime
config <- load_model_config(
  "models.yaml",
  overrides = list(
    defaults = list(validation_split = 0.2),
    plugins = list(
      list(name = "lgbm_base", config = list(learning_rate = 0.1))
    )
  )
)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Validate valid config | TRUE |
| TC-002 | Validate missing plugins | No error (optional) |
| TC-003 | Validate plugin missing name | Error |
| TC-004 | Validate invalid model type | Error |
| TC-005 | Validate invalid horizons | Error |
| TC-006 | Apply defaults empty config | Defaults added |
| TC-007 | Apply type-specific defaults | Hyperparams merged |
| TC-008 | get_model_config() existing | Returns config |
| TC-009 | get_model_config() missing | Returns NULL |
| TC-010 | load_model_config() valid | Processed config |
| TC-011 | load_model_config() with overrides | Merged |
| TC-012 | Resolve $ENV{VAR} | Environment value |
| TC-013 | Validate training schedule | Valid values |

---

## Definition of Done

- [ ] Configuration schema documented
- [ ] Validation functions implemented
- [ ] Default application working
- [ ] Type-specific hyperparameters
- [ ] Environment variable resolution
- [ ] YAML loading integrated
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Configuration format should align with EPIC-08 (Configuration System)
- Model type defaults provide sensible starting points
- Environment variables enable secrets and environment-specific config
- Consider adding config inheritance for model families
- The `enabled` flag allows toggling models without removing them
