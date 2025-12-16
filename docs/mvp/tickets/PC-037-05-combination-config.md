# PC-037-05: Combination Configuration Schema

**Epic:** [EPIC-05: Combination Infrastructure](../epics/EPIC-05-combination-layer.md)
**Task Reference:** T-05.6
**Priority:** Medium
**Estimated Effort:** 0.5 days

---

## Summary

Define and implement the YAML configuration schema for forecast combination, including combiner selection, weight settings, bias correction options, and model inclusion rules.

---

## Acceptance Criteria

- [ ] Configuration module created in `R/combination/config.R`
- [ ] YAML schema defined for combination settings
- [ ] `validate_combination_config()` validates structure
- [ ] `apply_combination_defaults()` fills missing values
- [ ] Support for per-horizon combination strategies
- [ ] Environment variable substitution

---

## Technical Specification

### File Location
```
R/combination/config.R
```

### Configuration Schema

```yaml
# combination.yaml - Forecast combination configuration schema
combination:
  # Default combination strategy
  strategy: weighted_average

  # Models to include in combination
  models:
    - name: lgbm_base
      enabled: true
      weight: null  # null = auto-optimize

    - name: lgbm_deep
      enabled: true
      weight: 0.3

    - name: rf_ensemble
      enabled: false  # Excluded from combination

    - name: holt_winters
      enabled: true
      weight: null

  # Weight optimization settings
  weight_optimization:
    enabled: true
    method: inverse_mse  # inverse_mse, inverse_mae, ols, constrained_ols
    lookback_days: 30
    min_samples: 168  # 1 week of hourly data
    update_frequency: daily  # daily, weekly, manual
    constraints:
      min_weight: 0.0
      max_weight: 1.0
      normalize: true

  # Bias correction settings
  bias_correction:
    enabled: true
    type: conditional  # simple, conditional, regression
    correction_type: additive  # additive, multiplicative
    grouping_cols:
      - hour
      - weekday
    lookback_days: 14

  # Per-horizon combination (optional override)
  horizons:
    D+0:
      strategy: weighted_average
      models: [lgbm_base, holt_winters]
      weight_optimization:
        method: inverse_mse
        lookback_days: 7

    D+1:
      strategy: weighted_average
      models: [lgbm_base, lgbm_deep, rf_ensemble]

    D+2_D+8:
      strategy: simple_average
      models: [lgbm_base, holt_winters]

  # Output settings
  output:
    save_weights: true
    save_individual_forecasts: true
    save_metrics: true
```

### Configuration Validator

```r
#' @title Combination Configuration Constants
#' @description Constants for combination configuration validation

#' Valid combination strategies
COMBINATION_STRATEGIES <- c(
  "simple_average",
  "weighted_average",
  "optimal_ols",
  "best_model",
  "stacking"
)

#' Valid weight optimization methods
WEIGHT_METHODS <- c(
  "inverse_mse",
  "inverse_mae",
  "ols",
  "constrained_ols",
  "equal"
)

#' Valid bias correction types
BIAS_CORRECTION_TYPES <- c(
  "simple",
  "conditional",
  "regression"
)


#' Validate combination configuration
#'
#' @param config Configuration list (parsed YAML)
#' @return TRUE if valid, throws error otherwise
#' @export
validate_combination_config <- function(config) {
  checkmate::assert_list(config)

  # Validate strategy
  if ("strategy" %in% names(config)) {
    checkmate::assert_choice(config$strategy, COMBINATION_STRATEGIES)
  }

  # Validate models section
  if ("models" %in% names(config)) {
    validate_models_config(config$models)
  }

  # Validate weight optimization
  if ("weight_optimization" %in% names(config)) {
    validate_weight_optimization_config(config$weight_optimization)
  }

  # Validate bias correction
  if ("bias_correction" %in% names(config)) {
    validate_bias_correction_config(config$bias_correction)
  }

  # Validate horizons
  if ("horizons" %in% names(config)) {
    validate_horizons_config(config$horizons)
  }

  invisible(TRUE)
}


#' Validate models configuration
#' @noRd
validate_models_config <- function(models) {
  checkmate::assert_list(models, min.len = 1)

  for (i in seq_along(models)) {
    model <- models[[i]]

    if (!"name" %in% names(model)) {
      stop(sprintf("Model at index %d missing 'name' field", i))
    }

    checkmate::assert_string(model$name, min.chars = 1)

    if ("enabled" %in% names(model)) {
      checkmate::assert_logical(model$enabled, len = 1)
    }

    if ("weight" %in% names(model) && !is.null(model$weight)) {
      checkmate::assert_number(model$weight, lower = 0, upper = 1)
    }
  }

  invisible(TRUE)
}


#' Validate weight optimization configuration
#' @noRd
validate_weight_optimization_config <- function(config) {
  checkmate::assert_list(config)

  if ("enabled" %in% names(config)) {
    checkmate::assert_logical(config$enabled, len = 1)
  }

  if ("method" %in% names(config)) {
    checkmate::assert_choice(config$method, WEIGHT_METHODS)
  }

  if ("lookback_days" %in% names(config)) {
    checkmate::assert_integerish(config$lookback_days, lower = 1)
  }

  if ("min_samples" %in% names(config)) {
    checkmate::assert_integerish(config$min_samples, lower = 1)
  }

  if ("constraints" %in% names(config)) {
    constraints <- config$constraints
    if ("min_weight" %in% names(constraints)) {
      checkmate::assert_number(constraints$min_weight, lower = 0, upper = 1)
    }
    if ("max_weight" %in% names(constraints)) {
      checkmate::assert_number(constraints$max_weight, lower = 0, upper = 1)
    }
  }

  invisible(TRUE)
}


#' Validate bias correction configuration
#' @noRd
validate_bias_correction_config <- function(config) {
  checkmate::assert_list(config)

  if ("enabled" %in% names(config)) {
    checkmate::assert_logical(config$enabled, len = 1)
  }

  if ("type" %in% names(config)) {
    checkmate::assert_choice(config$type, BIAS_CORRECTION_TYPES)
  }

  if ("correction_type" %in% names(config)) {
    checkmate::assert_choice(config$correction_type, c("additive", "multiplicative"))
  }

  if ("grouping_cols" %in% names(config)) {
    checkmate::assert_character(config$grouping_cols, min.len = 1)
  }

  invisible(TRUE)
}


#' Validate horizons configuration
#' @noRd
validate_horizons_config <- function(horizons) {
  checkmate::assert_list(horizons, names = "named")

  for (horizon_name in names(horizons)) {
    horizon_config <- horizons[[horizon_name]]

    if ("strategy" %in% names(horizon_config)) {
      checkmate::assert_choice(horizon_config$strategy, COMBINATION_STRATEGIES)
    }

    if ("models" %in% names(horizon_config)) {
      checkmate::assert_character(horizon_config$models, min.len = 1)
    }
  }

  invisible(TRUE)
}
```

### Configuration Defaults and Loading

```r
#' Apply default values to combination configuration
#'
#' @param config Configuration list
#' @return Configuration with defaults applied
#' @export
apply_combination_defaults <- function(config) {
  result <- config

  # Strategy default
  result$strategy <- result$strategy %||% "weighted_average"

  # Weight optimization defaults
  if (!"weight_optimization" %in% names(result)) {
    result$weight_optimization <- list()
  }

  wo_defaults <- list(
    enabled = TRUE,
    method = "inverse_mse",
    lookback_days = 30,
    min_samples = 168,
    update_frequency = "daily",
    constraints = list(
      min_weight = 0.0,
      max_weight = 1.0,
      normalize = TRUE
    )
  )

  for (key in names(wo_defaults)) {
    if (!key %in% names(result$weight_optimization)) {
      result$weight_optimization[[key]] <- wo_defaults[[key]]
    }
  }

  # Bias correction defaults
  if (!"bias_correction" %in% names(result)) {
    result$bias_correction <- list()
  }

  bc_defaults <- list(
    enabled = FALSE,
    type = "simple",
    correction_type = "additive",
    lookback_days = 14
  )

  for (key in names(bc_defaults)) {
    if (!key %in% names(result$bias_correction)) {
      result$bias_correction[[key]] <- bc_defaults[[key]]
    }
  }

  # Output defaults
  if (!"output" %in% names(result)) {
    result$output <- list()
  }

  output_defaults <- list(
    save_weights = TRUE,
    save_individual_forecasts = FALSE,
    save_metrics = TRUE
  )

  for (key in names(output_defaults)) {
    if (!key %in% names(result$output)) {
      result$output[[key]] <- output_defaults[[key]]
    }
  }

  result
}


#' Get enabled models from configuration
#'
#' @param config Combination configuration
#' @return Character vector of enabled model names
#' @export
get_enabled_models <- function(config) {
  if (!"models" %in% names(config)) {
    return(character())
  }

  sapply(
    Filter(function(m) m$enabled %||% TRUE, config$models),
    function(m) m$name
  )
}


#' Get manual weights from configuration
#'
#' @param config Combination configuration
#' @return Named numeric vector of weights (NULL values excluded)
#' @export
get_manual_weights <- function(config) {
  if (!"models" %in% names(config)) {
    return(NULL)
  }

  weights <- sapply(config$models, function(m) m$weight)
  names(weights) <- sapply(config$models, function(m) m$name)

  # Filter out NULLs
  weights <- weights[!sapply(weights, is.null)]

  if (length(weights) == 0) {
    return(NULL)
  }

  unlist(weights)
}


#' Get horizon-specific configuration
#'
#' @param config Full combination configuration
#' @param horizon Horizon name (e.g., "D+0", "D+1")
#' @return Merged configuration for horizon
#' @export
get_horizon_config <- function(config, horizon) {
  # Start with base config
  result <- config

  # Override with horizon-specific if present
  if ("horizons" %in% names(config) && horizon %in% names(config$horizons)) {
    horizon_config <- config$horizons[[horizon]]

    for (key in names(horizon_config)) {
      if (is.list(horizon_config[[key]]) && is.list(result[[key]])) {
        # Merge nested lists
        result[[key]] <- merge_lists(result[[key]], horizon_config[[key]])
      } else {
        result[[key]] <- horizon_config[[key]]
      }
    }
  }

  result
}


#' Load combination configuration from YAML
#'
#' @param path Path to YAML file
#' @param overrides Optional override configuration
#' @return Processed configuration
#' @export
load_combination_config <- function(path, overrides = NULL) {
  checkmate::assert_file_exists(path)

  config <- yaml::read_yaml(path)

  # Extract combination section if present
  if ("combination" %in% names(config)) {
    config <- config$combination
  }

  # Apply overrides
  if (!is.null(overrides)) {
    config <- merge_lists(config, overrides)
  }

  # Apply defaults
  config <- apply_combination_defaults(config)

  # Resolve environment variables
  config <- resolve_config_env_vars(config)

  # Validate
  validate_combination_config(config)

  config
}
```

### Usage Example

```r
# Load configuration
config <- load_combination_config("combination.yaml")

# Get enabled models
enabled <- get_enabled_models(config)
# [1] "lgbm_base" "lgbm_deep" "holt_winters"

# Get manual weights (models with explicit weights)
manual_weights <- get_manual_weights(config)
# lgbm_deep
#      0.3

# Get horizon-specific config
d0_config <- get_horizon_config(config, "D+0")
# Uses D+0 specific models and weight settings

# Override at runtime
config <- load_combination_config(
  "combination.yaml",
  overrides = list(
    strategy = "simple_average",
    weight_optimization = list(enabled = FALSE)
  )
)

# Create workflow from config
combiner <- get_combiner(config$strategy)

if (config$weight_optimization$enabled) {
  optimizer <- StandardWeightOptimizer$new(
    config = config$weight_optimization$constraints
  )
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | validate_combination_config() valid | TRUE |
| TC-002 | validate_combination_config() invalid strategy | Error |
| TC-003 | validate_models_config() missing name | Error |
| TC-004 | validate_weight_optimization_config() | Validates method |
| TC-005 | apply_combination_defaults() empty | All defaults applied |
| TC-006 | apply_combination_defaults() partial | Only missing filled |
| TC-007 | get_enabled_models() | Filters disabled |
| TC-008 | get_manual_weights() | Returns non-null weights |
| TC-009 | get_horizon_config() | Merges configs |
| TC-010 | load_combination_config() | Full pipeline |
| TC-011 | load_combination_config() with overrides | Overrides applied |

---

## Dependencies

- PC-032-05: BaseCombiner
- PC-033-05: CombinerRegistry

---

## Definition of Done

- [ ] Configuration schema documented
- [ ] Validation functions implemented
- [ ] Default application working
- [ ] Horizon-specific config support
- [ ] Environment variable resolution
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Configuration aligns with EPIC-08 (Configuration System)
- Per-horizon configs enable different strategies for D+0 vs D+1+
- Manual weights can override optimization
- Consider adding model exclusion rules (e.g., exclude if MAPE > threshold)
- Future: add config validation against registered models
