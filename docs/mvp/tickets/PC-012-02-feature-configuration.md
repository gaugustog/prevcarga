# PC-012-02: Feature Configuration Schema

**Epic:** [EPIC-02: Feature Engineering Infrastructure](../epics/EPIC-02-feature-engineering.md)
**Task Reference:** T-02.4
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Define and implement the YAML configuration schema for feature engineering, including validation, defaults, and integration with the FeaturePipeline.

---

## Acceptance Criteria

- [ ] Configuration module created in `R/features/config.R`
- [ ] YAML schema defined for feature configuration
- [ ] `validate_feature_config()` validates configuration structure
- [ ] `apply_config_defaults()` fills missing values with defaults
- [ ] `merge_configs()` combines base and override configurations
- [ ] Plugin ordering support via configuration
- [ ] Environment variable substitution support

---

## Technical Specification

### File Location
```
R/features/config.R
```

### Configuration Schema

```yaml
# features.yaml - Feature pipeline configuration schema
features:
 # Pipeline metadata
 pipeline_name: "default_pipeline"
 version: "1.0"

 # Global settings applied to all plugins
 defaults:
   datetime_col: "DataHora"
   target_col: "CargaGlobal"

 # Plugin definitions (executed in order)
 plugins:
   - name: temporal_basic
     enabled: true
     config:
       datetime_col: "${defaults.datetime_col}"

   - name: calendar
     enabled: true
     config:
       include_holidays: true
       holiday_types:
         - nacional
         - estadual

   - name: lag_features
     enabled: true
     config:
       target_col: "${defaults.target_col}"
       lags: [1, 24, 48, 168]
       rolling_windows: [24, 168]

   - name: weather
     enabled: false  # Disabled by default
     config:
       variables:
         - temperatura
         - umidade

 # Feature selection (optional)
 selection:
   method: "all"  # all, importance, manual
   top_k: null
   include: []
   exclude: []
```

### Configuration Validator

```r
#' @title Feature Configuration Schema
#' @description Constants for configuration validation

#' Required fields in plugin config
PLUGIN_CONFIG_REQUIRED <- c("name")

#' Optional fields with defaults
PLUGIN_CONFIG_DEFAULTS <- list(
 enabled = TRUE,
 config = list()
)

#' Valid selection methods
SELECTION_METHODS <- c("all", "importance", "correlation", "manual")


#' Validate feature configuration
#'
#' @param config Configuration list (parsed YAML)
#' @return TRUE if valid, throws error otherwise
#' @export
validate_feature_config <- function(config) {
 checkmate::assert_list(config)

 # Check top-level structure
 if (!"plugins" %in% names(config)) {
   stop("Configuration must contain 'plugins' list")
 }

 plugins <- config$plugins
 if (!is.list(plugins) || length(plugins) == 0) {
   stop("'plugins' must be a non-empty list")
 }

 # Validate each plugin configuration
 for (i in seq_along(plugins)) {
   plugin_conf <- plugins[[i]]
   validate_plugin_config(plugin_conf, index = i)
 }

 # Validate selection config if present
 if ("selection" %in% names(config)) {
   validate_selection_config(config$selection)
 }

 invisible(TRUE)
}


#' Validate single plugin configuration
#'
#' @param plugin_conf Plugin configuration list
#' @param index Plugin index for error messages
#' @return TRUE if valid
#' @noRd
validate_plugin_config <- function(plugin_conf, index = NULL) {
 idx_msg <- if (!is.null(index)) sprintf(" at index %d", index) else ""

 if (!is.list(plugin_conf)) {
   stop(sprintf("Plugin config%s must be a list", idx_msg))
 }

 # Check required fields
 if (!"name" %in% names(plugin_conf)) {
   stop(sprintf("Plugin config%s missing required 'name' field", idx_msg))
 }

 name <- plugin_conf$name
 checkmate::assert_string(name, min.chars = 1)

 # Validate enabled field if present
 if ("enabled" %in% names(plugin_conf)) {
   checkmate::assert_logical(plugin_conf$enabled, len = 1)
 }

 # Validate config field if present
 if ("config" %in% names(plugin_conf)) {
   checkmate::assert_list(plugin_conf$config)
 }

 invisible(TRUE)
}


#' Validate selection configuration
#'
#' @param selection_conf Selection configuration list
#' @return TRUE if valid
#' @noRd
validate_selection_config <- function(selection_conf) {
 checkmate::assert_list(selection_conf)

 if ("method" %in% names(selection_conf)) {
   checkmate::assert_choice(selection_conf$method, SELECTION_METHODS)
 }

 if ("top_k" %in% names(selection_conf)) {
   if (!is.null(selection_conf$top_k)) {
     checkmate::assert_integerish(selection_conf$top_k, lower = 1)
   }
 }

 if ("include" %in% names(selection_conf)) {
   checkmate::assert_character(selection_conf$include)
 }

 if ("exclude" %in% names(selection_conf)) {
   checkmate::assert_character(selection_conf$exclude)
 }

 invisible(TRUE)
}
```

### Configuration Defaults

```r
#' Apply default values to configuration
#'
#' @param config Configuration list
#' @return Configuration with defaults applied
#' @export
apply_config_defaults <- function(config) {
 result <- config

 # Apply pipeline-level defaults
 result$pipeline_name <- result$pipeline_name %||% "default_pipeline"
 result$version <- result$version %||% "1.0"
 result$defaults <- result$defaults %||% list()

 # Apply plugin-level defaults
 result$plugins <- lapply(result$plugins, function(plugin_conf) {
   for (field in names(PLUGIN_CONFIG_DEFAULTS)) {
     if (!field %in% names(plugin_conf)) {
       plugin_conf[[field]] <- PLUGIN_CONFIG_DEFAULTS[[field]]
     }
   }
   plugin_conf
 })

 # Apply selection defaults
 if (!"selection" %in% names(result)) {
   result$selection <- list(method = "all")
 }

 result
}


#' Merge base and override configurations
#'
#' @param base Base configuration
#' @param override Override configuration (takes precedence)
#' @return Merged configuration
#' @export
merge_configs <- function(base, override) {
 checkmate::assert_list(base)
 checkmate::assert_list(override)

 result <- base

 for (key in names(override)) {
   if (key == "plugins") {
     # Special handling for plugins: merge by name
     result$plugins <- merge_plugin_lists(
       base$plugins %||% list(),
       override$plugins
     )
   } else if (is.list(override[[key]]) && is.list(result[[key]])) {
     # Recursive merge for nested lists
     result[[key]] <- merge_configs(result[[key]], override[[key]])
   } else {
     # Direct override
     result[[key]] <- override[[key]]
   }
 }

 result
}


#' Merge plugin lists by name
#'
#' @param base_plugins Base plugin list
#' @param override_plugins Override plugin list
#' @return Merged plugin list
#' @noRd
merge_plugin_lists <- function(base_plugins, override_plugins) {
 # Create lookup by name
 base_by_name <- setNames(base_plugins, sapply(base_plugins, `[[`, "name"))

 for (override_plugin in override_plugins) {
   name <- override_plugin$name
   if (name %in% names(base_by_name)) {
     # Merge existing plugin config
     base_by_name[[name]] <- merge_configs(
       base_by_name[[name]],
       override_plugin
     )
   } else {
     # Add new plugin
     base_by_name[[name]] <- override_plugin
   }
 }

 unname(base_by_name)
}
```

### Variable Substitution

```r
#' Resolve variable references in configuration
#'
#' Supports ${variable} syntax for referencing other config values
#' and $ENV{VARIABLE} for environment variables.
#'
#' @param config Configuration list
#' @return Configuration with variables resolved
#' @export
resolve_config_variables <- function(config) {
 # First pass: collect defaults
 defaults <- config$defaults %||% list()

 # Recursive resolution function
 resolve_value <- function(value, context) {
   if (is.character(value) && length(value) == 1) {
     # Handle ${defaults.key} references
     while (grepl("\\$\\{defaults\\.([^}]+)\\}", value)) {
       key <- sub(".*\\$\\{defaults\\.([^}]+)\\}.*", "\\1", value)
       replacement <- defaults[[key]] %||% ""
       value <- sub("\\$\\{defaults\\.[^}]+\\}", replacement, value)
     }

     # Handle $ENV{VARIABLE} references
     while (grepl("\\$ENV\\{([^}]+)\\}", value)) {
       env_var <- sub(".*\\$ENV\\{([^}]+)\\}.*", "\\1", value)
       replacement <- Sys.getenv(env_var, unset = "")
       value <- sub("\\$ENV\\{[^}]+\\}", replacement, value)
     }
   } else if (is.list(value)) {
     value <- lapply(value, resolve_value, context = context)
   }
   value
 }

 resolve_value(config, defaults)
}


#' Load and process feature configuration from YAML
#'
#' @param path Path to YAML file
#' @param overrides Optional override configuration
#' @return Processed configuration
#' @export
load_feature_config <- function(path, overrides = NULL) {
 checkmate::assert_file_exists(path)

 config <- yaml::read_yaml(path)

 # Extract features section if present
 if ("features" %in% names(config)) {
   config <- config$features
 }

 # Apply overrides if provided
 if (!is.null(overrides)) {
   config <- merge_configs(config, overrides)
 }

 # Apply defaults
 config <- apply_config_defaults(config)

 # Resolve variables
 config <- resolve_config_variables(config)

 # Validate
 validate_feature_config(config)

 config
}
```

### Usage Example

```r
# Load configuration
config <- load_feature_config("features.yaml")

# Load with overrides
config <- load_feature_config(
 "features.yaml",
 overrides = list(
   plugins = list(
     list(name = "lag_features", config = list(lags = c(1, 24)))
   )
 )
)

# Build pipeline from config
pipeline <- FeaturePipeline$new()
pipeline$from_config(config)

# Validate configuration programmatically
my_config <- list(
 plugins = list(
   list(name = "temporal", enabled = TRUE),
   list(name = "calendar", config = list(holidays = TRUE))
 )
)
validate_feature_config(my_config)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Validate valid config | TRUE |
| TC-002 | Validate missing plugins | Error |
| TC-003 | Validate plugin missing name | Error |
| TC-004 | Validate invalid enabled type | Error |
| TC-005 | Apply defaults empty config | Defaults added |
| TC-006 | Apply defaults partial config | Missing filled |
| TC-007 | Merge configs simple | Override wins |
| TC-008 | Merge configs nested | Deep merge |
| TC-009 | Merge plugin lists by name | Merged correctly |
| TC-010 | Resolve ${defaults.x} | Substituted |
| TC-011 | Resolve $ENV{VAR} | Environment value |
| TC-012 | load_feature_config valid | Processed config |
| TC-013 | load_feature_config with overrides | Merged and processed |
| TC-014 | Validate selection config | All methods valid |

---

## Definition of Done

- [ ] Configuration schema documented
- [ ] Validation functions implemented
- [ ] Default application working
- [ ] Config merging working
- [ ] Variable substitution working
- [ ] YAML loading integrated
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Configuration format should align with EPIC-08 (Configuration System)
- Consider JSON Schema for formal validation in the future
- Environment variable substitution enables secrets management
- The `enabled` flag allows disabling plugins without removing them
- Plugin order in the YAML file determines execution order
