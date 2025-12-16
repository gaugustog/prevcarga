# PC-057-08: ConfigManager

**Epic:** [EPIC-08: Orchestrator](../epics/EPIC-08-orchestrator.md)
**Task Reference:** T-08.1
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `ConfigManager` R6 class for loading, validating, and accessing YAML configuration files. This is the central configuration management component for the orchestration layer.

---

## Acceptance Criteria

- [ ] Config module created in `R/orchestrator/config.R`
- [ ] `ConfigManager` R6 class with YAML loading
- [ ] Configuration validation with required sections
- [ ] Nested key access with dot notation
- [ ] Default value support
- [ ] Environment variable interpolation

---

## Technical Specification

### File Location
```
R/orchestrator/config.R
```

### ConfigManager R6 Class

```r
#' @title ConfigManager
#' @description Central configuration management for PrevCarga
#'
#' Loads YAML configuration files, validates required sections,
#' and provides convenient access to configuration values.
#'
#' @export
ConfigManager <- R6::R6Class(
  "ConfigManager",
  private = list(
    config = NULL,
    config_path = NULL,
    environment = NULL,
    loaded = FALSE
  ),
  public = list(
    #' @description Initialize config manager
    #' @param environment Environment name (production, staging, development)
    initialize = function(environment = NULL) {
      private$environment <- environment %||% Sys.getenv("PREVCARGA_ENV", "development")
    },

    #' @description Load configuration from YAML file
    #' @param path Path to YAML configuration file
    #' @return Invisible self
    load = function(path) {
      checkmate::assert_file_exists(path)

      private$config_path <- path
      private$config <- yaml::read_yaml(path)

      # Interpolate environment variables
      private$config <- self$interpolate_env(private$config)

      # Apply environment-specific overrides
      self$apply_environment_overrides()

      # Validate configuration
      self$validate()

      private$loaded <- TRUE
      message(sprintf("Configuration loaded from: %s", path))

      invisible(self)
    },

    #' @description Validate configuration
    #' @return TRUE if valid, error otherwise
    validate = function() {
      required_sections <- c("project", "storage", "regions", "models")

      missing <- setdiff(required_sections, names(private$config))
      if (length(missing) > 0) {
        stop(sprintf(
          "Missing required configuration sections: %s",
          paste(missing, collapse = ", ")
        ))
      }

      # Validate project section
      if (is.null(private$config$project$name)) {
        stop("Configuration missing 'project.name'")
      }

      # Validate storage section
      if (is.null(private$config$storage$backend)) {
        stop("Configuration missing 'storage.backend'")
      }

      # Validate regions section
      if (!is.list(private$config$regions$areas)) {
        stop("Configuration 'regions.areas' must be a list")
      }

      invisible(TRUE)
    },

    #' @description Get configuration value
    #' @param key Dot-notation key (e.g., "storage.backend")
    #' @return Configuration value or NULL
    get = function(key = NULL) {
      if (!private$loaded) {
        stop("Configuration not loaded. Call load() first.")
      }

      if (is.null(key)) {
        return(private$config)
      }

      # Support nested keys with dot notation
      keys <- strsplit(key, "\\.")[[1]]
      result <- private$config

      for (k in keys) {
        if (is.null(result[[k]])) {
          return(NULL)
        }
        result <- result[[k]]
      }

      result
    },

    #' @description Get configuration with default value
    #' @param key Configuration key
    #' @param default Default value if key not found
    #' @return Configuration value or default
    get_with_default = function(key, default) {
      result <- self$get(key)
      if (is.null(result)) default else result
    },

    #' @description Check if configuration key exists
    #' @param key Configuration key
    #' @return Logical
    has = function(key) {
      !is.null(self$get(key))
    },

    #' @description Get all keys at a level
    #' @param prefix Key prefix
    #' @return Character vector of keys
    keys = function(prefix = NULL) {
      if (is.null(prefix)) {
        return(names(private$config))
      }

      section <- self$get(prefix)
      if (is.list(section)) {
        names(section)
      } else {
        character(0)
      }
    },

    #' @description Interpolate environment variables
    #' @param config Configuration list
    #' @return Configuration with interpolated values
    interpolate_env = function(config) {
      if (is.character(config)) {
        # Replace ${VAR} or $VAR patterns
        pattern <- "\\$\\{([^}]+)\\}|\\$([A-Za-z_][A-Za-z0-9_]*)"
        matches <- gregexpr(pattern, config, perl = TRUE)

        if (matches[[1]][1] != -1) {
          config <- gsub(
            "\\$\\{([^}]+)\\}",
            "\\1",
            config,
            perl = TRUE
          )
          # Get env var value
          env_var <- regmatches(config, regexpr("[A-Za-z_][A-Za-z0-9_]*", config))
          if (length(env_var) > 0) {
            config <- Sys.getenv(env_var, unset = config)
          }
        }
        return(config)
      }

      if (is.list(config)) {
        return(lapply(config, self$interpolate_env))
      }

      config
    },

    #' @description Apply environment-specific overrides
    apply_environment_overrides = function() {
      env_section <- sprintf("environments.%s", private$environment)

      if (self$has(env_section)) {
        overrides <- self$get(env_section)
        private$config <- self$merge_config(private$config, overrides)
      }

      invisible(self)
    },

    #' @description Merge configuration with overrides
    #' @param base Base configuration
    #' @param overrides Override values
    #' @return Merged configuration
    merge_config = function(base, overrides) {
      for (key in names(overrides)) {
        if (is.list(base[[key]]) && is.list(overrides[[key]])) {
          base[[key]] <- self$merge_config(base[[key]], overrides[[key]])
        } else {
          base[[key]] <- overrides[[key]]
        }
      }
      base
    },

    #' @description Get model configuration
    #' @param model_name Model name
    #' @return Model configuration list
    get_model_config = function(model_name) {
      self$get(sprintf("models.plugins.%s", model_name))
    },

    #' @description Get feature configuration
    #' @param feature_name Feature plugin name
    #' @return Feature configuration list
    get_feature_config = function(feature_name) {
      self$get(sprintf("features.plugins.%s", feature_name))
    },

    #' @description Get areas list
    #' @return Character vector of area codes
    get_areas = function() {
      areas <- self$get("regions.areas")
      if (is.list(areas)) {
        unlist(areas, use.names = FALSE)
      } else {
        areas
      }
    },

    #' @description Get storage backend configuration
    #' @return Storage configuration list
    get_storage_config = function() {
      self$get("storage")
    },

    #' @description Get environment
    #' @return Environment name
    get_environment = function() {
      private$environment
    },

    #' @description Get configuration path
    #' @return Path to loaded config file
    get_path = function() {
      private$config_path
    },

    #' @description Reload configuration
    reload = function() {
      if (is.null(private$config_path)) {
        stop("No configuration loaded to reload")
      }
      self$load(private$config_path)
    },

    #' @description Export configuration as list
    #' @return Configuration list
    as_list = function() {
      private$config
    },

    #' @description Print configuration summary
    print = function() {
      cat("ConfigManager\n")

      if (private$loaded) {
        cat(sprintf("  Path: %s\n", private$config_path))
        cat(sprintf("  Environment: %s\n", private$environment))
        cat(sprintf("  Project: %s\n", self$get("project.name")))
        cat(sprintf("  Sections: %s\n", paste(names(private$config), collapse = ", ")))
      } else {
        cat("  Status: Not loaded\n")
      }

      invisible(self)
    }
  )
)
```

### Configuration Schema

```yaml
# config/config.yaml

project:
  name: "PrevCargaONS"
  version: "1.0.0"

storage:
  backend: "s3"  # s3, local, azure
  bucket: "${S3_BUCKET}"
  prefix: "prevcarga"
  local_cache: true
  cache_dir: ".cache"

regions:
  national: "SIN"
  subsystems:
    SECO: [RJ, SP, MG, ES, MT, MS, AC, RO, DF, GO]
    S: [PR, SC, RS]
    NE: [ALPE, PBRN, BASE, CE, PI, BAOE]
    N: [AM, PA, MA, TO, RR, AP]
  areas:
    - RJ
    - SP
    - MG
    # ... all 23 areas

models:
  default: "lgbm"
  plugins:
    lgbm:
      enabled: true
      config:
        num_leaves: 31
        learning_rate: 0.05
        n_estimators: 100
    rf:
      enabled: true
      config:
        num.trees: 500
        mtry: 10
    hw:
      enabled: true
      config:
        seasonal: "additive"
        m: 48

features:
  plugins:
    calendar:
      enabled: true
    lags:
      enabled: true
      config:
        lags: [1, 2, 24, 48, 168]
    temperature:
      enabled: true

training:
  parallel:
    enabled: true
    n_jobs: 4
  validation:
    split_ratio: 0.2
    cv_folds: 5

prediction:
  horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  reconciliation: true
  combination: true

evaluation:
  metrics: [mape, mae, rmse]
  drift_detection:
    enabled: true
    threshold_mape: 1.2

logging:
  level: "INFO"
  format: "json"
  file: "logs/prevcarga.log"

# Environment-specific overrides
environments:
  development:
    storage:
      backend: "local"
      cache_dir: ".dev_cache"
    training:
      parallel:
        n_jobs: 2
    logging:
      level: "DEBUG"

  production:
    training:
      parallel:
        n_jobs: 8
    logging:
      level: "INFO"
      file: "/var/log/prevcarga/prevcarga.log"
```

### Convenience Functions

```r
#' Load global configuration
#'
#' @param path Path to configuration file
#' @param environment Environment name
#' @return ConfigManager instance
#' @export
load_config <- function(path = "config/config.yaml", environment = NULL) {
  config <- ConfigManager$new(environment = environment)
  config$load(path)
  config
}


#' Get global config instance
#'
#' @return ConfigManager instance or NULL
#' @export
get_config <- function() {
  .config_manager
}


#' Global config manager instance
#' @noRd
.config_manager <- NULL


#' Initialize global config
#' @noRd
init_config <- function(path, environment = NULL) {
  .config_manager <<- load_config(path, environment)
}
```

### Usage Example

```r
# Load configuration
config <- load_config("config/config.yaml", environment = "development")

# Access values
project_name <- config$get("project.name")
# "PrevCargaONS"

storage_backend <- config$get("storage.backend")
# "local" (from development override)

# Access with default
n_jobs <- config$get_with_default("training.parallel.n_jobs", 4)
# 2 (from development override)

# Check if key exists
has_drift <- config$has("evaluation.drift_detection.enabled")
# TRUE

# Get model config
lgbm_config <- config$get_model_config("lgbm")
# $enabled: TRUE
# $config: list(num_leaves = 31, ...)

# Get all areas
areas <- config$get_areas()
# c("RJ", "SP", "MG", ...)

# Get keys at level
model_names <- config$keys("models.plugins")
# c("lgbm", "rf", "hw")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | load() valid YAML | Config loaded |
| TC-002 | load() missing file | Error thrown |
| TC-003 | validate() missing sections | Error with list |
| TC-004 | get() simple key | Value returned |
| TC-005 | get() nested key | Nested value |
| TC-006 | get() missing key | NULL returned |
| TC-007 | get_with_default() | Default used |
| TC-008 | has() existing key | TRUE |
| TC-009 | has() missing key | FALSE |
| TC-010 | interpolate_env() | Env vars replaced |
| TC-011 | Environment overrides | Overrides applied |
| TC-012 | reload() | Config refreshed |

---

## Dependencies

None (foundation module)

---

## Definition of Done

- [ ] ConfigManager R6 class implemented
- [ ] YAML loading and validation
- [ ] Nested key access working
- [ ] Environment variable interpolation
- [ ] Environment-specific overrides
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Environment variables use ${VAR} syntax
- Environment overrides allow per-env configuration
- Consider adding config schema validation with jsonvalidate
- May add config file watching for hot-reload
