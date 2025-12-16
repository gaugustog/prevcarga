# PC-044-06: Hierarchy Configuration Schema

**Epic:** [EPIC-06: Reconciliation Infrastructure](../epics/EPIC-06-reconciliation-layer.md)
**Task Reference:** T-06.6
**Priority:** Medium
**Estimated Effort:** 0.5 days

---

## Summary

Define the YAML configuration schema for hierarchical reconciliation, including hierarchy structure, reconciliation strategy, and loss area settings.

---

## Acceptance Criteria

- [ ] Configuration module created in `R/reconciliation/config.R`
- [ ] YAML schema for hierarchy and reconciliation
- [ ] `ReconciliationConfig` R6 class for config management
- [ ] Configuration validation
- [ ] Default SIN configuration
- [ ] Support for custom hierarchies

---

## Technical Specification

### File Location
```
R/reconciliation/config.R
```

### YAML Configuration Schema

```yaml
# config/reconciliation.yaml

reconciliation:
  # Reconciliation strategy: bottom_up, top_down, optimal, mint, ols
  strategy: bottom_up

  # Strategy-specific configuration
  strategy_config:
    # For optimal/mint/ols methods
    covariance: shrink  # diag, sample, shrink
    lambda: null        # Shrinkage parameter (null = auto)
    nonnegative: true   # Enforce non-negative forecasts

  # Hierarchy definition
  hierarchy:
    # Top level
    national: SIN

    # Subsystems with their areas
    subsystems:
      SECO:
        - RJ
        - SP
        - MG
        - ES
        - MT
        - MS
        - AC
        - RO
        - DF
        - GO
      S:
        - PR
        - SC
        - RS
      NE:
        - ALPE
        - PBRN
        - BASE
        - CE
        - PI
        - BAOE
      N:
        - AM
        - PA
        - MA
        - TO
        - RR
        - AP

    # Loss areas configuration
    loss_areas:
      PESE:
        subsystem: SECO
        method: by_difference
      PES:
        subsystem: S
        method: by_difference
      PENE:
        subsystem: NE
        method: model  # Uses model forecast, not calculated
      PEN:
        subsystem: N
        method: by_difference

  # Loss calculation settings
  loss_config:
    clamp_negative: true
    min_loss_threshold: 0
    warn_on_negative: true

  # Coherence validation settings
  coherence:
    validate: true
    strict: false          # true = error, false = warning
    tolerance: 1e-6        # Tolerance for numerical comparison

  # Workflow settings
  workflow:
    calculate_losses_first: true  # Before or after reconciliation
    validate_input: true
    validate_output: true
```

### ReconciliationConfig R6 Class

```r
#' @title ReconciliationConfig
#' @description Configuration management for reconciliation
#' @export
ReconciliationConfig <- R6::R6Class(
  "ReconciliationConfig",
  private = list(
    config = NULL,
    validated = FALSE
  ),
  public = list(
    #' @description Initialize configuration
    #' @param config Configuration list or path to YAML file
    initialize = function(config = NULL) {
      if (is.null(config)) {
        private$config <- private$default_config()
      } else if (is.character(config)) {
        private$config <- yaml::read_yaml(config)$reconciliation
      } else {
        private$config <- config
      }

      self$validate()
    },

    #' @description Validate configuration
    #' @return TRUE if valid, error otherwise
    validate = function() {
      config <- private$config

      # Validate strategy
      valid_strategies <- c("bottom_up", "top_down", "optimal", "mint", "ols", "wls")
      if (!config$strategy %in% valid_strategies) {
        stop(sprintf(
          "Invalid strategy '%s'. Must be one of: %s",
          config$strategy, paste(valid_strategies, collapse = ", ")
        ))
      }

      # Validate hierarchy
      if (is.null(config$hierarchy$national)) {
        stop("Hierarchy must have 'national' field")
      }

      if (is.null(config$hierarchy$subsystems) ||
          length(config$hierarchy$subsystems) == 0) {
        stop("Hierarchy must have 'subsystems' field with at least one subsystem")
      }

      # Validate each subsystem has areas
      for (subsystem in names(config$hierarchy$subsystems)) {
        areas <- config$hierarchy$subsystems[[subsystem]]
        if (length(areas) == 0) {
          stop(sprintf("Subsystem '%s' must have at least one area", subsystem))
        }
      }

      # Validate loss areas reference valid subsystems
      for (la in names(config$hierarchy$loss_areas)) {
        la_config <- config$hierarchy$loss_areas[[la]]

        if (!la_config$subsystem %in% names(config$hierarchy$subsystems)) {
          stop(sprintf(
            "Loss area '%s' references unknown subsystem '%s'",
            la, la_config$subsystem
          ))
        }

        if (!la_config$method %in% c("by_difference", "model")) {
          stop(sprintf(
            "Loss area '%s' has invalid method '%s'",
            la, la_config$method
          ))
        }
      }

      # Check for duplicate area codes
      all_areas <- c(
        unlist(config$hierarchy$subsystems, use.names = FALSE),
        names(config$hierarchy$loss_areas)
      )
      dupes <- all_areas[duplicated(all_areas)]
      if (length(dupes) > 0) {
        stop(sprintf("Duplicate area codes: %s", paste(dupes, collapse = ", ")))
      }

      private$validated <- TRUE
      invisible(TRUE)
    },

    #' @description Get full configuration
    #' @return Configuration list
    get = function() {
      private$config
    },

    #' @description Get strategy
    #' @return Strategy name
    get_strategy = function() {
      private$config$strategy
    },

    #' @description Get strategy configuration
    #' @return Strategy config list
    get_strategy_config = function() {
      private$config$strategy_config %||% list()
    },

    #' @description Get hierarchy configuration
    #' @return Hierarchy config list
    get_hierarchy = function() {
      private$config$hierarchy
    },

    #' @description Get loss configuration
    #' @return Loss config list
    get_loss_config = function() {
      private$config$loss_config %||% list()
    },

    #' @description Get coherence configuration
    #' @return Coherence config list
    get_coherence_config = function() {
      private$config$coherence %||% list(validate = TRUE, strict = FALSE)
    },

    #' @description Get workflow configuration
    #' @return Workflow config list
    get_workflow_config = function() {
      private$config$workflow %||% list()
    },

    #' @description Create HierarchyBuilder from config
    #' @return HierarchyBuilder instance
    create_hierarchy_builder = function() {
      HierarchyBuilder$new(config = self$get_hierarchy())
    },

    #' @description Create LossCalculator from config
    #' @return LossCalculator instance
    create_loss_calculator = function() {
      LossCalculator$new(config = self$get_loss_config())
    },

    #' @description Save configuration to YAML
    #' @param path Output file path
    save = function(path) {
      yaml::write_yaml(
        list(reconciliation = private$config),
        file = path
      )
      message(sprintf("Configuration saved to: %s", path))
      invisible(self)
    },

    #' @description Print configuration summary
    print = function() {
      config <- private$config

      cat("ReconciliationConfig\n")
      cat(sprintf("  Strategy: %s\n", config$strategy))
      cat(sprintf("  National: %s\n", config$hierarchy$national))
      cat(sprintf("  Subsystems: %d\n", length(config$hierarchy$subsystems)))

      for (subsystem in names(config$hierarchy$subsystems)) {
        n_areas <- length(config$hierarchy$subsystems[[subsystem]])
        cat(sprintf("    %s: %d areas\n", subsystem, n_areas))
      }

      cat(sprintf("  Loss areas: %d\n", length(config$hierarchy$loss_areas)))
      cat(sprintf("  Validated: %s\n", private$validated))

      invisible(self)
    }
  ),

  private = list(
    default_config = function() {
      list(
        strategy = "bottom_up",
        strategy_config = list(
          covariance = "shrink",
          lambda = NULL,
          nonnegative = TRUE
        ),
        hierarchy = list(
          national = "SIN",
          subsystems = list(
            SECO = c("RJ", "SP", "MG", "ES", "MT", "MS", "AC", "RO", "DF", "GO"),
            S = c("PR", "SC", "RS"),
            NE = c("ALPE", "PBRN", "BASE", "CE", "PI", "BAOE"),
            N = c("AM", "PA", "MA", "TO", "RR", "AP")
          ),
          loss_areas = list(
            PESE = list(subsystem = "SECO", method = "by_difference"),
            PES = list(subsystem = "S", method = "by_difference"),
            PENE = list(subsystem = "NE", method = "model"),
            PEN = list(subsystem = "N", method = "by_difference")
          )
        ),
        loss_config = list(
          clamp_negative = TRUE,
          min_loss_threshold = 0,
          warn_on_negative = TRUE
        ),
        coherence = list(
          validate = TRUE,
          strict = FALSE,
          tolerance = 1e-6
        ),
        workflow = list(
          calculate_losses_first = TRUE,
          validate_input = TRUE,
          validate_output = TRUE
        )
      )
    }
  )
)
```

### Convenience Functions

```r
#' Load reconciliation configuration
#'
#' @param path Path to YAML file (NULL for defaults)
#' @return ReconciliationConfig instance
#' @export
load_reconciliation_config <- function(path = NULL) {
  ReconciliationConfig$new(config = path)
}


#' Get default SIN hierarchy configuration
#'
#' @return List with SIN hierarchy structure
#' @export
get_sin_hierarchy_config <- function() {
  config <- ReconciliationConfig$new()
  config$get_hierarchy()
}


#' Validate reconciliation configuration file
#'
#' @param path Path to YAML file
#' @return TRUE if valid, error otherwise
#' @export
validate_reconciliation_config <- function(path) {
  config <- ReconciliationConfig$new(config = path)
  message("Configuration is valid")
  invisible(TRUE)
}
```

### Usage Example

```r
# Load default configuration
config <- load_reconciliation_config()

print(config)
# ReconciliationConfig
#   Strategy: bottom_up
#   National: SIN
#   Subsystems: 4
#     SECO: 10 areas
#     S: 3 areas
#     NE: 6 areas
#     N: 6 areas
#   Loss areas: 4
#   Validated: TRUE

# Load from YAML file
config <- load_reconciliation_config("config/reconciliation.yaml")

# Get specific sections
strategy <- config$get_strategy()
hierarchy <- config$get_hierarchy()

# Create components from config
hierarchy_builder <- config$create_hierarchy_builder()
loss_calculator <- config$create_loss_calculator()

# Save configuration
config$save("config/my_reconciliation.yaml")

# Custom configuration
custom_config <- list(
  strategy = "mint",
  strategy_config = list(covariance = "shrink"),
  hierarchy = list(
    national = "TOTAL",
    subsystems = list(
      REGION_A = c("A1", "A2", "A3"),
      REGION_B = c("B1", "B2")
    ),
    loss_areas = list(
      LOSS_A = list(subsystem = "REGION_A", method = "by_difference")
    )
  )
)
config <- ReconciliationConfig$new(config = custom_config)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | Valid SIN config |
| TC-002 | Initialize from YAML | Config loaded |
| TC-003 | Initialize with list | Config applied |
| TC-004 | validate() valid | TRUE |
| TC-005 | validate() invalid strategy | Error |
| TC-006 | validate() missing national | Error |
| TC-007 | validate() missing subsystems | Error |
| TC-008 | validate() empty subsystem | Error |
| TC-009 | validate() invalid loss subsystem | Error |
| TC-010 | validate() duplicate areas | Error |
| TC-011 | get_hierarchy() | Hierarchy config |
| TC-012 | create_hierarchy_builder() | HierarchyBuilder |
| TC-013 | create_loss_calculator() | LossCalculator |
| TC-014 | save() | YAML file created |

---

## Dependencies

None (configuration foundation)

---

## Definition of Done

- [ ] ReconciliationConfig R6 class implemented
- [ ] YAML schema defined
- [ ] Validation working
- [ ] Factory functions implemented
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Default config matches Brazilian SIN hierarchy
- Support custom hierarchies for testing/other grids
- Strategy config varies by reconciliation method
- Consider adding config versioning in future
