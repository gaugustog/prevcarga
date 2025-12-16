# PC-040-06: ReconcilerRegistry

**Epic:** [EPIC-06: Reconciliation Infrastructure](../epics/EPIC-06-reconciliation-layer.md)
**Task Reference:** T-06.2
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Implement the `ReconcilerRegistry` R6 class for managing and instantiating hierarchical reconciliation strategies, following the same registry pattern used for models, features, and combiners.

---

## Acceptance Criteria

- [ ] Registry module created in `R/reconciliation/registry.R`
- [ ] `ReconcilerRegistry` R6 class with CRUD operations
- [ ] Global registry instance `.reconciler_registry`
- [ ] Convenience function `get_reconciler()`
- [ ] Support for reconciler discovery and listing
- [ ] Validation that registered classes inherit from BaseReconciler

---

## Technical Specification

### File Location
```
R/reconciliation/registry.R
```

### ReconcilerRegistry R6 Class

```r
#' @title ReconcilerRegistry
#' @description Registry for hierarchical reconciliation strategies
#'
#' Manages registration, discovery, and instantiation of reconciler plugins.
#' Follows the same pattern as ModelRegistry, FeaturePluginRegistry, and CombinerRegistry.
#'
#' @export
ReconcilerRegistry <- R6::R6Class(
  "ReconcilerRegistry",
  private = list(
    reconcilers = NULL,
    metadata = NULL
  ),
  public = list(
    #' @description Initialize registry
    initialize = function() {
      private$reconcilers <- list()
      private$metadata <- list()
    },

    #' @description Register a reconciler class
    #' @param name Reconciler name
    #' @param reconciler_class R6 class generator (not instance)
    #' @param description Optional description
    #' @param tags Optional character vector of tags
    #' @return Invisible self
    register = function(name, reconciler_class, description = NULL, tags = NULL) {
      checkmate::assert_string(name, min.chars = 1)
      checkmate::assert_class(reconciler_class, "R6ClassGenerator")

      # Validate it's a BaseReconciler subclass
      test_instance <- tryCatch(
        reconciler_class$new(),
        error = function(e) {
          stop(sprintf(
            "Failed to instantiate reconciler '%s': %s",
            name, conditionMessage(e)
          ))
        }
      )

      if (!inherits(test_instance, "BaseReconciler")) {
        stop(sprintf(
          "Reconciler '%s' must inherit from BaseReconciler",
          name
        ))
      }

      # Check for duplicate registration
      if (name %in% names(private$reconcilers)) {
        warning(sprintf("Overwriting existing reconciler: %s", name))
      }

      private$reconcilers[[name]] <- reconciler_class
      private$metadata[[name]] <- list(
        description = description,
        tags = tags,
        registered_at = Sys.time(),
        class_name = class(test_instance)[1]
      )

      message(sprintf("Registered reconciler: %s", name))
      invisible(self)
    },

    #' @description Get a reconciler instance
    #' @param name Reconciler name
    #' @param config Optional configuration
    #' @return BaseReconciler instance
    get = function(name, config = list()) {
      checkmate::assert_string(name)

      if (!self$has(name)) {
        available <- paste(self$list_reconcilers(), collapse = ", ")
        stop(sprintf(
          "Reconciler '%s' not registered. Available: %s",
          name, available
        ))
      }

      private$reconcilers[[name]]$new(name = name, config = config)
    },

    #' @description Check if reconciler is registered
    #' @param name Reconciler name
    #' @return Logical
    has = function(name) {
      name %in% names(private$reconcilers)
    },

    #' @description List all registered reconcilers
    #' @return Character vector of reconciler names
    list_reconcilers = function() {
      names(private$reconcilers)
    },

    #' @description Get reconciler metadata
    #' @param name Reconciler name (NULL for all)
    #' @return List or data.table of metadata
    get_metadata = function(name = NULL) {
      if (is.null(name)) {
        if (length(private$metadata) == 0) {
          return(data.table::data.table())
        }

        data.table::rbindlist(lapply(names(private$metadata), function(n) {
          meta <- private$metadata[[n]]
          data.table::data.table(
            name = n,
            class_name = meta$class_name,
            description = meta$description %||% NA_character_,
            tags = paste(meta$tags, collapse = ", "),
            registered_at = meta$registered_at
          )
        }))
      } else {
        private$metadata[[name]]
      }
    },

    #' @description Find reconcilers by tag
    #' @param tag Tag to search for
    #' @return Character vector of matching reconciler names
    find_by_tag = function(tag) {
      checkmate::assert_string(tag)

      matching <- sapply(names(private$metadata), function(name) {
        tag %in% private$metadata[[name]]$tags
      })

      names(private$metadata)[matching]
    },

    #' @description Unregister a reconciler
    #' @param name Reconciler name
    #' @return Invisible self
    unregister = function(name) {
      if (!self$has(name)) {
        warning(sprintf("Reconciler '%s' not registered", name))
        return(invisible(self))
      }

      private$reconcilers[[name]] <- NULL
      private$metadata[[name]] <- NULL

      message(sprintf("Unregistered reconciler: %s", name))
      invisible(self)
    },

    #' @description Clear all registrations
    #' @return Invisible self
    clear = function() {
      private$reconcilers <- list()
      private$metadata <- list()
      message("Cleared all reconciler registrations")
      invisible(self)
    },

    #' @description Get number of registered reconcilers
    #' @return Integer
    size = function() {
      length(private$reconcilers)
    },

    #' @description Print registry summary
    print = function() {
      cat(sprintf("ReconcilerRegistry with %d reconcilers:\n", self$size()))

      if (self$size() > 0) {
        for (name in self$list_reconcilers()) {
          meta <- private$metadata[[name]]
          desc <- if (!is.null(meta$description)) {
            sprintf(" - %s", meta$description)
          } else ""
          cat(sprintf("  - %s%s\n", name, desc))
        }
      }

      invisible(self)
    }
  )
)
```

### Global Registry and Convenience Functions

```r
#' Global reconciler registry instance
#' @export
.reconciler_registry <- NULL

#' Get the global reconciler registry
#'
#' @return ReconcilerRegistry instance
#' @export
reconciler_registry <- function() {
  if (is.null(.reconciler_registry)) {
    .reconciler_registry <<- ReconcilerRegistry$new()
  }
  .reconciler_registry
}


#' Register a reconciler in the global registry
#'
#' @param name Reconciler name
#' @param reconciler_class R6 class generator
#' @param description Optional description
#' @param tags Optional tags
#' @return Invisible registry
#' @export
register_reconciler <- function(name, reconciler_class,
                                description = NULL, tags = NULL) {
  reconciler_registry()$register(
    name = name,
    reconciler_class = reconciler_class,
    description = description,
    tags = tags
  )
}


#' Get a reconciler from the global registry
#'
#' @param name Reconciler name
#' @param config Optional configuration
#' @return BaseReconciler instance
#' @export
#' @examples
#' reconciler <- get_reconciler("bottom_up")
#' reconciler <- get_reconciler("mint", config = list(
#'   covariance = "shrink"
#' ))
get_reconciler <- function(name, config = list()) {
  reconciler_registry()$get(name = name, config = config)
}


#' List available reconcilers
#'
#' @param tag Optional tag filter
#' @return Character vector or data.table
#' @export
list_reconcilers <- function(tag = NULL) {
  if (is.null(tag)) {
    reconciler_registry()$list_reconcilers()
  } else {
    reconciler_registry()$find_by_tag(tag)
  }
}


#' Check if reconciler exists
#'
#' @param name Reconciler name
#' @return Logical
#' @export
has_reconciler <- function(name) {
  reconciler_registry()$has(name)
}
```

### Auto-Registration on Package Load

```r
#' Register built-in reconcilers
#'
#' Called during package load to register default reconcilers.
#' @noRd
.register_builtin_reconcilers <- function() {
  # Bottom-up (always available, simplest)
  if (!has_reconciler("bottom_up")) {
    register_reconciler(
      "bottom_up",
      BottomUpReconciler,
      description = "Aggregates bottom-level forecasts",
      tags = c("simple", "default")
    )
  }

  # Top-down (simple proportions)
  if (!has_reconciler("top_down")) {
    register_reconciler(
      "top_down",
      TopDownReconciler,
      description = "Disaggregates top-level by historical proportions",
      tags = c("simple", "proportional")
    )
  }
}


#' Package load hook
#' @noRd
.onLoad <- function(libname, pkgname) {
  # Initialize registry
  .reconciler_registry <<- ReconcilerRegistry$new()

  # Register built-in reconcilers
  .register_builtin_reconcilers()
}
```

### Usage Example

```r
# Register a custom reconciler
register_reconciler(
  "mint_shrink",
  MinTReconciler,
  description = "MinT with shrinkage covariance",
  tags = c("optimal", "mint")
)

# List available reconcilers
list_reconcilers()
# [1] "bottom_up" "top_down" "mint_shrink"

# Get reconciler instance
reconciler <- get_reconciler("bottom_up")

# Get with configuration
mint_reconciler <- get_reconciler("mint_shrink", config = list(
  covariance = "shrink",
  lambda = 0.5
))

# Find reconcilers by tag
list_reconcilers(tag = "optimal")
# [1] "mint_shrink"

# Get metadata
reconciler_registry()$get_metadata()
#          name       class_name                            description
# 1:  bottom_up BottomUpReconciler Aggregates bottom-level forecasts
# 2:   top_down  TopDownReconciler Disaggregates top-level by historic...

# Check existence
has_reconciler("unknown")
# [1] FALSE
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize registry | Empty registry |
| TC-002 | register() valid reconciler | Reconciler registered |
| TC-003 | register() non-BaseReconciler | Error thrown |
| TC-004 | register() duplicate | Warning, overwrite |
| TC-005 | get() existing | Returns instance |
| TC-006 | get() missing | Error with list |
| TC-007 | has() existing | TRUE |
| TC-008 | has() missing | FALSE |
| TC-009 | list_reconcilers() | All names |
| TC-010 | get_metadata() all | data.table |
| TC-011 | find_by_tag() | Matching reconcilers |
| TC-012 | unregister() | Reconciler removed |
| TC-013 | Global registry functions | Work correctly |

---

## Dependencies

- PC-039-06: BaseReconciler

---

## Definition of Done

- [ ] ReconcilerRegistry R6 class implemented
- [ ] Global registry instance
- [ ] Convenience functions (get_reconciler, etc.)
- [ ] Tag-based discovery
- [ ] Auto-registration on package load
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Registry pattern consistent with other registries
- Tags enable discovery by reconciliation type
- Built-in reconcilers registered on package load
- Consider adding reconciler comparison utilities in future
