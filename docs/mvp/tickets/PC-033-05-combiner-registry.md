# PC-033-05: CombinerRegistry

**Epic:** [EPIC-05: Combination Infrastructure](../epics/EPIC-05-combination-layer.md)
**Task Reference:** T-05.2
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Implement the `CombinerRegistry` R6 class for managing and instantiating forecast combination strategies, following the same registry pattern used for models and features.

---

## Acceptance Criteria

- [ ] Registry module created in `R/combination/registry.R`
- [ ] `CombinerRegistry` R6 class with CRUD operations
- [ ] Global registry instance `.combiner_registry`
- [ ] Convenience function `get_combiner()`
- [ ] Support for combiner discovery and listing
- [ ] Validation that registered classes inherit from BaseCombiner

---

## Technical Specification

### File Location
```
R/combination/registry.R
```

### CombinerRegistry R6 Class

```r
#' @title CombinerRegistry
#' @description Registry for forecast combination strategies
#'
#' Manages registration, discovery, and instantiation of combiner plugins.
#' Follows the same pattern as ModelRegistry and FeaturePluginRegistry.
#'
#' @export
CombinerRegistry <- R6::R6Class(
  "CombinerRegistry",
  private = list(
    combiners = NULL,
    metadata = NULL
  ),
  public = list(
    #' @description Initialize registry
    initialize = function() {
      private$combiners <- list()
      private$metadata <- list()
    },

    #' @description Register a combiner class
    #' @param name Combiner name
    #' @param combiner_class R6 class generator (not instance)
    #' @param description Optional description
    #' @param tags Optional character vector of tags
    #' @return Invisible self
    register = function(name, combiner_class, description = NULL, tags = NULL) {
      checkmate::assert_string(name, min.chars = 1)
      checkmate::assert_class(combiner_class, "R6ClassGenerator")

      # Validate it's a BaseCombiner subclass
      test_instance <- tryCatch(
        combiner_class$new(),
        error = function(e) {
          stop(sprintf(
            "Failed to instantiate combiner '%s': %s",
            name, conditionMessage(e)
          ))
        }
      )

      if (!inherits(test_instance, "BaseCombiner")) {
        stop(sprintf(
          "Combiner '%s' must inherit from BaseCombiner",
          name
        ))
      }

      # Check for duplicate registration
      if (name %in% names(private$combiners)) {
        warning(sprintf("Overwriting existing combiner: %s", name))
      }

      private$combiners[[name]] <- combiner_class
      private$metadata[[name]] <- list(
        description = description,
        tags = tags,
        registered_at = Sys.time(),
        class_name = class(test_instance)[1]
      )

      message(sprintf("Registered combiner: %s", name))
      invisible(self)
    },

    #' @description Get a combiner instance
    #' @param name Combiner name
    #' @param config Optional configuration
    #' @return BaseCombiner instance
    get = function(name, config = list()) {
      checkmate::assert_string(name)

      if (!self$has(name)) {
        available <- paste(self$list_combiners(), collapse = ", ")
        stop(sprintf(
          "Combiner '%s' not registered. Available: %s",
          name, available
        ))
      }

      private$combiners[[name]]$new(name = name, config = config)
    },

    #' @description Check if combiner is registered
    #' @param name Combiner name
    #' @return Logical
    has = function(name) {
      name %in% names(private$combiners)
    },

    #' @description List all registered combiners
    #' @return Character vector of combiner names
    list_combiners = function() {
      names(private$combiners)
    },

    #' @description Get combiner metadata
    #' @param name Combiner name (NULL for all)
    #' @return List or data.table of metadata
    get_metadata = function(name = NULL) {
      if (is.null(name)) {
        # Return all metadata as data.table
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

    #' @description Find combiners by tag
    #' @param tag Tag to search for
    #' @return Character vector of matching combiner names
    find_by_tag = function(tag) {
      checkmate::assert_string(tag)

      matching <- sapply(names(private$metadata), function(name) {
        tag %in% private$metadata[[name]]$tags
      })

      names(private$metadata)[matching]
    },

    #' @description Unregister a combiner
    #' @param name Combiner name
    #' @return Invisible self
    unregister = function(name) {
      if (!self$has(name)) {
        warning(sprintf("Combiner '%s' not registered", name))
        return(invisible(self))
      }

      private$combiners[[name]] <- NULL
      private$metadata[[name]] <- NULL

      message(sprintf("Unregistered combiner: %s", name))
      invisible(self)
    },

    #' @description Clear all registrations
    #' @return Invisible self
    clear = function() {
      private$combiners <- list()
      private$metadata <- list()
      message("Cleared all combiner registrations")
      invisible(self)
    },

    #' @description Get number of registered combiners
    #' @return Integer
    size = function() {
      length(private$combiners)
    },

    #' @description Print registry summary
    print = function() {
      cat(sprintf("CombinerRegistry with %d combiners:\n", self$size()))

      if (self$size() > 0) {
        for (name in self$list_combiners()) {
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
#' Global combiner registry instance
#' @export
.combiner_registry <- NULL

#' Get the global combiner registry
#'
#' @return CombinerRegistry instance
#' @export
combiner_registry <- function() {
  if (is.null(.combiner_registry)) {
    .combiner_registry <<- CombinerRegistry$new()
  }
  .combiner_registry
}


#' Register a combiner in the global registry
#'
#' @param name Combiner name
#' @param combiner_class R6 class generator
#' @param description Optional description
#' @param tags Optional tags
#' @return Invisible registry
#' @export
register_combiner <- function(name, combiner_class,
                              description = NULL, tags = NULL) {
  combiner_registry()$register(
    name = name,
    combiner_class = combiner_class,
    description = description,
    tags = tags
  )
}


#' Get a combiner from the global registry
#'
#' @param name Combiner name
#' @param config Optional configuration
#' @return BaseCombiner instance
#' @export
#' @examples
#' combiner <- get_combiner("weighted_average", config = list(
#'   auto_normalize = TRUE
#' ))
get_combiner <- function(name, config = list()) {
  combiner_registry()$get(name = name, config = config)
}


#' List available combiners
#'
#' @param tag Optional tag filter
#' @return Character vector or data.table
#' @export
list_combiners <- function(tag = NULL) {
  if (is.null(tag)) {
    combiner_registry()$list_combiners()
  } else {
    combiner_registry()$find_by_tag(tag)
  }
}


#' Check if combiner exists
#'
#' @param name Combiner name
#' @return Logical
#' @export
has_combiner <- function(name) {
  combiner_registry()$has(name)
}
```

### Auto-Registration on Package Load

```r
#' Register built-in combiners
#'
#' Called during package load to register default combiners.
#' @noRd
.register_builtin_combiners <- function() {
  # Simple average (always available)
  if (!has_combiner("simple_average")) {
    register_combiner(
      "simple_average",
      SimpleAverageCombiner,
      description = "Equal-weight average of all models",
      tags = c("simple", "default")
    )
  }

  # Weighted average (always available)
  if (!has_combiner("weighted_average")) {
    register_combiner(
      "weighted_average",
      WeightedAverageCombiner,
      description = "Weighted average with custom weights",
      tags = c("weighted", "default")
    )
  }
}


#' Package load hook
#' @noRd
.onLoad <- function(libname, pkgname) {
  # Initialize registry
  .combiner_registry <<- CombinerRegistry$new()

  # Register built-in combiners
  .register_builtin_combiners()
}
```

### Usage Example

```r
# Register a custom combiner
register_combiner(
  "my_combiner",
  MyCustomCombiner,
  description = "My custom combination strategy",
  tags = c("custom", "experimental")
)

# List available combiners
list_combiners()
# [1] "simple_average" "weighted_average" "my_combiner"

# Get combiner instance
combiner <- get_combiner("weighted_average", config = list(
  auto_normalize = TRUE
))

# Find combiners by tag
list_combiners(tag = "default")
# [1] "simple_average" "weighted_average"

# Get metadata
combiner_registry()$get_metadata()
#              name        class_name                      description
# 1: simple_average SimpleAverageCombiner Equal-weight average of all...
# 2: weighted_average WeightedAverageCombiner Weighted average with custom...

# Check existence
has_combiner("unknown")
# [1] FALSE

# Unregister
combiner_registry()$unregister("my_combiner")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize registry | Empty registry |
| TC-002 | register() valid combiner | Combiner registered |
| TC-003 | register() non-BaseCombiner | Error thrown |
| TC-004 | register() duplicate | Warning, overwrite |
| TC-005 | get() existing | Returns instance |
| TC-006 | get() missing | Error with list |
| TC-007 | has() existing | TRUE |
| TC-008 | has() missing | FALSE |
| TC-009 | list_combiners() | All names |
| TC-010 | get_metadata() all | data.table |
| TC-011 | get_metadata() specific | List |
| TC-012 | find_by_tag() | Matching combiners |
| TC-013 | unregister() | Combiner removed |
| TC-014 | clear() | Registry empty |
| TC-015 | Global registry functions | Work correctly |

---

## Dependencies

- PC-032-05: BaseCombiner

---

## Definition of Done

- [ ] CombinerRegistry R6 class implemented
- [ ] Global registry instance
- [ ] Convenience functions (get_combiner, etc.)
- [ ] Tag-based discovery
- [ ] Auto-registration on package load
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Registry pattern matches ModelRegistry and FeaturePluginRegistry
- Tags enable discovery by category
- Built-in combiners are registered on package load
- Consider adding combiner versioning in future
- Metadata helps users discover appropriate combiners
