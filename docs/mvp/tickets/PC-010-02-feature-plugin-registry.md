# PC-010-02: FeaturePluginRegistry

**Epic:** [EPIC-02: Feature Engineering Infrastructure](../epics/EPIC-02-feature-engineering.md)
**Task Reference:** T-02.2
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Implement the `FeaturePluginRegistry` R6 class that manages registration, discovery, and instantiation of feature plugins using the registry pattern.

---

## Acceptance Criteria

- [ ] Registry module created in `R/features/registry.R`
- [ ] `FeaturePluginRegistry` R6 class with registration methods
- [ ] `register()` validates plugin class inherits from BaseFeaturePlugin
- [ ] `get()` instantiates plugins with configuration
- [ ] `list_plugins()` returns registered plugin names
- [ ] `has()` checks if plugin is registered
- [ ] Global registry instance created (`.feature_registry`)
- [ ] Export convenience functions for plugin contributors

---

## Technical Specification

### File Location
```
R/features/registry.R
```

### Registry Class

```r
#' @title FeaturePluginRegistry
#' @description Registry for feature plugin classes
#'
#' The registry maintains a collection of feature plugin classes that
#' can be instantiated on demand. Plugins are registered by name and
#' can be retrieved with optional configuration.
#'
#' @export
FeaturePluginRegistry <- R6::R6Class(
 "FeaturePluginRegistry",
 private = list(
   plugins = NULL,
   metadata = NULL
 ),
 public = list(
   #' @description Initialize registry
   initialize = function() {
     private$plugins <- list()
     private$metadata <- list()
   },

   #' @description Register a plugin class
   #' @param name Unique plugin name
   #' @param plugin_class R6 class generator (not instance)
   #' @param description Optional description
   #' @param version Optional version string
   #' @param author Optional author name
   #' @return Invisible self (for chaining)
   register = function(name, plugin_class,
                       description = NULL,
                       version = "1.0.0",
                       author = NULL) {
     checkmate::assert_string(name, min.chars = 1)

     # Validate that plugin_class is an R6 class generator
     if (!inherits(plugin_class, "R6ClassGenerator")) {
       stop(sprintf(
         "plugin_class must be an R6 class generator, got %s",
         class(plugin_class)[1]
       ))
     }

     # Validate that instances inherit from BaseFeaturePlugin
     test_instance <- tryCatch(
       plugin_class$new(),
       error = function(e) {
         stop(sprintf(
           "Failed to instantiate plugin '%s': %s",
           name, e$message
         ))
       }
     )

     if (!inherits(test_instance, "BaseFeaturePlugin")) {
       stop(sprintf(
         "Plugin '%s' must inherit from BaseFeaturePlugin",
         name
       ))
     }

     # Warn if overwriting existing registration
     if (name %in% names(private$plugins)) {
       warning(sprintf("Overwriting existing plugin registration: '%s'", name))
     }

     # Store plugin class and metadata
     private$plugins[[name]] <- plugin_class
     private$metadata[[name]] <- list(
       description = description,
       version = version,
       author = author,
       registered_at = Sys.time()
     )

     message(sprintf("Registered feature plugin: '%s'", name))
     invisible(self)
   },

   #' @description Get plugin instance
   #' @param name Plugin name
   #' @param config Configuration list for plugin
   #' @return Plugin instance
   get = function(name, config = list()) {
     checkmate::assert_string(name)
     checkmate::assert_list(config)

     if (!self$has(name)) {
       available <- paste(self$list_plugins(), collapse = ", ")
       stop(sprintf(
         "Plugin '%s' not registered. Available: %s",
         name,
         if (nchar(available) > 0) available else "(none)"
       ))
     }

     private$plugins[[name]]$new(name = name, config = config)
   },

   #' @description Check if plugin is registered
   #' @param name Plugin name
   #' @return Logical
   has = function(name) {
     name %in% names(private$plugins)
   },

   #' @description List registered plugin names
   #' @return Character vector
   list_plugins = function() {
     names(private$plugins)
   },

   #' @description Get plugin count
   #' @return Integer
   count = function() {
     length(private$plugins)
   },

   #' @description Get plugin metadata
   #' @param name Plugin name
   #' @return List with metadata or NULL
   get_metadata = function(name) {
     if (!self$has(name)) {
       return(NULL)
     }
     private$metadata[[name]]
   },

   #' @description Unregister a plugin
   #' @param name Plugin name
   #' @return Invisible self
   unregister = function(name) {
     if (!self$has(name)) {
       warning(sprintf("Plugin '%s' not registered, nothing to remove", name))
       return(invisible(self))
     }

     private$plugins[[name]] <- NULL
     private$metadata[[name]] <- NULL
     message(sprintf("Unregistered plugin: '%s'", name))
     invisible(self)
   },

   #' @description Clear all registrations
   #' @return Invisible self
   clear = function() {
     private$plugins <- list()
     private$metadata <- list()
     message("Cleared all plugin registrations")
     invisible(self)
   },

   #' @description Get summary of all plugins
   #' @return data.table with plugin information
   summary = function() {
     if (self$count() == 0) {
       return(data.table::data.table(
         name = character(),
         version = character(),
         description = character(),
         author = character()
       ))
     }

     data.table::rbindlist(lapply(names(private$plugins), function(name) {
       meta <- private$metadata[[name]]
       data.table::data.table(
         name = name,
         version = meta$version %||% NA_character_,
         description = meta$description %||% NA_character_,
         author = meta$author %||% NA_character_
       )
     }))
   },

   #' @description Print registry summary
   print = function() {
     cat(sprintf("<FeaturePluginRegistry: %d plugins>\n", self$count()))
     if (self$count() > 0) {
       for (name in self$list_plugins()) {
         meta <- private$metadata[[name]]
         desc <- if (!is.null(meta$description)) {
           sprintf(" - %s", meta$description)
         } else ""
         cat(sprintf("  - %s (v%s)%s\n", name, meta$version, desc))
       }
     }
     invisible(self)
   }
 )
)
```

### Global Registry Instance

```r
#' Global feature plugin registry
#'
#' This is the default registry instance used by the system.
#' Plugin developers should register their plugins here.
#'
#' @export
.feature_registry <- NULL

#' Initialize global registry on package load
#' @noRd
.onLoad <- function(libname, pkgname) {
 .feature_registry <<- FeaturePluginRegistry$new()
}
```

### Convenience Functions

```r
#' Register a feature plugin
#'
#' Convenience function to register a plugin in the global registry.
#'
#' @param name Plugin name
#' @param plugin_class R6 class generator
#' @param ... Additional metadata (description, version, author)
#' @return Invisible NULL
#' @export
#' @examples
#' \dontrun{
#' register_feature_plugin("my_plugin", MyPluginClass,
#'   description = "My awesome plugin",
#'   version = "1.0.0"
#' )
#' }
register_feature_plugin <- function(name, plugin_class, ...) {
 .feature_registry$register(name, plugin_class, ...)
 invisible(NULL)
}


#' Get a feature plugin instance
#'
#' @param name Plugin name
#' @param config Plugin configuration
#' @return Plugin instance
#' @export
get_feature_plugin <- function(name, config = list()) {
 .feature_registry$get(name, config)
}


#' List available feature plugins
#'
#' @return Character vector of plugin names
#' @export
list_feature_plugins <- function() {
 .feature_registry$list_plugins()
}


#' Check if feature plugin exists
#'
#' @param name Plugin name
#' @return Logical
#' @export
has_feature_plugin <- function(name) {
 .feature_registry$has(name)
}


#' Get feature plugin metadata
#'
#' @param name Plugin name
#' @return List with plugin metadata
#' @export
feature_plugin_info <- function(name) {
 .feature_registry$get_metadata(name)
}
```

### Usage Example

```r
# Define a custom plugin
MyTemporalPlugin <- R6::R6Class(
 "MyTemporalPlugin",
 inherit = BaseFeaturePlugin,
 public = list(
   transform = function(dt, ...) {
     result <- data.table::copy(dt)
     result[, hour := data.table::hour(DataHora)]
     result[, day_of_week := data.table::wday(DataHora)]
     result
   },
   get_feature_names = function() {
     c("hour", "day_of_week")
   }
 )
)

# Register plugin
register_feature_plugin(
 "temporal_basic",
 MyTemporalPlugin,
 description = "Basic temporal features",
 version = "1.0.0",
 author = "ONS Team"
)

# List available plugins
list_feature_plugins()
# [1] "temporal_basic"

# Get plugin instance
plugin <- get_feature_plugin("temporal_basic")
transformed <- plugin$transform(data)

# Get plugin info
feature_plugin_info("temporal_basic")
# $description: "Basic temporal features"
# $version: "1.0.0"
# $author: "ONS Team"
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize empty registry | count() == 0 |
| TC-002 | Register valid plugin | Success, count() == 1 |
| TC-003 | Register non-R6 class | Error |
| TC-004 | Register non-BaseFeaturePlugin | Error |
| TC-005 | Register duplicate name | Warning, overwrites |
| TC-006 | get() existing plugin | Returns instance |
| TC-007 | get() non-existent plugin | Error with available list |
| TC-008 | get() with config | Plugin receives config |
| TC-009 | has() existing | TRUE |
| TC-010 | has() non-existent | FALSE |
| TC-011 | list_plugins() | Returns all names |
| TC-012 | unregister() existing | Removes plugin |
| TC-013 | unregister() non-existent | Warning |
| TC-014 | clear() | Removes all |
| TC-015 | summary() | Returns data.table |

---

## Definition of Done

- [ ] FeaturePluginRegistry class implemented
- [ ] Global registry instance created
- [ ] Convenience functions exported
- [ ] Plugin validation working
- [ ] Metadata storage working
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The registry validates plugins at registration time, not at get() time
- Consider adding plugin dependency tracking in the future
- The `%||%` operator is used for null coalescing (from rlang or define locally)
- Registration messages can be suppressed with `suppressMessages()` if needed
