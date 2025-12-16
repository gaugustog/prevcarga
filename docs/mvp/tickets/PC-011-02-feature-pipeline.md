# PC-011-02: FeaturePipeline

**Epic:** [EPIC-02: Feature Engineering Infrastructure](../epics/EPIC-02-feature-engineering.md)
**Task Reference:** T-02.3
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `FeaturePipeline` R6 class that composes multiple feature plugins into an ordered execution pipeline, supporting both sequential transformation and parallel execution strategies.

---

## Acceptance Criteria

- [ ] Pipeline module created in `R/features/pipeline.R`
- [ ] `FeaturePipeline` R6 class with plugin composition
- [ ] `add()` adds plugins with optional position control
- [ ] `remove()` removes plugins by name
- [ ] `transform()` executes all plugins in order
- [ ] `fit()` fits all stateful plugins
- [ ] `get_all_feature_names()` aggregates feature names
- [ ] `from_config()` builds pipeline from configuration
- [ ] Support for pipeline cloning and modification

---

## Technical Specification

### File Location
```
R/features/pipeline.R
```

### Pipeline Class

```r
#' @title FeaturePipeline
#' @description Compose multiple feature plugins into a transformation pipeline
#'
#' A pipeline maintains an ordered list of feature plugins and executes
#' them sequentially. Each plugin receives the output of the previous plugin.
#'
#' @export
FeaturePipeline <- R6::R6Class(
 "FeaturePipeline",
 private = list(
   plugins = NULL,
   registry = NULL,
   is_fitted = FALSE
 ),
 public = list(
   #' @field name Pipeline name
   name = NULL,

   #' @description Initialize pipeline
   #' @param name Optional pipeline name
   #' @param registry Optional FeaturePluginRegistry for from_config()
   initialize = function(name = "pipeline", registry = NULL) {
     self$name <- name
     private$plugins <- list()
     private$registry <- registry
   },

   #' @description Add plugin to pipeline
   #' @param plugin BaseFeaturePlugin instance
   #' @param position Position in pipeline (NULL = append)
   #' @return Invisible self (for chaining)
   add = function(plugin, position = NULL) {
     checkmate::assert_class(plugin, "BaseFeaturePlugin")

     # Check for duplicate names
     if (plugin$name %in% self$list_plugins()) {
       warning(sprintf(
         "Plugin '%s' already in pipeline, adding with unique suffix",
         plugin$name
       ))
       plugin$name <- sprintf("%s_%d", plugin$name, length(private$plugins) + 1)
     }

     if (is.null(position)) {
       private$plugins <- c(private$plugins, list(plugin))
     } else {
       checkmate::assert_integerish(position, lower = 1,
                                     upper = length(private$plugins) + 1)
       private$plugins <- append(private$plugins, list(plugin), after = position - 1)
     }

     private$is_fitted <- FALSE
     invisible(self)
   },

   #' @description Remove plugin by name
   #' @param name Plugin name to remove
   #' @return Invisible self
   remove = function(name) {
     checkmate::assert_string(name)

     idx <- which(sapply(private$plugins, function(p) p$name) == name)
     if (length(idx) == 0) {
       warning(sprintf("Plugin '%s' not found in pipeline", name))
       return(invisible(self))
     }

     private$plugins <- private$plugins[-idx]
     private$is_fitted <- FALSE
     invisible(self)
   },

   #' @description Get plugin by name
   #' @param name Plugin name
   #' @return Plugin instance or NULL
   get = function(name) {
     idx <- which(sapply(private$plugins, function(p) p$name) == name)
     if (length(idx) == 0) {
       return(NULL)
     }
     private$plugins[[idx[1]]]
   },

   #' @description List plugin names in order
   #' @return Character vector
   list_plugins = function() {
     sapply(private$plugins, function(p) p$name)
   },

   #' @description Get plugin count
   #' @return Integer
   length = function() {
     length(private$plugins)
   },

   #' @description Fit all plugins on training data
   #' @param dt Training data.table
   #' @param ... Additional arguments passed to each plugin
   #' @return Invisible self
   fit = function(dt, ...) {
     checkmate::assert_data_table(dt)

     current_dt <- data.table::copy(dt)

     for (i in seq_along(private$plugins)) {
       plugin <- private$plugins[[i]]
       message(sprintf("[%d/%d] Fitting plugin: %s",
                       i, length(private$plugins), plugin$name))

       plugin$fit(current_dt, ...)

       # Transform for next plugin (if not last)
       if (i < length(private$plugins)) {
         current_dt <- plugin$transform(current_dt, ...)
       }
     }

     private$is_fitted <- TRUE
     invisible(self)
   },

   #' @description Transform data through all plugins
   #' @param dt data.table to transform
   #' @param ... Additional arguments passed to each plugin
   #' @return Transformed data.table
   transform = function(dt, ...) {
     checkmate::assert_data_table(dt)

     if (length(private$plugins) == 0) {
       warning("Pipeline has no plugins, returning input unchanged")
       return(data.table::copy(dt))
     }

     result <- data.table::copy(dt)

     for (i in seq_along(private$plugins)) {
       plugin <- private$plugins[[i]]

       # Validate input
       plugin$validate_input(result)

       # Transform
       result <- plugin$transform(result, ...)
     }

     result
   },

   #' @description Fit and transform in one step
   #' @param dt data.table
   #' @param ... Additional arguments
   #' @return Transformed data.table
   fit_transform = function(dt, ...) {
     self$fit(dt, ...)
     self$transform(dt, ...)
   },

   #' @description Get all feature names from all plugins
   #' @return Character vector of all feature names
   get_all_feature_names = function() {
     if (length(private$plugins) == 0) {
       return(character())
     }

     unique(unlist(lapply(private$plugins, function(p) {
       tryCatch(
         p$get_feature_names(),
         error = function(e) character()
       )
     })))
   },

   #' @description Build pipeline from configuration
   #' @param config List with plugin configurations
   #' @param registry FeaturePluginRegistry to use
   #' @return Invisible self
   from_config = function(config, registry = NULL) {
     checkmate::assert_list(config)

     reg <- registry %||% private$registry %||% .feature_registry

     if (is.null(reg)) {
       stop("No registry provided and global registry not available")
     }

     # Clear existing plugins
     private$plugins <- list()

     # Parse plugins from config
     plugins_config <- config$plugins %||% config

     if (!is.list(plugins_config)) {
       stop("Config must contain a 'plugins' list")
     }

     for (plugin_conf in plugins_config) {
       name <- plugin_conf$name
       if (is.null(name)) {
         stop("Each plugin config must have a 'name' field")
       }

       plugin_config <- plugin_conf$config %||% list()
       enabled <- plugin_conf$enabled %||% TRUE

       if (!enabled) {
         message(sprintf("Skipping disabled plugin: %s", name))
         next
       }

       plugin <- reg$get(name, plugin_config)
       self$add(plugin)
     }

     invisible(self)
   },

   #' @description Clone pipeline (deep copy)
   #' @return New FeaturePipeline instance
   clone_pipeline = function() {
     new_pipeline <- FeaturePipeline$new(
       name = paste0(self$name, "_copy"),
       registry = private$registry
     )

     for (plugin in private$plugins) {
       # Create new instance with same config
       new_plugin <- plugin$clone(deep = TRUE)
       new_pipeline$add(new_plugin)
     }

     new_pipeline
   },

   #' @description Get pipeline info
   #' @return List with pipeline metadata
   info = function() {
     list(
       name = self$name,
       plugin_count = self$length(),
       plugins = lapply(private$plugins, function(p) p$info()),
       is_fitted = private$is_fitted,
       feature_count = length(self$get_all_feature_names())
     )
   },

   #' @description Print pipeline summary
   print = function() {
     cat(sprintf("<FeaturePipeline: %s>\n", self$name))
     cat(sprintf("  Plugins: %d\n", self$length()))
     cat(sprintf("  Fitted: %s\n", private$is_fitted))

     if (self$length() > 0) {
       cat("  Order:\n")
       for (i in seq_along(private$plugins)) {
         plugin <- private$plugins[[i]]
         features <- tryCatch(
           length(plugin$get_feature_names()),
           error = function(e) "?"
         )
         cat(sprintf("    %d. %s (%s features)\n", i, plugin$name, features))
       }
     }

     invisible(self)
   }
 )
)
```

### Pipeline Builder (Fluent Interface)

```r
#' Create a new feature pipeline
#'
#' @param name Pipeline name
#' @param registry Optional registry for from_config()
#' @return FeaturePipeline instance
#' @export
feature_pipeline <- function(name = "pipeline", registry = NULL) {
 FeaturePipeline$new(name = name, registry = registry)
}


#' Build pipeline from YAML configuration
#'
#' @param config_path Path to YAML config file
#' @param registry Optional registry
#' @return FeaturePipeline instance
#' @export
pipeline_from_yaml <- function(config_path, registry = NULL) {
 checkmate::assert_file_exists(config_path)
 config <- yaml::read_yaml(config_path)

 pipeline <- FeaturePipeline$new(registry = registry)
 pipeline$from_config(config$features %||% config)
 pipeline
}
```

### Usage Example

```r
# Fluent pipeline construction
pipeline <- feature_pipeline("load_features")$
 add(get_feature_plugin("temporal_basic"))$
 add(get_feature_plugin("calendar"))$
 add(get_feature_plugin("lag_features", list(lags = c(24, 48, 168))))

# Fit and transform
train_features <- pipeline$fit_transform(train_data)
test_features <- pipeline$transform(test_data)

# Get all feature names
all_features <- pipeline$get_all_feature_names()

# From YAML config
# features.yaml:
#   plugins:
#     - name: temporal_basic
#       enabled: true
#     - name: calendar
#       config:
#         include_holidays: true
#     - name: lag_features
#       config:
#         lags: [24, 48, 168]

pipeline <- pipeline_from_yaml("features.yaml")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize empty pipeline | length() == 0 |
| TC-002 | Add single plugin | length() == 1 |
| TC-003 | Add multiple plugins | Correct order |
| TC-004 | Add with position | Inserted at position |
| TC-005 | Add duplicate name | Warning, unique suffix |
| TC-006 | Remove existing plugin | Removed |
| TC-007 | Remove non-existent | Warning |
| TC-008 | transform() empty pipeline | Returns copy with warning |
| TC-009 | transform() with plugins | All plugins applied |
| TC-010 | fit() stores state | is_fitted == TRUE |
| TC-011 | get_all_feature_names() | Aggregated unique names |
| TC-012 | from_config() valid | Pipeline built |
| TC-013 | from_config() disabled plugin | Skipped |
| TC-014 | from_config() missing name | Error |
| TC-015 | clone_pipeline() | Deep copy works |

---

## Definition of Done

- [ ] FeaturePipeline class implemented
- [ ] Builder functions created
- [ ] YAML config loading works
- [ ] Plugin ordering works correctly
- [ ] fit/transform flow works
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Plugins are executed in the order they were added
- Each plugin receives the output of the previous plugin
- Consider adding parallel execution option for independent plugins
- The pipeline should be serializable for caching fitted state
- YAML config format should match the CLI configuration system (EPIC-08)
