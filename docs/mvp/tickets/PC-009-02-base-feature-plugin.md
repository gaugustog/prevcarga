# PC-009-02: BaseFeaturePlugin Abstract Class

**Epic:** [EPIC-02: Feature Engineering Infrastructure](../epics/EPIC-02-feature-engineering.md)
**Task Reference:** T-02.1
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Create the `BaseFeaturePlugin` R6 abstract class that defines the contract for all feature engineering plugins. This is the foundation of the plugin architecture for feature transformation.

---

## Acceptance Criteria

- [ ] Base class created in `R/features/base.R`
- [ ] `BaseFeaturePlugin` R6 class with abstract methods
- [ ] `transform()` method enforced (throws if not implemented)
- [ ] `get_feature_names()` method enforced
- [ ] `validate_input()` method with default implementation
- [ ] `fit()` method for stateful plugins (optional)
- [ ] Configuration support via `config` field
- [ ] roxygen2 documentation for plugin developers

---

## Technical Specification

### File Location
```
R/features/base.R
```

### Class Definition

```r
#' @title BaseFeaturePlugin
#' @description Abstract base class for feature engineering plugins
#'
#' All feature plugins must inherit from this class and implement
#' the required methods: `transform()` and `get_feature_names()`.
#'
#' @details
#' Feature plugins transform input data.tables by adding new columns
#' (features) derived from existing data. Plugins can be stateless
#' (pure transformations) or stateful (requiring a fit step).
#'
#' @section Required Methods:
#' \describe{
#'   \item{transform(dt, ...)}{Transform input data, adding features}
#'   \item{get_feature_names()}{Return names of features created}
#' }
#'
#' @section Optional Methods:
#' \describe{
#'   \item{fit(dt, ...)}{Fit plugin state from training data}
#'   \item{validate_input(dt)}{Validate input data structure}
#' }
#'
#' @export
BaseFeaturePlugin <- R6::R6Class(
 "BaseFeaturePlugin",
 public = list(
   #' @field name Plugin name identifier
   name = NULL,

   #' @field config Plugin configuration list
   config = NULL,

   #' @field is_fitted Whether the plugin has been fitted
   is_fitted = FALSE,

   #' @description Initialize plugin
   #' @param name Plugin name (optional, defaults to class name)
   #' @param config Configuration list
   initialize = function(name = NULL, config = list()) {
     if (is.null(name)) {
       self$name <- class(self)[1]
     } else {
       self$name <- name
     }
     checkmate::assert_list(config)
     self$config <- config
   },

   #' @description Transform input data by adding features
   #' @param dt data.table to transform
   #' @param ... Additional arguments
   #' @return data.table with new feature columns added
   transform = function(dt, ...) {
     stop(sprintf(
       "transform() must be implemented by subclass '%s'",
       class(self)[1]
     ))
   },

   #' @description Get names of features created by this plugin
   #' @return Character vector of feature names
   get_feature_names = function() {
     stop(sprintf(
       "get_feature_names() must be implemented by subclass '%s'",
       class(self)[1]
     ))
   },

   #' @description Validate input data structure
   #' @param dt data.table to validate
   #' @return TRUE if valid, throws error otherwise
   validate_input = function(dt) {
     checkmate::assert_data_table(dt, min.rows = 1)
     invisible(TRUE)
   },

   #' @description Fit plugin state from training data
   #' @param dt Training data.table
   #' @param ... Additional arguments
   #' @return Invisible self (for chaining)
   fit = function(dt, ...) {
     # Default: no fitting required (stateless plugin)
     self$is_fitted <- TRUE
     invisible(self)
   },

   #' @description Fit and transform in one step
   #' @param dt data.table to fit and transform
   #' @param ... Additional arguments
   #' @return Transformed data.table
   fit_transform = function(dt, ...) {
     self$fit(dt, ...)
     self$transform(dt, ...)
   },

   #' @description Get configuration value
   #' @param key Configuration key

   #' @param default Default value if key not found
   #' @return Configuration value
   get_config = function(key, default = NULL) {
     if (key %in% names(self$config)) {
       self$config[[key]]
     } else {
       default
     }
   },

   #' @description Get plugin info as list
   #' @return List with plugin metadata
   info = function() {
     list(
       name = self$name,
       class = class(self)[1],
       is_fitted = self$is_fitted,
       config = self$config,
       features = tryCatch(
         self$get_feature_names(),
         error = function(e) character()
       )
     )
   },

   #' @description Print plugin summary
   print = function() {
     cat(sprintf("<%s>\n", class(self)[1]))
     cat(sprintf("  Name: %s\n", self$name))
     cat(sprintf("  Fitted: %s\n", self$is_fitted))
     if (length(self$config) > 0) {
       cat("  Config:\n")
       for (key in names(self$config)) {
         cat(sprintf("    %s: %s\n", key, self$config[[key]]))
       }
     }
     invisible(self)
   }
 ),

 private = list(
   #' Add columns to data.table safely
   #' @param dt data.table to modify
   #' @param cols Named list of columns to add
   add_columns = function(dt, cols) {
     for (col_name in names(cols)) {
       data.table::set(dt, j = col_name, value = cols[[col_name]])
     }
     invisible(dt)
   },

   #' Check if required columns exist
   #' @param dt data.table
   #' @param required Character vector of required column names
   require_columns = function(dt, required) {
     missing <- setdiff(required, names(dt))
     if (length(missing) > 0) {
       stop(sprintf(
         "Plugin '%s' requires columns: %s",
         self$name,
         paste(missing, collapse = ", ")
       ))
     }
     invisible(TRUE)
   }
 )
)
```

### Stateful Plugin Example

```r
#' Example: Plugin that normalizes features (requires fitting)
#'
#' @description
#' This is an example of a stateful plugin that learns parameters
#' from training data during the fit() step.
#'
#' @examples
#' \dontrun{
#' normalizer <- NormalizerPlugin$new(config = list(columns = c("CargaGlobal")))
#' normalizer$fit(train_data)
#' transformed <- normalizer$transform(test_data)
#' }
NormalizerPlugin <- R6::R6Class(
 "NormalizerPlugin",
 inherit = BaseFeaturePlugin,
 private = list(
   means = NULL,
   sds = NULL
 ),
 public = list(
   initialize = function(name = "normalizer", config = list()) {
     super$initialize(name, config)
   },

   fit = function(dt, ...) {
     columns <- self$get_config("columns", names(dt))
     numeric_cols <- intersect(columns, names(dt)[sapply(dt, is.numeric)])

     private$means <- sapply(numeric_cols, function(col) mean(dt[[col]], na.rm = TRUE))
     private$sds <- sapply(numeric_cols, function(col) sd(dt[[col]], na.rm = TRUE))

     self$is_fitted <- TRUE
     invisible(self)
   },

   transform = function(dt, ...) {
     if (!self$is_fitted) {
       stop("Plugin must be fitted before transform()")
     }

     self$validate_input(dt)
     result <- data.table::copy(dt)

     for (col in names(private$means)) {
       if (col %in% names(result)) {
         norm_col <- paste0(col, "_normalized")
         result[, (norm_col) := (get(col) - private$means[[col]]) / private$sds[[col]]]
       }
     }

     result
   },

   get_feature_names = function() {
     if (is.null(private$means)) {
       return(character())
     }
     paste0(names(private$means), "_normalized")
   }
 )
)
```

### Stateless Plugin Example

```r
#' Example: Simple stateless plugin
#'
#' @description
#' This is an example of a stateless plugin that performs pure
#' transformations without requiring a fit step.
#'
#' @examples
#' \dontrun{
#' hour_plugin <- HourOfDayPlugin$new()
#' transformed <- hour_plugin$transform(data)
#' }
HourOfDayPlugin <- R6::R6Class(
 "HourOfDayPlugin",
 inherit = BaseFeaturePlugin,
 public = list(
   initialize = function(name = "hour_of_day", config = list()) {
     super$initialize(name, config)
     self$is_fitted <- TRUE  # Stateless - always "fitted"
   },

   transform = function(dt, ...) {
     self$validate_input(dt)
     datetime_col <- self$get_config("datetime_col", "DataHora")

     private$require_columns(dt, datetime_col)

     result <- data.table::copy(dt)
     result[, hour := data.table::hour(get(datetime_col))]

     result
   },

   get_feature_names = function() {
     "hour"
   },

   validate_input = function(dt) {
     super$validate_input(dt)
     datetime_col <- self$get_config("datetime_col", "DataHora")
     if (!datetime_col %in% names(dt)) {
       stop(sprintf("DateTime column '%s' not found", datetime_col))
     }
     invisible(TRUE)
   }
 )
)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Instantiate BaseFeaturePlugin directly | Works (with abstract methods) |
| TC-002 | Call transform() on base class | Error: must be implemented |
| TC-003 | Call get_feature_names() on base class | Error: must be implemented |
| TC-004 | Subclass without transform() | Error when called |
| TC-005 | Subclass with proper implementation | Works correctly |
| TC-006 | validate_input() with valid data | TRUE |
| TC-007 | validate_input() with empty data.table | Error |
| TC-008 | fit_transform() combines both | Fit then transform |
| TC-009 | get_config() with existing key | Returns value |
| TC-010 | get_config() with missing key | Returns default |
| TC-011 | info() returns metadata | Correct list |
| TC-012 | print() shows summary | Formatted output |

---

## Definition of Done

- [ ] BaseFeaturePlugin class implemented
- [ ] Abstract method enforcement working
- [ ] Example plugins documented
- [ ] Private helper methods available
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The base class enforces the plugin contract via error-throwing default implementations
- Stateful plugins should implement `fit()` to learn parameters from training data
- Stateless plugins can set `is_fitted = TRUE` in the constructor
- The `private$require_columns()` helper makes input validation easier
- Consider adding `clone_plugin()` method for deep copying in the future
