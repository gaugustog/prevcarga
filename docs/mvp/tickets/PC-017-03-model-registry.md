# PC-017-03: ModelRegistry

**Epic:** [EPIC-03: Model Layer - Infrastructure](../epics/EPIC-03-model-layer-part1.md)
**Task Reference:** T-03.2
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Implement the `ModelRegistry` R6 class that manages registration, discovery, and instantiation of model plugins using the registry pattern.

---

## Acceptance Criteria

- [ ] Registry module created in `R/models/registry.R`
- [ ] `ModelRegistry` R6 class with registration methods
- [ ] `register()` validates model class inherits from BaseModel
- [ ] `get()` instantiates models with configuration
- [ ] `list_models()` returns registered model names
- [ ] `has()` checks if model is registered
- [ ] Global registry instance created (`.model_registry`)
- [ ] Export convenience functions for model contributors

---

## Technical Specification

### File Location
```
R/models/registry.R
```

### Registry Class

```r
#' @title ModelRegistry
#' @description Registry for model plugin classes
#'
#' The registry maintains a collection of model plugin classes that
#' can be instantiated on demand. Models are registered by name and
#' can be retrieved with optional configuration.
#'
#' @export
ModelRegistry <- R6::R6Class(
  "ModelRegistry",
  private = list(
    models = NULL,
    metadata = NULL
  ),
  public = list(
    #' @description Initialize registry
    initialize = function() {
      private$models <- list()
      private$metadata <- list()
    },

    #' @description Register a model class
    #' @param name Unique model name
    #' @param model_class R6 class generator (not instance)
    #' @param description Optional description
    #' @param version Optional version string
    #' @param author Optional author name
    #' @param model_type Type: "multi_horizon", "recursive", "ensemble"
    #' @return Invisible self (for chaining)
    register = function(name, model_class,
                        description = NULL,
                        version = "1.0.0",
                        author = NULL,
                        model_type = "multi_horizon") {
      checkmate::assert_string(name, min.chars = 1)
      checkmate::assert_choice(model_type,
        c("multi_horizon", "recursive", "ensemble", "custom"))

      # Validate that model_class is an R6 class generator
      if (!inherits(model_class, "R6ClassGenerator")) {
        stop(sprintf(
          "model_class must be an R6 class generator, got %s",
          class(model_class)[1]
        ))
      }

      # Validate that instances inherit from BaseModel
      test_instance <- tryCatch(
        model_class$new(),
        error = function(e) {
          stop(sprintf(
            "Failed to instantiate model '%s': %s",
            name, e$message
          ))
        }
      )

      if (!inherits(test_instance, "BaseModel")) {
        stop(sprintf(
          "Model '%s' must inherit from BaseModel",
          name
        ))
      }

      # Warn if overwriting existing registration
      if (name %in% names(private$models)) {
        warning(sprintf("Overwriting existing model registration: '%s'", name))
      }

      # Store model class and metadata
      private$models[[name]] <- model_class
      private$metadata[[name]] <- list(
        description = description,
        version = version,
        author = author,
        model_type = model_type,
        registered_at = Sys.time()
      )

      message(sprintf("Registered model: '%s' (%s)", name, model_type))
      invisible(self)
    },

    #' @description Get model instance
    #' @param name Model name
    #' @param horizons Forecast horizons (optional)
    #' @param config Configuration list for model
    #' @return Model instance
    get = function(name, horizons = NULL, config = list()) {
      checkmate::assert_string(name)
      checkmate::assert_list(config)

      if (!self$has(name)) {
        available <- paste(self$list_models(), collapse = ", ")
        stop(sprintf(
          "Model '%s' not registered. Available: %s",
          name,
          if (nchar(available) > 0) available else "(none)"
        ))
      }

      # Create instance with optional horizons
      if (!is.null(horizons)) {
        private$models[[name]]$new(name = name, horizons = horizons, config = config)
      } else {
        private$models[[name]]$new(name = name, config = config)
      }
    },

    #' @description Check if model is registered
    #' @param name Model name
    #' @return Logical
    has = function(name) {
      name %in% names(private$models)
    },

    #' @description List registered model names
    #' @param type Optional filter by model type
    #' @return Character vector
    list_models = function(type = NULL) {
      if (is.null(type)) {
        return(names(private$models))
      }

      # Filter by type
      names(private$models)[sapply(names(private$models), function(name) {
        private$metadata[[name]]$model_type == type
      })]
    },

    #' @description Get model count
    #' @return Integer
    count = function() {
      length(private$models)
    },

    #' @description Get model metadata
    #' @param name Model name
    #' @return List with metadata or NULL
    get_metadata = function(name) {
      if (!self$has(name)) {
        return(NULL)
      }
      private$metadata[[name]]
    },

    #' @description Get model class (not instance)
    #' @param name Model name
    #' @return R6 class generator or NULL
    get_class = function(name) {
      if (!self$has(name)) {
        return(NULL)
      }
      private$models[[name]]
    },

    #' @description Unregister a model
    #' @param name Model name
    #' @return Invisible self
    unregister = function(name) {
      if (!self$has(name)) {
        warning(sprintf("Model '%s' not registered, nothing to remove", name))
        return(invisible(self))
      }

      private$models[[name]] <- NULL
      private$metadata[[name]] <- NULL
      message(sprintf("Unregistered model: '%s'", name))
      invisible(self)
    },

    #' @description Clear all registrations
    #' @return Invisible self
    clear = function() {
      private$models <- list()
      private$metadata <- list()
      message("Cleared all model registrations")
      invisible(self)
    },

    #' @description Get summary of all models
    #' @return data.table with model information
    summary = function() {
      if (self$count() == 0) {
        return(data.table::data.table(
          name = character(),
          type = character(),
          version = character(),
          description = character()
        ))
      }

      data.table::rbindlist(lapply(names(private$models), function(name) {
        meta <- private$metadata[[name]]
        data.table::data.table(
          name = name,
          type = meta$model_type %||% NA_character_,
          version = meta$version %||% NA_character_,
          description = meta$description %||% NA_character_
        )
      }))
    },

    #' @description Print registry summary
    print = function() {
      cat(sprintf("<ModelRegistry: %d models>\n", self$count()))

      if (self$count() > 0) {
        # Group by type
        for (type in c("multi_horizon", "recursive", "ensemble", "custom")) {
          models <- self$list_models(type)
          if (length(models) > 0) {
            cat(sprintf("\n  %s:\n", type))
            for (name in models) {
              meta <- private$metadata[[name]]
              desc <- if (!is.null(meta$description)) {
                sprintf(" - %s", substr(meta$description, 1, 40))
              } else ""
              cat(sprintf("    - %s (v%s)%s\n", name, meta$version, desc))
            }
          }
        }
      }

      invisible(self)
    }
  )
)
```

### Global Registry Instance

```r
#' Global model registry
#'
#' This is the default registry instance used by the system.
#' Model developers should register their models here.
#'
#' @export
.model_registry <- NULL

#' Initialize global registry on package load
#' @noRd
.onLoad <- function(libname, pkgname) {
  .model_registry <<- ModelRegistry$new()
}
```

### Convenience Functions

```r
#' Register a model
#'
#' Convenience function to register a model in the global registry.
#'
#' @param name Model name
#' @param model_class R6 class generator
#' @param ... Additional metadata (description, version, author, model_type)
#' @return Invisible NULL
#' @export
#' @examples
#' \dontrun{
#' register_model("my_model", MyModelClass,
#'   description = "My custom forecasting model",
#'   version = "1.0.0",
#'   model_type = "multi_horizon"
#' )
#' }
register_model <- function(name, model_class, ...) {
  .model_registry$register(name, model_class, ...)
  invisible(NULL)
}


#' Get a model instance
#'
#' @param name Model name
#' @param horizons Forecast horizons (optional)
#' @param config Model configuration
#' @return Model instance
#' @export
get_model <- function(name, horizons = NULL, config = list()) {
  .model_registry$get(name, horizons = horizons, config = config)
}


#' List available models
#'
#' @param type Optional filter by model type
#' @return Character vector of model names
#' @export
list_models <- function(type = NULL) {
  .model_registry$list_models(type)
}


#' Check if model exists
#'
#' @param name Model name
#' @return Logical
#' @export
has_model <- function(name) {
  .model_registry$has(name)
}


#' Get model metadata
#'
#' @param name Model name
#' @return List with model metadata
#' @export
model_info <- function(name) {
  .model_registry$get_metadata(name)
}
```

### Usage Example

```r
# Define a custom model
MyForecaster <- R6::R6Class(
  "MyForecaster",
  inherit = BaseModel,
  public = list(
    train = function(X, y, ...) {
      # Training implementation
      self$is_trained <- TRUE
      invisible(self)
    },
    predict = function(X, horizon = NULL, ...) {
      # Prediction implementation
      rep(mean(X[[1]]), nrow(X))
    },
    save = function(path, version) {
      saveRDS(self, file.path(path, "model.rds"))
      invisible(self)
    },
    load = function(path) {
      # Load implementation
      invisible(self)
    }
  )
)

# Register model
register_model(
  "my_forecaster",
  MyForecaster,
  description = "Simple average forecaster",
  version = "1.0.0",
  model_type = "custom"
)

# List available models
list_models()
# [1] "my_forecaster"

# Get model instance
model <- get_model("my_forecaster", horizons = 0:4)
model$train(train_X, train_y)
predictions <- model$predict(test_X)

# Get model info
model_info("my_forecaster")
# $description: "Simple average forecaster"
# $model_type: "custom"
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize empty registry | count() == 0 |
| TC-002 | Register valid model | Success, count() == 1 |
| TC-003 | Register non-R6 class | Error |
| TC-004 | Register non-BaseModel | Error |
| TC-005 | Register duplicate name | Warning, overwrites |
| TC-006 | get() existing model | Returns instance |
| TC-007 | get() non-existent model | Error with available list |
| TC-008 | get() with horizons | Model has specified horizons |
| TC-009 | get() with config | Model receives config |
| TC-010 | has() existing | TRUE |
| TC-011 | has() non-existent | FALSE |
| TC-012 | list_models() all | Returns all names |
| TC-013 | list_models() by type | Returns filtered names |
| TC-014 | unregister() existing | Removes model |
| TC-015 | summary() | Returns data.table |

---

## Definition of Done

- [ ] ModelRegistry class implemented
- [ ] Global registry instance created
- [ ] Convenience functions exported
- [ ] Model type filtering working
- [ ] Metadata storage working
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The registry validates models at registration time
- Model types help organize and filter available models
- Consider adding model dependency tracking (e.g., required packages)
- The `%||%` operator is used for null coalescing
- Registration messages can be suppressed with `suppressMessages()`
