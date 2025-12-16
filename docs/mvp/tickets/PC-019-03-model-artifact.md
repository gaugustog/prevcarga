# PC-019-03: ModelArtifact

**Epic:** [EPIC-03: Model Layer - Infrastructure](../epics/EPIC-03-model-layer-part1.md)
**Task Reference:** T-03.4
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Implement the `ModelArtifact` R6 class for packaging, serializing, and deserializing complete model artifacts including the trained model, metadata, normalization parameters, and feature names.

---

## Acceptance Criteria

- [ ] Artifact module created in `R/models/artifact.R`
- [ ] `ModelArtifact` R6 class with save/load methods
- [ ] Save model.rds with fitted model object
- [ ] Save metadata.yaml with training metadata
- [ ] Save normalization.rds if normalization was applied
- [ ] Save feature_names.json with feature list
- [ ] Support for both local and S3 storage backends
- [ ] Atomic save operation (all or nothing)

---

## Technical Specification

### File Location
```
R/models/artifact.R
```

### Artifact Directory Structure

```
models/{model_name}/{version}/
├── model.rds              # Fitted model object(s)
├── metadata.yaml          # Training metadata
├── feature_names.json     # List of feature names
├── normalization.rds      # Normalization parameters (optional)
├── config.yaml            # Model configuration
└── metrics.json           # Training/validation metrics
```

### ModelArtifact Class

```r
#' @title ModelArtifact
#' @description Package for model serialization and deserialization
#'
#' Encapsulates all components needed to save and restore a trained model,
#' including the model itself, training metadata, feature information,
#' and any preprocessing parameters.
#'
#' @export
ModelArtifact <- R6::R6Class(
  "ModelArtifact",
  public = list(
    #' @field model Fitted model object
    model = NULL,

    #' @field model_name Name of the model
    model_name = NULL,

    #' @field version ModelVersion instance
    version = NULL,

    #' @field metadata Training metadata
    metadata = NULL,

    #' @field feature_names Character vector of feature names
    feature_names = NULL,

    #' @field normalization_params Normalization parameters (optional)
    normalization_params = NULL,

    #' @field config Model configuration
    config = NULL,

    #' @field metrics Training/validation metrics
    metrics = NULL,

    #' @description Initialize artifact
    #' @param model Fitted model object
    #' @param model_name Model name
    #' @param version ModelVersion or version string
    #' @param feature_names Feature names used during training
    #' @param config Model configuration
    initialize = function(model = NULL,
                          model_name = NULL,
                          version = NULL,
                          feature_names = NULL,
                          config = NULL) {
      self$model <- model
      self$model_name <- model_name

      if (is.character(version)) {
        self$version <- ModelVersion$new(version_string = version)
      } else if (inherits(version, "ModelVersion")) {
        self$version <- version
      } else {
        self$version <- ModelVersion$new()
      }

      self$feature_names <- feature_names
      self$config <- config %||% list()
      self$metadata <- private$create_default_metadata()
      self$metrics <- list()
    },

    #' @description Save artifact to storage
    #' @param storage StorageBackend instance
    #' @param base_path Base path for models (default: "models")
    #' @return Invisible artifact path
    save = function(storage, base_path = "models") {
      checkmate::assert_class(storage, "StorageBackend")
      checkmate::assert_string(base_path)

      if (is.null(self$model)) {
        stop("Cannot save artifact: model is NULL")
      }

      if (is.null(self$model_name)) {
        stop("Cannot save artifact: model_name is required")
      }

      # Build artifact path
      artifact_path <- file.path(
        base_path,
        self$model_name,
        self$version$to_string()
      )

      # Update metadata before saving
      self$metadata$saved_at <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
      self$metadata$artifact_path <- artifact_path

      tryCatch({
        # Save model object
        storage$write_rds(
          self$model,
          file.path(artifact_path, "model.rds")
        )

        # Save metadata as YAML
        storage$write_yaml(
          self$metadata,
          file.path(artifact_path, "metadata.yaml")
        )

        # Save feature names as JSON
        if (!is.null(self$feature_names)) {
          storage$write_json(
            list(features = self$feature_names),
            file.path(artifact_path, "feature_names.json")
          )
        }

        # Save normalization parameters if present
        if (!is.null(self$normalization_params)) {
          storage$write_rds(
            self$normalization_params,
            file.path(artifact_path, "normalization.rds")
          )
        }

        # Save config
        if (length(self$config) > 0) {
          storage$write_yaml(
            self$config,
            file.path(artifact_path, "config.yaml")
          )
        }

        # Save metrics if present
        if (length(self$metrics) > 0) {
          storage$write_json(
            self$metrics,
            file.path(artifact_path, "metrics.json")
          )
        }

        message(sprintf("Saved model artifact to: %s", artifact_path))

      }, error = function(e) {
        # Attempt cleanup on failure
        tryCatch(
          storage$delete_directory(artifact_path),
          error = function(e2) NULL
        )
        stop(sprintf("Failed to save artifact: %s", e$message))
      })

      invisible(artifact_path)
    },

    #' @description Load artifact from storage
    #' @param storage StorageBackend instance
    #' @param path Path to artifact directory
    #' @return Invisible self
    load = function(storage, path) {
      checkmate::assert_class(storage, "StorageBackend")
      checkmate::assert_string(path)

      # Check required files exist
      model_path <- file.path(path, "model.rds")
      if (!storage$exists(model_path)) {
        stop(sprintf("Model file not found: %s", model_path))
      }

      # Load model
      self$model <- storage$read_rds(model_path)

      # Load metadata
      metadata_path <- file.path(path, "metadata.yaml")
      if (storage$exists(metadata_path)) {
        self$metadata <- storage$read_yaml(metadata_path)
        self$model_name <- self$metadata$model_name
      }

      # Parse version from path
      version_str <- basename(path)
      self$version <- tryCatch(
        ModelVersion$new(version_string = version_str),
        error = function(e) ModelVersion$new()
      )

      # Load feature names
      features_path <- file.path(path, "feature_names.json")
      if (storage$exists(features_path)) {
        features_data <- storage$read_json(features_path)
        self$feature_names <- features_data$features
      }

      # Load normalization parameters
      norm_path <- file.path(path, "normalization.rds")
      if (storage$exists(norm_path)) {
        self$normalization_params <- storage$read_rds(norm_path)
      }

      # Load config
      config_path <- file.path(path, "config.yaml")
      if (storage$exists(config_path)) {
        self$config <- storage$read_yaml(config_path)
      }

      # Load metrics
      metrics_path <- file.path(path, "metrics.json")
      if (storage$exists(metrics_path)) {
        self$metrics <- storage$read_json(metrics_path)
      }

      message(sprintf("Loaded model artifact from: %s", path))
      invisible(self)
    },

    #' @description Set training metrics
    #' @param metrics Named list of metrics
    #' @return Invisible self
    set_metrics = function(metrics) {
      checkmate::assert_list(metrics, names = "named")
      self$metrics <- metrics
      invisible(self)
    },

    #' @description Add single metric
    #' @param name Metric name
    #' @param value Metric value
    #' @return Invisible self
    add_metric = function(name, value) {
      self$metrics[[name]] <- value
      invisible(self)
    },

    #' @description Set normalization parameters
    #' @param params Normalization parameters list
    #' @return Invisible self
    set_normalization = function(params) {
      self$normalization_params <- params
      invisible(self)
    },

    #' @description Update metadata
    #' @param key Metadata key
    #' @param value Metadata value
    #' @return Invisible self
    set_metadata = function(key, value) {
      self$metadata[[key]] <- value
      invisible(self)
    },

    #' @description Get artifact summary
    #' @return List with artifact information
    summary = function() {
      list(
        model_name = self$model_name,
        version = self$version$to_string(),
        feature_count = length(self$feature_names),
        has_normalization = !is.null(self$normalization_params),
        metrics = self$metrics,
        created_at = self$metadata$created_at,
        saved_at = self$metadata$saved_at
      )
    },

    #' @description Print artifact summary
    print = function() {
      cat(sprintf("<ModelArtifact: %s>\n", self$model_name %||% "(unnamed)"))
      cat(sprintf("  Version: %s\n", self$version$to_string()))
      cat(sprintf("  Features: %d\n", length(self$feature_names)))
      cat(sprintf("  Has Normalization: %s\n", !is.null(self$normalization_params)))

      if (length(self$metrics) > 0) {
        cat("  Metrics:\n")
        for (name in names(self$metrics)) {
          value <- self$metrics[[name]]
          if (is.numeric(value)) {
            cat(sprintf("    %s: %.4f\n", name, value))
          } else {
            cat(sprintf("    %s: %s\n", name, value))
          }
        }
      }

      invisible(self)
    }
  ),

  private = list(
    #' Create default metadata
    create_default_metadata = function() {
      list(
        model_name = self$model_name,
        version = self$version$to_string(),
        created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        r_version = R.version.string,
        package_version = packageVersion("prevcargaons") %||% "0.0.0",
        platform = Sys.info()["sysname"]
      )
    }
  )
)
```

### Convenience Functions

```r
#' Create a model artifact
#'
#' @param model Fitted model object
#' @param model_name Model name
#' @param version Version string or ModelVersion
#' @param feature_names Feature names
#' @param config Model configuration
#' @return ModelArtifact instance
#' @export
create_artifact <- function(model, model_name, version = NULL,
                            feature_names = NULL, config = NULL) {
  ModelArtifact$new(
    model = model,
    model_name = model_name,
    version = version,
    feature_names = feature_names,
    config = config
  )
}


#' Load a model artifact
#'
#' @param storage StorageBackend instance
#' @param model_name Model name
#' @param version Version string (default: "latest")
#' @param base_path Base models path
#' @return ModelArtifact instance
#' @export
load_artifact <- function(storage, model_name, version = "latest",
                          base_path = "models") {
  if (version == "latest") {
    version <- get_latest_model_version(storage, model_name, base_path)
    if (is.null(version)) {
      stop(sprintf("No versions found for model: %s", model_name))
    }
  }

  path <- file.path(base_path, model_name, version)
  artifact <- ModelArtifact$new()
  artifact$load(storage, path)
  artifact
}


#' List artifact versions for a model
#'
#' @param storage StorageBackend instance
#' @param model_name Model name
#' @param base_path Base models path
#' @return Character vector of version strings
#' @export
list_artifact_versions <- function(storage, model_name, base_path = "models") {
  model_path <- file.path(base_path, model_name)

  if (!storage$exists(model_path)) {
    return(character())
  }

  dirs <- storage$list_directories(model_path)
  # Filter to valid version directories
  dirs[grepl("^v\\d+\\.\\d+\\.\\d+", dirs)]
}
```

### Usage Example

```r
# Create artifact from trained model
artifact <- create_artifact(
  model = fitted_lgbm_model,
  model_name = "lgbm_rj",
  version = "v1.0.0",
  feature_names = c("hour", "day_of_week", "temp", "lag_24"),
  config = list(learning_rate = 0.1, num_leaves = 31)
)

# Add metrics
artifact$set_metrics(list(
  train_mape = 2.5,
  val_mape = 3.1,
  train_rmse = 150.2,
  val_rmse = 185.4
))

# Add normalization if used
artifact$set_normalization(list(
  means = list(temp = 25.0, load = 5000.0),
  stds = list(temp = 5.0, load = 1000.0)
))

# Save to storage
storage <- LocalStorageBackend$new(base_path = "./data")
artifact$save(storage, base_path = "models")
# Saved to: models/lgbm_rj/v1.0.0_20250117_143052/

# Load artifact
loaded <- load_artifact(storage, "lgbm_rj", version = "latest")
loaded$model  # The fitted model
loaded$feature_names  # Feature list
loaded$metrics  # Training metrics

# List versions
versions <- list_artifact_versions(storage, "lgbm_rj")
# c("v1.0.0_20250117_143052")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Create artifact with model | All fields set |
| TC-002 | Create artifact without model | Works (for loading) |
| TC-003 | Save complete artifact | All files created |
| TC-004 | Save artifact without model | Error |
| TC-005 | Save with normalization | normalization.rds created |
| TC-006 | Load complete artifact | All fields restored |
| TC-007 | Load artifact missing files | Error for required, skip optional |
| TC-008 | set_metrics() | Metrics stored |
| TC-009 | add_metric() | Single metric added |
| TC-010 | set_normalization() | Params stored |
| TC-011 | summary() | Correct structure |
| TC-012 | list_artifact_versions() | Returns versions |
| TC-013 | load_artifact() latest | Gets newest version |
| TC-014 | Atomic save failure | Cleanup attempted |

---

## Definition of Done

- [ ] ModelArtifact class implemented
- [ ] Save/load with storage backend
- [ ] All artifact components handled
- [ ] Convenience functions exported
- [ ] Atomic save operation
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Artifacts are immutable once saved (create new version for changes)
- Consider adding compression option for large models
- The storage backend abstraction enables both local and S3 storage
- Metadata includes R version for reproducibility tracking
- Future: add artifact signing for integrity verification
