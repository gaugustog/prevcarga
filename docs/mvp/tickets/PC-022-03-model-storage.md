# PC-022-03: Model Storage Utilities

**Epic:** [EPIC-03: Model Layer - Infrastructure](../epics/EPIC-03-model-layer-part1.md)
**Task Reference:** T-03.7
**Priority:** Medium
**Estimated Effort:** 0.5 days

---

## Summary

Implement utility functions for model path management, version discovery, and symlink handling to support model artifact storage and retrieval.

---

## Acceptance Criteria

- [ ] Storage utilities module created in `R/models/storage.R`
- [ ] `get_model_path()` builds paths for model artifacts
- [ ] `get_latest_version()` finds the latest model version
- [ ] `list_model_versions()` lists all versions for a model
- [ ] `create_latest_symlink()` manages "latest" pointer
- [ ] Support for version cleanup (keep N versions)
- [ ] Works with both local and S3 storage backends

---

## Technical Specification

### File Location
```
R/models/storage.R
```

### Path Management Functions

```r
#' Build path for model artifacts
#'
#' @param model_name Model name
#' @param version Version string (optional)
#' @param base_path Base models path
#' @return Full path to model artifacts
#' @export
#' @examples
#' get_model_path("lgbm_rj", "v1.0.0")
#' # "models/lgbm_rj/v1.0.0"
#'
#' get_model_path("lgbm_rj")
#' # "models/lgbm_rj"
get_model_path <- function(model_name, version = NULL, base_path = "models") {
  checkmate::assert_string(model_name, min.chars = 1)
  checkmate::assert_string(base_path)

  if (is.null(version)) {
    file.path(base_path, model_name)
  } else {
    file.path(base_path, model_name, version)
  }
}


#' Build path for specific artifact file
#'
#' @param model_name Model name
#' @param version Version string
#' @param filename Artifact filename
#' @param base_path Base models path
#' @return Full path to artifact file
#' @export
get_artifact_path <- function(model_name, version, filename,
                               base_path = "models") {
  file.path(get_model_path(model_name, version, base_path), filename)
}
```

### Version Discovery Functions

```r
#' Get latest version for a model
#'
#' @param storage StorageBackend instance
#' @param model_name Model name
#' @param base_path Base models path
#' @return Latest version string or NULL if no versions exist
#' @export
get_latest_model_version <- function(storage, model_name, base_path = "models") {
  checkmate::assert_class(storage, "StorageBackend")
  checkmate::assert_string(model_name)

  versions <- list_model_versions(storage, model_name, base_path)

  if (length(versions) == 0) {
    return(NULL)
  }

  # Check for "latest" symlink/pointer first
  latest_path <- file.path(base_path, model_name, "latest")
  if (storage$exists(latest_path)) {
    # Read symlink target or pointer file
    tryCatch({
      target <- storage$read_text(file.path(latest_path, ".version"))
      if (target %in% versions) {
        return(target)
      }
    }, error = function(e) NULL)
  }

  # Otherwise, find the latest by parsing versions
  get_latest_version(versions)
}


#' List all versions for a model
#'
#' @param storage StorageBackend instance
#' @param model_name Model name
#' @param base_path Base models path
#' @return Character vector of version strings (sorted newest first)
#' @export
list_model_versions <- function(storage, model_name, base_path = "models") {
  checkmate::assert_class(storage, "StorageBackend")
  checkmate::assert_string(model_name)

  model_path <- get_model_path(model_name, base_path = base_path)

  if (!storage$exists(model_path)) {
    return(character())
  }

  # List directories
  dirs <- storage$list_directories(model_path)

  # Filter to valid version directories
  version_pattern <- "^v\\d+\\.\\d+\\.\\d+"
  versions <- dirs[grepl(version_pattern, dirs)]

  # Exclude "latest" pointer
  versions <- versions[versions != "latest"]

  # Sort newest first
  if (length(versions) > 0) {
    versions <- rev(sort_versions(versions))
  }

  versions
}


#' Check if model version exists
#'
#' @param storage StorageBackend instance
#' @param model_name Model name
#' @param version Version string
#' @param base_path Base models path
#' @return Logical
#' @export
model_version_exists <- function(storage, model_name, version,
                                  base_path = "models") {
  path <- get_model_path(model_name, version, base_path)
  storage$exists(path)
}
```

### Latest Pointer Management

```r
#' Create or update "latest" pointer for a model
#'
#' @param storage StorageBackend instance
#' @param model_name Model name
#' @param version Version to point to
#' @param base_path Base models path
#' @return Invisible path to latest pointer
#' @export
create_latest_pointer <- function(storage, model_name, version,
                                   base_path = "models") {
  checkmate::assert_class(storage, "StorageBackend")
  checkmate::assert_string(model_name)
  checkmate::assert_string(version)

  latest_path <- file.path(base_path, model_name, "latest")

  # Create pointer file
  storage$write_text(
    version,
    file.path(latest_path, ".version")
  )

  message(sprintf("Updated 'latest' pointer for %s -> %s", model_name, version))
  invisible(latest_path)
}


#' Get version that "latest" points to
#'
#' @param storage StorageBackend instance
#' @param model_name Model name
#' @param base_path Base models path
#' @return Version string or NULL if no pointer exists
#' @export
get_latest_pointer <- function(storage, model_name, base_path = "models") {
  latest_path <- file.path(base_path, model_name, "latest", ".version")

  if (!storage$exists(latest_path)) {
    return(NULL)
  }

  tryCatch(
    trimws(storage$read_text(latest_path)),
    error = function(e) NULL
  )
}


#' Update latest pointer to newest version
#'
#' @param storage StorageBackend instance
#' @param model_name Model name
#' @param base_path Base models path
#' @return Invisible TRUE if updated, FALSE if no versions
#' @export
auto_update_latest <- function(storage, model_name, base_path = "models") {
  versions <- list_model_versions(storage, model_name, base_path)

  if (length(versions) == 0) {
    return(invisible(FALSE))
  }

  latest <- versions[1]  # Already sorted newest first
  create_latest_pointer(storage, model_name, latest, base_path)

  invisible(TRUE)
}
```

### Version Cleanup Functions

```r
#' Clean up old model versions
#'
#' @param storage StorageBackend instance
#' @param model_name Model name
#' @param keep Number of versions to keep
#' @param base_path Base models path
#' @param dry_run If TRUE, only report what would be deleted
#' @return Character vector of deleted (or would-be-deleted) versions
#' @export
cleanup_old_versions <- function(storage, model_name, keep = 5,
                                  base_path = "models", dry_run = FALSE) {
  checkmate::assert_class(storage, "StorageBackend")
  checkmate::assert_integerish(keep, lower = 1)

  versions <- list_model_versions(storage, model_name, base_path)

  if (length(versions) <= keep) {
    message(sprintf("Only %d versions exist, nothing to clean up", length(versions)))
    return(character())
  }

  # Versions to delete (keep newest `keep` versions)
  to_delete <- versions[(keep + 1):length(versions)]

  if (dry_run) {
    message(sprintf("Would delete %d versions:", length(to_delete)))
    for (v in to_delete) {
      message(sprintf("  - %s", v))
    }
    return(to_delete)
  }

  # Actually delete
  deleted <- character()
  for (version in to_delete) {
    path <- get_model_path(model_name, version, base_path)
    tryCatch({
      storage$delete_directory(path)
      deleted <- c(deleted, version)
      message(sprintf("Deleted: %s/%s", model_name, version))
    }, error = function(e) {
      warning(sprintf("Failed to delete %s: %s", version, e$message))
    })
  }

  deleted
}


#' Get disk usage for model versions
#'
#' @param storage StorageBackend instance
#' @param model_name Model name
#' @param base_path Base models path
#' @return data.table with version and size information
#' @export
get_model_storage_usage <- function(storage, model_name, base_path = "models") {
  versions <- list_model_versions(storage, model_name, base_path)

  if (length(versions) == 0) {
    return(data.table::data.table(
      version = character(),
      size_bytes = numeric(),
      size_mb = numeric()
    ))
  }

  data.table::rbindlist(lapply(versions, function(version) {
    path <- get_model_path(model_name, version, base_path)
    size <- tryCatch(
      storage$get_directory_size(path),
      error = function(e) NA_real_
    )

    data.table::data.table(
      version = version,
      size_bytes = size,
      size_mb = round(size / 1024^2, 2)
    )
  }))
}
```

### Model Discovery Functions

```r
#' List all models in storage
#'
#' @param storage StorageBackend instance
#' @param base_path Base models path
#' @return data.table with model information
#' @export
list_all_models <- function(storage, base_path = "models") {
  checkmate::assert_class(storage, "StorageBackend")

  if (!storage$exists(base_path)) {
    return(data.table::data.table(
      model_name = character(),
      version_count = integer(),
      latest_version = character()
    ))
  }

  model_dirs <- storage$list_directories(base_path)

  if (length(model_dirs) == 0) {
    return(data.table::data.table(
      model_name = character(),
      version_count = integer(),
      latest_version = character()
    ))
  }

  data.table::rbindlist(lapply(model_dirs, function(model_name) {
    versions <- list_model_versions(storage, model_name, base_path)
    latest <- if (length(versions) > 0) versions[1] else NA_character_

    data.table::data.table(
      model_name = model_name,
      version_count = length(versions),
      latest_version = latest
    )
  }))
}


#' Find models by name pattern
#'
#' @param storage StorageBackend instance
#' @param pattern Regex pattern for model names
#' @param base_path Base models path
#' @return Character vector of matching model names
#' @export
find_models <- function(storage, pattern, base_path = "models") {
  all_models <- list_all_models(storage, base_path)

  if (nrow(all_models) == 0) {
    return(character())
  }

  matches <- grepl(pattern, all_models$model_name)
  all_models$model_name[matches]
}
```

### Usage Example

```r
storage <- LocalStorageBackend$new(base_path = "./data")

# Build paths
get_model_path("lgbm_rj", "v1.0.0")
# "models/lgbm_rj/v1.0.0"

get_artifact_path("lgbm_rj", "v1.0.0", "model.rds")
# "models/lgbm_rj/v1.0.0/model.rds"

# List versions
versions <- list_model_versions(storage, "lgbm_rj")
# c("v1.2.0_20250117", "v1.1.0_20250110", "v1.0.0_20250103")

# Get latest
latest <- get_latest_model_version(storage, "lgbm_rj")
# "v1.2.0_20250117"

# Create latest pointer
create_latest_pointer(storage, "lgbm_rj", "v1.2.0_20250117")

# Cleanup old versions (keep 3)
cleanup_old_versions(storage, "lgbm_rj", keep = 3, dry_run = TRUE)
# Would delete: v1.0.0_20250103

cleanup_old_versions(storage, "lgbm_rj", keep = 3)
# Actually deletes

# List all models
list_all_models(storage)
#    model_name version_count latest_version
# 1:    lgbm_rj             3    v1.2.0_...
# 2:    lgbm_sp             2    v1.1.0_...

# Find models matching pattern
find_models(storage, "lgbm_")
# c("lgbm_rj", "lgbm_sp")

# Check storage usage
get_model_storage_usage(storage, "lgbm_rj")
#           version size_bytes size_mb
# 1: v1.2.0_...    52428800   50.00
# 2: v1.1.0_...    51380224   49.00
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | get_model_path() with version | Correct path |
| TC-002 | get_model_path() without version | Base model path |
| TC-003 | get_artifact_path() | Correct file path |
| TC-004 | list_model_versions() existing | Sorted list |
| TC-005 | list_model_versions() no versions | Empty vector |
| TC-006 | get_latest_model_version() | Newest version |
| TC-007 | get_latest_model_version() with pointer | Pointer target |
| TC-008 | model_version_exists() true | TRUE |
| TC-009 | model_version_exists() false | FALSE |
| TC-010 | create_latest_pointer() | Pointer created |
| TC-011 | cleanup_old_versions() dry_run | Reports only |
| TC-012 | cleanup_old_versions() actual | Deletes versions |
| TC-013 | list_all_models() | All models listed |
| TC-014 | find_models() | Matching models |

---

## Definition of Done

- [ ] All path management functions implemented
- [ ] Version discovery functions working
- [ ] Latest pointer management working
- [ ] Version cleanup working
- [ ] Model discovery functions implemented
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The "latest" pointer uses a simple text file for S3 compatibility
- Version cleanup helps manage storage costs in production
- Consider adding version tagging (e.g., "production", "staging")
- Future: add version comparison between models
- Storage backend must implement `list_directories()` and `delete_directory()`
