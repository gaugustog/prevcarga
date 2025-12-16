# PC-018-03: Semantic Versioning

**Epic:** [EPIC-03: Model Layer - Infrastructure](../epics/EPIC-03-model-layer-part1.md)
**Task Reference:** T-03.3
**Priority:** Medium
**Estimated Effort:** 0.5 days

---

## Summary

Implement the `ModelVersion` R6 class for semantic versioning of model artifacts, including version parsing, comparison, and timestamp support for tracking model evolution.

---

## Acceptance Criteria

- [ ] Versioning module created in `R/models/versioning.R`
- [ ] `ModelVersion` R6 class with major.minor.patch components
- [ ] `to_string()` generates version string with timestamp
- [ ] Version parsing from string
- [ ] `bump_major()`, `bump_minor()`, `bump_patch()` methods
- [ ] `compare()` for version comparison
- [ ] Support for timestamp tracking

---

## Technical Specification

### File Location
```
R/models/versioning.R
```

### ModelVersion Class

```r
#' @title ModelVersion
#' @description Semantic versioning for model artifacts
#'
#' Implements semantic versioning (major.minor.patch) with timestamp
#' support for model artifact management. Enables version comparison
#' and chronological tracking of model iterations.
#'
#' @details
#' Version format: `v{major}.{minor}.{patch}_{timestamp}`
#' Example: `v1.2.3_20250117_143052`
#'
#' @export
ModelVersion <- R6::R6Class(
  "ModelVersion",
  public = list(
    #' @field major Major version (breaking changes)
    major = 0L,

    #' @field minor Minor version (new features)
    minor = 0L,

    #' @field patch Patch version (bug fixes)
    patch = 0L,

    #' @field timestamp Creation timestamp
    timestamp = NULL,

    #' @field metadata Optional metadata list
    metadata = NULL,

    #' @description Initialize version
    #' @param version_string Optional version string to parse
    #' @param major Major version number
    #' @param minor Minor version number
    #' @param patch Patch version number
    #' @param timestamp Timestamp (default: current time)
    initialize = function(version_string = NULL,
                          major = 0L, minor = 0L, patch = 0L,
                          timestamp = NULL) {
      if (!is.null(version_string)) {
        parsed <- private$parse_version(version_string)
        self$major <- parsed$major
        self$minor <- parsed$minor
        self$patch <- parsed$patch
        self$timestamp <- parsed$timestamp
      } else {
        self$major <- as.integer(major)
        self$minor <- as.integer(minor)
        self$patch <- as.integer(patch)
        self$timestamp <- timestamp %||% Sys.time()
      }

      self$metadata <- list()
    },

    #' @description Convert to string representation
    #' @param include_timestamp Include timestamp in string
    #' @return Version string
    to_string = function(include_timestamp = TRUE) {
      base <- sprintf("v%d.%d.%d", self$major, self$minor, self$patch)

      if (include_timestamp && !is.null(self$timestamp)) {
        sprintf("%s_%s", base, format(self$timestamp, "%Y%m%d_%H%M%S"))
      } else {
        base
      }
    },

    #' @description Get short version (without timestamp)
    #' @return Short version string
    short = function() {
      sprintf("v%d.%d.%d", self$major, self$minor, self$patch)
    },

    #' @description Bump major version (resets minor and patch)
    #' @return New ModelVersion instance
    bump_major = function() {
      ModelVersion$new(
        major = self$major + 1L,
        minor = 0L,
        patch = 0L
      )
    },

    #' @description Bump minor version (resets patch)
    #' @return New ModelVersion instance
    bump_minor = function() {
      ModelVersion$new(
        major = self$major,
        minor = self$minor + 1L,
        patch = 0L
      )
    },

    #' @description Bump patch version
    #' @return New ModelVersion instance
    bump_patch = function() {
      ModelVersion$new(
        major = self$major,
        minor = self$minor,
        patch = self$patch + 1L
      )
    },

    #' @description Compare to another version
    #' @param other ModelVersion instance or version string
    #' @return Integer: -1 (less), 0 (equal), 1 (greater)
    compare = function(other) {
      if (is.character(other)) {
        other <- ModelVersion$new(version_string = other)
      }

      checkmate::assert_class(other, "ModelVersion")

      # Compare major
      if (self$major != other$major) {
        return(sign(self$major - other$major))
      }

      # Compare minor
      if (self$minor != other$minor) {
        return(sign(self$minor - other$minor))
      }

      # Compare patch
      if (self$patch != other$patch) {
        return(sign(self$patch - other$patch))
      }

      # Equal version - compare timestamps if available
      if (!is.null(self$timestamp) && !is.null(other$timestamp)) {
        return(sign(as.numeric(self$timestamp) - as.numeric(other$timestamp)))
      }

      0L
    },

    #' @description Check if this version is newer than other
    #' @param other ModelVersion or string
    #' @return Logical
    is_newer_than = function(other) {
      self$compare(other) > 0
    },

    #' @description Check if this version is older than other
    #' @param other ModelVersion or string
    #' @return Logical
    is_older_than = function(other) {
      self$compare(other) < 0
    },

    #' @description Check if versions are equal (ignoring timestamp)
    #' @param other ModelVersion or string
    #' @return Logical
    equals = function(other) {
      if (is.character(other)) {
        other <- ModelVersion$new(version_string = other)
      }

      self$major == other$major &&
        self$minor == other$minor &&
        self$patch == other$patch
    },

    #' @description Add metadata to version
    #' @param key Metadata key
    #' @param value Metadata value
    #' @return Invisible self
    add_metadata = function(key, value) {
      self$metadata[[key]] <- value
      invisible(self)
    },

    #' @description Get version as list
    #' @return List with version components
    as_list = function() {
      list(
        major = self$major,
        minor = self$minor,
        patch = self$patch,
        timestamp = self$timestamp,
        string = self$to_string(),
        metadata = self$metadata
      )
    },

    #' @description Print version
    print = function() {
      cat(sprintf("<ModelVersion: %s>\n", self$to_string()))
      if (length(self$metadata) > 0) {
        cat("  Metadata:\n")
        for (key in names(self$metadata)) {
          cat(sprintf("    %s: %s\n", key, self$metadata[[key]]))
        }
      }
      invisible(self)
    }
  ),

  private = list(
    #' Parse version string
    parse_version = function(version_string) {
      checkmate::assert_string(version_string)

      # Pattern: v{major}.{minor}.{patch}[_{timestamp}]
      pattern <- "^v?(\\d+)\\.(\\d+)\\.(\\d+)(?:_(\\d{8}_\\d{6}))?$"

      if (!grepl(pattern, version_string)) {
        stop(sprintf(
          "Invalid version string: '%s'. Expected format: v1.2.3 or v1.2.3_20250117_143052",
          version_string
        ))
      }

      matches <- regmatches(version_string, regexec(pattern, version_string))[[1]]

      timestamp <- if (length(matches) >= 5 && nchar(matches[5]) > 0) {
        as.POSIXct(matches[5], format = "%Y%m%d_%H%M%S")
      } else {
        NULL
      }

      list(
        major = as.integer(matches[2]),
        minor = as.integer(matches[3]),
        patch = as.integer(matches[4]),
        timestamp = timestamp
      )
    }
  )
)
```

### Utility Functions

```r
#' Create a new version
#'
#' @param major Major version
#' @param minor Minor version
#' @param patch Patch version
#' @return ModelVersion instance
#' @export
model_version <- function(major = 0, minor = 0, patch = 0) {
  ModelVersion$new(major = major, minor = minor, patch = patch)
}


#' Parse version string
#'
#' @param version_string Version string to parse
#' @return ModelVersion instance
#' @export
parse_model_version <- function(version_string) {
  ModelVersion$new(version_string = version_string)
}


#' Compare two versions
#'
#' @param v1 First version (ModelVersion or string)
#' @param v2 Second version (ModelVersion or string)
#' @return Integer: -1, 0, or 1
#' @export
compare_versions <- function(v1, v2) {
  if (is.character(v1)) v1 <- parse_model_version(v1)
  if (is.character(v2)) v2 <- parse_model_version(v2)

  v1$compare(v2)
}


#' Get the latest version from a list
#'
#' @param versions Character vector of version strings
#' @return Latest version string
#' @export
get_latest_version <- function(versions) {
  if (length(versions) == 0) return(NULL)

  parsed <- lapply(versions, parse_model_version)

  latest_idx <- 1
  for (i in seq_along(parsed)[-1]) {
    if (parsed[[i]]$is_newer_than(parsed[[latest_idx]])) {
      latest_idx <- i
    }
  }

  versions[latest_idx]
}


#' Sort versions from oldest to newest
#'
#' @param versions Character vector of version strings
#' @return Sorted version strings
#' @export
sort_versions <- function(versions) {
  if (length(versions) <= 1) return(versions)

  parsed <- lapply(versions, parse_model_version)

  # Create comparison matrix
  order_idx <- order(sapply(parsed, function(v) {
    v$major * 1e6 + v$minor * 1e3 + v$patch +
      as.numeric(v$timestamp %||% 0) / 1e10
  }))

  versions[order_idx]
}
```

### Usage Example

```r
# Create versions
v1 <- model_version(1, 0, 0)
v1$to_string()
# "v1.0.0_20250117_143052"

v1$short()
# "v1.0.0"

# Bump versions
v2 <- v1$bump_minor()
v2$short()
# "v1.1.0"

v3 <- v2$bump_patch()
v3$short()
# "v1.1.1"

v4 <- v3$bump_major()
v4$short()
# "v2.0.0"

# Parse from string
v_parsed <- parse_model_version("v1.2.3_20250117_143052")
v_parsed$major  # 1
v_parsed$minor  # 2
v_parsed$patch  # 3

# Compare versions
v1$compare(v2)  # -1 (v1 < v2)
v2$is_newer_than(v1)  # TRUE

# Find latest
versions <- c("v1.0.0", "v2.1.0", "v1.5.0", "v2.0.1")
get_latest_version(versions)
# "v2.1.0"

# Sort versions
sort_versions(versions)
# c("v1.0.0", "v1.5.0", "v2.0.1", "v2.1.0")

# Add metadata
v1$add_metadata("trained_by", "scheduler")
v1$add_metadata("dataset_hash", "abc123")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Create with defaults | v0.0.0 with timestamp |
| TC-002 | Create with values | Correct major.minor.patch |
| TC-003 | Parse valid string | Correct components |
| TC-004 | Parse string with timestamp | Timestamp parsed |
| TC-005 | Parse invalid string | Error |
| TC-006 | to_string() with timestamp | Full format |
| TC-007 | short() | Without timestamp |
| TC-008 | bump_major() | Minor/patch reset |
| TC-009 | bump_minor() | Patch reset |
| TC-010 | bump_patch() | Only patch incremented |
| TC-011 | compare() less than | Returns -1 |
| TC-012 | compare() greater than | Returns 1 |
| TC-013 | compare() equal | Returns 0 |
| TC-014 | get_latest_version() | Correct latest |
| TC-015 | sort_versions() | Correct order |

---

## Definition of Done

- [ ] ModelVersion class implemented
- [ ] Version parsing working
- [ ] Comparison methods working
- [ ] Utility functions exported
- [ ] Metadata support working
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥95% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Version format follows semantic versioning (semver.org)
- Timestamp ensures uniqueness even with same version numbers
- Metadata can store training context (dataset hash, trainer, etc.)
- Consider adding pre-release suffix support (e.g., v1.0.0-beta)
- Version comparison ignores timestamp unless versions are equal
