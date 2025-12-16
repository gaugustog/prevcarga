# PC-002-01: Hive Partitioning Utilities

**Epic:** [EPIC-01: Data Layer](../epics/EPIC-01-data-layer.md)
**Task Reference:** T-01.2
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Implement utility functions for working with Hive-style partitioned paths, enabling efficient path construction and parsing for partitioned parquet datasets.

---

## Acceptance Criteria

- [ ] Utility module created in `R/utils/hive.R`
- [ ] `build_hive_path()` constructs paths from base and partition values
- [ ] `parse_hive_path()` extracts partition values from paths
- [ ] `expand_partition_grid()` generates all partition combinations
- [ ] Functions handle edge cases (empty partitions, special characters)
- [ ] All functions are documented with roxygen2

---

## Technical Specification

### File Location
```
R/utils/hive.R
```

### Function Definitions

```r
#' Build a Hive-style partitioned path
#'
#' @param base_path Base directory path
#' @param partitions Named list of partition key-value pairs
#' @param filename Filename to append (default: "data.parquet")
#' @return Character string with full path
#' @export
#' @examples
#' build_hive_path("load", list(area_code = "RJ", year = 2024))
#' # Returns: "load/area_code=RJ/year=2024/data.parquet"
build_hive_path <- function(base_path, partitions, filename = "data.parquet") {
  checkmate::assert_string(base_path)
  checkmate::assert_list(partitions, names = "named")
  checkmate::assert_string(filename)

  path <- base_path
  for (name in names(partitions)) {
    value <- partitions[[name]]
    # Handle NULL or NA values
    if (is.null(value) || is.na(value)) {
      stop(sprintf("Partition value for '%s' cannot be NULL or NA", name))
    }
    path <- file.path(path, sprintf("%s=%s", name, as.character(value)))
  }

  file.path(path, filename)
}


#' Parse partition values from a Hive-style path
#'
#' @param path Character string with Hive-style path
#' @return Named list of partition values
#' @export
#' @examples
#' parse_hive_path("load/area_code=RJ/year=2024/data.parquet")
#' # Returns: list(area_code = "RJ", year = "2024")
parse_hive_path <- function(path) {
  checkmate::assert_string(path)

  # Split path into components
  components <- strsplit(path, "/|\\\\")[[1]]

  # Find partition components (contain "=")
  partition_components <- components[grepl("=", components)]

  if (length(partition_components) == 0) {
    return(list())
  }

  # Parse each partition
  partitions <- list()
  for (comp in partition_components) {
    parts <- strsplit(comp, "=", fixed = TRUE)[[1]]
    if (length(parts) == 2) {
      partitions[[parts[1]]] <- parts[2]
    }
  }

  partitions
}


#' Expand partition grid to all combinations
#'
#' @param ... Named vectors of partition values
#' @return data.table with all partition combinations
#' @export
#' @examples
#' expand_partition_grid(area_code = c("RJ", "SP"), year = c(2023, 2024))
#' # Returns data.table with 4 rows (all combinations)
expand_partition_grid <- function(...) {
  args <- list(...)
  checkmate::assert_list(args, names = "named", min.len = 1)

  data.table::CJ(..., sorted = FALSE)
}


#' List all partition values in a directory
#'
#' @param storage StorageBackend instance
#' @param base_path Base directory path
#' @param partition_name Name of the partition to list
#' @return Character vector of partition values
#' @export
list_partition_values <- function(storage, base_path, partition_name) {
  checkmate::assert_class(storage, "StorageBackend")
  checkmate::assert_string(base_path)
  checkmate::assert_string(partition_name)

  dirs <- storage$list_files(base_path)
  pattern <- sprintf("^%s=(.+)$", partition_name)

  matches <- regmatches(dirs, regexec(pattern, dirs))
  values <- vapply(matches, function(m) {
    if (length(m) == 2) m[2] else NA_character_
  }, character(1))

  values[!is.na(values)]
}


#' Validate partition values
#'
#' @param partitions Named list of partition values
#' @param allowed Named list of allowed values per partition (optional)
#' @return TRUE if valid, throws error otherwise
#' @export
validate_partitions <- function(partitions, allowed = NULL) {
  checkmate::assert_list(partitions, names = "named")

  for (name in names(partitions)) {
    value <- partitions[[name]]

    # Check for invalid characters
    if (grepl("[/\\\\=]", as.character(value))) {
      stop(sprintf("Partition value for '%s' contains invalid characters", name))
    }

    # Check against allowed values if provided
    if (!is.null(allowed) && name %in% names(allowed)) {
      if (!value %in% allowed[[name]]) {
        stop(sprintf(
          "Invalid value '%s' for partition '%s'. Allowed: %s",
          value, name, paste(allowed[[name]], collapse = ", ")
        ))
      }
    }
  }

  invisible(TRUE)
}
```

### Dependencies

- `checkmate` - Argument validation
- `data.table` - CJ for grid expansion

### Usage Examples

```r
# Build path for a single partition
path <- build_hive_path(
  "load",
  list(area_code = "RJ", year = 2024)
)
# "load/area_code=RJ/year=2024/data.parquet"

# Parse path back to partitions
partitions <- parse_hive_path(path)
# list(area_code = "RJ", year = "2024")

# Generate all partition combinations
grid <- expand_partition_grid(
  area_code = c("RJ", "SP", "MG"),
  year = c(2023, 2024)
)
# data.table with 6 rows

# Validate partitions
validate_partitions(
  list(area_code = "RJ"),
  allowed = list(area_code = c("RJ", "SP", "MG"))
)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | build_hive_path with single partition | Correct path |
| TC-002 | build_hive_path with multiple partitions | Correct path |
| TC-003 | build_hive_path with NULL partition | Error |
| TC-004 | parse_hive_path valid path | Correct list |
| TC-005 | parse_hive_path no partitions | Empty list |
| TC-006 | expand_partition_grid 2x2 | 4 rows |
| TC-007 | expand_partition_grid 3x3x2 | 18 rows |
| TC-008 | validate_partitions valid | TRUE |
| TC-009 | validate_partitions invalid chars | Error |
| TC-010 | validate_partitions not in allowed | Error |

---

## Definition of Done

- [ ] All utility functions implemented
- [ ] roxygen2 documentation complete
- [ ] Unit tests written (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- These utilities are used internally by `DataLoader` (PC-001-01)
- Consider memoization for `list_partition_values` if called frequently
- Path separator handling should work on both Windows and Unix
