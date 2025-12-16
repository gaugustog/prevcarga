# PC-006-01: DataCatalog

**Epic:** [EPIC-01: Data Layer](../epics/EPIC-01-data-layer.md)
**Task Reference:** T-01.6
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Implement the `DataCatalog` R6 class that serves as a central registry for dataset metadata, enabling consistent access patterns and automatic schema validation across all data types.

---

## Acceptance Criteria

- [ ] Catalog module created in `R/data/catalog.R`
- [ ] `DataCatalog` R6 class with dataset registration
- [ ] `register()` stores dataset metadata (path, schema, partitions)
- [ ] `get()` retrieves dataset configuration by name
- [ ] `list_datasets()` returns all registered datasets
- [ ] `validate()` validates data against registered schema
- [ ] Pre-registered datasets for: load, weather, holidays, dst_periods, load_tiers

---

## Technical Specification

### File Location
```
R/data/catalog.R
```

### Dataset Entry Class

```r
#' @title DatasetEntry
#' @description Represents a registered dataset in the catalog
DatasetEntry <- R6::R6Class(
  "DatasetEntry",
  public = list(
    #' @field name Dataset name
    name = NULL,
    #' @field base_path Base path in storage
    base_path = NULL,
    #' @field schema_name Schema name from SCHEMAS constant
    schema_name = NULL,
    #' @field partitions Character vector of partition column names
    partitions = NULL,
    #' @field description Human-readable description
    description = NULL,
    #' @field filename Filename pattern (default: data.parquet)
    filename = NULL,

    #' @description Initialize dataset entry
    initialize = function(name, base_path, schema_name,
                          partitions = character(),
                          description = "",
                          filename = "data.parquet") {
      self$name <- name
      self$base_path <- base_path
      self$schema_name <- schema_name
      self$partitions <- partitions
      self$description <- description
      self$filename <- filename
    }
  )
)
```

### DataCatalog Class

```r
#' @title DataCatalog
#' @description Central registry for dataset metadata
#' @export
DataCatalog <- R6::R6Class(
  "DataCatalog",
  private = list(
    datasets = NULL
  ),
  public = list(
    #' @description Initialize catalog
    initialize = function() {
      private$datasets <- list()
    },

    #' @description Register a dataset
    #' @param name Unique dataset name
    #' @param base_path Base path in storage
    #' @param schema_name Schema name for validation
    #' @param partitions Partition column names
    #' @param description Human-readable description
    #' @param filename Filename pattern
    #' @return Invisible self
    register = function(name, base_path, schema_name,
                        partitions = character(),
                        description = "",
                        filename = "data.parquet") {
      checkmate::assert_string(name)
      checkmate::assert_string(base_path)
      checkmate::assert_string(schema_name)
      checkmate::assert_character(partitions)
      checkmate::assert_string(description)
      checkmate::assert_string(filename)

      if (name %in% names(private$datasets)) {
        warning(sprintf("Dataset '%s' already registered, overwriting", name))
      }

      private$datasets[[name]] <- DatasetEntry$new(
        name = name,
        base_path = base_path,
        schema_name = schema_name,
        partitions = partitions,
        description = description,
        filename = filename
      )

      invisible(self)
    },

    #' @description Get dataset entry by name
    #' @param name Dataset name
    #' @return DatasetEntry or NULL if not found
    get = function(name) {
      checkmate::assert_string(name)
      private$datasets[[name]]
    },

    #' @description Check if dataset is registered
    #' @param name Dataset name
    #' @return Logical
    has = function(name) {
      name %in% names(private$datasets)
    },

    #' @description List all registered datasets
    #' @return Character vector of dataset names
    list_datasets = function() {
      names(private$datasets)
    },

    #' @description Get summary of all datasets
    #' @return data.table with dataset information
    summary = function() {
      if (length(private$datasets) == 0) {
        return(data.table::data.table(
          name = character(),
          base_path = character(),
          schema = character(),
          partitions = character(),
          description = character()
        ))
      }

      data.table::rbindlist(lapply(private$datasets, function(ds) {
        data.table::data.table(
          name = ds$name,
          base_path = ds$base_path,
          schema = ds$schema_name,
          partitions = paste(ds$partitions, collapse = ", "),
          description = ds$description
        )
      }))
    },

    #' @description Validate data against registered schema
    #' @param name Dataset name
    #' @param dt data.table to validate
    #' @param strict Strict mode for schema validation
    #' @return TRUE if valid, throws error otherwise
    validate = function(name, dt, strict = FALSE) {
      checkmate::assert_string(name)
      checkmate::assert_data_table(dt)

      dataset <- self$get(name)
      if (is.null(dataset)) {
        stop(sprintf("Dataset '%s' not registered in catalog", name))
      }

      # Call schema validator
      validate_schema(dt, dataset$schema_name, strict = strict)
    },

    #' @description Build path for a dataset with partition values
    #' @param name Dataset name
    #' @param partition_values Named list of partition values
    #' @return Character path
    build_path = function(name, partition_values = list()) {
      dataset <- self$get(name)
      if (is.null(dataset)) {
        stop(sprintf("Dataset '%s' not registered in catalog", name))
      }

      # Validate partition values
      expected <- dataset$partitions
      provided <- names(partition_values)

      missing <- setdiff(expected, provided)
      if (length(missing) > 0) {
        stop(sprintf(
          "Missing partition values for '%s': %s",
          name, paste(missing, collapse = ", ")
        ))
      }

      # Build path using Hive utilities
      build_hive_path(
        base_path = dataset$base_path,
        partitions = partition_values[expected],  # Preserve order
        filename = dataset$filename
      )
    }
  )
)
```

### Default Catalog Factory

```r
#' Create catalog with pre-registered datasets
#'
#' @return DataCatalog with standard datasets registered
#' @export
create_default_catalog <- function() {
  catalog <- DataCatalog$new()

  # Load data
  catalog$register(
    name = "load",
    base_path = "load",
    schema_name = "carga",
    partitions = c("area_code", "year"),
    description = "Hourly electricity load data by area"
  )

  # Weather forecast
  catalog$register(
    name = "weather_forecast",
    base_path = "weather/forecast",
    schema_name = "weather",
    partitions = c("area_code", "year"),
    description = "Weather forecast data by area"
  )

  # Weather observed
  catalog$register(
    name = "weather_observed",
    base_path = "weather/observed",
    schema_name = "weather",
    partitions = c("area_code", "year"),
    description = "Observed weather data by area"
  )

  # Holidays
  catalog$register(
    name = "holidays",
    base_path = "auxiliary/holidays",
    schema_name = "holidays",
    partitions = c("area_code", "year"),
    description = "Holiday calendar by area"
  )

  # DST periods
  catalog$register(
    name = "dst_periods",
    base_path = "auxiliary/dst_periods",
    schema_name = "dst_periods",
    partitions = character(),  # No partitions
    description = "Daylight saving time period definitions"
  )

  # Load tiers
  catalog$register(
    name = "load_tiers",
    base_path = "auxiliary/load_tiers",
    schema_name = "load_tiers",
    partitions = c("area_code"),
    description = "Load tier configuration by area"
  )

  catalog
}
```

### Usage Example

```r
# Create default catalog
catalog <- create_default_catalog()

# List available datasets
catalog$list_datasets()
# [1] "load" "weather_forecast" "weather_observed" "holidays" ...

# Get dataset info
load_ds <- catalog$get("load")
load_ds$base_path      # "load"
load_ds$partitions     # c("area_code", "year")
load_ds$schema_name    # "carga"

# Build path for specific partition
path <- catalog$build_path("load", list(area_code = "RJ", year = 2024))
# "load/area_code=RJ/year=2024/data.parquet"

# Validate loaded data
dt <- loader$load_carga("RJ", "2024-01-01", "2024-12-31")
catalog$validate("load", dt)  # Throws if invalid

# Get summary
catalog$summary()
#                name            base_path  schema    partitions
# 1:             load                 load   carga area_code, year
# 2: weather_forecast     weather/forecast weather area_code, year
# ...
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Register new dataset | Entry added to catalog |
| TC-002 | Register duplicate name | Warning, overwrites |
| TC-003 | Get existing dataset | Returns DatasetEntry |
| TC-004 | Get non-existent dataset | Returns NULL |
| TC-005 | has() for existing | Returns TRUE |
| TC-006 | has() for non-existent | Returns FALSE |
| TC-007 | list_datasets() | Returns all names |
| TC-008 | summary() empty catalog | Empty data.table |
| TC-009 | summary() with datasets | Populated data.table |
| TC-010 | validate() valid data | Returns TRUE |
| TC-011 | validate() invalid data | Throws error |
| TC-012 | build_path() complete | Correct Hive path |
| TC-013 | build_path() missing partition | Error |
| TC-014 | create_default_catalog() | All standard datasets |

---

## Definition of Done

- [ ] DatasetEntry class implemented
- [ ] DataCatalog class implemented
- [ ] Default catalog factory function
- [ ] Integration with schema validators
- [ ] Integration with Hive path utilities
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The catalog is a metadata registry; actual data loading is done by DataLoader
- Consider adding versioning support for dataset schemas in the future
- The catalog could be extended to support remote dataset discovery
- Integration with DataLoader should use the catalog for path building and validation
