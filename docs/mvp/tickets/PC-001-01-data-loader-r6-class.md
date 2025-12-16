# PC-001-01: DataLoader R6 Class

**Epic:** [EPIC-01: Data Layer](../epics/EPIC-01-data-layer.md)
**Task Reference:** T-01.1
**Priority:** High
**Estimated Effort:** 3 days

---

## Summary

Implement the `DataLoader` R6 class that provides a unified interface for loading time series data from the storage backend, supporting Hive-style partitioned parquet files.

---

## Acceptance Criteria

- [ ] `DataLoader` R6 class is created in `R/data/loader.R`
- [ ] Constructor accepts a `StorageBackend` instance via dependency injection
- [ ] `load_carga()` loads load data for specified areas and date range
- [ ] `load_weather()` loads weather data (forecast or observed) for areas
- [ ] `load_holidays()` loads holiday calendar for areas and years
- [ ] `load_dst_periods()` loads daylight saving time periods
- [ ] `load_load_tiers()` loads load tier configuration for areas
- [ ] All methods support Hive-style path resolution
- [ ] Data is returned as `data.table` objects
- [ ] Methods handle missing partitions gracefully (warning, not error)

---

## Technical Specification

### File Location
```
R/data/loader.R
```

### Class Definition

```r
#' @title DataLoader
#' @description Unified data loader using storage backend abstraction
#' @export
DataLoader <- R6::R6Class(
  "DataLoader",
  private = list(
    storage = NULL,

    #' Build Hive-style path for partitioned data
    build_path = function(base, partitions) {
      path <- base
      for (name in names(partitions)) {
        path <- file.path(path, sprintf("%s=%s", name, partitions[[name]]))
      }
      file.path(path, "data.parquet")
    },

    #' Load multiple partitions and combine
    load_partitions = function(base, partition_grid) {
      results <- lapply(seq_len(nrow(partition_grid)), function(i) {
        partitions <- as.list(partition_grid[i, ])
        path <- private$build_path(base, partitions)

        if (private$storage$exists(path)) {
          dt <- private$storage$read_parquet(path)
          # Add partition columns if not present
          for (name in names(partitions)) {
            if (!name %in% names(dt)) {
              dt[[name]] <- partitions[[name]]
            }
          }
          dt
        } else {
          warning(sprintf("Partition not found: %s", path))
          NULL
        }
      })

      results <- Filter(Negate(is.null), results)
      if (length(results) == 0) return(data.table::data.table())
      data.table::rbindlist(results, fill = TRUE)
    }
  ),

  public = list(
    #' @description Initialize DataLoader
    #' @param storage_backend A StorageBackend instance
    initialize = function(storage_backend) {
      checkmate::assert_class(storage_backend, "StorageBackend")
      private$storage <- storage_backend
    },

    #' @description Load electricity load data
    #' @param areas Character vector of area codes
    #' @param start_date Start date (Date or character)
    #' @param end_date End date (Date or character)
    #' @return data.table with load data
    load_carga = function(areas, start_date, end_date) {
      checkmate::assert_character(areas, min.len = 1)
      start_date <- as.Date(start_date)
      end_date <- as.Date(end_date)

      years <- seq(
        as.integer(format(start_date, "%Y")),
        as.integer(format(end_date, "%Y"))
      )

      partition_grid <- data.table::CJ(
        area_code = areas,
        year = years
      )

      dt <- private$load_partitions("load", partition_grid)

      if (nrow(dt) > 0) {
        dt <- dt[DataHora >= start_date & DataHora <= end_date]
      }

      dt
    },

    #' @description Load weather data
    #' @param areas Character vector of area codes
    #' @param type Weather type: "forecast" or "observed"
    #' @param start_date Start date
    #' @param end_date End date
    #' @return data.table with weather data
    load_weather = function(areas, type = c("forecast", "observed"),
                           start_date, end_date) {
      type <- match.arg(type)
      checkmate::assert_character(areas, min.len = 1)
      start_date <- as.Date(start_date)
      end_date <- as.Date(end_date)

      years <- seq(
        as.integer(format(start_date, "%Y")),
        as.integer(format(end_date, "%Y"))
      )

      partition_grid <- data.table::CJ(
        area_code = areas,
        year = years
      )

      base_path <- file.path("weather", type)
      dt <- private$load_partitions(base_path, partition_grid)

      if (nrow(dt) > 0) {
        dt <- dt[DataHora >= start_date & DataHora <= end_date]
      }

      dt
    },

    #' @description Load holiday calendar
    #' @param areas Character vector of area codes
    #' @param years Integer vector of years
    #' @return data.table with holiday data
    load_holidays = function(areas, years) {
      checkmate::assert_character(areas, min.len = 1)
      checkmate::assert_integerish(years, min.len = 1)

      partition_grid <- data.table::CJ(
        area_code = areas,
        year = as.integer(years)
      )

      private$load_partitions("auxiliary/holidays", partition_grid)
    },

    #' @description Load DST periods
    #' @return data.table with DST period definitions
    load_dst_periods = function() {
      path <- "auxiliary/dst_periods/data.parquet"
      if (private$storage$exists(path)) {
        private$storage$read_parquet(path)
      } else {
        warning("DST periods file not found")
        data.table::data.table()
      }
    },

    #' @description Load load tier configuration
    #' @param areas Character vector of area codes
    #' @return data.table with load tier definitions
    load_load_tiers = function(areas) {
      checkmate::assert_character(areas, min.len = 1)

      results <- lapply(areas, function(area) {
        path <- sprintf("auxiliary/load_tiers/area_code=%s/data.parquet", area)
        if (private$storage$exists(path)) {
          dt <- private$storage$read_parquet(path)
          dt$area_code <- area
          dt
        } else {
          NULL
        }
      })

      results <- Filter(Negate(is.null), results)
      if (length(results) == 0) return(data.table::data.table())
      data.table::rbindlist(results, fill = TRUE)
    }
  )
)
```

### Dependencies

- `R6` - R6 class system
- `data.table` - Data manipulation
- `checkmate` - Argument validation
- `StorageBackend` from EPIC-00

### Usage Example

```r
# Create storage backend
storage <- LocalStorageBackend$new(base_path = "./data")

# Create data loader
loader <- DataLoader$new(storage_backend = storage)

# Load electricity load data
carga <- loader$load_carga(
  areas = c("RJ", "SP"),
  start_date = "2024-01-01",
  end_date = "2024-12-31"
)

# Load weather forecast
weather <- loader$load_weather(
  areas = c("RJ", "SP"),
  type = "forecast",
  start_date = "2024-01-01",
  end_date = "2024-12-31"
)

# Load holidays
holidays <- loader$load_holidays(
  areas = c("RJ", "SP"),
  years = c(2024, 2025)
)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with valid StorageBackend | Success |
| TC-002 | Initialize with invalid object | Error |
| TC-003 | load_carga with existing partitions | Returns data.table |
| TC-004 | load_carga with missing partition | Warning, partial data |
| TC-005 | load_carga with date filtering | Only dates in range |
| TC-006 | load_weather forecast type | Returns forecast data |
| TC-007 | load_weather observed type | Returns observed data |
| TC-008 | load_holidays multiple areas/years | Combined data.table |
| TC-009 | load_dst_periods | Returns DST data |
| TC-010 | load_load_tiers | Returns tier config |

---

## Definition of Done

- [ ] R6 class implemented with all methods
- [ ] roxygen2 documentation complete
- [ ] Unit tests written (≥80% coverage)
- [ ] Integration test with mock StorageBackend
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- This class depends on `StorageBackend` from EPIC-00
- Hive partitioning utilities (PC-002-01) can be extracted if complexity grows
- Schema validation (PC-003-01) should be called after loading
