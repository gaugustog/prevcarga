# PC-008-01: Data Layer Tests

**Epic:** [EPIC-01: Data Layer](../epics/EPIC-01-data-layer.md)
**Task Reference:** T-01.8
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Create comprehensive test suite for the Data Layer (EPIC-01), covering DataLoader, validators, preprocessors, catalog, and area codes with ≥80% coverage target.

---

## Acceptance Criteria

- [ ] Test file created: `tests/testthat/test-data-loader.R`
- [ ] Test file created: `tests/testthat/test-validators.R`
- [ ] Test file created: `tests/testthat/test-preprocessors.R`
- [ ] Test file created: `tests/testthat/test-catalog.R`
- [ ] Test file created: `tests/testthat/test-areas.R`
- [ ] Test file created: `tests/testthat/test-hive.R`
- [ ] Mock StorageBackend for isolated testing
- [ ] Test fixtures for sample data
- [ ] Overall coverage ≥80% for Data Layer

---

## Technical Specification

### Test Directory Structure
```
tests/
├── testthat/
│   ├── test-data-loader.R
│   ├── test-validators.R
│   ├── test-preprocessors.R
│   ├── test-catalog.R
│   ├── test-areas.R
│   ├── test-hive.R
│   └── fixtures/
│       ├── sample_carga.parquet
│       ├── sample_weather.parquet
│       └── sample_holidays.parquet
└── testthat.R
```

### Mock Storage Backend

```r
# tests/testthat/helper-mocks.R

#' Mock Storage Backend for Testing
MockStorageBackend <- R6::R6Class(
  "MockStorageBackend",
  inherit = StorageBackend,
  private = list(
    files = NULL,
    data = NULL
  ),
  public = list(
    initialize = function() {
      private$files <- list()
      private$data <- list()
    },

    #' Add mock file
    add_file = function(path, dt) {
      private$files[[path]] <- TRUE
      private$data[[path]] <- dt
    },

    #' StorageBackend interface
    exists = function(path) {
      path %in% names(private$files)
    },

    read_parquet = function(path) {
      if (!self$exists(path)) {
        stop(sprintf("File not found: %s", path))
      }
      data.table::copy(private$data[[path]])
    },

    write_parquet = function(dt, path) {
      private$files[[path]] <- TRUE
      private$data[[path]] <- data.table::copy(dt)
    },

    list_files = function(path) {
      pattern <- sprintf("^%s/[^/]+$", path)
      grep(pattern, names(private$files), value = TRUE)
    }
  )
)
```

### Test Fixtures

```r
# tests/testthat/helper-fixtures.R

#' Create sample load data
create_sample_carga <- function(n = 24, area_code = "RJ") {
  data.table::data.table(
    DataHora = seq(
      as.POSIXct("2024-01-01 00:00:00", tz = "America/Sao_Paulo"),
      by = "hour",
      length.out = n
    ),
    CargaGlobal = runif(n, 1000, 5000),
    area_code = area_code
  )
}

#' Create sample weather data
create_sample_weather <- function(n = 24, area_code = "RJ") {
  data.table::data.table(
    DataHora = seq(
      as.POSIXct("2024-01-01 00:00:00", tz = "America/Sao_Paulo"),
      by = "hour",
      length.out = n
    ),
    temperatura = runif(n, 15, 35),
    umidade = runif(n, 30, 100),
    area_code = area_code
  )
}

#' Create sample holiday data
create_sample_holidays <- function(area_code = "RJ", year = 2024) {
  data.table::data.table(
    Data = as.Date(c("2024-01-01", "2024-04-21", "2024-05-01")),
    nome = c("Ano Novo", "Tiradentes", "Dia do Trabalho"),
    tipo = c("nacional", "nacional", "nacional"),
    area_code = area_code
  )
}

#' Create data with missing values
create_data_with_gaps <- function(gap_indices = c(5, 10, 15)) {
  dt <- create_sample_carga(24)
  dt[gap_indices, CargaGlobal := NA]
  dt
}
```

### DataLoader Tests

```r
# tests/testthat/test-data-loader.R

describe("DataLoader", {
  it("initializes with valid StorageBackend", {
    storage <- MockStorageBackend$new()
    loader <- DataLoader$new(storage_backend = storage)
    expect_s3_class(loader, "DataLoader")
  })

  it("throws error with invalid storage", {
    expect_error(
      DataLoader$new(storage_backend = "not_a_backend"),
      "storage_backend"
    )
  })

  describe("load_carga", {
    it("loads data for single area and year", {
      storage <- MockStorageBackend$new()
      storage$add_file(
        "load/area_code=RJ/year=2024/data.parquet",
        create_sample_carga(24, "RJ")
      )

      loader <- DataLoader$new(storage)
      result <- loader$load_carga(
        areas = "RJ",
        start_date = "2024-01-01",
        end_date = "2024-01-01"
      )

      expect_s3_class(result, "data.table")
      expect_equal(nrow(result), 24)
      expect_equal(unique(result$area_code), "RJ")
    })

    it("loads data for multiple areas", {
      storage <- MockStorageBackend$new()
      storage$add_file(
        "load/area_code=RJ/year=2024/data.parquet",
        create_sample_carga(24, "RJ")
      )
      storage$add_file(
        "load/area_code=SP/year=2024/data.parquet",
        create_sample_carga(24, "SP")
      )

      loader <- DataLoader$new(storage)
      result <- loader$load_carga(
        areas = c("RJ", "SP"),
        start_date = "2024-01-01",
        end_date = "2024-01-01"
      )

      expect_equal(nrow(result), 48)
      expect_setequal(unique(result$area_code), c("RJ", "SP"))
    })

    it("warns on missing partition", {
      storage <- MockStorageBackend$new()
      loader <- DataLoader$new(storage)

      expect_warning(
        loader$load_carga("XX", "2024-01-01", "2024-01-01"),
        "Partition not found"
      )
    })

    it("filters by date range", {
      storage <- MockStorageBackend$new()
      # Add 48 hours of data
      dt <- create_sample_carga(48, "RJ")
      dt$DataHora <- seq(
        as.POSIXct("2024-01-01 00:00:00"),
        by = "hour",
        length.out = 48
      )
      storage$add_file("load/area_code=RJ/year=2024/data.parquet", dt)

      loader <- DataLoader$new(storage)
      result <- loader$load_carga("RJ", "2024-01-01", "2024-01-01")

      # Should only have 24 hours (first day)
      expect_lte(nrow(result), 24)
    })
  })

  describe("load_weather", {
    it("loads forecast data", {
      storage <- MockStorageBackend$new()
      storage$add_file(
        "weather/forecast/area_code=RJ/year=2024/data.parquet",
        create_sample_weather(24, "RJ")
      )

      loader <- DataLoader$new(storage)
      result <- loader$load_weather(
        areas = "RJ",
        type = "forecast",
        start_date = "2024-01-01",
        end_date = "2024-01-01"
      )

      expect_s3_class(result, "data.table")
      expect_true("temperatura" %in% names(result))
    })

    it("loads observed data", {
      storage <- MockStorageBackend$new()
      storage$add_file(
        "weather/observed/area_code=RJ/year=2024/data.parquet",
        create_sample_weather(24, "RJ")
      )

      loader <- DataLoader$new(storage)
      result <- loader$load_weather(
        areas = "RJ",
        type = "observed",
        start_date = "2024-01-01",
        end_date = "2024-01-01"
      )

      expect_s3_class(result, "data.table")
    })
  })

  describe("load_holidays", {
    it("loads holidays for area and year", {
      storage <- MockStorageBackend$new()
      storage$add_file(
        "auxiliary/holidays/area_code=RJ/year=2024/data.parquet",
        create_sample_holidays("RJ", 2024)
      )

      loader <- DataLoader$new(storage)
      result <- loader$load_holidays(areas = "RJ", years = 2024)

      expect_s3_class(result, "data.table")
      expect_true("Data" %in% names(result))
      expect_true("nome" %in% names(result))
    })
  })

  describe("load_dst_periods", {
    it("loads DST periods", {
      storage <- MockStorageBackend$new()
      storage$add_file(
        "auxiliary/dst_periods/data.parquet",
        data.table::data.table(
          inicio = as.Date("2024-11-03"),
          fim = as.Date("2025-02-16"),
          ano = 2024L
        )
      )

      loader <- DataLoader$new(storage)
      result <- loader$load_dst_periods()

      expect_s3_class(result, "data.table")
      expect_true("inicio" %in% names(result))
    })

    it("warns when DST file not found", {
      storage <- MockStorageBackend$new()
      loader <- DataLoader$new(storage)

      expect_warning(
        result <- loader$load_dst_periods(),
        "DST periods file not found"
      )
      expect_equal(nrow(result), 0)
    })
  })
})
```

### Validator Tests

```r
# tests/testthat/test-validators.R

describe("Schema Validators", {
  describe("validate_carga_schema", {
    it("accepts valid carga data", {
      dt <- create_sample_carga()
      expect_true(validate_carga_schema(dt))
    })

    it("rejects missing required columns", {
      dt <- data.table::data.table(DataHora = Sys.time())
      expect_error(
        validate_carga_schema(dt),
        "Missing required columns"
      )
    })

    it("rejects wrong column types", {
      dt <- data.table::data.table(
        DataHora = "not a datetime",
        CargaGlobal = 1000,
        area_code = "RJ"
      )
      expect_error(
        validate_carga_schema(dt),
        "expected POSIXct"
      )
    })

    it("allows extra columns in non-strict mode", {
      dt <- create_sample_carga()
      dt$extra_column <- "test"
      expect_true(validate_carga_schema(dt, strict = FALSE))
    })

    it("rejects extra columns in strict mode", {
      dt <- create_sample_carga()
      dt$extra_column <- "test"
      expect_error(
        validate_carga_schema(dt, strict = TRUE),
        "Unexpected columns"
      )
    })
  })

  describe("validate_constraints", {
    it("accepts valid constraints", {
      dt <- create_sample_carga()
      expect_true(validate_constraints(dt, list(
        list(column = "CargaGlobal", min = 0, max = 100000)
      )))
    })

    it("rejects values below minimum", {
      dt <- create_sample_carga()
      dt[1, CargaGlobal := -100]
      expect_error(
        validate_constraints(dt, list(
          list(column = "CargaGlobal", min = 0)
        )),
        "below minimum"
      )
    })

    it("rejects values above maximum", {
      dt <- create_sample_carga()
      dt[1, CargaGlobal := 1000000]
      expect_error(
        validate_constraints(dt, list(
          list(column = "CargaGlobal", max = 100000)
        )),
        "above maximum"
      )
    })
  })
})
```

### Preprocessor Tests

```r
# tests/testthat/test-preprocessors.R

describe("Imputers", {
  describe("LOCFImputer", {
    it("fills gaps with previous value", {
      dt <- create_data_with_gaps(c(5))
      imputer <- LOCFImputer$new()

      result <- imputer$impute(dt, "CargaGlobal")

      expect_false(is.na(result[5, CargaGlobal]))
      expect_equal(result[5, CargaGlobal], result[4, CargaGlobal])
    })

    it("leaves NA at start unchanged", {
      dt <- create_sample_carga()
      dt[1, CargaGlobal := NA]
      imputer <- LOCFImputer$new()

      result <- imputer$impute(dt, "CargaGlobal")

      expect_true(is.na(result[1, CargaGlobal]))
    })
  })

  describe("LinearImputer", {
    it("interpolates between known values", {
      dt <- create_sample_carga()
      dt[5, CargaGlobal := NA]
      before <- dt[4, CargaGlobal]
      after <- dt[6, CargaGlobal]
      imputer <- LinearImputer$new()

      result <- imputer$impute(dt, "CargaGlobal")

      expect_false(is.na(result[5, CargaGlobal]))
      # Value should be between neighbors
      expect_gte(result[5, CargaGlobal], min(before, after))
      expect_lte(result[5, CargaGlobal], max(before, after))
    })
  })

  describe("ImputerChain", {
    it("applies imputers in sequence", {
      dt <- create_data_with_gaps(c(5, 10, 15))
      chain <- ImputerChain$new()$
        add(LinearImputer$new())$
        add(LOCFImputer$new())

      result <- chain$impute(dt, "CargaGlobal")

      expect_equal(sum(is.na(result$CargaGlobal)), 0)
    })

    it("stops when no NA remain", {
      dt <- create_sample_carga()  # No NAs
      chain <- create_default_imputer_chain()

      expect_message(
        chain$impute(dt, "CargaGlobal"),
        "No missing values"
      )
    })
  })
})

describe("Resampler", {
  it("upsamples hourly to semi-hourly", {
    dt <- create_sample_carga(24)
    resampler <- Resampler$new(method = "linear")

    result <- resampler$upsample(dt, "CargaGlobal")

    expect_equal(nrow(result), 47)  # 24 hours -> 47 half-hours
  })

  it("downsamples semi-hourly to hourly", {
    dt <- create_sample_carga(48)
    dt$DataHora <- seq(
      as.POSIXct("2024-01-01 00:00:00"),
      by = "30 min",
      length.out = 48
    )
    resampler <- Resampler$new()

    result <- resampler$downsample(dt, "CargaGlobal")

    expect_equal(nrow(result), 24)
  })

  it("detects hourly frequency", {
    dt <- create_sample_carga(24)
    expect_equal(detect_frequency(dt), "hourly")
  })
})
```

### Area Tests

```r
# tests/testthat/test-areas.R

describe("Area Codes", {
  it("has 21 base areas", {
    base_areas <- get_all_area_codes(
      include_subsystems = FALSE,
      include_national = FALSE
    )
    expect_equal(length(base_areas), 21)
  })

  it("has 26 total series", {
    all_codes <- get_all_area_codes()
    expect_equal(length(all_codes), 26)
  })

  it("validates valid area code", {
    expect_true(validate_area_code("RJ"))
    expect_true(validate_area_code("SECO"))
    expect_true(validate_area_code("SIN"))
  })

  it("rejects invalid area code", {
    expect_error(validate_area_code("XX"), "Invalid area code")
  })

  it("returns correct subsystem", {
    expect_equal(get_subsystem("RJ"), "SECO")
    expect_equal(get_subsystem("RS"), "S")
    expect_equal(get_subsystem("CE"), "NE")
    expect_equal(get_subsystem("PA"), "N")
  })

  it("identifies hierarchy levels", {
    expect_equal(get_hierarchy_level("RJ"), "area")
    expect_equal(get_hierarchy_level("SECO"), "subsystem")
    expect_equal(get_hierarchy_level("SIN"), "national")
  })
})
```

---

## Test Coverage Targets

| Module | Target Coverage |
|--------|-----------------|
| `R/data/loader.R` | ≥80% |
| `R/data/validators.R` | ≥90% |
| `R/data/preprocessors.R` | ≥85% |
| `R/data/catalog.R` | ≥85% |
| `R/data/areas.R` | ≥95% |
| `R/utils/hive.R` | ≥90% |
| **Overall Data Layer** | **≥80%** |

---

## Definition of Done

- [ ] All test files created
- [ ] MockStorageBackend implemented
- [ ] Test fixtures for all data types
- [ ] All tests passing
- [ ] Coverage ≥80% verified with covr
- [ ] Integration tests with real parquet files
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use `testthat::describe()` and `it()` for BDD-style tests
- Mock external dependencies (storage, file system)
- Test edge cases: empty data, NA values, boundary conditions
- Consider using `withr::with_tempdir()` for file-based tests
- Run coverage with: `covr::package_coverage()`
