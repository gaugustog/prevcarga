# EPIC-01: Data Layer

**Duration:** 2 weeks
**Dependencies:** EPIC-00
**Reference:** [MVP Plan R - Phase 1](../mvp-plan-r.md#phase-1-data-layer-2-weeks)

---

## Objective

Build the data loading and preprocessing infrastructure using the storage backend abstraction, supporting Hive-style partitioned parquet files.

---

## Scope

This epic covers:
- DataLoader R6 class using arrow
- Schema validation with checkmate
- Preprocessing pipeline (imputation, resampling)
- DataCatalog for dataset registry
- Support for both S3 and local backends

**Out of Scope:**
- Feature engineering (EPIC-02)
- Model-specific data transformations
- Plugin implementations

---

## Tasks

### T-01.1: Create DataLoader R6 Class
- [ ] Create `R/data/loader.R`
- [ ] Implement `DataLoader` R6 class with storage backend injection
- [ ] Methods to implement:
  ```r
  DataLoader <- R6::R6Class(
    "DataLoader",
    private = list(storage = NULL),
    public = list(
      initialize = function(storage_backend) { ... },
      load_carga = function(areas, start_date, end_date) { ... },
      load_weather = function(areas, type, start_date, end_date) { ... },
      load_holidays = function(areas, years) { ... },
      load_dst_periods = function() { ... },
      load_load_tiers = function(areas) { ... }
    )
  )
  ```
- [ ] Support Hive-style path resolution: `area_code={CODE}/year={YEAR}/data.parquet`

### T-01.2: Implement Hive Partitioning Utilities
- [ ] Create `R/utils/hive.R`
- [ ] Function to build Hive-style paths from parameters
- [ ] Function to parse partition values from paths
- [ ] Support for multi-partition queries (multiple areas, years)

### T-01.3: Create Schema Validators
- [ ] Create `R/data/validators.R`
- [ ] Define schemas using checkmate:
  ```r
  validate_carga_schema = function(dt) {
    checkmate::assert_data_table(dt)
    checkmate::assert_names(names(dt), must.include = c("DataHora", "CargaGlobal", "area_code"))
    checkmate::assert_posixct(dt$DataHora)
    checkmate::assert_numeric(dt$CargaGlobal)
    checkmate::assert_character(dt$area_code)
  }
  ```
- [ ] Validators for: carga, weather, holidays, dst_periods, load_tiers

### T-01.4: Implement Missing Value Imputation
- [ ] Create `R/data/preprocessors.R`
- [ ] Implement `ImputerChain` R6 class:
  ```r
  ImputerChain <- R6::R6Class(
    "ImputerChain",
    public = list(
      add = function(imputer) { ... },
      impute = function(dt) { ... }
    )
  )
  ```
- [ ] Imputation methods:
  - [ ] Last observation carried forward (LOCF)
  - [ ] Linear interpolation
  - [ ] Same-hour-previous-day

### T-01.5: Implement Resampling
- [ ] Add `ResamplerMixin` functionality
- [ ] Hourly to semi-hourly conversion
- [ ] Support for different interpolation methods
- [ ] Preserve data integrity during resampling

### T-01.6: Create DataCatalog
- [ ] Create `R/data/catalog.R`
- [ ] Implement `DataCatalog` R6 class:
  ```r
  DataCatalog <- R6::R6Class(
    "DataCatalog",
    public = list(
      register = function(name, path, schema, partitions) { ... },
      get = function(name) { ... },
      list_datasets = function() { ... },
      validate = function(name, dt) { ... }
    )
  )
  ```
- [ ] Pre-register standard datasets (load, weather, holidays, etc.)

### T-01.7: Configure Area Codes
- [ ] Create `R/data/areas.R` with area code constants
- [ ] Define area groups:
  ```r
  AREAS <- list(
    SECO = c("RJ", "SP", "MG", "ES", "MT", "MS", "AC", "RO", "DF", "GO"),
    S = c("PR", "SC", "RS"),
    NE = c("ALPE", "PBRN", "BASE", "CE", "PI", "BAOE"),
    N = c("AM", "PA", "MA", "TO", "RR", "AP")
  )
  LOSS_AREAS <- c("PESE", "PES", "PENE", "PEN")
  SUBSYSTEMS <- c("SECO", "S", "NE", "N")
  NATIONAL <- "SIN"
  ```
- [ ] Validation functions for area codes

### T-01.8: Write Tests
- [ ] Create `tests/testthat/test-data-loader.R`
- [ ] Test DataLoader with mock storage backend
- [ ] Test schema validation (valid and invalid data)
- [ ] Test imputation methods
- [ ] Test Hive path resolution
- [ ] Achieve ≥80% coverage

---

## Acceptance Criteria

- [ ] `DataLoader` loads parquet files from both local and S3 backends
- [ ] Hive-style partitioning works for area_code and year
- [ ] Schema validation catches invalid data with clear error messages
- [ ] Missing value imputation handles gaps in time series
- [ ] Resampling converts hourly to semi-hourly correctly
- [ ] `DataCatalog` registers and retrieves dataset metadata
- [ ] All 21 area codes are properly defined and validated
- [ ] Tests pass with ≥80% coverage

---

## Definition of Done

- [ ] All tasks completed
- [ ] Code reviewed and merged
- [ ] Tests passing (≥80% coverage)
- [ ] Documentation complete with roxygen2
- [ ] Integration tested with real parquet files

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `R/data/loader.R` | Create | DataLoader R6 class |
| `R/data/validators.R` | Create | Schema validators |
| `R/data/preprocessors.R` | Create | Imputers and resamplers |
| `R/data/catalog.R` | Create | DataCatalog |
| `R/data/areas.R` | Create | Area code definitions |
| `R/utils/hive.R` | Create | Hive partitioning utilities |
| `tests/testthat/test-data-loader.R` | Create | DataLoader tests |

---

## Data Schemas Reference

### Load Data (`load/area_code={CODE}/year={YEAR}/data.parquet`)
| Column | Type | Description |
|--------|------|-------------|
| `DataHora` | POSIXct | Timestamp (hourly) |
| `CargaGlobal` | numeric | Load in MW |
| `area_code` | character | Area identifier |

### Weather Data (`weather/{type}/area_code={CODE}/year={YEAR}/data.parquet`)
| Column | Type | Description |
|--------|------|-------------|
| `DataHora` | POSIXct | Timestamp |
| `temperatura` | numeric | Temperature (C) |
| `umidade` | numeric | Humidity (%) |
| `area_code` | character | Area identifier |

### Holiday Data (`auxiliary/holidays/area_code={CODE}/year={YEAR}/data.parquet`)
| Column | Type | Description |
|--------|------|-------------|
| `Data` | Date | Holiday date |
| `nome` | character | Holiday name |
| `tipo` | character | Holiday type |
| `area_code` | character | Area identifier |
