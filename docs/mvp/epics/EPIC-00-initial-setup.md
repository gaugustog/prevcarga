# EPIC-00: Initial Setup

**Duration:** 1 week
**Dependencies:** None
**Reference:** [MVP Plan R - Phase 0](../mvp-plan-r.md#phase-0-initial-setup-1-week)

---

## Objective

Establish the R package infrastructure, development environment, and core abstractions for the PrevCargaONS system.

---

## Scope

This epic covers:
- R package structure initialization
- Dependency management with renv
- Storage backend abstraction (R6 classes)
- Structured logging infrastructure
- Test framework setup

**Out of Scope:**
- Plugin implementations (models, features, combiners)
- Data loading logic (EPIC-01)
- CLI commands (EPIC-09)

---

## Tasks

### T-00.1: Initialize R Package Structure
- [ ] Create package with `usethis::create_package("prevcarga")`
- [ ] Set up DESCRIPTION file with metadata
- [ ] Configure NAMESPACE for exports
- [ ] Create `.Rprofile` for environment setup
- [ ] Initialize `renv` for dependency management

### T-00.2: Configure Dependencies
- [ ] Add core dependencies to DESCRIPTION:
  ```
  Imports:
    R6,
    data.table,
    arrow,
    yaml,
    paws,
    future,
    future.apply,
    cli,
    optparse,
    checkmate,
    jsonlite,
    highcharter,
    rmarkdown,
    htmlwidgets
  Suggests:
    testthat (>= 3.0.0),
    mockery,
    covr
  ```
- [ ] Run `renv::snapshot()` to lock versions
- [ ] Document minimum R version requirement (≥4.1.0)

### T-00.3: Create StorageBackend Abstract Class
- [ ] Implement `StorageBackend` R6 base class in `R/storage/base.R`
  ```r
  StorageBackend <- R6::R6Class(
    "StorageBackend",
    public = list(
      backend_type = NULL,
      read_parquet = function(path) stop("Must implement"),
      write_parquet = function(dt, path) stop("Must implement"),
      read_rds = function(path) stop("Must implement"),
      write_rds = function(obj, path) stop("Must implement"),
      exists = function(path) stop("Must implement"),
      list_files = function(path, pattern = NULL) stop("Must implement"),
      delete = function(path) stop("Must implement"),
      mkdir = function(path) stop("Must implement")
    )
  )
  ```

### T-00.4: Implement LocalStorageBackend
- [ ] Create `R/storage/local.R` inheriting from `StorageBackend`
- [ ] Implement all abstract methods for local filesystem
- [ ] Handle directory creation automatically
- [ ] Support Hive-style partitioning paths

### T-00.5: Implement S3StorageBackend
- [ ] Create `R/storage/s3.R` inheriting from `StorageBackend`
- [ ] Use `paws` package for S3 operations
- [ ] Implement temp file strategy for parquet read/write
- [ ] Handle S3 configuration (bucket, region, prefix)

### T-00.6: Create StorageFactory
- [ ] Implement `StorageFactory` R6 class in `R/storage/factory.R`
- [ ] Support creation from config YAML
- [ ] Support explicit backend creation
- [ ] Default to local backend if not specified

### T-00.7: Set Up Structured Logging
- [ ] Create `R/orchestrator/logger.R`
- [ ] Implement JSON-format structured logging
- [ ] Support log levels (DEBUG, INFO, WARN, ERROR)
- [ ] Support console and file handlers
- [ ] Include timestamp, level, message, context fields

### T-00.8: Set Up Test Infrastructure
- [ ] Configure `testthat` (edition 3)
- [ ] Create `tests/testthat.R` entry point
- [ ] Create test helper files in `tests/testthat/`
- [ ] Set up `covr` for coverage reporting
- [ ] Create first test: `test-storage-backend.R`

### T-00.9: Create Directory Structure
- [ ] Create all R source directories per MVP plan:
  ```
  R/
  ├── cli/
  ├── data/
  ├── features/
  ├── models/
  ├── combination/
  ├── reconciliation/
  ├── evaluation/
  ├── orchestrator/
  ├── storage/
  └── utils/
  ```
- [ ] Create `inst/config/` for default YAML configs
- [ ] Create `inst/shell/` for shell entry script

---

## Acceptance Criteria

- [ ] Package installs with `R CMD INSTALL .` without errors
- [ ] `renv::restore()` reproduces exact dependency versions
- [ ] `LocalStorageBackend` can read/write parquet files
- [ ] `S3StorageBackend` can read/write to S3 (with credentials)
- [ ] `StorageFactory$from_config()` creates correct backend from YAML
- [ ] Structured logger outputs valid JSON
- [ ] `testthat` runs with at least 5 passing tests
- [ ] All R6 classes are documented with roxygen2

---

## Definition of Done

- [ ] All tasks completed
- [ ] Code reviewed and merged
- [ ] Tests passing (≥80% coverage for new code)
- [ ] Documentation generated with `roxygen2::roxygenise()`
- [ ] No critical linting issues

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `DESCRIPTION` | Create | Package metadata |
| `NAMESPACE` | Create | Exports |
| `renv.lock` | Create | Dependency lock |
| `R/storage/base.R` | Create | StorageBackend abstract class |
| `R/storage/local.R` | Create | LocalStorageBackend |
| `R/storage/s3.R` | Create | S3StorageBackend |
| `R/storage/factory.R` | Create | StorageFactory |
| `R/orchestrator/logger.R` | Create | StructuredLogger |
| `tests/testthat.R` | Create | Test entry point |
| `tests/testthat/test-storage-backend.R` | Create | Storage tests |
