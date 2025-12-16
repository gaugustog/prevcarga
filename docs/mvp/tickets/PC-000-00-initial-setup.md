# PC-000-00: Initial Setup - R Package Infrastructure

**Epic:** [EPIC-00: Initial Setup](../epics/EPIC-00-initial-setup.md)
**Priority:** Critical
**Effort:** 5 days
**Dependencies:** None
**Blocks:** All other tickets

---

## Objective

Establish the R package infrastructure, development environment, storage abstractions, and core tooling for the PrevCargaONS system.

---

## Background

This foundational ticket creates the entire R package structure that all other components will build upon. It includes:
- Package initialization and metadata
- Dependency management with renv
- Storage backend abstraction (local and S3)
- Structured logging infrastructure
- Test framework configuration

---

## Requirements

### 1. R Package Structure

Initialize the R package with proper structure:

```
prevcarga/
├── DESCRIPTION
├── NAMESPACE
├── .Rbuildignore
├── .Rprofile
├── renv.lock
├── renv/
├── R/
│   ├── cli/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── combination/
│   ├── reconciliation/
│   ├── evaluation/
│   ├── orchestrator/
│   ├── storage/
│   └── utils/
├── inst/
│   ├── config/
│   └── shell/
├── tests/
│   └── testthat/
└── man/
```

### 2. DESCRIPTION File

```
Package: prevcarga
Title: Electric Load Forecasting System for Brazilian Grid
Version: 0.1.0
Authors@R: person("PrevCarga Team", email = "prevcarga@example.com", role = c("aut", "cre"))
Description: Unified forecasting system for Brazilian electric load prediction
    supporting multiple models, forecast combination, and hierarchical reconciliation.
License: Proprietary
Encoding: UTF-8
LazyData: true
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.2.3
Depends: R (>= 4.1.0)
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
    covr,
    lintr
Config/testthat/edition: 3
```

### 3. StorageBackend Abstract Class

Create `R/storage/base.R`:

```r
#' @title StorageBackend Abstract Class
#' @description Base class for storage operations (local, S3)
#' @export
StorageBackend <- R6::R6Class(
  "StorageBackend",
  public = list(
    #' @field backend_type Character identifying the backend type
    backend_type = NULL,

    #' @description Read a parquet file
    #' @param path Character path to the file
    #' @return data.table
    read_parquet = function(path) {
      stop("StorageBackend$read_parquet() must be implemented by subclass")
    },

    #' @description Write a data.table to parquet
    #' @param dt data.table to write
    #' @param path Character path for output
    write_parquet = function(dt, path) {
      stop("StorageBackend$write_parquet() must be implemented by subclass")
    },

    #' @description Read an RDS file
    #' @param path Character path to the file
    #' @return R object
    read_rds = function(path) {
      stop("StorageBackend$read_rds() must be implemented by subclass")
    },

    #' @description Write an object to RDS
    #' @param obj R object to write
    #' @param path Character path for output
    write_rds = function(obj, path) {
      stop("StorageBackend$write_rds() must be implemented by subclass")
    },

    #' @description Check if a path exists
    #' @param path Character path to check
    #' @return Logical
    exists = function(path) {
      stop("StorageBackend$exists() must be implemented by subclass")
    },

    #' @description List files in a directory
    #' @param path Character directory path
    #' @param pattern Optional regex pattern to filter
    #' @return Character vector of file paths
    list_files = function(path, pattern = NULL) {
      stop("StorageBackend$list_files() must be implemented by subclass")
    },

    #' @description Delete a file or directory
    #' @param path Character path to delete
    delete = function(path) {
      stop("StorageBackend$delete() must be implemented by subclass")
    },

    #' @description Create a directory
    #' @param path Character directory path
    mkdir = function(path) {
      stop("StorageBackend$mkdir() must be implemented by subclass")
    }
  )
)
```

### 4. LocalStorageBackend

Create `R/storage/local.R` implementing all abstract methods for local filesystem.

### 5. S3StorageBackend

Create `R/storage/s3.R` using `paws` package for AWS S3 operations.

### 6. StorageFactory

Create `R/storage/factory.R`:

```r
#' @title StorageFactory
#' @description Factory for creating storage backends
#' @export
StorageFactory <- R6::R6Class(
  "StorageFactory",
  public = list(
    #' @description Create backend from config
    #' @param config List with storage configuration
    #' @return StorageBackend instance
    from_config = function(config) {
      backend_type <- config$type %||% "local"
      switch(backend_type,
        "local" = LocalStorageBackend$new(base_path = config$base_path),
        "s3" = S3StorageBackend$new(
          bucket = config$bucket,
          prefix = config$prefix,
          region = config$region
        ),
        stop(sprintf("Unknown storage backend type: %s", backend_type))
      )
    },

    #' @description Create local backend
    #' @param base_path Character base directory path
    #' @return LocalStorageBackend instance
    create_local = function(base_path = ".") {
      LocalStorageBackend$new(base_path = base_path)
    },

    #' @description Create S3 backend
    #' @param bucket Character S3 bucket name
    #' @param prefix Character S3 prefix (optional)
    #' @param region Character AWS region
    #' @return S3StorageBackend instance
    create_s3 = function(bucket, prefix = "", region = "us-east-1") {
      S3StorageBackend$new(bucket = bucket, prefix = prefix, region = region)
    }
  )
)
```

### 7. StructuredLogger

Create `R/orchestrator/logger.R` with JSON-format structured logging supporting:
- Log levels: DEBUG, INFO, WARN, ERROR
- Console and file handlers
- Fields: timestamp, level, message, context

### 8. Test Infrastructure

- Configure testthat edition 3
- Create `tests/testthat.R` entry point
- Create `tests/testthat/helper.R` for shared fixtures
- Create initial tests for storage backends
- Configure covr for coverage reporting

---

## Implementation Steps

1. Create package with `usethis::create_package("prevcarga")`
2. Configure DESCRIPTION with all dependencies
3. Initialize renv with `renv::init()`
4. Create directory structure
5. Implement StorageBackend abstract class
6. Implement LocalStorageBackend
7. Implement S3StorageBackend
8. Implement StorageFactory
9. Implement StructuredLogger
10. Set up testthat infrastructure
11. Write initial tests
12. Generate documentation with roxygen2

---

## Acceptance Criteria

- [ ] Package installs with `R CMD INSTALL .` without errors
- [ ] `R CMD check .` passes with no errors or warnings
- [ ] `renv::restore()` reproduces exact dependency versions
- [ ] LocalStorageBackend can read/write parquet and RDS files
- [ ] S3StorageBackend can read/write to S3 (when credentials available)
- [ ] StorageFactory creates correct backend from YAML config
- [ ] StructuredLogger outputs valid JSON to console and file
- [ ] testthat runs with at least 10 passing tests
- [ ] All R6 classes have roxygen2 documentation
- [ ] Code coverage ≥80% for new code

---

## Test Cases

### Storage Tests

```r
test_that("LocalStorageBackend reads and writes parquet", {
  backend <- LocalStorageBackend$new(base_path = tempdir())
  dt <- data.table::data.table(a = 1:3, b = c("x", "y", "z"))

  path <- "test.parquet"
  backend$write_parquet(dt, path)
  expect_true(backend$exists(path))

  result <- backend$read_parquet(path)
  expect_equal(result, dt)

  backend$delete(path)
  expect_false(backend$exists(path))
})

test_that("StorageFactory creates LocalStorageBackend from config", {
  factory <- StorageFactory$new()
  config <- list(type = "local", base_path = tempdir())

  backend <- factory$from_config(config)
  expect_s3_class(backend, "LocalStorageBackend")
})

test_that("StructuredLogger outputs valid JSON", {
  logger <- StructuredLogger$new(level = "DEBUG")
  output <- capture.output(logger$info("test message", context = list(key = "value")))

  parsed <- jsonlite::fromJSON(output)
  expect_equal(parsed$level, "INFO")
  expect_equal(parsed$message, "test message")
  expect_equal(parsed$context$key, "value")
})
```

---

## Files Created

| File | Description |
|------|-------------|
| `DESCRIPTION` | Package metadata and dependencies |
| `NAMESPACE` | Package exports |
| `.Rbuildignore` | Build ignore patterns |
| `.Rprofile` | Environment setup |
| `renv.lock` | Dependency lock file |
| `R/storage/base.R` | StorageBackend abstract class |
| `R/storage/local.R` | LocalStorageBackend implementation |
| `R/storage/s3.R` | S3StorageBackend implementation |
| `R/storage/factory.R` | StorageFactory |
| `R/orchestrator/logger.R` | StructuredLogger |
| `tests/testthat.R` | Test entry point |
| `tests/testthat/helper.R` | Test helpers |
| `tests/testthat/test-storage-backend.R` | Storage tests |
| `tests/testthat/test-logger.R` | Logger tests |
| `inst/config/default.yaml` | Default configuration template |

---

## Notes

- This ticket must be completed before any other implementation work
- S3 tests require AWS credentials and should be skipped in CI if unavailable
- Use `checkmate` for input validation in all public methods
- Follow Google R Style Guide for code formatting
