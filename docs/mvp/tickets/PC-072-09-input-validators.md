# PC-072-09: Input Validators

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.9
**Priority:** Medium
**Estimated Effort:** 0.5 days

---

## Summary

Implement input validation utilities for CLI commands including date format validators, area code validators, model name validators, file path validators, and numeric range validators.

---

## Acceptance Criteria

- [ ] Validators module created in `R/cli/validators.R`
- [ ] Date format validation with multiple formats
- [ ] Area code validation against configuration
- [ ] Model name validation against registry
- [ ] File path validation (existence, permissions)
- [ ] Numeric range validation
- [ ] Clear error messages for validation failures

---

## Technical Specification

### File Location
```
R/cli/validators.R
```

### Date Validators

```r
#' Validate date string
#'
#' @param date_str Date string to validate
#' @param name Parameter name for error messages
#' @param allow_null Allow NULL values
#' @return Parsed Date object
#' @export
validate_date <- function(date_str, name = "date", allow_null = FALSE) {
  if (is.null(date_str)) {
    if (allow_null) return(NULL)
    cli_validation_error(name, "is required")
  }

  # Try common formats
  formats <- c(
    "%Y-%m-%d",       # 2024-01-15
    "%Y/%m/%d",       # 2024/01/15
    "%d-%m-%Y",       # 15-01-2024
    "%d/%m/%Y",       # 15/01/2024
    "%Y%m%d"          # 20240115
  )

  for (fmt in formats) {
    result <- tryCatch(
      as.Date(date_str, format = fmt),
      error = function(e) NA
    )
    if (!is.na(result)) {
      return(result)
    }
  }

  cli_validation_error(
    name,
    sprintf("invalid date format '%s'. Use YYYY-MM-DD", date_str)
  )
}


#' Validate date range
#'
#' @param start_date Start date
#' @param end_date End date
#' @param name Parameter name for error messages
#' @return List with start_date and end_date
#' @export
validate_date_range <- function(start_date, end_date, name = "date range") {
  start <- validate_date(start_date, "start_date")
  end <- validate_date(end_date, "end_date")

  if (start >= end) {
    cli_validation_error(
      name,
      "start_date must be before end_date"
    )
  }

  # Check reasonable range
  days <- as.integer(end - start)
  if (days > 3650) {  # ~10 years
    cli_validation_warning(
      name,
      sprintf("very large range (%d days). This may take a long time.", days)
    )
  }

  list(start_date = start, end_date = end)
}


#' Validate date not in future
#'
#' @param date Date to check
#' @param name Parameter name
#' @param allow_today Allow today's date
#' @return Validated date
#' @export
validate_date_not_future <- function(date, name = "date", allow_today = TRUE) {
  date <- validate_date(date, name)

  max_date <- if (allow_today) Sys.Date() else Sys.Date() - 1

  if (date > max_date) {
    cli_validation_error(
      name,
      sprintf("cannot be in the future (max: %s)", max_date)
    )
  }

  date
}
```

### Area Code Validators

```r
#' Validate area codes
#'
#' @param areas Character vector or comma-separated string
#' @param config ConfigManager instance
#' @param name Parameter name
#' @param allow_all Allow "all" as value
#' @return Character vector of valid area codes
#' @export
validate_areas <- function(areas, config = NULL, name = "areas", allow_all = TRUE) {
  if (is.null(areas)) {
    cli_validation_error(name, "is required")
  }

  # Handle comma-separated string
  if (is.character(areas) && length(areas) == 1) {
    if (allow_all && tolower(areas) == "all") {
      if (is.null(config)) {
        cli_validation_error(name, "'all' requires config to be loaded")
      }
      return(config$get_areas())
    }
    areas <- trimws(strsplit(areas, ",")[[1]])
  }

  # Validate against known areas
  if (!is.null(config)) {
    valid_areas <- config$get_areas()
    invalid <- setdiff(areas, valid_areas)

    if (length(invalid) > 0) {
      cli_validation_error(
        name,
        sprintf("invalid area codes: %s. Valid: %s",
                paste(invalid, collapse = ", "),
                paste(valid_areas, collapse = ", "))
      )
    }
  }

  # Check for duplicates
  if (any(duplicated(areas))) {
    areas <- unique(areas)
    cli_validation_warning(name, "duplicate areas removed")
  }

  areas
}


#' Validate single area code
#'
#' @param area Single area code
#' @param config ConfigManager instance
#' @param name Parameter name
#' @return Validated area code
#' @export
validate_area <- function(area, config = NULL, name = "area") {
  if (is.null(area)) {
    cli_validation_error(name, "is required")
  }

  areas <- validate_areas(area, config, name, allow_all = FALSE)

  if (length(areas) != 1) {
    cli_validation_error(name, "must be a single area code")
  }

  areas[1]
}
```

### Model Name Validators

```r
#' Validate model names
#'
#' @param models Character vector or comma-separated string
#' @param config ConfigManager instance (optional)
#' @param name Parameter name
#' @param allow_all Allow "all" as value
#' @return Character vector of valid model names
#' @export
validate_models <- function(models, config = NULL, name = "models", allow_all = TRUE) {
  if (is.null(models)) {
    cli_validation_error(name, "is required")
  }

  # Handle comma-separated string
  if (is.character(models) && length(models) == 1) {
    if (allow_all && tolower(models) == "all") {
      if (!is.null(config)) {
        # Get enabled models from config
        model_config <- config$get("models.plugins")
        return(names(Filter(function(m) isTRUE(m$enabled), model_config)))
      }
      # Return all registered models
      return(list_models())
    }
    models <- trimws(strsplit(models, ",")[[1]])
  }

  # Validate against registry
  for (model in models) {
    if (!has_model(model)) {
      available <- list_models()
      cli_validation_error(
        name,
        sprintf("unknown model '%s'. Available: %s",
                model, paste(available, collapse = ", "))
      )
    }
  }

  models
}


#' Validate single model name
#'
#' @param model Model name
#' @param name Parameter name
#' @return Validated model name
#' @export
validate_model <- function(model, name = "model") {
  if (is.null(model)) {
    cli_validation_error(name, "is required")
  }

  models <- validate_models(model, name = name, allow_all = FALSE)

  if (length(models) != 1) {
    cli_validation_error(name, "must be a single model name")
  }

  models[1]
}
```

### File Path Validators

```r
#' Validate file exists
#'
#' @param path File path
#' @param name Parameter name
#' @param must_exist Require file to exist
#' @return Validated path
#' @export
validate_file_path <- function(path, name = "file", must_exist = TRUE) {
  if (is.null(path)) {
    cli_validation_error(name, "is required")
  }

  # Expand path
  path <- path.expand(path)

  if (must_exist && !file.exists(path)) {
    cli_validation_error(name, sprintf("file not found: %s", path))
  }

  # Check readable if exists
  if (file.exists(path) && file.access(path, 4) != 0) {
    cli_validation_error(name, sprintf("file not readable: %s", path))
  }

  path
}


#' Validate directory exists
#'
#' @param path Directory path
#' @param name Parameter name
#' @param create Create if not exists
#' @return Validated path
#' @export
validate_dir_path <- function(path, name = "directory", create = FALSE) {
  if (is.null(path)) {
    cli_validation_error(name, "is required")
  }

  path <- path.expand(path)

  if (!dir.exists(path)) {
    if (create) {
      tryCatch({
        dir.create(path, recursive = TRUE)
        cli_validation_info(name, sprintf("created directory: %s", path))
      }, error = function(e) {
        cli_validation_error(name, sprintf("cannot create directory: %s", path))
      })
    } else {
      cli_validation_error(name, sprintf("directory not found: %s", path))
    }
  }

  # Check writable
  if (file.access(path, 2) != 0) {
    cli_validation_error(name, sprintf("directory not writable: %s", path))
  }

  path
}


#' Validate output file path
#'
#' @param path Output file path
#' @param name Parameter name
#' @param overwrite Allow overwriting
#' @return Validated path
#' @export
validate_output_path <- function(path, name = "output", overwrite = TRUE) {
  if (is.null(path)) {
    cli_validation_error(name, "is required")
  }

  path <- path.expand(path)

  # Check directory exists (or create)
  dir_path <- dirname(path)
  validate_dir_path(dir_path, sprintf("%s directory", name), create = TRUE)

  # Check if file exists
  if (!overwrite && file.exists(path)) {
    cli_validation_error(
      name,
      sprintf("file already exists: %s. Use --overwrite to replace.", path)
    )
  }

  path
}
```

### Numeric Validators

```r
#' Validate integer in range
#'
#' @param value Value to validate
#' @param name Parameter name
#' @param min Minimum value (inclusive)
#' @param max Maximum value (inclusive)
#' @param allow_null Allow NULL
#' @return Validated integer
#' @export
validate_integer <- function(value, name = "value", min = NULL, max = NULL,
                             allow_null = FALSE) {
  if (is.null(value)) {
    if (allow_null) return(NULL)
    cli_validation_error(name, "is required")
  }

  # Try to convert
  value <- suppressWarnings(as.integer(value))

  if (is.na(value)) {
    cli_validation_error(name, "must be an integer")
  }

  # Check range
  if (!is.null(min) && value < min) {
    cli_validation_error(name, sprintf("must be at least %d", min))
  }

  if (!is.null(max) && value > max) {
    cli_validation_error(name, sprintf("must be at most %d", max))
  }

  value
}


#' Validate numeric value
#'
#' @param value Value to validate
#' @param name Parameter name
#' @param min Minimum value
#' @param max Maximum value
#' @param allow_null Allow NULL
#' @return Validated numeric
#' @export
validate_numeric <- function(value, name = "value", min = NULL, max = NULL,
                             allow_null = FALSE) {
  if (is.null(value)) {
    if (allow_null) return(NULL)
    cli_validation_error(name, "is required")
  }

  value <- suppressWarnings(as.numeric(value))

  if (is.na(value)) {
    cli_validation_error(name, "must be a number")
  }

  if (!is.null(min) && value < min) {
    cli_validation_error(name, sprintf("must be at least %g", min))
  }

  if (!is.null(max) && value > max) {
    cli_validation_error(name, sprintf("must be at most %g", max))
  }

  value
}


#' Validate comma-separated integers
#'
#' @param value Comma-separated string or vector
#' @param name Parameter name
#' @param min Minimum value for each
#' @param max Maximum value for each
#' @return Integer vector
#' @export
validate_integer_list <- function(value, name = "values", min = NULL, max = NULL) {
  if (is.null(value)) {
    cli_validation_error(name, "is required")
  }

  # Parse if string
  if (is.character(value) && length(value) == 1) {
    value <- strsplit(value, ",")[[1]]
  }

  # Validate each
  result <- sapply(value, function(v) {
    validate_integer(v, name, min, max)
  })

  as.integer(result)
}
```

### Validation Helpers

```r
#' Throw validation error
#'
#' @param name Parameter name
#' @param message Error message
#' @noRd
cli_validation_error <- function(name, message) {
  stop(sprintf("Invalid %s: %s", name, message), call. = FALSE)
}


#' Print validation warning
#'
#' @param name Parameter name
#' @param message Warning message
#' @noRd
cli_validation_warning <- function(name, message) {
  warning(sprintf("Warning for %s: %s", name, message), call. = FALSE)
}


#' Print validation info
#'
#' @param name Parameter name
#' @param message Info message
#' @noRd
cli_validation_info <- function(name, message) {
  message(sprintf("Note for %s: %s", name, message))
}


#' Validate choice from list
#'
#' @param value Value to validate
#' @param choices Valid choices
#' @param name Parameter name
#' @param allow_null Allow NULL
#' @return Validated value
#' @export
validate_choice <- function(value, choices, name = "choice", allow_null = FALSE) {
  if (is.null(value)) {
    if (allow_null) return(NULL)
    cli_validation_error(name, "is required")
  }

  value <- tolower(value)
  choices_lower <- tolower(choices)

  if (!value %in% choices_lower) {
    cli_validation_error(
      name,
      sprintf("invalid value '%s'. Valid: %s", value, paste(choices, collapse = ", "))
    )
  }

  # Return original case
  choices[match(value, choices_lower)]
}


#' Validate with custom function
#'
#' @param value Value to validate
#' @param validator Custom validation function
#' @param name Parameter name
#' @param ... Additional arguments to validator
#' @return Validated value
#' @export
validate_custom <- function(value, validator, name = "value", ...) {
  result <- validator(value, ...)

  if (isFALSE(result)) {
    cli_validation_error(name, "failed custom validation")
  }

  if (is.character(result) && nchar(result) > 0) {
    cli_validation_error(name, result)
  }

  value
}
```

### Usage Examples

```r
# Date validation
date <- validate_date("2024-01-15")
range <- validate_date_range("2024-01-01", "2024-12-31")

# Area validation (with config)
areas <- validate_areas("RJ,SP,MG", config)
areas <- validate_areas("all", config)

# Model validation
models <- validate_models("lgbm,rf")
model <- validate_model("lgbm")

# File validation
path <- validate_file_path("config.yaml")
output <- validate_output_path("results/output.csv")

# Numeric validation
n_jobs <- validate_integer(4, "parallel", min = 1, max = 32)
horizons <- validate_integer_list("0,1,2,3", "horizons", min = 0, max = 8)

# Choice validation
mode <- validate_choice("batch", c("batch", "intraday"))
format <- validate_choice("csv", c("parquet", "csv", "rds"))
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | validate_date() valid | Date returned |
| TC-002 | validate_date() invalid | Error thrown |
| TC-003 | validate_date_range() valid | Range returned |
| TC-004 | validate_date_range() invalid | Error thrown |
| TC-005 | validate_areas() valid | Areas returned |
| TC-006 | validate_areas() "all" | All areas |
| TC-007 | validate_areas() invalid | Error thrown |
| TC-008 | validate_models() valid | Models returned |
| TC-009 | validate_file_path() exists | Path returned |
| TC-010 | validate_file_path() not exists | Error thrown |
| TC-011 | validate_integer() in range | Integer returned |
| TC-012 | validate_choice() valid | Choice returned |

---

## Dependencies

- PC-057-08: ConfigManager (for area validation)
- PC-017-03: ModelRegistry (for model validation)

---

## Definition of Done

- [ ] All validators implemented
- [ ] Clear error messages
- [ ] Support for common input formats
- [ ] Integration with ConfigManager
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Validators throw exceptions on failure
- Messages should help users fix the issue
- Support multiple date formats for flexibility
- Case-insensitive matching where sensible
