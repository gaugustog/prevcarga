# PC-003-01: Schema Validators

**Epic:** [EPIC-01: Data Layer](../epics/EPIC-01-data-layer.md)
**Task Reference:** T-01.3
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Implement schema validation functions using `checkmate` to ensure data integrity for all dataset types loaded by the DataLoader.

---

## Acceptance Criteria

- [ ] Validator module created in `R/data/validators.R`
- [ ] `validate_carga_schema()` validates load data
- [ ] `validate_weather_schema()` validates weather data
- [ ] `validate_holidays_schema()` validates holiday data
- [ ] `validate_dst_schema()` validates DST period data
- [ ] `validate_load_tiers_schema()` validates load tier data
- [ ] Clear, actionable error messages on validation failure
- [ ] Support for optional columns with defaults

---

## Technical Specification

### File Location
```
R/data/validators.R
```

### Schema Definitions

```r
#' @title Schema column specification
#' @description Define expected columns and their types
SCHEMAS <- list(
  carga = list(
    required = list(
      DataHora = "POSIXct",
      CargaGlobal = "numeric",
      area_code = "character"
    ),
    optional = list(
      CargaVerificada = "numeric"
    )
  ),

  weather = list(
    required = list(
      DataHora = "POSIXct",
      temperatura = "numeric",
      area_code = "character"
    ),
    optional = list(
      umidade = "numeric",
      pressao = "numeric",
      vento_velocidade = "numeric",
      vento_direcao = "numeric",
      precipitacao = "numeric",
      nebulosidade = "numeric"
    )
  ),

  holidays = list(
    required = list(
      Data = "Date",
      nome = "character",
      area_code = "character"
    ),
    optional = list(
      tipo = "character"
    )
  ),

  dst_periods = list(
    required = list(
      inicio = "Date",
      fim = "Date",
      ano = "integer"
    ),
    optional = list()
  ),

  load_tiers = list(
    required = list(
      area_code = "character",
      mes = "integer",
      ponta_inicio = "character",
      ponta_fim = "character"
    ),
    optional = list(
      intermediario_inicio = "character",
      intermediario_fim = "character"
    )
  )
)
```

### Validation Functions

```r
#' Validate data.table against a schema
#'
#' @param dt data.table to validate
#' @param schema_name Name of schema from SCHEMAS
#' @param strict If TRUE, fail on extra columns
#' @return TRUE invisibly if valid, throws error otherwise
#' @export
validate_schema <- function(dt, schema_name, strict = FALSE) {
  checkmate::assert_data_table(dt)
  checkmate::assert_choice(schema_name, names(SCHEMAS))

  schema <- SCHEMAS[[schema_name]]
  errors <- character()

  # Check required columns exist
  required_cols <- names(schema$required)
  missing_cols <- setdiff(required_cols, names(dt))
  if (length(missing_cols) > 0) {
    errors <- c(errors, sprintf(
      "Missing required columns: %s",
      paste(missing_cols, collapse = ", ")
    ))
  }

  # Validate column types for present columns
  for (col in intersect(required_cols, names(dt))) {
    expected_type <- schema$required[[col]]
    if (!check_column_type(dt[[col]], expected_type)) {
      errors <- c(errors, sprintf(
        "Column '%s': expected %s, got %s",
        col, expected_type, class(dt[[col]])[1]
      ))
    }
  }

  # Validate optional column types if present
  for (col in intersect(names(schema$optional), names(dt))) {
    expected_type <- schema$optional[[col]]
    if (!check_column_type(dt[[col]], expected_type)) {
      errors <- c(errors, sprintf(
        "Column '%s': expected %s, got %s",
        col, expected_type, class(dt[[col]])[1]
      ))
    }
  }

  # Check for unexpected columns in strict mode
  if (strict) {
    allowed_cols <- c(names(schema$required), names(schema$optional))
    extra_cols <- setdiff(names(dt), allowed_cols)
    if (length(extra_cols) > 0) {
      errors <- c(errors, sprintf(
        "Unexpected columns (strict mode): %s",
        paste(extra_cols, collapse = ", ")
      ))
    }
  }

  if (length(errors) > 0) {
    stop(sprintf(
      "Schema validation failed for '%s':\n  - %s",
      schema_name,
      paste(errors, collapse = "\n  - ")
    ))
  }

  invisible(TRUE)
}


#' Check if column matches expected type
#'
#' @param col Column vector
#' @param expected_type Expected type name
#' @return TRUE if matches
check_column_type <- function(col, expected_type) {
  switch(expected_type,
    "POSIXct" = inherits(col, "POSIXct"),
    "Date" = inherits(col, "Date"),
    "numeric" = is.numeric(col),
    "integer" = is.integer(col) || (is.numeric(col) && all(col == floor(col), na.rm = TRUE)),
    "character" = is.character(col),
    "logical" = is.logical(col),
    FALSE
  )
}


#' Convenience validators for each data type
#' @rdname validate_schema
#' @export
validate_carga_schema <- function(dt, strict = FALSE) {
  validate_schema(dt, "carga", strict)
}

#' @rdname validate_schema
#' @export
validate_weather_schema <- function(dt, strict = FALSE) {
  validate_schema(dt, "weather", strict)
}

#' @rdname validate_schema
#' @export
validate_holidays_schema <- function(dt, strict = FALSE) {
  validate_schema(dt, "holidays", strict)
}

#' @rdname validate_schema
#' @export
validate_dst_schema <- function(dt, strict = FALSE) {
  validate_schema(dt, "dst_periods", strict)
}

#' @rdname validate_schema
#' @export
validate_load_tiers_schema <- function(dt, strict = FALSE) {
  validate_schema(dt, "load_tiers", strict)
}
```

### Additional Validation Functions

```r
#' Validate data ranges and constraints
#'
#' @param dt data.table to validate
#' @param constraints List of constraint specifications
#' @return TRUE invisibly if valid
#' @export
validate_constraints <- function(dt, constraints) {
  errors <- character()

  for (constraint in constraints) {
    col <- constraint$column
    if (!col %in% names(dt)) next

    if (!is.null(constraint$min)) {
      if (any(dt[[col]] < constraint$min, na.rm = TRUE)) {
        errors <- c(errors, sprintf(
          "Column '%s' has values below minimum %s",
          col, constraint$min
        ))
      }
    }

    if (!is.null(constraint$max)) {
      if (any(dt[[col]] > constraint$max, na.rm = TRUE)) {
        errors <- c(errors, sprintf(
          "Column '%s' has values above maximum %s",
          col, constraint$max
        ))
      }
    }

    if (!is.null(constraint$allowed)) {
      invalid <- setdiff(unique(dt[[col]]), constraint$allowed)
      if (length(invalid) > 0) {
        errors <- c(errors, sprintf(
          "Column '%s' has invalid values: %s",
          col, paste(head(invalid, 5), collapse = ", ")
        ))
      }
    }
  }

  if (length(errors) > 0) {
    stop(sprintf(
      "Constraint validation failed:\n  - %s",
      paste(errors, collapse = "\n  - ")
    ))
  }

  invisible(TRUE)
}


#' Validate load data specific constraints
#' @export
validate_carga_constraints <- function(dt) {
  validate_constraints(dt, list(
    list(column = "CargaGlobal", min = 0, max = 100000),
    list(column = "area_code", allowed = c(
      get_all_area_codes()  # From PC-007-01
    ))
  ))
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Valid carga schema | TRUE |
| TC-002 | Missing required column | Error with column name |
| TC-003 | Wrong column type | Error with type info |
| TC-004 | Extra columns (non-strict) | TRUE |
| TC-005 | Extra columns (strict) | Error |
| TC-006 | Valid weather schema | TRUE |
| TC-007 | Valid holidays schema | TRUE |
| TC-008 | Constraint violation (min) | Error |
| TC-009 | Constraint violation (max) | Error |
| TC-010 | Invalid area_code | Error |

---

## Definition of Done

- [ ] All validators implemented
- [ ] SCHEMAS constant with all data types
- [ ] Constraint validation for business rules
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Schema definitions should be kept in sync with data sources
- Consider adding `validate_on_load` option to DataLoader
- Error messages should help identify the specific row/value causing issues
