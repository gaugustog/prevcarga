# PC-007-01: Area Codes

**Epic:** [EPIC-01: Data Layer](../epics/EPIC-01-data-layer.md)
**Task Reference:** T-01.7
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Define and validate area codes for the Brazilian interconnected power system (SIN), including individual areas, subsystems, and the national aggregate.

---

## Acceptance Criteria

- [ ] Areas module created in `R/data/areas.R`
- [ ] `AREAS` constant with all 21 area codes grouped by subsystem
- [ ] `LOSS_AREAS` constant for transmission loss areas
- [ ] `SUBSYSTEMS` constant for the 4 subsystems
- [ ] `NATIONAL` constant for national aggregate (SIN)
- [ ] `get_all_area_codes()` returns all valid codes
- [ ] `validate_area_code()` validates single code
- [ ] `validate_area_codes()` validates vector of codes
- [ ] `get_subsystem()` returns subsystem for an area
- [ ] `get_areas_in_subsystem()` returns areas for a subsystem

---

## Technical Specification

### File Location
```
R/data/areas.R
```

### Constants

```r
#' @title Area Definitions for SIN (Brazilian Interconnected System)
#' @description Constants and utilities for area code management
#' @name area-codes

#' Areas grouped by subsystem
#' @export
AREAS <- list(
  SECO = c("RJ", "SP", "MG", "ES", "MT", "MS", "AC", "RO", "DF", "GO"),
  S = c("PR", "SC", "RS"),
  NE = c("ALPE", "PBRN", "BASE", "CE", "PI", "BAOE"),
  N = c("AM", "PA", "MA", "TO", "RR", "AP")
)

#' Transmission loss areas (pseudo-areas)
#' @export
LOSS_AREAS <- c("PESE", "PES", "PENE", "PEN")

#' Subsystem codes
#' @export
SUBSYSTEMS <- c("SECO", "S", "NE", "N")

#' National aggregate code
#' @export
NATIONAL <- "SIN"

#' All forecast series (areas + subsystems + national)
#' @export
ALL_SERIES <- c(unlist(AREAS, use.names = FALSE), SUBSYSTEMS, NATIONAL)
```

### Utility Functions

```r
#' Get all valid area codes
#'
#' @param include_subsystems Include subsystem codes (default TRUE)
#' @param include_national Include national code (default TRUE)
#' @param include_loss Include loss areas (default FALSE)
#' @return Character vector of area codes
#' @export
get_all_area_codes <- function(include_subsystems = TRUE,
                                include_national = TRUE,
                                include_loss = FALSE) {
  codes <- unlist(AREAS, use.names = FALSE)

  if (include_subsystems) {
    codes <- c(codes, SUBSYSTEMS)
  }

  if (include_national) {
    codes <- c(codes, NATIONAL)
  }

  if (include_loss) {
    codes <- c(codes, LOSS_AREAS)
  }

  codes
}


#' Validate a single area code
#'
#' @param code Area code to validate
#' @param allow_subsystems Allow subsystem codes
#' @param allow_national Allow national code
#' @param allow_loss Allow loss areas
#' @return TRUE if valid, throws error otherwise
#' @export
validate_area_code <- function(code,
                                allow_subsystems = TRUE,
                                allow_national = TRUE,
                                allow_loss = FALSE) {
  checkmate::assert_string(code)

  valid_codes <- get_all_area_codes(
    include_subsystems = allow_subsystems,
    include_national = allow_national,
    include_loss = allow_loss
  )

  if (!code %in% valid_codes) {
    stop(sprintf(
      "Invalid area code: '%s'. Valid codes: %s",
      code,
      paste(head(valid_codes, 10), collapse = ", ")
    ))
  }

  invisible(TRUE)
}


#' Validate multiple area codes
#'
#' @param codes Character vector of area codes
#' @param allow_subsystems Allow subsystem codes
#' @param allow_national Allow national code
#' @param allow_loss Allow loss areas
#' @return TRUE if all valid, throws error otherwise
#' @export
validate_area_codes <- function(codes,
                                 allow_subsystems = TRUE,
                                 allow_national = TRUE,
                                 allow_loss = FALSE) {
  checkmate::assert_character(codes, min.len = 1)

  valid_codes <- get_all_area_codes(
    include_subsystems = allow_subsystems,
    include_national = allow_national,
    include_loss = allow_loss
  )

  invalid <- setdiff(codes, valid_codes)

  if (length(invalid) > 0) {
    stop(sprintf(
      "Invalid area codes: %s",
      paste(invalid, collapse = ", ")
    ))
  }

  invisible(TRUE)
}


#' Get subsystem for an area code
#'
#' @param code Area code
#' @return Subsystem code or NA if not found
#' @export
get_subsystem <- function(code) {
  checkmate::assert_string(code)

  # Check if it's already a subsystem or national
  if (code %in% SUBSYSTEMS) {
    return(code)
  }

  if (code == NATIONAL) {
    return(NA_character_)  # National has no single subsystem
  }

  # Find which subsystem contains this area
  for (subsys in names(AREAS)) {
    if (code %in% AREAS[[subsys]]) {
      return(subsys)
    }
  }

  NA_character_
}


#' Get all areas in a subsystem
#'
#' @param subsystem Subsystem code
#' @return Character vector of area codes
#' @export
get_areas_in_subsystem <- function(subsystem) {
  checkmate::assert_string(subsystem)
  checkmate::assert_choice(subsystem, SUBSYSTEMS)

  AREAS[[subsystem]]
}


#' Check if code is a subsystem
#'
#' @param code Area code
#' @return Logical
#' @export
is_subsystem <- function(code) {
  code %in% SUBSYSTEMS
}


#' Check if code is national aggregate
#'
#' @param code Area code
#' @return Logical
#' @export
is_national <- function(code) {
  code == NATIONAL
}


#' Check if code is a base area (not subsystem or national)
#'
#' @param code Area code
#' @return Logical
#' @export
is_base_area <- function(code) {
  code %in% unlist(AREAS, use.names = FALSE)
}


#' Get area hierarchy level
#'
#' @param code Area code
#' @return Character: "area", "subsystem", or "national"
#' @export
get_hierarchy_level <- function(code) {
  if (is_national(code)) {
    "national"
  } else if (is_subsystem(code)) {
    "subsystem"
  } else if (is_base_area(code)) {
    "area"
  } else {
    NA_character_
  }
}
```

### Area Metadata

```r
#' Area display names
#' @export
AREA_NAMES <- list(
  # SECO
  RJ = "Rio de Janeiro",
  SP = "São Paulo",
  MG = "Minas Gerais",
  ES = "Espírito Santo",
  MT = "Mato Grosso",
  MS = "Mato Grosso do Sul",
  AC = "Acre",
  RO = "Rondônia",
  DF = "Distrito Federal",
  GO = "Goiás",
  # S
  PR = "Paraná",
  SC = "Santa Catarina",
  RS = "Rio Grande do Sul",
  # NE
  ALPE = "Alagoas/Pernambuco",
  PBRN = "Paraíba/Rio Grande do Norte",
  BASE = "Bahia Sudeste",
  CE = "Ceará",
  PI = "Piauí",
  BAOE = "Bahia Oeste",
  # N
  AM = "Amazonas",
  PA = "Pará",
  MA = "Maranhão",
  TO = "Tocantins",
  RR = "Roraima",
  AP = "Amapá",
  # Subsystems
  SECO = "Sudeste/Centro-Oeste",
  S = "Sul",
  NE = "Nordeste",
  N = "Norte",
  # National
  SIN = "Sistema Interligado Nacional"
)


#' Get display name for area code
#'
#' @param code Area code
#' @return Display name or code if not found
#' @export
get_area_name <- function(code) {
  checkmate::assert_string(code)
  name <- AREA_NAMES[[code]]
  if (is.null(name)) code else name
}
```

### Usage Example

```r
# Get all areas
all_codes <- get_all_area_codes()  # 26 codes (21 areas + 4 subsystems + SIN)

# Get only base areas
base_areas <- get_all_area_codes(include_subsystems = FALSE, include_national = FALSE)

# Validate codes
validate_area_code("RJ")  # TRUE
validate_area_code("XX")  # Error: Invalid area code

# Get subsystem
get_subsystem("RJ")    # "SECO"
get_subsystem("SECO")  # "SECO"
get_subsystem("SIN")   # NA

# Get areas in subsystem
get_areas_in_subsystem("S")  # c("PR", "SC", "RS")

# Check hierarchy
get_hierarchy_level("RJ")   # "area"
get_hierarchy_level("NE")   # "subsystem"
get_hierarchy_level("SIN")  # "national"

# Get display name
get_area_name("RJ")    # "Rio de Janeiro"
get_area_name("SECO")  # "Sudeste/Centro-Oeste"
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | get_all_area_codes() default | 26 codes |
| TC-002 | get_all_area_codes() base only | 21 codes |
| TC-003 | get_all_area_codes() with loss | 30 codes |
| TC-004 | validate_area_code valid | TRUE |
| TC-005 | validate_area_code invalid | Error |
| TC-006 | validate_area_codes valid vector | TRUE |
| TC-007 | validate_area_codes mixed | Error on invalid |
| TC-008 | get_subsystem for area | Correct subsystem |
| TC-009 | get_subsystem for subsystem | Returns same |
| TC-010 | get_subsystem for SIN | NA |
| TC-011 | get_areas_in_subsystem | Correct areas |
| TC-012 | is_subsystem TRUE | TRUE |
| TC-013 | is_subsystem FALSE | FALSE |
| TC-014 | get_hierarchy_level | Correct level |
| TC-015 | get_area_name | Display name |

---

## Definition of Done

- [ ] All constants defined
- [ ] All utility functions implemented
- [ ] Area metadata (display names) complete
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥95% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The 21 areas are distribution concession areas, not strictly state boundaries
- Some areas combine multiple states (e.g., ALPE = Alagoas + Pernambuco)
- The system has 26 time series total: 21 areas + 4 subsystems + 1 national
- Loss areas (PESE, PES, etc.) are used for transmission loss calculation
- Subsystem groupings follow ONS (Operador Nacional do Sistema Elétrico) definitions
