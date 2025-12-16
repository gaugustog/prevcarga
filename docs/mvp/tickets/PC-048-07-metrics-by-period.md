# PC-048-07: Metrics by Period (Patamar)

**Epic:** [EPIC-07: Evaluation Layer](../epics/EPIC-07-evaluation-layer.md)
**Task Reference:** T-07.2
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Implement metrics calculation by time period (patamar), supporting peak/off-peak breakdown for Brazilian electricity market analysis. Patamares vary by subsystem and month.

---

## Acceptance Criteria

- [ ] Patamar classification function implemented
- [ ] Metrics by patamar calculation
- [ ] Support for subsystem-specific configurations
- [ ] Monthly variation in patamar hours
- [ ] Integration with MetricsCalculator

---

## Technical Specification

### File Location
```
R/evaluation/metrics_period.R
```

### Patamar Classification

```r
#' Classify datetime into patamar (peak/off-peak period)
#'
#' @param datetime POSIXct vector of datetimes
#' @param area_code Area code for subsystem lookup
#' @param config Patamar configuration list
#' @return Character vector with patamar classification
#' @export
classify_patamar <- function(datetime, area_code, config = NULL) {
  checkmate::assert_posixct(datetime)
  checkmate::assert_string(area_code)


  if (is.null(config)) {
    config <- get_default_patamar_config()
  }

  # Get subsystem for area
  subsystem <- get_subsystem_for_area(area_code)

  # Classify each datetime
  vapply(datetime, function(dt) {
    classify_single_datetime(dt, subsystem, config)
  }, character(1))
}


#' Classify a single datetime into patamar
#'
#' @param datetime Single POSIXct value
#' @param subsystem Subsystem code
#' @param config Patamar configuration
#' @return Patamar classification
#' @noRd
classify_single_datetime <- function(datetime, subsystem, config) {
  month_abbr <- format(datetime, "%b")
  hour_min <- format(datetime, "%H:%M")

  # Get patamar hours for this subsystem and month
  patamar_hours <- config[[subsystem]][[month_abbr]]

  if (is.null(patamar_hours)) {
    # Use default if not specified
    patamar_hours <- config$default
  }

  # Check if within peak hours
  ponta_inicio <- patamar_hours$ponta_inicio
  ponta_fim <- patamar_hours$ponta_fim

  if (hour_min >= ponta_inicio && hour_min < ponta_fim) {
    return("ponta")  # Peak
  }

  # Check intermediate period if defined
  if (!is.null(patamar_hours$intermediario_inicio)) {
    inter_inicio <- patamar_hours$intermediario_inicio
    inter_fim <- patamar_hours$intermediario_fim

    if (hour_min >= inter_inicio && hour_min < inter_fim) {
      return("intermediario")  # Intermediate
    }
  }

  return("fora_ponta")  # Off-peak
}


#' Get subsystem for a given area code
#'
#' @param area_code Area code
#' @return Subsystem code
#' @noRd
get_subsystem_for_area <- function(area_code) {
  area_map <- list(
    SECO = c("RJ", "SP", "MG", "ES", "MT", "MS", "AC", "RO", "DF", "GO", "PESE"),
    S = c("PR", "SC", "RS", "PES"),
    NE = c("ALPE", "PBRN", "BASE", "CE", "PI", "BAOE", "PENE"),
    N = c("AM", "PA", "MA", "TO", "RR", "AP", "PEN")
  )

  for (subsystem in names(area_map)) {
    if (area_code %in% area_map[[subsystem]]) {
      return(subsystem)
    }
  }

  # National or unknown
  if (area_code == "SIN") {
    return("SIN")
  }

  warning(sprintf("Unknown area code: %s, using SECO defaults", area_code))
  return("SECO")
}


#' Get default patamar configuration
#'
#' Based on ANEEL/ONS regulations for Brazilian electricity market.
#'
#' @return List with patamar configuration by subsystem and month
#' @export
get_default_patamar_config <- function() {
  # Default peak hours (can vary by month and region)
  list(
    SECO = list(
      default = list(ponta_inicio = "17:00", ponta_fim = "20:00"),
      # Summer months may have different hours
      Dez = list(ponta_inicio = "17:30", ponta_fim = "20:30"),
      Jan = list(ponta_inicio = "17:30", ponta_fim = "20:30"),
      Fev = list(ponta_inicio = "17:30", ponta_fim = "20:30")
    ),
    S = list(
      default = list(ponta_inicio = "18:00", ponta_fim = "21:00"),
      Jun = list(ponta_inicio = "18:00", ponta_fim = "21:00"),
      Jul = list(ponta_inicio = "18:00", ponta_fim = "21:00")
    ),
    NE = list(
      default = list(ponta_inicio = "17:30", ponta_fim = "20:30")
    ),
    N = list(
      default = list(ponta_inicio = "18:00", ponta_fim = "21:00")
    ),
    SIN = list(
      default = list(ponta_inicio = "17:00", ponta_fim = "21:00")
    ),
    default = list(ponta_inicio = "17:00", ponta_fim = "20:00")
  )
}
```

### Metrics by Patamar Calculator

```r
#' Calculate metrics by patamar (peak/off-peak)
#'
#' @param dt data.table with DataHora, actual, predicted columns
#' @param area_code Area code for patamar classification
#' @param patamar_config Optional patamar configuration
#' @return data.table with metrics by patamar
#' @export
calculate_metrics_by_patamar <- function(dt,
                                          area_code = NULL,
                                          patamar_config = NULL) {
  checkmate::assert_data_table(dt)

  # Make a copy to avoid modifying original

  dt <- data.table::copy(dt)

  # Classify patamar
  if ("area_code" %in% names(dt) && is.null(area_code)) {
    # Classify by row-level area_code
    dt[, patamar := classify_patamar(DataHora, area_code, patamar_config),
       by = area_code]
  } else {
    # Use single area_code for all
    area_code <- area_code %||% "SECO"
    dt[, patamar := classify_patamar(DataHora, area_code, patamar_config)]
  }

  # Calculate metrics by patamar
  metrics <- dt[, {
    m <- calculate_all_metrics(actual, predicted)
    data.table::data.table(
      mape = m$mape,
      mae = m$mae,
      rmse = m$rmse,
      mbe = m$mbe,
      n_obs = m$n_observations
    )
  }, by = patamar]

  # Add percentage of observations
  total_obs <- sum(metrics$n_obs)
  metrics[, pct_obs := n_obs / total_obs * 100]

  # Order by patamar type
  patamar_order <- c("ponta", "intermediario", "fora_ponta")
  metrics[, patamar := factor(patamar, levels = patamar_order)]
  data.table::setorder(metrics, patamar)

  metrics
}


#' Calculate metrics by patamar and area
#'
#' @param dt data.table with DataHora, actual, predicted, area_code
#' @param patamar_config Optional patamar configuration
#' @return data.table with metrics by patamar and area
#' @export
calculate_metrics_by_patamar_area <- function(dt, patamar_config = NULL) {
  checkmate::assert_data_table(dt)
  checkmate::assert_true("area_code" %in% names(dt))

  dt <- data.table::copy(dt)

  # Classify patamar for each area
  dt[, patamar := classify_patamar(DataHora, area_code[1], patamar_config),
     by = area_code]

  # Calculate metrics by patamar and area
  metrics <- dt[, {
    m <- calculate_all_metrics(actual, predicted)
    data.table::data.table(
      mape = m$mape,
      mae = m$mae,
      rmse = m$rmse,
      mbe = m$mbe,
      n_obs = m$n_observations
    )
  }, by = .(area_code, patamar)]

  # Order
  data.table::setorder(metrics, area_code, patamar)

  metrics
}
```

### PatamarClassifier R6 Class

```r
#' @title PatamarClassifier
#' @description R6 class for patamar (peak/off-peak) classification
#' @export
PatamarClassifier <- R6::R6Class(
  "PatamarClassifier",
  private = list(
    config = NULL
  ),
  public = list(
    #' @description Initialize classifier
    #' @param config Patamar configuration
    initialize = function(config = NULL) {
      private$config <- config %||% get_default_patamar_config()
    },

    #' @description Classify datetime vector
    #' @param datetime POSIXct vector
    #' @param area_code Area code
    #' @return Character vector of patamar classifications
    classify = function(datetime, area_code) {
      classify_patamar(datetime, area_code, private$config)
    },

    #' @description Check if datetime is peak period
    #' @param datetime POSIXct vector
    #' @param area_code Area code
    #' @return Logical vector
    is_peak = function(datetime, area_code) {
      self$classify(datetime, area_code) == "ponta"
    },

    #' @description Get peak hours for area/month
    #' @param area_code Area code
    #' @param month Month abbreviation (Jan, Feb, etc.)
    #' @return List with ponta_inicio and ponta_fim
    get_peak_hours = function(area_code, month = NULL) {
      subsystem <- get_subsystem_for_area(area_code)

      if (is.null(month)) {
        return(private$config[[subsystem]]$default)
      }

      private$config[[subsystem]][[month]] %||%
        private$config[[subsystem]]$default
    },

    #' @description Get configuration
    get_config = function() {
      private$config
    },

    #' @description Update configuration
    #' @param config New configuration
    set_config = function(config) {
      private$config <- config
      invisible(self)
    },

    #' @description Print summary
    print = function() {
      cat("PatamarClassifier\n")
      cat(sprintf("  Subsystems configured: %d\n",
                  length(private$config) - 1))  # Exclude 'default'
      invisible(self)
    }
  )
)
```

### Usage Example

```r
# Create test data
dt <- data.table::data.table(
  DataHora = seq(
    as.POSIXct("2024-01-15 00:00:00"),
    as.POSIXct("2024-01-15 23:30:00"),
    by = "30 min"
  ),
  actual = rnorm(48, mean = 1000, sd = 100),
  predicted = rnorm(48, mean = 1000, sd = 110)
)

# Classify patamar
dt$patamar <- classify_patamar(dt$DataHora, "RJ")
table(dt$patamar)
# fora_ponta      ponta
#         42          6

# Calculate metrics by patamar
metrics <- calculate_metrics_by_patamar(dt, area_code = "RJ")
#       patamar     mape      mae     rmse   mbe n_obs pct_obs
# 1:      ponta 10.23456 102.3456 125.4567  5.67     6   12.50
# 2: fora_ponta  9.87654  98.7654 118.9876 -3.21    42   87.50

# Use R6 classifier
classifier <- PatamarClassifier$new()
classifier$is_peak(dt$DataHora[1:5], "RJ")
# FALSE FALSE FALSE FALSE FALSE

classifier$get_peak_hours("RJ", "Jan")
# $ponta_inicio: "17:30"
# $ponta_fim: "20:30"
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | classify_patamar() peak hour | "ponta" |
| TC-002 | classify_patamar() off-peak | "fora_ponta" |
| TC-003 | classify_patamar() SECO summer | Summer hours used |
| TC-004 | classify_patamar() S region | Different hours |
| TC-005 | get_subsystem_for_area() RJ | "SECO" |
| TC-006 | get_subsystem_for_area() PR | "S" |
| TC-007 | calculate_metrics_by_patamar() | Metrics per period |
| TC-008 | calculate_metrics_by_patamar_area() | By area and period |
| TC-009 | PatamarClassifier$is_peak() | Logical vector |
| TC-010 | Unknown area fallback | Uses default |

---

## Dependencies

- PC-047-07: Metrics Calculator

---

## Definition of Done

- [ ] Patamar classification implemented
- [ ] Metrics by patamar working
- [ ] Subsystem-specific hours configured
- [ ] Monthly variation supported
- [ ] PatamarClassifier R6 class implemented
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Patamar hours are regulated by ANEEL and vary by subsystem
- Summer daylight saving can affect peak hours
- Consider holidays (may have different patamares)
- Intermediate period exists in some tariff structures
