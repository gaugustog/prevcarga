# PC-004-01: Missing Value Imputation

**Epic:** [EPIC-01: Data Layer](../epics/EPIC-01-data-layer.md)
**Task Reference:** T-01.4
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Implement the `ImputerChain` R6 class that provides a composable pipeline for handling missing values in time series data using multiple imputation strategies.

---

## Acceptance Criteria

- [ ] Preprocessors module created in `R/data/preprocessors.R`
- [ ] `ImputerChain` R6 class allows chaining multiple imputers
- [ ] `LOCFImputer` - Last observation carried forward
- [ ] `LinearImputer` - Linear interpolation between known values
- [ ] `SameHourPrevDayImputer` - Same hour from previous day
- [ ] Each imputer logs count of imputed values
- [ ] Chain stops when no missing values remain
- [ ] Original data is not modified (returns new data.table)

---

## Technical Specification

### File Location
```
R/data/preprocessors.R
```

### Base Imputer Class

```r
#' @title BaseImputer
#' @description Abstract base class for imputation strategies
BaseImputer <- R6::R6Class(
  "BaseImputer",
  public = list(
    #' @field name Human-readable imputer name
    name = NULL,

    #' @description Initialize imputer
    #' @param name Imputer name for logging
    initialize = function(name = "BaseImputer") {
      self$name <- name
    },

    #' @description Apply imputation to data
    #' @param dt data.table with missing values
    #' @param columns Character vector of columns to impute
    #' @return data.table with imputed values
    impute = function(dt, columns) {
      stop("impute() must be implemented by subclass")
    }
  )
)
```

### LOCF Imputer

```r
#' @title LOCFImputer
#' @description Last Observation Carried Forward imputation
#' @export
LOCFImputer <- R6::R6Class(
  "LOCFImputer",
  inherit = BaseImputer,
  public = list(
    initialize = function() {
      super$initialize("LOCF")
    },

    impute = function(dt, columns) {
      checkmate::assert_data_table(dt)
      checkmate::assert_character(columns, min.len = 1)

      result <- data.table::copy(dt)

      for (col in columns) {
        if (!col %in% names(result)) next

        na_before <- sum(is.na(result[[col]]))
        if (na_before == 0) next

        # Carry forward last observation
        result[, (col) := data.table::nafill(get(col), type = "locf")]

        na_after <- sum(is.na(result[[col]]))
        if (na_before > na_after) {
          message(sprintf("[%s] %s: imputed %d values",
                          self$name, col, na_before - na_after))
        }
      }

      result
    }
  )
)
```

### Linear Interpolation Imputer

```r
#' @title LinearImputer
#' @description Linear interpolation imputation
#' @export
LinearImputer <- R6::R6Class(
  "LinearImputer",
  inherit = BaseImputer,
  public = list(
    initialize = function() {
      super$initialize("Linear")
    },

    impute = function(dt, columns) {
      checkmate::assert_data_table(dt)
      checkmate::assert_character(columns, min.len = 1)

      result <- data.table::copy(dt)

      for (col in columns) {
        if (!col %in% names(result)) next

        na_before <- sum(is.na(result[[col]]))
        if (na_before == 0) next

        # Linear interpolation using approx
        x <- seq_len(nrow(result))
        y <- result[[col]]
        non_na <- !is.na(y)

        if (sum(non_na) >= 2) {
          interpolated <- stats::approx(
            x[non_na], y[non_na],
            xout = x,
            method = "linear",
            rule = 1  # NA outside bounds
          )$y

          result[is.na(get(col)), (col) := interpolated[is.na(get(col))]]
        }

        na_after <- sum(is.na(result[[col]]))
        if (na_before > na_after) {
          message(sprintf("[%s] %s: imputed %d values",
                          self$name, col, na_before - na_after))
        }
      }

      result
    }
  )
)
```

### Same Hour Previous Day Imputer

```r
#' @title SameHourPrevDayImputer
#' @description Impute using same hour from previous day
#' @export
SameHourPrevDayImputer <- R6::R6Class(
  "SameHourPrevDayImputer",
  inherit = BaseImputer,
  private = list(
    datetime_col = NULL
  ),
  public = list(
    #' @description Initialize imputer
    #' @param datetime_col Name of datetime column (default: "DataHora")
    initialize = function(datetime_col = "DataHora") {
      super$initialize("SameHourPrevDay")
      private$datetime_col <- datetime_col
    },

    impute = function(dt, columns) {
      checkmate::assert_data_table(dt)
      checkmate::assert_character(columns, min.len = 1)
      checkmate::assert_choice(private$datetime_col, names(dt))

      result <- data.table::copy(dt)

      # Ensure sorted by datetime
      data.table::setorderv(result, private$datetime_col)

      # Calculate hours per day for the dataset
      hours_per_day <- 24L

      for (col in columns) {
        if (!col %in% names(result)) next

        na_before <- sum(is.na(result[[col]]))
        if (na_before == 0) next

        # Create shifted column (24 hours back)
        prev_day_values <- data.table::shift(result[[col]], n = hours_per_day)

        # Fill NA with previous day value
        na_idx <- which(is.na(result[[col]]))
        for (i in na_idx) {
          if (!is.na(prev_day_values[i])) {
            data.table::set(result, i, col, prev_day_values[i])
          }
        }

        na_after <- sum(is.na(result[[col]]))
        if (na_before > na_after) {
          message(sprintf("[%s] %s: imputed %d values",
                          self$name, col, na_before - na_after))
        }
      }

      result
    }
  )
)
```

### Imputer Chain

```r
#' @title ImputerChain
#' @description Chain multiple imputers together
#' @export
ImputerChain <- R6::R6Class(
  "ImputerChain",
  private = list(
    imputers = NULL
  ),
  public = list(
    #' @description Initialize chain
    initialize = function() {
      private$imputers <- list()
    },

    #' @description Add imputer to chain
    #' @param imputer BaseImputer instance
    #' @return self (for chaining)
    add = function(imputer) {
      checkmate::assert_class(imputer, "BaseImputer")
      private$imputers <- c(private$imputers, list(imputer))
      invisible(self)
    },

    #' @description Apply all imputers in sequence
    #' @param dt data.table with missing values
    #' @param columns Character vector of columns to impute
    #' @return data.table with imputed values
    impute = function(dt, columns) {
      checkmate::assert_data_table(dt)
      checkmate::assert_character(columns, min.len = 1)

      if (length(private$imputers) == 0) {
        warning("ImputerChain has no imputers configured")
        return(data.table::copy(dt))
      }

      result <- data.table::copy(dt)

      for (imputer in private$imputers) {
        # Check if any missing values remain
        na_count <- sum(sapply(columns, function(col) {
          if (col %in% names(result)) sum(is.na(result[[col]])) else 0
        }))

        if (na_count == 0) {
          message("No missing values remaining, stopping chain")
          break
        }

        result <- imputer$impute(result, columns)
      }

      result
    },

    #' @description Get count of imputers in chain
    #' @return Integer count
    length = function() {
      length(private$imputers)
    }
  )
)
```

### Factory Function

```r
#' Create default imputer chain for load data
#'
#' @return ImputerChain configured for load data imputation
#' @export
create_default_imputer_chain <- function() {
  ImputerChain$new()$
    add(SameHourPrevDayImputer$new())$
    add(LinearImputer$new())$
    add(LOCFImputer$new())
}
```

### Usage Example

```r
# Create chain
chain <- ImputerChain$new()
chain$add(SameHourPrevDayImputer$new())
chain$add(LinearImputer$new())
chain$add(LOCFImputer$new())

# Or use fluent interface
chain <- ImputerChain$new()$
  add(SameHourPrevDayImputer$new())$
  add(LinearImputer$new())$
  add(LOCFImputer$new())

# Or use factory
chain <- create_default_imputer_chain()

# Apply to data
clean_data <- chain$impute(load_data, columns = c("CargaGlobal"))
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | LOCFImputer single gap | Gap filled with previous value |
| TC-002 | LOCFImputer multiple gaps | All gaps filled forward |
| TC-003 | LOCFImputer gap at start | Remains NA (no previous value) |
| TC-004 | LinearImputer single gap | Linear interpolation applied |
| TC-005 | LinearImputer boundary gap | Remains NA (can't interpolate) |
| TC-006 | SameHourPrevDayImputer | Uses value from 24 hours ago |
| TC-007 | SameHourPrevDayImputer no prev day | Remains NA |
| TC-008 | ImputerChain empty | Warning, returns copy |
| TC-009 | ImputerChain stops early | Stops when no NA remain |
| TC-010 | ImputerChain full pipeline | All gaps filled |

---

## Definition of Done

- [ ] All imputer classes implemented
- [ ] ImputerChain with fluent interface
- [ ] Factory function for default chain
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Imputers should be stateless (no fitting required)
- Consider adding `NOCBImputer` (Next Observation Carried Backward) for completeness
- The SameHourPrevDayImputer assumes hourly data; adjust for semi-hourly if needed
- Logging uses `message()` for now; consider using a proper logging framework later
