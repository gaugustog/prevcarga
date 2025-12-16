# PC-014-02: Common Feature Utilities

**Epic:** [EPIC-02: Feature Engineering Infrastructure](../epics/EPIC-02-feature-engineering.md)
**Task Reference:** T-02.6
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Create utility functions that plugin developers can use to build feature plugins. These are common operations like lag creation, rolling statistics, cyclical encoding, and one-hot encoding.

---

## Acceptance Criteria

- [ ] Common utilities module created in `R/features/common.R`
- [ ] `cyclical_encode()` - sin/cos encoding for cyclical features
- [ ] `create_lags()` - create lagged features for time series
- [ ] `create_rolling_stats()` - rolling window statistics
- [ ] `one_hot_encode()` - create dummy variables
- [ ] `create_interactions()` - feature interactions
- [ ] All functions work with data.table efficiently
- [ ] roxygen2 documentation with examples

---

## Technical Specification

### File Location
```
R/features/common.R
```

### Cyclical Encoding

```r
#' Encode cyclical features using sin/cos transformation
#'
#' Transforms a numeric value into two columns representing the sin and cos
#' of the value scaled to a period. This preserves the cyclical nature
#' (e.g., hour 23 is close to hour 0).
#'
#' @param values Numeric vector of values to encode
#' @param period The period of the cycle (e.g., 24 for hours, 7 for days)
#' @param prefix Prefix for output column names
#' @return data.table with sin and cos columns
#' @export
#' @examples
#' # Encode hours (0-23)
#' hours <- 0:23
#' encoded <- cyclical_encode(hours, period = 24, prefix = "hour")
#' # Returns: hour_sin, hour_cos
#'
#' # Encode day of week (1-7)
#' days <- 1:7
#' encoded <- cyclical_encode(days, period = 7, prefix = "dow")
cyclical_encode <- function(values, period, prefix = "cyclical") {
 checkmate::assert_numeric(values)
 checkmate::assert_number(period, lower = 1)
 checkmate::assert_string(prefix)

 # Scale to radians
 radians <- 2 * pi * values / period

 data.table::data.table(
   sin_col = sin(radians),
   cos_col = cos(radians)
 ) |>
   data.table::setnames(c(
     sprintf("%s_sin", prefix),
     sprintf("%s_cos", prefix)
   ))
}


#' Add cyclical encoding columns to data.table
#'
#' @param dt data.table to modify
#' @param column Column name to encode
#' @param period Cycle period
#' @param prefix Prefix for new columns (default: column name)
#' @return data.table with new columns added
#' @export
add_cyclical_encoding <- function(dt, column, period, prefix = NULL) {
 checkmate::assert_data_table(dt)
 checkmate::assert_choice(column, names(dt))

 prefix <- prefix %||% column
 encoded <- cyclical_encode(dt[[column]], period, prefix)

 result <- data.table::copy(dt)
 for (col in names(encoded)) {
   result[, (col) := encoded[[col]]]
 }

 result
}
```

### Lag Features

```r
#' Create lagged features for time series
#'
#' Creates multiple lagged versions of specified columns. Handles
#' proper sorting by datetime and optional grouping.
#'
#' @param dt data.table with time series data
#' @param columns Character vector of columns to lag
#' @param lags Integer vector of lag values (positive = past)
#' @param datetime_col DateTime column for sorting
#' @param group_cols Optional grouping columns (e.g., area_code)
#' @param suffix_format Format string for lag suffix (default: "_lag%d")
#' @return data.table with lag columns added
#' @export
#' @examples
#' # Create lag features
#' result <- create_lags(
#'   dt = load_data,
#'   columns = "CargaGlobal",
#'   lags = c(1, 24, 48, 168),
#'   datetime_col = "DataHora",
#'   group_cols = "area_code"
#' )
#' # Adds: CargaGlobal_lag1, CargaGlobal_lag24, etc.
create_lags <- function(dt, columns, lags,
                        datetime_col = "DataHora",
                        group_cols = NULL,
                        suffix_format = "_lag%d") {
 checkmate::assert_data_table(dt)
 checkmate::assert_character(columns, min.len = 1)
 checkmate::assert_integerish(lags, lower = 1, min.len = 1)
 checkmate::assert_choice(datetime_col, names(dt))

 result <- data.table::copy(dt)

 # Ensure sorted by datetime within groups
 if (!is.null(group_cols)) {
   data.table::setorderv(result, c(group_cols, datetime_col))
 } else {
   data.table::setorderv(result, datetime_col)
 }

 # Create lag features
 for (col in columns) {
   if (!col %in% names(result)) {
     warning(sprintf("Column '%s' not found, skipping", col))
     next
   }

   for (lag in lags) {
     lag_name <- sprintf("%s%s", col, sprintf(suffix_format, lag))

     if (!is.null(group_cols)) {
       result[, (lag_name) := data.table::shift(get(col), n = lag, type = "lag"),
              by = group_cols]
     } else {
       result[, (lag_name) := data.table::shift(get(col), n = lag, type = "lag")]
     }
   }
 }

 result
}


#' Create lead features (future values)
#'
#' @param dt data.table
#' @param columns Columns to lead
#' @param leads Lead values
#' @param datetime_col DateTime column
#' @param group_cols Grouping columns
#' @return data.table with lead columns
#' @export
create_leads <- function(dt, columns, leads,
                         datetime_col = "DataHora",
                         group_cols = NULL) {
 checkmate::assert_data_table(dt)
 checkmate::assert_character(columns, min.len = 1)
 checkmate::assert_integerish(leads, lower = 1, min.len = 1)

 result <- data.table::copy(dt)

 if (!is.null(group_cols)) {
   data.table::setorderv(result, c(group_cols, datetime_col))
 } else {
   data.table::setorderv(result, datetime_col)
 }

 for (col in columns) {
   if (!col %in% names(result)) next

   for (lead in leads) {
     lead_name <- sprintf("%s_lead%d", col, lead)

     if (!is.null(group_cols)) {
       result[, (lead_name) := data.table::shift(get(col), n = lead, type = "lead"),
              by = group_cols]
     } else {
       result[, (lead_name) := data.table::shift(get(col), n = lead, type = "lead")]
     }
   }
 }

 result
}
```

### Rolling Statistics

```r
#' Create rolling window statistics
#'
#' Computes statistics over rolling windows for specified columns.
#' Supports mean, sum, min, max, sd, and custom functions.
#'
#' @param dt data.table
#' @param columns Columns to compute statistics for
#' @param windows Window sizes (in number of observations)
#' @param stats Statistics to compute: "mean", "sum", "min", "max", "sd"
#' @param datetime_col DateTime column for ordering
#' @param group_cols Grouping columns
#' @param align Window alignment: "right" (default), "center", "left"
#' @return data.table with rolling statistic columns
#' @export
#' @examples
#' result <- create_rolling_stats(
#'   dt = load_data,
#'   columns = "CargaGlobal",
#'   windows = c(24, 168),
#'   stats = c("mean", "sd"),
#'   group_cols = "area_code"
#' )
#' # Adds: CargaGlobal_roll24_mean, CargaGlobal_roll24_sd, etc.
create_rolling_stats <- function(dt, columns, windows,
                                  stats = c("mean", "sd"),
                                  datetime_col = "DataHora",
                                  group_cols = NULL,
                                  align = "right") {
 checkmate::assert_data_table(dt)
 checkmate::assert_character(columns, min.len = 1)
 checkmate::assert_integerish(windows, lower = 2, min.len = 1)
 checkmate::assert_subset(stats, c("mean", "sum", "min", "max", "sd", "var"))
 checkmate::assert_choice(align, c("right", "center", "left"))

 result <- data.table::copy(dt)

 # Sort
 if (!is.null(group_cols)) {
   data.table::setorderv(result, c(group_cols, datetime_col))
 } else {
   data.table::setorderv(result, datetime_col)
 }

 # Stat functions
 stat_funs <- list(
   mean = function(x, n) data.table::frollmean(x, n, align = align, na.rm = TRUE),
   sum = function(x, n) data.table::frollsum(x, n, align = align, na.rm = TRUE),
   min = function(x, n) frollapply(x, n, min, align = align, na.rm = TRUE),
   max = function(x, n) frollapply(x, n, max, align = align, na.rm = TRUE),
   sd = function(x, n) frollapply(x, n, sd, align = align, na.rm = TRUE),
   var = function(x, n) frollapply(x, n, var, align = align, na.rm = TRUE)
 )

 for (col in columns) {
   if (!col %in% names(result)) next

   for (window in windows) {
     for (stat in stats) {
       col_name <- sprintf("%s_roll%d_%s", col, window, stat)

       if (!is.null(group_cols)) {
         result[, (col_name) := stat_funs[[stat]](get(col), window),
                by = group_cols]
       } else {
         result[, (col_name) := stat_funs[[stat]](get(col), window)]
       }
     }
   }
 }

 result
}


#' Generic rolling apply function
#'
#' @param x Numeric vector
#' @param n Window size
#' @param FUN Function to apply
#' @param align Alignment
#' @param ... Additional arguments to FUN
#' @return Numeric vector
#' @noRd
frollapply <- function(x, n, FUN, align = "right", ...) {
 result <- rep(NA_real_, length(x))

 for (i in seq_len(length(x))) {
   if (align == "right") {
     start <- max(1, i - n + 1)
     end <- i
   } else if (align == "left") {
     start <- i
     end <- min(length(x), i + n - 1)
   } else {
     half <- floor(n / 2)
     start <- max(1, i - half)
     end <- min(length(x), i + half)
   }

   if (end - start + 1 >= n) {
     result[i] <- FUN(x[start:end], ...)
   }
 }

 result
}
```

### One-Hot Encoding

```r
#' One-hot encode categorical column
#'
#' Creates binary dummy variables for each category. Optionally
#' drops one category to avoid multicollinearity.
#'
#' @param dt data.table
#' @param column Column to encode
#' @param drop_first Drop first category (for regression)
#' @param prefix Prefix for dummy columns
#' @param sparse Return sparse representation (list of indices)
#' @return data.table with dummy columns
#' @export
#' @examples
#' result <- one_hot_encode(
#'   dt = data,
#'   column = "day_type",
#'   drop_first = TRUE
#' )
one_hot_encode <- function(dt, column,
                           drop_first = FALSE,
                           prefix = NULL,
                           sparse = FALSE) {
 checkmate::assert_data_table(dt)
 checkmate::assert_choice(column, names(dt))

 result <- data.table::copy(dt)
 prefix <- prefix %||% column

 # Get unique categories
 categories <- sort(unique(na.omit(dt[[column]])))

 if (drop_first && length(categories) > 1) {
   categories <- categories[-1]
 }

 # Create dummy columns
 for (cat in categories) {
   col_name <- sprintf("%s_%s", prefix, make.names(as.character(cat)))
   result[, (col_name) := as.integer(get(column) == cat)]
 }

 result
}


#' Create interaction features
#'
#' Creates multiplicative interactions between numeric features.
#'
#' @param dt data.table
#' @param columns Column names to interact
#' @param degree Interaction degree (2 = pairwise, 3 = triplets)
#' @param include_self Include self-interactions (squares)
#' @return data.table with interaction columns
#' @export
create_interactions <- function(dt, columns,
                                 degree = 2,
                                 include_self = FALSE) {
 checkmate::assert_data_table(dt)
 checkmate::assert_character(columns, min.len = 2)
 checkmate::assert_integerish(degree, lower = 2, upper = 3)

 result <- data.table::copy(dt)

 # Filter to numeric columns that exist
 numeric_cols <- columns[columns %in% names(result)]
 numeric_cols <- numeric_cols[sapply(result[, ..numeric_cols], is.numeric)]

 # Generate combinations
 if (include_self) {
   combos <- utils::combn(c(numeric_cols, numeric_cols), degree, simplify = FALSE)
   combos <- unique(lapply(combos, sort))
 } else {
   combos <- utils::combn(numeric_cols, degree, simplify = FALSE)
 }

 for (combo in combos) {
   col_name <- paste(combo, collapse = "_x_")

   # Compute product
   result[, (col_name) := Reduce(`*`, .SD), .SDcols = combo]
 }

 result
}
```

### Difference Features

```r
#' Create difference features
#'
#' Computes differences between current and lagged values.
#'
#' @param dt data.table
#' @param columns Columns to difference
#' @param lags Difference lags
#' @param datetime_col DateTime column
#' @param group_cols Grouping columns
#' @param pct Use percentage change instead of absolute
#' @return data.table with difference columns
#' @export
create_differences <- function(dt, columns, lags = 1,
                                datetime_col = "DataHora",
                                group_cols = NULL,
                                pct = FALSE) {
 checkmate::assert_data_table(dt)
 checkmate::assert_character(columns, min.len = 1)
 checkmate::assert_integerish(lags, lower = 1)

 result <- data.table::copy(dt)

 if (!is.null(group_cols)) {
   data.table::setorderv(result, c(group_cols, datetime_col))
 } else {
   data.table::setorderv(result, datetime_col)
 }

 suffix <- if (pct) "_pct" else "_diff"

 for (col in columns) {
   if (!col %in% names(result)) next

   for (lag in lags) {
     col_name <- sprintf("%s%s%d", col, suffix, lag)

     if (!is.null(group_cols)) {
       lagged <- result[, data.table::shift(get(col), n = lag), by = group_cols]$V1
     } else {
       lagged <- data.table::shift(result[[col]], n = lag)
     }

     if (pct) {
       result[, (col_name) := (get(col) - lagged) / lagged]
     } else {
       result[, (col_name) := get(col) - lagged]
     }
   }
 }

 result
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | cyclical_encode hours | sin/cos values correct |
| TC-002 | cyclical_encode boundary | Hour 0 ≈ Hour 24 |
| TC-003 | create_lags single column | Correct lag values |
| TC-004 | create_lags with groups | Independent per group |
| TC-005 | create_lags with NA fill | First values are NA |
| TC-006 | create_rolling_stats mean | Correct rolling mean |
| TC-007 | create_rolling_stats multiple | All stats computed |
| TC-008 | one_hot_encode basic | Dummy columns created |
| TC-009 | one_hot_encode drop_first | One less category |
| TC-010 | create_interactions pairwise | All pairs created |
| TC-011 | create_differences absolute | Correct diffs |
| TC-012 | create_differences percent | Correct pct change |
| TC-013 | add_cyclical_encoding | Columns added to dt |
| TC-014 | create_leads | Future values correct |

---

## Definition of Done

- [ ] All utility functions implemented
- [ ] Efficient data.table operations
- [ ] Proper handling of NA values
- [ ] Group-aware computations working
- [ ] roxygen2 documentation with examples
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- All functions should work efficiently with large data.tables
- Consider adding parallel processing options for large datasets
- These utilities are building blocks; plugins compose them into features
- Ensure memory efficiency by avoiding unnecessary copies
- Group-aware operations are critical for multi-area forecasting
