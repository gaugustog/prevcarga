# PC-005-01: Resampling

**Epic:** [EPIC-01: Data Layer](../epics/EPIC-01-data-layer.md)
**Task Reference:** T-01.5
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Implement resampling functionality to convert time series data between different temporal resolutions (hourly to semi-hourly and vice versa), with support for multiple interpolation methods.

---

## Acceptance Criteria

- [ ] `Resampler` R6 class created in `R/data/preprocessors.R`
- [ ] `upsample()` converts hourly to semi-hourly (30-min intervals)
- [ ] `downsample()` converts semi-hourly to hourly (aggregation)
- [ ] Support for linear, step, and spline interpolation methods
- [ ] Preserve data integrity (no information loss in round-trip where applicable)
- [ ] Handle edge cases (partial hours, timezone-aware timestamps)

---

## Technical Specification

### File Location
```
R/data/preprocessors.R  (append to existing)
```

### Resampler Class

```r
#' @title Resampler
#' @description Resample time series to different frequencies
#' @export
Resampler <- R6::R6Class(
  "Resampler",
  private = list(
    datetime_col = NULL,
    method = NULL
  ),
  public = list(
    #' @description Initialize resampler
    #' @param datetime_col Name of datetime column
    #' @param method Interpolation method: "linear", "step", "spline"
    initialize = function(datetime_col = "DataHora",
                          method = c("linear", "step", "spline")) {
      private$datetime_col <- datetime_col
      private$method <- match.arg(method)
    },

    #' @description Upsample from hourly to semi-hourly
    #' @param dt data.table with hourly data
    #' @param value_cols Character vector of columns to interpolate
    #' @return data.table with semi-hourly data
    upsample = function(dt, value_cols) {
      checkmate::assert_data_table(dt)
      checkmate::assert_character(value_cols, min.len = 1)

      dt_col <- private$datetime_col
      checkmate::assert_choice(dt_col, names(dt))

      # Ensure sorted
      result <- data.table::copy(dt)
      data.table::setorderv(result, dt_col)

      # Get time range
      time_min <- min(result[[dt_col]])
      time_max <- max(result[[dt_col]])

      # Create semi-hourly sequence
      new_times <- seq(time_min, time_max, by = "30 min")

      # Create new data.table with semi-hourly times
      upsampled <- data.table::data.table(temp_dt_col = new_times)
      data.table::setnames(upsampled, "temp_dt_col", dt_col)

      # Interpolate each value column
      for (col in value_cols) {
        if (!col %in% names(result)) next

        x_orig <- as.numeric(result[[dt_col]])
        y_orig <- result[[col]]

        x_new <- as.numeric(upsampled[[dt_col]])

        interpolated <- private$interpolate(x_orig, y_orig, x_new)
        upsampled[, (col) := interpolated]
      }

      # Copy non-value columns (take from nearest hour)
      other_cols <- setdiff(names(result), c(dt_col, value_cols))
      if (length(other_cols) > 0) {
        # Merge non-value columns from original hourly data
        upsampled <- merge(
          upsampled,
          result[, c(dt_col, other_cols), with = FALSE],
          by = dt_col,
          all.x = TRUE
        )
        # Fill forward for the 30-minute marks
        for (col in other_cols) {
          upsampled[, (col) := data.table::nafill(get(col), type = "locf")]
        }
      }

      upsampled
    },

    #' @description Downsample from semi-hourly to hourly
    #' @param dt data.table with semi-hourly data
    #' @param value_cols Character vector of columns to aggregate
    #' @param agg_fun Aggregation function (default: mean)
    #' @return data.table with hourly data
    downsample = function(dt, value_cols, agg_fun = mean) {
      checkmate::assert_data_table(dt)
      checkmate::assert_character(value_cols, min.len = 1)
      checkmate::assert_function(agg_fun)

      dt_col <- private$datetime_col
      checkmate::assert_choice(dt_col, names(dt))

      result <- data.table::copy(dt)

      # Floor to hour
      result[, hour := lubridate::floor_date(get(dt_col), "hour")]

      # Aggregate value columns
      agg_exprs <- lapply(value_cols, function(col) {
        if (col %in% names(result)) {
          substitute(agg_fun(col_name, na.rm = TRUE), list(col_name = as.name(col)))
        }
      })
      names(agg_exprs) <- value_cols

      # Get first value for non-value columns
      other_cols <- setdiff(names(result), c(dt_col, value_cols, "hour"))
      first_exprs <- lapply(other_cols, function(col) {
        substitute(data.table::first(col_name), list(col_name = as.name(col)))
      })
      names(first_exprs) <- other_cols

      all_exprs <- c(agg_exprs, first_exprs)

      downsampled <- result[, lapply(all_exprs, eval), by = hour]
      data.table::setnames(downsampled, "hour", dt_col)

      downsampled
    }
  ),

  private = list(
    #' Perform interpolation
    interpolate = function(x, y, x_new) {
      non_na <- !is.na(y)
      if (sum(non_na) < 2) {
        return(rep(NA_real_, length(x_new)))
      }

      switch(private$method,
        "linear" = stats::approx(x[non_na], y[non_na], xout = x_new,
                                  method = "linear", rule = 2)$y,
        "step" = stats::approx(x[non_na], y[non_na], xout = x_new,
                               method = "constant", rule = 2)$y,
        "spline" = stats::spline(x[non_na], y[non_na], xout = x_new,
                                  method = "natural")$y
      )
    }
  )
)
```

### Convenience Functions

```r
#' Upsample hourly data to semi-hourly
#'
#' @param dt data.table with hourly data
#' @param value_cols Columns to interpolate
#' @param datetime_col DateTime column name
#' @param method Interpolation method
#' @return data.table with semi-hourly data
#' @export
upsample_to_semihourly <- function(dt, value_cols,
                                    datetime_col = "DataHora",
                                    method = "linear") {
  resampler <- Resampler$new(datetime_col = datetime_col, method = method)
  resampler$upsample(dt, value_cols)
}


#' Downsample semi-hourly data to hourly
#'
#' @param dt data.table with semi-hourly data
#' @param value_cols Columns to aggregate
#' @param datetime_col DateTime column name
#' @param agg_fun Aggregation function
#' @return data.table with hourly data
#' @export
downsample_to_hourly <- function(dt, value_cols,
                                  datetime_col = "DataHora",
                                  agg_fun = mean) {
  resampler <- Resampler$new(datetime_col = datetime_col)
  resampler$downsample(dt, value_cols, agg_fun)
}


#' Detect time series frequency
#'
#' @param dt data.table with datetime column
#' @param datetime_col DateTime column name
#' @return Character: "hourly", "semi-hourly", or "irregular"
#' @export
detect_frequency <- function(dt, datetime_col = "DataHora") {
  checkmate::assert_data_table(dt, min.rows = 2)
  checkmate::assert_choice(datetime_col, names(dt))

  diffs <- diff(as.numeric(dt[[datetime_col]]))
  median_diff <- stats::median(diffs, na.rm = TRUE)

  if (abs(median_diff - 3600) < 60) {
    "hourly"
  } else if (abs(median_diff - 1800) < 60) {
    "semi-hourly"
  } else {
    "irregular"
  }
}
```

### Usage Example

```r
# Create resampler
resampler <- Resampler$new(datetime_col = "DataHora", method = "linear")

# Upsample hourly to semi-hourly
semihourly_data <- resampler$upsample(
  hourly_data,
  value_cols = c("CargaGlobal", "temperatura")
)

# Downsample back to hourly (with mean aggregation)
hourly_again <- resampler$downsample(
  semihourly_data,
  value_cols = c("CargaGlobal", "temperatura"),
  agg_fun = mean
)

# Or use convenience functions
semihourly <- upsample_to_semihourly(hourly_data, c("CargaGlobal"))
hourly <- downsample_to_hourly(semihourly, c("CargaGlobal"))

# Detect frequency
freq <- detect_frequency(data)  # "hourly" or "semi-hourly"
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Upsample hourly to semi-hourly (linear) | Rows doubled, values interpolated |
| TC-002 | Upsample with step method | Values repeated at half-hour |
| TC-003 | Upsample with spline method | Smooth interpolation |
| TC-004 | Downsample semi-hourly to hourly (mean) | Rows halved, values averaged |
| TC-005 | Downsample with sum aggregation | Values summed per hour |
| TC-006 | Round-trip consistency | downsample(upsample(x)) ≈ x |
| TC-007 | Handle NA values | NAs preserved or interpolated |
| TC-008 | Preserve non-numeric columns | Categorical columns carried forward |
| TC-009 | detect_frequency hourly data | Returns "hourly" |
| TC-010 | detect_frequency semi-hourly data | Returns "semi-hourly" |

---

## Definition of Done

- [ ] Resampler class implemented
- [ ] Convenience functions for common operations
- [ ] Frequency detection function
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The system primarily uses hourly data; semi-hourly may be needed for BLF integration
- Spline interpolation can produce values outside the original range; consider clamping
- For aggregation, consider whether to use mean, sum, or other functions depending on the variable type (load = mean, precipitation = sum)
- Time zone handling should be consistent with the rest of the system
