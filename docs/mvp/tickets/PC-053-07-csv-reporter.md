# PC-053-07: CSVReporter

**Epic:** [EPIC-07: Evaluation Layer](../epics/EPIC-07-evaluation-layer.md)
**Task Reference:** T-07.7
**Priority:** Medium
**Estimated Effort:** 0.5 days

---

## Summary

Implement the `CSVReporter` R6 class for exporting evaluation metrics and results to CSV format for further analysis or integration with other tools.

---

## Acceptance Criteria

- [ ] `CSVReporter` R6 class in `R/evaluation/reporter.R`
- [ ] Export metrics to CSV
- [ ] Export comparisons to CSV
- [ ] Support for multiple output files
- [ ] Configurable separators and encoding
- [ ] Timestamp in filenames option

---

## Technical Specification

### File Location
```
R/evaluation/reporter.R (append to existing)
```

### CSVReporter R6 Class

```r
#' @title CSVReporter
#' @description Export evaluation results to CSV format
#'
#' Exports metrics, comparisons, and detailed results to CSV files
#' for further analysis or integration with other tools.
#'
#' @export
CSVReporter <- R6::R6Class(
  "CSVReporter",
  private = list(
    config = NULL
  ),
  public = list(
    #' @description Initialize CSV reporter
    #' @param config Configuration options
    initialize = function(config = list()) {
      private$config <- private$apply_default_config(config)
    },

    #' @description Export evaluation results to CSV
    #' @param evaluation_results List with evaluation data
    #' @param output_path Output file path or directory
    #' @param type Type of export: "single", "multiple"
    #' @return Vector of output file paths
    generate = function(evaluation_results,
                        output_path,
                        type = "single") {
      checkmate::assert_list(evaluation_results)
      checkmate::assert_string(output_path)

      if (type == "single") {
        self$export_single(evaluation_results, output_path)
      } else {
        self$export_multiple(evaluation_results, output_path)
      }
    },

    #' @description Export all results to a single CSV
    #' @param results Evaluation results list
    #' @param output_path Output file path
    #' @return Output file path
    export_single = function(results, output_path) {
      # Combine all metrics into one table
      combined <- self$combine_results(results)

      self$write_csv(combined, output_path)

      message(sprintf("Exported to: %s", output_path))
      invisible(output_path)
    },

    #' @description Export results to multiple CSV files
    #' @param results Evaluation results list
    #' @param output_dir Output directory
    #' @return Vector of output file paths
    export_multiple = function(results, output_dir) {
      # Create output directory if needed
      if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
      }

      output_paths <- character()
      timestamp <- if (private$config$add_timestamp) {
        format(Sys.time(), "_%Y%m%d_%H%M%S")
      } else ""

      # Export each component
      for (name in names(results)) {
        if (is.data.frame(results[[name]]) || is.data.table(results[[name]])) {
          filename <- sprintf("%s%s.csv", name, timestamp)
          filepath <- file.path(output_dir, filename)

          self$write_csv(results[[name]], filepath)
          output_paths <- c(output_paths, filepath)
        }
      }

      message(sprintf("Exported %d files to: %s", length(output_paths), output_dir))
      invisible(output_paths)
    },

    #' @description Combine results into single data.table
    #' @param results Evaluation results list
    #' @return Combined data.table
    combine_results = function(results) {
      combined <- list()

      for (name in names(results)) {
        if (is.data.frame(results[[name]]) || is.data.table(results[[name]])) {
          dt <- data.table::as.data.table(results[[name]])
          dt[, result_type := name]
          combined[[name]] <- dt
        }
      }

      if (length(combined) == 0) {
        stop("No data.frame or data.table objects found in results")
      }

      data.table::rbindlist(combined, fill = TRUE)
    },

    #' @description Export metrics summary
    #' @param metrics Metrics data.table
    #' @param output_path Output file path
    export_metrics = function(metrics, output_path) {
      checkmate::assert_data_table(metrics)
      self$write_csv(metrics, output_path)
    },

    #' @description Export model comparison
    #' @param comparison Comparison data.table
    #' @param output_path Output file path
    export_comparison = function(comparison, output_path) {
      checkmate::assert_data_table(comparison)
      self$write_csv(comparison, output_path)
    },

    #' @description Export time series data
    #' @param ts_data Time series data.table
    #' @param output_path Output file path
    export_timeseries = function(ts_data, output_path) {
      checkmate::assert_data_table(ts_data)
      self$write_csv(ts_data, output_path)
    },

    #' @description Write data.table to CSV
    #' @param dt data.table to export
    #' @param path Output file path
    write_csv = function(dt, path) {
      # Ensure directory exists
      output_dir <- dirname(path)
      if (!dir.exists(output_dir) && output_dir != ".") {
        dir.create(output_dir, recursive = TRUE)
      }

      data.table::fwrite(
        dt,
        file = path,
        sep = private$config$separator,
        na = private$config$na_string,
        quote = private$config$quote,
        dateTimeAs = private$config$datetime_format,
        bom = private$config$bom
      )

      invisible(path)
    },

    #' @description Read CSV back (for verification)
    #' @param path CSV file path
    #' @return data.table
    read_csv = function(path) {
      data.table::fread(
        path,
        sep = private$config$separator,
        na.strings = private$config$na_string
      )
    },

    #' @description Get configuration
    get_config = function() {
      private$config
    },

    #' @description Update configuration
    #' @param config New configuration
    set_config = function(config) {
      private$config <- private$apply_default_config(config)
      invisible(self)
    },

    #' @description Print summary
    print = function() {
      cat("CSVReporter\n")
      cat(sprintf("  Separator: '%s'\n", private$config$separator))
      cat(sprintf("  Add timestamp: %s\n", private$config$add_timestamp))
      cat(sprintf("  Datetime format: %s\n", private$config$datetime_format))
      invisible(self)
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        separator = ",",
        na_string = "NA",
        quote = TRUE,
        datetime_format = "ISO",
        add_timestamp = TRUE,
        bom = FALSE,  # Byte Order Mark for Excel compatibility
        encoding = "UTF-8"
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    }
  )
)
```

### Convenience Functions

```r
#' Export evaluation results to CSV
#'
#' @param results Evaluation results list
#' @param output_path Output file path or directory
#' @param type Export type: "single" or "multiple"
#' @return Output file path(s)
#' @export
export_to_csv <- function(results, output_path, type = "single") {
  reporter <- CSVReporter$new()
  reporter$generate(results, output_path, type)
}


#' Export metrics to CSV
#'
#' @param metrics Metrics data.table
#' @param output_path Output file path
#' @export
export_metrics_csv <- function(metrics, output_path) {
  reporter <- CSVReporter$new()
  reporter$export_metrics(metrics, output_path)
}
```

### Usage Example

```r
# Create reporter
reporter <- CSVReporter$new()

# Or with custom config
reporter <- CSVReporter$new(config = list(
  separator = ";",
  add_timestamp = FALSE
))

# Prepare evaluation results
evaluation_results <- list(
  metrics = data.table::data.table(
    area_code = c("RJ", "SP", "MG"),
    mape = c(3.5, 3.2, 4.1),
    mae = c(35, 32, 41),
    rmse = c(45, 42, 52)
  ),
  comparison = data.table::data.table(
    model = c("lgbm", "rf", "hw"),
    mape = c(3.2, 3.8, 4.5)
  ),
  metrics_by_horizon = data.table::data.table(
    horizon = 0:8,
    mape = c(2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5)
  )
)

# Export to single file
reporter$generate(
  evaluation_results,
  output_path = "results/metrics.csv",
  type = "single"
)

# Export to multiple files
reporter$generate(
  evaluation_results,
  output_path = "results/",
  type = "multiple"
)
# Creates:
#   results/metrics_20240115_143022.csv
#   results/comparison_20240115_143022.csv
#   results/metrics_by_horizon_20240115_143022.csv

# Export specific metrics
reporter$export_metrics(
  evaluation_results$metrics,
  "results/area_metrics.csv"
)

# Convenience function
export_to_csv(evaluation_results, "results/all.csv")
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | Default config |
| TC-002 | generate() single file | One CSV created |
| TC-003 | generate() multiple files | Multiple CSVs |
| TC-004 | export_metrics() | Metrics exported |
| TC-005 | Custom separator | Correct delimiter |
| TC-006 | add_timestamp option | Timestamp in filename |
| TC-007 | Directory creation | Created if missing |
| TC-008 | read_csv() verification | Data matches |
| TC-009 | NA handling | Correct NA string |
| TC-010 | combine_results() | Merged data.table |

---

## Dependencies

- data.table package

---

## Definition of Done

- [ ] CSVReporter R6 class implemented
- [ ] Single and multiple file export
- [ ] Configurable options working
- [ ] Convenience functions implemented
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- CSV is widely compatible for data exchange
- Consider adding Excel export option
- BOM option helps Excel recognize UTF-8
- Timestamps prevent accidental overwrites
