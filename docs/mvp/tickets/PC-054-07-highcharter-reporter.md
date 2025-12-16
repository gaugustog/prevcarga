# PC-054-07: HighcharterReporter

**Epic:** [EPIC-07: Evaluation Layer](../epics/EPIC-07-evaluation-layer.md)
**Task Reference:** T-07.9
**Priority:** Medium
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `HighcharterReporter` R6 class for generating interactive HTML reports using the highcharter package for rich visualizations.

---

## Acceptance Criteria

- [ ] HighcharterReporter in `R/evaluation/highcharter_reporter.R`
- [ ] Interactive time series charts
- [ ] Model comparison visualizations
- [ ] Error distribution charts
- [ ] Template-based report generation
- [ ] Export buttons for charts

---

## Technical Specification

### File Location
```
R/evaluation/highcharter_reporter.R
```

### Package Dependencies
```r
# In DESCRIPTION Imports:
highcharter,
rmarkdown,
htmlwidgets
```

### HighcharterReporter R6 Class

```r
#' @title HighcharterReporter
#' @description Generate interactive HTML reports using highcharter
#'
#' Creates rich, interactive visualizations for forecast evaluation
#' using the Highcharts JavaScript library via highcharter.
#'
#' @export
HighcharterReporter <- R6::R6Class(
  "HighcharterReporter",
  private = list(
    config = NULL,
    template_dir = NULL,
    theme = NULL
  ),
  public = list(
    #' @description Initialize reporter
    #' @param config Report configuration
    initialize = function(config = list()) {
      private$config <- private$apply_default_config(config)
      private$template_dir <- system.file(
        "templates/reports",
        package = "prevcarga"
      )
      private$theme <- private$create_theme()
    },

    #' @description Generate interactive report
    #' @param report_type Type of report
    #' @param data Report data
    #' @param output_path Output file path
    #' @param config Additional configuration
    #' @return Output file path
    generate = function(report_type, data, output_path, config = list()) {
      checkmate::assert_choice(report_type, self$available_reports())
      checkmate::assert_list(data)
      checkmate::assert_string(output_path)

      template <- self$get_template(report_type)

      if (!file.exists(template)) {
        stop(sprintf("Template not found: %s", template))
      }

      # Merge configs
      report_config <- c(private$config, config)

      # Prepare parameters
      params <- list(
        data = data,
        config = report_config,
        theme = private$theme,
        generated_at = Sys.time()
      )

      # Create output directory
      output_dir <- dirname(output_path)
      if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
      }

      # Render
      message(sprintf("Generating %s report...", report_type))

      rmarkdown::render(
        input = template,
        output_file = basename(output_path),
        output_dir = output_dir,
        params = params,
        envir = new.env(),
        quiet = TRUE
      )

      message(sprintf("Report saved to: %s", output_path))
      invisible(output_path)
    },

    #' @description Get template path
    #' @param type Report type
    #' @return Full path to template
    get_template = function(type) {
      template_name <- sprintf("%s_report.Rmd", type)
      file.path(private$template_dir, template_name)
    },

    #' @description List available report types
    #' @return Character vector of report types
    available_reports = function() {
      c("forecast", "backtest", "model_comparison",
        "area_dashboard", "drift")
    },

    #' @description Create forecast vs actual chart
    #' @param data data.table with DataHora, actual, predicted
    #' @param title Chart title
    #' @return highchart object
    chart_forecast_vs_actual = function(data, title = "Forecast vs Actual") {
      highcharter::highchart() %>%
        highcharter::hc_add_series(
          data = data,
          type = "line",
          highcharter::hcaes(x = DataHora, y = actual),
          name = "Actual",
          color = private$config$colors$actual
        ) %>%
        highcharter::hc_add_series(
          data = data,
          type = "line",
          highcharter::hcaes(x = DataHora, y = predicted),
          name = "Forecast",
          color = private$config$colors$forecast,
          dashStyle = "Dash"
        ) %>%
        highcharter::hc_title(text = title) %>%
        highcharter::hc_xAxis(type = "datetime") %>%
        highcharter::hc_yAxis(title = list(text = "Load (MW)")) %>%
        highcharter::hc_tooltip(shared = TRUE, crosshairs = TRUE) %>%
        highcharter::hc_exporting(enabled = TRUE) %>%
        highcharter::hc_add_theme(private$theme)
    },

    #' @description Create error distribution histogram
    #' @param errors Numeric vector of errors
    #' @param title Chart title
    #' @return highchart object
    chart_error_distribution = function(errors, title = "Error Distribution") {
      # Create histogram data
      hist_data <- hist(errors, breaks = 50, plot = FALSE)

      highcharter::highchart() %>%
        highcharter::hc_add_series(
          data = hist_data$counts,
          type = "column",
          name = "Frequency",
          color = private$config$colors$primary
        ) %>%
        highcharter::hc_xAxis(
          categories = round(hist_data$mids, 1),
          title = list(text = "Error")
        ) %>%
        highcharter::hc_yAxis(title = list(text = "Frequency")) %>%
        highcharter::hc_title(text = title) %>%
        highcharter::hc_exporting(enabled = TRUE) %>%
        highcharter::hc_add_theme(private$theme)
    },

    #' @description Create metrics by horizon chart
    #' @param metrics data.table with horizon and metrics
    #' @return highchart object
    chart_metrics_by_horizon = function(metrics) {
      highcharter::highchart() %>%
        highcharter::hc_add_series(
          data = metrics,
          type = "line",
          highcharter::hcaes(x = horizon, y = mape),
          name = "MAPE",
          color = private$config$colors$primary
        ) %>%
        highcharter::hc_xAxis(
          title = list(text = "Horizon"),
          categories = paste0("D+", metrics$horizon)
        ) %>%
        highcharter::hc_yAxis(
          title = list(text = "MAPE (%)"),
          min = 0
        ) %>%
        highcharter::hc_title(text = "MAPE by Forecast Horizon") %>%
        highcharter::hc_exporting(enabled = TRUE) %>%
        highcharter::hc_add_theme(private$theme)
    },

    #' @description Create model comparison bar chart
    #' @param comparison data.table with model and metrics
    #' @param metric Metric to compare
    #' @return highchart object
    chart_model_comparison = function(comparison, metric = "mape") {
      # Sort by metric
      data.table::setorderv(comparison, metric)

      colors <- ifelse(
        seq_len(nrow(comparison)) == 1,
        private$config$colors$success,
        private$config$colors$secondary
      )

      highcharter::highchart() %>%
        highcharter::hc_add_series(
          data = comparison[[metric]],
          type = "bar",
          name = toupper(metric),
          colorByPoint = TRUE,
          colors = colors
        ) %>%
        highcharter::hc_xAxis(
          categories = comparison$model,
          title = list(text = "Model")
        ) %>%
        highcharter::hc_yAxis(
          title = list(text = toupper(metric))
        ) %>%
        highcharter::hc_title(text = sprintf("Model Comparison - %s", toupper(metric))) %>%
        highcharter::hc_exporting(enabled = TRUE) %>%
        highcharter::hc_add_theme(private$theme)
    },

    #' @description Create metrics heatmap
    #' @param metrics data.table with area, hour, and metric
    #' @param metric Metric to display
    #' @return highchart object
    chart_metrics_heatmap = function(metrics, metric = "mape") {
      highcharter::hchart(
        metrics,
        type = "heatmap",
        highcharter::hcaes(x = hour, y = area_code, value = get(metric))
      ) %>%
        highcharter::hc_colorAxis(
          stops = highcharter::color_stops(
            n = 5,
            colors = c("#2ecc71", "#f1c40f", "#e74c3c")
          )
        ) %>%
        highcharter::hc_title(text = sprintf("%s by Hour and Area", toupper(metric))) %>%
        highcharter::hc_xAxis(title = list(text = "Hour")) %>%
        highcharter::hc_yAxis(title = list(text = "Area")) %>%
        highcharter::hc_exporting(enabled = TRUE) %>%
        highcharter::hc_add_theme(private$theme)
    },

    #' @description Create rolling metrics chart for drift detection
    #' @param history data.table with date and metrics
    #' @param threshold Threshold line value
    #' @return highchart object
    chart_rolling_metrics = function(history, threshold = NULL) {
      hc <- highcharter::highchart() %>%
        highcharter::hc_add_series(
          data = history,
          type = "line",
          highcharter::hcaes(x = date, y = mape_rolling),
          name = "Rolling MAPE",
          color = private$config$colors$primary
        ) %>%
        highcharter::hc_xAxis(type = "datetime") %>%
        highcharter::hc_yAxis(
          title = list(text = "MAPE (%)"),
          min = 0
        ) %>%
        highcharter::hc_title(text = "Rolling MAPE (Drift Detection)") %>%
        highcharter::hc_exporting(enabled = TRUE) %>%
        highcharter::hc_add_theme(private$theme)

      if (!is.null(threshold)) {
        hc <- hc %>%
          highcharter::hc_yAxis(
            plotLines = list(list(
              value = threshold,
              color = private$config$colors$error,
              width = 2,
              dashStyle = "Dash",
              label = list(text = "Threshold")
            ))
          )
      }

      hc
    },

    #' @description Save chart to file
    #' @param chart highchart object
    #' @param output_path Output file path
    save_chart = function(chart, output_path) {
      htmlwidgets::saveWidget(
        chart,
        output_path,
        selfcontained = TRUE
      )
      invisible(output_path)
    },

    #' @description Get configuration
    get_config = function() {
      private$config
    },

    #' @description Print summary
    print = function() {
      cat("HighcharterReporter\n")
      cat(sprintf("  Available reports: %s\n",
                  paste(self$available_reports(), collapse = ", ")))
      cat(sprintf("  Template dir: %s\n", private$template_dir))
      invisible(self)
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        width = 900,
        height = 500,
        export_buttons = TRUE,
        colors = list(
          primary = "#3498db",
          secondary = "#95a5a6",
          success = "#2ecc71",
          error = "#e74c3c",
          actual = "#2c3e50",
          forecast = "#3498db"
        )
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    },

    create_theme = function() {
      highcharter::hc_theme(
        chart = list(
          backgroundColor = "#ffffff",
          style = list(fontFamily = "Roboto, sans-serif")
        ),
        title = list(
          style = list(fontSize = "18px", fontWeight = "bold")
        ),
        legend = list(
          itemStyle = list(fontSize = "12px")
        )
      )
    }
  )
)
```

### Convenience Function

```r
#' Create interactive forecast chart
#'
#' @param data data.table with DataHora, actual, predicted
#' @param title Chart title
#' @return highchart object
#' @export
hc_forecast_chart <- function(data, title = "Forecast vs Actual") {
  reporter <- HighcharterReporter$new()
  reporter$chart_forecast_vs_actual(data, title)
}
```

### Usage Example

```r
# Create reporter
reporter <- HighcharterReporter$new()

# Check available reports
reporter$available_reports()
# "forecast" "backtest" "model_comparison" "area_dashboard" "drift"

# Prepare data
forecast_data <- data.table::data.table(
  DataHora = seq(Sys.time(), by = "hour", length.out = 168),
  actual = rnorm(168, 1000, 50),
  predicted = rnorm(168, 1000, 60)
)

data <- list(
  forecast_data = forecast_data,
  metrics = data.table::data.table(mape = 3.5, mae = 35, rmse = 45),
  metrics_by_horizon = data.table::data.table(
    horizon = 0:8,
    mape = c(2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5)
  )
)

# Generate full report
reporter$generate(
  report_type = "forecast",
  data = data,
  output_path = "reports/forecast_20240115.html"
)

# Create individual charts
chart <- reporter$chart_forecast_vs_actual(forecast_data)
print(chart)  # Display in RStudio viewer

# Save chart standalone
reporter$save_chart(chart, "charts/forecast.html")

# Model comparison chart
comparison <- data.table::data.table(
  model = c("LightGBM", "Random Forest", "Holt-Winters"),
  mape = c(3.2, 3.8, 4.5)
)
reporter$chart_model_comparison(comparison)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | Default config |
| TC-002 | available_reports() | 5 report types |
| TC-003 | generate() forecast | HTML created |
| TC-004 | chart_forecast_vs_actual() | highchart object |
| TC-005 | chart_error_distribution() | Histogram chart |
| TC-006 | chart_metrics_by_horizon() | Line chart |
| TC-007 | chart_model_comparison() | Bar chart |
| TC-008 | chart_metrics_heatmap() | Heatmap |
| TC-009 | save_chart() | HTML file saved |
| TC-010 | Custom colors | Colors applied |

---

## Dependencies

- PC-047-07: Metrics Calculator
- highcharter package
- rmarkdown package
- htmlwidgets package

---

## Definition of Done

- [ ] HighcharterReporter R6 class implemented
- [ ] All chart types working
- [ ] Report generation working
- [ ] Export functionality
- [ ] Custom theming support
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Highcharts requires license for commercial use
- Charts are interactive with zoom/pan/export
- Consider adding additional chart types
- May add Plotly alternative in future
