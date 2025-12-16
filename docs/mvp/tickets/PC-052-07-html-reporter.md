# PC-052-07: HTMLReporter

**Epic:** [EPIC-07: Evaluation Layer](../epics/EPIC-07-evaluation-layer.md)
**Task Reference:** T-07.6
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Implement the `HTMLReporter` R6 class for generating static HTML reports from evaluation results using R Markdown templates.

---

## Acceptance Criteria

- [ ] Reporter module created in `R/evaluation/reporter.R`
- [ ] `HTMLReporter` R6 class with report generation
- [ ] Basic report template in `inst/templates/report.Rmd`
- [ ] Support for metrics, comparisons, and charts
- [ ] Self-contained HTML output
- [ ] Customizable report sections

---

## Technical Specification

### File Location
```
R/evaluation/reporter.R
```

### HTMLReporter R6 Class

```r
#' @title HTMLReporter
#' @description Generate HTML reports from evaluation results
#'
#' Uses R Markdown templates to generate self-contained HTML reports
#' with metrics, tables, and visualizations.
#'
#' @export
HTMLReporter <- R6::R6Class(
  "HTMLReporter",
  private = list(
    config = NULL,
    template_dir = NULL
  ),
  public = list(
    #' @description Initialize reporter
    #' @param config Report configuration
    initialize = function(config = list()) {
      private$config <- private$apply_default_config(config)
      private$template_dir <- system.file(
        "templates",
        package = "prevcarga"
      )
    },

    #' @description Generate HTML report
    #' @param evaluation_results List with evaluation data
    #' @param output_path Output file path
    #' @param template Template name or path
    #' @param title Report title
    #' @return Output file path
    generate = function(evaluation_results,
                        output_path,
                        template = "report",
                        title = NULL) {
      checkmate::assert_list(evaluation_results)
      checkmate::assert_string(output_path)

      # Get template path
      template_path <- self$get_template_path(template)

      if (!file.exists(template_path)) {
        stop(sprintf("Template not found: %s", template_path))
      }

      # Prepare parameters
      params <- list(
        results = evaluation_results,
        title = title %||% private$config$default_title,
        config = private$config,
        generated_at = Sys.time()
      )

      # Create output directory if needed
      output_dir <- dirname(output_path)
      if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
      }

      # Render report
      message(sprintf("Generating report: %s", output_path))

      rmarkdown::render(
        input = template_path,
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
    #' @param template Template name or path
    #' @return Full path to template
    get_template_path = function(template) {
      # Check if it's already a full path
      if (file.exists(template)) {
        return(template)
      }

      # Check in package templates
      pkg_template <- file.path(private$template_dir, paste0(template, ".Rmd"))
      if (file.exists(pkg_template)) {
        return(pkg_template)
      }

      # Check in working directory
      wd_template <- file.path(getwd(), paste0(template, ".Rmd"))
      if (file.exists(wd_template)) {
        return(wd_template)
      }

      stop(sprintf("Template not found: %s", template))
    },

    #' @description List available templates
    #' @return Character vector of template names
    list_templates = function() {
      templates <- list.files(
        private$template_dir,
        pattern = "\\.Rmd$",
        full.names = FALSE
      )

      gsub("\\.Rmd$", "", templates)
    },

    #' @description Generate metrics summary section
    #' @param metrics Metrics data.table
    #' @return HTML string
    render_metrics_table = function(metrics) {
      knitr::kable(
        metrics,
        format = "html",
        digits = 2,
        caption = "Forecast Metrics"
      )
    },

    #' @description Generate comparison table
    #' @param comparison Comparison data.table
    #' @return HTML string
    render_comparison_table = function(comparison) {
      knitr::kable(
        comparison,
        format = "html",
        digits = 2,
        caption = "Model Comparison"
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
      cat("HTMLReporter\n")
      cat(sprintf("  Template dir: %s\n", private$template_dir))
      cat(sprintf("  Available templates: %s\n",
                  paste(self$list_templates(), collapse = ", ")))
      invisible(self)
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        default_title = "Forecast Evaluation Report",
        theme = "flatly",
        toc = TRUE,
        toc_depth = 2,
        code_folding = "hide",
        self_contained = TRUE,
        number_sections = TRUE
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

### Report Template

```
inst/templates/report.Rmd
```

```yaml
---
title: "`r params$title`"
date: "`r format(params$generated_at, '%Y-%m-%d %H:%M:%S')`"
output:
  html_document:
    theme: "`r params$config$theme`"
    toc: `r params$config$toc`
    toc_depth: `r params$config$toc_depth`
    toc_float: true
    code_folding: "`r params$config$code_folding`"
    self_contained: `r params$config$self_contained`
    number_sections: `r params$config$number_sections`
params:
  results: NULL
  title: "Forecast Evaluation Report"
  config: NULL
  generated_at: NULL
---

```r setup, include=FALSE
knitr::opts_chunk$set(
  echo = FALSE,
  warning = FALSE,
  message = FALSE,
  fig.width = 10,
  fig.height = 6
)

library(data.table)
library(ggplot2)

results <- params$results
```

## Executive Summary

This report presents the evaluation results for the forecasting models.

**Report generated:** `r format(params$generated_at, "%Y-%m-%d %H:%M:%S")`

---

## Metrics Summary

```r metrics-table
if (!is.null(results$metrics)) {
  knitr::kable(
    results$metrics,
    digits = 2,
    caption = "Overall Metrics"
  )
}
```

---

## Model Comparison

```r comparison-table
if (!is.null(results$comparison)) {
  knitr::kable(
    results$comparison,
    digits = 2,
    caption = "Model Comparison"
  )
}
```

---

## Metrics by Horizon

```r horizon-plot
if (!is.null(results$metrics_by_horizon)) {
  ggplot(results$metrics_by_horizon, aes(x = horizon, y = mape)) +
    geom_line(color = "steelblue", size = 1) +
    geom_point(color = "steelblue", size = 3) +
    labs(
      title = "MAPE by Forecast Horizon",
      x = "Horizon (days)",
      y = "MAPE (%)"
    ) +
    theme_minimal() +
    scale_x_continuous(breaks = 0:8, labels = paste0("D+", 0:8))
}
```

---

## Metrics by Period

```r patamar-table
if (!is.null(results$metrics_by_patamar)) {
  knitr::kable(
    results$metrics_by_patamar,
    digits = 2,
    caption = "Metrics by Period (Patamar)"
  )
}
```

---

## Forecast vs Actual

```r forecast-plot
if (!is.null(results$forecast_data)) {
  ggplot(results$forecast_data, aes(x = DataHora)) +
    geom_line(aes(y = actual, color = "Actual")) +
    geom_line(aes(y = predicted, color = "Forecast"), linetype = "dashed") +
    labs(
      title = "Forecast vs Actual",
      x = "Date/Time",
      y = "Load (MW)",
      color = ""
    ) +
    theme_minimal() +
    scale_color_manual(values = c("Actual" = "black", "Forecast" = "steelblue"))
}
```

---

## Error Distribution

```r error-hist
if (!is.null(results$errors)) {
  ggplot(data.frame(error = results$errors), aes(x = error)) +
    geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
    geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
    labs(
      title = "Error Distribution",
      x = "Error (Actual - Predicted)",
      y = "Frequency"
    ) +
    theme_minimal()
}
```

---

## Notes

- Report generated by PrevCarga evaluation framework
- All metrics calculated on test set
```

### Usage Example

```r
# Create reporter
reporter <- HTMLReporter$new()

# Prepare evaluation results
evaluation_results <- list(
  metrics = data.table::data.table(
    metric = c("MAPE", "MAE", "RMSE"),
    value = c(3.5, 35.2, 45.6)
  ),
  comparison = data.table::data.table(
    model = c("LightGBM", "Random Forest", "Holt-Winters"),
    mape = c(3.2, 3.8, 4.5),
    mae = c(32, 38, 45),
    rmse = c(42, 48, 55)
  ),
  metrics_by_horizon = data.table::data.table(
    horizon = 0:8,
    mape = c(2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5)
  ),
  forecast_data = data.table::data.table(
    DataHora = seq(Sys.time(), by = "hour", length.out = 48),
    actual = rnorm(48, 1000, 50),
    predicted = rnorm(48, 1000, 60)
  )
)

# Generate report
reporter$generate(
  evaluation_results,
  output_path = "reports/evaluation_2024-01-15.html",
  title = "Forecast Evaluation - January 15, 2024"
)

# List available templates
reporter$list_templates()
# "report"
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | Default config applied |
| TC-002 | generate() basic | HTML file created |
| TC-003 | generate() with custom title | Title in output |
| TC-004 | generate() missing template | Error thrown |
| TC-005 | get_template_path() package | Package path |
| TC-006 | get_template_path() custom | Custom path |
| TC-007 | list_templates() | Template list |
| TC-008 | render_metrics_table() | HTML table |
| TC-009 | Output self-contained | Single file |
| TC-010 | Output directory creation | Created if missing |

---

## Dependencies

- PC-047-07: Metrics Calculator
- rmarkdown package
- knitr package

---

## Definition of Done

- [ ] HTMLReporter R6 class implemented
- [ ] Basic report template created
- [ ] Self-contained HTML output
- [ ] Template discovery working
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Templates use R Markdown with parameterized reports
- Self-contained HTML includes all dependencies
- Consider adding PDF output option in future
- May add custom CSS theming support
