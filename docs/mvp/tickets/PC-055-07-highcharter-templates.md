# PC-055-07: Highcharter Report Templates

**Epic:** [EPIC-07: Evaluation Layer](../epics/EPIC-07-evaluation-layer.md)
**Task Reference:** T-07.10
**Priority:** Medium
**Estimated Effort:** 2 days

---

## Summary

Create R Markdown report templates for the HighcharterReporter, including forecast analysis, backtest results, model comparison, area dashboards, and drift detection reports.

---

## Acceptance Criteria

- [ ] Template directory created at `inst/templates/reports/`
- [ ] Common helpers in `_common.R`
- [ ] 5 report templates implemented
- [ ] All templates render without errors
- [ ] Consistent styling across templates
- [ ] Export buttons functional

---

## Technical Specification

### Directory Structure

```
inst/templates/reports/
├── _common.R                    # Shared helpers
├── forecast_report.Rmd          # Forecast analysis
├── backtest_report.Rmd          # Backtest results
├── model_comparison_report.Rmd  # Model comparison
├── area_dashboard_report.Rmd    # Area deep-dive
└── drift_report.Rmd             # Drift analysis
```

### Common Helpers

```r
# inst/templates/reports/_common.R

#' Common highcharter helpers for reports
#' @noRd

# Color palette
PREVCARGA_COLORS <- list(
  primary = "#3498db",
  secondary = "#95a5a6",
  success = "#2ecc71",
  warning = "#f1c40f",
  error = "#e74c3c",
  actual = "#2c3e50",
  forecast = "#3498db",
  gradient = c("#2ecc71", "#f1c40f", "#e74c3c")
)

#' Create prevcarga highcharter theme
#' @return hc_theme object
prevcarga_theme <- function() {
  highcharter::hc_theme(
    chart = list(
      backgroundColor = "#ffffff",
      style = list(fontFamily = "'Roboto', 'Helvetica', sans-serif")
    ),
    title = list(
      style = list(
        fontSize = "18px",
        fontWeight = "600",
        color = "#2c3e50"
      )
    ),
    subtitle = list(
      style = list(
        fontSize = "12px",
        color = "#7f8c8d"
      )
    ),
    legend = list(
      itemStyle = list(
        fontSize = "12px",
        color = "#34495e"
      )
    ),
    xAxis = list(
      labels = list(style = list(color = "#7f8c8d")),
      title = list(style = list(color = "#34495e"))
    ),
    yAxis = list(
      labels = list(style = list(color = "#7f8c8d")),
      title = list(style = list(color = "#34495e")),
      gridLineColor = "#ecf0f1"
    )
  )
}

#' Format metrics table for display
#' @param metrics data.table with metrics
#' @return kable object
format_metrics_table <- function(metrics) {
  knitr::kable(
    metrics,
    digits = 2,
    format = "html",
    table.attr = 'class="table table-striped table-hover"'
  ) %>%
    kableExtra::kable_styling(
      bootstrap_options = c("striped", "hover", "condensed"),
      full_width = FALSE
    )
}

#' Create summary card HTML
#' @param title Card title
#' @param value Card value
#' @param color Card color
#' @return HTML string
summary_card <- function(title, value, color = "primary") {
  sprintf(
    '<div class="card text-white bg-%s mb-3" style="max-width: 18rem;">
      <div class="card-body">
        <h5 class="card-title">%s</h5>
        <p class="card-text" style="font-size: 2em;">%s</p>
      </div>
    </div>',
    color, title, value
  )
}
```

### Forecast Report Template

```yaml
# inst/templates/reports/forecast_report.Rmd
---
title: "Forecast Report"
subtitle: "`r format(params$data$report_date, '%B %d, %Y')`"
output:
  html_document:
    theme: flatly
    toc: true
    toc_float: true
    toc_depth: 2
    self_contained: true
    code_folding: hide
params:
  data: NULL
  config: NULL
  theme: NULL
  generated_at: NULL
---
```

```r setup, include=FALSE
knitr::opts_chunk$set(
  echo = FALSE,
  warning = FALSE,
  message = FALSE,
  fig.width = 10,
  fig.height = 6
)

library(highcharter)
library(data.table)
library(knitr)
library(kableExtra)

source("_common.R")

data <- params$data
config <- params$config
```

# Executive Summary

```r summary-cards, results='asis'
if (!is.null(data$metrics)) {
  cat('<div class="row">')
  cat(summary_card("MAPE", sprintf("%.2f%%", data$metrics$mape), "primary"))
  cat(summary_card("MAE", sprintf("%.1f MW", data$metrics$mae), "info"))
  cat(summary_card("RMSE", sprintf("%.1f MW", data$metrics$rmse), "secondary"))
  cat('</div>')
}
```

---

# Forecast vs Actual

```r forecast-chart
if (!is.null(data$forecast_data)) {
  highchart() %>%
    hc_add_series(
      data = data$forecast_data,
      type = "line",
      hcaes(x = DataHora, y = actual),
      name = "Actual",
      color = PREVCARGA_COLORS$actual
    ) %>%
    hc_add_series(
      data = data$forecast_data,
      type = "line",
      hcaes(x = DataHora, y = predicted),
      name = "Forecast",
      color = PREVCARGA_COLORS$forecast,
      dashStyle = "Dash"
    ) %>%
    hc_xAxis(type = "datetime", title = list(text = "Date/Time")) %>%
    hc_yAxis(title = list(text = "Load (MW)")) %>%
    hc_tooltip(shared = TRUE, crosshairs = TRUE) %>%
    hc_exporting(enabled = TRUE) %>%
    hc_add_theme(prevcarga_theme())
}
```

---

# Error Distribution

```r error-chart
if (!is.null(data$forecast_data)) {
  errors <- data$forecast_data$actual - data$forecast_data$predicted

  hchart(errors, type = "histogram", name = "Error") %>%
    hc_xAxis(title = list(text = "Error (Actual - Predicted)")) %>%
    hc_yAxis(title = list(text = "Frequency")) %>%
    hc_exporting(enabled = TRUE) %>%
    hc_add_theme(prevcarga_theme())
}
```

---

# Metrics by Hour

```r metrics-hour
if (!is.null(data$metrics_by_hour)) {
  highchart() %>%
    hc_add_series(
      data = data$metrics_by_hour,
      type = "column",
      hcaes(x = hour, y = mape),
      name = "MAPE",
      color = PREVCARGA_COLORS$primary
    ) %>%
    hc_xAxis(categories = 0:23, title = list(text = "Hour")) %>%
    hc_yAxis(title = list(text = "MAPE (%)")) %>%
    hc_exporting(enabled = TRUE) %>%
    hc_add_theme(prevcarga_theme())
}
```

---

# Detailed Metrics

```r metrics-table
if (!is.null(data$detailed_metrics)) {
  format_metrics_table(data$detailed_metrics)
}
```

---

*Report generated: `r format(params$generated_at, "%Y-%m-%d %H:%M:%S")`*

### Backtest Report Template

```yaml
# inst/templates/reports/backtest_report.Rmd
---
title: "Backtest Report"
subtitle: "`r params$data$backtest_id`"
output:
  html_document:
    theme: flatly
    toc: true
    toc_float: true
    self_contained: true
params:
  data: NULL
  config: NULL
  theme: NULL
  generated_at: NULL
---
```

```r setup, include=FALSE
# Standard setup
knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)
library(highcharter)
library(data.table)
source("_common.R")

data <- params$data
```

# Backtest Summary

- **Period:** `r data$start_date` to `r data$end_date`
- **Model:** `r data$model_name`
- **Retraining:** `r data$retrain_frequency`

---

# Metrics by Horizon

```r horizon-chart
if (!is.null(data$metrics_by_horizon)) {
  highchart() %>%
    hc_add_series(
      data = data$metrics_by_horizon,
      type = "line",
      hcaes(x = horizon, y = mape),
      name = "MAPE",
      color = PREVCARGA_COLORS$primary,
      marker = list(enabled = TRUE, radius = 5)
    ) %>%
    hc_xAxis(
      categories = paste0("D+", data$metrics_by_horizon$horizon),
      title = list(text = "Horizon")
    ) %>%
    hc_yAxis(title = list(text = "MAPE (%)"), min = 0) %>%
    hc_exporting(enabled = TRUE) %>%
    hc_add_theme(prevcarga_theme())
}
```

---

# Rolling Performance

```r rolling-chart
if (!is.null(data$rolling_metrics)) {
  highchart() %>%
    hc_add_series(
      data = data$rolling_metrics,
      type = "line",
      hcaes(x = date, y = mape),
      name = "Daily MAPE",
      color = PREVCARGA_COLORS$secondary
    ) %>%
    hc_add_series(
      data = data$rolling_metrics,
      type = "line",
      hcaes(x = date, y = mape_rolling),
      name = "7-Day Rolling",
      color = PREVCARGA_COLORS$primary
    ) %>%
    hc_xAxis(type = "datetime") %>%
    hc_yAxis(title = list(text = "MAPE (%)")) %>%
    hc_exporting(enabled = TRUE) %>%
    hc_add_theme(prevcarga_theme())
}
```

### Model Comparison Template

```yaml
# inst/templates/reports/model_comparison_report.Rmd
---
title: "Model Comparison Report"
output:
  html_document:
    theme: flatly
    toc: true
    toc_float: true
    self_contained: true
params:
  data: NULL
  config: NULL
  theme: NULL
  generated_at: NULL
---
```

```r comparison-chart
if (!is.null(data$comparison)) {
  # Sort by MAPE
  comp <- data$comparison[order(mape)]

  colors <- ifelse(
    seq_len(nrow(comp)) == 1,
    PREVCARGA_COLORS$success,
    PREVCARGA_COLORS$secondary
  )

  highchart() %>%
    hc_add_series(
      data = comp$mape,
      type = "bar",
      name = "MAPE",
      colorByPoint = TRUE,
      colors = colors
    ) %>%
    hc_xAxis(categories = comp$model) %>%
    hc_yAxis(title = list(text = "MAPE (%)")) %>%
    hc_title(text = "Model Ranking by MAPE") %>%
    hc_exporting(enabled = TRUE) %>%
    hc_add_theme(prevcarga_theme())
}
```

### Area Dashboard Template

```yaml
# inst/templates/reports/area_dashboard_report.Rmd
---
title: "`r paste0(params$data$area_code, ' Dashboard')`"
output:
  html_document:
    theme: flatly
    toc: true
    toc_float: true
    self_contained: true
params:
  data: NULL
  config: NULL
  theme: NULL
  generated_at: NULL
---
```

```r load-profile
if (!is.null(data$load_data)) {
  highchart() %>%
    hc_add_series(
      data = data$load_data,
      type = "line",
      hcaes(x = DataHora, y = carga),
      name = "Load",
      color = PREVCARGA_COLORS$primary
    ) %>%
    hc_xAxis(type = "datetime") %>%
    hc_yAxis(title = list(text = "Load (MW)")) %>%
    hc_title(text = "Load Profile") %>%
    hc_exporting(enabled = TRUE) %>%
    hc_add_theme(prevcarga_theme())
}
```

### Drift Report Template

```yaml
# inst/templates/reports/drift_report.Rmd
---
title: "Drift Detection Report"
subtitle: "`r params$data$model_name`"
output:
  html_document:
    theme: flatly
    toc: true
    toc_float: true
    self_contained: true
params:
  data: NULL
  config: NULL
  theme: NULL
  generated_at: NULL
---
```

```r drift-chart
if (!is.null(data$history)) {
  baseline_mape <- data$baseline$mape
  threshold <- baseline_mape * data$threshold_ratio

  highchart() %>%
    hc_add_series(
      data = data$history,
      type = "line",
      hcaes(x = date, y = mape_rolling),
      name = "Rolling MAPE",
      color = PREVCARGA_COLORS$primary
    ) %>%
    hc_yAxis(
      title = list(text = "MAPE (%)"),
      plotLines = list(
        list(
          value = baseline_mape,
          color = PREVCARGA_COLORS$success,
          width = 2,
          dashStyle = "Dash",
          label = list(text = "Baseline")
        ),
        list(
          value = threshold,
          color = PREVCARGA_COLORS$error,
          width = 2,
          dashStyle = "Dash",
          label = list(text = "Threshold")
        )
      )
    ) %>%
    hc_xAxis(type = "datetime") %>%
    hc_exporting(enabled = TRUE) %>%
    hc_add_theme(prevcarga_theme())
}
```

---

## Usage Example

```r
# Generate forecast report
reporter <- HighcharterReporter$new()

forecast_data <- list(
  report_date = Sys.Date(),
  forecast_data = data.table(
    DataHora = seq(Sys.time(), by = "hour", length.out = 168),
    actual = rnorm(168, 1000, 50),
    predicted = rnorm(168, 1000, 60)
  ),
  metrics = list(mape = 3.5, mae = 35, rmse = 45),
  metrics_by_hour = data.table(
    hour = 0:23,
    mape = rnorm(24, 3.5, 0.5)
  )
)

reporter$generate(
  report_type = "forecast",
  data = forecast_data,
  output_path = "reports/forecast_2024-01-15.html"
)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | forecast_report renders | HTML created |
| TC-002 | backtest_report renders | HTML created |
| TC-003 | model_comparison_report renders | HTML created |
| TC-004 | area_dashboard_report renders | HTML created |
| TC-005 | drift_report renders | HTML created |
| TC-006 | _common.R loads | No errors |
| TC-007 | Charts export | Export works |
| TC-008 | Missing data handled | Graceful fallback |
| TC-009 | Theme applied | Consistent styling |
| TC-010 | Self-contained output | Single file |

---

## Dependencies

- PC-054-07: HighcharterReporter
- highcharter package
- rmarkdown package
- knitr package
- kableExtra package

---

## Definition of Done

- [ ] All 5 templates created
- [ ] Common helpers implemented
- [ ] All templates render without errors
- [ ] Export functionality working
- [ ] Consistent theming
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Templates are parameterized R Markdown
- Self-contained HTML for easy sharing
- Export buttons allow saving as PNG/PDF
- Consider adding print-friendly CSS
