# PC-085-10: Validation Report

**Epic:** [EPIC-10: Testing & Validation](../epics/EPIC-10-testing-validation.md)
**Task Reference:** T-10.10
**Priority:** High
**Estimated Effort:** 2 days

---

## Summary

Generate a comprehensive validation report that consolidates all validation results including model accuracy comparisons, performance benchmarks, test coverage summary, and recommendations for production deployment.

---

## Acceptance Criteria

- [ ] Validation report template in `reports/validation_report.Rmd`
- [ ] Model accuracy comparison included
- [ ] Performance benchmarks included
- [ ] Test coverage summary included
- [ ] Recommendations section included
- [ ] HTML report generation working
- [ ] Executive summary page

---

## Technical Specification

### File Location
```
reports/validation_report.Rmd
R/reporting/validation_report.R
```

### Validation Report Generator

```r
#' @title ValidationReportGenerator
#' @description Generates comprehensive validation report
#' @export
ValidationReportGenerator <- R6::R6Class(
  "ValidationReportGenerator",
  private = list(
    config = NULL,
    storage = NULL,
    results = list(),
    report_date = NULL
  ),

  public = list(
    #' @description Initialize report generator
    #' @param config ConfigManager instance
    initialize = function(config) {
      private$config <- config
      private$storage <- create_storage_backend(config$get_storage_config())
      private$report_date <- Sys.Date()
    },

    #' @description Collect all validation results
    #' @param results_dir Directory containing validation results
    #' @return Self for chaining
    collect_results = function(results_dir = "reports/validation") {
      cat("Collecting validation results...\n")

      # Load baseline validation
      baseline_path <- file.path(results_dir, "baseline_results.rds")
      if (file.exists(baseline_path)) {
        private$results$baseline <- readRDS(baseline_path)
        cat("  - Baseline validation loaded\n")
      }

      # Load reproduction results
      reproduction_path <- file.path(results_dir, "reproduction_results.rds")
      if (file.exists(reproduction_path)) {
        private$results$reproduction <- readRDS(reproduction_path)
        cat("  - Model reproduction results loaded\n")
      }

      # Load backtest results
      backtest_path <- file.path(results_dir, "backtest_results.rds")
      if (file.exists(backtest_path)) {
        private$results$backtest <- readRDS(backtest_path)
        cat("  - Backtest results loaded\n")
      }

      # Load combination validation
      combination_path <- file.path(results_dir, "combination_results.rds")
      if (file.exists(combination_path)) {
        private$results$combination <- readRDS(combination_path)
        cat("  - Combination validation loaded\n")
      }

      # Load reconciliation validation
      reconciliation_path <- file.path(results_dir, "reconciliation_results.rds")
      if (file.exists(reconciliation_path)) {
        private$results$reconciliation <- readRDS(reconciliation_path)
        cat("  - Reconciliation validation loaded\n")
      }

      # Load performance benchmarks
      benchmark_path <- file.path(results_dir, "benchmark_results.rds")
      if (file.exists(benchmark_path)) {
        private$results$benchmark <- readRDS(benchmark_path)
        cat("  - Performance benchmarks loaded\n")
      }

      # Load coverage results
      coverage_path <- file.path(results_dir, "coverage_results.rds")
      if (file.exists(coverage_path)) {
        private$results$coverage <- readRDS(coverage_path)
        cat("  - Coverage results loaded\n")
      }

      invisible(self)
    },

    #' @description Generate model accuracy summary
    #' @return data.table with model metrics
    summarize_model_accuracy = function() {
      if (is.null(private$results$backtest)) {
        return(data.table::data.table())
      }

      backtest <- private$results$backtest

      # By model
      by_model <- backtest$breakdowns$by_model

      # Add ranking
      by_model <- by_model[order(mape)]
      by_model$rank <- seq_len(nrow(by_model))

      by_model[, .(
        model,
        rank,
        mape = round(mape * 100, 2),
        mae = round(mae, 1),
        rmse = round(rmse, 1),
        n_predictions = n
      )]
    },

    #' @description Generate performance summary
    #' @return data.table with performance metrics
    summarize_performance = function() {
      if (is.null(private$results$benchmark)) {
        return(data.table::data.table())
      }

      benchmark <- private$results$benchmark

      data.table::data.table(
        metric = c("Training (single area)", "Intraday prediction", "1-year backtest"),
        target = c("< 30 min", "< 5 min", "< 8 hours"),
        actual = c(
          sprintf("%.1f min", benchmark$training$mean_seconds / 60),
          sprintf("%.1f min", benchmark$intraday_prediction$mean_seconds / 60),
          sprintf("%.1f hours", benchmark$backtest$elapsed_hours %||% NA)
        ),
        passed = c(
          benchmark$training$passed,
          benchmark$intraday_prediction$passed,
          benchmark$backtest$passed %||% NA
        )
      )
    },

    #' @description Generate recommendations
    #' @return List of recommendations
    generate_recommendations = function() {
      recommendations <- list()

      # Model recommendations
      if (!is.null(private$results$backtest)) {
        model_summary <- self$summarize_model_accuracy()

        if (nrow(model_summary) > 0) {
          best_model <- model_summary[rank == 1, model]
          best_mape <- model_summary[rank == 1, mape]

          recommendations$model <- sprintf(
            "Use %s as primary model (MAPE: %.2f%%)",
            best_model, best_mape
          )
        }
      }

      # Retraining interval recommendation
      if (!is.null(private$results$backtest)) {
        by_interval <- private$results$backtest$get_metrics_by_interval()

        if (nrow(by_interval) > 0) {
          best_interval <- by_interval[which.min(mean_mape), interval]
          recommendations$retraining <- sprintf(
            "Recommended retraining interval: %d days",
            best_interval
          )
        }
      }

      # Coverage recommendations
      if (!is.null(private$results$coverage)) {
        overall <- private$results$coverage$overall$coverage

        if (overall < 70) {
          recommendations$coverage <- sprintf(
            "CRITICAL: Test coverage (%.1f%%) below 70%% threshold - add more tests",
            overall
          )
        } else if (overall < 80) {
          recommendations$coverage <- sprintf(
            "Consider improving test coverage from %.1f%% to 80%%+",
            overall
          )
        }
      }

      # Performance recommendations
      if (!is.null(private$results$benchmark)) {
        if (!private$results$benchmark$training$passed) {
          recommendations$performance_training <-
            "Training exceeds target - consider model optimization"
        }
        if (!private$results$benchmark$intraday_prediction$passed) {
          recommendations$performance_prediction <-
            "Prediction exceeds target - consider caching or parallel processing"
        }
      }

      # Combination recommendations
      if (!is.null(private$results$combination)) {
        outperforming <- sum(sapply(private$results$combination, function(r) {
          isTRUE(r$outperforms_individuals)
        }))

        if (outperforming > 0) {
          recommendations$combination <- sprintf(
            "%d combination strategies outperform individual models - use ensemble",
            outperforming
          )
        }
      }

      recommendations
    },

    #' @description Generate overall validation status
    #' @return Status summary
    get_validation_status = function() {
      checks <- list()

      # Baseline validation
      if (!is.null(private$results$reproduction)) {
        checks$baseline <- all(sapply(private$results$reproduction, function(r) {
          isTRUE(r$passed)
        }))
      } else {
        checks$baseline <- NA
      }

      # Performance benchmarks
      if (!is.null(private$results$benchmark)) {
        checks$performance <- all(c(
          private$results$benchmark$training$passed,
          private$results$benchmark$intraday_prediction$passed,
          private$results$benchmark$backtest$passed %||% TRUE
        ))
      } else {
        checks$performance <- NA
      }

      # Test coverage
      if (!is.null(private$results$coverage)) {
        checks$coverage <- private$results$coverage$overall$coverage >= 70
      } else {
        checks$coverage <- NA
      }

      # Reconciliation
      if (!is.null(private$results$reconciliation)) {
        checks$reconciliation <- all(sapply(private$results$reconciliation, function(r) {
          isTRUE(r$hierarchical_consistent)
        }))
      } else {
        checks$reconciliation <- NA
      }

      # Overall status
      non_na_checks <- checks[!is.na(checks)]
      overall_passed <- all(unlist(non_na_checks))

      list(
        checks = checks,
        overall_passed = overall_passed,
        n_passed = sum(unlist(non_na_checks), na.rm = TRUE),
        n_total = length(non_na_checks)
      )
    },

    #' @description Generate HTML report
    #' @param output_path Output path for report
    #' @return Report path
    generate = function(output_path = "reports/validation_report.html") {
      cat("\nGenerating validation report...\n")

      # Collect results if not already done
      if (length(private$results) == 0) {
        self$collect_results()
      }

      # Prepare parameters
      params <- list(
        results = private$results,
        model_summary = self$summarize_model_accuracy(),
        performance_summary = self$summarize_performance(),
        recommendations = self$generate_recommendations(),
        status = self$get_validation_status(),
        config = private$config$as_list(),
        report_date = private$report_date
      )

      # Render report
      template_path <- system.file(
        "templates/validation_report.Rmd",
        package = "prevcargaons"
      )

      # Fallback to local template
      if (!file.exists(template_path)) {
        template_path <- "reports/validation_report.Rmd"
      }

      rmarkdown::render(
        input = template_path,
        output_file = output_path,
        params = params,
        envir = new.env()
      )

      cat(sprintf("Report generated: %s\n", output_path))
      output_path
    },

    #' @description Get results
    get_results = function() {
      private$results
    },

    #' @description Print summary
    print = function() {
      status <- self$get_validation_status()

      cat("\n")
      cat("=" |> rep(50) |> paste(collapse = ""))
      cat("\n  Validation Summary\n")
      cat("=" |> rep(50) |> paste(collapse = ""))
      cat("\n\n")

      overall <- if (status$overall_passed) "\u2713 PASSED" else "\u2717 FAILED"
      cat(sprintf("Overall Status: %s (%d/%d checks passed)\n\n",
                  overall, status$n_passed, status$n_total))

      cat("Validation Checks:\n")
      for (name in names(status$checks)) {
        check <- status$checks[[name]]
        symbol <- if (is.na(check)) "?" else if (check) "\u2713" else "\u2717"
        status_text <- if (is.na(check)) "Not run" else if (check) "Passed" else "Failed"
        cat(sprintf("  %s %s: %s\n", symbol, name, status_text))
      }

      cat("\nRecommendations:\n")
      recs <- self$generate_recommendations()
      for (rec in recs) {
        cat(sprintf("  - %s\n", rec))
      }

      invisible(self)
    }
  )
)
```

### R Markdown Report Template

```r
# reports/validation_report.Rmd

---
title: "PrevCarga Validation Report"
date: "`r params$report_date`"
output:
  html_document:
    toc: true
    toc_depth: 3
    toc_float: true
    theme: flatly
    highlight: tango
    css: validation_report.css
params:
  results: NULL
  model_summary: NULL
  performance_summary: NULL
  recommendations: NULL
  status: NULL
  config: NULL
  report_date: NULL
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)
library(data.table)
library(ggplot2)
library(knitr)
library(kableExtra)
```

# Executive Summary

```{r status-badge}
status <- params$status
badge_color <- if (status$overall_passed) "#28a745" else "#dc3545"
badge_text <- if (status$overall_passed) "PASSED" else "FAILED"

cat(sprintf('<div style="text-align: center; padding: 20px;">
  <span style="background-color: %s; color: white; padding: 10px 30px; font-size: 24px; border-radius: 5px;">
    %s
  </span>
  <p style="margin-top: 10px; font-size: 18px;">%d/%d validation checks passed</p>
</div>', badge_color, badge_text, status$n_passed, status$n_total))
```

## Key Findings

- **Best performing model:** `r if(nrow(params$model_summary) > 0) params$model_summary[rank == 1, model] else "N/A"`
- **Best model MAPE:** `r if(nrow(params$model_summary) > 0) sprintf("%.2f%%", params$model_summary[rank == 1, mape]) else "N/A"`
- **Training target met:** `r if(!is.null(params$performance_summary) && nrow(params$performance_summary) > 0) ifelse(params$performance_summary[metric == "Training (single area)", passed], "Yes", "No") else "N/A"`

## Recommendations

```{r recommendations}
if (length(params$recommendations) > 0) {
  for (i in seq_along(params$recommendations)) {
    cat(sprintf("- %s\n", params$recommendations[[i]]))
  }
} else {
  cat("No specific recommendations at this time.")
}
```

---

# Model Accuracy

## Model Comparison

```{r model-table}
if (nrow(params$model_summary) > 0) {
  params$model_summary %>%
    kable(caption = "Model Performance Summary") %>%
    kable_styling(bootstrap_options = c("striped", "hover"))
} else {
  cat("No model summary available.")
}
```

## Accuracy by Area

```{r accuracy-by-area, fig.width=10, fig.height=6}
if (!is.null(params$results$backtest) &&
    !is.null(params$results$backtest$breakdowns$by_area)) {

  by_area <- params$results$backtest$breakdowns$by_area

  ggplot(by_area, aes(x = reorder(area_code, -mape), y = mape * 100)) +
    geom_bar(stat = "identity", fill = "#3498db") +
    geom_hline(yintercept = 5, linetype = "dashed", color = "red") +
    coord_flip() +
    labs(
      title = "MAPE by Area",
      x = "Area",
      y = "MAPE (%)"
    ) +
    theme_minimal()
} else {
  cat("No area-level data available.")
}
```

## Accuracy by Horizon

```{r accuracy-by-horizon, fig.width=10, fig.height=6}
if (!is.null(params$results$backtest) &&
    !is.null(params$results$backtest$breakdowns$by_horizon)) {

  by_horizon <- params$results$backtest$breakdowns$by_horizon

  ggplot(by_horizon, aes(x = factor(horizon), y = mape * 100)) +
    geom_bar(stat = "identity", fill = "#2ecc71") +
    geom_line(aes(group = 1), color = "#27ae60", size = 1) +
    labs(
      title = "MAPE by Forecast Horizon",
      x = "Horizon (D+N)",
      y = "MAPE (%)"
    ) +
    theme_minimal()
} else {
  cat("No horizon-level data available.")
}
```

---

# Performance Benchmarks

```{r performance-table}
if (!is.null(params$performance_summary) && nrow(params$performance_summary) > 0) {
  params$performance_summary %>%
    mutate(
      status = ifelse(is.na(passed), "Not run",
                      ifelse(passed, "\u2713 Passed", "\u2717 Failed"))
    ) %>%
    select(-passed) %>%
    kable(caption = "Performance Benchmark Results") %>%
    kable_styling(bootstrap_options = c("striped", "hover"))
} else {
  cat("No performance benchmarks available.")
}
```

---

# Test Coverage

```{r coverage-summary}
if (!is.null(params$results$coverage)) {
  coverage <- params$results$coverage

  cat(sprintf("**Overall Coverage:** %.1f%%\n\n", coverage$overall$coverage))
  cat(sprintf("**Target:** %.0f%%\n\n", coverage$overall$threshold * 100))

  if (coverage$overall$passed) {
    cat('<span style="color: green;">Coverage target met!</span>')
  } else {
    cat('<span style="color: red;">Coverage below target.</span>')
  }
} else {
  cat("No coverage data available.")
}
```

---

# Combination Validation

```{r combination-table}
if (!is.null(params$results$combination)) {
  combination_df <- data.table::rbindlist(lapply(names(params$results$combination), function(name) {
    r <- params$results$combination[[name]]
    if (!is.null(r$combined_mape)) {
      data.table::data.table(
        combiner = name,
        mape = round(r$combined_mape * 100, 2),
        outperforms = r$outperforms_individuals,
        improvement = round(r$improvement * 100, 2)
      )
    }
  }), fill = TRUE)

  if (nrow(combination_df) > 0) {
    combination_df %>%
      kable(caption = "Combination Strategy Results") %>%
      kable_styling(bootstrap_options = c("striped", "hover"))
  }
} else {
  cat("No combination validation data available.")
}
```

---

# Reconciliation Validation

```{r reconciliation-table}
if (!is.null(params$results$reconciliation)) {
  rec_df <- data.table::rbindlist(lapply(names(params$results$reconciliation), function(name) {
    r <- params$results$reconciliation[[name]]
    if (!is.null(r$hierarchical_consistent)) {
      data.table::data.table(
        reconciler = name,
        consistent = r$hierarchical_consistent,
        max_discrepancy = round(r$max_sin_discrepancy * 100, 4)
      )
    }
  }), fill = TRUE)

  if (nrow(rec_df) > 0) {
    rec_df %>%
      kable(caption = "Reconciliation Validation Results") %>%
      kable_styling(bootstrap_options = c("striped", "hover"))
  }
} else {
  cat("No reconciliation validation data available.")
}
```

---

# Validation Checklist

| Check | Status | Notes |
|-------|--------|-------|
| Baseline validation | `r if(is.na(params$status$checks$baseline)) "Not run" else if(params$status$checks$baseline) "\u2713 Passed" else "\u2717 Failed"` | Models within ±5% MAPE |
| Performance benchmarks | `r if(is.na(params$status$checks$performance)) "Not run" else if(params$status$checks$performance) "\u2713 Passed" else "\u2717 Failed"` | All targets met |
| Test coverage | `r if(is.na(params$status$checks$coverage)) "Not run" else if(params$status$checks$coverage) "\u2713 Passed" else "\u2717 Failed"` | ≥70% coverage |
| Reconciliation | `r if(is.na(params$status$checks$reconciliation)) "Not run" else if(params$status$checks$reconciliation) "\u2713 Passed" else "\u2717 Failed"` | 100% hierarchical consistency |

---

# Appendix

## Configuration

```{r config}
if (!is.null(params$config)) {
  cat("```yaml\n")
  cat(yaml::as.yaml(params$config))
  cat("```\n")
}
```

## Report Metadata

- **Generated:** `r Sys.time()`
- **Package version:** `r packageVersion("prevcargaons")`
- **R version:** `r R.version.string`
```

### Report Generation Script

```r
#!/usr/bin/env Rscript
# reports/generate_validation_report.R
#
# Generate comprehensive validation report

library(prevcargaons)

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)

output_path <- "reports/validation_report.html"
results_dir <- "reports/validation"

if ("--output" %in% args) {
  idx <- which(args == "--output")
  output_path <- args[idx + 1]
}

if ("--results-dir" %in% args) {
  idx <- which(args == "--results-dir")
  results_dir <- args[idx + 1]
}

# Load config
config <- load_config("config/config.yaml")

# Generate report
generator <- ValidationReportGenerator$new(config)
generator$collect_results(results_dir)

# Print summary
print(generator)

# Generate HTML report
report_path <- generator$generate(output_path)

cat(sprintf("\nValidation report generated: %s\n", report_path))

# Open in browser if requested
if ("--open" %in% args) {
  browseURL(report_path)
}

# Exit based on validation status
status <- generator$get_validation_status()
if (status$overall_passed) {
  cat("\n\u2713 All validation checks passed\n")
  quit(status = 0)
} else {
  cat("\n\u2717 Some validation checks failed\n")
  quit(status = 1)
}
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Collect all validation results | All results loaded |
| TC-002 | Model accuracy summary | Table generated |
| TC-003 | Performance summary | Benchmarks shown |
| TC-004 | Recommendations generated | Based on results |
| TC-005 | Overall status calculated | Pass/fail determined |
| TC-006 | HTML report generated | File exists |
| TC-007 | Report contains all sections | TOC complete |
| TC-008 | Charts render correctly | No errors |
| TC-009 | Missing data handled | Graceful fallback |
| TC-010 | Report opens in browser | Renders correctly |

---

## Dependencies

- PC-076-10: Baseline Validation Framework
- PC-077-10: Model Results Reproduction
- PC-078-10: 1-Year Backtest Execution
- PC-079-10: Combination Validation
- PC-080-10: Reconciliation Validation
- PC-081-10: Performance Benchmarking
- PC-082-10: Test Coverage Verification

---

## Definition of Done

- [ ] ValidationReportGenerator class implemented
- [ ] Results collection working
- [ ] Model accuracy summary generated
- [ ] Performance summary generated
- [ ] Recommendations logic implemented
- [ ] R Markdown template created
- [ ] HTML report generation working
- [ ] All sections rendering
- [ ] Executive summary included
- [ ] roxygen2 documentation complete
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Report aggregates all validation results
- Executive summary critical for stakeholders
- Recommendations based on validation findings
- Consider adding PDF export option
- Report should be generated at end of validation cycle
