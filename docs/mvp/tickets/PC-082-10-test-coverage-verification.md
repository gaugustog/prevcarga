# PC-082-10: Test Coverage Verification

**Epic:** [EPIC-10: Testing & Validation](../epics/EPIC-10-testing-validation.md)
**Task Reference:** T-10.7
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Run coverage analysis using covr package and verify that test coverage meets quality standards: ≥70% overall package coverage and ≥80% coverage for critical infrastructure components.

---

## Acceptance Criteria

- [ ] Coverage analysis script in `tests/validation/coverage.R`
- [ ] Overall package coverage ≥70%
- [ ] Critical components coverage ≥80%
- [ ] Coverage report generated
- [ ] Coverage gaps identified and documented
- [ ] CI/CD integration configured

---

## Technical Specification

### File Location
```
tests/validation/coverage.R
tests/validation/coverage_report.Rmd
```

### Coverage Verification Framework

```r
#' @title CoverageValidator
#' @description Validates test coverage meets quality standards
#' @export
CoverageValidator <- R6::R6Class(
  "CoverageValidator",
  private = list(
    package_path = NULL,
    coverage = NULL,
    thresholds = list(
      overall = 0.70,           # 70% overall
      critical = 0.80,          # 80% critical components
      unit = 0.85,              # 85% unit test coverage
      integration = 0.60        # 60% integration coverage
    ),
    critical_components = c(
      "storage",                # Storage backends
      "models",                 # Model infrastructure
      "combination",            # Combination infrastructure
      "reconciliation",         # Reconciliation infrastructure
      "features"                # Feature pipeline
    )
  ),

  public = list(
    #' @description Initialize coverage validator
    #' @param package_path Path to package
    initialize = function(package_path = ".") {
      private$package_path <- package_path
    },

    #' @description Run coverage analysis
    #' @param type Coverage type ("unit", "integration", "all")
    #' @return Coverage object
    run_coverage = function(type = "all") {
      cat(sprintf("\nRunning %s coverage analysis...\n", type))

      test_path <- switch(type,
        "unit" = "tests/testthat",
        "integration" = "tests/integration",
        "all" = NULL  # All tests
      )

      if (!is.null(test_path)) {
        private$coverage <- covr::package_coverage(
          path = private$package_path,
          type = "tests",
          code = sprintf('testthat::test_dir("%s")', test_path)
        )
      } else {
        private$coverage <- covr::package_coverage(
          path = private$package_path
        )
      }

      cat("Coverage analysis complete.\n")
      private$coverage
    },

    #' @description Get overall coverage percentage
    #' @return Numeric coverage percentage
    get_overall_coverage = function() {
      if (is.null(private$coverage)) {
        stop("Run coverage analysis first")
      }

      covr::percent_coverage(private$coverage)
    },

    #' @description Get coverage by component
    #' @return data.table with component coverage
    get_component_coverage = function() {
      if (is.null(private$coverage)) {
        stop("Run coverage analysis first")
      }

      # Get coverage summary by file
      tally <- covr::tally_coverage(private$coverage)

      # Group by component (directory)
      tally$component <- sapply(tally$filename, function(f) {
        parts <- strsplit(f, "/")[[1]]
        if (length(parts) > 1) parts[1] else "root"
      })

      # Aggregate by component
      by_component <- aggregate(
        cbind(value, lines) ~ component,
        data = tally,
        FUN = sum
      )

      by_component$coverage <- by_component$value / by_component$lines

      data.table::as.data.table(by_component)
    },

    #' @description Check if coverage meets thresholds
    #' @return Validation result
    validate_coverage = function() {
      cat("\n=== Coverage Validation ===\n\n")

      overall <- self$get_overall_coverage()
      by_component <- self$get_component_coverage()

      results <- list(
        overall = list(
          coverage = overall,
          threshold = private$thresholds$overall,
          passed = overall >= private$thresholds$overall * 100
        ),
        critical = list(),
        all_passed = TRUE
      )

      # Check overall
      status <- if (results$overall$passed) "\u2713" else "\u2717"
      cat(sprintf("%s Overall coverage: %.1f%% (threshold: %.0f%%)\n",
                  status, overall, private$thresholds$overall * 100))

      if (!results$overall$passed) {
        results$all_passed <- FALSE
      }

      # Check critical components
      cat("\nCritical Component Coverage:\n")

      for (component in private$critical_components) {
        comp_data <- by_component[by_component$component == component, ]

        if (nrow(comp_data) > 0) {
          coverage <- comp_data$coverage * 100
          threshold <- private$thresholds$critical * 100
          passed <- coverage >= threshold

          status <- if (passed) "\u2713" else "\u2717"
          cat(sprintf("  %s %s: %.1f%% (threshold: %.0f%%)\n",
                      status, component, coverage, threshold))

          results$critical[[component]] <- list(
            coverage = coverage,
            threshold = threshold,
            passed = passed
          )

          if (!passed) {
            results$all_passed <- FALSE
          }
        } else {
          cat(sprintf("  ? %s: No coverage data\n", component))
          results$critical[[component]] <- list(
            coverage = 0,
            threshold = private$thresholds$critical * 100,
            passed = FALSE
          )
          results$all_passed <- FALSE
        }
      }

      results
    },

    #' @description Identify coverage gaps
    #' @return data.table of uncovered lines
    identify_gaps = function() {
      if (is.null(private$coverage)) {
        stop("Run coverage analysis first")
      }

      # Get zero coverage lines
      tally <- covr::tally_coverage(private$coverage)
      gaps <- tally[tally$value == 0, ]

      if (nrow(gaps) > 0) {
        # Group by file
        gap_summary <- aggregate(
          line ~ filename,
          data = gaps,
          FUN = function(x) length(x)
        )
        names(gap_summary) <- c("file", "uncovered_lines")

        # Sort by uncovered lines
        gap_summary <- gap_summary[order(-gap_summary$uncovered_lines), ]

        data.table::as.data.table(gap_summary)
      } else {
        data.table::data.table()
      }
    },

    #' @description Generate coverage report
    #' @param output_path Output path for report
    #' @return Report path
    generate_report = function(output_path = "reports/coverage_report.html") {
      if (is.null(private$coverage)) {
        stop("Run coverage analysis first")
      }

      cat(sprintf("\nGenerating coverage report: %s\n", output_path))

      # Generate HTML report
      covr::report(
        private$coverage,
        file = output_path,
        browse = FALSE
      )

      # Also generate summary
      summary_path <- gsub("\\.html$", "_summary.txt", output_path)

      sink(summary_path)
      cat("PrevCarga Coverage Summary\n")
      cat("==========================\n\n")

      cat(sprintf("Overall Coverage: %.1f%%\n\n", self$get_overall_coverage()))

      cat("Coverage by Component:\n")
      print(self$get_component_coverage())

      cat("\n\nUncovered Files (Top 20):\n")
      print(head(self$identify_gaps(), 20))
      sink()

      cat(sprintf("Summary saved: %s\n", summary_path))

      output_path
    },

    #' @description Get summary table
    #' @return data.table summary
    summary = function() {
      if (is.null(private$coverage)) {
        return(data.table::data.table())
      }

      by_component <- self$get_component_coverage()

      # Add pass/fail status
      by_component$threshold <- ifelse(
        by_component$component %in% private$critical_components,
        private$thresholds$critical,
        private$thresholds$overall
      )

      by_component$passed <- by_component$coverage >= by_component$threshold

      by_component[, .(
        component,
        lines,
        covered = value,
        coverage_pct = round(coverage * 100, 1),
        threshold_pct = round(threshold * 100, 0),
        passed
      )]
    },

    #' @description Print summary
    print = function() {
      cat("\nCoverage Validation Summary\n")
      cat("===========================\n\n")

      if (is.null(private$coverage)) {
        cat("No coverage data. Run coverage analysis first.\n")
        return(invisible(self))
      }

      # Overall
      overall <- self$get_overall_coverage()
      cat(sprintf("Overall Coverage: %.1f%%\n\n", overall))

      # By component
      summary_dt <- self$summary()
      print(summary_dt)

      # Pass/fail count
      n_passed <- sum(summary_dt$passed)
      n_total <- nrow(summary_dt)
      cat(sprintf("\n%d/%d components meet thresholds\n", n_passed, n_total))

      invisible(self)
    }
  )
)
```

### Coverage Runner Script

```r
#!/usr/bin/env Rscript
# tests/validation/coverage.R
#
# Run test coverage analysis

library(prevcargaons)

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)

type <- "all"
if ("--unit" %in% args) type <- "unit"
if ("--integration" %in% args) type <- "integration"

# Initialize validator
validator <- CoverageValidator$new(".")

# Run coverage
coverage <- validator$run_coverage(type = type)

# Validate against thresholds
results <- validator$validate_coverage()

# Print summary
print(validator)

# Generate report
validator$generate_report("reports/coverage_report.html")

# Identify gaps
cat("\n\nTop Coverage Gaps:\n")
gaps <- validator$identify_gaps()
if (nrow(gaps) > 0) {
  print(head(gaps, 15))
}

# Exit with appropriate code
if (results$all_passed) {
  cat("\n\u2713 All coverage thresholds met!\n")
  quit(status = 0)
} else {
  cat("\n\u2717 Some coverage thresholds not met\n")
  quit(status = 1)
}
```

### CI/CD Integration (GitHub Actions)

```yaml
# .github/workflows/coverage.yml
name: Test Coverage

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  coverage:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Setup R
      uses: r-lib/actions/setup-r@v2

    - name: Install dependencies
      run: |
        install.packages(c("covr", "testthat", "data.table"))
        devtools::install_deps(dependencies = TRUE)
      shell: Rscript {0}

    - name: Run coverage
      run: |
        cov <- covr::package_coverage()
        covr::codecov(coverage = cov)
      shell: Rscript {0}
      env:
        CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}

    - name: Check thresholds
      run: Rscript tests/validation/coverage.R
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Overall coverage ≥70% | Threshold met |
| TC-002 | Storage coverage ≥80% | Threshold met |
| TC-003 | Models coverage ≥80% | Threshold met |
| TC-004 | Combination coverage ≥80% | Threshold met |
| TC-005 | Reconciliation coverage ≥80% | Threshold met |
| TC-006 | Features coverage ≥80% | Threshold met |
| TC-007 | Coverage gaps identified | Files listed |
| TC-008 | HTML report generated | File exists |
| TC-009 | CI/CD integration | Workflow passes |
| TC-010 | Threshold enforcement | Build fails if below |

---

## Dependencies

- All test suites from EPIC-01 through EPIC-09

---

## Definition of Done

- [ ] CoverageValidator class implemented
- [ ] Coverage analysis working
- [ ] Threshold validation working
- [ ] Gap identification working
- [ ] HTML report generation
- [ ] CI/CD integration configured
- [ ] Overall coverage ≥70%
- [ ] Critical components ≥80%
- [ ] roxygen2 documentation complete
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Run coverage locally before CI/CD
- Coverage may vary between environments
- Exclude generated code from coverage
- Focus on meaningful coverage, not just percentage
- Critical components need higher coverage due to stability requirements
