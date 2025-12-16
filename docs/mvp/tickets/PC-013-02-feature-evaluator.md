# PC-013-02: FeatureEvaluator Structure

**Epic:** [EPIC-02: Feature Engineering Infrastructure](../epics/EPIC-02-feature-engineering.md)
**Task Reference:** T-02.5
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Create the `FeatureEvaluator` R6 class structure that defines the interface for feature importance analysis, stability evaluation, and feature selection. This epic defines the structure; actual implementation methods are contributed separately.

---

## Acceptance Criteria

- [ ] Evaluation module created in `R/features/evaluation.R`
- [ ] `FeatureEvaluator` R6 class with evaluation method stubs
- [ ] Interface for importance calculation (correlation, SHAP, permutation)
- [ ] Interface for stability analysis across folds
- [ ] Interface for feature selection strategies
- [ ] Report generation structure
- [ ] Extensible design for future evaluation methods

---

## Technical Specification

### File Location
```
R/features/evaluation.R
```

### Evaluator Class

```r
#' @title FeatureEvaluator
#' @description Framework for feature importance and selection analysis
#'
#' This class defines the structure for evaluating features. Actual
#' implementations of evaluation methods are contributed via the
#' plugin system or implemented in subclasses.
#'
#' @export
FeatureEvaluator <- R6::R6Class(
 "FeatureEvaluator",
 private = list(
   results = NULL,
   config = NULL
 ),
 public = list(
   #' @description Initialize evaluator
   #' @param config Evaluation configuration
   initialize = function(config = list()) {
     checkmate::assert_list(config)
     private$config <- config
     private$results <- list()
   },

   #' @description Evaluate feature importance
   #' @param features data.table with feature columns
   #' @param target Target variable vector
   #' @param method Importance method: "correlation", "mutual_info", "shap", "permutation"
   #' @param ... Method-specific arguments
   #' @return data.table with feature importance scores
   evaluate_importance = function(features, target,
                                   method = c("correlation", "mutual_info",
                                              "shap", "permutation"),
                                   ...) {
     method <- match.arg(method)

     result <- switch(method,
       "correlation" = private$importance_correlation(features, target, ...),
       "mutual_info" = private$importance_mutual_info(features, target, ...),
       "shap" = private$importance_shap(features, target, ...),
       "permutation" = private$importance_permutation(features, target, ...),
       stop(sprintf("Unknown importance method: %s", method))
     )

     private$results$importance <- result
     result
   },

   #' @description Evaluate feature stability across folds
   #' @param features data.table with feature columns
   #' @param folds List of fold indices
   #' @param method Stability method: "variance", "selection_frequency", "rank_correlation"
   #' @return data.table with stability metrics
   evaluate_stability = function(features, folds,
                                  method = c("variance", "selection_frequency",
                                             "rank_correlation")) {
     method <- match.arg(method)

     result <- switch(method,
       "variance" = private$stability_variance(features, folds),
       "selection_frequency" = private$stability_selection(features, folds),
       "rank_correlation" = private$stability_rank(features, folds),
       stop(sprintf("Unknown stability method: %s", method))
     )

     private$results$stability <- result
     result
   },

   #' @description Select features based on evaluation results
   #' @param features data.table with feature columns
   #' @param method Selection method: "top_k", "threshold", "cumulative"
   #' @param ... Method-specific arguments (k, threshold, etc.)
   #' @return Character vector of selected feature names
   select_features = function(features,
                               method = c("top_k", "threshold", "cumulative"),
                               ...) {
     method <- match.arg(method)

     if (is.null(private$results$importance)) {
       stop("Must run evaluate_importance() before select_features()")
     }

     switch(method,
       "top_k" = private$select_top_k(features, ...),
       "threshold" = private$select_threshold(features, ...),
       "cumulative" = private$select_cumulative(features, ...),
       stop(sprintf("Unknown selection method: %s", method))
     )
   },

   #' @description Generate evaluation report
   #' @param format Report format: "list", "data.table", "markdown"
   #' @return Report in specified format
   get_report = function(format = c("list", "data.table", "markdown")) {
     format <- match.arg(format)

     report <- list(
       importance = private$results$importance,
       stability = private$results$stability,
       config = private$config,
       timestamp = Sys.time()
     )

     switch(format,
       "list" = report,
       "data.table" = private$report_as_datatable(report),
       "markdown" = private$report_as_markdown(report),
       report
     )
   },

   #' @description Get cached results
   #' @return List with evaluation results
   get_results = function() {
     private$results
   },

   #' @description Clear cached results
   #' @return Invisible self
   clear_results = function() {
     private$results <- list()
     invisible(self)
   }
 ),

 private = list(
   #' Correlation-based importance
   #' @noRd
   importance_correlation = function(features, target, ...) {
     # Placeholder - to be implemented
     feature_names <- setdiff(names(features), "target")

     data.table::data.table(
       feature = feature_names,
       importance = NA_real_,
       method = "correlation"
     )
   },

   #' Mutual information importance
   #' @noRd
   importance_mutual_info = function(features, target, ...) {
     stop("Mutual information not yet implemented. Install suggested package or contribute implementation.")
   },

   #' SHAP importance
   #' @noRd
   importance_shap = function(features, target, model = NULL, ...) {
     if (is.null(model)) {
       stop("SHAP importance requires a trained model")
     }
     stop("SHAP importance not yet implemented. Install suggested package or contribute implementation.")
   },

   #' Permutation importance
   #' @noRd
   importance_permutation = function(features, target, model = NULL, ...) {
     if (is.null(model)) {
       stop("Permutation importance requires a trained model")
     }
     stop("Permutation importance not yet implemented. Contribute implementation.")
   },

   #' Variance-based stability
   #' @noRd
   stability_variance = function(features, folds) {
     stop("Variance stability not yet implemented. Contribute implementation.")
   },

   #' Selection frequency stability
   #' @noRd
   stability_selection = function(features, folds) {
     stop("Selection frequency stability not yet implemented. Contribute implementation.")
   },

   #' Rank correlation stability
   #' @noRd
   stability_rank = function(features, folds) {
     stop("Rank correlation stability not yet implemented. Contribute implementation.")
   },

   #' Select top k features
   #' @noRd
   select_top_k = function(features, k = 10, ...) {
     importance <- private$results$importance
     if (is.null(importance)) {
       stop("No importance results available")
     }

     data.table::setorder(importance, -importance)
     head(importance$feature, k)
   },

   #' Select by threshold
   #' @noRd
   select_threshold = function(features, threshold = 0.05, ...) {
     importance <- private$results$importance
     if (is.null(importance)) {
       stop("No importance results available")
     }

     importance[importance >= threshold, feature]
   },

   #' Select by cumulative importance
   #' @noRd
   select_cumulative = function(features, cumulative_pct = 0.95, ...) {
     importance <- private$results$importance

     if (is.null(importance)) {
       stop("No importance results available")
     }

     data.table::setorder(importance, -importance)
     importance[, cumsum := cumsum(importance) / sum(importance, na.rm = TRUE)]
     importance[cumsum <= cumulative_pct, feature]
   },

   #' Convert report to data.table
   #' @noRd
   report_as_datatable = function(report) {
     if (!is.null(report$importance)) {
       return(report$importance)
     }
     data.table::data.table()
   },

   #' Convert report to markdown
   #' @noRd
   report_as_markdown = function(report) {
     lines <- character()
     lines <- c(lines, "# Feature Evaluation Report")
     lines <- c(lines, "")
     lines <- c(lines, sprintf("Generated: %s", report$timestamp))
     lines <- c(lines, "")

     if (!is.null(report$importance)) {
       lines <- c(lines, "## Feature Importance")
       lines <- c(lines, "")
       lines <- c(lines, "| Feature | Importance | Method |")
       lines <- c(lines, "|---------|------------|--------|")

       for (i in seq_len(min(20, nrow(report$importance)))) {
         row <- report$importance[i]
         lines <- c(lines, sprintf("| %s | %.4f | %s |",
                                    row$feature, row$importance, row$method))
       }
     }

     paste(lines, collapse = "\n")
   }
 )
)
```

### Correlation Implementation

```r
#' Basic correlation-based feature importance
#'
#' @param features data.table with features
#' @param target Target vector
#' @return data.table with correlations
#' @export
calculate_correlation_importance <- function(features, target) {
 checkmate::assert_data_table(features)
 checkmate::assert_numeric(target, len = nrow(features))

 # Get numeric columns only
 numeric_cols <- names(features)[sapply(features, is.numeric)]

 correlations <- sapply(numeric_cols, function(col) {
   tryCatch(
     abs(cor(features[[col]], target, use = "pairwise.complete.obs")),
     error = function(e) NA_real_
   )
 })

 result <- data.table::data.table(
   feature = numeric_cols,
   importance = correlations,
   method = "correlation"
 )

 data.table::setorder(result, -importance)
 result
}
```

### Usage Example

```r
# Create evaluator
evaluator <- FeatureEvaluator$new()

# Evaluate importance using correlation
importance <- evaluator$evaluate_importance(
 features = train_features,
 target = train_data$CargaGlobal,
 method = "correlation"
)

# Select top 20 features
selected <- evaluator$select_features(
 features = train_features,
 method = "top_k",
 k = 20
)

# Generate report
report <- evaluator$get_report(format = "markdown")
cat(report)

# Or use standalone function
importance <- calculate_correlation_importance(
 features = train_features,
 target = train_data$CargaGlobal
)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize evaluator | Empty results |
| TC-002 | evaluate_importance correlation | Returns data.table |
| TC-003 | evaluate_importance unimplemented | Clear error message |
| TC-004 | evaluate_stability unimplemented | Clear error message |
| TC-005 | select_features without importance | Error |
| TC-006 | select_features top_k | Returns k features |
| TC-007 | select_features threshold | Returns filtered |
| TC-008 | get_report list format | Returns list |
| TC-009 | get_report data.table format | Returns data.table |
| TC-010 | get_report markdown | Returns string |
| TC-011 | clear_results | Empties cache |
| TC-012 | calculate_correlation_importance | Correct correlations |

---

## Definition of Done

- [ ] FeatureEvaluator class structure implemented
- [ ] Interface methods defined with clear errors
- [ ] Correlation importance implemented
- [ ] Feature selection methods working
- [ ] Report generation working
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Most evaluation methods are placeholders; implementations are contributed separately
- The correlation method is implemented as a baseline
- SHAP and permutation methods require model access
- Consider adding caching for expensive computations
- Future work: integrate with model training pipeline for automatic importance
