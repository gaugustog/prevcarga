# PC-068-09: Feature Commands

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.5
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Implement the feature-related CLI commands: `gen-features` for generating feature datasets and `eval-features` for evaluating feature importance using various methods (SHAP, correlation, stability).

---

## Acceptance Criteria

- [ ] Feature commands module created in `R/cli/commands/features.R`
- [ ] `gen-features` command for feature generation
- [ ] `eval-features` command for importance evaluation
- [ ] Support for multiple evaluation methods
- [ ] Output to storage or file
- [ ] Progress reporting for long operations

---

## Technical Specification

### File Location
```
R/cli/commands/features.R
```

### Gen-Features Command

```r
#' Gen-Features Command Handler
#'
#' Handles the 'prevcarga gen-features' command for generating features.
#'
#' @param args Command line arguments
#' @noRd
cmd_gen_features <- function(args) {
  parser <- create_gen_features_parser()
  opts <- optparse::parse_args(parser, args, positional_arguments = FALSE)

  validate_gen_features_options(opts)
  execute_gen_features(opts)
}


#' Create Gen-Features Parser
#' @noRd
create_gen_features_parser <- function() {
  parser <- optparse::OptionParser(
    usage = "prevcarga gen-features [options]",
    description = "Generate features for model training or prediction."
  )

  # Configuration
  parser <- optparse::add_option(
    parser, c("-c", "--config"),
    type = "character",
    default = "config/config.yaml",
    help = "Path to configuration YAML file [default: %default]"
  )

  # Model (determines feature set)
  parser <- optparse::add_option(
    parser, c("-m", "--model"),
    type = "character",
    default = NULL,
    help = "Model name to generate features for [required]"
  )

  # Areas
  parser <- optparse::add_option(
    parser, c("-a", "--areas"),
    type = "character",
    default = "all",
    help = "Comma-separated area codes or 'all' [default: %default]"
  )

  # Date range
  parser <- optparse::add_option(
    parser, "--start-date",
    type = "character",
    help = "Feature generation start date (YYYY-MM-DD) [required]"
  )

  parser <- optparse::add_option(
    parser, "--end-date",
    type = "character",
    help = "Feature generation end date (YYYY-MM-DD) [required]"
  )

  # Feature plugins
  parser <- optparse::add_option(
    parser, "--plugins",
    type = "character",
    default = NULL,
    help = "Comma-separated plugin names [default: from config]"
  )

  # Output
  parser <- optparse::add_option(
    parser, c("-o", "--output"),
    type = "character",
    default = NULL,
    help = "Output path for features [default: features/<model>/<date>.parquet]"
  )

  parser <- optparse::add_option(
    parser, "--format",
    type = "character",
    default = "parquet",
    help = "Output format: parquet, csv, rds [default: %default]"
  )

  # Options
  parser <- optparse::add_option(
    parser, "--include-target",
    action = "store_true",
    default = TRUE,
    help = "Include target variable in output [default: TRUE]"
  )

  parser <- optparse::add_option(
    parser, "--drop-na",
    action = "store_true",
    default = FALSE,
    help = "Drop rows with NA values"
  )

  # Verbosity
  parser <- optparse::add_option(
    parser, c("-v", "--verbose"),
    action = "store_true",
    default = FALSE,
    help = "Enable verbose output"
  )

  parser
}


#' Validate Gen-Features Options
#' @noRd
validate_gen_features_options <- function(opts) {
  if (is.null(opts$model)) {
    stop("--model is required")
  }

  if (is.null(opts$`start-date`)) {
    stop("--start-date is required")
  }

  if (is.null(opts$`end-date`)) {
    stop("--end-date is required")
  }

  # Validate dates
  start_date <- tryCatch(
    as.Date(opts$`start-date`),
    error = function(e) stop("Invalid --start-date format. Use YYYY-MM-DD")
  )

  end_date <- tryCatch(
    as.Date(opts$`end-date`),
    error = function(e) stop("Invalid --end-date format. Use YYYY-MM-DD")
  )

  if (start_date >= end_date) {
    stop("--start-date must be before --end-date")
  }

  invisible(TRUE)
}


#' Execute Gen-Features
#' @noRd
execute_gen_features <- function(opts) {
  # Load configuration
  cli_info(sprintf("Loading configuration: %s", opts$config))
  config <- ConfigManager$new()
  config$load(opts$config)

  # Resolve areas
  areas <- resolve_areas(opts$areas, config)
  cli_info(sprintf("Areas: %s", paste(areas, collapse = ", ")))

  # Load data
  cli_info("Loading source data...")
  storage <- create_storage_backend(config$get_storage_config())
  loader <- DataLoader$new(storage)

  data <- loader$load_carga(
    areas = areas,
    start_date = opts$`start-date`,
    end_date = opts$`end-date`
  )
  cli_info(sprintf("Loaded %d rows", nrow(data)))

  # Build feature pipeline
  cli_info("Building feature pipeline...")
  pipeline <- build_pipeline_from_config(opts$model, opts$plugins, config)

  # Generate features
  cli_info("Generating features...")
  features <- pipeline$transform(data)

  if (opts$`drop-na`) {
    n_before <- nrow(features$X)
    features$X <- na.omit(features$X)
    features$y <- features$y[!is.na(features$y)]
    cli_info(sprintf("Dropped %d rows with NA values", n_before - nrow(features$X)))
  }

  # Prepare output
  if (opts$`include-target`) {
    output_data <- cbind(features$X, target = features$y)
  } else {
    output_data <- features$X
  }

  # Save output
  output_path <- opts$output %||%
    sprintf("features/%s/%s_%s.%s",
            opts$model, opts$`start-date`, opts$`end-date`, opts$format)

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  switch(opts$format,
    "parquet" = arrow::write_parquet(output_data, output_path),
    "csv" = data.table::fwrite(output_data, output_path),
    "rds" = saveRDS(output_data, output_path)
  )

  cli_success(sprintf("Features saved to: %s", output_path))
  cli_info(sprintf("Shape: %d rows x %d columns", nrow(output_data), ncol(output_data)))
}


#' Build pipeline from config
#' @noRd
build_pipeline_from_config <- function(model_name, plugins_opt, config) {
  pipeline <- FeaturePipeline$new()

  # Get plugins from option or config
  if (!is.null(plugins_opt)) {
    plugin_names <- strsplit(plugins_opt, ",")[[1]]
  } else {
    feature_config <- config$get("features.plugins")
    plugin_names <- names(Filter(function(p) isTRUE(p$enabled), feature_config))
  }

  for (name in plugin_names) {
    plugin_config <- config$get_feature_config(name)$config %||% list()
    plugin <- get_feature_plugin(name, plugin_config)
    pipeline$add_plugin(plugin)
  }

  pipeline
}
```

### Eval-Features Command

```r
#' Eval-Features Command Handler
#'
#' Handles the 'prevcarga eval-features' command for evaluating feature importance.
#'
#' @param args Command line arguments
#' @noRd
cmd_eval_features <- function(args) {
  parser <- create_eval_features_parser()
  opts <- optparse::parse_args(parser, args, positional_arguments = FALSE)

  validate_eval_features_options(opts)
  execute_eval_features(opts)
}


#' Create Eval-Features Parser
#' @noRd
create_eval_features_parser <- function() {
  parser <- optparse::OptionParser(
    usage = "prevcarga eval-features [options]",
    description = "Evaluate feature importance using various methods."
  )

  # Configuration
  parser <- optparse::add_option(
    parser, c("-c", "--config"),
    type = "character",
    default = "config/config.yaml",
    help = "Path to configuration YAML file [default: %default]"
  )

  # Input features
  parser <- optparse::add_option(
    parser, c("-i", "--input"),
    type = "character",
    default = NULL,
    help = "Path to feature file (parquet/csv/rds) [required]"
  )

  # Or generate on-the-fly
  parser <- optparse::add_option(
    parser, c("-m", "--model"),
    type = "character",
    default = NULL,
    help = "Model name (to generate features)"
  )

  parser <- optparse::add_option(
    parser, c("-a", "--area"),
    type = "character",
    default = NULL,
    help = "Area code (for on-the-fly generation)"
  )

  # Evaluation method
  parser <- optparse::add_option(
    parser, "--method",
    type = "character",
    default = "all",
    help = "Method: shap, correlation, stability, permutation, or 'all' [default: %default]"
  )

  # SHAP options
  parser <- optparse::add_option(
    parser, "--shap-samples",
    type = "integer",
    default = 1000,
    help = "Number of samples for SHAP analysis [default: %default]"
  )

  # Correlation options
  parser <- optparse::add_option(
    parser, "--correlation-method",
    type = "character",
    default = "spearman",
    help = "Correlation method: pearson, spearman, kendall [default: %default]"
  )

  # Output
  parser <- optparse::add_option(
    parser, c("-o", "--output"),
    type = "character",
    default = NULL,
    help = "Output path for results [default: stdout]"
  )

  parser <- optparse::add_option(
    parser, "--format",
    type = "character",
    default = "table",
    help = "Output format: table, csv, json [default: %default]"
  )

  parser <- optparse::add_option(
    parser, "--top",
    type = "integer",
    default = 20,
    help = "Show top N features [default: %default]"
  )

  # Verbosity
  parser <- optparse::add_option(
    parser, c("-v", "--verbose"),
    action = "store_true",
    default = FALSE,
    help = "Enable verbose output"
  )

  parser
}


#' Validate Eval-Features Options
#' @noRd
validate_eval_features_options <- function(opts) {
  # Need either input file or model+area
  if (is.null(opts$input) && (is.null(opts$model) || is.null(opts$area))) {
    stop("Either --input or both --model and --area are required")
  }

  # Validate method
  valid_methods <- c("all", "shap", "correlation", "stability", "permutation")
  if (!opts$method %in% valid_methods) {
    stop(sprintf("Invalid --method. Valid: %s", paste(valid_methods, collapse = ", ")))
  }

  # Validate correlation method
  if (!opts$`correlation-method` %in% c("pearson", "spearman", "kendall")) {
    stop("--correlation-method must be pearson, spearman, or kendall")
  }

  invisible(TRUE)
}


#' Execute Eval-Features
#' @noRd
execute_eval_features <- function(opts) {
  # Load configuration
  config <- ConfigManager$new()
  config$load(opts$config)

  # Load or generate features
  if (!is.null(opts$input)) {
    cli_info(sprintf("Loading features from: %s", opts$input))
    features <- load_features_file(opts$input)
  } else {
    cli_info(sprintf("Generating features for %s/%s", opts$model, opts$area))
    features <- generate_features_for_eval(opts$model, opts$area, config)
  }

  # Determine methods to run
  methods <- if (opts$method == "all") {
    c("correlation", "stability", "permutation")
  } else {
    opts$method
  }

  # Run evaluations
  results <- list()

  for (method in methods) {
    cli_info(sprintf("Evaluating with method: %s", method))

    result <- switch(method,
      "correlation" = eval_correlation(features, opts),
      "stability" = eval_stability(features, opts),
      "permutation" = eval_permutation(features, opts, config),
      "shap" = eval_shap(features, opts, config)
    )

    results[[method]] <- result
  }

  # Output results
  output_eval_results(results, opts)
}


#' Evaluate feature correlation
#' @noRd
eval_correlation <- function(features, opts) {
  X <- features$X
  y <- features$y

  # Calculate correlations with target
  correlations <- sapply(names(X), function(col) {
    if (is.numeric(X[[col]])) {
      cor(X[[col]], y, method = opts$`correlation-method`, use = "complete.obs")
    } else {
      NA_real_
    }
  })

  data.table::data.table(
    feature = names(correlations),
    correlation = correlations,
    abs_correlation = abs(correlations)
  )[order(-abs_correlation)]
}


#' Evaluate feature stability
#' @noRd
eval_stability <- function(features, opts) {
  X <- features$X

  # Calculate coefficient of variation for each feature
  cv <- sapply(names(X), function(col) {
    if (is.numeric(X[[col]])) {
      vals <- X[[col]][!is.na(X[[col]])]
      if (length(vals) > 0 && mean(vals) != 0) {
        sd(vals) / abs(mean(vals))
      } else {
        NA_real_
      }
    } else {
      NA_real_
    }
  })

  # Calculate missing rate
  missing_rate <- sapply(names(X), function(col) {
    sum(is.na(X[[col]])) / nrow(X)
  })

  data.table::data.table(
    feature = names(cv),
    cv = cv,
    missing_rate = missing_rate,
    stability_score = 1 - (cv / max(cv, na.rm = TRUE) + missing_rate) / 2
  )[order(-stability_score)]
}


#' Evaluate feature importance via permutation
#' @noRd
eval_permutation <- function(features, opts, config) {
  # Requires a trained model
  model_config <- config$get_model_config("lgbm")  # Use default model
  model <- get_model("lgbm", model_config$config %||% list())

  # Train simple model
  model$train(features$X, features$y)

  # Baseline prediction
  baseline_pred <- model$predict(features$X)
  baseline_rmse <- sqrt(mean((baseline_pred - features$y)^2))

  # Permutation importance
  importance <- sapply(names(features$X), function(col) {
    X_perm <- data.table::copy(features$X)
    X_perm[[col]] <- sample(X_perm[[col]])

    perm_pred <- model$predict(X_perm)
    perm_rmse <- sqrt(mean((perm_pred - features$y)^2))

    perm_rmse - baseline_rmse
  })

  data.table::data.table(
    feature = names(importance),
    importance = importance,
    normalized = importance / max(importance)
  )[order(-importance)]
}


#' Output evaluation results
#' @noRd
output_eval_results <- function(results, opts) {
  for (method in names(results)) {
    cat(sprintf("\n=== %s ===\n\n", toupper(method)))

    result <- results[[method]]
    top_n <- head(result, opts$top)

    if (opts$format == "table") {
      print(top_n)
    } else if (opts$format == "csv") {
      cat(paste(capture.output(data.table::fwrite(top_n, "")), collapse = "\n"))
    } else if (opts$format == "json") {
      cat(jsonlite::toJSON(top_n, pretty = TRUE))
    }

    cat("\n")
  }

  # Save to file if specified
  if (!is.null(opts$output)) {
    saveRDS(results, opts$output)
    cli_success(sprintf("Results saved to: %s", opts$output))
  }
}
```

### Usage Examples

```bash
# Generate features for a model
prevcarga gen-features \
  --model lgbm \
  --areas RJ,SP \
  --start-date 2023-01-01 \
  --end-date 2024-01-01

# Generate with custom output
prevcarga gen-features \
  --model lgbm \
  --areas all \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --output features/lgbm_2023.parquet

# Evaluate feature importance
prevcarga eval-features \
  --input features/lgbm_2023.parquet \
  --method all

# Evaluate with specific method
prevcarga eval-features \
  --input features/lgbm_2023.parquet \
  --method correlation \
  --correlation-method spearman

# Generate and evaluate on-the-fly
prevcarga eval-features \
  --model lgbm \
  --area RJ \
  --method permutation

# Output top 10 features as JSON
prevcarga eval-features \
  --input features.parquet \
  --method correlation \
  --format json \
  --top 10
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | gen-features with valid args | Features generated |
| TC-002 | Missing --model | Error message |
| TC-003 | Missing date range | Error message |
| TC-004 | eval-features correlation | Rankings generated |
| TC-005 | eval-features stability | Stability scores |
| TC-006 | eval-features permutation | Importance scores |
| TC-007 | --method all | All methods run |
| TC-008 | Custom output path | File saved |
| TC-009 | --format csv | CSV output |
| TC-010 | --top N limit | Shows N features |

---

## Dependencies

- PC-064-09: CLI Entry Point
- PC-011-02: FeaturePipeline
- PC-010-02: FeaturePluginRegistry

---

## Definition of Done

- [ ] gen-features command implemented
- [ ] eval-features command implemented
- [ ] All evaluation methods working
- [ ] Multiple output formats supported
- [ ] Progress reporting for long operations
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- SHAP evaluation requires model training (expensive)
- Permutation importance is model-agnostic
- Correlation and stability are fast, no model needed
- Consider caching generated features for repeated evaluation
