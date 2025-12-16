# PC-069-09: Evaluation Commands

**Epic:** [EPIC-09: CLI](../epics/EPIC-09-cli.md)
**Task Reference:** T-09.6
**Priority:** Medium
**Estimated Effort:** 1.5 days

---

## Summary

Implement the evaluation-related CLI commands: `eval-model` for single model evaluation, `eval-combination` for forecast combination evaluation, and `combine` for combining forecasts from multiple models.

---

## Acceptance Criteria

- [ ] Evaluation commands module created in `R/cli/commands/evaluate.R`
- [ ] `eval-model` command for model performance evaluation
- [ ] `eval-combination` command for combination strategy evaluation
- [ ] `combine` command for forecast combination
- [ ] Multiple metrics support
- [ ] Report generation options

---

## Technical Specification

### File Location
```
R/cli/commands/evaluate.R
```

### Eval-Model Command

```r
#' Eval-Model Command Handler
#'
#' Handles the 'prevcarga eval-model' command for model evaluation.
#'
#' @param args Command line arguments
#' @noRd
cmd_eval_model <- function(args) {
  parser <- create_eval_model_parser()
  opts <- optparse::parse_args(parser, args, positional_arguments = FALSE)

  validate_eval_model_options(opts)
  execute_eval_model(opts)
}


#' Create Eval-Model Parser
#' @noRd
create_eval_model_parser <- function() {
  parser <- optparse::OptionParser(
    usage = "prevcarga eval-model [options]",
    description = "Evaluate model performance against actual values."
  )

  # Configuration
  parser <- optparse::add_option(
    parser, c("-c", "--config"),
    type = "character",
    default = "config/config.yaml",
    help = "Path to configuration YAML file [default: %default]"
  )

  # Model specification
  parser <- optparse::add_option(
    parser, c("-m", "--model"),
    type = "character",
    help = "Model name to evaluate [required]"
  )

  parser <- optparse::add_option(
    parser, "--model-version",
    type = "character",
    default = "latest",
    help = "Model version [default: %default]"
  )

  parser <- optparse::add_option(
    parser, "--model-path",
    type = "character",
    default = NULL,
    help = "Direct path to model artifact [alternative to --model]"
  )

  # Test data
  parser <- optparse::add_option(
    parser, "--test-data",
    type = "character",
    default = NULL,
    help = "Path to test data (parquet/csv)"
  )

  parser <- optparse::add_option(
    parser, "--start-date",
    type = "character",
    default = NULL,
    help = "Test period start date"
  )

  parser <- optparse::add_option(
    parser, "--end-date",
    type = "character",
    default = NULL,
    help = "Test period end date"
  )

  # Areas
  parser <- optparse::add_option(
    parser, c("-a", "--areas"),
    type = "character",
    default = "all",
    help = "Comma-separated area codes [default: %default]"
  )

  # Metrics
  parser <- optparse::add_option(
    parser, "--metrics",
    type = "character",
    default = "mape,mae,rmse,mbe",
    help = "Comma-separated metrics [default: %default]"
  )

  # Breakdown options
  parser <- optparse::add_option(
    parser, "--by-horizon",
    action = "store_true",
    default = FALSE,
    help = "Break down metrics by forecast horizon"
  )

  parser <- optparse::add_option(
    parser, "--by-period",
    action = "store_true",
    default = FALSE,
    help = "Break down metrics by patamar (peak/off-peak)"
  )

  parser <- optparse::add_option(
    parser, "--by-area",
    action = "store_true",
    default = FALSE,
    help = "Break down metrics by area"
  )

  # Output
  parser <- optparse::add_option(
    parser, c("-o", "--output"),
    type = "character",
    default = NULL,
    help = "Output path for results"
  )

  parser <- optparse::add_option(
    parser, "--format",
    type = "character",
    default = "table",
    help = "Output format: table, csv, json [default: %default]"
  )

  parser <- optparse::add_option(
    parser, "--report",
    type = "character",
    default = NULL,
    help = "Generate HTML report at path"
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


#' Execute Eval-Model
#' @noRd
execute_eval_model <- function(opts) {
  # Load configuration
  config <- ConfigManager$new()
  config$load(opts$config)

  # Load model
  model <- load_model_from_opts(opts, config)
  cli_info(sprintf("Loaded model: %s", opts$model %||% opts$`model-path`))

  # Load test data
  test_data <- load_test_data(opts, config)
  cli_info(sprintf("Test data: %d rows", nrow(test_data)))

  # Generate predictions
  cli_info("Generating predictions...")
  predictions <- generate_model_predictions(model, test_data, config)

  # Calculate metrics
  cli_info("Calculating metrics...")
  metrics <- strsplit(opts$metrics, ",")[[1]]

  calculator <- MetricsCalculator$new()

  results <- list()

  # Overall metrics
  results$overall <- calculator$calculate(
    actual = test_data$carga_mwmed,
    predicted = predictions,
    metrics = metrics
  )

  # By-horizon breakdown
  if (opts$`by-horizon` && "horizon" %in% names(test_data)) {
    results$by_horizon <- calculate_metrics_by_horizon(
      test_data, predictions, metrics, calculator
    )
  }

  # By-period breakdown
  if (opts$`by-period`) {
    results$by_period <- calculate_metrics_by_period(
      test_data, predictions, metrics, calculator
    )
  }

  # By-area breakdown
  if (opts$`by-area` && "area_code" %in% names(test_data)) {
    results$by_area <- calculate_metrics_by_area(
      test_data, predictions, metrics, calculator
    )
  }

  # Output results
  output_eval_model_results(results, opts)

  # Generate report if requested
  if (!is.null(opts$report)) {
    generate_eval_report(results, opts$report, opts$model)
    cli_success(sprintf("Report generated: %s", opts$report))
  }

  invisible(results)
}


#' Load model from options
#' @noRd
load_model_from_opts <- function(opts, config) {
  if (!is.null(opts$`model-path`)) {
    # Direct path
    artifact <- ModelArtifact$new()
    artifact$load(LocalStorageBackend$new(), opts$`model-path`)
    return(artifact$model)
  }

  if (!is.null(opts$model)) {
    # Load from registry
    storage <- create_storage_backend(config$get_storage_config())
    version <- opts$`model-version`

    if (version == "latest") {
      version <- get_latest_model_version(opts$model, storage)
    }

    artifact <- ModelArtifact$new()
    path <- sprintf("models/%s/%s/model.rds", opts$model, version)
    artifact$load(storage, path)
    return(artifact$model)
  }

  stop("Either --model or --model-path is required")
}
```

### Eval-Combination Command

```r
#' Eval-Combination Command Handler
#'
#' Evaluates different forecast combination strategies.
#'
#' @param args Command line arguments
#' @noRd
cmd_eval_combination <- function(args) {
  parser <- create_eval_combination_parser()
  opts <- optparse::parse_args(parser, args, positional_arguments = FALSE)

  validate_eval_combination_options(opts)
  execute_eval_combination(opts)
}


#' Create Eval-Combination Parser
#' @noRd
create_eval_combination_parser <- function() {
  parser <- optparse::OptionParser(
    usage = "prevcarga eval-combination [options]",
    description = "Evaluate forecast combination strategies."
  )

  # Configuration
  parser <- optparse::add_option(
    parser, c("-c", "--config"),
    type = "character",
    default = "config/config.yaml",
    help = "Path to configuration YAML file"
  )

  # Models to combine
  parser <- optparse::add_option(
    parser, c("-m", "--models"),
    type = "character",
    help = "Comma-separated model names to combine [required]"
  )

  # Combination strategies
  parser <- optparse::add_option(
    parser, "--strategies",
    type = "character",
    default = "simple_average,inverse_mape,optimal",
    help = "Comma-separated strategies to evaluate [default: %default]"
  )

  # Test period
  parser <- optparse::add_option(
    parser, "--start-date",
    type = "character",
    help = "Evaluation start date [required]"
  )

  parser <- optparse::add_option(
    parser, "--end-date",
    type = "character",
    help = "Evaluation end date [required]"
  )

  # Areas
  parser <- optparse::add_option(
    parser, c("-a", "--areas"),
    type = "character",
    default = "all",
    help = "Comma-separated area codes [default: %default]"
  )

  # Metrics
  parser <- optparse::add_option(
    parser, "--metrics",
    type = "character",
    default = "mape,mae,rmse",
    help = "Metrics to calculate [default: %default]"
  )

  # Output
  parser <- optparse::add_option(
    parser, c("-o", "--output"),
    type = "character",
    default = NULL,
    help = "Output path for results"
  )

  parser <- optparse::add_option(
    parser, "--report",
    type = "character",
    default = NULL,
    help = "Generate comparison HTML report"
  )

  parser
}


#' Execute Eval-Combination
#' @noRd
execute_eval_combination <- function(opts) {
  config <- ConfigManager$new()
  config$load(opts$config)

  models <- strsplit(opts$models, ",")[[1]]
  strategies <- strsplit(opts$strategies, ",")[[1]]
  metrics <- strsplit(opts$metrics, ",")[[1]]

  cli_info(sprintf("Evaluating combination of: %s", paste(models, collapse = ", ")))
  cli_info(sprintf("Strategies: %s", paste(strategies, collapse = ", ")))

  # Load predictions from each model
  storage <- create_storage_backend(config$get_storage_config())
  areas <- resolve_areas(opts$areas, config)

  model_predictions <- list()
  for (model in models) {
    preds <- load_model_predictions(model, opts$`start-date`, opts$`end-date`, areas, storage)
    model_predictions[[model]] <- preds
    cli_info(sprintf("Loaded predictions for %s: %d rows", model, nrow(preds)))
  }

  # Load actual values
  actual_data <- load_actual_values(opts$`start-date`, opts$`end-date`, areas, storage)

  # Evaluate each strategy
  results <- list()

  for (strategy in strategies) {
    cli_info(sprintf("Evaluating strategy: %s", strategy))

    combiner <- get_combiner(strategy)
    combined <- combiner$combine(model_predictions)

    # Calculate metrics
    calculator <- MetricsCalculator$new()
    strategy_metrics <- calculator$calculate(
      actual = actual_data$carga_mwmed,
      predicted = combined$prediction,
      metrics = metrics
    )

    results[[strategy]] <- list(
      metrics = strategy_metrics,
      weights = combined$weights
    )
  }

  # Add individual model results for comparison
  for (model in models) {
    calculator <- MetricsCalculator$new()
    model_metrics <- calculator$calculate(
      actual = actual_data$carga_mwmed,
      predicted = model_predictions[[model]]$prediction,
      metrics = metrics
    )
    results[[model]] <- list(metrics = model_metrics, weights = NULL)
  }

  # Output results
  output_combination_results(results, strategies, models, opts)

  invisible(results)
}
```

### Combine Command

```r
#' Combine Command Handler
#'
#' Combines forecasts from multiple models.
#'
#' @param args Command line arguments
#' @noRd
cmd_combine <- function(args) {
  parser <- create_combine_parser()
  opts <- optparse::parse_args(parser, args, positional_arguments = FALSE)

  validate_combine_options(opts)
  execute_combine(opts)
}


#' Create Combine Parser
#' @noRd
create_combine_parser <- function() {
  parser <- optparse::OptionParser(
    usage = "prevcarga combine [options]",
    description = "Combine forecasts from multiple models."
  )

  # Configuration
  parser <- optparse::add_option(
    parser, c("-c", "--config"),
    type = "character",
    default = "config/config.yaml",
    help = "Path to configuration YAML file"
  )

  # Models
  parser <- optparse::add_option(
    parser, c("-m", "--models"),
    type = "character",
    help = "Comma-separated model names [required]"
  )

  # Strategy
  parser <- optparse::add_option(
    parser, c("-s", "--strategy"),
    type = "character",
    default = "inverse_mape",
    help = "Combination strategy [default: %default]"
  )

  # Date
  parser <- optparse::add_option(
    parser, c("-d", "--date"),
    type = "character",
    default = NULL,
    help = "Prediction date [default: today]"
  )

  # Areas
  parser <- optparse::add_option(
    parser, c("-a", "--areas"),
    type = "character",
    default = "all",
    help = "Comma-separated areas [default: %default]"
  )

  # Custom weights
  parser <- optparse::add_option(
    parser, "--weights",
    type = "character",
    default = NULL,
    help = "Custom weights as comma-separated values (must match model count)"
  )

  # Output
  parser <- optparse::add_option(
    parser, c("-o", "--output"),
    type = "character",
    default = NULL,
    help = "Output path for combined forecast"
  )

  parser <- optparse::add_option(
    parser, "--format",
    type = "character",
    default = "parquet",
    help = "Output format [default: %default]"
  )

  parser
}


#' Execute Combine
#' @noRd
execute_combine <- function(opts) {
  config <- ConfigManager$new()
  config$load(opts$config)

  models <- strsplit(opts$models, ",")[[1]]
  date <- opts$date %||% as.character(Sys.Date())
  areas <- resolve_areas(opts$areas, config)

  cli_info(sprintf("Combining models: %s", paste(models, collapse = ", ")))
  cli_info(sprintf("Strategy: %s", opts$strategy))
  cli_info(sprintf("Date: %s", date))

  # Load predictions
  storage <- create_storage_backend(config$get_storage_config())

  model_predictions <- list()
  for (model in models) {
    preds <- load_model_predictions(model, date, date, areas, storage)
    model_predictions[[model]] <- preds
  }

  # Combine
  if (!is.null(opts$weights)) {
    weights <- as.numeric(strsplit(opts$weights, ",")[[1]])
    if (length(weights) != length(models)) {
      stop("Number of weights must match number of models")
    }
    combiner <- get_combiner("custom", weights = weights)
  } else {
    combiner <- get_combiner(opts$strategy)
  }

  combined <- combiner$combine(model_predictions)

  # Output
  output_path <- opts$output %||%
    sprintf("predictions/combined/%s/combined.%s", date, opts$format)

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  switch(opts$format,
    "parquet" = arrow::write_parquet(combined$predictions, output_path),
    "csv" = data.table::fwrite(combined$predictions, output_path),
    "rds" = saveRDS(combined, output_path)
  )

  cli_success(sprintf("Combined forecast saved to: %s", output_path))

  # Show weights used
  cat("\nWeights used:\n")
  for (i in seq_along(models)) {
    cat(sprintf("  %s: %.4f\n", models[i], combined$weights[i]))
  }
}
```

### Usage Examples

```bash
# Evaluate a model
prevcarga eval-model \
  --model lgbm \
  --start-date 2024-10-01 \
  --end-date 2024-12-31

# With all breakdowns
prevcarga eval-model \
  --model lgbm \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --by-horizon \
  --by-period \
  --by-area

# Evaluate from direct path
prevcarga eval-model \
  --model-path models/lgbm/v1.0.0/model.rds \
  --test-data test_data.parquet

# Evaluate combination strategies
prevcarga eval-combination \
  --models lgbm,rf,hw \
  --strategies simple_average,inverse_mape,optimal \
  --start-date 2024-10-01 \
  --end-date 2024-12-31

# Combine forecasts
prevcarga combine \
  --models lgbm,rf \
  --strategy inverse_mape \
  --date 2025-01-17

# Combine with custom weights
prevcarga combine \
  --models lgbm,rf,hw \
  --weights 0.5,0.3,0.2 \
  --date 2025-01-17
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | eval-model basic | Metrics calculated |
| TC-002 | eval-model --by-horizon | Horizon breakdown |
| TC-003 | eval-model --by-period | Patamar breakdown |
| TC-004 | eval-model --by-area | Area breakdown |
| TC-005 | eval-combination all strategies | All compared |
| TC-006 | combine inverse_mape | Weights calculated |
| TC-007 | combine custom weights | Uses provided weights |
| TC-008 | Invalid model name | Error message |
| TC-009 | --report flag | Report generated |
| TC-010 | Output formats | All formats work |

---

## Dependencies

- PC-064-09: CLI Entry Point
- PC-047-07: MetricsCalculator
- PC-032-05: BaseCombiner
- PC-051-07: ModelComparator

---

## Definition of Done

- [ ] eval-model command implemented
- [ ] eval-combination command implemented
- [ ] combine command implemented
- [ ] All breakdown options working
- [ ] Multiple output formats
- [ ] Report generation
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥80% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- eval-model supports both model name and direct path
- eval-combination compares strategies side by side
- combine applies the best strategy or custom weights
- Consider adding cross-validation for weight optimization
