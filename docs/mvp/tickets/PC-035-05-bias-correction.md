# PC-035-05: Bias Correction Infrastructure

**Epic:** [EPIC-05: Combination Infrastructure](../epics/EPIC-05-combination-layer.md)
**Task Reference:** T-05.4
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Create the `BiasCorrector` R6 abstract class infrastructure for systematic forecast bias correction, supporting both simple additive/multiplicative corrections and regression-based approaches.

---

## Acceptance Criteria

- [ ] Bias correction module created in `R/combination/bias_correction.R`
- [ ] `BiasCorrector` abstract R6 class
- [ ] Abstract methods: `fit()`, `correct()`, `get_correction_factors()`
- [ ] Support for additive and multiplicative correction
- [ ] Feature-conditional correction support
- [ ] Integration hooks for CombinationWorkflow

---

## Technical Specification

### File Location
```
R/combination/bias_correction.R
```

### BiasCorrector Abstract Class

```r
#' @title BiasCorrector
#' @description Abstract base class for forecast bias correction
#'
#' Bias correction adjusts systematic errors in forecasts:
#' - Additive bias: forecast consistently over/under predicts by fixed amount
#' - Multiplicative bias: forecast consistently over/under predicts by percentage
#' - Conditional bias: bias varies by hour, day, or other features
#'
#' @export
BiasCorrector <- R6::R6Class(
  "BiasCorrector",
  private = list(
    correction_model = NULL,
    is_fitted = FALSE,
    fit_metadata = NULL,
    correction_type = NULL
  ),
  public = list(
    #' @field name Corrector name
    name = NULL,

    #' @field config Configuration
    config = NULL,

    #' @description Initialize corrector
    #' @param name Corrector name
    #' @param config Configuration list
    initialize = function(name = NULL, config = list()) {
      self$name <- name %||% class(self)[1]
      self$config <- private$apply_default_config(config)
    },

    #' @description Fit correction model
    #' @param predictions Historical predictions
    #' @param actuals Corresponding actual values
    #' @param features Optional feature data for conditional correction
    #' @param ... Additional arguments
    #' @return Invisible self
    fit = function(predictions, actuals, features = NULL, ...) {
      stop("fit() must be implemented by subclass")
    },

    #' @description Apply correction to predictions
    #' @param predictions Predictions to correct
    #' @param features Optional features for conditional correction
    #' @param ... Additional arguments
    #' @return Corrected predictions
    correct = function(predictions, features = NULL, ...) {
      stop("correct() must be implemented by subclass")
    },

    #' @description Get correction factors/parameters
    #' @return List with correction information
    get_correction_factors = function() {
      stop("get_correction_factors() must be implemented by subclass")
    },

    #' @description Check if corrector is fitted
    #' @return Logical
    is_fitted = function() {
      private$is_fitted
    },

    #' @description Get fit metadata
    #' @return List
    get_fit_metadata = function() {
      private$fit_metadata
    },

    #' @description Calculate bias metrics
    #' @param predictions Predictions
    #' @param actuals Actuals
    #' @return List with bias metrics
    calculate_bias_metrics = function(predictions, actuals) {
      residuals <- actuals - predictions

      list(
        mean_bias = mean(residuals, na.rm = TRUE),
        median_bias = median(residuals, na.rm = TRUE),
        bias_std = sd(residuals, na.rm = TRUE),
        relative_bias = mean(residuals / actuals, na.rm = TRUE) * 100,
        positive_bias_pct = mean(residuals > 0, na.rm = TRUE) * 100
      )
    },

    #' @description Print corrector summary
    print = function() {
      cat(sprintf("<%s>\n", class(self)[1]))
      cat(sprintf("  Name: %s\n", self$name))
      cat(sprintf("  Fitted: %s\n", private$is_fitted))

      if (private$is_fitted) {
        factors <- tryCatch(
          self$get_correction_factors(),
          error = function(e) NULL
        )
        if (!is.null(factors)) {
          cat(sprintf("  Correction type: %s\n", private$correction_type))
        }
      }

      invisible(self)
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        correction_type = "additive",  # "additive", "multiplicative", "regression"
        min_samples = 24,
        regularization = 0.0
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    },

    set_fit_metadata = function(metadata) {
      private$fit_metadata <- c(
        metadata,
        list(fitted_at = Sys.time())
      )
    }
  )
)
```

### Simple Bias Corrector Implementation

```r
#' @title SimpleBiasCorrector
#' @description Simple additive or multiplicative bias correction
#' @export
SimpleBiasCorrector <- R6::R6Class(
  "SimpleBiasCorrector",
  inherit = BiasCorrector,
  private = list(
    bias_value = NULL
  ),
  public = list(
    #' @description Fit simple bias correction
    fit = function(predictions, actuals, features = NULL, ...) {
      checkmate::assert_numeric(predictions)
      checkmate::assert_numeric(actuals, len = length(predictions))

      if (length(predictions) < self$config$min_samples) {
        warning(sprintf(
          "Only %d samples for bias fitting (min: %d)",
          length(predictions), self$config$min_samples
        ))
      }

      residuals <- actuals - predictions

      if (self$config$correction_type == "additive") {
        # Additive: corrected = prediction + bias
        private$bias_value <- mean(residuals, na.rm = TRUE)
        private$correction_type <- "additive"

      } else if (self$config$correction_type == "multiplicative") {
        # Multiplicative: corrected = prediction * ratio
        private$bias_value <- mean(actuals / predictions, na.rm = TRUE)
        private$correction_type <- "multiplicative"

      } else {
        stop(sprintf("Unknown correction_type: %s", self$config$correction_type))
      }

      private$is_fitted <- TRUE
      private$set_fit_metadata(list(
        n_samples = length(predictions),
        bias_value = private$bias_value,
        pre_correction_bias = mean(residuals, na.rm = TRUE)
      ))

      message(sprintf(
        "Fitted %s bias correction: %.4f",
        private$correction_type, private$bias_value
      ))

      invisible(self)
    },

    #' @description Apply correction
    correct = function(predictions, features = NULL, ...) {
      if (!private$is_fitted) {
        stop("Corrector must be fitted before use")
      }

      if (private$correction_type == "additive") {
        predictions + private$bias_value
      } else {
        predictions * private$bias_value
      }
    },

    #' @description Get correction factors
    get_correction_factors = function() {
      list(
        type = private$correction_type,
        value = private$bias_value
      )
    }
  )
)
```

### Conditional Bias Corrector

```r
#' @title ConditionalBiasCorrector
#' @description Bias correction conditioned on features (hour, weekday, etc.)
#' @export
ConditionalBiasCorrector <- R6::R6Class(
  "ConditionalBiasCorrector",
  inherit = BiasCorrector,
  private = list(
    group_corrections = NULL,
    grouping_cols = NULL
  ),
  public = list(
    #' @description Initialize with grouping columns
    #' @param grouping_cols Columns to group by (e.g., c("hour", "weekday"))
    #' @param config Configuration
    initialize = function(grouping_cols = "hour", config = list()) {
      super$initialize(config = config)
      private$grouping_cols <- grouping_cols
    },

    #' @description Fit conditional bias correction
    fit = function(predictions, actuals, features, ...) {
      checkmate::assert_numeric(predictions)
      checkmate::assert_numeric(actuals)
      checkmate::assert_data_table(features)

      # Validate grouping columns exist
      for (col in private$grouping_cols) {
        if (!col %in% names(features)) {
          stop(sprintf("Grouping column '%s' not found in features", col))
        }
      }

      # Create data.table for grouping
      dt <- data.table::data.table(
        prediction = predictions,
        actual = actuals
      )
      dt <- cbind(dt, features[, ..private$grouping_cols])
      dt[, residual := actual - prediction]

      # Calculate bias per group
      if (self$config$correction_type == "additive") {
        corrections <- dt[, .(
          bias = mean(residual, na.rm = TRUE),
          n = .N
        ), by = private$grouping_cols]
      } else {
        corrections <- dt[, .(
          bias = mean(actual / prediction, na.rm = TRUE),
          n = .N
        ), by = private$grouping_cols]
      }

      private$group_corrections <- corrections
      private$correction_type <- self$config$correction_type
      private$is_fitted <- TRUE

      private$set_fit_metadata(list(
        n_groups = nrow(corrections),
        grouping_cols = private$grouping_cols,
        mean_group_size = mean(corrections$n)
      ))

      message(sprintf(
        "Fitted conditional bias correction with %d groups",
        nrow(corrections)
      ))

      invisible(self)
    },

    #' @description Apply conditional correction
    correct = function(predictions, features, ...) {
      if (!private$is_fitted) {
        stop("Corrector must be fitted before use")
      }

      checkmate::assert_data_table(features)

      # Join predictions with corrections
      dt <- data.table::data.table(
        prediction = predictions,
        .row_id = seq_along(predictions)
      )
      dt <- cbind(dt, features[, ..private$grouping_cols])

      dt <- merge(
        dt,
        private$group_corrections,
        by = private$grouping_cols,
        all.x = TRUE
      )

      # Handle missing groups with global average
      global_bias <- mean(private$group_corrections$bias, na.rm = TRUE)
      dt[is.na(bias), bias := global_bias]

      # Apply correction
      data.table::setorder(dt, .row_id)

      if (private$correction_type == "additive") {
        dt$prediction + dt$bias
      } else {
        dt$prediction * dt$bias
      }
    },

    #' @description Get correction factors
    get_correction_factors = function() {
      list(
        type = private$correction_type,
        grouping_cols = private$grouping_cols,
        group_corrections = private$group_corrections
      )
    }
  )
)
```

### Regression-Based Corrector

```r
#' @title RegressionBiasCorrector
#' @description Linear regression-based bias correction
#' @export
RegressionBiasCorrector <- R6::R6Class(
  "RegressionBiasCorrector",
  inherit = BiasCorrector,
  private = list(
    regression_model = NULL,
    feature_cols = NULL
  ),
  public = list(
    #' @description Fit regression model for bias correction
    fit = function(predictions, actuals, features, ...) {
      checkmate::assert_numeric(predictions)
      checkmate::assert_numeric(actuals)
      checkmate::assert_data_table(features)

      private$feature_cols <- names(features)

      # Build regression data
      reg_data <- data.table::copy(features)
      reg_data[, `:=`(
        prediction = predictions,
        actual = actuals
      )]

      # Fit linear model: actual ~ prediction + features
      formula_str <- sprintf(
        "actual ~ prediction + %s",
        paste(private$feature_cols, collapse = " + ")
      )

      private$regression_model <- lm(
        as.formula(formula_str),
        data = reg_data
      )

      private$correction_type <- "regression"
      private$is_fitted <- TRUE

      private$set_fit_metadata(list(
        n_samples = nrow(reg_data),
        r_squared = summary(private$regression_model)$r.squared,
        feature_cols = private$feature_cols
      ))

      message(sprintf(
        "Fitted regression bias correction (R² = %.4f)",
        private$fit_metadata$r_squared
      ))

      invisible(self)
    },

    #' @description Apply regression-based correction
    correct = function(predictions, features, ...) {
      if (!private$is_fitted) {
        stop("Corrector must be fitted before use")
      }

      pred_data <- data.table::copy(features)
      pred_data[, prediction := predictions]

      predict(private$regression_model, newdata = pred_data)
    },

    #' @description Get correction factors
    get_correction_factors = function() {
      list(
        type = "regression",
        coefficients = coef(private$regression_model),
        r_squared = summary(private$regression_model)$r.squared
      )
    }
  )
)
```

### Usage Example

```r
# Simple additive correction
corrector <- SimpleBiasCorrector$new(config = list(
  correction_type = "additive"
))

corrector$fit(historical_predictions, historical_actuals)
# Fitted additive bias correction: 234.5

corrected <- corrector$correct(new_predictions)


# Conditional correction by hour
hourly_corrector <- ConditionalBiasCorrector$new(
  grouping_cols = c("hour", "weekday"),
  config = list(correction_type = "additive")
)

hourly_corrector$fit(predictions, actuals, features)
# Fitted conditional bias correction with 168 groups

corrected <- hourly_corrector$correct(new_predictions, new_features)


# Regression-based correction
reg_corrector <- RegressionBiasCorrector$new()
reg_corrector$fit(predictions, actuals, features)
# Fitted regression bias correction (R² = 0.9876)

corrected <- reg_corrector$correct(new_predictions, new_features)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | BiasCorrector abstract methods | Errors thrown |
| TC-002 | SimpleBiasCorrector additive fit | Bias calculated |
| TC-003 | SimpleBiasCorrector multiplicative | Ratio calculated |
| TC-004 | SimpleBiasCorrector correct() | Values adjusted |
| TC-005 | ConditionalBiasCorrector fit | Group biases |
| TC-006 | ConditionalBiasCorrector correct | Conditional adjustment |
| TC-007 | ConditionalBiasCorrector missing group | Global fallback |
| TC-008 | RegressionBiasCorrector fit | Model fitted |
| TC-009 | RegressionBiasCorrector correct | Regression applied |
| TC-010 | calculate_bias_metrics() | Metrics computed |
| TC-011 | is_fitted() before/after | FALSE/TRUE |
| TC-012 | get_correction_factors() | Factors returned |

---

## Dependencies

- PC-032-05: BaseCombiner (integration)

---

## Definition of Done

- [ ] BiasCorrector abstract class implemented
- [ ] SimpleBiasCorrector implemented
- [ ] ConditionalBiasCorrector implemented
- [ ] RegressionBiasCorrector implemented
- [ ] Integration with CombinationWorkflow
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Bias correction is applied after model combination
- Conditional correction by hour captures diurnal patterns
- Regression approach can model complex bias patterns
- Consider adding rolling window for adaptive correction
- Future: add Bayesian bias correction
