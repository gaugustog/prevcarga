# PC-030-04: Dynamic Feature Selection Support

**Epic:** [EPIC-04: Model Layer - Hierarchical Support](../epics/EPIC-04-model-layer-part2.md)
**Task Reference:** T-04.7
**Priority:** Medium
**Estimated Effort:** 1 day

---

## Summary

Add infrastructure for per-horizon and per-target dynamic feature selection, supporting horizon-aware models (216/432 models) where different forecast horizons benefit from different feature subsets.

---

## Acceptance Criteria

- [ ] Feature selection module created in `R/models/feature_selection.R`
- [ ] Infrastructure for per-horizon feature selection
- [ ] Storage/loading of feature selections per target
- [ ] Integration with MultiModelManager
- [ ] Feature importance-based selection
- [ ] Support for 216-model (9 days × 24 hours) pattern

---

## Technical Specification

### File Location
```
R/models/feature_selection.R
```

### FeatureSelector R6 Class

```r
#' @title FeatureSelector
#' @description Dynamic feature selection for multi-horizon forecasting
#'
#' Manages per-horizon/per-target feature selections:
#' - Lag features become less useful as horizon increases
#' - Calendar features remain useful across all horizons
#' - Temperature features may have different optimal lags
#'
#' @export
FeatureSelector <- R6::R6Class(
  "FeatureSelector",
  private = list(
    selections = NULL,       # Named list: {target: feature_names}
    importance_scores = NULL, # Named list: {target: importance_df}
    config = NULL
  ),
  public = list(
    #' @description Initialize feature selector
    #' @param config Selection configuration
    initialize = function(config = list()) {
      private$selections <- list()
      private$importance_scores <- list()
      private$config <- private$apply_default_config(config)
    },

    #' @description Select features for a specific target/horizon
    #' @param X Feature matrix
    #' @param y Target vector
    #' @param target Target name (e.g., "y_h024" for hour 24)
    #' @param method Selection method
    #' @param ... Additional arguments for selection method
    #' @return Character vector of selected features
    select = function(X, y, target, method = "importance", ...) {
      checkmate::assert_data_table(X)
      checkmate::assert_string(target)

      message(sprintf("Selecting features for target: %s", target))

      selected <- switch(method,
        importance = self$select_by_importance(X, y, target, ...),
        correlation = self$select_by_correlation(X, y, target, ...),
        horizon_aware = self$select_horizon_aware(X, y, target, ...),
        manual = self$get_manual_selection(target),
        stop(sprintf("Unknown selection method: %s", method))
      )

      # Store selection
      private$selections[[target]] <- selected

      message(sprintf("  Selected %d features", length(selected)))
      selected
    },

    #' @description Select features by model importance
    #' @param X Feature matrix
    #' @param y Target vector
    #' @param target Target name
    #' @param model_type Model type for importance calculation
    #' @param top_n Number of top features to select
    #' @param min_importance Minimum importance threshold
    #' @return Character vector
    select_by_importance = function(X, y, target,
                                    model_type = "lightgbm",
                                    top_n = NULL,
                                    min_importance = 0.01) {
      # Train a quick model for importance
      model <- get_model(model_type, config = list(
        num_iterations = 100,  # Quick training
        verbose = -1
      ))

      model$train(X, y)
      importance <- model$get_feature_importance()

      if (is.null(importance)) {
        warning("Model doesn't support feature importance, returning all features")
        return(names(X))
      }

      # Store importance scores
      private$importance_scores[[target]] <- importance

      # Filter by importance
      importance <- importance[importance$importance >= min_importance, ]

      # Select top N if specified
      if (!is.null(top_n)) {
        importance <- importance[order(-importance$importance), ][1:min(top_n, nrow(importance)), ]
      }

      importance$feature
    },

    #' @description Select features by correlation with target
    #' @param X Feature matrix
    #' @param y Target vector
    #' @param target Target name
    #' @param min_correlation Minimum absolute correlation
    #' @param max_features Maximum features to select
    #' @return Character vector
    select_by_correlation = function(X, y, target,
                                     min_correlation = 0.1,
                                     max_features = 50) {
      correlations <- sapply(names(X), function(col) {
        if (is.numeric(X[[col]])) {
          abs(cor(X[[col]], y, use = "pairwise.complete.obs"))
        } else {
          0
        }
      })

      # Filter and sort
      correlations <- correlations[correlations >= min_correlation]
      correlations <- sort(correlations, decreasing = TRUE)

      # Limit to max features
      if (length(correlations) > max_features) {
        correlations <- correlations[1:max_features]
      }

      names(correlations)
    },

    #' @description Horizon-aware feature selection
    #' @param X Feature matrix
    #' @param y Target vector
    #' @param target Target name (expected format: y_hNNN)
    #' @param ... Additional arguments
    #' @return Character vector
    select_horizon_aware = function(X, y, target, ...) {
      # Extract horizon from target name
      horizon <- private$parse_horizon(target)

      # Get all available features
      all_features <- names(X)

      # Always include: calendar, categorical
      base_features <- grep(
        "^(hour|weekday|month|is_holiday|is_weekend|area)",
        all_features,
        value = TRUE
      )

      # Lag features: only include if lag <= horizon (in hours)
      horizon_hours <- horizon  # horizon is already in hours
      lag_pattern <- "^lag_(\\d+)"
      lag_features <- grep(lag_pattern, all_features, value = TRUE)

      valid_lag_features <- sapply(lag_features, function(f) {
        lag_hours <- as.integer(sub(lag_pattern, "\\1", f))
        lag_hours >= horizon_hours  # Only use lags that are "available" at forecast time
      })

      lag_features <- lag_features[valid_lag_features]

      # Rolling features: similar logic
      rolling_pattern <- "^rolling_(\\d+)"
      rolling_features <- grep(rolling_pattern, all_features, value = TRUE)

      # Weather features: include all (assuming they are forecasts)
      weather_features <- grep(
        "^(temp|humidity|pressure|wind)",
        all_features,
        value = TRUE
      )

      # Combine
      selected <- unique(c(
        base_features,
        lag_features,
        rolling_features,
        weather_features
      ))

      # Store metadata
      attr(selected, "horizon") <- horizon
      attr(selected, "method") <- "horizon_aware"

      selected
    },

    #' @description Get manual/configured selection for target
    #' @param target Target name
    #' @return Character vector or NULL
    get_manual_selection = function(target) {
      private$config$manual_selections[[target]]
    },

    #' @description Set manual selection for target
    #' @param target Target name
    #' @param features Feature names
    #' @return Invisible self
    set_manual_selection = function(target, features) {
      checkmate::assert_string(target)
      checkmate::assert_character(features)

      if (is.null(private$config$manual_selections)) {
        private$config$manual_selections <- list()
      }

      private$config$manual_selections[[target]] <- features
      private$selections[[target]] <- features

      invisible(self)
    },

    #' @description Get stored selection for target
    #' @param target Target name
    #' @return Character vector or NULL
    get_selection = function(target) {
      private$selections[[target]]
    },

    #' @description Get all selections
    #' @return Named list of feature selections
    get_all_selections = function() {
      private$selections
    },

    #' @description Get importance scores for target
    #' @param target Target name
    #' @return data.frame with feature importance
    get_importance = function(target) {
      private$importance_scores[[target]]
    },

    #' @description Generate selections for all horizons
    #' @param X Feature matrix
    #' @param y_matrix Target matrix (columns = targets)
    #' @param method Selection method
    #' @param ... Additional arguments
    #' @return Named list of selections
    select_all_horizons = function(X, y_matrix, method = "horizon_aware", ...) {
      target_names <- colnames(y_matrix)

      if (is.null(target_names)) {
        target_names <- sprintf("y_h%03d", seq_len(ncol(y_matrix)))
      }

      for (i in seq_along(target_names)) {
        target <- target_names[i]
        y <- y_matrix[, i]

        self$select(X, y, target, method = method, ...)
      }

      private$selections
    },

    #' @description Save selections to file
    #' @param path File path
    #' @return Invisible path
    save = function(path) {
      data <- list(
        selections = private$selections,
        importance_scores = private$importance_scores,
        config = private$config,
        saved_at = Sys.time()
      )

      saveRDS(data, path)
      message(sprintf("Saved feature selections to: %s", path))
      invisible(path)
    },

    #' @description Load selections from file
    #' @param path File path
    #' @return Invisible self
    load = function(path) {
      checkmate::assert_file_exists(path)

      data <- readRDS(path)
      private$selections <- data$selections
      private$importance_scores <- data$importance_scores
      private$config <- data$config

      message(sprintf("Loaded %d feature selections from: %s",
                      length(private$selections), path))
      invisible(self)
    },

    #' @description Summary of selections
    #' @return data.table with selection summary
    summary = function() {
      if (length(private$selections) == 0) {
        return(data.table::data.table())
      }

      data.table::rbindlist(lapply(names(private$selections), function(target) {
        data.table::data.table(
          target = target,
          n_features = length(private$selections[[target]]),
          features_sample = paste(
            head(private$selections[[target]], 3),
            collapse = ", "
          )
        )
      }))
    }
  ),

  private = list(
    apply_default_config = function(config) {
      defaults <- list(
        default_method = "horizon_aware",
        min_importance = 0.01,
        min_correlation = 0.1,
        max_features = 100,
        manual_selections = list()
      )

      for (key in names(defaults)) {
        if (!key %in% names(config)) {
          config[[key]] <- defaults[[key]]
        }
      }

      config
    },

    parse_horizon = function(target) {
      # Parse horizon from target name like "y_h024" -> 24
      if (grepl("^y_h(\\d+)$", target)) {
        as.integer(sub("^y_h(\\d+)$", "\\1", target))
      } else if (grepl("^target_(\\d+)$", target)) {
        as.integer(sub("^target_(\\d+)$", "\\1", target))
      } else {
        # Default: assume it's a direct horizon number
        0
      }
    }
  )
)
```

### Horizon-Aware Feature Configuration

```r
#' @title HorizonFeatureConfig
#' @description Configuration for horizon-aware feature selection
#' @export
HorizonFeatureConfig <- R6::R6Class(
  "HorizonFeatureConfig",
  public = list(
    #' @description Get recommended features for a horizon
    #' @param horizon Forecast horizon (hours ahead)
    #' @param all_features Available feature names
    #' @return Character vector of recommended features
    get_recommended_features = function(horizon, all_features) {
      # Tier 1: Always available (calendar, weather forecasts)
      tier1 <- c(
        "hour", "weekday", "month", "day_of_year",
        "is_holiday", "is_weekend", "is_bridge_day",
        "temp_forecast", "humidity_forecast"
      )

      # Tier 2: Available based on horizon (lags)
      # For D+1 (24h ahead), we can use lag_24 and beyond
      # For D+2 (48h ahead), we can use lag_48 and beyond
      min_lag <- horizon
      tier2 <- grep(sprintf("^lag_(\\d+)$"), all_features, value = TRUE)
      tier2 <- Filter(function(f) {
        lag_val <- as.integer(sub("^lag_(\\d+)$", "\\1", f))
        lag_val >= min_lag
      }, tier2)

      # Tier 3: Rolling statistics (window must end before forecast time)
      tier3 <- grep("^rolling_", all_features, value = TRUE)

      # Tier 4: Interaction features
      tier4 <- grep("^(hour_x_|temp_x_)", all_features, value = TRUE)

      # Combine and filter to available
      recommended <- unique(c(tier1, tier2, tier3, tier4))
      intersect(recommended, all_features)
    },

    #' @description Create horizon-feature matrix
    #' @param horizons Vector of horizons
    #' @param all_features All available features
    #' @return data.table showing which features apply to which horizons
    create_feature_matrix = function(horizons, all_features) {
      matrix_list <- lapply(horizons, function(h) {
        features <- self$get_recommended_features(h, all_features)
        data.table::data.table(
          feature = all_features,
          selected = all_features %in% features
        )[, horizon := h]
      })

      data.table::rbindlist(matrix_list)
    }
  )
)
```

### Integration with MultiModelManager

```r
#' Train MultiModelManager with dynamic feature selection
#'
#' @param manager MultiModelManager instance
#' @param X Feature matrix
#' @param y Target matrix
#' @param target_cols Target column names
#' @param selector FeatureSelector instance
#' @param selection_method Selection method
#' @param ... Additional arguments
#' @return MultiModelManager with trained models
#' @export
train_with_feature_selection <- function(manager, X, y, target_cols,
                                         selector = NULL,
                                         selection_method = "horizon_aware",
                                         ...) {
  checkmate::assert_class(manager, "MultiModelManager")

  # Create selector if not provided
  if (is.null(selector)) {
    selector <- FeatureSelector$new()
  }

  # Generate selections for all targets
  y_dt <- as.data.table(y)
  names(y_dt) <- target_cols

  feature_selections <- list()

  for (target in target_cols) {
    selected <- selector$select(
      X = X,
      y = y_dt[[target]],
      target = target,
      method = selection_method,
      ...
    )
    feature_selections[[target]] <- selected
  }

  # Train manager with selections
  manager$train_multi(
    X = X,
    y = y_dt,
    target_cols = target_cols,
    feature_selections = feature_selections,
    ...
  )

  # Store selector in manager config
  manager$config$feature_selector <- selector

  manager
}
```

### Usage Example

```r
# Create feature selector
selector <- FeatureSelector$new(config = list(
  min_importance = 0.02,
  max_features = 50
))

# Select features for a single target
features_h024 <- selector$select(
  X = feature_matrix,
  y = targets$y_h024,
  target = "y_h024",
  method = "horizon_aware"
)

# Select features for all horizons
targets <- sprintf("y_h%03d", 1:216)
y_matrix <- as.matrix(target_data[, ..targets])

all_selections <- selector$select_all_horizons(
  X = feature_matrix,
  y_matrix = y_matrix,
  method = "horizon_aware"
)

# View summary
selector$summary()
#        target n_features        features_sample
# 1:   y_h001         45    hour, weekday, lag_1
# 2:   y_h024         38   hour, weekday, lag_24
# 3:   y_h048         32   hour, weekday, lag_48
# ...

# Save selections
selector$save("models/feature_selections.rds")

# Use with MultiModelManager
manager <- create_horizon_model_manager("lgbm_horizon", "lightgbm")
manager <- train_with_feature_selection(
  manager = manager,
  X = feature_matrix,
  y = target_matrix,
  target_cols = targets,
  selector = selector,
  selection_method = "horizon_aware"
)

# Horizon-aware configuration
config <- HorizonFeatureConfig$new()
h24_features <- config$get_recommended_features(
  horizon = 24,
  all_features = names(feature_matrix)
)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize selector | Empty selections |
| TC-002 | select() by importance | Top features selected |
| TC-003 | select() by correlation | Correlated features |
| TC-004 | select() horizon_aware | Appropriate lags |
| TC-005 | select_all_horizons() | All targets have selections |
| TC-006 | get_selection() existing | Returns selection |
| TC-007 | get_selection() missing | Returns NULL |
| TC-008 | save()/load() | Selections preserved |
| TC-009 | HorizonFeatureConfig | Horizon-appropriate features |
| TC-010 | train_with_feature_selection() | Manager trained |
| TC-011 | Parse horizon from target | Correct extraction |
| TC-012 | Manual selection | Manual features used |

---

## Dependencies

- PC-025-04: MultiModelManager (integration)
- PC-017-03: ModelRegistry (for get_model)

---

## Definition of Done

- [ ] FeatureSelector R6 class implemented
- [ ] Multiple selection methods
- [ ] Horizon-aware selection logic
- [ ] HorizonFeatureConfig utility
- [ ] Integration with MultiModelManager
- [ ] save()/load() persistence
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Horizon-aware selection is critical for multi-step forecasting
- Lag features become unavailable as horizon increases
- Consider caching importance scores for performance
- Future: add automated feature selection pipelines
- The 216-model pattern creates many selection tasks - use parallel processing
