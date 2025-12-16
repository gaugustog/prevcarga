# PC-024-04: HierarchicalModel Base Class

**Epic:** [EPIC-04: Model Layer - Hierarchical Support](../epics/EPIC-04-model-layer-part2.md)
**Task Reference:** T-04.1
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Create the `HierarchicalModel` abstract base class that extends `BaseModel` to support the Demand Mean → Profile (DM→Profile) forecasting pattern used in Brazilian load forecasting.

---

## Acceptance Criteria

- [ ] Hierarchical model module created in `R/models/hierarchical.R`
- [ ] `HierarchicalModel` R6 class extends `BaseModel`
- [ ] Abstract methods for DM training/prediction
- [ ] Abstract methods for profile training/prediction
- [ ] `compose()` method combines DM and profile forecasts
- [ ] Support for 48 half-hour profile models
- [ ] Integration with `ModelArtifact` for saving/loading

---

## Technical Specification

### File Location
```
R/models/hierarchical.R
```

### HierarchicalModel R6 Class

```r
#' @title HierarchicalModel
#' @description Abstract base class for hierarchical DM→Profile forecasting models
#'
#' The hierarchical pattern decomposes forecasting into:
#' 1. Demand Mean (DM): Forecasts daily average load
#' 2. Profile Models: Forecast intraday load shape (48 half-hour periods)
#' 3. Composition: Final = DM × Profile
#'
#' @export
HierarchicalModel <- R6::R6Class(
  "HierarchicalModel",
  inherit = BaseModel,
  private = list(
    dm_model = NULL,
    profile_models = NULL,
    n_profiles = 48L,
    composition_method = "multiplicative"
  ),
  public = list(
    #' @description Initialize hierarchical model
    #' @param name Model name
    #' @param horizons Forecast horizons (D+0 to D+8)
    #' @param n_profiles Number of profile periods (default 48 for half-hourly)
    #' @param composition_method How to combine DM and profile ("multiplicative" or "additive")
    #' @param config Additional configuration
    initialize = function(name,
                          horizons = 0:8,
                          n_profiles = 48L,
                          composition_method = "multiplicative",
                          config = list()) {
      checkmate::assert_string(name, min.chars = 1)
      checkmate::assert_integerish(horizons, lower = 0, upper = 8)
      checkmate::assert_integerish(n_profiles, lower = 1)
      checkmate::assert_choice(composition_method, c("multiplicative", "additive"))

      super$initialize(name = name, horizons = horizons, config = config)

      private$n_profiles <- as.integer(n_profiles)
      private$composition_method <- composition_method
      private$profile_models <- vector("list", private$n_profiles)
    },

    #' @description Train demand mean model
    #' @param X Feature matrix for DM model
    #' @param y Target vector (daily average load)
    #' @param ... Additional arguments
    #' @return Invisible self
    train_dm = function(X, y, ...) {
      stop("train_dm() must be implemented by subclass")
    },

    #' @description Train profile models
    #' @param dm_residuals Residuals from DM model or profile ratios
    #' @param X Feature matrix for profile models (optional)
    #' @param ... Additional arguments
    #' @return Invisible self
    train_profiles = function(dm_residuals, X = NULL, ...) {
      stop("train_profiles() must be implemented by subclass")
    },

    #' @description Full training workflow
    #' @param X Feature data
    #' @param y_dm DM target (daily average)
    #' @param y_profiles Profile targets (48-column matrix or list)
    #' @param ... Additional arguments
    #' @return Invisible self
    train = function(X, y_dm, y_profiles = NULL, ...) {
      checkmate::assert_data_table(X)

      message(sprintf("Training hierarchical model: %s", self$name))

      # Step 1: Train DM model
      message("  Step 1: Training demand mean model...")
      self$train_dm(X, y_dm, ...)

      # Step 2: Get DM predictions/residuals for profile training
      dm_fitted <- self$predict_dm(X)
      dm_residuals <- if (private$composition_method == "multiplicative") {
        y_dm / dm_fitted  # Ratios for multiplicative
      } else {
        y_dm - dm_fitted  # Residuals for additive
      }

      # Step 3: Train profile models
      message("  Step 2: Training profile models...")
      self$train_profiles(dm_residuals, X = X, y_profiles = y_profiles, ...)

      self$is_trained <- TRUE
      message("  Hierarchical model training complete.")

      invisible(self)
    },

    #' @description Predict demand mean
    #' @param X Feature matrix
    #' @param ... Additional arguments
    #' @return Numeric vector of DM predictions
    predict_dm = function(X, ...) {
      stop("predict_dm() must be implemented by subclass")
    },

    #' @description Predict profiles
    #' @param dm_forecast DM forecast values
    #' @param X Feature matrix (optional)
    #' @param ... Additional arguments
    #' @return Matrix of profile predictions (n_obs × n_profiles)
    predict_profiles = function(dm_forecast, X = NULL, ...) {
      stop("predict_profiles() must be implemented by subclass")
    },

    #' @description Full prediction workflow
    #' @param X Feature matrix
    #' @param horizon Forecast horizon (optional)
    #' @param ... Additional arguments
    #' @return Matrix of final predictions (n_obs × n_profiles)
    predict = function(X, horizon = NULL, ...) {
      if (!self$is_trained) {
        stop("Model must be trained before prediction")
      }

      # Step 1: Predict DM
      dm_forecast <- self$predict_dm(X, ...)

      # Step 2: Predict profiles
      profile_forecast <- self$predict_profiles(dm_forecast, X = X, ...)

      # Step 3: Compose final forecast
      self$compose(dm_forecast, profile_forecast)
    },

    #' @description Compose DM and profile forecasts
    #' @param dm_forecast Vector of DM forecasts
    #' @param profile_forecast Matrix of profile forecasts
    #' @return Matrix of final forecasts
    compose = function(dm_forecast, profile_forecast) {
      checkmate::assert_numeric(dm_forecast)
      checkmate::assert_matrix(profile_forecast)

      if (length(dm_forecast) != nrow(profile_forecast)) {
        stop("DM forecast length must match profile forecast rows")
      }

      if (private$composition_method == "multiplicative") {
        # Final = DM × Profile (broadcast DM across profiles)
        sweep(profile_forecast, 1, dm_forecast, FUN = "*")
      } else {
        # Final = DM + Profile
        sweep(profile_forecast, 1, dm_forecast, FUN = "+")
      }
    },

    #' @description Get DM model
    #' @return DM model object
    get_dm_model = function() {
      private$dm_model
    },

    #' @description Get profile models
    #' @return List of profile models
    get_profile_models = function() {
      private$profile_models
    },

    #' @description Get number of profiles
    #' @return Integer
    get_n_profiles = function() {
      private$n_profiles
    },

    #' @description Get composition method
    #' @return String
    get_composition_method = function() {
      private$composition_method
    },

    #' @description Save hierarchical model
    #' @param path Base path for saving
    #' @param version ModelVersion object
    #' @return Invisible path
    save = function(path, version = NULL) {
      checkmate::assert_string(path)

      # Create directory structure
      if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
      }

      # Save DM model
      dm_path <- file.path(path, "dm_model.rds")
      saveRDS(private$dm_model, dm_path)

      # Save profile models
      profiles_path <- file.path(path, "profile_models.rds")
      saveRDS(private$profile_models, profiles_path)

      # Save metadata
      metadata <- list(
        name = self$name,
        horizons = self$horizons,
        n_profiles = private$n_profiles,
        composition_method = private$composition_method,
        config = self$config,
        is_trained = self$is_trained,
        version = if (!is.null(version)) version$to_string() else NULL,
        saved_at = Sys.time()
      )
      yaml::write_yaml(metadata, file.path(path, "metadata.yaml"))

      message(sprintf("Saved hierarchical model to: %s", path))
      invisible(path)
    },

    #' @description Load hierarchical model
    #' @param path Path to load from
    #' @return Invisible self
    load = function(path) {
      checkmate::assert_directory_exists(path)

      # Load DM model
      dm_path <- file.path(path, "dm_model.rds")
      if (!file.exists(dm_path)) {
        stop("DM model file not found: ", dm_path)
      }
      private$dm_model <- readRDS(dm_path)

      # Load profile models
      profiles_path <- file.path(path, "profile_models.rds")
      if (!file.exists(profiles_path)) {
        stop("Profile models file not found: ", profiles_path)
      }
      private$profile_models <- readRDS(profiles_path)

      # Load metadata
      metadata_path <- file.path(path, "metadata.yaml")
      if (file.exists(metadata_path)) {
        metadata <- yaml::read_yaml(metadata_path)
        self$name <- metadata$name
        self$horizons <- metadata$horizons
        private$n_profiles <- metadata$n_profiles
        private$composition_method <- metadata$composition_method
        self$config <- metadata$config
        self$is_trained <- metadata$is_trained
      }

      message(sprintf("Loaded hierarchical model from: %s", path))
      invisible(self)
    }
  )
)
```

### Helper Functions

```r
#' Create profile target matrix from hourly data
#'
#' @param data data.table with hourly load data
#' @param datetime_col Column name for datetime
#' @param load_col Column name for load values
#' @param n_profiles Number of profiles per day (48 for half-hourly)
#' @return Matrix with profile targets
#' @export
create_profile_targets <- function(data, datetime_col = "DataHora",
                                   load_col = "CargaGlobal",
                                   n_profiles = 48L) {
  checkmate::assert_data_table(data)
  checkmate::assert_choice(datetime_col, names(data))
  checkmate::assert_choice(load_col, names(data))

  # Extract date and period
  dt <- data.table::copy(data)
  dt[, `:=`(
    date = as.Date(get(datetime_col)),
    period = ((lubridate::hour(get(datetime_col)) * 60 +
               lubridate::minute(get(datetime_col))) %/% (1440 / n_profiles)) + 1L
  )]

  # Pivot to wide format
  profile_wide <- data.table::dcast(
    dt,
    date ~ period,
    value.var = load_col,
    fun.aggregate = mean
  )

  # Return as matrix (exclude date column)
  as.matrix(profile_wide[, -1])
}


#' Calculate daily demand mean from hourly data
#'
#' @param data data.table with hourly load data
#' @param datetime_col Column name for datetime
#' @param load_col Column name for load values
#' @return data.table with date and daily_mean
#' @export
calculate_daily_dm <- function(data, datetime_col = "DataHora",
                               load_col = "CargaGlobal") {
  checkmate::assert_data_table(data)

  dt <- data.table::copy(data)
  dt[, date := as.Date(get(datetime_col))]

  dt[, .(daily_mean = mean(get(load_col), na.rm = TRUE)), by = date]
}


#' Validate hierarchical model structure
#'
#' @param model HierarchicalModel instance
#' @return TRUE if valid, error otherwise
#' @export
validate_hierarchical_model <- function(model) {
  checkmate::assert_class(model, "HierarchicalModel")

  if (is.null(model$get_dm_model())) {
    stop("DM model is not set")
  }

  profiles <- model$get_profile_models()
  if (length(profiles) != model$get_n_profiles()) {
    stop(sprintf(
      "Expected %d profile models, got %d",
      model$get_n_profiles(),
      length(profiles)
    ))
  }

  invisible(TRUE)
}
```

### Usage Example

```r
# Example subclass implementation
ExampleHierarchicalModel <- R6::R6Class(
  "ExampleHierarchicalModel",
  inherit = HierarchicalModel,
  public = list(
    train_dm = function(X, y, ...) {
      # Train a simple linear model for DM
      private$dm_model <- lm(y ~ ., data = as.data.frame(X))
      invisible(self)
    },

    train_profiles = function(dm_residuals, X = NULL, y_profiles = NULL, ...) {
      # Train a model for each profile period
      for (i in seq_len(private$n_profiles)) {
        # Simple: use mean profile ratio
        private$profile_models[[i]] <- mean(y_profiles[, i], na.rm = TRUE)
      }
      invisible(self)
    },

    predict_dm = function(X, ...) {
      predict(private$dm_model, newdata = as.data.frame(X))
    },

    predict_profiles = function(dm_forecast, X = NULL, ...) {
      # Return constant profiles
      profile_vals <- sapply(private$profile_models, identity)
      matrix(profile_vals, nrow = length(dm_forecast),
             ncol = private$n_profiles, byrow = TRUE)
    }
  )
)

# Usage
model <- ExampleHierarchicalModel$new(
  name = "example_hierarchical",
  horizons = 0:8,
  n_profiles = 48,
  composition_method = "multiplicative"
)

# Prepare data
dm_targets <- calculate_daily_dm(hourly_data)
profile_targets <- create_profile_targets(hourly_data)

# Train
model$train(X = features, y_dm = dm_targets$daily_mean,
            y_profiles = profile_targets)

# Predict
forecasts <- model$predict(X = new_features)
# Returns matrix: n_days × 48 periods
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with defaults | n_profiles=48, multiplicative |
| TC-002 | Initialize with additive | composition_method set |
| TC-003 | train_dm() abstract | Error thrown |
| TC-004 | train_profiles() abstract | Error thrown |
| TC-005 | predict_dm() abstract | Error thrown |
| TC-006 | predict_profiles() abstract | Error thrown |
| TC-007 | compose() multiplicative | DM × Profile |
| TC-008 | compose() additive | DM + Profile |
| TC-009 | compose() dimension check | Error if mismatch |
| TC-010 | save() creates files | dm_model.rds, profile_models.rds, metadata.yaml |
| TC-011 | load() restores state | All fields restored |
| TC-012 | create_profile_targets() | Matrix with n_profiles columns |
| TC-013 | calculate_daily_dm() | Daily averages |
| TC-014 | validate_hierarchical_model() | Validates structure |

---

## Dependencies

- PC-016-03: BaseModel (parent class)
- PC-019-03: ModelArtifact (for persistence patterns)

---

## Definition of Done

- [ ] HierarchicalModel R6 class implemented
- [ ] Abstract methods defined for subclasses
- [ ] compose() method working
- [ ] save()/load() methods working
- [ ] Helper functions implemented
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥85% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The 48-profile pattern matches Brazilian half-hourly metering
- Multiplicative composition is more common for load forecasting
- Consider adding profile smoothing options
- Future: support for 24-profile (hourly) and 96-profile (15-min) patterns
- The DM→Profile decomposition is fundamental to RegDin+SVM and HW models
