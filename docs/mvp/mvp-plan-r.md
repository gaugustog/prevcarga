# MACRO PLAN - Unified Electric Load Forecasting System (R CLI)

**Version:** 1.0.0
**Date:** 2025-12-07
**Objective:** Modernize and unify short-term electric load forecasting systems (D+0 to D+8) as an R-native CLI implementation

---

## Overview

### Context
Migration from PrevCargaDESSEM system to a unified R platform that integrates multiple forecasting models with hierarchical architectures (DM→Profile) and end-to-end, supporting:
- 26 time series (17 areas + 4 subsystems + 4 losses + 1 national)
- Multiple base models (pluggable architecture)
- Bottom-up hierarchical reconciliation
- Advanced model combination (Stacking, Voting, Markov Chain)
- Operational intraday execution (every 30min) and daily batch

### Architectural Principles
- **Modularity**: Independent and pluggable R6 components
- **Extensibility**: Plugin system for models, features, combination and metrics
- **Reproducibility**: Configurable seeds, semantic versioning
- **Flexibility**: YAML configuration, support for multiple scenarios
- **Performance**: Parallel execution with `future` package
- **Observability**: Structured logging, comparative metrics, drift detection

---

## System Architecture

```
+---------------------------------------------------------------------+
|                     R CLI Shell (readline/cli)                       |
|  Commands: train | predict | backtest | combine | eval | forecast    |
|            --storage-backend [s3|local]                              |
+--------------------------------+------------------------------------+
                                 |
              +------------------+------------------+
              |          Orchestrator Layer         |
              |  - Config Management (yaml/config)  |
              |  - Workflow Execution (R6)          |
              |  - Parallel Coordination (future)   |
              +------------------+------------------+
                                 |
       +-------------------------+-------------------------+
       |                         |                         |
+------v------+          +-------v-------+          +------v------+
|    Data     |          |    Feature    |          |    Model    |
|    Layer    |          |    Layer      |          |    Layer    |
| (arrow/DBI) |          |  (data.table) |          |    (R6)     |
+------+------+          +-------+-------+          +------+------+
       |                         |                         |
+------v-------------------------v-------------------------v------+
|              Storage Backend (R6 Abstract Class)                 |
|   +-------------------+        +--------------------+            |
|   | S3Backend (paws)  |        | LocalBackend       |            |
|   | (Production)      |        | (Development)      |            |
|   +-------------------+        +--------------------+            |
|   raw/ features/ models/ results/                                |
+-------+------------------------+------------------------+--------+
        |                        |                        |
+-------v-------+        +-------v--------+       +-------v-------+
|   Combiner    |        |  Reconciler    |       |   Evaluator   |
|    Layer      |        |    Layer       |       |    Layer      |
+---------------+        +----------------+       +---------------+
```

---

## Storage Structure

**Supports both S3 (production) and Local Filesystem (development) using Hive-style partitioning**

### Data Structure (Hive-style Partitioning)
```yaml
./data/                                    # Base path (configurable)
│
├── load/                                  # Load data (primary dataset)
│   └── area_code={CODE}/                  # Ex: area_code=RJ, area_code=SP
│       └── year={YEAR}/                   # Ex: year=2024
│           └── data.parquet               # Hourly load values
│
├── auxiliary/                             # Supporting metadata
│   ├── dst_periods/
│   │   └── data.parquet                   # DST transition periods
│   ├── holidays/
│   │   └── area_code={CODE}/
│   │       └── year={YEAR}/
│   │           └── data.parquet           # Holiday calendar
│   └── load_tiers/
│       └── area_code={CODE}/
│           └── data.parquet               # Load tier configuration (static)
│
├── weather/
│   ├── forecast/                          # Weather forecasts
│   │   └── area_code={CODE}/
│   │       └── year={YEAR}/
│   │           └── data.parquet
│   └── observed/                          # Observed weather
│       └── area_code={CODE}/
│           └── year={YEAR}/
│               └── data.parquet
│
├── models/                                # Trained models
│   └── {model}/                           # Ex: lgbm, rf, holt_winters
│       └── {version}/                     # Ex: v1.2.3_20250117_143022
│           ├── metadata.yaml              # Config, metrics, training date
│           ├── model.rds                  # Serialized model
│           ├── normalization.rds          # Normalization parameters
│           ├── feature_names.json
│           └── latest -> symlink          # Points to latest version
│
├── results/                               # Forecasts and evaluations
│   ├── predictions/
│   │   └── {execution_date}/              # Ex: 20250117_080000
│   │       ├── individual/                # Predictions by model
│   │       ├── combined/                  # Combined predictions
│   │       └── reconciled/                # Reconciled predictions
│   └── backtests/
│       └── {backtest_id}/
│           ├── config.yaml
│           ├── metrics_summary.csv
│           ├── predictions_full.parquet
│           └── report.html
│
└── cache/                                 # Temporary cache (configurable TTL)
    └── {execution_id}/
```

### S3 Structure
```yaml
s3://prevcarga-bucket-sandbox/
│
├── load/                                  # Same Hive-style partitioning
│   └── area_code={CODE}/year={YEAR}/data.parquet
├── auxiliary/
├── weather/
├── models/
├── results/
└── cache/
```

**Note:** Both S3 and local use identical Hive-style partitioning, enabling transparent backend switching.

---

## R Project Directory Structure

```
prevcargaons/
│
├── DESCRIPTION                    # Package metadata
├── NAMESPACE                      # Exported functions
├── .Rprofile                      # Environment setup
├── renv.lock                      # Dependency lock file
│
├── R/                             # Source code
│   ├── cli/                       # Command-line interface
│   │   ├── main.R                 # CLI entry point and shell
│   │   ├── commands/
│   │   │   ├── train.R            # train command
│   │   │   ├── predict.R          # predict command
│   │   │   ├── backtest.R         # backtest command
│   │   │   ├── forecast.R         # quick forecast
│   │   │   ├── combine.R          # combination commands
│   │   │   ├── evaluate.R         # evaluation commands
│   │   │   └── features.R         # feature engineering commands
│   │   ├── shell.R                # Interactive shell with REPL
│   │   ├── display.R              # ANSI colors, progress bars, charts
│   │   └── validators.R           # Input validation
│   │
│   ├── data/                      # Data layer
│   │   ├── loader.R               # DataLoader (uses StorageBackend)
│   │   ├── validators.R           # Schema validation (checkmate)
│   │   ├── preprocessors.R        # Imputers, resampling
│   │   └── catalog.R              # DataCatalog (dataset registry)
│   │
│   ├── features/                  # Feature engineering
│   │   ├── base.R                 # BaseFeaturePlugin (R6)
│   │   ├── registry.R             # FeaturePluginRegistry
│   │   ├── common.R               # Lag, dummy, cyclical features
│   │   ├── pipeline.R             # FeaturePipeline composer
│   │   ├── plugins/
│   │   │   ├── wavelet.R          # Haar wavelet transform
│   │   │   ├── loess.R            # LOESS smoothing
│   │   │   ├── blf.R              # BLF strategy features
│   │   │   ├── rf_selector.R      # RF dynamic feature selection
│   │   │   └── hierarchical.R     # DM->Profile features
│   │   └── evaluation.R           # Feature importance analysis
│   │
│   ├── models/                    # Model implementations
│   │   ├── base.R                 # BaseModel (R6), ModelArtifact
│   │   ├── registry.R             # ModelRegistry
│   │   ├── versioning.R           # Semantic versioning
│   │   └── trainer.R              # UniversalTrainer
│   │
│   ├── combination/               # Ensemble methods
│   │   ├── base.R                 # BaseCombiner (R6)
│   │   ├── registry.R             # CombinerRegistry
│   │   ├── voting.R               # WeightedVoting, MedianVoting
│   │   ├── stacking.R             # StackingCombiner
│   │   ├── markov.R               # MarkovChainCombiner
│   │   ├── bias_correction.R      # BiasCorrector
│   │   └── optimizer.R            # WeightOptimizer
│   │
│   ├── reconciliation/            # Hierarchical reconciliation
│   │   ├── base.R                 # BaseReconciler (R6)
│   │   ├── mint.R                 # MinT reconciliation
│   │   ├── ols.R                  # OLS reconciliation
│   │   ├── wls.R                  # WLS reconciliation
│   │   ├── hierarchy.R            # HierarchyDefinition
│   │   └── loss_calculation.R     # Loss calculator
│   │
│   ├── evaluation/                # Metrics and reporting
│   │   ├── metrics.R              # MAPE, MAE, RMSE, percentiles
│   │   ├── comparator.R           # ModelComparator
│   │   ├── drift.R                # DriftDetector
│   │   └── reporter.R             # HTML/CSV reporters
│   │
│   ├── orchestrator/              # Workflow coordination
│   │   ├── workflows/
│   │   │   ├── train.R            # TrainWorkflow
│   │   │   ├── predict.R          # PredictWorkflow
│   │   │   └── backtest.R         # BacktestWorkflow
│   │   ├── parallel.R             # ParallelExecutor (future)
│   │   ├── config.R               # ConfigManager
│   │   └── logger.R               # StructuredLogger
│   │
│   ├── storage/                   # Storage abstraction
│   │   ├── base.R                 # StorageBackend (R6 abstract)
│   │   ├── s3.R                   # S3StorageBackend (paws)
│   │   ├── local.R                # LocalStorageBackend
│   │   └── factory.R              # StorageFactory
│   │
│   └── utils/                     # Utilities
│       ├── config.R               # Configuration helpers
│       ├── dates.R                # Date/time utilities
│       ├── wavelets.R             # Wavelet functions
│       └── normalization.R        # Min-max, z-score
│
├── inst/
│   ├── config/
│   │   ├── default.yaml           # Default configuration
│   │   ├── prod.yaml              # Production config
│   │   └── dev.yaml               # Development config
│   ├── shell/
│   │   └── prevcarga              # Shell entry script
│   └── templates/
│       └── reports/               # Highcharter report templates
│           ├── _common.R          # Shared helpers and theme
│           ├── forecast_report.Rmd
│           ├── backtest_report.Rmd
│           ├── model_comparison.Rmd
│           ├── area_dashboard.Rmd
│           └── drift_report.Rmd
│
├── tests/                         # Test suite (testthat)
│   ├── testthat/
│   │   ├── test-data-loader.R
│   │   ├── test-models.R
│   │   ├── test-combination.R
│   │   ├── test-reconciliation.R
│   │   └── test-cli.R
│   └── testthat.R
│
├── man/                           # Documentation (roxygen2)
├── vignettes/                     # Long-form documentation
├── data-raw/                      # Raw data scripts
├── scripts/                       # Utility scripts
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── .github/
    └── workflows/
        ├── ci.yml
        └── release.yml
```

---

## Main Components

### 1. Data Layer (`R/data/`)

**Responsibilities:**
- Load raw data from any storage backend (load, temperature, holidays)
- Schema validation (checkmate)
- Missing values imputation
- Filtering by area/period
- Hourly → semi-hourly conversion
- Backend-agnostic data operations

**Modules:**
```r
data/
├── loader.R          # DataLoader (unified, uses StorageBackend)
├── validators.R      # CargaSchema, TemperaturaSchema
├── preprocessors.R   # ImputerChain, ResamplerMixin
└── catalog.R         # DataCatalog (dataset registry)
```

**Storage Backend Integration:**
```r
library(R6)

# Auto-detect or explicit backend
backend <- StorageFactory$new()$from_config(config)
# or
backend <- StorageFactory$new()$create("local", base_path = "./data")
# or
backend <- StorageFactory$new()$create("s3", bucket = "prevcarga-bucket")

# DataLoader works with any backend
loader <- DataLoader$new(storage_backend = backend)
df <- loader$load_carga(c("RJ", "SP"), start_date, end_date)
```

---

### 2. Feature Engineering Layer (`R/features/`)

**Responsibilities:**
- Common feature engineering (lags, dummies, cyclical)
- Model-specific feature engineering (plugins)
- Feature selection/evaluation
- Modular and composable pipeline

**Base Plugin Interface:**
```r
#' @title BaseFeaturePlugin
#' @description Abstract base class for feature plugins
BaseFeaturePlugin <- R6::R6Class(
  "BaseFeaturePlugin",
  public = list(
    name = NULL,
    config = NULL,

    transform = function(dt, ...) {
      stop("transform() must be implemented by subclass")
    },

    get_feature_names = function() {
      stop("get_feature_names() must be implemented by subclass")
    }
  )
)
```

> **See:** `docs/mvp/plugin-guide/02-feature-engineering.md` for detailed plugin implementation guide.

**Modules:**
```r
features/
├── base.R               # BaseFeaturePlugin (R6)
├── registry.R           # FeaturePluginRegistry
├── pipeline.R           # FeaturePipeline (compose plugins)
└── plugins/             # Plugin implementations (contributed)
```

---

### 3. Model Layer (`R/models/`)

**Responsibilities:**
- Plugin system for forecasting models
- Training, prediction, serialization
- Semantic versioning and registry

**Base Model Interface:**
```r
#' @title BaseModel
#' @description Abstract base class for forecasting models
BaseModel <- R6::R6Class(
  "BaseModel",
  public = list(
    name = NULL,
    horizons = NULL,
    config = NULL,
    is_trained = FALSE,

    train = function(X, y, ...) {
      stop("train() must be implemented by subclass")
    },

    predict = function(X, ...) {
      stop("predict() must be implemented by subclass")
    },

    save = function(path, version) {
      stop("save() must be implemented by subclass")
    },

    load = function(path) {
      stop("load() must be implemented by subclass")
    }
  )
)
```

> **See:** `docs/mvp/plugin-guide/03-model-training.md` and `docs/mvp/plugin-guide/04-model-inference.md` for detailed model implementation guides.

**Modules:**
```r
models/
├── base.R               # BaseModel (R6)
├── registry.R           # ModelRegistry
└── implementations/     # Model implementations (contributed)
```

---

### 4. Combination Layer (`R/combination/`)

**Responsibilities:**
- Combination of predictions from multiple models
- Pluggable strategies (Voting, Stacking, Markov)
- Bias correction
- Weight optimization by context

**Available Strategies:**
- Voting (simple, weighted, inverse_error)
- Stacking (meta-model: LightGBM or LinearRegression)
- Markov Chain (dynamic weights per interval)
- Bias Correction (residual adjustment)

**Base Combiner Interface:**
```r
#' @title BaseCombiner
#' @description Abstract base class for combination strategies
BaseCombiner <- R6::R6Class(
  "BaseCombiner",
  public = list(
    fit = function(predictions, actual, context, ...) {
      stop("fit() must be implemented by subclass")
    },

    combine = function(predictions, context, ...) {
      stop("combine() must be implemented by subclass")
    },

    get_weights = function() {
      stop("get_weights() must be implemented by subclass")
    }
  )
)
```

**Modules:**
```r
combination/
├── base.R               # BaseCombiner (R6)
├── registry.R           # CombinerRegistry
└── implementations/     # Strategy implementations (voting, stacking, markov, etc.)
```

---

### 5. Reconciliation Layer (`R/reconciliation/`)

**Responsibilities:**
- Bottom-up hierarchical reconciliation
- Ensure Subsystem = Σ(Areas)
- Losses by difference (absorb discrepancies)
- Methods: MinT, OLS, WLS

**Hierarchy:**
```
SIN (National) - area_code=SIN, RECONCILIATION ONLY (no models)
├── SECO (Subsystem) - RECONCILIATION ONLY (no models)
│   ├── RJ, SP, MG, ES, MT, MS, AC, RO, DF, GO (Areas)
│   └── PESE (Loss Area - models OR by difference)
├── S (Subsystem) - RECONCILIATION ONLY (no models)
│   ├── PR, SC, RS (Areas)
│   └── PES (Loss Area - models OR by difference)
├── NE (Subsystem) - RECONCILIATION ONLY (no models)
│   ├── ALPE, PBRN, BASE, CE, PI, BAOE (Areas)
│   └── PENE (Loss Area - models OR by difference)
└── N (Subsystem) - RECONCILIATION ONLY (no models)
    ├── AM, PA, MA, TO, RR, AP (Areas)
    └── PEN (Loss Area - models OR by difference)
```

**Key Design Decisions:**
- **Areas (21 total)**: Can have models run (17 regular + 4 loss areas)
- **Loss Areas (PESE, PES, PENE, PEN)**: Can have models OR be calculated by difference
- **Subsystems (SECO, S, NE, N)**: No models, only reconciled from area forecasts
- **SIN (National)**: area_code=SIN, no models, reconciled from subsystem totals

**Algorithm (with loss areas by difference):**
```r
1. Forecast areas independently (up to 21 series if losses have models)
2. Aggregate areas by subsystem: SECO_base = Σ(RJ, SP, ..., PESE)
3. Reconcile using MinT/OLS/WLS: SECO_rec, areas_rec
4. If loss has no model: PESE = SECO_rec - Σ(other_areas_rec)
5. SIN (National) = Σ(subsystems_rec)
```

**Modules:**
```r
reconciliation/
├── base.R               # BaseReconciler (R6)
├── mint.R               # MinTReconciler (default)
├── ols.R                # OLSReconciler
├── wls.R                # WLSReconciler (variance/error weights)
├── hierarchy.R          # HierarchyDefinition (YAML → Graph)
└── loss_calculation.R   # LossCalculator (losses by difference)
```

---

### 6. Evaluation Layer (`R/evaluation/`)

**Responsibilities:**
- Metric calculation (MAPE, MAE, RMSE, percentiles)
- Metrics by time period (peak/off-peak)
- Drift detection (temporal degradation)
- Report generation

**Implemented Metrics:**
```r
calculate_metrics <- function(actual, predicted) {
  list(
    mape = mean(abs((actual - predicted) / actual)) * 100,
    mae = mean(abs(actual - predicted)),
    rmse = sqrt(mean((actual - predicted)^2)),
    percentiles = quantile(abs(actual - predicted),
                           probs = c(0.05, 0.25, 0.5, 0.75, 0.95)),
    max_deviation_abs = max(abs(actual - predicted)),
    max_deviation_rel = max(abs((actual - predicted) / actual)) * 100
  )
}
```

**Drift Detection:**
```r
DriftDetector <- R6::R6Class(
  "DriftDetector",
  public = list(
    window = 30,
    threshold_mape = 1.2,  # +20%
    threshold_mae = 1.15,  # +15%

    detect = function(current_metrics, baseline_metrics) {
      drift_detected <- FALSE

      if (current_metrics$mape / baseline_metrics$mape > self$threshold_mape) {
        drift_detected <- TRUE
      }
      if (current_metrics$mae / baseline_metrics$mae > self$threshold_mae) {
        drift_detected <- TRUE
      }

      list(
        drift_detected = drift_detected,
        mape_ratio = current_metrics$mape / baseline_metrics$mape,
        mae_ratio = current_metrics$mae / baseline_metrics$mae
      )
    }
  )
)
```

**Modules:**
```r
evaluation/
├── metrics.R            # MAPE, MAE, RMSE, Percentiles
├── comparator.R         # ModelComparator (comparison matrix)
├── drift.R              # DriftDetector
├── reporter.R           # HTMLReporter, CSVReporter
└── highcharter_reporter.R  # HighcharterReporter (interactive HTML)
```

---

### 7. Orchestrator Layer (`R/orchestrator/`)

**Responsibilities:**
- Workflow coordination (train, predict, backtest)
- Multi-area/multi-model parallelization
- Configuration management
- Logging and monitoring

**Workflows:**

#### 7.1 Train Workflow
```r
TrainWorkflow <- R6::R6Class(
  "TrainWorkflow",
  public = list(
    run = function(config, storage) {
      # 1. Load historical data (3 years)
      loader <- DataLoader$new(storage)
      data <- loader$load_carga(config$areas, config$start_date, config$end_date)

      # 2. For each area (parallelization)
      future::plan(future::multisession, workers = config$parallel$n_jobs)

      results <- future.apply::future_lapply(config$areas, function(area) {
        area_data <- data[area_code == area]

        # a. Apply feature engineering (model plugins)
        pipeline <- FeaturePipeline$new(config$feature_plugins)
        features <- pipeline$transform(area_data)

        # b. Train models (intra-model parallelization)
        lapply(config$models, function(model_name) {
          model <- .model_registry$create(model_name, config$model_config[[model_name]])
          model$train(features$X, features$y)

          # c. Validate and calculate metrics
          # d. Version and save
          model$save(storage, config$version)
          model
        })
      }, future.seed = TRUE)

      # 3. Train combination meta-models
      # 4. Generate training report

      results
    }
  )
)
```

#### 7.2 Predict Workflow
```r
PredictWorkflow:
  1. Load latest models from storage
  2. Load data up to D+0 (or intraday up to current hour)
  3. For each area (parallelization):
     a. Apply feature engineering
     b. Predict with each model (specific horizon)
  4. Combine predictions (per 30min interval)
  5. Reconcile hierarchically
  6. Save results
```

#### 7.3 Backtest Workflow
```r
BacktestWorkflow:
  1. Define period (start_date, end_date)
  2. For each day in period:
     a. Simulate forecast D+0→D+8
     b. Compare with actual
     c. Calculate metrics
     d. Detect drift
  3. For each retraining interval [1,3,7,10,15]:
     a. Simulate periodic retraining
     b. Compare performance
  4. Generate comparative report
  5. Recommend optimal interval
```

**Modules:**
```r
orchestrator/
├── workflows/
│   ├── train.R          # TrainWorkflow
│   ├── predict.R        # PredictWorkflow
│   └── backtest.R       # BacktestWorkflow
├── parallel.R           # ParallelExecutor (future)
├── config.R             # ConfigManager (load/validate YAML)
└── logger.R             # StructuredLogger (JSON logs)
```

---

### 8. CLI Layer (`R/cli/`)

**Responsibilities:**
- Command-line interface (readline/optparse)
- Argument mode (pipelines) and interactive mode (dev)
- Input validation
- Progress bars and feedback

**Main Commands:**

```bash
# === TRAINING ===
prevcarga train \
  --config config.yaml \
  --storage-backend s3 \
  --areas SECO_RJ,SECO_SP \
  --models lgbm,rf \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --version 1.2.0 \
  --parallel 8

# === TRAINING (Local Development) ===
prevcarga train \
  --config config_dev.yaml \
  --storage-backend local \
  --areas SECO_RJ \
  --models lgbm \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --version 0.1.0-dev

# === PREDICTION ===
prevcarga predict \
  --config config.yaml \
  --storage-backend s3 \
  --date 2025-01-17 \
  --mode intraday \
  --areas all \
  --models all \
  --reconcile \
  --output results/20250117/

# === BACKTEST ===
prevcarga backtest \
  --config config.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --retrain-intervals 1,3,7,10,15 \
  --walk-forward \
  --models all \
  --report report.html

# === FEATURE ENGINEERING ===
prevcarga gen-features \
  --config config.yaml \
  --model lgbm \
  --areas SECO_RJ \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --save-s3

prevcarga eval-features \
  --config config.yaml \
  --model lgbm \
  --areas SECO_RJ \
  --methods shap,correlation,stability

# === EVALUATION ===
prevcarga eval-model \
  --model-path s3://bucket/models/lgbm/latest \
  --test-data s3://bucket/test/2024Q4.parquet \
  --metrics all \
  --by-patamar

# === COMBINATION ===
prevcarga combine \
  --predictions-path s3://bucket/results/20250117/individual/ \
  --strategy markov \
  --output-path s3://bucket/results/20250117/combined/ \
  --train-weights

# === REPORT GENERATION (Interactive HTML with highcharter) ===
prevcarga report forecast \
  --date 2025-01-17 \
  --areas RJ,SP \
  --output reports/forecast_20250117.html

prevcarga report backtest \
  --backtest-id bt_2024Q4 \
  --output reports/backtest_2024Q4.html

prevcarga report compare \
  --models lgbm,rf,hw \
  --period 2024-10-01:2024-12-31 \
  --output reports/comparison_2024Q4.html

prevcarga report dashboard \
  --area RJ \
  --period 2024-01-01:2024-12-31 \
  --output reports/dashboard_RJ_2024.html

prevcarga report drift \
  --model lgbm \
  --baseline-date 2024-06-01 \
  --output reports/drift_lgbm.html \
  --open

# === INTERACTIVE MODE ===
prevcarga interactive
> Choose command: [train/predict/backtest/eval]
> ...
```

**Interactive Shell (based on demo):**
```r
PrevCargaShell <- R6::R6Class(
  "PrevCargaShell",
  private = list(
    config = NULL,
    storage = NULL,
    running = FALSE,

    # ANSI color codes
    colors = list(
      reset = "\033[0m",
      bold = "\033[1m",
      red = "\033[31m",
      green = "\033[32m",
      yellow = "\033[33m",
      cyan = "\033[36m"
    ),

    # Unicode symbols
    symbols = list(
      check = "\u2713",
      cross = "\u2717",
      arrow = "\u2192",
      bullet = "\u2022",
      bar_full = "\u2588",
      bar_empty = "\u2591"
    )
  ),
  public = list(
    run = function() {
      private$running <- TRUE
      self$print_banner()

      while (private$running) {
        self$print_prompt()
        input <- readline()
        self$execute_command(input)
      }
    },

    print_banner = function() {
      cat("\n")
      cat(private$colors$cyan, "  PrevCargaONS v1.0.0\n", private$colors$reset)
      cat("  Electric Load Forecasting System\n")
      cat("  Type 'help' for available commands\n\n")
    },

    print_prompt = function() {
      cat(private$colors$green, "prevcarga", private$colors$cyan, "> ",
          private$colors$reset, sep = "")
    },

    progress_bar = function(current, total, width = 40) {
      pct <- current / total
      filled <- floor(pct * width)
      bar <- paste0(
        "[",
        strrep(private$symbols$bar_full, filled),
        strrep(private$symbols$bar_empty, width - filled),
        "]"
      )
      sprintf("%s %3.0f%%", bar, pct * 100)
    },

    sparkline = function(values) {
      chars <- c("\u2581", "\u2582", "\u2583", "\u2584",
                 "\u2585", "\u2586", "\u2587", "\u2588")
      rng <- range(values, na.rm = TRUE)
      if (rng[1] == rng[2]) return(strrep(chars[4], length(values)))
      scaled <- (values - rng[1]) / (rng[2] - rng[1])
      indices <- pmin(floor(scaled * 7) + 1, 8)
      paste(chars[indices], collapse = "")
    }
  )
)
```

**Modules:**
```r
cli/
├── main.R               # Main entry point
├── commands/
│   ├── train.R
│   ├── predict.R
│   ├── backtest.R
│   ├── features.R
│   ├── evaluate.R
│   ├── combine.R
│   └── report.R         # Interactive HTML report generation
├── shell.R              # InteractiveCLI
├── display.R            # ANSI, progress, sparklines
└── validators.R         # InputValidators
```

---

## Storage Backend (R6 Abstract)

```r
#' @title StorageBackend
#' @description Abstract base class for storage backends
StorageBackend <- R6::R6Class(
  "StorageBackend",
  public = list(
    backend_type = NULL,

    read_parquet = function(path) {
      stop("read_parquet() must be implemented by subclass")
    },

    write_parquet = function(dt, path) {
      stop("write_parquet() must be implemented by subclass")
    },

    read_rds = function(path) {
      stop("read_rds() must be implemented by subclass")
    },

    write_rds = function(obj, path) {
      stop("write_rds() must be implemented by subclass")
    },

    exists = function(path) {
      stop("exists() must be implemented by subclass")
    },

    list_files = function(path, pattern = NULL) {
      stop("list_files() must be implemented by subclass")
    },

    delete = function(path) {
      stop("delete() must be implemented by subclass")
    },

    mkdir = function(path) {
      stop("mkdir() must be implemented by subclass")
    }
  )
)

#' @title LocalStorageBackend
LocalStorageBackend <- R6::R6Class(
  "LocalStorageBackend",
  inherit = StorageBackend,
  private = list(
    base_path = NULL
  ),
  public = list(
    initialize = function(base_path = "./data") {
      self$backend_type <- "local"
      private$base_path <- normalizePath(base_path, mustWork = FALSE)
      if (!dir.exists(private$base_path)) {
        dir.create(private$base_path, recursive = TRUE)
      }
    },

    read_parquet = function(path) {
      full_path <- file.path(private$base_path, path)
      arrow::read_parquet(full_path) |> data.table::setDT()
    },

    write_parquet = function(dt, path) {
      full_path <- file.path(private$base_path, path)
      dir.create(dirname(full_path), recursive = TRUE, showWarnings = FALSE)
      arrow::write_parquet(dt, full_path)
    },

    read_rds = function(path) {
      full_path <- file.path(private$base_path, path)
      readRDS(full_path)
    },

    write_rds = function(obj, path) {
      full_path <- file.path(private$base_path, path)
      dir.create(dirname(full_path), recursive = TRUE, showWarnings = FALSE)
      saveRDS(obj, full_path)
    },

    exists = function(path) {
      file.exists(file.path(private$base_path, path))
    },

    list_files = function(path, pattern = NULL) {
      full_path <- file.path(private$base_path, path)
      list.files(full_path, pattern = pattern, full.names = FALSE)
    },

    delete = function(path) {
      full_path <- file.path(private$base_path, path)
      unlink(full_path, recursive = TRUE)
    },

    mkdir = function(path) {
      full_path <- file.path(private$base_path, path)
      dir.create(full_path, recursive = TRUE, showWarnings = FALSE)
    }
  )
)

#' @title S3StorageBackend
S3StorageBackend <- R6::R6Class(
  "S3StorageBackend",
  inherit = StorageBackend,
  private = list(
    bucket = NULL,
    prefix = NULL,
    s3_client = NULL
  ),
  public = list(
    initialize = function(bucket, prefix = "", region = "us-east-1") {
      self$backend_type <- "s3"
      private$bucket <- bucket
      private$prefix <- prefix
      private$s3_client <- paws::s3(config = list(region = region))
    },

    read_parquet = function(path) {
      full_key <- file.path(private$prefix, path)
      tmp_file <- tempfile(fileext = ".parquet")
      on.exit(unlink(tmp_file))

      private$s3_client$download_file(
        Bucket = private$bucket,
        Key = full_key,
        Filename = tmp_file
      )
      arrow::read_parquet(tmp_file) |> data.table::setDT()
    },

    write_parquet = function(dt, path) {
      full_key <- file.path(private$prefix, path)
      tmp_file <- tempfile(fileext = ".parquet")
      on.exit(unlink(tmp_file))

      arrow::write_parquet(dt, tmp_file)
      private$s3_client$upload_file(
        Filename = tmp_file,
        Bucket = private$bucket,
        Key = full_key
      )
    },

    exists = function(path) {
      full_key <- file.path(private$prefix, path)
      tryCatch({
        private$s3_client$head_object(
          Bucket = private$bucket,
          Key = full_key
        )
        TRUE
      }, error = function(e) FALSE)
    }
  )
)

#' @title StorageFactory
StorageFactory <- R6::R6Class(
  "StorageFactory",
  public = list(
    from_config = function(config) {
      backend_type <- config$storage$backend %||% "local"

      if (backend_type == "s3") {
        s3_config <- config$storage$s3
        S3StorageBackend$new(
          bucket = s3_config$bucket,
          prefix = s3_config$prefix %||% "",
          region = s3_config$region %||% "us-east-1"
        )
      } else {
        local_config <- config$storage$local
        LocalStorageBackend$new(
          base_path = local_config$base_path %||% "./data"
        )
      }
    },

    create = function(type, ...) {
      if (type == "s3") {
        S3StorageBackend$new(...)
      } else if (type == "local") {
        LocalStorageBackend$new(...)
      } else {
        stop(sprintf("Unknown backend type: %s", type))
      }
    }
  )
)
```

---

## Configuration (config.yaml)

```yaml
# === GENERAL ===
project:
  name: "PrevCargaONS"
  version: "1.0.0"
  seed: 42

# === STORAGE ===
storage:
  backend: local  # Options: 's3' or 'local'

  paths:
    raw_data: raw_data/
    features: features/
    models: models/
    results: results/

  s3:
    bucket: prevcarga-bucket
    region: us-east-1
    endpoint_url: null
    connect_timeout: 60
    read_timeout: 300
    max_retries: 3

  local:
    base_path: ./data
    create_dirs: true

# === REGIONAL HIERARCHY ===
regions:
  # National level (reconciliation only, no models)
  national: SIN                      # Use SIN as area_code

  # Subsystems (reconciliation only, no models - aggregate from areas)
  subsystems:
    SECO: [RJ, SP, MG, ES, MT, MS, AC, RO, DF, GO, PESE]
    S: [PR, SC, RS, PES]
    NE: [ALPE, PBRN, BASE, CE, PI, BAOE, PENE]
    N: [AM, PA, MA, TO, RR, AP, PEN]

  # Areas where models CAN run (21 areas total)
  areas:
    # SECO areas
    - RJ
    - SP
    - MG
    - ES
    - MT
    - MS
    - AC
    - RO
    - DF
    - GO
    # S areas
    - PR
    - SC
    - RS
    # NE areas
    - ALPE
    - PBRN
    - BASE
    - CE
    - PI
    - BAOE
    # N areas
    - AM
    - PA
    - MA
    - TO
    - RR
    - AP
    # Loss areas (can have models OR be calculated by difference)
    - PESE
    - PES
    - PENE
    - PEN

  # Loss areas configuration (models OR by_difference)
  loss_areas:
    PESE: {subsystem: SECO, method: by_difference}  # or method: model
    PES: {subsystem: S, method: by_difference}
    PENE: {subsystem: NE, method: by_difference}
    PEN: {subsystem: N, method: by_difference}

# === TIME PERIODS ===
patamares:
  SECO:
    Jan: {ponta_inicio: "17:30", ponta_fim: "20:30"}
    Fev: {ponta_inicio: "17:30", ponta_fim: "20:30"}
    # ... other months

# === MODELS ===
models:
  # Models are registered via plugin system
  # Configuration per model
  default_hyperparameters:
    seed: 42
    validation_split: 0.15
    test_split: 0.15

# === COMBINATION ===
combination:
  default_strategy: markov_chain

  strategies:
    voting:
      type: weighted
      weight_method: inverse_error

    stacking:
      meta_model: lgbm
      meta_features: [predictions, hora, dia_semana, temperatura]

    markov_chain:
      history_window: 30
      state_features: [intervalo_30min, dia_semana, hora, feriado]
      transition_smoothing: 0.1

  bias_correction:
    enabled: true
    features: [hora, dia_semana, temperatura, modelo_usado]

# === RECONCILIATION ===
reconciliation:
  method: mint  # mint | ols | wls

  wls_weights:
    type: variance
    window: 90

# === TRAINING ===
training:
  historical_window: 1095  # 3 years in days
  validation_split: 0.15
  test_split: 0.15

  retrain_evaluation:
    intervals: [1, 3, 7, 10, 15]
    method: walk_forward

  parallel:
    areas: true
    models: true
    n_jobs: 8

# === EVALUATION ===
evaluation:
  metrics:
    - mape
    - mae
    - rmse
    - percentiles: [5, 25, 50, 75, 95]
    - max_deviation_abs
    - max_deviation_rel

  by_patamar: true
  by_horizon: true

  drift_detection:
    enabled: true
    baseline_window: 30
    threshold_mape: 1.2
    threshold_mae: 1.15

# === REPORTS (Interactive HTML with highcharter) ===
reports:
  theme: "prevcarga"  # Custom highcharter theme

  defaults:
    width: 900
    height: 500
    export_buttons: true  # Enable PNG/SVG/PDF export

  colors:
    primary: "#1f77b4"
    secondary: "#ff7f0e"
    error: "#d62728"
    success: "#2ca02c"
    forecast: "#17becf"
    actual: "#7f7f7f"

  charts:
    forecast:
      show_confidence: true
      confidence_levels: [80, 95]
    metrics:
      show_baseline: true
    comparison:
      sort_by: mape

  templates:
    output_dir: reports/
    open_after_generation: false

# === LOGGING ===
logging:
  level: INFO
  format: json
  handlers:
    - console
    - file: logs/prevcarga.log
```

---

## Development Phases

### **PHASE 0: Initial Setup** (1 week)
- [ ] Initialize R package structure (`usethis::create_package()`)
- [ ] Set up renv for dependency management
- [ ] Configure roxygen2 for documentation
- [ ] Create DESCRIPTION with dependencies:
  ```
  Imports:
    R6,
    data.table,
    arrow,
    yaml,
    paws,
    future,
    future.apply,
    cli,
    optparse,
    checkmate,
    jsonlite,
    highcharter,
    rmarkdown,
    htmlwidgets
  Suggests:
    testthat (>= 3.0.0),
    mockery,
    covr
  ```
- [ ] Create StorageBackend R6 classes (base, local, S3)
- [ ] Create StorageFactory
- [ ] Set up structured logging
- [ ] Create test infrastructure (testthat)

### **PHASE 1: Data Layer** (2 weeks)
- [ ] Create DataLoader R6 class using arrow for parquet
- [ ] Implement schema validation with checkmate
- [ ] Create preprocessing pipeline:
  - Missing value imputation (locf, linear interpolation)
  - Outlier detection (IQR, k-means MO1)
  - Resampling (hourly to half-hourly)
- [ ] Create DataCatalog for dataset registry
- [ ] Port existing `R/data/loader.R` to new architecture
- [ ] Implement deck_mdl format compatibility
- [ ] Tests for both S3 and local backends

### **PHASE 2: Feature Engineering** (3 weeks)
- [ ] Create BaseFeaturePlugin R6 class
- [ ] Create FeaturePluginRegistry
- [ ] Implement common features plugin:
  - Lag features (D-1, D-7, same hour)
  - Calendar dummies (day of week, month, holiday)
  - Cyclical encoding (hour_sin, hour_cos)
- [ ] Implement wavelet plugin
- [ ] Implement LOESS smoothing plugin
- [ ] Implement BLF strategy plugin
- [ ] Implement RF feature selector (per-lead-hour)
- [ ] Create FeaturePipeline composer
- [ ] Feature importance evaluation
- [ ] Tests + benchmarks

### **PHASE 3: Model Layer - Part 1** (4 weeks)
- [ ] Create BaseModel R6 class
- [ ] Create ModelRegistry
- [ ] Implement semantic versioning
- [ ] Model serialization/deserialization
- [ ] Model versioning with metadata
- [ ] Tests + validation

### **PHASE 4: Model Layer - Part 2** (3 weeks)
- [ ] Hierarchical Pipeline (DM→Profile):
  - ARIMA demand mean (forecast package)
  - Profile models (48 models)
  - DM to Profile composition
- [ ] Port existing models from `R/models/`
- [ ] Integration tests

### **PHASE 5: Combination Layer** (3 weeks)
- [ ] Create BaseCombiner R6 class
- [ ] Create CombinerRegistry
- [ ] Voting combiners (simple, weighted, inverse_error)
- [ ] Stacking combiner (meta-model)
- [ ] Markov chain combiner:
  - Transition matrix
  - Dynamic weights
- [ ] Bias corrector
- [ ] Weight optimizer
- [ ] Tests + validation

### **PHASE 6: Reconciliation Layer** (2 weeks)
- [ ] Create BaseReconciler R6 class
- [ ] Define hierarchy from YAML
- [ ] MinT reconciler
- [ ] OLS reconciler
- [ ] WLS reconciler (variance/error weights)
- [ ] Loss calculator
- [ ] Numerical tests

### **PHASE 7: Evaluation Layer** (2 weeks)
- [ ] Base metrics (MAPE, MAE, RMSE)
- [ ] Percentiles and deviations
- [ ] Metrics by patamar (peak/off-peak)
- [ ] Drift detector
- [ ] Model comparator
- [ ] HTML reporter (rmarkdown)
- [ ] CSV reporter
- [ ] Highcharter reporter (interactive HTML reports)
- [ ] Report templates (forecast, backtest, compare, dashboard, drift)
- [ ] Tests + examples

### **PHASE 8: Orchestrator** (3 weeks)
- [ ] ConfigManager (YAML loading with validation)
- [ ] TrainWorkflow
- [ ] PredictWorkflow (batch + intraday)
- [ ] BacktestWorkflow
- [ ] ParallelExecutor using future
- [ ] Structured logger
- [ ] Integration tests

### **PHASE 9: CLI** (2 weeks)
- [ ] Interactive shell implementation (based on demo)
- [ ] Command-line interface with optparse
- [ ] Commands: train, predict, backtest, forecast
- [ ] Commands: gen-features, eval-features
- [ ] Commands: eval-model, eval-combination, combine
- [ ] Command: report (forecast, backtest, compare, dashboard, drift)
- [ ] Progress bars, colors, sparklines
- [ ] Input validators
- [ ] Complete help texts
- [ ] E2E tests

### **PHASE 10: Testing & Validation** (3 weeks)
- [ ] Reproduce existing R model results
- [ ] Complete 2024 backtest (1 year)
- [ ] Model vs legacy comparison
- [ ] Combination evaluation
- [ ] Reconciliation validation
- [ ] Performance benchmarks

### **PHASE 11: One-Line Installer** (1 week)
- [ ] Create installer script structure (`scripts/install.sh`)
- [ ] Implement OS detection (Ubuntu, Debian, Fedora, CentOS, macOS)
- [ ] Implement automatic R installation per platform
- [ ] Create directory structure (`~/.prevcarga/`)
- [ ] Base64 embed and extract R shell script
- [ ] Create launcher script (`~/.prevcarga/bin/prevcarga`)
- [ ] Configure renv environment with locked dependencies
- [ ] Configure user PATH (bash/zsh)
- [ ] Create default configuration
- [ ] Create build script for installer (`scripts/build_installer.sh`)
- [ ] Write installer tests (OS detection, R check, PATH config)
- [ ] Integration test on Docker containers

**Installation Command:**
```bash
curl -fsSL https://ons.org.br/prevcarga/install.sh | bash
```

### **PHASE 12: Documentation & Deploy** (2 weeks)
- [ ] Complete README
- [ ] Sphinx documentation site with ReadTheDocs theme
- [ ] Installation guides: Docker, WSL, curl installer
- [ ] Documentation pages for each component
- [ ] Dockerfile and docker-compose
- [ ] CI/CD with GitHub Actions

---

## Milestones and Deliverables

| Milestone | Week | Deliverable |
|-----------|------|-------------|
| M1: MVP Data + Features | 6 | Data pipeline + common features working |
| M2: First Model | 8 | Model training and predicting via plugin |
| M3: All Core Models | 15 | Multiple models working independently |
| M4: Combination and Reconciliation | 20 | Complete system (without CLI) |
| M5: Functional CLI | 23 | CLI with all commands |
| M6: Complete Validation | 26 | Results validated vs baseline |
| M7: One-Line Installer | 28 | curl installer for easy deployment |
| M8: Production | 30 | Docker deploy + Sphinx documentation |

**Total Estimated: ~7.5 months (30 weeks)**

---

## Testing Strategy

### Test Pyramid
```
          /\
         /E2E\        10% - End-to-End (CLI commands)
        /------\
       /  INT   \     30% - Integration (workflows)
      /----------\
     /   UNIT     \   60% - Unit (functions, R6 classes)
    /--------------\
```

### Minimum Coverage
- **Unit**: 80%
- **Integration**: 70%
- **E2E**: Critical cases (happy path + common errors)

### Test Framework
```r
# tests/testthat.R
library(testthat)
library(prevcargaons)

test_check("prevcargaons")
```

### Unit Test Examples
```r
# tests/testthat/test-storage-backend.R
test_that("LocalStorageBackend reads and writes parquet", {
  backend <- LocalStorageBackend$new(tempdir())

  dt <- data.table::data.table(
    date = as.Date("2024-01-01") + 0:9,
    value = rnorm(10)
  )

  backend$write_parquet(dt, "test/data.parquet")
  expect_true(backend$exists("test/data.parquet"))

  dt_read <- backend$read_parquet("test/data.parquet")
  expect_equal(nrow(dt_read), 10)

  backend$delete("test")
})

# tests/testthat/test-model-base.R
test_that("Model registry creates instances", {
  # Register a test model
  TestModel <- R6::R6Class("TestModel", inherit = BaseModel)
  .model_registry$register("test", TestModel)

  model <- .model_registry$create("test")
  expect_s3_class(model, "BaseModel")
})
```

### Tools
- `testthat` (framework)
- `covr` (coverage)
- `mockery` (mocking)

---

## Plugin Development Guide

For detailed guides on creating plugins for this system, see the **Plugin Guide** documentation:

| Guide | Description |
|-------|-------------|
| [`docs/mvp/plugin-guide/00-overview.md`](plugin-guide/00-overview.md) | Plugin architecture overview, R6 classes, registry pattern |
| [`docs/mvp/plugin-guide/01-data-importing.md`](plugin-guide/01-data-importing.md) | Data importing, storage backends, Hive partitioning |
| [`docs/mvp/plugin-guide/02-feature-engineering.md`](plugin-guide/02-feature-engineering.md) | Feature engineering plugins, complexity tiers |
| [`docs/mvp/plugin-guide/03-model-training.md`](plugin-guide/03-model-training.md) | Model training plugins, custom objectives, multi-model patterns |
| [`docs/mvp/plugin-guide/04-model-inference.md`](plugin-guide/04-model-inference.md) | Inference plugins, normalization, post-processing |

> **Note:** These guides will evolve into the project's CONTRIBUTING documentation.

---

## Containerization

### Dockerfile
```dockerfile
FROM rocker/r-ver:4.3.0

# System dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN R -e "install.packages(c( \
    'R6', \
    'data.table', \
    'arrow', \
    'yaml', \
    'paws', \
    'future', \
    'future.apply', \
    'cli', \
    'optparse', \
    'checkmate', \
    'jsonlite', \
    'forecast' \
), repos='https://cloud.r-project.org/')"

# Working directory
WORKDIR /app

# Copy package
COPY . /app/

# Install package
RUN R CMD INSTALL .

# Entry point
ENTRYPOINT ["Rscript", "-e", "prevcargaons::main()"]
```

---

## Reproducibility

### Best Practices
- [ ] Fixed seeds in all random components
- [ ] Model versioning (semantic + timestamp)
- [ ] Configuration registry used (metadata.yaml)
- [ ] Dependency freeze (renv.lock)
- [ ] Tagged Docker images

---

## Final Acceptance Checklist

### Core Functionalities
- [ ] Training of models via plugin system
- [ ] Batch prediction (D+0 to D+8)
- [ ] Intraday prediction (30min)
- [ ] Combination (Voting, Stacking, Markov)
- [ ] Reconciliation (MinT, OLS, WLS)
- [ ] Backtest with retraining evaluation
- [ ] Pluggable feature engineering
- [ ] Complete CLI (10+ commands)

### Performance
- [ ] Training 1 area + 1 model: <30min
- [ ] Intraday prediction: <5min
- [ ] 1-year backtest: <8h
- [ ] Parallelization working

### Quality
- [ ] Test coverage >70%
- [ ] Complete documentation
- [ ] Reproduces baseline results ±5% MAPE
- [ ] Drift detection working
- [ ] Structured logs

### Deploy
- [ ] Functional Dockerfile
- [ ] docker-compose tested locally
- [ ] One-line installer tested
- [ ] CI/CD pipeline working

---

## Next Steps After M8

1. **Performance Optimizations**
   - GPU acceleration for ML models
   - Dask-like parallelization
   - Redis cache for features

2. **New Models**
   - Transformers (temporal attention)
   - N-BEATS
   - Prophet

3. **Advanced Features**
   - Solar/wind generation data
   - Additional meteorological variables
   - Custom special events

4. **Combination Improvements**
   - AutoML for meta-model
   - Advanced ensemble learning
   - Hierarchical combination

5. **Operationalization**
   - REST API (plumber)
   - Real-time dashboard (shiny)
   - Alert system
   - Automatic retraining

---

**END OF R CLI MACRO PLAN**
