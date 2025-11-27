# Usage Guide

Complete guide to using PrevCarga for electric load forecasting.

## Table of Contents

- [CLI Overview](#cli-overview)
- [Training Models](#training-models)
- [Generating Predictions](#generating-predictions)
- [Feature Engineering](#feature-engineering)
- [Data Management](#data-management)
- [Configuration](#configuration)
- [Monitoring and Logging](#monitoring-and-logging)
- [Best Practices](#best-practices)

---

## CLI Overview

PrevCarga provides a comprehensive command-line interface for all forecasting operations.

### Basic Command Structure

```bash
prevcarga [COMMAND] [SUBCOMMAND] [OPTIONS]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `predict` | Generate load forecasts |
| `train` | Train forecasting models |
| `backtest` | Run historical backtests |
| `evaluate` | Calculate performance metrics |
| `compare` | Compare model performance |
| `data` | Data management operations |
| `feature` | Feature engineering |
| `config` | Configuration management |
| `status` | System status and health |

### Getting Help

```bash
# General help
prevcarga --help

# Command-specific help
prevcarga predict --help
prevcarga train --help

# Subcommand help
prevcarga train model --help
```

### Global Options

| Option | Description |
|--------|-------------|
| `--config` | Path to configuration file |
| `--storage-backend` | Storage backend (local, s3) |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `--dry-run` | Preview operations without executing |
| `--version` | Show version information |

---

## Training Models

### Single Model Training

Train a single model for a specific area and date range:

```bash
prevcarga train \
    --config config/config.yaml \
    --storage-backend local \
    --areas SECO_RJ \
    --models lgbm \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--models` | Model type(s): lgbm, random_forest, regdin_svm, holt_winters, all | lgbm |
| `--areas` | Target area(s) or "all" | Required |
| `--start-date` | Training period start (YYYY-MM-DD) | Required |
| `--end-date` | Training period end (YYYY-MM-DD) | Required |
| `--parallel` | Number of parallel workers | 4 |
| `--force` | Force retrain existing models | false |

### Training Multiple Areas

```bash
prevcarga train \
    --config config/config.yaml \
    --areas SECO_RJ \
    --areas SECO_SP \
    --areas SECO_MG \
    --models lgbm \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --parallel 4
```

### Training All Models for All Areas

```bash
prevcarga train \
    --config config/config.yaml \
    --areas all \
    --models all \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --parallel 8
```

### Batch Training from Configuration

Create a training configuration file:

```yaml
# training_config.yaml
training:
  models:
    - lgbm
    - random_forest
    - regdin_svm

  areas:
    - SECO_RJ
    - SECO_SP
    - S_PR

  date_range:
    start: "2023-01-01"
    end: "2023-12-31"

  options:
    parallel: 8
    force_retrain: false
    validation_split: 0.2
```

Execute batch training:

```bash
prevcarga train batch --config training_config.yaml
```

### Dry Run (Preview Training Plan)

```bash
prevcarga train \
    --areas SECO_RJ \
    --models all \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --dry-run
```

Output:
```
Training Plan (Dry Run)
=======================
Models to train: 4
  - lgbm
  - random_forest
  - regdin_svm
  - holt_winters

Areas: 1
  - SECO_RJ

Date Range: 2023-01-01 to 2023-12-31 (365 days)

Estimated time: ~45 minutes (with 4 parallel workers)
Estimated storage: ~250 MB

No changes made (dry run mode).
```

---

## Generating Predictions

### Single Model Prediction

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas SECO_RJ \
    --models lgbm
```

### Multi-Model Prediction

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas SECO_RJ \
    --models lgbm \
    --models random_forest \
    --models regdin_svm
```

### All Models with Reconciliation

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas all \
    --models all \
    --reconcile
```

### Intraday Prediction (BLF Strategy)

For operational intraday forecasting with Baseline Load Forecast:

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --mode intraday \
    --areas SECO_RJ \
    --blf-strategy morning
```

**BLF Strategies:**
- `morning`: Use morning actual values to adjust forecast
- `midday`: Use values up to midday
- `afternoon`: Use values up to afternoon
- `evening`: Use values up to evening

### Batch Predictions

Create prediction configuration:

```yaml
# predictions_config.yaml
predictions:
  models:
    - lgbm
    - random_forest

  areas:
    - SECO_RJ
    - SECO_SP

  dates:
    start: "2024-01-01"
    end: "2024-01-31"

  options:
    reconcile: true
    parallel: 4

  output:
    format: parquet
    path: ./predictions/
```

Execute:

```bash
prevcarga predict batch --config predictions_config.yaml
```

### Prediction Output Formats

| Format | Option | Description |
|--------|--------|-------------|
| Parquet | `--output-format parquet` | Efficient columnar storage |
| CSV | `--output-format csv` | Human-readable |
| JSON | `--output-format json` | API-friendly |

---

## Feature Engineering

### Build Feature Pipeline

```bash
prevcarga feature build \
    --config config/config.yaml \
    --areas SECO_RJ \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

### List Available Plugins

```bash
prevcarga feature list-plugins
```

Output:
```
Available Feature Plugins:
==========================
Plugin                  | Description                           | Status
------------------------|---------------------------------------|--------
temporal_features       | Basic temporal features (hour, day)   | enabled
calendar_features       | Calendar and holiday features         | enabled
lag_features           | Lag and rolling window features        | enabled
cyclical_encoding      | Cyclical encoding for periodic data    | enabled
loess_smoothing        | LOESS signal smoothing                 | enabled
wavelet_transform      | Wavelet decomposition features         | enabled
blf_strategy           | Baseline Load Forecast features        | enabled
seasonality_features   | Seasonal decomposition features        | enabled
```

### Configure Feature Pipeline

```yaml
# feature_config.yaml
feature_pipeline:
  plugins:
    - name: temporal_features
      enabled: true

    - name: calendar_features
      enabled: true
      params:
        include_holidays: true
        country: BR

    - name: lag_features
      enabled: true
      params:
        lags: [1, 2, 24, 48, 168]
        rolling_windows: [24, 48, 168]
        statistics: [mean, std, min, max]

    - name: cyclical_encoding
      enabled: true
      params:
        columns:
          - hour
          - day_of_week
          - month

    - name: loess_smoothing
      enabled: false
      params:
        frac: 0.1
```

Apply configuration:

```bash
prevcarga feature build \
    --config feature_config.yaml \
    --areas SECO_RJ \
    --start-date 2024-01-01
```

### Feature Importance Analysis

```bash
prevcarga feature importance \
    --model lgbm \
    --areas SECO_RJ \
    --top 20
```

Output:
```
Feature Importance (Top 20)
===========================
Rank | Feature                    | Importance
-----|----------------------------|------------
1    | load_lag_1                 | 0.1523
2    | load_lag_24                | 0.1245
3    | hour_sin                   | 0.0892
4    | temperature                | 0.0756
5    | load_rolling_mean_24       | 0.0698
...
```

---

## Data Management

### Validate Data Quality

```bash
prevcarga data validate \
    --config config/config.yaml \
    --areas SECO_RJ \
    --date 2024-01-15
```

Output:
```
Data Validation Report
======================
Area: SECO_RJ
Date: 2024-01-15

Checks:
  [PASS] Data completeness: 100% (48/48 half-hours)
  [PASS] Value range: All values within expected bounds
  [PASS] No missing values
  [PASS] No duplicate timestamps
  [WARN] Outliers detected: 2 values flagged

Overall: PASS (with warnings)
```

### Check Data Availability

```bash
prevcarga data check \
    --config config/config.yaml \
    --areas SECO_RJ \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

### Show Data Statistics

```bash
prevcarga data stats \
    --config config/config.yaml \
    --areas SECO_RJ \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

Output:
```
Data Statistics: SECO_RJ
========================
Period: 2024-01-01 to 2024-01-31 (31 days)

Load (MWh):
  Mean:    12,456.78
  Std:     1,234.56
  Min:     8,901.23
  Max:     15,678.90

Completeness: 99.8% (1,485/1,488 records)
Missing: 3 records (2024-01-15 03:00, 03:30, 04:00)
```

---

## Configuration

### View Current Configuration

```bash
prevcarga config show
```

### Validate Configuration

```bash
prevcarga config validate --config config/config.yaml
```

### Configuration File Structure

```yaml
# config/config.yaml

# Storage configuration
storage:
  backend: s3  # or 'local'
  s3:
    bucket: prevcarga-data-production
    region: us-east-1
    prefix: prevcarga/
  local:
    base_path: ./data

# Logging configuration
logging:
  level: INFO
  format: json
  output: stdout

# Model configuration
models:
  default: lgbm
  enabled:
    - lgbm
    - random_forest
    - regdin_svm
    - holt_winters

  lgbm:
    n_estimators: 1000
    learning_rate: 0.05
    max_depth: 10
    num_leaves: 31
    early_stopping_rounds: 50

  random_forest:
    n_estimators: 200
    max_depth: 15
    n_jobs: -1

# Feature configuration
features:
  plugins:
    - temporal
    - calendar
    - lag
    - cyclical_encoding

  cache_enabled: true
  parallel_processing: true

# Training configuration
training:
  validation_split: 0.2
  cross_validation_folds: 5
  parallel_workers: 4

# Prediction configuration
prediction:
  default_horizon: 48  # half-hours
  reconciliation_method: mint
  output_format: parquet
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `wJalr...` |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `PREVCARGA_CONFIG` | Config file path | `/etc/prevcarga/config.yaml` |
| `PREVCARGA_LOG_LEVEL` | Log level | `INFO` |
| `PREVCARGA_STORAGE_BACKEND` | Storage backend | `s3` |

---

## Monitoring and Logging

### Check System Status

```bash
prevcarga status
```

Output:
```
PrevCarga System Status
=======================
Version: 1.0.0
Python: 3.11.7
Platform: Linux

Storage:
  Backend: s3
  Bucket: prevcarga-data-production
  Access: Connected

Models Available:
  lgbm: 17 areas trained
  random_forest: 17 areas trained
  regdin_svm: 17 areas trained
  holt_winters: 17 areas trained

System Health: OK
Last Updated: 2024-01-15 10:30:00 UTC
```

### Log Configuration

Configure structured JSON logging:

```yaml
logging:
  level: INFO
  format: json
  output: file
  file_path: /var/log/prevcarga/prevcarga.log
  rotation:
    max_size: 100MB
    backup_count: 10
```

### View and Filter Logs

```bash
# View real-time logs
tail -f /var/log/prevcarga/prevcarga.log

# Parse JSON logs
cat prevcarga.log | jq .

# Filter by level
cat prevcarga.log | jq 'select(.level == "ERROR")'

# Filter by operation
cat prevcarga.log | jq 'select(.operation == "predict")'
```

---

## Best Practices

### Model Training

1. **Use Sufficient Data**: Minimum 12 months of historical data for seasonal patterns
2. **Regular Retraining**: Retrain models weekly or when performance degrades
3. **Leverage Parallelism**: Use `--parallel` option for multi-core systems
4. **Monitor Metrics**: Track MAPE, MAE, and RMSE during training
5. **Version Models**: Use date-based naming for model artifacts

### Predictions

1. **Use Ensemble Methods**: Combine multiple models for robustness
2. **Apply Reconciliation**: Ensure hierarchical consistency with `--reconcile`
3. **Validate Inputs**: Check data quality before prediction
4. **Monitor Confidence**: Review prediction intervals
5. **Cache Results**: Reuse predictions when possible

### Production Deployment

1. **Containerization**: Use Docker for consistent environments
2. **Configuration Management**: Externalize all configuration
3. **Structured Logging**: Use JSON format for log aggregation
4. **Health Monitoring**: Implement health checks and alerts
5. **Automated Workflows**: Schedule training and prediction jobs

### Performance Optimization

1. **Parallel Processing**: Use multiple workers for training/prediction
2. **Memory Management**: Monitor memory usage, reduce batch sizes if needed
3. **Data Caching**: Enable feature caching for repeated operations
4. **Storage Optimization**: Use Parquet format for efficient I/O
5. **Resource Allocation**: Match worker count to available CPU cores

---

## Advanced Usage

### Python API Integration

```python
from src.models import LightGBMModel
from src.features import FeatureRegistry
from src.storage import StorageFactory

# Initialize storage
storage = StorageFactory.create("local", base_path="./data")

# Load trained model
model = LightGBMModel.load(storage, area="SECO_RJ")

# Generate features
registry = FeatureRegistry()
features = registry.transform(input_data)

# Make prediction
predictions = model.predict(features)
```

### Custom Scripts

```python
from src.orchestration import TrainingWorkflow, PredictionWorkflow
from src.config import SystemConfig

# Load configuration
config = SystemConfig.from_yaml("config/config.yaml")

# Training workflow
training = TrainingWorkflow(config)
results = training.run(
    models=["lgbm", "random_forest"],
    areas=["SECO_RJ"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Prediction workflow
prediction = PredictionWorkflow(config)
forecasts = prediction.run(
    models=["lgbm"],
    areas=["SECO_RJ"],
    date="2024-01-15"
)
```

### Scheduled Workflows

Example cron jobs for automated operations:

```bash
# Daily training update (every Sunday at 2 AM)
0 2 * * 0 prevcarga train --config /etc/prevcarga/config.yaml --areas all --models all

# Intraday predictions (every 30 minutes)
*/30 * * * * prevcarga predict --config /etc/prevcarga/config.yaml --mode intraday --areas all

# Daily performance evaluation (every day at 6 AM)
0 6 * * * prevcarga evaluate --config /etc/prevcarga/config.yaml --date yesterday
```

---

## Next Steps

- **[Installation Guide](installation.md)**: Setup instructions
- **[Quick Start](quick-start.md)**: Get started in 5 minutes
- **[API Reference](../api/index.rst)**: Programmatic usage
- **[Tutorials](../tutorials/index.rst)**: Step-by-step tutorials
- **[Troubleshooting](troubleshooting.md)**: Common issues

---

**Need Help?** Check [Troubleshooting](troubleshooting.md) or open an issue on [GitHub](https://github.com/gaugustog/prevcarga/issues)
