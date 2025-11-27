# Quick Start Guide

Get up and running with PrevCarga in 5 minutes.

## Prerequisites

Before you begin, ensure you have:

- Python 3.11+ installed
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials configured (for S3 backend)

```{seealso}
See the [Installation Guide](installation.md) for detailed setup instructions.
```

---

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/gaugustog/prevcarga.git
cd prevcarga

# Install dependencies
uv sync

# Verify installation
prevcarga --version
```

---

## 2. Configuration

### Local Development (No AWS Required)

Create a minimal configuration for local development:

```yaml
# config/config_dev.yaml
storage:
  backend: local
  local:
    base_path: ./data

logging:
  level: INFO
  format: json

models:
  default: lgbm
  enabled:
    - lgbm
    - random_forest

features:
  plugins:
    - temporal
    - calendar
    - lag
```

### Production (S3 Backend)

```yaml
# config/config.yaml
storage:
  backend: s3
  s3:
    bucket: your-prevcarga-bucket
    region: us-east-1

logging:
  level: INFO
  format: json

models:
  default: lgbm
  enabled:
    - lgbm
    - random_forest
    - regdin_svm
    - holt_winters
```

---

## 3. Your First Prediction

### Generate a Forecast

```bash
# Single area prediction
prevcarga predict \
    --config config/config_dev.yaml \
    --storage-backend local \
    --date 2024-01-15 \
    --areas SECO_RJ \
    --models lgbm
```

### Batch Prediction (Multiple Areas)

```bash
# All areas prediction
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas all \
    --models all \
    --reconcile
```

### Intraday Prediction

```bash
# Intraday mode with BLF strategy
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --mode intraday \
    --areas SECO_RJ \
    --blf-strategy morning
```

---

## 4. Training a Model

### Train Single Model

```bash
# Train LightGBM for one area
prevcarga train \
    --config config/config_dev.yaml \
    --storage-backend local \
    --areas SECO_RJ \
    --models lgbm \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

### Batch Training

```bash
# Train all models for all areas (parallel)
prevcarga train \
    --config config/config.yaml \
    --areas all \
    --models all \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --parallel 4
```

---

## 5. Running a Backtest

```bash
# Backtest with monthly retraining
prevcarga backtest \
    --config config/config.yaml \
    --start-date 2024-01-01 \
    --end-date 2024-06-30 \
    --areas SECO_RJ \
    --models lgbm \
    --retrain-interval 30 \
    --report backtest_report.html
```

---

## 6. Evaluating Performance

### Generate Metrics Report

```bash
# Evaluate model performance
prevcarga evaluate \
    --config config/config.yaml \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --areas SECO_RJ \
    --metrics mape,mae,rmse \
    --output metrics.json
```

### Compare Models

```bash
# Compare multiple models
prevcarga compare \
    --config config/config.yaml \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --areas SECO_RJ \
    --models lgbm,random_forest,regdin_svm \
    --report comparison.html
```

---

## Common CLI Commands Reference

| Command | Description |
|---------|-------------|
| `prevcarga predict` | Generate load forecasts |
| `prevcarga train` | Train forecasting models |
| `prevcarga backtest` | Run historical backtests |
| `prevcarga evaluate` | Calculate performance metrics |
| `prevcarga compare` | Compare model performance |
| `prevcarga config show` | Display current configuration |
| `prevcarga data validate` | Validate input data quality |

Get help for any command:

```bash
prevcarga <command> --help
```

---

## Python API Quick Start

### Basic Prediction

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
print(predictions)
```

### Feature Engineering Pipeline

```python
from src.features import FeaturePipelineComposer
from src.features.plugins import (
    TemporalFeaturePlugin,
    CalendarFeaturePlugin,
    LagFeaturePlugin,
)

# Create pipeline
composer = FeaturePipelineComposer()
composer.add_plugin(TemporalFeaturePlugin())
composer.add_plugin(CalendarFeaturePlugin())
composer.add_plugin(LagFeaturePlugin(lags=[1, 2, 24, 48]))

# Transform data
df_features = composer.fit_transform(df_raw)
print(f"Generated {len(composer.get_feature_names())} features")
```

### Model Training

```python
from src.models import LightGBMModel
from src.training import TrainingWorkflow

# Configure training
workflow = TrainingWorkflow(
    model_class=LightGBMModel,
    config={
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 10,
    }
)

# Train model
model = workflow.train(
    X_train=features,
    y_train=targets,
    validation_data=(X_val, y_val),
)

# Save model
model.save(storage, area="SECO_RJ")
```

---

## Example Workflows

### Daily Operational Forecast

```bash
#!/bin/bash
# daily_forecast.sh - Run daily operational forecasts

DATE=$(date +%Y-%m-%d)

# Generate predictions for all areas
prevcarga predict \
    --config config/config.yaml \
    --date $DATE \
    --areas all \
    --models all \
    --reconcile \
    --output predictions_${DATE}.parquet

# Evaluate against actuals (when available)
prevcarga evaluate \
    --config config/config.yaml \
    --date $(date -d "yesterday" +%Y-%m-%d) \
    --output daily_metrics.json
```

### Weekly Model Retraining

```bash
#!/bin/bash
# weekly_retrain.sh - Retrain models weekly

END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "1 year ago" +%Y-%m-%d)

# Retrain all models
prevcarga train \
    --config config/config.yaml \
    --areas all \
    --models all \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --parallel 4

# Validate new models
prevcarga evaluate \
    --config config/config.yaml \
    --start-date $(date -d "1 month ago" +%Y-%m-%d) \
    --end-date $END_DATE \
    --output validation_metrics.json
```

---

## Next Steps

Now that you've run your first predictions:

1. **[Installation Guide](installation.md)**: Detailed setup for different environments
2. **[API Reference](../api/index.rst)**: Integrate PrevCarga into your applications
3. **[Architecture Overview](../architecture/index.rst)**: Understand the system design
4. **[Tutorials](../tutorials/index.rst)**: Step-by-step tutorials
5. **[Troubleshooting](troubleshooting.md)**: Solutions for common issues

---

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [GitHub Issues](https://github.com/gaugustog/prevcarga/issues)
3. Ask in [GitHub Discussions](https://github.com/gaugustog/prevcarga/discussions)

---

**Ready for more?** Check out the [API Reference](../api/index.rst) for comprehensive documentation.
