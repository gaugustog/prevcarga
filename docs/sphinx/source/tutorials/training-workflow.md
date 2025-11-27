# Training Workflow Tutorial

A comprehensive tutorial on training forecasting models with PrevCarga.

## What You'll Learn

In this tutorial, you will:

1. Understand the training pipeline architecture
2. Configure training parameters
3. Train multiple model types
4. Implement cross-validation
5. Optimize hyperparameters
6. Monitor training progress
7. Evaluate and compare models

**Estimated time**: 45 minutes

---

## Prerequisites

- Completed the [Getting Started Tutorial](getting-started.md)
- Historical load data available
- Understanding of basic ML concepts

---

## Training Pipeline Overview

The PrevCarga training pipeline consists of several stages:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Data        │ --> │ Feature      │ --> │ Model       │
│ Loading     │     │ Engineering  │     │ Training    │
└─────────────┘     └──────────────┘     └─────────────┘
                                               │
                                               v
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Model       │ <-- │ Validation   │ <-- │ Cross-      │
│ Saving      │     │ Metrics      │     │ Validation  │
└─────────────┘     └──────────────┘     └─────────────┘
```

---

## Step 1: Understanding Model Types

PrevCarga supports multiple model architectures:

### End-to-End Models

Direct forecasting from features to predictions:

| Model | Description | Best For |
|-------|-------------|----------|
| `lgbm` | LightGBM gradient boosting | General purpose, fast training |
| `random_forest` | Random Forest ensemble | Robust, interpretable |

### Hierarchical Models

Decompose forecast into daily mean and hourly profile:

| Model | Description | Best For |
|-------|-------------|----------|
| `regdin_svm` | ARIMA + SVM for profile | Capturing trend + profile |
| `holt_winters` | ETS + HW decomposition | Strong seasonality |

---

## Step 2: Basic Training Configuration

### Minimal Training

```bash
prevcarga train \
    --config config/config.yaml \
    --areas SECO_RJ \
    --models lgbm \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

### Training Configuration File

Create `config/training_advanced.yaml`:

```yaml
# Storage configuration
storage:
  backend: local
  local:
    base_path: ./data

# Model-specific configurations
models:
  lgbm:
    # Model hyperparameters
    n_estimators: 1000
    learning_rate: 0.05
    max_depth: 10
    num_leaves: 31
    min_child_samples: 20
    subsample: 0.8
    colsample_bytree: 0.8

    # Training options
    early_stopping_rounds: 50
    verbose: -1

  random_forest:
    n_estimators: 200
    max_depth: 15
    min_samples_split: 5
    min_samples_leaf: 2
    n_jobs: -1

  regdin_svm:
    arima_order: [2, 1, 2]
    seasonal_order: [1, 1, 1, 48]
    svm_kernel: rbf
    svm_C: 1.0

  holt_winters:
    seasonal_periods: 48
    trend: add
    seasonal: add
    damped_trend: true

# Feature configuration
features:
  plugins:
    - temporal
    - calendar
    - lag
    - cyclical_encoding

  lag:
    lags: [1, 2, 24, 48, 168, 336]
    rolling_windows: [24, 48, 168]
    statistics: [mean, std, min, max]

# Training configuration
training:
  validation_split: 0.2
  shuffle: false  # Time series - don't shuffle
  parallel_workers: 4

  # Cross-validation
  cross_validation:
    enabled: false
    n_splits: 5
    method: time_series  # Time series split

# Logging
logging:
  level: INFO
  format: json
```

---

## Step 3: Training Multiple Models

### Train All Models for One Area

```bash
prevcarga train \
    --config config/training_advanced.yaml \
    --areas SECO_RJ \
    --models all \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --parallel 4
```

### Train One Model for All Areas

```bash
prevcarga train \
    --config config/training_advanced.yaml \
    --areas all \
    --models lgbm \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --parallel 8
```

### Train All Models for All Areas

```bash
prevcarga train \
    --config config/training_advanced.yaml \
    --areas all \
    --models all \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --parallel 8
```

This will train: `4 models × 17 areas = 68 model instances`

---

## Step 4: Cross-Validation

### Enable Cross-Validation

Update configuration:

```yaml
training:
  cross_validation:
    enabled: true
    n_splits: 5
    method: time_series
```

### Run Training with CV

```bash
prevcarga train \
    --config config/training_advanced.yaml \
    --areas SECO_RJ \
    --models lgbm \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --cross-validate
```

Output:
```
Cross-Validation Results
========================
Model: lgbm
Area: SECO_RJ
Folds: 5

Fold Results:
  Fold 1: MAPE 3.21%, MAE 41.23, RMSE 55.67
  Fold 2: MAPE 3.45%, MAE 43.56, RMSE 58.12
  Fold 3: MAPE 3.12%, MAE 40.89, RMSE 54.34
  Fold 4: MAPE 3.67%, MAE 45.12, RMSE 60.45
  Fold 5: MAPE 3.34%, MAE 42.78, RMSE 57.23

Average Metrics:
  MAPE: 3.36% (±0.21%)
  MAE: 42.72 (±1.67)
  RMSE: 57.16 (±2.34)
```

### Time Series Cross-Validation Strategy

```
Training Data     |  Validation
=====================================
Fold 1: [Jan-Jun] | [Jul-Aug]
Fold 2: [Jan-Aug] | [Sep-Oct]
Fold 3: [Jan-Oct] | [Nov-Dec]
...
```

---

## Step 5: Hyperparameter Tuning

### Manual Hyperparameter Search

Create multiple configurations and compare:

```yaml
# config/lgbm_v1.yaml
models:
  lgbm:
    n_estimators: 500
    learning_rate: 0.1
    max_depth: 8

# config/lgbm_v2.yaml
models:
  lgbm:
    n_estimators: 1000
    learning_rate: 0.05
    max_depth: 10

# config/lgbm_v3.yaml
models:
  lgbm:
    n_estimators: 2000
    learning_rate: 0.01
    max_depth: 12
```

Run and compare:

```bash
# Train each configuration
prevcarga train --config config/lgbm_v1.yaml --areas SECO_RJ --models lgbm ...
prevcarga train --config config/lgbm_v2.yaml --areas SECO_RJ --models lgbm ...
prevcarga train --config config/lgbm_v3.yaml --areas SECO_RJ --models lgbm ...

# Compare results
prevcarga compare \
    --models lgbm_v1 lgbm_v2 lgbm_v3 \
    --areas SECO_RJ \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

### Python API for Grid Search

```python
from src.orchestration import TrainingWorkflow
from src.config import TrainingConfig
from itertools import product

# Define parameter grid
param_grid = {
    'n_estimators': [500, 1000, 2000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [8, 10, 12]
}

results = []

for n_est, lr, depth in product(*param_grid.values()):
    config = TrainingConfig(
        model_params={
            'n_estimators': n_est,
            'learning_rate': lr,
            'max_depth': depth
        }
    )

    workflow = TrainingWorkflow(config)
    metrics = workflow.train_with_cv(
        model='lgbm',
        area='SECO_RJ',
        start_date='2023-01-01',
        end_date='2023-12-31',
        n_splits=3
    )

    results.append({
        'params': {'n_estimators': n_est, 'learning_rate': lr, 'max_depth': depth},
        'mape': metrics['mape'],
        'mae': metrics['mae']
    })

# Find best configuration
best = min(results, key=lambda x: x['mape'])
print(f"Best params: {best['params']}")
print(f"Best MAPE: {best['mape']:.2f}%")
```

---

## Step 6: Monitoring Training Progress

### Real-Time Progress

Training displays real-time progress:

```
Training Models
===============
Model: lgbm | Area: SECO_RJ | Period: 2023-01-01 to 2023-12-31

Stage: Feature Engineering
  [████████████████████] 100% | Features generated: 156

Stage: Model Training
  [████████████░░░░░░░░] 60% | Iteration: 600/1000
  Current metrics:
    Train RMSE: 45.23
    Valid RMSE: 52.67
    Best iteration: 543

Elapsed: 5m 23s | ETA: 3m 45s
```

### Logging Training Metrics

Enable detailed logging:

```yaml
logging:
  level: DEBUG
  training_metrics: true
  output: file
  file_path: logs/training.log
```

View training logs:

```bash
# Real-time log monitoring
tail -f logs/training.log | jq 'select(.event == "training_iteration")'

# Extract final metrics
cat logs/training.log | jq 'select(.event == "training_complete")' | jq '.metrics'
```

### Training Callbacks (Python API)

```python
from src.orchestration import TrainingWorkflow

def on_iteration(iteration, metrics):
    """Called after each training iteration."""
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: RMSE = {metrics['rmse']:.2f}")

def on_early_stop(iteration, best_iteration):
    """Called when early stopping triggers."""
    print(f"Early stopping at iteration {iteration}")
    print(f"Best iteration: {best_iteration}")

workflow = TrainingWorkflow(config)
workflow.train(
    model='lgbm',
    area='SECO_RJ',
    callbacks={
        'on_iteration': on_iteration,
        'on_early_stop': on_early_stop
    }
)
```

---

## Step 7: Model Evaluation and Comparison

### Evaluate Single Model

```bash
prevcarga evaluate \
    --config config/config.yaml \
    --areas SECO_RJ \
    --models lgbm \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --output eval_lgbm.json
```

### Compare Multiple Models

```bash
prevcarga compare \
    --config config/config.yaml \
    --areas SECO_RJ \
    --models lgbm random_forest regdin_svm holt_winters \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --report comparison_report.html
```

Output:
```
Model Comparison Report
=======================
Area: SECO_RJ
Period: 2024-01-01 to 2024-01-31

Performance Summary:
┌───────────────┬─────────┬──────────┬──────────┐
│ Model         │ MAPE    │ MAE      │ RMSE     │
├───────────────┼─────────┼──────────┼──────────┤
│ lgbm          │ 3.45%   │ 42.31    │ 58.92    │
│ random_forest │ 3.67%   │ 45.12    │ 62.34    │
│ regdin_svm    │ 3.89%   │ 47.56    │ 65.78    │
│ holt_winters  │ 4.12%   │ 50.23    │ 69.45    │
└───────────────┴─────────┴──────────┴──────────┘

Best Model: lgbm (MAPE: 3.45%)

Detailed report saved to: comparison_report.html
```

---

## Step 8: Saving and Loading Models

### Model Storage Structure

```
data/models/
├── lgbm/
│   └── SECO_RJ/
│       ├── model.pkl
│       ├── config.yaml
│       ├── metadata.json
│       └── feature_importance.json
├── random_forest/
│   └── SECO_RJ/
│       ├── model.pkl
│       └── ...
└── ...
```

### Load and Use Model (Python API)

```python
from src.models import ModelRegistry
from src.storage import StorageFactory

# Initialize storage
storage = StorageFactory.create("local", base_path="./data")

# Load model
model = ModelRegistry.load(
    storage=storage,
    model_type="lgbm",
    area="SECO_RJ"
)

# View model metadata
print(f"Model: {model.model_type}")
print(f"Trained: {model.metadata['trained_at']}")
print(f"Training MAPE: {model.metadata['metrics']['mape']:.2f}%")

# Make predictions
predictions = model.predict(features)
```

### Export Model for Deployment

```bash
prevcarga model export \
    --model lgbm \
    --area SECO_RJ \
    --output ./exports/lgbm_SECO_RJ.tar.gz
```

---

## Best Practices

### Data Quality

1. **Minimum 12 months** of historical data for seasonal patterns
2. **Check for missing values** before training
3. **Handle outliers** appropriately
4. **Verify data consistency** across areas

### Training Configuration

1. **Use early stopping** to prevent overfitting
2. **Reserve validation data** (20% recommended)
3. **Don't shuffle time series data**
4. **Monitor training metrics** for anomalies

### Resource Management

1. **Parallel training** for multiple areas/models
2. **Batch training** during off-peak hours
3. **Monitor memory usage** for large datasets
4. **Use incremental training** when possible

### Model Management

1. **Version your models** with timestamps
2. **Document hyperparameters** used
3. **Keep training logs** for reproducibility
4. **Regular retraining** (weekly recommended)

---

## Troubleshooting

### Training Runs Out of Memory

```bash
# Reduce parallel workers
prevcarga train --parallel 1 ...

# Or reduce data batch size in config
training:
  batch_size: 10000
```

### Training Takes Too Long

```bash
# Use fewer estimators for initial experiments
models:
  lgbm:
    n_estimators: 200  # Reduce from 1000

# Enable early stopping
    early_stopping_rounds: 30
```

### Poor Model Performance

1. Check data quality: `prevcarga data validate --area SECO_RJ`
2. Review feature importance
3. Try different model types
4. Adjust hyperparameters
5. Increase training data

---

## Next Steps

- **[Prediction Workflow Tutorial](prediction-workflow.md)**: Generate and evaluate predictions
- **[Usage Guide](../guides/usage.md)**: Complete CLI reference
- **[API Reference](../api/index.rst)**: Programmatic usage

---

**Need help?** Check the [Troubleshooting Guide](../guides/troubleshooting.md)
