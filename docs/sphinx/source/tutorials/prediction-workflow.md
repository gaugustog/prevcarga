# Prediction Workflow Tutorial

A comprehensive tutorial on generating and managing load forecasts with PrevCarga.

## What You'll Learn

In this tutorial, you will:

1. Understand prediction modes and horizons
2. Generate single and batch predictions
3. Use ensemble methods for better accuracy
4. Apply hierarchical reconciliation
5. Implement intraday BLF strategy
6. Evaluate prediction quality
7. Automate prediction workflows

**Estimated time**: 45 minutes

---

## Prerequisites

- Completed the [Getting Started Tutorial](getting-started.md)
- Trained models available for your target areas
- Understanding of forecasting horizons

---

## Prediction Pipeline Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Load        │ --> │ Feature      │ --> │ Model       │
│ Model       │     │ Generation   │     │ Inference   │
└─────────────┘     └──────────────┘     └─────────────┘
                                               │
                                               v
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Save        │ <-- │ Post-        │ <-- │ Hierarchical│
│ Results     │     │ Processing   │     │ Reconcile   │
└─────────────┘     └──────────────┘     └─────────────┘
```

---

## Step 1: Understanding Prediction Modes

### Forecast Horizons

PrevCarga supports multiple forecast horizons:

| Horizon | Description | Update Frequency |
|---------|-------------|------------------|
| D+0 | Intraday (same day) | Every 30 minutes |
| D+1 | Next day | Daily at 00:00 |
| D+2 to D+8 | Medium-term | Daily |

### Prediction Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `batch` | Standard batch prediction | Daily operational forecasts |
| `intraday` | Real-time with BLF | Operational intraday updates |
| `backtest` | Historical evaluation | Model performance analysis |

---

## Step 2: Basic Predictions

### Single Model, Single Area

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas SECO_RJ \
    --models lgbm
```

Output:
```
Generating Predictions
======================
Configuration: config/config.yaml
Date: 2024-01-15
Area: SECO_RJ
Model: lgbm

Loading model: lgbm/SECO_RJ
Generating features...
Making predictions...

Predictions Generated
=====================
Records: 48 (half-hourly)
Output: data/predictions/2024-01-15/SECO_RJ_lgbm.parquet

Summary Statistics:
  Mean Load: 12,456.78 MWh
  Min Load: 9,234.56 MWh (04:00)
  Max Load: 15,678.90 MWh (19:30)
```

### Multiple Areas

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas SECO_RJ SECO_SP SECO_MG S_PR \
    --models lgbm \
    --parallel 4
```

### All Areas

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas all \
    --models lgbm
```

---

## Step 3: Ensemble Predictions

### Multiple Models

Combine predictions from multiple models:

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas SECO_RJ \
    --models lgbm random_forest regdin_svm holt_winters \
    --combiner weighted_voting
```

### Available Combination Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| `simple_average` | Equal weight average | Quick baseline |
| `weighted_voting` | Performance-weighted | Production use |
| `best_model` | Select single best | Conservative approach |
| `stacking` | Meta-model combination | Maximum accuracy |

### Weighted Voting Configuration

```yaml
# config/ensemble.yaml
combination:
  method: weighted_voting
  weights:
    lgbm: 0.35
    random_forest: 0.25
    regdin_svm: 0.25
    holt_winters: 0.15

  # Or use automatic weights based on validation performance
  auto_weights:
    enabled: true
    metric: mape
    lookback_days: 30
```

### Stacking Ensemble

```yaml
combination:
  method: stacking
  meta_model: lgbm
  meta_features:
    - base_predictions
    - hour
    - day_of_week
    - temperature
```

---

## Step 4: Hierarchical Reconciliation

Ensure predictions are consistent across the hierarchy:

```
National (SIN) = Sum of all Subsystems
Subsystem = Sum of Areas + Losses
```

### Enable Reconciliation

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas all \
    --models lgbm \
    --reconcile
```

### Reconciliation Methods

| Method | Description | Accuracy |
|--------|-------------|----------|
| `bottom_up` | Sum from areas | Fast, simple |
| `ols` | OLS regression | Good balance |
| `wls` | Weighted least squares | Better accuracy |
| `mint` | Minimum trace | Best accuracy |

### Configuration

```yaml
# config/reconciliation.yaml
reconciliation:
  enabled: true
  method: mint

  hierarchy:
    - level: national
      name: SIN
      children: [SECO, S, NE, N]

    - level: subsystem
      name: SECO
      children: [SECO_RJ, SECO_SP, SECO_MG, SECO_ES]

    - level: subsystem
      name: S
      children: [S_PR, S_SC, S_RS]

    # ... more subsystems

  # Reconciliation options
  options:
    preserve_area_forecasts: false
    min_adjustment_threshold: 0.01
```

### View Reconciliation Results

```bash
prevcarga predict \
    --config config/reconciliation.yaml \
    --date 2024-01-15 \
    --areas all \
    --models lgbm \
    --reconcile \
    --verbose
```

Output:
```
Hierarchical Reconciliation
===========================
Method: MinT (Minimum Trace)

Pre-Reconciliation:
  Sum of areas: 45,678.90 MWh
  National forecast: 45,234.56 MWh
  Difference: 444.34 MWh (0.98%)

Post-Reconciliation:
  National total: 45,456.73 MWh
  Coherent: Yes

Area Adjustments:
  SECO_RJ: -12.34 MWh (-0.10%)
  SECO_SP: +8.56 MWh (+0.05%)
  ...
```

---

## Step 5: Intraday Predictions (BLF Strategy)

For operational intraday forecasting, use the Baseline Load Forecast (BLF) strategy.

### BLF Concept

```
Time      | Strategy
----------|--------------------
00:00     | Use D+0 model forecast
06:00     | Blend actual (00-06) + forecast (06-24)
12:00     | Blend actual (00-12) + forecast (12-24)
18:00     | Blend actual (00-18) + forecast (18-24)
```

### Intraday Prediction

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --mode intraday \
    --areas SECO_RJ \
    --models lgbm \
    --blf-strategy auto
```

### BLF Strategies

| Strategy | Description |
|----------|-------------|
| `auto` | Automatically select based on current time |
| `morning` | Use actuals 00:00-06:00 |
| `midday` | Use actuals 00:00-12:00 |
| `afternoon` | Use actuals 00:00-18:00 |
| `evening` | Use actuals 00:00-21:00 |

### BLF Configuration

```yaml
# config/intraday.yaml
intraday:
  blf:
    enabled: true
    strategy: auto
    correction_method: ratio  # or 'difference'

    # Time windows for each strategy
    windows:
      morning: "06:00"
      midday: "12:00"
      afternoon: "18:00"
      evening: "21:00"

    # Smoothing for transition
    transition_smoothing:
      enabled: true
      window: 2  # hours
```

### Scheduled Intraday Updates

Run predictions every 30 minutes:

```bash
# crontab entry
*/30 * * * * prevcarga predict --config config/intraday.yaml --mode intraday --areas all
```

---

## Step 6: Batch Predictions

### Date Range Predictions

Generate predictions for multiple dates:

```bash
prevcarga predict batch \
    --config config/config.yaml \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --areas all \
    --models all \
    --reconcile \
    --parallel 8
```

### Batch Configuration

```yaml
# config/batch_predictions.yaml
batch:
  dates:
    start: "2024-01-01"
    end: "2024-01-31"

  models:
    - lgbm
    - random_forest

  areas:
    - SECO_RJ
    - SECO_SP
    - S_PR

  options:
    reconcile: true
    parallel: 8
    save_intermediate: true

  output:
    format: parquet
    path: ./predictions/batch/
    partitioning:
      - date
      - area
```

Execute:

```bash
prevcarga predict batch --config config/batch_predictions.yaml
```

---

## Step 7: Prediction Evaluation

### Compare Predictions to Actuals

```bash
prevcarga evaluate \
    --config config/config.yaml \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --areas SECO_RJ \
    --models lgbm \
    --metrics mape mae rmse \
    --output evaluation.json
```

Output:
```
Prediction Evaluation
=====================
Area: SECO_RJ
Model: lgbm
Period: 2024-01-01 to 2024-01-31

Overall Metrics:
  MAPE: 3.45%
  MAE: 42.31 MWh
  RMSE: 58.92 MWh
  Bias: -2.34 MWh

By Horizon:
  D+0: MAPE 2.12%
  D+1: MAPE 3.45%
  D+2: MAPE 4.23%

By Time Period:
  Peak (18-21h): MAPE 4.56%
  Off-peak: MAPE 2.89%
  Night (00-06h): MAPE 2.34%

By Day Type:
  Weekday: MAPE 3.23%
  Weekend: MAPE 4.12%
```

### Generate Evaluation Report

```bash
prevcarga evaluate \
    --config config/config.yaml \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --areas all \
    --models all \
    --report evaluation_report.html
```

### Percentile Analysis

```bash
prevcarga evaluate \
    --config config/config.yaml \
    --areas SECO_RJ \
    --models lgbm \
    --percentiles 5 25 50 75 95 \
    --output percentiles.json
```

Output:
```
Percentile Analysis
===================
Error Distribution (MWh):
  P5:  -89.23 (5% of errors below this)
  P25: -34.56
  P50: -2.34 (median)
  P75: +31.23
  P95: +78.90 (5% of errors above this)

Interquartile Range: 65.79 MWh
```

---

## Step 8: Output Formats and Storage

### Available Output Formats

| Format | Option | Use Case |
|--------|--------|----------|
| Parquet | `--output-format parquet` | Efficient storage, analytics |
| CSV | `--output-format csv` | Human-readable, spreadsheets |
| JSON | `--output-format json` | APIs, web applications |

### Custom Output Path

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas SECO_RJ \
    --models lgbm \
    --output-path ./custom/predictions/
```

### S3 Output

```bash
prevcarga predict \
    --config config/config.yaml \
    --date 2024-01-15 \
    --areas SECO_RJ \
    --models lgbm \
    --storage-backend s3 \
    --output-path s3://prevcarga-predictions/2024-01-15/
```

---

## Step 9: Python API Integration

### Basic Prediction

```python
from src.orchestration import PredictionWorkflow
from src.config import SystemConfig

# Load configuration
config = SystemConfig.from_yaml("config/config.yaml")

# Initialize workflow
workflow = PredictionWorkflow(config)

# Generate predictions
predictions = workflow.predict(
    date="2024-01-15",
    areas=["SECO_RJ", "SECO_SP"],
    models=["lgbm"],
    reconcile=True
)

# Access results
for area, df in predictions.items():
    print(f"{area}: {len(df)} predictions")
    print(df.head())
```

### Ensemble Prediction

```python
from src.orchestration import PredictionWorkflow
from src.combination import WeightedVoting

# Create combiner
combiner = WeightedVoting(
    weights={"lgbm": 0.4, "random_forest": 0.3, "regdin_svm": 0.3}
)

# Generate ensemble predictions
predictions = workflow.predict(
    date="2024-01-15",
    areas=["SECO_RJ"],
    models=["lgbm", "random_forest", "regdin_svm"],
    combiner=combiner
)
```

### Streaming Predictions

```python
from src.orchestration import PredictionWorkflow

workflow = PredictionWorkflow(config)

# Stream predictions for large date ranges
for date_predictions in workflow.predict_stream(
    start_date="2024-01-01",
    end_date="2024-12-31",
    areas=["SECO_RJ"],
    models=["lgbm"],
    batch_size=7  # days per batch
):
    # Process batch
    save_to_database(date_predictions)
    send_to_api(date_predictions)
```

---

## Automation Examples

### Daily Operational Workflow

```bash
#!/bin/bash
# daily_forecast.sh

DATE=$(date +%Y-%m-%d)
CONFIG=/etc/prevcarga/config.yaml

echo "Starting daily forecast for $DATE"

# Generate predictions for all areas
prevcarga predict \
    --config $CONFIG \
    --date $DATE \
    --areas all \
    --models all \
    --reconcile \
    --output-path /data/predictions/$DATE/

# Evaluate previous day's predictions
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
prevcarga evaluate \
    --config $CONFIG \
    --date $YESTERDAY \
    --areas all \
    --models all \
    --output /data/evaluations/$YESTERDAY.json

echo "Daily forecast complete"
```

### Intraday Scheduler

```python
"""Intraday prediction scheduler."""
import schedule
import subprocess
from datetime import datetime

def run_intraday_prediction():
    """Run intraday prediction."""
    date = datetime.now().strftime("%Y-%m-%d")
    cmd = [
        "prevcarga", "predict",
        "--config", "/etc/prevcarga/config.yaml",
        "--date", date,
        "--mode", "intraday",
        "--areas", "all",
        "--blf-strategy", "auto"
    ]
    subprocess.run(cmd, check=True)
    print(f"Intraday prediction completed at {datetime.now()}")

# Schedule every 30 minutes
schedule.every(30).minutes.do(run_intraday_prediction)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60)
```

---

## Best Practices

### Prediction Quality

1. **Use appropriate models** for each horizon
2. **Apply reconciliation** for hierarchical consistency
3. **Monitor prediction errors** daily
4. **Retrain models** when performance degrades

### Operational Reliability

1. **Implement health checks** for prediction services
2. **Set up alerts** for failed predictions
3. **Maintain backup predictions** for system failures
4. **Log all prediction operations**

### Performance Optimization

1. **Cache features** for repeated predictions
2. **Parallelize** multi-area predictions
3. **Use efficient storage** formats (Parquet)
4. **Batch predictions** when possible

---

## Troubleshooting

### Model Not Found

```bash
# Check available models
prevcarga status

# Train missing model
prevcarga train --areas SECO_RJ --models lgbm ...
```

### Reconciliation Fails

```bash
# Check hierarchy configuration
prevcarga config validate --config config/reconciliation.yaml

# Verify all required areas are predicted
prevcarga predict --areas all ...
```

### Poor Prediction Accuracy

1. Check data quality: `prevcarga data validate`
2. Evaluate recent model performance
3. Consider retraining with recent data
4. Review feature engineering configuration

---

## Next Steps

- **[Usage Guide](../guides/usage.md)**: Complete CLI reference
- **[API Reference](../api/index.rst)**: Programmatic usage
- **[Troubleshooting](../guides/troubleshooting.md)**: Common issues

---

**Need help?** Check the [Troubleshooting Guide](../guides/troubleshooting.md)
