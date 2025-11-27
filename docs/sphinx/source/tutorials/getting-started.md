# Getting Started Tutorial

A step-by-step tutorial to get you up and running with PrevCarga.

## What You'll Learn

In this tutorial, you will:

1. Set up your development environment
2. Configure PrevCarga for local development
3. Train your first forecasting model
4. Generate predictions
5. Evaluate model performance

**Estimated time**: 30 minutes

---

## Prerequisites

Before starting, ensure you have:

- Python 3.11+ installed
- [uv](https://github.com/astral-sh/uv) package manager
- Git installed
- 8GB+ RAM available

---

## Step 1: Installation

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/gaugustog/prevcarga.git
cd prevcarga

# Install dependencies
uv sync

# Verify installation
prevcarga --version
```

Expected output:
```
PrevCarga v1.0.0
```

### Verify Setup

```bash
prevcarga --help
```

You should see a list of available commands.

---

## Step 2: Configuration

### Create Local Configuration

Create a configuration file for local development:

```bash
mkdir -p config
```

Create `config/config_dev.yaml`:

```yaml
# Development configuration
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
    - cyclical_encoding

training:
  validation_split: 0.2
  parallel_workers: 2
```

### Create Data Directory

```bash
mkdir -p data/raw data/models data/predictions
```

---

## Step 3: Prepare Sample Data

For this tutorial, we'll create sample data. In a real scenario, you would load actual historical load data.

### Create Sample Data Script

Create `scripts/create_sample_data.py`:

```python
"""Create sample data for the getting started tutorial."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def create_sample_load_data(
    area: str,
    start_date: str,
    end_date: str,
    output_dir: Path
) -> None:
    """Generate synthetic load data for testing."""

    # Create date range (half-hourly)
    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq='30min'
    )

    # Base load pattern (daily seasonality)
    hours = dates.hour + dates.minute / 60
    daily_pattern = 1000 + 500 * np.sin(2 * np.pi * (hours - 6) / 24)

    # Weekly pattern
    weekday = dates.dayofweek
    weekly_factor = np.where(weekday < 5, 1.1, 0.85)

    # Add trend and noise
    trend = np.linspace(0, 100, len(dates))
    noise = np.random.normal(0, 50, len(dates))

    # Combine patterns
    load = daily_pattern * weekly_factor + trend + noise
    load = np.maximum(load, 500)  # Minimum load

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'area': area,
        'load_mwh': load.round(2),
        'temperature': 20 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, len(dates))
    })

    # Save to parquet
    output_path = output_dir / f"{area}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Created sample data: {output_path}")
    print(f"  Records: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

if __name__ == "__main__":
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data for one area
    create_sample_load_data(
        area="SECO_RJ",
        start_date="2023-01-01",
        end_date="2024-01-31",
        output_dir=output_dir
    )
```

### Generate Sample Data

```bash
python scripts/create_sample_data.py
```

Expected output:
```
Created sample data: data/raw/SECO_RJ.parquet
  Records: 26352
  Date range: 2023-01-01 00:00:00 to 2024-01-31 23:30:00
```

---

## Step 4: Train Your First Model

Now let's train a LightGBM model on the sample data.

### Run Training

```bash
prevcarga train \
    --config config/config_dev.yaml \
    --storage-backend local \
    --areas SECO_RJ \
    --models lgbm \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

Expected output:
```
Training Models
===============
Configuration: config/config_dev.yaml
Storage: local

Training lgbm for SECO_RJ...
  Loading data: 2023-01-01 to 2023-12-31
  Generating features...
  Training model...
  [████████████████████] 100%

Training Complete
=================
Model: lgbm
Area: SECO_RJ
Duration: 2m 15s

Validation Metrics:
  MAPE: 3.45%
  MAE: 42.31 MWh
  RMSE: 58.92 MWh
  R²: 0.94

Model saved to: data/models/lgbm/SECO_RJ/
```

### Verify Model

```bash
ls -la data/models/lgbm/SECO_RJ/
```

You should see the saved model files.

---

## Step 5: Generate Predictions

### Single Day Prediction

```bash
prevcarga predict \
    --config config/config_dev.yaml \
    --storage-backend local \
    --date 2024-01-15 \
    --areas SECO_RJ \
    --models lgbm
```

Expected output:
```
Generating Predictions
======================
Model: lgbm
Area: SECO_RJ
Date: 2024-01-15

Loading model...
Generating features...
Making predictions...
  [████████████████████] 100%

Predictions Generated
=====================
Area: SECO_RJ
Date: 2024-01-15
Horizon: 48 half-hours

Summary:
  Mean predicted load: 1,234.56 MWh
  Min: 892.34 MWh
  Max: 1,567.89 MWh

Output: data/predictions/2024-01-15/SECO_RJ_lgbm.parquet
```

### View Predictions

```python
import pandas as pd

# Load predictions
df = pd.read_parquet("data/predictions/2024-01-15/SECO_RJ_lgbm.parquet")
print(df.head(10))
```

Output:
```
             timestamp     area    model  predicted_load
0  2024-01-15 00:00:00  SECO_RJ     lgbm        1045.23
1  2024-01-15 00:30:00  SECO_RJ     lgbm        1023.45
2  2024-01-15 01:00:00  SECO_RJ     lgbm         998.67
3  2024-01-15 01:30:00  SECO_RJ     lgbm         967.89
4  2024-01-15 02:00:00  SECO_RJ     lgbm         945.12
...
```

---

## Step 6: Evaluate Performance

### Run Evaluation

```bash
prevcarga evaluate \
    --config config/config_dev.yaml \
    --storage-backend local \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --areas SECO_RJ \
    --models lgbm \
    --output metrics.json
```

Expected output:
```
Model Evaluation
================
Area: SECO_RJ
Model: lgbm
Period: 2024-01-01 to 2024-01-31

Performance Metrics:
  MAPE: 3.67%
  MAE: 45.23 MWh
  RMSE: 62.45 MWh

By Horizon:
  D+0: MAPE 2.12%
  D+1: MAPE 3.45%
  D+2: MAPE 4.89%

By Period:
  Peak hours (18-21h): MAPE 4.23%
  Off-peak: MAPE 3.12%

Results saved to: metrics.json
```

---

## Step 7: Visualize Results (Optional)

Create a simple visualization script:

```python
"""Visualize predictions vs actuals."""
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions and actuals
predictions = pd.read_parquet("data/predictions/2024-01-15/SECO_RJ_lgbm.parquet")
actuals = pd.read_parquet("data/raw/SECO_RJ.parquet")

# Filter actuals for the prediction date
actuals = actuals[actuals['timestamp'].dt.date == pd.Timestamp('2024-01-15').date()]

# Merge
df = predictions.merge(
    actuals[['timestamp', 'load_mwh']],
    on='timestamp',
    how='left'
)
df = df.rename(columns={'load_mwh': 'actual_load'})

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['actual_load'], label='Actual', linewidth=2)
plt.plot(df['timestamp'], df['predicted_load'], label='Predicted', linewidth=2, linestyle='--')
plt.xlabel('Time')
plt.ylabel('Load (MWh)')
plt.title('Load Forecast vs Actual - SECO_RJ - 2024-01-15')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('forecast_comparison.png', dpi=150)
print("Plot saved to: forecast_comparison.png")
```

---

## Summary

Congratulations! You have successfully:

1. Installed and configured PrevCarga
2. Created sample training data
3. Trained a LightGBM model
4. Generated predictions for a future date
5. Evaluated model performance

### What's Next?

- **[Training Workflow Tutorial](training-workflow.md)**: Deep dive into training options
- **[Prediction Workflow Tutorial](prediction-workflow.md)**: Advanced prediction techniques
- **[Usage Guide](../guides/usage.md)**: Complete CLI reference
- **[API Reference](../api/index.rst)**: Programmatic usage

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Ensure you activated the virtual environment:
```bash
source .venv/bin/activate
```

**Issue**: `FileNotFoundError: Model not found`

**Solution**: Train the model first before predicting:
```bash
prevcarga train --areas SECO_RJ --models lgbm ...
```

**Issue**: Memory errors during training

**Solution**: Reduce parallel workers or batch size:
```bash
prevcarga train --parallel 1 ...
```

---

**Need help?** Check the [Troubleshooting Guide](../guides/troubleshooting.md)
