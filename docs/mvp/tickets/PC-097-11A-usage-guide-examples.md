# PC-097-11A: Usage Guide and Examples

**Ticket ID:** PC-097-11A  
**Epic:** [Epic-11A: Documentation & Infrastructure](../epics/Epic-11A.md)  
**User Story:** US-11A.1 - Comprehensive Documentation Suite  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Create comprehensive usage guide with practical examples, tutorials, and best practices for using the PrevCarga system. Cover CLI commands, common workflows, configuration management, and advanced usage patterns.

**As a** PrevCarga user  
**I want** detailed usage examples and tutorials  
**So that** I can effectively use the system for my forecasting needs

---

## ✅ Acceptance Criteria

- [ ] Complete CLI command reference created
- [ ] Training workflow examples documented
- [ ] Prediction workflow examples documented
- [ ] Feature engineering examples provided
- [ ] Configuration management guide created
- [ ] Common use cases documented
- [ ] Best practices section included
- [ ] Troubleshooting guide for runtime issues
- [ ] Performance optimization tips documented
- [ ] Interactive code examples that can be copied and run

---

## 🔧 Implementation Tasks

### 1. Create Usage Guide Structure
- [ ] Create `docs/source/guides/usage.md`
- [ ] Add table of contents
- [ ] Structure sections logically
- [ ] Add navigation links

### 2. Document CLI Commands
- [ ] `prevcarga predict` command with all options
- [ ] `prevcarga train` command group
- [ ] `prevcarga data` command group
- [ ] `prevcarga feature` command group
- [ ] `prevcarga config` command group
- [ ] `prevcarga status` command
- [ ] Command option explanations
- [ ] Exit codes and error handling

### 3. Create Training Tutorials
- [ ] Single model training walkthrough
- [ ] Batch training tutorial
- [ ] Cross-validation setup
- [ ] Hyperparameter tuning guide
- [ ] Model evaluation examples
- [ ] Training monitoring and logging

### 4. Create Prediction Tutorials
- [ ] Single prediction example
- [ ] Batch prediction tutorial
- [ ] Ensemble prediction guide
- [ ] Real-time prediction workflow
- [ ] Prediction confidence intervals
- [ ] Result interpretation

### 5. Feature Engineering Examples
- [ ] Temporal features usage
- [ ] Calendar features configuration
- [ ] Custom feature plugins
- [ ] Feature selection strategies
- [ ] Feature importance analysis

### 6. Configuration Management
- [ ] Configuration file structure
- [ ] Environment variable usage
- [ ] Multi-environment setup
- [ ] Configuration validation
- [ ] Configuration best practices

### 7. Advanced Usage Patterns
- [ ] Custom model integration
- [ ] Pipeline customization
- [ ] Parallel processing optimization
- [ ] Memory management
- [ ] Production deployment patterns

### 8. Create Tutorials
- [ ] Getting started tutorial (basic workflow)
- [ ] End-to-end forecasting tutorial
- [ ] Custom plugin development tutorial
- [ ] Production deployment tutorial

---

## 📂 Files to Create

```
docs/source/guides/
├── usage.md                    # Main usage guide
├── tutorials/
│   ├── getting-started.md      # Basic tutorial
│   ├── training-workflow.md    # Training tutorial
│   ├── prediction-workflow.md  # Prediction tutorial
│   ├── custom-plugins.md       # Plugin development
│   └── production-deployment.md # Deployment tutorial
└── examples/
    ├── basic-prediction.md
    ├── batch-training.md
    ├── ensemble-models.md
    └── advanced-configuration.md
```

---

## 🔧 Technical Implementation

### Main Usage Guide (`docs/source/guides/usage.md`)

```markdown
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
- [Troubleshooting](#troubleshooting)

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

---

## Training Models

### Single Model Training

Train a single model for a specific area and date range:

```bash
prevcarga train model \
    --model lgbm \
    --area area001 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --parallel 4
```

**Options:**
- `--model`: Model type (lgbm, rf, regdin_svm, holt_winters, all)
- `--area`: Target area(s) - can specify multiple
- `--start-date`: Training period start (YYYY-MM-DD)
- `--end-date`: Training period end (YYYY-MM-DD)
- `--parallel`: Number of parallel workers (default: 4)
- `--force`: Force retrain existing models

### Training Multiple Areas

```bash
prevcarga train model \
    --model lgbm \
    --area area001 \
    --area area002 \
    --area area003 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

### Training All Models

```bash
prevcarga train model \
    --model all \
    --area area001 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --parallel 8
```

### Batch Training from Configuration

Create a training configuration file (`training_config.yaml`):

```yaml
training:
  models:
    - lgbm
    - rf
    - regdin_svm
  
  areas:
    - area001
    - area002
    - area003
  
  date_ranges:
    - start: "2023-01-01"
      end: "2023-06-30"
    - start: "2023-07-01"
      end: "2023-12-31"
  
  options:
    parallel: 8
    force_retrain: false
```

Execute batch training:

```bash
prevcarga train batch --config training_config.yaml
```

### Dry Run (Preview Training Plan)

```bash
prevcarga train model \
    --model all \
    --area area001 \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --dry-run
```

---

## Generating Predictions

### Single Model Prediction

```bash
prevcarga predict \
    --model lgbm \
    --area area001 \
    --date 2024-01-01
```

### Ensemble Prediction

Use multiple models and combine predictions:

```bash
prevcarga predict \
    --model all \
    --area area001 \
    --date 2024-01-01 \
    --combiner weighted_voting
```

**Available Combiners:**
- `simple_average`: Simple average of all predictions
- `weighted_voting`: Performance-weighted combination
- `best_model`: Select single best performing model

### Batch Predictions

Create prediction configuration (`predictions_config.yaml`):

```yaml
predictions:
  models:
    - lgbm
    - rf
  
  areas:
    - area001
    - area002
  
  dates:
    - "2024-01-01"
    - "2024-01-02"
    - "2024-01-03"
  
  combiner: weighted_voting
  
  output:
    format: parquet
    path: s3://prevcarga-predictions/
```

Execute batch predictions:

```bash
prevcarga predict batch --config predictions_config.yaml
```

### Prediction with Custom Horizon

```bash
prevcarga predict \
    --model lgbm \
    --area area001 \
    --date 2024-01-01 \
    --horizon 24  # 24-hour forecast
```

---

## Feature Engineering

### Build Feature Pipeline

```bash
prevcarga feature build \
    --area area001 \
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
- temporal_features: Basic temporal features (hour, day, month)
- calendar_features: Calendar and holiday features
- lag_features: Lag and rolling window features
- cyclical_encoding: Cyclical encoding for periodic features
- loess_smoothing: LOESS signal smoothing
- wavelet_transform: Wavelet decomposition features
```

### Configure Feature Pipeline

Create `feature_config.yaml`:

```yaml
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
        lags: [1, 7, 24, 168]
        rolling_windows: [24, 168]
    
    - name: cyclical_encoding
      enabled: true
      params:
        columns: [hour, day_of_week, month]
```

Apply configuration:

```bash
prevcarga feature build \
    --config feature_config.yaml \
    --area area001 \
    --start-date 2024-01-01
```

---

## Data Management

### Validate Data Quality

```bash
prevcarga data validate \
    --area area001 \
    --date 2024-01-01
```

### Check Data Availability

```bash
prevcarga data check \
    --area area001 \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

### Show Data Statistics

```bash
prevcarga data stats \
    --area area001 \
    --date 2024-01-01
```

---

## Configuration

### View Current Configuration

```bash
prevcarga config show
```

### Validate Configuration

```bash
prevcarga config validate
```

### Set Configuration Values

```bash
prevcarga config set aws.region us-east-1
prevcarga config set storage.s3_bucket prevcarga-data
```

### Configuration File

Create `config/prevcarga.yaml`:

```yaml
aws:
  region: us-east-1
  profile: prevcarga

storage:
  s3_bucket: prevcarga-data-production
  cache_dir: /tmp/prevcarga_cache

logging:
  level: INFO
  format: json
  output: stdout

models:
  default_parallel: 4
  cache_enabled: true

features:
  cache_enabled: true
  parallel_processing: true
```

Use custom configuration:

```bash
export PREVCARGA_CONFIG=config/prevcarga.yaml
prevcarga predict --model lgbm --area area001 --date 2024-01-01
```

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
AWS Region: us-east-1
S3 Access: ✓ Connected

Models Available:
- lgbm: 15 trained models
- rf: 15 trained models
- regdin_svm: 10 trained models
- holt_winters: 12 trained models

System Health: OK
```

### View Logs

```bash
# View real-time logs
tail -f logs/prevcarga.log

# View JSON logs
jq . logs/prevcarga.jsonl

# Filter by level
jq 'select(.level == "ERROR")' logs/prevcarga.jsonl
```

### Training Progress Monitoring

Training commands show real-time progress:

```
Training Models
===============
Model: lgbm | Area: area001 | Date Range: 2023-01-01 to 2023-12-31

[████████████████████----] 80% | 4/5 months complete
Current: 2023-10-01 | ETA: 2m 15s

Performance Metrics:
- RMSE: 125.43
- MAE: 89.12
- R²: 0.92
```

---

## Troubleshooting

### Common Issues

#### AWS Credentials Not Found

```bash
# Check credentials
aws sts get-caller-identity

# Set credentials
export AWS_PROFILE=prevcarga
export AWS_DEFAULT_REGION=us-east-1
```

#### Model Not Found

```bash
# Check available models
prevcarga status

# Train missing model
prevcarga train model --model lgbm --area area001 \
    --start-date 2023-01-01 --end-date 2023-12-31
```

#### Out of Memory

```bash
# Reduce parallel workers
prevcarga train model --parallel 1

# Use smaller batch size
prevcarga train model --batch-size 1000
```

#### Data Validation Fails

```bash
# Check data quality
prevcarga data validate --area area001 --date 2024-01-01

# View validation report
cat logs/validation_report.json
```

---

## Best Practices

### Model Training

1. **Use appropriate date ranges**: Minimum 6 months for seasonal patterns
2. **Leverage parallel processing**: Use `--parallel` for faster training
3. **Monitor training metrics**: Check logs for performance indicators
4. **Version your models**: Use date-based naming conventions

### Predictions

1. **Use ensemble methods**: Combine multiple models for robustness
2. **Validate inputs**: Always check data quality before prediction
3. **Monitor confidence**: Review prediction intervals
4. **Cache predictions**: Reuse predictions when possible

### Production Deployment

1. **Use Docker containers**: Consistent environment
2. **Configure logging**: JSON format for parsing
3. **Set up monitoring**: Health checks and alerts
4. **Automate workflows**: Scheduled training and prediction

---

## Advanced Usage

### Custom Python Scripts

```python
from prevcarga.workflows import TrainingWorkflow, PredictionWorkflow
from prevcarga.config import SystemConfig

# Initialize configuration
config = SystemConfig.load()

# Training workflow
workflow = TrainingWorkflow(config)
results = workflow.train_model(
    model_type='lgbm',
    area='area001',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Prediction workflow
pred_workflow = PredictionWorkflow(config)
predictions = pred_workflow.predict(
    model_type='lgbm',
    area='area001',
    date='2024-01-01'
)
```

### Integration with Data Pipelines

```python
# Apache Airflow DAG example
from airflow import DAG
from airflow.operators.bash import BashOperator

dag = DAG('prevcarga_daily_forecast', schedule_interval='@daily')

train_task = BashOperator(
    task_id='train_models',
    bash_command='prevcarga train batch --config daily_training.yaml',
    dag=dag
)

predict_task = BashOperator(
    task_id='generate_predictions',
    bash_command='prevcarga predict batch --config daily_predictions.yaml',
    dag=dag
)

train_task >> predict_task
```

---

## Next Steps

- **API Reference**: See [API Documentation](../api/index.rst) for programmatic usage
- **Architecture**: Read [Architecture Guide](../architecture/overview.md) for system design
- **Extensions**: Learn [Plugin Development](extensions.md) for customization
- **Deployment**: Follow [Deployment Guide](tutorials/production-deployment.md)

---

**Need Help?** Check the [FAQ](faq.md) or [open an issue](https://github.com/your-org/prevcarga/issues)
```

---

## 🧪 Testing & Validation

### Validation Commands
```bash
# Test all CLI examples
# Extract code blocks and execute them

# Verify configuration examples
prevcarga config validate --config examples/config.yaml

# Test training examples (dry-run)
prevcarga train model --model lgbm --area area001 \
    --start-date 2023-01-01 --end-date 2023-01-31 --dry-run

# Test prediction examples
prevcarga predict --model lgbm --area area001 --date 2024-01-01 --dry-run

# Verify all links
markdown-link-check docs/source/guides/usage.md
```

### Success Criteria
- [ ] All code examples are syntactically correct
- [ ] All commands execute without errors (or with expected errors)
- [ ] Configuration examples validate successfully
- [ ] Navigation links work correctly
- [ ] Examples cover common use cases
- [ ] Troubleshooting section addresses real issues

---

## 📝 Technical Notes

- Use realistic example data (area001, dates, etc.)
- Include expected outputs for commands
- Show both success and error scenarios
- Use consistent formatting for code blocks
- Add comments to complex examples
- Cross-reference related sections
- Keep examples copy-paste ready

---

## 🔗 Dependencies

**Depends On:**
- PC-095-11A: Documentation Structure Setup
- PC-096-11A: README and Installation Guide
- All CLI implementation tickets (Epic-09A, Epic-09B)

**Blocks:**
- PC-098-11A: API Reference Documentation
- PC-099-11A: Architecture and Extension Documentation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] All implementation tasks completed
- [ ] Usage guide created and reviewed
- [ ] All tutorials written and tested
- [ ] Code examples verified to work
- [ ] Links validated
- [ ] Technical review approved
- [ ] User acceptance testing completed
- [ ] Committed to repository

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-19 | 1.0.0 | Initial ticket created from Epic-11A | System |

---

**Next Ticket:** [PC-098-11A: API Reference Documentation](PC-098-11A-api-reference-docs.md)
