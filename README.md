# ⚡ PrevCargaONS - Unified Electric Load Forecasting System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Modern, unified platform for short-term electric load forecasting (D+0 to D+8) for the Brazilian National Interconnected System**

## 📋 Overview

PrevCargaONS is a next-generation electric load forecasting system that modernizes and unifies multiple forecasting approaches into a single, extensible Python platform. The system combines traditional statistical methods with modern machine learning techniques, supporting both end-to-end and hierarchical forecasting architectures.

### Key Features

- 🎯 **Multi-Model Architecture**: 5 base models (LGBM, Random Forest, ARIMA+SVM, Holt-Winters, and extensible)
- 🌳 **Hierarchical Reconciliation**: Bottom-up reconciliation ensuring consistency across 26 time series
- 🔄 **Intraday Forecasting**: Operational execution every 30 minutes with BLF strategy
- 🧩 **Plugin System**: Extensible feature engineering, models, and combination strategies
- 📊 **Advanced Combination**: Stacking, Voting, and Markov Chain ensemble methods
- ☁️ **Dual Storage**: Seamless switching between S3 (production) and local filesystem (development)
- 🔍 **Comprehensive Evaluation**: Drift detection, backtesting, and detailed performance metrics
- 🚀 **Production Ready**: Containerized, cloud-native architecture with AWS Fargate support

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Python (Click)                        │
│  Commands: train | predict | backtest | combine | eval-*        │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │   Orchestrator Layer     │
        │  - Config Management     │
        │  - Workflow Execution    │
        │  - Parallel Coordination │
        └────────────┬────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
┌────▼─────┐  ┌─────▼──────┐  ┌────▼─────┐
│  Data    │  │  Feature   │  │  Model   │
│  Layer   │  │  Layer     │  │  Layer   │
└────┬─────┘  └─────┬──────┘  └────┬─────┘
     │              │               │
┌────▼──────────────▼───────────────▼─────┐
│       Storage Backend (Abstract)         │
│  ┌──────────────┐    ┌────────────────┐ │
│  │ S3 Backend   │    │ Local Backend  │ │
│  │ (Production) │    │ (Development)  │ │
│  └──────────────┘    └────────────────┘ │
└──────────────────────────────────────────┘
     │              │               │
┌────▼─────┐  ┌─────▼──────┐  ┌────▼─────┐
│Combiner  │  │Reconciler  │  │Evaluator │
│  Layer   │  │  Layer     │  │  Layer   │
└──────────┘  └────────────┘  └──────────┘
```

## 🎯 Coverage

### Geographic Scope
- **17 Areas**: RJ, SP, MG, ES, PR, SC, RS, BA, PE, CE, and more
- **4 Subsystems**: SECO (Southeast/Midwest), S (South), NE (Northeast), N (North)
- **4 Loss Components**: System losses calculated by difference
- **1 National Series**: Total SIN (Sistema Interligado Nacional)

### Forecasting Horizons
- **D+0 (Intraday)**: Updates every 30 minutes with BLF strategy
- **D+1 to D+8**: Daily batch forecasts for medium-term planning

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- AWS credentials (for S3 backend)

### Installation

```bash
# Clone the repository
git clone https://github.com/gaugustog/prevcarga.git
cd prevcarga

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Basic Usage

```bash
# Train a model (local development)
prevcarga train \
  --config config_dev.yaml \
  --storage-backend local \
  --areas SECO_RJ \
  --models lgbm \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# Generate predictions
prevcarga predict \
  --config config.yaml \
  --date 2025-01-17 \
  --mode intraday \
  --areas all \
  --reconcile

# Run backtest
prevcarga backtest \
  --config config.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --retrain-intervals 1,3,7 \
  --report report.html
```

## 📊 Models

### End-to-End Models
- **LightGBM**: Gradient boosting with custom asymmetric loss and intraday BLF strategy
- **Random Forest**: 216 specialized models (9 horizons × 24 lead hours)

### Hierarchical Models (DM→Profile)
- **RegDin + SVM**: ARIMA for daily mean + 48 SVM models for half-hourly profile
- **Holt-Winters**: ETS for daily mean + 48 HW models for profile decomposition

### Future Models
- Extensible plugin system for Transformers, Neural Networks, Prophet, and more

## 🧩 Plugin System

### Feature Engineering Plugins

```python
from src.features.base import BaseFeaturePlugin, register_feature_plugin

@register_feature_plugin(name="custom_features")
class CustomFeaturePlugin(BaseFeaturePlugin):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Your custom feature engineering logic
        return df_with_features
    
    def get_feature_names(self) -> List[str]:
        return ["feature1", "feature2", "feature3"]
```

### Model Plugins

```python
from src.models.base import BaseModel, register_model

@register_model(name="custom_model", type="end_to_end", horizon=[0,1,2,3])
class CustomModel(BaseModel):
    def train(self, X, y, **kwargs):
        # Training logic
        pass
    
    def predict(self, X, **kwargs):
        # Prediction logic
        pass
```

## 🔄 Combination Strategies

- **Weighted Voting**: Fixed or learned weights via linear regression
- **Stacking**: Meta-model (LGBM/LinearRegression) trained on base predictions
- **Markov Chain**: Dynamic weights based on state transitions and historical performance
- **Bias Correction**: Systematic error correction by context (hour, day, temperature)

## 📈 Reconciliation

Bottom-up hierarchical reconciliation ensures consistency:

```
National = Σ(Subsystems)
Subsystem = Σ(Areas) + Losses
```

Methods supported:
- **MinT**: Minimum Trace reconciliation (default)
- **OLS**: Ordinary Least Squares
- **WLS**: Weighted Least Squares (variance/error weights)

## 📊 Evaluation & Metrics

- **Standard Metrics**: MAPE, MAE, RMSE
- **Percentile Analysis**: P5, P25, P50, P75, P95 deviations
- **Period-Specific**: Peak vs off-peak performance
- **Drift Detection**: Automatic monitoring for model degradation
- **Comparative Analysis**: Model comparison matrices and reports

## ☁️ Storage Backends

### S3 (Production)
```yaml
storage:
  backend: s3
  s3:
    bucket: prevcarga-bucket
    region: us-east-1
```

### Local Filesystem (Development)
```yaml
storage:
  backend: local
  local:
    base_path: ./data
```

Both backends share identical path structures for seamless switching.

## 🐳 Deployment

### Docker

```bash
# Build image
docker build -t prevcarga-ons:latest .

# Run training
docker run -v $(pwd)/config:/app/config \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  prevcarga-ons:latest train --config /app/config/config.yaml
```

### AWS Fargate

The system is designed for serverless deployment on AWS Fargate with EventBridge scheduling for operational execution.

```bash
# Deploy to Fargate
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster prevcarga --service-name prevcarga-predict --task-definition prevcarga:1
```

## 📁 Project Structure

```
prevcarga-ons/
├── src/
│   ├── cli/              # Command-line interface
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering plugins
│   ├── models/           # Model implementations
│   ├── combination/      # Ensemble strategies
│   ├── reconciliation/   # Hierarchical reconciliation
│   ├── evaluation/       # Metrics and reporting
│   ├── orchestrator/     # Workflow coordination
│   └── storage/          # Storage backend abstraction
├── config/               # Configuration files
├── data/                 # Local data storage
├── tests/                # Test suite
├── docs/                 # Documentation
├── scripts/              # Utility scripts
└── pyproject.toml        # Project dependencies
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test suite
uv run pytest tests/unit/models/
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Getting Started**: Installation and quickstart guides
- **User Guide**: Detailed usage instructions for all commands
- **Developer Guide**: Architecture, plugin development, and contribution guidelines
- **API Reference**: Complete API documentation

Generate documentation:
```bash
cd docs
mkdocs serve
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Run linting
uv run ruff check src/

# Run formatting
uv run black src/

# Run type checking
uv run mypy src/
```

## 📊 Performance Benchmarks

| Operation | Duration | Hardware |
|-----------|----------|----------|
| Train 1 area (LGBM) | ~15 min | 8 cores, 16GB RAM |
| Train all areas (LGBM) | ~2 hours | Parallel, 32 cores |
| Intraday prediction | <3 min | 4 cores, 8GB RAM |
| Full backtest (1 year) | ~6 hours | 16 cores, 32GB RAM |

## 🗺️ Roadmap

### Current (v1.0.0)
- ✅ Core model implementations
- ✅ Hierarchical reconciliation
- ✅ Combination strategies
- ✅ Dual storage backends
- ✅ Complete CLI

### Planned (v1.1.0)
- 🔜 REST API with FastAPI
- 🔜 Real-time monitoring dashboard
- 🔜 GPU acceleration for training
- 🔜 MLflow integration

### Future (v2.0.0)
- 🔮 Transformer-based models
- 🔮 AutoML for meta-model optimization
- 🔮 Multi-modal features (solar, wind, economic indicators)
- 🔮 Probabilistic forecasting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original PrevCargaDESSEM system in R
- ONS (Operador Nacional do Sistema Elétrico) for domain knowledge
- Open-source community for excellent tools and libraries

## 📞 Contact

- **Project Lead**: Gabriel Gonçalves
- **ML Engineers**: Danilo Lopes - Sofia Lopes
- **Email**: gabriel.goncalves@ons.org.br
- **Issues**: [GitHub Issues](https://github.com/gaugustog/prevcarga/issues)

---

**Built with ❤️ for the Brazilian electric sector**