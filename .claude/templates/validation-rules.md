# PrevCarga Validation Rules

> Domain-specific validation rules for ticket implementation in the PrevCarga forecasting system.

---

## 1. Code Standards

### Python Version & Style
- **Python**: 3.12+ required
- **Type Hints**: All public functions and methods MUST have type hints
- **Docstrings**: Google-style docstrings for all public APIs
- **Formatting**: Black (default settings) + ruff
- **Line Length**: 88 characters (Black default)

### Import Order (enforced by ruff)
```python
# 1. Standard library
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# 2. Third-party
import pandas as pd
import numpy as np
from pydantic import BaseModel

# 3. Local
from prevcarga.data import DataLoader
from prevcarga.models import BaseModel
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `LGBMModel`, `FeaturePlugin` |
| Functions | snake_case | `generate_features`, `fit_model` |
| Constants | UPPER_SNAKE | `DEFAULT_HORIZONS`, `AREA_CODES` |
| Private | _prefix | `_validate_input`, `_cache` |
| Modules | snake_case | `feature_plugins.py`, `model_registry.py` |

---

## 2. Architecture Patterns

### Plugin Pattern (Features & Models)
All feature plugins MUST inherit from `BaseFeaturePlugin`:
```python
class BaseFeaturePlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def generate_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame: ...

    @abstractmethod
    def get_feature_names(self, config: Dict) -> List[str]: ...
```

All models MUST inherit from `BaseModel`:
```python
class BaseModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def supported_horizons(self) -> List[int]: ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, config: Dict) -> 'BaseModel': ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, horizons: List[int]) -> pd.DataFrame: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(cls, path: str) -> 'BaseModel': ...
```

### Registry Pattern
- All plugins/models MUST be registered via decorators
- Registration must happen at import time
- Example:
```python
@model_registry.register("lgbm")
class LGBMModel(BaseModel):
    ...
```

### Factory Pattern (Storage)
- Storage backends selected via `StorageFactory.create(backend_type)`
- Valid backends: `"s3"`, `"local"`

### Validation Rules for Patterns
- [ ] Every plugin class inherits from appropriate ABC
- [ ] Every plugin is registered in its registry
- [ ] No direct instantiation of storage backends (use Factory)
- [ ] Configuration loaded via Pydantic models
- [ ] All I/O operations use storage abstraction

---

## 3. Testing Requirements

### Coverage
- **Minimum**: 70% line coverage
- **Target**: 85% for core modules (`models/`, `features/`, `combination/`)
- **Critical paths**: 100% coverage for reconciliation logic

### Test Structure
```
tests/
├── unit/
│   ├── test_features/
│   │   ├── test_temporal_plugin.py
│   │   └── test_calendar_plugin.py
│   ├── test_models/
│   │   ├── test_lgbm_model.py
│   │   └── test_random_forest_model.py
│   └── test_data/
│       └── test_loaders.py
├── integration/
│   ├── test_workflows/
│   └── test_cli/
├── fixtures/
│   ├── sample_load_data.parquet
│   └── sample_weather_data.parquet
└── conftest.py
```

### Mocking Requirements
- **S3 operations**: Use `moto` or mock `boto3` client
- **Time-dependent tests**: Use `freezegun` or `time_machine`
- **External APIs**: Always mock, never call real endpoints in tests

### Test Naming
```python
def test_{method_name}_{scenario}_{expected_result}():
    # test_fit_with_valid_data_returns_trained_model
    # test_predict_with_missing_features_raises_validation_error
```

---

## 4. Domain Constraints

### Area Codes (17 areas)
```python
VALID_AREA_CODES = {
    # Southeast/Midwest (SECO)
    "SP", "RJ", "MG", "ES",
    # South (S)
    "PR", "SC", "RS",
    # Northeast (NE)
    "BA", "PE", "CE", "ALPE", "PBRN", "BAOE",
    # North (N)
    "PA", "AM", "AP", "AC", "RO", "TON"
}
```

### Subsystems (4 subsystems + losses)
```python
SUBSYSTEMS = {"SECO", "S", "NE", "N"}
LOSS_COMPONENTS = {"Losses_SECO", "Losses_S", "Losses_NE", "Losses_N"}
NATIONAL = {"SIN"}  # National total
```

### Forecast Horizons
```python
VALID_HORIZONS = list(range(0, 9))  # D+0 to D+8
INTRADAY_HORIZONS = [0, 1]  # D+0 and D+1 support intraday updates
```

### Model Names
```python
VALID_MODEL_NAMES = {
    "lgbm",           # LightGBM
    "random_forest",  # Random Forest
    "regdin_svm",     # ARIMA + SVM
    "holt_winters",   # Holt-Winters ETS
}
```

### Time Resolution
- **Half-hourly**: 48 periods per day (00:00-00:30, 00:30-01:00, ...)
- **Hourly**: 24 periods per day
- **Timestamps**: UTC-3 (Brasília time), no DST handling in storage

### Data Validation Rules
- [ ] Area codes must be in `VALID_AREA_CODES`
- [ ] Horizons must be in range [0, 8]
- [ ] Load values must be positive (> 0 MWh)
- [ ] Temperature values must be in range [-10, 50] °C
- [ ] Timestamps must be half-hourly aligned

---

## 5. File Structure Validation

### Expected Source Layout
```
src/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── main.py              # Click application entry
│   ├── train.py             # train command
│   ├── predict.py           # predict command
│   └── backtest.py          # backtest command
├── data/
│   ├── __init__.py
│   ├── loaders.py           # DataLoader classes
│   ├── validators.py        # Pydantic schemas
│   └── imputation.py        # Missing data handling
├── features/
│   ├── __init__.py
│   ├── base.py              # BaseFeaturePlugin
│   ├── registry.py          # FeatureRegistry
│   ├── temporal.py          # TemporalFeaturePlugin
│   ├── calendar.py          # CalendarFeaturePlugin
│   ├── lags.py              # LagsFeaturePlugin
│   └── advanced/
│       ├── loess.py         # LOESSPlugin
│       ├── wavelets.py      # WaveletTransformPlugin
│       └── blf.py           # BLFStrategyPlugin
├── models/
│   ├── __init__.py
│   ├── base.py              # BaseModel
│   ├── registry.py          # ModelRegistry
│   ├── lgbm.py              # LGBMModel
│   ├── random_forest.py     # RandomForestModel
│   ├── regdin_svm.py        # RegDinSVMModel
│   └── holt_winters.py      # HoltWintersModel
├── combination/
│   ├── __init__.py
│   ├── base.py              # BaseCombiner
│   ├── weighted.py          # WeightedAverageCombiner
│   ├── stacking.py          # StackingCombiner
│   └── markov.py            # MarkovChainCombiner
├── reconciliation/
│   ├── __init__.py
│   ├── base.py              # BaseReconciler
│   ├── hierarchy.py         # HierarchyDefinition
│   ├── mint.py              # MinTReconciler
│   ├── ols.py               # OLSReconciler
│   └── wls.py               # WLSReconciler
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # MAPE, MAE, RMSE
│   ├── reports.py           # ReportGenerator
│   └── drift.py             # DriftDetector
├── orchestrator/
│   ├── __init__.py
│   ├── workflows.py         # TrainingWorkflow, PredictionWorkflow
│   └── executor.py          # ParallelExecutor
└── storage/
    ├── __init__.py
    ├── base.py              # StorageBackend ABC
    ├── factory.py           # StorageFactory
    ├── s3.py                # S3StorageBackend
    └── local.py             # LocalStorageBackend
```

### Configuration Files
```
config/
├── default.yaml             # Default configuration
├── areas/
│   ├── SP.yaml              # Area-specific overrides
│   └── ...
├── models/
│   ├── lgbm.yaml            # Model hyperparameters
│   └── ...
└── logging.yaml             # Logging configuration
```

---

## 6. Dependency Validation

### Ticket Dependencies
Each ticket specifies:
- `depends_on`: List of tickets that must be completed first
- `blocks`: List of tickets blocked by this one

### Validation Rules
- [ ] No circular dependencies in ticket graph
- [ ] All referenced tickets exist
- [ ] Epic order is respected (Epic-00 before Epic-01, etc.)
- [ ] Infrastructure tickets (Epic-00, 01) complete before feature tickets

### Critical Path
```
Epic-00 (Foundation)
    → Epic-01 (Data)
    → Epic-02A (Core Features)
    → Epic-03 (Models)
    → Epic-05A (Combination)
    → Epic-06A (Reconciliation)
    → Epic-08A (Workflows)
    → Epic-09A (CLI)
```

---

## 7. Quality Gates

### Before Marking Ticket Complete
1. [ ] All acceptance criteria checked
2. [ ] All implementation tasks completed
3. [ ] Tests pass: `pytest tests/ -v`
4. [ ] Coverage meets threshold: `pytest --cov=src --cov-fail-under=70`
5. [ ] Linting passes: `ruff check src/`
6. [ ] Formatting correct: `black --check src/`
7. [ ] Type checking passes: `mypy src/`
8. [ ] No security issues: `bandit -r src/`

### Commit Convention
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
Scope: `data`, `features`, `models`, `cli`, `storage`, etc.

Example:
```
feat(models): implement LGBMModel with asymmetric loss

- Add custom asymmetric loss function for under-prediction penalty
- Implement hyperparameter optimization via Optuna
- Add support for D+0 to D+1 horizons

Refs: PC-025-03-lgbm-model
```

---

## 8. Performance Requirements

### Training
- Single model + area: < 30 minutes
- All 5 models × 17 areas: < 24 hours (parallelized)

### Prediction
- Intraday update (all 26 series): < 5 minutes
- Single prediction batch: < 30 seconds

### Memory
- Peak memory: < 16 GB for training
- Prediction memory: < 4 GB

---

## 9. Security Validation

### Forbidden Patterns
- [ ] No hardcoded credentials
- [ ] No `eval()` or `exec()` on user input
- [ ] No pickle for untrusted data (use Parquet)
- [ ] No SQL injection vulnerabilities
- [ ] AWS credentials via environment/IAM only

### Required Security Measures
- [ ] Input validation on all external data
- [ ] Logging excludes sensitive information
- [ ] Dependencies pinned in `pyproject.toml`
