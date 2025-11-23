# Contributing to PrevCargaONS

First off, thank you for considering contributing to PrevCargaONS! 🎉

The following is a set of guidelines for contributing to this electric load forecasting system. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Plugin Development](#plugin-development)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project and everyone participating in it is governed by respect, collaboration, and professionalism. By participating, you are expected to uphold this standard. Please report unacceptable behavior to gabriel.goncalves@ons.org.br.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- Git
- Basic understanding of electric load forecasting (helpful but not required)

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/gaugustog/prevcarga.git
cd prevcarga
```

2. **Create a virtual environment and install dependencies**

```bash
# Using uv (recommended)
uv sync --dev

# Or using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

3. **Install pre-commit hooks**

```bash
uv run pre-commit install
```

4. **Verify your setup**

```bash
# Run tests
uv run pytest

# Run linting
uv run ruff check src/

# Run type checking
uv run mypy src/
```

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, configuration files, data samples)
- **Describe the behavior you observed** and what you expected
- **Include error messages and stack traces**
- **Specify your environment** (OS, Python version, package versions)

**Bug Report Template:**

```markdown
## Description
A clear description of the bug.

## Steps to Reproduce
1. Load data with...
2. Train model with...
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- PrevCargaONS version: [e.g., 1.0.0]
- Storage backend: [S3/Local]

## Additional Context
Any other relevant information.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a step-by-step description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List any alternatives you've considered**

### Contributing Code

We welcome code contributions! Here are the main areas where you can contribute:

1. **New Models**: Implement new forecasting models (Neural Networks, Transformers, etc.)
2. **Feature Engineering**: Add new feature plugins (wavelets, decompositions, external data)
3. **Combination Strategies**: Implement new ensemble methods
4. **Reconciliation Methods**: Add new hierarchical reconciliation algorithms
5. **Performance Optimization**: Improve computational efficiency
6. **Bug Fixes**: Fix identified issues
7. **Documentation**: Improve existing documentation or add new guides
8. **Tests**: Increase test coverage

## 🔄 Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Urgent fixes for production
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements

### Workflow Steps

1. **Create a new branch** from `main`

```bash
git checkout -b feature/new-model-plugin
```

2. **Make your changes** following our coding standards

3. **Write/update tests** for your changes

```bash
# Run tests frequently during development
uv run pytest tests/

# Run tests for specific module
uv run pytest tests/unit/models/test_new_model.py
```

4. **Update documentation** if needed

5. **Run quality checks**

```bash
# Format code
uv run black src/ tests/

# Check linting
uv run ruff check src/ tests/

# Type checking
uv run mypy src/

# Run all tests with coverage
uv run pytest --cov=src --cov-report=html
```

6. **Commit your changes** with clear commit messages

7. **Push to your fork** and create a pull request

## 💻 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Formatter**: Black (automatic formatting)
- **Linter**: Ruff (replaces flake8, pylint, isort)
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google style, required for all public APIs

### Example Code Style

```python
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


class ExampleModel:
    """Brief description of the model.
    
    Longer description explaining the model's purpose, algorithm,
    and use cases within the PrevCargaONS system.
    
    Args:
        param1: Description of parameter 1.
        param2: Description of parameter 2.
        config: Optional configuration dictionary.
    
    Attributes:
        attribute1: Description of attribute 1.
        attribute2: Description of attribute 2.
    
    Example:
        >>> model = ExampleModel(param1="value", param2=10)
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        param1: str,
        param2: int = 10,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the model with given parameters."""
        self.param1 = param1
        self.param2 = param2
        self.config = config or {}
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on provided data.
        
        Args:
            X: Feature DataFrame with shape (n_samples, n_features).
            y: Target Series with shape (n_samples,).
        
        Raises:
            ValueError: If X and y have different lengths.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        # Training logic here
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input data.
        
        Args:
            X: Feature DataFrame with shape (n_samples, n_features).
        
        Returns:
            Array of predictions with shape (n_samples,).
        """
        # Prediction logic here
        return np.zeros(len(X))
```

### Type Hints

Always use type hints for function signatures:

```python
from typing import List, Dict, Optional, Union
import pandas as pd

def process_data(
    df: pd.DataFrame,
    areas: List[str],
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Process data for specified areas."""
    ...
```

### Error Handling

- Use specific exception types
- Provide informative error messages
- Log errors appropriately

```python
from src.utils.logging import get_logger

logger = get_logger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Load data from specified path."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise
    
    if df.empty:
        raise ValueError(f"Loaded data is empty: {path}")
    
    return df
```

## 🧪 Testing Guidelines

### Test Structure

We use pytest with the following structure:

```
tests/
├── unit/              # Unit tests (isolated component tests)
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── integration/       # Integration tests (component interaction)
│   ├── test_workflows.py
│   └── test_storage.py
└── e2e/              # End-to-end tests (full workflows)
    └── test_train_predict.py
```

### Writing Tests

**Unit Test Example:**

```python
import pytest
import pandas as pd
from src.features.common import LagFeatures


class TestLagFeatures:
    """Test suite for LagFeatures plugin."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='30T'),
            'load': range(100),
            'temperature': range(100, 200)
        })
    
    def test_create_lags(self, sample_data):
        """Test lag feature creation."""
        plugin = LagFeatures(lags=[1, 2, 48])
        result = plugin.transform(sample_data)
        
        assert 'load_lag_1' in result.columns
        assert 'load_lag_2' in result.columns
        assert 'load_lag_48' in result.columns
        assert len(result) == len(sample_data)
    
    def test_handles_missing_values(self, sample_data):
        """Test handling of missing values in lag creation."""
        sample_data.loc[5, 'load'] = None
        plugin = LagFeatures(lags=[1])
        result = plugin.transform(sample_data)
        
        # Lag of missing value should be NaN
        assert pd.isna(result.loc[6, 'load_lag_1'])
    
    def test_invalid_lag_raises_error(self):
        """Test that invalid lag values raise appropriate errors."""
        with pytest.raises(ValueError, match="Lags must be positive"):
            LagFeatures(lags=[-1, 0])
```

**Integration Test Example:**

```python
import pytest
from src.orchestrator.workflows.train import TrainWorkflow
from src.storage.factory import StorageFactory


class TestTrainWorkflow:
    """Integration tests for training workflow."""
    
    @pytest.fixture
    def storage_backend(self, tmp_path):
        """Create temporary storage backend."""
        return StorageFactory.create("local", base_path=str(tmp_path))
    
    def test_train_lgbm_workflow(self, storage_backend, sample_train_data):
        """Test complete LGBM training workflow."""
        workflow = TrainWorkflow(
            storage_backend=storage_backend,
            config={"model": "lgbm", "areas": ["RJ"]}
        )
        
        result = workflow.run(sample_train_data)
        
        assert result.success
        assert result.model_version is not None
        assert storage_backend.exists(result.model_path)
```

### Test Coverage Requirements

- **Minimum overall coverage**: 70%
- **New code**: 80% coverage required
- **Critical paths** (model training, prediction): 90% coverage

Run coverage reports:

```bash
# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Fixtures and Mocking

Use pytest fixtures for reusable test data:

```python
@pytest.fixture
def sample_config():
    """Provide sample configuration for tests."""
    return {
        "storage": {"backend": "local"},
        "models": {"lgbm": {"enabled": True}},
        "training": {"historical_window": 90}
    }

@pytest.fixture
def mock_s3_backend(mocker):
    """Mock S3 backend for testing."""
    mock = mocker.Mock()
    mock.exists.return_value = True
    mock.load.return_value = pd.DataFrame()
    return mock
```

## 📚 Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def forecast_load(
    area: str,
    horizon: int,
    model: str = "lgbm"
) -> pd.DataFrame:
    """Generate load forecast for specified area and horizon.
    
    This function orchestrates the complete forecasting process including
    data loading, feature engineering, model prediction, and result formatting.
    
    Args:
        area: Area code (e.g., "RJ", "SP", "MG").
        horizon: Forecast horizon in days (0-8).
        model: Model name to use. Defaults to "lgbm".
    
    Returns:
        DataFrame with columns ['timestamp', 'forecast', 'lower_bound', 'upper_bound']
        indexed by datetime.
    
    Raises:
        ValueError: If area is not recognized or horizon is out of range.
        ModelNotFoundError: If specified model is not available.
    
    Example:
        >>> forecast = forecast_load(area="RJ", horizon=1, model="lgbm")
        >>> print(forecast.head())
    """
    ...
```

### Documentation Files

When adding new features, update relevant documentation:

- **README.md**: For major features that affect user experience
- **API Reference**: For new public APIs
- **User Guide**: For new commands or workflows
- **Developer Guide**: For new plugin types or architectural changes

## 🔌 Plugin Development

### Creating a New Feature Plugin

```python
from src.features.base import BaseFeaturePlugin, register_feature_plugin
from typing import List
import pandas as pd


@register_feature_plugin(name="my_custom_features")
class MyCustomFeaturePlugin(BaseFeaturePlugin):
    """Custom feature engineering plugin.
    
    Detailed description of what features this plugin creates
    and when they should be used.
    """
    
    def __init__(self, param1: int = 10):
        """Initialize plugin with parameters."""
        super().__init__()
        self.param1 = param1
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data by adding custom features.
        
        Args:
            df: Input DataFrame with required columns.
        
        Returns:
            DataFrame with additional feature columns.
        """
        df = df.copy()
        # Add your feature engineering logic
        df['custom_feature_1'] = df['load'] * self.param1
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names created by this plugin."""
        return ['custom_feature_1']
    
    def get_config(self) -> dict:
        """Return plugin configuration for reproducibility."""
        return {'param1': self.param1}
```

### Creating a New Model Plugin

```python
from src.models.base import BaseModel, register_model, ModelArtifact
from typing import Dict, Any
import numpy as np


@register_model(
    name="my_custom_model",
    type="end_to_end",
    horizon=[0, 1, 2, 3, 4, 5, 6, 7, 8]
)
class MyCustomModel(BaseModel):
    """Custom forecasting model.
    
    Detailed description of the model algorithm, use cases,
    and performance characteristics.
    """
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> ModelArtifact:
        """Train the model on provided data."""
        # Training logic
        return ModelArtifact(
            model=self,
            metrics={'mae': 0.0, 'mape': 0.0},
            metadata={'trained_at': datetime.now()}
        )
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Generate predictions."""
        # Prediction logic
        return np.zeros(len(X))
    
    def save(self, path: str, version: str) -> None:
        """Save model to disk."""
        # Serialization logic
        pass
    
    @classmethod
    def load(cls, path: str) -> 'MyCustomModel':
        """Load model from disk."""
        # Deserialization logic
        return cls()
```

## 💬 Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring (no feature change or bug fix)
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates

### Examples

```bash
feat(models): add transformer-based forecasting model

Implement attention-based temporal forecasting model with
multi-head attention mechanism. Includes training pipeline
and integration with existing model registry.

Closes #123

---

fix(storage): correct S3 path handling for Windows

Fix path separator issues when running on Windows systems
with S3 storage backend.

Fixes #456

---

docs(readme): update installation instructions

Add instructions for uv-based installation and clarify
AWS credentials setup for S3 backend.

---

test(features): increase coverage for lag features

Add tests for edge cases including missing values,
insufficient history, and invalid lag specifications.
```

## 🔀 Pull Request Process

### Before Submitting

- [ ] Code follows the project's coding standards
- [ ] All tests pass locally
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated (if applicable)
- [ ] Commit messages follow conventions
- [ ] Branch is up-to-date with `main`

### PR Description Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Related Issues
Closes #XXX

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
Describe the tests you ran and how to reproduce them.

## Screenshots (if applicable)
Add screenshots demonstrating the change.

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where necessary
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing unit tests pass locally
- [ ] Any dependent changes have been merged
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests, linting, and type checking
2. **Code Review**: At least one maintainer reviews the code
3. **Feedback**: Address any comments or requested changes
4. **Approval**: PR is approved by maintainer(s)
5. **Merge**: PR is merged into `main`

### Review Criteria

Reviewers will check for:

- **Correctness**: Does the code work as intended?
- **Quality**: Is the code well-structured and maintainable?
- **Tests**: Are there adequate tests with good coverage?
- **Documentation**: Is the code and functionality well-documented?
- **Performance**: Are there any performance concerns?
- **Security**: Are there any security implications?

## 🎯 Priority Areas

We're particularly interested in contributions in these areas:

### High Priority

- 🔥 **Performance optimization** for large-scale training
- 🔥 **GPU acceleration** for LGBM and neural network models
- 🔥 **Additional model implementations** (Transformers, LSTM, etc.)
- 🔥 **Enhanced feature engineering** plugins

### Medium Priority

- 📊 **Visualization tools** for model comparison and diagnostics
- 📊 **REST API** development with FastAPI
- 📊 **Real-time monitoring** dashboard
- 📊 **MLflow integration** for experiment tracking

### Always Welcome

- 🐛 Bug fixes and issue resolutions
- 📚 Documentation improvements
- ✅ Test coverage improvements
- 🎨 Code quality improvements

## 💡 Questions?

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: gabriel.goncalves@ons.org.br for direct contact

## 🙏 Thank You!

Your contributions help make PrevCargaONS better for the entire Brazilian electric sector community. We appreciate your time and effort! ⚡

---

**Happy coding!** 🚀
