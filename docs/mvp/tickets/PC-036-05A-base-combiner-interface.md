# PC-036-05A: Base Combiner Interface

**Ticket ID:** PC-036-05A  
**Epic:** [Epic-05A: Base Combination Strategies](../epics/Epic-05A.md)  
**User Story:** US-1  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement foundational `BaseCombiner` interface that defines the standard contract for all model combination strategies. Provides abstract methods for combining predictions, fitting combination parameters, and handling metadata. Supports both point forecasts and prediction intervals, handles missing predictions gracefully, and enables 26 time series (17 areas + 4 subsystems + 4 losses + 1 national) across 9 horizons (D+0 to D+8).

**As a** ML engineer  
**I want** a standardized interface for combining model predictions  
**So that** I can easily integrate multiple combination strategies

---

## ✅ Acceptance Criteria

- [ ] `BaseCombiner` abstract class with standard combination contract
- [ ] Supports both point forecasts and prediction intervals
- [ ] Combination metadata (weights, performance) tracked
- [ ] Handles missing predictions gracefully
- [ ] Extensible design allows new combination methods
- [ ] Support for 26 time series and 9 horizons
- [ ] Input validation for prediction formats
- [ ] Logging and monitoring integration
- [ ] Comprehensive tests with mock implementations
- [ ] Documentation with usage examples

---

## 🔧 Implementation Tasks

### 1. Create Combination Module Structure
- [ ] Create `src/models/combination/` directory
- [ ] Create `__init__.py` with module exports
- [ ] Create `base_combiner.py` for interface
- [ ] Add module docstrings

### 2. Define CombinationResult Class
- [ ] Create `CombinationResult` dataclass
- [ ] Fields: combined_predictions, weights, metadata
- [ ] Fields: model_contributions, confidence_intervals
- [ ] Add validation methods
- [ ] Add serialization support

### 3. Define CombinationMetadata Class
- [ ] Create `CombinationMetadata` dataclass
- [ ] Fields: timestamp, models_used, combination_method
- [ ] Fields: performance_metrics, processing_time
- [ ] Track input data quality
- [ ] Add logging helpers

### 4. Implement BaseCombiner Abstract Class
- [ ] Create abstract base class
- [ ] Define abstract `combine()` method
- [ ] Define abstract `fit()` method
- [ ] Define abstract `get_weights()` method
- [ ] Add `is_fitted()` property
- [ ] Add `name` and `version` properties

### 5. Implement Input Validation
- [ ] Create `_validate_predictions()` method
- [ ] Check prediction format and shape
- [ ] Validate model names and metadata
- [ ] Check for required horizons
- [ ] Validate time series alignment
- [ ] Raise informative errors

### 6. Implement Missing Value Handling
- [ ] Create `_handle_missing_predictions()` method
- [ ] Detect missing model predictions
- [ ] Strategy: exclude missing models
- [ ] Strategy: impute with model average
- [ ] Log missing prediction warnings
- [ ] Return cleaned predictions dict

### 7. Implement Metadata Tracking
- [ ] Create `_create_metadata()` method
- [ ] Track combination timestamp
- [ ] Record models used and excluded
- [ ] Calculate processing time
- [ ] Store combination parameters
- [ ] Return CombinationMetadata instance

### 8. Implement Weight Management
- [ ] Create `_initialize_weights()` method
- [ ] Store weights per time series
- [ ] Store weights per horizon
- [ ] Validate weight constraints (sum=1, ≥0)
- [ ] Support weight persistence
- [ ] Add weight versioning

### 9. Implement Performance Tracking
- [ ] Create `_track_performance()` method
- [ ] Calculate combination metrics
- [ ] Compare to individual models
- [ ] Store historical performance
- [ ] Generate performance summaries
- [ ] Support metric visualization

### 10. Implement Prediction Interval Support
- [ ] Create `_combine_intervals()` method
- [ ] Support interval propagation
- [ ] Calculate combined confidence bounds
- [ ] Validate interval coverage
- [ ] Handle missing intervals
- [ ] Return interval DataFrame

### 11. Implement Logging Integration
- [ ] Add structured logging
- [ ] Log combination start/end
- [ ] Log model availability
- [ ] Log weight changes
- [ ] Log performance metrics
- [ ] Add debug logging option

### 12. Implement Configuration Management
- [ ] Create `CombinerConfig` base class
- [ ] Support YAML/JSON configuration
- [ ] Validate configuration schema
- [ ] Handle default values
- [ ] Support configuration versioning
- [ ] Add configuration examples

### 13. Implement Model Serialization
- [ ] Create `save()` method
- [ ] Serialize weights and metadata
- [ ] Save configuration
- [ ] Support multiple formats (pickle, joblib)
- [ ] Add version compatibility

### 14. Implement Model Loading
- [ ] Create `load()` class method
- [ ] Deserialize weights and metadata
- [ ] Restore configuration
- [ ] Validate loaded state
- [ ] Handle version migration

### 15. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_base_combiner.py`
- [ ] Test abstract interface enforcement
- [ ] Test input validation
- [ ] Test missing value handling
- [ ] Test metadata tracking
- [ ] Test serialization/deserialization
- [ ] Create mock combiner implementation

### 16. Write Integration Tests
- [ ] Test with mock model predictions
- [ ] Test with 26 time series
- [ ] Test with multiple horizons
- [ ] Test missing prediction scenarios
- [ ] Test configuration loading

### 17. Create Usage Documentation
- [ ] Create `docs/combination_guide.md`
- [ ] Document BaseCombiner interface
- [ ] Provide implementation examples
- [ ] Document configuration options
- [ ] Add troubleshooting guide

---

## 💻 Implementation Details

### CombinationResult Dataclass

```python
"""Data structures for model combination."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class CombinationResult:
    """
    Result of model combination operation.
    
    Contains combined predictions, weights used, and metadata
    about the combination process.
    
    Attributes:
        combined_predictions: DataFrame with combined forecasts
        weights: Dictionary mapping model names to weights
        metadata: CombinationMetadata instance
        model_contributions: Optional breakdown by model
        confidence_intervals: Optional prediction intervals
    """
    
    combined_predictions: pd.DataFrame
    weights: Dict[str, float]
    metadata: 'CombinationMetadata'
    model_contributions: Optional[Dict[str, pd.DataFrame]] = None
    confidence_intervals: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """Validate result after initialization."""
        if self.combined_predictions.empty:
            raise ValueError("Combined predictions cannot be empty")
        
        if not self.weights:
            raise ValueError("Weights dictionary cannot be empty")
        
        # Validate weights sum to 1 (within tolerance)
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1, got {weight_sum}")
    
    def get_model_contribution(self, model_name: str) -> pd.DataFrame:
        """Get specific model's contribution to combination."""
        if self.model_contributions is None:
            raise ValueError("Model contributions not available")
        
        if model_name not in self.model_contributions:
            raise KeyError(f"Model {model_name} not in contributions")
        
        return self.model_contributions[model_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'combined_predictions': self.combined_predictions.to_dict(),
            'weights': self.weights,
            'metadata': self.metadata.to_dict(),
            'has_contributions': self.model_contributions is not None,
            'has_intervals': self.confidence_intervals is not None
        }


@dataclass
class CombinationMetadata:
    """
    Metadata about model combination process.
    
    Tracks which models were used, timing information,
    and quality metrics.
    """
    
    timestamp: datetime
    combination_method: str
    models_used: List[str]
    models_excluded: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def add_warning(self, warning: str):
        """Add warning message to metadata."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'combination_method': self.combination_method,
            'models_used': self.models_used,
            'models_excluded': self.models_excluded,
            'processing_time_seconds': self.processing_time_seconds,
            'performance_metrics': self.performance_metrics,
            'configuration': self.configuration,
            'warnings': self.warnings
        }
```

### BaseCombiner Abstract Class

```python
"""Base interface for model combination strategies."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import time

from src.models.combination.data_structures import (
    CombinationResult, CombinationMetadata
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseCombiner(ABC):
    """
    Abstract base class for model combination strategies.
    
    Defines standard interface for combining predictions from
    multiple forecasting models. Subclasses implement specific
    combination logic (averaging, weighted voting, selection, etc.).
    
    Key Features:
    - Standard combine() and fit() interface
    - Input validation and missing value handling
    - Metadata tracking and logging
    - Weight management and persistence
    - Support for prediction intervals
    
    Example:
        >>> class SimpleCombiner(BaseCombiner):
        ...     def combine(self, predictions, metadata):
        ...         # Implementation
        ...         pass
        ...     
        ...     def fit(self, train_predictions, train_targets):
        ...         # Implementation
        ...         pass
        >>> 
        >>> combiner = SimpleCombiner()
        >>> combiner.fit(train_preds, train_targets)
        >>> result = combiner.combine(test_preds)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize combiner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._weights: Dict[str, Dict[str, float]] = {}  # {ts: {model: weight}}
        self._fitted = False
        self._fit_timestamp: Optional[datetime] = None
        self._performance_history: List[Dict[str, Any]] = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return combiner name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Return combiner version."""
        pass
    
    @property
    def is_fitted(self) -> bool:
        """Check if combiner has been fitted."""
        return self._fitted
    
    @abstractmethod
    def combine(
        self,
        predictions: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CombinationResult:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: Dictionary mapping model names to prediction DataFrames
                Each DataFrame should have columns like 'pred_h0', 'pred_h1', etc.
            metadata: Optional metadata about predictions
        
        Returns:
            CombinationResult with combined predictions and metadata
        
        Raises:
            ValueError: If predictions format is invalid
            RuntimeError: If combiner not fitted (when required)
        """
        pass
    
    @abstractmethod
    def fit(
        self,
        train_predictions: Dict[str, pd.DataFrame],
        train_targets: pd.DataFrame,
        validation_predictions: Optional[Dict[str, pd.DataFrame]] = None,
        validation_targets: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Fit/train the combination strategy.
        
        Args:
            train_predictions: Dictionary mapping model names to predictions
            train_targets: True target values for training period
            validation_predictions: Optional validation predictions
            validation_targets: Optional validation targets
        
        Raises:
            ValueError: If input format is invalid
        """
        pass
    
    def get_weights(
        self,
        time_series: Optional[str] = None,
        horizon: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get combination weights.
        
        Args:
            time_series: Optional specific time series
            horizon: Optional specific horizon
        
        Returns:
            Dictionary mapping model names to weights
        """
        if not self.is_fitted:
            raise RuntimeError("Combiner not fitted")
        
        if time_series is not None:
            return self._weights.get(time_series, {})
        
        # Return average weights across all time series
        if not self._weights:
            return {}
        
        all_models = set()
        for ts_weights in self._weights.values():
            all_models.update(ts_weights.keys())
        
        avg_weights = {}
        for model in all_models:
            weights = [
                ts_weights.get(model, 0.0)
                for ts_weights in self._weights.values()
            ]
            avg_weights[model] = np.mean(weights)
        
        return avg_weights
    
    def _validate_predictions(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Validate prediction format and consistency.
        
        Args:
            predictions: Dictionary of model predictions
        
        Raises:
            ValueError: If predictions are invalid
        """
        if not predictions:
            raise ValueError("Predictions dictionary is empty")
        
        # Check each model's predictions
        first_model = next(iter(predictions.keys()))
        first_pred = predictions[first_model]
        
        if first_pred.empty:
            raise ValueError(f"Predictions for {first_model} are empty")
        
        # Get expected columns (pred_h0, pred_h1, etc.)
        expected_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        if not expected_cols:
            raise ValueError("No prediction columns found (expected pred_h0, pred_h1, etc.)")
        
        # Validate consistency across models
        for model_name, pred_df in predictions.items():
            if pred_df.empty:
                raise ValueError(f"Predictions for {model_name} are empty")
            
            # Check columns
            model_cols = [col for col in pred_df.columns if col.startswith('pred_h')]
            if set(model_cols) != set(expected_cols):
                raise ValueError(
                    f"Model {model_name} has inconsistent columns: "
                    f"expected {expected_cols}, got {model_cols}"
                )
            
            # Check index alignment
            if not pred_df.index.equals(first_pred.index):
                raise ValueError(
                    f"Model {model_name} has misaligned index"
                )
        
        logger.debug(f"Validated predictions from {len(predictions)} models")
    
    def _handle_missing_predictions(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Handle missing predictions from some models.
        
        Args:
            predictions: Dictionary of model predictions
        
        Returns:
            Tuple of (cleaned_predictions, excluded_models)
        """
        cleaned = {}
        excluded = []
        
        for model_name, pred_df in predictions.items():
            # Check for excessive missing values
            pred_cols = [col for col in pred_df.columns if col.startswith('pred_h')]
            missing_pct = pred_df[pred_cols].isna().mean().mean()
            
            if missing_pct > 0.5:  # More than 50% missing
                excluded.append(model_name)
                logger.warning(
                    f"Excluding {model_name}: {missing_pct:.1%} missing predictions"
                )
            else:
                cleaned[model_name] = pred_df
        
        if not cleaned:
            raise ValueError("All models excluded due to missing predictions")
        
        return cleaned, excluded
    
    def _create_metadata(
        self,
        models_used: List[str],
        models_excluded: List[str],
        processing_time: float,
        **kwargs
    ) -> CombinationMetadata:
        """
        Create combination metadata.
        
        Args:
            models_used: List of models used in combination
            models_excluded: List of models excluded
            processing_time: Processing time in seconds
            **kwargs: Additional metadata fields
        
        Returns:
            CombinationMetadata instance
        """
        metadata = CombinationMetadata(
            timestamp=datetime.now(),
            combination_method=self.name,
            models_used=models_used,
            models_excluded=models_excluded,
            processing_time_seconds=processing_time,
            configuration=self.config.copy()
        )
        
        # Add any additional fields
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        return metadata
    
    def save(self, filepath: str) -> None:
        """
        Save combiner state to file.
        
        Args:
            filepath: Path to save file
        """
        import joblib
        
        state = {
            'name': self.name,
            'version': self.version,
            'config': self.config,
            'weights': self._weights,
            'fitted': self._fitted,
            'fit_timestamp': self._fit_timestamp,
            'performance_history': self._performance_history
        }
        
        joblib.dump(state, filepath)
        logger.info(f"Saved combiner to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseCombiner':
        """
        Load combiner from file.
        
        Args:
            filepath: Path to load file
        
        Returns:
            Loaded combiner instance
        """
        import joblib
        
        state = joblib.load(filepath)
        
        # Create instance
        instance = cls(config=state['config'])
        instance._weights = state['weights']
        instance._fitted = state['fitted']
        instance._fit_timestamp = state['fit_timestamp']
        instance._performance_history = state['performance_history']
        
        logger.info(f"Loaded combiner from {filepath}")
        
        return instance
```

---

## 🧪 Testing & Validation

```python
"""Tests for BaseCombiner interface."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.models.combination.base_combiner import BaseCombiner
from src.models.combination.data_structures import CombinationResult, CombinationMetadata


class MockCombiner(BaseCombiner):
    """Mock combiner for testing."""
    
    @property
    def name(self) -> str:
        return "mock_combiner"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def combine(self, predictions, metadata=None):
        self._validate_predictions(predictions)
        
        # Simple average
        pred_cols = [col for col in predictions[next(iter(predictions))].columns 
                     if col.startswith('pred_h')]
        
        combined_df = pd.DataFrame(index=predictions[next(iter(predictions))].index)
        for col in pred_cols:
            values = np.mean([pred[col].values for pred in predictions.values()], axis=0)
            combined_df[col] = values
        
        weights = {model: 1.0/len(predictions) for model in predictions.keys()}
        
        metadata_obj = CombinationMetadata(
            timestamp=datetime.now(),
            combination_method=self.name,
            models_used=list(predictions.keys()),
            models_excluded=[]
        )
        
        return CombinationResult(
            combined_predictions=combined_df,
            weights=weights,
            metadata=metadata_obj
        )
    
    def fit(self, train_predictions, train_targets, 
            validation_predictions=None, validation_targets=None):
        self._fitted = True
        self._fit_timestamp = datetime.now()


@pytest.fixture
def sample_predictions():
    """Create sample model predictions."""
    dates = pd.date_range('2024-01-01', periods=48, freq='30min')
    
    predictions = {
        'lgbm': pd.DataFrame({
            'pred_h0': np.random.randn(48) * 100 + 1000,
            'pred_h1': np.random.randn(48) * 100 + 1000
        }, index=dates),
        'rf': pd.DataFrame({
            'pred_h0': np.random.randn(48) * 100 + 1000,
            'pred_h1': np.random.randn(48) * 100 + 1000
        }, index=dates)
    }
    
    return predictions


def test_base_combiner_interface():
    """Test BaseCombiner cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseCombiner()


def test_mock_combiner_initialization():
    """Test mock combiner initialization."""
    combiner = MockCombiner()
    
    assert combiner.name == "mock_combiner"
    assert combiner.version == "1.0.0"
    assert not combiner.is_fitted


def test_mock_combiner_combination(sample_predictions):
    """Test mock combiner combination."""
    combiner = MockCombiner()
    
    result = combiner.combine(sample_predictions)
    
    assert isinstance(result, CombinationResult)
    assert 'pred_h0' in result.combined_predictions.columns
    assert len(result.weights) == 2
    assert np.isclose(sum(result.weights.values()), 1.0)


def test_prediction_validation(sample_predictions):
    """Test prediction validation."""
    combiner = MockCombiner()
    
    # Valid predictions
    combiner._validate_predictions(sample_predictions)
    
    # Empty predictions
    with pytest.raises(ValueError, match="empty"):
        combiner._validate_predictions({})
    
    # Misaligned indices
    bad_predictions = sample_predictions.copy()
    bad_predictions['bad_model'] = sample_predictions['lgbm'].iloc[:10]
    
    with pytest.raises(ValueError, match="misaligned"):
        combiner._validate_predictions(bad_predictions)


def test_save_load(sample_predictions, tmp_path):
    """Test combiner save and load."""
    combiner = MockCombiner()
    combiner.fit(sample_predictions, None)
    
    filepath = tmp_path / "combiner.pkl"
    combiner.save(str(filepath))
    
    loaded = MockCombiner.load(str(filepath))
    
    assert loaded.name == combiner.name
    assert loaded.is_fitted == combiner.is_fitted
```

---

## 📝 Technical Notes

### Design Patterns
- Abstract Base Class for interface enforcement
- Template Method for common workflows
- Strategy Pattern for different combination approaches

### Weight Management
- Weights stored per time series for flexibility
- Support for global and local weights
- Validation ensures sum=1 and non-negative

---

## 🔗 Dependencies

**Depends On:**
- Epic-04: All model outputs available

**Blocks:**
- PC-037-05A: Simple Averaging Combiner
- PC-038-05A: Weighted Voting Combiner
- All other Epic-05A tickets

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] BaseCombiner abstract class implemented
- [ ] CombinationResult and CombinationMetadata defined
- [ ] Input validation working
- [ ] Missing value handling functional
- [ ] Serialization/deserialization working
- [ ] Mock combiner tests pass
- [ ] Unit tests >85% coverage
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-05A: Base Combination Strategies](../epics/Epic-05A.md)  
**Next:** [PC-037-05A: Simple Averaging Combiner](PC-037-05A-simple-averaging-combiner.md)
