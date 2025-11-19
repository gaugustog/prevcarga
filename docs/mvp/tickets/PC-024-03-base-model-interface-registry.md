# PC-024-03: Base Model Interface & Registry

**Ticket ID:** PC-024-03  
**Epic:** [Epic-03: Model Layer - End-to-End Models](../epics/Epic-03.md)  
**User Story:** US-1  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement the foundational model infrastructure including the `BaseModel` abstract class, `ModelRegistry` for centralized model management, and semantic versioning system. This provides a standardized interface for all forecasting models with robust versioning, metadata tracking, and thread-safe model loading.

**As an** ML engineer  
**I want** a standardized model interface with registry system  
**So that** I can manage multiple models consistently with versioning and metadata

---

## ✅ Acceptance Criteria

- [ ] `BaseModel` abstract class defines common interface for all models
- [ ] `ModelRegistry` manages model registration and discovery
- [ ] Semantic versioning system tracks model evolution
- [ ] Model metadata includes training timestamp, performance metrics, feature dependencies
- [ ] Registry supports model comparison and selection
- [ ] Thread-safe model loading and caching
- [ ] Model discovery by name, version, or criteria
- [ ] Comprehensive tests for all registry operations

---

## 🔧 Implementation Tasks

### 1. Create Base Model Module Structure
- [ ] Create `src/models/` directory
- [ ] Create `src/models/base/` subdirectory
- [ ] Create `src/models/base/__init__.py`
- [ ] Create `src/models/base/model.py`
- [ ] Create `src/models/base/registry.py`
- [ ] Add module docstrings

### 2. Implement BaseModel Abstract Class
- [ ] Create `BaseModel` abstract base class
- [ ] Add `@property` abstract methods: `name`, `version`, `supported_horizons`
- [ ] Add abstract method: `fit(X, y, config)`
- [ ] Add abstract method: `predict(X, horizons)`
- [ ] Add abstract method: `get_feature_importance()`
- [ ] Add optional method: `get_metadata()`
- [ ] Add `_is_fitted` internal flag
- [ ] Document expected behavior for each abstract method

### 3. Implement Model Metadata Schema
- [ ] Create `ModelMetadata` Pydantic model
- [ ] Add `model_id` unique identifier field
- [ ] Add `model_name` and `model_version` fields
- [ ] Add `model_class` class name field
- [ ] Add `trained_at` timestamp field
- [ ] Add `supported_horizons` list field
- [ ] Add `feature_dependencies` list field
- [ ] Add `performance_metrics` dict field
- [ ] Add `training_config` dict field
- [ ] Add `custom_metadata` dict for extensibility

### 4. Implement Semantic Versioning
- [ ] Create `SemanticVersion` class
- [ ] Parse version strings (major.minor.patch)
- [ ] Implement version comparison operators
- [ ] Support version constraints (>=, <, ~, ^)
- [ ] Validate version string format
- [ ] Handle pre-release versions (alpha, beta, rc)

### 5. Implement ModelRegistry Core
- [ ] Create `ModelRegistry` class
- [ ] Initialize with storage path
- [ ] Add `_models` internal registry dict
- [ ] Add `_lock` threading lock for thread-safety
- [ ] Implement singleton pattern (optional)
- [ ] Create storage directory structure

### 6. Implement Model Registration
- [ ] Create `register_model()` method
- [ ] Generate unique model ID
- [ ] Validate model is fitted
- [ ] Create model storage directory
- [ ] Serialize model to disk
- [ ] Save metadata as JSON
- [ ] Update in-memory registry
- [ ] Return model ID

### 7. Implement Model Loading
- [ ] Create `load_model()` method
- [ ] Check in-memory cache first
- [ ] Load metadata from disk
- [ ] Dynamically import model class
- [ ] Deserialize model object
- [ ] Validate model integrity
- [ ] Cache in memory with lock
- [ ] Return loaded model

### 8. Implement Model Discovery
- [ ] Create `list_models()` method
- [ ] Filter by model name (optional)
- [ ] Filter by version constraint (optional)
- [ ] Filter by metadata criteria (optional)
- [ ] Return list of model metadata
- [ ] Sort by version or training date

### 9. Implement Model Comparison
- [ ] Create `compare_models()` method
- [ ] Load multiple models by IDs
- [ ] Extract performance metrics
- [ ] Create comparison DataFrame
- [ ] Rank models by metrics
- [ ] Return comparison results

### 10. Implement Model Selection
- [ ] Create `get_best_model()` method
- [ ] Filter by criteria (horizon, area, etc.)
- [ ] Select by performance metric
- [ ] Return best model or model ID
- [ ] Handle ties with timestamp

### 11. Implement Model Deregistration
- [ ] Create `deregister_model()` method
- [ ] Remove from in-memory registry
- [ ] Optionally delete from disk
- [ ] Update registry index
- [ ] Handle model in use (warnings)

### 12. Implement Model Class Registry
- [ ] Create `_model_classes` class registry
- [ ] Implement `register_model_class()` decorator
- [ ] Dynamic class lookup by name
- [ ] Support custom model classes
- [ ] Validate class inherits from BaseModel

### 13. Implement Registry Persistence
- [ ] Create registry index file
- [ ] Save index on registration
- [ ] Load index on initialization
- [ ] Handle index corruption
- [ ] Rebuild index from disk

### 14. Write Comprehensive Tests
- [ ] Create `tests/models/base/test_model.py`
- [ ] Test BaseModel interface compliance
- [ ] Create mock model implementation
- [ ] Test abstract methods enforcement
- [ ] Test metadata generation

### 15. Write Registry Tests
- [ ] Create `tests/models/base/test_registry.py`
- [ ] Test model registration and loading
- [ ] Test thread-safe operations
- [ ] Test model discovery and filtering
- [ ] Test model comparison
- [ ] Test deregistration
- [ ] Test registry persistence
- [ ] Test cache invalidation

### 16. Write Versioning Tests
- [ ] Test semantic version parsing
- [ ] Test version comparison
- [ ] Test version constraints
- [ ] Test invalid version handling

### 17. Create Usage Examples
- [ ] Create `examples/model_registry_demo.py`
- [ ] Show model registration workflow
- [ ] Show model loading and caching
- [ ] Show model discovery
- [ ] Show model comparison

---

## 💻 Implementation Details

### BaseModel Abstract Class

```python
"""Base model interface for all forecasting models."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    Provides standardized interface for model training, prediction,
    and metadata management. All forecasting models must inherit
    from this class and implement the required abstract methods.
    
    Example:
        >>> class MyModel(BaseModel):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_model"
        ...     
        ...     def fit(self, X, y, config):
        ...         # Training logic
        ...         return self
    """
    
    def __init__(self):
        """Initialize base model."""
        self._is_fitted = False
        self._feature_names = []
        self._training_metadata = {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Model name for identification.
        
        Returns:
            Unique model name (e.g., "lgbm", "random_forest")
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """
        Model version using semantic versioning.
        
        Returns:
            Version string (e.g., "1.0.0", "2.1.0-beta")
        """
        pass
    
    @property
    @abstractmethod
    def supported_horizons(self) -> List[int]:
        """
        List of supported forecast horizons.
        
        Returns:
            List of horizon values in days (e.g., [0, 1] for D+0, D+1)
        """
        pass
    
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ) -> 'BaseModel':
        """
        Train the model on provided data.
        
        Args:
            X: Feature DataFrame with predictors
            y: Target Series with values to predict
            config: Training configuration dictionary
        
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Generate predictions for specified horizons.
        
        Args:
            X: Feature DataFrame for prediction
            horizons: List of horizons to predict
        
        Returns:
            DataFrame with predictions columns (pred_h0, pred_h1, etc.)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names used in training."""
        return self._feature_names
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'name': self.name,
            'version': self.version,
            'supported_horizons': self.supported_horizons,
            'is_fitted': self._is_fitted,
            'feature_names': self._feature_names,
            'training_metadata': self._training_metadata
        }
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Default implementation using joblib. Can be overridden.
        
        Args:
            path: File path to save model
        """
        import joblib
        joblib.dump(self, path, compress=3)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Default implementation using joblib. Can be overridden.
        
        Args:
            path: File path to load model from
        
        Returns:
            Loaded model instance
        """
        import joblib
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
    
    def __repr__(self) -> str:
        """String representation of model."""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}', fitted={self._is_fitted})"
```

### Model Metadata Schema

```python
"""Model metadata schema."""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Metadata for registered models."""
    
    model_id: str = Field(
        ...,
        description="Unique model identifier"
    )
    model_name: str = Field(
        ...,
        description="Model name"
    )
    model_version: str = Field(
        ...,
        description="Semantic version"
    )
    model_class: str = Field(
        ...,
        description="Model class name"
    )
    trained_at: datetime = Field(
        default_factory=datetime.now,
        description="Training timestamp"
    )
    registered_at: datetime = Field(
        default_factory=datetime.now,
        description="Registration timestamp"
    )
    supported_horizons: List[int] = Field(
        default_factory=list,
        description="Supported forecast horizons"
    )
    feature_dependencies: List[str] = Field(
        default_factory=list,
        description="Required feature names"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Model performance metrics"
    )
    training_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training configuration"
    )
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata fields"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

### Model Registry Implementation

```python
"""Model registry for centralized model management."""
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import json

from src.models.base.model import BaseModel
from src.models.base.metadata import ModelMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Central registry for model management.
    
    Provides thread-safe model registration, loading, and discovery
    with automatic caching and metadata tracking.
    
    Example:
        >>> registry = ModelRegistry(storage_path="models/")
        >>> 
        >>> # Register a trained model
        >>> model_id = registry.register_model(
        ...     model=trained_model,
        ...     metadata={'area': 'SE', 'horizon': 1}
        ... )
        >>> 
        >>> # Load model later
        >>> loaded_model = registry.load_model(model_id)
    """
    
    # Class-level model class registry
    _model_classes: Dict[str, Type[BaseModel]] = {}
    
    def __init__(self, storage_path: str = "models/registry"):
        """
        Initialize model registry.
        
        Args:
            storage_path: Base directory for model storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._loaded_models: Dict[str, BaseModel] = {}
        self._metadata_cache: Dict[str, ModelMetadata] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load registry index
        self._load_registry_index()
        
        logger.info(f"ModelRegistry initialized at {self.storage_path}")
    
    @classmethod
    def register_model_class(cls, model_class: Type[BaseModel]) -> Type[BaseModel]:
        """
        Register a model class for dynamic loading.
        
        Decorator for model class registration.
        
        Example:
            >>> @ModelRegistry.register_model_class
            >>> class MyModel(BaseModel):
            ...     pass
        """
        class_name = model_class.__name__
        cls._model_classes[class_name] = model_class
        logger.debug(f"Registered model class: {class_name}")
        return model_class
    
    def register_model(
        self,
        model: BaseModel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a trained model with metadata.
        
        Args:
            model: Trained model instance
            metadata: Additional metadata dictionary
        
        Returns:
            Unique model ID
        """
        if not model.is_fitted():
            raise ValueError("Model must be fitted before registration")
        
        # Generate unique model ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"{model.name}_v{model.version}_{timestamp}"
        
        with self._lock:
            # Create model directory
            model_dir = self.storage_path / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            model.save(str(model_path))
            
            # Prepare metadata
            model_metadata = ModelMetadata(
                model_id=model_id,
                model_name=model.name,
                model_version=model.version,
                model_class=model.__class__.__name__,
                trained_at=datetime.now(),
                registered_at=datetime.now(),
                supported_horizons=model.supported_horizons,
                feature_dependencies=model.get_feature_names(),
                performance_metrics=metadata.get('performance_metrics', {}) if metadata else {},
                training_config=metadata.get('training_config', {}) if metadata else {},
                custom_metadata=metadata or {}
            )
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata.dict(), f, indent=2, default=str)
            
            # Update cache
            self._metadata_cache[model_id] = model_metadata
            
            # Update registry index
            self._save_registry_index()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def load_model(self, model_id: str, use_cache: bool = True) -> BaseModel:
        """
        Load model from registry.
        
        Args:
            model_id: Model identifier
            use_cache: Use in-memory cache if available
        
        Returns:
            Loaded model instance
        """
        # Check cache
        if use_cache and model_id in self._loaded_models:
            logger.debug(f"Model loaded from cache: {model_id}")
            return self._loaded_models[model_id]
        
        model_dir = self.storage_path / model_id
        
        if not model_dir.exists():
            raise ValueError(f"Model not found: {model_id}")
        
        # Load metadata
        metadata = self._load_metadata(model_id)
        
        # Get model class
        model_class = self._get_model_class(metadata.model_class)
        
        # Load model
        model_path = model_dir / "model.pkl"
        model = model_class.load(str(model_path))
        
        # Cache model
        with self._lock:
            self._loaded_models[model_id] = model
        
        logger.info(f"Loaded model: {model_id}")
        return model
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        version_constraint: Optional[str] = None,
        **filters
    ) -> List[ModelMetadata]:
        """
        List registered models with optional filtering.
        
        Args:
            model_name: Filter by model name
            version_constraint: Filter by version (e.g., ">=1.0.0")
            **filters: Additional metadata filters
        
        Returns:
            List of model metadata
        """
        models = []
        
        for model_id in self._metadata_cache:
            metadata = self._metadata_cache[model_id]
            
            # Apply filters
            if model_name and metadata.model_name != model_name:
                continue
            
            if version_constraint:
                # TODO: Implement version constraint checking
                pass
            
            # Apply custom filters
            if filters:
                match = all(
                    metadata.custom_metadata.get(k) == v
                    for k, v in filters.items()
                )
                if not match:
                    continue
            
            models.append(metadata)
        
        # Sort by registration time (newest first)
        models.sort(key=lambda m: m.registered_at, reverse=True)
        
        return models
    
    def get_best_model(
        self,
        metric: str = "mae",
        model_name: Optional[str] = None,
        **filters
    ) -> Optional[str]:
        """
        Get best performing model by metric.
        
        Args:
            metric: Performance metric name
            model_name: Filter by model name
            **filters: Additional filters
        
        Returns:
            Best model ID or None
        """
        models = self.list_models(model_name=model_name, **filters)
        
        if not models:
            return None
        
        # Filter models with the metric
        models_with_metric = [
            m for m in models
            if metric in m.performance_metrics
        ]
        
        if not models_with_metric:
            return None
        
        # Sort by metric (lower is better for mae, mape, rmse)
        best_model = min(
            models_with_metric,
            key=lambda m: m.performance_metrics[metric]
        )
        
        return best_model.model_id
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            model_ids: List of model IDs to compare
        
        Returns:
            DataFrame with model comparison
        """
        import pandas as pd
        
        comparison_data = []
        
        for model_id in model_ids:
            metadata = self._load_metadata(model_id)
            
            row = {
                'model_id': model_id,
                'model_name': metadata.model_name,
                'version': metadata.model_version,
                'trained_at': metadata.trained_at,
                **metadata.performance_metrics
            }
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def deregister_model(
        self,
        model_id: str,
        delete_files: bool = False
    ) -> None:
        """
        Remove model from registry.
        
        Args:
            model_id: Model identifier
            delete_files: Also delete model files
        """
        with self._lock:
            # Remove from cache
            self._loaded_models.pop(model_id, None)
            self._metadata_cache.pop(model_id, None)
            
            # Delete files
            if delete_files:
                model_dir = self.storage_path / model_id
                if model_dir.exists():
                    import shutil
                    shutil.rmtree(model_dir)
            
            # Update index
            self._save_registry_index()
        
        logger.info(f"Deregistered model: {model_id}")
    
    def _load_metadata(self, model_id: str) -> ModelMetadata:
        """Load metadata for model."""
        if model_id in self._metadata_cache:
            return self._metadata_cache[model_id]
        
        metadata_path = self.storage_path / model_id / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = ModelMetadata(**metadata_dict)
        
        with self._lock:
            self._metadata_cache[model_id] = metadata
        
        return metadata
    
    def _get_model_class(self, class_name: str) -> Type[BaseModel]:
        """Get model class by name."""
        if class_name not in self._model_classes:
            raise ValueError(f"Model class not registered: {class_name}")
        
        return self._model_classes[class_name]
    
    def _load_registry_index(self) -> None:
        """Load registry index from disk."""
        index_path = self.storage_path / "registry_index.json"
        
        if not index_path.exists():
            return
        
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            
            # Load metadata for each model
            for model_id in index_data.get('model_ids', []):
                try:
                    self._load_metadata(model_id)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_id}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load registry index: {e}")
    
    def _save_registry_index(self) -> None:
        """Save registry index to disk."""
        index_path = self.storage_path / "registry_index.json"
        
        index_data = {
            'model_ids': list(self._metadata_cache.keys()),
            'updated_at': datetime.now().isoformat()
        }
        
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for base model interface and registry."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from src.models.base.model import BaseModel
from src.models.base.registry import ModelRegistry


# Mock model implementation for testing
@ModelRegistry.register_model_class
class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self):
        super().__init__()
        self.model_data = None
    
    @property
    def name(self) -> str:
        return "mock_model"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return [0, 1]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> 'MockModel':
        self.model_data = {'mean': y.mean()}
        self._feature_names = X.columns.tolist()
        self._is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        predictions = {}
        for horizon in horizons:
            predictions[f'pred_h{horizon}'] = [self.model_data['mean']] * len(X)
        return pd.DataFrame(predictions, index=X.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        return {name: 1.0 / len(self._feature_names) for name in self._feature_names}


@pytest.fixture
def registry(tmp_path):
    """Create temporary registry."""
    return ModelRegistry(storage_path=str(tmp_path / "test_registry"))


@pytest.fixture
def trained_model():
    """Create trained mock model."""
    X = pd.DataFrame({'feature1': np.random.randn(100), 'feature2': np.random.randn(100)})
    y = pd.Series(np.random.randn(100))
    
    model = MockModel()
    model.fit(X, y, {})
    
    return model


def test_base_model_interface():
    """Test BaseModel interface."""
    model = MockModel()
    
    assert model.name == "mock_model"
    assert model.version == "1.0.0"
    assert model.supported_horizons == [0, 1]
    assert not model.is_fitted()


def test_model_training(trained_model):
    """Test model training."""
    assert trained_model.is_fitted()
    assert len(trained_model.get_feature_names()) == 2


def test_model_prediction(trained_model):
    """Test model prediction."""
    X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    
    predictions = trained_model.predict(X_test, horizons=[0, 1])
    
    assert 'pred_h0' in predictions.columns
    assert 'pred_h1' in predictions.columns
    assert len(predictions) == 3


def test_registry_registration(registry, trained_model):
    """Test model registration."""
    model_id = registry.register_model(
        model=trained_model,
        metadata={'area': 'SE', 'mae': 10.5}
    )
    
    assert model_id.startswith('mock_model_v1.0.0_')


def test_registry_loading(registry, trained_model):
    """Test model loading."""
    model_id = registry.register_model(model=trained_model)
    
    loaded_model = registry.load_model(model_id)
    
    assert loaded_model.is_fitted()
    assert loaded_model.name == "mock_model"


def test_registry_caching(registry, trained_model):
    """Test model caching."""
    model_id = registry.register_model(model=trained_model)
    
    # First load
    model1 = registry.load_model(model_id)
    
    # Second load (from cache)
    model2 = registry.load_model(model_id)
    
    # Should be same instance
    assert model1 is model2


def test_registry_list_models(registry, trained_model):
    """Test model listing."""
    model_id1 = registry.register_model(trained_model, {'area': 'SE'})
    model_id2 = registry.register_model(trained_model, {'area': 'NE'})
    
    all_models = registry.list_models()
    assert len(all_models) == 2
    
    se_models = registry.list_models(area='SE')
    assert len(se_models) == 1


def test_registry_best_model(registry, trained_model):
    """Test best model selection."""
    model_id1 = registry.register_model(trained_model, {'performance_metrics': {'mae': 10.0}})
    model_id2 = registry.register_model(trained_model, {'performance_metrics': {'mae': 5.0}})
    
    best_id = registry.get_best_model(metric='mae')
    
    assert best_id == model_id2


def test_registry_persistence(tmp_path, trained_model):
    """Test registry persistence."""
    storage_path = tmp_path / "test_registry"
    
    # First registry instance
    registry1 = ModelRegistry(storage_path=str(storage_path))
    model_id = registry1.register_model(trained_model)
    
    # Second registry instance (simulating restart)
    registry2 = ModelRegistry(storage_path=str(storage_path))
    
    # Should load from index
    models = registry2.list_models()
    assert len(models) == 1
    assert models[0].model_id == model_id


def test_unfitted_model_registration(registry):
    """Test registration of unfitted model fails."""
    model = MockModel()
    
    with pytest.raises(ValueError, match="must be fitted"):
        registry.register_model(model)


def test_model_deregistration(registry, trained_model):
    """Test model deregistration."""
    model_id = registry.register_model(trained_model)
    
    registry.deregister_model(model_id, delete_files=True)
    
    models = registry.list_models()
    assert len(models) == 0
```

---

## 📝 Technical Notes

### Thread Safety
- Registry uses `threading.Lock()` for all cache operations
- Model loading is synchronized to prevent race conditions
- Safe for concurrent access from multiple threads

### Caching Strategy
- In-memory cache for loaded models
- Metadata cache for fast listing
- Lazy loading: models loaded on demand
- Cache invalidation on deregistration

### Versioning Best Practices
- Use semantic versioning: MAJOR.MINOR.PATCH
- Increment MAJOR for breaking changes
- Increment MINOR for new features
- Increment PATCH for bug fixes

---

## 🔗 Dependencies

**Depends On:**
- Python 3.11+
- Core infrastructure (logging, configuration)

**External Dependencies:**
- `pydantic>=2.0.0`
- `pandas>=2.0.0`
- `joblib>=1.3.0`

**Blocks:**
- PC-025-03: LGBM Model (needs base interface)
- PC-026-03: Random Forest Model (needs base interface)
- All other Epic-03 tickets

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] BaseModel abstract class fully documented
- [ ] ModelRegistry supports all operations
- [ ] Thread-safe model loading validated
- [ ] Unit tests pass with >90% coverage
- [ ] Integration tests with mock models
- [ ] Documentation complete with examples
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next:** [PC-025-03: LGBM Model Implementation](PC-025-03-lgbm-model.md)
