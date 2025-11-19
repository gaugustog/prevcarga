# PC-028-03: Model Serialization & Version Management

**Ticket ID:** PC-028-03  
**Epic:** [Epic-03: Model Layer - End-to-End Models](../epics/Epic-03.md)  
**User Story:** US-5  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement robust model serialization with complete metadata capture, semantic versioning system, and integrity validation. Enables model comparison, rollback capabilities, and efficient storage of large models with compression.

**As a** MLOps engineer  
**I want** robust model serialization with versioning and validation  
**So that** I can manage model lifecycle and ensure deployment integrity

---

## ✅ Acceptance Criteria

- [ ] Models serialize with complete metadata and dependencies
- [ ] Semantic versioning tracks model evolution and compatibility
- [ ] Serialization includes feature names, preprocessing steps, and performance metrics
- [ ] Supports model comparison and rollback capabilities
- [ ] Handles large models efficiently with compression
- [ ] Validates model integrity on loading (hash checking)
- [ ] Supports models > 1GB in size
- [ ] Load time < 5 seconds for typical models
- [ ] Comprehensive tests with version migration scenarios

---

## 🔧 Implementation Tasks

### 1. Create Serialization Module Structure
- [ ] Create `src/models/serialization/model_serializer.py`
- [ ] Create `src/models/serialization/version_manager.py`
- [ ] Create `src/models/serialization/schemas.py`
- [ ] Add module docstrings

### 2. Implement Serialization Metadata Schema
- [ ] Create `SerializedModelMetadata` Pydantic model
- [ ] Add `model_id` field with UUID
- [ ] Add `model_type` field (lgbm, rf, etc.)
- [ ] Add `version` field (semantic version)
- [ ] Add `created_at` timestamp
- [ ] Add `serialized_at` timestamp
- [ ] Add `serialization_format` field
- [ ] Add `compression` field
- [ ] Add `checksum` field (SHA256)
- [ ] Add `file_size_bytes` field
- [ ] Add `python_version` field
- [ ] Add `dependencies` dict

### 3. Implement Training Metadata Schema
- [ ] Extend `ModelMetadata` from PC-024
- [ ] Add `training_config` dict
- [ ] Add `feature_names` list
- [ ] Add `preprocessing_steps` list
- [ ] Add `training_duration_seconds` field
- [ ] Add `n_samples_train` field
- [ ] Add `n_features` field
- [ ] Add `performance_metrics` dict (MAPE, RMSE, MAE)
- [ ] Add `cv_scores` dict
- [ ] Add `best_hyperparams` dict (if optimized)

### 4. Implement ModelSerializer Class
- [ ] Create static `save_model()` method
- [ ] Accept model, path, metadata
- [ ] Generate model ID if not provided
- [ ] Calculate file checksum
- [ ] Compress model data
- [ ] Save metadata separately
- [ ] Create version manifest
- [ ] Return serialization info

### 5. Implement Model Loading
- [ ] Create static `load_model()` method
- [ ] Accept path and validate existence
- [ ] Load and validate metadata
- [ ] Verify checksum integrity
- [ ] Decompress model data
- [ ] Restore model state
- [ ] Log loading statistics
- [ ] Return loaded model

### 6. Implement Integrity Validation
- [ ] Create `_calculate_checksum()` method
- [ ] Use SHA256 for file hashing
- [ ] Include model weights and metadata
- [ ] Store checksum in metadata
- [ ] Create `validate_integrity()` method
- [ ] Compare checksums on load
- [ ] Raise error if mismatch

### 7. Implement Compression Support
- [ ] Use joblib with compression level 3
- [ ] Support gzip, lz4, zstd
- [ ] Add compression parameter to save
- [ ] Benchmark compression ratios
- [ ] Log compression statistics

### 8. Implement VersionManager Class
- [ ] Create `create_version()` method
- [ ] Parse semantic version string (major.minor.patch)
- [ ] Generate version from timestamp if needed
- [ ] Validate version format
- [ ] Store version metadata

### 9. Implement Version Comparison
- [ ] Create `compare_versions()` method
- [ ] Parse version strings
- [ ] Compare major.minor.patch
- [ ] Return comparison result (-1, 0, 1)
- [ ] Handle pre-release versions

### 10. Implement Version History
- [ ] Create `get_version_history()` method
- [ ] Scan model directory for versions
- [ ] Parse version from filenames
- [ ] Sort chronologically
- [ ] Return DataFrame with metadata

### 11. Implement Model Rollback
- [ ] Create `rollback_to_version()` method
- [ ] Validate target version exists
- [ ] Copy versioned model to active
- [ ] Update deployment manifest
- [ ] Log rollback event

### 12. Implement Model Comparison
- [ ] Create `compare_models()` method
- [ ] Load metadata for both models
- [ ] Compare performance metrics
- [ ] Compare feature sets
- [ ] Compare configurations
- [ ] Return comparison report

### 13. Implement Dependency Tracking
- [ ] Create `_capture_dependencies()` method
- [ ] Extract Python version
- [ ] Get installed package versions
- [ ] Store in metadata
- [ ] Validate on load

### 14. Implement Model Registry Integration
- [ ] Create `register_serialized_model()` method
- [ ] Extract metadata from serialized file
- [ ] Register with ModelRegistry
- [ ] Update registry index
- [ ] Handle version conflicts

### 15. Handle Large Model Optimization
- [ ] Implement chunked serialization
- [ ] Use memory-efficient compression
- [ ] Add progress callback for large files
- [ ] Implement streaming load
- [ ] Test with >1GB models

### 16. Write Comprehensive Tests
- [ ] Create `tests/models/serialization/test_serializer.py`
- [ ] Test save/load round-trip
- [ ] Test integrity validation
- [ ] Test compression formats
- [ ] Test version parsing
- [ ] Test version comparison
- [ ] Test rollback functionality
- [ ] Test large model handling
- [ ] Test corrupted file detection

### 17. Write Integration Tests
- [ ] Test with LGBM model
- [ ] Test with Random Forest model
- [ ] Test version migration
- [ ] Test multi-model comparison

### 18. Create Usage Examples
- [ ] Create `examples/model_versioning_demo.py`
- [ ] Show model save with versioning
- [ ] Show integrity validation
- [ ] Show version comparison
- [ ] Show rollback scenario

---

## 💻 Implementation Details

### Serialization Schemas

```python
"""Schemas for model serialization."""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class SerializedModelMetadata(BaseModel):
    """Metadata for serialized model."""
    
    # Identity
    model_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique model identifier"
    )
    model_type: str = Field(
        description="Model type (lgbm, random_forest, etc.)"
    )
    model_name: str = Field(
        description="Human-readable model name"
    )
    
    # Versioning
    version: str = Field(
        description="Semantic version (major.minor.patch)"
    )
    
    # Timestamps
    created_at: datetime = Field(
        description="Model training timestamp"
    )
    serialized_at: datetime = Field(
        default_factory=datetime.now,
        description="Serialization timestamp"
    )
    
    # Serialization info
    serialization_format: str = Field(
        default="joblib",
        description="Serialization format"
    )
    compression: str = Field(
        default="gzip",
        description="Compression algorithm"
    )
    compression_level: int = Field(
        default=3,
        description="Compression level (0-9)"
    )
    
    # Integrity
    checksum: str = Field(
        description="SHA256 checksum of model file"
    )
    file_size_bytes: int = Field(
        description="Serialized file size"
    )
    
    # Environment
    python_version: str = Field(
        description="Python version used"
    )
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Package name to version mapping"
    )


class TrainingMetadata(BaseModel):
    """Extended training metadata."""
    
    # Training data
    n_samples_train: int = Field(
        description="Number of training samples"
    )
    n_samples_val: Optional[int] = Field(
        default=None,
        description="Number of validation samples"
    )
    n_features: int = Field(
        description="Number of features"
    )
    feature_names: List[str] = Field(
        description="List of feature names"
    )
    
    # Training process
    training_config: Dict[str, Any] = Field(
        description="Training configuration"
    )
    preprocessing_steps: List[str] = Field(
        default_factory=list,
        description="Preprocessing steps applied"
    )
    training_duration_seconds: float = Field(
        description="Training duration"
    )
    
    # Performance
    performance_metrics: Dict[str, float] = Field(
        description="Performance metrics (MAPE, RMSE, etc.)"
    )
    cv_scores: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Cross-validation scores"
    )
    
    # Optimization
    hyperparameter_optimization: bool = Field(
        default=False,
        description="Whether hyperparameters were optimized"
    )
    best_hyperparams: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Best hyperparameters found"
    )
```

### Model Serializer Implementation

```python
"""Model serialization with versioning and integrity validation."""
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
import joblib
import json
import sys

from src.models.base.model import BaseModel
from src.models.serialization.schemas import (
    SerializedModelMetadata,
    TrainingMetadata
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelSerializer:
    """
    Serializes and loads models with comprehensive metadata.
    
    Features:
    - Complete metadata capture
    - Integrity validation via checksums
    - Compression support
    - Dependency tracking
    - Version management
    
    Example:
        >>> model = LGBMModel()
        >>> model.fit(X, y, config)
        >>> 
        >>> metadata = TrainingMetadata(
        ...     n_samples_train=len(X),
        ...     n_features=len(X.columns),
        ...     feature_names=X.columns.tolist(),
        ...     training_config=config,
        ...     performance_metrics={'mape': 2.5, 'rmse': 45.2}
        ... )
        >>> 
        >>> ModelSerializer.save_model(
        ...     model=model,
        ...     path='models/lgbm_v1.0.0.pkl',
        ...     training_metadata=metadata,
        ...     version='1.0.0'
        ... )
    """
    
    @staticmethod
    def save_model(
        model: BaseModel,
        path: str,
        training_metadata: TrainingMetadata,
        version: str,
        compression: str = "gzip",
        compression_level: int = 3
    ) -> SerializedModelMetadata:
        """
        Save model with comprehensive metadata.
        
        Args:
            model: Fitted model to save
            path: Target file path
            training_metadata: Training metadata
            version: Semantic version string
            compression: Compression algorithm
            compression_level: Compression level (0-9)
        
        Returns:
            Serialization metadata
        """
        if not model.is_fitted():
            raise ValueError("Model must be fitted before saving")
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {path} with version {version}")
        
        # Capture dependencies
        dependencies = ModelSerializer._capture_dependencies()
        
        # Save model
        joblib.dump(
            model,
            path,
            compress=(compression, compression_level)
        )
        
        # Calculate checksum
        checksum = ModelSerializer._calculate_checksum(path)
        file_size = path_obj.stat().st_size
        
        # Create serialization metadata
        serialization_metadata = SerializedModelMetadata(
            model_type=model.name,
            model_name=f"{model.name}_v{version}",
            version=version,
            created_at=training_metadata.training_config.get(
                'training_started_at',
                datetime.now()
            ),
            serialization_format="joblib",
            compression=compression,
            compression_level=compression_level,
            checksum=checksum,
            file_size_bytes=file_size,
            python_version=sys.version,
            dependencies=dependencies
        )
        
        # Save metadata separately
        metadata_path = path_obj.with_suffix('.json')
        combined_metadata = {
            'serialization': serialization_metadata.dict(),
            'training': training_metadata.dict()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(combined_metadata, f, indent=2, default=str)
        
        logger.info(
            f"Model saved successfully: "
            f"{file_size / 1024 / 1024:.2f} MB, "
            f"checksum: {checksum[:8]}..."
        )
        
        return serialization_metadata
    
    @staticmethod
    def load_model(
        path: str,
        validate_integrity: bool = True
    ) -> BaseModel:
        """
        Load model with integrity validation.
        
        Args:
            path: Model file path
            validate_integrity: Whether to validate checksum
        
        Returns:
            Loaded model
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        logger.info(f"Loading model from {path}")
        
        # Load metadata
        metadata_path = path_obj.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            serialization_meta = SerializedModelMetadata(**metadata['serialization'])
            
            # Validate integrity
            if validate_integrity:
                current_checksum = ModelSerializer._calculate_checksum(path)
                if current_checksum != serialization_meta.checksum:
                    raise ValueError(
                        f"Checksum mismatch! "
                        f"Expected: {serialization_meta.checksum[:8]}..., "
                        f"Got: {current_checksum[:8]}..."
                    )
                logger.info("Integrity validation passed")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
        
        # Load model
        model = joblib.load(path)
        
        logger.info(f"Model loaded successfully: {model.name} v{model.version}")
        
        return model
    
    @staticmethod
    def _calculate_checksum(path: str) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        
        with open(path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    @staticmethod
    def _capture_dependencies() -> Dict[str, str]:
        """Capture installed package versions."""
        try:
            import pkg_resources
            
            dependencies = {}
            for package in ['pandas', 'numpy', 'scikit-learn', 
                          'lightgbm', 'pydantic', 'joblib']:
                try:
                    version = pkg_resources.get_distribution(package).version
                    dependencies[package] = version
                except:
                    pass
            
            return dependencies
        except:
            return {}
```

### Version Manager Implementation

```python
"""Model version management."""
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VersionManager:
    """
    Manages model versioning using semantic versioning.
    
    Semantic versioning format: MAJOR.MINOR.PATCH
    - MAJOR: Incompatible API/architecture changes
    - MINOR: Backward-compatible functionality additions
    - PATCH: Backward-compatible bug fixes
    
    Example:
        >>> vm = VersionManager(models_dir='models/lgbm')
        >>> 
        >>> # Get version history
        >>> history = vm.get_version_history()
        >>> 
        >>> # Compare versions
        >>> result = vm.compare_versions('1.2.0', '1.1.5')
        >>> # result = 1 (first is newer)
        >>> 
        >>> # Get latest version
        >>> latest = vm.get_latest_version()
    """
    
    VERSION_PATTERN = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')
    
    def __init__(self, models_dir: str):
        """
        Initialize version manager.
        
        Args:
            models_dir: Directory containing versioned models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def parse_version(version_str: str) -> Tuple[int, int, int]:
        """
        Parse semantic version string.
        
        Args:
            version_str: Version string (e.g., '1.2.3')
        
        Returns:
            Tuple of (major, minor, patch)
        """
        match = VersionManager.VERSION_PATTERN.match(version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        
        return tuple(map(int, match.groups()))
    
    @staticmethod
    def compare_versions(v1: str, v2: str) -> int:
        """
        Compare two version strings.
        
        Args:
            v1: First version
            v2: Second version
        
        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        major1, minor1, patch1 = VersionManager.parse_version(v1)
        major2, minor2, patch2 = VersionManager.parse_version(v2)
        
        if (major1, minor1, patch1) < (major2, minor2, patch2):
            return -1
        elif (major1, minor1, patch1) > (major2, minor2, patch2):
            return 1
        else:
            return 0
    
    def get_version_history(self) -> pd.DataFrame:
        """
        Get history of all versioned models.
        
        Returns:
            DataFrame with version information
        """
        versions = []
        
        for model_file in self.models_dir.glob('*.pkl'):
            # Extract version from filename
            version_match = re.search(
                r'v?(\d+\.\d+\.\d+)',
                model_file.stem
            )
            
            if version_match:
                version = version_match.group(1)
                
                # Load metadata if available
                metadata_file = model_file.with_suffix('.json')
                if metadata_file.exists():
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    versions.append({
                        'version': version,
                        'file': str(model_file),
                        'size_mb': model_file.stat().st_size / 1024 / 1024,
                        'created_at': metadata['serialization'].get('created_at'),
                        'model_type': metadata['serialization'].get('model_type'),
                        'checksum': metadata['serialization'].get('checksum', '')[:8]
                    })
        
        if not versions:
            return pd.DataFrame()
        
        df = pd.DataFrame(versions)
        
        # Sort by version
        df['version_tuple'] = df['version'].apply(self.parse_version)
        df = df.sort_values('version_tuple', ascending=False)
        df = df.drop('version_tuple', axis=1)
        
        return df
    
    def get_latest_version(self) -> Optional[str]:
        """Get latest version string."""
        history = self.get_version_history()
        
        if history.empty:
            return None
        
        return history.iloc[0]['version']
    
    def increment_version(
        self,
        current_version: str,
        bump: str = 'patch'
    ) -> str:
        """
        Increment version.
        
        Args:
            current_version: Current version string
            bump: 'major', 'minor', or 'patch'
        
        Returns:
            New version string
        """
        major, minor, patch = self.parse_version(current_version)
        
        if bump == 'major':
            return f"{major + 1}.0.0"
        elif bump == 'minor':
            return f"{major}.{minor + 1}.0"
        elif bump == 'patch':
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Invalid bump type: {bump}")
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for model serialization."""
import pytest
import pandas as pd
import numpy as np

from src.models.serialization.model_serializer import ModelSerializer
from src.models.serialization.version_manager import VersionManager
from src.models.serialization.schemas import TrainingMetadata


def test_version_parsing():
    """Test semantic version parsing."""
    version = '1.2.3'
    major, minor, patch = VersionManager.parse_version(version)
    
    assert (major, minor, patch) == (1, 2, 3)


def test_version_comparison():
    """Test version comparison."""
    assert VersionManager.compare_versions('1.2.3', '1.2.2') == 1
    assert VersionManager.compare_versions('1.2.3', '1.2.3') == 0
    assert VersionManager.compare_versions('1.2.3', '1.3.0') == -1


def test_model_save_load(sample_model, tmp_path):
    """Test model serialization round-trip."""
    model_path = tmp_path / "model_v1.0.0.pkl"
    
    metadata = TrainingMetadata(
        n_samples_train=1000,
        n_features=50,
        feature_names=['f1', 'f2'],
        training_config={},
        training_duration_seconds=120.5,
        performance_metrics={'mape': 2.5}
    )
    
    # Save
    ser_meta = ModelSerializer.save_model(
        model=sample_model,
        path=str(model_path),
        training_metadata=metadata,
        version='1.0.0'
    )
    
    assert ser_meta.version == '1.0.0'
    assert model_path.exists()
    
    # Load
    loaded_model = ModelSerializer.load_model(str(model_path))
    
    assert loaded_model.name == sample_model.name


def test_integrity_validation(sample_model, tmp_path):
    """Test checksum validation."""
    model_path = tmp_path / "model.pkl"
    
    metadata = TrainingMetadata(
        n_samples_train=1000,
        n_features=50,
        feature_names=[],
        training_config={},
        training_duration_seconds=60.0,
        performance_metrics={}
    )
    
    ModelSerializer.save_model(
        model=sample_model,
        path=str(model_path),
        training_metadata=metadata,
        version='1.0.0'
    )
    
    # Corrupt file
    with open(model_path, 'ab') as f:
        f.write(b'corrupt')
    
    # Should raise error
    with pytest.raises(ValueError, match="Checksum mismatch"):
        ModelSerializer.load_model(str(model_path), validate_integrity=True)
```

---

## 📝 Technical Notes

### Semantic Versioning
- MAJOR: Breaking changes (incompatible model architecture)
- MINOR: New features (additional horizons, features)
- PATCH: Bug fixes, training improvements

### Integrity Validation
- SHA256 checksums prevent silent corruption
- Validates entire model file including metadata
- Critical for production deployments

---

## 🔗 Dependencies

**Depends On:**
- PC-024-03: Base Model Interface

**External Dependencies:**
- `joblib>=1.3.0`
- `pydantic>=2.0.0`

**Blocks:**
- Epic-05 deployment components

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Save/load with complete metadata
- [ ] Semantic versioning functional
- [ ] Checksum validation works
- [ ] Handles large models (>1GB)
- [ ] Unit tests pass with >85% coverage
- [ ] Load time < 5s for typical models
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-027-03: BLF Intraday Predictor](PC-027-03-blf-predictor.md)  
**Next:** [PC-029-03: Universal Trainer with Parallelization](PC-029-03-universal-trainer.md)
