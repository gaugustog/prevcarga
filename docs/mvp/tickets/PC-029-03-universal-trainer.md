# PC-029-03: Universal Trainer with Parallelization

**Ticket ID:** PC-029-03  
**Epic:** [Epic-03: Model Layer - End-to-End Models](../epics/Epic-03.md)  
**User Story:** US-6  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement universal training interface that handles multiple model types with parallel execution across areas and horizons. Includes checkpointing, progress tracking, automated hyperparameter optimization, and memory-efficient execution for large-scale training operations.

**As a** ML engineer  
**I want** universal trainer with parallelization and checkpointing  
**So that** I can efficiently train models across all areas and horizons

---

## ✅ Acceptance Criteria

- [ ] Supports training multiple areas in parallel
- [ ] Handles different model types with unified interface
- [ ] Implements checkpointing and resume capabilities
- [ ] Provides progress tracking and logging
- [ ] Manages memory efficiently during parallel training
- [ ] Includes automated hyperparameter optimization
- [ ] Achieves >50% speedup over sequential training
- [ ] Supports GPU acceleration where applicable
- [ ] Comprehensive tests with multi-area scenarios

---

## 🔧 Implementation Tasks

### 1. Create Trainer Module Structure
- [ ] Create `src/training/universal_trainer.py`
- [ ] Create `src/training/training_config.py`
- [ ] Create `src/training/checkpoint_manager.py`
- [ ] Add module docstrings

### 2. Implement Training Configuration Schema
- [ ] Create `TrainingConfig` Pydantic model
- [ ] Add `model_type` field (lgbm, random_forest)
- [ ] Add `model_config` dict
- [ ] Add `areas` list
- [ ] Add `horizons` list
- [ ] Add `parallel_workers` field (default: 4)
- [ ] Add `enable_checkpointing` boolean (default: True)
- [ ] Add `checkpoint_frequency` field (default: 10)
- [ ] Add `optimize_hyperparameters` boolean
- [ ] Add `optimization_trials` field (default: 100)
- [ ] Add `cv_folds` field (default: 5)
- [ ] Add `memory_limit_gb` field

### 3. Implement UniversalTrainer Class
- [ ] Initialize with configuration
- [ ] Create model registry reference
- [ ] Initialize checkpoint manager
- [ ] Setup logging
- [ ] Initialize progress tracker
- [ ] Create thread pool/process pool

### 4. Implement Single Area Training
- [ ] Create `_train_single_area()` static method
- [ ] Accept area, data, config parameters
- [ ] Load model class from registry
- [ ] Perform cross-validation
- [ ] Train final model
- [ ] Calculate performance metrics
- [ ] Return trained model and metrics

### 5. Implement Multi-Area Parallel Training
- [ ] Create `train_multiple_areas()` method
- [ ] Split areas across workers
- [ ] Use ProcessPoolExecutor for CPU parallelization
- [ ] Track progress with tqdm or custom tracker
- [ ] Handle worker failures gracefully
- [ ] Aggregate results
- [ ] Return training summary

### 6. Implement Hyperparameter Optimization
- [ ] Create `train_with_hyperopt()` method
- [ ] Integrate with Optuna
- [ ] Define objective function
- [ ] Use TimeSeriesSplit for CV
- [ ] Run optimization trials
- [ ] Store best hyperparameters
- [ ] Train final model with best params

### 7. Implement Cross-Validation
- [ ] Create `_cross_validate_model()` method
- [ ] Use TimeSeriesSplit for temporal data
- [ ] Calculate metrics per fold
- [ ] Aggregate mean and std
- [ ] Return CV scores
- [ ] Log fold performance

### 8. Implement Checkpointing
- [ ] Create CheckpointManager class
- [ ] Implement `save_checkpoint()` method
- [ ] Save training state (models, metrics, progress)
- [ ] Save config and metadata
- [ ] Implement `load_checkpoint()` method
- [ ] Restore training state
- [ ] Resume from last checkpoint

### 9. Implement Progress Tracking
- [ ] Create ProgressTracker class
- [ ] Track completed/total areas
- [ ] Track elapsed time
- [ ] Estimate time remaining
- [ ] Log progress updates
- [ ] Support callbacks for external monitoring

### 10. Implement Memory Management
- [ ] Monitor memory usage during training
- [ ] Limit number of parallel workers based on memory
- [ ] Clear model cache between areas
- [ ] Use memory-efficient data loading
- [ ] Implement garbage collection triggers

### 11. Implement GPU Support
- [ ] Detect GPU availability
- [ ] Allocate models to GPUs
- [ ] Handle multi-GPU training
- [ ] Fallback to CPU if GPU unavailable
- [ ] Log GPU utilization

### 12. Implement Training Pipeline
- [ ] Create `run_training_pipeline()` method
- [ ] Load data for all areas
- [ ] Optionally run hyperparameter optimization
- [ ] Train models for all areas in parallel
- [ ] Validate trained models
- [ ] Save models with versioning
- [ ] Generate training report

### 13. Implement Training Report
- [ ] Create `generate_training_report()` method
- [ ] Summarize performance metrics per area
- [ ] Show training duration
- [ ] Display cross-validation scores
- [ ] List hyperparameters used
- [ ] Save report as JSON and HTML

### 14. Implement Model Validation
- [ ] Create `validate_trained_models()` method
- [ ] Check model is fitted
- [ ] Verify prediction functionality
- [ ] Validate performance thresholds
- [ ] Flag low-performing models

### 15. Implement Resume Capability
- [ ] Create `resume_training()` method
- [ ] Load checkpoint
- [ ] Identify completed areas
- [ ] Resume training for remaining areas
- [ ] Preserve original configuration

### 16. Write Comprehensive Tests
- [ ] Create `tests/training/test_universal_trainer.py`
- [ ] Test single area training
- [ ] Test multi-area parallel training
- [ ] Test checkpointing and resume
- [ ] Test hyperparameter optimization
- [ ] Test memory management
- [ ] Test cross-validation
- [ ] Test progress tracking

### 17. Write Integration Tests
- [ ] Test with LGBM model
- [ ] Test with Random Forest model
- [ ] Test full pipeline with 24 areas
- [ ] Test checkpoint recovery
- [ ] Benchmark parallelization speedup

### 18. Create Usage Examples
- [ ] Create `examples/universal_trainer_demo.py`
- [ ] Show single model training
- [ ] Show multi-area training
- [ ] Show hyperparameter optimization
- [ ] Show checkpoint resume
- [ ] Visualize training progress

---

## 💻 Implementation Details

### Training Configuration

```python
"""Configuration for universal trainer."""
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for universal model training."""
    
    # Model configuration
    model_type: Literal["lgbm", "random_forest"] = Field(
        description="Type of model to train"
    )
    model_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific configuration"
    )
    
    # Data configuration
    areas: List[str] = Field(
        description="List of areas to train"
    )
    horizons: Optional[List[int]] = Field(
        default=None,
        description="Horizons to train (None = all supported)"
    )
    
    # Parallel execution
    parallel_workers: int = Field(
        default=4,
        description="Number of parallel workers"
    )
    backend: Literal["multiprocessing", "threading"] = Field(
        default="multiprocessing",
        description="Parallel backend"
    )
    
    # Checkpointing
    enable_checkpointing: bool = Field(
        default=True,
        description="Enable checkpoint saving"
    )
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory for checkpoints"
    )
    checkpoint_frequency: int = Field(
        default=10,
        description="Save checkpoint every N areas"
    )
    
    # Hyperparameter optimization
    optimize_hyperparameters: bool = Field(
        default=False,
        description="Run hyperparameter optimization"
    )
    optimization_trials: int = Field(
        default=100,
        description="Number of Optuna trials"
    )
    
    # Cross-validation
    cv_folds: int = Field(
        default=5,
        description="Number of CV folds"
    )
    
    # Resource management
    memory_limit_gb: Optional[float] = Field(
        default=None,
        description="Memory limit in GB"
    )
    gpu_enabled: bool = Field(
        default=False,
        description="Enable GPU acceleration"
    )
```

### Universal Trainer Implementation

```python
"""Universal trainer for parallel model training."""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from datetime import datetime

from src.models.base.model import BaseModel
from src.models.base.registry import ModelRegistry
from src.training.training_config import TrainingConfig
from src.training.checkpoint_manager import CheckpointManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UniversalTrainer:
    """
    Universal trainer for parallel model training across areas.
    
    Features:
    - Multi-area parallel training
    - Checkpointing and resume
    - Hyperparameter optimization
    - Cross-validation
    - Progress tracking
    - Memory management
    
    Example:
        >>> config = TrainingConfig(
        ...     model_type='lgbm',
        ...     areas=['area1', 'area2', 'area3'],
        ...     parallel_workers=3,
        ...     enable_checkpointing=True
        ... )
        >>> 
        >>> trainer = UniversalTrainer(config)
        >>> results = trainer.train_multiple_areas(data_dict)
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize universal trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.registry = ModelRegistry()
        
        if self.config.enable_checkpointing:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=self.config.checkpoint_dir
            )
        else:
            self.checkpoint_manager = None
        
        self.training_start_time: Optional[datetime] = None
        self.training_results: Dict[str, Any] = {}
        
        logger.info(
            f"UniversalTrainer initialized: "
            f"{self.config.model_type}, "
            f"{len(self.config.areas)} areas, "
            f"{self.config.parallel_workers} workers"
        )
    
    def train_multiple_areas(
        self,
        data_dict: Dict[str, Tuple[pd.DataFrame, pd.Series]]
    ) -> Dict[str, Any]:
        """
        Train models for multiple areas in parallel.
        
        Args:
            data_dict: Dictionary mapping area to (X, y) tuples
        
        Returns:
            Training results summary
        """
        self.training_start_time = datetime.now()
        
        areas_to_train = [
            area for area in self.config.areas
            if area in data_dict
        ]
        
        logger.info(f"Training {len(areas_to_train)} areas")
        
        # Check for checkpoint
        if self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                completed_areas = checkpoint.get('completed_areas', [])
                areas_to_train = [
                    a for a in areas_to_train
                    if a not in completed_areas
                ]
                logger.info(f"Resuming: {len(areas_to_train)} areas remaining")
                self.training_results = checkpoint.get('results', {})
        
        # Train in parallel
        with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = {}
            
            for area in areas_to_train:
                X, y = data_dict[area]
                
                future = executor.submit(
                    self._train_single_area,
                    area=area,
                    X=X,
                    y=y,
                    config=self.config
                )
                futures[future] = area
            
            # Track progress
            completed = 0
            total = len(areas_to_train)
            
            for future in as_completed(futures):
                area = futures[future]
                
                try:
                    model, metrics = future.result()
                    
                    # Save model
                    model_path = f"models/{self.config.model_type}/{area}_v1.0.0.pkl"
                    model.save(model_path)
                    
                    self.training_results[area] = {
                        'model_path': model_path,
                        'metrics': metrics,
                        'status': 'success'
                    }
                    
                    completed += 1
                    logger.info(
                        f"[{completed}/{total}] Trained {area}: "
                        f"MAPE={metrics.get('mape', 0):.2f}%"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to train {area}: {e}")
                    self.training_results[area] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    completed += 1
                
                # Checkpoint
                if self.checkpoint_manager and completed % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(list(self.training_results.keys()))
        
        # Final checkpoint
        if self.checkpoint_manager:
            self._save_checkpoint(list(self.training_results.keys()))
        
        # Generate summary
        summary = self._generate_summary()
        
        return summary
    
    @staticmethod
    def _train_single_area(
        area: str,
        X: pd.DataFrame,
        y: pd.Series,
        config: TrainingConfig
    ) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Train model for single area.
        
        Args:
            area: Area identifier
            X: Features
            y: Target
            config: Training configuration
        
        Returns:
            Trained model and metrics
        """
        logger.info(f"Training {area} with {config.model_type}")
        
        # Get model class
        registry = ModelRegistry()
        model_class = registry.get_model_class(config.model_type)
        
        # Create model instance
        model = model_class()
        
        # Cross-validation
        cv_scores = UniversalTrainer._cross_validate_model(
            model=model,
            X=X,
            y=y,
            config=config
        )
        
        # Train final model
        model.fit(X, y, config.model_config)
        
        # Calculate metrics on training data
        predictions = model.predict(X, horizons=model.supported_horizons[:1])
        
        if f'pred_h0' in predictions.columns:
            pred_col = 'pred_h0'
            mape = np.mean(np.abs((y - predictions[pred_col]) / y)) * 100
            rmse = np.sqrt(np.mean((y - predictions[pred_col]) ** 2))
        else:
            mape = 0
            rmse = 0
        
        metrics = {
            'mape': mape,
            'rmse': rmse,
            'cv_scores': cv_scores
        }
        
        return model, metrics
    
    @staticmethod
    def _cross_validate_model(
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        config: TrainingConfig
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            model: Model instance
            X: Features
            y: Target
            config: Training configuration
        
        Returns:
            CV scores dictionary
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=config.cv_folds)
        
        cv_scores = {
            'mape': [],
            'rmse': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train on fold
            model_fold = type(model)()
            model_fold.fit(X_train, y_train, config.model_config)
            
            # Predict
            predictions = model_fold.predict(
                X_val,
                horizons=model_fold.supported_horizons[:1]
            )
            
            if f'pred_h0' in predictions.columns:
                pred_col = 'pred_h0'
                mape = np.mean(np.abs((y_val - predictions[pred_col]) / y_val)) * 100
                rmse = np.sqrt(np.mean((y_val - predictions[pred_col]) ** 2))
                
                cv_scores['mape'].append(mape)
                cv_scores['rmse'].append(rmse)
        
        return cv_scores
    
    def train_with_hyperopt(
        self,
        area: str,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Train model with hyperparameter optimization.
        
        Args:
            area: Area identifier
            X: Features
            y: Target
        
        Returns:
            Trained model with optimized hyperparameters
        """
        import optuna
        
        logger.info(f"Optimizing hyperparameters for {area}")
        
        # Get model class
        model_class = self.registry.get_model_class(self.config.model_type)
        
        def objective(trial):
            # Define hyperparameter search space
            # (This would be model-specific)
            config = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10)
            }
            
            # Cross-validation
            model = model_class()
            cv_scores = self._cross_validate_model(model, X, y, self.config)
            
            return np.mean(cv_scores['mape'])
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config.optimization_trials)
        
        # Train final model with best params
        best_config = {**self.config.model_config, **study.best_params}
        
        model = model_class()
        model.fit(X, y, best_config)
        
        return model, {'best_params': study.best_params, 'best_score': study.best_value}
    
    def _save_checkpoint(self, completed_areas: List[str]) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'config': self.config.dict(),
            'completed_areas': completed_areas,
            'results': self.training_results,
            'timestamp': datetime.now()
        }
        
        self.checkpoint_manager.save_checkpoint(checkpoint)
        logger.info(f"Checkpoint saved: {len(completed_areas)} areas completed")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate training summary."""
        elapsed = (datetime.now() - self.training_start_time).total_seconds()
        
        successful = [
            r for r in self.training_results.values()
            if r.get('status') == 'success'
        ]
        
        failed = [
            r for r in self.training_results.values()
            if r.get('status') == 'failed'
        ]
        
        if successful:
            avg_mape = np.mean([r['metrics']['mape'] for r in successful])
            avg_rmse = np.mean([r['metrics']['rmse'] for r in successful])
        else:
            avg_mape = 0
            avg_rmse = 0
        
        summary = {
            'total_areas': len(self.config.areas),
            'successful': len(successful),
            'failed': len(failed),
            'elapsed_seconds': elapsed,
            'avg_mape': avg_mape,
            'avg_rmse': avg_rmse,
            'results_by_area': self.training_results
        }
        
        logger.info(
            f"Training complete: "
            f"{len(successful)}/{len(self.config.areas)} successful, "
            f"avg MAPE={avg_mape:.2f}%, "
            f"elapsed={elapsed/60:.1f} min"
        )
        
        return summary
```

### Checkpoint Manager

```python
"""Checkpoint manager for training resume."""
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages training checkpoints."""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "training_checkpoint.pkl"
    
    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save training checkpoint."""
        joblib.dump(checkpoint, self.checkpoint_path)
        logger.info(f"Checkpoint saved to {self.checkpoint_path}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load training checkpoint."""
        if not self.checkpoint_path.exists():
            return None
        
        checkpoint = joblib.load(self.checkpoint_path)
        logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
        
        return checkpoint
    
    def clear_checkpoint(self) -> None:
        """Remove checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Checkpoint cleared")
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for universal trainer."""
import pytest
import pandas as pd
import numpy as np

from src.training.universal_trainer import UniversalTrainer
from src.training.training_config import TrainingConfig


@pytest.fixture
def training_data():
    """Create sample training data."""
    data_dict = {}
    
    for area in ['area1', 'area2', 'area3']:
        X = pd.DataFrame({
            'hour': np.tile(np.arange(24), 100),
            'feature1': np.random.randn(2400)
        })
        
        y = pd.Series(1000 + np.random.randn(2400) * 50)
        
        data_dict[area] = (X, y)
    
    return data_dict


def test_trainer_initialization():
    """Test trainer initialization."""
    config = TrainingConfig(
        model_type='lgbm',
        areas=['area1', 'area2']
    )
    
    trainer = UniversalTrainer(config)
    
    assert trainer.config.model_type == 'lgbm'
    assert len(trainer.config.areas) == 2


def test_multi_area_training(training_data):
    """Test parallel training."""
    config = TrainingConfig(
        model_type='lgbm',
        areas=['area1', 'area2'],
        parallel_workers=2,
        enable_checkpointing=False
    )
    
    trainer = UniversalTrainer(config)
    results = trainer.train_multiple_areas(training_data)
    
    assert results['total_areas'] == 2
    assert results['successful'] > 0


def test_checkpoint_save_load(training_data, tmp_path):
    """Test checkpointing."""
    config = TrainingConfig(
        model_type='lgbm',
        areas=['area1', 'area2', 'area3'],
        parallel_workers=1,
        checkpoint_dir=str(tmp_path),
        checkpoint_frequency=1
    )
    
    trainer = UniversalTrainer(config)
    
    # Train (will checkpoint)
    results = trainer.train_multiple_areas(training_data)
    
    # Check checkpoint exists
    checkpoint_path = tmp_path / "training_checkpoint.pkl"
    assert checkpoint_path.exists()
```

---

## 📝 Technical Notes

### Parallelization Strategy
- ProcessPoolExecutor for CPU-bound training
- Each worker trains one area independently
- Results collected and aggregated
- >50% speedup expected with 4+ workers

### Memory Management
- Monitor memory per worker
- Limit concurrent workers if memory constrained
- Clear model cache between areas
- Use generators for data loading

---

## 🔗 Dependencies

**Depends On:**
- PC-024-03: Base Model Interface & Registry
- PC-025-03: LGBM Model
- PC-026-03: Random Forest Model

**External Dependencies:**
- `concurrent.futures` (Python stdlib)
- `optuna>=3.0.0` (for hyperopt)

**Blocks:**
- Epic-05 deployment pipeline

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Multi-area parallel training works
- [ ] Checkpointing and resume functional
- [ ] Hyperparameter optimization integrated
- [ ] Memory management implemented
- [ ] >50% speedup over sequential training
- [ ] Unit tests pass with >85% coverage
- [ ] Integration tests with 24 areas pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-028-03: Model Serialization & Version Management](PC-028-03-serialization-version-management.md)  
**Next:** Epic-04 Hierarchical Models
