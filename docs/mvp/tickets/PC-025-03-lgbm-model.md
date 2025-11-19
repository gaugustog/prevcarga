# PC-025-03: LGBM Model Implementation

**Ticket ID:** PC-025-03  
**Epic:** [Epic-03: Model Layer - End-to-End Models](../epics/Epic-03.md)  
**User Story:** US-2  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement LightGBM model for short-term electric load forecasting (D+0, D+1 horizons) with Optuna-based hyperparameter optimization, custom objective functions, and integration with BLF strategy for intraday updates. Achieves high accuracy for critical near-term predictions.

**As a** forecasting engineer  
**I want** LGBM model for short-term load forecasting (D+0, D+1)  
**So that** I can achieve high accuracy for critical near-term predictions

---

## ✅ Acceptance Criteria

- [ ] Implements LightGBM with custom objective function for load forecasting
- [ ] Supports D+0 and D+1 horizon predictions
- [ ] Includes Optuna hyperparameter optimization
- [ ] Handles categorical features and missing values appropriately
- [ ] Provides feature importance analysis and SHAP values
- [ ] Integrates with BLF strategy for intraday updates
- [ ] Achieves MAPE < 3% on validation set
- [ ] Training completes in < 30 minutes per area
- [ ] Comprehensive tests with real and synthetic data

---

## 🔧 Implementation Tasks

### 1. Create LGBM Module Structure
- [ ] Create `src/models/end_to_end/` directory
- [ ] Create `src/models/end_to_end/__init__.py`
- [ ] Create `src/models/end_to_end/lgbm_model.py`
- [ ] Create `src/models/config/lgbm_config.py`
- [ ] Add module docstrings

### 2. Implement LGBM Configuration Schema
- [ ] Create `LGBMConfig` Pydantic model
- [ ] Add `n_estimators` field (default: 1000)
- [ ] Add `learning_rate` field (default: 0.05)
- [ ] Add `num_leaves` field (default: 31)
- [ ] Add `max_depth` field (default: -1)
- [ ] Add `min_child_samples` field (default: 20)
- [ ] Add `feature_fraction` field (default: 0.8)
- [ ] Add `bagging_fraction` field (default: 0.8)
- [ ] Add `bagging_freq` field (default: 5)
- [ ] Add `early_stopping_rounds` field (default: 50)
- [ ] Add `categorical_features` list field
- [ ] Add `optimize_hyperparams` boolean (default: True)
- [ ] Add `n_trials` for Optuna (default: 100)

### 3. Implement LGBMModel Class
- [ ] Inherit from `BaseModel`
- [ ] Initialize with empty model
- [ ] Implement `name` property returning "lgbm"
- [ ] Implement `version` property returning "1.0.0"
- [ ] Implement `supported_horizons` property returning [0, 1]
- [ ] Add internal attributes for model storage

### 4. Implement Data Preparation
- [ ] Create `_prepare_data()` method
- [ ] Handle categorical features encoding
- [ ] Create lgb.Dataset with proper parameters
- [ ] Handle missing values (NaN strategy)
- [ ] Validate feature alignment
- [ ] Store feature names and types

### 5. Implement Hyperparameter Optimization
- [ ] Create `_optimize_hyperparams()` method
- [ ] Define Optuna study with MAE objective
- [ ] Create trial parameter space
- [ ] Implement objective function with CV
- [ ] Use TimeSeriesSplit for validation
- [ ] Return best parameters
- [ ] Log optimization progress

### 6. Implement Training Method
- [ ] Implement `fit()` method
- [ ] Validate input data
- [ ] Optionally run hyperparameter optimization
- [ ] Create LightGBM dataset
- [ ] Configure training parameters
- [ ] Add callbacks (early stopping, logging)
- [ ] Train model with lgb.train()
- [ ] Store training metadata
- [ ] Set `_is_fitted` flag

### 7. Implement Prediction Method
- [ ] Implement `predict()` method
- [ ] Validate horizons are supported
- [ ] Generate base predictions
- [ ] Apply horizon-specific adjustments
- [ ] Handle D+0 (same-day) predictions
- [ ] Handle D+1 (next-day) predictions
- [ ] Return DataFrame with pred_h0, pred_h1

### 8. Implement Horizon Adjustments
- [ ] Create `_calculate_horizon_adjustment()` method
- [ ] Compute temporal decay factors
- [ ] Adjust for horizon-specific patterns
- [ ] Use calendar effects (weekday vs weekend)
- [ ] Apply learned correction factors

### 9. Implement Feature Importance
- [ ] Implement `get_feature_importance()` method
- [ ] Extract gain-based importance
- [ ] Extract split-based importance
- [ ] Sort by importance value
- [ ] Return dictionary mapping features to scores

### 10. Implement SHAP Integration
- [ ] Create `get_shap_values()` method
- [ ] Import shap library
- [ ] Create TreeExplainer
- [ ] Calculate SHAP values for sample
- [ ] Return SHAP values array
- [ ] Add visualization helper

### 11. Implement Custom Objective Function
- [ ] Create `_custom_objective()` method
- [ ] Implement asymmetric loss (under-prediction penalty)
- [ ] Handle gradient and hessian calculation
- [ ] Support load-specific constraints
- [ ] Return (gradient, hessian) tuple

### 12. Implement Model Serialization
- [ ] Override `save()` method if needed
- [ ] Save model with lgb.Booster.save_model()
- [ ] Save additional artifacts (feature names, config)
- [ ] Handle large model files efficiently
- [ ] Store SHAP explainer if available

### 13. Implement Model Loading
- [ ] Override `load()` method if needed
- [ ] Load LightGBM booster
- [ ] Restore feature names and metadata
- [ ] Validate model integrity
- [ ] Restore configuration

### 14. Handle Categorical Features
- [ ] Auto-detect categorical columns
- [ ] Encode categorical features properly
- [ ] Pass categorical_feature parameter to lgb.Dataset
- [ ] Handle unseen categories in prediction
- [ ] Document categorical feature requirements

### 15. Implement Cross-Validation
- [ ] Create `_cross_validate()` method
- [ ] Use TimeSeriesSplit (preserves temporal order)
- [ ] Calculate MAE, MAPE, RMSE per fold
- [ ] Return aggregated scores
- [ ] Store CV results in metadata

### 16. Write Comprehensive Tests
- [ ] Create `tests/models/end_to_end/test_lgbm_model.py`
- [ ] Test model initialization
- [ ] Test training with default params
- [ ] Test training with hyperopt
- [ ] Test D+0 and D+1 predictions
- [ ] Test with categorical features
- [ ] Test missing value handling
- [ ] Test feature importance extraction
- [ ] Test model save/load round-trip

### 17. Write Integration Tests
- [ ] Test with real feature pipeline output
- [ ] Test multi-area training
- [ ] Test prediction workflow
- [ ] Test SHAP value calculation
- [ ] Test performance benchmarks

### 18. Create Usage Examples
- [ ] Create `examples/lgbm_training_demo.py`
- [ ] Show basic training workflow
- [ ] Show hyperparameter optimization
- [ ] Show prediction generation
- [ ] Show feature importance analysis
- [ ] Show SHAP visualization

---

## 💻 Implementation Details

### LGBM Configuration

```python
"""Configuration for LGBM model."""
from typing import List, Optional
from pydantic import BaseModel, Field


class LGBMConfig(BaseModel):
    """Configuration for LightGBM forecasting model."""
    
    # Boosting parameters
    n_estimators: int = Field(
        default=1000,
        description="Number of boosting rounds"
    )
    learning_rate: float = Field(
        default=0.05,
        description="Learning rate"
    )
    num_leaves: int = Field(
        default=31,
        description="Maximum number of leaves in one tree"
    )
    max_depth: int = Field(
        default=-1,
        description="Maximum tree depth (-1 means no limit)"
    )
    min_child_samples: int = Field(
        default=20,
        description="Minimum number of samples in a leaf"
    )
    
    # Feature subsampling
    feature_fraction: float = Field(
        default=0.8,
        description="Fraction of features to use per tree"
    )
    bagging_fraction: float = Field(
        default=0.8,
        description="Fraction of data to use per tree"
    )
    bagging_freq: int = Field(
        default=5,
        description="Frequency for bagging"
    )
    
    # Regularization
    lambda_l1: float = Field(
        default=0.0,
        description="L1 regularization"
    )
    lambda_l2: float = Field(
        default=0.0,
        description="L2 regularization"
    )
    
    # Early stopping
    early_stopping_rounds: int = Field(
        default=50,
        description="Stop if no improvement for N rounds"
    )
    
    # Features
    categorical_features: List[str] = Field(
        default_factory=list,
        description="List of categorical feature names"
    )
    
    # Hyperparameter optimization
    optimize_hyperparams: bool = Field(
        default=True,
        description="Run Optuna optimization"
    )
    n_trials: int = Field(
        default=100,
        description="Number of Optuna trials"
    )
    cv_folds: int = Field(
        default=5,
        description="Cross-validation folds"
    )
    
    # Model behavior
    objective: str = Field(
        default="regression",
        description="Objective function"
    )
    metric: str = Field(
        default="mae",
        description="Evaluation metric"
    )
    verbosity: int = Field(
        default=-1,
        description="Verbosity level"
    )
    n_jobs: int = Field(
        default=-1,
        description="Number of parallel threads"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )
```

### LGBM Model Implementation

```python
"""LightGBM model for short-term load forecasting."""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from src.models.base.model import BaseModel
from src.models.config.lgbm_config import LGBMConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LGBMModel(BaseModel):
    """
    LightGBM model for short-term electric load forecasting.
    
    Optimized for D+0 and D+1 horizons with Optuna hyperparameter
    optimization and support for intraday BLF updates.
    
    Features:
    - Automatic hyperparameter tuning with Optuna
    - Custom objective functions for asymmetric loss
    - Categorical feature handling
    - SHAP value integration for interpretability
    - Efficient training with early stopping
    
    Example:
        >>> model = LGBMModel()
        >>> config = {'optimize_hyperparams': True, 'n_trials': 50}
        >>> model.fit(X_train, y_train, config)
        >>> predictions = model.predict(X_test, horizons=[0, 1])
    """
    
    def __init__(self):
        """Initialize LGBM model."""
        super().__init__()
        self.model: Optional[lgb.Booster] = None
        self.config: Optional[LGBMConfig] = None
        self.best_iteration: int = 0
        self.cv_scores: Dict[str, List[float]] = {}
    
    @property
    def name(self) -> str:
        return "lgbm"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return [0, 1]  # D+0 and D+1
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ) -> 'LGBMModel':
        """
        Train LGBM model with optional hyperparameter optimization.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            config: Training configuration
        
        Returns:
            Self for method chaining
        """
        # Validate configuration
        self.config = LGBMConfig(**config)
        
        logger.info(f"Training LGBM model with {len(X)} samples and {len(X.columns)} features")
        
        # Store feature names
        self._feature_names = X.columns.tolist()
        
        # Prepare data
        train_data = self._prepare_data(X, y)
        
        # Hyperparameter optimization
        if self.config.optimize_hyperparams:
            logger.info(f"Starting hyperparameter optimization with {self.config.n_trials} trials")
            best_params = self._optimize_hyperparams(train_data)
        else:
            best_params = self._get_default_params()
        
        logger.info(f"Training final model with params: {best_params}")
        
        # Train final model
        self.model = lgb.train(
            params=best_params,
            train_set=train_data,
            num_boost_round=self.config.n_estimators,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )
        
        self.best_iteration = self.model.best_iteration
        self._is_fitted = True
        
        # Store training metadata
        self._training_metadata = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'best_iteration': self.best_iteration,
            'params': best_params
        }
        
        logger.info(f"Training complete. Best iteration: {self.best_iteration}")
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Generate predictions for specified horizons.
        
        Args:
            X: Feature DataFrame
            horizons: List of horizons (0, 1)
        
        Returns:
            DataFrame with predictions
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Validate horizons
        invalid = [h for h in horizons if h not in self.supported_horizons]
        if invalid:
            raise ValueError(f"Unsupported horizons: {invalid}. Supported: {self.supported_horizons}")
        
        logger.debug(f"Generating predictions for {len(X)} samples, horizons: {horizons}")
        
        # Generate base predictions
        base_pred = self.model.predict(X, num_iteration=self.best_iteration)
        
        predictions = {}
        
        for horizon in horizons:
            if horizon == 0:
                # D+0: Same-day prediction (no adjustment)
                predictions['pred_h0'] = base_pred
            elif horizon == 1:
                # D+1: Next-day prediction (apply adjustment)
                adjustment = self._calculate_horizon_adjustment(X, horizon)
                predictions['pred_h1'] = base_pred * adjustment
        
        return pd.DataFrame(predictions, index=X.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping features to importance
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting importance")
        
        # Get gain-based importance
        importance = self.model.feature_importance(importance_type='gain')
        
        # Map to feature names
        feature_importance = dict(zip(self._feature_names, importance))
        
        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance
    
    def get_shap_values(
        self,
        X: pd.DataFrame,
        sample_size: int = 1000
    ) -> np.ndarray:
        """
        Calculate SHAP values for interpretability.
        
        Args:
            X: Feature DataFrame
            sample_size: Number of samples to use
        
        Returns:
            SHAP values array
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted")
        
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP library required. Install with: pip install shap")
        
        # Sample data for speed
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        logger.info(f"Calculating SHAP values for {len(X_sample)} samples")
        
        # Create explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values
    
    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> lgb.Dataset:
        """
        Prepare LightGBM dataset.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            LightGBM Dataset
        """
        # Identify categorical features
        categorical_indices = []
        if self.config.categorical_features:
            for i, col in enumerate(X.columns):
                if col in self.config.categorical_features:
                    categorical_indices.append(i)
        
        # Create dataset
        dataset = lgb.Dataset(
            data=X,
            label=y,
            categorical_feature=categorical_indices,
            free_raw_data=False
        )
        
        return dataset
    
    def _optimize_hyperparams(
        self,
        train_data: lgb.Dataset
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            train_data: Training dataset
        
        Returns:
            Best parameters dictionary
        """
        import optuna
        
        def objective(trial):
            """Optuna objective function."""
            params = {
                'objective': self.config.objective,
                'metric': self.config.metric,
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            }
            
            # Cross-validation
            cv_results = lgb.cv(
                params=params,
                train_set=train_data,
                num_boost_round=500,
                nfold=self.config.cv_folds,
                stratified=False,
                shuffle=False,  # Time series - preserve order
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
                return_cvbooster=False
            )
            
            # Get best score
            best_score = cv_results[f'{self.config.metric}-mean'][-1]
            
            return best_score
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            show_progress_bar=True
        )
        
        logger.info(f"Best trial: {study.best_trial.number}, Best MAE: {study.best_value:.4f}")
        
        # Merge with base params
        best_params = self._get_default_params()
        best_params.update(study.best_params)
        
        return best_params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default LightGBM parameters."""
        return {
            'objective': self.config.objective,
            'metric': self.config.metric,
            'boosting_type': 'gbdt',
            'num_leaves': self.config.num_leaves,
            'learning_rate': self.config.learning_rate,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,
            'min_child_samples': self.config.min_child_samples,
            'lambda_l1': self.config.lambda_l1,
            'lambda_l2': self.config.lambda_l2,
            'verbosity': self.config.verbosity,
            'n_jobs': self.config.n_jobs,
            'random_state': self.config.random_state,
        }
    
    def _calculate_horizon_adjustment(
        self,
        X: pd.DataFrame,
        horizon: int
    ) -> np.ndarray:
        """
        Calculate horizon-specific adjustment factor.
        
        For D+1, apply slight adjustment based on temporal patterns.
        
        Args:
            X: Feature DataFrame
            horizon: Forecast horizon
        
        Returns:
            Adjustment factors array
        """
        # Simple adjustment: slight decay for next-day
        if horizon == 1:
            # Check if we have hour information
            if 'hour' in X.columns:
                # Peak hours get smaller adjustment
                hour = X['hour'].values
                adjustment = np.where(
                    (hour >= 17) & (hour <= 20),  # Peak hours
                    1.0,
                    0.98  # Slight decay for off-peak
                )
            else:
                adjustment = np.ones(len(X)) * 0.99
        else:
            adjustment = np.ones(len(X))
        
        return adjustment
    
    def save(self, path: str) -> None:
        """Save LGBM model."""
        from pathlib import Path
        import joblib
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save booster
        self.model.save_model(str(path_obj))
        
        # Save metadata
        metadata = {
            'feature_names': self._feature_names,
            'config': self.config.dict() if self.config else {},
            'training_metadata': self._training_metadata,
            'best_iteration': self.best_iteration
        }
        
        metadata_path = path_obj.parent / f"{path_obj.stem}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"LGBM model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LGBMModel':
        """Load LGBM model."""
        from pathlib import Path
        import joblib
        
        path_obj = Path(path)
        
        # Load booster
        booster = lgb.Booster(model_file=str(path_obj))
        
        # Load metadata
        metadata_path = path_obj.parent / f"{path_obj.stem}_metadata.pkl"
        metadata = joblib.load(metadata_path)
        
        # Reconstruct model
        model = cls()
        model.model = booster
        model._feature_names = metadata['feature_names']
        model.config = LGBMConfig(**metadata['config'])
        model._training_metadata = metadata['training_metadata']
        model.best_iteration = metadata['best_iteration']
        model._is_fitted = True
        
        logger.info(f"LGBM model loaded from {path}")
        
        return model
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for LGBM model."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.end_to_end.lgbm_model import LGBMModel
from src.models.config.lgbm_config import LGBMConfig


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'hour': np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples],
        'day_of_week': np.tile(np.arange(7), n_samples // 7 + 1)[:n_samples],
        'temperature': np.random.randn(n_samples) * 10 + 20,
        'lag_24': np.random.randn(n_samples) * 100 + 1000,
        'lag_48': np.random.randn(n_samples) * 100 + 1000,
    })
    
    # Target with patterns
    y = (
        1000 +
        50 * np.sin(2 * np.pi * X['hour'] / 24) +  # Daily pattern
        20 * (X['day_of_week'] < 5) +  # Weekday effect
        5 * X['temperature'] +  # Temperature effect
        np.random.randn(n_samples) * 20  # Noise
    )
    
    return X, pd.Series(y)


def test_lgbm_initialization():
    """Test LGBM model initialization."""
    model = LGBMModel()
    
    assert model.name == "lgbm"
    assert model.version == "1.0.0"
    assert model.supported_horizons == [0, 1]
    assert not model.is_fitted()


def test_lgbm_training_no_hyperopt(sample_data):
    """Test LGBM training without hyperparameter optimization."""
    X, y = sample_data
    
    model = LGBMModel()
    config = {
        'optimize_hyperparams': False,
        'n_estimators': 100,
        'early_stopping_rounds': 10
    }
    
    model.fit(X, y, config)
    
    assert model.is_fitted()
    assert model.best_iteration > 0
    assert len(model.get_feature_names()) == 5


def test_lgbm_training_with_hyperopt(sample_data):
    """Test LGBM training with hyperparameter optimization."""
    X, y = sample_data
    
    model = LGBMModel()
    config = {
        'optimize_hyperparams': True,
        'n_trials': 10,  # Small for testing
        'cv_folds': 3
    }
    
    model.fit(X, y, config)
    
    assert model.is_fitted()


def test_lgbm_prediction(sample_data):
    """Test LGBM predictions."""
    X_train, y_train = sample_data
    X_test = X_train.iloc[:100]
    
    model = LGBMModel()
    model.fit(X_train, y_train, {'optimize_hyperparams': False, 'n_estimators': 50})
    
    predictions = model.predict(X_test, horizons=[0, 1])
    
    assert 'pred_h0' in predictions.columns
    assert 'pred_h1' in predictions.columns
    assert len(predictions) == 100
    assert predictions['pred_h0'].notna().all()


def test_lgbm_feature_importance(sample_data):
    """Test feature importance extraction."""
    X, y = sample_data
    
    model = LGBMModel()
    model.fit(X, y, {'optimize_hyperparams': False, 'n_estimators': 50})
    
    importance = model.get_feature_importance()
    
    assert len(importance) == 5
    assert all(v >= 0 for v in importance.values())


def test_lgbm_save_load(sample_data, tmp_path):
    """Test model serialization."""
    X, y = sample_data
    
    # Train model
    model = LGBMModel()
    model.fit(X, y, {'optimize_hyperparams': False, 'n_estimators': 50})
    
    # Save
    model_path = tmp_path / "lgbm_model.txt"
    model.save(str(model_path))
    
    # Load
    loaded_model = LGBMModel.load(str(model_path))
    
    assert loaded_model.is_fitted()
    assert loaded_model.get_feature_names() == model.get_feature_names()
    
    # Compare predictions
    X_test = X.iloc[:10]
    pred_original = model.predict(X_test, horizons=[0])
    pred_loaded = loaded_model.predict(X_test, horizons=[0])
    
    np.testing.assert_array_almost_equal(
        pred_original['pred_h0'].values,
        pred_loaded['pred_h0'].values
    )


def test_lgbm_unsupported_horizon(sample_data):
    """Test error for unsupported horizon."""
    X, y = sample_data
    
    model = LGBMModel()
    model.fit(X, y, {'optimize_hyperparams': False, 'n_estimators': 50})
    
    with pytest.raises(ValueError, match="Unsupported horizons"):
        model.predict(X.iloc[:10], horizons=[5])


def test_lgbm_prediction_before_training():
    """Test error when predicting before training."""
    model = LGBMModel()
    X_test = pd.DataFrame({'feature1': [1, 2, 3]})
    
    with pytest.raises(ValueError, match="must be fitted"):
        model.predict(X_test, horizons=[0])
```

---

## 📝 Technical Notes

### Hyperparameter Optimization
- Optuna uses Tree-structured Parzen Estimator (TPE) sampler
- Typical optimization: 100 trials takes ~20 minutes
- Cross-validation uses TimeSeriesSplit to preserve temporal order
- Best params stored in model metadata

### Performance Considerations
- Early stopping prevents overfitting
- Feature/bagging fraction reduce computation
- Categorical features encoded automatically
- Multi-threading enabled by default (n_jobs=-1)

### SHAP Integration
- TreeExplainer is fast for tree-based models
- Sample size of 1000 recommended for speed
- Useful for model debugging and stakeholder communication

---

## 🔗 Dependencies

**Depends On:**
- PC-024-03: Base Model Interface & Registry

**External Dependencies:**
- `lightgbm>=4.0.0`
- `optuna>=3.0.0`
- `shap>=0.42.0` (optional)
- `scikit-learn>=1.3.0`

**Blocks:**
- PC-027-03: BLF Intraday Predictor (provides base model)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] LGBM model trains successfully
- [ ] Hyperparameter optimization functional
- [ ] D+0 and D+1 predictions working
- [ ] Feature importance extraction working
- [ ] Model save/load validated
- [ ] Unit tests pass with >85% coverage
- [ ] Training time < 30 min per area
- [ ] MAPE < 3% on validation set
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-024-03: Base Model Interface & Registry](PC-024-03-base-model-interface-registry.md)  
**Next:** [PC-026-03: Random Forest Multi-Horizon Model](PC-026-03-random-forest-model.md)
