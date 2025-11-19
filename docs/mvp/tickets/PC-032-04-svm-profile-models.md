# PC-032-04: SVM Profile Models

**Ticket ID:** PC-032-04  
**Epic:** [Epic-04: Model Layer - Hierarchical Models](../epics/Epic-04.md)  
**User Story:** US-2  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement 48 Support Vector Machine (SVM) models for semi-hourly profile forecasting. Each model predicts profile ratios (load / demand_mean) for a specific semi-hourly period, using RBF kernel with hyperparameter optimization. Ensures profile ratios sum to reasonable daily totals and handles categorical/temporal features.

**As a** ML engineer  
**I want** SVM models for semi-hourly profile forecasting  
**So that** I can capture intraday patterns and their variations around demand mean

---

## ✅ Acceptance Criteria

- [ ] Implements 48 SVM models (one per semi-hourly period 0-47)
- [ ] Each model predicts profile ratio (load / demand_mean)
- [ ] Uses RBF kernel with hyperparameter optimization via GridSearchCV
- [ ] Handles categorical features (weekday, holiday, season)
- [ ] Includes temporal features (hour, day type, calendar effects)
- [ ] Ensures profile ratios sum to reasonable daily totals (0.5-2.0 range)
- [ ] Training completes in <20 minutes for all 48 models
- [ ] Parallel training across hours functional
- [ ] Comprehensive tests with profile validation
- [ ] Documentation explains profile ratio concept

---

## 🔧 Implementation Tasks

### 1. Create Profile Module Structure
- [ ] Create `src/models/profile/__init__.py`
- [ ] Create `src/models/profile/svm_profile.py`
- [ ] Create `src/models/profile/profile_normalizer.py`
- [ ] Add module docstrings

### 2. Implement SVM Configuration Schema
- [ ] Create `SVMProfileConfig` Pydantic model
- [ ] Add `kernel` field (default: "rbf")
- [ ] Add `C` field (regularization)
- [ ] Add `gamma` field (kernel coefficient)
- [ ] Add `epsilon` field (epsilon-tube)
- [ ] Add `optimize_hyperparams` boolean
- [ ] Add `cv_folds` field (default: 5)
- [ ] Add `n_jobs` field for parallelization

### 3. Implement SVMProfileModel Class
- [ ] Inherit from `BaseModel`
- [ ] Initialize models dict {hour: SVR}
- [ ] Initialize scalers dict {hour: StandardScaler}
- [ ] Initialize profile_statistics dict
- [ ] Implement `name` property
- [ ] Implement `version` property
- [ ] Implement `supported_horizons` property ([0])

### 4. Implement Profile Data Preparation
- [ ] Create `_prepare_profile_data()` method
- [ ] Group data by semi-hourly period (hour)
- [ ] Extract features for each hour
- [ ] Return dict {hour: (X_hour, y_hour)}
- [ ] Handle missing hours gracefully

### 5. Implement Feature Engineering for Profiles
- [ ] Create `_prepare_hour_features()` method
- [ ] Extract temporal features (dayofweek, day, month)
- [ ] Add calendar features (weekend, holiday, bridge_day)
- [ ] Add weather features if available (temperature, humidity)
- [ ] Add quarter and season indicators
- [ ] Return feature array

### 6. Implement SVM Training Per Hour
- [ ] Implement `fit()` method
- [ ] Prepare profile data grouped by hour
- [ ] Loop through 48 semi-hourly periods
- [ ] Scale features with StandardScaler
- [ ] Train SVM model per hour
- [ ] Store models, scalers, and statistics
- [ ] Log training progress

### 7. Implement Hyperparameter Optimization
- [ ] Create `_optimize_svm_hyperparams()` method
- [ ] Define parameter grid (C, gamma, epsilon)
- [ ] Use GridSearchCV with cross-validation
- [ ] Optimize using neg_mean_squared_error
- [ ] Return best estimator
- [ ] Log best parameters

### 8. Implement Profile Statistics Tracking
- [ ] Calculate mean, std per hour
- [ ] Calculate min, max per hour
- [ ] Calculate quartiles (q25, q75)
- [ ] Store in profile_statistics dict
- [ ] Use for prediction bounds validation

### 9. Implement Prediction Method
- [ ] Implement `predict()` method
- [ ] Validate horizons (only h=0 supported)
- [ ] Loop through timestamps in X
- [ ] Extract hour from timestamp
- [ ] Prepare features for timestamp
- [ ] Scale features
- [ ] Predict profile ratio
- [ ] Apply bounds clipping
- [ ] Return DataFrame with predictions

### 10. Implement Profile Bounds Validation
- [ ] Create `_apply_profile_bounds()` method
- [ ] Clip predictions using training statistics
- [ ] Use q25 - 2*std as lower bound
- [ ] Use q75 + 2*std as upper bound
- [ ] Ensure profiles in [0.1, 5.0] range
- [ ] Log bound violations

### 11. Implement Parallel Training
- [ ] Create `_train_hour_model()` static method
- [ ] Accept hour, data, config as parameters
- [ ] Return trained model, scaler, statistics
- [ ] Use joblib.Parallel for parallel execution
- [ ] Handle worker failures gracefully

### 12. Implement Profile Normalization Utilities
- [ ] Create `ProfileNormalizer` class
- [ ] Implement `normalize_daily_profiles()` method
- [ ] Ensure profiles sum to 48 (conservation)
- [ ] Implement `denormalize_profiles()` method
- [ ] Add smoothing methods

### 13. Implement Model Serialization
- [ ] Override `save()` method
- [ ] Save all 48 SVM models
- [ ] Save all scalers
- [ ] Save profile statistics
- [ ] Save configuration
- [ ] Use consistent directory structure

### 14. Implement Model Loading
- [ ] Override `load()` classmethod
- [ ] Load all SVM models
- [ ] Load all scalers
- [ ] Restore profile statistics
- [ ] Restore configuration
- [ ] Validate integrity

### 15. Write Comprehensive Tests
- [ ] Create `tests/models/profile/test_svm_profile.py`
- [ ] Test profile data preparation
- [ ] Test single hour training
- [ ] Test all 48 hours training
- [ ] Test hyperparameter optimization
- [ ] Test profile bounds validation
- [ ] Test parallel training
- [ ] Test save/load

### 16. Write Integration Tests
- [ ] Test with hierarchical model
- [ ] Test profile ratio calculation
- [ ] Test energy conservation
- [ ] Benchmark training time

### 17. Create Usage Examples
- [ ] Create `examples/svm_profile_demo.py`
- [ ] Show profile ratio calculation
- [ ] Show 48-model training
- [ ] Show profile prediction
- [ ] Visualize profiles by hour
- [ ] Show seasonal patterns

---

## 💻 Implementation Details

### SVM Profile Configuration

```python
"""Configuration for SVM profile models."""
from typing import Dict, List
from pydantic import BaseModel, Field


class SVMProfileConfig(BaseModel):
    """Configuration for SVM profile forecasting."""
    
    # SVM parameters
    kernel: str = Field(
        default="rbf",
        description="SVM kernel type"
    )
    C: float = Field(
        default=1.0,
        description="Regularization parameter"
    )
    gamma: str = Field(
        default="scale",
        description="Kernel coefficient"
    )
    epsilon: float = Field(
        default=0.1,
        description="Epsilon in epsilon-SVR"
    )
    
    # Hyperparameter optimization
    optimize_hyperparams: bool = Field(
        default=True,
        description="Run GridSearchCV optimization"
    )
    param_grid: Dict[str, List] = Field(
        default={
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2]
        },
        description="Parameter grid for optimization"
    )
    cv_folds: int = Field(
        default=5,
        description="Cross-validation folds"
    )
    
    # Profile bounds
    min_profile_ratio: float = Field(
        default=0.1,
        description="Minimum profile ratio"
    )
    max_profile_ratio: float = Field(
        default=5.0,
        description="Maximum profile ratio"
    )
    
    # Parallelization
    n_jobs: int = Field(
        default=-1,
        description="Parallel jobs (-1 = all cores)"
    )
```

### SVM Profile Model Implementation

```python
"""SVM model for semi-hourly profile forecasting."""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from src.models.base.model import BaseModel
from src.models.profile.profile_normalizer import ProfileNormalizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SVMProfileModel(BaseModel):
    """
    SVM model for semi-hourly profile forecasting.
    
    Trains 48 independent SVM models (one per semi-hourly period).
    Each model predicts profile ratio: load / demand_mean
    
    Profile ratios represent intraday load distribution patterns:
    - Ratio < 1: Below daily average
    - Ratio = 1: At daily average
    - Ratio > 1: Above daily average
    
    Features:
    - RBF kernel SVM with hyperparameter optimization
    - Feature scaling per hour
    - Profile bounds validation
    - Parallel training across hours
    
    Example:
        >>> model = SVMProfileModel()
        >>> config = {'optimize_hyperparams': True, 'n_jobs': 4}
        >>> 
        >>> # y should be profile ratios (load / demand_mean)
        >>> model.fit(X_train, profile_ratios, config)
        >>> 
        >>> # Predict profile ratios
        >>> predictions = model.predict(X_test, horizons=[0])
    """
    
    def __init__(self):
        """Initialize SVM profile model."""
        super().__init__()
        self.models: Dict[int, SVR] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.profile_statistics: Dict[int, Dict[str, float]] = {}
        self.config: Optional[SVMProfileConfig] = None
        self.normalizer = ProfileNormalizer()
    
    @property
    def name(self) -> str:
        return "svm_profile"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return [0]  # SVM profiles predict current period only
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ) -> 'SVMProfileModel':
        """
        Train SVM models for each semi-hourly period.
        
        Args:
            X: Feature DataFrame with datetime index
            y: Profile ratios Series (load / demand_mean)
            config: SVM configuration
        
        Returns:
            Self for method chaining
        """
        self.config = SVMProfileConfig(**config)
        
        logger.info("Training 48 SVM profile models")
        
        # Prepare profile data grouped by hour
        profile_data = self._prepare_profile_data(X, y)
        
        # Train models for each hour
        trained_count = 0
        failed_count = 0
        
        for hour in range(48):
            if hour not in profile_data:
                logger.warning(f"No data for hour {hour}")
                failed_count += 1
                continue
            
            hour_X, hour_y = profile_data[hour]
            
            if len(hour_y) < 50:  # Minimum samples for SVM
                logger.warning(f"Insufficient samples for hour {hour}: {len(hour_y)}")
                failed_count += 1
                continue
            
            try:
                # Feature scaling
                scaler = StandardScaler()
                hour_X_scaled = scaler.fit_transform(hour_X)
                
                # Train SVM model
                if self.config.optimize_hyperparams:
                    model = self._optimize_svm_hyperparams(
                        hour_X_scaled, hour_y
                    )
                else:
                    model = SVR(
                        kernel=self.config.kernel,
                        C=self.config.C,
                        gamma=self.config.gamma,
                        epsilon=self.config.epsilon
                    )
                    model.fit(hour_X_scaled, hour_y)
                
                # Store model and scaler
                self.models[hour] = model
                self.scalers[hour] = scaler
                
                # Store profile statistics
                self.profile_statistics[hour] = {
                    'mean': hour_y.mean(),
                    'std': hour_y.std(),
                    'min': hour_y.min(),
                    'max': hour_y.max(),
                    'q25': hour_y.quantile(0.25),
                    'q75': hour_y.quantile(0.75),
                    'n_samples': len(hour_y)
                }
                
                trained_count += 1
                
                if (trained_count % 10 == 0):
                    logger.info(f"Trained {trained_count}/48 profile models")
            
            except Exception as e:
                logger.error(f"Failed to train hour {hour}: {e}")
                failed_count += 1
                continue
        
        logger.info(
            f"SVM profile training complete: "
            f"{trained_count} successful, {failed_count} failed"
        )
        
        self._is_fitted = True
        
        # Store training metadata
        self._training_metadata = {
            'models_trained': trained_count,
            'coverage': trained_count / 48,
            'failed_hours': failed_count
        }
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Generate SVM profile predictions.
        
        Args:
            X: Feature DataFrame with datetime index
            horizons: List of horizons (only [0] supported)
        
        Returns:
            DataFrame with profile ratio predictions
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        
        for horizon in horizons:
            if horizon != 0:
                logger.warning(f"SVM profiles only support horizon 0, got {horizon}")
                continue
            
            # Predict profile for each timestamp
            profile_preds = []
            
            for i, timestamp in enumerate(X.index):
                hour = timestamp.hour * 2 + (timestamp.minute // 30)
                
                if hour in self.models:
                    # Prepare features for this timestamp
                    hour_X = self._prepare_timestamp_features(X.iloc[[i]], timestamp)
                    hour_X_scaled = self.scalers[hour].transform(hour_X)
                    
                    # Predict profile ratio
                    profile_pred = self.models[hour].predict(hour_X_scaled)[0]
                    
                    # Apply bounds
                    profile_pred = self._apply_profile_bounds(profile_pred, hour)
                    
                    profile_preds.append(profile_pred)
                else:
                    # Use neutral profile if no model
                    profile_preds.append(1.0)
            
            predictions[f'pred_h{horizon}'] = profile_preds
        
        return pd.DataFrame(predictions, index=X.index)
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance summary.
        
        SVMs don't provide direct feature importance, but we can
        report model statistics per hour.
        
        Returns:
            Dictionary with model statistics
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted")
        
        importance = {
            'model_type': 'SVM',
            'models_trained': len(self.models),
            'hours_covered': list(self.models.keys()),
            'note': 'SVM with RBF kernel does not provide feature importance'
        }
        
        return importance
    
    def _prepare_profile_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare training data grouped by semi-hourly periods.
        
        Args:
            X: Feature DataFrame
            y: Profile ratios Series
        
        Returns:
            Dictionary mapping hour to (X_hour, y_hour)
        """
        profile_data = {}
        
        for hour in range(48):
            # Filter data for this semi-hourly period
            mask = (X.index.hour * 2 + (X.index.minute // 30)) == hour
            
            if mask.sum() == 0:
                continue
            
            hour_X = X[mask]
            hour_y = y[mask]
            
            # Prepare features for this hour
            features = self._prepare_hour_features(hour_X)
            
            # Remove outliers from profile ratios
            valid_mask = (hour_y >= 0.05) & (hour_y <= 10.0)
            
            profile_data[hour] = (
                features[valid_mask.values],
                hour_y.values[valid_mask.values]
            )
        
        return profile_data
    
    def _prepare_hour_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for a specific hour.
        
        Args:
            X: Feature DataFrame for specific hour
        
        Returns:
            Feature array
        """
        features = []
        
        for idx, row in X.iterrows():
            feature_vector = [
                idx.dayofweek,  # Day of week (0-6)
                idx.day,        # Day of month (1-31)
                idx.month,      # Month (1-12)
                idx.quarter,    # Quarter (1-4)
                int(idx.dayofweek >= 5),  # Weekend flag
                # Weather features if available
                row.get('temperature', 20.0),
                row.get('humidity', 50.0),
                # Calendar features if available
                int(row.get('is_holiday', 0)),
                int(row.get('is_bridge_day', 0)),
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _prepare_timestamp_features(
        self,
        X: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> np.ndarray:
        """Prepare features for single timestamp."""
        return self._prepare_hour_features(X)
    
    def _optimize_svm_hyperparams(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> SVR:
        """
        Optimize SVM hyperparameters using GridSearchCV.
        
        Args:
            X: Scaled feature array
            y: Target array
        
        Returns:
            Best SVM estimator
        """
        svm = SVR(kernel=self.config.kernel)
        
        grid_search = GridSearchCV(
            svm,
            self.config.param_grid,
            cv=self.config.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=self.config.n_jobs,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_
    
    def _apply_profile_bounds(
        self,
        profile_pred: float,
        hour: int
    ) -> float:
        """
        Apply bounds to profile prediction.
        
        Args:
            profile_pred: Raw profile prediction
            hour: Semi-hourly period (0-47)
        
        Returns:
            Bounded profile prediction
        """
        if hour in self.profile_statistics:
            stats = self.profile_statistics[hour]
            
            # Use training statistics for bounds
            lower_bound = max(
                self.config.min_profile_ratio,
                stats['q25'] - 2 * stats['std']
            )
            upper_bound = min(
                self.config.max_profile_ratio,
                stats['q75'] + 2 * stats['std']
            )
            
            profile_pred = np.clip(profile_pred, lower_bound, upper_bound)
        else:
            # Use config bounds
            profile_pred = np.clip(
                profile_pred,
                self.config.min_profile_ratio,
                self.config.max_profile_ratio
            )
        
        return profile_pred
    
    def save(self, path: str) -> None:
        """Save SVM profile models."""
        import joblib
        from pathlib import Path
        
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # Save all components
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'profile_statistics': self.profile_statistics,
            'config': self.config.dict() if self.config else {},
            'training_metadata': self._training_metadata
        }
        
        joblib.dump(model_data, path_obj / "svm_profile_models.pkl")
        logger.info(f"SVM profile models saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SVMProfileModel':
        """Load SVM profile models."""
        import joblib
        from pathlib import Path
        
        path_obj = Path(path)
        model_data = joblib.load(path_obj / "svm_profile_models.pkl")
        
        model = cls()
        model.models = model_data['models']
        model.scalers = model_data['scalers']
        model.profile_statistics = model_data['profile_statistics']
        model.config = SVMProfileConfig(**model_data['config'])
        model._training_metadata = model_data['training_metadata']
        model._is_fitted = True
        
        logger.info(f"SVM profile models loaded from {path}")
        
        return model
```

### Profile Normalizer

```python
"""Profile normalization utilities."""
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProfileNormalizer:
    """Normalize and denormalize profile ratios."""
    
    def normalize_daily_profiles(
        self,
        profiles: pd.Series,
        target_sum: float = 48.0
    ) -> pd.Series:
        """
        Normalize profiles to sum to target.
        
        Ensures energy conservation: sum of 48 semi-hourly profiles = target
        
        Args:
            profiles: Profile ratios Series (48 values per day)
            target_sum: Target daily sum (default 48 for neutral)
        
        Returns:
            Normalized profiles
        """
        # Group by date
        daily_groups = profiles.groupby(profiles.index.date)
        
        normalized = []
        
        for date, group in daily_groups:
            current_sum = group.sum()
            if current_sum > 0:
                scaling_factor = target_sum / current_sum
                normalized.append(group * scaling_factor)
            else:
                # Uniform if all zeros
                normalized.append(pd.Series(
                    [target_sum / 48] * len(group),
                    index=group.index
                ))
        
        return pd.concat(normalized)
```

---

## 🧪 Testing & Validation

```python
"""Tests for SVM profile models."""
import pytest
import pandas as pd
import numpy as np

from src.models.profile.svm_profile import SVMProfileModel


def test_svm_profile_initialization():
    """Test SVM profile model initialization."""
    model = SVMProfileModel()
    
    assert model.name == "svm_profile"
    assert len(model.models) == 0
    assert not model.is_fitted()


def test_svm_profile_training():
    """Test SVM profile training."""
    # Create sample data with 48 hours
    dates = pd.date_range('2024-01-01', periods=480, freq='30min')
    X = pd.DataFrame({
        'feature1': np.random.randn(480)
    }, index=dates)
    
    # Create profile ratios
    y = pd.Series(1.0 + np.random.randn(480) * 0.2, index=dates)
    y = y.clip(0.5, 2.0)
    
    model = SVMProfileModel()
    config = {'optimize_hyperparams': False, 'n_jobs': 1}
    
    model.fit(X, y, config)
    
    assert model.is_fitted()
    assert len(model.models) > 0


def test_svm_profile_prediction():
    """Test SVM profile prediction."""
    dates = pd.date_range('2024-01-01', periods=240, freq='30min')
    X = pd.DataFrame({'feature1': np.random.randn(240)}, index=dates)
    y = pd.Series(1.0 + np.random.randn(240) * 0.1, index=dates)
    
    model = SVMProfileModel()
    model.fit(X, y, {'optimize_hyperparams': False})
    
    # Predict
    X_test = X.iloc[:48]
    predictions = model.predict(X_test, horizons=[0])
    
    assert 'pred_h0' in predictions.columns
    assert len(predictions) == 48
    assert (predictions['pred_h0'] >= 0.1).all()
    assert (predictions['pred_h0'] <= 5.0).all()
```

---

## 📝 Technical Notes

### Profile Ratio Concept
- Profile ratio = load / daily_demand_mean
- Represents how current period compares to daily average
- Typical range: 0.5 to 2.0
- Sum of 48 ratios ≈ 48 (energy conservation)

### SVM Advantages for Profiles
- Handles non-linear patterns with RBF kernel
- Robust to outliers with epsilon-insensitive loss
- Fast prediction after training

---

## 🔗 Dependencies

**Depends On:**
- PC-030-04: Base Hierarchical Model Interface

**External Dependencies:**
- `scikit-learn>=1.3.0`

**Blocks:**
- PC-033-04: RegDin+SVM Pipeline

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] 48 SVM models train successfully
- [ ] Hyperparameter optimization functional
- [ ] Profile bounds validation working
- [ ] Training time <20 min for 48 models
- [ ] Unit tests pass with >85% coverage
- [ ] Profile ratios in reasonable range
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Previous:** [PC-031-04: ARIMA Demand Mean Model](PC-031-04-arima-demand-mean.md)  
**Next:** [PC-033-04: RegDin+SVM Hierarchical Pipeline](PC-033-04-regdin-svm-pipeline.md)
