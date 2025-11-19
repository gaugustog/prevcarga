# PC-030-04: Base Hierarchical Model Interface

**Ticket ID:** PC-030-04  
**Epic:** [Epic-04: Model Layer - Hierarchical Models](../epics/Epic-04.md)  
**User Story:** Foundation for US-1 through US-6  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement abstract `BaseHierarchicalModel` class that defines the interface for two-stage hierarchical forecasting models. Provides orchestration framework for demand mean forecasting followed by profile modeling, with proper data preparation and combination logic.

**As a** forecasting architect  
**I want** base hierarchical model interface  
**So that** I can implement consistent demand mean + profile decomposition patterns

---

## ✅ Acceptance Criteria

- [ ] Abstract `BaseHierarchicalModel` inherits from `BaseModel` (PC-024-03)
- [ ] Defines abstract methods for demand mean and profile model classes
- [ ] Implements two-stage fit() orchestration (demand mean → profiles)
- [ ] Implements two-stage predict() combination (demand mean × profiles)
- [ ] Provides abstract data preparation methods
- [ ] Includes `ProfileCombiner` utility for forecast reconstruction
- [ ] Validates hierarchical consistency (energy conservation)
- [ ] Comprehensive tests with mock demand mean and profile models
- [ ] Documentation explains hierarchical modeling theory

---

## 🔧 Implementation Tasks

### 1. Create Hierarchical Module Structure
- [ ] Create `src/models/hierarchical/__init__.py`
- [ ] Create `src/models/hierarchical/base_hierarchical.py`
- [ ] Create `src/models/hierarchical/profile_combiner.py`
- [ ] Add module docstrings

### 2. Implement BaseHierarchicalModel Abstract Class
- [ ] Inherit from `BaseModel` (PC-024-03)
- [ ] Add `demand_mean_model` attribute
- [ ] Add `profile_models` dict attribute {hour: model}
- [ ] Add `combiner` attribute (ProfileCombiner)
- [ ] Define abstract property `demand_mean_model_class`
- [ ] Define abstract property `profile_model_class`

### 3. Implement Abstract Data Preparation Methods
- [ ] Define `_prepare_demand_mean_data()` abstract method
- [ ] Define `_prepare_profile_data()` abstract method
- [ ] Define `_prepare_profile_features()` abstract method
- [ ] Add type hints for all methods
- [ ] Document expected data transformations

### 4. Implement Two-Stage fit() Method
- [ ] Accept X, y, config parameters
- [ ] Stage 1: Prepare demand mean data
- [ ] Stage 1: Train demand mean model
- [ ] Stage 1: Generate demand mean predictions
- [ ] Stage 2: Prepare profile data (using demand mean)
- [ ] Stage 2: Train profile models per hour
- [ ] Stage 3: Initialize ProfileCombiner
- [ ] Set `_is_fitted` flag
- [ ] Return self for chaining

### 5. Implement Two-Stage predict() Method
- [ ] Validate model is fitted
- [ ] Stage 1: Predict demand mean for horizons
- [ ] Stage 2: Predict profiles per semi-hourly period
- [ ] Stage 3: Combine demand mean with profiles
- [ ] Return DataFrame with predictions per horizon
- [ ] Include decomposition components

### 6. Implement ProfileCombiner Class
- [ ] Create `combine()` method
- [ ] Accept demand_mean, profiles dict, horizon
- [ ] Match timestamps to semi-hourly periods
- [ ] Apply formula: load = demand_mean × profile_ratio
- [ ] Handle missing profile models gracefully
- [ ] Ensure non-negative values
- [ ] Return combined Series

### 7. Implement Energy Conservation Validation
- [ ] Create `_validate_energy_conservation()` method
- [ ] Calculate daily sums of predictions
- [ ] Compare to daily demand mean × 48
- [ ] Compute conservation error percentage
- [ ] Raise warning if error > 10%
- [ ] Log conservation metrics

### 8. Implement Model Serialization
- [ ] Override `save()` method
- [ ] Save demand mean model
- [ ] Save all profile models
- [ ] Save combiner state
- [ ] Save hierarchical metadata
- [ ] Use consistent directory structure

### 9. Implement Model Loading
- [ ] Override `load()` classmethod
- [ ] Load demand mean model
- [ ] Load all profile models
- [ ] Restore combiner
- [ ] Restore metadata
- [ ] Validate model integrity

### 10. Add Helper Methods
- [ ] Create `get_demand_mean_prediction()` method
- [ ] Create `get_profile_predictions()` method
- [ ] Create `get_decomposition()` method
- [ ] Create `summarize_components()` method
- [ ] Log component statistics

### 11. Write Comprehensive Tests
- [ ] Create `tests/models/hierarchical/test_base_hierarchical.py`
- [ ] Create mock demand mean model
- [ ] Create mock profile model
- [ ] Test two-stage training
- [ ] Test two-stage prediction
- [ ] Test energy conservation validation
- [ ] Test component decomposition
- [ ] Test save/load round-trip

### 12. Write Documentation
- [ ] Document hierarchical modeling theory
- [ ] Explain demand mean + profile decomposition
- [ ] Provide usage examples
- [ ] Document energy conservation principle
- [ ] Add troubleshooting guide

---

## 💻 Implementation Details

### Base Hierarchical Model Interface

```python
"""Base class for hierarchical forecasting models."""
from typing import Dict, Any, List, Tuple, Type, Optional
from abc import abstractmethod
import pandas as pd
import numpy as np

from src.models.base.model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseHierarchicalModel(BaseModel):
    """
    Abstract base class for hierarchical forecasting models.
    
    Hierarchical models decompose load forecasting into two stages:
    1. Demand Mean: Daily average load forecast (time series model)
    2. Profile: Semi-hourly load distribution ratios (ML model)
    
    Final forecast = Demand Mean × Profile Ratio
    
    This ensures:
    - Energy conservation (daily sum matches demand mean)
    - Interpretable decomposition (trend vs intraday pattern)
    - Flexible component modeling (different algorithms per stage)
    
    Subclasses must implement:
    - demand_mean_model_class: Class for demand mean forecasting
    - profile_model_class: Class for profile modeling
    - _prepare_demand_mean_data: Data transformation for stage 1
    - _prepare_profile_data: Data transformation for stage 2
    
    Example:
        >>> class RegDinSVMModel(BaseHierarchicalModel):
        ...     @property
        ...     def demand_mean_model_class(self):
        ...         return ARIMADemandMeanModel
        ...     
        ...     @property
        ...     def profile_model_class(self):
        ...         return SVMProfileModel
        >>> 
        >>> model = RegDinSVMModel()
        >>> model.fit(X_train, y_train, config)
        >>> predictions = model.predict(X_test, horizons=[0, 1, 2])
    """
    
    def __init__(self):
        """Initialize hierarchical model components."""
        super().__init__()
        self.demand_mean_model: Optional[BaseModel] = None
        self.profile_models: Dict[int, BaseModel] = {}
        self.combiner: Optional['ProfileCombiner'] = None
        self._is_fitted = False
    
    @property
    @abstractmethod
    def demand_mean_model_class(self) -> Type[BaseModel]:
        """
        Class for demand mean forecasting.
        
        Returns:
            Model class for daily demand mean (e.g., ARIMADemandMeanModel)
        """
        pass
    
    @property
    @abstractmethod
    def profile_model_class(self) -> Type[BaseModel]:
        """
        Class for profile forecasting.
        
        Returns:
            Model class for semi-hourly profiles (e.g., SVMProfileModel)
        """
        pass
    
    @abstractmethod
    def _prepare_demand_mean_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for demand mean modeling.
        
        Typically aggregates semi-hourly data to daily means.
        
        Args:
            X: Semi-hourly feature DataFrame
            y: Semi-hourly target Series
        
        Returns:
            Tuple of (daily_X, daily_y) for demand mean model
        """
        pass
    
    @abstractmethod
    def _prepare_profile_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        demand_mean: pd.Series
    ) -> Dict[int, Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare data for profile modeling.
        
        Calculates profile ratios: y / demand_mean
        Groups by semi-hourly period (0-47).
        
        Args:
            X: Semi-hourly feature DataFrame
            y: Semi-hourly target Series
            demand_mean: Demand mean predictions for training period
        
        Returns:
            Dictionary mapping hour (0-47) to (X_hour, profile_ratios_hour)
        """
        pass
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ) -> 'BaseHierarchicalModel':
        """
        Train hierarchical model with two-stage approach.
        
        Stage 1: Train demand mean model on daily aggregated data
        Stage 2: Train profile models on semi-hourly ratios
        
        Args:
            X: Feature DataFrame
            y: Target Series
            config: Training configuration with 'demand_mean' and 'profile' keys
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Starting hierarchical training for {self.name}")
        
        # Stage 1: Train demand mean model
        logger.info("Stage 1: Training demand mean model")
        dm_X, dm_y = self._prepare_demand_mean_data(X, y)
        
        self.demand_mean_model = self.demand_mean_model_class()
        self.demand_mean_model.fit(dm_X, dm_y, config.get('demand_mean', {}))
        
        logger.info(f"Demand mean model trained: {self.demand_mean_model.name}")
        
        # Generate demand mean predictions for profile training
        dm_predictions = self.demand_mean_model.predict(dm_X, [0])
        dm_pred = dm_predictions['pred_h0']
        
        # Stage 2: Train profile models
        logger.info("Stage 2: Training profile models")
        profile_data = self._prepare_profile_data(X, y, dm_pred)
        
        trained_count = 0
        for hour, (prof_X, prof_y) in profile_data.items():
            if len(prof_y) < 10:  # Minimum samples
                logger.warning(f"Insufficient data for hour {hour}: {len(prof_y)} samples")
                continue
            
            try:
                profile_model = self.profile_model_class()
                profile_model.fit(prof_X, prof_y, config.get('profile', {}))
                self.profile_models[hour] = profile_model
                trained_count += 1
            except Exception as e:
                logger.error(f"Failed to train profile model for hour {hour}: {e}")
                continue
        
        logger.info(f"Profile models trained: {trained_count}/48 hours")
        
        # Stage 3: Initialize combiner
        self.combiner = ProfileCombiner()
        
        # Validate energy conservation
        self._validate_energy_conservation(X, y)
        
        self._is_fitted = True
        
        # Store training metadata
        self._training_metadata = {
            'demand_mean_model': self.demand_mean_model.name,
            'profile_models_count': len(self.profile_models),
            'profile_coverage': len(self.profile_models) / 48,
            'n_samples': len(y)
        }
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Generate hierarchical predictions.
        
        Stage 1: Predict demand mean for horizons
        Stage 2: Predict profiles for each semi-hourly period
        Stage 3: Combine: load = demand_mean × profile_ratio
        
        Args:
            X: Feature DataFrame
            horizons: List of horizons to predict
        
        Returns:
            DataFrame with predictions and decomposition components
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        
        # Stage 1: Predict demand mean
        dm_X, _ = self._prepare_demand_mean_data(X, pd.Series(index=X.index))
        dm_predictions = self.demand_mean_model.predict(dm_X, horizons)
        
        # Stage 2 & 3: Predict profiles and combine
        for horizon in horizons:
            dm_col = f'pred_h{horizon}'
            
            if dm_col not in dm_predictions.columns:
                logger.warning(f"No demand mean prediction for horizon {horizon}")
                continue
            
            # Get demand mean for this horizon
            demand_mean = dm_predictions[dm_col]
            
            # Predict profiles for each timestamp
            hourly_profiles = {}
            
            for timestamp in X.index:
                hour = timestamp.hour * 2 + (timestamp.minute // 30)
                
                if hour in self.profile_models:
                    # Get single timestamp features
                    hour_X = X.loc[[timestamp]]
                    
                    # Predict profile ratio
                    profile_pred = self.profile_models[hour].predict(hour_X, [0])
                    
                    if 'pred_h0' in profile_pred.columns:
                        if hour not in hourly_profiles:
                            hourly_profiles[hour] = []
                        hourly_profiles[hour].append(profile_pred['pred_h0'].iloc[0])
            
            # Convert hourly_profiles to Series per hour
            profile_series = {}
            idx_per_hour = {}
            
            for timestamp in X.index:
                hour = timestamp.hour * 2 + (timestamp.minute // 30)
                if hour not in idx_per_hour:
                    idx_per_hour[hour] = []
                idx_per_hour[hour].append(timestamp)
            
            for hour, indices in idx_per_hour.items():
                if hour in hourly_profiles:
                    profile_series[hour] = pd.Series(
                        hourly_profiles[hour][:len(indices)],
                        index=indices
                    )
            
            # Combine demand mean with profiles
            combined_forecast = self.combiner.combine(
                demand_mean=demand_mean,
                profiles=profile_series,
                horizon=horizon,
                index=X.index
            )
            
            predictions[f'pred_h{horizon}'] = combined_forecast
            
            # Add decomposition components
            predictions[f'demand_mean_h{horizon}'] = demand_mean
        
        return pd.DataFrame(predictions, index=X.index)
    
    def _validate_energy_conservation(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """
        Validate energy conservation in hierarchical predictions.
        
        Daily sum of semi-hourly predictions should match demand mean × 48.
        
        Args:
            X: Feature DataFrame
            y: Actual target Series
        """
        try:
            # Generate predictions
            predictions = self.predict(X, [0])
            
            if 'pred_h0' not in predictions.columns:
                logger.warning("Cannot validate energy conservation: no predictions")
                return
            
            # Calculate daily sums
            daily_pred = predictions['pred_h0'].groupby(
                predictions.index.date
            ).sum()
            
            daily_actual = y.groupby(y.index.date).sum()
            
            # Align dates
            common_dates = daily_pred.index.intersection(daily_actual.index)
            
            if len(common_dates) == 0:
                logger.warning("No common dates for energy conservation validation")
                return
            
            pred_aligned = daily_pred.loc[common_dates]
            actual_aligned = daily_actual.loc[common_dates]
            
            # Calculate conservation error
            relative_error = np.abs(pred_aligned - actual_aligned) / actual_aligned
            mean_error = relative_error.mean()
            
            if mean_error > 0.10:  # 10% threshold
                logger.warning(
                    f"High energy conservation error: {mean_error:.2%}. "
                    "Check demand mean and profile consistency."
                )
            else:
                logger.info(f"Energy conservation validated: {mean_error:.2%} error")
        
        except Exception as e:
            logger.error(f"Energy conservation validation failed: {e}")
    
    def get_decomposition(
        self,
        X: pd.DataFrame,
        horizon: int = 0
    ) -> Dict[str, pd.Series]:
        """
        Get forecast decomposition into components.
        
        Args:
            X: Feature DataFrame
            horizon: Forecast horizon
        
        Returns:
            Dictionary with 'demand_mean', 'profile', and 'combined' Series
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted")
        
        predictions = self.predict(X, [horizon])
        
        decomposition = {
            'combined': predictions[f'pred_h{horizon}'],
            'demand_mean': predictions.get(f'demand_mean_h{horizon}')
        }
        
        # Calculate profile component
        if decomposition['demand_mean'] is not None:
            decomposition['profile'] = (
                decomposition['combined'] / decomposition['demand_mean']
            )
        
        return decomposition
    
    def save(self, path: str) -> None:
        """Save hierarchical model."""
        from pathlib import Path
        import joblib
        
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # Save demand mean model
        dm_path = path_obj / "demand_mean_model.pkl"
        self.demand_mean_model.save(str(dm_path))
        
        # Save profile models
        profile_dir = path_obj / "profile_models"
        profile_dir.mkdir(exist_ok=True)
        
        for hour, model in self.profile_models.items():
            model_path = profile_dir / f"profile_h{hour}.pkl"
            model.save(str(model_path))
        
        # Save metadata
        metadata = {
            'model_name': self.name,
            'version': self.version,
            'profile_models_hours': list(self.profile_models.keys()),
            'training_metadata': self._training_metadata
        }
        
        metadata_path = path_obj / "hierarchical_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Hierarchical model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseHierarchicalModel':
        """Load hierarchical model."""
        from pathlib import Path
        import joblib
        
        path_obj = Path(path)
        
        # Load metadata
        metadata_path = path_obj / "hierarchical_metadata.pkl"
        metadata = joblib.load(metadata_path)
        
        # Create model instance
        model = cls()
        
        # Load demand mean model
        dm_path = path_obj / "demand_mean_model.pkl"
        model.demand_mean_model = model.demand_mean_model_class.load(str(dm_path))
        
        # Load profile models
        profile_dir = path_obj / "profile_models"
        
        for hour in metadata['profile_models_hours']:
            model_path = profile_dir / f"profile_h{hour}.pkl"
            model.profile_models[hour] = model.profile_model_class.load(str(model_path))
        
        # Restore state
        model.combiner = ProfileCombiner()
        model._is_fitted = True
        model._training_metadata = metadata.get('training_metadata', {})
        
        logger.info(f"Hierarchical model loaded from {path}")
        
        return model
```

### Profile Combiner Implementation

```python
"""Profile combiner for hierarchical forecasting."""
import pandas as pd
import numpy as np
from typing import Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProfileCombiner:
    """
    Combines demand mean forecasts with profile predictions.
    
    Formula: load_forecast = demand_mean × profile_ratio
    
    Ensures:
    - Energy conservation (daily sum matches demand mean)
    - Non-negative values
    - Handles missing profiles gracefully
    
    Example:
        >>> combiner = ProfileCombiner()
        >>> forecast = combiner.combine(
        ...     demand_mean=daily_dm_series,
        ...     profiles={0: profile_h0, 1: profile_h1, ...},
        ...     horizon=0,
        ...     index=semi_hourly_index
        ... )
    """
    
    def __init__(self):
        """Initialize profile combiner."""
        self.default_profile = 1.0  # Neutral profile ratio
    
    def combine(
        self,
        demand_mean: pd.Series,
        profiles: Dict[int, pd.Series],
        horizon: int,
        index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Combine demand mean with profiles.
        
        Args:
            demand_mean: Daily demand mean forecast (replicated to semi-hourly)
            profiles: Dictionary mapping hour (0-47) to profile Series
            horizon: Forecast horizon
            index: Target DatetimeIndex for output
        
        Returns:
            Combined load forecast Series
        """
        combined_forecast = []
        
        for timestamp in index:
            hour = timestamp.hour * 2 + (timestamp.minute // 30)
            
            # Get demand mean for this timestamp
            if timestamp in demand_mean.index:
                dm_value = demand_mean.loc[timestamp]
            else:
                # Use first available value (daily forecast)
                dm_value = demand_mean.iloc[0] if len(demand_mean) > 0 else 1000
            
            # Get profile ratio for this hour
            if hour in profiles and timestamp in profiles[hour].index:
                profile_ratio = profiles[hour].loc[timestamp]
            else:
                # Use default profile
                profile_ratio = self.default_profile
                
            # Clip profile to reasonable bounds
            profile_ratio = np.clip(profile_ratio, 0.1, 5.0)
            
            # Combine: load = demand_mean × profile
            load_forecast = dm_value * profile_ratio
            
            # Ensure non-negative
            load_forecast = max(0, load_forecast)
            
            combined_forecast.append(load_forecast)
        
        return pd.Series(combined_forecast, index=index)
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for base hierarchical model."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel, ProfileCombiner
from src.models.base.model import BaseModel


class MockDemandMeanModel(BaseModel):
    """Mock demand mean model for testing."""
    
    @property
    def name(self) -> str:
        return "mock_demand_mean"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self):
        return [0, 1, 2]
    
    def fit(self, X, y, config):
        self._is_fitted = True
        self.mean_value = y.mean()
        return self
    
    def predict(self, X, horizons):
        pred = {}
        for h in horizons:
            pred[f'pred_h{h}'] = pd.Series([self.mean_value] * len(X), index=X.index)
        return pd.DataFrame(pred)


class MockProfileModel(BaseModel):
    """Mock profile model for testing."""
    
    @property
    def name(self) -> str:
        return "mock_profile"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self):
        return [0]
    
    def fit(self, X, y, config):
        self._is_fitted = True
        self.mean_profile = y.mean()
        return self
    
    def predict(self, X, horizons):
        return pd.DataFrame({
            'pred_h0': [self.mean_profile] * len(X)
        }, index=X.index)


class TestHierarchicalModel(BaseHierarchicalModel):
    """Test implementation of hierarchical model."""
    
    @property
    def name(self):
        return "test_hierarchical"
    
    @property
    def version(self):
        return "1.0.0"
    
    @property
    def supported_horizons(self):
        return [0, 1, 2]
    
    @property
    def demand_mean_model_class(self):
        return MockDemandMeanModel
    
    @property
    def profile_model_class(self):
        return MockProfileModel
    
    def _prepare_demand_mean_data(self, X, y):
        daily_X = X.groupby(X.index.date).first()
        daily_y = y.groupby(y.index.date).mean()
        daily_X.index = pd.to_datetime(daily_X.index)
        daily_y.index = pd.to_datetime(daily_y.index)
        return daily_X, daily_y
    
    def _prepare_profile_data(self, X, y, demand_mean):
        profile_data = {}
        
        for hour in range(48):
            mask = (X.index.hour * 2 + (X.index.minute // 30)) == hour
            if mask.sum() > 0:
                hour_X = X[mask]
                hour_y = y[mask]
                profile_data[hour] = (hour_X, hour_y)
        
        return profile_data


def test_hierarchical_initialization():
    """Test hierarchical model initialization."""
    model = TestHierarchicalModel()
    
    assert model.demand_mean_model is None
    assert len(model.profile_models) == 0
    assert not model.is_fitted()


def test_hierarchical_training():
    """Test two-stage hierarchical training."""
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=96, freq='30min')
    X = pd.DataFrame({'feature1': np.random.randn(96)}, index=dates)
    y = pd.Series(1000 + np.random.randn(96) * 50, index=dates)
    
    model = TestHierarchicalModel()
    config = {'demand_mean': {}, 'profile': {}}
    
    model.fit(X, y, config)
    
    assert model.is_fitted()
    assert model.demand_mean_model is not None
    assert len(model.profile_models) > 0


def test_profile_combiner():
    """Test profile combiner."""
    combiner = ProfileCombiner()
    
    dates = pd.date_range('2024-01-01', periods=48, freq='30min')
    demand_mean = pd.Series([1000] * 48, index=dates)
    profiles = {
        0: pd.Series([1.2] * 2, index=dates[:2]),
        1: pd.Series([0.8] * 2, index=dates[2:4])
    }
    
    combined = combiner.combine(demand_mean, profiles, 0, dates)
    
    assert len(combined) == 48
    assert combined.iloc[0] == 1000 * 1.2  # With profile
```

---

## 📝 Technical Notes

### Hierarchical Decomposition Theory
- **Demand Mean**: Captures daily energy consumption trends
- **Profile**: Captures intraday distribution patterns
- **Energy Conservation**: Daily sum = demand_mean × 48 (semi-hourly periods)

### Two-Stage Training
1. Train demand mean model on daily aggregated data
2. Generate demand mean predictions for training period
3. Calculate profile ratios: load / demand_mean
4. Train 48 profile models (one per semi-hourly period)

---

## 🔗 Dependencies

**Depends On:**
- PC-024-03: Base Model Interface & Registry

**Blocks:**
- PC-031-04: ARIMA Demand Mean Model
- PC-032-04: SVM Profile Models
- PC-033-04: RegDin+SVM Pipeline
- PC-034-04: Holt-Winters Model

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] BaseHierarchicalModel abstract class complete
- [ ] Two-stage fit() and predict() working
- [ ] ProfileCombiner functional
- [ ] Energy conservation validation
- [ ] Unit tests pass with >85% coverage
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** Epic-03 Tickets  
**Next:** [PC-031-04: ARIMA Demand Mean Model](PC-031-04-arima-demand-mean.md)
