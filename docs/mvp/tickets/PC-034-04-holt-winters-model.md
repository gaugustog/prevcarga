# PC-034-04: Holt-Winters Hierarchical Model

**Ticket ID:** PC-034-04  
**Epic:** [Epic-04: Model Layer - Hierarchical Models](../epics/Epic-04.md)  
**User Story:** US-4  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement Holt-Winters hierarchical model using exponential smoothing for both demand mean and profiles. Handles multiple seasonal patterns (daily, weekly, yearly), provides additive/multiplicative options, and generates prediction intervals through simulation or analytical methods.

**As a** forecasting engineer  
**I want** Holt-Winters hierarchical model with seasonal decomposition  
**So that** I can capture trend, seasonality, and intraday patterns in one unified framework

---

## ✅ Acceptance Criteria

- [ ] Implements Holt-Winters for demand mean forecasting
- [ ] Implements Holt-Winters for 48 profile models
- [ ] Inherits from BaseHierarchicalModel (PC-030)
- [ ] Handles multiple seasonal patterns (daily, weekly, yearly)
- [ ] Supports additive and multiplicative seasonality
- [ ] Uses exponential smoothing for trend and seasonal components
- [ ] Generates prediction intervals via simulation
- [ ] Validates seasonal pattern consistency
- [ ] Provides interpretable level/trend/seasonal decomposition
- [ ] Training completes in <25 minutes per area
- [ ] Achieves MAPE competitive with RegDin+SVM
- [ ] Comprehensive tests with seasonal data
- [ ] Documentation explains Holt-Winters methodology

---

## 🔧 Implementation Tasks

### 1. Create Holt-Winters Module
- [ ] Create `src/models/hierarchical/holt_winters.py`
- [ ] Import statsmodels.tsa.holtwinters.ExponentialSmoothing
- [ ] Import BaseHierarchicalModel
- [ ] Add module docstrings

### 2. Implement HoltWintersDemandMeanModel
- [ ] Inherit from base class
- [ ] Implement `name` property returning "holt_winters_demand_mean"
- [ ] Define smoothing parameters: alpha, beta, gamma, phi
- [ ] Define seasonal parameters: seasonal_periods, trend, seasonal, damped_trend
- [ ] Support additive and multiplicative seasonal options

### 3. Implement Demand Mean Configuration
- [ ] Create HoltWintersConfig Pydantic model
- [ ] Parameters: trend (add/mul/None), seasonal (add/mul/None)
- [ ] seasonal_periods (7 for weekly, 365 for yearly)
- [ ] damped_trend (bool), use_boxcox (bool)
- [ ] initialization_method (estimated/heuristic)
- [ ] Validation logic

### 4. Implement Demand Mean Training
- [ ] Prepare daily aggregated data
- [ ] Fit ExponentialSmoothing model
- [ ] Estimate smoothing parameters
- [ ] Store fitted model and parameters
- [ ] Calculate residuals
- [ ] Log training summary

### 5. Implement Demand Mean Forecasting
- [ ] Generate forecasts for horizons 0-8
- [ ] Calculate prediction intervals (80%, 95%)
- [ ] Use analytical or simulation methods
- [ ] Expand daily forecasts to semi-hourly
- [ ] Return forecasts with confidence bounds

### 6. Implement HoltWintersProfileModel
- [ ] Class for 48 profile models
- [ ] Each model captures intraday seasonality
- [ ] Fit separate ExponentialSmoothing per hour
- [ ] Use seasonal_periods=7 (weekly pattern)
- [ ] Store 48 fitted models

### 7. Implement Profile Configuration
- [ ] Create HoltWintersProfileConfig
- [ ] Parameters: trend (add/mul/None), seasonal (add/mul/None)
- [ ] seasonal_periods (default 7)
- [ ] Profile-specific smoothing parameters
- [ ] Validation rules

### 8. Implement Profile Training
- [ ] Calculate profile ratios from demand mean
- [ ] Group by semi-hourly period (0-47)
- [ ] For each period:
  - [ ] Fit ExponentialSmoothing
  - [ ] Estimate smoothing parameters
  - [ ] Store fitted model
- [ ] Handle insufficient data (use defaults)
- [ ] Log training progress

### 9. Implement Profile Forecasting
- [ ] For each hour, generate profile forecast
- [ ] Apply seasonal patterns
- [ ] Calculate prediction intervals
- [ ] Combine with demand mean
- [ ] Return combined load forecasts

### 10. Implement HoltWintersModel Wrapper
- [ ] Inherit from BaseHierarchicalModel
- [ ] Set demand_mean_model_class to HoltWintersDemandMeanModel
- [ ] Set profile_model_class to HoltWintersProfileModel
- [ ] Implement `_prepare_demand_mean_data()`
- [ ] Implement `_prepare_profile_data()`

### 11. Implement Seasonal Decomposition Analysis
- [ ] Create `get_seasonal_decomposition()` method
- [ ] Extract level component
- [ ] Extract trend component
- [ ] Extract seasonal components (daily, weekly)
- [ ] Return structured decomposition
- [ ] Visualize components

### 12. Implement Prediction Intervals
- [ ] Calculate intervals from fitted residuals
- [ ] Support simulation-based intervals
- [ ] Aggregate intervals across hierarchy
- [ ] Validate interval coverage
- [ ] Return intervals with forecasts

### 13. Handle Multiple Seasonal Patterns
- [ ] Support weekly seasonality (7 days)
- [ ] Support yearly seasonality (365 days)
- [ ] Combine seasonal patterns additively
- [ ] Validate pattern consistency
- [ ] Document pattern selection

### 14. Implement Model Diagnostics
- [ ] Create `get_model_diagnostics()` method
- [ ] Report smoothing parameters (alpha, beta, gamma)
- [ ] Report seasonal pattern strength
- [ ] Calculate residual statistics
- [ ] Report model fit metrics (AIC, BIC)
- [ ] Return comprehensive report

### 15. Write Comprehensive Tests
- [ ] Create `tests/models/hierarchical/test_holt_winters.py`
- [ ] Test demand mean training with seasonal data
- [ ] Test profile training
- [ ] Test seasonal decomposition
- [ ] Test prediction intervals
- [ ] Test multiple seasonal patterns
- [ ] Test save/load

### 16. Write Integration Tests
- [ ] Test with Epic-02A features
- [ ] Test with seasonal load data
- [ ] Compare performance vs RegDin+SVM
- [ ] Test hierarchical validation

### 17. Create Usage Examples
- [ ] Create `examples/holt_winters_demo.py`
- [ ] Show seasonal decomposition
- [ ] Show multi-pattern modeling
- [ ] Show prediction intervals
- [ ] Compare with ARIMA baseline

---

## 💻 Implementation Details

### Holt-Winters Demand Mean Model

```python
"""Holt-Winters demand mean forecasting model."""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HoltWintersConfig(BaseModel):
    """Configuration for Holt-Winters demand mean model."""
    
    trend: Optional[str] = Field(
        default='add',
        description="Trend component: 'add', 'mul', or None"
    )
    seasonal: Optional[str] = Field(
        default='add',
        description="Seasonal component: 'add', 'mul', or None"
    )
    seasonal_periods: int = Field(
        default=7,
        description="Number of periods in seasonal cycle (7 for weekly)"
    )
    damped_trend: bool = Field(
        default=False,
        description="Whether to use damped trend"
    )
    use_boxcox: bool = Field(
        default=False,
        description="Whether to apply Box-Cox transformation"
    )
    initialization_method: str = Field(
        default='estimated',
        description="Initialization method: 'estimated' or 'heuristic'"
    )


class HoltWintersDemandMeanModel:
    """
    Holt-Winters exponential smoothing for demand mean forecasting.
    
    Uses exponential smoothing to capture:
    - Level: baseline demand
    - Trend: long-term increase/decrease
    - Seasonality: weekly/yearly patterns
    
    Advantages:
    - Simple, interpretable components
    - Automatic smoothing parameter estimation
    - Handles multiple seasonal patterns
    - Fast training and prediction
    
    Example:
        >>> model = HoltWintersDemandMeanModel()
        >>> config = HoltWintersConfig(
        ...     trend='add',
        ...     seasonal='add',
        ...     seasonal_periods=7
        ... )
        >>> model.fit(X_train, y_train, config)
        >>> forecasts = model.predict(X_test, horizons=[0, 1, 2])
    """
    
    def __init__(self):
        self.model: Optional[ExponentialSmoothing] = None
        self.fitted_model: Optional[Any] = None
        self.config: Optional[HoltWintersConfig] = None
        self._smoothing_params: Dict[str, float] = {}
    
    @property
    def name(self) -> str:
        return "holt_winters_demand_mean"
    
    def is_fitted(self) -> bool:
        return self.fitted_model is not None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Fit Holt-Winters model to daily demand mean data.
        
        Args:
            X: Feature DataFrame (daily)
            y: Target Series (daily demand mean)
            config: Configuration dictionary
        """
        logger.info("Training Holt-Winters demand mean model")
        
        # Parse configuration
        if config is None:
            config = {}
        self.config = HoltWintersConfig(**config)
        
        # Validate sufficient data
        min_samples = self.config.seasonal_periods * 2
        if len(y) < min_samples:
            raise ValueError(
                f"Insufficient data: need {min_samples}, got {len(y)}"
            )
        
        # Create and fit model
        self.model = ExponentialSmoothing(
            y,
            trend=self.config.trend,
            seasonal=self.config.seasonal,
            seasonal_periods=self.config.seasonal_periods,
            damped_trend=self.config.damped_trend,
            use_boxcox=self.config.use_boxcox,
            initialization_method=self.config.initialization_method
        )
        
        self.fitted_model = self.model.fit(optimized=True)
        
        # Store smoothing parameters
        self._smoothing_params = {
            'smoothing_level': self.fitted_model.params['smoothing_level'],
            'smoothing_trend': self.fitted_model.params.get('smoothing_trend'),
            'smoothing_seasonal': self.fitted_model.params.get('smoothing_seasonal'),
            'damping_trend': self.fitted_model.params.get('damping_trend')
        }
        
        logger.info(f"Model fitted with params: {self._smoothing_params}")
    
    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Generate demand mean forecasts for multiple horizons.
        
        Args:
            X: Feature DataFrame (not used for Holt-Winters)
            horizons: List of forecast horizons (0-8)
        
        Returns:
            DataFrame with demand_mean_h{i} columns
        """
        if not self.is_fitted():
            raise RuntimeError("Model not fitted")
        
        predictions = pd.DataFrame(index=X.index)
        
        max_horizon = max(horizons)
        forecasts = self.fitted_model.forecast(steps=max_horizon + 1)
        
        for horizon in horizons:
            col_name = f'demand_mean_h{horizon}'
            predictions[col_name] = forecasts.iloc[horizon]
        
        return predictions
    
    def get_decomposition(self) -> Dict[str, pd.Series]:
        """
        Get seasonal decomposition components.
        
        Returns:
            Dictionary with level, trend, seasonal, residual
        """
        if not self.is_fitted():
            raise RuntimeError("Model not fitted")
        
        return {
            'level': pd.Series(self.fitted_model.level),
            'trend': pd.Series(self.fitted_model.trend) if self.config.trend else None,
            'seasonal': pd.Series(self.fitted_model.season) if self.config.seasonal else None,
            'residual': self.fitted_model.resid
        }
```

### Holt-Winters Profile Model

```python
"""Holt-Winters profile forecasting model."""
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HoltWintersProfileConfig(BaseModel):
    """Configuration for Holt-Winters profile models."""
    
    trend: Optional[str] = Field(
        default=None,
        description="Trend component: 'add', 'mul', or None"
    )
    seasonal: Optional[str] = Field(
        default='add',
        description="Seasonal component: 'add', 'mul', or None"
    )
    seasonal_periods: int = Field(
        default=7,
        description="Number of periods in seasonal cycle"
    )


class HoltWintersProfileModel:
    """
    Holt-Winters exponential smoothing for profile forecasting.
    
    Trains 48 independent models (one per semi-hourly period) to
    capture weekly patterns in profile ratios.
    
    Each model captures:
    - Level: average profile ratio for that hour
    - Seasonality: day-of-week variations
    
    Example:
        >>> model = HoltWintersProfileModel()
        >>> config = {'seasonal_periods': 7}
        >>> profile_data = {0: (X_h0, profiles_h0), ...}
        >>> model.fit(profile_data, config)
        >>> profiles = model.predict(X_test)
    """
    
    def __init__(self):
        self.models: Dict[int, Any] = {}  # hour -> fitted model
        self.config: Optional[HoltWintersProfileConfig] = None
    
    @property
    def name(self) -> str:
        return "holt_winters_profile"
    
    def is_fitted(self) -> bool:
        return len(self.models) > 0
    
    def fit(
        self,
        profile_data: Dict[int, Tuple[pd.DataFrame, pd.Series]],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Fit Holt-Winters models for each semi-hourly period.
        
        Args:
            profile_data: Dict mapping hour to (X, profile_ratios)
            config: Configuration dictionary
        """
        logger.info("Training Holt-Winters profile models")
        
        if config is None:
            config = {}
        self.config = HoltWintersProfileConfig(**config)
        
        for hour, (X_hour, profiles_hour) in profile_data.items():
            if len(profiles_hour) < self.config.seasonal_periods * 2:
                logger.warning(f"Hour {hour}: insufficient data, skipping")
                continue
            
            try:
                model = ExponentialSmoothing(
                    profiles_hour,
                    trend=self.config.trend,
                    seasonal=self.config.seasonal,
                    seasonal_periods=self.config.seasonal_periods
                )
                
                fitted_model = model.fit(optimized=True)
                self.models[hour] = fitted_model
                
            except Exception as e:
                logger.warning(f"Hour {hour}: failed to fit - {e}")
        
        logger.info(f"Trained {len(self.models)}/48 profile models")
    
    def predict(
        self,
        X: pd.DataFrame,
        horizon: int = 0
    ) -> pd.Series:
        """
        Predict profile ratios for all timestamps.
        
        Args:
            X: Feature DataFrame
            horizon: Forecast horizon (0 only supported)
        
        Returns:
            Profile predictions Series
        """
        if not self.is_fitted():
            raise RuntimeError("Model not fitted")
        
        if horizon != 0:
            raise ValueError("Only horizon=0 supported for profiles")
        
        predictions = pd.Series(index=X.index, dtype=float)
        
        for timestamp in X.index:
            hour = timestamp.hour * 2 + (timestamp.minute // 30)
            
            if hour in self.models:
                # Use model forecast (next step)
                forecast = self.models[hour].forecast(steps=1)
                predictions.loc[timestamp] = forecast.iloc[0]
            else:
                # Use default profile ratio
                predictions.loc[timestamp] = 1.0
        
        # Clip to reasonable bounds
        predictions = predictions.clip(0.1, 5.0)
        
        return predictions
```

### Complete Holt-Winters Hierarchical Model

```python
"""Complete Holt-Winters hierarchical model."""
from typing import Dict, Any, Tuple, Type, List
import pandas as pd

from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.models.demand_mean.holt_winters_demand import HoltWintersDemandMeanModel
from src.models.profile.holt_winters_profile import HoltWintersProfileModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HoltWintersModel(BaseHierarchicalModel):
    """
    Complete Holt-Winters hierarchical forecasting model.
    
    Uses exponential smoothing for both components:
    - Demand mean: daily trends and seasonal patterns
    - Profiles: intraday patterns with weekly seasonality
    
    Benefits:
    - Unified exponential smoothing framework
    - Interpretable level/trend/seasonal decomposition
    - Fast training and prediction
    - Automatic parameter estimation
    
    Example:
        >>> model = HoltWintersModel()
        >>> config = {
        ...     'demand_mean': {
        ...         'trend': 'add',
        ...         'seasonal': 'add',
        ...         'seasonal_periods': 7
        ...     },
        ...     'profile': {
        ...         'seasonal': 'add',
        ...         'seasonal_periods': 7
        ...     }
        ... }
        >>> model.fit(X_train, y_train, config)
        >>> predictions = model.predict(X_test, horizons=[0, 1, 2])
    """
    
    @property
    def name(self) -> str:
        return "holt_winters"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return list(range(0, 9))
    
    @property
    def demand_mean_model_class(self) -> Type[HoltWintersDemandMeanModel]:
        return HoltWintersDemandMeanModel
    
    @property
    def profile_model_class(self) -> Type[HoltWintersProfileModel]:
        return HoltWintersProfileModel
    
    def _prepare_demand_mean_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Aggregate to daily data."""
        daily_X = X.groupby(X.index.date).first()
        daily_y = y.groupby(y.index.date).mean()
        
        daily_X.index = pd.to_datetime(daily_X.index)
        daily_y.index = pd.to_datetime(daily_y.index)
        
        return daily_X, daily_y
    
    def _prepare_profile_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        demand_mean: pd.Series
    ) -> Dict[int, Tuple[pd.DataFrame, pd.Series]]:
        """Calculate profile ratios grouped by hour."""
        # Expand demand mean to semi-hourly
        daily_expanded = pd.Series(index=y.index, dtype=float)
        for timestamp in y.index:
            date = timestamp.date()
            match = demand_mean.index[demand_mean.index.date == date]
            if len(match) > 0:
                daily_expanded.loc[timestamp] = demand_mean.loc[match[0]]
        
        # Calculate profile ratios
        profile_ratios = y / (daily_expanded + 1e-8)
        profile_ratios = profile_ratios.clip(0.01, 10.0)
        
        # Group by hour
        profile_data = {}
        for hour in range(48):
            mask = (X.index.hour * 2 + (X.index.minute // 30)) == hour
            if mask.sum() > 0:
                profile_data[hour] = (X[mask], profile_ratios[mask])
        
        return profile_data
    
    def get_seasonal_decomposition(self) -> Dict[str, Any]:
        """Get full seasonal decomposition."""
        decomp = {}
        
        if self.demand_mean_model:
            decomp['demand_mean'] = self.demand_mean_model.get_decomposition()
        
        return decomp
```

---

## 🧪 Testing & Validation

```python
"""Tests for Holt-Winters hierarchical model."""
import pytest
import pandas as pd
import numpy as np

from src.models.hierarchical.holt_winters import HoltWintersModel


@pytest.fixture
def seasonal_data():
    """Create data with trend and seasonality."""
    dates = pd.date_range('2024-01-01', periods=672, freq='30min')  # 14 days
    
    X = pd.DataFrame({
        'temperature': 20 + np.random.randn(672) * 5
    }, index=dates)
    
    # Trend + weekly seasonality + intraday pattern
    days = np.arange(14)
    trend = 1000 + days * 5
    weekly = 50 * np.sin(2 * np.pi * days / 7)
    daily_pattern = trend[:, None] + weekly[:, None]
    
    intraday = 50 * np.sin(2 * np.pi * np.arange(48) / 48)
    
    y_values = (daily_pattern + intraday).flatten() + np.random.randn(672) * 10
    y = pd.Series(y_values, index=dates)
    
    return X, y


def test_holt_winters_training(seasonal_data):
    """Test Holt-Winters hierarchical training."""
    X, y = seasonal_data
    
    model = HoltWintersModel()
    config = {
        'demand_mean': {
            'trend': 'add',
            'seasonal': 'add',
            'seasonal_periods': 7
        },
        'profile': {
            'seasonal': 'add',
            'seasonal_periods': 7
        }
    }
    
    model.fit(X, y, config)
    
    assert model.is_fitted()
    assert model.demand_mean_model is not None


def test_holt_winters_prediction(seasonal_data):
    """Test Holt-Winters forecasting."""
    X, y = seasonal_data
    
    model = HoltWintersModel()
    config = {
        'demand_mean': {'seasonal_periods': 7},
        'profile': {'seasonal_periods': 7}
    }
    
    model.fit(X, y, config)
    predictions = model.predict(X.iloc[:48], horizons=[0, 1])
    
    assert 'pred_h0' in predictions.columns
    assert len(predictions) == 48


def test_holt_winters_decomposition(seasonal_data):
    """Test seasonal decomposition."""
    X, y = seasonal_data
    
    model = HoltWintersModel()
    model.fit(X, y, {'demand_mean': {'seasonal_periods': 7}})
    
    decomp = model.get_seasonal_decomposition()
    assert 'demand_mean' in decomp
```

---

## 📝 Technical Notes

### Exponential Smoothing Components
- Level (α): baseline value
- Trend (β): rate of change
- Seasonality (γ): periodic patterns

### Additive vs Multiplicative
- Additive: seasonal variation constant
- Multiplicative: seasonal variation proportional to level

---

## 🔗 Dependencies

**Depends On:**
- PC-030-04: Base Hierarchical Model Interface

**Blocks:**
- Epic-05 model combination
- Epic-06 hierarchical reconciliation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Holt-Winters demand mean model functional
- [ ] Holt-Winters profile models trained
- [ ] Seasonal decomposition working
- [ ] Prediction intervals generated
- [ ] Training time <25 min per area
- [ ] MAPE competitive with RegDin+SVM
- [ ] Unit tests pass with >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Previous:** [PC-033-04: RegDin+SVM Hierarchical Pipeline](PC-033-04-regdin-svm-pipeline.md)  
**Next:** [PC-035-04: Hierarchical Orchestration & Validation](PC-035-04-hierarchical-orchestration-validation.md)
