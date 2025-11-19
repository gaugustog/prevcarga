# PC-031-04: ARIMA Demand Mean Model

**Ticket ID:** PC-031-04  
**Epic:** [Epic-04: Model Layer - Hierarchical Models](../epics/Epic-04.md)  
**User Story:** US-1  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement ARIMA/SARIMA model for demand mean forecasting using statsforecast library. Automatically selects optimal (p,d,q) parameters with information criteria, handles seasonal patterns (weekly, yearly), and generates forecasts for D+0 to D+8 horizons with prediction intervals.

**As a** time series analyst  
**I want** ARIMA model for demand mean forecasting  
**So that** I can capture long-term trends and seasonal patterns in daily energy demand

---

## ✅ Acceptance Criteria

- [ ] Implements ARIMA using statsforecast AutoARIMA
- [ ] Automatically selects optimal (p,d,q) parameters using AIC/BIC
- [ ] Handles seasonal ARIMA (SARIMA) for weekly and yearly patterns
- [ ] Processes daily aggregated load data (24-hour means)
- [ ] Generates forecasts for D+0 to D+8 horizons
- [ ] Provides prediction intervals with 80% and 95% confidence bounds
- [ ] Training completes in <10 minutes per area
- [ ] Comprehensive tests with synthetic seasonal data
- [ ] Documentation explains ARIMA parameter selection

---

## 🔧 Implementation Tasks

### 1. Create Demand Mean Module Structure
- [ ] Create `src/models/demand_mean/__init__.py`
- [ ] Create `src/models/demand_mean/arima_model.py`
- [ ] Create `src/models/demand_mean/demand_processor.py`
- [ ] Add module docstrings

### 2. Implement ARIMA Configuration Schema
- [ ] Create `ARIMAConfig` Pydantic model
- [ ] Add `season_length` field (default: 7 for weekly)
- [ ] Add `max_p`, `max_d`, `max_q` fields
- [ ] Add `max_P`, `max_D`, `max_Q` fields (seasonal)
- [ ] Add `stepwise` boolean (default: True)
- [ ] Add `approximation` boolean
- [ ] Add `information_criterion` enum (aic/bic/aicc)
- [ ] Add `seasonal` boolean

### 3. Implement ARIMADemandMeanModel Class
- [ ] Inherit from `BaseModel`
- [ ] Initialize with empty model
- [ ] Implement `name` property returning "arima_demand_mean"
- [ ] Implement `version` property returning "1.0.0"
- [ ] Implement `supported_horizons` property (0-8)
- [ ] Store seasonal periods configuration

### 4. Implement Data Conversion
- [ ] Create `_convert_to_daily_mean()` method
- [ ] Group semi-hourly data by date
- [ ] Calculate mean per day
- [ ] Handle timezone-aware indices
- [ ] Validate minimum data length
- [ ] Return daily Series with datetime index

### 5. Implement Model Training
- [ ] Implement `fit()` method
- [ ] Convert y to daily means
- [ ] Prepare statsforecast DataFrame format
- [ ] Configure AutoARIMA with parameters
- [ ] Create StatsForecast instance
- [ ] Fit model with daily frequency
- [ ] Store fitted parameters
- [ ] Log selected (p,d,q) orders

### 6. Implement Forecasting
- [ ] Implement `predict()` method
- [ ] Determine maximum horizon
- [ ] Generate forecasts with statsforecast
- [ ] Extract point forecasts per horizon
- [ ] Extract prediction intervals (80%, 95%)
- [ ] Replicate to semi-hourly resolution
- [ ] Return DataFrame with all components

### 7. Implement Seasonal Pattern Detection
- [ ] Create `_detect_seasonality()` method
- [ ] Test for weekly seasonality (7-day cycle)
- [ ] Test for yearly seasonality (365-day cycle)
- [ ] Use statistical tests (e.g., periodogram)
- [ ] Return detected seasonal periods
- [ ] Log seasonality findings

### 8. Implement Parameter Optimization
- [ ] Create `_optimize_parameters()` method
- [ ] Use AutoARIMA stepwise search
- [ ] Evaluate models with information criteria
- [ ] Handle stationarity requirements
- [ ] Store best parameters
- [ ] Log optimization results

### 9. Implement Model Diagnostics
- [ ] Create `get_diagnostics()` method
- [ ] Return fitted values
- [ ] Return residuals
- [ ] Calculate residual statistics (mean, std, ACF)
- [ ] Perform Ljung-Box test
- [ ] Return diagnostic DataFrame

### 10. Implement Demand Processor Utilities
- [ ] Create `DemandProcessor` class
- [ ] Implement `aggregate_to_daily()` method
- [ ] Implement `handle_missing_days()` method
- [ ] Implement `detect_outliers()` method
- [ ] Implement `smooth_demand()` method
- [ ] Add logging for processing steps

### 11. Handle Edge Cases
- [ ] Handle insufficient data (< 14 days)
- [ ] Handle missing days (interpolation)
- [ ] Handle outliers (winsorization)
- [ ] Handle non-stationary series
- [ ] Handle seasonal period mismatches
- [ ] Raise informative errors

### 12. Implement Model Serialization
- [ ] Override `save()` method
- [ ] Save statsforecast model
- [ ] Save configuration
- [ ] Save fitted parameters
- [ ] Save training metadata
- [ ] Use joblib for persistence

### 13. Implement Model Loading
- [ ] Override `load()` classmethod
- [ ] Load statsforecast model
- [ ] Restore configuration
- [ ] Restore fitted parameters
- [ ] Validate model integrity
- [ ] Return loaded model

### 14. Write Comprehensive Tests
- [ ] Create `tests/models/demand_mean/test_arima_model.py`
- [ ] Test daily aggregation
- [ ] Test ARIMA training with synthetic data
- [ ] Test seasonal pattern detection
- [ ] Test multi-horizon forecasting
- [ ] Test prediction intervals
- [ ] Test model save/load
- [ ] Test edge cases

### 15. Write Integration Tests
- [ ] Test with real semi-hourly data
- [ ] Test with multiple seasonal patterns
- [ ] Test performance benchmarks
- [ ] Test hierarchical integration

### 16. Create Usage Examples
- [ ] Create `examples/arima_demand_mean_demo.py`
- [ ] Show daily aggregation
- [ ] Show ARIMA training
- [ ] Show multi-horizon forecasting
- [ ] Visualize predictions with intervals
- [ ] Show seasonal decomposition

---

## 💻 Implementation Details

### ARIMA Configuration

```python
"""Configuration for ARIMA demand mean model."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ARIMAConfig(BaseModel):
    """Configuration for ARIMA demand mean forecasting."""
    
    # Seasonal parameters
    seasonal: bool = Field(
        default=True,
        description="Enable seasonal ARIMA (SARIMA)"
    )
    season_length: int = Field(
        default=7,
        description="Seasonal period (7 = weekly)"
    )
    
    # ARIMA order limits
    max_p: int = Field(
        default=5,
        description="Maximum AR order"
    )
    max_d: int = Field(
        default=2,
        description="Maximum differencing order"
    )
    max_q: int = Field(
        default=5,
        description="Maximum MA order"
    )
    
    # Seasonal order limits
    max_P: int = Field(
        default=2,
        description="Maximum seasonal AR order"
    )
    max_D: int = Field(
        default=1,
        description="Maximum seasonal differencing order"
    )
    max_Q: int = Field(
        default=2,
        description="Maximum seasonal MA order"
    )
    
    # Optimization parameters
    stepwise: bool = Field(
        default=True,
        description="Use stepwise search (faster)"
    )
    approximation: bool = Field(
        default=False,
        description="Use approximation for speed"
    )
    information_criterion: Literal["aic", "bic", "aicc"] = Field(
        default="aic",
        description="Information criterion for model selection"
    )
    
    # Validation parameters
    min_training_days: int = Field(
        default=14,
        description="Minimum days required for training"
    )
```

### ARIMA Demand Mean Model Implementation

```python
"""ARIMA model for demand mean forecasting."""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.models.base.model import BaseModel
from src.models.demand_mean.demand_processor import DemandProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ARIMADemandMeanModel(BaseModel):
    """
    ARIMA model for daily demand mean forecasting.
    
    Uses statsforecast AutoARIMA for automatic parameter selection.
    Handles seasonal patterns (weekly, yearly) with SARIMA.
    Provides prediction intervals for uncertainty quantification.
    
    Features:
    - Automatic (p,d,q) parameter selection
    - Seasonal ARIMA support
    - Multi-horizon forecasting (D+0 to D+8)
    - Prediction intervals (80%, 95%)
    - Model diagnostics
    
    Example:
        >>> model = ARIMADemandMeanModel()
        >>> config = {'season_length': 7, 'stepwise': True}
        >>> model.fit(X_daily, y_daily, config)
        >>> forecasts = model.predict(X_future, horizons=[0, 1, 2, 3])
        >>> # forecasts contains 'pred_h0', 'pred_h0_lo80', 'pred_h0_hi80', etc.
    """
    
    def __init__(self):
        """Initialize ARIMA demand mean model."""
        super().__init__()
        self.model: Optional[Any] = None  # StatsForecast instance
        self.sf_model: Optional[Any] = None  # AutoARIMA model
        self.config: Optional[ARIMAConfig] = None
        self.fitted_params: Dict[str, Any] = {}
        self.processor = DemandProcessor()
    
    @property
    def name(self) -> str:
        return "arima_demand_mean"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return list(range(0, 9))  # D+0 to D+8
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ) -> 'ARIMADemandMeanModel':
        """
        Train ARIMA model on daily demand mean data.
        
        Args:
            X: Feature DataFrame (semi-hourly or daily)
            y: Target Series (semi-hourly or daily)
            config: ARIMA configuration
        
        Returns:
            Self for method chaining
        """
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA
        
        # Validate configuration
        self.config = ARIMAConfig(**config)
        
        # Convert to daily demand mean if needed
        daily_demand = self._convert_to_daily_mean(y)
        
        # Validate minimum data
        if len(daily_demand) < self.config.min_training_days:
            raise ValueError(
                f"Insufficient training data: {len(daily_demand)} days "
                f"(minimum: {self.config.min_training_days})"
            )
        
        logger.info(f"Training ARIMA on {len(daily_demand)} days of demand mean data")
        
        # Process data (handle outliers, missing values)
        daily_demand = self.processor.process_demand_series(daily_demand)
        
        # Prepare data for statsforecast
        sf_data = pd.DataFrame({
            'unique_id': 'demand_mean',
            'ds': daily_demand.index,
            'y': daily_demand.values
        })
        
        # Configure AutoARIMA
        self.sf_model = AutoARIMA(
            season_length=self.config.season_length if self.config.seasonal else 1,
            stepwise=self.config.stepwise,
            approximation=self.config.approximation,
            max_p=self.config.max_p,
            max_q=self.config.max_q,
            max_d=self.config.max_d,
            max_P=self.config.max_P if self.config.seasonal else 0,
            max_Q=self.config.max_Q if self.config.seasonal else 0,
            max_D=self.config.max_D if self.config.seasonal else 0,
            information_criterion=self.config.information_criterion
        )
        
        # Create StatsForecast instance
        self.model = StatsForecast(
            models=[self.sf_model],
            freq='D',  # Daily frequency
            n_jobs=1
        )
        
        # Fit model
        logger.info("Fitting AutoARIMA model...")
        self.model.fit(sf_data)
        
        # Store fitted parameters
        self.fitted_params = {
            'model_type': 'AutoARIMA',
            'training_periods': len(daily_demand),
            'first_date': str(daily_demand.index[0]),
            'last_date': str(daily_demand.index[-1]),
            'seasonal': self.config.seasonal,
            'season_length': self.config.season_length
        }
        
        self._is_fitted = True
        
        logger.info(f"ARIMA model trained successfully on {len(daily_demand)} days")
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Generate ARIMA demand mean forecasts.
        
        Args:
            X: Feature DataFrame (for index alignment)
            horizons: List of horizons to predict (days ahead)
        
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Determine maximum horizon for forecasting
        max_horizon = max(horizons)
        
        # Generate forecasts with prediction intervals
        logger.info(f"Generating forecasts for horizons: {horizons}")
        
        forecasts = self.model.forecast(
            h=max_horizon + 1,
            level=[80, 95]  # Confidence levels
        )
        
        # Convert forecasts to semi-hourly resolution
        predictions = {}
        
        for horizon in horizons:
            if horizon >= len(forecasts):
                logger.warning(f"Horizon {horizon} exceeds forecast length")
                continue
            
            # Get daily forecast for this horizon
            daily_forecast = forecasts.iloc[horizon]['AutoARIMA']
            
            # Replicate to semi-hourly (uniform distribution across day)
            semi_hourly_forecast = pd.Series(
                [daily_forecast] * len(X),
                index=X.index
            )
            
            predictions[f'pred_h{horizon}'] = semi_hourly_forecast
            
            # Add confidence intervals if available
            if 'AutoARIMA-lo-80' in forecasts.columns:
                predictions[f'pred_h{horizon}_lo80'] = pd.Series(
                    [forecasts.iloc[horizon]['AutoARIMA-lo-80']] * len(X),
                    index=X.index
                )
                predictions[f'pred_h{horizon}_hi80'] = pd.Series(
                    [forecasts.iloc[horizon]['AutoARIMA-hi-80']] * len(X),
                    index=X.index
                )
            
            if 'AutoARIMA-lo-95' in forecasts.columns:
                predictions[f'pred_h{horizon}_lo95'] = pd.Series(
                    [forecasts.iloc[horizon]['AutoARIMA-lo-95']] * len(X),
                    index=X.index
                )
                predictions[f'pred_h{horizon}_hi95'] = pd.Series(
                    [forecasts.iloc[horizon]['AutoARIMA-hi-95']] * len(X),
                    index=X.index
                )
        
        return pd.DataFrame(predictions, index=X.index)
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get ARIMA model parameters as 'importance'.
        
        Returns:
            Dictionary with fitted parameters
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted")
        
        return {
            'model_type': 'ARIMA',
            'parameters': self.fitted_params,
            'note': 'ARIMA uses time series patterns, not feature importance'
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get ARIMA model diagnostics.
        
        Returns:
            Dictionary with diagnostic statistics
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted")
        
        # This is a simplified version - statsforecast diagnostics may vary
        diagnostics = {
            'fitted_params': self.fitted_params,
            'model_summary': 'AutoARIMA selected best model'
        }
        
        return diagnostics
    
    def _convert_to_daily_mean(self, semi_hourly_data: pd.Series) -> pd.Series:
        """
        Convert semi-hourly data to daily means.
        
        Args:
            semi_hourly_data: Series with semi-hourly resolution
        
        Returns:
            Series with daily means
        """
        # Group by date and calculate mean
        daily_mean = semi_hourly_data.groupby(semi_hourly_data.index.date).mean()
        
        # Convert index back to datetime
        daily_mean.index = pd.to_datetime(daily_mean.index)
        
        return daily_mean
    
    def save(self, path: str) -> None:
        """Save ARIMA model."""
        import joblib
        from pathlib import Path
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'sf_model': self.sf_model,
            'config': self.config.dict() if self.config else {},
            'fitted_params': self.fitted_params
        }
        
        joblib.dump(model_data, path)
        logger.info(f"ARIMA model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ARIMADemandMeanModel':
        """Load ARIMA model."""
        import joblib
        
        model_data = joblib.load(path)
        
        model = cls()
        model.model = model_data['model']
        model.sf_model = model_data['sf_model']
        model.config = ARIMAConfig(**model_data['config'])
        model.fitted_params = model_data['fitted_params']
        model._is_fitted = True
        
        logger.info(f"ARIMA model loaded from {path}")
        
        return model
```

### Demand Processor Utilities

```python
"""Utilities for demand mean data processing."""
import pandas as pd
import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DemandProcessor:
    """Processes demand mean data for ARIMA modeling."""
    
    def process_demand_series(
        self,
        demand: pd.Series,
        handle_outliers: bool = True,
        fill_missing: bool = True
    ) -> pd.Series:
        """
        Process demand series for robust modeling.
        
        Args:
            demand: Daily demand Series
            handle_outliers: Whether to handle outliers
            fill_missing: Whether to fill missing values
        
        Returns:
            Processed demand Series
        """
        demand = demand.copy()
        
        # Handle missing values
        if fill_missing and demand.isna().any():
            logger.info(f"Filling {demand.isna().sum()} missing values")
            demand = self._fill_missing_values(demand)
        
        # Handle outliers
        if handle_outliers:
            n_outliers = self._detect_outliers(demand).sum()
            if n_outliers > 0:
                logger.info(f"Handling {n_outliers} outliers")
                demand = self._handle_outliers(demand)
        
        return demand
    
    def _fill_missing_values(self, demand: pd.Series) -> pd.Series:
        """Fill missing values with interpolation."""
        return demand.interpolate(method='linear', limit=3)
    
    def _detect_outliers(
        self,
        demand: pd.Series,
        z_threshold: float = 3.0
    ) -> pd.Series:
        """Detect outliers using z-score method."""
        z_scores = np.abs(stats.zscore(demand.dropna()))
        outliers = pd.Series(False, index=demand.index)
        outliers.loc[demand.dropna().index] = z_scores > z_threshold
        return outliers
    
    def _handle_outliers(
        self,
        demand: pd.Series,
        z_threshold: float = 3.0
    ) -> pd.Series:
        """Handle outliers with winsorization."""
        outliers = self._detect_outliers(demand, z_threshold)
        
        if outliers.sum() > 0:
            # Winsorize: cap at 3 std from mean
            mean = demand.mean()
            std = demand.std()
            
            demand = demand.clip(
                lower=mean - z_threshold * std,
                upper=mean + z_threshold * std
            )
        
        return demand
```

---

## 🧪 Testing & Validation

```python
"""Tests for ARIMA demand mean model."""
import pytest
import pandas as pd
import numpy as np

from src.models.demand_mean.arima_model import ARIMADemandMeanModel


def test_arima_initialization():
    """Test ARIMA model initialization."""
    model = ARIMADemandMeanModel()
    
    assert model.name == "arima_demand_mean"
    assert not model.is_fitted()


def test_arima_training():
    """Test ARIMA training with synthetic data."""
    # Create synthetic daily data with trend and seasonality
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    trend = np.linspace(1000, 1100, 100)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(100) / 7)  # Weekly
    noise = np.random.randn(100) * 10
    
    y = pd.Series(trend + seasonal + noise, index=dates)
    X = pd.DataFrame({'dummy': 1}, index=dates)
    
    model = ARIMADemandMeanModel()
    config = {'season_length': 7, 'stepwise': True}
    
    model.fit(X, y, config)
    
    assert model.is_fitted()
    assert model.fitted_params['training_periods'] == 100


def test_arima_forecasting():
    """Test ARIMA multi-horizon forecasting."""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    y = pd.Series(1000 + np.random.randn(50) * 20, index=dates)
    X = pd.DataFrame({'dummy': 1}, index=dates)
    
    model = ARIMADemandMeanModel()
    model.fit(X, y, {'season_length': 7})
    
    # Forecast
    future_dates = pd.date_range('2023-02-20', periods=10, freq='D')
    X_future = pd.DataFrame({'dummy': 1}, index=future_dates)
    
    forecasts = model.predict(X_future, horizons=[0, 1, 2])
    
    assert 'pred_h0' in forecasts.columns
    assert 'pred_h0_lo80' in forecasts.columns
    assert len(forecasts) == 10
```

---

## 📝 Technical Notes

### ARIMA Model Selection
- AutoARIMA uses stepwise search for efficiency
- AIC/BIC penalizes model complexity
- Seasonal patterns detected automatically

### Prediction Intervals
- Based on forecast error variance
- 80% interval: ±1.28σ
- 95% interval: ±1.96σ

---

## 🔗 Dependencies

**Depends On:**
- PC-030-04: Base Hierarchical Model Interface

**External Dependencies:**
- `statsforecast>=1.5.0`
- `scipy>=1.10.0`

**Blocks:**
- PC-033-04: RegDin+SVM Pipeline

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] ARIMA training with AutoARIMA functional
- [ ] Multi-horizon forecasting with intervals
- [ ] Daily aggregation correct
- [ ] Unit tests pass with >85% coverage
- [ ] Training time <10 min per area
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Previous:** [PC-030-04: Base Hierarchical Model Interface](PC-030-04-base-hierarchical-interface.md)  
**Next:** [PC-032-04: SVM Profile Models](PC-032-04-svm-profile-models.md)
