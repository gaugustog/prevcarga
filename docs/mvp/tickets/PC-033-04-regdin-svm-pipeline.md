# PC-033-04: RegDin+SVM Hierarchical Pipeline

**Ticket ID:** PC-033-04  
**Epic:** [Epic-04: Model Layer - Hierarchical Models](../epics/Epic-04.md)  
**User Story:** US-3  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement complete RegDin+SVM hierarchical model that integrates ARIMA demand mean with SVM profile models. Orchestrates two-stage training, combines predictions maintaining energy conservation, and provides interpretable decomposition of mean vs profile components for all horizons D+0 to D+8.

**As a** forecasting engineer  
**I want** complete RegDin+SVM hierarchical model  
**So that** I can combine ARIMA demand mean with SVM profiles for accurate load forecasting

---

## ✅ Acceptance Criteria

- [ ] Integrates ARIMA demand mean (PC-031) with SVM profiles (PC-032)
- [ ] Inherits from BaseHierarchicalModel (PC-030)
- [ ] Orchestrates two-stage training (demand mean → profiles)
- [ ] Combines predictions maintaining energy conservation
- [ ] Handles horizon-specific demand mean forecasting (D+0 to D+8)
- [ ] Provides interpretable decomposition (mean vs profile components)
- [ ] Validates hierarchical consistency (<5% energy error)
- [ ] Training completes in <30 minutes per area
- [ ] Achieves MAPE within 10% of end-to-end models (baseline)
- [ ] Comprehensive tests with multi-horizon validation
- [ ] Documentation explains RegDin+SVM methodology

---

## 🔧 Implementation Tasks

### 1. Create RegDin+SVM Module
- [ ] Create `src/models/hierarchical/regdin_svm.py`
- [ ] Import ARIMADemandMeanModel
- [ ] Import SVMProfileModel
- [ ] Import BaseHierarchicalModel
- [ ] Add module docstrings

### 2. Implement RegDinSVMModel Class
- [ ] Inherit from BaseHierarchicalModel
- [ ] Implement `name` property returning "regdin_svm"
- [ ] Implement `version` property returning "1.0.0"
- [ ] Implement `supported_horizons` property (0-8)
- [ ] Define demand_mean_model_class as ARIMADemandMeanModel
- [ ] Define profile_model_class as SVMProfileModel

### 3. Implement Demand Mean Data Preparation
- [ ] Override `_prepare_demand_mean_data()` method
- [ ] Aggregate semi-hourly data to daily means
- [ ] Group X by date, take first row
- [ ] Group y by date, calculate mean
- [ ] Convert indices to datetime
- [ ] Return aligned daily X, y
- [ ] Log data transformation

### 4. Implement Profile Data Preparation
- [ ] Override `_prepare_profile_data()` method
- [ ] Expand daily demand mean to semi-hourly
- [ ] Calculate profile ratios: y / demand_mean
- [ ] Handle division by zero (add epsilon)
- [ ] Group by semi-hourly period (0-47)
- [ ] Remove outliers (beyond 3 std)
- [ ] Return dict {hour: (X_hour, profile_ratios)}

### 5. Implement Two-Stage Training
- [ ] Use inherited `fit()` method from BaseHierarchicalModel
- [ ] Stage 1 trains ARIMA on daily data
- [ ] Stage 2 trains 48 SVM models on profiles
- [ ] Validate energy conservation after training
- [ ] Store training metadata
- [ ] Log training progress

### 6. Implement Prediction Combination
- [ ] Use inherited `predict()` method
- [ ] Get ARIMA demand mean forecasts per horizon
- [ ] For each horizon:
  - [ ] Get demand mean value
  - [ ] Predict profiles for each timestamp
  - [ ] Combine: load = demand_mean × profile_ratio
  - [ ] Ensure non-negative values
- [ ] Return DataFrame with all horizons

### 7. Implement Profile Ratio Calculation
- [ ] Create `_calculate_profile_ratios()` method
- [ ] Accept semi-hourly load and daily demand mean
- [ ] Expand demand mean to semi-hourly resolution
- [ ] Calculate ratios with epsilon protection
- [ ] Clip ratios to reasonable bounds
- [ ] Return profile ratios Series

### 8. Implement Energy Conservation Validation
- [ ] Create `validate_energy_conservation()` method
- [ ] Calculate daily sums of predictions
- [ ] Compare to demand mean × 48
- [ ] Compute relative error percentage
- [ ] Raise warning if error > 5%
- [ ] Log conservation metrics

### 9. Implement Outlier Removal
- [ ] Create `_remove_profile_outliers()` method
- [ ] Calculate z-scores for profile ratios
- [ ] Remove samples beyond 3 std from mean
- [ ] Log number of outliers removed
- [ ] Return cleaned data

### 10. Implement Decomposition Analysis
- [ ] Create `get_forecast_decomposition()` method
- [ ] Return demand mean component
- [ ] Return profile component
- [ ] Return combined forecast
- [ ] Calculate contribution percentages
- [ ] Return as structured dict

### 11. Implement Model Diagnostics
- [ ] Create `get_model_diagnostics()` method
- [ ] Get ARIMA diagnostics
- [ ] Get SVM profile statistics
- [ ] Calculate energy conservation error
- [ ] Calculate component correlations
- [ ] Return comprehensive report

### 12. Handle Edge Cases
- [ ] Handle missing demand mean predictions
- [ ] Handle missing profile models (use default)
- [ ] Handle zero demand mean (use epsilon)
- [ ] Handle extreme profile ratios (clip)
- [ ] Validate sufficient training data
- [ ] Raise informative errors

### 13. Implement Model Serialization
- [ ] Use inherited `save()` from BaseHierarchicalModel
- [ ] Save ARIMA demand mean model
- [ ] Save all 48 SVM profile models
- [ ] Save RegDin+SVM metadata
- [ ] Use hierarchical directory structure

### 14. Implement Model Loading
- [ ] Use inherited `load()` from BaseHierarchicalModel
- [ ] Load ARIMA demand mean model
- [ ] Load all SVM profile models
- [ ] Restore RegDin+SVM state
- [ ] Validate model integrity

### 15. Write Comprehensive Tests
- [ ] Create `tests/models/hierarchical/test_regdin_svm.py`
- [ ] Test demand mean data preparation
- [ ] Test profile data preparation
- [ ] Test profile ratio calculation
- [ ] Test two-stage training
- [ ] Test multi-horizon prediction
- [ ] Test energy conservation
- [ ] Test decomposition analysis
- [ ] Test save/load

### 16. Write Integration Tests
- [ ] Test with Epic-02A features
- [ ] Test with real multi-area data
- [ ] Test performance benchmarks vs LGBM
- [ ] Test hierarchical validation suite

### 17. Create Usage Examples
- [ ] Create `examples/regdin_svm_demo.py`
- [ ] Show two-stage training
- [ ] Show multi-horizon forecasting
- [ ] Show decomposition visualization
- [ ] Show energy conservation validation
- [ ] Compare with end-to-end models

---

## 💻 Implementation Details

### RegDin+SVM Model Implementation

```python
"""RegDin+SVM hierarchical model combining ARIMA and SVM."""
from typing import Dict, Any, List, Tuple, Type
import pandas as pd
import numpy as np

from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.models.demand_mean.arima_model import ARIMADemandMeanModel
from src.models.profile.svm_profile import SVMProfileModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RegDinSVMModel(BaseHierarchicalModel):
    """
    RegDin+SVM hierarchical forecasting model.
    
    Combines:
    - ARIMA (RegDin): Demand mean forecasting (daily trends/seasonality)
    - SVM: Profile forecasting (intraday patterns)
    
    Two-stage approach:
    1. Train ARIMA on daily aggregated load data
    2. Calculate profile ratios: load / demand_mean
    3. Train 48 SVM models (one per semi-hourly period)
    4. Forecast: demand_mean × profile_ratio
    
    Benefits:
    - Interpretable decomposition (trend vs pattern)
    - Energy conservation (daily sum matches demand mean)
    - Flexible components (can improve each independently)
    
    Example:
        >>> model = RegDinSVMModel()
        >>> config = {
        ...     'demand_mean': {'season_length': 7},
        ...     'profile': {'optimize_hyperparams': True}
        ... }
        >>> model.fit(X_train, y_train, config)
        >>> predictions = model.predict(X_test, horizons=[0, 1, 2])
        >>> 
        >>> # Get decomposition
        >>> decomp = model.get_decomposition(X_test, horizon=0)
        >>> # decomp['demand_mean'], decomp['profile'], decomp['combined']
    """
    
    @property
    def name(self) -> str:
        return "regdin_svm"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return list(range(0, 9))  # D+0 to D+8
    
    @property
    def demand_mean_model_class(self) -> Type[ARIMADemandMeanModel]:
        return ARIMADemandMeanModel
    
    @property
    def profile_model_class(self) -> Type[SVMProfileModel]:
        return SVMProfileModel
    
    def _prepare_demand_mean_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare daily aggregated data for ARIMA demand mean.
        
        Aggregates semi-hourly data to daily means for ARIMA modeling.
        
        Args:
            X: Semi-hourly feature DataFrame
            y: Semi-hourly target Series
        
        Returns:
            Tuple of (daily_X, daily_y)
        """
        logger.info("Preparing demand mean data (daily aggregation)")
        
        # Aggregate to daily means
        daily_X = X.groupby(X.index.date).first()
        daily_y = y.groupby(y.index.date).mean()
        
        # Align indices
        daily_X.index = pd.to_datetime(daily_X.index)
        daily_y.index = pd.to_datetime(daily_y.index)
        
        logger.info(f"Aggregated to {len(daily_y)} daily observations")
        
        return daily_X, daily_y
    
    def _prepare_profile_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        demand_mean: pd.Series
    ) -> Dict[int, Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare profile ratios (load / demand_mean) by hour.
        
        Calculates semi-hourly profile ratios and groups by period.
        Removes outliers for robust SVM training.
        
        Args:
            X: Semi-hourly feature DataFrame
            y: Semi-hourly target Series
            demand_mean: Daily demand mean predictions
        
        Returns:
            Dictionary mapping hour (0-47) to (X_hour, profile_ratios_hour)
        """
        logger.info("Preparing profile data (ratio calculation)")
        
        # Calculate profile ratios
        profile_ratios = self._calculate_profile_ratios(y, demand_mean)
        
        # Group by semi-hourly periods
        profile_data = {}
        
        for hour in range(48):
            # Filter for this semi-hourly period
            mask = (X.index.hour * 2 + (X.index.minute // 30)) == hour
            
            if mask.sum() == 0:
                continue
            
            hour_X = X[mask]
            hour_profiles = profile_ratios[mask]
            
            # Remove outliers (beyond 3 std from mean)
            mean_ratio = hour_profiles.mean()
            std_ratio = hour_profiles.std()
            
            if std_ratio > 0:
                outlier_mask = np.abs(hour_profiles - mean_ratio) <= 3 * std_ratio
                
                clean_X = hour_X[outlier_mask]
                clean_profiles = hour_profiles[outlier_mask]
                
                if len(clean_profiles) > 10:  # Minimum samples
                    profile_data[hour] = (clean_X, clean_profiles)
                    
                    outliers_removed = (~outlier_mask).sum()
                    if outliers_removed > 0:
                        logger.debug(
                            f"Hour {hour}: removed {outliers_removed} outliers"
                        )
        
        logger.info(f"Prepared profile data for {len(profile_data)}/48 hours")
        
        return profile_data
    
    def _calculate_profile_ratios(
        self,
        y: pd.Series,
        demand_mean_daily: pd.Series
    ) -> pd.Series:
        """
        Calculate profile ratios: y / demand_mean.
        
        Expands daily demand mean to semi-hourly resolution and
        calculates ratios for each timestamp.
        
        Args:
            y: Semi-hourly load Series
            demand_mean_daily: Daily demand mean Series
        
        Returns:
            Profile ratios Series (semi-hourly)
        """
        # Expand daily demand mean to semi-hourly
        # Map each timestamp to its daily demand mean
        daily_demand_mean_expanded = pd.Series(
            index=y.index,
            dtype=float
        )
        
        for timestamp in y.index:
            date = timestamp.date()
            if date in demand_mean_daily.index.date:
                # Find matching date
                date_match = demand_mean_daily.index[
                    demand_mean_daily.index.date == date
                ][0]
                daily_demand_mean_expanded.loc[timestamp] = demand_mean_daily.loc[date_match]
            else:
                # Use nearest available
                daily_demand_mean_expanded.loc[timestamp] = demand_mean_daily.iloc[0]
        
        # Calculate profile ratios with epsilon to avoid division by zero
        epsilon = 1e-8
        profile_ratios = y / (daily_demand_mean_expanded + epsilon)
        
        # Clip extreme ratios
        profile_ratios = profile_ratios.clip(0.01, 10.0)
        
        return profile_ratios
    
    def validate_energy_conservation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.05
    ) -> Dict[str, float]:
        """
        Validate energy conservation in predictions.
        
        Daily sum of semi-hourly predictions should match demand mean × 48.
        
        Args:
            X: Feature DataFrame
            y: Actual target Series
            threshold: Maximum acceptable relative error (default 5%)
        
        Returns:
            Dictionary with conservation metrics
        """
        # Generate predictions
        predictions = self.predict(X, [0])
        
        if 'pred_h0' not in predictions.columns:
            logger.warning("No predictions to validate")
            return {}
        
        # Calculate daily sums
        daily_pred = predictions['pred_h0'].groupby(
            predictions.index.date
        ).sum()
        
        daily_actual = y.groupby(y.index.date).sum()
        
        # Get demand mean predictions
        if 'demand_mean_h0' in predictions.columns:
            daily_dm = predictions['demand_mean_h0'].groupby(
                predictions.index.date
            ).first() * 48  # Demand mean × 48 periods
        else:
            daily_dm = daily_actual  # Fallback
        
        # Align dates
        common_dates = daily_pred.index.intersection(daily_dm.index)
        
        if len(common_dates) == 0:
            logger.warning("No common dates for validation")
            return {}
        
        pred_aligned = daily_pred.loc[common_dates]
        dm_aligned = daily_dm.loc[common_dates]
        
        # Calculate conservation error
        relative_error = np.abs(pred_aligned - dm_aligned) / dm_aligned
        mean_error = relative_error.mean()
        max_error = relative_error.max()
        
        metrics = {
            'mean_conservation_error': mean_error,
            'max_conservation_error': max_error,
            'within_threshold': mean_error <= threshold
        }
        
        if mean_error > threshold:
            logger.warning(
                f"Energy conservation error {mean_error:.2%} exceeds threshold {threshold:.2%}"
            )
        else:
            logger.info(f"Energy conservation validated: {mean_error:.2%} error")
        
        return metrics
    
    def get_forecast_decomposition(
        self,
        X: pd.DataFrame,
        horizon: int = 0
    ) -> Dict[str, pd.Series]:
        """
        Get detailed forecast decomposition.
        
        Args:
            X: Feature DataFrame
            horizon: Forecast horizon
        
        Returns:
            Dictionary with decomposition components
        """
        decomposition = self.get_decomposition(X, horizon)
        
        # Add additional analysis
        if decomposition['demand_mean'] is not None:
            decomposition['profile_contribution_pct'] = (
                (decomposition['profile'] - 1.0) * 100
            )
            
            decomposition['statistics'] = {
                'mean_profile': decomposition['profile'].mean(),
                'std_profile': decomposition['profile'].std(),
                'mean_demand': decomposition['demand_mean'].mean()
            }
        
        return decomposition
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive model diagnostics.
        
        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            'model_name': self.name,
            'version': self.version
        }
        
        # Demand mean diagnostics
        if self.demand_mean_model:
            diagnostics['demand_mean'] = {
                'model_type': self.demand_mean_model.name,
                'fitted': self.demand_mean_model.is_fitted()
            }
        
        # Profile diagnostics
        if self.profile_models:
            diagnostics['profiles'] = {
                'models_trained': len(self.profile_models),
                'coverage': len(self.profile_models) / 48,
                'hours_covered': list(self.profile_models.keys())
            }
        
        # Training metadata
        if hasattr(self, '_training_metadata'):
            diagnostics['training'] = self._training_metadata
        
        return diagnostics
```

---

## 🧪 Testing & Validation

```python
"""Tests for RegDin+SVM hierarchical model."""
import pytest
import pandas as pd
import numpy as np

from src.models.hierarchical.regdin_svm import RegDinSVMModel


@pytest.fixture
def sample_hierarchical_data():
    """Create sample data for hierarchical model."""
    # Create 10 days of semi-hourly data
    dates = pd.date_range('2024-01-01', periods=480, freq='30min')
    
    # Features
    X = pd.DataFrame({
        'temperature': 20 + np.random.randn(480) * 5,
        'humidity': 50 + np.random.randn(480) * 10
    }, index=dates)
    
    # Target with daily trend and intraday pattern
    daily_trend = np.repeat(1000 + np.arange(10) * 10, 48)
    intraday_pattern = np.tile(
        50 * np.sin(2 * np.pi * np.arange(48) / 48),
        10
    )
    noise = np.random.randn(480) * 20
    
    y = pd.Series(daily_trend + intraday_pattern + noise, index=dates)
    
    return X, y


def test_regdin_svm_initialization():
    """Test RegDin+SVM model initialization."""
    model = RegDinSVMModel()
    
    assert model.name == "regdin_svm"
    assert model.supported_horizons == list(range(0, 9))
    assert not model.is_fitted()


def test_regdin_svm_training(sample_hierarchical_data):
    """Test two-stage hierarchical training."""
    X, y = sample_hierarchical_data
    
    model = RegDinSVMModel()
    config = {
        'demand_mean': {
            'season_length': 7,
            'stepwise': True,
            'max_p': 2,
            'max_q': 2
        },
        'profile': {
            'optimize_hyperparams': False,
            'n_jobs': 1
        }
    }
    
    model.fit(X, y, config)
    
    assert model.is_fitted()
    assert model.demand_mean_model is not None
    assert len(model.profile_models) > 0


def test_regdin_svm_prediction(sample_hierarchical_data):
    """Test multi-horizon prediction."""
    X_train, y_train = sample_hierarchical_data
    
    model = RegDinSVMModel()
    config = {
        'demand_mean': {'season_length': 7},
        'profile': {'optimize_hyperparams': False}
    }
    
    model.fit(X_train, y_train, config)
    
    # Predict
    X_test = X_train.iloc[:48]
    predictions = model.predict(X_test, horizons=[0, 1, 2])
    
    assert 'pred_h0' in predictions.columns
    assert 'pred_h1' in predictions.columns
    assert 'demand_mean_h0' in predictions.columns
    assert len(predictions) == 48


def test_regdin_svm_energy_conservation(sample_hierarchical_data):
    """Test energy conservation validation."""
    X, y = sample_hierarchical_data
    
    model = RegDinSVMModel()
    config = {
        'demand_mean': {'season_length': 7},
        'profile': {'optimize_hyperparams': False}
    }
    
    model.fit(X, y, config)
    
    # Validate conservation
    metrics = model.validate_energy_conservation(X, y, threshold=0.15)
    
    assert 'mean_conservation_error' in metrics
    assert metrics['mean_conservation_error'] < 0.20  # 20% tolerance for test


def test_regdin_svm_decomposition(sample_hierarchical_data):
    """Test forecast decomposition."""
    X, y = sample_hierarchical_data
    
    model = RegDinSVMModel()
    config = {
        'demand_mean': {'season_length': 7},
        'profile': {'optimize_hyperparams': False}
    }
    
    model.fit(X, y, config)
    
    # Get decomposition
    decomp = model.get_forecast_decomposition(X.iloc[:48], horizon=0)
    
    assert 'demand_mean' in decomp
    assert 'profile' in decomp
    assert 'combined' in decomp
    assert 'statistics' in decomp
```

---

## 📝 Technical Notes

### RegDin Methodology
- RegDin = Regression of Dynamic components
- Models daily demand mean as time series
- Profiles capture deviations from mean

### Energy Conservation
- Daily sum of loads = demand_mean × 48
- Critical for physical consistency
- Validates hierarchical structure

---

## 🔗 Dependencies

**Depends On:**
- PC-030-04: Base Hierarchical Model Interface
- PC-031-04: ARIMA Demand Mean Model
- PC-032-04: SVM Profile Models

**Blocks:**
- Epic-05 model combination
- Epic-06 hierarchical reconciliation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] RegDin+SVM pipeline trains successfully
- [ ] Multi-horizon prediction functional
- [ ] Energy conservation validated (<5% error)
- [ ] Decomposition analysis working
- [ ] Training time <30 min per area
- [ ] MAPE within 10% of LGBM baseline
- [ ] Unit tests pass with >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Previous:** [PC-032-04: SVM Profile Models](PC-032-04-svm-profile-models.md)  
**Next:** [PC-034-04: Holt-Winters Hierarchical Model](PC-034-04-holt-winters-model.md)
