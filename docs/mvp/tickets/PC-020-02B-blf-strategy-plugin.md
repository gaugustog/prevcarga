# PC-020-02B: BLF Strategy Plugin

**Ticket ID:** PC-020-02B  
**Epic:** [Epic-02B: Advanced Feature Transformations](../epics/Epic-02B.md)  
**User Story:** US-3  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a Baseline Load Forecast (BLF) strategy plugin for generating intraday correction features. Enables real-time forecast updates by calculating corrections based on recent forecast errors during operational day (D+0).

**As an** operations forecaster  
**I want** intraday baseline load forecast (BLF) features  
**So that** models can incorporate real-time corrections during the day

---

## ✅ Acceptance Criteria

- [ ] `BLFStrategyPlugin` implements sliding window BLF updates
- [ ] Calculates forecast corrections based on recent observations
- [ ] Supports configurable correction horizons (1h, 2h, 4h ahead)
- [ ] Handles missing real-time data gracefully
- [ ] Provides confidence intervals for BLF corrections
- [ ] Integrates with intraday prediction workflows
- [ ] Generates intraday position features (progress, hours remaining)
- [ ] Unit tests validate correction accuracy
- [ ] Performance: <5 seconds for 1 year hourly data

---

## 🔧 Implementation Tasks

### 1. Create BLF Plugin Module
- [ ] Create `src/features/plugins/blf_strategy.py`
- [ ] Add module docstring explaining BLF methodology
- [ ] Document use case for intraday forecasting

### 2. Implement Configuration Schema
- [ ] Create `BLFStrategyConfig` Pydantic model
- [ ] Add `target_column` string field
- [ ] Add `forecast_columns` list (e.g., ["forecast_h1", "forecast_h2"])
- [ ] Add `correction_horizons` list of integers (default: [1, 2, 4])
- [ ] Add `blf_window` integer (rolling window size, default: 12)
- [ ] Add `confidence_level` float (default: 0.95)
- [ ] Add `min_observations` integer (minimum for correction)
- [ ] Validate horizons are positive
- [ ] Validate confidence level in (0, 1)

### 3. Implement BLFStrategyPlugin Class
- [ ] Inherit from `AdvancedFeaturePlugin`
- [ ] Implement `name` property returning "blf_strategy"
- [ ] Implement `version` property
- [ ] Implement `computational_complexity` returning "low"
- [ ] Add class docstring with BLF explanation

### 4. Implement Forecast Error Calculation
- [ ] Create `_calculate_forecast_errors()` method
- [ ] Calculate error: actual - forecast for each horizon
- [ ] Handle missing forecast columns gracefully
- [ ] Return DataFrame with error columns
- [ ] Log error statistics

### 5. Implement BLF Correction Calculation
- [ ] Create `_calculate_blf_correction()` method
- [ ] Calculate rolling mean of recent errors
- [ ] Use configurable window size
- [ ] Shift by 1 to avoid data leakage (use only past errors)
- [ ] Handle insufficient observations
- [ ] Return correction series

### 6. Implement Confidence Interval Calculation
- [ ] Create `_calculate_confidence_intervals()` method
- [ ] Calculate rolling standard deviation of errors
- [ ] Use confidence level to determine interval width
- [ ] Formula: mean ± z * std where z from confidence level
- [ ] Return lower and upper bounds
- [ ] Handle edge cases (constant errors)

### 7. Implement Adaptive Correction Strength
- [ ] Create `_calculate_correction_strength()` method
- [ ] Calculate recent MAPE (Mean Absolute Percentage Error)
- [ ] Compute strength: 1 / (1 + MAPE)
- [ ] Higher accuracy → stronger correction
- [ ] Return strength series (0 to 1)

### 8. Implement Intraday Position Features
- [ ] Create `_generate_intraday_features()` method
- [ ] Calculate intraday progress: hour / 24
- [ ] Calculate hours since midnight
- [ ] Calculate remaining hours: 24 - hour
- [ ] Add is_morning, is_afternoon, is_evening flags
- [ ] Return DataFrame with position features

### 9. Implement generate_features() Method
- [ ] Validate input DataFrame
- [ ] Check target column exists
- [ ] Check forecast columns exist
- [ ] Calculate forecast errors for all horizons
- [ ] Calculate BLF corrections
- [ ] Calculate confidence intervals
- [ ] Calculate adaptive correction strength
- [ ] Generate intraday position features
- [ ] Combine all features
- [ ] Return DataFrame with BLF features

### 10. Implement get_feature_names() Method
- [ ] Generate names for correction features per horizon
- [ ] Generate names for confidence features per horizon
- [ ] Generate names for correction strength features
- [ ] Generate names for intraday position features
- [ ] Return complete list

### 11. Implement supports_incremental_updates()
- [ ] Override to return True
- [ ] BLF is designed for incremental updates
- [ ] Document incremental update procedure

### 12. Create BLF Evaluation Utilities
- [ ] Create `src/features/advanced/blf_evaluator.py`
- [ ] Add `evaluate_blf_improvement()` function
- [ ] Compare accuracy with/without BLF corrections
- [ ] Calculate improvement percentage
- [ ] Generate evaluation report

### 13. Write Comprehensive Tests
- [ ] Create `tests/features/plugins/test_blf_strategy.py`
- [ ] Test basic BLF correction calculation
- [ ] Test with perfect forecasts (corrections should be zero)
- [ ] Test with consistent errors (corrections should match)
- [ ] Test confidence interval coverage
- [ ] Test adaptive correction strength
- [ ] Test missing forecast data handling
- [ ] Test insufficient observations handling
- [ ] Test intraday position features
- [ ] Test incremental update capability

### 14. Create BLF Integration Examples
- [ ] Create `examples/blf_intraday_demo.py`
- [ ] Show BLF correction application
- [ ] Visualize forecast improvement with BLF
- [ ] Demonstrate real-time update scenario
- [ ] Include accuracy metrics

---

## 💻 Implementation Details

### Configuration Schema

```python
"""Configuration for BLF strategy plugin."""
from typing import List
from pydantic import BaseModel, Field, field_validator


class BLFStrategyConfig(BaseModel):
    """Configuration for BLF strategy plugin."""
    
    target_column: str = Field(
        default="carga",
        description="Target column (actual load)"
    )
    forecast_columns: List[str] = Field(
        default_factory=lambda: ["forecast_h1", "forecast_h2", "forecast_h4"],
        description="Forecast columns for different horizons"
    )
    correction_horizons: List[int] = Field(
        default_factory=lambda: [1, 2, 4],
        description="Correction horizons in hours"
    )
    blf_window: int = Field(
        default=12,
        description="Rolling window size for error calculation (periods)"
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for intervals (0-1)"
    )
    min_observations: int = Field(
        default=3,
        description="Minimum observations required for correction"
    )
    
    @field_validator('correction_horizons')
    @classmethod
    def validate_horizons(cls, v):
        """Validate horizons are positive."""
        if not all(h > 0 for h in v):
            raise ValueError("All correction horizons must be positive")
        return sorted(v)
    
    @field_validator('confidence_level')
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence level."""
        if not 0 < v < 1:
            raise ValueError(f"Confidence level must be in (0, 1), got {v}")
        return v
    
    @field_validator('blf_window')
    @classmethod
    def validate_window(cls, v):
        """Validate window size."""
        if v < 1:
            raise ValueError(f"BLF window must be positive, got {v}")
        return v
```

### BLFStrategyPlugin Implementation

```python
"""BLF strategy plugin for intraday forecast corrections."""
from typing import Dict, Any, List, Literal
import pandas as pd
import numpy as np
from scipy import stats

from src.features.base.plugin import AdvancedFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BLFStrategyPlugin(AdvancedFeaturePlugin):
    """
    Plugin for Baseline Load Forecast (BLF) strategy features.
    
    BLF Strategy enables intraday forecast corrections by learning
    from recent forecast errors. As the operational day (D+0) progresses,
    the system accumulates real-time observations and adjusts future
    forecasts based on observed error patterns.
    
    Methodology:
    1. Calculate forecast errors: actual - forecast
    2. Compute rolling statistics of recent errors
    3. Generate correction: mean of recent errors
    4. Apply adaptive weighting based on recent accuracy
    5. Provide confidence intervals for uncertainty quantification
    
    Key Features:
    - blf_correction_h{N}: Mean correction for horizon N
    - blf_confidence_h{N}: Confidence interval width
    - blf_strength_h{N}: Adaptive correction strength (0-1)
    - Intraday position features (progress, hours remaining)
    
    Use Case:
    During operational day at 10:00, if forecasts have consistently
    underestimated load by 50 MW, BLF applies +50 MW correction to
    remaining forecasts for the day.
    
    Example:
        >>> plugin = BLFStrategyPlugin()
        >>> config = {
        ...     "target_column": "carga",
        ...     "forecast_columns": ["forecast_h1", "forecast_h2"],
        ...     "correction_horizons": [1, 2],
        ...     "blf_window": 12
        ... }
        >>> features = plugin.generate_features(df, config)
    """
    
    @property
    def name(self) -> str:
        return "blf_strategy"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def computational_complexity(self) -> Literal["low", "medium", "high"]:
        return "low"
    
    def estimate_compute_time(self, data_size: int) -> float:
        """BLF is very fast: rolling operations are O(n)."""
        return 0.0001 * data_size
    
    def supports_incremental_updates(self) -> bool:
        """BLF is designed for incremental real-time updates."""
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration."""
        BLFStrategyConfig(**config)
        logger.debug("BLF strategy config validated")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate BLF strategy features.
        
        Args:
            df: Input DataFrame with actual and forecast columns
            config: Plugin configuration
        
        Returns:
            DataFrame with BLF features
        """
        # Validate config
        validated_config = BLFStrategyConfig(**config)
        
        # Validate target column
        if validated_config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{validated_config.target_column}' not found"
            )
        
        logger.info(
            f"Generating BLF strategy features for "
            f"{len(validated_config.correction_horizons)} horizons"
        )
        
        # Initialize features
        features = {}
        
        # Process each correction horizon
        for i, horizon in enumerate(validated_config.correction_horizons):
            # Get corresponding forecast column
            if i < len(validated_config.forecast_columns):
                forecast_col = validated_config.forecast_columns[i]
            else:
                logger.warning(
                    f"No forecast column for horizon {horizon}, skipping"
                )
                continue
            
            if forecast_col not in df.columns:
                logger.warning(
                    f"Forecast column '{forecast_col}' not found, skipping"
                )
                continue
            
            # Calculate forecast errors
            errors = df[validated_config.target_column] - df[forecast_col]
            
            # BLF correction (rolling mean of past errors)
            blf_correction = errors.rolling(
                window=validated_config.blf_window,
                min_periods=validated_config.min_observations
            ).mean().shift(1)  # Shift to use only past
            
            features[f'blf_correction_h{horizon}'] = blf_correction
            
            # Confidence intervals (rolling std)
            error_std = errors.rolling(
                window=validated_config.blf_window,
                min_periods=validated_config.min_observations
            ).std().shift(1)
            
            # Z-score for confidence level
            z_score = stats.norm.ppf((1 + validated_config.confidence_level) / 2)
            confidence_width = z_score * error_std
            
            features[f'blf_confidence_h{horizon}'] = confidence_width
            
            # Adaptive correction strength (based on recent accuracy)
            recent_mape = (
                np.abs(errors / (df[validated_config.target_column] + 1e-6))
                .rolling(
                    window=validated_config.blf_window,
                    min_periods=validated_config.min_observations
                )
                .mean()
                .shift(1)
            )
            
            correction_strength = 1 / (1 + recent_mape)
            features[f'blf_strength_h{horizon}'] = correction_strength
        
        # Intraday position features
        if isinstance(df.index, pd.DatetimeIndex):
            features['intraday_progress'] = df.index.hour / 24.0
            features['hours_since_midnight'] = df.index.hour
            features['remaining_hours'] = 24 - df.index.hour
            
            # Time of day flags
            features['is_morning'] = (df.index.hour >= 6) & (df.index.hour < 12)
            features['is_afternoon'] = (df.index.hour >= 12) & (df.index.hour < 18)
            features['is_evening'] = (df.index.hour >= 18) & (df.index.hour < 24)
            features['is_night'] = (df.index.hour >= 0) & (df.index.hour < 6)
        
        # Create result DataFrame
        result = pd.DataFrame(features, index=df.index)
        
        logger.info(f"Generated {len(result.columns)} BLF strategy features")
        return result
    
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """Get feature names."""
        validated_config = BLFStrategyConfig(**config)
        
        names = []
        
        for horizon in validated_config.correction_horizons:
            names.extend([
                f'blf_correction_h{horizon}',
                f'blf_confidence_h{horizon}',
                f'blf_strength_h{horizon}'
            ])
        
        # Intraday position features
        names.extend([
            'intraday_progress',
            'hours_since_midnight',
            'remaining_hours',
            'is_morning',
            'is_afternoon',
            'is_evening',
            'is_night'
        ])
        
        return names
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for BLF strategy plugin."""
import pytest
import pandas as pd
import numpy as np

from src.features.plugins.blf_strategy import BLFStrategyPlugin, BLFStrategyConfig


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return BLFStrategyPlugin()


@pytest.fixture
def sample_df_with_forecasts():
    """Create sample with actual and forecast values."""
    dates = pd.date_range("2024-01-01", periods=200, freq="h")
    
    # Actual load
    actual = 1000 + np.random.randn(200) * 20
    
    # Forecast with systematic bias (underestimate by 30 MW)
    forecast_h1 = actual - 30 + np.random.randn(200) * 10
    forecast_h2 = actual - 25 + np.random.randn(200) * 15
    
    return pd.DataFrame({
        "carga": actual,
        "forecast_h1": forecast_h1,
        "forecast_h2": forecast_h2
    }, index=dates)


def test_basic_blf_correction(plugin, sample_df_with_forecasts):
    """Test basic BLF correction calculation."""
    config = {
        "target_column": "carga",
        "forecast_columns": ["forecast_h1"],
        "correction_horizons": [1],
        "blf_window": 12
    }
    
    result = plugin.generate_features(sample_df_with_forecasts, config)
    
    assert "blf_correction_h1" in result.columns
    assert "blf_confidence_h1" in result.columns
    assert "blf_strength_h1" in result.columns


def test_correction_detects_bias(plugin, sample_df_with_forecasts):
    """Test that BLF correction detects systematic bias."""
    config = {
        "target_column": "carga",
        "forecast_columns": ["forecast_h1"],
        "correction_horizons": [1],
        "blf_window": 12
    }
    
    result = plugin.generate_features(sample_df_with_forecasts, config)
    
    # After sufficient window, correction should be around +30 (bias)
    correction = result["blf_correction_h1"].dropna()
    mean_correction = correction.iloc[50:].mean()
    
    # Should detect the -30 MW bias and suggest +30 correction
    assert 20 < mean_correction < 40


def test_perfect_forecast_zero_correction(plugin):
    """Test that perfect forecasts have zero correction."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    
    # Perfect forecast
    actual = 1000 + np.random.randn(100) * 5
    forecast = actual.copy()
    
    df = pd.DataFrame({
        "carga": actual,
        "forecast_h1": forecast
    }, index=dates)
    
    config = {
        "target_column": "carga",
        "forecast_columns": ["forecast_h1"],
        "correction_horizons": [1]
    }
    
    result = plugin.generate_features(df, config)
    
    correction = result["blf_correction_h1"].dropna()
    
    # Correction should be close to zero
    assert abs(correction.mean()) < 1.0


def test_confidence_intervals(plugin, sample_df_with_forecasts):
    """Test confidence interval calculation."""
    config = {
        "target_column": "carga",
        "forecast_columns": ["forecast_h1"],
        "correction_horizons": [1],
        "confidence_level": 0.95
    }
    
    result = plugin.generate_features(sample_df_with_forecasts, config)
    
    confidence = result["blf_confidence_h1"].dropna()
    
    # Confidence should be positive
    assert (confidence > 0).all()


def test_adaptive_correction_strength(plugin, sample_df_with_forecasts):
    """Test adaptive correction strength calculation."""
    config = {
        "target_column": "carga",
        "forecast_columns": ["forecast_h1"],
        "correction_horizons": [1]
    }
    
    result = plugin.generate_features(sample_df_with_forecasts, config)
    
    strength = result["blf_strength_h1"].dropna()
    
    # Strength should be in (0, 1)
    assert (strength > 0).all()
    assert (strength <= 1).all()


def test_intraday_position_features(plugin, sample_df_with_forecasts):
    """Test intraday position features."""
    config = {
        "target_column": "carga",
        "forecast_columns": ["forecast_h1"],
        "correction_horizons": [1]
    }
    
    result = plugin.generate_features(sample_df_with_forecasts, config)
    
    assert "intraday_progress" in result.columns
    assert "hours_since_midnight" in result.columns
    assert "remaining_hours" in result.columns
    
    # Progress should be in [0, 1)
    assert result["intraday_progress"].between(0, 1).all()


def test_missing_forecast_column(plugin, sample_df_with_forecasts):
    """Test handling of missing forecast column."""
    config = {
        "target_column": "carga",
        "forecast_columns": ["nonexistent_forecast"],
        "correction_horizons": [1]
    }
    
    # Should not raise, but log warning
    result = plugin.generate_features(sample_df_with_forecasts, config)
    
    # Should only have intraday features
    assert "intraday_progress" in result.columns
    assert "blf_correction_h1" not in result.columns


def test_insufficient_observations(plugin):
    """Test handling of insufficient observations."""
    dates = pd.date_range("2024-01-01", periods=5, freq="h")
    df = pd.DataFrame({
        "carga": [1000, 1010, 1020, 1030, 1040],
        "forecast_h1": [990, 1000, 1010, 1020, 1030]
    }, index=dates)
    
    config = {
        "target_column": "carga",
        "forecast_columns": ["forecast_h1"],
        "correction_horizons": [1],
        "blf_window": 12,
        "min_observations": 10
    }
    
    result = plugin.generate_features(df, config)
    
    # Should have NaN for correction (insufficient data)
    assert result["blf_correction_h1"].isna().all()


def test_multiple_horizons(plugin, sample_df_with_forecasts):
    """Test multiple correction horizons."""
    config = {
        "target_column": "carga",
        "forecast_columns": ["forecast_h1", "forecast_h2"],
        "correction_horizons": [1, 2]
    }
    
    result = plugin.generate_features(sample_df_with_forecasts, config)
    
    assert "blf_correction_h1" in result.columns
    assert "blf_correction_h2" in result.columns


def test_incremental_update_support(plugin):
    """Test that plugin supports incremental updates."""
    assert plugin.supports_incremental_updates() is True


def test_config_validation():
    """Test configuration validation."""
    # Invalid confidence level
    with pytest.raises(ValueError, match="Confidence level must be in"):
        BLFStrategyConfig(confidence_level=1.5)
    
    # Invalid horizon
    with pytest.raises(ValueError, match="positive"):
        BLFStrategyConfig(correction_horizons=[0, 1])


def test_feature_names(plugin):
    """Test feature name generation."""
    config = {
        "correction_horizons": [1, 2, 4]
    }
    
    names = plugin.get_feature_names(config)
    
    assert "blf_correction_h1" in names
    assert "blf_confidence_h2" in names
    assert "blf_strength_h4" in names
    assert "intraday_progress" in names
```

---

## 📝 Technical Notes

### BLF Strategy Methodology
- **Rolling Error Analysis**: Tracks recent forecast errors
- **Adaptive Correction**: Stronger corrections when recent accuracy is good
- **Confidence Quantification**: Provides uncertainty estimates
- **Intraday Awareness**: Features track position within operational day

### Use Cases
- Real-time intraday forecast corrections (D+0)
- Adaptive learning from recent forecast performance
- Uncertainty quantification for decision-making
- Operational forecasting improvements

### Integration Points
- Requires forecast columns in input DataFrame
- Works best with rolling/incremental predictions
- Can be combined with model ensemble strategies

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation
- PC-018-02B: LOESS Smoothing Plugin (AdvancedFeaturePlugin)

**External Dependencies:**
- `scipy>=1.11.0` (statistics for confidence intervals)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] BLFStrategyPlugin implemented
- [ ] Correction calculation working correctly
- [ ] Confidence intervals accurate
- [ ] Adaptive correction strength functional
- [ ] Intraday position features generated
- [ ] Incremental update support verified
- [ ] Unit tests pass with >90% coverage
- [ ] Performance benchmark met (<5s for 1 year)
- [ ] Integration with forecasting workflow documented
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-019-02B: Wavelet Transform Plugin](PC-019-02B-wavelet-transform-plugin.md)  
**Next:** [PC-021-02B: RF Feature Selector Plugin](PC-021-02B-rf-feature-selector-plugin.md)
