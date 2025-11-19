# PC-015-02A: Lag Features Plugin

**Ticket ID:** PC-015-02A  
**Epic:** [Epic-02A: Core Feature Engineering](../epics/Epic-02A.md)  
**User Story:** US-4  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a plugin for generating lag (autoregressive) features from the target variable and auxiliary variables. Include configurable lag periods, rolling statistics (mean, std, min, max), and efficient computation for large datasets.

**As a** data scientist  
**I want** lag features and rolling statistics  
**So that** the model can learn temporal dependencies and autoregressive patterns

---

## ✅ Acceptance Criteria

- [ ] `LagFeaturesPlugin` creates lag features for target variable
- [ ] Support multiple lag periods (e.g., [1, 2, 24, 168])
- [ ] Rolling window statistics (mean, std, min, max)
- [ ] Configurable window sizes (e.g., 24h, 168h)
- [ ] Efficient computation using pandas shift and rolling
- [ ] NaN handling for initial periods
- [ ] Support multiple input columns (carga, temperatura, etc.)
- [ ] Memory-efficient implementation for large datasets
- [ ] Unit tests with edge cases (small datasets, single lags)
- [ ] Performance: <500ms for 1 year hourly data with 10 lags

---

## 🔧 Implementation Tasks

### 1. Create Plugin Module
- [ ] Create `src/features/plugins/lag.py`
- [ ] Import BaseFeaturePlugin and dependencies
- [ ] Add module docstring with lag feature explanation

### 2. Implement LagFeaturesPlugin Class
- [ ] Inherit from `BaseFeaturePlugin`
- [ ] Implement `name` property returning "lag_features"
- [ ] Implement `version` property with semantic versioning
- [ ] Add class docstring with usage examples

### 3. Implement Configuration Schema
- [ ] Create `LagFeaturesConfig` Pydantic model
- [ ] Add `lag_periods` list of integers (e.g., [1, 2, 24, 168])
- [ ] Add `rolling_windows` list of integers (e.g., [24, 168])
- [ ] Add `rolling_stats` list of strings (e.g., ["mean", "std", "min", "max"])
- [ ] Add `target_column` string (column to lag)
- [ ] Add `additional_columns` list of additional columns to lag
- [ ] Add `fill_method` enum ("none", "forward", "backward")
- [ ] Validate lag_periods are positive integers
- [ ] Validate rolling_windows are positive integers

### 4. Implement Simple Lag Features
- [ ] Create `_create_lag_features()` method
- [ ] Use pandas `shift()` for efficient lagging
- [ ] Generate lags for target column
- [ ] Generate lags for additional columns if specified
- [ ] Name features as `{column}_lag_{period}`
- [ ] Handle edge case: lag period exceeds data length
- [ ] Return DataFrame with lag features

### 5. Implement Rolling Statistics
- [ ] Create `_create_rolling_features()` method
- [ ] Use pandas `rolling()` for efficient computation
- [ ] Implement rolling mean
- [ ] Implement rolling std (standard deviation)
- [ ] Implement rolling min
- [ ] Implement rolling max
- [ ] Name features as `{column}_rolling_{window}_{stat}`
- [ ] Handle edge case: window exceeds data length
- [ ] Return DataFrame with rolling features

### 6. Implement NaN Handling
- [ ] Create `_handle_nans()` method
- [ ] Support "none" strategy (leave NaNs)
- [ ] Support "forward" strategy (forward fill)
- [ ] Support "backward" strategy (backward fill)
- [ ] Log warnings for excessive NaNs
- [ ] Return cleaned DataFrame

### 7. Implement Efficient Batch Processing
- [ ] Create `_batch_create_features()` method
- [ ] Process all lags in single vectorized operation when possible
- [ ] Avoid explicit loops over lag periods
- [ ] Use pandas concat for efficient merging
- [ ] Monitor memory usage during processing

### 8. Implement generate_features() Method
- [ ] Validate input DataFrame structure
- [ ] Check target column exists
- [ ] Generate simple lag features
- [ ] Generate rolling statistics features
- [ ] Apply NaN handling strategy
- [ ] Ensure output index matches input
- [ ] Add error handling for missing columns
- [ ] Log feature generation summary

### 9. Implement get_feature_names() Method
- [ ] Generate names for lag features
- [ ] Generate names for rolling features
- [ ] Return complete list based on configuration
- [ ] Maintain consistent naming convention

### 10. Add Feature Metadata Tracking
- [ ] Create `_generate_metadata()` method
- [ ] Track which lags were created
- [ ] Track which rolling windows were used
- [ ] Record NaN statistics (count, percentage)
- [ ] Include computation time
- [ ] Return metadata dictionary

### 11. Write Comprehensive Tests
- [ ] Create `tests/features/plugins/test_lag.py`
- [ ] Test simple lag creation
- [ ] Test rolling statistics correctness
- [ ] Test multiple lag periods
- [ ] Test NaN handling strategies
- [ ] Test edge case: lag_1 equals previous value
- [ ] Test edge case: single row dataset
- [ ] Test with missing data in input
- [ ] Test memory efficiency with large dataset
- [ ] Test performance benchmarks

### 12. Create Optimization Documentation
- [ ] Document vectorization strategies
- [ ] Explain memory vs computation tradeoffs
- [ ] Provide guidelines for lag selection
- [ ] Include performance benchmarks
- [ ] Add troubleshooting section

---

## 💻 Implementation Details

### Configuration Schema

```python
"""Configuration for lag features plugin."""
from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class LagFeaturesConfig(BaseModel):
    """Configuration for lag features plugin."""
    
    lag_periods: List[int] = Field(
        default_factory=lambda: [1, 2, 24, 168],
        description="List of lag periods (in time steps)"
    )
    rolling_windows: List[int] = Field(
        default_factory=lambda: [24, 168],
        description="List of rolling window sizes"
    )
    rolling_stats: List[str] = Field(
        default_factory=lambda: ["mean", "std", "min", "max"],
        description="Rolling statistics to compute"
    )
    target_column: str = Field(
        default="carga",
        description="Target column to create lags for"
    )
    additional_columns: List[str] = Field(
        default_factory=list,
        description="Additional columns to create lags for"
    )
    fill_method: Literal["none", "forward", "backward"] = Field(
        default="none",
        description="Method to handle NaN values"
    )
    
    @field_validator('lag_periods')
    @classmethod
    def validate_lag_periods(cls, v):
        """Validate lag periods are positive."""
        if not all(lag > 0 for lag in v):
            raise ValueError("All lag periods must be positive integers")
        return sorted(v)  # Sort for consistency
    
    @field_validator('rolling_windows')
    @classmethod
    def validate_rolling_windows(cls, v):
        """Validate rolling windows are positive."""
        if not all(window > 0 for window in v):
            raise ValueError("All rolling windows must be positive integers")
        return sorted(v)
    
    @field_validator('rolling_stats')
    @classmethod
    def validate_rolling_stats(cls, v):
        """Validate rolling stats are supported."""
        valid_stats = {"mean", "std", "min", "max", "median", "sum"}
        for stat in v:
            if stat not in valid_stats:
                raise ValueError(
                    f"Invalid rolling stat: {stat}. "
                    f"Valid options: {valid_stats}"
                )
        return v
```

### LagFeaturesPlugin Implementation

```python
"""Lag features plugin for autoregressive feature engineering."""
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LagFeaturesPlugin(BaseFeaturePlugin):
    """
    Plugin for generating lag features and rolling statistics.
    
    Creates autoregressive features by shifting time series data
    and computing rolling window statistics.
    
    Features include:
    - Simple lags: {column}_lag_{period}
    - Rolling statistics: {column}_rolling_{window}_{stat}
    
    Example:
        >>> plugin = LagFeaturesPlugin()
        >>> config = {
        ...     "lag_periods": [1, 2, 24, 168],
        ...     "rolling_windows": [24, 168],
        ...     "rolling_stats": ["mean", "std"],
        ...     "target_column": "carga"
        ... }
        >>> features = plugin.generate_features(df, config)
    """
    
    @property
    def name(self) -> str:
        return "lag_features"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration using Pydantic model."""
        LagFeaturesConfig(**config)
        logger.debug("Lag features config validated")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate lag features and rolling statistics.
        
        Args:
            df: Input DataFrame with time series data
            config: Plugin configuration
        
        Returns:
            DataFrame with lag features
        
        Raises:
            ValueError: If target column not in DataFrame
            KeyError: If configuration is invalid
        """
        # Validate config
        validated_config = LagFeaturesConfig(**config)
        
        # Validate target column exists
        if validated_config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{validated_config.target_column}' "
                f"not found in DataFrame"
            )
        
        # Get all columns to process
        columns_to_lag = [validated_config.target_column]
        for col in validated_config.additional_columns:
            if col in df.columns:
                columns_to_lag.append(col)
            else:
                logger.warning(f"Additional column '{col}' not found, skipping")
        
        logger.info(
            f"Generating lag features for {len(columns_to_lag)} columns "
            f"with {len(validated_config.lag_periods)} lags and "
            f"{len(validated_config.rolling_windows)} rolling windows"
        )
        
        # Initialize feature list
        lag_dfs = []
        
        # Create simple lag features
        for column in columns_to_lag:
            lag_features = self._create_lag_features(
                df[column],
                validated_config.lag_periods,
                column
            )
            lag_dfs.append(lag_features)
        
        # Create rolling statistics features
        for column in columns_to_lag:
            rolling_features = self._create_rolling_features(
                df[column],
                validated_config.rolling_windows,
                validated_config.rolling_stats,
                column
            )
            lag_dfs.append(rolling_features)
        
        # Concatenate all features
        result = pd.concat(lag_dfs, axis=1)
        
        # Handle NaN values
        if validated_config.fill_method != "none":
            result = self._handle_nans(result, validated_config.fill_method)
        
        # Log NaN statistics
        nan_counts = result.isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(
                f"Generated features contain NaNs: "
                f"{nan_counts[nan_counts > 0].to_dict()}"
            )
        
        logger.info(f"Generated {len(result.columns)} lag features")
        return result
    
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """Get list of feature names based on configuration."""
        validated_config = LagFeaturesConfig(**config)
        
        feature_names = []
        
        # Get columns to process
        columns = [validated_config.target_column] + validated_config.additional_columns
        
        # Add lag feature names
        for column in columns:
            for lag in validated_config.lag_periods:
                feature_names.append(f"{column}_lag_{lag}")
        
        # Add rolling feature names
        for column in columns:
            for window in validated_config.rolling_windows:
                for stat in validated_config.rolling_stats:
                    feature_names.append(f"{column}_rolling_{window}_{stat}")
        
        return feature_names
    
    @staticmethod
    def _create_lag_features(
        series: pd.Series,
        lag_periods: List[int],
        column_name: str
    ) -> pd.DataFrame:
        """
        Create simple lag features for a series.
        
        Args:
            series: Input time series
            lag_periods: List of lag periods
            column_name: Original column name
        
        Returns:
            DataFrame with lag features
        """
        lag_dict = {}
        
        for lag in lag_periods:
            feature_name = f"{column_name}_lag_{lag}"
            lag_dict[feature_name] = series.shift(lag)
        
        return pd.DataFrame(lag_dict, index=series.index)
    
    @staticmethod
    def _create_rolling_features(
        series: pd.Series,
        windows: List[int],
        stats: List[str],
        column_name: str
    ) -> pd.DataFrame:
        """
        Create rolling statistics features for a series.
        
        Args:
            series: Input time series
            windows: List of rolling window sizes
            stats: List of statistics to compute
            column_name: Original column name
        
        Returns:
            DataFrame with rolling features
        """
        rolling_dict = {}
        
        for window in windows:
            rolling_obj = series.rolling(window=window)
            
            for stat in stats:
                feature_name = f"{column_name}_rolling_{window}_{stat}"
                
                if stat == "mean":
                    rolling_dict[feature_name] = rolling_obj.mean()
                elif stat == "std":
                    rolling_dict[feature_name] = rolling_obj.std()
                elif stat == "min":
                    rolling_dict[feature_name] = rolling_obj.min()
                elif stat == "max":
                    rolling_dict[feature_name] = rolling_obj.max()
                elif stat == "median":
                    rolling_dict[feature_name] = rolling_obj.median()
                elif stat == "sum":
                    rolling_dict[feature_name] = rolling_obj.sum()
        
        return pd.DataFrame(rolling_dict, index=series.index)
    
    @staticmethod
    def _handle_nans(df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Handle NaN values in feature DataFrame.
        
        Args:
            df: DataFrame with potential NaNs
            method: Fill method ("forward" or "backward")
        
        Returns:
            DataFrame with NaNs handled
        """
        if method == "forward":
            return df.fillna(method="ffill")
        elif method == "backward":
            return df.fillna(method="bfill")
        else:
            return df
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for lag features plugin."""
import pytest
import pandas as pd
import numpy as np

from src.features.plugins.lag import LagFeaturesPlugin, LagFeaturesConfig


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return LagFeaturesPlugin()


@pytest.fixture
def sample_series():
    """Create sample time series."""
    dates = pd.date_range("2024-01-01", periods=200, freq="h")
    return pd.DataFrame({
        "carga": np.arange(200),
        "temperatura": np.random.randn(200) + 25
    }, index=dates)


def test_simple_lag_creation(plugin, sample_series):
    """Test creation of simple lag features."""
    config = {
        "lag_periods": [1, 2],
        "rolling_windows": [],
        "target_column": "carga"
    }
    result = plugin.generate_features(sample_series, config)
    
    assert "carga_lag_1" in result.columns
    assert "carga_lag_2" in result.columns
    
    # Verify lag_1 is previous value
    assert result["carga_lag_1"].iloc[1] == sample_series["carga"].iloc[0]
    assert result["carga_lag_2"].iloc[2] == sample_series["carga"].iloc[0]


def test_rolling_statistics(plugin, sample_series):
    """Test rolling statistics computation."""
    config = {
        "lag_periods": [],
        "rolling_windows": [24],
        "rolling_stats": ["mean", "std"],
        "target_column": "carga"
    }
    result = plugin.generate_features(sample_series, config)
    
    assert "carga_rolling_24_mean" in result.columns
    assert "carga_rolling_24_std" in result.columns
    
    # Verify rolling mean calculation
    expected_mean = sample_series["carga"].iloc[0:24].mean()
    actual_mean = result["carga_rolling_24_mean"].iloc[23]
    np.testing.assert_almost_equal(actual_mean, expected_mean)


def test_nan_handling_forward_fill(plugin, sample_series):
    """Test forward fill NaN handling."""
    config = {
        "lag_periods": [1],
        "rolling_windows": [],
        "target_column": "carga",
        "fill_method": "forward"
    }
    result = plugin.generate_features(sample_series, config)
    
    # First value should still be NaN (nothing to forward fill from)
    assert pd.isna(result["carga_lag_1"].iloc[0])
    
    # Second value should be filled
    assert not pd.isna(result["carga_lag_1"].iloc[1])


def test_multiple_columns(plugin, sample_series):
    """Test lagging multiple columns."""
    config = {
        "lag_periods": [1],
        "rolling_windows": [],
        "target_column": "carga",
        "additional_columns": ["temperatura"]
    }
    result = plugin.generate_features(sample_series, config)
    
    assert "carga_lag_1" in result.columns
    assert "temperatura_lag_1" in result.columns


def test_feature_names(plugin):
    """Test feature name generation."""
    config = {
        "lag_periods": [1, 24],
        "rolling_windows": [24],
        "rolling_stats": ["mean"],
        "target_column": "carga",
        "additional_columns": []
    }
    names = plugin.get_feature_names(config)
    
    assert "carga_lag_1" in names
    assert "carga_lag_24" in names
    assert "carga_rolling_24_mean" in names


def test_invalid_target_column(plugin, sample_series):
    """Test error when target column doesn't exist."""
    config = {
        "target_column": "nonexistent_column"
    }
    
    with pytest.raises(ValueError, match="not found in DataFrame"):
        plugin.generate_features(sample_series, config)


def test_config_validation_negative_lag():
    """Test that negative lags are rejected."""
    with pytest.raises(ValueError, match="positive integers"):
        LagFeaturesConfig(lag_periods=[1, -1, 24])


def test_config_validation_invalid_stat():
    """Test that invalid rolling stats are rejected."""
    with pytest.raises(ValueError, match="Invalid rolling stat"):
        LagFeaturesConfig(rolling_stats=["mean", "invalid_stat"])


def test_large_lag_period(plugin):
    """Test lag period larger than dataset."""
    small_df = pd.DataFrame({
        "carga": [1, 2, 3]
    }, index=pd.date_range("2024-01-01", periods=3, freq="h"))
    
    config = {
        "lag_periods": [5],  # Larger than dataset
        "rolling_windows": [],
        "target_column": "carga"
    }
    result = plugin.generate_features(small_df, config)
    
    # Should produce all NaNs
    assert result["carga_lag_5"].isna().all()


def test_performance(plugin):
    """Test performance with large dataset."""
    import time
    
    # 1 year of hourly data
    dates = pd.date_range("2024-01-01", periods=8760, freq="h")
    large_df = pd.DataFrame({
        "carga": np.random.randn(8760) + 1000,
        "temperatura": np.random.randn(8760) + 25
    }, index=dates)
    
    config = {
        "lag_periods": [1, 2, 24, 48, 168],
        "rolling_windows": [24, 168],
        "rolling_stats": ["mean", "std", "min", "max"],
        "target_column": "carga"
    }
    
    start = time.time()
    result = plugin.generate_features(large_df, config)
    elapsed = time.time() - start
    
    # Should complete in <500ms
    assert elapsed < 0.5
    assert len(result) == len(large_df)
    
    # Should have all expected features
    expected_features = len(config["lag_periods"]) + \
                       len(config["rolling_windows"]) * len(config["rolling_stats"])
    assert len(result.columns) == expected_features
```

---

## 📝 Technical Notes

- Use pandas `shift()` for memory-efficient lagging
- Use pandas `rolling()` with vectorized operations
- Initial rows will have NaN for lag features (unavoidable)
- Large lag periods (e.g., 168 = 1 week) useful for weekly patterns
- Rolling windows smooth noisy data
- Standard deviation measures volatility/variance
- Typical lags for hourly load forecasting:
  - 1, 2: Recent trend
  - 24: Same hour yesterday
  - 168: Same hour last week
  - 8760: Same hour last year (if available)

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation

**Blocks:**
- PC-017-02A: Feature Pipeline Composer

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] LagFeaturesPlugin implemented
- [ ] Simple lag features working
- [ ] Rolling statistics working (mean, std, min, max)
- [ ] NaN handling strategies implemented
- [ ] Configuration validation complete
- [ ] Unit tests pass with >90% coverage
- [ ] Performance benchmark met (<500ms for 1 year with 10 lags)
- [ ] Code reviewed and approved
- [ ] Memory efficiency validated

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-014-02A: Calendar/Holiday Features Plugin](PC-014-02A-calendar-holiday-features-plugin.md)  
**Next:** [PC-016-02A: Cyclical Encoding Plugin](PC-016-02A-cyclical-encoding-plugin.md)
