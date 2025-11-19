# PC-018-02B: LOESS Smoothing Plugin

**Ticket ID:** PC-018-02B  
**Epic:** [Epic-02B: Advanced Feature Transformations](../epics/Epic-02B.md)  
**User Story:** US-1  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a LOESS (Locally Weighted Scatterplot Smoothing) plugin for generating trend and seasonal features by filtering noise from time series data. Uses locally weighted regression to capture underlying patterns at multiple time scales.

**As a** time series analyst  
**I want** LOESS-smoothed trend features  
**So that** models can capture underlying trends while filtering out noise

---

## ✅ Acceptance Criteria

- [ ] `LOESSFeaturePlugin` implements LOESS/LOWESS smoothing
- [ ] Configurable smoothing parameters (span, degree, iterations)
- [ ] Handles missing values gracefully without artifacts
- [ ] Generates trend, seasonal, and residual components
- [ ] Supports multiple smoothing windows (0.1, 0.3, 0.7 spans)
- [ ] Performance optimized for large datasets (vectorized operations)
- [ ] Unit tests validate smoothing properties
- [ ] Performance: <30 seconds for 1 year hourly data

---

## 🔧 Implementation Tasks

### 1. Create Advanced Plugin Module Structure
- [ ] Create `src/features/plugins/loess.py`
- [ ] Create `src/features/advanced/` directory
- [ ] Create `src/features/advanced/__init__.py`
- [ ] Create `src/features/advanced/signal_processing.py`
- [ ] Add module docstrings explaining signal processing

### 2. Extend Plugin Base Class
- [ ] Create `AdvancedFeaturePlugin` in `src/features/base/plugin.py`
- [ ] Add `computational_complexity` abstract property
- [ ] Add `estimate_compute_time()` abstract method
- [ ] Add `supports_incremental_updates()` method
- [ ] Add `get_memory_requirements()` method
- [ ] Document advanced plugin interface

### 3. Implement Configuration Schema
- [ ] Create `LOESSFeaturesConfig` Pydantic model
- [ ] Add `target_column` string field
- [ ] Add `smoothing_spans` list of floats (e.g., [0.1, 0.3, 0.7])
- [ ] Add `seasonal_period` integer (default: 48 for semi-hourly)
- [ ] Add `degree` integer (polynomial degree, default: 1)
- [ ] Add `iterations` integer (robustness iterations, default: 2)
- [ ] Add `min_data_points` validation
- [ ] Validate span values in (0, 1] range

### 4. Implement LOESSFeaturePlugin Class
- [ ] Inherit from `AdvancedFeaturePlugin`
- [ ] Implement `name` property returning "loess_features"
- [ ] Implement `version` property with semantic versioning
- [ ] Implement `computational_complexity` returning "medium"
- [ ] Add class docstring with mathematical explanation

### 5. Implement LOESS Smoothing Core
- [ ] Import `statsmodels.nonparametric.smoothers_lowess.lowess`
- [ ] Create `_apply_loess()` method with span parameter
- [ ] Handle missing values by dropping NaN before smoothing
- [ ] Reindex smoothed values to original DataFrame index
- [ ] Add error handling for insufficient data points
- [ ] Log smoothing parameters and data statistics

### 6. Implement Trend Component Extraction
- [ ] Generate LOESS trend for each configured span
- [ ] Create feature names as `loess_trend_{span}`
- [ ] Store trend values aligned with original index
- [ ] Handle edge effects at beginning/end of series
- [ ] Validate trend smoothness properties

### 7. Implement Residual Component
- [ ] Calculate residuals: original - trend
- [ ] Create feature names as `loess_residual_{span}`
- [ ] Store residual values aligned with original index
- [ ] Validate residuals sum to approximately zero
- [ ] Check residual variance reduction

### 8. Implement Seasonal Extraction
- [ ] Create `_extract_seasonality()` method
- [ ] Use rolling mean over seasonal period
- [ ] Extract repeating seasonal pattern from trend
- [ ] Create feature names as `loess_seasonal_{span}`
- [ ] Handle edge cases (insufficient periods)
- [ ] Validate seasonal pattern stability

### 9. Implement Multiple Span Processing
- [ ] Loop through configured smoothing spans
- [ ] Generate all components for each span
- [ ] Combine features into single DataFrame
- [ ] Log feature counts per span
- [ ] Optimize memory by avoiding duplication

### 10. Implement generate_features() Method
- [ ] Validate input DataFrame structure
- [ ] Check target column exists
- [ ] Verify sufficient data points (>= 2 * seasonal_period)
- [ ] Apply LOESS for all configured spans
- [ ] Generate trend, residual, and seasonal components
- [ ] Return combined DataFrame with all features
- [ ] Add comprehensive error handling

### 11. Implement get_feature_names() Method
- [ ] Generate names for trend features
- [ ] Generate names for residual features
- [ ] Generate names for seasonal features
- [ ] Return complete list based on configuration
- [ ] Maintain consistent naming convention

### 12. Implement Performance Estimation
- [ ] Implement `estimate_compute_time()` method
- [ ] Use empirical formula based on data size and spans
- [ ] Consider complexity O(n * span * n) for LOESS
- [ ] Return estimated seconds
- [ ] Implement `get_memory_requirements()` method
- [ ] Estimate based on number of spans and data size

### 13. Create Signal Processing Utilities
- [ ] Create `src/features/advanced/signal_processing.py`
- [ ] Add `validate_signal_quality()` function
- [ ] Add `detect_outliers_in_signal()` function
- [ ] Add `interpolate_missing_values()` function
- [ ] Add `calculate_smoothness_metric()` function
- [ ] Document signal processing theory

### 14. Write Comprehensive Tests
- [ ] Create `tests/features/plugins/test_loess.py`
- [ ] Test basic LOESS smoothing correctness
- [ ] Test trend extraction for different spans
- [ ] Test residual properties (mean ≈ 0)
- [ ] Test seasonal extraction with known patterns
- [ ] Test missing value handling
- [ ] Test insufficient data handling
- [ ] Test edge effects at boundaries
- [ ] Test with synthetic data (known smooth function)
- [ ] Test performance with large datasets

### 15. Create Usage Examples
- [ ] Create `examples/loess_features_demo.py`
- [ ] Show basic LOESS smoothing usage
- [ ] Visualize trend extraction for multiple spans
- [ ] Plot original vs smoothed signals
- [ ] Show residual analysis
- [ ] Include seasonal pattern visualization

---

## 💻 Implementation Details

### AdvancedFeaturePlugin Base Class

```python
"""Extended base class for advanced feature plugins."""
from abc import abstractmethod
from typing import Literal

from src.features.base.plugin import BaseFeaturePlugin


class AdvancedFeaturePlugin(BaseFeaturePlugin):
    """
    Extended base class for advanced feature plugins.
    
    Adds performance estimation and complexity tracking
    for computationally intensive transformations.
    """
    
    @property
    @abstractmethod
    def computational_complexity(self) -> Literal["low", "medium", "high"]:
        """
        Computational complexity level.
        
        Returns:
            'low': <1s for typical datasets
            'medium': 1-30s for typical datasets
            'high': >30s for typical datasets
        """
        pass
    
    @abstractmethod
    def estimate_compute_time(self, data_size: int) -> float:
        """
        Estimate computation time in seconds.
        
        Args:
            data_size: Number of data points
        
        Returns:
            Estimated computation time in seconds
        """
        pass
    
    def supports_incremental_updates(self) -> bool:
        """
        Whether plugin supports incremental feature updates.
        
        Returns:
            True if plugin can update features incrementally
        """
        return False
    
    def get_memory_requirements(self, data_size: int) -> int:
        """
        Estimate memory requirements in MB.
        
        Args:
            data_size: Number of data points
        
        Returns:
            Estimated memory usage in megabytes
        """
        # Default: 0.1 MB per 1000 data points
        return int(data_size * 0.1 / 1000)
```

### Configuration Schema

```python
"""Configuration for LOESS features plugin."""
from typing import List
from pydantic import BaseModel, Field, field_validator


class LOESSFeaturesConfig(BaseModel):
    """Configuration for LOESS smoothing plugin."""
    
    target_column: str = Field(
        default="carga",
        description="Target column to smooth"
    )
    smoothing_spans: List[float] = Field(
        default_factory=lambda: [0.1, 0.3, 0.7],
        description="LOESS smoothing span parameters (0, 1]"
    )
    seasonal_period: int = Field(
        default=48,
        description="Seasonal period for pattern extraction (e.g., 48 for semi-hourly daily)"
    )
    degree: int = Field(
        default=1,
        description="Polynomial degree for local regression (1 or 2)"
    )
    iterations: int = Field(
        default=2,
        description="Number of robustness iterations"
    )
    min_data_points: int = Field(
        default=100,
        description="Minimum data points required for smoothing"
    )
    
    @field_validator('smoothing_spans')
    @classmethod
    def validate_spans(cls, v):
        """Validate span values are in valid range."""
        for span in v:
            if span <= 0 or span > 1:
                raise ValueError(f"Smoothing span must be in (0, 1], got {span}")
        return sorted(v)
    
    @field_validator('degree')
    @classmethod
    def validate_degree(cls, v):
        """Validate polynomial degree."""
        if v not in [1, 2]:
            raise ValueError(f"Degree must be 1 or 2, got {v}")
        return v
    
    @field_validator('seasonal_period')
    @classmethod
    def validate_seasonal_period(cls, v):
        """Validate seasonal period is positive."""
        if v <= 0:
            raise ValueError(f"Seasonal period must be positive, got {v}")
        return v
```

### LOESSFeaturePlugin Implementation

```python
"""LOESS smoothing plugin for trend and seasonal features."""
from typing import Dict, Any, List, Literal
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.features.base.plugin import AdvancedFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LOESSFeaturePlugin(AdvancedFeaturePlugin):
    """
    Plugin for generating LOESS-smoothed trend and seasonal features.
    
    LOESS (Locally Weighted Scatterplot Smoothing) is a non-parametric
    regression method that fits a smooth curve through data points using
    local weighted regression.
    
    Features include:
    - Trend components at multiple smoothing levels
    - Residual components (original - trend)
    - Seasonal components extracted from smoothed trends
    
    Mathematical Background:
    - Uses locally weighted polynomial regression
    - Span parameter controls neighborhood size (% of data)
    - Smaller spans = more detail, larger spans = smoother trends
    
    Example:
        >>> plugin = LOESSFeaturePlugin()
        >>> config = {
        ...     "target_column": "carga",
        ...     "smoothing_spans": [0.1, 0.3, 0.7],
        ...     "seasonal_period": 48
        ... }
        >>> features = plugin.generate_features(df, config)
    """
    
    @property
    def name(self) -> str:
        return "loess_features"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def computational_complexity(self) -> Literal["low", "medium", "high"]:
        return "medium"
    
    def estimate_compute_time(self, data_size: int) -> float:
        """
        Estimate LOESS computation time.
        
        LOESS complexity is approximately O(n²) due to local regressions.
        
        Args:
            data_size: Number of data points
        
        Returns:
            Estimated seconds
        """
        # Empirical formula: 0.00001 * n² seconds
        return 0.00001 * (data_size ** 2)
    
    def get_memory_requirements(self, data_size: int) -> int:
        """Estimate memory for LOESS computations."""
        # Multiple spans, each requiring storage
        return int(data_size * 0.5 / 1000)  # 0.5 MB per 1000 points
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration using Pydantic model."""
        LOESSFeaturesConfig(**config)
        logger.debug("LOESS features config validated")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate LOESS-smoothed features.
        
        Args:
            df: Input DataFrame with time series data
            config: Plugin configuration
        
        Returns:
            DataFrame with LOESS features
        
        Raises:
            ValueError: If target column not found or insufficient data
        """
        # Validate config
        validated_config = LOESSFeaturesConfig(**config)
        
        # Validate target column
        if validated_config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{validated_config.target_column}' "
                f"not found in DataFrame"
            )
        
        # Get target series
        target = df[validated_config.target_column]
        
        # Check sufficient data
        valid_data = target.dropna()
        if len(valid_data) < validated_config.min_data_points:
            raise ValueError(
                f"Insufficient data points: {len(valid_data)} < "
                f"{validated_config.min_data_points}"
            )
        
        logger.info(
            f"Generating LOESS features with {len(validated_config.smoothing_spans)} spans "
            f"on {len(valid_data)} valid data points"
        )
        
        # Initialize feature dictionary
        features = {}
        
        # Process each smoothing span
        for span in validated_config.smoothing_spans:
            logger.debug(f"Processing LOESS with span={span}")
            
            # Apply LOESS smoothing
            trend = self._apply_loess(
                target,
                span=span,
                degree=validated_config.degree,
                iterations=validated_config.iterations
            )
            
            # Store trend feature
            span_str = str(span).replace('.', '_')
            features[f'loess_trend_{span_str}'] = trend
            
            # Calculate residuals
            residual = target - trend
            features[f'loess_residual_{span_str}'] = residual
            
            # Extract seasonal component
            seasonal = self._extract_seasonality(
                trend,
                period=validated_config.seasonal_period
            )
            features[f'loess_seasonal_{span_str}'] = seasonal
        
        # Create result DataFrame
        result = pd.DataFrame(features, index=df.index)
        
        logger.info(f"Generated {len(result.columns)} LOESS features")
        return result
    
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """Get list of feature names based on configuration."""
        validated_config = LOESSFeaturesConfig(**config)
        
        feature_names = []
        
        for span in validated_config.smoothing_spans:
            span_str = str(span).replace('.', '_')
            feature_names.extend([
                f'loess_trend_{span_str}',
                f'loess_residual_{span_str}',
                f'loess_seasonal_{span_str}'
            ])
        
        return feature_names
    
    def _apply_loess(
        self,
        series: pd.Series,
        span: float,
        degree: int = 1,
        iterations: int = 2
    ) -> pd.Series:
        """
        Apply LOESS smoothing to time series.
        
        Args:
            series: Input time series
            span: Smoothing span (fraction of data to use)
            degree: Polynomial degree (1 or 2)
            iterations: Robustness iterations
        
        Returns:
            Smoothed series aligned with original index
        """
        # Drop NaN values for smoothing
        valid_data = series.dropna()
        
        if len(valid_data) == 0:
            logger.warning("No valid data for LOESS smoothing")
            return pd.Series(np.nan, index=series.index)
        
        # Create x values (indices)
        x = np.arange(len(valid_data))
        y = valid_data.values
        
        # Apply LOESS
        smoothed = lowess(
            y,
            x,
            frac=span,
            it=iterations,
            delta=0.0,
            return_sorted=False
        )
        
        # Create series with valid indices
        smoothed_series = pd.Series(smoothed, index=valid_data.index)
        
        # Reindex to original index (will have NaN where original had NaN)
        result = smoothed_series.reindex(series.index)
        
        return result
    
    def _extract_seasonality(
        self,
        series: pd.Series,
        period: int
    ) -> pd.Series:
        """
        Extract seasonal component from smoothed trend.
        
        Args:
            series: Smoothed trend series
            period: Seasonal period
        
        Returns:
            Seasonal component series
        """
        if len(series.dropna()) < 2 * period:
            logger.warning(
                f"Insufficient data for seasonal extraction: "
                f"{len(series.dropna())} < {2 * period}"
            )
            return pd.Series(0.0, index=series.index)
        
        # Calculate rolling mean over seasonal period
        seasonal_mean = series.rolling(
            window=period,
            center=True,
            min_periods=period // 2
        ).mean()
        
        # Seasonal component is deviation from rolling mean
        seasonal = series - seasonal_mean
        
        # Fill NaN at edges with 0
        seasonal = seasonal.fillna(0.0)
        
        return seasonal
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for LOESS features plugin."""
import pytest
import pandas as pd
import numpy as np

from src.features.plugins.loess import LOESSFeaturePlugin, LOESSFeaturesConfig


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return LOESSFeaturePlugin()


@pytest.fixture
def sample_df():
    """Create sample time series with known trend."""
    dates = pd.date_range("2024-01-01", periods=1000, freq="h")
    
    # Generate synthetic data: trend + seasonal + noise
    t = np.arange(1000)
    trend = 0.5 * t + 100
    seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    noise = np.random.randn(1000) * 2
    
    signal = trend + seasonal + noise
    
    return pd.DataFrame({
        "carga": signal,
        "true_trend": trend,
        "true_seasonal": seasonal
    }, index=dates)


def test_basic_loess_smoothing(plugin, sample_df):
    """Test basic LOESS smoothing."""
    config = {
        "target_column": "carga",
        "smoothing_spans": [0.3],
        "seasonal_period": 24
    }
    
    result = plugin.generate_features(sample_df, config)
    
    assert "loess_trend_0_3" in result.columns
    assert "loess_residual_0_3" in result.columns
    assert "loess_seasonal_0_3" in result.columns


def test_trend_extraction_accuracy(plugin, sample_df):
    """Test that LOESS trend approximates true trend."""
    config = {
        "target_column": "carga",
        "smoothing_spans": [0.5],  # Large span for smooth trend
        "seasonal_period": 24
    }
    
    result = plugin.generate_features(sample_df, config)
    
    # LOESS trend should be close to true trend
    trend = result["loess_trend_0_5"].dropna()
    true_trend = sample_df["true_trend"].loc[trend.index]
    
    # Correlation should be high
    correlation = np.corrcoef(trend, true_trend)[0, 1]
    assert correlation > 0.99


def test_residual_properties(plugin, sample_df):
    """Test residual has approximately zero mean."""
    config = {
        "target_column": "carga",
        "smoothing_spans": [0.3]
    }
    
    result = plugin.generate_features(sample_df, config)
    
    residual = result["loess_residual_0_3"].dropna()
    
    # Mean should be close to zero
    assert abs(residual.mean()) < 1.0
    
    # Variance should be reduced compared to original
    original_var = sample_df["carga"].var()
    residual_var = residual.var()
    assert residual_var < original_var


def test_multiple_spans(plugin, sample_df):
    """Test multiple smoothing spans."""
    config = {
        "target_column": "carga",
        "smoothing_spans": [0.1, 0.3, 0.7]
    }
    
    result = plugin.generate_features(sample_df, config)
    
    # Should have 3 features per span
    assert len(result.columns) == 9
    
    # Larger spans should produce smoother trends
    trend_01 = result["loess_trend_0_1"].dropna()
    trend_07 = result["loess_trend_0_7"].dropna()
    
    # Measure smoothness by variance of differences
    smoothness_01 = np.var(np.diff(trend_01))
    smoothness_07 = np.var(np.diff(trend_07))
    
    assert smoothness_07 < smoothness_01


def test_missing_value_handling(plugin):
    """Test handling of missing values."""
    dates = pd.date_range("2024-01-01", periods=200, freq="h")
    data = np.random.randn(200) + 100
    
    # Introduce missing values
    data[50:60] = np.nan
    
    df = pd.DataFrame({"carga": data}, index=dates)
    
    config = {"target_column": "carga", "smoothing_spans": [0.3]}
    
    result = plugin.generate_features(df, config)
    
    # Should handle gracefully
    assert result["loess_trend_0_3"].notna().sum() > 0


def test_insufficient_data_error(plugin):
    """Test error when insufficient data."""
    df = pd.DataFrame({
        "carga": np.random.randn(50) + 100
    }, index=pd.date_range("2024-01-01", periods=50, freq="h"))
    
    config = {
        "target_column": "carga",
        "min_data_points": 100
    }
    
    with pytest.raises(ValueError, match="Insufficient data points"):
        plugin.generate_features(df, config)


def test_config_validation():
    """Test configuration validation."""
    # Invalid span
    with pytest.raises(ValueError, match="Smoothing span must be in"):
        LOESSFeaturesConfig(smoothing_spans=[0.0, 0.5])
    
    with pytest.raises(ValueError, match="Smoothing span must be in"):
        LOESSFeaturesConfig(smoothing_spans=[1.5])
    
    # Invalid degree
    with pytest.raises(ValueError, match="Degree must be 1 or 2"):
        LOESSFeaturesConfig(degree=3)


def test_feature_names(plugin):
    """Test feature name generation."""
    config = {
        "smoothing_spans": [0.1, 0.5]
    }
    
    names = plugin.get_feature_names(config)
    
    assert "loess_trend_0_1" in names
    assert "loess_residual_0_1" in names
    assert "loess_seasonal_0_1" in names
    assert "loess_trend_0_5" in names


def test_performance_estimation(plugin):
    """Test computation time estimation."""
    # Small dataset
    time_1k = plugin.estimate_compute_time(1000)
    
    # Large dataset
    time_10k = plugin.estimate_compute_time(10000)
    
    # Should scale quadratically
    assert time_10k > time_1k * 50  # Approximately 100x


def test_seasonal_extraction(plugin, sample_df):
    """Test seasonal pattern extraction."""
    config = {
        "target_column": "carga",
        "smoothing_spans": [0.5],
        "seasonal_period": 24
    }
    
    result = plugin.generate_features(sample_df, config)
    
    seasonal = result["loess_seasonal_0_5"].dropna()
    
    # Seasonal component should have approximately zero mean
    assert abs(seasonal.mean()) < 1.0
    
    # Should have periodic pattern (check autocorrelation)
    autocorr_24 = seasonal.autocorr(lag=24)
    assert autocorr_24 > 0.5  # Should be positively correlated with period


def test_performance_benchmark(plugin):
    """Test performance with large dataset."""
    import time
    
    # 1 year of hourly data
    dates = pd.date_range("2024-01-01", periods=8760, freq="h")
    df = pd.DataFrame({
        "carga": np.random.randn(8760) + 1000
    }, index=dates)
    
    config = {
        "target_column": "carga",
        "smoothing_spans": [0.1, 0.3, 0.7]
    }
    
    start = time.time()
    result = plugin.generate_features(df, config)
    elapsed = time.time() - start
    
    # Should complete in <30 seconds
    assert elapsed < 30.0
    assert len(result) == len(df)
```

---

## 📝 Technical Notes

### LOESS Algorithm
- **Local Regression**: Fits polynomial at each point using nearby data
- **Weighting**: Points closer to target have higher weight (tricube function)
- **Span Parameter**: Controls neighborhood size (fraction of data)
- **Robustness Iterations**: Reduce influence of outliers
- **Complexity**: O(n²) due to local regressions at each point

### Smoothing Spans
- **0.1**: Captures short-term variations, follows data closely
- **0.3**: Balanced smoothing, good for trend extraction
- **0.7**: Strong smoothing, reveals long-term trends only

### Use Cases
- Denoising time series while preserving trends
- Extracting underlying patterns from volatile data
- Seasonal decomposition after trend removal
- Visual data exploration and smoothing

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation (AdvancedFeaturePlugin extends BaseFeaturePlugin)
- Epic-02A: Core Feature Engineering (provides pipeline infrastructure)

**Blocks:**
- PC-023-02B: Performance Optimization (LOESS is computationally intensive)

**External Dependencies:**
- `statsmodels>=0.14.0` (LOESS/LOWESS implementation)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] LOESSFeaturePlugin implemented with AdvancedFeaturePlugin interface
- [ ] LOESS smoothing produces mathematically correct trends
- [ ] Multiple smoothing spans working correctly
- [ ] Trend, residual, and seasonal components generated
- [ ] Missing value handling robust
- [ ] Unit tests pass with >90% coverage
- [ ] Performance benchmark met (<30s for 1 year hourly)
- [ ] Mathematical validation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next:** [PC-019-02B: Wavelet Transform Plugin](PC-019-02B-wavelet-transform-plugin.md)
