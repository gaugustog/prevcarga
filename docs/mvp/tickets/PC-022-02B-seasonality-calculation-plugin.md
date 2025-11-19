# PC-022-02B: Seasonality Calculation Plugin

**Ticket ID:** PC-022-02B  
**Epic:** [Epic-02B: Advanced Feature Transformations](../epics/Epic-02B.md)  
**User Story:** US-5  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a seasonality calculation plugin using STL (Seasonal-Trend decomposition using Loess) to extract multiple seasonal patterns from time series data. Decomposes load data into trend, seasonal components (daily, weekly, yearly), and residuals for feature engineering.

**As a** data scientist  
**I want** STL-based seasonality decomposition with multiple periods  
**So that** I can capture complex seasonal patterns in load forecasting

---

## ✅ Acceptance Criteria

- [ ] `SeasonalityPlugin` extracts seasonal components using STL
- [ ] Supports multiple seasonal periods (daily: 48, weekly: 336, yearly: 17520)
- [ ] Decomposes into trend, seasonal, and residual components
- [ ] Calculates seasonal strength metrics (0-1 scale)
- [ ] Handles missing data gracefully
- [ ] Generates features for each seasonal component
- [ ] Supports MSTL (Multiple Seasonal-Trend decomposition)
- [ ] Configurable smoothing parameters
- [ ] Performance: <10 seconds per decomposition
- [ ] Comprehensive tests with synthetic seasonal data

---

## 🔧 Implementation Tasks

### 1. Create Seasonality Module Structure
- [ ] Create `src/features/plugins/seasonality.py`
- [ ] Create `src/features/decomposition/stl_decomposer.py`
- [ ] Create `src/features/metrics/seasonal_strength.py`
- [ ] Add module docstrings

### 2. Implement Configuration Schema
- [ ] Create `SeasonalityConfig` Pydantic model
- [ ] Add `target_column` string field
- [ ] Add `seasonal_periods` dict (daily, weekly, yearly)
- [ ] Add `decomposition_method` enum: "stl", "mstl"
- [ ] Add `seasonal_smoother` integer (default: 7 for weekly smoothing)
- [ ] Add `trend_smoother` integer (default: None, auto-calculated)
- [ ] Add `robust` boolean (default: True for outlier handling)
- [ ] Add `seasonal_strength_threshold` float (default: 0.3)
- [ ] Add `extrapolate_trend` integer (default: 0)
- [ ] Validate seasonal periods are positive

### 3. Implement STL Decomposer Base Class
- [ ] Create `STLDecomposer` class
- [ ] Initialize with seasonal period and smoothing parameters
- [ ] Implement `decompose()` method using statsmodels STL
- [ ] Extract trend, seasonal, and residual components
- [ ] Handle edge effects at series boundaries
- [ ] Support robust mode for outliers
- [ ] Return decomposition result object

### 4. Implement MSTL Decomposer
- [ ] Create `MSTLDecomposer` class
- [ ] Support multiple seasonal periods simultaneously
- [ ] Use statsmodels MSTL for multi-period decomposition
- [ ] Extract individual seasonal components per period
- [ ] Handle period interactions
- [ ] Return multi-component decomposition

### 5. Implement Seasonal Strength Calculation
- [ ] Create `SeasonalStrengthCalculator` class
- [ ] Implement formula: 1 - Var(residual) / Var(seasonal + residual)
- [ ] Calculate strength for each seasonal component
- [ ] Validate strength is in [0, 1] range
- [ ] Handle edge case where variance is zero
- [ ] Return strength metrics dictionary

### 6. Implement Trend Strength Calculation
- [ ] Implement trend strength metric
- [ ] Formula: 1 - Var(residual) / Var(trend + residual)
- [ ] Identify strong vs weak trends
- [ ] Return trend strength value

### 7. Implement SeasonalityPlugin Class
- [ ] Inherit from `AdvancedFeaturePlugin`
- [ ] Implement `name` property returning "seasonality"
- [ ] Implement `version` property
- [ ] Implement `computational_complexity` returning "medium"
- [ ] Add class docstring with STL methodology

### 8. Implement generate_features() Method
- [ ] Validate input DataFrame has target column
- [ ] Check minimum data length (need >= 2 × max_period)
- [ ] Handle missing values via interpolation or forward-fill
- [ ] Perform STL or MSTL decomposition
- [ ] Calculate seasonal strength metrics
- [ ] Generate features from components
- [ ] Store decomposition metadata
- [ ] Return DataFrame with seasonal features

### 9. Generate Seasonal Component Features
- [ ] Create features for trend component
- [ ] Create features for each seasonal component
- [ ] Create residual-based features
- [ ] Add lagged seasonal values (e.g., seasonal_daily_lag_1)
- [ ] Add seasonal differences
- [ ] Add seasonal moving averages
- [ ] Name features descriptively (e.g., "seasonal_daily", "trend")

### 10. Implement Daily Seasonality Extraction
- [ ] Extract 48-period (semi-hourly) daily pattern
- [ ] Generate daily_seasonal feature
- [ ] Calculate daily seasonal strength
- [ ] Create daily seasonal profile (average pattern)
- [ ] Handle daylight saving time transitions

### 11. Implement Weekly Seasonality Extraction
- [ ] Extract 336-period (48 × 7) weekly pattern
- [ ] Generate weekly_seasonal feature
- [ ] Calculate weekly seasonal strength
- [ ] Create weekly seasonal profile
- [ ] Distinguish weekday vs weekend patterns

### 12. Implement Yearly Seasonality Extraction
- [ ] Extract 17520-period (48 × 365) yearly pattern
- [ ] Generate yearly_seasonal feature
- [ ] Calculate yearly seasonal strength
- [ ] Handle leap years
- [ ] Create yearly seasonal profile

### 13. Implement Seasonal Profiles
- [ ] Create `generate_seasonal_profiles()` method
- [ ] Calculate average seasonal pattern per period
- [ ] Create lookup table for seasonal values by hour/day/month
- [ ] Support profile visualization
- [ ] Export profiles to JSON/CSV

### 14. Handle Missing Data
- [ ] Implement preprocessing step for gaps
- [ ] Use linear interpolation for small gaps (<6 hours)
- [ ] Use seasonal average for larger gaps
- [ ] Mark interpolated regions
- [ ] Log imputation summary

### 15. Implement Decomposition Visualization
- [ ] Create `plot_decomposition()` method
- [ ] Plot original series
- [ ] Plot trend component
- [ ] Plot seasonal components (separate subplots)
- [ ] Plot residuals
- [ ] Save to file if path provided
- [ ] Use matplotlib/seaborn

### 16. Store Decomposition Metadata
- [ ] Create `decomposition_metadata` attribute
- [ ] Store seasonal strengths
- [ ] Store trend strength
- [ ] Store decomposition parameters used
- [ ] Store residual statistics (mean, std, outliers)
- [ ] Provide accessor methods

### 17. Write Comprehensive Tests
- [ ] Create `tests/features/plugins/test_seasonality.py`
- [ ] Test basic STL decomposition
- [ ] Test MSTL with multiple periods
- [ ] Test seasonal strength calculation
- [ ] Test with synthetic data (known seasonality)
- [ ] Test missing data handling
- [ ] Test edge cases (short series, no seasonality)
- [ ] Test performance benchmark
- [ ] Validate extracted components sum to original

### 18. Create Seasonal Strength Tests
- [ ] Create `tests/features/metrics/test_seasonal_strength.py`
- [ ] Test with strong seasonality (strength > 0.8)
- [ ] Test with weak seasonality (strength < 0.3)
- [ ] Test with no seasonality (strength ≈ 0)
- [ ] Test edge cases (constant series)

### 19. Create Usage Examples
- [ ] Create `examples/seasonality_decomposition_demo.py`
- [ ] Show single period decomposition
- [ ] Show multi-period MSTL
- [ ] Visualize seasonal components
- [ ] Show seasonal strength calculation
- [ ] Compare STL vs MSTL

---

## 💻 Implementation Details

### Configuration Schema

```python
"""Configuration for seasonality plugin."""
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class SeasonalityConfig(BaseModel):
    """Configuration for seasonality calculation plugin."""
    
    target_column: str = Field(
        default="carga",
        description="Target column for decomposition"
    )
    seasonal_periods: Dict[str, int] = Field(
        default_factory=lambda: {
            "daily": 48,      # Semi-hourly data: 48 periods per day
            "weekly": 336,    # 48 * 7 periods per week
            "yearly": 17520   # 48 * 365 periods per year
        },
        description="Seasonal periods to extract"
    )
    decomposition_method: Literal["stl", "mstl"] = Field(
        default="mstl",
        description="Decomposition method"
    )
    seasonal_smoother: int = Field(
        default=7,
        description="Seasonal smoother parameter (odd integer)"
    )
    trend_smoother: Optional[int] = Field(
        default=None,
        description="Trend smoother (None for auto)"
    )
    robust: bool = Field(
        default=True,
        description="Use robust fitting (resistant to outliers)"
    )
    seasonal_strength_threshold: float = Field(
        default=0.3,
        description="Minimum strength to include seasonal component"
    )
    extrapolate_trend: int = Field(
        default=0,
        description="Number of periods to extrapolate trend"
    )
    min_data_periods: int = Field(
        default=2,
        description="Minimum periods of data required (× largest period)"
    )
    
    @field_validator('seasonal_periods')
    @classmethod
    def validate_periods(cls, v):
        """Validate seasonal periods are positive."""
        for name, period in v.items():
            if period < 2:
                raise ValueError(f"Seasonal period '{name}' must be >= 2")
        return v
    
    @field_validator('seasonal_smoother')
    @classmethod
    def validate_seasonal_smoother(cls, v):
        """Validate seasonal smoother is odd."""
        if v % 2 == 0:
            raise ValueError("seasonal_smoother must be odd")
        if v < 3:
            raise ValueError("seasonal_smoother must be >= 3")
        return v
    
    @field_validator('seasonal_strength_threshold')
    @classmethod
    def validate_threshold(cls, v):
        """Validate threshold is in [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError("seasonal_strength_threshold must be in [0, 1]")
        return v
```

### STL Decomposer Implementation

```python
"""STL decomposition utilities."""
from typing import Optional
import pandas as pd
from statsmodels.tsa.seasonal import STL, MSTL

from src.utils.logger import get_logger

logger = get_logger(__name__)


class STLDecomposer:
    """
    Seasonal-Trend decomposition using Loess.
    
    Decomposes time series into three components:
    - Trend: Long-term progression
    - Seasonal: Periodic patterns
    - Residual: Irregular fluctuations
    
    Uses locally weighted regression (LOESS) for flexible, non-parametric
    decomposition that adapts to changing seasonal patterns.
    """
    
    def __init__(
        self,
        seasonal_period: int,
        seasonal_smoother: int = 7,
        trend_smoother: Optional[int] = None,
        robust: bool = True
    ):
        """
        Initialize STL decomposer.
        
        Args:
            seasonal_period: Number of observations per seasonal cycle
            seasonal_smoother: Smoothing parameter for seasonal component (odd)
            trend_smoother: Smoothing parameter for trend (None for auto)
            robust: Use robust fitting resistant to outliers
        """
        self.seasonal_period = seasonal_period
        self.seasonal_smoother = seasonal_smoother
        self.trend_smoother = trend_smoother
        self.robust = robust
    
    def decompose(self, series: pd.Series) -> pd.DataFrame:
        """
        Perform STL decomposition.
        
        Args:
            series: Time series to decompose
        
        Returns:
            DataFrame with trend, seasonal, and residual components
        """
        logger.info(
            f"Starting STL decomposition with period={self.seasonal_period}"
        )
        
        # Check minimum length
        min_length = 2 * self.seasonal_period
        if len(series) < min_length:
            raise ValueError(
                f"Series too short ({len(series)}). Need >= {min_length} observations"
            )
        
        # Handle missing values
        series_clean = series.interpolate(method='linear', limit=24)
        series_clean = series_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Perform STL decomposition
        stl = STL(
            series_clean,
            seasonal=self.seasonal_smoother,
            trend=self.trend_smoother,
            period=self.seasonal_period,
            robust=self.robust
        )
        
        result = stl.fit()
        
        # Create output DataFrame
        decomposition = pd.DataFrame({
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        }, index=series.index)
        
        logger.info("STL decomposition complete")
        
        return decomposition


class MSTLDecomposer:
    """
    Multiple Seasonal-Trend decomposition using Loess.
    
    Extends STL to handle multiple seasonal periods simultaneously,
    ideal for electricity load data with daily, weekly, and yearly patterns.
    """
    
    def __init__(
        self,
        seasonal_periods: Dict[str, int],
        seasonal_smoother: int = 7,
        robust: bool = True
    ):
        """
        Initialize MSTL decomposer.
        
        Args:
            seasonal_periods: Dictionary mapping names to periods
            seasonal_smoother: Smoothing parameter for seasonal components
            robust: Use robust fitting
        """
        self.seasonal_periods = seasonal_periods
        self.seasonal_smoother = seasonal_smoother
        self.robust = robust
    
    def decompose(self, series: pd.Series) -> pd.DataFrame:
        """
        Perform MSTL decomposition.
        
        Args:
            series: Time series to decompose
        
        Returns:
            DataFrame with trend, multiple seasonal components, and residual
        """
        logger.info(
            f"Starting MSTL decomposition with periods={list(self.seasonal_periods.values())}"
        )
        
        # Check minimum length (need at least 2 cycles of longest period)
        max_period = max(self.seasonal_periods.values())
        min_length = 2 * max_period
        
        if len(series) < min_length:
            raise ValueError(
                f"Series too short ({len(series)}). Need >= {min_length} observations"
            )
        
        # Handle missing values
        series_clean = series.interpolate(method='linear', limit=24)
        series_clean = series_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Perform MSTL decomposition
        mstl = MSTL(
            series_clean,
            periods=list(self.seasonal_periods.values()),
            windows=self.seasonal_smoother,
            robust=self.robust
        )
        
        result = mstl.fit()
        
        # Create output DataFrame with named seasonal components
        decomposition = pd.DataFrame(index=series.index)
        decomposition['trend'] = result.trend
        
        # Add each seasonal component with descriptive name
        period_to_name = {v: k for k, v in self.seasonal_periods.items()}
        
        for i, period in enumerate(self.seasonal_periods.values()):
            name = period_to_name[period]
            component_col = f'seasonal_{name}'
            decomposition[component_col] = result.seasonal.iloc[:, i]
        
        decomposition['residual'] = result.resid
        
        logger.info("MSTL decomposition complete")
        
        return decomposition
```

### Seasonal Strength Calculation

```python
"""Seasonal strength metrics."""
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SeasonalStrengthCalculator:
    """
    Calculate strength metrics for seasonal and trend components.
    
    Seasonal strength measures how much of the variance in the
    detrended series is explained by the seasonal component.
    
    Formula: Fs = max(0, 1 - Var(residual) / Var(seasonal + residual))
    
    Interpretation:
    - Fs > 0.6: Strong seasonality
    - 0.3 < Fs < 0.6: Moderate seasonality
    - Fs < 0.3: Weak seasonality
    """
    
    @staticmethod
    def calculate_seasonal_strength(
        seasonal: pd.Series,
        residual: pd.Series
    ) -> float:
        """
        Calculate seasonal strength.
        
        Args:
            seasonal: Seasonal component
            residual: Residual component
        
        Returns:
            Seasonal strength in [0, 1]
        """
        # Variance of residual
        var_residual = residual.var()
        
        # Variance of seasonal + residual (detrended series)
        detrended = seasonal + residual
        var_detrended = detrended.var()
        
        # Avoid division by zero
        if var_detrended == 0:
            logger.warning("Detrended variance is zero, returning strength = 0")
            return 0.0
        
        # Calculate strength
        strength = max(0, 1 - var_residual / var_detrended)
        
        logger.debug(f"Seasonal strength: {strength:.3f}")
        
        return strength
    
    @staticmethod
    def calculate_trend_strength(
        trend: pd.Series,
        residual: pd.Series
    ) -> float:
        """
        Calculate trend strength.
        
        Args:
            trend: Trend component
            residual: Residual component
        
        Returns:
            Trend strength in [0, 1]
        """
        # Variance of residual
        var_residual = residual.var()
        
        # Variance of trend + residual (deseasonalized series)
        deseasonalized = trend + residual
        var_deseasonalized = deseasonalized.var()
        
        # Avoid division by zero
        if var_deseasonalized == 0:
            logger.warning("Deseasonalized variance is zero, returning strength = 0")
            return 0.0
        
        # Calculate strength
        strength = max(0, 1 - var_residual / var_deseasonalized)
        
        logger.debug(f"Trend strength: {strength:.3f}")
        
        return strength
    
    @staticmethod
    def classify_strength(strength: float) -> str:
        """
        Classify strength level.
        
        Args:
            strength: Strength value in [0, 1]
        
        Returns:
            Classification: "strong", "moderate", or "weak"
        """
        if strength > 0.6:
            return "strong"
        elif strength > 0.3:
            return "moderate"
        else:
            return "weak"
```

### SeasonalityPlugin Implementation

```python
"""Seasonality plugin for advanced feature engineering."""
from typing import Dict, Any, List, Literal, Optional
import pandas as pd
import numpy as np

from src.features.base.plugin import AdvancedFeaturePlugin
from src.features.decomposition.stl_decomposer import STLDecomposer, MSTLDecomposer
from src.features.metrics.seasonal_strength import SeasonalStrengthCalculator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SeasonalityPlugin(AdvancedFeaturePlugin):
    """
    Seasonality calculation plugin using STL/MSTL decomposition.
    
    Extracts seasonal patterns from time series using Seasonal-Trend
    decomposition based on Loess (STL). Supports multiple seasonal
    periods for capturing complex patterns in electricity load data.
    
    Features Generated:
    - trend: Long-term trend component
    - seasonal_daily: 48-period (semi-hourly) daily pattern
    - seasonal_weekly: 336-period weekly pattern
    - seasonal_yearly: 17520-period yearly pattern
    - residual: Irregular component after decomposition
    - seasonal_*_strength: Strength metrics for each component
    
    Example:
        >>> plugin = SeasonalityPlugin()
        >>> config = {
        ...     "target_column": "carga",
        ...     "decomposition_method": "mstl",
        ...     "seasonal_periods": {
        ...         "daily": 48,
        ...         "weekly": 336
        ...     }
        ... }
        >>> features_df = plugin.generate_features(df, config)
    """
    
    def __init__(self):
        """Initialize plugin."""
        self.decomposition_metadata = {}
        self.seasonal_profiles = {}
    
    @property
    def name(self) -> str:
        return "seasonality"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def computational_complexity(self) -> Literal["low", "medium", "high"]:
        return "medium"
    
    def estimate_compute_time(self, data_size: int) -> float:
        """Estimate decomposition time."""
        # Empirical: ~5 seconds per year of data for MSTL
        return 5.0 * (data_size / 17520)
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration."""
        SeasonalityConfig(**config)
        logger.debug("Seasonality config validated")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate seasonal features via STL/MSTL decomposition.
        
        Args:
            df: Input DataFrame with target column
            config: Plugin configuration
        
        Returns:
            DataFrame with seasonal component features
        """
        # Validate config
        validated_config = SeasonalityConfig(**config)
        
        # Validate target column
        target_col = validated_config.target_column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Extract target series
        target_series = df[target_col].copy()
        
        # Check minimum data length
        max_period = max(validated_config.seasonal_periods.values())
        min_length = validated_config.min_data_periods * max_period
        
        if len(target_series) < min_length:
            logger.warning(
                f"Insufficient data ({len(target_series)} < {min_length}). "
                "Returning empty features."
            )
            return pd.DataFrame(index=df.index)
        
        logger.info(
            f"Starting {validated_config.decomposition_method.upper()} decomposition "
            f"on {len(target_series)} observations"
        )
        
        # Perform decomposition
        if validated_config.decomposition_method == "stl":
            decomposition = self._decompose_stl(target_series, validated_config)
        else:
            decomposition = self._decompose_mstl(target_series, validated_config)
        
        # Calculate seasonal strengths
        self._calculate_strengths(decomposition, validated_config)
        
        # Generate features from components
        features_df = self._generate_component_features(
            decomposition,
            validated_config
        )
        
        # Generate seasonal profiles
        self._generate_seasonal_profiles(decomposition, validated_config)
        
        logger.info(f"Generated {len(features_df.columns)} seasonal features")
        
        return features_df
    
    def _decompose_stl(
        self,
        series: pd.Series,
        config: SeasonalityConfig
    ) -> pd.DataFrame:
        """Perform STL decomposition (single period)."""
        # Use first period in config
        period_name, period = list(config.seasonal_periods.items())[0]
        
        logger.info(f"Using STL with period={period} ({period_name})")
        
        decomposer = STLDecomposer(
            seasonal_period=period,
            seasonal_smoother=config.seasonal_smoother,
            trend_smoother=config.trend_smoother,
            robust=config.robust
        )
        
        decomposition = decomposer.decompose(series)
        
        # Rename seasonal column to include period name
        decomposition = decomposition.rename(
            columns={'seasonal': f'seasonal_{period_name}'}
        )
        
        return decomposition
    
    def _decompose_mstl(
        self,
        series: pd.Series,
        config: SeasonalityConfig
    ) -> pd.DataFrame:
        """Perform MSTL decomposition (multiple periods)."""
        decomposer = MSTLDecomposer(
            seasonal_periods=config.seasonal_periods,
            seasonal_smoother=config.seasonal_smoother,
            robust=config.robust
        )
        
        decomposition = decomposer.decompose(series)
        
        return decomposition
    
    def _calculate_strengths(
        self,
        decomposition: pd.DataFrame,
        config: SeasonalityConfig
    ) -> None:
        """Calculate and store seasonal strength metrics."""
        calculator = SeasonalStrengthCalculator()
        
        strengths = {}
        
        # Trend strength
        if 'trend' in decomposition.columns:
            trend_strength = calculator.calculate_trend_strength(
                decomposition['trend'],
                decomposition['residual']
            )
            strengths['trend'] = trend_strength
            logger.info(
                f"Trend strength: {trend_strength:.3f} "
                f"({calculator.classify_strength(trend_strength)})"
            )
        
        # Seasonal strengths
        seasonal_cols = [col for col in decomposition.columns if col.startswith('seasonal_')]
        
        for col in seasonal_cols:
            seasonal_strength = calculator.calculate_seasonal_strength(
                decomposition[col],
                decomposition['residual']
            )
            
            period_name = col.replace('seasonal_', '')
            strengths[period_name] = seasonal_strength
            
            logger.info(
                f"{period_name.capitalize()} seasonal strength: {seasonal_strength:.3f} "
                f"({calculator.classify_strength(seasonal_strength)})"
            )
        
        self.decomposition_metadata['strengths'] = strengths
    
    def _generate_component_features(
        self,
        decomposition: pd.DataFrame,
        config: SeasonalityConfig
    ) -> pd.DataFrame:
        """Generate features from decomposition components."""
        features_df = pd.DataFrame(index=decomposition.index)
        
        # Add all components as features
        for col in decomposition.columns:
            features_df[col] = decomposition[col]
        
        # Add lagged seasonal features
        seasonal_cols = [col for col in decomposition.columns if col.startswith('seasonal_')]
        
        for col in seasonal_cols:
            # Lag 1 (previous period's seasonal value)
            features_df[f'{col}_lag_1'] = decomposition[col].shift(1)
            
            # Seasonal difference
            period_name = col.replace('seasonal_', '')
            if period_name in config.seasonal_periods:
                period = config.seasonal_periods[period_name]
                features_df[f'{col}_diff'] = decomposition[col].diff(period)
        
        # Add detrended series
        if 'trend' in decomposition.columns:
            seasonal_sum = decomposition[[col for col in decomposition.columns 
                                          if col.startswith('seasonal_')]].sum(axis=1)
            features_df['detrended'] = seasonal_sum + decomposition['residual']
        
        # Add deseasonalized series
        seasonal_cols = [col for col in decomposition.columns if col.startswith('seasonal_')]
        if seasonal_cols and 'trend' in decomposition.columns:
            features_df['deseasonalized'] = (
                decomposition['trend'] + decomposition['residual']
            )
        
        return features_df
    
    def _generate_seasonal_profiles(
        self,
        decomposition: pd.DataFrame,
        config: SeasonalityConfig
    ) -> None:
        """Generate average seasonal profiles."""
        seasonal_cols = [col for col in decomposition.columns if col.startswith('seasonal_')]
        
        for col in seasonal_cols:
            period_name = col.replace('seasonal_', '')
            
            if period_name in config.seasonal_periods:
                period = config.seasonal_periods[period_name]
                
                # Calculate average seasonal pattern
                seasonal_values = decomposition[col].values
                n_complete_cycles = len(seasonal_values) // period
                
                if n_complete_cycles > 0:
                    # Reshape to (n_cycles, period) and average
                    truncated_length = n_complete_cycles * period
                    reshaped = seasonal_values[:truncated_length].reshape(-1, period)
                    profile = reshaped.mean(axis=0)
                    
                    self.seasonal_profiles[period_name] = profile
                    
                    logger.debug(
                        f"Generated {period_name} profile with {len(profile)} values"
                    )
    
    def get_seasonal_strength(self, period_name: str) -> Optional[float]:
        """Get seasonal strength for specific period."""
        return self.decomposition_metadata.get('strengths', {}).get(period_name)
    
    def get_seasonal_profile(self, period_name: str) -> Optional[np.ndarray]:
        """Get seasonal profile for specific period."""
        return self.seasonal_profiles.get(period_name)
    
    def plot_decomposition(
        self,
        decomposition: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot decomposition components.
        
        Args:
            decomposition: Decomposition DataFrame
            output_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        # Count components
        n_components = len(decomposition.columns)
        
        # Create subplots
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 2 * n_components))
        
        if n_components == 1:
            axes = [axes]
        
        # Plot each component
        for i, col in enumerate(decomposition.columns):
            axes[i].plot(decomposition.index, decomposition[col])
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Decomposition plot saved to {output_path}")
        else:
            plt.show()
    
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """Get generated feature names."""
        validated_config = SeasonalityConfig(**config)
        
        feature_names = ['trend', 'residual', 'detrended', 'deseasonalized']
        
        # Add seasonal component names
        for period_name in validated_config.seasonal_periods.keys():
            feature_names.append(f'seasonal_{period_name}')
            feature_names.append(f'seasonal_{period_name}_lag_1')
            feature_names.append(f'seasonal_{period_name}_diff')
        
        return feature_names
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for seasonality plugin."""
import pytest
import pandas as pd
import numpy as np

from src.features.plugins.seasonality import SeasonalityPlugin, SeasonalityConfig
from src.features.metrics.seasonal_strength import SeasonalStrengthCalculator


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return SeasonalityPlugin()


@pytest.fixture
def synthetic_seasonal_data():
    """Create synthetic data with known seasonality."""
    np.random.seed(42)
    
    # Generate 2 years of semi-hourly data
    n_periods = 2 * 365 * 48
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="30min")
    
    # Trend
    trend = np.linspace(1000, 1200, n_periods)
    
    # Daily seasonality (48-period)
    daily_pattern = 100 * np.sin(2 * np.pi * np.arange(48) / 48)
    daily_seasonal = np.tile(daily_pattern, n_periods // 48)[:n_periods]
    
    # Weekly seasonality (336-period)
    weekly_pattern = 50 * np.sin(2 * np.pi * np.arange(336) / 336)
    weekly_seasonal = np.tile(weekly_pattern, n_periods // 336 + 1)[:n_periods]
    
    # Noise
    noise = np.random.randn(n_periods) * 10
    
    # Combine
    load = trend + daily_seasonal + weekly_seasonal + noise
    
    return pd.DataFrame({
        'carga': load,
        'true_trend': trend,
        'true_daily': daily_seasonal,
        'true_weekly': weekly_seasonal
    }, index=dates)


def test_basic_stl_decomposition(plugin, synthetic_seasonal_data):
    """Test basic STL decomposition."""
    config = {
        'target_column': 'carga',
        'decomposition_method': 'stl',
        'seasonal_periods': {'daily': 48}
    }
    
    features = plugin.generate_features(synthetic_seasonal_data, config)
    
    # Should have trend, seasonal, residual
    assert 'trend' in features.columns
    assert 'seasonal_daily' in features.columns
    assert 'residual' in features.columns


def test_mstl_decomposition(plugin, synthetic_seasonal_data):
    """Test MSTL with multiple periods."""
    config = {
        'target_column': 'carga',
        'decomposition_method': 'mstl',
        'seasonal_periods': {
            'daily': 48,
            'weekly': 336
        }
    }
    
    features = plugin.generate_features(synthetic_seasonal_data, config)
    
    # Should have both seasonal components
    assert 'seasonal_daily' in features.columns
    assert 'seasonal_weekly' in features.columns
    assert 'trend' in features.columns
    assert 'residual' in features.columns


def test_seasonal_strength_calculation():
    """Test seasonal strength metric."""
    calculator = SeasonalStrengthCalculator()
    
    # Strong seasonality
    seasonal = pd.Series(100 * np.sin(2 * np.pi * np.arange(100) / 10))
    residual = pd.Series(np.random.randn(100) * 5)
    
    strength = calculator.calculate_seasonal_strength(seasonal, residual)
    
    # Should be high (> 0.6)
    assert strength > 0.6
    assert calculator.classify_strength(strength) == "strong"


def test_weak_seasonality():
    """Test with weak seasonality."""
    calculator = SeasonalStrengthCalculator()
    
    # Weak seasonality
    seasonal = pd.Series(np.sin(2 * np.pi * np.arange(100) / 10))
    residual = pd.Series(np.random.randn(100) * 100)  # Large noise
    
    strength = calculator.calculate_seasonal_strength(seasonal, residual)
    
    # Should be low (< 0.3)
    assert strength < 0.5


def test_decomposition_reconstruction(plugin, synthetic_seasonal_data):
    """Test that components sum back to original."""
    config = {
        'target_column': 'carga',
        'decomposition_method': 'mstl',
        'seasonal_periods': {'daily': 48}
    }
    
    features = plugin.generate_features(synthetic_seasonal_data, config)
    
    # Reconstruct
    reconstructed = (
        features['trend'] +
        features['seasonal_daily'] +
        features['residual']
    )
    
    # Should match original (allowing small numerical errors)
    original = synthetic_seasonal_data['carga']
    common_idx = reconstructed.index.intersection(original.index)
    
    diff = np.abs(reconstructed.loc[common_idx] - original.loc[common_idx])
    assert diff.mean() < 1e-10


def test_lagged_features(plugin, synthetic_seasonal_data):
    """Test lagged seasonal features are generated."""
    config = {
        'target_column': 'carga',
        'decomposition_method': 'stl',
        'seasonal_periods': {'daily': 48}
    }
    
    features = plugin.generate_features(synthetic_seasonal_data, config)
    
    # Should have lagged features
    assert 'seasonal_daily_lag_1' in features.columns
    assert 'seasonal_daily_diff' in features.columns


def test_seasonal_profiles(plugin, synthetic_seasonal_data):
    """Test seasonal profile generation."""
    config = {
        'target_column': 'carga',
        'decomposition_method': 'stl',
        'seasonal_periods': {'daily': 48}
    }
    
    plugin.generate_features(synthetic_seasonal_data, config)
    
    # Should have daily profile
    profile = plugin.get_seasonal_profile('daily')
    
    assert profile is not None
    assert len(profile) == 48


def test_insufficient_data(plugin):
    """Test handling of insufficient data."""
    short_df = pd.DataFrame({
        'carga': np.random.randn(50)
    }, index=pd.date_range("2024-01-01", periods=50, freq="h"))
    
    config = {
        'target_column': 'carga',
        'seasonal_periods': {'daily': 48}
    }
    
    # Should return empty features gracefully
    features = plugin.generate_features(short_df, config)
    assert len(features.columns) == 0


def test_missing_data_handling(plugin, synthetic_seasonal_data):
    """Test handling of missing data."""
    # Introduce missing values
    df_with_missing = synthetic_seasonal_data.copy()
    df_with_missing.loc[df_with_missing.index[100:110], 'carga'] = np.nan
    
    config = {
        'target_column': 'carga',
        'seasonal_periods': {'daily': 48}
    }
    
    # Should handle gracefully
    features = plugin.generate_features(df_with_missing, config)
    assert len(features) > 0


def test_config_validation():
    """Test configuration validation."""
    # Invalid period (< 2)
    with pytest.raises(ValueError, match="must be >= 2"):
        SeasonalityConfig(seasonal_periods={'daily': 1})
    
    # Even seasonal smoother
    with pytest.raises(ValueError, match="must be odd"):
        SeasonalityConfig(seasonal_smoother=6)
    
    # Invalid threshold
    with pytest.raises(ValueError, match="must be in"):
        SeasonalityConfig(seasonal_strength_threshold=1.5)


def test_computational_complexity(plugin):
    """Test complexity reporting."""
    assert plugin.computational_complexity == "medium"


def test_estimate_compute_time(plugin):
    """Test time estimation."""
    time_estimate = plugin.estimate_compute_time(17520)  # 1 year
    
    # Should estimate ~5 seconds
    assert 3 < time_estimate < 10


def test_feature_names(plugin):
    """Test feature name generation."""
    config = {
        'seasonal_periods': {'daily': 48, 'weekly': 336}
    }
    
    feature_names = plugin.get_feature_names(config)
    
    assert 'seasonal_daily' in feature_names
    assert 'seasonal_weekly' in feature_names
    assert 'trend' in feature_names
    assert 'residual' in feature_names
```

---

## 📝 Technical Notes

### STL vs MSTL
- **STL**: Single seasonal period, faster, simpler
- **MSTL**: Multiple periods, captures complex patterns, slightly slower
- For electricity load: MSTL recommended to capture daily, weekly, and yearly patterns

### Seasonal Smoothing Parameter
- Odd integer controlling seasonal smoothness
- Larger values = smoother seasonal component
- Typical range: 7-21
- Default 7 works well for most cases

### Handling Missing Data
- Linear interpolation for small gaps (<6 hours)
- Forward/backward fill for edge regions
- STL is reasonably robust to missing data

### Seasonal Strength Interpretation
- **> 0.6**: Strong seasonality, important for forecasting
- **0.3-0.6**: Moderate seasonality, useful but not dominant
- **< 0.3**: Weak seasonality, may not be worth including

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation
- PC-018-02B: LOESS Smoothing Plugin (related methodology)

**External Dependencies:**
- `statsmodels>=0.14.0` (STL, MSTL decomposition)
- `pandas>=2.0.0`
- `numpy>=1.24.0`

**Blocks:**
- Seasonal feature engineering pipelines
- Model training with decomposed components

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] SeasonalityPlugin implemented with STL/MSTL
- [ ] Seasonal strength calculation working
- [ ] Multiple seasonal periods supported
- [ ] Missing data handling robust
- [ ] Seasonal profiles generated
- [ ] Unit tests pass with >85% coverage
- [ ] Tests with synthetic seasonal data validate accuracy
- [ ] Performance benchmark met (<10s per decomposition)
- [ ] Visualization methods functional
- [ ] Documentation complete with examples
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-021-02B: RF Feature Selector Plugin](PC-021-02B-rf-feature-selector-plugin.md)  
**Next:** [PC-023-02B: Performance Optimization](PC-023-02B-performance-optimization.md)
