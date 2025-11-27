"""Seasonality plugin for advanced feature engineering.

This module provides the SeasonalityPlugin for extracting seasonal patterns
from time series using STL (Seasonal-Trend decomposition using Loess) and
MSTL (Multiple Seasonal-Trend decomposition using Loess).

The plugin decomposes load data into:
- Trend: Long-term progression patterns
- Seasonal components: Daily, weekly, and yearly patterns
- Residuals: Irregular fluctuations

These components serve as powerful features for electricity load forecasting,
capturing complex temporal patterns that simple features cannot represent.

Example:
    ```python
    from src.features.plugins.seasonality import (
        SeasonalityPlugin,
        SeasonalityConfig,
    )
    import pandas as pd

    # Create plugin
    plugin = SeasonalityPlugin()

    # Configure for MSTL with multiple periods
    config = SeasonalityConfig(
        target_column="carga",
        decomposition_method="mstl",
        seasonal_periods={
            "daily": 48,
            "weekly": 336
        },
        robust=True
    )

    # Generate features
    df = pd.DataFrame(...)  # Load data
    features = plugin.generate_features(df, config.model_dump())
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features.base.advanced_plugin import AdvancedFeaturePlugin
from src.features.decomposition.stl_decomposer import MSTLDecomposer, STLDecomposer
from src.features.metrics.seasonal_strength import SeasonalStrengthCalculator
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


class SeasonalityConfig(BaseModel):
    """Configuration for seasonality calculation plugin.

    This configuration model defines all parameters for STL/MSTL decomposition
    including seasonal periods, smoothing parameters, and feature generation
    options.

    Attributes:
        target_column: Target column name for decomposition.
        seasonal_periods: Dictionary of seasonal periods to extract.
        decomposition_method: Method to use ("stl" or "mstl").
        seasonal_smoother: Seasonal smoother parameter (odd integer).
        trend_smoother: Trend smoother parameter (None for auto).
        robust: Use robust fitting (resistant to outliers).
        seasonal_strength_threshold: Minimum strength to include component.
        extrapolate_trend: Number of periods to extrapolate trend.
        min_data_periods: Minimum periods of data required.

    Example:
        ```python
        config = SeasonalityConfig(
            target_column="carga",
            decomposition_method="mstl",
            seasonal_periods={
                "daily": 48,
                "weekly": 336,
                "yearly": 17520
            },
            seasonal_smoother=7,
            robust=True
        )
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    target_column: str = Field(
        default="carga",
        description="Target column for decomposition",
    )
    seasonal_periods: dict[str, int] = Field(
        default_factory=lambda: {
            "daily": 48,  # Semi-hourly data: 48 periods per day
            "weekly": 336,  # 48 * 7 periods per week
            "yearly": 17520,  # 48 * 365 periods per year
        },
        description="Seasonal periods to extract",
    )
    decomposition_method: Literal["stl", "mstl"] = Field(
        default="mstl",
        description="Decomposition method",
    )
    seasonal_smoother: int = Field(
        default=7,
        description="Seasonal smoother parameter (odd integer)",
    )
    trend_smoother: int | None = Field(
        default=None,
        description="Trend smoother (None for auto)",
    )
    robust: bool = Field(
        default=True,
        description="Use robust fitting (resistant to outliers)",
    )
    seasonal_strength_threshold: float = Field(
        default=0.3,
        description="Minimum strength to include seasonal component",
    )
    extrapolate_trend: int = Field(
        default=0,
        description="Number of periods to extrapolate trend",
    )
    min_data_periods: int = Field(
        default=2,
        description="Minimum periods of data required (x largest period)",
    )

    @field_validator("seasonal_periods")
    @classmethod
    def validate_periods(cls, v: dict[str, int]) -> dict[str, int]:
        """Validate seasonal periods are positive."""
        for name, period in v.items():
            if period < 2:
                msg = f"Seasonal period '{name}' must be >= 2"
                raise ValueError(msg)
        return v

    @field_validator("seasonal_smoother")
    @classmethod
    def validate_seasonal_smoother(cls, v: int) -> int:
        """Validate seasonal smoother is odd."""
        if v % 2 == 0:
            msg = "seasonal_smoother must be odd"
            raise ValueError(msg)
        if v < 3:
            msg = "seasonal_smoother must be >= 3"
            raise ValueError(msg)
        return v

    @field_validator("seasonal_strength_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold is in [0, 1]."""
        if not 0 <= v <= 1:
            msg = "seasonal_strength_threshold must be in [0, 1]"
            raise ValueError(msg)
        return v


class SeasonalityPlugin(AdvancedFeaturePlugin):
    """Seasonality calculation plugin using STL/MSTL decomposition.

    Extracts seasonal patterns from time series using Seasonal-Trend
    decomposition based on Loess (STL). Supports multiple seasonal
    periods for capturing complex patterns in electricity load data.

    Features Generated:
    - trend: Long-term trend component
    - seasonal_daily: 48-period (semi-hourly) daily pattern
    - seasonal_weekly: 336-period weekly pattern
    - seasonal_yearly: 17520-period yearly pattern
    - residual: Irregular component after decomposition
    - seasonal_*_lag_1: Lagged seasonal values
    - seasonal_*_diff: Seasonal differences
    - detrended: Series with trend removed
    - deseasonalized: Series with seasonality removed

    The plugin also calculates and stores:
    - Seasonal strength metrics for each component
    - Trend strength metric
    - Seasonal profiles (average patterns)

    Example:
        ```python
        plugin = SeasonalityPlugin()
        config = {
            "target_column": "carga",
            "decomposition_method": "mstl",
            "seasonal_periods": {
                "daily": 48,
                "weekly": 336
            }
        }
        features_df = plugin.generate_features(df, config)

        # Access metadata
        daily_strength = plugin.get_seasonal_strength("daily")
        daily_profile = plugin.get_seasonal_profile("daily")
        ```
    """

    def __init__(self):
        """Initialize plugin."""
        self.decomposition_metadata: dict[str, Any] = {}
        self.seasonal_profiles: dict[str, NDArray[np.floating]] = {}

    @property
    def name(self) -> str:
        """Return plugin name."""
        return "seasonality"

    @property
    def version(self) -> str:
        """Return plugin version."""
        return "1.0.0"

    @property
    def computational_complexity(self) -> Literal["low", "medium", "high"]:
        """Return computational complexity level."""
        return "medium"

    def estimate_compute_time(self, data_size: int) -> float:
        """Estimate decomposition time.

        Args:
            data_size: Number of data points.

        Returns:
            Estimated computation time in seconds.

        Note:
            Based on empirical measurements: ~5 seconds per year of
            semi-hourly data for MSTL decomposition.
        """
        # Empirical: ~5 seconds per year of data for MSTL
        return 5.0 * (data_size / 17520)

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate configuration.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            True if configuration is valid.

        Raises:
            ValidationError: If configuration is invalid.
        """
        SeasonalityConfig(**config)
        logger.debug("Seasonality config validated")
        return True

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate seasonal features via STL/MSTL decomposition.

        Args:
            df: Input DataFrame with target column.
            config: Plugin configuration.

        Returns:
            DataFrame with seasonal component features.

        Raises:
            ValueError: If target column not found or data is insufficient.
        """
        # Validate config
        validated_config = SeasonalityConfig(**config)

        # Validate target column
        target_col = validated_config.target_column
        if target_col not in df.columns:
            msg = f"Target column '{target_col}' not found"
            raise ValueError(msg)

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
            validated_config,
        )

        # Generate seasonal profiles
        self._generate_seasonal_profiles(decomposition, validated_config)

        logger.info(f"Generated {len(features_df.columns)} seasonal features")

        return features_df

    def _decompose_stl(
        self,
        series: pd.Series,
        config: SeasonalityConfig,
    ) -> pd.DataFrame:
        """Perform STL decomposition (single period).

        Args:
            series: Time series to decompose.
            config: Configuration object.

        Returns:
            DataFrame with decomposition components.
        """
        # Use first period in config
        period_name, period = next(iter(config.seasonal_periods.items()))

        logger.info(f"Using STL with period={period} ({period_name})")

        decomposer = STLDecomposer(
            seasonal_period=period,
            seasonal_smoother=config.seasonal_smoother,
            trend_smoother=config.trend_smoother,
            robust=config.robust,
        )

        decomposition = decomposer.decompose(series)

        # Rename seasonal column to include period name
        return decomposition.rename(
            columns={"seasonal": f"seasonal_{period_name}"}
        )

    def _decompose_mstl(
        self,
        series: pd.Series,
        config: SeasonalityConfig,
    ) -> pd.DataFrame:
        """Perform MSTL decomposition (multiple periods).

        Args:
            series: Time series to decompose.
            config: Configuration object.

        Returns:
            DataFrame with decomposition components.
        """
        decomposer = MSTLDecomposer(
            seasonal_periods=config.seasonal_periods,
            seasonal_smoother=config.seasonal_smoother,
            robust=config.robust,
        )

        return decomposer.decompose(series)

    def _calculate_strengths(
        self,
        decomposition: pd.DataFrame,
        config: SeasonalityConfig,  # noqa: ARG002
    ) -> None:
        """Calculate and store seasonal strength metrics.

        Args:
            decomposition: Decomposition DataFrame.
            config: Configuration object.
        """
        calculator = SeasonalStrengthCalculator()

        strengths: dict[str, float] = {}

        # Trend strength
        if "trend" in decomposition.columns:
            trend_strength = calculator.calculate_trend_strength(
                decomposition["trend"],
                decomposition["residual"],
            )
            strengths["trend"] = trend_strength
            logger.info(
                f"Trend strength: {trend_strength:.3f} "
                f"({calculator.classify_strength(trend_strength)})"
            )

        # Seasonal strengths
        seasonal_cols = [
            col for col in decomposition.columns if col.startswith("seasonal_")
        ]

        for col in seasonal_cols:
            seasonal_strength = calculator.calculate_seasonal_strength(
                decomposition[col],
                decomposition["residual"],
            )

            period_name = col.replace("seasonal_", "")
            strengths[period_name] = seasonal_strength

            logger.info(
                f"{period_name.capitalize()} seasonal strength: {seasonal_strength:.3f} "
                f"({calculator.classify_strength(seasonal_strength)})"
            )

        self.decomposition_metadata["strengths"] = strengths

    def _generate_component_features(
        self,
        decomposition: pd.DataFrame,
        config: SeasonalityConfig,
    ) -> pd.DataFrame:
        """Generate features from decomposition components.

        Args:
            decomposition: Decomposition DataFrame.
            config: Configuration object.

        Returns:
            DataFrame with component-based features.
        """
        features_df = pd.DataFrame(index=decomposition.index)

        # Add all components as features
        for col in decomposition.columns:
            features_df[col] = decomposition[col]

        # Add lagged seasonal features
        seasonal_cols = [
            col for col in decomposition.columns if col.startswith("seasonal_")
        ]

        for col in seasonal_cols:
            # Lag 1 (previous period's seasonal value)
            features_df[f"{col}_lag_1"] = decomposition[col].shift(1)

            # Seasonal difference
            period_name = col.replace("seasonal_", "")
            if period_name in config.seasonal_periods:
                period = config.seasonal_periods[period_name]
                features_df[f"{col}_diff"] = decomposition[col].diff(period)

        # Add detrended series
        if "trend" in decomposition.columns:
            seasonal_sum = decomposition[
                [col for col in decomposition.columns if col.startswith("seasonal_")]
            ].sum(axis=1)
            features_df["detrended"] = seasonal_sum + decomposition["residual"]

        # Add deseasonalized series
        seasonal_cols = [
            col for col in decomposition.columns if col.startswith("seasonal_")
        ]
        if seasonal_cols and "trend" in decomposition.columns:
            features_df["deseasonalized"] = (
                decomposition["trend"] + decomposition["residual"]
            )

        return features_df

    def _generate_seasonal_profiles(
        self,
        decomposition: pd.DataFrame,
        config: SeasonalityConfig,
    ) -> None:
        """Generate average seasonal profiles.

        Args:
            decomposition: Decomposition DataFrame.
            config: Configuration object.
        """
        seasonal_cols = [
            col for col in decomposition.columns if col.startswith("seasonal_")
        ]

        for col in seasonal_cols:
            period_name = col.replace("seasonal_", "")

            if period_name in config.seasonal_periods:
                period = config.seasonal_periods[period_name]

                # Calculate average seasonal pattern
                seasonal_values = decomposition[col].to_numpy()
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

    def get_seasonal_strength(self, period_name: str) -> float | None:
        """Get seasonal strength for specific period.

        Args:
            period_name: Name of the seasonal period (e.g., "daily", "weekly").

        Returns:
            Seasonal strength value in [0, 1], or None if not available.
        """
        return self.decomposition_metadata.get("strengths", {}).get(period_name)

    def get_seasonal_profile(self, period_name: str) -> NDArray[np.floating] | None:
        """Get seasonal profile for specific period.

        Args:
            period_name: Name of the seasonal period (e.g., "daily", "weekly").

        Returns:
            Array containing the average seasonal pattern, or None if not available.
        """
        return self.seasonal_profiles.get(period_name)

    def plot_decomposition(
        self,
        decomposition: pd.DataFrame,
        output_path: str | None = None,
    ) -> None:
        """Plot decomposition components.

        Args:
            decomposition: Decomposition DataFrame.
            output_path: Optional path to save plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return

        # Count components
        n_components = len(decomposition.columns)

        # Create subplots
        _fig, axes = plt.subplots(n_components, 1, figsize=(12, 2 * n_components))

        if n_components == 1:
            axes = [axes]

        # Plot each component
        for i, col in enumerate(decomposition.columns):
            axes[i].plot(decomposition.index, decomposition[col])
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Decomposition plot saved to {output_path}")
        else:
            plt.show()

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """Get generated feature names.

        Args:
            config: Plugin configuration.

        Returns:
            List of feature names that will be generated.
        """
        validated_config = SeasonalityConfig(**config)

        feature_names = ["trend", "residual", "detrended", "deseasonalized"]

        # Add seasonal component names
        for period_name in validated_config.seasonal_periods:
            feature_names.append(f"seasonal_{period_name}")
            feature_names.append(f"seasonal_{period_name}_lag_1")
            feature_names.append(f"seasonal_{period_name}_diff")

        return feature_names
