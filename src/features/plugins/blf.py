"""BLF (Baseline Load Forecast) strategy feature plugin.

This module provides the BLFStrategyPlugin for generating features based on
baseline load forecast patterns and correction strategies used in Brazilian
electric load forecasting.

BLF features help models learn from historical baseline patterns and
corrections, which is essential for accurate D+1 to D+8 forecasting.

Example:
    ```python
    from src.features.plugins.blf import (
        BLFStrategyPlugin,
        BLFStrategyConfig,
    )
    import pandas as pd

    # Create plugin and config
    plugin = BLFStrategyPlugin()
    config = BLFStrategyConfig(
        load_column="load",
        reference_days=7,
        include_ratios=True,
        include_corrections=True,
    )

    # Generate BLF features
    df = pd.DataFrame(
        {"load": [100, 105, 95, 110, 102, 98, 115]},
        index=pd.date_range("2024-01-01", periods=7, freq="h"),
    )
    result = plugin.generate_features(df, config.model_dump())
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

# Default BLF parameters
DEFAULT_REFERENCE_DAYS = 7  # One week of reference
DEFAULT_CORRECTION_WINDOWS = [24, 48, 168]  # 1 day, 2 days, 1 week


class BLFStrategyConfig(BaseModel):
    """Configuration model for BLFStrategyPlugin.

    This model defines configuration options for BLF feature generation,
    including the load column, reference period, and which feature types
    to include.

    Attributes:
        load_column: Name of the column containing load values.
        reference_days: Number of reference days for baseline calculation.
        correction_windows: List of window sizes (hours) for corrections.
        include_baseline: Whether to include baseline estimate features.
        include_ratios: Whether to include ratio-to-baseline features.
        include_corrections: Whether to include correction features.
        include_profile: Whether to include hourly profile features.
        suffix_baseline: Suffix for baseline feature names.
        suffix_ratio: Suffix for ratio feature names.
        suffix_correction: Suffix for correction feature names.

    Example:
        ```python
        config = BLFStrategyConfig(
            load_column="load",
            reference_days=7,
            include_ratios=True,
            include_corrections=True,
        )
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    load_column: str = Field(
        default="load",
        description="Name of the column containing load values",
    )
    reference_days: int = Field(
        default=DEFAULT_REFERENCE_DAYS,
        ge=1,
        le=30,
        description="Number of reference days for baseline calculation",
    )
    correction_windows: list[int] = Field(
        default_factory=lambda: list(DEFAULT_CORRECTION_WINDOWS),
        description="Window sizes (hours) for correction calculations",
    )
    include_baseline: bool = Field(
        default=True,
        description="Include baseline estimate features",
    )
    include_ratios: bool = Field(
        default=True,
        description="Include ratio-to-baseline features",
    )
    include_corrections: bool = Field(
        default=True,
        description="Include correction features (actual - baseline)",
    )
    include_profile: bool = Field(
        default=True,
        description="Include hourly profile features (normalized)",
    )
    suffix_baseline: str = Field(
        default="_baseline",
        description="Suffix for baseline feature column names",
    )
    suffix_ratio: str = Field(
        default="_ratio",
        description="Suffix for ratio feature column names",
    )
    suffix_correction: str = Field(
        default="_correction",
        description="Suffix for correction feature column names",
    )

    @field_validator("load_column")
    @classmethod
    def validate_load_column(cls, v: str) -> str:
        """Validate that load column name is non-empty."""
        if not v or not v.strip():
            msg = "load_column cannot be empty"
            raise ValueError(msg)
        return v.strip()

    @field_validator("correction_windows")
    @classmethod
    def validate_correction_windows(cls, v: list[int]) -> list[int]:
        """Validate correction window sizes."""
        for window in v:
            if window < 1:
                msg = "Correction window sizes must be >= 1"
                raise ValueError(msg)
        return v

    @field_validator("suffix_baseline", "suffix_ratio", "suffix_correction")
    @classmethod
    def validate_suffix(cls, v: str) -> str:
        """Validate that suffixes are non-empty."""
        if not v or not v.strip():
            msg = "Suffixes cannot be empty strings"
            raise ValueError(msg)
        return v


class BLFStrategyPlugin(BaseFeaturePlugin):
    """Feature plugin for BLF (Baseline Load Forecast) strategy features.

    BLF features are based on the concept of baseline load patterns in
    Brazilian electric load forecasting. The plugin generates:

    - Baseline estimates: Average load for same hour/day-of-week pattern
    - Ratio features: Current load relative to baseline
    - Correction features: Deviation from baseline
    - Profile features: Normalized hourly load patterns

    These features help models understand:
    - Typical load patterns by hour and day of week
    - Recent deviations from expected patterns
    - Correction factors for baseline adjustments

    Example:
        ```python
        plugin = BLFStrategyPlugin()
        config = {
            "load_column": "load",
            "reference_days": 7,
            "include_baseline": True,
            "include_ratios": True,
        }
        result = plugin.generate_features(df, config)
        ```

    Attributes:
        name: Plugin identifier ("blf_strategy").
        version: Plugin version ("1.0.0").
    """

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "blf_strategy"

    @property
    def version(self) -> str:
        """Return the plugin version."""
        return "1.0.0"

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate BLF strategy features.

        Args:
            df: Input DataFrame with load column and datetime index.
            config: Plugin configuration dictionary.

        Returns:
            DataFrame with BLF features added.

        Raises:
            ValueError: If load column is missing or invalid index.
        """
        parsed_config = BLFStrategyConfig(**config)
        load_col = parsed_config.load_column

        # Validate load column exists
        if load_col not in df.columns:
            msg = f"Load column '{load_col}' not found in DataFrame"
            raise ValueError(msg)

        # Validate index is datetime-like
        if not isinstance(df.index, pd.DatetimeIndex):
            msg = "DataFrame must have a DatetimeIndex"
            raise ValueError(msg)

        result_df = df.copy()
        load_values = df[load_col].to_numpy(dtype=np.float64)

        # Calculate baseline features
        if parsed_config.include_baseline:
            baseline = self._calculate_baseline(
                df=df,
                load_col=load_col,
                reference_days=parsed_config.reference_days,
            )
            result_df[f"{load_col}{parsed_config.suffix_baseline}"] = baseline

            # Also needed for ratio and correction calculations
        else:
            baseline = self._calculate_baseline(
                df=df,
                load_col=load_col,
                reference_days=parsed_config.reference_days,
            )

        # Calculate ratio features
        if parsed_config.include_ratios:
            ratio = self._calculate_ratio(load_values, baseline)
            result_df[f"{load_col}{parsed_config.suffix_ratio}"] = ratio

        # Calculate correction features
        if parsed_config.include_corrections:
            for window in parsed_config.correction_windows:
                correction = self._calculate_correction(
                    df=df,
                    load_col=load_col,
                    baseline=baseline,
                    window=window,
                )
                col_name = f"{load_col}{parsed_config.suffix_correction}_{window}h"
                result_df[col_name] = correction

        # Calculate profile features
        if parsed_config.include_profile:
            profile = self._calculate_profile(df, load_col)
            result_df[f"{load_col}_profile"] = profile
            result_df[f"{load_col}_profile_deviation"] = load_values - profile

        logger.debug(
            "BLF features generated for '%s': ref_days=%d, windows=%s",
            load_col,
            parsed_config.reference_days,
            parsed_config.correction_windows,
        )

        return result_df

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """Return the list of feature names generated by this plugin.

        Args:
            config: Plugin configuration dictionary.

        Returns:
            List of feature column names that will be generated.
        """
        parsed_config = BLFStrategyConfig(**config)
        load_col = parsed_config.load_column
        feature_names: list[str] = []

        if parsed_config.include_baseline:
            feature_names.append(f"{load_col}{parsed_config.suffix_baseline}")

        if parsed_config.include_ratios:
            feature_names.append(f"{load_col}{parsed_config.suffix_ratio}")

        if parsed_config.include_corrections:
            for window in parsed_config.correction_windows:
                feature_names.append(f"{load_col}{parsed_config.suffix_correction}_{window}h")

        if parsed_config.include_profile:
            feature_names.append(f"{load_col}_profile")
            feature_names.append(f"{load_col}_profile_deviation")

        return feature_names

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate the plugin configuration.

        Args:
            config: Plugin configuration dictionary.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If configuration is invalid.
        """
        BLFStrategyConfig(**config)
        return True

    @staticmethod
    def _calculate_baseline(
        df: pd.DataFrame,
        load_col: str,
        reference_days: int,
    ) -> NDArray[np.floating[Any]]:
        """Calculate baseline load for each timestamp.

        The baseline is the rolling mean of the same hour of day and day of week
        over the reference period.

        Args:
            df: DataFrame with load column and datetime index.
            load_col: Name of the load column.
            reference_days: Number of days for reference calculation.

        Returns:
            Array of baseline values for each timestamp.
        """
        # Create hour and day-of-week features
        hour = df.index.hour
        day_of_week = df.index.dayofweek

        # Calculate rolling baseline by hour and day of week pattern
        baseline = np.full(len(df), np.nan)
        reference_hours = reference_days * 24

        for i in range(len(df)):
            # Find matching hour/day_of_week patterns in the reference period
            start_idx = max(0, i - reference_hours)
            mask = (
                (hour[start_idx:i] == hour[i])
                & (day_of_week[start_idx:i] == day_of_week[i])
            )

            matching_values = df[load_col].iloc[start_idx:i][mask]
            if len(matching_values) > 0:
                baseline[i] = matching_values.mean()
            elif i > 0:
                # Fall back to simple rolling mean
                baseline[i] = df[load_col].iloc[max(0, i - 24) : i].mean()
            else:
                baseline[i] = df[load_col].iloc[i]

        return baseline

    @staticmethod
    def _calculate_ratio(
        load: NDArray[np.floating[Any]],
        baseline: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Calculate ratio of actual load to baseline.

        Args:
            load: Actual load values.
            baseline: Baseline values.

        Returns:
            Array of ratio values (load / baseline).
        """
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(baseline != 0, load / baseline, 1.0)
            return np.clip(ratio, 0.1, 10.0)  # Clip extreme ratios

    @staticmethod
    def _calculate_correction(
        df: pd.DataFrame,
        load_col: str,
        baseline: NDArray[np.floating[Any]],
        window: int,
    ) -> NDArray[np.floating[Any]]:
        """Calculate rolling correction factor.

        The correction is the rolling mean of (actual - baseline) over the window.

        Args:
            df: DataFrame with load column.
            load_col: Name of the load column.
            baseline: Baseline values.
            window: Window size in hours for rolling calculation.

        Returns:
            Array of correction values.
        """
        deviation = df[load_col].to_numpy() - baseline
        # Rolling mean of deviation
        return pd.Series(deviation).rolling(window=window, min_periods=1).mean().to_numpy()

    @staticmethod
    def _calculate_profile(
        df: pd.DataFrame,
        load_col: str,
    ) -> NDArray[np.floating[Any]]:
        """Calculate normalized hourly profile.

        The profile represents the typical load for each hour, normalized
        by the daily mean.

        Args:
            df: DataFrame with load column and datetime index.
            load_col: Name of the load column.

        Returns:
            Array of profile values (expected load for each hour).
        """
        # Calculate mean load by hour
        hourly_mean = df.groupby(df.index.hour)[load_col].transform("mean")
        return hourly_mean.to_numpy()
