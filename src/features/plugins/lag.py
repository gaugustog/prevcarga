"""Lag features plugin for generating autoregressive and rolling statistics features.

This module provides the LagFeaturesPlugin for generating lag (autoregressive) features
and rolling statistics from time series data. It supports configurable lag periods,
multiple rolling window sizes, and various aggregation statistics.

Example:
    ```python
    from src.features.plugins.lag import (
        LagFeaturesPlugin,
        LagFeaturesConfig,
    )
    import pandas as pd

    # Create plugin and config
    plugin = LagFeaturesPlugin()
    config = LagFeaturesConfig(
        lag_periods=[1, 2, 24, 168],
        rolling_windows=[24, 168],
        rolling_stats=["mean", "std"],
        target_column="carga",
    )

    # Generate features
    df = pd.DataFrame(
        {"carga": [100, 200, 300, 400, 500]},
        index=pd.date_range("2024-01-01", periods=5, freq="h"),
    )
    result = plugin.generate_features(df, config.model_dump())
    ```
"""

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported rolling statistics
SUPPORTED_ROLLING_STATS = frozenset({"mean", "std", "min", "max", "median", "sum"})

# Default configuration values
DEFAULT_LAG_PERIODS = [1, 2, 24, 168]
DEFAULT_ROLLING_WINDOWS = [24, 168]
DEFAULT_ROLLING_STATS = ["mean", "std", "min", "max"]


class LagFeaturesConfig(BaseModel):
    """Configuration model for LagFeaturesPlugin.

    This model defines configuration options for lag feature generation,
    including lag periods, rolling window sizes, and aggregation statistics.

    Attributes:
        lag_periods: List of lag periods (in time steps) to generate.
            Default: [1, 2, 24, 168] (1h, 2h, 1 day, 1 week for hourly data).
        rolling_windows: List of rolling window sizes for statistics.
            Default: [24, 168] (1 day, 1 week for hourly data).
        rolling_stats: List of rolling statistics to compute.
            Default: ["mean", "std", "min", "max"].
            Supported: "mean", "std", "min", "max", "median", "sum".
        target_column: Name of the target column to generate features from.
            Default: "carga".
        additional_columns: List of additional columns to generate features from.
            Default: []. Missing columns will be skipped with a warning.
        fill_method: Method to handle NaN values after feature generation.
            - "none": Leave NaN values (default, initial periods will have NaN)
            - "forward": Forward fill (ffill)
            - "backward": Backward fill (bfill)

    Example:
        ```python
        config = LagFeaturesConfig(
            lag_periods=[1, 24, 48],
            rolling_windows=[12, 24],
            rolling_stats=["mean", "median"],
            target_column="carga",
            fill_method="none",
        )
        ```
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
    )

    lag_periods: list[int] = Field(
        default_factory=lambda: DEFAULT_LAG_PERIODS.copy(),
        description="List of lag periods (positive integers) to generate",
    )
    rolling_windows: list[int] = Field(
        default_factory=lambda: DEFAULT_ROLLING_WINDOWS.copy(),
        description="List of rolling window sizes (positive integers) for statistics",
    )
    rolling_stats: list[str] = Field(
        default_factory=lambda: DEFAULT_ROLLING_STATS.copy(),
        description="List of rolling statistics to compute (mean, std, min, max, median, sum)",
    )
    target_column: str = Field(
        default="carga",
        min_length=1,
        description="Name of the target column to generate features from",
    )
    additional_columns: list[str] = Field(
        default_factory=list,
        description="List of additional columns to generate features from",
    )
    fill_method: Literal["none", "forward", "backward"] = Field(
        default="none",
        description="Method to handle NaN values: none, forward, or backward",
    )

    @field_validator("lag_periods")
    @classmethod
    def validate_lag_periods(cls, v: list[int]) -> list[int]:
        """Validate that all lag periods are positive integers.

        Args:
            v: List of lag periods to validate.

        Returns:
            The validated lag periods list.

        Raises:
            ValueError: If any lag period is not positive.
        """
        if not v:
            return v

        for period in v:
            if period <= 0:
                msg = f"All lag periods must be positive integers, got: {period}"
                raise ValueError(msg)

        return v

    @field_validator("rolling_windows")
    @classmethod
    def validate_rolling_windows(cls, v: list[int]) -> list[int]:
        """Validate that all rolling windows are positive integers.

        Args:
            v: List of rolling window sizes to validate.

        Returns:
            The validated rolling windows list.

        Raises:
            ValueError: If any rolling window is not positive.
        """
        if not v:
            return v

        for window in v:
            if window <= 0:
                msg = f"All rolling windows must be positive integers, got: {window}"
                raise ValueError(msg)

        return v

    @field_validator("rolling_stats")
    @classmethod
    def validate_rolling_stats(cls, v: list[str]) -> list[str]:
        """Validate that all rolling statistics are supported.

        Args:
            v: List of rolling statistics to validate.

        Returns:
            The validated rolling statistics list.

        Raises:
            ValueError: If any statistic is not supported.
        """
        if not v:
            return v

        invalid = [stat for stat in v if stat not in SUPPORTED_ROLLING_STATS]
        if invalid:
            msg = (
                f"Invalid rolling statistics: {invalid}. "
                f"Supported: {sorted(SUPPORTED_ROLLING_STATS)}"
            )
            raise ValueError(msg)

        return v


class LagFeaturesPlugin(BaseFeaturePlugin):
    """Plugin for generating lag and rolling statistics features.

    This plugin generates autoregressive (lag) features and rolling window
    statistics from time series data. It is designed for electric load
    forecasting where historical patterns are important predictors.

    Features generated:
    - Lag features: `{column}_lag_{period}` for each column and lag period
    - Rolling features: `{column}_rolling_{window}_{stat}` for each column,
      window size, and statistic

    The plugin uses pandas shift() for efficient lagging and pandas rolling()
    for computing rolling statistics.

    Example:
        ```python
        plugin = LagFeaturesPlugin()
        config = {"lag_periods": [1, 24], "rolling_windows": [24]}
        result = plugin.generate_features(df, config)
        ```
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this plugin.

        Returns:
            Plugin name "lag_features".
        """
        return "lag_features"

    @property
    def version(self) -> str:
        """Return the plugin version string.

        Returns:
            Version string "1.0.0".
        """
        return "1.0.0"

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate lag and rolling statistics features from the input DataFrame.

        This method generates lag features using pandas shift() and rolling
        statistics using pandas rolling(). Features are generated for the
        target column and any additional columns specified in the configuration.

        Args:
            df: Input DataFrame with time series data. Must contain the
                target column specified in the configuration.
            config: Plugin configuration dictionary. Will be validated against
                LagFeaturesConfig.

        Returns:
            DataFrame with additional lag and rolling feature columns.

        Raises:
            ValueError: If the target column is not found in the DataFrame.
        """
        # Validate and parse configuration
        parsed_config = LagFeaturesConfig(**config)
        logger.debug(
            "Generating lag features with config: lags=%s, windows=%s, stats=%s",
            parsed_config.lag_periods,
            parsed_config.rolling_windows,
            parsed_config.rolling_stats,
        )

        # Make a copy to avoid modifying the original
        result = df.copy()

        # Validate target column exists
        if parsed_config.target_column not in result.columns:
            msg = (
                f"Target column '{parsed_config.target_column}' not found in DataFrame. "
                f"Available columns: {list(result.columns)}"
            )
            raise ValueError(msg)

        # Collect columns to process
        columns_to_process = [parsed_config.target_column]

        # Add additional columns, skipping missing ones with warning
        for col in parsed_config.additional_columns:
            if col in result.columns:
                columns_to_process.append(col)
            else:
                logger.warning(
                    "Additional column '%s' not found in DataFrame, skipping",
                    col,
                )

        # Generate features for each column
        for column in columns_to_process:
            # Generate lag features
            result = self._generate_lag_features(result, column, parsed_config.lag_periods)

            # Generate rolling statistics features
            result = self._generate_rolling_features(
                result,
                column,
                parsed_config.rolling_windows,
                parsed_config.rolling_stats,
            )

        # Apply fill method for NaN handling
        result = self._apply_fill_method(result, parsed_config.fill_method)

        # Log NaN statistics
        self._log_nan_statistics(result)

        logger.debug(
            "Generated %d lag/rolling features for %d rows",
            len(self.get_feature_names(config)),
            len(result),
        )
        return result

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """Return the list of feature column names this plugin generates.

        Args:
            config: Plugin configuration dictionary.

        Returns:
            List of feature column names that will be generated.
        """
        # Parse configuration to get settings
        parsed_config = LagFeaturesConfig(**config)

        feature_names: list[str] = []

        # Collect columns (target + additional that would be processed)
        columns = [parsed_config.target_column, *parsed_config.additional_columns]

        for column in columns:
            # Add lag feature names
            for period in parsed_config.lag_periods:
                feature_names.append(f"{column}_lag_{period}")

            # Add rolling feature names
            for window in parsed_config.rolling_windows:
                for stat in parsed_config.rolling_stats:
                    feature_names.append(f"{column}_rolling_{window}_{stat}")

        return feature_names

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate the plugin configuration.

        Args:
            config: Plugin configuration dictionary to validate.

        Returns:
            True if the configuration is valid.

        Raises:
            ValueError: If the configuration is invalid.
        """
        # First call parent validation
        super().validate_config(config)

        # Then validate with Pydantic model (will raise on invalid config)
        LagFeaturesConfig(**config)
        logger.debug("Configuration validated for plugin '%s'", self.name)
        return True

    def _generate_lag_features(
        self,
        df: pd.DataFrame,
        column: str,
        lag_periods: list[int],
    ) -> pd.DataFrame:
        """Generate lag features for a single column.

        Args:
            df: DataFrame to add features to.
            column: Column name to create lags from.
            lag_periods: List of lag periods to generate.

        Returns:
            DataFrame with lag features added.
        """
        for period in lag_periods:
            feature_name = f"{column}_lag_{period}"
            df[feature_name] = df[column].shift(period)
            logger.debug("Generated lag feature: %s", feature_name)

        return df

    def _generate_rolling_features(
        self,
        df: pd.DataFrame,
        column: str,
        rolling_windows: list[int],
        rolling_stats: list[str],
    ) -> pd.DataFrame:
        """Generate rolling statistics features for a single column.

        Args:
            df: DataFrame to add features to.
            column: Column name to compute rolling statistics from.
            rolling_windows: List of rolling window sizes.
            rolling_stats: List of statistics to compute.

        Returns:
            DataFrame with rolling features added.
        """
        for window in rolling_windows:
            # Create rolling window object once per window size
            rolling = df[column].rolling(window=window)

            for stat in rolling_stats:
                feature_name = f"{column}_rolling_{window}_{stat}"

                if stat == "mean":
                    df[feature_name] = rolling.mean()
                elif stat == "std":
                    df[feature_name] = rolling.std()
                elif stat == "min":
                    df[feature_name] = rolling.min()
                elif stat == "max":
                    df[feature_name] = rolling.max()
                elif stat == "median":
                    df[feature_name] = rolling.median()
                elif stat == "sum":
                    df[feature_name] = rolling.sum()

                logger.debug("Generated rolling feature: %s", feature_name)

        return df

    def _apply_fill_method(
        self,
        df: pd.DataFrame,
        fill_method: Literal["none", "forward", "backward"],
    ) -> pd.DataFrame:
        """Apply the specified fill method to handle NaN values.

        Args:
            df: DataFrame with potential NaN values.
            fill_method: Method to use for filling NaN values.

        Returns:
            DataFrame with NaN values handled according to fill method.
        """
        if fill_method == "forward":
            df = df.ffill()
            logger.debug("Applied forward fill to handle NaN values")
        elif fill_method == "backward":
            df = df.bfill()
            logger.debug("Applied backward fill to handle NaN values")
        # "none" leaves NaN values as is

        return df

    def _log_nan_statistics(self, df: pd.DataFrame) -> None:
        """Log statistics about NaN values in the DataFrame.

        Args:
            df: DataFrame to analyze for NaN values.
        """
        nan_counts = df.isna().sum()
        total_nans = nan_counts.sum()

        if total_nans > 0:
            columns_with_nans = nan_counts[nan_counts > 0]
            logger.debug(
                "NaN statistics after feature generation: %d total NaN values across %d columns",
                total_nans,
                len(columns_with_nans),
            )
            for col, count in columns_with_nans.items():
                logger.debug("  %s: %d NaN values", col, count)
