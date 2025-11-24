"""Temporal features plugin for generating time-based features.

This module provides the TemporalFeaturesPlugin for generating temporal features
from datetime indexes, including basic temporal components (hour, day of week, etc.),
cyclical encodings (sin/cos), and Southern Hemisphere season mappings.

Example:
    ```python
    from src.features.plugins.temporal import (
        TemporalFeaturesPlugin,
        TemporalFeaturesConfig,
    )
    import pandas as pd

    # Create plugin and config
    plugin = TemporalFeaturesPlugin()
    config = TemporalFeaturesConfig(
        include_cyclical=True,
        include_season=True,
        timezone="America/Sao_Paulo",
    )

    # Generate features
    df = pd.DataFrame(
        {"load": [100, 200, 300]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )
    result = plugin.generate_features(df, config.model_dump())
    ```
"""

from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default basic temporal features
DEFAULT_BASIC_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_business_day",
]

# Day of week constants (Monday=0, Sunday=6)
SATURDAY_DAYOFWEEK = 5  # Saturday is day 5 in pandas dayofweek (0-indexed)

# Southern Hemisphere season mapping (month -> season)
# Summer: Dec(12), Jan(1), Feb(2) -> 1
# Autumn: Mar(3), Apr(4), May(5) -> 2
# Winter: Jun(6), Jul(7), Aug(8) -> 3
# Spring: Sep(9), Oct(10), Nov(11) -> 4
SOUTHERN_HEMISPHERE_SEASONS = {
    12: 1,  # December -> Summer
    1: 1,  # January -> Summer
    2: 1,  # February -> Summer
    3: 2,  # March -> Autumn
    4: 2,  # April -> Autumn
    5: 2,  # May -> Autumn
    6: 3,  # June -> Winter
    7: 3,  # July -> Winter
    8: 3,  # August -> Winter
    9: 4,  # September -> Spring
    10: 4,  # October -> Spring
    11: 4,  # November -> Spring
}


class TemporalFeaturesConfig(BaseModel):
    """Configuration model for TemporalFeaturesPlugin.

    This model defines configuration options for temporal feature generation,
    including cyclical encodings, season mappings, and timezone handling.

    Attributes:
        include_cyclical: Whether to include cyclical (sin/cos) encodings.
        include_season: Whether to include Southern Hemisphere season feature.
        timezone: Timezone for datetime localization (validates with zoneinfo).
        features: List of basic temporal features to generate.

    Example:
        ```python
        config = TemporalFeaturesConfig(
            include_cyclical=True,
            include_season=True,
            timezone="America/Sao_Paulo",
            features=["hour", "day_of_week", "month"],
        )
        ```
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
    )

    include_cyclical: bool = Field(
        default=True,
        description="Whether to include cyclical (sin/cos) encodings for hour, day_of_week, and month",
    )
    include_season: bool = Field(
        default=True,
        description="Whether to include Southern Hemisphere season feature",
    )
    timezone: str = Field(
        default="America/Sao_Paulo",
        description="Timezone for datetime localization (IANA timezone name)",
    )
    features: list[str] = Field(
        default_factory=lambda: DEFAULT_BASIC_FEATURES.copy(),
        description="List of basic temporal features to generate",
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate that the timezone is a valid IANA timezone.

        Args:
            v: Timezone string to validate.

        Returns:
            The validated timezone string.

        Raises:
            ValueError: If the timezone is not valid.
        """
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as e:
            msg = f"Invalid timezone '{v}'. Must be a valid IANA timezone name."
            raise ValueError(msg) from e
        return v

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: list[str]) -> list[str]:
        """Validate that all requested features are supported.

        Args:
            v: List of feature names to validate.

        Returns:
            The validated feature list.

        Raises:
            ValueError: If any feature is not supported.
        """
        valid_features = set(DEFAULT_BASIC_FEATURES)
        invalid = [f for f in v if f not in valid_features]
        if invalid:
            msg = f"Invalid features: {invalid}. Valid options: {sorted(valid_features)}"
            raise ValueError(msg)
        return v


class TemporalFeaturesPlugin(BaseFeaturePlugin):
    """Plugin for generating temporal features from datetime indexes.

    This plugin generates various temporal features including:
    - Basic features: hour, day_of_week, day_of_month, month, quarter, year
    - Flags: is_weekend, is_business_day
    - Season: Southern Hemisphere season mapping (1=Summer, 2=Autumn, 3=Winter, 4=Spring)
    - Cyclical encodings: sin/cos for hour (24-cycle), day_of_week (7-cycle), month (12-cycle)

    The plugin handles both DatetimeIndex and timestamp columns, and supports
    timezone localization for naive datetimes.

    Example:
        ```python
        plugin = TemporalFeaturesPlugin()
        config = {"include_cyclical": True, "include_season": True}
        result = plugin.generate_features(df, config)
        ```
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this plugin.

        Returns:
            Plugin name "temporal_features".
        """
        return "temporal_features"

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
        """Generate temporal features from the input DataFrame.

        This method extracts temporal features from the DataFrame's DatetimeIndex
        or a 'timestamp' column if present. It generates basic temporal features,
        optional cyclical encodings, and optional Southern Hemisphere seasons.

        Args:
            df: Input DataFrame with DatetimeIndex or 'timestamp' column.
            config: Plugin configuration dictionary. Will be validated against
                TemporalFeaturesConfig.

        Returns:
            DataFrame with additional temporal feature columns.

        Raises:
            ValueError: If the DataFrame has no valid datetime index or column.
            TypeError: If the input DataFrame has incorrect structure.
        """
        # Validate and parse configuration
        parsed_config = TemporalFeaturesConfig(**config)
        logger.debug(
            "Generating temporal features with config: cyclical=%s, season=%s, tz=%s",
            parsed_config.include_cyclical,
            parsed_config.include_season,
            parsed_config.timezone,
        )

        # Make a copy to avoid modifying the original
        result = df.copy()

        # Get datetime index, handling both DatetimeIndex and timestamp column
        dt_index = self._get_datetime_index(result, parsed_config.timezone)

        # Generate basic temporal features
        result = self._generate_basic_features(result, dt_index, parsed_config.features)

        # Generate season feature if enabled
        if parsed_config.include_season:
            result = self._generate_season_feature(result, dt_index)

        # Generate cyclical encodings if enabled
        if parsed_config.include_cyclical:
            result = self._generate_cyclical_features(result, dt_index)

        logger.debug(
            "Generated %d temporal features for %d rows",
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
        parsed_config = TemporalFeaturesConfig(**config)

        feature_names: list[str] = []

        # Add basic features
        feature_names.extend(parsed_config.features)

        # Add season feature if enabled
        if parsed_config.include_season:
            feature_names.append("season")

        # Add cyclical features if enabled
        if parsed_config.include_cyclical:
            cyclical_features = [
                "hour_sin",
                "hour_cos",
                "day_of_week_sin",
                "day_of_week_cos",
                "month_sin",
                "month_cos",
            ]
            feature_names.extend(cyclical_features)

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
        TemporalFeaturesConfig(**config)
        logger.debug("Configuration validated for plugin '%s'", self.name)
        return True

    def _get_datetime_index(
        self,
        df: pd.DataFrame,
        timezone: str,
    ) -> pd.DatetimeIndex:
        """Extract and localize datetime index from DataFrame.

        Args:
            df: Input DataFrame.
            timezone: Target timezone for localization.

        Returns:
            DatetimeIndex localized to the specified timezone.

        Raises:
            ValueError: If no valid datetime index or column is found.
        """
        # Check if index is DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            dt_index = df.index
        # Check for timestamp column
        elif "timestamp" in df.columns:
            dt_index = pd.DatetimeIndex(df["timestamp"])
        else:
            msg = (
                "DataFrame must have a DatetimeIndex or a 'timestamp' column. "
                f"Found index type: {type(df.index).__name__}, "
                f"columns: {list(df.columns)}"
            )
            raise ValueError(msg)

        # Localize naive datetimes to the specified timezone
        if dt_index.tz is None:
            dt_index = dt_index.tz_localize(timezone)
            logger.debug("Localized naive datetime index to %s", timezone)
        elif str(dt_index.tz) != timezone:
            dt_index = dt_index.tz_convert(timezone)
            logger.debug("Converted datetime index to %s", timezone)

        return dt_index

    def _generate_basic_features(
        self,
        df: pd.DataFrame,
        dt_index: pd.DatetimeIndex,
        features: list[str],
    ) -> pd.DataFrame:
        """Generate basic temporal features.

        Args:
            df: DataFrame to add features to.
            dt_index: DatetimeIndex to extract features from.
            features: List of features to generate.

        Returns:
            DataFrame with basic temporal features added.
        """
        for feature in features:
            if feature == "hour":
                df["hour"] = dt_index.hour
            elif feature == "day_of_week":
                df["day_of_week"] = dt_index.dayofweek
            elif feature == "day_of_month":
                df["day_of_month"] = dt_index.day
            elif feature == "month":
                df["month"] = dt_index.month
            elif feature == "quarter":
                df["quarter"] = dt_index.quarter
            elif feature == "year":
                df["year"] = dt_index.year
            elif feature == "is_weekend":
                df["is_weekend"] = (dt_index.dayofweek >= SATURDAY_DAYOFWEEK).astype(np.int32)
            elif feature == "is_business_day":
                # Business day: Monday-Friday (dayofweek 0-4)
                df["is_business_day"] = (dt_index.dayofweek < SATURDAY_DAYOFWEEK).astype(np.int32)

        return df

    def _generate_season_feature(
        self,
        df: pd.DataFrame,
        dt_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate Southern Hemisphere season feature.

        Season values:
        - 1: Summer (Dec, Jan, Feb)
        - 2: Autumn (Mar, Apr, May)
        - 3: Winter (Jun, Jul, Aug)
        - 4: Spring (Sep, Oct, Nov)

        Args:
            df: DataFrame to add feature to.
            dt_index: DatetimeIndex to extract month from.

        Returns:
            DataFrame with season feature added.
        """
        months = dt_index.month.to_numpy()
        df["season"] = np.vectorize(SOUTHERN_HEMISPHERE_SEASONS.get)(months)
        return df

    def _generate_cyclical_features(
        self,
        df: pd.DataFrame,
        dt_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate cyclical sin/cos encodings for temporal features.

        Cyclical encoding formula:
        - sin(2 * pi * value / max_value)
        - cos(2 * pi * value / max_value)

        This encoding preserves the cyclical nature of time features,
        e.g., hour 23 is close to hour 0.

        Args:
            df: DataFrame to add features to.
            dt_index: DatetimeIndex to extract values from.

        Returns:
            DataFrame with cyclical features added.
        """
        two_pi = 2 * np.pi

        # Hour cyclical encoding (24-cycle)
        hours = dt_index.hour.to_numpy()
        df["hour_sin"] = np.sin(two_pi * hours / 24)
        df["hour_cos"] = np.cos(two_pi * hours / 24)

        # Day of week cyclical encoding (7-cycle)
        day_of_week = dt_index.dayofweek.to_numpy()
        df["day_of_week_sin"] = np.sin(two_pi * day_of_week / 7)
        df["day_of_week_cos"] = np.cos(two_pi * day_of_week / 7)

        # Month cyclical encoding (12-cycle, using 1-indexed month)
        months = dt_index.month.to_numpy()
        df["month_sin"] = np.sin(two_pi * (months - 1) / 12)
        df["month_cos"] = np.cos(two_pi * (months - 1) / 12)

        return df
