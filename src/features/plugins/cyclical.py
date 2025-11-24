"""Cyclical encoding plugin for periodic feature transformation.

This module provides the CyclicalEncodingPlugin for encoding periodic features
using sine and cosine transformations. This encoding preserves the cyclical
nature of periodic data, where values at the end of a period are close to
values at the beginning (e.g., hour 23 is close to hour 0).

Example:
    ```python
    from src.features.plugins.cyclical import (
        CyclicalEncodingPlugin,
        CyclicalEncodingConfig,
    )
    import pandas as pd

    # Create plugin and config
    plugin = CyclicalEncodingPlugin()
    config = CyclicalEncodingConfig(
        features={"hour": 24, "day_of_week": 7, "month": 12},
        keep_original=True,
    )

    # Generate features
    df = pd.DataFrame({
        "hour": [0, 6, 12, 18, 23],
        "day_of_week": [0, 1, 2, 3, 4],
    })
    result = plugin.generate_features(df, config.model_dump())
    ```
"""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CyclicalEncodingConfig(BaseModel):
    """Configuration model for CyclicalEncodingPlugin.

    This model defines configuration options for cyclical feature encoding,
    including the features to encode, their periods, and naming options.

    Attributes:
        features: Dictionary mapping feature names to their periods.
            Period must be a positive integer representing the cycle length.
            For example, {"hour": 24, "day_of_week": 7, "month": 12}.
        keep_original: Whether to keep the original feature columns.
        suffix_sin: Suffix for sine-encoded columns.
        suffix_cos: Suffix for cosine-encoded columns.

    Example:
        ```python
        config = CyclicalEncodingConfig(
            features={"hour": 24, "day_of_week": 7},
            keep_original=False,
            suffix_sin="_sin",
            suffix_cos="_cos",
        )
        ```
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
    )

    features: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of feature names to their periods (cycle lengths)",
    )
    keep_original: bool = Field(
        default=True,
        description="Whether to keep the original feature columns",
    )
    suffix_sin: str = Field(
        default="_sin",
        description="Suffix for sine-encoded columns",
    )
    suffix_cos: str = Field(
        default="_cos",
        description="Suffix for cosine-encoded columns",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: dict[str, int]) -> dict[str, int]:
        """Validate that all periods are positive integers.

        Args:
            v: Dictionary of feature names to periods.

        Returns:
            The validated features dictionary.

        Raises:
            ValueError: If any period is not positive.
        """
        for feature_name, period in v.items():
            if period <= 0:
                msg = f"Period for feature '{feature_name}' must be positive, got {period}"
                raise ValueError(msg)
        return v


class CyclicalEncodingPlugin(BaseFeaturePlugin):
    """Plugin for encoding periodic features using sine/cosine transformations.

    This plugin transforms periodic features into cyclical encodings using:
    - sin(2 * pi * value / period)
    - cos(2 * pi * value / period)

    This encoding preserves the cyclical nature of periodic data, ensuring that
    values at opposite ends of the period (e.g., hour 0 and hour 23) are
    represented as nearby points in the encoded space.

    Mathematical properties:
    - sin(0) = 0, cos(0) = 1
    - sin(2*pi) = 0, cos(2*pi) = 1
    - sin^2 + cos^2 = 1 for all values

    Example:
        ```python
        plugin = CyclicalEncodingPlugin()
        config = {"features": {"hour": 24}, "keep_original": True}
        result = plugin.generate_features(df, config)
        # Creates hour_sin and hour_cos columns
        ```
    """

    @property
    def name(self) -> str:
        """Return the unique identifier for this plugin.

        Returns:
            Plugin name "cyclical_encoding".
        """
        return "cyclical_encoding"

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
        """Generate cyclical encoded features from the input DataFrame.

        This method encodes specified periodic features using sine and cosine
        transformations. Each feature is transformed into two new columns:
        {feature_name}{suffix_sin} and {feature_name}{suffix_cos}.

        Args:
            df: Input DataFrame containing the features to encode.
            config: Plugin configuration dictionary. Will be validated against
                CyclicalEncodingConfig.

        Returns:
            DataFrame with additional cyclical encoded feature columns.
            If keep_original is False, the original feature columns are removed.

        Raises:
            ValueError: If a specified feature does not exist in the DataFrame.
        """
        # Validate and parse configuration
        parsed_config = CyclicalEncodingConfig(**config)

        # Handle empty features dictionary
        if not parsed_config.features:
            logger.debug("No features specified for cyclical encoding, returning copy")
            return df.copy()

        logger.debug(
            "Generating cyclical encodings for %d features: %s",
            len(parsed_config.features),
            list(parsed_config.features.keys()),
        )

        # Make a copy to avoid modifying the original
        result = df.copy()

        # Validate that all features exist in the DataFrame
        missing_features = [f for f in parsed_config.features if f not in result.columns]
        if missing_features:
            msg = (
                f"Features not found in DataFrame: {missing_features}. "
                f"Available columns: {list(result.columns)}"
            )
            raise ValueError(msg)

        # Generate cyclical encodings for each feature
        two_pi = 2 * np.pi
        encoded_count = 0

        for feature_name, period in parsed_config.features.items():
            values = result[feature_name].to_numpy()
            normalized = values / period

            # Calculate sin and cos encodings
            sin_values = np.sin(two_pi * normalized)
            cos_values = np.cos(two_pi * normalized)

            # Create column names
            sin_col = f"{feature_name}{parsed_config.suffix_sin}"
            cos_col = f"{feature_name}{parsed_config.suffix_cos}"

            # Add encoded columns
            result[sin_col] = sin_values
            result[cos_col] = cos_values
            encoded_count += 1

            logger.debug(
                "Encoded feature '%s' (period=%d) -> '%s', '%s'",
                feature_name,
                period,
                sin_col,
                cos_col,
            )

        # Remove original columns if requested
        if not parsed_config.keep_original:
            columns_to_drop = list(parsed_config.features.keys())
            result = result.drop(columns=columns_to_drop)
            logger.debug(
                "Removed %d original feature columns: %s",
                len(columns_to_drop),
                columns_to_drop,
            )

        logger.debug(
            "Generated cyclical encodings: %d features -> %d encoded columns",
            encoded_count,
            encoded_count * 2,
        )

        return result

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """Return the list of feature column names this plugin generates.

        Args:
            config: Plugin configuration dictionary.

        Returns:
            List of feature column names that will be generated.
            Includes sin/cos pairs for each feature, and optionally
            the original feature names if keep_original is True.
        """
        parsed_config = CyclicalEncodingConfig(**config)

        feature_names: list[str] = []

        # Add original feature names if keeping them
        if parsed_config.keep_original:
            feature_names.extend(parsed_config.features.keys())

        # Add cyclical encoded feature names
        for feature_name in parsed_config.features:
            feature_names.append(f"{feature_name}{parsed_config.suffix_sin}")
            feature_names.append(f"{feature_name}{parsed_config.suffix_cos}")

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
        CyclicalEncodingConfig(**config)
        logger.debug("Configuration validated for plugin '%s'", self.name)
        return True
