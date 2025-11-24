"""Wavelet transform feature plugin for time series decomposition.

This module provides the WaveletTransformPlugin for generating features
using wavelet decomposition, which decomposes signals into different
frequency components at various scales.

Example:
    ```python
    from src.features.plugins.wavelet import (
        WaveletTransformPlugin,
        WaveletTransformConfig,
    )
    import pandas as pd

    # Create plugin and config
    plugin = WaveletTransformPlugin()
    config = WaveletTransformConfig(
        columns=["load"],
        wavelet="db4",
        level=3,
        include_approximation=True,
        include_details=True,
    )

    # Generate wavelet features
    df = pd.DataFrame(
        {"load": [100, 105, 95, 110, 102, 98, 115, 108]},
        index=pd.date_range("2024-01-01", periods=8, freq="h"),
    )
    result = plugin.generate_features(df, config.model_dump())
    # Result includes: load_approx, load_detail_1, load_detail_2, load_detail_3
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

# Supported wavelet families with their filter coefficients
# Using Daubechies wavelets as they are commonly used for time series
WAVELET_FILTERS: dict[str, tuple[list[float], list[float]]] = {
    "haar": (
        [0.7071067811865476, 0.7071067811865476],  # Low-pass (approximation)
        [0.7071067811865476, -0.7071067811865476],  # High-pass (detail)
    ),
    "db2": (
        [0.3415063509461096, 0.5915063509461097, 0.1584936490538903, -0.0915063509461096],
        [-0.0915063509461096, -0.1584936490538903, 0.5915063509461097, -0.3415063509461096],
    ),
    "db4": (
        [
            0.23037781330885523,
            0.7148465705525415,
            0.6308807679295904,
            -0.02798376941698385,
            -0.18703481171888114,
            0.030841381835986965,
            0.032883011666982945,
            -0.010597401784997278,
        ],
        [
            -0.010597401784997278,
            -0.032883011666982945,
            0.030841381835986965,
            0.18703481171888114,
            -0.02798376941698385,
            -0.6308807679295904,
            0.7148465705525415,
            -0.23037781330885523,
        ],
    ),
}

# Supported wavelet names
SUPPORTED_WAVELETS = list(WAVELET_FILTERS.keys())

# Default decomposition level
DEFAULT_LEVEL = 3


class WaveletTransformConfig(BaseModel):
    """Configuration model for WaveletTransformPlugin.

    This model defines configuration options for wavelet decomposition,
    including the wavelet type, decomposition level, and which components
    to include in the output.

    Attributes:
        columns: List of column names to apply wavelet transform to.
        wavelet: Wavelet type (haar, db2, db4).
        level: Decomposition level (1-8).
        include_approximation: Whether to include approximation coefficients.
        include_details: Whether to include detail coefficients.
        suffix_approx: Suffix for approximation feature names.
        suffix_detail: Suffix for detail feature names.

    Example:
        ```python
        config = WaveletTransformConfig(
            columns=["load"],
            wavelet="db4",
            level=3,
            include_approximation=True,
            include_details=True,
        )
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    columns: list[str] = Field(
        default_factory=list,
        description="Column names to apply wavelet transform to",
    )
    wavelet: str = Field(
        default="db4",
        description=f"Wavelet type. Supported: {SUPPORTED_WAVELETS}",
    )
    level: int = Field(
        default=DEFAULT_LEVEL,
        ge=1,
        le=8,
        description="Decomposition level (1-8)",
    )
    include_approximation: bool = Field(
        default=True,
        description="Include approximation (low-frequency) coefficients",
    )
    include_details: bool = Field(
        default=True,
        description="Include detail (high-frequency) coefficients",
    )
    suffix_approx: str = Field(
        default="_approx",
        description="Suffix for approximation feature column names",
    )
    suffix_detail: str = Field(
        default="_detail",
        description="Suffix for detail feature column names",
    )

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v: list[str]) -> list[str]:
        """Validate that column names are non-empty strings."""
        for col in v:
            if not col or not col.strip():
                msg = "Column names cannot be empty strings"
                raise ValueError(msg)
        return v

    @field_validator("wavelet")
    @classmethod
    def validate_wavelet(cls, v: str) -> str:
        """Validate that wavelet type is supported."""
        v_lower = v.lower()
        if v_lower not in SUPPORTED_WAVELETS:
            msg = f"Unsupported wavelet '{v}'. Supported: {SUPPORTED_WAVELETS}"
            raise ValueError(msg)
        return v_lower

    @field_validator("suffix_approx", "suffix_detail")
    @classmethod
    def validate_suffix(cls, v: str) -> str:
        """Validate that suffixes are non-empty."""
        if not v or not v.strip():
            msg = "Suffixes cannot be empty strings"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_at_least_one_output(self) -> WaveletTransformConfig:
        """Validate that at least one of approximation or details is enabled."""
        if not self.include_approximation and not self.include_details:
            msg = "At least one of include_approximation or include_details must be True"
            raise ValueError(msg)
        return self


class WaveletTransformPlugin(BaseFeaturePlugin):
    """Feature plugin for wavelet transform decomposition of time series.

    Wavelet decomposition breaks down a signal into approximation (low-frequency
    trend) and detail (high-frequency variations) components at multiple scales.
    This is useful for:
    - Separating trend from noise
    - Multi-scale feature extraction
    - Pattern recognition at different time scales

    The plugin generates:
    - Approximation coefficients (overall trend)
    - Detail coefficients at each level (variations at that scale)

    Example:
        ```python
        plugin = WaveletTransformPlugin()
        config = {
            "columns": ["load"],
            "wavelet": "db4",
            "level": 3,
        }
        result = plugin.generate_features(df, config)
        # Generates: load_approx, load_detail_1, load_detail_2, load_detail_3
        ```

    Attributes:
        name: Plugin identifier ("wavelet_transform").
        version: Plugin version ("1.0.0").

    Note:
        The implementation uses pure NumPy and does not require PyWavelets.
        Supported wavelets: haar, db2, db4.
    """

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "wavelet_transform"

    @property
    def version(self) -> str:
        """Return the plugin version."""
        return "1.0.0"

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate wavelet transform features.

        Args:
            df: Input DataFrame with numeric columns to transform.
            config: Plugin configuration dictionary.

        Returns:
            DataFrame with wavelet decomposition columns added.

        Raises:
            ValueError: If required columns are missing or insufficient data.
        """
        parsed_config = WaveletTransformConfig(**config)

        if not parsed_config.columns:
            logger.debug("No columns specified for wavelet transform, returning unchanged")
            return df

        # Validate columns exist
        missing_cols = [c for c in parsed_config.columns if c not in df.columns]
        if missing_cols:
            msg = f"Columns not found in DataFrame: {missing_cols}"
            raise ValueError(msg)

        # Get filter coefficients
        low_pass, high_pass = WAVELET_FILTERS[parsed_config.wavelet]

        result_df = df.copy()

        for col in parsed_config.columns:
            # Get numeric values
            y = df[col].to_numpy(dtype=np.float64)

            # Perform wavelet decomposition
            coeffs = self._wavedec(
                data=y,
                low_pass=low_pass,
                high_pass=high_pass,
                level=parsed_config.level,
            )

            # Reconstruct signals at original length
            reconstructed = self._reconstruct_signals(
                coeffs=coeffs,
                low_pass=low_pass,
                high_pass=high_pass,
                original_length=len(y),
            )

            # Add approximation feature
            if parsed_config.include_approximation:
                approx_col = f"{col}{parsed_config.suffix_approx}"
                result_df[approx_col] = reconstructed["approximation"]

            # Add detail features
            if parsed_config.include_details:
                for level, detail_signal in enumerate(reconstructed["details"], start=1):
                    detail_col = f"{col}{parsed_config.suffix_detail}_{level}"
                    result_df[detail_col] = detail_signal

            logger.debug(
                "Wavelet decomposition applied to '%s': wavelet=%s, level=%d",
                col,
                parsed_config.wavelet,
                parsed_config.level,
            )

        return result_df

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """Return the list of feature names generated by this plugin.

        Args:
            config: Plugin configuration dictionary.

        Returns:
            List of feature column names that will be generated.
        """
        parsed_config = WaveletTransformConfig(**config)
        feature_names: list[str] = []

        for col in parsed_config.columns:
            if parsed_config.include_approximation:
                feature_names.append(f"{col}{parsed_config.suffix_approx}")
            if parsed_config.include_details:
                for level in range(1, parsed_config.level + 1):
                    feature_names.append(f"{col}{parsed_config.suffix_detail}_{level}")

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
        WaveletTransformConfig(**config)
        return True

    @staticmethod
    def _convolve_down(
        data: NDArray[np.floating[Any]],
        filter_coef: list[float],
    ) -> NDArray[np.floating[Any]]:
        """Convolve and downsample by 2 (decimation).

        Args:
            data: Input signal.
            filter_coef: Filter coefficients.

        Returns:
            Downsampled filtered signal.
        """
        # Pad the signal for circular convolution
        filter_len = len(filter_coef)
        padded = np.concatenate([data[-filter_len + 1 :], data, data[: filter_len - 1]])

        # Convolve
        result = np.convolve(padded, filter_coef, mode="valid")

        # Downsample by 2
        return result[::2]

    @staticmethod
    def _upsample_convolve(
        data: NDArray[np.floating[Any]],
        filter_coef: list[float],
        target_length: int,
    ) -> NDArray[np.floating[Any]]:
        """Upsample by 2 and convolve (interpolation).

        Args:
            data: Input coefficients.
            filter_coef: Filter coefficients.
            target_length: Target length after reconstruction.

        Returns:
            Upsampled and filtered signal.
        """
        # Upsample by inserting zeros
        upsampled = np.zeros(len(data) * 2)
        upsampled[::2] = data

        # Convolve
        result = np.convolve(upsampled, filter_coef, mode="full")

        # Trim to target length (center the result)
        start = (len(result) - target_length) // 2
        start = max(start, 0)
        end = start + target_length
        if end > len(result):
            # Pad with zeros if needed
            padded = np.zeros(target_length)
            padded[: len(result) - start] = result[start:]
            return padded
        return result[start:end]

    def _wavedec(
        self,
        data: NDArray[np.floating[Any]],
        low_pass: list[float],
        high_pass: list[float],
        level: int,
    ) -> list[NDArray[np.floating[Any]]]:
        """Perform multi-level wavelet decomposition.

        Args:
            data: Input signal.
            low_pass: Low-pass filter coefficients.
            high_pass: High-pass filter coefficients.
            level: Number of decomposition levels.

        Returns:
            List of coefficients [cA_n, cD_n, cD_n-1, ..., cD_1]
            where cA_n is the final approximation and cD_i are details.
        """
        coefficients: list[NDArray[np.floating[Any]]] = []
        current = data

        for _ in range(level):
            # Decompose into approximation and detail
            approx = self._convolve_down(current, low_pass)
            detail = self._convolve_down(current, high_pass)

            # Store detail coefficients
            coefficients.insert(0, detail)

            # Continue with approximation
            current = approx

        # Add final approximation
        coefficients.insert(0, current)

        return coefficients

    def _reconstruct_signals(
        self,
        coeffs: list[NDArray[np.floating[Any]]],
        low_pass: list[float],
        high_pass: list[float],
        original_length: int,
    ) -> dict[str, Any]:
        """Reconstruct approximation and detail signals at original length.

        Args:
            coeffs: Wavelet coefficients from _wavedec.
            low_pass: Low-pass filter coefficients.
            high_pass: High-pass filter coefficients.
            original_length: Original signal length.

        Returns:
            Dictionary with 'approximation' and 'details' arrays.
        """
        # Reconstruction filters (reversed)
        low_recon = list(reversed(low_pass))
        high_recon = list(reversed(high_pass))

        num_levels = len(coeffs) - 1
        details: list[NDArray[np.floating[Any]]] = []

        # Reconstruct each detail level
        for level_idx in range(num_levels):
            detail_coef = coeffs[level_idx + 1]

            # Zero out all other coefficients
            zero_coeffs = [np.zeros_like(c) for c in coeffs]
            zero_coeffs[level_idx + 1] = detail_coef
            zero_coeffs[0] = np.zeros_like(coeffs[0])

            # Reconstruct
            detail_signal = self._waverec(zero_coeffs, low_recon, high_recon, original_length)
            details.append(detail_signal)

        # Reconstruct approximation (using only approximation coefficients)
        approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        approximation = self._waverec(approx_coeffs, low_recon, high_recon, original_length)

        return {"approximation": approximation, "details": details}

    def _waverec(
        self,
        coeffs: list[NDArray[np.floating[Any]]],
        low_recon: list[float],
        high_recon: list[float],
        target_length: int,
    ) -> NDArray[np.floating[Any]]:
        """Perform multi-level wavelet reconstruction.

        Args:
            coeffs: Wavelet coefficients.
            low_recon: Low-pass reconstruction filter.
            high_recon: High-pass reconstruction filter.
            target_length: Target signal length.

        Returns:
            Reconstructed signal.
        """
        current = coeffs[0]
        levels = len(coeffs) - 1

        for level_idx in range(levels):
            detail = coeffs[level_idx + 1]

            # Target length is 2x the current length
            level_target = len(current) * 2
            if level_idx == levels - 1:
                level_target = target_length

            # Upsample and filter both with same target
            approx_up = self._upsample_convolve(current, low_recon, level_target)
            detail_up = self._upsample_convolve(detail, high_recon, level_target)

            # Ensure same length for addition
            min_len = min(len(approx_up), len(detail_up))
            current = approx_up[:min_len] + detail_up[:min_len]

        # Pad or trim to target length
        if len(current) < target_length:
            result = np.zeros(target_length)
            result[: len(current)] = current
            return result
        return current[:target_length]
