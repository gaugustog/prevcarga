"""LOESS smoothing feature plugin for time series data.

This module provides the LoessSmoothingPlugin for generating smoothed features
using LOESS (Locally Estimated Scatterplot Smoothing) regression. LOESS is a
non-parametric method that fits local polynomial regressions to smooth data.

Example:
    ```python
    from src.features.plugins.smoothing import (
        LoessSmoothingPlugin,
        LoessSmoothingConfig,
    )
    import pandas as pd

    # Create plugin and config
    plugin = LoessSmoothingPlugin()
    config = LoessSmoothingConfig(
        columns=["load"],
        frac=0.1,
        degree=1,
    )

    # Generate smoothed features
    df = pd.DataFrame(
        {"load": [100, 105, 95, 110, 102, 98, 115]},
        index=pd.date_range("2024-01-01", periods=7, freq="h"),
    )
    result = plugin.generate_features(df, config.model_dump())
    # Result includes: load_loess (smoothed), load_residual (original - smoothed)
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

# Default LOESS parameters
DEFAULT_FRAC = 0.1  # 10% of data points in local regression window
DEFAULT_DEGREE = 1  # Linear local regression (1=linear, 2=quadratic)
DEFAULT_ITERATIONS = 3  # Number of robustifying iterations
MIN_POINTS_FOR_LOESS = 10  # Minimum data points required for LOESS


class LoessSmoothingConfig(BaseModel):
    """Configuration model for LoessSmoothingPlugin.

    This model defines configuration options for LOESS smoothing,
    including the fraction of data points to use in local regression,
    the polynomial degree, and robustifying iterations.

    Attributes:
        columns: List of column names to smooth.
        frac: Fraction of data to use in local regression (0 < frac <= 1).
        degree: Degree of local polynomial (1=linear, 2=quadratic).
        iterations: Number of robustifying iterations.
        include_residuals: Whether to include residual features.
        suffix_smooth: Suffix for smoothed feature names.
        suffix_residual: Suffix for residual feature names.

    Example:
        ```python
        config = LoessSmoothingConfig(
            columns=["load", "temperature"],
            frac=0.15,
            degree=1,
            include_residuals=True,
        )
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    columns: list[str] = Field(
        default_factory=list,
        description="Column names to apply LOESS smoothing to",
    )
    frac: float = Field(
        default=DEFAULT_FRAC,
        ge=0.01,
        le=1.0,
        description="Fraction of data to use in local regression (0 < frac <= 1)",
    )
    degree: int = Field(
        default=DEFAULT_DEGREE,
        ge=1,
        le=2,
        description="Degree of local polynomial (1=linear, 2=quadratic)",
    )
    iterations: int = Field(
        default=DEFAULT_ITERATIONS,
        ge=1,
        le=10,
        description="Number of robustifying iterations",
    )
    include_residuals: bool = Field(
        default=True,
        description="Whether to generate residual features (original - smoothed)",
    )
    suffix_smooth: str = Field(
        default="_loess",
        description="Suffix for smoothed feature column names",
    )
    suffix_residual: str = Field(
        default="_residual",
        description="Suffix for residual feature column names",
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

    @field_validator("suffix_smooth", "suffix_residual")
    @classmethod
    def validate_suffix(cls, v: str) -> str:
        """Validate that suffixes are non-empty."""
        if not v or not v.strip():
            msg = "Suffixes cannot be empty strings"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_suffixes_different(self) -> LoessSmoothingConfig:
        """Validate that smooth and residual suffixes are different."""
        if self.suffix_smooth == self.suffix_residual:
            msg = "suffix_smooth and suffix_residual must be different"
            raise ValueError(msg)
        return self


class LoessSmoothingPlugin(BaseFeaturePlugin):
    """Feature plugin for LOESS smoothing of time series data.

    LOESS (Locally Estimated Scatterplot Smoothing) performs local
    polynomial regression to smooth data. This is useful for:
    - Extracting trend components from noisy time series
    - Creating smoothed features for machine learning models
    - Identifying anomalies through residual analysis

    The plugin generates:
    - Smoothed values using LOESS regression
    - Residual values (original - smoothed) for anomaly detection

    Example:
        ```python
        plugin = LoessSmoothingPlugin()
        config = {
            "columns": ["load"],
            "frac": 0.1,
            "degree": 1,
            "include_residuals": True,
        }
        result = plugin.generate_features(df, config)
        ```

    Attributes:
        name: Plugin identifier ("loess_smoothing").
        version: Plugin version ("1.0.0").

    Note:
        The implementation uses a pure-numpy LOESS algorithm that
        does not require statsmodels or other heavy dependencies.
    """

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "loess_smoothing"

    @property
    def version(self) -> str:
        """Return the plugin version."""
        return "1.0.0"

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate LOESS smoothed features.

        Args:
            df: Input DataFrame with numeric columns to smooth.
            config: Plugin configuration dictionary.

        Returns:
            DataFrame with smoothed and optionally residual columns added.

        Raises:
            ValueError: If required columns are missing or insufficient data.
        """
        parsed_config = LoessSmoothingConfig(**config)

        if not parsed_config.columns:
            logger.debug("No columns specified for LOESS smoothing, returning unchanged")
            return df

        # Validate columns exist
        missing_cols = [c for c in parsed_config.columns if c not in df.columns]
        if missing_cols:
            msg = f"Columns not found in DataFrame: {missing_cols}"
            raise ValueError(msg)

        # Check minimum data points
        if len(df) < MIN_POINTS_FOR_LOESS:
            msg = f"Insufficient data points for LOESS: {len(df)} < {MIN_POINTS_FOR_LOESS}"
            raise ValueError(msg)

        result_df = df.copy()

        for col in parsed_config.columns:
            # Get numeric values
            y = df[col].to_numpy(dtype=np.float64)
            x = np.arange(len(y), dtype=np.float64)

            # Apply LOESS smoothing
            smoothed = self._loess_smooth(
                x=x,
                y=y,
                frac=parsed_config.frac,
                degree=parsed_config.degree,
                iterations=parsed_config.iterations,
            )

            # Add smoothed column
            smooth_col = f"{col}{parsed_config.suffix_smooth}"
            result_df[smooth_col] = smoothed

            # Add residual column if requested
            if parsed_config.include_residuals:
                residual_col = f"{col}{parsed_config.suffix_residual}"
                result_df[residual_col] = y - smoothed

            logger.debug(
                "LOESS smoothing applied to '%s': frac=%.3f, degree=%d",
                col,
                parsed_config.frac,
                parsed_config.degree,
            )

        return result_df

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """Return the list of feature names generated by this plugin.

        Args:
            config: Plugin configuration dictionary.

        Returns:
            List of feature column names that will be generated.
        """
        parsed_config = LoessSmoothingConfig(**config)
        feature_names: list[str] = []

        for col in parsed_config.columns:
            feature_names.append(f"{col}{parsed_config.suffix_smooth}")
            if parsed_config.include_residuals:
                feature_names.append(f"{col}{parsed_config.suffix_residual}")

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
        # Pydantic validation is performed during parsing
        LoessSmoothingConfig(**config)
        return True

    @staticmethod
    def _tricube_weights(distances: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Calculate tricube weights for LOESS.

        The tricube weight function is: (1 - |d|^3)^3 for |d| < 1, 0 otherwise.

        Args:
            distances: Normalized distances (0 to 1 range).

        Returns:
            Array of tricube weights.
        """
        # Clip distances to [0, 1] and apply tricube function
        d = np.clip(np.abs(distances), 0, 1)
        return np.power(1 - np.power(d, 3), 3)

    @staticmethod
    def _bisquare_weights(
        residuals: NDArray[np.floating[Any]],
        mad: float,
    ) -> NDArray[np.floating[Any]]:
        """Calculate bisquare weights for robustifying iterations.

        The bisquare weight function reduces the influence of outliers.

        Args:
            residuals: Residuals from the current fit.
            mad: Median absolute deviation of residuals.

        Returns:
            Array of bisquare weights.
        """
        if mad == 0:
            return np.ones_like(residuals)

        # Scale residuals by 6 * MAD
        scaled = residuals / (6 * mad)
        scaled = np.clip(np.abs(scaled), 0, 1)
        return np.power(1 - np.power(scaled, 2), 2)

    def _loess_smooth(
        self,
        x: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        frac: float,
        degree: int,
        iterations: int,
    ) -> NDArray[np.floating[Any]]:
        """Apply LOESS smoothing to the data.

        Args:
            x: Independent variable (typically time indices).
            y: Dependent variable (values to smooth).
            frac: Fraction of data points to use in local regression.
            degree: Degree of local polynomial.
            iterations: Number of robustifying iterations.

        Returns:
            Array of smoothed values.
        """
        n = len(y)
        k = max(3, int(np.ceil(frac * n)))  # Number of points in local window

        # Handle NaN values
        nan_mask = np.isnan(y)
        y_filled = np.where(nan_mask, 0, y)  # Fill NaN with 0 for calculations

        smoothed = np.zeros_like(y, dtype=np.float64)
        robustness_weights = np.ones(n, dtype=np.float64)

        for iteration in range(iterations):
            for i in range(n):
                if nan_mask[i]:
                    smoothed[i] = np.nan
                    continue

                # Find k nearest neighbors
                distances = np.abs(x - x[i])
                sorted_indices = np.argsort(distances)
                neighbor_indices = sorted_indices[:k]

                # Calculate local bandwidth (distance to k-th nearest neighbor)
                bandwidth = distances[neighbor_indices[-1]]
                if bandwidth == 0:
                    bandwidth = 1.0

                # Normalize distances and calculate tricube weights
                normalized_distances = distances[neighbor_indices] / bandwidth
                local_weights = self._tricube_weights(normalized_distances)

                # Apply robustness weights from previous iteration
                local_weights *= robustness_weights[neighbor_indices]

                # Fit local polynomial
                x_local = x[neighbor_indices]
                y_local = y_filled[neighbor_indices]

                if degree == 1:
                    # Linear regression
                    smoothed[i] = self._weighted_linear_fit(
                        x_local, y_local, local_weights, x[i]
                    )
                else:
                    # Quadratic regression
                    smoothed[i] = self._weighted_polynomial_fit(
                        x_local, y_local, local_weights, x[i], degree
                    )

            # Update robustness weights for next iteration (except last)
            if iteration < iterations - 1:
                residuals = np.where(nan_mask, 0, y_filled - smoothed)
                mad = np.median(np.abs(residuals[~nan_mask]))
                robustness_weights = np.where(
                    nan_mask,
                    1.0,
                    self._bisquare_weights(np.abs(residuals), mad),
                )

        return smoothed

    @staticmethod
    def _weighted_linear_fit(
        x: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
        x_eval: float,
    ) -> float:
        """Perform weighted linear regression and evaluate at x_eval.

        Args:
            x: Independent variable values.
            y: Dependent variable values.
            weights: Weights for each point.
            x_eval: Point at which to evaluate the fit.

        Returns:
            Fitted value at x_eval.
        """
        # Weighted means
        sum_w = np.sum(weights)
        if sum_w == 0:
            return np.mean(y)

        mean_x = np.sum(weights * x) / sum_w
        mean_y = np.sum(weights * y) / sum_w

        # Weighted covariance and variance
        x_centered = x - mean_x
        cov_xy = np.sum(weights * x_centered * (y - mean_y))
        var_x = np.sum(weights * x_centered**2)

        if var_x == 0:
            return mean_y

        # Slope and intercept
        slope = cov_xy / var_x
        intercept = mean_y - slope * mean_x

        return intercept + slope * x_eval

    @staticmethod
    def _weighted_polynomial_fit(
        x: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
        x_eval: float,
        degree: int,
    ) -> float:
        """Perform weighted polynomial regression and evaluate at x_eval.

        Args:
            x: Independent variable values.
            y: Dependent variable values.
            weights: Weights for each point.
            x_eval: Point at which to evaluate the fit.
            degree: Polynomial degree.

        Returns:
            Fitted value at x_eval.
        """
        # Build Vandermonde matrix
        vander_matrix = np.column_stack([x**i for i in range(degree + 1)])

        # Weight matrix
        weight_diag = np.diag(weights)

        # Weighted least squares: (X'WX)^-1 X'Wy
        try:
            xt_w = vander_matrix.T @ weight_diag
            xt_w_x = xt_w @ vander_matrix
            xt_w_y = xt_w @ y
            coeffs = np.linalg.solve(xt_w_x, xt_w_y)
        except np.linalg.LinAlgError:
            # Fall back to mean if matrix is singular
            return float(np.average(y, weights=weights))

        # Evaluate polynomial at x_eval
        x_powers = np.array([x_eval**i for i in range(degree + 1)])
        return float(np.dot(coeffs, x_powers))
