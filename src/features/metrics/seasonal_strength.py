"""Seasonal strength metrics for time series decomposition.

This module provides utilities for calculating seasonal and trend strength
metrics from decomposed time series components. These metrics quantify how
much of the variance in the data is explained by seasonal or trend patterns.

Seasonal strength is a useful metric for:
- Determining if seasonal patterns are significant
- Deciding whether to include seasonal components in models
- Comparing different seasonal patterns (daily vs weekly vs yearly)
- Evaluating decomposition quality

Example:
    ```python
    from src.features.metrics import SeasonalStrengthCalculator
    import pandas as pd

    calculator = SeasonalStrengthCalculator()

    # Calculate seasonal strength
    seasonal = pd.Series([...])  # Seasonal component
    residual = pd.Series([...])   # Residual component
    strength = calculator.calculate_seasonal_strength(seasonal, residual)

    # Classify strength
    classification = calculator.classify_strength(strength)
    # Returns: "strong", "moderate", or "weak"
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from typing import Literal

logger = get_logger(__name__)


class SeasonalStrengthCalculator:
    """Calculate strength metrics for seasonal and trend components.

    Seasonal strength measures how much of the variance in the
    detrended series is explained by the seasonal component.

    Formula: Fs = max(0, 1 - Var(residual) / Var(seasonal + residual))

    Interpretation:
    - Fs > 0.6: Strong seasonality (seasonal patterns dominate)
    - 0.3 < Fs < 0.6: Moderate seasonality (seasonal patterns present)
    - Fs < 0.3: Weak seasonality (seasonal patterns minimal)

    Trend strength uses a similar formula:
    Ft = max(0, 1 - Var(residual) / Var(trend + residual))

    Example:
        ```python
        calculator = SeasonalStrengthCalculator()

        # Strong seasonality example
        seasonal = pd.Series(100 * np.sin(2 * np.pi * np.arange(100) / 10))
        residual = pd.Series(np.random.randn(100) * 5)
        strength = calculator.calculate_seasonal_strength(seasonal, residual)
        # Returns: ~0.95 (strong)
        ```
    """

    @staticmethod
    def calculate_seasonal_strength(
        seasonal: pd.Series,
        residual: pd.Series,
    ) -> float:
        """Calculate seasonal strength.

        Args:
            seasonal: Seasonal component from decomposition.
            residual: Residual component from decomposition.

        Returns:
            Seasonal strength in [0, 1] range.
                - 1.0 indicates perfect seasonality (all variance explained)
                - 0.0 indicates no seasonality (no variance explained)

        Note:
            If the detrended variance is zero, returns 0.0 to avoid
            division by zero errors.
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

        return float(strength)

    @staticmethod
    def calculate_trend_strength(
        trend: pd.Series,
        residual: pd.Series,
    ) -> float:
        """Calculate trend strength.

        Args:
            trend: Trend component from decomposition.
            residual: Residual component from decomposition.

        Returns:
            Trend strength in [0, 1] range.
                - 1.0 indicates strong trend (all variance explained)
                - 0.0 indicates no trend (no variance explained)

        Note:
            If the deseasonalized variance is zero, returns 0.0 to avoid
            division by zero errors.
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

        return float(strength)

    @staticmethod
    def classify_strength(strength: float) -> Literal["strong", "moderate", "weak"]:
        """Classify strength level.

        Classifies a strength value into one of three categories based on
        commonly used thresholds in time series analysis.

        Args:
            strength: Strength value in [0, 1] range.

        Returns:
            Classification as "strong", "moderate", or "weak".

        Example:
            ```python
            calculator = SeasonalStrengthCalculator()
            calculator.classify_strength(0.85)  # Returns: "strong"
            calculator.classify_strength(0.45)  # Returns: "moderate"
            calculator.classify_strength(0.15)  # Returns: "weak"
            ```
        """
        if strength > 0.6:
            return "strong"
        if strength > 0.3:
            return "moderate"
        return "weak"
