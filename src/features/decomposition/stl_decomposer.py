"""STL decomposition utilities.

This module provides classes for time series decomposition using STL
(Seasonal-Trend decomposition using Loess) and MSTL (Multiple Seasonal-Trend
decomposition using Loess).

STL is a versatile and robust method for decomposing time series that:
- Uses locally weighted regression (LOESS) for flexible decomposition
- Adapts to changing seasonal patterns over time
- Handles outliers robustly
- Works well with electricity load data patterns

Example:
    ```python
    from src.features.decomposition import STLDecomposer
    import pandas as pd

    # Create decomposer
    decomposer = STLDecomposer(
        seasonal_period=48,  # Daily pattern for semi-hourly data
        seasonal_smoother=7,
        robust=True
    )

    # Decompose series
    series = pd.Series([...], index=pd.date_range(...))
    decomposition = decomposer.decompose(series)

    # Access components
    trend = decomposition['trend']
    seasonal = decomposition['seasonal']
    residual = decomposition['residual']
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from statsmodels.tsa.seasonal import MSTL, STL

from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class STLDecomposer:
    """Seasonal-Trend decomposition using Loess.

    Decomposes time series into three components:
    - Trend: Long-term progression
    - Seasonal: Periodic patterns
    - Residual: Irregular fluctuations

    Uses locally weighted regression (LOESS) for flexible, non-parametric
    decomposition that adapts to changing seasonal patterns.

    Attributes:
        seasonal_period: Number of observations per seasonal cycle.
        seasonal_smoother: Smoothing parameter for seasonal component (odd).
        trend_smoother: Smoothing parameter for trend (None for auto).
        robust: Use robust fitting resistant to outliers.

    Example:
        ```python
        decomposer = STLDecomposer(
            seasonal_period=48,
            seasonal_smoother=7,
            robust=True
        )
        decomposition = decomposer.decompose(series)
        ```
    """

    def __init__(
        self,
        seasonal_period: int,
        seasonal_smoother: int = 7,
        trend_smoother: int | None = None,
        robust: bool = True,
    ):
        """Initialize STL decomposer.

        Args:
            seasonal_period: Number of observations per seasonal cycle.
            seasonal_smoother: Smoothing parameter for seasonal component (odd).
            trend_smoother: Smoothing parameter for trend (None for auto).
            robust: Use robust fitting resistant to outliers.

        Raises:
            ValueError: If seasonal_period < 2 or seasonal_smoother is even.
        """
        if seasonal_period < 2:
            msg = f"seasonal_period must be >= 2, got {seasonal_period}"
            raise ValueError(msg)

        if seasonal_smoother % 2 == 0:
            msg = f"seasonal_smoother must be odd, got {seasonal_smoother}"
            raise ValueError(msg)

        self.seasonal_period = seasonal_period
        self.seasonal_smoother = seasonal_smoother
        self.trend_smoother = trend_smoother
        self.robust = robust

    def decompose(self, series: pd.Series) -> pd.DataFrame:
        """Perform STL decomposition.

        Args:
            series: Time series to decompose.

        Returns:
            DataFrame with trend, seasonal, and residual components.

        Raises:
            ValueError: If series is too short (< 2 x seasonal_period).
        """
        logger.info(
            f"Starting STL decomposition with period={self.seasonal_period}"
        )

        # Check minimum length
        min_length = 2 * self.seasonal_period
        if len(series) < min_length:
            msg = (
                f"Series too short ({len(series)}). "
                f"Need >= {min_length} observations"
            )
            raise ValueError(msg)

        # Handle missing values
        series_clean = series.interpolate(method="linear", limit=24)
        series_clean = series_clean.ffill().bfill()

        # Perform STL decomposition
        stl = STL(
            series_clean,
            seasonal=self.seasonal_smoother,
            trend=self.trend_smoother,
            period=self.seasonal_period,
            robust=self.robust,
        )

        result = stl.fit()

        # Create output DataFrame
        decomposition = pd.DataFrame(
            {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "residual": result.resid,
            },
            index=series.index,
        )

        logger.info("STL decomposition complete")

        return decomposition


class MSTLDecomposer:
    """Multiple Seasonal-Trend decomposition using Loess.

    Extends STL to handle multiple seasonal periods simultaneously,
    ideal for electricity load data with daily, weekly, and yearly patterns.

    MSTL performs sequential STL decompositions for each seasonal period,
    allowing extraction of complex nested seasonal patterns.

    Attributes:
        seasonal_periods: Dictionary mapping names to periods.
        seasonal_smoother: Smoothing parameter for seasonal components.
        robust: Use robust fitting.

    Example:
        ```python
        decomposer = MSTLDecomposer(
            seasonal_periods={
                "daily": 48,
                "weekly": 336,
                "yearly": 17520
            },
            seasonal_smoother=7,
            robust=True
        )
        decomposition = decomposer.decompose(series)
        # Result has: trend, seasonal_daily, seasonal_weekly, seasonal_yearly, residual
        ```
    """

    def __init__(
        self,
        seasonal_periods: dict[str, int],
        seasonal_smoother: int = 7,
        robust: bool = True,
    ):
        """Initialize MSTL decomposer.

        Args:
            seasonal_periods: Dictionary mapping names to periods.
            seasonal_smoother: Smoothing parameter for seasonal components.
            robust: Use robust fitting.

        Raises:
            ValueError: If seasonal_periods is empty or contains invalid periods.
        """
        if not seasonal_periods:
            msg = "seasonal_periods cannot be empty"
            raise ValueError(msg)

        for name, period in seasonal_periods.items():
            if period < 2:
                msg = f"Period '{name}' must be >= 2, got {period}"
                raise ValueError(msg)

        if seasonal_smoother % 2 == 0:
            msg = f"seasonal_smoother must be odd, got {seasonal_smoother}"
            raise ValueError(msg)

        self.seasonal_periods = seasonal_periods
        self.seasonal_smoother = seasonal_smoother
        self.robust = robust

    def decompose(self, series: pd.Series) -> pd.DataFrame:
        """Perform MSTL decomposition.

        Args:
            series: Time series to decompose.

        Returns:
            DataFrame with trend, multiple seasonal components, and residual.

        Raises:
            ValueError: If series is too short for the largest period.
        """
        logger.info(
            f"Starting MSTL decomposition with periods={list(self.seasonal_periods.values())}"
        )

        # Check minimum length (need at least 2 cycles of longest period)
        max_period = max(self.seasonal_periods.values())
        min_length = 2 * max_period

        if len(series) < min_length:
            msg = (
                f"Series too short ({len(series)}). "
                f"Need >= {min_length} observations"
            )
            raise ValueError(msg)

        # Handle missing values
        series_clean = series.interpolate(method="linear", limit=24)
        series_clean = series_clean.ffill().bfill()

        # Perform MSTL decomposition
        # Note: MSTL doesn't support robust parameter directly
        # windows must be a list/tuple matching the number of periods
        n_periods = len(self.seasonal_periods)
        windows = [self.seasonal_smoother] * n_periods

        mstl = MSTL(
            series_clean,
            periods=list(self.seasonal_periods.values()),
            windows=windows,
        )

        result = mstl.fit()

        # Create output DataFrame with named seasonal components
        decomposition = pd.DataFrame(index=series.index)
        decomposition["trend"] = result.trend

        # Add each seasonal component with descriptive name
        period_to_name = {v: k for k, v in self.seasonal_periods.items()}

        # Handle single period case (returns Series) vs multiple periods (returns DataFrame)
        if len(self.seasonal_periods) == 1:
            # Single period: result.seasonal is a Series
            name = next(iter(period_to_name.values()))
            component_col = f"seasonal_{name}"
            decomposition[component_col] = result.seasonal
        else:
            # Multiple periods: result.seasonal is a DataFrame
            for i, period in enumerate(self.seasonal_periods.values()):
                name = period_to_name[period]
                component_col = f"seasonal_{name}"
                decomposition[component_col] = result.seasonal.iloc[:, i]

        decomposition["residual"] = result.resid

        logger.info("MSTL decomposition complete")

        return decomposition
