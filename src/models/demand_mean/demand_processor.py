"""Demand data processing utilities for ARIMA models.

This module provides the DemandProcessor class for preprocessing demand time series
data before ARIMA/SARIMA model training. It handles missing value imputation,
outlier detection and treatment, and data quality validation.

Example:
    ```python
    from src.models.demand_mean.demand_processor import DemandProcessor
    import pandas as pd

    # Create processor
    processor = DemandProcessor()

    # Process demand series
    demand_series = pd.Series(
        [100.0, 102.0, np.nan, 105.0, 500.0, 103.0],
        index=pd.date_range("2024-01-01", periods=6, freq="D")
    )

    processed = processor.process_demand_series(
        demand_series,
        fill_missing=True,
        handle_outliers=True
    )
    ```
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DemandProcessor:
    """Processor for cleaning and preparing demand time series data.

    This class provides methods for handling common data quality issues in
    demand time series:
    - Missing values: Linear interpolation with forward/backward fill for edges
    - Outliers: Z-score detection with winsorization treatment
    - Validation: Checks for sufficient data and reasonable value ranges

    The processor is designed to be stateless and can be reused across
    multiple time series.

    Attributes:
        None (stateless processor)
    """

    def __init__(self) -> None:
        """Initialize the demand processor."""
        logger.debug("Initialized DemandProcessor")

    def process_demand_series(
        self,
        series: pd.Series,
        fill_missing: bool = True,
        handle_outliers: bool = True,
        outlier_z_threshold: float = 3.5,
    ) -> pd.Series:
        """Process demand time series with missing value and outlier handling.

        This is the main entry point for demand data preprocessing. It applies
        a sequence of transformations to clean the data.

        Args:
            series: Input demand time series with DatetimeIndex.
            fill_missing: Whether to fill missing values via interpolation.
            handle_outliers: Whether to detect and treat outliers.
            outlier_z_threshold: Z-score threshold for outlier detection.
                Values with |z-score| > threshold are considered outliers.

        Returns:
            Processed demand series with same index as input.

        Raises:
            ValueError: If series is empty or has no valid values.

        Example:
            ```python
            processor = DemandProcessor()
            clean_demand = processor.process_demand_series(
                raw_demand,
                fill_missing=True,
                handle_outliers=True,
                outlier_z_threshold=3.0
            )
            ```
        """
        if len(series) == 0:
            msg = "Input series is empty"
            raise ValueError(msg)

        if series.isna().all():
            msg = "Input series has no valid values"
            raise ValueError(msg)

        logger.debug(
            "Processing demand series: %d points, %d missing (%.1f%%)",
            len(series),
            series.isna().sum(),
            100 * series.isna().sum() / len(series),
        )

        processed = series.copy()

        # Step 1: Fill missing values
        if fill_missing and processed.isna().any():
            n_missing_before = processed.isna().sum()
            processed = self._fill_missing_values(processed)
            n_filled = n_missing_before - processed.isna().sum()
            logger.debug("Filled %d missing values via interpolation", n_filled)

        # Step 2: Detect and handle outliers
        if handle_outliers:
            outlier_mask = self._detect_outliers(processed, z_threshold=outlier_z_threshold)
            n_outliers = outlier_mask.sum()

            if n_outliers > 0:
                logger.debug(
                    "Detected %d outliers (%.1f%%) with z-threshold=%.1f",
                    n_outliers,
                    100 * n_outliers / len(processed),
                    outlier_z_threshold,
                )
                processed = self._handle_outliers(processed, outlier_mask)

        # Step 3: Final validation
        if processed.isna().any():
            logger.warning(
                "Processed series still has %d missing values",
                processed.isna().sum(),
            )

        logger.info(
            "Demand series processed: %d points, min=%.2f, max=%.2f, mean=%.2f",
            len(processed),
            processed.min(),
            processed.max(),
            processed.mean(),
        )

        return processed

    def _fill_missing_values(self, series: pd.Series) -> pd.Series:
        """Fill missing values using linear interpolation.

        Uses pandas interpolate with linear method, then forward fill and
        backward fill to handle missing values at the edges.

        Args:
            series: Input series with potential missing values.

        Returns:
            Series with missing values filled.

        Example:
            ```python
            # Series: [10.0, NaN, 20.0, NaN, NaN, 30.0]
            # Result: [10.0, 15.0, 20.0, 23.3, 26.7, 30.0]
            filled = processor._fill_missing_values(series)
            ```
        """
        if not series.isna().any():
            return series

        filled = series.copy()

        # Linear interpolation for interior missing values
        filled = filled.interpolate(
            method="linear",
            limit_direction="both",
            limit_area=None,
        )

        # Forward fill for any remaining missing at the start
        filled = filled.ffill()

        # Backward fill for any remaining missing at the end
        return filled.bfill()

    def _detect_outliers(
        self,
        series: pd.Series,
        z_threshold: float = 3.5,
    ) -> pd.Series:
        """Detect outliers using Modified Z-score method (robust to outliers).

        Uses median and MAD (Median Absolute Deviation) instead of mean and
        standard deviation to be more robust against extreme outliers.

        The Modified Z-score is: 0.6745 * (x - median) / MAD
        where 0.6745 is a constant that makes MAD consistent with std for normal data.

        Args:
            series: Input series to check for outliers.
            z_threshold: Modified Z-score threshold. Common values: 2.5-3.5.

        Returns:
            Boolean Series with True for outliers, False otherwise.

        Example:
            ```python
            # Series: [10, 12, 11, 500, 13, 12]
            # Outlier mask: [False, False, False, True, False, False]
            outliers = processor._detect_outliers(series, z_threshold=3.0)
            ```
        """
        # Remove NaN before computing scores
        valid_values = series.dropna()
        min_data_points = 3

        if len(valid_values) < min_data_points:
            # Not enough data for meaningful outlier detection
            return pd.Series(False, index=series.index)

        # Compute Modified Z-scores using median and MAD (more robust)
        median = valid_values.median()
        mad = np.abs(valid_values - median).median()

        # Avoid division by zero
        if mad == 0:
            # All values are equal or very close, no outliers
            return pd.Series(False, index=series.index)

        # Modified Z-score: 0.6745 is the constant for normal distribution consistency
        modified_z_scores = 0.6745 * np.abs(valid_values - median) / mad

        # Create boolean mask
        outlier_mask = pd.Series(False, index=series.index)
        outlier_mask.loc[valid_values.index] = modified_z_scores > z_threshold

        return outlier_mask

    def _handle_outliers(
        self,
        series: pd.Series,
        outlier_mask: pd.Series,
    ) -> pd.Series:
        """Handle outliers using winsorization.

        Winsorization replaces outliers with the nearest non-outlier value
        within a reasonable range (percentile-based clipping).

        Args:
            series: Input series with outliers.
            outlier_mask: Boolean mask indicating outlier positions.

        Returns:
            Series with outliers replaced by clipped values.

        Example:
            ```python
            # Series: [10, 12, 11, 500, 13, 12]
            # Outlier mask: [False, False, False, True, False, False]
            # Result: [10, 12, 11, 13, 13, 12]  # 500 clipped to 95th percentile
            treated = processor._handle_outliers(series, outlier_mask)
            ```
        """
        if not outlier_mask.any():
            return series

        treated = series.copy()
        valid_values = series[~outlier_mask].dropna()

        if len(valid_values) == 0:
            logger.warning("All values are outliers, cannot winsorize")
            return treated

        # Compute percentile-based bounds (5th and 95th percentiles)
        lower_bound = valid_values.quantile(0.05)
        upper_bound = valid_values.quantile(0.95)

        # Replace outliers with clipped values
        treated[outlier_mask] = treated[outlier_mask].clip(
            lower=lower_bound,
            upper=upper_bound,
        )

        logger.debug(
            "Winsorized %d outliers to range [%.2f, %.2f]",
            outlier_mask.sum(),
            lower_bound,
            upper_bound,
        )

        return treated

    def validate_for_arima(
        self,
        series: pd.Series,
        min_observations: int = 14,
    ) -> dict[str, Any]:
        """Validate demand series for ARIMA model training.

        Checks various data quality criteria:
        - Sufficient number of observations
        - No missing values
        - No constant values
        - Positive values (demand cannot be negative)

        Args:
            series: Demand series to validate.
            min_observations: Minimum required number of observations.

        Returns:
            Dictionary with validation results:
                - "is_valid": bool, overall validation status
                - "n_observations": int, number of observations
                - "n_missing": int, number of missing values
                - "has_variance": bool, whether series has variation
                - "all_positive": bool, whether all values are positive
                - "issues": list of str, validation issues found

        Example:
            ```python
            processor = DemandProcessor()
            validation = processor.validate_for_arima(demand_series, min_observations=14)

            if not validation["is_valid"]:
                print(f"Validation failed: {validation['issues']}")
            ```
        """
        issues = []

        # Check length
        n_observations = len(series)
        if n_observations < min_observations:
            issues.append(
                f"Insufficient data: {n_observations} observations "
                f"(minimum {min_observations} required)"
            )

        # Check for missing values
        n_missing = series.isna().sum()
        if n_missing > 0:
            issues.append(f"Contains {n_missing} missing values")

        # Check for variance
        valid_values = series.dropna()
        has_variance = len(valid_values) > 1 and valid_values.std() > 0
        if not has_variance:
            issues.append("Series has no variance (constant or single value)")

        # Check for positive values
        all_positive = (valid_values >= 0).all()
        if not all_positive:
            n_negative = (valid_values < 0).sum()
            issues.append(f"Contains {n_negative} negative values (demand must be >= 0)")

        is_valid = len(issues) == 0

        result = {
            "is_valid": is_valid,
            "n_observations": n_observations,
            "n_missing": n_missing,
            "has_variance": has_variance,
            "all_positive": all_positive,
            "issues": issues,
        }

        if not is_valid:
            logger.warning("ARIMA validation failed: %s", issues)
        else:
            logger.debug("ARIMA validation passed: %d observations", n_observations)

        return result
