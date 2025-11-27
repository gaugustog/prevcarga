"""Profile combiner for hierarchical forecasting models.

This module provides the ProfileCombiner class that combines demand mean forecasts
with profile ratios to produce semi-hourly load predictions. It handles edge cases
such as missing profiles and ensures reasonable bounds on profile ratios.

Example:
    ```python
    from src.models.hierarchical.profile_combiner import ProfileCombiner
    import pandas as pd

    combiner = ProfileCombiner()

    # Demand mean predictions (daily average load)
    demand_mean = pd.Series([1000.0, 1100.0, 1200.0])

    # Profile ratios for each semi-hourly period (48 periods/day)
    profiles = {
        0: pd.Series([0.8, 0.85, 0.9]),   # Period 0 (00:00-00:30)
        1: pd.Series([0.75, 0.8, 0.85]),  # Period 1 (00:30-01:00)
        # ... up to period 47
    }

    # Combine to get semi-hourly predictions
    load = combiner.combine(demand_mean, profiles)
    ```
"""

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants for Brazilian SIN semi-hourly resolution
PERIODS_PER_DAY = 48  # Number of semi-hourly periods in a day


class ProfileCombiner:
    """Combines demand mean forecasts with profile ratios.

    This class implements the combination logic for two-stage hierarchical models:
        load = demand_mean * profile_ratio

    It handles missing profiles gracefully by defaulting to 1.0 (flat profile),
    clips profile ratios to reasonable bounds (0.1 to 5.0), and ensures
    non-negative outputs.

    Attributes:
        min_profile_ratio: Minimum allowed profile ratio (default: 0.1).
        max_profile_ratio: Maximum allowed profile ratio (default: 5.0).
    """

    def __init__(
        self,
        min_profile_ratio: float = 0.1,
        max_profile_ratio: float = 5.0,
    ) -> None:
        """Initialize ProfileCombiner with ratio bounds.

        Args:
            min_profile_ratio: Minimum allowed profile ratio. Values below this
                               will be clipped.
            max_profile_ratio: Maximum allowed profile ratio. Values above this
                               will be clipped.

        Raises:
            ValueError: If min_profile_ratio >= max_profile_ratio or if either
                       bound is non-positive.
        """
        if min_profile_ratio >= max_profile_ratio:
            msg = (
                f"min_profile_ratio ({min_profile_ratio}) must be less than "
                f"max_profile_ratio ({max_profile_ratio})"
            )
            raise ValueError(msg)

        if min_profile_ratio <= 0:
            msg = f"min_profile_ratio must be positive, got {min_profile_ratio}"
            raise ValueError(msg)

        if max_profile_ratio <= 0:
            msg = f"max_profile_ratio must be positive, got {max_profile_ratio}"
            raise ValueError(msg)

        self.min_profile_ratio = min_profile_ratio
        self.max_profile_ratio = max_profile_ratio

        logger.debug(
            "Initialized ProfileCombiner with ratio bounds [%.2f, %.2f]",
            min_profile_ratio,
            max_profile_ratio,
        )

    def combine(
        self,
        demand_mean: pd.Series,
        profiles: dict[int, pd.Series],
    ) -> pd.DataFrame:
        """Combine demand mean with profile ratios to produce load predictions.

        This method multiplies the demand mean by the profile ratio for each
        semi-hourly period (0-47). Missing profiles are handled by using a
        default ratio of 1.0 (flat profile).

        Args:
            demand_mean: Series of demand mean predictions (daily average load).
                        Index should match profile series indices.
            profiles: Dictionary mapping period (0-47) to profile ratio series.
                     Each series should have the same index as demand_mean.

        Returns:
            DataFrame with semi-hourly load predictions. Shape (n_samples, 48).
            Columns are named "period_0", "period_1", ..., "period_47".

        Raises:
            ValueError: If demand_mean is empty or contains NaN/infinite values.
            ValueError: If profile indices don't match demand_mean index.
        """
        # Validate demand_mean
        if demand_mean.empty:
            msg = "demand_mean cannot be empty"
            raise ValueError(msg)

        if demand_mean.isna().any():
            n_nan = demand_mean.isna().sum()
            msg = f"demand_mean contains {n_nan} NaN values"
            raise ValueError(msg)

        if not np.isfinite(demand_mean).all():
            n_inf = (~np.isfinite(demand_mean)).sum()
            msg = f"demand_mean contains {n_inf} infinite values"
            raise ValueError(msg)

        n_samples = len(demand_mean)
        logger.debug("Combining demand_mean with profiles for %d samples", n_samples)

        # Initialize result DataFrame with 48 periods
        result_data = {}

        # Process each semi-hourly period (0-47)
        for period in range(48):
            if period in profiles:
                profile_series = profiles[period]

                # Validate profile index matches demand_mean index
                if not profile_series.index.equals(demand_mean.index):
                    msg = (
                        f"Profile series for period {period} has mismatched index. "
                        f"Expected {len(demand_mean)} rows, got {len(profile_series)}"
                    )
                    raise ValueError(msg)

                # Clip profile ratios to reasonable bounds
                clipped_profile = profile_series.clip(
                    lower=self.min_profile_ratio,
                    upper=self.max_profile_ratio,
                )

                # Count how many values were clipped
                n_clipped_low = (profile_series < self.min_profile_ratio).sum()
                n_clipped_high = (profile_series > self.max_profile_ratio).sum()

                if n_clipped_low > 0:
                    logger.debug(
                        "Period %d: Clipped %d values below %.2f",
                        period,
                        n_clipped_low,
                        self.min_profile_ratio,
                    )

                if n_clipped_high > 0:
                    logger.debug(
                        "Period %d: Clipped %d values above %.2f",
                        period,
                        n_clipped_high,
                        self.max_profile_ratio,
                    )

                # Combine: load = demand_mean * profile_ratio
                load = demand_mean * clipped_profile

            else:
                # Missing profile: use flat profile (ratio = 1.0)
                logger.debug(
                    "Period %d: Profile not found, using default ratio of 1.0",
                    period,
                )
                load = demand_mean * 1.0

            # Ensure non-negative outputs
            load = load.clip(lower=0.0)

            result_data[f"period_{period}"] = load

        result_df = pd.DataFrame(result_data, index=demand_mean.index)

        logger.info(
            "Combined %d demand_mean predictions with %d profiles to produce "
            "semi-hourly load predictions",
            n_samples,
            len(profiles),
        )

        return result_df

    def validate_energy_conservation(
        self,
        demand_mean: pd.Series,
        combined_load: pd.DataFrame,
        tolerance: float = 0.1,
    ) -> dict[str, Any]:
        """Validate that daily energy is conserved in the combination.

        The expected daily energy is demand_mean * 48 (48 semi-hourly periods).
        The actual daily energy is the sum of all 48 semi-hourly loads.
        This method checks that the relative error is within tolerance.

        Args:
            demand_mean: Series of demand mean predictions (daily average load).
            combined_load: DataFrame with semi-hourly load predictions (48 columns).
            tolerance: Maximum allowed relative error (default: 0.1 = 10%).

        Returns:
            Dictionary with validation results:
                - "is_valid": bool, whether all errors are within tolerance
                - "mean_error": float, mean relative error across samples
                - "max_error": float, maximum relative error
                - "n_violations": int, number of samples exceeding tolerance
                - "tolerance": float, the tolerance threshold used

        Raises:
            ValueError: If shapes don't match or combined_load doesn't have 48 columns.
        """
        if len(demand_mean) != len(combined_load):
            msg = (
                f"Shape mismatch: demand_mean has {len(demand_mean)} rows, "
                f"combined_load has {len(combined_load)} rows"
            )
            raise ValueError(msg)

        if combined_load.shape[1] != PERIODS_PER_DAY:
            msg = f"combined_load must have 48 columns, got {combined_load.shape[1]}"
            raise ValueError(msg)

        # Expected daily energy: demand_mean * PERIODS_PER_DAY
        expected_energy = demand_mean * PERIODS_PER_DAY

        # Actual daily energy: sum of all 48 semi-hourly loads
        actual_energy = combined_load.sum(axis=1)

        # Relative error: |actual - expected| / expected
        relative_error = np.abs(actual_energy - expected_energy) / expected_energy

        # Statistics
        mean_error = relative_error.mean()
        max_error = relative_error.max()
        n_violations = (relative_error > tolerance).sum()
        is_valid = n_violations == 0

        if not is_valid:
            logger.warning(
                "Energy conservation validation failed: %d/%d samples exceed "
                "%.1f%% tolerance (mean error: %.2f%%, max error: %.2f%%)",
                n_violations,
                len(demand_mean),
                tolerance * 100,
                mean_error * 100,
                max_error * 100,
            )
        else:
            logger.info(
                "Energy conservation validated: all samples within %.1f%% tolerance "
                "(mean error: %.2f%%, max error: %.2f%%)",
                tolerance * 100,
                mean_error * 100,
                max_error * 100,
            )

        return {
            "is_valid": is_valid,
            "mean_error": float(mean_error),
            "max_error": float(max_error),
            "n_violations": int(n_violations),
            "tolerance": tolerance,
        }

    def __repr__(self) -> str:
        """Return string representation of ProfileCombiner.

        Returns:
            String representation with ratio bounds.
        """
        return (
            f"ProfileCombiner("
            f"min_ratio={self.min_profile_ratio}, "
            f"max_ratio={self.max_profile_ratio})"
        )
