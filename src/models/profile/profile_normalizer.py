"""Profile normalization utilities for energy conservation.

This module provides the ProfileNormalizer class for ensuring that predicted
profile ratios satisfy energy conservation constraints. Profile ratios should
sum to 48 (one value per semi-hourly period) to maintain consistency with
daily demand mean predictions.

Example:
    ```python
    from src.models.profile.profile_normalizer import ProfileNormalizer
    import pandas as pd
    import numpy as np

    normalizer = ProfileNormalizer()

    # Create sample profile ratios (48 periods)
    profiles = pd.DataFrame({
        f"period_{i}": np.random.uniform(0.8, 1.2, 10)
        for i in range(48)
    })

    # Normalize to sum to 48
    normalized = normalizer.normalize_daily_profiles(profiles, target_sum=48.0)

    # Verify sum
    assert np.allclose(normalized.sum(axis=1), 48.0)
    ```
"""

from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProfileNormalizer:
    """Utility class for normalizing and smoothing profile ratios.

    This class provides methods to:
    1. Normalize profile ratios to ensure they sum to a target value (48)
    2. Denormalize profiles back to original scale
    3. Apply optional smoothing to reduce noise in profile predictions

    Profile normalization ensures energy conservation: the sum of 48 semi-hourly
    profile ratios should equal 48, so that when multiplied by the daily demand
    mean, we recover the correct total daily energy.
    """

    def __init__(self) -> None:
        """Initialize ProfileNormalizer."""
        logger.debug("Initialized ProfileNormalizer")

    def normalize_daily_profiles(
        self,
        profiles: pd.DataFrame,
        target_sum: float = 48.0,
        method: Literal["proportional", "additive"] = "proportional",
    ) -> pd.DataFrame:
        """Normalize profile ratios to sum to target value.

        This ensures energy conservation: sum of 48 semi-hourly ratios = 48.

        Args:
            profiles: DataFrame with profile ratios. Each row is a day,
                each column is a semi-hourly period (should have 48 columns).
            target_sum: Target sum for each row. Default 48.0 for 48 periods.
            method: Normalization method:
                - "proportional": Multiply all values by (target / current_sum)
                - "additive": Add (target - current_sum) / 48 to all values

        Returns:
            DataFrame with normalized profiles. Same shape as input.

        Raises:
            ValueError: If profiles DataFrame is invalid.

        Example:
            ```python
            # Proportional normalization (recommended)
            normalized = normalizer.normalize_daily_profiles(
                profiles, target_sum=48.0, method="proportional"
            )

            # Additive normalization
            normalized = normalizer.normalize_daily_profiles(
                profiles, target_sum=48.0, method="additive"
            )
            ```
        """
        if profiles.empty:
            msg = "Profiles DataFrame is empty"
            raise ValueError(msg)

        if profiles.shape[1] != 48:  # noqa: PLR2004
            logger.warning(
                "Profiles has %d columns, expected 48 for semi-hourly periods",
                profiles.shape[1],
            )

        # Calculate current sums
        current_sums = profiles.sum(axis=1)

        # Avoid division by zero
        valid_mask = current_sums > 0
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(
                "Found %d rows with sum <= 0. These will be set to uniform profile.",
                n_invalid,
            )

        # Normalize based on method
        normalized = profiles.copy()

        if method == "proportional":
            # Multiply by scaling factor
            scaling_factors = np.where(
                valid_mask,
                target_sum / current_sums,
                1.0,
            )
            normalized = profiles.mul(scaling_factors, axis=0)

            # For invalid rows, set uniform profile
            if not valid_mask.all():
                uniform_value = target_sum / profiles.shape[1]
                normalized.loc[~valid_mask, :] = uniform_value

        elif method == "additive":
            # Add adjustment to each value
            adjustments = (target_sum - current_sums) / profiles.shape[1]
            normalized = profiles.add(adjustments, axis=0)

            # For invalid rows, set uniform profile
            if not valid_mask.all():
                uniform_value = target_sum / profiles.shape[1]
                normalized.loc[~valid_mask, :] = uniform_value

        else:
            msg = f"Unknown normalization method: {method}"
            raise ValueError(msg)

        # Verify normalization
        final_sums = normalized.sum(axis=1)
        max_error = (final_sums - target_sum).abs().max()

        if max_error > 1e-6:
            logger.warning(
                "Normalization error: max deviation from target = %.2e",
                max_error,
            )

        logger.debug(
            "Normalized %d profiles using %s method (target_sum=%.1f)",
            len(profiles),
            method,
            target_sum,
        )

        return normalized

    def denormalize_profiles(
        self,
        normalized_profiles: pd.DataFrame,
        original_sums: pd.Series,
    ) -> pd.DataFrame:
        """Denormalize profiles back to original scale.

        Reverses the normalization by multiplying by (original_sum / 48).

        Args:
            normalized_profiles: DataFrame with normalized profiles (sum = 48).
            original_sums: Series with original sum for each row.

        Returns:
            DataFrame with denormalized profiles.

        Raises:
            ValueError: If shapes don't match.

        Example:
            ```python
            # Normalize
            original_sums = profiles.sum(axis=1)
            normalized = normalizer.normalize_daily_profiles(profiles)

            # Denormalize
            recovered = normalizer.denormalize_profiles(normalized, original_sums)

            # recovered should be very close to original profiles
            ```
        """
        if len(normalized_profiles) != len(original_sums):
            msg = (
                f"Shape mismatch: {len(normalized_profiles)} rows vs "
                f"{len(original_sums)} sums"
            )
            raise ValueError(msg)

        # Current sum should be ~48 for normalized profiles
        current_sums = normalized_profiles.sum(axis=1)

        # Scale back to original sums
        scaling_factors = original_sums / current_sums

        denormalized = normalized_profiles.mul(scaling_factors, axis=0)

        logger.debug(
            "Denormalized %d profiles back to original scale",
            len(normalized_profiles),
        )

        return denormalized

    def smooth_profiles(
        self,
        profiles: pd.DataFrame,
        sigma: float = 1.0,
        preserve_sum: bool = True,
    ) -> pd.DataFrame:
        """Apply Gaussian smoothing to reduce noise in profiles.

        This can help reduce overfitting and create more realistic profile shapes.

        Args:
            profiles: DataFrame with profile ratios. Each row is a day.
            sigma: Standard deviation for Gaussian kernel. Higher values =
                more smoothing. Typical range: 0.5 to 3.0.
            preserve_sum: If True, renormalize after smoothing to maintain
                the same sum per row.

        Returns:
            DataFrame with smoothed profiles. Same shape as input.

        Raises:
            ValueError: If sigma is invalid.

        Example:
            ```python
            # Light smoothing
            smoothed = normalizer.smooth_profiles(profiles, sigma=1.0)

            # Heavy smoothing
            smoothed = normalizer.smooth_profiles(profiles, sigma=2.5)
            ```
        """
        if sigma <= 0:
            msg = f"Sigma must be positive, got {sigma}"
            raise ValueError(msg)

        if profiles.empty:
            msg = "Profiles DataFrame is empty"
            raise ValueError(msg)

        # Store original sums
        if preserve_sum:
            original_sums = profiles.sum(axis=1)

        # Apply Gaussian filter to each row
        smoothed_values = np.apply_along_axis(
            lambda x: gaussian_filter1d(x, sigma=sigma, mode="wrap"),
            axis=1,
            arr=profiles.to_numpy(),
        )

        smoothed = pd.DataFrame(
            smoothed_values,
            index=profiles.index,
            columns=profiles.columns,
        )

        # Restore original sums if requested
        if preserve_sum:
            current_sums = smoothed.sum(axis=1)
            scaling_factors = original_sums / current_sums
            smoothed = smoothed.mul(scaling_factors, axis=0)

        logger.debug(
            "Smoothed %d profiles with sigma=%.2f (preserve_sum=%s)",
            len(profiles),
            sigma,
            preserve_sum,
        )

        return smoothed

    def validate_profiles(
        self,
        profiles: pd.DataFrame,
        min_ratio: float = 0.1,
        max_ratio: float = 5.0,
        expected_sum: float = 48.0,
        tolerance: float = 0.1,
    ) -> dict[str, bool | list[str]]:
        """Validate profile ratios for common issues.

        Args:
            profiles: DataFrame with profile ratios.
            min_ratio: Minimum allowable ratio value.
            max_ratio: Maximum allowable ratio value.
            expected_sum: Expected sum per row (default 48 for 48 periods).
            tolerance: Tolerance for sum validation (as fraction of expected_sum).

        Returns:
            Dictionary with validation results:
                - "is_valid": Boolean indicating if all checks passed
                - "issues": List of issue descriptions
                - "n_out_of_bounds": Number of values outside [min_ratio, max_ratio]
                - "n_wrong_sum": Number of rows with sum far from expected_sum

        Example:
            ```python
            validation = normalizer.validate_profiles(profiles)
            if not validation["is_valid"]:
                print("Issues found:", validation["issues"])
            ```
        """
        issues = []

        # Check for NaN/inf
        if profiles.isna().any().any():
            n_nan = profiles.isna().sum().sum()
            issues.append(f"Found {n_nan} NaN values")

        if np.isinf(profiles.to_numpy()).any():
            n_inf = np.isinf(profiles.to_numpy()).sum()
            issues.append(f"Found {n_inf} infinite values")

        # Check bounds
        n_below_min = (profiles < min_ratio).sum().sum()
        n_above_max = (profiles > max_ratio).sum().sum()
        n_out_of_bounds = n_below_min + n_above_max

        if n_out_of_bounds > 0:
            issues.append(
                f"Found {n_out_of_bounds} values outside [{min_ratio}, {max_ratio}] "
                f"({n_below_min} below min, {n_above_max} above max)"
            )

        # Check sums
        row_sums = profiles.sum(axis=1)
        sum_tolerance = expected_sum * tolerance
        wrong_sum_mask = (row_sums - expected_sum).abs() > sum_tolerance
        n_wrong_sum = wrong_sum_mask.sum()

        if n_wrong_sum > 0:
            issues.append(
                f"Found {n_wrong_sum} rows with sum far from {expected_sum} "
                f"(tolerance={sum_tolerance:.2f})"
            )

        is_valid = len(issues) == 0

        validation_result = {
            "is_valid": is_valid,
            "issues": issues,
            "n_out_of_bounds": n_out_of_bounds,
            "n_wrong_sum": n_wrong_sum,
        }

        if not is_valid:
            logger.warning("Profile validation failed: %s", "; ".join(issues))
        else:
            logger.debug("Profile validation passed")

        return validation_result
