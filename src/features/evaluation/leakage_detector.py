"""Data leakage detection utilities for time series forecasting.

This module provides the DataLeakageDetector class for identifying
temporal data leakage in feature sets. Temporal leakage occurs when
features that would not be available at forecast time are included
in the training data, leading to overly optimistic model performance
that doesn't translate to production.

Example:
    ```python
    from src.features.evaluation import DataLeakageDetector
    import pandas as pd

    # Initialize detector for semi-hourly data
    detector = DataLeakageDetector(periods_per_day=48)

    # Check which features are safe for horizon D+1
    features = ["hour", "carga_lag_24", "carga_lag_48", "future_value"]
    safe_features = detector.get_horizon_safe_features(
        features=features,
        horizon=1,
        target_column="carga"
    )
    # Returns: ["hour", "carga_lag_48"]
    # Excludes: "carga_lag_24" (insufficient lag), "future_value" (leakage pattern)

    # Comprehensive leakage check
    leakage_report = detector.check_temporal_leakage(
        df=data_df,
        feature_cols=features,
        target_col="carga",
        horizon=1
    )
    ```
"""

import re
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLeakageDetector:
    """Detector for temporal data leakage in feature sets.

    Identifies features that would not be available at forecast time,
    preventing future information from leaking into the model. This is
    critical for multi-horizon forecasting where different horizons
    have different feature availability constraints.

    The detector checks for:
        1. Features with suspicious naming patterns (e.g., "future", "forward", "actual")
        2. Lag features with insufficient lag for the forecast horizon
        3. Target variable and its direct derivatives

    Attributes:
        periods_per_day: Number of periods per day (48 for semi-hourly data).
        leakage_patterns: List of regex patterns that indicate potential leakage.

    Example:
        >>> detector = DataLeakageDetector(periods_per_day=48)
        >>> features = ["hour", "carga_lag_24", "future_temp"]
        >>> safe = detector.get_horizon_safe_features(features, horizon=2, target_column="carga")
        >>> # "carga_lag_24" filtered (needs lag >= 96 for D+2)
        >>> # "future_temp" filtered (matches leakage pattern)
    """

    def __init__(self, periods_per_day: int = 48) -> None:
        """Initialize detector with data frequency information.

        Args:
            periods_per_day: Number of periods per day (48 for semi-hourly,
                24 for hourly, 1 for daily data).
        """
        self.periods_per_day = periods_per_day
        self.leakage_patterns = [
            r".*future.*",
            r".*forward.*",
            r".*actual.*",  # Avoid actual values
            r".*target.*",  # Avoid target derivatives
        ]

    def get_horizon_safe_features(
        self,
        features: list[str],
        horizon: int,
        target_column: str = "carga",
    ) -> list[str]:
        """Get features that are safe for given forecast horizon.

        Filters the feature list to include only features that would be
        available at forecast time for the specified horizon.

        Args:
            features: List of feature names to check.
            horizon: Forecast horizon in days (0 for D+0, 1 for D+1, etc.).
            target_column: Name of target column to exclude.

        Returns:
            List of safe feature names that can be used for the horizon.

        Example:
            >>> detector = DataLeakageDetector(periods_per_day=48)
            >>> features = ["hour", "day_of_week", "carga_lag_24", "carga_lag_96"]
            >>> safe = detector.get_horizon_safe_features(features, horizon=2)
            >>> # For D+2 (2 days ahead), only carga_lag_96 is safe
            >>> # carga_lag_24 requires at least 96 periods of lag
        """
        safe_features = []

        for feature in features:
            # Skip target column
            if feature == target_column:
                logger.debug("Rejecting '%s': is target column", feature)
                continue

            # Check leakage patterns
            if self._matches_leakage_pattern(feature):
                logger.debug("Rejecting '%s': matches leakage pattern", feature)
                continue

            # Validate lag features
            if "lag" in feature.lower():
                if not self._is_lag_safe_for_horizon(feature, horizon):
                    logger.debug(
                        "Rejecting '%s': insufficient lag for horizon %d",
                        feature,
                        horizon,
                    )
                    continue

            safe_features.append(feature)

        logger.info(
            "Horizon %d: %d/%d features safe",
            horizon,
            len(safe_features),
            len(features),
        )

        return safe_features

    def _matches_leakage_pattern(self, feature_name: str) -> bool:
        """Check if feature name matches known leakage patterns.

        Args:
            feature_name: Name of the feature to check.

        Returns:
            True if feature name matches a leakage pattern, False otherwise.
        """
        feature_lower = feature_name.lower()

        for pattern in self.leakage_patterns:
            if re.match(pattern, feature_lower):
                return True

        return False

    def _is_lag_safe_for_horizon(self, feature_name: str, horizon: int) -> bool:
        """Check if lag feature has sufficient lag for horizon.

        For a forecast horizon of D+N, lag features must have a lag of
        at least N × periods_per_day to ensure they don't use information
        from the future.

        Args:
            feature_name: Feature name (e.g., "carga_lag_24", "temp_lag_48").
            horizon: Forecast horizon in days.

        Returns:
            True if lag is sufficient for the horizon, False otherwise.

        Example:
            >>> detector = DataLeakageDetector(periods_per_day=48)
            >>> detector._is_lag_safe_for_horizon("carga_lag_48", horizon=1)
            True  # 48 >= 1 * 48
            >>> detector._is_lag_safe_for_horizon("carga_lag_24", horizon=2)
            False  # 24 < 2 * 48 = 96
        """
        # Extract lag value from feature name
        lag_match = re.search(r"lag[_-](\d+)", feature_name, re.IGNORECASE)

        if not lag_match:
            # Not a standard lag feature, allow it
            return True

        lag_value = int(lag_match.group(1))

        # Required lag for horizon (in periods)
        required_lag = horizon * self.periods_per_day

        # Lag must be at least as large as horizon requirement
        return lag_value >= required_lag

    def check_temporal_leakage(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        horizon: int,
    ) -> dict[str, Any]:
        """Comprehensive temporal leakage check for a dataset.

        Performs a full analysis of potential temporal leakage in the
        provided feature set for a specific forecast horizon.

        Args:
            df: DataFrame with features and target.
            feature_cols: List of feature column names to check.
            target_col: Target column name.
            horizon: Forecast horizon in days.

        Returns:
            Dictionary with leakage analysis results containing:
                - has_leakage: Boolean indicating if leakage was detected
                - unsafe_features: List of features that may cause leakage
                - warnings: List of warning messages

        Example:
            >>> detector = DataLeakageDetector(periods_per_day=48)
            >>> result = detector.check_temporal_leakage(
            ...     df=data_df,
            ...     feature_cols=["hour", "carga_lag_24", "future_value"],
            ...     target_col="carga",
            ...     horizon=2
            ... )
            >>> if result['has_leakage']:
            ...     print(f"Unsafe features: {result['unsafe_features']}")
        """
        results = {
            "has_leakage": False,
            "unsafe_features": [],
            "warnings": [],
        }

        # Get safe features
        safe_features = self.get_horizon_safe_features(
            feature_cols,
            horizon,
            target_col,
        )

        # Identify unsafe features
        unsafe = set(feature_cols) - set(safe_features)

        if unsafe:
            results["has_leakage"] = True
            results["unsafe_features"] = list(unsafe)
            results["warnings"].append(
                f"Found {len(unsafe)} potentially unsafe features for horizon {horizon}"
            )
            logger.warning(
                "Temporal leakage detected for horizon %d: %s",
                horizon,
                ", ".join(list(unsafe)[:5]),  # Show first 5 unsafe features
            )

        return results
