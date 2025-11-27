"""Tests for DataLeakageDetector class.

This module contains comprehensive tests for the temporal data leakage
detection functionality used in multi-horizon forecasting feature selection.
"""

import pandas as pd
import pytest

from src.features.evaluation.leakage_detector import DataLeakageDetector


class TestDataLeakageDetector:
    """Test suite for DataLeakageDetector."""

    @pytest.fixture
    def detector(self) -> DataLeakageDetector:
        """Create detector instance with semi-hourly frequency."""
        return DataLeakageDetector(periods_per_day=48)

    @pytest.fixture
    def detector_hourly(self) -> DataLeakageDetector:
        """Create detector instance with hourly frequency."""
        return DataLeakageDetector(periods_per_day=24)

    def test_initialization(self, detector: DataLeakageDetector) -> None:
        """Test detector initialization."""
        assert detector.periods_per_day == 48
        assert len(detector.leakage_patterns) > 0

    def test_matches_leakage_pattern_future(self, detector: DataLeakageDetector) -> None:
        """Test detection of 'future' pattern in feature names."""
        assert detector._matches_leakage_pattern("future_value")
        assert detector._matches_leakage_pattern("feature_future_1")
        assert detector._matches_leakage_pattern("FUTURE_TEMP")

    def test_matches_leakage_pattern_forward(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test detection of 'forward' pattern in feature names."""
        assert detector._matches_leakage_pattern("forward_looking")
        assert detector._matches_leakage_pattern("temp_forward")

    def test_matches_leakage_pattern_actual(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test detection of 'actual' pattern in feature names."""
        assert detector._matches_leakage_pattern("actual_value")
        assert detector._matches_leakage_pattern("temperature_actual")

    def test_matches_leakage_pattern_target(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test detection of 'target' pattern in feature names."""
        assert detector._matches_leakage_pattern("target_derived")
        assert detector._matches_leakage_pattern("feature_target")

    def test_no_false_positive_leakage_patterns(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test that safe feature names don't match leakage patterns."""
        safe_features = [
            "hour",
            "day_of_week",
            "temperature",
            "carga_lag_24",
            "rolling_mean_7d",
        ]
        for feature in safe_features:
            assert not detector._matches_leakage_pattern(feature)

    def test_lag_safe_for_horizon_sufficient_lag(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test that sufficient lag passes validation."""
        # For semi-hourly data (48 periods/day)
        # Horizon 0: requires lag >= 0
        assert detector._is_lag_safe_for_horizon("carga_lag_1", horizon=0)
        assert detector._is_lag_safe_for_horizon("carga_lag_24", horizon=0)

        # Horizon 1: requires lag >= 48
        assert detector._is_lag_safe_for_horizon("carga_lag_48", horizon=1)
        assert detector._is_lag_safe_for_horizon("carga_lag_96", horizon=1)

        # Horizon 2: requires lag >= 96
        assert detector._is_lag_safe_for_horizon("carga_lag_96", horizon=2)
        assert detector._is_lag_safe_for_horizon("carga_lag_144", horizon=2)

    def test_lag_safe_for_horizon_insufficient_lag(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test that insufficient lag fails validation."""
        # For semi-hourly data (48 periods/day)
        # Horizon 1: requires lag >= 48
        assert not detector._is_lag_safe_for_horizon("carga_lag_24", horizon=1)
        assert not detector._is_lag_safe_for_horizon("carga_lag_1", horizon=1)

        # Horizon 2: requires lag >= 96
        assert not detector._is_lag_safe_for_horizon("carga_lag_48", horizon=2)
        assert not detector._is_lag_safe_for_horizon("carga_lag_72", horizon=2)

    def test_lag_safe_for_horizon_hourly_data(
        self, detector_hourly: DataLeakageDetector
    ) -> None:
        """Test lag validation with hourly data frequency."""
        # For hourly data (24 periods/day)
        # Horizon 1: requires lag >= 24
        assert detector_hourly._is_lag_safe_for_horizon("carga_lag_24", horizon=1)
        assert not detector_hourly._is_lag_safe_for_horizon("carga_lag_12", horizon=1)

        # Horizon 2: requires lag >= 48
        assert detector_hourly._is_lag_safe_for_horizon("carga_lag_48", horizon=2)
        assert not detector_hourly._is_lag_safe_for_horizon("carga_lag_24", horizon=2)

    def test_lag_safe_for_horizon_non_standard_naming(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test that non-standard lag features are allowed."""
        # Features without standard lag pattern should pass
        assert detector._is_lag_safe_for_horizon("temperature", horizon=5)
        assert detector._is_lag_safe_for_horizon("hour_of_day", horizon=5)
        assert detector._is_lag_safe_for_horizon("rolling_mean", horizon=5)

    def test_lag_safe_for_horizon_different_separators(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test lag detection with different naming conventions."""
        # Test underscore separator
        assert detector._is_lag_safe_for_horizon("carga_lag_48", horizon=1)
        assert not detector._is_lag_safe_for_horizon("carga_lag_24", horizon=1)

        # Test hyphen separator
        assert detector._is_lag_safe_for_horizon("carga-lag-48", horizon=1)
        assert not detector._is_lag_safe_for_horizon("carga-lag-24", horizon=1)

    def test_get_horizon_safe_features_horizon_zero(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test feature filtering for horizon D+0."""
        features = [
            "hour",
            "day_of_week",
            "carga_lag_1",
            "carga_lag_24",
            "future_value",
            "temperature",
        ]

        safe = detector.get_horizon_safe_features(
            features, horizon=0, target_column="carga"
        )

        # Should include temporal and lag features
        assert "hour" in safe
        assert "day_of_week" in safe
        assert "carga_lag_1" in safe
        assert "carga_lag_24" in safe
        assert "temperature" in safe

        # Should exclude future-looking features
        assert "future_value" not in safe

    def test_get_horizon_safe_features_horizon_one(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test feature filtering for horizon D+1."""
        features = [
            "hour",
            "carga_lag_24",
            "carga_lag_48",
            "carga_lag_96",
            "future_value",
        ]

        safe = detector.get_horizon_safe_features(
            features, horizon=1, target_column="carga"
        )

        # Hour should be safe (temporal feature)
        assert "hour" in safe

        # Only lag >= 48 should be safe
        assert "carga_lag_24" not in safe
        assert "carga_lag_48" in safe
        assert "carga_lag_96" in safe

        # Future-looking features should be excluded
        assert "future_value" not in safe

    def test_get_horizon_safe_features_horizon_two(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test feature filtering for horizon D+2."""
        features = [
            "hour",
            "carga_lag_48",
            "carga_lag_96",
            "carga_lag_144",
            "forward_temp",
        ]

        safe = detector.get_horizon_safe_features(
            features, horizon=2, target_column="carga"
        )

        # Only lag >= 96 should be safe
        assert "carga_lag_48" not in safe
        assert "carga_lag_96" in safe
        assert "carga_lag_144" in safe

        # Forward-looking features should be excluded
        assert "forward_temp" not in safe

    def test_get_horizon_safe_features_excludes_target(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test that target column is always excluded."""
        features = ["hour", "carga", "carga_lag_48", "temperature"]

        safe = detector.get_horizon_safe_features(
            features, horizon=0, target_column="carga"
        )

        assert "carga" not in safe
        assert "hour" in safe
        assert "carga_lag_48" in safe

    def test_get_horizon_safe_features_mixed_patterns(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test filtering with mixture of safe and unsafe features."""
        features = [
            "hour",
            "day_of_week",
            "carga_lag_24",
            "carga_lag_96",
            "future_value",
            "actual_temp",
            "target_derived",
            "temperature",
            "rolling_mean_7d",
        ]

        safe = detector.get_horizon_safe_features(
            features, horizon=2, target_column="carga"
        )

        # Safe features
        assert "hour" in safe
        assert "day_of_week" in safe
        assert "temperature" in safe
        assert "rolling_mean_7d" in safe
        assert "carga_lag_96" in safe

        # Unsafe features
        assert "carga_lag_24" not in safe  # Insufficient lag
        assert "future_value" not in safe  # Leakage pattern
        assert "actual_temp" not in safe  # Leakage pattern
        assert "target_derived" not in safe  # Leakage pattern

    def test_check_temporal_leakage_no_leakage(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test leakage check with safe features only."""
        df = pd.DataFrame(
            {
                "carga": [100, 105, 110],
                "hour": [0, 1, 2],
                "carga_lag_48": [95, 100, 105],
                "temperature": [25, 26, 25],
            }
        )

        result = detector.check_temporal_leakage(
            df=df,
            feature_cols=["hour", "carga_lag_48", "temperature"],
            target_col="carga",
            horizon=1,
        )

        assert not result["has_leakage"]
        assert len(result["unsafe_features"]) == 0
        assert len(result["warnings"]) == 0

    def test_check_temporal_leakage_with_leakage(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test leakage check with unsafe features."""
        df = pd.DataFrame(
            {
                "carga": [100, 105, 110],
                "hour": [0, 1, 2],
                "carga_lag_24": [95, 100, 105],
                "future_value": [110, 115, 120],
            }
        )

        result = detector.check_temporal_leakage(
            df=df,
            feature_cols=["hour", "carga_lag_24", "future_value"],
            target_col="carga",
            horizon=1,
        )

        assert result["has_leakage"]
        assert "carga_lag_24" in result["unsafe_features"]
        assert "future_value" in result["unsafe_features"]
        assert "hour" not in result["unsafe_features"]
        assert len(result["warnings"]) > 0

    def test_check_temporal_leakage_multiple_unsafe_features(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test leakage check with multiple unsafe features."""
        df = pd.DataFrame(
            {
                "carga": [100, 105, 110],
                "carga_lag_12": [90, 95, 100],
                "future_temp": [26, 27, 28],
                "actual_value": [100, 105, 110],
                "target_next": [105, 110, 115],
            }
        )

        result = detector.check_temporal_leakage(
            df=df,
            feature_cols=[
                "carga_lag_12",
                "future_temp",
                "actual_value",
                "target_next",
            ],
            target_col="carga",
            horizon=2,
        )

        assert result["has_leakage"]
        # All features should be unsafe
        assert len(result["unsafe_features"]) == 4
        assert "carga_lag_12" in result["unsafe_features"]  # Insufficient lag
        assert "future_temp" in result["unsafe_features"]  # Future pattern
        assert "actual_value" in result["unsafe_features"]  # Actual pattern
        assert "target_next" in result["unsafe_features"]  # Target pattern

    def test_get_horizon_safe_features_empty_list(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test with empty feature list."""
        safe = detector.get_horizon_safe_features(
            features=[], horizon=1, target_column="carga"
        )

        assert safe == []

    def test_get_horizon_safe_features_all_unsafe(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test when all features are unsafe."""
        features = ["future_value", "actual_temp", "target_derived", "carga"]

        safe = detector.get_horizon_safe_features(
            features, horizon=1, target_column="carga"
        )

        assert len(safe) == 0

    def test_detector_case_insensitive_patterns(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test that leakage pattern matching is case-insensitive."""
        assert detector._matches_leakage_pattern("FUTURE_value")
        assert detector._matches_leakage_pattern("Future_Value")
        assert detector._matches_leakage_pattern("forward_TEMP")
        assert detector._matches_leakage_pattern("ACTUAL_value")

    def test_lag_extraction_with_case_variations(
        self, detector: DataLeakageDetector
    ) -> None:
        """Test lag value extraction with different case variations."""
        # Should work with uppercase LAG
        assert detector._is_lag_safe_for_horizon("carga_LAG_48", horizon=1)
        assert not detector._is_lag_safe_for_horizon("carga_LAG_24", horizon=1)

        # Should work with mixed case
        assert detector._is_lag_safe_for_horizon("carga_Lag_48", horizon=1)
