"""Tests for RFFeatureSelectorPlugin class.

This module contains comprehensive tests for the Random Forest-based
feature selection plugin with horizon-aware leakage prevention.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.features.evaluation.leakage_detector import DataLeakageDetector
from src.features.plugins.rf_selector import (
    RFFeatureSelectorConfig,
    RFFeatureSelectorPlugin,
)


class TestRFFeatureSelectorConfig:
    """Test suite for RFFeatureSelectorConfig."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        config = RFFeatureSelectorConfig()

        assert config.target_column == "carga"
        assert config.horizons == list(range(0, 9))
        assert config.max_features_per_horizon == 50
        assert config.selection_method == "rfecv"
        assert config.cv_folds == 5
        assert config.importance_threshold == 0.01
        assert config.n_estimators == 50
        assert config.random_state == 42
        assert config.n_jobs == -1
        assert config.periods_per_day == 48

    def test_config_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = RFFeatureSelectorConfig(
            target_column="load",
            horizons=[0, 1, 2],
            max_features_per_horizon=30,
            selection_method="permutation",
            cv_folds=3,
            n_estimators=100,
        )

        assert config.target_column == "load"
        assert config.horizons == [0, 1, 2]
        assert config.max_features_per_horizon == 30
        assert config.selection_method == "permutation"
        assert config.cv_folds == 3
        assert config.n_estimators == 100

    def test_config_validates_horizons_non_negative(self) -> None:
        """Test that negative horizons are rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            RFFeatureSelectorConfig(horizons=[-1, 0, 1])

    def test_config_validates_max_features(self) -> None:
        """Test that max_features must be positive."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            RFFeatureSelectorConfig(max_features_per_horizon=0)

    def test_config_sorts_horizons(self) -> None:
        """Test that horizons are automatically sorted."""
        config = RFFeatureSelectorConfig(horizons=[3, 1, 2, 0])
        assert config.horizons == [0, 1, 2, 3]

    def test_config_frozen(self) -> None:
        """Test that config is immutable."""
        config = RFFeatureSelectorConfig()
        with pytest.raises(Exception):  # Pydantic ValidationError
            config.horizons = [0, 1]  # type: ignore


class TestRFFeatureSelectorPlugin:
    """Test suite for RFFeatureSelectorPlugin."""

    @pytest.fixture
    def plugin(self) -> RFFeatureSelectorPlugin:
        """Create plugin instance."""
        return RFFeatureSelectorPlugin()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with features and target."""
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="h")

        # Create target with some pattern
        target = 1000 + 100 * np.sin(np.arange(n_samples) * 2 * np.pi / 24)
        target += np.random.randn(n_samples) * 10

        # Safe features (temporal, calendar)
        hour = dates.hour
        day_of_week = dates.dayofweek
        month = dates.month

        # Lag features
        lag_24 = pd.Series(target).shift(24).values
        lag_48 = pd.Series(target).shift(48).values
        lag_96 = pd.Series(target).shift(96).values

        # Additional features
        temperature = 25 + 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 24)
        temperature += np.random.randn(n_samples) * 2

        # Unsafe features (should be filtered)
        future_value = pd.Series(target).shift(-24).values

        return pd.DataFrame(
            {
                "carga": target,
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "carga_lag_24": lag_24,
                "carga_lag_48": lag_48,
                "carga_lag_96": lag_96,
                "temperature": temperature,
                "future_value": future_value,
            },
            index=dates,
        )

    def test_plugin_initialization(self, plugin: RFFeatureSelectorPlugin) -> None:
        """Test plugin initialization."""
        assert plugin.name == "rf_feature_selector"
        assert plugin.version == "1.0.0"
        assert plugin.computational_complexity == "high"
        assert plugin.selection_metadata == {}
        assert plugin.leakage_detector is None

    def test_plugin_estimate_compute_time(
        self, plugin: RFFeatureSelectorPlugin
    ) -> None:
        """Test computation time estimation."""
        # For 1 year of semi-hourly data (8760 samples)
        time_estimate = plugin.estimate_compute_time(8760)
        assert 30 < time_estimate < 300  # 30s to 5min is reasonable

        # Smaller dataset should have proportionally less time
        time_small = plugin.estimate_compute_time(1000)
        assert time_small < time_estimate

    def test_plugin_validate_config_valid(
        self, plugin: RFFeatureSelectorPlugin
    ) -> None:
        """Test config validation with valid config."""
        config = {
            "target_column": "carga",
            "horizons": [0, 1],
            "max_features_per_horizon": 10,
        }

        assert plugin.validate_config(config) is True

    def test_plugin_validate_config_invalid(
        self, plugin: RFFeatureSelectorPlugin
    ) -> None:
        """Test config validation with invalid config."""
        config = {"horizons": [-1, 0]}  # Negative horizon

        with pytest.raises(ValueError):
            plugin.validate_config(config)

    def test_basic_feature_selection_single_horizon(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test basic feature selection for single horizon."""
        config = {
            "target_column": "carga",
            "horizons": [0],
            "max_features_per_horizon": 5,
            "selection_method": "permutation",
            "n_estimators": 10,  # Fast for testing
        }

        result = plugin.generate_features(sample_df, config)

        # Result should be empty DataFrame (results in metadata)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

        # Should have selection metadata
        selected = plugin.get_selected_features(horizon=0)
        assert isinstance(selected, list)
        assert len(selected) > 0
        assert len(selected) <= 5

    def test_multi_horizon_selection(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test selection for multiple horizons."""
        config = {
            "target_column": "carga",
            "horizons": [0, 1, 2],
            "max_features_per_horizon": 5,
            "selection_method": "permutation",
            "n_estimators": 10,
        }

        plugin.generate_features(sample_df, config)

        # Should have selections for all horizons
        for horizon in [0, 1, 2]:
            selected = plugin.get_selected_features(horizon)
            assert len(selected) > 0
            assert len(selected) <= 5

    def test_leakage_detection_filters_unsafe_features(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test that future-looking features are filtered."""
        config = {
            "target_column": "carga",
            "horizons": [1],
            "max_features_per_horizon": 10,
            "selection_method": "permutation",
            "n_estimators": 10,
        }

        plugin.generate_features(sample_df, config)
        selected = plugin.get_selected_features(horizon=1)

        # 'future_value' should not be selected (leakage pattern)
        assert "future_value" not in selected

        # 'carga_lag_24' should not be selected (insufficient lag for D+1)
        assert "carga_lag_24" not in selected

    def test_lag_feature_validation_for_horizons(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test that lag features are properly validated per horizon."""
        config = {
            "target_column": "carga",
            "horizons": [0, 1, 2],
            "max_features_per_horizon": 10,
            "selection_method": "permutation",
            "n_estimators": 10,
        }

        plugin.generate_features(sample_df, config)

        # Horizon 0: lag_24 should be available
        selected_h0 = plugin.get_selected_features(horizon=0)
        # Note: may or may not be selected based on importance, but should be allowed

        # Horizon 1 (requires lag >= 48): lag_24 should not be available
        selected_h1 = plugin.get_selected_features(horizon=1)
        assert "carga_lag_24" not in selected_h1

        # Horizon 2 (requires lag >= 96): lag_48 should not be available
        selected_h2 = plugin.get_selected_features(horizon=2)
        assert "carga_lag_24" not in selected_h2
        assert "carga_lag_48" not in selected_h2

    def test_rfecv_selection_method(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test RFECV selection method."""
        config = {
            "target_column": "carga",
            "horizons": [0],
            "selection_method": "rfecv",
            "max_features_per_horizon": 5,
            "n_estimators": 10,
            "cv_folds": 3,
        }

        plugin.generate_features(sample_df, config)
        selected = plugin.get_selected_features(horizon=0)

        assert len(selected) > 0
        assert len(selected) <= 5

    def test_shap_selection_method(
        self,
        plugin: RFFeatureSelectorPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test SHAP selection method (fallback to permutation)."""
        # Test the fallback behavior when SHAP is not available
        config = {
            "target_column": "carga",
            "horizons": [0],
            "selection_method": "shap",
            "max_features_per_horizon": 5,
            "n_estimators": 10,
        }

        # This should fall back to permutation if shap is not installed
        plugin.generate_features(sample_df, config)
        selected = plugin.get_selected_features(horizon=0)

        assert len(selected) > 0

    def test_combined_selection_method(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test combined selection method."""
        config = {
            "target_column": "carga",
            "horizons": [0],
            "selection_method": "combined",
            "max_features_per_horizon": 5,
            "n_estimators": 10,
            "cv_folds": 3,
        }

        plugin.generate_features(sample_df, config)
        selected = plugin.get_selected_features(horizon=0)

        assert len(selected) > 0
        assert len(selected) <= 5

        # Check that metadata includes info from both methods
        metadata = plugin.selection_metadata["horizon_0"]
        assert "rfecv_selected" in metadata or "perm_selected" in metadata

    def test_get_importance_scores(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test feature importance score retrieval."""
        config = {
            "target_column": "carga",
            "horizons": [0],
            "max_features_per_horizon": 5,
            "selection_method": "permutation",
            "n_estimators": 10,
        }

        plugin.generate_features(sample_df, config)
        importance = plugin.get_importance_scores(horizon=0)

        assert isinstance(importance, dict)
        assert len(importance) > 0
        # All importance values should be non-negative
        assert all(v >= 0 for v in importance.values())

    def test_generate_selection_report(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test selection report generation."""
        config = {
            "target_column": "carga",
            "horizons": [0, 1],
            "max_features_per_horizon": 5,
            "selection_method": "permutation",
            "n_estimators": 10,
        }

        plugin.generate_features(sample_df, config)
        report = plugin.generate_selection_report()

        assert isinstance(report, pd.DataFrame)
        assert len(report) == 2  # Two horizons
        assert "horizon" in report.columns
        assert "n_features" in report.columns
        assert "top_feature" in report.columns
        assert "mean_importance" in report.columns

    def test_insufficient_data_handling(
        self, plugin: RFFeatureSelectorPlugin
    ) -> None:
        """Test handling of insufficient data."""
        small_df = pd.DataFrame(
            {
                "carga": np.random.randn(50),
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
            },
            index=pd.date_range("2024-01-01", periods=50, freq="h"),
        )

        config = {
            "target_column": "carga",
            "horizons": [0],
            "max_features_per_horizon": 3,
            "selection_method": "permutation",
            "n_estimators": 10,
        }

        # Should not crash
        plugin.generate_features(small_df, config)

        # Should still return some features (fallback behavior)
        selected = plugin.get_selected_features(horizon=0)
        assert isinstance(selected, list)

    def test_missing_target_column_raises_error(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test that missing target column raises error."""
        config = {
            "target_column": "nonexistent_column",
            "horizons": [0],
        }

        with pytest.raises(ValueError, match="not found"):
            plugin.generate_features(sample_df, config)

    def test_get_selected_features_nonexistent_horizon(
        self, plugin: RFFeatureSelectorPlugin
    ) -> None:
        """Test getting features for horizon that wasn't processed."""
        selected = plugin.get_selected_features(horizon=99)
        assert selected == []

    def test_get_importance_scores_nonexistent_horizon(
        self, plugin: RFFeatureSelectorPlugin
    ) -> None:
        """Test getting importance for horizon that wasn't processed."""
        importance = plugin.get_importance_scores(horizon=99)
        assert importance == {}

    def test_get_feature_names_returns_empty(
        self, plugin: RFFeatureSelectorPlugin
    ) -> None:
        """Test that get_feature_names returns empty list."""
        # This plugin doesn't generate features, only selects them
        feature_names = plugin.get_feature_names({})
        assert feature_names == []

    def test_max_features_enforcement(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test that max_features_per_horizon limit is enforced."""
        config = {
            "target_column": "carga",
            "horizons": [0],
            "max_features_per_horizon": 3,
            "selection_method": "permutation",
            "n_estimators": 10,
        }

        plugin.generate_features(sample_df, config)
        selected = plugin.get_selected_features(horizon=0)

        # Should respect the limit
        assert len(selected) <= 3

    def test_selection_with_many_nan_values(
        self, plugin: RFFeatureSelectorPlugin
    ) -> None:
        """Test feature selection with dataset containing NaN values."""
        np.random.seed(42)
        n_samples = 300
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="h")

        # Create data with some NaN values
        target = 1000 + np.random.randn(n_samples) * 50
        feature1 = np.random.randn(n_samples)
        feature2 = np.random.randn(n_samples)

        # Introduce NaN values in early rows (will be handled by lag creation)
        feature1[:50] = np.nan

        df = pd.DataFrame(
            {
                "carga": target,
                "feature1": feature1,
                "feature2": feature2,
                "hour": dates.hour,
            },
            index=dates,
        )

        config = {
            "target_column": "carga",
            "horizons": [0],
            "max_features_per_horizon": 3,
            "selection_method": "permutation",
            "n_estimators": 10,
        }

        # Should handle NaN gracefully
        plugin.generate_features(df, config)
        selected = plugin.get_selected_features(horizon=0)

        assert isinstance(selected, list)

    def test_plugin_metadata(self, plugin: RFFeatureSelectorPlugin) -> None:
        """Test plugin metadata retrieval."""
        metadata = plugin.get_metadata()

        assert metadata["name"] == "rf_feature_selector"
        assert metadata["version"] == "1.0.0"
        assert metadata["computational_complexity"] == "high"
        assert "supports_incremental" in metadata

    def test_selection_error_handling(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test that errors during selection are handled gracefully."""
        config = {
            "target_column": "carga",
            "horizons": [0, 1, 2],
            "max_features_per_horizon": 5,
            "selection_method": "permutation",
            "n_estimators": 10,
        }

        # Even if one horizon fails, others should succeed
        plugin.generate_features(sample_df, config)

        # Should have attempted all horizons
        assert len(plugin.selection_metadata) == 3

    def test_selection_reproducibility(
        self, plugin: RFFeatureSelectorPlugin, sample_df: pd.DataFrame
    ) -> None:
        """Test that selection is reproducible with same random_state."""
        config = {
            "target_column": "carga",
            "horizons": [0],
            "max_features_per_horizon": 5,
            "selection_method": "permutation",
            "n_estimators": 10,
            "random_state": 42,
        }

        # First run
        plugin.generate_features(sample_df, config)
        selected1 = plugin.get_selected_features(horizon=0)

        # Second run with same random state
        plugin2 = RFFeatureSelectorPlugin()
        plugin2.generate_features(sample_df, config)
        selected2 = plugin2.get_selected_features(horizon=0)

        # Should produce same results (note: order might differ)
        assert set(selected1) == set(selected2)
