"""Tests for seasonality plugin.

This module tests the SeasonalityPlugin which performs STL/MSTL decomposition
to extract seasonal patterns from time series data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.plugins.seasonality import SeasonalityConfig, SeasonalityPlugin


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return SeasonalityPlugin()


@pytest.fixture
def synthetic_seasonal_data():
    """Create synthetic data with known seasonality.

    Generates 2 years of semi-hourly data with:
    - Linear trend
    - Daily seasonality (48-period)
    - Weekly seasonality (336-period)
    - Random noise
    """
    np.random.seed(42)

    # Generate 2 years of semi-hourly data
    n_periods = 2 * 365 * 48
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="30min")

    # Trend
    trend = np.linspace(1000, 1200, n_periods)

    # Daily seasonality (48-period)
    daily_pattern = 100 * np.sin(2 * np.pi * np.arange(48) / 48)
    daily_seasonal = np.tile(daily_pattern, n_periods // 48)[:n_periods]

    # Weekly seasonality (336-period)
    weekly_pattern = 50 * np.sin(2 * np.pi * np.arange(336) / 336)
    weekly_seasonal = np.tile(weekly_pattern, n_periods // 336 + 1)[:n_periods]

    # Noise
    noise = np.random.randn(n_periods) * 10

    # Combine
    load = trend + daily_seasonal + weekly_seasonal + noise

    return pd.DataFrame(
        {
            "carga": load,
            "true_trend": trend,
            "true_daily": daily_seasonal,
            "true_weekly": weekly_seasonal,
        },
        index=dates,
    )


class TestSeasonalityConfig:
    """Tests for SeasonalityConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = SeasonalityConfig()

        assert config.target_column == "carga"
        assert config.decomposition_method == "mstl"
        assert config.seasonal_smoother == 7
        assert config.robust is True
        assert config.seasonal_strength_threshold == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = SeasonalityConfig(
            target_column="load",
            decomposition_method="stl",
            seasonal_periods={"daily": 48},
            seasonal_smoother=9,
            robust=False,
        )

        assert config.target_column == "load"
        assert config.decomposition_method == "stl"
        assert config.seasonal_periods == {"daily": 48}
        assert config.seasonal_smoother == 9
        assert config.robust is False

    def test_invalid_period(self):
        """Test validation of invalid seasonal period."""
        with pytest.raises(ValueError, match="must be >= 2"):
            SeasonalityConfig(seasonal_periods={"daily": 1})

    def test_even_seasonal_smoother(self):
        """Test validation of even seasonal smoother."""
        with pytest.raises(ValueError, match="must be odd"):
            SeasonalityConfig(seasonal_smoother=6)

    def test_small_seasonal_smoother(self):
        """Test validation of small seasonal smoother."""
        with pytest.raises(ValueError, match="must be >= 3"):
            SeasonalityConfig(seasonal_smoother=1)

    def test_invalid_threshold(self):
        """Test validation of invalid threshold."""
        with pytest.raises(ValueError, match="must be in"):
            SeasonalityConfig(seasonal_strength_threshold=1.5)

        with pytest.raises(ValueError, match="must be in"):
            SeasonalityConfig(seasonal_strength_threshold=-0.1)


class TestSeasonalityPlugin:
    """Tests for SeasonalityPlugin functionality."""

    def test_plugin_properties(self, plugin):
        """Test plugin properties."""
        assert plugin.name == "seasonality"
        assert plugin.version == "1.0.0"
        assert plugin.computational_complexity == "medium"

    def test_estimate_compute_time(self, plugin):
        """Test time estimation."""
        # 1 year of data
        time_estimate = plugin.estimate_compute_time(17520)

        # Should estimate ~5 seconds
        assert 3 < time_estimate < 10

        # 2 years should take ~10 seconds
        time_estimate_2y = plugin.estimate_compute_time(2 * 17520)
        assert 8 < time_estimate_2y < 15

    def test_validate_config(self, plugin):
        """Test configuration validation."""
        config = {
            "target_column": "carga",
            "seasonal_periods": {"daily": 48},
        }

        assert plugin.validate_config(config) is True

    def test_validate_invalid_config(self, plugin):
        """Test validation of invalid config."""
        config = {
            "seasonal_periods": {"daily": 1},  # Invalid period
        }

        with pytest.raises(ValueError):
            plugin.validate_config(config)

    def test_basic_stl_decomposition(self, plugin, synthetic_seasonal_data):
        """Test basic STL decomposition."""
        config = {
            "target_column": "carga",
            "decomposition_method": "stl",
            "seasonal_periods": {"daily": 48},
        }

        features = plugin.generate_features(synthetic_seasonal_data, config)

        # Should have trend, seasonal, residual
        assert "trend" in features.columns
        assert "seasonal_daily" in features.columns
        assert "residual" in features.columns

        # Should have derived features
        assert "seasonal_daily_lag_1" in features.columns
        assert "seasonal_daily_diff" in features.columns
        assert "detrended" in features.columns
        assert "deseasonalized" in features.columns

    def test_mstl_decomposition(self, plugin, synthetic_seasonal_data):
        """Test MSTL with multiple periods."""
        config = {
            "target_column": "carga",
            "decomposition_method": "mstl",
            "seasonal_periods": {"daily": 48, "weekly": 336},
        }

        features = plugin.generate_features(synthetic_seasonal_data, config)

        # Should have both seasonal components
        assert "seasonal_daily" in features.columns
        assert "seasonal_weekly" in features.columns
        assert "trend" in features.columns
        assert "residual" in features.columns

        # Should have lagged features for both
        assert "seasonal_daily_lag_1" in features.columns
        assert "seasonal_weekly_lag_1" in features.columns

    def test_decomposition_reconstruction(self, plugin, synthetic_seasonal_data):
        """Test that components sum back to original."""
        config = {
            "target_column": "carga",
            "decomposition_method": "mstl",
            "seasonal_periods": {"daily": 48},
        }

        features = plugin.generate_features(synthetic_seasonal_data, config)

        # Reconstruct
        reconstructed = (
            features["trend"] + features["seasonal_daily"] + features["residual"]
        )

        # Should match original (allowing small numerical errors)
        original = synthetic_seasonal_data["carga"]
        common_idx = reconstructed.index.intersection(original.index)

        diff = np.abs(reconstructed.loc[common_idx] - original.loc[common_idx])
        assert diff.mean() < 1e-10

    def test_lagged_features(self, plugin, synthetic_seasonal_data):
        """Test lagged seasonal features are generated."""
        config = {
            "target_column": "carga",
            "decomposition_method": "stl",
            "seasonal_periods": {"daily": 48},
        }

        features = plugin.generate_features(synthetic_seasonal_data, config)

        # Should have lagged features
        assert "seasonal_daily_lag_1" in features.columns
        assert "seasonal_daily_diff" in features.columns

        # Lag 1 should be shifted by 1
        assert pd.isna(features["seasonal_daily_lag_1"].iloc[0])
        assert features["seasonal_daily_lag_1"].iloc[1] == features["seasonal_daily"].iloc[0]

    def test_seasonal_profiles(self, plugin, synthetic_seasonal_data):
        """Test seasonal profile generation."""
        config = {
            "target_column": "carga",
            "decomposition_method": "stl",
            "seasonal_periods": {"daily": 48},
        }

        plugin.generate_features(synthetic_seasonal_data, config)

        # Should have daily profile
        profile = plugin.get_seasonal_profile("daily")

        assert profile is not None
        assert len(profile) == 48

    def test_seasonal_strength_calculation(self, plugin, synthetic_seasonal_data):
        """Test seasonal strength calculation."""
        config = {
            "target_column": "carga",
            "decomposition_method": "stl",
            "seasonal_periods": {"daily": 48},
        }

        plugin.generate_features(synthetic_seasonal_data, config)

        # Should have calculated strength
        daily_strength = plugin.get_seasonal_strength("daily")
        trend_strength = plugin.get_seasonal_strength("trend")

        assert daily_strength is not None
        assert trend_strength is not None

        # With synthetic data, should have strong seasonality
        assert daily_strength > 0.5

    def test_insufficient_data(self, plugin):
        """Test handling of insufficient data."""
        short_df = pd.DataFrame(
            {"carga": np.random.randn(50)},
            index=pd.date_range("2024-01-01", periods=50, freq="h"),
        )

        config = {
            "target_column": "carga",
            "seasonal_periods": {"daily": 48},
        }

        # Should return empty features gracefully
        features = plugin.generate_features(short_df, config)
        assert len(features.columns) == 0

    def test_missing_data_handling(self, plugin, synthetic_seasonal_data):
        """Test handling of missing data."""
        # Introduce missing values
        df_with_missing = synthetic_seasonal_data.copy()
        df_with_missing.loc[df_with_missing.index[100:110], "carga"] = np.nan

        config = {
            "target_column": "carga",
            "seasonal_periods": {"daily": 48},
        }

        # Should handle gracefully
        features = plugin.generate_features(df_with_missing, config)
        assert len(features) > 0

    def test_missing_target_column(self, plugin, synthetic_seasonal_data):
        """Test error on missing target column."""
        config = {
            "target_column": "nonexistent",
            "seasonal_periods": {"daily": 48},
        }

        with pytest.raises(ValueError, match="not found"):
            plugin.generate_features(synthetic_seasonal_data, config)

    def test_feature_names(self, plugin):
        """Test feature name generation."""
        config = {"seasonal_periods": {"daily": 48, "weekly": 336}}

        feature_names = plugin.get_feature_names(config)

        assert "seasonal_daily" in feature_names
        assert "seasonal_weekly" in feature_names
        assert "seasonal_daily_lag_1" in feature_names
        assert "seasonal_weekly_lag_1" in feature_names
        assert "seasonal_daily_diff" in feature_names
        assert "seasonal_weekly_diff" in feature_names
        assert "trend" in feature_names
        assert "residual" in feature_names
        assert "detrended" in feature_names
        assert "deseasonalized" in feature_names

    def test_get_metadata(self, plugin):
        """Test metadata retrieval."""
        metadata = plugin.get_metadata()

        assert metadata["name"] == "seasonality"
        assert metadata["version"] == "1.0.0"
        assert metadata["computational_complexity"] == "medium"
        assert "class" in metadata
        assert "module" in metadata


class TestSeasonalityWithRealPatterns:
    """Tests with realistic electricity load patterns."""

    def test_strong_daily_pattern(self, plugin):
        """Test with strong daily pattern."""
        np.random.seed(123)

        # Create data with strong daily pattern
        n_days = 30
        n_periods = n_days * 48
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="30min")

        # Strong daily pattern
        base_load = 1000
        daily_amplitude = 300
        hour_of_day = np.arange(n_periods) % 48
        daily_pattern = daily_amplitude * np.sin(2 * np.pi * hour_of_day / 48)

        # Small noise
        noise = np.random.randn(n_periods) * 20

        load = base_load + daily_pattern + noise

        df = pd.DataFrame({"carga": load}, index=dates)

        config = {
            "target_column": "carga",
            "decomposition_method": "stl",
            "seasonal_periods": {"daily": 48},
        }

        plugin.generate_features(df, config)

        # Should detect strong daily seasonality
        daily_strength = plugin.get_seasonal_strength("daily")
        assert daily_strength is not None
        assert daily_strength > 0.7  # Strong seasonality

    def test_weak_seasonality(self, plugin):
        """Test with weak seasonality."""
        np.random.seed(456)

        # Create data with weak seasonality
        n_periods = 30 * 48
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="30min")

        # Weak seasonal pattern
        base_load = 1000
        seasonal_amplitude = 10  # Very small
        hour_of_day = np.arange(n_periods) % 48
        seasonal_pattern = seasonal_amplitude * np.sin(2 * np.pi * hour_of_day / 48)

        # Large noise
        noise = np.random.randn(n_periods) * 100

        load = base_load + seasonal_pattern + noise

        df = pd.DataFrame({"carga": load}, index=dates)

        config = {
            "target_column": "carga",
            "decomposition_method": "stl",
            "seasonal_periods": {"daily": 48},
        }

        plugin.generate_features(df, config)

        # Should detect weak seasonality
        daily_strength = plugin.get_seasonal_strength("daily")
        assert daily_strength is not None
        assert daily_strength < 0.5  # Weak seasonality

    def test_multiple_periods(self, plugin):
        """Test with multiple seasonal periods."""
        np.random.seed(789)

        # Create 60 days of data with daily and weekly patterns
        n_days = 60
        n_periods = n_days * 48
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="30min")

        base_load = 1000

        # Daily pattern
        hour_of_day = np.arange(n_periods) % 48
        daily_pattern = 200 * np.sin(2 * np.pi * hour_of_day / 48)

        # Weekly pattern
        hour_of_week = np.arange(n_periods) % 336
        weekly_pattern = 100 * np.sin(2 * np.pi * hour_of_week / 336)

        # Noise
        noise = np.random.randn(n_periods) * 30

        load = base_load + daily_pattern + weekly_pattern + noise

        df = pd.DataFrame({"carga": load}, index=dates)

        config = {
            "target_column": "carga",
            "decomposition_method": "mstl",
            "seasonal_periods": {"daily": 48, "weekly": 336},
        }

        features = plugin.generate_features(df, config)

        # Should have both components
        assert "seasonal_daily" in features.columns
        assert "seasonal_weekly" in features.columns

        # Both should have reasonable strength
        daily_strength = plugin.get_seasonal_strength("daily")
        weekly_strength = plugin.get_seasonal_strength("weekly")

        assert daily_strength is not None
        assert weekly_strength is not None
        assert daily_strength > 0.5
        assert weekly_strength > 0.3
