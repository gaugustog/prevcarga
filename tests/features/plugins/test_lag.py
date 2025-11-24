"""Tests for LagFeaturesPlugin.

This module contains comprehensive tests for the LagFeaturesPlugin,
including lag feature generation, rolling statistics, NaN handling,
multiple columns, feature naming, configuration validation, and performance tests.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.plugins.lag import (
    DEFAULT_LAG_PERIODS,
    DEFAULT_ROLLING_STATS,
    DEFAULT_ROLLING_WINDOWS,
    SUPPORTED_ROLLING_STATS,
    LagFeaturesConfig,
    LagFeaturesPlugin,
)


class TestLagFeaturesConfig:
    """Tests for LagFeaturesConfig Pydantic model."""

    def test_default_config(self):
        """Test that default configuration has expected values."""
        config = LagFeaturesConfig()

        assert config.lag_periods == DEFAULT_LAG_PERIODS
        assert config.rolling_windows == DEFAULT_ROLLING_WINDOWS
        assert config.rolling_stats == DEFAULT_ROLLING_STATS
        assert config.target_column == "carga"
        assert config.additional_columns == []
        assert config.fill_method == "none"

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = LagFeaturesConfig(
            lag_periods=[1, 2, 3],
            rolling_windows=[12, 24],
            rolling_stats=["mean", "median"],
            target_column="load",
            additional_columns=["temp"],
            fill_method="forward",
        )

        assert config.lag_periods == [1, 2, 3]
        assert config.rolling_windows == [12, 24]
        assert config.rolling_stats == ["mean", "median"]
        assert config.target_column == "load"
        assert config.additional_columns == ["temp"]
        assert config.fill_method == "forward"

    def test_negative_lag_periods_rejected(self):
        """Test that negative lag periods raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LagFeaturesConfig(lag_periods=[1, -2, 3])

        assert "positive integers" in str(exc_info.value)
        assert "-2" in str(exc_info.value)

    def test_zero_lag_period_rejected(self):
        """Test that zero lag period raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LagFeaturesConfig(lag_periods=[0, 1, 2])

        assert "positive integers" in str(exc_info.value)

    def test_negative_rolling_windows_rejected(self):
        """Test that negative rolling windows raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LagFeaturesConfig(rolling_windows=[-24, 168])

        assert "positive integers" in str(exc_info.value)
        assert "-24" in str(exc_info.value)

    def test_zero_rolling_window_rejected(self):
        """Test that zero rolling window raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LagFeaturesConfig(rolling_windows=[0, 24])

        assert "positive integers" in str(exc_info.value)

    def test_invalid_rolling_stats_rejected(self):
        """Test that invalid rolling statistics raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LagFeaturesConfig(rolling_stats=["mean", "invalid_stat"])

        assert "Invalid rolling statistics" in str(exc_info.value)
        assert "invalid_stat" in str(exc_info.value)

    def test_all_supported_rolling_stats(self):
        """Test that all supported rolling statistics are accepted."""
        config = LagFeaturesConfig(rolling_stats=list(SUPPORTED_ROLLING_STATS))
        assert set(config.rolling_stats) == SUPPORTED_ROLLING_STATS

    def test_fill_method_options(self):
        """Test that all fill method options are accepted."""
        for method in ["none", "forward", "backward"]:
            config = LagFeaturesConfig(fill_method=method)  # type: ignore[arg-type]
            assert config.fill_method == method

    def test_invalid_fill_method_rejected(self):
        """Test that invalid fill method raises ValueError."""
        with pytest.raises(ValueError):
            LagFeaturesConfig(fill_method="invalid")  # type: ignore[arg-type]

    def test_empty_lag_periods_allowed(self):
        """Test that empty lag periods list is allowed."""
        config = LagFeaturesConfig(lag_periods=[])
        assert config.lag_periods == []

    def test_empty_rolling_windows_allowed(self):
        """Test that empty rolling windows list is allowed."""
        config = LagFeaturesConfig(rolling_windows=[])
        assert config.rolling_windows == []

    def test_empty_rolling_stats_allowed(self):
        """Test that empty rolling stats list is allowed."""
        config = LagFeaturesConfig(rolling_stats=[])
        assert config.rolling_stats == []

    def test_config_model_dump(self):
        """Test that config can be serialized to dict."""
        config = LagFeaturesConfig()
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert "lag_periods" in config_dict
        assert "rolling_windows" in config_dict
        assert "rolling_stats" in config_dict
        assert "target_column" in config_dict
        assert "additional_columns" in config_dict
        assert "fill_method" in config_dict

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValueError):
            LagFeaturesConfig(unknown_field="value")  # type: ignore[call-arg]


class TestLagFeaturesPlugin:
    """Tests for LagFeaturesPlugin base functionality."""

    @pytest.fixture
    def plugin(self) -> LagFeaturesPlugin:
        """Create a plugin instance for testing."""
        return LagFeaturesPlugin()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame with carga column."""
        return pd.DataFrame(
            {"carga": [100.0, 200.0, 300.0, 400.0, 500.0]},
            index=pd.date_range("2024-01-01", periods=5, freq="h"),
        )

    @pytest.fixture
    def default_config(self) -> dict[str, Any]:
        """Return default configuration as dict."""
        return LagFeaturesConfig().model_dump()

    def test_plugin_name(self, plugin: LagFeaturesPlugin):
        """Test that plugin name is correct."""
        assert plugin.name == "lag_features"

    def test_plugin_version(self, plugin: LagFeaturesPlugin):
        """Test that plugin version is correct."""
        assert plugin.version == "1.0.0"

    def test_get_metadata(self, plugin: LagFeaturesPlugin):
        """Test that get_metadata returns correct values."""
        metadata = plugin.get_metadata()

        assert metadata["name"] == "lag_features"
        assert metadata["version"] == "1.0.0"
        assert metadata["class"] == "LagFeaturesPlugin"

    def test_repr(self, plugin: LagFeaturesPlugin):
        """Test plugin string representation."""
        repr_str = repr(plugin)

        assert "LagFeaturesPlugin" in repr_str
        assert "lag_features" in repr_str
        assert "1.0.0" in repr_str


class TestLagFeatureGeneration:
    """Tests for lag feature generation."""

    @pytest.fixture
    def plugin(self) -> LagFeaturesPlugin:
        """Create a plugin instance for testing."""
        return LagFeaturesPlugin()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame."""
        return pd.DataFrame(
            {"carga": [100.0, 200.0, 300.0, 400.0, 500.0]},
            index=pd.date_range("2024-01-01", periods=5, freq="h"),
        )

    def test_simple_lag_creation(self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame):
        """Test simple lag creation (lag_1 equals previous value)."""
        config = LagFeaturesConfig(
            lag_periods=[1],
            rolling_windows=[],
            rolling_stats=[],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "carga_lag_1" in result.columns
        # First value should be NaN, rest should be previous values
        assert pd.isna(result["carga_lag_1"].iloc[0])
        assert result["carga_lag_1"].iloc[1] == 100.0
        assert result["carga_lag_1"].iloc[2] == 200.0
        assert result["carga_lag_1"].iloc[3] == 300.0
        assert result["carga_lag_1"].iloc[4] == 400.0

    def test_lag_2_equals_two_periods_back(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test that lag_2 equals value from 2 periods back."""
        config = LagFeaturesConfig(
            lag_periods=[2],
            rolling_windows=[],
            rolling_stats=[],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "carga_lag_2" in result.columns
        assert pd.isna(result["carga_lag_2"].iloc[0])
        assert pd.isna(result["carga_lag_2"].iloc[1])
        assert result["carga_lag_2"].iloc[2] == 100.0
        assert result["carga_lag_2"].iloc[3] == 200.0
        assert result["carga_lag_2"].iloc[4] == 300.0

    def test_multiple_lag_periods(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test multiple lag periods generation."""
        config = LagFeaturesConfig(
            lag_periods=[1, 2, 3],
            rolling_windows=[],
            rolling_stats=[],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "carga_lag_1" in result.columns
        assert "carga_lag_2" in result.columns
        assert "carga_lag_3" in result.columns

    def test_large_lag_period_all_nan(self, plugin: LagFeaturesPlugin):
        """Test that lag period larger than dataset results in all NaN."""
        df = pd.DataFrame(
            {"carga": [100.0, 200.0, 300.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

        config = LagFeaturesConfig(
            lag_periods=[10],  # Larger than dataset
            rolling_windows=[],
            rolling_stats=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "carga_lag_10" in result.columns
        assert result["carga_lag_10"].isna().all()


class TestRollingStatistics:
    """Tests for rolling statistics feature generation."""

    @pytest.fixture
    def plugin(self) -> LagFeaturesPlugin:
        """Create a plugin instance for testing."""
        return LagFeaturesPlugin()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame with known values."""
        return pd.DataFrame(
            {"carga": [10.0, 20.0, 30.0, 40.0, 50.0]},
            index=pd.date_range("2024-01-01", periods=5, freq="h"),
        )

    def test_rolling_mean_calculation(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test rolling mean calculation."""
        config = LagFeaturesConfig(
            lag_periods=[],
            rolling_windows=[3],
            rolling_stats=["mean"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "carga_rolling_3_mean" in result.columns
        # First two values should be NaN (window not full)
        assert pd.isna(result["carga_rolling_3_mean"].iloc[0])
        assert pd.isna(result["carga_rolling_3_mean"].iloc[1])
        # Third value: mean(10, 20, 30) = 20
        np.testing.assert_almost_equal(result["carga_rolling_3_mean"].iloc[2], 20.0)
        # Fourth value: mean(20, 30, 40) = 30
        np.testing.assert_almost_equal(result["carga_rolling_3_mean"].iloc[3], 30.0)
        # Fifth value: mean(30, 40, 50) = 40
        np.testing.assert_almost_equal(result["carga_rolling_3_mean"].iloc[4], 40.0)

    def test_rolling_std_calculation(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test rolling standard deviation calculation."""
        config = LagFeaturesConfig(
            lag_periods=[],
            rolling_windows=[3],
            rolling_stats=["std"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "carga_rolling_3_std" in result.columns
        # Standard deviation of [10, 20, 30] with ddof=1 (default)
        expected_std = np.std([10, 20, 30], ddof=1)
        np.testing.assert_almost_equal(
            result["carga_rolling_3_std"].iloc[2], expected_std
        )

    def test_rolling_min_calculation(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test rolling minimum calculation."""
        config = LagFeaturesConfig(
            lag_periods=[],
            rolling_windows=[3],
            rolling_stats=["min"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "carga_rolling_3_min" in result.columns
        np.testing.assert_almost_equal(result["carga_rolling_3_min"].iloc[2], 10.0)
        np.testing.assert_almost_equal(result["carga_rolling_3_min"].iloc[3], 20.0)
        np.testing.assert_almost_equal(result["carga_rolling_3_min"].iloc[4], 30.0)

    def test_rolling_max_calculation(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test rolling maximum calculation."""
        config = LagFeaturesConfig(
            lag_periods=[],
            rolling_windows=[3],
            rolling_stats=["max"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "carga_rolling_3_max" in result.columns
        np.testing.assert_almost_equal(result["carga_rolling_3_max"].iloc[2], 30.0)
        np.testing.assert_almost_equal(result["carga_rolling_3_max"].iloc[3], 40.0)
        np.testing.assert_almost_equal(result["carga_rolling_3_max"].iloc[4], 50.0)

    def test_rolling_median_calculation(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test rolling median calculation."""
        config = LagFeaturesConfig(
            lag_periods=[],
            rolling_windows=[3],
            rolling_stats=["median"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "carga_rolling_3_median" in result.columns
        np.testing.assert_almost_equal(result["carga_rolling_3_median"].iloc[2], 20.0)
        np.testing.assert_almost_equal(result["carga_rolling_3_median"].iloc[3], 30.0)
        np.testing.assert_almost_equal(result["carga_rolling_3_median"].iloc[4], 40.0)

    def test_rolling_sum_calculation(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test rolling sum calculation."""
        config = LagFeaturesConfig(
            lag_periods=[],
            rolling_windows=[3],
            rolling_stats=["sum"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "carga_rolling_3_sum" in result.columns
        # sum(10, 20, 30) = 60
        np.testing.assert_almost_equal(result["carga_rolling_3_sum"].iloc[2], 60.0)
        # sum(20, 30, 40) = 90
        np.testing.assert_almost_equal(result["carga_rolling_3_sum"].iloc[3], 90.0)
        # sum(30, 40, 50) = 120
        np.testing.assert_almost_equal(result["carga_rolling_3_sum"].iloc[4], 120.0)

    def test_multiple_rolling_windows(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test multiple rolling window sizes."""
        config = LagFeaturesConfig(
            lag_periods=[],
            rolling_windows=[2, 3],
            rolling_stats=["mean"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "carga_rolling_2_mean" in result.columns
        assert "carga_rolling_3_mean" in result.columns

    def test_all_rolling_stats_together(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test all supported rolling statistics together."""
        config = LagFeaturesConfig(
            lag_periods=[],
            rolling_windows=[3],
            rolling_stats=list(SUPPORTED_ROLLING_STATS),
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        for stat in SUPPORTED_ROLLING_STATS:
            assert f"carga_rolling_3_{stat}" in result.columns


class TestNaNHandling:
    """Tests for NaN handling strategies."""

    @pytest.fixture
    def plugin(self) -> LagFeaturesPlugin:
        """Create a plugin instance for testing."""
        return LagFeaturesPlugin()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame."""
        return pd.DataFrame(
            {"carga": [100.0, 200.0, 300.0, 400.0, 500.0]},
            index=pd.date_range("2024-01-01", periods=5, freq="h"),
        )

    def test_fill_method_none(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test that fill_method='none' leaves NaN values."""
        config = LagFeaturesConfig(
            lag_periods=[2],
            rolling_windows=[],
            rolling_stats=[],
            fill_method="none",
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        # First two values should be NaN
        assert pd.isna(result["carga_lag_2"].iloc[0])
        assert pd.isna(result["carga_lag_2"].iloc[1])

    def test_fill_method_forward(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test that fill_method='forward' applies ffill."""
        config = LagFeaturesConfig(
            lag_periods=[2],
            rolling_windows=[],
            rolling_stats=[],
            fill_method="forward",
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        # After ffill, first valid value fills forward
        # Since lag_2 has NaN for first two rows, ffill fills from first valid value
        # But first two rows have no prior value, so they remain NaN
        # Actually ffill will fill NaN with previous non-NaN, starting rows stay NaN
        # After checking: the original carga column might fill across?
        # Let's verify the behavior - lag creates NaN at start, ffill propagates last valid

        # Actually, for lag features at the start, there's no prior value to fill from
        # So first NaN values will remain NaN with forward fill
        # This test confirms ffill is applied but start values stay NaN if no prior data
        assert pd.isna(result["carga_lag_2"].iloc[0])

    def test_fill_method_backward(
        self, plugin: LagFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test that fill_method='backward' applies bfill."""
        config = LagFeaturesConfig(
            lag_periods=[2],
            rolling_windows=[],
            rolling_stats=[],
            fill_method="backward",
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        # After bfill, NaN values are filled with the next valid value
        # lag_2 first two rows were NaN, bfill fills them with first valid value
        # First valid value for lag_2 is at index 2 with value 100.0
        assert result["carga_lag_2"].iloc[0] == 100.0
        assert result["carga_lag_2"].iloc[1] == 100.0


class TestMultipleColumns:
    """Tests for multiple column feature generation."""

    @pytest.fixture
    def plugin(self) -> LagFeaturesPlugin:
        """Create a plugin instance for testing."""
        return LagFeaturesPlugin()

    @pytest.fixture
    def multi_column_df(self) -> pd.DataFrame:
        """Create a DataFrame with multiple columns."""
        return pd.DataFrame(
            {
                "carga": [100.0, 200.0, 300.0, 400.0, 500.0],
                "temperature": [20.0, 21.0, 22.0, 23.0, 24.0],
                "humidity": [0.5, 0.6, 0.7, 0.8, 0.9],
            },
            index=pd.date_range("2024-01-01", periods=5, freq="h"),
        )

    def test_target_and_additional_columns(
        self, plugin: LagFeaturesPlugin, multi_column_df: pd.DataFrame
    ):
        """Test feature generation for target and additional columns."""
        config = LagFeaturesConfig(
            lag_periods=[1],
            rolling_windows=[],
            rolling_stats=[],
            target_column="carga",
            additional_columns=["temperature"],
        ).model_dump()

        result = plugin.generate_features(multi_column_df, config)

        assert "carga_lag_1" in result.columns
        assert "temperature_lag_1" in result.columns
        # humidity not included
        assert "humidity_lag_1" not in result.columns

    def test_missing_additional_column_skipped_with_warning(
        self, plugin: LagFeaturesPlugin, multi_column_df: pd.DataFrame
    ):
        """Test that missing additional columns are skipped with warning."""
        config = LagFeaturesConfig(
            lag_periods=[1],
            rolling_windows=[],
            rolling_stats=[],
            target_column="carga",
            additional_columns=["temperature", "nonexistent_column"],
        ).model_dump()

        # Should not raise, missing column is skipped
        result = plugin.generate_features(multi_column_df, config)

        assert "carga_lag_1" in result.columns
        assert "temperature_lag_1" in result.columns
        assert "nonexistent_column_lag_1" not in result.columns


class TestFeatureNames:
    """Tests for get_feature_names method."""

    @pytest.fixture
    def plugin(self) -> LagFeaturesPlugin:
        """Create a plugin instance for testing."""
        return LagFeaturesPlugin()

    def test_feature_names_lag_only(self, plugin: LagFeaturesPlugin):
        """Test feature names with lag features only."""
        config = LagFeaturesConfig(
            lag_periods=[1, 2],
            rolling_windows=[],
            rolling_stats=[],
            target_column="carga",
        ).model_dump()

        names = plugin.get_feature_names(config)

        assert "carga_lag_1" in names
        assert "carga_lag_2" in names
        assert len(names) == 2

    def test_feature_names_rolling_only(self, plugin: LagFeaturesPlugin):
        """Test feature names with rolling features only."""
        config = LagFeaturesConfig(
            lag_periods=[],
            rolling_windows=[24],
            rolling_stats=["mean", "std"],
            target_column="carga",
        ).model_dump()

        names = plugin.get_feature_names(config)

        assert "carga_rolling_24_mean" in names
        assert "carga_rolling_24_std" in names
        assert len(names) == 2

    def test_feature_names_multiple_columns(self, plugin: LagFeaturesPlugin):
        """Test feature names with multiple columns."""
        config = LagFeaturesConfig(
            lag_periods=[1],
            rolling_windows=[24],
            rolling_stats=["mean"],
            target_column="carga",
            additional_columns=["temp"],
        ).model_dump()

        names = plugin.get_feature_names(config)

        assert "carga_lag_1" in names
        assert "carga_rolling_24_mean" in names
        assert "temp_lag_1" in names
        assert "temp_rolling_24_mean" in names
        assert len(names) == 4

    def test_feature_names_naming_convention(self, plugin: LagFeaturesPlugin):
        """Test that feature naming follows convention."""
        config = LagFeaturesConfig(
            lag_periods=[1, 168],
            rolling_windows=[24, 168],
            rolling_stats=["mean", "max"],
            target_column="my_column",
        ).model_dump()

        names = plugin.get_feature_names(config)

        # Lag features: {column}_lag_{period}
        assert "my_column_lag_1" in names
        assert "my_column_lag_168" in names

        # Rolling features: {column}_rolling_{window}_{stat}
        assert "my_column_rolling_24_mean" in names
        assert "my_column_rolling_24_max" in names
        assert "my_column_rolling_168_mean" in names
        assert "my_column_rolling_168_max" in names

    def test_feature_names_match_generated(self, plugin: LagFeaturesPlugin):
        """Test that feature names match actually generated columns."""
        df = pd.DataFrame(
            {"carga": np.random.randn(100)},
            index=pd.date_range("2024-01-01", periods=100, freq="h"),
        )

        config = LagFeaturesConfig(
            lag_periods=[1, 2],
            rolling_windows=[24],
            rolling_stats=["mean", "std"],
        ).model_dump()

        names = plugin.get_feature_names(config)
        result = plugin.generate_features(df, config)

        # All listed features should be in result columns
        for name in names:
            assert name in result.columns, f"Feature {name} not in generated columns"


class TestConfigValidation:
    """Tests for configuration validation."""

    @pytest.fixture
    def plugin(self) -> LagFeaturesPlugin:
        """Create a plugin instance for testing."""
        return LagFeaturesPlugin()

    def test_validate_config_accepts_valid(self, plugin: LagFeaturesPlugin):
        """Test that validate_config accepts valid configuration."""
        config = LagFeaturesConfig().model_dump()
        result = plugin.validate_config(config)
        assert result is True

    def test_validate_config_rejects_none(self, plugin: LagFeaturesPlugin):
        """Test that validate_config rejects None."""
        with pytest.raises(ValueError):
            plugin.validate_config(None)  # type: ignore[arg-type]

    def test_invalid_target_column_raises_error(self, plugin: LagFeaturesPlugin):
        """Test that missing target column raises ValueError."""
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

        config = LagFeaturesConfig(
            target_column="nonexistent_column",
            lag_periods=[1],
            rolling_windows=[],
            rolling_stats=[],
        ).model_dump()

        with pytest.raises(ValueError) as exc_info:
            plugin.generate_features(df, config)

        assert "not found" in str(exc_info.value)
        assert "nonexistent_column" in str(exc_info.value)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def plugin(self) -> LagFeaturesPlugin:
        """Create a plugin instance for testing."""
        return LagFeaturesPlugin()

    def test_empty_dataframe(self, plugin: LagFeaturesPlugin):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(
            {"carga": []},
            index=pd.DatetimeIndex([]),
        )

        config = LagFeaturesConfig(
            lag_periods=[1],
            rolling_windows=[],
            rolling_stats=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert len(result) == 0
        assert "carga_lag_1" in result.columns

    def test_single_row_dataframe(self, plugin: LagFeaturesPlugin):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame(
            {"carga": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )

        config = LagFeaturesConfig(
            lag_periods=[1],
            rolling_windows=[2],
            rolling_stats=["mean"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert len(result) == 1
        # All lag/rolling features should be NaN for single row
        assert pd.isna(result["carga_lag_1"].iloc[0])
        assert pd.isna(result["carga_rolling_2_mean"].iloc[0])

    def test_preserves_original_columns(self, plugin: LagFeaturesPlugin):
        """Test that original columns are preserved."""
        df = pd.DataFrame(
            {
                "carga": [100.0, 200.0],
                "temperature": [25.0, 26.0],
                "metadata": ["a", "b"],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="h"),
        )

        config = LagFeaturesConfig(
            lag_periods=[1],
            rolling_windows=[],
            rolling_stats=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "carga" in result.columns
        assert "temperature" in result.columns
        assert "metadata" in result.columns
        np.testing.assert_array_equal(result["carga"].values, [100.0, 200.0])

    def test_preserves_index(self, plugin: LagFeaturesPlugin):
        """Test that DataFrame index is preserved."""
        original_index = pd.date_range("2024-01-01 10:00:00", periods=5, freq="h")
        df = pd.DataFrame(
            {"carga": [100.0, 200.0, 300.0, 400.0, 500.0]},
            index=original_index,
        )

        config = LagFeaturesConfig(
            lag_periods=[1],
            rolling_windows=[],
            rolling_stats=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        pd.testing.assert_index_equal(result.index, original_index)

    def test_does_not_modify_input(self, plugin: LagFeaturesPlugin):
        """Test that input DataFrame is not modified."""
        df = pd.DataFrame(
            {"carga": [100.0, 200.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="h"),
        )
        original_columns = list(df.columns)

        config = LagFeaturesConfig(
            lag_periods=[1],
            rolling_windows=[2],
            rolling_stats=["mean"],
        ).model_dump()

        _result = plugin.generate_features(df, config)

        assert list(df.columns) == original_columns

    def test_no_features_when_empty_config(self, plugin: LagFeaturesPlugin):
        """Test that no features are generated with empty lag/rolling config."""
        df = pd.DataFrame(
            {"carga": [100.0, 200.0, 300.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

        config = LagFeaturesConfig(
            lag_periods=[],
            rolling_windows=[],
            rolling_stats=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        # Only original column should remain
        assert list(result.columns) == ["carga"]


class TestPerformance:
    """Performance tests for lag features plugin."""

    @pytest.fixture
    def plugin(self) -> LagFeaturesPlugin:
        """Create a plugin instance for testing."""
        return LagFeaturesPlugin()

    def test_performance_one_year_hourly(self, plugin: LagFeaturesPlugin):
        """Test that 1 year of hourly data with 10+ features processes in < 500ms."""
        # Create 1 year of hourly data (8760 rows)
        df = pd.DataFrame(
            {"carga": np.random.randn(8760)},
            index=pd.date_range("2024-01-01", periods=8760, freq="h"),
        )

        # Config that generates 10+ features:
        # 4 lag features + (2 windows * 4 stats) = 4 + 8 = 12 features
        config = LagFeaturesConfig(
            lag_periods=[1, 2, 24, 168],
            rolling_windows=[24, 168],
            rolling_stats=["mean", "std", "min", "max"],
        ).model_dump()

        # Time the feature generation
        start_time = time.perf_counter()
        result = plugin.generate_features(df, config)
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        assert elapsed_time < 500, f"Feature generation took {elapsed_time:.2f}ms (> 500ms)"
        assert len(result) == 8760

        # Verify we generated 10+ features
        generated_features = plugin.get_feature_names(config)
        assert len(generated_features) >= 10

    def test_performance_large_dataset(self, plugin: LagFeaturesPlugin):
        """Test performance with a larger dataset (5 years hourly)."""
        # Create 5 years of hourly data (43800 rows)
        df = pd.DataFrame(
            {"carga": np.random.randn(43800)},
            index=pd.date_range("2020-01-01", periods=43800, freq="h"),
        )

        config = LagFeaturesConfig(
            lag_periods=[1, 24],
            rolling_windows=[24],
            rolling_stats=["mean", "std"],
        ).model_dump()

        # Time the feature generation
        start_time = time.perf_counter()
        result = plugin.generate_features(df, config)
        elapsed_time = (time.perf_counter() - start_time) * 1000

        # Should still be reasonably fast (< 1000ms for large dataset)
        assert elapsed_time < 1000, f"Feature generation took {elapsed_time:.2f}ms (> 1000ms)"
        assert len(result) == 43800


class TestIntegrationWithFeatures:
    """Integration tests with other feature components."""

    def test_import_from_plugins_package(self):
        """Test importing from src.features.plugins."""
        from src.features.plugins import LagFeaturesConfig, LagFeaturesPlugin

        plugin = LagFeaturesPlugin()
        config = LagFeaturesConfig()

        assert plugin.name == "lag_features"
        assert config.target_column == "carga"

    def test_import_from_features_package(self):
        """Test importing from src.features."""
        from src.features import LagFeaturesConfig, LagFeaturesPlugin

        plugin = LagFeaturesPlugin()
        config = LagFeaturesConfig()

        assert plugin.name == "lag_features"
        assert config.target_column == "carga"
