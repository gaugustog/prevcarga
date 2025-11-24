"""Tests for CyclicalEncodingPlugin.

This module contains comprehensive tests for the CyclicalEncodingPlugin,
including basic sin/cos encoding, mathematical properties verification,
configuration validation, and performance tests.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.plugins.cyclical import (
    CyclicalEncodingConfig,
    CyclicalEncodingPlugin,
)


class TestCyclicalEncodingConfig:
    """Tests for CyclicalEncodingConfig Pydantic model."""

    def test_default_config(self):
        """Test that default configuration has expected values."""
        config = CyclicalEncodingConfig()

        assert config.features == {}
        assert config.keep_original is True
        assert config.suffix_sin == "_sin"
        assert config.suffix_cos == "_cos"

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = CyclicalEncodingConfig(
            features={"hour": 24, "day_of_week": 7},
            keep_original=False,
            suffix_sin="_sine",
            suffix_cos="_cosine",
        )

        assert config.features == {"hour": 24, "day_of_week": 7}
        assert config.keep_original is False
        assert config.suffix_sin == "_sine"
        assert config.suffix_cos == "_cosine"

    def test_config_model_dump(self):
        """Test that config can be serialized to dict."""
        config = CyclicalEncodingConfig(features={"hour": 24})
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert "features" in config_dict
        assert "keep_original" in config_dict
        assert "suffix_sin" in config_dict
        assert "suffix_cos" in config_dict

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValueError):
            CyclicalEncodingConfig(unknown_field="value")  # type: ignore[call-arg]

    def test_invalid_period_zero_rejected(self):
        """Test that zero period is rejected."""
        with pytest.raises(ValueError) as exc_info:
            CyclicalEncodingConfig(features={"hour": 0})

        assert "positive" in str(exc_info.value).lower()

    def test_invalid_period_negative_rejected(self):
        """Test that negative period is rejected."""
        with pytest.raises(ValueError) as exc_info:
            CyclicalEncodingConfig(features={"hour": -24})

        assert "positive" in str(exc_info.value).lower()

    def test_multiple_features_one_invalid(self):
        """Test that validation fails if any period is invalid."""
        with pytest.raises(ValueError) as exc_info:
            CyclicalEncodingConfig(features={"hour": 24, "day": 0, "month": 12})

        assert "day" in str(exc_info.value)

    def test_custom_suffixes(self):
        """Test custom suffix configuration."""
        config = CyclicalEncodingConfig(
            features={"hour": 24},
            suffix_sin="_s",
            suffix_cos="_c",
        )

        assert config.suffix_sin == "_s"
        assert config.suffix_cos == "_c"


class TestCyclicalEncodingPlugin:
    """Tests for CyclicalEncodingPlugin base functionality."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame with hour and day_of_week columns."""
        return pd.DataFrame({
            "hour": [0, 6, 12, 18, 23],
            "day_of_week": [0, 1, 2, 3, 4],
            "load": [100.0, 200.0, 300.0, 400.0, 500.0],
        })

    @pytest.fixture
    def default_config(self) -> dict[str, Any]:
        """Return default configuration as dict."""
        return CyclicalEncodingConfig(features={"hour": 24}).model_dump()

    def test_plugin_name(self, plugin: CyclicalEncodingPlugin):
        """Test that plugin name is correct."""
        assert plugin.name == "cyclical_encoding"

    def test_plugin_version(self, plugin: CyclicalEncodingPlugin):
        """Test that plugin version is correct."""
        assert plugin.version == "1.0.0"

    def test_get_metadata(self, plugin: CyclicalEncodingPlugin):
        """Test that get_metadata returns correct values."""
        metadata = plugin.get_metadata()

        assert metadata["name"] == "cyclical_encoding"
        assert metadata["version"] == "1.0.0"
        assert metadata["class"] == "CyclicalEncodingPlugin"

    def test_repr(self, plugin: CyclicalEncodingPlugin):
        """Test plugin string representation."""
        repr_str = repr(plugin)

        assert "CyclicalEncodingPlugin" in repr_str
        assert "cyclical_encoding" in repr_str
        assert "1.0.0" in repr_str


class TestBasicSinCosEncoding:
    """Tests for basic sin/cos encoding functionality."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_basic_encoding_hour(self, plugin: CyclicalEncodingPlugin):
        """Test basic sine/cosine encoding for hours."""
        df = pd.DataFrame({"hour": [0, 6, 12, 18]})
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()

        result = plugin.generate_features(df, config)

        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

        # hour=0: sin=0, cos=1
        np.testing.assert_almost_equal(result["hour_sin"].values[0], 0.0, decimal=10)
        np.testing.assert_almost_equal(result["hour_cos"].values[0], 1.0, decimal=10)

        # hour=6 (quarter period): sin=1, cos=0
        np.testing.assert_almost_equal(result["hour_sin"].values[1], 1.0, decimal=10)
        np.testing.assert_almost_equal(result["hour_cos"].values[1], 0.0, decimal=10)

        # hour=12 (half period): sin=0, cos=-1
        np.testing.assert_almost_equal(result["hour_sin"].values[2], 0.0, decimal=10)
        np.testing.assert_almost_equal(result["hour_cos"].values[2], -1.0, decimal=10)

        # hour=18 (three-quarter period): sin=-1, cos=0
        np.testing.assert_almost_equal(result["hour_sin"].values[3], -1.0, decimal=10)
        np.testing.assert_almost_equal(result["hour_cos"].values[3], 0.0, decimal=10)

    def test_encoding_day_of_week(self, plugin: CyclicalEncodingPlugin):
        """Test encoding for day of week (period=7)."""
        df = pd.DataFrame({"day_of_week": [0, 1, 2, 3, 4, 5, 6]})
        config = CyclicalEncodingConfig(features={"day_of_week": 7}).model_dump()

        result = plugin.generate_features(df, config)

        assert "day_of_week_sin" in result.columns
        assert "day_of_week_cos" in result.columns

        # day=0: sin=0, cos=1
        np.testing.assert_almost_equal(
            result["day_of_week_sin"].values[0], 0.0, decimal=10
        )
        np.testing.assert_almost_equal(
            result["day_of_week_cos"].values[0], 1.0, decimal=10
        )

    def test_encoding_multiple_features(self, plugin: CyclicalEncodingPlugin):
        """Test encoding multiple features simultaneously."""
        df = pd.DataFrame({
            "hour": [0, 12],
            "day_of_week": [0, 3],
            "month": [1, 7],
        })
        config = CyclicalEncodingConfig(
            features={"hour": 24, "day_of_week": 7, "month": 12}
        ).model_dump()

        result = plugin.generate_features(df, config)

        # Check all encoded columns exist
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "day_of_week_sin" in result.columns
        assert "day_of_week_cos" in result.columns
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns


class TestKeepOriginalFlag:
    """Tests for keep_original configuration option."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_keep_original_true(self, plugin: CyclicalEncodingPlugin):
        """Test that original columns are kept when keep_original=True."""
        df = pd.DataFrame({"hour": [0, 6, 12]})
        config = CyclicalEncodingConfig(
            features={"hour": 24},
            keep_original=True,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "hour" in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_keep_original_false(self, plugin: CyclicalEncodingPlugin):
        """Test that original columns are removed when keep_original=False."""
        df = pd.DataFrame({"hour": [0, 6, 12], "other": [1, 2, 3]})
        config = CyclicalEncodingConfig(
            features={"hour": 24},
            keep_original=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "hour" not in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "other" in result.columns  # Unrelated columns should be kept

    def test_keep_original_false_multiple_features(self, plugin: CyclicalEncodingPlugin):
        """Test keep_original=False with multiple features."""
        df = pd.DataFrame({
            "hour": [0, 6],
            "day_of_week": [0, 3],
            "load": [100.0, 200.0],
        })
        config = CyclicalEncodingConfig(
            features={"hour": 24, "day_of_week": 7},
            keep_original=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "hour" not in result.columns
        assert "day_of_week" not in result.columns
        assert "load" in result.columns


class TestValueRanges:
    """Tests for verifying encoded value ranges."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_values_in_range_minus_one_to_one(self, plugin: CyclicalEncodingPlugin):
        """Test that sin/cos values are in range [-1, 1]."""
        # Create all 24 hours
        df = pd.DataFrame({"hour": list(range(24))})
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()

        result = plugin.generate_features(df, config)

        assert result["hour_sin"].min() >= -1.0
        assert result["hour_sin"].max() <= 1.0
        assert result["hour_cos"].min() >= -1.0
        assert result["hour_cos"].max() <= 1.0

    def test_values_in_range_large_dataset(self, plugin: CyclicalEncodingPlugin):
        """Test value ranges with a larger dataset."""
        # Create data with various values
        df = pd.DataFrame({
            "hour": np.tile(range(24), 100),
            "day": np.tile(range(7), 343)[:2400],
        })
        config = CyclicalEncodingConfig(
            features={"hour": 24, "day": 7}
        ).model_dump()

        result = plugin.generate_features(df, config)

        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
            assert result[col].min() >= -1.0, f"{col} min < -1"
            assert result[col].max() <= 1.0, f"{col} max > 1"


class TestPythagoreanIdentity:
    """Tests for sin^2 + cos^2 = 1 property."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_sin_squared_plus_cos_squared_equals_one(
        self, plugin: CyclicalEncodingPlugin
    ):
        """Test that sin^2 + cos^2 = 1 for all values."""
        df = pd.DataFrame({"hour": list(range(24))})
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()

        result = plugin.generate_features(df, config)

        identity = result["hour_sin"] ** 2 + result["hour_cos"] ** 2
        np.testing.assert_array_almost_equal(identity.values, np.ones(24), decimal=10)

    def test_pythagorean_identity_various_periods(
        self, plugin: CyclicalEncodingPlugin
    ):
        """Test Pythagorean identity for various periods."""
        periods = [7, 12, 24, 52, 365]

        for period in periods:
            df = pd.DataFrame({"value": list(range(period))})
            config = CyclicalEncodingConfig(features={"value": period}).model_dump()

            result = plugin.generate_features(df, config)

            identity = result["value_sin"] ** 2 + result["value_cos"] ** 2
            np.testing.assert_array_almost_equal(
                identity.values,
                np.ones(period),
                decimal=10,
                err_msg=f"Identity failed for period={period}",
            )

    def test_pythagorean_identity_fractional_values(
        self, plugin: CyclicalEncodingPlugin
    ):
        """Test Pythagorean identity with fractional values."""
        df = pd.DataFrame({"value": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]})
        config = CyclicalEncodingConfig(features={"value": 3}).model_dump()

        result = plugin.generate_features(df, config)

        identity = result["value_sin"] ** 2 + result["value_cos"] ** 2
        np.testing.assert_array_almost_equal(identity.values, np.ones(6), decimal=10)


class TestCyclicDistance:
    """Tests for cyclic distance property (hour 0 close to hour 23)."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_hour_0_close_to_hour_23(self, plugin: CyclicalEncodingPlugin):
        """Test that hour 0 and hour 23 are close in encoded space."""
        df = pd.DataFrame({"hour": [0, 1, 12, 22, 23]})
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()

        result = plugin.generate_features(df, config)

        # Get encoded values
        hour_0 = np.array([
            result["hour_sin"].values[0],
            result["hour_cos"].values[0],
        ])
        hour_1 = np.array([
            result["hour_sin"].values[1],
            result["hour_cos"].values[1],
        ])
        hour_12 = np.array([
            result["hour_sin"].values[2],
            result["hour_cos"].values[2],
        ])
        hour_23 = np.array([
            result["hour_sin"].values[4],
            result["hour_cos"].values[4],
        ])

        # Calculate Euclidean distances
        dist_0_1 = np.linalg.norm(hour_0 - hour_1)
        dist_0_12 = np.linalg.norm(hour_0 - hour_12)
        dist_0_23 = np.linalg.norm(hour_0 - hour_23)

        # Hour 0 and 23 should be closer than hour 0 and 12
        assert dist_0_23 < dist_0_12, (
            f"Hour 0 to 23 ({dist_0_23:.4f}) should be closer than "
            f"hour 0 to 12 ({dist_0_12:.4f})"
        )

        # Hour 0 and 23 should be similar distance to hour 0 and 1
        np.testing.assert_almost_equal(
            dist_0_23, dist_0_1, decimal=2,
            err_msg="Hour 0-23 distance should be similar to hour 0-1 distance"
        )

    def test_day_6_close_to_day_0(self, plugin: CyclicalEncodingPlugin):
        """Test that day 6 (Sunday) is close to day 0 (Monday) in encoded space."""
        df = pd.DataFrame({"day": [0, 1, 3, 5, 6]})
        config = CyclicalEncodingConfig(features={"day": 7}).model_dump()

        result = plugin.generate_features(df, config)

        day_0 = np.array([result["day_sin"].values[0], result["day_cos"].values[0]])
        day_3 = np.array([result["day_sin"].values[2], result["day_cos"].values[2]])
        day_6 = np.array([result["day_sin"].values[4], result["day_cos"].values[4]])

        dist_0_3 = np.linalg.norm(day_0 - day_3)
        dist_0_6 = np.linalg.norm(day_0 - day_6)

        # Day 0 and 6 should be closer than day 0 and 3
        assert dist_0_6 < dist_0_3


class TestCustomSuffixes:
    """Tests for custom suffix configuration."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_custom_suffixes(self, plugin: CyclicalEncodingPlugin):
        """Test that custom suffixes are applied correctly."""
        df = pd.DataFrame({"hour": [0, 6, 12]})
        config = CyclicalEncodingConfig(
            features={"hour": 24},
            suffix_sin="_sine_enc",
            suffix_cos="_cosine_enc",
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "hour_sine_enc" in result.columns
        assert "hour_cosine_enc" in result.columns
        assert "hour_sin" not in result.columns
        assert "hour_cos" not in result.columns

    def test_short_suffixes(self, plugin: CyclicalEncodingPlugin):
        """Test single character suffixes."""
        df = pd.DataFrame({"hour": [0, 6, 12]})
        config = CyclicalEncodingConfig(
            features={"hour": 24},
            suffix_sin="_s",
            suffix_cos="_c",
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "hour_s" in result.columns
        assert "hour_c" in result.columns


class TestFeatureNamesGeneration:
    """Tests for get_feature_names method."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_feature_names_with_keep_original_true(
        self, plugin: CyclicalEncodingPlugin
    ):
        """Test feature names when keep_original=True."""
        config = CyclicalEncodingConfig(
            features={"hour": 24, "day": 7},
            keep_original=True,
        ).model_dump()

        names = plugin.get_feature_names(config)

        assert "hour" in names
        assert "day" in names
        assert "hour_sin" in names
        assert "hour_cos" in names
        assert "day_sin" in names
        assert "day_cos" in names
        assert len(names) == 6  # 2 original + 4 encoded

    def test_feature_names_with_keep_original_false(
        self, plugin: CyclicalEncodingPlugin
    ):
        """Test feature names when keep_original=False."""
        config = CyclicalEncodingConfig(
            features={"hour": 24, "day": 7},
            keep_original=False,
        ).model_dump()

        names = plugin.get_feature_names(config)

        assert "hour" not in names
        assert "day" not in names
        assert "hour_sin" in names
        assert "hour_cos" in names
        assert "day_sin" in names
        assert "day_cos" in names
        assert len(names) == 4  # Only encoded

    def test_feature_names_custom_suffixes(self, plugin: CyclicalEncodingPlugin):
        """Test feature names with custom suffixes."""
        config = CyclicalEncodingConfig(
            features={"hour": 24},
            suffix_sin="_s",
            suffix_cos="_c",
            keep_original=False,
        ).model_dump()

        names = plugin.get_feature_names(config)

        assert "hour_s" in names
        assert "hour_c" in names
        assert "hour_sin" not in names

    def test_feature_names_empty_features(self, plugin: CyclicalEncodingPlugin):
        """Test feature names with empty features dict."""
        config = CyclicalEncodingConfig(features={}).model_dump()

        names = plugin.get_feature_names(config)

        assert names == []

    def test_feature_names_match_generated(self, plugin: CyclicalEncodingPlugin):
        """Test that feature names match actually generated columns."""
        df = pd.DataFrame({"hour": [0, 6, 12], "day": [0, 1, 2]})
        config = CyclicalEncodingConfig(
            features={"hour": 24, "day": 7},
            keep_original=True,
        ).model_dump()

        names = plugin.get_feature_names(config)
        result = plugin.generate_features(df, config)

        # All listed features should be in result columns
        for name in names:
            assert name in result.columns, f"Feature {name} not in generated columns"


class TestMissingFeatureError:
    """Tests for error handling when feature is missing."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_missing_feature_raises_valueerror(self, plugin: CyclicalEncodingPlugin):
        """Test that ValueError is raised when feature doesn't exist."""
        df = pd.DataFrame({"day": [0, 1, 2]})
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()

        with pytest.raises(ValueError) as exc_info:
            plugin.generate_features(df, config)

        assert "hour" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()

    def test_partial_missing_features(self, plugin: CyclicalEncodingPlugin):
        """Test error when some features exist and some don't."""
        df = pd.DataFrame({"hour": [0, 6, 12]})
        config = CyclicalEncodingConfig(
            features={"hour": 24, "day": 7}
        ).model_dump()

        with pytest.raises(ValueError) as exc_info:
            plugin.generate_features(df, config)

        assert "day" in str(exc_info.value)


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_empty_features_dict(self, plugin: CyclicalEncodingPlugin):
        """Test handling of empty features dictionary."""
        df = pd.DataFrame({"hour": [0, 6, 12], "load": [100.0, 200.0, 300.0]})
        config = CyclicalEncodingConfig(features={}).model_dump()

        result = plugin.generate_features(df, config)

        # Should return a copy without modifications
        assert list(result.columns) == ["hour", "load"]
        np.testing.assert_array_equal(result["hour"].values, [0, 6, 12])

    def test_value_zero_sin_zero_cos_one(self, plugin: CyclicalEncodingPlugin):
        """Test that value=0 gives sin=0, cos=1."""
        df = pd.DataFrame({"value": [0]})
        config = CyclicalEncodingConfig(features={"value": 24}).model_dump()

        result = plugin.generate_features(df, config)

        np.testing.assert_almost_equal(result["value_sin"].values[0], 0.0, decimal=10)
        np.testing.assert_almost_equal(result["value_cos"].values[0], 1.0, decimal=10)

    def test_value_equals_period_sin_approx_zero_cos_approx_one(
        self, plugin: CyclicalEncodingPlugin
    ):
        """Test that value=period gives sin~0, cos~1 (full cycle)."""
        df = pd.DataFrame({"value": [24]})
        config = CyclicalEncodingConfig(features={"value": 24}).model_dump()

        result = plugin.generate_features(df, config)

        # After full cycle, should return to start (sin=0, cos=1)
        np.testing.assert_almost_equal(result["value_sin"].values[0], 0.0, decimal=10)
        np.testing.assert_almost_equal(result["value_cos"].values[0], 1.0, decimal=10)

    def test_preserves_other_columns(self, plugin: CyclicalEncodingPlugin):
        """Test that non-encoded columns are preserved."""
        df = pd.DataFrame({
            "hour": [0, 6, 12],
            "load": [100.0, 200.0, 300.0],
            "temperature": [25.0, 26.0, 27.0],
        })
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()

        result = plugin.generate_features(df, config)

        assert "load" in result.columns
        assert "temperature" in result.columns
        np.testing.assert_array_equal(result["load"].values, [100.0, 200.0, 300.0])

    def test_preserves_index(self, plugin: CyclicalEncodingPlugin):
        """Test that DataFrame index is preserved."""
        original_index = pd.date_range("2024-01-01", periods=3, freq="h")
        df = pd.DataFrame({"hour": [0, 1, 2]}, index=original_index)
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()

        result = plugin.generate_features(df, config)

        pd.testing.assert_index_equal(result.index, original_index)

    def test_does_not_modify_input(self, plugin: CyclicalEncodingPlugin):
        """Test that input DataFrame is not modified."""
        df = pd.DataFrame({"hour": [0, 6, 12]})
        original_columns = list(df.columns)
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()

        _result = plugin.generate_features(df, config)

        assert list(df.columns) == original_columns

    def test_single_row_dataframe(self, plugin: CyclicalEncodingPlugin):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({"hour": [12]})
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()

        result = plugin.generate_features(df, config)

        assert len(result) == 1
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_empty_dataframe(self, plugin: CyclicalEncodingPlugin):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"hour": pd.Series([], dtype=float)})
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()

        result = plugin.generate_features(df, config)

        assert len(result) == 0
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_float_values(self, plugin: CyclicalEncodingPlugin):
        """Test encoding with float values."""
        df = pd.DataFrame({"value": [0.0, 0.5, 1.0, 1.5, 2.0]})
        config = CyclicalEncodingConfig(features={"value": 2}).model_dump()

        result = plugin.generate_features(df, config)

        # value=0.5 is quarter period: sin=1, cos=0
        np.testing.assert_almost_equal(result["value_sin"].values[1], 1.0, decimal=10)
        np.testing.assert_almost_equal(result["value_cos"].values[1], 0.0, decimal=10)


class TestConfigValidation:
    """Tests for configuration validation."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_validate_config_accepts_valid(self, plugin: CyclicalEncodingPlugin):
        """Test that validate_config accepts valid configuration."""
        config = CyclicalEncodingConfig(features={"hour": 24}).model_dump()
        result = plugin.validate_config(config)
        assert result is True

    def test_validate_config_rejects_none(self, plugin: CyclicalEncodingPlugin):
        """Test that validate_config rejects None."""
        with pytest.raises(ValueError):
            plugin.validate_config(None)  # type: ignore[arg-type]

    def test_validate_config_rejects_invalid_period(
        self, plugin: CyclicalEncodingPlugin
    ):
        """Test that validate_config rejects invalid period."""
        config = {"features": {"hour": -1}}

        with pytest.raises(ValueError):
            plugin.validate_config(config)


class TestPerformance:
    """Performance tests for cyclical encoding plugin."""

    @pytest.fixture
    def plugin(self) -> CyclicalEncodingPlugin:
        """Create a plugin instance for testing."""
        return CyclicalEncodingPlugin()

    def test_performance_one_year_hourly(self, plugin: CyclicalEncodingPlugin):
        """Test that 1 year of hourly data processes in < 10ms."""
        # Create 1 year of hourly data (8760 rows)
        df = pd.DataFrame({
            "hour": np.tile(range(24), 365),
            "day_of_week": np.tile(range(7), 1252)[:8760],
        })

        config = CyclicalEncodingConfig(
            features={"hour": 24, "day_of_week": 7}
        ).model_dump()

        # Time the feature generation
        start_time = time.perf_counter()
        result = plugin.generate_features(df, config)
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        assert elapsed_time < 10, f"Feature generation took {elapsed_time:.2f}ms (> 10ms)"
        assert len(result) == 8760

    def test_performance_large_dataset(self, plugin: CyclicalEncodingPlugin):
        """Test performance with a larger dataset (5 years hourly)."""
        # Create 5 years of hourly data (43800 rows)
        df = pd.DataFrame({
            "hour": np.tile(range(24), 1825),
            "day_of_week": np.tile(range(7), 6258)[:43800],
            "month": np.tile(range(12), 3650)[:43800],
        })

        config = CyclicalEncodingConfig(
            features={"hour": 24, "day_of_week": 7, "month": 12}
        ).model_dump()

        # Time the feature generation
        start_time = time.perf_counter()
        result = plugin.generate_features(df, config)
        elapsed_time = (time.perf_counter() - start_time) * 1000

        # Should still be reasonably fast (< 50ms)
        assert elapsed_time < 50, f"Feature generation took {elapsed_time:.2f}ms (> 50ms)"
        assert len(result) == 43800
