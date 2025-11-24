"""Tests for TemporalFeaturesPlugin.

This module contains comprehensive tests for the TemporalFeaturesPlugin,
including basic temporal features, cyclical encodings, Southern Hemisphere
seasons, timezone handling, and performance tests.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.plugins.temporal import (
    DEFAULT_BASIC_FEATURES,
    SOUTHERN_HEMISPHERE_SEASONS,
    TemporalFeaturesConfig,
    TemporalFeaturesPlugin,
)


class TestTemporalFeaturesConfig:
    """Tests for TemporalFeaturesConfig Pydantic model."""

    def test_default_config(self):
        """Test that default configuration has expected values."""
        config = TemporalFeaturesConfig()

        assert config.include_cyclical is True
        assert config.include_season is True
        assert config.timezone == "America/Sao_Paulo"
        assert config.features == DEFAULT_BASIC_FEATURES

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            timezone="UTC",
            features=["hour", "day_of_week"],
        )

        assert config.include_cyclical is False
        assert config.include_season is False
        assert config.timezone == "UTC"
        assert config.features == ["hour", "day_of_week"]

    def test_valid_timezone(self):
        """Test that valid timezones are accepted."""
        valid_timezones = [
            "America/Sao_Paulo",
            "UTC",
            "America/New_York",
            "Europe/London",
            "Asia/Tokyo",
        ]

        for tz in valid_timezones:
            config = TemporalFeaturesConfig(timezone=tz)
            assert config.timezone == tz

    def test_invalid_timezone_raises_error(self):
        """Test that invalid timezone raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            TemporalFeaturesConfig(timezone="Invalid/Timezone")

        assert "Invalid timezone" in str(exc_info.value)

    def test_invalid_feature_raises_error(self):
        """Test that invalid feature names raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            TemporalFeaturesConfig(features=["hour", "invalid_feature"])

        assert "Invalid features" in str(exc_info.value)

    def test_config_model_dump(self):
        """Test that config can be serialized to dict."""
        config = TemporalFeaturesConfig()
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert "include_cyclical" in config_dict
        assert "include_season" in config_dict
        assert "timezone" in config_dict
        assert "features" in config_dict

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValueError):
            TemporalFeaturesConfig(unknown_field="value")  # type: ignore[call-arg]


class TestTemporalFeaturesPlugin:
    """Tests for TemporalFeaturesPlugin base functionality."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame with DatetimeIndex."""
        return pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.date_range("2024-01-15 10:00:00", periods=3, freq="h"),
        )

    @pytest.fixture
    def default_config(self) -> dict[str, Any]:
        """Return default configuration as dict."""
        return TemporalFeaturesConfig().model_dump()

    def test_plugin_name(self, plugin: TemporalFeaturesPlugin):
        """Test that plugin name is correct."""
        assert plugin.name == "temporal_features"

    def test_plugin_version(self, plugin: TemporalFeaturesPlugin):
        """Test that plugin version is correct."""
        assert plugin.version == "1.0.0"

    def test_get_metadata(self, plugin: TemporalFeaturesPlugin):
        """Test that get_metadata returns correct values."""
        metadata = plugin.get_metadata()

        assert metadata["name"] == "temporal_features"
        assert metadata["version"] == "1.0.0"
        assert metadata["class"] == "TemporalFeaturesPlugin"

    def test_repr(self, plugin: TemporalFeaturesPlugin):
        """Test plugin string representation."""
        repr_str = repr(plugin)

        assert "TemporalFeaturesPlugin" in repr_str
        assert "temporal_features" in repr_str
        assert "1.0.0" in repr_str


class TestBasicTemporalFeatures:
    """Tests for basic temporal feature generation."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame spanning different times."""
        # Create data for Monday Jan 15, 2024 at 10:00, 11:00, 12:00
        return pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.date_range("2024-01-15 10:00:00", periods=3, freq="h"),
        )

    def test_hour_feature(self, plugin: TemporalFeaturesPlugin, sample_df: pd.DataFrame):
        """Test hour feature generation."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["hour"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "hour" in result.columns
        np.testing.assert_array_equal(result["hour"].values, [10, 11, 12])

    def test_day_of_week_feature(
        self, plugin: TemporalFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test day_of_week feature generation (Monday = 0)."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["day_of_week"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "day_of_week" in result.columns
        # Jan 15, 2024 is a Monday (0)
        np.testing.assert_array_equal(result["day_of_week"].values, [0, 0, 0])

    def test_day_of_month_feature(
        self, plugin: TemporalFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test day_of_month feature generation."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["day_of_month"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "day_of_month" in result.columns
        np.testing.assert_array_equal(result["day_of_month"].values, [15, 15, 15])

    def test_month_feature(self, plugin: TemporalFeaturesPlugin, sample_df: pd.DataFrame):
        """Test month feature generation."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["month"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "month" in result.columns
        np.testing.assert_array_equal(result["month"].values, [1, 1, 1])

    def test_quarter_feature(self, plugin: TemporalFeaturesPlugin, sample_df: pd.DataFrame):
        """Test quarter feature generation."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["quarter"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "quarter" in result.columns
        np.testing.assert_array_equal(result["quarter"].values, [1, 1, 1])

    def test_year_feature(self, plugin: TemporalFeaturesPlugin, sample_df: pd.DataFrame):
        """Test year feature generation."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["year"],
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        assert "year" in result.columns
        np.testing.assert_array_equal(result["year"].values, [2024, 2024, 2024])

    def test_all_basic_features(
        self, plugin: TemporalFeaturesPlugin, sample_df: pd.DataFrame
    ):
        """Test that all basic features are generated."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
        ).model_dump()

        result = plugin.generate_features(sample_df, config)

        for feature in DEFAULT_BASIC_FEATURES:
            assert feature in result.columns


class TestWeekendAndBusinessDay:
    """Tests for weekend and business day flag features."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    @pytest.fixture
    def week_df(self) -> pd.DataFrame:
        """Create a DataFrame spanning Mon-Sun."""
        # Jan 15-21, 2024: Mon, Tue, Wed, Thu, Fri, Sat, Sun
        return pd.DataFrame(
            {"load": [100.0] * 7},
            index=pd.date_range("2024-01-15", periods=7, freq="D"),
        )

    def test_is_weekend_feature(self, plugin: TemporalFeaturesPlugin, week_df: pd.DataFrame):
        """Test is_weekend flag (Sat=1, Sun=1, else=0)."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["is_weekend"],
        ).model_dump()

        result = plugin.generate_features(week_df, config)

        assert "is_weekend" in result.columns
        # Mon, Tue, Wed, Thu, Fri, Sat, Sun
        expected = [0, 0, 0, 0, 0, 1, 1]
        np.testing.assert_array_equal(result["is_weekend"].values, expected)

    def test_is_business_day_feature(
        self, plugin: TemporalFeaturesPlugin, week_df: pd.DataFrame
    ):
        """Test is_business_day flag (Mon-Fri=1, else=0)."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["is_business_day"],
        ).model_dump()

        result = plugin.generate_features(week_df, config)

        assert "is_business_day" in result.columns
        # Mon, Tue, Wed, Thu, Fri, Sat, Sun
        expected = [1, 1, 1, 1, 1, 0, 0]
        np.testing.assert_array_equal(result["is_business_day"].values, expected)

    def test_weekend_and_business_day_are_complementary(
        self, plugin: TemporalFeaturesPlugin, week_df: pd.DataFrame
    ):
        """Test that is_weekend + is_business_day = 1 for all rows."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["is_weekend", "is_business_day"],
        ).model_dump()

        result = plugin.generate_features(week_df, config)

        sum_values = result["is_weekend"].values + result["is_business_day"].values
        np.testing.assert_array_equal(sum_values, [1, 1, 1, 1, 1, 1, 1])


class TestSouthernHemisphereSeason:
    """Tests for Southern Hemisphere season feature."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    def test_season_mapping_dictionary(self):
        """Test that season mapping covers all 12 months."""
        assert len(SOUTHERN_HEMISPHERE_SEASONS) == 12
        for month in range(1, 13):
            assert month in SOUTHERN_HEMISPHERE_SEASONS

    def test_summer_months(self, plugin: TemporalFeaturesPlugin):
        """Test Summer season (Dec, Jan, Feb) -> 1."""
        # Create data for Dec, Jan, Feb
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.DatetimeIndex(
                ["2024-12-15", "2024-01-15", "2024-02-15"]
            ),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=True,
            features=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "season" in result.columns
        np.testing.assert_array_equal(result["season"].values, [1, 1, 1])

    def test_autumn_months(self, plugin: TemporalFeaturesPlugin):
        """Test Autumn season (Mar, Apr, May) -> 2."""
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.DatetimeIndex(
                ["2024-03-15", "2024-04-15", "2024-05-15"]
            ),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=True,
            features=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        np.testing.assert_array_equal(result["season"].values, [2, 2, 2])

    def test_winter_months(self, plugin: TemporalFeaturesPlugin):
        """Test Winter season (Jun, Jul, Aug) -> 3."""
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.DatetimeIndex(
                ["2024-06-15", "2024-07-15", "2024-08-15"]
            ),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=True,
            features=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        np.testing.assert_array_equal(result["season"].values, [3, 3, 3])

    def test_spring_months(self, plugin: TemporalFeaturesPlugin):
        """Test Spring season (Sep, Oct, Nov) -> 4."""
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.DatetimeIndex(
                ["2024-09-15", "2024-10-15", "2024-11-15"]
            ),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=True,
            features=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        np.testing.assert_array_equal(result["season"].values, [4, 4, 4])

    def test_all_months_seasons(self, plugin: TemporalFeaturesPlugin):
        """Test season values for all 12 months."""
        # Create data for all 12 months
        dates = [f"2024-{month:02d}-15" for month in range(1, 13)]
        df = pd.DataFrame(
            {"load": [100.0] * 12},
            index=pd.DatetimeIndex(dates),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=True,
            features=["month"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        # Expected seasons: Jan=1, Feb=1, Mar=2, Apr=2, May=2, Jun=3,
        # Jul=3, Aug=3, Sep=4, Oct=4, Nov=4, Dec=1
        expected = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
        np.testing.assert_array_equal(result["season"].values, expected)


class TestCyclicalEncoding:
    """Tests for cyclical sin/cos encoding features."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    def test_cyclical_features_generated(self, plugin: TemporalFeaturesPlugin):
        """Test that cyclical features are generated when enabled."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.date_range("2024-01-15 10:00:00", periods=1, freq="h"),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=True,
            include_season=False,
            features=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        expected_features = [
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
        ]
        for feat in expected_features:
            assert feat in result.columns

    def test_cyclical_values_in_range(self, plugin: TemporalFeaturesPlugin):
        """Test that sin/cos values are in range [-1, 1]."""
        # Create data spanning a full year
        df = pd.DataFrame(
            {"load": [100.0] * 365 * 24},
            index=pd.date_range("2024-01-01", periods=365 * 24, freq="h"),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=True,
            include_season=False,
            features=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        cyclical_cols = [
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
        ]

        for col in cyclical_cols:
            assert result[col].min() >= -1.0, f"{col} has value < -1"
            assert result[col].max() <= 1.0, f"{col} has value > 1"

    def test_sin_cos_identity(self, plugin: TemporalFeaturesPlugin):
        """Test that sin^2 + cos^2 = 1 for all cyclical encodings."""
        df = pd.DataFrame(
            {"load": [100.0] * 100},
            index=pd.date_range("2024-01-01", periods=100, freq="h"),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=True,
            include_season=False,
            features=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        # Test sin^2 + cos^2 = 1 for hour encoding
        hour_identity = result["hour_sin"] ** 2 + result["hour_cos"] ** 2
        np.testing.assert_array_almost_equal(hour_identity.values, np.ones(100))

        # Test for day_of_week encoding
        dow_identity = result["day_of_week_sin"] ** 2 + result["day_of_week_cos"] ** 2
        np.testing.assert_array_almost_equal(dow_identity.values, np.ones(100))

        # Test for month encoding
        month_identity = result["month_sin"] ** 2 + result["month_cos"] ** 2
        np.testing.assert_array_almost_equal(month_identity.values, np.ones(100))

    def test_hour_cyclical_values(self, plugin: TemporalFeaturesPlugin):
        """Test specific hour cyclical values."""
        # Create data at hours 0, 6, 12, 18
        df = pd.DataFrame(
            {"load": [100.0] * 4},
            index=pd.DatetimeIndex([
                "2024-01-15 00:00:00",
                "2024-01-15 06:00:00",
                "2024-01-15 12:00:00",
                "2024-01-15 18:00:00",
            ]),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=True,
            include_season=False,
            features=[],
        ).model_dump()

        result = plugin.generate_features(df, config)

        # hour=0: sin=0, cos=1
        # hour=6: sin=1, cos=0
        # hour=12: sin=0, cos=-1
        # hour=18: sin=-1, cos=0
        expected_sin = [0.0, 1.0, 0.0, -1.0]
        expected_cos = [1.0, 0.0, -1.0, 0.0]

        np.testing.assert_array_almost_equal(
            result["hour_sin"].values, expected_sin, decimal=10
        )
        np.testing.assert_array_almost_equal(
            result["hour_cos"].values, expected_cos, decimal=10
        )

    def test_cyclical_disabled(self, plugin: TemporalFeaturesPlugin):
        """Test that cyclical features are not generated when disabled."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.date_range("2024-01-15", periods=1, freq="h"),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["hour"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "hour_sin" not in result.columns
        assert "hour_cos" not in result.columns


class TestTimezoneHandling:
    """Tests for timezone handling."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    def test_naive_datetime_localization(self, plugin: TemporalFeaturesPlugin):
        """Test that naive datetimes are localized to configured timezone."""
        # Create naive datetime
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.date_range("2024-01-15 10:00:00", periods=1, freq="h"),
        )

        assert df.index.tz is None  # Confirm naive

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            timezone="America/Sao_Paulo",
            features=["hour"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        # Should have generated hour feature
        assert "hour" in result.columns
        assert result["hour"].values[0] == 10

    def test_aware_datetime_conversion(self, plugin: TemporalFeaturesPlugin):
        """Test that aware datetimes are converted to configured timezone."""
        # Create UTC datetime
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.date_range("2024-01-15 15:00:00", periods=1, freq="h", tz="UTC"),
        )

        assert df.index.tz is not None  # Confirm aware

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            timezone="America/Sao_Paulo",  # UTC-3
            features=["hour"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        # UTC 15:00 -> Sao Paulo 12:00 (UTC-3)
        assert result["hour"].values[0] == 12

    def test_different_timezones(self, plugin: TemporalFeaturesPlugin):
        """Test feature generation with different timezones."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.date_range("2024-01-15 12:00:00", periods=1, freq="h", tz="UTC"),
        )

        # Test with UTC
        config_utc = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            timezone="UTC",
            features=["hour"],
        ).model_dump()

        result_utc = plugin.generate_features(df, config_utc)
        assert result_utc["hour"].values[0] == 12

        # Test with Tokyo (UTC+9)
        config_tokyo = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            timezone="Asia/Tokyo",
            features=["hour"],
        ).model_dump()

        result_tokyo = plugin.generate_features(df, config_tokyo)
        assert result_tokyo["hour"].values[0] == 21  # UTC 12:00 -> Tokyo 21:00


class TestTimestampColumn:
    """Tests for DataFrame with timestamp column instead of DatetimeIndex."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    def test_timestamp_column_support(self, plugin: TemporalFeaturesPlugin):
        """Test that timestamp column is used when present."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-15 10:00:00", periods=3, freq="h"),
            "load": [100.0, 200.0, 300.0],
        })

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["hour"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "hour" in result.columns
        np.testing.assert_array_equal(result["hour"].values, [10, 11, 12])

    def test_no_datetime_raises_error(self, plugin: TemporalFeaturesPlugin):
        """Test that error is raised when no datetime source exists."""
        df = pd.DataFrame({
            "load": [100.0, 200.0, 300.0],
            "value": [1, 2, 3],
        })

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["hour"],
        ).model_dump()

        with pytest.raises(ValueError) as exc_info:
            plugin.generate_features(df, config)

        assert "DatetimeIndex" in str(exc_info.value)


class TestFeatureNames:
    """Tests for get_feature_names method."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    def test_feature_names_default_config(self, plugin: TemporalFeaturesPlugin):
        """Test feature names with default configuration."""
        config = TemporalFeaturesConfig().model_dump()
        names = plugin.get_feature_names(config)

        # Should include all basic features
        for feat in DEFAULT_BASIC_FEATURES:
            assert feat in names

        # Should include season
        assert "season" in names

        # Should include cyclical features
        cyclical = ["hour_sin", "hour_cos", "day_of_week_sin",
                    "day_of_week_cos", "month_sin", "month_cos"]
        for feat in cyclical:
            assert feat in names

    def test_feature_names_no_cyclical(self, plugin: TemporalFeaturesPlugin):
        """Test feature names without cyclical features."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=True,
        ).model_dump()
        names = plugin.get_feature_names(config)

        assert "hour_sin" not in names
        assert "hour_cos" not in names
        assert "season" in names

    def test_feature_names_no_season(self, plugin: TemporalFeaturesPlugin):
        """Test feature names without season feature."""
        config = TemporalFeaturesConfig(
            include_cyclical=True,
            include_season=False,
        ).model_dump()
        names = plugin.get_feature_names(config)

        assert "season" not in names
        assert "hour_sin" in names

    def test_feature_names_custom_features(self, plugin: TemporalFeaturesPlugin):
        """Test feature names with custom feature list."""
        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["hour", "day_of_week"],
        ).model_dump()
        names = plugin.get_feature_names(config)

        assert names == ["hour", "day_of_week"]

    def test_feature_names_match_generated(self, plugin: TemporalFeaturesPlugin):
        """Test that feature names match actually generated columns."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.date_range("2024-01-15", periods=1, freq="h"),
        )

        config = TemporalFeaturesConfig().model_dump()
        names = plugin.get_feature_names(config)
        result = plugin.generate_features(df, config)

        # All listed features should be in result columns
        for name in names:
            assert name in result.columns, f"Feature {name} not in generated columns"


class TestConfigValidation:
    """Tests for configuration validation."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    def test_validate_config_accepts_valid(self, plugin: TemporalFeaturesPlugin):
        """Test that validate_config accepts valid configuration."""
        config = TemporalFeaturesConfig().model_dump()
        result = plugin.validate_config(config)
        assert result is True

    def test_validate_config_rejects_none(self, plugin: TemporalFeaturesPlugin):
        """Test that validate_config rejects None."""
        with pytest.raises(ValueError):
            plugin.validate_config(None)  # type: ignore[arg-type]

    def test_validate_config_rejects_invalid_timezone(
        self, plugin: TemporalFeaturesPlugin
    ):
        """Test that validate_config rejects invalid timezone."""
        config = {"timezone": "Invalid/Timezone"}

        with pytest.raises(ValueError):
            plugin.validate_config(config)


class TestPerformance:
    """Performance tests for temporal features plugin."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    def test_performance_one_year_hourly(self, plugin: TemporalFeaturesPlugin):
        """Test that 1 year of hourly data processes in < 100ms."""
        # Create 1 year of hourly data (8760 rows)
        df = pd.DataFrame(
            {"load": np.random.randn(8760)},
            index=pd.date_range("2024-01-01", periods=8760, freq="h"),
        )

        config = TemporalFeaturesConfig().model_dump()

        # Time the feature generation
        start_time = time.perf_counter()
        _result = plugin.generate_features(df, config)
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        assert elapsed_time < 100, f"Feature generation took {elapsed_time:.2f}ms (> 100ms)"

    def test_performance_large_dataset(self, plugin: TemporalFeaturesPlugin):
        """Test performance with a larger dataset (5 years hourly)."""
        # Create 5 years of hourly data (43800 rows)
        df = pd.DataFrame(
            {"load": np.random.randn(43800)},
            index=pd.date_range("2020-01-01", periods=43800, freq="h"),
        )

        config = TemporalFeaturesConfig().model_dump()

        # Time the feature generation
        start_time = time.perf_counter()
        result = plugin.generate_features(df, config)
        elapsed_time = (time.perf_counter() - start_time) * 1000

        # Should still be reasonably fast (< 500ms)
        assert elapsed_time < 500, f"Feature generation took {elapsed_time:.2f}ms (> 500ms)"
        assert len(result) == 43800


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def plugin(self) -> TemporalFeaturesPlugin:
        """Create a plugin instance for testing."""
        return TemporalFeaturesPlugin()

    def test_empty_dataframe(self, plugin: TemporalFeaturesPlugin):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(
            {"load": []},
            index=pd.DatetimeIndex([]),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=False,
            features=["hour"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert len(result) == 0
        assert "hour" in result.columns

    def test_single_row_dataframe(self, plugin: TemporalFeaturesPlugin):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.date_range("2024-01-15 10:00:00", periods=1, freq="h"),
        )

        config = TemporalFeaturesConfig().model_dump()
        result = plugin.generate_features(df, config)

        assert len(result) == 1
        assert result["hour"].values[0] == 10

    def test_preserves_original_columns(self, plugin: TemporalFeaturesPlugin):
        """Test that original columns are preserved."""
        df = pd.DataFrame(
            {
                "load": [100.0, 200.0],
                "temperature": [25.0, 26.0],
                "humidity": [0.7, 0.8],
            },
            index=pd.date_range("2024-01-15", periods=2, freq="h"),
        )

        config = TemporalFeaturesConfig().model_dump()
        result = plugin.generate_features(df, config)

        assert "load" in result.columns
        assert "temperature" in result.columns
        assert "humidity" in result.columns
        np.testing.assert_array_equal(result["load"].values, [100.0, 200.0])

    def test_preserves_index(self, plugin: TemporalFeaturesPlugin):
        """Test that DataFrame index is preserved."""
        original_index = pd.date_range("2024-01-15 10:00:00", periods=3, freq="h")
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=original_index,
        )

        config = TemporalFeaturesConfig().model_dump()
        result = plugin.generate_features(df, config)

        pd.testing.assert_index_equal(result.index, original_index)

    def test_does_not_modify_input(self, plugin: TemporalFeaturesPlugin):
        """Test that input DataFrame is not modified."""
        df = pd.DataFrame(
            {"load": [100.0, 200.0]},
            index=pd.date_range("2024-01-15", periods=2, freq="h"),
        )
        original_columns = list(df.columns)

        config = TemporalFeaturesConfig().model_dump()
        _result = plugin.generate_features(df, config)

        assert list(df.columns) == original_columns

    def test_leap_year_handling(self, plugin: TemporalFeaturesPlugin):
        """Test handling of leap year dates."""
        # Feb 29, 2024 (leap year)
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-02-29 12:00:00"]),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=True,
            features=["day_of_month", "month"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["day_of_month"].values[0] == 29
        assert result["month"].values[0] == 2
        assert result["season"].values[0] == 1  # February = Summer

    def test_year_boundary(self, plugin: TemporalFeaturesPlugin):
        """Test handling of year boundary (Dec 31 -> Jan 1)."""
        df = pd.DataFrame(
            {"load": [100.0, 200.0]},
            index=pd.DatetimeIndex([
                "2023-12-31 23:00:00",
                "2024-01-01 00:00:00",
            ]),
        )

        config = TemporalFeaturesConfig(
            include_cyclical=False,
            include_season=True,
            features=["year", "month", "day_of_month", "hour"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        # Dec 31, 2023
        assert result["year"].values[0] == 2023
        assert result["month"].values[0] == 12
        assert result["day_of_month"].values[0] == 31
        assert result["hour"].values[0] == 23
        assert result["season"].values[0] == 1  # December = Summer

        # Jan 1, 2024
        assert result["year"].values[1] == 2024
        assert result["month"].values[1] == 1
        assert result["day_of_month"].values[1] == 1
        assert result["hour"].values[1] == 0
        assert result["season"].values[1] == 1  # January = Summer
