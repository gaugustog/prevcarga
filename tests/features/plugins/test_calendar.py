"""Tests for CalendarFeaturesPlugin.

This module contains comprehensive tests for the CalendarFeaturesPlugin,
including Brazilian holiday detection, bridge days, pre/post holiday indicators,
days until next holiday, custom holidays, and performance tests.
"""

import time
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.plugins.calendar import (
    BrazilianHolidays,
    CalendarFeaturesConfig,
    CalendarFeaturesPlugin,
)


class TestBrazilianHolidays:
    """Tests for BrazilianHolidays class."""

    @pytest.fixture
    def holidays(self) -> BrazilianHolidays:
        """Create a BrazilianHolidays instance for testing."""
        return BrazilianHolidays()

    def test_fixed_holidays_count(self, holidays: BrazilianHolidays):
        """Test that all 9 fixed holidays are defined."""
        assert len(BrazilianHolidays.FIXED_HOLIDAYS) == 9

    def test_new_year(self, holidays: BrazilianHolidays):
        """Test New Year holiday (Jan 1)."""
        name = holidays.get_holiday_name(date(2024, 1, 1))
        assert name == "Ano Novo"

    def test_tiradentes(self, holidays: BrazilianHolidays):
        """Test Tiradentes holiday (Apr 21)."""
        name = holidays.get_holiday_name(date(2024, 4, 21))
        assert name == "Tiradentes"

    def test_labor_day(self, holidays: BrazilianHolidays):
        """Test Labor Day (May 1)."""
        name = holidays.get_holiday_name(date(2024, 5, 1))
        assert name == "Dia do Trabalho"

    def test_independence_day(self, holidays: BrazilianHolidays):
        """Test Independence Day (Sep 7)."""
        name = holidays.get_holiday_name(date(2024, 9, 7))
        assert name == "Independencia"

    def test_aparecida(self, holidays: BrazilianHolidays):
        """Test Nossa Senhora Aparecida (Oct 12)."""
        name = holidays.get_holiday_name(date(2024, 10, 12))
        assert name == "Nossa Senhora Aparecida"

    def test_all_souls_day(self, holidays: BrazilianHolidays):
        """Test All Souls Day (Nov 2)."""
        name = holidays.get_holiday_name(date(2024, 11, 2))
        assert name == "Finados"

    def test_republic_day(self, holidays: BrazilianHolidays):
        """Test Republic Day (Nov 15)."""
        name = holidays.get_holiday_name(date(2024, 11, 15))
        assert name == "Proclamacao da Republica"

    def test_black_consciousness_day(self, holidays: BrazilianHolidays):
        """Test Black Consciousness Day (Nov 20)."""
        name = holidays.get_holiday_name(date(2024, 11, 20))
        assert name == "Consciencia Negra"

    def test_christmas(self, holidays: BrazilianHolidays):
        """Test Christmas (Dec 25)."""
        name = holidays.get_holiday_name(date(2024, 12, 25))
        assert name == "Natal"


class TestMovableHolidays:
    """Tests for movable holidays based on Easter."""

    @pytest.fixture
    def holidays(self) -> BrazilianHolidays:
        """Create a BrazilianHolidays instance for testing."""
        return BrazilianHolidays()

    def test_carnival_2024(self, holidays: BrazilianHolidays):
        """Test Carnival 2024 (Feb 13, 47 days before Easter Mar 31)."""
        # Easter 2024 is March 31
        # Carnival = Easter - 47 days = February 13
        name = holidays.get_holiday_name(date(2024, 2, 13))
        assert name == "Carnaval"

    def test_good_friday_2024(self, holidays: BrazilianHolidays):
        """Test Good Friday 2024 (Mar 29, 2 days before Easter)."""
        # Easter 2024 is March 31
        # Good Friday = Easter - 2 days = March 29
        name = holidays.get_holiday_name(date(2024, 3, 29))
        assert name == "Sexta-feira Santa"

    def test_corpus_christi_2024(self, holidays: BrazilianHolidays):
        """Test Corpus Christi 2024 (May 30, 60 days after Easter)."""
        # Easter 2024 is March 31
        # Corpus Christi = Easter + 60 days = May 30
        name = holidays.get_holiday_name(date(2024, 5, 30))
        assert name == "Corpus Christi"

    def test_carnival_2023(self, holidays: BrazilianHolidays):
        """Test Carnival 2023 (Feb 21, 47 days before Easter Apr 9)."""
        # Easter 2023 is April 9
        # Carnival = Easter - 47 days = February 21
        name = holidays.get_holiday_name(date(2023, 2, 21))
        assert name == "Carnaval"

    def test_good_friday_2023(self, holidays: BrazilianHolidays):
        """Test Good Friday 2023 (Apr 7, 2 days before Easter)."""
        # Easter 2023 is April 9
        # Good Friday = Easter - 2 days = April 7
        name = holidays.get_holiday_name(date(2023, 4, 7))
        assert name == "Sexta-feira Santa"

    def test_corpus_christi_2023(self, holidays: BrazilianHolidays):
        """Test Corpus Christi 2023 (Jun 8, 60 days after Easter)."""
        # Easter 2023 is April 9
        # Corpus Christi = Easter + 60 days = June 8
        name = holidays.get_holiday_name(date(2023, 6, 8))
        assert name == "Corpus Christi"

    def test_carnival_2025(self, holidays: BrazilianHolidays):
        """Test Carnival 2025."""
        # Easter 2025 is April 20
        # Carnival = Easter - 47 days = March 4
        name = holidays.get_holiday_name(date(2025, 3, 4))
        assert name == "Carnaval"


class TestHolidayMethods:
    """Tests for BrazilianHolidays utility methods."""

    @pytest.fixture
    def holidays(self) -> BrazilianHolidays:
        """Create a BrazilianHolidays instance for testing."""
        return BrazilianHolidays()

    def test_get_holidays_returns_dict(self, holidays: BrazilianHolidays):
        """Test that get_holidays returns a dictionary."""
        result = holidays.get_holidays(2024)
        assert isinstance(result, dict)

    def test_get_holidays_count(self, holidays: BrazilianHolidays):
        """Test that get_holidays returns 12 holidays (9 fixed + 3 movable)."""
        result = holidays.get_holidays(2024)
        assert len(result) == 12

    def test_is_holiday_true(self, holidays: BrazilianHolidays):
        """Test is_holiday returns True for a holiday."""
        assert holidays.is_holiday(date(2024, 12, 25)) is True

    def test_is_holiday_false(self, holidays: BrazilianHolidays):
        """Test is_holiday returns False for a non-holiday."""
        assert holidays.is_holiday(date(2024, 3, 15)) is False

    def test_get_holiday_name_none(self, holidays: BrazilianHolidays):
        """Test get_holiday_name returns None for non-holiday."""
        assert holidays.get_holiday_name(date(2024, 3, 15)) is None

    def test_get_next_holiday(self, holidays: BrazilianHolidays):
        """Test get_next_holiday returns correct next holiday."""
        # Jan 2 should return Carnival as next holiday
        result = holidays.get_next_holiday(date(2024, 1, 2))
        assert result is not None
        holiday_date, holiday_name = result
        assert holiday_date == date(2024, 2, 13)  # Carnival 2024
        assert holiday_name == "Carnaval"

    def test_get_next_holiday_on_holiday(self, holidays: BrazilianHolidays):
        """Test get_next_holiday returns same day when called on a holiday."""
        result = holidays.get_next_holiday(date(2024, 1, 1))
        assert result is not None
        holiday_date, holiday_name = result
        assert holiday_date == date(2024, 1, 1)
        assert holiday_name == "Ano Novo"

    def test_get_next_holiday_year_boundary(self, holidays: BrazilianHolidays):
        """Test get_next_holiday handles year boundary correctly."""
        # Dec 26 should return New Year as next holiday
        result = holidays.get_next_holiday(date(2024, 12, 26))
        assert result is not None
        holiday_date, holiday_name = result
        assert holiday_date == date(2025, 1, 1)
        assert holiday_name == "Ano Novo"

    def test_holiday_cache(self, holidays: BrazilianHolidays):
        """Test that holidays are cached."""
        # First call
        result1 = holidays.get_holidays(2024)
        # Second call should return cached result
        result2 = holidays.get_holidays(2024)

        assert result1 is result2  # Same object (cached)

    def test_clear_cache(self, holidays: BrazilianHolidays):
        """Test cache clearing."""
        # Populate cache
        holidays.get_holidays(2024)
        assert 2024 in holidays._cache

        # Clear cache
        holidays.clear_cache()
        assert 2024 not in holidays._cache


class TestCalendarFeaturesConfig:
    """Tests for CalendarFeaturesConfig Pydantic model."""

    def test_default_config(self):
        """Test that default configuration has expected values."""
        config = CalendarFeaturesConfig()

        assert config.include_bridge_days is True
        assert config.include_pre_post_indicators is True
        assert config.include_days_until is True
        assert config.custom_holidays == []
        assert config.timezone == "America/Sao_Paulo"

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
            custom_holidays=["2024-06-10"],
            timezone="UTC",
        )

        assert config.include_bridge_days is False
        assert config.include_pre_post_indicators is False
        assert config.include_days_until is False
        assert config.custom_holidays == ["2024-06-10"]
        assert config.timezone == "UTC"

    def test_valid_timezone(self):
        """Test that valid timezones are accepted."""
        valid_timezones = [
            "America/Sao_Paulo",
            "UTC",
            "America/New_York",
        ]

        for tz in valid_timezones:
            config = CalendarFeaturesConfig(timezone=tz)
            assert config.timezone == tz

    def test_invalid_timezone_raises_error(self):
        """Test that invalid timezone raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CalendarFeaturesConfig(timezone="Invalid/Timezone")

        assert "Invalid timezone" in str(exc_info.value)

    def test_valid_custom_holidays(self):
        """Test that valid custom holiday format is accepted."""
        config = CalendarFeaturesConfig(
            custom_holidays=["2024-01-02", "2024-06-10", "2024-12-31"]
        )
        assert len(config.custom_holidays) == 3

    def test_invalid_custom_holiday_format(self):
        """Test that invalid custom holiday format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CalendarFeaturesConfig(custom_holidays=["01-02-2024"])

        assert "Invalid date format" in str(exc_info.value)

    def test_config_model_dump(self):
        """Test that config can be serialized to dict."""
        config = CalendarFeaturesConfig()
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert "include_bridge_days" in config_dict
        assert "include_pre_post_indicators" in config_dict
        assert "include_days_until" in config_dict
        assert "custom_holidays" in config_dict
        assert "timezone" in config_dict

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValueError):
            CalendarFeaturesConfig(unknown_field="value")  # type: ignore[call-arg]


class TestCalendarFeaturesPlugin:
    """Tests for CalendarFeaturesPlugin base functionality."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame with DatetimeIndex."""
        return pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

    @pytest.fixture
    def default_config(self) -> dict[str, Any]:
        """Return default configuration as dict."""
        return CalendarFeaturesConfig().model_dump()

    def test_plugin_name(self, plugin: CalendarFeaturesPlugin):
        """Test that plugin name is correct."""
        assert plugin.name == "calendar_features"

    def test_plugin_version(self, plugin: CalendarFeaturesPlugin):
        """Test that plugin version is correct."""
        assert plugin.version == "1.0.0"

    def test_get_metadata(self, plugin: CalendarFeaturesPlugin):
        """Test that get_metadata returns correct values."""
        metadata = plugin.get_metadata()

        assert metadata["name"] == "calendar_features"
        assert metadata["version"] == "1.0.0"
        assert metadata["class"] == "CalendarFeaturesPlugin"

    def test_repr(self, plugin: CalendarFeaturesPlugin):
        """Test plugin string representation."""
        repr_str = repr(plugin)

        assert "CalendarFeaturesPlugin" in repr_str
        assert "calendar_features" in repr_str
        assert "1.0.0" in repr_str


class TestIsHolidayFeature:
    """Tests for is_holiday feature generation."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_is_holiday_new_year(self, plugin: CalendarFeaturesPlugin):
        """Test is_holiday detects New Year."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-01-01"]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "is_holiday" in result.columns
        assert result["is_holiday"].values[0] == 1

    def test_is_holiday_non_holiday(self, plugin: CalendarFeaturesPlugin):
        """Test is_holiday returns 0 for non-holiday."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_holiday"].values[0] == 0

    def test_is_holiday_movable(self, plugin: CalendarFeaturesPlugin):
        """Test is_holiday detects movable holidays."""
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.DatetimeIndex([
                "2024-02-13",  # Carnival
                "2024-03-29",  # Good Friday
                "2024-05-30",  # Corpus Christi
            ]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        np.testing.assert_array_equal(result["is_holiday"].values, [1, 1, 1])

    def test_is_holiday_dtype(self, plugin: CalendarFeaturesPlugin):
        """Test that is_holiday feature is int32 dtype."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-01-01"]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_holiday"].dtype == np.int32


class TestBridgeDayFeature:
    """Tests for is_bridge_day feature generation."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_monday_before_tuesday_holiday(self, plugin: CalendarFeaturesPlugin):
        """Test Monday is a bridge day when Tuesday is a holiday."""
        # Tiradentes (Apr 21, 2026) falls on Tuesday
        # So Monday Apr 20 should be a bridge day
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2026-04-20"]),  # Monday
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=True,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_bridge_day"].values[0] == 1

    def test_friday_after_thursday_holiday(self, plugin: CalendarFeaturesPlugin):
        """Test Friday is a bridge day when Thursday is a holiday."""
        # Christmas 2025 (Dec 25) falls on Thursday
        # So Friday Dec 26 should be a bridge day
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2025-12-26"]),  # Friday
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=True,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_bridge_day"].values[0] == 1

    def test_non_bridge_monday(self, plugin: CalendarFeaturesPlugin):
        """Test Monday is not a bridge day when Tuesday is not a holiday."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-01-15"]),  # Monday, Jan 16 is not a holiday
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=True,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_bridge_day"].values[0] == 0

    def test_bridge_day_disabled(self, plugin: CalendarFeaturesPlugin):
        """Test that bridge day feature is not generated when disabled."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-01-01"]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "is_bridge_day" not in result.columns


class TestPrePostHolidayFeatures:
    """Tests for is_pre_holiday and is_post_holiday features."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_pre_holiday_day_before(self, plugin: CalendarFeaturesPlugin):
        """Test pre-holiday for day immediately before holiday."""
        # Dec 24 is pre-holiday (before Christmas Dec 25)
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-12-24"]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=True,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_pre_holiday"].values[0] == 1

    def test_post_holiday_day_after(self, plugin: CalendarFeaturesPlugin):
        """Test post-holiday for day immediately after holiday."""
        # Jan 2 is post-holiday (after New Year Jan 1)
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=True,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_post_holiday"].values[0] == 1

    def test_friday_before_monday_holiday(self, plugin: CalendarFeaturesPlugin):
        """Test Friday is pre-holiday when Monday is a holiday."""
        # Labor Day 2028 (May 1) falls on Monday
        # Friday Apr 28 should be pre-holiday
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2028-04-28"]),  # Friday
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=True,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_pre_holiday"].values[0] == 1

    def test_monday_after_friday_holiday(self, plugin: CalendarFeaturesPlugin):
        """Test Monday is post-holiday when Friday is a holiday."""
        # Tiradentes 2025 (Apr 21) falls on Monday - need a Friday holiday
        # Good Friday 2024 (Mar 29) falls on Friday
        # Monday Apr 1 should be post-holiday
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-04-01"]),  # Monday after Good Friday
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=True,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_post_holiday"].values[0] == 1

    def test_pre_post_disabled(self, plugin: CalendarFeaturesPlugin):
        """Test that pre/post holiday features are not generated when disabled."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-01-01"]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "is_pre_holiday" not in result.columns
        assert "is_post_holiday" not in result.columns


class TestDaysUntilHoliday:
    """Tests for days_until_next_holiday feature."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_days_until_on_holiday(self, plugin: CalendarFeaturesPlugin):
        """Test days_until returns 0 on a holiday."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-01-01"]),  # New Year
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=True,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["days_until_next_holiday"].values[0] == 0

    def test_days_until_before_holiday(self, plugin: CalendarFeaturesPlugin):
        """Test days_until returns correct count before holiday."""
        # Jan 2, 2024: next holiday is Carnival (Feb 13)
        # Days: Feb 13 - Jan 2 = 42 days
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=True,
        ).model_dump()

        result = plugin.generate_features(df, config)

        # Calculate expected days
        expected_days = (date(2024, 2, 13) - date(2024, 1, 2)).days
        assert result["days_until_next_holiday"].values[0] == expected_days

    def test_days_until_day_before(self, plugin: CalendarFeaturesPlugin):
        """Test days_until returns 1 for day before holiday."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-12-24"]),  # Day before Christmas
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=True,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["days_until_next_holiday"].values[0] == 1

    def test_days_until_year_boundary(self, plugin: CalendarFeaturesPlugin):
        """Test days_until handles year boundary correctly."""
        # Dec 26, 2024: next holiday is New Year 2025 (6 days)
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-12-26"]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=True,
        ).model_dump()

        result = plugin.generate_features(df, config)

        expected_days = (date(2025, 1, 1) - date(2024, 12, 26)).days
        assert result["days_until_next_holiday"].values[0] == expected_days

    def test_days_until_disabled(self, plugin: CalendarFeaturesPlugin):
        """Test that days_until feature is not generated when disabled."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-01-01"]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "days_until_next_holiday" not in result.columns


class TestCustomHolidays:
    """Tests for custom holiday support."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_custom_holiday_detected(self, plugin: CalendarFeaturesPlugin):
        """Test that custom holidays are detected."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-06-10"]),  # Custom holiday
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
            custom_holidays=["2024-06-10"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_holiday"].values[0] == 1

    def test_custom_holiday_bridge_day(self, plugin: CalendarFeaturesPlugin):
        """Test bridge day detection with custom holidays."""
        # Add Tuesday Jun 11 as custom holiday, so Monday Jun 10 is bridge day
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-06-10"]),  # Monday
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=True,
            include_pre_post_indicators=False,
            include_days_until=False,
            custom_holidays=["2024-06-11"],  # Tuesday
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_bridge_day"].values[0] == 1

    def test_custom_holiday_pre_holiday(self, plugin: CalendarFeaturesPlugin):
        """Test pre-holiday detection with custom holidays."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-06-09"]),  # Day before custom holiday
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=True,
            include_days_until=False,
            custom_holidays=["2024-06-10"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["is_pre_holiday"].values[0] == 1

    def test_custom_holiday_days_until(self, plugin: CalendarFeaturesPlugin):
        """Test days_until with custom holidays."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-06-05"]),  # 5 days before custom holiday
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=True,
            custom_holidays=["2024-06-10"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert result["days_until_next_holiday"].values[0] == 5

    def test_multiple_custom_holidays(self, plugin: CalendarFeaturesPlugin):
        """Test with multiple custom holidays."""
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.DatetimeIndex([
                "2024-06-10",
                "2024-07-15",
                "2024-08-20",
            ]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
            custom_holidays=["2024-06-10", "2024-07-15", "2024-08-20"],
        ).model_dump()

        result = plugin.generate_features(df, config)

        np.testing.assert_array_equal(result["is_holiday"].values, [1, 1, 1])


class TestFeatureNames:
    """Tests for get_feature_names method."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_feature_names_all_enabled(self, plugin: CalendarFeaturesPlugin):
        """Test feature names with all features enabled."""
        config = CalendarFeaturesConfig().model_dump()
        names = plugin.get_feature_names(config)

        assert "is_holiday" in names
        assert "is_bridge_day" in names
        assert "is_pre_holiday" in names
        assert "is_post_holiday" in names
        assert "days_until_next_holiday" in names
        assert len(names) == 5

    def test_feature_names_minimal(self, plugin: CalendarFeaturesPlugin):
        """Test feature names with minimal features."""
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()
        names = plugin.get_feature_names(config)

        assert names == ["is_holiday"]

    def test_feature_names_match_generated(self, plugin: CalendarFeaturesPlugin):
        """Test that feature names match actually generated columns."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.date_range("2024-01-15", periods=1, freq="D"),
        )

        config = CalendarFeaturesConfig().model_dump()
        names = plugin.get_feature_names(config)
        result = plugin.generate_features(df, config)

        for name in names:
            assert name in result.columns, f"Feature {name} not in generated columns"


class TestTimezoneHandling:
    """Tests for timezone handling."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_naive_datetime_localization(self, plugin: CalendarFeaturesPlugin):
        """Test that naive datetimes are localized to configured timezone."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.date_range("2024-01-01 10:00:00", periods=1, freq="h"),
        )

        assert df.index.tz is None  # Confirm naive

        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
            timezone="America/Sao_Paulo",
        ).model_dump()

        result = plugin.generate_features(df, config)

        # Should have generated is_holiday feature for Jan 1 (New Year)
        assert result["is_holiday"].values[0] == 1

    def test_timestamp_column_support(self, plugin: CalendarFeaturesPlugin):
        """Test that timestamp column is used when present."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "load": [100.0, 200.0, 300.0],
        })

        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        assert "is_holiday" in result.columns
        # Jan 1 is holiday, Jan 2 and 3 are not
        np.testing.assert_array_equal(result["is_holiday"].values, [1, 0, 0])

    def test_no_datetime_raises_error(self, plugin: CalendarFeaturesPlugin):
        """Test that error is raised when no datetime source exists."""
        df = pd.DataFrame({
            "load": [100.0, 200.0, 300.0],
            "value": [1, 2, 3],
        })

        config = CalendarFeaturesConfig().model_dump()

        with pytest.raises(ValueError) as exc_info:
            plugin.generate_features(df, config)

        assert "DatetimeIndex" in str(exc_info.value)


class TestConfigValidation:
    """Tests for configuration validation."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_validate_config_accepts_valid(self, plugin: CalendarFeaturesPlugin):
        """Test that validate_config accepts valid configuration."""
        config = CalendarFeaturesConfig().model_dump()
        result = plugin.validate_config(config)
        assert result is True

    def test_validate_config_rejects_none(self, plugin: CalendarFeaturesPlugin):
        """Test that validate_config rejects None."""
        with pytest.raises(ValueError):
            plugin.validate_config(None)  # type: ignore[arg-type]

    def test_validate_config_rejects_invalid_timezone(
        self, plugin: CalendarFeaturesPlugin
    ):
        """Test that validate_config rejects invalid timezone."""
        config = {"timezone": "Invalid/Timezone"}

        with pytest.raises(ValueError):
            plugin.validate_config(config)


class TestPerformance:
    """Performance tests for calendar features plugin."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_performance_one_year_daily(self, plugin: CalendarFeaturesPlugin):
        """Test that 1 year of daily data processes in < 50ms."""
        # Create 1 year of daily data (365 rows)
        df = pd.DataFrame(
            {"load": np.random.randn(365)},
            index=pd.date_range("2024-01-01", periods=365, freq="D"),
        )

        config = CalendarFeaturesConfig().model_dump()

        # Time the feature generation
        start_time = time.perf_counter()
        _result = plugin.generate_features(df, config)
        elapsed_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        assert elapsed_time < 50, f"Feature generation took {elapsed_time:.2f}ms (> 50ms)"

    def test_performance_one_year_hourly(self, plugin: CalendarFeaturesPlugin):
        """Test performance with 1 year of hourly data."""
        # Create 1 year of hourly data (8760 rows)
        df = pd.DataFrame(
            {"load": np.random.randn(8760)},
            index=pd.date_range("2024-01-01", periods=8760, freq="h"),
        )

        config = CalendarFeaturesConfig().model_dump()

        # Time the feature generation
        start_time = time.perf_counter()
        result = plugin.generate_features(df, config)
        elapsed_time = (time.perf_counter() - start_time) * 1000

        # Should still be reasonably fast (< 500ms)
        assert elapsed_time < 500, f"Feature generation took {elapsed_time:.2f}ms (> 500ms)"
        assert len(result) == 8760


class TestMultipleYears:
    """Tests for handling multiple years of data."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_holidays_across_years(self, plugin: CalendarFeaturesPlugin):
        """Test holiday detection across multiple years."""
        # Test New Year across 3 years
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.DatetimeIndex([
                "2022-01-01",
                "2023-01-01",
                "2024-01-01",
            ]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        np.testing.assert_array_equal(result["is_holiday"].values, [1, 1, 1])

    def test_movable_holidays_different_years(self, plugin: CalendarFeaturesPlugin):
        """Test movable holidays calculated correctly for different years."""
        # Carnival dates: 2022=Mar1, 2023=Feb21, 2024=Feb13
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=pd.DatetimeIndex([
                "2022-03-01",  # Carnival 2022
                "2023-02-21",  # Carnival 2023
                "2024-02-13",  # Carnival 2024
            ]),
        )
        config = CalendarFeaturesConfig(
            include_bridge_days=False,
            include_pre_post_indicators=False,
            include_days_until=False,
        ).model_dump()

        result = plugin.generate_features(df, config)

        np.testing.assert_array_equal(result["is_holiday"].values, [1, 1, 1])


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def plugin(self) -> CalendarFeaturesPlugin:
        """Create a plugin instance for testing."""
        return CalendarFeaturesPlugin()

    def test_empty_dataframe(self, plugin: CalendarFeaturesPlugin):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(
            {"load": []},
            index=pd.DatetimeIndex([]),
        )

        config = CalendarFeaturesConfig().model_dump()

        result = plugin.generate_features(df, config)

        assert len(result) == 0
        assert "is_holiday" in result.columns

    def test_single_row_dataframe(self, plugin: CalendarFeaturesPlugin):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="D"),
        )

        config = CalendarFeaturesConfig().model_dump()
        result = plugin.generate_features(df, config)

        assert len(result) == 1
        assert result["is_holiday"].values[0] == 1

    def test_preserves_original_columns(self, plugin: CalendarFeaturesPlugin):
        """Test that original columns are preserved."""
        df = pd.DataFrame(
            {
                "load": [100.0, 200.0],
                "temperature": [25.0, 26.0],
                "humidity": [0.7, 0.8],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )

        config = CalendarFeaturesConfig().model_dump()
        result = plugin.generate_features(df, config)

        assert "load" in result.columns
        assert "temperature" in result.columns
        assert "humidity" in result.columns
        np.testing.assert_array_equal(result["load"].values, [100.0, 200.0])

    def test_preserves_index(self, plugin: CalendarFeaturesPlugin):
        """Test that DataFrame index is preserved."""
        original_index = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {"load": [100.0, 200.0, 300.0]},
            index=original_index,
        )

        config = CalendarFeaturesConfig().model_dump()
        result = plugin.generate_features(df, config)

        pd.testing.assert_index_equal(result.index, original_index)

    def test_does_not_modify_input(self, plugin: CalendarFeaturesPlugin):
        """Test that input DataFrame is not modified."""
        df = pd.DataFrame(
            {"load": [100.0, 200.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )
        original_columns = list(df.columns)

        config = CalendarFeaturesConfig().model_dump()
        _result = plugin.generate_features(df, config)

        assert list(df.columns) == original_columns

    def test_leap_year_handling(self, plugin: CalendarFeaturesPlugin):
        """Test handling of leap year dates."""
        # Feb 29, 2024 (leap year)
        df = pd.DataFrame(
            {"load": [100.0]},
            index=pd.DatetimeIndex(["2024-02-29 12:00:00"]),
        )

        config = CalendarFeaturesConfig().model_dump()
        result = plugin.generate_features(df, config)

        # Feb 29 is not a holiday
        assert result["is_holiday"].values[0] == 0

    def test_year_boundary(self, plugin: CalendarFeaturesPlugin):
        """Test handling of year boundary (Dec 31 -> Jan 1)."""
        df = pd.DataFrame(
            {"load": [100.0, 200.0]},
            index=pd.DatetimeIndex([
                "2023-12-31 23:00:00",
                "2024-01-01 00:00:00",
            ]),
        )

        config = CalendarFeaturesConfig().model_dump()
        result = plugin.generate_features(df, config)

        # Dec 31 is not a holiday, Jan 1 is (New Year)
        assert result["is_holiday"].values[0] == 0
        assert result["is_holiday"].values[1] == 1

        # Dec 31 is pre-holiday
        assert result["is_pre_holiday"].values[0] == 1

        # Jan 1 is a holiday so not post-holiday for itself
        # But Jan 2 would be post-holiday


class TestHolidayIntegrity:
    """Tests to verify holiday dates are calculated correctly for multiple years."""

    @pytest.fixture
    def holidays(self) -> BrazilianHolidays:
        """Create a BrazilianHolidays instance for testing."""
        return BrazilianHolidays()

    def test_total_holidays_per_year(self, holidays: BrazilianHolidays):
        """Test that each year has exactly 12 holidays."""
        for year in range(2020, 2030):
            holiday_dict = holidays.get_holidays(year)
            assert len(holiday_dict) == 12, f"Year {year} has {len(holiday_dict)} holidays"

    def test_fixed_holidays_constant(self, holidays: BrazilianHolidays):
        """Test that fixed holidays are on the same date each year."""
        fixed_dates = [
            (1, 1),   # New Year
            (4, 21),  # Tiradentes
            (5, 1),   # Labor Day
            (9, 7),   # Independence
            (10, 12), # Aparecida
            (11, 2),  # All Souls
            (11, 15), # Republic
            (11, 20), # Black Consciousness
            (12, 25), # Christmas
        ]

        for year in range(2020, 2030):
            holiday_dict = holidays.get_holidays(year)
            for month, day in fixed_dates:
                d = date(year, month, day)
                assert d in holiday_dict, f"{d} not in holidays for {year}"
