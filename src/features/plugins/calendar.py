"""Calendar and holiday features plugin for Brazilian load forecasting.

This module provides the CalendarFeaturesPlugin for generating holiday and calendar
features specific to Brazil, including fixed holidays, movable holidays (based on
Easter), and derived features like bridge days and pre/post holiday indicators.

Example:
    ```python
    from src.features.plugins.calendar import (
        CalendarFeaturesPlugin,
        CalendarFeaturesConfig,
        BrazilianHolidays,
    )
    import pandas as pd

    # Create plugin and config
    plugin = CalendarFeaturesPlugin()
    config = CalendarFeaturesConfig(
        include_bridge_days=True,
        include_pre_post_indicators=True,
        include_days_until=True,
    )

    # Generate features
    df = pd.DataFrame(
        {"load": [100, 200, 300]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    result = plugin.generate_features(df, config.model_dump())

    # Check holidays directly
    holidays = BrazilianHolidays()
    print(holidays.get_holidays(2024))
    print(holidays.get_holiday_name(date(2024, 12, 25)))  # "Natal"
    ```
"""

from datetime import date, timedelta
from typing import Any, ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from dateutil.easter import easter
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Day of week constants (Monday=0, Sunday=6)
MONDAY = 0
THURSDAY = 3
FRIDAY = 4
SATURDAY = 5
SUNDAY = 6

# Days offset from Easter for movable holidays
CARNIVAL_OFFSET = -47  # 47 days before Easter
GOOD_FRIDAY_OFFSET = -2  # 2 days before Easter
CORPUS_CHRISTI_OFFSET = 60  # 60 days after Easter


class BrazilianHolidays:
    """Manager for Brazilian national holidays.

    This class handles both fixed and movable Brazilian national holidays.
    Holiday data is cached by year for performance.

    Fixed holidays:
        - New Year (Jan 1): Ano Novo
        - Tiradentes (Apr 21): Tiradentes
        - Labor Day (May 1): Dia do Trabalho
        - Independence Day (Sep 7): Independencia
        - Nossa Senhora Aparecida (Oct 12): Nossa Senhora Aparecida
        - All Souls Day (Nov 2): Finados
        - Republic Day (Nov 15): Proclamacao da Republica
        - Black Consciousness Day (Nov 20): Consciencia Negra
        - Christmas (Dec 25): Natal

    Movable holidays (based on Easter):
        - Carnival (47 days before Easter): Carnaval
        - Good Friday (2 days before Easter): Sexta-feira Santa
        - Corpus Christi (60 days after Easter): Corpus Christi

    Example:
        ```python
        holidays = BrazilianHolidays()

        # Get all holidays for 2024
        for d, name in holidays.get_holidays(2024).items():
            print(f"{d}: {name}")

        # Get holiday name for a specific date
        name = holidays.get_holiday_name(date(2024, 12, 25))
        print(name)  # "Natal"

        # Check if a date is a holiday
        is_holiday = holidays.is_holiday(date(2024, 1, 1))
        print(is_holiday)  # True
        ```
    """

    # Fixed holidays: (month, day) -> name
    FIXED_HOLIDAYS: ClassVar[dict[tuple[int, int], str]] = {
        (1, 1): "Ano Novo",
        (4, 21): "Tiradentes",
        (5, 1): "Dia do Trabalho",
        (9, 7): "Independencia",
        (10, 12): "Nossa Senhora Aparecida",
        (11, 2): "Finados",
        (11, 15): "Proclamacao da Republica",
        (11, 20): "Consciencia Negra",
        (12, 25): "Natal",
    }

    def __init__(self) -> None:
        """Initialize the BrazilianHolidays manager with empty cache."""
        self._cache: dict[int, dict[date, str]] = {}

    def get_holidays(self, year: int) -> dict[date, str]:
        """Get all Brazilian national holidays for a given year.

        This method returns both fixed and movable holidays. Results are
        cached for performance.

        Args:
            year: The year to get holidays for.

        Returns:
            Dictionary mapping holiday dates to holiday names.

        Example:
            ```python
            holidays = BrazilianHolidays()
            holiday_dict = holidays.get_holidays(2024)
            for d, name in sorted(holiday_dict.items()):
                print(f"{d}: {name}")
            ```
        """
        if year in self._cache:
            return self._cache[year]

        holidays: dict[date, str] = {}

        # Add fixed holidays
        for (month, day), name in self.FIXED_HOLIDAYS.items():
            holidays[date(year, month, day)] = name

        # Calculate Easter and add movable holidays
        easter_date = easter(year)

        # Carnival (47 days before Easter)
        carnival_date = easter_date + timedelta(days=CARNIVAL_OFFSET)
        holidays[carnival_date] = "Carnaval"

        # Good Friday (2 days before Easter)
        good_friday_date = easter_date + timedelta(days=GOOD_FRIDAY_OFFSET)
        holidays[good_friday_date] = "Sexta-feira Santa"

        # Corpus Christi (60 days after Easter)
        corpus_christi_date = easter_date + timedelta(days=CORPUS_CHRISTI_OFFSET)
        holidays[corpus_christi_date] = "Corpus Christi"

        # Cache the result
        self._cache[year] = holidays
        logger.debug("Cached %d holidays for year %d", len(holidays), year)

        return holidays

    def get_holiday_name(self, d: date) -> str | None:
        """Get the holiday name for a specific date.

        Args:
            d: The date to check.

        Returns:
            The holiday name if the date is a holiday, None otherwise.

        Example:
            ```python
            holidays = BrazilianHolidays()
            name = holidays.get_holiday_name(date(2024, 12, 25))
            print(name)  # "Natal"
            ```
        """
        holidays = self.get_holidays(d.year)
        return holidays.get(d)

    def is_holiday(self, d: date) -> bool:
        """Check if a date is a Brazilian national holiday.

        Args:
            d: The date to check.

        Returns:
            True if the date is a holiday, False otherwise.
        """
        return self.get_holiday_name(d) is not None

    def get_next_holiday(self, d: date) -> tuple[date, str] | None:
        """Get the next holiday on or after the given date.

        Args:
            d: The starting date.

        Returns:
            Tuple of (holiday_date, holiday_name), or None if no holiday found
            within the same or next year.
        """
        # Check current year
        holidays = self.get_holidays(d.year)
        for holiday_date in sorted(holidays.keys()):
            if holiday_date >= d:
                return (holiday_date, holidays[holiday_date])

        # Check next year if no holiday found in current year
        next_year_holidays = self.get_holidays(d.year + 1)
        for holiday_date in sorted(next_year_holidays.keys()):
            return (holiday_date, next_year_holidays[holiday_date])

        return None

    def clear_cache(self) -> None:
        """Clear the holiday cache."""
        self._cache.clear()
        logger.debug("Holiday cache cleared")


class CalendarFeaturesConfig(BaseModel):
    """Configuration model for CalendarFeaturesPlugin.

    This model defines configuration options for calendar and holiday feature
    generation, including bridge days, pre/post holiday indicators, and
    days until next holiday.

    Attributes:
        include_bridge_days: Whether to include bridge day detection feature.
        include_pre_post_indicators: Whether to include pre/post holiday flags.
        include_days_until: Whether to include days until next holiday feature.
        custom_holidays: List of additional custom holidays in YYYY-MM-DD format.
        timezone: Timezone for datetime localization (validates with zoneinfo).

    Example:
        ```python
        config = CalendarFeaturesConfig(
            include_bridge_days=True,
            include_pre_post_indicators=True,
            include_days_until=True,
            custom_holidays=["2024-06-10"],
            timezone="America/Sao_Paulo",
        )
        ```
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
    )

    include_bridge_days: bool = Field(
        default=True,
        description="Whether to include bridge day detection feature",
    )
    include_pre_post_indicators: bool = Field(
        default=True,
        description="Whether to include pre-holiday and post-holiday flags",
    )
    include_days_until: bool = Field(
        default=True,
        description="Whether to include days until next holiday feature",
    )
    custom_holidays: list[str] = Field(
        default_factory=list,
        description="List of additional custom holidays in YYYY-MM-DD format",
    )
    timezone: str = Field(
        default="America/Sao_Paulo",
        description="Timezone for datetime localization (IANA timezone name)",
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate that the timezone is a valid IANA timezone.

        Args:
            v: Timezone string to validate.

        Returns:
            The validated timezone string.

        Raises:
            ValueError: If the timezone is not valid.
        """
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as e:
            msg = f"Invalid timezone '{v}'. Must be a valid IANA timezone name."
            raise ValueError(msg) from e
        return v

    @field_validator("custom_holidays")
    @classmethod
    def validate_custom_holidays(cls, v: list[str]) -> list[str]:
        """Validate that custom holidays are in YYYY-MM-DD format.

        Args:
            v: List of date strings to validate.

        Returns:
            The validated list of date strings.

        Raises:
            ValueError: If any date string is not in YYYY-MM-DD format.
        """
        for date_str in v:
            try:
                date.fromisoformat(date_str)
            except ValueError as e:
                msg = (
                    f"Invalid date format '{date_str}'. "
                    "Custom holidays must be in YYYY-MM-DD format."
                )
                raise ValueError(msg) from e
        return v


class CalendarFeaturesPlugin(BaseFeaturePlugin):
    """Plugin for generating Brazilian holiday and calendar features.

    This plugin generates various calendar-related features including:
    - is_holiday: Binary flag indicating if the date is a Brazilian national holiday
    - is_bridge_day: Binary flag for bridge days (Monday with Tuesday holiday,
      or Friday with Thursday holiday)
    - is_pre_holiday: Binary flag for days before holidays (considering weekends)
    - is_post_holiday: Binary flag for days after holidays (considering weekends)
    - days_until_next_holiday: Number of days until the next holiday (0 if current
      day is a holiday)

    The plugin handles both DatetimeIndex and timestamp columns, and supports
    timezone localization for naive datetimes.

    Example:
        ```python
        plugin = CalendarFeaturesPlugin()
        config = {
            "include_bridge_days": True,
            "include_pre_post_indicators": True,
            "include_days_until": True,
        }
        result = plugin.generate_features(df, config)
        ```
    """

    def __init__(self) -> None:
        """Initialize the CalendarFeaturesPlugin with BrazilianHolidays instance."""
        self._holidays = BrazilianHolidays()

    @property
    def name(self) -> str:
        """Return the unique identifier for this plugin.

        Returns:
            Plugin name "calendar_features".
        """
        return "calendar_features"

    @property
    def version(self) -> str:
        """Return the plugin version string.

        Returns:
            Version string "1.0.0".
        """
        return "1.0.0"

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate calendar and holiday features from the input DataFrame.

        This method extracts date information from the DataFrame's DatetimeIndex
        or a 'timestamp' column if present, then generates holiday-related features.

        Args:
            df: Input DataFrame with DatetimeIndex or 'timestamp' column.
            config: Plugin configuration dictionary. Will be validated against
                CalendarFeaturesConfig.

        Returns:
            DataFrame with additional calendar feature columns.

        Raises:
            ValueError: If the DataFrame has no valid datetime index or column.
            TypeError: If the input DataFrame has incorrect structure.
        """
        # Validate and parse configuration
        parsed_config = CalendarFeaturesConfig(**config)
        logger.debug(
            "Generating calendar features with config: bridge_days=%s, "
            "pre_post=%s, days_until=%s, custom_holidays=%d",
            parsed_config.include_bridge_days,
            parsed_config.include_pre_post_indicators,
            parsed_config.include_days_until,
            len(parsed_config.custom_holidays),
        )

        # Make a copy to avoid modifying the original
        result = df.copy()

        # Get datetime index, handling both DatetimeIndex and timestamp column
        dt_index = self._get_datetime_index(result, parsed_config.timezone)

        # Parse custom holidays
        custom_holidays_set = self._parse_custom_holidays(parsed_config.custom_holidays)

        # Generate is_holiday feature (always included)
        result = self._generate_is_holiday(result, dt_index, custom_holidays_set)

        # Generate bridge day feature if enabled
        if parsed_config.include_bridge_days:
            result = self._generate_bridge_day(result, dt_index, custom_holidays_set)

        # Generate pre/post holiday features if enabled
        if parsed_config.include_pre_post_indicators:
            result = self._generate_pre_post_holiday(result, dt_index, custom_holidays_set)

        # Generate days until next holiday if enabled
        if parsed_config.include_days_until:
            result = self._generate_days_until_holiday(result, dt_index, custom_holidays_set)

        logger.debug(
            "Generated %d calendar features for %d rows",
            len(self.get_feature_names(config)),
            len(result),
        )
        return result

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """Return the list of feature column names this plugin generates.

        Args:
            config: Plugin configuration dictionary.

        Returns:
            List of feature column names that will be generated.
        """
        # Parse configuration to get settings
        parsed_config = CalendarFeaturesConfig(**config)

        feature_names: list[str] = ["is_holiday"]

        if parsed_config.include_bridge_days:
            feature_names.append("is_bridge_day")

        if parsed_config.include_pre_post_indicators:
            feature_names.extend(["is_pre_holiday", "is_post_holiday"])

        if parsed_config.include_days_until:
            feature_names.append("days_until_next_holiday")

        return feature_names

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate the plugin configuration.

        Args:
            config: Plugin configuration dictionary to validate.

        Returns:
            True if the configuration is valid.

        Raises:
            ValueError: If the configuration is invalid.
        """
        # First call parent validation
        super().validate_config(config)

        # Then validate with Pydantic model (will raise on invalid config)
        CalendarFeaturesConfig(**config)
        logger.debug("Configuration validated for plugin '%s'", self.name)
        return True

    def _get_datetime_index(
        self,
        df: pd.DataFrame,
        timezone: str,
    ) -> pd.DatetimeIndex:
        """Extract and localize datetime index from DataFrame.

        Args:
            df: Input DataFrame.
            timezone: Target timezone for localization.

        Returns:
            DatetimeIndex localized to the specified timezone.

        Raises:
            ValueError: If no valid datetime index or column is found.
        """
        # Check if index is DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            dt_index = df.index
        # Check for timestamp column
        elif "timestamp" in df.columns:
            dt_index = pd.DatetimeIndex(df["timestamp"])
        else:
            msg = (
                "DataFrame must have a DatetimeIndex or a 'timestamp' column. "
                f"Found index type: {type(df.index).__name__}, "
                f"columns: {list(df.columns)}"
            )
            raise ValueError(msg)

        # Localize naive datetimes to the specified timezone
        if dt_index.tz is None:
            dt_index = dt_index.tz_localize(timezone)
            logger.debug("Localized naive datetime index to %s", timezone)
        elif str(dt_index.tz) != timezone:
            dt_index = dt_index.tz_convert(timezone)
            logger.debug("Converted datetime index to %s", timezone)

        return dt_index

    def _parse_custom_holidays(self, custom_holidays: list[str]) -> set[date]:
        """Parse custom holiday strings into date objects.

        Args:
            custom_holidays: List of date strings in YYYY-MM-DD format.

        Returns:
            Set of date objects.
        """
        return {date.fromisoformat(d) for d in custom_holidays}

    def _is_holiday_date(self, d: date, custom_holidays: set[date]) -> bool:
        """Check if a date is a holiday (including custom holidays).

        Args:
            d: The date to check.
            custom_holidays: Set of custom holiday dates.

        Returns:
            True if the date is a holiday.
        """
        return self._holidays.is_holiday(d) or d in custom_holidays

    def _generate_is_holiday(
        self,
        df: pd.DataFrame,
        dt_index: pd.DatetimeIndex,
        custom_holidays: set[date],
    ) -> pd.DataFrame:
        """Generate is_holiday feature.

        Args:
            df: DataFrame to add feature to.
            dt_index: DatetimeIndex to extract dates from.
            custom_holidays: Set of custom holiday dates.

        Returns:
            DataFrame with is_holiday feature added.
        """
        dates = dt_index.date
        is_holiday_values = np.array(
            [self._is_holiday_date(d, custom_holidays) for d in dates],
            dtype=np.int32,
        )
        df["is_holiday"] = is_holiday_values
        return df

    def _generate_bridge_day(
        self,
        df: pd.DataFrame,
        dt_index: pd.DatetimeIndex,
        custom_holidays: set[date],
    ) -> pd.DataFrame:
        """Generate is_bridge_day feature.

        Bridge day logic:
        - Monday is a bridge day if Tuesday is a holiday
        - Friday is a bridge day if Thursday is a holiday

        Args:
            df: DataFrame to add feature to.
            dt_index: DatetimeIndex to extract dates from.
            custom_holidays: Set of custom holiday dates.

        Returns:
            DataFrame with is_bridge_day feature added.
        """
        dates = dt_index.date
        day_of_week = dt_index.dayofweek

        is_bridge_values = np.zeros(len(dates), dtype=np.int32)

        for i, (d, dow) in enumerate(zip(dates, day_of_week, strict=True)):
            # Monday with Tuesday holiday
            if dow == MONDAY:
                next_day = d + timedelta(days=1)
                if self._is_holiday_date(next_day, custom_holidays):
                    is_bridge_values[i] = 1

            # Friday with Thursday holiday
            elif dow == FRIDAY:
                prev_day = d - timedelta(days=1)
                if self._is_holiday_date(prev_day, custom_holidays):
                    is_bridge_values[i] = 1

        df["is_bridge_day"] = is_bridge_values
        return df

    def _generate_pre_post_holiday(
        self,
        df: pd.DataFrame,
        dt_index: pd.DatetimeIndex,
        custom_holidays: set[date],
    ) -> pd.DataFrame:
        """Generate is_pre_holiday and is_post_holiday features.

        Pre-holiday logic:
        - Day before a holiday is pre-holiday
        - Friday is pre-holiday if Monday is a holiday

        Post-holiday logic:
        - Day after a holiday is post-holiday
        - Monday is post-holiday if Friday is a holiday

        Args:
            df: DataFrame to add features to.
            dt_index: DatetimeIndex to extract dates from.
            custom_holidays: Set of custom holiday dates.

        Returns:
            DataFrame with pre/post holiday features added.
        """
        dates = dt_index.date
        day_of_week = dt_index.dayofweek

        is_pre_holiday = np.zeros(len(dates), dtype=np.int32)
        is_post_holiday = np.zeros(len(dates), dtype=np.int32)

        for i, (d, dow) in enumerate(zip(dates, day_of_week, strict=True)):
            # Check pre-holiday: next day is a holiday
            next_day = d + timedelta(days=1)
            if self._is_holiday_date(next_day, custom_holidays):
                is_pre_holiday[i] = 1

            # Friday before Monday holiday (considering weekend)
            if dow == FRIDAY:
                monday = d + timedelta(days=3)
                if self._is_holiday_date(monday, custom_holidays):
                    is_pre_holiday[i] = 1

            # Check post-holiday: previous day is a holiday
            prev_day = d - timedelta(days=1)
            if self._is_holiday_date(prev_day, custom_holidays):
                is_post_holiday[i] = 1

            # Monday after Friday holiday (considering weekend)
            if dow == MONDAY:
                friday = d - timedelta(days=3)
                if self._is_holiday_date(friday, custom_holidays):
                    is_post_holiday[i] = 1

        df["is_pre_holiday"] = is_pre_holiday
        df["is_post_holiday"] = is_post_holiday
        return df

    def _generate_days_until_holiday(
        self,
        df: pd.DataFrame,
        dt_index: pd.DatetimeIndex,
        custom_holidays: set[date],
    ) -> pd.DataFrame:
        """Generate days_until_next_holiday feature.

        Returns 0 if the current day is a holiday.

        Args:
            df: DataFrame to add feature to.
            dt_index: DatetimeIndex to extract dates from.
            custom_holidays: Set of custom holiday dates.

        Returns:
            DataFrame with days_until_next_holiday feature added.
        """
        dates = dt_index.date
        days_until = np.zeros(len(dates), dtype=np.int32)

        # Handle empty DataFrame
        if len(dates) == 0:
            df["days_until_next_holiday"] = days_until
            return df

        # Build a set of all holidays for relevant years
        # Get unique years from the data plus next year for boundary handling
        unique_years = {d.year for d in dates}
        max_year = max(unique_years) + 1

        # Build combined holiday set
        all_holidays: set[date] = set(custom_holidays)
        for year in range(min(unique_years), max_year + 1):
            all_holidays.update(self._holidays.get_holidays(year).keys())

        # Sort holidays for efficient searching
        sorted_holidays = sorted(all_holidays)

        for i, d in enumerate(dates):
            if self._is_holiday_date(d, custom_holidays):
                days_until[i] = 0
            else:
                # Find next holiday using binary search
                for holiday in sorted_holidays:
                    if holiday > d:
                        days_until[i] = (holiday - d).days
                        break

        df["days_until_next_holiday"] = days_until
        return df
