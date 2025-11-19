# PC-014-02A: Calendar/Holiday Features Plugin

**Ticket ID:** PC-014-02A  
**Epic:** [Epic-02A: Core Feature Engineering](../epics/Epic-02A.md)  
**User Story:** US-3  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a plugin for generating calendar and holiday features specific to Brazilian holidays and observances. Include national holidays, bridge days (pontes), pre/post-holiday indicators, and days until next holiday.

**As a** data scientist  
**I want** holiday and calendar features  
**So that** the model can learn special day patterns affecting electricity load

---

## ✅ Acceptance Criteria

- [ ] `CalendarFeaturesPlugin` identifies Brazilian national holidays
- [ ] Bridge day detection (ponte - Monday/Friday adjacent to holidays)
- [ ] Pre-holiday indicator (day before holiday)
- [ ] Post-holiday indicator (day after holiday)
- [ ] Days until next holiday counter
- [ ] Carnival detection (movable holiday)
- [ ] Corpus Christi detection (movable holiday)
- [ ] Configurable holiday calendar with custom dates
- [ ] Integration with existing feriados data
- [ ] Unit tests with multiple years of holidays
- [ ] Performance: <50ms for 1 year of hourly data

---

## 🔧 Implementation Tasks

### 1. Create Plugin Module
- [ ] Create `src/features/plugins/calendar.py`
- [ ] Import BaseFeaturePlugin and dependencies
- [ ] Add module docstring with Brazilian holiday list

### 2. Define Brazilian Holiday Calendar
- [ ] Create `BrazilianHolidays` class
- [ ] Define fixed national holidays:
  - New Year (Jan 1)
  - Tiradentes Day (Apr 21)
  - Labor Day (May 1)
  - Independence Day (Sep 7)
  - Our Lady of Aparecida (Oct 12)
  - All Souls' Day (Nov 2)
  - Republic Day (Nov 15)
  - Black Consciousness Day (Nov 20)
  - Christmas (Dec 25)
- [ ] Implement movable holiday calculations:
  - Carnival (47 days before Easter)
  - Good Friday (2 days before Easter)
  - Corpus Christi (60 days after Easter)
- [ ] Support custom regional holidays
- [ ] Cache holiday dates by year

### 3. Implement CalendarFeaturesPlugin Class
- [ ] Inherit from `BaseFeaturePlugin`
- [ ] Implement `name` property returning "calendar_features"
- [ ] Implement `version` property with semantic versioning
- [ ] Add class docstring with usage examples

### 4. Implement Holiday Detection
- [ ] Create `_is_holiday()` method
- [ ] Check against national holiday list
- [ ] Check against custom holiday list
- [ ] Handle timezone-aware dates
- [ ] Return binary indicator (0/1)

### 5. Implement Bridge Day Detection
- [ ] Create `_is_bridge_day()` method
- [ ] Check if Monday with Tuesday holiday
- [ ] Check if Friday with Thursday holiday
- [ ] Consider weekend adjacency
- [ ] Return binary indicator (0/1)

### 6. Implement Pre/Post Holiday Indicators
- [ ] Create `_is_pre_holiday()` method (day before)
- [ ] Create `_is_post_holiday()` method (day after)
- [ ] Handle weekends (pre-holiday Friday if holiday Monday)
- [ ] Return binary indicators (0/1)

### 7. Implement Days Until Next Holiday
- [ ] Create `_days_until_next_holiday()` method
- [ ] Calculate days from current date to next holiday
- [ ] Handle year boundaries
- [ ] Return integer count (0 if today is holiday)

### 8. Implement Configuration Schema
- [ ] Create `CalendarFeaturesConfig` Pydantic model
- [ ] Add `include_bridge_days` boolean flag (default: True)
- [ ] Add `include_pre_post_indicators` boolean flag (default: True)
- [ ] Add `include_days_until` boolean flag (default: True)
- [ ] Add `custom_holidays` list of dates
- [ ] Add `feriados_df` optional DataFrame parameter
- [ ] Add `timezone` string field (default: "America/Sao_Paulo")

### 9. Implement generate_features() Method
- [ ] Validate input DataFrame has DatetimeIndex
- [ ] Convert timezone if needed
- [ ] Generate is_holiday feature
- [ ] Generate is_bridge_day if configured
- [ ] Generate is_pre_holiday if configured
- [ ] Generate is_post_holiday if configured
- [ ] Generate days_until_next_holiday if configured
- [ ] Return DataFrame with same index as input
- [ ] Add error handling for invalid inputs

### 10. Implement get_feature_names() Method
- [ ] Return list based on configuration
- [ ] Include is_holiday
- [ ] Include bridge day features if enabled
- [ ] Include pre/post indicators if enabled
- [ ] Include days until counter if enabled
- [ ] Maintain consistent naming convention

### 11. Integrate with Feriados Data
- [ ] Support loading from PC-011-01 auxiliary data
- [ ] Merge external feriados DataFrame
- [ ] Deduplicate holiday dates
- [ ] Validate feriados schema

### 12. Write Comprehensive Tests
- [ ] Create `tests/features/plugins/test_calendar.py`
- [ ] Test fixed holiday detection
- [ ] Test movable holiday calculation (Carnival, Corpus Christi)
- [ ] Test bridge day logic
- [ ] Test pre/post holiday indicators
- [ ] Test days until next holiday
- [ ] Test custom holiday addition
- [ ] Test feriados integration
- [ ] Test edge cases (year boundaries, leap years)
- [ ] Test performance with large datasets

### 13. Create Holiday Calendar Documentation
- [ ] Document all Brazilian national holidays
- [ ] Document movable holiday calculation methods
- [ ] Create examples with visualization
- [ ] Add regional holiday configuration guide

---

## 💻 Implementation Details

### Brazilian Holidays Class

```python
"""Brazilian holiday calendar utilities."""
from datetime import date, timedelta
from typing import Set, List
import pandas as pd
from dateutil.easter import easter


class BrazilianHolidays:
    """
    Brazilian national holiday calendar.
    
    Includes fixed and movable holidays according to Brazilian law.
    """
    
    # Fixed national holidays (month, day)
    FIXED_HOLIDAYS = {
        (1, 1): "Ano Novo",
        (4, 21): "Tiradentes",
        (5, 1): "Dia do Trabalho",
        (9, 7): "Independência do Brasil",
        (10, 12): "Nossa Senhora Aparecida",
        (11, 2): "Finados",
        (11, 15): "Proclamação da República",
        (11, 20): "Consciência Negra",
        (12, 25): "Natal"
    }
    
    @classmethod
    def get_holidays(cls, year: int) -> Set[date]:
        """
        Get all national holidays for a given year.
        
        Args:
            year: Year to get holidays for
        
        Returns:
            Set of holiday dates
        """
        holidays = set()
        
        # Add fixed holidays
        for (month, day), name in cls.FIXED_HOLIDAYS.items():
            holidays.add(date(year, month, day))
        
        # Add movable holidays
        easter_date = easter(year)
        
        # Carnival (47 days before Easter)
        carnival = easter_date - timedelta(days=47)
        holidays.add(carnival)
        
        # Good Friday (2 days before Easter)
        good_friday = easter_date - timedelta(days=2)
        holidays.add(good_friday)
        
        # Corpus Christi (60 days after Easter)
        corpus_christi = easter_date + timedelta(days=60)
        holidays.add(corpus_christi)
        
        return holidays
    
    @classmethod
    def get_holiday_name(cls, dt: date, year_holidays: Set[date]) -> str:
        """Get holiday name for a given date."""
        if dt not in year_holidays:
            return ""
        
        # Check fixed holidays
        key = (dt.month, dt.day)
        if key in cls.FIXED_HOLIDAYS:
            return cls.FIXED_HOLIDAYS[key]
        
        # Check movable holidays
        easter_date = easter(dt.year)
        if dt == easter_date - timedelta(days=47):
            return "Carnaval"
        elif dt == easter_date - timedelta(days=2):
            return "Sexta-feira Santa"
        elif dt == easter_date + timedelta(days=60):
            return "Corpus Christi"
        
        return "Feriado"
```

### CalendarFeaturesPlugin Implementation

```python
"""Calendar and holiday features plugin."""
from typing import Dict, Any, List, Optional, Set
from datetime import date, timedelta
import pandas as pd
from pydantic import BaseModel, Field
import pytz

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CalendarFeaturesConfig(BaseModel):
    """Configuration for calendar features plugin."""
    
    include_bridge_days: bool = Field(
        default=True,
        description="Include bridge day indicators"
    )
    include_pre_post_indicators: bool = Field(
        default=True,
        description="Include pre/post holiday indicators"
    )
    include_days_until: bool = Field(
        default=True,
        description="Include days until next holiday counter"
    )
    custom_holidays: List[str] = Field(
        default_factory=list,
        description="Custom holiday dates (YYYY-MM-DD format)"
    )
    timezone: str = Field(
        default="America/Sao_Paulo",
        description="Timezone for datetime localization"
    )


class CalendarFeaturesPlugin(BaseFeaturePlugin):
    """
    Plugin for generating calendar and holiday features.
    
    Features include:
    - is_holiday: Binary indicator for Brazilian national holidays
    - is_bridge_day: Bridge day indicator (Monday/Friday adjacent to holiday)
    - is_pre_holiday: Day before holiday
    - is_post_holiday: Day after holiday
    - days_until_next_holiday: Days until next holiday
    
    Example:
        >>> plugin = CalendarFeaturesPlugin()
        >>> config = {
        ...     "include_bridge_days": True,
        ...     "include_pre_post_indicators": True,
        ...     "custom_holidays": ["2024-02-13"]  # Carnival
        ... }
        >>> features = plugin.generate_features(df, config)
    """
    
    def __init__(self):
        """Initialize plugin with holiday cache."""
        self._holiday_cache: Dict[int, Set[date]] = {}
    
    @property
    def name(self) -> str:
        return "calendar_features"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration using Pydantic model."""
        CalendarFeaturesConfig(**config)
        logger.debug("Calendar features config validated")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate calendar and holiday features.
        
        Args:
            df: Input DataFrame with DatetimeIndex
            config: Plugin configuration
        
        Returns:
            DataFrame with calendar features
        
        Raises:
            ValueError: If DataFrame doesn't have DatetimeIndex
        """
        # Validate config
        validated_config = CalendarFeaturesConfig(**config)
        
        # Validate input
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        logger.info(f"Generating calendar features for {len(df)} records")
        
        # Ensure timezone
        dt_index = df.index
        if dt_index.tz is None:
            tz = pytz.timezone(validated_config.timezone)
            dt_index = dt_index.tz_localize(tz)
        
        # Get unique years and load holidays
        years = dt_index.year.unique()
        for year in years:
            if year not in self._holiday_cache:
                self._holiday_cache[year] = BrazilianHolidays.get_holidays(year)
        
        # Add custom holidays
        custom_dates = set()
        for holiday_str in validated_config.custom_holidays:
            custom_dates.add(pd.to_datetime(holiday_str).date())
        
        # Merge all holidays
        all_holidays = set()
        for year_holidays in self._holiday_cache.values():
            all_holidays.update(year_holidays)
        all_holidays.update(custom_dates)
        
        # Initialize feature dictionary
        features = {}
        
        # Generate is_holiday feature
        features["is_holiday"] = dt_index.date.map(
            lambda d: 1 if d in all_holidays else 0
        )
        
        # Generate bridge day feature
        if validated_config.include_bridge_days:
            features["is_bridge_day"] = dt_index.date.map(
                lambda d: self._is_bridge_day(d, all_holidays)
            )
        
        # Generate pre/post holiday indicators
        if validated_config.include_pre_post_indicators:
            features["is_pre_holiday"] = dt_index.date.map(
                lambda d: self._is_pre_holiday(d, all_holidays)
            )
            features["is_post_holiday"] = dt_index.date.map(
                lambda d: self._is_post_holiday(d, all_holidays)
            )
        
        # Generate days until next holiday
        if validated_config.include_days_until:
            sorted_holidays = sorted(all_holidays)
            features["days_until_next_holiday"] = dt_index.date.map(
                lambda d: self._days_until_next_holiday(d, sorted_holidays)
            )
        
        # Create DataFrame
        result = pd.DataFrame(features, index=df.index)
        
        logger.info(f"Generated {len(result.columns)} calendar features")
        return result
    
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """Get list of feature names based on configuration."""
        validated_config = CalendarFeaturesConfig(**config)
        
        feature_names = ["is_holiday"]
        
        if validated_config.include_bridge_days:
            feature_names.append("is_bridge_day")
        
        if validated_config.include_pre_post_indicators:
            feature_names.extend(["is_pre_holiday", "is_post_holiday"])
        
        if validated_config.include_days_until:
            feature_names.append("days_until_next_holiday")
        
        return feature_names
    
    @staticmethod
    def _is_bridge_day(dt: date, holidays: Set[date]) -> int:
        """
        Check if date is a bridge day.
        
        Bridge day = Monday with Tuesday holiday or Friday with Thursday holiday
        
        Args:
            dt: Date to check
            holidays: Set of holiday dates
        
        Returns:
            1 if bridge day, 0 otherwise
        """
        weekday = dt.weekday()
        
        # Monday (0) with Tuesday holiday
        if weekday == 0:
            next_day = dt + timedelta(days=1)
            if next_day in holidays:
                return 1
        
        # Friday (4) with Thursday holiday
        elif weekday == 4:
            prev_day = dt - timedelta(days=1)
            if prev_day in holidays:
                return 1
        
        return 0
    
    @staticmethod
    def _is_pre_holiday(dt: date, holidays: Set[date]) -> int:
        """Check if date is day before holiday."""
        next_day = dt + timedelta(days=1)
        
        # If next day is holiday (not Saturday/Sunday)
        if next_day in holidays and next_day.weekday() < 5:
            return 1
        
        # If Friday and Monday is holiday
        if dt.weekday() == 4:  # Friday
            monday = dt + timedelta(days=3)
            if monday in holidays:
                return 1
        
        return 0
    
    @staticmethod
    def _is_post_holiday(dt: date, holidays: Set[date]) -> int:
        """Check if date is day after holiday."""
        prev_day = dt - timedelta(days=1)
        
        # If previous day is holiday (not Saturday/Sunday)
        if prev_day in holidays and prev_day.weekday() < 5:
            return 1
        
        # If Monday and Friday was holiday
        if dt.weekday() == 0:  # Monday
            friday = dt - timedelta(days=3)
            if friday in holidays:
                return 1
        
        return 0
    
    @staticmethod
    def _days_until_next_holiday(dt: date, sorted_holidays: List[date]) -> int:
        """
        Calculate days until next holiday.
        
        Args:
            dt: Current date
            sorted_holidays: List of holiday dates (sorted)
        
        Returns:
            Number of days until next holiday (0 if today is holiday)
        """
        if dt in sorted_holidays:
            return 0
        
        # Find next holiday
        for holiday in sorted_holidays:
            if holiday > dt:
                return (holiday - dt).days
        
        # No more holidays this year, return large number
        return 365
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for calendar features plugin."""
import pytest
import pandas as pd
from datetime import date, datetime

from src.features.plugins.calendar import (
    CalendarFeaturesPlugin,
    BrazilianHolidays
)


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return CalendarFeaturesPlugin()


@pytest.fixture
def sample_df():
    """Create sample DataFrame with full year."""
    dates = pd.date_range(
        start="2024-01-01",
        end="2024-12-31",
        freq="D",
        tz="America/Sao_Paulo"
    )
    return pd.DataFrame({"value": range(len(dates))}, index=dates)


def test_fixed_holidays():
    """Test fixed holiday detection."""
    holidays_2024 = BrazilianHolidays.get_holidays(2024)
    
    assert date(2024, 1, 1) in holidays_2024  # New Year
    assert date(2024, 4, 21) in holidays_2024  # Tiradentes
    assert date(2024, 5, 1) in holidays_2024  # Labor Day
    assert date(2024, 9, 7) in holidays_2024  # Independence
    assert date(2024, 12, 25) in holidays_2024  # Christmas


def test_movable_holidays():
    """Test movable holiday calculations."""
    holidays_2024 = BrazilianHolidays.get_holidays(2024)
    
    # Carnival 2024: February 13
    assert date(2024, 2, 13) in holidays_2024
    
    # Good Friday 2024: March 29
    assert date(2024, 3, 29) in holidays_2024
    
    # Corpus Christi 2024: May 30
    assert date(2024, 5, 30) in holidays_2024


def test_is_holiday_feature(plugin, sample_df):
    """Test holiday detection in features."""
    config = {}
    result = plugin.generate_features(sample_df, config)
    
    assert "is_holiday" in result.columns
    
    # Check New Year is marked
    assert result.loc["2024-01-01", "is_holiday"] == 1
    
    # Check regular day is not marked
    assert result.loc["2024-01-02", "is_holiday"] == 0


def test_bridge_day_detection(plugin):
    """Test bridge day logic."""
    # Create DataFrame with known bridge day scenario
    # If April 21 (Tiradentes) is Tuesday, April 20 (Monday) is bridge day
    dates = pd.date_range("2020-04-19", "2020-04-22", freq="D")
    df = pd.DataFrame({"value": [1, 2, 3, 4]}, index=dates)
    
    config = {"include_bridge_days": True}
    result = plugin.generate_features(df, config)
    
    # April 21, 2020 is Tuesday (holiday)
    # April 20, 2020 is Monday (should be bridge day)
    assert result.loc["2020-04-20", "is_bridge_day"] == 1


def test_pre_post_holiday_indicators(plugin, sample_df):
    """Test pre and post holiday indicators."""
    config = {"include_pre_post_indicators": True}
    result = plugin.generate_features(sample_df, config)
    
    assert "is_pre_holiday" in result.columns
    assert "is_post_holiday" in result.columns
    
    # December 24 should be pre-holiday (Christmas is Dec 25)
    assert result.loc["2024-12-24", "is_pre_holiday"] == 1
    
    # December 26 should be post-holiday
    assert result.loc["2024-12-26", "is_post_holiday"] == 1


def test_days_until_next_holiday(plugin, sample_df):
    """Test days until next holiday counter."""
    config = {"include_days_until": True}
    result = plugin.generate_features(sample_df, config)
    
    assert "days_until_next_holiday" in result.columns
    
    # January 1 is holiday, should be 0
    assert result.loc["2024-01-01", "days_until_next_holiday"] == 0
    
    # Days should decrease as approaching holiday
    dec_23 = result.loc["2024-12-23", "days_until_next_holiday"]
    dec_24 = result.loc["2024-12-24", "days_until_next_holiday"]
    assert dec_24 < dec_23


def test_custom_holidays(plugin):
    """Test adding custom holidays."""
    dates = pd.date_range("2024-06-01", "2024-06-10", freq="D")
    df = pd.DataFrame({"value": range(len(dates))}, index=dates)
    
    config = {"custom_holidays": ["2024-06-05"]}
    result = plugin.generate_features(df, config)
    
    assert result.loc["2024-06-05", "is_holiday"] == 1


def test_feature_names(plugin):
    """Test feature name generation."""
    config = {
        "include_bridge_days": True,
        "include_pre_post_indicators": True,
        "include_days_until": True
    }
    names = plugin.get_feature_names(config)
    
    assert "is_holiday" in names
    assert "is_bridge_day" in names
    assert "is_pre_holiday" in names
    assert "is_post_holiday" in names
    assert "days_until_next_holiday" in names


def test_performance(plugin, sample_df):
    """Test performance with full year of daily data."""
    import time
    
    config = {}
    start = time.time()
    result = plugin.generate_features(sample_df, config)
    elapsed = time.time() - start
    
    # Should process 1 year in <50ms
    assert elapsed < 0.05
    assert len(result) == len(sample_df)
```

---

## 📝 Technical Notes

- Easter date calculation uses `dateutil.easter` for accuracy
- Bridge days are common in Brazil (extending holidays)
- Carnival date: 47 days before Easter (Shrove Tuesday)
- Corpus Christi: 60 days after Easter (always Thursday)
- Cache holidays by year to avoid recalculation
- Handle timezone-aware dates consistently

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation
- PC-011-01: Load Auxiliary Data (optional integration)

**Blocks:**
- PC-017-02A: Feature Pipeline Composer

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] CalendarFeaturesPlugin implemented
- [ ] Brazilian holiday calendar complete
- [ ] Movable holiday calculations correct
- [ ] Bridge day detection working
- [ ] Unit tests pass with >90% coverage
- [ ] Performance benchmark met (<50ms for 1 year daily data)
- [ ] Holiday documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-013-02A: Temporal Features Plugin](PC-013-02A-temporal-features-plugin.md)  
**Next:** [PC-015-02A: Lag Features Plugin](PC-015-02A-lag-features-plugin.md)
