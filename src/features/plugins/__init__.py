"""Feature engineering plugins.

This package contains concrete implementations of feature engineering plugins
that extend the BaseFeaturePlugin abstract class.

Available plugins:
    - TemporalFeaturesPlugin: Generates temporal features including hour,
      day of week, month, season, and cyclical encodings.
    - CalendarFeaturesPlugin: Generates Brazilian holiday and calendar features
      including bridge days, pre/post holiday indicators, and days until holiday.
"""

from src.features.plugins.calendar import (
    BrazilianHolidays,
    CalendarFeaturesConfig,
    CalendarFeaturesPlugin,
)
from src.features.plugins.temporal import (
    TemporalFeaturesConfig,
    TemporalFeaturesPlugin,
)

__all__: list[str] = [
    "BrazilianHolidays",
    "CalendarFeaturesConfig",
    "CalendarFeaturesPlugin",
    "TemporalFeaturesConfig",
    "TemporalFeaturesPlugin",
]
