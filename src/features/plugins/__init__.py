"""Feature engineering plugins.

This package contains concrete implementations of feature engineering plugins
that extend the BaseFeaturePlugin abstract class.

Available plugins:
    - TemporalFeaturesPlugin: Generates temporal features including hour,
      day of week, month, season, and cyclical encodings.
    - CalendarFeaturesPlugin: Generates Brazilian holiday and calendar features
      including bridge days, pre/post holiday indicators, and days until holiday.
    - LagFeaturesPlugin: Generates lag (autoregressive) features and rolling
      statistics for time series forecasting.
    - CyclicalEncodingPlugin: Generates cyclical sin/cos encodings for periodic
      features like hour, day of week, and month.
    - LoessSmoothingPlugin: Generates smoothed features using LOESS regression
      for trend extraction and residual analysis.
"""

from src.features.plugins.calendar import (
    BrazilianHolidays,
    CalendarFeaturesConfig,
    CalendarFeaturesPlugin,
)
from src.features.plugins.cyclical import (
    CyclicalEncodingConfig,
    CyclicalEncodingPlugin,
)
from src.features.plugins.lag import (
    LagFeaturesConfig,
    LagFeaturesPlugin,
)
from src.features.plugins.smoothing import (
    LoessSmoothingConfig,
    LoessSmoothingPlugin,
)
from src.features.plugins.temporal import (
    TemporalFeaturesConfig,
    TemporalFeaturesPlugin,
)

__all__: list[str] = [
    "BrazilianHolidays",
    "CalendarFeaturesConfig",
    "CalendarFeaturesPlugin",
    "CyclicalEncodingConfig",
    "CyclicalEncodingPlugin",
    "LagFeaturesConfig",
    "LagFeaturesPlugin",
    "LoessSmoothingConfig",
    "LoessSmoothingPlugin",
    "TemporalFeaturesConfig",
    "TemporalFeaturesPlugin",
]
