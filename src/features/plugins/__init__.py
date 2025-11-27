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
    - RFFeatureSelectorPlugin: Performs Random Forest-based feature selection
      with horizon-aware leakage prevention for multi-horizon forecasting.
    - SeasonalityPlugin: Generates seasonal decomposition features using STL/MSTL
      for extracting trend, seasonal patterns, and residuals.
"""

from src.features.plugins.blf import (
    BLFStrategyConfig,
    BLFStrategyPlugin,
)
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
from src.features.plugins.rf_selector import (
    RFFeatureSelectorConfig,
    RFFeatureSelectorPlugin,
)
from src.features.plugins.seasonality import (
    SeasonalityConfig,
    SeasonalityPlugin,
)
from src.features.plugins.smoothing import (
    LoessSmoothingConfig,
    LoessSmoothingPlugin,
)
from src.features.plugins.temporal import (
    TemporalFeaturesConfig,
    TemporalFeaturesPlugin,
)
from src.features.plugins.wavelet import (
    WaveletTransformConfig,
    WaveletTransformPlugin,
)

__all__: list[str] = [
    "BLFStrategyConfig",
    "BLFStrategyPlugin",
    "BrazilianHolidays",
    "CalendarFeaturesConfig",
    "CalendarFeaturesPlugin",
    "CyclicalEncodingConfig",
    "CyclicalEncodingPlugin",
    "LagFeaturesConfig",
    "LagFeaturesPlugin",
    "LoessSmoothingConfig",
    "LoessSmoothingPlugin",
    "RFFeatureSelectorConfig",
    "RFFeatureSelectorPlugin",
    "SeasonalityConfig",
    "SeasonalityPlugin",
    "TemporalFeaturesConfig",
    "TemporalFeaturesPlugin",
    "WaveletTransformConfig",
    "WaveletTransformPlugin",
]
