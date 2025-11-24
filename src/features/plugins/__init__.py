"""Feature engineering plugins.

This package contains concrete implementations of feature engineering plugins
that extend the BaseFeaturePlugin abstract class.

Available plugins:
    - TemporalFeaturesPlugin: Generates temporal features including hour,
      day of week, month, season, and cyclical encodings.
"""

from src.features.plugins.temporal import (
    TemporalFeaturesConfig,
    TemporalFeaturesPlugin,
)

__all__: list[str] = [
    "TemporalFeaturesConfig",
    "TemporalFeaturesPlugin",
]
