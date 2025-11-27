"""Metrics and statistics for feature engineering.

This module provides various metrics and statistical measures used in
feature engineering, including seasonal strength calculation and other
decomposition quality metrics.

Modules:
    seasonal_strength: Seasonal and trend strength calculation.
"""

from src.features.metrics.seasonal_strength import SeasonalStrengthCalculator

__all__ = [
    "SeasonalStrengthCalculator",
]
