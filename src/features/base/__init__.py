"""Base classes for feature engineering plugins.

This module provides the foundational components for the plugin-based
feature engineering architecture, including abstract base classes,
configuration models, and validators.
"""

from src.features.base.config import PluginConfig
from src.features.base.plugin import BaseFeaturePlugin
from src.features.base.validator import (
    FeatureValidator,
    ValidationReport,
    ValidationResult,
)

__all__ = [
    "BaseFeaturePlugin",
    "FeatureValidator",
    "PluginConfig",
    "ValidationReport",
    "ValidationResult",
]
