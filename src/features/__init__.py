"""Feature engineering plugins and pipeline composer.

This package provides the plugin-based feature engineering architecture
for the PrevCarga electric load forecasting system. It includes:

- BaseFeaturePlugin: Abstract base class for all feature plugins
- PluginConfig: Pydantic configuration model for plugins
- PluginRegistry: Singleton registry for managing plugins
- FeatureValidator: Validation utilities for plugin outputs
- Utility functions for working with feature DataFrames

Example:
    ```python
    from src.features import (
        BaseFeaturePlugin,
        PluginConfig,
        PluginRegistry,
        FeatureValidator,
    )

    # Create a custom feature plugin
    class MyPlugin(BaseFeaturePlugin):
        @property
        def name(self) -> str:
            return "my_plugin"

        @property
        def version(self) -> str:
            return "1.0.0"

        def generate_features(self, df, config):
            df["my_feature"] = df["value"] * 2
            return df

        def get_feature_names(self, config):
            return ["my_feature"]

    # Register the plugin
    registry = PluginRegistry()
    registry.register(MyPlugin())

    # Use the plugin
    plugin = registry.get_plugin("my_plugin")
    output_df = plugin.generate_features(input_df, {})

    # Validate the output
    validator = FeatureValidator()
    report = validator.validate(
        output_df,
        plugin_name="my_plugin",
        expected_columns=["my_feature"]
    )
    ```
"""

from src.features.base.config import PluginConfig
from src.features.base.plugin import BaseFeaturePlugin
from src.features.base.validator import (
    FeatureValidator,
    ValidationReport,
    ValidationResult,
)
from src.features.registry import (
    DuplicatePluginError,
    InvalidPluginNameError,
    InvalidVersionError,
    PluginNotFoundError,
    PluginRegistry,
    PluginRegistryError,
    get_plugin,
    register_plugin,
)
from src.features.plugins.calendar import (
    BrazilianHolidays,
    CalendarFeaturesConfig,
    CalendarFeaturesPlugin,
)
from src.features.plugins.temporal import (
    TemporalFeaturesConfig,
    TemporalFeaturesPlugin,
)
from src.features.utils import (
    FeatureNameCollisionError,
    FeatureUtilsError,
    InvalidIndexError,
    check_feature_names,
    get_feature_statistics,
    merge_feature_dataframes,
    validate_dataframe_index,
)

__all__ = [
    "BaseFeaturePlugin",
    "BrazilianHolidays",
    "CalendarFeaturesConfig",
    "CalendarFeaturesPlugin",
    "DuplicatePluginError",
    "FeatureNameCollisionError",
    "FeatureUtilsError",
    "FeatureValidator",
    "InvalidIndexError",
    "InvalidPluginNameError",
    "InvalidVersionError",
    "PluginConfig",
    "PluginNotFoundError",
    "PluginRegistry",
    "PluginRegistryError",
    "TemporalFeaturesConfig",
    "TemporalFeaturesPlugin",
    "ValidationReport",
    "ValidationResult",
    "check_feature_names",
    "get_feature_statistics",
    "get_plugin",
    "merge_feature_dataframes",
    "register_plugin",
    "validate_dataframe_index",
]
