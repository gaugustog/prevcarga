"""Abstract base class for feature engineering plugins.

This module defines the BaseFeaturePlugin abstract class that all feature
engineering plugins must inherit from. It establishes the contract for
feature generation, configuration validation, and metadata retrieval.

Example:
    ```python
    from src.features.base import BaseFeaturePlugin, PluginConfig

    class TemporalPlugin(BaseFeaturePlugin):
        @property
        def name(self) -> str:
            return "temporal"

        @property
        def version(self) -> str:
            return "1.0.0"

        def generate_features(
            self,
            df: pd.DataFrame,
            config: dict[str, Any]
        ) -> pd.DataFrame:
            # Add hour, day of week features
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            return df

        def get_feature_names(self, config: dict[str, Any]) -> list[str]:
            return ["hour", "day_of_week"]
    ```
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseFeaturePlugin(ABC):
    """Abstract base class for feature engineering plugins.

    This class defines the interface that all feature engineering plugins
    must implement. It provides abstract methods for feature generation
    and concrete methods for configuration validation and metadata.

    Subclasses must implement:
        - name: Unique plugin identifier (lowercase with underscores)
        - version: Semantic version string (e.g., "1.0.0")
        - generate_features: Feature generation logic
        - get_feature_names: List of generated feature column names

    Attributes:
        name: Unique identifier for the plugin.
        version: Plugin version following semantic versioning.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this plugin.

        The name must be lowercase with underscores only, matching the
        pattern ^[a-z][a-z0-9_]*$.

        Returns:
            Unique plugin name string.
        """
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the plugin version string.

        The version must follow semantic versioning format: MAJOR.MINOR.PATCH.

        Returns:
            Version string in semantic versioning format (e.g., "1.0.0").
        """
        ...

    @abstractmethod
    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate features from the input DataFrame.

        This method implements the core feature engineering logic for the plugin.
        It should add new columns to the DataFrame or return a new DataFrame
        with additional feature columns.

        Args:
            df: Input DataFrame with timestamp index and data columns.
            config: Plugin-specific configuration dictionary.

        Returns:
            DataFrame with additional feature columns. May be the same
            DataFrame with new columns or a new DataFrame.

        Raises:
            ValueError: If required columns are missing from input DataFrame.
            TypeError: If input DataFrame has incorrect structure.
        """
        ...

    @abstractmethod
    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """Return the list of feature column names this plugin generates.

        This method returns the names of all feature columns that will be
        created by generate_features(). This is used for validation and
        collision detection.

        Args:
            config: Plugin-specific configuration dictionary.

        Returns:
            List of feature column names that will be generated.
        """
        ...

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate the plugin configuration.

        This method validates that the provided configuration is valid
        for this plugin. Override this method in subclasses to add
        custom validation logic.

        Args:
            config: Plugin-specific configuration dictionary to validate.

        Returns:
            True if the configuration is valid.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if config is None:
            msg = "Configuration cannot be None"
            raise ValueError(msg)

        logger.debug("Configuration validated for plugin '%s'", self.name)
        return True

    def get_metadata(self) -> dict[str, Any]:
        """Return plugin metadata as a dictionary.

        This method returns metadata about the plugin including its name,
        version, and class information.

        Returns:
            Dictionary containing plugin metadata:
                - name: Plugin identifier
                - version: Plugin version
                - class: Full class name
                - module: Module containing the plugin
        """
        return {
            "name": self.name,
            "version": self.version,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
        }

    def __repr__(self) -> str:
        """Return string representation of the plugin.

        Returns:
            String in format: PluginClassName(name='plugin_name', version='1.0.0')
        """
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
