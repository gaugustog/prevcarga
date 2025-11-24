"""Plugin registry for feature engineering plugins.

This module provides a singleton registry for managing feature engineering
plugins, including registration, retrieval, and validation of plugin names
and versions.

Example:
    ```python
    from src.features.registry import PluginRegistry
    from src.features.base import BaseFeaturePlugin

    # Get the singleton registry
    registry = PluginRegistry()

    # Register a plugin
    registry.register(MyFeaturePlugin())

    # Retrieve a plugin
    plugin = registry.get_plugin("my_feature")

    # List all plugins
    for name, metadata in registry.list_plugins().items():
        print(f"{name}: v{metadata['version']}")
    ```
"""

import re
import threading
from typing import Any

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Regex pattern for valid plugin names: lowercase letters, digits, underscores
# Must start with a lowercase letter
PLUGIN_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

# Regex pattern for semantic versioning: MAJOR.MINOR.PATCH
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class PluginRegistryError(Exception):
    """Base exception for plugin registry errors.

    Attributes:
        message: Error message.
        plugin_name: Name of the plugin involved (if applicable).
    """

    def __init__(self, message: str, plugin_name: str | None = None) -> None:
        """Initialize PluginRegistryError.

        Args:
            message: Error message.
            plugin_name: Name of the plugin involved.
        """
        self.plugin_name = plugin_name
        super().__init__(message)


class PluginNotFoundError(PluginRegistryError):
    """Exception raised when a plugin is not found in the registry."""

    def __init__(self, plugin_name: str) -> None:
        """Initialize PluginNotFoundError.

        Args:
            plugin_name: Name of the plugin that was not found.
        """
        available = PluginRegistry().list_plugins().keys()
        msg = f"Plugin '{plugin_name}' not found. Available: {list(available)}"
        super().__init__(msg, plugin_name=plugin_name)


class InvalidPluginNameError(PluginRegistryError):
    """Exception raised when a plugin name is invalid."""

    def __init__(self, plugin_name: str) -> None:
        """Initialize InvalidPluginNameError.

        Args:
            plugin_name: The invalid plugin name.
        """
        msg = (
            f"Invalid plugin name '{plugin_name}'. "
            f"Must match pattern: {PLUGIN_NAME_PATTERN.pattern} "
            "(lowercase letters, digits, underscores; must start with letter)"
        )
        super().__init__(msg, plugin_name=plugin_name)


class InvalidVersionError(PluginRegistryError):
    """Exception raised when a plugin version is invalid."""

    def __init__(self, plugin_name: str, version: str) -> None:
        """Initialize InvalidVersionError.

        Args:
            plugin_name: Name of the plugin with invalid version.
            version: The invalid version string.
        """
        msg = (
            f"Invalid version '{version}' for plugin '{plugin_name}'. "
            f"Must match semantic versioning: {VERSION_PATTERN.pattern}"
        )
        super().__init__(msg, plugin_name=plugin_name)


class DuplicatePluginError(PluginRegistryError):
    """Exception raised when attempting to register a duplicate plugin."""

    def __init__(self, plugin_name: str) -> None:
        """Initialize DuplicatePluginError.

        Args:
            plugin_name: Name of the duplicate plugin.
        """
        msg = f"Plugin '{plugin_name}' is already registered"
        super().__init__(msg, plugin_name=plugin_name)


class PluginRegistry:
    """Singleton registry for feature engineering plugins.

    This class implements the singleton pattern to provide a single,
    global registry for all feature engineering plugins. It handles
    plugin registration, retrieval, and metadata management.

    The registry enforces:
        - Unique plugin names
        - Valid plugin name format (lowercase with underscores)
        - Valid semantic version format
        - Thread-safe operations

    Example:
        ```python
        registry = PluginRegistry()

        # Register a plugin
        registry.register(my_plugin)

        # Get a plugin by name
        plugin = registry.get_plugin("my_plugin")

        # List all registered plugins
        for name, meta in registry.list_plugins().items():
            print(f"{name} v{meta['version']}")

        # Unregister a plugin
        registry.unregister("my_plugin")
        ```

    Note:
        This class uses the singleton pattern. All instances of
        PluginRegistry reference the same underlying registry data.
    """

    _instance: "PluginRegistry | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "PluginRegistry":
        """Create or return the singleton instance.

        Returns:
            The singleton PluginRegistry instance.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._plugins: dict[str, BaseFeaturePlugin] = {}
                    cls._instance._registry_lock = threading.RLock()
        return cls._instance

    def register(self, plugin: BaseFeaturePlugin) -> None:
        """Register a feature plugin.

        Args:
            plugin: Plugin instance to register.

        Raises:
            InvalidPluginNameError: If the plugin name format is invalid.
            InvalidVersionError: If the plugin version format is invalid.
            DuplicatePluginError: If a plugin with the same name exists.
            TypeError: If plugin is not a BaseFeaturePlugin instance.
        """
        if not isinstance(plugin, BaseFeaturePlugin):
            msg = f"Plugin must be a BaseFeaturePlugin instance, got {type(plugin)}"
            raise TypeError(msg)

        name = plugin.name
        version = plugin.version

        # Validate plugin name format
        if not PLUGIN_NAME_PATTERN.match(name):
            raise InvalidPluginNameError(name)

        # Validate version format
        if not VERSION_PATTERN.match(version):
            raise InvalidVersionError(name, version)

        with self._registry_lock:
            # Check for duplicates
            if name in self._plugins:
                raise DuplicatePluginError(name)

            self._plugins[name] = plugin
            logger.info("Registered plugin '%s' v%s", name, version)

    def get_plugin(self, name: str) -> BaseFeaturePlugin:
        """Retrieve a plugin by name.

        Args:
            name: The plugin name to retrieve.

        Returns:
            The registered plugin instance.

        Raises:
            PluginNotFoundError: If the plugin is not registered.
        """
        with self._registry_lock:
            if name not in self._plugins:
                raise PluginNotFoundError(name)
            return self._plugins[name]

    def list_plugins(self) -> dict[str, dict[str, Any]]:
        """List all registered plugins with their metadata.

        Returns:
            Dictionary mapping plugin names to their metadata.
            Each metadata dict contains: name, version, class, module.
        """
        with self._registry_lock:
            return {name: plugin.get_metadata() for name, plugin in self._plugins.items()}

    def unregister(self, name: str) -> None:
        """Remove a plugin from the registry.

        Args:
            name: The plugin name to unregister.

        Raises:
            PluginNotFoundError: If the plugin is not registered.
        """
        with self._registry_lock:
            if name not in self._plugins:
                raise PluginNotFoundError(name)

            del self._plugins[name]
            logger.info("Unregistered plugin '%s'", name)

    def clear(self) -> None:
        """Remove all plugins from the registry.

        This method is primarily intended for testing purposes.
        """
        with self._registry_lock:
            count = len(self._plugins)
            self._plugins.clear()
            logger.debug("Cleared %d plugins from registry", count)

    def has_plugin(self, name: str) -> bool:
        """Check if a plugin is registered.

        Args:
            name: The plugin name to check.

        Returns:
            True if the plugin is registered, False otherwise.
        """
        with self._registry_lock:
            return name in self._plugins

    def get_plugin_count(self) -> int:
        """Get the number of registered plugins.

        Returns:
            Number of registered plugins.
        """
        with self._registry_lock:
            return len(self._plugins)

    def __repr__(self) -> str:
        """Return string representation of the registry.

        Returns:
            String showing the number of registered plugins.
        """
        return f"PluginRegistry(plugins={self.get_plugin_count()})"


def register_plugin(plugin: BaseFeaturePlugin) -> BaseFeaturePlugin:
    """Decorator/function to register a plugin with the global registry.

    This can be used as a function or as a decorator on a plugin class.

    Args:
        plugin: Plugin instance to register.

    Returns:
        The registered plugin instance.

    Example:
        ```python
        # As a function
        register_plugin(MyPlugin())

        # In class definition
        @register_plugin
        class MyPlugin(BaseFeaturePlugin):
            ...
        ```
    """
    registry = PluginRegistry()
    registry.register(plugin)
    return plugin


def get_plugin(name: str) -> BaseFeaturePlugin:
    """Get a plugin from the global registry.

    Convenience function for PluginRegistry().get_plugin(name).

    Args:
        name: The plugin name to retrieve.

    Returns:
        The registered plugin instance.

    Raises:
        PluginNotFoundError: If the plugin is not registered.
    """
    return PluginRegistry().get_plugin(name)
