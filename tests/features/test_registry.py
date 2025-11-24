"""Tests for PluginRegistry.

This module contains comprehensive tests for the PluginRegistry singleton class,
including registration, retrieval, validation, and error handling.
"""

from typing import Any

import pandas as pd
import pytest

from src.features.base.plugin import BaseFeaturePlugin
from src.features.registry import (
    PLUGIN_NAME_PATTERN,
    VERSION_PATTERN,
    DuplicatePluginError,
    InvalidPluginNameError,
    InvalidVersionError,
    PluginNotFoundError,
    PluginRegistry,
    get_plugin,
    register_plugin,
)


class MockFeaturePlugin(BaseFeaturePlugin):
    """Mock feature plugin for testing."""

    def __init__(
        self,
        name: str = "mock_plugin",
        version: str = "1.0.0",
    ) -> None:
        """Initialize mock plugin.

        Args:
            name: Plugin name.
            version: Plugin version.
        """
        self._name = name
        self._version = version

    @property
    def name(self) -> str:
        """Return mock plugin name."""
        return self._name

    @property
    def version(self) -> str:
        """Return mock plugin version."""
        return self._version

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate mock features."""
        return df.copy()

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """Return mock feature names."""
        return ["mock_feature"]


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the registry before and after each test."""
    registry = PluginRegistry()
    registry.clear()
    yield
    registry.clear()


class TestPluginRegistrySingleton:
    """Tests for PluginRegistry singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """Test that multiple calls return the same instance."""
        registry1 = PluginRegistry()
        registry2 = PluginRegistry()

        assert registry1 is registry2

    def test_singleton_shares_plugins(self):
        """Test that singleton instances share plugin data."""
        registry1 = PluginRegistry()
        registry2 = PluginRegistry()

        plugin = MockFeaturePlugin(name="shared_plugin")
        registry1.register(plugin)

        assert registry2.has_plugin("shared_plugin")

    def test_repr(self):
        """Test __repr__ method."""
        registry = PluginRegistry()
        registry.register(MockFeaturePlugin(name="plugin_a"))
        registry.register(MockFeaturePlugin(name="plugin_b"))

        repr_str = repr(registry)
        assert "PluginRegistry" in repr_str
        assert "plugins=2" in repr_str


class TestPluginRegistration:
    """Tests for plugin registration."""

    def test_register_valid_plugin(self):
        """Test registering a valid plugin."""
        registry = PluginRegistry()
        plugin = MockFeaturePlugin()

        registry.register(plugin)

        assert registry.has_plugin("mock_plugin")

    def test_register_returns_none(self):
        """Test that register returns None."""
        registry = PluginRegistry()
        plugin = MockFeaturePlugin()

        result = registry.register(plugin)

        assert result is None

    def test_register_plugin_is_registered(self):
        """Test that registration adds plugin to registry."""
        registry = PluginRegistry()
        plugin = MockFeaturePlugin(name="logging_test")

        registry.register(plugin)

        # Verify plugin was registered successfully
        assert registry.has_plugin("logging_test")
        retrieved = registry.get_plugin("logging_test")
        assert retrieved is plugin

    def test_register_multiple_plugins(self):
        """Test registering multiple unique plugins."""
        registry = PluginRegistry()

        registry.register(MockFeaturePlugin(name="plugin_a"))
        registry.register(MockFeaturePlugin(name="plugin_b"))
        registry.register(MockFeaturePlugin(name="plugin_c"))

        assert registry.get_plugin_count() == 3
        assert registry.has_plugin("plugin_a")
        assert registry.has_plugin("plugin_b")
        assert registry.has_plugin("plugin_c")


class TestPluginNameValidation:
    """Tests for plugin name validation."""

    @pytest.mark.parametrize(
        "valid_name",
        [
            "simple",
            "with_underscore",
            "plugin123",
            "a",
            "feature_extraction_v2",
            "temporal_features",
        ],
    )
    def test_valid_plugin_names(self, valid_name: str):
        """Test that valid plugin names are accepted."""
        registry = PluginRegistry()
        plugin = MockFeaturePlugin(name=valid_name)

        registry.register(plugin)

        assert registry.has_plugin(valid_name)

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "Invalid",  # uppercase
            "UPPERCASE",  # all uppercase
            "with-hyphen",  # hyphen
            "with.dot",  # dot
            "with space",  # space
            "123starts_with_number",  # starts with number
            "_underscore_start",  # starts with underscore
            "",  # empty
        ],
    )
    def test_invalid_plugin_names(self, invalid_name: str):
        """Test that invalid plugin names are rejected."""
        registry = PluginRegistry()
        plugin = MockFeaturePlugin(name=invalid_name)

        with pytest.raises(InvalidPluginNameError) as exc_info:
            registry.register(plugin)

        assert invalid_name in str(exc_info.value) or "Invalid plugin name" in str(
            exc_info.value
        )

    def test_plugin_name_pattern(self):
        """Test the plugin name regex pattern."""
        # Valid names
        assert PLUGIN_NAME_PATTERN.match("valid")
        assert PLUGIN_NAME_PATTERN.match("with_underscore")
        assert PLUGIN_NAME_PATTERN.match("a1b2c3")

        # Invalid names
        assert not PLUGIN_NAME_PATTERN.match("Invalid")
        assert not PLUGIN_NAME_PATTERN.match("123invalid")
        assert not PLUGIN_NAME_PATTERN.match("_invalid")
        assert not PLUGIN_NAME_PATTERN.match("")


class TestVersionValidation:
    """Tests for version validation."""

    @pytest.mark.parametrize(
        "valid_version",
        [
            "0.0.0",
            "1.0.0",
            "1.2.3",
            "10.20.30",
            "123.456.789",
        ],
    )
    def test_valid_versions(self, valid_version: str):
        """Test that valid semantic versions are accepted."""
        registry = PluginRegistry()
        plugin = MockFeaturePlugin(name="version_test", version=valid_version)

        registry.register(plugin)

        registered = registry.get_plugin("version_test")
        assert registered.version == valid_version

    @pytest.mark.parametrize(
        "invalid_version",
        [
            "1.0",  # missing patch
            "1",  # only major
            "v1.0.0",  # prefix
            "1.0.0-alpha",  # suffix
            "1.0.0.0",  # extra segment
            "a.b.c",  # non-numeric
            "",  # empty
            "1.0.0-rc1",  # pre-release suffix
        ],
    )
    def test_invalid_versions(self, invalid_version: str):
        """Test that invalid versions are rejected."""
        registry = PluginRegistry()
        plugin = MockFeaturePlugin(name="version_test", version=invalid_version)

        with pytest.raises(InvalidVersionError) as exc_info:
            registry.register(plugin)

        assert invalid_version in str(exc_info.value) or "Invalid version" in str(
            exc_info.value
        )

    def test_version_pattern(self):
        """Test the version regex pattern."""
        # Valid versions
        assert VERSION_PATTERN.match("1.0.0")
        assert VERSION_PATTERN.match("0.0.1")
        assert VERSION_PATTERN.match("99.99.99")

        # Invalid versions
        assert not VERSION_PATTERN.match("1.0")
        assert not VERSION_PATTERN.match("v1.0.0")
        assert not VERSION_PATTERN.match("1.0.0-alpha")


class TestDuplicateRegistration:
    """Tests for duplicate plugin registration."""

    def test_duplicate_registration_raises(self):
        """Test that registering a duplicate plugin raises error."""
        registry = PluginRegistry()
        plugin1 = MockFeaturePlugin(name="duplicate_test")
        plugin2 = MockFeaturePlugin(name="duplicate_test", version="2.0.0")

        registry.register(plugin1)

        with pytest.raises(DuplicatePluginError) as exc_info:
            registry.register(plugin2)

        assert "duplicate_test" in str(exc_info.value)
        assert "already registered" in str(exc_info.value)

    def test_can_register_after_unregister(self):
        """Test that a plugin can be re-registered after unregistration."""
        registry = PluginRegistry()
        plugin1 = MockFeaturePlugin(name="reregister_test", version="1.0.0")
        plugin2 = MockFeaturePlugin(name="reregister_test", version="2.0.0")

        registry.register(plugin1)
        registry.unregister("reregister_test")
        registry.register(plugin2)

        registered = registry.get_plugin("reregister_test")
        assert registered.version == "2.0.0"


class TestPluginRetrieval:
    """Tests for plugin retrieval."""

    def test_get_plugin_returns_registered_plugin(self):
        """Test that get_plugin returns the registered plugin."""
        registry = PluginRegistry()
        plugin = MockFeaturePlugin(name="retrieval_test")
        registry.register(plugin)

        retrieved = registry.get_plugin("retrieval_test")

        assert retrieved is plugin
        assert retrieved.name == "retrieval_test"

    def test_get_plugin_not_found_raises(self):
        """Test that get_plugin raises for non-existent plugin."""
        registry = PluginRegistry()

        with pytest.raises(PluginNotFoundError) as exc_info:
            registry.get_plugin("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_has_plugin_true(self):
        """Test has_plugin returns True for registered plugin."""
        registry = PluginRegistry()
        registry.register(MockFeaturePlugin(name="exists"))

        assert registry.has_plugin("exists") is True

    def test_has_plugin_false(self):
        """Test has_plugin returns False for non-existent plugin."""
        registry = PluginRegistry()

        assert registry.has_plugin("nonexistent") is False


class TestPluginListing:
    """Tests for listing plugins."""

    def test_list_plugins_empty(self):
        """Test list_plugins returns empty dict when no plugins."""
        registry = PluginRegistry()

        plugins = registry.list_plugins()

        assert plugins == {}

    def test_list_plugins_returns_metadata(self):
        """Test list_plugins returns metadata for all plugins."""
        registry = PluginRegistry()
        registry.register(MockFeaturePlugin(name="plugin_a", version="1.0.0"))
        registry.register(MockFeaturePlugin(name="plugin_b", version="2.0.0"))

        plugins = registry.list_plugins()

        assert len(plugins) == 2
        assert "plugin_a" in plugins
        assert "plugin_b" in plugins
        assert plugins["plugin_a"]["version"] == "1.0.0"
        assert plugins["plugin_b"]["version"] == "2.0.0"

    def test_list_plugins_metadata_structure(self):
        """Test that list_plugins metadata has correct structure."""
        registry = PluginRegistry()
        registry.register(MockFeaturePlugin(name="metadata_test"))

        plugins = registry.list_plugins()

        metadata = plugins["metadata_test"]
        assert "name" in metadata
        assert "version" in metadata
        assert "class" in metadata
        assert "module" in metadata


class TestPluginUnregistration:
    """Tests for plugin unregistration."""

    def test_unregister_removes_plugin(self):
        """Test that unregister removes the plugin."""
        registry = PluginRegistry()
        registry.register(MockFeaturePlugin(name="unregister_test"))

        registry.unregister("unregister_test")

        assert registry.has_plugin("unregister_test") is False

    def test_unregister_nonexistent_raises(self):
        """Test that unregister raises for non-existent plugin."""
        registry = PluginRegistry()

        with pytest.raises(PluginNotFoundError):
            registry.unregister("nonexistent")

    def test_unregister_plugin_is_removed(self):
        """Test that unregistration removes plugin from registry."""
        registry = PluginRegistry()
        registry.register(MockFeaturePlugin(name="unregister_log"))

        # Verify plugin is registered before unregister
        assert registry.has_plugin("unregister_log")

        registry.unregister("unregister_log")

        # Verify plugin was removed
        assert not registry.has_plugin("unregister_log")


class TestRegistryClear:
    """Tests for clearing the registry."""

    def test_clear_removes_all_plugins(self):
        """Test that clear removes all plugins."""
        registry = PluginRegistry()
        registry.register(MockFeaturePlugin(name="clear_a"))
        registry.register(MockFeaturePlugin(name="clear_b"))

        registry.clear()

        assert registry.get_plugin_count() == 0
        assert registry.list_plugins() == {}

    def test_clear_empty_registry(self):
        """Test that clear works on empty registry."""
        registry = PluginRegistry()

        registry.clear()  # Should not raise

        assert registry.get_plugin_count() == 0


class TestRegistryTypeValidation:
    """Tests for type validation in registry."""

    def test_register_non_plugin_raises(self):
        """Test that registering non-plugin raises TypeError."""
        registry = PluginRegistry()

        with pytest.raises(TypeError) as exc_info:
            registry.register("not a plugin")  # type: ignore[arg-type]

        assert "BaseFeaturePlugin" in str(exc_info.value)

    def test_register_dict_raises(self):
        """Test that registering dict raises TypeError."""
        registry = PluginRegistry()

        with pytest.raises(TypeError):
            registry.register({"name": "fake"})  # type: ignore[arg-type]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_register_plugin_function(self):
        """Test register_plugin convenience function."""
        plugin = MockFeaturePlugin(name="convenience_test")

        result = register_plugin(plugin)

        assert result is plugin
        assert PluginRegistry().has_plugin("convenience_test")

    def test_get_plugin_function(self):
        """Test get_plugin convenience function."""
        plugin = MockFeaturePlugin(name="get_convenience")
        PluginRegistry().register(plugin)

        retrieved = get_plugin("get_convenience")

        assert retrieved is plugin

    def test_get_plugin_function_not_found(self):
        """Test get_plugin raises for non-existent plugin."""
        with pytest.raises(PluginNotFoundError):
            get_plugin("nonexistent_convenience")


class TestPluginCount:
    """Tests for plugin counting."""

    def test_get_plugin_count_zero(self):
        """Test get_plugin_count returns 0 when empty."""
        registry = PluginRegistry()

        assert registry.get_plugin_count() == 0

    def test_get_plugin_count_increments(self):
        """Test get_plugin_count increments with registrations."""
        registry = PluginRegistry()

        registry.register(MockFeaturePlugin(name="count_a"))
        assert registry.get_plugin_count() == 1

        registry.register(MockFeaturePlugin(name="count_b"))
        assert registry.get_plugin_count() == 2

    def test_get_plugin_count_decrements(self):
        """Test get_plugin_count decrements with unregistrations."""
        registry = PluginRegistry()
        registry.register(MockFeaturePlugin(name="count_c"))
        registry.register(MockFeaturePlugin(name="count_d"))

        registry.unregister("count_c")

        assert registry.get_plugin_count() == 1
