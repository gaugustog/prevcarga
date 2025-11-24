"""Tests for BaseFeaturePlugin abstract class.

This module contains comprehensive tests for the BaseFeaturePlugin abstract class,
including testing that it cannot be instantiated directly and that concrete
implementations work correctly.
"""

from typing import Any

import pandas as pd
import pytest

from src.features.base.plugin import BaseFeaturePlugin


class TestBaseFeaturePluginAbstract:
    """Tests for abstract nature of BaseFeaturePlugin."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseFeaturePlugin cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseFeaturePlugin()  # type: ignore[abstract]

        # Error message should mention abstract methods
        assert "abstract" in str(exc_info.value).lower()

    def test_abstract_methods_listed(self):
        """Test that all expected abstract methods are defined."""
        # Get abstract methods from the class
        abstract_methods = set()
        for name in dir(BaseFeaturePlugin):
            method = getattr(BaseFeaturePlugin, name)
            if getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)

        # Should have these abstract methods/properties
        expected = {"name", "version", "generate_features", "get_feature_names"}
        assert expected.issubset(abstract_methods)


class MockFeaturePlugin(BaseFeaturePlugin):
    """Mock feature plugin for testing."""

    def __init__(
        self,
        name: str = "mock_plugin",
        version: str = "1.0.0",
        features: list[str] | None = None,
    ) -> None:
        """Initialize mock plugin.

        Args:
            name: Plugin name.
            version: Plugin version.
            features: List of feature names to generate.
        """
        self._name = name
        self._version = version
        self._features = features or ["feature_a", "feature_b"]

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
        result = df.copy()
        for feat in self._features:
            result[feat] = 1.0
        return result

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        """Return mock feature names."""
        return self._features.copy()


class TestMockFeaturePlugin:
    """Tests for concrete implementation of BaseFeaturePlugin."""

    @pytest.fixture
    def plugin(self) -> MockFeaturePlugin:
        """Create a mock plugin for testing."""
        return MockFeaturePlugin()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {"value": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

    def test_name_property(self, plugin: MockFeaturePlugin):
        """Test that name property returns correct value."""
        assert plugin.name == "mock_plugin"

    def test_version_property(self, plugin: MockFeaturePlugin):
        """Test that version property returns correct value."""
        assert plugin.version == "1.0.0"

    def test_custom_name_and_version(self):
        """Test creating plugin with custom name and version."""
        plugin = MockFeaturePlugin(name="custom_plugin", version="2.3.4")
        assert plugin.name == "custom_plugin"
        assert plugin.version == "2.3.4"

    def test_generate_features(
        self, plugin: MockFeaturePlugin, sample_df: pd.DataFrame
    ):
        """Test that generate_features adds feature columns."""
        result = plugin.generate_features(sample_df, {})

        assert "feature_a" in result.columns
        assert "feature_b" in result.columns
        assert len(result) == len(sample_df)

    def test_generate_features_preserves_original_columns(
        self, plugin: MockFeaturePlugin, sample_df: pd.DataFrame
    ):
        """Test that generate_features preserves original columns."""
        result = plugin.generate_features(sample_df, {})

        assert "value" in result.columns
        assert result["value"].equals(sample_df["value"])

    def test_get_feature_names(self, plugin: MockFeaturePlugin):
        """Test that get_feature_names returns correct list."""
        names = plugin.get_feature_names({})

        assert names == ["feature_a", "feature_b"]

    def test_get_feature_names_custom(self):
        """Test get_feature_names with custom features."""
        plugin = MockFeaturePlugin(features=["hour", "day_of_week", "month"])
        names = plugin.get_feature_names({})

        assert names == ["hour", "day_of_week", "month"]


class TestBaseFeaturePluginConcreteMethods:
    """Tests for concrete methods in BaseFeaturePlugin."""

    @pytest.fixture
    def plugin(self) -> MockFeaturePlugin:
        """Create a mock plugin for testing."""
        return MockFeaturePlugin()

    def test_validate_config_accepts_valid_config(self, plugin: MockFeaturePlugin):
        """Test that validate_config accepts valid configuration."""
        result = plugin.validate_config({"option": "value"})
        assert result is True

    def test_validate_config_accepts_empty_dict(self, plugin: MockFeaturePlugin):
        """Test that validate_config accepts empty dictionary."""
        result = plugin.validate_config({})
        assert result is True

    def test_validate_config_rejects_none(self, plugin: MockFeaturePlugin):
        """Test that validate_config rejects None configuration."""
        with pytest.raises(ValueError) as exc_info:
            plugin.validate_config(None)  # type: ignore[arg-type]

        assert "cannot be None" in str(exc_info.value)

    def test_get_metadata_returns_dict(self, plugin: MockFeaturePlugin):
        """Test that get_metadata returns a dictionary."""
        metadata = plugin.get_metadata()

        assert isinstance(metadata, dict)

    def test_get_metadata_contains_required_keys(self, plugin: MockFeaturePlugin):
        """Test that get_metadata contains all required keys."""
        metadata = plugin.get_metadata()

        assert "name" in metadata
        assert "version" in metadata
        assert "class" in metadata
        assert "module" in metadata

    def test_get_metadata_correct_values(self, plugin: MockFeaturePlugin):
        """Test that get_metadata returns correct values."""
        metadata = plugin.get_metadata()

        assert metadata["name"] == "mock_plugin"
        assert metadata["version"] == "1.0.0"
        assert metadata["class"] == "MockFeaturePlugin"
        assert "test_plugin" in metadata["module"]

    def test_repr(self, plugin: MockFeaturePlugin):
        """Test __repr__ returns expected format."""
        repr_str = repr(plugin)

        assert "MockFeaturePlugin" in repr_str
        assert "name='mock_plugin'" in repr_str
        assert "version='1.0.0'" in repr_str

    def test_repr_with_custom_values(self):
        """Test __repr__ with custom name and version."""
        plugin = MockFeaturePlugin(name="custom", version="3.2.1")
        repr_str = repr(plugin)

        assert "name='custom'" in repr_str
        assert "version='3.2.1'" in repr_str


class TestBaseFeaturePluginInheritance:
    """Tests for proper inheritance behavior."""

    def test_incomplete_implementation_raises(self):
        """Test that incomplete implementation raises TypeError."""

        class IncompletePlugin(BaseFeaturePlugin):
            @property
            def name(self) -> str:
                return "incomplete"

            # Missing version, generate_features, get_feature_names

        with pytest.raises(TypeError):
            IncompletePlugin()  # type: ignore[abstract]

    def test_partial_implementation_raises(self):
        """Test that partial implementation raises TypeError."""

        class PartialPlugin(BaseFeaturePlugin):
            @property
            def name(self) -> str:
                return "partial"

            @property
            def version(self) -> str:
                return "1.0.0"

            # Missing generate_features, get_feature_names

        with pytest.raises(TypeError):
            PartialPlugin()  # type: ignore[abstract]

    def test_complete_implementation_works(self):
        """Test that complete implementation can be instantiated."""
        # MockFeaturePlugin is complete, so this should work
        plugin = MockFeaturePlugin()
        assert plugin.name == "mock_plugin"
        assert plugin.version == "1.0.0"
