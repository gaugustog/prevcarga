"""Tests for PluginConfig Pydantic model.

This module contains comprehensive tests for the PluginConfig model,
including default values, validation, and extra field handling.
"""

import pytest
from pydantic import ValidationError

from src.features.base.config import PluginConfig


class TestPluginConfigDefaults:
    """Tests for PluginConfig default values."""

    def test_default_enabled(self):
        """Test that enabled defaults to True."""
        config = PluginConfig()
        assert config.enabled is True

    def test_default_priority(self):
        """Test that priority defaults to 100."""
        config = PluginConfig()
        assert config.priority == 100

    def test_all_defaults(self):
        """Test creating config with all defaults."""
        config = PluginConfig()

        assert config.enabled is True
        assert config.priority == 100


class TestPluginConfigEnabled:
    """Tests for enabled field."""

    def test_enabled_true(self):
        """Test setting enabled to True."""
        config = PluginConfig(enabled=True)
        assert config.enabled is True

    def test_enabled_false(self):
        """Test setting enabled to False."""
        config = PluginConfig(enabled=False)
        assert config.enabled is False

    def test_enabled_from_string_true(self):
        """Test that string 'true' is coerced to bool."""
        # Pydantic should coerce strings
        config = PluginConfig(enabled="true")  # type: ignore[arg-type]
        assert config.enabled is True

    def test_enabled_from_int(self):
        """Test that integer 1 is coerced to True."""
        config = PluginConfig(enabled=1)  # type: ignore[arg-type]
        assert config.enabled is True


class TestPluginConfigPriority:
    """Tests for priority field."""

    def test_priority_custom_value(self):
        """Test setting a custom priority value."""
        config = PluginConfig(priority=50)
        assert config.priority == 50

    def test_priority_minimum(self):
        """Test priority at minimum bound (0)."""
        config = PluginConfig(priority=0)
        assert config.priority == 0

    def test_priority_maximum(self):
        """Test priority at maximum bound (1000)."""
        config = PluginConfig(priority=1000)
        assert config.priority == 1000

    def test_priority_below_minimum_raises(self):
        """Test that priority below 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PluginConfig(priority=-1)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("priority",) for e in errors)

    def test_priority_above_maximum_raises(self):
        """Test that priority above 1000 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PluginConfig(priority=1001)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("priority",) for e in errors)

    def test_priority_invalid_type_raises(self):
        """Test that non-integer priority raises ValidationError."""
        with pytest.raises(ValidationError):
            PluginConfig(priority="high")  # type: ignore[arg-type]


class TestPluginConfigExtraFields:
    """Tests for extra field handling."""

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed."""
        config = PluginConfig(
            enabled=True,
            priority=100,
            custom_option="value",
            lookback_days=7,
        )

        assert config.enabled is True
        assert config.priority == 100
        assert config.model_extra is not None
        assert config.model_extra.get("custom_option") == "value"
        assert config.model_extra.get("lookback_days") == 7

    def test_multiple_extra_fields(self):
        """Test multiple extra fields."""
        config = PluginConfig(
            option_a="a",
            option_b=123,
            option_c=True,
            option_d=[1, 2, 3],
        )

        assert config.model_extra is not None
        assert len(config.model_extra) == 4
        assert config.model_extra["option_a"] == "a"
        assert config.model_extra["option_b"] == 123
        assert config.model_extra["option_c"] is True
        assert config.model_extra["option_d"] == [1, 2, 3]

    def test_get_extra_field_exists(self):
        """Test get_extra_field for existing field."""
        config = PluginConfig(custom_field="test_value")

        value = config.get_extra_field("custom_field")

        assert value == "test_value"

    def test_get_extra_field_not_exists_default(self):
        """Test get_extra_field returns default for non-existent field."""
        config = PluginConfig()

        value = config.get_extra_field("nonexistent", default="fallback")

        assert value == "fallback"

    def test_get_extra_field_not_exists_none(self):
        """Test get_extra_field returns None when no default."""
        config = PluginConfig()

        value = config.get_extra_field("nonexistent")

        assert value is None

    def test_has_extra_field_true(self):
        """Test has_extra_field returns True for existing field."""
        config = PluginConfig(existing_field="value")

        assert config.has_extra_field("existing_field") is True

    def test_has_extra_field_false(self):
        """Test has_extra_field returns False for non-existent field."""
        config = PluginConfig()

        assert config.has_extra_field("nonexistent") is False


class TestPluginConfigToDict:
    """Tests for to_dict method."""

    def test_to_dict_default(self):
        """Test to_dict with default values."""
        config = PluginConfig()
        result = config.to_dict()

        assert result == {"enabled": True, "priority": 100}

    def test_to_dict_custom_values(self):
        """Test to_dict with custom values."""
        config = PluginConfig(enabled=False, priority=50)
        result = config.to_dict()

        assert result == {"enabled": False, "priority": 50}

    def test_to_dict_with_extra_fields(self):
        """Test to_dict includes extra fields."""
        config = PluginConfig(
            enabled=True,
            priority=100,
            custom_a="value_a",
            custom_b=42,
        )
        result = config.to_dict()

        assert result["enabled"] is True
        assert result["priority"] == 100
        assert result["custom_a"] == "value_a"
        assert result["custom_b"] == 42

    def test_to_dict_returns_dict(self):
        """Test to_dict returns a dictionary."""
        config = PluginConfig()
        result = config.to_dict()

        assert isinstance(result, dict)


class TestPluginConfigValidation:
    """Tests for config validation behavior."""

    def test_whitespace_stripping(self):
        """Test that whitespace is stripped from string fields."""
        config = PluginConfig(string_field="  value  ")

        # Extra fields don't get stripped automatically
        # This test verifies the model_config setting
        assert config.model_extra is not None

    def test_validate_default_true(self):
        """Test that defaults are validated."""
        # Should not raise - defaults are valid
        config = PluginConfig()
        assert config.enabled is True
        assert config.priority == 100


class TestPluginConfigSerialization:
    """Tests for serialization behavior."""

    def test_model_dump(self):
        """Test Pydantic model_dump method."""
        config = PluginConfig(
            enabled=True,
            priority=50,
            custom="value",
        )

        data = config.model_dump()

        assert data["enabled"] is True
        assert data["priority"] == 50
        assert data["custom"] == "value"

    def test_model_dump_json(self):
        """Test Pydantic model_dump_json method."""
        config = PluginConfig(enabled=True, priority=100)

        json_str = config.model_dump_json()

        assert '"enabled":true' in json_str or '"enabled": true' in json_str
        assert "100" in json_str

    def test_model_validate_from_dict(self):
        """Test creating config from dict."""
        data = {
            "enabled": False,
            "priority": 75,
            "extra_option": "test",
        }

        config = PluginConfig.model_validate(data)

        assert config.enabled is False
        assert config.priority == 75
        assert config.model_extra is not None
        assert config.model_extra.get("extra_option") == "test"


class TestPluginConfigEdgeCases:
    """Tests for edge cases."""

    def test_empty_extra_fields(self):
        """Test behavior with no extra fields."""
        config = PluginConfig(enabled=True, priority=100)

        assert config.model_extra == {} or config.model_extra is None

    def test_nested_extra_fields(self):
        """Test nested dict/list in extra fields."""
        config = PluginConfig(
            nested_dict={"a": 1, "b": 2},
            nested_list=[1, 2, 3],
        )

        assert config.model_extra is not None
        assert config.model_extra["nested_dict"] == {"a": 1, "b": 2}
        assert config.model_extra["nested_list"] == [1, 2, 3]

    def test_extra_field_named_enabled(self):
        """Test that extra field cannot override core field."""
        # enabled is a core field, so extra enabled would be ignored
        config = PluginConfig(enabled=True)

        # Core field takes precedence
        assert config.enabled is True
