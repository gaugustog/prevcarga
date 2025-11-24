"""Pydantic configuration models for feature plugins.

This module provides the base configuration model for feature plugins,
supporting common fields like enabled/disabled state and priority,
while allowing plugin-specific additional configuration.

Example:
    ```python
    from src.features.base import PluginConfig

    # Basic usage
    config = PluginConfig(enabled=True, priority=50)

    # With extra fields for plugin-specific config
    config = PluginConfig(
        enabled=True,
        priority=100,
        lookback_days=7,
        include_holidays=True
    )
    ```
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Priority thresholds for logging
PRIORITY_HIGH_THRESHOLD = 10
PRIORITY_LOW_THRESHOLD = 900


class PluginConfig(BaseModel):
    """Base configuration model for feature plugins.

    This model provides common configuration fields that all plugins share,
    while allowing extra fields for plugin-specific configuration through
    the extra="allow" setting.

    Attributes:
        enabled: Whether the plugin is enabled for feature generation.
        priority: Execution priority (lower values run first). Default is 100.
            Valid range is 0-1000.

    Example:
        ```python
        # Create config with defaults
        config = PluginConfig()
        assert config.enabled is True
        assert config.priority == 100

        # Create config with plugin-specific options
        config = PluginConfig(
            enabled=True,
            priority=50,
            custom_option="value",
            lookback_days=14
        )
        print(config.model_extra)  # {'custom_option': 'value', 'lookback_days': 14}
        ```
    """

    model_config = ConfigDict(
        extra="allow",
        validate_default=True,
        str_strip_whitespace=True,
    )

    enabled: bool = Field(
        default=True,
        description="Whether the plugin is enabled for feature generation",
    )
    priority: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Execution priority (lower values run first)",
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: int) -> int:
        """Validate and log warnings for extreme priority values.

        Args:
            v: Priority value to validate.

        Returns:
            The validated priority value.
        """
        if v < PRIORITY_HIGH_THRESHOLD:
            logger.debug("Very high priority plugin (priority=%d)", v)
        elif v > PRIORITY_LOW_THRESHOLD:
            logger.debug("Very low priority plugin (priority=%d)", v)
        return v

    def get_extra_field(self, field_name: str, default: Any = None) -> Any:
        """Get a plugin-specific extra field value.

        Args:
            field_name: Name of the extra field to retrieve.
            default: Default value if field doesn't exist.

        Returns:
            The field value or the default.
        """
        if self.model_extra is not None:
            return self.model_extra.get(field_name, default)
        return default

    def has_extra_field(self, field_name: str) -> bool:
        """Check if a plugin-specific extra field exists.

        Args:
            field_name: Name of the extra field to check.

        Returns:
            True if the field exists, False otherwise.
        """
        if self.model_extra is not None:
            return field_name in self.model_extra
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary including extra fields.

        Returns:
            Dictionary with all configuration fields including extras.
        """
        result = {
            "enabled": self.enabled,
            "priority": self.priority,
        }
        if self.model_extra:
            result.update(self.model_extra)
        return result
