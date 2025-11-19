# PC-012-02A: Plugin Architecture Foundation

**Ticket ID:** PC-012-02A  
**Epic:** [Epic-02A: Core Feature Engineering](../epics/Epic-02A.md)  
**User Story:** US-1  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Establish the foundational plugin architecture for feature engineering, including the abstract base class, plugin registry, versioning system, and configuration validation to enable flexible and extensible feature development.

**As a** data scientist  
**I want** a flexible plugin system for feature engineering  
**So that** I can easily add new feature transformations without modifying core code

---

## ✅ Acceptance Criteria

- [ ] `BaseFeaturePlugin` abstract class defines plugin interface
- [ ] `PluginRegistry` manages plugin registration and discovery
- [ ] Plugin versioning system supports semantic versioning
- [ ] Plugin configuration validation prevents runtime errors
- [ ] Documentation explains how to create new plugins
- [ ] Unit tests cover plugin registration and validation

---

## 🔧 Implementation Tasks

### 1. Create Feature Engineering Directory Structure
- [ ] Create `src/features/` directory
- [ ] Create `src/features/base/` subdirectory
- [ ] Create `src/features/plugins/` subdirectory
- [ ] Create `tests/features/` directory
- [ ] Add `__init__.py` files to all directories

### 2. Implement BaseFeaturePlugin Abstract Class
- [ ] Create `src/features/base/plugin.py`
- [ ] Define abstract `BaseFeaturePlugin` class with ABC
- [ ] Add abstract `name` property (plugin identifier)
- [ ] Add abstract `version` property (semantic versioning)
- [ ] Add abstract `generate_features()` method
- [ ] Add abstract `get_feature_names()` method
- [ ] Add `validate_config()` method with default implementation
- [ ] Add `get_metadata()` method for plugin description
- [ ] Add type hints for all methods
- [ ] Add comprehensive docstrings with examples

### 3. Implement PluginRegistry
- [ ] Create `src/features/registry.py`
- [ ] Implement `PluginRegistry` class with singleton pattern
- [ ] Add `register()` method for plugin registration
- [ ] Add `get_plugin()` method for plugin retrieval by name
- [ ] Add `list_plugins()` method returning all registered plugins
- [ ] Add `unregister()` method for plugin removal
- [ ] Implement version conflict detection
- [ ] Add plugin name validation (no duplicates)
- [ ] Support plugin discovery via entry points

### 4. Create Configuration Schema
- [ ] Create `src/features/base/config.py`
- [ ] Define `PluginConfig` Pydantic base model
- [ ] Add common configuration fields (enabled, priority)
- [ ] Implement configuration inheritance for plugin-specific configs
- [ ] Add validation for required vs optional parameters
- [ ] Create configuration examples in docstrings

### 5. Implement Plugin Validator
- [ ] Create `src/features/base/validator.py`
- [ ] Define `FeatureValidator` class
- [ ] Implement schema validation for plugin outputs
- [ ] Add feature name validation (no duplicates, valid characters)
- [ ] Add feature value validation (NaN detection, type checking)
- [ ] Create validation report with warnings and errors
- [ ] Add configuration validation helper methods

### 6. Create Feature Utilities
- [ ] Create `src/features/utils.py`
- [ ] Add `validate_dataframe_index()` for datetime index checks
- [ ] Add `check_feature_names()` for name collision detection
- [ ] Add `generate_feature_metadata()` for tracking feature lineage
- [ ] Add `merge_feature_dataframes()` for combining plugin outputs
- [ ] Add logging utilities for feature generation tracking

### 7. Write Comprehensive Tests
- [ ] Create `tests/features/test_base_plugin.py`
- [ ] Test abstract class cannot be instantiated
- [ ] Create mock plugin implementations for testing
- [ ] Test plugin registration and retrieval
- [ ] Test version conflict detection
- [ ] Test configuration validation
- [ ] Test plugin discovery mechanisms
- [ ] Create fixtures for common test scenarios

### 8. Create Developer Documentation
- [ ] Create `docs/developer/plugin-development-guide.md`
- [ ] Document plugin creation process step-by-step
- [ ] Include example plugin implementation
- [ ] Document configuration schema patterns
- [ ] Add troubleshooting section
- [ ] Include best practices for plugin development

---

## 💻 Implementation Details

### BaseFeaturePlugin Abstract Class

```python
"""Base abstract class for feature engineering plugins."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseFeaturePlugin(ABC):
    """
    Abstract base class for all feature engineering plugins.
    
    All feature plugins must inherit from this class and implement
    the required abstract methods.
    
    Example:
        >>> class MyFeaturePlugin(BaseFeaturePlugin):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_feature"
        ...     
        ...     @property
        ...     def version(self) -> str:
        ...         return "1.0.0"
        ...     
        ...     def generate_features(self, df, config):
        ...         # Feature generation logic
        ...         return pd.DataFrame({"my_feature": [1, 2, 3]})
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Plugin name for registration and identification.
        
        Returns:
            Unique plugin identifier (lowercase, underscores allowed)
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """
        Plugin version following semantic versioning (MAJOR.MINOR.PATCH).
        
        Returns:
            Version string (e.g., "1.0.0")
        """
        pass
    
    @abstractmethod
    def generate_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate features from input DataFrame.
        
        Args:
            df: Input DataFrame with datetime index
            config: Plugin-specific configuration dictionary
        
        Returns:
            DataFrame with generated features (same index as input)
        
        Raises:
            ValueError: If input DataFrame is invalid
            KeyError: If required config keys are missing
        """
        pass
    
    @abstractmethod
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """
        Get list of feature names that will be generated.
        
        Args:
            config: Plugin-specific configuration dictionary
        
        Returns:
            List of feature names
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate plugin configuration.
        
        Override this method to add custom validation logic.
        
        Args:
            config: Plugin configuration to validate
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(config)}")
        
        logger.debug(f"Config validation passed for plugin '{self.name}'")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.
        
        Returns:
            Dictionary with plugin metadata (name, version, description)
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.__doc__ or "No description available"
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
```

### PluginRegistry Implementation

```python
"""Plugin registry for feature engineering plugins."""
from typing import Dict, List, Optional
import re

from src.features.base.plugin import BaseFeaturePlugin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PluginRegistry:
    """
    Central registry for feature engineering plugins.
    
    Implements singleton pattern to ensure single source of truth
    for plugin management.
    
    Example:
        >>> registry = PluginRegistry()
        >>> registry.register(MyFeaturePlugin())
        >>> plugin = registry.get_plugin("my_feature")
        >>> plugins = registry.list_plugins()
    """
    
    _instance = None
    _plugins: Dict[str, BaseFeaturePlugin] = {}
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = {}
        return cls._instance
    
    def register(self, plugin: BaseFeaturePlugin) -> None:
        """
        Register a feature plugin.
        
        Args:
            plugin: Plugin instance to register
        
        Raises:
            ValueError: If plugin name is invalid or already registered
            TypeError: If plugin doesn't inherit from BaseFeaturePlugin
        """
        # Type check
        if not isinstance(plugin, BaseFeaturePlugin):
            raise TypeError(
                f"Plugin must inherit from BaseFeaturePlugin, "
                f"got {type(plugin)}"
            )
        
        # Name validation
        if not self._is_valid_plugin_name(plugin.name):
            raise ValueError(
                f"Invalid plugin name: '{plugin.name}'. "
                f"Must be lowercase with underscores only."
            )
        
        # Version validation
        if not self._is_valid_semver(plugin.version):
            raise ValueError(
                f"Invalid version: '{plugin.version}'. "
                f"Must follow semantic versioning (e.g., '1.0.0')"
            )
        
        # Check for duplicates
        if plugin.name in self._plugins:
            existing = self._plugins[plugin.name]
            logger.warning(
                f"Plugin '{plugin.name}' already registered "
                f"(version {existing.version}). Overwriting with version {plugin.version}."
            )
        
        self._plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name} (v{plugin.version})")
    
    def get_plugin(self, name: str) -> BaseFeaturePlugin:
        """
        Retrieve plugin by name.
        
        Args:
            name: Plugin name
        
        Returns:
            Plugin instance
        
        Raises:
            KeyError: If plugin not found
        """
        if name not in self._plugins:
            raise KeyError(
                f"Plugin '{name}' not found. "
                f"Available plugins: {list(self._plugins.keys())}"
            )
        
        return self._plugins[name]
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all registered plugins with metadata.
        
        Returns:
            List of plugin metadata dictionaries
        """
        return [plugin.get_metadata() for plugin in self._plugins.values()]
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name to unregister
        
        Returns:
            True if plugin was removed, False if not found
        """
        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered plugins (mainly for testing)."""
        self._plugins.clear()
        logger.debug("Cleared all registered plugins")
    
    @staticmethod
    def _is_valid_plugin_name(name: str) -> bool:
        """Validate plugin name (lowercase, underscores only)."""
        return bool(re.match(r'^[a-z][a-z0-9_]*$', name))
    
    @staticmethod
    def _is_valid_semver(version: str) -> bool:
        """Validate semantic version string."""
        pattern = r'^\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))
```

### Configuration Base Schema

```python
"""Configuration schemas for feature plugins."""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class PluginConfig(BaseModel):
    """
    Base configuration for feature plugins.
    
    All plugin-specific configs should inherit from this.
    """
    
    enabled: bool = Field(default=True, description="Whether plugin is enabled")
    priority: int = Field(default=100, description="Execution priority (lower = earlier)")
    
    class Config:
        extra = "allow"  # Allow plugin-specific fields
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for plugin architecture."""
import pytest
from src.features.base.plugin import BaseFeaturePlugin
from src.features.registry import PluginRegistry


class MockFeaturePlugin(BaseFeaturePlugin):
    """Mock plugin for testing."""
    
    @property
    def name(self) -> str:
        return "mock_feature"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def generate_features(self, df, config):
        return pd.DataFrame({"mock_feature": [1, 2, 3]}, index=df.index)
    
    def get_feature_names(self, config):
        return ["mock_feature"]


def test_base_plugin_cannot_instantiate():
    """Test that abstract base class cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseFeaturePlugin()


def test_plugin_registration():
    """Test plugin registration in registry."""
    registry = PluginRegistry()
    registry.clear()
    
    plugin = MockFeaturePlugin()
    registry.register(plugin)
    
    assert "mock_feature" in [p["name"] for p in registry.list_plugins()]


def test_plugin_retrieval():
    """Test retrieving registered plugin."""
    registry = PluginRegistry()
    registry.clear()
    
    plugin = MockFeaturePlugin()
    registry.register(plugin)
    
    retrieved = registry.get_plugin("mock_feature")
    assert retrieved.name == "mock_feature"
    assert retrieved.version == "1.0.0"


def test_invalid_plugin_name():
    """Test that invalid plugin names are rejected."""
    registry = PluginRegistry()
    
    class InvalidPlugin(BaseFeaturePlugin):
        @property
        def name(self):
            return "Invalid-Name"  # Uppercase and hyphen not allowed
        
        @property
        def version(self):
            return "1.0.0"
        
        def generate_features(self, df, config):
            return pd.DataFrame()
        
        def get_feature_names(self, config):
            return []
    
    with pytest.raises(ValueError, match="Invalid plugin name"):
        registry.register(InvalidPlugin())


def test_invalid_version_format():
    """Test that invalid versions are rejected."""
    registry = PluginRegistry()
    
    class InvalidVersionPlugin(BaseFeaturePlugin):
        @property
        def name(self):
            return "test_plugin"
        
        @property
        def version(self):
            return "v1.0"  # Invalid format
        
        def generate_features(self, df, config):
            return pd.DataFrame()
        
        def get_feature_names(self, config):
            return []
    
    with pytest.raises(ValueError, match="Invalid version"):
        registry.register(InvalidVersionPlugin())
```

---

## 📝 Technical Notes

- Use ABC (Abstract Base Class) for enforcing plugin interface
- Singleton pattern ensures single registry across application
- Semantic versioning enables version tracking and compatibility checks
- Configuration validation happens before feature generation
- Plugin metadata enables introspection and documentation generation

---

## 🔗 Dependencies

**Depends On:**
- PC-001-00: Repository Structure Setup
- PC-002-00: Python Environment with uv
- PC-004-00: Structured Logging Framework

**Blocks:**
- PC-013-02A: Temporal Features Plugin
- PC-014-02A: Calendar/Holiday Features Plugin
- PC-015-02A: Lag Features Plugin
- PC-016-02A: Cyclical Encoding Plugin
- PC-017-02A: Feature Pipeline Composer

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] BaseFeaturePlugin abstract class implemented
- [ ] PluginRegistry with registration/discovery
- [ ] Configuration validation working
- [ ] Unit tests pass with >90% coverage
- [ ] Mock plugin created for testing
- [ ] Developer documentation complete
- [ ] Code reviewed and approved
- [ ] Ready for plugin implementations (PC-013 to PC-016)

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next Ticket:** [PC-013-02A: Temporal Features Plugin](PC-013-02A-temporal-features-plugin.md)
