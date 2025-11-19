# PC-017-02A: Feature Pipeline Composer

**Ticket ID:** PC-017-02A  
**Epic:** [Epic-02A: Core Feature Engineering](../epics/Epic-02A.md)  
**User Story:** US-6  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a feature pipeline composer that orchestrates multiple feature engineering plugins, manages dependencies, handles feature conflicts, validates outputs, and provides a unified interface for feature generation with configuration management.

**As a** data scientist  
**I want** an orchestrated feature pipeline  
**So that** I can combine multiple feature plugins with proper dependency management

---

## ✅ Acceptance Criteria

- [ ] `FeaturePipeline` class orchestrates multiple plugins
- [ ] Plugin execution respects priority order
- [ ] Feature name collision detection and resolution
- [ ] Feature validation (NaN checks, type validation)
- [ ] Configuration file support (YAML)
- [ ] Pipeline state persistence and reloading
- [ ] Feature metadata tracking (lineage, timing)
- [ ] Parallel plugin execution when possible
- [ ] Comprehensive error handling and logging
- [ ] Unit tests with multiple plugin combinations
- [ ] Integration tests with real plugins
- [ ] Performance: <2s for full pipeline on 1 year hourly data

---

## 🔧 Implementation Tasks

### 1. Create Pipeline Module
- [ ] Create `src/features/pipeline.py`
- [ ] Import PluginRegistry and BaseFeaturePlugin
- [ ] Add module docstring with pipeline overview

### 2. Implement FeaturePipeline Class
- [ ] Create `FeaturePipeline` class
- [ ] Add `__init__()` with registry integration
- [ ] Add `add_plugin()` method for plugin registration
- [ ] Add `remove_plugin()` method
- [ ] Add `get_plugins()` method for listing
- [ ] Add internal plugin storage with priority
- [ ] Add configuration storage

### 3. Implement Configuration Schema
- [ ] Create `PipelineConfig` Pydantic model
- [ ] Add `plugins` list with plugin configurations
- [ ] Each plugin config includes: name, enabled, priority, config
- [ ] Add `validation` settings (nan_threshold, type_checking)
- [ ] Add `parallelization` settings (enabled, max_workers)
- [ ] Add `logging` settings (level, output_file)
- [ ] Validate plugin names exist in registry

### 4. Implement Plugin Ordering
- [ ] Create `_sort_plugins_by_priority()` method
- [ ] Sort plugins by priority (lower = earlier)
- [ ] Handle ties (same priority) with stable sort
- [ ] Log execution order
- [ ] Return sorted list of plugins

### 5. Implement Feature Generation Pipeline
- [ ] Create `generate_features()` main method
- [ ] Accept input DataFrame
- [ ] Execute plugins in priority order
- [ ] Collect features from each plugin
- [ ] Merge features into single DataFrame
- [ ] Handle feature name collisions
- [ ] Track execution timing per plugin
- [ ] Log progress and completion
- [ ] Return combined feature DataFrame

### 6. Implement Feature Name Collision Detection
- [ ] Create `_check_feature_collisions()` method
- [ ] Track all generated feature names
- [ ] Detect duplicate feature names
- [ ] Strategies: error, warn, suffix, keep_first, keep_last
- [ ] Log collision warnings
- [ ] Return collision report

### 7. Implement Feature Validation
- [ ] Create `FeatureValidator` class
- [ ] Implement `validate_features()` method
- [ ] Check for excessive NaNs (configurable threshold)
- [ ] Check for infinite values
- [ ] Check for constant columns (zero variance)
- [ ] Validate data types (numeric, categorical)
- [ ] Generate validation report
- [ ] Raise warnings or errors based on severity

### 8. Implement Metadata Tracking
- [ ] Create `FeatureMetadata` dataclass
- [ ] Track plugin name for each feature
- [ ] Track generation timestamp
- [ ] Track execution time
- [ ] Track NaN count/percentage
- [ ] Track feature statistics (min, max, mean, std)
- [ ] Create `get_metadata()` method returning metadata

### 9. Implement Configuration File Support
- [ ] Create `load_config()` class method
- [ ] Support YAML configuration files
- [ ] Parse plugin configurations
- [ ] Instantiate plugins from config
- [ ] Validate configuration schema
- [ ] Create `save_config()` method for exporting config
- [ ] Support environment variable substitution

### 10. Implement Pipeline Persistence
- [ ] Create `save_pipeline()` method
- [ ] Serialize pipeline state to disk
- [ ] Save configuration and plugin list
- [ ] Create `load_pipeline()` class method
- [ ] Restore pipeline from saved state
- [ ] Validate loaded plugins against registry

### 11. Implement Parallel Execution (Optional)
- [ ] Create `_execute_parallel()` method
- [ ] Identify plugins with no dependencies
- [ ] Use ThreadPoolExecutor or multiprocessing
- [ ] Collect results from parallel execution
- [ ] Handle exceptions in parallel tasks
- [ ] Fall back to sequential on error

### 12. Write Comprehensive Tests
- [ ] Create `tests/features/test_pipeline.py`
- [ ] Test single plugin execution
- [ ] Test multiple plugins with priority ordering
- [ ] Test feature collision detection
- [ ] Test feature validation
- [ ] Test configuration loading from YAML
- [ ] Test pipeline persistence and loading
- [ ] Test error handling (missing plugin, invalid config)
- [ ] Integration test with real plugins
- [ ] Test performance with full pipeline

### 13. Create Usage Examples
- [ ] Create `examples/feature_pipeline_demo.py`
- [ ] Show single plugin usage
- [ ] Show multi-plugin pipeline
- [ ] Show configuration file usage
- [ ] Show pipeline persistence
- [ ] Include visualization of generated features

### 14. Create Pipeline Documentation
- [ ] Document pipeline architecture
- [ ] Document plugin execution order
- [ ] Document configuration format
- [ ] Document feature validation rules
- [ ] Include troubleshooting guide
- [ ] Add best practices section

---

## 💻 Implementation Details

### Configuration Schema

```python
"""Configuration schemas for feature pipeline."""
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class PluginConfigItem(BaseModel):
    """Configuration for a single plugin in pipeline."""
    
    name: str = Field(description="Plugin name (must be registered)")
    enabled: bool = Field(default=True, description="Whether plugin is enabled")
    priority: int = Field(default=100, description="Execution priority (lower = earlier)")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration"
    )


class PipelineConfig(BaseModel):
    """Configuration for feature pipeline."""
    
    plugins: List[PluginConfigItem] = Field(
        default_factory=list,
        description="List of plugin configurations"
    )
    validation: Dict[str, Any] = Field(
        default_factory=lambda: {
            "nan_threshold": 0.5,
            "check_infinite": True,
            "check_constant": True
        },
        description="Feature validation settings"
    )
    collision_strategy: Literal["error", "warn", "suffix", "keep_first", "keep_last"] = Field(
        default="error",
        description="Strategy for handling feature name collisions"
    )
    parallelization: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "max_workers": 4
        },
        description="Parallel execution settings"
    )
```

### FeaturePipeline Implementation

```python
"""Feature pipeline for orchestrating multiple plugins."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import yaml
from pathlib import Path

from src.features.base.plugin import BaseFeaturePlugin
from src.features.registry import PluginRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for generated features."""
    
    feature_name: str
    plugin_name: str
    timestamp: datetime
    execution_time_ms: float
    nan_count: int
    nan_percentage: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None


class FeatureValidator:
    """Validator for generated features."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with configuration.
        
        Args:
            config: Validation configuration
        """
        self.nan_threshold = config.get("nan_threshold", 0.5)
        self.check_infinite = config.get("check_infinite", True)
        self.check_constant = config.get("check_constant", True)
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate feature DataFrame.
        
        Args:
            df: DataFrame with features to validate
        
        Returns:
            Dictionary with validation warnings and errors
        """
        warnings = []
        errors = []
        
        for col in df.columns:
            # Check NaN percentage
            nan_pct = df[col].isna().sum() / len(df)
            if nan_pct > self.nan_threshold:
                warnings.append(
                    f"Feature '{col}' has {nan_pct:.1%} NaN values "
                    f"(threshold: {self.nan_threshold:.1%})"
                )
            
            # Check for infinite values
            if self.check_infinite and pd.api.types.is_numeric_dtype(df[col]):
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    errors.append(
                        f"Feature '{col}' contains {inf_count} infinite values"
                    )
            
            # Check for constant columns
            if self.check_constant and pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() == 1:
                    warnings.append(f"Feature '{col}' is constant (zero variance)")
        
        return {"warnings": warnings, "errors": errors}


class FeaturePipeline:
    """
    Feature engineering pipeline orchestrator.
    
    Manages multiple feature plugins, execution order, validation,
    and feature metadata tracking.
    
    Example:
        >>> pipeline = FeaturePipeline()
        >>> pipeline.add_plugin("temporal_features", priority=10, config={...})
        >>> pipeline.add_plugin("lag_features", priority=20, config={...})
        >>> features = pipeline.generate_features(df)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize feature pipeline.
        
        Args:
            config: Optional pipeline configuration
        """
        self.registry = PluginRegistry()
        self.plugins: List[Dict[str, Any]] = []
        self.metadata: List[FeatureMetadata] = []
        
        if config:
            self._load_from_config(config)
    
    def add_plugin(
        self,
        plugin_name: str,
        priority: int = 100,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ) -> None:
        """
        Add plugin to pipeline.
        
        Args:
            plugin_name: Name of registered plugin
            priority: Execution priority (lower = earlier)
            config: Plugin-specific configuration
            enabled: Whether plugin is enabled
        
        Raises:
            KeyError: If plugin not found in registry
        """
        # Validate plugin exists
        plugin = self.registry.get_plugin(plugin_name)
        
        # Validate plugin config
        if config:
            plugin.validate_config(config)
        
        self.plugins.append({
            "name": plugin_name,
            "priority": priority,
            "config": config or {},
            "enabled": enabled
        })
        
        logger.info(
            f"Added plugin '{plugin_name}' with priority {priority} "
            f"(enabled: {enabled})"
        )
    
    def remove_plugin(self, plugin_name: str) -> bool:
        """
        Remove plugin from pipeline.
        
        Args:
            plugin_name: Name of plugin to remove
        
        Returns:
            True if removed, False if not found
        """
        initial_count = len(self.plugins)
        self.plugins = [p for p in self.plugins if p["name"] != plugin_name]
        
        if len(self.plugins) < initial_count:
            logger.info(f"Removed plugin '{plugin_name}' from pipeline")
            return True
        return False
    
    def get_plugins(self) -> List[Dict[str, Any]]:
        """Get list of plugins in pipeline."""
        return self.plugins.copy()
    
    def generate_features(
        self,
        df: pd.DataFrame,
        validate: bool = True,
        collision_strategy: str = "error"
    ) -> pd.DataFrame:
        """
        Generate features using all enabled plugins.
        
        Args:
            df: Input DataFrame
            validate: Whether to validate generated features
            collision_strategy: How to handle feature name collisions
        
        Returns:
            DataFrame with all generated features
        
        Raises:
            ValueError: If feature collision detected and strategy is "error"
        """
        logger.info(f"Starting feature generation pipeline with {len(self.plugins)} plugins")
        
        # Sort plugins by priority
        sorted_plugins = self._sort_plugins_by_priority()
        
        # Filter enabled plugins
        enabled_plugins = [p for p in sorted_plugins if p["enabled"]]
        logger.info(f"Executing {len(enabled_plugins)} enabled plugins")
        
        # Track all features and metadata
        all_features = []
        feature_names_seen = set()
        self.metadata = []
        
        # Execute each plugin
        for plugin_info in enabled_plugins:
            plugin_name = plugin_info["name"]
            plugin_config = plugin_info["config"]
            
            try:
                # Get plugin from registry
                plugin = self.registry.get_plugin(plugin_name)
                
                # Execute plugin
                start_time = datetime.now()
                features = plugin.generate_features(df, plugin_config)
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                logger.info(
                    f"Plugin '{plugin_name}' generated {len(features.columns)} features "
                    f"in {execution_time:.1f}ms"
                )
                
                # Check for collisions
                collisions = feature_names_seen & set(features.columns)
                if collisions:
                    self._handle_collisions(
                        collisions,
                        plugin_name,
                        collision_strategy
                    )
                
                # Update seen features
                feature_names_seen.update(features.columns)
                
                # Track metadata
                for col in features.columns:
                    nan_count = features[col].isna().sum()
                    nan_pct = nan_count / len(features)
                    
                    metadata = FeatureMetadata(
                        feature_name=col,
                        plugin_name=plugin_name,
                        timestamp=start_time,
                        execution_time_ms=execution_time / len(features.columns),
                        nan_count=nan_count,
                        nan_percentage=nan_pct
                    )
                    
                    # Add statistics for numeric features
                    if pd.api.types.is_numeric_dtype(features[col]):
                        metadata.min_value = float(features[col].min())
                        metadata.max_value = float(features[col].max())
                        metadata.mean_value = float(features[col].mean())
                        metadata.std_value = float(features[col].std())
                    
                    self.metadata.append(metadata)
                
                # Store features
                all_features.append(features)
                
            except Exception as e:
                logger.error(f"Plugin '{plugin_name}' failed: {e}")
                raise
        
        # Combine all features
        if not all_features:
            logger.warning("No features generated")
            return pd.DataFrame(index=df.index)
        
        result = pd.concat(all_features, axis=1)
        
        # Validate if requested
        if validate:
            validation_config = {"nan_threshold": 0.5, "check_infinite": True}
            validator = FeatureValidator(validation_config)
            validation_result = validator.validate_features(result)
            
            for warning in validation_result["warnings"]:
                logger.warning(f"Validation: {warning}")
            
            for error in validation_result["errors"]:
                logger.error(f"Validation: {error}")
        
        logger.info(
            f"Feature generation complete: {len(result.columns)} features, "
            f"{len(result)} records"
        )
        
        return result
    
    def get_metadata(self) -> List[FeatureMetadata]:
        """Get feature generation metadata."""
        return self.metadata.copy()
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "FeaturePipeline":
        """
        Load pipeline from YAML configuration file.
        
        Args:
            config_path: Path to YAML config file
        
        Returns:
            Configured FeaturePipeline instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = PipelineConfig(**config_dict)
        
        pipeline = cls()
        for plugin_config in config.plugins:
            pipeline.add_plugin(
                plugin_name=plugin_config.name,
                priority=plugin_config.priority,
                config=plugin_config.config,
                enabled=plugin_config.enabled
            )
        
        logger.info(f"Loaded pipeline from {config_path}")
        return pipeline
    
    def save_config(self, config_path: str) -> None:
        """
        Save pipeline configuration to YAML file.
        
        Args:
            config_path: Path to save config file
        """
        config_dict = {
            "plugins": self.plugins
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Saved pipeline config to {config_path}")
    
    def _sort_plugins_by_priority(self) -> List[Dict[str, Any]]:
        """Sort plugins by priority (lower = earlier)."""
        return sorted(self.plugins, key=lambda p: p["priority"])
    
    def _handle_collisions(
        self,
        collisions: set,
        plugin_name: str,
        strategy: str
    ) -> None:
        """Handle feature name collisions based on strategy."""
        if strategy == "error":
            raise ValueError(
                f"Plugin '{plugin_name}' generated features with "
                f"duplicate names: {collisions}"
            )
        elif strategy == "warn":
            logger.warning(
                f"Plugin '{plugin_name}' generated features with "
                f"duplicate names: {collisions}. Later values will overwrite."
            )
        # Other strategies (suffix, keep_first, keep_last) would be implemented here
    
    def _load_from_config(self, config: PipelineConfig) -> None:
        """Load pipeline from PipelineConfig object."""
        for plugin_config in config.plugins:
            self.add_plugin(
                plugin_name=plugin_config.name,
                priority=plugin_config.priority,
                config=plugin_config.config,
                enabled=plugin_config.enabled
            )
```

### Example Configuration File

```yaml
# config/feature_pipeline.yaml
plugins:
  - name: temporal_features
    enabled: true
    priority: 10
    config:
      include_cyclical: true
      include_season: true
      timezone: "America/Sao_Paulo"
  
  - name: calendar_features
    enabled: true
    priority: 20
    config:
      include_bridge_days: true
      include_pre_post_indicators: true
      custom_holidays: []
  
  - name: lag_features
    enabled: true
    priority: 30
    config:
      lag_periods: [1, 2, 24, 168]
      rolling_windows: [24, 168]
      rolling_stats: ["mean", "std", "min", "max"]
      target_column: "carga"
      fill_method: "none"

validation:
  nan_threshold: 0.5
  check_infinite: true
  check_constant: true

collision_strategy: "error"
```

---

## 🧪 Testing & Validation

### Integration Test

```python
"""Integration tests for feature pipeline."""
import pytest
import pandas as pd
import numpy as np

from src.features.pipeline import FeaturePipeline
from src.features.registry import PluginRegistry
from src.features.plugins.temporal import TemporalFeaturesPlugin
from src.features.plugins.lag import LagFeaturesPlugin


@pytest.fixture
def sample_df():
    """Create sample DataFrame."""
    dates = pd.date_range("2024-01-01", periods=1000, freq="h", tz="America/Sao_Paulo")
    return pd.DataFrame({
        "carga": np.random.randn(1000) + 1000
    }, index=dates)


def test_single_plugin(sample_df):
    """Test pipeline with single plugin."""
    # Register plugin
    registry = PluginRegistry()
    registry.register(TemporalFeaturesPlugin())
    
    # Create pipeline
    pipeline = FeaturePipeline()
    pipeline.add_plugin("temporal_features", priority=10, config={})
    
    # Generate features
    result = pipeline.generate_features(sample_df)
    
    assert len(result.columns) > 0
    assert len(result) == len(sample_df)


def test_multiple_plugins(sample_df):
    """Test pipeline with multiple plugins."""
    # Register plugins
    registry = PluginRegistry()
    registry.clear()
    registry.register(TemporalFeaturesPlugin())
    registry.register(LagFeaturesPlugin())
    
    # Create pipeline
    pipeline = FeaturePipeline()
    pipeline.add_plugin("temporal_features", priority=10)
    pipeline.add_plugin("lag_features", priority=20, config={
        "lag_periods": [1, 24],
        "rolling_windows": [],
        "target_column": "carga"
    })
    
    # Generate features
    result = pipeline.generate_features(sample_df)
    
    # Should have features from both plugins
    assert "hour" in result.columns
    assert "carga_lag_1" in result.columns


def test_priority_ordering(sample_df):
    """Test plugins execute in priority order."""
    registry = PluginRegistry()
    registry.clear()
    registry.register(TemporalFeaturesPlugin())
    
    pipeline = FeaturePipeline()
    pipeline.add_plugin("temporal_features", priority=50)
    
    # Lower priority should execute first
    sorted_plugins = pipeline._sort_plugins_by_priority()
    assert sorted_plugins[0]["priority"] == 50


def test_config_file_loading(tmp_path):
    """Test loading pipeline from YAML config."""
    config_file = tmp_path / "pipeline_config.yaml"
    config_content = """
plugins:
  - name: temporal_features
    enabled: true
    priority: 10
    config:
      include_cyclical: true
"""
    config_file.write_text(config_content)
    
    # Register plugin
    registry = PluginRegistry()
    registry.clear()
    registry.register(TemporalFeaturesPlugin())
    
    # Load pipeline
    pipeline = FeaturePipeline.from_config_file(str(config_file))
    
    assert len(pipeline.get_plugins()) == 1


def test_metadata_tracking(sample_df):
    """Test feature metadata is tracked."""
    registry = PluginRegistry()
    registry.clear()
    registry.register(TemporalFeaturesPlugin())
    
    pipeline = FeaturePipeline()
    pipeline.add_plugin("temporal_features", priority=10)
    
    result = pipeline.generate_features(sample_df)
    metadata = pipeline.get_metadata()
    
    assert len(metadata) == len(result.columns)
    assert all(m.plugin_name == "temporal_features" for m in metadata)
```

---

## 📝 Technical Notes

- Plugin priority determines execution order (lower = earlier)
- Feature validation helps catch data quality issues early
- Metadata tracking enables feature lineage and debugging
- YAML configuration enables easy experimentation
- Collision detection prevents silent feature overwriting
- Parallel execution optional (may have ordering constraints)

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation
- PC-013-02A: Temporal Features Plugin
- PC-014-02A: Calendar/Holiday Features Plugin
- PC-015-02A: Lag Features Plugin
- PC-016-02A: Cyclical Encoding Plugin

**Enables:**
- Complete feature engineering workflow
- Reproducible feature generation
- Easy plugin combinations and experimentation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] FeaturePipeline class implemented
- [ ] Plugin orchestration working
- [ ] Priority-based execution
- [ ] Feature collision detection
- [ ] Feature validation
- [ ] Configuration file support (YAML)
- [ ] Metadata tracking
- [ ] Unit tests pass with >85% coverage
- [ ] Integration tests with real plugins pass
- [ ] Performance benchmark met (<2s for full pipeline)
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-016-02A: Cyclical Encoding Plugin](PC-016-02A-cyclical-encoding-plugin.md)  
**Epic:** [Epic-02A: Core Feature Engineering](../epics/Epic-02A.md)

---

## 📈 Story Points Summary for Epic-02A

| Ticket | Story Points |
|--------|-------------|
| PC-012-02A | 5 |
| PC-013-02A | 5 |
| PC-014-02A | 5 |
| PC-015-02A | 8 |
| PC-016-02A | 3 |
| PC-017-02A | 8 |
| **Total** | **34** |
