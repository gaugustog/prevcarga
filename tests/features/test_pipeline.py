"""Tests for the feature pipeline composer.

This module tests the FeaturePipeline class including:
- Pipeline construction and configuration
- Plugin execution and chaining
- Feature collision detection
- Error handling and recovery
- Execution metrics and timing
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.base.plugin import BaseFeaturePlugin
from src.features.pipeline import (
    FeatureCollisionError,
    FeaturePipeline,
    PipelineConfig,
    PipelineConfigError,
    PipelineExecutionError,
    PluginStep,
)
from src.features.registry import PluginNotFoundError, PluginRegistry

# Test fixtures and helper classes


class SimplePlugin(BaseFeaturePlugin):
    """Simple plugin for testing that doubles a column value."""

    @property
    def name(self) -> str:
        return "simple_plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        column = config.get("column", "value")
        df[f"{column}_doubled"] = df[column] * 2
        return df

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        column = config.get("column", "value")
        return [f"{column}_doubled"]


class AnotherPlugin(BaseFeaturePlugin):
    """Another simple plugin for testing that triples a column value."""

    @property
    def name(self) -> str:
        return "another_plugin"

    @property
    def version(self) -> str:
        return "2.0.0"

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        column = config.get("column", "value")
        df[f"{column}_tripled"] = df[column] * 3
        return df

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        column = config.get("column", "value")
        return [f"{column}_tripled"]


class FailingPlugin(BaseFeaturePlugin):
    """Plugin that always fails for error testing."""

    @property
    def name(self) -> str:
        return "failing_plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        msg = "Intentional failure for testing"
        raise RuntimeError(msg)

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        return ["will_fail"]


class CollidingPlugin(BaseFeaturePlugin):
    """Plugin that generates the same feature name as SimplePlugin."""

    @property
    def name(self) -> str:
        return "colliding_plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        column = config.get("column", "value")
        df[f"{column}_doubled"] = df[column] * 4  # Different calculation, same name
        return df

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        column = config.get("column", "value")
        return [f"{column}_doubled"]


class MultiFeaturePlugin(BaseFeaturePlugin):
    """Plugin that generates multiple features."""

    @property
    def name(self) -> str:
        return "multi_feature"

    @property
    def version(self) -> str:
        return "1.0.0"

    def generate_features(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        column = config.get("column", "value")
        df[f"{column}_squared"] = df[column] ** 2
        df[f"{column}_sqrt"] = np.sqrt(np.abs(df[column]))
        df[f"{column}_log"] = np.log1p(np.abs(df[column]))
        return df

    def get_feature_names(self, config: dict[str, Any]) -> list[str]:
        column = config.get("column", "value")
        return [f"{column}_squared", f"{column}_sqrt", f"{column}_log"]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=24, freq="h")
    return pd.DataFrame(
        {"value": np.arange(1, 25), "other": np.arange(100, 124)},
        index=dates,
    )


@pytest.fixture
def simple_plugin() -> SimplePlugin:
    """Create a SimplePlugin instance."""
    return SimplePlugin()


@pytest.fixture
def another_plugin() -> AnotherPlugin:
    """Create an AnotherPlugin instance."""
    return AnotherPlugin()


@pytest.fixture
def registry() -> PluginRegistry:
    """Create a clean registry for testing."""
    reg = PluginRegistry()
    reg.clear()
    return reg


# Tests for PipelineConfig


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.fail_fast is True
        assert config.validate_features is True
        assert config.detect_collisions is True
        assert config.copy_input is True
        assert config.preserve_input_columns is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = PipelineConfig(
            fail_fast=False,
            validate_features=False,
            detect_collisions=False,
            copy_input=False,
            preserve_input_columns=False,
        )
        assert config.fail_fast is False
        assert config.validate_features is False
        assert config.detect_collisions is False
        assert config.copy_input is False
        assert config.preserve_input_columns is False

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable."""
        from pydantic import ValidationError

        config = PipelineConfig()
        with pytest.raises(ValidationError):
            config.fail_fast = False  # type: ignore[misc]


# Tests for PluginStep


class TestPluginStep:
    """Tests for PluginStep."""

    def test_basic_step(self) -> None:
        """Test basic PluginStep creation."""
        step = PluginStep(plugin_name="test_plugin")
        assert step.plugin_name == "test_plugin"
        assert step.config == {}
        assert step.enabled is True

    def test_step_with_config(self) -> None:
        """Test PluginStep with configuration."""
        step = PluginStep(
            plugin_name="test_plugin",
            config={"column": "value", "option": True},
            enabled=False,
        )
        assert step.plugin_name == "test_plugin"
        assert step.config == {"column": "value", "option": True}
        assert step.enabled is False

    def test_empty_plugin_name_fails(self) -> None:
        """Test that empty plugin name fails validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PluginStep(plugin_name="")

    def test_whitespace_plugin_name_fails(self) -> None:
        """Test that whitespace-only plugin name fails validation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PluginStep(plugin_name="   ")

    def test_plugin_name_trimmed(self) -> None:
        """Test that plugin name is trimmed."""
        step = PluginStep(plugin_name="  test_plugin  ")
        assert step.plugin_name == "test_plugin"


# Tests for FeaturePipeline construction


class TestFeaturePipelineConstruction:
    """Tests for FeaturePipeline construction."""

    def test_empty_pipeline(self) -> None:
        """Test creating an empty pipeline."""
        pipeline = FeaturePipeline()
        assert pipeline.plugin_count == 0
        assert len(pipeline) == 0

    def test_pipeline_with_plugins(
        self,
        simple_plugin: SimplePlugin,
        another_plugin: AnotherPlugin,
    ) -> None:
        """Test creating a pipeline with plugins."""
        pipeline = FeaturePipeline(
            plugins=[
                (simple_plugin, {"column": "value"}),
                (another_plugin, {}),
            ]
        )
        assert pipeline.plugin_count == 2

    def test_add_plugin(self, simple_plugin: SimplePlugin) -> None:
        """Test adding plugins to pipeline."""
        pipeline = FeaturePipeline()
        result = pipeline.add_plugin(simple_plugin, {"column": "value"})

        # Should return self for chaining
        assert result is pipeline
        assert pipeline.plugin_count == 1

    def test_add_plugin_chaining(
        self,
        simple_plugin: SimplePlugin,
        another_plugin: AnotherPlugin,
    ) -> None:
        """Test method chaining when adding plugins."""
        pipeline = (
            FeaturePipeline()
            .add_plugin(simple_plugin, {})
            .add_plugin(another_plugin, {})
        )
        assert pipeline.plugin_count == 2

    def test_add_invalid_plugin_type(self) -> None:
        """Test that adding non-plugin raises TypeError."""
        pipeline = FeaturePipeline()
        with pytest.raises(TypeError, match="Expected BaseFeaturePlugin"):
            pipeline.add_plugin("not a plugin", {})  # type: ignore[arg-type]

    def test_remove_plugin(
        self,
        simple_plugin: SimplePlugin,
        another_plugin: AnotherPlugin,
    ) -> None:
        """Test removing plugins from pipeline."""
        pipeline = FeaturePipeline(
            plugins=[
                (simple_plugin, {}),
                (another_plugin, {}),
            ]
        )

        result = pipeline.remove_plugin("simple_plugin")
        assert result is True
        assert pipeline.plugin_count == 1

    def test_remove_nonexistent_plugin(self, simple_plugin: SimplePlugin) -> None:
        """Test removing a plugin that doesn't exist."""
        pipeline = FeaturePipeline(plugins=[(simple_plugin, {})])

        result = pipeline.remove_plugin("nonexistent")
        assert result is False
        assert pipeline.plugin_count == 1

    def test_get_plugins(
        self,
        simple_plugin: SimplePlugin,
        another_plugin: AnotherPlugin,
    ) -> None:
        """Test getting list of plugins."""
        pipeline = FeaturePipeline(
            plugins=[
                (simple_plugin, {}),
                (another_plugin, {}),
            ]
        )

        plugins = pipeline.get_plugins()
        assert plugins == [("simple_plugin", "1.0.0"), ("another_plugin", "2.0.0")]

    def test_pipeline_repr(self, simple_plugin: SimplePlugin) -> None:
        """Test pipeline string representation."""
        pipeline = FeaturePipeline(plugins=[(simple_plugin, {})])
        repr_str = repr(pipeline)
        assert "FeaturePipeline" in repr_str
        assert "simple_plugin" in repr_str


# Tests for FeaturePipeline.from_registry


class TestFeaturePipelineFromRegistry:
    """Tests for creating pipelines from registry."""

    def test_from_registry_basic(
        self,
        registry: PluginRegistry,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test creating pipeline from registry."""
        registry.register(simple_plugin)

        pipeline = FeaturePipeline.from_registry(
            steps=[PluginStep(plugin_name="simple_plugin")],
            registry=registry,
        )
        assert pipeline.plugin_count == 1

    def test_from_registry_with_config(
        self,
        registry: PluginRegistry,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test creating pipeline from registry with config."""
        registry.register(simple_plugin)

        pipeline = FeaturePipeline.from_registry(
            steps=[
                PluginStep(
                    plugin_name="simple_plugin",
                    config={"column": "other"},
                )
            ],
            registry=registry,
        )
        assert pipeline.plugin_count == 1

    def test_from_registry_disabled_step(
        self,
        registry: PluginRegistry,
        simple_plugin: SimplePlugin,
        another_plugin: AnotherPlugin,
    ) -> None:
        """Test that disabled steps are skipped."""
        registry.register(simple_plugin)
        registry.register(another_plugin)

        pipeline = FeaturePipeline.from_registry(
            steps=[
                PluginStep(plugin_name="simple_plugin", enabled=True),
                PluginStep(plugin_name="another_plugin", enabled=False),
            ],
            registry=registry,
        )
        assert pipeline.plugin_count == 1

    def test_from_registry_plugin_not_found(self, registry: PluginRegistry) -> None:
        """Test error when plugin not found in registry."""
        with pytest.raises(PluginNotFoundError):
            FeaturePipeline.from_registry(
                steps=[PluginStep(plugin_name="nonexistent")],
                registry=registry,
            )


# Tests for feature collision detection


class TestFeatureCollisionDetection:
    """Tests for feature collision detection."""

    def test_no_collisions(
        self,
        simple_plugin: SimplePlugin,
        another_plugin: AnotherPlugin,
    ) -> None:
        """Test detection when no collisions exist."""
        pipeline = FeaturePipeline(
            plugins=[
                (simple_plugin, {"column": "value"}),
                (another_plugin, {"column": "value"}),
            ]
        )

        collisions = pipeline.detect_feature_collisions()
        assert collisions == {}

    def test_detect_collisions(self, simple_plugin: SimplePlugin) -> None:
        """Test detection of feature collisions."""
        colliding_plugin = CollidingPlugin()
        pipeline = FeaturePipeline(
            plugins=[
                (simple_plugin, {"column": "value"}),
                (colliding_plugin, {"column": "value"}),
            ]
        )

        collisions = pipeline.detect_feature_collisions()
        assert "value_doubled" in collisions
        assert set(collisions["value_doubled"]) == {"simple_plugin", "colliding_plugin"}

    def test_get_all_feature_names(
        self,
        simple_plugin: SimplePlugin,
        another_plugin: AnotherPlugin,
    ) -> None:
        """Test getting all feature names."""
        pipeline = FeaturePipeline(
            plugins=[
                (simple_plugin, {"column": "value"}),
                (another_plugin, {"column": "value"}),
            ]
        )

        names = pipeline.get_all_feature_names()
        assert "value_doubled" in names
        assert "value_tripled" in names


# Tests for pipeline validation


class TestPipelineValidation:
    """Tests for pipeline validation."""

    def test_validate_empty_pipeline(self) -> None:
        """Test validation of empty pipeline."""
        pipeline = FeaturePipeline()
        errors = pipeline.validate()
        assert any("no plugins" in e.lower() for e in errors)

    def test_validate_valid_pipeline(self, simple_plugin: SimplePlugin) -> None:
        """Test validation of valid pipeline."""
        pipeline = FeaturePipeline(plugins=[(simple_plugin, {})])
        errors = pipeline.validate()
        assert errors == []

    def test_validate_with_collisions(self, simple_plugin: SimplePlugin) -> None:
        """Test validation detects collisions."""
        colliding_plugin = CollidingPlugin()
        config = PipelineConfig(detect_collisions=True)
        pipeline = FeaturePipeline(
            plugins=[
                (simple_plugin, {"column": "value"}),
                (colliding_plugin, {"column": "value"}),
            ],
            config=config,
        )

        errors = pipeline.validate()
        assert any("multiple plugins" in e for e in errors)


# Tests for pipeline execution


class TestPipelineExecution:
    """Tests for pipeline execution."""

    def test_basic_execution(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test basic pipeline execution."""
        pipeline = FeaturePipeline(plugins=[(simple_plugin, {"column": "value"})])
        result = pipeline.run(sample_df)

        assert result.success is True
        assert "value_doubled" in result.df.columns
        assert result.plugins_executed == 1
        assert "value_doubled" in result.feature_names

    def test_multi_plugin_execution(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
        another_plugin: AnotherPlugin,
    ) -> None:
        """Test execution with multiple plugins."""
        pipeline = FeaturePipeline(
            plugins=[
                (simple_plugin, {"column": "value"}),
                (another_plugin, {"column": "value"}),
            ]
        )
        result = pipeline.run(sample_df)

        assert result.success is True
        assert "value_doubled" in result.df.columns
        assert "value_tripled" in result.df.columns
        assert result.plugins_executed == 2

    def test_execution_preserves_input_columns(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test that input columns are preserved by default."""
        config = PipelineConfig(preserve_input_columns=True)
        pipeline = FeaturePipeline(
            plugins=[(simple_plugin, {"column": "value"})],
            config=config,
        )
        result = pipeline.run(sample_df)

        assert "value" in result.df.columns
        assert "other" in result.df.columns

    def test_execution_removes_input_columns(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test that input columns can be removed."""
        config = PipelineConfig(preserve_input_columns=False)
        pipeline = FeaturePipeline(
            plugins=[(simple_plugin, {"column": "value"})],
            config=config,
        )
        result = pipeline.run(sample_df)

        assert "value" not in result.df.columns
        assert "other" not in result.df.columns
        assert "value_doubled" in result.df.columns

    def test_execution_copies_input(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test that input DataFrame is copied by default."""
        config = PipelineConfig(copy_input=True)
        pipeline = FeaturePipeline(
            plugins=[(simple_plugin, {"column": "value"})],
            config=config,
        )
        original_columns = list(sample_df.columns)
        pipeline.run(sample_df)

        # Original should be unchanged
        assert list(sample_df.columns) == original_columns

    def test_execution_modifies_input_when_no_copy(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test that input DataFrame is modified when copy_input=False."""
        config = PipelineConfig(copy_input=False, preserve_input_columns=True)
        pipeline = FeaturePipeline(
            plugins=[(simple_plugin, {"column": "value"})],
            config=config,
        )
        pipeline.run(sample_df)

        # Original should be modified
        assert "value_doubled" in sample_df.columns

    def test_execution_with_override_configs(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test execution with config overrides."""
        pipeline = FeaturePipeline(
            plugins=[(simple_plugin, {"column": "value"})]
        )
        result = pipeline.run(
            sample_df,
            override_configs={"simple_plugin": {"column": "other"}},
        )

        assert "other_doubled" in result.df.columns

    def test_execution_timing(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test that execution timing is recorded."""
        pipeline = FeaturePipeline(plugins=[(simple_plugin, {})])
        result = pipeline.run(sample_df)

        assert result.execution_time_ms > 0
        assert result.plugin_results[0].execution_time_ms > 0


# Tests for error handling


class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    def test_fail_fast_on_error(self, sample_df: pd.DataFrame) -> None:
        """Test that pipeline stops on error when fail_fast=True."""
        config = PipelineConfig(fail_fast=True)
        pipeline = FeaturePipeline(
            plugins=[
                (FailingPlugin(), {}),
                (SimplePlugin(), {}),
            ],
            config=config,
        )

        with pytest.raises(PipelineExecutionError):
            pipeline.run(sample_df)

    def test_continue_on_error(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test that pipeline continues on error when fail_fast=False."""
        config = PipelineConfig(fail_fast=False)
        pipeline = FeaturePipeline(
            plugins=[
                (FailingPlugin(), {}),
                (simple_plugin, {}),
            ],
            config=config,
        )

        result = pipeline.run(sample_df)
        assert result.success is False
        assert result.plugins_executed == 1  # Only simple_plugin succeeded
        assert "value_doubled" in result.df.columns

    def test_empty_pipeline_execution_fails(self, sample_df: pd.DataFrame) -> None:
        """Test that empty pipeline raises error."""
        pipeline = FeaturePipeline()

        with pytest.raises(PipelineConfigError, match="no plugins"):
            pipeline.run(sample_df)

    def test_collision_error_raised(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test that collision error is raised when detect_collisions=True."""
        config = PipelineConfig(detect_collisions=True)
        pipeline = FeaturePipeline(
            plugins=[
                (simple_plugin, {"column": "value"}),
                (CollidingPlugin(), {"column": "value"}),
            ],
            config=config,
        )

        with pytest.raises(FeatureCollisionError) as exc_info:
            pipeline.run(sample_df)

        assert "value_doubled" in exc_info.value.collisions


# Tests for PipelineResult


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_result_summary(
        self,
        sample_df: pd.DataFrame,
        simple_plugin: SimplePlugin,
    ) -> None:
        """Test getting result summary."""
        pipeline = FeaturePipeline(plugins=[(simple_plugin, {})])
        result = pipeline.run(sample_df)
        summary = result.get_summary()

        assert summary["success"] is True
        assert summary["plugins_executed"] == 1
        assert summary["features_generated"] == 1
        assert summary["execution_time_ms"] > 0
        assert summary["input_rows"] == 24
        assert len(summary["plugin_details"]) == 1

    def test_plugin_execution_result_on_failure(
        self,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test PluginExecutionResult captures failure info."""
        config = PipelineConfig(fail_fast=False)
        pipeline = FeaturePipeline(
            plugins=[(FailingPlugin(), {})],
            config=config,
        )

        result = pipeline.run(sample_df)
        plugin_result = result.plugin_results[0]

        assert plugin_result.success is False
        assert plugin_result.error_message is not None
        assert "Intentional failure" in plugin_result.error_message


# Tests for multi-feature plugins


class TestMultiFeaturePlugin:
    """Tests for plugins that generate multiple features."""

    def test_multi_feature_execution(self, sample_df: pd.DataFrame) -> None:
        """Test execution with multi-feature plugin."""
        plugin = MultiFeaturePlugin()
        pipeline = FeaturePipeline(plugins=[(plugin, {"column": "value"})])
        result = pipeline.run(sample_df)

        assert result.success is True
        assert "value_squared" in result.df.columns
        assert "value_sqrt" in result.df.columns
        assert "value_log" in result.df.columns
        assert len(result.feature_names) == 3


# Integration tests


class TestPipelineIntegration:
    """Integration tests for the pipeline."""

    def test_complex_pipeline(self, sample_df: pd.DataFrame) -> None:
        """Test complex pipeline with multiple plugins."""
        pipeline = FeaturePipeline(
            plugins=[
                (SimplePlugin(), {"column": "value"}),
                (AnotherPlugin(), {"column": "value"}),
                (MultiFeaturePlugin(), {"column": "value"}),
            ]
        )

        result = pipeline.run(sample_df)

        assert result.success is True
        assert result.plugins_executed == 3
        assert len(result.feature_names) == 5  # 1 + 1 + 3

    def test_pipeline_with_real_plugins(self, sample_df: pd.DataFrame) -> None:
        """Test pipeline with real project plugins if available."""
        from src.features.plugins.temporal import TemporalFeaturesPlugin

        plugin = TemporalFeaturesPlugin()
        pipeline = FeaturePipeline(plugins=[(plugin, {})])

        result = pipeline.run(sample_df)
        assert result.success is True
        assert result.plugins_executed == 1
        assert "hour" in result.df.columns

    def test_chained_plugin_dependencies(self, sample_df: pd.DataFrame) -> None:
        """Test plugins that depend on output of previous plugins."""
        # First plugin doubles value
        simple = SimplePlugin()

        # Custom plugin that uses the doubled value
        class DependentPlugin(BaseFeaturePlugin):
            @property
            def name(self) -> str:
                return "dependent_plugin"

            @property
            def version(self) -> str:
                return "1.0.0"

            def generate_features(
                self,
                df: pd.DataFrame,
                config: dict[str, Any],
            ) -> pd.DataFrame:
                # Uses output of SimplePlugin
                df["quadrupled"] = df["value_doubled"] * 2
                return df

            def get_feature_names(self, config: dict[str, Any]) -> list[str]:
                return ["quadrupled"]

        pipeline = FeaturePipeline(
            plugins=[
                (simple, {"column": "value"}),
                (DependentPlugin(), {}),
            ]
        )

        result = pipeline.run(sample_df)
        assert result.success is True
        assert result.df["quadrupled"].iloc[0] == 4  # 1 * 2 * 2
