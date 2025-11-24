"""Tests for the BLF strategy feature plugin."""

import numpy as np
import pandas as pd
import pytest

from src.features.plugins.blf import (
    BLFStrategyConfig,
    BLFStrategyPlugin,
)


@pytest.fixture
def plugin() -> BLFStrategyPlugin:
    """Create a BLFStrategyPlugin instance."""
    return BLFStrategyPlugin()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with daily load pattern."""
    # Create 14 days of hourly data
    dates = pd.date_range("2024-01-01", periods=14 * 24, freq="h")
    # Simple daily pattern: higher during day, lower at night
    hour = dates.hour
    load = 100 + 20 * np.sin((hour - 6) * np.pi / 12)
    return pd.DataFrame({"load": load}, index=dates)


class TestBLFStrategyConfig:
    """Tests for BLFStrategyConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = BLFStrategyConfig()
        assert config.load_column == "load"
        assert config.reference_days == 7
        assert config.include_baseline is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BLFStrategyConfig(
            load_column="demand",
            reference_days=14,
            include_baseline=False,
        )
        assert config.load_column == "demand"
        assert config.reference_days == 14


class TestBLFStrategyPlugin:
    """Tests for BLFStrategyPlugin."""

    def test_plugin_name(self, plugin: BLFStrategyPlugin) -> None:
        """Test plugin name."""
        assert plugin.name == "blf_strategy"

    def test_plugin_version(self, plugin: BLFStrategyPlugin) -> None:
        """Test plugin version."""
        assert plugin.version == "1.0.0"

    def test_generate_features(
        self,
        plugin: BLFStrategyPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test feature generation."""
        config = {"load_column": "load"}
        result = plugin.generate_features(sample_df, config)

        assert "load_baseline" in result.columns
        assert "load_ratio" in result.columns
        assert len(result) == len(sample_df)

    def test_get_feature_names(self, plugin: BLFStrategyPlugin) -> None:
        """Test feature name retrieval."""
        config = {
            "load_column": "load",
            "include_baseline": True,
            "include_ratios": True,
        }
        names = plugin.get_feature_names(config)
        assert "load_baseline" in names
        assert "load_ratio" in names
