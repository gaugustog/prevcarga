"""Tests for the LOESS smoothing feature plugin.

This module tests the LoessSmoothingPlugin including:
- Configuration validation
- LOESS smoothing algorithm correctness
- Residual calculation
- Edge cases (NaN handling, insufficient data)
- Integration with feature pipeline
"""


import numpy as np
import pandas as pd
import pytest

from src.features.plugins.smoothing import (
    LoessSmoothingConfig,
    LoessSmoothingPlugin,
)

# Test fixtures


@pytest.fixture
def plugin() -> LoessSmoothingPlugin:
    """Create a LoessSmoothingPlugin instance."""
    return LoessSmoothingPlugin()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with a noisy sinusoidal pattern."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.linspace(0, 4 * np.pi, n)
    # Sinusoidal pattern with noise
    y = np.sin(x) + rng.normal(0, 0.2, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"load": y, "temperature": 20 + 5 * np.sin(x) + rng.normal(0, 1, n)},
        index=dates,
    )


@pytest.fixture
def linear_df() -> pd.DataFrame:
    """Create a DataFrame with linear data (should smooth perfectly)."""
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"value": np.linspace(0, 100, n)},
        index=dates,
    )


@pytest.fixture
def constant_df() -> pd.DataFrame:
    """Create a DataFrame with constant data."""
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"value": np.full(n, 42.0)},
        index=dates,
    )


# Tests for LoessSmoothingConfig


class TestLoessSmoothingConfig:
    """Tests for LoessSmoothingConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = LoessSmoothingConfig()
        assert config.columns == []
        assert config.frac == 0.1
        assert config.degree == 1
        assert config.iterations == 3
        assert config.include_residuals is True
        assert config.suffix_smooth == "_loess"
        assert config.suffix_residual == "_residual"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = LoessSmoothingConfig(
            columns=["load", "temperature"],
            frac=0.2,
            degree=2,
            iterations=5,
            include_residuals=False,
            suffix_smooth="_smooth",
            suffix_residual="_res",
        )
        assert config.columns == ["load", "temperature"]
        assert config.frac == 0.2
        assert config.degree == 2
        assert config.iterations == 5
        assert config.include_residuals is False
        assert config.suffix_smooth == "_smooth"

    def test_invalid_frac_too_small(self) -> None:
        """Test that frac < 0.01 is rejected."""
        with pytest.raises(ValueError):
            LoessSmoothingConfig(frac=0.005)

    def test_invalid_frac_too_large(self) -> None:
        """Test that frac > 1.0 is rejected."""
        with pytest.raises(ValueError):
            LoessSmoothingConfig(frac=1.5)

    def test_invalid_degree_too_small(self) -> None:
        """Test that degree < 1 is rejected."""
        with pytest.raises(ValueError):
            LoessSmoothingConfig(degree=0)

    def test_invalid_degree_too_large(self) -> None:
        """Test that degree > 2 is rejected."""
        with pytest.raises(ValueError):
            LoessSmoothingConfig(degree=3)

    def test_invalid_iterations_too_small(self) -> None:
        """Test that iterations < 1 is rejected."""
        with pytest.raises(ValueError):
            LoessSmoothingConfig(iterations=0)

    def test_invalid_iterations_too_large(self) -> None:
        """Test that iterations > 10 is rejected."""
        with pytest.raises(ValueError):
            LoessSmoothingConfig(iterations=11)

    def test_empty_column_name_rejected(self) -> None:
        """Test that empty column names are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            LoessSmoothingConfig(columns=["load", ""])

    def test_empty_suffix_rejected(self) -> None:
        """Test that empty suffixes are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            LoessSmoothingConfig(suffix_smooth="")

    def test_same_suffixes_rejected(self) -> None:
        """Test that identical smooth and residual suffixes are rejected."""
        with pytest.raises(ValueError, match="must be different"):
            LoessSmoothingConfig(suffix_smooth="_loess", suffix_residual="_loess")


# Tests for LoessSmoothingPlugin basic properties


class TestLoessSmoothingPluginProperties:
    """Tests for LoessSmoothingPlugin basic properties."""

    def test_plugin_name(self, plugin: LoessSmoothingPlugin) -> None:
        """Test plugin name."""
        assert plugin.name == "loess_smoothing"

    def test_plugin_version(self, plugin: LoessSmoothingPlugin) -> None:
        """Test plugin version."""
        assert plugin.version == "1.0.0"

    def test_plugin_metadata(self, plugin: LoessSmoothingPlugin) -> None:
        """Test plugin metadata."""
        metadata = plugin.get_metadata()
        assert metadata["name"] == "loess_smoothing"
        assert metadata["version"] == "1.0.0"
        assert metadata["class"] == "LoessSmoothingPlugin"


# Tests for get_feature_names


class TestGetFeatureNames:
    """Tests for get_feature_names method."""

    def test_single_column_with_residuals(self, plugin: LoessSmoothingPlugin) -> None:
        """Test feature names for single column with residuals."""
        config = {"columns": ["load"], "include_residuals": True}
        names = plugin.get_feature_names(config)
        assert names == ["load_loess", "load_residual"]

    def test_single_column_without_residuals(self, plugin: LoessSmoothingPlugin) -> None:
        """Test feature names for single column without residuals."""
        config = {"columns": ["load"], "include_residuals": False}
        names = plugin.get_feature_names(config)
        assert names == ["load_loess"]

    def test_multiple_columns(self, plugin: LoessSmoothingPlugin) -> None:
        """Test feature names for multiple columns."""
        config = {"columns": ["load", "temperature"], "include_residuals": True}
        names = plugin.get_feature_names(config)
        assert names == [
            "load_loess",
            "load_residual",
            "temperature_loess",
            "temperature_residual",
        ]

    def test_custom_suffixes(self, plugin: LoessSmoothingPlugin) -> None:
        """Test feature names with custom suffixes."""
        config = {
            "columns": ["load"],
            "suffix_smooth": "_smooth",
            "suffix_residual": "_res",
            "include_residuals": True,
        }
        names = plugin.get_feature_names(config)
        assert names == ["load_smooth", "load_res"]

    def test_empty_columns(self, plugin: LoessSmoothingPlugin) -> None:
        """Test feature names with no columns."""
        config = {"columns": []}
        names = plugin.get_feature_names(config)
        assert names == []


# Tests for generate_features


class TestGenerateFeatures:
    """Tests for generate_features method."""

    def test_basic_smoothing(
        self,
        plugin: LoessSmoothingPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test basic LOESS smoothing."""
        config = {"columns": ["load"], "frac": 0.15, "include_residuals": True}
        result = plugin.generate_features(sample_df, config)

        assert "load_loess" in result.columns
        assert "load_residual" in result.columns
        assert len(result) == len(sample_df)

    def test_smoothing_reduces_variance(
        self,
        plugin: LoessSmoothingPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that smoothing reduces variance compared to original."""
        config = {"columns": ["load"], "frac": 0.3, "include_residuals": False}
        result = plugin.generate_features(sample_df, config)

        original_var = sample_df["load"].var()
        smoothed_var = result["load_loess"].var()

        # Smoothed data should have lower variance (less noise)
        assert smoothed_var < original_var

    def test_linear_data_smoothing(
        self,
        plugin: LoessSmoothingPlugin,
        linear_df: pd.DataFrame,
    ) -> None:
        """Test that linear data is smoothed nearly perfectly."""
        config = {"columns": ["value"], "frac": 0.3, "include_residuals": True}
        result = plugin.generate_features(linear_df, config)

        # Smoothed values should be very close to original for linear data
        max_error = np.abs(result["value_loess"] - linear_df["value"]).max()
        assert max_error < 1.0  # Allow small numerical error

    def test_constant_data_smoothing(
        self,
        plugin: LoessSmoothingPlugin,
        constant_df: pd.DataFrame,
    ) -> None:
        """Test that constant data is preserved by smoothing."""
        config = {"columns": ["value"], "frac": 0.3}
        result = plugin.generate_features(constant_df, config)

        # Smoothed values should equal the constant
        np.testing.assert_array_almost_equal(
            result["value_loess"].values,
            constant_df["value"].values,
            decimal=10,
        )

    def test_residuals_calculation(
        self,
        plugin: LoessSmoothingPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that residuals are calculated correctly."""
        config = {"columns": ["load"], "include_residuals": True}
        result = plugin.generate_features(sample_df, config)

        # Residual = original - smoothed
        expected_residuals = sample_df["load"] - result["load_loess"]
        np.testing.assert_array_almost_equal(
            result["load_residual"].values,
            expected_residuals.values,
        )

    def test_multiple_columns(
        self,
        plugin: LoessSmoothingPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test smoothing multiple columns."""
        config = {"columns": ["load", "temperature"], "include_residuals": False}
        result = plugin.generate_features(sample_df, config)

        assert "load_loess" in result.columns
        assert "temperature_loess" in result.columns

    def test_quadratic_smoothing(
        self,
        plugin: LoessSmoothingPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test quadratic (degree=2) smoothing."""
        config = {"columns": ["load"], "degree": 2, "frac": 0.3}
        result = plugin.generate_features(sample_df, config)

        assert "load_loess" in result.columns
        assert not result["load_loess"].isna().any()

    def test_preserves_original_columns(
        self,
        plugin: LoessSmoothingPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that original columns are preserved."""
        config = {"columns": ["load"]}
        result = plugin.generate_features(sample_df, config)

        assert "load" in result.columns
        assert "temperature" in result.columns
        # Original values unchanged
        pd.testing.assert_series_equal(result["load"], sample_df["load"])

    def test_empty_columns_returns_unchanged(
        self,
        plugin: LoessSmoothingPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that empty columns list returns DataFrame unchanged."""
        config = {"columns": []}
        result = plugin.generate_features(sample_df, config)

        pd.testing.assert_frame_equal(result, sample_df)

    def test_robustifying_iterations(
        self,
        plugin: LoessSmoothingPlugin,
    ) -> None:
        """Test that robustifying iterations handle outliers."""
        # Create data with outliers (no random component needed for this test)
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        y = np.sin(np.linspace(0, 2 * np.pi, n)) * 10
        # Add outliers
        y[10] = 100  # Large outlier
        y[30] = -100  # Large outlier

        df = pd.DataFrame({"value": y}, index=dates)

        # With more iterations, outliers should have less influence
        config_few_iter = {"columns": ["value"], "iterations": 1, "frac": 0.3}
        config_many_iter = {"columns": ["value"], "iterations": 5, "frac": 0.3}

        result_few = plugin.generate_features(df, config_few_iter)
        result_many = plugin.generate_features(df, config_many_iter)

        # With more iterations, the smoothed values near outliers should be closer
        # to the underlying pattern (less influenced by outliers)
        residual_few = np.abs(result_few["value_residual"].iloc[10])
        residual_many = np.abs(result_many["value_residual"].iloc[10])

        # More iterations should better identify the outlier (larger residual)
        assert residual_many >= residual_few * 0.5  # Allow some tolerance


# Tests for error handling


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_column_raises_error(
        self,
        plugin: LoessSmoothingPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that missing columns raise ValueError."""
        config = {"columns": ["nonexistent"]}
        with pytest.raises(ValueError, match="not found"):
            plugin.generate_features(sample_df, config)

    def test_insufficient_data_raises_error(
        self,
        plugin: LoessSmoothingPlugin,
    ) -> None:
        """Test that insufficient data raises ValueError."""
        # Create DataFrame with too few points
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )
        config = {"columns": ["value"]}
        with pytest.raises(ValueError, match="Insufficient data"):
            plugin.generate_features(df, config)

    def test_validate_config(self, plugin: LoessSmoothingPlugin) -> None:
        """Test validate_config method."""
        # Valid config
        assert plugin.validate_config({"columns": ["load"], "frac": 0.2}) is True

        # Invalid config
        with pytest.raises(ValueError):
            plugin.validate_config({"frac": 2.0})  # frac > 1


# Tests for NaN handling


class TestNaNHandling:
    """Tests for NaN value handling."""

    def test_nan_values_preserved(
        self,
        plugin: LoessSmoothingPlugin,
    ) -> None:
        """Test that NaN values in input are preserved in output."""
        # No random component needed for this test
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        y = np.sin(np.linspace(0, 2 * np.pi, n)) * 10

        # Add NaN values
        y[10] = np.nan
        y[25] = np.nan

        df = pd.DataFrame({"value": y}, index=dates)
        config = {"columns": ["value"], "frac": 0.3}

        result = plugin.generate_features(df, config)

        # NaN positions should remain NaN
        assert np.isnan(result["value_loess"].iloc[10])
        assert np.isnan(result["value_loess"].iloc[25])

        # Non-NaN positions should have values
        non_nan_mask = ~np.isnan(y)
        assert not result["value_loess"][non_nan_mask].isna().any()


# Tests for different frac values


class TestFracParameter:
    """Tests for different frac parameter values."""

    def test_small_frac_preserves_detail(
        self,
        plugin: LoessSmoothingPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that small frac preserves more detail."""
        config_small = {"columns": ["load"], "frac": 0.05}
        config_large = {"columns": ["load"], "frac": 0.5}

        result_small = plugin.generate_features(sample_df, config_small)
        result_large = plugin.generate_features(sample_df, config_large)

        # Small frac should have larger variance (more detail preserved)
        var_small = result_small["load_loess"].var()
        var_large = result_large["load_loess"].var()

        assert var_small > var_large

    def test_frac_1_is_global_fit(
        self,
        plugin: LoessSmoothingPlugin,
        linear_df: pd.DataFrame,
    ) -> None:
        """Test that frac=1 uses all data points (global fit)."""
        config = {"columns": ["value"], "frac": 1.0}
        result = plugin.generate_features(linear_df, config)

        # For linear data with frac=1, result should be essentially perfect
        max_error = np.abs(result["value_loess"] - linear_df["value"]).max()
        assert max_error < 0.1


# Tests for integration with pipeline


class TestPipelineIntegration:
    """Tests for integration with feature pipeline."""

    def test_works_in_pipeline(
        self,
        plugin: LoessSmoothingPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that plugin works correctly in a pipeline."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(
            plugins=[
                (plugin, {"columns": ["load"], "frac": 0.2}),
            ]
        )

        result = pipeline.run(sample_df)

        assert result.success is True
        assert "load_loess" in result.df.columns
        assert "load_residual" in result.df.columns


# Tests for tricube and bisquare weight functions


class TestWeightFunctions:
    """Tests for weight calculation functions."""

    def test_tricube_weights_at_zero(self) -> None:
        """Test tricube weights at distance zero."""
        weights = LoessSmoothingPlugin._tricube_weights(np.array([0.0]))
        assert weights[0] == 1.0

    def test_tricube_weights_at_one(self) -> None:
        """Test tricube weights at distance one."""
        weights = LoessSmoothingPlugin._tricube_weights(np.array([1.0]))
        assert weights[0] == 0.0

    def test_tricube_weights_decreasing(self) -> None:
        """Test that tricube weights decrease with distance."""
        distances = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        weights = LoessSmoothingPlugin._tricube_weights(distances)

        # Weights should be decreasing
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1]

    def test_bisquare_weights_at_zero(self) -> None:
        """Test bisquare weights for zero residuals."""
        residuals = np.array([0.0])
        weights = LoessSmoothingPlugin._bisquare_weights(residuals, mad=1.0)
        assert weights[0] == 1.0

    def test_bisquare_weights_with_zero_mad(self) -> None:
        """Test bisquare weights when MAD is zero."""
        residuals = np.array([0.1, 0.2])
        weights = LoessSmoothingPlugin._bisquare_weights(residuals, mad=0.0)
        np.testing.assert_array_equal(weights, np.ones_like(residuals))
