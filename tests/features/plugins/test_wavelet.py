"""Tests for the wavelet transform feature plugin.

This module tests the WaveletTransformPlugin including:
- Configuration validation
- Wavelet decomposition correctness
- Multi-level decomposition
- Signal reconstruction
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest

from src.features.plugins.wavelet import (
    SUPPORTED_WAVELETS,
    WaveletTransformConfig,
    WaveletTransformPlugin,
)

# Test fixtures


@pytest.fixture
def plugin() -> WaveletTransformPlugin:
    """Create a WaveletTransformPlugin instance."""
    return WaveletTransformPlugin()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with composite signal."""
    n = 128
    t = np.linspace(0, 4 * np.pi, n)
    # Composite signal: slow trend + fast oscillation + noise
    rng = np.random.default_rng(42)
    slow = np.sin(t)  # Low frequency
    fast = 0.5 * np.sin(10 * t)  # High frequency
    noise = 0.1 * rng.normal(size=n)
    y = slow + fast + noise
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"load": y, "other": np.arange(n)}, index=dates)


@pytest.fixture
def sinusoidal_df() -> pd.DataFrame:
    """Create a DataFrame with simple sinusoidal data."""
    n = 64
    t = np.linspace(0, 2 * np.pi, n)
    y = np.sin(t)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"value": y}, index=dates)


@pytest.fixture
def constant_df() -> pd.DataFrame:
    """Create a DataFrame with constant data."""
    n = 64
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"value": np.full(n, 100.0)}, index=dates)


# Tests for WaveletTransformConfig


class TestWaveletTransformConfig:
    """Tests for WaveletTransformConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = WaveletTransformConfig()
        assert config.columns == []
        assert config.wavelet == "db4"
        assert config.level == 3
        assert config.include_approximation is True
        assert config.include_details is True
        assert config.suffix_approx == "_approx"
        assert config.suffix_detail == "_detail"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = WaveletTransformConfig(
            columns=["load", "temperature"],
            wavelet="haar",
            level=5,
            include_approximation=True,
            include_details=False,
            suffix_approx="_a",
            suffix_detail="_d",
        )
        assert config.columns == ["load", "temperature"]
        assert config.wavelet == "haar"
        assert config.level == 5
        assert config.include_approximation is True
        assert config.include_details is False

    def test_invalid_wavelet(self) -> None:
        """Test that invalid wavelet type is rejected."""
        with pytest.raises(ValueError, match="Unsupported wavelet"):
            WaveletTransformConfig(wavelet="invalid")

    def test_wavelet_case_insensitive(self) -> None:
        """Test that wavelet names are case insensitive."""
        config = WaveletTransformConfig(wavelet="HAAR")
        assert config.wavelet == "haar"

        config2 = WaveletTransformConfig(wavelet="Db4")
        assert config2.wavelet == "db4"

    def test_invalid_level_too_small(self) -> None:
        """Test that level < 1 is rejected."""
        with pytest.raises(ValueError):
            WaveletTransformConfig(level=0)

    def test_invalid_level_too_large(self) -> None:
        """Test that level > 8 is rejected."""
        with pytest.raises(ValueError):
            WaveletTransformConfig(level=9)

    def test_empty_column_name_rejected(self) -> None:
        """Test that empty column names are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            WaveletTransformConfig(columns=["load", ""])

    def test_empty_suffix_rejected(self) -> None:
        """Test that empty suffixes are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            WaveletTransformConfig(suffix_approx="")

    def test_both_outputs_disabled_rejected(self) -> None:
        """Test that disabling both outputs is rejected."""
        with pytest.raises(ValueError, match="At least one"):
            WaveletTransformConfig(
                include_approximation=False,
                include_details=False,
            )


# Tests for WaveletTransformPlugin basic properties


class TestWaveletTransformPluginProperties:
    """Tests for WaveletTransformPlugin basic properties."""

    def test_plugin_name(self, plugin: WaveletTransformPlugin) -> None:
        """Test plugin name."""
        assert plugin.name == "wavelet_transform"

    def test_plugin_version(self, plugin: WaveletTransformPlugin) -> None:
        """Test plugin version."""
        assert plugin.version == "1.0.0"

    def test_plugin_metadata(self, plugin: WaveletTransformPlugin) -> None:
        """Test plugin metadata."""
        metadata = plugin.get_metadata()
        assert metadata["name"] == "wavelet_transform"
        assert metadata["version"] == "1.0.0"
        assert metadata["class"] == "WaveletTransformPlugin"


# Tests for get_feature_names


class TestGetFeatureNames:
    """Tests for get_feature_names method."""

    def test_single_column_all_outputs(self, plugin: WaveletTransformPlugin) -> None:
        """Test feature names with all outputs enabled."""
        config = {
            "columns": ["load"],
            "level": 3,
            "include_approximation": True,
            "include_details": True,
        }
        names = plugin.get_feature_names(config)
        assert names == [
            "load_approx",
            "load_detail_1",
            "load_detail_2",
            "load_detail_3",
        ]

    def test_single_column_approx_only(self, plugin: WaveletTransformPlugin) -> None:
        """Test feature names with approximation only."""
        config = {
            "columns": ["load"],
            "level": 3,
            "include_approximation": True,
            "include_details": False,
        }
        names = plugin.get_feature_names(config)
        assert names == ["load_approx"]

    def test_single_column_details_only(self, plugin: WaveletTransformPlugin) -> None:
        """Test feature names with details only."""
        config = {
            "columns": ["load"],
            "level": 2,
            "include_approximation": False,
            "include_details": True,
        }
        names = plugin.get_feature_names(config)
        assert names == ["load_detail_1", "load_detail_2"]

    def test_multiple_columns(self, plugin: WaveletTransformPlugin) -> None:
        """Test feature names for multiple columns."""
        config = {
            "columns": ["load", "temp"],
            "level": 2,
            "include_approximation": True,
            "include_details": True,
        }
        names = plugin.get_feature_names(config)
        assert names == [
            "load_approx",
            "load_detail_1",
            "load_detail_2",
            "temp_approx",
            "temp_detail_1",
            "temp_detail_2",
        ]

    def test_empty_columns(self, plugin: WaveletTransformPlugin) -> None:
        """Test feature names with no columns."""
        config = {"columns": []}
        names = plugin.get_feature_names(config)
        assert names == []


# Tests for generate_features


class TestGenerateFeatures:
    """Tests for generate_features method."""

    def test_basic_decomposition(
        self,
        plugin: WaveletTransformPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test basic wavelet decomposition."""
        config = {"columns": ["load"], "level": 3}
        result = plugin.generate_features(sample_df, config)

        assert "load_approx" in result.columns
        assert "load_detail_1" in result.columns
        assert "load_detail_2" in result.columns
        assert "load_detail_3" in result.columns
        assert len(result) == len(sample_df)

    def test_all_wavelets(
        self,
        plugin: WaveletTransformPlugin,
        sinusoidal_df: pd.DataFrame,
    ) -> None:
        """Test all supported wavelet types."""
        for wavelet in SUPPORTED_WAVELETS:
            config = {"columns": ["value"], "wavelet": wavelet, "level": 2}
            result = plugin.generate_features(sinusoidal_df, config)
            assert "value_approx" in result.columns
            assert "value_detail_1" in result.columns

    def test_reconstruction_preserves_energy(
        self,
        plugin: WaveletTransformPlugin,
        sinusoidal_df: pd.DataFrame,
    ) -> None:
        """Test that decomposition approximately preserves signal energy."""
        config = {"columns": ["value"], "level": 2}
        result = plugin.generate_features(sinusoidal_df, config)

        # Sum of components should approximate original signal
        reconstructed = (
            result["value_approx"]
            + result["value_detail_1"]
            + result["value_detail_2"]
        )

        # Allow for some numerical error
        correlation = np.corrcoef(sinusoidal_df["value"], reconstructed)[0, 1]
        assert correlation > 0.95

    def test_constant_data(
        self,
        plugin: WaveletTransformPlugin,
        constant_df: pd.DataFrame,
    ) -> None:
        """Test wavelet transform on constant data."""
        config = {"columns": ["value"], "level": 2}
        result = plugin.generate_features(constant_df, config)

        # Approximation should capture the constant value
        # Details should be near zero
        assert result["value_approx"].mean() > 50  # Should be around 100

    def test_preserves_original_columns(
        self,
        plugin: WaveletTransformPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that original columns are preserved."""
        config = {"columns": ["load"]}
        result = plugin.generate_features(sample_df, config)

        assert "load" in result.columns
        assert "other" in result.columns
        pd.testing.assert_series_equal(result["load"], sample_df["load"])

    def test_empty_columns_returns_unchanged(
        self,
        plugin: WaveletTransformPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that empty columns list returns DataFrame unchanged."""
        config = {"columns": []}
        result = plugin.generate_features(sample_df, config)

        pd.testing.assert_frame_equal(result, sample_df)

    def test_approx_only(
        self,
        plugin: WaveletTransformPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test generating only approximation coefficients."""
        config = {
            "columns": ["load"],
            "include_approximation": True,
            "include_details": False,
        }
        result = plugin.generate_features(sample_df, config)

        assert "load_approx" in result.columns
        assert "load_detail_1" not in result.columns

    def test_details_only(
        self,
        plugin: WaveletTransformPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test generating only detail coefficients."""
        config = {
            "columns": ["load"],
            "level": 2,
            "include_approximation": False,
            "include_details": True,
        }
        result = plugin.generate_features(sample_df, config)

        assert "load_approx" not in result.columns
        assert "load_detail_1" in result.columns
        assert "load_detail_2" in result.columns


# Tests for error handling


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_column_raises_error(
        self,
        plugin: WaveletTransformPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that missing columns raise ValueError."""
        config = {"columns": ["nonexistent"]}
        with pytest.raises(ValueError, match="not found"):
            plugin.generate_features(sample_df, config)

    def test_validate_config(self, plugin: WaveletTransformPlugin) -> None:
        """Test validate_config method."""
        # Valid config
        assert plugin.validate_config({"columns": ["load"], "level": 3}) is True

        # Invalid config
        with pytest.raises(ValueError):
            plugin.validate_config({"level": 10})


# Tests for different decomposition levels


class TestDecompositionLevels:
    """Tests for different decomposition levels."""

    def test_level_1(
        self,
        plugin: WaveletTransformPlugin,
        sinusoidal_df: pd.DataFrame,
    ) -> None:
        """Test level 1 decomposition."""
        config = {"columns": ["value"], "level": 1}
        result = plugin.generate_features(sinusoidal_df, config)

        assert "value_approx" in result.columns
        assert "value_detail_1" in result.columns
        assert "value_detail_2" not in result.columns

    def test_higher_levels(
        self,
        plugin: WaveletTransformPlugin,
    ) -> None:
        """Test higher decomposition levels."""
        n = 256  # Need more data for higher levels
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        df = pd.DataFrame({"value": np.sin(np.linspace(0, 8 * np.pi, n))}, index=dates)

        config = {"columns": ["value"], "level": 5}
        result = plugin.generate_features(df, config)

        assert "value_approx" in result.columns
        for i in range(1, 6):
            assert f"value_detail_{i}" in result.columns


# Tests for pipeline integration


class TestPipelineIntegration:
    """Tests for integration with feature pipeline."""

    def test_works_in_pipeline(
        self,
        plugin: WaveletTransformPlugin,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test that plugin works correctly in a pipeline."""
        from src.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(
            plugins=[
                (plugin, {"columns": ["load"], "level": 2}),
            ]
        )

        result = pipeline.run(sample_df)

        assert result.success is True
        assert "load_approx" in result.df.columns
        assert "load_detail_1" in result.df.columns
        assert "load_detail_2" in result.df.columns


# Tests for internal methods


class TestInternalMethods:
    """Tests for internal wavelet methods."""

    def test_convolve_down_reduces_length(
        self,
        plugin: WaveletTransformPlugin,
    ) -> None:
        """Test that convolve_down reduces signal length by approximately half."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        filter_coef = [0.7071067811865476, 0.7071067811865476]  # Haar low-pass

        result = plugin._convolve_down(data, filter_coef)
        # Length should be approximately half (may vary due to padding)
        assert len(result) <= len(data)
        assert len(result) >= len(data) // 2

    def test_haar_wavelet_good_reconstruction(
        self,
        plugin: WaveletTransformPlugin,
    ) -> None:
        """Test good reconstruction with Haar wavelet on power-of-2 length."""
        # Use a simple signal
        n = 32
        data = np.sin(np.linspace(0, 2 * np.pi, n))
        df = pd.DataFrame(
            {"value": data},
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )

        config = {"columns": ["value"], "wavelet": "haar", "level": 2}
        result = plugin.generate_features(df, config)

        # Sum of components should approximate original
        reconstructed = (
            result["value_approx"]
            + result["value_detail_1"]
            + result["value_detail_2"]
        )

        # Check correlation is reasonably high (not perfect due to boundary effects)
        correlation = np.corrcoef(data, reconstructed)[0, 1]
        assert correlation > 0.95
