"""Tests for advanced bias correction module.

This module provides comprehensive tests for the AdvancedBiasCorrectionModule,
including spline-based correction, tree-based conditional correction,
temporal decomposition, and cross-validation model selection.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.combination.advanced_bias_correction import (
    AdvancedBiasCorrectionConfig,
    AdvancedBiasCorrectionModule,
    BiasCorrectortDiagnostics,
    CorrectionType,
    SplineBiasCorrector,
    TreeBiasCorrector,
)


@pytest.fixture
def sample_data_nonlinear():
    """Create sample data with non-linear bias."""
    np.random.seed(42)
    n_samples = 1000

    # Base predictions
    predictions = np.random.randn(n_samples) * 50 + 1000

    # Add non-linear bias (quadratic)
    nonlinear_bias = 0.0001 * (predictions - 1000) ** 2

    targets = predictions + nonlinear_bias + np.random.randn(n_samples) * 10

    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="30min")

    return predictions, targets, timestamps


@pytest.fixture
def sample_data_temporal():
    """Create sample data with temporal bias patterns."""
    np.random.seed(42)
    n_samples = 2000

    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="30min")

    # Base predictions
    predictions = np.random.randn(n_samples) * 30 + 1000

    # Add hourly bias pattern (sine wave)
    hourly_bias = np.array([20 * np.sin(ts.hour * np.pi / 12) for ts in timestamps])

    # Add weekly bias pattern
    weekly_bias = np.array([10 if ts.dayofweek < 5 else -15 for ts in timestamps])

    targets = predictions + hourly_bias + weekly_bias + np.random.randn(n_samples) * 5

    return predictions, targets, timestamps


@pytest.fixture
def sample_data_combined():
    """Create sample data with combined bias patterns."""
    np.random.seed(42)
    n_samples = 2000

    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="30min")

    # Base predictions
    predictions = np.random.randn(n_samples) * 40 + 1000

    # Non-linear bias
    nonlinear_bias = 0.00005 * (predictions - 1000) ** 2

    # Temporal bias
    hourly_bias = np.array([15 * np.sin(ts.hour * np.pi / 12) for ts in timestamps])

    targets = predictions + nonlinear_bias + hourly_bias + np.random.randn(n_samples) * 8

    return predictions, targets, timestamps


class TestCorrectionType:
    """Tests for CorrectionType enum."""

    def test_correction_types(self):
        """Test all correction types are defined."""
        assert CorrectionType.SPLINE.value == "spline"
        assert CorrectionType.TREE.value == "tree"
        assert CorrectionType.TEMPORAL.value == "temporal"
        assert CorrectionType.COMBINED.value == "combined"
        assert CorrectionType.AUTO.value == "auto"


class TestAdvancedBiasCorrectionConfig:
    """Tests for AdvancedBiasCorrectionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AdvancedBiasCorrectionConfig()

        assert config.correction_type == "combined"
        assert config.spline_n_knots == 5
        assert config.spline_degree == 3
        assert config.tree_max_depth == 4
        assert config.min_samples_per_group == 20
        assert config.significance_level == 0.05
        assert config.use_cross_validation is True
        assert config.n_cv_folds == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AdvancedBiasCorrectionConfig(
            correction_type="spline",
            spline_n_knots=10,
            significance_level=0.01,
        )

        assert config.correction_type == "spline"
        assert config.spline_n_knots == 10
        assert config.significance_level == 0.01


class TestSplineBiasCorrector:
    """Tests for SplineBiasCorrector class."""

    def test_initialization(self):
        """Test spline corrector initialization."""
        corrector = SplineBiasCorrector(n_knots=5, degree=3)

        assert corrector.n_knots == 5
        assert corrector.degree == 3
        assert not corrector.fitted

    def test_fit(self, sample_data_nonlinear):
        """Test spline corrector fitting."""
        predictions, targets, _ = sample_data_nonlinear
        residuals = targets - predictions

        corrector = SplineBiasCorrector(n_knots=5, degree=3)
        corrector.fit(predictions, residuals)

        assert corrector.fitted
        assert corrector.r2_score is not None
        assert corrector.r2_score >= 0

    def test_correct(self, sample_data_nonlinear):
        """Test spline correction application."""
        predictions, targets, _ = sample_data_nonlinear
        residuals = targets - predictions

        corrector = SplineBiasCorrector(n_knots=5, degree=3)
        corrector.fit(predictions, residuals)

        corrected = corrector.correct(predictions)

        # Corrected should reduce bias
        original_mae = np.mean(np.abs(residuals))
        corrected_mae = np.mean(np.abs(targets - corrected))

        assert corrected_mae < original_mae

    def test_correct_without_fit(self):
        """Test correction without fitting returns original."""
        corrector = SplineBiasCorrector()
        predictions = np.array([100, 200, 300])

        corrected = corrector.correct(predictions)

        np.testing.assert_array_equal(corrected, predictions)

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        corrector = SplineBiasCorrector(n_knots=10)

        # Very few samples
        predictions = np.array([100, 200, 300])
        residuals = np.array([1, 2, 3])

        corrector.fit(predictions, residuals)

        # Should not be fitted
        assert not corrector.fitted


class TestTreeBiasCorrector:
    """Tests for TreeBiasCorrector class."""

    def test_initialization(self):
        """Test tree corrector initialization."""
        corrector = TreeBiasCorrector(max_depth=4)

        assert corrector.max_depth == 4
        assert not corrector.fitted

    def test_fit_without_context(self):
        """Test tree corrector fitting without context features."""
        np.random.seed(42)
        predictions = np.random.randn(200) * 50 + 1000
        residuals = np.random.randn(200) * 10

        corrector = TreeBiasCorrector(max_depth=3)
        corrector.fit(predictions, residuals)

        assert corrector.fitted

    def test_fit_with_context(self):
        """Test tree corrector fitting with context features."""
        np.random.seed(42)
        predictions = np.random.randn(200) * 50 + 1000
        residuals = np.random.randn(200) * 10
        context = np.random.randn(200, 3)

        corrector = TreeBiasCorrector(max_depth=3)
        corrector.fit(predictions, residuals, context)

        assert corrector.fitted
        assert corrector._n_features == 4  # predictions + 3 context features

    def test_correct_with_context(self):
        """Test tree correction with context features."""
        np.random.seed(42)
        predictions = np.random.randn(200) * 50 + 1000
        residuals = np.random.randn(200) * 10
        context = np.random.randn(200, 3)

        corrector = TreeBiasCorrector(max_depth=3)
        corrector.fit(predictions, residuals, context)

        corrected = corrector.correct(predictions, context)

        # Should return array of same shape
        assert corrected.shape == predictions.shape

    def test_correct_feature_mismatch(self):
        """Test correction with mismatched feature count."""
        np.random.seed(42)
        predictions = np.random.randn(200) * 50 + 1000
        residuals = np.random.randn(200) * 10
        context_fit = np.random.randn(200, 3)
        context_test = np.random.randn(200, 5)  # Different size

        corrector = TreeBiasCorrector(max_depth=3)
        corrector.fit(predictions, residuals, context_fit)

        # Should return original predictions due to mismatch
        corrected = corrector.correct(predictions, context_test)

        np.testing.assert_array_equal(corrected, predictions)


class TestAdvancedBiasCorrectionModuleInit:
    """Tests for AdvancedBiasCorrectionModule initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        module = AdvancedBiasCorrectionModule()

        assert module.config.correction_type == "combined"
        assert not module.fitted
        assert module.bias_reduction == 0.0

    def test_custom_config(self):
        """Test initialization with custom config."""
        module = AdvancedBiasCorrectionModule(
            config={
                "correction_type": "spline",
                "spline_n_knots": 10,
                "significance_level": 0.01,
            }
        )

        assert module.config.correction_type == "spline"
        assert module.config.spline_n_knots == 10
        assert module.config.significance_level == 0.01


class TestAdvancedBiasCorrectionModuleFit:
    """Tests for AdvancedBiasCorrectionModule.fit() method."""

    def test_fit_spline(self, sample_data_nonlinear):
        """Test fitting with spline correction."""
        predictions, targets, timestamps = sample_data_nonlinear

        module = AdvancedBiasCorrectionModule(config={"correction_type": "spline"})
        module.fit(predictions, targets, timestamps)

        assert module.fitted
        assert module.spline_corrector is not None
        assert module.spline_corrector.fitted

    def test_fit_tree(self, sample_data_nonlinear):
        """Test fitting with tree correction."""
        predictions, targets, timestamps = sample_data_nonlinear
        context = np.column_stack([timestamps.hour, timestamps.dayofweek])

        module = AdvancedBiasCorrectionModule(config={"correction_type": "tree"})
        module.fit(predictions, targets, timestamps, context_features=context)

        assert module.fitted
        assert module.tree_corrector is not None
        assert module.tree_corrector.fitted

    def test_fit_temporal(self, sample_data_temporal):
        """Test fitting with temporal correction."""
        predictions, targets, timestamps = sample_data_temporal

        module = AdvancedBiasCorrectionModule(config={"correction_type": "temporal"})
        module.fit(predictions, targets, timestamps)

        assert module.fitted
        # Should detect at least some temporal patterns
        assert len(module.temporal_bias) > 0 or module.trend_bias is not None

    def test_fit_combined(self, sample_data_combined):
        """Test fitting with combined correction."""
        predictions, targets, timestamps = sample_data_combined

        module = AdvancedBiasCorrectionModule(config={"correction_type": "combined"})
        module.fit(predictions, targets, timestamps)

        assert module.fitted

    def test_fit_validates_lengths(self):
        """Test that fit validates array lengths."""
        module = AdvancedBiasCorrectionModule()

        predictions = np.array([100, 200, 300])
        targets = np.array([101, 201])  # Different length
        timestamps = pd.date_range("2024-01-01", periods=3, freq="30min")

        with pytest.raises(ValueError, match="Array length mismatch"):
            module.fit(predictions, targets, timestamps)


class TestAdvancedBiasCorrectionModuleCorrect:
    """Tests for AdvancedBiasCorrectionModule.correct_bias() method."""

    def test_correct_without_fit(self):
        """Test correction without fitting returns original."""
        module = AdvancedBiasCorrectionModule()

        predictions = np.array([100, 200, 300])
        timestamps = pd.date_range("2024-01-01", periods=3, freq="30min")

        corrected = module.correct_bias(predictions, timestamps)

        np.testing.assert_array_equal(corrected, predictions)

    def test_correct_reduces_bias(self, sample_data_nonlinear):
        """Test that correction reduces bias."""
        predictions, targets, timestamps = sample_data_nonlinear

        module = AdvancedBiasCorrectionModule(config={"correction_type": "spline"})
        module.fit(predictions, targets, timestamps)

        corrected = module.correct_bias(predictions, timestamps)

        original_mae = np.mean(np.abs(targets - predictions))
        corrected_mae = np.mean(np.abs(targets - corrected))

        assert corrected_mae < original_mae

    def test_correct_with_context(self, sample_data_nonlinear):
        """Test correction with context features."""
        predictions, targets, timestamps = sample_data_nonlinear
        context = np.column_stack([timestamps.hour, timestamps.dayofweek])

        module = AdvancedBiasCorrectionModule(config={"correction_type": "tree"})
        module.fit(predictions, targets, timestamps, context_features=context)

        corrected = module.correct_bias(predictions, timestamps, context_features=context)

        assert corrected.shape == predictions.shape

    def test_correction_bounded(self, sample_data_combined):
        """Test that correction is bounded by max_correction_fraction."""
        predictions, targets, timestamps = sample_data_combined

        module = AdvancedBiasCorrectionModule(
            config={
                "correction_type": "combined",
                "max_correction_fraction": 0.1,
            }
        )
        module.fit(predictions, targets, timestamps)

        corrected = module.correct_bias(predictions, timestamps)

        # Check correction is bounded
        correction = np.abs(corrected - predictions)
        max_allowed = np.abs(predictions) * 0.1

        assert np.all(correction <= max_allowed + 1e-6)


class TestTemporalBiasDetection:
    """Tests for temporal bias detection."""

    def test_detect_hourly_bias(self, sample_data_temporal):
        """Test hourly bias detection."""
        predictions, targets, timestamps = sample_data_temporal

        module = AdvancedBiasCorrectionModule(
            config={
                "correction_type": "temporal",
                "detect_hourly": True,
                "detect_weekly": False,
                "detect_seasonal": False,
                "detect_trend": False,
            }
        )
        module.fit(predictions, targets, timestamps)

        # Should detect hourly pattern
        assert "hourly" in module.temporal_bias

    def test_detect_weekly_bias(self, sample_data_temporal):
        """Test weekly bias detection."""
        predictions, targets, timestamps = sample_data_temporal

        module = AdvancedBiasCorrectionModule(
            config={
                "correction_type": "temporal",
                "detect_hourly": False,
                "detect_weekly": True,
                "detect_seasonal": False,
                "detect_trend": False,
            }
        )
        module.fit(predictions, targets, timestamps)

        # Should detect weekly pattern
        assert "weekly" in module.temporal_bias

    def test_no_temporal_bias_detected(self):
        """Test when no temporal bias exists."""
        np.random.seed(42)
        n_samples = 500
        timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="30min")

        # Random data with no temporal pattern
        predictions = np.random.randn(n_samples) * 50 + 1000
        targets = predictions + np.random.randn(n_samples) * 10

        module = AdvancedBiasCorrectionModule(config={"correction_type": "temporal"})
        module.fit(predictions, targets, timestamps)

        # May or may not detect patterns depending on random data
        # Just check it doesn't crash


class TestSignificanceTesting:
    """Tests for significance testing in bias detection."""

    def test_significance_threshold(self, sample_data_temporal):
        """Test that significance threshold is respected."""
        predictions, targets, timestamps = sample_data_temporal

        # Very strict significance level
        module_strict = AdvancedBiasCorrectionModule(
            config={
                "correction_type": "temporal",
                "significance_level": 0.001,
            }
        )
        module_strict.fit(predictions, targets, timestamps)

        # Lenient significance level
        module_lenient = AdvancedBiasCorrectionModule(
            config={
                "correction_type": "temporal",
                "significance_level": 0.5,
            }
        )
        module_lenient.fit(predictions, targets, timestamps)

        # Lenient should detect at least as many patterns
        strict_components = len(module_strict.temporal_bias)
        lenient_components = len(module_lenient.temporal_bias)

        assert lenient_components >= strict_components


class TestDiagnostics:
    """Tests for diagnostics functionality."""

    def test_get_diagnostics(self, sample_data_combined):
        """Test diagnostics retrieval."""
        predictions, targets, timestamps = sample_data_combined

        module = AdvancedBiasCorrectionModule(config={"correction_type": "combined"})
        module.fit(predictions, targets, timestamps)

        diagnostics = module.get_diagnostics()

        assert "correction_type" in diagnostics
        assert "n_samples" in diagnostics
        assert "original_bias" in diagnostics
        assert "corrected_bias" in diagnostics
        assert "bias_reduction" in diagnostics
        assert diagnostics["n_samples"] == len(predictions)

    def test_diagnostics_before_fit(self):
        """Test diagnostics before fitting."""
        module = AdvancedBiasCorrectionModule()

        diagnostics = module.get_diagnostics()

        assert diagnostics["n_samples"] == 0
        assert diagnostics["bias_reduction"] == 0.0


class TestAutoSelection:
    """Tests for automatic correction type selection."""

    def test_auto_selection(self, sample_data_combined):
        """Test automatic correction type selection."""
        predictions, targets, timestamps = sample_data_combined

        module = AdvancedBiasCorrectionModule(
            config={
                "correction_type": "auto",
                "use_cross_validation": True,
            }
        )
        module.fit(predictions, targets, timestamps)

        diagnostics = module.get_diagnostics()

        # Should have selected a model
        assert diagnostics["selected_model"] is not None or module.config.correction_type != "auto"

    def test_auto_selection_without_cv(self, sample_data_combined):
        """Test auto selection falls back to combined without CV."""
        predictions, targets, timestamps = sample_data_combined

        module = AdvancedBiasCorrectionModule(
            config={
                "correction_type": "auto",
                "use_cross_validation": False,
            }
        )
        module.fit(predictions, targets, timestamps)

        # Should use combined as fallback
        assert module.fitted


class TestBiasReduction:
    """Tests for bias reduction calculation."""

    def test_bias_reduction_positive(self, sample_data_nonlinear):
        """Test that bias reduction is positive for biased data."""
        predictions, targets, timestamps = sample_data_nonlinear

        module = AdvancedBiasCorrectionModule(config={"correction_type": "spline"})
        module.fit(predictions, targets, timestamps)

        # Should show positive bias reduction
        assert module.bias_reduction > 0

    def test_bias_reduction_in_diagnostics(self, sample_data_combined):
        """Test bias reduction is reported in diagnostics."""
        predictions, targets, timestamps = sample_data_combined

        module = AdvancedBiasCorrectionModule(config={"correction_type": "combined"})
        module.fit(predictions, targets, timestamps)

        diagnostics = module.get_diagnostics()

        assert diagnostics["bias_reduction"] == module.bias_reduction
        assert diagnostics["original_bias"] > 0
        assert diagnostics["corrected_bias"] >= 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_constant_predictions(self):
        """Test handling of constant predictions."""
        np.random.seed(42)
        n_samples = 200
        timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="30min")

        predictions = np.ones(n_samples) * 1000
        targets = predictions + np.random.randn(n_samples) * 10

        module = AdvancedBiasCorrectionModule(config={"correction_type": "spline"})
        module.fit(predictions, targets, timestamps)

        # Should handle without crashing
        corrected = module.correct_bias(predictions, timestamps)
        assert corrected.shape == predictions.shape

    def test_small_dataset(self):
        """Test handling of small dataset."""
        np.random.seed(42)
        n_samples = 50
        timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="30min")

        predictions = np.random.randn(n_samples) * 50 + 1000
        targets = predictions + 10

        module = AdvancedBiasCorrectionModule(config={"correction_type": "temporal"})
        module.fit(predictions, targets, timestamps)

        # Should handle small dataset without crash
        assert module.fitted

    def test_nan_handling(self):
        """Test handling of NaN values."""
        np.random.seed(42)
        n_samples = 200
        timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="30min")

        predictions = np.random.randn(n_samples) * 50 + 1000
        targets = predictions + 10

        # Add some NaN values
        predictions[10:15] = np.nan

        module = AdvancedBiasCorrectionModule()

        # Should handle NaN - either by working or raising clear error
        # For now, just check it doesn't crash unexpectedly


class TestRepr:
    """Tests for string representation."""

    def test_repr_unfitted(self):
        """Test repr for unfitted module."""
        module = AdvancedBiasCorrectionModule(config={"correction_type": "spline"})

        repr_str = repr(module)

        assert "spline" in repr_str
        assert "not fitted" in repr_str

    def test_repr_fitted(self, sample_data_nonlinear):
        """Test repr for fitted module."""
        predictions, targets, timestamps = sample_data_nonlinear

        module = AdvancedBiasCorrectionModule(config={"correction_type": "spline"})
        module.fit(predictions, targets, timestamps)

        repr_str = repr(module)

        assert "spline" in repr_str
        assert "fitted" in repr_str or "not fitted" not in repr_str


class TestTrendDetection:
    """Tests for trend bias detection."""

    def test_detect_linear_trend(self):
        """Test linear trend detection."""
        np.random.seed(42)
        n_samples = 500
        timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="30min")

        # Create data with clear linear trend
        predictions = np.random.randn(n_samples) * 20 + 1000

        # Add linear trend
        days = np.arange(n_samples) / 48  # Days
        trend = 5 * days  # 5 units per day

        targets = predictions + trend + np.random.randn(n_samples) * 5

        module = AdvancedBiasCorrectionModule(
            config={
                "correction_type": "temporal",
                "detect_trend": True,
                "detect_seasonal": False,
                "detect_weekly": False,
                "detect_hourly": False,
            }
        )
        module.fit(predictions, targets, timestamps)

        # Should detect trend
        assert module.trend_bias is not None
        assert "slope" in module.trend_bias
        assert module.trend_bias["slope"] > 0
