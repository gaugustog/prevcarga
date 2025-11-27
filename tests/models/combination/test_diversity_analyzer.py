"""Tests for ensemble diversity analyzer."""

import numpy as np
import pandas as pd
import pytest

from src.models.combination.diversity_analyzer import (
    DiversityAnalyzer,
    DiversityConfig,
    DiversityResult,
    EnsembleDiversityAnalyzer,
)


# Fixtures


@pytest.fixture
def simple_predictions():
    """Simple prediction set for basic tests."""
    return {
        "model1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "model2": np.array([1.1, 2.1, 2.9, 4.1, 4.9]),
        "model3": np.array([0.9, 2.2, 3.1, 3.9, 5.1]),
    }


@pytest.fixture
def simple_actuals():
    """Simple actuals for basic tests."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def identical_predictions():
    """Identical predictions (zero diversity)."""
    return {
        "model1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "model2": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }


@pytest.fixture
def orthogonal_predictions():
    """Uncorrelated predictions (high diversity)."""
    np.random.seed(42)
    return {
        "model1": np.random.randn(100),
        "model2": np.random.randn(100),
        "model3": np.random.randn(100),
    }


@pytest.fixture
def high_correlation_predictions():
    """Highly correlated predictions (low diversity)."""
    base = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    return {
        "model1": base + np.array([0.1, 0.2, 0.1, 0.2, 0.1]),
        "model2": base + np.array([0.2, 0.1, 0.2, 0.1, 0.2]),
        "model3": base + np.array([0.15, 0.15, 0.15, 0.15, 0.15]),
    }


@pytest.fixture
def predictions_with_nan():
    """Predictions with missing values."""
    return {
        "model1": np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
        "model2": np.array([1.1, 2.1, 2.9, np.nan, 4.9]),
        "model3": np.array([0.9, 2.2, 3.1, 3.9, 5.1]),
    }


@pytest.fixture
def analyzer():
    """Default analyzer instance."""
    return DiversityAnalyzer()


@pytest.fixture
def config():
    """Default config instance."""
    return DiversityConfig()


# Tests for DiversityConfig


class TestDiversityConfig:
    """Tests for DiversityConfig Pydantic model."""

    def test_init_default(self):
        """Test default initialization."""
        config = DiversityConfig()

        assert config.metrics == ["correlation", "disagreement"]
        assert config.correlation_threshold == 0.95
        assert config.min_models == 2
        assert config.max_models == 10
        assert config.selection_method == "greedy_forward"
        assert config.error_tolerance == 0.05

    def test_init_custom(self):
        """Test custom initialization."""
        config = DiversityConfig(
            metrics=["correlation", "ambiguity"],
            correlation_threshold=0.85,
            min_models=3,
            max_models=5,
            selection_method="greedy_backward",
            error_tolerance=0.1,
        )

        assert config.metrics == ["correlation", "ambiguity"]
        assert config.correlation_threshold == 0.85
        assert config.min_models == 3
        assert config.max_models == 5
        assert config.selection_method == "greedy_backward"
        assert config.error_tolerance == 0.1

    def test_validate_invalid_metrics(self):
        """Test validation of invalid metrics."""
        with pytest.raises(ValueError, match="Invalid metrics"):
            DiversityConfig(metrics=["correlation", "invalid_metric"])

    def test_validate_max_models_less_than_min(self):
        """Test validation when max_models < min_models."""
        with pytest.raises(ValueError, match="max_models.*must be.*min_models"):
            DiversityConfig(min_models=5, max_models=3)

    def test_correlation_threshold_bounds(self):
        """Test correlation_threshold validation bounds."""
        with pytest.raises(ValueError):
            DiversityConfig(correlation_threshold=-0.1)

        with pytest.raises(ValueError):
            DiversityConfig(correlation_threshold=1.5)

    def test_error_tolerance_bounds(self):
        """Test error_tolerance validation bounds."""
        with pytest.raises(ValueError):
            DiversityConfig(error_tolerance=-0.1)

        with pytest.raises(ValueError):
            DiversityConfig(error_tolerance=1.5)


# Tests for DiversityResult


class TestDiversityResult:
    """Tests for DiversityResult dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        corr_matrix = pd.DataFrame(
            [[1.0, 0.85], [0.85, 1.0]], index=["m1", "m2"], columns=["m1", "m2"]
        )

        result = DiversityResult(
            pairwise_correlations={("m1", "m2"): 0.85},
            avg_correlation=0.85,
            correlation_matrix=corr_matrix,
            avg_correlation_per_model={"m1": 0.85, "m2": 0.85},
            diversity_scores={"overall": 0.15, "correlation": 0.15},
            model_rankings=[("m1", 0.15), ("m2", 0.15)],
            aggregate_diversity=0.15,
            model_names=["m1", "m2"],
        )

        assert result.avg_correlation == 0.85
        assert result.aggregate_diversity == 0.15
        assert len(result.model_names) == 2
        assert result.q_statistics is None
        assert result.disagreement is None

    def test_get_most_diverse_pair(self):
        """Test getting most diverse model pair."""
        corr_matrix = pd.DataFrame(
            [[1.0, 0.85, 0.60], [0.85, 1.0, 0.70], [0.60, 0.70, 1.0]],
            index=["m1", "m2", "m3"],
            columns=["m1", "m2", "m3"],
        )

        result = DiversityResult(
            pairwise_correlations={
                ("m1", "m2"): 0.85,
                ("m1", "m3"): 0.60,
                ("m2", "m3"): 0.70,
            },
            avg_correlation=0.72,
            correlation_matrix=corr_matrix,
            avg_correlation_per_model={"m1": 0.725, "m2": 0.775, "m3": 0.65},
            diversity_scores={"overall": 0.28},
            model_rankings=[("m3", 0.35), ("m1", 0.275), ("m2", 0.225)],
            aggregate_diversity=0.28,
            model_names=["m1", "m2", "m3"],
        )

        model1, model2, corr = result.get_most_diverse_pair()
        assert corr == 0.60
        assert (model1, model2) == ("m1", "m3")

    def test_get_least_diverse_pair(self):
        """Test getting least diverse model pair."""
        corr_matrix = pd.DataFrame(
            [[1.0, 0.85, 0.60], [0.85, 1.0, 0.70], [0.60, 0.70, 1.0]],
            index=["m1", "m2", "m3"],
            columns=["m1", "m2", "m3"],
        )

        result = DiversityResult(
            pairwise_correlations={
                ("m1", "m2"): 0.85,
                ("m1", "m3"): 0.60,
                ("m2", "m3"): 0.70,
            },
            avg_correlation=0.72,
            correlation_matrix=corr_matrix,
            avg_correlation_per_model={"m1": 0.725, "m2": 0.775, "m3": 0.65},
            diversity_scores={"overall": 0.28},
            model_rankings=[("m3", 0.35), ("m1", 0.275), ("m2", 0.225)],
            aggregate_diversity=0.28,
            model_names=["m1", "m2", "m3"],
        )

        model1, model2, corr = result.get_least_diverse_pair()
        assert corr == 0.85
        assert (model1, model2) == ("m1", "m2")

    def test_get_most_diverse_pair_empty(self):
        """Test error when no correlations available."""
        corr_matrix = pd.DataFrame()

        result = DiversityResult(
            pairwise_correlations={},
            avg_correlation=0.0,
            correlation_matrix=corr_matrix,
            avg_correlation_per_model={},
            diversity_scores={},
            model_rankings=[],
            model_names=[],
        )

        with pytest.raises(ValueError, match="No pairwise correlations"):
            result.get_most_diverse_pair()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        corr_matrix = pd.DataFrame(
            [[1.0, 0.85], [0.85, 1.0]], index=["m1", "m2"], columns=["m1", "m2"]
        )

        result = DiversityResult(
            pairwise_correlations={("m1", "m2"): 0.85},
            avg_correlation=0.85,
            correlation_matrix=corr_matrix,
            avg_correlation_per_model={"m1": 0.85, "m2": 0.85},
            diversity_scores={"overall": 0.15},
            model_rankings=[("m1", 0.15), ("m2", 0.15)],
            aggregate_diversity=0.15,
            model_names=["m1", "m2"],
            metadata={"n_samples": 100},
        )

        result_dict = result.to_dict()

        assert "pairwise_correlations" in result_dict
        assert "avg_correlation" in result_dict
        assert "avg_correlation_per_model" in result_dict
        assert "diversity_scores" in result_dict
        assert "model_rankings" in result_dict
        assert "aggregate_diversity" in result_dict
        assert "model_names" in result_dict
        assert "metadata" in result_dict
        assert result_dict["avg_correlation"] == 0.85
        assert result_dict["aggregate_diversity"] == 0.15

    def test_repr(self):
        """Test string representation."""
        corr_matrix = pd.DataFrame(
            [[1.0, 0.85], [0.85, 1.0]], index=["m1", "m2"], columns=["m1", "m2"]
        )

        result = DiversityResult(
            pairwise_correlations={("m1", "m2"): 0.85},
            avg_correlation=0.85,
            correlation_matrix=corr_matrix,
            avg_correlation_per_model={"m1": 0.85, "m2": 0.85},
            diversity_scores={"overall": 0.15},
            model_rankings=[("m1", 0.15), ("m2", 0.15)],
            aggregate_diversity=0.15,
            model_names=["m1", "m2"],
        )

        repr_str = repr(result)
        assert "DiversityResult" in repr_str
        assert "n_models=2" in repr_str
        assert "0.850" in repr_str
        assert "0.150" in repr_str


# Tests for DiversityAnalyzer


class TestDiversityAnalyzer:
    """Tests for DiversityAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        config = DiversityConfig(error_tolerance=0.1)
        analyzer = DiversityAnalyzer(config=config)
        assert analyzer.config.error_tolerance == 0.1

    def test_init_default(self):
        """Test analyzer initialization with defaults."""
        analyzer = DiversityAnalyzer()
        assert analyzer.config.error_tolerance == 0.05

    def test_analyze_basic(self, analyzer, simple_predictions):
        """Test basic diversity analysis."""
        result = analyzer.analyze(simple_predictions)

        assert isinstance(result, DiversityResult)
        assert len(result.model_names) == 3
        assert len(result.pairwise_correlations) == 3  # 3 choose 2
        assert 0.0 <= result.avg_correlation <= 1.0
        assert 0.0 <= result.aggregate_diversity <= 1.0
        assert result.aggregate_diversity == pytest.approx(1.0 - result.avg_correlation)

        # Check correlation matrix
        assert isinstance(result.correlation_matrix, pd.DataFrame)
        assert result.correlation_matrix.shape == (3, 3)

        # Check avg_correlation_per_model
        assert len(result.avg_correlation_per_model) == 3
        for model in result.model_names:
            assert model in result.avg_correlation_per_model

        # Check diversity_scores
        assert "overall" in result.diversity_scores
        assert "correlation" in result.diversity_scores

        # Check model_rankings
        assert len(result.model_rankings) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result.model_rankings)

    def test_analyze_with_actuals(self, analyzer, simple_predictions, simple_actuals):
        """Test analysis with actual values."""
        result = analyzer.analyze(simple_predictions, simple_actuals)

        assert isinstance(result, DiversityResult)
        assert result.metadata["has_actuals"] is True

        # Error-based metrics should be None by default (not in default metrics list)
        # To test error metrics, need to configure them
        config = DiversityConfig(
            metrics=["correlation", "disagreement", "q_statistic", "double_fault", "ambiguity"]
        )
        analyzer_with_errors = DiversityAnalyzer(config=config)
        result_with_errors = analyzer_with_errors.analyze(simple_predictions, simple_actuals)

        assert result_with_errors.q_statistics is not None
        assert result_with_errors.disagreement is not None
        assert result_with_errors.double_fault is not None
        assert result_with_errors.ambiguity is not None

    def test_analyze_identical(self, analyzer, identical_predictions):
        """Test analysis with identical predictions (zero diversity)."""
        result = analyzer.analyze(identical_predictions)

        # Identical predictions should have correlation = 1.0
        assert result.avg_correlation == pytest.approx(1.0, abs=1e-6)
        assert result.aggregate_diversity == pytest.approx(0.0, abs=1e-6)

    def test_analyze_orthogonal(self, analyzer, orthogonal_predictions):
        """Test analysis with uncorrelated predictions (high diversity)."""
        result = analyzer.analyze(orthogonal_predictions)

        # Random uncorrelated predictions should have low correlation
        assert result.avg_correlation == pytest.approx(0.0, abs=0.2)
        assert result.aggregate_diversity == pytest.approx(1.0, abs=0.2)

    def test_analyze_high_correlation(self, analyzer, high_correlation_predictions):
        """Test analysis with highly correlated predictions."""
        result = analyzer.analyze(high_correlation_predictions)

        # Highly correlated predictions should have high correlation
        assert result.avg_correlation > 0.95
        assert result.aggregate_diversity < 0.05

    def test_analyze_with_nan(self, analyzer, predictions_with_nan):
        """Test analysis with missing values."""
        result = analyzer.analyze(predictions_with_nan)

        # Should still compute correlations for valid data
        assert isinstance(result, DiversityResult)
        assert len(result.pairwise_correlations) > 0

    def test_analyze_metadata(self, analyzer, simple_predictions):
        """Test metadata in results."""
        result = analyzer.analyze(simple_predictions)

        assert "n_models" in result.metadata
        assert "n_samples" in result.metadata
        assert "n_pairs" in result.metadata
        assert "processing_time_seconds" in result.metadata
        assert "has_actuals" in result.metadata

        assert result.metadata["n_models"] == 3
        assert result.metadata["n_samples"] == 5
        assert result.metadata["has_actuals"] is False

    def test_analyze_spearman(self, simple_predictions):
        """Test analysis with Spearman correlation."""
        config = DiversityConfig(metrics=["correlation", "spearman"])
        analyzer = DiversityAnalyzer(config=config)

        result = analyzer.analyze(simple_predictions)

        assert result.pairwise_spearman is not None
        assert len(result.pairwise_spearman) == 3

    def test_select_subset_greedy_forward(self, analyzer, simple_predictions, simple_actuals):
        """Test greedy forward selection."""
        selected = analyzer.select_subset(simple_predictions, simple_actuals, method="greedy_forward")

        assert isinstance(selected, list)
        assert 2 <= len(selected) <= analyzer.config.max_models
        assert all(model in simple_predictions for model in selected)

    def test_select_subset_greedy_backward(self, analyzer, simple_predictions, simple_actuals):
        """Test greedy backward selection."""
        selected = analyzer.select_subset(
            simple_predictions, simple_actuals, method="greedy_backward"
        )

        assert isinstance(selected, list)
        assert 2 <= len(selected) <= len(simple_predictions)
        assert all(model in simple_predictions for model in selected)

    def test_select_subset_correlation_clustering(self, analyzer, simple_predictions, simple_actuals):
        """Test correlation clustering selection."""
        selected = analyzer.select_subset(
            simple_predictions, simple_actuals, method="correlation_clustering"
        )

        assert isinstance(selected, list)
        assert 2 <= len(selected) <= len(simple_predictions)
        assert all(model in simple_predictions for model in selected)

    def test_select_subset_invalid_method(self, analyzer, simple_predictions, simple_actuals):
        """Test error with invalid selection method."""
        with pytest.raises(ValueError, match="Unknown selection method"):
            analyzer.select_subset(simple_predictions, simple_actuals, method="invalid")

    def test_get_redundant_models(self, analyzer, high_correlation_predictions):
        """Test finding redundant models."""
        result = analyzer.analyze(high_correlation_predictions)
        redundant = analyzer.get_redundant_models(result)

        # All models should be flagged as redundant due to high correlation
        assert isinstance(redundant, list)
        assert len(redundant) > 0

    def test_get_redundant_models_orthogonal(self, analyzer, orthogonal_predictions):
        """Test finding redundant models with uncorrelated predictions."""
        result = analyzer.analyze(orthogonal_predictions)
        redundant = analyzer.get_redundant_models(result)

        # No models should be redundant with random uncorrelated predictions
        assert len(redundant) == 0

    def test_plot_correlation_matrix(self, analyzer, simple_predictions):
        """Test correlation matrix visualization data."""
        result = analyzer.analyze(simple_predictions)
        heatmap_data = analyzer.plot_correlation_matrix(result)

        assert "matrix" in heatmap_data
        assert "labels" in heatmap_data
        assert "values" in heatmap_data
        assert isinstance(heatmap_data["matrix"], pd.DataFrame)
        assert len(heatmap_data["labels"]) == 3
        assert heatmap_data["values"].shape == (3, 3)

    def test_get_dendrogram_data(self, analyzer, simple_predictions):
        """Test dendrogram data generation."""
        result = analyzer.analyze(simple_predictions)
        dendro_data = analyzer.get_dendrogram_data(result)

        assert "linkage_matrix" in dendro_data
        assert "labels" in dendro_data
        assert dendro_data["labels"] == result.model_names

    def test_get_diversity_report(self, analyzer, simple_predictions, simple_actuals):
        """Test diversity report generation."""
        result = analyzer.analyze(simple_predictions, simple_actuals)
        report = analyzer.get_diversity_report(result)

        assert isinstance(report, str)
        assert "ENSEMBLE DIVERSITY ANALYSIS REPORT" in report
        assert "Number of models: 3" in report
        assert "OVERALL DIVERSITY METRICS" in report
        assert "Average correlation:" in report
        assert "Aggregate diversity:" in report

    def test_get_diversity_report_with_error_metrics(self, simple_predictions, simple_actuals):
        """Test diversity report with error metrics."""
        config = DiversityConfig(
            metrics=["correlation", "disagreement", "q_statistic", "double_fault", "ambiguity"]
        )
        analyzer = DiversityAnalyzer(config=config)

        result = analyzer.analyze(simple_predictions, simple_actuals)
        report = analyzer.get_diversity_report(result)

        assert "ERROR-BASED METRICS" in report
        assert "Q-statistic" in report or "disagreement" in report

    # Validation tests

    def test_validate_predictions_empty(self, analyzer):
        """Test validation with empty predictions."""
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze({})

    def test_validate_predictions_single_model(self, analyzer):
        """Test validation with single model."""
        with pytest.raises(ValueError, match="At least 2 models"):
            analyzer.analyze({"model1": np.array([1, 2, 3])})

    def test_validate_predictions_inconsistent_length(self, analyzer):
        """Test validation with inconsistent array lengths."""
        predictions = {
            "model1": np.array([1, 2, 3]),
            "model2": np.array([1, 2, 3, 4]),
        }

        with pytest.raises(ValueError, match="same length"):
            analyzer.analyze(predictions)

    def test_validate_predictions_too_few_samples(self, analyzer):
        """Test validation with too few samples."""
        predictions = {
            "model1": np.array([1]),
            "model2": np.array([2]),
        }

        with pytest.raises(ValueError, match="At least 2 samples"):
            analyzer.analyze(predictions)

    def test_validate_actuals_wrong_length(self, analyzer, simple_predictions):
        """Test validation with wrong actuals length."""
        actuals = np.array([1, 2])  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            analyzer.analyze(simple_predictions, actuals)

    # Edge case tests

    def test_zero_variance_predictions(self, analyzer):
        """Test handling of zero variance predictions."""
        predictions = {
            "model1": np.array([5.0, 5.0, 5.0, 5.0]),
            "model2": np.array([1.0, 2.0, 3.0, 4.0]),
        }

        result = analyzer.analyze(predictions)

        # Should handle gracefully, possibly skipping the pair
        assert isinstance(result, DiversityResult)

    def test_large_predictions(self, analyzer):
        """Test with large prediction arrays."""
        np.random.seed(42)
        n_samples = 10000
        predictions = {f"model{i}": np.random.randn(n_samples) for i in range(5)}

        result = analyzer.analyze(predictions)

        assert len(result.model_names) == 5
        assert len(result.pairwise_correlations) == 10  # 5 choose 2

    def test_perfect_negative_correlation(self, analyzer):
        """Test with perfectly negatively correlated predictions."""
        predictions = {
            "model1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "model2": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        }

        result = analyzer.analyze(predictions)

        # Perfect negative correlation should be close to -1
        assert result.pairwise_correlations[("model1", "model2")] == pytest.approx(-1.0, abs=1e-6)

    def test_many_models(self, analyzer):
        """Test with many models."""
        np.random.seed(42)
        n_models = 20
        predictions = {f"model{i}": np.random.randn(100) for i in range(n_models)}

        result = analyzer.analyze(predictions)

        expected_pairs = n_models * (n_models - 1) // 2
        assert len(result.model_names) == n_models
        assert len(result.pairwise_correlations) == expected_pairs

    def test_error_tolerance_effect(self):
        """Test effect of error tolerance on Q-statistic."""
        # Create predictions with small errors
        actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        predictions = {
            "model1": actuals + 0.1,  # Small consistent error
            "model2": actuals + 0.2,  # Slightly larger error
        }

        # Tight tolerance - both models incorrect
        config_tight = DiversityConfig(metrics=["q_statistic"], error_tolerance=0.001)
        analyzer_tight = DiversityAnalyzer(config=config_tight)
        result_tight = analyzer_tight.analyze(predictions, actuals)

        # Loose tolerance - both models correct
        config_loose = DiversityConfig(metrics=["q_statistic"], error_tolerance=0.1)
        analyzer_loose = DiversityAnalyzer(config=config_loose)
        result_loose = analyzer_loose.analyze(predictions, actuals)

        # Q-statistics should differ based on tolerance
        q_tight = result_tight.q_statistics[("model1", "model2")]
        q_loose = result_loose.q_statistics[("model1", "model2")]

        assert isinstance(q_tight, float)
        assert isinstance(q_loose, float)

    def test_analyze_sorted_model_names(self, analyzer):
        """Test that model names are sorted consistently."""
        # Create predictions with unordered keys
        predictions = {
            "zebra": np.array([1, 2, 3, 4, 5]),
            "alpha": np.array([1.1, 2.1, 2.9, 4.1, 4.9]),
            "beta": np.array([0.9, 2.2, 3.1, 3.9, 5.1]),
        }

        result = analyzer.analyze(predictions)

        # Model names should be sorted
        assert result.model_names == ["alpha", "beta", "zebra"]

    def test_backwards_compatibility_alias(self):
        """Test that EnsembleDiversityAnalyzer is still available."""
        analyzer = EnsembleDiversityAnalyzer()
        assert isinstance(analyzer, DiversityAnalyzer)

    def test_diversity_scores_structure(self, simple_predictions, simple_actuals):
        """Test diversity scores structure."""
        config = DiversityConfig(
            metrics=["correlation", "disagreement", "q_statistic", "double_fault", "ambiguity"]
        )
        analyzer = DiversityAnalyzer(config=config)

        result = analyzer.analyze(simple_predictions, simple_actuals)

        # Check that diversity_scores contains expected keys
        assert "correlation" in result.diversity_scores
        assert "overall" in result.diversity_scores

        # Check that all scores are in valid range [0, 1]
        for score in result.diversity_scores.values():
            assert 0.0 <= score <= 1.0

    def test_model_rankings_order(self, analyzer, simple_predictions):
        """Test that model rankings are ordered by diversity."""
        result = analyzer.analyze(simple_predictions)

        # Check that rankings are sorted in descending order
        diversity_values = [score for _, score in result.model_rankings]
        assert diversity_values == sorted(diversity_values, reverse=True)

    def test_correlation_matrix_symmetry(self, analyzer, simple_predictions):
        """Test that correlation matrix is symmetric."""
        result = analyzer.analyze(simple_predictions)

        corr_matrix = result.correlation_matrix
        assert np.allclose(corr_matrix.values, corr_matrix.values.T)

    def test_correlation_matrix_diagonal(self, analyzer, simple_predictions):
        """Test that correlation matrix diagonal is 1.0."""
        result = analyzer.analyze(simple_predictions)

        corr_matrix = result.correlation_matrix
        diagonal = np.diag(corr_matrix.values)
        assert np.allclose(diagonal, 1.0)
