"""Tests for model comparison framework.

This module tests the ModelComparator class and associated components
for comparing and ranking multiple forecast models.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.model_comparator import (
    ComparisonConfig,
    ComparisonMetric,
    ComparisonResult,
    ModelComparator,
    ModelRanking,
    PairwiseComparison,
    RankingMethod,
)


class TestComparisonConfig:
    """Tests for ComparisonConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ComparisonConfig()

        assert "mape" in config.metrics
        assert "mae" in config.metrics
        assert "rmse" in config.metrics
        assert config.primary_metric == "mape"
        assert config.ranking_method == "single_metric"
        assert config.significance_level == 0.05

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ComparisonConfig(
            metrics=["rmse", "r2"],
            primary_metric="rmse",
            significance_level=0.01,
            ranking_method="weighted_average",
        )

        assert config.metrics == ["rmse", "r2"]
        assert config.primary_metric == "rmse"
        assert config.significance_level == 0.01
        assert config.ranking_method == "weighted_average"

    def test_invalid_metric(self) -> None:
        """Test validation of metrics."""
        with pytest.raises(ValueError, match="Invalid metric"):
            ComparisonConfig(metrics=["invalid_metric"])

    def test_invalid_primary_metric(self) -> None:
        """Test validation of primary metric."""
        with pytest.raises(ValueError, match="Invalid primary_metric"):
            ComparisonConfig(primary_metric="invalid")

    def test_invalid_significance_level(self) -> None:
        """Test validation of significance level."""
        with pytest.raises(ValueError, match="significance_level"):
            ComparisonConfig(significance_level=0.0)

        with pytest.raises(ValueError, match="significance_level"):
            ComparisonConfig(significance_level=1.0)

    def test_invalid_ranking_method(self) -> None:
        """Test validation of ranking method."""
        with pytest.raises(ValueError, match="Invalid ranking_method"):
            ComparisonConfig(ranking_method="invalid")

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = ComparisonConfig()
        config_dict = config.to_dict()

        assert "metrics" in config_dict
        assert "primary_metric" in config_dict
        assert "significance_level" in config_dict


class TestModelRanking:
    """Tests for ModelRanking dataclass."""

    def test_basic_ranking(self) -> None:
        """Test creating a basic ranking."""
        ranking = ModelRanking(
            model_name="lgbm",
            rank=1,
            metrics={"mape": 5.2, "rmse": 150.0},
            score=0.95,
        )

        assert ranking.model_name == "lgbm"
        assert ranking.rank == 1
        assert ranking.metrics["mape"] == 5.2
        assert ranking.score == 0.95

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        ranking = ModelRanking(
            model_name="lgbm",
            rank=1,
            metrics={"mape": 5.2},
            confidence_intervals={"mape": (4.5, 5.9)},
        )
        ranking_dict = ranking.to_dict()

        assert ranking_dict["model_name"] == "lgbm"
        assert ranking_dict["rank"] == 1
        assert "metrics" in ranking_dict
        assert "confidence_intervals" in ranking_dict

    def test_repr(self) -> None:
        """Test string representation."""
        ranking = ModelRanking(
            model_name="lgbm",
            rank=1,
            metrics={"mape": 5.2, "rmse": 150.0},
        )
        repr_str = repr(ranking)

        assert "lgbm" in repr_str
        assert "rank=1" in repr_str


class TestPairwiseComparison:
    """Tests for PairwiseComparison dataclass."""

    def test_basic_comparison(self) -> None:
        """Test creating a basic pairwise comparison."""
        comparison = PairwiseComparison(
            model_a="lgbm",
            model_b="rf",
            metric="mape",
            mean_diff=-0.5,
            t_p_value=0.02,
            is_significant=True,
            better_model="lgbm",
        )

        assert comparison.model_a == "lgbm"
        assert comparison.model_b == "rf"
        assert comparison.is_significant is True
        assert comparison.better_model == "lgbm"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        comparison = PairwiseComparison(
            model_a="lgbm",
            model_b="rf",
            metric="mape",
            mean_diff=-0.5,
        )
        comp_dict = comparison.to_dict()

        assert comp_dict["model_a"] == "lgbm"
        assert comp_dict["model_b"] == "rf"
        assert "mean_diff" in comp_dict

    def test_repr(self) -> None:
        """Test string representation."""
        comparison = PairwiseComparison(
            model_a="lgbm",
            model_b="rf",
            metric="mape",
            mean_diff=-0.5,
            is_significant=True,
            better_model="lgbm",
        )
        repr_str = repr(comparison)

        assert "lgbm" in repr_str
        assert "rf" in repr_str
        assert "SIGNIFICANT" in repr_str


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_basic_result(self) -> None:
        """Test creating a basic result."""
        ranking1 = ModelRanking(model_name="lgbm", rank=1, metrics={"mape": 5.0})
        ranking2 = ModelRanking(model_name="rf", rank=2, metrics={"mape": 6.0})

        result = ComparisonResult(
            rankings=[ranking1, ranking2],
            best_model="lgbm",
        )

        assert result.best_model == "lgbm"
        assert result.num_models == 2
        assert result.model_names == ["lgbm", "rf"]

    def test_get_ranking(self) -> None:
        """Test getting ranking for a specific model."""
        ranking1 = ModelRanking(model_name="lgbm", rank=1, metrics={"mape": 5.0})
        ranking2 = ModelRanking(model_name="rf", rank=2, metrics={"mape": 6.0})

        result = ComparisonResult(
            rankings=[ranking1, ranking2],
            best_model="lgbm",
        )

        ranking = result.get_ranking("lgbm")
        assert ranking is not None
        assert ranking.rank == 1

        ranking = result.get_ranking("nonexistent")
        assert ranking is None

    def test_get_comparison(self) -> None:
        """Test getting pairwise comparison."""
        comparison = PairwiseComparison(
            model_a="lgbm",
            model_b="rf",
            metric="mape",
            mean_diff=-0.5,
        )

        result = ComparisonResult(
            rankings=[],
            best_model="lgbm",
            pairwise_comparisons=[comparison],
        )

        comp = result.get_comparison("lgbm", "rf")
        assert comp is not None
        assert comp.mean_diff == -0.5

        # Should also work with reversed order
        comp = result.get_comparison("rf", "lgbm")
        assert comp is not None

        comp = result.get_comparison("lgbm", "nonexistent")
        assert comp is None

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        ranking = ModelRanking(model_name="lgbm", rank=1, metrics={"mape": 5.0})
        result = ComparisonResult(
            rankings=[ranking],
            best_model="lgbm",
        )
        result_dict = result.to_dict()

        assert "rankings" in result_dict
        assert "best_model" in result_dict
        assert result_dict["best_model"] == "lgbm"

    def test_repr(self) -> None:
        """Test string representation."""
        ranking = ModelRanking(model_name="lgbm", rank=1, metrics={"mape": 5.0})
        result = ComparisonResult(
            rankings=[ranking],
            best_model="lgbm",
        )
        repr_str = repr(result)

        assert "lgbm" in repr_str
        assert "n_models=1" in repr_str


class TestModelComparator:
    """Tests for ModelComparator class."""

    @pytest.fixture
    def comparator(self) -> ModelComparator:
        """Create a comparator with default config."""
        return ModelComparator()

    @pytest.fixture
    def sample_data(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Generate sample prediction data."""
        rng = np.random.default_rng(42)
        n = 200
        actuals = rng.normal(100, 10, n)

        # Model A: Best model (low error)
        pred_a = actuals + rng.normal(0, 3, n)
        # Model B: Moderate model
        pred_b = actuals + rng.normal(0, 5, n)
        # Model C: Worst model (high error)
        pred_c = actuals + rng.normal(2, 8, n)

        predictions = {
            "model_a": pred_a,
            "model_b": pred_b,
            "model_c": pred_c,
        }

        return predictions, actuals

    def test_init_default(self, comparator: ModelComparator) -> None:
        """Test initialization with default config."""
        assert comparator.config is not None
        assert comparator.config.primary_metric == "mape"

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = ComparisonConfig(primary_metric="rmse")
        comparator = ModelComparator(config)

        assert comparator.config.primary_metric == "rmse"

    def test_compare_basic(
        self,
        comparator: ModelComparator,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test basic model comparison."""
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        assert result.best_model in predictions.keys()
        assert len(result.rankings) == 3
        assert result.rankings[0].rank == 1
        assert result.rankings[1].rank == 2
        assert result.rankings[2].rank == 3

    def test_compare_rankings_order(
        self,
        comparator: ModelComparator,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test that rankings are ordered correctly."""
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        # Model A should rank best (lowest error)
        assert result.rankings[0].model_name == "model_a"

    def test_compare_metrics_calculated(
        self,
        comparator: ModelComparator,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test that metrics are calculated for all models."""
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        for ranking in result.rankings:
            assert "mape" in ranking.metrics
            assert "mae" in ranking.metrics
            assert "rmse" in ranking.metrics

    def test_compare_pairwise(
        self,
        comparator: ModelComparator,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test pairwise comparisons are generated."""
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        # With 3 models, we should have 3 pairwise comparisons
        assert len(result.pairwise_comparisons) == 3

        # Each comparison should have both models
        for comp in result.pairwise_comparisons:
            assert comp.model_a in predictions.keys()
            assert comp.model_b in predictions.keys()
            assert comp.model_a != comp.model_b

    def test_compare_statistical_tests(
        self,
        comparator: ModelComparator,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test that statistical tests are performed."""
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        for comp in result.pairwise_comparisons:
            assert comp.t_statistic != 0.0 or comp.mean_diff == 0.0
            assert 0.0 <= comp.t_p_value <= 1.0

    def test_compare_insufficient_models(self, comparator: ModelComparator) -> None:
        """Test comparison with less than 2 models."""
        predictions = {"model_a": np.array([1.0, 2.0, 3.0])}
        actuals = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="at least 2 models"):
            comparator.compare(predictions, actuals)

    def test_compare_length_mismatch(self, comparator: ModelComparator) -> None:
        """Test comparison with mismatched prediction lengths."""
        predictions = {
            "model_a": np.array([1.0, 2.0, 3.0]),
            "model_b": np.array([1.0, 2.0]),
        }
        actuals = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="length mismatch"):
            comparator.compare(predictions, actuals)

    def test_ranking_method_single_metric(
        self,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test single metric ranking method."""
        config = ComparisonConfig(
            ranking_method="single_metric",
            primary_metric="mape",
        )
        comparator = ModelComparator(config)
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        # Best model should have lowest MAPE
        best = result.rankings[0]
        for ranking in result.rankings[1:]:
            assert best.metrics["mape"] <= ranking.metrics["mape"]

    def test_ranking_method_weighted_average(
        self,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test weighted average ranking method."""
        config = ComparisonConfig(
            ranking_method="weighted_average",
            metrics=["mape", "rmse"],
            metric_weights={"mape": 0.7, "rmse": 0.3},
        )
        comparator = ModelComparator(config)
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        assert len(result.rankings) == 3
        # Scores should be between 0 and 1
        for ranking in result.rankings:
            assert 0.0 <= ranking.score <= 1.0

    def test_ranking_method_borda_count(
        self,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test Borda count ranking method."""
        config = ComparisonConfig(
            ranking_method="borda_count",
            metrics=["mape", "mae", "rmse"],
        )
        comparator = ModelComparator(config)
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        # Borda scores should be integers
        for ranking in result.rankings:
            assert ranking.score == int(ranking.score)

    def test_ranking_method_pareto(
        self,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test Pareto ranking method."""
        config = ComparisonConfig(
            ranking_method="pareto",
            metrics=["mape", "rmse"],
        )
        comparator = ModelComparator(config)
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        # At least one model should be Pareto-optimal
        pareto_models = [r for r in result.rankings if r.is_pareto_optimal]
        assert len(pareto_models) >= 1

    def test_recommendation_generated(
        self,
        comparator: ModelComparator,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test that recommendation is generated."""
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        assert result.recommendation != ""
        assert result.best_model in result.recommendation

    def test_confidence_intervals(
        self,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test confidence intervals are calculated."""
        config = ComparisonConfig(bootstrap_iterations=100)
        comparator = ModelComparator(config)
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        for ranking in result.rankings:
            if ranking.confidence_intervals:
                for metric, ci in ranking.confidence_intervals.items():
                    lower, upper = ci
                    assert lower <= upper

    def test_cross_validation(
        self,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test cross-validation functionality."""
        config = ComparisonConfig(cv_folds=3)
        comparator = ModelComparator(config)
        predictions, actuals = sample_data

        timestamps = pd.date_range("2023-01-01", periods=len(actuals), freq="h")

        result = comparator.compare(predictions, actuals, timestamps)

        assert result.cv_results is not None
        assert "n_folds" in result.cv_results
        assert "summary" in result.cv_results

    def test_metadata_included(
        self,
        comparator: ModelComparator,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test that metadata is included in result."""
        predictions, actuals = sample_data

        result = comparator.compare(predictions, actuals)

        assert "timestamp" in result.metadata
        assert "n_samples" in result.metadata
        assert "config" in result.metadata

    def test_get_metric_comparison_matrix(
        self,
        comparator: ModelComparator,
        sample_data: tuple[dict[str, np.ndarray], np.ndarray],
    ) -> None:
        """Test metric comparison matrix generation."""
        predictions, actuals = sample_data

        df = comparator.get_metric_comparison_matrix(predictions, actuals)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 3 models
        assert "mape" in df.columns
        assert "mae" in df.columns
        assert "rmse" in df.columns

    def test_repr(self, comparator: ModelComparator) -> None:
        """Test string representation."""
        repr_str = repr(comparator)

        assert "mape" in repr_str
        assert "single_metric" in repr_str


class TestModelComparatorStatisticalTests:
    """Tests for statistical test functionality."""

    @pytest.fixture
    def comparator(self) -> ModelComparator:
        """Create a comparator with DM test enabled."""
        config = ComparisonConfig(use_diebold_mariano=True)
        return ModelComparator(config)

    def test_diebold_mariano_test(self, comparator: ModelComparator) -> None:
        """Test Diebold-Mariano test is performed."""
        rng = np.random.default_rng(42)
        n = 200
        actuals = rng.normal(100, 10, n)
        pred_a = actuals + rng.normal(0, 3, n)
        pred_b = actuals + rng.normal(0, 5, n)

        predictions = {"model_a": pred_a, "model_b": pred_b}
        result = comparator.compare(predictions, actuals)

        for comp in result.pairwise_comparisons:
            assert comp.dm_statistic is not None
            assert comp.dm_p_value is not None
            assert 0.0 <= comp.dm_p_value <= 1.0

    def test_wilcoxon_test(self, comparator: ModelComparator) -> None:
        """Test Wilcoxon test is performed for sufficient samples."""
        rng = np.random.default_rng(42)
        n = 100
        actuals = rng.normal(100, 10, n)
        pred_a = actuals + rng.normal(0, 3, n)
        pred_b = actuals + rng.normal(0, 5, n)

        predictions = {"model_a": pred_a, "model_b": pred_b}
        result = comparator.compare(predictions, actuals)

        for comp in result.pairwise_comparisons:
            # With 100 samples, Wilcoxon should be performed
            if comp.wilcoxon_statistic is not None:
                assert 0.0 <= comp.wilcoxon_p_value <= 1.0

    def test_effect_size_calculated(self, comparator: ModelComparator) -> None:
        """Test effect size (Cohen's d) is calculated."""
        rng = np.random.default_rng(42)
        n = 100
        actuals = rng.normal(100, 10, n)
        pred_a = actuals + rng.normal(0, 3, n)
        pred_b = actuals + rng.normal(5, 5, n)  # Clearly worse

        predictions = {"model_a": pred_a, "model_b": pred_b}
        result = comparator.compare(predictions, actuals)

        for comp in result.pairwise_comparisons:
            # Effect size should be non-zero for different models
            assert comp.effect_size != 0.0


class TestModelComparatorEdgeCases:
    """Edge case tests for ModelComparator."""

    def test_identical_predictions(self) -> None:
        """Test with identical predictions from all models."""
        comparator = ModelComparator()
        rng = np.random.default_rng(42)
        actuals = rng.normal(100, 10, 100)

        # All models make the same predictions
        predictions = {
            "model_a": actuals + 1,
            "model_b": actuals + 1,
            "model_c": actuals + 1,
        }

        result = comparator.compare(predictions, actuals)

        # All models should have the same metrics
        assert result.rankings[0].metrics["mape"] == result.rankings[1].metrics["mape"]

    def test_perfect_model(self) -> None:
        """Test with a perfect model (zero error)."""
        comparator = ModelComparator()
        rng = np.random.default_rng(42)
        actuals = rng.normal(100, 10, 100)

        predictions = {
            "perfect": actuals.copy(),  # Perfect predictions
            "imperfect": actuals + rng.normal(0, 5, 100),
        }

        result = comparator.compare(predictions, actuals)

        # Perfect model should be best
        assert result.best_model == "perfect"
        assert result.rankings[0].metrics["mape"] < 0.01  # Nearly zero

    def test_two_models(self) -> None:
        """Test with exactly two models."""
        comparator = ModelComparator()
        rng = np.random.default_rng(42)
        actuals = rng.normal(100, 10, 100)

        predictions = {
            "model_a": actuals + rng.normal(0, 3, 100),
            "model_b": actuals + rng.normal(0, 5, 100),
        }

        result = comparator.compare(predictions, actuals)

        assert len(result.rankings) == 2
        assert len(result.pairwise_comparisons) == 1

    def test_small_sample_size(self) -> None:
        """Test with small sample size."""
        comparator = ModelComparator()
        rng = np.random.default_rng(42)
        n = 20
        actuals = rng.normal(100, 10, n)

        predictions = {
            "model_a": actuals + rng.normal(0, 3, n),
            "model_b": actuals + rng.normal(0, 5, n),
        }

        result = comparator.compare(predictions, actuals)

        # Should still work with small samples
        assert len(result.rankings) == 2

    def test_many_models(self) -> None:
        """Test with many models."""
        comparator = ModelComparator()
        rng = np.random.default_rng(42)
        actuals = rng.normal(100, 10, 100)

        # Create 10 models with varying error levels
        predictions = {}
        for i in range(10):
            predictions[f"model_{i}"] = actuals + rng.normal(0, i + 1, 100)

        result = comparator.compare(predictions, actuals)

        assert len(result.rankings) == 10
        # Number of pairwise comparisons = n(n-1)/2 = 45
        assert len(result.pairwise_comparisons) == 45


class TestComparisonEnums:
    """Tests for comparison enums."""

    def test_comparison_metric_values(self) -> None:
        """Test ComparisonMetric enum values."""
        assert ComparisonMetric.MAPE.value == "mape"
        assert ComparisonMetric.MAE.value == "mae"
        assert ComparisonMetric.RMSE.value == "rmse"
        assert ComparisonMetric.MSE.value == "mse"
        assert ComparisonMetric.R2.value == "r2"

    def test_ranking_method_values(self) -> None:
        """Test RankingMethod enum values."""
        assert RankingMethod.SINGLE_METRIC.value == "single_metric"
        assert RankingMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert RankingMethod.PARETO.value == "pareto"
        assert RankingMethod.BORDA_COUNT.value == "borda_count"


class TestModelComparatorIntegration:
    """Integration tests for model comparison."""

    def test_electric_load_scenario(self) -> None:
        """Test realistic electric load forecasting scenario."""
        comparator = ModelComparator(
            ComparisonConfig(
                metrics=["mape", "rmse"],
                ranking_method="weighted_average",
                metric_weights={"mape": 0.6, "rmse": 0.4},
            )
        )
        rng = np.random.default_rng(42)

        # Simulate hourly load data
        n = 24 * 30  # 30 days of hourly data
        actuals = rng.normal(5000, 500, n)

        # Different model types with different characteristics
        lgbm_pred = actuals + rng.normal(0, 200, n)  # Best overall
        rf_pred = actuals + rng.normal(50, 250, n)  # Slight bias, higher variance
        arima_pred = actuals + rng.normal(-50, 300, n)  # Different bias

        predictions = {
            "lgbm": lgbm_pred,
            "random_forest": rf_pred,
            "arima": arima_pred,
        }

        result = comparator.compare(predictions, actuals)

        # LGBM should be best
        assert result.best_model == "lgbm"

        # Recommendation should be meaningful
        assert len(result.recommendation) > 0

        # All pairwise comparisons should be valid
        for comp in result.pairwise_comparisons:
            assert comp.t_p_value >= 0
            assert comp.t_p_value <= 1

    def test_horizon_comparison(self) -> None:
        """Test comparison across multiple horizons."""
        comparator = ModelComparator()
        rng = np.random.default_rng(42)

        horizons = [0, 1, 2, 3]  # D+0 to D+3
        predictions_by_horizon = {}
        actuals_by_horizon = {}

        for h in horizons:
            n = 100
            actuals = rng.normal(100 + h * 10, 10, n)
            actuals_by_horizon[h] = actuals

            # Error increases with horizon
            predictions_by_horizon[h] = {
                "lgbm": actuals + rng.normal(0, 3 + h, n),
                "rf": actuals + rng.normal(0, 4 + h, n),
            }

        results = comparator.compare_horizons(predictions_by_horizon, actuals_by_horizon)

        assert len(results) == 4
        for h, result in results.items():
            assert result.best_model in ["lgbm", "rf"]
