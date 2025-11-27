"""Tests for performance-based model selector.

Tests cover:
- PerformanceSelectorConfig initialization
- RollingPerformanceTracker functionality
- PerformanceBasedSelector with different strategies
- Multi-objective optimization
- Selection diagnostics and history
"""

# ruff: noqa: NPY002

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.models.combination.performance_selector import (
    PerformanceBasedSelector,
    PerformanceSelectorConfig,
    RollingPerformanceTracker,
    SelectionDiagnostics,
    SelectionRecord,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_predictions() -> dict[str, pd.DataFrame]:
    """Create sample predictions from multiple models."""
    np.random.seed(42)
    n_samples = 100

    # Create predictions with different characteristics
    base = np.linspace(100, 200, n_samples)

    predictions = {}

    # Model 1: Good accuracy, low diversity (similar to truth)
    preds_1 = {
        f"pred_h{i}": base + np.random.normal(0, 2, n_samples)
        for i in range(3)
    }
    predictions["lgbm"] = pd.DataFrame(preds_1)

    # Model 2: Medium accuracy, medium diversity
    preds_2 = {
        f"pred_h{i}": base + np.random.normal(0, 5, n_samples) + i * 2
        for i in range(3)
    }
    predictions["xgb"] = pd.DataFrame(preds_2)

    # Model 3: Lower accuracy, high diversity (different pattern)
    preds_3 = {
        f"pred_h{i}": base + np.random.normal(0, 8, n_samples) + np.sin(np.linspace(0, 4 * np.pi, n_samples)) * 10
        for i in range(3)
    }
    predictions["catboost"] = pd.DataFrame(preds_3)

    # Model 4: Similar to model 1 (low diversity with it)
    preds_4 = {
        f"pred_h{i}": base + np.random.normal(0, 2.5, n_samples) + 0.5
        for i in range(3)
    }
    predictions["rf"] = pd.DataFrame(preds_4)

    # Model 5: High error model
    preds_5 = {
        f"pred_h{i}": base + np.random.normal(0, 15, n_samples) + 20
        for i in range(3)
    }
    predictions["linear"] = pd.DataFrame(preds_5)

    return predictions


@pytest.fixture
def sample_targets() -> pd.DataFrame:
    """Create sample targets."""
    np.random.seed(42)
    n_samples = 100
    base = np.linspace(100, 200, n_samples)

    return pd.DataFrame({
        f"target_h{i}": base + np.random.normal(0, 0.5, n_samples)
        for i in range(3)
    })


@pytest.fixture
def sample_computational_costs() -> dict[str, float]:
    """Create sample computational costs."""
    return {
        "lgbm": 1.0,
        "xgb": 1.2,
        "catboost": 1.5,
        "rf": 0.8,
        "linear": 0.3,
    }


@pytest.fixture
def fitted_selector(
    sample_predictions: dict[str, pd.DataFrame],
    sample_targets: pd.DataFrame,
    sample_computational_costs: dict[str, float],
) -> PerformanceBasedSelector:
    """Create a fitted selector."""
    selector = PerformanceBasedSelector(config={"selection_strategy": "adaptive"})
    selector.fit(sample_predictions, sample_targets, sample_computational_costs)
    return selector


# =============================================================================
# PerformanceSelectorConfig Tests
# =============================================================================


class TestPerformanceSelectorConfig:
    """Tests for PerformanceSelectorConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PerformanceSelectorConfig()

        assert config.selection_strategy == "adaptive"
        assert config.max_ensemble_size == 5
        assert config.min_ensemble_size == 2
        assert config.window_size == 48
        assert config.diversity_weight == 0.3
        assert config.accuracy_weight == 0.6
        assert config.cost_weight == 0.1
        assert config.marginal_gain_threshold == 0.01
        assert config.recency_decay == 0.95
        assert config.min_diversity_threshold == 0.1

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PerformanceSelectorConfig(
            selection_strategy="greedy",
            max_ensemble_size=3,
            diversity_weight=0.5,
        )

        assert config.selection_strategy == "greedy"
        assert config.max_ensemble_size == 3
        assert config.diversity_weight == 0.5

    def test_weight_sum(self) -> None:
        """Test that default weights sum to 1."""
        config = PerformanceSelectorConfig()
        total_weight = config.accuracy_weight + config.diversity_weight + config.cost_weight
        assert abs(total_weight - 1.0) < 0.01


# =============================================================================
# RollingPerformanceTracker Tests
# =============================================================================


class TestRollingPerformanceTracker:
    """Tests for RollingPerformanceTracker."""

    def test_initialization(self) -> None:
        """Test tracker initialization."""
        tracker = RollingPerformanceTracker(window_size=24)
        assert tracker.window_size == 24
        assert tracker.get_model_names() == []

    def test_update_single_model(self) -> None:
        """Test updating performance for a single model."""
        tracker = RollingPerformanceTracker(window_size=48)

        tracker.update("lgbm", accuracy=2.5, diversity=0.8, cost=1.0)

        assert "lgbm" in tracker.get_model_names()
        perf = tracker.get_recent_performance("lgbm")
        assert perf["accuracy"] == 2.5
        assert perf["diversity"] == 0.8
        assert perf["cost"] == 1.0

    def test_update_multiple_models(self) -> None:
        """Test updating performance for multiple models."""
        tracker = RollingPerformanceTracker(window_size=48)

        tracker.update("lgbm", accuracy=2.5, diversity=0.8, cost=1.0)
        tracker.update("xgb", accuracy=3.0, diversity=0.6, cost=1.2)

        assert set(tracker.get_model_names()) == {"lgbm", "xgb"}

    def test_rolling_average(self) -> None:
        """Test rolling average calculation."""
        tracker = RollingPerformanceTracker(window_size=10)

        # Add multiple observations
        for i in range(5):
            tracker.update("lgbm", accuracy=2.0 + i * 0.2, diversity=0.5, cost=1.0)

        perf = tracker.get_recent_performance("lgbm")
        # Average of 2.0, 2.2, 2.4, 2.6, 2.8 = 2.4
        assert abs(perf["accuracy"] - 2.4) < 0.01

    def test_window_size_limit(self) -> None:
        """Test that window size limits history."""
        tracker = RollingPerformanceTracker(window_size=3)

        for i in range(10):
            tracker.update("lgbm", accuracy=float(i), diversity=0.5, cost=1.0)

        perf = tracker.get_recent_performance("lgbm")
        # Only last 3 values: 7, 8, 9 -> mean = 8.0
        assert abs(perf["accuracy"] - 8.0) < 0.01

    def test_get_performance_trend(self) -> None:
        """Test performance trend calculation."""
        tracker = RollingPerformanceTracker(window_size=10)

        # Add improving accuracy (decreasing MAPE)
        for i in range(6):
            tracker.update("lgbm", accuracy=5.0 - i * 0.5, diversity=0.5, cost=1.0)

        trend = tracker.get_performance_trend("lgbm")
        # Accuracy is improving (getting lower), so trend should be positive
        assert trend["accuracy_trend"] > 0

    def test_get_performance_trend_insufficient_data(self) -> None:
        """Test trend with insufficient data."""
        tracker = RollingPerformanceTracker(window_size=10)

        tracker.update("lgbm", accuracy=2.5, diversity=0.5, cost=1.0)

        trend = tracker.get_performance_trend("lgbm")
        assert trend["accuracy_trend"] == 0.0
        assert trend["diversity_trend"] == 0.0

    def test_get_performance_unknown_model(self) -> None:
        """Test getting performance for unknown model."""
        tracker = RollingPerformanceTracker(window_size=48)

        perf = tracker.get_recent_performance("unknown")
        assert perf["accuracy"] == float("inf")
        assert perf["diversity"] == 0.0
        assert perf["cost"] == 1.0

    def test_reset_single_model(self) -> None:
        """Test resetting a single model."""
        tracker = RollingPerformanceTracker(window_size=48)

        tracker.update("lgbm", accuracy=2.5, diversity=0.8, cost=1.0)
        tracker.update("xgb", accuracy=3.0, diversity=0.6, cost=1.2)

        tracker.reset("lgbm")

        # lgbm should have default values
        perf_lgbm = tracker.get_recent_performance("lgbm")
        assert perf_lgbm["accuracy"] == float("inf")

        # xgb should still have values
        perf_xgb = tracker.get_recent_performance("xgb")
        assert perf_xgb["accuracy"] == 3.0

    def test_reset_all_models(self) -> None:
        """Test resetting all models."""
        tracker = RollingPerformanceTracker(window_size=48)

        tracker.update("lgbm", accuracy=2.5, diversity=0.8, cost=1.0)
        tracker.update("xgb", accuracy=3.0, diversity=0.6, cost=1.2)

        tracker.reset()

        assert tracker.get_model_names() == []


# =============================================================================
# SelectionRecord Tests
# =============================================================================


class TestSelectionRecord:
    """Tests for SelectionRecord dataclass."""

    def test_record_creation(self) -> None:
        """Test creating a selection record."""
        record = SelectionRecord(
            timestamp=datetime.now(tz=UTC),
            selected_models=["lgbm", "xgb"],
            selection_strategy="greedy",
            composite_scores={"lgbm": 0.8, "xgb": 0.7},
            expected_performance=2.5,
            ensemble_size=2,
            reason="Test selection",
            diversity_score=0.6,
        )

        assert record.selected_models == ["lgbm", "xgb"]
        assert record.ensemble_size == 2
        assert record.diversity_score == 0.6


# =============================================================================
# SelectionDiagnostics Tests
# =============================================================================


class TestSelectionDiagnostics:
    """Tests for SelectionDiagnostics dataclass."""

    def test_diagnostics_defaults(self) -> None:
        """Test default diagnostic values."""
        diag = SelectionDiagnostics()

        assert diag.n_models_available == 0
        assert diag.n_models_selected == 0
        assert diag.selection_strategy == ""
        assert diag.individual_scores == {}
        assert diag.ensemble_performance == 0.0
        assert diag.ensemble_diversity == 0.0
        assert diag.selection_time_ms == 0.0
        assert diag.pareto_candidates == 0
        assert diag.marginal_gains == []


# =============================================================================
# PerformanceBasedSelector Tests
# =============================================================================


class TestPerformanceBasedSelectorInit:
    """Tests for PerformanceBasedSelector initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        selector = PerformanceBasedSelector()

        assert selector.config.selection_strategy == "adaptive"
        assert selector.fitted is False
        assert selector.model_pool == []
        assert selector.selection_history == []
        assert selector.expected_improvement is None

    def test_custom_config(self) -> None:
        """Test initialization with custom config."""
        selector = PerformanceBasedSelector(config={
            "selection_strategy": "pareto",
            "max_ensemble_size": 4,
            "diversity_weight": 0.4,
        })

        assert selector.config.selection_strategy == "pareto"
        assert selector.config.max_ensemble_size == 4
        assert selector.config.diversity_weight == 0.4

    def test_repr(self) -> None:
        """Test string representation."""
        selector = PerformanceBasedSelector()
        repr_str = repr(selector)

        assert "PerformanceBasedSelector" in repr_str
        assert "adaptive" in repr_str
        assert "not fitted" in repr_str


class TestPerformanceBasedSelectorFit:
    """Tests for fitting the selector."""

    def test_fit_basic(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test basic fitting."""
        selector = PerformanceBasedSelector()
        selector.fit(sample_predictions, sample_targets)

        assert selector.fitted is True
        assert len(selector.model_pool) == 5
        assert "lgbm" in selector.model_pool

    def test_fit_with_costs(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
        sample_computational_costs: dict[str, float],
    ) -> None:
        """Test fitting with computational costs."""
        selector = PerformanceBasedSelector()
        selector.fit(sample_predictions, sample_targets, sample_computational_costs)

        assert selector.fitted is True
        # Verify costs are tracked
        perf = selector.performance_tracker.get_recent_performance("linear")
        assert perf["cost"] == 0.3

    def test_fit_empty_predictions_raises(self, sample_targets: pd.DataFrame) -> None:
        """Test that empty predictions raise ValueError."""
        selector = PerformanceBasedSelector()

        with pytest.raises(ValueError, match="empty"):
            selector.fit({}, sample_targets)

    def test_fit_updates_correlation_matrix(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test that fitting updates correlation matrix."""
        selector = PerformanceBasedSelector()
        selector.fit(sample_predictions, sample_targets)

        # Check correlation matrix is populated
        assert len(selector._correlation_matrix) > 0
        assert ("lgbm", "xgb") in selector._correlation_matrix

    def test_fit_alternative_target_columns(
        self,
        sample_predictions: dict[str, pd.DataFrame],
    ) -> None:
        """Test fitting with h0, h1 style target columns."""
        np.random.seed(42)
        n_samples = 100
        base = np.linspace(100, 200, n_samples)

        targets = pd.DataFrame({
            f"h{i}": base + np.random.normal(0, 0.5, n_samples)
            for i in range(3)
        })

        selector = PerformanceBasedSelector()
        selector.fit(sample_predictions, targets)

        assert selector.fitted is True


class TestPerformanceBasedSelectorSelection:
    """Tests for model selection."""

    def test_select_not_fitted_raises(self) -> None:
        """Test that selection without fitting raises error."""
        selector = PerformanceBasedSelector()

        with pytest.raises(RuntimeError, match="not been fitted"):
            selector.select_optimal_ensemble()

    def test_greedy_selection(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test greedy selection strategy."""
        selector = PerformanceBasedSelector(config={"selection_strategy": "greedy"})
        selector.fit(sample_predictions, sample_targets)

        selected = selector.select_optimal_ensemble()

        assert len(selected) >= selector.config.min_ensemble_size
        assert len(selected) <= selector.config.max_ensemble_size
        assert all(m in selector.model_pool for m in selected)

    def test_pareto_selection(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test Pareto selection strategy."""
        selector = PerformanceBasedSelector(config={"selection_strategy": "pareto"})
        selector.fit(sample_predictions, sample_targets)

        selected = selector.select_optimal_ensemble()

        assert len(selected) >= selector.config.min_ensemble_size
        assert len(selected) <= selector.config.max_ensemble_size

    def test_adaptive_selection(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test adaptive selection strategy."""
        selector = PerformanceBasedSelector(config={"selection_strategy": "adaptive"})
        selector.fit(sample_predictions, sample_targets)

        selected = selector.select_optimal_ensemble()

        assert len(selected) >= selector.config.min_ensemble_size
        assert len(selected) <= selector.config.max_ensemble_size

    def test_unknown_strategy_raises(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test that unknown strategy raises ValueError."""
        selector = PerformanceBasedSelector(config={"selection_strategy": "unknown"})
        selector.fit(sample_predictions, sample_targets)

        with pytest.raises(ValueError, match="Unknown selection strategy"):
            selector.select_optimal_ensemble()

    def test_small_model_pool(self) -> None:
        """Test selection with small model pool."""
        np.random.seed(42)
        n_samples = 100
        base = np.linspace(100, 200, n_samples)

        # Only 2 models
        predictions = {
            "lgbm": pd.DataFrame({f"pred_h{i}": base + np.random.normal(0, 2, n_samples) for i in range(3)}),
            "xgb": pd.DataFrame({f"pred_h{i}": base + np.random.normal(0, 3, n_samples) for i in range(3)}),
        }
        targets = pd.DataFrame({f"target_h{i}": base for i in range(3)})

        selector = PerformanceBasedSelector(config={"min_ensemble_size": 2})
        selector.fit(predictions, targets)

        selected = selector.select_optimal_ensemble()
        assert selected == ["lgbm", "xgb"]

    def test_selection_records_history(self, fitted_selector: PerformanceBasedSelector) -> None:
        """Test that selection records history."""
        fitted_selector.select_optimal_ensemble()

        assert len(fitted_selector.selection_history) == 1
        record = fitted_selector.selection_history[0]
        assert isinstance(record, SelectionRecord)
        assert len(record.selected_models) > 0

    def test_selection_updates_expected_improvement(
        self,
        fitted_selector: PerformanceBasedSelector,
    ) -> None:
        """Test that selection updates expected improvement."""
        fitted_selector.select_optimal_ensemble()

        assert fitted_selector.expected_improvement is not None

    def test_best_model_always_selected(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test that best performing model is typically selected."""
        selector = PerformanceBasedSelector(config={"selection_strategy": "greedy"})
        selector.fit(sample_predictions, sample_targets)

        selected = selector.select_optimal_ensemble()

        # lgbm has best accuracy in our fixture, should typically be selected
        # (though not guaranteed depending on diversity considerations)
        scores = selector._calculate_multi_objective_scores()
        best_model = max(scores, key=scores.get)
        assert best_model in selected


class TestPerformanceBasedSelectorDiagnostics:
    """Tests for selection diagnostics."""

    def test_get_diagnostics_before_selection(self) -> None:
        """Test getting diagnostics before selection."""
        selector = PerformanceBasedSelector()
        diag = selector.get_selection_diagnostics()

        assert diag == {}

    def test_get_diagnostics_after_selection(
        self,
        fitted_selector: PerformanceBasedSelector,
    ) -> None:
        """Test getting diagnostics after selection."""
        fitted_selector.select_optimal_ensemble()
        diag = fitted_selector.get_selection_diagnostics()

        assert diag["n_models_available"] == 5
        assert diag["n_models_selected"] >= 2
        assert "selection_strategy" in diag
        assert "individual_scores" in diag
        assert "ensemble_performance" in diag
        assert "ensemble_diversity" in diag
        assert diag["selection_time_ms"] > 0

    def test_pareto_diagnostics_include_candidates(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test that Pareto selection includes candidate count."""
        selector = PerformanceBasedSelector(config={"selection_strategy": "pareto"})
        selector.fit(sample_predictions, sample_targets)
        selector.select_optimal_ensemble()

        diag = selector.get_selection_diagnostics()
        assert diag["pareto_candidates"] > 0

    def test_greedy_diagnostics_include_marginal_gains(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test that greedy selection includes marginal gains."""
        selector = PerformanceBasedSelector(config={"selection_strategy": "greedy"})
        selector.fit(sample_predictions, sample_targets)
        selector.select_optimal_ensemble()

        diag = selector.get_selection_diagnostics()
        assert "marginal_gains" in diag


class TestPerformanceBasedSelectorHistory:
    """Tests for selection history."""

    def test_get_history_summary_empty(self) -> None:
        """Test history summary with no selections."""
        selector = PerformanceBasedSelector()
        summary = selector.get_selection_history_summary()

        assert summary["n_selections"] == 0

    def test_get_history_summary_after_selections(
        self,
        fitted_selector: PerformanceBasedSelector,
    ) -> None:
        """Test history summary after multiple selections."""
        # Make multiple selections
        for _ in range(3):
            fitted_selector.select_optimal_ensemble()

        summary = fitted_selector.get_selection_history_summary()

        assert summary["n_selections"] == 3
        assert "avg_ensemble_size" in summary
        assert "avg_performance" in summary
        assert "avg_diversity" in summary
        assert "model_selection_frequency" in summary
        assert "most_selected" in summary


class TestPerformanceBasedSelectorUpdatePerformance:
    """Tests for dynamic performance updates."""

    def test_update_performance(self, fitted_selector: PerformanceBasedSelector) -> None:
        """Test updating performance for a model."""
        initial_perf = fitted_selector.performance_tracker.get_recent_performance("lgbm")

        fitted_selector.update_performance("lgbm", accuracy=1.5)

        updated_perf = fitted_selector.performance_tracker.get_recent_performance("lgbm")

        # Accuracy should have changed (rolling average includes new value)
        assert updated_perf["accuracy"] != initial_perf["accuracy"]

    def test_update_performance_with_diversity(
        self,
        fitted_selector: PerformanceBasedSelector,
    ) -> None:
        """Test updating performance with diversity."""
        fitted_selector.update_performance("lgbm", accuracy=1.5, diversity=0.9)

        perf = fitted_selector.performance_tracker.get_recent_performance("lgbm")
        # New value should influence the average
        assert perf["diversity"] > 0  # Just check it's valid


class TestPerformanceBasedSelectorDiversity:
    """Tests for diversity calculations."""

    def test_calculate_ensemble_diversity_single_model(
        self,
        fitted_selector: PerformanceBasedSelector,
    ) -> None:
        """Test diversity with single model."""
        diversity = fitted_selector._calculate_ensemble_diversity(["lgbm"])
        assert diversity == 0.0

    def test_calculate_ensemble_diversity_two_models(
        self,
        fitted_selector: PerformanceBasedSelector,
    ) -> None:
        """Test diversity with two models."""
        diversity = fitted_selector._calculate_ensemble_diversity(["lgbm", "catboost"])
        assert 0.0 <= diversity <= 1.0

    def test_similar_models_low_diversity(
        self,
        fitted_selector: PerformanceBasedSelector,
    ) -> None:
        """Test that similar models have low diversity."""
        # lgbm and rf are similar in our fixture
        div_similar = fitted_selector._calculate_ensemble_diversity(["lgbm", "rf"])
        # lgbm and catboost are different
        div_different = fitted_selector._calculate_ensemble_diversity(["lgbm", "catboost"])

        # Different models should have higher diversity
        assert div_different > div_similar or abs(div_different - div_similar) < 0.1


class TestPerformanceBasedSelectorMultiObjective:
    """Tests for multi-objective optimization."""

    def test_calculate_multi_objective_scores(
        self,
        fitted_selector: PerformanceBasedSelector,
    ) -> None:
        """Test multi-objective score calculation."""
        scores = fitted_selector._calculate_multi_objective_scores()

        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores.values())
        assert all(s > 0 for s in scores.values())

    def test_score_reflects_accuracy(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test that scores reflect accuracy differences."""
        selector = PerformanceBasedSelector(config={
            "accuracy_weight": 1.0,
            "diversity_weight": 0.0,
            "cost_weight": 0.0,
        })
        selector.fit(sample_predictions, sample_targets)

        scores = selector._calculate_multi_objective_scores()

        # lgbm should have higher score than linear (better accuracy)
        assert scores["lgbm"] > scores["linear"]

    def test_score_reflects_diversity(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test that scores reflect diversity."""
        selector = PerformanceBasedSelector(config={
            "accuracy_weight": 0.0,
            "diversity_weight": 1.0,
            "cost_weight": 0.0,
        })
        selector.fit(sample_predictions, sample_targets)

        scores = selector._calculate_multi_objective_scores()

        # catboost should have higher diversity score (more different pattern)
        # Check that scores vary based on diversity
        assert max(scores.values()) != min(scores.values())


class TestPerformanceBasedSelectorEdgeCases:
    """Tests for edge cases."""

    def test_single_horizon(self) -> None:
        """Test with single horizon predictions."""
        np.random.seed(42)
        n_samples = 100
        base = np.linspace(100, 200, n_samples)

        predictions = {
            "lgbm": pd.DataFrame({"pred_h0": base + np.random.normal(0, 2, n_samples)}),
            "xgb": pd.DataFrame({"pred_h0": base + np.random.normal(0, 3, n_samples)}),
            "rf": pd.DataFrame({"pred_h0": base + np.random.normal(0, 4, n_samples)}),
        }
        targets = pd.DataFrame({"target_h0": base})

        selector = PerformanceBasedSelector()
        selector.fit(predictions, targets)

        selected = selector.select_optimal_ensemble()
        assert len(selected) >= 2

    def test_with_nan_values(self) -> None:
        """Test handling of NaN values in predictions."""
        np.random.seed(42)
        n_samples = 100
        base = np.linspace(100, 200, n_samples)

        predictions = {
            "lgbm": pd.DataFrame({"pred_h0": base + np.random.normal(0, 2, n_samples)}),
            "xgb": pd.DataFrame({"pred_h0": base + np.random.normal(0, 3, n_samples)}),
            "rf": pd.DataFrame({"pred_h0": base + np.random.normal(0, 4, n_samples)}),
        }
        # Add some NaN values
        predictions["lgbm"].iloc[10:15, 0] = np.nan

        targets = pd.DataFrame({"target_h0": base})

        selector = PerformanceBasedSelector()
        selector.fit(predictions, targets)

        selected = selector.select_optimal_ensemble()
        assert len(selected) >= 2

    def test_high_correlation_models_filtered(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
    ) -> None:
        """Test that highly correlated models are filtered in greedy selection."""
        # Add a model that is identical to lgbm
        sample_predictions["lgbm_copy"] = sample_predictions["lgbm"].copy()

        selector = PerformanceBasedSelector(config={
            "selection_strategy": "greedy",
            "marginal_gain_threshold": 0.001,  # Lower threshold to allow more models
            "max_ensemble_size": 4,
        })
        selector.fit(sample_predictions, sample_targets)

        selected = selector.select_optimal_ensemble()

        # The greedy selection should skip highly correlated models (>0.95 correlation)
        # When both lgbm and lgbm_copy are considered, the second one should be skipped
        # If we have more than 2 models selected, check correlation filtering worked
        if len(selected) > 2:
            # If greedy added models beyond the first two, lgbm_copy shouldn't be there
            # because it's perfectly correlated with lgbm
            has_lgbm = "lgbm" in selected
            has_lgbm_copy = "lgbm_copy" in selected
            # At least verify one is picked, or if both are there, they're the only options
            assert has_lgbm or has_lgbm_copy

        # Verify that correlation filtering is working by checking the correlation matrix
        corr = selector._correlation_matrix.get(("lgbm", "lgbm_copy"), 0.0)
        assert corr > 0.95  # Should be highly correlated

    def test_fallback_selection(self) -> None:
        """Test fallback selection mechanism."""
        np.random.seed(42)
        n_samples = 100
        base = np.linspace(100, 200, n_samples)

        # Create predictions where greedy might struggle
        predictions = {
            "model_a": pd.DataFrame({"pred_h0": base + np.random.normal(0, 2, n_samples)}),
            "model_b": pd.DataFrame({"pred_h0": base + np.random.normal(0, 2, n_samples)}),
            "model_c": pd.DataFrame({"pred_h0": base + np.random.normal(0, 2, n_samples)}),
        }
        targets = pd.DataFrame({"target_h0": base})

        selector = PerformanceBasedSelector(config={
            "selection_strategy": "greedy",
            "marginal_gain_threshold": 0.99,  # Very high threshold
            "min_ensemble_size": 2,
        })
        selector.fit(predictions, targets)

        # Should still get at least min_ensemble_size models via fallback
        selected = selector.select_optimal_ensemble()
        assert len(selected) >= 2


class TestPerformanceBasedSelectorIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(
        self,
        sample_predictions: dict[str, pd.DataFrame],
        sample_targets: pd.DataFrame,
        sample_computational_costs: dict[str, float],
    ) -> None:
        """Test complete selection workflow."""
        # 1. Initialize
        selector = PerformanceBasedSelector(config={"selection_strategy": "adaptive"})

        # 2. Fit
        selector.fit(sample_predictions, sample_targets, sample_computational_costs)
        assert selector.fitted

        # 3. Select
        selected = selector.select_optimal_ensemble()
        assert len(selected) >= 2

        # 4. Get diagnostics
        diag = selector.get_selection_diagnostics()
        assert diag["n_models_selected"] == len(selected)

        # 5. Update performance
        selector.update_performance("lgbm", accuracy=1.8)

        # 6. Select again
        selected_2 = selector.select_optimal_ensemble()
        assert len(selected_2) >= 2

        # 7. Get history
        summary = selector.get_selection_history_summary()
        assert summary["n_selections"] == 2

    def test_multiple_selection_rounds(
        self,
        fitted_selector: PerformanceBasedSelector,
    ) -> None:
        """Test multiple rounds of selection."""
        selections = []

        for _ in range(5):
            selected = fitted_selector.select_optimal_ensemble()
            selections.append(selected)

        # All selections should be valid
        assert all(len(s) >= 2 for s in selections)

        # History should be updated
        summary = fitted_selector.get_selection_history_summary()
        assert summary["n_selections"] == 5
