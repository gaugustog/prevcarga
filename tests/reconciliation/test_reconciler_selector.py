# ruff: noqa: NPY002
"""Tests for ReconcilerSelector automatic method selection.

Tests cover:
- Method performance evaluation
- Scoring functions (accuracy, consistency, efficiency)
- Method selection logic
- Rolling window averaging
- Selection history and reporting
- Save/load persistence
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.reconciliation.data_structures import ReconciliationConfig, ReconciliationResult
from src.reconciliation.hierarchy import HierarchyDefinition
from src.reconciliation.reconciler_selector import (
    MethodPerformance,
    ReconcilerSelector,
    SelectionResult,
    SelectorConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_hierarchy(tmp_path):
    """Create a sample hierarchy for testing."""
    yaml_content = """
hierarchy:
  national:
    name: "Total"
    children: ["North", "South"]

  subsystems:
    North:
      name: "North"
      children: ["City_A", "City_B"]
    South:
      name: "South"
      children: ["City_C", "City_D"]

  areas:
    City_A:
      name: "City_A"
      type: "consumption"
    City_B:
      name: "City_B"
      type: "consumption"
    City_C:
      name: "City_C"
      type: "consumption"
    City_D:
      name: "City_D"
      type: "consumption"
"""
    yaml_file = tmp_path / "test_hierarchy.yaml"
    yaml_file.write_text(yaml_content)
    return HierarchyDefinition(str(yaml_file))


@pytest.fixture
def sample_forecasts():
    """Create sample base forecasts."""
    np.random.seed(42)
    n_timesteps = 20

    city_a = 100 + np.random.randn(n_timesteps) * 10
    city_b = 150 + np.random.randn(n_timesteps) * 15
    city_c = 80 + np.random.randn(n_timesteps) * 8
    city_d = 120 + np.random.randn(n_timesteps) * 12

    north = city_a + city_b + np.random.randn(n_timesteps) * 5
    south = city_c + city_d + np.random.randn(n_timesteps) * 5
    total = north + south + np.random.randn(n_timesteps) * 10

    return pd.DataFrame(
        {
            "Total": total,
            "North": north,
            "South": south,
            "City_A": city_a,
            "City_B": city_b,
            "City_C": city_c,
            "City_D": city_d,
        }
    )


@pytest.fixture
def sample_actuals(sample_forecasts):
    """Create sample actual values (close to forecasts with some noise)."""
    np.random.seed(43)
    actuals = sample_forecasts.copy()
    for col in actuals.columns:
        actuals[col] += np.random.randn(len(actuals)) * 5
    return actuals


@pytest.fixture
def mock_reconciler():
    """Create a mock reconciler."""

    def create_mock(name: str, reconcile_time: float = 0.1, coherence_error: float = 0.0):
        mock = MagicMock()
        mock.name = name

        def mock_reconcile(base_forecasts, hierarchy, **kwargs):
            result = MagicMock(spec=ReconciliationResult)
            # Return modified forecasts (small random adjustment)
            reconciled = base_forecasts.copy()
            for col in reconciled.columns:
                reconciled[col] += np.random.randn(len(reconciled)) * 0.01
            result.reconciled_forecasts = reconciled
            return result

        mock.reconcile = mock_reconcile
        mock.check_coherence = MagicMock(return_value=coherence_error)

        return mock

    return create_mock


# =============================================================================
# SelectorConfig Tests
# =============================================================================


class TestSelectorConfig:
    """Tests for SelectorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SelectorConfig()

        # Weights should sum to 1
        total = config.accuracy_weight + config.consistency_weight + config.efficiency_weight
        assert abs(total - 1.0) < 1e-10

    def test_weight_normalization(self):
        """Test that weights are normalized."""
        config = SelectorConfig(
            accuracy_weight=6.0,
            consistency_weight=3.0,
            efficiency_weight=1.0,
        )

        total = config.accuracy_weight + config.consistency_weight + config.efficiency_weight
        assert abs(total - 1.0) < 1e-10
        assert abs(config.accuracy_weight - 0.6) < 1e-10
        assert abs(config.consistency_weight - 0.3) < 1e-10
        assert abs(config.efficiency_weight - 0.1) < 1e-10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SelectorConfig(rolling_window_size=5)
        d = config.to_dict()

        assert "accuracy_weight" in d
        assert "consistency_weight" in d
        assert "efficiency_weight" in d
        assert d["rolling_window_size"] == 5


# =============================================================================
# MethodPerformance Tests
# =============================================================================


class TestMethodPerformance:
    """Tests for MethodPerformance dataclass."""

    def test_creation(self):
        """Test basic creation."""
        perf = MethodPerformance(
            method_name="test",
            accuracy_score=0.9,
            consistency_score=1.0,
            efficiency_score=0.8,
            overall_score=0.9,
        )

        assert perf.method_name == "test"
        assert perf.accuracy_score == 0.9
        assert perf.consistency_score == 1.0
        assert perf.efficiency_score == 0.8
        assert perf.overall_score == 0.9

    def test_to_dict(self):
        """Test conversion to dictionary."""
        perf = MethodPerformance(
            method_name="test",
            accuracy_score=0.9,
            consistency_score=1.0,
            efficiency_score=0.8,
            overall_score=0.9,
            mape=0.05,
            computation_time=1.5,
        )

        d = perf.to_dict()

        assert d["method_name"] == "test"
        assert d["mape"] == 0.05
        assert d["computation_time"] == 1.5


# =============================================================================
# SelectionResult Tests
# =============================================================================


class TestSelectionResult:
    """Tests for SelectionResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        perf = MethodPerformance(
            method_name="best",
            accuracy_score=0.9,
            consistency_score=1.0,
            efficiency_score=0.8,
            overall_score=0.9,
        )

        result = SelectionResult(
            selected_method_name="best",
            selected_method_index=0,
            all_performances=[perf],
            selection_reason="Best accuracy",
            selection_timestamp="2025-01-01T00:00:00",
        )

        assert result.selected_method_name == "best"
        assert result.selected_method_index == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        perf = MethodPerformance(
            method_name="best",
            accuracy_score=0.9,
            consistency_score=1.0,
            efficiency_score=0.8,
            overall_score=0.9,
        )

        result = SelectionResult(
            selected_method_name="best",
            selected_method_index=0,
            all_performances=[perf],
            selection_reason="Best accuracy",
            selection_timestamp="2025-01-01T00:00:00",
        )

        d = result.to_dict()

        assert d["selected_method_name"] == "best"
        assert len(d["all_performances"]) == 1


# =============================================================================
# ReconcilerSelector Tests
# =============================================================================


class TestReconcilerSelector:
    """Tests for ReconcilerSelector class."""

    def test_initialization(self, mock_reconciler):
        """Test selector initialization."""
        methods = [mock_reconciler("method_1"), mock_reconciler("method_2")]
        selector = ReconcilerSelector(methods=methods)

        assert len(selector.methods) == 2
        assert "method_1" in selector.method_names
        assert "method_2" in selector.method_names

    def test_initialization_empty_methods(self):
        """Test that empty methods list raises error."""
        with pytest.raises(ValueError, match="At least one method"):
            ReconcilerSelector(methods=[])

    def test_initialization_with_config(self, mock_reconciler):
        """Test initialization with custom config."""
        methods = [mock_reconciler("m1")]
        config = SelectorConfig(
            accuracy_weight=0.8,
            consistency_weight=0.1,
            efficiency_weight=0.1,
        )
        selector = ReconcilerSelector(methods=methods, config=config)

        assert abs(selector.config.accuracy_weight - 0.8) < 1e-10

    def test_initialization_with_dict_config(self, mock_reconciler):
        """Test initialization with dict config."""
        methods = [mock_reconciler("m1")]
        config = {"accuracy_weight": 0.5, "rolling_window_size": 5}
        selector = ReconcilerSelector(methods=methods, config=config)

        assert selector.config.rolling_window_size == 5

    def test_select_method(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test method selection."""
        # Create methods with different simulated performance
        method1 = mock_reconciler("fast_method", reconcile_time=0.1, coherence_error=0.0)
        method2 = mock_reconciler("slow_method", reconcile_time=1.0, coherence_error=0.0)

        selector = ReconcilerSelector(methods=[method1, method2])

        selected = selector.select_method(
            base_forecasts=sample_forecasts,
            hierarchy=sample_hierarchy,
            actuals=sample_actuals,
        )

        assert selected in [method1, method2]
        assert selector.get_last_selection() is not None

    def test_select_method_stores_history(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test that selection stores history."""
        methods = [mock_reconciler("m1"), mock_reconciler("m2")]
        selector = ReconcilerSelector(methods=methods)

        # Run multiple selections
        for _ in range(3):
            selector.select_method(
                base_forecasts=sample_forecasts,
                hierarchy=sample_hierarchy,
                actuals=sample_actuals,
            )

        assert len(selector.performance_history) == 3

    def test_select_method_trims_history(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test that history is trimmed to window size."""
        methods = [mock_reconciler("m1")]
        config = SelectorConfig(rolling_window_size=3)
        selector = ReconcilerSelector(methods=methods, config=config)

        # Run more selections than window size
        for _ in range(5):
            selector.select_method(
                base_forecasts=sample_forecasts,
                hierarchy=sample_hierarchy,
                actuals=sample_actuals,
            )

        assert len(selector.performance_history) == 3

    def test_selection_counts(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test selection count tracking."""
        methods = [mock_reconciler("m1"), mock_reconciler("m2")]
        selector = ReconcilerSelector(methods=methods)

        # Run several selections
        for _ in range(5):
            selector.select_method(
                base_forecasts=sample_forecasts,
                hierarchy=sample_hierarchy,
                actuals=sample_actuals,
            )

        counts = selector.get_selection_counts()
        total = sum(counts.values())
        assert total == 5

    def test_method_rankings(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test method rankings calculation."""
        methods = [mock_reconciler("m1"), mock_reconciler("m2")]
        selector = ReconcilerSelector(methods=methods)

        # Run selection
        selector.select_method(
            base_forecasts=sample_forecasts,
            hierarchy=sample_hierarchy,
            actuals=sample_actuals,
        )

        rankings = selector.get_method_rankings()

        assert len(rankings) == 2
        # Rankings should be sorted by score descending
        assert rankings[0][1] >= rankings[1][1]

    def test_reset_history(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test history reset."""
        methods = [mock_reconciler("m1")]
        selector = ReconcilerSelector(methods=methods)

        selector.select_method(
            base_forecasts=sample_forecasts,
            hierarchy=sample_hierarchy,
            actuals=sample_actuals,
        )

        assert len(selector.performance_history) > 0

        selector.reset_history()

        assert len(selector.performance_history) == 0
        assert selector.get_last_selection() is None


# =============================================================================
# Scoring Function Tests
# =============================================================================


class TestScoringFunctions:
    """Tests for scoring functions."""

    def test_score_accuracy_perfect(self, mock_reconciler):
        """Test accuracy score with zero MAPE."""
        methods = [mock_reconciler("m1")]
        selector = ReconcilerSelector(methods=methods)

        score = selector._score_accuracy(0.0)
        assert score == 1.0

    def test_score_accuracy_high_mape(self, mock_reconciler):
        """Test accuracy score with high MAPE."""
        methods = [mock_reconciler("m1")]
        selector = ReconcilerSelector(methods=methods)

        score = selector._score_accuracy(1.0)
        assert score == 0.0

    def test_score_accuracy_threshold(self, mock_reconciler):
        """Test accuracy score at threshold."""
        methods = [mock_reconciler("m1")]
        config = SelectorConfig(mape_threshold=0.05)
        selector = ReconcilerSelector(methods=methods, config=config)

        # At threshold, score should be ~0.5
        score = selector._score_accuracy(0.05)
        assert 0.4 < score < 0.6

    def test_score_consistency_perfect(self, mock_reconciler):
        """Test consistency score with zero coherence error."""
        methods = [mock_reconciler("m1")]
        selector = ReconcilerSelector(methods=methods)

        score = selector._score_consistency(0.0)
        assert score == 1.0

    def test_score_consistency_large_error(self, mock_reconciler):
        """Test consistency score with large coherence error."""
        methods = [mock_reconciler("m1")]
        selector = ReconcilerSelector(methods=methods)

        score = selector._score_consistency(1000.0)
        assert score < 0.01

    def test_score_efficiency_fast(self, mock_reconciler):
        """Test efficiency score with fast computation."""
        methods = [mock_reconciler("m1")]
        config = SelectorConfig(target_computation_time=5.0)
        selector = ReconcilerSelector(methods=methods, config=config)

        score = selector._score_efficiency(0.1)
        # Fast computation should have score > 0.7
        assert score > 0.7

    def test_score_efficiency_slow(self, mock_reconciler):
        """Test efficiency score with slow computation."""
        methods = [mock_reconciler("m1")]
        config = SelectorConfig(target_computation_time=5.0)
        selector = ReconcilerSelector(methods=methods, config=config)

        score = selector._score_efficiency(50.0)
        assert score < 0.1


# =============================================================================
# Error Metric Tests
# =============================================================================


class TestErrorMetrics:
    """Tests for error metric calculations."""

    def test_calculate_mape(self, mock_reconciler):
        """Test MAPE calculation."""
        methods = [mock_reconciler("m1")]
        selector = ReconcilerSelector(methods=methods)

        reconciled = pd.DataFrame({"A": [100, 200, 300]})
        actuals = pd.DataFrame({"A": [100, 200, 300]})

        mape = selector._calculate_mape(reconciled, actuals)
        assert mape == 0.0

    def test_calculate_mape_with_error(self, mock_reconciler):
        """Test MAPE calculation with error."""
        methods = [mock_reconciler("m1")]
        selector = ReconcilerSelector(methods=methods)

        reconciled = pd.DataFrame({"A": [110, 220, 330]})
        actuals = pd.DataFrame({"A": [100, 200, 300]})

        mape = selector._calculate_mape(reconciled, actuals)
        assert abs(mape - 0.1) < 0.01  # ~10% error

    def test_calculate_mae(self, mock_reconciler):
        """Test MAE calculation."""
        methods = [mock_reconciler("m1")]
        selector = ReconcilerSelector(methods=methods)

        reconciled = pd.DataFrame({"A": [100, 200, 300]})
        actuals = pd.DataFrame({"A": [105, 195, 305]})

        mae = selector._calculate_mae(reconciled, actuals)
        assert mae == 5.0

    def test_calculate_rmse(self, mock_reconciler):
        """Test RMSE calculation."""
        methods = [mock_reconciler("m1")]
        selector = ReconcilerSelector(methods=methods)

        reconciled = pd.DataFrame({"A": [100, 200]})
        actuals = pd.DataFrame({"A": [103, 196]})

        rmse = selector._calculate_rmse(reconciled, actuals)
        expected = np.sqrt((9 + 16) / 2)
        assert abs(rmse - expected) < 0.01


# =============================================================================
# Report Generation Tests
# =============================================================================


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test report generation."""
        methods = [mock_reconciler("m1"), mock_reconciler("m2")]
        selector = ReconcilerSelector(methods=methods)

        selector.select_method(
            base_forecasts=sample_forecasts,
            hierarchy=sample_hierarchy,
            actuals=sample_actuals,
        )

        report = selector.generate_report()

        assert "n_methods" in report
        assert report["n_methods"] == 2
        assert "method_names" in report
        assert "config" in report
        assert "selection_counts" in report
        assert "rankings" in report
        assert "last_selection" in report

    def test_generate_report_with_statistics(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test report with method statistics."""
        methods = [mock_reconciler("m1")]
        selector = ReconcilerSelector(methods=methods)

        # Run multiple selections
        for _ in range(3):
            selector.select_method(
                base_forecasts=sample_forecasts,
                hierarchy=sample_hierarchy,
                actuals=sample_actuals,
            )

        report = selector.generate_report()

        assert "method_statistics" in report
        assert "m1" in report["method_statistics"]
        stats = report["method_statistics"]["m1"]
        assert "avg_accuracy" in stats
        assert "avg_consistency" in stats
        assert "avg_efficiency" in stats


# =============================================================================
# Save/Load Tests
# =============================================================================


class TestSaveLoad:
    """Tests for save/load functionality."""

    def test_save_and_load(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test saving and loading selector state."""
        methods = [mock_reconciler("m1"), mock_reconciler("m2")]
        selector = ReconcilerSelector(methods=methods)

        # Run some selections
        selector.select_method(
            base_forecasts=sample_forecasts,
            hierarchy=sample_hierarchy,
            actuals=sample_actuals,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "selector.json"
            selector.save(filepath)

            # Create new methods with same names
            loaded_methods = [mock_reconciler("m1"), mock_reconciler("m2")]
            loaded = ReconcilerSelector.load(filepath, methods=loaded_methods)

            assert loaded.method_names == selector.method_names
            assert loaded._selection_counts == selector._selection_counts
            assert len(loaded.performance_history) == len(selector.performance_history)

    def test_load_nonexistent_file(self, mock_reconciler):
        """Test loading from nonexistent file."""
        methods = [mock_reconciler("m1")]

        with pytest.raises(FileNotFoundError):
            ReconcilerSelector.load("/nonexistent/path.json", methods=methods)

    def test_load_method_mismatch(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test loading with mismatched methods."""
        methods = [mock_reconciler("m1"), mock_reconciler("m2")]
        selector = ReconcilerSelector(methods=methods)

        selector.select_method(
            base_forecasts=sample_forecasts,
            hierarchy=sample_hierarchy,
            actuals=sample_actuals,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "selector.json"
            selector.save(filepath)

            # Try to load with different methods
            wrong_methods = [mock_reconciler("m3")]
            with pytest.raises(ValueError, match="Method mismatch"):
                ReconcilerSelector.load(filepath, methods=wrong_methods)


# =============================================================================
# Rolling Average Tests
# =============================================================================


class TestRollingAverage:
    """Tests for rolling average functionality."""

    def test_rolling_scores_calculated(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test that rolling scores are calculated."""
        methods = [mock_reconciler("m1"), mock_reconciler("m2")]
        config = SelectorConfig(use_rolling_average=True, rolling_window_size=3)
        selector = ReconcilerSelector(methods=methods, config=config)

        # Run multiple selections
        for _ in range(3):
            selector.select_method(
                base_forecasts=sample_forecasts,
                hierarchy=sample_hierarchy,
                actuals=sample_actuals,
            )

        rolling_scores = selector._calculate_rolling_scores()
        assert len(rolling_scores) == 2

    def test_rolling_disabled(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test selection without rolling average."""
        methods = [mock_reconciler("m1")]
        config = SelectorConfig(use_rolling_average=False)
        selector = ReconcilerSelector(methods=methods, config=config)

        # Run selection
        selector.select_method(
            base_forecasts=sample_forecasts,
            hierarchy=sample_hierarchy,
            actuals=sample_actuals,
        )

        assert selector.get_last_selection() is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_method_failure_handled(
        self, sample_hierarchy, sample_forecasts, sample_actuals, mock_reconciler
    ):
        """Test handling of method failure during evaluation."""
        good_method = mock_reconciler("good")
        bad_method = mock_reconciler("bad")
        bad_method.reconcile = MagicMock(side_effect=RuntimeError("Method failed"))

        methods = [good_method, bad_method]
        selector = ReconcilerSelector(methods=methods)

        # Should still select the working method
        selected = selector.select_method(
            base_forecasts=sample_forecasts,
            hierarchy=sample_hierarchy,
            actuals=sample_actuals,
        )

        assert selected == good_method

    def test_all_methods_fail(self, sample_hierarchy, sample_forecasts, sample_actuals):
        """Test that error is raised when all methods fail."""
        bad_method = MagicMock()
        bad_method.name = "bad"
        bad_method.reconcile = MagicMock(side_effect=RuntimeError("Failed"))

        selector = ReconcilerSelector(methods=[bad_method])

        with pytest.raises(ValueError, match="All methods failed"):
            selector.select_method(
                base_forecasts=sample_forecasts,
                hierarchy=sample_hierarchy,
                actuals=sample_actuals,
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with real reconcilers."""

    def test_with_real_reconcilers(self, sample_hierarchy, sample_forecasts, sample_actuals):
        """Test with actual reconciler implementations."""
        from src.reconciliation.ols_reconciler import OLSReconciler
        from src.reconciliation.wls_reconciler import WLSReconciler

        # Create real reconcilers
        methods = [
            OLSReconciler(),
            WLSReconciler(),
        ]

        config = SelectorConfig(
            accuracy_weight=0.5,
            consistency_weight=0.3,
            efficiency_weight=0.2,
        )

        selector = ReconcilerSelector(methods=methods, config=config)

        # Select method
        selected = selector.select_method(
            base_forecasts=sample_forecasts,
            hierarchy=sample_hierarchy,
            actuals=sample_actuals,
        )

        assert selected in methods

        # Check selection result
        result = selector.get_last_selection()
        assert result is not None
        assert len(result.all_performances) == 2

        # Generate report
        report = selector.generate_report()
        assert report["n_methods"] == 2
