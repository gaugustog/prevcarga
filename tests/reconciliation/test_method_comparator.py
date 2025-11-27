# ruff: noqa: NPY002
"""Tests for ReconciliationComparator method comparison framework.

Tests cover:
- Cross-validation with time-aware splits
- Performance benchmarking
- Statistical significance testing
- Method ranking
- Recommendations generation
- Save/load functionality
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.reconciliation.data_structures import ReconciliationResult
from src.reconciliation.hierarchy import HierarchyDefinition
from src.reconciliation.method_comparator import (
    ComparatorConfig,
    ComparisonReport,
    CVFoldResult,
    PairwiseTest,
    ReconciliationComparator,
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
def sample_data():
    """Create sample forecast data with sufficient samples."""
    np.random.seed(42)
    n_timesteps = 100  # Enough for CV

    # Create bottom-level data
    city_a = 100 + np.random.randn(n_timesteps) * 5
    city_b = 150 + np.random.randn(n_timesteps) * 7
    city_c = 80 + np.random.randn(n_timesteps) * 4
    city_d = 120 + np.random.randn(n_timesteps) * 6

    # Base forecasts (with some error)
    forecasts = pd.DataFrame({
        "City_A": city_a + np.random.randn(n_timesteps) * 2,
        "City_B": city_b + np.random.randn(n_timesteps) * 3,
        "City_C": city_c + np.random.randn(n_timesteps) * 2,
        "City_D": city_d + np.random.randn(n_timesteps) * 2,
        "North": city_a + city_b + np.random.randn(n_timesteps) * 3,
        "South": city_c + city_d + np.random.randn(n_timesteps) * 3,
        "Total": city_a + city_b + city_c + city_d + np.random.randn(n_timesteps) * 5,
    })

    # Actuals (the truth)
    actuals = pd.DataFrame({
        "City_A": city_a,
        "City_B": city_b,
        "City_C": city_c,
        "City_D": city_d,
        "North": city_a + city_b,
        "South": city_c + city_d,
        "Total": city_a + city_b + city_c + city_d,
    })

    return forecasts, actuals


@pytest.fixture
def mock_reconciler():
    """Create a mock reconciler."""

    def create_mock(name: str, error_factor: float = 1.0):
        mock = MagicMock()
        mock.name = name

        def mock_reconcile(base_forecasts, hierarchy, **kwargs):
            result = MagicMock(spec=ReconciliationResult)
            # Return modified forecasts (add some random noise scaled by error_factor)
            reconciled = base_forecasts.copy()
            for col in reconciled.columns:
                reconciled[col] = reconciled[col] + np.random.randn(len(reconciled)) * error_factor
            result.reconciled_forecasts = reconciled
            return result

        mock.reconcile = mock_reconcile
        mock.check_coherence = MagicMock(return_value=0.001)

        return mock

    return create_mock


# =============================================================================
# ComparatorConfig Tests
# =============================================================================


class TestComparatorConfig:
    """Tests for ComparatorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = ComparatorConfig()

        assert config.cv_folds == 5
        assert config.significance_level == 0.05
        assert config.metric == "mape"

    def test_custom_values(self):
        """Test custom configuration."""
        config = ComparatorConfig(
            cv_folds=3,
            significance_level=0.01,
            metric="mae",
        )

        assert config.cv_folds == 3
        assert config.significance_level == 0.01
        assert config.metric == "mae"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ComparatorConfig()
        d = config.to_dict()

        assert "cv_folds" in d
        assert "significance_level" in d
        assert "metric" in d


# =============================================================================
# CVFoldResult Tests
# =============================================================================


class TestCVFoldResult:
    """Tests for CVFoldResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = CVFoldResult(
            fold_idx=0,
            method_name="test",
            mape=0.05,
            mae=10.0,
            rmse=15.0,
            coherence_error=0.001,
            computation_time=0.5,
        )

        assert result.fold_idx == 0
        assert result.method_name == "test"
        assert result.mape == 0.05

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = CVFoldResult(
            fold_idx=0,
            method_name="test",
            mape=0.05,
            mae=10.0,
            rmse=15.0,
            coherence_error=0.001,
            computation_time=0.5,
        )

        d = result.to_dict()

        assert d["method_name"] == "test"
        assert d["mape"] == 0.05


# =============================================================================
# PairwiseTest Tests
# =============================================================================


class TestPairwiseTest:
    """Tests for PairwiseTest dataclass."""

    def test_creation(self):
        """Test pairwise test creation."""
        test = PairwiseTest(
            method_a="A",
            method_b="B",
            ttest_statistic=2.5,
            ttest_pvalue=0.02,
            wilcoxon_statistic=15.0,
            wilcoxon_pvalue=0.03,
            significant=True,
            better_method="A",
        )

        assert test.method_a == "A"
        assert test.significant is True
        assert test.better_method == "A"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        test = PairwiseTest(
            method_a="A",
            method_b="B",
            ttest_statistic=2.5,
            ttest_pvalue=0.02,
            wilcoxon_statistic=15.0,
            wilcoxon_pvalue=0.03,
            significant=True,
            better_method="A",
        )

        d = test.to_dict()

        assert d["method_a"] == "A"
        assert d["significant"] is True


# =============================================================================
# ComparisonReport Tests
# =============================================================================


class TestComparisonReport:
    """Tests for ComparisonReport dataclass."""

    def test_creation(self):
        """Test report creation."""
        report = ComparisonReport(
            method_rankings={"A": 1, "B": 2},
            performance_matrix=[[0.05, 0.06], [0.08, 0.09]],
            method_stats={"A": {"mean": 0.055}},
            pairwise_tests=[],
            recommendations=["Use A"],
        )

        assert report.method_rankings["A"] == 1
        assert len(report.performance_matrix) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = ComparisonReport(
            method_rankings={"A": 1},
            performance_matrix=[[0.05]],
            method_stats={"A": {"mean": 0.05}},
            pairwise_tests=[],
            recommendations=["Test"],
        )

        d = report.to_dict()

        assert "method_rankings" in d
        assert "performance_matrix" in d


# =============================================================================
# ReconciliationComparator Tests
# =============================================================================


class TestReconciliationComparator:
    """Tests for ReconciliationComparator class."""

    def test_initialization(self):
        """Test comparator initialization."""
        comparator = ReconciliationComparator()

        assert comparator.config is not None
        assert comparator._last_report is None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = ComparatorConfig(cv_folds=3)
        comparator = ReconciliationComparator(config=config)

        assert comparator.config.cv_folds == 3

    def test_initialization_with_dict_config(self):
        """Test initialization with dict config."""
        config = {"cv_folds": 3, "significance_level": 0.01}
        comparator = ReconciliationComparator(config=config)

        assert comparator.config.cv_folds == 3
        assert comparator.config.significance_level == 0.01

    def test_compare_methods_empty_list(self, sample_hierarchy, sample_data):
        """Test that empty methods list raises error."""
        comparator = ReconciliationComparator()
        forecasts, actuals = sample_data

        with pytest.raises(ValueError, match="At least one method"):
            comparator.compare_methods([], forecasts, actuals, sample_hierarchy)

    def test_compare_methods_insufficient_data(
        self, sample_hierarchy, mock_reconciler
    ):
        """Test that insufficient data raises error."""
        comparator = ReconciliationComparator(
            config=ComparatorConfig(cv_folds=5, min_samples_per_fold=20)
        )

        # Only 10 samples, need 100 (5 folds * 20 per fold)
        forecasts = pd.DataFrame({"A": np.random.randn(10)})
        actuals = pd.DataFrame({"A": np.random.randn(10)})

        methods = [mock_reconciler("m1")]

        with pytest.raises(ValueError, match="Insufficient data"):
            comparator.compare_methods(methods, forecasts, actuals, sample_hierarchy)

    def test_compare_methods(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test method comparison."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [
            mock_reconciler("method_1", error_factor=1.0),
            mock_reconciler("method_2", error_factor=2.0),
        ]

        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        assert "method_1" in report.method_rankings
        assert "method_2" in report.method_rankings
        assert len(report.recommendations) > 0

    def test_compare_methods_stores_report(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test that comparison stores last report."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [mock_reconciler("m1")]

        comparator.compare_methods(methods, forecasts, actuals, sample_hierarchy)

        assert comparator.get_last_report() is not None

    def test_method_rankings(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test method rankings are correct."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data

        # Create methods with different error levels
        # method_1 should be better (lower error)
        methods = [
            mock_reconciler("better_method", error_factor=0.5),
            mock_reconciler("worse_method", error_factor=5.0),
        ]

        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        # Better method should have rank 1
        assert report.method_rankings["better_method"] < report.method_rankings["worse_method"]


# =============================================================================
# Cross-Validation Tests
# =============================================================================


class TestCrossValidation:
    """Tests for cross-validation functionality."""

    def test_cv_fold_results(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test CV fold results are generated."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [mock_reconciler("m1")]

        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        # Should have 3 fold results for 1 method
        assert len(report.cv_results) == 3

    def test_performance_matrix_shape(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test performance matrix has correct shape."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [mock_reconciler("m1"), mock_reconciler("m2")]

        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        # 2 methods x 3 folds
        assert len(report.performance_matrix) == 2
        assert len(report.performance_matrix[0]) == 3


# =============================================================================
# Statistical Tests
# =============================================================================


class TestStatisticalComparison:
    """Tests for statistical comparison."""

    def test_pairwise_tests_generated(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test pairwise tests are generated."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [
            mock_reconciler("m1"),
            mock_reconciler("m2"),
            mock_reconciler("m3"),
        ]

        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        # 3 methods -> 3 pairwise comparisons (3 choose 2)
        assert len(report.pairwise_tests) == 3

    def test_pairwise_test_structure(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test pairwise test structure."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [mock_reconciler("m1"), mock_reconciler("m2")]

        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        test = report.pairwise_tests[0]
        assert hasattr(test, "ttest_pvalue")
        assert hasattr(test, "wilcoxon_pvalue")
        assert hasattr(test, "significant")


# =============================================================================
# Recommendations Tests
# =============================================================================


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_recommendations_generated(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test recommendations are generated."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [mock_reconciler("m1")]

        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        assert len(report.recommendations) > 0

    def test_recommendations_include_best_method(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test recommendations mention best method."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [mock_reconciler("test_method")]

        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        # Should recommend the best method
        assert any("test_method" in rec for rec in report.recommendations)


# =============================================================================
# Method Statistics Tests
# =============================================================================


class TestMethodStatistics:
    """Tests for method statistics calculation."""

    def test_stats_calculated(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test statistics are calculated for each method."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [mock_reconciler("m1")]

        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        assert "m1" in report.method_stats
        stats = report.method_stats["m1"]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats


# =============================================================================
# Save/Load Tests
# =============================================================================


class TestSaveLoad:
    """Tests for save/load functionality."""

    def test_save_report(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test saving comparison report."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [mock_reconciler("m1")]

        comparator.compare_methods(methods, forecasts, actuals, sample_hierarchy)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            comparator.save_report(filepath)

            assert filepath.exists()

    def test_save_report_no_comparison(self):
        """Test save fails without comparison."""
        comparator = ReconciliationComparator()

        with pytest.raises(ValueError, match="No report to save"):
            comparator.save_report("/tmp/test.json")

    def test_load_report(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test loading comparison report."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [mock_reconciler("m1")]

        original_report = comparator.compare_methods(
            methods, forecasts, actuals, sample_hierarchy
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            comparator.save_report(filepath)

            loaded_report = ReconciliationComparator.load_report(filepath)

            assert loaded_report.method_rankings == original_report.method_rankings
            assert len(loaded_report.recommendations) == len(original_report.recommendations)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            ReconciliationComparator.load_report("/nonexistent/path.json")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_method_failure_handled(
        self, sample_hierarchy, sample_data, mock_reconciler
    ):
        """Test handling of method failure during CV."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data

        # Create a failing method
        good_method = mock_reconciler("good")
        bad_method = mock_reconciler("bad")
        bad_method.reconcile = MagicMock(side_effect=RuntimeError("Failed"))

        report = comparator.compare_methods(
            methods=[good_method, bad_method],
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        # Bad method should have inf scores
        assert report.method_stats["bad"]["n_failed_folds"] > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with real reconcilers."""

    def test_with_real_reconcilers(self, sample_hierarchy, sample_data):
        """Test with actual reconciler implementations."""
        from src.reconciliation.ols_reconciler import OLSReconciler
        from src.reconciliation.wls_reconciler import WLSReconciler

        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [OLSReconciler(), WLSReconciler()]

        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        assert len(report.method_rankings) == 2
        assert len(report.pairwise_tests) == 1
        assert report.timestamp != ""

    def test_full_workflow(self, sample_hierarchy, sample_data, mock_reconciler):
        """Test complete workflow."""
        config = ComparatorConfig(cv_folds=3, min_samples_per_fold=10)
        comparator = ReconciliationComparator(config=config)

        forecasts, actuals = sample_data
        methods = [
            mock_reconciler("m1"),
            mock_reconciler("m2"),
        ]

        # Run comparison
        report = comparator.compare_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        # Verify structure
        assert report.method_rankings
        assert report.performance_matrix
        assert report.method_stats
        assert report.pairwise_tests
        assert report.recommendations
        assert report.cv_results
        assert report.timestamp

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            comparator.save_report(filepath)

            loaded = ReconciliationComparator.load_report(filepath)
            assert loaded.method_rankings == report.method_rankings
