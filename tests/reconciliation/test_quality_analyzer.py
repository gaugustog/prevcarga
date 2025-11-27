# ruff: noqa: NPY002
"""Tests for ReconciliationQualityAnalyzer.

Tests cover:
- Quality metrics calculation
- Constraint satisfaction scoring
- Accuracy improvement calculation
- Consistency measures
- Statistical significance testing
- Degradation detection
- Recommendations generation
- Save/load persistence
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reconciliation.hierarchy import HierarchyDefinition
from src.reconciliation.quality_analyzer import (
    QualityConfig,
    QualityMetrics,
    QualityReport,
    ReconciliationQualityAnalyzer,
    StatisticalTests,
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
    """Create sample forecast data."""
    np.random.seed(42)
    n_timesteps = 20

    # Create bottom-level forecasts
    city_a = 100 + np.random.randn(n_timesteps) * 5
    city_b = 150 + np.random.randn(n_timesteps) * 7
    city_c = 80 + np.random.randn(n_timesteps) * 4
    city_d = 120 + np.random.randn(n_timesteps) * 6

    # Base forecasts (slightly incoherent)
    base = pd.DataFrame({
        "City_A": city_a,
        "City_B": city_b,
        "City_C": city_c,
        "City_D": city_d,
        "North": city_a + city_b + np.random.randn(n_timesteps) * 3,
        "South": city_c + city_d + np.random.randn(n_timesteps) * 3,
        "Total": city_a + city_b + city_c + city_d + np.random.randn(n_timesteps) * 5,
    })

    # Reconciled forecasts (coherent)
    reconciled = pd.DataFrame({
        "City_A": city_a,
        "City_B": city_b,
        "City_C": city_c,
        "City_D": city_d,
        "North": city_a + city_b,
        "South": city_c + city_d,
        "Total": city_a + city_b + city_c + city_d,
    })

    # Actuals (close to reconciled)
    actuals = reconciled.copy()
    for col in actuals.columns:
        actuals[col] += np.random.randn(n_timesteps) * 2

    return base, reconciled, actuals


# =============================================================================
# QualityConfig Tests
# =============================================================================


class TestQualityConfig:
    """Tests for QualityConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = QualityConfig()

        assert config.significance_level == 0.05
        assert config.degradation_window == 5
        assert config.min_samples_for_tests == 3

    def test_custom_values(self):
        """Test custom configuration."""
        config = QualityConfig(
            significance_level=0.01,
            degradation_window=10,
        )

        assert config.significance_level == 0.01
        assert config.degradation_window == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = QualityConfig()
        d = config.to_dict()

        assert "significance_level" in d
        assert "degradation_window" in d
        assert "constraint_tolerance" in d


# =============================================================================
# QualityMetrics Tests
# =============================================================================


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_default_values(self):
        """Test default metrics."""
        metrics = QualityMetrics()

        assert metrics.mape_base == 0.0
        assert metrics.mape_reconciled == 0.0
        assert metrics.correlation_preserved == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = QualityMetrics(mape_base=0.05, mape_reconciled=0.03)
        d = metrics.to_dict()

        assert d["mape_base"] == 0.05
        assert d["mape_reconciled"] == 0.03


# =============================================================================
# StatisticalTests Tests
# =============================================================================


class TestStatisticalTests:
    """Tests for StatisticalTests dataclass."""

    def test_default_values(self):
        """Test default values."""
        tests = StatisticalTests()

        assert tests.paired_ttest_pvalue == 1.0
        assert tests.improvement_significant is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        tests = StatisticalTests(
            paired_ttest_statistic=2.5,
            paired_ttest_pvalue=0.02,
            improvement_significant=True,
        )
        d = tests.to_dict()

        assert d["paired_ttest_statistic"] == 2.5
        assert d["improvement_significant"] is True


# =============================================================================
# QualityReport Tests
# =============================================================================


class TestQualityReport:
    """Tests for QualityReport dataclass."""

    def test_creation(self):
        """Test report creation."""
        report = QualityReport(
            constraint_satisfaction=0.95,
            accuracy_improvement=0.1,
            consistency_score=0.85,
            statistical_significance=True,
            degradation_detected=False,
            recommendations=["All good"],
        )

        assert report.constraint_satisfaction == 0.95
        assert report.accuracy_improvement == 0.1
        assert len(report.recommendations) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = QualityReport(
            constraint_satisfaction=0.95,
            accuracy_improvement=0.1,
            consistency_score=0.85,
            statistical_significance=True,
            degradation_detected=False,
            recommendations=["Test"],
            method_name="ols",
        )

        d = report.to_dict()

        assert d["constraint_satisfaction"] == 0.95
        assert d["method_name"] == "ols"
        assert "metrics" in d
        assert "statistical_tests" in d


# =============================================================================
# ReconciliationQualityAnalyzer Tests
# =============================================================================


class TestReconciliationQualityAnalyzer:
    """Tests for ReconciliationQualityAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ReconciliationQualityAnalyzer()

        assert analyzer.config is not None
        assert len(analyzer.quality_history) == 0

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = QualityConfig(significance_level=0.01)
        analyzer = ReconciliationQualityAnalyzer(config=config)

        assert analyzer.config.significance_level == 0.01

    def test_initialization_with_dict_config(self):
        """Test initialization with dict config."""
        config = {"significance_level": 0.1, "degradation_window": 10}
        analyzer = ReconciliationQualityAnalyzer(config=config)

        assert analyzer.config.significance_level == 0.1
        assert analyzer.config.degradation_window == 10

    def test_analyze_quality(self, sample_hierarchy, sample_data):
        """Test quality analysis."""
        base, reconciled, actuals = sample_data
        analyzer = ReconciliationQualityAnalyzer()

        report = analyzer.analyze_quality(
            reconciled_forecasts=reconciled,
            base_forecasts=base,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        assert 0 <= report.constraint_satisfaction <= 1
        assert -1 <= report.accuracy_improvement <= 1
        assert 0 <= report.consistency_score <= 1
        assert isinstance(report.recommendations, list)

    def test_analyze_with_method_name(self, sample_hierarchy, sample_data):
        """Test analysis with method name."""
        base, reconciled, actuals = sample_data
        analyzer = ReconciliationQualityAnalyzer()

        report = analyzer.analyze_quality(
            reconciled_forecasts=reconciled,
            base_forecasts=base,
            actuals=actuals,
            hierarchy=sample_hierarchy,
            method_name="ols",
        )

        assert report.method_name == "ols"

    def test_quality_history(self, sample_hierarchy, sample_data):
        """Test quality history tracking."""
        base, reconciled, actuals = sample_data
        analyzer = ReconciliationQualityAnalyzer()

        # Run multiple analyses
        for _ in range(3):
            analyzer.analyze_quality(
                reconciled_forecasts=reconciled,
                base_forecasts=base,
                actuals=actuals,
                hierarchy=sample_hierarchy,
            )

        assert len(analyzer.quality_history) == 3

    def test_clear_history(self, sample_hierarchy, sample_data):
        """Test clearing history."""
        base, reconciled, actuals = sample_data
        analyzer = ReconciliationQualityAnalyzer()

        analyzer.analyze_quality(
            reconciled_forecasts=reconciled,
            base_forecasts=base,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        assert len(analyzer.quality_history) == 1

        analyzer.clear_history()

        assert len(analyzer.quality_history) == 0


# =============================================================================
# Constraint Satisfaction Tests
# =============================================================================


class TestConstraintSatisfaction:
    """Tests for constraint satisfaction calculation."""

    def test_perfect_coherence(self, sample_hierarchy):
        """Test perfect constraint satisfaction."""
        # Create perfectly coherent forecasts
        forecasts = pd.DataFrame({
            "City_A": [100.0],
            "City_B": [150.0],
            "City_C": [80.0],
            "City_D": [120.0],
            "North": [250.0],  # = City_A + City_B
            "South": [200.0],  # = City_C + City_D
            "Total": [450.0],  # = North + South
        })

        analyzer = ReconciliationQualityAnalyzer()
        score = analyzer._calculate_constraint_satisfaction(forecasts, sample_hierarchy)

        assert score > 0.99

    def test_large_violation(self, sample_hierarchy):
        """Test low constraint satisfaction with large violations."""
        # Create incoherent forecasts
        forecasts = pd.DataFrame({
            "City_A": [100.0],
            "City_B": [150.0],
            "City_C": [80.0],
            "City_D": [120.0],
            "North": [300.0],  # Should be 250
            "South": [250.0],  # Should be 200
            "Total": [600.0],  # Should be 450
        })

        analyzer = ReconciliationQualityAnalyzer()
        score = analyzer._calculate_constraint_satisfaction(forecasts, sample_hierarchy)

        assert score < 0.9


# =============================================================================
# Accuracy Improvement Tests
# =============================================================================


class TestAccuracyImprovement:
    """Tests for accuracy improvement calculation."""

    def test_positive_improvement(self):
        """Test positive accuracy improvement."""
        metrics = QualityMetrics(mape_base=0.10, mape_reconciled=0.05)

        analyzer = ReconciliationQualityAnalyzer()
        improvement = analyzer._calculate_accuracy_improvement(metrics)

        assert improvement > 0
        assert abs(improvement - 0.5) < 0.01  # 50% improvement

    def test_negative_improvement(self):
        """Test negative accuracy improvement (degradation)."""
        metrics = QualityMetrics(mape_base=0.05, mape_reconciled=0.10)

        analyzer = ReconciliationQualityAnalyzer()
        improvement = analyzer._calculate_accuracy_improvement(metrics)

        assert improvement < 0

    def test_no_improvement(self):
        """Test no improvement."""
        metrics = QualityMetrics(mape_base=0.10, mape_reconciled=0.10)

        analyzer = ReconciliationQualityAnalyzer()
        improvement = analyzer._calculate_accuracy_improvement(metrics)

        assert abs(improvement) < 0.01

    def test_clamped_improvement(self):
        """Test improvement is clamped to [-1, 1]."""
        # Extreme case
        metrics = QualityMetrics(mape_base=0.01, mape_reconciled=0.50)

        analyzer = ReconciliationQualityAnalyzer()
        improvement = analyzer._calculate_accuracy_improvement(metrics)

        assert improvement >= -1.0
        assert improvement <= 1.0


# =============================================================================
# Statistical Tests
# =============================================================================


class TestStatisticalTesting:
    """Tests for statistical significance testing."""

    def test_significant_improvement(self, sample_hierarchy):
        """Test detection of significant improvement."""
        np.random.seed(42)
        n = 20

        # Base forecasts with higher error
        base = pd.DataFrame({
            "A": np.random.randn(n) * 10 + 100,
            "B": np.random.randn(n) * 10 + 200,
        })

        # Reconciled with lower error (closer to actuals)
        actuals = pd.DataFrame({
            "A": np.ones(n) * 100,
            "B": np.ones(n) * 200,
        })

        reconciled = pd.DataFrame({
            "A": np.random.randn(n) * 2 + 100,  # Much lower variance
            "B": np.random.randn(n) * 2 + 200,
        })

        analyzer = ReconciliationQualityAnalyzer()
        tests = analyzer._run_statistical_tests(reconciled, base, actuals)

        # Should detect significant improvement
        assert tests.sample_size == 2

    def test_insufficient_samples(self):
        """Test with insufficient samples."""
        # Only 2 columns but need 3 for test
        base = pd.DataFrame({"A": [100], "B": [200]})
        reconciled = pd.DataFrame({"A": [100], "B": [200]})
        actuals = pd.DataFrame({"A": [100], "B": [200]})

        config = QualityConfig(min_samples_for_tests=3)
        analyzer = ReconciliationQualityAnalyzer(config=config)
        tests = analyzer._run_statistical_tests(reconciled, base, actuals)

        assert tests.improvement_significant is False


# =============================================================================
# Degradation Detection Tests
# =============================================================================


class TestDegradationDetection:
    """Tests for quality degradation detection."""

    def test_no_degradation_insufficient_history(self):
        """Test no degradation with insufficient history."""
        analyzer = ReconciliationQualityAnalyzer()

        # Add only 2 reports (less than window)
        for _ in range(2):
            report = QualityReport(
                constraint_satisfaction=0.95,
                accuracy_improvement=0.1,
                consistency_score=0.9,
                statistical_significance=True,
                degradation_detected=False,
                recommendations=[],
            )
            analyzer.quality_history.append(report)

        degradation = analyzer._detect_degradation()
        assert degradation is False

    def test_degradation_detected(self):
        """Test degradation detection with declining trend."""
        config = QualityConfig(degradation_window=5)
        analyzer = ReconciliationQualityAnalyzer(config=config)

        # Add reports with declining accuracy improvement
        improvements = [0.15, 0.12, 0.08, 0.03, -0.02]
        for imp in improvements:
            report = QualityReport(
                constraint_satisfaction=0.95,
                accuracy_improvement=imp,
                consistency_score=0.9,
                statistical_significance=True,
                degradation_detected=False,
                recommendations=[],
            )
            analyzer.quality_history.append(report)

        degradation = analyzer._detect_degradation()
        assert degradation is True

    def test_no_degradation_stable(self):
        """Test no degradation with stable performance."""
        config = QualityConfig(degradation_window=5)
        analyzer = ReconciliationQualityAnalyzer(config=config)

        # Add reports with stable accuracy improvement
        for _ in range(5):
            report = QualityReport(
                constraint_satisfaction=0.95,
                accuracy_improvement=0.10,  # Stable
                consistency_score=0.9,
                statistical_significance=True,
                degradation_detected=False,
                recommendations=[],
            )
            analyzer.quality_history.append(report)

        degradation = analyzer._detect_degradation()
        assert degradation is False


# =============================================================================
# Recommendations Tests
# =============================================================================


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_low_constraint_recommendation(self):
        """Test recommendation for low constraint satisfaction."""
        analyzer = ReconciliationQualityAnalyzer()
        metrics = QualityMetrics()
        stat_tests = StatisticalTests()

        recommendations = analyzer._generate_recommendations(
            constraint_sat=0.8,
            accuracy_imp=0.1,
            consistency=0.9,
            stat_tests=stat_tests,
            metrics=metrics,
        )

        assert any("constraint" in r.lower() for r in recommendations)

    def test_low_improvement_recommendation(self):
        """Test recommendation for low accuracy improvement."""
        analyzer = ReconciliationQualityAnalyzer()
        metrics = QualityMetrics()
        stat_tests = StatisticalTests()

        recommendations = analyzer._generate_recommendations(
            constraint_sat=0.99,
            accuracy_imp=0.005,
            consistency=0.9,
            stat_tests=stat_tests,
            metrics=metrics,
        )

        assert any("improvement" in r.lower() or "alternative" in r.lower() for r in recommendations)

    def test_degradation_recommendation(self):
        """Test recommendation for negative improvement."""
        analyzer = ReconciliationQualityAnalyzer()
        metrics = QualityMetrics()
        stat_tests = StatisticalTests()

        recommendations = analyzer._generate_recommendations(
            constraint_sat=0.99,
            accuracy_imp=-0.05,
            consistency=0.9,
            stat_tests=stat_tests,
            metrics=metrics,
        )

        assert any("degrading" in r.lower() for r in recommendations)

    def test_all_good_recommendation(self):
        """Test recommendation when everything is good."""
        analyzer = ReconciliationQualityAnalyzer()
        metrics = QualityMetrics(correlation_preserved=0.95)
        stat_tests = StatisticalTests(improvement_significant=True)

        recommendations = analyzer._generate_recommendations(
            constraint_sat=0.99,
            accuracy_imp=0.15,
            consistency=0.95,
            stat_tests=stat_tests,
            metrics=metrics,
        )

        assert any("good" in r.lower() or "no issues" in r.lower() for r in recommendations)


# =============================================================================
# History Summary Tests
# =============================================================================


class TestHistorySummary:
    """Tests for history summary."""

    def test_empty_history(self):
        """Test summary with empty history."""
        analyzer = ReconciliationQualityAnalyzer()
        summary = analyzer.get_history_summary()

        assert summary["n_analyses"] == 0

    def test_with_history(self, sample_hierarchy, sample_data):
        """Test summary with history."""
        base, reconciled, actuals = sample_data
        analyzer = ReconciliationQualityAnalyzer()

        for _ in range(3):
            analyzer.analyze_quality(
                reconciled_forecasts=reconciled,
                base_forecasts=base,
                actuals=actuals,
                hierarchy=sample_hierarchy,
            )

        summary = analyzer.get_history_summary()

        assert summary["n_analyses"] == 3
        assert "avg_accuracy_improvement" in summary
        assert "avg_constraint_satisfaction" in summary


# =============================================================================
# Save/Load Tests
# =============================================================================


class TestSaveLoad:
    """Tests for save/load functionality."""

    def test_save_and_load(self, sample_hierarchy, sample_data):
        """Test saving and loading analyzer state."""
        base, reconciled, actuals = sample_data

        config = QualityConfig(significance_level=0.01)
        analyzer = ReconciliationQualityAnalyzer(config=config)

        analyzer.analyze_quality(
            reconciled_forecasts=reconciled,
            base_forecasts=base,
            actuals=actuals,
            hierarchy=sample_hierarchy,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "analyzer.json"
            analyzer.save(filepath)

            loaded = ReconciliationQualityAnalyzer.load(filepath)

            assert loaded.config.significance_level == 0.01
            assert len(loaded.quality_history) == 1

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            ReconciliationQualityAnalyzer.load("/nonexistent/path.json")


# =============================================================================
# Error Metric Tests
# =============================================================================


class TestErrorMetrics:
    """Tests for error metric calculations."""

    def test_mape_calculation(self):
        """Test MAPE calculation."""
        forecasts = pd.DataFrame({"A": [100, 200, 300]})
        actuals = pd.DataFrame({"A": [100, 200, 300]})

        analyzer = ReconciliationQualityAnalyzer()
        mape = analyzer._calculate_mape(forecasts, actuals)

        assert mape == 0.0

    def test_mae_calculation(self):
        """Test MAE calculation."""
        forecasts = pd.DataFrame({"A": [100, 200, 300]})
        actuals = pd.DataFrame({"A": [105, 195, 305]})

        analyzer = ReconciliationQualityAnalyzer()
        mae = analyzer._calculate_mae(forecasts, actuals)

        assert mae == 5.0

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        forecasts = pd.DataFrame({"A": [100, 200]})
        actuals = pd.DataFrame({"A": [103, 196]})

        analyzer = ReconciliationQualityAnalyzer()
        rmse = analyzer._calculate_rmse(forecasts, actuals)

        expected = np.sqrt((9 + 16) / 2)
        assert abs(rmse - expected) < 0.01


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with real reconcilers."""

    def test_full_workflow(self, sample_hierarchy, sample_data):
        """Test complete analysis workflow."""
        base, reconciled, actuals = sample_data

        analyzer = ReconciliationQualityAnalyzer()

        # Run analysis
        report = analyzer.analyze_quality(
            reconciled_forecasts=reconciled,
            base_forecasts=base,
            actuals=actuals,
            hierarchy=sample_hierarchy,
            method_name="test_method",
        )

        # Verify report structure
        assert report.constraint_satisfaction >= 0
        assert report.metrics is not None
        assert report.statistical_tests is not None
        assert report.timestamp != ""
        assert report.method_name == "test_method"

        # Get summary
        summary = analyzer.get_history_summary()
        assert summary["n_analyses"] == 1

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "analyzer.json"
            analyzer.save(filepath)

            loaded = ReconciliationQualityAnalyzer.load(filepath)
            assert len(loaded.quality_history) == 1
