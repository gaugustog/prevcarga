"""Tests for the model performance analyzer module.

Tests cover:
- PerformanceAnalysisConfig validation and defaults
- IndividualModelAnalysis methods and serialization
- CombinationValidationResult methods
- ReconciliationImpactAnalysis methods
- DriftDetectionResult serialization
- ModelPerformanceAnalyzer comprehensive analysis
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.validation.comprehensive_backtester import (
    BacktestPeriod,
    BacktestPeriodResult,
    YearlyBacktestResult,
)
from src.validation.model_performance_analyzer import (
    CombinationValidationResult,
    DriftDetectionResult,
    IndividualModelAnalysis,
    ModelPerformanceAnalyzer,
    PerformanceAnalysisConfig,
    ReconciliationImpactAnalysis,
    StatisticalTestResult,
)


class TestPerformanceAnalysisConfig:
    """Tests for PerformanceAnalysisConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PerformanceAnalysisConfig()

        assert len(config.models) == 5
        assert len(config.areas) == 5
        assert config.significance_level == 0.05
        assert config.effect_size_threshold == 0.3
        assert config.drift_window_size == 10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PerformanceAnalysisConfig(
            models=["lgbm", "rf"],
            areas=["SECO", "S"],
            significance_level=0.01,
            effect_size_threshold=0.5,
            drift_window_size=20,
        )

        assert config.models == ["lgbm", "rf"]
        assert config.areas == ["SECO", "S"]
        assert config.significance_level == 0.01
        assert config.effect_size_threshold == 0.5
        assert config.drift_window_size == 20

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = PerformanceAnalysisConfig()
        result = config.to_dict()

        assert "models" in result
        assert "areas" in result
        assert "significance_level" in result
        assert "effect_size_threshold" in result


class TestStatisticalTestResult:
    """Tests for StatisticalTestResult dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = StatisticalTestResult(
            test_type="paired_t_test",
            statistic=2.5,
            p_value=0.02,
            is_significant=True,
            effect_size=0.8,
        )

        d = result.to_dict()

        assert d["test_type"] == "paired_t_test"
        assert d["statistic"] == 2.5
        assert d["p_value"] == 0.02
        assert d["is_significant"] is True
        assert d["effect_size"] == 0.8


class TestIndividualModelAnalysis:
    """Tests for IndividualModelAnalysis dataclass."""

    @pytest.fixture
    def sample_analysis(self) -> IndividualModelAnalysis:
        """Create sample analysis."""
        return IndividualModelAnalysis(
            performances={
                "lgbm": {"overall_mape": 3.5, "mape_std": 0.5},
                "rf": {"overall_mape": 4.0, "mape_std": 0.6},
                "arima": {"overall_mape": 5.0, "mape_std": 0.8},
            },
            rankings={"lgbm": 1, "rf": 2, "arima": 3},
            seasonal_patterns={
                "summer": {"best_model": "lgbm", "worst_model": "arima"}
            },
        )

    def test_get_best_model(self, sample_analysis: IndividualModelAnalysis) -> None:
        """Test getting best model."""
        assert sample_analysis.get_best_model() == "lgbm"

    def test_get_best_model_empty(self) -> None:
        """Test getting best model with empty rankings."""
        analysis = IndividualModelAnalysis(
            performances={},
            rankings={},
            seasonal_patterns={},
        )
        assert analysis.get_best_model() == ""

    def test_get_model_summary(self, sample_analysis: IndividualModelAnalysis) -> None:
        """Test getting model summary."""
        summary = sample_analysis.get_model_summary("lgbm")
        assert summary["overall_mape"] == 3.5

    def test_get_model_summary_not_found(
        self, sample_analysis: IndividualModelAnalysis
    ) -> None:
        """Test getting summary for non-existent model."""
        summary = sample_analysis.get_model_summary("nonexistent")
        assert summary == {}

    def test_to_dict(self, sample_analysis: IndividualModelAnalysis) -> None:
        """Test conversion to dictionary."""
        result = sample_analysis.to_dict()

        assert "performances" in result
        assert "rankings" in result
        assert "seasonal_patterns" in result
        assert result["best_model"] == "lgbm"


class TestCombinationValidationResult:
    """Tests for CombinationValidationResult dataclass."""

    @pytest.fixture
    def passing_result(self) -> CombinationValidationResult:
        """Create passing validation result."""
        return CombinationValidationResult(
            validations={
                "simple_average": {"passes_validation": True, "improvement": 0.5},
                "weighted_average": {"passes_validation": True, "improvement": 0.7},
            },
            best_combination="weighted_average",
            improvement_over_best_individual=0.7,
        )

    @pytest.fixture
    def failing_result(self) -> CombinationValidationResult:
        """Create failing validation result."""
        return CombinationValidationResult(
            validations={
                "simple_average": {"passes_validation": False, "improvement": -0.1},
            },
            best_combination=None,
            improvement_over_best_individual=-0.1,
        )

    def test_passes_validation_true(
        self, passing_result: CombinationValidationResult
    ) -> None:
        """Test passes validation when all pass."""
        assert passing_result.passes_validation()

    def test_passes_validation_false(
        self, failing_result: CombinationValidationResult
    ) -> None:
        """Test passes validation when some fail."""
        assert not failing_result.passes_validation()

    def test_passes_validation_empty(self) -> None:
        """Test passes validation with empty validations."""
        result = CombinationValidationResult(
            validations={},
            best_combination=None,
            improvement_over_best_individual=0.0,
        )
        assert not result.passes_validation()

    def test_to_dict(self, passing_result: CombinationValidationResult) -> None:
        """Test conversion to dictionary."""
        result = passing_result.to_dict()

        assert "validations" in result
        assert "best_combination" in result
        assert "improvement_over_best_individual" in result
        assert "passes_validation" in result


class TestReconciliationImpactAnalysis:
    """Tests for ReconciliationImpactAnalysis dataclass."""

    @pytest.fixture
    def sample_analysis(self) -> ReconciliationImpactAnalysis:
        """Create sample analysis."""
        return ReconciliationImpactAnalysis(
            area_impacts={
                "SECO": {"improvement": 0.5, "base_mape": 4.0, "reconciled_mape": 3.5},
                "S": {"improvement": -0.1, "base_mape": 3.0, "reconciled_mape": 3.1},
                "NE": {"improvement": 0.3, "base_mape": 5.0, "reconciled_mape": 4.7},
            },
            consistency_validation={"system_consistency": True},
            overall_improvement=0.23,
        )

    def test_get_improved_areas(
        self, sample_analysis: ReconciliationImpactAnalysis
    ) -> None:
        """Test getting improved areas."""
        improved = sample_analysis.get_improved_areas()
        assert "SECO" in improved
        assert "NE" in improved
        assert "S" not in improved

    def test_is_consistent_true(
        self, sample_analysis: ReconciliationImpactAnalysis
    ) -> None:
        """Test consistency check when consistent."""
        assert sample_analysis.is_consistent()

    def test_is_consistent_false(self) -> None:
        """Test consistency check when inconsistent."""
        analysis = ReconciliationImpactAnalysis(
            area_impacts={},
            consistency_validation={"system_consistency": False},
            overall_improvement=0.0,
        )
        assert not analysis.is_consistent()

    def test_to_dict(self, sample_analysis: ReconciliationImpactAnalysis) -> None:
        """Test conversion to dictionary."""
        result = sample_analysis.to_dict()

        assert "area_impacts" in result
        assert "consistency_validation" in result
        assert "overall_improvement" in result
        assert "improved_areas" in result
        assert "is_consistent" in result


class TestDriftDetectionResult:
    """Tests for DriftDetectionResult dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = DriftDetectionResult(
            model_drift={
                "lgbm": {
                    "mean_drift": {"drift_absolute": 0.5, "is_significant": True}
                }
            },
            overall_drift_detected=True,
            recommendations=["Consider retraining lgbm"],
        )

        d = result.to_dict()

        assert "model_drift" in d
        assert "overall_drift_detected" in d
        assert "recommendations" in d
        assert d["overall_drift_detected"] is True


class TestModelPerformanceAnalyzer:
    """Tests for ModelPerformanceAnalyzer class."""

    @pytest.fixture
    def config(self) -> PerformanceAnalysisConfig:
        """Create test configuration."""
        return PerformanceAnalysisConfig(
            models=["lgbm", "rf"],
            areas=["SECO", "S"],
        )

    @pytest.fixture
    def analyzer(self, config: PerformanceAnalysisConfig) -> ModelPerformanceAnalyzer:
        """Create analyzer instance."""
        return ModelPerformanceAnalyzer(config)

    @pytest.fixture
    def sample_backtest_results(self) -> YearlyBacktestResult:
        """Create sample backtest results."""
        period_results = []

        for i in range(10):
            period = BacktestPeriod(
                train_start=datetime(2023, 1, 1) + timedelta(days=i * 7),
                train_end=datetime(2023, 12, 31) + timedelta(days=i * 7),
                test_start=datetime(2024, 1, 1) + timedelta(days=i * 7),
                test_end=datetime(2024, 1, 7) + timedelta(days=i * 7),
                period_id=i,
            )

            evaluation = {
                "mape": {
                    "SECO_lgbm": 3.0 + np.random.uniform(-0.5, 0.5),
                    "SECO_rf": 4.0 + np.random.uniform(-0.5, 0.5),
                    "S_lgbm": 3.5 + np.random.uniform(-0.5, 0.5),
                    "S_rf": 4.5 + np.random.uniform(-0.5, 0.5),
                },
                "mae": {
                    "SECO_lgbm": 100 + np.random.uniform(-10, 10),
                    "SECO_rf": 120 + np.random.uniform(-10, 10),
                    "S_lgbm": 110 + np.random.uniform(-10, 10),
                    "S_rf": 130 + np.random.uniform(-10, 10),
                },
                "rmse": {
                    "SECO_lgbm": 150 + np.random.uniform(-15, 15),
                    "SECO_rf": 180 + np.random.uniform(-15, 15),
                    "S_lgbm": 160 + np.random.uniform(-15, 15),
                    "S_rf": 190 + np.random.uniform(-15, 15),
                },
            }

            period_result = BacktestPeriodResult(
                period=period,
                training_result={"duration": 10.0},
                predictions={"duration": 5.0},
                evaluation=evaluation,
                performance_metrics={},
                execution_time=20.0,
            )
            period_results.append(period_result)

        return YearlyBacktestResult(
            backtest_year=2024,
            total_periods=10,
            period_results=period_results,
            total_duration=200.0,
            average_mape=0.04,
            summary_metrics={},
        )

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        analyzer = ModelPerformanceAnalyzer()
        assert analyzer.config is not None
        assert len(analyzer.config.models) == 5

    def test_init_custom_config(
        self, config: PerformanceAnalysisConfig
    ) -> None:
        """Test initialization with custom config."""
        analyzer = ModelPerformanceAnalyzer(config)
        assert analyzer.config.models == ["lgbm", "rf"]

    def test_analyze_individual_models(
        self,
        analyzer: ModelPerformanceAnalyzer,
        sample_backtest_results: YearlyBacktestResult,
    ) -> None:
        """Test individual model analysis."""
        analysis = analyzer.analyze_individual_models(sample_backtest_results)

        assert isinstance(analysis, IndividualModelAnalysis)
        assert "lgbm" in analysis.performances
        assert "rf" in analysis.performances
        assert "lgbm" in analysis.rankings
        assert analysis.rankings["lgbm"] == 1  # lgbm should be better

    def test_analyze_individual_models_metrics(
        self,
        analyzer: ModelPerformanceAnalyzer,
        sample_backtest_results: YearlyBacktestResult,
    ) -> None:
        """Test that analysis includes all expected metrics."""
        analysis = analyzer.analyze_individual_models(sample_backtest_results)

        lgbm_perf = analysis.performances["lgbm"]

        assert "overall_mape" in lgbm_perf
        assert "mape_std" in lgbm_perf
        assert "overall_mae" in lgbm_perf
        assert "overall_rmse" in lgbm_perf
        assert "seasonal_performance" in lgbm_perf
        assert "area_performance" in lgbm_perf
        assert "stability_metrics" in lgbm_perf
        assert "outlier_analysis" in lgbm_perf

    def test_analyze_empty_backtest_results(
        self, analyzer: ModelPerformanceAnalyzer
    ) -> None:
        """Test analysis with empty backtest results."""
        empty_results = YearlyBacktestResult(
            backtest_year=2024,
            total_periods=0,
            period_results=[],
            total_duration=0.0,
            average_mape=0.0,
            summary_metrics={},
        )

        analysis = analyzer.analyze_individual_models(empty_results)

        assert analysis.performances == {}
        assert analysis.rankings == {}

    def test_validate_combination_effectiveness(
        self,
        analyzer: ModelPerformanceAnalyzer,
        sample_backtest_results: YearlyBacktestResult,
    ) -> None:
        """Test combination effectiveness validation."""
        individual_analysis = analyzer.analyze_individual_models(sample_backtest_results)

        combination_mapes = {
            "simple_average": 2.8,  # Better than best individual
            "weighted_average": 2.5,  # Even better
        }

        result = analyzer.validate_combination_effectiveness(
            individual_analysis,
            combination_mapes,
        )

        assert isinstance(result, CombinationValidationResult)
        assert result.best_combination == "weighted_average"
        assert result.improvement_over_best_individual > 0

    def test_validate_combination_with_samples(
        self,
        analyzer: ModelPerformanceAnalyzer,
        sample_backtest_results: YearlyBacktestResult,
    ) -> None:
        """Test combination validation with sample data."""
        individual_analysis = analyzer.analyze_individual_models(sample_backtest_results)

        combination_mapes = {
            "simple_average": 2.8,
        }

        combination_samples = {
            "simple_average": list(np.random.normal(2.8, 0.3, 30)),
        }

        result = analyzer.validate_combination_effectiveness(
            individual_analysis,
            combination_mapes,
            combination_samples,
        )

        assert "simple_average" in result.validations
        # Should have statistical significance results
        validation = result.validations["simple_average"]
        assert "statistical_significance" in validation

    def test_analyze_reconciliation_impact(
        self, analyzer: ModelPerformanceAnalyzer
    ) -> None:
        """Test reconciliation impact analysis."""
        base_forecasts = {
            "SECO": np.array([10000, 10500, 11000, 10200, 10800]),
            "S": np.array([5000, 5200, 5100, 4900, 5300]),
            "SIN": np.array([15000, 15700, 16100, 15100, 16100]),
        }

        reconciled_forecasts = {
            "SECO": np.array([10050, 10480, 10980, 10220, 10780]),
            "S": np.array([4950, 5220, 5120, 4880, 5320]),
            "SIN": np.array([15000, 15700, 16100, 15100, 16100]),
        }

        actuals = {
            "SECO": np.array([10100, 10400, 11050, 10250, 10750]),
            "S": np.array([4900, 5300, 5050, 4850, 5350]),
            "SIN": np.array([15000, 15700, 16100, 15100, 16100]),
        }

        result = analyzer.analyze_reconciliation_impact(
            base_forecasts,
            reconciled_forecasts,
            actuals,
        )

        assert isinstance(result, ReconciliationImpactAnalysis)
        assert "SECO" in result.area_impacts
        assert "S" in result.area_impacts

    def test_detect_performance_drift(
        self,
        analyzer: ModelPerformanceAnalyzer,
        sample_backtest_results: YearlyBacktestResult,
    ) -> None:
        """Test performance drift detection."""
        # Add more periods for drift detection
        extended_results = []
        for i in range(20):
            period = BacktestPeriod(
                train_start=datetime(2023, 1, 1) + timedelta(days=i * 7),
                train_end=datetime(2023, 12, 31) + timedelta(days=i * 7),
                test_start=datetime(2024, 1, 1) + timedelta(days=i * 7),
                test_end=datetime(2024, 1, 7) + timedelta(days=i * 7),
                period_id=i,
            )

            # Add drift: performance degrades over time
            drift_factor = i * 0.1

            evaluation = {
                "mape": {
                    "SECO_lgbm": 3.0 + drift_factor,
                    "SECO_rf": 4.0 + drift_factor,
                    "S_lgbm": 3.5 + drift_factor,
                    "S_rf": 4.5 + drift_factor,
                },
            }

            period_result = BacktestPeriodResult(
                period=period,
                training_result={"duration": 10.0},
                predictions={"duration": 5.0},
                evaluation=evaluation,
                performance_metrics={},
                execution_time=20.0,
            )
            extended_results.append(period_result)

        result = analyzer.detect_performance_drift(extended_results, window_size=5)

        assert isinstance(result, DriftDetectionResult)
        assert "lgbm" in result.model_drift
        # Should detect drift since we added degradation
        assert result.model_drift["lgbm"]["trend_analysis"]["is_degrading"]

    def test_compare_models_statistically(
        self, analyzer: ModelPerformanceAnalyzer
    ) -> None:
        """Test statistical comparison between models."""
        model1_samples = np.random.normal(3.0, 0.5, 30)
        model2_samples = np.random.normal(4.0, 0.5, 30)

        result = analyzer.compare_models_statistically(
            model1_samples,
            model2_samples,
            test_type="paired_t_test",
        )

        assert isinstance(result, StatisticalTestResult)
        assert result.test_type == "paired_t_test"
        assert result.is_significant  # Means are different

    def test_compare_models_wilcoxon(
        self, analyzer: ModelPerformanceAnalyzer
    ) -> None:
        """Test Wilcoxon test comparison."""
        model1_samples = np.random.normal(3.0, 0.5, 30)
        model2_samples = np.random.normal(4.0, 0.5, 30)

        result = analyzer.compare_models_statistically(
            model1_samples,
            model2_samples,
            test_type="wilcoxon",
        )

        assert result.test_type == "wilcoxon"

    def test_compare_models_mann_whitney(
        self, analyzer: ModelPerformanceAnalyzer
    ) -> None:
        """Test Mann-Whitney test comparison."""
        model1_samples = np.random.normal(3.0, 0.5, 30)
        model2_samples = np.random.normal(4.0, 0.5, 30)

        result = analyzer.compare_models_statistically(
            model1_samples,
            model2_samples,
            test_type="mann_whitney",
        )

        assert result.test_type == "mann_whitney"


class TestAnalyzerHelperMethods:
    """Tests for analyzer helper methods."""

    @pytest.fixture
    def analyzer(self) -> ModelPerformanceAnalyzer:
        """Create analyzer instance."""
        return ModelPerformanceAnalyzer(
            PerformanceAnalysisConfig(
                models=["lgbm", "rf"],
                areas=["SECO", "S"],
            )
        )

    def test_get_season_southern_hemisphere(
        self, analyzer: ModelPerformanceAnalyzer
    ) -> None:
        """Test season mapping for Southern Hemisphere."""
        assert analyzer._get_season(1) == "summer"
        assert analyzer._get_season(2) == "summer"
        assert analyzer._get_season(12) == "summer"
        assert analyzer._get_season(4) == "fall"
        assert analyzer._get_season(7) == "winter"
        assert analyzer._get_season(10) == "spring"

    def test_calculate_mape(self, analyzer: ModelPerformanceAnalyzer) -> None:
        """Test MAPE calculation."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([105, 195, 310])

        mape = analyzer._calculate_mape(predictions, actuals)

        assert mape > 0
        assert mape < 10

    def test_calculate_mape_empty(self, analyzer: ModelPerformanceAnalyzer) -> None:
        """Test MAPE with empty arrays."""
        mape = analyzer._calculate_mape(np.array([]), np.array([]))
        assert mape == 0.0

    def test_calculate_mae(self, analyzer: ModelPerformanceAnalyzer) -> None:
        """Test MAE calculation."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([110, 190, 310])

        mae = analyzer._calculate_mae(predictions, actuals)

        assert mae == 10.0

    def test_calculate_effect_size(self, analyzer: ModelPerformanceAnalyzer) -> None:
        """Test Cohen's d effect size calculation."""
        sample1 = np.array([3.0, 3.5, 3.2, 2.8, 3.3])
        sample2 = np.array([4.0, 4.5, 4.2, 3.8, 4.3])

        effect_size = analyzer._calculate_effect_size(sample1, sample2)

        # Should be a large negative effect (sample1 < sample2)
        assert effect_size < 0

    def test_rank_models(self, analyzer: ModelPerformanceAnalyzer) -> None:
        """Test model ranking."""
        performances = {
            "model_a": {"overall_mape": 5.0},
            "model_b": {"overall_mape": 3.0},
            "model_c": {"overall_mape": 4.0},
        }

        rankings = analyzer._rank_models(performances)

        assert rankings["model_b"] == 1
        assert rankings["model_c"] == 2
        assert rankings["model_a"] == 3

    def test_rank_models_empty(self, analyzer: ModelPerformanceAnalyzer) -> None:
        """Test ranking with empty performances."""
        rankings = analyzer._rank_models({})
        assert rankings == {}

    def test_calculate_stability_metrics(
        self, analyzer: ModelPerformanceAnalyzer
    ) -> None:
        """Test stability metrics calculation."""
        metrics = {
            "mape": [3.0, 3.5, 3.2, 2.8, 3.3, 3.1, 3.4, 3.0, 3.2, 3.1],
            "dates": [],
            "areas": [],
        }

        stability = analyzer._calculate_stability_metrics(metrics)

        assert "coefficient_of_variation" in stability
        assert "range" in stability
        assert "iqr" in stability
        assert "stability_score" in stability
        assert stability["coefficient_of_variation"] > 0

    def test_analyze_outliers(self, analyzer: ModelPerformanceAnalyzer) -> None:
        """Test outlier analysis."""
        metrics = {
            "mape": [3.0, 3.1, 3.2, 3.0, 3.1, 10.0, 3.2, 3.0, 3.1, 0.5],  # 10.0 and 0.5 are outliers
            "dates": [],
            "areas": [],
        }

        outliers = analyzer._analyze_outliers(metrics)

        assert outliers["outlier_count"] >= 1
        assert "outlier_percentage" in outliers
        assert "lower_bound" in outliers
        assert "upper_bound" in outliers

    def test_detect_mean_drift(self, analyzer: ModelPerformanceAnalyzer) -> None:
        """Test mean drift detection."""
        # Create series with drift
        time_series = np.array([3.0, 3.1, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0])

        drift = analyzer._detect_mean_drift(time_series, window_size=3)

        assert "drift_absolute" in drift
        assert "drift_percentage" in drift
        assert "is_significant" in drift
        assert drift["drift_absolute"] > 0

    def test_detect_variance_drift(self, analyzer: ModelPerformanceAnalyzer) -> None:
        """Test variance drift detection."""
        # Create series with increasing variance
        time_series = np.array([3.0, 3.1, 3.0, 3.5, 2.5, 4.0, 2.0, 5.0, 1.5, 5.5])

        drift = analyzer._detect_variance_drift(time_series, window_size=3)

        assert "variance_drift" in drift
        assert "is_increasing" in drift

    def test_analyze_performance_trend(
        self, analyzer: ModelPerformanceAnalyzer
    ) -> None:
        """Test performance trend analysis."""
        # Degrading performance
        time_series = np.array([3.0, 3.2, 3.5, 3.8, 4.0, 4.3, 4.5, 4.8, 5.0, 5.2])

        trend = analyzer._analyze_performance_trend(time_series)

        assert "trend_slope" in trend
        assert "is_improving" in trend
        assert "is_degrading" in trend
        assert trend["is_degrading"]  # MAPE increasing = degrading

    def test_detect_anomaly_periods(self, analyzer: ModelPerformanceAnalyzer) -> None:
        """Test anomaly period detection."""
        time_series = np.array([3.0, 3.1, 3.2, 10.0, 3.0, 3.1, 3.2, 0.5, 3.0, 3.1])

        anomalies = analyzer._detect_anomaly_periods(time_series)

        assert len(anomalies) >= 1  # Should detect at least one outlier (10.0)


class TestEdgeCases:
    """Edge case tests."""

    def test_hierarchical_consistency_no_sin(self) -> None:
        """Test consistency check without SIN area."""
        analyzer = ModelPerformanceAnalyzer()

        forecasts = {
            "SECO": np.array([10000, 10500]),
            "S": np.array([5000, 5200]),
        }

        consistency = analyzer._validate_hierarchical_consistency(forecasts)
        assert consistency["system_consistency"]

    def test_coherence_calculation_without_subsystems(self) -> None:
        """Test coherence calculation without subsystems."""
        analyzer = ModelPerformanceAnalyzer()

        forecasts = {
            "SIN": np.array([15000, 15700]),
        }

        coherence = analyzer._calculate_coherence(forecasts, "SIN")
        assert coherence == 1.0

    def test_statistical_test_insufficient_samples(self) -> None:
        """Test statistical test with insufficient samples."""
        analyzer = ModelPerformanceAnalyzer()

        sample1 = np.array([3.0])
        sample2 = np.array([4.0])

        result = analyzer._test_significance(sample1, sample2)

        assert result.p_value == 1.0
        assert not result.is_significant
