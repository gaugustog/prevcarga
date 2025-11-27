"""Model performance analyzer for PrevCarga system.

This module provides comprehensive model performance analysis capabilities
including individual model analysis, combination validation, reconciliation
impact assessment, and performance drift detection.

Key Features:
- Individual model performance analysis
- Model combination effectiveness validation
- Hierarchical reconciliation impact assessment
- Performance drift detection
- Statistical significance testing

Example:
    ```python
    from src.validation.model_performance_analyzer import (
        ModelPerformanceAnalyzer,
        PerformanceAnalysisConfig,
    )

    config = PerformanceAnalysisConfig(
        models=["lgbm", "rf", "arima"],
        areas=["SECO", "S", "NE", "N", "SIN"],
    )
    analyzer = ModelPerformanceAnalyzer(config)

    # Analyze individual models
    analysis = analyzer.analyze_individual_models(backtest_results)
    print(f"Best model: {analysis.get_best_model()}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.validation.comprehensive_backtester import (
        BacktestPeriodResult,
        YearlyBacktestResult,
    )

logger = get_logger(__name__)


# Default areas and models
DEFAULT_AREAS = ["SECO", "S", "NE", "N", "SIN"]
DEFAULT_MODELS = ["lgbm", "rf", "arima", "holt_winters", "regdin_svm"]
DEFAULT_COMBINATION_METHODS = ["simple_average", "weighted_average", "best_model"]


@dataclass
class PerformanceAnalysisConfig:
    """Configuration for performance analysis.

    Attributes:
        models: List of models to analyze.
        areas: List of areas to analyze.
        combination_methods: List of combination methods to validate.
        significance_level: Statistical significance level (alpha).
        effect_size_threshold: Minimum effect size for practical significance.
        drift_window_size: Window size for drift detection.
    """

    models: list[str] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    areas: list[str] = field(default_factory=lambda: DEFAULT_AREAS.copy())
    combination_methods: list[str] = field(
        default_factory=lambda: DEFAULT_COMBINATION_METHODS.copy()
    )
    significance_level: float = 0.05
    effect_size_threshold: float = 0.3
    drift_window_size: int = 10

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "models": self.models,
            "areas": self.areas,
            "combination_methods": self.combination_methods,
            "significance_level": self.significance_level,
            "effect_size_threshold": self.effect_size_threshold,
            "drift_window_size": self.drift_window_size,
        }


@dataclass
class StatisticalTestResult:
    """Result from statistical significance test.

    Attributes:
        test_type: Type of test performed.
        statistic: Test statistic value.
        p_value: P-value from the test.
        is_significant: Whether result is statistically significant.
        effect_size: Calculated effect size (Cohen's d).
    """

    test_type: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "test_type": self.test_type,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "effect_size": self.effect_size,
        }


@dataclass
class IndividualModelAnalysis:
    """Analysis results for individual models.

    Attributes:
        performances: Dictionary of model performances.
        rankings: Model rankings (1 = best).
        seasonal_patterns: Seasonal performance patterns.
    """

    performances: dict[str, dict[str, Any]]
    rankings: dict[str, int]
    seasonal_patterns: dict[str, dict[str, Any]]

    def get_best_model(self) -> str:
        """Get best performing model.

        Returns:
            Name of best model.
        """
        if not self.rankings:
            return ""
        return min(self.rankings.items(), key=lambda x: x[1])[0]

    def get_model_summary(self, model: str) -> dict[str, Any]:
        """Get summary for specific model.

        Args:
            model: Model name.

        Returns:
            Model performance summary.
        """
        return self.performances.get(model, {})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "performances": self.performances,
            "rankings": self.rankings,
            "seasonal_patterns": self.seasonal_patterns,
            "best_model": self.get_best_model(),
        }


@dataclass
class CombinationValidationResult:
    """Results from combination effectiveness validation.

    Attributes:
        validations: Validation results by combination method.
        best_combination: Best performing combination method.
        improvement_over_best_individual: Improvement percentage.
    """

    validations: dict[str, dict[str, Any]]
    best_combination: str | None
    improvement_over_best_individual: float

    def passes_validation(self) -> bool:
        """Check if combinations pass validation criteria.

        Returns:
            True if all combinations pass validation.
        """
        if not self.validations:
            return False
        return all(v.get("passes_validation", False) for v in self.validations.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "validations": self.validations,
            "best_combination": self.best_combination,
            "improvement_over_best_individual": self.improvement_over_best_individual,
            "passes_validation": self.passes_validation(),
        }


@dataclass
class ReconciliationImpactAnalysis:
    """Analysis of reconciliation impact.

    Attributes:
        area_impacts: Impact metrics by area.
        consistency_validation: Consistency validation results.
        overall_improvement: Overall MAPE improvement.
    """

    area_impacts: dict[str, dict[str, float]]
    consistency_validation: dict[str, bool]
    overall_improvement: float

    def get_improved_areas(self) -> list[str]:
        """Get areas where reconciliation improved accuracy.

        Returns:
            List of improved area names.
        """
        return [
            area
            for area, metrics in self.area_impacts.items()
            if metrics.get("improvement", 0) > 0
        ]

    def is_consistent(self) -> bool:
        """Check if hierarchical consistency is maintained.

        Returns:
            True if all consistency checks pass.
        """
        return all(self.consistency_validation.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "area_impacts": self.area_impacts,
            "consistency_validation": self.consistency_validation,
            "overall_improvement": self.overall_improvement,
            "improved_areas": self.get_improved_areas(),
            "is_consistent": self.is_consistent(),
        }


@dataclass
class DriftDetectionResult:
    """Result from performance drift detection.

    Attributes:
        model_drift: Drift results by model.
        overall_drift_detected: Whether significant drift was detected.
        recommendations: Recommendations based on drift analysis.
    """

    model_drift: dict[str, dict[str, Any]]
    overall_drift_detected: bool
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model_drift": self.model_drift,
            "overall_drift_detected": self.overall_drift_detected,
            "recommendations": self.recommendations,
        }


class ModelPerformanceAnalyzer:
    """Analyzes and compares model performance with statistical rigor.

    This class provides comprehensive model performance analysis including
    individual model analysis, combination validation, reconciliation impact
    assessment, and performance drift detection.

    Attributes:
        config: Performance analysis configuration.
    """

    def __init__(self, config: PerformanceAnalysisConfig | None = None) -> None:
        """Initialize analyzer.

        Args:
            config: Performance analysis configuration.
        """
        self.config = config or PerformanceAnalysisConfig()

    def analyze_individual_models(
        self,
        backtest_results: YearlyBacktestResult,
    ) -> IndividualModelAnalysis:
        """Analyze performance of each model individually.

        Args:
            backtest_results: Results from yearly backtesting.

        Returns:
            IndividualModelAnalysis with comprehensive metrics.
        """
        logger.info("Analyzing individual model performance")

        model_performances: dict[str, dict[str, Any]] = {}

        for model_type in self.config.models:
            logger.debug(f"Analyzing {model_type}")

            # Extract model-specific metrics
            model_metrics = self._extract_model_metrics(backtest_results, model_type)

            if not model_metrics["mape"]:
                logger.warning(f"No metrics found for {model_type}")
                continue

            # Perform comprehensive analysis
            analysis: dict[str, Any] = {
                "overall_mape": float(np.mean(model_metrics["mape"])),
                "mape_std": float(np.std(model_metrics["mape"])),
                "overall_mae": float(np.mean(model_metrics["mae"])) if model_metrics["mae"] else 0.0,
                "overall_rmse": float(np.mean(model_metrics["rmse"])) if model_metrics["rmse"] else 0.0,
                "seasonal_performance": self._analyze_seasonal_patterns(model_metrics),
                "area_performance": self._analyze_area_performance(model_metrics),
                "temporal_performance": self._analyze_temporal_patterns(model_metrics),
                "stability_metrics": self._calculate_stability_metrics(model_metrics),
                "outlier_analysis": self._analyze_outliers(model_metrics),
            }

            model_performances[model_type] = analysis

        # Rank models
        rankings = self._rank_models(model_performances)

        # Analyze seasonal patterns across all models
        seasonal_patterns = self._compare_seasonal_patterns(model_performances)

        return IndividualModelAnalysis(
            performances=model_performances,
            rankings=rankings,
            seasonal_patterns=seasonal_patterns,
        )

    def validate_combination_effectiveness(
        self,
        individual_results: IndividualModelAnalysis,
        combination_mapes: dict[str, float],
        combination_samples: dict[str, list[float]] | None = None,
    ) -> CombinationValidationResult:
        """Validate that model combinations outperform individual models.

        Args:
            individual_results: Analysis of individual models.
            combination_mapes: MAPE values for each combination method.
            combination_samples: Optional MAPE samples for statistical testing.

        Returns:
            CombinationValidationResult with statistical validation.
        """
        logger.info("Validating model combination effectiveness")

        validation_results: dict[str, dict[str, Any]] = {}
        best_improvement = float("-inf")
        best_combination: str | None = None

        # Get best individual model performance
        if not individual_results.performances:
            return CombinationValidationResult(
                validations={},
                best_combination=None,
                improvement_over_best_individual=0.0,
            )

        best_individual_mape = min(
            perf["overall_mape"]
            for perf in individual_results.performances.values()
        )

        for combination_method, combination_mape in combination_mapes.items():
            logger.debug(f"Validating {combination_method}")

            # Calculate improvement
            improvement = best_individual_mape - combination_mape
            improvement_pct = (improvement / best_individual_mape) * 100 if best_individual_mape > 0 else 0.0

            # Statistical significance testing if samples provided
            significance_result: dict[str, Any] = {}
            effect_size = 0.0

            if combination_samples and combination_method in combination_samples:
                best_model = individual_results.get_best_model()
                if best_model and best_model in individual_results.performances:
                    # Create mock samples for testing
                    best_samples = self._generate_samples_from_stats(
                        individual_results.performances[best_model]
                    )
                    comb_samples = np.array(combination_samples[combination_method])

                    if len(best_samples) >= 5 and len(comb_samples) >= 5:
                        test_result = self._test_significance(
                            best_samples,
                            comb_samples,
                        )
                        significance_result = test_result.to_dict()
                        effect_size = test_result.effect_size

            # Validation criteria
            passes_validation = (
                improvement > 0.01  # >1% improvement
                and (not significance_result or significance_result.get("p_value", 1.0) < self.config.significance_level)
                and (effect_size > self.config.effect_size_threshold or not significance_result)
            )

            validation_results[combination_method] = {
                "combination_mape": combination_mape,
                "best_individual_mape": best_individual_mape,
                "improvement": improvement,
                "improvement_percentage": improvement_pct,
                "statistical_significance": significance_result,
                "effect_size": effect_size,
                "passes_validation": passes_validation,
            }

            if improvement > best_improvement:
                best_improvement = improvement
                best_combination = combination_method

        return CombinationValidationResult(
            validations=validation_results,
            best_combination=best_combination,
            improvement_over_best_individual=best_improvement,
        )

    def analyze_reconciliation_impact(
        self,
        base_forecasts: dict[str, np.ndarray],
        reconciled_forecasts: dict[str, np.ndarray],
        actuals: dict[str, np.ndarray],
    ) -> ReconciliationImpactAnalysis:
        """Analyze the impact of hierarchical reconciliation.

        Args:
            base_forecasts: Base forecasts before reconciliation by area.
            reconciled_forecasts: Forecasts after reconciliation by area.
            actuals: Actual load values by area.

        Returns:
            ReconciliationImpactAnalysis with impact metrics.
        """
        logger.info("Analyzing reconciliation impact")

        impact_metrics: dict[str, dict[str, float]] = {}

        for area in self.config.areas:
            if area not in base_forecasts or area not in reconciled_forecasts or area not in actuals:
                continue

            base_preds = np.array(base_forecasts[area])
            recon_preds = np.array(reconciled_forecasts[area])
            actual_vals = np.array(actuals[area])

            if len(base_preds) == 0 or len(actual_vals) == 0:
                continue

            # Calculate metrics before reconciliation
            base_mape = self._calculate_mape(base_preds, actual_vals)
            base_mae = self._calculate_mae(base_preds, actual_vals)

            # Calculate metrics after reconciliation
            reconciled_mape = self._calculate_mape(recon_preds, actual_vals)
            reconciled_mae = self._calculate_mae(recon_preds, actual_vals)

            # Calculate coherence metrics
            coherence_before = self._calculate_coherence(base_forecasts, area)
            coherence_after = self._calculate_coherence(reconciled_forecasts, area)

            mape_improvement = base_mape - reconciled_mape

            impact_metrics[area] = {
                "base_mape": base_mape,
                "reconciled_mape": reconciled_mape,
                "improvement": mape_improvement,
                "mape_improvement_pct": (mape_improvement / base_mape * 100) if base_mape > 0 else 0.0,
                "base_mae": base_mae,
                "reconciled_mae": reconciled_mae,
                "mae_improvement": base_mae - reconciled_mae,
                "coherence_before": coherence_before,
                "coherence_after": coherence_after,
                "coherence_improvement": coherence_after - coherence_before,
            }

        # Validate hierarchical consistency
        consistency_validation = self._validate_hierarchical_consistency(reconciled_forecasts)

        # Calculate overall improvement
        improvements = [m["improvement"] for m in impact_metrics.values()]
        overall_improvement = float(np.mean(improvements)) if improvements else 0.0

        return ReconciliationImpactAnalysis(
            area_impacts=impact_metrics,
            consistency_validation=consistency_validation,
            overall_improvement=overall_improvement,
        )

    def detect_performance_drift(
        self,
        historical_results: list[BacktestPeriodResult],
        window_size: int | None = None,
    ) -> DriftDetectionResult:
        """Detect performance degradation and drift over time.

        Args:
            historical_results: Historical backtest results.
            window_size: Size of rolling window for drift detection.

        Returns:
            DriftDetectionResult with drift detection results.
        """
        logger.info("Detecting performance drift")

        window = window_size or self.config.drift_window_size
        drift_results: dict[str, dict[str, Any]] = {}
        overall_drift = False
        recommendations: list[str] = []

        for model_type in self.config.models:
            # Extract time series of performance metrics
            mape_series = self._extract_metric_time_series(
                historical_results, model_type, "mape"
            )

            if len(mape_series) < window:
                logger.warning(
                    f"Insufficient data for drift detection in {model_type}"
                )
                continue

            # Detect drift using various methods
            drift_analysis = {
                "mean_drift": self._detect_mean_drift(mape_series, window),
                "variance_drift": self._detect_variance_drift(mape_series, window),
                "trend_analysis": self._analyze_performance_trend(mape_series),
                "anomaly_periods": self._detect_anomaly_periods(mape_series),
            }

            drift_results[model_type] = drift_analysis

            # Check for significant drift
            if drift_analysis["mean_drift"]["is_significant"]:
                overall_drift = True
                if drift_analysis["trend_analysis"]["is_degrading"]:
                    recommendations.append(
                        f"Consider retraining {model_type} - performance degradation detected"
                    )

            if drift_analysis["variance_drift"]["is_increasing"]:
                recommendations.append(
                    f"Investigate {model_type} stability - variance increasing"
                )

        return DriftDetectionResult(
            model_drift=drift_results,
            overall_drift_detected=overall_drift,
            recommendations=recommendations,
        )

    def compare_models_statistically(
        self,
        model1_samples: np.ndarray,
        model2_samples: np.ndarray,
        test_type: str = "paired_t_test",
    ) -> StatisticalTestResult:
        """Perform statistical comparison between two models.

        Args:
            model1_samples: MAPE samples from model 1.
            model2_samples: MAPE samples from model 2.
            test_type: Type of test ('paired_t_test', 'wilcoxon', 'mann_whitney').

        Returns:
            StatisticalTestResult with test results.
        """
        return self._test_significance(model1_samples, model2_samples, test_type)

    # Helper methods

    def _extract_model_metrics(
        self,
        backtest_results: YearlyBacktestResult,
        model_type: str,
    ) -> dict[str, list]:
        """Extract metrics for specific model from backtest results.

        Args:
            backtest_results: Yearly backtest results.
            model_type: Model type to extract.

        Returns:
            Dictionary of metrics lists.
        """
        metrics: dict[str, list] = {
            "mape": [],
            "mae": [],
            "rmse": [],
            "dates": [],
            "areas": [],
        }

        for period_result in backtest_results.period_results:
            for area in self.config.areas:
                key = f"{area}_{model_type}"
                if key in period_result.evaluation.get("mape", {}):
                    metrics["mape"].append(period_result.evaluation["mape"][key])
                    if "mae" in period_result.evaluation:
                        metrics["mae"].append(
                            period_result.evaluation["mae"].get(key, 0)
                        )
                    if "rmse" in period_result.evaluation:
                        metrics["rmse"].append(
                            period_result.evaluation["rmse"].get(key, 0)
                        )
                    metrics["dates"].append(period_result.period.test_start)
                    metrics["areas"].append(area)

        return metrics

    def _analyze_seasonal_patterns(
        self,
        model_metrics: dict[str, list],
    ) -> dict[str, dict[str, float]]:
        """Analyze seasonal performance patterns.

        Args:
            model_metrics: Model metrics dictionary.

        Returns:
            Seasonal performance statistics.
        """
        if not model_metrics["mape"] or not model_metrics["dates"]:
            return {}

        df = pd.DataFrame(
            {
                "mape": model_metrics["mape"],
                "date": pd.to_datetime(model_metrics["dates"]),
            }
        )

        df["month"] = df["date"].dt.month
        df["season"] = df["month"].apply(self._get_season)

        seasonal_performance: dict[str, dict[str, float]] = {}
        for season, group in df.groupby("season"):
            seasonal_performance[str(season)] = {
                "mean": float(group["mape"].mean()),
                "std": float(group["mape"].std()),
                "min": float(group["mape"].min()),
                "max": float(group["mape"].max()),
                "count": int(len(group)),
            }

        return seasonal_performance

    def _analyze_area_performance(
        self,
        model_metrics: dict[str, list],
    ) -> dict[str, dict[str, float]]:
        """Analyze performance by geographic area.

        Args:
            model_metrics: Model metrics dictionary.

        Returns:
            Area performance statistics.
        """
        if not model_metrics["mape"] or not model_metrics["areas"]:
            return {}

        df = pd.DataFrame(
            {
                "mape": model_metrics["mape"],
                "area": model_metrics["areas"],
            }
        )

        area_performance: dict[str, dict[str, float]] = {}
        for area, group in df.groupby("area"):
            area_performance[str(area)] = {
                "mean": float(group["mape"].mean()),
                "std": float(group["mape"].std()),
                "min": float(group["mape"].min()),
                "max": float(group["mape"].max()),
                "count": int(len(group)),
            }

        return area_performance

    def _analyze_temporal_patterns(
        self,
        model_metrics: dict[str, list],
    ) -> dict[str, float]:
        """Analyze temporal performance patterns.

        Args:
            model_metrics: Model metrics dictionary.

        Returns:
            Temporal pattern statistics.
        """
        if not model_metrics["mape"] or not model_metrics["dates"]:
            return {}

        df = pd.DataFrame(
            {
                "mape": model_metrics["mape"],
                "date": pd.to_datetime(model_metrics["dates"]),
            }
        )

        df["weekday"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["weekday"].isin([5, 6])

        weekday_mape = df[~df["is_weekend"]]["mape"]
        weekend_mape = df[df["is_weekend"]]["mape"]

        return {
            "weekday_mean": float(weekday_mape.mean()) if len(weekday_mape) > 0 else 0.0,
            "weekend_mean": float(weekend_mape.mean()) if len(weekend_mape) > 0 else 0.0,
            "weekday_weekend_diff": float(
                weekend_mape.mean() - weekday_mape.mean()
            ) if len(weekday_mape) > 0 and len(weekend_mape) > 0 else 0.0,
        }

    def _calculate_stability_metrics(
        self,
        model_metrics: dict[str, list],
    ) -> dict[str, float]:
        """Calculate stability metrics for model performance.

        Args:
            model_metrics: Model metrics dictionary.

        Returns:
            Stability metrics.
        """
        if not model_metrics["mape"]:
            return {
                "coefficient_of_variation": 0.0,
                "range": 0.0,
                "iqr": 0.0,
                "stability_score": 0.0,
            }

        mape_values = np.array(model_metrics["mape"])
        mean_val = np.mean(mape_values)
        std_val = np.std(mape_values)

        return {
            "coefficient_of_variation": float(std_val / mean_val) if mean_val > 0 else 0.0,
            "range": float(np.max(mape_values) - np.min(mape_values)),
            "iqr": float(
                np.percentile(mape_values, 75) - np.percentile(mape_values, 25)
            ),
            "stability_score": float(1 - (std_val / mean_val)) if mean_val > 0 else 0.0,
        }

    def _analyze_outliers(
        self,
        model_metrics: dict[str, list],
    ) -> dict[str, Any]:
        """Analyze outliers in model performance.

        Args:
            model_metrics: Model metrics dictionary.

        Returns:
            Outlier analysis results.
        """
        if not model_metrics["mape"]:
            return {
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "outlier_values": [],
                "lower_bound": 0.0,
                "upper_bound": 0.0,
            }

        mape_values = np.array(model_metrics["mape"])

        q1 = float(np.percentile(mape_values, 25))
        q3 = float(np.percentile(mape_values, 75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = mape_values[
            (mape_values < lower_bound) | (mape_values > upper_bound)
        ]

        return {
            "outlier_count": len(outliers),
            "outlier_percentage": float(len(outliers) / len(mape_values) * 100),
            "outlier_values": outliers.tolist(),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    def _rank_models(
        self,
        model_performances: dict[str, dict[str, Any]],
    ) -> dict[str, int]:
        """Rank models by overall performance.

        Args:
            model_performances: Model performance dictionary.

        Returns:
            Model rankings (1 = best).
        """
        if not model_performances:
            return {}

        model_mapes = {
            model: perf["overall_mape"]
            for model, perf in model_performances.items()
        }

        sorted_models = sorted(model_mapes.items(), key=lambda x: x[1])

        return {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}

    def _compare_seasonal_patterns(
        self,
        model_performances: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Compare seasonal patterns across all models.

        Args:
            model_performances: Model performance dictionary.

        Returns:
            Seasonal comparison results.
        """
        seasonal_comparison: dict[str, dict[str, Any]] = {}
        seasons = ["summer", "fall", "winter", "spring"]

        for season in seasons:
            season_data: dict[str, float] = {}
            for model, perf in model_performances.items():
                seasonal_perf = perf.get("seasonal_performance", {})
                if season in seasonal_perf:
                    season_data[model] = seasonal_perf[season]["mean"]

            if season_data:
                best_model = min(season_data.items(), key=lambda x: x[1])
                worst_model = max(season_data.items(), key=lambda x: x[1])
                seasonal_comparison[season] = {
                    "model_performances": season_data,
                    "best_model": best_model[0],
                    "worst_model": worst_model[0],
                    "best_mape": best_model[1],
                    "worst_mape": worst_model[1],
                }

        return seasonal_comparison

    def _get_season(self, month: int) -> str:
        """Map month to season (Southern Hemisphere).

        Args:
            month: Month number (1-12).

        Returns:
            Season name.
        """
        if month in [12, 1, 2]:
            return "summer"
        elif month in [3, 4, 5]:
            return "fall"
        elif month in [6, 7, 8]:
            return "winter"
        else:
            return "spring"

    def _test_significance(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        test_type: str = "paired_t_test",
    ) -> StatisticalTestResult:
        """Perform statistical significance testing.

        Args:
            sample1: First sample.
            sample2: Second sample.
            test_type: Type of test.

        Returns:
            StatisticalTestResult.
        """
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)

        # Ensure equal lengths for paired tests
        min_len = min(len(sample1), len(sample2))
        sample1 = sample1[:min_len]
        sample2 = sample2[:min_len]

        if min_len < 2:
            return StatisticalTestResult(
                test_type=test_type,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                effect_size=0.0,
            )

        try:
            if test_type == "paired_t_test":
                statistic, p_value = stats.ttest_rel(sample1, sample2)
            elif test_type == "wilcoxon":
                statistic, p_value = stats.wilcoxon(sample1, sample2)
            elif test_type == "mann_whitney":
                statistic, p_value = stats.mannwhitneyu(sample1, sample2)
            else:
                raise ValueError(f"Unknown test type: {test_type}")

            effect_size = self._calculate_effect_size(sample1, sample2)

            return StatisticalTestResult(
                test_type=test_type,
                statistic=float(statistic),
                p_value=float(p_value),
                is_significant=p_value < self.config.significance_level,
                effect_size=effect_size,
            )
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return StatisticalTestResult(
                test_type=test_type,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                effect_size=0.0,
            )

    def _calculate_effect_size(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
    ) -> float:
        """Calculate Cohen's d effect size.

        Args:
            sample1: First sample.
            sample2: Second sample.

        Returns:
            Effect size value.
        """
        mean_diff = np.mean(sample1) - np.mean(sample2)
        pooled_std = np.sqrt((np.var(sample1) + np.var(sample2)) / 2)
        return float(mean_diff / pooled_std) if pooled_std > 0 else 0.0

    def _generate_samples_from_stats(
        self,
        performance: dict[str, Any],
        n_samples: int = 30,
    ) -> np.ndarray:
        """Generate sample values from performance statistics.

        Args:
            performance: Performance statistics dictionary.
            n_samples: Number of samples to generate.

        Returns:
            Array of sample values.
        """
        mean = performance.get("overall_mape", 0.0)
        std = performance.get("mape_std", 1.0)
        return np.random.normal(mean, std, n_samples)

    def _calculate_mape(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Calculate MAPE.

        Args:
            predictions: Predicted values.
            actuals: Actual values.

        Returns:
            MAPE percentage.
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return 0.0

        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

        mask = actuals != 0
        if not mask.any():
            return 0.0

        return float(np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100)

    def _calculate_mae(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Calculate MAE.

        Args:
            predictions: Predicted values.
            actuals: Actual values.

        Returns:
            MAE value.
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return 0.0

        min_len = min(len(predictions), len(actuals))
        return float(np.mean(np.abs(actuals[:min_len] - predictions[:min_len])))

    def _calculate_coherence(
        self,
        forecasts: dict[str, np.ndarray],
        area: str,
    ) -> float:
        """Calculate hierarchical coherence.

        Args:
            forecasts: Forecast dictionary by area.
            area: Area to check.

        Returns:
            Coherence score (0-1).
        """
        # For SIN, check if subsystems sum to total
        if area == "SIN" and "SIN" in forecasts:
            subsystem_areas = ["SECO", "S", "NE", "N"]
            available_subsystems = [a for a in subsystem_areas if a in forecasts]

            if not available_subsystems:
                return 1.0

            sin_total = np.array(forecasts["SIN"])
            subsystem_sum = sum(np.array(forecasts[a]) for a in available_subsystems)

            if len(sin_total) > 0 and sin_total.sum() > 0:
                relative_error = np.abs(sin_total - subsystem_sum) / np.abs(sin_total)
                return float(1 - np.mean(relative_error))

        return 1.0

    def _validate_hierarchical_consistency(
        self,
        reconciled_forecasts: dict[str, np.ndarray],
    ) -> dict[str, bool]:
        """Validate hierarchical consistency after reconciliation.

        Args:
            reconciled_forecasts: Reconciled forecasts by area.

        Returns:
            Consistency validation results.
        """
        consistency_checks: dict[str, bool] = {}

        # Check if subsystems sum to SIN
        if "SIN" in reconciled_forecasts:
            subsystem_areas = ["SECO", "S", "NE", "N"]
            available_subsystems = [a for a in subsystem_areas if a in reconciled_forecasts]

            if available_subsystems:
                sin_total = np.array(reconciled_forecasts["SIN"])
                subsystem_sum = sum(np.array(reconciled_forecasts[a]) for a in available_subsystems)

                consistency_checks["system_consistency"] = bool(
                    np.allclose(sin_total, subsystem_sum, rtol=1e-3)
                )
            else:
                consistency_checks["system_consistency"] = True
        else:
            consistency_checks["system_consistency"] = True

        return consistency_checks

    def _extract_metric_time_series(
        self,
        historical_results: list[BacktestPeriodResult],
        model_type: str,
        metric: str,
    ) -> np.ndarray:
        """Extract time series of specific metric for model.

        Args:
            historical_results: Historical backtest results.
            model_type: Model type.
            metric: Metric name.

        Returns:
            Time series array.
        """
        values: list[float] = []

        for result in historical_results:
            if metric in result.evaluation:
                # Aggregate across areas
                metric_values = [
                    v
                    for k, v in result.evaluation[metric].items()
                    if model_type in k
                ]
                if metric_values:
                    values.append(float(np.mean(metric_values)))

        return np.array(values)

    def _detect_mean_drift(
        self,
        time_series: np.ndarray,
        window_size: int,
    ) -> dict[str, Any]:
        """Detect drift in mean performance.

        Args:
            time_series: Performance time series.
            window_size: Rolling window size.

        Returns:
            Mean drift analysis results.
        """
        if len(time_series) < window_size:
            return {
                "drift_absolute": 0.0,
                "drift_percentage": 0.0,
                "is_significant": False,
            }

        rolling_mean = pd.Series(time_series).rolling(window=window_size).mean()

        first_window_mean = float(rolling_mean.iloc[window_size - 1])
        last_window_mean = float(rolling_mean.iloc[-1])

        drift = last_window_mean - first_window_mean
        drift_pct = (drift / first_window_mean * 100) if first_window_mean > 0 else 0.0

        return {
            "drift_absolute": drift,
            "drift_percentage": drift_pct,
            "is_significant": abs(drift_pct) > 10,  # >10% change
        }

    def _detect_variance_drift(
        self,
        time_series: np.ndarray,
        window_size: int,
    ) -> dict[str, Any]:
        """Detect drift in performance variance.

        Args:
            time_series: Performance time series.
            window_size: Rolling window size.

        Returns:
            Variance drift analysis results.
        """
        if len(time_series) < window_size:
            return {
                "variance_drift": 0.0,
                "is_increasing": False,
            }

        rolling_std = pd.Series(time_series).rolling(window=window_size).std()

        first_window_std = float(rolling_std.iloc[window_size - 1])
        last_window_std = float(rolling_std.iloc[-1])

        drift = last_window_std - first_window_std

        return {
            "variance_drift": drift,
            "is_increasing": drift > 0,
        }

    def _analyze_performance_trend(
        self,
        time_series: np.ndarray,
    ) -> dict[str, Any]:
        """Analyze overall performance trend.

        Args:
            time_series: Performance time series.

        Returns:
            Trend analysis results.
        """
        if len(time_series) < 2:
            return {
                "trend_slope": 0.0,
                "is_improving": False,
                "is_degrading": False,
            }

        x = np.arange(len(time_series))
        y = time_series

        # Linear regression
        slope, _ = np.polyfit(x, y, 1)

        return {
            "trend_slope": float(slope),
            "is_improving": slope < 0,  # Lower MAPE is better
            "is_degrading": slope > 0,
        }

    def _detect_anomaly_periods(
        self,
        time_series: np.ndarray,
    ) -> list[int]:
        """Detect anomaly periods in performance.

        Args:
            time_series: Performance time series.

        Returns:
            List of anomaly period indices.
        """
        if len(time_series) < 3:
            return []

        mean = np.mean(time_series)
        std = np.std(time_series)

        if std == 0:
            return []

        # Periods where performance is >2 std from mean
        anomalies = []
        for i, value in enumerate(time_series):
            if abs(value - mean) > 2 * std:
                anomalies.append(i)

        return anomalies
