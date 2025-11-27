"""Model comparison framework for forecast model evaluation.

This module provides comprehensive model comparison capabilities for evaluating
and ranking multiple forecast models using statistical tests, cross-validation,
and multi-metric analysis.

Key Components:
- ModelComparator: Main class for comparing multiple models
- ComparisonConfig: Configuration for comparison settings
- ComparisonResult: Container for comparison results
- ModelRanking: Ranking information for a single model
- PairwiseComparison: Results from pairwise model comparisons

Supported Features:
- Multi-model comparison
- Pairwise statistical tests (t-test, Wilcoxon, Diebold-Mariano)
- Cross-validation for robust comparison
- Multiple metric support (MAPE, MAE, RMSE, R²)
- Model ranking with confidence intervals
- Visualization-ready data structures

Example:
    ```python
    from src.evaluation.model_comparator import ModelComparator, ComparisonConfig

    # Configure comparator
    config = ComparisonConfig(
        metrics=["mape", "rmse"],
        significance_level=0.05,
        cv_folds=5,
    )

    # Create comparator
    comparator = ModelComparator(config)

    # Compare models
    predictions = {
        "model_a": pred_a,
        "model_b": pred_b,
        "model_c": pred_c,
    }
    result = comparator.compare(predictions, actuals, timestamps)

    # Get rankings
    print(f"Best model: {result.best_model}")
    for ranking in result.rankings:
        print(f"{ranking.rank}. {ranking.model_name}: MAPE={ranking.mape:.2f}%")
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import MetricsCalculator, MetricsConfig, MetricsResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ComparisonMetric(Enum):
    """Metrics available for model comparison.

    Attributes:
        MAPE: Mean Absolute Percentage Error.
        MAE: Mean Absolute Error.
        RMSE: Root Mean Square Error.
        MSE: Mean Square Error.
        R2: R-squared (coefficient of determination).
    """

    MAPE = "mape"
    MAE = "mae"
    RMSE = "rmse"
    MSE = "mse"
    R2 = "r2"


class RankingMethod(Enum):
    """Methods for ranking models.

    Attributes:
        SINGLE_METRIC: Rank by a single primary metric.
        WEIGHTED_AVERAGE: Rank by weighted average of multiple metrics.
        PARETO: Pareto-based ranking (non-dominated solutions).
        BORDA_COUNT: Borda count method across all metrics.
    """

    SINGLE_METRIC = "single_metric"
    WEIGHTED_AVERAGE = "weighted_average"
    PARETO = "pareto"
    BORDA_COUNT = "borda_count"


@dataclass
class ComparisonConfig:
    """Configuration for model comparison.

    Attributes:
        metrics: List of metrics to use for comparison.
        primary_metric: Primary metric for ranking (default: MAPE).
        ranking_method: Method for ranking models.
        significance_level: Significance level for statistical tests.
        cv_folds: Number of cross-validation folds (0 for no CV).
        cv_gap: Gap between train and test in CV for time series.
        metric_weights: Weights for weighted average ranking.
        bootstrap_iterations: Number of bootstrap samples for CI.
        use_diebold_mariano: Use Diebold-Mariano test for pairwise comparison.
        random_seed: Random seed for reproducibility.

    Example:
        >>> config = ComparisonConfig(
        ...     metrics=["mape", "rmse"],
        ...     primary_metric="mape",
        ...     significance_level=0.05
        ... )
    """

    metrics: list[str] = field(default_factory=lambda: ["mape", "mae", "rmse"])
    primary_metric: str = "mape"
    ranking_method: str = "single_metric"
    significance_level: float = 0.05
    cv_folds: int = 0
    cv_gap: int = 0
    metric_weights: dict[str, float] | None = None
    bootstrap_iterations: int = 1000
    use_diebold_mariano: bool = True
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_metrics = {"mape", "mae", "rmse", "mse", "r2"}
        for metric in self.metrics:
            if metric not in valid_metrics:
                msg = f"Invalid metric '{metric}'. Must be one of {valid_metrics}"
                raise ValueError(msg)

        if self.primary_metric not in valid_metrics:
            msg = f"Invalid primary_metric '{self.primary_metric}'"
            raise ValueError(msg)

        if not 0.0 < self.significance_level < 1.0:
            msg = f"significance_level must be between 0 and 1, got {self.significance_level}"
            raise ValueError(msg)

        valid_ranking_methods = {"single_metric", "weighted_average", "pareto", "borda_count"}
        if self.ranking_method not in valid_ranking_methods:
            msg = f"Invalid ranking_method '{self.ranking_method}'"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "metrics": self.metrics.copy(),
            "primary_metric": self.primary_metric,
            "ranking_method": self.ranking_method,
            "significance_level": self.significance_level,
            "cv_folds": self.cv_folds,
            "cv_gap": self.cv_gap,
            "metric_weights": self.metric_weights.copy() if self.metric_weights else None,
            "bootstrap_iterations": self.bootstrap_iterations,
            "use_diebold_mariano": self.use_diebold_mariano,
        }


@dataclass
class ModelRanking:
    """Ranking information for a single model.

    Attributes:
        model_name: Name of the model.
        rank: Overall rank (1 = best).
        metrics: Dictionary of metric values.
        confidence_intervals: Dictionary of metric to (lower, upper) CI.
        score: Composite score used for ranking.
        is_significantly_best: Whether significantly better than all others.
        is_pareto_optimal: Whether on the Pareto frontier.
        metadata: Additional ranking metadata.

    Example:
        >>> ranking = ModelRanking(
        ...     model_name="lgbm",
        ...     rank=1,
        ...     metrics={"mape": 5.2, "rmse": 150.0},
        ...     score=0.95
        ... )
    """

    model_name: str
    rank: int
    metrics: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    score: float = 0.0
    is_significantly_best: bool = False
    is_pareto_optimal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model_name": self.model_name,
            "rank": self.rank,
            "metrics": self.metrics.copy(),
            "confidence_intervals": {k: list(v) for k, v in self.confidence_intervals.items()},
            "score": self.score,
            "is_significantly_best": self.is_significantly_best,
            "is_pareto_optimal": self.is_pareto_optimal,
            "metadata": self.metadata.copy(),
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        metrics_str = ", ".join(f"{k}={v:.2f}" for k, v in self.metrics.items())
        return f"ModelRanking(rank={self.rank}, model={self.model_name!r}, {metrics_str})"


@dataclass
class PairwiseComparison:
    """Results from pairwise model comparison.

    Attributes:
        model_a: Name of first model.
        model_b: Name of second model.
        metric: Metric used for comparison.
        mean_diff: Mean difference (A - B).
        t_statistic: T-statistic from paired t-test.
        t_p_value: P-value from paired t-test.
        wilcoxon_statistic: Statistic from Wilcoxon test.
        wilcoxon_p_value: P-value from Wilcoxon test.
        dm_statistic: Diebold-Mariano statistic.
        dm_p_value: P-value from Diebold-Mariano test.
        is_significant: Whether difference is statistically significant.
        better_model: Name of the better model (or None if not significant).
        effect_size: Cohen's d effect size.

    Example:
        >>> comparison = PairwiseComparison(
        ...     model_a="lgbm",
        ...     model_b="rf",
        ...     metric="mape",
        ...     mean_diff=-0.5,
        ...     t_p_value=0.02,
        ...     is_significant=True,
        ...     better_model="lgbm"
        ... )
    """

    model_a: str
    model_b: str
    metric: str
    mean_diff: float
    t_statistic: float = 0.0
    t_p_value: float = 1.0
    wilcoxon_statistic: float | None = None
    wilcoxon_p_value: float | None = None
    dm_statistic: float | None = None
    dm_p_value: float | None = None
    is_significant: bool = False
    better_model: str | None = None
    effect_size: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "mean_diff": self.mean_diff,
            "t_statistic": self.t_statistic,
            "t_p_value": self.t_p_value,
            "wilcoxon_statistic": self.wilcoxon_statistic,
            "wilcoxon_p_value": self.wilcoxon_p_value,
            "dm_statistic": self.dm_statistic,
            "dm_p_value": self.dm_p_value,
            "is_significant": self.is_significant,
            "better_model": self.better_model,
            "effect_size": self.effect_size,
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        sig = "SIGNIFICANT" if self.is_significant else "not significant"
        return f"PairwiseComparison({self.model_a} vs {self.model_b}: {sig}, better={self.better_model})"


@dataclass
class ComparisonResult:
    """Results from model comparison.

    Attributes:
        rankings: List of ModelRanking sorted by rank.
        best_model: Name of the best model.
        pairwise_comparisons: List of pairwise comparison results.
        metrics_summary: Summary statistics for each model.
        cv_results: Cross-validation results if CV was used.
        recommendation: Text recommendation based on results.
        metadata: Additional comparison metadata.

    Example:
        >>> result = ComparisonResult(
        ...     rankings=[ranking1, ranking2],
        ...     best_model="lgbm",
        ...     recommendation="Use lgbm for production."
        ... )
    """

    rankings: list[ModelRanking]
    best_model: str
    pairwise_comparisons: list[PairwiseComparison] = field(default_factory=list)
    metrics_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    cv_results: dict[str, Any] | None = None
    recommendation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_models(self) -> int:
        """Get number of models compared.

        Returns:
            Number of models.
        """
        return len(self.rankings)

    @property
    def model_names(self) -> list[str]:
        """Get list of model names.

        Returns:
            List of model names.
        """
        return [r.model_name for r in self.rankings]

    def get_ranking(self, model_name: str) -> ModelRanking | None:
        """Get ranking for a specific model.

        Args:
            model_name: Name of the model.

        Returns:
            ModelRanking or None if not found.
        """
        for ranking in self.rankings:
            if ranking.model_name == model_name:
                return ranking
        return None

    def get_comparison(self, model_a: str, model_b: str) -> PairwiseComparison | None:
        """Get pairwise comparison between two models.

        Args:
            model_a: Name of first model.
            model_b: Name of second model.

        Returns:
            PairwiseComparison or None if not found.
        """
        for comp in self.pairwise_comparisons:
            if (comp.model_a == model_a and comp.model_b == model_b) or (
                comp.model_a == model_b and comp.model_b == model_a
            ):
                return comp
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "rankings": [r.to_dict() for r in self.rankings],
            "best_model": self.best_model,
            "pairwise_comparisons": [c.to_dict() for c in self.pairwise_comparisons],
            "metrics_summary": self.metrics_summary.copy(),
            "cv_results": self.cv_results,
            "recommendation": self.recommendation,
            "metadata": self.metadata.copy(),
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return f"ComparisonResult(best={self.best_model!r}, n_models={self.num_models})"


class ModelComparator:
    """Model comparison framework for forecast evaluation.

    Provides comprehensive model comparison capabilities including:
    - Multi-metric comparison
    - Pairwise statistical tests
    - Cross-validation support
    - Model ranking with multiple methods
    - Visualization-ready outputs

    Example:
        >>> comparator = ModelComparator()

        >>> # Prepare predictions
        >>> predictions = {
        ...     "lgbm": lgbm_pred,
        ...     "rf": rf_pred,
        ...     "arima": arima_pred,
        ... }

        >>> # Compare models
        >>> result = comparator.compare(predictions, actual)

        >>> # Get best model
        >>> print(f"Best model: {result.best_model}")

        >>> # Check if difference is significant
        >>> comp = result.get_comparison("lgbm", "rf")
        >>> if comp.is_significant:
        ...     print(f"{comp.better_model} is significantly better")
    """

    def __init__(self, config: ComparisonConfig | None = None) -> None:
        """Initialize comparator.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or ComparisonConfig()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._metrics_calculator = MetricsCalculator(
            MetricsConfig(
                bootstrap_iterations=self.config.bootstrap_iterations,
                random_seed=self.config.random_seed,
            )
        )

        logger.debug(
            "Initialized ModelComparator with config: %s",
            self.config.to_dict(),
        )

    def compare(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
        timestamps: pd.DatetimeIndex | None = None,
    ) -> ComparisonResult:
        """Compare multiple models.

        Args:
            predictions: Dictionary mapping model names to prediction arrays.
            actuals: Array of actual values.
            timestamps: Optional timestamps for time-series CV.

        Returns:
            ComparisonResult with rankings and comparisons.

        Raises:
            ValueError: If inputs are invalid.
        """
        if len(predictions) < 2:
            msg = "Need at least 2 models to compare"
            raise ValueError(msg)

        actuals = np.asarray(actuals, dtype=np.float64)

        # Validate all predictions have same length as actuals
        for name, pred in predictions.items():
            pred = np.asarray(pred, dtype=np.float64)
            if len(pred) != len(actuals):
                msg = f"Prediction length mismatch for {name}: {len(pred)} vs {len(actuals)}"
                raise ValueError(msg)
            predictions[name] = pred

        logger.info(
            "Comparing %d models with %d samples",
            len(predictions),
            len(actuals),
        )

        # Calculate metrics for each model
        metrics_summary = self._calculate_metrics(predictions, actuals)

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(predictions, actuals)

        # Perform pairwise comparisons
        pairwise_comparisons = self._pairwise_comparisons(predictions, actuals)

        # Cross-validation if configured
        cv_results = None
        if self.config.cv_folds > 0 and timestamps is not None:
            cv_results = self._cross_validate(predictions, actuals, timestamps)

        # Rank models
        rankings = self._rank_models(metrics_summary, confidence_intervals, pairwise_comparisons)

        # Determine if best model is significantly better
        best_model = rankings[0].model_name
        rankings[0].is_significantly_best = self._is_significantly_best(
            best_model, pairwise_comparisons
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(rankings, pairwise_comparisons)

        return ComparisonResult(
            rankings=rankings,
            best_model=best_model,
            pairwise_comparisons=pairwise_comparisons,
            metrics_summary=metrics_summary,
            cv_results=cv_results,
            recommendation=recommendation,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "n_samples": len(actuals),
                "config": self.config.to_dict(),
            },
        )

    def _calculate_metrics(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Calculate metrics for all models.

        Args:
            predictions: Model predictions.
            actuals: Actual values.

        Returns:
            Dictionary mapping model names to metric values.
        """
        metrics_summary: dict[str, dict[str, float]] = {}

        for name, pred in predictions.items():
            result = self._metrics_calculator.calculate_single(actuals, pred)
            metrics_summary[name] = {
                "mape": result.mape,
                "mae": result.mae,
                "rmse": result.rmse,
                "mse": result.mse,
                "r2": result.r2,
            }

        return metrics_summary

    def _calculate_confidence_intervals(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> dict[str, dict[str, tuple[float, float]]]:
        """Calculate confidence intervals for metrics.

        Args:
            predictions: Model predictions.
            actuals: Actual values.

        Returns:
            Dictionary mapping model names to metric CIs.
        """
        ci_results: dict[str, dict[str, tuple[float, float]]] = {}

        for name, pred in predictions.items():
            result = self._metrics_calculator.calculate_single(actuals, pred)
            ci_results[name] = result.confidence_intervals

        return ci_results

    def _pairwise_comparisons(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> list[PairwiseComparison]:
        """Perform pairwise statistical comparisons.

        Args:
            predictions: Model predictions.
            actuals: Actual values.

        Returns:
            List of pairwise comparison results.
        """
        comparisons = []
        model_names = list(predictions.keys())

        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1 :]:
                pred_a = predictions[model_a]
                pred_b = predictions[model_b]

                # Calculate errors
                errors_a = np.abs(actuals - pred_a)
                errors_b = np.abs(actuals - pred_b)

                # Mean difference
                mean_diff = float(np.mean(errors_a) - np.mean(errors_b))

                # Paired t-test
                t_stat, t_p = stats.ttest_rel(errors_a, errors_b)

                # Wilcoxon test (if enough samples)
                wilcoxon_stat = None
                wilcoxon_p = None
                if len(errors_a) >= 20:
                    try:
                        diff = errors_a - errors_b
                        if not np.allclose(diff, 0):
                            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                                errors_a, errors_b, alternative="two-sided"
                            )
                    except ValueError:
                        pass

                # Diebold-Mariano test
                dm_stat = None
                dm_p = None
                if self.config.use_diebold_mariano:
                    dm_stat, dm_p = self._diebold_mariano_test(
                        actuals, pred_a, pred_b
                    )

                # Effect size (Cohen's d)
                diff = errors_a - errors_b
                std_diff = np.std(diff, ddof=1)
                effect_size = float(np.mean(diff) / (std_diff + 1e-10))

                # Determine significance (using the most conservative p-value)
                p_values = [float(t_p)]
                if wilcoxon_p is not None:
                    p_values.append(float(wilcoxon_p))
                if dm_p is not None:
                    p_values.append(float(dm_p))

                # Use maximum p-value (most conservative)
                max_p = max(p_values)
                is_significant = max_p < self.config.significance_level

                # Determine better model
                better_model = None
                if is_significant:
                    if mean_diff < 0:
                        better_model = model_a  # A has lower errors
                    else:
                        better_model = model_b  # B has lower errors

                comparison = PairwiseComparison(
                    model_a=model_a,
                    model_b=model_b,
                    metric=self.config.primary_metric,
                    mean_diff=mean_diff,
                    t_statistic=float(t_stat),
                    t_p_value=float(t_p),
                    wilcoxon_statistic=float(wilcoxon_stat) if wilcoxon_stat is not None else None,
                    wilcoxon_p_value=float(wilcoxon_p) if wilcoxon_p is not None else None,
                    dm_statistic=dm_stat,
                    dm_p_value=dm_p,
                    is_significant=is_significant,
                    better_model=better_model,
                    effect_size=effect_size,
                )
                comparisons.append(comparison)

        return comparisons

    def _diebold_mariano_test(
        self,
        actuals: np.ndarray,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        h: int = 1,
    ) -> tuple[float, float]:
        """Perform Diebold-Mariano test for forecast comparison.

        Tests whether two forecasts have equal predictive accuracy.

        Args:
            actuals: Actual values.
            pred_a: Predictions from model A.
            pred_b: Predictions from model B.
            h: Forecast horizon (for variance correction).

        Returns:
            Tuple of (DM statistic, p-value).
        """
        # Calculate loss differentials (squared errors)
        loss_a = (actuals - pred_a) ** 2
        loss_b = (actuals - pred_b) ** 2
        d = loss_a - loss_b

        n = len(d)
        d_mean = np.mean(d)

        # Estimate variance with Newey-West correction for autocorrelation
        gamma_0 = np.var(d, ddof=1)
        gamma = np.zeros(h)
        for k in range(1, h):
            gamma[k - 1] = np.cov(d[:-k], d[k:])[0, 1]

        # Long-run variance
        var_d = gamma_0 + 2 * np.sum(gamma)
        var_d = max(var_d, 1e-10)  # Prevent division by zero

        # DM statistic
        dm_stat = d_mean / np.sqrt(var_d / n)

        # P-value (two-sided)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return float(dm_stat), float(p_value)

    def _cross_validate(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> dict[str, Any]:
        """Perform time-series cross-validation.

        Args:
            predictions: Model predictions.
            actuals: Actual values.
            timestamps: Timestamps for temporal splitting.

        Returns:
            Dictionary of CV results.
        """
        n = len(actuals)
        n_folds = self.config.cv_folds
        gap = self.config.cv_gap

        # Calculate fold sizes
        fold_size = n // (n_folds + 1)

        cv_metrics: dict[str, list[dict[str, float]]] = {
            name: [] for name in predictions.keys()
        }

        for fold in range(n_folds):
            # Time series split: train on earlier data, test on later
            test_start = (fold + 1) * fold_size + gap
            test_end = min(test_start + fold_size, n)

            if test_end <= test_start:
                continue

            test_actuals = actuals[test_start:test_end]

            for name, pred in predictions.items():
                test_pred = pred[test_start:test_end]
                result = self._metrics_calculator.calculate_single(test_actuals, test_pred)
                cv_metrics[name].append({
                    "fold": fold,
                    "mape": result.mape,
                    "mae": result.mae,
                    "rmse": result.rmse,
                })

        # Aggregate CV results
        cv_summary = {}
        for name, fold_metrics in cv_metrics.items():
            if fold_metrics:
                mapes = [m["mape"] for m in fold_metrics]
                maes = [m["mae"] for m in fold_metrics]
                rmses = [m["rmse"] for m in fold_metrics]

                cv_summary[name] = {
                    "mape_mean": float(np.mean(mapes)),
                    "mape_std": float(np.std(mapes)),
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes)),
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses)),
                    "n_folds": len(fold_metrics),
                }

        return {
            "n_folds": n_folds,
            "gap": gap,
            "fold_size": fold_size,
            "summary": cv_summary,
        }

    def _rank_models(
        self,
        metrics_summary: dict[str, dict[str, float]],
        confidence_intervals: dict[str, dict[str, tuple[float, float]]],
        pairwise_comparisons: list[PairwiseComparison],
    ) -> list[ModelRanking]:
        """Rank models based on configured method.

        Args:
            metrics_summary: Metrics for each model.
            confidence_intervals: CIs for each model.
            pairwise_comparisons: Pairwise comparison results.

        Returns:
            Sorted list of ModelRanking.
        """
        rankings = []

        if self.config.ranking_method == "single_metric":
            rankings = self._rank_by_single_metric(metrics_summary)
        elif self.config.ranking_method == "weighted_average":
            rankings = self._rank_by_weighted_average(metrics_summary)
        elif self.config.ranking_method == "borda_count":
            rankings = self._rank_by_borda_count(metrics_summary)
        elif self.config.ranking_method == "pareto":
            rankings = self._rank_by_pareto(metrics_summary)
        else:
            rankings = self._rank_by_single_metric(metrics_summary)

        # Add confidence intervals
        for ranking in rankings:
            if ranking.model_name in confidence_intervals:
                ranking.confidence_intervals = confidence_intervals[ranking.model_name]

        # Identify Pareto-optimal models
        pareto_set = self._find_pareto_optimal(metrics_summary)
        for ranking in rankings:
            ranking.is_pareto_optimal = ranking.model_name in pareto_set

        return rankings

    def _rank_by_single_metric(
        self,
        metrics_summary: dict[str, dict[str, float]],
    ) -> list[ModelRanking]:
        """Rank by a single primary metric.

        Args:
            metrics_summary: Metrics for each model.

        Returns:
            Sorted list of ModelRanking.
        """
        metric = self.config.primary_metric

        # For R², higher is better; for errors, lower is better
        reverse = metric == "r2"

        sorted_models = sorted(
            metrics_summary.items(),
            key=lambda x: x[1].get(metric, float("inf")),
            reverse=reverse,
        )

        rankings = []
        for rank, (name, metrics) in enumerate(sorted_models, 1):
            rankings.append(
                ModelRanking(
                    model_name=name,
                    rank=rank,
                    metrics=metrics,
                    score=float(metrics.get(metric, 0)),
                )
            )

        return rankings

    def _rank_by_weighted_average(
        self,
        metrics_summary: dict[str, dict[str, float]],
    ) -> list[ModelRanking]:
        """Rank by weighted average of normalized metrics.

        Args:
            metrics_summary: Metrics for each model.

        Returns:
            Sorted list of ModelRanking.
        """
        weights = self.config.metric_weights or {m: 1.0 for m in self.config.metrics}

        # Normalize each metric to [0, 1]
        normalized: dict[str, dict[str, float]] = {}
        for metric in self.config.metrics:
            values = [metrics_summary[m].get(metric, 0) for m in metrics_summary]
            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val + 1e-10

            for name in metrics_summary:
                if name not in normalized:
                    normalized[name] = {}
                val = metrics_summary[name].get(metric, 0)

                # For R², higher is better; for errors, lower is better
                if metric == "r2":
                    normalized[name][metric] = (val - min_val) / range_val
                else:
                    normalized[name][metric] = (max_val - val) / range_val

        # Calculate weighted scores
        scores: dict[str, float] = {}
        for name in metrics_summary:
            score = 0.0
            total_weight = 0.0
            for metric in self.config.metrics:
                w = weights.get(metric, 1.0)
                score += w * normalized[name].get(metric, 0)
                total_weight += w
            scores[name] = score / total_weight if total_weight > 0 else 0

        # Sort by score (higher is better)
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        rankings = []
        for rank, (name, score) in enumerate(sorted_models, 1):
            rankings.append(
                ModelRanking(
                    model_name=name,
                    rank=rank,
                    metrics=metrics_summary[name],
                    score=score,
                )
            )

        return rankings

    def _rank_by_borda_count(
        self,
        metrics_summary: dict[str, dict[str, float]],
    ) -> list[ModelRanking]:
        """Rank by Borda count method.

        Each model gets points based on its rank for each metric.
        n-1 points for 1st place, n-2 for 2nd, etc.

        Args:
            metrics_summary: Metrics for each model.

        Returns:
            Sorted list of ModelRanking.
        """
        n_models = len(metrics_summary)
        borda_scores: dict[str, int] = {name: 0 for name in metrics_summary}

        for metric in self.config.metrics:
            # Sort models by this metric
            reverse = metric == "r2"  # Higher is better for R²
            sorted_models = sorted(
                metrics_summary.items(),
                key=lambda x: x[1].get(metric, float("inf")),
                reverse=reverse,
            )

            # Assign Borda points
            for rank, (name, _) in enumerate(sorted_models):
                borda_scores[name] += n_models - 1 - rank

        # Sort by Borda score (higher is better)
        sorted_scores = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)

        rankings = []
        for rank, (name, score) in enumerate(sorted_scores, 1):
            rankings.append(
                ModelRanking(
                    model_name=name,
                    rank=rank,
                    metrics=metrics_summary[name],
                    score=float(score),
                )
            )

        return rankings

    def _rank_by_pareto(
        self,
        metrics_summary: dict[str, dict[str, float]],
    ) -> list[ModelRanking]:
        """Rank by Pareto dominance.

        Args:
            metrics_summary: Metrics for each model.

        Returns:
            Sorted list of ModelRanking.
        """
        # Find Pareto-optimal set
        pareto_set = self._find_pareto_optimal(metrics_summary)

        # Rank Pareto-optimal first, then others by primary metric
        pareto_models = [name for name in metrics_summary if name in pareto_set]
        non_pareto_models = [name for name in metrics_summary if name not in pareto_set]

        # Sort within groups by primary metric
        metric = self.config.primary_metric
        reverse = metric == "r2"

        pareto_models.sort(
            key=lambda x: metrics_summary[x].get(metric, float("inf")),
            reverse=reverse,
        )
        non_pareto_models.sort(
            key=lambda x: metrics_summary[x].get(metric, float("inf")),
            reverse=reverse,
        )

        rankings = []
        for rank, name in enumerate(pareto_models + non_pareto_models, 1):
            rankings.append(
                ModelRanking(
                    model_name=name,
                    rank=rank,
                    metrics=metrics_summary[name],
                    is_pareto_optimal=name in pareto_set,
                )
            )

        return rankings

    def _find_pareto_optimal(
        self,
        metrics_summary: dict[str, dict[str, float]],
    ) -> set[str]:
        """Find Pareto-optimal models.

        A model is Pareto-optimal if no other model is better in all metrics.

        Args:
            metrics_summary: Metrics for each model.

        Returns:
            Set of Pareto-optimal model names.
        """
        pareto_set: set[str] = set()
        model_names = list(metrics_summary.keys())

        for name_a in model_names:
            is_dominated = False
            for name_b in model_names:
                if name_a == name_b:
                    continue

                # Check if B dominates A
                dominates = True
                strictly_better_in_one = False

                for metric in self.config.metrics:
                    val_a = metrics_summary[name_a].get(metric, float("inf"))
                    val_b = metrics_summary[name_b].get(metric, float("inf"))

                    # For R², higher is better
                    if metric == "r2":
                        if val_b < val_a:
                            dominates = False
                            break
                        if val_b > val_a:
                            strictly_better_in_one = True
                    else:
                        if val_b > val_a:
                            dominates = False
                            break
                        if val_b < val_a:
                            strictly_better_in_one = True

                if dominates and strictly_better_in_one:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_set.add(name_a)

        return pareto_set

    def _is_significantly_best(
        self,
        best_model: str,
        pairwise_comparisons: list[PairwiseComparison],
    ) -> bool:
        """Check if best model is significantly better than all others.

        Args:
            best_model: Name of the best model.
            pairwise_comparisons: List of pairwise comparisons.

        Returns:
            True if best model is significantly better than all others.
        """
        for comp in pairwise_comparisons:
            if best_model in (comp.model_a, comp.model_b):
                if comp.is_significant and comp.better_model == best_model:
                    continue
                elif not comp.is_significant:
                    return False
                else:
                    return False
        return True

    def _generate_recommendation(
        self,
        rankings: list[ModelRanking],
        pairwise_comparisons: list[PairwiseComparison],
    ) -> str:
        """Generate text recommendation based on results.

        Args:
            rankings: Ranked models.
            pairwise_comparisons: Pairwise comparison results.

        Returns:
            Recommendation text.
        """
        if not rankings:
            return "No models to compare."

        best = rankings[0]
        second = rankings[1] if len(rankings) > 1 else None

        rec_parts = []

        # Best model recommendation
        if best.is_significantly_best:
            rec_parts.append(
                f"Strongly recommend '{best.model_name}' - significantly better than all alternatives."
            )
        elif best.is_pareto_optimal:
            rec_parts.append(
                f"Recommend '{best.model_name}' - Pareto-optimal across metrics."
            )
        else:
            rec_parts.append(
                f"'{best.model_name}' ranks first but consider alternatives."
            )

        # Comparison with second best
        if second:
            comp = None
            for c in pairwise_comparisons:
                if (c.model_a == best.model_name and c.model_b == second.model_name) or (
                    c.model_b == best.model_name and c.model_a == second.model_name
                ):
                    comp = c
                    break

            if comp and not comp.is_significant:
                rec_parts.append(
                    f"No significant difference vs '{second.model_name}'."
                )

        # Metric-specific insights
        primary = self.config.primary_metric.upper()
        rec_parts.append(
            f"Based on {primary}: {best.metrics.get(self.config.primary_metric, 0):.2f}"
        )

        return " ".join(rec_parts)

    def compare_horizons(
        self,
        predictions_by_horizon: dict[int, dict[str, np.ndarray]],
        actuals_by_horizon: dict[int, np.ndarray],
    ) -> dict[int, ComparisonResult]:
        """Compare models across different forecast horizons.

        Args:
            predictions_by_horizon: Dict mapping horizon to model predictions.
            actuals_by_horizon: Dict mapping horizon to actual values.

        Returns:
            Dictionary mapping horizon to ComparisonResult.
        """
        results = {}

        for horizon in sorted(predictions_by_horizon.keys()):
            predictions = predictions_by_horizon[horizon]
            actuals = actuals_by_horizon.get(horizon)

            if actuals is None:
                logger.warning("No actuals for horizon %d", horizon)
                continue

            results[horizon] = self.compare(predictions, actuals)

        return results

    def get_metric_comparison_matrix(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> pd.DataFrame:
        """Get comparison matrix of all models vs all metrics.

        Args:
            predictions: Model predictions.
            actuals: Actual values.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        metrics_summary = self._calculate_metrics(predictions, actuals)

        data = []
        for name, metrics in metrics_summary.items():
            row = {"model": name}
            row.update(metrics)
            data.append(row)

        df = pd.DataFrame(data).set_index("model")
        return df

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return f"ModelComparator(primary_metric={self.config.primary_metric!r}, ranking={self.config.ranking_method!r})"
