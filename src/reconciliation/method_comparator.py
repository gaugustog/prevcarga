"""Reconciliation method comparison framework.

This module implements the ReconciliationComparator class for systematic
benchmarking and comparison of hierarchical forecast reconciliation methods.

Key features:
    - Cross-validation: Time-aware splits for unbiased evaluation
    - Benchmarking: Standardized performance evaluation across methods
    - Statistical testing: Pairwise tests for significance
    - Ranking: Performance-based method rankings
    - Recommendations: Data-driven method selection guidance

Example:
    ```python
    from src.reconciliation import (
        ReconciliationComparator,
        ComparatorConfig,
        OLSReconciler,
        WLSReconciler
    )
    import pandas as pd

    # Create methods to compare
    methods = [OLSReconciler(), WLSReconciler()]

    # Configure comparator
    config = ComparatorConfig(cv_folds=5, significance_level=0.05)
    comparator = ReconciliationComparator(config=config)

    # Run comparison
    report = comparator.compare_methods(
        methods=methods,
        forecasts=forecast_df,
        actuals=actual_df,
        hierarchy=hierarchy
    )

    # View rankings
    for method, rank in report.method_rankings.items():
        print(f"{method}: Rank {rank}")

    # Get recommendations
    for rec in report.recommendations:
        print(f"  - {rec}")
    ```

References:
    - Bergmeir, C., & Benitez, J.M. (2012). "On the use of cross-validation
      for time series predictor evaluation", Information Sciences.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.reconciliation.base_reconciler import BaseReconciler
from src.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CVFoldResult:
    """Results for a single cross-validation fold.

    Attributes:
        fold_idx: Index of the fold.
        method_name: Name of the method.
        mape: Mean Absolute Percentage Error.
        mae: Mean Absolute Error.
        rmse: Root Mean Squared Error.
        coherence_error: Aggregation coherence error.
        computation_time: Time to reconcile (seconds).
    """

    fold_idx: int
    method_name: str
    mape: float
    mae: float
    rmse: float
    coherence_error: float
    computation_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fold_idx": self.fold_idx,
            "method_name": self.method_name,
            "mape": float(self.mape),
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "coherence_error": float(self.coherence_error),
            "computation_time": float(self.computation_time),
        }


@dataclass
class PairwiseTest:
    """Results of pairwise statistical test.

    Attributes:
        method_a: First method name.
        method_b: Second method name.
        ttest_statistic: Paired t-test statistic.
        ttest_pvalue: Paired t-test p-value.
        wilcoxon_statistic: Wilcoxon signed-rank statistic.
        wilcoxon_pvalue: Wilcoxon signed-rank p-value.
        significant: Whether difference is significant.
        better_method: Name of statistically better method (or None).
    """

    method_a: str
    method_b: str
    ttest_statistic: float
    ttest_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    significant: bool
    better_method: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "ttest_statistic": float(self.ttest_statistic),
            "ttest_pvalue": float(self.ttest_pvalue),
            "wilcoxon_statistic": float(self.wilcoxon_statistic),
            "wilcoxon_pvalue": float(self.wilcoxon_pvalue),
            "significant": bool(self.significant),
            "better_method": self.better_method,
        }


@dataclass
class ComparisonReport:
    """Comprehensive method comparison results.

    Attributes:
        method_rankings: Dictionary mapping method names to ranks (1 = best).
        performance_matrix: Matrix of (n_methods x n_folds) MAPE scores.
        method_stats: Statistics per method (mean, std, min, max).
        pairwise_tests: Pairwise statistical test results.
        recommendations: List of recommendation strings.
        cv_results: All cross-validation fold results.
        timestamp: When comparison was performed.
        config: Configuration used for comparison.
    """

    method_rankings: dict[str, int]
    performance_matrix: list[list[float]]
    method_stats: dict[str, dict[str, float]]
    pairwise_tests: list[PairwiseTest]
    recommendations: list[str]
    cv_results: list[CVFoldResult] = field(default_factory=list)
    timestamp: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_rankings": self.method_rankings,
            "performance_matrix": self.performance_matrix,
            "method_stats": self.method_stats,
            "pairwise_tests": [t.to_dict() for t in self.pairwise_tests],
            "recommendations": self.recommendations,
            "cv_results": [r.to_dict() for r in self.cv_results],
            "timestamp": self.timestamp,
            "config": self.config,
        }


@dataclass
class ComparatorConfig:
    """Configuration for reconciliation comparator.

    Attributes:
        cv_folds: Number of cross-validation folds.
        significance_level: P-value threshold for significance tests.
        min_samples_per_fold: Minimum samples required per fold.
        metric: Primary metric for ranking (mape, mae, rmse).
        include_coherence: Whether to include coherence in evaluation.
        random_state: Random state for reproducibility.
    """

    cv_folds: int = 5
    significance_level: float = 0.05
    min_samples_per_fold: int = 10
    metric: str = "mape"
    include_coherence: bool = True
    random_state: int | None = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cv_folds": self.cv_folds,
            "significance_level": self.significance_level,
            "min_samples_per_fold": self.min_samples_per_fold,
            "metric": self.metric,
            "include_coherence": self.include_coherence,
            "random_state": self.random_state,
        }


class ReconciliationComparator:
    """Systematic comparison of reconciliation methods.

    Provides cross-validation, statistical testing, ranking, and
    recommendations for selecting reconciliation methods.

    Example:
        >>> methods = [OLSReconciler(), WLSReconciler()]
        >>> comparator = ReconciliationComparator()
        >>> report = comparator.compare_methods(
        ...     methods=methods,
        ...     forecasts=forecast_df,
        ...     actuals=actual_df,
        ...     hierarchy=hierarchy
        ... )
        >>> print(report.method_rankings)
    """

    def __init__(
        self,
        config: ComparatorConfig | dict[str, Any] | None = None,
    ) -> None:
        """Initialize comparator.

        Args:
            config: Configuration for comparison.
        """
        if config is None:
            self.config = ComparatorConfig()
        elif isinstance(config, dict):
            self.config = ComparatorConfig(**config)
        else:
            self.config = config

        self._last_report: ComparisonReport | None = None
        self._method_names: list[str] = []

        logger.info(
            "Initialized ReconciliationComparator (cv_folds=%d, significance=%.2f)",
            self.config.cv_folds,
            self.config.significance_level,
        )

    def compare_methods(
        self,
        methods: list[BaseReconciler],
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> ComparisonReport:
        """Compare reconciliation methods systematically.

        Args:
            methods: List of reconciliation methods to compare.
            forecasts: Base forecasts DataFrame (time x nodes).
            actuals: Actual values DataFrame (time x nodes).
            hierarchy: HierarchyDefinition.

        Returns:
            ComparisonReport with rankings and analysis.

        Raises:
            ValueError: If methods list is empty or data insufficient.
        """
        if not methods:
            msg = "At least one method must be provided"
            raise ValueError(msg)

        n_samples = len(forecasts)
        min_required = self.config.cv_folds * self.config.min_samples_per_fold
        if n_samples < min_required:
            msg = f"Insufficient data: {n_samples} samples, need at least {min_required}"
            raise ValueError(msg)

        logger.info("Starting comparison of %d methods with %d samples", len(methods), n_samples)
        timestamp = datetime.now(tz=UTC).isoformat()

        # Get method names
        self._method_names = [self._get_method_name(m) for m in methods]

        # Cross-validate all methods
        cv_results = self._cross_validate_methods(
            methods=methods,
            forecasts=forecasts,
            actuals=actuals,
            hierarchy=hierarchy,
        )

        # Build performance matrix
        performance_matrix = self._build_performance_matrix(cv_results)

        # Calculate method statistics
        method_stats = self._calculate_method_stats(cv_results)

        # Pairwise statistical tests
        pairwise_tests = self._statistical_comparison(cv_results)

        # Rank methods
        rankings = self._rank_methods(method_stats)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            rankings, method_stats, pairwise_tests
        )

        report = ComparisonReport(
            method_rankings=rankings,
            performance_matrix=performance_matrix,
            method_stats=method_stats,
            pairwise_tests=pairwise_tests,
            recommendations=recommendations,
            cv_results=cv_results,
            timestamp=timestamp,
            config=self.config.to_dict(),
        )

        self._last_report = report

        logger.info(
            "Comparison complete. Top method: %s",
            min(rankings, key=rankings.get) if rankings else "N/A",
        )

        return report

    @staticmethod
    def _get_method_name(method: BaseReconciler) -> str:
        """Get name for a reconciler method.

        Uses class name to ensure uniqueness across different reconciler types.

        Args:
            method: Reconciler instance.

        Returns:
            Method name string.
        """
        # First, try explicit name attribute (for mocks)
        if hasattr(method, "name") and method.name:
            return method.name

        # Use class name for uniqueness across different reconciler types
        class_name = method.__class__.__name__

        # Optionally append config method if different
        if (
            hasattr(method, "config")
            and method.config is not None
            and hasattr(method.config, "method")
            and method.config.method
        ):
            config_method = method.config.method
            # Append if config method adds info (e.g., "OLSReconciler_ledoit_wolf")
            if config_method.lower() not in class_name.lower():
                return f"{class_name}_{config_method}"

        return class_name

    def _cross_validate_methods(
        self,
        methods: list[BaseReconciler],
        forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> list[CVFoldResult]:
        """Perform time-series cross-validation for all methods.

        Args:
            methods: List of reconciliation methods.
            forecasts: Base forecasts DataFrame.
            actuals: Actual values DataFrame.
            hierarchy: HierarchyDefinition.

        Returns:
            List of CVFoldResult for all methods and folds.
        """
        n_samples = len(forecasts)
        n_folds = self.config.cv_folds

        # Calculate fold sizes for time series split
        fold_size = n_samples // (n_folds + 1)

        cv_results = []

        for fold_idx in range(n_folds):
            # Time series split: train on past, test on future
            train_end = (fold_idx + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)

            if test_end <= test_start:
                continue

            # Get test data
            test_forecasts = forecasts.iloc[test_start:test_end]
            test_actuals = actuals.iloc[test_start:test_end]

            for method_idx, method in enumerate(methods):
                method_name = self._method_names[method_idx]

                try:
                    # Time the reconciliation
                    start = time.time()
                    result = method.reconcile(test_forecasts, hierarchy)
                    computation_time = time.time() - start

                    reconciled = result.reconciled_forecasts

                    # Calculate metrics
                    mape = self._calculate_mape(reconciled, test_actuals)
                    mae = self._calculate_mae(reconciled, test_actuals)
                    rmse = self._calculate_rmse(reconciled, test_actuals)
                    coherence = method.check_coherence(reconciled, hierarchy)

                    cv_results.append(
                        CVFoldResult(
                            fold_idx=fold_idx,
                            method_name=method_name,
                            mape=mape,
                            mae=mae,
                            rmse=rmse,
                            coherence_error=coherence,
                            computation_time=computation_time,
                        )
                    )

                except Exception as e:
                    logger.warning(
                        "Method %s failed on fold %d: %s", method_name, fold_idx, e
                    )
                    cv_results.append(
                        CVFoldResult(
                            fold_idx=fold_idx,
                            method_name=method_name,
                            mape=float("inf"),
                            mae=float("inf"),
                            rmse=float("inf"),
                            coherence_error=float("inf"),
                            computation_time=0.0,
                        )
                    )

        return cv_results

    def _calculate_mape(self, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> float:
        """Calculate Mean Absolute Percentage Error.

        Args:
            forecasts: Forecast values.
            actuals: Actual values.

        Returns:
            MAPE value (decimal).
        """
        common_cols = list(set(forecasts.columns) & set(actuals.columns))
        if not common_cols:
            return float("inf")

        epsilon = 1e-10
        errors = []

        for col in common_cols:
            y_true = actuals[col].to_numpy()
            y_pred = forecasts[col].to_numpy()

            min_len = min(len(y_true), len(y_pred))
            with np.errstate(divide="ignore", invalid="ignore"):
                ape = np.abs((y_true[:min_len] - y_pred[:min_len]) / (np.abs(y_true[:min_len]) + epsilon))
                errors.append(np.nanmean(ape))

        return float(np.nanmean(errors))

    def _calculate_mae(self, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> float:
        """Calculate Mean Absolute Error.

        Args:
            forecasts: Forecast values.
            actuals: Actual values.

        Returns:
            MAE value.
        """
        common_cols = list(set(forecasts.columns) & set(actuals.columns))
        if not common_cols:
            return float("inf")

        errors = []
        for col in common_cols:
            y_true = actuals[col].to_numpy()
            y_pred = forecasts[col].to_numpy()
            min_len = min(len(y_true), len(y_pred))
            errors.append(np.nanmean(np.abs(y_true[:min_len] - y_pred[:min_len])))

        return float(np.nanmean(errors))

    def _calculate_rmse(self, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> float:
        """Calculate Root Mean Squared Error.

        Args:
            forecasts: Forecast values.
            actuals: Actual values.

        Returns:
            RMSE value.
        """
        common_cols = list(set(forecasts.columns) & set(actuals.columns))
        if not common_cols:
            return float("inf")

        errors = []
        for col in common_cols:
            y_true = actuals[col].to_numpy()
            y_pred = forecasts[col].to_numpy()
            min_len = min(len(y_true), len(y_pred))
            mse = np.nanmean((y_true[:min_len] - y_pred[:min_len]) ** 2)
            errors.append(np.sqrt(mse))

        return float(np.nanmean(errors))

    def _build_performance_matrix(
        self,
        cv_results: list[CVFoldResult],
    ) -> list[list[float]]:
        """Build performance matrix (n_methods x n_folds).

        Args:
            cv_results: Cross-validation results.

        Returns:
            Nested list of MAPE scores.
        """
        # Group by method
        method_scores: dict[str, list[float]] = {name: [] for name in self._method_names}

        for result in cv_results:
            score = getattr(result, self.config.metric)
            method_scores[result.method_name].append(score)

        # Build matrix
        return [method_scores[name] for name in self._method_names]

    def _calculate_method_stats(
        self,
        cv_results: list[CVFoldResult],
    ) -> dict[str, dict[str, float]]:
        """Calculate statistics for each method.

        Args:
            cv_results: Cross-validation results.

        Returns:
            Dictionary of method statistics.
        """
        # Group results by method
        method_results: dict[str, list[CVFoldResult]] = {
            name: [] for name in self._method_names
        }
        for result in cv_results:
            method_results[result.method_name].append(result)

        stats_dict = {}

        for method_name, results in method_results.items():
            if not results:
                continue

            metric_values = [getattr(r, self.config.metric) for r in results]
            finite_values = [v for v in metric_values if np.isfinite(v)]

            if finite_values:
                stats_dict[method_name] = {
                    "mean": float(np.mean(finite_values)),
                    "std": float(np.std(finite_values)),
                    "min": float(np.min(finite_values)),
                    "max": float(np.max(finite_values)),
                    "median": float(np.median(finite_values)),
                    "n_successful_folds": len(finite_values),
                    "n_failed_folds": len(metric_values) - len(finite_values),
                    "avg_coherence": float(np.mean([r.coherence_error for r in results if np.isfinite(r.coherence_error)])) if any(np.isfinite(r.coherence_error) for r in results) else float("inf"),
                    "avg_time": float(np.mean([r.computation_time for r in results])),
                }
            else:
                stats_dict[method_name] = {
                    "mean": float("inf"),
                    "std": 0.0,
                    "min": float("inf"),
                    "max": float("inf"),
                    "median": float("inf"),
                    "n_successful_folds": 0,
                    "n_failed_folds": len(results),
                    "avg_coherence": float("inf"),
                    "avg_time": 0.0,
                }

        return stats_dict

    def _statistical_comparison(
        self,
        cv_results: list[CVFoldResult],
    ) -> list[PairwiseTest]:
        """Perform pairwise statistical tests between methods.

        Args:
            cv_results: Cross-validation results.

        Returns:
            List of pairwise test results.
        """
        # Group scores by method
        method_scores: dict[str, list[float]] = {name: [] for name in self._method_names}
        for result in cv_results:
            method_scores[result.method_name].append(getattr(result, self.config.metric))

        pairwise_tests = []
        n_methods = len(self._method_names)
        min_samples_for_test = 3

        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                method_a = self._method_names[i]
                method_b = self._method_names[j]

                scores_a = method_scores[method_a]
                scores_b = method_scores[method_b]

                # Align by fold
                n_folds = min(len(scores_a), len(scores_b))

                if n_folds < min_samples_for_test:
                    pairwise_tests.append(
                        PairwiseTest(
                            method_a=method_a,
                            method_b=method_b,
                            ttest_statistic=0.0,
                            ttest_pvalue=1.0,
                            wilcoxon_statistic=0.0,
                            wilcoxon_pvalue=1.0,
                            significant=False,
                            better_method=None,
                        )
                    )
                    continue

                scores_a = scores_a[:n_folds]
                scores_b = scores_b[:n_folds]

                # Replace inf with large values for tests
                scores_a_clean = [s if np.isfinite(s) else 1e10 for s in scores_a]
                scores_b_clean = [s if np.isfinite(s) else 1e10 for s in scores_b]

                # Paired t-test
                try:
                    t_stat, t_pvalue = stats.ttest_rel(scores_a_clean, scores_b_clean)
                except Exception:
                    t_stat, t_pvalue = 0.0, 1.0

                # Wilcoxon signed-rank test
                try:
                    w_stat, w_pvalue = stats.wilcoxon(scores_a_clean, scores_b_clean)
                except Exception:
                    w_stat, w_pvalue = 0.0, 1.0

                # Determine significance and better method
                significant = t_pvalue < self.config.significance_level
                better_method = None
                if significant:
                    mean_a = np.mean(scores_a_clean)
                    mean_b = np.mean(scores_b_clean)
                    # Lower error is better
                    better_method = method_a if mean_a < mean_b else method_b

                pairwise_tests.append(
                    PairwiseTest(
                        method_a=method_a,
                        method_b=method_b,
                        ttest_statistic=float(t_stat) if np.isfinite(t_stat) else 0.0,
                        ttest_pvalue=float(t_pvalue) if np.isfinite(t_pvalue) else 1.0,
                        wilcoxon_statistic=float(w_stat) if np.isfinite(w_stat) else 0.0,
                        wilcoxon_pvalue=float(w_pvalue) if np.isfinite(w_pvalue) else 1.0,
                        significant=significant,
                        better_method=better_method,
                    )
                )

        return pairwise_tests

    def _rank_methods(
        self,
        method_stats: dict[str, dict[str, float]],
    ) -> dict[str, int]:
        """Rank methods by average performance.

        Args:
            method_stats: Statistics for each method.

        Returns:
            Dictionary mapping method names to ranks (1 = best).
        """
        # Sort by mean metric (lower is better)
        sorted_methods = sorted(
            method_stats.items(),
            key=lambda x: x[1]["mean"],
        )

        return {method: rank + 1 for rank, (method, _) in enumerate(sorted_methods)}

    def _generate_recommendations(
        self,
        rankings: dict[str, int],
        method_stats: dict[str, dict[str, float]],
        pairwise_tests: list[PairwiseTest],
    ) -> list[str]:
        """Generate data-driven recommendations.

        Args:
            rankings: Method rankings.
            method_stats: Method statistics.
            pairwise_tests: Pairwise test results.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        if not rankings:
            recommendations.append("No methods could be evaluated successfully")
            return recommendations

        # Find best method
        best_method = min(rankings, key=rankings.get)
        best_stats = method_stats.get(best_method, {})

        recommendations.append(
            f"Recommended method: {best_method} "
            f"(mean {self.config.metric.upper()}: {best_stats.get('mean', 'N/A'):.4f})"
        )

        # Check if best method is significantly better than others
        significant_wins = sum(
            1
            for t in pairwise_tests
            if t.significant and t.better_method == best_method
        )

        if significant_wins > 0:
            recommendations.append(
                f"{best_method} is statistically significantly better than "
                f"{significant_wins} other method(s)"
            )
        else:
            recommendations.append(
                "No statistically significant differences between methods - "
                "consider using the fastest method"
            )

        # Check for failed methods
        for method, stats_vals in method_stats.items():
            if stats_vals.get("n_failed_folds", 0) > 0:
                recommendations.append(
                    f"Warning: {method} failed on "
                    f"{stats_vals['n_failed_folds']} fold(s)"
                )

        # Coherence-based recommendation
        if self.config.include_coherence:
            coherence_values = {
                m: s.get("avg_coherence", float("inf"))
                for m, s in method_stats.items()
            }
            best_coherence = min(coherence_values, key=coherence_values.get)
            if best_coherence != best_method:
                recommendations.append(
                    f"Note: {best_coherence} has best coherence "
                    f"({coherence_values[best_coherence]:.2e})"
                )

        # Speed-based recommendation
        fastest_method = min(
            method_stats.keys(),
            key=lambda m: method_stats[m].get("avg_time", float("inf")),
        )
        if fastest_method != best_method:
            best_time = method_stats[best_method].get("avg_time", 1)
            fastest_time = method_stats[fastest_method].get("avg_time", 1)
            if fastest_time > 0 and best_time > 0:
                speedup = best_time / fastest_time
                speedup_threshold = 2
                if speedup > speedup_threshold:
                    recommendations.append(
                        f"Consider {fastest_method} if speed is critical "
                        f"({speedup:.1f}x faster than {best_method})"
                    )

        return recommendations

    def get_last_report(self) -> ComparisonReport | None:
        """Get the last comparison report.

        Returns:
            ComparisonReport or None if no comparison done yet.
        """
        return self._last_report

    def save_report(self, path: str | Path) -> None:
        """Save comparison report to disk.

        Args:
            path: Output file path.

        Raises:
            ValueError: If no report to save.
        """
        if self._last_report is None:
            msg = "No report to save - run compare_methods first"
            raise ValueError(msg)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self._last_report.to_dict(), f, indent=2)

        logger.info("Saved comparison report to %s", path)

    @classmethod
    def load_report(cls, path: str | Path) -> ComparisonReport:
        """Load comparison report from disk.

        Args:
            path: Path to saved report.

        Returns:
            ComparisonReport instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(path)

        if not path.exists():
            msg = f"Report file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open() as f:
            data = json.load(f)

        # Reconstruct dataclasses
        cv_results = [CVFoldResult(**r) for r in data.get("cv_results", [])]
        pairwise_tests = [PairwiseTest(**t) for t in data.get("pairwise_tests", [])]

        return ComparisonReport(
            method_rankings=data["method_rankings"],
            performance_matrix=data["performance_matrix"],
            method_stats=data["method_stats"],
            pairwise_tests=pairwise_tests,
            recommendations=data["recommendations"],
            cv_results=cv_results,
            timestamp=data.get("timestamp", ""),
            config=data.get("config", {}),
        )
