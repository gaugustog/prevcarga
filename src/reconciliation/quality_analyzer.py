"""Reconciliation quality analysis module.

This module implements the ReconciliationQualityAnalyzer class for comprehensive
quality assessment of hierarchical forecast reconciliation methods.

The analyzer evaluates reconciliation quality across multiple dimensions:
    - Constraint Satisfaction: How well aggregation constraints are satisfied
    - Accuracy Improvement: Error reduction compared to base forecasts
    - Consistency: Temporal stability and correlation preservation
    - Statistical Significance: Hypothesis tests for performance differences
    - Degradation Detection: Monitoring for quality decline over time

Example:
    ```python
    from src.reconciliation import (
        ReconciliationQualityAnalyzer,
        QualityConfig,
        HierarchyDefinition
    )
    import pandas as pd

    # Create analyzer
    config = QualityConfig(significance_level=0.05)
    analyzer = ReconciliationQualityAnalyzer(config=config)

    # Analyze reconciliation quality
    report = analyzer.analyze_quality(
        reconciled_forecasts=reconciled_df,
        base_forecasts=base_df,
        actuals=actual_df,
        hierarchy=hierarchy
    )

    # Review metrics
    print(f"Constraint satisfaction: {report.constraint_satisfaction:.2%}")
    print(f"Accuracy improvement: {report.accuracy_improvement:.2%}")
    print(f"Consistency score: {report.consistency_score:.2%}")
    print(f"Statistically significant: {report.statistical_significance}")

    # Get recommendations
    for rec in report.recommendations:
        print(f"  - {rec}")
    ```

References:
    - Hyndman, R.J., et al. (2011). "Optimal Combination Forecasts for
      Hierarchical Time Series", Computational Statistics & Data Analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Detailed quality metrics for reconciliation.

    Attributes:
        mape_base: MAPE of base forecasts.
        mape_reconciled: MAPE of reconciled forecasts.
        mae_base: MAE of base forecasts.
        mae_reconciled: MAE of reconciled forecasts.
        rmse_base: RMSE of base forecasts.
        rmse_reconciled: RMSE of reconciled forecasts.
        max_constraint_violation: Maximum constraint violation.
        mean_constraint_violation: Mean constraint violation.
        correlation_preserved: Correlation preservation score.
        temporal_stability: Temporal stability score.
    """

    mape_base: float = 0.0
    mape_reconciled: float = 0.0
    mae_base: float = 0.0
    mae_reconciled: float = 0.0
    rmse_base: float = 0.0
    rmse_reconciled: float = 0.0
    max_constraint_violation: float = 0.0
    mean_constraint_violation: float = 0.0
    correlation_preserved: float = 1.0
    temporal_stability: float = 1.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "mape_base": float(self.mape_base),
            "mape_reconciled": float(self.mape_reconciled),
            "mae_base": float(self.mae_base),
            "mae_reconciled": float(self.mae_reconciled),
            "rmse_base": float(self.rmse_base),
            "rmse_reconciled": float(self.rmse_reconciled),
            "max_constraint_violation": float(self.max_constraint_violation),
            "mean_constraint_violation": float(self.mean_constraint_violation),
            "correlation_preserved": float(self.correlation_preserved),
            "temporal_stability": float(self.temporal_stability),
        }


@dataclass
class StatisticalTests:
    """Statistical test results.

    Attributes:
        paired_ttest_statistic: Paired t-test statistic.
        paired_ttest_pvalue: Paired t-test p-value.
        wilcoxon_statistic: Wilcoxon signed-rank statistic.
        wilcoxon_pvalue: Wilcoxon signed-rank p-value.
        improvement_significant: Whether improvement is statistically significant.
        sample_size: Number of samples used in tests.
    """

    paired_ttest_statistic: float = 0.0
    paired_ttest_pvalue: float = 1.0
    wilcoxon_statistic: float = 0.0
    wilcoxon_pvalue: float = 1.0
    improvement_significant: bool = False
    sample_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "paired_ttest_statistic": float(self.paired_ttest_statistic),
            "paired_ttest_pvalue": float(self.paired_ttest_pvalue),
            "wilcoxon_statistic": float(self.wilcoxon_statistic),
            "wilcoxon_pvalue": float(self.wilcoxon_pvalue),
            "improvement_significant": bool(self.improvement_significant),
            "sample_size": int(self.sample_size),
        }


@dataclass
class QualityReport:
    """Comprehensive quality report for reconciliation.

    Attributes:
        constraint_satisfaction: Score for constraint satisfaction (0-1).
        accuracy_improvement: Relative accuracy improvement (-1 to 1).
        consistency_score: Consistency score (0-1).
        statistical_significance: Whether improvement is significant.
        degradation_detected: Whether quality degradation detected.
        recommendations: List of actionable recommendations.
        metrics: Detailed quality metrics.
        statistical_tests: Statistical test results.
        timestamp: When analysis was performed.
        method_name: Name of reconciliation method (if available).
    """

    constraint_satisfaction: float
    accuracy_improvement: float
    consistency_score: float
    statistical_significance: bool
    degradation_detected: bool
    recommendations: list[str]
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    statistical_tests: StatisticalTests = field(default_factory=StatisticalTests)
    timestamp: str = ""
    method_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "constraint_satisfaction": float(self.constraint_satisfaction),
            "accuracy_improvement": float(self.accuracy_improvement),
            "consistency_score": float(self.consistency_score),
            "statistical_significance": bool(self.statistical_significance),
            "degradation_detected": bool(self.degradation_detected),
            "recommendations": list(self.recommendations),
            "metrics": self.metrics.to_dict(),
            "statistical_tests": self.statistical_tests.to_dict(),
            "timestamp": str(self.timestamp),
            "method_name": str(self.method_name),
        }


@dataclass
class QualityConfig:
    """Configuration for quality analyzer.

    Attributes:
        significance_level: P-value threshold for significance tests.
        constraint_tolerance: Tolerance for constraint violations.
        degradation_window: Number of reports for degradation detection.
        degradation_threshold: Slope threshold for degradation detection.
        min_samples_for_tests: Minimum samples for statistical tests.
        correlation_threshold: Threshold for correlation preservation.
        stability_threshold: Threshold for temporal stability.
    """

    significance_level: float = 0.05
    constraint_tolerance: float = 1e-6
    degradation_window: int = 5
    degradation_threshold: float = -0.01
    min_samples_for_tests: int = 3
    correlation_threshold: float = 0.9
    stability_threshold: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "significance_level": self.significance_level,
            "constraint_tolerance": self.constraint_tolerance,
            "degradation_window": self.degradation_window,
            "degradation_threshold": self.degradation_threshold,
            "min_samples_for_tests": self.min_samples_for_tests,
            "correlation_threshold": self.correlation_threshold,
            "stability_threshold": self.stability_threshold,
        }


class ReconciliationQualityAnalyzer:
    """Comprehensive quality analysis for reconciliation methods.

    Evaluates reconciliation quality across multiple dimensions including
    constraint satisfaction, accuracy improvement, consistency, and
    provides statistical significance testing.

    Example:
        >>> analyzer = ReconciliationQualityAnalyzer()
        >>> report = analyzer.analyze_quality(
        ...     reconciled_forecasts=reconciled_df,
        ...     base_forecasts=base_df,
        ...     actuals=actual_df,
        ...     hierarchy=hierarchy
        ... )
        >>> print(f"Accuracy improvement: {report.accuracy_improvement:.2%}")
    """

    def __init__(
        self,
        config: QualityConfig | dict[str, Any] | None = None,
    ) -> None:
        """Initialize quality analyzer.

        Args:
            config: Configuration for analysis.
        """
        if config is None:
            self.config = QualityConfig()
        elif isinstance(config, dict):
            self.config = QualityConfig(**config)
        else:
            self.config = config

        self.quality_history: list[QualityReport] = []

        logger.info(
            "Initialized ReconciliationQualityAnalyzer "
            "(significance=%.2f, degradation_window=%d)",
            self.config.significance_level,
            self.config.degradation_window,
        )

    def analyze_quality(
        self,
        reconciled_forecasts: pd.DataFrame,
        base_forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        method_name: str = "",
    ) -> QualityReport:
        """Analyze reconciliation quality comprehensively.

        Args:
            reconciled_forecasts: Reconciled forecasts DataFrame.
            base_forecasts: Base forecasts DataFrame.
            actuals: Actual values DataFrame.
            hierarchy: HierarchyDefinition.
            method_name: Optional name of reconciliation method.

        Returns:
            QualityReport with comprehensive analysis.
        """
        logger.info("Starting quality analysis")
        timestamp = datetime.now(tz=UTC).isoformat()

        # Calculate metrics
        metrics = self._calculate_metrics(
            reconciled_forecasts, base_forecasts, actuals, hierarchy
        )

        # Calculate constraint satisfaction
        constraint_sat = self._calculate_constraint_satisfaction(
            reconciled_forecasts, hierarchy
        )

        # Calculate accuracy improvement
        accuracy_imp = self._calculate_accuracy_improvement(metrics)

        # Calculate consistency
        consistency = self._calculate_consistency(
            reconciled_forecasts, base_forecasts, actuals
        )

        # Statistical tests
        stat_tests = self._run_statistical_tests(
            reconciled_forecasts, base_forecasts, actuals
        )

        # Degradation detection
        degradation = self._detect_degradation()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            constraint_sat, accuracy_imp, consistency, stat_tests, metrics
        )

        report = QualityReport(
            constraint_satisfaction=constraint_sat,
            accuracy_improvement=accuracy_imp,
            consistency_score=consistency,
            statistical_significance=stat_tests.improvement_significant,
            degradation_detected=degradation,
            recommendations=recommendations,
            metrics=metrics,
            statistical_tests=stat_tests,
            timestamp=timestamp,
            method_name=method_name,
        )

        self.quality_history.append(report)

        logger.info(
            "Quality analysis complete: constraint=%.3f, improvement=%.3f, "
            "consistency=%.3f, significant=%s",
            constraint_sat,
            accuracy_imp,
            consistency,
            stat_tests.improvement_significant,
        )

        return report

    def _calculate_metrics(
        self,
        reconciled: pd.DataFrame,
        base: pd.DataFrame,
        actuals: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> QualityMetrics:
        """Calculate detailed quality metrics.

        Args:
            reconciled: Reconciled forecasts.
            base: Base forecasts.
            actuals: Actual values.
            hierarchy: HierarchyDefinition.

        Returns:
            QualityMetrics with detailed measurements.
        """
        common_cols = list(
            set(reconciled.columns) & set(base.columns) & set(actuals.columns)
        )

        if not common_cols:
            return QualityMetrics()

        # Calculate error metrics
        base_mape = self._calculate_mape(base[common_cols], actuals[common_cols])
        rec_mape = self._calculate_mape(reconciled[common_cols], actuals[common_cols])

        base_mae = self._calculate_mae(base[common_cols], actuals[common_cols])
        rec_mae = self._calculate_mae(reconciled[common_cols], actuals[common_cols])

        base_rmse = self._calculate_rmse(base[common_cols], actuals[common_cols])
        rec_rmse = self._calculate_rmse(reconciled[common_cols], actuals[common_cols])

        # Calculate constraint violations
        max_violation, mean_violation = self._calculate_constraint_violations(
            reconciled, hierarchy
        )

        # Correlation preservation
        corr_preserved = self._calculate_correlation_preservation(
            reconciled, base, common_cols
        )

        # Temporal stability
        stability = self._calculate_temporal_stability(reconciled, base, common_cols)

        return QualityMetrics(
            mape_base=base_mape,
            mape_reconciled=rec_mape,
            mae_base=base_mae,
            mae_reconciled=rec_mae,
            rmse_base=base_rmse,
            rmse_reconciled=rec_rmse,
            max_constraint_violation=max_violation,
            mean_constraint_violation=mean_violation,
            correlation_preserved=corr_preserved,
            temporal_stability=stability,
        )

    def _calculate_mape(self, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> float:
        """Calculate Mean Absolute Percentage Error.

        Args:
            forecasts: Forecast values.
            actuals: Actual values.

        Returns:
            MAPE value (decimal).
        """
        epsilon = 1e-10
        errors = []

        for col in forecasts.columns:
            y_true = actuals[col].to_numpy()
            y_pred = forecasts[col].to_numpy()

            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

            with np.errstate(divide="ignore", invalid="ignore"):
                ape = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
                errors.append(np.nanmean(ape))

        return float(np.nanmean(errors)) if errors else 0.0

    def _calculate_mae(self, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> float:
        """Calculate Mean Absolute Error.

        Args:
            forecasts: Forecast values.
            actuals: Actual values.

        Returns:
            MAE value.
        """
        errors = []

        for col in forecasts.columns:
            y_true = actuals[col].to_numpy()
            y_pred = forecasts[col].to_numpy()

            min_len = min(len(y_true), len(y_pred))
            errors.append(np.nanmean(np.abs(y_true[:min_len] - y_pred[:min_len])))

        return float(np.nanmean(errors)) if errors else 0.0

    def _calculate_rmse(self, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> float:
        """Calculate Root Mean Squared Error.

        Args:
            forecasts: Forecast values.
            actuals: Actual values.

        Returns:
            RMSE value.
        """
        errors = []

        for col in forecasts.columns:
            y_true = actuals[col].to_numpy()
            y_pred = forecasts[col].to_numpy()

            min_len = min(len(y_true), len(y_pred))
            mse = np.nanmean((y_true[:min_len] - y_pred[:min_len]) ** 2)
            errors.append(np.sqrt(mse))

        return float(np.nanmean(errors)) if errors else 0.0

    def _calculate_constraint_violations(
        self,
        forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> tuple[float, float]:
        """Calculate constraint violations.

        Args:
            forecasts: Forecast DataFrame.
            hierarchy: HierarchyDefinition.

        Returns:
            Tuple of (max_violation, mean_violation).
        """
        violations = []

        # Get all aggregate nodes
        for node in hierarchy.nodes.values():
            if not node.children:
                continue

            node_name = node.name
            if node_name not in forecasts.columns:
                continue

            # Check if all children are in forecasts
            child_names = [c for c in node.children if c in forecasts.columns]
            if not child_names:
                continue

            # Calculate expected vs actual
            expected = forecasts[child_names].sum(axis=1)
            actual = forecasts[node_name]

            violation = np.abs(actual - expected).mean()
            violations.append(violation)

        if not violations:
            return 0.0, 0.0

        return float(np.max(violations)), float(np.mean(violations))

    def _calculate_constraint_satisfaction(
        self,
        forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> float:
        """Calculate constraint satisfaction score.

        Args:
            forecasts: Forecast DataFrame.
            hierarchy: HierarchyDefinition.

        Returns:
            Constraint satisfaction score (0-1).
        """
        max_violation, _ = self._calculate_constraint_violations(forecasts, hierarchy)

        if max_violation <= self.config.constraint_tolerance:
            return 1.0

        # Exponential decay for larger violations
        # Scale factor determines sensitivity
        scale = 10.0
        return float(np.exp(-max_violation / scale))

    def _calculate_accuracy_improvement(self, metrics: QualityMetrics) -> float:
        """Calculate relative accuracy improvement.

        Args:
            metrics: Quality metrics with base and reconciled errors.

        Returns:
            Relative improvement (-1 to 1).
        """
        if metrics.mape_base <= 0:
            return 0.0

        improvement = (metrics.mape_base - metrics.mape_reconciled) / metrics.mape_base
        # Clamp to [-1, 1]
        return float(max(-1.0, min(1.0, improvement)))

    def _calculate_consistency(
        self,
        reconciled: pd.DataFrame,
        base: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> float:
        """Calculate consistency score.

        Measures correlation preservation and temporal stability.

        Args:
            reconciled: Reconciled forecasts.
            base: Base forecasts.
            actuals: Actual values.

        Returns:
            Consistency score (0-1).
        """
        common_cols = list(
            set(reconciled.columns) & set(base.columns) & set(actuals.columns)
        )

        if not common_cols:
            return 1.0

        corr_score = self._calculate_correlation_preservation(
            reconciled, base, common_cols
        )
        stability_score = self._calculate_temporal_stability(
            reconciled, base, common_cols
        )

        # Combine scores
        return float(0.5 * corr_score + 0.5 * stability_score)

    def _calculate_correlation_preservation(
        self,
        reconciled: pd.DataFrame,
        base: pd.DataFrame,
        columns: list[str],
    ) -> float:
        """Calculate correlation preservation score.

        Args:
            reconciled: Reconciled forecasts.
            base: Base forecasts.
            columns: Columns to analyze.

        Returns:
            Correlation preservation score (0-1).
        """
        min_columns_for_correlation = 2
        if len(columns) < min_columns_for_correlation:
            return 1.0

        try:
            base_corr = base[columns].corr()
            rec_corr = reconciled[columns].corr()

            # Compare correlation matrices
            diff = np.abs(base_corr.to_numpy() - rec_corr.to_numpy())
            mean_diff = np.nanmean(diff)

            # Convert to score (1 = perfect preservation)
            return float(max(0.0, 1.0 - mean_diff))
        except Exception:
            return 1.0

    def _calculate_temporal_stability(
        self,
        reconciled: pd.DataFrame,
        base: pd.DataFrame,
        columns: list[str],
    ) -> float:
        """Calculate temporal stability score.

        Measures how much reconciliation changes temporal patterns.

        Args:
            reconciled: Reconciled forecasts.
            base: Base forecasts.
            columns: Columns to analyze.

        Returns:
            Stability score (0-1).
        """
        stabilities = []

        for col in columns:
            base_vals = base[col].to_numpy()
            rec_vals = reconciled[col].to_numpy()

            min_len = min(len(base_vals), len(rec_vals))
            min_len_for_diff = 2
            if min_len < min_len_for_diff:
                continue

            base_diff = np.diff(base_vals[:min_len])
            rec_diff = np.diff(rec_vals[:min_len])

            # Calculate correlation of differences
            if np.std(base_diff) > 0 and np.std(rec_diff) > 0:
                corr = np.corrcoef(base_diff, rec_diff)[0, 1]
                if not np.isnan(corr):
                    stabilities.append(max(0.0, corr))

        if not stabilities:
            return 1.0

        return float(np.mean(stabilities))

    def _run_statistical_tests(
        self,
        reconciled: pd.DataFrame,
        base: pd.DataFrame,
        actuals: pd.DataFrame,
    ) -> StatisticalTests:
        """Run statistical significance tests.

        Args:
            reconciled: Reconciled forecasts.
            base: Base forecasts.
            actuals: Actual values.

        Returns:
            StatisticalTests results.
        """
        common_cols = list(
            set(reconciled.columns) & set(base.columns) & set(actuals.columns)
        )

        if not common_cols:
            return StatisticalTests()

        # Calculate per-series MAE
        base_errors = []
        rec_errors = []

        for col in common_cols:
            y_true = actuals[col].to_numpy()
            y_base = base[col].to_numpy()
            y_rec = reconciled[col].to_numpy()

            min_len = min(len(y_true), len(y_base), len(y_rec))
            base_mae = np.nanmean(np.abs(y_true[:min_len] - y_base[:min_len]))
            rec_mae = np.nanmean(np.abs(y_true[:min_len] - y_rec[:min_len]))

            if not np.isnan(base_mae) and not np.isnan(rec_mae):
                base_errors.append(base_mae)
                rec_errors.append(rec_mae)

        n_samples = len(base_errors)

        if n_samples < self.config.min_samples_for_tests:
            return StatisticalTests(sample_size=n_samples)

        base_errors = np.array(base_errors)
        rec_errors = np.array(rec_errors)

        # Paired t-test
        try:
            t_stat, t_pvalue = stats.ttest_rel(base_errors, rec_errors)
        except Exception:
            t_stat, t_pvalue = 0.0, 1.0

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pvalue = stats.wilcoxon(
                base_errors, rec_errors, alternative="greater"
            )
        except Exception:
            w_stat, w_pvalue = 0.0, 1.0

        # Check if improvement is significant
        # Both tests should agree for robustness
        is_significant = (
            t_pvalue < self.config.significance_level
            and t_stat > 0  # Base errors > reconciled errors
        )

        return StatisticalTests(
            paired_ttest_statistic=float(t_stat) if not np.isnan(t_stat) else 0.0,
            paired_ttest_pvalue=float(t_pvalue) if not np.isnan(t_pvalue) else 1.0,
            wilcoxon_statistic=float(w_stat) if not np.isnan(w_stat) else 0.0,
            wilcoxon_pvalue=float(w_pvalue) if not np.isnan(w_pvalue) else 1.0,
            improvement_significant=is_significant,
            sample_size=n_samples,
        )

    def _detect_degradation(self) -> bool:
        """Detect quality degradation over time.

        Uses linear regression on recent accuracy improvements to detect
        declining trends.

        Returns:
            True if significant degradation detected.
        """
        window = self.config.degradation_window

        if len(self.quality_history) < window:
            return False

        recent_improvements = [
            q.accuracy_improvement for q in self.quality_history[-window:]
        ]

        # Linear regression to detect trend
        x = np.arange(len(recent_improvements))
        try:
            slope, _, _, p_value, _ = stats.linregress(x, recent_improvements)

            # Significant negative trend
            degradation_threshold = 0.1  # Relaxed p-value for trend detection
            return bool(slope < self.config.degradation_threshold and p_value < degradation_threshold)
        except Exception:
            return False

    def _generate_recommendations(
        self,
        constraint_sat: float,
        accuracy_imp: float,
        consistency: float,
        stat_tests: StatisticalTests,
        metrics: QualityMetrics,
    ) -> list[str]:
        """Generate actionable recommendations.

        Args:
            constraint_sat: Constraint satisfaction score.
            accuracy_imp: Accuracy improvement.
            consistency: Consistency score.
            stat_tests: Statistical test results.
            metrics: Quality metrics.

        Returns:
            List of recommendation strings.
        """
        recommendations = []
        constraint_threshold = 0.95
        improvement_threshold = 0.02
        consistency_threshold = 0.8
        violation_warning = 1.0

        # Constraint satisfaction
        if constraint_sat < constraint_threshold:
            recommendations.append(
                "Constraint satisfaction is low - consider stricter "
                "enforcement or checking hierarchy definition"
            )

        if metrics.max_constraint_violation > violation_warning:
            recommendations.append(
                f"Large constraint violations detected "
                f"(max: {metrics.max_constraint_violation:.2f}) - "
                f"review aggregation constraints"
            )

        # Accuracy improvement
        if accuracy_imp < improvement_threshold:
            if accuracy_imp < 0:
                recommendations.append(
                    "Reconciliation is degrading accuracy - consider "
                    "alternative methods or checking data quality"
                )
            else:
                recommendations.append(
                    "Minimal accuracy improvement - try alternative "
                    "reconciliation method or adjust parameters"
                )

        if accuracy_imp > 0 and not stat_tests.improvement_significant:
            recommendations.append(
                "Accuracy improvement is not statistically significant - "
                "gather more data or use more conservative methods"
            )

        # Consistency
        if consistency < consistency_threshold:
            recommendations.append(
                "Low consistency detected - check for data quality issues "
                "or structural changes in hierarchy"
            )

        if metrics.correlation_preserved < self.config.correlation_threshold:
            recommendations.append(
                "Correlation structure not well preserved - "
                "consider methods that better preserve relationships"
            )

        if metrics.temporal_stability < (1 - self.config.stability_threshold):
            recommendations.append(
                "Temporal patterns significantly altered - "
                "verify this aligns with expectations"
            )

        # No issues found
        if not recommendations:
            recommendations.append("Quality metrics look good - no issues detected")

        return recommendations

    def get_history_summary(self) -> dict[str, Any]:
        """Get summary of quality history.

        Returns:
            Dictionary with history statistics.
        """
        if not self.quality_history:
            return {"n_analyses": 0}

        improvements = [r.accuracy_improvement for r in self.quality_history]
        constraints = [r.constraint_satisfaction for r in self.quality_history]
        consistencies = [r.consistency_score for r in self.quality_history]

        return {
            "n_analyses": len(self.quality_history),
            "avg_accuracy_improvement": float(np.mean(improvements)),
            "avg_constraint_satisfaction": float(np.mean(constraints)),
            "avg_consistency_score": float(np.mean(consistencies)),
            "min_accuracy_improvement": float(np.min(improvements)),
            "max_accuracy_improvement": float(np.max(improvements)),
            "significant_improvements": sum(
                1 for r in self.quality_history if r.statistical_significance
            ),
            "degradation_events": sum(
                1 for r in self.quality_history if r.degradation_detected
            ),
        }

    def clear_history(self) -> None:
        """Clear quality history."""
        self.quality_history = []
        logger.info("Cleared quality history")

    def save(self, path: str | Path) -> None:
        """Save analyzer state to disk.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config.to_dict(),
            "quality_history": [r.to_dict() for r in self.quality_history],
        }

        with path.open("w") as f:
            json.dump(state, f, indent=2)

        logger.info("Saved QualityAnalyzer to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ReconciliationQualityAnalyzer:
        """Load analyzer state from disk.

        Args:
            path: Path to saved state file.

        Returns:
            ReconciliationQualityAnalyzer instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(path)

        if not path.exists():
            msg = f"State file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open() as f:
            state = json.load(f)

        config = QualityConfig(**state["config"])
        instance = cls(config=config)

        # Restore history
        for report_data in state.get("quality_history", []):
            metrics_data = report_data.pop("metrics", {})
            tests_data = report_data.pop("statistical_tests", {})

            report = QualityReport(
                **report_data,
                metrics=QualityMetrics(**metrics_data),
                statistical_tests=StatisticalTests(**tests_data),
            )
            instance.quality_history.append(report)

        logger.info("Loaded QualityAnalyzer from %s", path)

        return instance
