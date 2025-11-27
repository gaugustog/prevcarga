"""Automatic reconciliation method selection system.

This module implements the ReconcilerSelector class for automatically selecting
the optimal hierarchical reconciliation method based on performance metrics,
data characteristics, and computational efficiency.

The selection system evaluates multiple methods using:
    - Accuracy: Forecast error metrics (MAPE, MAE, RMSE)
    - Consistency: Constraint satisfaction (aggregation coherence)
    - Efficiency: Computational performance (time)

Selection Strategy:
    1. Evaluate all methods on rolling windows
    2. Calculate weighted score: w_acc*accuracy + w_cons*consistency + w_eff*efficiency
    3. Select method with highest score
    4. Adapt to changing data characteristics

Example:
    ```python
    from src.reconciliation import (
        MintReconciler, OLSReconciler, WLSReconciler,
        ReconcilerSelector, SelectorConfig
    )

    # Define candidate methods
    methods = [
        MintReconciler(),
        OLSReconciler(),
        WLSReconciler()
    ]

    # Configure selector with custom weights
    config = SelectorConfig(
        accuracy_weight=0.6,
        consistency_weight=0.3,
        efficiency_weight=0.1,
        rolling_window_size=10
    )

    # Create selector
    selector = ReconcilerSelector(methods=methods, config=config)

    # Select best method based on evaluation data
    best_method = selector.select_method(
        base_forecasts=forecasts_df,
        hierarchy=hierarchy,
        actuals=actuals_df
    )

    # Use selected method
    result = best_method.reconcile(new_forecasts, hierarchy)
    ```

References:
    - Hyndman, R.J., & Athanasopoulos, G. (2021). "Forecasting: Principles
      and Practice", 3rd edition, OTexts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.reconciliation.base_reconciler import BaseReconciler
from src.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MethodPerformance:
    """Performance metrics for a reconciliation method.

    Attributes:
        method_name: Name of the reconciliation method.
        accuracy_score: Accuracy score (higher is better, 0-1 range).
        consistency_score: Consistency score (higher is better, 0-1 range).
        efficiency_score: Efficiency score (higher is better, 0-1 range).
        overall_score: Weighted overall score.
        mape: Mean Absolute Percentage Error.
        mae: Mean Absolute Error.
        rmse: Root Mean Squared Error.
        coherence_error: Aggregation coherence error.
        computation_time: Time to reconcile (seconds).
        evaluation_timestamp: When evaluation was performed.
    """

    method_name: str
    accuracy_score: float
    consistency_score: float
    efficiency_score: float
    overall_score: float
    mape: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    coherence_error: float = 0.0
    computation_time: float = 0.0
    evaluation_timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_name": self.method_name,
            "accuracy_score": self.accuracy_score,
            "consistency_score": self.consistency_score,
            "efficiency_score": self.efficiency_score,
            "overall_score": self.overall_score,
            "mape": self.mape,
            "mae": self.mae,
            "rmse": self.rmse,
            "coherence_error": self.coherence_error,
            "computation_time": self.computation_time,
            "evaluation_timestamp": self.evaluation_timestamp,
        }


@dataclass
class SelectionResult:
    """Result of method selection.

    Attributes:
        selected_method_name: Name of selected method.
        selected_method_index: Index of selected method in methods list.
        all_performances: Performance metrics for all evaluated methods.
        selection_reason: Human-readable explanation of selection.
        selection_timestamp: When selection was made.
        config_used: Configuration used for selection.
    """

    selected_method_name: str
    selected_method_index: int
    all_performances: list[MethodPerformance]
    selection_reason: str
    selection_timestamp: str
    config_used: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selected_method_name": self.selected_method_name,
            "selected_method_index": self.selected_method_index,
            "all_performances": [p.to_dict() for p in self.all_performances],
            "selection_reason": self.selection_reason,
            "selection_timestamp": self.selection_timestamp,
            "config_used": self.config_used,
        }


@dataclass
class SelectorConfig:
    """Configuration for reconciler selector.

    Attributes:
        accuracy_weight: Weight for accuracy in overall score (0-1).
        consistency_weight: Weight for consistency in overall score (0-1).
        efficiency_weight: Weight for efficiency in overall score (0-1).
        rolling_window_size: Number of historical evaluations to consider.
        target_computation_time: Target time for efficiency scoring (seconds).
        mape_threshold: MAPE threshold for good accuracy (decimal).
        coherence_tolerance: Tolerance for coherence violations.
        min_evaluation_samples: Minimum samples needed for evaluation.
        use_rolling_average: Whether to use rolling average for selection.
        adapt_to_patterns: Whether to adapt to changing data patterns.
    """

    accuracy_weight: float = 0.6
    consistency_weight: float = 0.3
    efficiency_weight: float = 0.1
    rolling_window_size: int = 10
    target_computation_time: float = 5.0
    mape_threshold: float = 0.05
    coherence_tolerance: float = 1e-6
    min_evaluation_samples: int = 10
    use_rolling_average: bool = True
    adapt_to_patterns: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Normalize weights
        total_weight = self.accuracy_weight + self.consistency_weight + self.efficiency_weight
        if total_weight > 0:
            self.accuracy_weight /= total_weight
            self.consistency_weight /= total_weight
            self.efficiency_weight /= total_weight

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy_weight": self.accuracy_weight,
            "consistency_weight": self.consistency_weight,
            "efficiency_weight": self.efficiency_weight,
            "rolling_window_size": self.rolling_window_size,
            "target_computation_time": self.target_computation_time,
            "mape_threshold": self.mape_threshold,
            "coherence_tolerance": self.coherence_tolerance,
            "min_evaluation_samples": self.min_evaluation_samples,
            "use_rolling_average": self.use_rolling_average,
            "adapt_to_patterns": self.adapt_to_patterns,
        }


class ReconcilerSelector:
    """Automatic selection of optimal reconciliation method.

    Evaluates multiple reconciliation methods and selects the best one based on:
        - Accuracy: How well reconciled forecasts match actuals
        - Consistency: How well constraints are satisfied
        - Efficiency: How fast the method is

    The selector uses a weighted scoring system with configurable weights
    and supports rolling window evaluation to adapt to changing data patterns.

    Example:
        >>> from src.reconciliation import MintReconciler, OLSReconciler
        >>> methods = [MintReconciler(), OLSReconciler()]
        >>> selector = ReconcilerSelector(methods)
        >>> best = selector.select_method(forecasts, hierarchy, actuals)
        >>> result = best.reconcile(forecasts, hierarchy)
    """

    def __init__(
        self,
        methods: list[BaseReconciler],
        config: SelectorConfig | dict[str, Any] | None = None,
    ) -> None:
        """Initialize reconciler selector.

        Args:
            methods: List of reconciliation methods to evaluate.
            config: Configuration for selection (SelectorConfig or dict).

        Raises:
            ValueError: If methods list is empty.
        """
        if not methods:
            msg = "At least one method must be provided"
            raise ValueError(msg)

        self.methods = methods
        self.method_names = [self._get_method_name(m) for m in methods]

        # Parse configuration
        if config is None:
            self.config = SelectorConfig()
        elif isinstance(config, dict):
            self.config = SelectorConfig(**config)
        else:
            self.config = config

        # State
        self.performance_history: list[list[MethodPerformance]] = []
        self._last_selection: SelectionResult | None = None
        self._selection_counts: dict[str, int] = dict.fromkeys(self.method_names, 0)

        logger.info(
            "Initialized ReconcilerSelector with %d methods: %s",
            len(methods),
            ", ".join(self.method_names),
        )

    @staticmethod
    def _get_method_name(method: BaseReconciler) -> str:
        """Get name for a reconciler method.

        Tries multiple sources: name attribute, config.method, class name.

        Args:
            method: Reconciler instance.

        Returns:
            Method name string.
        """
        # Try name attribute first (for mocks and custom implementations)
        if hasattr(method, "name") and method.name:
            return method.name

        # Try config.method
        if (
            hasattr(method, "config")
            and method.config is not None
            and hasattr(method.config, "method")
            and method.config.method
        ):
            return method.config.method

        # Fall back to class name
        return method.__class__.__name__

    def select_method(
        self,
        base_forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        actuals: pd.DataFrame,
        **kwargs: Any,
    ) -> BaseReconciler:
        """Select optimal reconciliation method.

        Evaluates all methods on the provided data and selects the one
        with the highest weighted score.

        Args:
            base_forecasts: Base forecasts DataFrame (time x nodes).
            hierarchy: HierarchyDefinition with structure.
            actuals: Actual values for evaluation (time x nodes).
            **kwargs: Additional arguments passed to reconcile methods.

        Returns:
            Selected reconciler with best performance.

        Raises:
            ValueError: If evaluation fails for all methods.
        """
        start_time = time.time()
        logger.info("Starting method selection evaluation")

        performances = []
        timestamp = pd.Timestamp.now().isoformat()

        for method in self.methods:
            try:
                perf = self._evaluate_method(
                    method=method,
                    base_forecasts=base_forecasts,
                    hierarchy=hierarchy,
                    actuals=actuals,
                    timestamp=timestamp,
                    **kwargs,
                )
                performances.append(perf)
            except Exception as e:
                method_name = self._get_method_name(method)
                logger.warning("Method %s failed evaluation: %s", method_name, e)
                # Add dummy performance with zero score
                performances.append(
                    MethodPerformance(
                        method_name=method_name,
                        accuracy_score=0.0,
                        consistency_score=0.0,
                        efficiency_score=0.0,
                        overall_score=0.0,
                        evaluation_timestamp=timestamp,
                    )
                )

        if all(p.overall_score == 0 for p in performances):
            msg = "All methods failed evaluation"
            raise ValueError(msg)

        # Store in history
        self.performance_history.append(performances)

        # Trim history to window size
        window_size = self.config.rolling_window_size
        if len(self.performance_history) > window_size:
            self.performance_history = self.performance_history[-window_size:]

        # Calculate scores (with optional rolling average)
        if self.config.use_rolling_average and len(self.performance_history) > 1:
            final_scores = self._calculate_rolling_scores()
        else:
            final_scores = [p.overall_score for p in performances]

        # Select best method
        best_idx = int(np.argmax(final_scores))
        selected_method = self.methods[best_idx]

        # Generate selection reason
        reason = self._generate_selection_reason(performances, best_idx, final_scores)

        # Store selection result
        self._last_selection = SelectionResult(
            selected_method_name=self._get_method_name(selected_method),
            selected_method_index=best_idx,
            all_performances=performances,
            selection_reason=reason,
            selection_timestamp=timestamp,
            config_used=self.config.to_dict(),
        )

        # Update selection counts
        selected_name = self._get_method_name(selected_method)
        self._selection_counts[selected_name] += 1

        selection_time = time.time() - start_time
        logger.info(
            "Selected method: %s (score: %.4f) in %.2fs",
            selected_name,
            final_scores[best_idx],
            selection_time,
        )

        return selected_method

    def _evaluate_method(
        self,
        method: BaseReconciler,
        base_forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        actuals: pd.DataFrame,
        timestamp: str,
        **kwargs: Any,
    ) -> MethodPerformance:
        """Evaluate a single reconciliation method.

        Args:
            method: Reconciler to evaluate.
            base_forecasts: Base forecasts DataFrame.
            hierarchy: HierarchyDefinition.
            actuals: Actual values for evaluation.
            timestamp: Evaluation timestamp.
            **kwargs: Additional reconciliation arguments.

        Returns:
            MethodPerformance with evaluation metrics.
        """
        method_name = self._get_method_name(method)
        logger.debug("Evaluating method: %s", method_name)

        # Time the reconciliation
        start = time.time()
        result = method.reconcile(base_forecasts, hierarchy, **kwargs)
        computation_time = time.time() - start

        reconciled = result.reconciled_forecasts

        # Calculate metrics
        mape = self._calculate_mape(reconciled, actuals)
        mae = self._calculate_mae(reconciled, actuals)
        rmse = self._calculate_rmse(reconciled, actuals)
        coherence_error = method.check_coherence(reconciled, hierarchy)

        # Calculate component scores
        accuracy_score = self._score_accuracy(mape)
        consistency_score = self._score_consistency(coherence_error)
        efficiency_score = self._score_efficiency(computation_time)

        # Calculate weighted overall score
        overall_score = (
            self.config.accuracy_weight * accuracy_score
            + self.config.consistency_weight * consistency_score
            + self.config.efficiency_weight * efficiency_score
        )

        return MethodPerformance(
            method_name=method_name,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            efficiency_score=efficiency_score,
            overall_score=overall_score,
            mape=mape,
            mae=mae,
            rmse=rmse,
            coherence_error=coherence_error,
            computation_time=computation_time,
            evaluation_timestamp=timestamp,
        )

    def _calculate_mape(self, reconciled: pd.DataFrame, actuals: pd.DataFrame) -> float:
        """Calculate Mean Absolute Percentage Error.

        Args:
            reconciled: Reconciled forecasts.
            actuals: Actual values.

        Returns:
            MAPE value (decimal, not percentage).
        """
        common_cols = list(set(reconciled.columns) & set(actuals.columns))
        if not common_cols:
            return 1.0  # Maximum error

        errors = []
        epsilon = 1e-10

        for col in common_cols:
            y_true = actuals[col].to_numpy()
            y_pred = reconciled[col].to_numpy()

            # Align shapes
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

            # Calculate MAPE for this series
            with np.errstate(divide="ignore", invalid="ignore"):
                ape = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
                errors.append(np.nanmean(ape))

        return float(np.nanmean(errors))

    def _calculate_mae(self, reconciled: pd.DataFrame, actuals: pd.DataFrame) -> float:
        """Calculate Mean Absolute Error.

        Args:
            reconciled: Reconciled forecasts.
            actuals: Actual values.

        Returns:
            MAE value.
        """
        common_cols = list(set(reconciled.columns) & set(actuals.columns))
        if not common_cols:
            return float("inf")

        errors = []
        for col in common_cols:
            y_true = actuals[col].to_numpy()
            y_pred = reconciled[col].to_numpy()

            min_len = min(len(y_true), len(y_pred))
            errors.append(np.nanmean(np.abs(y_true[:min_len] - y_pred[:min_len])))

        return float(np.nanmean(errors))

    def _calculate_rmse(self, reconciled: pd.DataFrame, actuals: pd.DataFrame) -> float:
        """Calculate Root Mean Squared Error.

        Args:
            reconciled: Reconciled forecasts.
            actuals: Actual values.

        Returns:
            RMSE value.
        """
        common_cols = list(set(reconciled.columns) & set(actuals.columns))
        if not common_cols:
            return float("inf")

        errors = []
        for col in common_cols:
            y_true = actuals[col].to_numpy()
            y_pred = reconciled[col].to_numpy()

            min_len = min(len(y_true), len(y_pred))
            mse = np.nanmean((y_true[:min_len] - y_pred[:min_len]) ** 2)
            errors.append(np.sqrt(mse))

        return float(np.nanmean(errors))

    def _score_accuracy(self, mape: float) -> float:
        """Convert MAPE to accuracy score (0-1, higher is better).

        Uses sigmoid-like transformation where MAPE at threshold gives 0.5 score.

        Args:
            mape: Mean Absolute Percentage Error (decimal).

        Returns:
            Accuracy score in [0, 1] range.
        """
        threshold = self.config.mape_threshold

        if mape <= 0:
            return 1.0
        if mape >= 1.0:
            return 0.0

        # Exponential decay from 1 to 0
        # At threshold MAPE, score is ~0.5
        return float(np.exp(-mape / threshold * np.log(2)))

    def _score_consistency(self, coherence_error: float) -> float:
        """Convert coherence error to consistency score (0-1, higher is better).

        Args:
            coherence_error: Sum of absolute coherence violations.

        Returns:
            Consistency score in [0, 1] range.
        """
        tolerance = self.config.coherence_tolerance

        if coherence_error <= tolerance:
            return 1.0

        # Exponential decay for larger errors
        return float(np.exp(-coherence_error / 100.0))

    def _score_efficiency(self, computation_time: float) -> float:
        """Convert computation time to efficiency score (0-1, higher is better).

        Uses sigmoid function where target time gives 0.5 score.

        Args:
            computation_time: Time to reconcile (seconds).

        Returns:
            Efficiency score in [0, 1] range.
        """
        target = self.config.target_computation_time

        if computation_time <= 0:
            return 1.0

        # Sigmoid function: fast methods score high
        return float(1.0 / (1.0 + np.exp((computation_time - target) / target)))

    def _calculate_rolling_scores(self) -> list[float]:
        """Calculate rolling average scores for each method.

        Returns:
            List of rolling average overall scores per method.
        """
        n_methods = len(self.methods)
        rolling_scores = []

        for method_idx in range(n_methods):
            scores = [
                eval_round[method_idx].overall_score
                for eval_round in self.performance_history
            ]
            rolling_scores.append(float(np.mean(scores)))

        return rolling_scores

    def _generate_selection_reason(
        self,
        performances: list[MethodPerformance],
        best_idx: int,
        final_scores: list[float],
    ) -> str:
        """Generate human-readable selection reason.

        Args:
            performances: Performance metrics for all methods.
            best_idx: Index of selected method.
            final_scores: Final scores (possibly with rolling average).

        Returns:
            Selection reason string.
        """
        best_perf = performances[best_idx]
        reason_parts = [
            f"Selected {best_perf.method_name} with overall score {final_scores[best_idx]:.4f}."
        ]

        # Add component scores
        reason_parts.append(
            f"Accuracy: {best_perf.accuracy_score:.3f} (MAPE: {best_perf.mape:.4f})"
        )
        reason_parts.append(
            f"Consistency: {best_perf.consistency_score:.3f} "
            f"(coherence: {best_perf.coherence_error:.2e})"
        )
        reason_parts.append(
            f"Efficiency: {best_perf.efficiency_score:.3f} "
            f"(time: {best_perf.computation_time:.2f}s)"
        )

        # Comparison with other methods
        other_scores = [
            (p.method_name, s)
            for i, (p, s) in enumerate(zip(performances, final_scores, strict=True))
            if i != best_idx
        ]
        if other_scores:
            comparisons = [f"{name}: {score:.4f}" for name, score in other_scores]
            reason_parts.append(f"Other methods: {', '.join(comparisons)}")

        return " ".join(reason_parts)

    def get_last_selection(self) -> SelectionResult | None:
        """Get the last selection result.

        Returns:
            SelectionResult or None if no selection made yet.
        """
        return self._last_selection

    def get_selection_counts(self) -> dict[str, int]:
        """Get count of selections per method.

        Returns:
            Dictionary mapping method names to selection counts.
        """
        return self._selection_counts.copy()

    def get_method_rankings(self) -> list[tuple[str, float]]:
        """Get current method rankings based on rolling history.

        Returns:
            List of (method_name, average_score) tuples sorted by score descending.
        """
        if not self.performance_history:
            return [(name, 0.0) for name in self.method_names]

        scores = self._calculate_rolling_scores()
        rankings = list(zip(self.method_names, scores, strict=True))
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def reset_history(self) -> None:
        """Reset performance history and selection counts."""
        self.performance_history = []
        self._last_selection = None
        self._selection_counts = dict.fromkeys(self.method_names, 0)
        logger.info("Reset selector history")

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive selection report.

        Returns:
            Dictionary with selection statistics and recommendations.
        """
        report: dict[str, Any] = {
            "n_methods": len(self.methods),
            "method_names": self.method_names,
            "config": self.config.to_dict(),
            "n_evaluations": len(self.performance_history),
            "selection_counts": self.get_selection_counts(),
            "rankings": self.get_method_rankings(),
        }

        if self._last_selection:
            report["last_selection"] = self._last_selection.to_dict()

        # Calculate aggregate statistics
        if self.performance_history:
            method_stats = {}
            for method_idx, method_name in enumerate(self.method_names):
                perfs = [
                    eval_round[method_idx] for eval_round in self.performance_history
                ]
                method_stats[method_name] = {
                    "avg_accuracy": float(np.mean([p.accuracy_score for p in perfs])),
                    "avg_consistency": float(np.mean([p.consistency_score for p in perfs])),
                    "avg_efficiency": float(np.mean([p.efficiency_score for p in perfs])),
                    "avg_overall": float(np.mean([p.overall_score for p in perfs])),
                    "avg_mape": float(np.mean([p.mape for p in perfs])),
                    "avg_time": float(np.mean([p.computation_time for p in perfs])),
                }
            report["method_statistics"] = method_stats

        return report

    def save(self, path: str | Path) -> None:
        """Save selector state to disk.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "method_names": self.method_names,
            "config": self.config.to_dict(),
            "selection_counts": self._selection_counts,
            "performance_history": [
                [p.to_dict() for p in eval_round]
                for eval_round in self.performance_history
            ],
        }

        if self._last_selection:
            state["last_selection"] = self._last_selection.to_dict()

        with path.open("w") as f:
            json.dump(state, f, indent=2)

        logger.info("Saved ReconcilerSelector to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        methods: list[BaseReconciler],
    ) -> ReconcilerSelector:
        """Load selector state from disk.

        Args:
            path: Path to saved state file.
            methods: List of reconciler instances (must match saved method names).

        Returns:
            ReconcilerSelector instance with restored state.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If method names don't match.
        """
        path = Path(path)

        if not path.exists():
            msg = f"State file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open() as f:
            state = json.load(f)

        # Validate methods
        saved_names = state["method_names"]
        provided_names = [m.name for m in methods]
        if set(saved_names) != set(provided_names):
            msg = f"Method mismatch: saved={saved_names}, provided={provided_names}"
            raise ValueError(msg)

        # Create instance
        config = SelectorConfig(**state["config"])
        instance = cls(methods=methods, config=config)

        # Restore state
        instance._selection_counts = state["selection_counts"]

        # Restore performance history
        instance.performance_history = [
            [MethodPerformance(**p) for p in eval_round]
            for eval_round in state.get("performance_history", [])
        ]

        # Restore last selection
        if "last_selection" in state:
            sel_data = state["last_selection"]
            instance._last_selection = SelectionResult(
                selected_method_name=sel_data["selected_method_name"],
                selected_method_index=sel_data["selected_method_index"],
                all_performances=[
                    MethodPerformance(**p) for p in sel_data["all_performances"]
                ],
                selection_reason=sel_data["selection_reason"],
                selection_timestamp=sel_data["selection_timestamp"],
                config_used=sel_data.get("config_used", {}),
            )

        logger.info("Loaded ReconcilerSelector from %s", path)

        return instance
