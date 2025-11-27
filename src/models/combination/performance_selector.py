"""Performance-based model selection for optimal ensemble composition.

This module provides intelligent model selection that dynamically optimizes
ensemble composition through multi-objective optimization balancing accuracy,
diversity, and computational cost.

Key Features:
- Multi-objective optimization (accuracy, diversity, computational cost)
- Rolling window evaluation to prevent selection bias
- Ensemble size optimization (2-5 models)
- Multiple selection strategies (greedy, Pareto, adaptive)
- Time series characteristic-aware selection
- Selection history tracking and analysis
- Fallback mechanisms for robust selection

Example:
    ```python
    from src.models.combination.performance_selector import PerformanceBasedSelector

    # Create selector with adaptive strategy
    selector = PerformanceBasedSelector(config={
        'selection_strategy': 'adaptive',
        'max_ensemble_size': 5
    })

    # Fit on historical data
    selector.fit(predictions, targets)

    # Select optimal ensemble
    selected_models = selector.select_optimal_ensemble()

    print(f"Selected models: {selected_models}")
    print(f"Expected improvement: {selector.expected_improvement:.1%}")
    ```
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

# Ruff: Ignore magic numbers in validation thresholds
# ruff: noqa: PLR2004

logger = get_logger(__name__)


@dataclass
class PerformanceSelectorConfig:
    """Configuration for performance-based selector.

    Attributes:
        selection_strategy: Strategy for model selection
            ('greedy', 'pareto', 'adaptive').
        max_ensemble_size: Maximum number of models in ensemble (2-10).
        min_ensemble_size: Minimum number of models in ensemble.
        window_size: Rolling window size for performance tracking.
        diversity_weight: Weight for diversity in multi-objective score (0-1).
        accuracy_weight: Weight for accuracy in multi-objective score (0-1).
        cost_weight: Weight for computational cost in score (0-1).
        marginal_gain_threshold: Minimum performance gain to add model.
        recency_decay: Exponential decay for older performance samples.
        min_diversity_threshold: Minimum ensemble diversity required.
    """

    selection_strategy: str = "adaptive"
    max_ensemble_size: int = 5
    min_ensemble_size: int = 2
    window_size: int = 48
    diversity_weight: float = 0.3
    accuracy_weight: float = 0.6
    cost_weight: float = 0.1
    marginal_gain_threshold: float = 0.01
    recency_decay: float = 0.95
    min_diversity_threshold: float = 0.1


@dataclass
class SelectionRecord:
    """Record of ensemble selection decision.

    Attributes:
        timestamp: When selection was made.
        selected_models: List of selected model names.
        selection_strategy: Strategy used for selection.
        composite_scores: Multi-objective scores for selected models.
        expected_performance: Expected ensemble MAPE.
        ensemble_size: Number of models selected.
        reason: Description of selection decision.
        diversity_score: Overall diversity of selected ensemble.
    """

    timestamp: datetime
    selected_models: list[str]
    selection_strategy: str
    composite_scores: dict[str, float]
    expected_performance: float
    ensemble_size: int
    reason: str
    diversity_score: float = 0.0


class RollingPerformanceTracker:
    """Track rolling performance metrics for models.

    Maintains a rolling window of performance metrics (accuracy, diversity, cost)
    for each model, enabling temporal analysis of model performance.

    Attributes:
        window_size: Size of rolling window for metrics.

    Example:
        >>> tracker = RollingPerformanceTracker(window_size=48)
        >>> tracker.update("lgbm", accuracy=2.5, diversity=0.8, cost=1.0)
        >>> perf = tracker.get_recent_performance("lgbm")
        >>> print(perf['accuracy'])
    """

    def __init__(self, window_size: int = 48) -> None:
        """Initialize performance tracker.

        Args:
            window_size: Size of rolling window for performance history.
        """
        self.window_size = window_size
        self._accuracy_history: dict[str, deque[float]] = {}
        self._diversity_history: dict[str, deque[float]] = {}
        self._cost_history: dict[str, deque[float]] = {}

    def update(
        self,
        model_name: str,
        accuracy: float,
        diversity: float = 0.5,
        cost: float = 1.0,
    ) -> None:
        """Update performance metrics for a model.

        Args:
            model_name: Name of the model.
            accuracy: Accuracy metric (MAPE, lower is better).
            diversity: Diversity contribution score (0-1, higher is better).
            cost: Computational cost metric (relative, lower is better).
        """
        if model_name not in self._accuracy_history:
            self._accuracy_history[model_name] = deque(maxlen=self.window_size)
            self._diversity_history[model_name] = deque(maxlen=self.window_size)
            self._cost_history[model_name] = deque(maxlen=self.window_size)

        self._accuracy_history[model_name].append(accuracy)
        self._diversity_history[model_name].append(diversity)
        self._cost_history[model_name].append(cost)

    def get_recent_performance(self, model_name: str) -> dict[str, float]:
        """Get recent average performance for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary with accuracy, diversity, and cost averages.
        """
        if model_name not in self._accuracy_history or len(self._accuracy_history[model_name]) == 0:
            return {"accuracy": float("inf"), "diversity": 0.0, "cost": 1.0}

        return {
            "accuracy": float(np.mean(list(self._accuracy_history[model_name]))),
            "diversity": float(np.mean(list(self._diversity_history[model_name]))),
            "cost": float(np.mean(list(self._cost_history[model_name]))),
        }

    def get_performance_trend(self, model_name: str) -> dict[str, float]:
        """Get performance trend (recent vs older).

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary with trend indicators (-1 to 1, positive = improving).
        """
        if model_name not in self._accuracy_history:
            return {"accuracy_trend": 0.0, "diversity_trend": 0.0}

        history = list(self._accuracy_history[model_name])
        if len(history) < 4:
            return {"accuracy_trend": 0.0, "diversity_trend": 0.0}

        mid = len(history) // 2
        recent_acc = np.mean(history[mid:])
        older_acc = np.mean(history[:mid])

        div_history = list(self._diversity_history[model_name])
        recent_div = np.mean(div_history[mid:])
        older_div = np.mean(div_history[:mid])

        # For accuracy, lower is better, so negative trend is improvement
        acc_trend = (older_acc - recent_acc) / (older_acc + 1e-8)
        # For diversity, higher is better
        div_trend = (recent_div - older_div) / (older_div + 1e-8)

        return {
            "accuracy_trend": float(np.clip(acc_trend, -1, 1)),
            "diversity_trend": float(np.clip(div_trend, -1, 1)),
        }

    def get_model_names(self) -> list[str]:
        """Get list of tracked model names.

        Returns:
            List of model names with performance history.
        """
        return list(self._accuracy_history.keys())

    def reset(self, model_name: str | None = None) -> None:
        """Reset performance history.

        Args:
            model_name: If provided, reset only this model. Otherwise reset all.
        """
        if model_name is not None:
            if model_name in self._accuracy_history:
                self._accuracy_history[model_name].clear()
                self._diversity_history[model_name].clear()
                self._cost_history[model_name].clear()
        else:
            self._accuracy_history.clear()
            self._diversity_history.clear()
            self._cost_history.clear()


@dataclass
class SelectionDiagnostics:
    """Diagnostics from model selection process.

    Attributes:
        n_models_available: Number of models in pool.
        n_models_selected: Number of models selected.
        selection_strategy: Strategy used.
        individual_scores: Multi-objective scores per model.
        ensemble_performance: Expected ensemble performance.
        ensemble_diversity: Overall ensemble diversity.
        selection_time_ms: Time taken for selection.
        pareto_candidates: Number of Pareto-optimal candidates (if applicable).
        marginal_gains: Marginal gains when adding each model (greedy).
    """

    n_models_available: int = 0
    n_models_selected: int = 0
    selection_strategy: str = ""
    individual_scores: dict[str, float] = field(default_factory=dict)
    ensemble_performance: float = 0.0
    ensemble_diversity: float = 0.0
    selection_time_ms: float = 0.0
    pareto_candidates: int = 0
    marginal_gains: list[float] = field(default_factory=list)


class PerformanceBasedSelector:
    """Performance-based model selection for optimal ensemble composition.

    Dynamically selects optimal ensemble members by balancing multiple objectives:
    accuracy, diversity, and computational cost. Supports multiple selection
    strategies and includes rolling window evaluation to prevent selection bias.

    Multi-Objective Scoring:
        score = w_acc * (1/accuracy) + w_div * diversity - w_cost * cost

    Selection Strategies:
        - Greedy: Iteratively add models with highest marginal gain
        - Pareto: Find Pareto-optimal ensemble combinations
        - Adaptive: Choose strategy based on model pool size

    Ensemble Size Optimization:
        - Start with best performing model
        - Add models while marginal_gain > threshold
        - Limit ensemble size to [min_size, max_size]

    Attributes:
        config: Configuration parameters.
        performance_tracker: Rolling performance tracker.
        model_pool: List of available model names.
        selection_history: History of selection decisions.
        expected_improvement: Expected improvement from selection.
        fitted: Whether selector has been fitted.

    Example:
        >>> selector = PerformanceBasedSelector(
        ...     config={'selection_strategy': 'adaptive'}
        ... )
        >>> selector.fit(predictions, targets)
        >>> selected = selector.select_optimal_ensemble()
        >>> print(f"Selected: {selected}")
        >>> print(f"Expected improvement: {selector.expected_improvement:.1%}")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize performance-based selector.

        Args:
            config: Configuration dictionary. See PerformanceSelectorConfig
                for available options.
        """
        if config is None:
            config = {}

        self.config = PerformanceSelectorConfig(**config)
        self.performance_tracker = RollingPerformanceTracker(self.config.window_size)

        self.model_pool: list[str] = []
        self.selection_history: list[SelectionRecord] = []
        self.expected_improvement: float | None = None
        self._fitted = False
        self._last_diagnostics: SelectionDiagnostics | None = None

        # Cached correlation matrix for diversity calculations
        self._correlation_matrix: dict[tuple[str, str], float] = {}

        logger.debug(
            "Initialized PerformanceBasedSelector with strategy=%s",
            self.config.selection_strategy,
        )

    @property
    def fitted(self) -> bool:
        """Return whether selector has been fitted."""
        return self._fitted

    def fit(
        self,
        predictions: dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        computational_costs: dict[str, float] | None = None,
    ) -> None:
        """Fit selector using historical predictions and targets.

        Calculates accuracy and diversity metrics for each model based on
        historical performance, storing them in the rolling tracker.

        Args:
            predictions: Dictionary mapping model names to prediction DataFrames.
                Each DataFrame should have columns like 'pred_h0', 'pred_h1', etc.
            targets: Target values DataFrame with columns 'target_h0', 'target_h1', etc.
                or 'h0', 'h1', etc.
            computational_costs: Optional dictionary of relative costs per model.

        Raises:
            ValueError: If predictions dictionary is empty.
        """
        if not predictions:
            msg = "Predictions dictionary cannot be empty"
            raise ValueError(msg)

        logger.info("Fitting performance-based selector on %d models", len(predictions))

        self.model_pool = sorted(predictions.keys())

        # Default costs
        if computational_costs is None:
            computational_costs = dict.fromkeys(self.model_pool, 1.0)

        # Get prediction columns
        first_pred = next(iter(predictions.values()))
        pred_cols = sorted([col for col in first_pred.columns if col.startswith("pred_h")])

        if not pred_cols:
            msg = "No prediction columns found (expected pred_h0, pred_h1, etc.)"
            raise ValueError(msg)

        # Calculate accuracy for each model
        model_accuracies = self._calculate_model_accuracies(predictions, targets, pred_cols)

        # Calculate pairwise correlations for diversity
        self._correlation_matrix = self._calculate_correlation_matrix(predictions, pred_cols)

        # Calculate diversity contribution for each model
        model_diversities = self._calculate_diversity_contributions()

        # Update performance tracker
        for model_name in self.model_pool:
            self.performance_tracker.update(
                model_name=model_name,
                accuracy=model_accuracies.get(model_name, float("inf")),
                diversity=model_diversities.get(model_name, 0.5),
                cost=computational_costs.get(model_name, 1.0),
            )

        self._fitted = True
        logger.info("Performance selector fitted successfully")

    def _calculate_model_accuracies(
        self,
        predictions: dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        pred_cols: list[str],
    ) -> dict[str, float]:
        """Calculate MAPE accuracy for each model.

        Args:
            predictions: Model predictions.
            targets: Target values.
            pred_cols: List of prediction columns.

        Returns:
            Dictionary mapping model names to MAPE values.
        """
        model_accuracies = {}

        for model_name in self.model_pool:
            errors = []

            for col in pred_cols:
                horizon = col.split("_h")[1]

                # Try different target column naming conventions
                target_col = None
                for prefix in ["target_h", "h"]:
                    candidate = f"{prefix}{horizon}"
                    if candidate in targets.columns:
                        target_col = candidate
                        break

                if target_col is None:
                    continue

                pred_values = predictions[model_name][col].to_numpy()
                true_values = targets[target_col].to_numpy()

                # Calculate MAPE
                valid_mask = ~(np.isnan(pred_values) | np.isnan(true_values) | (true_values == 0))

                if valid_mask.sum() > 0:
                    mape = (
                        np.abs((true_values[valid_mask] - pred_values[valid_mask]) / true_values[valid_mask]).mean()
                        * 100
                    )
                    errors.append(mape)

            model_accuracies[model_name] = np.mean(errors) if errors else float("inf")

        return model_accuracies

    def _calculate_correlation_matrix(
        self,
        predictions: dict[str, pd.DataFrame],
        pred_cols: list[str],
    ) -> dict[tuple[str, str], float]:
        """Calculate pairwise correlation matrix between models.

        Args:
            predictions: Model predictions.
            pred_cols: List of prediction columns.

        Returns:
            Dictionary mapping model pairs to correlation values.
        """
        correlations: dict[tuple[str, str], float] = {}

        for m1, m2 in combinations(self.model_pool, 2):
            # Calculate average correlation across horizons
            corrs = []
            for col in pred_cols:
                p1 = predictions[m1][col].to_numpy()
                p2 = predictions[m2][col].to_numpy()

                valid_mask = ~(np.isnan(p1) | np.isnan(p2))
                if valid_mask.sum() > 10:
                    corr = np.corrcoef(p1[valid_mask], p2[valid_mask])[0, 1]
                    if not np.isnan(corr):
                        corrs.append(corr)

            avg_corr = np.mean(corrs) if corrs else 0.0
            correlations[(m1, m2)] = avg_corr
            correlations[(m2, m1)] = avg_corr

        # Self-correlation is 1
        for m in self.model_pool:
            correlations[(m, m)] = 1.0

        return correlations

    def _calculate_diversity_contributions(self) -> dict[str, float]:
        """Calculate diversity contribution for each model.

        Diversity is based on average dissimilarity (1 - correlation)
        with other models.

        Returns:
            Dictionary mapping model names to diversity scores (0-1).
        """
        diversities = {}

        for model in self.model_pool:
            dissimilarities = []
            for other in self.model_pool:
                if other != model:
                    corr = self._correlation_matrix.get((model, other), 0.0)
                    dissimilarities.append(1.0 - abs(corr))

            diversities[model] = np.mean(dissimilarities) if dissimilarities else 0.5

        return diversities

    def select_optimal_ensemble(self) -> list[str]:
        """Select optimal ensemble composition.

        Uses configured selection strategy to choose models that
        maximize the multi-objective score balancing accuracy,
        diversity, and computational cost.

        Returns:
            List of selected model names.

        Raises:
            RuntimeError: If selector has not been fitted.
        """
        if not self._fitted:
            msg = "Selector has not been fitted. Call fit() first."
            raise RuntimeError(msg)

        start_time = time.time()

        logger.info(
            "Selecting optimal ensemble (%s strategy) from %d models",
            self.config.selection_strategy,
            len(self.model_pool),
        )

        # Initialize diagnostics
        self._last_diagnostics = SelectionDiagnostics(
            n_models_available=len(self.model_pool),
            selection_strategy=self.config.selection_strategy,
        )

        # Handle small model pools
        if len(self.model_pool) <= self.config.min_ensemble_size:
            logger.info("Insufficient models, using all available")
            selected = self.model_pool.copy()
        else:
            # Choose selection strategy
            if self.config.selection_strategy == "greedy":
                selected = self._greedy_select()
            elif self.config.selection_strategy == "pareto":
                selected = self._pareto_select()
            elif self.config.selection_strategy == "adaptive":
                selected = self._adaptive_select()
            else:
                msg = f"Unknown selection strategy: {self.config.selection_strategy}"
                raise ValueError(msg)

            # Validate selection
            if len(selected) < self.config.min_ensemble_size:
                selected = self._fallback_selection()

        # Update diagnostics
        self._last_diagnostics.n_models_selected = len(selected)
        self._last_diagnostics.selection_time_ms = (time.time() - start_time) * 1000
        self._last_diagnostics.individual_scores = self._calculate_multi_objective_scores()
        self._last_diagnostics.ensemble_performance = self._estimate_ensemble_performance(selected)
        self._last_diagnostics.ensemble_diversity = self._calculate_ensemble_diversity(selected)

        # Record selection
        self._record_selection(selected)

        logger.info("Selected %d models: %s", len(selected), selected)

        return selected

    def _greedy_select(self) -> list[str]:
        """Greedy selection: iteratively add models with highest marginal gain.

        Returns:
            List of selected model names.
        """
        # Rank models by multi-objective score
        scores = self._calculate_multi_objective_scores()
        ranked_models = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)

        # Start with best model
        selected = [ranked_models[0]]
        base_performance = self._estimate_ensemble_performance(selected)

        marginal_gains = []

        # Add models while improving performance
        for candidate in ranked_models[1:]:
            if len(selected) >= self.config.max_ensemble_size:
                break

            # Check diversity constraint
            avg_corr = self._calculate_avg_correlation(selected, candidate)
            if avg_corr > 0.95:  # Skip highly correlated models
                continue

            # Test adding this model
            test_ensemble = [*selected, candidate]
            test_performance = self._estimate_ensemble_performance(test_ensemble)

            # Calculate marginal gain (performance improvement)
            if base_performance > 0:
                marginal_gain = (base_performance - test_performance) / base_performance
            else:
                marginal_gain = 0.0

            marginal_gains.append(marginal_gain)

            if marginal_gain > self.config.marginal_gain_threshold:
                selected.append(candidate)
                base_performance = test_performance

        if self._last_diagnostics:
            self._last_diagnostics.marginal_gains = marginal_gains

        return selected

    def _pareto_select(self) -> list[str]:
        """Pareto optimization: find Pareto-optimal ensemble.

        Evaluates all possible combinations and selects the one
        with the best composite score from the Pareto frontier.

        Returns:
            List of selected model names.
        """
        best_ensemble: list[str] | None = None
        best_score = float("-inf")
        pareto_candidates = 0

        for size in range(
            self.config.min_ensemble_size,
            min(self.config.max_ensemble_size, len(self.model_pool)) + 1,
        ):
            for combo in combinations(self.model_pool, size):
                combo_list = list(combo)

                # Calculate composite score
                accuracy = self._estimate_ensemble_performance(combo_list)
                diversity = self._calculate_ensemble_diversity(combo_list)
                cost = len(combo_list)

                # Multi-objective score (higher is better)
                # Accuracy is MAPE so invert it
                score = (
                    self.config.accuracy_weight * (1.0 / (accuracy + 1.0))
                    + self.config.diversity_weight * diversity
                    - self.config.cost_weight * cost / self.config.max_ensemble_size
                )

                pareto_candidates += 1

                if score > best_score:
                    best_score = score
                    best_ensemble = combo_list

        if self._last_diagnostics:
            self._last_diagnostics.pareto_candidates = pareto_candidates

        return best_ensemble if best_ensemble else self.model_pool[: self.config.min_ensemble_size]

    def _adaptive_select(self) -> list[str]:
        """Adaptive selection: choose strategy based on characteristics.

        Uses greedy for large model pools (>6 models) for efficiency,
        and Pareto for smaller pools where exhaustive search is feasible.

        Returns:
            List of selected model names.
        """
        # Use greedy for large pools, Pareto for small
        if len(self.model_pool) > 6:
            return self._greedy_select()
        return self._pareto_select()

    def _calculate_multi_objective_scores(self) -> dict[str, float]:
        """Calculate multi-objective composite score for each model.

        Returns:
            Dictionary mapping model names to composite scores.
        """
        scores = {}

        for model in self.model_pool:
            perf = self.performance_tracker.get_recent_performance(model)

            # Normalize and combine components
            # Accuracy: lower MAPE is better, so invert
            accuracy_score = 1.0 / (perf["accuracy"] + 1.0)
            # Diversity: higher is better
            diversity_score = perf["diversity"]
            # Cost: lower is better, so invert
            cost_score = 1.0 / (perf["cost"] + 0.1)

            composite = (
                self.config.accuracy_weight * accuracy_score
                + self.config.diversity_weight * diversity_score
                + self.config.cost_weight * cost_score
            )

            scores[model] = composite

        return scores

    def _estimate_ensemble_performance(self, models: list[str]) -> float:
        """Estimate ensemble MAPE performance.

        Uses weighted average of individual performances with
        diversity benefit adjustment.

        Args:
            models: List of model names in ensemble.

        Returns:
            Estimated ensemble MAPE (lower is better).
        """
        if not models:
            return float("inf")

        performances = [self.performance_tracker.get_recent_performance(m)["accuracy"] for m in models]

        # Calculate diversity benefit
        diversity = self._calculate_ensemble_diversity(models)
        # More diverse ensembles get bonus (up to 10% improvement)
        diversity_benefit = 0.9 + 0.1 * min(diversity, 1.0)

        return float(np.mean(performances) * diversity_benefit)

    def _calculate_ensemble_diversity(self, models: list[str]) -> float:
        """Calculate overall diversity score for ensemble.

        Based on average pairwise dissimilarity (1 - correlation).

        Args:
            models: List of model names.

        Returns:
            Diversity score (0-1, higher is more diverse).
        """
        if len(models) < 2:
            return 0.0

        dissimilarities = []
        for m1, m2 in combinations(models, 2):
            corr = self._correlation_matrix.get((m1, m2), 0.0)
            dissimilarities.append(1.0 - abs(corr))

        return float(np.mean(dissimilarities)) if dissimilarities else 0.0

    def _calculate_avg_correlation(self, selected: list[str], candidate: str) -> float:
        """Calculate average correlation between candidate and selected models.

        Args:
            selected: Currently selected models.
            candidate: Candidate model to add.

        Returns:
            Average absolute correlation.
        """
        if not selected:
            return 0.0

        correlations = []
        for model in selected:
            corr = self._correlation_matrix.get((model, candidate), 0.0)
            correlations.append(abs(corr))

        return float(np.mean(correlations))

    def _fallback_selection(self) -> list[str]:
        """Fallback selection when primary strategy fails.

        Returns top models by accuracy.

        Returns:
            List of selected model names.
        """
        logger.warning("Using fallback selection")

        scores = self._calculate_multi_objective_scores()
        ranked = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)

        return ranked[: self.config.min_ensemble_size]

    def _record_selection(self, selected: list[str]) -> None:
        """Record selection decision in history.

        Args:
            selected: List of selected model names.
        """
        scores = self._calculate_multi_objective_scores()

        record = SelectionRecord(
            timestamp=datetime.now(tz=UTC),
            selected_models=selected.copy(),
            selection_strategy=self.config.selection_strategy,
            composite_scores={m: scores[m] for m in selected},
            expected_performance=self._estimate_ensemble_performance(selected),
            ensemble_size=len(selected),
            reason=f"Selected via {self.config.selection_strategy} optimization",
            diversity_score=self._calculate_ensemble_diversity(selected),
        )

        self.selection_history.append(record)

        # Calculate expected improvement over average
        all_performances = [
            self.performance_tracker.get_recent_performance(m)["accuracy"] for m in self.model_pool
        ]
        avg_performance = np.mean(all_performances)

        if avg_performance > 0:
            self.expected_improvement = (avg_performance - record.expected_performance) / avg_performance
        else:
            self.expected_improvement = 0.0

    def get_selection_diagnostics(self) -> dict[str, Any]:
        """Get diagnostics from most recent selection.

        Returns:
            Dictionary with diagnostic information.
        """
        if self._last_diagnostics is None:
            return {}

        return {
            "n_models_available": self._last_diagnostics.n_models_available,
            "n_models_selected": self._last_diagnostics.n_models_selected,
            "selection_strategy": self._last_diagnostics.selection_strategy,
            "individual_scores": self._last_diagnostics.individual_scores.copy(),
            "ensemble_performance": self._last_diagnostics.ensemble_performance,
            "ensemble_diversity": self._last_diagnostics.ensemble_diversity,
            "selection_time_ms": self._last_diagnostics.selection_time_ms,
            "pareto_candidates": self._last_diagnostics.pareto_candidates,
            "marginal_gains": self._last_diagnostics.marginal_gains.copy(),
        }

    def get_selection_history_summary(self) -> dict[str, Any]:
        """Get summary of selection history.

        Returns:
            Dictionary with history summary statistics.
        """
        if not self.selection_history:
            return {"n_selections": 0}

        sizes = [r.ensemble_size for r in self.selection_history]
        performances = [r.expected_performance for r in self.selection_history]
        diversities = [r.diversity_score for r in self.selection_history]

        # Calculate model selection frequency
        model_counts: dict[str, int] = {}
        for record in self.selection_history:
            for model in record.selected_models:
                model_counts[model] = model_counts.get(model, 0) + 1

        return {
            "n_selections": len(self.selection_history),
            "avg_ensemble_size": float(np.mean(sizes)),
            "avg_performance": float(np.mean(performances)),
            "avg_diversity": float(np.mean(diversities)),
            "model_selection_frequency": model_counts,
            "most_selected": max(model_counts, key=model_counts.get) if model_counts else None,
        }

    def update_performance(
        self,
        model_name: str,
        accuracy: float,
        diversity: float | None = None,
        cost: float | None = None,
    ) -> None:
        """Update performance metrics for a single model.

        Call this to update the rolling performance tracker with
        new observations during inference.

        Args:
            model_name: Name of the model.
            accuracy: New accuracy observation (MAPE).
            diversity: Optional new diversity score.
            cost: Optional new cost value.
        """
        current = self.performance_tracker.get_recent_performance(model_name)

        self.performance_tracker.update(
            model_name=model_name,
            accuracy=accuracy,
            diversity=diversity if diversity is not None else current.get("diversity", 0.5),
            cost=cost if cost is not None else current.get("cost", 1.0),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"PerformanceBasedSelector("
            f"strategy={self.config.selection_strategy}, "
            f"models={len(self.model_pool)}, "
            f"status={status})"
        )
