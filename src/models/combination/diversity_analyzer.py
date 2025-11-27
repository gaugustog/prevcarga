"""Ensemble diversity analyzer for model combination.

This module provides tools to measure diversity and complementarity among ensemble
members. Understanding model diversity helps select effective model combinations and
diagnose ensemble performance.

Key Features:
    - Correlation-based diversity metrics (Pearson, Spearman)
    - Error-based diversity metrics (Q-statistic, disagreement, double fault, ambiguity)
    - Pairwise and aggregate diversity analysis
    - Redundant model identification
    - Optimal ensemble size recommendation
    - Model selection strategies (greedy forward/backward, correlation clustering)
    - Visualization support (correlation matrix, dendrogram data)
    - Diversity report generation

Example:
    ```python
    from src.models.combination import DiversityAnalyzer, DiversityConfig
    import numpy as np

    # Model predictions
    predictions = {
        "lgbm": np.array([100, 200, 300, 400]),
        "rf": np.array([110, 190, 310, 395]),
        "arima": np.array([95, 205, 295, 405]),
    }
    actuals = np.array([102, 198, 302, 401])

    # Configure analyzer
    config = DiversityConfig(
        metrics=["correlation", "disagreement", "ambiguity"],
        correlation_threshold=0.95,
        max_models=3
    )

    # Analyze diversity
    analyzer = DiversityAnalyzer(config=config)
    result = analyzer.analyze(predictions, actuals)

    print(f"Average correlation: {result.avg_correlation:.3f}")
    print(f"Diversity score: {result.diversity_scores['overall']:.3f}")

    # Select optimal subset
    subset = analyzer.select_subset(predictions, actuals, method="greedy_forward")
    print(f"Recommended ensemble: {subset}")
    ```
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiversityConfig(BaseModel):
    """Configuration for diversity analyzer.

    Provides standardized configuration options for diversity analysis,
    including which metrics to compute, thresholds for redundancy detection,
    and parameters for model selection.

    Attributes:
        metrics: List of diversity metrics to compute. Options:
            - "correlation": Pearson correlation between predictions
            - "spearman": Spearman rank correlation
            - "disagreement": Proportion of disagreeing predictions
            - "double_fault": Both models wrong simultaneously
            - "q_statistic": Q-statistic (adapted for regression)
            - "ambiguity": Ambiguity decomposition (ensemble variance)
        correlation_threshold: Correlation above which models are considered redundant (0-1).
            For Brazilian load forecasting, 0.85 is typical.
        min_models: Minimum number of models in selected ensemble subset.
        max_models: Maximum number of models in selected ensemble subset.
        selection_method: Method for selecting optimal model subset:
            - "greedy_forward": Start with best pair, add most diverse iteratively
            - "greedy_backward": Start with all, remove most redundant iteratively
            - "correlation_clustering": Cluster by correlation, pick representative from each
        error_tolerance: Tolerance for classifying predictions as correct/incorrect
            in error-based metrics (relative error).
        clustering_distance_metric: Distance metric for hierarchical clustering
            ("euclidean", "correlation", "manhattan").

    Example:
        >>> config = DiversityConfig(
        ...     metrics=["correlation", "disagreement", "ambiguity"],
        ...     correlation_threshold=0.85,
        ...     min_models=2,
        ...     max_models=5,
        ...     selection_method="greedy_forward"
        ... )
        >>> analyzer = DiversityAnalyzer(config=config)
    """

    metrics: list[str] = Field(
        default=["correlation", "disagreement"],
        description="List of diversity metrics to compute",
    )
    correlation_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Threshold for redundant model detection",
    )
    min_models: int = Field(
        default=2,
        ge=2,
        description="Minimum ensemble size",
    )
    max_models: int = Field(
        default=10,
        ge=2,
        description="Maximum ensemble size for subset selection",
    )
    selection_method: Literal["greedy_forward", "greedy_backward", "correlation_clustering"] = Field(
        default="greedy_forward",
        description="Model selection strategy",
    )
    error_tolerance: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Relative error tolerance for error-based metrics",
    )
    clustering_distance_metric: Literal["euclidean", "correlation", "manhattan"] = Field(
        default="correlation",
        description="Distance metric for hierarchical clustering",
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Validate metric names."""
        valid_metrics = {
            "correlation",
            "spearman",
            "disagreement",
            "double_fault",
            "q_statistic",
            "ambiguity",
        }
        invalid = set(v) - valid_metrics
        if invalid:
            msg = f"Invalid metrics: {invalid}. Valid options: {valid_metrics}"
            raise ValueError(msg)
        return v

    @field_validator("max_models")
    @classmethod
    def validate_max_models(cls, v: int, info: Any) -> int:
        """Validate max_models >= min_models."""
        if "min_models" in info.data and v < info.data["min_models"]:
            msg = f"max_models ({v}) must be >= min_models ({info.data['min_models']})"
            raise ValueError(msg)
        return v


@dataclass
class DiversityResult:
    """Result of ensemble diversity analysis.

    Contains computed diversity metrics, pairwise relationships, model rankings,
    and metadata about the analysis process.

    Attributes:
        pairwise_correlations: Dictionary mapping model pairs to Pearson correlation.
            Keys are (model1, model2) tuples.
        avg_correlation: Average pairwise correlation across all model pairs.
        correlation_matrix: n_models × n_models DataFrame with all correlations.
        avg_correlation_per_model: Dictionary mapping each model to its average
            correlation with all other models.
        diversity_scores: Dictionary of diversity scores per metric type.
            Keys: "overall", "correlation", "error_based", etc.
        model_rankings: List of (model_name, diversity_contribution) tuples,
            sorted by diversity contribution (higher = more diverse).
        recommended_subset: Optional list of recommended model names for ensembling.
        pairwise_spearman: Optional dictionary of Spearman correlations.
        q_statistics: Optional dictionary of Q-statistics (error-based).
            Requires actual targets for computation.
        disagreement: Optional dictionary of disagreement measures (error-based).
            Requires actual targets for computation.
        double_fault: Optional dictionary of double fault measures (both models wrong).
            Requires actual targets for computation.
        ambiguity: Optional dictionary of ambiguity scores (ensemble variance component).
            Requires actual targets for computation.
        aggregate_diversity: Single diversity score for ensemble (1 - avg_correlation).
        metadata: Dictionary with computation details (time, n_samples, etc.).
        model_names: List of model names in the analysis.

    Example:
        >>> result = DiversityResult(
        ...     pairwise_correlations={("lgbm", "rf"): 0.85},
        ...     avg_correlation=0.85,
        ...     correlation_matrix=corr_matrix_df,
        ...     avg_correlation_per_model={"lgbm": 0.82, "rf": 0.88},
        ...     diversity_scores={"overall": 0.15, "correlation": 0.15},
        ...     model_rankings=[("lgbm", 0.18), ("rf", 0.12)],
        ...     aggregate_diversity=0.15,
        ...     model_names=["lgbm", "rf"]
        ... )
        >>> model1, model2, score = result.get_most_diverse_pair()
    """

    pairwise_correlations: dict[tuple[str, str], float]
    avg_correlation: float
    correlation_matrix: pd.DataFrame
    avg_correlation_per_model: dict[str, float]
    diversity_scores: dict[str, float]
    model_rankings: list[tuple[str, float]]
    pairwise_spearman: dict[tuple[str, str], float] | None = None
    q_statistics: dict[tuple[str, str], float] | None = None
    disagreement: dict[tuple[str, str], float] | None = None
    double_fault: dict[tuple[str, str], float] | None = None
    ambiguity: dict[tuple[str, str], float] | None = None
    aggregate_diversity: float = 0.0
    recommended_subset: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    model_names: list[str] = field(default_factory=list)

    def get_most_diverse_pair(self) -> tuple[str, str, float]:
        """Return model pair with lowest correlation (highest diversity).

        Returns:
            Tuple of (model1, model2, correlation_score) where correlation_score
            is the lowest among all pairs.

        Raises:
            ValueError: If no pairwise correlations available.

        Example:
            >>> model1, model2, score = result.get_most_diverse_pair()
            >>> print(f"Most diverse: {model1} & {model2} (corr: {score:.3f})")
        """
        if not self.pairwise_correlations:
            msg = "No pairwise correlations available"
            raise ValueError(msg)

        # Find pair with minimum correlation (maximum diversity)
        min_pair = min(self.pairwise_correlations.items(), key=lambda x: x[1])
        model1, model2 = min_pair[0]
        correlation = min_pair[1]

        return model1, model2, correlation

    def get_least_diverse_pair(self) -> tuple[str, str, float]:
        """Return model pair with highest correlation (lowest diversity).

        Returns:
            Tuple of (model1, model2, correlation_score) where correlation_score
            is the highest among all pairs.

        Raises:
            ValueError: If no pairwise correlations available.

        Example:
            >>> model1, model2, score = result.get_least_diverse_pair()
            >>> print(f"Least diverse: {model1} & {model2} (corr: {score:.3f})")
        """
        if not self.pairwise_correlations:
            msg = "No pairwise correlations available"
            raise ValueError(msg)

        # Find pair with maximum correlation (minimum diversity)
        max_pair = max(self.pairwise_correlations.items(), key=lambda x: x[1])
        model1, model2 = max_pair[0]
        correlation = max_pair[1]

        return model1, model2, correlation

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization.

        Returns:
            Dictionary representation of the diversity result.
        """
        return {
            "pairwise_correlations": {
                f"{k[0]}__{k[1]}": v for k, v in self.pairwise_correlations.items()
            },
            "avg_correlation": self.avg_correlation,
            "avg_correlation_per_model": self.avg_correlation_per_model.copy(),
            "diversity_scores": self.diversity_scores.copy(),
            "model_rankings": self.model_rankings.copy(),
            "aggregate_diversity": self.aggregate_diversity,
            "model_names": self.model_names.copy(),
            "metadata": self.metadata.copy(),
            "has_spearman": self.pairwise_spearman is not None,
            "has_q_statistics": self.q_statistics is not None,
            "has_disagreement": self.disagreement is not None,
            "has_double_fault": self.double_fault is not None,
            "has_ambiguity": self.ambiguity is not None,
            "recommended_subset": self.recommended_subset.copy() if self.recommended_subset else None,
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the diversity result.
        """
        return (
            f"DiversityResult("
            f"n_models={len(self.model_names)}, "
            f"avg_correlation={self.avg_correlation:.3f}, "
            f"diversity={self.aggregate_diversity:.3f})"
        )


class DiversityAnalyzer:
    """Analyzer for measuring ensemble diversity and complementarity.

    This is the main diversity analysis class (formerly EnsembleDiversityAnalyzer).
    Provides comprehensive diversity metrics, model selection, and visualization support.

    Key Features:
        - Correlation-based metrics (Pearson, Spearman)
        - Error-based metrics (Q-statistic, disagreement, double fault, ambiguity)
        - Redundant model detection
        - Optimal model subset selection
        - Visualization data generation
        - Diversity reporting

    Attributes:
        config: DiversityConfig instance with analysis configuration.

    Example:
        >>> config = DiversityConfig(metrics=["correlation", "ambiguity"])
        >>> analyzer = DiversityAnalyzer(config=config)
        >>> predictions = {
        ...     "model1": np.array([1, 2, 3]),
        ...     "model2": np.array([1.1, 2.1, 2.9])
        ... }
        >>> result = analyzer.analyze(predictions, np.array([1, 2, 3]))
        >>> print(f"Diversity: {result.aggregate_diversity:.3f}")
    """

    def __init__(self, config: DiversityConfig | None = None) -> None:
        """Initialize diversity analyzer.

        Args:
            config: Configuration for diversity analysis. If None, uses defaults.
        """
        self.config = config or DiversityConfig()
        logger.debug(
            "Initialized DiversityAnalyzer with config: metrics=%s, threshold=%.2f",
            self.config.metrics,
            self.config.correlation_threshold,
        )

    def analyze(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray | None = None,
    ) -> DiversityResult:
        """Analyze diversity of ensemble members.

        Computes comprehensive diversity metrics based on configuration.

        Args:
            predictions: Dictionary mapping model names to prediction arrays.
                All arrays must have the same length.
            actuals: Optional ground truth for error-based metrics. If None,
                only correlation-based metrics are computed.

        Returns:
            DiversityResult with all computed metrics.

        Raises:
            ValueError: If predictions are invalid or have inconsistent lengths.

        Example:
            >>> predictions = {
            ...     "lgbm": np.array([100, 200, 300]),
            ...     "rf": np.array([110, 190, 310])
            ... }
            >>> actuals = np.array([102, 198, 302])
            >>> result = analyzer.analyze(predictions, actuals)
            >>> print(f"Average correlation: {result.avg_correlation:.3f}")
        """
        start_time = time.time()

        # Validate inputs
        self._validate_predictions(predictions)
        if actuals is not None:
            self._validate_actuals(predictions, actuals)

        model_names = sorted(predictions.keys())
        n_samples = len(predictions[model_names[0]])

        # 1. Compute correlation matrix
        pairwise_correlations, correlation_matrix = self._compute_correlations(predictions)

        # 2. Compute average correlation per model
        avg_correlation_per_model = self._compute_avg_correlation_per_model(
            correlation_matrix, model_names
        )

        # 3. Compute optional Spearman correlations
        pairwise_spearman = None
        if "spearman" in self.config.metrics:
            pairwise_spearman = self._compute_spearman_correlations(predictions)

        # 4. Compute error-based metrics (if actuals provided)
        q_statistics = None
        disagreement = None
        double_fault = None
        ambiguity = None

        if actuals is not None:
            if "q_statistic" in self.config.metrics:
                q_statistics = self._compute_q_statistics(predictions, actuals)

            if "disagreement" in self.config.metrics:
                disagreement = self._compute_disagreement_all(predictions, actuals)

            if "double_fault" in self.config.metrics:
                double_fault = self._compute_double_fault_all(predictions, actuals)

            if "ambiguity" in self.config.metrics:
                ambiguity = self._compute_ambiguity_all(predictions, actuals)

        # 5. Aggregate diversity scores
        diversity_scores = self._aggregate_diversity_scores(
            pairwise_correlations,
            q_statistics,
            disagreement,
            double_fault,
            ambiguity,
        )

        # 6. Rank models by diversity contribution
        model_rankings = self._rank_models_by_diversity(avg_correlation_per_model)

        # 7. Compute average correlation and aggregate diversity
        avg_correlation = float(np.mean(list(pairwise_correlations.values())))
        aggregate_diversity = 1.0 - avg_correlation

        # 8. Create metadata
        processing_time = time.time() - start_time
        metadata = {
            "n_models": len(model_names),
            "n_samples": n_samples,
            "n_pairs": len(pairwise_correlations),
            "processing_time_seconds": processing_time,
            "has_actuals": actuals is not None,
            "timestamp": datetime.now().isoformat(),
            "metrics_computed": self.config.metrics.copy(),
            "config": {
                "correlation_threshold": self.config.correlation_threshold,
                "error_tolerance": self.config.error_tolerance,
            },
        }

        result = DiversityResult(
            pairwise_correlations=pairwise_correlations,
            avg_correlation=avg_correlation,
            correlation_matrix=correlation_matrix,
            avg_correlation_per_model=avg_correlation_per_model,
            diversity_scores=diversity_scores,
            model_rankings=model_rankings,
            pairwise_spearman=pairwise_spearman,
            q_statistics=q_statistics,
            disagreement=disagreement,
            double_fault=double_fault,
            ambiguity=ambiguity,
            aggregate_diversity=aggregate_diversity,
            metadata=metadata,
            model_names=model_names,
        )

        logger.info(
            "Computed diversity for %d models: avg_corr=%.3f, diversity=%.3f (%.2fs)",
            len(model_names),
            avg_correlation,
            aggregate_diversity,
            processing_time,
        )

        return result

    def select_subset(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
        method: str | None = None,
    ) -> list[str]:
        """Select optimal subset of models for ensemble.

        Uses the configured or specified selection strategy to choose
        the most diverse and effective subset of models.

        Args:
            predictions: Dictionary of model predictions.
            actuals: Ground truth values for performance evaluation.
            method: Selection strategy. If None, uses config.selection_method.
                Options: "greedy_forward", "greedy_backward", "correlation_clustering".

        Returns:
            List of selected model names.

        Raises:
            ValueError: If method is not supported.

        Example:
            >>> subset = analyzer.select_subset(
            ...     predictions, actuals, method="greedy_forward"
            ... )
            >>> print(f"Selected models: {subset}")
        """
        method = method or self.config.selection_method

        if method == "greedy_forward":
            return self._greedy_forward_selection(predictions, actuals)
        if method == "greedy_backward":
            return self._greedy_backward_selection(predictions, actuals)
        if method == "correlation_clustering":
            return self._correlation_clustering_selection(predictions, actuals)
        msg = f"Unknown selection method: {method}"
        raise ValueError(msg)

    def get_redundant_models(
        self,
        result: DiversityResult,
    ) -> list[str]:
        """Identify redundant models based on correlation threshold.

        Args:
            result: DiversityResult from analyze().

        Returns:
            List of model names that are potentially redundant.

        Example:
            >>> redundant = analyzer.get_redundant_models(result)
            >>> print(f"Redundant models: {redundant}")
        """
        # Count how many high correlations each model has
        redundancy_score = dict.fromkeys(result.model_names, 0)

        for (model1, model2), corr in result.pairwise_correlations.items():
            if corr >= self.config.correlation_threshold:
                redundancy_score[model1] += 1
                redundancy_score[model2] += 1

        # Models with any high correlation are potentially redundant
        redundant = [name for name, score in redundancy_score.items() if score > 0]

        logger.info(
            "Found %d potentially redundant models (threshold=%.2f): %s",
            len(redundant),
            self.config.correlation_threshold,
            redundant,
        )

        return redundant

    def plot_correlation_matrix(
        self,
        result: DiversityResult,
    ) -> dict[str, Any]:
        """Generate correlation heatmap data for visualization.

        Args:
            result: DiversityResult from analyze().

        Returns:
            Dictionary with visualization data:
                - "matrix": Correlation matrix DataFrame
                - "labels": Model names
                - "values": 2D numpy array of correlations

        Example:
            >>> heatmap_data = analyzer.plot_correlation_matrix(result)
            >>> # Use with matplotlib: plt.imshow(heatmap_data["values"])
        """
        matrix = result.correlation_matrix

        return {
            "matrix": matrix,
            "labels": list(matrix.index),
            "values": matrix.values,
            "title": "Model Prediction Correlation Matrix",
            "xlabel": "Model",
            "ylabel": "Model",
            "cmap": "coolwarm",
            "vmin": -1.0,
            "vmax": 1.0,
        }

    def get_dendrogram_data(
        self,
        result: DiversityResult,
    ) -> dict[str, Any]:
        """Generate dendrogram data for hierarchical clustering visualization.

        Args:
            result: DiversityResult from analyze().

        Returns:
            Dictionary with dendrogram data compatible with scipy/matplotlib.

        Example:
            >>> dendro_data = analyzer.get_dendrogram_data(result)
            >>> # Use with scipy: dendrogram(dendro_data["linkage_matrix"])
        """
        # Convert correlation to distance (1 - correlation)
        corr_matrix = result.correlation_matrix.values
        distance_matrix = 1.0 - corr_matrix

        # Convert to condensed distance matrix for linkage
        condensed_dist = squareform(distance_matrix, checks=False)

        # Compute hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method="average")

        return {
            "linkage_matrix": linkage_matrix,
            "labels": result.model_names,
            "distance_metric": "1 - correlation",
            "linkage_method": "average",
        }

    def get_diversity_report(
        self,
        result: DiversityResult,
    ) -> str:
        """Generate text summary of diversity analysis.

        Args:
            result: DiversityResult from analyze().

        Returns:
            Formatted text report summarizing diversity metrics.

        Example:
            >>> report = analyzer.get_diversity_report(result)
            >>> print(report)
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ENSEMBLE DIVERSITY ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append(f"Number of models: {len(result.model_names)}")
        lines.append(f"Models: {', '.join(result.model_names)}")
        lines.append(f"Number of samples: {result.metadata.get('n_samples', 'N/A')}")
        lines.append("")

        # Overall diversity
        lines.append("OVERALL DIVERSITY METRICS")
        lines.append("-" * 60)
        lines.append(f"Average correlation: {result.avg_correlation:.4f}")
        lines.append(f"Aggregate diversity: {result.aggregate_diversity:.4f}")
        lines.append("")

        # Diversity scores
        if result.diversity_scores:
            lines.append("DIVERSITY SCORES BY METRIC")
            lines.append("-" * 60)
            for metric, score in sorted(result.diversity_scores.items()):
                lines.append(f"  {metric:20s}: {score:.4f}")
            lines.append("")

        # Model rankings
        lines.append("MODEL RANKINGS (by diversity contribution)")
        lines.append("-" * 60)
        for i, (model, score) in enumerate(result.model_rankings, 1):
            avg_corr = result.avg_correlation_per_model.get(model, 0.0)
            lines.append(f"  {i}. {model:15s}  diversity={score:.4f}  avg_corr={avg_corr:.4f}")
        lines.append("")

        # Most/least diverse pairs
        if result.pairwise_correlations:
            try:
                m1, m2, corr = result.get_most_diverse_pair()
                lines.append(f"Most diverse pair: {m1} & {m2} (correlation: {corr:.4f})")
            except ValueError:
                pass

            try:
                m1, m2, corr = result.get_least_diverse_pair()
                lines.append(f"Least diverse pair: {m1} & {m2} (correlation: {corr:.4f})")
            except ValueError:
                pass
            lines.append("")

        # Redundant models
        redundant = self.get_redundant_models(result)
        if redundant:
            lines.append(f"REDUNDANT MODELS (corr > {self.config.correlation_threshold:.2f})")
            lines.append("-" * 60)
            for model in redundant:
                lines.append(f"  - {model}")
            lines.append("")

        # Recommended subset
        if result.recommended_subset:
            lines.append("RECOMMENDED ENSEMBLE SUBSET")
            lines.append("-" * 60)
            lines.append(f"Selected models: {', '.join(result.recommended_subset)}")
            lines.append(f"Subset size: {len(result.recommended_subset)}")
            lines.append("")

        # Error-based metrics (if available)
        if result.metadata.get("has_actuals"):
            lines.append("ERROR-BASED METRICS")
            lines.append("-" * 60)
            if result.q_statistics:
                avg_q = float(np.mean(list(result.q_statistics.values())))
                lines.append(f"  Average Q-statistic: {avg_q:.4f}")
            if result.disagreement:
                avg_disagree = float(np.mean(list(result.disagreement.values())))
                lines.append(f"  Average disagreement: {avg_disagree:.4f}")
            if result.double_fault:
                avg_df = float(np.mean(list(result.double_fault.values())))
                lines.append(f"  Average double fault: {avg_df:.4f}")
            if result.ambiguity:
                avg_amb = float(np.mean(list(result.ambiguity.values())))
                lines.append(f"  Average ambiguity: {avg_amb:.4f}")
            lines.append("")

        # Metadata
        lines.append("ANALYSIS METADATA")
        lines.append("-" * 60)
        lines.append(f"Processing time: {result.metadata.get('processing_time_seconds', 0):.3f}s")
        if "timestamp" in result.metadata:
            lines.append(f"Timestamp: {result.metadata['timestamp']}")
        lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    # ========================================================================
    # PRIVATE METHODS - Metric Computation
    # ========================================================================

    def _compute_correlations(
        self,
        predictions: dict[str, np.ndarray],
    ) -> tuple[dict[tuple[str, str], float], pd.DataFrame]:
        """Compute pairwise Pearson correlations and matrix.

        Args:
            predictions: Dictionary of model predictions.

        Returns:
            Tuple of (pairwise_correlations_dict, correlation_matrix_df).
        """
        model_names = sorted(predictions.keys())
        pairwise_correlations = {}

        for model1, model2 in combinations(model_names, 2):
            pred1 = predictions[model1]
            pred2 = predictions[model2]

            # Remove NaN values
            mask = ~(np.isnan(pred1) | np.isnan(pred2))
            if mask.sum() < 2:
                logger.warning(
                    "Not enough valid samples for correlation between %s and %s",
                    model1,
                    model2,
                )
                continue

            pred1_clean = pred1[mask]
            pred2_clean = pred2[mask]

            # Compute Pearson correlation
            if np.std(pred1_clean) > 0 and np.std(pred2_clean) > 0:
                corr, _ = pearsonr(pred1_clean, pred2_clean)
                pairwise_correlations[(model1, model2)] = float(corr)
            else:
                logger.warning(
                    "Zero variance in predictions for %s or %s, skipping correlation",
                    model1,
                    model2,
                )

        # Build correlation matrix
        correlation_matrix = self._build_correlation_matrix(pairwise_correlations, model_names)

        return pairwise_correlations, correlation_matrix

    def _build_correlation_matrix(
        self,
        pairwise_correlations: dict[tuple[str, str], float],
        model_names: list[str],
    ) -> pd.DataFrame:
        """Build symmetric correlation matrix from pairwise correlations.

        Args:
            pairwise_correlations: Dictionary of pairwise correlations.
            model_names: List of model names.

        Returns:
            Correlation matrix as DataFrame.
        """
        n_models = len(model_names)
        matrix = np.eye(n_models)

        name_to_idx = {name: idx for idx, name in enumerate(model_names)}

        for (model1, model2), corr in pairwise_correlations.items():
            idx1 = name_to_idx[model1]
            idx2 = name_to_idx[model2]
            matrix[idx1, idx2] = corr
            matrix[idx2, idx1] = corr  # Symmetric

        return pd.DataFrame(matrix, index=model_names, columns=model_names)

    def _compute_avg_correlation_per_model(
        self,
        correlation_matrix: pd.DataFrame,
        model_names: list[str],
    ) -> dict[str, float]:
        """Compute average correlation for each model with all others.

        Args:
            correlation_matrix: Correlation matrix DataFrame.
            model_names: List of model names.

        Returns:
            Dictionary mapping model names to their average correlations.
        """
        avg_corr_per_model = {}

        for model in model_names:
            # Get correlations with other models (exclude self-correlation of 1.0)
            corrs = correlation_matrix.loc[model, [m for m in model_names if m != model]]
            avg_corr_per_model[model] = float(corrs.mean())

        return avg_corr_per_model

    def _compute_spearman_correlations(
        self,
        predictions: dict[str, np.ndarray],
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise Spearman rank correlations.

        Args:
            predictions: Dictionary of model predictions.

        Returns:
            Dictionary of pairwise Spearman correlations.
        """
        model_names = sorted(predictions.keys())
        pairwise_spearman = {}

        for model1, model2 in combinations(model_names, 2):
            pred1 = predictions[model1]
            pred2 = predictions[model2]

            # Remove NaN values
            mask = ~(np.isnan(pred1) | np.isnan(pred2))
            if mask.sum() < 3:
                continue

            pred1_clean = pred1[mask]
            pred2_clean = pred2[mask]

            try:
                spearman_corr, _ = spearmanr(pred1_clean, pred2_clean)
                pairwise_spearman[(model1, model2)] = float(spearman_corr)
            except Exception as e:
                logger.warning(
                    "Failed to compute Spearman correlation for %s and %s: %s",
                    model1,
                    model2,
                    e,
                )

        return pairwise_spearman

    def _compute_q_statistics(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> dict[tuple[str, str], float]:
        """Compute Q-statistics for all model pairs.

        Args:
            predictions: Dictionary of model predictions.
            actuals: Ground truth values.

        Returns:
            Dictionary of Q-statistics.
        """
        model_names = sorted(predictions.keys())
        q_statistics = {}

        for model1, model2 in combinations(model_names, 2):
            q_stat = self._compute_q_statistic(
                predictions[model1],
                predictions[model2],
                actuals,
            )
            q_statistics[(model1, model2)] = q_stat

        return q_statistics

    def _compute_q_statistic(
        self,
        pred1: np.ndarray,
        pred2: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Compute Q-statistic for two models (adapted for regression).

        The Q-statistic measures whether two models tend to make correct/incorrect
        predictions on the same samples. For regression, we threshold errors to
        classify predictions as correct/incorrect.

        Q ranges from -1 to 1:
        - Q = 1: Models always agree (both correct or both wrong)
        - Q = 0: Models are independent
        - Q = -1: Models always disagree (one correct, other wrong)

        Args:
            pred1: Predictions from first model.
            pred2: Predictions from second model.
            actuals: Actual target values.

        Returns:
            Q-statistic value.
        """
        # Classify predictions as correct (within tolerance) or incorrect
        error1 = np.abs(pred1 - actuals)
        error2 = np.abs(pred2 - actuals)

        # Use relative error tolerance
        tolerance = self.config.error_tolerance * np.abs(actuals)
        tolerance = np.where(tolerance < 1e-6, 1e-6, tolerance)  # Minimum tolerance

        correct1 = error1 <= tolerance
        correct2 = error2 <= tolerance

        # Count agreement patterns
        n_both_correct = np.sum(correct1 & correct2)
        n_both_incorrect = np.sum(~correct1 & ~correct2)
        n_1correct_2incorrect = np.sum(correct1 & ~correct2)
        n_1incorrect_2correct = np.sum(~correct1 & correct2)

        # Compute Q-statistic
        numerator = (n_both_correct * n_both_incorrect) - (n_1correct_2incorrect * n_1incorrect_2correct)
        denominator = (n_both_correct * n_both_incorrect) + (n_1correct_2incorrect * n_1incorrect_2correct)

        if denominator == 0:
            return 0.0

        q_stat = numerator / denominator

        return float(q_stat)

    def _compute_disagreement_all(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> dict[tuple[str, str], float]:
        """Compute disagreement measures for all model pairs.

        Args:
            predictions: Dictionary of model predictions.
            actuals: Ground truth values.

        Returns:
            Dictionary of disagreement measures.
        """
        model_names = sorted(predictions.keys())
        disagreement = {}

        for model1, model2 in combinations(model_names, 2):
            disagree = self._compute_disagreement(
                predictions[model1],
                predictions[model2],
                actuals,
            )
            disagreement[(model1, model2)] = disagree

        return disagreement

    def _compute_disagreement(
        self,
        pred1: np.ndarray,
        pred2: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Compute disagreement measure for two models.

        Disagreement is the proportion of samples where one model is correct
        and the other is incorrect. Higher disagreement indicates more diversity.

        Args:
            pred1: Predictions from first model.
            pred2: Predictions from second model.
            actuals: Actual target values.

        Returns:
            Disagreement value (0 to 1).
        """
        # Classify predictions as correct (within tolerance) or incorrect
        error1 = np.abs(pred1 - actuals)
        error2 = np.abs(pred2 - actuals)

        # Use relative error tolerance
        tolerance = self.config.error_tolerance * np.abs(actuals)
        tolerance = np.where(tolerance < 1e-6, 1e-6, tolerance)

        correct1 = error1 <= tolerance
        correct2 = error2 <= tolerance

        # Count disagreements
        n_disagree = np.sum(correct1 != correct2)
        n_total = len(pred1)

        disagreement = n_disagree / n_total

        return float(disagreement)

    def _compute_double_fault_all(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> dict[tuple[str, str], float]:
        """Compute double fault measures for all model pairs.

        Args:
            predictions: Dictionary of model predictions.
            actuals: Ground truth values.

        Returns:
            Dictionary of double fault measures.
        """
        model_names = sorted(predictions.keys())
        double_fault = {}

        for model1, model2 in combinations(model_names, 2):
            df = self._compute_double_fault(
                predictions[model1],
                predictions[model2],
                actuals,
            )
            double_fault[(model1, model2)] = df

        return double_fault

    def _compute_double_fault(
        self,
        pred1: np.ndarray,
        pred2: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Compute double fault measure for two models.

        Double fault is the proportion of samples where both models are incorrect.
        Lower double fault indicates better complementarity.

        Args:
            pred1: Predictions from first model.
            pred2: Predictions from second model.
            actuals: Actual target values.

        Returns:
            Double fault value (0 to 1).
        """
        # Classify predictions as correct (within tolerance) or incorrect
        error1 = np.abs(pred1 - actuals)
        error2 = np.abs(pred2 - actuals)

        # Use relative error tolerance
        tolerance = self.config.error_tolerance * np.abs(actuals)
        tolerance = np.where(tolerance < 1e-6, 1e-6, tolerance)

        correct1 = error1 <= tolerance
        correct2 = error2 <= tolerance

        # Count both incorrect
        n_both_incorrect = np.sum(~correct1 & ~correct2)
        n_total = len(pred1)

        double_fault = n_both_incorrect / n_total

        return float(double_fault)

    def _compute_ambiguity_all(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> dict[tuple[str, str], float]:
        """Compute ambiguity measures for all model pairs.

        Args:
            predictions: Dictionary of model predictions.
            actuals: Ground truth values.

        Returns:
            Dictionary of ambiguity measures.
        """
        model_names = sorted(predictions.keys())
        ambiguity = {}

        for model1, model2 in combinations(model_names, 2):
            amb = self._compute_ambiguity(
                predictions[model1],
                predictions[model2],
                actuals,
            )
            ambiguity[(model1, model2)] = amb

        return ambiguity

    def _compute_ambiguity(
        self,
        pred1: np.ndarray,
        pred2: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Compute ambiguity decomposition for two models.

        Ambiguity is the variance of individual predictions around the ensemble mean.
        From bias-variance decomposition: MSE = bias² + variance - ambiguity.
        Higher ambiguity means more diverse predictions.

        Args:
            pred1: Predictions from first model.
            pred2: Predictions from second model.
            actuals: Actual target values.

        Returns:
            Ambiguity value (higher = more diverse).
        """
        # Ensemble prediction (simple average of two models)
        ensemble_pred = (pred1 + pred2) / 2.0

        # Ambiguity: average squared deviation from ensemble mean
        return float(np.mean((pred1 - ensemble_pred) ** 2 + (pred2 - ensemble_pred) ** 2))


    def _aggregate_diversity_scores(
        self,
        pairwise_correlations: dict[tuple[str, str], float],
        q_statistics: dict[tuple[str, str], float] | None,
        disagreement: dict[tuple[str, str], float] | None,
        double_fault: dict[tuple[str, str], float] | None,
        ambiguity: dict[tuple[str, str], float] | None,
    ) -> dict[str, float]:
        """Aggregate diversity scores from different metrics.

        Args:
            pairwise_correlations: Correlation-based diversity.
            q_statistics: Q-statistic based diversity.
            disagreement: Disagreement based diversity.
            double_fault: Double fault based diversity.
            ambiguity: Ambiguity based diversity.

        Returns:
            Dictionary of aggregated diversity scores.
        """
        scores = {}

        # Correlation-based diversity (1 - correlation)
        if pairwise_correlations:
            avg_corr = float(np.mean(list(pairwise_correlations.values())))
            scores["correlation"] = 1.0 - avg_corr

        # Error-based diversity
        error_based_scores = []

        if q_statistics:
            # Q-statistic diversity (lower Q = more diverse, so use 1 - Q)
            # But Q can be negative, so normalize to [0, 1]
            avg_q = float(np.mean(list(q_statistics.values())))
            scores["q_statistic"] = (1.0 - avg_q) / 2.0  # Normalize from [-1, 1] to [0, 1]
            error_based_scores.append(scores["q_statistic"])

        if disagreement:
            avg_disagree = float(np.mean(list(disagreement.values())))
            scores["disagreement"] = avg_disagree
            error_based_scores.append(avg_disagree)

        if double_fault:
            # Lower double fault is better (more complementary)
            avg_df = float(np.mean(list(double_fault.values())))
            scores["double_fault"] = 1.0 - avg_df
            error_based_scores.append(1.0 - avg_df)

        if ambiguity:
            # Normalize ambiguity to [0, 1] range (higher is more diverse)
            ambiguity_values = list(ambiguity.values())
            if ambiguity_values:
                max_amb = max(ambiguity_values) if ambiguity_values else 1.0
                avg_amb = float(np.mean(ambiguity_values))
                scores["ambiguity"] = avg_amb / max_amb if max_amb > 0 else 0.0
                error_based_scores.append(scores["ambiguity"])

        # Aggregate error-based score
        if error_based_scores:
            scores["error_based"] = float(np.mean(error_based_scores))

        # Overall diversity (weighted average of correlation and error-based)
        overall_components = []
        if "correlation" in scores:
            overall_components.append(scores["correlation"])
        if "error_based" in scores:
            overall_components.append(scores["error_based"])

        if overall_components:
            scores["overall"] = float(np.mean(overall_components))
        else:
            scores["overall"] = scores.get("correlation", 0.0)

        return scores

    def _rank_models_by_diversity(
        self,
        avg_correlation_per_model: dict[str, float],
    ) -> list[tuple[str, float]]:
        """Rank models by their diversity contribution.

        Models with lower average correlation are more diverse.

        Args:
            avg_correlation_per_model: Average correlation for each model.

        Returns:
            List of (model_name, diversity_contribution) tuples, sorted by diversity.
        """
        # Diversity contribution = 1 - avg_correlation
        diversity_contributions = [
            (model, 1.0 - avg_corr) for model, avg_corr in avg_correlation_per_model.items()
        ]

        # Sort by diversity contribution (descending)
        return sorted(diversity_contributions, key=lambda x: x[1], reverse=True)


    # ========================================================================
    # PRIVATE METHODS - Model Selection
    # ========================================================================

    def _greedy_forward_selection(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> list[str]:
        """Select models using greedy forward selection.

        Start with the best performing pair, then iteratively add the model
        that maximizes diversity while maintaining good performance.

        Args:
            predictions: Dictionary of model predictions.
            actuals: Ground truth values.

        Returns:
            List of selected model names.
        """
        model_names = list(predictions.keys())

        # 1. Find best performing pair (lowest RMSE)
        best_pair = None
        best_rmse = float("inf")

        for model1, model2 in combinations(model_names, 2):
            ensemble_pred = (predictions[model1] + predictions[model2]) / 2.0
            rmse = float(np.sqrt(np.mean((ensemble_pred - actuals) ** 2)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_pair = [model1, model2]

        selected = best_pair if best_pair else model_names[:2]
        remaining = [m for m in model_names if m not in selected]

        # 2. Iteratively add most diverse model
        while remaining and len(selected) < self.config.max_models:
            best_model = None
            best_diversity = -float("inf")
            best_rmse = float("inf")

            for candidate in remaining:
                # Compute diversity with already selected models
                candidate_preds = {**{m: predictions[m] for m in selected}, candidate: predictions[candidate]}
                result = self.analyze(candidate_preds, actuals)

                # Compute ensemble RMSE
                ensemble_pred = np.mean([predictions[m] for m in [*selected, candidate]], axis=0)
                rmse = float(np.sqrt(np.mean((ensemble_pred - actuals) ** 2)))

                # Select based on diversity (with RMSE as tiebreaker)
                if result.aggregate_diversity > best_diversity or (
                    np.isclose(result.aggregate_diversity, best_diversity) and rmse < best_rmse
                ):
                    best_diversity = result.aggregate_diversity
                    best_rmse = rmse
                    best_model = candidate

            if best_model:
                selected.append(best_model)
                remaining.remove(best_model)
            else:
                break

        # Ensure we have at least min_models
        while len(selected) < self.config.min_models and remaining:
            selected.append(remaining.pop(0))

        logger.info("Greedy forward selection: selected %d models: %s", len(selected), selected)

        return selected

    def _greedy_backward_selection(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> list[str]:
        """Select models using greedy backward elimination.

        Start with all models, then iteratively remove the most redundant model
        until reaching the optimal size or min_models.

        Args:
            predictions: Dictionary of model predictions.
            actuals: Ground truth values.

        Returns:
            List of selected model names.
        """
        selected = list(predictions.keys())

        # Remove most redundant models until we reach max_models
        while len(selected) > self.config.max_models:
            worst_model = None
            worst_diversity = float("inf")

            for candidate in selected:
                # Analyze without this model
                remaining = [m for m in selected if m != candidate]
                remaining_preds = {m: predictions[m] for m in remaining}

                if len(remaining) < 2:
                    break

                result = self.analyze(remaining_preds, actuals)

                # Model with lowest impact on diversity when removed is most redundant
                # (i.e., diversity stays high without it)
                if result.aggregate_diversity < worst_diversity:
                    worst_diversity = result.aggregate_diversity
                    worst_model = candidate

            if worst_model and len(selected) > self.config.min_models:
                selected.remove(worst_model)
            else:
                break

        logger.info("Greedy backward selection: selected %d models: %s", len(selected), selected)

        return selected

    def _correlation_clustering_selection(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> list[str]:
        """Select models using correlation-based clustering.

        Cluster models by correlation, then pick the best representative
        from each cluster.

        Args:
            predictions: Dictionary of model predictions.
            actuals: Ground truth values.

        Returns:
            List of selected model names.
        """
        # Analyze to get correlation matrix
        result = self.analyze(predictions, actuals)

        # Convert correlation to distance
        distance_matrix = 1.0 - result.correlation_matrix.values
        condensed_dist = squareform(distance_matrix, checks=False)

        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method="average")

        # Cut tree to get desired number of clusters
        from scipy.cluster.hierarchy import fcluster

        n_clusters = min(self.config.max_models, len(predictions))
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

        # Select best model from each cluster based on RMSE
        selected = []
        for cluster_id in range(1, n_clusters + 1):
            cluster_models = [
                result.model_names[i] for i, label in enumerate(cluster_labels) if label == cluster_id
            ]

            # Find best performing model in this cluster
            best_model = None
            best_rmse = float("inf")

            for model in cluster_models:
                rmse = float(np.sqrt(np.mean((predictions[model] - actuals) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model

            if best_model:
                selected.append(best_model)

        # Ensure we have at least min_models
        remaining = [m for m in predictions if m not in selected]
        while len(selected) < self.config.min_models and remaining:
            selected.append(remaining.pop(0))

        logger.info("Correlation clustering selection: selected %d models: %s", len(selected), selected)

        return selected

    # ========================================================================
    # PRIVATE METHODS - Validation
    # ========================================================================

    def _validate_predictions(self, predictions: dict[str, np.ndarray]) -> None:
        """Validate prediction dictionary.

        Args:
            predictions: Dictionary of model predictions.

        Raises:
            ValueError: If predictions are invalid.
        """
        if not predictions:
            msg = "Predictions dictionary is empty"
            raise ValueError(msg)

        if len(predictions) < 2:
            msg = f"At least 2 models required for diversity analysis, got {len(predictions)}"
            raise ValueError(msg)

        # Check all arrays have same length
        lengths = [len(pred) for pred in predictions.values()]
        if len(set(lengths)) > 1:
            msg = f"All prediction arrays must have same length, got lengths: {lengths}"
            raise ValueError(msg)

        # Check for at least some valid samples
        first_pred = next(iter(predictions.values()))
        if len(first_pred) < 2:
            msg = f"At least 2 samples required for diversity analysis, got {len(first_pred)}"
            raise ValueError(msg)

    def _validate_actuals(
        self,
        predictions: dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> None:
        """Validate actuals array.

        Args:
            predictions: Dictionary of model predictions.
            actuals: Array of actual values.

        Raises:
            ValueError: If actuals are invalid.
        """
        first_pred = next(iter(predictions.values()))
        if len(actuals) != len(first_pred):
            msg = (
                f"Actuals length ({len(actuals)}) must match "
                f"predictions length ({len(first_pred)})"
            )
            raise ValueError(msg)


# Backwards compatibility alias
EnsembleDiversityAnalyzer = DiversityAnalyzer
