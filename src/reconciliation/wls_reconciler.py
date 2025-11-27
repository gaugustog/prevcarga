"""WLS (Weighted Least Squares) reconciliation with variance-based weighting.

This module implements the WLSReconciler class for hierarchical forecast
reconciliation using Weighted Least Squares methodology with multiple
weighting schemes based on forecast variance/uncertainty.

The WLS reconciliation formula:
    G = S(S'W^{-1}S)^{-1}S'W^{-1}

where:
    - S = aggregation/summing matrix from hierarchy
    - W = weight matrix (diagonal, computed from variances)
    - G = reconciliation matrix
    - reconciled = G @ base_forecasts

WLS gives higher influence to more reliable forecasts (lower variance) in the
reconciliation process, making it more efficient than equal-weighted methods.

Weighting Schemes:
    - inverse_variance: w = 1/σ² (optimal for known variances)
    - prediction_interval: w = 1/(PI_width)² (from prediction bounds)
    - ensemble_variance: w = 1/var(ensemble) (from ensemble agreement)
    - model_confidence: w = confidence_score² (from model scores)
    - equal: uniform weights (fallback)

Example:
    ```python
    from src.reconciliation import WLSReconciler, ReconciliationConfig
    import pandas as pd
    import numpy as np

    # Calculate variances from historical residuals
    residual_variances = residuals.var(axis=0)

    # Configure WLS reconciler
    config = ReconciliationConfig(
        method="wls",
        weighting_scheme="inverse_variance",
        normalize_weights=True
    )

    # Create reconciler
    reconciler = WLSReconciler(config=config)

    # Fit to historical data to learn variances
    reconciler.fit(historical_forecasts, actuals, hierarchy)

    # Reconcile new forecasts
    result = reconciler.reconcile(base_forecasts, hierarchy)

    # Check weight diagnostics
    diagnostics = reconciler.get_weight_diagnostics()
    print(f"Weight ratio: {diagnostics['weight_ratio']:.2f}")
    ```

References:
    Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
    "Optimal Forecast Reconciliation for Hierarchical and Grouped Time
    Series Through Trace Minimization", Journal of the American Statistical
    Association, 114:526, 804-819.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from src.reconciliation.base_reconciler import BaseReconciler
from src.reconciliation.data_structures import ReconciliationConfig, ReconciliationResult
from src.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

# ruff: noqa: PLR2004

logger = get_logger(__name__)


class WeightingScheme(Enum):
    """Weighting schemes for WLS reconciliation.

    Attributes:
        INVERSE_VARIANCE: w = 1/σ² (optimal for known variances).
        PREDICTION_INTERVAL: w = 1/(PI_width)² (from prediction bounds).
        MODEL_CONFIDENCE: w = confidence_score² (from model scores).
        ENSEMBLE_VARIANCE: w = 1/var(ensemble) (from ensemble agreement).
        EQUAL: Uniform weights (fallback).
    """

    INVERSE_VARIANCE = "inverse_variance"
    PREDICTION_INTERVAL = "prediction_interval"
    MODEL_CONFIDENCE = "model_confidence"
    ENSEMBLE_VARIANCE = "ensemble_variance"
    EQUAL = "equal"


@dataclass
class WeightDiagnostics:
    """Diagnostics for weight calculation.

    Attributes:
        weighting_scheme: The scheme used for weighting.
        n_series: Number of series weighted.
        min_weight: Minimum weight value.
        max_weight: Maximum weight value.
        mean_weight: Mean weight value.
        std_weight: Standard deviation of weights.
        weight_ratio: Ratio of max to min weight.
        normalized: Whether weights were normalized.
        heavily_weighted_count: Number of heavily weighted series.
        down_weighted_count: Number of down-weighted series.
        fallback_used: Whether fallback was applied.
    """

    weighting_scheme: str
    n_series: int
    min_weight: float
    max_weight: float
    mean_weight: float
    std_weight: float
    weight_ratio: float
    normalized: bool
    heavily_weighted_count: int = 0
    down_weighted_count: int = 0
    fallback_used: bool = False


@dataclass
class WLSConfig:
    """Configuration for WLS reconciler.

    Attributes:
        weighting_scheme: Weighting scheme to use.
        normalize_weights: Whether to normalize weights.
        min_weight: Minimum allowed weight.
        max_weight: Maximum allowed weight.
        fallback_equal_weights: Use equal weights if variance unavailable.
        regularization: Ridge regularization for numerical stability.
        residual_window: Window size for variance estimation from residuals.
    """

    weighting_scheme: str = "inverse_variance"
    normalize_weights: bool = False
    min_weight: float = 1e-6
    max_weight: float = 1e6
    fallback_equal_weights: bool = True
    regularization: float = 1e-6
    residual_window: int = 100


class VarianceWeightCalculator:
    """Calculate weights from forecast variance estimates.

    Weighting Principles:
        - Lower variance -> Higher weight
        - More reliable forecasts influence reconciliation more
        - Weights inversely proportional to variance

    Schemes:
        - Inverse Variance: w = 1/σ²
        - Prediction Interval: w = 1/(PI_width)²
        - Ensemble Variance: w = 1/var(ensemble)
        - Model Confidence: w = confidence_score²

    Attributes:
        scheme: Weighting scheme to use.
        min_weight: Minimum weight (prevents zeros).
        max_weight: Maximum weight (prevents extreme values).
        normalize: Whether to normalize weights.
    """

    def __init__(
        self,
        scheme: WeightingScheme = WeightingScheme.INVERSE_VARIANCE,
        min_weight: float = 1e-6,
        max_weight: float = 1e6,
        normalize: bool = False,
    ) -> None:
        """Initialize weight calculator.

        Args:
            scheme: Weighting scheme to use.
            min_weight: Minimum weight (prevents zeros).
            max_weight: Maximum weight (prevents extreme values).
            normalize: Whether to normalize weights.
        """
        self.scheme = scheme
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.normalize = normalize

    def calculate_weights(
        self,
        variances: dict[str, np.ndarray] | None = None,
        prediction_intervals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        confidence_scores: dict[str, float] | None = None,
        ensemble_forecasts: dict[str, list[np.ndarray]] | None = None,
        node_order: list[str] | None = None,
    ) -> np.ndarray:
        """Calculate weights based on selected scheme.

        Args:
            variances: Forecast variances for each series.
            prediction_intervals: Prediction intervals (lower, upper) per series.
            confidence_scores: Model confidence scores per series.
            ensemble_forecasts: Ensemble member forecasts per series.
            node_order: Ordered list of node names.

        Returns:
            Weight vector with same length as node_order.

        Raises:
            ValueError: If required data is missing for selected scheme.
        """
        if node_order is None:
            msg = "node_order is required"
            raise ValueError(msg)

        if self.scheme == WeightingScheme.INVERSE_VARIANCE:
            if variances is None:
                msg = "Variances required for inverse variance weighting"
                raise ValueError(msg)
            weights = self._inverse_variance_weights(variances, node_order)

        elif self.scheme == WeightingScheme.PREDICTION_INTERVAL:
            if prediction_intervals is None:
                msg = "Prediction intervals required for PI weighting"
                raise ValueError(msg)
            weights = self._pi_weights(prediction_intervals, node_order)

        elif self.scheme == WeightingScheme.ENSEMBLE_VARIANCE:
            if ensemble_forecasts is None:
                msg = "Ensemble forecasts required for ensemble weighting"
                raise ValueError(msg)
            weights = self._ensemble_weights(ensemble_forecasts, node_order)

        elif self.scheme == WeightingScheme.MODEL_CONFIDENCE:
            if confidence_scores is None:
                msg = "Confidence scores required for confidence weighting"
                raise ValueError(msg)
            weights = self._confidence_weights(confidence_scores, node_order)

        elif self.scheme == WeightingScheme.EQUAL:
            weights = np.ones(len(node_order))

        else:
            msg = f"Unknown weighting scheme: {self.scheme}"
            raise ValueError(msg)

        # Validate and clip weights
        weights = self._validate_and_clip_weights(weights)

        # Normalize if requested
        if self.normalize:
            weights = weights / np.sum(weights)

        return weights

    def _inverse_variance_weights(
        self,
        variances: dict[str, np.ndarray],
        node_order: list[str],
    ) -> np.ndarray:
        """Calculate inverse variance weights: w = 1/σ².

        Args:
            variances: Variance array for each node.
            node_order: Ordered list of node names.

        Returns:
            Weight vector.
        """
        weights = []
        epsilon = 1e-10

        for node_name in node_order:
            if node_name in variances:
                var = np.mean(variances[node_name])  # Average variance over time
                # Avoid division by zero
                weight = 1.0 / (var + epsilon)
            else:
                # Fallback to unit weight
                weight = 1.0
                logger.warning("Missing variance for %s, using unit weight", node_name)

            weights.append(weight)

        return np.array(weights)

    def _pi_weights(
        self,
        prediction_intervals: dict[str, tuple[np.ndarray, np.ndarray]],
        node_order: list[str],
    ) -> np.ndarray:
        """Calculate weights from prediction interval widths.

        Wider intervals indicate more uncertainty, so they get lower weight.

        Args:
            prediction_intervals: (lower, upper) bounds for each node.
            node_order: Ordered list of node names.

        Returns:
            Weight vector.
        """
        weights = []
        epsilon = 1e-10

        for node_name in node_order:
            if node_name in prediction_intervals:
                lower, upper = prediction_intervals[node_name]
                pi_width = np.mean(upper - lower)
                # Wider interval -> lower weight
                weight = 1.0 / (pi_width**2 + epsilon)
            else:
                weight = 1.0
                logger.warning("Missing PI for %s, using unit weight", node_name)

            weights.append(weight)

        return np.array(weights)

    def _ensemble_weights(
        self,
        ensemble_forecasts: dict[str, list[np.ndarray]],
        node_order: list[str],
    ) -> np.ndarray:
        """Calculate weights from ensemble variance.

        Higher agreement among ensemble members indicates lower uncertainty.

        Args:
            ensemble_forecasts: List of ensemble member forecasts per node.
            node_order: Ordered list of node names.

        Returns:
            Weight vector.
        """
        weights = []
        epsilon = 1e-10

        for node_name in node_order:
            if node_name in ensemble_forecasts:
                ensemble = np.array(ensemble_forecasts[node_name])
                var = np.var(ensemble, axis=0).mean()  # Average variance across time
                weight = 1.0 / (var + epsilon)
            else:
                weight = 1.0
                logger.warning("Missing ensemble for %s, using unit weight", node_name)

            weights.append(weight)

        return np.array(weights)

    def _confidence_weights(
        self,
        confidence_scores: dict[str, float],
        node_order: list[str],
    ) -> np.ndarray:
        """Calculate weights from model confidence scores.

        Higher confidence scores indicate more reliable forecasts.

        Args:
            confidence_scores: Confidence score (0-1) per node.
            node_order: Ordered list of node names.

        Returns:
            Weight vector.
        """
        weights = []

        for node_name in node_order:
            if node_name in confidence_scores:
                # Confidence score should be in [0, 1]
                confidence = confidence_scores[node_name]
                # Higher confidence -> higher weight (squared to emphasize differences)
                weight = confidence**2
            else:
                weight = 0.5  # Neutral weight
                logger.warning("Missing confidence for %s, using neutral weight", node_name)

            weights.append(weight)

        return np.array(weights)

    def _validate_and_clip_weights(self, weights: np.ndarray) -> np.ndarray:
        """Validate and clip weights to acceptable range.

        Args:
            weights: Raw weight vector.

        Returns:
            Validated and clipped weight vector.
        """
        # Check for NaN or Inf
        if np.any(~np.isfinite(weights)):
            logger.warning("Non-finite weights detected, replacing with unit weights")
            weights = np.where(np.isfinite(weights), weights, 1.0)

        # Clip to valid range
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # Check weight ratio
        if np.max(weights) > 0 and np.min(weights) > 0:
            weight_ratio = np.max(weights) / np.min(weights)
            if weight_ratio > 1e6:
                logger.warning("Extreme weight ratio: %.2e", weight_ratio)

        return weights


class WLSReconciler(BaseReconciler):
    """Weighted Least Squares (WLS) hierarchical reconciliation.

    WLS reconciliation uses forecast variance to optimally weight different
    forecasts in the reconciliation process. More reliable forecasts (lower
    variance) receive higher weight.

    Algorithm:
        1. Calculate weights from forecast variances: w_i = 1/variance_i
        2. Construct weight matrix W = diag(w)
        3. Compute reconciliation matrix: G = S(S'W^{-1}S)^{-1}S'W^{-1}
        4. Apply reconciliation: y_reconciled = G @ y_base

    Weighting Schemes:
        - inverse_variance: w = 1/σ² (optimal for known variances)
        - prediction_interval: w = 1/(PI_width)²
        - ensemble_variance: w = 1/var(ensemble)
        - model_confidence: w = confidence_score²
        - equal: uniform weights (fallback)

    Advantages:
        - Adapts to forecast quality differences
        - Gives appropriate influence to reliable forecasts
        - Can leverage model uncertainty estimates
        - More efficient than equal weighting

    Example:
        >>> reconciler = WLSReconciler(config=ReconciliationConfig(
        ...     method="wls",
        ...     weighting_scheme="inverse_variance",
        ...     normalize_weights=True
        ... ))
        >>> reconciler.fit(historical_forecasts, actuals, hierarchy)
        >>> result = reconciler.reconcile(base_forecasts, hierarchy)
    """

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        """Initialize WLS reconciler.

        Args:
            config: Configuration with optional keys:
                - weighting_scheme: 'inverse_variance', 'prediction_interval', etc.
                - normalize_weights: Whether to normalize weights
                - min_weight: Minimum weight value
                - max_weight: Maximum weight value
                - fallback_equal_weights: Use equal weights if variances unavailable
                - regularization: Ridge regularization for numerical stability
        """
        super().__init__(config)

        # Parse configuration
        scheme_str = self.config.get_extra_field("weighting_scheme", "inverse_variance")
        self.weighting_scheme = WeightingScheme(scheme_str)
        self.normalize_weights = self.config.get_extra_field("normalize_weights", False)
        self.fallback_equal = self.config.get_extra_field("fallback_equal_weights", True)
        self.regularization = self.config.get_extra_field("regularization", 1e-6)
        self.residual_window = self.config.get_extra_field("residual_window", 100)

        self.weight_calculator = VarianceWeightCalculator(
            scheme=self.weighting_scheme,
            min_weight=self.config.get_extra_field("min_weight", 1e-6),
            max_weight=self.config.get_extra_field("max_weight", 1e6),
            normalize=self.normalize_weights,
        )

        # State
        self._computed_weights: np.ndarray | None = None
        self._learned_variances: dict[str, np.ndarray] | None = None
        self._node_order: list[str] | None = None
        self._reconciliation_matrix: np.ndarray | None = None

        logger.debug(
            "Initialized WLSReconciler with scheme=%s, normalize=%s",
            self.weighting_scheme.value,
            self.normalize_weights,
        )

    def fit(
        self,
        historical_forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> None:
        """Learn variance weights from historical forecast errors.

        Computes the variance of forecast errors for each node, which are
        used to construct the weight matrix during reconciliation.

        Args:
            historical_forecasts: Historical base forecasts (time x nodes).
            actuals: Historical actual values (time x nodes).
            hierarchy: HierarchyDefinition for structure.

        Raises:
            ValueError: If inputs are invalid.
        """
        start_time = time.time()
        logger.info("Fitting WLS reconciler from historical data")

        # Validate inputs
        self.validate_forecasts(historical_forecasts, hierarchy)
        self.validate_forecasts(actuals, hierarchy)

        # Get node ordering
        all_nodes = hierarchy.get_node_names_sorted()
        self._node_order = all_nodes

        # Calculate residuals
        residuals = historical_forecasts[all_nodes] - actuals[all_nodes]

        # Use most recent observations for variance estimation
        if len(residuals) > self.residual_window:
            residuals = residuals.iloc[-self.residual_window :]

        # Compute variance for each node
        self._learned_variances = {}
        for node in all_nodes:
            node_residuals = residuals[node].to_numpy()
            var = np.var(node_residuals)

            # Avoid zero variance (can happen with constant series)
            if var < 1e-10:
                var = 1e-10
                logger.warning("Near-zero variance for %s, using epsilon", node)

            self._learned_variances[node] = np.array([var])

        self.fitted = True
        fit_time = time.time() - start_time

        logger.info(
            "WLS reconciler fitted in %.2fs (learned variances for %d nodes)",
            fit_time,
            len(self._learned_variances),
        )

    def reconcile(
        self,
        base_forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        forecast_variances: dict[str, np.ndarray] | None = None,
        prediction_intervals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        confidence_scores: dict[str, float] | None = None,
        ensemble_forecasts: dict[str, list[np.ndarray]] | None = None,
    ) -> ReconciliationResult:
        """Perform WLS reconciliation.

        Args:
            base_forecasts: Base forecasts DataFrame (time x nodes).
            hierarchy: HierarchyDefinition with structure.
            forecast_variances: Optional explicit variances (overrides learned).
            prediction_intervals: Optional prediction intervals for weighting.
            confidence_scores: Optional confidence scores for weighting.
            ensemble_forecasts: Optional ensemble forecasts for weighting.

        Returns:
            ReconciliationResult with reconciled forecasts.

        Raises:
            ValueError: If validation fails or weights cannot be computed.
        """
        start_time = time.time()
        logger.info("Starting WLS reconciliation (scheme: %s)", self.weighting_scheme.value)

        # Validate inputs
        self.validate_forecasts(base_forecasts, hierarchy)

        # Get node ordering
        all_nodes = hierarchy.get_node_names_sorted()
        self._node_order = all_nodes
        n_nodes = len(all_nodes)

        # Check coherence before
        coherence_before = self.check_coherence(base_forecasts, hierarchy)
        logger.info("Coherence error before: %.6e", coherence_before)

        # Get aggregation matrix S
        if hierarchy.aggregation_matrix is None:
            hierarchy.build_aggregation_matrix()

        # Prepare forecast array (nodes x timesteps)
        y_base = base_forecasts[all_nodes].to_numpy().T  # (n_nodes, n_timesteps)

        # Determine variances to use
        variances = self._get_variances(
            forecast_variances=forecast_variances,
            n_nodes=n_nodes,
        )

        # Calculate weights
        fallback_used = False
        try:
            weights = self.weight_calculator.calculate_weights(
                variances=variances,
                prediction_intervals=prediction_intervals,
                confidence_scores=confidence_scores,
                ensemble_forecasts=ensemble_forecasts,
                node_order=all_nodes,
            )
            self._computed_weights = weights

        except ValueError as e:
            if self.fallback_equal:
                logger.warning("Weight calculation failed (%s), using equal weights", e)
                weights = np.ones(n_nodes)
                self._computed_weights = weights
                fallback_used = True
            else:
                msg = f"Weight calculation failed: {e}"
                raise ValueError(msg) from e

        logger.info("Weight range: [%.2e, %.2e]", weights.min(), weights.max())

        # Construct weight matrix W = diag(weights)
        w_matrix = np.diag(weights)

        # Compute WLS reconciliation matrix G
        g_matrix = self._compute_reconciliation_matrix(hierarchy, weight_matrix=w_matrix)
        self._reconciliation_matrix = g_matrix

        # Apply reconciliation: y_reconciled = G @ y_base
        y_reconciled = g_matrix @ y_base

        # Create reconciled DataFrame
        reconciled_df = pd.DataFrame(
            y_reconciled.T,
            index=base_forecasts.index,
            columns=all_nodes,
        )

        # Apply non-negativity if configured
        reconciled_df = self.apply_non_negativity(reconciled_df)

        # Check coherence after
        coherence_after = self.check_coherence(reconciled_df, hierarchy)
        logger.info("Coherence error after: %.6e", coherence_after)

        # Create metadata
        processing_time = time.time() - start_time
        metadata = {
            "method": "wls",
            "weighting_scheme": self.weighting_scheme.value,
            "coherence_before": coherence_before,
            "coherence_after": coherence_after,
            "processing_time_seconds": processing_time,
            "n_nodes": n_nodes,
            "n_timesteps": len(base_forecasts),
            "weight_ratio": float(weights.max() / (weights.min() + 1e-10)),
            "mean_weight": float(weights.mean()),
            "normalized_weights": self.normalize_weights,
            "fallback_used": fallback_used,
            "regularization": self.regularization,
        }

        logger.info("WLS reconciliation complete in %.2fs", processing_time)

        return ReconciliationResult(
            reconciled_forecasts=reconciled_df,
            base_forecasts=base_forecasts,
            metadata=metadata,
        )

    def _compute_reconciliation_matrix(
        self,
        hierarchy: HierarchyDefinition,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute WLS reconciliation matrix G = S(S'W^{-1}S)^{-1}S'W^{-1}.

        Args:
            hierarchy: HierarchyDefinition with aggregation matrix S.
            **kwargs: Must include 'weight_matrix' key with diagonal weight matrix W.

        Returns:
            WLS reconciliation matrix G of shape (n_total, n_total).

        Raises:
            ValueError: If weight_matrix is not provided.
        """
        w_matrix = kwargs.get("weight_matrix")
        if w_matrix is None:
            msg = "weight_matrix required for WLS reconciliation"
            raise ValueError(msg)

        s_matrix = hierarchy.aggregation_matrix

        # Invert weight matrix (easy for diagonal)
        w_diag = np.diag(w_matrix)
        w_inv_diag = 1.0 / w_diag
        w_inv = np.diag(w_inv_diag)

        # Compute S'W^{-1}S
        sws = s_matrix.T @ w_inv @ s_matrix

        # Add regularization for numerical stability
        sws += self.regularization * np.eye(sws.shape[0])

        # Check condition number
        cond_num = np.linalg.cond(sws)
        logger.info("Condition number of S'W^{-1}S: %.2e", cond_num)

        # Invert S'W^{-1}S using stable method
        sws_inv = self._stable_inversion(sws)

        # Compute full reconciliation matrix: G = S(S'W^{-1}S)^{-1}S'W^{-1}
        return s_matrix @ sws_inv @ s_matrix.T @ w_inv

    def _stable_inversion(self, matrix: np.ndarray) -> np.ndarray:
        """Invert matrix using stable method with fallbacks.

        Args:
            matrix: Square matrix to invert.

        Returns:
            Inverted matrix.
        """
        n = matrix.shape[0]

        # Try Cholesky decomposition first (most efficient)
        try:
            chol_lower = linalg.cholesky(matrix, lower=True)
            return linalg.cho_solve((chol_lower, True), np.eye(n))
        except linalg.LinAlgError:
            logger.debug("Cholesky failed, trying LU decomposition")

        # Try LU decomposition
        try:
            lu, piv = linalg.lu_factor(matrix)
            return linalg.lu_solve((lu, piv), np.eye(n))
        except linalg.LinAlgError:
            logger.warning("LU decomposition failed, using pseudo-inverse")

        # Fallback to pseudo-inverse
        return linalg.pinv(matrix)

    def _get_variances(
        self,
        forecast_variances: dict[str, np.ndarray] | None,
        n_nodes: int,  # noqa: ARG002
    ) -> dict[str, np.ndarray] | None:
        """Get variances for weight calculation.

        Uses provided variances, learned variances, or returns None for fallback.

        Args:
            forecast_variances: Explicitly provided variances.
            n_nodes: Number of nodes (for validation).

        Returns:
            Variance dictionary or None if unavailable.
        """
        # Use provided variances if available
        if forecast_variances is not None:
            return forecast_variances

        # Use learned variances if fitted
        if self.fitted and self._learned_variances is not None:
            return self._learned_variances

        # Return None to trigger fallback
        logger.warning(
            "No variances available (fitted=%s), will use fallback if enabled",
            self.fitted,
        )
        return None

    def get_weight_diagnostics(self) -> WeightDiagnostics | dict[str, Any]:
        """Get diagnostic information about computed weights.

        Returns:
            WeightDiagnostics object or dict with error if no weights computed.
        """
        if self._computed_weights is None:
            return {"error": "No weights computed yet"}

        weights = self._computed_weights

        # Identify extreme weights
        median_weight = np.median(weights)
        heavily_weighted = np.where(weights > 10 * median_weight)[0]
        down_weighted = np.where(weights < 0.1 * median_weight)[0]

        return WeightDiagnostics(
            weighting_scheme=self.weighting_scheme.value,
            n_series=len(weights),
            min_weight=float(weights.min()),
            max_weight=float(weights.max()),
            mean_weight=float(weights.mean()),
            std_weight=float(weights.std()),
            weight_ratio=float(weights.max() / (weights.min() + 1e-10)),
            normalized=self.normalize_weights,
            heavily_weighted_count=len(heavily_weighted),
            down_weighted_count=len(down_weighted),
        )

    def get_weights_by_node(self) -> dict[str, float] | None:
        """Get weights mapped to node names.

        Returns:
            Dictionary mapping node names to weights, or None if not computed.
        """
        if self._computed_weights is None or self._node_order is None:
            return None

        return {
            node: float(weight)
            for node, weight in zip(self._node_order, self._computed_weights, strict=True)
        }

    def save(self, path: str | Path) -> None:
        """Save reconciler state to disk.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create state dict
        state = {
            "config": self.config.to_dict(),
            "fitted": self.fitted,
            "weighting_scheme": self.weighting_scheme.value,
            "normalize_weights": self.normalize_weights,
            "fallback_equal": self.fallback_equal,
            "regularization": self.regularization,
            "residual_window": self.residual_window,
            "learned_variances": (
                {k: v.tolist() for k, v in self._learned_variances.items()}
                if self._learned_variances
                else None
            ),
            "node_order": self._node_order,
        }

        with path.open("w") as f:
            json.dump(state, f, indent=2)

        logger.info("Saved WLS reconciler to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> WLSReconciler:
        """Load reconciler state from disk.

        Args:
            path: Path to saved state file.

        Returns:
            WLSReconciler instance with restored state.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(path)

        if not path.exists():
            msg = f"State file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open() as f:
            state = json.load(f)

        # Create config from saved state
        config = ReconciliationConfig(**state["config"])

        # Create instance
        instance = cls(config=config)

        # Restore state
        instance.fitted = state["fitted"]
        instance._node_order = state.get("node_order")

        # Restore learned variances if present
        if state.get("learned_variances"):
            instance._learned_variances = {
                k: np.array(v) for k, v in state["learned_variances"].items()
            }

        logger.info("Loaded WLS reconciler from %s (fitted=%s)", path, instance.fitted)

        return instance
