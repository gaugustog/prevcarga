"""OLS (Ordinary Least Squares) reconciliation with advanced covariance estimation.

This module implements the OLSReconciler class for hierarchical forecast
reconciliation using Ordinary Least Squares methodology with multiple
covariance estimation methods.

The OLS reconciliation formula:
    G = S(S'Σ^{-1}S)^{-1}S'Σ^{-1}

where:
    - S = aggregation/summing matrix from hierarchy
    - Σ = forecast error covariance matrix
    - G = reconciliation matrix
    - reconciled = G @ base_forecasts

OLS provides optimal (minimum variance unbiased) reconciled forecasts when the
error covariance structure is known or can be accurately estimated.

Covariance Methods:
    - sample: Empirical covariance from residuals
    - ledoit_wolf: Ledoit-Wolf shrinkage (optimal for high-dim scenarios)
    - factor: Factor-based covariance (structured errors)
    - diagonal: Diagonal covariance (independence assumption)

Example:
    ```python
    from src.reconciliation import OLSReconciler, ReconciliationConfig
    import pandas as pd

    # Configure OLS reconciler
    config = ReconciliationConfig(
        method="ols",
        covariance_method="ledoit_wolf",
        regularization=1e-6,
        check_coherence=True
    )

    # Create reconciler
    reconciler = OLSReconciler(config=config)

    # Fit to historical data
    reconciler.fit(historical_forecasts, actuals, hierarchy)

    # Reconcile new forecasts
    result = reconciler.reconcile(base_forecasts, hierarchy)
    ```

References:
    Hyndman, R. J., & Athanasopoulos, G. (2021). "Forecasting: Principles
    and Practice", 3rd edition. Chapter 11: Hierarchical and grouped
    time series.
"""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.covariance import LedoitWolf

from src.reconciliation.base_reconciler import BaseReconciler
from src.reconciliation.data_structures import ReconciliationConfig, ReconciliationResult
from src.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

# ruff: noqa: PLR2004

logger = get_logger(__name__)


# Type alias for covariance estimation methods
CovarianceMethod = Literal["sample", "ledoit_wolf", "factor", "diagonal"]


class OLSCovarianceMethod(Enum):
    """Covariance estimation methods for OLS reconciliation.

    Attributes:
        SAMPLE: Empirical covariance from residuals.
        LEDOIT_WOLF: Ledoit-Wolf shrinkage estimator.
        FACTOR: Factor-based covariance.
        DIAGONAL: Diagonal covariance (independence assumption).
    """

    SAMPLE = "sample"
    LEDOIT_WOLF = "ledoit_wolf"
    FACTOR = "factor"
    DIAGONAL = "diagonal"


class SampleCovarianceEstimator:
    """Sample covariance estimator with bias correction.

    Computes empirical covariance matrix from residuals with Bessel's
    correction for small sample bias.

    Example:
        >>> estimator = SampleCovarianceEstimator()
        >>> cov = estimator.estimate(residuals)
        >>> print(f"Covariance shape: {cov.shape}")
    """

    def estimate(self, residuals: np.ndarray) -> np.ndarray:
        """Estimate sample covariance.

        Args:
            residuals: Historical forecast residuals (n_samples, n_series).

        Returns:
            Sample covariance matrix (n_series, n_series).

        Raises:
            ValueError: If fewer than 2 samples provided.
        """
        n_samples = residuals.shape[0]

        if n_samples < 2:
            msg = f"Need at least 2 samples for covariance estimation, got {n_samples}"
            raise ValueError(msg)

        # Center residuals
        residuals_centered = residuals - residuals.mean(axis=0)

        # Compute sample covariance with Bessel's bias correction
        return (residuals_centered.T @ residuals_centered) / (n_samples - 1)


class LedoitWolfEstimator:
    """Ledoit-Wolf shrinkage estimator.

    Optimal shrinkage towards identity matrix for high-dimensional cases.
    Particularly useful when n_samples is not much larger than n_features.

    The estimator automatically determines the optimal shrinkage intensity
    by minimizing the expected loss.

    Attributes:
        assume_centered: Whether residuals are already centered.
        estimator: sklearn LedoitWolf estimator instance.

    Example:
        >>> estimator = LedoitWolfEstimator()
        >>> cov = estimator.estimate(residuals)
        >>> print(f"Shrinkage: {estimator.estimator.shrinkage_:.3f}")
    """

    def __init__(self, assume_centered: bool = False) -> None:
        """Initialize Ledoit-Wolf estimator.

        Args:
            assume_centered: Whether residuals are already centered.
        """
        self.assume_centered = assume_centered
        self.estimator = LedoitWolf(assume_centered=assume_centered)

    def estimate(self, residuals: np.ndarray) -> np.ndarray:
        """Estimate shrunk covariance using Ledoit-Wolf.

        Args:
            residuals: Historical forecast residuals (n_samples, n_series).

        Returns:
            Shrunk covariance matrix (n_series, n_series).
        """
        self.estimator.fit(residuals)

        logger.info(
            "Ledoit-Wolf shrinkage intensity: %.3f",
            self.estimator.shrinkage_,
        )

        return self.estimator.covariance_


class FactorCovarianceEstimator:
    """Factor-based covariance estimator.

    Models covariance using factor structure: Σ = ΛΛ' + Ψ

    where Λ are factor loadings and Ψ is the diagonal specific variance matrix.
    Useful when errors have latent common factors.

    Attributes:
        n_factors: Number of factors to extract (None for automatic selection).

    Example:
        >>> estimator = FactorCovarianceEstimator(n_factors=3)
        >>> cov = estimator.estimate(residuals)
    """

    def __init__(self, n_factors: int | None = None) -> None:
        """Initialize factor estimator.

        Args:
            n_factors: Number of factors (None for automatic selection
                using Kaiser criterion).
        """
        self.n_factors = n_factors

    def estimate(self, residuals: np.ndarray) -> np.ndarray:
        """Estimate factor-based covariance.

        Args:
            residuals: Historical forecast residuals (n_samples, n_series).

        Returns:
            Factor-based covariance matrix (n_series, n_series).
        """
        _n_samples, n_series = residuals.shape

        # Determine number of factors
        if self.n_factors is None:
            # Use Kaiser criterion: keep factors with eigenvalue > 1
            sample_cov = SampleCovarianceEstimator().estimate(residuals)
            eigenvalues = np.linalg.eigvalsh(sample_cov)
            self.n_factors = max(1, int(np.sum(eigenvalues > 1)))
            logger.info("Auto-selected %d factors using Kaiser criterion", self.n_factors)

        # Perform SVD for factor extraction
        residuals_centered = residuals - residuals.mean(axis=0)
        _u, s, vt = np.linalg.svd(residuals_centered, full_matrices=False)

        # Extract top k factors
        k = min(self.n_factors, len(s), n_series)
        # Factor loadings: ΛΛ' = V[:k]' * s[:k]² * V[:k]
        factor_loadings = vt[:k, :].T * (s[:k] / np.sqrt(residuals.shape[0] - 1))

        # Estimate factor covariance component
        factor_cov = factor_loadings @ factor_loadings.T

        # Estimate specific variances (diagonal)
        sample_cov = SampleCovarianceEstimator().estimate(residuals)
        specific_var = np.diag(np.maximum(np.diag(sample_cov - factor_cov), 1e-8))

        # Combine: Σ = ΛΛ' + Ψ
        return factor_cov + specific_var


class OLSReconciler(BaseReconciler):
    """OLS (Ordinary Least Squares) hierarchical reconciliation.

    OLS reconciliation produces optimal (minimum variance unbiased) reconciled
    forecasts when the forecast error covariance structure is known or can be
    accurately estimated.

    Algorithm:
        1. Estimate forecast error covariance Σ from historical residuals
        2. Compute reconciliation matrix: G = S(S'Σ^{-1}S)^{-1}S'Σ^{-1}
        3. Apply reconciliation: y_reconciled = G @ y_base

    Covariance Methods:
        - sample: Empirical covariance from residuals
        - ledoit_wolf: Ledoit-Wolf shrinkage (optimal for high-dim)
        - factor: Factor-based covariance (structured errors)
        - diagonal: Diagonal covariance (independent errors)

    Numerical Stability:
        - Ridge regularization: Σ_reg = Σ + λI
        - Multiple decomposition methods (Cholesky, LU, SVD)
        - Condition number monitoring and adaptive regularization
        - Graceful degradation to pseudo-inverse

    Attributes:
        config: ReconciliationConfig with method="ols".
        covariance_method: Method for covariance estimation.
        regularization: Ridge regularization parameter.
        adaptive_regularization: Whether to auto-adjust regularization.
        n_factors: Number of factors for factor method.
        covariance_matrix: Fitted error covariance matrix.
        fitted: Whether reconciler has been fitted.

    Example:
        >>> config = ReconciliationConfig(
        ...     method="ols",
        ...     covariance_method="ledoit_wolf",
        ...     regularization=1e-6
        ... )
        >>> reconciler = OLSReconciler(config=config)
        >>> reconciler.fit(historical_forecasts, actuals, hierarchy)
        >>> result = reconciler.reconcile(base_forecasts, hierarchy)
    """

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        """Initialize OLS reconciler.

        Args:
            config: Reconciliation configuration. Can include extra fields:
                - covariance_method: 'sample', 'ledoit_wolf', 'factor', 'diagonal'
                - regularization: Ridge regularization parameter (default: 1e-6)
                - n_factors: Number of factors for factor method
                - adaptive_regularization: Auto-adjust based on condition number
                - cache_matrices: Cache computed reconciliation matrices
        """
        super().__init__(config)

        # Get configuration parameters
        self.covariance_method: CovarianceMethod = self.config.get_extra_field(
            "covariance_method", "ledoit_wolf"
        )
        self.regularization: float = self.config.get_extra_field("regularization", 1e-6)
        self.n_factors: int | None = self.config.get_extra_field("n_factors", None)
        self.adaptive_regularization: bool = self.config.get_extra_field(
            "adaptive_regularization", True
        )
        self.cache_matrices: bool = self.config.get_extra_field("cache_matrices", True)

        # Initialize covariance estimator
        self._cov_estimator = self._create_covariance_estimator()

        # State
        self.covariance_matrix: np.ndarray | None = None
        self._cached_reconciliation_matrix: np.ndarray | None = None
        self._last_diagnostics: dict[str, Any] = {}

        logger.info(
            "Initialized OLS reconciler (covariance_method=%s, regularization=%.2e)",
            self.covariance_method,
            self.regularization,
        )

    def _create_covariance_estimator(
        self,
    ) -> SampleCovarianceEstimator | LedoitWolfEstimator | FactorCovarianceEstimator | None:
        """Create appropriate covariance estimator based on method.

        Returns:
            Covariance estimator instance or None for diagonal method.
        """
        if self.covariance_method == "sample":
            return SampleCovarianceEstimator()
        if self.covariance_method == "ledoit_wolf":
            return LedoitWolfEstimator()
        if self.covariance_method == "factor":
            return FactorCovarianceEstimator(n_factors=self.n_factors)
        if self.covariance_method == "diagonal":
            return None  # Will use diagonal extraction from sample
        msg = f"Unknown covariance method: {self.covariance_method}"
        raise ValueError(msg)

    def fit(
        self,
        historical_forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        _hierarchy: HierarchyDefinition,
    ) -> None:
        """Estimate error covariance matrix from historical data.

        Computes residuals from historical forecasts and actuals, then
        estimates the covariance matrix using the configured method.

        Args:
            historical_forecasts: DataFrame with historical forecasts (time x nodes).
            actuals: DataFrame with actual values (time x nodes).
            _hierarchy: HierarchyDefinition with structure (currently unused).

        Raises:
            ValueError: If data shapes don't match or insufficient data.
        """
        logger.info(
            "Fitting OLS reconciler with %d historical observations",
            len(historical_forecasts),
        )

        # Validate inputs
        if len(historical_forecasts) != len(actuals):
            msg = (
                f"Forecasts and actuals must have same length: "
                f"{len(historical_forecasts)} vs {len(actuals)}"
            )
            raise ValueError(msg)

        # Compute residuals
        common_cols = list(
            set(historical_forecasts.columns) & set(actuals.columns)
        )
        residuals = historical_forecasts[common_cols] - actuals[common_cols]

        # Estimate covariance
        residuals_array = residuals.to_numpy()
        self.covariance_matrix = self._estimate_covariance(residuals_array)

        # Clear cached matrices
        self._cached_reconciliation_matrix = None

        self.fitted = True
        logger.info(
            "OLS reconciler fitted. Covariance shape: %s",
            self.covariance_matrix.shape,
        )

    def reconcile(
        self,
        base_forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        residuals: np.ndarray | None = None,
    ) -> ReconciliationResult:
        """Reconcile base forecasts to ensure hierarchical coherence.

        Applies OLS reconciliation to transform base forecasts into
        coherent reconciled forecasts that satisfy aggregation constraints.

        Args:
            base_forecasts: DataFrame with base forecasts (time x nodes).
            hierarchy: HierarchyDefinition with structure and aggregation matrix.
            residuals: Optional historical residuals for covariance estimation
                if reconciler not pre-fitted.

        Returns:
            ReconciliationResult with reconciled forecasts and metadata.

        Raises:
            ValueError: If base_forecasts doesn't match hierarchy.
        """
        start_time = time.time()
        logger.info(
            "Starting OLS reconciliation (method: %s)",
            self.covariance_method,
        )

        # Validate inputs
        errors = self._validate_inputs(base_forecasts, hierarchy)
        if errors:
            logger.warning("Validation issues: %s", errors)

        # Get node order from hierarchy
        node_order = hierarchy.get_node_names_sorted()
        n_nodes = len(node_order)

        # Get aggregation matrix
        if hierarchy.aggregation_matrix is None:
            hierarchy.build_aggregation_matrix()
        s_matrix = hierarchy.aggregation_matrix
        logger.info("Aggregation matrix shape: %s", s_matrix.shape)

        # Get or estimate covariance matrix
        if self.covariance_matrix is not None:
            sigma = self.covariance_matrix
        elif residuals is not None:
            sigma = self._estimate_covariance(residuals)
        else:
            # Use identity (equivalent to OLS with equal weights)
            logger.warning("No covariance provided, using identity matrix")
            sigma = np.eye(n_nodes)

        # Ensure covariance matches node count
        if sigma.shape[0] != n_nodes:
            logger.warning(
                "Covariance dimension mismatch: %d vs %d nodes. Using identity.",
                sigma.shape[0],
                n_nodes,
            )
            sigma = np.eye(n_nodes)

        # Apply regularization
        sigma_reg = self._apply_regularization(sigma)

        # Compute or retrieve reconciliation matrix
        if self._cached_reconciliation_matrix is None or not self.cache_matrices:
            g_matrix = self._compute_reconciliation_matrix(
                hierarchy,
                covariance_matrix=sigma_reg,
            )
            if self.cache_matrices:
                self._cached_reconciliation_matrix = g_matrix
        else:
            g_matrix = self._cached_reconciliation_matrix
            logger.debug("Using cached reconciliation matrix")

        # Apply reconciliation - G is (n_nodes x n_nodes), forecasts (n_nodes x T)
        n_timesteps = len(base_forecasts)

        # Reshape for batch processing
        y_base_matrix = base_forecasts[node_order].to_numpy().T  # (n_nodes, T)
        y_reconciled_matrix = g_matrix @ y_base_matrix  # (n_nodes, T)

        # Reconstruct DataFrame
        reconciled_forecasts = pd.DataFrame(
            y_reconciled_matrix.T,
            index=base_forecasts.index,
            columns=node_order,
        )

        # Apply non-negativity if configured
        if self.config.non_negative:
            reconciled_forecasts = reconciled_forecasts.clip(lower=0)

        # Calculate coherence metrics
        coherence_before = self._calculate_coherence_error(base_forecasts, hierarchy)
        coherence_after = self._calculate_coherence_error(reconciled_forecasts, hierarchy)

        # Compute processing time
        processing_time = time.time() - start_time

        # Store diagnostics
        condition_number = float(np.linalg.cond(sigma_reg))
        self._last_diagnostics = {
            "covariance_method": self.covariance_method,
            "condition_number": condition_number,
            "regularization_applied": self.regularization,
            "n_nodes": n_nodes,
            "n_timesteps": n_timesteps,
            "coherence_before": coherence_before,
            "coherence_after": coherence_after,
        }

        # Create metadata
        metadata = {
            "method": "ols",
            "covariance_method": self.covariance_method,
            "coherence_before": coherence_before,
            "coherence_after": coherence_after,
            "processing_time_s": processing_time,
            "condition_number": condition_number,
            "regularization": self.regularization,
        }

        logger.info(
            "OLS reconciliation complete in %.3fs. "
            "Coherence: %.2e -> %.2e",
            processing_time,
            coherence_before,
            coherence_after,
        )

        return ReconciliationResult(
            reconciled_forecasts=reconciled_forecasts,
            base_forecasts=base_forecasts,
            metadata=metadata,
        )

    def _compute_reconciliation_matrix(
        self,
        hierarchy: HierarchyDefinition,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute OLS reconciliation matrix G = S(S'Σ^{-1}S)^{-1}S'Σ^{-1}.

        Args:
            hierarchy: HierarchyDefinition with aggregation matrix S.
            **kwargs: Must include 'covariance_matrix' with Σ.

        Returns:
            Reconciliation matrix G of shape (n_total, n_total).
        """
        sigma = kwargs.get("covariance_matrix")
        if sigma is None:
            msg = "covariance_matrix required for OLS reconciliation"
            raise ValueError(msg)

        s_matrix = hierarchy.aggregation_matrix

        # Invert covariance matrix
        sigma_inv = self._stable_inversion(sigma)

        # Compute S'Σ^{-1}S
        sts = s_matrix.T @ sigma_inv @ s_matrix

        # Check condition number
        cond_num = np.linalg.cond(sts)
        logger.debug("Condition number of S'Σ^{-1}S: %.2e", cond_num)

        if cond_num > 1e10:
            logger.warning("S'Σ^{-1}S is ill-conditioned (cond=%.2e)", cond_num)

        # Invert S'Σ^{-1}S
        sts_inv = self._stable_inversion(sts)

        # Compute full reconciliation matrix: G = S(S'Σ^{-1}S)^{-1}S'Σ^{-1}
        return s_matrix @ sts_inv @ s_matrix.T @ sigma_inv

    def _estimate_covariance(self, residuals: np.ndarray) -> np.ndarray:
        """Estimate covariance matrix from residuals.

        Args:
            residuals: Historical forecast residuals (n_samples, n_series).

        Returns:
            Estimated covariance matrix (n_series, n_series).
        """
        if self.covariance_method == "diagonal":
            # Diagonal covariance (independent errors)
            sample_cov = SampleCovarianceEstimator().estimate(residuals)
            return np.diag(np.diag(sample_cov))

        if self._cov_estimator is not None:
            return self._cov_estimator.estimate(residuals)

        # Fallback to sample covariance
        return SampleCovarianceEstimator().estimate(residuals)

    def _apply_regularization(self, sigma: np.ndarray) -> np.ndarray:
        """Apply ridge regularization to covariance matrix.

        Adds λI to the covariance matrix for numerical stability.
        With adaptive regularization, λ is adjusted based on condition number.

        Args:
            sigma: Covariance matrix.

        Returns:
            Regularized covariance matrix.
        """
        if self.regularization <= 0:
            return sigma

        reg_param = self.regularization

        # Adaptive regularization based on condition number
        if self.adaptive_regularization:
            cond_num = np.linalg.cond(sigma)
            if cond_num > 1e8:
                # Increase regularization for ill-conditioned matrices
                reg_param = self.regularization * (cond_num / 1e8)
                logger.info("Adaptive regularization: λ=%.2e (cond=%.2e)", reg_param, cond_num)

        n = sigma.shape[0]
        return sigma + reg_param * np.eye(n)

    def _stable_inversion(self, matrix: np.ndarray) -> np.ndarray:
        """Compute numerically stable matrix inversion.

        Attempts Cholesky decomposition first (fastest for positive definite),
        then falls back to LU decomposition, and finally pseudo-inverse.

        Args:
            matrix: Square matrix to invert.

        Returns:
            Inverted matrix.
        """
        try:
            # Try Cholesky (fastest for positive definite)
            cho_factor = linalg.cholesky(matrix, lower=True)
            inv_matrix = linalg.cho_solve(
                (cho_factor, True), np.eye(matrix.shape[0])
            )
            logger.debug("Used Cholesky decomposition for inversion")
            return inv_matrix

        except linalg.LinAlgError:
            pass

        try:
            # Fallback to LU decomposition
            inv_matrix = linalg.inv(matrix)
            logger.debug("Used LU decomposition for inversion")
            return inv_matrix

        except linalg.LinAlgError:
            # Ultimate fallback to pseudo-inverse
            logger.warning("Matrix singular, using pseudo-inverse")
            return linalg.pinv(matrix)

    def _validate_inputs(
        self,
        base_forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> list[str]:
        """Validate reconciliation inputs.

        Args:
            base_forecasts: Base forecasts DataFrame.
            hierarchy: Hierarchy definition.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        node_names = hierarchy.get_node_names_sorted()

        # Check all nodes have forecasts
        missing_nodes = set(node_names) - set(base_forecasts.columns)
        if missing_nodes:
            errors.append(f"Missing forecasts for nodes: {missing_nodes}")

        # Check for NaN values
        nan_cols = base_forecasts.columns[base_forecasts.isna().any()].tolist()
        if nan_cols:
            errors.append(f"NaN values in columns: {nan_cols}")

        return errors

    def _calculate_coherence_error(
        self,
        forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> float:
        """Calculate coherence error for forecasts.

        Coherence error measures how well the forecasts satisfy the
        hierarchical aggregation constraints.

        Args:
            forecasts: Forecasts DataFrame.
            hierarchy: Hierarchy definition.

        Returns:
            Mean absolute coherence error.
        """
        try:
            node_order = hierarchy.get_node_names_sorted()
            y_matrix = forecasts[node_order].to_numpy().T

            if hierarchy.aggregation_matrix is None:
                hierarchy.build_aggregation_matrix()
            s_matrix = hierarchy.aggregation_matrix

            # Check S @ bottom_level == full (simplified check)
            n_bottom = s_matrix.shape[1]
            y_bottom = y_matrix[-n_bottom:, :]
            y_reconstructed = s_matrix @ y_bottom
            return float(np.mean(np.abs(y_matrix - y_reconstructed)))
        except Exception as e:
            logger.warning("Could not calculate coherence error: %s", e)
            return float("inf")

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information from last reconciliation.

        Returns:
            Dictionary with diagnostic metrics including covariance method,
            condition number, regularization applied, and coherence metrics.
        """
        return self._last_diagnostics.copy()

    def compare_with_mint(
        self,
        base_forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        residuals: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compare OLS reconciliation with MinT.

        Runs both OLS and MinT reconciliation on the same data and
        compares their performance.

        Args:
            base_forecasts: Base forecasts DataFrame.
            actuals: Actual values for error calculation.
            hierarchy: Hierarchy definition.
            residuals: Optional historical residuals for covariance estimation.

        Returns:
            Dictionary with comparison metrics.
        """
        # Import here to avoid circular import
        from src.reconciliation.mint_reconciler import (  # noqa: PLC0415
            MintReconciler,
        )

        # OLS reconciliation
        ols_result = self.reconcile(base_forecasts, hierarchy, residuals=residuals)

        # MinT reconciliation
        mint = MintReconciler(ReconciliationConfig(method="mint"))
        if residuals is not None:
            # Create historical forecasts/actuals for fitting
            # For simplicity, use residuals directly
            mint.fitted = True
            mint.covariance_matrix = self._estimate_covariance(residuals)

        mint_result = mint.reconcile(base_forecasts, hierarchy)

        # Calculate errors
        node_order = hierarchy.get_node_names_sorted()
        common_cols = [c for c in node_order if c in actuals.columns]

        ols_mae = float(
            np.mean(np.abs(ols_result.reconciled_forecasts[common_cols] - actuals[common_cols]))
        )
        mint_mae = float(
            np.mean(np.abs(mint_result.reconciled_forecasts[common_cols] - actuals[common_cols]))
        )

        improvement = (mint_mae - ols_mae) / mint_mae * 100 if mint_mae > 0 else 0

        return {
            "ols_mae": ols_mae,
            "mint_mae": mint_mae,
            "improvement_pct": improvement,
            "ols_coherence": ols_result.metadata.get("coherence_after", 0),
            "mint_coherence": mint_result.metadata.get("coherence_after", 0),
        }

    def save(self, path: str | Path) -> None:
        """Save reconciler state to file.

        Args:
            path: Path to save file (JSON format).
        """
        state = {
            "covariance_method": self.covariance_method,
            "regularization": self.regularization,
            "n_factors": self.n_factors,
            "adaptive_regularization": self.adaptive_regularization,
            "fitted": self.fitted,
            "covariance_matrix": (
                self.covariance_matrix.tolist() if self.covariance_matrix is not None else None
            ),
        }

        path = Path(path)
        with path.open("w") as f:
            json.dump(state, f, indent=2)

        logger.info("Saved OLS reconciler to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> OLSReconciler:
        """Load reconciler state from file.

        Args:
            path: Path to saved state file.

        Returns:
            Loaded OLSReconciler instance.
        """
        path = Path(path)
        with path.open() as f:
            state = json.load(f)

        config = ReconciliationConfig(
            method="ols",
            extra_fields={
                "covariance_method": state["covariance_method"],
                "regularization": state["regularization"],
                "n_factors": state["n_factors"],
                "adaptive_regularization": state["adaptive_regularization"],
            },
        )

        reconciler = cls(config=config)
        reconciler.fitted = state["fitted"]

        if state["covariance_matrix"] is not None:
            reconciler.covariance_matrix = np.array(state["covariance_matrix"])

        logger.info("Loaded OLS reconciler from %s", path)
        return reconciler
