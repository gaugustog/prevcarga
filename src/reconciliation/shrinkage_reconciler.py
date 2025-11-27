"""Shrinkage-based hierarchical forecast reconciliation.

This module implements the ShrinkageReconciler class that uses James-Stein shrinkage
for robust covariance estimation in hierarchical forecast reconciliation. It is
particularly effective when historical data is limited (<100 samples).

The shrinkage formula combines sample covariance with a structured target:
    Σ_shrunk = (1-λ)Σ_sample + λΣ_target

where λ ∈ [0,1] is the shrinkage intensity that balances:
    - λ=0: Full sample covariance (unstable with small samples)
    - λ=1: Full shrinkage to target (stable but biased)
    - λ optimal: Ledoit-Wolf formula minimizes MSE

Shrinkage Targets:
    - IDENTITY: Σ_target = σ²I (independence assumption)
    - DIAGONAL: Σ_target = diag(Σ_sample) (no correlations)
    - FACTOR: Σ_target from factor model
    - CONSTANT_CORRELATION: Common correlation ρ

Lambda Selection Methods:
    - ledoit_wolf: Analytical formula (fast, recommended)
    - cross_validation: Data-driven K-fold CV (slower, flexible)
    - fixed: User-specified constant λ

Example:
    ```python
    from src.reconciliation import ShrinkageReconciler, ReconciliationConfig
    import pandas as pd

    # Configure shrinkage reconciler
    config = ReconciliationConfig(
        method="shrinkage",
        covariance_method="shrinkage",
        extra_fields={
            "shrinkage_target": "diagonal",
            "auto_lambda": True,
            "lambda_method": "ledoit_wolf"
        }
    )

    # Create reconciler
    reconciler = ShrinkageReconciler(config=config)

    # Fit with limited historical data
    reconciler.fit(historical_forecasts, actuals, hierarchy)
    print(f"Optimal λ: {reconciler.computed_lambda:.3f}")

    # Reconcile forecasts
    result = reconciler.reconcile(base_forecasts, hierarchy)
    ```

References:
    Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional
    covariance matrices." Journal of Multivariate Analysis, 88(2), 365-411.

    Schäfer, J., & Strimmer, K. (2005). "A shrinkage approach to large-scale covariance
    matrix estimation and implications for functional genomics." Statistical Applications
    in Genetics and Molecular Biology, 4(1).
"""

from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import linalg

from src.reconciliation.base_reconciler import BaseReconciler
from src.reconciliation.data_structures import ReconciliationConfig, ReconciliationResult
from src.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type aliases
ShrinkageTargetType = Literal["identity", "diagonal", "factor", "constant_correlation"]
LambdaMethod = Literal["ledoit_wolf", "cross_validation", "fixed"]


class ShrinkageTarget(Enum):
    """Enumeration of shrinkage target structures."""

    IDENTITY = "identity"
    DIAGONAL = "diagonal"
    FACTOR = "factor"
    CONSTANT_CORRELATION = "constant_correlation"


class ShrinkageReconciler(BaseReconciler):
    """Shrinkage-based hierarchical forecast reconciliation.

    This reconciler uses James-Stein shrinkage to robustly estimate the error
    covariance matrix when historical data is limited. It shrinks the unstable
    sample covariance towards a structured target, improving stability and
    reconciliation quality.

    The reconciliation uses MinT formula with shrunk covariance:
        G = S(S'W_shrunk^{-1}S)^{-1}S'W_shrunk^{-1}

    where W_shrunk^{-1} = Σ_shrunk is the shrinkage-based covariance estimate.

    Attributes:
        config: ReconciliationConfig with shrinkage parameters.
        shrinkage_target: Target structure for shrinkage.
        auto_lambda: Whether to automatically select λ.
        lambda_method: Method for λ selection.
        fixed_lambda: Fixed λ value if auto_lambda=False.
        computed_lambda: Optimal λ computed during fit().
        covariance_matrix: Shrunk covariance matrix.
        fitted: Whether reconciler has been fitted.

    Example:
        ```python
        from src.reconciliation import ShrinkageReconciler, ReconciliationConfig

        # Create reconciler
        config = ReconciliationConfig(
            method="shrinkage",
            extra_fields={
                "shrinkage_target": "diagonal",
                "auto_lambda": True,
                "lambda_method": "ledoit_wolf",
                "min_lambda": 0.0,
                "max_lambda": 1.0
            }
        )
        reconciler = ShrinkageReconciler(config=config)

        # Fit and reconcile
        reconciler.fit(historical_forecasts, actuals, hierarchy)
        result = reconciler.reconcile(base_forecasts, hierarchy)

        print(f"Shrinkage intensity: {reconciler.computed_lambda:.3f}")
        print(f"Coherence: {result.metadata['coherence_after']:.6e}")
        ```
    """

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        """Initialize shrinkage reconciler.

        Args:
            config: ReconciliationConfig with shrinkage parameters.
                Pass extra parameters directly:
                    - shrinkage_target: Target structure (default: "diagonal")
                    - auto_lambda: Auto-select λ (default: True)
                    - lambda_method: Method for λ (default: "ledoit_wolf")
                    - fixed_lambda: Fixed λ if auto_lambda=False (default: 0.5)
                    - min_lambda: Minimum λ (default: 0.0)
                    - max_lambda: Maximum λ (default: 1.0)
        """
        super().__init__(config)

        # Extract shrinkage parameters from config (using get_extra_field for model_extra)
        target_str: ShrinkageTargetType = self.config.get_extra_field(
            "shrinkage_target", "diagonal"
        )
        self.shrinkage_target = ShrinkageTarget(target_str)

        self.auto_lambda: bool = bool(self.config.get_extra_field("auto_lambda", True))
        self.lambda_method: LambdaMethod = self.config.get_extra_field(
            "lambda_method", "ledoit_wolf"
        )
        self.fixed_lambda: float = float(self.config.get_extra_field("fixed_lambda", 0.5))

        # Lambda bounds
        self.min_lambda: float = float(self.config.get_extra_field("min_lambda", 0.0))
        self.max_lambda: float = float(self.config.get_extra_field("max_lambda", 1.0))

        # Validate lambda bounds
        if not 0.0 <= self.min_lambda <= 1.0:
            msg = f"min_lambda must be in [0,1], got {self.min_lambda}"
            raise ValueError(msg)
        if not 0.0 <= self.max_lambda <= 1.0:
            msg = f"max_lambda must be in [0,1], got {self.max_lambda}"
            raise ValueError(msg)
        if self.min_lambda > self.max_lambda:
            msg = f"min_lambda ({self.min_lambda}) > max_lambda ({self.max_lambda})"
            raise ValueError(msg)

        # State variables
        self.computed_lambda: float | None = None
        self.covariance_matrix: np.ndarray | None = None
        self.sample_covariance: np.ndarray | None = None
        self.target_covariance: np.ndarray | None = None

        logger.info(
            "Initialized ShrinkageReconciler (target=%s, auto_lambda=%s, method=%s)",
            self.shrinkage_target.value,
            self.auto_lambda,
            self.lambda_method,
        )

    def fit(
        self,
        historical_forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> None:
        """Estimate shrinkage-based covariance matrix from historical data.

        Computes residuals, estimates sample covariance, constructs shrinkage
        target, selects optimal λ, and applies shrinkage.

        Args:
            historical_forecasts: Historical base forecasts (n_periods × n_nodes).
            actuals: Actual values (n_periods × n_nodes).
            hierarchy: HierarchyDefinition with structure.

        Raises:
            ValueError: If inputs invalid or insufficient data.

        Example:
            ```python
            # Limited historical data
            historical_forecasts = pd.DataFrame({
                "BR_National": [1000, 1100, 1050, 1080],  # Only 4 periods
                "SE": [400, 440, 420, 430],
                # ...
            })

            reconciler.fit(historical_forecasts, actuals, hierarchy)
            print(f"Fitted with λ={reconciler.computed_lambda:.3f}")
            ```
        """
        logger.info(
            "Fitting ShrinkageReconciler with %d historical periods", len(historical_forecasts)
        )

        # Validate inputs
        self.validate_forecasts(historical_forecasts, hierarchy)
        self.validate_forecasts(actuals, hierarchy)

        if historical_forecasts.shape != actuals.shape:
            msg = (
                f"Shape mismatch: historical_forecasts {historical_forecasts.shape} "
                f"vs actuals {actuals.shape}"
            )
            raise ValueError(msg)

        if len(historical_forecasts) < 2:
            msg = "Need at least 2 historical periods to estimate covariance"
            raise ValueError(msg)

        # Get nodes in sorted order
        all_nodes = hierarchy.get_node_names_sorted()
        n_nodes = len(all_nodes)
        n_samples = len(historical_forecasts)

        # Compute residuals
        residuals = historical_forecasts[all_nodes] - actuals[all_nodes]
        residuals_array = residuals.values  # (n_samples, n_nodes)

        logger.debug(
            "Computed residuals: n_samples=%d, n_nodes=%d, condition: %.1f sample/node",
            n_samples,
            n_nodes,
            n_samples / n_nodes,
        )

        # Estimate sample covariance
        self.sample_covariance = self._compute_sample_covariance(residuals_array)

        # Construct shrinkage target
        self.target_covariance = self._construct_target(
            self.sample_covariance, residuals_array
        )

        # Determine shrinkage intensity λ
        if self.auto_lambda:
            if self.lambda_method == "ledoit_wolf":
                self.computed_lambda = self._ledoit_wolf_lambda(
                    self.sample_covariance, self.target_covariance, residuals_array
                )
            elif self.lambda_method == "cross_validation":
                self.computed_lambda = self._cv_lambda(
                    residuals_array, self.sample_covariance, self.target_covariance
                )
            else:  # fixed
                self.computed_lambda = self.fixed_lambda
        else:
            self.computed_lambda = self.fixed_lambda

        # Clip to bounds
        self.computed_lambda = np.clip(
            self.computed_lambda, self.min_lambda, self.max_lambda
        )

        logger.info("Shrinkage intensity λ = %.4f", self.computed_lambda)

        # Apply shrinkage: Σ_shrunk = (1-λ)Σ_sample + λΣ_target
        self.covariance_matrix = (
            (1 - self.computed_lambda) * self.sample_covariance
            + self.computed_lambda * self.target_covariance
        )

        # Ensure positive definiteness
        self.covariance_matrix = self._ensure_positive_definite(self.covariance_matrix)

        # Mark as fitted
        self.fitted = True

        # Log diagnostics
        logger.info(
            "Fitted shrinkage covariance: shape=%s, trace=%.2e, det=%.2e, condition=%.2e",
            self.covariance_matrix.shape,
            np.trace(self.covariance_matrix),
            np.linalg.det(self.covariance_matrix),
            np.linalg.cond(self.covariance_matrix),
        )

    def reconcile(
        self,
        base_forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> ReconciliationResult:
        """Reconcile base forecasts using shrinkage-based MinT.

        Applies MinT reconciliation with the shrinkage-based covariance estimate.

        Args:
            base_forecasts: Base forecasts to reconcile (n_timesteps × n_nodes).
            hierarchy: HierarchyDefinition with aggregation matrix.

        Returns:
            ReconciliationResult with reconciled forecasts and metadata.

        Raises:
            ValueError: If reconciler not fitted or inputs invalid.

        Example:
            ```python
            base_forecasts = pd.DataFrame({
                "BR_National": [1000, 1100],
                "SE": [400, 440],
                # ...
            })

            result = reconciler.reconcile(base_forecasts, hierarchy)
            print(f"Coherence: {result.metadata['coherence_after']:.6e}")
            print(f"Used λ: {result.metadata['shrinkage_lambda']:.3f}")
            ```
        """
        logger.info("Reconciling %d timesteps with ShrinkageReconciler", len(base_forecasts))

        # Validate inputs
        self.validate_forecasts(base_forecasts, hierarchy)

        if not self.fitted:
            msg = (
                "ShrinkageReconciler not fitted. "
                "Call fit(historical_forecasts, actuals, hierarchy) first."
            )
            raise ValueError(msg)

        # Compute coherence before reconciliation
        coherence_before = self.check_coherence(base_forecasts, hierarchy)

        # Compute reconciliation matrix G
        G = self._compute_reconciliation_matrix(hierarchy)

        # Apply reconciliation
        all_nodes = hierarchy.get_node_names_sorted()
        y_base = base_forecasts[all_nodes].values.T  # (n_total, T)
        y_reconciled = (G @ y_base).T  # (T, n_total)

        # Create DataFrame
        reconciled_df = pd.DataFrame(
            y_reconciled,
            index=base_forecasts.index,
            columns=all_nodes,
        )

        # Apply non-negativity constraint if configured
        if self.config.non_negative:
            reconciled_df = self.apply_non_negativity(reconciled_df)
            logger.debug("Applied non-negativity constraint")

        # Compute coherence after reconciliation
        coherence_after = self.check_coherence(reconciled_df, hierarchy)

        # Validate coherence if configured
        if self.config.check_coherence:
            if coherence_after > self.config.tolerance:
                logger.warning(
                    "Coherence check failed: %.6e > %.6e",
                    coherence_after,
                    self.config.tolerance,
                )
            else:
                logger.info(
                    "Coherence validated: %.6e <= %.6e", coherence_after, self.config.tolerance
                )

        # Create result with metadata
        metadata = {
            "method": "shrinkage",
            "shrinkage_target": self.shrinkage_target.value,
            "shrinkage_lambda": float(self.computed_lambda),
            "lambda_method": self.lambda_method,
            "coherence_before": float(coherence_before),
            "coherence_after": float(coherence_after),
            "fitted": self.fitted,
            "n_timesteps": len(base_forecasts),
            "n_nodes": len(all_nodes),
        }

        result = ReconciliationResult(
            reconciled_forecasts=reconciled_df,
            metadata=metadata,
            base_forecasts=base_forecasts.copy(),
            residual_covariance=self.covariance_matrix.copy(),
            reconciliation_matrix=G,
        )

        logger.info(
            "Shrinkage reconciliation complete: λ=%.4f, coherence %.6e → %.6e",
            self.computed_lambda,
            coherence_before,
            coherence_after,
        )

        return result

    def _compute_reconciliation_matrix(
        self,
        hierarchy: HierarchyDefinition,
    ) -> np.ndarray:
        """Compute MinT reconciliation matrix with shrunk covariance.

        Formula: G = S(S'Σ^{-1}S)^{-1}S'Σ^{-1}

        Args:
            hierarchy: HierarchyDefinition with aggregation matrix S.

        Returns:
            Reconciliation matrix G (n_total × n_total).
        """
        S = hierarchy.aggregation_matrix

        if S is None:
            msg = "Hierarchy aggregation matrix is None"
            raise ValueError(msg)

        # Compute Σ^{-1} using pseudo-inverse for stability
        Sigma_inv = linalg.pinv(self.covariance_matrix, rcond=1e-10)

        # Compute S'Σ^{-1}S
        STS = S.T @ Sigma_inv @ S

        # Invert using pseudo-inverse
        STS_inv = linalg.pinv(STS, rcond=1e-10)

        # Compute G = S(S'Σ^{-1}S)^{-1}S'Σ^{-1}
        G = S @ STS_inv @ S.T @ Sigma_inv

        logger.debug(
            "Computed reconciliation matrix: shape=%s, rank=%d, condition=%.2e",
            G.shape,
            np.linalg.matrix_rank(G),
            np.linalg.cond(G),
        )

        return G

    def _construct_target(
        self, sample_cov: np.ndarray, residuals: np.ndarray
    ) -> np.ndarray:
        """Construct shrinkage target covariance matrix.

        Args:
            sample_cov: Sample covariance matrix (n_nodes × n_nodes).
            residuals: Residual array (n_samples × n_nodes).

        Returns:
            Target covariance matrix (n_nodes × n_nodes).
        """
        n_nodes = sample_cov.shape[0]

        if self.shrinkage_target == ShrinkageTarget.IDENTITY:
            # Scaled identity matrix
            scale = np.trace(sample_cov) / n_nodes
            target = scale * np.eye(n_nodes)
            logger.debug("Constructed IDENTITY target (scale=%.2e)", scale)

        elif self.shrinkage_target == ShrinkageTarget.DIAGONAL:
            # Diagonal of sample covariance
            target = np.diag(np.diag(sample_cov))
            logger.debug("Constructed DIAGONAL target")

        elif self.shrinkage_target == ShrinkageTarget.CONSTANT_CORRELATION:
            # Constant correlation model
            variances = np.diag(sample_cov)
            std_devs = np.sqrt(variances)

            # Compute average correlation
            corr_matrix = sample_cov / np.outer(std_devs, std_devs)
            avg_corr = (np.sum(corr_matrix) - n_nodes) / (n_nodes * (n_nodes - 1))

            # Construct target with constant correlation
            target = avg_corr * np.outer(std_devs, std_devs)
            np.fill_diagonal(target, variances)

            logger.debug(
                "Constructed CONSTANT_CORRELATION target (avg_corr=%.4f)", avg_corr
            )

        elif self.shrinkage_target == ShrinkageTarget.FACTOR:
            # Simple factor model (single factor)
            target = self._factor_target(residuals, sample_cov)
            logger.debug("Constructed FACTOR target")

        else:
            msg = f"Unknown shrinkage target: {self.shrinkage_target}"
            raise ValueError(msg)

        return target

    def _factor_target(
        self, residuals: np.ndarray, sample_cov: np.ndarray
    ) -> np.ndarray:
        """Construct factor model target covariance.

        Uses single-factor model: Σ = ββ' + D
        where β are factor loadings and D is diagonal specific variance.

        Args:
            residuals: Residual array (n_samples × n_nodes).
            sample_cov: Sample covariance matrix.

        Returns:
            Factor model target covariance.
        """
        n_nodes = residuals.shape[1]

        # Compute first principal component as factor
        centered = residuals - residuals.mean(axis=0)
        cov_est = centered.T @ centered / (residuals.shape[0] - 1)

        # Get first eigenvector as factor loadings
        eigenvalues, eigenvectors = np.linalg.eigh(cov_est)
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Factor loadings (first eigenvector scaled by sqrt(eigenvalue))
        factor_loadings = eigenvectors[:, 0] * np.sqrt(eigenvalues[0])

        # Construct ββ'
        factor_part = np.outer(factor_loadings, factor_loadings)

        # Specific variance (diagonal)
        specific_var = np.diag(sample_cov) - np.diag(factor_part)
        specific_var = np.maximum(specific_var, 1e-6)  # Ensure positive

        # Combine: Σ = ββ' + D
        target = factor_part + np.diag(specific_var)

        return target

    def _ledoit_wolf_lambda(
        self,
        sample_cov: np.ndarray,
        target_cov: np.ndarray,
        residuals: np.ndarray,
    ) -> float:
        """Compute optimal shrinkage intensity using Ledoit-Wolf formula.

        The optimal λ minimizes the expected MSE between the estimator and
        the true covariance matrix.

        Formula:
            λ* = φ / δ
        where:
            δ = ||Σ_sample - Σ_target||²_F (distance to target)
            φ = variance of sample covariance estimator

        Args:
            sample_cov: Sample covariance matrix (n_nodes × n_nodes).
            target_cov: Target covariance matrix (n_nodes × n_nodes).
            residuals: Residual array (n_samples × n_nodes).

        Returns:
            Optimal shrinkage intensity λ ∈ [0, 1].

        References:
            Ledoit, O., & Wolf, M. (2004). Journal of Multivariate Analysis.
        """
        n_samples, n_nodes = residuals.shape

        # Compute δ: squared Frobenius distance to target
        delta = np.linalg.norm(sample_cov - target_cov, "fro") ** 2

        if delta < 1e-10:
            # Sample covariance already at target
            logger.debug("Sample covariance ≈ target, λ=0")
            return 0.0

        # Compute φ: variance of sample covariance estimator
        # φ = (1/n²) Σ_t ||r_t r_t' - Σ_sample||²_F
        centered = residuals - residuals.mean(axis=0)
        phi = 0.0

        for t in range(n_samples):
            r_t = centered[t:t + 1, :].T  # (n_nodes, 1)
            outer_prod = r_t @ r_t.T  # (n_nodes, n_nodes)
            phi += np.linalg.norm(outer_prod - sample_cov, "fro") ** 2

        phi /= n_samples**2

        # Optimal λ = min(φ/δ, 1)
        lambda_opt = phi / (delta + 1e-10)
        lambda_opt = min(lambda_opt, 1.0)
        lambda_opt = max(lambda_opt, 0.0)

        logger.debug(
            "Ledoit-Wolf: δ=%.2e, φ=%.2e, λ*=%.4f", delta, phi, lambda_opt
        )

        return float(lambda_opt)

    def _cv_lambda(
        self,
        residuals: np.ndarray,
        sample_cov: np.ndarray,
        target_cov: np.ndarray,
    ) -> float:
        """Select optimal shrinkage intensity via cross-validation.

        Uses time-series split to avoid look-ahead bias.

        Args:
            residuals: Residual array (n_samples × n_nodes).
            sample_cov: Sample covariance from full data.
            target_cov: Target covariance.

        Returns:
            Optimal shrinkage intensity λ ∈ [0, 1].
        """
        n_samples = residuals.shape[0]

        if n_samples < 10:
            logger.warning("Too few samples for CV (%d), using λ=0.5", n_samples)
            return 0.5

        # Try λ values
        lambda_grid = np.linspace(self.min_lambda, self.max_lambda, 11)

        # Simple holdout: train on first 70%, validate on last 30%
        split_idx = int(0.7 * n_samples)
        train_residuals = residuals[:split_idx]
        val_residuals = residuals[split_idx:]

        if len(train_residuals) < 2 or len(val_residuals) < 2:
            logger.warning("Insufficient data for CV split, using λ=0.5")
            return 0.5

        # Compute train covariances
        train_sample_cov = self._compute_sample_covariance(train_residuals)
        train_target_cov = self._construct_target(train_sample_cov, train_residuals)

        # Compute validation covariance (ground truth proxy)
        val_cov = self._compute_sample_covariance(val_residuals)

        # Evaluate each λ
        best_lambda = 0.5
        best_loss = float("inf")

        for lam in lambda_grid:
            # Shrunk estimator
            shrunk_cov = (1 - lam) * train_sample_cov + lam * train_target_cov

            # Loss: Frobenius distance to validation covariance
            loss = np.linalg.norm(shrunk_cov - val_cov, "fro")

            if loss < best_loss:
                best_loss = loss
                best_lambda = lam

        logger.debug("CV selected λ=%.4f (loss=%.2e)", best_lambda, best_loss)

        return float(best_lambda)

    def _compute_sample_covariance(self, residuals: np.ndarray) -> np.ndarray:
        """Compute sample covariance with bias correction.

        Args:
            residuals: Residual array (n_samples × n_nodes).

        Returns:
            Sample covariance matrix (n_nodes × n_nodes).
        """
        n_samples = residuals.shape[0]
        centered = residuals - residuals.mean(axis=0)
        cov = (centered.T @ centered) / (n_samples - 1)
        return cov

    def _ensure_positive_definite(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is positive definite.

        Adds small regularization to diagonal if needed.

        Args:
            cov_matrix: Covariance matrix.

        Returns:
            Positive definite covariance matrix.
        """
        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        min_eigenvalue = eigenvalues.min()

        if min_eigenvalue < 1e-8:
            # Add regularization
            reg = max(1e-8 - min_eigenvalue, 1e-8)
            cov_matrix = cov_matrix + reg * np.eye(cov_matrix.shape[0])
            logger.debug(
                "Added regularization %.2e to ensure positive definiteness", reg
            )

        return cov_matrix

    def get_diagnostics(self) -> dict[str, Any]:
        """Get shrinkage diagnostics.

        Returns:
            Dictionary with diagnostic information.

        Example:
            ```python
            diag = reconciler.get_diagnostics()
            print(f"Lambda: {diag['lambda']}")
            print(f"Condition number improvement: {diag['condition_improvement']:.1f}x")
            ```
        """
        if not self.fitted:
            return {"fitted": False}

        sample_cond = np.linalg.cond(self.sample_covariance)
        shrunk_cond = np.linalg.cond(self.covariance_matrix)

        return {
            "fitted": True,
            "shrinkage_target": self.shrinkage_target.value,
            "lambda": self.computed_lambda,
            "lambda_method": self.lambda_method,
            "sample_covariance_condition": float(sample_cond),
            "shrunk_covariance_condition": float(shrunk_cond),
            "condition_improvement": float(sample_cond / shrunk_cond),
            "sample_covariance_trace": float(np.trace(self.sample_covariance)),
            "shrunk_covariance_trace": float(np.trace(self.covariance_matrix)),
            "sample_covariance_det": float(np.linalg.det(self.sample_covariance)),
            "shrunk_covariance_det": float(np.linalg.det(self.covariance_matrix)),
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        if self.fitted:
            return (
                f"ShrinkageReconciler("
                f"target={self.shrinkage_target.value!r}, "
                f"λ={self.computed_lambda:.4f}, "
                f"fitted=True)"
            )
        else:
            return (
                f"ShrinkageReconciler("
                f"target={self.shrinkage_target.value!r}, "
                f"fitted=False)"
            )
