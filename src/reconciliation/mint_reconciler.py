"""MinT (Minimum Trace) reconciliation method.

This module implements the MinT (Minimum Trace) reconciliation method, which is
the state-of-the-art approach for hierarchical forecast reconciliation. MinT
optimally reconciles forecasts by minimizing the trace of the reconciled forecast
error covariance matrix.

The MinT reconciliation formula:
    G = S(S'WS)^{-1}S'W

where:
    - S = aggregation/summing matrix from hierarchy
    - W = Σ^{-1} (inverse of error covariance matrix)
    - G = reconciliation matrix
    - reconciled = G @ base_forecasts

The method requires estimating the error covariance matrix Σ from historical
residuals. Multiple estimation methods are supported:
    - sample: Sample covariance from residuals
    - shrinkage: Ledoit-Wolf shrinkage estimator (robust for small samples)
    - diagonal: Diagonal covariance (independence assumption)
    - identity: Identity matrix (fallback to OLS reconciliation)
    - structural: Variance proportional to forecast level

Example:
    ```python
    from src.reconciliation import MintReconciler, ReconciliationConfig
    import pandas as pd

    # Configure MinT reconciler
    config = ReconciliationConfig(
        method="mint",
        covariance_method="shrinkage",
        check_coherence=True,
        non_negative=True
    )

    # Create reconciler
    reconciler = MintReconciler(config=config)

    # Fit to historical data
    reconciler.fit(historical_forecasts, actuals, hierarchy)

    # Reconcile new forecasts
    result = reconciler.reconcile(base_forecasts, hierarchy)

    # Check results
    print(f"Coherence after: {result.metadata['coherence_after']:.6e}")
    assert result.validate_coherence(hierarchy, tolerance=1e-6)
    ```

References:
    Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
    "Optimal forecast reconciliation for hierarchical and grouped time series
    through trace minimization." Journal of the American Statistical Association,
    114(526), 804-819.
"""

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.reconciliation.base_reconciler import BaseReconciler
from src.reconciliation.data_structures import ReconciliationConfig, ReconciliationResult
from src.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for covariance estimation methods
CovarianceMethod = Literal["sample", "shrinkage", "diagonal", "identity", "structural"]


class MintReconciler(BaseReconciler):
    """MinT (Minimum Trace) reconciliation for hierarchical forecasts.

    MinT is the state-of-the-art reconciliation method that minimizes the trace
    of the reconciled forecast error covariance matrix. It accounts for
    heteroskedasticity and correlation in forecast errors.

    The reconciliation matrix is computed as:
        G = S(S'Σ^{-1}S)^{-1}S'Σ^{-1}

    where Σ is the error covariance matrix estimated from historical residuals.

    Attributes:
        config: ReconciliationConfig with method="mint" and covariance_method.
        covariance_matrix: Fitted error covariance matrix Σ (n_nodes × n_nodes).
        covariance_method: Method for estimating covariance matrix.
        fitted: Whether the reconciler has been fitted with historical data.
        regularization: Regularization parameter for numerical stability.

    Example:
        ```python
        from src.reconciliation import MintReconciler, ReconciliationConfig

        # Configure
        config = ReconciliationConfig(
            method="mint",
            covariance_method="shrinkage",
            regularization=1e-8,
            check_coherence=True
        )

        # Create and fit
        reconciler = MintReconciler(config=config)
        reconciler.fit(historical_forecasts, actuals, hierarchy)

        # Reconcile
        result = reconciler.reconcile(base_forecasts, hierarchy)
        ```
    """

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        """Initialize MinT reconciler.

        Args:
            config: Reconciliation configuration. Must have method="mint".
                Can include extra field "covariance_method" for covariance
                estimation method (default: "shrinkage").
        """
        super().__init__(config)

        # Get covariance method from config
        self.covariance_method: CovarianceMethod = self.config.get_extra_field(
            "covariance_method", "shrinkage"
        )

        # Get regularization parameter
        self.regularization: float = self.config.get_extra_field("regularization", 1e-8)

        # Initialize covariance matrix (fitted during fit())
        self.covariance_matrix: np.ndarray | None = None

        logger.info(
            "Initialized MinT reconciler (covariance_method=%s, regularization=%.2e)",
            self.covariance_method,
            self.regularization,
        )

    def fit(
        self,
        historical_forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> None:
        """Estimate error covariance matrix from historical data.

        This method computes residuals from historical forecasts and actuals,
        then estimates the error covariance matrix using the configured method.

        Args:
            historical_forecasts: Historical base forecasts (n_periods × n_nodes).
                Must contain all nodes in the hierarchy.
            actuals: Actual values (n_periods × n_nodes). Must have same shape
                as historical_forecasts.
            hierarchy: HierarchyDefinition with structure and node ordering.

        Raises:
            ValueError: If inputs are invalid or have mismatched shapes.

        Example:
            ```python
            # Historical data
            historical_forecasts = pd.DataFrame({
                "BR_National": [1000, 1100, 1050],
                "SE": [400, 440, 420],
                # ... more nodes
            })
            actuals = pd.DataFrame({
                "BR_National": [1010, 1090, 1060],
                "SE": [405, 435, 425],
                # ... more nodes
            })

            # Fit
            reconciler.fit(historical_forecasts, actuals, hierarchy)
            print(f"Fitted covariance shape: {reconciler.covariance_matrix.shape}")
            ```
        """
        logger.info("Fitting MinT reconciler with %d historical periods", len(historical_forecasts))

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

        # Compute residuals: e = forecasts - actuals
        residuals = historical_forecasts[all_nodes] - actuals[all_nodes]
        logger.debug("Computed residuals with shape %s", residuals.shape)

        # Estimate covariance matrix
        self.covariance_matrix = self._estimate_covariance(
            residuals.values, method=self.covariance_method
        )

        # Ensure positive definiteness
        self.covariance_matrix = self._ensure_positive_definite(self.covariance_matrix)

        # Mark as fitted
        self.fitted = True

        # Log diagnostics
        cov_trace = np.trace(self.covariance_matrix)
        cov_det = np.linalg.det(self.covariance_matrix)
        logger.info(
            "Fitted covariance matrix: shape=%s, trace=%.2e, det=%.2e",
            self.covariance_matrix.shape,
            cov_trace,
            cov_det,
        )

    def reconcile(
        self,
        base_forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> ReconciliationResult:
        """Reconcile base forecasts using MinT method.

        Applies the MinT reconciliation formula to ensure forecasts satisfy
        hierarchical aggregation constraints while minimizing the trace of
        the error covariance matrix.

        Args:
            base_forecasts: Base forecasts to reconcile (n_timesteps × n_nodes).
                Must contain all nodes in the hierarchy.
            hierarchy: HierarchyDefinition with aggregation matrix.

        Returns:
            ReconciliationResult with reconciled forecasts and metadata.

        Raises:
            ValueError: If reconciler hasn't been fitted or inputs are invalid.

        Example:
            ```python
            # Base forecasts (potentially incoherent)
            base_forecasts = pd.DataFrame({
                "BR_National": [1000, 1100],
                "SE": [400, 440],
                # ... more nodes
            })

            # Reconcile
            result = reconciler.reconcile(base_forecasts, hierarchy)

            # Access results
            print(result.reconciled_forecasts)
            print(f"Coherence: {result.metadata['coherence_after']:.6e}")
            ```
        """
        logger.info("Reconciling %d timesteps with MinT method", len(base_forecasts))

        # Validate inputs
        self.validate_forecasts(base_forecasts, hierarchy)

        if not self.fitted:
            logger.warning("MinT reconciler not fitted. Falling back to identity covariance (OLS).")
            # Use identity covariance (equivalent to OLS)
            all_nodes = hierarchy.get_node_names_sorted()
            n_nodes = len(all_nodes)
            self.covariance_matrix = np.eye(n_nodes)

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
            "method": "mint",
            "covariance_method": self.covariance_method,
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
            residual_covariance=(
                self.covariance_matrix.copy() if self.covariance_matrix is not None else None
            ),
            reconciliation_matrix=G,
        )

        logger.info(
            "MinT reconciliation complete: coherence %.6e → %.6e",
            coherence_before,
            coherence_after,
        )

        return result

    def _compute_reconciliation_matrix(
        self,
        hierarchy: HierarchyDefinition,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute MinT reconciliation matrix G.

        The MinT reconciliation matrix is computed as:
            G = S(S'WS)^{-1}S'W

        where:
            - S = aggregation matrix from hierarchy
            - W = Σ^{-1} (inverse of error covariance matrix)
            - Σ = fitted covariance matrix

        Args:
            hierarchy: HierarchyDefinition with aggregation matrix S.
            **kwargs: Additional parameters (unused, for interface compatibility).

        Returns:
            Reconciliation matrix G of shape (n_total, n_total).

        Raises:
            ValueError: If hierarchy has no aggregation matrix or covariance
                matrix is not fitted.

        Example:
            ```python
            # Compute G matrix
            G = reconciler._compute_reconciliation_matrix(hierarchy)

            # Apply to forecasts
            y_reconciled = G @ y_base
            ```
        """
        if hierarchy.aggregation_matrix is None:
            msg = "Hierarchy must have aggregation matrix built"
            raise ValueError(msg)

        if self.covariance_matrix is None:
            msg = "Covariance matrix not fitted. Call fit() first or use identity."
            raise ValueError(msg)

        # Get aggregation matrix S
        S = hierarchy.aggregation_matrix  # (n_total, n_bottom)

        # Compute W = Σ^{-1} (inverse covariance)
        try:
            W = np.linalg.inv(self.covariance_matrix)
        except np.linalg.LinAlgError as e:
            logger.error("Failed to invert covariance matrix: %s", e)
            # Add regularization and retry
            logger.warning("Adding regularization %.2e to diagonal", self.regularization * 10)
            cov_reg = (
                self.covariance_matrix
                + np.eye(len(self.covariance_matrix)) * self.regularization * 10
            )
            W = np.linalg.inv(cov_reg)

        # Compute S'WS
        STWS = S.T @ W @ S  # (n_bottom, n_bottom)

        # Invert (S'WS)
        try:
            STWS_inv = np.linalg.inv(STWS)
        except np.linalg.LinAlgError as e:
            logger.error("Failed to invert S'WS: %s", e)
            # Add regularization and retry
            logger.warning("Adding regularization %.2e to S'WS diagonal", self.regularization * 10)
            STWS_reg = STWS + np.eye(len(STWS)) * self.regularization * 10
            STWS_inv = np.linalg.inv(STWS_reg)

        # Compute G = S(S'WS)^{-1}S'W
        G = S @ STWS_inv @ S.T @ W  # (n_total, n_total)

        logger.debug("Computed MinT reconciliation matrix G with shape %s", G.shape)

        return G

    def _estimate_covariance(
        self,
        residuals: np.ndarray,
        method: CovarianceMethod = "shrinkage",
    ) -> np.ndarray:
        """Estimate error covariance matrix from residuals.

        Supports multiple estimation methods optimized for different scenarios.

        Args:
            residuals: Residual matrix (n_periods × n_nodes).
            method: Covariance estimation method. Options:
                - "sample": Sample covariance from residuals
                - "shrinkage": Ledoit-Wolf shrinkage estimator (robust)
                - "diagonal": Diagonal covariance (independence assumption)
                - "identity": Identity matrix (OLS reconciliation fallback)
                - "structural": Variance proportional to forecast level

        Returns:
            Covariance matrix Σ of shape (n_nodes, n_nodes).

        Raises:
            ValueError: If method is unknown or residuals are invalid.

        Example:
            ```python
            # Estimate using different methods
            cov_sample = reconciler._estimate_covariance(residuals, "sample")
            cov_shrink = reconciler._estimate_covariance(residuals, "shrinkage")
            cov_diag = reconciler._estimate_covariance(residuals, "diagonal")
            ```
        """
        n_periods, n_nodes = residuals.shape

        logger.debug(
            "Estimating covariance with method=%s from %d periods, %d nodes",
            method,
            n_periods,
            n_nodes,
        )

        if method == "sample":
            # Simple sample covariance: Σ = (1/n) X'X
            cov = np.cov(residuals.T, bias=False)
            logger.debug("Computed sample covariance")

        elif method == "shrinkage":
            # Ledoit-Wolf shrinkage estimator (robust for small samples)
            try:
                from sklearn.covariance import LedoitWolf

                lw = LedoitWolf()
                lw.fit(residuals)
                cov = lw.covariance_
                shrinkage = lw.shrinkage_
                logger.debug("Computed Ledoit-Wolf covariance (shrinkage=%.4f)", shrinkage)
            except ImportError:
                logger.warning("sklearn not available, falling back to sample covariance")
                cov = np.cov(residuals.T, bias=False)

        elif method == "diagonal":
            # Diagonal covariance (assume independence)
            variances = np.var(residuals, axis=0, ddof=1)
            # Ensure minimum variance
            variances = np.maximum(variances, self.config.min_variance)
            cov = np.diag(variances)
            logger.debug("Computed diagonal covariance")

        elif method == "identity":
            # Identity matrix (OLS reconciliation)
            cov = np.eye(n_nodes)
            logger.debug("Using identity covariance (OLS)")

        elif method == "structural":
            # Variance proportional to squared mean residual
            variances = np.mean(residuals**2, axis=0)
            # Ensure minimum variance
            variances = np.maximum(variances, self.config.min_variance)
            cov = np.diag(variances)
            logger.debug("Computed structural covariance")

        else:
            msg = (
                f"Unknown covariance method: {method}. "
                f"Valid: sample, shrinkage, diagonal, identity, structural"
            )
            raise ValueError(msg)

        return cov

    def _ensure_positive_definite(
        self,
        cov: np.ndarray,
        epsilon: float | None = None,
    ) -> np.ndarray:
        """Ensure covariance matrix is positive definite.

        Adds regularization to the diagonal if the matrix is not positive
        definite or is poorly conditioned.

        Args:
            cov: Covariance matrix to regularize.
            epsilon: Regularization parameter. If None, uses self.regularization.

        Returns:
            Positive definite covariance matrix.

        Example:
            ```python
            # Ensure positive definiteness
            cov_pd = reconciler._ensure_positive_definite(cov)

            # Check
            eigenvalues = np.linalg.eigvals(cov_pd)
            assert all(eigenvalues > 0)
            ```
        """
        if epsilon is None:
            epsilon = self.regularization

        # Check if already positive definite
        try:
            eigenvalues = np.linalg.eigvals(cov)
            min_eigenvalue = np.min(eigenvalues)

            if min_eigenvalue <= 0:
                logger.warning(
                    "Covariance matrix not positive definite (min eigenvalue=%.2e). "
                    "Adding regularization %.2e",
                    min_eigenvalue,
                    epsilon,
                )
                cov = cov + np.eye(len(cov)) * epsilon

            elif min_eigenvalue < epsilon * 10:
                logger.warning(
                    "Covariance matrix poorly conditioned (min eigenvalue=%.2e). "
                    "Adding regularization %.2e",
                    min_eigenvalue,
                    epsilon,
                )
                cov = cov + np.eye(len(cov)) * epsilon

            else:
                logger.debug(
                    "Covariance matrix is positive definite (min eigenvalue=%.2e)",
                    min_eigenvalue,
                )

        except np.linalg.LinAlgError as e:
            logger.error("Failed to compute eigenvalues: %s. Adding regularization.", e)
            cov = cov + np.eye(len(cov)) * epsilon

        return cov

    def save(self, path: str | Path) -> None:
        """Save reconciler configuration and fitted parameters to disk.

        Saves both the configuration and the fitted covariance matrix.

        Args:
            path: Output file path (should end with .json).

        Example:
            ```python
            reconciler.save("config/mint_reconciler.json")
            ```
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Base config
        config_dict = self.config.to_dict()
        config_dict["fitted"] = self.fitted
        config_dict["reconciler_class"] = type(self).__name__
        config_dict["covariance_method"] = self.covariance_method
        config_dict["regularization"] = self.regularization

        # Save covariance matrix separately if fitted
        if self.covariance_matrix is not None:
            cov_path = path.with_suffix(".cov.npy")
            np.save(cov_path, self.covariance_matrix)
            config_dict["covariance_matrix_path"] = str(cov_path)
            logger.info("Saved covariance matrix to %s", cov_path)

        # Save config
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info("Saved MinT reconciler config to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MintReconciler":
        """Load reconciler configuration and fitted parameters from disk.

        Args:
            path: Path to saved configuration file.

        Returns:
            MintReconciler instance with loaded configuration and covariance.

        Raises:
            FileNotFoundError: If file doesn't exist.

        Example:
            ```python
            reconciler = MintReconciler.load("config/mint_reconciler.json")
            result = reconciler.reconcile(base_forecasts, hierarchy)
            ```
        """
        path = Path(path)

        if not path.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)

        with open(path) as f:
            config_dict = json.load(f)

        # Extract MinT-specific fields
        fitted = config_dict.pop("fitted", False)
        reconciler_class = config_dict.pop("reconciler_class", None)
        covariance_method = config_dict.pop("covariance_method", "shrinkage")
        regularization = config_dict.pop("regularization", 1e-8)
        cov_path = config_dict.pop("covariance_matrix_path", None)

        # Create config (include extra fields for covariance_method)
        config = ReconciliationConfig(**config_dict)
        if config.model_extra is None:
            config.model_extra = {}
        config.model_extra["covariance_method"] = covariance_method
        config.model_extra["regularization"] = regularization

        # Create instance
        instance = cls(config=config)
        instance.fitted = fitted

        # Load covariance matrix if available
        if cov_path and Path(cov_path).exists():
            instance.covariance_matrix = np.load(cov_path)
            logger.info("Loaded covariance matrix from %s", cov_path)
        elif fitted:
            logger.warning("Config indicates fitted=True but no covariance matrix found")

        logger.info("Loaded MinT reconciler from %s (class: %s)", path, reconciler_class)

        return instance

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the reconciler.
        """
        return (
            f"MintReconciler(method={self.config.method!r}, "
            f"covariance_method={self.covariance_method!r}, "
            f"fitted={self.fitted})"
        )
