"""Abstract base class for hierarchical forecast reconciliation.

This module defines the BaseReconciler abstract class that all reconciliation
methods must implement. It provides common functionality for validation,
coherence checking, and forecast manipulation.

Reconciliation methods transform base forecasts to ensure they satisfy
hierarchical aggregation constraints:
    reconciled = G @ base

Where G is a method-specific reconciliation matrix.

Example:
    ```python
    from src.reconciliation import BaseReconciler, ReconciliationConfig
    import numpy as np

    class OLSReconciler(BaseReconciler):
        def reconcile(self, base_forecasts, hierarchy):
            # Validate inputs
            self.validate_forecasts(base_forecasts, hierarchy)

            # Compute reconciliation matrix G = S(S'S)^{-1}S'
            G = self._compute_reconciliation_matrix(hierarchy)

            # Apply reconciliation
            all_nodes = hierarchy.get_node_names_sorted()
            y_base = base_forecasts[all_nodes].values.T  # (n_total, T)
            y_reconciled = (G @ y_base).T  # (T, n_total)

            # Create result
            reconciled_df = pd.DataFrame(
                y_reconciled, index=base_forecasts.index, columns=all_nodes
            )

            return ReconciliationResult(
                reconciled_forecasts=reconciled_df,
                metadata={"method": "ols"},
                base_forecasts=base_forecasts
            )

        def _compute_reconciliation_matrix(self, hierarchy, **kwargs):
            S = hierarchy.aggregation_matrix
            return S @ np.linalg.inv(S.T @ S) @ S.T
    ```
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.reconciliation.data_structures import ReconciliationConfig, ReconciliationResult
from src.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseReconciler(ABC):
    """Abstract base class for hierarchical forecast reconciliation.

    This class defines the interface that all reconciliation methods must implement,
    including MinT, OLS, WLS, and others. It provides common helper methods for
    forecast validation, coherence checking, and data manipulation.

    Subclasses must implement:
    - reconcile(): Apply reconciliation to base forecasts
    - _compute_reconciliation_matrix(): Compute method-specific G matrix

    Optionally, subclasses can implement:
    - fit(): Learn parameters from historical data

    Attributes:
        config: ReconciliationConfig with method parameters.
        fitted: Whether the reconciler has been fitted with historical data.
        logger: Logger instance for this reconciler.

    Example:
        ```python
        from src.reconciliation import BaseReconciler, ReconciliationConfig

        # Configuration
        config = ReconciliationConfig(
            method="mint",
            non_negative=True,
            check_coherence=True
        )

        # Subclass implements reconcile()
        reconciler = MintReconciler(config=config)

        # Optional: Fit with historical data
        reconciler.fit(historical_forecasts, actuals, hierarchy)

        # Reconcile new forecasts
        result = reconciler.reconcile(base_forecasts, hierarchy)

        # Access results
        print(result.reconciled_forecasts)
        print(f"Coherence error: {result.metadata['coherence_after']:.6f}")
        ```
    """

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        """Initialize base reconciler.

        Args:
            config: Reconciliation configuration. If None, uses default config.
        """
        self.config = config or ReconciliationConfig()
        self.fitted = False
        self.logger = get_logger(__name__)

    @abstractmethod
    def reconcile(
        self,
        base_forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> ReconciliationResult:
        """Reconcile base forecasts to ensure hierarchical coherence.

        This is the main method that applies reconciliation to base forecasts.
        Subclasses must implement the specific reconciliation algorithm.

        Args:
            base_forecasts: DataFrame with base forecasts (time × nodes).
                Must contain all nodes in the hierarchy as columns.
            hierarchy: HierarchyDefinition with structure and aggregation matrix.

        Returns:
            ReconciliationResult with reconciled forecasts and metadata.

        Raises:
            ValueError: If base_forecasts shape doesn't match hierarchy.

        Example:
            ```python
            # Base forecasts (potentially incoherent)
            base_forecasts = pd.DataFrame({
                "BR_National": [1000, 1100],
                "SE": [400, 440],
                "S": [300, 330],
                # ... more nodes
            })

            # Reconcile
            result = reconciler.reconcile(base_forecasts, hierarchy)

            # Check coherence
            assert result.validate_coherence(hierarchy, tolerance=1e-6)
            ```
        """

    @abstractmethod
    def _compute_reconciliation_matrix(
        self,
        hierarchy: HierarchyDefinition,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute method-specific reconciliation matrix G.

        The reconciliation matrix G transforms base forecasts to reconciled forecasts:
            reconciled = G @ base

        Different methods compute G differently:
        - OLS: G = S(S'S)^{-1}S'
        - WLS: G = S(S'W^{-1}S)^{-1}S'W^{-1}
        - MinT: G = S(S'Σ^{-1}S)^{-1}S'Σ^{-1} where Σ is error covariance

        Args:
            hierarchy: HierarchyDefinition with aggregation matrix S.
            **kwargs: Method-specific parameters (e.g., weights, covariance).

        Returns:
            Reconciliation matrix G of shape (n_total, n_total).

        Raises:
            ValueError: If hierarchy is invalid or parameters are missing.

        Example:
            ```python
            # OLS reconciliation matrix
            G = reconciler._compute_reconciliation_matrix(hierarchy)

            # MinT with covariance matrix
            G = reconciler._compute_reconciliation_matrix(
                hierarchy,
                covariance_matrix=Sigma
            )
            ```
        """

    def fit(
        self,
        historical_forecasts: pd.DataFrame,
        actuals: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> None:
        """Learn reconciliation parameters from historical data.

        This is an optional method for learning-based reconcilers (e.g., MinT
        which learns error covariance from historical residuals). Methods that
        don't require learning (OLS, bottom-up) can leave this as a no-op.

        Args:
            historical_forecasts: Historical base forecasts (time × nodes).
            actuals: Historical actual values (time × nodes).
            hierarchy: HierarchyDefinition for structure.

        Example:
            ```python
            # Fit MinT reconciler with historical data
            reconciler.fit(historical_forecasts, actuals, hierarchy)

            # Now the reconciler uses learned covariance for reconciliation
            result = reconciler.reconcile(new_forecasts, hierarchy)
            ```
        """
        logger.info("fit() called but not implemented for %s", type(self).__name__)
        self.fitted = False

    def validate_forecasts(
        self,
        forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> None:
        """Validate forecast DataFrame against hierarchy structure.

        Checks:
        - All hierarchy nodes are present as columns
        - No NaN values in forecasts
        - DataFrame is not empty
        - Shape matches hierarchy expectations

        Args:
            forecasts: Forecast DataFrame to validate.
            hierarchy: HierarchyDefinition to validate against.

        Raises:
            ValueError: If validation fails.

        Example:
            ```python
            # Validate before reconciliation
            reconciler.validate_forecasts(base_forecasts, hierarchy)

            # If validation passes, proceed with reconciliation
            result = reconciler.reconcile(base_forecasts, hierarchy)
            ```
        """
        # Check empty
        if forecasts.empty:
            msg = "Forecasts DataFrame is empty"
            raise ValueError(msg)

        # Check for NaN values
        if forecasts.isna().any().any():
            nan_cols = forecasts.columns[forecasts.isna().any()].tolist()
            msg = f"Forecasts contain NaN values in columns: {nan_cols}"
            raise ValueError(msg)

        # Check that all hierarchy nodes are present
        all_nodes = set(hierarchy.get_node_names_sorted())
        forecast_cols = set(forecasts.columns)

        missing_nodes = all_nodes - forecast_cols
        if missing_nodes:
            msg = f"Missing nodes in forecasts: {sorted(missing_nodes)}"
            raise ValueError(msg)

        extra_nodes = forecast_cols - all_nodes
        if extra_nodes:
            logger.warning("Extra nodes in forecasts (will be ignored): %s", sorted(extra_nodes))

        # Check shape
        n_timesteps, n_cols = forecasts.shape
        n_expected_nodes = hierarchy.n_nodes

        logger.debug(
            "Validated forecasts: %d timesteps, %d nodes (expected %d)",
            n_timesteps,
            len(forecast_cols & all_nodes),
            n_expected_nodes,
        )

    def check_coherence(
        self,
        forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> float:
        """Compute coherence error for forecasts.

        Coherence error measures how well forecasts satisfy aggregation constraints:
            error = ||S @ y_bottom - y_all|| / ||y_all||

        Where S is the aggregation matrix, y_bottom are bottom-level forecasts,
        and y_all are all forecasts.

        Args:
            forecasts: Forecast DataFrame to check (time × nodes).
            hierarchy: HierarchyDefinition with aggregation matrix.

        Returns:
            Normalized coherence error (0 = perfectly coherent).

        Raises:
            ValueError: If hierarchy has no aggregation matrix.

        Example:
            ```python
            # Check coherence before reconciliation
            error_before = reconciler.check_coherence(base_forecasts, hierarchy)
            print(f"Coherence error before: {error_before:.6f}")

            # Reconcile
            result = reconciler.reconcile(base_forecasts, hierarchy)

            # Check coherence after
            error_after = reconciler.check_coherence(
                result.reconciled_forecasts, hierarchy
            )
            print(f"Coherence error after: {error_after:.6e}")
            ```
        """
        if hierarchy.aggregation_matrix is None:
            msg = "Hierarchy must have aggregation matrix built"
            raise ValueError(msg)

        # Get nodes in correct order
        bottom_nodes = sorted(hierarchy.get_bottom_level_nodes())
        all_nodes = hierarchy.get_node_names_sorted()

        # Extract forecasts
        try:
            y_bottom = forecasts[bottom_nodes].values  # (T, n_bottom)
            y_all = forecasts[all_nodes].values  # (T, n_total)
        except KeyError as e:
            msg = f"Missing nodes in forecasts: {e}"
            raise ValueError(msg) from e

        # Compute aggregated values: S @ y_bottom.T -> (n_total, T)
        S = hierarchy.aggregation_matrix
        y_aggregated = (S @ y_bottom.T).T  # (T, n_total)

        # Compute normalized coherence error
        diff = y_aggregated - y_all
        coherence_error = np.linalg.norm(diff) / np.linalg.norm(y_all)

        return float(coherence_error)

    def extract_bottom_level(
        self,
        forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> pd.DataFrame:
        """Extract bottom-level forecasts from full forecast DataFrame.

        Args:
            forecasts: Full forecast DataFrame (time × nodes).
            hierarchy: HierarchyDefinition with node structure.

        Returns:
            DataFrame with only bottom-level forecasts (time × bottom_nodes).

        Example:
            ```python
            # Extract bottom-level forecasts
            bottom_forecasts = reconciler.extract_bottom_level(
                all_forecasts, hierarchy
            )

            # Use for bottom-up reconciliation
            aggregated = reconciler.aggregate_to_top(bottom_forecasts, hierarchy)
            ```
        """
        bottom_nodes = sorted(hierarchy.get_bottom_level_nodes())

        try:
            return forecasts[bottom_nodes].copy()
        except KeyError as e:
            msg = f"Missing bottom-level nodes in forecasts: {e}"
            raise ValueError(msg) from e

    def aggregate_to_top(
        self,
        bottom_forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> pd.DataFrame:
        """Aggregate bottom-level forecasts to all hierarchy levels.

        Uses the aggregation matrix S to compute all forecasts from bottom-level:
            y_all = S @ y_bottom

        Args:
            bottom_forecasts: Bottom-level forecasts (time × bottom_nodes).
            hierarchy: HierarchyDefinition with aggregation matrix.

        Returns:
            DataFrame with forecasts for all nodes (time × all_nodes).

        Raises:
            ValueError: If bottom_forecasts columns don't match hierarchy.

        Example:
            ```python
            # Bottom-up reconciliation
            bottom_forecasts = reconciler.extract_bottom_level(
                base_forecasts, hierarchy
            )
            reconciled = reconciler.aggregate_to_top(bottom_forecasts, hierarchy)
            ```
        """
        if hierarchy.aggregation_matrix is None:
            msg = "Hierarchy must have aggregation matrix built"
            raise ValueError(msg)

        # Get node ordering
        bottom_nodes = sorted(hierarchy.get_bottom_level_nodes())
        all_nodes = hierarchy.get_node_names_sorted()

        # Validate columns
        if not all(node in bottom_forecasts.columns for node in bottom_nodes):
            missing = [n for n in bottom_nodes if n not in bottom_forecasts.columns]
            msg = f"Missing bottom-level nodes: {missing}"
            raise ValueError(msg)

        # Extract bottom-level values in correct order
        y_bottom = bottom_forecasts[bottom_nodes].values  # (T, n_bottom)

        # Aggregate using S matrix: S @ y_bottom.T -> (n_total, T)
        S = hierarchy.aggregation_matrix
        y_all = (S @ y_bottom.T).T  # (T, n_total)

        # Create DataFrame with all nodes
        return pd.DataFrame(
            y_all,
            index=bottom_forecasts.index,
            columns=all_nodes,
        )

    def apply_non_negativity(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Apply non-negativity constraint to forecasts.

        Replaces negative values with zero (or small positive value).

        Args:
            forecasts: Forecast DataFrame (time × nodes).

        Returns:
            DataFrame with non-negative values.

        Example:
            ```python
            # Apply non-negativity after reconciliation
            if self.config.non_negative:
                reconciled_df = self.apply_non_negativity(reconciled_df)
            ```
        """
        if self.config.non_negative:
            # Replace negative values with small positive value
            forecasts = forecasts.clip(lower=0.0)
            logger.debug("Applied non-negativity constraint")

        return forecasts

    def save(self, path: str | Path) -> None:
        """Save reconciler configuration to disk.

        Args:
            path: Output file path (should end with .json or .yaml).

        Example:
            ```python
            reconciler.save("config/reconciler_config.json")
            ```
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.config.to_dict()
        config_dict["fitted"] = self.fitted
        config_dict["reconciler_class"] = type(self).__name__

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info("Saved reconciler config to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseReconciler":
        """Load reconciler configuration from disk.

        Note: This only loads the configuration. Subclasses may need to
        override to load additional fitted parameters.

        Args:
            path: Path to saved configuration file.

        Returns:
            Reconciler instance with loaded configuration.

        Raises:
            FileNotFoundError: If file doesn't exist.

        Example:
            ```python
            reconciler = MintReconciler.load("config/reconciler_config.json")
            ```
        """
        import json

        path = Path(path)

        if not path.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)

        with open(path) as f:
            config_dict = json.load(f)

        # Extract fitted flag
        fitted = config_dict.pop("fitted", False)
        reconciler_class = config_dict.pop("reconciler_class", None)

        # Create config
        config = ReconciliationConfig(**config_dict)

        # Create instance
        instance = cls(config=config)
        instance.fitted = fitted

        logger.info("Loaded reconciler config from %s (class: %s)", path, reconciler_class)

        return instance

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the reconciler.
        """
        return (
            f"{type(self).__name__}(method={self.config.method!r}, "
            f"fitted={self.fitted})"
        )
