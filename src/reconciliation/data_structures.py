"""Data structures for hierarchical forecast reconciliation.

This module provides the core data structures used in the reconciliation subsystem:
- ReconciliationResult: Stores reconciled forecasts and metadata
- ReconciliationConfig: Configuration for reconciliation methods

Example:
    ```python
    from src.reconciliation import ReconciliationResult, ReconciliationConfig
    import pandas as pd
    import numpy as np

    # Configuration
    config = ReconciliationConfig(
        method="mint",
        non_negative=True,
        check_coherence=True
    )

    # Result
    reconciled_df = pd.DataFrame(...)
    result = ReconciliationResult(
        reconciled_forecasts=reconciled_df,
        metadata={"method": "mint", "coherence_after": 0.0},
        base_forecasts=base_df
    )

    # Validate coherence
    is_coherent = result.validate_coherence(hierarchy, tolerance=1e-6)
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReconciliationConfig(BaseModel):
    """Configuration for hierarchical forecast reconciliation.

    This Pydantic model defines the configuration parameters for reconciliation
    methods, including method selection, constraints, and validation options.

    Attributes:
        method: Reconciliation method name (e.g., "mint", "ols", "wls", "bottom_up").
        bottom_up: Force bottom-up reconciliation (ignore base forecasts for upper levels).
        non_negative: Enforce non-negativity constraint on reconciled forecasts.
        check_coherence: Validate coherence after reconciliation.
        hierarchy_path: Path to hierarchy YAML configuration file.
        tolerance: Tolerance for coherence validation (default: 1e-6).
        min_variance: Minimum variance threshold for covariance estimation (default: 1e-8).

    Example:
        ```python
        # MinT configuration
        config = ReconciliationConfig(
            method="mint",
            non_negative=True,
            check_coherence=True,
            hierarchy_path="config/hierarchy_brazil.yaml"
        )

        # Bottom-up configuration
        config = ReconciliationConfig(
            method="bottom_up",
            bottom_up=True,
            non_negative=True
        )

        # OLS configuration with custom tolerance
        config = ReconciliationConfig(
            method="ols",
            tolerance=1e-4,
            check_coherence=True
        )
        ```
    """

    model_config = ConfigDict(
        extra="allow",
        validate_default=True,
        str_strip_whitespace=True,
    )

    method: str = Field(
        default="ols",
        description="Reconciliation method (mint, ols, wls, bottom_up)",
    )
    bottom_up: bool = Field(
        default=False,
        description="Force bottom-up reconciliation",
    )
    non_negative: bool = Field(
        default=True,
        description="Enforce non-negativity constraint",
    )
    check_coherence: bool = Field(
        default=True,
        description="Validate coherence after reconciliation",
    )
    hierarchy_path: str | None = Field(
        default=None,
        description="Path to hierarchy YAML file",
    )
    tolerance: float = Field(
        default=1e-6,
        gt=0,
        description="Tolerance for coherence validation",
    )
    min_variance: float = Field(
        default=1e-8,
        gt=0,
        description="Minimum variance threshold for covariance estimation",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate reconciliation method name.

        Args:
            v: Method name to validate.

        Returns:
            Validated method name (lowercase).

        Raises:
            ValueError: If method is not recognized.
        """
        valid_methods = {"mint", "ols", "wls", "bottom_up", "top_down"}
        v_lower = v.lower()
        if v_lower not in valid_methods:
            msg = f"Unknown reconciliation method: {v}. Valid: {valid_methods}"
            logger.warning(msg)
        return v_lower

    @field_validator("tolerance", "min_variance")
    @classmethod
    def validate_positive(cls, v: float, info: Any) -> float:
        """Validate positive values for tolerance and min_variance.

        Args:
            v: Value to validate.
            info: Field info from Pydantic.

        Returns:
            Validated value.
        """
        if v <= 0:
            msg = f"{info.field_name} must be positive, got {v}"
            raise ValueError(msg)
        if v > 1.0:
            logger.warning("%s is unusually large: %f", info.field_name, v)
        return v

    def get_extra_field(self, field_name: str, default: Any = None) -> Any:
        """Get a method-specific extra field value.

        Args:
            field_name: Name of the extra field to retrieve.
            default: Default value if field doesn't exist.

        Returns:
            The field value or the default.
        """
        if self.model_extra is not None:
            return self.model_extra.get(field_name, default)
        return default

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary with all configuration fields including extras.
        """
        result = {
            "method": self.method,
            "bottom_up": self.bottom_up,
            "non_negative": self.non_negative,
            "check_coherence": self.check_coherence,
            "hierarchy_path": self.hierarchy_path,
            "tolerance": self.tolerance,
            "min_variance": self.min_variance,
        }
        if self.model_extra:
            result.update(self.model_extra)
        return result


@dataclass
class ReconciliationResult:
    """Result container for hierarchical forecast reconciliation.

    This dataclass stores the reconciled forecasts along with metadata about
    the reconciliation process, including coherence metrics and timing information.

    Attributes:
        reconciled_forecasts: DataFrame with reconciled forecasts (time × nodes).
        metadata: Dictionary with reconciliation metadata (method, timing, coherence).
        base_forecasts: Original base forecasts before reconciliation (optional).
        residual_covariance: Residual error covariance matrix for MinT (optional).
        reconciliation_matrix: The G matrix used for reconciliation (optional).

    Example:
        ```python
        import pandas as pd
        from src.reconciliation import ReconciliationResult

        # Create result
        result = ReconciliationResult(
            reconciled_forecasts=reconciled_df,
            metadata={
                "method": "mint",
                "timestamp": "2025-11-25T10:00:00",
                "coherence_before": 15.3,
                "coherence_after": 0.0,
                "elapsed_time": 0.123
            },
            base_forecasts=base_df
        )

        # Access reconciled forecasts
        print(result.reconciled_forecasts.head())

        # Check coherence
        is_coherent = result.validate_coherence(hierarchy)

        # Convert to dict for serialization
        result_dict = result.to_dict()
        ```
    """

    reconciled_forecasts: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    base_forecasts: pd.DataFrame | None = None
    residual_covariance: np.ndarray | None = None
    reconciliation_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        if self.reconciled_forecasts is None or self.reconciled_forecasts.empty:
            msg = "reconciled_forecasts cannot be empty"
            raise ValueError(msg)

        # Ensure metadata has required fields
        if "method" not in self.metadata:
            self.metadata["method"] = "unknown"
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()

    def validate_coherence(
        self,
        hierarchy: Any,  # HierarchyDefinition, but avoid circular import
        tolerance: float = 1e-6,
    ) -> bool:
        """Validate that reconciled forecasts satisfy coherence constraints.

        Uses the hierarchy aggregation matrix to check if the forecasts satisfy
        the aggregation constraints: Sy ≈ reconciled_forecasts (within tolerance).

        Args:
            hierarchy: HierarchyDefinition instance with aggregation matrix.
            tolerance: Maximum allowed coherence error (default: 1e-6).

        Returns:
            True if forecasts are coherent within tolerance, False otherwise.

        Raises:
            ValueError: If hierarchy has no aggregation matrix.

        Example:
            ```python
            from src.reconciliation import HierarchyDefinition

            hierarchy = HierarchyDefinition("config/hierarchy_brazil.yaml")
            is_coherent = result.validate_coherence(hierarchy, tolerance=1e-6)

            if is_coherent:
                print("Forecasts are coherent!")
            else:
                print("Coherence violation detected")
            ```
        """
        if hierarchy.aggregation_matrix is None:
            msg = "Hierarchy must have aggregation matrix built"
            raise ValueError(msg)

        # Get bottom-level nodes
        bottom_nodes = sorted(hierarchy.get_bottom_level_nodes())
        all_nodes = hierarchy.get_node_names_sorted()

        # Extract bottom-level forecasts from reconciled forecasts
        try:
            y_bottom = self.reconciled_forecasts[bottom_nodes].values  # (T, n_bottom)
        except KeyError as e:
            logger.error("Missing bottom-level nodes in reconciled forecasts: %s", e)
            return False

        # Extract all forecasts in the correct order
        try:
            y_all = self.reconciled_forecasts[all_nodes].values  # (T, n_total)
        except KeyError as e:
            logger.error("Missing nodes in reconciled forecasts: %s", e)
            return False

        # Compute aggregated values: S @ y_bottom.T -> (n_total, T)
        S = hierarchy.aggregation_matrix
        y_aggregated = (S @ y_bottom.T).T  # (T, n_total)

        # Compute coherence error
        coherence_error = np.linalg.norm(y_aggregated - y_all) / np.linalg.norm(y_all)

        # Store coherence error in metadata
        self.metadata["coherence_error_validated"] = float(coherence_error)

        is_coherent = bool(coherence_error <= tolerance)

        if not is_coherent:
            logger.warning(
                "Coherence validation failed: error=%.6e > tolerance=%.6e",
                coherence_error,
                tolerance,
            )
        else:
            logger.debug("Coherence validated: error=%.6e", coherence_error)

        return is_coherent

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization.

        Returns:
            Dictionary with all result data. DataFrames and arrays are converted
            to lists for JSON serialization.

        Example:
            ```python
            result_dict = result.to_dict()

            # Save to JSON
            import json
            with open("result.json", "w") as f:
                json.dump(result_dict, f, indent=2)
            ```
        """
        result = {
            "reconciled_forecasts": self.reconciled_forecasts.to_dict(orient="split"),
            "metadata": self.metadata.copy(),
            "base_forecasts": (
                self.base_forecasts.to_dict(orient="split")
                if self.base_forecasts is not None
                else None
            ),
            "residual_covariance": (
                self.residual_covariance.tolist()
                if self.residual_covariance is not None
                else None
            ),
            "reconciliation_matrix": (
                self.reconciliation_matrix.tolist()
                if self.reconciliation_matrix is not None
                else None
            ),
        }

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReconciliationResult":
        """Create result from dictionary.

        Args:
            data: Dictionary with result data (from to_dict()).

        Returns:
            ReconciliationResult instance.

        Example:
            ```python
            import json

            # Load from JSON
            with open("result.json") as f:
                result_dict = json.load(f)

            result = ReconciliationResult.from_dict(result_dict)
            ```
        """
        # Convert reconciled_forecasts
        recon_data = data["reconciled_forecasts"]
        reconciled_df = pd.DataFrame(
            recon_data["data"],
            columns=recon_data["columns"],
        )
        # Restore index if it was datetime
        if recon_data["index"]:
            try:
                reconciled_df.index = pd.to_datetime(recon_data["index"])
            except (ValueError, TypeError):
                reconciled_df.index = recon_data["index"]

        # Convert base_forecasts if present
        base_df = None
        if "base_forecasts" in data and data["base_forecasts"] is not None:
            base_data = data["base_forecasts"]
            base_df = pd.DataFrame(
                base_data["data"],
                columns=base_data["columns"],
            )
            # Restore index if it was datetime
            if base_data["index"]:
                try:
                    base_df.index = pd.to_datetime(base_data["index"])
                except (ValueError, TypeError):
                    base_df.index = base_data["index"]

        # Convert covariance if present
        residual_cov = None
        if "residual_covariance" in data and data["residual_covariance"] is not None:
            residual_cov = np.array(data["residual_covariance"])

        # Convert reconciliation matrix if present
        recon_matrix = None
        if "reconciliation_matrix" in data and data["reconciliation_matrix"] is not None:
            recon_matrix = np.array(data["reconciliation_matrix"])

        return cls(
            reconciled_forecasts=reconciled_df,
            metadata=data.get("metadata", {}),
            base_forecasts=base_df,
            residual_covariance=residual_cov,
            reconciliation_matrix=recon_matrix,
        )

    def save(self, path: str | Path) -> None:
        """Save result to disk.

        Saves the result as a compressed pickle file with all DataFrames and arrays.

        Args:
            path: Output file path (should end with .pkl or .pickle).

        Example:
            ```python
            result.save("outputs/reconciliation_result.pkl")
            ```
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

        logger.info("Saved reconciliation result to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ReconciliationResult":
        """Load result from disk.

        Args:
            path: Path to saved result file.

        Returns:
            Loaded ReconciliationResult instance.

        Raises:
            FileNotFoundError: If file doesn't exist.

        Example:
            ```python
            result = ReconciliationResult.load("outputs/reconciliation_result.pkl")
            print(result.metadata)
            ```
        """
        import pickle

        path = Path(path)

        if not path.exists():
            msg = f"Result file not found: {path}"
            raise FileNotFoundError(msg)

        with open(path, "rb") as f:
            data = pickle.load(f)

        logger.info("Loaded reconciliation result from %s", path)

        return cls.from_dict(data)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the result.
        """
        method = self.metadata.get("method", "unknown")
        n_timesteps, n_nodes = self.reconciled_forecasts.shape
        return (
            f"ReconciliationResult(method={method!r}, "
            f"timesteps={n_timesteps}, nodes={n_nodes})"
        )
