"""Data structures for model combination.

This module provides dataclasses for storing combination results, metadata,
and configuration for the model combination framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CombinerConfig:
    """Configuration for model combiners.

    Provides standardized configuration options for all combiner implementations.

    Attributes:
        missing_threshold: Threshold for excluding models based on missing data (0.0-1.0).
            Models with missing rate above this threshold are excluded.
        require_fit: Whether fit() must be called before combine().
        validate_weights: Whether to validate that weights sum to 1.
        store_contributions: Whether to store individual model contributions.
        store_intervals: Whether to compute and store prediction intervals.
        log_level: Logging level for the combiner.
        custom_config: Additional custom configuration parameters.

    Example:
        >>> config = CombinerConfig(
        ...     missing_threshold=0.3,
        ...     require_fit=True,
        ...     store_contributions=True
        ... )
        >>> combiner = SomeCombiber(config=config)
    """

    missing_threshold: float = 0.5
    require_fit: bool = True
    validate_weights: bool = True
    store_contributions: bool = False
    store_intervals: bool = False
    log_level: str = "INFO"
    custom_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0.0 <= self.missing_threshold <= 1.0:
            msg = f"missing_threshold must be between 0 and 1, got {self.missing_threshold}"
            raise ValueError(msg)

        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            msg = f"log_level must be one of {valid_log_levels}, got {self.log_level}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "missing_threshold": self.missing_threshold,
            "require_fit": self.require_fit,
            "validate_weights": self.validate_weights,
            "store_contributions": self.store_contributions,
            "store_intervals": self.store_intervals,
            "log_level": self.log_level,
            "custom_config": self.custom_config.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CombinerConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            CombinerConfig instance.
        """
        return cls(
            missing_threshold=data.get("missing_threshold", 0.5),
            require_fit=data.get("require_fit", True),
            validate_weights=data.get("validate_weights", True),
            store_contributions=data.get("store_contributions", False),
            store_intervals=data.get("store_intervals", False),
            log_level=data.get("log_level", "INFO"),
            custom_config=data.get("custom_config", {}),
        )


@dataclass
class CombinationMetadata:
    """Metadata about model combination process.

    Tracks which models were used, timing information, quality metrics,
    and any warnings generated during the combination process.

    Attributes:
        timestamp: When the combination was performed.
        combination_method: Name of the combination method used.
        models_used: List of model names included in combination.
        models_excluded: List of model names excluded from combination.
        processing_time_seconds: Time taken to perform combination.
        performance_metrics: Dictionary of performance metrics.
        configuration: Configuration used for the combination.
        warnings: List of warning messages generated during combination.

    Example:
        >>> metadata = CombinationMetadata(
        ...     timestamp=datetime.now(),
        ...     combination_method="weighted_average",
        ...     models_used=["lgbm", "rf", "xgb"],
        ...     models_excluded=["arima"],
        ...     processing_time_seconds=0.5
        ... )
    """

    timestamp: datetime
    combination_method: str
    models_used: list[str]
    models_excluded: list[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    performance_metrics: dict[str, float] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, warning: str) -> None:
        """Add warning message to metadata.

        Args:
            warning: Warning message to add.
        """
        self.warnings.append(warning)

    def add_metric(self, name: str, value: float) -> None:
        """Add performance metric to metadata.

        Args:
            name: Metric name.
            value: Metric value.
        """
        self.performance_metrics[name] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary.

        Returns:
            Dictionary representation of the metadata.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "combination_method": self.combination_method,
            "models_used": self.models_used.copy(),
            "models_excluded": self.models_excluded.copy(),
            "processing_time_seconds": self.processing_time_seconds,
            "performance_metrics": self.performance_metrics.copy(),
            "configuration": self.configuration.copy(),
            "warnings": self.warnings.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CombinationMetadata":
        """Create metadata from dictionary.

        Args:
            data: Dictionary with metadata values.

        Returns:
            CombinationMetadata instance.
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            timestamp=timestamp,
            combination_method=data.get("combination_method", "unknown"),
            models_used=data.get("models_used", []),
            models_excluded=data.get("models_excluded", []),
            processing_time_seconds=data.get("processing_time_seconds", 0.0),
            performance_metrics=data.get("performance_metrics", {}),
            configuration=data.get("configuration", {}),
            warnings=data.get("warnings", []),
        )

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the metadata.
        """
        return (
            f"CombinationMetadata("
            f"method={self.combination_method!r}, "
            f"models_used={len(self.models_used)}, "
            f"models_excluded={len(self.models_excluded)}, "
            f"time={self.processing_time_seconds:.3f}s)"
        )


@dataclass
class CombinationResult:
    """Result of model combination operation.

    Contains combined predictions, weights used, and metadata about
    the combination process. Optionally includes individual model
    contributions and confidence intervals.

    Attributes:
        combined_predictions: DataFrame with combined forecasts.
            Columns typically include pred_h0, pred_h1, etc. for horizons.
        weights: Dictionary mapping model names to their weights.
        metadata: CombinationMetadata instance with process information.
        model_contributions: Optional breakdown of each model's contribution.
        confidence_intervals: Optional DataFrame with prediction intervals.

    Example:
        >>> result = CombinationResult(
        ...     combined_predictions=combined_df,
        ...     weights={"lgbm": 0.4, "rf": 0.3, "xgb": 0.3},
        ...     metadata=metadata
        ... )
        >>> print(f"Combined {len(result.weights)} models")
    """

    combined_predictions: pd.DataFrame
    weights: dict[str, float]
    metadata: CombinationMetadata
    model_contributions: dict[str, pd.DataFrame] | None = None
    confidence_intervals: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        if self.combined_predictions.empty:
            msg = "Combined predictions cannot be empty"
            raise ValueError(msg)

        if not self.weights:
            msg = "Weights dictionary cannot be empty"
            raise ValueError(msg)

        # Validate weights sum to 1 (within tolerance)
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            msg = f"Weights must sum to 1, got {weight_sum:.6f}"
            raise ValueError(msg)

        # Validate all weights are non-negative
        if any(w < 0 for w in self.weights.values()):
            msg = "All weights must be non-negative"
            raise ValueError(msg)

    @property
    def n_models(self) -> int:
        """Get number of models in combination.

        Returns:
            Number of models used in combination.
        """
        return len(self.weights)

    @property
    def n_samples(self) -> int:
        """Get number of samples in predictions.

        Returns:
            Number of samples (rows) in combined predictions.
        """
        return len(self.combined_predictions)

    @property
    def horizons(self) -> list[int]:
        """Get list of forecast horizons.

        Returns:
            List of horizon indices from column names.
        """
        pred_cols = [col for col in self.combined_predictions.columns if col.startswith("pred_h")]
        return [int(col.replace("pred_h", "")) for col in pred_cols]

    def get_model_contribution(self, model_name: str) -> pd.DataFrame:
        """Get specific model's contribution to combination.

        Args:
            model_name: Name of the model.

        Returns:
            DataFrame with the model's weighted contribution.

        Raises:
            ValueError: If model contributions not available.
            KeyError: If model name not found.
        """
        if self.model_contributions is None:
            msg = "Model contributions not available (set store_contributions=True in config)"
            raise ValueError(msg)

        if model_name not in self.model_contributions:
            msg = f"Model {model_name!r} not in contributions. Available: {list(self.model_contributions.keys())}"
            raise KeyError(msg)

        return self.model_contributions[model_name]

    def get_dominant_model(self) -> str:
        """Get the model with highest weight.

        Returns:
            Name of the model with the highest combination weight.
        """
        return max(self.weights, key=self.weights.get)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization.

        Returns:
            Dictionary representation of the result.
        """
        result = {
            "combined_predictions": self.combined_predictions.to_dict(),
            "weights": self.weights.copy(),
            "metadata": self.metadata.to_dict(),
            "has_contributions": self.model_contributions is not None,
            "has_intervals": self.confidence_intervals is not None,
        }

        if self.model_contributions is not None:
            result["model_contributions"] = {
                name: df.to_dict() for name, df in self.model_contributions.items()
            }

        if self.confidence_intervals is not None:
            result["confidence_intervals"] = self.confidence_intervals.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CombinationResult":
        """Create result from dictionary.

        Args:
            data: Dictionary with result values.

        Returns:
            CombinationResult instance.
        """
        combined_predictions = pd.DataFrame(data["combined_predictions"])
        metadata = CombinationMetadata.from_dict(data["metadata"])

        model_contributions = None
        if "model_contributions" in data and data["model_contributions"]:
            model_contributions = {
                name: pd.DataFrame(df_dict) for name, df_dict in data["model_contributions"].items()
            }

        confidence_intervals = None
        if "confidence_intervals" in data and data["confidence_intervals"]:
            confidence_intervals = pd.DataFrame(data["confidence_intervals"])

        return cls(
            combined_predictions=combined_predictions,
            weights=data["weights"],
            metadata=metadata,
            model_contributions=model_contributions,
            confidence_intervals=confidence_intervals,
        )

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the result.
        """
        return (
            f"CombinationResult("
            f"n_models={self.n_models}, "
            f"n_samples={self.n_samples}, "
            f"horizons={self.horizons}, "
            f"dominant={self.get_dominant_model()!r})"
        )
