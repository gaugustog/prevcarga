"""Serialization metadata schemas using Pydantic for validation.

This module defines Pydantic models for capturing comprehensive metadata during
model serialization. It includes schemas for serialization metadata (file format,
compression, checksums) and training metadata (training statistics, performance
metrics, hyperparameters).

Example:
    ```python
    from src.models.serialization.schemas import (
        SerializedModelMetadata,
        TrainingMetadata,
    )
    from datetime import datetime, UTC
    from uuid import uuid4

    # Create training metadata
    training_meta = TrainingMetadata(
        n_samples_train=10000,
        n_samples_val=2000,
        n_features=50,
        feature_names=["hour", "day_of_week", "temperature"],
        training_config={"learning_rate": 0.1, "max_depth": 7},
        preprocessing_steps=["standardization", "outlier_removal"],
        training_duration_seconds=120.5,
        performance_metrics={"mape": 2.5, "rmse": 150.0, "mae": 100.0},
        cv_scores={"mean_cv_mape": 2.8, "std_cv_mape": 0.3},
        hyperparameter_optimization=True,
        best_hyperparams={"learning_rate": 0.08, "max_depth": 8},
    )

    # Create serialization metadata
    serial_meta = SerializedModelMetadata(
        model_id=str(uuid4()),
        model_type="lgbm",
        model_name="lgbm_production_v1",
        version="1.0.0",
        created_at=datetime.now(UTC),
        serialized_at=datetime.now(UTC),
        serialization_format="joblib",
        compression="gzip",
        compression_level=6,
        checksum="abc123...",
        file_size_bytes=1024000,
        python_version="3.12.0",
        dependencies={"lightgbm": "4.1.0", "pandas": "2.0.0"},
    )
    ```
"""

import sys
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingMetadata(BaseModel):
    """Comprehensive training metadata for model serialization.

    This Pydantic model captures detailed information about the model training
    process including dataset statistics, training configuration, performance
    metrics, cross-validation results, and hyperparameter optimization details.

    Attributes:
        n_samples_train: Number of samples in training set.
        n_samples_val: Number of samples in validation set.
        n_features: Number of features used for training.
        feature_names: List of feature column names.
        training_config: Dictionary of training configuration parameters
                        (hyperparameters, optimization settings, etc.).
        preprocessing_steps: List of preprocessing steps applied to data.
        training_duration_seconds: Total training duration in seconds.
        performance_metrics: Dictionary of performance metrics on validation/test set.
                            Common keys: "mape", "rmse", "mae", "r2".
        cv_scores: Dictionary of cross-validation scores.
                  Common keys: "mean_cv_mape", "std_cv_mape", etc.
        hyperparameter_optimization: Whether hyperparameter optimization was used.
        best_hyperparams: Dictionary of best hyperparameters found during optimization.
                         Empty dict if hyperparameter_optimization is False.
    """

    n_samples_train: int = Field(
        ...,
        ge=1,
        description="Number of samples in training set",
    )
    n_samples_val: int = Field(
        ...,
        ge=0,
        description="Number of samples in validation set",
    )
    n_features: int = Field(
        ...,
        ge=1,
        description="Number of features used for training",
    )
    feature_names: list[str] = Field(
        default_factory=list,
        description="List of feature column names",
    )
    training_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Training configuration parameters",
    )
    preprocessing_steps: list[str] = Field(
        default_factory=list,
        description="List of preprocessing steps applied",
    )
    training_duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total training duration in seconds",
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics (MAPE, RMSE, MAE, etc.)",
    )
    cv_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Cross-validation scores",
    )
    hyperparameter_optimization: bool = Field(
        default=False,
        description="Whether hyperparameter optimization was used",
    )
    best_hyperparams: dict[str, Any] = Field(
        default_factory=dict,
        description="Best hyperparameters from optimization",
    )

    @field_validator("feature_names")
    @classmethod
    def validate_feature_names(cls, v: list[str]) -> list[str]:
        """Validate feature names list.

        Args:
            v: List of feature names to validate.

        Returns:
            Validated feature names list.

        Raises:
            ValueError: If feature names contain duplicates or empty strings.
        """
        if not v:
            logger.warning("No feature names provided in training metadata")
            return v

        # Check for empty strings
        if any(not name.strip() for name in v):
            msg = "Feature names cannot be empty strings"
            raise ValueError(msg)

        # Check for duplicates
        if len(v) != len(set(v)):
            duplicates = [name for name in v if v.count(name) > 1]
            msg = f"Duplicate feature names found: {set(duplicates)}"
            raise ValueError(msg)

        return v

    @field_validator("performance_metrics", "cv_scores")
    @classmethod
    def validate_metrics(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that all metric values are finite numbers.

        Args:
            v: Dictionary of metrics to validate.

        Returns:
            Validated metrics dictionary.

        Raises:
            ValueError: If any metric value is not finite.
        """
        import math

        for metric_name, metric_value in v.items():
            if not isinstance(metric_value, (int, float)):
                msg = f"Metric '{metric_name}' has non-numeric value: {metric_value}"
                raise ValueError(msg)

            if not math.isfinite(metric_value):
                msg = f"Metric '{metric_name}' has non-finite value: {metric_value}"
                raise ValueError(msg)

        return v

    @field_validator("best_hyperparams")
    @classmethod
    def validate_best_hyperparams(cls, v: dict[str, Any], info) -> dict[str, Any]:
        """Validate best_hyperparams consistency with hyperparameter_optimization.

        Args:
            v: Best hyperparameters dict to validate.
            info: Validation context with other field values.

        Returns:
            Validated best_hyperparams dict.
        """
        # Get hyperparameter_optimization value from validation context
        data = info.data
        hyperparameter_optimization = data.get("hyperparameter_optimization", False)

        if hyperparameter_optimization and not v:
            logger.warning("hyperparameter_optimization=True but best_hyperparams is empty")
        elif not hyperparameter_optimization and v:
            logger.warning("hyperparameter_optimization=False but best_hyperparams is not empty")

        return v


class SerializedModelMetadata(BaseModel):
    """Comprehensive serialization metadata for saved models.

    This Pydantic model captures complete information about the serialized model
    file including model identity, versioning, serialization format, compression
    settings, integrity checksums, and Python environment dependencies.

    Attributes:
        model_id: Unique identifier for the model (UUID format).
        model_type: Model type identifier (e.g., "lgbm", "rf", "regdin_svm").
        model_name: Human-readable model name.
        version: Semantic version string (MAJOR.MINOR.PATCH).
        created_at: UTC timestamp when model was first created/trained.
        serialized_at: UTC timestamp when model was serialized to disk.
        serialization_format: Serialization format used (default: "joblib").
        compression: Compression algorithm (gzip, lz4, zstd, or None).
        compression_level: Compression level (0-9, where 0=no compression, 9=max).
        checksum: SHA256 checksum of serialized model file for integrity validation.
        file_size_bytes: Size of serialized model file in bytes.
        python_version: Python version used during serialization (e.g., "3.12.0").
        dependencies: Dictionary of package names to version strings.
    """

    model_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique model identifier (UUID)",
    )
    model_type: str = Field(
        ...,
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Model type identifier (lowercase with underscores)",
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Human-readable model name",
    )
    version: str = Field(
        ...,
        min_length=1,
        description="Semantic version string (e.g., '1.0.0')",
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when model was created/trained",
    )
    serialized_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when model was serialized",
    )
    serialization_format: str = Field(
        default="joblib",
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Serialization format (e.g., 'joblib', 'pickle')",
    )
    compression: str | None = Field(
        default="gzip",
        description="Compression algorithm (gzip, lz4, zstd, or None)",
    )
    compression_level: int = Field(
        default=6,
        ge=0,
        le=9,
        description="Compression level (0=none, 9=maximum)",
    )
    checksum: str = Field(
        ...,
        min_length=1,
        description="SHA256 checksum for integrity validation",
    )
    file_size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of serialized model file in bytes",
    )
    python_version: str = Field(
        default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        description="Python version used during serialization",
    )
    dependencies: dict[str, str] = Field(
        default_factory=dict,
        description="Package dependencies (name -> version)",
    )

    @field_validator("version")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Validate that version follows semantic versioning format.

        Args:
            v: Version string to validate.

        Returns:
            Validated version string.

        Raises:
            ValueError: If version string format is invalid.
        """
        # Import here to avoid circular dependency
        from src.models.base.versioning import SemanticVersion

        try:
            SemanticVersion(v)
        except ValueError as e:
            msg = f"Invalid semantic version: {v}"
            raise ValueError(msg) from e

        return v

    @field_validator("created_at", "serialized_at")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        """Ensure datetime has timezone information (prefer UTC).

        Args:
            v: Datetime to validate.

        Returns:
            Datetime with timezone information.
        """
        if v.tzinfo is None:
            logger.warning("Datetime has no timezone, assuming UTC: %s", v)
            return v.replace(tzinfo=UTC)
        return v

    @field_validator("compression")
    @classmethod
    def validate_compression(cls, v: str | None) -> str | None:
        """Validate compression algorithm is supported.

        Args:
            v: Compression algorithm to validate.

        Returns:
            Validated compression algorithm.

        Raises:
            ValueError: If compression algorithm is not supported.
        """
        if v is None:
            return v

        supported_compression = {"gzip", "lz4", "zstd", "none"}
        if v not in supported_compression:
            msg = f"Unsupported compression algorithm: {v}. " f"Supported: {supported_compression}"
            raise ValueError(msg)

        return v

    @field_validator("compression_level")
    @classmethod
    def validate_compression_level(cls, v: int, info) -> int:
        """Validate compression level is appropriate for compression algorithm.

        Args:
            v: Compression level to validate.
            info: Validation context with other field values.

        Returns:
            Validated compression level.
        """
        data = info.data
        compression = data.get("compression")

        if compression in (None, "none") and v > 0:
            logger.warning("compression=None but compression_level=%d (will be ignored)", v)

        return v

    @field_validator("checksum")
    @classmethod
    def validate_checksum_format(cls, v: str) -> str:
        """Validate checksum is a valid SHA256 hash.

        Args:
            v: Checksum string to validate.

        Returns:
            Validated checksum string.

        Raises:
            ValueError: If checksum format is invalid.
        """
        # SHA256 produces 64 hexadecimal characters
        if len(v) != 64:
            msg = f"Invalid SHA256 checksum length: {len(v)} (expected 64)"
            raise ValueError(msg)

        # Check if all characters are hexadecimal
        try:
            int(v, 16)
        except ValueError as e:
            msg = f"Invalid SHA256 checksum format: {v} (not hexadecimal)"
            raise ValueError(msg) from e

        return v.lower()  # Normalize to lowercase

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary with serialized datetimes.

        Returns:
            Dictionary representation with ISO format timestamps.
        """
        data = self.model_dump()
        # Convert datetimes to ISO format strings
        data["created_at"] = self.created_at.isoformat()
        data["serialized_at"] = self.serialized_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SerializedModelMetadata":
        """Create SerializedModelMetadata instance from dictionary.

        This method handles datetime parsing from ISO format strings.

        Args:
            data: Dictionary containing metadata fields.

        Returns:
            SerializedModelMetadata instance.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Convert ISO format strings to datetime objects if needed
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("serialized_at"), str):
            data["serialized_at"] = datetime.fromisoformat(data["serialized_at"])

        return cls(**data)
