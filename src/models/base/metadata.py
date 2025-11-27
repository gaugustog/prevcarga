"""Model metadata schema using Pydantic for validation and serialization.

This module defines the ModelMetadata Pydantic model that stores comprehensive
information about trained models including versioning, performance metrics,
feature dependencies, and training configuration.

Example:
    ```python
    from datetime import datetime, timezone
    from src.models.base.metadata import ModelMetadata

    # Create metadata for a trained model
    metadata = ModelMetadata(
        model_id="lgbm_20241124_123456",
        model_name="lgbm",
        model_version="1.0.0",
        model_class="LGBMModel",
        trained_at=datetime.now(timezone.utc),
        supported_horizons=[0, 1, 2, 3],
        feature_dependencies=["hour", "day_of_week", "temperature"],
        performance_metrics={"mape": 2.5, "rmse": 150.0},
        training_config={"max_depth": 7, "learning_rate": 0.1},
    )

    # Serialize to JSON
    json_str = metadata.model_dump_json()

    # Deserialize from dict
    metadata_copy = ModelMetadata(**metadata.model_dump())
    ```
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelMetadata(BaseModel):
    """Comprehensive metadata for trained forecasting models.

    This Pydantic model defines the schema for model metadata, providing
    validation, serialization, and deserialization capabilities. It stores
    information about model identity, versioning, performance, and configuration.

    Attributes:
        model_id: Unique identifier for the model instance (typically includes
                  timestamp or hash). Example: "lgbm_20241124_123456".
        model_name: Human-readable model name (e.g., "lgbm", "random_forest").
        model_version: Semantic version string (e.g., "1.0.0", "2.1.3-beta.1").
        model_class: Fully qualified class name for dynamic loading.
                     Example: "LGBMModel".
        trained_at: UTC timestamp when model training completed.
        registered_at: UTC timestamp when model was registered. Auto-set to
                       current time if not provided.
        supported_horizons: List of forecast horizons this model supports
                            (0 to 8 for D+0 to D+8).
        feature_dependencies: List of feature column names required for
                              predictions. Empty list if no dependencies.
        performance_metrics: Dictionary of performance metrics from validation/test.
                             Keys are metric names (e.g., "mape", "rmse"),
                             values are metric values.
        training_config: Dictionary of training configuration parameters.
                         Can include hyperparameters, data preprocessing settings, etc.
        custom_metadata: Dictionary for extensibility and plugin-specific metadata.
                         Allows storing arbitrary additional information.
    """

    model_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the model instance",
    )
    model_name: str = Field(
        ...,
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Model name (lowercase with underscores)",
    )
    model_version: str = Field(
        ...,
        min_length=1,
        description="Semantic version string (e.g., '1.0.0')",
    )
    model_class: str = Field(
        ...,
        min_length=1,
        description="Class name for dynamic loading",
    )
    trained_at: datetime = Field(
        ...,
        description="UTC timestamp when model training completed",
    )
    registered_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when model was registered",
    )
    supported_horizons: list[int] = Field(
        default_factory=list,
        description="List of forecast horizons (0-8 for D+0 to D+8)",
    )
    feature_dependencies: list[str] = Field(
        default_factory=list,
        description="List of required feature column names",
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of performance metrics",
    )
    training_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Training configuration parameters",
    )
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata for extensibility",
    )

    @field_validator("model_version")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Validate that model_version follows semantic versioning format.

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

    @field_validator("supported_horizons")
    @classmethod
    def validate_horizons(cls, v: list[int]) -> list[int]:
        """Validate that horizons are within valid range (0-8).

        Args:
            v: List of horizon values to validate.

        Returns:
            Validated list of horizons.

        Raises:
            ValueError: If any horizon is outside the valid range.
        """
        if not v:
            logger.warning("Model has no supported horizons defined")
            return v

        invalid = [h for h in v if h < 0 or h > 8]
        if invalid:
            msg = f"Invalid horizons {invalid}. Must be in range 0-8 (D+0 to D+8)"
            raise ValueError(msg)

        # Remove duplicates and sort
        unique_sorted = sorted(set(v))
        if len(unique_sorted) != len(v):
            logger.debug("Removed duplicate horizons: %s -> %s", v, unique_sorted)

        return unique_sorted

    @field_validator("trained_at", "registered_at")
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

    @field_validator("performance_metrics")
    @classmethod
    def validate_metrics(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that all metric values are finite numbers.

        Args:
            v: Dictionary of metrics to validate.

        Returns:
            Validated metrics dictionary.

        Raises:
            ValueError: If any metric value is not a finite number.
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

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary with serialized datetimes.

        Returns:
            Dictionary representation with ISO format timestamps.
        """
        data = self.model_dump()
        # Convert datetimes to ISO format strings
        data["trained_at"] = self.trained_at.isoformat()
        data["registered_at"] = self.registered_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create ModelMetadata instance from dictionary.

        This method handles datetime parsing from ISO format strings.

        Args:
            data: Dictionary containing metadata fields.

        Returns:
            ModelMetadata instance.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Convert ISO format strings to datetime objects if needed
        if isinstance(data.get("trained_at"), str):
            data["trained_at"] = datetime.fromisoformat(data["trained_at"])
        if isinstance(data.get("registered_at"), str):
            data["registered_at"] = datetime.fromisoformat(data["registered_at"])

        return cls(**data)

    def get_version_info(self) -> dict[str, Any]:
        """Get detailed version information.

        Returns:
            Dictionary with version details including major, minor, patch.
        """
        from src.models.base.versioning import SemanticVersion

        version = SemanticVersion(self.model_version)
        return version.to_dict()

    def get_best_metric(self, metric_name: str, lower_is_better: bool = True) -> float | None:
        """Get a specific performance metric value.

        Args:
            metric_name: Name of the metric to retrieve.
            lower_is_better: Whether lower values are better for this metric.
                            Used for logging purposes only.

        Returns:
            Metric value if exists, None otherwise.
        """
        value = self.performance_metrics.get(metric_name)
        if value is not None:
            logger.debug(
                "Metric '%s' for model '%s': %.4f (lower_is_better=%s)",
                metric_name,
                self.model_id,
                value,
                lower_is_better,
            )
        return value

    def update_metrics(self, new_metrics: dict[str, float]) -> None:
        """Update performance metrics with new values.

        Args:
            new_metrics: Dictionary of new metrics to add/update.

        Raises:
            ValueError: If any metric value is not finite.
        """
        # Validate new metrics
        validated = self.validate_metrics(new_metrics)

        # Update existing metrics
        self.performance_metrics.update(validated)
        logger.info("Updated metrics for model '%s': %s", self.model_id, list(validated.keys()))

    def __str__(self) -> str:
        """Return human-readable string representation.

        Returns:
            String with model ID, name, and version.
        """
        return f"{self.model_name} v{self.model_version} (ID: {self.model_id})"

    def __repr__(self) -> str:
        """Return detailed string representation for debugging.

        Returns:
            String representation with key metadata fields.
        """
        return (
            f"ModelMetadata(id='{self.model_id}', "
            f"name='{self.model_name}', "
            f"version='{self.model_version}', "
            f"horizons={self.supported_horizons})"
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
    )
