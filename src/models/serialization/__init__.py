"""Model serialization with metadata and version management.

This package provides comprehensive model serialization capabilities including:
- Metadata schemas for serialization and training information
- Model serialization with compression and integrity validation
- Semantic version management and comparison
- Version history tracking across saved models

Key Components:
    - SerializedModelMetadata: Pydantic schema for serialization metadata
    - TrainingMetadata: Pydantic schema for training information
    - ModelSerializer: Serialize/deserialize models with metadata
    - VersionManager: Parse, compare, and manage semantic versions

Example:
    ```python
    from src.models.serialization import (
        ModelSerializer,
        VersionManager,
        TrainingMetadata,
        SerializedModelMetadata,
    )

    # Create training metadata
    training_meta = TrainingMetadata(
        n_samples_train=10000,
        n_samples_val=2000,
        n_features=50,
        feature_names=["hour", "day_of_week"],
        training_config={"learning_rate": 0.1},
        preprocessing_steps=["standardization"],
        training_duration_seconds=120.0,
        performance_metrics={"mape": 2.5, "rmse": 150.0},
    )

    # Save model with metadata
    serializer = ModelSerializer()
    serializer.save_model(
        model=my_model,
        path="models/lgbm_v1.0.0.pkl",
        training_metadata=training_meta,
        version="1.0.0",
        compression="gzip",
    )

    # Load model
    model, metadata = serializer.load_model("models/lgbm_v1.0.0.pkl")

    # Version management
    vm = VersionManager()
    latest = vm.get_latest_version("models/", model_type="lgbm")
    new_version = vm.increment_version(latest, bump="minor")
    ```
"""

from src.models.serialization.model_serializer import ModelSerializer
from src.models.serialization.schemas import (
    SerializedModelMetadata,
    TrainingMetadata,
)
from src.models.serialization.version_manager import VersionManager

__all__ = [
    "ModelSerializer",
    "SerializedModelMetadata",
    "TrainingMetadata",
    "VersionManager",
]
