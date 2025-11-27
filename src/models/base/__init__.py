"""Base model infrastructure for PrevCarga forecasting system.

This package provides the foundational components for model management:
- BaseModel: Abstract base class for all forecasting models
- ModelMetadata: Pydantic schema for model metadata
- SemanticVersion: Version parsing and comparison
- ModelRegistry: Centralized model management with persistence
- Model class registration utilities

Example:
    ```python
    from src.models.base import (
        BaseModel,
        ModelMetadata,
        ModelRegistry,
        SemanticVersion,
        register_model_class,
    )

    # Register a model class
    @register_model_class("my_model")
    class MyModel(BaseModel):
        ...

    # Create a registry
    registry = ModelRegistry("models/registry")

    # Register a trained model
    model_id = registry.register_model(trained_model)

    # Load a model
    loaded_model = registry.load_model(model_id)
    ```
"""

from src.models.base.metadata import ModelMetadata
from src.models.base.model import BaseModel
from src.models.base.registry import (
    ModelRegistry,
    get_model_class,
    register_model_class,
)
from src.models.base.versioning import SemanticVersion

__all__ = [
    "BaseModel",
    "ModelMetadata",
    "ModelRegistry",
    "SemanticVersion",
    "get_model_class",
    "register_model_class",
]
