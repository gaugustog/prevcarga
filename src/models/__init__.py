"""Forecasting models (end-to-end and hierarchical).

This package provides the complete model infrastructure for PrevCarga:
- Base model classes and interfaces
- Model implementations (LGBM, Random Forest, etc.)
- Model registry and management
- Model versioning and metadata

Example:
    ```python
    from src.models.base import BaseModel, ModelRegistry, register_model_class
    from src.models.config import LGBMConfig
    from src.models.end_to_end import LGBMModel

    # Access base infrastructure
    registry = ModelRegistry("models/registry")

    # Use model implementations
    model = LGBMModel()
    config = LGBMConfig()
    ```
"""

from src.models.base import (
    BaseModel,
    ModelMetadata,
    ModelRegistry,
    SemanticVersion,
    get_model_class,
    register_model_class,
)
from src.models.config import LGBMConfig
from src.models.end_to_end import LGBMModel

__all__ = [
    "BaseModel",
    "LGBMConfig",
    "LGBMModel",
    "ModelMetadata",
    "ModelRegistry",
    "SemanticVersion",
    "get_model_class",
    "register_model_class",
]
