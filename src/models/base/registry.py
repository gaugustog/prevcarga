"""Model registry for centralized model management and persistence.

This module provides the ModelRegistry class for managing trained models,
including registration, loading, listing, comparison, and persistence.
It supports thread-safe operations, in-memory caching, and model versioning.

Example:
    ```python
    from src.models.base.registry import ModelRegistry, register_model_class
    from src.models.base.model import BaseModel

    # Register model class
    @register_model_class("lgbm")
    class LGBMModel(BaseModel):
        ...

    # Create registry instance
    registry = ModelRegistry(base_path="models/registry")

    # Register a trained model
    model_id = registry.register_model(trained_model, metadata)

    # Load model
    loaded_model = registry.load_model(model_id)

    # List models with filters
    models = registry.list_models(model_name="lgbm", version_constraint=">=1.0.0")

    # Get best model by metric
    best_id = registry.get_best_model(metric="mape", model_name="lgbm")

    # Compare models
    comparison = registry.compare_models([model_id_1, model_id_2])
    ```
"""

import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.base.metadata import ModelMetadata
from src.models.base.model import BaseModel
from src.models.base.versioning import SemanticVersion
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global model class registry
_MODEL_CLASS_REGISTRY: dict[str, type[BaseModel]] = {}
_REGISTRY_LOCK = threading.Lock()


def register_model_class(model_name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Decorator to register a model class with the global registry.

    This decorator allows models to be dynamically loaded by name.

    Args:
        model_name: Unique name for the model class (e.g., "lgbm").

    Returns:
        Decorator function that registers the class.

    Example:
        ```python
        @register_model_class("lgbm")
        class LGBMModel(BaseModel):
            ...
        ```
    """

    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        with _REGISTRY_LOCK:
            if model_name in _MODEL_CLASS_REGISTRY:
                logger.warning(
                    "Overwriting existing model class registration: %s",
                    model_name,
                )
            _MODEL_CLASS_REGISTRY[model_name] = cls
            logger.debug("Registered model class: %s -> %s", model_name, cls.__name__)
        return cls

    return decorator


def get_model_class(model_name: str) -> type[BaseModel]:
    """Get registered model class by name.

    Args:
        model_name: Name of the model class to retrieve.

    Returns:
        Model class type.

    Raises:
        KeyError: If model class is not registered.
    """
    with _REGISTRY_LOCK:
        if model_name not in _MODEL_CLASS_REGISTRY:
            available = list(_MODEL_CLASS_REGISTRY.keys())
            msg = f"Unknown model class: {model_name}. Available: {available}"
            raise KeyError(msg)
        return _MODEL_CLASS_REGISTRY[model_name]


class ModelRegistry:
    """Registry for managing trained forecasting models.

    This class provides centralized management of trained models including:
    - Thread-safe registration and loading
    - Persistent storage with metadata
    - Version-based filtering and comparison
    - In-memory caching for performance
    - Model lifecycle management

    The registry stores:
    - Model files (.joblib) in base_path/models/
    - Metadata files (.json) in base_path/metadata/
    - Index file (index.json) in base_path/

    Attributes:
        base_path: Root directory for model storage.
        _index: In-memory index mapping model_id to metadata.
        _model_cache: LRU cache of loaded models.
        _lock: Thread lock for safe concurrent access.
    """

    def __init__(
        self,
        base_path: str | Path = "models/registry",
        cache_size: int = 10,
    ) -> None:
        """Initialize model registry.

        Args:
            base_path: Root directory for model storage. Created if doesn't exist.
            cache_size: Maximum number of models to keep in memory cache.
        """
        self.base_path = Path(base_path)
        self._cache_size = cache_size

        # Create directory structure
        self._models_dir = self.base_path / "models"
        self._metadata_dir = self.base_path / "metadata"
        self._index_file = self.base_path / "index.json"

        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_dir.mkdir(parents=True, exist_ok=True)

        # Initialize index and caches
        self._index: dict[str, ModelMetadata] = {}
        self._model_cache: dict[str, BaseModel] = {}
        self._cache_order: list[str] = []  # For LRU eviction
        self._lock = threading.Lock()

        # Load existing index
        self._load_index()

        logger.info(
            "Initialized ModelRegistry at %s with %d models",
            self.base_path,
            len(self._index),
        )

    def _load_index(self) -> None:
        """Load registry index from disk."""
        if not self._index_file.exists():
            logger.debug("No existing index file, starting with empty registry")
            return

        try:
            with self._index_file.open() as f:
                index_data = json.load(f)

            for model_id, metadata_dict in index_data.items():
                try:
                    metadata = ModelMetadata.from_dict(metadata_dict)
                    self._index[model_id] = metadata
                except Exception as e:
                    logger.error("Failed to load metadata for %s: %s", model_id, e)

            logger.info("Loaded registry index with %d models", len(self._index))
        except Exception as e:
            logger.error("Failed to load registry index: %s", e)
            self._index = {}

    def _save_index(self) -> None:
        """Save registry index to disk."""
        try:
            index_data = {
                model_id: metadata.to_dict() for model_id, metadata in self._index.items()
            }

            with self._index_file.open("w") as f:
                json.dump(index_data, f, indent=2)

            logger.debug("Saved registry index with %d models", len(self._index))
        except Exception as e:
            logger.error("Failed to save registry index: %s", e)

    def _get_model_path(self, model_id: str) -> Path:
        """Get file path for model pickle file.

        Args:
            model_id: Model identifier.

        Returns:
            Path to model file.
        """
        return self._models_dir / f"{model_id}.joblib"

    def _get_metadata_path(self, model_id: str) -> Path:
        """Get file path for metadata JSON file.

        Args:
            model_id: Model identifier.

        Returns:
            Path to metadata file.
        """
        return self._metadata_dir / f"{model_id}.json"

    def _evict_cache(self) -> None:
        """Evict oldest model from cache if at capacity."""
        if len(self._model_cache) >= self._cache_size and self._cache_order:
            oldest_id = self._cache_order.pop(0)
            if oldest_id in self._model_cache:
                del self._model_cache[oldest_id]
                logger.debug("Evicted model %s from cache", oldest_id)

    def register_model(
        self,
        model: BaseModel,
        metadata: ModelMetadata | None = None,
    ) -> str:
        """Register a trained model in the registry.

        This method:
        1. Validates the model is fitted
        2. Generates or validates metadata
        3. Saves model to disk
        4. Updates registry index
        5. Adds model to cache

        Args:
            model: Trained model instance to register.
            metadata: Optional metadata. If None, generated from model.

        Returns:
            Model ID assigned to the registered model.

        Raises:
            ValueError: If model is not fitted or metadata is invalid.
            OSError: If model cannot be saved to disk.
        """
        if not model.is_fitted():
            msg = f"Cannot register unfitted model: {model.name}"
            raise ValueError(msg)

        # Generate metadata if not provided
        if metadata is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            model_id = f"{model.name}_{timestamp}"
            metadata = model.get_metadata(model_id=model_id)
        else:
            model_id = metadata.model_id

        with self._lock:
            # Check if model_id already exists
            if model_id in self._index:
                logger.warning("Overwriting existing model: %s", model_id)

            # Save model to disk
            model_path = self._get_model_path(model_id)
            model.save(model_path)

            # Save metadata to disk
            metadata_path = self._get_metadata_path(model_id)
            with metadata_path.open("w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Update index
            self._index[model_id] = metadata
            self._save_index()

            # Add to cache
            self._evict_cache()
            self._model_cache[model_id] = model
            if model_id in self._cache_order:
                self._cache_order.remove(model_id)
            self._cache_order.append(model_id)

        logger.info("Registered model: %s (version %s)", model_id, metadata.model_version)
        return model_id

    def load_model(
        self,
        model_id: str,
        use_cache: bool = True,
    ) -> BaseModel:
        """Load a model from the registry.

        Args:
            model_id: Model identifier.
            use_cache: Whether to use in-memory cache. If False, always load from disk.

        Returns:
            Loaded model instance.

        Raises:
            KeyError: If model_id is not in registry.
            FileNotFoundError: If model file is missing.
            OSError: If model cannot be loaded.
        """
        with self._lock:
            if model_id not in self._index:
                msg = f"Model not found in registry: {model_id}"
                raise KeyError(msg)

            # Check cache first
            if use_cache and model_id in self._model_cache:
                logger.debug("Loading model %s from cache", model_id)
                # Update cache order (LRU)
                self._cache_order.remove(model_id)
                self._cache_order.append(model_id)
                return self._model_cache[model_id]

            # Load from disk
            model_path = self._get_model_path(model_id)
            if not model_path.exists():
                msg = f"Model file not found: {model_path}"
                raise FileNotFoundError(msg)

            model = BaseModel.load(model_path)

            # Add to cache
            if use_cache:
                self._evict_cache()
                self._model_cache[model_id] = model
                if model_id in self._cache_order:
                    self._cache_order.remove(model_id)
                self._cache_order.append(model_id)

            logger.info("Loaded model: %s", model_id)
            return model

    def list_models(
        self,
        model_name: str | None = None,
        version_constraint: str | None = None,
        **filters: Any,
    ) -> list[ModelMetadata]:
        """List models in the registry with optional filtering.

        Args:
            model_name: Filter by model name (exact match).
            version_constraint: Filter by version constraint (e.g., ">=1.0.0").
            **filters: Additional filters on metadata fields.
                      Example: min_horizon=0, metric_threshold={"mape": 5.0}

        Returns:
            List of ModelMetadata matching the filters, sorted by registration time.

        Example:
            ```python
            # Get all LGBM models version >= 1.0.0
            models = registry.list_models(
                model_name="lgbm",
                version_constraint=">=1.0.0"
            )

            # Get models supporting specific horizon
            models = registry.list_models(
                model_name="lgbm",
                min_horizon=3
            )
            ```
        """
        with self._lock:
            results = list(self._index.values())

        # Filter by model name
        if model_name is not None:
            results = [m for m in results if m.model_name == model_name]

        # Filter by version constraint
        if version_constraint is not None:
            results = [
                m for m in results if SemanticVersion(m.model_version).satisfies(version_constraint)
            ]

        # Apply custom filters
        for key, value in filters.items():
            if key == "min_horizon":
                # Filter models supporting at least this horizon
                results = [m for m in results if value in m.supported_horizons]
            elif key == "has_feature":
                # Filter models with specific feature dependency
                results = [m for m in results if value in m.feature_dependencies]
            elif key == "metric_threshold":
                # Filter by metric thresholds (dict of metric: max_value)
                if isinstance(value, dict):
                    for metric, threshold in value.items():
                        results = [
                            m
                            for m in results
                            if m.performance_metrics.get(metric, float("inf")) <= threshold
                        ]

        # Sort by registration time (newest first)
        results.sort(key=lambda m: m.registered_at, reverse=True)

        logger.debug(
            "Listed %d models with filters: name=%s, version=%s, filters=%s",
            len(results),
            model_name,
            version_constraint,
            filters,
        )

        return results

    def get_best_model(
        self,
        metric: str,
        model_name: str | None = None,
        lower_is_better: bool = True,
        **filters: Any,
    ) -> str | None:
        """Get the best model ID based on a performance metric.

        Args:
            metric: Metric name to optimize (e.g., "mape", "rmse").
            model_name: Optional filter by model name.
            lower_is_better: Whether lower metric values are better.
            **filters: Additional filters passed to list_models().

        Returns:
            Model ID of the best model, or None if no models match filters.

        Example:
            ```python
            # Get best LGBM model by MAPE
            best_id = registry.get_best_model(
                metric="mape",
                model_name="lgbm",
                lower_is_better=True
            )
            ```
        """
        models = self.list_models(model_name=model_name, **filters)

        # Filter models that have the metric
        models_with_metric = [m for m in models if metric in m.performance_metrics]

        if not models_with_metric:
            logger.warning(
                "No models found with metric '%s' (name=%s, filters=%s)",
                metric,
                model_name,
                filters,
            )
            return None

        # Find best model
        if lower_is_better:
            best_model = min(
                models_with_metric,
                key=lambda m: m.performance_metrics[metric],
            )
        else:
            best_model = max(
                models_with_metric,
                key=lambda m: m.performance_metrics[metric],
            )

        logger.info(
            "Best model by %s: %s (value=%.4f)",
            metric,
            best_model.model_id,
            best_model.performance_metrics[metric],
        )

        return best_model.model_id

    def compare_models(self, model_ids: list[str]) -> pd.DataFrame:
        """Compare multiple models by their metadata and metrics.

        Args:
            model_ids: List of model IDs to compare.

        Returns:
            DataFrame with model comparison. Rows are models, columns include
            metadata fields and all available metrics.

        Raises:
            KeyError: If any model_id is not in registry.

        Example:
            ```python
            comparison = registry.compare_models([id1, id2, id3])
            print(comparison[["model_name", "model_version", "mape", "rmse"]])
            ```
        """
        with self._lock:
            metadata_list = []
            for model_id in model_ids:
                if model_id not in self._index:
                    msg = f"Model not found in registry: {model_id}"
                    raise KeyError(msg)
                metadata_list.append(self._index[model_id])

        # Build comparison dataframe
        rows = []
        for metadata in metadata_list:
            row = {
                "model_id": metadata.model_id,
                "model_name": metadata.model_name,
                "model_version": metadata.model_version,
                "model_class": metadata.model_class,
                "trained_at": metadata.trained_at,
                "registered_at": metadata.registered_at,
                "supported_horizons": len(metadata.supported_horizons),
                "num_features": len(metadata.feature_dependencies),
            }

            # Add all performance metrics
            row.update(metadata.performance_metrics)

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index("model_id")

        logger.info("Compared %d models", len(model_ids))
        return df

    def deregister_model(
        self,
        model_id: str,
        delete_files: bool = False,
    ) -> None:
        """Remove a model from the registry.

        Args:
            model_id: Model identifier to remove.
            delete_files: Whether to delete model and metadata files from disk.
                         If False, only removes from index and cache.

        Raises:
            KeyError: If model_id is not in registry.
        """
        with self._lock:
            if model_id not in self._index:
                msg = f"Model not found in registry: {model_id}"
                raise KeyError(msg)

            # Remove from index
            del self._index[model_id]
            self._save_index()

            # Remove from cache
            if model_id in self._model_cache:
                del self._model_cache[model_id]
            if model_id in self._cache_order:
                self._cache_order.remove(model_id)

            # Delete files if requested
            if delete_files:
                model_path = self._get_model_path(model_id)
                metadata_path = self._get_metadata_path(model_id)

                if model_path.exists():
                    model_path.unlink()
                    logger.debug("Deleted model file: %s", model_path)

                if metadata_path.exists():
                    metadata_path.unlink()
                    logger.debug("Deleted metadata file: %s", metadata_path)

        logger.info("Deregistered model: %s (delete_files=%s)", model_id, delete_files)

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get metadata for a specific model.

        Args:
            model_id: Model identifier.

        Returns:
            ModelMetadata instance.

        Raises:
            KeyError: If model_id is not in registry.
        """
        with self._lock:
            if model_id not in self._index:
                msg = f"Model not found in registry: {model_id}"
                raise KeyError(msg)
            return self._index[model_id]

    def __len__(self) -> int:
        """Return number of models in registry.

        Returns:
            Count of registered models.
        """
        return len(self._index)

    def __contains__(self, model_id: str) -> bool:
        """Check if model_id exists in registry.

        Args:
            model_id: Model identifier to check.

        Returns:
            True if model exists in registry, False otherwise.
        """
        return model_id in self._index

    def __repr__(self) -> str:
        """Return string representation of registry.

        Returns:
            String with registry path and model count.
        """
        return f"ModelRegistry(path='{self.base_path}', models={len(self._index)})"
