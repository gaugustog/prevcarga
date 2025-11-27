"""Tests for ModelRegistry class."""

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from src.models.base.metadata import ModelMetadata
from src.models.base.model import BaseModel
from src.models.base.registry import (
    ModelRegistry,
    get_model_class,
    register_model_class,
)


# Mock models for testing
class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, model_name: str = "mock", version: str = "1.0.0"):
        super().__init__()
        self._name = model_name
        self._version = version

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def supported_horizons(self) -> list[int]:
        return [0, 1, 2, 3]

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame, config: dict[str, Any]) -> None:
        self._feature_names = list(X.columns)
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_config": config,
            "performance_metrics": config.get("metrics", {"mape": 5.0}),
        }
        self._is_fitted = True

    def predict(self, X: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
        self._check_is_fitted()
        if horizons is None:
            horizons = self.supported_horizons
        return pd.DataFrame(index=X.index, columns=[f"h{h}" for h in horizons], data=100.0)

    def get_feature_importance(self) -> dict[str, float]:
        self._check_is_fitted()
        n = len(self._feature_names)
        return dict.fromkeys(self._feature_names, 1.0 / n)


@pytest.fixture
def temp_registry_path(tmp_path):
    """Create temporary registry path."""
    return tmp_path / "test_registry"


@pytest.fixture
def fitted_model():
    """Create a fitted mock model."""
    model = MockModel(model_name="test_model", version="1.0.0")
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = pd.Series([10, 11, 12])
    model.fit(X, y, {"lr": 0.1, "metrics": {"mape": 2.5, "rmse": 150.0}})
    return model


class TestModelRegistryInitialization:
    """Test ModelRegistry initialization."""

    def test_init_creates_directories(self, temp_registry_path):
        """Test that registry creates necessary directories."""
        registry = ModelRegistry(base_path=temp_registry_path)

        assert temp_registry_path.exists()
        assert (temp_registry_path / "models").exists()
        assert (temp_registry_path / "metadata").exists()
        assert isinstance(registry, ModelRegistry)

    def test_init_empty_registry(self, temp_registry_path):
        """Test initialization of empty registry."""
        registry = ModelRegistry(base_path=temp_registry_path)

        assert len(registry) == 0
        assert not (temp_registry_path / "index.json").exists()

    def test_init_loads_existing_index(self, temp_registry_path, fitted_model):
        """Test that registry loads existing index on initialization."""
        # Create registry and register a model
        registry1 = ModelRegistry(base_path=temp_registry_path)
        model_id = registry1.register_model(fitted_model)

        # Create new registry instance (should load from disk)
        registry2 = ModelRegistry(base_path=temp_registry_path)

        assert len(registry2) == 1
        assert model_id in registry2


class TestModelRegistryRegistration:
    """Test model registration."""

    def test_register_fitted_model(self, temp_registry_path, fitted_model):
        """Test registering a fitted model."""
        registry = ModelRegistry(base_path=temp_registry_path)

        model_id = registry.register_model(fitted_model)

        assert isinstance(model_id, str)
        assert "test_model_" in model_id
        assert len(registry) == 1
        assert model_id in registry

    def test_register_unfitted_model_raises_error(self, temp_registry_path):
        """Test that registering unfitted model raises error."""
        registry = ModelRegistry(base_path=temp_registry_path)
        unfitted_model = MockModel()

        with pytest.raises(ValueError, match="Cannot register unfitted model"):
            registry.register_model(unfitted_model)

    def test_register_with_custom_metadata(self, temp_registry_path, fitted_model):
        """Test registering model with custom metadata."""
        registry = ModelRegistry(base_path=temp_registry_path)

        metadata = ModelMetadata(
            model_id="custom_id_001",
            model_name="test_model",
            model_version="1.0.0",
            model_class="MockModel",
            trained_at=datetime.now(UTC),
            supported_horizons=[0, 1, 2],
            performance_metrics={"mape": 1.5},
        )

        model_id = registry.register_model(fitted_model, metadata=metadata)

        assert model_id == "custom_id_001"
        loaded_metadata = registry.get_metadata(model_id)
        assert loaded_metadata.performance_metrics["mape"] == 1.5

    def test_register_creates_files(self, temp_registry_path, fitted_model):
        """Test that registration creates model and metadata files."""
        registry = ModelRegistry(base_path=temp_registry_path)

        model_id = registry.register_model(fitted_model)

        model_file = temp_registry_path / "models" / f"{model_id}.joblib"
        metadata_file = temp_registry_path / "metadata" / f"{model_id}.json"
        index_file = temp_registry_path / "index.json"

        assert model_file.exists()
        assert metadata_file.exists()
        assert index_file.exists()

    def test_register_overwrites_existing(self, temp_registry_path, fitted_model):
        """Test that registering with same ID overwrites existing."""
        registry = ModelRegistry(base_path=temp_registry_path)

        # Register first time
        metadata1 = ModelMetadata(
            model_id="duplicate_id",
            model_name="test_model",
            model_version="1.0.0",
            model_class="MockModel",
            trained_at=datetime.now(UTC),
            performance_metrics={"mape": 5.0},
        )
        registry.register_model(fitted_model, metadata=metadata1)

        # Register again with same ID
        metadata2 = ModelMetadata(
            model_id="duplicate_id",
            model_name="test_model",
            model_version="1.0.0",
            model_class="MockModel",
            trained_at=datetime.now(UTC),
            performance_metrics={"mape": 2.0},
        )
        registry.register_model(fitted_model, metadata=metadata2)

        # Should have only one model
        assert len(registry) == 1
        loaded_metadata = registry.get_metadata("duplicate_id")
        assert loaded_metadata.performance_metrics["mape"] == 2.0


class TestModelRegistryLoading:
    """Test model loading."""

    def test_load_registered_model(self, temp_registry_path, fitted_model):
        """Test loading a registered model."""
        registry = ModelRegistry(base_path=temp_registry_path)
        model_id = registry.register_model(fitted_model)

        loaded_model = registry.load_model(model_id)

        assert isinstance(loaded_model, BaseModel)
        assert loaded_model.is_fitted()
        assert loaded_model.name == "test_model"
        assert loaded_model.get_feature_names() == ["a", "b"]

    def test_load_nonexistent_model_raises_error(self, temp_registry_path):
        """Test that loading nonexistent model raises error."""
        registry = ModelRegistry(base_path=temp_registry_path)

        with pytest.raises(KeyError, match="Model not found"):
            registry.load_model("nonexistent_id")

    def test_load_uses_cache(self, temp_registry_path, fitted_model):
        """Test that load uses in-memory cache."""
        registry = ModelRegistry(base_path=temp_registry_path, cache_size=5)
        model_id = registry.register_model(fitted_model)

        # First load (from disk)
        model1 = registry.load_model(model_id, use_cache=True)

        # Second load (from cache - should be same object)
        model2 = registry.load_model(model_id, use_cache=True)

        assert model1 is model2

    def test_load_without_cache(self, temp_registry_path, fitted_model):
        """Test loading without cache."""
        registry = ModelRegistry(base_path=temp_registry_path)
        model_id = registry.register_model(fitted_model)

        # Load without cache
        model1 = registry.load_model(model_id, use_cache=False)
        model2 = registry.load_model(model_id, use_cache=False)

        # Should be different objects
        assert model1 is not model2

    def test_cache_eviction(self, temp_registry_path):
        """Test that cache evicts old models when full."""
        registry = ModelRegistry(base_path=temp_registry_path, cache_size=2)

        # Register and load 3 models
        model_ids = []
        for i in range(3):
            model = MockModel(model_name=f"model_{i}", version="1.0.0")
            X = pd.DataFrame({"a": [1]})
            y = pd.Series([2])
            model.fit(X, y, {})
            model_id = registry.register_model(model)
            model_ids.append(model_id)
            registry.load_model(model_id)

        # Cache should only have last 2 models
        assert len(registry._model_cache) <= 2


class TestModelRegistryListing:
    """Test model listing and filtering."""

    @pytest.fixture
    def registry_with_models(self, temp_registry_path):
        """Create registry with multiple models."""
        registry = ModelRegistry(base_path=temp_registry_path)

        # Register models with different names and versions
        import time

        for i in range(3):
            model = MockModel(model_name="lgbm", version=f"{i + 1}.0.0")
            X = pd.DataFrame({"a": [1]})
            y = pd.Series([2])
            metrics = {"mape": float(5 - i), "rmse": 100.0 + i * 10}
            model.fit(X, y, {"metrics": metrics})
            # Provide unique metadata to avoid ID collisions
            metadata = model.get_metadata(model_id=f"lgbm_{i + 1}")
            registry.register_model(model, metadata=metadata)
            time.sleep(0.001)  # Small delay to ensure different timestamps

        for i in range(2):
            model = MockModel(model_name="random_forest", version="1.0.0")
            X = pd.DataFrame({"a": [1]})
            y = pd.Series([2])
            metrics = {"mape": float(3 + i)}
            model.fit(X, y, {"metrics": metrics})
            # Provide unique metadata
            metadata = model.get_metadata(model_id=f"random_forest_{i + 1}")
            registry.register_model(model, metadata=metadata)
            time.sleep(0.001)

        return registry

    def test_list_all_models(self, registry_with_models):
        """Test listing all models."""
        models = registry_with_models.list_models()

        assert len(models) == 5

    def test_list_by_model_name(self, registry_with_models):
        """Test filtering by model name."""
        models = registry_with_models.list_models(model_name="lgbm")

        assert len(models) == 3
        assert all(m.model_name == "lgbm" for m in models)

    def test_list_by_version_constraint(self, registry_with_models):
        """Test filtering by version constraint."""
        models = registry_with_models.list_models(model_name="lgbm", version_constraint=">=2.0.0")

        assert len(models) == 2
        assert all(m.model_name == "lgbm" for m in models)

    def test_list_by_horizon(self, registry_with_models):
        """Test filtering by supported horizon."""
        models = registry_with_models.list_models(min_horizon=0)

        # All test models support horizon 0
        assert len(models) == 5

    def test_list_sorted_by_registered_at(self, registry_with_models):
        """Test that results are sorted by registration time."""
        models = registry_with_models.list_models()

        # Should be sorted newest first
        for i in range(len(models) - 1):
            assert models[i].registered_at >= models[i + 1].registered_at


class TestModelRegistryBestModel:
    """Test getting best model by metric."""

    @pytest.fixture
    def registry_with_metrics(self, temp_registry_path):
        """Create registry with models having different metrics."""
        registry = ModelRegistry(base_path=temp_registry_path)

        metrics_list = [
            {"mape": 5.0, "rmse": 200.0},
            {"mape": 2.5, "rmse": 150.0},
            {"mape": 3.5, "rmse": 175.0},
        ]

        for i, metrics in enumerate(metrics_list):
            model = MockModel(model_name="lgbm", version=f"{i + 1}.0.0")
            X = pd.DataFrame({"a": [1]})
            y = pd.Series([2])
            model.fit(X, y, {"metrics": metrics})
            # Provide unique metadata
            metadata = model.get_metadata(model_id=f"lgbm_metrics_{i + 1}")
            registry.register_model(model, metadata=metadata)

        return registry

    def test_get_best_model_lower_is_better(self, registry_with_metrics):
        """Test getting best model when lower metric is better."""
        best_id = registry_with_metrics.get_best_model(
            metric="mape", model_name="lgbm", lower_is_better=True
        )

        assert best_id is not None
        metadata = registry_with_metrics.get_metadata(best_id)
        assert metadata.performance_metrics["mape"] == 2.5

    def test_get_best_model_higher_is_better(self, registry_with_metrics):
        """Test getting best model when higher metric is better."""
        best_id = registry_with_metrics.get_best_model(
            metric="mape", model_name="lgbm", lower_is_better=False
        )

        assert best_id is not None
        metadata = registry_with_metrics.get_metadata(best_id)
        assert metadata.performance_metrics["mape"] == 5.0

    def test_get_best_model_nonexistent_metric(self, registry_with_metrics):
        """Test that get_best_model returns None for nonexistent metric."""
        best_id = registry_with_metrics.get_best_model(metric="nonexistent", model_name="lgbm")

        assert best_id is None


class TestModelRegistryComparison:
    """Test model comparison."""

    @pytest.fixture
    def registry_for_comparison(self, temp_registry_path):
        """Create registry for comparison tests."""
        registry = ModelRegistry(base_path=temp_registry_path)

        model_ids = []
        for i in range(3):
            model = MockModel(model_name="lgbm", version=f"{i + 1}.0.0")
            X = pd.DataFrame({"a": [1]})
            y = pd.Series([2])
            metrics = {"mape": float(i + 1), "rmse": 100.0 + i * 50}
            model.fit(X, y, {"metrics": metrics})
            # Provide unique metadata
            metadata = model.get_metadata(model_id=f"lgbm_compare_{i + 1}")
            model_id = registry.register_model(model, metadata=metadata)
            model_ids.append(model_id)

        return registry, model_ids

    def test_compare_models(self, registry_for_comparison):
        """Test comparing multiple models."""
        registry, model_ids = registry_for_comparison

        comparison = registry.compare_models(model_ids)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert "model_name" in comparison.columns
        assert "model_version" in comparison.columns
        assert "mape" in comparison.columns
        assert "rmse" in comparison.columns

    def test_compare_nonexistent_model_raises_error(self, registry_for_comparison):
        """Test that comparing with nonexistent model raises error."""
        registry, model_ids = registry_for_comparison

        with pytest.raises(KeyError, match="Model not found"):
            registry.compare_models([*model_ids, "nonexistent_id"])


class TestModelRegistryDeregistration:
    """Test model deregistration."""

    def test_deregister_model(self, temp_registry_path, fitted_model):
        """Test deregistering a model (keeping files)."""
        registry = ModelRegistry(base_path=temp_registry_path)
        model_id = registry.register_model(fitted_model)

        assert model_id in registry
        assert len(registry) == 1

        registry.deregister_model(model_id, delete_files=False)

        assert model_id not in registry
        assert len(registry) == 0

        # Files should still exist
        model_file = temp_registry_path / "models" / f"{model_id}.joblib"
        assert model_file.exists()

    def test_deregister_with_delete_files(self, temp_registry_path, fitted_model):
        """Test deregistering a model and deleting files."""
        registry = ModelRegistry(base_path=temp_registry_path)
        model_id = registry.register_model(fitted_model)

        model_file = temp_registry_path / "models" / f"{model_id}.joblib"
        metadata_file = temp_registry_path / "metadata" / f"{model_id}.json"

        assert model_file.exists()
        assert metadata_file.exists()

        registry.deregister_model(model_id, delete_files=True)

        assert model_id not in registry
        assert not model_file.exists()
        assert not metadata_file.exists()

    def test_deregister_nonexistent_model_raises_error(self, temp_registry_path):
        """Test that deregistering nonexistent model raises error."""
        registry = ModelRegistry(base_path=temp_registry_path)

        with pytest.raises(KeyError, match="Model not found"):
            registry.deregister_model("nonexistent_id")

    def test_deregister_removes_from_cache(self, temp_registry_path, fitted_model):
        """Test that deregistration removes model from cache."""
        registry = ModelRegistry(base_path=temp_registry_path)
        model_id = registry.register_model(fitted_model)

        # Load into cache
        registry.load_model(model_id)
        assert model_id in registry._model_cache

        # Deregister
        registry.deregister_model(model_id)
        assert model_id not in registry._model_cache


class TestModelRegistryMetadata:
    """Test metadata retrieval."""

    def test_get_metadata(self, temp_registry_path, fitted_model):
        """Test getting metadata for a model."""
        registry = ModelRegistry(base_path=temp_registry_path)
        model_id = registry.register_model(fitted_model)

        metadata = registry.get_metadata(model_id)

        assert isinstance(metadata, ModelMetadata)
        assert metadata.model_id == model_id
        assert metadata.model_name == "test_model"

    def test_get_metadata_nonexistent_raises_error(self, temp_registry_path):
        """Test that getting metadata for nonexistent model raises error."""
        registry = ModelRegistry(base_path=temp_registry_path)

        with pytest.raises(KeyError, match="Model not found"):
            registry.get_metadata("nonexistent_id")


class TestModelRegistryUtilities:
    """Test utility methods and operators."""

    def test_len_operator(self, temp_registry_path, fitted_model):
        """Test __len__ operator."""
        registry = ModelRegistry(base_path=temp_registry_path)

        assert len(registry) == 0

        registry.register_model(fitted_model)
        assert len(registry) == 1

    def test_contains_operator(self, temp_registry_path, fitted_model):
        """Test __contains__ operator."""
        registry = ModelRegistry(base_path=temp_registry_path)
        model_id = registry.register_model(fitted_model)

        assert model_id in registry
        assert "nonexistent_id" not in registry

    def test_repr(self, temp_registry_path):
        """Test __repr__ method."""
        registry = ModelRegistry(base_path=temp_registry_path)

        r = repr(registry)
        assert "ModelRegistry" in r
        assert str(temp_registry_path) in r
        assert "models=0" in r


class TestModelClassRegistry:
    """Test global model class registration."""

    def test_register_model_class_decorator(self):
        """Test registering model class with decorator."""

        @register_model_class("test_registered")
        class TestRegisteredModel(BaseModel):
            @property
            def name(self) -> str:
                return "test_registered"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def supported_horizons(self) -> list[int]:
                return [0]

            def fit(self, X, y, config):
                pass

            def predict(self, X, horizons=None):
                pass

            def get_feature_importance(self):
                return {}

        # Should be retrievable
        cls = get_model_class("test_registered")
        assert cls == TestRegisteredModel

    def test_get_model_class_nonexistent_raises_error(self):
        """Test that getting nonexistent model class raises error."""
        with pytest.raises(KeyError, match="Unknown model class"):
            get_model_class("nonexistent_model_class")


class TestModelRegistryThreadSafety:
    """Test thread-safety of registry operations."""

    def test_concurrent_registration(self, temp_registry_path):
        """Test that concurrent registrations are thread-safe."""
        import threading

        registry = ModelRegistry(base_path=temp_registry_path)
        model_ids = []
        lock = threading.Lock()

        def register_model(i):
            model = MockModel(model_name=f"model_{i}", version="1.0.0")
            X = pd.DataFrame({"a": [1]})
            y = pd.Series([2])
            model.fit(X, y, {})
            model_id = registry.register_model(model)

            with lock:
                model_ids.append(model_id)

        # Create and start threads
        threads = [threading.Thread(target=register_model, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All models should be registered
        assert len(registry) == 5
        assert len(model_ids) == 5


class TestModelRegistryPersistence:
    """Test registry persistence across instances."""

    def test_persistence_across_instances(self, temp_registry_path, fitted_model):
        """Test that registry state persists across instances."""
        # Create registry and register model
        registry1 = ModelRegistry(base_path=temp_registry_path)
        model_id = registry1.register_model(fitted_model)

        # Create new registry instance
        registry2 = ModelRegistry(base_path=temp_registry_path)

        # Should have the same model
        assert len(registry2) == 1
        assert model_id in registry2

        # Should be able to load the model
        loaded_model = registry2.load_model(model_id)
        assert loaded_model.name == "test_model"
