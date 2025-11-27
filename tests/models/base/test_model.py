"""Tests for BaseModel abstract class."""

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from src.models.base.metadata import ModelMetadata
from src.models.base.model import BaseModel


# Mock implementation of BaseModel for testing
class MockModel(BaseModel):
    """Mock model implementation for testing."""

    def __init__(self):
        super().__init__()
        self._mock_model = None

    @property
    def name(self) -> str:
        return "mock_model"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_horizons(self) -> list[int]:
        return [0, 1, 2, 3]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Mock fit implementation."""
        self._feature_names = list(X.columns)
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_config": config,
            "performance_metrics": {"mape": 2.5, "rmse": 150.0},
        }
        self._mock_model = "fitted"
        self._is_fitted = True

    def predict(
        self,
        X: pd.DataFrame,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Mock predict implementation."""
        self._check_is_fitted()
        self.validate_features(X)

        if horizons is None:
            horizons = self.supported_horizons

        # Validate horizons
        invalid = set(horizons) - set(self.supported_horizons)
        if invalid:
            msg = f"Invalid horizons: {invalid}"
            raise ValueError(msg)

        # Generate mock predictions
        return pd.DataFrame(
            index=X.index,
            columns=[f"h{h}" for h in horizons],
            data=100.0,
        )

    def get_feature_importance(self) -> dict[str, float]:
        """Mock feature importance implementation."""
        self._check_is_fitted()

        if not self._feature_names:
            return {}

        # Return equal importance for all features
        n_features = len(self._feature_names)
        importance = 1.0 / n_features
        return dict.fromkeys(self._feature_names, importance)


class TestBaseModelInterface:
    """Test BaseModel abstract interface."""

    def test_cannot_instantiate_base_model(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()

    def test_mock_model_implements_interface(self):
        """Test that MockModel properly implements BaseModel interface."""
        model = MockModel()
        assert isinstance(model, BaseModel)
        assert model.name == "mock_model"
        assert model.version == "1.0.0"
        assert model.supported_horizons == [0, 1, 2, 3]


class TestBaseModelInitialization:
    """Test BaseModel initialization."""

    def test_initial_state(self):
        """Test that new model has correct initial state."""
        model = MockModel()

        assert not model.is_fitted()
        assert model.get_feature_names() == []
        assert model._training_metadata == {}


class TestBaseModelFitting:
    """Test model fitting functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = pd.DataFrame(
            {
                "hour": [0, 1, 2, 3, 4],
                "day_of_week": [0, 0, 0, 0, 0],
                "temperature": [20.0, 21.0, 22.0, 23.0, 24.0],
            }
        )
        y = pd.Series([100, 105, 110, 115, 120])
        config = {"learning_rate": 0.1, "max_depth": 5}
        return X, y, config

    def test_fit_model(self, sample_data):
        """Test fitting a model."""
        X, y, config = sample_data
        model = MockModel()

        assert not model.is_fitted()

        model.fit(X, y, config)

        assert model.is_fitted()
        assert model.get_feature_names() == ["hour", "day_of_week", "temperature"]

    def test_fit_stores_metadata(self, sample_data):
        """Test that fit stores training metadata."""
        X, y, config = sample_data
        model = MockModel()

        model.fit(X, y, config)

        assert "trained_at" in model._training_metadata
        assert "training_config" in model._training_metadata
        assert model._training_metadata["training_config"] == config


class TestBaseModelPrediction:
    """Test model prediction functionality."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        X = pd.DataFrame(
            {
                "hour": [0, 1, 2],
                "day_of_week": [0, 0, 0],
                "temperature": [20.0, 21.0, 22.0],
            }
        )
        y = pd.Series([100, 105, 110])
        config = {"learning_rate": 0.1}

        model = MockModel()
        model.fit(X, y, config)
        return model

    def test_predict_all_horizons(self, fitted_model):
        """Test prediction with all supported horizons."""
        X_test = pd.DataFrame(
            {
                "hour": [5, 6],
                "day_of_week": [0, 0],
                "temperature": [25.0, 26.0],
            }
        )

        predictions = fitted_model.predict(X_test)

        assert predictions.shape == (2, 4)  # 2 samples, 4 horizons
        assert list(predictions.columns) == ["h0", "h1", "h2", "h3"]

    def test_predict_specific_horizons(self, fitted_model):
        """Test prediction with specific horizons."""
        X_test = pd.DataFrame(
            {
                "hour": [5, 6],
                "day_of_week": [0, 0],
                "temperature": [25.0, 26.0],
            }
        )

        predictions = fitted_model.predict(X_test, horizons=[0, 2])

        assert predictions.shape == (2, 2)
        assert list(predictions.columns) == ["h0", "h2"]

    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error if model not fitted."""
        model = MockModel()
        X = pd.DataFrame({"hour": [0], "day_of_week": [0], "temperature": [20.0]})

        with pytest.raises(ValueError, match="has not been fitted"):
            model.predict(X)

    def test_predict_invalid_horizons(self, fitted_model):
        """Test that invalid horizons raise error."""
        X_test = pd.DataFrame(
            {
                "hour": [5],
                "day_of_week": [0],
                "temperature": [25.0],
            }
        )

        with pytest.raises(ValueError, match="Invalid horizons"):
            fitted_model.predict(X_test, horizons=[0, 10])


class TestBaseModelFeatures:
    """Test feature-related functionality."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        y = pd.Series([10, 11, 12])
        model = MockModel()
        model.fit(X, y, {})
        return model

    def test_get_feature_names(self, fitted_model):
        """Test getting feature names."""
        feature_names = fitted_model.get_feature_names()

        assert feature_names == ["a", "b", "c"]
        # Should return a copy, not reference
        feature_names.append("d")
        assert fitted_model.get_feature_names() == ["a", "b", "c"]

    def test_validate_features_valid(self, fitted_model):
        """Test feature validation with valid features."""
        X_test = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        fitted_model.validate_features(X_test)  # Should not raise

    def test_validate_features_extra_columns(self, fitted_model):
        """Test feature validation with extra columns (should be ok)."""
        X_test = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        fitted_model.validate_features(X_test)  # Should not raise (extra ignored)

    def test_validate_features_missing_columns(self, fitted_model):
        """Test feature validation with missing required columns."""
        X_test = pd.DataFrame({"a": [1], "b": [2]})  # Missing 'c'

        with pytest.raises(ValueError, match="Missing required features"):
            fitted_model.validate_features(X_test)

    def test_validate_features_before_fit_raises_error(self):
        """Test that validate_features raises error if not fitted."""
        model = MockModel()
        X = pd.DataFrame({"a": [1]})

        with pytest.raises(ValueError, match="has not been fitted"):
            model.validate_features(X)

    def test_get_feature_importance(self, fitted_model):
        """Test getting feature importance."""
        importance = fitted_model.get_feature_importance()

        assert isinstance(importance, dict)
        assert set(importance.keys()) == {"a", "b", "c"}
        # Equal importance for all features in mock
        assert all(abs(v - 1 / 3) < 0.001 for v in importance.values())

    def test_get_feature_importance_before_fit_raises_error(self):
        """Test that get_feature_importance raises error if not fitted."""
        model = MockModel()

        with pytest.raises(ValueError, match="has not been fitted"):
            model.get_feature_importance()


class TestBaseModelMetadata:
    """Test metadata generation."""

    def test_get_metadata_default(self):
        """Test metadata generation with auto-generated ID."""
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([5, 6])
        model = MockModel()
        model.fit(X, y, {"lr": 0.1})

        metadata = model.get_metadata()

        assert isinstance(metadata, ModelMetadata)
        assert metadata.model_name == "mock_model"
        assert metadata.model_version == "1.0.0"
        assert metadata.model_class == "MockModel"
        assert metadata.supported_horizons == [0, 1, 2, 3]
        assert metadata.feature_dependencies == ["a", "b"]
        assert "mock_model_" in metadata.model_id

    def test_get_metadata_custom_id(self):
        """Test metadata generation with custom ID."""
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([2])
        model = MockModel()
        model.fit(X, y, {})

        metadata = model.get_metadata(model_id="custom_id_001")

        assert metadata.model_id == "custom_id_001"

    def test_get_metadata_includes_training_info(self):
        """Test that metadata includes training information."""
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([2])
        model = MockModel()
        model.fit(X, y, {"lr": 0.1, "epochs": 100})

        metadata = model.get_metadata()

        assert metadata.training_config == {"lr": 0.1, "epochs": 100}
        assert metadata.performance_metrics == {"mape": 2.5, "rmse": 150.0}


class TestBaseModelSerialization:
    """Test model saving and loading."""

    def test_save_model(self, tmp_path):
        """Test saving a fitted model."""
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([5, 6])
        model = MockModel()
        model.fit(X, y, {})

        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        assert save_path.exists()

    def test_save_unfitted_model_raises_error(self, tmp_path):
        """Test that saving unfitted model raises error."""
        model = MockModel()
        save_path = tmp_path / "model.joblib"

        with pytest.raises(ValueError, match="has not been fitted"):
            model.save(save_path)

    def test_save_creates_parent_directory(self, tmp_path):
        """Test that save creates parent directories."""
        model = MockModel()
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([2])
        model.fit(X, y, {})

        save_path = tmp_path / "nested" / "dir" / "model.joblib"
        model.save(save_path)

        assert save_path.exists()

    def test_load_model(self, tmp_path):
        """Test loading a saved model."""
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([5, 6])
        original_model = MockModel()
        original_model.fit(X, y, {"lr": 0.1})

        save_path = tmp_path / "model.joblib"
        original_model.save(save_path)

        loaded_model = BaseModel.load(save_path)

        assert isinstance(loaded_model, MockModel)
        assert loaded_model.is_fitted()
        assert loaded_model.name == "mock_model"
        assert loaded_model.version == "1.0.0"
        assert loaded_model.get_feature_names() == ["a", "b"]

    def test_load_nonexistent_file_raises_error(self, tmp_path):
        """Test that loading nonexistent file raises error."""
        save_path = tmp_path / "nonexistent.joblib"

        with pytest.raises(FileNotFoundError):
            BaseModel.load(save_path)

    def test_save_load_roundtrip(self, tmp_path):
        """Test that save/load preserves model state."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([7, 8, 9])
        original = MockModel()
        original.fit(X, y, {"lr": 0.1})

        # Save
        save_path = tmp_path / "model.joblib"
        original.save(save_path)

        # Load
        loaded = BaseModel.load(save_path)

        # Test prediction works
        X_test = pd.DataFrame({"a": [10], "b": [11]})
        predictions = loaded.predict(X_test)

        assert predictions.shape == (1, 4)
        assert loaded.get_feature_names() == ["a", "b"]


class TestBaseModelRepresentation:
    """Test string representations."""

    def test_str_representation(self):
        """Test __str__ method."""
        model = MockModel()
        s = str(model)

        assert "mock_model" in s
        assert "1.0.0" in s

    def test_repr_representation_unfitted(self):
        """Test __repr__ for unfitted model."""
        model = MockModel()
        r = repr(model)

        assert "MockModel" in r
        assert "mock_model" in r
        assert "1.0.0" in r
        assert "fitted=False" in r

    def test_repr_representation_fitted(self):
        """Test __repr__ for fitted model."""
        model = MockModel()
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([2])
        model.fit(X, y, {})

        r = repr(model)
        assert "fitted=True" in r


class TestBaseModelEdgeCases:
    """Test edge cases and error conditions."""

    def test_multiple_fits(self):
        """Test that model can be refitted."""
        model = MockModel()
        X1 = pd.DataFrame({"a": [1], "b": [2]})
        y1 = pd.Series([3])
        model.fit(X1, y1, {})

        assert model.get_feature_names() == ["a", "b"]

        # Refit with different features
        X2 = pd.DataFrame({"c": [4], "d": [5]})
        y2 = pd.Series([6])
        model.fit(X2, y2, {})

        assert model.get_feature_names() == ["c", "d"]
        assert model.is_fitted()

    def test_empty_training_metadata(self):
        """Test model with minimal training metadata."""

        class MinimalModel(MockModel):
            def fit(self, X, y, config):
                self._feature_names = list(X.columns)
                self._is_fitted = True
                # Don't set training_metadata

        model = MinimalModel()
        X = pd.DataFrame({"a": [1]})
        y = pd.Series([2])
        model.fit(X, y, {})

        metadata = model.get_metadata()
        # Should still work with defaults
        assert metadata.training_config == {}
        assert metadata.performance_metrics == {}
