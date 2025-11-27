"""Tests for LGBMModel implementation.

This module tests the LightGBM model including:
- Model initialization and properties
- Training without hyperparameter optimization
- Training with Optuna optimization (small n_trials)
- Predictions for D+0 and D+1 horizons
- Feature importance extraction
- SHAP values computation
- Model save/load round-trip
- Error cases (unfitted model, unsupported horizons)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.config.lgbm_config import LGBMConfig
from src.models.end_to_end.lgbm_model import LGBMModel


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic load data with patterns for testing.

    Returns:
        Tuple of (features DataFrame, target Series) with:
        - Daily patterns (hour of day)
        - Weekly patterns (day of week)
        - Temperature effects
        - Trend component
    """
    np.random.seed(42)
    n_samples = 1000

    # Create time-based features
    hours = np.arange(n_samples) % 24
    days = (np.arange(n_samples) // 24) % 7

    # Create features
    features = pd.DataFrame({
        "hour": hours,
        "day_of_week": days,
        "temperature": 20 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.randn(n_samples) * 2,
        "lag_1": np.random.randn(n_samples) * 5 + 50,
        "lag_24": np.random.randn(n_samples) * 3 + 50,
        "rolling_mean_7d": 48 + np.random.randn(n_samples) * 4,
    })

    # Create target with patterns
    # Daily pattern: higher load during day (7am-10pm)
    daily_pattern = 20 * (1 + np.sin(2 * np.pi * (hours - 6) / 24))

    # Weekly pattern: lower on weekends
    weekly_pattern = np.where(days >= 5, -5, 0)

    # Temperature effect: higher load with extreme temperatures
    temp_effect = 0.5 * (features["temperature"] - 20) ** 2

    # Noise
    noise = np.random.randn(n_samples) * 3

    # Combine components
    target = pd.Series(
        40 + daily_pattern + weekly_pattern + temp_effect + noise,
        name="load"
    )

    return features, target


@pytest.fixture
def train_test_split(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> dict[str, pd.DataFrame | pd.Series]:
    """Split synthetic data into train and test sets.

    Args:
        synthetic_data: Fixture providing synthetic data.

    Returns:
        Dictionary with X_train, X_test, y_train, y_test.
    """
    X, y = synthetic_data

    # Use first 80% for training (time series, no shuffle)
    train_size = int(0.8 * len(X))

    return {
        "X_train": X.iloc[:train_size],
        "X_test": X.iloc[train_size:],
        "y_train": y.iloc[:train_size],
        "y_test": y.iloc[train_size:],
    }


class TestLGBMModelInitialization:
    """Test model initialization and properties."""

    def test_initialization(self) -> None:
        """Test that LGBMModel initializes correctly."""
        model = LGBMModel()

        assert model.name == "lgbm"
        assert model.version == "1.0.0"
        assert model.supported_horizons == [0, 1]
        assert not model.is_fitted()
        assert model.get_feature_names() == []

    def test_model_repr(self) -> None:
        """Test model string representation."""
        model = LGBMModel()

        repr_str = repr(model)
        assert "LGBMModel" in repr_str
        assert "lgbm" in repr_str
        assert "1.0.0" in repr_str
        assert "fitted=False" in repr_str

        str_str = str(model)
        assert "lgbm" in str_str
        assert "1.0.0" in str_str


class TestLGBMModelTraining:
    """Test model training functionality."""

    def test_fit_basic(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test basic model training without optimization."""
        model = LGBMModel()
        config = LGBMConfig(
            n_estimators=50,  # Fast training
            optimize_hyperparams=False,
            early_stopping_rounds=10,
            verbosity=-1,
        )

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        assert model.is_fitted()
        assert len(model.get_feature_names()) == 6
        assert set(model.get_feature_names()) == set(train_test_split["X_train"].columns)

    def test_fit_with_dataframe_target(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test training with DataFrame target (should use first column)."""
        model = LGBMModel()
        config = LGBMConfig(n_estimators=50, optimize_hyperparams=False)

        # Convert target to DataFrame
        y_df = pd.DataFrame({
            "load": train_test_split["y_train"],
            "extra_col": train_test_split["y_train"] * 2,
        })

        model.fit(
            train_test_split["X_train"],
            y_df,
            config.model_dump(),
        )

        assert model.is_fitted()

    def test_fit_with_optimization(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test training with Optuna hyperparameter optimization."""
        model = LGBMModel()
        config = LGBMConfig(
            n_estimators=50,
            optimize_hyperparams=True,
            n_trials=3,  # Small number for fast testing
            cv_folds=2,  # Small number for fast testing
            early_stopping_rounds=10,
        )

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        assert model.is_fitted()
        assert model._cv_scores is not None
        assert "best_value" in model._cv_scores
        assert "n_trials" in model._cv_scores
        assert model._cv_scores["n_trials"] == 3

    def test_fit_stores_metadata(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that training stores appropriate metadata."""
        model = LGBMModel()
        config = LGBMConfig(n_estimators=50, optimize_hyperparams=False)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        metadata = model._training_metadata

        assert "trained_at" in metadata
        assert "training_samples" in metadata
        assert "training_features" in metadata
        assert "best_iteration" in metadata
        assert "training_config" in metadata

        assert metadata["training_samples"] == len(train_test_split["X_train"])
        assert metadata["training_features"] == len(train_test_split["X_train"].columns)

    def test_fit_with_invalid_config(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that invalid config raises error."""
        model = LGBMModel()

        with pytest.raises(ValueError, match="Invalid configuration"):
            model.fit(
                train_test_split["X_train"],
                train_test_split["y_train"],
                {"n_estimators": -1},  # Invalid
            )

    def test_fit_with_empty_data(self) -> None:
        """Test that empty training data raises error."""
        model = LGBMModel()
        config = LGBMConfig()

        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="cannot be empty"):
            model.fit(empty_X, empty_y, config.model_dump())

    def test_fit_with_mismatched_lengths(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that mismatched X and y lengths raise error."""
        model = LGBMModel()
        config = LGBMConfig()

        X = train_test_split["X_train"]
        y = train_test_split["y_train"].iloc[:10]  # Different length

        with pytest.raises(ValueError, match="same length"):
            model.fit(X, y, config.model_dump())


class TestLGBMModelPrediction:
    """Test model prediction functionality."""

    @pytest.fixture
    def fitted_model(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> LGBMModel:
        """Create a fitted model for testing predictions."""
        model = LGBMModel()
        config = LGBMConfig(n_estimators=50, optimize_hyperparams=False)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        return model

    def test_predict_all_horizons(
        self,
        fitted_model: LGBMModel,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Test prediction for all supported horizons."""
        predictions = fitted_model.predict(train_test_split["X_test"])

        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(train_test_split["X_test"])
        assert list(predictions.columns) == ["h0", "h1"]
        assert not predictions.isna().any().any()

    def test_predict_specific_horizons(
        self,
        fitted_model: LGBMModel,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Test prediction for specific horizons."""
        # D+0 only
        predictions_h0 = fitted_model.predict(train_test_split["X_test"], horizons=[0])
        assert list(predictions_h0.columns) == ["h0"]

        # D+1 only
        predictions_h1 = fitted_model.predict(train_test_split["X_test"], horizons=[1])
        assert list(predictions_h1.columns) == ["h1"]

        # Both
        predictions_both = fitted_model.predict(train_test_split["X_test"], horizons=[0, 1])
        assert list(predictions_both.columns) == ["h0", "h1"]

    def test_predict_reasonable_values(
        self,
        fitted_model: LGBMModel,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Test that predictions are in reasonable range."""
        predictions = fitted_model.predict(train_test_split["X_test"])

        # Load values should be positive and in reasonable range (20-100)
        assert (predictions["h0"] > 0).all()
        assert (predictions["h0"] < 200).all()

        # D+1 predictions should be similar to D+0
        diff = (predictions["h1"] - predictions["h0"]).abs()
        assert diff.mean() < 10  # Average difference should be small

    def test_predict_unfitted_model(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that predicting with unfitted model raises error."""
        model = LGBMModel()

        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(train_test_split["X_test"])

    def test_predict_invalid_horizons(
        self,
        fitted_model: LGBMModel,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Test that invalid horizons raise error."""
        with pytest.raises(ValueError, match="Invalid horizons"):
            fitted_model.predict(train_test_split["X_test"], horizons=[2, 3])

        with pytest.raises(ValueError, match="Invalid horizons"):
            fitted_model.predict(train_test_split["X_test"], horizons=[0, 1, 2])

    def test_predict_missing_features(
        self,
        fitted_model: LGBMModel,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Test that missing features raise error."""
        X_missing = train_test_split["X_test"].drop(columns=["hour"])

        with pytest.raises(ValueError, match="Missing required features"):
            fitted_model.predict(X_missing)

    def test_predict_with_extra_features(
        self,
        fitted_model: LGBMModel,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Test prediction with extra features (should work, extras ignored)."""
        X_extra = train_test_split["X_test"].copy()
        X_extra["extra_feature"] = 1.0

        predictions = fitted_model.predict(X_extra)

        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(X_extra)


class TestLGBMModelFeatureImportance:
    """Test feature importance extraction."""

    @pytest.fixture
    def fitted_model(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> LGBMModel:
        """Create a fitted model for testing."""
        model = LGBMModel()
        config = LGBMConfig(n_estimators=50, optimize_hyperparams=False)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        return model

    def test_get_feature_importance(self, fitted_model: LGBMModel) -> None:
        """Test feature importance extraction."""
        importance = fitted_model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 6  # All features
        assert set(importance.keys()) == set(fitted_model.get_feature_names())

        # Check normalization (sum to 1.0)
        total = sum(importance.values())
        assert abs(total - 1.0) < 1e-6

        # All values should be non-negative
        assert all(v >= 0 for v in importance.values())

    def test_feature_importance_unfitted(self) -> None:
        """Test that feature importance on unfitted model raises error."""
        model = LGBMModel()

        with pytest.raises(ValueError, match="not been fitted"):
            model.get_feature_importance()


class TestLGBMModelSHAPValues:
    """Test SHAP value computation."""

    @pytest.fixture
    def fitted_model(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> LGBMModel:
        """Create a fitted model for testing."""
        model = LGBMModel()
        config = LGBMConfig(n_estimators=50, optimize_hyperparams=False)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        return model

    def test_get_shap_values(
        self,
        fitted_model: LGBMModel,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Test SHAP value computation."""
        X_test = train_test_split["X_test"]

        shap_values = fitted_model.get_shap_values(X_test, sample_size=50)

        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape == (50, 6)  # sample_size x n_features

    def test_get_shap_values_full(
        self,
        fitted_model: LGBMModel,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Test SHAP value computation without sampling."""
        X_test = train_test_split["X_test"]

        shap_values = fitted_model.get_shap_values(X_test)

        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape == (len(X_test), 6)

    def test_get_shap_values_unfitted(
        self,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Test that SHAP values on unfitted model raise error."""
        model = LGBMModel()

        with pytest.raises(ValueError, match="not been fitted"):
            model.get_shap_values(train_test_split["X_test"])


class TestLGBMModelSerialization:
    """Test model save/load functionality."""

    @pytest.fixture
    def fitted_model(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> LGBMModel:
        """Create a fitted model for testing."""
        model = LGBMModel()
        config = LGBMConfig(n_estimators=50, optimize_hyperparams=False)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        return model

    def test_save_load_roundtrip(
        self,
        fitted_model: LGBMModel,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
        tmp_path: Path,
    ) -> None:
        """Test save and load round-trip."""
        # Save model
        model_path = tmp_path / "test_model.joblib"
        fitted_model.save(model_path)

        # Check files were created
        assert model_path.with_suffix(".lgb").exists()
        assert model_path.with_suffix(".meta").exists()

        # Load model
        loaded_model = LGBMModel.load(model_path)

        # Check properties preserved
        assert loaded_model.name == fitted_model.name
        assert loaded_model.version == fitted_model.version
        assert loaded_model.is_fitted()
        assert loaded_model.get_feature_names() == fitted_model.get_feature_names()

        # Check predictions match
        X_test = train_test_split["X_test"]
        original_preds = fitted_model.predict(X_test)
        loaded_preds = loaded_model.predict(X_test)

        pd.testing.assert_frame_equal(original_preds, loaded_preds)

    def test_save_creates_directory(self, fitted_model: LGBMModel, tmp_path: Path) -> None:
        """Test that save creates parent directory if needed."""
        model_path = tmp_path / "subdir" / "nested" / "model.joblib"

        fitted_model.save(model_path)

        assert model_path.with_suffix(".lgb").exists()
        assert model_path.with_suffix(".meta").exists()

    def test_save_unfitted_model(self, tmp_path: Path) -> None:
        """Test that saving unfitted model raises error."""
        model = LGBMModel()
        model_path = tmp_path / "model.joblib"

        with pytest.raises(ValueError, match="not been fitted"):
            model.save(model_path)

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that loading nonexistent file raises error."""
        model_path = tmp_path / "nonexistent.joblib"

        with pytest.raises(FileNotFoundError):
            LGBMModel.load(model_path)

    def test_load_missing_lgb_file(self, fitted_model: LGBMModel, tmp_path: Path) -> None:
        """Test loading with missing .lgb file."""
        model_path = tmp_path / "model.joblib"
        fitted_model.save(model_path)

        # Remove .lgb file
        model_path.with_suffix(".lgb").unlink()

        with pytest.raises(FileNotFoundError, match="LightGBM model file"):
            LGBMModel.load(model_path)

    def test_load_missing_meta_file(self, fitted_model: LGBMModel, tmp_path: Path) -> None:
        """Test loading with missing .meta file."""
        model_path = tmp_path / "model.joblib"
        fitted_model.save(model_path)

        # Remove .meta file
        model_path.with_suffix(".meta").unlink()

        with pytest.raises(FileNotFoundError, match="Metadata file"):
            LGBMModel.load(model_path)


class TestLGBMModelMetadata:
    """Test model metadata generation."""

    def test_get_metadata(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test metadata generation for fitted model."""
        model = LGBMModel()
        config = LGBMConfig(n_estimators=50, optimize_hyperparams=False)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        metadata = model.get_metadata()

        assert metadata.model_name == "lgbm"
        assert metadata.model_version == "1.0.0"
        assert metadata.model_class == "LGBMModel"
        assert metadata.supported_horizons == [0, 1]
        assert len(metadata.feature_dependencies) == 6
        assert metadata.trained_at is not None


class TestLGBMModelRegistration:
    """Test model class registration."""

    def test_model_registered(self) -> None:
        """Test that LGBMModel is registered with decorator."""
        from src.models.base import get_model_class

        # Should be able to retrieve registered class
        model_class = get_model_class("lgbm")
        assert model_class == LGBMModel

    def test_can_instantiate_from_registry(self) -> None:
        """Test that model can be instantiated from registry."""
        from src.models.base import get_model_class

        model_class = get_model_class("lgbm")
        model = model_class()

        assert isinstance(model, LGBMModel)
        assert model.name == "lgbm"
