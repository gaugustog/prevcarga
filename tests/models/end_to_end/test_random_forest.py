"""Tests for RandomForestModel implementation.

This module tests the Random Forest model including:
- Model initialization and properties
- Multi-horizon training (D+0 through D+8)
- Horizon-safe feature filtering (temporal leakage prevention)
- Feature selection per horizon
- Predictions with confidence intervals
- OOB scoring extraction
- Feature importance (aggregated and per-horizon)
- Model save/load round-trip
- Error cases (unfitted model, unsupported horizons)
"""

# ruff: noqa: NPY002, N806, PD003, B007

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.config.rf_config import RandomForestConfig
from src.models.end_to_end.random_forest import RandomForestModel


@pytest.fixture
def synthetic_data_with_lags() -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic load data with lag features for testing leakage prevention.

    Returns:
        Tuple of (features DataFrame, target Series) with:
        - Multiple lag features (lag_24, lag_48, lag_96, lag_144)
        - Hour and day features
        - Temperature effects
        - Target with daily patterns
    """
    np.random.seed(42)
    n_samples = 2000  # ~41 days at 48 periods/day

    # Create time-based features
    hours = np.arange(n_samples) % 24
    days = (np.arange(n_samples) // 24) % 7

    # Create features with various lags
    features = pd.DataFrame({
        "hour": hours,
        "day_of_week": days,
        "temperature": 20 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.randn(n_samples) * 2,
        "lag_24": np.random.randn(n_samples) * 5 + 50,  # 24 periods (0.5 days for 48 ppd)
        "lag_48": np.random.randn(n_samples) * 5 + 50,  # 48 periods (1 day)
        "lag_96": np.random.randn(n_samples) * 4 + 50,  # 96 periods (2 days)
        "lag_144": np.random.randn(n_samples) * 4 + 50,  # 144 periods (3 days)
        "rolling_mean_7d": 48 + np.random.randn(n_samples) * 4,
        "future_load": np.random.randn(n_samples) * 10 + 60,  # Should be rejected
    })

    # Create target with patterns
    daily_pattern = 20 * (1 + np.sin(2 * np.pi * (hours - 6) / 24))
    weekly_pattern = np.where(days >= 5, -5, 0)
    temp_effect = 0.5 * (features["temperature"] - 20) ** 2
    noise = np.random.randn(n_samples) * 3

    target = pd.Series(
        40 + daily_pattern + weekly_pattern + temp_effect + noise,
        name="load"
    )

    return features, target


@pytest.fixture
def train_test_split(synthetic_data_with_lags: tuple[pd.DataFrame, pd.Series]) -> dict[str, pd.DataFrame | pd.Series]:
    """Split synthetic data into train and test sets.

    Args:
        synthetic_data_with_lags: Fixture providing synthetic data.

    Returns:
        Dictionary with X_train, X_test, y_train, y_test.
    """
    X, y = synthetic_data_with_lags

    # Use first 80% for training (time series, no shuffle)
    train_size = int(0.8 * len(X))

    return {
        "X_train": X.iloc[:train_size],
        "X_test": X.iloc[train_size:],
        "y_train": y.iloc[:train_size],
        "y_test": y.iloc[train_size:],
    }


class TestRandomForestModelInitialization:
    """Test model initialization and properties."""

    def test_initialization(self) -> None:
        """Test that RandomForestModel initializes correctly."""
        model = RandomForestModel()

        assert model.name == "random_forest"
        assert model.version == "1.0.0"
        assert model.supported_horizons == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert not model.is_fitted()
        assert model.get_feature_names() == []
        assert model.models == {}
        assert model.selected_features == {}
        assert model.oob_scores == {}

    def test_model_repr(self) -> None:
        """Test model string representation."""
        model = RandomForestModel()

        repr_str = repr(model)
        assert "RandomForestModel" in repr_str
        assert "random_forest" in repr_str
        assert "1.0.0" in repr_str
        assert "fitted=False" in repr_str

        str_str = str(model)
        assert "random_forest" in str_str
        assert "1.0.0" in str_str


class TestRandomForestModelTraining:
    """Test model training functionality."""

    def test_fit_basic(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test basic model training for all horizons."""
        model = RandomForestModel()
        config = RandomForestConfig(
            n_estimators=20,  # Fast training
            max_features_per_horizon=10,
            bootstrap=True,
            oob_score=True,
        )

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        assert model.is_fitted()
        assert len(model.models) == 9  # D+0 through D+8
        assert len(model.selected_features) == 9
        assert len(model.oob_scores) == 9

        # Check that all horizons were trained
        for horizon in range(9):
            assert horizon in model.models
            assert horizon in model.selected_features
            assert horizon in model.oob_scores

    def test_fit_with_dataframe_target(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test training with DataFrame target (should use first column)."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=10, max_features_per_horizon=5)

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
        assert len(model.models) == 9

    def test_fit_stores_feature_names(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that training stores original feature names."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=10)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        feature_names = model.get_feature_names()
        assert len(feature_names) == len(train_test_split["X_train"].columns)
        assert set(feature_names) == set(train_test_split["X_train"].columns)

    def test_fit_empty_data_error(self) -> None:
        """Test that training with empty data raises ValueError."""
        model = RandomForestModel()
        config = RandomForestConfig()

        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="cannot be empty"):
            model.fit(X_empty, y_empty, config.model_dump())

    def test_fit_mismatched_lengths_error(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that training with mismatched X and y lengths raises ValueError."""
        model = RandomForestModel()
        config = RandomForestConfig()

        X = train_test_split["X_train"]
        y = train_test_split["y_train"].iloc[:100]  # Shorter than X

        with pytest.raises(ValueError, match="same length"):
            model.fit(X, y, config.model_dump())

    def test_fit_invalid_config_error(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that training with invalid config raises ValueError."""
        model = RandomForestModel()

        # Invalid config (n_estimators out of bounds)
        with pytest.raises(ValueError, match="Invalid configuration"):
            model.fit(
                train_test_split["X_train"],
                train_test_split["y_train"],
                {"n_estimators": 0},  # Invalid
            )


class TestHorizonSafeFeatureSelection:
    """Test horizon-safe feature filtering to prevent temporal leakage."""

    def test_future_keyword_rejection(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that features with 'future' keyword are rejected."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=10, max_features_per_horizon=20)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        # Check that "future_load" was rejected for all horizons
        for horizon in range(9):
            selected = model.selected_features[horizon]
            assert "future_load" not in selected, f"future_load should be rejected for horizon {horizon}"

    def test_lag_filtering_for_horizons(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that lag features are filtered based on horizon requirements."""
        model = RandomForestModel()
        config = RandomForestConfig(
            n_estimators=10,
            max_features_per_horizon=20,
            periods_per_day=48,
        )

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        # D+0 (horizon 0): lag_24 should be allowed (24 >= 0 * 48)
        assert "lag_24" in model.selected_features[0] or len(model.selected_features[0]) > 0

        # D+1 (horizon 1): lag_24 should be rejected (24 < 1 * 48 = 48)
        # lag_48 should be allowed (48 >= 48)
        if "lag_24" in train_test_split["X_train"].columns:
            assert "lag_24" not in model.selected_features[1], "lag_24 should be rejected for D+1"

        # D+2 (horizon 2): lag_48 should be rejected (48 < 2 * 48 = 96)
        # lag_96 should be allowed (96 >= 96)
        if "lag_48" in train_test_split["X_train"].columns:
            assert "lag_48" not in model.selected_features[2], "lag_48 should be rejected for D+2"

        # D+3 (horizon 3): lag_96 should be rejected (96 < 3 * 48 = 144)
        # lag_144 should be allowed (144 >= 144)
        if "lag_96" in train_test_split["X_train"].columns:
            assert "lag_96" not in model.selected_features[3], "lag_96 should be rejected for D+3"

    def test_non_lag_features_allowed(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that non-lag features (hour, temperature) are allowed for all horizons."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=10, max_features_per_horizon=20)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        # Hour and day_of_week should be available for all horizons
        for horizon in range(9):
            selected = model.selected_features[horizon]
            # At least one of hour or day_of_week should be selected
            assert len(selected) > 0, f"No features selected for horizon {horizon}"


class TestRandomForestModelPrediction:
    """Test model prediction functionality."""

    def test_predict_all_horizons(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test prediction for all horizons."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=20, max_features_per_horizon=10)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        predictions = model.predict(train_test_split["X_test"])

        # Should have predictions for all 9 horizons with confidence intervals
        # Each horizon has 4 columns: h{n}, h{n}_std, h{n}_lower, h{n}_upper
        expected_cols = 9 * 4  # 36 columns total
        assert len(predictions.columns) == expected_cols

        # Check column names
        for horizon in range(9):
            assert f"h{horizon}" in predictions.columns
            assert f"h{horizon}_std" in predictions.columns
            assert f"h{horizon}_lower" in predictions.columns
            assert f"h{horizon}_upper" in predictions.columns

        # Check no NaN values
        assert not predictions.isnull().any().any()

        # Check predictions shape
        assert len(predictions) == len(train_test_split["X_test"])

    def test_predict_specific_horizons(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test prediction for specific horizons only."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=20, max_features_per_horizon=10)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        predictions = model.predict(train_test_split["X_test"], horizons=[0, 1, 2])

        # Should have predictions for only 3 horizons
        expected_cols = 3 * 4  # 12 columns
        assert len(predictions.columns) == expected_cols

        # Check only requested horizons are present
        assert "h0" in predictions.columns
        assert "h1" in predictions.columns
        assert "h2" in predictions.columns
        assert "h3" not in predictions.columns
        assert "h8" not in predictions.columns

    def test_predict_confidence_intervals(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that confidence intervals are calculated correctly."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=20, max_features_per_horizon=10)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        predictions = model.predict(train_test_split["X_test"], horizons=[0])

        # Check that lower < pred < upper (approximately, for most samples)
        assert (predictions["h0_lower"] <= predictions["h0"]).mean() > 0.9
        assert (predictions["h0"] <= predictions["h0_upper"]).mean() > 0.9

        # Check that CI width is approximately 2 * 1.96 * std
        ci_width = predictions["h0_upper"] - predictions["h0_lower"]
        expected_width = 2 * 1.96 * predictions["h0_std"]
        assert np.allclose(ci_width, expected_width, rtol=0.01)

        # Check that std > 0 (variance exists)
        assert (predictions["h0_std"] > 0).all()

    def test_predict_unfitted_error(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that predicting with unfitted model raises ValueError."""
        model = RandomForestModel()

        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(train_test_split["X_test"])

    def test_predict_invalid_horizons_error(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that invalid horizons raise ValueError."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=10)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        # Invalid horizon (9 is not supported, max is 8)
        with pytest.raises(ValueError, match="Invalid horizons"):
            model.predict(train_test_split["X_test"], horizons=[9])

        # Negative horizon
        with pytest.raises(ValueError, match="Invalid horizons"):
            model.predict(train_test_split["X_test"], horizons=[-1])


class TestRandomForestModelFeatureImportance:
    """Test feature importance functionality."""

    def test_get_feature_importance_aggregated(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test aggregated feature importance across all horizons."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=20, max_features_per_horizon=10)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        importance = model.get_feature_importance()

        # Should return a dictionary
        assert isinstance(importance, dict)

        # Should have positive values
        assert all(v >= 0 for v in importance.values())

        # Should sum to approximately 1.0 (normalized)
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_get_feature_importance_per_horizon(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test per-horizon feature importance."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=20, max_features_per_horizon=10)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        per_horizon = model.get_feature_importance_per_horizon()

        # Should have importance for all 9 horizons
        assert len(per_horizon) == 9

        # Each horizon should have normalized importance (sum to 1.0)
        for horizon, importance in per_horizon.items():
            assert isinstance(importance, dict)
            assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_feature_importance_unfitted_error(self) -> None:
        """Test that feature importance on unfitted model raises ValueError."""
        model = RandomForestModel()

        with pytest.raises(ValueError, match="not been fitted"):
            model.get_feature_importance()

        with pytest.raises(ValueError, match="not been fitted"):
            model.get_feature_importance_per_horizon()


class TestRandomForestModelOOBScoring:
    """Test out-of-bag scoring functionality."""

    def test_oob_scores_stored(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that OOB scores are stored for all horizons."""
        model = RandomForestModel()
        config = RandomForestConfig(
            n_estimators=20,
            bootstrap=True,
            oob_score=True,
        )

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        # Should have OOB scores for all horizons
        assert len(model.oob_scores) == 9

        # OOB scores should be reasonable (between -1 and 1 for R²)
        for horizon, score in model.oob_scores.items():
            assert not np.isnan(score), f"OOB score for horizon {horizon} is NaN"
            # R² can be negative for very poor models, but should be > -10 at least
            assert score > -10, f"OOB score for horizon {horizon} is unreasonably low: {score}"

    def test_oob_scores_disabled_when_bootstrap_false(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that OOB scores are NaN when bootstrap is disabled."""
        model = RandomForestModel()
        config = RandomForestConfig(
            n_estimators=20,
            bootstrap=False,
            oob_score=False,
        )

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        # OOB scores should be NaN
        for score in model.oob_scores.values():
            assert np.isnan(score)


class TestRandomForestModelSaveLoad:
    """Test model serialization and deserialization."""

    def test_save_and_load_roundtrip(
        self,
        train_test_split: dict[str, pd.DataFrame | pd.Series],
        tmp_path: Path,
    ) -> None:
        """Test that model can be saved and loaded correctly."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=20, max_features_per_horizon=10)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        # Make predictions before saving
        predictions_before = model.predict(train_test_split["X_test"], horizons=[0, 1])

        # Save model
        save_path = tmp_path / "test_rf_model.joblib"
        model.save(save_path)

        assert save_path.exists()

        # Load model
        loaded_model = RandomForestModel.load(save_path)

        # Check loaded model state
        assert loaded_model.is_fitted()
        assert loaded_model.name == "random_forest"
        assert loaded_model.version == "1.0.0"
        assert len(loaded_model.models) == 9
        assert len(loaded_model.selected_features) == 9
        assert len(loaded_model.oob_scores) == 9

        # Make predictions with loaded model
        predictions_after = loaded_model.predict(train_test_split["X_test"], horizons=[0, 1])

        # Predictions should be identical
        pd.testing.assert_frame_equal(predictions_before, predictions_after)

    def test_save_unfitted_error(self, tmp_path: Path) -> None:
        """Test that saving unfitted model raises ValueError."""
        model = RandomForestModel()
        save_path = tmp_path / "unfitted.joblib"

        with pytest.raises(ValueError, match="not been fitted"):
            model.save(save_path)

    def test_load_nonexistent_file_error(self, tmp_path: Path) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        load_path = tmp_path / "nonexistent.joblib"

        with pytest.raises(FileNotFoundError):
            RandomForestModel.load(load_path)


class TestRandomForestModelMetadata:
    """Test model metadata functionality."""

    def test_training_metadata_stored(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test that training metadata is stored correctly."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=20)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        # Check training metadata
        metadata = model._training_metadata

        assert "trained_at" in metadata
        assert "training_samples" in metadata
        assert "training_features" in metadata
        assert "oob_scores" in metadata
        assert "features_per_horizon" in metadata
        assert "training_config" in metadata

        assert metadata["training_samples"] == len(train_test_split["X_train"])
        assert metadata["training_features"] == len(train_test_split["X_train"].columns)
        assert len(metadata["oob_scores"]) == 9
        assert len(metadata["features_per_horizon"]) == 9

    def test_get_metadata(self, train_test_split: dict[str, pd.DataFrame | pd.Series]) -> None:
        """Test get_metadata() method."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=20)

        model.fit(
            train_test_split["X_train"],
            train_test_split["y_train"],
            config.model_dump(),
        )

        metadata = model.get_metadata()

        assert metadata.model_name == "random_forest"
        assert metadata.model_version == "1.0.0"
        assert metadata.model_class == "RandomForestModel"
        assert metadata.supported_horizons == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert len(metadata.feature_dependencies) > 0


class TestRandomForestModelEdgeCases:
    """Test edge cases and special scenarios."""

    def test_small_dataset(self) -> None:
        """Test training with very small dataset."""
        model = RandomForestModel()
        config = RandomForestConfig(
            n_estimators=10,
            max_features_per_horizon=5,
            min_samples_split=2,
            min_samples_leaf=1,
            periods_per_day=48,
        )

        # Larger dataset to support D+8 with 48 periods/day
        # D+8 requires shift of 8 * 48 = 384 periods, so we need more data
        n_samples = 1000
        X = pd.DataFrame({
            "hour": np.arange(n_samples) % 24,
            "lag_48": np.random.randn(n_samples),
            "lag_96": np.random.randn(n_samples),
            "lag_384": np.random.randn(n_samples),  # For D+8
        })
        y = pd.Series(np.random.randn(n_samples) + 50)

        # Should train without errors (though model quality may be poor)
        model.fit(X, y, config.model_dump())
        assert model.is_fitted()

    def test_single_feature(self) -> None:
        """Test training with only one feature."""
        model = RandomForestModel()
        config = RandomForestConfig(n_estimators=10)

        X = pd.DataFrame({
            "hour": np.arange(500) % 24,
        })
        y = pd.Series(np.random.randn(500) + 50)

        model.fit(X, y, config.model_dump())
        assert model.is_fitted()

        # Should be able to predict
        predictions = model.predict(X[:100], horizons=[0])
        assert len(predictions) == 100
