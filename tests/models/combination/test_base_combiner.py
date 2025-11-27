"""Tests for BaseCombiner interface."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.combination.base_combiner import BaseCombiner
from src.models.combination.data_structures import (
    CombinationMetadata,
    CombinationResult,
    CombinerConfig,
)


# Mock Combiner Implementation for Testing


class MockCombiner(BaseCombiner):
    """Mock combiner for testing the base interface."""

    @property
    def name(self) -> str:
        return "mock_combiner"

    @property
    def version(self) -> str:
        return "1.0.0"

    def combine(
        self,
        predictions: dict[str, pd.DataFrame],
        metadata: dict[str, Any] | None = None,
    ) -> CombinationResult:
        """Simple average combination for testing."""
        self._check_is_fitted()
        self._validate_predictions(predictions)

        # Handle missing predictions
        cleaned, excluded = self._handle_missing_predictions(predictions)

        start_time = datetime.now()

        # Get prediction columns
        first_pred = next(iter(cleaned.values()))
        pred_cols = [col for col in first_pred.columns if col.startswith("pred_h")]

        # Simple average combination
        combined_df = pd.DataFrame(index=first_pred.index)
        for col in pred_cols:
            values = np.mean([pred[col].values for pred in cleaned.values()], axis=0)
            combined_df[col] = values

        # Use learned weights or equal weights
        if self._global_weights:
            weights = {m: self._global_weights.get(m, 0.0) for m in cleaned.keys()}
            # Normalize
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                weights = {m: 1.0 / len(cleaned) for m in cleaned.keys()}
        else:
            weights = {m: 1.0 / len(cleaned) for m in cleaned.keys()}

        processing_time = (datetime.now() - start_time).total_seconds()

        metadata_obj = self._create_metadata(
            models_used=list(cleaned.keys()),
            models_excluded=excluded,
            processing_time=processing_time,
        )

        return CombinationResult(
            combined_predictions=combined_df,
            weights=weights,
            metadata=metadata_obj,
        )

    def fit(
        self,
        train_predictions: dict[str, pd.DataFrame],
        train_targets: pd.DataFrame,
        validation_predictions: dict[str, pd.DataFrame] | None = None,
        validation_targets: pd.DataFrame | None = None,
    ) -> None:
        """Mock fit that sets equal weights."""
        self._validate_predictions(train_predictions)

        self._model_names = list(train_predictions.keys())
        self._global_weights = self._initialize_weights(self._model_names, equal=True)
        self._fitted = True
        self._fit_timestamp = datetime.now()


# Fixtures


@pytest.fixture
def sample_predictions():
    """Create sample model predictions."""
    dates = pd.date_range("2024-01-01", periods=48, freq="30min")

    np.random.seed(42)
    predictions = {
        "lgbm": pd.DataFrame(
            {
                "pred_h0": np.random.randn(48) * 100 + 1000,
                "pred_h1": np.random.randn(48) * 100 + 1000,
                "pred_h2": np.random.randn(48) * 100 + 1000,
            },
            index=dates,
        ),
        "rf": pd.DataFrame(
            {
                "pred_h0": np.random.randn(48) * 100 + 1050,
                "pred_h1": np.random.randn(48) * 100 + 1050,
                "pred_h2": np.random.randn(48) * 100 + 1050,
            },
            index=dates,
        ),
        "xgb": pd.DataFrame(
            {
                "pred_h0": np.random.randn(48) * 100 + 980,
                "pred_h1": np.random.randn(48) * 100 + 980,
                "pred_h2": np.random.randn(48) * 100 + 980,
            },
            index=dates,
        ),
    }

    return predictions


@pytest.fixture
def sample_targets():
    """Create sample target values."""
    dates = pd.date_range("2024-01-01", periods=48, freq="30min")

    np.random.seed(42)
    targets = pd.DataFrame(
        {
            "h0": np.random.randn(48) * 100 + 1000,
            "h1": np.random.randn(48) * 100 + 1000,
            "h2": np.random.randn(48) * 100 + 1000,
        },
        index=dates,
    )

    return targets


@pytest.fixture
def combiner():
    """Create mock combiner instance."""
    return MockCombiner()


@pytest.fixture
def fitted_combiner(sample_predictions, sample_targets):
    """Create fitted mock combiner."""
    combiner = MockCombiner()
    combiner.fit(sample_predictions, sample_targets)
    return combiner


# Tests for BaseCombiner Interface


class TestBaseCombinerInterface:
    """Tests for BaseCombiner abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseCombiner cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseCombiner()

    def test_mock_combiner_initialization(self):
        """Test mock combiner initialization."""
        combiner = MockCombiner()

        assert combiner.name == "mock_combiner"
        assert combiner.version == "1.0.0"
        assert not combiner.is_fitted
        assert combiner.fit_timestamp is None
        assert combiner.model_names == []

    def test_initialization_with_config(self):
        """Test initialization with configuration."""
        config = CombinerConfig(
            missing_threshold=0.3,
            require_fit=False,
            store_contributions=True,
        )
        combiner = MockCombiner(config=config)

        assert combiner.config.missing_threshold == 0.3
        assert combiner.config.require_fit is False
        assert combiner.config.store_contributions is True

    def test_initialization_with_dict_config(self):
        """Test initialization with dictionary config."""
        config_dict = {
            "missing_threshold": 0.4,
            "require_fit": True,
            "log_level": "DEBUG",
        }
        combiner = MockCombiner(config=config_dict)

        assert combiner.config.missing_threshold == 0.4
        assert combiner.config.require_fit is True
        assert combiner.config.log_level == "DEBUG"


class TestCombinerFit:
    """Tests for combiner fit functionality."""

    def test_fit_basic(self, combiner, sample_predictions, sample_targets):
        """Test basic fit operation."""
        combiner.fit(sample_predictions, sample_targets)

        assert combiner.is_fitted
        assert combiner.fit_timestamp is not None
        assert set(combiner.model_names) == {"lgbm", "rf", "xgb"}

    def test_fit_with_validation_data(self, combiner, sample_predictions, sample_targets):
        """Test fit with validation data."""
        combiner.fit(
            train_predictions=sample_predictions,
            train_targets=sample_targets,
            validation_predictions=sample_predictions,
            validation_targets=sample_targets,
        )

        assert combiner.is_fitted

    def test_fit_invalid_predictions(self, combiner, sample_targets):
        """Test fit with invalid predictions."""
        with pytest.raises(ValueError, match="empty"):
            combiner.fit({}, sample_targets)

    def test_get_weights_after_fit(self, fitted_combiner):
        """Test getting weights after fit."""
        weights = fitted_combiner.get_weights()

        assert len(weights) == 3
        assert np.isclose(sum(weights.values()), 1.0)
        assert all(w >= 0 for w in weights.values())

    def test_get_weights_not_fitted(self, combiner):
        """Test getting weights when not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            combiner.get_weights()


class TestCombinerCombine:
    """Tests for combiner combine functionality."""

    def test_combine_basic(self, fitted_combiner, sample_predictions):
        """Test basic combination operation."""
        result = fitted_combiner.combine(sample_predictions)

        assert isinstance(result, CombinationResult)
        assert "pred_h0" in result.combined_predictions.columns
        assert "pred_h1" in result.combined_predictions.columns
        assert "pred_h2" in result.combined_predictions.columns
        assert len(result.weights) == 3
        assert np.isclose(sum(result.weights.values()), 1.0)

    def test_combine_result_properties(self, fitted_combiner, sample_predictions):
        """Test CombinationResult properties."""
        result = fitted_combiner.combine(sample_predictions)

        assert result.n_models == 3
        assert result.n_samples == 48
        assert result.horizons == [0, 1, 2]
        assert result.get_dominant_model() in ["lgbm", "rf", "xgb"]

    def test_combine_not_fitted(self, combiner, sample_predictions):
        """Test combine when not fitted raises error."""
        with pytest.raises(RuntimeError, match="not fitted"):
            combiner.combine(sample_predictions)

    def test_combine_with_require_fit_false(self, sample_predictions, sample_targets):
        """Test combine with require_fit=False."""
        config = CombinerConfig(require_fit=False)
        combiner = MockCombiner(config=config)

        # Should work without fitting (mock implementation handles this)
        combiner.fit(sample_predictions, sample_targets)
        result = combiner.combine(sample_predictions)

        assert isinstance(result, CombinationResult)

    def test_combine_metadata(self, fitted_combiner, sample_predictions):
        """Test combination metadata."""
        result = fitted_combiner.combine(sample_predictions)

        assert result.metadata.combination_method == "mock_combiner"
        assert len(result.metadata.models_used) == 3
        assert result.metadata.models_excluded == []
        assert result.metadata.processing_time_seconds >= 0


class TestPredictionValidation:
    """Tests for prediction validation."""

    def test_validate_empty_predictions(self, combiner):
        """Test validation with empty predictions."""
        with pytest.raises(ValueError, match="empty"):
            combiner._validate_predictions({})

    def test_validate_empty_dataframe(self, combiner):
        """Test validation with empty DataFrame."""
        predictions = {"model1": pd.DataFrame()}

        with pytest.raises(ValueError, match="empty"):
            combiner._validate_predictions(predictions)

    def test_validate_no_prediction_columns(self, combiner):
        """Test validation with no pred_h columns."""
        dates = pd.date_range("2024-01-01", periods=10, freq="30min")
        predictions = {
            "model1": pd.DataFrame({"other_col": np.random.randn(10)}, index=dates),
        }

        with pytest.raises(ValueError, match="No prediction columns"):
            combiner._validate_predictions(predictions)

    def test_validate_inconsistent_columns(self, combiner):
        """Test validation with inconsistent columns."""
        dates = pd.date_range("2024-01-01", periods=10, freq="30min")
        predictions = {
            "model1": pd.DataFrame({"pred_h0": np.random.randn(10)}, index=dates),
            "model2": pd.DataFrame(
                {"pred_h0": np.random.randn(10), "pred_h1": np.random.randn(10)},
                index=dates,
            ),
        }

        with pytest.raises(ValueError, match="inconsistent columns"):
            combiner._validate_predictions(predictions)

    def test_validate_misaligned_index(self, combiner):
        """Test validation with misaligned indices."""
        dates1 = pd.date_range("2024-01-01", periods=10, freq="30min")
        dates2 = pd.date_range("2024-01-02", periods=10, freq="30min")

        predictions = {
            "model1": pd.DataFrame({"pred_h0": np.random.randn(10)}, index=dates1),
            "model2": pd.DataFrame({"pred_h0": np.random.randn(10)}, index=dates2),
        }

        with pytest.raises(ValueError, match="misaligned"):
            combiner._validate_predictions(predictions)


class TestMissingPredictionHandling:
    """Tests for missing prediction handling."""

    def test_handle_missing_excludes_high_missing(self, combiner):
        """Test that models with high missing rate are excluded."""
        dates = pd.date_range("2024-01-01", periods=10, freq="30min")

        predictions = {
            "good_model": pd.DataFrame({"pred_h0": np.random.randn(10)}, index=dates),
            "bad_model": pd.DataFrame(
                {"pred_h0": [np.nan] * 8 + [1.0, 2.0]},  # 80% missing
                index=dates,
            ),
        }

        cleaned, excluded = combiner._handle_missing_predictions(predictions)

        assert "good_model" in cleaned
        assert "bad_model" in excluded
        assert "bad_model" not in cleaned

    def test_handle_missing_all_excluded_raises(self, combiner):
        """Test error when all models are excluded."""
        dates = pd.date_range("2024-01-01", periods=10, freq="30min")

        predictions = {
            "bad1": pd.DataFrame({"pred_h0": [np.nan] * 10}, index=dates),
            "bad2": pd.DataFrame({"pred_h0": [np.nan] * 10}, index=dates),
        }

        with pytest.raises(ValueError, match="All models excluded"):
            combiner._handle_missing_predictions(predictions)


class TestWeightManagement:
    """Tests for weight management."""

    def test_initialize_equal_weights(self, combiner):
        """Test equal weight initialization."""
        weights = combiner._initialize_weights(["m1", "m2", "m3"], equal=True)

        assert len(weights) == 3
        assert np.isclose(sum(weights.values()), 1.0)
        assert all(np.isclose(w, 1 / 3) for w in weights.values())

    def test_validate_weights_valid(self, combiner):
        """Test weight validation with valid weights."""
        weights = {"m1": 0.5, "m2": 0.3, "m3": 0.2}
        combiner._validate_weights(weights)  # Should not raise

    def test_validate_weights_negative(self, combiner):
        """Test weight validation with negative weights."""
        weights = {"m1": 0.5, "m2": -0.1, "m3": 0.6}

        with pytest.raises(ValueError, match="non-negative"):
            combiner._validate_weights(weights)

    def test_validate_weights_wrong_sum(self, combiner):
        """Test weight validation with weights not summing to 1."""
        weights = {"m1": 0.5, "m2": 0.3, "m3": 0.3}  # Sum = 1.1

        with pytest.raises(ValueError, match="sum to 1"):
            combiner._validate_weights(weights)

    def test_normalize_weights(self, combiner):
        """Test weight normalization."""
        weights = {"m1": 2.0, "m2": 3.0, "m3": 5.0}
        normalized = combiner._normalize_weights(weights)

        assert np.isclose(sum(normalized.values()), 1.0)
        assert np.isclose(normalized["m1"], 0.2)
        assert np.isclose(normalized["m2"], 0.3)
        assert np.isclose(normalized["m3"], 0.5)


class TestSaveLoad:
    """Tests for save and load functionality."""

    def test_save_and_load(self, fitted_combiner, tmp_path):
        """Test combiner save and load."""
        filepath = tmp_path / "combiner.pkl"

        fitted_combiner.save(filepath)
        assert filepath.exists()

        loaded = MockCombiner.load(filepath)

        assert loaded.name == fitted_combiner.name
        assert loaded.version == fitted_combiner.version
        assert loaded.is_fitted == fitted_combiner.is_fitted
        assert loaded.model_names == fitted_combiner.model_names

    def test_save_load_preserves_weights(self, fitted_combiner, tmp_path):
        """Test that save/load preserves weights."""
        filepath = tmp_path / "combiner.pkl"

        original_weights = fitted_combiner.get_weights()
        fitted_combiner.save(filepath)

        loaded = MockCombiner.load(filepath)
        loaded_weights = loaded.get_weights()

        assert original_weights == loaded_weights


class TestReset:
    """Tests for combiner reset functionality."""

    def test_reset(self, fitted_combiner):
        """Test combiner reset."""
        assert fitted_combiner.is_fitted

        fitted_combiner.reset()

        assert not fitted_combiner.is_fitted
        assert fitted_combiner.fit_timestamp is None
        assert fitted_combiner.model_names == []

    def test_reset_clears_performance_history(self, fitted_combiner, sample_predictions):
        """Test that reset clears performance history."""
        # Generate some history
        _ = fitted_combiner.combine(sample_predictions)

        fitted_combiner.reset()

        assert fitted_combiner.get_performance_history() == []


class TestRepr:
    """Tests for string representation."""

    def test_repr_not_fitted(self, combiner):
        """Test repr for unfitted combiner."""
        repr_str = repr(combiner)

        assert "MockCombiner" in repr_str
        assert "mock_combiner" in repr_str
        assert "not fitted" in repr_str

    def test_repr_fitted(self, fitted_combiner):
        """Test repr for fitted combiner."""
        repr_str = repr(fitted_combiner)

        assert "MockCombiner" in repr_str
        assert "fitted" in repr_str
        assert "n_models=3" in repr_str
