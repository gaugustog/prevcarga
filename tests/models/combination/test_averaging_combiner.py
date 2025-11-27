"""Tests for SimpleAveragingCombiner."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.combination.averaging_combiner import SimpleAveragingCombiner
from src.models.combination.data_structures import (
    CombinationResult,
    CombinerConfig,
)


# Fixtures


@pytest.fixture
def sample_predictions():
    """Create sample model predictions with 3 models and 3 horizons."""
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
def sample_predictions_2_models():
    """Create sample predictions with 2 models."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    np.random.seed(42)
    predictions = {
        "model_a": pd.DataFrame(
            {
                "pred_h0": np.array([100.0] * 24),
                "pred_h1": np.array([200.0] * 24),
            },
            index=dates,
        ),
        "model_b": pd.DataFrame(
            {
                "pred_h0": np.array([200.0] * 24),
                "pred_h1": np.array([400.0] * 24),
            },
            index=dates,
        ),
    }

    return predictions


@pytest.fixture
def sample_predictions_5_models():
    """Create sample predictions with 5 models."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    np.random.seed(42)
    predictions = {}
    for i in range(5):
        predictions[f"model_{i}"] = pd.DataFrame(
            {
                "pred_h0": np.random.randn(24) * 10 + 1000 + i * 100,
                "pred_h1": np.random.randn(24) * 10 + 1000 + i * 100,
            },
            index=dates,
        )

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
def predictions_with_nans():
    """Create predictions with missing values."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    predictions = {
        "model_a": pd.DataFrame(
            {
                "pred_h0": [100.0] * 12 + [np.nan] * 12,
                "pred_h1": [200.0] * 24,
            },
            index=dates,
        ),
        "model_b": pd.DataFrame(
            {
                "pred_h0": [200.0] * 24,
                "pred_h1": [np.nan] * 6 + [400.0] * 18,
            },
            index=dates,
        ),
        "model_c": pd.DataFrame(
            {
                "pred_h0": [150.0] * 24,
                "pred_h1": [300.0] * 24,
            },
            index=dates,
        ),
    }

    return predictions


# Test Initialization


def test_initialization_simple():
    """Test simple averaging combiner initialization."""
    combiner = SimpleAveragingCombiner(weighted=False)

    assert combiner.name == "simple_averaging"
    assert combiner.version == "1.0.0"
    assert combiner.weighted is False
    assert not combiner.is_fitted


def test_initialization_weighted():
    """Test weighted averaging combiner initialization."""
    combiner = SimpleAveragingCombiner(weighted=True)

    assert combiner.name == "weighted_averaging"
    assert combiner.version == "1.0.0"
    assert combiner.weighted is True
    assert not combiner.is_fitted


def test_initialization_with_config():
    """Test initialization with custom config."""
    config = CombinerConfig(
        missing_threshold=0.3,
        require_fit=False,
        validate_weights=True,
    )
    combiner = SimpleAveragingCombiner(config=config, weighted=True)

    assert combiner.config.missing_threshold == 0.3
    assert combiner.config.require_fit is False
    assert combiner.config.validate_weights is True


def test_initialization_with_dict_config():
    """Test initialization with dictionary config."""
    config_dict = {
        "missing_threshold": 0.4,
        "require_fit": True,
        "validate_weights": False,
    }
    combiner = SimpleAveragingCombiner(config=config_dict, weighted=False)

    assert combiner.config.missing_threshold == 0.4
    assert combiner.config.require_fit is True
    assert combiner.config.validate_weights is False


# Test Fitting


def test_fit_simple_averaging(sample_predictions, sample_targets):
    """Test fitting simple averaging combiner."""
    combiner = SimpleAveragingCombiner(weighted=False)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted
    assert combiner.fit_timestamp is not None
    assert len(combiner.model_names) == 3

    # Check equal weights
    weights = combiner.get_weights()
    assert len(weights) == 3
    for model, weight in weights.items():
        assert pytest.approx(weight, abs=1e-6) == 1.0 / 3.0


def test_fit_weighted_averaging(sample_predictions, sample_targets):
    """Test fitting weighted averaging combiner."""
    combiner = SimpleAveragingCombiner(weighted=True)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted
    assert combiner.fit_timestamp is not None
    assert len(combiner.model_names) == 3

    # Currently uses equal weights as baseline
    weights = combiner.get_weights()
    assert len(weights) == 3
    for model, weight in weights.items():
        assert pytest.approx(weight, abs=1e-6) == 1.0 / 3.0


def test_fit_validates_predictions(sample_targets):
    """Test that fit validates prediction format."""
    combiner = SimpleAveragingCombiner()

    # Empty predictions
    with pytest.raises(ValueError, match="Predictions dictionary is empty"):
        combiner.fit({}, sample_targets)

    # Predictions with no pred columns
    bad_predictions = {
        "model_a": pd.DataFrame({"value": [1, 2, 3]}),
    }
    with pytest.raises(ValueError, match="No prediction columns found"):
        combiner.fit(bad_predictions, sample_targets)


# Test Simple Averaging Combination


def test_simple_averaging_2_models(sample_predictions_2_models, sample_targets):
    """Test simple averaging with 2 models."""
    combiner = SimpleAveragingCombiner(weighted=False)
    combiner.fit(sample_predictions_2_models, sample_targets)

    result = combiner.combine(sample_predictions_2_models)

    # Check result structure
    assert isinstance(result, CombinationResult)
    assert result.n_models == 2
    assert result.n_samples == 24

    # Check weights are equal
    assert len(result.weights) == 2
    assert pytest.approx(result.weights["model_a"]) == 0.5
    assert pytest.approx(result.weights["model_b"]) == 0.5

    # Check averaging is correct
    # model_a has [100, 200], model_b has [200, 400]
    # Average should be [150, 300]
    assert np.allclose(result.combined_predictions["pred_h0"], 150.0)
    assert np.allclose(result.combined_predictions["pred_h1"], 300.0)


def test_simple_averaging_3_models(sample_predictions, sample_targets):
    """Test simple averaging with 3 models."""
    combiner = SimpleAveragingCombiner(weighted=False)
    combiner.fit(sample_predictions, sample_targets)

    result = combiner.combine(sample_predictions)

    # Check result structure
    assert result.n_models == 3
    assert result.n_samples == 48
    assert len(result.weights) == 3

    # Check equal weights
    for weight in result.weights.values():
        assert pytest.approx(weight, abs=1e-6) == 1.0 / 3.0

    # Check metadata
    assert result.metadata.combination_method == "simple_averaging"
    assert len(result.metadata.models_used) == 3
    assert len(result.metadata.models_excluded) == 0


def test_simple_averaging_5_models(sample_predictions_5_models, sample_targets):
    """Test simple averaging with 5 models."""
    combiner = SimpleAveragingCombiner(weighted=False)
    combiner.fit(sample_predictions_5_models, sample_targets)

    result = combiner.combine(sample_predictions_5_models)

    assert result.n_models == 5
    assert len(result.weights) == 5

    # Check equal weights
    for weight in result.weights.values():
        assert pytest.approx(weight, abs=1e-6) == 1.0 / 5.0


# Test Weighted Averaging Combination


def test_weighted_averaging_custom_weights(sample_predictions_2_models, sample_targets):
    """Test weighted averaging with custom weights."""
    config = CombinerConfig(require_fit=False)
    combiner = SimpleAveragingCombiner(config=config, weighted=True)

    weights = {"model_a": 0.7, "model_b": 0.3}
    result = combiner.combine(sample_predictions_2_models, weights=weights)

    # Check weights are applied
    assert pytest.approx(result.weights["model_a"]) == 0.7
    assert pytest.approx(result.weights["model_b"]) == 0.3

    # Check weighted average is correct
    # model_a: [100, 200], model_b: [200, 400]
    # Weighted: 0.7*[100, 200] + 0.3*[200, 400] = [130, 260]
    assert np.allclose(result.combined_predictions["pred_h0"], 130.0)
    assert np.allclose(result.combined_predictions["pred_h1"], 260.0)


def test_weighted_averaging_equal_weights(sample_predictions_2_models):
    """Test weighted averaging with equal weights."""
    config = CombinerConfig(require_fit=False)
    combiner = SimpleAveragingCombiner(config=config, weighted=True)

    weights = {"model_a": 0.5, "model_b": 0.5}
    result = combiner.combine(sample_predictions_2_models, weights=weights)

    # Should be same as simple averaging
    assert np.allclose(result.combined_predictions["pred_h0"], 150.0)
    assert np.allclose(result.combined_predictions["pred_h1"], 300.0)


def test_weighted_averaging_validates_weight_sum(sample_predictions_2_models):
    """Test that weighted averaging validates weight sum."""
    config = CombinerConfig(require_fit=False, validate_weights=True)
    combiner = SimpleAveragingCombiner(config=config, weighted=True)

    # Weights don't sum to 1
    weights = {"model_a": 0.5, "model_b": 0.6}

    # Should normalize automatically
    result = combiner.combine(sample_predictions_2_models, weights=weights)

    # Weights should be normalized
    assert pytest.approx(sum(result.weights.values())) == 1.0
    assert pytest.approx(result.weights["model_a"], abs=1e-6) == 0.5 / 1.1
    assert pytest.approx(result.weights["model_b"], abs=1e-6) == 0.6 / 1.1


def test_weighted_averaging_missing_model_in_weights(sample_predictions):
    """Test error when custom weights missing a model."""
    config = CombinerConfig(require_fit=False)
    combiner = SimpleAveragingCombiner(config=config, weighted=True)

    # Missing 'xgb' in weights
    weights = {"lgbm": 0.6, "rf": 0.4}

    with pytest.raises(ValueError, match="Custom weights missing for models"):
        combiner.combine(sample_predictions, weights=weights)


def test_weighted_averaging_uses_fitted_weights(sample_predictions, sample_targets):
    """Test that weighted averaging uses fitted weights when custom not provided."""
    combiner = SimpleAveragingCombiner(weighted=True)
    combiner.fit(sample_predictions, sample_targets)

    # Don't provide custom weights
    result = combiner.combine(sample_predictions)

    # Should use fitted weights (currently equal)
    assert len(result.weights) == 3
    for weight in result.weights.values():
        assert pytest.approx(weight, abs=1e-6) == 1.0 / 3.0


# Test Missing Value Handling


def test_handles_nan_in_predictions(predictions_with_nans):
    """Test that NaN values are handled gracefully."""
    config = CombinerConfig(require_fit=False, missing_threshold=1.0)
    combiner = SimpleAveragingCombiner(config=config, weighted=False)

    result = combiner.combine(predictions_with_nans)

    # Check that result has valid values
    assert result.n_models == 3
    assert result.n_samples == 24

    # For h0: model_a has NaN in second half, model_b and model_c are valid
    # First 12: average of 100, 200, 150 = 150
    # Second 12: average of 200, 150 = 175 (model_a excluded)
    expected_h0 = [150.0] * 12 + [175.0] * 12
    assert np.allclose(result.combined_predictions["pred_h0"], expected_h0)

    # For h1: model_b has NaN in first 6, all have values in remaining
    # First 6: average of 200, 300 = 250 (model_b excluded)
    # Remaining 18: average of 200, 400, 300 = 300
    expected_h1 = [250.0] * 6 + [300.0] * 18
    assert np.allclose(result.combined_predictions["pred_h1"], expected_h1)


def test_excludes_models_with_excessive_nans():
    """Test that models with too many NaNs are excluded."""
    dates = pd.date_range("2024-01-01", periods=10, freq="30min")

    predictions = {
        "model_a": pd.DataFrame(
            {
                "pred_h0": [100.0] * 10,
            },
            index=dates,
        ),
        "model_b": pd.DataFrame(
            {
                "pred_h0": [np.nan] * 9 + [200.0],  # 90% missing
            },
            index=dates,
        ),
    }

    config = CombinerConfig(require_fit=False, missing_threshold=0.5)
    combiner = SimpleAveragingCombiner(config=config, weighted=False)

    result = combiner.combine(predictions)

    # model_b should be excluded
    assert result.n_models == 1
    assert "model_a" in result.weights
    assert "model_b" not in result.weights
    assert "model_b" in result.metadata.models_excluded


def test_all_nan_produces_nan():
    """Test that when all models have NaN for a value, result is NaN."""
    dates = pd.date_range("2024-01-01", periods=2, freq="30min")

    predictions = {
        "model_a": pd.DataFrame(
            {
                "pred_h0": [100.0, np.nan],
            },
            index=dates,
        ),
        "model_b": pd.DataFrame(
            {
                "pred_h0": [200.0, np.nan],
            },
            index=dates,
        ),
    }

    config = CombinerConfig(require_fit=False)
    combiner = SimpleAveragingCombiner(config=config, weighted=False)

    result = combiner.combine(predictions)

    # First value should be average, second should be NaN
    assert pytest.approx(result.combined_predictions["pred_h0"].iloc[0]) == 150.0
    assert np.isnan(result.combined_predictions["pred_h0"].iloc[1])


# Test Edge Cases


def test_single_model():
    """Test combination with single model."""
    dates = pd.date_range("2024-01-01", periods=10, freq="30min")

    predictions = {
        "only_model": pd.DataFrame(
            {
                "pred_h0": np.array([100.0] * 10),
            },
            index=dates,
        ),
    }

    config = CombinerConfig(require_fit=False)
    combiner = SimpleAveragingCombiner(config=config, weighted=False)

    result = combiner.combine(predictions)

    assert result.n_models == 1
    assert result.weights["only_model"] == 1.0
    assert np.allclose(result.combined_predictions["pred_h0"], 100.0)


def test_empty_predictions_raises_error():
    """Test that empty predictions raise error."""
    config = CombinerConfig(require_fit=False)
    combiner = SimpleAveragingCombiner(config=config)

    with pytest.raises(ValueError, match="Predictions dictionary is empty"):
        combiner.combine({})


def test_requires_fit_when_configured(sample_predictions):
    """Test that combine requires fit when configured."""
    config = CombinerConfig(require_fit=True)
    combiner = SimpleAveragingCombiner(config=config, weighted=False)

    # Should raise error if not fitted
    with pytest.raises(RuntimeError, match="combiner not fitted"):
        combiner.combine(sample_predictions)


def test_allows_combine_without_fit_when_configured(sample_predictions):
    """Test that combine works without fit when configured."""
    config = CombinerConfig(require_fit=False)
    combiner = SimpleAveragingCombiner(config=config, weighted=False)

    # Should work without fit
    result = combiner.combine(sample_predictions)
    assert result.n_models == 3


# Test Metadata and Result Properties


def test_metadata_correctness(sample_predictions_2_models, sample_targets):
    """Test that metadata is correctly populated."""
    combiner = SimpleAveragingCombiner(weighted=False)
    combiner.fit(sample_predictions_2_models, sample_targets)

    result = combiner.combine(sample_predictions_2_models)

    # Check metadata
    assert result.metadata.combination_method == "simple_averaging"
    assert result.metadata.models_used == ["model_a", "model_b"]
    assert result.metadata.models_excluded == []
    assert result.metadata.processing_time_seconds > 0
    assert "weighted" in result.metadata.configuration
    assert result.metadata.configuration["weighted"] is False


def test_weighted_metadata(sample_predictions_2_models):
    """Test metadata for weighted averaging."""
    config = CombinerConfig(require_fit=False)
    combiner = SimpleAveragingCombiner(config=config, weighted=True)

    weights = {"model_a": 0.6, "model_b": 0.4}
    result = combiner.combine(sample_predictions_2_models, weights=weights)

    assert result.metadata.combination_method == "weighted_averaging"
    assert result.metadata.configuration["weighted"] is True


def test_model_contributions_when_enabled(sample_predictions_2_models):
    """Test that model contributions are stored when enabled."""
    config = CombinerConfig(require_fit=False, store_contributions=True)
    combiner = SimpleAveragingCombiner(config=config, weighted=False)

    result = combiner.combine(sample_predictions_2_models)

    # Check contributions exist
    assert result.model_contributions is not None
    assert "model_a" in result.model_contributions
    assert "model_b" in result.model_contributions

    # Check contribution values (weighted by 0.5 each)
    # model_a has pred_h0=100, weighted by 0.5 = 50
    assert np.allclose(result.model_contributions["model_a"]["pred_h0"], 50.0)
    # model_b has pred_h0=200, weighted by 0.5 = 100
    assert np.allclose(result.model_contributions["model_b"]["pred_h0"], 100.0)


def test_model_contributions_not_stored_by_default(sample_predictions_2_models):
    """Test that model contributions are not stored by default."""
    config = CombinerConfig(require_fit=False, store_contributions=False)
    combiner = SimpleAveragingCombiner(config=config, weighted=False)

    result = combiner.combine(sample_predictions_2_models)

    assert result.model_contributions is None


def test_confidence_intervals_when_enabled():
    """Test that confidence intervals are combined when enabled."""
    dates = pd.date_range("2024-01-01", periods=10, freq="30min")

    predictions = {
        "model_a": pd.DataFrame(
            {
                "pred_h0": [100.0] * 10,
                "pred_h0_lower": [90.0] * 10,
                "pred_h0_upper": [110.0] * 10,
            },
            index=dates,
        ),
        "model_b": pd.DataFrame(
            {
                "pred_h0": [200.0] * 10,
                "pred_h0_lower": [180.0] * 10,
                "pred_h0_upper": [220.0] * 10,
            },
            index=dates,
        ),
    }

    config = CombinerConfig(require_fit=False, store_intervals=True)
    combiner = SimpleAveragingCombiner(config=config, weighted=False)

    result = combiner.combine(predictions)

    # Check intervals exist
    assert result.confidence_intervals is not None
    assert "pred_h0_lower" in result.confidence_intervals.columns
    assert "pred_h0_upper" in result.confidence_intervals.columns

    # Check interval values (average of lower and upper)
    # Lower: (90 + 180) / 2 = 135
    # Upper: (110 + 220) / 2 = 165
    assert np.allclose(result.confidence_intervals["pred_h0_lower"], 135.0)
    assert np.allclose(result.confidence_intervals["pred_h0_upper"], 165.0)


# Test Persistence (Save/Load)


def test_save_and_load(sample_predictions, sample_targets, tmp_path):
    """Test saving and loading combiner."""
    combiner = SimpleAveragingCombiner(weighted=True)
    combiner.fit(sample_predictions, sample_targets)

    # Save
    filepath = tmp_path / "combiner.pkl"
    combiner.save(filepath)

    assert filepath.exists()

    # Load
    loaded_combiner = SimpleAveragingCombiner.load(filepath)

    assert loaded_combiner.name == combiner.name
    assert loaded_combiner.version == combiner.version
    assert loaded_combiner.is_fitted
    assert loaded_combiner.model_names == combiner.model_names
    assert loaded_combiner.get_weights() == combiner.get_weights()


def test_loaded_combiner_works(sample_predictions, sample_targets, tmp_path):
    """Test that loaded combiner can be used for combination."""
    combiner = SimpleAveragingCombiner(weighted=False)
    combiner.fit(sample_predictions, sample_targets)

    # Get result before saving
    result_before = combiner.combine(sample_predictions)

    # Save and load
    filepath = tmp_path / "combiner.pkl"
    combiner.save(filepath)
    loaded_combiner = SimpleAveragingCombiner.load(filepath)

    # Get result after loading
    result_after = loaded_combiner.combine(sample_predictions)

    # Results should be identical
    assert result_before.n_models == result_after.n_models
    assert result_before.weights == result_after.weights
    assert np.allclose(
        result_before.combined_predictions.values,
        result_after.combined_predictions.values,
    )


# Test Result Properties


def test_result_has_correct_horizons(sample_predictions, sample_targets):
    """Test that result has correct horizon information."""
    combiner = SimpleAveragingCombiner(weighted=False)
    combiner.fit(sample_predictions, sample_targets)

    result = combiner.combine(sample_predictions)

    assert result.horizons == [0, 1, 2]


def test_result_dominant_model(sample_predictions_2_models):
    """Test identifying dominant model."""
    config = CombinerConfig(require_fit=False)
    combiner = SimpleAveragingCombiner(config=config, weighted=True)

    weights = {"model_a": 0.7, "model_b": 0.3}
    result = combiner.combine(sample_predictions_2_models, weights=weights)

    assert result.get_dominant_model() == "model_a"


# Test Combiner Properties


def test_get_weights_after_fit(sample_predictions, sample_targets):
    """Test getting weights after fitting."""
    combiner = SimpleAveragingCombiner(weighted=False)
    combiner.fit(sample_predictions, sample_targets)

    weights = combiner.get_weights()

    assert len(weights) == 3
    assert "lgbm" in weights
    assert "rf" in weights
    assert "xgb" in weights
    assert pytest.approx(sum(weights.values())) == 1.0


def test_reset_combiner(sample_predictions, sample_targets):
    """Test resetting combiner to unfitted state."""
    combiner = SimpleAveragingCombiner(weighted=False)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted

    combiner.reset()

    assert not combiner.is_fitted
    assert combiner.model_names == []
    assert combiner.fit_timestamp is None


def test_repr(sample_predictions, sample_targets):
    """Test string representation."""
    combiner = SimpleAveragingCombiner(weighted=False)

    repr_before = repr(combiner)
    assert "not fitted" in repr_before

    combiner.fit(sample_predictions, sample_targets)

    repr_after = repr(combiner)
    assert "fitted" in repr_after
    assert "n_models=3" in repr_after


# Test Simple vs Weighted Behavior


def test_simple_ignores_custom_weights(sample_predictions_2_models, sample_targets):
    """Test that simple averaging ignores custom weights."""
    combiner = SimpleAveragingCombiner(weighted=False)
    combiner.fit(sample_predictions_2_models, sample_targets)

    # Try to provide custom weights (should be ignored)
    weights = {"model_a": 0.8, "model_b": 0.2}
    result = combiner.combine(sample_predictions_2_models, weights=weights)

    # Should still use equal weights
    assert pytest.approx(result.weights["model_a"]) == 0.5
    assert pytest.approx(result.weights["model_b"]) == 0.5


def test_weighted_requires_weights_or_fit(sample_predictions_2_models):
    """Test that weighted averaging requires either custom weights or fitting."""
    config = CombinerConfig(require_fit=True)
    combiner = SimpleAveragingCombiner(config=config, weighted=True)

    # Should fail without fit and without custom weights
    with pytest.raises(RuntimeError, match="combiner not fitted"):
        combiner.combine(sample_predictions_2_models)


# Test Validation


def test_validates_prediction_columns(sample_targets):
    """Test validation of prediction column consistency."""
    dates = pd.date_range("2024-01-01", periods=10, freq="30min")

    predictions = {
        "model_a": pd.DataFrame(
            {
                "pred_h0": [100.0] * 10,
                "pred_h1": [200.0] * 10,
            },
            index=dates,
        ),
        "model_b": pd.DataFrame(
            {
                "pred_h0": [100.0] * 10,
                # Missing pred_h1
            },
            index=dates,
        ),
    }

    combiner = SimpleAveragingCombiner(weighted=False)

    with pytest.raises(ValueError, match="inconsistent columns"):
        combiner.fit(predictions, sample_targets)


def test_validates_prediction_index_alignment(sample_targets):
    """Test validation of prediction index alignment."""
    predictions = {
        "model_a": pd.DataFrame(
            {
                "pred_h0": [100.0] * 10,
            },
            index=pd.date_range("2024-01-01", periods=10, freq="30min"),
        ),
        "model_b": pd.DataFrame(
            {
                "pred_h0": [100.0] * 10,
            },
            index=pd.date_range("2024-01-02", periods=10, freq="30min"),  # Different dates
        ),
    }

    combiner = SimpleAveragingCombiner(weighted=False)

    with pytest.raises(ValueError, match="misaligned index"):
        combiner.fit(predictions, sample_targets)
