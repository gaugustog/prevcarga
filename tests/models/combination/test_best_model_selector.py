"""Tests for BestModelSelector."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.combination.best_model_selector import BestModelSelector
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
                "pred_h0": np.random.randn(48) * 10 + 1000,  # Good model
                "pred_h1": np.random.randn(48) * 10 + 1000,
                "pred_h2": np.random.randn(48) * 10 + 1000,
            },
            index=dates,
        ),
        "rf": pd.DataFrame(
            {
                "pred_h0": np.random.randn(48) * 20 + 1050,  # Medium model
                "pred_h1": np.random.randn(48) * 20 + 1050,
                "pred_h2": np.random.randn(48) * 20 + 1050,
            },
            index=dates,
        ),
        "xgb": pd.DataFrame(
            {
                "pred_h0": np.random.randn(48) * 30 + 980,  # Worst model
                "pred_h1": np.random.randn(48) * 30 + 980,
                "pred_h2": np.random.randn(48) * 30 + 980,
            },
            index=dates,
        ),
    }

    return predictions


@pytest.fixture
def sample_predictions_2_models():
    """Create sample predictions with 2 models for simple testing."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    # model_a is better (lower error)
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
                "pred_h0": np.array([200.0] * 24),  # Higher error
                "pred_h1": np.array([400.0] * 24),
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
            "h0": np.random.randn(48) * 10 + 1000,
            "h1": np.random.randn(48) * 10 + 1000,
            "h2": np.random.randn(48) * 10 + 1000,
        },
        index=dates,
    )

    return targets


@pytest.fixture
def sample_targets_2_models():
    """Create target values for 2 model test."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    targets = pd.DataFrame(
        {
            "h0": np.array([105.0] * 24),  # Closer to model_a
            "h1": np.array([210.0] * 24),
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


@pytest.fixture
def perfect_and_poor_models():
    """Create predictions where one model is perfect and one is poor."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    # Perfect model
    perfect_values_h0 = np.array([100.0] * 24)
    perfect_values_h1 = np.array([200.0] * 24)

    # Poor model (off by 50)
    poor_values_h0 = perfect_values_h0 + 50
    poor_values_h1 = perfect_values_h1 + 50

    predictions = {
        "perfect": pd.DataFrame(
            {
                "pred_h0": perfect_values_h0,
                "pred_h1": perfect_values_h1,
            },
            index=dates,
        ),
        "poor": pd.DataFrame(
            {
                "pred_h0": poor_values_h0,
                "pred_h1": poor_values_h1,
            },
            index=dates,
        ),
    }

    targets = pd.DataFrame(
        {
            "h0": perfect_values_h0,
            "h1": perfect_values_h1,
        },
        index=dates,
    )

    return predictions, targets


# Test Initialization


def test_initialization_default():
    """Test selector initialization with default parameters."""
    selector = BestModelSelector()

    assert selector.name == "best_model_selector"
    assert selector.version == "1.0.0"
    assert selector.selection_criterion == "min_mape"
    assert selector.per_horizon is False
    assert selector.fallback_strategy == "error"
    assert not selector.is_fitted


def test_initialization_with_criterion():
    """Test initialization with different selection criteria."""
    # Test each criterion
    for criterion in ["min_mape", "min_mae", "min_rmse", "max_r2", "min_combined"]:
        selector = BestModelSelector(selection_criterion=criterion)
        assert selector.selection_criterion == criterion


def test_initialization_per_horizon():
    """Test initialization with per_horizon=True."""
    selector = BestModelSelector(per_horizon=True)
    assert selector.per_horizon is True


def test_initialization_with_config():
    """Test initialization with custom config."""
    config = CombinerConfig(
        missing_threshold=0.3,
        require_fit=True,
        validate_weights=False,
    )
    selector = BestModelSelector(config=config)

    assert selector.config.missing_threshold == 0.3
    assert selector.config.require_fit is True
    assert selector.config.validate_weights is False


def test_initialization_with_dict_config():
    """Test initialization with dictionary config."""
    config_dict = {
        "missing_threshold": 0.4,
        "require_fit": False,
    }
    selector = BestModelSelector(config=config_dict)

    assert selector.config.missing_threshold == 0.4
    assert selector.config.require_fit is False


def test_initialization_combined_weights():
    """Test initialization with custom combined weights."""
    weights = {"mape": 0.5, "mae": 0.25, "rmse": 0.25}
    selector = BestModelSelector(
        selection_criterion="min_combined",
        combined_weights=weights,
    )
    assert selector.combined_weights == weights


def test_initialization_invalid_combined_weights():
    """Test that invalid combined weights raise error."""
    with pytest.raises(ValueError, match="combined_weights must sum to 1"):
        BestModelSelector(
            selection_criterion="min_combined",
            combined_weights={"mape": 0.5, "mae": 0.3},  # Sum = 0.8
        )


# Test fit() Method


def test_fit_selects_best_model(sample_predictions, sample_targets):
    """Test that fit() correctly selects best model."""
    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(sample_predictions, sample_targets)

    assert selector.is_fitted
    assert selector.fit_timestamp is not None
    assert selector._best_model is not None
    assert selector._best_model in sample_predictions


def test_fit_min_mape_criterion(perfect_and_poor_models):
    """Test fit with min_mape criterion."""
    predictions, targets = perfect_and_poor_models

    selector = BestModelSelector(selection_criterion="min_mape")
    selector.fit(predictions, targets)

    assert selector.get_best_model() == "perfect"
    scores = selector.get_model_scores()
    assert scores["perfect"]["mape"] < scores["poor"]["mape"]


def test_fit_min_mae_criterion(perfect_and_poor_models):
    """Test fit with min_mae criterion."""
    predictions, targets = perfect_and_poor_models

    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(predictions, targets)

    assert selector.get_best_model() == "perfect"
    scores = selector.get_model_scores()
    assert scores["perfect"]["mae"] < scores["poor"]["mae"]


def test_fit_min_rmse_criterion(perfect_and_poor_models):
    """Test fit with min_rmse criterion."""
    predictions, targets = perfect_and_poor_models

    selector = BestModelSelector(selection_criterion="min_rmse")
    selector.fit(predictions, targets)

    assert selector.get_best_model() == "perfect"
    scores = selector.get_model_scores()
    assert scores["perfect"]["rmse"] < scores["poor"]["rmse"]


def test_fit_max_r2_criterion(perfect_and_poor_models):
    """Test fit with max_r2 criterion."""
    predictions, targets = perfect_and_poor_models

    selector = BestModelSelector(selection_criterion="max_r2")
    selector.fit(predictions, targets)

    assert selector.get_best_model() == "perfect"
    scores = selector.get_model_scores()
    # Perfect model should have R² = 1.0 or very close
    assert scores["perfect"]["r2"] > scores["poor"]["r2"]


def test_fit_min_combined_criterion(perfect_and_poor_models):
    """Test fit with min_combined criterion."""
    predictions, targets = perfect_and_poor_models

    selector = BestModelSelector(
        selection_criterion="min_combined",
        combined_weights={"mape": 0.4, "mae": 0.3, "rmse": 0.3},
    )
    selector.fit(predictions, targets)

    assert selector.get_best_model() == "perfect"


def test_fit_per_horizon(sample_predictions, sample_targets):
    """Test fit with per_horizon=True."""
    selector = BestModelSelector(
        selection_criterion="min_mae",
        per_horizon=True,
    )
    selector.fit(sample_predictions, sample_targets)

    assert selector.is_fitted
    best_models = selector.get_best_models()
    assert "pred_h0" in best_models
    assert "pred_h1" in best_models
    assert "pred_h2" in best_models
    assert all(model in sample_predictions for model in best_models.values())


def test_fit_with_validation_set(sample_predictions, sample_targets):
    """Test fit using validation set for selection."""
    # Create separate validation set
    dates = pd.date_range("2024-02-01", periods=24, freq="30min")
    val_predictions = {
        "lgbm": pd.DataFrame(
            {
                "pred_h0": np.random.randn(24) * 10 + 1000,
                "pred_h1": np.random.randn(24) * 10 + 1000,
            },
            index=dates,
        ),
        "rf": pd.DataFrame(
            {
                "pred_h0": np.random.randn(24) * 20 + 1050,
                "pred_h1": np.random.randn(24) * 20 + 1050,
            },
            index=dates,
        ),
    }
    val_targets = pd.DataFrame(
        {
            "h0": np.random.randn(24) * 10 + 1000,
            "h1": np.random.randn(24) * 10 + 1000,
        },
        index=dates,
    )

    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(
        sample_predictions,
        sample_targets,
        validation_predictions=val_predictions,
        validation_targets=val_targets,
    )

    assert selector.is_fitted
    assert selector._best_model in val_predictions


def test_fit_invalid_predictions():
    """Test that fit validates prediction format."""
    selector = BestModelSelector()

    # Empty predictions
    with pytest.raises(ValueError, match="empty"):
        selector.fit({}, pd.DataFrame())


def test_fit_stores_model_scores(sample_predictions_2_models, sample_targets_2_models):
    """Test that fit stores model scores."""
    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    scores = selector.get_model_scores()
    assert "model_a" in scores
    assert "model_b" in scores
    assert "mape" in scores["model_a"]
    assert "mae" in scores["model_a"]
    assert "rmse" in scores["model_a"]
    assert "r2" in scores["model_a"]


# Test combine() Method


def test_combine_returns_best_model_forecast(sample_predictions_2_models, sample_targets_2_models):
    """Test that combine() returns selected model's forecast."""
    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    result = selector.combine(sample_predictions_2_models)

    assert isinstance(result, CombinationResult)
    assert result.n_models == 1
    best_model = selector.get_best_model()
    assert best_model in result.weights
    assert result.weights[best_model] == 1.0

    # Check that predictions match selected model
    expected = sample_predictions_2_models[best_model]
    pd.testing.assert_frame_equal(result.combined_predictions, expected)


def test_combine_with_override_model(sample_predictions_2_models, sample_targets_2_models):
    """Test combine with override_model parameter."""
    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    # Override to use different model
    result = selector.combine(sample_predictions_2_models, override_model="model_b")

    assert result.weights == {"model_b": 1.0}
    expected = sample_predictions_2_models["model_b"]
    pd.testing.assert_frame_equal(result.combined_predictions, expected)


def test_combine_not_fitted_with_require_fit():
    """Test that combine raises error if not fitted when required."""
    config = CombinerConfig(require_fit=True)
    selector = BestModelSelector(config=config)

    dates = pd.date_range("2024-01-01", periods=24, freq="30min")
    predictions = {
        "model_a": pd.DataFrame({"pred_h0": [100.0] * 24}, index=dates),
    }

    with pytest.raises(RuntimeError, match="not fitted"):
        selector.combine(predictions)


def test_combine_per_horizon(sample_predictions, sample_targets):
    """Test combine with per_horizon selection."""
    selector = BestModelSelector(
        selection_criterion="min_mae",
        per_horizon=True,
    )
    selector.fit(sample_predictions, sample_targets)

    result = selector.combine(sample_predictions)

    assert isinstance(result, CombinationResult)
    assert result.combined_predictions.shape == (48, 3)

    # Check metadata contains per-horizon selections
    assert "selected_models" in result.metadata.configuration
    selected_models = result.metadata.configuration["selected_models"]
    assert "pred_h0" in selected_models
    assert "pred_h1" in selected_models
    assert "pred_h2" in selected_models


def test_combine_metadata_contains_selection_info(
    sample_predictions_2_models, sample_targets_2_models
):
    """Test that combine result metadata contains selection information."""
    selector = BestModelSelector(selection_criterion="min_mape")
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    result = selector.combine(sample_predictions_2_models)

    assert "selected_model" in result.metadata.configuration
    assert "selection_criterion" in result.metadata.configuration
    assert "model_scores" in result.metadata.configuration
    assert result.metadata.configuration["selection_criterion"] == "min_mape"


def test_combine_handles_missing_best_model_error_strategy():
    """Test combine raises error when best model missing (error strategy)."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    train_preds = {
        "model_a": pd.DataFrame({"pred_h0": [100.0] * 24}, index=dates),
        "model_b": pd.DataFrame({"pred_h0": [200.0] * 24}, index=dates),
    }
    targets = pd.DataFrame({"h0": [105.0] * 24}, index=dates)

    selector = BestModelSelector(
        selection_criterion="min_mae",
        fallback_strategy="error",
    )
    selector.fit(train_preds, targets)

    # Test predictions missing the best model
    test_preds = {
        "model_b": pd.DataFrame({"pred_h0": [200.0] * 24}, index=dates),
    }

    with pytest.raises(KeyError, match="not available"):
        selector.combine(test_preds)


def test_combine_handles_missing_best_model_second_best_strategy():
    """Test combine uses second-best when best model missing."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    train_preds = {
        "model_a": pd.DataFrame({"pred_h0": [100.0] * 24}, index=dates),
        "model_b": pd.DataFrame({"pred_h0": [110.0] * 24}, index=dates),
        "model_c": pd.DataFrame({"pred_h0": [200.0] * 24}, index=dates),
    }
    targets = pd.DataFrame({"h0": [105.0] * 24}, index=dates)

    selector = BestModelSelector(
        selection_criterion="min_mae",
        fallback_strategy="second_best",
    )
    selector.fit(train_preds, targets)

    # Test predictions missing the best model
    test_preds = {
        "model_b": pd.DataFrame({"pred_h0": [110.0] * 24}, index=dates),
        "model_c": pd.DataFrame({"pred_h0": [200.0] * 24}, index=dates),
    }

    result = selector.combine(test_preds)
    # Should use model_b (second best)
    assert "model_b" in result.weights
    assert result.weights["model_b"] == 1.0


def test_combine_handles_missing_best_model_simple_average_strategy():
    """Test combine uses fallback when best model missing (simple_average strategy)."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    train_preds = {
        "model_a": pd.DataFrame({"pred_h0": [100.0] * 24}, index=dates),
        "model_b": pd.DataFrame({"pred_h0": [200.0] * 24}, index=dates),
    }
    targets = pd.DataFrame({"h0": [105.0] * 24}, index=dates)

    selector = BestModelSelector(
        selection_criterion="min_mae",
        fallback_strategy="simple_average",
    )
    selector.fit(train_preds, targets)

    # Test predictions missing the best model
    test_preds = {
        "model_b": pd.DataFrame({"pred_h0": [200.0] * 24}, index=dates),
    }

    result = selector.combine(test_preds)
    # Should use model_b as fallback
    assert "model_b" in result.weights


def test_combine_with_nan_predictions(predictions_with_nans, sample_targets_2_models):
    """Test combine handles NaN predictions correctly."""
    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(predictions_with_nans, sample_targets_2_models)

    result = selector.combine(predictions_with_nans)

    assert isinstance(result, CombinationResult)
    # Should select model with fewest NaN values and best performance


# Test Getter Methods


def test_get_best_model(sample_predictions_2_models, sample_targets_2_models):
    """Test get_best_model() returns correct model."""
    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    best_model = selector.get_best_model()
    assert best_model in sample_predictions_2_models
    assert isinstance(best_model, str)


def test_get_best_model_not_fitted():
    """Test get_best_model() raises error if not fitted."""
    selector = BestModelSelector()

    with pytest.raises(RuntimeError, match="not fitted"):
        selector.get_best_model()


def test_get_best_model_with_per_horizon_raises_error(sample_predictions, sample_targets):
    """Test get_best_model() raises error when per_horizon=True."""
    selector = BestModelSelector(per_horizon=True)
    selector.fit(sample_predictions, sample_targets)

    with pytest.raises(RuntimeError, match="not available for per_horizon=True"):
        selector.get_best_model()


def test_get_best_models(sample_predictions, sample_targets):
    """Test get_best_models() returns correct dict."""
    selector = BestModelSelector(
        selection_criterion="min_mae",
        per_horizon=True,
    )
    selector.fit(sample_predictions, sample_targets)

    best_models = selector.get_best_models()
    assert isinstance(best_models, dict)
    assert "pred_h0" in best_models
    assert all(isinstance(model, str) for model in best_models.values())


def test_get_best_models_not_fitted():
    """Test get_best_models() raises error if not fitted."""
    selector = BestModelSelector(per_horizon=True)

    with pytest.raises(RuntimeError, match="not fitted"):
        selector.get_best_models()


def test_get_best_models_without_per_horizon_raises_error(
    sample_predictions_2_models, sample_targets_2_models
):
    """Test get_best_models() raises error when per_horizon=False."""
    selector = BestModelSelector(per_horizon=False)
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    with pytest.raises(RuntimeError, match="only available for per_horizon=True"):
        selector.get_best_models()


def test_get_model_scores(sample_predictions_2_models, sample_targets_2_models):
    """Test get_model_scores() returns correct scores."""
    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    scores = selector.get_model_scores()
    assert isinstance(scores, dict)
    assert "model_a" in scores
    assert "model_b" in scores
    assert all(isinstance(model_scores, dict) for model_scores in scores.values())
    assert all("mape" in model_scores for model_scores in scores.values())
    assert all("mae" in model_scores for model_scores in scores.values())


def test_get_model_scores_not_fitted():
    """Test get_model_scores() raises error if not fitted."""
    selector = BestModelSelector()

    with pytest.raises(RuntimeError, match="not fitted"):
        selector.get_model_scores()


# Test Persistence (save/load)


def test_save_and_load(tmp_path, sample_predictions_2_models, sample_targets_2_models):
    """Test saving and loading selector."""
    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    # Save
    filepath = tmp_path / "selector.pkl"
    selector.save(filepath)
    assert filepath.exists()

    # Load
    loaded_selector = BestModelSelector.load(filepath)
    assert loaded_selector.name == selector.name
    assert loaded_selector.version == selector.version
    assert loaded_selector.is_fitted
    assert loaded_selector.selection_criterion == selector.selection_criterion
    assert loaded_selector._best_model == selector._best_model
    assert loaded_selector._model_scores == selector._model_scores


def test_save_and_load_per_horizon(tmp_path, sample_predictions, sample_targets):
    """Test saving and loading per-horizon selector."""
    selector = BestModelSelector(
        selection_criterion="min_rmse",
        per_horizon=True,
    )
    selector.fit(sample_predictions, sample_targets)

    # Save
    filepath = tmp_path / "selector_per_horizon.pkl"
    selector.save(filepath)

    # Load
    loaded_selector = BestModelSelector.load(filepath)
    assert loaded_selector.per_horizon is True
    assert loaded_selector._best_models_per_horizon == selector._best_models_per_horizon


def test_save_creates_directory(tmp_path, sample_predictions_2_models, sample_targets_2_models):
    """Test that save creates parent directories."""
    selector = BestModelSelector()
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    filepath = tmp_path / "subdir" / "selector.pkl"
    selector.save(filepath)

    assert filepath.exists()
    assert filepath.parent.exists()


# Test Edge Cases


def test_single_model():
    """Test selector with only one model."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    predictions = {
        "model_a": pd.DataFrame({"pred_h0": [100.0] * 24}, index=dates),
    }
    targets = pd.DataFrame({"h0": [105.0] * 24}, index=dates)

    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(predictions, targets)

    assert selector.get_best_model() == "model_a"

    result = selector.combine(predictions)
    assert result.weights == {"model_a": 1.0}


def test_all_models_equal_performance():
    """Test selector when all models have equal performance."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    # All models have identical predictions
    predictions = {
        "model_a": pd.DataFrame({"pred_h0": [100.0] * 24}, index=dates),
        "model_b": pd.DataFrame({"pred_h0": [100.0] * 24}, index=dates),
        "model_c": pd.DataFrame({"pred_h0": [100.0] * 24}, index=dates),
    }
    targets = pd.DataFrame({"h0": [105.0] * 24}, index=dates)

    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(predictions, targets)

    # Should select one (first one that matches min criterion)
    best_model = selector.get_best_model()
    assert best_model in predictions

    # All models should have same scores
    scores = selector.get_model_scores()
    mae_values = [scores[model]["mae"] for model in scores]
    assert len(set(mae_values)) == 1  # All equal


def test_multiple_horizons_different_best():
    """Test per-horizon selection with different best models per horizon."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    # model_a best for h0, model_b best for h1
    predictions = {
        "model_a": pd.DataFrame(
            {
                "pred_h0": np.array([100.0] * 24),
                "pred_h1": np.array([300.0] * 24),
            },
            index=dates,
        ),
        "model_b": pd.DataFrame(
            {
                "pred_h0": np.array([150.0] * 24),
                "pred_h1": np.array([200.0] * 24),
            },
            index=dates,
        ),
    }
    targets = pd.DataFrame(
        {
            "h0": np.array([105.0] * 24),
            "h1": np.array([205.0] * 24),
        },
        index=dates,
    )

    selector = BestModelSelector(
        selection_criterion="min_mae",
        per_horizon=True,
    )
    selector.fit(predictions, targets)

    best_models = selector.get_best_models()
    assert best_models["pred_h0"] == "model_a"
    assert best_models["pred_h1"] == "model_b"

    result = selector.combine(predictions)
    # Should use model_a for h0 and model_b for h1
    np.testing.assert_array_equal(
        result.combined_predictions["pred_h0"].values,
        predictions["model_a"]["pred_h0"].values,
    )
    np.testing.assert_array_equal(
        result.combined_predictions["pred_h1"].values,
        predictions["model_b"]["pred_h1"].values,
    )


def test_with_many_models():
    """Test selector with many models (5+)."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    np.random.seed(42)
    predictions = {}
    for i in range(10):
        predictions[f"model_{i}"] = pd.DataFrame(
            {
                "pred_h0": np.random.randn(24) * 10 + 1000 + i * 5,
            },
            index=dates,
        )

    targets = pd.DataFrame(
        {"h0": np.random.randn(24) * 10 + 1000},
        index=dates,
    )

    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(predictions, targets)

    best_model = selector.get_best_model()
    assert best_model in predictions

    # Check all models have scores
    scores = selector.get_model_scores()
    assert len(scores) == 10


# Test Integration


def test_full_workflow(sample_predictions, sample_targets):
    """Test complete workflow: fit, combine, save, load."""
    # Fit
    selector = BestModelSelector(
        selection_criterion="min_mape",
        fallback_strategy="second_best",
    )
    selector.fit(sample_predictions, sample_targets)

    # Get best model
    best_model = selector.get_best_model()
    assert best_model in sample_predictions

    # Combine
    result = selector.combine(sample_predictions)
    assert result.weights[best_model] == 1.0

    # Verify metadata
    assert result.metadata.configuration["selected_model"] == best_model
    assert result.metadata.configuration["selection_criterion"] == "min_mape"

    # Get scores
    scores = selector.get_model_scores()
    assert len(scores) == len(sample_predictions)


def test_comparison_with_different_criteria(perfect_and_poor_models):
    """Test that all criteria select the perfect model."""
    predictions, targets = perfect_and_poor_models

    for criterion in ["min_mape", "min_mae", "min_rmse", "max_r2"]:
        selector = BestModelSelector(selection_criterion=criterion)
        selector.fit(predictions, targets)

        best_model = selector.get_best_model()
        assert best_model == "perfect", f"Failed for criterion: {criterion}"


def test_reset_after_fit(sample_predictions_2_models, sample_targets_2_models):
    """Test reset() clears fitted state."""
    selector = BestModelSelector(selection_criterion="min_mae")
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    assert selector.is_fitted
    assert selector._best_model is not None

    selector.reset()

    assert not selector.is_fitted
    assert selector._best_model is None
    assert selector._model_scores == {}
    assert selector._ranked_models == []


def test_model_names_property(sample_predictions_2_models, sample_targets_2_models):
    """Test model_names property returns fitted models."""
    selector = BestModelSelector()
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    model_names = selector.model_names
    assert set(model_names) == {"model_a", "model_b"}


def test_repr(sample_predictions_2_models, sample_targets_2_models):
    """Test string representation."""
    selector = BestModelSelector()
    selector.fit(sample_predictions_2_models, sample_targets_2_models)

    repr_str = repr(selector)
    assert "BestModelSelector" in repr_str
    assert "fitted" in repr_str
    assert "n_models=2" in repr_str
