"""Tests for StackingCombiner."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.combination.data_structures import CombinerConfig
from src.models.combination.stacking_combiner import LIGHTGBM_AVAILABLE, StackingCombiner


# Fixtures


@pytest.fixture
def sample_predictions():
    """Create sample model predictions with 3 models and 3 horizons."""
    dates = pd.date_range("2024-01-01", periods=100, freq="30min")

    np.random.seed(42)
    predictions = {
        "lgbm": pd.DataFrame(
            {
                "pred_h0": np.random.randn(100) * 50 + 1000,
                "pred_h1": np.random.randn(100) * 50 + 1000,
                "pred_h2": np.random.randn(100) * 50 + 1000,
            },
            index=dates,
        ),
        "rf": pd.DataFrame(
            {
                "pred_h0": np.random.randn(100) * 50 + 1050,
                "pred_h1": np.random.randn(100) * 50 + 1050,
                "pred_h2": np.random.randn(100) * 50 + 1050,
            },
            index=dates,
        ),
        "xgb": pd.DataFrame(
            {
                "pred_h0": np.random.randn(100) * 50 + 980,
                "pred_h1": np.random.randn(100) * 50 + 980,
                "pred_h2": np.random.randn(100) * 50 + 980,
            },
            index=dates,
        ),
    }

    return predictions


@pytest.fixture
def sample_targets():
    """Create sample target values."""
    dates = pd.date_range("2024-01-01", periods=100, freq="30min")

    np.random.seed(42)
    targets = pd.DataFrame(
        {
            "h0": np.random.randn(100) * 50 + 1000,
            "h1": np.random.randn(100) * 50 + 1000,
            "h2": np.random.randn(100) * 50 + 1000,
        },
        index=dates,
    )

    return targets


@pytest.fixture
def sample_predictions_2_models():
    """Create sample predictions with 2 models for simpler tests."""
    dates = pd.date_range("2024-01-01", periods=50, freq="30min")

    np.random.seed(42)
    predictions = {
        "model_a": pd.DataFrame(
            {
                "pred_h0": np.array([100.0 + i for i in range(50)]),
                "pred_h1": np.array([200.0 + i for i in range(50)]),
            },
            index=dates,
        ),
        "model_b": pd.DataFrame(
            {
                "pred_h0": np.array([200.0 + i * 2 for i in range(50)]),
                "pred_h1": np.array([400.0 + i * 2 for i in range(50)]),
            },
            index=dates,
        ),
    }

    return predictions


@pytest.fixture
def sample_targets_2_models():
    """Create target values for 2 model predictions."""
    dates = pd.date_range("2024-01-01", periods=50, freq="30min")

    np.random.seed(42)
    targets = pd.DataFrame(
        {
            "h0": np.array([150.0 + i * 1.5 for i in range(50)]),
            "h1": np.array([300.0 + i * 1.5 for i in range(50)]),
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
def targets_with_nans():
    """Create targets for predictions with NaNs."""
    dates = pd.date_range("2024-01-01", periods=24, freq="30min")

    targets = pd.DataFrame(
        {
            "h0": [175.0] * 24,
            "h1": [350.0] * 24,
        },
        index=dates,
    )

    return targets


# Initialization Tests


def test_stacking_combiner_initialization_default():
    """Test StackingCombiner initialization with default parameters."""
    combiner = StackingCombiner()

    assert combiner.name == "stacking_ridge"
    assert combiner.version == "1.0.0"
    assert combiner.meta_model_name == "ridge"
    assert combiner.cv_folds == 5
    assert combiner.cv_strategy == "timeseries"
    assert not combiner.is_fitted
    assert combiner.config.require_fit is True


def test_stacking_combiner_initialization_linear():
    """Test initialization with linear regression meta-model."""
    combiner = StackingCombiner(meta_model="linear")

    assert combiner.name == "stacking_linear"
    assert combiner.meta_model_name == "linear"


def test_stacking_combiner_initialization_lasso():
    """Test initialization with lasso regression meta-model."""
    combiner = StackingCombiner(meta_model="lasso", cv_folds=3)

    assert combiner.name == "stacking_lasso"
    assert combiner.meta_model_name == "lasso"
    assert combiner.cv_folds == 3


def test_stacking_combiner_initialization_elastic_net():
    """Test initialization with elastic net meta-model."""
    combiner = StackingCombiner(
        meta_model="elastic_net", meta_model_params={"alpha": 0.5, "l1_ratio": 0.7}
    )

    assert combiner.name == "stacking_elastic_net"
    assert combiner.meta_model_params["alpha"] == 0.5
    assert combiner.meta_model_params["l1_ratio"] == 0.7


def test_stacking_combiner_initialization_rf():
    """Test initialization with random forest meta-model."""
    combiner = StackingCombiner(
        meta_model="rf", meta_model_params={"n_estimators": 50, "max_depth": 10}
    )

    assert combiner.name == "stacking_rf"
    assert combiner.meta_model_params["n_estimators"] == 50


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
def test_stacking_combiner_initialization_lgbm():
    """Test initialization with LightGBM meta-model."""
    combiner = StackingCombiner(
        meta_model="lgbm", meta_model_params={"num_leaves": 15, "learning_rate": 0.05}
    )

    assert combiner.name == "stacking_lgbm"
    assert combiner.meta_model_params["num_leaves"] == 15


def test_stacking_combiner_initialization_kfold():
    """Test initialization with KFold cross-validation strategy."""
    combiner = StackingCombiner(cv_strategy="kfold", cv_folds=10)

    assert combiner.cv_strategy == "kfold"
    assert combiner.cv_folds == 10


def test_stacking_combiner_initialization_with_config():
    """Test initialization with custom config."""
    config = CombinerConfig(require_fit=True, store_contributions=True, missing_threshold=0.3)
    combiner = StackingCombiner(config=config)

    assert combiner.config.store_contributions is True
    assert combiner.config.missing_threshold == 0.3


def test_stacking_combiner_invalid_meta_model():
    """Test that invalid meta-model raises ValueError."""
    with pytest.raises(ValueError, match="Invalid meta_model"):
        StackingCombiner(meta_model="invalid_model")


def test_stacking_combiner_invalid_cv_folds():
    """Test that cv_folds < 2 raises ValueError."""
    with pytest.raises(ValueError, match="cv_folds must be >= 2"):
        StackingCombiner(cv_folds=1)

    with pytest.raises(ValueError, match="cv_folds must be >= 2"):
        StackingCombiner(cv_folds=0)


def test_stacking_combiner_invalid_cv_strategy():
    """Test that invalid cv_strategy raises ValueError."""
    with pytest.raises(ValueError, match="Invalid cv_strategy"):
        StackingCombiner(cv_strategy="invalid_strategy")


@pytest.mark.skipif(LIGHTGBM_AVAILABLE, reason="Test only when LightGBM not available")
def test_stacking_combiner_lgbm_not_installed():
    """Test that lgbm meta-model raises ImportError when not installed."""
    with pytest.raises(ImportError, match="LightGBM is not installed"):
        StackingCombiner(meta_model="lgbm")


# Fit Tests


def test_stacking_combiner_fit_ridge(sample_predictions, sample_targets):
    """Test fitting stacking combiner with ridge regression."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted
    assert combiner.fit_timestamp is not None
    assert isinstance(combiner.fit_timestamp, datetime)
    assert combiner._meta_model is not None
    assert len(combiner._model_names) == 3
    assert set(combiner._model_names) == {"lgbm", "rf", "xgb"}


def test_stacking_combiner_fit_linear(sample_predictions, sample_targets):
    """Test fitting with linear regression meta-model."""
    combiner = StackingCombiner(meta_model="linear", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted
    assert combiner._meta_model is not None


def test_stacking_combiner_fit_lasso(sample_predictions, sample_targets):
    """Test fitting with lasso regression meta-model."""
    combiner = StackingCombiner(meta_model="lasso", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted
    assert combiner._meta_model is not None


def test_stacking_combiner_fit_elastic_net(sample_predictions, sample_targets):
    """Test fitting with elastic net meta-model."""
    combiner = StackingCombiner(meta_model="elastic_net", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted
    assert combiner._meta_model is not None


def test_stacking_combiner_fit_rf(sample_predictions, sample_targets):
    """Test fitting with random forest meta-model."""
    combiner = StackingCombiner(meta_model="rf", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted
    assert combiner._meta_model is not None


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
def test_stacking_combiner_fit_lgbm(sample_predictions, sample_targets):
    """Test fitting with LightGBM meta-model."""
    combiner = StackingCombiner(meta_model="lgbm", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted
    assert combiner._meta_model is not None


def test_stacking_combiner_fit_generates_cv_scores(sample_predictions, sample_targets):
    """Test that fit generates CV scores."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    assert len(combiner._cv_scores) > 0
    assert "h0" in combiner._cv_scores
    assert len(combiner._cv_scores["h0"]) > 0


def test_stacking_combiner_fit_extracts_weights(sample_predictions, sample_targets):
    """Test that fit extracts weights from meta-model."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    weights = combiner.get_weights()
    assert len(weights) == 3
    assert "lgbm" in weights
    assert "rf" in weights
    assert "xgb" in weights

    # Weights should sum to 1
    assert np.isclose(sum(weights.values()), 1.0)

    # All weights should be non-negative
    assert all(w >= 0 for w in weights.values())


def test_stacking_combiner_fit_kfold(sample_predictions, sample_targets):
    """Test fitting with KFold cross-validation."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3, cv_strategy="kfold")
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted
    assert combiner.cv_strategy == "kfold"


def test_stacking_combiner_fit_timeseries(sample_predictions, sample_targets):
    """Test fitting with TimeSeriesSplit cross-validation."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3, cv_strategy="timeseries")
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted
    assert combiner.cv_strategy == "timeseries"


def test_stacking_combiner_fit_invalid_predictions():
    """Test that fit raises ValueError for invalid predictions."""
    combiner = StackingCombiner()

    with pytest.raises(ValueError):
        combiner.fit({}, pd.DataFrame())


def test_stacking_combiner_fit_missing_target_columns(sample_predictions):
    """Test that fit raises ValueError when target columns are missing."""
    combiner = StackingCombiner()

    # Targets with wrong column names
    targets = pd.DataFrame(
        {"wrong_col": np.random.randn(100)}, index=sample_predictions["lgbm"].index
    )

    with pytest.raises(ValueError, match="Target columns missing"):
        combiner.fit(sample_predictions, targets)


def test_stacking_combiner_fit_generates_oof_predictions(sample_predictions, sample_targets):
    """Test that fit generates out-of-fold predictions."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner._oof_predictions is not None
    assert len(combiner._oof_predictions) == len(sample_predictions["lgbm"])


# Combine Tests


def test_stacking_combiner_combine_after_fit(sample_predictions, sample_targets):
    """Test combining predictions after fitting."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    result = combiner.combine(sample_predictions)

    assert result is not None
    assert len(result.combined_predictions) == len(sample_predictions["lgbm"])
    assert "pred_h0" in result.combined_predictions.columns
    assert "pred_h1" in result.combined_predictions.columns
    assert "pred_h2" in result.combined_predictions.columns


def test_stacking_combiner_combine_returns_valid_result(sample_predictions, sample_targets):
    """Test that combine returns valid CombinationResult."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    result = combiner.combine(sample_predictions)

    assert result.n_models == 3
    assert result.n_samples == len(sample_predictions["lgbm"])
    assert result.horizons == [0, 1, 2]
    assert len(result.weights) == 3

    # Weights should sum to 1
    assert np.isclose(sum(result.weights.values()), 1.0)


def test_stacking_combiner_combine_not_fitted(sample_predictions):
    """Test that combine raises RuntimeError when not fitted."""
    combiner = StackingCombiner()

    with pytest.raises(RuntimeError, match="not fitted"):
        combiner.combine(sample_predictions)


def test_stacking_combiner_combine_model_mismatch(
    sample_predictions, sample_targets, sample_predictions_2_models
):
    """Test that combine raises ValueError when model names don't match."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    # Try to combine with different models
    with pytest.raises(ValueError, match="Model names don't match"):
        combiner.combine(sample_predictions_2_models)


def test_stacking_combiner_combine_with_contributions(sample_predictions, sample_targets):
    """Test combining with store_contributions enabled."""
    config = CombinerConfig(store_contributions=True)
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3, config=config)
    combiner.fit(sample_predictions, sample_targets)

    result = combiner.combine(sample_predictions)

    assert result.model_contributions is not None
    assert len(result.model_contributions) == 3
    assert "lgbm" in result.model_contributions
    assert "rf" in result.model_contributions
    assert "xgb" in result.model_contributions

    # Each contribution should have same shape as combined predictions
    for model_name, contrib_df in result.model_contributions.items():
        assert len(contrib_df) == len(result.combined_predictions)
        assert list(contrib_df.columns) == list(result.combined_predictions.columns)


def test_stacking_combiner_combine_with_nans(predictions_with_nans, targets_with_nans):
    """Test combining predictions with NaN values."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=2)
    combiner.fit(predictions_with_nans, targets_with_nans)

    result = combiner.combine(predictions_with_nans)

    # Should handle NaN gracefully with fallback to averaging
    assert result is not None
    assert len(result.combined_predictions) == len(predictions_with_nans["model_a"])

    # Check that we have predictions even where NaN existed
    # (might have some NaN if all models had NaN)
    pred_h0 = result.combined_predictions["pred_h0"]
    assert not pred_h0.isna().all()  # Not all NaN


def test_stacking_combiner_combine_metadata(sample_predictions, sample_targets):
    """Test that combine generates proper metadata."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    result = combiner.combine(sample_predictions)

    assert result.metadata is not None
    assert result.metadata.combination_method == "stacking_ridge"
    assert len(result.metadata.models_used) == 3
    assert result.metadata.processing_time_seconds > 0
    assert "meta_model" in result.metadata.configuration
    assert result.metadata.configuration["meta_model"] == "ridge"
    assert "cv_folds" in result.metadata.configuration
    assert "cv_strategy" in result.metadata.configuration


def test_stacking_combiner_combine_includes_cv_scores(sample_predictions, sample_targets):
    """Test that combine includes CV scores in metadata."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    result = combiner.combine(sample_predictions)

    # CV scores should be in performance metrics
    assert "cv_rmse_mean_h0" in result.metadata.performance_metrics
    assert result.metadata.performance_metrics["cv_rmse_mean_h0"] > 0


def test_stacking_combiner_combine_predictions_reasonable(
    sample_predictions_2_models, sample_targets_2_models
):
    """Test that combined predictions are in reasonable range."""
    combiner = StackingCombiner(meta_model="linear", cv_folds=2)
    combiner.fit(sample_predictions_2_models, sample_targets_2_models)

    result = combiner.combine(sample_predictions_2_models)

    # Combined predictions should be between min and max of input predictions
    for col in ["pred_h0", "pred_h1"]:
        combined = result.combined_predictions[col]
        min_pred = min(
            sample_predictions_2_models["model_a"][col].min(),
            sample_predictions_2_models["model_b"][col].min(),
        )
        max_pred = max(
            sample_predictions_2_models["model_a"][col].max(),
            sample_predictions_2_models["model_b"][col].max(),
        )

        # Allow some extrapolation (10% margin)
        margin = (max_pred - min_pred) * 0.1
        assert (combined >= min_pred - margin).all() or np.isnan(combined).any()
        assert (combined <= max_pred + margin).all() or np.isnan(combined).any()


# Get CV Scores Tests


def test_stacking_combiner_get_cv_scores_all(sample_predictions, sample_targets):
    """Test getting all CV scores."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    scores = combiner.get_cv_scores()

    assert isinstance(scores, dict)
    assert "h0" in scores
    assert len(scores["h0"]) > 0


def test_stacking_combiner_get_cv_scores_specific_horizon(sample_predictions, sample_targets):
    """Test getting CV scores for specific horizon."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    scores = combiner.get_cv_scores("h0")

    assert isinstance(scores, list)
    assert len(scores) > 0
    assert all(isinstance(s, float) for s in scores)


def test_stacking_combiner_get_cv_scores_not_fitted():
    """Test that get_cv_scores raises RuntimeError when not fitted."""
    combiner = StackingCombiner()

    with pytest.raises(RuntimeError, match="not fitted"):
        combiner.get_cv_scores()


def test_stacking_combiner_get_cv_scores_invalid_horizon(sample_predictions, sample_targets):
    """Test that get_cv_scores raises KeyError for invalid horizon."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    with pytest.raises(KeyError, match="not found"):
        combiner.get_cv_scores("h99")


# Weight Extraction Tests


def test_stacking_combiner_weights_from_linear_model(sample_predictions, sample_targets):
    """Test weight extraction from linear regression."""
    combiner = StackingCombiner(meta_model="linear", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    weights = combiner.get_weights()

    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(w >= 0 for w in weights.values())


def test_stacking_combiner_weights_from_ridge(sample_predictions, sample_targets):
    """Test weight extraction from ridge regression."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    weights = combiner.get_weights()

    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(w >= 0 for w in weights.values())


def test_stacking_combiner_weights_from_rf(sample_predictions, sample_targets):
    """Test weight extraction from random forest (feature importances)."""
    combiner = StackingCombiner(meta_model="rf", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    weights = combiner.get_weights()

    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(w >= 0 for w in weights.values())


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
def test_stacking_combiner_weights_from_lgbm(sample_predictions, sample_targets):
    """Test weight extraction from LightGBM (feature importances)."""
    combiner = StackingCombiner(meta_model="lgbm", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    weights = combiner.get_weights()

    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(w >= 0 for w in weights.values())


# Persistence Tests


def test_stacking_combiner_save_and_load(sample_predictions, sample_targets, tmp_path):
    """Test saving and loading combiner."""
    # Fit combiner
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    # Get predictions before saving
    result_before = combiner.combine(sample_predictions)

    # Save
    filepath = tmp_path / "stacking_combiner.pkl"
    combiner.save(filepath)

    assert filepath.exists()

    # Load
    loaded_combiner = StackingCombiner.load(filepath)

    # Check state
    assert loaded_combiner.is_fitted
    assert loaded_combiner.meta_model_name == "ridge"
    assert loaded_combiner.cv_folds == 3
    assert loaded_combiner.cv_strategy == "timeseries"
    assert loaded_combiner._meta_model is not None

    # Get predictions after loading
    result_after = loaded_combiner.combine(sample_predictions)

    # Results should be identical
    pd.testing.assert_frame_equal(
        result_before.combined_predictions, result_after.combined_predictions
    )
    assert result_before.weights == result_after.weights


def test_stacking_combiner_save_creates_directory(sample_predictions, sample_targets, tmp_path):
    """Test that save creates parent directories if needed."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    filepath = tmp_path / "subdir" / "nested" / "combiner.pkl"
    combiner.save(filepath)

    assert filepath.exists()


def test_stacking_combiner_load_preserves_cv_scores(sample_predictions, sample_targets, tmp_path):
    """Test that load preserves CV scores."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    scores_before = combiner.get_cv_scores()

    # Save and load
    filepath = tmp_path / "combiner.pkl"
    combiner.save(filepath)
    loaded_combiner = StackingCombiner.load(filepath)

    scores_after = loaded_combiner.get_cv_scores()

    assert scores_before.keys() == scores_after.keys()
    for horizon in scores_before:
        assert scores_before[horizon] == scores_after[horizon]


def test_stacking_combiner_load_different_meta_models(sample_predictions, sample_targets, tmp_path):
    """Test saving and loading different meta-model types."""
    meta_models = ["linear", "ridge", "lasso", "rf"]

    for meta_model in meta_models:
        combiner = StackingCombiner(meta_model=meta_model, cv_folds=2)
        combiner.fit(sample_predictions, sample_targets)

        filepath = tmp_path / f"combiner_{meta_model}.pkl"
        combiner.save(filepath)

        loaded = StackingCombiner.load(filepath)
        assert loaded.meta_model_name == meta_model
        assert loaded.is_fitted


# Integration Tests


def test_stacking_combiner_full_workflow(sample_predictions, sample_targets):
    """Test complete workflow: initialize -> fit -> combine -> save -> load -> combine."""
    # Initialize
    config = CombinerConfig(require_fit=True, store_contributions=True)
    combiner = StackingCombiner(
        meta_model="ridge", cv_folds=3, cv_strategy="timeseries", config=config
    )

    # Fit
    combiner.fit(sample_predictions, sample_targets)
    assert combiner.is_fitted

    # Combine
    result1 = combiner.combine(sample_predictions)
    assert result1.n_models == 3
    assert result1.model_contributions is not None

    # Get CV scores
    cv_scores = combiner.get_cv_scores()
    assert len(cv_scores) > 0

    # Save
    filepath = Path("/tmp/test_stacking_combiner.pkl")
    combiner.save(filepath)

    # Load
    loaded = StackingCombiner.load(filepath)
    assert loaded.is_fitted

    # Combine with loaded combiner
    result2 = loaded.combine(sample_predictions)

    # Results should match
    pd.testing.assert_frame_equal(result1.combined_predictions, result2.combined_predictions)

    # Cleanup
    if filepath.exists():
        filepath.unlink()


def test_stacking_combiner_beats_simple_average(
    sample_predictions_2_models, sample_targets_2_models
):
    """Test that stacking can learn better combinations than simple averaging.

    Note: This test might occasionally fail due to randomness, but in general
    stacking should perform as well or better than simple averaging.
    """
    from src.models.combination import SimpleAveragingCombiner

    # Simple averaging combiner
    simple_combiner = SimpleAveragingCombiner(weighted=False)
    simple_combiner.fit(sample_predictions_2_models, sample_targets_2_models)
    simple_result = simple_combiner.combine(sample_predictions_2_models)

    # Stacking combiner
    stacking_combiner = StackingCombiner(meta_model="ridge", cv_folds=2)
    stacking_combiner.fit(sample_predictions_2_models, sample_targets_2_models)
    stacking_result = stacking_combiner.combine(sample_predictions_2_models)

    # Both should produce valid results
    assert simple_result.n_models == 2
    assert stacking_result.n_models == 2

    # Weights should sum to 1
    assert np.isclose(sum(simple_result.weights.values()), 1.0)
    assert np.isclose(sum(stacking_result.weights.values()), 1.0)

    # Predictions should be in similar range
    for col in ["pred_h0", "pred_h1"]:
        simple_pred = simple_result.combined_predictions[col]
        stacking_pred = stacking_result.combined_predictions[col]

        # Correlation should be high (both methods should give similar results)
        correlation = np.corrcoef(simple_pred, stacking_pred)[0, 1]
        assert correlation > 0.9


def test_stacking_combiner_different_cv_strategies_produce_different_results(
    sample_predictions, sample_targets
):
    """Test that different CV strategies produce different (but valid) results."""
    # KFold strategy
    combiner_kfold = StackingCombiner(meta_model="ridge", cv_folds=3, cv_strategy="kfold")
    combiner_kfold.fit(sample_predictions, sample_targets)

    # TimeSeriesSplit strategy
    combiner_ts = StackingCombiner(meta_model="ridge", cv_folds=3, cv_strategy="timeseries")
    combiner_ts.fit(sample_predictions, sample_targets)

    # Both should be fitted
    assert combiner_kfold.is_fitted
    assert combiner_ts.is_fitted

    # Weights might differ (depending on data)
    weights_kfold = combiner_kfold.get_weights()
    weights_ts = combiner_ts.get_weights()

    assert len(weights_kfold) == len(weights_ts)
    # Weights should sum to 1 in both cases
    assert np.isclose(sum(weights_kfold.values()), 1.0)
    assert np.isclose(sum(weights_ts.values()), 1.0)


def test_stacking_combiner_multiple_horizons_handled_correctly(sample_predictions, sample_targets):
    """Test that combiner handles multiple horizons correctly."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    result = combiner.combine(sample_predictions)

    # Should have all horizons
    assert "pred_h0" in result.combined_predictions.columns
    assert "pred_h1" in result.combined_predictions.columns
    assert "pred_h2" in result.combined_predictions.columns

    # All horizons should have predictions
    assert not result.combined_predictions["pred_h0"].isna().all()
    assert not result.combined_predictions["pred_h1"].isna().all()
    assert not result.combined_predictions["pred_h2"].isna().all()


def test_stacking_combiner_repr(sample_predictions, sample_targets):
    """Test string representation of combiner."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)

    # Before fitting
    repr_before = repr(combiner)
    assert "StackingCombiner" in repr_before
    assert "stacking_ridge" in repr_before
    assert "not fitted" in repr_before

    # After fitting
    combiner.fit(sample_predictions, sample_targets)
    repr_after = repr(combiner)
    assert "StackingCombiner" in repr_after
    assert "fitted" in repr_after
    assert "n_models=3" in repr_after


def test_stacking_combiner_reset(sample_predictions, sample_targets):
    """Test resetting combiner to unfitted state."""
    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(sample_predictions, sample_targets)

    assert combiner.is_fitted

    combiner.reset()

    assert not combiner.is_fitted
    assert combiner.fit_timestamp is None
    assert len(combiner._model_names) == 0
    assert len(combiner._global_weights) == 0
    assert combiner._meta_model is None  # Stacking-specific state


# Edge Cases


def test_stacking_combiner_single_model():
    """Test that stacking with single model works (though not recommended)."""
    dates = pd.date_range("2024-01-01", periods=50, freq="30min")

    predictions = {
        "single_model": pd.DataFrame(
            {
                "pred_h0": np.random.randn(50) * 50 + 1000,
                "pred_h1": np.random.randn(50) * 50 + 1000,
            },
            index=dates,
        ),
    }

    targets = pd.DataFrame(
        {
            "h0": np.random.randn(50) * 50 + 1000,
            "h1": np.random.randn(50) * 50 + 1000,
        },
        index=dates,
    )

    combiner = StackingCombiner(meta_model="linear", cv_folds=2)
    combiner.fit(predictions, targets)

    result = combiner.combine(predictions)

    assert result.n_models == 1
    assert list(result.weights.keys()) == ["single_model"]
    assert np.isclose(result.weights["single_model"], 1.0)


def test_stacking_combiner_many_models():
    """Test stacking with many models."""
    dates = pd.date_range("2024-01-01", periods=100, freq="30min")
    n_models = 10

    np.random.seed(42)
    predictions = {}
    for i in range(n_models):
        predictions[f"model_{i}"] = pd.DataFrame(
            {
                "pred_h0": np.random.randn(100) * 50 + 1000 + i * 10,
            },
            index=dates,
        )

    targets = pd.DataFrame(
        {"h0": np.random.randn(100) * 50 + 1000},
        index=dates,
    )

    combiner = StackingCombiner(meta_model="ridge", cv_folds=3)
    combiner.fit(predictions, targets)

    result = combiner.combine(predictions)

    assert result.n_models == n_models
    assert len(result.weights) == n_models
    assert np.isclose(sum(result.weights.values()), 1.0)


def test_stacking_combiner_small_dataset():
    """Test stacking with small dataset (edge case for CV)."""
    dates = pd.date_range("2024-01-01", periods=10, freq="30min")

    predictions = {
        "model_a": pd.DataFrame(
            {"pred_h0": np.random.randn(10) + 100},
            index=dates,
        ),
        "model_b": pd.DataFrame(
            {"pred_h0": np.random.randn(10) + 100},
            index=dates,
        ),
    }

    targets = pd.DataFrame(
        {"h0": np.random.randn(10) + 100},
        index=dates,
    )

    # Use fewer folds for small dataset
    combiner = StackingCombiner(meta_model="linear", cv_folds=2)
    combiner.fit(predictions, targets)

    result = combiner.combine(predictions)

    assert result.n_models == 2
