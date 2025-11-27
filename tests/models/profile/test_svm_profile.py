"""Tests for SVMProfileModel class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVR

from src.models.config.svm_config import SVMProfileConfig
from src.models.profile.svm_profile import SVMProfileModel


@pytest.fixture
def svm_config():
    """Create SVMProfileConfig for testing."""
    return SVMProfileConfig(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        epsilon=0.1,
        optimize_hyperparams=False,  # Disable for faster tests
        min_samples_per_period=10,  # Lower threshold for tests
        n_jobs=1,
        random_state=42,
    )


@pytest.fixture
def sample_data():
    """Create sample semi-hourly data with profile ratios."""
    np.random.seed(42)

    # Create 30 days of semi-hourly data (48 periods per day)
    n_days = 30
    n_samples = n_days * 48

    # Create timestamps
    timestamps = pd.date_range(
        "2024-01-01",
        periods=n_samples,
        freq="30min",
    )

    # Create profile ratios with realistic patterns
    # Nighttime (periods 0-14): lower ratios (0.6-0.9)
    # Daytime (periods 15-40): higher ratios (1.0-1.4)
    # Evening (periods 41-47): medium ratios (0.9-1.1)
    profile_ratios = []

    for ts in timestamps:
        period = ts.hour * 2 + (ts.minute // 30)

        if period < 15:  # Nighttime
            base = 0.75
            noise = np.random.normal(0, 0.05)
        elif period < 41:  # Daytime
            base = 1.2
            noise = np.random.normal(0, 0.1)
        else:  # Evening
            base = 1.0
            noise = np.random.normal(0, 0.05)

        ratio = max(0.5, min(2.0, base + noise))
        profile_ratios.append(ratio)

    # Create DataFrame
    df = pd.DataFrame(index=timestamps)
    df["dayofweek"] = timestamps.dayofweek
    df["day"] = timestamps.day
    df["month"] = timestamps.month
    df["is_weekend"] = (timestamps.dayofweek >= 5).astype(int)

    y = pd.Series(profile_ratios, index=timestamps, name="profile_ratio")

    return df, y


@pytest.fixture
def trained_model(svm_config, sample_data):
    """Create and train a model for testing."""
    X, y = sample_data
    model = SVMProfileModel()
    config = {"svm_config": svm_config}
    model.fit(X, y, config)
    return model


def test_svm_profile_initialization():
    """Test SVMProfileModel initialization."""
    model = SVMProfileModel()

    assert model.name == "svm_profile"
    assert model.version == "1.0.0"
    assert model.supported_horizons == [0]
    assert not model.is_fitted()


def test_svm_config_validation():
    """Test SVMProfileConfig validation."""
    # Valid config
    config = SVMProfileConfig(kernel="rbf", C=1.0)
    assert config.kernel == "rbf"
    assert config.C == 1.0

    # Invalid kernel
    with pytest.raises(ValueError, match="not supported"):
        SVMProfileConfig(kernel="invalid")

    # Invalid C (must be positive)
    with pytest.raises(ValueError):
        SVMProfileConfig(C=-1.0)

    # Invalid gamma string
    with pytest.raises(ValueError, match="not supported"):
        SVMProfileConfig(gamma="invalid")

    # Valid gamma as float
    config = SVMProfileConfig(gamma=0.01)
    assert config.gamma == 0.01

    # Invalid gamma (negative float)
    with pytest.raises(ValueError, match="must be positive"):
        SVMProfileConfig(gamma=-0.1)


def test_svm_config_param_grid():
    """Test SVMProfileConfig parameter grid generation."""
    config = SVMProfileConfig()

    param_grid = config.get_param_grid()

    assert "C" in param_grid
    assert "gamma" in param_grid
    assert "epsilon" in param_grid
    assert isinstance(param_grid["C"], list)


def test_svm_config_custom_param_grid():
    """Test SVMProfileConfig with custom parameter grid."""
    custom_grid = {
        "C": [1.0, 10.0],
        "gamma": ["scale"],
        "epsilon": [0.1],
    }

    config = SVMProfileConfig(param_grid=custom_grid)
    param_grid = config.get_param_grid()

    assert param_grid == custom_grid


def test_svm_config_to_svr_params():
    """Test conversion to SVR parameters."""
    config = SVMProfileConfig(kernel="rbf", C=10.0, gamma="auto", epsilon=0.2)

    params = config.to_svr_params()

    assert params["kernel"] == "rbf"
    assert params["C"] == 10.0
    assert params["gamma"] == "auto"
    assert params["epsilon"] == 0.2


def test_fit_basic(svm_config, sample_data):
    """Test basic model training."""
    X, y = sample_data
    model = SVMProfileModel()

    config = {"svm_config": svm_config}
    model.fit(X, y, config)

    assert model.is_fitted()
    assert len(model._models) > 0  # At least some periods trained
    assert len(model._scalers) == len(model._models)
    assert len(model._profile_statistics) == len(model._models)


def test_fit_with_dataframe_target(svm_config, sample_data):
    """Test training with DataFrame target (single column)."""
    X, y = sample_data
    model = SVMProfileModel()

    # Convert y to DataFrame
    y_df = y.to_frame(name="target")

    config = {"svm_config": svm_config}
    model.fit(X, y_df, config)

    assert model.is_fitted()


def test_fit_with_multi_column_target_fails(svm_config, sample_data):
    """Test that multi-column target raises error."""
    X, y = sample_data
    model = SVMProfileModel()

    # Create multi-column target
    y_df = pd.DataFrame({"col1": y, "col2": y})

    config = {"svm_config": svm_config}

    with pytest.raises(ValueError, match="single column"):
        model.fit(X, y_df, config)


def test_fit_length_mismatch(svm_config, sample_data):
    """Test that X and y length mismatch raises error."""
    X, y = sample_data
    model = SVMProfileModel()

    # Truncate y
    y_short = y.iloc[:100]

    config = {"svm_config": svm_config}

    with pytest.raises(ValueError, match="length mismatch"):
        model.fit(X, y_short, config)


def test_fit_without_datetime_index(svm_config):
    """Test that non-DatetimeIndex raises error."""
    model = SVMProfileModel()

    # Create data with integer index
    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.Series([1.0, 1.0, 1.0])

    config = {"svm_config": svm_config}

    with pytest.raises(ValueError, match="DatetimeIndex"):
        model.fit(X, y, config)


def test_fit_insufficient_samples(svm_config):
    """Test handling of insufficient samples per period."""
    # Create data with only 5 samples (below min_samples_per_period)
    timestamps = pd.date_range("2024-01-01", periods=5, freq="30min")
    X = pd.DataFrame({"feature": range(5)}, index=timestamps)
    y = pd.Series([1.0] * 5, index=timestamps)

    model = SVMProfileModel()
    config = {"svm_config": svm_config}

    # Should raise error because no models can be trained
    with pytest.raises(RuntimeError, match="No models were successfully trained"):
        model.fit(X, y, config)


def test_prepare_profile_data(svm_config, sample_data):
    """Test _prepare_profile_data method."""
    X, y = sample_data
    model = SVMProfileModel()
    model._svm_config = svm_config

    period_data = model._prepare_profile_data(X, y)

    # Should have data for 48 periods
    assert len(period_data) == 48

    # Each period should have X and y
    for period, data in period_data.items():
        assert "X" in data
        assert "y" in data
        assert len(data["X"]) == len(data["y"])
        assert len(data["X"]) == 30  # 30 days

    # Check statistics are computed
    assert len(model._profile_statistics) == 48
    for period, stats in model._profile_statistics.items():
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "q25" in stats
        assert "q75" in stats
        assert "n_samples" in stats


def test_prepare_period_features():
    """Test _prepare_period_features method."""
    model = SVMProfileModel()

    timestamps = pd.date_range("2024-01-01", periods=100, freq="30min")
    features = model._prepare_period_features(timestamps)

    assert len(features) == 100
    assert "dayofweek" in features.columns
    assert "day" in features.columns
    assert "month" in features.columns
    assert "quarter" in features.columns
    assert "is_weekend" in features.columns
    assert "month_sin" in features.columns
    assert "month_cos" in features.columns
    assert "dow_sin" in features.columns
    assert "dow_cos" in features.columns

    # Check value ranges
    assert (features["dayofweek"] >= 0).all()
    assert (features["dayofweek"] <= 6).all()
    assert (features["month"] >= 1).all()
    assert (features["month"] <= 12).all()
    assert features["is_weekend"].isin([0, 1]).all()


def test_extract_periods():
    """Test _extract_periods method."""
    model = SVMProfileModel()

    # Test various timestamps
    timestamps = pd.DatetimeIndex([
        "2024-01-01 00:00:00",  # Period 0
        "2024-01-01 00:30:00",  # Period 1
        "2024-01-01 01:00:00",  # Period 2
        "2024-01-01 12:00:00",  # Period 24
        "2024-01-01 23:30:00",  # Period 47
    ])

    periods = model._extract_periods(timestamps)

    assert periods[0] == 0
    assert periods[1] == 1
    assert periods[2] == 2
    assert periods[3] == 24
    assert periods[4] == 47


def test_single_period_training(svm_config, sample_data):
    """Test training a single period model."""
    X, y = sample_data
    model = SVMProfileModel()
    model._svm_config = svm_config

    # Prepare data for one period
    period_data = model._prepare_profile_data(X, y)

    # Train period 24 (noon)
    period = 24
    model._train_period_model(period, period_data[period])

    # Check model and scaler are created
    assert period in model._models
    assert period in model._scalers
    assert isinstance(model._models[period], SVR)


def test_predict_basic(trained_model, sample_data):
    """Test basic prediction."""
    X, y = sample_data

    # Use subset for prediction
    X_test = X.iloc[:48]  # First day

    predictions = trained_model.predict(X_test, horizons=[0])

    # Check output shape
    assert predictions.shape == (48, 1)
    assert "h0" in predictions.columns

    # Check predictions are reasonable
    assert (predictions["h0"] > 0).all()
    assert (predictions["h0"] < 5.0).all()


def test_predict_without_features(svm_config):
    """Test prediction with empty features (uses temporal features)."""
    # Train model WITHOUT features so it uses temporal features
    np.random.seed(42)
    n_days = 30
    n_samples = n_days * 48
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="30min")

    # Training data WITHOUT features (empty DataFrame)
    X_train = pd.DataFrame(index=timestamps)
    y_train = pd.Series(
        np.random.uniform(0.8, 1.2, n_samples),
        index=timestamps,
        name="profile_ratio"
    )

    model = SVMProfileModel()
    model.fit(X_train, y_train, config={"svm_config": svm_config})

    # Create test data without features
    test_timestamps = pd.date_range("2024-02-01", periods=48, freq="30min")
    X_test = pd.DataFrame(index=test_timestamps)

    predictions = model.predict(X_test, horizons=[0])

    assert predictions.shape == (48, 1)
    assert "h0" in predictions.columns


def test_predict_invalid_horizons(trained_model, sample_data):
    """Test prediction with invalid horizons."""
    X, _ = sample_data
    X_test = X.iloc[:48]

    # Invalid horizon
    with pytest.raises(ValueError, match="Invalid horizons"):
        trained_model.predict(X_test, horizons=[1])

    with pytest.raises(ValueError, match="Invalid horizons"):
        trained_model.predict(X_test, horizons=[0, 1])


def test_predict_not_fitted():
    """Test that prediction fails if model not fitted."""
    model = SVMProfileModel()

    timestamps = pd.date_range("2024-01-01", periods=10, freq="30min")
    X = pd.DataFrame(index=timestamps)

    with pytest.raises(ValueError, match="not been fitted"):
        model.predict(X, horizons=[0])


def test_predict_without_datetime_index(trained_model):
    """Test that prediction requires DatetimeIndex."""
    X = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(ValueError, match="DatetimeIndex"):
        trained_model.predict(X)


def test_hyperparameter_optimization():
    """Test hyperparameter optimization with GridSearchCV."""
    np.random.seed(42)

    # Create larger dataset to have enough samples per period
    timestamps = pd.date_range("2024-01-01", periods=1440, freq="30min")  # 30 days
    X = pd.DataFrame({"feature": range(1440)}, index=timestamps)
    y = pd.Series(np.random.uniform(0.8, 1.2, 1440), index=timestamps)

    # Config with optimization enabled but small grid
    config = SVMProfileConfig(
        optimize_hyperparams=True,
        param_grid={
            "C": [1.0, 10.0],
            "gamma": ["scale"],
            "epsilon": [0.1],
        },
        cv_folds=2,  # Small number of folds
        min_samples_per_period=10,  # Minimum valid threshold
        n_jobs=1,
    )

    model = SVMProfileModel()

    # Mock GridSearchCV to avoid slow actual optimization in tests
    with patch("src.models.profile.svm_profile.GridSearchCV") as mock_grid:
        # Create mock that returns a fitted SVR
        mock_estimator = SVR(kernel="rbf", C=1.0)
        mock_estimator.fit(X.iloc[:10].values, y.iloc[:10].values)

        mock_grid.return_value.fit = MagicMock()
        mock_grid.return_value.best_estimator_ = mock_estimator
        mock_grid.return_value.best_params_ = {"C": 1.0, "gamma": "scale"}
        mock_grid.return_value.best_score_ = -0.01

        model.fit(X, y, config={"svm_config": config})

        # Verify GridSearchCV was called
        assert mock_grid.called


def test_profile_bounds_validation(trained_model):
    """Test _apply_profile_bounds method."""
    # Test bounding for a period that exists
    period = 24

    # Test that bounded values stay within reasonable range
    bounded = trained_model._apply_profile_bounds(1.0, period)
    # Value should be within config bounds (may be clipped to period-specific bounds)
    assert trained_model._svm_config.min_profile_ratio <= bounded <= trained_model._svm_config.max_profile_ratio

    # Test value below lower bound
    bounded = trained_model._apply_profile_bounds(0.01, period)
    assert bounded >= trained_model._svm_config.min_profile_ratio

    # Test value above upper bound
    bounded = trained_model._apply_profile_bounds(10.0, period)
    assert bounded <= trained_model._svm_config.max_profile_ratio

    # Test for non-existent period (should use config bounds)
    bounded = trained_model._apply_profile_bounds(1.0, 999)
    assert trained_model._svm_config.min_profile_ratio <= bounded <= trained_model._svm_config.max_profile_ratio


def test_missing_period_handling(trained_model, sample_data):
    """Test prediction when some periods weren't trained."""
    X, _ = sample_data

    # Manually remove a period from models
    if 20 in trained_model._models:
        del trained_model._models[20]

    # Create test data for period 20 with the same features as training
    timestamps = pd.DatetimeIndex(["2024-02-01 10:00:00"])  # Period 20
    X_test = pd.DataFrame({
        "dayofweek": [timestamps[0].dayofweek],
        "day": [timestamps[0].day],
        "month": [timestamps[0].month],
        "is_weekend": [int(timestamps[0].dayofweek >= 5)],
    }, index=timestamps)

    # Should still work (uses global mean)
    predictions = trained_model.predict(X_test, horizons=[0])
    assert len(predictions) == 1
    assert predictions["h0"].iloc[0] > 0


def test_get_feature_importance(trained_model):
    """Test get_feature_importance method."""
    importance = trained_model.get_feature_importance()

    # SVM doesn't have feature importance
    assert isinstance(importance, dict)
    assert len(importance) == 0


def test_get_diagnostics(trained_model):
    """Test get_diagnostics method."""
    diagnostics = trained_model.get_diagnostics()

    assert "model_name" in diagnostics
    assert diagnostics["model_name"] == "svm_profile"
    assert "model_version" in diagnostics
    assert "config" in diagnostics
    assert "n_periods_trained" in diagnostics
    assert "trained_periods" in diagnostics
    assert "profile_statistics" in diagnostics
    assert "training_metadata" in diagnostics

    # Check statistics structure
    assert isinstance(diagnostics["profile_statistics"], dict)
    for period, stats in diagnostics["profile_statistics"].items():
        assert "mean" in stats
        assert "std" in stats
        assert "n_samples" in stats


def test_model_save_load(trained_model, tmp_path, sample_data):
    """Test model save and load."""
    # Save model
    model_path = tmp_path / "svm_profile_model.pkl"
    trained_model.save(model_path)

    assert model_path.exists()

    # Load model
    loaded_model = SVMProfileModel.load(model_path)

    assert loaded_model.is_fitted()
    assert loaded_model.name == trained_model.name
    assert loaded_model.version == trained_model.version
    assert len(loaded_model._models) == len(trained_model._models)
    assert len(loaded_model._scalers) == len(trained_model._scalers)
    assert loaded_model._svm_config.kernel == trained_model._svm_config.kernel

    # Test prediction works with loaded model
    # Use the same feature columns as training data
    X_train, _ = sample_data
    X_test = X_train.iloc[:48]  # Use first day from training data

    predictions = loaded_model.predict(X_test, horizons=[0])
    assert len(predictions) == 48


def test_model_save_not_fitted():
    """Test that saving unfitted model raises error."""
    model = SVMProfileModel()

    with pytest.raises(ValueError, match="not been fitted"):
        model.save("model.pkl")


def test_model_load_nonexistent_file():
    """Test loading from non-existent file."""
    with pytest.raises(FileNotFoundError):
        SVMProfileModel.load("nonexistent.pkl")


def test_fit_with_invalid_config_type(sample_data):
    """Test fit with invalid config type."""
    X, y = sample_data
    model = SVMProfileModel()

    with pytest.raises(ValueError, match="Invalid svm_config type"):
        model.fit(X, y, config={"svm_config": "invalid"})


def test_fit_with_dict_config(sample_data):
    """Test fit with config as dictionary."""
    X, y = sample_data
    model = SVMProfileModel()

    config = {
        "svm_config": {
            "kernel": "rbf",
            "C": 1.0,
            "optimize_hyperparams": False,
            "min_samples_per_period": 10,
        }
    }

    model.fit(X, y, config)
    assert model.is_fitted()
    assert model._svm_config.kernel == "rbf"


def test_all_48_periods_training(svm_config):
    """Test that all 48 periods can be trained with sufficient data."""
    np.random.seed(42)

    # Create 100 days of data to ensure all periods have enough samples
    n_days = 100
    n_samples = n_days * 48

    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="30min")
    X = pd.DataFrame({"feature": range(n_samples)}, index=timestamps)
    y = pd.Series(np.random.uniform(0.8, 1.2, n_samples), index=timestamps)

    model = SVMProfileModel()
    config = {"svm_config": svm_config}
    model.fit(X, y, config)

    # Should have trained all 48 periods
    assert len(model._models) == 48
    assert set(model._models.keys()) == set(range(48))


def test_profile_statistics_computation(svm_config, sample_data):
    """Test that profile statistics are computed correctly."""
    X, y = sample_data
    model = SVMProfileModel()
    config = {"svm_config": svm_config}

    model.fit(X, y, config)

    # Check statistics for a specific period
    for period in model._profile_statistics:
        stats = model._profile_statistics[period]

        assert stats["mean"] > 0
        assert stats["std"] >= 0
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["q25"] <= stats["mean"]
        assert stats["q75"] >= stats["mean"]
        assert stats["n_samples"] > 0


def test_feature_names_storage(svm_config, sample_data):
    """Test that feature names are stored correctly."""
    X, y = sample_data
    model = SVMProfileModel()
    config = {"svm_config": svm_config}

    model.fit(X, y, config)

    feature_names = model.get_feature_names()
    assert len(feature_names) > 0
    assert "dayofweek" in feature_names or list(X.columns)[0] in feature_names


def test_model_repr(trained_model):
    """Test model string representation."""
    repr_str = repr(trained_model)
    assert "SVMProfileModel" in repr_str
    assert "svm_profile" in repr_str
    assert "fitted=True" in repr_str

    str_str = str(trained_model)
    assert "svm_profile" in str_str
