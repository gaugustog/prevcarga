"""Tests for ARIMADemandMeanModel class."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.config.arima_config import ARIMAConfig
from src.models.demand_mean.arima_model import ARIMADemandMeanModel


class TestARIMADemandMeanModel:
    """Test suite for ARIMADemandMeanModel."""

    @pytest.fixture
    def synthetic_daily_data(self):
        """Create synthetic daily demand data with trend and seasonality."""
        np.random.seed(42)
        n_days = 60

        # Time index
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

        # Components: trend + weekly seasonality + noise
        trend = np.linspace(100, 120, n_days)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        noise = np.random.normal(0, 2, n_days)

        y = trend + seasonal + noise

        # Create DataFrame with features (even though ARIMA doesn't use them)
        X = pd.DataFrame(
            {
                "day_of_week": dates.dayofweek,
                "day_of_year": dates.dayofyear,
            },
            index=dates,
        )

        y_series = pd.Series(y, index=dates, name="demand")

        return X, y_series

    @pytest.fixture
    def synthetic_semihourly_data(self):
        """Create synthetic semi-hourly demand data for aggregation tests."""
        np.random.seed(42)
        n_days = 30
        n_periods_per_day = 48

        # Time index (semi-hourly)
        dates = pd.date_range("2024-01-01", periods=n_days * n_periods_per_day, freq="30min")

        # Daily pattern
        daily_mean = np.linspace(100, 110, n_days)
        daily_mean_expanded = np.repeat(daily_mean, n_periods_per_day)

        # Intraday pattern (higher during day, lower at night)
        intraday_pattern = np.tile(
            np.concatenate([
                np.linspace(0.7, 1.0, 24),  # Morning rise
                np.linspace(1.0, 0.7, 24),  # Evening fall
            ]),
            n_days,
        )

        y = daily_mean_expanded * intraday_pattern + np.random.normal(0, 1, len(dates))

        X = pd.DataFrame(
            {
                "period": (dates.hour * 2 + dates.minute // 30).values,
            },
            index=dates,
        )

        y_series = pd.Series(y, index=dates, name="demand")

        return X, y_series

    def test_arima_initialization(self):
        """Test ARIMA model initialization."""
        model = ARIMADemandMeanModel()

        assert model is not None
        assert model.name == "arima_demand_mean"
        assert model.version == "1.0.0"
        assert model.supported_horizons == list(range(9))
        assert not model.is_fitted()

    def test_arima_config_validation(self):
        """Test ARIMAConfig validation."""
        # Valid config
        config = ARIMAConfig(
            seasonal=True,
            season_length=7,
            max_p=3,
            max_d=1,
            max_q=3,
        )
        assert config.seasonal
        assert config.season_length == 7

        # Invalid prediction interval
        with pytest.raises(ValueError, match="must be between 1 and 99"):
            ARIMAConfig(prediction_intervals=[0, 100])

    def test_daily_aggregation(self, synthetic_semihourly_data):
        """Test conversion from semi-hourly to daily mean."""
        X, y = synthetic_semihourly_data

        model = ARIMADemandMeanModel()
        y_daily = model._convert_to_daily_mean(y)

        # Check daily aggregation
        assert len(y_daily) == 30  # 30 days
        assert isinstance(y_daily.index, pd.DatetimeIndex)

        # Check that daily mean is reasonable
        expected_first_day_mean = y[y.index.date == y.index[0].date()].mean()
        assert abs(y_daily.iloc[0] - expected_first_day_mean) < 0.01

    def test_arima_training_with_synthetic_data(self, synthetic_daily_data):
        """Test ARIMA training with synthetic daily data."""
        X, y = synthetic_daily_data

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(
            seasonal=True,
            season_length=7,
            max_p=2,
            max_d=1,
            max_q=2,
            max_P=1,
            max_D=1,
            max_Q=1,
            stepwise=True,
            min_training_days=14,
        )

        # Train model
        model.fit(X, y, config={"arima_config": config})

        assert model.is_fitted()
        assert model._training_series is not None
        assert len(model._training_series) == 60

    def test_arima_training_with_config_dict(self, synthetic_daily_data):
        """Test ARIMA training with config as dictionary."""
        X, y = synthetic_daily_data

        model = ARIMADemandMeanModel()
        config_dict = {
            "seasonal": True,
            "season_length": 7,
            "max_p": 2,
            "max_d": 1,
            "max_q": 2,
            "min_training_days": 14,
        }

        # Train model with dict config
        model.fit(X, y, config={"arima_config": config_dict})

        assert model.is_fitted()

    def test_multi_horizon_forecasting(self, synthetic_daily_data):
        """Test multi-horizon forecasting."""
        X_train, y_train = synthetic_daily_data

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(
            seasonal=False,  # Faster for testing
            max_p=2,
            max_d=1,
            max_q=2,
            stepwise=True,
            min_training_days=14,
        )

        # Train model
        model.fit(X_train, y_train, config={"arima_config": config})

        # Predict multiple horizons
        horizons = [0, 1, 2, 3]
        predictions = model.predict(X_train, horizons=horizons)

        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[1] == len(horizons)
        assert all(f"h{h}" in predictions.columns for h in horizons)

        # Check predictions are reasonable (not NaN or extreme)
        assert not predictions.isna().any().any()
        assert (predictions > 0).all().all()  # Demand should be positive

    def test_prediction_intervals(self, synthetic_daily_data):
        """Test prediction intervals generation."""
        X_train, y_train = synthetic_daily_data

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(
            seasonal=False,
            max_p=2,
            max_d=1,
            max_q=2,
            stepwise=True,
            prediction_intervals=[80, 95],
            min_training_days=14,
        )

        # Train model
        model.fit(X_train, y_train, config={"arima_config": config})

        # Get prediction intervals
        horizons = [0, 1, 2]
        intervals = model.get_prediction_intervals(X_train, horizons=horizons)

        assert isinstance(intervals, pd.DataFrame)

        # Check point forecasts exist
        assert all(f"h{h}" in intervals.columns for h in horizons)

        # Check intervals exist
        for h in horizons:
            assert f"h{h}_lo80" in intervals.columns
            assert f"h{h}_hi80" in intervals.columns
            assert f"h{h}_lo95" in intervals.columns
            assert f"h{h}_hi95" in intervals.columns

        # Check interval ordering: lo95 < lo80 < point < hi80 < hi95
        for h in horizons:
            point = intervals[f"h{h}"].iloc[0]
            lo80 = intervals[f"h{h}_lo80"].iloc[0]
            hi80 = intervals[f"h{h}_hi80"].iloc[0]
            lo95 = intervals[f"h{h}_lo95"].iloc[0]
            hi95 = intervals[f"h{h}_hi95"].iloc[0]

            assert lo95 <= lo80 <= point <= hi80 <= hi95

    def test_model_save_load(self, synthetic_daily_data, tmp_path):
        """Test model save and load functionality."""
        X_train, y_train = synthetic_daily_data

        # Train model
        model = ARIMADemandMeanModel()
        config = ARIMAConfig(
            seasonal=False,
            max_p=2,
            max_d=1,
            max_q=2,
            min_training_days=14,
        )
        model.fit(X_train, y_train, config={"arima_config": config})

        # Save model
        model_path = tmp_path / "arima_model.joblib"
        model.save(model_path)
        assert model_path.exists()

        # Load model
        loaded_model = ARIMADemandMeanModel.load(model_path)
        assert loaded_model.is_fitted()
        assert loaded_model.name == model.name
        assert loaded_model.version == model.version

        # Check predictions are the same
        pred_original = model.predict(X_train, horizons=[0, 1, 2])
        pred_loaded = loaded_model.predict(X_train, horizons=[0, 1, 2])

        pd.testing.assert_frame_equal(pred_original, pred_loaded)

    def test_insufficient_data_error(self):
        """Test error when insufficient training data."""
        # Very short series
        X = pd.DataFrame(
            {"day_of_week": [0, 1, 2]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        y = pd.Series([100.0, 110.0, 105.0], index=X.index)

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(min_training_days=14)

        with pytest.raises(ValueError, match="Data validation failed"):
            model.fit(X, y, config={"arima_config": config})

    def test_predict_before_fit_error(self, synthetic_daily_data):
        """Test error when predicting before fitting."""
        X, y = synthetic_daily_data

        model = ARIMADemandMeanModel()

        with pytest.raises(ValueError, match="has not been fitted"):
            model.predict(X, horizons=[0, 1])

    def test_invalid_horizons_error(self, synthetic_daily_data):
        """Test error with invalid horizons."""
        X_train, y_train = synthetic_daily_data

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(seasonal=False, min_training_days=14)
        model.fit(X_train, y_train, config={"arima_config": config})

        with pytest.raises(ValueError, match="Invalid horizons"):
            model.predict(X_train, horizons=[0, 1, 99])  # 99 is invalid

    def test_get_diagnostics(self, synthetic_daily_data):
        """Test model diagnostics retrieval."""
        X_train, y_train = synthetic_daily_data

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(seasonal=True, season_length=7, min_training_days=14)
        model.fit(X_train, y_train, config={"arima_config": config})

        diagnostics = model.get_diagnostics()

        assert "model_name" in diagnostics
        assert diagnostics["model_name"] == "arima_demand_mean"
        assert "config" in diagnostics
        assert "training_stats" in diagnostics
        assert diagnostics["training_stats"]["n_samples"] == 60
        assert "mean" in diagnostics["training_stats"]
        assert "std" in diagnostics["training_stats"]

    def test_get_feature_importance(self, synthetic_daily_data):
        """Test feature importance (should be empty for ARIMA)."""
        X_train, y_train = synthetic_daily_data

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(seasonal=False, min_training_days=14)
        model.fit(X_train, y_train, config={"arima_config": config})

        importance = model.get_feature_importance()

        # ARIMA doesn't have feature importance
        assert isinstance(importance, dict)
        assert len(importance) == 0

    def test_dataframe_target(self, synthetic_daily_data):
        """Test training with DataFrame target (single column)."""
        X, y = synthetic_daily_data

        # Convert y to DataFrame
        y_df = y.to_frame(name="demand")

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(seasonal=False, min_training_days=14)

        # Should work with single-column DataFrame
        model.fit(X, y_df, config={"arima_config": config})
        assert model.is_fitted()

    def test_dataframe_target_multiple_columns_error(self, synthetic_daily_data):
        """Test error with multi-column DataFrame target."""
        X, y = synthetic_daily_data

        # Create multi-column DataFrame
        y_df = pd.DataFrame(
            {"demand1": y, "demand2": y * 1.1},
            index=y.index,
        )

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(seasonal=False, min_training_days=14)

        with pytest.raises(ValueError, match="must be single column"):
            model.fit(X, y_df, config={"arima_config": config})

    def test_load_nonexistent_file_error(self):
        """Test error when loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            ARIMADemandMeanModel.load("/nonexistent/path.joblib")

    def test_aggregation_semihourly_to_daily(self, synthetic_semihourly_data):
        """Test full training pipeline with semi-hourly data."""
        X, y = synthetic_semihourly_data

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(
            seasonal=True,
            season_length=7,
            max_p=2,
            max_d=1,
            max_q=2,
            min_training_days=14,
        )

        # Train on semi-hourly data (model will aggregate internally)
        model.fit(X, y, config={"arima_config": config})

        assert model.is_fitted()
        # Should have 30 daily observations
        assert len(model._training_series) == 30

    def test_prediction_all_horizons(self, synthetic_daily_data):
        """Test prediction with all supported horizons."""
        X_train, y_train = synthetic_daily_data

        model = ARIMADemandMeanModel()
        config = ARIMAConfig(seasonal=False, min_training_days=14)
        model.fit(X_train, y_train, config={"arima_config": config})

        # Predict all horizons (default)
        predictions = model.predict(X_train)

        assert predictions.shape[1] == 9  # D+0 through D+8
        assert all(f"h{h}" in predictions.columns for h in range(9))
