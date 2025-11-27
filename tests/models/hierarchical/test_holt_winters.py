"""Tests for HoltWintersModel hierarchical forecasting model."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from src.models.config.holtwinters_config import (
    HoltWintersConfig,
    HoltWintersProfileConfig,
)
from src.models.demand_mean.holt_winters import HoltWintersDemandMeanModel
from src.models.hierarchical.holt_winters import HoltWintersModel
from src.models.profile.holt_winters_profile import HoltWintersProfileModel


# Test Fixtures


@pytest.fixture
def synthetic_semi_hourly_data():
    """Generate synthetic semi-hourly load data with realistic patterns.

    Creates data with:
    - Daily trend (increasing over time)
    - Intraday pattern (higher during day, lower at night)
    - Weekly seasonality
    """
    # Generate 30 days of semi-hourly data
    n_days = 30
    periods_per_day = 48
    n_samples = n_days * periods_per_day

    # Create timestamp index
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=30 * i) for i in range(n_samples)]
    index = pd.DatetimeIndex(timestamps)

    # Base daily demand mean (MW) with increasing trend
    daily_trend = np.linspace(1000, 1200, n_days)

    # Weekly seasonality (weekdays higher than weekends)
    day_of_week = np.array([ts.weekday() for ts in timestamps[: periods_per_day * 7]])
    weekly_pattern = np.where(day_of_week < 5, 1.1, 0.9)  # Weekday/weekend multiplier
    weekly_pattern = np.tile(weekly_pattern, n_days // 7 + 1)[:n_samples]

    # Intraday pattern (semi-hourly profile ratios)
    # Higher during business hours (8am-6pm), lower at night
    hours = np.array([i // 2 for i in range(periods_per_day)])
    intraday_pattern = np.where(
        (hours >= 8) & (hours < 18),
        1.3,  # Peak hours
        0.7,  # Off-peak hours
    )

    # Add smooth transitions
    for i in range(len(intraday_pattern)):
        hour = hours[i]
        if 6 <= hour < 8:  # Morning ramp-up
            intraday_pattern[i] = 0.7 + 0.6 * (hour - 6) / 2
        elif 18 <= hour < 22:  # Evening ramp-down
            intraday_pattern[i] = 1.3 - 0.6 * (hour - 18) / 4

    # Tile intraday pattern for all days
    intraday_pattern_full = np.tile(intraday_pattern, n_days)

    # Expand daily trend to semi-hourly
    demand_mean_expanded = np.repeat(daily_trend, periods_per_day)

    # Combine components
    load = demand_mean_expanded * weekly_pattern * intraday_pattern_full

    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 20, n_samples)  # 20 MW noise
    load = load + noise

    # Ensure non-negative
    load = np.maximum(load, 10)

    # Create DataFrame
    df = pd.DataFrame({"load": load}, index=index)

    return df


@pytest.fixture
def holt_winters_config():
    """Create configuration for Holt-Winters model."""
    config = {
        "demand_mean_config": {
            "holtwinters_config": HoltWintersConfig(
                trend="add",
                seasonal="add",
                seasonal_periods=7,  # Weekly seasonality
                damped_trend=False,
                use_boxcox=False,
                initialization_method="estimated",
                min_training_days=14,
            )
        },
        "profile_config": {
            "holtwinters_config": HoltWintersProfileConfig(
                trend=None,  # Profiles typically don't have trend
                seasonal="add",
                seasonal_periods=7,
                damped_trend=False,
                use_boxcox=False,
                initialization_method="estimated",
                min_samples_per_period=10,
                min_profile_ratio=0.1,
                max_profile_ratio=5.0,
            )
        },
    }
    return config


@pytest.fixture
def trained_holt_winters_model(synthetic_semi_hourly_data, holt_winters_config):
    """Create and train a HoltWintersModel for testing."""
    # Split data
    train_size = int(len(synthetic_semi_hourly_data) * 0.7)
    train_data = synthetic_semi_hourly_data.iloc[:train_size]

    # Create model
    model = HoltWintersModel()

    # Prepare data (X is empty, y is load)
    X_train = pd.DataFrame(index=train_data.index)
    y_train = train_data["load"]

    # Train model
    model.fit(X_train, y_train, holt_winters_config)

    return model


# Tests for HoltWintersModel


class TestHoltWintersInitialization:
    """Tests for HoltWintersModel initialization and properties."""

    def test_holt_winters_initialization(self):
        """Test that HoltWintersModel initializes correctly."""
        model = HoltWintersModel()

        assert model is not None
        assert not model._is_fitted
        assert model.demand_mean_model is None
        assert len(model.profile_models) == 0

    def test_holt_winters_properties(self):
        """Test HoltWintersModel properties."""
        model = HoltWintersModel()

        assert model.name == "holt_winters"
        assert model.version == "1.0.0"
        assert model.supported_horizons == list(range(9))  # D+0 to D+8

    def test_demand_mean_model_class(self):
        """Test that demand_mean_model_class returns correct class."""
        model = HoltWintersModel()

        assert model.demand_mean_model_class == HoltWintersDemandMeanModel

    def test_profile_model_class(self):
        """Test that profile_model_class returns correct class."""
        model = HoltWintersModel()

        assert model.profile_model_class == HoltWintersProfileModel


class TestPrepareData:
    """Tests for data preparation methods."""

    def test_prepare_demand_mean_data(self, synthetic_semi_hourly_data):
        """Test preparation of demand mean data from semi-hourly data."""
        model = HoltWintersModel()

        # Add load column
        df = synthetic_semi_hourly_data.copy()

        # Prepare demand mean data
        X_daily, y_daily = model._prepare_demand_mean_data(df)

        # Check that data is aggregated to daily
        assert len(y_daily) == len(df) // 48  # 48 periods per day
        assert isinstance(y_daily.index, pd.DatetimeIndex)
        assert isinstance(X_daily, pd.DataFrame)

        # Check that y_daily contains reasonable values
        assert y_daily.min() > 0
        assert not y_daily.isna().any()

    def test_prepare_demand_mean_data_prediction(self, synthetic_semi_hourly_data):
        """Test preparation of demand mean data for prediction (no load column)."""
        model = HoltWintersModel()

        # Create df without load column
        df = pd.DataFrame(index=synthetic_semi_hourly_data.index)

        # Prepare demand mean data for prediction
        X_daily, y_daily = model._prepare_demand_mean_data(df)

        # Check that daily data is created
        assert len(X_daily) == len(df) // 48
        assert isinstance(X_daily.index, pd.DatetimeIndex)
        assert isinstance(y_daily, pd.Series)

    def test_prepare_profile_data(self, synthetic_semi_hourly_data):
        """Test preparation of profile data for a specific period."""
        model = HoltWintersModel()

        # Add load column
        df = synthetic_semi_hourly_data.copy()

        # Create dummy demand mean (daily)
        dates = df.index.date
        unique_dates = pd.Series(dates).unique()
        demand_mean = pd.Series(
            data=1000.0,
            index=pd.to_datetime(unique_dates),
        )

        # Prepare profile data for period 0
        period = 0
        X_period, y_period = model._prepare_profile_data(df, demand_mean, period)

        # Check that data is filtered for period 0
        # Period 0 is 00:00-00:30, so we expect one sample per day
        assert len(y_period) == len(demand_mean)
        assert isinstance(y_period, pd.Series)
        assert isinstance(X_period, pd.DataFrame)

        # Check that profile ratios are reasonable (> 0)
        assert (y_period > 0).all()

    def test_prepare_profile_data_invalid_period(self, synthetic_semi_hourly_data):
        """Test that invalid period raises error."""
        model = HoltWintersModel()

        df = synthetic_semi_hourly_data.copy()
        demand_mean = pd.Series(data=1000.0, index=pd.to_datetime(["2024-01-01"]))

        # Invalid period
        with pytest.raises(ValueError, match="Period must be in range"):
            model._prepare_profile_data(df, demand_mean, period=48)

        with pytest.raises(ValueError, match="Period must be in range"):
            model._prepare_profile_data(df, demand_mean, period=-1)

    def test_calculate_profile_ratios(self, synthetic_semi_hourly_data):
        """Test calculation of profile ratios."""
        model = HoltWintersModel()

        # Get semi-hourly load
        y = synthetic_semi_hourly_data["load"]

        # Create daily demand mean
        daily_mean = y.groupby(y.index.date).mean()
        daily_mean.index = pd.to_datetime(daily_mean.index)

        # Calculate profile ratios
        profile_ratios = model._calculate_profile_ratios(y, daily_mean)

        # Check output
        assert len(profile_ratios) == len(y)
        assert isinstance(profile_ratios, pd.Series)

        # Check that ratios are positive and reasonable
        assert (profile_ratios > 0).all()
        assert profile_ratios.min() >= 0.01  # MIN_PROFILE_RATIO
        assert profile_ratios.max() <= 10.0  # MAX_PROFILE_RATIO

        # Check that mean ratio is close to 1.0 (by definition)
        # Average of all periods should be close to 1.0
        assert 0.8 < profile_ratios.mean() < 1.2


class TestTraining:
    """Tests for model training."""

    def test_two_stage_training(self, synthetic_semi_hourly_data, holt_winters_config):
        """Test that two-stage training completes successfully."""
        model = HoltWintersModel()

        # Prepare data
        X = pd.DataFrame(index=synthetic_semi_hourly_data.index)
        y = synthetic_semi_hourly_data["load"]

        # Train model
        model.fit(X, y, holt_winters_config)

        # Check that model is fitted
        assert model._is_fitted
        assert model.demand_mean_model is not None
        assert model.demand_mean_model._is_fitted
        assert len(model.profile_models) > 0

        # Check that profile models are trained for all periods
        # (at least most of them, some may be skipped if insufficient data)
        assert len(model.profile_models) >= 40  # At least 40 out of 48 periods

    def test_training_with_insufficient_data(self):
        """Test training with insufficient data."""
        model = HoltWintersModel()

        # Create very small dataset (less than minimum required)
        timestamps = pd.date_range("2024-01-01", periods=100, freq="30min")
        X = pd.DataFrame(index=timestamps)
        y = pd.Series(data=1000.0, index=timestamps)

        config = {
            "demand_mean_config": {
                "holtwinters_config": HoltWintersConfig(
                    seasonal="add",
                    seasonal_periods=7,
                    min_training_days=7  # Minimum allowed is 7
                )
            },
            "profile_config": {
                "holtwinters_config": HoltWintersProfileConfig(
                    seasonal="add",
                    seasonal_periods=7,
                    min_samples_per_period=10  # Minimum allowed is 10
                )
            },
        }

        # Training should fail due to insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            model.fit(X, y, config)

    def test_training_diagnostics(self, trained_holt_winters_model):
        """Test that training metadata is stored correctly."""
        model = trained_holt_winters_model

        assert "n_daily_samples" in model._training_metadata
        assert "n_semi_hourly_samples" in model._training_metadata
        assert "n_profile_models" in model._training_metadata

        # Demand mean model should have diagnostics
        dm_diagnostics = model.demand_mean_model.get_diagnostics()
        assert "model_params" in dm_diagnostics
        assert "smoothing_level" in dm_diagnostics["model_params"]


class TestPrediction:
    """Tests for model prediction."""

    def test_multi_horizon_prediction(
        self,
        trained_holt_winters_model,
        synthetic_semi_hourly_data,
    ):
        """Test prediction for multiple horizons."""
        model = trained_holt_winters_model

        # Use last 7 days for testing
        test_data = synthetic_semi_hourly_data.iloc[-7 * 48 :]
        X_test = pd.DataFrame(index=test_data.index)

        # Predict for horizons D+0 to D+3
        horizons = [0, 1, 2, 3]
        predictions = model.predict(X_test, horizons=horizons)

        # Check output shape
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[1] == 48  # 48 semi-hourly periods

        # Check that predictions are reasonable
        assert (predictions > 0).all().all()
        assert not predictions.isna().any().any()

    def test_single_horizon_prediction(
        self,
        trained_holt_winters_model,
        synthetic_semi_hourly_data,
    ):
        """Test prediction for single horizon (D+0)."""
        model = trained_holt_winters_model

        test_data = synthetic_semi_hourly_data.iloc[-48:]  # Last day
        X_test = pd.DataFrame(index=test_data.index)

        # Predict for D+0 only
        predictions = model.predict(X_test, horizons=[0])

        # Check output
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[1] == 48

    def test_prediction_not_fitted_raises_error(self):
        """Test that prediction fails if model is not fitted."""
        model = HoltWintersModel()

        X_test = pd.DataFrame(index=pd.date_range("2024-01-01", periods=48, freq="30min"))

        with pytest.raises(ValueError, match="has not been fitted"):
            model.predict(X_test)


class TestSeasonalDecomposition:
    """Tests for seasonal decomposition."""

    def test_seasonal_decomposition(self, trained_holt_winters_model):
        """Test that seasonal decomposition returns correct components."""
        model = trained_holt_winters_model

        # Get decomposition
        decomp = model.get_seasonal_decomposition()

        # Check that demand mean decomposition is present
        assert "demand_mean" in decomp
        dm_decomp = decomp["demand_mean"]

        # Check components
        assert "level" in dm_decomp
        assert "trend" in dm_decomp
        assert "seasonal" in dm_decomp
        assert "fitted_values" in dm_decomp
        assert "residuals" in dm_decomp

        # Level should be a Series
        assert isinstance(dm_decomp["level"], pd.Series)

        # Trend should be present (we configured trend="add")
        assert dm_decomp["trend"] is not None
        assert isinstance(dm_decomp["trend"], pd.Series)

        # Seasonal should be present (we configured seasonal="add")
        assert dm_decomp["seasonal"] is not None
        assert isinstance(dm_decomp["seasonal"], pd.Series)

    def test_decomposition_profile_samples(self, trained_holt_winters_model):
        """Test that profile sample decompositions are included."""
        model = trained_holt_winters_model

        decomp = model.get_seasonal_decomposition()

        # Check for profile samples
        if "profile_samples" in decomp:
            assert isinstance(decomp["profile_samples"], dict)
            # Should have at least one sample period
            assert len(decomp["profile_samples"]) > 0


class TestModelDiagnostics:
    """Tests for model diagnostics."""

    def test_model_diagnostics_not_fitted_raises_error(self):
        """Test that diagnostics fails if model is not fitted."""
        model = HoltWintersModel()

        with pytest.raises(ValueError, match="has not been fitted"):
            model.get_feature_importance()

    def test_feature_importance(self, trained_holt_winters_model):
        """Test feature importance (should return empty dict for Holt-Winters)."""
        model = trained_holt_winters_model

        importance = model.get_feature_importance()

        # Should return empty dict (Holt-Winters doesn't have feature importance)
        assert isinstance(importance, dict)
        assert len(importance) == 0


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_model_save_load(self, trained_holt_winters_model):
        """Test that model can be saved and loaded."""
        model = trained_holt_winters_model

        with TemporaryDirectory() as tmpdir:
            # Save model
            model_path = Path(tmpdir) / "holt_winters_model.pkl"
            model.save(model_path)

            # Check that file exists
            assert model_path.exists()

            # Load model
            loaded_model = HoltWintersModel.load(model_path)

            # Check that loaded model has same properties
            assert loaded_model.name == model.name
            assert loaded_model.version == model.version
            assert loaded_model._is_fitted == model._is_fitted
            assert len(loaded_model.profile_models) == len(model.profile_models)

    def test_save_not_fitted_raises_error(self):
        """Test that saving unfitted model raises error."""
        model = HoltWintersModel()

        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            with pytest.raises(ValueError, match="has not been fitted"):
                model.save(model_path)

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            HoltWintersModel.load("/nonexistent/path/model.pkl")


class TestRepr:
    """Tests for string representation."""

    def test_repr_unfitted(self):
        """Test repr for unfitted model."""
        model = HoltWintersModel()
        repr_str = repr(model)

        assert "HoltWintersModel" in repr_str
        assert "fitted=False" in repr_str

    def test_repr_fitted(self, trained_holt_winters_model):
        """Test repr for fitted model."""
        model = trained_holt_winters_model
        repr_str = repr(model)

        assert "HoltWintersModel" in repr_str
        assert "fitted=True" in repr_str
        assert "n_profile_models=" in repr_str


class TestDemandMeanModel:
    """Tests specific to HoltWintersDemandMeanModel."""

    def test_demand_mean_initialization(self):
        """Test demand mean model initialization."""
        model = HoltWintersDemandMeanModel()

        assert model.name == "holt_winters_demand_mean"
        assert model.version == "1.0.0"
        assert model.supported_horizons == list(range(9))

    def test_demand_mean_training(self, synthetic_semi_hourly_data):
        """Test demand mean model training."""
        # Aggregate to daily
        y_daily = synthetic_semi_hourly_data["load"].groupby(
            synthetic_semi_hourly_data.index.date
        ).mean()
        y_daily.index = pd.to_datetime(y_daily.index)

        X_daily = pd.DataFrame(index=y_daily.index)

        config = {
            "holtwinters_config": HoltWintersConfig(
                trend="add",
                seasonal="add",
                seasonal_periods=7,
            )
        }

        model = HoltWintersDemandMeanModel()
        model.fit(X_daily, y_daily, config)

        assert model._is_fitted
        assert model._fitted_result is not None

    def test_demand_mean_prediction(self, synthetic_semi_hourly_data):
        """Test demand mean model prediction."""
        # Aggregate to daily
        y_daily = synthetic_semi_hourly_data["load"].groupby(
            synthetic_semi_hourly_data.index.date
        ).mean()
        y_daily.index = pd.to_datetime(y_daily.index)

        X_daily = pd.DataFrame(index=y_daily.index)

        config = {
            "holtwinters_config": HoltWintersConfig(
                trend="add",
                seasonal="add",
                seasonal_periods=7,
            )
        }

        model = HoltWintersDemandMeanModel()
        model.fit(X_daily, y_daily, config)

        # Predict
        predictions = model.predict(X_daily, horizons=[0, 1, 2])

        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[1] == 3  # 3 horizons
        assert "h0" in predictions.columns
        assert "h1" in predictions.columns
        assert "h2" in predictions.columns

    def test_demand_mean_prediction_intervals(self, synthetic_semi_hourly_data):
        """Test demand mean model prediction intervals."""
        # Aggregate to daily
        y_daily = synthetic_semi_hourly_data["load"].groupby(
            synthetic_semi_hourly_data.index.date
        ).mean()
        y_daily.index = pd.to_datetime(y_daily.index)

        X_daily = pd.DataFrame(index=y_daily.index)

        config = {
            "holtwinters_config": HoltWintersConfig(
                trend="add",
                seasonal="add",
                seasonal_periods=7,
                prediction_intervals=[80, 95],
            )
        }

        model = HoltWintersDemandMeanModel()
        model.fit(X_daily, y_daily, config)

        # Predict with intervals
        predictions = model.get_prediction_intervals(X_daily, horizons=[0, 1])

        assert isinstance(predictions, pd.DataFrame)
        # Should have point forecasts and intervals
        assert "h0" in predictions.columns
        assert "h1" in predictions.columns


class TestProfileModel:
    """Tests specific to HoltWintersProfileModel."""

    def test_profile_initialization(self):
        """Test profile model initialization."""
        model = HoltWintersProfileModel()

        assert model.name == "holt_winters_profile"
        assert model.version == "1.0.0"
        assert model.supported_horizons == [0]  # Only D+0

    def test_profile_training(self, synthetic_semi_hourly_data):
        """Test profile model training."""
        # Calculate profile ratios
        y = synthetic_semi_hourly_data["load"]
        daily_mean = y.groupby(y.index.date).mean()
        daily_mean.index = pd.to_datetime(daily_mean.index)

        # Expand to semi-hourly
        demand_mean_expanded = pd.Series(index=y.index, dtype=float)
        for i, date in enumerate(y.index.date):
            date_ts = pd.Timestamp(date)
            if date_ts in daily_mean.index:
                demand_mean_expanded.iloc[i] = daily_mean.loc[date_ts]

        profile_ratios = y / (demand_mean_expanded + 1e-8)
        profile_ratios = profile_ratios.clip(0.1, 5.0)

        X = pd.DataFrame(index=y.index)

        config = {
            "holtwinters_config": HoltWintersProfileConfig(
                seasonal="add",
                seasonal_periods=7,
                trend=None,
                min_samples_per_period=10,
            )
        }

        model = HoltWintersProfileModel()
        model.fit(X, profile_ratios, config)

        assert model._is_fitted
        assert len(model._models) > 0

    def test_profile_prediction(self, synthetic_semi_hourly_data):
        """Test profile model prediction."""
        # Calculate profile ratios
        y = synthetic_semi_hourly_data["load"]
        daily_mean = y.groupby(y.index.date).mean()
        daily_mean.index = pd.to_datetime(daily_mean.index)

        # Expand to semi-hourly
        demand_mean_expanded = pd.Series(index=y.index, dtype=float)
        for i, date in enumerate(y.index.date):
            date_ts = pd.Timestamp(date)
            if date_ts in daily_mean.index:
                demand_mean_expanded.iloc[i] = daily_mean.loc[date_ts]

        profile_ratios = y / (demand_mean_expanded + 1e-8)
        profile_ratios = profile_ratios.clip(0.1, 5.0)

        X = pd.DataFrame(index=y.index)

        config = {
            "holtwinters_config": HoltWintersProfileConfig(
                seasonal="add",
                seasonal_periods=7,
                trend=None,
                min_samples_per_period=10,
            )
        }

        model = HoltWintersProfileModel()
        model.fit(X, profile_ratios, config)

        # Predict
        X_test = pd.DataFrame(index=y.index[-48:])
        predictions = model.predict(X_test, horizons=[0])

        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape == (48, 1)
        assert "h0" in predictions.columns


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_prediction_with_missing_periods(
        self,
        trained_holt_winters_model,
        synthetic_semi_hourly_data,
    ):
        """Test prediction when some periods have no trained models."""
        model = trained_holt_winters_model

        # Manually remove some period models to simulate missing periods
        periods_to_remove = [5, 10, 15, 20]
        for period in periods_to_remove:
            if period in model.profile_models:
                del model.profile_models[period]
            if period in model._profile_model._models:
                del model._profile_model._models[period]

        # Prediction should still work (using default ratios for missing periods)
        test_data = synthetic_semi_hourly_data.iloc[-48:]
        X_test = pd.DataFrame(index=test_data.index)

        predictions = model.predict(X_test, horizons=[0])

        # Check that predictions are still generated
        assert predictions.shape[1] == 48
        assert (predictions > 0).all().all()

    def test_multiplicative_seasonality_with_negative_values(self):
        """Test that multiplicative model handles non-positive values gracefully."""
        # Create data with negative values at the daily level
        timestamps = pd.date_range("2024-01-01", periods=14 * 48, freq="30min")
        np.random.seed(123)

        # Create semi-hourly data with a pattern where some days will have negative mean
        y_values = []
        for day in range(14):
            # Days 5-7 will have negative daily mean
            if 5 <= day < 7:
                daily_base = -100  # Negative base
            else:
                daily_base = 1000  # Positive base

            # Add intraday variation
            for period in range(48):
                value = daily_base + np.random.randn() * 50
                y_values.append(value)

        y = pd.Series(data=y_values, index=timestamps)

        X = pd.DataFrame(index=timestamps)

        config = {
            "demand_mean_config": {
                "holtwinters_config": HoltWintersConfig(
                    trend="mul",  # Multiplicative trend
                    seasonal="mul",  # Multiplicative seasonal
                    seasonal_periods=7,
                )
            },
            "profile_config": {
                "holtwinters_config": HoltWintersProfileConfig(
                    seasonal="add",
                    seasonal_periods=7,
                )
            },
        }

        model = HoltWintersModel()

        # Should raise error for multiplicative with non-positive values
        # The error comes from the demand mean aggregation step
        with pytest.raises(ValueError, match="Multiplicative.*requires all positive values"):
            model.fit(X, y, config)
