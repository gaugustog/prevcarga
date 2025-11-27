"""Tests for RegDinSVMModel hierarchical forecasting model."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from src.models.config.arima_config import ARIMAConfig
from src.models.config.svm_config import SVMProfileConfig
from src.models.demand_mean.arima_model import ARIMADemandMeanModel
from src.models.hierarchical.regdin_svm import RegDinSVMModel
from src.models.profile.svm_profile import SVMProfileModel


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
def regdin_svm_config():
    """Create configuration for RegDin+SVM model."""
    config = {
        "demand_mean_config": {
            "arima_config": ARIMAConfig(
                seasonal=True,
                season_length=7,  # Weekly seasonality
                stepwise=True,
                approximation=False,
                max_p=2,
                max_q=2,
                max_P=1,
                max_Q=1,
                min_training_days=14,
            )
        },
        "profile_config": {
            "svm_config": SVMProfileConfig(
                kernel="rbf",
                C=1.0,
                epsilon=0.1,
                gamma="scale",
                optimize_hyperparams=False,  # Disable for faster tests
                min_samples_per_period=10,  # Minimum allowed is 10
                min_profile_ratio=0.1,
                max_profile_ratio=5.0,
            )
        },
    }
    return config


@pytest.fixture
def trained_regdin_svm_model(synthetic_semi_hourly_data, regdin_svm_config):
    """Create and train a RegDinSVMModel for testing."""
    # Split data
    train_size = int(len(synthetic_semi_hourly_data) * 0.7)
    train_data = synthetic_semi_hourly_data.iloc[:train_size]

    # Create model
    model = RegDinSVMModel()

    # Prepare data (X is empty, y is load)
    X_train = pd.DataFrame(index=train_data.index)
    y_train = train_data["load"]

    # Train model
    model.fit(X_train, y_train, regdin_svm_config)

    return model


# Tests for RegDinSVMModel


class TestRegDinSVMInitialization:
    """Tests for RegDinSVMModel initialization and properties."""

    def test_regdin_svm_initialization(self):
        """Test that RegDinSVMModel initializes correctly."""
        model = RegDinSVMModel()

        assert model is not None
        assert not model._is_fitted
        assert model.demand_mean_model is None
        assert len(model.profile_models) == 0

    def test_regdin_svm_properties(self):
        """Test RegDinSVMModel properties."""
        model = RegDinSVMModel()

        assert model.name == "regdin_svm"
        assert model.version == "1.0.0"
        assert model.supported_horizons == list(range(9))  # D+0 to D+8

    def test_demand_mean_model_class(self):
        """Test that demand_mean_model_class returns correct class."""
        model = RegDinSVMModel()

        assert model.demand_mean_model_class == ARIMADemandMeanModel

    def test_profile_model_class(self):
        """Test that profile_model_class returns correct class."""
        model = RegDinSVMModel()

        assert model.profile_model_class == SVMProfileModel


class TestPrepareData:
    """Tests for data preparation methods."""

    def test_prepare_demand_mean_data(self, synthetic_semi_hourly_data):
        """Test preparation of demand mean data from semi-hourly data."""
        model = RegDinSVMModel()

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
        model = RegDinSVMModel()

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
        model = RegDinSVMModel()

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
        model = RegDinSVMModel()

        df = synthetic_semi_hourly_data.copy()
        demand_mean = pd.Series(data=1000.0, index=pd.to_datetime(["2024-01-01"]))

        # Invalid period
        with pytest.raises(ValueError, match="Period must be in range"):
            model._prepare_profile_data(df, demand_mean, period=48)

        with pytest.raises(ValueError, match="Period must be in range"):
            model._prepare_profile_data(df, demand_mean, period=-1)

    def test_calculate_profile_ratios(self, synthetic_semi_hourly_data):
        """Test calculation of profile ratios."""
        model = RegDinSVMModel()

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

    def test_two_stage_training(self, synthetic_semi_hourly_data, regdin_svm_config):
        """Test that two-stage training completes successfully."""
        model = RegDinSVMModel()

        # Prepare data
        X = pd.DataFrame(index=synthetic_semi_hourly_data.index)
        y = synthetic_semi_hourly_data["load"]

        # Train model
        model.fit(X, y, regdin_svm_config)

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
        model = RegDinSVMModel()

        # Create very small dataset (less than minimum required)
        timestamps = pd.date_range("2024-01-01", periods=100, freq="30min")
        X = pd.DataFrame(index=timestamps)
        y = pd.Series(data=1000.0, index=timestamps)

        config = {
            "demand_mean_config": {
                "arima_config": ARIMAConfig(min_training_days=7)  # Minimum allowed is 7
            },
            "profile_config": {
                "svm_config": SVMProfileConfig(min_samples_per_period=10)  # Minimum allowed is 10
            },
        }

        # Training should fail or warn about insufficient data
        # Depending on implementation, this may raise an error or log warnings
        try:
            model.fit(X, y, config)
            # If it doesn't raise, check that some models were trained
            assert model._is_fitted or len(model.profile_models) > 0
        except (ValueError, RuntimeError) as e:
            # Expected behavior for insufficient data
            assert "validation failed" in str(e).lower() or "insufficient" in str(e).lower()


class TestPrediction:
    """Tests for model prediction."""

    def test_multi_horizon_prediction(
        self,
        trained_regdin_svm_model,
        synthetic_semi_hourly_data,
    ):
        """Test prediction for multiple horizons."""
        model = trained_regdin_svm_model

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
        trained_regdin_svm_model,
        synthetic_semi_hourly_data,
    ):
        """Test prediction for single horizon (D+0)."""
        model = trained_regdin_svm_model

        test_data = synthetic_semi_hourly_data.iloc[-48:]  # Last day
        X_test = pd.DataFrame(index=test_data.index)

        # Predict for D+0 only
        predictions = model.predict(X_test, horizons=[0])

        # Check output
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[1] == 48

    def test_prediction_not_fitted_raises_error(self):
        """Test that prediction fails if model is not fitted."""
        model = RegDinSVMModel()

        X_test = pd.DataFrame(index=pd.date_range("2024-01-01", periods=48, freq="30min"))

        with pytest.raises(ValueError, match="has not been fitted"):
            model.predict(X_test)


class TestEnergyConservation:
    """Tests for energy conservation validation."""

    def test_energy_conservation(
        self,
        trained_regdin_svm_model,
        synthetic_semi_hourly_data,
    ):
        """Test that energy conservation is maintained."""
        model = trained_regdin_svm_model

        # Get test data
        test_data = synthetic_semi_hourly_data.iloc[-7 * 48 :]
        X_test = pd.DataFrame(index=test_data.index)

        # Get decomposition
        decomp = model.get_forecast_decomposition(X_test)
        demand_mean = decomp["demand_mean"]
        combined_load = decomp["combined_load"]

        # Validate energy conservation
        validation = model.validate_energy_conservation(
            demand_mean,
            combined_load,
            tolerance=0.05,  # 5%
        )

        # Check validation results
        assert isinstance(validation, dict)
        assert "is_valid" in validation
        assert "mean_error" in validation
        assert "max_error" in validation
        assert "n_violations" in validation

        # Energy conservation should be reasonably good
        # (may not be perfect due to clipping, approximation, and forecasting errors)
        # ARIMA forecasts and SVM profile predictions introduce additional variance
        assert validation["mean_error"] < 0.15  # Mean error < 15%

    def test_energy_conservation_with_custom_tolerance(
        self,
        trained_regdin_svm_model,
        synthetic_semi_hourly_data,
    ):
        """Test energy conservation with custom tolerance."""
        model = trained_regdin_svm_model

        test_data = synthetic_semi_hourly_data.iloc[-48:]
        X_test = pd.DataFrame(index=test_data.index)

        decomp = model.get_forecast_decomposition(X_test)
        demand_mean = decomp["demand_mean"]
        combined_load = decomp["combined_load"]

        # Very strict tolerance
        validation_strict = model.validate_energy_conservation(
            demand_mean,
            combined_load,
            tolerance=0.01,  # 1%
        )

        # Loose tolerance
        validation_loose = model.validate_energy_conservation(
            demand_mean,
            combined_load,
            tolerance=0.50,  # 50%
        )

        # Loose tolerance should have fewer violations
        assert validation_loose["n_violations"] <= validation_strict["n_violations"]


class TestForecastDecomposition:
    """Tests for forecast decomposition."""

    def test_forecast_decomposition(
        self,
        trained_regdin_svm_model,
        synthetic_semi_hourly_data,
    ):
        """Test that forecast decomposition returns correct components."""
        model = trained_regdin_svm_model

        test_data = synthetic_semi_hourly_data.iloc[-7 * 48 :]
        X_test = pd.DataFrame(index=test_data.index)

        # Get decomposition
        decomp = model.get_forecast_decomposition(X_test)

        # Check that all components are present
        assert "demand_mean" in decomp
        assert "profiles" in decomp
        assert "combined_load" in decomp
        assert "horizons" in decomp

        # Check demand_mean
        demand_mean = decomp["demand_mean"]
        assert isinstance(demand_mean, pd.Series)
        assert len(demand_mean) == 7  # 7 days

        # Check profiles
        profiles = decomp["profiles"]
        assert isinstance(profiles, dict)
        assert len(profiles) > 0  # At least some periods

        # Check combined_load
        combined_load = decomp["combined_load"]
        assert isinstance(combined_load, pd.DataFrame)
        assert combined_load.shape[1] == 48


class TestModelDiagnostics:
    """Tests for model diagnostics."""

    def test_model_diagnostics(self, trained_regdin_svm_model):
        """Test that model diagnostics returns comprehensive information."""
        model = trained_regdin_svm_model

        diagnostics = model.get_model_diagnostics()

        # Check that all expected fields are present
        assert "model_name" in diagnostics
        assert "model_version" in diagnostics
        assert "supported_horizons" in diagnostics
        assert "demand_mean_model" in diagnostics
        assert "profile_models" in diagnostics
        assert "training_metadata" in diagnostics
        assert "n_profile_models" in diagnostics

        # Check values
        assert diagnostics["model_name"] == "regdin_svm"
        assert diagnostics["model_version"] == "1.0.0"
        assert diagnostics["supported_horizons"] == list(range(9))
        assert diagnostics["n_profile_models"] > 0

        # Check demand mean model diagnostics
        dm_diag = diagnostics["demand_mean_model"]
        assert dm_diag is not None
        assert "model_name" in dm_diag

        # Check profile models diagnostics
        prof_diag = diagnostics["profile_models"]
        assert "n_models" in prof_diag
        assert prof_diag["n_models"] > 0

    def test_diagnostics_not_fitted_raises_error(self):
        """Test that diagnostics fails if model is not fitted."""
        model = RegDinSVMModel()

        with pytest.raises(ValueError, match="has not been fitted"):
            model.get_model_diagnostics()


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_model_save_load(self, trained_regdin_svm_model):
        """Test that model can be saved and loaded."""
        model = trained_regdin_svm_model

        with TemporaryDirectory() as tmpdir:
            # Save model
            model_path = Path(tmpdir) / "regdin_svm_model.pkl"
            model.save(model_path)

            # Check that file exists
            assert model_path.exists()

            # Load model
            loaded_model = RegDinSVMModel.load(model_path)

            # Check that loaded model has same properties
            assert loaded_model.name == model.name
            assert loaded_model.version == model.version
            assert loaded_model._is_fitted == model._is_fitted
            assert len(loaded_model.profile_models) == len(model.profile_models)

    def test_save_not_fitted_raises_error(self):
        """Test that saving unfitted model raises error."""
        model = RegDinSVMModel()

        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            with pytest.raises(ValueError, match="has not been fitted"):
                model.save(model_path)

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            RegDinSVMModel.load("/nonexistent/path/model.pkl")


class TestRepr:
    """Tests for string representation."""

    def test_repr_unfitted(self):
        """Test repr for unfitted model."""
        model = RegDinSVMModel()
        repr_str = repr(model)

        assert "RegDinSVMModel" in repr_str
        assert "fitted=False" in repr_str

    def test_repr_fitted(self, trained_regdin_svm_model):
        """Test repr for fitted model."""
        model = trained_regdin_svm_model
        repr_str = repr(model)

        assert "RegDinSVMModel" in repr_str
        assert "fitted=True" in repr_str
        assert "n_profile_models=" in repr_str


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_prediction_with_missing_periods(
        self,
        trained_regdin_svm_model,
        synthetic_semi_hourly_data,
    ):
        """Test prediction when some periods have no trained models."""
        model = trained_regdin_svm_model

        # Manually remove some period models to simulate missing periods
        periods_to_remove = [5, 10, 15, 20]
        for period in periods_to_remove:
            if period in model.profile_models:
                del model.profile_models[period]

        # Prediction should still work (using default ratios for missing periods)
        test_data = synthetic_semi_hourly_data.iloc[-48:]
        X_test = pd.DataFrame(index=test_data.index)

        predictions = model.predict(X_test, horizons=[0])

        # Check that predictions are still generated
        assert predictions.shape[1] == 48
        assert (predictions > 0).all().all()

    def test_feature_importance(self, trained_regdin_svm_model):
        """Test feature importance aggregation."""
        model = trained_regdin_svm_model

        importance = model.get_feature_importance()

        # Should return a dict (may be empty since ARIMA and SVM don't provide importance)
        assert isinstance(importance, dict)
