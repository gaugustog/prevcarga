"""Tests for BaseHierarchicalModel abstract class and ProfileCombiner."""

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.base.model import BaseModel
from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.models.hierarchical.profile_combiner import ProfileCombiner


# Mock Models for Testing


class MockDemandMeanModel(BaseModel):
    """Mock model for demand mean forecasting."""

    def __init__(self):
        super().__init__()
        self._multiplier = 1.0

    @property
    def name(self) -> str:
        return "mock_demand_mean"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_horizons(self) -> list[int]:
        return [0, 1, 2, 3]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Mock fit implementation."""
        self._feature_names = list(X.columns)
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_config": config,
        }
        # Store mean of y for predictions
        self._multiplier = y.mean() if len(y) > 0 else 1000.0
        self._is_fitted = True

    def predict(
        self,
        X: pd.DataFrame,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Mock predict implementation - returns constant based on training data."""
        self._check_is_fitted()

        if horizons is None:
            horizons = self.supported_horizons

        # Return constant prediction based on training mean
        return pd.DataFrame(
            index=X.index,
            columns=[f"h{h}" for h in horizons],
            data=self._multiplier,
        )

    def get_feature_importance(self) -> dict[str, float]:
        """Mock feature importance."""
        self._check_is_fitted()

        if not self._feature_names:
            return {}

        n_features = len(self._feature_names)
        importance = 1.0 / n_features
        return dict.fromkeys(self._feature_names, importance)


class MockProfileModel(BaseModel):
    """Mock model for profile ratio forecasting."""

    def __init__(self):
        super().__init__()
        self._mean_ratio = 1.0

    @property
    def name(self) -> str:
        return "mock_profile"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_horizons(self) -> list[int]:
        return [0, 1, 2, 3]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Mock fit implementation."""
        self._feature_names = list(X.columns)
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_config": config,
        }
        # Store mean ratio for predictions
        self._mean_ratio = y.mean() if len(y) > 0 else 1.0
        self._is_fitted = True

    def predict(
        self,
        X: pd.DataFrame,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Mock predict implementation - returns constant ratio."""
        self._check_is_fitted()

        if horizons is None:
            horizons = self.supported_horizons

        return pd.DataFrame(
            index=X.index,
            columns=[f"h{h}" for h in horizons],
            data=self._mean_ratio,
        )

    def get_feature_importance(self) -> dict[str, float]:
        """Mock feature importance."""
        self._check_is_fitted()

        if not self._feature_names:
            return {}

        n_features = len(self._feature_names)
        importance = 1.0 / n_features
        return dict.fromkeys(self._feature_names, importance)


# Concrete Test Implementation of BaseHierarchicalModel


class TestHierarchicalModel(BaseHierarchicalModel):
    """Concrete implementation of BaseHierarchicalModel for testing."""

    @property
    def name(self) -> str:
        return "test_hierarchical"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_horizons(self) -> list[int]:
        return [0, 1, 2, 3]

    @property
    def demand_mean_model_class(self) -> type[BaseModel]:
        return MockDemandMeanModel

    @property
    def profile_model_class(self) -> type[BaseModel]:
        return MockProfileModel

    def _prepare_demand_mean_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare daily aggregated data for demand mean model."""
        # Validate required columns
        if "timestamp" not in df.columns:
            msg = "DataFrame must contain 'timestamp' column"
            raise ValueError(msg)

        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        # For prediction, load column may not exist
        has_load = "load" in df.columns

        # Aggregate to daily
        if has_load:
            daily = df.groupby("date").agg(
                {
                    "load": "mean",  # Demand mean is daily average
                    "hour": "mean",  # Average hour as feature
                }
            )
            X_daily = daily[["hour"]]
            y_daily = daily["load"]
        else:
            # During prediction, no target
            daily = df.groupby("date").agg(
                {
                    "hour": "mean",  # Average hour as feature
                }
            )
            X_daily = daily[["hour"]]
            y_daily = pd.Series(dtype=float)  # Empty series

        return X_daily, y_daily

    def _prepare_profile_data(
        self,
        df: pd.DataFrame,
        demand_mean: pd.Series,
        period: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare profile data for a specific semi-hourly period."""
        if period < 0 or period >= 48:
            msg = f"Period must be in range 0-47, got {period}"
            raise ValueError(msg)

        # Filter by period
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        # Calculate period from timestamp
        df["period"] = (
            pd.to_datetime(df["timestamp"]).dt.hour * 2
            + (pd.to_datetime(df["timestamp"]).dt.minute >= 30).astype(int)
        )

        period_data = df[df["period"] == period].copy()

        if len(period_data) == 0:
            # Return empty data
            return pd.DataFrame(columns=["hour"]), pd.Series(dtype=float)

        # For prediction, load column may not exist
        has_load = "load" in period_data.columns

        # Merge with demand mean
        period_data = period_data.set_index("date")
        period_data["demand_mean"] = demand_mean

        if has_load:
            # Calculate profile ratio
            period_data["profile_ratio"] = period_data["load"] / period_data["demand_mean"]

            # Handle division by zero
            period_data["profile_ratio"] = period_data["profile_ratio"].replace(
                [np.inf, -np.inf], 1.0
            )
            period_data["profile_ratio"] = period_data["profile_ratio"].fillna(1.0)

            X_period = period_data[["hour"]]
            y_period = period_data["profile_ratio"]
        else:
            # During prediction, no target
            X_period = period_data[["hour"]]
            y_period = pd.Series(dtype=float)  # Empty series

        return X_period, y_period


# Tests for ProfileCombiner


class TestProfileCombinerInitialization:
    """Test ProfileCombiner initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        combiner = ProfileCombiner()

        assert combiner.min_profile_ratio == 0.1
        assert combiner.max_profile_ratio == 5.0

    def test_custom_bounds(self):
        """Test initialization with custom bounds."""
        combiner = ProfileCombiner(min_profile_ratio=0.2, max_profile_ratio=3.0)

        assert combiner.min_profile_ratio == 0.2
        assert combiner.max_profile_ratio == 3.0

    def test_invalid_bounds_raises_error(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError, match="must be less than"):
            ProfileCombiner(min_profile_ratio=5.0, max_profile_ratio=0.1)

    def test_non_positive_min_raises_error(self):
        """Test that non-positive min bound raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            ProfileCombiner(min_profile_ratio=0.0, max_profile_ratio=5.0)

        with pytest.raises(ValueError, match="must be positive"):
            ProfileCombiner(min_profile_ratio=-0.1, max_profile_ratio=5.0)

    def test_non_positive_max_raises_error(self):
        """Test that non-positive max bound raises error."""
        with pytest.raises(ValueError, match="must be less than|must be positive"):
            ProfileCombiner(min_profile_ratio=0.1, max_profile_ratio=0.0)


class TestProfileCombinerCombine:
    """Test ProfileCombiner.combine() method."""

    @pytest.fixture
    def combiner(self):
        """Create combiner instance."""
        return ProfileCombiner()

    @pytest.fixture
    def sample_demand_mean(self):
        """Create sample demand mean data."""
        return pd.Series([1000.0, 1100.0, 1200.0], index=pd.date_range("2024-01-01", periods=3))

    def test_combine_with_all_profiles(self, combiner, sample_demand_mean):
        """Test combining with all 48 profiles."""
        # Create profiles (constant ratio of 1.2 for all periods)
        profiles = {
            period: pd.Series([1.2, 1.2, 1.2], index=sample_demand_mean.index)
            for period in range(48)
        }

        result = combiner.combine(sample_demand_mean, profiles)

        assert result.shape == (3, 48)
        assert list(result.columns) == [f"period_{i}" for i in range(48)]
        # All values should be demand_mean * 1.2
        expected = np.tile(sample_demand_mean.values[:, None], (1, 48)) * 1.2
        np.testing.assert_allclose(result.values, expected, rtol=1e-6)

    def test_combine_with_missing_profiles(self, combiner, sample_demand_mean):
        """Test combining with some profiles missing."""
        # Only provide profiles for periods 0, 1, 2
        profiles = {
            0: pd.Series([1.2, 1.2, 1.2], index=sample_demand_mean.index),
            1: pd.Series([0.8, 0.8, 0.8], index=sample_demand_mean.index),
            2: pd.Series([1.5, 1.5, 1.5], index=sample_demand_mean.index),
        }

        result = combiner.combine(sample_demand_mean, profiles)

        assert result.shape == (3, 48)
        # Periods 0-2 should have specified ratios
        np.testing.assert_allclose(
            result["period_0"].values,
            sample_demand_mean.values * 1.2,
            rtol=1e-6,
        )
        # Period 3 onwards should use default ratio of 1.0
        np.testing.assert_allclose(
            result["period_3"].values,
            sample_demand_mean.values * 1.0,
            rtol=1e-6,
        )

    def test_combine_clips_extreme_ratios(self, sample_demand_mean):
        """Test that extreme ratios are clipped."""
        combiner = ProfileCombiner(min_profile_ratio=0.1, max_profile_ratio=5.0)

        profiles = {
            0: pd.Series([0.05, 10.0, 2.0], index=sample_demand_mean.index),  # Will be clipped
        }

        result = combiner.combine(sample_demand_mean, profiles)

        # First value clipped to 0.1
        assert result["period_0"].iloc[0] == pytest.approx(1000.0 * 0.1)
        # Second value clipped to 5.0
        assert result["period_0"].iloc[1] == pytest.approx(1100.0 * 5.0)
        # Third value unchanged
        assert result["period_0"].iloc[2] == pytest.approx(1200.0 * 2.0)

    def test_combine_ensures_non_negative(self, combiner):
        """Test that negative loads are clipped to zero."""
        demand_mean = pd.Series([100.0, -50.0, 200.0])  # One negative value
        profiles = {0: pd.Series([1.0, 1.0, 1.0], index=demand_mean.index)}

        result = combiner.combine(demand_mean, profiles)

        # All values should be non-negative
        assert (result.values >= 0).all()

    def test_combine_empty_demand_mean_raises_error(self, combiner):
        """Test that empty demand_mean raises error."""
        demand_mean = pd.Series([], dtype=float)
        profiles = {}

        with pytest.raises(ValueError, match="cannot be empty"):
            combiner.combine(demand_mean, profiles)

    def test_combine_nan_demand_mean_raises_error(self, combiner, sample_demand_mean):
        """Test that NaN in demand_mean raises error."""
        demand_mean = sample_demand_mean.copy()
        demand_mean.iloc[1] = np.nan
        profiles = {}

        with pytest.raises(ValueError, match="NaN values"):
            combiner.combine(demand_mean, profiles)

    def test_combine_infinite_demand_mean_raises_error(self, combiner, sample_demand_mean):
        """Test that infinite values in demand_mean raise error."""
        demand_mean = sample_demand_mean.copy()
        demand_mean.iloc[1] = np.inf
        profiles = {}

        with pytest.raises(ValueError, match="infinite values"):
            combiner.combine(demand_mean, profiles)

    def test_combine_mismatched_index_raises_error(self, combiner, sample_demand_mean):
        """Test that mismatched profile index raises error."""
        profiles = {
            0: pd.Series([1.0, 1.0], index=[0, 1]),  # Wrong index
        }

        with pytest.raises(ValueError, match="mismatched index"):
            combiner.combine(sample_demand_mean, profiles)


class TestProfileCombinerEnergyConservation:
    """Test ProfileCombiner.validate_energy_conservation() method."""

    @pytest.fixture
    def combiner(self):
        """Create combiner instance."""
        return ProfileCombiner()

    def test_validate_perfect_conservation(self, combiner):
        """Test validation with perfect energy conservation."""
        demand_mean = pd.Series([1000.0, 1100.0])

        # Create combined load where each period has exactly demand_mean value
        # So daily sum = demand_mean * 48
        combined_load = pd.DataFrame(
            {f"period_{i}": demand_mean for i in range(48)}
        )

        result = combiner.validate_energy_conservation(demand_mean, combined_load)

        assert result["is_valid"]
        assert result["mean_error"] == pytest.approx(0.0, abs=1e-6)
        assert result["max_error"] == pytest.approx(0.0, abs=1e-6)
        assert result["n_violations"] == 0
        assert result["tolerance"] == 0.1

    def test_validate_within_tolerance(self, combiner):
        """Test validation within tolerance."""
        demand_mean = pd.Series([1000.0, 1100.0])

        # Create load with 5% error (within 10% tolerance)
        combined_load = pd.DataFrame(
            {f"period_{i}": demand_mean * 1.05 for i in range(48)}
        )

        result = combiner.validate_energy_conservation(demand_mean, combined_load)

        assert result["is_valid"]
        assert result["mean_error"] == pytest.approx(0.05, abs=1e-6)
        assert result["n_violations"] == 0

    def test_validate_exceeds_tolerance(self, combiner):
        """Test validation exceeding tolerance."""
        demand_mean = pd.Series([1000.0, 1100.0])

        # Create load with 15% error (exceeds 10% tolerance)
        combined_load = pd.DataFrame(
            {f"period_{i}": demand_mean * 1.15 for i in range(48)}
        )

        result = combiner.validate_energy_conservation(demand_mean, combined_load)

        assert not result["is_valid"]
        assert result["mean_error"] == pytest.approx(0.15, abs=1e-6)
        assert result["n_violations"] == 2  # Both samples exceed tolerance

    def test_validate_custom_tolerance(self, combiner):
        """Test validation with custom tolerance."""
        demand_mean = pd.Series([1000.0])
        combined_load = pd.DataFrame(
            {f"period_{i}": demand_mean * 1.05 for i in range(48)}
        )

        # 5% error should pass with 10% tolerance
        result = combiner.validate_energy_conservation(
            demand_mean, combined_load, tolerance=0.1
        )
        assert result["is_valid"]

        # But fail with 3% tolerance
        result = combiner.validate_energy_conservation(
            demand_mean, combined_load, tolerance=0.03
        )
        assert not result["is_valid"]

    def test_validate_shape_mismatch_raises_error(self, combiner):
        """Test that shape mismatch raises error."""
        demand_mean = pd.Series([1000.0, 1100.0])
        combined_load = pd.DataFrame(
            {f"period_{i}": [1000.0] for i in range(48)}  # Only 1 row
        )

        with pytest.raises(ValueError, match="Shape mismatch"):
            combiner.validate_energy_conservation(demand_mean, combined_load)

    def test_validate_wrong_num_columns_raises_error(self, combiner):
        """Test that wrong number of columns raises error."""
        demand_mean = pd.Series([1000.0, 1100.0])
        combined_load = pd.DataFrame(
            {f"period_{i}": demand_mean for i in range(24)}  # Only 24 columns
        )

        with pytest.raises(ValueError, match="must have 48 columns"):
            combiner.validate_energy_conservation(demand_mean, combined_load)


# Tests for BaseHierarchicalModel


class TestBaseHierarchicalModelInterface:
    """Test BaseHierarchicalModel abstract interface."""

    def test_cannot_instantiate_base_hierarchical_model(self):
        """Test that BaseHierarchicalModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseHierarchicalModel()

    def test_test_model_implements_interface(self):
        """Test that TestHierarchicalModel properly implements interface."""
        model = TestHierarchicalModel()

        assert isinstance(model, BaseHierarchicalModel)
        assert isinstance(model, BaseModel)
        assert model.name == "test_hierarchical"
        assert model.version == "1.0.0"
        assert model.supported_horizons == [0, 1, 2, 3]
        assert model.demand_mean_model_class == MockDemandMeanModel
        assert model.profile_model_class == MockProfileModel


class TestBaseHierarchicalModelInitialization:
    """Test BaseHierarchicalModel initialization."""

    def test_initial_state(self):
        """Test that new model has correct initial state."""
        model = TestHierarchicalModel()

        assert not model.is_fitted()
        assert model.demand_mean_model is None
        assert model.profile_models == {}
        assert isinstance(model.combiner, ProfileCombiner)


class TestBaseHierarchicalModelFitting:
    """Test hierarchical model fitting functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample semi-hourly training data."""
        # 3 days of semi-hourly data (3 * 48 = 144 samples)
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        timestamps = []
        loads = []

        for date in dates:
            for hour in range(24):
                for minute in [0, 30]:
                    timestamp = date + pd.Timedelta(hours=hour, minutes=minute)
                    timestamps.append(timestamp)
                    # Load varies by hour (simple pattern)
                    load = 1000 + hour * 50 + np.random.randn() * 10
                    loads.append(load)

        X = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        y = pd.Series(loads)

        config = {
            "demand_mean_config": {},
            "profile_config": {},
        }

        return X, y, config

    def test_fit_model(self, sample_data):
        """Test fitting hierarchical model."""
        X, y, config = sample_data
        model = TestHierarchicalModel()

        assert not model.is_fitted()

        model.fit(X, y, config)

        assert model.is_fitted()
        assert model.demand_mean_model is not None
        assert model.demand_mean_model.is_fitted()
        assert len(model.profile_models) > 0

    def test_fit_trains_demand_mean_model(self, sample_data):
        """Test that fit trains demand mean model."""
        X, y, config = sample_data
        model = TestHierarchicalModel()

        model.fit(X, y, config)

        assert model.demand_mean_model is not None
        assert isinstance(model.demand_mean_model, MockDemandMeanModel)
        assert model.demand_mean_model.is_fitted()

    def test_fit_trains_profile_models(self, sample_data):
        """Test that fit trains profile models."""
        X, y, config = sample_data
        model = TestHierarchicalModel()

        model.fit(X, y, config)

        # Should have trained profile models for multiple periods
        assert len(model.profile_models) > 0

        # Check that they are the correct type
        for period, profile_model in model.profile_models.items():
            assert isinstance(profile_model, MockProfileModel)
            assert profile_model.is_fitted()
            assert 0 <= period < 48

    def test_fit_stores_metadata(self, sample_data):
        """Test that fit stores training metadata."""
        X, y, config = sample_data
        model = TestHierarchicalModel()

        model.fit(X, y, config)

        assert "trained_at" in model._training_metadata
        assert "training_config" in model._training_metadata
        assert "n_samples" in model._training_metadata
        assert "n_daily_samples" in model._training_metadata
        assert "n_profile_periods" in model._training_metadata

    def test_fit_with_dataframe_target(self, sample_data):
        """Test fitting with DataFrame target."""
        X, y, config = sample_data
        model = TestHierarchicalModel()

        # Convert y to single-column DataFrame
        y_df = y.to_frame(name="load")

        model.fit(X, y_df, config)

        assert model.is_fitted()

    def test_fit_with_multi_column_target_raises_error(self, sample_data):
        """Test that multi-column target raises error."""
        X, y, config = sample_data
        model = TestHierarchicalModel()

        # Create multi-column target
        y_df = pd.DataFrame({"load1": y, "load2": y})

        with pytest.raises(ValueError, match="must be single column"):
            model.fit(X, y_df, config)


class TestBaseHierarchicalModelPrediction:
    """Test hierarchical model prediction functionality."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted hierarchical model."""
        # 3 days of semi-hourly data
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        timestamps = []
        loads = []

        for date in dates:
            for hour in range(24):
                for minute in [0, 30]:
                    timestamp = date + pd.Timedelta(hours=hour, minutes=minute)
                    timestamps.append(timestamp)
                    load = 1000 + hour * 50
                    loads.append(load)

        X = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        y = pd.Series(loads)

        config = {
            "demand_mean_config": {},
            "profile_config": {},
        }

        model = TestHierarchicalModel()
        model.fit(X, y, config)

        return model

    def test_predict_returns_48_periods(self, fitted_model):
        """Test that predict returns 48 semi-hourly periods."""
        # 1 day of test data
        timestamps = pd.date_range("2024-01-04", periods=48, freq="30min")

        X_test = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        predictions = fitted_model.predict(X_test)

        # Should have 48 columns (one per semi-hourly period)
        assert predictions.shape[1] == 48
        assert list(predictions.columns) == [f"period_{i}" for i in range(48)]

    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error if model not fitted."""
        model = TestHierarchicalModel()
        X = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=48, freq="30min"), "hour": range(48)})

        with pytest.raises(ValueError, match="has not been fitted"):
            model.predict(X)

    def test_predict_with_horizons(self, fitted_model):
        """Test prediction with specific horizons."""
        timestamps = pd.date_range("2024-01-04", periods=48, freq="30min")

        X_test = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        predictions = fitted_model.predict(X_test, horizons=[0, 1])

        # Should still return 48 periods (horizons are used internally)
        assert predictions.shape[1] == 48


class TestBaseHierarchicalModelFeatureImportance:
    """Test feature importance aggregation."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        timestamps = []
        loads = []

        for day in range(3):
            for hour in range(24):
                for minute in [0, 30]:
                    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(
                        days=day, hours=hour, minutes=minute
                    )
                    timestamps.append(timestamp)
                    load = 1000 + hour * 50
                    loads.append(load)

        X = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        y = pd.Series(loads)

        model = TestHierarchicalModel()
        model.fit(X, y, {"demand_mean_config": {}, "profile_config": {}})

        return model

    def test_get_feature_importance(self, fitted_model):
        """Test getting aggregated feature importance."""
        importance = fitted_model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) > 0
        # Should contain 'hour' feature
        assert "hour" in importance
        # Importance should be normalized
        assert abs(sum(importance.values()) - 1.0) < 1e-6

    def test_get_feature_importance_before_fit_raises_error(self):
        """Test that get_feature_importance raises error if not fitted."""
        model = TestHierarchicalModel()

        with pytest.raises(ValueError, match="has not been fitted"):
            model.get_feature_importance()


class TestBaseHierarchicalModelDecomposition:
    """Test get_decomposition() helper method."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model."""
        timestamps = []
        loads = []

        for day in range(3):
            for hour in range(24):
                for minute in [0, 30]:
                    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(
                        days=day, hours=hour, minutes=minute
                    )
                    timestamps.append(timestamp)
                    load = 1000 + hour * 50
                    loads.append(load)

        X = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        y = pd.Series(loads)

        model = TestHierarchicalModel()
        model.fit(X, y, {"demand_mean_config": {}, "profile_config": {}})

        return model

    def test_get_decomposition(self, fitted_model):
        """Test getting prediction decomposition."""
        timestamps = pd.date_range("2024-01-04", periods=48, freq="30min")

        X_test = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        decomp = fitted_model.get_decomposition(X_test)

        assert "demand_mean" in decomp
        assert "profiles" in decomp
        assert "combined_load" in decomp

        # Check shapes
        assert isinstance(decomp["demand_mean"], pd.Series)
        assert isinstance(decomp["profiles"], dict)
        assert isinstance(decomp["combined_load"], pd.DataFrame)
        assert decomp["combined_load"].shape[1] == 48

    def test_get_decomposition_before_fit_raises_error(self):
        """Test that get_decomposition raises error if not fitted."""
        model = TestHierarchicalModel()
        X = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=48, freq="30min"), "hour": range(48)})

        with pytest.raises(ValueError, match="has not been fitted"):
            model.get_decomposition(X)


class TestBaseHierarchicalModelSerialization:
    """Test model saving and loading."""

    def test_save_model(self, tmp_path):
        """Test saving a fitted hierarchical model."""
        # Create and fit model
        timestamps = []
        loads = []

        for day in range(2):
            for hour in range(24):
                for minute in [0, 30]:
                    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(
                        days=day, hours=hour, minutes=minute
                    )
                    timestamps.append(timestamp)
                    load = 1000 + hour * 50
                    loads.append(load)

        X = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        y = pd.Series(loads)

        model = TestHierarchicalModel()
        model.fit(X, y, {"demand_mean_config": {}, "profile_config": {}})

        # Save
        save_path = tmp_path / "hierarchical_model.joblib"
        model.save(save_path)

        assert save_path.exists()

    def test_save_unfitted_model_raises_error(self, tmp_path):
        """Test that saving unfitted model raises error."""
        model = TestHierarchicalModel()
        save_path = tmp_path / "model.joblib"

        with pytest.raises(ValueError, match="has not been fitted"):
            model.save(save_path)

    def test_load_model(self, tmp_path):
        """Test loading a saved hierarchical model."""
        # Create and fit model
        timestamps = []
        loads = []

        for day in range(2):
            for hour in range(24):
                for minute in [0, 30]:
                    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(
                        days=day, hours=hour, minutes=minute
                    )
                    timestamps.append(timestamp)
                    load = 1000 + hour * 50
                    loads.append(load)

        X = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        y = pd.Series(loads)

        original_model = TestHierarchicalModel()
        original_model.fit(X, y, {"demand_mean_config": {}, "profile_config": {}})

        # Save
        save_path = tmp_path / "model.joblib"
        original_model.save(save_path)

        # Load
        loaded_model = BaseHierarchicalModel.load(save_path)

        assert isinstance(loaded_model, TestHierarchicalModel)
        assert loaded_model.is_fitted()
        assert loaded_model.name == "test_hierarchical"
        assert len(loaded_model.profile_models) == len(original_model.profile_models)

    def test_save_load_roundtrip(self, tmp_path):
        """Test that save/load preserves model state."""
        # Create and fit model
        timestamps = []
        loads = []

        for day in range(2):
            for hour in range(24):
                for minute in [0, 30]:
                    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(
                        days=day, hours=hour, minutes=minute
                    )
                    timestamps.append(timestamp)
                    load = 1000 + hour * 50
                    loads.append(load)

        X = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        y = pd.Series(loads)

        original = TestHierarchicalModel()
        original.fit(X, y, {"demand_mean_config": {}, "profile_config": {}})

        # Save
        save_path = tmp_path / "model.joblib"
        original.save(save_path)

        # Load
        loaded = BaseHierarchicalModel.load(save_path)

        # Test prediction works - need 2 days of data to get 2 rows of predictions
        X_test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-03", periods=96, freq="30min"),  # 2 days
                "hour": [ts.hour for ts in pd.date_range("2024-01-03", periods=96, freq="30min")],
            }
        )

        predictions = loaded.predict(X_test)

        # Should have 2 daily predictions, each with 48 periods
        assert predictions.shape == (2, 48)


class TestBaseHierarchicalModelRepresentation:
    """Test string representations."""

    def test_repr_unfitted(self):
        """Test __repr__ for unfitted model."""
        model = TestHierarchicalModel()
        r = repr(model)

        assert "TestHierarchicalModel" in r
        assert "test_hierarchical" in r
        assert "fitted=False" in r
        assert "n_profile_models=0" in r

    def test_repr_fitted(self):
        """Test __repr__ for fitted model."""
        timestamps = []
        loads = []

        for day in range(2):
            for hour in range(24):
                for minute in [0, 30]:
                    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(
                        days=day, hours=hour, minutes=minute
                    )
                    timestamps.append(timestamp)
                    load = 1000 + hour * 50
                    loads.append(load)

        X = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hour": [ts.hour for ts in timestamps],
            }
        )

        y = pd.Series(loads)

        model = TestHierarchicalModel()
        model.fit(X, y, {"demand_mean_config": {}, "profile_config": {}})

        r = repr(model)

        assert "fitted=True" in r
        assert "n_profile_models=" in r
