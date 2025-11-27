"""Tests for hierarchical model orchestration."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.base.model import BaseModel
from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.models.hierarchical.orchestration import HierarchicalTrainer


# Mock Models for Testing


class MockDemandMeanModel(BaseModel):
    """Mock model for demand mean forecasting."""

    def __init__(self):
        super().__init__()
        self._multiplier = 1000.0

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
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Mock fit implementation."""
        self._feature_names = list(X.columns) if not X.empty else []
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_config": config,
        }
        self._multiplier = y.mean() if len(y) > 0 else 1000.0
        self._is_fitted = True

    def predict(
        self,
        X: pd.DataFrame,  # noqa: N803
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Mock predict implementation."""
        self._check_is_fitted()

        if horizons is None:
            horizons = self.supported_horizons

        return pd.DataFrame(
            index=X.index,
            columns=[f"h{h}" for h in horizons],
            data=self._multiplier,
        )

    def get_feature_importance(self) -> dict[str, float]:
        """Mock feature importance."""
        self._check_is_fitted()
        return {}


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
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Mock fit implementation."""
        self._feature_names = list(X.columns) if not X.empty else []
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_config": config,
        }
        self._mean_ratio = y.mean() if len(y) > 0 else 1.0
        self._is_fitted = True

    def predict(
        self,
        X: pd.DataFrame,  # noqa: N803
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Mock predict implementation."""
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
        return {}


class TestHierarchicalModel(BaseHierarchicalModel):
    """Concrete implementation for testing."""

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
        """Prepare daily aggregated data."""
        if not isinstance(df.index, pd.DatetimeIndex):
            msg = "DataFrame must have DatetimeIndex"
            raise ValueError(msg)

        has_load = "load" in df.columns

        if has_load:
            daily_y = df["load"].groupby(df.index.date).mean()
            daily_y.index = pd.to_datetime(daily_y.index)
            daily_X = pd.DataFrame(index=daily_y.index)  # noqa: N806
            return daily_X, daily_y

        # Prediction
        dates = df.index.date
        unique_dates = pd.Series(dates).unique()
        daily_index = pd.to_datetime(unique_dates)
        daily_X = pd.DataFrame(index=daily_index)  # noqa: N806
        daily_y = pd.Series(index=daily_index, dtype=float)
        return daily_X, daily_y

    def _prepare_profile_data(
        self,
        df: pd.DataFrame,
        demand_mean: pd.Series,
        period: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare profile data for specific period."""
        if not 0 <= period < 48:
            msg = f"Period must be in range [0, 47], got {period}"
            raise ValueError(msg)

        if not isinstance(df.index, pd.DatetimeIndex):
            msg = "DataFrame must have DatetimeIndex"
            raise ValueError(msg)

        # Extract period from timestamps
        periods = df.index.hour * 2 + (df.index.minute // 30)
        mask = periods == period
        df_period = df[mask].copy()

        if len(df_period) == 0:
            empty_index = pd.DatetimeIndex([])
            return pd.DataFrame(index=empty_index), pd.Series(dtype=float, index=empty_index)

        has_load = "load" in df.columns

        if has_load:
            # Training
            df_period["date"] = df_period.index.date
            profile_data = []

            period_hour = period // 2
            period_minute = (period % 2) * 30

            for date in df_period["date"].unique():
                date_mask = df_period["date"] == date
                load_value = df_period.loc[date_mask, "load"].iloc[0] if date_mask.sum() > 0 else np.nan

                date_ts = pd.Timestamp(date)
                if date_ts in demand_mean.index:
                    demand_mean_value = demand_mean.loc[date_ts]
                    profile_ratio = load_value / (demand_mean_value + 1e-8)
                    profile_ratio = np.clip(profile_ratio, 0.01, 10.0)

                    timestamp_with_time = date_ts.replace(hour=period_hour, minute=period_minute)
                    profile_data.append(
                        {
                            "timestamp": timestamp_with_time,
                            "profile_ratio": profile_ratio,
                        }
                    )

            if len(profile_data) == 0:
                empty_index = pd.DatetimeIndex([])
                return pd.DataFrame(index=empty_index), pd.Series(dtype=float, index=empty_index)

            profile_df = pd.DataFrame(profile_data)
            profile_df = profile_df.set_index("timestamp")
            X_period = pd.DataFrame(index=profile_df.index)  # noqa: N806
            y_period = profile_df["profile_ratio"]

            return X_period, y_period

        # Prediction
        period_hour = period // 2
        period_minute = (period % 2) * 30

        unique_dates = sorted(set(df_period.index.date))
        timestamps = []
        for date in unique_dates:
            date_ts = pd.Timestamp(date)
            timestamp_with_time = date_ts.replace(hour=period_hour, minute=period_minute)
            timestamps.append(timestamp_with_time)

        timestamp_index = pd.DatetimeIndex(timestamps)
        X_period = pd.DataFrame(index=timestamp_index)  # noqa: N806
        y_period = pd.Series(index=timestamp_index, dtype=float)

        return X_period, y_period


# Fixtures


@pytest.fixture
def sample_data():
    """Create sample semi-hourly data for testing."""
    # 7 days of semi-hourly data
    dates = pd.date_range("2023-01-01", periods=7 * 48, freq="30min")

    # Simulate load with daily pattern
    hours = dates.hour + dates.minute / 60
    base_load = 1000 + 200 * np.sin((hours - 6) * np.pi / 12)
    noise = np.random.RandomState(42).normal(0, 50, len(dates))
    load = base_load + noise

    X = pd.DataFrame(index=dates)  # noqa: N806
    y = pd.Series(load, index=dates)

    return X, y


@pytest.fixture
def trainer():
    """Create HierarchicalTrainer instance."""
    return HierarchicalTrainer(n_jobs=2, verbose=False)


# Tests


class TestHierarchicalTrainerInit:
    """Tests for HierarchicalTrainer initialization."""

    def test_init_default(self):
        """Test trainer initialization with default parameters."""
        trainer = HierarchicalTrainer()

        assert trainer.n_jobs == 4
        assert trainer.verbose is True

    def test_init_custom(self):
        """Test trainer initialization with custom parameters."""
        trainer = HierarchicalTrainer(n_jobs=8, verbose=False)

        assert trainer.n_jobs == 8
        assert trainer.verbose is False

    def test_init_invalid_n_jobs(self):
        """Test trainer initialization with invalid n_jobs."""
        with pytest.raises(ValueError, match="n_jobs must be"):
            HierarchicalTrainer(n_jobs=0)

        with pytest.raises(ValueError, match="n_jobs must be"):
            HierarchicalTrainer(n_jobs=-2)

    def test_repr(self):
        """Test string representation."""
        trainer = HierarchicalTrainer(n_jobs=4, verbose=True)
        repr_str = repr(trainer)

        assert "HierarchicalTrainer" in repr_str
        assert "n_jobs=4" in repr_str
        assert "verbose=True" in repr_str


class TestTrainHierarchicalModel:
    """Tests for train_hierarchical_model method."""

    def test_train_basic(self, trainer, sample_data):
        """Test basic hierarchical model training."""
        X, y = sample_data  # noqa: N806
        model = TestHierarchicalModel()

        config = {
            "demand_mean_config": {},
            "profile_config": {},
        }

        trained_model = trainer.train_hierarchical_model(model, X, y, config)

        # Check model is fitted
        assert trained_model._is_fitted
        assert trained_model.demand_mean_model is not None
        assert len(trained_model.profile_models) > 0

    def test_train_with_dataframe_y(self, trainer, sample_data):
        """Test training with DataFrame y."""
        X, y = sample_data  # noqa: N806
        y_df = pd.DataFrame({"load": y})
        model = TestHierarchicalModel()

        config = {
            "demand_mean_config": {},
            "profile_config": {},
        }

        trained_model = trainer.train_hierarchical_model(model, X, y_df, config)

        assert trained_model._is_fitted

    def test_train_invalid_y_shape(self, trainer, sample_data):
        """Test training with invalid y shape."""
        X, y = sample_data  # noqa: N806
        y_df = pd.DataFrame({"load1": y, "load2": y})  # Two columns
        model = TestHierarchicalModel()

        config = {}

        with pytest.raises(ValueError, match="y must be single column"):
            trainer.train_hierarchical_model(model, X, y_df, config)

    def test_train_empty_data(self, trainer):
        """Test training with empty data."""
        X = pd.DataFrame()  # noqa: N806
        y = pd.Series(dtype=float)
        model = TestHierarchicalModel()

        config = {}

        with pytest.raises(ValueError, match="Training data cannot be empty"):
            trainer.train_hierarchical_model(model, X, y, config)

    def test_train_mismatched_lengths(self, trainer, sample_data):
        """Test training with mismatched X and y lengths."""
        X, y = sample_data  # noqa: N806
        y_short = y[:100]
        model = TestHierarchicalModel()

        config = {}

        with pytest.raises(ValueError, match="X and y must have same length"):
            trainer.train_hierarchical_model(model, X, y_short, config)


class TestTrainProfilesParallel:
    """Tests for train_profiles_parallel method."""

    def test_parallel_training_basic(self, trainer):
        """Test basic parallel profile training."""
        model = TestHierarchicalModel()
        model.demand_mean_model = MockDemandMeanModel()
        model.demand_mean_model._is_fitted = True

        # Create mock profile data for a few periods
        dates = pd.date_range("2023-01-01", periods=7, freq="D")
        profile_data = {}

        for period in [0, 12, 24, 36]:  # Test 4 periods
            X_period = pd.DataFrame(index=dates)  # noqa: N806
            y_period = pd.Series(np.random.uniform(0.8, 1.2, len(dates)), index=dates)
            profile_data[period] = (X_period, y_period)

        config = {}

        profile_models = trainer.train_profiles_parallel(model, profile_data, config)

        # Check that models were trained
        assert len(profile_models) > 0
        assert all(period in [0, 12, 24, 36] for period in profile_models.keys())

    def test_parallel_training_empty_data(self, trainer):
        """Test parallel training with empty profile data."""
        model = TestHierarchicalModel()
        profile_data = {}
        config = {}

        with pytest.raises(ValueError, match="profile_data cannot be empty"):
            trainer.train_profiles_parallel(model, profile_data, config)

    def test_parallel_training_with_failures(self, trainer):
        """Test parallel training handles individual failures gracefully."""
        model = TestHierarchicalModel()

        # Create mock profile data
        dates = pd.date_range("2023-01-01", periods=7, freq="D")
        profile_data = {}

        for period in range(5):
            X_period = pd.DataFrame(index=dates)  # noqa: N806
            # Create some invalid data for period 2
            if period == 2:
                y_period = pd.Series([], dtype=float)  # Empty series
            else:
                y_period = pd.Series(np.random.uniform(0.8, 1.2, len(dates)), index=dates)
            profile_data[period] = (X_period, y_period)

        config = {}

        # Should not raise, but should train fewer models
        profile_models = trainer.train_profiles_parallel(model, profile_data, config)

        # At least some models should be trained
        assert len(profile_models) >= 3  # At least the valid ones


class TestTrainSingleProfileModel:
    """Tests for _train_single_profile_model static method."""

    def test_train_single_success(self):
        """Test successful single profile model training."""
        dates = pd.date_range("2023-01-01", periods=7, freq="D")
        X = pd.DataFrame(index=dates)  # noqa: N806
        y = pd.Series(np.random.uniform(0.8, 1.2, len(dates)), index=dates)

        config = {}

        model = HierarchicalTrainer._train_single_profile_model(
            MockProfileModel,
            12,  # period
            X,
            y,
            config,
        )

        assert model is not None
        assert model._is_fitted

    def test_train_single_failure(self):
        """Test single profile model training failure handling."""
        # Use a mock class that raises an error on fit
        class FailingModel(MockProfileModel):
            def fit(self, X, y, config):  # noqa: N803
                raise ValueError("Simulated training failure")

        dates = pd.date_range("2023-01-01", periods=7, freq="D")
        X = pd.DataFrame(index=dates)  # noqa: N806
        y = pd.Series(np.random.uniform(0.8, 1.2, len(dates)), index=dates)

        config = {}

        model = HierarchicalTrainer._train_single_profile_model(
            FailingModel,
            12,
            X,
            y,
            config,
        )

        # Should return None on failure
        assert model is None


class TestValidateTrainingResults:
    """Tests for _validate_training_results method."""

    def test_validate_success(self, trainer):
        """Test validation with successfully trained model."""
        model = TestHierarchicalModel()
        model.demand_mean_model = MockDemandMeanModel()
        model.demand_mean_model._is_fitted = True

        # Create mock profile models
        for period in range(48):
            profile_model = MockProfileModel()
            profile_model._is_fitted = True
            model.profile_models[period] = profile_model

        # Should not raise
        trainer._validate_training_results(model)

    def test_validate_no_demand_mean_model(self, trainer):
        """Test validation failure with no demand mean model."""
        model = TestHierarchicalModel()
        model.demand_mean_model = None
        model.profile_models = {}

        with pytest.raises(RuntimeError, match="Demand mean model was not trained"):
            trainer._validate_training_results(model)

    def test_validate_demand_mean_not_fitted(self, trainer):
        """Test validation failure with unfitted demand mean model."""
        model = TestHierarchicalModel()
        model.demand_mean_model = MockDemandMeanModel()
        model.demand_mean_model._is_fitted = False
        model.profile_models = {}

        with pytest.raises(RuntimeError, match="Demand mean model is not fitted"):
            trainer._validate_training_results(model)

    def test_validate_no_profile_models(self, trainer):
        """Test validation failure with no profile models."""
        model = TestHierarchicalModel()
        model.demand_mean_model = MockDemandMeanModel()
        model.demand_mean_model._is_fitted = True
        model.profile_models = {}

        with pytest.raises(RuntimeError, match="No profile models were trained"):
            trainer._validate_training_results(model)

    def test_validate_few_profile_models(self, trainer):
        """Test validation warning with few profile models."""
        model = TestHierarchicalModel()
        model.demand_mean_model = MockDemandMeanModel()
        model.demand_mean_model._is_fitted = True

        # Only 10 profile models (< 24 threshold)
        for period in range(10):
            profile_model = MockProfileModel()
            profile_model._is_fitted = True
            model.profile_models[period] = profile_model

        # Should not raise, but will log warning
        trainer._validate_training_results(model)
