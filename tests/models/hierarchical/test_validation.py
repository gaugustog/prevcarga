"""Tests for hierarchical model validation."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.base.model import BaseModel
from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.models.hierarchical.profile_combiner import ProfileCombiner
from src.models.hierarchical.validation import HierarchicalValidator


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

        # Extract dates from timestamps if needed
        if isinstance(X.index, pd.DatetimeIndex):
            # Convert timestamps to dates for alignment with demand_mean
            dates = pd.to_datetime(X.index.date)
            # Remove duplicates while preserving order
            unique_dates = dates[~dates.duplicated()]
            index = unique_dates
        else:
            index = X.index

        return pd.DataFrame(
            index=index,
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
    # 14 days of semi-hourly data (2 weeks for seasonal consistency)
    dates = pd.date_range("2023-01-01", periods=14 * 48, freq="30min")

    # Simulate load with daily pattern
    hours = dates.hour + dates.minute / 60
    base_load = 1000 + 200 * np.sin((hours - 6) * np.pi / 12)
    noise = np.random.RandomState(42).normal(0, 50, len(dates))
    load = base_load + noise

    X = pd.DataFrame(index=dates)  # noqa: N806
    y = pd.Series(load, index=dates)

    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create a trained hierarchical model."""
    X, y = sample_data  # noqa: N806
    model = TestHierarchicalModel()

    config = {
        "demand_mean_config": {},
        "profile_config": {},
    }

    model.fit(X, y, config)

    return model


@pytest.fixture
def validator():
    """Create HierarchicalValidator instance."""
    return HierarchicalValidator()


# Tests


class TestHierarchicalValidatorInit:
    """Tests for HierarchicalValidator initialization."""

    def test_init(self):
        """Test validator initialization."""
        validator = HierarchicalValidator()

        assert validator.validation_results == {}

    def test_repr(self):
        """Test string representation."""
        validator = HierarchicalValidator()
        repr_str = repr(validator)

        assert "HierarchicalValidator" in repr_str


class TestValidateModel:
    """Tests for validate_model method."""

    def test_validate_basic(self, validator, trained_model, sample_data):
        """Test basic model validation."""
        X, y = sample_data  # noqa: N806

        results = validator.validate_model(trained_model, X, y)

        # Check results structure
        assert "model_name" in results
        assert "timestamp" in results
        assert "energy_conservation" in results
        assert "profile_bounds" in results
        assert "seasonal_consistency" in results
        assert "component_analysis" in results
        assert "passed" in results

        # Model name should match
        assert results["model_name"] == "test_hierarchical"

        # Overall status should be boolean
        assert isinstance(results["passed"], bool)

    def test_validate_with_dataframe_y(self, validator, trained_model, sample_data):
        """Test validation with DataFrame y."""
        X, y = sample_data  # noqa: N806
        y_df = pd.DataFrame({"load": y})

        results = validator.validate_model(trained_model, X, y_df)

        assert "passed" in results
        assert isinstance(results["passed"], bool)

    def test_validate_invalid_y_shape(self, validator, trained_model, sample_data):
        """Test validation with invalid y shape."""
        X, y = sample_data  # noqa: N806
        y_df = pd.DataFrame({"load1": y, "load2": y})  # Two columns

        with pytest.raises(ValueError, match="y must be single column"):
            validator.validate_model(trained_model, X, y_df)

    def test_validate_unfitted_model(self, validator, sample_data):
        """Test validation of unfitted model."""
        X, y = sample_data  # noqa: N806
        model = TestHierarchicalModel()

        with pytest.raises(ValueError, match="Model must be fitted"):
            validator.validate_model(model, X, y)

    def test_validate_with_config(self, validator, trained_model, sample_data):
        """Test validation with custom config."""
        X, y = sample_data  # noqa: N806

        config = {
            "energy_tolerance": 0.10,  # 10% tolerance
            "profile_min": 0.05,
            "profile_max": 10.0,
            "seasonal_std_threshold": 1.0,
        }

        results = validator.validate_model(trained_model, X, y, config)

        # Check threshold if test passed
        if "error" not in results["energy_conservation"]:
            assert results["energy_conservation"]["threshold"] == 0.10


class TestEnergyConservation:
    """Tests for _test_energy_conservation method."""

    def test_energy_conservation_pass(self, validator, trained_model, sample_data):
        """Test energy conservation with passing model."""
        X, _ = sample_data  # noqa: N806

        result = validator._test_energy_conservation(trained_model, X, threshold=0.10)

        assert "passed" in result
        assert "mean_error" in result
        assert "max_error" in result
        assert "n_violations" in result
        assert "threshold" in result

        # Errors should be numeric
        assert isinstance(result["mean_error"], float)
        assert isinstance(result["max_error"], float)

    def test_energy_conservation_threshold(self, validator, trained_model, sample_data):
        """Test energy conservation with different thresholds."""
        X, _ = sample_data  # noqa: N806

        # Strict threshold
        result_strict = validator._test_energy_conservation(trained_model, X, threshold=0.001)

        # Loose threshold
        result_loose = validator._test_energy_conservation(trained_model, X, threshold=0.50)

        # Loose threshold should have fewer violations
        assert result_loose["n_violations"] <= result_strict["n_violations"]


class TestProfileBounds:
    """Tests for _test_profile_bounds method."""

    def test_profile_bounds_pass(self, validator, trained_model, sample_data):
        """Test profile bounds with passing model."""
        X, _ = sample_data  # noqa: N806

        result = validator._test_profile_bounds(
            trained_model,
            X,
            min_ratio=0.1,
            max_ratio=5.0,
        )

        assert "passed" in result
        assert "violation_rate" in result
        assert "lower_violations" in result
        assert "upper_violations" in result
        assert "profile_min" in result
        assert "profile_max" in result

        # Violations should be numeric
        assert isinstance(result["violation_rate"], float)
        assert isinstance(result["lower_violations"], int)
        assert isinstance(result["upper_violations"], int)

    def test_profile_bounds_custom_range(self, validator, trained_model, sample_data):
        """Test profile bounds with custom range."""
        X, _ = sample_data  # noqa: N806

        # Narrow range
        result_narrow = validator._test_profile_bounds(
            trained_model,
            X,
            min_ratio=0.5,
            max_ratio=2.0,
        )

        # Wide range
        result_wide = validator._test_profile_bounds(
            trained_model,
            X,
            min_ratio=0.01,
            max_ratio=100.0,
        )

        # Narrow range should have more violations
        assert result_narrow["violation_rate"] >= result_wide["violation_rate"]


class TestSeasonalConsistency:
    """Tests for _test_seasonal_consistency method."""

    def test_seasonal_consistency_pass(self, validator, trained_model, sample_data):
        """Test seasonal consistency with passing model."""
        X, _ = sample_data  # noqa: N806

        result = validator._test_seasonal_consistency(
            trained_model,
            X,
            threshold=0.5,
        )

        assert "passed" in result
        assert "mean_std" in result
        assert "max_std" in result
        assert "unstable_periods" in result
        assert "threshold" in result

        # Stats should be numeric
        assert isinstance(result["mean_std"], float)
        assert isinstance(result["max_std"], float)
        assert isinstance(result["unstable_periods"], int)

    def test_seasonal_consistency_no_datetime_index(self, validator, trained_model):
        """Test seasonal consistency without DatetimeIndex."""
        X = pd.DataFrame({"feat1": [1, 2, 3]})  # noqa: N806

        with pytest.raises(ValueError, match="must have DatetimeIndex"):
            validator._test_seasonal_consistency(trained_model, X)


class TestAnalyzeComponents:
    """Tests for analyze_components method."""

    def test_analyze_components_basic(self, validator, trained_model, sample_data):
        """Test basic component analysis."""
        X, _ = sample_data  # noqa: N806

        analysis = validator.analyze_components(trained_model, X)

        assert "demand_mean_stats" in analysis
        assert "profile_stats" in analysis
        assert "combined_stats" in analysis
        assert "n_profile_models" in analysis
        assert "n_profiles_predicted" in analysis

    def test_analyze_components_demand_mean_stats(self, validator, trained_model, sample_data):
        """Test demand mean statistics in analysis."""
        X, _ = sample_data  # noqa: N806

        analysis = validator.analyze_components(trained_model, X)

        dm_stats = analysis["demand_mean_stats"]
        assert "mean" in dm_stats
        assert "std" in dm_stats
        assert "min" in dm_stats
        assert "max" in dm_stats
        assert "n_samples" in dm_stats

        # Stats should be reasonable
        assert dm_stats["mean"] > 0
        assert dm_stats["std"] >= 0
        assert dm_stats["min"] <= dm_stats["max"]

    def test_analyze_components_profile_stats(self, validator, trained_model, sample_data):
        """Test profile statistics in analysis."""
        X, _ = sample_data  # noqa: N806

        analysis = validator.analyze_components(trained_model, X)

        profile_stats = analysis["profile_stats"]

        # Should have stats for multiple periods
        assert len(profile_stats) > 0

        # Each period should have stats
        for period, stats in profile_stats.items():
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats


class TestGenerateValidationReport:
    """Tests for generate_validation_report method."""

    def test_generate_report_basic(self, validator, trained_model, sample_data):
        """Test basic report generation."""
        X, y = sample_data  # noqa: N806

        results = validator.validate_model(trained_model, X, y)
        report = validator.generate_validation_report(results)

        # Report should be a string
        assert isinstance(report, str)

        # Report should contain key sections
        assert "VALIDATION REPORT" in report
        assert "test_hierarchical" in report
        assert "Energy Conservation" in report
        assert "Profile Bounds" in report
        assert "Seasonal Consistency" in report

    def test_generate_report_with_file(self, validator, trained_model, sample_data, tmp_path):
        """Test report generation with file output."""
        X, y = sample_data  # noqa: N806

        results = validator.validate_model(trained_model, X, y)

        output_file = tmp_path / "validation_report.txt"
        report = validator.generate_validation_report(results, output_path=output_file)

        # File should be created
        assert output_file.exists()

        # File content should match returned report
        with open(output_file) as f:
            file_content = f.read()

        assert file_content == report

    def test_generate_report_no_results(self, validator):
        """Test report generation with no validation results."""
        with pytest.raises(ValueError, match="No validation results available"):
            validator.generate_validation_report()

    def test_generate_report_uses_stored_results(self, validator, trained_model, sample_data):
        """Test report generation uses stored results."""
        X, y = sample_data  # noqa: N806

        # Run validation
        results = validator.validate_model(trained_model, X, y)

        # Generate report without passing results
        report = validator.generate_validation_report()

        # Report should use stored results
        assert "test_hierarchical" in report
        assert "VALIDATION REPORT" in report


class TestValidationReportFormatting:
    """Tests for validation report formatting."""

    def test_report_contains_all_sections(self, validator, trained_model, sample_data):
        """Test that report contains all expected sections."""
        X, y = sample_data  # noqa: N806

        results = validator.validate_model(trained_model, X, y)
        report = validator.generate_validation_report(results)

        # Check for all main sections
        assert "Test 1: Energy Conservation" in report
        assert "Test 2: Profile Bounds" in report
        assert "Test 3: Seasonal Consistency" in report
        assert "Component Analysis" in report

    def test_report_shows_pass_fail_status(self, validator, trained_model, sample_data):
        """Test that report shows pass/fail status."""
        X, y = sample_data  # noqa: N806

        results = validator.validate_model(trained_model, X, y)
        report = validator.generate_validation_report(results)

        # Should contain status indicators
        assert "Status:" in report
        assert ("PASSED" in report or "FAILED" in report)

    def test_report_shows_metrics(self, validator, trained_model, sample_data):
        """Test that report shows key metrics."""
        X, y = sample_data  # noqa: N806

        results = validator.validate_model(trained_model, X, y)
        report = validator.generate_validation_report(results)

        # Should contain key metrics
        assert "Mean Error:" in report
        assert "Violation Rate:" in report
        assert "Mean Std:" in report
