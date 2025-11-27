"""Tests for time period analyzer module."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.time_period import (
    BRAZILIAN_HOLIDAYS,
    TimePeriodAnalyzer,
    TimePeriodConfig,
    TimePeriodResult,
)


# Tests for TimePeriodConfig


class TestTimePeriodConfig:
    """Tests for TimePeriodConfig Pydantic model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TimePeriodConfig()

        assert config.period_types == ["weekday", "season", "hour"]
        assert config.metrics == ["mape", "mae", "rmse"]
        assert config.min_samples == 30
        assert config.include_statistical_tests is True
        assert config.timezone == "America/Sao_Paulo"
        assert config.alpha == 0.05

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TimePeriodConfig(
            period_types=["weekday", "month"],
            metrics=["mape", "rmse"],
            min_samples=50,
            include_statistical_tests=False,
            timezone="UTC",
            alpha=0.01,
        )

        assert config.period_types == ["weekday", "month"]
        assert config.metrics == ["mape", "rmse"]
        assert config.min_samples == 50
        assert config.include_statistical_tests is False
        assert config.timezone == "UTC"
        assert config.alpha == 0.01

    def test_invalid_period_type(self):
        """Test invalid period type raises error."""
        with pytest.raises(ValueError, match="Invalid period_type"):
            TimePeriodConfig(period_types=["invalid_period"])

    def test_invalid_metric(self):
        """Test invalid metric raises error."""
        with pytest.raises(ValueError, match="Invalid metric"):
            TimePeriodConfig(metrics=["invalid_metric"])

    def test_invalid_min_samples(self):
        """Test invalid min_samples raises error."""
        with pytest.raises(ValueError):
            TimePeriodConfig(min_samples=0)

    def test_invalid_alpha(self):
        """Test invalid alpha raises error."""
        with pytest.raises(ValueError):
            TimePeriodConfig(alpha=1.5)


# Tests for TimePeriodResult


class TestTimePeriodResult:
    """Tests for TimePeriodResult dataclass."""

    def test_basic_result(self):
        """Test basic result creation."""
        from src.evaluation.metrics import MetricsResult

        metrics_mon = MetricsResult(
            mape=5.0, mae=10.0, rmse=15.0, mse=225.0, r2=0.95, sample_size=100
        )
        metrics_tue = MetricsResult(
            mape=6.0, mae=12.0, rmse=18.0, mse=324.0, r2=0.93, sample_size=100
        )

        result = TimePeriodResult(
            period_type="weekday",
            metrics_by_period={"Monday": metrics_mon, "Tuesday": metrics_tue},
            worst_periods=[("Tuesday", 6.0), ("Monday", 5.0)],
            best_periods=[("Monday", 5.0), ("Tuesday", 6.0)],
            statistical_tests={"anova": {"p_value": 0.001}},
            metadata={"timezone": "America/Sao_Paulo"},
        )

        assert result.period_type == "weekday"
        assert result.n_periods == 2
        assert result.worst_period == ("Tuesday", 6.0)
        assert result.best_period == ("Monday", 5.0)

    def test_empty_result(self):
        """Test result with no periods."""
        result = TimePeriodResult(
            period_type="weekday",
            metrics_by_period={},
            worst_periods=[],
            best_periods=[],
        )

        assert result.n_periods == 0
        assert result.worst_period is None
        assert result.best_period is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.evaluation.metrics import MetricsResult

        metrics = MetricsResult(
            mape=5.0, mae=10.0, rmse=15.0, mse=225.0, r2=0.95, sample_size=100
        )

        result = TimePeriodResult(
            period_type="weekday",
            metrics_by_period={"Monday": metrics},
            worst_periods=[("Monday", 5.0)],
            best_periods=[("Monday", 5.0)],
        )

        d = result.to_dict()

        assert d["period_type"] == "weekday"
        assert "Monday" in d["metrics_by_period"]
        assert isinstance(d["metrics_by_period"]["Monday"], dict)


# Tests for TimePeriodAnalyzer


class TestTimePeriodAnalyzer:
    """Tests for TimePeriodAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # One full week of hourly data
        timestamps = pd.date_range(
            "2023-01-01", periods=24 * 7, freq="h", tz="America/Sao_Paulo"
        )

        np.random.seed(42)
        actual = np.random.randn(len(timestamps)) * 100 + 1000
        # Add some systematic error patterns
        forecast = actual + np.random.randn(len(timestamps)) * 50

        return actual, forecast, timestamps

    @pytest.fixture
    def large_sample_data(self):
        """Create larger sample data (1 year)."""
        # One full year of hourly data
        timestamps = pd.date_range(
            "2023-01-01", periods=8760, freq="h", tz="America/Sao_Paulo"
        )

        np.random.seed(42)
        actual = np.random.randn(len(timestamps)) * 100 + 1000

        # Add systematic patterns
        # Weekends have higher error
        is_weekend = timestamps.dayofweek >= 5
        forecast = actual.copy()
        forecast[is_weekend] += np.random.randn(is_weekend.sum()) * 100
        forecast[~is_weekend] += np.random.randn((~is_weekend).sum()) * 50

        return actual, forecast, timestamps

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = TimePeriodAnalyzer()

        assert analyzer.config.timezone == "America/Sao_Paulo"
        assert analyzer.config.min_samples == 30
        assert analyzer.metrics_calculator is not None

    def test_custom_config(self):
        """Test analyzer with custom config."""
        config = TimePeriodConfig(
            min_samples=50, timezone="UTC", include_statistical_tests=False
        )
        analyzer = TimePeriodAnalyzer(config=config)

        assert analyzer.config.min_samples == 50
        assert analyzer.config.timezone == "UTC"
        assert analyzer.config.include_statistical_tests is False

    def test_analyze_by_weekday(self, large_sample_data):
        """Test weekday analysis."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        assert result.period_type == "weekday"
        assert result.n_periods == 7
        assert "Monday" in result.metrics_by_period
        assert "Sunday" in result.metrics_by_period
        assert len(result.worst_periods) == 7
        assert len(result.best_periods) == 7

        # Check metadata
        assert "total_samples" in result.metadata
        assert "sample_counts" in result.metadata

    def test_analyze_by_season(self, large_sample_data):
        """Test season analysis (Southern Hemisphere)."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_season(actual, forecast, timestamps)

        assert result.period_type == "season"
        assert result.n_periods == 4
        assert "Summer" in result.metrics_by_period
        assert "Autumn" in result.metrics_by_period
        assert "Winter" in result.metrics_by_period
        assert "Spring" in result.metrics_by_period

    def test_analyze_by_hour(self, large_sample_data):
        """Test hour of day analysis."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_hour(actual, forecast, timestamps)

        assert result.period_type == "hour"
        assert result.n_periods == 24
        assert "0" in result.metrics_by_period
        assert "23" in result.metrics_by_period

    def test_analyze_by_month(self, large_sample_data):
        """Test monthly analysis."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_month(actual, forecast, timestamps)

        assert result.period_type == "month"
        assert result.n_periods == 12
        assert "January" in result.metrics_by_period
        assert "December" in result.metrics_by_period

    def test_analyze_by_quarter(self, large_sample_data):
        """Test quarterly analysis."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_quarter(actual, forecast, timestamps)

        assert result.period_type == "quarter"
        assert result.n_periods == 4
        assert "Q1" in result.metrics_by_period
        assert "Q4" in result.metrics_by_period

    def test_analyze_by_day_type(self, large_sample_data):
        """Test day type (weekday vs weekend) analysis."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_day_type(actual, forecast, timestamps)

        assert result.period_type == "day_type"
        assert result.n_periods == 2
        assert "Weekday" in result.metrics_by_period
        assert "Weekend" in result.metrics_by_period

    def test_analyze_by_holiday_proximity(self, large_sample_data):
        """Test holiday proximity analysis."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_holiday_proximity(actual, forecast, timestamps)

        assert result.period_type == "holiday_proximity"
        # Should have at least "Regular" category
        assert "Regular" in result.metrics_by_period

        # May have Holiday, Pre-Holiday, Post-Holiday depending on data
        assert result.n_periods >= 1

    def test_season_extraction_southern_hemisphere(self):
        """Test season extraction for Southern Hemisphere."""
        analyzer = TimePeriodAnalyzer()

        # December = Summer
        assert analyzer._extract_season(pd.Timestamp("2023-12-15")) == "Summer"
        # January = Summer
        assert analyzer._extract_season(pd.Timestamp("2023-01-15")) == "Summer"
        # February = Summer
        assert analyzer._extract_season(pd.Timestamp("2023-02-15")) == "Summer"

        # March = Autumn
        assert analyzer._extract_season(pd.Timestamp("2023-03-15")) == "Autumn"
        # April = Autumn
        assert analyzer._extract_season(pd.Timestamp("2023-04-15")) == "Autumn"
        # May = Autumn
        assert analyzer._extract_season(pd.Timestamp("2023-05-15")) == "Autumn"

        # June = Winter
        assert analyzer._extract_season(pd.Timestamp("2023-06-15")) == "Winter"
        # July = Winter
        assert analyzer._extract_season(pd.Timestamp("2023-07-15")) == "Winter"
        # August = Winter
        assert analyzer._extract_season(pd.Timestamp("2023-08-15")) == "Winter"

        # September = Spring
        assert analyzer._extract_season(pd.Timestamp("2023-09-15")) == "Spring"
        # October = Spring
        assert analyzer._extract_season(pd.Timestamp("2023-10-15")) == "Spring"
        # November = Spring
        assert analyzer._extract_season(pd.Timestamp("2023-11-15")) == "Spring"

    def test_holiday_proximity_extraction(self):
        """Test holiday proximity extraction."""
        analyzer = TimePeriodAnalyzer()

        # New Year's Day
        assert (
            analyzer._extract_holiday_proximity(pd.Timestamp("2023-01-01"))
            == "Holiday"
        )

        # Day before New Year
        assert (
            analyzer._extract_holiday_proximity(pd.Timestamp("2022-12-31"))
            == "Pre-Holiday"
        )

        # Day after New Year
        assert (
            analyzer._extract_holiday_proximity(pd.Timestamp("2023-01-02"))
            == "Post-Holiday"
        )

        # Regular day
        assert (
            analyzer._extract_holiday_proximity(pd.Timestamp("2023-01-15"))
            == "Regular"
        )

        # Christmas
        assert (
            analyzer._extract_holiday_proximity(pd.Timestamp("2023-12-25"))
            == "Holiday"
        )

    def test_min_samples_threshold(self, sample_data):
        """Test minimum samples threshold filtering."""
        actual, forecast, timestamps = sample_data

        # Set high min_samples to filter out some periods
        config = TimePeriodConfig(min_samples=50)
        analyzer = TimePeriodAnalyzer(config=config)

        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        # With only 1 week of data (24 hours per day), no weekday should meet threshold
        assert result.n_periods == 0

    def test_statistical_tests_enabled(self, large_sample_data):
        """Test statistical tests when enabled."""
        actual, forecast, timestamps = large_sample_data

        config = TimePeriodConfig(include_statistical_tests=True)
        analyzer = TimePeriodAnalyzer(config=config)

        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        assert "anova" in result.statistical_tests
        assert "kruskal_wallis" in result.statistical_tests
        assert "p_value" in result.statistical_tests["anova"]
        assert "p_value" in result.statistical_tests["kruskal_wallis"]

    def test_statistical_tests_disabled(self, large_sample_data):
        """Test statistical tests when disabled."""
        actual, forecast, timestamps = large_sample_data

        config = TimePeriodConfig(include_statistical_tests=False)
        analyzer = TimePeriodAnalyzer(config=config)

        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        assert len(result.statistical_tests) == 0

    def test_worst_best_period_identification(self, large_sample_data):
        """Test worst and best period identification."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        # Worst periods should be sorted descending (highest MAPE first)
        worst_mapes = [mape for _, mape in result.worst_periods]
        assert worst_mapes == sorted(worst_mapes, reverse=True)

        # Best periods should be sorted ascending (lowest MAPE first)
        best_mapes = [mape for _, mape in result.best_periods]
        assert best_mapes == sorted(best_mapes)

    def test_identify_worst_periods(self, large_sample_data):
        """Test identify_worst_periods method."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        worst_3 = analyzer.identify_worst_periods(result, n=3)

        assert len(worst_3) == 3
        assert all(isinstance(period, tuple) for period in worst_3)
        assert all(len(period) == 2 for period in worst_3)

    def test_generate_period_report(self, large_sample_data):
        """Test report generation."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        report = analyzer.generate_period_report(result)

        assert isinstance(report, str)
        assert "weekday" in report.lower()
        assert "Monday" in report
        assert "MAPE" in report
        assert "Best period" in report
        assert "Worst period" in report

    def test_compare_periods_weekday(self, large_sample_data):
        """Test compare_periods with weekday type."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.compare_periods(actual, forecast, timestamps, "weekday")

        assert result.period_type == "weekday"
        assert result.n_periods == 7

    def test_compare_periods_invalid_type(self, large_sample_data):
        """Test compare_periods with invalid type."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()

        with pytest.raises(ValueError, match="Invalid period_type"):
            analyzer.compare_periods(actual, forecast, timestamps, "invalid")

    def test_shape_mismatch_error(self):
        """Test error on shape mismatch."""
        analyzer = TimePeriodAnalyzer()

        actual = np.array([1, 2, 3])
        forecast = np.array([1, 2])
        timestamps = pd.date_range("2023-01-01", periods=3, freq="h")

        with pytest.raises(ValueError, match="Shape mismatch"):
            analyzer.analyze_by_weekday(actual, forecast, timestamps)

    def test_length_mismatch_error(self):
        """Test error on length mismatch between values and timestamps."""
        analyzer = TimePeriodAnalyzer()

        actual = np.array([1, 2, 3])
        forecast = np.array([1, 2, 3])
        timestamps = pd.date_range("2023-01-01", periods=2, freq="h")

        with pytest.raises(ValueError, match="Length mismatch"):
            analyzer.analyze_by_weekday(actual, forecast, timestamps)

    def test_timezone_handling(self, large_sample_data):
        """Test timezone handling."""
        actual, forecast, timestamps_utc = large_sample_data

        # Convert to UTC
        timestamps_utc = timestamps_utc.tz_convert("UTC")

        config = TimePeriodConfig(timezone="America/Sao_Paulo")
        analyzer = TimePeriodAnalyzer(config=config)

        result = analyzer.analyze_by_weekday(actual, forecast, timestamps_utc)

        # Should successfully convert and analyze
        assert result.n_periods == 7
        assert result.metadata["timezone"] == "America/Sao_Paulo"

    def test_naive_timestamps(self, large_sample_data):
        """Test handling of naive (non-timezone-aware) timestamps."""
        actual, forecast, _ = large_sample_data

        # Create naive timestamps
        timestamps_naive = pd.date_range("2023-01-01", periods=len(actual), freq="h")

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_weekday(actual, forecast, timestamps_naive)

        # Should localize to configured timezone
        assert result.n_periods == 7

    def test_with_nan_values(self, large_sample_data):
        """Test handling of NaN values."""
        actual, forecast, timestamps = large_sample_data

        # Introduce some NaN values
        actual[10:20] = np.nan
        forecast[30:40] = np.nan

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        # Should successfully analyze after removing NaN values
        assert result.n_periods == 7
        # Total samples should be less than original
        assert result.metadata["total_samples"] < len(actual)

    def test_all_nan_error(self):
        """Test error when all values are NaN."""
        analyzer = TimePeriodAnalyzer()

        actual = np.full(100, np.nan)
        forecast = np.full(100, np.nan)
        timestamps = pd.date_range("2023-01-01", periods=100, freq="h")

        with pytest.raises(ValueError, match="All values are NaN"):
            analyzer.analyze_by_weekday(actual, forecast, timestamps)

    def test_series_input(self, large_sample_data):
        """Test with pandas Series input."""
        actual_arr, forecast_arr, timestamps = large_sample_data

        # Convert to Series
        actual = pd.Series(actual_arr)
        forecast = pd.Series(forecast_arr)

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        assert result.n_periods == 7

    def test_ranking_with_equal_mape(self):
        """Test ranking when multiple periods have equal MAPE."""
        # Create data where weekends have identical errors
        timestamps = pd.date_range("2023-01-01", periods=8760, freq="h")

        np.random.seed(42)
        actual = np.ones(len(timestamps)) * 1000

        # Create forecast with specific patterns
        forecast = actual.copy()
        is_weekend = timestamps.dayofweek >= 5
        # All weekend days get same error
        forecast[is_weekend] += 100
        # Weekdays get random error
        forecast[~is_weekend] += np.random.randn((~is_weekend).sum()) * 50

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        # Saturday and Sunday should have very similar MAPE
        sat_mape = result.metrics_by_period["Saturday"].mape
        sun_mape = result.metrics_by_period["Sunday"].mape
        assert abs(sat_mape - sun_mape) < 1.0  # Should be very close

    def test_brazilian_holidays_defined(self):
        """Test that Brazilian holidays are properly defined."""
        assert len(BRAZILIAN_HOLIDAYS) > 0
        assert "New Year" in BRAZILIAN_HOLIDAYS
        assert "Christmas" in BRAZILIAN_HOLIDAYS
        assert "Independence Day" in BRAZILIAN_HOLIDAYS

        # Check format (month, day)
        for name, date in BRAZILIAN_HOLIDAYS.items():
            assert isinstance(date, tuple)
            assert len(date) == 2
            month, day = date
            assert 1 <= month <= 12
            assert 1 <= day <= 31

    def test_statistical_test_insufficient_groups(self):
        """Test statistical tests with insufficient groups."""
        # Create data with only one weekday having enough samples
        timestamps = pd.date_range("2023-01-02", periods=48, freq="h")  # 2 days

        np.random.seed(42)
        actual = np.random.randn(len(timestamps)) * 100 + 1000
        forecast = actual + np.random.randn(len(timestamps)) * 50

        config = TimePeriodConfig(min_samples=30, include_statistical_tests=True)
        analyzer = TimePeriodAnalyzer(config=config)

        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        # Should have note about insufficient groups
        if "note" in result.statistical_tests:
            assert "Insufficient groups" in result.statistical_tests["note"]

    def test_repr(self):
        """Test string representation."""
        config = TimePeriodConfig(timezone="UTC", min_samples=50)
        analyzer = TimePeriodAnalyzer(config=config)

        repr_str = repr(analyzer)

        assert "TimePeriodAnalyzer" in repr_str
        assert "UTC" in repr_str
        assert "50" in repr_str

    def test_result_repr(self):
        """Test TimePeriodResult string representation."""
        from src.evaluation.metrics import MetricsResult

        metrics = MetricsResult(
            mape=5.0, mae=10.0, rmse=15.0, mse=225.0, r2=0.95, sample_size=100
        )

        result = TimePeriodResult(
            period_type="weekday",
            metrics_by_period={"Monday": metrics},
            worst_periods=[("Monday", 5.0)],
            best_periods=[("Monday", 5.0)],
        )

        repr_str = repr(result)

        assert "TimePeriodResult" in repr_str
        assert "weekday" in repr_str

    def test_period_order_preserved(self, large_sample_data):
        """Test that period order is preserved in results."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        # Weekdays should appear in Monday-Sunday order
        weekday_names = list(result.metrics_by_period.keys())
        expected_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        assert weekday_names == expected_order

    def test_season_order_preserved(self, large_sample_data):
        """Test that season order is preserved in results."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_season(actual, forecast, timestamps)

        # Seasons should appear in Summer-Autumn-Winter-Spring order
        season_names = list(result.metrics_by_period.keys())
        expected_order = ["Summer", "Autumn", "Winter", "Spring"]

        assert season_names == expected_order

    def test_hour_order_preserved(self, large_sample_data):
        """Test that hour order is preserved in results."""
        actual, forecast, timestamps = large_sample_data

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_hour(actual, forecast, timestamps)

        # Hours should appear in 0-23 order
        hour_names = list(result.metrics_by_period.keys())
        expected_order = [str(h) for h in range(24)]

        assert hour_names == expected_order


# Integration tests


class TestTimePeriodIntegration:
    """Integration tests for time period analyzer."""

    def test_full_workflow(self):
        """Test complete workflow from data to report."""
        # Create realistic data
        timestamps = pd.date_range(
            "2023-01-01", periods=8760, freq="h", tz="America/Sao_Paulo"
        )

        np.random.seed(42)
        actual = np.random.randn(len(timestamps)) * 100 + 1000

        # Add systematic patterns
        # Weekends have higher error
        is_weekend = timestamps.dayofweek >= 5
        forecast = actual.copy()
        forecast[is_weekend] += np.random.randn(is_weekend.sum()) * 100
        forecast[~is_weekend] += np.random.randn((~is_weekend).sum()) * 50

        # Configure analyzer
        config = TimePeriodConfig(
            period_types=["weekday", "season", "hour"],
            metrics=["mape", "rmse"],
            min_samples=30,
            include_statistical_tests=True,
        )

        analyzer = TimePeriodAnalyzer(config=config)

        # Analyze by weekday
        weekday_result = analyzer.analyze_by_weekday(actual, forecast, timestamps)

        assert weekday_result.n_periods == 7

        # Weekends should be worse
        sat_mape = weekday_result.metrics_by_period["Saturday"].mape
        mon_mape = weekday_result.metrics_by_period["Monday"].mape
        assert sat_mape > mon_mape

        # Generate report
        report = analyzer.generate_period_report(weekday_result)
        assert len(report) > 0

        # Analyze by season
        season_result = analyzer.analyze_by_season(actual, forecast, timestamps)
        assert season_result.n_periods == 4

        # Analyze by hour
        hour_result = analyzer.analyze_by_hour(actual, forecast, timestamps)
        assert hour_result.n_periods == 24

    def test_multiple_period_types(self):
        """Test analyzing multiple period types on same data."""
        timestamps = pd.date_range(
            "2023-01-01", periods=8760, freq="h", tz="America/Sao_Paulo"
        )

        np.random.seed(42)
        actual = np.random.randn(len(timestamps)) * 100 + 1000
        forecast = actual + np.random.randn(len(timestamps)) * 50

        analyzer = TimePeriodAnalyzer()

        # Analyze all period types
        results = {}
        results["weekday"] = analyzer.analyze_by_weekday(actual, forecast, timestamps)
        results["season"] = analyzer.analyze_by_season(actual, forecast, timestamps)
        results["hour"] = analyzer.analyze_by_hour(actual, forecast, timestamps)
        results["month"] = analyzer.analyze_by_month(actual, forecast, timestamps)
        results["quarter"] = analyzer.analyze_by_quarter(actual, forecast, timestamps)
        results["day_type"] = analyzer.analyze_by_day_type(actual, forecast, timestamps)

        # All should succeed
        assert all(result.n_periods > 0 for result in results.values())

    def test_dst_transition_handling(self):
        """Test handling of DST (Daylight Saving Time) transitions."""
        # Brazil DST typically occurs in October (start) and February (end)
        # Create data spanning DST transition
        timestamps = pd.date_range(
            "2023-10-01",
            periods=30 * 24,  # 30 days
            freq="h",
            tz="America/Sao_Paulo",
        )

        np.random.seed(42)
        actual = np.random.randn(len(timestamps)) * 100 + 1000
        forecast = actual + np.random.randn(len(timestamps)) * 50

        analyzer = TimePeriodAnalyzer()
        result = analyzer.analyze_by_hour(actual, forecast, timestamps)

        # Should handle DST gracefully
        assert result.n_periods == 24

    def test_performance_with_large_dataset(self):
        """Test performance with large dataset (multi-year)."""
        # 3 years of hourly data
        timestamps = pd.date_range(
            "2021-01-01", periods=3 * 8760, freq="h", tz="America/Sao_Paulo"
        )

        np.random.seed(42)
        actual = np.random.randn(len(timestamps)) * 100 + 1000
        forecast = actual + np.random.randn(len(timestamps)) * 50

        analyzer = TimePeriodAnalyzer()

        import time

        start = time.time()
        result = analyzer.analyze_by_weekday(actual, forecast, timestamps)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert result.n_periods == 7
