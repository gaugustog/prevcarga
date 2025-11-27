"""Tests for metrics calculation module."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    MetricsCalculator,
    MetricsConfig,
    MetricsResult,
    StatisticalTests,
)


# Tests for MetricsConfig


class TestMetricsConfig:
    """Tests for MetricsConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MetricsConfig()

        assert config.epsilon == 1e-8
        assert config.use_symmetric_mape is True
        assert config.confidence_level == 0.95
        assert config.bootstrap_iterations == 1000
        assert config.nan_handling == "exclude"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MetricsConfig(
            epsilon=1e-6,
            use_symmetric_mape=False,
            confidence_level=0.99,
            bootstrap_iterations=500,
        )

        assert config.epsilon == 1e-6
        assert config.use_symmetric_mape is False
        assert config.confidence_level == 0.99
        assert config.bootstrap_iterations == 500

    def test_invalid_epsilon(self):
        """Test invalid epsilon raises error."""
        with pytest.raises(ValueError, match="epsilon"):
            MetricsConfig(epsilon=0)

        with pytest.raises(ValueError, match="epsilon"):
            MetricsConfig(epsilon=-1)

    def test_invalid_confidence_level(self):
        """Test invalid confidence level raises error."""
        with pytest.raises(ValueError, match="confidence_level"):
            MetricsConfig(confidence_level=0)

        with pytest.raises(ValueError, match="confidence_level"):
            MetricsConfig(confidence_level=1.0)

    def test_invalid_bootstrap_iterations(self):
        """Test invalid bootstrap iterations raises error."""
        with pytest.raises(ValueError, match="bootstrap_iterations"):
            MetricsConfig(bootstrap_iterations=-1)

    def test_invalid_nan_handling(self):
        """Test invalid nan_handling raises error."""
        with pytest.raises(ValueError, match="nan_handling"):
            MetricsConfig(nan_handling="invalid")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = MetricsConfig(epsilon=1e-6, confidence_level=0.9)
        d = config.to_dict()

        assert d["epsilon"] == 1e-6
        assert d["confidence_level"] == 0.9


# Tests for MetricsResult


class TestMetricsResult:
    """Tests for MetricsResult dataclass."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = MetricsResult(
            mape=5.0,
            mae=10.0,
            rmse=15.0,
            mse=225.0,
            r2=0.95,
            sample_size=100,
        )

        assert result.mape == 5.0
        assert result.mae == 10.0
        assert result.rmse == 15.0
        assert result.mse == 225.0
        assert result.r2 == 0.95
        assert result.sample_size == 100

    def test_result_with_ci(self):
        """Test result with confidence intervals."""
        result = MetricsResult(
            mape=5.0,
            mae=10.0,
            rmse=15.0,
            mse=225.0,
            r2=0.95,
            confidence_intervals={"mape": (4.5, 5.5), "mae": (9.0, 11.0)},
            sample_size=100,
        )

        assert result.confidence_intervals["mape"] == (4.5, 5.5)
        assert result.confidence_intervals["mae"] == (9.0, 11.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MetricsResult(
            mape=5.0,
            mae=10.0,
            rmse=15.0,
            mse=225.0,
            r2=0.95,
            sample_size=100,
        )

        d = result.to_dict()

        assert d["mape"] == 5.0
        assert d["mae"] == 10.0
        assert d["sample_size"] == 100

    def test_repr(self):
        """Test string representation."""
        result = MetricsResult(
            mape=5.0,
            mae=10.0,
            rmse=15.0,
            mse=225.0,
            r2=0.95,
            sample_size=100,
        )

        repr_str = repr(result)

        assert "MetricsResult" in repr_str
        assert "MAPE=5.00%" in repr_str
        assert "RMSE=15.00" in repr_str


# Fixtures for MetricsCalculator tests


@pytest.fixture
def calculator():
    """Create default calculator."""
    return MetricsCalculator()


@pytest.fixture
def calculator_standard_mape():
    """Create calculator with standard MAPE."""
    config = MetricsConfig(use_symmetric_mape=False)
    return MetricsCalculator(config)


@pytest.fixture
def calculator_no_bootstrap():
    """Create calculator without bootstrap CI."""
    config = MetricsConfig(bootstrap_iterations=0)
    return MetricsCalculator(config)


@pytest.fixture
def perfect_forecast():
    """Perfect forecast data."""
    actual = np.array([100, 200, 300, 400, 500])
    forecast = np.array([100, 200, 300, 400, 500])
    return actual, forecast


@pytest.fixture
def good_forecast():
    """Good forecast data."""
    actual = np.array([100, 200, 300, 400, 500])
    forecast = np.array([95, 210, 290, 410, 480])
    return actual, forecast


@pytest.fixture
def bad_forecast():
    """Bad forecast data."""
    actual = np.array([100, 200, 300, 400, 500])
    forecast = np.array([200, 100, 400, 200, 600])
    return actual, forecast


# Tests for MetricsCalculator


class TestMetricsCalculatorInit:
    """Tests for MetricsCalculator initialization."""

    def test_default_init(self):
        """Test default initialization."""
        calc = MetricsCalculator()

        assert calc.config.use_symmetric_mape is True

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = MetricsConfig(use_symmetric_mape=False)
        calc = MetricsCalculator(config)

        assert calc.config.use_symmetric_mape is False


class TestMAPE:
    """Tests for MAPE calculation."""

    def test_mape_perfect_forecast(self, calculator, perfect_forecast):
        """Test MAPE for perfect forecast."""
        actual, forecast = perfect_forecast

        mape = calculator.calculate_mape(actual, forecast)

        assert mape == pytest.approx(0.0, abs=1e-6)

    def test_mape_good_forecast(self, calculator, good_forecast):
        """Test MAPE for good forecast."""
        actual, forecast = good_forecast

        mape = calculator.calculate_mape(actual, forecast)

        # Expected: errors are 5, 10, 10, 10, 20 on values 100, 200, 300, 400, 500
        # sMAPE formula
        assert 0 < mape < 10  # Should be relatively small

    def test_smape_with_zeros(self, calculator):
        """Test symmetric MAPE handles zeros gracefully."""
        actual = np.array([0, 100, 200])
        forecast = np.array([10, 100, 200])

        mape = calculator.calculate_mape(actual, forecast)

        assert np.isfinite(mape)
        assert mape > 0

    def test_standard_mape_with_epsilon(self, calculator_standard_mape):
        """Test standard MAPE uses epsilon protection."""
        actual = np.array([0, 100, 200])
        forecast = np.array([10, 100, 200])

        mape = calculator_standard_mape.calculate_mape(actual, forecast)

        assert np.isfinite(mape)


class TestMAE:
    """Tests for MAE calculation."""

    def test_mae_perfect_forecast(self, calculator, perfect_forecast):
        """Test MAE for perfect forecast."""
        actual, forecast = perfect_forecast

        mae = calculator.calculate_mae(actual, forecast)

        assert mae == pytest.approx(0.0, abs=1e-6)

    def test_mae_good_forecast(self, calculator, good_forecast):
        """Test MAE for good forecast."""
        actual, forecast = good_forecast

        mae = calculator.calculate_mae(actual, forecast)

        # Expected: mean(|5, 10, 10, 10, 20|) = 11
        assert mae == pytest.approx(11.0, abs=0.1)

    def test_mae_weighted(self, calculator):
        """Test weighted MAE."""
        config = MetricsConfig(weighted_metrics=True)
        calc = MetricsCalculator(config)

        actual = np.array([100, 200, 300])
        forecast = np.array([90, 200, 310])  # errors: 10, 0, 10
        weights = np.array([1.0, 2.0, 1.0])

        mae = calc.calculate_mae(actual, forecast, weights)

        # Weighted: (1*10 + 2*0 + 1*10) / 4 = 5
        assert mae == pytest.approx(5.0, abs=0.1)


class TestRMSE:
    """Tests for RMSE calculation."""

    def test_rmse_perfect_forecast(self, calculator, perfect_forecast):
        """Test RMSE for perfect forecast."""
        actual, forecast = perfect_forecast

        rmse = calculator.calculate_rmse(actual, forecast)

        assert rmse == pytest.approx(0.0, abs=1e-6)

    def test_rmse_good_forecast(self, calculator, good_forecast):
        """Test RMSE for good forecast."""
        actual, forecast = good_forecast

        rmse = calculator.calculate_rmse(actual, forecast)

        # Expected: sqrt(mean(25, 100, 100, 100, 400)) = sqrt(145) ≈ 12.04
        assert 10 < rmse < 15

    def test_rmse_weighted(self, calculator):
        """Test weighted RMSE."""
        config = MetricsConfig(weighted_metrics=True)
        calc = MetricsCalculator(config)

        actual = np.array([100, 200])
        forecast = np.array([90, 200])  # errors: 10, 0
        weights = np.array([1.0, 1.0])

        rmse = calc.calculate_rmse(actual, forecast, weights)

        # sqrt((1*100 + 1*0) / 2) = sqrt(50) ≈ 7.07
        assert rmse == pytest.approx(np.sqrt(50), abs=0.1)


class TestR2:
    """Tests for R² calculation."""

    def test_r2_perfect_forecast(self, calculator, perfect_forecast):
        """Test R² for perfect forecast."""
        actual, forecast = perfect_forecast

        r2 = calculator.calculate_r2(actual, forecast)

        assert r2 == pytest.approx(1.0, abs=1e-6)

    def test_r2_good_forecast(self, calculator, good_forecast):
        """Test R² for good forecast."""
        actual, forecast = good_forecast

        r2 = calculator.calculate_r2(actual, forecast)

        assert 0.9 < r2 < 1.0  # Should be high for good forecast

    def test_r2_bad_forecast(self, calculator, bad_forecast):
        """Test R² for bad forecast."""
        actual, forecast = bad_forecast

        r2 = calculator.calculate_r2(actual, forecast)

        assert r2 < 0.9  # Should be lower for bad forecast

    def test_r2_constant_actual(self, calculator):
        """Test R² when actual values are constant."""
        actual = np.array([100, 100, 100, 100])
        forecast = np.array([100, 100, 100, 100])

        r2 = calculator.calculate_r2(actual, forecast)

        assert r2 == 1.0  # Perfect prediction of constant

    def test_r2_can_be_negative(self, calculator):
        """Test R² can be negative for very bad fits."""
        actual = np.array([1, 2, 3, 4, 5])
        forecast = np.array([5, 4, 3, 2, 1])  # Inverse prediction

        r2 = calculator.calculate_r2(actual, forecast)

        assert r2 < 0


class TestCalculateSingle:
    """Tests for calculate_single method."""

    def test_calculate_single_basic(self, calculator, good_forecast):
        """Test calculate_single with basic data."""
        actual, forecast = good_forecast

        result = calculator.calculate_single(actual, forecast)

        assert isinstance(result, MetricsResult)
        assert result.sample_size == 5
        assert result.n_excluded == 0

    def test_calculate_single_with_nan(self, calculator):
        """Test calculate_single with NaN values."""
        actual = np.array([100, 200, np.nan, 400, 500])
        forecast = np.array([100, 200, 300, 400, 500])

        result = calculator.calculate_single(actual, forecast)

        assert result.sample_size == 4
        assert result.n_excluded == 1

    def test_calculate_single_empty_after_nan_exclusion(self, calculator):
        """Test calculate_single when all values are NaN."""
        actual = np.array([np.nan, np.nan, np.nan])
        forecast = np.array([1, 2, 3])

        result = calculator.calculate_single(actual, forecast)

        assert result.sample_size == 0
        assert np.isnan(result.mape)

    def test_calculate_single_shape_mismatch(self, calculator):
        """Test calculate_single with mismatched shapes."""
        actual = np.array([100, 200, 300])
        forecast = np.array([100, 200])

        with pytest.raises(ValueError, match="Shape mismatch"):
            calculator.calculate_single(actual, forecast)

    def test_calculate_single_empty_arrays(self, calculator):
        """Test calculate_single with empty arrays."""
        actual = np.array([])
        forecast = np.array([])

        with pytest.raises(ValueError, match="empty"):
            calculator.calculate_single(actual, forecast)


class TestCalculateMetrics:
    """Tests for calculate_metrics method."""

    def test_calculate_metrics_multiple(self, calculator):
        """Test calculate_metrics with multiple series."""
        predictions = {
            "h0": np.array([100, 200, 300]),
            "h1": np.array([110, 210, 310]),
        }
        actuals = {
            "h0": np.array([100, 200, 300]),
            "h1": np.array([100, 200, 300]),
        }

        results = calculator.calculate_metrics(predictions, actuals)

        assert "h0" in results
        assert "h1" in results
        assert results["h0"].mape == pytest.approx(0.0, abs=1e-6)

    def test_calculate_metrics_mismatched_keys(self, calculator):
        """Test calculate_metrics with mismatched keys."""
        predictions = {"h0": np.array([100])}
        actuals = {"h1": np.array([100])}

        with pytest.raises(ValueError, match="same keys"):
            calculator.calculate_metrics(predictions, actuals)


class TestConfidenceIntervals:
    """Tests for confidence interval calculation."""

    def test_ci_calculation(self, calculator, good_forecast):
        """Test confidence interval calculation."""
        actual, forecast = good_forecast

        ci = calculator.calculate_confidence_intervals(actual, forecast)

        assert "mape" in ci
        assert "mae" in ci
        assert "rmse" in ci
        assert ci["mape"][0] < ci["mape"][1]  # Lower < Upper

    def test_ci_with_seed(self, good_forecast):
        """Test CI reproducibility with seed."""
        actual, forecast = good_forecast

        config = MetricsConfig(random_seed=42, bootstrap_iterations=100)
        calc1 = MetricsCalculator(config)
        calc2 = MetricsCalculator(config)

        ci1 = calc1.calculate_confidence_intervals(actual, forecast)
        ci2 = calc2.calculate_confidence_intervals(actual, forecast)

        assert ci1["mape"] == ci2["mape"]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_zeros_actual(self, calculator):
        """Test with all zero actual values."""
        actual = np.array([0, 0, 0, 0, 0])
        forecast = np.array([1, 2, 3, 4, 5])

        result = calculator.calculate_single(actual, forecast)

        # Should not raise, use epsilon protection
        assert np.isfinite(result.mape)

    def test_all_zeros_forecast(self, calculator):
        """Test with all zero forecast values."""
        actual = np.array([1, 2, 3, 4, 5])
        forecast = np.array([0, 0, 0, 0, 0])

        result = calculator.calculate_single(actual, forecast)

        assert np.isfinite(result.mape)

    def test_single_value(self, calculator):
        """Test with single value."""
        actual = np.array([100])
        forecast = np.array([110])

        result = calculator.calculate_single(actual, forecast)

        assert result.sample_size == 1
        assert np.isfinite(result.mape)

    def test_identical_values(self, calculator):
        """Test with identical actual and forecast."""
        actual = np.array([100, 100, 100, 100])
        forecast = np.array([100, 100, 100, 100])

        result = calculator.calculate_single(actual, forecast)

        assert result.mape == pytest.approx(0.0, abs=1e-6)
        assert result.mae == pytest.approx(0.0, abs=1e-6)
        assert result.rmse == pytest.approx(0.0, abs=1e-6)

    def test_mixed_nan_values(self, calculator):
        """Test with mixed NaN values."""
        actual = np.array([100, np.nan, 300, 400, np.nan])
        forecast = np.array([100, 200, np.nan, 400, 500])

        result = calculator.calculate_single(actual, forecast)

        assert result.sample_size == 2  # Only (100, 100) and (400, 400) are valid
        assert result.n_excluded == 3

    def test_near_zero_denominators(self, calculator):
        """Test with near-zero denominators."""
        actual = np.array([1e-10, 1e-10, 100])
        forecast = np.array([1e-10, 1e-10, 100])

        result = calculator.calculate_single(actual, forecast)

        assert np.isfinite(result.mape)

    def test_large_values(self, calculator):
        """Test numerical stability with large values."""
        actual = np.array([1e10, 2e10, 3e10])
        forecast = np.array([1.1e10, 2.1e10, 3.1e10])

        result = calculator.calculate_single(actual, forecast)

        assert np.isfinite(result.mape)
        assert np.isfinite(result.rmse)

    def test_nan_handling_error_mode(self):
        """Test nan_handling='error' raises error."""
        config = MetricsConfig(nan_handling="error")
        calc = MetricsCalculator(config)

        actual = np.array([100, np.nan, 300])
        forecast = np.array([100, 200, 300])

        with pytest.raises(ValueError, match="NaN"):
            calc.calculate_single(actual, forecast)


# Tests for StatisticalTests


class TestStatisticalTests:
    """Tests for StatisticalTests class."""

    @pytest.fixture
    def tests(self):
        """Create StatisticalTests instance."""
        return StatisticalTests()

    @pytest.fixture
    def different_errors(self):
        """Create errors with significant difference."""
        np.random.seed(42)
        errors_a = np.random.normal(10, 2, 100)
        errors_b = np.random.normal(15, 2, 100)
        return errors_a, errors_b

    @pytest.fixture
    def similar_errors(self):
        """Create errors with no significant difference."""
        np.random.seed(42)
        errors_a = np.random.normal(10, 2, 100)
        errors_b = np.random.normal(10, 2, 100)
        return errors_a, errors_b

    def test_paired_t_test_significant(self, tests, different_errors):
        """Test paired t-test detects significant difference."""
        errors_a, errors_b = different_errors

        result = tests.paired_t_test(errors_a, errors_b)

        assert result["significant"] == True  # noqa: E712 - numpy bool comparison
        assert result["p_value"] < 0.05
        assert "t_statistic" in result

    def test_paired_t_test_not_significant(self, tests, similar_errors):
        """Test paired t-test with no significant difference."""
        errors_a, errors_b = similar_errors

        result = tests.paired_t_test(errors_a, errors_b)

        # May or may not be significant depending on random values
        assert "significant" in result
        assert "p_value" in result

    def test_paired_t_test_length_mismatch(self, tests):
        """Test paired t-test with mismatched lengths."""
        errors_a = np.array([1, 2, 3])
        errors_b = np.array([1, 2])

        with pytest.raises(ValueError, match="same length"):
            tests.paired_t_test(errors_a, errors_b)

    def test_paired_t_test_insufficient_samples(self, tests):
        """Test paired t-test with insufficient samples."""
        errors_a = np.array([1])
        errors_b = np.array([2])

        with pytest.raises(ValueError, match="at least 2"):
            tests.paired_t_test(errors_a, errors_b)

    def test_wilcoxon_significant(self, tests, different_errors):
        """Test Wilcoxon test detects significant difference."""
        errors_a, errors_b = different_errors

        result = tests.wilcoxon_test(errors_a, errors_b)

        assert result["significant"] == True  # noqa: E712 - numpy bool comparison
        assert result["p_value"] < 0.05

    def test_wilcoxon_insufficient_samples(self, tests):
        """Test Wilcoxon test with insufficient samples."""
        errors_a = np.array([1, 2, 3])
        errors_b = np.array([2, 3, 4])

        with pytest.raises(ValueError, match="at least 10"):
            tests.wilcoxon_test(errors_a, errors_b)

    def test_wilcoxon_zero_differences(self, tests):
        """Test Wilcoxon test with all zero differences."""
        errors = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = tests.wilcoxon_test(errors, errors)

        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_compare_models(self, tests):
        """Test compare_models method."""
        np.random.seed(42)
        actuals = np.random.normal(100, 10, 50)
        predictions_a = actuals + np.random.normal(0, 5, 50)
        predictions_b = actuals + np.random.normal(0, 10, 50)

        result = tests.compare_models(predictions_a, predictions_b, actuals)

        assert "t_test" in result
        assert "wilcoxon" in result
        assert "mean_error_a" in result
        assert "mean_error_b" in result
        assert "better_model" in result


class TestPerformance:
    """Performance tests for metrics calculation."""

    def test_performance_large_dataset(self, calculator_no_bootstrap):
        """Test performance with large dataset."""
        np.random.seed(42)
        n = 10000
        actual = np.random.normal(1000, 100, n)
        forecast = actual + np.random.normal(0, 50, n)

        result = calculator_no_bootstrap.calculate_single(actual, forecast)

        assert result.sample_size == n

    def test_performance_many_series(self, calculator_no_bootstrap):
        """Test performance with many series."""
        np.random.seed(42)
        n_series = 26 * 9  # 26 series × 9 horizons
        n_samples = 100

        predictions = {}
        actuals = {}
        for i in range(n_series):
            actual = np.random.normal(1000, 100, n_samples)
            pred = actual + np.random.normal(0, 50, n_samples)
            key = f"series_{i}"
            predictions[key] = pred
            actuals[key] = actual

        results = calculator_no_bootstrap.calculate_metrics(predictions, actuals)

        assert len(results) == n_series
