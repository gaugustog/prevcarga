"""Tests for the percentile analysis module.

Tests cover:
- OutlierAnalysis dataclass functionality
- DistributionAnalysis dataclass functionality
- PercentileAnalyzer initialization and configuration
- Distribution analysis methods
- Outlier detection methods (IQR, Z-score, Modified Z-score)
- Normality testing
- Edge cases and error handling
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from src.evaluation.percentile import (
    DEFAULT_PERCENTILES,
    DistributionAnalysis,
    OutlierAnalysis,
    PercentileAnalyzer,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def normal_data() -> np.ndarray:
    """Generate normally distributed data."""
    np.random.seed(42)
    return np.random.normal(loc=0.0, scale=1.0, size=100)


@pytest.fixture
def skewed_data() -> np.ndarray:
    """Generate right-skewed data."""
    np.random.seed(42)
    return np.random.exponential(scale=2.0, size=100)


@pytest.fixture
def data_with_outliers() -> np.ndarray:
    """Generate data with known outliers."""
    np.random.seed(42)
    data = np.random.normal(loc=0.0, scale=1.0, size=100)
    # Add clear outliers
    data[0] = 10.0
    data[1] = -10.0
    data[2] = 15.0
    return data


@pytest.fixture
def forecast_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate actual and forecast data pairs."""
    np.random.seed(42)
    actual = np.random.uniform(100, 200, size=50)
    forecast = actual + np.random.normal(0, 5, size=50)
    return actual, forecast


@pytest.fixture
def analyzer() -> PercentileAnalyzer:
    """Create default analyzer."""
    return PercentileAnalyzer()


# =============================================================================
# OutlierAnalysis Tests
# =============================================================================


class TestOutlierAnalysis:
    """Tests for OutlierAnalysis dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic OutlierAnalysis creation."""
        analysis = OutlierAnalysis(
            outlier_indices=[2, 5, 10],
            outlier_values=np.array([10.0, -8.0, 12.0]),
            outlier_scores=np.array([3.5, 3.2, 4.0]),
            threshold_used=3.0,
            method="zscore",
        )

        assert analysis.n_outliers == 3
        assert analysis.method == "zscore"
        assert analysis.threshold_used == 3.0
        assert analysis.outlier_indices == [2, 5, 10]

    def test_n_outliers_property(self) -> None:
        """Test n_outliers property."""
        analysis = OutlierAnalysis(
            outlier_indices=[1, 2, 3, 4, 5],
            outlier_values=np.array([1, 2, 3, 4, 5]),
            outlier_scores=np.array([1, 1, 1, 1, 1]),
            threshold_used=1.5,
            method="iqr",
        )
        assert analysis.n_outliers == 5

    def test_empty_outliers(self) -> None:
        """Test OutlierAnalysis with no outliers."""
        analysis = OutlierAnalysis(
            outlier_indices=[],
            outlier_values=np.array([]),
            outlier_scores=np.array([]),
            threshold_used=3.0,
            method="zscore",
        )

        assert analysis.n_outliers == 0
        assert len(analysis.outlier_values) == 0

    def test_outlier_rate_without_total(self) -> None:
        """Test outlier_rate without total count set."""
        analysis = OutlierAnalysis(
            outlier_indices=[1, 2],
            outlier_values=np.array([1, 2]),
            outlier_scores=np.array([1, 1]),
            threshold_used=1.5,
            method="iqr",
        )
        # Without _total_count, rate is 0
        assert analysis.outlier_rate == 0.0

    def test_outlier_rate_with_total(self) -> None:
        """Test outlier_rate with total count set."""
        analysis = OutlierAnalysis(
            outlier_indices=[1, 2],
            outlier_values=np.array([1, 2]),
            outlier_scores=np.array([1, 1]),
            threshold_used=1.5,
            method="iqr",
        )
        analysis._total_count = 100  # type: ignore[attr-defined]
        assert analysis.outlier_rate == 0.02

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        analysis = OutlierAnalysis(
            outlier_indices=[0, 1],
            outlier_values=np.array([10.0, -10.0]),
            outlier_scores=np.array([3.5, 3.5]),
            threshold_used=3.0,
            method="zscore",
            lower_bound=-3.0,
            upper_bound=3.0,
        )

        d = analysis.to_dict()
        assert d["outlier_indices"] == [0, 1]
        assert d["outlier_values"] == [10.0, -10.0]
        assert d["outlier_scores"] == [3.5, 3.5]
        assert d["threshold_used"] == 3.0
        assert d["method"] == "zscore"
        assert d["n_outliers"] == 2
        assert d["lower_bound"] == -3.0
        assert d["upper_bound"] == 3.0

    def test_repr(self) -> None:
        """Test string representation."""
        analysis = OutlierAnalysis(
            outlier_indices=[0, 1, 2],
            outlier_values=np.array([1, 2, 3]),
            outlier_scores=np.array([1, 1, 1]),
            threshold_used=1.5,
            method="iqr",
        )

        repr_str = repr(analysis)
        assert "OutlierAnalysis" in repr_str
        assert "iqr" in repr_str
        assert "n_outliers=3" in repr_str
        assert "threshold=1.5" in repr_str


# =============================================================================
# DistributionAnalysis Tests
# =============================================================================


class TestDistributionAnalysis:
    """Tests for DistributionAnalysis dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic DistributionAnalysis creation."""
        analysis = DistributionAnalysis(
            percentiles={5: -1.5, 25: -0.5, 50: 0.0, 75: 0.5, 95: 1.5},
            mean=0.0,
            std=1.0,
            skewness=0.0,
            kurtosis=3.0,
            normality_test={"is_normal": True},
            sample_size=100,
        )

        assert analysis.mean == 0.0
        assert analysis.std == 1.0
        assert analysis.sample_size == 100

    def test_median_property(self) -> None:
        """Test median property."""
        analysis = DistributionAnalysis(
            percentiles={50: 5.5},
            mean=5.0,
            std=1.0,
            skewness=0.0,
            kurtosis=3.0,
            normality_test={},
            sample_size=10,
        )

        assert analysis.median == 5.5

    def test_median_missing(self) -> None:
        """Test median when P50 not calculated."""
        analysis = DistributionAnalysis(
            percentiles={25: 2.0, 75: 8.0},
            mean=5.0,
            std=1.0,
            skewness=0.0,
            kurtosis=3.0,
            normality_test={},
            sample_size=10,
        )

        assert np.isnan(analysis.median)

    def test_iqr_property(self) -> None:
        """Test IQR property."""
        analysis = DistributionAnalysis(
            percentiles={25: 2.0, 50: 5.0, 75: 8.0},
            mean=5.0,
            std=2.0,
            skewness=0.0,
            kurtosis=3.0,
            normality_test={},
            sample_size=100,
        )

        assert analysis.iqr == 6.0

    def test_is_normal_property(self) -> None:
        """Test is_normal property."""
        # Normal distribution
        analysis_normal = DistributionAnalysis(
            percentiles={50: 0.0},
            mean=0.0,
            std=1.0,
            skewness=0.0,
            kurtosis=3.0,
            normality_test={"is_normal": True},
            sample_size=100,
        )
        assert analysis_normal.is_normal == True  # noqa: E712

        # Non-normal distribution
        analysis_nonnormal = DistributionAnalysis(
            percentiles={50: 0.0},
            mean=0.0,
            std=1.0,
            skewness=2.0,
            kurtosis=10.0,
            normality_test={"is_normal": False},
            sample_size=100,
        )
        assert analysis_nonnormal.is_normal == False  # noqa: E712

    def test_outlier_count_property(self) -> None:
        """Test outlier_count property."""
        outlier_analysis = OutlierAnalysis(
            outlier_indices=[1, 2, 3],
            outlier_values=np.array([1, 2, 3]),
            outlier_scores=np.array([1, 1, 1]),
            threshold_used=1.5,
            method="iqr",
        )

        analysis = DistributionAnalysis(
            percentiles={50: 0.0},
            mean=0.0,
            std=1.0,
            skewness=0.0,
            kurtosis=3.0,
            normality_test={},
            outlier_analysis=outlier_analysis,
            sample_size=100,
        )

        assert analysis.outlier_count == 3

    def test_outlier_count_no_analysis(self) -> None:
        """Test outlier_count when no outlier analysis."""
        analysis = DistributionAnalysis(
            percentiles={50: 0.0},
            mean=0.0,
            std=1.0,
            skewness=0.0,
            kurtosis=3.0,
            normality_test={},
            outlier_analysis=None,
            sample_size=100,
        )

        assert analysis.outlier_count == 0

    def test_outlier_indices_property(self) -> None:
        """Test outlier_indices property."""
        outlier_analysis = OutlierAnalysis(
            outlier_indices=[5, 10, 15],
            outlier_values=np.array([1, 2, 3]),
            outlier_scores=np.array([1, 1, 1]),
            threshold_used=1.5,
            method="iqr",
        )

        analysis = DistributionAnalysis(
            percentiles={50: 0.0},
            mean=0.0,
            std=1.0,
            skewness=0.0,
            kurtosis=3.0,
            normality_test={},
            outlier_analysis=outlier_analysis,
            sample_size=100,
        )

        assert analysis.outlier_indices == [5, 10, 15]

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        analysis = DistributionAnalysis(
            percentiles={5: -1.5, 50: 0.0, 95: 1.5},
            mean=0.1,
            std=1.0,
            skewness=0.05,
            kurtosis=3.0,
            normality_test={"is_normal": True, "p_value": 0.5},
            sample_size=100,
            metadata={"n_excluded": 0},
        )

        d = analysis.to_dict()
        assert d["percentiles"] == {5: -1.5, 50: 0.0, 95: 1.5}
        assert d["mean"] == 0.1
        assert d["std"] == 1.0
        assert d["skewness"] == 0.05
        assert d["kurtosis"] == 3.0
        assert d["normality_test"]["is_normal"] == True  # noqa: E712
        assert d["sample_size"] == 100
        assert d["outlier_analysis"] is None

    def test_repr(self) -> None:
        """Test string representation."""
        analysis = DistributionAnalysis(
            percentiles={50: 5.0},
            mean=5.0,
            std=2.0,
            skewness=0.5,
            kurtosis=3.5,
            normality_test={},
            sample_size=50,
        )

        repr_str = repr(analysis)
        assert "DistributionAnalysis" in repr_str
        assert "mean=5.00" in repr_str
        assert "std=2.00" in repr_str
        assert "n=50" in repr_str


# =============================================================================
# PercentileAnalyzer Initialization Tests
# =============================================================================


class TestPercentileAnalyzerInit:
    """Tests for PercentileAnalyzer initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        analyzer = PercentileAnalyzer()

        assert analyzer.percentiles == DEFAULT_PERCENTILES
        assert analyzer.outlier_method == "iqr"
        assert analyzer.outlier_threshold == 1.5

    def test_custom_percentiles(self) -> None:
        """Test custom percentile configuration."""
        custom_percentiles = [10, 50, 90]
        analyzer = PercentileAnalyzer(percentiles=custom_percentiles)

        assert analyzer.percentiles == [10, 50, 90]

    def test_custom_outlier_method(self) -> None:
        """Test custom outlier method configuration."""
        analyzer = PercentileAnalyzer(outlier_method="zscore", outlier_threshold=3.0)

        assert analyzer.outlier_method == "zscore"
        assert analyzer.outlier_threshold == 3.0

    def test_invalid_percentile(self) -> None:
        """Test invalid percentile raises error."""
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            PercentileAnalyzer(percentiles=[50, 110])

    def test_invalid_percentile_negative(self) -> None:
        """Test negative percentile raises error."""
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            PercentileAnalyzer(percentiles=[-5, 50])

    def test_invalid_outlier_method(self) -> None:
        """Test invalid outlier method raises error."""
        with pytest.raises(ValueError, match="outlier_method must be one of"):
            PercentileAnalyzer(outlier_method="invalid")

    def test_repr(self) -> None:
        """Test string representation."""
        analyzer = PercentileAnalyzer(percentiles=[25, 50, 75], outlier_method="zscore")

        repr_str = repr(analyzer)
        assert "PercentileAnalyzer" in repr_str
        assert "[25, 50, 75]" in repr_str
        assert "zscore" in repr_str


# =============================================================================
# Distribution Analysis Tests
# =============================================================================


class TestAnalyzeDistribution:
    """Tests for analyze_distribution method."""

    def test_basic_analysis(self, normal_data: np.ndarray, analyzer: PercentileAnalyzer) -> None:
        """Test basic distribution analysis."""
        result = analyzer.analyze_distribution(normal_data)

        assert isinstance(result, DistributionAnalysis)
        assert result.sample_size == len(normal_data)
        assert all(p in result.percentiles for p in DEFAULT_PERCENTILES)

    def test_percentile_values(self, analyzer: PercentileAnalyzer) -> None:
        """Test percentile values are correct."""
        data = np.arange(1, 101)  # 1 to 100
        result = analyzer.analyze_distribution(data, include_outliers=False)

        # P50 should be close to 50
        assert 49 < result.percentiles[50] < 52
        # P25 should be close to 25
        assert 24 < result.percentiles[25] < 27
        # P75 should be close to 75
        assert 74 < result.percentiles[75] < 77

    def test_mean_std(self, analyzer: PercentileAnalyzer) -> None:
        """Test mean and std calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyzer.analyze_distribution(data, include_outliers=False)

        assert result.mean == 3.0
        assert result.std == pytest.approx(np.std(data, ddof=1))

    def test_skewness_symmetric(self, analyzer: PercentileAnalyzer) -> None:
        """Test skewness for symmetric distribution."""
        np.random.seed(42)
        symmetric_data = np.random.normal(0, 1, 1000)
        result = analyzer.analyze_distribution(symmetric_data, include_outliers=False)

        # Skewness should be close to 0 for symmetric data
        assert abs(result.skewness) < 0.5

    def test_skewness_right_skewed(self, skewed_data: np.ndarray, analyzer: PercentileAnalyzer) -> None:
        """Test skewness for right-skewed distribution."""
        result = analyzer.analyze_distribution(skewed_data, include_outliers=False)

        # Exponential distribution is right-skewed (positive skewness)
        assert result.skewness > 0

    def test_kurtosis_normal(self, normal_data: np.ndarray, analyzer: PercentileAnalyzer) -> None:
        """Test kurtosis for normal distribution."""
        result = analyzer.analyze_distribution(normal_data, include_outliers=False)

        # Normal distribution has kurtosis of 3 (non-excess)
        assert 2.0 < result.kurtosis < 5.0

    def test_with_outliers_included(
        self, data_with_outliers: np.ndarray, analyzer: PercentileAnalyzer
    ) -> None:
        """Test analysis with outlier detection."""
        result = analyzer.analyze_distribution(data_with_outliers, include_outliers=True)

        assert result.outlier_analysis is not None
        assert result.outlier_count > 0

    def test_without_outliers(self, normal_data: np.ndarray, analyzer: PercentileAnalyzer) -> None:
        """Test analysis without outlier detection."""
        result = analyzer.analyze_distribution(normal_data, include_outliers=False)

        assert result.outlier_analysis is None
        assert result.outlier_count == 0

    def test_handles_nan_values(self, analyzer: PercentileAnalyzer) -> None:
        """Test handling of NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.inf])
        result = analyzer.analyze_distribution(data, include_outliers=False)

        assert result.sample_size == 4  # Only valid values
        assert result.metadata["n_excluded"] == 2

    def test_empty_data_error(self, analyzer: PercentileAnalyzer) -> None:
        """Test empty data raises error."""
        with pytest.raises(ValueError, match="No valid"):
            analyzer.analyze_distribution(np.array([]))

    def test_all_nan_error(self, analyzer: PercentileAnalyzer) -> None:
        """Test all NaN data raises error."""
        with pytest.raises(ValueError, match="No valid"):
            analyzer.analyze_distribution(np.array([np.nan, np.nan, np.nan]))

    def test_small_sample(self, analyzer: PercentileAnalyzer) -> None:
        """Test small sample handling."""
        # 2 samples - std calculation should use ddof=1
        data = np.array([1.0, 2.0])
        result = analyzer.analyze_distribution(data, include_outliers=False)

        assert result.sample_size == 2
        assert result.std == pytest.approx(np.std(data, ddof=1))
        # With only 2 samples, skewness and kurtosis default
        assert result.skewness == 0.0
        assert result.kurtosis == 3.0

    def test_single_value(self, analyzer: PercentileAnalyzer) -> None:
        """Test single value data."""
        data = np.array([5.0])
        result = analyzer.analyze_distribution(data, include_outliers=False)

        assert result.sample_size == 1
        assert result.mean == 5.0
        assert result.std == 0.0  # Single value has no variation


# =============================================================================
# Error Distribution Analysis Tests
# =============================================================================


class TestAnalyzeErrorDistribution:
    """Tests for analyze_error_distribution method."""

    def test_basic_error_analysis(
        self, forecast_data: tuple[np.ndarray, np.ndarray], analyzer: PercentileAnalyzer
    ) -> None:
        """Test basic error distribution analysis."""
        actual, forecast = forecast_data
        result = analyzer.analyze_error_distribution(actual, forecast)

        assert isinstance(result, DistributionAnalysis)
        assert result.sample_size == len(actual)

    def test_error_calculation(self, analyzer: PercentileAnalyzer) -> None:
        """Test errors are calculated correctly (forecast - actual)."""
        actual = np.array([100.0, 200.0, 300.0])
        forecast = np.array([110.0, 190.0, 305.0])
        # Expected errors: [10, -10, 5]

        result = analyzer.analyze_error_distribution(actual, forecast, include_outliers=False)

        # Mean error should be (10 - 10 + 5) / 3 = 1.67
        assert result.mean == pytest.approx(5.0 / 3, rel=1e-5)

    def test_shape_mismatch_error(self, analyzer: PercentileAnalyzer) -> None:
        """Test shape mismatch raises error."""
        actual = np.array([1.0, 2.0, 3.0])
        forecast = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            analyzer.analyze_error_distribution(actual, forecast)

    def test_2d_arrays(self, analyzer: PercentileAnalyzer) -> None:
        """Test 2D array shape mismatch."""
        actual = np.array([[1.0, 2.0], [3.0, 4.0]])
        forecast = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="Shape mismatch"):
            analyzer.analyze_error_distribution(actual, forecast)


# =============================================================================
# Normality Testing Tests
# =============================================================================


class TestNormalityTesting:
    """Tests for normality testing."""

    def test_normal_distribution(self, analyzer: PercentileAnalyzer) -> None:
        """Test normality detection for normal distribution."""
        np.random.seed(42)
        normal = np.random.normal(0, 1, 500)
        result = analyzer.analyze_distribution(normal, include_outliers=False)

        # Should pass normality tests
        assert "shapiro_wilk" in result.normality_test
        assert "anderson_darling" in result.normality_test
        assert "shapiro_wilk" in result.normality_test["tests_performed"]
        assert "anderson_darling" in result.normality_test["tests_performed"]

    def test_non_normal_distribution(self, skewed_data: np.ndarray, analyzer: PercentileAnalyzer) -> None:
        """Test normality detection for non-normal distribution."""
        result = analyzer.analyze_distribution(skewed_data, include_outliers=False)

        # Exponential distribution should fail normality tests
        assert result.normality_test["is_normal"] == False  # noqa: E712

    def test_large_sample_no_shapiro(self, analyzer: PercentileAnalyzer) -> None:
        """Test Shapiro-Wilk not used for large samples."""
        np.random.seed(42)
        large_data = np.random.normal(0, 1, 6000)
        result = analyzer.analyze_distribution(large_data, include_outliers=False)

        # Shapiro-Wilk should not be used for n >= 5000
        assert "shapiro_wilk" not in result.normality_test
        assert "anderson_darling" in result.normality_test

    def test_small_sample_normality(self, analyzer: PercentileAnalyzer) -> None:
        """Test normality testing with small sample."""
        data = np.array([1.0, 2.0])  # Only 2 samples
        result = analyzer.analyze_distribution(data, include_outliers=False)

        # Should have error message due to insufficient samples
        assert "error" in result.normality_test or len(result.normality_test["tests_performed"]) == 0


# =============================================================================
# Outlier Detection Tests
# =============================================================================


class TestOutlierDetection:
    """Tests for outlier detection methods."""

    def test_iqr_method(self, data_with_outliers: np.ndarray, analyzer: PercentileAnalyzer) -> None:
        """Test IQR outlier detection."""
        outliers = analyzer.detect_outliers(data_with_outliers, method="iqr")

        assert outliers.method == "iqr"
        assert outliers.n_outliers > 0
        assert outliers.lower_bound is not None
        assert outliers.upper_bound is not None

    def test_zscore_method(self, data_with_outliers: np.ndarray, analyzer: PercentileAnalyzer) -> None:
        """Test Z-score outlier detection."""
        outliers = analyzer.detect_outliers(data_with_outliers, method="zscore", threshold=3.0)

        assert outliers.method == "zscore"
        assert outliers.threshold_used == 3.0
        # Known outliers (10, -10, 15) should be detected
        assert outliers.n_outliers >= 2

    def test_modified_zscore_method(
        self, data_with_outliers: np.ndarray, analyzer: PercentileAnalyzer
    ) -> None:
        """Test modified Z-score outlier detection."""
        outliers = analyzer.detect_outliers(data_with_outliers, method="modified_zscore", threshold=3.5)

        assert outliers.method == "modified_zscore"
        assert outliers.n_outliers > 0

    def test_outlier_values_match_indices(self, data_with_outliers: np.ndarray, analyzer: PercentileAnalyzer) -> None:
        """Test outlier values match their indices."""
        outliers = analyzer.detect_outliers(data_with_outliers, method="iqr")

        for idx, val in zip(outliers.outlier_indices, outliers.outlier_values):
            assert data_with_outliers[idx] == val

    def test_iqr_bounds(self, analyzer: PercentileAnalyzer) -> None:
        """Test IQR bounds are calculated correctly."""
        data = np.arange(1, 101, dtype=float)  # 1 to 100
        outliers = analyzer.detect_outliers(data, method="iqr", threshold=1.5)

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        assert outliers.lower_bound == pytest.approx(q1 - 1.5 * iqr)
        assert outliers.upper_bound == pytest.approx(q3 + 1.5 * iqr)

    def test_zscore_bounds(self, analyzer: PercentileAnalyzer) -> None:
        """Test Z-score bounds are calculated correctly."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # Clear outlier
        outliers = analyzer.detect_outliers(data, method="zscore", threshold=2.0)

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        assert outliers.lower_bound == pytest.approx(mean - 2.0 * std)
        assert outliers.upper_bound == pytest.approx(mean + 2.0 * std)

    def test_no_variation_zscore(self, analyzer: PercentileAnalyzer) -> None:
        """Test Z-score with no variation."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        outliers = analyzer.detect_outliers(data, method="zscore")

        assert outliers.n_outliers == 0

    def test_no_variation_modified_zscore(self, analyzer: PercentileAnalyzer) -> None:
        """Test modified Z-score with no variation."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        outliers = analyzer.detect_outliers(data, method="modified_zscore")

        assert outliers.n_outliers == 0

    def test_small_data_no_outliers(self, analyzer: PercentileAnalyzer) -> None:
        """Test small dataset returns no outliers."""
        data = np.array([1.0, 2.0, 3.0])  # Less than 4 samples
        outliers = analyzer.detect_outliers(data)

        assert outliers.n_outliers == 0

    def test_default_method_used(self, analyzer: PercentileAnalyzer) -> None:
        """Test default method is used when none specified."""
        data = np.random.normal(0, 1, 100)
        outliers = analyzer.detect_outliers(data)

        assert outliers.method == "iqr"

    def test_custom_threshold(self, data_with_outliers: np.ndarray) -> None:
        """Test custom threshold affects detection."""
        analyzer = PercentileAnalyzer(outlier_method="iqr", outlier_threshold=3.0)

        # With higher threshold, fewer outliers should be detected
        outliers_high = analyzer.detect_outliers(data_with_outliers)

        analyzer_low = PercentileAnalyzer(outlier_method="iqr", outlier_threshold=1.0)
        outliers_low = analyzer_low.detect_outliers(data_with_outliers)

        assert outliers_low.n_outliers >= outliers_high.n_outliers

    def test_invalid_method_error(self, analyzer: PercentileAnalyzer) -> None:
        """Test invalid method raises error."""
        data = np.random.normal(0, 1, 100)
        with pytest.raises(ValueError, match="Unknown outlier method"):
            analyzer.detect_outliers(data, method="invalid_method")


# =============================================================================
# Compare Outlier Methods Tests
# =============================================================================


class TestCompareOutlierMethods:
    """Tests for compare_outlier_methods."""

    def test_returns_all_methods(
        self, data_with_outliers: np.ndarray, analyzer: PercentileAnalyzer
    ) -> None:
        """Test all methods are returned."""
        results = analyzer.compare_outlier_methods(data_with_outliers)

        assert "iqr" in results
        assert "zscore" in results
        assert "modified_zscore" in results

    def test_method_types(self, data_with_outliers: np.ndarray, analyzer: PercentileAnalyzer) -> None:
        """Test result types are correct."""
        results = analyzer.compare_outlier_methods(data_with_outliers)

        for method, analysis in results.items():
            assert isinstance(analysis, OutlierAnalysis)
            assert analysis.method == method

    def test_different_methods_may_differ(
        self, data_with_outliers: np.ndarray, analyzer: PercentileAnalyzer
    ) -> None:
        """Test different methods may detect different outliers."""
        results = analyzer.compare_outlier_methods(data_with_outliers)

        # Get outlier counts
        counts = {method: analysis.n_outliers for method, analysis in results.items()}

        # At least one method should detect outliers
        assert max(counts.values()) > 0


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_large_dataset(self, analyzer: PercentileAnalyzer) -> None:
        """Test with large dataset."""
        np.random.seed(42)
        large_data = np.random.normal(0, 1, 10000)
        result = analyzer.analyze_distribution(large_data)

        assert result.sample_size == 10000
        assert result.outlier_analysis is not None

    def test_extreme_values(self, analyzer: PercentileAnalyzer) -> None:
        """Test with extreme values."""
        data = np.array([1e-10, 1e10, 0, 1e10, 1e-10])
        result = analyzer.analyze_distribution(data)

        assert result.sample_size == 5
        assert np.isfinite(result.mean)

    def test_negative_values(self, analyzer: PercentileAnalyzer) -> None:
        """Test with all negative values."""
        data = np.array([-100.0, -50.0, -25.0, -10.0, -5.0])
        result = analyzer.analyze_distribution(data, include_outliers=False)

        assert result.mean < 0
        assert all(v < 0 for v in result.percentiles.values())

    def test_mixed_sign_values(self, analyzer: PercentileAnalyzer) -> None:
        """Test with mixed positive and negative values."""
        data = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = analyzer.analyze_distribution(data, include_outliers=False)

        assert result.mean == 0.0
        assert result.median == 0.0

    def test_custom_percentiles_preserved(self) -> None:
        """Test custom percentiles are used."""
        custom_percentiles = [1, 99]
        analyzer = PercentileAnalyzer(percentiles=custom_percentiles)
        data = np.random.normal(0, 1, 100)
        result = analyzer.analyze_distribution(data, include_outliers=False)

        assert 1 in result.percentiles
        assert 99 in result.percentiles
        assert 50 not in result.percentiles  # Default P50 not included

    def test_identical_values(self, analyzer: PercentileAnalyzer) -> None:
        """Test with all identical values."""
        data = np.array([5.0] * 100)
        result = analyzer.analyze_distribution(data)

        assert result.mean == 5.0
        assert result.std == 0.0
        assert result.median == 5.0
        assert result.iqr == 0.0

    def test_list_input(self, analyzer: PercentileAnalyzer) -> None:
        """Test with list input (should be converted to array)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = analyzer.analyze_distribution(data, include_outliers=False)

        assert result.sample_size == 5
        assert result.mean == 3.0


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_forecast_analysis_workflow(self) -> None:
        """Test complete forecast analysis workflow."""
        np.random.seed(42)

        # Generate realistic forecast data
        actual = np.random.uniform(100, 200, size=100)
        forecast = actual + np.random.normal(0, 5, size=100)

        # Add some outliers
        forecast[0] = actual[0] + 50
        forecast[1] = actual[1] - 40

        # Create analyzer and analyze
        analyzer = PercentileAnalyzer(
            percentiles=[5, 25, 50, 75, 95],
            outlier_method="iqr",
            outlier_threshold=1.5,
        )

        result = analyzer.analyze_error_distribution(actual, forecast)

        # Verify complete analysis
        assert result.sample_size == 100
        assert all(p in result.percentiles for p in [5, 25, 50, 75, 95])
        assert result.outlier_analysis is not None
        assert result.outlier_count >= 2  # At least our injected outliers
        assert "is_normal" in result.normality_test

        # Convert to dict for reporting
        d = result.to_dict()
        assert "percentiles" in d
        assert "outlier_analysis" in d

    def test_method_comparison_workflow(self) -> None:
        """Test outlier method comparison workflow."""
        np.random.seed(42)

        # Generate data with clear outliers
        data = np.random.normal(0, 1, 100)
        data[0] = 100  # Extreme outlier

        analyzer = PercentileAnalyzer()
        comparison = analyzer.compare_outlier_methods(data)

        # All methods should detect the extreme outlier
        for method, analysis in comparison.items():
            assert 0 in analysis.outlier_indices, f"{method} should detect index 0"
