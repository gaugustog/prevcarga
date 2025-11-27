"""Tests for the statistical tests module.

Tests cover:
- StatisticalTestResult dataclass
- NormalityTestResult dataclass
- ValidationResult dataclass
- StatisticalTestSuite class methods
"""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.statistical_tests import (
    NormalityTestResult,
    StatisticalTestResult,
    StatisticalTestSuite,
    TestType,
    ValidationResult,
)


class TestTestType:
    """Tests for TestType enum."""

    def test_all_types_defined(self) -> None:
        """Test all test types are defined."""
        expected = [
            "PAIRED_T_TEST",
            "WILCOXON",
            "MANN_WHITNEY",
            "SHAPIRO_WILK",
            "ANDERSON_DARLING",
            "LEVENE",
            "KOLMOGOROV_SMIRNOV",
        ]
        for name in expected:
            assert hasattr(TestType, name)


class TestStatisticalTestResult:
    """Tests for StatisticalTestResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating test result."""
        result = StatisticalTestResult(
            test_type="paired_t_test",
            statistic=2.5,
            p_value=0.02,
            is_significant=True,
            effect_size=0.5,
            confidence_interval=(-0.1, 0.9),
        )

        assert result.test_type == "paired_t_test"
        assert result.statistic == 2.5
        assert result.p_value == 0.02
        assert result.is_significant is True
        assert result.effect_size == 0.5
        assert result.confidence_interval == (-0.1, 0.9)

    def test_result_to_dict(self) -> None:
        """Test to_dict method."""
        result = StatisticalTestResult(
            test_type="wilcoxon",
            statistic=100,
            p_value=0.05,
            is_significant=True,
            effect_size=0.3,
            confidence_interval=(0.1, 0.5),
            additional_info={"n": 50},
        )

        d = result.to_dict()
        assert d["test_type"] == "wilcoxon"
        assert d["statistic"] == 100
        assert d["p_value"] == 0.05
        assert d["is_significant"] is True
        assert d["effect_size"] == 0.3
        assert d["confidence_interval"] == (0.1, 0.5)
        assert d["additional_info"] == {"n": 50}


class TestNormalityTestResult:
    """Tests for NormalityTestResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating normality test result."""
        result = NormalityTestResult(
            test_type="shapiro_wilk",
            statistic=0.98,
            p_value=0.15,
            is_normal=True,
            sample_size=100,
        )

        assert result.test_type == "shapiro_wilk"
        assert result.statistic == 0.98
        assert result.p_value == 0.15
        assert result.is_normal is True
        assert result.sample_size == 100

    def test_result_to_dict(self) -> None:
        """Test to_dict method."""
        result = NormalityTestResult(
            test_type="anderson_darling",
            statistic=0.5,
            p_value=0.10,
            is_normal=True,
            sample_size=50,
        )

        d = result.to_dict()
        assert d["test_type"] == "anderson_darling"
        assert "statistic" in d
        assert "p_value" in d
        assert "is_normal" in d
        assert "sample_size" in d


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating validation result."""
        result = ValidationResult(
            is_valid=True,
            validations={
                "SECO_lgbm": {"mape_valid": True, "all_valid": True},
                "S_lgbm": {"mape_valid": True, "all_valid": True},
            },
            summary={"pass_rate": 1.0},
        )

        assert result.is_valid is True
        assert len(result.validations) == 2
        assert result.summary["pass_rate"] == 1.0

    def test_result_to_dict(self) -> None:
        """Test to_dict method."""
        result = ValidationResult(
            is_valid=False,
            validations={"key": {"valid": False}},
            summary={"pass_rate": 0.0},
        )

        d = result.to_dict()
        assert d["is_valid"] is False
        assert "validations" in d
        assert "summary" in d


class TestStatisticalTestSuite:
    """Tests for StatisticalTestSuite class."""

    @pytest.fixture
    def suite(self) -> StatisticalTestSuite:
        """Create test suite."""
        return StatisticalTestSuite(confidence_level=0.95)

    def test_init(self, suite: StatisticalTestSuite) -> None:
        """Test suite initialization."""
        assert suite.confidence_level == 0.95
        assert suite.alpha == pytest.approx(0.05)

    def test_init_invalid_confidence_level(self) -> None:
        """Test init with invalid confidence level."""
        with pytest.raises(ValueError):
            StatisticalTestSuite(confidence_level=1.5)

        with pytest.raises(ValueError):
            StatisticalTestSuite(confidence_level=0)

    def test_validate_baseline_all_valid(self, suite: StatisticalTestSuite) -> None:
        """Test baseline validation when all valid."""
        baseline_metrics = {
            "SECO_lgbm": {"mape": 0.03, "mae": 50, "rmse": 80},
            "S_lgbm": {"mape": 0.04, "mae": 60, "rmse": 90},
        }
        tolerance_thresholds = {"mape": 0.05, "mae": 100, "rmse": 150}

        result = suite.validate_baseline(baseline_metrics, tolerance_thresholds)

        assert result.is_valid is True
        assert result.summary["pass_rate"] == 1.0

    def test_validate_baseline_some_invalid(self, suite: StatisticalTestSuite) -> None:
        """Test baseline validation when some invalid."""
        baseline_metrics = {
            "SECO_lgbm": {"mape": 0.03, "mae": 50, "rmse": 80},
            "S_lgbm": {"mape": 0.10, "mae": 250, "rmse": 300},  # Exceeds
        }
        tolerance_thresholds = {"mape": 0.05, "mae": 100, "rmse": 150}

        result = suite.validate_baseline(baseline_metrics, tolerance_thresholds)

        assert result.is_valid is False
        assert result.summary["pass_rate"] == 0.5

    def test_validate_baseline_empty_metrics(self, suite: StatisticalTestSuite) -> None:
        """Test baseline validation with empty metrics."""
        result = suite.validate_baseline({}, {})

        assert result.is_valid is False
        assert result.summary["pass_rate"] == 0.0

    def test_test_significance_paired_t_test(self, suite: StatisticalTestSuite) -> None:
        """Test paired t-test."""
        np.random.seed(42)
        sample1 = np.random.normal(100, 10, 50)
        sample2 = np.random.normal(100, 10, 50)

        result = suite.test_significance(sample1, sample2, test_type="paired_t_test")

        assert result.test_type == "paired_t_test"
        assert 0 <= result.p_value <= 1
        assert result.is_significant in (True, False)  # numpy bool compatible
        assert isinstance(result.effect_size, (float, np.floating))

    def test_test_significance_wilcoxon(self, suite: StatisticalTestSuite) -> None:
        """Test Wilcoxon signed-rank test."""
        np.random.seed(42)
        sample1 = np.random.normal(100, 10, 50)
        sample2 = sample1 + np.random.normal(0, 5, 50)

        result = suite.test_significance(sample1, sample2, test_type="wilcoxon")

        assert result.test_type == "wilcoxon"
        assert 0 <= result.p_value <= 1

    def test_test_significance_mann_whitney(self, suite: StatisticalTestSuite) -> None:
        """Test Mann-Whitney U test."""
        np.random.seed(42)
        sample1 = np.random.normal(100, 10, 50)
        sample2 = np.random.normal(105, 10, 50)

        result = suite.test_significance(sample1, sample2, test_type="mann_whitney")

        assert result.test_type == "mann_whitney"
        assert 0 <= result.p_value <= 1

    def test_test_significance_unknown_type(self, suite: StatisticalTestSuite) -> None:
        """Test with unknown test type."""
        sample1 = np.array([1, 2, 3])
        sample2 = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Unknown test type"):
            suite.test_significance(sample1, sample2, test_type="unknown")

    def test_test_significance_small_sample(self, suite: StatisticalTestSuite) -> None:
        """Test with small sample size."""
        sample1 = np.array([1, 2])
        sample2 = np.array([1, 2])

        result = suite.test_significance(sample1, sample2)

        assert np.isnan(result.statistic)
        assert result.p_value == 1.0
        assert result.is_significant is False

    def test_test_significance_with_nans(self, suite: StatisticalTestSuite) -> None:
        """Test with NaN values in samples."""
        sample1 = np.array([1, 2, 3, np.nan, 5, 6, 7, 8])
        sample2 = np.array([1, 2, np.nan, 4, 5, 6, 7, 8])

        result = suite.test_significance(sample1, sample2)

        # Should handle NaNs (but may still have issues with small samples after filtering)
        # Just verify it runs without error
        assert result.test_type == "paired_t_test"

    def test_test_normality_shapiro_wilk(self, suite: StatisticalTestSuite) -> None:
        """Test Shapiro-Wilk normality test."""
        np.random.seed(42)
        sample = np.random.normal(100, 10, 100)

        result = suite.test_normality(sample, test_type="shapiro_wilk")

        assert result.test_type == "shapiro_wilk"
        assert 0 <= result.p_value <= 1
        assert result.sample_size == 100
        assert result.is_normal  # Normal data should pass

    def test_test_normality_anderson_darling(self, suite: StatisticalTestSuite) -> None:
        """Test Anderson-Darling normality test."""
        np.random.seed(42)
        sample = np.random.normal(100, 10, 100)

        result = suite.test_normality(sample, test_type="anderson_darling")

        assert result.test_type == "anderson_darling"

    def test_test_normality_non_normal(self, suite: StatisticalTestSuite) -> None:
        """Test normality with non-normal data."""
        np.random.seed(42)
        # Exponential distribution is not normal
        sample = np.random.exponential(10, 100)

        result = suite.test_normality(sample)

        assert not result.is_normal

    def test_test_normality_small_sample(self, suite: StatisticalTestSuite) -> None:
        """Test normality with small sample."""
        sample = np.array([1, 2])

        result = suite.test_normality(sample)

        assert np.isnan(result.statistic)
        assert result.is_normal is False

    def test_test_normality_unknown_type(self, suite: StatisticalTestSuite) -> None:
        """Test normality with unknown test type."""
        with pytest.raises(ValueError, match="Unknown normality test type"):
            suite.test_normality(np.array([1, 2, 3]), test_type="unknown")

    def test_test_normality_large_sample(self, suite: StatisticalTestSuite) -> None:
        """Test normality with large sample (>5000)."""
        np.random.seed(42)
        sample = np.random.normal(100, 10, 10000)

        result = suite.test_normality(sample)

        # Should downsample to 5000
        assert result.sample_size == 5000

    def test_test_variance_equality(self, suite: StatisticalTestSuite) -> None:
        """Test Levene's test for variance equality."""
        np.random.seed(42)
        sample1 = np.random.normal(100, 10, 50)
        sample2 = np.random.normal(100, 10, 50)

        result = suite.test_variance_equality(sample1, sample2)

        assert result.test_type == "levene"
        assert 0 <= result.p_value <= 1

    def test_test_variance_equality_different_variance(
        self, suite: StatisticalTestSuite
    ) -> None:
        """Test Levene's test with different variances."""
        np.random.seed(42)
        sample1 = np.random.normal(100, 10, 100)
        sample2 = np.random.normal(100, 50, 100)  # Much larger variance

        result = suite.test_variance_equality(sample1, sample2)

        assert result.is_significant  # numpy bool compatible

    def test_test_variance_equality_small_sample(
        self, suite: StatisticalTestSuite
    ) -> None:
        """Test Levene's test with small sample."""
        sample1 = np.array([1, 2])
        sample2 = np.array([1, 2])

        result = suite.test_variance_equality(sample1, sample2)

        assert np.isnan(result.statistic)

    def test_test_distribution_equality(self, suite: StatisticalTestSuite) -> None:
        """Test K-S test for distribution equality."""
        np.random.seed(42)
        sample1 = np.random.normal(100, 10, 50)
        sample2 = np.random.normal(100, 10, 50)

        result = suite.test_distribution_equality(sample1, sample2)

        assert result.test_type == "kolmogorov_smirnov"
        assert 0 <= result.p_value <= 1

    def test_test_distribution_equality_different_distributions(
        self, suite: StatisticalTestSuite
    ) -> None:
        """Test K-S test with different distributions."""
        np.random.seed(42)
        sample1 = np.random.normal(100, 10, 100)
        sample2 = np.random.exponential(100, 100)

        result = suite.test_distribution_equality(sample1, sample2)

        assert result.is_significant  # numpy bool compatible

    def test_calculate_effect_size(self, suite: StatisticalTestSuite) -> None:
        """Test effect size calculation."""
        sample1 = np.array([100, 110, 120, 130, 140])
        sample2 = np.array([90, 100, 110, 120, 130])

        effect_size = suite._calculate_effect_size(sample1, sample2)

        # Mean difference is 10, std is same
        assert effect_size > 0

    def test_calculate_effect_size_zero_std(self, suite: StatisticalTestSuite) -> None:
        """Test effect size with zero std."""
        sample1 = np.array([100, 100, 100])
        sample2 = np.array([100, 100, 100])

        effect_size = suite._calculate_effect_size(sample1, sample2)

        assert effect_size == 0.0

    def test_calculate_confidence_interval(self, suite: StatisticalTestSuite) -> None:
        """Test confidence interval calculation."""
        np.random.seed(42)
        sample = np.random.normal(100, 10, 100)

        ci = suite._calculate_confidence_interval(sample)

        assert len(ci) == 2
        assert ci[0] < ci[1]
        # Mean should be within CI
        assert ci[0] < np.mean(sample) < ci[1]

    def test_calculate_confidence_interval_small_sample(
        self, suite: StatisticalTestSuite
    ) -> None:
        """Test CI with very small sample."""
        sample = np.array([100])

        ci = suite._calculate_confidence_interval(sample)

        assert np.isnan(ci[0]) and np.isnan(ci[1])

    def test_create_validation_summary_normal(
        self, suite: StatisticalTestSuite
    ) -> None:
        """Test validation summary creation."""
        validations = {
            "key1": {"mape_valid": True, "mae_valid": True, "rmse_valid": True, "all_valid": True},
            "key2": {"mape_valid": True, "mae_valid": False, "rmse_valid": True, "all_valid": False},
            "key3": {"mape_valid": False, "mae_valid": False, "rmse_valid": False, "all_valid": False},
        }

        summary = suite._create_validation_summary(validations)

        assert summary["total_combinations"] == 3
        assert summary["valid_combinations"] == 1
        assert summary["pass_rate"] == pytest.approx(1 / 3)
        assert summary["mape_pass_rate"] == pytest.approx(2 / 3)

    def test_create_validation_summary_empty(self, suite: StatisticalTestSuite) -> None:
        """Test validation summary with empty data."""
        summary = suite._create_validation_summary({})

        assert summary["total_combinations"] == 0
        assert summary["pass_rate"] == 0.0


class TestSignificanceWithRealPatterns:
    """Test significance tests with realistic data patterns."""

    @pytest.fixture
    def suite(self) -> StatisticalTestSuite:
        """Create test suite."""
        return StatisticalTestSuite(confidence_level=0.95)

    def test_detect_significant_difference(self, suite: StatisticalTestSuite) -> None:
        """Test detection of significant difference."""
        np.random.seed(42)
        # Two clearly different distributions
        sample1 = np.random.normal(100, 10, 100)
        sample2 = np.random.normal(120, 10, 100)  # 20 unit difference

        result = suite.test_significance(sample1, sample2, test_type="mann_whitney")

        assert result.is_significant  # numpy bool compatible
        assert abs(result.effect_size) > 1  # Large effect

    def test_no_significant_difference(self, suite: StatisticalTestSuite) -> None:
        """Test no significant difference detected."""
        np.random.seed(42)
        # Same distribution
        sample1 = np.random.normal(100, 10, 100)
        sample2 = np.random.normal(100, 10, 100)

        result = suite.test_significance(sample1, sample2)

        # Not guaranteed to be non-significant, but effect should be small
        assert abs(result.effect_size) < 0.5
