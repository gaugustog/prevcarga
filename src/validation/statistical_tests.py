"""Statistical test suite for baseline validation.

This module provides statistical tests for validating baseline reproduction
and comparing system predictions against reference standards.

Key Features:
- Normality testing (Shapiro-Wilk, Anderson-Darling)
- Significance testing (paired t-test, Wilcoxon, Mann-Whitney)
- Effect size calculation (Cohen's d)
- Confidence interval computation
- Validation result aggregation

Example:
    ```python
    from src.validation.statistical_tests import StatisticalTestSuite
    import numpy as np

    suite = StatisticalTestSuite(confidence_level=0.95)

    # Test significance
    sample1 = np.random.normal(100, 10, 100)
    sample2 = np.random.normal(105, 10, 100)
    result = suite.test_significance(sample1, sample2)
    print(f"P-value: {result.p_value:.4f}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestType(Enum):
    """Statistical test types."""

    PAIRED_T_TEST = "paired_t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    SHAPIRO_WILK = "shapiro_wilk"
    ANDERSON_DARLING = "anderson_darling"
    LEVENE = "levene"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"


@dataclass
class StatisticalTestResult:
    """Results from statistical test.

    Attributes:
        test_type: Type of statistical test performed.
        statistic: Test statistic value.
        p_value: P-value from test.
        is_significant: Whether result is statistically significant.
        effect_size: Effect size (e.g., Cohen's d).
        confidence_interval: Confidence interval tuple.
        additional_info: Additional test-specific information.
    """

    test_type: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    confidence_interval: tuple[float, float]
    additional_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "test_type": self.test_type,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "effect_size": self.effect_size,
            "confidence_interval": self.confidence_interval,
            "additional_info": self.additional_info,
        }


@dataclass
class NormalityTestResult:
    """Results from normality test.

    Attributes:
        test_type: Type of normality test.
        statistic: Test statistic.
        p_value: P-value.
        is_normal: Whether sample is normally distributed.
        sample_size: Size of sample tested.
    """

    test_type: str
    statistic: float
    p_value: float
    is_normal: bool
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_type": self.test_type,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_normal": self.is_normal,
            "sample_size": self.sample_size,
        }


class StatisticalTestSuite:
    """Suite of statistical tests for validation.

    This class provides various statistical tests for validating baseline
    reproduction and comparing system predictions.

    Attributes:
        confidence_level: Confidence level for tests (default 0.95).
        alpha: Significance level (1 - confidence_level).
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        """Initialize test suite.

        Args:
            confidence_level: Confidence level for tests.

        Raises:
            ValueError: If confidence_level not in (0, 1).
        """
        if not 0 < confidence_level < 1:
            raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")

        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def validate_baseline(
        self,
        baseline_metrics: dict[str, dict[str, float]],
        tolerance_thresholds: dict[str, float],
    ) -> "ValidationResult":
        """Validate baseline reproduction against tolerance thresholds.

        Args:
            baseline_metrics: Calculated metrics from baseline.
            tolerance_thresholds: Acceptable tolerance thresholds.

        Returns:
            ValidationResult with overall validation status.
        """
        logger.info("Validating baseline against tolerance thresholds")

        validations: dict[str, dict[str, bool]] = {}

        mape_threshold = tolerance_thresholds.get("mape", 0.15)
        mae_threshold = tolerance_thresholds.get("mae", 200.0)
        rmse_threshold = tolerance_thresholds.get("rmse", 300.0)

        for key, metrics in baseline_metrics.items():
            mape_valid = metrics.get("mape", float("inf")) <= mape_threshold
            mae_valid = metrics.get("mae", float("inf")) <= mae_threshold
            rmse_valid = metrics.get("rmse", float("inf")) <= rmse_threshold

            validations[key] = {
                "mape_valid": mape_valid,
                "mae_valid": mae_valid,
                "rmse_valid": rmse_valid,
                "all_valid": mape_valid and mae_valid and rmse_valid,
            }

        overall_valid = all(v["all_valid"] for v in validations.values()) if validations else False

        summary = self._create_validation_summary(validations)

        logger.info(f"Validation complete. Overall valid: {overall_valid}")

        return ValidationResult(
            is_valid=overall_valid,
            validations=validations,
            summary=summary,
        )

    def test_significance(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        test_type: str = "paired_t_test",
    ) -> StatisticalTestResult:
        """Perform statistical significance testing.

        Args:
            sample1: First sample.
            sample2: Second sample.
            test_type: Type of test ('paired_t_test', 'wilcoxon', 'mann_whitney').

        Returns:
            StatisticalTestResult with test outcomes.

        Raises:
            ValueError: If unknown test type.
        """
        sample1 = np.asarray(sample1).flatten()
        sample2 = np.asarray(sample2).flatten()

        # Remove NaN values
        mask1 = ~np.isnan(sample1)
        mask2 = ~np.isnan(sample2)

        if test_type in ["paired_t_test", "wilcoxon"]:
            # Need paired samples - use common mask
            mask = mask1 & mask2
            sample1 = sample1[mask]
            sample2 = sample2[mask]
        else:
            sample1 = sample1[mask1]
            sample2 = sample2[mask2]

        if len(sample1) < 3 or len(sample2) < 3:
            logger.warning("Sample size too small for significance test")
            return StatisticalTestResult(
                test_type=test_type,
                statistic=np.nan,
                p_value=1.0,
                is_significant=False,
                effect_size=0.0,
                confidence_interval=(np.nan, np.nan),
                additional_info={"warning": "Sample size too small"},
            )

        if test_type == "paired_t_test":
            statistic, p_value = stats.ttest_rel(sample1, sample2)
        elif test_type == "wilcoxon":
            try:
                statistic, p_value = stats.wilcoxon(sample1, sample2)
            except ValueError:
                # All zeros or identical samples
                statistic, p_value = 0.0, 1.0
        elif test_type == "mann_whitney":
            statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        effect_size = self._calculate_effect_size(sample1, sample2)

        if test_type == "paired_t_test":
            diff = sample1 - sample2
            confidence_interval = self._calculate_confidence_interval(diff)
        else:
            confidence_interval = self._calculate_confidence_interval(sample1)

        return StatisticalTestResult(
            test_type=test_type,
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < self.alpha,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            additional_info={
                "sample1_size": len(sample1),
                "sample2_size": len(sample2),
                "sample1_mean": float(np.mean(sample1)),
                "sample2_mean": float(np.mean(sample2)),
            },
        )

    def test_normality(
        self,
        sample: np.ndarray,
        test_type: str = "shapiro_wilk",
    ) -> NormalityTestResult:
        """Test if sample follows normal distribution.

        Args:
            sample: Sample to test.
            test_type: Type of test ('shapiro_wilk' or 'anderson_darling').

        Returns:
            NormalityTestResult with test outcomes.
        """
        sample = np.asarray(sample).flatten()
        sample = sample[~np.isnan(sample)]

        if len(sample) < 3:
            logger.warning("Sample size too small for normality test")
            return NormalityTestResult(
                test_type=test_type,
                statistic=np.nan,
                p_value=1.0,
                is_normal=False,
                sample_size=len(sample),
            )

        if test_type == "shapiro_wilk":
            # Shapiro-Wilk limited to 5000 samples
            if len(sample) > 5000:
                sample = np.random.choice(sample, 5000, replace=False)
            statistic, p_value = stats.shapiro(sample)
        elif test_type == "anderson_darling":
            result = stats.anderson(sample, dist="norm")
            statistic = result.statistic
            # Use 5% significance level
            p_value = 0.05 if result.statistic > result.critical_values[2] else 0.10
        else:
            raise ValueError(f"Unknown normality test type: {test_type}")

        return NormalityTestResult(
            test_type=test_type,
            statistic=float(statistic),
            p_value=float(p_value),
            is_normal=p_value > self.alpha,
            sample_size=len(sample),
        )

    def test_variance_equality(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
    ) -> StatisticalTestResult:
        """Test equality of variances using Levene's test.

        Args:
            sample1: First sample.
            sample2: Second sample.

        Returns:
            StatisticalTestResult with test outcomes.
        """
        sample1 = np.asarray(sample1).flatten()
        sample2 = np.asarray(sample2).flatten()

        sample1 = sample1[~np.isnan(sample1)]
        sample2 = sample2[~np.isnan(sample2)]

        if len(sample1) < 3 or len(sample2) < 3:
            return StatisticalTestResult(
                test_type="levene",
                statistic=np.nan,
                p_value=1.0,
                is_significant=False,
                effect_size=0.0,
                confidence_interval=(np.nan, np.nan),
            )

        statistic, p_value = stats.levene(sample1, sample2)

        return StatisticalTestResult(
            test_type="levene",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < self.alpha,
            effect_size=0.0,  # Not applicable for variance test
            confidence_interval=(np.nan, np.nan),
            additional_info={
                "sample1_var": float(np.var(sample1)),
                "sample2_var": float(np.var(sample2)),
            },
        )

    def test_distribution_equality(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
    ) -> StatisticalTestResult:
        """Test distribution equality using Kolmogorov-Smirnov test.

        Args:
            sample1: First sample.
            sample2: Second sample.

        Returns:
            StatisticalTestResult with test outcomes.
        """
        sample1 = np.asarray(sample1).flatten()
        sample2 = np.asarray(sample2).flatten()

        sample1 = sample1[~np.isnan(sample1)]
        sample2 = sample2[~np.isnan(sample2)]

        if len(sample1) < 3 or len(sample2) < 3:
            return StatisticalTestResult(
                test_type="kolmogorov_smirnov",
                statistic=np.nan,
                p_value=1.0,
                is_significant=False,
                effect_size=0.0,
                confidence_interval=(np.nan, np.nan),
            )

        statistic, p_value = stats.ks_2samp(sample1, sample2)

        return StatisticalTestResult(
            test_type="kolmogorov_smirnov",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < self.alpha,
            effect_size=self._calculate_effect_size(sample1, sample2),
            confidence_interval=(np.nan, np.nan),
        )

    def _calculate_effect_size(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
    ) -> float:
        """Calculate Cohen's d effect size.

        Args:
            sample1: First sample.
            sample2: Second sample.

        Returns:
            Effect size (Cohen's d).
        """
        mean_diff = np.mean(sample1) - np.mean(sample2)
        pooled_std = np.sqrt((np.var(sample1, ddof=1) + np.var(sample2, ddof=1)) / 2)
        return float(mean_diff / pooled_std) if pooled_std > 0 else 0.0

    def _calculate_confidence_interval(
        self,
        sample: np.ndarray,
    ) -> tuple[float, float]:
        """Calculate confidence interval for sample mean.

        Args:
            sample: Sample data.

        Returns:
            Tuple of (lower, upper) bounds.
        """
        if len(sample) < 2:
            return (np.nan, np.nan)

        mean = np.mean(sample)
        se = stats.sem(sample)

        if np.isnan(se) or se == 0:
            return (mean, mean)

        ci = stats.t.interval(
            self.confidence_level,
            len(sample) - 1,
            loc=mean,
            scale=se,
        )
        return (float(ci[0]), float(ci[1]))

    def _create_validation_summary(
        self,
        validations: dict[str, dict[str, bool]],
    ) -> dict[str, float]:
        """Create summary statistics from validations.

        Args:
            validations: Dictionary of validation results.

        Returns:
            Summary statistics dictionary.
        """
        if not validations:
            return {
                "total_combinations": 0,
                "valid_combinations": 0,
                "pass_rate": 0.0,
                "mape_pass_rate": 0.0,
                "mae_pass_rate": 0.0,
                "rmse_pass_rate": 0.0,
            }

        total = len(validations)
        valid = sum(1 for v in validations.values() if v["all_valid"])
        mape_valid = sum(1 for v in validations.values() if v.get("mape_valid", False))
        mae_valid = sum(1 for v in validations.values() if v.get("mae_valid", False))
        rmse_valid = sum(1 for v in validations.values() if v.get("rmse_valid", False))

        return {
            "total_combinations": total,
            "valid_combinations": valid,
            "pass_rate": valid / total if total > 0 else 0.0,
            "mape_pass_rate": mape_valid / total if total > 0 else 0.0,
            "mae_pass_rate": mae_valid / total if total > 0 else 0.0,
            "rmse_pass_rate": rmse_valid / total if total > 0 else 0.0,
        }


@dataclass
class ValidationResult:
    """Result of baseline validation.

    Attributes:
        is_valid: Whether overall validation passed.
        validations: Per-combination validation results.
        summary: Summary statistics.
    """

    is_valid: bool
    validations: dict[str, dict[str, bool]]
    summary: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "is_valid": self.is_valid,
            "validations": self.validations,
            "summary": self.summary,
        }
