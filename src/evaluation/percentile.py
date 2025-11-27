"""Percentile analysis framework for forecast error distributions.

This module provides the PercentileAnalyzer class that performs detailed
distribution analysis of forecast errors including percentile calculations,
normality testing, and outlier detection.

Key Features:
- Percentile calculations (P5-P95)
- Normality testing (Shapiro-Wilk, Anderson-Darling)
- Multiple outlier detection methods (IQR, Z-score, modified Z-score)
- Skewness and kurtosis calculations
- Visualization-ready data structures

Example:
    ```python
    from src.evaluation.percentile import PercentileAnalyzer
    import numpy as np

    # Create analyzer
    analyzer = PercentileAnalyzer()

    # Analyze error distribution
    actual = np.array([100, 200, 300, 400, 500])
    forecast = np.array([95, 210, 290, 410, 480])
    result = analyzer.analyze_error_distribution(actual, forecast)

    print(f"Median error: {result.percentiles[50]:.2f}")
    print(f"Is normal: {result.normality_test['is_normal']}")
    ```
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default percentile levels
DEFAULT_PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


@dataclass
class OutlierAnalysis:
    """Results from outlier detection.

    Attributes:
        outlier_indices: Indices of detected outliers in the original array.
        outlier_values: The actual values that are outliers.
        outlier_scores: Scores indicating how extreme each outlier is.
        threshold_used: The threshold value used for detection.
        method: The detection method used ("iqr", "zscore", "modified_zscore").
        lower_bound: Lower bound for outlier detection (if applicable).
        upper_bound: Upper bound for outlier detection (if applicable).

    Example:
        >>> analysis = OutlierAnalysis(
        ...     outlier_indices=[2, 15, 23],
        ...     outlier_values=np.array([100.5, -50.2, 120.3]),
        ...     outlier_scores=np.array([3.5, 4.2, 3.8]),
        ...     threshold_used=3.0,
        ...     method="zscore"
        ... )
    """

    outlier_indices: list[int]
    outlier_values: np.ndarray
    outlier_scores: np.ndarray
    threshold_used: float
    method: str
    lower_bound: float | None = None
    upper_bound: float | None = None

    @property
    def n_outliers(self) -> int:
        """Get number of outliers detected.

        Returns:
            Number of outliers.
        """
        return len(self.outlier_indices)

    @property
    def outlier_rate(self) -> float:
        """Get outlier rate (assuming known total count).

        Note: This requires knowing the total sample size, which
        is stored in metadata if available.

        Returns:
            Outlier rate as a fraction.
        """
        if hasattr(self, "_total_count") and self._total_count > 0:
            return self.n_outliers / self._total_count
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "outlier_indices": self.outlier_indices,
            "outlier_values": self.outlier_values.tolist(),
            "outlier_scores": self.outlier_scores.tolist(),
            "threshold_used": self.threshold_used,
            "method": self.method,
            "n_outliers": self.n_outliers,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return f"OutlierAnalysis(method={self.method!r}, n_outliers={self.n_outliers}, threshold={self.threshold_used})"


@dataclass
class DistributionAnalysis:
    """Results from distribution analysis.

    Attributes:
        percentiles: Dictionary mapping percentile levels to values.
        mean: Mean of the distribution.
        std: Standard deviation.
        skewness: Skewness (0 = symmetric, >0 = right-skewed, <0 = left-skewed).
        kurtosis: Kurtosis (3 = normal, >3 = heavy-tailed, <3 = light-tailed).
        normality_test: Results from normality tests.
        outlier_analysis: Optional outlier detection results.
        sample_size: Number of samples in the analysis.
        metadata: Additional analysis metadata.

    Example:
        >>> analysis = DistributionAnalysis(
        ...     percentiles={5: -10.2, 50: 0.5, 95: 12.3},
        ...     mean=0.3,
        ...     std=5.2,
        ...     skewness=0.1,
        ...     kurtosis=3.0,
        ...     normality_test={"is_normal": True, "shapiro_p": 0.12},
        ...     sample_size=100
        ... )
    """

    percentiles: dict[int, float]
    mean: float
    std: float
    skewness: float
    kurtosis: float
    normality_test: dict[str, Any]
    outlier_analysis: OutlierAnalysis | None = None
    sample_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def median(self) -> float:
        """Get median (P50).

        Returns:
            Median value.
        """
        return self.percentiles.get(50, np.nan)

    @property
    def iqr(self) -> float:
        """Get interquartile range (P75 - P25).

        Returns:
            IQR value.
        """
        q1 = self.percentiles.get(25, np.nan)
        q3 = self.percentiles.get(75, np.nan)
        return q3 - q1

    @property
    def is_normal(self) -> bool:
        """Check if distribution appears normal.

        Returns:
            True if normality tests suggest normal distribution.
        """
        return self.normality_test.get("is_normal", False)

    @property
    def outlier_count(self) -> int:
        """Get number of outliers detected.

        Returns:
            Number of outliers, or 0 if no outlier analysis.
        """
        if self.outlier_analysis is not None:
            return self.outlier_analysis.n_outliers
        return 0

    @property
    def outlier_indices(self) -> list[int]:
        """Get indices of outliers.

        Returns:
            List of outlier indices, or empty list.
        """
        if self.outlier_analysis is not None:
            return self.outlier_analysis.outlier_indices
        return []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "percentiles": self.percentiles.copy(),
            "mean": self.mean,
            "std": self.std,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "normality_test": self.normality_test.copy(),
            "outlier_analysis": self.outlier_analysis.to_dict() if self.outlier_analysis else None,
            "sample_size": self.sample_size,
            "metadata": self.metadata.copy(),
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return (
            f"DistributionAnalysis("
            f"mean={self.mean:.2f}, "
            f"std={self.std:.2f}, "
            f"median={self.median:.2f}, "
            f"skewness={self.skewness:.2f}, "
            f"n={self.sample_size})"
        )


class PercentileAnalyzer:
    """Percentile analysis framework for forecast errors.

    Provides detailed distribution analysis including percentile
    calculations, normality testing, and outlier detection.

    Supported Features:
        - Configurable percentile levels (default: P5-P95)
        - Normality testing (Shapiro-Wilk, Anderson-Darling)
        - Three outlier detection methods (IQR, Z-score, modified Z-score)
        - Distribution shape analysis (skewness, kurtosis)

    Example:
        >>> analyzer = PercentileAnalyzer()

        >>> # Analyze error distribution
        >>> errors = forecast - actual
        >>> result = analyzer.analyze_distribution(errors)
        >>> print(f"P95 error: {result.percentiles[95]:.2f}")

        >>> # Detect outliers
        >>> outliers = analyzer.detect_outliers(errors, method="iqr")
        >>> print(f"Found {outliers.n_outliers} outliers")
    """

    def __init__(
        self,
        percentiles: list[int] | None = None,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
    ) -> None:
        """Initialize analyzer.

        Args:
            percentiles: List of percentile levels to calculate.
                Defaults to [5, 10, 25, 50, 75, 90, 95].
            outlier_method: Default outlier detection method.
                One of "iqr", "zscore", "modified_zscore".
            outlier_threshold: Default threshold for outlier detection.
                For IQR: multiplier (default 1.5).
                For Z-score: number of standard deviations (default 3.0).
                For modified Z-score: threshold (default 3.5).
        """
        self.percentiles = percentiles or DEFAULT_PERCENTILES.copy()
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

        # Validate percentiles
        for p in self.percentiles:
            if not 0 <= p <= 100:
                msg = f"Percentile must be between 0 and 100, got {p}"
                raise ValueError(msg)

        # Validate outlier method
        valid_methods = {"iqr", "zscore", "modified_zscore"}
        if outlier_method not in valid_methods:
            msg = f"outlier_method must be one of {valid_methods}, got {outlier_method}"
            raise ValueError(msg)

        logger.debug(
            "Initialized PercentileAnalyzer: percentiles=%s, outlier_method=%s",
            self.percentiles,
            outlier_method,
        )

    def analyze_error_distribution(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
        include_outliers: bool = True,
    ) -> DistributionAnalysis:
        """Analyze distribution of forecast errors.

        Calculates errors as (forecast - actual) and performs
        comprehensive distribution analysis.

        Args:
            actual: Array of actual values.
            forecast: Array of forecast values.
            include_outliers: Whether to include outlier detection.

        Returns:
            DistributionAnalysis with all statistics.

        Raises:
            ValueError: If arrays have different lengths or are empty.
        """
        actual = np.asarray(actual, dtype=np.float64)
        forecast = np.asarray(forecast, dtype=np.float64)

        if actual.shape != forecast.shape:
            msg = f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}"
            raise ValueError(msg)

        # Calculate errors
        errors = forecast - actual

        return self.analyze_distribution(errors, include_outliers=include_outliers)

    def analyze_distribution(
        self,
        data: np.ndarray,
        include_outliers: bool = True,
    ) -> DistributionAnalysis:
        """Analyze distribution of data.

        Args:
            data: Array of values to analyze.
            include_outliers: Whether to include outlier detection.

        Returns:
            DistributionAnalysis with all statistics.

        Raises:
            ValueError: If data is empty.
        """
        data = np.asarray(data, dtype=np.float64)

        # Remove NaN values
        valid_data = data[np.isfinite(data)]
        n_excluded = len(data) - len(valid_data)

        if len(valid_data) == 0:
            msg = "No valid (non-NaN) values in data"
            raise ValueError(msg)

        if n_excluded > 0:
            logger.warning("Excluded %d NaN/Inf values from analysis", n_excluded)

        # Calculate percentiles
        percentile_values = {}
        for p in self.percentiles:
            percentile_values[p] = float(np.percentile(valid_data, p))

        # Calculate basic statistics
        mean = float(np.mean(valid_data))
        std = float(np.std(valid_data, ddof=1)) if len(valid_data) > 1 else 0.0

        # Calculate skewness and kurtosis
        if len(valid_data) >= 3:
            skewness = float(stats.skew(valid_data))
            kurtosis = float(stats.kurtosis(valid_data, fisher=False))  # Non-excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 3.0  # Normal distribution

        # Normality testing
        normality_test = self._test_normality(valid_data)

        # Outlier detection
        outlier_analysis = None
        if include_outliers and len(valid_data) >= 4:
            outlier_analysis = self.detect_outliers(
                valid_data,
                method=self.outlier_method,
                threshold=self.outlier_threshold,
            )

        return DistributionAnalysis(
            percentiles=percentile_values,
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            normality_test=normality_test,
            outlier_analysis=outlier_analysis,
            sample_size=len(valid_data),
            metadata={"n_excluded": n_excluded},
        )

    def _test_normality(self, data: np.ndarray) -> dict[str, Any]:
        """Test normality of distribution.

        Uses Shapiro-Wilk for n < 5000 and Anderson-Darling for all sizes.

        Args:
            data: Array of values to test.

        Returns:
            Dictionary with test results.
        """
        result: dict[str, Any] = {
            "is_normal": False,
            "tests_performed": [],
        }

        n = len(data)

        if n < 3:
            result["error"] = "Insufficient samples for normality testing (need >= 3)"
            return result

        # Shapiro-Wilk test (n < 5000)
        if n < 5000:
            try:
                stat, p_value = stats.shapiro(data)
                result["shapiro_wilk"] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05,
                }
                result["tests_performed"].append("shapiro_wilk")
            except Exception as e:
                result["shapiro_wilk_error"] = str(e)

        # Anderson-Darling test (all sample sizes)
        try:
            ad_result = stats.anderson(data, dist="norm")
            # Use 5% significance level (index 2)
            is_normal_ad = ad_result.statistic < ad_result.critical_values[2]
            result["anderson_darling"] = {
                "statistic": float(ad_result.statistic),
                "critical_values": {
                    f"{sl}%": float(cv)
                    for sl, cv in zip(ad_result.significance_level, ad_result.critical_values)
                },
                "is_normal": is_normal_ad,
            }
            result["tests_performed"].append("anderson_darling")
        except Exception as e:
            result["anderson_darling_error"] = str(e)

        # Overall normality decision
        # If both tests performed, require both to indicate normality
        # If only one, use that one
        normality_indicators = []
        if "shapiro_wilk" in result:
            normality_indicators.append(result["shapiro_wilk"]["is_normal"])
        if "anderson_darling" in result:
            normality_indicators.append(result["anderson_darling"]["is_normal"])

        if normality_indicators:
            result["is_normal"] = all(normality_indicators)
        else:
            result["is_normal"] = False

        return result

    def detect_outliers(
        self,
        data: np.ndarray,
        method: str | None = None,
        threshold: float | None = None,
    ) -> OutlierAnalysis:
        """Detect outliers in data.

        Args:
            data: Array of values.
            method: Detection method. One of "iqr", "zscore", "modified_zscore".
                If None, uses default method from initialization.
            threshold: Threshold for detection.
                If None, uses default threshold from initialization.

        Returns:
            OutlierAnalysis with detected outliers.

        Raises:
            ValueError: If invalid method or insufficient data.
        """
        data = np.asarray(data, dtype=np.float64)
        method = method or self.outlier_method
        threshold = threshold if threshold is not None else self.outlier_threshold

        if len(data) < 4:
            # Not enough data for meaningful outlier detection
            return OutlierAnalysis(
                outlier_indices=[],
                outlier_values=np.array([]),
                outlier_scores=np.array([]),
                threshold_used=threshold,
                method=method,
            )

        if method == "iqr":
            return self._detect_outliers_iqr(data, threshold)
        elif method == "zscore":
            return self._detect_outliers_zscore(data, threshold)
        elif method == "modified_zscore":
            return self._detect_outliers_modified_zscore(data, threshold)
        else:
            msg = f"Unknown outlier method: {method}"
            raise ValueError(msg)

    def _detect_outliers_iqr(
        self,
        data: np.ndarray,
        threshold: float = 1.5,
    ) -> OutlierAnalysis:
        """Detect outliers using IQR method.

        Outliers are values outside [Q1 - threshold*IQR, Q3 + threshold*IQR].

        Args:
            data: Array of values.
            threshold: IQR multiplier (default 1.5).

        Returns:
            OutlierAnalysis with results.
        """
        q1 = float(np.percentile(data, 25))
        q3 = float(np.percentile(data, 75))
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        # Find outliers
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = list(np.where(outlier_mask)[0])
        outlier_values = data[outlier_mask]

        # Calculate scores (distance from nearest bound, normalized by IQR)
        if iqr > 0:
            scores = np.zeros(len(outlier_values))
            for i, val in enumerate(outlier_values):
                if val < lower_bound:
                    scores[i] = (lower_bound - val) / iqr
                else:
                    scores[i] = (val - upper_bound) / iqr
        else:
            scores = np.zeros(len(outlier_values))

        result = OutlierAnalysis(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_scores=scores,
            threshold_used=threshold,
            method="iqr",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        result._total_count = len(data)  # type: ignore[attr-defined]

        return result

    def _detect_outliers_zscore(
        self,
        data: np.ndarray,
        threshold: float = 3.0,
    ) -> OutlierAnalysis:
        """Detect outliers using Z-score method.

        Outliers are values with |z-score| > threshold.

        Args:
            data: Array of values.
            threshold: Number of standard deviations (default 3.0).

        Returns:
            OutlierAnalysis with results.
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std < 1e-10:
            # No variation, no outliers
            return OutlierAnalysis(
                outlier_indices=[],
                outlier_values=np.array([]),
                outlier_scores=np.array([]),
                threshold_used=threshold,
                method="zscore",
            )

        z_scores = (data - mean) / std
        outlier_mask = np.abs(z_scores) > threshold
        outlier_indices = list(np.where(outlier_mask)[0])
        outlier_values = data[outlier_mask]
        outlier_scores = np.abs(z_scores[outlier_mask])

        result = OutlierAnalysis(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_scores=outlier_scores,
            threshold_used=threshold,
            method="zscore",
            lower_bound=float(mean - threshold * std),
            upper_bound=float(mean + threshold * std),
        )
        result._total_count = len(data)  # type: ignore[attr-defined]

        return result

    def _detect_outliers_modified_zscore(
        self,
        data: np.ndarray,
        threshold: float = 3.5,
    ) -> OutlierAnalysis:
        """Detect outliers using modified Z-score (MAD-based).

        Uses median and Median Absolute Deviation (MAD) for robustness.
        Modified Z-score = 0.6745 * (x - median) / MAD

        Args:
            data: Array of values.
            threshold: Modified Z-score threshold (default 3.5).

        Returns:
            OutlierAnalysis with results.
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad < 1e-10:
            # No variation, no outliers
            return OutlierAnalysis(
                outlier_indices=[],
                outlier_values=np.array([]),
                outlier_scores=np.array([]),
                threshold_used=threshold,
                method="modified_zscore",
            )

        # 0.6745 is the constant that makes MAD consistent with std for normal distribution
        modified_z = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z) > threshold
        outlier_indices = list(np.where(outlier_mask)[0])
        outlier_values = data[outlier_mask]
        outlier_scores = np.abs(modified_z[outlier_mask])

        result = OutlierAnalysis(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_scores=outlier_scores,
            threshold_used=threshold,
            method="modified_zscore",
        )
        result._total_count = len(data)  # type: ignore[attr-defined]

        return result

    def compare_outlier_methods(
        self,
        data: np.ndarray,
    ) -> dict[str, OutlierAnalysis]:
        """Compare all outlier detection methods on the same data.

        Args:
            data: Array of values.

        Returns:
            Dictionary mapping method names to OutlierAnalysis.
        """
        data = np.asarray(data, dtype=np.float64)

        return {
            "iqr": self.detect_outliers(data, method="iqr", threshold=1.5),
            "zscore": self.detect_outliers(data, method="zscore", threshold=3.0),
            "modified_zscore": self.detect_outliers(data, method="modified_zscore", threshold=3.5),
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return (
            f"PercentileAnalyzer("
            f"percentiles={self.percentiles}, "
            f"outlier_method={self.outlier_method!r})"
        )
