"""Drift detection framework for forecast model monitoring.

This module provides comprehensive drift detection capabilities for monitoring
forecast models in production. It detects both data drift (changes in input
feature distributions) and concept drift (changes in the relationship between
features and target).

Key Components:
- DriftDetector: Main class for drift detection
- DriftConfig: Configuration for drift detection methods
- DriftResult: Container for drift detection results
- DriftType: Enum for types of drift
- DriftAlert: Alert dataclass for drift events

Supported Methods:
- Statistical tests: KS test, Chi-square test, Population Stability Index (PSI)
- Sequential methods: CUSUM, ADWIN, Page-Hinkley
- Distribution comparison: Wasserstein distance, KL divergence

Example:
    ```python
    from src.evaluation.drift import DriftDetector, DriftConfig

    # Configure detector
    config = DriftConfig(
        reference_window_size=1000,
        test_window_size=100,
        significance_level=0.05,
    )

    # Create detector
    detector = DriftDetector(config)

    # Set reference distribution
    detector.fit(reference_data)

    # Detect drift in new data
    result = detector.detect(new_data)
    if result.drift_detected:
        print(f"Drift detected! Score: {result.drift_score:.4f}")
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected.

    Attributes:
        DATA_DRIFT: Changes in input feature distributions.
        CONCEPT_DRIFT: Changes in the target distribution or feature-target relationship.
        PREDICTION_DRIFT: Changes in model prediction distribution.
        PERFORMANCE_DRIFT: Degradation in model performance metrics.
    """

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class DriftMethod(Enum):
    """Drift detection methods.

    Attributes:
        KS_TEST: Kolmogorov-Smirnov two-sample test.
        CHI_SQUARE: Chi-square test for categorical distributions.
        PSI: Population Stability Index.
        WASSERSTEIN: Wasserstein (Earth Mover's) distance.
        KL_DIVERGENCE: Kullback-Leibler divergence.
        CUSUM: Cumulative Sum control chart.
        PAGE_HINKLEY: Page-Hinkley test for change detection.
    """

    KS_TEST = "ks_test"
    CHI_SQUARE = "chi_square"
    PSI = "psi"
    WASSERSTEIN = "wasserstein"
    KL_DIVERGENCE = "kl_divergence"
    CUSUM = "cusum"
    PAGE_HINKLEY = "page_hinkley"


@dataclass
class DriftConfig:
    """Configuration for drift detection.

    Attributes:
        reference_window_size: Size of reference window for comparison.
        test_window_size: Size of test window to evaluate.
        significance_level: Significance level for statistical tests (alpha).
        psi_threshold: PSI threshold for drift detection (default 0.1).
        wasserstein_threshold: Wasserstein distance threshold.
        cusum_threshold: CUSUM threshold for change detection.
        n_bins: Number of bins for histogram-based methods.
        min_samples: Minimum samples required for detection.

    Example:
        >>> config = DriftConfig(
        ...     reference_window_size=1000,
        ...     test_window_size=100,
        ...     significance_level=0.05
        ... )
    """

    reference_window_size: int = 1000
    test_window_size: int = 100
    significance_level: float = 0.05
    psi_threshold: float = 0.1
    wasserstein_threshold: float | None = None
    cusum_threshold: float = 5.0
    page_hinkley_threshold: float = 50.0
    page_hinkley_delta: float = 0.005
    n_bins: int = 10
    min_samples: int = 30
    epsilon: float = 1e-10

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.reference_window_size < 10:
            msg = f"reference_window_size must be >= 10, got {self.reference_window_size}"
            raise ValueError(msg)

        if self.test_window_size < 10:
            msg = f"test_window_size must be >= 10, got {self.test_window_size}"
            raise ValueError(msg)

        if not 0.0 < self.significance_level < 1.0:
            msg = f"significance_level must be between 0 and 1, got {self.significance_level}"
            raise ValueError(msg)

        if self.psi_threshold <= 0:
            msg = f"psi_threshold must be positive, got {self.psi_threshold}"
            raise ValueError(msg)

        if self.n_bins < 2:
            msg = f"n_bins must be >= 2, got {self.n_bins}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "reference_window_size": self.reference_window_size,
            "test_window_size": self.test_window_size,
            "significance_level": self.significance_level,
            "psi_threshold": self.psi_threshold,
            "wasserstein_threshold": self.wasserstein_threshold,
            "cusum_threshold": self.cusum_threshold,
            "page_hinkley_threshold": self.page_hinkley_threshold,
            "page_hinkley_delta": self.page_hinkley_delta,
            "n_bins": self.n_bins,
            "min_samples": self.min_samples,
        }


@dataclass
class DriftAlert:
    """Alert for detected drift.

    Attributes:
        drift_type: Type of drift detected.
        severity: Severity level ("low", "medium", "high", "critical").
        drift_score: Numeric score indicating drift magnitude.
        feature_name: Name of feature/variable with drift (if applicable).
        timestamp: When the drift was detected.
        details: Additional details about the drift.
        recommended_action: Suggested action to take.

    Example:
        >>> alert = DriftAlert(
        ...     drift_type=DriftType.DATA_DRIFT,
        ...     severity="high",
        ...     drift_score=0.25,
        ...     feature_name="temperature",
        ...     recommended_action="Retrain model with recent data"
        ... )
    """

    drift_type: DriftType
    severity: str
    drift_score: float
    feature_name: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""

    def __post_init__(self) -> None:
        """Validate alert."""
        valid_severities = {"low", "medium", "high", "critical"}
        if self.severity not in valid_severities:
            msg = f"severity must be one of {valid_severities}, got {self.severity}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "drift_type": self.drift_type.value,
            "severity": self.severity,
            "drift_score": self.drift_score,
            "feature_name": self.feature_name,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details.copy(),
            "recommended_action": self.recommended_action,
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return (
            f"DriftAlert({self.drift_type.value}, "
            f"severity={self.severity!r}, "
            f"score={self.drift_score:.4f})"
        )


@dataclass
class DriftResult:
    """Results from drift detection.

    Attributes:
        drift_detected: Whether drift was detected.
        drift_score: Numeric score indicating drift magnitude.
        p_value: P-value from statistical test (if applicable).
        method: Detection method used.
        drift_type: Type of drift detected.
        alerts: List of drift alerts if drift detected.
        feature_scores: Dict of feature names to their drift scores.
        reference_stats: Statistics from reference distribution.
        test_stats: Statistics from test distribution.
        metadata: Additional detection metadata.

    Example:
        >>> result = DriftResult(
        ...     drift_detected=True,
        ...     drift_score=0.15,
        ...     p_value=0.001,
        ...     method=DriftMethod.KS_TEST,
        ...     drift_type=DriftType.DATA_DRIFT
        ... )
    """

    drift_detected: bool
    drift_score: float
    p_value: float | None
    method: DriftMethod
    drift_type: DriftType
    alerts: list[DriftAlert] = field(default_factory=list)
    feature_scores: dict[str, float] = field(default_factory=dict)
    reference_stats: dict[str, float] = field(default_factory=dict)
    test_stats: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        """Get severity level based on drift score.

        Returns:
            Severity string ("low", "medium", "high", "critical").
        """
        if self.drift_score < 0.1:
            return "low"
        elif self.drift_score < 0.2:
            return "medium"
        elif self.drift_score < 0.3:
            return "high"
        else:
            return "critical"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "method": self.method.value,
            "drift_type": self.drift_type.value,
            "severity": self.severity,
            "alerts": [a.to_dict() for a in self.alerts],
            "feature_scores": self.feature_scores.copy(),
            "reference_stats": self.reference_stats.copy(),
            "test_stats": self.test_stats.copy(),
            "metadata": self.metadata.copy(),
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return (
            f"DriftResult("
            f"detected={self.drift_detected}, "
            f"score={self.drift_score:.4f}, "
            f"method={self.method.value})"
        )


class DriftDetector:
    """Drift detection framework for forecast model monitoring.

    Provides comprehensive drift detection capabilities including:
    - Statistical tests (KS, Chi-square)
    - Distribution metrics (PSI, Wasserstein, KL divergence)
    - Sequential change detection (CUSUM, Page-Hinkley)

    The detector compares a reference distribution against a test distribution
    to identify significant changes that may require model retraining.

    Example:
        >>> detector = DriftDetector()

        >>> # Set reference distribution
        >>> detector.fit(reference_data)

        >>> # Detect drift
        >>> result = detector.detect(test_data)
        >>> if result.drift_detected:
        ...     print(f"Drift detected! PSI={result.drift_score:.4f}")

        >>> # Multi-feature detection
        >>> results = detector.detect_multivariate(features_reference, features_test)
    """

    def __init__(self, config: DriftConfig | None = None) -> None:
        """Initialize detector.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or DriftConfig()
        self._reference: np.ndarray | None = None
        self._reference_stats: dict[str, float] = {}
        self._is_fitted = False

        # CUSUM state
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._cusum_mean: float = 0.0

        # Page-Hinkley state
        self._ph_sum: float = 0.0
        self._ph_min: float = float("inf")
        self._ph_count: int = 0
        self._ph_mean: float = 0.0

        logger.debug(
            "Initialized DriftDetector with config: %s",
            self.config.to_dict(),
        )

    def fit(self, reference_data: np.ndarray) -> "DriftDetector":
        """Fit detector on reference data.

        Computes reference statistics for comparison against test data.

        Args:
            reference_data: Reference data array.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If reference data is too small.
        """
        reference_data = np.asarray(reference_data, dtype=np.float64).flatten()
        reference_data = reference_data[np.isfinite(reference_data)]

        if len(reference_data) < self.config.min_samples:
            msg = f"Reference data needs at least {self.config.min_samples} samples, got {len(reference_data)}"
            raise ValueError(msg)

        self._reference = reference_data
        self._reference_stats = self._compute_stats(reference_data)

        # Initialize sequential methods
        self._cusum_mean = float(np.mean(reference_data))
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0

        self._ph_mean = float(np.mean(reference_data))
        self._ph_sum = 0.0
        self._ph_min = float("inf")
        self._ph_count = 0

        self._is_fitted = True

        logger.info(
            "Fitted DriftDetector on %d samples, mean=%.4f, std=%.4f",
            len(reference_data),
            self._reference_stats["mean"],
            self._reference_stats["std"],
        )

        return self

    def _compute_stats(self, data: np.ndarray) -> dict[str, float]:
        """Compute summary statistics for data.

        Args:
            data: Data array.

        Returns:
            Dictionary of statistics.
        """
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
            "median": float(np.median(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
            "n_samples": len(data),
        }

    def detect(
        self,
        test_data: np.ndarray,
        method: DriftMethod = DriftMethod.PSI,
        drift_type: DriftType = DriftType.DATA_DRIFT,
    ) -> DriftResult:
        """Detect drift in test data.

        Args:
            test_data: Test data to check for drift.
            method: Detection method to use.
            drift_type: Type of drift to report.

        Returns:
            DriftResult with detection results.

        Raises:
            ValueError: If detector not fitted or test data invalid.
        """
        if not self._is_fitted or self._reference is None:
            msg = "Detector not fitted. Call fit() first."
            raise ValueError(msg)

        test_data = np.asarray(test_data, dtype=np.float64).flatten()
        test_data = test_data[np.isfinite(test_data)]

        if len(test_data) < self.config.min_samples:
            logger.warning(
                "Test data has only %d samples, minimum is %d",
                len(test_data),
                self.config.min_samples,
            )
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                p_value=None,
                method=method,
                drift_type=drift_type,
                metadata={"error": "Insufficient test samples"},
            )

        # Compute test statistics
        test_stats = self._compute_stats(test_data)

        # Run detection based on method
        if method == DriftMethod.KS_TEST:
            result = self._detect_ks_test(test_data, drift_type)
        elif method == DriftMethod.PSI:
            result = self._detect_psi(test_data, drift_type)
        elif method == DriftMethod.WASSERSTEIN:
            result = self._detect_wasserstein(test_data, drift_type)
        elif method == DriftMethod.KL_DIVERGENCE:
            result = self._detect_kl_divergence(test_data, drift_type)
        elif method == DriftMethod.CUSUM:
            result = self._detect_cusum(test_data, drift_type)
        elif method == DriftMethod.PAGE_HINKLEY:
            result = self._detect_page_hinkley(test_data, drift_type)
        elif method == DriftMethod.CHI_SQUARE:
            result = self._detect_chi_square(test_data, drift_type)
        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        # Add statistics
        result.reference_stats = self._reference_stats.copy()
        result.test_stats = test_stats
        result.metadata["timestamp"] = datetime.now().isoformat()

        # Generate alerts if drift detected
        if result.drift_detected:
            alert = DriftAlert(
                drift_type=drift_type,
                severity=result.severity,
                drift_score=result.drift_score,
                details={"method": method.value, "p_value": result.p_value},
                recommended_action=self._get_recommended_action(result),
            )
            result.alerts.append(alert)

        return result

    def _detect_ks_test(
        self,
        test_data: np.ndarray,
        drift_type: DriftType,
    ) -> DriftResult:
        """Detect drift using Kolmogorov-Smirnov test.

        Args:
            test_data: Test data.
            drift_type: Type of drift.

        Returns:
            DriftResult.
        """
        assert self._reference is not None

        # Two-sample KS test
        statistic, p_value = stats.ks_2samp(self._reference, test_data)

        drift_detected = p_value < self.config.significance_level

        return DriftResult(
            drift_detected=drift_detected,
            drift_score=float(statistic),
            p_value=float(p_value),
            method=DriftMethod.KS_TEST,
            drift_type=drift_type,
        )

    def _detect_psi(
        self,
        test_data: np.ndarray,
        drift_type: DriftType,
    ) -> DriftResult:
        """Detect drift using Population Stability Index.

        PSI formula: sum((test% - ref%) * ln(test% / ref%))
        Interpretation:
        - PSI < 0.1: No significant drift
        - 0.1 <= PSI < 0.2: Moderate drift
        - PSI >= 0.2: Significant drift

        Args:
            test_data: Test data.
            drift_type: Type of drift.

        Returns:
            DriftResult.
        """
        assert self._reference is not None

        # Create bins from reference data
        min_val = min(np.min(self._reference), np.min(test_data))
        max_val = max(np.max(self._reference), np.max(test_data))

        # Add small buffer to include edge values
        buffer = (max_val - min_val) * 0.01 + self.config.epsilon
        bins = np.linspace(min_val - buffer, max_val + buffer, self.config.n_bins + 1)

        # Calculate histograms
        ref_hist, _ = np.histogram(self._reference, bins=bins)
        test_hist, _ = np.histogram(test_data, bins=bins)

        # Convert to proportions with epsilon for stability
        ref_pct = (ref_hist + self.config.epsilon) / (
            len(self._reference) + self.config.epsilon * self.config.n_bins
        )
        test_pct = (test_hist + self.config.epsilon) / (
            len(test_data) + self.config.epsilon * self.config.n_bins
        )

        # Calculate PSI
        psi = float(np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct)))

        drift_detected = psi >= self.config.psi_threshold

        return DriftResult(
            drift_detected=drift_detected,
            drift_score=psi,
            p_value=None,  # PSI doesn't have a p-value
            method=DriftMethod.PSI,
            drift_type=drift_type,
            metadata={"threshold": self.config.psi_threshold},
        )

    def _detect_wasserstein(
        self,
        test_data: np.ndarray,
        drift_type: DriftType,
    ) -> DriftResult:
        """Detect drift using Wasserstein distance.

        Also known as Earth Mover's Distance.

        Args:
            test_data: Test data.
            drift_type: Type of drift.

        Returns:
            DriftResult.
        """
        assert self._reference is not None

        # Calculate Wasserstein distance
        distance = stats.wasserstein_distance(self._reference, test_data)

        # Normalize by reference standard deviation for interpretability
        ref_std = self._reference_stats.get("std", 1.0)
        if ref_std > self.config.epsilon:
            normalized_distance = distance / ref_std
        else:
            normalized_distance = distance

        # Determine threshold
        threshold = self.config.wasserstein_threshold
        if threshold is None:
            # Default: drift if distance > 0.5 std
            threshold = 0.5

        drift_detected = normalized_distance >= threshold

        return DriftResult(
            drift_detected=drift_detected,
            drift_score=float(normalized_distance),
            p_value=None,
            method=DriftMethod.WASSERSTEIN,
            drift_type=drift_type,
            metadata={"raw_distance": float(distance), "threshold": threshold},
        )

    def _detect_kl_divergence(
        self,
        test_data: np.ndarray,
        drift_type: DriftType,
    ) -> DriftResult:
        """Detect drift using Kullback-Leibler divergence.

        Note: KL divergence is asymmetric, so we use the symmetric version:
        KL_sym = (KL(P||Q) + KL(Q||P)) / 2

        Args:
            test_data: Test data.
            drift_type: Type of drift.

        Returns:
            DriftResult.
        """
        assert self._reference is not None

        # Create bins
        min_val = min(np.min(self._reference), np.min(test_data))
        max_val = max(np.max(self._reference), np.max(test_data))
        buffer = (max_val - min_val) * 0.01 + self.config.epsilon
        bins = np.linspace(min_val - buffer, max_val + buffer, self.config.n_bins + 1)

        # Calculate histograms
        ref_hist, _ = np.histogram(self._reference, bins=bins)
        test_hist, _ = np.histogram(test_data, bins=bins)

        # Convert to probability distributions with epsilon
        ref_prob = (ref_hist + self.config.epsilon) / (
            np.sum(ref_hist) + self.config.epsilon * self.config.n_bins
        )
        test_prob = (test_hist + self.config.epsilon) / (
            np.sum(test_hist) + self.config.epsilon * self.config.n_bins
        )

        # Calculate symmetric KL divergence
        kl_pq = float(np.sum(ref_prob * np.log(ref_prob / test_prob)))
        kl_qp = float(np.sum(test_prob * np.log(test_prob / ref_prob)))
        kl_symmetric = (kl_pq + kl_qp) / 2

        # Threshold based on typical values
        threshold = 0.1
        drift_detected = kl_symmetric >= threshold

        return DriftResult(
            drift_detected=drift_detected,
            drift_score=kl_symmetric,
            p_value=None,
            method=DriftMethod.KL_DIVERGENCE,
            drift_type=drift_type,
            metadata={"kl_pq": kl_pq, "kl_qp": kl_qp, "threshold": threshold},
        )

    def _detect_chi_square(
        self,
        test_data: np.ndarray,
        drift_type: DriftType,
    ) -> DriftResult:
        """Detect drift using Chi-square test.

        Args:
            test_data: Test data.
            drift_type: Type of drift.

        Returns:
            DriftResult.
        """
        assert self._reference is not None

        # Create bins
        min_val = min(np.min(self._reference), np.min(test_data))
        max_val = max(np.max(self._reference), np.max(test_data))
        buffer = (max_val - min_val) * 0.01 + self.config.epsilon
        bins = np.linspace(min_val - buffer, max_val + buffer, self.config.n_bins + 1)

        # Calculate histograms
        ref_hist, _ = np.histogram(self._reference, bins=bins)
        test_hist, _ = np.histogram(test_data, bins=bins)

        # Scale test histogram to reference size for comparison
        scale_factor = len(self._reference) / len(test_data)
        test_hist_scaled = test_hist * scale_factor

        # Add epsilon to avoid division by zero
        ref_hist = ref_hist + self.config.epsilon
        test_hist_scaled = test_hist_scaled + self.config.epsilon

        # Chi-square statistic
        chi2_stat = float(np.sum((test_hist_scaled - ref_hist) ** 2 / ref_hist))
        df = self.config.n_bins - 1
        p_value = float(1 - stats.chi2.cdf(chi2_stat, df))

        drift_detected = p_value < self.config.significance_level

        return DriftResult(
            drift_detected=drift_detected,
            drift_score=chi2_stat,
            p_value=p_value,
            method=DriftMethod.CHI_SQUARE,
            drift_type=drift_type,
            metadata={"degrees_of_freedom": df},
        )

    def _detect_cusum(
        self,
        test_data: np.ndarray,
        drift_type: DriftType,
    ) -> DriftResult:
        """Detect drift using CUSUM (Cumulative Sum) control chart.

        CUSUM detects small persistent shifts in the mean.

        Args:
            test_data: Test data.
            drift_type: Type of drift.

        Returns:
            DriftResult.
        """
        assert self._reference is not None

        # Reset CUSUM state for new detection
        cusum_pos = 0.0
        cusum_neg = 0.0
        max_cusum = 0.0

        target = self._cusum_mean
        std = self._reference_stats.get("std", 1.0)

        # Allowable slack (typically 0.5 * shift to detect)
        k = 0.5 * std

        for x in test_data:
            diff = x - target
            cusum_pos = max(0.0, cusum_pos + diff - k)
            cusum_neg = max(0.0, cusum_neg - diff - k)
            max_cusum = max(max_cusum, cusum_pos, cusum_neg)

        # Normalize by std for interpretability
        if std > self.config.epsilon:
            normalized_cusum = max_cusum / std
        else:
            normalized_cusum = max_cusum

        drift_detected = normalized_cusum >= self.config.cusum_threshold

        return DriftResult(
            drift_detected=drift_detected,
            drift_score=float(normalized_cusum),
            p_value=None,
            method=DriftMethod.CUSUM,
            drift_type=drift_type,
            metadata={
                "max_cusum_raw": float(max_cusum),
                "threshold": self.config.cusum_threshold,
            },
        )

    def _detect_page_hinkley(
        self,
        test_data: np.ndarray,
        drift_type: DriftType,
    ) -> DriftResult:
        """Detect drift using Page-Hinkley test.

        Page-Hinkley is a sequential change detection algorithm.

        Args:
            test_data: Test data.
            drift_type: Type of drift.

        Returns:
            DriftResult.
        """
        assert self._reference is not None

        # Reset state for new detection
        ph_sum = 0.0
        ph_min = float("inf")
        count = 0
        running_mean = self._ph_mean
        delta = self.config.page_hinkley_delta

        max_ph = 0.0

        for x in test_data:
            count += 1
            running_mean = running_mean + (x - running_mean) / count
            ph_sum = ph_sum + (x - running_mean - delta)
            ph_min = min(ph_min, ph_sum)

            ph_test = ph_sum - ph_min
            max_ph = max(max_ph, ph_test)

        drift_detected = max_ph >= self.config.page_hinkley_threshold

        return DriftResult(
            drift_detected=drift_detected,
            drift_score=float(max_ph),
            p_value=None,
            method=DriftMethod.PAGE_HINKLEY,
            drift_type=drift_type,
            metadata={"threshold": self.config.page_hinkley_threshold},
        )

    def detect_multivariate(
        self,
        reference_features: dict[str, np.ndarray],
        test_features: dict[str, np.ndarray],
        method: DriftMethod = DriftMethod.PSI,
    ) -> DriftResult:
        """Detect drift across multiple features.

        Args:
            reference_features: Dict of feature names to reference data.
            test_features: Dict of feature names to test data.
            method: Detection method to use.

        Returns:
            Combined DriftResult with per-feature scores.

        Raises:
            ValueError: If feature sets don't match.
        """
        if set(reference_features.keys()) != set(test_features.keys()):
            msg = "Reference and test features must have same keys"
            raise ValueError(msg)

        feature_scores: dict[str, float] = {}
        feature_results: list[DriftResult] = []
        alerts: list[DriftAlert] = []

        for feature_name, ref_data in reference_features.items():
            test_data = test_features[feature_name]

            # Fit on reference
            self.fit(ref_data)

            # Detect drift
            result = self.detect(test_data, method=method)

            feature_scores[feature_name] = result.drift_score
            feature_results.append(result)

            if result.drift_detected:
                alert = DriftAlert(
                    drift_type=DriftType.DATA_DRIFT,
                    severity=result.severity,
                    drift_score=result.drift_score,
                    feature_name=feature_name,
                    details={"p_value": result.p_value},
                    recommended_action=f"Review feature '{feature_name}' distribution",
                )
                alerts.append(alert)

        # Aggregate results
        max_score = max(feature_scores.values()) if feature_scores else 0.0
        any_drift = any(r.drift_detected for r in feature_results)

        return DriftResult(
            drift_detected=any_drift,
            drift_score=max_score,
            p_value=None,
            method=method,
            drift_type=DriftType.DATA_DRIFT,
            alerts=alerts,
            feature_scores=feature_scores,
            metadata={
                "n_features": len(feature_scores),
                "n_drifted": sum(1 for r in feature_results if r.drift_detected),
            },
        )

    def detect_performance_drift(
        self,
        reference_errors: np.ndarray,
        test_errors: np.ndarray,
    ) -> DriftResult:
        """Detect drift in model performance metrics.

        Args:
            reference_errors: Error values from reference period.
            test_errors: Error values from test period.

        Returns:
            DriftResult for performance drift.
        """
        # Fit on reference errors
        self.fit(reference_errors)

        # Detect using KS test (good for error distributions)
        result = self.detect(
            test_errors,
            method=DriftMethod.KS_TEST,
            drift_type=DriftType.PERFORMANCE_DRIFT,
        )

        # Additional check: mean error increase
        ref_mean = float(np.mean(reference_errors))
        test_mean = float(np.mean(test_errors))
        mean_increase = (test_mean - ref_mean) / (ref_mean + self.config.epsilon)

        result.metadata["reference_mean_error"] = ref_mean
        result.metadata["test_mean_error"] = test_mean
        result.metadata["mean_increase_pct"] = mean_increase * 100

        # Significant performance drift if mean error increased > 10%
        if mean_increase > 0.1:
            result.drift_detected = True
            alert = DriftAlert(
                drift_type=DriftType.PERFORMANCE_DRIFT,
                severity="high" if mean_increase > 0.2 else "medium",
                drift_score=mean_increase,
                details={"mean_increase_pct": mean_increase * 100},
                recommended_action="Model performance degraded. Consider retraining.",
            )
            result.alerts.append(alert)

        return result

    def _get_recommended_action(self, result: DriftResult) -> str:
        """Get recommended action based on drift result.

        Args:
            result: Drift detection result.

        Returns:
            Recommended action string.
        """
        if result.severity == "critical":
            return "Immediate model retraining recommended. Significant drift detected."
        elif result.severity == "high":
            return "Schedule model retraining. Consider data quality investigation."
        elif result.severity == "medium":
            return "Monitor closely. Evaluate if retraining is needed."
        else:
            return "Continue monitoring. No immediate action required."

    def get_drift_summary(
        self,
        test_data: np.ndarray,
    ) -> dict[str, DriftResult]:
        """Run all drift detection methods and return summary.

        Args:
            test_data: Test data to analyze.

        Returns:
            Dictionary mapping method names to their results.
        """
        if not self._is_fitted:
            msg = "Detector not fitted. Call fit() first."
            raise ValueError(msg)

        methods = [
            DriftMethod.KS_TEST,
            DriftMethod.PSI,
            DriftMethod.WASSERSTEIN,
            DriftMethod.KL_DIVERGENCE,
            DriftMethod.CHI_SQUARE,
            DriftMethod.CUSUM,
            DriftMethod.PAGE_HINKLEY,
        ]

        results = {}
        for method in methods:
            try:
                results[method.value] = self.detect(test_data, method=method)
            except Exception as e:
                logger.warning("Failed to run %s: %s", method.value, str(e))

        return results

    def reset(self) -> None:
        """Reset detector state.

        Clears reference data and all internal state.
        """
        self._reference = None
        self._reference_stats = {}
        self._is_fitted = False

        # Reset sequential states
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._cusum_mean = 0.0
        self._ph_sum = 0.0
        self._ph_min = float("inf")
        self._ph_count = 0
        self._ph_mean = 0.0

        logger.debug("Reset DriftDetector state")

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        fitted_status = "fitted" if self._is_fitted else "not fitted"
        n_samples = len(self._reference) if self._reference is not None else 0
        return f"DriftDetector({fitted_status}, n_ref={n_samples})"
