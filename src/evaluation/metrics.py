"""Core metrics calculation engine for forecast evaluation.

This module provides the MetricsCalculator class that computes standard
forecasting metrics (MAPE, MAE, RMSE, R²) with robust handling of edge
cases including zeros, near-zero values, and missing data.

Key Features:
- Standard and symmetric MAPE calculation
- Weighted metrics support
- Bootstrap-based confidence intervals
- Statistical significance testing (paired t-test, Wilcoxon)
- Robust NaN/infinity handling

Example:
    ```python
    from src.evaluation.metrics import MetricsCalculator, MetricsConfig
    import numpy as np

    # Create calculator
    config = MetricsConfig(use_symmetric_mape=True)
    calculator = MetricsCalculator(config)

    # Prepare data
    actual = np.array([100, 200, 300, 400, 500])
    forecast = np.array([95, 210, 290, 410, 480])

    # Calculate all metrics
    result = calculator.calculate_single(actual, forecast)
    print(f"MAPE: {result.mape:.2f}%")
    print(f"RMSE: {result.rmse:.2f}")
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation.

    Attributes:
        epsilon: Small value to prevent division by zero.
        use_symmetric_mape: Use symmetric MAPE for better zero handling.
        confidence_level: Confidence level for intervals (0.0-1.0).
        bootstrap_iterations: Number of bootstrap iterations for CI.
        weighted_metrics: Enable weighted metrics calculation.
        nan_handling: How to handle NaN values ("exclude", "error", "zero").
        random_seed: Random seed for reproducibility of bootstrap.

    Example:
        >>> config = MetricsConfig(
        ...     use_symmetric_mape=True,
        ...     confidence_level=0.95,
        ...     bootstrap_iterations=1000
        ... )
    """

    epsilon: float = 1e-8
    use_symmetric_mape: bool = True
    confidence_level: float = 0.95
    bootstrap_iterations: int = 1000
    weighted_metrics: bool = False
    nan_handling: str = "exclude"
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.epsilon <= 0:
            msg = f"epsilon must be positive, got {self.epsilon}"
            raise ValueError(msg)

        if not 0.0 < self.confidence_level < 1.0:
            msg = f"confidence_level must be between 0 and 1, got {self.confidence_level}"
            raise ValueError(msg)

        if self.bootstrap_iterations < 0:
            msg = f"bootstrap_iterations must be non-negative, got {self.bootstrap_iterations}"
            raise ValueError(msg)

        valid_nan_handling = {"exclude", "error", "zero"}
        if self.nan_handling not in valid_nan_handling:
            msg = f"nan_handling must be one of {valid_nan_handling}, got {self.nan_handling}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "epsilon": self.epsilon,
            "use_symmetric_mape": self.use_symmetric_mape,
            "confidence_level": self.confidence_level,
            "bootstrap_iterations": self.bootstrap_iterations,
            "weighted_metrics": self.weighted_metrics,
            "nan_handling": self.nan_handling,
            "random_seed": self.random_seed,
        }


@dataclass
class MetricsResult:
    """Container for calculated metrics.

    Attributes:
        mape: Mean Absolute Percentage Error (%).
        mae: Mean Absolute Error.
        rmse: Root Mean Square Error.
        mse: Mean Square Error.
        r2: R-squared (coefficient of determination).
        confidence_intervals: Dict of metric name to (lower, upper) CI bounds.
        sample_size: Number of valid samples used.
        n_excluded: Number of samples excluded (NaN/Inf).
        metadata: Additional metadata about the calculation.

    Example:
        >>> result = MetricsResult(
        ...     mape=5.2,
        ...     mae=10.5,
        ...     rmse=15.3,
        ...     mse=234.09,
        ...     r2=0.95,
        ...     confidence_intervals={"mape": (4.8, 5.6)},
        ...     sample_size=100
        ... )
    """

    mape: float
    mae: float
    rmse: float
    mse: float
    r2: float
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    sample_size: int = 0
    n_excluded: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result values."""
        # Check for invalid values
        for name, value in [("mape", self.mape), ("mae", self.mae), ("rmse", self.rmse)]:
            if np.isnan(value) or np.isinf(value):
                logger.warning("%s has invalid value: %s", name, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "mape": self.mape,
            "mae": self.mae,
            "rmse": self.rmse,
            "mse": self.mse,
            "r2": self.r2,
            "confidence_intervals": self.confidence_intervals.copy(),
            "sample_size": self.sample_size,
            "n_excluded": self.n_excluded,
            "metadata": self.metadata.copy(),
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return (
            f"MetricsResult("
            f"MAPE={self.mape:.2f}%, "
            f"MAE={self.mae:.2f}, "
            f"RMSE={self.rmse:.2f}, "
            f"R²={self.r2:.3f}, "
            f"n={self.sample_size})"
        )


class MetricsCalculator:
    """Core metrics calculation engine.

    Provides accurate and efficient computation of standard forecasting
    metrics with robust handling of edge cases.

    Supported Metrics:
        - MAPE (Mean Absolute Percentage Error)
        - sMAPE (Symmetric MAPE)
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Square Error)
        - MSE (Mean Square Error)
        - R² (Coefficient of Determination)

    Features:
        - Weighted metrics support
        - Bootstrap confidence intervals
        - NaN/Inf handling
        - Zero/near-zero protection

    Example:
        >>> calculator = MetricsCalculator()

        >>> # Single prediction
        >>> result = calculator.calculate_single(actual, forecast)
        >>> print(f"MAPE: {result.mape:.2f}%")

        >>> # Multiple models
        >>> results = calculator.calculate_metrics(
        ...     predictions={"model_a": pred_a, "model_b": pred_b},
        ...     actuals={"model_a": act, "model_b": act}
        ... )
    """

    def __init__(self, config: MetricsConfig | None = None) -> None:
        """Initialize calculator.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or MetricsConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        logger.debug(
            "Initialized MetricsCalculator with config: %s",
            self.config.to_dict(),
        )

    def calculate_metrics(
        self,
        predictions: dict[str, np.ndarray],
        actuals: dict[str, np.ndarray],
        weights: dict[str, np.ndarray] | None = None,
    ) -> dict[str, MetricsResult]:
        """Calculate metrics for multiple models/series.

        Args:
            predictions: Dict mapping names to prediction arrays.
            actuals: Dict mapping names to actual value arrays.
            weights: Optional dict mapping names to weight arrays.

        Returns:
            Dict mapping names to MetricsResult instances.

        Raises:
            ValueError: If predictions and actuals keys don't match.

        Example:
            >>> results = calculator.calculate_metrics(
            ...     predictions={"h0": pred_h0, "h1": pred_h1},
            ...     actuals={"h0": act_h0, "h1": act_h1}
            ... )
            >>> print(results["h0"].mape)
        """
        if set(predictions.keys()) != set(actuals.keys()):
            msg = "predictions and actuals must have same keys"
            raise ValueError(msg)

        results = {}
        for name in predictions:
            pred = predictions[name]
            actual = actuals[name]
            weight = weights.get(name) if weights else None

            results[name] = self.calculate_single(actual, pred, weight)

        return results

    def calculate_single(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> MetricsResult:
        """Calculate all metrics for a single prediction.

        Args:
            actual: Array of actual values.
            forecast: Array of predicted values.
            weights: Optional array of weights.

        Returns:
            MetricsResult with all calculated metrics.

        Raises:
            ValueError: If arrays have different lengths or are empty.
        """
        actual = np.asarray(actual, dtype=np.float64)
        forecast = np.asarray(forecast, dtype=np.float64)

        # Validate inputs
        if actual.shape != forecast.shape:
            msg = f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}"
            raise ValueError(msg)

        if len(actual) == 0:
            msg = "Input arrays cannot be empty"
            raise ValueError(msg)

        # Handle weights
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != actual.shape:
                msg = f"Weights shape {weights.shape} must match actual shape {actual.shape}"
                raise ValueError(msg)

        # Handle NaN values
        actual_clean, forecast_clean, weights_clean, n_excluded = self._handle_nan(
            actual, forecast, weights
        )

        if len(actual_clean) == 0:
            # All values excluded
            return MetricsResult(
                mape=np.nan,
                mae=np.nan,
                rmse=np.nan,
                mse=np.nan,
                r2=np.nan,
                sample_size=0,
                n_excluded=n_excluded,
                metadata={"error": "All values excluded"},
            )

        # Calculate metrics
        mape = self.calculate_mape(actual_clean, forecast_clean)
        mae = self.calculate_mae(actual_clean, forecast_clean, weights_clean)
        mse = self.calculate_mse(actual_clean, forecast_clean, weights_clean)
        rmse = np.sqrt(mse)
        r2 = self.calculate_r2(actual_clean, forecast_clean)

        # Calculate confidence intervals if enabled
        ci = {}
        if self.config.bootstrap_iterations > 0:
            ci = self.calculate_confidence_intervals(actual_clean, forecast_clean)

        return MetricsResult(
            mape=mape,
            mae=mae,
            rmse=rmse,
            mse=mse,
            r2=r2,
            confidence_intervals=ci,
            sample_size=len(actual_clean),
            n_excluded=n_excluded,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "config": self.config.to_dict(),
            },
        )

    def _handle_nan(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
        weights: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
        """Handle NaN and Inf values in inputs.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            weights: Optional weights.

        Returns:
            Tuple of (clean_actual, clean_forecast, clean_weights, n_excluded).

        Raises:
            ValueError: If nan_handling is "error" and NaN values present.
        """
        # Find valid (non-NaN, non-Inf) indices
        valid_mask = np.isfinite(actual) & np.isfinite(forecast)
        if weights is not None:
            valid_mask &= np.isfinite(weights)

        n_excluded = int(np.sum(~valid_mask))

        if n_excluded > 0:
            if self.config.nan_handling == "error":
                msg = f"Found {n_excluded} NaN/Inf values with nan_handling='error'"
                raise ValueError(msg)
            elif self.config.nan_handling == "zero":
                # Replace with zeros
                actual = np.where(valid_mask, actual, 0.0)
                forecast = np.where(valid_mask, forecast, 0.0)
                if weights is not None:
                    weights = np.where(valid_mask, weights, 0.0)
                return actual, forecast, weights, n_excluded
            # else: exclude

        if n_excluded > 0 and self.config.nan_handling == "exclude":
            actual = actual[valid_mask]
            forecast = forecast[valid_mask]
            if weights is not None:
                weights = weights[valid_mask]

        return actual, forecast, weights, n_excluded

    def calculate_mape(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
    ) -> float:
        """Calculate MAPE (Mean Absolute Percentage Error).

        Uses symmetric MAPE if configured, otherwise standard MAPE
        with epsilon protection for zero denominators.

        Args:
            actual: Array of actual values.
            forecast: Array of forecast values.

        Returns:
            MAPE as percentage (e.g., 5.0 for 5%).
        """
        if self.config.use_symmetric_mape:
            return self._calculate_smape(actual, forecast)
        return self._calculate_standard_mape(actual, forecast)

    def _calculate_standard_mape(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
    ) -> float:
        """Calculate standard MAPE.

        Formula: MAPE = 100 * mean(|actual - forecast| / max(|actual|, ε))

        Args:
            actual: Actual values.
            forecast: Forecast values.

        Returns:
            MAPE percentage.
        """
        error = np.abs(actual - forecast)
        denominator = np.maximum(np.abs(actual), self.config.epsilon)
        mape = 100.0 * np.mean(error / denominator)
        return float(mape)

    def _calculate_smape(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
    ) -> float:
        """Calculate symmetric MAPE.

        Formula: sMAPE = 100 * mean(|actual - forecast| / ((|actual| + |forecast|) / 2))

        This version handles zeros more robustly than standard MAPE.

        Args:
            actual: Actual values.
            forecast: Forecast values.

        Returns:
            sMAPE percentage.
        """
        error = np.abs(actual - forecast)
        denominator = (np.abs(actual) + np.abs(forecast)) / 2.0
        denominator = np.maximum(denominator, self.config.epsilon)
        smape = 100.0 * np.mean(error / denominator)
        return float(smape)

    def calculate_mae(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """Calculate MAE (Mean Absolute Error).

        Formula: MAE = sum(weights * |actual - forecast|) / sum(weights)
        If no weights, uses simple mean.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            weights: Optional weights.

        Returns:
            MAE value.
        """
        error = np.abs(actual - forecast)

        if weights is not None and self.config.weighted_metrics:
            mae = np.sum(weights * error) / np.sum(weights)
        else:
            mae = np.mean(error)

        return float(mae)

    def calculate_mse(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """Calculate MSE (Mean Square Error).

        Formula: MSE = sum(weights * (actual - forecast)²) / sum(weights)
        If no weights, uses simple mean.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            weights: Optional weights.

        Returns:
            MSE value.
        """
        error_sq = (actual - forecast) ** 2

        if weights is not None and self.config.weighted_metrics:
            mse = np.sum(weights * error_sq) / np.sum(weights)
        else:
            mse = np.mean(error_sq)

        return float(mse)

    def calculate_rmse(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """Calculate RMSE (Root Mean Square Error).

        Formula: RMSE = sqrt(MSE)

        Args:
            actual: Actual values.
            forecast: Forecast values.
            weights: Optional weights.

        Returns:
            RMSE value.
        """
        mse = self.calculate_mse(actual, forecast, weights)
        return float(np.sqrt(mse))

    def calculate_r2(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
    ) -> float:
        """Calculate R² (coefficient of determination).

        Formula: R² = 1 - SS_res / SS_tot
        where SS_res = sum((actual - forecast)²)
              SS_tot = sum((actual - mean(actual))²)

        Args:
            actual: Actual values.
            forecast: Forecast values.

        Returns:
            R² value (can be negative for very bad fits).
        """
        ss_res = np.sum((actual - forecast) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        if ss_tot < self.config.epsilon:
            # Constant actual values
            if ss_res < self.config.epsilon:
                return 1.0  # Perfect prediction of constant
            return 0.0  # Can't define R² for constant

        r2 = 1.0 - (ss_res / ss_tot)
        return float(r2)

    def calculate_confidence_intervals(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
    ) -> dict[str, tuple[float, float]]:
        """Calculate bootstrap confidence intervals for metrics.

        Uses percentile method with the configured number of iterations.

        Args:
            actual: Actual values.
            forecast: Forecast values.

        Returns:
            Dict mapping metric names to (lower, upper) bounds.
        """
        n = len(actual)
        alpha = 1.0 - self.config.confidence_level
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100

        # Store bootstrap samples
        bootstrap_mape = []
        bootstrap_mae = []
        bootstrap_rmse = []

        for _ in range(self.config.bootstrap_iterations):
            # Resample with replacement
            indices = self._rng.integers(0, n, size=n)
            actual_boot = actual[indices]
            forecast_boot = forecast[indices]

            # Calculate metrics for bootstrap sample
            bootstrap_mape.append(self.calculate_mape(actual_boot, forecast_boot))
            bootstrap_mae.append(self.calculate_mae(actual_boot, forecast_boot))
            bootstrap_rmse.append(self.calculate_rmse(actual_boot, forecast_boot))

        # Calculate percentile CIs
        ci = {
            "mape": (
                float(np.percentile(bootstrap_mape, lower_pct)),
                float(np.percentile(bootstrap_mape, upper_pct)),
            ),
            "mae": (
                float(np.percentile(bootstrap_mae, lower_pct)),
                float(np.percentile(bootstrap_mae, upper_pct)),
            ),
            "rmse": (
                float(np.percentile(bootstrap_rmse, lower_pct)),
                float(np.percentile(bootstrap_rmse, upper_pct)),
            ),
        }

        return ci


class StatisticalTests:
    """Statistical tests for model comparison.

    Provides paired t-test and Wilcoxon signed-rank test for
    comparing forecast errors between two models.

    Example:
        >>> tests = StatisticalTests()
        >>> result = tests.paired_t_test(errors_a, errors_b)
        >>> if result["significant"]:
        ...     print(f"Significant difference, p={result['p_value']:.4f}")
    """

    def paired_t_test(
        self,
        errors_a: np.ndarray,
        errors_b: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Perform paired t-test on forecast errors.

        Tests if there is a significant difference between the mean
        errors of two models.

        Args:
            errors_a: Absolute errors from model A.
            errors_b: Absolute errors from model B.
            alpha: Significance level.

        Returns:
            Dict with t_statistic, p_value, significant, effect_size.

        Raises:
            ValueError: If arrays have different lengths or insufficient samples.
        """
        errors_a = np.asarray(errors_a, dtype=np.float64)
        errors_b = np.asarray(errors_b, dtype=np.float64)

        if len(errors_a) != len(errors_b):
            msg = f"Arrays must have same length: {len(errors_a)} vs {len(errors_b)}"
            raise ValueError(msg)

        if len(errors_a) < 2:
            msg = "Need at least 2 samples for t-test"
            raise ValueError(msg)

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(errors_a, errors_b)

        # Calculate effect size (Cohen's d)
        diff = errors_a - errors_b
        effect_size = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "effect_size": effect_size,
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "n_samples": len(errors_a),
            "alpha": alpha,
        }

    def wilcoxon_test(
        self,
        errors_a: np.ndarray,
        errors_b: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Perform Wilcoxon signed-rank test on forecast errors.

        Non-parametric alternative to paired t-test. Useful when
        error distributions are not normal.

        Args:
            errors_a: Absolute errors from model A.
            errors_b: Absolute errors from model B.
            alpha: Significance level.

        Returns:
            Dict with statistic, p_value, significant.

        Raises:
            ValueError: If arrays have different lengths or insufficient samples.
        """
        errors_a = np.asarray(errors_a, dtype=np.float64)
        errors_b = np.asarray(errors_b, dtype=np.float64)

        if len(errors_a) != len(errors_b):
            msg = f"Arrays must have same length: {len(errors_a)} vs {len(errors_b)}"
            raise ValueError(msg)

        if len(errors_a) < 10:
            msg = "Wilcoxon test needs at least 10 samples for reliable results"
            raise ValueError(msg)

        # Check for zero differences (Wilcoxon can't handle all zeros)
        diff = errors_a - errors_b
        if np.allclose(diff, 0):
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "n_samples": len(errors_a),
                "alpha": alpha,
                "note": "All differences are zero",
            }

        # Perform Wilcoxon signed-rank test
        try:
            statistic, p_value = stats.wilcoxon(errors_a, errors_b, alternative="two-sided")
        except ValueError as e:
            # Handle edge cases (e.g., all zeros after removing ties)
            return {
                "statistic": np.nan,
                "p_value": 1.0,
                "significant": False,
                "n_samples": len(errors_a),
                "alpha": alpha,
                "error": str(e),
            }

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "median_diff": float(np.median(diff)),
            "n_samples": len(errors_a),
            "alpha": alpha,
        }

    def compare_models(
        self,
        predictions_a: np.ndarray,
        predictions_b: np.ndarray,
        actuals: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Compare two models using multiple statistical tests.

        Args:
            predictions_a: Predictions from model A.
            predictions_b: Predictions from model B.
            actuals: Actual values.
            alpha: Significance level.

        Returns:
            Dict with results from both tests and summary.
        """
        predictions_a = np.asarray(predictions_a, dtype=np.float64)
        predictions_b = np.asarray(predictions_b, dtype=np.float64)
        actuals = np.asarray(actuals, dtype=np.float64)

        # Calculate absolute errors
        errors_a = np.abs(actuals - predictions_a)
        errors_b = np.abs(actuals - predictions_b)

        # Run tests
        t_test_result = self.paired_t_test(errors_a, errors_b, alpha)

        wilcoxon_result = None
        if len(errors_a) >= 10:
            wilcoxon_result = self.wilcoxon_test(errors_a, errors_b, alpha)

        # Summary
        mean_error_a = float(np.mean(errors_a))
        mean_error_b = float(np.mean(errors_b))
        better_model = "A" if mean_error_a < mean_error_b else "B"

        return {
            "t_test": t_test_result,
            "wilcoxon": wilcoxon_result,
            "mean_error_a": mean_error_a,
            "mean_error_b": mean_error_b,
            "better_model": better_model,
            "error_reduction": float(abs(mean_error_a - mean_error_b) / max(mean_error_a, mean_error_b) * 100),
        }
