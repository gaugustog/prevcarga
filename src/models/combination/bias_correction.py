"""Basic bias correction for model combinations.

This module provides systematic bias identification and correction for
forecast combinations in the PrevCarga system. Supports additive and
multiplicative corrections with time-dependent patterns.

Example:
    >>> corrector = BasicBiasCorrectionModule(
    ...     config={'correction_type': 'additive'}
    ... )
    >>> corrector.fit(predictions, targets)
    >>> corrected = corrector.correct(new_predictions)
    >>> print(f"Bias reduced by {corrector.bias_reduction:.1%}")
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BiasCorrectionConfig:
    """Configuration for bias correction.

    Attributes:
        correction_type: Type of correction ('additive', 'multiplicative', 'both')
        window_size: Rolling window size for bias estimation
        detect_seasonal: Enable seasonal bias detection
        detect_weekly: Enable weekly bias detection
        detect_hourly: Enable hourly bias detection
        min_samples: Minimum samples required for bias estimation
        max_correction: Maximum correction magnitude as fraction
        significance_level: Significance level for statistical tests
    """

    correction_type: str = 'additive'
    window_size: int = 48
    detect_seasonal: bool = True
    detect_weekly: bool = True
    detect_hourly: bool = True
    min_samples: int = 20
    max_correction: float = 0.2
    significance_level: float = 0.05


class BasicBiasCorrectionModule:
    """Basic bias correction for forecasts.

    Identifies and corrects systematic biases in predictions using
    additive or multiplicative adjustments with optional time-dependent
    patterns.

    Correction Methods:
        Additive:
            y_corrected = y_forecast + bias_offset
            where bias_offset = mean(y_true - y_forecast)

        Multiplicative:
            y_corrected = y_forecast * bias_factor
            where bias_factor = mean(y_true / y_forecast)

    Time-Dependent Patterns:
        - Seasonal (monthly)
        - Weekly (day of week)
        - Hourly (hour of day or semi-hourly period)

    Overfitting Prevention:
        - Maximum correction magnitude limits
        - Statistical significance tests
        - Outlier filtering
        - Regularization of extreme values

    Example:
        >>> config = {'correction_type': 'both', 'detect_seasonal': True}
        >>> corrector = BasicBiasCorrectionModule(config=config)
        >>>
        >>> # Fit on training data
        >>> corrector.fit(train_predictions, train_targets)
        >>>
        >>> # Apply to new predictions
        >>> corrected = corrector.correct(test_predictions)
        >>>
        >>> # Validate effectiveness
        >>> metrics = corrector.validate_bias_correction(
        ...     test_predictions, test_targets
        ... )
        >>> print(f"Bias reduction: {metrics['bias_reduction']:.1%}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize bias correction module.

        Args:
            config: Configuration dictionary for bias correction parameters.
                If None, uses default configuration.
        """
        if config is None:
            config = {}

        self.config = BiasCorrectionConfig(**config)

        self._fitted = False
        self._bias_offset: Optional[float] = None
        self._bias_factor: Optional[float] = None
        self._seasonal_bias: Dict[int, float] = {}
        self._weekly_bias: Dict[int, float] = {}
        self._hourly_bias: Dict[int, float] = {}
        self.bias_reduction: float = 0.0

    def fit(
        self,
        predictions: pd.DataFrame,
        targets: pd.DataFrame
    ) -> None:
        """Fit bias correction on training data.

        Estimates systematic biases by comparing predictions to targets.
        Detects global biases and time-dependent patterns based on
        configuration.

        Args:
            predictions: Forecast predictions DataFrame with columns
                'pred_h0', 'pred_h1', etc.
            targets: True target values DataFrame with columns
                'target_h0', 'target_h1', etc.

        Raises:
            ValueError: If no valid prediction columns found
        """
        logger.info("Fitting bias correction module")

        pred_cols = [col for col in predictions.columns if col.startswith('pred_h')]

        if not pred_cols:
            raise ValueError("No prediction columns found with 'pred_h' prefix")

        # Collect all errors and values
        all_errors = []
        all_pred_values = []
        all_true_values = []
        timestamps = []

        for col in pred_cols:
            horizon = int(col.split('_h')[1])
            target_col = f'target_h{horizon}'

            if target_col not in targets.columns:
                logger.warning(f"Target column {target_col} not found, skipping")
                continue

            pred_values = predictions[col].values
            true_values = targets[target_col].values

            # Remove NaN values
            valid_mask = ~(np.isnan(pred_values) | np.isnan(true_values))
            pred_values_valid = pred_values[valid_mask]
            true_values_valid = true_values[valid_mask]

            errors = true_values_valid - pred_values_valid

            all_errors.extend(errors)
            all_pred_values.extend(pred_values_valid)
            all_true_values.extend(true_values_valid)
            timestamps.extend(predictions.index[valid_mask])

        # Convert to arrays
        all_errors = np.array(all_errors)
        all_pred_values = np.array(all_pred_values)
        all_true_values = np.array(all_true_values)
        timestamps = pd.DatetimeIndex(timestamps)

        if len(all_errors) < self.config.min_samples:
            logger.warning(
                f"Insufficient samples for bias correction: "
                f"{len(all_errors)} < {self.config.min_samples}"
            )
            self._fitted = False
            return

        # Estimate global biases
        if self.config.correction_type in ['additive', 'both']:
            self._bias_offset = self._estimate_additive_bias(all_errors)
            logger.info(f"Estimated additive bias: {self._bias_offset:.2f}")

        if self.config.correction_type in ['multiplicative', 'both']:
            self._bias_factor = self._estimate_multiplicative_bias(
                all_pred_values, all_true_values
            )
            logger.info(f"Estimated multiplicative bias factor: {self._bias_factor:.4f}")

        # Detect time-dependent biases
        if self.config.detect_seasonal:
            self._seasonal_bias = self._detect_seasonal_bias(timestamps, all_errors)
            if self._seasonal_bias:
                logger.info(f"Detected seasonal bias in {len(self._seasonal_bias)} months")

        if self.config.detect_weekly:
            self._weekly_bias = self._detect_weekly_bias(timestamps, all_errors)
            if self._weekly_bias:
                logger.info(f"Detected weekly bias in {len(self._weekly_bias)} days")

        if self.config.detect_hourly:
            self._hourly_bias = self._detect_hourly_bias(timestamps, all_errors)
            if self._hourly_bias:
                logger.info(f"Detected hourly bias in {len(self._hourly_bias)} periods")

        self._fitted = True
        logger.info("Bias correction module fitted successfully")

    def correct(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply bias correction to predictions.

        Applies fitted bias corrections (global and time-dependent) to
        new predictions. Ensures non-negative values for load forecasts.

        Args:
            predictions: Predictions DataFrame to correct

        Returns:
            Corrected predictions DataFrame with same structure as input
        """
        if not self._fitted:
            logger.warning("Bias correction not fitted, returning original predictions")
            return predictions.copy()

        corrected = predictions.copy()
        pred_cols = [col for col in corrected.columns if col.startswith('pred_h')]

        for col in pred_cols:
            values = corrected[col].values.copy()

            # Apply global corrections
            if self.config.correction_type == 'additive' and self._bias_offset is not None:
                values = values + self._bias_offset

            elif self.config.correction_type == 'multiplicative' and self._bias_factor is not None:
                values = values * self._bias_factor

            elif self.config.correction_type == 'both':
                if self._bias_offset is not None:
                    values = values + self._bias_offset
                if self._bias_factor is not None:
                    values = values * self._bias_factor

            # Apply time-dependent corrections
            if self._seasonal_bias or self._weekly_bias or self._hourly_bias:
                values = self._apply_time_dependent_correction(values, corrected.index)

            # Ensure non-negative for load forecasts
            values = np.maximum(values, 0)

            corrected[col] = values

        logger.debug(f"Applied bias correction to {len(pred_cols)} columns")

        return corrected

    def validate_bias_correction(
        self,
        predictions: pd.DataFrame,
        targets: pd.DataFrame
    ) -> Dict[str, float]:
        """Validate bias correction effectiveness.

        Compares bias before and after correction to measure effectiveness.
        Calculates mean absolute error reduction.

        Args:
            predictions: Original predictions DataFrame
            targets: True target values DataFrame

        Returns:
            Dictionary containing:
                - original_bias: Mean absolute error before correction
                - corrected_bias: Mean absolute error after correction
                - bias_reduction: Fractional reduction (0-1)
                - n_samples: Number of samples used
        """
        pred_cols = [col for col in predictions.columns if col.startswith('pred_h')]

        original_bias = 0.0
        corrected_bias = 0.0
        n_samples = 0

        corrected_predictions = self.correct(predictions)

        for col in pred_cols:
            horizon = int(col.split('_h')[1])
            target_col = f'target_h{horizon}'

            if target_col not in targets.columns:
                continue

            orig_errors = targets[target_col].values - predictions[col].values
            corr_errors = targets[target_col].values - corrected_predictions[col].values

            valid_mask = ~(np.isnan(orig_errors) | np.isnan(corr_errors))

            original_bias += np.sum(np.abs(orig_errors[valid_mask]))
            corrected_bias += np.sum(np.abs(corr_errors[valid_mask]))
            n_samples += valid_mask.sum()

        if n_samples == 0:
            logger.warning("No valid samples for bias validation")
            return {
                'original_bias': 0.0,
                'corrected_bias': 0.0,
                'bias_reduction': 0.0,
                'n_samples': 0
            }

        original_bias /= n_samples
        corrected_bias /= n_samples

        bias_reduction = (original_bias - corrected_bias) / original_bias if original_bias > 0 else 0.0
        self.bias_reduction = bias_reduction

        logger.info(f"Bias reduction: {bias_reduction:.1%} (MAE: {original_bias:.2f} → {corrected_bias:.2f})")

        return {
            'original_bias': float(original_bias),
            'corrected_bias': float(corrected_bias),
            'bias_reduction': float(bias_reduction),
            'n_samples': int(n_samples)
        }

    def _estimate_additive_bias(self, errors: np.ndarray) -> float:
        """Estimate additive bias offset.

        Calculates mean error with magnitude limiting to prevent
        overfitting.

        Args:
            errors: Array of errors (true - predicted)

        Returns:
            Bias offset value to add to predictions
        """
        bias = np.mean(errors)

        # Limit correction magnitude to prevent overcorrection
        max_abs_bias = np.abs(errors).mean() * self.config.max_correction
        bias = np.clip(bias, -max_abs_bias, max_abs_bias)

        return float(bias)

    def _estimate_multiplicative_bias(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray
    ) -> float:
        """Estimate multiplicative bias factor.

        Calculates mean ratio of true/predicted with outlier filtering
        and magnitude limiting.

        Args:
            predictions: Predicted values
            true_values: True values

        Returns:
            Bias factor to multiply with predictions
        """
        epsilon = 1e-8
        ratios = true_values / (predictions + epsilon)

        # Remove outliers using IQR method
        q25, q75 = np.percentile(ratios, [25, 75])
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr

        filtered_ratios = ratios[(ratios >= lower) & (ratios <= upper)]

        if len(filtered_ratios) == 0:
            return 1.0

        bias_factor = np.mean(filtered_ratios)

        # Limit correction magnitude
        max_factor = 1.0 + self.config.max_correction
        min_factor = 1.0 - self.config.max_correction
        bias_factor = np.clip(bias_factor, min_factor, max_factor)

        return float(bias_factor)

    def _detect_seasonal_bias(
        self,
        timestamps: pd.DatetimeIndex,
        errors: np.ndarray
    ) -> Dict[int, float]:
        """Detect monthly/seasonal bias patterns.

        Groups errors by month and tests for significant variation.

        Args:
            timestamps: Timestamps for errors
            errors: Error values

        Returns:
            Dictionary mapping month (1-12) to bias offset
        """
        monthly_errors = {}

        for month in range(1, 13):
            mask = timestamps.month == month
            if mask.sum() >= self.config.min_samples:
                monthly_errors[month] = float(np.mean(errors[mask]))

        # Need at least half the months
        if len(monthly_errors) < 6:
            return {}

        # Test for significant variation
        overall_mean = np.mean(errors)
        monthly_means = list(monthly_errors.values())

        # If standard deviation of monthly means is significant, return biases
        if np.std(monthly_means) > 0.1 * np.abs(overall_mean):
            return monthly_errors

        return {}

    def _detect_weekly_bias(
        self,
        timestamps: pd.DatetimeIndex,
        errors: np.ndarray
    ) -> Dict[int, float]:
        """Detect day-of-week bias patterns.

        Groups errors by day of week and tests for weekday/weekend
        differences.

        Args:
            timestamps: Timestamps for errors
            errors: Error values

        Returns:
            Dictionary mapping day-of-week (0=Mon, 6=Sun) to bias offset
        """
        weekly_errors = {}

        for dow in range(7):
            mask = timestamps.dayofweek == dow
            if mask.sum() >= self.config.min_samples:
                weekly_errors[dow] = float(np.mean(errors[mask]))

        # Need most days
        if len(weekly_errors) < 5:
            return {}

        # Test for weekday/weekend difference
        weekday_errors = [weekly_errors[d] for d in range(5) if d in weekly_errors]
        weekend_errors = [weekly_errors[d] for d in [5, 6] if d in weekly_errors]

        if weekday_errors and weekend_errors:
            try:
                t_stat, p_value = stats.ttest_ind(weekday_errors, weekend_errors)

                if p_value < self.config.significance_level:
                    logger.debug(f"Significant weekly bias detected (p={p_value:.4f})")
                    return weekly_errors
            except Exception as e:
                logger.warning(f"Failed to test weekly bias significance: {e}")

        return {}

    def _detect_hourly_bias(
        self,
        timestamps: pd.DatetimeIndex,
        errors: np.ndarray
    ) -> Dict[int, float]:
        """Detect hour-of-day bias patterns.

        For semi-hourly data, groups errors by semi-hourly period (0-47).
        Tests for significant variation across periods.

        Args:
            timestamps: Timestamps for errors
            errors: Error values

        Returns:
            Dictionary mapping period index (0-47) to bias offset
        """
        hourly_errors = {}

        # For semi-hourly data (48 periods per day)
        for period in range(48):
            mask = (timestamps.hour * 2 + (timestamps.minute // 30)) == period
            if mask.sum() >= self.config.min_samples:
                hourly_errors[period] = float(np.mean(errors[mask]))

        # Need at least half the periods
        if len(hourly_errors) < 24:
            return {}

        # Test for variation
        period_std = np.std(list(hourly_errors.values()))
        overall_mean_abs = np.abs(np.mean(errors))

        if period_std > 0.05 * overall_mean_abs:
            return hourly_errors

        return {}

    def _apply_time_dependent_correction(
        self,
        values: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> np.ndarray:
        """Apply time-dependent bias corrections.

        Combines seasonal, weekly, and hourly corrections for each
        timestamp.

        Args:
            values: Values to correct
            timestamps: Timestamps for values

        Returns:
            Corrected values array
        """
        corrected = values.copy()

        for i, ts in enumerate(timestamps):
            correction = 0.0

            # Apply seasonal correction
            if self._seasonal_bias and ts.month in self._seasonal_bias:
                correction += self._seasonal_bias[ts.month]

            # Apply weekly correction
            if self._weekly_bias and ts.dayofweek in self._weekly_bias:
                correction += self._weekly_bias[ts.dayofweek]

            # Apply hourly correction
            period_idx = ts.hour * 2 + (ts.minute // 30)
            if self._hourly_bias and period_idx in self._hourly_bias:
                correction += self._hourly_bias[period_idx]

            corrected[i] += correction

        return corrected

    def __repr__(self) -> str:
        """String representation of bias correction module."""
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"BasicBiasCorrectionModule("
            f"type={self.config.correction_type}, "
            f"status={status})"
        )
