# PC-040-05A: Basic Bias Correction

**Ticket ID:** PC-040-05A  
**Epic:** [Epic-05A: Base Combination Strategies](../epics/Epic-05A.md)  
**User Story:** US-5  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `BasicBiasCorrectionModule` that identifies and corrects systematic biases in model combinations. Supports additive and multiplicative bias correction, time-dependent bias patterns (seasonal, weekly), validates against overfitting, and integrates with all combination methods to reduce systematic forecast errors.

**As a** ML engineer  
**I want** bias correction for model combinations  
**So that** I can reduce systematic forecast errors and improve accuracy

---

## ✅ Acceptance Criteria

- [ ] `BasicBiasCorrectionModule` identifies systematic biases
- [ ] Additive bias correction: `y_corrected = y_forecast + bias_offset`
- [ ] Multiplicative bias correction: `y_corrected = y_forecast * bias_factor`
- [ ] Time-dependent bias correction (seasonal, weekly patterns)
- [ ] Validation prevents overfitting to correction data
- [ ] Integration with all combination methods
- [ ] Rolling window bias estimation
- [ ] Bias pattern detection and analysis
- [ ] Cross-validation for correction validation
- [ ] Bias reduction >20% on validation set

---

## 🔧 Implementation Tasks

### 1. Create Bias Correction Module
- [ ] Create `src/models/combination/bias_correction.py`
- [ ] Import statistical utilities
- [ ] Import BaseCombiner
- [ ] Add module docstrings

### 2. Implement BasicBiasCorrectionModule Class
- [ ] Create standalone bias correction class
- [ ] Support additive and multiplicative methods
- [ ] Initialize with configuration
- [ ] Define bias estimation parameters

### 3. Implement Additive Bias Correction
- [ ] Create `_estimate_additive_bias()` method
- [ ] Calculate: bias = mean(y_true - y_pred)
- [ ] Support per-horizon bias
- [ ] Support time-dependent bias
- [ ] Return bias offset

### 4. Implement Multiplicative Bias Correction
- [ ] Create `_estimate_multiplicative_bias()` method
- [ ] Calculate: bias_factor = mean(y_true / y_pred)
- [ ] Handle zero predictions (add epsilon)
- [ ] Support per-horizon bias
- [ ] Return bias multiplier

### 5. Implement Rolling Window Bias Estimation
- [ ] Create `_rolling_bias_estimation()` method
- [ ] Use sliding window for recent bias
- [ ] Update bias estimates incrementally
- [ ] Weight recent observations higher
- [ ] Return time-varying bias

### 6. Implement Seasonal Bias Detection
- [ ] Create `_detect_seasonal_bias()` method
- [ ] Group errors by season/month
- [ ] Calculate seasonal bias patterns
- [ ] Test for statistical significance
- [ ] Return seasonal correction factors

### 7. Implement Weekly Bias Detection
- [ ] Create `_detect_weekly_bias()` method
- [ ] Group errors by day of week
- [ ] Calculate day-specific bias
- [ ] Test for weekday/weekend patterns
- [ ] Return weekly correction factors

### 8. Implement Hourly Bias Detection
- [ ] Create `_detect_hourly_bias()` method
- [ ] Group errors by hour of day
- [ ] Calculate hour-specific bias
- [ ] Identify peak/off-peak patterns
- [ ] Return hourly correction factors

### 9. Implement fit() Method
- [ ] Accept predictions and targets
- [ ] Estimate bias using training data
- [ ] Detect time-dependent patterns
- [ ] Validate bias significance
- [ ] Store correction parameters
- [ ] Mark as fitted

### 10. Implement correct() Method
- [ ] Accept predictions to correct
- [ ] Apply additive or multiplicative correction
- [ ] Apply time-dependent corrections
- [ ] Handle edge cases (negatives, zeros)
- [ ] Return corrected predictions
- [ ] Log correction summary

### 11. Implement Bias Validation
- [ ] Create `validate_bias_correction()` method
- [ ] Calculate pre-correction bias
- [ ] Calculate post-correction bias
- [ ] Compare bias reduction
- [ ] Test for overfitting
- [ ] Return validation metrics

### 12. Implement Overfitting Prevention
- [ ] Use cross-validation for bias estimation
- [ ] Regularize extreme correction factors
- [ ] Limit correction magnitude
- [ ] Test on hold-out data
- [ ] Reject corrections that degrade performance

### 13. Implement Integration with Combiners
- [ ] Create `BiasCorrectingCombiner` wrapper
- [ ] Accept any BaseCombiner instance
- [ ] Apply bias correction after combination
- [ ] Transparent integration
- [ ] Preserve combiner interface

### 14. Implement Bias Analysis Tools
- [ ] Create `analyze_bias_patterns()` method
- [ ] Generate bias distribution plots
- [ ] Identify systematic patterns
- [ ] Calculate bias statistics
- [ ] Return analysis report

### 15. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_bias_correction.py`
- [ ] Test additive correction
- [ ] Test multiplicative correction
- [ ] Test seasonal bias detection
- [ ] Test weekly/hourly patterns
- [ ] Test overfitting prevention
- [ ] Test integration with combiners

### 16. Write Integration Tests
- [ ] Test with real model predictions
- [ ] Test bias correction effectiveness
- [ ] Validate >20% bias reduction
- [ ] Test with different combination methods
- [ ] Cross-validation performance

### 17. Create Usage Examples
- [ ] Create `examples/bias_correction_demo.py`
- [ ] Show additive correction
- [ ] Show multiplicative correction
- [ ] Show seasonal patterns
- [ ] Compare before/after correction

---

## 💻 Implementation Details

### BasicBiasCorrectionModule Implementation

```python
"""Basic bias correction for model combinations."""
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BiasCorrectionConfig:
    """Configuration for bias correction."""
    
    correction_type: str = 'additive'  # 'additive', 'multiplicative', 'both'
    window_size: int = 48  # Rolling window for bias estimation
    detect_seasonal: bool = True
    detect_weekly: bool = True
    detect_hourly: bool = True
    min_samples: int = 20  # Minimum samples for bias estimation
    max_correction: float = 0.2  # Maximum correction as fraction (20%)
    significance_level: float = 0.05  # For statistical tests


class BasicBiasCorrectionModule:
    """
    Basic bias correction for forecasts.
    
    Identifies and corrects systematic biases in predictions
    using additive or multiplicative adjustments.
    
    Additive Correction:
        y_corrected = y_forecast + bias_offset
        bias_offset = mean(y_true - y_forecast)
    
    Multiplicative Correction:
        y_corrected = y_forecast * bias_factor
        bias_factor = mean(y_true / y_forecast)
    
    Time-Dependent Correction:
        Estimates different biases for:
        - Seasonal patterns (monthly)
        - Weekly patterns (day of week)
        - Hourly patterns (hour of day)
    
    Example:
        >>> corrector = BasicBiasCorrectionModule(
        ...     config={'correction_type': 'additive'}
        ... )
        >>> corrector.fit(predictions, targets)
        >>> corrected = corrector.correct(new_predictions)
        >>> print(f"Bias reduced by {corrector.bias_reduction:.1%}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize bias correction module.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = {}
        
        self.config = BiasCorrectionConfig(**config)
        
        self._fitted = False
        self._bias_offset: Optional[float] = None
        self._bias_factor: Optional[float] = None
        self._seasonal_bias: Dict[int, float] = {}  # {month: bias}
        self._weekly_bias: Dict[int, float] = {}  # {dow: bias}
        self._hourly_bias: Dict[int, float] = {}  # {hour: bias}
    
    def fit(
        self,
        predictions: pd.DataFrame,
        targets: pd.DataFrame
    ):
        """
        Fit bias correction on training data.
        
        Args:
            predictions: Forecast predictions DataFrame
            targets: True target values DataFrame
        """
        logger.info("Fitting bias correction module")
        
        pred_cols = [col for col in predictions.columns if col.startswith('pred_h')]
        
        if not pred_cols:
            raise ValueError("No prediction columns found")
        
        # Calculate errors
        all_errors = []
        all_pred_values = []
        all_true_values = []
        timestamps = []
        
        for col in pred_cols:
            horizon = int(col.split('_h')[1])
            target_col = f'target_h{horizon}'
            
            if target_col not in targets.columns:
                continue
            
            pred_values = predictions[col].values
            true_values = targets[target_col].values
            
            # Remove NaN
            valid_mask = ~(np.isnan(pred_values) | np.isnan(true_values))
            pred_values = pred_values[valid_mask]
            true_values = true_values[valid_mask]
            
            errors = true_values - pred_values
            
            all_errors.extend(errors)
            all_pred_values.extend(pred_values)
            all_true_values.extend(true_values)
            timestamps.extend(predictions.index[valid_mask])
        
        all_errors = np.array(all_errors)
        all_pred_values = np.array(all_pred_values)
        all_true_values = np.array(all_true_values)
        timestamps = pd.DatetimeIndex(timestamps)
        
        if len(all_errors) < self.config.min_samples:
            logger.warning(f"Insufficient samples for bias correction: {len(all_errors)}")
            self._fitted = False
            return
        
        # Estimate global bias
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
            self._seasonal_bias = self._detect_seasonal_bias(
                timestamps, all_errors
            )
        
        if self.config.detect_weekly:
            self._weekly_bias = self._detect_weekly_bias(
                timestamps, all_errors
            )
        
        if self.config.detect_hourly:
            self._hourly_bias = self._detect_hourly_bias(
                timestamps, all_errors
            )
        
        self._fitted = True
        logger.info("Bias correction module fitted successfully")
    
    def correct(
        self,
        predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply bias correction to predictions.
        
        Args:
            predictions: Predictions to correct
        
        Returns:
            Corrected predictions DataFrame
        """
        if not self._fitted:
            logger.warning("Bias correction not fitted, returning original predictions")
            return predictions.copy()
        
        corrected = predictions.copy()
        pred_cols = [col for col in corrected.columns if col.startswith('pred_h')]
        
        for col in pred_cols:
            values = corrected[col].values
            
            # Apply global correction
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
                values = self._apply_time_dependent_correction(
                    values, corrected.index
                )
            
            # Ensure non-negative for load forecasts
            values = np.maximum(values, 0)
            
            corrected[col] = values
        
        logger.debug(f"Applied bias correction to {len(pred_cols)} columns")
        
        return corrected
    
    def _estimate_additive_bias(self, errors: np.ndarray) -> float:
        """
        Estimate additive bias offset.
        
        Args:
            errors: Array of errors (true - predicted)
        
        Returns:
            Bias offset value
        """
        bias = np.mean(errors)
        
        # Limit correction magnitude
        max_abs_bias = np.abs(errors).mean() * self.config.max_correction
        bias = np.clip(bias, -max_abs_bias, max_abs_bias)
        
        return float(bias)
    
    def _estimate_multiplicative_bias(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray
    ) -> float:
        """
        Estimate multiplicative bias factor.
        
        Args:
            predictions: Predicted values
            true_values: True values
        
        Returns:
            Bias factor (multiplier)
        """
        epsilon = 1e-8
        ratios = true_values / (predictions + epsilon)
        
        # Remove outliers
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
        """
        Detect monthly/seasonal bias patterns.
        
        Args:
            timestamps: Timestamps for errors
            errors: Error values
        
        Returns:
            Dictionary mapping month to bias
        """
        monthly_errors = {}
        
        for month in range(1, 13):
            mask = timestamps.month == month
            if mask.sum() >= self.config.min_samples:
                monthly_errors[month] = np.mean(errors[mask])
        
        # Test for significance
        if len(monthly_errors) < 6:  # Need at least half the months
            return {}
        
        # Simple ANOVA-like test
        overall_mean = np.mean(errors)
        monthly_means = list(monthly_errors.values())
        
        # If variation is significant, return biases
        if np.std(monthly_means) > 0.1 * np.abs(overall_mean):
            logger.info(f"Detected seasonal bias pattern in {len(monthly_errors)} months")
            return monthly_errors
        
        return {}
    
    def _detect_weekly_bias(
        self,
        timestamps: pd.DatetimeIndex,
        errors: np.ndarray
    ) -> Dict[int, float]:
        """
        Detect day-of-week bias patterns.
        
        Args:
            timestamps: Timestamps for errors
            errors: Error values
        
        Returns:
            Dictionary mapping day-of-week to bias
        """
        weekly_errors = {}
        
        for dow in range(7):  # Monday=0, Sunday=6
            mask = timestamps.dayofweek == dow
            if mask.sum() >= self.config.min_samples:
                weekly_errors[dow] = np.mean(errors[mask])
        
        if len(weekly_errors) < 5:  # Need most days
            return {}
        
        # Test for weekday/weekend difference
        weekday_errors = [weekly_errors[d] for d in range(5) if d in weekly_errors]
        weekend_errors = [weekly_errors[d] for d in [5, 6] if d in weekly_errors]
        
        if weekday_errors and weekend_errors:
            t_stat, p_value = stats.ttest_ind(weekday_errors, weekend_errors)
            
            if p_value < self.config.significance_level:
                logger.info("Detected significant weekly bias pattern")
                return weekly_errors
        
        return {}
    
    def _detect_hourly_bias(
        self,
        timestamps: pd.DatetimeIndex,
        errors: np.ndarray
    ) -> Dict[int, float]:
        """
        Detect hour-of-day bias patterns.
        
        Args:
            timestamps: Timestamps for errors
            errors: Error values
        
        Returns:
            Dictionary mapping hour to bias
        """
        # For semi-hourly data, use semi-hourly periods (0-47)
        hourly_errors = {}
        
        for hour in range(48):
            mask = (timestamps.hour * 2 + (timestamps.minute // 30)) == hour
            if mask.sum() >= self.config.min_samples:
                hourly_errors[hour] = np.mean(errors[mask])
        
        if len(hourly_errors) < 24:  # Need at least half the hours
            return {}
        
        # Test for variation
        if np.std(list(hourly_errors.values())) > 0.05 * np.abs(np.mean(errors)):
            logger.info(f"Detected hourly bias pattern in {len(hourly_errors)} periods")
            return hourly_errors
        
        return {}
    
    def _apply_time_dependent_correction(
        self,
        values: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Apply time-dependent bias corrections.
        
        Args:
            values: Values to correct
            timestamps: Timestamps for values
        
        Returns:
            Corrected values
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
            hour_idx = ts.hour * 2 + (ts.minute // 30)
            if self._hourly_bias and hour_idx in self._hourly_bias:
                correction += self._hourly_bias[hour_idx]
            
            corrected[i] += correction
        
        return corrected
    
    def validate_bias_correction(
        self,
        predictions: pd.DataFrame,
        targets: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate bias correction effectiveness.
        
        Args:
            predictions: Original predictions
            targets: True targets
        
        Returns:
            Dictionary with validation metrics
        """
        # Calculate original bias
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
        
        original_bias /= n_samples
        corrected_bias /= n_samples
        
        bias_reduction = (original_bias - corrected_bias) / original_bias
        
        return {
            'original_bias': original_bias,
            'corrected_bias': corrected_bias,
            'bias_reduction': bias_reduction,
            'n_samples': n_samples
        }
```

---

## 🧪 Testing & Validation

```python
"""Tests for bias correction module."""
import pytest
import pandas as pd
import numpy as np

from src.models.combination.bias_correction import BasicBiasCorrectionModule


@pytest.fixture
def biased_predictions():
    """Create predictions with systematic bias."""
    dates = pd.date_range('2024-01-01', periods=100, freq='30min')
    
    # Predictions with positive bias (+50)
    predictions = pd.DataFrame({
        'pred_h0': np.random.randn(100) * 50 + 1050,
        'pred_h1': np.random.randn(100) * 50 + 1050
    }, index=dates)
    
    # True values
    targets = pd.DataFrame({
        'target_h0': np.random.randn(100) * 50 + 1000,
        'target_h1': np.random.randn(100) * 50 + 1000
    }, index=dates)
    
    return predictions, targets


def test_additive_correction(biased_predictions):
    """Test additive bias correction."""
    predictions, targets = biased_predictions
    
    corrector = BasicBiasCorrectionModule(config={'correction_type': 'additive'})
    corrector.fit(predictions, targets)
    
    assert corrector._fitted
    assert corrector._bias_offset is not None
    
    # Bias should be approximately -50
    assert -60 < corrector._bias_offset < -40


def test_bias_reduction(biased_predictions):
    """Test bias reduction effectiveness."""
    predictions, targets = biased_predictions
    
    corrector = BasicBiasCorrectionModule()
    corrector.fit(predictions, targets)
    
    validation = corrector.validate_bias_correction(predictions, targets)
    
    # Should achieve >20% bias reduction
    assert validation['bias_reduction'] > 0.2
```

---

## 📝 Technical Notes

### Bias Types
- **Additive:** Constant offset (e.g., +50 MW)
- **Multiplicative:** Proportional scaling (e.g., ×1.05)
- **Time-dependent:** Varies by season/day/hour

### Overfitting Prevention
- Cross-validation for bias estimation
- Limit correction magnitude (20% max)
- Statistical significance testing
- Hold-out validation

---

## 🔗 Dependencies

**Depends On:**
- PC-036-05A: Base Combiner Interface

**Integrates With:**
- PC-037-05A: Simple Averaging
- PC-038-05A: Weighted Voting
- PC-039-05A: Best Model Selector

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Additive correction working
- [ ] Multiplicative correction working
- [ ] Time-dependent bias detection functional
- [ ] Overfitting prevention validated
- [ ] Bias reduction >20% achieved
- [ ] Integration with combiners working
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Previous:** [PC-039-05A: Best Model Selector](PC-039-05A-best-model-selector.md)  
**Next:** [PC-041-05A: Weight Validation Framework](PC-041-05A-weight-validation-framework.md)
