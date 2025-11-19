# PC-045-05B: Advanced Bias Correction Module

**Ticket ID:** PC-045-05B  
**Epic:** [Epic-05B: Advanced Ensemble Methods](../epics/Epic-05B.md)  
**User Story:** US-4  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `AdvancedBiasCorrectionModule` that handles complex systematic errors using non-linear methods (splines, decision trees), conditional bias correction based on forecast context, and temporal bias decomposition (trend, seasonal, weekly, daily). Includes cross-validation for bias model selection and significance testing to prevent overfitting.

**As a** forecasting analyst  
**I want** sophisticated bias correction that handles complex systematic errors  
**So that** I can eliminate non-linear, time-dependent, and conditional biases

---

## ✅ Acceptance Criteria

- [ ] `AdvancedBiasCorrectionModule` handles multiple bias types
- [ ] Non-linear bias correction using spline regression or tree methods
- [ ] Conditional bias correction based on forecast context
- [ ] Temporal bias patterns (hourly, daily, seasonal) identified and corrected
- [ ] Bias correction validation prevents overfitting
- [ ] Spline-based non-linear modeling
- [ ] Decision tree conditional bias
- [ ] Temporal decomposition (STL-style)
- [ ] Cross-validation for model selection
- [ ] Significance testing for bias reduction

---

## 🔧 Implementation Tasks

### 1. Setup Advanced Bias Module
- [ ] Create `src/models/combination/advanced_bias_correction.py`
- [ ] Import sklearn preprocessing (SplineTransformer)
- [ ] Import sklearn tree models
- [ ] Import statsmodels for decomposition
- [ ] Add module docstrings

### 2. Implement AdvancedBiasCorrectionModule Class
- [ ] Define configuration schema
- [ ] Initialize bias correction models
- [ ] Setup temporal decomposition
- [ ] Configure validation strategy
- [ ] Store bias patterns

### 3. Implement Non-Linear Bias Detection
- [ ] Create `_detect_nonlinear_bias()` method
- [ ] Calculate residuals (true - pred)
- [ ] Test for non-linear patterns
- [ ] Use spline regression for curve fitting
- [ ] Test improvement over linear bias
- [ ] Return non-linear bias function

### 4. Implement Spline-Based Correction
- [ ] Create `SplineBiasCorrector` class
- [ ] Use SplineTransformer for basis functions
- [ ] Fit spline to residuals vs predictions
- [ ] Configure knot placement
- [ ] Apply spline correction: y_corrected = y + spline(y)
- [ ] Return corrected predictions

### 5. Implement Tree-Based Conditional Correction
- [ ] Create `TreeBiasCorrector` class
- [ ] Train DecisionTreeRegressor on residuals
- [ ] Use context features as inputs
- [ ] Configure tree depth (avoid overfitting)
- [ ] Apply conditional correction
- [ ] Return corrected predictions

### 6. Implement Temporal Bias Decomposition
- [ ] Create `_decompose_temporal_bias()` method
- [ ] Decompose into trend, seasonal, weekly, daily
- [ ] Use STL decomposition or custom
- [ ] Extract each component separately
- [ ] Test significance of each component
- [ ] Return decomposition dict

### 7. Implement Trend Bias Correction
- [ ] Create `_correct_trend_bias()` method
- [ ] Detect systematic trend in residuals
- [ ] Fit linear or polynomial trend
- [ ] Apply detrending correction
- [ ] Validate trend significance
- [ ] Return detrended predictions

### 8. Implement Seasonal Bias Correction
- [ ] Create `_correct_seasonal_bias()` method
- [ ] Group residuals by month/season
- [ ] Calculate seasonal bias per period
- [ ] Test seasonal effect significance
- [ ] Apply seasonal correction
- [ ] Return corrected predictions

### 9. Implement Weekly Bias Correction
- [ ] Create `_correct_weekly_bias()` method
- [ ] Group residuals by day of week
- [ ] Calculate weekday vs weekend bias
- [ ] Test weekly effect significance
- [ ] Apply weekly correction
- [ ] Return corrected predictions

### 10. Implement Daily/Hourly Bias Correction
- [ ] Create `_correct_hourly_bias()` method
- [ ] Group residuals by hour of day
- [ ] Calculate hourly bias pattern
- [ ] Test hourly effect significance
- [ ] Apply hourly correction
- [ ] Return corrected predictions

### 11. Implement Combined Bias Correction
- [ ] Create `correct_bias()` method
- [ ] Apply corrections in sequence
- [ ] Combine non-linear + temporal corrections
- [ ] Handle interaction between corrections
- [ ] Validate combined improvement
- [ ] Return fully corrected predictions

### 12. Implement Bias Model Selection
- [ ] Create `_select_bias_model()` method
- [ ] Cross-validate different bias models
- [ ] Compare linear vs non-linear vs conditional
- [ ] Select best performing model
- [ ] Prevent overfitting with validation
- [ ] Return selected model

### 13. Implement Bias Significance Testing
- [ ] Create `_test_bias_significance()` method
- [ ] Calculate bias magnitude
- [ ] Perform statistical test (t-test)
- [ ] Test if bias reduction is significant
- [ ] Require p < 0.05 for correction
- [ ] Return significance results

### 14. Implement Overfitting Prevention
- [ ] Use cross-validation for all bias models
- [ ] Limit model complexity (tree depth, spline degrees)
- [ ] Require minimum sample size per group
- [ ] Test on hold-out validation set
- [ ] Monitor train vs validation performance
- [ ] Reject correction if validation worse

### 15. Handle Edge Cases
- [ ] Handle insufficient data for decomposition
- [ ] Handle missing temporal periods
- [ ] Handle constant bias (no pattern)
- [ ] Handle extreme outliers in residuals
- [ ] Validate correction bounds

### 16. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_advanced_bias_correction.py`
- [ ] Test non-linear bias detection
- [ ] Test spline correction
- [ ] Test tree-based conditional correction
- [ ] Test temporal decomposition
- [ ] Test significance testing
- [ ] Test overfitting prevention

### 17. Write Integration Tests
- [ ] Test with synthetic bias patterns
- [ ] Test with real forecast errors
- [ ] Test combined corrections
- [ ] Validate bias reduction >30%
- [ ] Test computational performance

### 18. Create Usage Examples
- [ ] Create `examples/advanced_bias_correction_demo.py`
- [ ] Show non-linear bias correction
- [ ] Show temporal decomposition
- [ ] Show conditional correction
- [ ] Compare with basic bias correction

---

## 💻 Implementation Details

### AdvancedBiasCorrectionModule Implementation

```python
"""Advanced bias correction with non-linear and temporal methods."""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AdvancedBiasCorrectionConfig:
    """Configuration for advanced bias correction."""
    
    correction_type: str = 'combined'  # 'spline', 'tree', 'temporal', 'combined'
    spline_n_knots: int = 5
    spline_degree: int = 3
    tree_max_depth: int = 4
    min_samples_per_group: int = 20
    significance_level: float = 0.05
    use_cross_validation: bool = True
    n_cv_folds: int = 5


class SplineBiasCorrector:
    """Spline-based non-linear bias correction."""
    
    def __init__(self, n_knots: int = 5, degree: int = 3):
        self.n_knots = n_knots
        self.degree = degree
        self.spline_transformer = SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            include_bias=True
        )
        self.ridge_model = Ridge(alpha=1.0)
        self.fitted = False
    
    def fit(self, predictions: np.ndarray, residuals: np.ndarray):
        """
        Fit spline to residuals.
        
        Args:
            predictions: Model predictions
            residuals: True - predicted
        """
        X_spline = self.spline_transformer.fit_transform(predictions.reshape(-1, 1))
        self.ridge_model.fit(X_spline, residuals)
        self.fitted = True
    
    def correct(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply spline correction.
        
        Args:
            predictions: Predictions to correct
        
        Returns:
            Corrected predictions
        """
        if not self.fitted:
            return predictions
        
        X_spline = self.spline_transformer.transform(predictions.reshape(-1, 1))
        bias_estimate = self.ridge_model.predict(X_spline)
        
        return predictions + bias_estimate


class TreeBiasCorrector:
    """Tree-based conditional bias correction."""
    
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.tree_model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=20,
            random_state=42
        )
        self.fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray,
        context_features: np.ndarray
    ):
        """
        Fit decision tree to residuals.
        
        Args:
            predictions: Model predictions
            residuals: True - predicted
            context_features: Context features (hour, dow, etc.)
        """
        # Combine predictions and context as features
        X = np.column_stack([predictions, context_features])
        self.tree_model.fit(X, residuals)
        self.fitted = True
    
    def correct(
        self,
        predictions: np.ndarray,
        context_features: np.ndarray
    ) -> np.ndarray:
        """
        Apply conditional correction.
        
        Args:
            predictions: Predictions to correct
            context_features: Context features
        
        Returns:
            Corrected predictions
        """
        if not self.fitted:
            return predictions
        
        X = np.column_stack([predictions, context_features])
        bias_estimate = self.tree_model.predict(X)
        
        return predictions + bias_estimate


class AdvancedBiasCorrectionModule:
    """
    Advanced bias correction with non-linear and temporal methods.
    
    Bias Types:
    - Non-linear: Spline regression on residuals
    - Conditional: Decision tree based on context
    - Temporal: Decomposition into trend/seasonal/weekly/daily
    
    Correction Function:
        y_corrected = y + bias_spline(y) + bias_temporal(t) + bias_conditional(context)
    
    Overfitting Prevention:
    - Cross-validation for model selection
    - Minimum sample size per group
    - Significance testing (p < 0.05)
    - Regularization in spline fitting
    
    Validation:
        bias_reduction = (original_bias - corrected_bias) / original_bias
        require: bias_reduction > 30% AND p < 0.05
    
    Example:
        >>> bias_corrector = AdvancedBiasCorrectionModule(
        ...     config={'correction_type': 'combined'}
        ... )
        >>> bias_corrector.fit(predictions, targets, timestamps)
        >>> corrected = bias_corrector.correct_bias(predictions, timestamps)
        >>> print(f"Bias reduction: {bias_corrector.bias_reduction:.1%}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize advanced bias correction module.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = {}
        
        config_obj = AdvancedBiasCorrectionConfig(**config)
        self.correction_type = config_obj.correction_type
        self.spline_n_knots = config_obj.spline_n_knots
        self.spline_degree = config_obj.spline_degree
        self.tree_max_depth = config_obj.tree_max_depth
        self.min_samples_per_group = config_obj.min_samples_per_group
        self.significance_level = config_obj.significance_level
        self.use_cross_validation = config_obj.use_cross_validation
        self.n_cv_folds = config_obj.n_cv_folds
        
        # Bias correction models
        self.spline_corrector: Optional[SplineBiasCorrector] = None
        self.tree_corrector: Optional[TreeBiasCorrector] = None
        self.temporal_bias: Dict[str, Any] = {}
        
        # Performance tracking
        self.bias_reduction: Optional[float] = None
        self.fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: pd.DatetimeIndex,
        context_features: Optional[np.ndarray] = None
    ):
        """
        Fit bias correction models.
        
        Args:
            predictions: Model predictions
            targets: True values
            timestamps: Datetime index
            context_features: Optional context features
        """
        logger.info(f"Fitting advanced bias correction ({self.correction_type})")
        
        # Calculate residuals
        residuals = targets - predictions
        
        # Fit spline correction
        if self.correction_type in ['spline', 'combined']:
            logger.info("Fitting spline bias correction")
            self.spline_corrector = SplineBiasCorrector(
                n_knots=self.spline_n_knots,
                degree=self.spline_degree
            )
            self.spline_corrector.fit(predictions, residuals)
        
        # Fit tree-based conditional correction
        if self.correction_type in ['tree', 'combined'] and context_features is not None:
            logger.info("Fitting tree-based conditional correction")
            self.tree_corrector = TreeBiasCorrector(max_depth=self.tree_max_depth)
            self.tree_corrector.fit(predictions, residuals, context_features)
        
        # Fit temporal corrections
        if self.correction_type in ['temporal', 'combined']:
            logger.info("Decomposing temporal bias patterns")
            self.temporal_bias = self._decompose_temporal_bias(
                residuals, timestamps
            )
        
        # Calculate bias reduction
        self._calculate_bias_reduction(predictions, targets, timestamps, context_features)
        
        self.fitted = True
        logger.info(f"Bias reduction: {self.bias_reduction:.1%}")
    
    def correct_bias(
        self,
        predictions: np.ndarray,
        timestamps: pd.DatetimeIndex,
        context_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply bias correction.
        
        Args:
            predictions: Predictions to correct
            timestamps: Datetime index
            context_features: Optional context features
        
        Returns:
            Corrected predictions
        """
        if not self.fitted:
            logger.warning("Bias corrector not fitted, returning original predictions")
            return predictions
        
        corrected = predictions.copy()
        
        # Apply spline correction
        if self.spline_corrector is not None:
            corrected = self.spline_corrector.correct(corrected)
        
        # Apply tree-based conditional correction
        if self.tree_corrector is not None and context_features is not None:
            corrected = self.tree_corrector.correct(corrected, context_features)
        
        # Apply temporal corrections
        if self.temporal_bias:
            corrected = self._apply_temporal_correction(corrected, timestamps)
        
        return corrected
    
    def _decompose_temporal_bias(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """
        Decompose temporal bias into components.
        
        Args:
            residuals: Prediction residuals
            timestamps: Datetime index
        
        Returns:
            Dictionary with temporal bias components
        """
        temporal_bias = {}
        
        # Seasonal bias (monthly)
        if len(residuals) >= 12 * self.min_samples_per_group:
            seasonal_bias = {}
            for month in range(1, 13):
                mask = timestamps.month == month
                if mask.sum() >= self.min_samples_per_group:
                    seasonal_bias[month] = np.mean(residuals[mask])
            
            # Test significance
            if self._test_seasonal_significance(residuals, timestamps):
                temporal_bias['seasonal'] = seasonal_bias
        
        # Weekly bias
        if len(residuals) >= 7 * self.min_samples_per_group:
            weekly_bias = {}
            for dow in range(7):
                mask = timestamps.dayofweek == dow
                if mask.sum() >= self.min_samples_per_group:
                    weekly_bias[dow] = np.mean(residuals[mask])
            
            # Test significance
            if self._test_weekly_significance(residuals, timestamps):
                temporal_bias['weekly'] = weekly_bias
        
        # Hourly bias
        if len(residuals) >= 24 * self.min_samples_per_group:
            hourly_bias = {}
            for hour in range(24):
                mask = timestamps.hour == hour
                if mask.sum() >= self.min_samples_per_group:
                    hourly_bias[hour] = np.mean(residuals[mask])
            
            # Test significance
            if self._test_hourly_significance(residuals, timestamps):
                temporal_bias['hourly'] = hourly_bias
        
        return temporal_bias
    
    def _apply_temporal_correction(
        self,
        predictions: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Apply temporal bias corrections.
        
        Args:
            predictions: Predictions to correct
            timestamps: Datetime index
        
        Returns:
            Corrected predictions
        """
        corrected = predictions.copy()
        
        # Apply seasonal correction
        if 'seasonal' in self.temporal_bias:
            for i, ts in enumerate(timestamps):
                month = ts.month
                if month in self.temporal_bias['seasonal']:
                    corrected[i] += self.temporal_bias['seasonal'][month]
        
        # Apply weekly correction
        if 'weekly' in self.temporal_bias:
            for i, ts in enumerate(timestamps):
                dow = ts.dayofweek
                if dow in self.temporal_bias['weekly']:
                    corrected[i] += self.temporal_bias['weekly'][dow]
        
        # Apply hourly correction
        if 'hourly' in self.temporal_bias:
            for i, ts in enumerate(timestamps):
                hour = ts.hour
                if hour in self.temporal_bias['hourly']:
                    corrected[i] += self.temporal_bias['hourly'][hour]
        
        return corrected
    
    def _test_seasonal_significance(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> bool:
        """Test if seasonal bias is statistically significant."""
        groups = [residuals[timestamps.month == m] for m in range(1, 13) 
                  if (timestamps.month == m).sum() >= self.min_samples_per_group]
        
        if len(groups) < 2:
            return False
        
        # ANOVA-like test
        f_stat, p_value = stats.f_oneway(*groups)
        return p_value < self.significance_level
    
    def _test_weekly_significance(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> bool:
        """Test if weekly bias is statistically significant."""
        groups = [residuals[timestamps.dayofweek == d] for d in range(7)
                  if (timestamps.dayofweek == d).sum() >= self.min_samples_per_group]
        
        if len(groups) < 2:
            return False
        
        f_stat, p_value = stats.f_oneway(*groups)
        return p_value < self.significance_level
    
    def _test_hourly_significance(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> bool:
        """Test if hourly bias is statistically significant."""
        groups = [residuals[timestamps.hour == h] for h in range(24)
                  if (timestamps.hour == h).sum() >= self.min_samples_per_group]
        
        if len(groups) < 2:
            return False
        
        f_stat, p_value = stats.f_oneway(*groups)
        return p_value < self.significance_level
    
    def _calculate_bias_reduction(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: pd.DatetimeIndex,
        context_features: Optional[np.ndarray]
    ):
        """Calculate bias reduction achieved."""
        original_bias = np.abs(np.mean(targets - predictions))
        
        corrected_predictions = self.correct_bias(
            predictions, timestamps, context_features
        )
        corrected_bias = np.abs(np.mean(targets - corrected_predictions))
        
        if original_bias > 0:
            self.bias_reduction = (original_bias - corrected_bias) / original_bias
        else:
            self.bias_reduction = 0.0
```

---

## 🧪 Testing & Validation

```python
"""Tests for advanced bias correction."""
import pytest
import pandas as pd
import numpy as np

from src.models.combination.advanced_bias_correction import (
    AdvancedBiasCorrectionModule, SplineBiasCorrector
)


@pytest.fixture
def sample_bias_data():
    """Create sample data with systematic bias."""
    dates = pd.date_range('2024-01-01', periods=1000, freq='30min')
    
    # Non-linear bias pattern
    true_values = np.random.randn(1000) * 50 + 1000
    predictions = true_values + 50 * np.sin(true_values / 200)  # Non-linear bias
    
    return predictions, true_values, dates


def test_spline_correction(sample_bias_data):
    """Test spline-based bias correction."""
    predictions, targets, timestamps = sample_bias_data
    
    corrector = SplineBiasCorrector(n_knots=5, degree=3)
    residuals = targets - predictions
    
    corrector.fit(predictions, residuals)
    corrected = corrector.correct(predictions)
    
    # Should reduce bias
    original_bias = np.abs(np.mean(residuals))
    corrected_bias = np.abs(np.mean(targets - corrected))
    
    assert corrected_bias < original_bias


def test_temporal_decomposition():
    """Test temporal bias decomposition."""
    dates = pd.date_range('2024-01-01', periods=2000, freq='30min')
    
    # Hourly bias pattern
    true_values = np.random.randn(2000) * 30 + 1000
    hourly_bias = np.array([20 * np.sin(h * np.pi / 12) for h in dates.hour])
    predictions = true_values + hourly_bias
    
    bias_corrector = AdvancedBiasCorrectionModule(
        config={'correction_type': 'temporal'}
    )
    
    residuals = true_values - predictions
    bias_corrector.fit(predictions, true_values, dates)
    
    assert 'hourly' in bias_corrector.temporal_bias


def test_combined_correction(sample_bias_data):
    """Test combined bias correction."""
    predictions, targets, timestamps = sample_bias_data
    
    bias_corrector = AdvancedBiasCorrectionModule(
        config={'correction_type': 'combined'}
    )
    
    bias_corrector.fit(predictions, targets, timestamps)
    corrected = bias_corrector.correct_bias(predictions, timestamps)
    
    # Should show significant bias reduction
    assert bias_corrector.bias_reduction > 0.2
```

---

## 📝 Technical Notes

### Non-Linear Bias
- Spline regression captures curves
- Prevents overfitting with regularization

### Temporal Decomposition
- Seasonal: Monthly patterns
- Weekly: Day-of-week patterns
- Hourly: Intraday patterns

---

## 🔗 Dependencies

**Depends On:**
- PC-036-05A: Base Combiner Interface
- PC-040-05A: Basic Bias Correction

**Blocks:**
- PC-046-05B: Ensemble Diversity Analyzer
- Epic-06A: Hierarchical reconciliation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Non-linear bias detection working
- [ ] Spline correction functional
- [ ] Tree-based conditional correction working
- [ ] Temporal decomposition implemented
- [ ] Significance testing preventing overfitting
- [ ] >30% bias reduction on validation data
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-05B: Advanced Ensemble Methods](../epics/Epic-05B.md)  
**Previous:** [PC-044-05B: Context-Aware Combiner](PC-044-05B-context-aware-combiner.md)  
**Next:** [PC-046-05B: Ensemble Diversity Metrics](PC-046-05B-ensemble-diversity-analyzer.md)
