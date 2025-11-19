# PC-055-06B: WLS Reconciler Implementation

**Ticket ID:** PC-055-06B  
**Epic:** [Epic-06B: Advanced Reconciliation Methods](../epics/Epic-06B.md)  
**User Story:** US-2  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `WLSReconciler` class that performs Weighted Least Squares reconciliation using forecast variance for optimal weighting. Dynamically calculates weights based on forecast uncertainty estimates (prediction intervals, ensemble variance, historical residuals), supports multiple weighting schemes, handles missing uncertainty with intelligent fallbacks, and gives higher weight to more reliable forecasts in the reconciliation process.

**As a** ML engineer  
**I want** a WLS reconciler that leverages forecast variance for optimal weighting  
**So that** more reliable forecasts receive appropriate weight in reconciliation

---

## ✅ Acceptance Criteria

- [ ] `WLSReconciler` implements weighted least squares reconciliation
- [ ] Dynamic weight calculation from forecast variances
- [ ] Multiple weighting schemes (inverse variance, prediction intervals, confidence)
- [ ] Fallback mechanisms for missing uncertainty estimates
- [ ] Weight validation and normalization
- [ ] Performance improvement over equal-weight methods
- [ ] Integration with model uncertainty estimates
- [ ] Efficient weight matrix operations
- [ ] Performance <5s for 26 series
- [ ] Comprehensive weight diagnostics

---

## 🔧 Implementation Tasks

### 1. Create WLS Reconciliation Module
- [ ] Create `src/models/reconciliation/wls_reconciler.py`
- [ ] Import scipy for linear algebra
- [ ] Import numpy for array operations
- [ ] Import BaseReconciler interface
- [ ] Add module docstrings

### 2. Implement WeightingScheme Enum
- [ ] Create `WeightingScheme` enum
- [ ] INVERSE_VARIANCE: w = 1/σ²
- [ ] PREDICTION_INTERVAL: weights from PI width
- [ ] MODEL_CONFIDENCE: weights from model scores
- [ ] ENSEMBLE_VARIANCE: weights from ensemble spread
- [ ] EQUAL: fallback equal weighting

### 3. Implement VarianceWeightCalculator Class
- [ ] Create `VarianceWeightCalculator` class
- [ ] Initialize with weighting scheme
- [ ] Add weight validation parameters
- [ ] Add normalization options
- [ ] Add fallback configuration

### 4. Implement Inverse Variance Weighting
- [ ] Create `_calculate_inverse_variance_weights()` method
- [ ] Compute: w_i = 1/σ_i²
- [ ] Handle zero variance (add small epsilon)
- [ ] Normalize weights if needed
- [ ] Validate weight magnitudes
- [ ] Return weight vector

### 5. Implement Prediction Interval Weighting
- [ ] Create `_calculate_pi_weights()` method
- [ ] Extract PI widths from forecast data
- [ ] Wider intervals → lower weights
- [ ] w_i = 1/(PI_width_i)²
- [ ] Handle missing PIs
- [ ] Return weight vector

### 6. Implement Ensemble Variance Weighting
- [ ] Create `_calculate_ensemble_weights()` method
- [ ] Calculate variance across ensemble members
- [ ] w_i = 1/var(ensemble_i)
- [ ] Higher agreement → higher weight
- [ ] Handle single-model forecasts
- [ ] Return weight vector

### 7. Implement Model Confidence Weighting
- [ ] Create `_calculate_confidence_weights()` method
- [ ] Use model confidence scores
- [ ] Confidence from historical performance
- [ ] Normalize to valid weight range
- [ ] Combine multiple confidence signals
- [ ] Return weight vector

### 8. Implement WLS Reconciliation Core
- [ ] Create `reconcile()` method
- [ ] Validate inputs (forecasts, hierarchy, variances)
- [ ] Calculate weights from variances
- [ ] Construct weight matrix W
- [ ] Extract aggregation matrix S
- [ ] Compute WLS matrix: G = S(S'W^(-1)S)^(-1)S'W^(-1)
- [ ] Apply: y_reconciled = G * y_base
- [ ] Return ReconciliationResult

### 9. Implement Weight Matrix Construction
- [ ] Create `_build_weight_matrix()` method
- [ ] Convert weight vector to diagonal matrix
- [ ] Validate positive definiteness
- [ ] Handle zero weights (epsilon floor)
- [ ] Check matrix dimensions
- [ ] Return weight matrix W

### 10. Implement Weight Validation
- [ ] Create `_validate_weights()` method
- [ ] Check all weights positive
- [ ] Check weight magnitudes reasonable
- [ ] Warn if extreme weight ratios
- [ ] Check for numerical issues
- [ ] Return validation status

### 11. Implement Fallback Weight Mechanisms
- [ ] Create `_apply_fallback_weights()` method
- [ ] Use equal weights when variance missing
- [ ] Use historical average variance
- [ ] Interpolate from available weights
- [ ] Log fallback applications
- [ ] Return complete weight vector

### 12. Implement Weight Normalization
- [ ] Create `_normalize_weights()` method
- [ ] Optional normalization to sum=1
- [ ] Optional scaling to [0, 1]
- [ ] Preserve relative magnitudes
- [ ] Handle edge cases
- [ ] Return normalized weights

### 13. Implement Variance Estimation from Residuals
- [ ] Create `_estimate_variance_from_residuals()` method
- [ ] Calculate rolling variance from history
- [ ] Window size configuration
- [ ] Handle limited history
- [ ] Return variance estimates

### 14. Implement WLS Matrix Computation
- [ ] Create `_compute_wls_matrix()` method
- [ ] Compute S'W^(-1)S
- [ ] Invert using stable method
- [ ] Compute full reconciliation matrix G
- [ ] Handle ill-conditioning
- [ ] Return WLS matrix

### 15. Implement Weight Diagnostics
- [ ] Create `get_weight_diagnostics()` method
- [ ] Report weight distribution
- [ ] Calculate weight ratios (max/min)
- [ ] Identify heavily weighted series
- [ ] Identify down-weighted series
- [ ] Return diagnostic report

### 16. Handle Edge Cases
- [ ] Handle all-zero variances
- [ ] Handle infinite variances
- [ ] Handle missing variance estimates
- [ ] Handle single-series scenarios
- [ ] Graceful degradation to equal weights

### 17. Write Comprehensive Tests
- [ ] Create `tests/models/reconciliation/test_wls_reconciler.py`
- [ ] Test each weighting scheme
- [ ] Test weight calculation accuracy
- [ ] Test fallback mechanisms
- [ ] Test weight validation
- [ ] Test integration with variances
- [ ] Test edge cases

### 18. Create Weight Analysis Tools
- [ ] Create `tools/analyze_weights.py`
- [ ] Visualize weight distributions
- [ ] Compare weighting schemes
- [ ] Analyze weight impact on reconciliation
- [ ] Document weight selection guidance

---

## 💻 Implementation Details

### WLSReconciler Implementation

```python
"""WLS reconciliation with variance-based weighting."""
from typing import Dict, Any, Optional, List
from enum import Enum
import numpy as np
from scipy import linalg
import time

from src.models.reconciliation.base_reconciler import (
    BaseReconciler, ReconciliationResult, ReconciliationError
)
from src.models.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WeightingScheme(Enum):
    """Weighting schemes for WLS reconciliation."""
    INVERSE_VARIANCE = "inverse_variance"
    PREDICTION_INTERVAL = "prediction_interval"
    MODEL_CONFIDENCE = "model_confidence"
    ENSEMBLE_VARIANCE = "ensemble_variance"
    EQUAL = "equal"


class VarianceWeightCalculator:
    """
    Calculates weights from forecast variance estimates.
    
    Weighting Principles:
        - Lower variance → Higher weight
        - More reliable forecasts influence reconciliation more
        - Weights inversely proportional to variance
    
    Schemes:
        - Inverse Variance: w = 1/σ²
        - Prediction Interval: w = 1/(PI_width)²
        - Ensemble Variance: w = 1/var(ensemble)
        - Model Confidence: w = confidence_score
    """
    
    def __init__(
        self,
        scheme: WeightingScheme = WeightingScheme.INVERSE_VARIANCE,
        min_weight: float = 1e-6,
        max_weight: float = 1e6,
        normalize: bool = False
    ):
        """
        Initialize weight calculator.
        
        Args:
            scheme: Weighting scheme to use
            min_weight: Minimum weight (prevents zeros)
            max_weight: Maximum weight (prevents extreme values)
            normalize: Whether to normalize weights
        """
        self.scheme = scheme
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.normalize = normalize
    
    def calculate_weights(
        self,
        variances: Optional[Dict[str, np.ndarray]] = None,
        prediction_intervals: Optional[Dict[str, tuple]] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        ensemble_forecasts: Optional[Dict[str, List[np.ndarray]]] = None,
        node_order: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Calculate weights based on selected scheme.
        
        Args:
            variances: Forecast variances for each series
            prediction_intervals: Prediction intervals (lower, upper)
            confidence_scores: Model confidence scores
            ensemble_forecasts: Ensemble member forecasts
            node_order: Ordered list of node names
        
        Returns:
            Weight vector
        """
        if self.scheme == WeightingScheme.INVERSE_VARIANCE:
            if variances is None:
                raise ValueError("Variances required for inverse variance weighting")
            weights = self._inverse_variance_weights(variances, node_order)
        
        elif self.scheme == WeightingScheme.PREDICTION_INTERVAL:
            if prediction_intervals is None:
                raise ValueError("Prediction intervals required")
            weights = self._pi_weights(prediction_intervals, node_order)
        
        elif self.scheme == WeightingScheme.ENSEMBLE_VARIANCE:
            if ensemble_forecasts is None:
                raise ValueError("Ensemble forecasts required")
            weights = self._ensemble_weights(ensemble_forecasts, node_order)
        
        elif self.scheme == WeightingScheme.MODEL_CONFIDENCE:
            if confidence_scores is None:
                raise ValueError("Confidence scores required")
            weights = self._confidence_weights(confidence_scores, node_order)
        
        elif self.scheme == WeightingScheme.EQUAL:
            n = len(node_order) if node_order else 1
            weights = np.ones(n)
        
        else:
            raise ValueError(f"Unknown weighting scheme: {self.scheme}")
        
        # Validate and clip weights
        weights = self._validate_and_clip_weights(weights)
        
        # Normalize if requested
        if self.normalize:
            weights = weights / np.sum(weights)
        
        return weights
    
    def _inverse_variance_weights(
        self,
        variances: Dict[str, np.ndarray],
        node_order: List[str]
    ) -> np.ndarray:
        """Calculate inverse variance weights: w = 1/σ²."""
        weights = []
        
        for node_name in node_order:
            if node_name in variances:
                var = np.mean(variances[node_name])  # Average variance over time
                # Avoid division by zero
                weight = 1.0 / (var + 1e-10)
            else:
                # Fallback to unit weight
                weight = 1.0
                logger.warning(f"Missing variance for {node_name}, using unit weight")
            
            weights.append(weight)
        
        return np.array(weights)
    
    def _pi_weights(
        self,
        prediction_intervals: Dict[str, tuple],
        node_order: List[str]
    ) -> np.ndarray:
        """Calculate weights from prediction interval widths."""
        weights = []
        
        for node_name in node_order:
            if node_name in prediction_intervals:
                lower, upper = prediction_intervals[node_name]
                pi_width = np.mean(upper - lower)
                # Wider interval → lower weight
                weight = 1.0 / (pi_width ** 2 + 1e-10)
            else:
                weight = 1.0
                logger.warning(f"Missing PI for {node_name}")
            
            weights.append(weight)
        
        return np.array(weights)
    
    def _ensemble_weights(
        self,
        ensemble_forecasts: Dict[str, List[np.ndarray]],
        node_order: List[str]
    ) -> np.ndarray:
        """Calculate weights from ensemble variance."""
        weights = []
        
        for node_name in node_order:
            if node_name in ensemble_forecasts:
                ensemble = np.array(ensemble_forecasts[node_name])
                var = np.var(ensemble, axis=0).mean()  # Average variance across time
                weight = 1.0 / (var + 1e-10)
            else:
                weight = 1.0
                logger.warning(f"Missing ensemble for {node_name}")
            
            weights.append(weight)
        
        return np.array(weights)
    
    def _confidence_weights(
        self,
        confidence_scores: Dict[str, float],
        node_order: List[str]
    ) -> np.ndarray:
        """Calculate weights from model confidence scores."""
        weights = []
        
        for node_name in node_order:
            if node_name in confidence_scores:
                # Confidence score should be in [0, 1]
                confidence = confidence_scores[node_name]
                # Higher confidence → higher weight
                weight = confidence ** 2  # Squaring emphasizes differences
            else:
                weight = 0.5  # Neutral weight
                logger.warning(f"Missing confidence for {node_name}")
            
            weights.append(weight)
        
        return np.array(weights)
    
    def _validate_and_clip_weights(self, weights: np.ndarray) -> np.ndarray:
        """Validate and clip weights to acceptable range."""
        # Check for NaN or Inf
        if np.any(~np.isfinite(weights)):
            logger.warning("Non-finite weights detected, replacing with unit weights")
            weights = np.where(np.isfinite(weights), weights, 1.0)
        
        # Clip to valid range
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        # Check weight ratio
        if np.max(weights) > 0:
            weight_ratio = np.max(weights) / (np.min(weights) + 1e-10)
            if weight_ratio > 1e6:
                logger.warning(f"Extreme weight ratio: {weight_ratio:.2e}")
        
        return weights


class WLSReconciler(BaseReconciler):
    """
    Weighted Least Squares (WLS) hierarchical reconciliation.
    
    WLS reconciliation uses forecast variance to optimally weight
    different forecasts in the reconciliation process. More reliable
    forecasts (lower variance) receive higher weight.
    
    Algorithm:
        1. Calculate weights from forecast variances: w_i = 1/σ_i²
        2. Construct weight matrix W = diag(w)
        3. Compute reconciliation matrix: G = S(S'W^(-1)S)^(-1)S'W^(-1)
        4. Apply reconciliation: y_reconciled = G * y_base
    
    Weighting Schemes:
        - inverse_variance: w = 1/σ² (optimal for known variances)
        - prediction_interval: w = 1/(PI_width)²
        - ensemble_variance: w = 1/var(ensemble)
        - model_confidence: w = confidence_score
        - equal: uniform weights (fallback)
    
    Advantages:
        - Adapts to forecast quality differences
        - Gives appropriate influence to reliable forecasts
        - Can leverage model uncertainty estimates
        - More efficient than equal weighting
    
    Example:
        >>> # Calculate variances from historical residuals
        >>> variances = {
        ...     'National': np.array([100, 110, 120]),
        ...     'SE': np.array([50, 55, 60]),
        ...     'S': np.array([30, 33, 36])
        ... }
        >>> 
        >>> reconciler = WLSReconciler(config={
        ...     'weighting_scheme': 'inverse_variance',
        ...     'normalize_weights': True
        ... })
        >>> 
        >>> result = reconciler.reconcile(
        ...     base_forecasts=forecasts,
        ...     hierarchy=hierarchy,
        ...     forecast_variances=variances
        ... )
        >>> 
        >>> # Check weight diagnostics
        >>> diagnostics = reconciler.get_weight_diagnostics()
        >>> print(f"Weight ratio: {diagnostics['weight_ratio']:.2f}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize WLS reconciler.
        
        Args:
            config: Configuration dictionary with optional keys:
                - weighting_scheme: 'inverse_variance', 'prediction_interval', etc.
                - normalize_weights: Whether to normalize weights
                - min_weight: Minimum weight value
                - max_weight: Maximum weight value
                - fallback_equal_weights: Use equal weights if variances unavailable
        """
        super().__init__(config)
        
        scheme_str = self.config.get('weighting_scheme', 'inverse_variance')
        self.weighting_scheme = WeightingScheme(scheme_str)
        self.normalize_weights = self.config.get('normalize_weights', False)
        self.fallback_equal = self.config.get('fallback_equal_weights', True)
        
        self.weight_calculator = VarianceWeightCalculator(
            scheme=self.weighting_scheme,
            min_weight=self.config.get('min_weight', 1e-6),
            max_weight=self.config.get('max_weight', 1e6),
            normalize=self.normalize_weights
        )
        
        self._computed_weights: Optional[np.ndarray] = None
    
    def reconcile(
        self,
        base_forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition,
        forecast_variances: Optional[Dict[str, np.ndarray]] = None,
        prediction_intervals: Optional[Dict[str, tuple]] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        ensemble_forecasts: Optional[Dict[str, List[np.ndarray]]] = None
    ) -> ReconciliationResult:
        """
        Perform WLS reconciliation.
        
        Args:
            base_forecasts: Dictionary of base forecasts
            hierarchy: Hierarchy definition
            forecast_variances: Forecast variances for weighting
            prediction_intervals: Prediction intervals (lower, upper)
            confidence_scores: Model confidence scores
            ensemble_forecasts: Ensemble member forecasts
        
        Returns:
            ReconciliationResult with reconciled forecasts
        
        Raises:
            ReconciliationError: If reconciliation fails
        """
        start_time = time.time()
        logger.info(f"Starting WLS reconciliation (scheme: {self.weighting_scheme.value})")
        
        # Validate inputs
        errors = self._validate_forecasts(base_forecasts, hierarchy)
        if errors:
            raise ReconciliationError(f"Validation errors: {errors}")
        
        # Get aggregation matrix S
        if hierarchy.aggregation_matrix is None:
            hierarchy.build_aggregation_matrix()
        S = hierarchy.aggregation_matrix
        
        # Prepare forecast vector
        y_base, node_order = self._prepare_forecast_array(base_forecasts, hierarchy)
        
        # Calculate weights
        try:
            weights = self.weight_calculator.calculate_weights(
                variances=forecast_variances,
                prediction_intervals=prediction_intervals,
                confidence_scores=confidence_scores,
                ensemble_forecasts=ensemble_forecasts,
                node_order=node_order
            )
            self._computed_weights = weights
        
        except (ValueError, KeyError) as e:
            if self.fallback_equal:
                logger.warning(f"Weight calculation failed ({e}), using equal weights")
                weights = np.ones(len(node_order))
                self._computed_weights = weights
            else:
                raise ReconciliationError(f"Weight calculation failed: {e}")
        
        logger.info(f"Weight range: [{weights.min():.2e}, {weights.max():.2e}]")
        
        # Construct weight matrix W
        W = np.diag(weights)
        
        # Compute WLS reconciliation matrix
        G = self._compute_wls_matrix(S, W)
        
        # Apply reconciliation
        y_reconciled = G @ y_base
        
        # Reconstruct forecast dictionary
        reconciled_forecasts = self._reconstruct_forecasts(
            y_reconciled, node_order, base_forecasts
        )
        
        # Validate constraints
        violations = self._validate_constraints(reconciled_forecasts, hierarchy)
        
        # Create metadata
        processing_time = time.time() - start_time
        metadata = self._create_metadata(
            method_name='WLS',
            hierarchy=hierarchy,
            processing_time=processing_time,
            weighting_scheme=self.weighting_scheme.value
        )
        
        # Add weight diagnostics
        metadata.performance_metrics['weight_ratio'] = float(weights.max() / (weights.min() + 1e-10))
        metadata.performance_metrics['mean_weight'] = float(weights.mean())
        
        logger.info(f"WLS reconciliation complete in {processing_time:.2f}s")
        
        return ReconciliationResult(
            reconciled_forecasts=reconciled_forecasts,
            original_forecasts=base_forecasts,
            metadata=metadata,
            constraint_violations=violations
        )
    
    def _compute_wls_matrix(self, S: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute WLS reconciliation matrix: G = S(S'W^(-1)S)^(-1)S'W^(-1).
        
        Args:
            S: Aggregation matrix
            W: Weight matrix (diagonal)
        
        Returns:
            WLS reconciliation matrix G
        """
        # Invert weight matrix (easy for diagonal)
        W_inv = np.diag(1.0 / np.diag(W))
        
        # Compute S'W^(-1)S
        SWS = S.T @ W_inv @ S
        
        # Check condition number
        cond_num = np.linalg.cond(SWS)
        logger.info(f"Condition number of S'W^(-1)S: {cond_num:.2e}")
        
        # Invert S'W^(-1)S
        try:
            L = linalg.cholesky(SWS, lower=True)
            SWS_inv = linalg.cho_solve((L, True), np.eye(SWS.shape[0]))
        except linalg.LinAlgError:
            logger.warning("Cholesky failed, using pseudo-inverse")
            SWS_inv = linalg.pinv(SWS)
        
        # Compute full reconciliation matrix
        G = S @ SWS_inv @ S.T @ W_inv
        
        return G
    
    def get_weight_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about computed weights.
        
        Returns:
            Dictionary with weight diagnostics
        """
        if self._computed_weights is None:
            return {'error': 'No weights computed yet'}
        
        weights = self._computed_weights
        
        diagnostics = {
            'weighting_scheme': self.weighting_scheme.value,
            'n_series': len(weights),
            'min_weight': float(weights.min()),
            'max_weight': float(weights.max()),
            'mean_weight': float(weights.mean()),
            'std_weight': float(weights.std()),
            'weight_ratio': float(weights.max() / (weights.min() + 1e-10)),
            'normalized': self.normalize_weights
        }
        
        # Identify extreme weights
        median_weight = np.median(weights)
        heavily_weighted = np.where(weights > 10 * median_weight)[0]
        down_weighted = np.where(weights < 0.1 * median_weight)[0]
        
        diagnostics['heavily_weighted_count'] = len(heavily_weighted)
        diagnostics['down_weighted_count'] = len(down_weighted)
        
        return diagnostics
    
    def _prepare_forecast_array(self, forecasts, hierarchy):
        """Prepare forecast array (same as other reconcilers)."""
        node_order = sorted(hierarchy.nodes.keys(),
                          key=lambda x: (hierarchy.nodes[x].level, x))
        
        forecast_arrays = []
        for node_name in node_order:
            if node_name in forecasts:
                forecast_arrays.append(forecasts[node_name])
            else:
                first_shape = next(iter(forecasts.values())).shape
                forecast_arrays.append(np.zeros(first_shape))
        
        forecast_array = np.concatenate(forecast_arrays)
        return forecast_array, node_order
    
    def _reconstruct_forecasts(self, reconciled_array, node_order, original_forecasts):
        """Reconstruct forecast dictionary (same as other reconcilers)."""
        first_forecast = next(iter(original_forecasts.values()))
        n_timesteps = len(first_forecast)
        
        reconciled_forecasts = {}
        offset = 0
        
        for node_name in node_order:
            reconciled_forecasts[node_name] = reconciled_array[offset:offset+n_timesteps]
            offset += n_timesteps
        
        return reconciled_forecasts
```

---

## 🧪 Testing & Validation

```python
"""Tests for WLS reconciler."""
import pytest
import numpy as np

from src.models.reconciliation.wls_reconciler import (
    WLSReconciler, VarianceWeightCalculator, WeightingScheme
)


def test_inverse_variance_weights():
    """Test inverse variance weight calculation."""
    variances = {
        'Series_1': np.array([100, 100, 100]),  # High variance → low weight
        'Series_2': np.array([25, 25, 25]),      # Low variance → high weight
        'Series_3': np.array([50, 50, 50])       # Medium variance → medium weight
    }
    
    calculator = VarianceWeightCalculator(scheme=WeightingScheme.INVERSE_VARIANCE)
    weights = calculator.calculate_weights(
        variances=variances,
        node_order=['Series_1', 'Series_2', 'Series_3']
    )
    
    # Series_2 should have highest weight (lowest variance)
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]


def test_wls_reconciliation(sample_hierarchy_yaml):
    """Test WLS reconciliation."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    forecasts = {
        'BR_National': np.array([1000, 1100]),
        'SE_Subsystem': np.array([600, 650]),
        'S_Subsystem': np.array([400, 450])
    }
    
    variances = {
        'BR_National': np.array([100, 100]),
        'SE_Subsystem': np.array([50, 50]),
        'S_Subsystem': np.array([30, 30])
    }
    
    reconciler = WLSReconciler(config={'weighting_scheme': 'inverse_variance'})
    result = reconciler.reconcile(forecasts, hierarchy, forecast_variances=variances)
    
    assert len(result.reconciled_forecasts) >= 3
    assert result.metadata.reconciliation_method == 'WLS'


def test_fallback_equal_weights(sample_hierarchy_yaml):
    """Test fallback to equal weights when variances missing."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    forecasts = {
        'BR_National': np.array([1000]),
        'SE_Subsystem': np.array([600]),
        'S_Subsystem': np.array([400])
    }
    
    reconciler = WLSReconciler(config={
        'weighting_scheme': 'inverse_variance',
        'fallback_equal_weights': True
    })
    
    # No variances provided, should use equal weights
    result = reconciler.reconcile(forecasts, hierarchy)
    
    assert result.metadata.reconciliation_method == 'WLS'


def test_weight_diagnostics():
    """Test weight diagnostics."""
    reconciler = WLSReconciler()
    
    # Mock computed weights
    reconciler._computed_weights = np.array([1.0, 10.0, 0.1, 5.0])
    
    diagnostics = reconciler.get_weight_diagnostics()
    
    assert diagnostics['n_series'] == 4
    assert diagnostics['weight_ratio'] == 100  # 10.0 / 0.1
    assert diagnostics['min_weight'] == 0.1
    assert diagnostics['max_weight'] == 10.0
```

---

## 📝 Technical Notes

### WLS Formula
- Reconciliation matrix: G = S(S'W^(-1)S)^(-1)S'W^(-1)
- Weight matrix W = diag(w) where w_i = 1/σ_i²
- More reliable forecasts get higher influence

### Weight Selection
- Inverse variance optimal when variances known
- Prediction intervals provide uncertainty estimates
- Ensemble variance shows model agreement

---

## 🔗 Dependencies

**Depends On:**
- PC-048-06A: Hierarchy Definition System
- PC-049-06A: Base Reconciler Interface
- PC-054-06B: OLS Reconciler (for comparison)

**Blocks:**
- PC-056-06B: Shrinkage Reconciler
- PC-057-06B: Reconciler Selection

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] WLS algorithm implemented correctly
- [ ] Multiple weighting schemes working
- [ ] Dynamic weight calculation functional
- [ ] Fallback mechanisms tested
- [ ] Weight validation comprehensive
- [ ] Performance improvement demonstrated
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Weight diagnostics operational
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-06B: Advanced Reconciliation Methods](../epics/Epic-06B.md)  
**Previous:** [PC-054-06B: OLS Reconciler Implementation](PC-054-06B-ols-reconciler.md)  
**Next:** [PC-056-06B: Shrinkage Reconciler Implementation](PC-056-06B-shrinkage-reconciler.md)
