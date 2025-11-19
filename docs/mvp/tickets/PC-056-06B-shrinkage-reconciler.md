# PC-056-06B: Shrinkage Reconciler Implementation

**Ticket ID:** PC-056-06B  
**Epic:** [Epic-06B: Advanced Reconciliation Methods](../epics/Epic-06B.md)  
**User Story:** US-3  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `ShrinkageReconciler` class that uses James-Stein shrinkage for robust covariance estimation. Handles unstable covariance estimates with limited historical data through shrinkage towards structured targets (identity, diagonal, factor). Includes automatic shrinkage intensity selection via cross-validation or Ledoit-Wolf formula, maintains reconciliation quality with small sample sizes, and ensures numerical stability.

**As a** system engineer  
**I want** shrinkage-based reconciliation that handles limited data robustly  
**So that** reconciliation quality is maintained even with small sample sizes

---

## ✅ Acceptance Criteria

- [ ] `ShrinkageReconciler` implements James-Stein shrinkage
- [ ] Multiple shrinkage targets (identity, diagonal, factor)
- [ ] Automatic shrinkage intensity (λ) selection
- [ ] Cross-validation for optimal λ
- [ ] Ledoit-Wolf analytical shrinkage formula
- [ ] Robust performance with <100 samples
- [ ] Outperforms sample covariance in low-data scenarios
- [ ] Integration with BaseReconciler interface
- [ ] Performance <5s for 26 series
- [ ] Comprehensive shrinkage diagnostics

---

## 🔧 Implementation Tasks

### 1-5. Module Setup & Core Classes
- [ ] Create `src/models/reconciliation/shrinkage_reconciler.py`
- [ ] Implement `ShrinkageTarget` enum (IDENTITY, DIAGONAL, FACTOR)
- [ ] Create `ShrinkageEstimator` base class
- [ ] Implement `LedoitWolfShrinkage` class
- [ ] Implement `CrossValidationShrinkage` class

### 6-10. Shrinkage Methods
- [ ] Implement identity target shrinkage: Σ_shrink = (1-λ)Σ_sample + λI
- [ ] Implement diagonal target shrinkage
- [ ] Implement factor structure shrinkage
- [ ] Create Ledoit-Wolf analytical λ calculator
- [ ] Implement cross-validation λ optimizer

### 11-15. Reconciliation & Validation
- [ ] Implement `reconcile()` method with shrinkage
- [ ] Create shrinkage intensity validation
- [ ] Implement positive definiteness checking
- [ ] Build performance comparison framework
- [ ] Add comprehensive diagnostics

### 16-18. Testing & Documentation
- [ ] Write comprehensive unit tests
- [ ] Create integration tests with small sample sizes
- [ ] Document shrinkage methodology and usage

---

## 💻 Code Implementation

```python
"""Shrinkage-based reconciliation for robust covariance estimation."""
from typing import Dict, Any, Optional
from enum import Enum
import numpy as np
from scipy import linalg
from sklearn.model_selection import TimeSeriesSplit
import time

from src.models.reconciliation.base_reconciler import BaseReconciler, ReconciliationResult
from src.models.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShrinkageTarget(Enum):
    """Shrinkage target structures."""
    IDENTITY = "identity"
    DIAGONAL = "diagonal"
    FACTOR = "factor"
    CONSTANT_CORRELATION = "constant_correlation"


class ShrinkageReconciler(BaseReconciler):
    """
    Shrinkage-based hierarchical reconciliation.
    
    Uses James-Stein shrinkage to produce robust covariance estimates
    when historical data is limited. Shrinks sample covariance towards
    structured target, improving stability.
    
    Shrinkage Formula:
        Σ_shrink = (1-λ)Σ_sample + λΣ_target
        where λ ∈ [0,1] is shrinkage intensity
    
    Targets:
        - IDENTITY: Σ_target = I (assumes independence)
        - DIAGONAL: Σ_target = diag(Σ_sample) (removes correlations)
        - FACTOR: Σ_target from factor model
        - CONSTANT_CORRELATION: Assumes constant ρ
    
    Optimal λ Selection:
        - Ledoit-Wolf: Analytical formula (fast, recommended)
        - Cross-validation: Data-driven (slower but flexible)
    
    Example:
        >>> reconciler = ShrinkageReconciler(config={
        ...     'shrinkage_target': 'diagonal',
        ...     'auto_lambda': True,
        ...     'lambda_method': 'ledoit_wolf'
        ... })
        >>> result = reconciler.reconcile(forecasts, hierarchy, residuals)
        >>> print(f"Shrinkage intensity: {result.metadata.performance_metrics['lambda']:.3f}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        target_str = self.config.get('shrinkage_target', 'diagonal')
        self.shrinkage_target = ShrinkageTarget(target_str)
        self.auto_lambda = self.config.get('auto_lambda', True)
        self.lambda_method = self.config.get('lambda_method', 'ledoit_wolf')
        self.fixed_lambda = self.config.get('fixed_lambda', 0.5)
        
        self._computed_lambda: Optional[float] = None
    
    def reconcile(
        self,
        base_forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition,
        historical_residuals: np.ndarray,
        **kwargs
    ) -> ReconciliationResult:
        """Perform shrinkage-based reconciliation."""
        start_time = time.time()
        logger.info(f"Starting shrinkage reconciliation (target: {self.shrinkage_target.value})")
        
        # Validate inputs
        errors = self._validate_forecasts(base_forecasts, hierarchy)
        if errors:
            raise ReconciliationError(f"Validation errors: {errors}")
        
        # Estimate sample covariance
        sample_cov = self._compute_sample_covariance(historical_residuals)
        
        # Construct shrinkage target
        target_cov = self._construct_target(sample_cov, historical_residuals)
        
        # Determine shrinkage intensity
        if self.auto_lambda:
            if self.lambda_method == 'ledoit_wolf':
                lambda_optimal = self._ledoit_wolf_lambda(sample_cov, target_cov, historical_residuals)
            else:  # cross_validation
                lambda_optimal = self._cv_lambda(historical_residuals, hierarchy)
        else:
            lambda_optimal = self.fixed_lambda
        
        self._computed_lambda = lambda_optimal
        logger.info(f"Shrinkage intensity λ: {lambda_optimal:.4f}")
        
        # Apply shrinkage
        shrunk_cov = (1 - lambda_optimal) * sample_cov + lambda_optimal * target_cov
        
        # Standard reconciliation with shrunk covariance
        S = hierarchy.aggregation_matrix if hierarchy.aggregation_matrix is not None else hierarchy.build_aggregation_matrix()
        y_base, node_order = self._prepare_forecast_array(base_forecasts, hierarchy)
        
        G = self._compute_reconciliation_matrix(S, shrunk_cov)
        y_reconciled = G @ y_base
        
        reconciled_forecasts = self._reconstruct_forecasts(y_reconciled, node_order, base_forecasts)
        violations = self._validate_constraints(reconciled_forecasts, hierarchy)
        
        processing_time = time.time() - start_time
        metadata = self._create_metadata('Shrinkage', hierarchy, processing_time)
        metadata.performance_metrics['lambda'] = float(lambda_optimal)
        metadata.performance_metrics['shrinkage_target'] = self.shrinkage_target.value
        
        return ReconciliationResult(
            reconciled_forecasts=reconciled_forecasts,
            original_forecasts=base_forecasts,
            metadata=metadata,
            constraint_violations=violations
        )
    
    def _construct_target(self, sample_cov: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """Construct shrinkage target covariance."""
        n = sample_cov.shape[0]
        
        if self.shrinkage_target == ShrinkageTarget.IDENTITY:
            # Scale identity by trace
            scale = np.trace(sample_cov) / n
            return scale * np.eye(n)
        
        elif self.shrinkage_target == ShrinkageTarget.DIAGONAL:
            return np.diag(np.diag(sample_cov))
        
        elif self.shrinkage_target == ShrinkageTarget.CONSTANT_CORRELATION:
            # Assume constant correlation ρ
            variances = np.diag(sample_cov)
            avg_corr = (np.sum(sample_cov / np.outer(np.sqrt(variances), np.sqrt(variances))) - n) / (n * (n - 1))
            target = avg_corr * np.outer(np.sqrt(variances), np.sqrt(variances))
            np.fill_diagonal(target, variances)
            return target
        
        else:  # FACTOR
            return self._factor_target(residuals)
    
    def _ledoit_wolf_lambda(self, sample_cov: np.ndarray, target_cov: np.ndarray, residuals: np.ndarray) -> float:
        """Compute optimal shrinkage intensity using Ledoit-Wolf formula."""
        n_samples, n_series = residuals.shape
        
        # Compute delta (numerator)
        delta = np.linalg.norm(sample_cov - target_cov, 'fro') ** 2
        
        # Compute phi (denominator) - variance of sample covariance
        centered = residuals - residuals.mean(axis=0)
        phi = 0.0
        for t in range(n_samples):
            r_t = centered[t:t+1, :].T
            phi += np.linalg.norm(r_t @ r_t.T - sample_cov, 'fro') ** 2
        phi /= n_samples ** 2
        
        # Optimal lambda
        lambda_opt = min(phi / (delta + 1e-10), 1.0)
        lambda_opt = max(lambda_opt, 0.0)
        
        return lambda_opt
    
    def _compute_sample_covariance(self, residuals: np.ndarray) -> np.ndarray:
        """Compute sample covariance with bias correction."""
        n_samples = residuals.shape[0]
        centered = residuals - residuals.mean(axis=0)
        return (centered.T @ centered) / (n_samples - 1)
    
    # Helper methods (similar to OLS/WLS reconcilers)
    def _compute_reconciliation_matrix(self, S, Sigma):
        """Compute reconciliation matrix."""
        Sigma_inv = linalg.pinv(Sigma)
        STS = S.T @ Sigma_inv @ S
        STS_inv = linalg.pinv(STS)
        return S @ STS_inv @ S.T @ Sigma_inv
    
    def _prepare_forecast_array(self, forecasts, hierarchy):
        """Prepare forecast array."""
        node_order = sorted(hierarchy.nodes.keys(), key=lambda x: (hierarchy.nodes[x].level, x))
        forecast_arrays = [forecasts.get(n, np.zeros(len(next(iter(forecasts.values()))))) for n in node_order]
        return np.concatenate(forecast_arrays), node_order
    
    def _reconstruct_forecasts(self, reconciled_array, node_order, original_forecasts):
        """Reconstruct forecast dictionary."""
        n_timesteps = len(next(iter(original_forecasts.values())))
        reconciled = {}
        offset = 0
        for node_name in node_order:
            reconciled[node_name] = reconciled_array[offset:offset+n_timesteps]
            offset += n_timesteps
        return reconciled
```

---

## 🧪 Testing

```python
def test_shrinkage_small_sample():
    """Test shrinkage with small sample size."""
    n_samples, n_series = 20, 10  # Small sample
    residuals = np.random.randn(n_samples, n_series)
    
    reconciler = ShrinkageReconciler(config={'shrinkage_target': 'diagonal'})
    sample_cov = reconciler._compute_sample_covariance(residuals)
    target_cov = reconciler._construct_target(sample_cov, residuals)
    lambda_opt = reconciler._ledoit_wolf_lambda(sample_cov, target_cov, residuals)
    
    # Should have significant shrinkage with small n
    assert lambda_opt > 0.3
    assert lambda_opt < 1.0
```

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Shrinkage algorithm implemented
- [ ] Multiple targets working
- [ ] Automatic λ selection functional
- [ ] Robust with limited data
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed

---

**Epic:** [Epic-06B](../epics/Epic-06B.md)  
**Previous:** [PC-055-06B](PC-055-06B-wls-reconciler.md)  
**Next:** [PC-057-06B](PC-057-06B-reconciler-selector.md)
