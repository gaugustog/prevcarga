# PC-050-06A: MinT Reconciler Implementation

**Ticket ID:** PC-050-06A  
**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**User Story:** US-3  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `MinTReconciler` class that performs Minimum Trace reconciliation using optimal weighted least squares. Includes covariance estimation (sample, structural, shrinkage), numerically stable matrix operations with regularization, efficient linear algebra for performance, and handles ill-conditioned matrices through Cholesky decomposition and pseudo-inverse fallbacks.

**As a** forecasting analyst  
**I want** a robust MinT reconciliation method implementation  
**So that** I can achieve optimal forecast reconciliation with minimum trace covariance

---

## ✅ Acceptance Criteria

- [ ] `MinTReconciler` implements minimum trace reconciliation
- [ ] Handles both structural and sampling covariance estimation
- [ ] Optimized matrix operations for computational efficiency
- [ ] Numerical stability for ill-conditioned covariance matrices
- [ ] Configurable shrinkage parameters for robustness
- [ ] MinT formula: `y_reconciled = S * (S'ΣS)^(-1) * S' * Σ * y_base`
- [ ] Multiple covariance estimation methods
- [ ] Ridge regularization support
- [ ] Performance <5s for 26 series
- [ ] Integration with BaseReconciler interface

---

## 🔧 Implementation Tasks

### 1. Create MinT Reconciler Module
- [ ] Create `src/models/reconciliation/mint_reconciler.py`
- [ ] Import scipy for linear algebra
- [ ] Import numpy for matrix operations
- [ ] Import BaseReconciler interface
- [ ] Add module docstrings

### 2. Implement MinTReconciler Class
- [ ] Inherit from BaseReconciler
- [ ] Implement `__init__()` with configuration
- [ ] Add shrinkage_param attribute
- [ ] Add covariance_method attribute
- [ ] Add regularization_param attribute

### 3. Implement Covariance Estimator Component
- [ ] Create `CovarianceEstimator` class
- [ ] Support 'sample' method
- [ ] Support 'structural' method  
- [ ] Support 'shrinkage' method
- [ ] Support 'identity' method (OLS)

### 4. Implement Sample Covariance Estimation
- [ ] Create `_estimate_sample_covariance()` method
- [ ] Calculate residuals from historical data
- [ ] Compute sample covariance: Σ = (1/n) * R' * R
- [ ] Handle small sample sizes
- [ ] Apply bias correction
- [ ] Return covariance matrix

### 5. Implement Structural Covariance Estimation
- [ ] Create `_estimate_structural_covariance()` method
- [ ] Use variance proportional to level in hierarchy
- [ ] Bottom level: σ²_i for each series
- [ ] Aggregate level: sum of children variances
- [ ] Construct diagonal covariance matrix
- [ ] Return covariance matrix

### 6. Implement Shrinkage Covariance Estimation
- [ ] Create `_estimate_shrinkage_covariance()` method
- [ ] Combine sample and structural estimates
- [ ] Σ_shrink = λ * Σ_structural + (1-λ) * Σ_sample
- [ ] Optimize λ using cross-validation or fixed
- [ ] Ensure positive definiteness
- [ ] Return shrinkage covariance

### 7. Implement MinT Reconciliation Core
- [ ] Create `reconcile()` method
- [ ] Validate inputs (forecasts, hierarchy)
- [ ] Extract aggregation matrix S from hierarchy
- [ ] Estimate or use provided covariance matrix Σ
- [ ] Compute reconciliation matrix G = S(S'ΣS)^(-1)S'Σ
- [ ] Apply: y_reconciled = G * y_base
- [ ] Return ReconciliationResult

### 8. Implement Matrix Inversion with Stability
- [ ] Create `_stable_inversion()` method
- [ ] Try Cholesky decomposition first (fast for PD matrices)
- [ ] Fallback to pseudo-inverse if singular
- [ ] Apply ridge regularization if needed
- [ ] Check condition number
- [ ] Return inverted matrix

### 9. Implement Ridge Regularization
- [ ] Create `_apply_regularization()` method
- [ ] Add λ*I to covariance matrix: Σ_reg = Σ + λ*I
- [ ] Configure λ from shrinkage_param
- [ ] Improve numerical stability
- [ ] Return regularized matrix

### 10. Implement Forecast Array Preparation
- [ ] Create `_prepare_forecast_array()` method
- [ ] Convert forecast dict to ordered array
- [ ] Match hierarchy node ordering
- [ ] Handle missing forecasts
- [ ] Validate dimensions
- [ ] Return forecast vector

### 11. Implement Result Reconstruction
- [ ] Create `_reconstruct_forecasts()` method
- [ ] Convert reconciled array back to dict
- [ ] Map to hierarchy node names
- [ ] Preserve original structure
- [ ] Return forecast dictionary

### 12. Implement Covariance Validation
- [ ] Create `_validate_covariance()` method
- [ ] Check positive definiteness
- [ ] Check symmetry
- [ ] Check dimensions match hierarchy
- [ ] Check condition number
- [ ] Log warnings for ill-conditioned matrices

### 13. Implement Numerical Stability Diagnostics
- [ ] Create `_check_numerical_stability()` method
- [ ] Calculate condition number
- [ ] Check for near-singularity
- [ ] Warn if condition number > 1e10
- [ ] Log matrix properties
- [ ] Return stability report

### 14. Implement Performance Optimization
- [ ] Use sparse matrices where possible
- [ ] Cache computed reconciliation matrix G
- [ ] Vectorize operations
- [ ] Use BLAS/LAPACK through numpy/scipy
- [ ] Profile critical sections

### 15. Handle Edge Cases
- [ ] Handle singular covariance matrices
- [ ] Handle nearly collinear forecasts
- [ ] Handle extreme forecast values
- [ ] Handle empty covariance (use identity)
- [ ] Validate reconciled forecasts non-negative (if needed)

### 16. Implement Reconciliation Diagnostics
- [ ] Override `get_diagnostics()` from base
- [ ] Report covariance method used
- [ ] Report condition number
- [ ] Report regularization applied
- [ ] Report numerical warnings
- [ ] Return diagnostic dict

### 17. Write Comprehensive Tests
- [ ] Create `tests/models/reconciliation/test_mint_reconciler.py`
- [ ] Test each covariance estimation method
- [ ] Test numerical stability
- [ ] Test with well-conditioned matrices
- [ ] Test with ill-conditioned matrices
- [ ] Test performance benchmarks
- [ ] Test constraint satisfaction

### 18. Write Integration Tests
- [ ] Test with real hierarchy (4 subsystems, 17 areas)
- [ ] Test with historical residuals
- [ ] Test end-to-end reconciliation pipeline
- [ ] Validate aggregation constraints satisfied
- [ ] Compare against unreconciled forecasts

---

## 💻 Implementation Details

### MinTReconciler Implementation

```python
"""Minimum Trace (MinT) reconciliation implementation."""
from typing import Dict, Any, Optional
import numpy as np
from scipy import linalg
import time

from src.models.reconciliation.base_reconciler import (
    BaseReconciler, ReconciliationResult, ReconciliationError
)
from src.models.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CovarianceEstimator:
    """Estimates forecast error covariance matrices."""
    
    def __init__(self, method: str = 'sample'):
        """
        Initialize covariance estimator.
        
        Args:
            method: Estimation method ('sample', 'structural', 'shrinkage', 'identity')
        """
        self.method = method
    
    def estimate(
        self,
        residuals: Optional[np.ndarray] = None,
        hierarchy: Optional[HierarchyDefinition] = None,
        shrinkage_param: float = 0.5
    ) -> np.ndarray:
        """
        Estimate covariance matrix.
        
        Args:
            residuals: Historical forecast residuals (n_samples x n_series)
            hierarchy: Hierarchy definition for structural method
            shrinkage_param: Shrinkage parameter for shrinkage method
        
        Returns:
            Covariance matrix
        """
        if self.method == 'identity':
            n = residuals.shape[1] if residuals is not None else len(hierarchy.nodes)
            return np.eye(n)
        
        elif self.method == 'sample':
            return self._estimate_sample(residuals)
        
        elif self.method == 'structural':
            return self._estimate_structural(residuals, hierarchy)
        
        elif self.method == 'shrinkage':
            sample_cov = self._estimate_sample(residuals)
            structural_cov = self._estimate_structural(residuals, hierarchy)
            return (shrinkage_param * structural_cov + 
                   (1 - shrinkage_param) * sample_cov)
        
        else:
            raise ValueError(f"Unknown covariance method: {self.method}")
    
    def _estimate_sample(self, residuals: np.ndarray) -> np.ndarray:
        """Estimate sample covariance."""
        if residuals is None:
            raise ValueError("Residuals required for sample covariance")
        
        # Sample covariance with bias correction
        n_samples = residuals.shape[0]
        cov = (residuals.T @ residuals) / (n_samples - 1)
        
        return cov
    
    def _estimate_structural(
        self,
        residuals: np.ndarray,
        hierarchy: HierarchyDefinition
    ) -> np.ndarray:
        """Estimate structural (diagonal) covariance."""
        if residuals is None:
            raise ValueError("Residuals required for structural covariance")
        
        # Diagonal covariance (variance for each series)
        variances = np.var(residuals, axis=0, ddof=1)
        cov = np.diag(variances)
        
        return cov


class MinTReconciler(BaseReconciler):
    """
    Minimum Trace (MinT) reconciliation.
    
    MinT produces optimal (minimum variance) reconciled forecasts by
    minimizing the trace of the forecast error covariance matrix.
    
    Algorithm:
        1. Construct aggregation matrix S from hierarchy
        2. Estimate forecast error covariance matrix Σ
        3. Compute reconciliation matrix: G = S(S'ΣS)^(-1)S'Σ
        4. Apply reconciliation: y_reconciled = G * y_base
    
    Covariance Methods:
        - identity: OLS reconciliation (Σ = I)
        - sample: Sample covariance from residuals
        - structural: Diagonal covariance (independent errors)
        - shrinkage: Convex combination of sample and structural
    
    Numerical Stability:
        - Ridge regularization: Σ_reg = Σ + λ*I
        - Cholesky decomposition for positive definite matrices
        - Pseudo-inverse fallback for singular matrices
        - Condition number monitoring
    
    Performance:
        - Target: <5 seconds for 26 series
        - Sparse matrix operations where possible
        - Cached reconciliation matrix
    
    Example:
        >>> reconciler = MinTReconciler(config={
        ...     'covariance_method': 'shrinkage',
        ...     'shrinkage_param': 0.5,
        ...     'regularization': 0.01
        ... })
        >>> result = reconciler.reconcile(base_forecasts, hierarchy, residuals)
        >>> print(f"Constraint violations: {len(result.constraint_violations)}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MinT reconciler.
        
        Args:
            config: Configuration dictionary with optional keys:
                - covariance_method: 'sample', 'structural', 'shrinkage', 'identity'
                - shrinkage_param: Shrinkage parameter (0-1)
                - regularization: Ridge regularization parameter
                - cache_reconciliation_matrix: Whether to cache G matrix
        """
        super().__init__(config)
        
        self.covariance_method = self.config.get('covariance_method', 'shrinkage')
        self.shrinkage_param = self.config.get('shrinkage_param', 0.5)
        self.regularization = self.config.get('regularization', 0.01)
        self.cache_matrix = self.config.get('cache_reconciliation_matrix', True)
        
        self.covariance_estimator = CovarianceEstimator(self.covariance_method)
        self.reconciliation_matrix: Optional[np.ndarray] = None
    
    def reconcile(
        self,
        base_forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition,
        covariance_matrix: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None
    ) -> ReconciliationResult:
        """
        Perform MinT reconciliation.
        
        Args:
            base_forecasts: Dictionary of base forecasts
            hierarchy: Hierarchy definition
            covariance_matrix: Optional pre-computed covariance
            residuals: Optional historical residuals for covariance estimation
        
        Returns:
            ReconciliationResult with reconciled forecasts
        
        Raises:
            ReconciliationError: If reconciliation fails
        """
        start_time = time.time()
        logger.info(f"Starting MinT reconciliation ({self.covariance_method})")
        
        # Validate inputs
        errors = self._validate_forecasts(base_forecasts, hierarchy)
        if errors:
            raise ReconciliationError(f"Validation errors: {errors}")
        
        # Get aggregation matrix S
        if hierarchy.aggregation_matrix is None:
            hierarchy.build_aggregation_matrix()
        S = hierarchy.aggregation_matrix
        
        logger.info(f"Aggregation matrix shape: {S.shape}")
        
        # Prepare forecast vector
        y_base, node_order = self._prepare_forecast_array(base_forecasts, hierarchy)
        
        # Estimate or use provided covariance matrix
        if covariance_matrix is None:
            logger.info("Estimating covariance matrix")
            Sigma = self.covariance_estimator.estimate(
                residuals=residuals,
                hierarchy=hierarchy,
                shrinkage_param=self.shrinkage_param
            )
        else:
            Sigma = covariance_matrix
        
        # Validate covariance
        self._validate_covariance(Sigma, len(node_order))
        
        # Apply regularization if needed
        if self.regularization > 0:
            Sigma = self._apply_regularization(Sigma)
        
        # Compute reconciliation matrix G = S(S'ΣS)^(-1)S'Σ
        if self.reconciliation_matrix is None or not self.cache_matrix:
            logger.info("Computing reconciliation matrix")
            G = self._compute_reconciliation_matrix(S, Sigma)
            
            if self.cache_matrix:
                self.reconciliation_matrix = G
        else:
            logger.info("Using cached reconciliation matrix")
            G = self.reconciliation_matrix
        
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
            method_name='MinT',
            hierarchy=hierarchy,
            processing_time=processing_time,
            covariance_method=self.covariance_method
        )
        
        logger.info(f"MinT reconciliation complete in {processing_time:.2f}s")
        
        return ReconciliationResult(
            reconciled_forecasts=reconciled_forecasts,
            original_forecasts=base_forecasts,
            metadata=metadata,
            constraint_violations=violations
        )
    
    def _compute_reconciliation_matrix(
        self,
        S: np.ndarray,
        Sigma: np.ndarray
    ) -> np.ndarray:
        """
        Compute MinT reconciliation matrix G = S(S'ΣS)^(-1)S'Σ.
        
        Args:
            S: Aggregation matrix
            Sigma: Covariance matrix
        
        Returns:
            Reconciliation matrix G
        """
        # Compute S'ΣS
        STS = S.T @ Sigma @ S
        
        # Check condition number
        cond_num = np.linalg.cond(STS)
        logger.info(f"Condition number of S'ΣS: {cond_num:.2e}")
        
        if cond_num > 1e10:
            logger.warning("Matrix is ill-conditioned, using pseudo-inverse")
        
        # Compute inverse using stable method
        STS_inv = self._stable_inversion(STS)
        
        # Compute reconciliation matrix
        G = S @ STS_inv @ S.T @ Sigma
        
        return G
    
    def _stable_inversion(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute stable matrix inversion.
        
        Args:
            matrix: Matrix to invert
        
        Returns:
            Inverted matrix
        """
        try:
            # Try Cholesky decomposition (fast for PD matrices)
            L = linalg.cholesky(matrix, lower=True)
            inv_matrix = linalg.cho_solve((L, True), np.eye(matrix.shape[0]))
            logger.debug("Used Cholesky decomposition")
        
        except linalg.LinAlgError:
            # Fallback to pseudo-inverse
            logger.warning("Cholesky failed, using pseudo-inverse")
            inv_matrix = linalg.pinv(matrix)
        
        return inv_matrix
    
    def _apply_regularization(self, Sigma: np.ndarray) -> np.ndarray:
        """Apply ridge regularization to covariance matrix."""
        n = Sigma.shape[0]
        Sigma_reg = Sigma + self.regularization * np.eye(n)
        logger.debug(f"Applied ridge regularization: λ={self.regularization}")
        return Sigma_reg
    
    def _prepare_forecast_array(
        self,
        forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition
    ) -> tuple[np.ndarray, List[str]]:
        """
        Prepare forecast array in consistent order.
        
        Args:
            forecasts: Forecast dictionary
            hierarchy: Hierarchy definition
        
        Returns:
            Tuple of (forecast_array, node_order)
        """
        # Order nodes by hierarchy (topological sort would be better)
        node_order = sorted(hierarchy.nodes.keys(), 
                           key=lambda x: (hierarchy.nodes[x].level, x))
        
        # Stack forecasts in order
        forecast_arrays = []
        for node_name in node_order:
            if node_name in forecasts:
                forecast_arrays.append(forecasts[node_name])
            else:
                # Use zeros for missing forecasts
                logger.warning(f"Missing forecast for {node_name}, using zeros")
                first_shape = next(iter(forecasts.values())).shape
                forecast_arrays.append(np.zeros(first_shape))
        
        forecast_array = np.concatenate(forecast_arrays)
        
        return forecast_array, node_order
    
    def _reconstruct_forecasts(
        self,
        reconciled_array: np.ndarray,
        node_order: List[str],
        original_forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Reconstruct forecast dictionary from array."""
        # Get original shape
        first_forecast = next(iter(original_forecasts.values()))
        n_timesteps = len(first_forecast)
        
        # Split reconciled array back into dict
        reconciled_forecasts = {}
        offset = 0
        
        for node_name in node_order:
            reconciled_forecasts[node_name] = reconciled_array[offset:offset+n_timesteps]
            offset += n_timesteps
        
        return reconciled_forecasts
    
    def _validate_covariance(self, Sigma: np.ndarray, expected_size: int) -> None:
        """Validate covariance matrix properties."""
        # Check dimensions
        if Sigma.shape[0] != Sigma.shape[1]:
            raise ReconciliationError("Covariance matrix must be square")
        
        if Sigma.shape[0] != expected_size:
            raise ReconciliationError(
                f"Covariance dimension mismatch: expected {expected_size}, "
                f"got {Sigma.shape[0]}"
            )
        
        # Check symmetry
        if not np.allclose(Sigma, Sigma.T):
            logger.warning("Covariance matrix is not symmetric")
        
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(Sigma)
        if np.any(eigenvalues < -1e-10):
            logger.warning("Covariance matrix has negative eigenvalues")
```

---

## 🧪 Testing & Validation

```python
"""Tests for MinT reconciler."""
import pytest
import numpy as np

from src.models.reconciliation.mint_reconciler import MinTReconciler
from src.models.reconciliation.hierarchy import HierarchyDefinition


@pytest.fixture
def sample_residuals():
    """Create sample residuals for covariance estimation."""
    np.random.seed(42)
    return np.random.randn(100, 6) * 50  # 100 samples, 6 series


def test_mint_reconciliation(sample_hierarchy_yaml, sample_residuals):
    """Test MinT reconciliation."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    forecasts = {
        'BR_National': np.array([1000, 1100, 1200]),
        'SE_Subsystem': np.array([600, 650, 700]),
        'S_Subsystem': np.array([400, 450, 500]),
        'Area_1': np.array([300, 325, 350]),
        'Area_2': np.array([300, 325, 350]),
        'Area_3': np.array([400, 450, 500])
    }
    
    reconciler = MinTReconciler(config={'covariance_method': 'identity'})
    result = reconciler.reconcile(forecasts, hierarchy)
    
    assert len(result.reconciled_forecasts) == 6
    assert result.metadata.reconciliation_method == 'MinT'


def test_covariance_estimation_methods(sample_residuals):
    """Test different covariance estimation methods."""
    from src.models.reconciliation.mint_reconciler import CovarianceEstimator
    
    # Sample covariance
    estimator = CovarianceEstimator('sample')
    cov_sample = estimator.estimate(residuals=sample_residuals)
    assert cov_sample.shape == (6, 6)
    
    # Structural covariance
    estimator = CovarianceEstimator('structural')
    cov_struct = estimator.estimate(residuals=sample_residuals)
    assert np.allclose(cov_struct, np.diag(np.diag(cov_struct)))  # Diagonal


def test_numerical_stability():
    """Test numerical stability with ill-conditioned matrices."""
    # Create ill-conditioned matrix
    A = np.array([[1.0, 0.9999], [0.9999, 1.0]])
    
    reconciler = MinTReconciler(config={'regularization': 0.01})
    A_reg = reconciler._apply_regularization(A)
    
    # Regularized matrix should have better condition number
    assert np.linalg.cond(A_reg) < np.linalg.cond(A)


def test_constraint_satisfaction(sample_hierarchy_yaml):
    """Test that reconciled forecasts satisfy constraints."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    forecasts = {
        'BR_National': np.array([1000]),
        'SE_Subsystem': np.array([550]),  # Violates constraint
        'S_Subsystem': np.array([400]),
        'Area_1': np.array([300]),
        'Area_2': np.array([300]),
        'Area_3': np.array([400])
    }
    
    reconciler = MinTReconciler()
    result = reconciler.reconcile(forecasts, hierarchy)
    
    # Check if constraints are better satisfied
    se_reconciled = result.reconciled_forecasts['SE_Subsystem'][0]
    area1_reconciled = result.reconciled_forecasts['Area_1'][0]
    area2_reconciled = result.reconciled_forecasts['Area_2'][0]
    
    # SE should equal sum of its areas (within tolerance)
    assert abs(se_reconciled - (area1_reconciled + area2_reconciled)) < 1e-6
```

---

## 📝 Technical Notes

### MinT Formula
- Reconciliation matrix: G = S(S'ΣS)^(-1)S'Σ
- Reconciled forecasts: y = G * y_base
- Minimizes trace of error covariance

### Numerical Stability
- Ridge regularization for ill-conditioned matrices
- Cholesky decomposition when possible
- Pseudo-inverse fallback

---

## 🔗 Dependencies

**Depends On:**
- PC-048-06A: Hierarchy Definition System
- PC-049-06A: Base Reconciler Interface

**Blocks:**
- PC-051-06A: Hierarchy Validator
- PC-052-06A: Constraint Enforcer
- Epic-06B: Advanced Reconciliation Methods

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] MinT algorithm implemented correctly
- [ ] Multiple covariance estimation methods working
- [ ] Numerical stability ensured
- [ ] Ridge regularization functional
- [ ] Performance <5s for 26 series
- [ ] Constraint satisfaction validated
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**Previous:** [PC-049-06A: Base Reconciler Interface](PC-049-06A-base-reconciler-interface.md)  
**Next:** [PC-051-06A: Hierarchy Validation Framework](PC-051-06A-hierarchy-validator.md)
