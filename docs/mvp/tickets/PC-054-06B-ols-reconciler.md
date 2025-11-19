# PC-054-06B: OLS Reconciliation Implementation

**Ticket ID:** PC-054-06B  
**Epic:** [Epic-06B: Advanced Reconciliation Methods](../epics/Epic-06B.md)  
**User Story:** US-1  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `OLSReconciler` class that performs Ordinary Least Squares reconciliation with sophisticated covariance estimation. Supports multiple covariance methods (sample, robust Ledoit-Wolf, factor-based), handles numerical stability through regularization, and optimizes performance using efficient linear algebra. OLS provides optimal reconciliation when forecast error structure is known or can be accurately estimated.

**As a** forecasting analyst  
**I want** an OLS reconciler with multiple covariance estimation methods  
**So that** I can achieve superior reconciliation performance when error structure is known

---

## ✅ Acceptance Criteria

- [ ] `OLSReconciler` implements generalized least squares reconciliation
- [ ] Multiple covariance estimation methods (sample, Ledoit-Wolf, factor)
- [ ] Diagonal and full covariance matrix support
- [ ] Numerical stability with ridge regularization
- [ ] Performance >5% better than MinT in structured error scenarios
- [ ] Efficient sparse matrix operations
- [ ] Integration with BaseReconciler interface
- [ ] Comprehensive covariance validation
- [ ] Performance <5s for 26 series
- [ ] Robust handling of ill-conditioned matrices

---

## 🔧 Implementation Tasks

### 1. Create OLS Reconciliation Module
- [ ] Create `src/models/reconciliation/ols_reconciler.py`
- [ ] Import scipy for linear algebra
- [ ] Import sklearn.covariance for robust estimators
- [ ] Import BaseReconciler interface
- [ ] Add module docstrings

### 2. Implement OLSReconciler Class
- [ ] Inherit from BaseReconciler
- [ ] Implement `__init__()` with covariance_method parameter
- [ ] Add regularization_param attribute
- [ ] Add covariance_estimator component
- [ ] Add numerical stability configuration

### 3. Implement Sample Covariance Estimator
- [ ] Create `SampleCovarianceEstimator` class
- [ ] Calculate empirical covariance: Σ = (1/n) R'R
- [ ] Apply bias correction for small samples
- [ ] Handle missing data in residuals
- [ ] Validate positive semi-definiteness
- [ ] Return sample covariance matrix

### 4. Implement Ledoit-Wolf Shrinkage Estimator
- [ ] Create `LedoitWolfEstimator` class
- [ ] Implement Ledoit-Wolf shrinkage formula
- [ ] Calculate optimal shrinkage intensity analytically
- [ ] Shrink towards identity or diagonal target
- [ ] Ensure positive definiteness
- [ ] Return shrunk covariance

### 5. Implement Factor-Based Covariance Estimator
- [ ] Create `FactorCovarianceEstimator` class
- [ ] Perform factor analysis on residuals
- [ ] Extract k principal factors
- [ ] Construct covariance: Σ = ΛΛ' + Ψ
- [ ] Validate factor loadings
- [ ] Return factor-based covariance

### 6. Implement OLS Reconciliation Core
- [ ] Create `reconcile()` method
- [ ] Validate inputs (forecasts, hierarchy, residuals)
- [ ] Estimate covariance matrix Σ using selected method
- [ ] Extract aggregation matrix S from hierarchy
- [ ] Compute OLS reconciliation matrix: G = S(S'Σ^(-1)S)^(-1)S'Σ^(-1)
- [ ] Apply reconciliation: y_reconciled = G * y_base
- [ ] Return ReconciliationResult

### 7. Implement Covariance Matrix Inversion
- [ ] Create `_invert_covariance()` method
- [ ] Try Cholesky decomposition first (fastest)
- [ ] Fallback to LU decomposition
- [ ] Ultimate fallback to pseudo-inverse
- [ ] Apply ridge regularization if needed
- [ ] Monitor condition number
- [ ] Return inverted covariance

### 8. Implement Ridge Regularization
- [ ] Create `_apply_ridge_regularization()` method
- [ ] Add regularization: Σ_reg = Σ + λI
- [ ] Configure λ from regularization_param
- [ ] Adaptive λ based on condition number
- [ ] Validate improved conditioning
- [ ] Return regularized covariance

### 9. Implement Design Matrix Construction
- [ ] Create `_build_design_matrix()` method
- [ ] Construct X matrix from hierarchy structure
- [ ] Handle aggregation relationships
- [ ] Ensure proper dimensions
- [ ] Validate rank
- [ ] Return design matrix

### 10. Implement Covariance Validation
- [ ] Create `_validate_covariance()` method
- [ ] Check symmetry
- [ ] Check positive definiteness (eigenvalues > 0)
- [ ] Check condition number
- [ ] Warn if ill-conditioned (cond > 1e10)
- [ ] Return validation status

### 11. Implement Efficient Matrix Operations
- [ ] Use scipy.linalg for optimized operations
- [ ] Exploit sparsity in S matrix
- [ ] Cache computed matrices where possible
- [ ] Use BLAS/LAPACK through numpy/scipy
- [ ] Profile critical operations
- [ ] Optimize memory usage

### 12. Implement Covariance Method Selection
- [ ] Create `_select_covariance_method()` helper
- [ ] Auto-select based on data characteristics
- [ ] Consider sample size vs dimensionality
- [ ] Use Ledoit-Wolf for high-dimensional cases
- [ ] Use sample for large sample sizes
- [ ] Return selected method

### 13. Implement Residual Preprocessing
- [ ] Create `_preprocess_residuals()` method
- [ ] Remove outliers (optional)
- [ ] Handle missing values
- [ ] Center residuals
- [ ] Validate temporal structure
- [ ] Return cleaned residuals

### 14. Implement Performance Comparison
- [ ] Create `compare_with_mint()` method
- [ ] Run both OLS and MinT on same data
- [ ] Calculate performance difference
- [ ] Identify scenarios where OLS superior
- [ ] Return comparison report

### 15. Handle Edge Cases
- [ ] Handle singular covariance matrices
- [ ] Handle rank-deficient design matrices
- [ ] Handle extremely small sample sizes
- [ ] Handle zero-variance series
- [ ] Graceful degradation to simpler methods

### 16. Implement Diagnostics
- [ ] Override `get_diagnostics()` from base
- [ ] Report covariance method used
- [ ] Report condition number
- [ ] Report regularization applied
- [ ] Report performance improvement
- [ ] Return diagnostic dict

### 17. Write Comprehensive Tests
- [ ] Create `tests/models/reconciliation/test_ols_reconciler.py`
- [ ] Test each covariance estimation method
- [ ] Test numerical stability
- [ ] Test with well/ill-conditioned matrices
- [ ] Test performance vs MinT
- [ ] Test edge cases
- [ ] Test integration with hierarchy

### 18. Create Performance Benchmarks
- [ ] Create `benchmarks/ols_reconciler_benchmark.py`
- [ ] Benchmark computational performance
- [ ] Compare with MinT baseline
- [ ] Test scalability with hierarchy size
- [ ] Memory usage profiling
- [ ] Document performance characteristics

---

## 💻 Implementation Details

### OLSReconciler Implementation

```python
"""OLS reconciliation with advanced covariance estimation."""
from typing import Dict, Any, Optional
import numpy as np
from scipy import linalg
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import time

from src.models.reconciliation.base_reconciler import (
    BaseReconciler, ReconciliationResult, ReconciliationError
)
from src.models.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CovarianceMethod:
    """Covariance estimation methods."""
    SAMPLE = "sample"
    LEDOIT_WOLF = "ledoit_wolf"
    FACTOR = "factor"
    DIAGONAL = "diagonal"


class SampleCovarianceEstimator:
    """
    Sample covariance estimator with bias correction.
    
    Computes empirical covariance matrix from residuals.
    """
    
    def estimate(self, residuals: np.ndarray) -> np.ndarray:
        """
        Estimate sample covariance.
        
        Args:
            residuals: Historical forecast residuals (n_samples, n_series)
        
        Returns:
            Sample covariance matrix (n_series, n_series)
        """
        n_samples = residuals.shape[0]
        
        if n_samples < 2:
            raise ValueError("Need at least 2 samples for covariance estimation")
        
        # Center residuals
        residuals_centered = residuals - residuals.mean(axis=0)
        
        # Compute sample covariance with bias correction
        cov = (residuals_centered.T @ residuals_centered) / (n_samples - 1)
        
        return cov


class LedoitWolfEstimator:
    """
    Ledoit-Wolf shrinkage estimator.
    
    Optimal shrinkage towards identity matrix for high-dimensional cases.
    """
    
    def __init__(self, assume_centered: bool = False):
        """
        Initialize Ledoit-Wolf estimator.
        
        Args:
            assume_centered: Whether residuals are already centered
        """
        self.assume_centered = assume_centered
        self.estimator = LedoitWolf(assume_centered=assume_centered)
    
    def estimate(self, residuals: np.ndarray) -> np.ndarray:
        """
        Estimate shrunk covariance using Ledoit-Wolf.
        
        Args:
            residuals: Historical forecast residuals
        
        Returns:
            Shrunk covariance matrix
        """
        self.estimator.fit(residuals)
        
        logger.info(f"Ledoit-Wolf shrinkage intensity: {self.estimator.shrinkage_:.3f}")
        
        return self.estimator.covariance_


class FactorCovarianceEstimator:
    """
    Factor-based covariance estimator.
    
    Models covariance using factor structure: Σ = ΛΛ' + Ψ
    """
    
    def __init__(self, n_factors: Optional[int] = None):
        """
        Initialize factor estimator.
        
        Args:
            n_factors: Number of factors (None for automatic selection)
        """
        self.n_factors = n_factors
    
    def estimate(self, residuals: np.ndarray) -> np.ndarray:
        """
        Estimate factor-based covariance.
        
        Args:
            residuals: Historical forecast residuals
        
        Returns:
            Factor-based covariance matrix
        """
        n_samples, n_series = residuals.shape
        
        # Determine number of factors
        if self.n_factors is None:
            # Use Kaiser criterion: keep factors with eigenvalue > 1
            sample_cov = SampleCovarianceEstimator().estimate(residuals)
            eigenvalues = np.linalg.eigvalsh(sample_cov)
            self.n_factors = np.sum(eigenvalues > 1)
            logger.info(f"Auto-selected {self.n_factors} factors")
        
        # Perform SVD for factor extraction
        U, s, Vt = np.linalg.svd(residuals, full_matrices=False)
        
        # Extract top k factors
        k = min(self.n_factors, len(s))
        Lambda = Vt[:k, :].T * s[:k]  # Factor loadings
        
        # Estimate specific variances (diagonal)
        factor_cov = Lambda @ Lambda.T
        sample_cov = SampleCovarianceEstimator().estimate(residuals)
        specific_var = np.diag(np.maximum(np.diag(sample_cov - factor_cov), 1e-6))
        
        # Combine: Σ = ΛΛ' + Ψ
        cov = factor_cov + specific_var
        
        return cov


class OLSReconciler(BaseReconciler):
    """
    Ordinary Least Squares (OLS) hierarchical reconciliation.
    
    OLS reconciliation produces optimal (minimum variance) reconciled
    forecasts when the forecast error covariance structure is known
    or can be accurately estimated.
    
    Algorithm:
        1. Estimate forecast error covariance Σ from historical residuals
        2. Compute reconciliation matrix: G = S(S'Σ^(-1)S)^(-1)S'Σ^(-1)
        3. Apply reconciliation: y_reconciled = G * y_base
    
    Covariance Methods:
        - sample: Empirical covariance from residuals
        - ledoit_wolf: Ledoit-Wolf shrinkage (optimal for high-dim)
        - factor: Factor-based covariance (structured errors)
        - diagonal: Diagonal covariance (independent errors)
    
    Numerical Stability:
        - Ridge regularization: Σ_reg = Σ + λI
        - Multiple decomposition methods (Cholesky, LU, SVD)
        - Condition number monitoring and adaptive regularization
        - Graceful degradation to pseudo-inverse
    
    Performance:
        - Target: <5 seconds for 26 series
        - Sparse matrix operations where possible
        - Efficient BLAS/LAPACK through scipy
        - Optional matrix caching
    
    Example:
        >>> reconciler = OLSReconciler(config={
        ...     'covariance_method': 'ledoit_wolf',
        ...     'regularization': 0.01
        ... })
        >>> 
        >>> result = reconciler.reconcile(
        ...     base_forecasts=forecasts,
        ...     hierarchy=hierarchy,
        ...     historical_residuals=residuals
        ... )
        >>> 
        >>> print(f"Method: {result.metadata.reconciliation_method}")
        >>> print(f"Improvement over MinT: {result.metadata.performance_metrics['improvement']:.1%}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OLS reconciler.
        
        Args:
            config: Configuration dictionary with optional keys:
                - covariance_method: 'sample', 'ledoit_wolf', 'factor', 'diagonal'
                - regularization: Ridge regularization parameter (default: 1e-6)
                - n_factors: Number of factors for factor method
                - adaptive_regularization: Auto-adjust based on condition number
                - cache_matrices: Cache computed reconciliation matrices
        """
        super().__init__(config)
        
        self.covariance_method = self.config.get('covariance_method', 'ledoit_wolf')
        self.regularization = self.config.get('regularization', 1e-6)
        self.n_factors = self.config.get('n_factors', None)
        self.adaptive_regularization = self.config.get('adaptive_regularization', True)
        self.cache_matrices = self.config.get('cache_matrices', True)
        
        # Initialize covariance estimator
        self._initialize_covariance_estimator()
        
        # Cache
        self._cached_covariance: Optional[np.ndarray] = None
        self._cached_reconciliation_matrix: Optional[np.ndarray] = None
    
    def _initialize_covariance_estimator(self):
        """Initialize covariance estimator based on method."""
        if self.covariance_method == CovarianceMethod.SAMPLE:
            self.cov_estimator = SampleCovarianceEstimator()
        elif self.covariance_method == CovarianceMethod.LEDOIT_WOLF:
            self.cov_estimator = LedoitWolfEstimator()
        elif self.covariance_method == CovarianceMethod.FACTOR:
            self.cov_estimator = FactorCovarianceEstimator(n_factors=self.n_factors)
        elif self.covariance_method == CovarianceMethod.DIAGONAL:
            self.cov_estimator = None  # Will use diagonal extraction
        else:
            raise ValueError(f"Unknown covariance method: {self.covariance_method}")
    
    def reconcile(
        self,
        base_forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition,
        historical_residuals: Optional[np.ndarray] = None,
        covariance_matrix: Optional[np.ndarray] = None
    ) -> ReconciliationResult:
        """
        Perform OLS reconciliation.
        
        Args:
            base_forecasts: Dictionary of base forecasts
            hierarchy: Hierarchy definition
            historical_residuals: Historical forecast residuals for covariance estimation
            covariance_matrix: Optional pre-computed covariance (overrides estimation)
        
        Returns:
            ReconciliationResult with reconciled forecasts
        
        Raises:
            ReconciliationError: If reconciliation fails
        """
        start_time = time.time()
        logger.info(f"Starting OLS reconciliation (method: {self.covariance_method})")
        
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
        if covariance_matrix is not None:
            Sigma = covariance_matrix
        elif historical_residuals is not None:
            logger.info("Estimating covariance from historical residuals")
            Sigma = self._estimate_covariance(historical_residuals)
        else:
            # Use identity (equivalent to standard OLS)
            logger.warning("No residuals provided, using identity covariance")
            n = len(node_order)
            Sigma = np.eye(n)
        
        # Validate and regularize covariance
        self._validate_covariance(Sigma, len(node_order))
        Sigma = self._apply_regularization(Sigma)
        
        # Compute reconciliation matrix G = S(S'Σ^(-1)S)^(-1)S'Σ^(-1)
        if self._cached_reconciliation_matrix is None or not self.cache_matrices:
            logger.info("Computing OLS reconciliation matrix")
            G = self._compute_ols_matrix(S, Sigma)
            
            if self.cache_matrices:
                self._cached_reconciliation_matrix = G
                self._cached_covariance = Sigma
        else:
            logger.info("Using cached reconciliation matrix")
            G = self._cached_reconciliation_matrix
        
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
            method_name='OLS',
            hierarchy=hierarchy,
            processing_time=processing_time,
            covariance_method=self.covariance_method
        )
        
        # Add performance metrics
        metadata.performance_metrics['condition_number'] = float(np.linalg.cond(Sigma))
        metadata.performance_metrics['regularization_applied'] = self.regularization
        
        logger.info(f"OLS reconciliation complete in {processing_time:.2f}s")
        
        return ReconciliationResult(
            reconciled_forecasts=reconciled_forecasts,
            original_forecasts=base_forecasts,
            metadata=metadata,
            constraint_violations=violations
        )
    
    def _estimate_covariance(self, residuals: np.ndarray) -> np.ndarray:
        """
        Estimate covariance matrix from residuals.
        
        Args:
            residuals: Historical forecast residuals (n_samples, n_series)
        
        Returns:
            Estimated covariance matrix
        """
        if self.covariance_method == CovarianceMethod.DIAGONAL:
            # Diagonal covariance (independent errors)
            sample_cov = SampleCovarianceEstimator().estimate(residuals)
            cov = np.diag(np.diag(sample_cov))
        else:
            # Use configured estimator
            cov = self.cov_estimator.estimate(residuals)
        
        return cov
    
    def _compute_ols_matrix(self, S: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """
        Compute OLS reconciliation matrix: G = S(S'Σ^(-1)S)^(-1)S'Σ^(-1).
        
        Args:
            S: Aggregation matrix
            Sigma: Covariance matrix
        
        Returns:
            OLS reconciliation matrix G
        """
        # Invert covariance matrix
        Sigma_inv = self._invert_covariance(Sigma)
        
        # Compute S'Σ^(-1)S
        STS = S.T @ Sigma_inv @ S
        
        # Check condition number
        cond_num = np.linalg.cond(STS)
        logger.info(f"Condition number of S'Σ^(-1)S: {cond_num:.2e}")
        
        if cond_num > 1e10:
            logger.warning("Matrix is ill-conditioned")
        
        # Invert S'Σ^(-1)S
        STS_inv = self._stable_inversion(STS)
        
        # Compute full reconciliation matrix
        G = S @ STS_inv @ S.T @ Sigma_inv
        
        return G
    
    def _invert_covariance(self, Sigma: np.ndarray) -> np.ndarray:
        """
        Invert covariance matrix using stable method.
        
        Args:
            Sigma: Covariance matrix
        
        Returns:
            Inverted covariance matrix
        """
        try:
            # Try Cholesky (fastest for PD matrices)
            L = linalg.cholesky(Sigma, lower=True)
            Sigma_inv = linalg.cho_solve((L, True), np.eye(Sigma.shape[0]))
            logger.debug("Used Cholesky decomposition for inversion")
        
        except linalg.LinAlgError:
            try:
                # Fallback to LU decomposition
                Sigma_inv = linalg.inv(Sigma)
                logger.debug("Used LU decomposition for inversion")
            
            except linalg.LinAlgError:
                # Ultimate fallback to pseudo-inverse
                logger.warning("Using pseudo-inverse (matrix singular)")
                Sigma_inv = linalg.pinv(Sigma)
        
        return Sigma_inv
    
    def _stable_inversion(self, matrix: np.ndarray) -> np.ndarray:
        """Compute stable matrix inversion (same as MinT)."""
        try:
            L = linalg.cholesky(matrix, lower=True)
            inv_matrix = linalg.cho_solve((L, True), np.eye(matrix.shape[0]))
        except linalg.LinAlgError:
            logger.warning("Cholesky failed, using pseudo-inverse")
            inv_matrix = linalg.pinv(matrix)
        
        return inv_matrix
    
    def _apply_regularization(self, Sigma: np.ndarray) -> np.ndarray:
        """Apply ridge regularization to covariance matrix."""
        if self.regularization <= 0:
            return Sigma
        
        # Check if adaptive regularization needed
        if self.adaptive_regularization:
            cond_num = np.linalg.cond(Sigma)
            if cond_num > 1e8:
                # Increase regularization for ill-conditioned matrices
                reg_param = self.regularization * (cond_num / 1e8)
                logger.info(f"Adaptive regularization: λ={reg_param:.2e}")
            else:
                reg_param = self.regularization
        else:
            reg_param = self.regularization
        
        n = Sigma.shape[0]
        Sigma_reg = Sigma + reg_param * np.eye(n)
        
        return Sigma_reg
    
    def _prepare_forecast_array(
        self,
        forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition
    ) -> tuple:
        """Prepare forecast array (same as MinT implementation)."""
        node_order = sorted(hierarchy.nodes.keys(),
                          key=lambda x: (hierarchy.nodes[x].level, x))
        
        forecast_arrays = []
        for node_name in node_order:
            if node_name in forecasts:
                forecast_arrays.append(forecasts[node_name])
            else:
                logger.warning(f"Missing forecast for {node_name}, using zeros")
                first_shape = next(iter(forecasts.values())).shape
                forecast_arrays.append(np.zeros(first_shape))
        
        forecast_array = np.concatenate(forecast_arrays)
        
        return forecast_array, node_order
    
    def _reconstruct_forecasts(
        self,
        reconciled_array: np.ndarray,
        node_order: list,
        original_forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Reconstruct forecast dictionary (same as MinT)."""
        first_forecast = next(iter(original_forecasts.values()))
        n_timesteps = len(first_forecast)
        
        reconciled_forecasts = {}
        offset = 0
        
        for node_name in node_order:
            reconciled_forecasts[node_name] = reconciled_array[offset:offset+n_timesteps]
            offset += n_timesteps
        
        return reconciled_forecasts
    
    def _validate_covariance(self, Sigma: np.ndarray, expected_size: int) -> None:
        """Validate covariance matrix (same as MinT implementation)."""
        if Sigma.shape[0] != Sigma.shape[1]:
            raise ReconciliationError("Covariance matrix must be square")
        
        if Sigma.shape[0] != expected_size:
            raise ReconciliationError(
                f"Covariance dimension mismatch: expected {expected_size}, "
                f"got {Sigma.shape[0]}"
            )
        
        if not np.allclose(Sigma, Sigma.T):
            logger.warning("Covariance matrix is not symmetric")
        
        eigenvalues = np.linalg.eigvalsh(Sigma)
        if np.any(eigenvalues < -1e-10):
            logger.warning("Covariance matrix has negative eigenvalues")
```

---

## 🧪 Testing & Validation

```python
"""Tests for OLS reconciler."""
import pytest
import numpy as np

from src.models.reconciliation.ols_reconciler import (
    OLSReconciler, SampleCovarianceEstimator, LedoitWolfEstimator
)
from src.models.reconciliation.hierarchy import HierarchyDefinition


@pytest.fixture
def sample_residuals():
    """Create sample residuals."""
    np.random.seed(42)
    # Structured covariance (correlated errors)
    cov_true = np.array([[1.0, 0.5, 0.3],
                        [0.5, 1.0, 0.4],
                        [0.3, 0.4, 1.0]])
    return np.random.multivariate_normal([0, 0, 0], cov_true, size=100)


def test_sample_covariance_estimation(sample_residuals):
    """Test sample covariance estimation."""
    estimator = SampleCovarianceEstimator()
    cov = estimator.estimate(sample_residuals)
    
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)  # Symmetric


def test_ledoit_wolf_estimation(sample_residuals):
    """Test Ledoit-Wolf shrinkage."""
    estimator = LedoitWolfEstimator()
    cov = estimator.estimate(sample_residuals)
    
    assert cov.shape == (3, 3)
    # Shrunk covariance should be more stable
    assert np.linalg.cond(cov) < np.linalg.cond(
        SampleCovarianceEstimator().estimate(sample_residuals)
    )


def test_ols_reconciliation(sample_hierarchy_yaml, sample_residuals):
    """Test OLS reconciliation."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    forecasts = {
        'BR_National': np.array([1000, 1100, 1200]),
        'SE_Subsystem': np.array([600, 650, 700]),
        'S_Subsystem': np.array([400, 450, 500])
    }
    
    reconciler = OLSReconciler(config={'covariance_method': 'sample'})
    result = reconciler.reconcile(forecasts, hierarchy, sample_residuals)
    
    assert len(result.reconciled_forecasts) >= 3
    assert result.metadata.reconciliation_method == 'OLS'


def test_numerical_stability():
    """Test numerical stability with ill-conditioned covariance."""
    # Create nearly singular covariance
    cov = np.array([[1.0, 0.9999], [0.9999, 1.0]])
    residuals = np.random.multivariate_normal([0, 0], cov, size=50)
    
    reconciler = OLSReconciler(config={
        'covariance_method': 'sample',
        'regularization': 0.01,
        'adaptive_regularization': True
    })
    
    # Should handle gracefully with regularization
    cov_estimated = SampleCovarianceEstimator().estimate(residuals)
    cov_reg = reconciler._apply_regularization(cov_estimated)
    
    # Regularized should have better condition number
    assert np.linalg.cond(cov_reg) < np.linalg.cond(cov_estimated)


def test_ols_vs_mint_comparison(sample_hierarchy_yaml, sample_residuals):
    """Test OLS performance comparison with MinT."""
    from src.models.reconciliation.mint_reconciler import MinTReconciler
    
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    forecasts = {
        'BR_National': np.array([1000]),
        'SE_Subsystem': np.array([550]),
        'S_Subsystem': np.array([400])
    }
    
    # OLS reconciliation
    ols = OLSReconciler(config={'covariance_method': 'ledoit_wolf'})
    ols_result = ols.reconcile(forecasts, hierarchy, sample_residuals)
    
    # MinT reconciliation
    mint = MinTReconciler(config={'covariance_method': 'shrinkage'})
    mint_result = mint.reconcile(forecasts, hierarchy, residuals=sample_residuals)
    
    # Both should satisfy constraints
    assert len(ols_result.constraint_violations) <= len(mint_result.constraint_violations)
```

---

## 📝 Technical Notes

### OLS Formula
- Reconciliation matrix: G = S(S'Σ^(-1)S)^(-1)S'Σ^(-1)
- Optimal when Σ accurately estimated
- Reduces to MinT when Σ = I

### Covariance Methods
- Sample: simple but unstable with small n
- Ledoit-Wolf: optimal shrinkage for high dimensions
- Factor: captures structured error patterns

### Numerical Stability
- Ridge regularization essential
- Adaptive regularization based on condition number
- Multiple decomposition fallbacks

---

## 🔗 Dependencies

**Depends On:**
- PC-048-06A: Hierarchy Definition System
- PC-049-06A: Base Reconciler Interface
- PC-050-06A: MinT Reconciler (for comparison)

**Blocks:**
- PC-055-06B: WLS Reconciler Implementation
- PC-057-06B: Reconciler Selection System

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] OLS algorithm implemented correctly
- [ ] Multiple covariance methods working
- [ ] Numerical stability ensured
- [ ] Performance >5% better than MinT in structured scenarios
- [ ] Efficient implementation <5s
- [ ] Ridge regularization functional
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Performance benchmarks documented
- [ ] Code reviewed and approved

---

**Epic:** [Epic-06B: Advanced Reconciliation Methods](../epics/Epic-06B.md)  
**Previous:** [PC-053-06A: Loss Calculation System](../PC-053-06A-loss-calculator.md)  
**Next:** [PC-055-06B: WLS Reconciler Implementation](PC-055-06B-wls-reconciler.md)
