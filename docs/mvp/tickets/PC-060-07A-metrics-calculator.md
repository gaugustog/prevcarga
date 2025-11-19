# PC-060-07A: Core Metrics Calculator Implementation

**Ticket ID:** PC-060-07A  
**Epic:** Epic-07A - Core Metrics & Analysis  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Sprint:** Week 19 (Days 1-2)

---

## 📋 Description

Implement the core metrics calculation engine (`MetricsCalculator`) that provides accurate and efficient computation of standard forecasting metrics (MAPE, MAE, RMSE, R²) with robust handling of edge cases including zeros, near-zero values, and missing data. Include weighted metrics support and bootstrap-based confidence interval calculation.

---

## 🎯 Acceptance Criteria

- [ ] `MetricsCalculator` class implemented with all core metrics (MAPE, MAE, RMSE, R²)
- [ ] Symmetric MAPE (sMAPE) option for robust zero handling
- [ ] Weighted metrics support with custom weight arrays
- [ ] Bootstrap confidence intervals (1000 iterations, 95% level)
- [ ] Statistical significance testing (paired t-test, Wilcoxon)
- [ ] Handles edge cases: NaN values, zeros, near-zeros, single values
- [ ] Performance: <5 seconds for 26 series × 9 horizons evaluation
- [ ] Unit tests with 85%+ coverage including edge cases

---

## 📐 Technical Specifications

### Core Components

#### 1. MetricsConfig
```python
@dataclass
class MetricsConfig:
    epsilon: float = 1e-8
    use_symmetric_mape: bool = True
    confidence_level: float = 0.95
    bootstrap_iterations: int = 1000
    weighted_metrics: bool = False
```

#### 2. MetricsResult
```python
@dataclass
class MetricsResult:
    mape: float
    mae: float
    rmse: float
    mse: float
    r2: float
    confidence_interval: Dict[str, tuple]
    sample_size: int
    metadata: Dict[str, any]
```

#### 3. MetricsCalculator
```python
class MetricsCalculator:
    def __init__(self, config: MetricsConfig)
    
    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray],
        weights: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, MetricsResult]
    
    def calculate_mape_robust(self, actual: np.ndarray, forecast: np.ndarray) -> float
    def calculate_mae(self, actual: np.ndarray, forecast: np.ndarray, weights: Optional[np.ndarray]) -> float
    def calculate_rmse(self, actual: np.ndarray, forecast: np.ndarray, weights: Optional[np.ndarray]) -> float
    def calculate_r2(self, actual: np.ndarray, forecast: np.ndarray) -> float
    def calculate_confidence_intervals(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, tuple]
```

#### 4. StatisticalTests
```python
class StatisticalTests:
    def paired_t_test(self, errors_a: np.ndarray, errors_b: np.ndarray, alpha: float = 0.05) -> Dict[str, any]
    def wilcoxon_test(self, errors_a: np.ndarray, errors_b: np.ndarray, alpha: float = 0.05) -> Dict[str, any]
```

### Metric Formulas

1. **Standard MAPE:**
   ```
   MAPE = 100 * mean(|actual - forecast| / max(|actual|, ε))
   ```

2. **Symmetric MAPE (sMAPE):**
   ```
   sMAPE = 100 * mean(|actual - forecast| / ((|actual| + |forecast|) / 2))
   ```

3. **MAE (weighted):**
   ```
   MAE = sum(weights * |actual - forecast|) / sum(weights)
   ```

4. **RMSE (weighted):**
   ```
   RMSE = sqrt(sum(weights * (actual - forecast)²) / sum(weights))
   ```

5. **R² Score:**
   ```
   R² = 1 - (SS_res / SS_tot)
   where SS_res = sum((actual - forecast)²)
         SS_tot = sum((actual - mean(actual))²)
   ```

---

## 🔧 Implementation Tasks

### Task 1: Core Data Structures (2 hours)
- [ ] Implement `MetricsConfig` dataclass with validation
- [ ] Implement `MetricsResult` dataclass with proper typing
- [ ] Add configuration validation logic

### Task 2: Basic Metrics Implementation (4 hours)
- [ ] Implement `calculate_mae()` with optional weighting
- [ ] Implement `calculate_rmse()` with optional weighting
- [ ] Implement `calculate_r2()` with edge case handling
- [ ] Add NaN filtering for all metrics

### Task 3: Robust MAPE Implementation (3 hours)
- [ ] Implement standard MAPE with epsilon protection
- [ ] Implement symmetric MAPE alternative
- [ ] Add configuration switch between methods
- [ ] Handle zero/near-zero denominators

### Task 4: Confidence Intervals (4 hours)
- [ ] Implement bootstrap resampling logic
- [ ] Calculate percentile-based confidence intervals
- [ ] Optimize bootstrap performance (vectorization)
- [ ] Add early stopping for convergence

### Task 5: Statistical Tests (3 hours)
- [ ] Implement paired t-test using scipy.stats
- [ ] Implement Wilcoxon signed-rank test
- [ ] Add result interpretation and p-value reporting
- [ ] Handle insufficient sample sizes gracefully

### Task 6: Integration and Testing (4 hours)
- [ ] Write unit tests for each metric calculation
- [ ] Test edge cases: zeros, NaNs, single values, identical values
- [ ] Performance testing with 26 series × 9 horizons
- [ ] Integration tests with sample data from Epic-04/05/06

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_mape_standard_calculation()
def test_mape_symmetric_with_zeros()
def test_mae_weighted_calculation()
def test_rmse_with_nan_handling()
def test_r2_perfect_fit()
def test_r2_no_fit()
def test_confidence_intervals_coverage()
def test_bootstrap_convergence()
def test_paired_ttest_significance()
def test_wilcoxon_nonparametric()
```

### Edge Case Tests
```python
def test_all_zeros_actual()
def test_all_zeros_forecast()
def test_single_value()
def test_identical_values()
def test_all_nan_values()
def test_mixed_nan_values()
def test_near_zero_denominators()
def test_large_value_stability()
```

### Performance Tests
```python
def test_performance_26_series_9_horizons()
def test_memory_usage_large_dataset()
def test_bootstrap_performance()
```

---

## 📊 Success Metrics

### Performance Targets
- Full evaluation (26 series × 9 horizons): **<5 seconds**
- Memory usage: **<500 MB**
- Bootstrap confidence intervals: **<2 seconds per series**

### Quality Targets
- Unit test coverage: **≥85%**
- Edge case handling: **100% of identified cases**
- Numerical stability: **No NaN/Inf in valid inputs**
- Statistical accuracy: **CI coverage ≥94% (validated via simulation)**

---

## 📚 Dependencies

### Required Packages
```python
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0
```

### Epic Dependencies
- Epic-04: Model training outputs (for testing)
- Epic-05A: Base combination outputs (for integration)
- Epic-06A: Reconciliation outputs (for validation)

---

## 🔗 Related Tickets
- **Blocks:** PC-061-07A (Percentile Analyzer)
- **Blocks:** PC-062-07A (Time Period Analyzer)
- **Blocks:** PC-063-07A (Horizon Analyzer)
- **Relates to:** Epic-05A, Epic-06A outputs

---

## 📖 Documentation Requirements

- [ ] API documentation with docstrings for all public methods
- [ ] Mathematical formulas and references
- [ ] Edge case handling documentation
- [ ] Performance optimization notes
- [ ] Usage examples with Epic-05/06 outputs

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥85% coverage
- [ ] Edge case tests passing (100%)
- [ ] Performance benchmarks met
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] Integration validated with Epic-05/06 outputs
- [ ] Ready for PC-061-07A (Percentile Analyzer)
