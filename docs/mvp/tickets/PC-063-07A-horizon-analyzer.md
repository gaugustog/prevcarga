# PC-063-07A: Horizon Performance Analyzer

**Ticket ID:** PC-063-07A  
**Epic:** Epic-07A - Core Metrics & Analysis  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Sprint:** Week 19 (Day 5)

---

## 📋 Description

Implement the horizon-specific performance analyzer (`HorizonAnalyzer`) that tracks forecast accuracy degradation across horizons D+0 to D+8, models accuracy decay using multiple regression approaches (linear, polynomial, exponential), and enables model comparison to identify superior short-term vs long-term performers.

---

## 🎯 Acceptance Criteria

- [ ] `HorizonAnalyzer` tracks performance across D+0 to D+8 horizons
- [ ] Three decay models implemented: linear, polynomial, exponential
- [ ] Model selection based on R² goodness-of-fit
- [ ] Comparison of multiple models' horizon-specific performance
- [ ] Identification of best model per horizon
- [ ] Horizon-specific confidence intervals
- [ ] Performance: <2 seconds for full horizon analysis
- [ ] Unit tests with 80%+ coverage

---

## 📐 Technical Specifications

### Core Components

#### 1. HorizonAnalysis
```python
@dataclass
class HorizonAnalysis:
    horizon_metrics: Dict[int, MetricsResult]  # horizon -> metrics
    decay_model: Dict[str, any]
    model_ranking: List[tuple]  # [(model_name, mape), ...]
```

#### 2. HorizonAnalyzer
```python
class HorizonAnalyzer:
    def __init__(self)
    
    def analyze_horizon_performance(
        self,
        predictions_by_horizon: Dict[int, Dict[str, np.ndarray]],
        actuals_by_horizon: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[str, HorizonAnalysis]
    
    def model_accuracy_decay(
        self,
        horizon_metrics: Dict[int, MetricsResult]
    ) -> Dict[str, any]
    
    def compare_models_by_horizon(
        self,
        model_results: Dict[str, Dict[int, MetricsResult]]
    ) -> Dict[int, List[tuple]]
```

### Decay Models

1. **Linear Decay Model:**
   ```
   MAPE(h) = a * h + b
   
   where:
   - h = horizon (0-8)
   - a = decay rate (slope)
   - b = baseline accuracy (intercept)
   
   Fit using: np.polyfit(horizons, mapes, 1)
   ```

2. **Polynomial Decay Model (Degree 2):**
   ```
   MAPE(h) = a * h² + b * h + c
   
   Fit using: np.polyfit(horizons, mapes, 2)
   ```

3. **Exponential Decay Model:**
   ```
   MAPE(h) = a * exp(b * h)
   
   Linearize: log(MAPE) = log(a) + b * h
   Fit using: np.polyfit(horizons, log(mapes), 1)
   then transform back
   ```

### Model Selection
Select best model based on R² (coefficient of determination):
```
R² = 1 - (SS_res / SS_tot)

where:
- SS_res = sum((observed - predicted)²)
- SS_tot = sum((observed - mean)²)

Choose model with highest R²
```

---

## 🔧 Implementation Tasks

### Task 1: Core Data Structures (1 hour)
- [ ] Implement `HorizonAnalysis` dataclass
- [ ] Add validation for horizon range (0-8)
- [ ] Create result formatting utilities

### Task 2: Horizon Metrics Calculation (2 hours)
- [ ] Implement horizon-specific metrics extraction
- [ ] Handle missing horizons gracefully
- [ ] Calculate metrics using MetricsCalculator
- [ ] Store results in structured format

### Task 3: Linear Decay Model (2 hours)
- [ ] Implement linear regression fitting
- [ ] Calculate slope and intercept
- [ ] Compute R² goodness-of-fit
- [ ] Generate interpretable formula string

### Task 4: Polynomial Decay Model (2 hours)
- [ ] Implement polynomial regression (degree 2)
- [ ] Handle minimum sample size requirements (≥4 horizons)
- [ ] Calculate R² for polynomial fit
- [ ] Compare with linear model

### Task 5: Exponential Decay Model (2 hours)
- [ ] Implement log-linear transformation
- [ ] Fit exponential model via linearization
- [ ] Transform coefficients back to exponential form
- [ ] Handle fitting failures gracefully
- [ ] Calculate R² for exponential model

### Task 6: Model Comparison Framework (2 hours)
- [ ] Implement multi-model comparison logic
- [ ] Rank models by horizon performance
- [ ] Identify best model per horizon
- [ ] Generate comparison summaries

### Task 7: Integration and Testing (3 hours)
- [ ] Unit tests for each decay model
- [ ] Test model selection logic
- [ ] Test multi-model comparison
- [ ] Performance testing with full dataset
- [ ] Integration with MetricsCalculator

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_horizon_metrics_extraction()
def test_linear_decay_fitting()
def test_polynomial_decay_fitting()
def test_exponential_decay_fitting()
def test_model_selection_by_r2()
def test_model_comparison_ranking()
def test_insufficient_horizons_handling()
def test_missing_horizon_handling()
```

### Decay Model Tests
```python
def test_linear_perfect_fit()
def test_polynomial_better_than_linear()
def test_exponential_monotonic_increase()
def test_decay_model_r2_calculation()
```

### Edge Case Tests
```python
def test_single_horizon()
def test_two_horizons_only()
def test_all_horizons_missing()
def test_identical_mape_across_horizons()
def test_decreasing_accuracy_pattern()
```

---

## 📊 Success Metrics

### Performance Targets
- Full horizon analysis per series: **<2 seconds**
- Decay model fitting: **<200 ms per model**
- Multi-model comparison: **<500 ms**

### Quality Targets
- Unit test coverage: **≥80%**
- R² for decay models: **>0.80** (on typical patterns)
- Model selection accuracy: **>90%** (best model chosen)

---

## 📚 Dependencies

### Required Packages
```python
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0  # Optional for advanced fitting
```

### Code Dependencies
- **Requires:** PC-060-07A (MetricsCalculator)

---

## 🔗 Related Tickets
- **Depends on:** PC-060-07A (Metrics Calculator)
- **Relates to:** PC-063-07A completes Epic-07A
- **Enables:** Epic-07B (Monitoring & Reporting)

---

## 📖 Documentation Requirements

- [ ] API documentation for all public methods
- [ ] Decay model mathematical descriptions
- [ ] R² interpretation guidelines
- [ ] Model selection methodology
- [ ] Usage examples with multi-horizon forecasts
- [ ] Visualization recommendations

---

## 💡 Implementation Notes

### Minimum Horizons for Fitting
- Linear model: ≥2 horizons
- Polynomial model: ≥4 horizons (degree 2 requires 3+ params)
- Exponential model: ≥3 horizons

### Handling Fitting Failures
Exponential model may fail if MAPE values are zero or negative:
```python
try:
    log_mapes = np.log(mapes)
    # fit exponential
except (ValueError, RuntimeWarning):
    # Skip exponential model
    pass
```

### R² Interpretation
- R² > 0.9: Excellent fit
- R² > 0.8: Good fit
- R² > 0.6: Moderate fit
- R² < 0.6: Poor fit (consider if decay pattern exists)

### Visualization Output Format
Prepare data for plotting:
```python
{
    'horizons': [0, 1, 2, ..., 8],
    'observed_mape': [2.5, 2.8, 3.1, ...],
    'linear_fit': [2.4, 2.7, 3.0, ...],
    'polynomial_fit': [2.5, 2.8, 3.2, ...],
    'exponential_fit': [2.5, 2.9, 3.4, ...]
}
```

### Model Ranking Output
```python
{
    'horizon_0': [('lgbm', 2.5), ('rf', 2.7), ('hw', 3.1)],
    'horizon_1': [('lgbm', 2.8), ('rf', 2.9), ('hw', 3.3)],
    ...
}
```

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥80% coverage
- [ ] All three decay models implemented and tested
- [ ] Model selection working correctly
- [ ] Multi-model comparison validated
- [ ] Performance benchmarks met
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] Integration tested with PC-060-07A
- [ ] Epic-07A complete and ready for Epic-07B
