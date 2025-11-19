# PC-065-07B: Model Comparison Framework

**Ticket ID:** PC-065-07B  
**Epic:** Epic-07B - Monitoring & Reporting  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Sprint:** Week 20 (Day 3)

---

## 📋 Description

Implement systematic model comparison framework with statistical significance testing to enable data-driven model selection. Includes pairwise model comparisons, performance rankings with confidence intervals, statistical tests (paired t-test, Wilcoxon), and effect size calculations (Cohen's d) with Bonferroni correction for multiple comparisons.

---

## 🎯 Acceptance Criteria

- [ ] `ModelComparator` compares models across multiple criteria (MAPE, MAE, RMSE)
- [ ] Pairwise comparison matrices generated for all model pairs
- [ ] Statistical significance testing: paired t-test and Wilcoxon signed-rank
- [ ] Effect size calculation using Cohen's d
- [ ] Bonferroni correction for family-wise error rate control
- [ ] Performance rankings with winner identification
- [ ] Visualization-ready data structures for comparison charts
- [ ] Unit tests with 80%+ coverage

---

## 📐 Technical Specifications

### Core Components

#### 1. Comparison Results
```python
@dataclass
class ComparisonResult:
    pairwise_comparisons: Dict[tuple, Dict[str, any]]
    rankings: List[tuple]  # [(model_name, MetricsResult), ...]
    best_model: str
    statistical_summary: Dict[str, any]

@dataclass
class SignificanceTest:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float  # Cohen's d
    interpretation: str
```

#### 2. ModelComparator
```python
class ModelComparator:
    def __init__(self, alpha: float = 0.05)
    
    def compare_models(
        self,
        model_results: Dict[str, MetricsResult],
        comparison_criteria: List[str] = ['mape', 'mae', 'rmse']
    ) -> ComparisonResult
    
    def statistical_significance_test(
        self,
        errors_a: np.ndarray,
        errors_b: np.ndarray,
        test_type: str = 'paired_t'  # or 'wilcoxon'
    ) -> SignificanceTest
    
    def _interpret_results(
        self,
        p_value: float,
        cohens_d: float,
        significant: bool
    ) -> str
```

### Statistical Methods

#### 1. Pairwise Comparisons
For each pair of models (A, B), compare on each criterion:
```python
# For each criterion (MAPE, MAE, RMSE):
value_a = model_a.mape
value_b = model_b.mape

# Lower is better for error metrics
winner = 'model_a' if value_a < value_b else 'model_b'
improvement = abs(value_a - value_b) / max(value_a, value_b)

result = {
    'model_a_value': value_a,
    'model_b_value': value_b,
    'winner': winner,
    'improvement_pct': improvement * 100
}
```

#### 2. Paired T-Test
```python
# Compare errors from two models on same test set
from scipy import stats

t_statistic, p_value = stats.ttest_rel(errors_a, errors_b)

# Reject null hypothesis if p_value < alpha
significant = p_value < alpha
```

#### 3. Wilcoxon Signed-Rank Test (Non-parametric)
```python
# Non-parametric alternative to t-test
statistic, p_value = stats.wilcoxon(errors_a, errors_b)

# Use when normality assumption questionable
```

#### 4. Cohen's d Effect Size
```python
mean_diff = np.mean(errors_a - errors_b)
pooled_std = np.sqrt((np.var(errors_a) + np.var(errors_b)) / 2)
cohens_d = mean_diff / pooled_std

# Interpretation:
# |d| < 0.5: small effect
# 0.5 ≤ |d| < 0.8: medium effect
# |d| ≥ 0.8: large effect
```

#### 5. Bonferroni Correction
```python
# For k comparisons, use adjusted alpha
alpha_bonferroni = alpha / k

# Example: 3 models → 3 pairwise comparisons
# alpha = 0.05 → alpha_bonferroni = 0.05 / 3 = 0.0167
```

---

## 🔧 Implementation Tasks

### Task 1: Core Data Structures (1 hour)
- [ ] Implement `ComparisonResult` dataclass
- [ ] Implement `SignificanceTest` dataclass
- [ ] Add validation and type hints

### Task 2: Pairwise Comparison Logic (3 hours)
- [ ] Implement pairwise model comparison
- [ ] Compare across multiple criteria (MAPE, MAE, RMSE)
- [ ] Calculate improvement percentages
- [ ] Identify winner for each criterion
- [ ] Store results in structured dictionary

### Task 3: Statistical Significance Tests (3 hours)
- [ ] Implement paired t-test using scipy.stats
- [ ] Implement Wilcoxon signed-rank test
- [ ] Add NaN pair handling and cleaning
- [ ] Calculate test statistics and p-values
- [ ] Add sample size validation

### Task 4: Effect Size Calculation (2 hours)
- [ ] Implement Cohen's d calculation
- [ ] Handle edge cases (zero variance)
- [ ] Add interpretation of effect size
- [ ] Document effect size thresholds

### Task 5: Result Interpretation (2 hours)
- [ ] Implement result interpretation logic
- [ ] Generate human-readable interpretations
- [ ] Include statistical significance and effect size
- [ ] Add direction of difference (better/worse)

### Task 6: Model Ranking (2 hours)
- [ ] Implement ranking by primary criterion (MAPE)
- [ ] Sort models by performance
- [ ] Identify best model
- [ ] Calculate statistical summary (mean, std, range, CV)

### Task 7: Testing and Validation (3 hours)
- [ ] Unit tests for each statistical test
- [ ] Test pairwise comparison logic
- [ ] Test ranking algorithm
- [ ] Validate Bonferroni correction
- [ ] Integration tests with MetricsCalculator

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_pairwise_comparison_basic()
def test_pairwise_comparison_multiple_criteria()
def test_paired_ttest_significant()
def test_paired_ttest_not_significant()
def test_wilcoxon_test()
def test_cohens_d_calculation()
def test_cohens_d_small_effect()
def test_cohens_d_medium_effect()
def test_cohens_d_large_effect()
def test_result_interpretation()
def test_model_ranking()
def test_best_model_identification()
def test_statistical_summary()
```

### Edge Case Tests
```python
def test_identical_models()
def test_single_model()
def test_insufficient_samples()
def test_nan_handling()
def test_zero_variance()
```

### Integration Tests
```python
def test_full_comparison_pipeline()
def test_integration_with_metrics_calculator()
def test_multiple_models_comparison()
def test_bonferroni_correction_application()
```

---

## 📊 Success Metrics

### Performance Targets
- Comparison computation: **<1 second for 5 models**
- Statistical tests: **<200 ms per test**
- Memory usage: **<50 MB for full comparison**

### Quality Targets
- Unit test coverage: **≥80%**
- Type I error control: **α ≤ 0.05** (with Bonferroni)
- Statistical power: **>80%** for detecting 10% differences

---

## 📚 Dependencies

### Required Packages
```python
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0
```

### Code Dependencies
- **Requires:** PC-060-07A (MetricsCalculator - MetricsResult)
- **Uses:** Error arrays from predictions vs actuals

---

## 🔗 Related Tickets
- **Depends on:** PC-060-07A (Metrics Calculator)
- **Relates to:** PC-064-07B (Drift Detector)
- **Blocks:** PC-066-07B (Report Generator - comparison sections)

---

## 📖 Documentation Requirements

- [ ] API documentation for ModelComparator
- [ ] Statistical test descriptions and assumptions
- [ ] Effect size interpretation guidelines
- [ ] Bonferroni correction explanation
- [ ] Usage examples with multiple models
- [ ] Model selection decision flowchart

---

## 💡 Implementation Notes

### When to Use Each Test

**Paired T-Test:**
- Use when: Errors are approximately normally distributed
- Assumes: Normality of differences, paired observations
- More powerful when assumptions met

**Wilcoxon Signed-Rank:**
- Use when: Normality assumption questionable
- Non-parametric: No distribution assumptions
- More robust to outliers

**Recommendation:** Provide both tests, let user choose or default to t-test

### Bonferroni Correction Example
```python
# 5 models → 10 pairwise comparisons (combination of 5 choose 2)
n_models = 5
n_comparisons = n_models * (n_models - 1) // 2  # 10

alpha = 0.05
alpha_bonferroni = alpha / n_comparisons  # 0.005

# Use alpha_bonferroni for each test to control family-wise error rate
```

### Cohen's d Interpretation
```python
def interpret_effect_size(cohens_d: float) -> str:
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
```

### Example Output Structure
```python
{
    'pairwise_comparisons': {
        ('lgbm', 'random_forest'): {
            'mape': {
                'lgbm_value': 2.5,
                'random_forest_value': 2.8,
                'winner': 'lgbm',
                'improvement_pct': 10.7
            },
            'mae': {...},
            'rmse': {...}
        },
        ('lgbm', 'holt_winters'): {...}
    },
    'rankings': [
        ('lgbm', MetricsResult(mape=2.5, ...)),
        ('random_forest', MetricsResult(mape=2.8, ...)),
        ('holt_winters', MetricsResult(mape=3.1, ...))
    ],
    'best_model': 'lgbm',
    'statistical_summary': {
        'mean_mape': 2.8,
        'std_mape': 0.25,
        'range_mape': 0.6,
        'cv_mape': 0.089
    }
}
```

### Statistical Interpretation Template
```python
if not significant:
    return "No statistically significant difference detected"

effect = "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
direction = "worse" if cohens_d > 0 else "better"

return f"Statistically significant difference (p={p_value:.4f}), {effect} effect size (d={cohens_d:.3f}), Model B performs {direction}"
```

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥80% coverage
- [ ] Both statistical tests (t-test, Wilcoxon) implemented
- [ ] Effect size calculation working
- [ ] Bonferroni correction applied correctly
- [ ] Model ranking accurate
- [ ] Result interpretation clear and actionable
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] Integration tested with PC-060-07A
- [ ] Ready for PC-066-07B (Report Generator)
