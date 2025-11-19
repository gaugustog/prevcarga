# PC-061-07A: Percentile Analysis Framework

**Ticket ID:** PC-061-07A  
**Epic:** Epic-07A - Core Metrics & Analysis  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Sprint:** Week 19 (Day 3)

---

## 📋 Description

Implement the percentile analysis framework (`PercentileAnalyzer`) that provides detailed distribution analysis of forecast errors including percentile calculations (P5-P95), normality testing, and multiple outlier detection methods (IQR, Z-score, modified Z-score).

---

## 🎯 Acceptance Criteria

- [ ] `PercentileAnalyzer` calculates P5, P10, P25, P50, P75, P90, P95 percentiles
- [ ] Distribution analysis with Shapiro-Wilk and Anderson-Darling tests
- [ ] Three outlier detection methods implemented: IQR, Z-score, modified Z-score
- [ ] Skewness and kurtosis calculations for distribution shape
- [ ] Visualization-ready data structures (pandas DataFrames)
- [ ] Performance: <2 seconds for full distribution analysis per series
- [ ] Unit tests with 80%+ coverage

---

## 📐 Technical Specifications

### Core Components

#### 1. DistributionAnalysis
```python
@dataclass
class DistributionAnalysis:
    percentiles: Dict[int, float]  # P5, P10, ..., P95
    mean: float
    std: float
    skewness: float
    kurtosis: float
    normality_test: Dict[str, any]
    outlier_count: int
    outlier_indices: List[int]
```

#### 2. OutlierAnalysis
```python
@dataclass
class OutlierAnalysis:
    outlier_indices: List[int]
    outlier_values: np.ndarray
    outlier_scores: np.ndarray
    threshold_used: float
    method: str  # 'iqr', 'zscore', 'modified_zscore'
```

#### 3. PercentileAnalyzer
```python
class PercentileAnalyzer:
    def __init__(self, percentiles: List[int] = [5, 10, 25, 50, 75, 90, 95])
    
    def analyze_error_distribution(
        self,
        actual: np.ndarray,
        forecast: np.ndarray
    ) -> DistributionAnalysis
    
    def detect_outliers(
        self,
        errors: np.ndarray,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> OutlierAnalysis
    
    def _test_normality(self, errors: np.ndarray) -> Dict[str, any]
    def _detect_outliers_iqr(self, errors: np.ndarray, threshold: float) -> OutlierAnalysis
    def _detect_outliers_zscore(self, errors: np.ndarray, threshold: float) -> OutlierAnalysis
    def _detect_outliers_modified_zscore(self, errors: np.ndarray, threshold: float) -> OutlierAnalysis
```

### Statistical Methods

1. **Percentile Calculation:**
   - Use `numpy.percentile()` with linear interpolation
   - Handle missing values via masking

2. **Normality Tests:**
   - Shapiro-Wilk: For n < 5000 samples
   - Anderson-Darling: For all sample sizes
   - Significance level: α = 0.05

3. **Outlier Detection Methods:**

   **a) IQR Method:**
   ```
   Q1 = 25th percentile
   Q3 = 75th percentile
   IQR = Q3 - Q1
   Lower bound = Q1 - 1.5 * IQR
   Upper bound = Q3 + 1.5 * IQR
   ```

   **b) Z-score Method:**
   ```
   z = (x - mean) / std
   Outlier if |z| > 3.0
   ```

   **c) Modified Z-score:**
   ```
   Modified z = 0.6745 * (x - median) / MAD
   where MAD = median(|x - median|)
   Outlier if |modified z| > 3.5
   ```

---

## 🔧 Implementation Tasks

### Task 1: Core Data Structures (1 hour)
- [ ] Implement `DistributionAnalysis` dataclass
- [ ] Implement `OutlierAnalysis` dataclass
- [ ] Add validation and type hints

### Task 2: Percentile Calculations (2 hours)
- [ ] Implement percentile calculation logic
- [ ] Add configurable percentile levels
- [ ] Calculate mean, std, skewness, kurtosis
- [ ] Handle NaN values properly

### Task 3: Normality Testing (2 hours)
- [ ] Implement Shapiro-Wilk test integration
- [ ] Implement Anderson-Darling test integration
- [ ] Add sample size checks (Shapiro-Wilk: n < 5000)
- [ ] Format test results for interpretation

### Task 4: IQR Outlier Detection (2 hours)
- [ ] Implement IQR calculation
- [ ] Calculate outlier bounds
- [ ] Identify outlier indices and values
- [ ] Calculate outlier scores (distance from bounds)

### Task 5: Z-score Methods (2 hours)
- [ ] Implement standard Z-score outlier detection
- [ ] Implement modified Z-score (MAD-based)
- [ ] Add threshold configuration
- [ ] Calculate outlier scores

### Task 6: Integration and Testing (3 hours)
- [ ] Write unit tests for each method
- [ ] Test with various distributions (normal, skewed, bimodal)
- [ ] Performance testing with large datasets
- [ ] Integration with MetricsCalculator outputs

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_percentile_calculation()
def test_distribution_statistics()
def test_shapiro_wilk_normal_data()
def test_shapiro_wilk_non_normal_data()
def test_anderson_darling_test()
def test_iqr_outlier_detection()
def test_zscore_outlier_detection()
def test_modified_zscore_outlier_detection()
def test_outlier_scores_calculation()
```

### Distribution Tests
```python
def test_normal_distribution_analysis()
def test_skewed_distribution_analysis()
def test_bimodal_distribution_analysis()
def test_uniform_distribution_analysis()
```

### Edge Case Tests
```python
def test_insufficient_samples()
def test_all_identical_values()
def test_single_outlier()
def test_multiple_outliers()
def test_no_outliers()
```

---

## 📊 Success Metrics

### Performance Targets
- Distribution analysis per series: **<2 seconds**
- Percentile calculation: **<100 ms**
- Outlier detection: **<500 ms**

### Quality Targets
- Unit test coverage: **≥80%**
- Outlier detection accuracy: **>90%** (validated on synthetic data)
- Normality test reliability: **Type I error ≤ 0.05**

---

## 📚 Dependencies

### Required Packages
```python
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0
```

### Code Dependencies
- **Requires:** PC-060-07A (MetricsCalculator)
- **Uses:** Error arrays from metrics calculation

---

## 🔗 Related Tickets
- **Depends on:** PC-060-07A (Metrics Calculator)
- **Blocks:** PC-064-07B (Drift Detector)
- **Relates to:** PC-066-07B (Report Generator)

---

## 📖 Documentation Requirements

- [ ] API documentation for all public methods
- [ ] Statistical method descriptions and references
- [ ] Outlier detection method comparison guide
- [ ] Interpretation guidelines for normality tests
- [ ] Usage examples with real forecast errors

---

## 💡 Implementation Notes

### Percentile Interpolation
Use linear interpolation for percentiles to ensure smooth transitions:
```python
np.percentile(data, q, interpolation='linear')
```

### Normality Test Interpretation
- **Shapiro-Wilk:** p > 0.05 suggests normal distribution
- **Anderson-Darling:** Compare statistic to critical values at different significance levels

### Outlier Method Selection
- **IQR:** Robust, works well for skewed distributions
- **Z-score:** Assumes normality, sensitive to extreme outliers
- **Modified Z-score:** More robust to outliers than standard Z-score

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥80% coverage
- [ ] All three outlier detection methods working
- [ ] Normality tests validated
- [ ] Performance benchmarks met
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] Integration tested with PC-060-07A outputs
- [ ] Ready for drift detection (PC-064-07B)
