# PC-062-07A: Time Period Performance Analyzer

**Ticket ID:** PC-062-07A  
**Epic:** Epic-07A - Core Metrics & Analysis  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Sprint:** Week 19 (Day 4)

---

## 📋 Description

Implement the time period performance analyzer (`TimePeriodAnalyzer`) that segments model performance by operational time periods including peak/off-peak hours, seasons (summer/winter/transition), day types (weekday/weekend/holiday), and provides statistical comparison across periods using ANOVA and t-tests.

---

## 🎯 Acceptance Criteria

- [ ] `PeriodClassifier` categorizes timestamps into peak/off-peak periods
- [ ] Seasonal classification: summer (Dec-Mar), winter (Jun-Sep), transition
- [ ] Day type classification: weekday, weekend, holiday
- [ ] Integration with Epic-01 holiday calendar
- [ ] `TimePeriodAnalyzer` calculates metrics for each period type
- [ ] Statistical comparison (ANOVA, t-tests) between periods
- [ ] Multi-dimensional analysis: period × area × horizon
- [ ] Performance: <3 seconds for full period analysis

---

## 📐 Technical Specifications

### Core Components

#### 1. PeriodAnalysis
```python
@dataclass
class PeriodAnalysis:
    period_metrics: Dict[str, MetricsResult]
    period_comparison: Dict[str, any]
    statistical_tests: Dict[str, any]
```

#### 2. PeriodClassifier
```python
class PeriodClassifier:
    def __init__(self):
        self.peak_hours = [(10, 12), (18, 22)]
    
    def classify_period(self, timestamp: pd.Timestamp) -> str
    def classify_season(self, timestamp: pd.Timestamp) -> str
    def classify_day_type(self, timestamp: pd.Timestamp, holiday_calendar: Optional[any]) -> str
```

#### 3. TimePeriodAnalyzer
```python
class TimePeriodAnalyzer:
    def __init__(self, holiday_calendar: Optional[any] = None)
    
    def analyze_by_periods(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray],
        timestamps: pd.DatetimeIndex
    ) -> Dict[str, PeriodAnalysis]
    
    def _compare_periods(self, period_metrics: Dict[str, MetricsResult]) -> Dict[str, any]
    def _perform_statistical_tests(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        peak_mask: np.ndarray
    ) -> Dict[str, any]
```

### Period Definitions

1. **Peak/Off-Peak Hours:**
   - Peak: 10:00-12:00, 18:00-22:00
   - Off-peak: All other hours

2. **Seasons (Southern Hemisphere):**
   - Summer: December, January, February, March
   - Winter: June, July, August, September
   - Transition: April, May, October, November

3. **Day Types:**
   - Weekday: Monday-Friday (not holiday)
   - Weekend: Saturday-Sunday
   - Holiday: From holiday calendar (Epic-01)

### Statistical Comparisons

1. **Peak vs Off-Peak:**
   ```
   Difference = MAPE_peak - MAPE_off_peak
   Relative difference (%) = 100 * Difference / MAPE_off_peak
   Statistical test: Independent t-test
   ```

2. **Seasonal Comparison:**
   ```
   ANOVA F-test across three seasons
   Post-hoc pairwise t-tests with Bonferroni correction
   ```

3. **Day Type Comparison:**
   ```
   ANOVA F-test: weekday vs weekend vs holiday
   ```

---

## 🔧 Implementation Tasks

### Task 1: Period Classification Logic (3 hours)
- [ ] Implement `PeriodClassifier` class
- [ ] Define peak hour ranges (configurable)
- [ ] Implement seasonal classification (Southern Hemisphere)
- [ ] Implement day type classification
- [ ] Add configuration for custom period definitions

### Task 2: Holiday Calendar Integration (2 hours)
- [ ] Integrate with Epic-01 holiday calendar
- [ ] Handle missing holiday calendar gracefully
- [ ] Add holiday detection logic
- [ ] Test with Brazilian holidays

### Task 3: Period Metrics Calculation (3 hours)
- [ ] Implement period-wise data filtering
- [ ] Calculate metrics for each period using MetricsCalculator
- [ ] Handle periods with insufficient data
- [ ] Create multi-dimensional results structure

### Task 4: Statistical Comparison (3 hours)
- [ ] Implement peak vs off-peak comparison
- [ ] Calculate relative differences
- [ ] Implement seasonal ANOVA
- [ ] Add t-tests for pairwise comparisons
- [ ] Format comparison results

### Task 5: Multi-dimensional Analysis (2 hours)
- [ ] Implement area × period analysis
- [ ] Implement horizon × period analysis
- [ ] Create summary statistics
- [ ] Identify best/worst performing periods

### Task 6: Testing and Integration (3 hours)
- [ ] Unit tests for period classification
- [ ] Integration tests with MetricsCalculator
- [ ] Statistical test validation
- [ ] Performance testing with full dataset

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_peak_period_classification()
def test_off_peak_period_classification()
def test_summer_season_classification()
def test_winter_season_classification()
def test_transition_season_classification()
def test_weekday_classification()
def test_weekend_classification()
def test_holiday_classification()
def test_period_metrics_calculation()
def test_peak_offpeak_comparison()
```

### Integration Tests
```python
def test_holiday_calendar_integration()
def test_metrics_calculator_integration()
def test_full_period_analysis()
def test_multidimensional_analysis()
```

### Statistical Tests
```python
def test_ttest_significance()
def test_anova_multiple_periods()
def test_insufficient_samples_handling()
```

---

## 📊 Success Metrics

### Performance Targets
- Full period analysis (26 series): **<3 seconds**
- Period classification: **<100 ms for 1000 timestamps**
- Statistical tests: **<500 ms per comparison**

### Quality Targets
- Unit test coverage: **≥80%**
- Period classification accuracy: **100%** (deterministic)
- Statistical test Type I error: **≤0.05**

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
- **Requires:** Epic-01 (Holiday Calendar)

---

## 🔗 Related Tickets
- **Depends on:** PC-060-07A (Metrics Calculator)
- **Depends on:** Epic-01 (Data Infrastructure - Holiday Calendar)
- **Relates to:** PC-066-07B (Report Generator)

---

## 📖 Documentation Requirements

- [ ] API documentation for all public classes
- [ ] Period definition documentation
- [ ] Statistical comparison methodology
- [ ] Holiday calendar integration guide
- [ ] Usage examples with operational insights

---

## 💡 Implementation Notes

### Peak Hour Configuration
Allow configurable peak hours for different regions:
```python
config = {
    'peak_hours': [(10, 12), (18, 22)],
    'custom_periods': {
        'super_peak': [(19, 21)]
    }
}
```

### Seasonal Adjustment
Consider regional differences in seasonal patterns:
- Southern Hemisphere: Summer in December-March
- Can be extended for other regions if needed

### Statistical Interpretation
Provide clear interpretation of statistical tests:
- p < 0.05: Statistically significant difference
- Effect size: Cohen's d for practical significance

### Handling Insufficient Data
Skip periods with <10 samples and report in metadata:
```python
if len(period_data) < 10:
    warnings.warn(f"Insufficient data for {period_name}")
    continue
```

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥80% coverage
- [ ] Holiday calendar integration working
- [ ] Statistical comparisons validated
- [ ] Performance benchmarks met
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] Integration tested with Epic-01 and PC-060-07A
- [ ] Operational insights validated with stakeholders
