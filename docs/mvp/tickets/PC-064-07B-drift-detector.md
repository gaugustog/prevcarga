# PC-064-07B: Drift Detection System

**Ticket ID:** PC-064-07B  
**Epic:** Epic-07B - Monitoring & Reporting  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Sprint:** Week 20 (Days 1-2)

---

## 📋 Description

Implement comprehensive drift detection system with multiple statistical methods (CUSUM, Kolmogorov-Smirnov, Page-Hinkley) to monitor model performance degradation over time. Include drift type classification (gradual, abrupt, cyclical), severity assessment, and automated alerting with configurable thresholds.

---

## 🎯 Acceptance Criteria

- [ ] `DriftDetector` monitors performance metrics using rolling windows
- [ ] Three detection methods implemented: CUSUM, KS test, Page-Hinkley
- [ ] Ensemble detection with majority voting across methods
- [ ] Drift classification: gradual (trend), abrupt (step), cyclical (seasonal)
- [ ] Severity levels: INFO, WARNING (5%), CRITICAL (10%), EMERGENCY (20%)
- [ ] `AlertSystem` with configurable handlers for notifications
- [ ] Baseline setting from historical performance data
- [ ] >90% accuracy in detecting 10%+ performance degradation
- [ ] Unit tests with 85%+ coverage

---

## 📐 Technical Specifications

### Core Components

#### 1. Configuration and Results
```python
class DriftType(Enum):
    GRADUAL = "gradual"
    ABRUPT = "abrupt"
    CYCLICAL = "cyclical"
    NONE = "none"

class SeverityLevel(Enum):
    INFO = "info"
    WARNING = "warning"        # 5-10% degradation
    CRITICAL = "critical"      # 10-20% degradation
    EMERGENCY = "emergency"    # >20% degradation

@dataclass
class DriftConfig:
    window_size: int = 30
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.5
    ks_alpha: float = 0.05
    ph_threshold: float = 10.0
    ph_lambda: float = 0.1
    warning_threshold: float = 0.05
    critical_threshold: float = 0.10

@dataclass
class DriftDetectionResult:
    drift_detected: bool
    drift_type: DriftType
    severity: SeverityLevel
    detection_time: str
    baseline_value: float
    current_value: float
    degradation_pct: float
    confidence: float
    method_used: str
    details: Dict[str, any]
```

#### 2. Detection Methods

**CUSUMDetector (Cumulative Sum):**
```python
class CUSUMDetector:
    def __init__(self, threshold: float = 5.0, drift: float = 0.5)
    def set_baseline(self, baseline_data: np.ndarray)
    def detect(self, value: float) -> bool
    def get_state(self) -> Dict[str, float]
    
    # Algorithm:
    # z = (value - baseline_mean) / baseline_std
    # cusum_pos = max(0, cusum_pos + z - drift)
    # cusum_neg = max(0, cusum_neg - z - drift)
    # Alert if cusum_pos > threshold or cusum_neg > threshold
```

**KSTestDetector (Kolmogorov-Smirnov):**
```python
class KSTestDetector:
    def __init__(self, alpha: float = 0.05)
    def set_baseline(self, baseline_data: np.ndarray)
    def detect(self, current_data: np.ndarray) -> tuple
    
    # Uses scipy.stats.ks_2samp to compare distributions
    # Returns (drift_detected, p_value, statistic)
```

**PageHinkleyDetector:**
```python
class PageHinkleyDetector:
    def __init__(self, threshold: float = 10.0, lambda_param: float = 0.1)
    def reset(self)
    def detect(self, value: float) -> bool
    def get_state(self) -> Dict[str, float]
    
    # Detects abrupt changes in mean
    # PH statistic = sum - min_sum
    # Alert if PH statistic > threshold
```

#### 3. DriftDetector Main Class
```python
class DriftDetector:
    def __init__(self, detection_config: DriftConfig)
    
    def set_baseline(self, historical_metrics: List[float])
    
    def detect_drift(
        self,
        current_metric: float,
        method: str = 'cusum'  # or 'ks_test', 'page_hinkley', 'all'
    ) -> DriftDetectionResult
    
    def classify_drift_type(
        self,
        performance_history: List[float]
    ) -> DriftClassification
    
    def _detect_drift_ensemble(self, current_metric: float) -> DriftDetectionResult
    def _assess_severity(self, degradation_pct: float) -> SeverityLevel
```

#### 4. Drift Classification
```python
@dataclass
class DriftClassification:
    drift_type: DriftType
    characteristics: Dict[str, any]
    recommended_action: str
    urgency: SeverityLevel

# Classification Logic:
# - GRADUAL: Linear regression p<0.05, |r|>0.7 (strong trend)
# - ABRUPT: t-test p<0.05 between first/second half, >10% change
# - CYCLICAL: Neither gradual nor abrupt patterns
```

#### 5. Alert System
```python
class AlertSystem:
    def __init__(self)
    def add_handler(self, handler: callable)
    def send_alert(self, drift_result: DriftDetectionResult)
    def _format_alert(self, drift_result: DriftDetectionResult) -> str
```

---

## 🔧 Implementation Tasks

### Task 1: Core Data Structures (2 hours)
- [ ] Implement `DriftType` and `SeverityLevel` enums
- [ ] Implement `DriftConfig` with validation
- [ ] Implement `DriftDetectionResult` dataclass
- [ ] Implement `DriftClassification` dataclass

### Task 2: CUSUM Detector (3 hours)
- [ ] Implement `CUSUMDetector` class
- [ ] Baseline statistics calculation (mean, std)
- [ ] Standardization and CUSUM update logic
- [ ] Threshold checking and state management
- [ ] Unit tests for CUSUM algorithm

### Task 3: KS Test Detector (2 hours)
- [ ] Implement `KSTestDetector` class
- [ ] Baseline distribution storage
- [ ] scipy.stats.ks_2samp integration
- [ ] Result interpretation and formatting
- [ ] Unit tests for KS test

### Task 4: Page-Hinkley Detector (3 hours)
- [ ] Implement `PageHinkleyDetector` class
- [ ] Running mean calculation
- [ ] PH statistic computation
- [ ] State reset and management
- [ ] Unit tests for Page-Hinkley

### Task 5: Main DriftDetector (4 hours)
- [ ] Implement `DriftDetector` coordinator class
- [ ] Baseline setting across all detectors
- [ ] Single-method detection logic
- [ ] Ensemble detection with majority voting
- [ ] Severity assessment based on degradation
- [ ] Integration with AlertSystem

### Task 6: Drift Classification (3 hours)
- [ ] Implement trend detection (linear regression)
- [ ] Implement abrupt change detection (t-test on halves)
- [ ] Classify drift types with characteristics
- [ ] Generate recommended actions
- [ ] Unit tests for classification logic

### Task 7: Alert System (2 hours)
- [ ] Implement `AlertSystem` class
- [ ] Handler registration mechanism
- [ ] Alert message formatting
- [ ] Alert history tracking
- [ ] Multiple handler support (logging, email, webhook)

### Task 8: Integration and Testing (5 hours)
- [ ] Integration tests with all detectors
- [ ] Ensemble method validation
- [ ] Performance testing with streaming data
- [ ] False positive/negative rate analysis
- [ ] Integration with Epic-07A MetricsCalculator

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_cusum_baseline_setting()
def test_cusum_drift_detection()
def test_cusum_state_management()
def test_ks_test_distribution_change()
def test_ks_test_no_change()
def test_page_hinkley_abrupt_change()
def test_page_hinkley_gradual_change()
def test_drift_detector_single_method()
def test_drift_detector_ensemble()
def test_severity_assessment()
def test_drift_classification_gradual()
def test_drift_classification_abrupt()
def test_drift_classification_cyclical()
def test_alert_system_handlers()
def test_alert_message_formatting()
```

### Integration Tests
```python
def test_full_drift_detection_pipeline()
def test_baseline_from_historical_data()
def test_multiple_consecutive_detections()
def test_ensemble_majority_voting()
def test_alert_triggering_on_detection()
```

### Validation Tests
```python
def test_detection_sensitivity_5_percent()
def test_detection_sensitivity_10_percent()
def test_detection_sensitivity_20_percent()
def test_false_positive_rate()
def test_false_negative_rate()
def test_detection_latency()
```

---

## 📊 Success Metrics

### Performance Targets
- Detection accuracy for 10% degradation: **>90%**
- False positive rate: **<10%**
- Detection latency: **<1 minute from drift occurrence**
- Memory usage: **<100 MB for 30-day rolling window**

### Quality Targets
- Unit test coverage: **≥85%**
- Type I error rate (false positives): **≤0.10**
- Type II error rate (false negatives): **≤0.10** for 10%+ degradation
- CUSUM detection delay: **<5 observations** for 2σ shift

---

## 📚 Dependencies

### Required Packages
```python
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0
```

### Code Dependencies
- **Requires:** Epic-07A (MetricsCalculator for metrics input)
- **Integrates with:** Logging infrastructure

---

## 🔗 Related Tickets
- **Depends on:** PC-060-07A (Metrics Calculator)
- **Blocks:** PC-066-07B (Report Generator - drift metrics)
- **Relates to:** PC-065-07B (Model Comparator)

---

## 📖 Documentation Requirements

- [ ] API documentation for all detector classes
- [ ] Statistical method descriptions and references
- [ ] Threshold tuning guidelines
- [ ] Drift type classification examples
- [ ] Alert system handler implementation guide
- [ ] Configuration parameter recommendations

---

## 💡 Implementation Notes

### CUSUM Threshold Selection
Default threshold of 5.0 provides good balance:
- Smaller threshold: More sensitive, higher false positives
- Larger threshold: Less sensitive, higher detection delay

### KS Test Sample Size
Requires minimum 10 observations in current window:
```python
if len(self.performance_history) < 10:
    return False  # Insufficient data
```

### Page-Hinkley vs CUSUM
- **Page-Hinkley:** Better for abrupt changes
- **CUSUM:** Better for gradual drift
- **Ensemble:** Combines strengths of both

### Drift Classification Logic
```python
# Gradual: Linear trend with p<0.05, |r|>0.7
slope, _, r_value, p_value, _ = stats.linregress(x, values)
if p_value < 0.05 and abs(r_value) > 0.7:
    return DriftType.GRADUAL

# Abrupt: Significant difference between halves
t_stat, t_pvalue = stats.ttest_ind(first_half, second_half)
if t_pvalue < 0.05 and change > 0.1 * baseline:
    return DriftType.ABRUPT
```

### Alert Handler Examples
```python
# Logging handler
def log_handler(message, severity):
    logger.log(severity.value.upper(), message)

# Email handler (placeholder)
def email_handler(message, severity):
    if severity in [SeverityLevel.CRITICAL, SeverityLevel.EMERGENCY]:
        send_email(to="ops@example.com", subject="Drift Alert", body=message)

alert_system.add_handler(log_handler)
alert_system.add_handler(email_handler)
```

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥85% coverage
- [ ] All three detection methods implemented and tested
- [ ] Ensemble detection working with majority voting
- [ ] Drift classification accurate for all types
- [ ] Alert system operational with multiple handlers
- [ ] Detection accuracy >90% for 10%+ degradation
- [ ] False positive rate <10%
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] Integration tested with Epic-07A
- [ ] Ready for PC-065-07B (Model Comparator)
