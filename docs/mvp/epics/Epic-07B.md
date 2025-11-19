# Epic-07B: Monitoring & Reporting

**Epic ID:** Epic-07B  
**Epic Name:** Monitoring & Reporting  
**Phase:** 7B  
**Duration:** 1 week (Week 20)  
**Dependencies:** Epic-07A (Core Metrics & Analysis)  
**Priority:** Medium  

---

## 🎯 Epic Overview

Implement advanced monitoring capabilities with drift detection and comprehensive multi-format reporting system. This epic builds on Epic-07A's metrics foundation to provide production-ready monitoring, alerting, and stakeholder reporting capabilities essential for operational deployment and ongoing system maintenance.

### Business Value
- **Proactive Monitoring:** Early detection of model drift and performance degradation
- **Automated Alerting:** Timely notifications of quality issues with severity classification
- **Stakeholder Communication:** Automated reports in multiple formats for different audiences
- **Continuous Improvement:** Data-driven insights for model selection and optimization
- **Operational Compliance:** Auditable performance tracking for regulatory requirements

---

## 📋 User Stories

### **User Story 1: Drift Detection System**
**As a** system operator  
**I want** automated detection of model performance drift and degradation  
**So that** I can take corrective action before forecast quality impacts operations  

**Acceptance Criteria:**
- [ ] `DriftDetector` monitors performance metrics over rolling windows
- [ ] Statistical tests for drift detection (CUSUM, Page-Hinkley, KS tests)
- [ ] Multiple drift types: gradual degradation, sudden shifts, concept drift
- [ ] Configurable sensitivity levels and detection thresholds
- [ ] Automated alerting system with severity levels and escalation procedures

**Technical Requirements:**
- Rolling window monitoring: configurable window sizes (daily, weekly, monthly)
- Statistical tests: CUSUM for mean shifts, Kolmogorov-Smirnov for distribution changes
- Drift classification: gradual (trend), abrupt (step change), cyclical (seasonal)
- Alert thresholds: warning (5% degradation), critical (10% degradation)
- Integration: alerting via logging, email, or monitoring systems

**Definition of Done:**
- [ ] `DriftDetector` class implemented with multiple detection methods
- [ ] Statistical drift tests working accurately
- [ ] Alert system with configurable thresholds operational
- [ ] Drift classification and severity assessment working
- [ ] Integration with monitoring infrastructure complete

---

### **User Story 2: Model Comparison Framework**
**As a** ML engineer  
**I want** systematic comparison of models with statistical significance testing  
**So that** I can make data-driven decisions about model selection and deployment  

**Acceptance Criteria:**
- [ ] `ModelComparator` compares multiple models across performance criteria
- [ ] Statistical significance testing (paired t-tests, Wilcoxon tests)
- [ ] Performance ranking with confidence intervals
- [ ] Pairwise model comparison matrices
- [ ] Visualization-ready comparison data structures

**Technical Requirements:**
- Multiple comparison criteria: MAPE, MAE, RMSE, bias, consistency
- Statistical tests: paired t-test, Wilcoxon signed-rank, Friedman test
- Bonferroni correction for multiple comparisons
- Effect size calculation (Cohen's d)
- Data structures: comparison matrices, ranking tables

**Definition of Done:**
- [ ] `ModelComparator` class implemented with statistical tests
- [ ] Pairwise comparison matrices generated correctly
- [ ] Statistical significance properly controlled (family-wise error rate)
- [ ] Performance rankings with uncertainty quantification
- [ ] Integration with Epic-07A metrics working

---

### **User Story 3: Comprehensive Reporting System**
**As a** stakeholder  
**I want** automated generation of performance reports in multiple formats  
**So that** I can share insights with technical and business audiences  

**Acceptance Criteria:**
- [ ] `ReportGenerator` creates HTML reports with interactive visualizations
- [ ] CSV exports for detailed analysis and data sharing
- [ ] PDF summaries for executive reporting and archival
- [ ] Automated report scheduling and distribution capabilities
- [ ] Customizable report templates for different stakeholder needs

**Technical Requirements:**
- HTML generation: Jinja2 templates with Plotly.js interactive charts
- CSV export: pandas to_csv with proper formatting and metadata
- PDF generation: WeasyPrint or reportlab for professional layouts
- Scheduling: cron-compatible scheduling with configurable frequencies
- Template system: modular report components for different audiences

**Definition of Done:**
- [ ] `ReportGenerator` class implemented with all output formats
- [ ] HTML reports with interactive visualizations working
- [ ] CSV and PDF generation functional
- [ ] Automated scheduling and distribution operational
- [ ] Customizable templates for different stakeholder needs

---

## 🏗️ Technical Architecture

### Core Components

```python
# Drift Detection System
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import numpy as np
from scipy import stats
from collections import deque

class DriftType(Enum):
    """Types of detected drift."""
    GRADUAL = "gradual"        # Slow degradation over time
    ABRUPT = "abrupt"          # Sudden performance shift
    CYCLICAL = "cyclical"      # Recurring pattern changes
    NONE = "none"              # No drift detected

class SeverityLevel(Enum):
    """Alert severity levels."""
    INFO = "info"              # FYI, no action needed
    WARNING = "warning"        # 5-10% degradation
    CRITICAL = "critical"      # >10% degradation
    EMERGENCY = "emergency"    # >20% degradation

@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    window_size: int = 30              # Rolling window size (days)
    cusum_threshold: float = 5.0       # CUSUM detection threshold
    cusum_drift: float = 0.5          # CUSUM drift parameter
    ks_alpha: float = 0.05            # KS test significance level
    ph_threshold: float = 10.0        # Page-Hinkley threshold
    ph_lambda: float = 0.1            # Page-Hinkley lambda
    warning_threshold: float = 0.05   # 5% degradation warning
    critical_threshold: float = 0.10  # 10% degradation critical

@dataclass
class DriftDetectionResult:
    """Results of drift detection."""
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

@dataclass
class DriftClassification:
    """Classification of detected drift."""
    drift_type: DriftType
    characteristics: Dict[str, any]
    recommended_action: str
    urgency: SeverityLevel

class CUSUMDetector:
    """Cumulative Sum (CUSUM) control chart for drift detection."""
    
    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.baseline_mean = None
        self.baseline_std = None
    
    def set_baseline(self, baseline_data: np.ndarray):
        """Set baseline statistics from historical data."""
        self.baseline_mean = np.mean(baseline_data)
        self.baseline_std = np.std(baseline_data)
    
    def detect(self, value: float) -> bool:
        """
        Detect drift for a single new observation.
        
        Returns:
            True if drift detected, False otherwise
        """
        if self.baseline_mean is None:
            raise ValueError("Baseline not set. Call set_baseline() first.")
        
        # Standardize observation
        z = (value - self.baseline_mean) / (self.baseline_std + 1e-8)
        
        # Update CUSUM statistics
        self.cusum_pos = max(0, self.cusum_pos + z - self.drift)
        self.cusum_neg = max(0, self.cusum_neg - z - self.drift)
        
        # Check thresholds
        return (self.cusum_pos > self.threshold) or (self.cusum_neg > self.threshold)
    
    def get_state(self) -> Dict[str, float]:
        """Get current CUSUM state."""
        return {
            'cusum_pos': self.cusum_pos,
            'cusum_neg': self.cusum_neg,
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std
        }

class KSTestDetector:
    """Kolmogorov-Smirnov test for distribution change detection."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.baseline_data = None
    
    def set_baseline(self, baseline_data: np.ndarray):
        """Set baseline distribution."""
        self.baseline_data = baseline_data.copy()
    
    def detect(self, current_data: np.ndarray) -> tuple:
        """
        Detect distribution drift.
        
        Returns:
            (drift_detected: bool, p_value: float, statistic: float)
        """
        if self.baseline_data is None:
            raise ValueError("Baseline not set.")
        
        statistic, p_value = stats.ks_2samp(self.baseline_data, current_data)
        drift_detected = p_value < self.alpha
        
        return drift_detected, p_value, statistic

class PageHinkleyDetector:
    """Page-Hinkley test for abrupt change detection."""
    
    def __init__(self, threshold: float = 10.0, lambda_param: float = 0.1):
        self.threshold = threshold
        self.lambda_param = lambda_param
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.sum = 0.0
        self.min_sum = 0.0
        self.mean = 0.0
        self.n = 0
    
    def detect(self, value: float) -> bool:
        """
        Detect abrupt change.
        
        Returns:
            True if change detected, False otherwise
        """
        # Update running mean
        self.n += 1
        self.mean = self.mean + (value - self.mean) / self.n
        
        # Update Page-Hinkley statistic
        self.sum += value - self.mean - self.lambda_param
        self.min_sum = min(self.min_sum, self.sum)
        
        # Check threshold
        ph_value = self.sum - self.min_sum
        return ph_value > self.threshold
    
    def get_state(self) -> Dict[str, float]:
        """Get current detector state."""
        return {
            'ph_statistic': self.sum - self.min_sum,
            'threshold': self.threshold,
            'mean': self.mean,
            'n': self.n
        }

class DriftDetector:
    """Advanced drift detection with multiple statistical methods."""
    
    def __init__(self, detection_config: DriftConfig):
        self.config = detection_config
        
        # Initialize detection methods
        self.detection_methods = {
            'cusum': CUSUMDetector(
                threshold=detection_config.cusum_threshold,
                drift=detection_config.cusum_drift
            ),
            'ks_test': KSTestDetector(alpha=detection_config.ks_alpha),
            'page_hinkley': PageHinkleyDetector(
                threshold=detection_config.ph_threshold,
                lambda_param=detection_config.ph_lambda
            )
        }
        
        self.alert_system = AlertSystem()
        self.performance_history = deque(maxlen=detection_config.window_size)
    
    def set_baseline(self, historical_metrics: List[float]):
        """Set baseline performance from historical data."""
        historical_array = np.array(historical_metrics)
        
        # Set baseline for each detector
        self.detection_methods['cusum'].set_baseline(historical_array)
        self.detection_methods['ks_test'].set_baseline(historical_array)
        
        self.baseline_mean = np.mean(historical_array)
        self.baseline_std = np.std(historical_array)
    
    def detect_drift(self, 
                    current_metric: float,
                    method: str = 'cusum') -> DriftDetectionResult:
        """
        Detect performance drift using specified statistical method.
        
        Args:
            current_metric: Current performance metric value (e.g., MAPE)
            method: Detection method ('cusum', 'ks_test', 'page_hinkley', 'all')
            
        Returns:
            DriftDetectionResult with detection details
        """
        # Add to history
        self.performance_history.append(current_metric)
        
        if method == 'all':
            # Use ensemble of all methods
            return self._detect_drift_ensemble(current_metric)
        
        # Use single method
        detector = self.detection_methods[method]
        
        if method == 'cusum':
            drift_detected = detector.detect(current_metric)
            details = detector.get_state()
        elif method == 'page_hinkley':
            drift_detected = detector.detect(current_metric)
            details = detector.get_state()
        elif method == 'ks_test':
            if len(self.performance_history) < 10:
                drift_detected = False
                details = {'error': 'Insufficient data'}
            else:
                drift_detected, p_value, statistic = detector.detect(
                    np.array(self.performance_history)
                )
                details = {'p_value': p_value, 'statistic': statistic}
        
        # Calculate degradation
        degradation_pct = (current_metric - self.baseline_mean) / self.baseline_mean
        
        # Determine severity
        severity = self._assess_severity(degradation_pct)
        
        # Classify drift type if detected
        drift_type = DriftType.NONE
        if drift_detected:
            drift_classification = self.classify_drift_type(
                list(self.performance_history)
            )
            drift_type = drift_classification.drift_type
        
        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            severity=severity,
            detection_time=pd.Timestamp.now().isoformat(),
            baseline_value=self.baseline_mean,
            current_value=current_metric,
            degradation_pct=float(degradation_pct),
            confidence=0.95 if drift_detected else 0.0,
            method_used=method,
            details=details
        )
        
        # Trigger alert if drift detected
        if drift_detected:
            self.alert_system.send_alert(result)
        
        return result
    
    def _detect_drift_ensemble(self, current_metric: float) -> DriftDetectionResult:
        """Use ensemble of all detection methods."""
        detections = []
        
        # CUSUM
        cusum_detected = self.detection_methods['cusum'].detect(current_metric)
        detections.append(cusum_detected)
        
        # Page-Hinkley
        ph_detected = self.detection_methods['page_hinkley'].detect(current_metric)
        detections.append(ph_detected)
        
        # KS test (if enough data)
        if len(self.performance_history) >= 10:
            ks_detected, _, _ = self.detection_methods['ks_test'].detect(
                np.array(self.performance_history)
            )
            detections.append(ks_detected)
        
        # Majority vote
        drift_detected = sum(detections) >= 2
        
        degradation_pct = (current_metric - self.baseline_mean) / self.baseline_mean
        severity = self._assess_severity(degradation_pct)
        
        drift_type = DriftType.NONE
        if drift_detected:
            drift_classification = self.classify_drift_type(
                list(self.performance_history)
            )
            drift_type = drift_classification.drift_type
        
        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            severity=severity,
            detection_time=pd.Timestamp.now().isoformat(),
            baseline_value=self.baseline_mean,
            current_value=current_metric,
            degradation_pct=float(degradation_pct),
            confidence=sum(detections) / len(detections),
            method_used='ensemble',
            details={'votes': {'cusum': cusum_detected, 'page_hinkley': ph_detected}}
        )
    
    def _assess_severity(self, degradation_pct: float) -> SeverityLevel:
        """Assess severity level based on degradation percentage."""
        if degradation_pct > 0.20:  # >20% degradation
            return SeverityLevel.EMERGENCY
        elif degradation_pct > self.config.critical_threshold:  # >10%
            return SeverityLevel.CRITICAL
        elif degradation_pct > self.config.warning_threshold:  # >5%
            return SeverityLevel.WARNING
        else:
            return SeverityLevel.INFO
    
    def classify_drift_type(self, 
                          performance_history: List[float]) -> DriftClassification:
        """
        Classify detected drift by type and characteristics.
        
        Args:
            performance_history: Recent performance metric values
            
        Returns:
            DriftClassification with type and recommended actions
        """
        if len(performance_history) < 10:
            return DriftClassification(
                drift_type=DriftType.NONE,
                characteristics={},
                recommended_action="Collect more data",
                urgency=SeverityLevel.INFO
            )
        
        values = np.array(performance_history)
        
        # Test for gradual trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Test for abrupt change (compare first/last halves)
        mid = len(values) // 2
        first_half_mean = np.mean(values[:mid])
        second_half_mean = np.mean(values[mid:])
        t_stat, t_pvalue = stats.ttest_ind(values[:mid], values[mid:])
        
        # Classify drift type
        if p_value < 0.05 and abs(r_value) > 0.7:  # Strong trend
            drift_type = DriftType.GRADUAL
            characteristics = {
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'trend_p_value': float(p_value)
            }
            recommended_action = "Monitor closely, consider retraining if trend continues"
            urgency = SeverityLevel.WARNING
            
        elif t_pvalue < 0.05 and abs(first_half_mean - second_half_mean) > 0.1 * first_half_mean:
            drift_type = DriftType.ABRUPT
            characteristics = {
                'first_half_mean': float(first_half_mean),
                'second_half_mean': float(second_half_mean),
                'change_magnitude': float(second_half_mean - first_half_mean),
                't_statistic': float(t_stat),
                't_p_value': float(t_pvalue)
            }
            recommended_action = "Immediate investigation required, likely retrain needed"
            urgency = SeverityLevel.CRITICAL
            
        else:
            drift_type = DriftType.CYCLICAL
            characteristics = {
                'pattern': 'Unknown cyclical pattern detected'
            }
            recommended_action = "Analyze for seasonal or operational patterns"
            urgency = SeverityLevel.WARNING
        
        return DriftClassification(
            drift_type=drift_type,
            characteristics=characteristics,
            recommended_action=recommended_action,
            urgency=urgency
        )

class AlertSystem:
    """Alert generation and distribution system."""
    
    def __init__(self):
        self.alert_handlers = []
        self.alert_history = []
    
    def add_handler(self, handler: callable):
        """Add alert handler (e.g., logging, email, webhook)."""
        self.alert_handlers.append(handler)
    
    def send_alert(self, drift_result: DriftDetectionResult):
        """Send alert through all configured handlers."""
        alert_message = self._format_alert(drift_result)
        
        # Store in history
        self.alert_history.append({
            'timestamp': drift_result.detection_time,
            'severity': drift_result.severity.value,
            'message': alert_message
        })
        
        # Send through handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_message, drift_result.severity)
            except Exception as e:
                print(f"Alert handler failed: {e}")
    
    def _format_alert(self, drift_result: DriftDetectionResult) -> str:
        """Format alert message."""
        return f"""
        [DRIFT ALERT - {drift_result.severity.value.upper()}]
        
        Drift Type: {drift_result.drift_type.value}
        Detection Method: {drift_result.method_used}
        Detection Time: {drift_result.detection_time}
        
        Performance Degradation: {drift_result.degradation_pct:.2%}
        Baseline: {drift_result.baseline_value:.4f}
        Current: {drift_result.current_value:.4f}
        
        Confidence: {drift_result.confidence:.2%}
        Details: {drift_result.details}
        """


# Model Comparison Framework
@dataclass
class ComparisonResult:
    """Results of model comparison."""
    pairwise_comparisons: Dict[tuple, Dict[str, any]]
    rankings: List[tuple]
    best_model: str
    statistical_summary: Dict[str, any]

@dataclass
class SignificanceTest:
    """Statistical significance test result."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    interpretation: str

class ModelComparator:
    """Systematic comparison and ranking of models/methods."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def compare_models(self, 
                      model_results: Dict[str, 'MetricsResult'],
                      comparison_criteria: List[str] = ['mape', 'mae', 'rmse']) -> ComparisonResult:
        """
        Compare models across multiple performance criteria.
        
        Args:
            model_results: Dict mapping model_name to MetricsResult
            comparison_criteria: List of metrics to compare
            
        Returns:
            ComparisonResult with pairwise comparisons and rankings
        """
        model_names = list(model_results.keys())
        
        # Pairwise comparisons
        pairwise = {}
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                comparison_key = (model_a, model_b)
                
                # Compare on each criterion
                criterion_results = {}
                for criterion in comparison_criteria:
                    value_a = getattr(model_results[model_a], criterion)
                    value_b = getattr(model_results[model_b], criterion)
                    
                    # Lower is better for error metrics
                    winner = model_a if value_a < value_b else model_b
                    improvement = abs(value_a - value_b) / max(value_a, value_b)
                    
                    criterion_results[criterion] = {
                        f'{model_a}_value': value_a,
                        f'{model_b}_value': value_b,
                        'winner': winner,
                        'improvement_pct': improvement * 100
                    }
                
                pairwise[comparison_key] = criterion_results
        
        # Overall rankings (by MAPE)
        rankings = sorted(
            model_results.items(),
            key=lambda x: x[1].mape
        )
        
        best_model = rankings[0][0]
        
        # Statistical summary
        mapes = [result.mape for result in model_results.values()]
        statistical_summary = {
            'mean_mape': np.mean(mapes),
            'std_mape': np.std(mapes),
            'range_mape': max(mapes) - min(mapes),
            'cv_mape': np.std(mapes) / np.mean(mapes) if np.mean(mapes) > 0 else 0
        }
        
        return ComparisonResult(
            pairwise_comparisons=pairwise,
            rankings=rankings,
            best_model=best_model,
            statistical_summary=statistical_summary
        )
    
    def statistical_significance_test(self,
                                     errors_a: np.ndarray,
                                     errors_b: np.ndarray,
                                     test_type: str = 'paired_t') -> SignificanceTest:
        """
        Test statistical significance of model differences.
        
        Args:
            errors_a: Errors from model A
            errors_b: Errors from model B
            test_type: 'paired_t', 'wilcoxon', or 'both'
            
        Returns:
            SignificanceTest with test results
        """
        # Remove NaN pairs
        mask = ~(np.isnan(errors_a) | np.isnan(errors_b))
        errors_a_clean = errors_a[mask]
        errors_b_clean = errors_b[mask]
        
        if len(errors_a_clean) < 2:
            return SignificanceTest(
                test_name='error',
                statistic=np.nan,
                p_value=np.nan,
                significant=False,
                effect_size=np.nan,
                interpretation='Insufficient data'
            )
        
        if test_type == 'paired_t':
            statistic, p_value = stats.ttest_rel(errors_a_clean, errors_b_clean)
            test_name = 'Paired t-test'
            
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(errors_a_clean, errors_b_clean)
            test_name = 'Wilcoxon signed-rank test'
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(errors_a_clean - errors_b_clean)
        pooled_std = np.sqrt((np.var(errors_a_clean) + np.var(errors_b_clean)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        significant = p_value < self.alpha
        
        # Bonferroni correction if needed (optional, can be applied externally)
        interpretation = self._interpret_results(p_value, cohens_d, significant)
        
        return SignificanceTest(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            significant=significant,
            effect_size=float(cohens_d),
            interpretation=interpretation
        )
    
    def _interpret_results(self, p_value: float, cohens_d: float, significant: bool) -> str:
        """Interpret statistical test results."""
        if not significant:
            return "No statistically significant difference detected"
        
        effect_magnitude = (
            "small" if abs(cohens_d) < 0.5
            else "medium" if abs(cohens_d) < 0.8
            else "large"
        )
        
        direction = "worse" if cohens_d > 0 else "better"
        
        return f"Statistically significant difference (p={p_value:.4f}), {effect_magnitude} effect size (d={cohens_d:.3f}), Model B performs {direction}"


# Reporting System
@dataclass
class ReportConfig:
    """Configuration for report generation."""
    template_dir: str = "templates"
    output_dir: str = "reports"
    include_charts: bool = True
    chart_format: str = "png"  # or "svg"
    report_title: str = "Forecast Performance Report"
    author: str = "PrevCarga System"

class ReportGenerator:
    """Multi-format report generation with visualization."""
    
    def __init__(self, template_config: ReportConfig):
        self.config = template_config
        self.template_engine = self._setup_jinja2()
        self.chart_generator = ChartGenerator()
    
    def _setup_jinja2(self):
        """Setup Jinja2 template environment."""
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        
        return Environment(
            loader=FileSystemLoader(self.config.template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
    
    def generate_html_report(self,
                           evaluation_results: Dict[str, any],
                           template_name: str = 'standard') -> str:
        """
        Generate interactive HTML report.
        
        Args:
            evaluation_results: Dictionary with metrics, analyses, comparisons
            template_name: Name of template to use
            
        Returns:
            Path to generated HTML file
        """
        # Generate charts
        charts = {}
        if self.config.include_charts:
            charts = self._generate_report_charts(evaluation_results)
        
        # Render template
        template = self.template_engine.get_template(f'{template_name}.html')
        html_content = template.render(
            title=self.config.report_title,
            author=self.config.author,
            generation_time=pd.Timestamp.now().isoformat(),
            results=evaluation_results,
            charts=charts
        )
        
        # Save to file
        output_path = f"{self.config.output_dir}/report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def export_csv_data(self,
                       evaluation_results: Dict[str, any],
                       output_path: str) -> None:
        """
        Export detailed metrics to CSV format.
        
        Args:
            evaluation_results: Dictionary with metrics data
            output_path: Path for CSV output
        """
        # Convert results to DataFrame
        records = []
        
        for series_id, metrics in evaluation_results.get('metrics', {}).items():
            record = {
                'series_id': series_id,
                'mape': metrics.mape,
                'mae': metrics.mae,
                'rmse': metrics.rmse,
                'r2': metrics.r2,
                'sample_size': metrics.sample_size
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Add metadata
        metadata = pd.DataFrame([{
            'report_title': self.config.report_title,
            'generation_time': pd.Timestamp.now().isoformat(),
            'author': self.config.author
        }])
        
        # Save with metadata
        with open(output_path, 'w') as f:
            f.write(f"# {self.config.report_title}\n")
            f.write(f"# Generated: {pd.Timestamp.now().isoformat()}\n\n")
            df.to_csv(f, index=False)
    
    def generate_pdf_summary(self,
                           evaluation_results: Dict[str, any],
                           output_path: str) -> None:
        """
        Generate executive PDF summary.
        
        Args:
            evaluation_results: Dictionary with summary metrics
            output_path: Path for PDF output
        """
        # This is a placeholder - actual implementation would use
        # WeasyPrint or reportlab for PDF generation
        
        # For now, generate HTML and note conversion needed
        html_path = self.generate_html_report(
            evaluation_results, 
            template_name='executive_summary'
        )
        
        print(f"HTML summary generated at: {html_path}")
        print(f"PDF conversion to {output_path} requires WeasyPrint or reportlab")
        
        # Actual PDF generation would be:
        # from weasyprint import HTML
        # HTML(html_path).write_pdf(output_path)
    
    def _generate_report_charts(self, evaluation_results: Dict[str, any]) -> Dict[str, str]:
        """Generate all charts for the report."""
        charts = {}
        
        # Performance by series chart
        if 'metrics' in evaluation_results:
            charts['performance_by_series'] = self.chart_generator.create_performance_chart(
                evaluation_results['metrics']
            )
        
        # Horizon decay chart
        if 'horizon_analysis' in evaluation_results:
            charts['horizon_decay'] = self.chart_generator.create_horizon_chart(
                evaluation_results['horizon_analysis']
            )
        
        # Time period comparison
        if 'period_analysis' in evaluation_results:
            charts['period_comparison'] = self.chart_generator.create_period_chart(
                evaluation_results['period_analysis']
            )
        
        return charts

class ChartGenerator:
    """Generate visualizations for reports."""
    
    def create_performance_chart(self, metrics: Dict[str, 'MetricsResult']) -> str:
        """Create performance comparison chart (returns HTML/PNG path)."""
        # Placeholder - would use plotly or matplotlib
        return "<div>Performance Chart Placeholder</div>"
    
    def create_horizon_chart(self, horizon_analysis: Dict) -> str:
        """Create horizon decay chart."""
        return "<div>Horizon Chart Placeholder</div>"
    
    def create_period_chart(self, period_analysis: Dict) -> str:
        """Create time period comparison chart."""
        return "<div>Period Chart Placeholder</div>"
```

### Integration with Epic-07A

```python
# Complete evaluation pipeline
from epic_07a.metrics_calculator import MetricsCalculator, MetricsConfig
from epic_07a.horizon_analyzer import HorizonAnalyzer
from epic_07a.time_period_analyzer import TimePeriodAnalyzer

# Initialize all components
metrics_calculator = MetricsCalculator(MetricsConfig())
horizon_analyzer = HorizonAnalyzer()
period_analyzer = TimePeriodAnalyzer(holiday_calendar)
drift_detector = DriftDetector(DriftConfig())
model_comparator = ModelComparator()
report_generator = ReportGenerator(ReportConfig())

# Set drift detection baseline
historical_mapes = [calculate historical performance]
drift_detector.set_baseline(historical_mapes)

# Calculate current metrics
current_metrics = metrics_calculator.calculate_metrics(predictions, actuals)

# Check for drift
for series_id, metrics in current_metrics.items():
    drift_result = drift_detector.detect_drift(metrics.mape, method='ensemble')
    
    if drift_result.drift_detected:
        print(f"Drift detected for {series_id}: {drift_result}")

# Compare models
comparison_result = model_comparator.compare_models(
    model_results=current_metrics,
    comparison_criteria=['mape', 'mae', 'rmse']
)

# Generate comprehensive report
evaluation_results = {
    'metrics': current_metrics,
    'horizon_analysis': horizon_analyzer.analyze_horizon_performance(...),
    'period_analysis': period_analyzer.analyze_by_periods(...),
    'model_comparison': comparison_result,
    'drift_detection': drift_detector.performance_history
}

report_path = report_generator.generate_html_report(evaluation_results)
report_generator.export_csv_data(evaluation_results, 'metrics.csv')
```

---

## 🔧 Implementation Plan

### Week 1: Monitoring and Reporting (Days 1-5)

- **Day 1:** Implement drift detection algorithms
  - CUSUMDetector class with state management
  - KSTestDetector for distribution changes
  - PageHinkleyDetector for abrupt changes
  - DriftDetector ensemble coordinator
  
- **Day 2:** Implement drift classification and alerting
  - Drift type classification (gradual, abrupt, cyclical)
  - Severity assessment and thresholds
  - AlertSystem with multiple handlers
  - Integration with logging infrastructure
  
- **Day 3:** Implement model comparison framework
  - ModelComparator with pairwise comparisons
  - Statistical significance testing (t-test, Wilcoxon)
  - Effect size calculation (Cohen's d)
  - Bonferroni correction for multiple comparisons
  
- **Day 4:** Implement reporting system
  - ReportGenerator with Jinja2 templates
  - HTML report generation with interactive charts
  - CSV export with proper formatting
  - PDF summary generation (basic implementation)
  
- **Day 5:** Integration testing and optimization
  - End-to-end testing with Epic-07A components
  - Performance optimization for large datasets
  - Alert system testing with various scenarios
  - Report generation performance tuning

---

## 📊 Success Metrics

### Performance Targets
- **Drift Detection Accuracy:** >90% accuracy in detecting 10% performance degradation
- **Report Generation Speed:** Complete HTML report in <30 seconds
- **Alert Latency:** Drift alerts triggered within 1 minute of detection
- **Memory Efficiency:** Peak memory usage <500MB for monitoring system

### Quality Gates
- [ ] Drift detection correctly identifies degradation with high sensitivity
- [ ] False positive rate for drift detection <10%
- [ ] Statistical tests properly control Type I error (α = 0.05)
- [ ] Reports are readable, actionable, and properly formatted
- [ ] Alert system delivers notifications reliably

### Acceptance Criteria
- [ ] Integration with Epic-07A metrics system successful
- [ ] Drift detection working with multiple statistical methods
- [ ] Model comparison framework provides statistical rigor
- [ ] Multi-format reporting operational (HTML, CSV, PDF)
- [ ] All 3 user stories completed with comprehensive testing
- [ ] Ready for Epic-08 (Orchestration & Workflows)

---

## 🧪 Testing Strategy

### Unit Tests
- Drift detection algorithm correctness (CUSUM, KS, Page-Hinkley)
- Statistical test implementations (t-test, Wilcoxon)
- Alert system notification delivery
- Report generation for all formats
- Template rendering accuracy

### Integration Tests
- End-to-end drift detection pipeline
- Integration with Epic-07A metrics outputs
- Alert system with multiple handlers
- Report generation with real evaluation data
- Multi-format export consistency

### Performance Tests
- Drift detection performance with streaming data
- Report generation speed benchmarks
- Memory usage during continuous monitoring
- Alert system throughput

### Validation Tests
- Drift detection sensitivity analysis
- False positive/negative rate assessment
- Statistical test power analysis
- Report accuracy verification

---

## 📚 Dependencies & Risks

### External Dependencies
- **Epic-07A Completion:** Core metrics and analysis system
- **Historical Data:** Baseline performance for drift detection
- **Template Files:** Jinja2 templates for report generation
- **Monitoring Infrastructure:** Integration points for alerts

### Python Dependencies
```
numpy >= 1.24.0              # Core numerical operations
pandas >= 2.0.0              # Data structures
scipy >= 1.10.0              # Statistical tests
jinja2 >= 3.1.0              # Template engine for reports
plotly >= 5.14.0             # Interactive visualizations (optional)
weasyprint >= 59.0           # PDF generation (optional)
```

### Technical Risks & Mitigation
1. **Drift Detection Sensitivity:** Tune thresholds via validation, provide configurable parameters
2. **Alert Fatigue:** Implement severity levels and aggregation, configurable thresholds
3. **Report Generation Performance:** Template caching, lazy chart generation
4. **PDF Generation Complexity:** Provide HTML as primary format, PDF as optional

### Business Risks & Mitigation
1. **False Alerts:** Rigorous validation, adjustable sensitivity, ensemble methods
2. **Report Complexity:** Multiple report templates for different audiences
3. **Integration Challenges:** Well-defined interfaces, comprehensive documentation

---

## 🔄 Handoff Criteria

### Deliverables for Epic-08
- [ ] Drift detection system operational and tested
- [ ] Alert system integrated with monitoring infrastructure
- [ ] Model comparison framework providing statistical rigor
- [ ] Multi-format reporting system functional
- [ ] All components documented and ready for orchestration

### Documentation Requirements
- [ ] Drift detection methodology and threshold tuning guide
- [ ] Alert system configuration and handler setup
- [ ] Report template customization guide
- [ ] Statistical test descriptions and interpretation
- [ ] Integration guide for operational monitoring

---

## 📈 Success Definition

Epic-07B is successful when:
1. **Drift detection system** reliably identifies performance degradation with >90% accuracy
2. **Alert system** provides timely notifications with appropriate severity levels
3. **Model comparison framework** enables statistically rigorous model selection
4. **Reporting system** generates actionable insights in multiple formats
5. **Performance benchmarks** consistently met (reports in <30s, alerts in <1min)
6. **Integration** with Epic-07A seamless and robust
7. **Foundation established** for Epic-08 operational orchestration and workflows

**Ready for Epic-08 when:** Monitoring system operational, drift detection validated, reporting automated, alert system integrated, and all acceptance criteria met with comprehensive testing.
