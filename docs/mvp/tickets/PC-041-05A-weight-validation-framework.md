# PC-041-05A: Weight Validation Framework

**Ticket ID:** PC-041-05A  
**Epic:** [Epic-05A: Base Combination Strategies](../epics/Epic-05A.md)  
**User Story:** US-6  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `WeightValidationFramework` that provides comprehensive validation for combination weights and performance. Validates weight stability and reasonableness, performs statistical significance testing for combination improvements, detects weight drift, and generates combination diagnostics and performance reports.

**As a** ML engineer  
**I want** comprehensive validation for combination weights and performance  
**So that** I can ensure combination quality and detect performance degradation

---

## ✅ Acceptance Criteria

- [ ] `WeightValidationFramework` validates weight stability and reasonableness
- [ ] Performance validation against individual models and benchmarks
- [ ] Statistical significance testing for combination improvements
- [ ] Weight drift detection and alerting
- [ ] Combination diagnostics and reporting
- [ ] Bootstrapping for weight stability tests
- [ ] t-tests and Wilcoxon tests for performance
- [ ] Statistical process control for drift detection
- [ ] Diagnostic plots and performance reports
- [ ] Alert system for significant performance changes

---

## 🔧 Implementation Tasks

### 1. Create Validation Framework Module
- [ ] Create `src/models/combination/validation_framework.py`
- [ ] Import statistical test utilities (scipy.stats)
- [ ] Import visualization libraries
- [ ] Add module docstrings

### 2. Implement WeightValidationFramework Class
- [ ] Create main validation class
- [ ] Initialize with validation configuration
- [ ] Track validation history
- [ ] Define validation test suite

### 3. Implement Weight Stability Tests
- [ ] Create `test_weight_stability()` method
- [ ] Use bootstrapping to resample data
- [ ] Calculate weight variance across samples
- [ ] Test if variance exceeds threshold
- [ ] Return stability metrics

### 4. Implement Weight Reasonableness Tests
- [ ] Create `test_weight_reasonableness()` method
- [ ] Check for extreme weights (>0.8 or <0.05)
- [ ] Detect zero-weight models
- [ ] Check for near-uniform weights
- [ ] Validate weight distribution
- [ ] Return reasonableness scores

### 5. Implement Performance Significance Tests
- [ ] Create `test_performance_significance()` method
- [ ] Paired t-test vs individual models
- [ ] Wilcoxon signed-rank test
- [ ] Calculate p-values
- [ ] Determine statistical significance
- [ ] Return test results

### 6. Implement Weight Drift Detection
- [ ] Create `detect_weight_drift()` method
- [ ] Track weight history over time
- [ ] Calculate moving statistics (mean, std)
- [ ] Apply statistical process control (SPC)
- [ ] Detect out-of-control points
- [ ] Generate drift alerts

### 7. Implement Combination Diagnostics
- [ ] Create `generate_diagnostics()` method
- [ ] Calculate combination metrics
- [ ] Analyze weight patterns
- [ ] Compare to baselines
- [ ] Identify potential issues
- [ ] Return diagnostic report

### 8. Implement Bootstrap Weight Validation
- [ ] Create `_bootstrap_weights()` method
- [ ] Resample training data B times
- [ ] Optimize weights on each sample
- [ ] Calculate weight confidence intervals
- [ ] Test for stability
- [ ] Return bootstrap statistics

### 9. Implement Performance Comparison Tests
- [ ] Create `compare_performance()` method
- [ ] Calculate metrics for combination
- [ ] Calculate metrics for individual models
- [ ] Statistical tests for differences
- [ ] Calculate improvement percentages
- [ ] Return comparison results

### 10. Implement SPC Charts
- [ ] Create `_create_spc_chart()` method
- [ ] Calculate control limits (±3σ)
- [ ] Plot weight trajectories
- [ ] Highlight out-of-control points
- [ ] Save chart to file
- [ ] Return chart object

### 11. Implement Alert System
- [ ] Create `AlertSystem` class
- [ ] Define alert rules and thresholds
- [ ] Check for violations
- [ ] Generate alert messages
- [ ] Store alert history
- [ ] Support notification callbacks

### 12. Implement Validation Reports
- [ ] Create `generate_validation_report()` method
- [ ] Aggregate all validation results
- [ ] Format as structured report
- [ ] Include visualizations
- [ ] Calculate summary scores
- [ ] Save to file (HTML/PDF)

### 13. Implement Diagnostic Plots
- [ ] Create `generate_diagnostic_plots()` method
- [ ] Weight distribution plot
- [ ] Weight evolution over time
- [ ] Performance comparison plot
- [ ] Error distribution plot
- [ ] Save plots to directory

### 14. Implement Validation Suite Runner
- [ ] Create `run_validation_suite()` method
- [ ] Execute all validation tests
- [ ] Aggregate results
- [ ] Generate overall pass/fail
- [ ] Create summary report
- [ ] Return validation results

### 15. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_validation_framework.py`
- [ ] Test weight stability validation
- [ ] Test performance significance tests
- [ ] Test drift detection
- [ ] Test bootstrap validation
- [ ] Test alert generation
- [ ] Test report generation

### 16. Write Integration Tests
- [ ] Test with real combination results
- [ ] Test alert triggering
- [ ] Test report generation end-to-end
- [ ] Validate statistical tests
- [ ] Performance benchmarks

### 17. Create Usage Examples
- [ ] Create `examples/validation_framework_demo.py`
- [ ] Show validation workflow
- [ ] Show alert system usage
- [ ] Show report generation
- [ ] Demonstrate drift detection

---

## 💻 Implementation Details

### WeightValidationFramework Implementation

```python
"""Validation framework for model combinations."""
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    
    weight_stability_threshold: float = 0.1  # Max acceptable std dev
    extreme_weight_threshold: float = 0.8  # Flag weights > this
    min_weight_threshold: float = 0.05  # Flag weights < this
    significance_level: float = 0.05  # For statistical tests
    bootstrap_samples: int = 100
    drift_window_size: int = 20
    drift_sigma: float = 3.0  # Control limit in standard deviations


@dataclass
class ValidationResult:
    """Results from validation tests."""
    
    passed: bool
    test_name: str
    metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class AlertSystem:
    """Alert system for validation failures."""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
    
    def add_alert(
        self,
        severity: str,  # 'info', 'warning', 'critical'
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Add alert to system."""
        alert = {
            'timestamp': pd.Timestamp.now(),
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        self.alerts.append(alert)
        logger.log(
            {'info': 'INFO', 'warning': 'WARNING', 'critical': 'ERROR'}[severity],
            f"ALERT: {message}"
        )
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by severity."""
        if severity is None:
            return self.alerts
        return [a for a in self.alerts if a['severity'] == severity]
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []


class WeightValidationFramework:
    """
    Comprehensive validation framework for model combinations.
    
    Validates:
    - Weight stability (bootstrapping)
    - Weight reasonableness (distribution checks)
    - Performance improvement (statistical tests)
    - Weight drift over time (SPC)
    - Combination diagnostics
    
    Features:
    - Statistical significance testing
    - Automatic alert generation
    - Comprehensive reporting
    - Visualization support
    
    Example:
        >>> validator = WeightValidationFramework()
        >>> 
        >>> # Validate weights
        >>> stability_result = validator.test_weight_stability(
        ...     combiner, predictions, targets
        ... )
        >>> 
        >>> # Run full validation suite
        >>> results = validator.run_validation_suite(
        ...     combiner, predictions, targets
        ... )
        >>> 
        >>> # Generate report
        >>> validator.generate_validation_report(results, 'report.html')
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validation framework.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = {}
        
        self.config = ValidationConfig(**config)
        self.alert_system = AlertSystem()
        self.validation_history: List[Dict[str, Any]] = []
    
    def test_weight_stability(
        self,
        combiner,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame
    ) -> ValidationResult:
        """
        Test weight stability using bootstrapping.
        
        Args:
            combiner: Fitted combiner instance
            predictions: Model predictions
            targets: True targets
        
        Returns:
            ValidationResult with stability metrics
        """
        logger.info("Testing weight stability via bootstrapping")
        
        bootstrap_weights = self._bootstrap_weights(
            combiner, predictions, targets
        )
        
        # Calculate statistics
        weight_means = {
            model: np.mean([w[model] for w in bootstrap_weights])
            for model in bootstrap_weights[0].keys()
        }
        
        weight_stds = {
            model: np.std([w[model] for w in bootstrap_weights])
            for model in bootstrap_weights[0].keys()
        }
        
        # Check if any weights are unstable
        max_std = max(weight_stds.values())
        passed = max_std < self.config.weight_stability_threshold
        
        warnings = []
        for model, std in weight_stds.items():
            if std > self.config.weight_stability_threshold:
                warnings.append(
                    f"Model {model} has unstable weight (std={std:.3f})"
                )
                self.alert_system.add_alert(
                    'warning',
                    f"Unstable weight for {model}",
                    {'std': std, 'mean': weight_means[model]}
                )
        
        return ValidationResult(
            passed=passed,
            test_name='weight_stability',
            metrics={'max_std': max_std, 'mean_std': np.mean(list(weight_stds.values()))},
            warnings=warnings,
            details={'weight_means': weight_means, 'weight_stds': weight_stds}
        )
    
    def test_weight_reasonableness(
        self,
        weights: Dict[str, float]
    ) -> ValidationResult:
        """
        Test if weights are reasonable.
        
        Args:
            weights: Dictionary of model weights
        
        Returns:
            ValidationResult with reasonableness checks
        """
        logger.info("Testing weight reasonableness")
        
        warnings = []
        
        # Check for extreme weights
        for model, weight in weights.items():
            if weight > self.config.extreme_weight_threshold:
                warnings.append(f"Extreme weight for {model}: {weight:.3f}")
                self.alert_system.add_alert(
                    'warning',
                    f"Extreme weight for {model}",
                    {'weight': weight}
                )
            
            if weight < self.config.min_weight_threshold and weight > 0:
                warnings.append(f"Very low weight for {model}: {weight:.3f}")
        
        # Check for near-uniform weights
        weight_values = list(weights.values())
        weight_variance = np.var(weight_values)
        
        if weight_variance < 0.01:  # Essentially uniform
            warnings.append("Weights are nearly uniform, simple averaging may be sufficient")
        
        passed = len(warnings) == 0
        
        return ValidationResult(
            passed=passed,
            test_name='weight_reasonableness',
            metrics={'weight_variance': weight_variance},
            warnings=warnings,
            details={'weights': weights}
        )
    
    def test_performance_significance(
        self,
        combination_errors: np.ndarray,
        individual_errors: Dict[str, np.ndarray]
    ) -> ValidationResult:
        """
        Test if combination significantly outperforms individual models.
        
        Args:
            combination_errors: Errors from combination
            individual_errors: Dict of errors from individual models
        
        Returns:
            ValidationResult with significance tests
        """
        logger.info("Testing performance significance")
        
        test_results = {}
        warnings = []
        
        for model_name, model_errors in individual_errors.items():
            # Paired t-test
            t_stat, t_pvalue = stats.ttest_rel(model_errors, combination_errors)
            
            # Wilcoxon signed-rank test
            w_stat, w_pvalue = stats.wilcoxon(model_errors, combination_errors)
            
            test_results[model_name] = {
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_pvalue': w_pvalue,
                'significant': t_pvalue < self.config.significance_level
            }
            
            if t_pvalue >= self.config.significance_level:
                warnings.append(
                    f"Not significantly better than {model_name} (p={t_pvalue:.3f})"
                )
        
        # Check if combination beats at least some models significantly
        significant_improvements = sum(
            1 for r in test_results.values() if r['significant']
        )
        
        passed = significant_improvements > 0
        
        if not passed:
            self.alert_system.add_alert(
                'critical',
                "Combination not significantly better than any individual model",
                {'test_results': test_results}
            )
        
        return ValidationResult(
            passed=passed,
            test_name='performance_significance',
            metrics={'significant_improvements': significant_improvements},
            warnings=warnings,
            details={'test_results': test_results}
        )
    
    def detect_weight_drift(
        self,
        weight_history: List[Dict[str, float]]
    ) -> ValidationResult:
        """
        Detect weight drift using statistical process control.
        
        Args:
            weight_history: List of weight dictionaries over time
        
        Returns:
            ValidationResult with drift detection
        """
        logger.info("Detecting weight drift")
        
        if len(weight_history) < self.config.drift_window_size:
            return ValidationResult(
                passed=True,
                test_name='weight_drift',
                metrics={},
                warnings=['Insufficient history for drift detection'],
                details={}
            )
        
        # Convert to DataFrame for easier analysis
        weight_df = pd.DataFrame(weight_history)
        
        # Calculate control limits for each model
        drift_detected = {}
        warnings = []
        
        for model in weight_df.columns:
            weights = weight_df[model].values
            
            # Calculate moving statistics
            mean = np.mean(weights)
            std = np.std(weights)
            
            # Control limits (±3σ)
            ucl = mean + self.config.drift_sigma * std
            lcl = mean - self.config.drift_sigma * std
            
            # Check recent weights
            recent_weights = weights[-5:]  # Last 5 observations
            out_of_control = np.any((recent_weights > ucl) | (recent_weights < lcl))
            
            drift_detected[model] = out_of_control
            
            if out_of_control:
                warnings.append(f"Weight drift detected for {model}")
                self.alert_system.add_alert(
                    'warning',
                    f"Weight drift detected for {model}",
                    {
                        'mean': mean,
                        'std': std,
                        'recent_weights': recent_weights.tolist()
                    }
                )
        
        passed = not any(drift_detected.values())
        
        return ValidationResult(
            passed=passed,
            test_name='weight_drift',
            metrics={'models_with_drift': sum(drift_detected.values())},
            warnings=warnings,
            details={'drift_detected': drift_detected}
        )
    
    def run_validation_suite(
        self,
        combiner,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        weight_history: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Run complete validation suite.
        
        Args:
            combiner: Fitted combiner
            predictions: Model predictions
            targets: True targets
            weight_history: Optional weight history for drift detection
        
        Returns:
            Dictionary of validation results
        """
        logger.info("Running validation suite")
        
        results = {}
        
        # Test 1: Weight stability
        results['stability'] = self.test_weight_stability(
            combiner, predictions, targets
        )
        
        # Test 2: Weight reasonableness
        weights = combiner.get_weights()
        results['reasonableness'] = self.test_weight_reasonableness(weights)
        
        # Test 3: Performance significance
        # (Would need to calculate errors from predictions/targets)
        
        # Test 4: Weight drift (if history available)
        if weight_history:
            results['drift'] = self.detect_weight_drift(weight_history)
        
        # Overall pass/fail
        all_passed = all(r.passed for r in results.values())
        
        logger.info(f"Validation suite: {'PASSED' if all_passed else 'FAILED'}")
        
        # Store in history
        self.validation_history.append({
            'timestamp': pd.Timestamp.now(),
            'results': results,
            'passed': all_passed
        })
        
        return results
    
    def generate_validation_report(
        self,
        results: Dict[str, ValidationResult],
        output_path: str
    ):
        """
        Generate comprehensive validation report.
        
        Args:
            results: Validation results
            output_path: Path to save report
        """
        logger.info(f"Generating validation report: {output_path}")
        
        # Create HTML report
        html = ["<html><head><title>Validation Report</title>"]
        html.append("<style>")
        html.append("body { font-family: Arial; margin: 20px; }")
        html.append(".passed { color: green; }")
        html.append(".failed { color: red; }")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("</style></head><body>")
        
        html.append("<h1>Model Combination Validation Report</h1>")
        html.append(f"<p>Generated: {pd.Timestamp.now()}</p>")
        
        # Summary table
        html.append("<h2>Validation Summary</h2>")
        html.append("<table>")
        html.append("<tr><th>Test</th><th>Status</th><th>Warnings</th></tr>")
        
        for test_name, result in results.items():
            status_class = 'passed' if result.passed else 'failed'
            status_text = 'PASSED' if result.passed else 'FAILED'
            
            html.append(f"<tr>")
            html.append(f"<td>{result.test_name}</td>")
            html.append(f"<td class='{status_class}'>{status_text}</td>")
            html.append(f"<td>{len(result.warnings)}</td>")
            html.append(f"</tr>")
        
        html.append("</table>")
        
        # Detailed results
        html.append("<h2>Detailed Results</h2>")
        
        for test_name, result in results.items():
            html.append(f"<h3>{result.test_name}</h3>")
            html.append(f"<p>Status: <span class='{'passed' if result.passed else 'failed'}'>{status_text}</span></p>")
            
            if result.metrics:
                html.append("<p><strong>Metrics:</strong></p><ul>")
                for metric, value in result.metrics.items():
                    html.append(f"<li>{metric}: {value:.4f}</li>")
                html.append("</ul>")
            
            if result.warnings:
                html.append("<p><strong>Warnings:</strong></p><ul>")
                for warning in result.warnings:
                    html.append(f"<li>{warning}</li>")
                html.append("</ul>")
        
        # Alerts
        alerts = self.alert_system.get_alerts()
        if alerts:
            html.append("<h2>Alerts</h2>")
            html.append("<table>")
            html.append("<tr><th>Timestamp</th><th>Severity</th><th>Message</th></tr>")
            
            for alert in alerts:
                html.append(f"<tr>")
                html.append(f"<td>{alert['timestamp']}</td>")
                html.append(f"<td>{alert['severity']}</td>")
                html.append(f"<td>{alert['message']}</td>")
                html.append(f"</tr>")
            
            html.append("</table>")
        
        html.append("</body></html>")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(html))
        
        logger.info(f"Report saved to {output_path}")
    
    def _bootstrap_weights(
        self,
        combiner,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame
    ) -> List[Dict[str, float]]:
        """
        Bootstrap weights by resampling data.
        
        Args:
            combiner: Combiner instance
            predictions: Model predictions
            targets: True targets
        
        Returns:
            List of weight dictionaries from bootstrap samples
        """
        n_samples = len(next(iter(predictions.values())))
        bootstrap_weights = []
        
        for _ in range(self.config.bootstrap_samples):
            # Resample indices
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Resample data
            boot_predictions = {
                model: pred.iloc[indices]
                for model, pred in predictions.items()
            }
            boot_targets = targets.iloc[indices]
            
            # Fit combiner on bootstrap sample
            boot_combiner = combiner.__class__(config=combiner.config)
            boot_combiner.fit(boot_predictions, boot_targets)
            
            # Get weights
            weights = boot_combiner.get_weights()
            bootstrap_weights.append(weights)
        
        return bootstrap_weights
```

---

## 🧪 Testing & Validation

```python
"""Tests for validation framework."""
import pytest
import pandas as pd
import numpy as np

from src.models.combination.validation_framework import WeightValidationFramework
from src.models.combination.weighted_voting import WeightedVotingCombiner


@pytest.fixture
def sample_validation_data():
    """Create sample data for validation."""
    dates = pd.date_range('2024-01-01', periods=100, freq='30min')
    
    predictions = {
        'model1': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 50 + 1000,
            'pred_h1': np.random.randn(100) * 50 + 1000
        }, index=dates),
        'model2': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 50 + 1000,
            'pred_h1': np.random.randn(100) * 50 + 1000
        }, index=dates)
    }
    
    targets = pd.DataFrame({
        'target_h0': np.random.randn(100) * 30 + 1000,
        'target_h1': np.random.randn(100) * 30 + 1000
    }, index=dates)
    
    return predictions, targets


def test_weight_stability(sample_validation_data):
    """Test weight stability validation."""
    predictions, targets = sample_validation_data
    
    combiner = WeightedVotingCombiner()
    combiner.fit(predictions, targets)
    
    validator = WeightValidationFramework()
    result = validator.test_weight_stability(combiner, predictions, targets)
    
    assert result.test_name == 'weight_stability'
    assert 'max_std' in result.metrics


def test_validation_suite(sample_validation_data):
    """Test full validation suite."""
    predictions, targets = sample_validation_data
    
    combiner = WeightedVotingCombiner()
    combiner.fit(predictions, targets)
    
    validator = WeightValidationFramework()
    results = validator.run_validation_suite(combiner, predictions, targets)
    
    assert 'stability' in results
    assert 'reasonableness' in results


def test_report_generation(sample_validation_data, tmp_path):
    """Test report generation."""
    predictions, targets = sample_validation_data
    
    combiner = WeightedVotingCombiner()
    combiner.fit(predictions, targets)
    
    validator = WeightValidationFramework()
    results = validator.run_validation_suite(combiner, predictions, targets)
    
    report_path = tmp_path / "validation_report.html"
    validator.generate_validation_report(results, str(report_path))
    
    assert report_path.exists()
```

---

## 📝 Technical Notes

### Statistical Tests
- **t-test:** Parametric test for mean differences
- **Wilcoxon:** Non-parametric alternative
- **Bootstrap:** Resampling for confidence intervals

### SPC Charts
- Control limits: mean ± 3σ
- Detects systematic drift
- Triggers alerts for investigation

---

## 🔗 Dependencies

**Depends On:**
- PC-036-05A: Base Combiner Interface
- PC-038-05A: Weighted Voting Combiner

**Integrates With:**
- All combination methods

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Weight stability tests working
- [ ] Performance significance tests functional
- [ ] Weight drift detection implemented
- [ ] Alert system operational
- [ ] Validation reports generating
- [ ] Bootstrap validation working
- [ ] SPC charts created
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Previous:** [PC-040-05A: Basic Bias Correction](PC-040-05A-basic-bias-correction.md)  
**Epic:** [Epic-05A: Base Combination Strategies](../epics/Epic-05A.md)
