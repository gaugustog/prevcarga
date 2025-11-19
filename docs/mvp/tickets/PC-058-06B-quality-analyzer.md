# PC-058-06B: Reconciliation Quality Analyzer

**Ticket ID:** PC-058-06B  
**Epic:** [Epic-06B: Advanced Reconciliation Methods](../epics/Epic-06B.md)  
**User Story:** US-5  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `ReconciliationQualityAnalyzer` that provides comprehensive quality metrics for reconciliation methods including constraint satisfaction, forecast accuracy improvements, consistency measures, quality degradation detection with statistical testing, comparative analysis across methods, and automated reporting with actionable insights.

**As a** system analyst  
**I want** comprehensive quality metrics for reconciliation  
**So that** I can monitor and validate performance over time

---

## ✅ Acceptance Criteria

- [ ] Comprehensive quality metric calculation
- [ ] Constraint satisfaction metrics
- [ ] Accuracy improvement metrics
- [ ] Consistency measures
- [ ] Statistical significance testing
- [ ] Quality degradation detection
- [ ] Comparative analysis across methods
- [ ] Automated reporting
- [ ] Actionable insights
- [ ] Real-time quality monitoring

---

## 💻 Implementation (Condensed)

```python
"""Quality analysis for reconciliation methods."""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy import stats


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    constraint_satisfaction: float
    accuracy_improvement: float
    consistency_score: float
    statistical_significance: bool
    degradation_detected: bool
    recommendations: List[str]


class ReconciliationQualityAnalyzer:
    """
    Comprehensive quality analysis for reconciliation.
    
    Metrics:
        - Constraint Satisfaction: How well aggregation rules satisfied
        - Accuracy: Improvement over base forecasts (MAPE, MAE, RMSE)
        - Consistency: Temporal stability, correlation preservation
        - Statistical: Hypothesis tests for performance differences
    
    Example:
        >>> analyzer = ReconciliationQualityAnalyzer()
        >>> report = analyzer.analyze_quality(
        ...     reconciled_forecasts=result.reconciled_forecasts,
        ...     base_forecasts=base_forecasts,
        ...     targets=actual_values,
        ...     hierarchy=hierarchy
        ... )
        >>> print(f"Constraint satisfaction: {report.constraint_satisfaction:.2%}")
        >>> print(f"Accuracy improvement: {report.accuracy_improvement:.2%}")
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.quality_history = []
    
    def analyze_quality(
        self,
        reconciled_forecasts: Dict[str, np.ndarray],
        base_forecasts: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        hierarchy: 'HierarchyDefinition'
    ) -> QualityReport:
        """Analyze reconciliation quality comprehensively."""
        
        # Constraint satisfaction
        constraint_sat = self._calculate_constraint_satisfaction(reconciled_forecasts, hierarchy)
        
        # Accuracy improvement
        accuracy_imp = self._calculate_accuracy_improvement(reconciled_forecasts, base_forecasts, targets)
        
        # Consistency
        consistency = self._calculate_consistency(reconciled_forecasts)
        
        # Statistical significance
        is_significant = self._test_significance(reconciled_forecasts, base_forecasts, targets)
        
        # Degradation detection
        degradation = self._detect_degradation()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(constraint_sat, accuracy_imp, consistency)
        
        report = QualityReport(
            constraint_satisfaction=constraint_sat,
            accuracy_improvement=accuracy_imp,
            consistency_score=consistency,
            statistical_significance=is_significant,
            degradation_detected=degradation,
            recommendations=recommendations
        )
        
        self.quality_history.append(report)
        
        return report
    
    def _calculate_constraint_satisfaction(self, forecasts: Dict, hierarchy) -> float:
        """Calculate how well constraints satisfied."""
        violations = []
        for node_name, node in hierarchy.nodes.items():
            if node.node_type != 'aggregate' or node_name not in forecasts:
                continue
            
            expected = sum(forecasts.get(child, 0) for child in node.children)
            actual = forecasts[node_name]
            error = np.abs(actual - expected).mean()
            violations.append(error)
        
        if not violations:
            return 1.0
        
        max_violation = max(violations)
        # Exponential decay: small violations OK
        return np.exp(-max_violation / 10.0)
    
    def _calculate_accuracy_improvement(self, reconciled, base, targets) -> float:
        """Calculate accuracy improvement over base forecasts."""
        base_errors = []
        reconciled_errors = []
        
        for key in targets:
            if key in base and key in reconciled:
                base_mape = np.mean(np.abs((targets[key] - base[key]) / (targets[key] + 1e-10)))
                rec_mape = np.mean(np.abs((targets[key] - reconciled[key]) / (targets[key] + 1e-10)))
                
                base_errors.append(base_mape)
                reconciled_errors.append(rec_mape)
        
        if not base_errors:
            return 0.0
        
        avg_base = np.mean(base_errors)
        avg_reconciled = np.mean(reconciled_errors)
        
        improvement = (avg_base - avg_reconciled) / avg_base if avg_base > 0 else 0.0
        return max(-1.0, min(1.0, improvement))  # Clamp to [-1, 1]
    
    def _test_significance(self, reconciled, base, targets) -> bool:
        """Test if improvement is statistically significant."""
        base_errors = []
        reconciled_errors = []
        
        for key in targets:
            if key in base and key in reconciled:
                base_err = np.abs(targets[key] - base[key])
                rec_err = np.abs(targets[key] - reconciled[key])
                base_errors.append(base_err.mean())
                reconciled_errors.append(rec_err.mean())
        
        if len(base_errors) < 3:
            return False
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(base_errors, reconciled_errors)
        
        return p_value < self.significance_level and t_stat > 0
    
    def _detect_degradation(self) -> bool:
        """Detect quality degradation over time."""
        if len(self.quality_history) < 5:
            return False
        
        recent_quality = [q.accuracy_improvement for q in self.quality_history[-5:]]
        
        # Check for declining trend
        from scipy.stats import linregress
        x = np.arange(len(recent_quality))
        slope, _, _, p_value, _ = linregress(x, recent_quality)
        
        # Significant negative trend
        return slope < -0.01 and p_value < 0.1
    
    def _generate_recommendations(self, constraint_sat, accuracy_imp, consistency) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        if constraint_sat < 0.95:
            recs.append("Constraint satisfaction low - consider stricter enforcement")
        
        if accuracy_imp < 0.02:
            recs.append("Minimal accuracy improvement - try alternative reconciliation method")
        
        if consistency < 0.8:
            recs.append("Low consistency - check for data quality issues")
        
        return recs
```

---

## 🧪 Testing

```python
def test_quality_analysis():
    """Test quality analyzer."""
    analyzer = ReconciliationQualityAnalyzer()
    
    reconciled = {'National': np.array([1000, 1100]), 'SE': np.array([600, 650])}
    base = {'National': np.array([980, 1080]), 'SE': np.array([590, 640])}
    targets = {'National': np.array([1010, 1110]), 'SE': np.array([605, 655])}
    
    hierarchy = HierarchyDefinition(sample_yaml)
    
    report = analyzer.analyze_quality(reconciled, base, targets, hierarchy)
    
    assert 0 <= report.constraint_satisfaction <= 1
    assert -1 <= report.accuracy_improvement <= 1
```

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Quality metrics accurate
- [ ] Statistical testing working
- [ ] Degradation detection functional
- [ ] Recommendations actionable
- [ ] Unit tests >85% coverage
- [ ] Documentation complete

---

**Epic:** [Epic-06B](../epics/Epic-06B.md)  
**Previous:** [PC-057-06B](PC-057-06B-reconciler-selector.md)  
**Next:** [PC-059-06B](PC-059-06B-method-comparator.md)
