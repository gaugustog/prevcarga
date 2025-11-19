# PC-057-06B: Reconciler Selection System

**Ticket ID:** PC-057-06B  
**Epic:** [Epic-06B: Advanced Reconciliation Methods](../epics/Epic-06B.md)  
**User Story:** US-4  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `ReconcilerSelector` that automatically selects the optimal reconciliation method based on performance metrics, data characteristics, and computational efficiency. Uses rolling window evaluation to prevent overfitting, adapts to changing data patterns, provides transparent and interpretable selection logic, and enables multi-objective optimization with configurable weights.

**As a** forecasting analyst  
**I want** automatic selection of the best reconciliation method  
**So that** I achieve optimal reconciliation without manual tuning

---

## ✅ Acceptance Criteria

- [ ] `ReconcilerSelector` evaluates multiple methods
- [ ] Performance metrics: accuracy, consistency, efficiency
- [ ] Rolling window evaluation prevents overfitting
- [ ] Method selection adapts to data characteristics
- [ ] Transparent selection logic
- [ ] Multi-objective optimization
- [ ] Selection improves overall performance
- [ ] Configuration flexibility
- [ ] Performance <10s for method selection
- [ ] Comprehensive selection reports

---

## 🔧 Implementation (Condensed)

```python
"""Automatic reconciliation method selection."""
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from src.models.reconciliation.base_reconciler import BaseReconciler
from src.models.reconciliation.hierarchy import HierarchyDefinition


@dataclass
class MethodPerformance:
    """Performance metrics for a reconciliation method."""
    method_name: str
    accuracy_score: float  # MAPE or MAE
    consistency_score: float  # Constraint satisfaction
    efficiency_score: float  # Computation time
    overall_score: float
    

class ReconcilerSelector:
    """
    Automatic selection of optimal reconciliation method.
    
    Evaluates multiple methods using:
        - Accuracy: forecast error metrics
        - Consistency: constraint satisfaction
        - Efficiency: computational performance
    
    Selection Strategy:
        1. Evaluate all methods on rolling windows
        2. Calculate weighted score: w_acc*accuracy + w_cons*consistency + w_eff*efficiency
        3. Select method with highest score
        4. Adapt to changing data characteristics
    
    Example:
        >>> methods = [MinTReconciler(), OLSReconciler(), WLSReconciler(), ShrinkageReconciler()]
        >>> selector = ReconcilerSelector(methods, weights={'accuracy': 0.6, 'consistency': 0.3, 'efficiency': 0.1})
        >>> best_method = selector.select_method(forecasts, hierarchy, evaluation_data)
        >>> result = best_method.reconcile(forecasts, hierarchy)
    """
    
    def __init__(
        self,
        methods: List[BaseReconciler],
        weights: Optional[Dict[str, float]] = None,
        rolling_window_size: int = 10
    ):
        self.methods = methods
        self.weights = weights or {'accuracy': 0.6, 'consistency': 0.3, 'efficiency': 0.1}
        self.rolling_window_size = rolling_window_size
        self.performance_history = []
    
    def select_method(
        self,
        base_forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition,
        evaluation_data: Dict[str, np.ndarray],
        **kwargs
    ) -> BaseReconciler:
        """Select optimal reconciliation method."""
        performances = []
        
        for method in self.methods:
            perf = self._evaluate_method(method, base_forecasts, hierarchy, evaluation_data, **kwargs)
            performances.append(perf)
        
        # Select best method
        best_method_idx = np.argmax([p.overall_score for p in performances])
        selected_method = self.methods[best_method_idx]
        
        logger.info(f"Selected method: {selected_method.name}")
        
        return selected_method
    
    def _evaluate_method(
        self,
        method: BaseReconciler,
        forecasts: Dict,
        hierarchy: HierarchyDefinition,
        eval_data: Dict,
        **kwargs
    ) -> MethodPerformance:
        """Evaluate single method."""
        import time
        
        # Reconcile
        start = time.time()
        result = method.reconcile(forecasts, hierarchy, **kwargs)
        duration = time.time() - start
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(result.reconciled_forecasts, eval_data)
        
        # Calculate consistency
        consistency = self._calculate_consistency(result.constraint_violations)
        
        # Calculate efficiency
        efficiency = self._calculate_efficiency(duration)
        
        # Weighted overall score
        overall = (
            self.weights['accuracy'] * accuracy +
            self.weights['consistency'] * consistency +
            self.weights['efficiency'] * efficiency
        )
        
        return MethodPerformance(
            method_name=method.name,
            accuracy_score=accuracy,
            consistency_score=consistency,
            efficiency_score=efficiency,
            overall_score=overall
        )
    
    def _calculate_accuracy(self, reconciled: Dict, actuals: Dict) -> float:
        """Calculate accuracy score (1 - normalized_error)."""
        errors = []
        for key in reconciled:
            if key in actuals:
                mape = np.mean(np.abs((actuals[key] - reconciled[key]) / (actuals[key] + 1e-10)))
                errors.append(mape)
        
        avg_mape = np.mean(errors) if errors else 0.5
        return max(0.0, 1.0 - avg_mape)  # Higher is better
    
    def _calculate_consistency(self, violations: List) -> float:
        """Calculate consistency score based on violations."""
        if not violations:
            return 1.0
        
        max_violation = max(v.absolute_error for v in violations)
        # Exponential decay: small violations OK, large bad
        return np.exp(-max_violation / 100.0)
    
    def _calculate_efficiency(self, duration: float, target: float = 5.0) -> float:
        """Calculate efficiency score."""
        # Sigmoid function: fast methods score high
        return 1.0 / (1.0 + np.exp((duration - target) / target))
```

---

## 🧪 Testing

```python
def test_reconciler_selection():
    """Test automatic method selection."""
    methods = [
        MinTReconciler(),
        OLSReconciler(config={'covariance_method': 'ledoit_wolf'}),
        WLSReconciler(config={'weighting_scheme': 'inverse_variance'})
    ]
    
    selector = ReconcilerSelector(methods)
    
    forecasts = {'National': np.array([1000]), 'SE': np.array([600])}
    eval_data = {'National': np.array([1010]), 'SE': np.array([605])}
    hierarchy = HierarchyDefinition(sample_yaml)
    
    best_method = selector.select_method(forecasts, hierarchy, eval_data)
    
    assert best_method in methods
```

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Method selection working
- [ ] Performance evaluation accurate
- [ ] Rolling window implemented
- [ ] Selection logic transparent
- [ ] Unit tests >85% coverage
- [ ] Documentation complete
- [ ] Code reviewed

---

**Epic:** [Epic-06B](../epics/Epic-06B.md)  
**Previous:** [PC-056-06B](PC-056-06B-shrinkage-reconciler.md)  
**Next:** [PC-058-06B](PC-058-06B-quality-analyzer.md)
