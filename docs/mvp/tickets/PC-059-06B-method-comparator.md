# PC-059-06B: Method Comparison Framework

**Ticket ID:** PC-059-06B  
**Epic:** [Epic-06B: Advanced Reconciliation Methods](../epics/Epic-06B.md)  
**User Story:** US-6  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `ReconciliationComparator` that provides systematic benchmarking and comparison of reconciliation methods. Includes cross-validation framework for unbiased performance estimation, statistical significance testing for method differences, performance visualization and ranking, and recommendation engine based on data characteristics.

**As a** ML engineer  
**I want** systematic comparison of reconciliation methods  
**So that** I can validate performance and guide strategy

---

## ✅ Acceptance Criteria

- [ ] Systematic method benchmarking
- [ ] Time-aware cross-validation
- [ ] Statistical significance testing
- [ ] Performance visualization
- [ ] Method ranking capabilities
- [ ] Recommendation engine
- [ ] Comparison reports
- [ ] Integration with all reconcilers
- [ ] Reproducible results
- [ ] Comprehensive documentation

---

## 💻 Implementation (Condensed)

```python
"""Framework for systematic reconciliation method comparison."""
from typing import List, Dict
import numpy as np
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt


@dataclass
class ComparisonReport:
    """Method comparison results."""
    method_rankings: Dict[str, int]
    performance_matrix: np.ndarray
    statistical_tests: Dict[str, Dict]
    recommendations: List[str]


class ReconciliationComparator:
    """
    Systematic comparison of reconciliation methods.
    
    Features:
        - Cross-validation: Time-aware splits
        - Benchmarking: Standardized evaluation
        - Statistical testing: Paired tests for significance
        - Visualization: Performance charts
        - Recommendations: Data-driven method selection
    
    Example:
        >>> methods = [MinTReconciler(), OLSReconciler(), WLSReconciler(), ShrinkageReconciler()]
        >>> comparator = ReconciliationComparator()
        >>> report = comparator.compare_methods(
        ...     methods=methods,
        ...     evaluation_data={'forecasts': ..., 'targets': ...},
        ...     hierarchy=hierarchy
        ... )
        >>> print("Method Rankings:")
        >>> for method, rank in report.method_rankings.items():
        ...     print(f"  {method}: {rank}")
    """
    
    def __init__(self, cv_folds: int = 5, significance_level: float = 0.05):
        self.cv_folds = cv_folds
        self.significance_level = significance_level
    
    def compare_methods(
        self,
        methods: List[BaseReconciler],
        evaluation_data: Dict,
        hierarchy: 'HierarchyDefinition'
    ) -> ComparisonReport:
        """Compare reconciliation methods systematically."""
        
        # Cross-validate all methods
        cv_results = self.cross_validate_methods(methods, evaluation_data, hierarchy)
        
        # Calculate performance matrix
        perf_matrix = self._build_performance_matrix(cv_results)
        
        # Statistical testing
        stat_tests = self._statistical_comparison(cv_results)
        
        # Rank methods
        rankings = self._rank_methods(perf_matrix)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(rankings, evaluation_data)
        
        return ComparisonReport(
            method_rankings=rankings,
            performance_matrix=perf_matrix,
            statistical_tests=stat_tests,
            recommendations=recommendations
        )
    
    def cross_validate_methods(
        self,
        methods: List[BaseReconciler],
        data: Dict,
        hierarchy: 'HierarchyDefinition'
    ) -> Dict[str, List[float]]:
        """Cross-validate methods using time-aware splits."""
        from sklearn.model_selection import TimeSeriesSplit
        
        results = {method.name: [] for method in methods}
        
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Time series data
        forecasts = data['forecasts']
        targets = data['targets']
        
        # Split by time
        for train_idx, test_idx in tscv.split(list(forecasts.values())[0]):
            train_forecasts = {k: v[train_idx] for k, v in forecasts.items()}
            test_forecasts = {k: v[test_idx] for k, v in forecasts.items()}
            test_targets = {k: v[test_idx] for k, v in targets.items()}
            
            for method in methods:
                try:
                    result = method.reconcile(test_forecasts, hierarchy)
                    
                    # Calculate performance
                    mape = self._calculate_mape(result.reconciled_forecasts, test_targets)
                    results[method.name].append(mape)
                
                except Exception as e:
                    logger.warning(f"Method {method.name} failed: {e}")
                    results[method.name].append(np.inf)
        
        return results
    
    def _statistical_comparison(self, cv_results: Dict[str, List[float]]) -> Dict:
        """Perform pairwise statistical tests."""
        method_names = list(cv_results.keys())
        tests = {}
        
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                errors1 = cv_results[method1]
                errors2 = cv_results[method2]
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(errors1, errors2)
                
                # Wilcoxon signed-rank test (non-parametric)
                w_stat, w_pvalue = stats.wilcoxon(errors1, errors2)
                
                tests[f"{method1}_vs_{method2}"] = {
                    't_test': {'statistic': t_stat, 'p_value': p_value},
                    'wilcoxon': {'statistic': w_stat, 'p_value': w_pvalue},
                    'significant': p_value < self.significance_level
                }
        
        return tests
    
    def _rank_methods(self, perf_matrix: np.ndarray) -> Dict[str, int]:
        """Rank methods by average performance."""
        avg_performance = perf_matrix.mean(axis=1)
        rankings = {}
        
        # Lower error is better, so sort ascending
        sorted_indices = np.argsort(avg_performance)
        
        for rank, idx in enumerate(sorted_indices, start=1):
            method_name = list(self.cv_results.keys())[idx]
            rankings[method_name] = rank
        
        return rankings
    
    def visualize_comparison(self, report: ComparisonReport, save_path: str = None):
        """Visualize method comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Performance boxplot
        ax1 = axes[0]
        ax1.boxplot(report.performance_matrix.T)
        ax1.set_xlabel('Method')
        ax1.set_ylabel('MAPE')
        ax1.set_title('Method Performance Distribution')
        
        # Rankings bar chart
        ax2 = axes[1]
        methods = list(report.method_rankings.keys())
        ranks = list(report.method_rankings.values())
        ax2.barh(methods, ranks)
        ax2.set_xlabel('Rank')
        ax2.set_title('Method Rankings')
        ax2.invert_xaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def _calculate_mape(self, forecasts: Dict, targets: Dict) -> float:
        """Calculate MAPE across all series."""
        errors = []
        for key in targets:
            if key in forecasts:
                mape = np.mean(np.abs((targets[key] - forecasts[key]) / (targets[key] + 1e-10)))
                errors.append(mape)
        
        return np.mean(errors) if errors else np.inf
```

---

## 🧪 Testing

```python
def test_method_comparison():
    """Test method comparison framework."""
    methods = [MinTReconciler(), OLSReconciler()]
    
    evaluation_data = {
        'forecasts': {'National': np.random.randn(100), 'SE': np.random.randn(100)},
        'targets': {'National': np.random.randn(100), 'SE': np.random.randn(100)}
    }
    
    comparator = ReconciliationComparator(cv_folds=3)
    hierarchy = HierarchyDefinition(sample_yaml)
    
    report = comparator.compare_methods(methods, evaluation_data, hierarchy)
    
    assert len(report.method_rankings) == 2
    assert set(report.method_rankings.values()) == {1, 2}
```

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Cross-validation working
- [ ] Statistical testing functional
- [ ] Visualization implemented
- [ ] Rankings accurate
- [ ] Recommendations useful
- [ ] Unit tests >85% coverage
- [ ] Documentation complete

---

**Epic:** [Epic-06B](../epics/Epic-06B.md)  
**Previous:** [PC-058-06B](PC-058-06B-quality-analyzer.md)  
**Next:** Epic-07 Evaluation System
