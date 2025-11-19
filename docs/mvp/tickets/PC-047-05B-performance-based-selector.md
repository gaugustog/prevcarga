# PC-047-05B: Performance-Based Model Selection

**Ticket ID:** PC-047-05B  
**Epic:** [Epic-05B: Advanced Ensemble Methods](../epics/Epic-05B.md)  
**User Story:** US-6  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `PerformanceBasedSelector` that optimizes ensemble composition dynamically through multi-objective optimization balancing accuracy, diversity, and computational cost. Uses rolling window evaluation to prevent selection bias, optimizes ensemble size (2-5 models), and adapts selection strategy to different time series characteristics.

**As a** forecasting analyst  
**I want** intelligent model selection that maximizes ensemble performance  
**So that** I can automatically compose optimal ensembles under different conditions

---

## ✅ Acceptance Criteria

- [ ] `PerformanceBasedSelector` optimizes ensemble composition dynamically
- [ ] Multi-objective optimization balances accuracy, diversity, and robustness
- [ ] Rolling window evaluation prevents selection bias
- [ ] Ensemble size optimization (2-5 models) based on performance gains
- [ ] Selection strategy adapts to different time series characteristics
- [ ] Pareto optimization for multi-objective selection
- [ ] Rolling performance tracking
- [ ] Diversity-aware selection
- [ ] Computational cost consideration
- [ ] Time series clustering for specialization

---

## 🔧 Implementation Tasks

### 1. Setup Selection Module
- [ ] Create `src/models/combination/performance_selector.py`
- [ ] Import optimization libraries
- [ ] Import diversity analyzer
- [ ] Import performance metrics
- [ ] Add module docstrings

### 2. Implement PerformanceBasedSelector Class
- [ ] Define configuration schema
- [ ] Initialize performance tracking
- [ ] Setup multi-objective optimizer
- [ ] Configure selection strategy
- [ ] Store selection history

### 3. Implement Rolling Performance Tracking
- [ ] Create `RollingPerformanceTracker` class
- [ ] Track accuracy per model (rolling window)
- [ ] Track diversity contribution per model
- [ ] Track computational cost per model
- [ ] Calculate recent performance trends
- [ ] Update performance statistics

### 4. Implement Multi-Objective Scoring
- [ ] Create `_calculate_multi_objective_score()` method
- [ ] Score component 1: Accuracy (MAPE)
- [ ] Score component 2: Diversity contribution
- [ ] Score component 3: Computational efficiency
- [ ] Weighted combination of objectives
- [ ] Normalize scores to [0, 1]
- [ ] Return composite score

### 5. Implement Model Ranking
- [ ] Create `_rank_models()` method
- [ ] Rank by multi-objective score
- [ ] Apply recency weighting (recent > old)
- [ ] Consider performance stability
- [ ] Handle ties with diversity tiebreaker
- [ ] Return ranked model list

### 6. Implement Ensemble Size Optimization
- [ ] Create `_optimize_ensemble_size()` method
- [ ] Start with top model
- [ ] Iteratively add models (2-5 models)
- [ ] Calculate marginal performance gain
- [ ] Stop when gain < threshold
- [ ] Return optimal ensemble size

### 7. Implement Greedy Selection Algorithm
- [ ] Create `_greedy_select()` method
- [ ] Initialize with best performing model
- [ ] Add models maximizing multi-objective score
- [ ] Check diversity constraint (not too correlated)
- [ ] Limit to optimal ensemble size
- [ ] Return selected model subset

### 8. Implement Pareto Optimization Selection
- [ ] Create `_pareto_select()` method
- [ ] Evaluate all model combinations
- [ ] Calculate Pareto frontier (accuracy, diversity, cost)
- [ ] Select from Pareto frontier
- [ ] Balance objectives based on configuration
- [ ] Return Pareto-optimal ensemble

### 9. Implement Time Series Clustering
- [ ] Create `_cluster_time_series()` method
- [ ] Extract time series features
- [ ] Cluster into groups (k-means)
- [ ] Identify cluster characteristics
- [ ] Specialize selection per cluster
- [ ] Return cluster assignments

### 10. Implement Adaptive Selection Strategy
- [ ] Create `select_optimal_ensemble()` method
- [ ] Detect time series characteristics
- [ ] Select appropriate strategy (greedy vs Pareto)
- [ ] Apply time series-specific selection
- [ ] Adapt to recent performance patterns
- [ ] Return selected ensemble

### 11. Implement Fallback Mechanism
- [ ] Create `_fallback_selection()` method
- [ ] Detect when optimal models unavailable
- [ ] Fallback to next best alternatives
- [ ] Ensure minimum diversity
- [ ] Guarantee at least 2 models selected
- [ ] Return fallback ensemble

### 12. Implement Selection Validation
- [ ] Create `_validate_selection()` method
- [ ] Check ensemble has sufficient diversity
- [ ] Verify performance improvement over best individual
- [ ] Check computational budget
- [ ] Test ensemble on validation set
- [ ] Return validation results

### 13. Implement Selection History Tracking
- [ ] Create `SelectionHistory` class
- [ ] Store selected ensembles over time
- [ ] Track selection reasons
- [ ] Monitor selection stability
- [ ] Analyze selection patterns
- [ ] Return history summary

### 14. Implement Computational Cost Estimation
- [ ] Create `_estimate_computational_cost()` method
- [ ] Estimate training time per model
- [ ] Estimate inference time per model
- [ ] Calculate total ensemble cost
- [ ] Compare against budget
- [ ] Return cost estimates

### 15. Handle Edge Cases
- [ ] Handle single available model
- [ ] Handle all models poor performance
- [ ] Handle insufficient diversity
- [ ] Handle computational budget constraints
- [ ] Validate selection robustness

### 16. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_performance_selector.py`
- [ ] Test rolling performance tracking
- [ ] Test multi-objective scoring
- [ ] Test model ranking
- [ ] Test ensemble size optimization
- [ ] Test greedy selection
- [ ] Test Pareto optimization
- [ ] Test adaptive selection

### 17. Write Integration Tests
- [ ] Test with real model performance data
- [ ] Test with different time series
- [ ] Test selection stability over time
- [ ] Validate performance improvement
- [ ] Test computational performance

### 18. Create Usage Examples
- [ ] Create `examples/performance_selector_demo.py`
- [ ] Show multi-objective selection
- [ ] Show ensemble size optimization
- [ ] Show adaptive selection
- [ ] Compare selection strategies

---

## 💻 Implementation Details

### PerformanceBasedSelector Implementation

```python
"""Performance-based model selection for optimal ensemble composition."""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import deque
from dataclasses import dataclass
from itertools import combinations

from src.models.combination.diversity_analyzer import EnsembleDiversityAnalyzer
from src.evaluation.metrics import calculate_mape
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceSelectorConfig:
    """Configuration for performance-based selector."""
    
    selection_strategy: str = 'adaptive'  # 'greedy', 'pareto', 'adaptive'
    max_ensemble_size: int = 5
    min_ensemble_size: int = 2
    window_size: int = 48  # Rolling window for performance
    diversity_weight: float = 0.3  # Weight for diversity in scoring
    accuracy_weight: float = 0.6  # Weight for accuracy in scoring
    cost_weight: float = 0.1  # Weight for computational cost
    marginal_gain_threshold: float = 0.01  # Minimum gain to add model
    recency_decay: float = 0.95  # Exponential decay for older performance


class RollingPerformanceTracker:
    """Track rolling performance metrics."""
    
    def __init__(self, window_size: int = 48):
        self.window_size = window_size
        self.accuracy_history: Dict[str, deque] = {}
        self.diversity_history: Dict[str, deque] = {}
        self.cost_history: Dict[str, deque] = {}
    
    def update(
        self,
        model_name: str,
        accuracy: float,
        diversity: float,
        cost: float = 1.0
    ):
        """Update performance metrics for model."""
        if model_name not in self.accuracy_history:
            self.accuracy_history[model_name] = deque(maxlen=self.window_size)
            self.diversity_history[model_name] = deque(maxlen=self.window_size)
            self.cost_history[model_name] = deque(maxlen=self.window_size)
        
        self.accuracy_history[model_name].append(accuracy)
        self.diversity_history[model_name].append(diversity)
        self.cost_history[model_name].append(cost)
    
    def get_recent_performance(self, model_name: str) -> Dict[str, float]:
        """Get recent average performance."""
        if model_name not in self.accuracy_history:
            return {'accuracy': np.inf, 'diversity': 0.0, 'cost': 1.0}
        
        return {
            'accuracy': np.mean(list(self.accuracy_history[model_name])),
            'diversity': np.mean(list(self.diversity_history[model_name])),
            'cost': np.mean(list(self.cost_history[model_name]))
        }


@dataclass
class SelectionRecord:
    """Record of ensemble selection."""
    
    timestamp: pd.Timestamp
    selected_models: List[str]
    selection_strategy: str
    composite_scores: Dict[str, float]
    expected_performance: float
    ensemble_size: int
    reason: str


class PerformanceBasedSelector:
    """
    Performance-based model selection for optimal ensemble composition.
    
    Multi-Objective Scoring:
        score = w_acc * accuracy + w_div * diversity - w_cost * cost
    
    Selection Strategies:
    - Greedy: Iteratively add best scoring models
    - Pareto: Find Pareto-optimal ensemble combinations
    - Adaptive: Choose strategy based on data characteristics
    
    Ensemble Size Optimization:
        Add models while: marginal_gain > threshold
        Optimal size: argmax(performance / cost)
    
    Rolling Evaluation:
        Use window_size recent predictions to prevent overfitting
    
    Time Series Specialization:
        Cluster time series → specialized selection per cluster
    
    Example:
        >>> selector = PerformanceBasedSelector(
        ...     config={'selection_strategy': 'adaptive'}
        ... )
        >>> selector.fit(predictions, targets)
        >>> selected = selector.select_optimal_ensemble()
        >>> print(f"Selected models: {selected}")
        >>> print(f"Expected improvement: {selector.expected_improvement:.1%}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance-based selector.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = {}
        
        config_obj = PerformanceSelectorConfig(**config)
        self.selection_strategy = config_obj.selection_strategy
        self.max_ensemble_size = config_obj.max_ensemble_size
        self.min_ensemble_size = config_obj.min_ensemble_size
        self.window_size = config_obj.window_size
        self.diversity_weight = config_obj.diversity_weight
        self.accuracy_weight = config_obj.accuracy_weight
        self.cost_weight = config_obj.cost_weight
        self.marginal_gain_threshold = config_obj.marginal_gain_threshold
        self.recency_decay = config_obj.recency_decay
        
        # Components
        self.performance_tracker = RollingPerformanceTracker(window_size)
        self.diversity_analyzer = EnsembleDiversityAnalyzer()
        
        # Storage
        self.selection_history: List[SelectionRecord] = []
        self.model_pool: List[str] = []
        self.expected_improvement: Optional[float] = None
    
    def fit(
        self,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame
    ):
        """
        Fit selector using historical predictions.
        
        Args:
            predictions: Model predictions
            targets: True targets
        """
        logger.info("Fitting performance-based selector")
        
        self.model_pool = list(predictions.keys())
        
        # Calculate diversity metrics
        diversity_metrics = self.diversity_analyzer.calculate_diversity(predictions)
        diversity_scores = diversity_metrics.get('diversity_scores', {})
        
        # Update performance tracking
        pred_cols = [col for col in next(iter(predictions.values())).columns 
                     if col.startswith('pred_h')]
        
        for model_name in self.model_pool:
            errors = []
            for col in pred_cols:
                horizon = int(col.split('_h')[1])
                target_col = f'target_h{horizon}'
                if target_col in targets.columns:
                    mape = calculate_mape(targets[target_col], predictions[model_name][col])
                    errors.append(mape)
            
            avg_accuracy = np.mean(errors) if errors else np.inf
            diversity = diversity_scores.get(model_name, 0.0)
            
            self.performance_tracker.update(model_name, avg_accuracy, diversity)
        
        logger.info(f"Selector fitted with {len(self.model_pool)} models")
    
    def select_optimal_ensemble(self) -> List[str]:
        """
        Select optimal ensemble composition.
        
        Returns:
            List of selected model names
        """
        logger.info(f"Selecting optimal ensemble ({self.selection_strategy})")
        
        if len(self.model_pool) <= self.min_ensemble_size:
            logger.info("Insufficient models, using all available")
            return self.model_pool
        
        # Choose selection strategy
        if self.selection_strategy == 'greedy':
            selected = self._greedy_select()
        elif self.selection_strategy == 'pareto':
            selected = self._pareto_select()
        elif self.selection_strategy == 'adaptive':
            selected = self._adaptive_select()
        else:
            raise ValueError(f"Unknown strategy: {self.selection_strategy}")
        
        # Validate selection
        if len(selected) < self.min_ensemble_size:
            selected = self._fallback_selection()
        
        # Record selection
        self._record_selection(selected)
        
        logger.info(f"Selected {len(selected)} models: {selected}")
        return selected
    
    def _greedy_select(self) -> List[str]:
        """Greedy selection: iteratively add best models."""
        # Rank models by multi-objective score
        scores = self._calculate_multi_objective_scores()
        ranked_models = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
        
        # Start with best model
        selected = [ranked_models[0]]
        base_performance = self._estimate_ensemble_performance(selected)
        
        # Add models while improving performance
        for candidate in ranked_models[1:]:
            if len(selected) >= self.max_ensemble_size:
                break
            
            # Test adding this model
            test_ensemble = selected + [candidate]
            test_performance = self._estimate_ensemble_performance(test_ensemble)
            
            # Calculate marginal gain
            marginal_gain = (base_performance - test_performance) / base_performance
            
            if marginal_gain > self.marginal_gain_threshold:
                selected.append(candidate)
                base_performance = test_performance
        
        return selected
    
    def _pareto_select(self) -> List[str]:
        """Pareto optimization: find Pareto-optimal ensemble."""
        # Evaluate all possible combinations (computationally expensive)
        best_ensemble = None
        best_score = -np.inf
        
        for size in range(self.min_ensemble_size, 
                         min(self.max_ensemble_size, len(self.model_pool)) + 1):
            for combo in combinations(self.model_pool, size):
                # Calculate composite score
                accuracy = self._estimate_ensemble_performance(list(combo))
                diversity = self._calculate_ensemble_diversity(list(combo))
                cost = len(combo)
                
                score = (self.accuracy_weight * (1.0 / accuracy) +
                        self.diversity_weight * diversity -
                        self.cost_weight * cost)
                
                if score > best_score:
                    best_score = score
                    best_ensemble = list(combo)
        
        return best_ensemble if best_ensemble else self.model_pool[:self.min_ensemble_size]
    
    def _adaptive_select(self) -> List[str]:
        """Adaptive selection: choose strategy based on characteristics."""
        # Simple heuristic: use greedy for many models, Pareto for few
        if len(self.model_pool) > 6:
            return self._greedy_select()
        else:
            return self._pareto_select()
    
    def _calculate_multi_objective_scores(self) -> Dict[str, float]:
        """Calculate multi-objective score for each model."""
        scores = {}
        
        for model in self.model_pool:
            perf = self.performance_tracker.get_recent_performance(model)
            
            # Normalize components
            accuracy_score = 1.0 / (perf['accuracy'] + 1.0)  # Lower error = higher score
            diversity_score = perf['diversity']
            cost_score = 1.0 / perf['cost']
            
            # Weighted combination
            composite = (
                self.accuracy_weight * accuracy_score +
                self.diversity_weight * diversity_score +
                self.cost_weight * cost_score
            )
            
            scores[model] = composite
        
        return scores
    
    def _estimate_ensemble_performance(self, models: List[str]) -> float:
        """Estimate ensemble MAPE (lower is better)."""
        # Simple estimate: weighted average of individual performances
        performances = [
            self.performance_tracker.get_recent_performance(m)['accuracy']
            for m in models
        ]
        
        # Account for diversity benefit
        diversity = self._calculate_ensemble_diversity(models)
        diversity_benefit = 0.9 + 0.1 * min(diversity, 1.0)  # Up to 10% benefit
        
        return np.mean(performances) * diversity_benefit
    
    def _calculate_ensemble_diversity(self, models: List[str]) -> float:
        """Calculate diversity score for model subset."""
        if len(models) < 2:
            return 0.0
        
        # Average diversity contribution
        diversity_scores = []
        for model in models:
            perf = self.performance_tracker.get_recent_performance(model)
            diversity_scores.append(perf['diversity'])
        
        return np.mean(diversity_scores)
    
    def _fallback_selection(self) -> List[str]:
        """Fallback: select top N models by accuracy."""
        scores = self._calculate_multi_objective_scores()
        ranked = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
        
        return ranked[:self.min_ensemble_size]
    
    def _record_selection(self, selected: List[str]):
        """Record selection decision."""
        scores = self._calculate_multi_objective_scores()
        
        record = SelectionRecord(
            timestamp=pd.Timestamp.now(),
            selected_models=selected,
            selection_strategy=self.selection_strategy,
            composite_scores={m: scores[m] for m in selected},
            expected_performance=self._estimate_ensemble_performance(selected),
            ensemble_size=len(selected),
            reason=f"Selected via {self.selection_strategy} optimization"
        )
        
        self.selection_history.append(record)
        
        # Calculate expected improvement
        all_model_performance = np.mean([
            self.performance_tracker.get_recent_performance(m)['accuracy']
            for m in self.model_pool
        ])
        
        self.expected_improvement = (
            (all_model_performance - record.expected_performance) / 
            all_model_performance
        )
```

---

## 🧪 Testing & Validation

```python
"""Tests for performance-based selector."""
import pytest
import pandas as pd
import numpy as np

from src.models.combination.performance_selector import PerformanceBasedSelector


@pytest.fixture
def sample_selection_data():
    """Create sample data for model selection."""
    dates = pd.date_range('2024-01-01', periods=200, freq='30min')
    
    # Models with different characteristics
    predictions = {
        'accurate_model': pd.DataFrame({
            'pred_h0': np.random.randn(200) * 20 + 1000  # Low error
        }, index=dates),
        'diverse_model': pd.DataFrame({
            'pred_h0': np.random.randn(200) * 80 + 1050  # High diversity
        }, index=dates),
        'stable_model': pd.DataFrame({
            'pred_h0': np.random.randn(200) * 30 + 1000  # Moderate
        }, index=dates),
        'poor_model': pd.DataFrame({
            'pred_h0': np.random.randn(200) * 100 + 1100  # High error
        }, index=dates)
    }
    
    targets = pd.DataFrame({
        'target_h0': np.random.randn(200) * 30 + 1000
    }, index=dates)
    
    return predictions, targets


def test_selector_fitting(sample_selection_data):
    """Test selector fitting."""
    predictions, targets = sample_selection_data
    
    selector = PerformanceBasedSelector()
    selector.fit(predictions, targets)
    
    assert len(selector.model_pool) == 4


def test_greedy_selection(sample_selection_data):
    """Test greedy selection strategy."""
    predictions, targets = sample_selection_data
    
    selector = PerformanceBasedSelector(config={'selection_strategy': 'greedy'})
    selector.fit(predictions, targets)
    
    selected = selector.select_optimal_ensemble()
    
    assert 2 <= len(selected) <= 5
    assert 'accurate_model' in selected  # Should select best model


def test_ensemble_size_optimization(sample_selection_data):
    """Test ensemble size optimization."""
    predictions, targets = sample_selection_data
    
    selector = PerformanceBasedSelector(config={'max_ensemble_size': 3})
    selector.fit(predictions, targets)
    
    selected = selector.select_optimal_ensemble()
    
    assert len(selected) <= 3


def test_selection_history(sample_selection_data):
    """Test selection history tracking."""
    predictions, targets = sample_selection_data
    
    selector = PerformanceBasedSelector()
    selector.fit(predictions, targets)
    selector.select_optimal_ensemble()
    
    assert len(selector.selection_history) > 0
    assert selector.expected_improvement is not None
```

---

## 📝 Technical Notes

### Multi-Objective Optimization
- Balance accuracy, diversity, cost
- Weighted scoring function
- Pareto frontier for optimal trade-offs

### Ensemble Size
- Start with best model
- Add while marginal gain > threshold
- Limit to 2-5 models

---

## 🔗 Dependencies

**Depends On:**
- PC-036-05A: Base Combiner Interface
- PC-046-05B: Ensemble Diversity Analyzer
- All combination methods (PC-037 through PC-045)

**Blocks:**
- Epic-06A: Hierarchical reconciliation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Multi-objective optimization working
- [ ] Rolling performance tracking functional
- [ ] Ensemble size optimization operational
- [ ] Multiple selection strategies supported
- [ ] Adaptive selection working
- [ ] Fallback mechanism robust
- [ ] Selection validation preventing poor choices
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-05B: Advanced Ensemble Methods](../epics/Epic-05B.md)  
**Previous:** [PC-046-05B: Ensemble Diversity Analyzer](PC-046-05B-ensemble-diversity-analyzer.md)  
**Next Epic:** [Epic-06A: Hierarchical Reconciliation](../epics/Epic-06A.md)
