# PC-046-05B: Ensemble Diversity Metrics

**Ticket ID:** PC-046-05B  
**Epic:** [Epic-05B: Advanced Ensemble Methods](../epics/Epic-05B.md)  
**User Story:** US-5  
**Story Points:** 5  
**Priority:** Medium  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `EnsembleDiversityAnalyzer` that calculates quantitative measures of ensemble diversity and quality including disagreement metrics, prediction correlation, spread analysis, quality-diversity trade-off, and diversity monitoring with alerts when ensemble becomes too homogeneous.

**As a** ML engineer  
**I want** quantitative measures of ensemble diversity and quality  
**So that** I can optimize ensemble composition and detect when diversity is insufficient

---

## ✅ Acceptance Criteria

- [ ] `EnsembleDiversityAnalyzer` calculates multiple diversity metrics
- [ ] Diversity measures include disagreement, correlation, and prediction spread
- [ ] Quality-diversity trade-off analysis guides ensemble optimization
- [ ] Diversity monitoring alerts when ensemble becomes too homogeneous
- [ ] Diversity-guided model selection for optimal ensemble composition
- [ ] Pairwise prediction correlations
- [ ] Disagreement measures (MAD, entropy)
- [ ] Quality-diversity Pareto frontier
- [ ] Statistical significance tests
- [ ] Integration with model selection

---

## 🔧 Implementation Tasks

### 1. Setup Diversity Module
- [ ] Create `src/models/combination/diversity_analyzer.py`
- [ ] Import scipy for statistics
- [ ] Import sklearn metrics
- [ ] Import visualization libraries
- [ ] Add module docstrings

### 2. Implement EnsembleDiversityAnalyzer Class
- [ ] Define configuration schema
- [ ] Initialize diversity metrics storage
- [ ] Setup monitoring thresholds
- [ ] Configure alert system
- [ ] Store analysis history

### 3. Implement Pairwise Correlation Analysis
- [ ] Create `calculate_pairwise_correlations()` method
- [ ] Calculate Pearson correlation between all model pairs
- [ ] Calculate Spearman rank correlation
- [ ] Create correlation matrix
- [ ] Identify highly correlated models (r > 0.9)
- [ ] Return correlation metrics

### 4. Implement Disagreement Measures
- [ ] Create `calculate_disagreement()` method
- [ ] Calculate mean absolute difference between predictions
- [ ] Calculate pairwise disagreement for all pairs
- [ ] Calculate average disagreement per model
- [ ] Normalize disagreement by prediction scale
- [ ] Return disagreement metrics

### 5. Implement Prediction Spread Analysis
- [ ] Create `calculate_prediction_spread()` method
- [ ] Calculate variance across model predictions
- [ ] Calculate coefficient of variation
- [ ] Calculate range (max - min) per timestamp
- [ ] Calculate interquartile range
- [ ] Return spread metrics

### 6. Implement Entropy-Based Diversity
- [ ] Create `calculate_diversity_entropy()` method
- [ ] Discretize predictions into bins
- [ ] Calculate vote entropy
- [ ] Measure prediction distribution uniformity
- [ ] Higher entropy = more diversity
- [ ] Return entropy score

### 7. Implement Quality Metrics
- [ ] Create `calculate_quality_metrics()` method
- [ ] Calculate accuracy (MAPE, MAE, RMSE) per model
- [ ] Calculate ensemble accuracy
- [ ] Calculate improvement over best individual
- [ ] Rank models by performance
- [ ] Return quality metrics

### 8. Implement Quality-Diversity Trade-Off Analysis
- [ ] Create `analyze_quality_diversity_tradeoff()` method
- [ ] Plot accuracy vs diversity for each model
- [ ] Calculate Pareto frontier
- [ ] Identify optimal accuracy-diversity balance
- [ ] Recommend model subset for ensemble
- [ ] Return trade-off analysis

### 9. Implement Diversity Monitoring
- [ ] Create `monitor_diversity()` method
- [ ] Track diversity over time
- [ ] Calculate rolling diversity statistics
- [ ] Compare against threshold
- [ ] Detect diversity degradation
- [ ] Return monitoring status

### 10. Implement Alert System
- [ ] Create `DiversityAlert` class
- [ ] Define alert levels (info, warning, critical)
- [ ] Trigger alert if diversity < threshold
- [ ] Trigger alert if correlation > threshold
- [ ] Provide actionable recommendations
- [ ] Store alert history

### 11. Implement Diversity-Guided Selection
- [ ] Create `select_diverse_subset()` method
- [ ] Start with best performing model
- [ ] Iteratively add models maximizing diversity
- [ ] Balance accuracy and diversity
- [ ] Limit ensemble size (2-5 models)
- [ ] Return selected model subset

### 12. Implement Statistical Significance Testing
- [ ] Create `test_diversity_significance()` method
- [ ] Test if diversity is significant vs random
- [ ] Use permutation test or bootstrap
- [ ] Calculate p-value for diversity metrics
- [ ] Require p < 0.05 for significance
- [ ] Return test results

### 13. Implement Diversity Visualization
- [ ] Create `visualize_diversity()` method
- [ ] Plot correlation matrix heatmap
- [ ] Plot quality-diversity scatter
- [ ] Plot diversity over time
- [ ] Plot pairwise disagreement
- [ ] Save plots to file

### 14. Implement Diversity Report Generation
- [ ] Create `generate_diversity_report()` method
- [ ] Summarize all diversity metrics
- [ ] Highlight low diversity warnings
- [ ] Recommend ensemble improvements
- [ ] Include visualizations
- [ ] Return formatted report

### 15. Handle Edge Cases
- [ ] Handle single model (no diversity)
- [ ] Handle identical predictions
- [ ] Handle missing predictions
- [ ] Handle constant predictions
- [ ] Validate diversity calculations

### 16. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_diversity_analyzer.py`
- [ ] Test correlation calculations
- [ ] Test disagreement metrics
- [ ] Test spread analysis
- [ ] Test quality-diversity trade-off
- [ ] Test diversity monitoring
- [ ] Test alert system

### 17. Write Integration Tests
- [ ] Test with real model predictions
- [ ] Test with high vs low diversity ensembles
- [ ] Test diversity-guided selection
- [ ] Validate recommendations
- [ ] Test computational performance

### 18. Create Usage Examples
- [ ] Create `examples/diversity_analysis_demo.py`
- [ ] Show diversity calculation workflow
- [ ] Show quality-diversity analysis
- [ ] Show diversity monitoring
- [ ] Show diversity-guided selection

---

## 💻 Implementation Details

### EnsembleDiversityAnalyzer Implementation

```python
"""Ensemble diversity and quality analysis."""
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DiversityAlert:
    """Diversity monitoring alert."""
    
    level: AlertLevel
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: pd.Timestamp
    recommendations: List[str]


@dataclass
class DiversityConfig:
    """Configuration for diversity analyzer."""
    
    correlation_threshold: float = 0.9  # High correlation warning
    min_diversity_threshold: float = 0.1  # Minimum acceptable diversity
    diversity_metric: str = 'disagreement'  # 'disagreement', 'correlation', 'spread'
    enable_monitoring: bool = True
    enable_alerts: bool = True


class EnsembleDiversityAnalyzer:
    """
    Ensemble diversity and quality analyzer.
    
    Diversity Metrics:
    - Pairwise Correlation: E[corr(pred_i, pred_j)] for i≠j
    - Disagreement: E[|pred_i - pred_j|] for i≠j
    - Prediction Spread: Var(predictions) across models
    - Vote Entropy: H(predictions) for discretized predictions
    
    Quality Metrics:
    - Individual Accuracy: MAPE per model
    - Ensemble Accuracy: MAPE of combination
    - Improvement: (best_individual - ensemble) / best_individual
    
    Quality-Diversity Trade-Off:
        Pareto Frontier: {(accuracy, diversity) | no domination}
    
    Diversity Monitoring:
        Alert if: diversity < threshold OR correlation > threshold
    
    Example:
        >>> analyzer = EnsembleDiversityAnalyzer()
        >>> metrics = analyzer.calculate_diversity(predictions)
        >>> print(f"Average correlation: {metrics['avg_correlation']:.3f}")
        >>> print(f"Disagreement: {metrics['disagreement']:.3f}")
        >>> 
        >>> analysis = analyzer.analyze_quality_diversity(predictions, targets)
        >>> print(f"Optimal subset: {analysis['recommended_models']}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize diversity analyzer.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = {}
        
        config_obj = DiversityConfig(**config)
        self.correlation_threshold = config_obj.correlation_threshold
        self.min_diversity_threshold = config_obj.min_diversity_threshold
        self.diversity_metric = config_obj.diversity_metric
        self.enable_monitoring = config_obj.enable_monitoring
        self.enable_alerts = config_obj.enable_alerts
        
        # Storage
        self.diversity_history: List[Dict] = []
        self.alerts: List[DiversityAlert] = []
    
    def calculate_diversity(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive diversity metrics.
        
        Args:
            predictions: Model predictions
        
        Returns:
            Dictionary of diversity metrics
        """
        logger.info(f"Calculating diversity for {len(predictions)} models")
        
        model_names = list(predictions.keys())
        
        if len(model_names) < 2:
            logger.warning("Need at least 2 models for diversity analysis")
            return {'error': 'insufficient_models'}
        
        # Get prediction columns
        first_pred = predictions[model_names[0]]
        pred_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        metrics = {}
        
        # Pairwise correlations
        correlations = self.calculate_pairwise_correlations(predictions, pred_cols)
        metrics['correlations'] = correlations
        metrics['avg_correlation'] = np.mean([v for v in correlations.values()])
        
        # Disagreement
        disagreement = self.calculate_disagreement(predictions, pred_cols)
        metrics['disagreement'] = disagreement
        
        # Prediction spread
        spread = self.calculate_prediction_spread(predictions, pred_cols)
        metrics['spread'] = spread
        
        # Entropy
        entropy = self.calculate_diversity_entropy(predictions, pred_cols)
        metrics['entropy'] = entropy
        
        # Store history
        self.diversity_history.append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics
        })
        
        # Monitoring and alerts
        if self.enable_monitoring:
            self._check_diversity_alerts(metrics)
        
        return metrics
    
    def calculate_pairwise_correlations(
        self,
        predictions: Dict[str, pd.DataFrame],
        pred_cols: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate pairwise prediction correlations.
        
        Args:
            predictions: Model predictions
            pred_cols: Prediction columns
        
        Returns:
            Dictionary of pairwise correlations
        """
        model_names = list(predictions.keys())
        correlations = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # Only upper triangle
                    # Concatenate predictions across horizons
                    pred1 = np.concatenate([
                        predictions[model1][col].values for col in pred_cols
                    ])
                    pred2 = np.concatenate([
                        predictions[model2][col].values for col in pred_cols
                    ])
                    
                    # Calculate correlation
                    corr, _ = pearsonr(pred1, pred2)
                    correlations[(model1, model2)] = corr
        
        return correlations
    
    def calculate_disagreement(
        self,
        predictions: Dict[str, pd.DataFrame],
        pred_cols: List[str]
    ) -> float:
        """
        Calculate average pairwise disagreement.
        
        Args:
            predictions: Model predictions
            pred_cols: Prediction columns
        
        Returns:
            Average disagreement score
        """
        model_names = list(predictions.keys())
        disagreements = []
        
        for col in pred_cols:
            # Stack predictions for this horizon
            pred_matrix = np.column_stack([
                predictions[model][col].values for model in model_names
            ])
            
            # Calculate pairwise absolute differences
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    diff = np.abs(pred_matrix[:, i] - pred_matrix[:, j])
                    disagreements.append(np.mean(diff))
        
        return np.mean(disagreements)
    
    def calculate_prediction_spread(
        self,
        predictions: Dict[str, pd.DataFrame],
        pred_cols: List[str]
    ) -> Dict[str, float]:
        """
        Calculate prediction spread metrics.
        
        Args:
            predictions: Model predictions
            pred_cols: Prediction columns
        
        Returns:
            Dictionary of spread metrics
        """
        model_names = list(predictions.keys())
        spreads = []
        cvs = []
        ranges = []
        
        for col in pred_cols:
            # Stack predictions
            pred_matrix = np.column_stack([
                predictions[model][col].values for model in model_names
            ])
            
            # Calculate spread metrics per timestamp
            std_vals = np.std(pred_matrix, axis=1)
            mean_vals = np.mean(pred_matrix, axis=1)
            cv_vals = std_vals / (mean_vals + 1e-8)
            range_vals = np.ptp(pred_matrix, axis=1)
            
            spreads.extend(std_vals)
            cvs.extend(cv_vals)
            ranges.extend(range_vals)
        
        return {
            'std': np.mean(spreads),
            'cv': np.mean(cvs),
            'range': np.mean(ranges)
        }
    
    def calculate_diversity_entropy(
        self,
        predictions: Dict[str, pd.DataFrame],
        pred_cols: List[str]
    ) -> float:
        """
        Calculate diversity using entropy.
        
        Args:
            predictions: Model predictions
            pred_cols: Prediction columns
        
        Returns:
            Entropy score
        """
        model_names = list(predictions.keys())
        entropies = []
        
        for col in pred_cols:
            # Stack predictions
            pred_matrix = np.column_stack([
                predictions[model][col].values for model in model_names
            ])
            
            # Calculate entropy per timestamp
            for i in range(len(pred_matrix)):
                preds = pred_matrix[i, :]
                
                # Discretize into bins
                hist, _ = np.histogram(preds, bins=10)
                prob = hist / hist.sum()
                prob = prob[prob > 0]  # Remove zeros
                
                # Calculate entropy
                entropy = -np.sum(prob * np.log(prob))
                entropies.append(entropy)
        
        return np.mean(entropies)
    
    def analyze_quality_diversity_tradeoff(
        self,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze quality-diversity trade-off.
        
        Args:
            predictions: Model predictions
            targets: True targets
        
        Returns:
            Trade-off analysis results
        """
        from src.evaluation.metrics import calculate_mape
        
        model_names = list(predictions.keys())
        pred_cols = [col for col in next(iter(predictions.values())).columns 
                     if col.startswith('pred_h')]
        
        # Calculate quality (accuracy) per model
        quality_scores = {}
        for model in model_names:
            errors = []
            for col in pred_cols:
                horizon = int(col.split('_h')[1])
                target_col = f'target_h{horizon}'
                if target_col in targets.columns:
                    mape = calculate_mape(targets[target_col], predictions[model][col])
                    errors.append(mape)
            quality_scores[model] = np.mean(errors)
        
        # Calculate diversity contribution per model
        diversity_scores = {}
        for model in model_names:
            # Diversity = average disagreement with other models
            other_models = [m for m in model_names if m != model]
            disagreements = []
            
            for col in pred_cols:
                pred_model = predictions[model][col].values
                for other in other_models:
                    pred_other = predictions[other][col].values
                    disagreements.append(np.mean(np.abs(pred_model - pred_other)))
            
            diversity_scores[model] = np.mean(disagreements)
        
        # Find Pareto frontier
        pareto_models = self._find_pareto_frontier(quality_scores, diversity_scores)
        
        # Recommend model subset
        recommended = self._recommend_model_subset(
            quality_scores, diversity_scores, max_models=5
        )
        
        return {
            'quality_scores': quality_scores,
            'diversity_scores': diversity_scores,
            'pareto_frontier': pareto_models,
            'recommended_models': recommended
        }
    
    def _find_pareto_frontier(
        self,
        quality_scores: Dict[str, float],
        diversity_scores: Dict[str, float]
    ) -> List[str]:
        """Find Pareto optimal models (minimize error, maximize diversity)."""
        pareto = []
        
        for model in quality_scores.keys():
            dominated = False
            
            for other in quality_scores.keys():
                if model == other:
                    continue
                
                # Other dominates if: better quality AND better diversity
                if (quality_scores[other] <= quality_scores[model] and
                    diversity_scores[other] >= diversity_scores[model] and
                    (quality_scores[other] < quality_scores[model] or
                     diversity_scores[other] > diversity_scores[model])):
                    dominated = True
                    break
            
            if not dominated:
                pareto.append(model)
        
        return pareto
    
    def _recommend_model_subset(
        self,
        quality_scores: Dict[str, float],
        diversity_scores: Dict[str, float],
        max_models: int = 5
    ) -> List[str]:
        """Recommend optimal model subset."""
        # Start with best performing model
        selected = [min(quality_scores, key=quality_scores.get)]
        available = [m for m in quality_scores.keys() if m not in selected]
        
        # Iteratively add models maximizing diversity
        while len(selected) < max_models and available:
            best_addition = None
            best_score = -np.inf
            
            for candidate in available:
                # Calculate diversity gain
                diversity_gain = diversity_scores[candidate]
                quality_penalty = quality_scores[candidate]
                
                # Combined score (balance accuracy and diversity)
                score = diversity_gain - 0.5 * quality_penalty
                
                if score > best_score:
                    best_score = score
                    best_addition = candidate
            
            if best_addition:
                selected.append(best_addition)
                available.remove(best_addition)
            else:
                break
        
        return selected
    
    def _check_diversity_alerts(self, metrics: Dict[str, Any]):
        """Check diversity metrics and generate alerts."""
        if not self.enable_alerts:
            return
        
        # Check correlation
        avg_corr = metrics.get('avg_correlation', 0)
        if avg_corr > self.correlation_threshold:
            alert = DiversityAlert(
                level=AlertLevel.WARNING,
                message=f"High correlation detected: {avg_corr:.3f}",
                metric_name='avg_correlation',
                metric_value=avg_corr,
                threshold=self.correlation_threshold,
                timestamp=pd.Timestamp.now(),
                recommendations=[
                    "Consider removing highly correlated models",
                    "Add more diverse models to ensemble"
                ]
            )
            self.alerts.append(alert)
            logger.warning(alert.message)
        
        # Check diversity
        disagreement = metrics.get('disagreement', 0)
        if disagreement < self.min_diversity_threshold:
            alert = DiversityAlert(
                level=AlertLevel.CRITICAL,
                message=f"Low diversity detected: {disagreement:.3f}",
                metric_name='disagreement',
                metric_value=disagreement,
                threshold=self.min_diversity_threshold,
                timestamp=pd.Timestamp.now(),
                recommendations=[
                    "Ensemble is too homogeneous",
                    "Add models with different architectures",
                    "Check for overfitting to same patterns"
                ]
            )
            self.alerts.append(alert)
            logger.critical(alert.message)
    
    def get_diversity_report(self) -> str:
        """Generate comprehensive diversity report."""
        if not self.diversity_history:
            return "No diversity analysis performed yet."
        
        latest = self.diversity_history[-1]
        metrics = latest['metrics']
        
        report = "=== Ensemble Diversity Report ===\n\n"
        report += f"Timestamp: {latest['timestamp']}\n\n"
        report += f"Average Correlation: {metrics.get('avg_correlation', 0):.3f}\n"
        report += f"Disagreement: {metrics.get('disagreement', 0):.3f}\n"
        report += f"Spread (Std): {metrics.get('spread', {}).get('std', 0):.3f}\n"
        report += f"Entropy: {metrics.get('entropy', 0):.3f}\n\n"
        
        if self.alerts:
            report += "=== Alerts ===\n"
            for alert in self.alerts[-5:]:  # Last 5 alerts
                report += f"[{alert.level.value.upper()}] {alert.message}\n"
        
        return report
```

---

## 🧪 Testing & Validation

```python
"""Tests for diversity analyzer."""
import pytest
import pandas as pd
import numpy as np

from src.models.combination.diversity_analyzer import EnsembleDiversityAnalyzer


@pytest.fixture
def sample_diverse_predictions():
    """Create predictions with varying diversity."""
    dates = pd.date_range('2024-01-01', periods=100, freq='30min')
    
    # Diverse models
    predictions_diverse = {
        'model1': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 50 + 1000
        }, index=dates),
        'model2': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 60 + 1010
        }, index=dates),
        'model3': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 40 + 990
        }, index=dates)
    }
    
    # Homogeneous models (highly correlated)
    base = np.random.randn(100) * 50 + 1000
    predictions_homogeneous = {
        'model1': pd.DataFrame({'pred_h0': base + np.random.randn(100) * 5}, index=dates),
        'model2': pd.DataFrame({'pred_h0': base + np.random.randn(100) * 5}, index=dates),
        'model3': pd.DataFrame({'pred_h0': base + np.random.randn(100) * 5}, index=dates)
    }
    
    return predictions_diverse, predictions_homogeneous


def test_diversity_calculation(sample_diverse_predictions):
    """Test diversity metric calculation."""
    predictions_diverse, _ = sample_diverse_predictions
    
    analyzer = EnsembleDiversityAnalyzer()
    metrics = analyzer.calculate_diversity(predictions_diverse)
    
    assert 'avg_correlation' in metrics
    assert 'disagreement' in metrics
    assert 'spread' in metrics


def test_high_vs_low_diversity(sample_diverse_predictions):
    """Test that analyzer distinguishes high vs low diversity."""
    predictions_diverse, predictions_homogeneous = sample_diverse_predictions
    
    analyzer = EnsembleDiversityAnalyzer()
    
    metrics_diverse = analyzer.calculate_diversity(predictions_diverse)
    metrics_homo = analyzer.calculate_diversity(predictions_homogeneous)
    
    # Diverse should have lower correlation
    assert metrics_diverse['avg_correlation'] < metrics_homo['avg_correlation']
    
    # Diverse should have higher disagreement
    assert metrics_diverse['disagreement'] > metrics_homo['disagreement']


def test_diversity_alerts(sample_diverse_predictions):
    """Test diversity alert system."""
    _, predictions_homogeneous = sample_diverse_predictions
    
    analyzer = EnsembleDiversityAnalyzer(config={
        'correlation_threshold': 0.8,
        'enable_alerts': True
    })
    
    analyzer.calculate_diversity(predictions_homogeneous)
    
    # Should trigger high correlation alert
    assert len(analyzer.alerts) > 0
```

---

## 📝 Technical Notes

### Diversity Metrics
- **Correlation:** Measures prediction similarity
- **Disagreement:** Measures prediction differences
- **Entropy:** Measures prediction distribution

### Quality-Diversity Trade-Off
- Pareto frontier identifies optimal models
- Balance accuracy with diversity

---

## 🔗 Dependencies

**Depends On:**
- PC-036-05A: Base Combiner Interface
- All combination methods (PC-037 through PC-044)

**Blocks:**
- PC-047-05B: Performance-Based Selector
- Epic-06A: Hierarchical reconciliation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Multiple diversity metrics calculated
- [ ] Pairwise correlations working
- [ ] Disagreement metrics functional
- [ ] Quality-diversity trade-off analysis complete
- [ ] Diversity monitoring operational
- [ ] Alert system triggering appropriately
- [ ] Diversity-guided selection working
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-05B: Advanced Ensemble Methods](../epics/Epic-05B.md)  
**Previous:** [PC-045-05B: Advanced Bias Correction](PC-045-05B-advanced-bias-correction.md)  
**Next:** [PC-047-05B: Performance-Based Model Selection](PC-047-05B-performance-based-selector.md)
