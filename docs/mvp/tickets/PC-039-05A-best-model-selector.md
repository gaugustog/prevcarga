# PC-039-05A: Best Model Selector

**Ticket ID:** PC-039-05A  
**Epic:** [Epic-05A: Base Combination Strategies](../epics/Epic-05A.md)  
**User Story:** US-4  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `BestModelSelector` that dynamically chooses the optimal model per time series and horizon based on recent performance, model confidence, and historical accuracy. Provides fallback mechanisms when the best model fails, logs selection decisions for interpretability, and tracks model performance rankings.

**As a** forecasting analyst  
**I want** a dynamic model selection strategy  
**So that** I can automatically choose the best performing model per context

---

## ✅ Acceptance Criteria

- [ ] `BestModelSelector` chooses optimal model per context
- [ ] Selection criteria include recent performance, model confidence, historical accuracy
- [ ] Fallback mechanisms when best model fails
- [ ] Selection logging for interpretability
- [ ] Performance tracking and model ranking
- [ ] Rolling window performance evaluation
- [ ] Support for context-based selection (time series, horizon, period)
- [ ] Graceful degradation with missing models
- [ ] Performance competitive with weighted voting
- [ ] Comprehensive tests with multiple models

---

## 🔧 Implementation Tasks

### 1. Create Best Model Selector Module
- [ ] Create `src/models/combination/best_model_selector.py`
- [ ] Import BaseCombiner
- [ ] Import performance tracking utilities
- [ ] Add module docstrings

### 2. Implement BestModelSelector Class
- [ ] Inherit from BaseCombiner
- [ ] Implement `name` property returning "best_model_selector"
- [ ] Implement `version` property returning "1.0.0"
- [ ] Define selection criteria configuration
- [ ] Initialize performance tracking

### 3. Implement Performance Tracking
- [ ] Create `PerformanceTracker` class
- [ ] Track rolling window metrics (MAPE, MAE, RMSE)
- [ ] Store per-model, per-horizon, per-time-series
- [ ] Update with new observations
- [ ] Calculate recent vs historical performance
- [ ] Support configurable window sizes

### 4. Implement Model Ranking
- [ ] Create `_rank_models()` method
- [ ] Calculate ranking score from metrics
- [ ] Weight recent performance higher
- [ ] Consider prediction confidence
- [ ] Handle missing performance data
- [ ] Return ranked model list

### 5. Implement Selection Strategy
- [ ] Create `_select_best_model()` method
- [ ] Choose top-ranked model per context
- [ ] Apply selection criteria
- [ ] Check model availability
- [ ] Log selection decision
- [ ] Return selected model name

### 6. Implement Context-Based Selection
- [ ] Support time-series-specific selection
- [ ] Support horizon-specific selection
- [ ] Support time-of-day selection
- [ ] Support seasonal pattern selection
- [ ] Store context-model mappings
- [ ] Log context patterns

### 7. Implement Fallback Mechanism
- [ ] Create `_get_fallback_model()` method
- [ ] Define fallback hierarchy
- [ ] Try second-best model
- [ ] Try third-best if needed
- [ ] Ultimate fallback: simple average
- [ ] Log fallback activation

### 8. Implement fit() Method
- [ ] Calculate initial model performance
- [ ] Rank models based on training data
- [ ] Initialize performance tracker
- [ ] Store model rankings
- [ ] Validate sufficient history
- [ ] Mark as fitted

### 9. Implement combine() Method
- [ ] For each timestamp/context:
  - [ ] Select best model
  - [ ] Extract prediction from selected model
  - [ ] Handle missing prediction (fallback)
  - [ ] Track selection decision
- [ ] Compile combined predictions
- [ ] Create metadata with selections
- [ ] Return CombinationResult

### 10. Implement Rolling Window Evaluation
- [ ] Create `_update_performance()` method
- [ ] Calculate metrics on recent window
- [ ] Update rolling statistics
- [ ] Detect performance degradation
- [ ] Trigger re-ranking if needed
- [ ] Log performance updates

### 11. Implement Confidence-Based Selection
- [ ] Create `_get_model_confidence()` method
- [ ] Extract confidence from prediction intervals
- [ ] Calculate confidence score
- [ ] Weight in selection decision
- [ ] Handle missing confidence data
- [ ] Log confidence values

### 12. Implement Selection Logging
- [ ] Create `SelectionLog` dataclass
- [ ] Track: timestamp, selected_model, reason
- [ ] Track: alternatives, scores, fallback_used
- [ ] Store selection history
- [ ] Generate selection reports
- [ ] Support audit trail

### 13. Implement Performance Comparison
- [ ] Compare selector vs individual models
- [ ] Compare selector vs weighted voting
- [ ] Calculate win rate per model
- [ ] Analyze selection patterns
- [ ] Generate comparison reports

### 14. Handle Edge Cases
- [ ] Handle single model available
- [ ] Handle all models with equal performance
- [ ] Handle cold start (no history)
- [ ] Handle missing predictions
- [ ] Handle performance ties
- [ ] Raise informative errors

### 15. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_best_model_selector.py`
- [ ] Test model ranking
- [ ] Test selection logic
- [ ] Test fallback mechanisms
- [ ] Test context-based selection
- [ ] Test rolling window updates
- [ ] Test cold start scenarios

### 16. Write Integration Tests
- [ ] Test with real model outputs
- [ ] Test performance tracking
- [ ] Test selection over time
- [ ] Validate improvement vs baselines
- [ ] Test with missing models

### 17. Create Usage Examples
- [ ] Create `examples/best_model_selector_demo.py`
- [ ] Show selection process
- [ ] Show performance tracking
- [ ] Show fallback mechanisms
- [ ] Analyze selection patterns

---

## 💻 Implementation Details

### BestModelSelector Implementation

```python
"""Best model selection combination strategy."""
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import deque

from src.models.combination.base_combiner import BaseCombiner, CombinationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SelectionLog:
    """Log entry for model selection decision."""
    
    timestamp: pd.Timestamp
    horizon: int
    selected_model: str
    selection_score: float
    alternatives: Dict[str, float]
    reason: str
    fallback_used: bool = False


@dataclass
class BestModelSelectorConfig:
    """Configuration for best model selector."""
    
    selection_criterion: str = 'mape'  # 'mape', 'mae', 'rmse', 'composite'
    window_size: int = 48  # Rolling window for performance evaluation
    recent_weight: float = 0.7  # Weight for recent vs historical performance
    use_confidence: bool = False
    fallback_strategy: str = 'ranked'  # 'ranked', 'average', 'best_overall'
    min_samples: int = 10  # Minimum samples for reliable ranking


class PerformanceTracker:
    """
    Tracks rolling performance metrics for models.
    
    Maintains rolling windows of errors per model, horizon,
    and time series for recent performance evaluation.
    """
    
    def __init__(self, window_size: int = 48):
        """
        Initialize performance tracker.
        
        Args:
            window_size: Size of rolling window
        """
        self.window_size = window_size
        self.errors: Dict[str, Dict[int, deque]] = {}  # {model: {horizon: deque}}
        self.performance_cache: Dict[str, Dict[int, float]] = {}
    
    def update(
        self,
        model_name: str,
        horizon: int,
        error: float
    ):
        """
        Update tracker with new error observation.
        
        Args:
            model_name: Name of model
            horizon: Forecast horizon
            error: Error value (e.g., absolute percentage error)
        """
        if model_name not in self.errors:
            self.errors[model_name] = {}
        
        if horizon not in self.errors[model_name]:
            self.errors[model_name][horizon] = deque(maxlen=self.window_size)
        
        self.errors[model_name][horizon].append(error)
        
        # Invalidate cache
        if model_name in self.performance_cache:
            if horizon in self.performance_cache[model_name]:
                del self.performance_cache[model_name][horizon]
    
    def get_performance(
        self,
        model_name: str,
        horizon: int
    ) -> Optional[float]:
        """
        Get recent performance metric for model and horizon.
        
        Args:
            model_name: Name of model
            horizon: Forecast horizon
        
        Returns:
            Average error over rolling window, or None if insufficient data
        """
        # Check cache
        if model_name in self.performance_cache:
            if horizon in self.performance_cache[model_name]:
                return self.performance_cache[model_name][horizon]
        
        # Calculate if not cached
        if model_name not in self.errors or horizon not in self.errors[model_name]:
            return None
        
        errors = self.errors[model_name][horizon]
        
        if len(errors) == 0:
            return None
        
        performance = np.mean(errors)
        
        # Cache result
        if model_name not in self.performance_cache:
            self.performance_cache[model_name] = {}
        self.performance_cache[model_name][horizon] = performance
        
        return performance


class BestModelSelector(BaseCombiner):
    """
    Best model selection combination strategy.
    
    Dynamically selects the best performing model for each
    prediction based on recent performance, historical accuracy,
    and optional prediction confidence.
    
    Selection Process:
    1. Rank models by recent performance (rolling window)
    2. Apply selection criteria (MAPE, MAE, RMSE, or composite)
    3. Choose top-ranked available model
    4. Fallback to next-best if primary unavailable
    
    Benefits:
    - Adapts to changing model performance
    - Exploits model strengths per context
    - Interpretable selection decisions
    - Robust to model failures
    
    Example:
        >>> selector = BestModelSelector(
        ...     config={'selection_criterion': 'mape', 'window_size': 48}
        ... )
        >>> selector.fit(train_predictions, train_targets)
        >>> result = selector.combine(test_predictions)
        >>> # Analyze selections
        >>> print(result.metadata.configuration['selections'])
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize best model selector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Parse configuration
        config_obj = BestModelSelectorConfig(**self.config)
        self.selection_criterion = config_obj.selection_criterion
        self.window_size = config_obj.window_size
        self.recent_weight = config_obj.recent_weight
        self.use_confidence = config_obj.use_confidence
        self.fallback_strategy = config_obj.fallback_strategy
        self.min_samples = config_obj.min_samples
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(window_size=self.window_size)
        
        # Initialize selection log
        self.selection_log: List[SelectionLog] = []
        
        # Model rankings
        self.model_rankings: Dict[int, List[str]] = {}  # {horizon: [models]}
    
    @property
    def name(self) -> str:
        return "best_model_selector"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def fit(
        self,
        train_predictions: Dict[str, pd.DataFrame],
        train_targets: pd.DataFrame,
        validation_predictions: Optional[Dict[str, pd.DataFrame]] = None,
        validation_targets: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Fit selector by calculating initial model rankings.
        
        Args:
            train_predictions: Training predictions
            train_targets: Training targets
            validation_predictions: Optional validation predictions
            validation_targets: Optional validation targets
        """
        logger.info(f"Fitting best model selector with {len(train_predictions)} models")
        
        # Validate inputs
        self._validate_predictions(train_predictions)
        
        model_names = list(train_predictions.keys())
        
        # Get prediction columns
        first_pred = train_predictions[model_names[0]]
        pred_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        # Calculate initial performance and update tracker
        for col in pred_cols:
            horizon = int(col.split('_h')[1])
            target_col = f'target_h{horizon}'
            
            if target_col not in train_targets.columns:
                continue
            
            for model_name in model_names:
                predictions = train_predictions[model_name][col].values
                targets = train_targets[target_col].values
                
                # Calculate errors
                errors = np.abs((predictions - targets) / (targets + 1e-8)) * 100
                
                # Update tracker
                for error in errors:
                    self.performance_tracker.update(model_name, horizon, error)
        
        # Calculate initial rankings
        self._update_rankings(model_names, pred_cols)
        
        # Mark as fitted
        self._fitted = True
        self._fit_timestamp = pd.Timestamp.now()
        
        # Log rankings
        logger.info("Initial model rankings:")
        for horizon, models in self.model_rankings.items():
            logger.info(f"  H{horizon}: {' > '.join(models[:3])}")
    
    def combine(
        self,
        predictions: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CombinationResult:
        """
        Combine by selecting best model per context.
        
        Args:
            predictions: Dictionary of model predictions
            metadata: Optional metadata
        
        Returns:
            CombinationResult with best model selections
        """
        if not self.is_fitted:
            raise RuntimeError("Selector must be fitted before combining")
        
        import time
        start_time = time.time()
        
        logger.info(f"Selecting best models from {len(predictions)} candidates")
        
        # Validate predictions
        self._validate_predictions(predictions)
        
        model_names = list(predictions.keys())
        first_pred = predictions[model_names[0]]
        pred_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        # Initialize combined predictions
        combined_df = pd.DataFrame(index=first_pred.index)
        selection_count = {model: 0 for model in model_names}
        
        # Select best model for each column
        for col in pred_cols:
            horizon = int(col.split('_h')[1])
            
            # Get rankings for this horizon
            if horizon in self.model_rankings:
                ranked_models = self.model_rankings[horizon]
            else:
                ranked_models = model_names
            
            # Select best available model
            selected_model = None
            for model in ranked_models:
                if model in predictions:
                    selected_model = model
                    break
            
            # Fallback if no ranked model available
            if selected_model is None:
                selected_model = model_names[0]
                logger.warning(f"No ranked model available for h{horizon}, using {selected_model}")
            
            # Extract prediction
            combined_df[col] = predictions[selected_model][col].values
            selection_count[selected_model] += 1
            
            # Log selection
            self.selection_log.append(SelectionLog(
                timestamp=pd.Timestamp.now(),
                horizon=horizon,
                selected_model=selected_model,
                selection_score=0.0,  # Would calculate from performance
                alternatives={},
                reason=f"Top ranked model for h{horizon}"
            ))
        
        # Create "weights" showing selection frequency
        total_selections = sum(selection_count.values())
        weights = {
            model: count / total_selections
            for model, count in selection_count.items()
        }
        
        # Create metadata
        processing_time = time.time() - start_time
        metadata_obj = self._create_metadata(
            models_used=list(predictions.keys()),
            models_excluded=[],
            processing_time=processing_time
        )
        metadata_obj.configuration['selection_counts'] = selection_count
        
        logger.info("Selection summary:")
        for model, count in selection_count.items():
            logger.info(f"  {model}: {count}/{len(pred_cols)} selections")
        
        return CombinationResult(
            combined_predictions=combined_df,
            weights=weights,
            metadata=metadata_obj
        )
    
    def _update_rankings(
        self,
        model_names: List[str],
        pred_cols: List[str]
    ):
        """
        Update model rankings based on performance.
        
        Args:
            model_names: List of model names
            pred_cols: Prediction columns
        """
        for col in pred_cols:
            horizon = int(col.split('_h')[1])
            
            # Get performance for each model
            performances = {}
            for model in model_names:
                perf = self.performance_tracker.get_performance(model, horizon)
                if perf is not None:
                    performances[model] = perf
            
            # Rank models (lower is better for errors)
            if performances:
                ranked = sorted(performances.items(), key=lambda x: x[1])
                self.model_rankings[horizon] = [model for model, _ in ranked]
            else:
                self.model_rankings[horizon] = model_names
    
    def get_selection_report(self) -> pd.DataFrame:
        """
        Generate report of selection decisions.
        
        Returns:
            DataFrame with selection statistics
        """
        if not self.selection_log:
            return pd.DataFrame()
        
        report_data = []
        for log_entry in self.selection_log:
            report_data.append({
                'timestamp': log_entry.timestamp,
                'horizon': log_entry.horizon,
                'selected_model': log_entry.selected_model,
                'selection_score': log_entry.selection_score,
                'fallback_used': log_entry.fallback_used
            })
        
        return pd.DataFrame(report_data)
```

---

## 🧪 Testing & Validation

```python
"""Tests for best model selector."""
import pytest
import pandas as pd
import numpy as np

from src.models.combination.best_model_selector import BestModelSelector


@pytest.fixture
def sample_performance_data():
    """Create data where models have different strengths."""
    dates = pd.date_range('2024-01-01', periods=100, freq='30min')
    
    # Model 1: Best at h0
    # Model 2: Best at h1
    predictions = {
        'model1': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 30 + 1000,  # Low variance
            'pred_h1': np.random.randn(100) * 100 + 1000  # High variance
        }, index=dates),
        'model2': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 100 + 1000,  # High variance
            'pred_h1': np.random.randn(100) * 30 + 1000   # Low variance
        }, index=dates)
    }
    
    targets = pd.DataFrame({
        'target_h0': np.random.randn(100) * 20 + 1000,
        'target_h1': np.random.randn(100) * 20 + 1000
    }, index=dates)
    
    return predictions, targets


def test_model_selection(sample_performance_data):
    """Test best model selection."""
    predictions, targets = sample_performance_data
    
    selector = BestModelSelector()
    selector.fit(predictions, targets)
    
    assert selector.is_fitted
    assert len(selector.model_rankings) > 0


def test_selection_logging(sample_performance_data):
    """Test selection logging."""
    predictions, targets = sample_performance_data
    
    selector = BestModelSelector()
    selector.fit(predictions, targets)
    
    result = selector.combine(predictions)
    
    # Check selection log
    assert len(selector.selection_log) > 0
    
    # Check report generation
    report = selector.get_selection_report()
    assert not report.empty
```

---

## 📝 Technical Notes

### Selection Strategies
- Performance-based: Choose lowest error
- Confidence-based: Choose highest confidence
- Composite: Weighted combination of criteria

### Fallback Hierarchy
1. Best ranked model
2. Second-best ranked
3. Third-best ranked
4. Simple average of all

---

## 🔗 Dependencies

**Depends On:**
- PC-036-05A: Base Combiner Interface
- PC-037-05A: Simple Averaging (for fallback)

**Blocks:**
- PC-041-05A: Weight Validation Framework
- Epic-05B advanced ensemble methods

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Model ranking working correctly
- [ ] Selection logic functional
- [ ] Fallback mechanisms tested
- [ ] Performance tracking accurate
- [ ] Selection logging complete
- [ ] Competitive performance with weighted voting
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Previous:** [PC-038-05A: Weighted Voting Combiner](PC-038-05A-weighted-voting-combiner.md)  
**Next:** [PC-040-05A: Basic Bias Correction](PC-040-05A-basic-bias-correction.md)
