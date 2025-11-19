# PC-043-05B: Markov Chain Dynamic Weighting

**Ticket ID:** PC-043-05B  
**Epic:** [Epic-05B: Advanced Ensemble Methods](../epics/Epic-05B.md)  
**User Story:** US-2  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `MarkovChainCombiner` with Hidden Markov Model (HMM) for dynamic weight adaptation based on performance states. Models transition between performance states (excellent, good, fair, poor) and weights adapt based on current state, transition probabilities, and context features. Enables temporal patterns in model performance to influence combination weights.

**As a** forecasting analyst  
**I want** dynamic model weights that adapt based on recent performance and context  
**So that** I can leverage temporal patterns in model performance for better combinations

---

## ✅ Acceptance Criteria

- [ ] `MarkovChainCombiner` implements state-based dynamic weighting
- [ ] Model performance states (excellent, good, fair, poor) defined and tracked
- [ ] Transition probabilities learned from historical performance patterns
- [ ] Context features influence state transitions and weight adaptation
- [ ] Weights adapt smoothly without erratic switching
- [ ] HMM with Gaussian emissions for performance modeling
- [ ] Rolling window performance tracking (48 periods)
- [ ] Weight smoothing to prevent rapid changes
- [ ] Integration with BaseCombiner interface
- [ ] Comprehensive tests with state transitions

---

## 🔧 Implementation Tasks

### 1. Setup Markov Chain Module
- [ ] Create `src/models/combination/markov_chain.py`
- [ ] Import hmmlearn for HMM
- [ ] Import scipy for statistics
- [ ] Import BaseCombiner
- [ ] Add module docstrings

### 2. Implement MarkovChainCombiner Class
- [ ] Inherit from BaseCombiner
- [ ] Implement `name` property returning "markov_chain"
- [ ] Implement `version` property returning "1.0.0"
- [ ] Define configuration schema
- [ ] Initialize HMM model

### 3. Define Performance States
- [ ] Create `PerformanceState` enum (EXCELLENT, GOOD, FAIR, POOR)
- [ ] Create `_define_state_thresholds()` method
- [ ] Map MAPE ranges to states
- [ ] Configure state count (default 4)
- [ ] Store state definitions

### 4. Implement Performance Tracking
- [ ] Create `PerformanceTracker` class
- [ ] Track rolling window errors (deque)
- [ ] Calculate current performance metrics
- [ ] Classify current performance state
- [ ] Update performance history

### 5. Implement State Classification
- [ ] Create `_classify_performance()` method
- [ ] Calculate rolling MAPE per model
- [ ] Map MAPE to performance state
- [ ] Handle edge cases (insufficient data)
- [ ] Return state for each model

### 6. Implement HMM Training
- [ ] Create `_train_hmm()` method
- [ ] Prepare observation sequences from performance history
- [ ] Configure HMM (n_components=4 states)
- [ ] Fit HMM using Baum-Welch algorithm
- [ ] Store transition matrix
- [ ] Store emission parameters

### 7. Implement Transition Probability Learning
- [ ] Create `_learn_transition_probabilities()` method
- [ ] Analyze state sequences over time
- [ ] Calculate P(state_t | state_t-1)
- [ ] Smooth probabilities (Laplace smoothing)
- [ ] Validate transition matrix (rows sum to 1)

### 8. Implement Context-Aware Transitions
- [ ] Create `_context_aware_transition()` method
- [ ] Extract context features (hour, dow, temp)
- [ ] Condition transitions on context
- [ ] Calculate P(state_t | state_t-1, context)
- [ ] Return adjusted transition probabilities

### 9. Implement fit() Method
- [ ] Validate input predictions and targets
- [ ] Initialize performance tracking per model
- [ ] Build performance state sequences
- [ ] Train HMM for each model
- [ ] Learn transition probabilities
- [ ] Calculate initial state distribution
- [ ] Store fitted HMM models

### 10. Implement Dynamic Weight Calculation
- [ ] Create `_calculate_dynamic_weights()` method
- [ ] Get current performance state per model
- [ ] Calculate weights based on states
- [ ] Weight function: higher for better states
- [ ] Normalize weights to sum to 1
- [ ] Return weight dictionary

### 11. Implement Weight Smoothing
- [ ] Create `_smooth_weights()` method
- [ ] Store previous weights
- [ ] Apply exponential smoothing: w_t = α*w_new + (1-α)*w_prev
- [ ] Configure smoothing factor α (default 0.3)
- [ ] Prevent erratic weight changes
- [ ] Return smoothed weights

### 12. Implement combine() Method
- [ ] Validate input predictions
- [ ] Check if HMM is fitted
- [ ] Update performance tracking with recent data
- [ ] Classify current performance states
- [ ] Calculate dynamic weights
- [ ] Apply weight smoothing
- [ ] Combine predictions using weights
- [ ] Create metadata
- [ ] Return CombinationResult

### 13. Implement State Prediction
- [ ] Create `_predict_next_state()` method
- [ ] Use HMM forward algorithm
- [ ] Predict most likely next state
- [ ] Consider context features
- [ ] Return predicted state distribution

### 14. Implement Weight Adaptation Diagnostics
- [ ] Create `get_adaptation_diagnostics()` method
- [ ] Return current states per model
- [ ] Return transition probabilities
- [ ] Return weight evolution over time
- [ ] Return state change history

### 15. Handle Edge Cases
- [ ] Handle insufficient performance history
- [ ] Handle constant performance (no transitions)
- [ ] Handle missing context features
- [ ] Validate HMM convergence
- [ ] Fallback to static weights if needed

### 16. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_markov_chain.py`
- [ ] Test performance state classification
- [ ] Test HMM training and convergence
- [ ] Test transition probability learning
- [ ] Test dynamic weight calculation
- [ ] Test weight smoothing
- [ ] Test context-aware transitions

### 17. Write Integration Tests
- [ ] Test with real performance sequences
- [ ] Test adaptation over time
- [ ] Test with different state configurations
- [ ] Validate smooth weight transitions
- [ ] Test computational performance

### 18. Create Usage Examples
- [ ] Create `examples/markov_chain_demo.py`
- [ ] Show HMM training workflow
- [ ] Show dynamic weight adaptation
- [ ] Show state transition analysis
- [ ] Compare with static weighting

---

## 💻 Implementation Details

### MarkovChainCombiner Implementation

```python
"""Markov Chain dynamic weighting combiner."""
from typing import Dict, Any, Optional, List
from enum import Enum
from collections import deque
import pandas as pd
import numpy as np
from hmmlearn import hmm
from dataclasses import dataclass

from src.models.combination.base_combiner import BaseCombiner, CombinationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceState(Enum):
    """Performance state classifications."""
    EXCELLENT = 0
    GOOD = 1
    FAIR = 2
    POOR = 3


@dataclass
class MarkovChainConfig:
    """Configuration for Markov Chain combiner."""
    
    n_states: int = 4
    window_size: int = 48  # Rolling window for performance tracking
    smoothing_factor: float = 0.3  # Weight smoothing (0=no change, 1=instant)
    state_thresholds: Dict[str, float] = None  # MAPE thresholds for states
    use_context_features: bool = True
    hmm_n_iter: int = 100
    hmm_tol: float = 1e-4


class PerformanceTracker:
    """Track model performance over time."""
    
    def __init__(self, window_size: int = 48):
        self.window_size = window_size
        self.errors: deque = deque(maxlen=window_size)
        self.states: deque = deque(maxlen=window_size)
    
    def update(self, error: float, state: PerformanceState):
        """Update performance tracking."""
        self.errors.append(error)
        self.states.append(state)
    
    def get_current_performance(self) -> float:
        """Get current rolling MAPE."""
        if len(self.errors) == 0:
            return np.inf
        return np.mean(list(self.errors))
    
    def get_state_sequence(self) -> List[int]:
        """Get sequence of performance states."""
        return [state.value for state in self.states]


class MarkovChainCombiner(BaseCombiner):
    """
    Dynamic weight adaptation using Hidden Markov Models.
    
    Performance States:
    - EXCELLENT: MAPE < 2%
    - GOOD: MAPE 2-4%
    - FAIR: MAPE 4-8%
    - POOR: MAPE > 8%
    
    State Transitions:
        P(state_t | state_t-1, context_t)
    
    Weight Calculation:
        w_i = f(current_state_i, transition_probs, context)
    
    Weight Smoothing:
        w_t = α * w_new + (1 - α) * w_prev
    
    Benefits:
    - Adapts to changing model performance
    - Leverages temporal performance patterns
    - Smooth weight transitions
    - Context-aware adaptation
    
    Example:
        >>> mc_combiner = MarkovChainCombiner(
        ...     config={'n_states': 4, 'smoothing_factor': 0.3}
        ... )
        >>> mc_combiner.fit(train_predictions, train_targets)
        >>> result = mc_combiner.combine(test_predictions)
        >>> print(mc_combiner.get_current_states())
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Markov Chain combiner.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Default state thresholds
        if self.config.get('state_thresholds') is None:
            self.config['state_thresholds'] = {
                'excellent': 2.0,
                'good': 4.0,
                'fair': 8.0
            }
        
        config_obj = MarkovChainConfig(**self.config)
        self.n_states = config_obj.n_states
        self.window_size = config_obj.window_size
        self.smoothing_factor = config_obj.smoothing_factor
        self.state_thresholds = config_obj.state_thresholds
        self.use_context_features = config_obj.use_context_features
        self.hmm_n_iter = config_obj.hmm_n_iter
        self.hmm_tol = config_obj.hmm_tol
        
        # State tracking
        self.hmm_models: Dict[str, hmm.GaussianHMM] = {}
        self.performance_trackers: Dict[str, PerformanceTracker] = {}
        self.transition_matrices: Dict[str, np.ndarray] = {}
        self.previous_weights: Optional[Dict[str, float]] = None
    
    @property
    def name(self) -> str:
        return "markov_chain"
    
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
        Fit Markov Chain combiner by learning performance states and transitions.
        
        Args:
            train_predictions: Base model predictions
            train_targets: True target values
            validation_predictions: Optional validation predictions
            validation_targets: Optional validation targets
        """
        logger.info(f"Fitting Markov Chain combiner with {self.n_states} states")
        
        # Validate inputs
        self._validate_predictions(train_predictions)
        
        model_names = list(train_predictions.keys())
        
        # Initialize performance trackers
        for model_name in model_names:
            self.performance_trackers[model_name] = PerformanceTracker(
                window_size=self.window_size
            )
        
        # Get prediction columns
        first_pred = train_predictions[model_names[0]]
        pred_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        # Build performance sequences
        performance_sequences = {model: [] for model in model_names}
        
        for idx in range(len(first_pred)):
            for model_name in model_names:
                # Calculate errors for this timestamp
                errors = []
                for col in pred_cols:
                    horizon = int(col.split('_h')[1])
                    target_col = f'target_h{horizon}'
                    
                    if target_col in train_targets.columns:
                        pred_val = train_predictions[model_name][col].iloc[idx]
                        true_val = train_targets[target_col].iloc[idx]
                        
                        if true_val != 0:
                            mape = np.abs((true_val - pred_val) / true_val) * 100
                            errors.append(mape)
                
                if errors:
                    avg_error = np.mean(errors)
                    state = self._classify_performance(avg_error)
                    performance_sequences[model_name].append(state.value)
                    self.performance_trackers[model_name].update(avg_error, state)
        
        # Train HMM for each model
        for model_name in model_names:
            logger.info(f"Training HMM for {model_name}")
            
            sequence = np.array(performance_sequences[model_name]).reshape(-1, 1)
            
            # Create and train HMM
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=self.hmm_n_iter,
                tol=self.hmm_tol,
                random_state=42
            )
            
            model.fit(sequence)
            
            self.hmm_models[model_name] = model
            self.transition_matrices[model_name] = model.transmat_
            
            logger.info(f"  Converged: {model.monitor_.converged}")
        
        # Mark as fitted
        self._fitted = True
        self._fit_timestamp = pd.Timestamp.now()
        
        logger.info("Markov Chain combiner trained successfully")
    
    def combine(
        self,
        predictions: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CombinationResult:
        """
        Combine predictions using dynamic Markov Chain weights.
        
        Args:
            predictions: Base model predictions
            metadata: Optional metadata
        
        Returns:
            CombinationResult with dynamically weighted predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Markov Chain combiner must be fitted before combining")
        
        import time
        start_time = time.time()
        
        logger.info(f"Combining {len(predictions)} models with Markov Chain weights")
        
        # Validate predictions
        self._validate_predictions(predictions)
        
        # Get prediction columns
        first_pred = next(iter(predictions.values()))
        pred_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        # Calculate dynamic weights
        dynamic_weights = self._calculate_dynamic_weights(predictions)
        
        # Apply weight smoothing
        if self.previous_weights is not None:
            dynamic_weights = self._smooth_weights(dynamic_weights, self.previous_weights)
        
        self.previous_weights = dynamic_weights.copy()
        
        # Combine predictions
        combined_df = pd.DataFrame(index=first_pred.index)
        
        for col in pred_cols:
            weighted_sum = np.zeros(len(first_pred))
            
            for model_name, weight in dynamic_weights.items():
                weighted_sum += weight * predictions[model_name][col].values
            
            combined_df[col] = weighted_sum
        
        # Create metadata
        processing_time = time.time() - start_time
        metadata_obj = self._create_metadata(
            models_used=list(predictions.keys()),
            models_excluded=[],
            processing_time=processing_time
        )
        metadata_obj.configuration['current_states'] = self.get_current_states()
        
        return CombinationResult(
            combined_predictions=combined_df,
            weights=dynamic_weights,
            metadata=metadata_obj
        )
    
    def _classify_performance(self, mape: float) -> PerformanceState:
        """
        Classify MAPE into performance state.
        
        Args:
            mape: Mean Absolute Percentage Error
        
        Returns:
            Performance state
        """
        if mape < self.state_thresholds['excellent']:
            return PerformanceState.EXCELLENT
        elif mape < self.state_thresholds['good']:
            return PerformanceState.GOOD
        elif mape < self.state_thresholds['fair']:
            return PerformanceState.FAIR
        else:
            return PerformanceState.POOR
    
    def _calculate_dynamic_weights(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calculate dynamic weights based on current performance states.
        
        Args:
            predictions: Model predictions
        
        Returns:
            Dictionary of dynamic weights
        """
        model_names = list(predictions.keys())
        
        # Get current performance for each model
        current_performances = {}
        for model_name in model_names:
            if model_name in self.performance_trackers:
                perf = self.performance_trackers[model_name].get_current_performance()
                current_performances[model_name] = perf
            else:
                current_performances[model_name] = np.inf
        
        # Convert performance to weights (inverse of error)
        raw_weights = {}
        for model_name, perf in current_performances.items():
            if np.isinf(perf):
                raw_weights[model_name] = 0.0
            else:
                # Weight inversely proportional to error
                raw_weights[model_name] = 1.0 / (perf + 1.0)
        
        # Normalize weights
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            weights = {m: w/total_weight for m, w in raw_weights.items()}
        else:
            # Fallback to equal weights
            weights = {m: 1.0/len(model_names) for m in model_names}
        
        return weights
    
    def _smooth_weights(
        self,
        new_weights: Dict[str, float],
        prev_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply exponential smoothing to weights.
        
        Args:
            new_weights: Newly calculated weights
            prev_weights: Previous weights
        
        Returns:
            Smoothed weights
        """
        smoothed = {}
        
        for model_name in new_weights.keys():
            w_new = new_weights[model_name]
            w_prev = prev_weights.get(model_name, w_new)
            
            # Exponential smoothing
            smoothed[model_name] = (
                self.smoothing_factor * w_new + 
                (1 - self.smoothing_factor) * w_prev
            )
        
        # Renormalize
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {m: w/total for m, w in smoothed.items()}
        
        return smoothed
    
    def get_current_states(self) -> Dict[str, str]:
        """Get current performance states for all models."""
        states = {}
        
        for model_name, tracker in self.performance_trackers.items():
            perf = tracker.get_current_performance()
            state = self._classify_performance(perf)
            states[model_name] = state.name
        
        return states
    
    def get_adaptation_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about weight adaptation."""
        if not self.is_fitted:
            return {}
        
        diagnostics = {
            'n_states': self.n_states,
            'current_states': self.get_current_states(),
            'current_weights': self.previous_weights,
            'transition_matrices': {}
        }
        
        for model_name, trans_mat in self.transition_matrices.items():
            diagnostics['transition_matrices'][model_name] = trans_mat.tolist()
        
        return diagnostics
```

---

## 🧪 Testing & Validation

```python
"""Tests for Markov Chain combiner."""
import pytest
import pandas as pd
import numpy as np

from src.models.combination.markov_chain import (
    MarkovChainCombiner, PerformanceState
)


@pytest.fixture
def sample_markov_data():
    """Create sample data with performance patterns."""
    dates = pd.date_range('2024-01-01', periods=200, freq='30min')
    
    # Model with degrading performance
    predictions = {
        'model1': pd.DataFrame({
            'pred_h0': np.concatenate([
                np.random.randn(100) * 30 + 1000,  # Good initially
                np.random.randn(100) * 80 + 1050   # Worse later
            ]),
            'pred_h1': np.concatenate([
                np.random.randn(100) * 30 + 1000,
                np.random.randn(100) * 80 + 1050
            ])
        }, index=dates),
        'model2': pd.DataFrame({
            'pred_h0': np.random.randn(200) * 40 + 1000,  # Stable
            'pred_h1': np.random.randn(200) * 40 + 1000
        }, index=dates)
    }
    
    targets = pd.DataFrame({
        'target_h0': np.random.randn(200) * 30 + 1000,
        'target_h1': np.random.randn(200) * 30 + 1000
    }, index=dates)
    
    return predictions, targets


def test_markov_chain_training(sample_markov_data):
    """Test Markov Chain HMM training."""
    predictions, targets = sample_markov_data
    
    mc_combiner = MarkovChainCombiner()
    mc_combiner.fit(predictions, targets)
    
    assert mc_combiner.is_fitted
    assert len(mc_combiner.hmm_models) == 2


def test_performance_state_classification():
    """Test performance state classification."""
    mc_combiner = MarkovChainCombiner()
    
    assert mc_combiner._classify_performance(1.5) == PerformanceState.EXCELLENT
    assert mc_combiner._classify_performance(3.0) == PerformanceState.GOOD
    assert mc_combiner._classify_performance(6.0) == PerformanceState.FAIR
    assert mc_combiner._classify_performance(10.0) == PerformanceState.POOR


def test_dynamic_weight_adaptation(sample_markov_data):
    """Test dynamic weight calculation."""
    predictions, targets = sample_markov_data
    
    mc_combiner = MarkovChainCombiner()
    mc_combiner.fit(predictions, targets)
    
    result = mc_combiner.combine(predictions)
    
    # Weights should sum to 1
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6


def test_weight_smoothing(sample_markov_data):
    """Test weight smoothing mechanism."""
    predictions, targets = sample_markov_data
    
    mc_combiner = MarkovChainCombiner(config={'smoothing_factor': 0.5})
    mc_combiner.fit(predictions, targets)
    
    # First combination
    result1 = mc_combiner.combine(predictions)
    weights1 = result1.weights.copy()
    
    # Second combination with different data
    result2 = mc_combiner.combine(predictions)
    weights2 = result2.weights
    
    # Weights should change but not drastically
    for model in weights1.keys():
        assert abs(weights2[model] - weights1[model]) < 0.5
```

---

## 📝 Technical Notes

### HMM Architecture
- **States:** Performance levels (excellent/good/fair/poor)
- **Observations:** Rolling MAPE values
- **Transitions:** Learned from historical performance patterns

### Weight Adaptation Strategy
- Inverse relationship with error
- Exponential smoothing for stability
- Context-aware adjustments

---

## 🔗 Dependencies

**Depends On:**
- PC-036-05A: Base Combiner Interface
- PC-042-05B: Stacking Combiner (for comparison)

**Blocks:**
- PC-044-05B: Context-Aware Combiner
- Epic-06A: Hierarchical reconciliation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] HMM training working correctly
- [ ] Performance state classification accurate
- [ ] Transition probabilities learned
- [ ] Dynamic weight calculation functional
- [ ] Weight smoothing prevents erratic changes
- [ ] Context-aware transitions optional
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-05B: Advanced Ensemble Methods](../epics/Epic-05B.md)  
**Previous:** [PC-042-05B: Stacking Combiner](PC-042-05B-stacking-combiner.md)  
**Next:** [PC-044-05B: Context-Aware Weight Adaptation](PC-044-05B-context-aware-combiner.md)
