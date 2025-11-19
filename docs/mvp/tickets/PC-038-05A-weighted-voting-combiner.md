# PC-038-05A: Weighted Voting Combiner

**Ticket ID:** PC-038-05A  
**Epic:** [Epic-05A: Base Combination Strategies](../epics/Epic-05A.md)  
**User Story:** US-3  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `WeightedVotingCombiner` with learned weights optimized from validation data. Uses constrained optimization to find optimal weights that minimize prediction error (MAPE, MAE, or RMSE), ensures weights sum to 1 and are non-negative, validates weight stability, and calculates weights based on historical model performance.

**As a** ML engineer  
**I want** a weighted combination method with optimized weights  
**So that** I can leverage historical model performance for better combinations

---

## ✅ Acceptance Criteria

- [ ] `WeightedVotingCombiner` with learned weights from validation data
- [ ] Weight optimization using multiple objective functions (MAPE, MAE, RMSE)
- [ ] Constraints ensure weights sum to 1 and are non-negative
- [ ] Weight stability analysis and validation
- [ ] Historical performance-based weight calculation
- [ ] Optimization converges within 100 iterations
- [ ] Cross-validation for weight robustness
- [ ] Weight persistence and versioning
- [ ] Performance improves over simple averaging (>3%)
- [ ] Comprehensive tests with multiple objectives

---

## 🔧 Implementation Tasks

### 1. Create Weighted Voting Module
- [ ] Create `src/models/combination/weighted_voting.py`
- [ ] Import scipy.optimize
- [ ] Import BaseCombiner
- [ ] Add module docstrings

### 2. Implement WeightedVotingCombiner Class
- [ ] Inherit from BaseCombiner
- [ ] Implement `name` property returning "weighted_voting"
- [ ] Implement `version` property returning "1.0.0"
- [ ] Define optimization objective (MAPE/MAE/RMSE)
- [ ] Initialize with optimization configuration

### 3. Implement Weight Optimization
- [ ] Create `_optimize_weights()` method
- [ ] Define objective function for minimization
- [ ] Set up constraints: Σw_i = 1, w_i ≥ 0
- [ ] Use scipy.optimize.minimize with SLSQP
- [ ] Initialize weights (equal or performance-based)
- [ ] Return optimized weights

### 4. Implement MAPE-Based Objective
- [ ] Create `_mape_objective()` method
- [ ] Calculate: Σ|y_true - Σ(w_i * y_i)| / y_true
- [ ] Handle zero targets (add epsilon)
- [ ] Return mean absolute percentage error
- [ ] Support per-horizon optimization

### 5. Implement MAE-Based Objective
- [ ] Create `_mae_objective()` method
- [ ] Calculate: Σ|y_true - Σ(w_i * y_i)|
- [ ] Return mean absolute error
- [ ] Support per-horizon optimization

### 6. Implement RMSE-Based Objective
- [ ] Create `_rmse_objective()` method
- [ ] Calculate: sqrt(Σ(y_true - Σ(w_i * y_i))²)
- [ ] Return root mean squared error
- [ ] Support per-horizon optimization

### 7. Implement fit() Method
- [ ] Validate train_predictions and train_targets
- [ ] Select objective function from config
- [ ] Optimize weights on training data
- [ ] Optionally validate on validation set
- [ ] Store optimized weights
- [ ] Calculate performance metrics
- [ ] Mark as fitted

### 8. Implement combine() Method
- [ ] Validate predictions input
- [ ] Check if fitted
- [ ] Apply learned weights to predictions
- [ ] Calculate weighted combination
- [ ] Track model contributions
- [ ] Create metadata
- [ ] Return CombinationResult

### 9. Implement Per-Time-Series Weights
- [ ] Support different weights per time series
- [ ] Optimize weights separately for each series
- [ ] Store in _weights dict: {ts: {model: weight}}
- [ ] Apply appropriate weights during combination
- [ ] Log time-series-specific optimization

### 10. Implement Per-Horizon Weights
- [ ] Support different weights per horizon
- [ ] Optimize weights separately for each horizon
- [ ] Store horizon-specific weights
- [ ] Apply during combination
- [ ] Analyze weight patterns across horizons

### 11. Implement Weight Initialization Strategies
- [ ] Create `_initialize_weights()` method
- [ ] Strategy: equal weights (1/n)
- [ ] Strategy: inverse error weights
- [ ] Strategy: performance-based weights
- [ ] Return initial weight vector

### 12. Implement Cross-Validation
- [ ] Create `_cross_validate_weights()` method
- [ ] Split training data into folds
- [ ] Optimize weights on each fold
- [ ] Calculate weight variance
- [ ] Detect unstable weights
- [ ] Return CV results

### 13. Implement Weight Stability Analysis
- [ ] Create `analyze_weight_stability()` method
- [ ] Compare weights across CV folds
- [ ] Calculate weight standard deviation
- [ ] Test for significant weight changes
- [ ] Generate stability report
- [ ] Flag unstable configurations

### 14. Implement Constraint Validation
- [ ] Create `_validate_constraints()` method
- [ ] Check weights sum to 1 (within tolerance)
- [ ] Check all weights ≥ 0
- [ ] Check no weights > 1
- [ ] Raise error if constraints violated
- [ ] Log validation results

### 15. Implement Performance Comparison
- [ ] Compare weighted vs simple averaging
- [ ] Calculate improvement percentage
- [ ] Statistical significance testing
- [ ] Log comparison metrics
- [ ] Store in metadata

### 16. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_weighted_voting.py`
- [ ] Test weight optimization convergence
- [ ] Test constraint satisfaction
- [ ] Test different objective functions
- [ ] Test per-time-series optimization
- [ ] Test weight stability
- [ ] Test save/load with weights

### 17. Write Integration Tests
- [ ] Test with real model outputs
- [ ] Test optimization on multiple time series
- [ ] Validate improvement over simple averaging
- [ ] Test cross-validation
- [ ] Performance benchmarks

### 18. Create Usage Examples
- [ ] Create `examples/weighted_voting_demo.py`
- [ ] Show weight optimization
- [ ] Show different objective functions
- [ ] Show weight stability analysis
- [ ] Compare performance improvements

---

## 💻 Implementation Details

### WeightedVotingCombiner Implementation

```python
"""Weighted voting combination with optimized weights."""
from typing import Dict, Any, Optional, List, Callable
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

from src.models.combination.base_combiner import BaseCombiner, CombinationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WeightedVotingConfig:
    """Configuration for weighted voting combiner."""
    
    objective: str = 'mape'  # 'mape', 'mae', 'rmse'
    optimization_method: str = 'SLSQP'
    max_iterations: int = 100
    tolerance: float = 1e-6
    per_time_series: bool = False
    per_horizon: bool = False
    initial_weights: str = 'equal'  # 'equal', 'inverse_error', 'performance'
    cross_validate: bool = True
    cv_folds: int = 5


class WeightedVotingCombiner(BaseCombiner):
    """
    Weighted voting combination with learned weights.
    
    Optimizes weights to minimize prediction error on training data
    subject to constraints:
    - Σw_i = 1 (weights sum to one)
    - w_i ≥ 0 (non-negative weights)
    
    Optimization Problem:
        min_w L(y_true, Σ(w_i * y_i))
        subject to: Σw_i = 1, w_i ≥ 0
    
    Where L is the loss function (MAPE, MAE, or RMSE).
    
    Benefits:
    - Learns optimal weights from data
    - Leverages model strengths
    - Typically 3-10% better than simple averaging
    
    Example:
        >>> combiner = WeightedVotingCombiner(
        ...     config={'objective': 'mape'}
        ... )
        >>> combiner.fit(train_predictions, train_targets)
        >>> print(f"Optimized weights: {combiner.get_weights()}")
        >>> result = combiner.combine(test_predictions)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize weighted voting combiner.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Parse configuration
        config_obj = WeightedVotingConfig(**self.config)
        self.objective = config_obj.objective
        self.optimization_method = config_obj.optimization_method
        self.max_iterations = config_obj.max_iterations
        self.tolerance = config_obj.tolerance
        self.per_time_series = config_obj.per_time_series
        self.per_horizon = config_obj.per_horizon
        self.initial_weights_strategy = config_obj.initial_weights
        self.cross_validate = config_obj.cross_validate
        self.cv_folds = config_obj.cv_folds
        
        # Select objective function
        self._objective_func = self._get_objective_function(self.objective)
    
    @property
    def name(self) -> str:
        return "weighted_voting"
    
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
        Fit weighted voting combiner by optimizing weights.
        
        Args:
            train_predictions: Training predictions from models
            train_targets: Training targets
            validation_predictions: Optional validation predictions
            validation_targets: Optional validation targets
        """
        logger.info(f"Fitting weighted voting combiner with {self.objective} objective")
        
        # Validate inputs
        self._validate_predictions(train_predictions)
        
        model_names = list(train_predictions.keys())
        n_models = len(model_names)
        
        logger.info(f"Optimizing weights for {n_models} models")
        
        # Get prediction columns
        first_pred = train_predictions[model_names[0]]
        pred_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        # Optimize weights
        if self.per_time_series:
            # Optimize separately for each time series
            # (Would need time series identifier in data)
            optimized_weights = self._optimize_global_weights(
                train_predictions, train_targets, model_names, pred_cols
            )
            self._weights['global'] = optimized_weights
        else:
            # Global weights for all time series
            optimized_weights = self._optimize_global_weights(
                train_predictions, train_targets, model_names, pred_cols
            )
            self._weights['global'] = optimized_weights
        
        # Cross-validation for weight stability
        if self.cross_validate:
            cv_results = self._cross_validate_weights(
                train_predictions, train_targets, model_names, pred_cols
            )
            logger.info(f"Weight stability (CV std): {cv_results['weight_std']}")
        
        # Mark as fitted
        self._fitted = True
        self._fit_timestamp = pd.Timestamp.now()
        
        # Log optimized weights
        logger.info("Optimized weights:")
        for model, weight in optimized_weights.items():
            logger.info(f"  {model}: {weight:.4f}")
    
    def combine(
        self,
        predictions: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CombinationResult:
        """
        Combine predictions using learned weights.
        
        Args:
            predictions: Dictionary of model predictions
            metadata: Optional metadata
        
        Returns:
            CombinationResult with weighted combination
        """
        if not self.is_fitted:
            raise RuntimeError("Combiner must be fitted before combining")
        
        import time
        start_time = time.time()
        
        logger.info(f"Combining {len(predictions)} models with learned weights")
        
        # Validate predictions
        self._validate_predictions(predictions)
        
        # Get weights
        weights = self._weights.get('global', {})
        
        # Filter to available models
        available_models = set(predictions.keys())
        weights = {k: v for k, v in weights.items() if k in available_models}
        
        # Renormalize weights if some models missing
        weight_sum = sum(weights.values())
        if not np.isclose(weight_sum, 1.0):
            weights = {k: v / weight_sum for k, v in weights.items()}
            logger.warning(f"Renormalized weights due to missing models")
        
        # Perform weighted combination
        combined_df = self._weighted_combine(predictions, weights)
        
        # Track contributions
        contributions = {}
        for model_name, pred_df in predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                pred_cols = [col for col in pred_df.columns if col.startswith('pred_h')]
                contributions[model_name] = pred_df[pred_cols] * weight
        
        # Create metadata
        processing_time = time.time() - start_time
        metadata_obj = self._create_metadata(
            models_used=list(predictions.keys()),
            models_excluded=[],
            processing_time=processing_time
        )
        
        return CombinationResult(
            combined_predictions=combined_df,
            weights=weights,
            metadata=metadata_obj,
            model_contributions=contributions
        )
    
    def _optimize_global_weights(
        self,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        model_names: List[str],
        pred_cols: List[str]
    ) -> Dict[str, float]:
        """
        Optimize weights globally across all data.
        
        Args:
            predictions: Model predictions
            targets: Target values
            model_names: List of model names
            pred_cols: Prediction column names
        
        Returns:
            Dictionary of optimized weights
        """
        n_models = len(model_names)
        
        # Initialize weights
        initial_weights = self._initialize_weights(
            predictions, targets, model_names, pred_cols
        )
        
        # Prepare data for optimization
        pred_arrays = []
        for model in model_names:
            model_preds = []
            for col in pred_cols:
                model_preds.append(predictions[model][col].values)
            pred_arrays.append(np.concatenate(model_preds))
        
        pred_matrix = np.array(pred_arrays).T  # Shape: (n_samples, n_models)
        
        target_arrays = []
        for col in pred_cols:
            horizon = int(col.split('_h')[1])
            target_col = f'target_h{horizon}'
            if target_col in targets.columns:
                target_arrays.append(targets[target_col].values)
        
        target_vector = np.concatenate(target_arrays)
        
        # Define objective function
        def objective(weights):
            combined = pred_matrix @ weights
            return self._objective_func(target_vector, combined)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
        ]
        
        # Define bounds (non-negative)
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method=self.optimization_method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        logger.info(f"Optimization converged in {result.nit} iterations")
        logger.info(f"Final objective value: {result.fun:.6f}")
        
        # Create weights dictionary
        optimized_weights = {
            model: float(weight)
            for model, weight in zip(model_names, result.x)
        }
        
        # Validate constraints
        self._validate_constraints(optimized_weights)
        
        return optimized_weights
    
    def _initialize_weights(
        self,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        model_names: List[str],
        pred_cols: List[str]
    ) -> np.ndarray:
        """
        Initialize weights for optimization.
        
        Args:
            predictions: Model predictions
            targets: Target values
            model_names: List of model names
            pred_cols: Prediction columns
        
        Returns:
            Initial weight vector
        """
        n_models = len(model_names)
        
        if self.initial_weights_strategy == 'equal':
            return np.ones(n_models) / n_models
        
        elif self.initial_weights_strategy == 'inverse_error':
            # Weight inversely proportional to error
            errors = []
            for model in model_names:
                model_error = 0
                for col in pred_cols:
                    horizon = int(col.split('_h')[1])
                    target_col = f'target_h{horizon}'
                    if target_col in targets.columns:
                        pred = predictions[model][col].values
                        target = targets[target_col].values
                        model_error += np.mean(np.abs(pred - target))
                errors.append(model_error / len(pred_cols))
            
            # Inverse weights
            inv_errors = 1.0 / (np.array(errors) + 1e-8)
            weights = inv_errors / inv_errors.sum()
            
            return weights
        
        elif self.initial_weights_strategy == 'performance':
            # Based on individual model performance
            # (Similar to inverse_error but could use different metrics)
            return self._initialize_weights(
                predictions, targets, model_names, pred_cols
            )
        
        else:
            return np.ones(n_models) / n_models
    
    def _get_objective_function(self, objective: str) -> Callable:
        """Get objective function based on configuration."""
        if objective == 'mape':
            return self._mape_objective
        elif objective == 'mae':
            return self._mae_objective
        elif objective == 'rmse':
            return self._rmse_objective
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def _mape_objective(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE objective."""
        epsilon = 1e-8
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    def _mae_objective(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAE objective."""
        return np.mean(np.abs(y_true - y_pred))
    
    def _rmse_objective(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE objective."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _weighted_combine(
        self,
        predictions: Dict[str, pd.DataFrame],
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Combine predictions using weights.
        
        Args:
            predictions: Model predictions
            weights: Model weights
        
        Returns:
            Combined predictions DataFrame
        """
        first_pred = next(iter(predictions.values()))
        pred_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        combined_df = pd.DataFrame(index=first_pred.index)
        
        for col in pred_cols:
            weighted_sum = np.zeros(len(first_pred))
            
            for model_name, weight in weights.items():
                if model_name in predictions:
                    weighted_sum += weight * predictions[model_name][col].values
            
            combined_df[col] = weighted_sum
        
        return combined_df
    
    def _cross_validate_weights(
        self,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        model_names: List[str],
        pred_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Cross-validate weight stability.
        
        Args:
            predictions: Model predictions
            targets: Target values
            model_names: Model names
            pred_cols: Prediction columns
        
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import KFold
        
        # Get combined data
        first_pred = predictions[model_names[0]]
        n_samples = len(first_pred)
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        fold_weights = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
            # Split predictions
            fold_train_preds = {}
            for model in model_names:
                fold_train_preds[model] = predictions[model].iloc[train_idx]
            
            fold_train_targets = targets.iloc[train_idx]
            
            # Optimize weights on this fold
            fold_weights_dict = self._optimize_global_weights(
                fold_train_preds, fold_train_targets, model_names, pred_cols
            )
            
            fold_weights.append([fold_weights_dict[m] for m in model_names])
        
        # Calculate statistics
        fold_weights_array = np.array(fold_weights)
        weight_means = fold_weights_array.mean(axis=0)
        weight_stds = fold_weights_array.std(axis=0)
        
        return {
            'fold_weights': fold_weights,
            'weight_means': {m: weight_means[i] for i, m in enumerate(model_names)},
            'weight_stds': {m: weight_stds[i] for i, m in enumerate(model_names)},
            'weight_std': weight_stds.mean()
        }
    
    def _validate_constraints(self, weights: Dict[str, float]) -> None:
        """Validate weight constraints."""
        weight_sum = sum(weights.values())
        
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1, got {weight_sum}")
        
        for model, weight in weights.items():
            if weight < -1e-6:
                raise ValueError(f"Weight for {model} is negative: {weight}")
            if weight > 1.0 + 1e-6:
                raise ValueError(f"Weight for {model} exceeds 1: {weight}")
```

---

## 🧪 Testing & Validation

```python
"""Tests for weighted voting combiner."""
import pytest
import pandas as pd
import numpy as np

from src.models.combination.weighted_voting import WeightedVotingCombiner


@pytest.fixture
def sample_train_data():
    """Create sample training data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='30min')
    
    # Model 1: Good at h0, poor at h1
    predictions = {
        'model1': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 50 + 1000,
            'pred_h1': np.random.randn(100) * 100 + 1000
        }, index=dates),
        # Model 2: Poor at h0, good at h1
        'model2': pd.DataFrame({
            'pred_h0': np.random.randn(100) * 100 + 1000,
            'pred_h1': np.random.randn(100) * 50 + 1000
        }, index=dates)
    }
    
    targets = pd.DataFrame({
        'target_h0': np.random.randn(100) * 30 + 1000,
        'target_h1': np.random.randn(100) * 30 + 1000
    }, index=dates)
    
    return predictions, targets


def test_weight_optimization(sample_train_data):
    """Test weight optimization convergence."""
    predictions, targets = sample_train_data
    
    combiner = WeightedVotingCombiner(config={'objective': 'mae'})
    combiner.fit(predictions, targets)
    
    assert combiner.is_fitted
    weights = combiner.get_weights()
    
    # Check constraints
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(w >= 0 for w in weights.values())


def test_different_objectives(sample_train_data):
    """Test different optimization objectives."""
    predictions, targets = sample_train_data
    
    for objective in ['mape', 'mae', 'rmse']:
        combiner = WeightedVotingCombiner(config={'objective': objective})
        combiner.fit(predictions, targets)
        
        assert combiner.is_fitted
        weights = combiner.get_weights()
        assert np.isclose(sum(weights.values()), 1.0)


def test_weighted_combination(sample_train_data):
    """Test weighted combination."""
    predictions, targets = sample_train_data
    
    combiner = WeightedVotingCombiner()
    combiner.fit(predictions, targets)
    
    result = combiner.combine(predictions)
    
    assert 'pred_h0' in result.combined_predictions.columns
    assert len(result.weights) == 2
```

---

## 📝 Technical Notes

### Optimization Methods
- SLSQP: Sequential Least Squares Programming
- Handles equality and inequality constraints
- Typically converges in 20-50 iterations

### Weight Initialization
- Equal weights: baseline
- Inverse error: favors better models
- Performance-based: uses validation metrics

---

## 🔗 Dependencies

**Depends On:**
- PC-036-05A: Base Combiner Interface
- PC-037-05A: Simple Averaging Combiner (for comparison)

**Blocks:**
- PC-040-05A: Basic Bias Correction
- Epic-05B advanced ensemble methods

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Weight optimization converging reliably
- [ ] Multiple objective functions working
- [ ] Constraint validation functional
- [ ] Cross-validation implemented
- [ ] Weight stability analysis working
- [ ] Performance >3% better than simple averaging
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Previous:** [PC-037-05A: Simple Averaging Combiner](PC-037-05A-simple-averaging-combiner.md)  
**Next:** [PC-039-05A: Best Model Selector](PC-039-05A-best-model-selector.md)
