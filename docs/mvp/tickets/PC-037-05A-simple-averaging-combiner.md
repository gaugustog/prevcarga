# PC-037-05A: Simple Averaging Combiner

**Ticket ID:** PC-037-05A  
**Epic:** [Epic-05A: Base Combination Strategies](../epics/Epic-05A.md)  
**User Story:** US-2  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `SimpleAveragingCombiner` that provides equal-weight averaging of model predictions using arithmetic and geometric mean methods. Handles missing predictions by using available model subset, tracks performance compared to individual models, and supports configurable model subset selection.

**As a** forecasting analyst  
**I want** a simple averaging combination method  
**So that** I can quickly combine models without complex weight optimization

---

## ✅ Acceptance Criteria

- [ ] `SimpleAveragingCombiner` provides equal-weight averaging
- [ ] Supports arithmetic mean: `(1/n) * Σ(y_i)`
- [ ] Supports geometric mean: `(Π(y_i))^(1/n)`
- [ ] Handles missing predictions with available model subset
- [ ] Performance tracking compared to individual models
- [ ] Configurable model subset selection
- [ ] Handles zero/negative values in geometric mean
- [ ] No fitting required (equal weights)
- [ ] Performance metrics (MAPE, MAE, RMSE)
- [ ] Comprehensive tests with multiple models

---

## 🔧 Implementation Tasks

### 1. Create Simple Averaging Module
- [ ] Create `src/models/combination/simple_averaging.py`
- [ ] Import BaseCombiner
- [ ] Import numpy and pandas
- [ ] Add module docstrings

### 2. Implement SimpleAveragingCombiner Class
- [ ] Inherit from BaseCombiner
- [ ] Implement `name` property returning "simple_averaging"
- [ ] Implement `version` property returning "1.0.0"
- [ ] Define configuration schema
- [ ] Initialize with averaging method (arithmetic/geometric)

### 3. Implement Arithmetic Mean Combination
- [ ] Create `_combine_arithmetic()` method
- [ ] Calculate mean across model predictions
- [ ] Handle per-column averaging (pred_h0, pred_h1, etc.)
- [ ] Preserve index and column structure
- [ ] Return combined DataFrame

### 4. Implement Geometric Mean Combination
- [ ] Create `_combine_geometric()` method
- [ ] Calculate product of predictions
- [ ] Take n-th root for n models
- [ ] Handle zero values (add epsilon)
- [ ] Handle negative values (raise warning, use abs)
- [ ] Return combined DataFrame

### 5. Implement combine() Method
- [ ] Validate predictions input
- [ ] Handle missing predictions
- [ ] Select combination method (arithmetic/geometric)
- [ ] Calculate combined predictions
- [ ] Create equal weights dict
- [ ] Track model contributions
- [ ] Create metadata
- [ ] Return CombinationResult

### 6. Implement fit() Method
- [ ] Accept train_predictions and train_targets
- [ ] Mark as fitted (no actual optimization needed)
- [ ] Calculate baseline performance metrics
- [ ] Store individual model performance
- [ ] Compare averaging performance
- [ ] Log fitting summary

### 7. Implement Model Subset Selection
- [ ] Create `_select_model_subset()` method
- [ ] Support include_models config parameter
- [ ] Support exclude_models config parameter
- [ ] Validate selected models exist
- [ ] Log subset selection
- [ ] Return filtered predictions dict

### 8. Implement Missing Value Handling
- [ ] Override `_handle_missing_predictions()`
- [ ] Count available models per timestamp
- [ ] Exclude models with >50% missing
- [ ] Log missing value warnings
- [ ] Adjust weights for available models
- [ ] Return cleaned predictions

### 9. Implement Performance Comparison
- [ ] Create `calculate_performance_metrics()` method
- [ ] Calculate MAPE, MAE, RMSE
- [ ] Compare combination vs individual models
- [ ] Calculate improvement percentages
- [ ] Store metrics in metadata
- [ ] Return performance dict

### 10. Implement Model Contribution Tracking
- [ ] Create `_track_contributions()` method
- [ ] Calculate each model's contribution (1/n)
- [ ] Store contribution DataFrames
- [ ] Visualize contribution breakdown
- [ ] Return contributions dict

### 11. Handle Edge Cases
- [ ] Handle single model (return as-is)
- [ ] Handle all models with same prediction
- [ ] Handle extreme outliers
- [ ] Validate non-negative predictions
- [ ] Handle empty predictions
- [ ] Raise informative errors

### 12. Implement Configuration Options
- [ ] Create SimpleAveragingConfig dataclass
- [ ] Parameter: averaging_method (arithmetic/geometric)
- [ ] Parameter: include_models (Optional[List[str]])
- [ ] Parameter: exclude_models (Optional[List[str]])
- [ ] Parameter: handle_negatives (abs/warn/error)
- [ ] Validate configuration

### 13. Implement Logging
- [ ] Log combination method selected
- [ ] Log number of models combined
- [ ] Log missing prediction warnings
- [ ] Log performance comparison
- [ ] Add debug logging for weights

### 14. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_simple_averaging.py`
- [ ] Test arithmetic mean combination
- [ ] Test geometric mean combination
- [ ] Test missing prediction handling
- [ ] Test model subset selection
- [ ] Test performance calculation
- [ ] Test edge cases (single model, negatives)

### 15. Write Integration Tests
- [ ] Test with real model outputs (LGBM, RF)
- [ ] Test with 26 time series
- [ ] Test with multiple horizons
- [ ] Compare performance vs individual models
- [ ] Test serialization/deserialization

### 16. Create Usage Examples
- [ ] Create `examples/simple_averaging_demo.py`
- [ ] Show arithmetic mean usage
- [ ] Show geometric mean usage
- [ ] Show performance comparison
- [ ] Show model subset selection

---

## 💻 Implementation Details

### SimpleAveragingCombiner Implementation

```python
"""Simple averaging combination strategy."""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.models.combination.base_combiner import BaseCombiner, CombinationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SimpleAveragingConfig:
    """Configuration for simple averaging combiner."""
    
    averaging_method: str = 'arithmetic'  # 'arithmetic' or 'geometric'
    include_models: Optional[List[str]] = None
    exclude_models: Optional[List[str]] = None
    handle_negatives: str = 'warn'  # 'abs', 'warn', 'error'
    epsilon: float = 1e-10  # For handling zeros in geometric mean


class SimpleAveragingCombiner(BaseCombiner):
    """
    Simple averaging combination strategy.
    
    Combines model predictions using equal weights (1/n for n models).
    Supports both arithmetic and geometric mean.
    
    Arithmetic Mean:
        y_combined = (1/n) * Σ(y_i) for i in models
    
    Geometric Mean:
        y_combined = (Π(y_i))^(1/n) for i in models
    
    Benefits:
    - No training required
    - Robust to outliers (geometric mean)
    - Simple and interpretable
    - Fast computation
    
    Limitations:
    - Equal weights may not be optimal
    - Geometric mean sensitive to zeros/negatives
    
    Example:
        >>> combiner = SimpleAveragingCombiner(
        ...     config={'averaging_method': 'arithmetic'}
        ... )
        >>> # No fitting required
        >>> result = combiner.combine({
        ...     'lgbm': lgbm_predictions,
        ...     'rf': rf_predictions
        ... })
        >>> print(f"Combined MAPE: {result.metadata.performance_metrics['mape']:.2f}%")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize simple averaging combiner.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Parse configuration
        config_obj = SimpleAveragingConfig(**self.config)
        self.averaging_method = config_obj.averaging_method
        self.include_models = config_obj.include_models
        self.exclude_models = config_obj.exclude_models
        self.handle_negatives = config_obj.handle_negatives
        self.epsilon = config_obj.epsilon
        
        if self.averaging_method not in ['arithmetic', 'geometric']:
            raise ValueError(
                f"averaging_method must be 'arithmetic' or 'geometric', "
                f"got {self.averaging_method}"
            )
    
    @property
    def name(self) -> str:
        return "simple_averaging"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def combine(
        self,
        predictions: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CombinationResult:
        """
        Combine predictions using simple averaging.
        
        Args:
            predictions: Dictionary mapping model names to predictions
            metadata: Optional metadata (not used)
        
        Returns:
            CombinationResult with averaged predictions
        """
        import time
        start_time = time.time()
        
        logger.info(f"Combining {len(predictions)} models using {self.averaging_method} mean")
        
        # Validate predictions
        self._validate_predictions(predictions)
        
        # Apply model subset selection
        predictions = self._select_model_subset(predictions)
        
        # Handle missing predictions
        predictions, excluded = self._handle_missing_predictions(predictions)
        
        # Perform combination
        if self.averaging_method == 'arithmetic':
            combined_df = self._combine_arithmetic(predictions)
        else:
            combined_df = self._combine_geometric(predictions)
        
        # Create equal weights
        n_models = len(predictions)
        weights = {model: 1.0 / n_models for model in predictions.keys()}
        
        # Track contributions
        contributions = self._track_contributions(predictions, weights)
        
        # Create metadata
        processing_time = time.time() - start_time
        metadata_obj = self._create_metadata(
            models_used=list(predictions.keys()),
            models_excluded=excluded,
            processing_time=processing_time
        )
        
        logger.info(f"Combination complete in {processing_time:.2f}s")
        
        return CombinationResult(
            combined_predictions=combined_df,
            weights=weights,
            metadata=metadata_obj,
            model_contributions=contributions
        )
    
    def fit(
        self,
        train_predictions: Dict[str, pd.DataFrame],
        train_targets: pd.DataFrame,
        validation_predictions: Optional[Dict[str, pd.DataFrame]] = None,
        validation_targets: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Fit combiner (no optimization needed for simple averaging).
        
        Calculates baseline performance metrics for comparison.
        
        Args:
            train_predictions: Training predictions
            train_targets: Training targets
            validation_predictions: Optional validation predictions
            validation_targets: Optional validation targets
        """
        logger.info("Fitting simple averaging combiner (calculating baseline metrics)")
        
        # Validate inputs
        self._validate_predictions(train_predictions)
        
        # Calculate individual model performance
        individual_performance = {}
        for model_name, predictions in train_predictions.items():
            metrics = self._calculate_metrics(predictions, train_targets)
            individual_performance[model_name] = metrics
        
        # Calculate averaged performance
        combined_result = self.combine(train_predictions)
        combined_metrics = self._calculate_metrics(
            combined_result.combined_predictions,
            train_targets
        )
        
        # Store performance
        self._performance_history.append({
            'individual': individual_performance,
            'combined': combined_metrics,
            'timestamp': combined_result.metadata.timestamp
        })
        
        # Mark as fitted
        self._fitted = True
        self._fit_timestamp = combined_result.metadata.timestamp
        
        # Log performance summary
        logger.info("Performance Summary:")
        for model, metrics in individual_performance.items():
            logger.info(f"  {model}: MAPE={metrics['mape']:.2f}%")
        logger.info(f"  Combined: MAPE={combined_metrics['mape']:.2f}%")
    
    def _combine_arithmetic(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine predictions using arithmetic mean.
        
        Args:
            predictions: Dictionary of model predictions
        
        Returns:
            Combined predictions DataFrame
        """
        # Get prediction columns
        first_pred = next(iter(predictions.values()))
        pred_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        # Initialize result
        combined_df = pd.DataFrame(index=first_pred.index)
        
        # Average each column
        for col in pred_cols:
            values = np.array([pred[col].values for pred in predictions.values()])
            combined_df[col] = np.mean(values, axis=0)
        
        logger.debug(f"Arithmetic mean calculated for {len(pred_cols)} horizons")
        
        return combined_df
    
    def _combine_geometric(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine predictions using geometric mean.
        
        Args:
            predictions: Dictionary of model predictions
        
        Returns:
            Combined predictions DataFrame
        """
        # Get prediction columns
        first_pred = next(iter(predictions.values()))
        pred_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        # Initialize result
        combined_df = pd.DataFrame(index=first_pred.index)
        
        n_models = len(predictions)
        
        # Calculate geometric mean for each column
        for col in pred_cols:
            values = np.array([pred[col].values for pred in predictions.values()])
            
            # Handle negatives
            if np.any(values < 0):
                if self.handle_negatives == 'error':
                    raise ValueError(f"Negative values found in {col}, cannot use geometric mean")
                elif self.handle_negatives == 'warn':
                    logger.warning(f"Negative values in {col}, using absolute values")
                    values = np.abs(values)
                elif self.handle_negatives == 'abs':
                    values = np.abs(values)
            
            # Handle zeros (add epsilon)
            values = values + self.epsilon
            
            # Calculate geometric mean: (product)^(1/n)
            product = np.prod(values, axis=0)
            geometric_mean = np.power(product, 1.0 / n_models)
            
            combined_df[col] = geometric_mean
        
        logger.debug(f"Geometric mean calculated for {len(pred_cols)} horizons")
        
        return combined_df
    
    def _select_model_subset(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Select subset of models based on configuration.
        
        Args:
            predictions: Dictionary of all predictions
        
        Returns:
            Filtered predictions dictionary
        """
        # Apply include filter
        if self.include_models:
            predictions = {
                k: v for k, v in predictions.items()
                if k in self.include_models
            }
            logger.info(f"Included models: {list(predictions.keys())}")
        
        # Apply exclude filter
        if self.exclude_models:
            predictions = {
                k: v for k, v in predictions.items()
                if k not in self.exclude_models
            }
            logger.info(f"After exclusions: {list(predictions.keys())}")
        
        if not predictions:
            raise ValueError("No models remaining after subset selection")
        
        return predictions
    
    def _track_contributions(
        self,
        predictions: Dict[str, pd.DataFrame],
        weights: Dict[str, float]
    ) -> Dict[str, pd.DataFrame]:
        """
        Track each model's contribution to final prediction.
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
        
        Returns:
            Dictionary mapping models to weighted contributions
        """
        contributions = {}
        
        for model_name, pred_df in predictions.items():
            weight = weights[model_name]
            
            # Calculate weighted contribution
            pred_cols = [col for col in pred_df.columns if col.startswith('pred_h')]
            contribution_df = pred_df[pred_cols].copy()
            
            if self.averaging_method == 'arithmetic':
                contribution_df = contribution_df * weight
            # For geometric mean, contribution is more complex
            
            contributions[model_name] = contribution_df
        
        return contributions
    
    def _calculate_metrics(
        self,
        predictions: pd.DataFrame,
        targets: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            predictions: Prediction DataFrame
            targets: Target DataFrame
        
        Returns:
            Dictionary of metrics
        """
        from src.evaluation.metrics import calculate_mape, calculate_mae, calculate_rmse
        
        metrics = {}
        
        # Calculate for each horizon
        pred_cols = [col for col in predictions.columns if col.startswith('pred_h')]
        
        for col in pred_cols:
            horizon = int(col.split('_h')[1])
            target_col = f'target_h{horizon}'
            
            if target_col in targets.columns:
                mape = calculate_mape(targets[target_col], predictions[col])
                mae = calculate_mae(targets[target_col], predictions[col])
                rmse = calculate_rmse(targets[target_col], predictions[col])
                
                metrics[f'mape_h{horizon}'] = mape
                metrics[f'mae_h{horizon}'] = mae
                metrics[f'rmse_h{horizon}'] = rmse
        
        # Calculate average metrics
        metrics['mape'] = np.mean([v for k, v in metrics.items() if k.startswith('mape_h')])
        metrics['mae'] = np.mean([v for k, v in metrics.items() if k.startswith('mae_h')])
        metrics['rmse'] = np.mean([v for k, v in metrics.items() if k.startswith('rmse_h')])
        
        return metrics
```

---

## 🧪 Testing & Validation

```python
"""Tests for simple averaging combiner."""
import pytest
import pandas as pd
import numpy as np

from src.models.combination.simple_averaging import SimpleAveragingCombiner


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    dates = pd.date_range('2024-01-01', periods=48, freq='30min')
    
    return {
        'lgbm': pd.DataFrame({
            'pred_h0': np.array([1000, 1100, 1200] * 16),
            'pred_h1': np.array([1050, 1150, 1250] * 16)
        }, index=dates),
        'rf': pd.DataFrame({
            'pred_h0': np.array([900, 1000, 1100] * 16),
            'pred_h1': np.array([950, 1050, 1150] * 16)
        }, index=dates),
        'regdin': pd.DataFrame({
            'pred_h0': np.array([1100, 1200, 1300] * 16),
            'pred_h1': np.array([1150, 1250, 1350] * 16)
        }, index=dates)
    }


def test_arithmetic_mean(sample_predictions):
    """Test arithmetic mean combination."""
    combiner = SimpleAveragingCombiner(config={'averaging_method': 'arithmetic'})
    
    result = combiner.combine(sample_predictions)
    
    # Check equal weights
    assert all(w == pytest.approx(1/3) for w in result.weights.values())
    
    # Check arithmetic mean calculation
    expected_h0 = (1000 + 900 + 1100) / 3
    assert result.combined_predictions['pred_h0'].iloc[0] == pytest.approx(expected_h0)


def test_geometric_mean(sample_predictions):
    """Test geometric mean combination."""
    combiner = SimpleAveragingCombiner(config={'averaging_method': 'geometric'})
    
    result = combiner.combine(sample_predictions)
    
    # Check geometric mean calculation
    expected_h0 = (1000 * 900 * 1100) ** (1/3)
    assert result.combined_predictions['pred_h0'].iloc[0] == pytest.approx(expected_h0, rel=1e-3)


def test_model_subset_selection(sample_predictions):
    """Test model subset selection."""
    combiner = SimpleAveragingCombiner(config={
        'averaging_method': 'arithmetic',
        'include_models': ['lgbm', 'rf']
    })
    
    result = combiner.combine(sample_predictions)
    
    assert len(result.weights) == 2
    assert 'regdin' not in result.weights


def test_missing_predictions():
    """Test handling of missing predictions."""
    dates = pd.date_range('2024-01-01', periods=48, freq='30min')
    
    predictions = {
        'lgbm': pd.DataFrame({
            'pred_h0': [1000] * 48,
            'pred_h1': [1050] * 48
        }, index=dates),
        'rf': pd.DataFrame({
            'pred_h0': [np.nan] * 24 + [1000] * 24,  # 50% missing
            'pred_h1': [1050] * 48
        }, index=dates)
    }
    
    combiner = SimpleAveragingCombiner()
    result = combiner.combine(predictions)
    
    # RF should be excluded due to >50% missing
    assert 'rf' in result.metadata.models_excluded
```

---

## 📝 Technical Notes

### Arithmetic vs Geometric Mean
- **Arithmetic:** Best for additive errors, simple interpretation
- **Geometric:** Robust to outliers, better for multiplicative errors

### Performance Expectations
- Typically 5-10% improvement over worst model
- May not beat best individual model
- Baseline for advanced methods

---

## 🔗 Dependencies

**Depends On:**
- PC-036-05A: Base Combiner Interface

**Blocks:**
- PC-038-05A: Weighted Voting Combiner
- Epic-05B advanced ensemble methods

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Arithmetic mean working correctly
- [ ] Geometric mean working correctly
- [ ] Model subset selection functional
- [ ] Missing value handling tested
- [ ] Performance metrics calculated
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Previous:** [PC-036-05A: Base Combiner Interface](PC-036-05A-base-combiner-interface.md)  
**Next:** [PC-038-05A: Weighted Voting Combiner](PC-038-05A-weighted-voting-combiner.md)
