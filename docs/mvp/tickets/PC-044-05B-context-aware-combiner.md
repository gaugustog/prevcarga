# PC-044-05B: Context-Aware Weight Adaptation

**Ticket ID:** PC-044-05B  
**Epic:** [Epic-05B: Advanced Ensemble Methods](../epics/Epic-05B.md)  
**User Story:** US-3  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `ContextAwareCombiner` that adjusts combination weights based on external conditions and forecasting context. Uses temporal features (hour, day of week, month), meteorological features (temperature), and load characteristics (level, volatility) to adapt weights. Supports multiple adaptation strategies (linear, tree-based, neural network) with interpretability analysis.

**As a** ML engineer  
**I want** combination weights that adapt based on forecasting context  
**So that** I can leverage different model strengths under different conditions

---

## ✅ Acceptance Criteria

- [ ] `ContextAwareCombiner` adjusts weights based on external conditions
- [ ] Context features include temporal, meteorological, and load characteristics
- [ ] Weight adaptation rules learned from historical performance correlations
- [ ] Multiple adaptation strategies (linear, tree-based, neural network)
- [ ] Context-weight relationships are interpretable and stable
- [ ] Ridge regression for linear adaptation
- [ ] Random Forest for non-linear patterns
- [ ] Shallow neural network option
- [ ] Feature importance analysis
- [ ] Integration with BaseCombiner interface

---

## 🔧 Implementation Tasks

### 1. Setup Context-Aware Module
- [ ] Create `src/models/combination/context_aware.py`
- [ ] Import sklearn models (Ridge, RandomForest)
- [ ] Import neural network library (optional)
- [ ] Import BaseCombiner
- [ ] Add module docstrings

### 2. Implement ContextAwareCombiner Class
- [ ] Inherit from BaseCombiner
- [ ] Implement `name` property returning "context_aware"
- [ ] Implement `version` property returning "1.0.0"
- [ ] Define configuration schema
- [ ] Initialize adaptation models

### 3. Implement Context Feature Extraction
- [ ] Create `ContextFeatureExtractor` class
- [ ] Extract temporal features (hour, day_of_week, month, is_weekend)
- [ ] Extract load characteristics (load_level, volatility)
- [ ] Extract meteorological features (temperature, if available)
- [ ] Normalize features
- [ ] Return feature DataFrame

### 4. Implement Temporal Feature Engineering
- [ ] Create `_extract_temporal_features()` method
- [ ] Extract hour of day (0-23)
- [ ] Extract day of week (0-6)
- [ ] Extract month (1-12)
- [ ] Create cyclic encoding (sin/cos)
- [ ] Flag weekends and holidays
- [ ] Return temporal feature dict

### 5. Implement Load Characteristic Features
- [ ] Create `_extract_load_features()` method
- [ ] Calculate load level (current vs average)
- [ ] Calculate load volatility (rolling std)
- [ ] Calculate load trend (recent direction)
- [ ] Categorize load regime (low/medium/high)
- [ ] Return load feature dict

### 6. Implement Meteorological Features
- [ ] Create `_extract_meteo_features()` method
- [ ] Extract temperature (if available)
- [ ] Calculate temperature deviation from normal
- [ ] Flag extreme weather conditions
- [ ] Handle missing meteorological data
- [ ] Return meteo feature dict

### 7. Create Ridge Adaptation Model
- [ ] Create `RidgeWeightAdapter` class
- [ ] Train Ridge regression: weights = f(context)
- [ ] Implement fit() with regularization
- [ ] Implement predict() for weight adaptation
- [ ] Ensure predicted weights sum to 1
- [ ] Return adapted weights

### 8. Create Random Forest Adaptation Model
- [ ] Create `RandomForestWeightAdapter` class
- [ ] Train RF regressor per model weight
- [ ] Configure RF hyperparameters
- [ ] Implement fit() with cross-validation
- [ ] Implement predict() with constraints
- [ ] Calculate feature importances
- [ ] Return adapted weights

### 9. Create Neural Network Adaptation Model
- [ ] Create `NeuralWeightAdapter` class (optional)
- [ ] Design shallow network (2-3 layers)
- [ ] Add softmax output layer for weights
- [ ] Implement training with early stopping
- [ ] Implement prediction
- [ ] Return adapted weights

### 10. Implement fit() Method
- [ ] Validate input predictions and targets
- [ ] Extract context features from timestamps
- [ ] Calculate base weights for each context
- [ ] Train adaptation model (Ridge/RF/NN)
- [ ] Validate adaptation performance
- [ ] Store trained adaptation model
- [ ] Mark as fitted

### 11. Implement combine() Method
- [ ] Validate input predictions
- [ ] Check if adaptation model is fitted
- [ ] Extract context features for current timestamp
- [ ] Get base weights (e.g., from weighted voting)
- [ ] Apply context-aware adaptation
- [ ] Ensure weight constraints (sum=1, non-negative)
- [ ] Combine predictions using adapted weights
- [ ] Create metadata
- [ ] Return CombinationResult

### 12. Implement Weight Constraint Enforcement
- [ ] Create `_enforce_weight_constraints()` method
- [ ] Clip negative weights to zero
- [ ] Normalize weights to sum to 1
- [ ] Apply minimum weight threshold
- [ ] Apply maximum weight threshold
- [ ] Return constrained weights

### 13. Implement Feature Importance Analysis
- [ ] Create `get_feature_importance()` method
- [ ] Calculate importance for Ridge (coefficients)
- [ ] Calculate importance for RF (feature_importances_)
- [ ] Rank features by importance
- [ ] Visualize top features
- [ ] Return importance dict

### 14. Implement Context-Weight Relationship Analysis
- [ ] Create `analyze_context_relationships()` method
- [ ] Calculate correlation between context and weights
- [ ] Identify strongest context-weight relationships
- [ ] Generate interpretability plots
- [ ] Test relationship stability
- [ ] Return analysis report

### 15. Handle Edge Cases
- [ ] Handle missing context features
- [ ] Handle unseen context values
- [ ] Handle adaptation model failures
- [ ] Fallback to base weights when needed
- [ ] Validate adapted weights

### 16. Write Comprehensive Tests
- [ ] Create `tests/models/combination/test_context_aware.py`
- [ ] Test context feature extraction
- [ ] Test different adaptation strategies
- [ ] Test weight constraint enforcement
- [ ] Test feature importance calculation
- [ ] Test with various context scenarios

### 17. Write Integration Tests
- [ ] Test with real temporal patterns
- [ ] Test with meteorological data
- [ ] Test adaptation performance improvement
- [ ] Validate interpretability
- [ ] Test computational performance

### 18. Create Usage Examples
- [ ] Create `examples/context_aware_demo.py`
- [ ] Show context feature engineering
- [ ] Show different adaptation models
- [ ] Show feature importance analysis
- [ ] Compare with static weighting

---

## 💻 Implementation Details

### ContextAwareCombiner Implementation

```python
"""Context-aware weight adaptation combiner."""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from src.models.combination.base_combiner import BaseCombiner, CombinationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContextAwareConfig:
    """Configuration for context-aware combiner."""
    
    adaptation_method: str = 'ridge'  # 'ridge', 'random_forest', 'neural'
    ridge_alpha: float = 1.0
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    use_temporal_features: bool = True
    use_load_features: bool = True
    use_meteo_features: bool = False
    min_weight: float = 0.01
    max_weight: float = 0.95


class ContextFeatureExtractor:
    """Extract context features for weight adaptation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
    
    def extract(
        self,
        timestamps: pd.DatetimeIndex,
        load_data: Optional[pd.Series] = None,
        meteo_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract context features.
        
        Args:
            timestamps: Datetime index
            load_data: Optional load values
            meteo_data: Optional meteorological data
        
        Returns:
            DataFrame with context features
        """
        features = pd.DataFrame(index=timestamps)
        
        # Temporal features
        if self.config.get('use_temporal_features', True):
            features['hour'] = timestamps.hour
            features['day_of_week'] = timestamps.dayofweek
            features['month'] = timestamps.month
            features['is_weekend'] = (timestamps.dayofweek >= 5).astype(int)
            
            # Cyclic encoding
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Load characteristics
        if self.config.get('use_load_features', True) and load_data is not None:
            features['load_level'] = load_data.values
            features['load_volatility'] = load_data.rolling(window=48).std().fillna(0)
            features['load_trend'] = load_data.diff(periods=1).fillna(0)
        
        # Meteorological features
        if self.config.get('use_meteo_features', False) and meteo_data is not None:
            if 'temperature' in meteo_data.columns:
                features['temperature'] = meteo_data['temperature'].values
        
        # Fill any remaining NaN
        features = features.fillna(0)
        
        self.feature_names = list(features.columns)
        
        return features


class ContextAwareCombiner(BaseCombiner):
    """
    Context-aware weight adaptation combiner.
    
    Adaptation Function:
        w(context) = adaptation_model(context_features)
    
    Context Features:
    - Temporal: hour, day_of_week, month, is_weekend
    - Load: load_level, volatility, trend
    - Meteorological: temperature (optional)
    
    Adaptation Models:
    - Ridge: Linear adaptation with regularization
    - Random Forest: Non-linear pattern learning
    - Neural Network: Deep adaptation (optional)
    
    Weight Constraints:
    - Σw_i = 1
    - w_i ∈ [min_weight, max_weight]
    
    Benefits:
    - Leverages model strengths in different contexts
    - Interpretable context-weight relationships
    - Adapts to time-varying conditions
    - Improves over static weighting
    
    Example:
        >>> ca_combiner = ContextAwareCombiner(
        ...     config={'adaptation_method': 'ridge'}
        ... )
        >>> ca_combiner.fit(train_predictions, train_targets)
        >>> result = ca_combiner.combine(test_predictions)
        >>> print(ca_combiner.get_feature_importance())
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize context-aware combiner.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        config_obj = ContextAwareConfig(**self.config)
        self.adaptation_method = config_obj.adaptation_method
        self.ridge_alpha = config_obj.ridge_alpha
        self.rf_n_estimators = config_obj.rf_n_estimators
        self.rf_max_depth = config_obj.rf_max_depth
        self.min_weight = config_obj.min_weight
        self.max_weight = config_obj.max_weight
        
        # Initialize components
        self.feature_extractor = ContextFeatureExtractor(self.config)
        self.adaptation_models: Dict[str, Any] = {}
        self.base_weights: Optional[Dict[str, float]] = None
    
    @property
    def name(self) -> str:
        return "context_aware"
    
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
        Fit context-aware adaptation model.
        
        Args:
            train_predictions: Base model predictions
            train_targets: True target values
            validation_predictions: Optional validation predictions
            validation_targets: Optional validation targets
        """
        logger.info(f"Fitting context-aware combiner with {self.adaptation_method}")
        
        # Validate inputs
        self._validate_predictions(train_predictions)
        
        model_names = list(train_predictions.keys())
        first_pred = train_predictions[model_names[0]]
        
        # Extract context features
        context_features = self.feature_extractor.extract(
            timestamps=first_pred.index
        )
        
        logger.info(f"Extracted {len(context_features.columns)} context features")
        
        # Calculate optimal weights for each context using validation
        optimal_weights = self._calculate_optimal_weights_by_context(
            train_predictions, train_targets, context_features
        )
        
        # Train adaptation model for each model's weight
        for model_name in model_names:
            logger.info(f"Training adaptation model for {model_name}")
            
            # Target: optimal weight for this model
            y_target = optimal_weights[model_name].values
            X_features = context_features.values
            
            # Create and train adaptation model
            adaptation_model = self._create_adaptation_model()
            adaptation_model.fit(X_features, y_target)
            
            self.adaptation_models[model_name] = adaptation_model
        
        # Store base weights (mean optimal weights)
        self.base_weights = {
            model: optimal_weights[model].mean()
            for model in model_names
        }
        
        # Mark as fitted
        self._fitted = True
        self._fit_timestamp = pd.Timestamp.now()
        
        logger.info("Context-aware combiner trained successfully")
    
    def combine(
        self,
        predictions: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CombinationResult:
        """
        Combine predictions with context-aware weights.
        
        Args:
            predictions: Base model predictions
            metadata: Optional metadata
        
        Returns:
            CombinationResult with context-adapted weights
        """
        if not self.is_fitted:
            raise RuntimeError("Context-aware combiner must be fitted before combining")
        
        import time
        start_time = time.time()
        
        logger.info(f"Combining {len(predictions)} models with context-aware weights")
        
        # Validate predictions
        self._validate_predictions(predictions)
        
        first_pred = next(iter(predictions.values()))
        pred_cols = [col for col in first_pred.columns if col.startswith('pred_h')]
        
        # Extract context features for current data
        context_features = self.feature_extractor.extract(
            timestamps=first_pred.index
        )
        
        # Predict adapted weights for each timestamp
        adapted_weights_by_timestamp = self._predict_adapted_weights(
            context_features
        )
        
        # Combine predictions with adapted weights
        combined_df = pd.DataFrame(index=first_pred.index)
        
        for col in pred_cols:
            combined_values = []
            
            for i, timestamp in enumerate(first_pred.index):
                timestamp_weights = adapted_weights_by_timestamp[i]
                
                weighted_sum = sum(
                    timestamp_weights[model] * predictions[model][col].iloc[i]
                    for model in predictions.keys()
                )
                
                combined_values.append(weighted_sum)
            
            combined_df[col] = combined_values
        
        # Average weights for metadata
        avg_weights = {
            model: np.mean([w[model] for w in adapted_weights_by_timestamp])
            for model in predictions.keys()
        }
        
        # Create metadata
        processing_time = time.time() - start_time
        metadata_obj = self._create_metadata(
            models_used=list(predictions.keys()),
            models_excluded=[],
            processing_time=processing_time
        )
        metadata_obj.configuration['adaptation_method'] = self.adaptation_method
        
        return CombinationResult(
            combined_predictions=combined_df,
            weights=avg_weights,
            metadata=metadata_obj
        )
    
    def _create_adaptation_model(self):
        """Create adaptation model based on configuration."""
        if self.adaptation_method == 'ridge':
            return Ridge(alpha=self.ridge_alpha)
        
        elif self.adaptation_method == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown adaptation method: {self.adaptation_method}")
    
    def _calculate_optimal_weights_by_context(
        self,
        predictions: Dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        context_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate optimal weights for each timestamp.
        
        Args:
            predictions: Model predictions
            targets: True targets
            context_features: Context features
        
        Returns:
            DataFrame with optimal weights per timestamp
        """
        from scipy.optimize import minimize
        
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        pred_cols = [col for col in next(iter(predictions.values())).columns 
                     if col.startswith('pred_h')]
        
        optimal_weights = pd.DataFrame(index=context_features.index)
        
        # For each timestamp, find optimal weights
        for idx in range(len(context_features)):
            # Get predictions and targets for this timestamp
            preds_array = []
            targets_array = []
            
            for col in pred_cols:
                horizon = int(col.split('_h')[1])
                target_col = f'target_h{horizon}'
                
                if target_col in targets.columns:
                    pred_vals = [predictions[m][col].iloc[idx] for m in model_names]
                    preds_array.append(pred_vals)
                    targets_array.append(targets[target_col].iloc[idx])
            
            if len(preds_array) == 0:
                # No targets, use equal weights
                for model in model_names:
                    optimal_weights.loc[context_features.index[idx], model] = 1.0 / n_models
                continue
            
            preds_array = np.array(preds_array)  # (n_horizons, n_models)
            targets_array = np.array(targets_array)  # (n_horizons,)
            
            # Optimize weights for this timestamp
            def objective(w):
                combined = preds_array @ w
                return np.mean(np.abs(targets_array - combined))
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            bounds = [(0.0, 1.0) for _ in range(n_models)]
            
            result = minimize(
                objective,
                x0=np.ones(n_models) / n_models,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )
            
            # Store optimal weights
            for j, model in enumerate(model_names):
                optimal_weights.loc[context_features.index[idx], model] = result.x[j]
        
        return optimal_weights
    
    def _predict_adapted_weights(
        self,
        context_features: pd.DataFrame
    ) -> List[Dict[str, float]]:
        """
        Predict adapted weights for given contexts.
        
        Args:
            context_features: Context features
        
        Returns:
            List of weight dictionaries per timestamp
        """
        X_features = context_features.values
        adapted_weights = []
        
        for i in range(len(X_features)):
            timestamp_weights = {}
            
            for model_name, adaptation_model in self.adaptation_models.items():
                predicted_weight = adaptation_model.predict(X_features[i:i+1])[0]
                timestamp_weights[model_name] = predicted_weight
            
            # Enforce constraints
            timestamp_weights = self._enforce_weight_constraints(timestamp_weights)
            
            adapted_weights.append(timestamp_weights)
        
        return adapted_weights
    
    def _enforce_weight_constraints(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Enforce weight constraints (sum=1, bounds).
        
        Args:
            weights: Raw predicted weights
        
        Returns:
            Constrained weights
        """
        # Clip to bounds
        constrained = {
            model: np.clip(w, self.min_weight, self.max_weight)
            for model, w in weights.items()
        }
        
        # Normalize to sum to 1
        total = sum(constrained.values())
        if total > 0:
            constrained = {model: w/total for model, w in constrained.items()}
        else:
            # Fallback to equal weights
            n = len(constrained)
            constrained = {model: 1.0/n for model in constrained.keys()}
        
        return constrained
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for weight adaptation."""
        if not self.is_fitted:
            return {}
        
        importances = {}
        
        for model_name, adaptation_model in self.adaptation_models.items():
            if hasattr(adaptation_model, 'coef_'):
                # Ridge: use coefficients
                importances[model_name] = {
                    feature: float(coef)
                    for feature, coef in zip(
                        self.feature_extractor.feature_names,
                        adaptation_model.coef_
                    )
                }
            elif hasattr(adaptation_model, 'feature_importances_'):
                # Random Forest: use importances
                importances[model_name] = {
                    feature: float(imp)
                    for feature, imp in zip(
                        self.feature_extractor.feature_names,
                        adaptation_model.feature_importances_
                    )
                }
        
        return importances
```

---

## 🧪 Testing & Validation

```python
"""Tests for context-aware combiner."""
import pytest
import pandas as pd
import numpy as np

from src.models.combination.context_aware import ContextAwareCombiner


@pytest.fixture
def sample_context_data():
    """Create sample data with contextual patterns."""
    dates = pd.date_range('2024-01-01', periods=200, freq='30min')
    
    # Model 1: Better at night
    predictions = {
        'model1': pd.DataFrame({
            'pred_h0': 1000 + (dates.hour < 12) * 50 + np.random.randn(200) * 20,
            'pred_h1': 1000 + (dates.hour < 12) * 50 + np.random.randn(200) * 20
        }, index=dates),
        # Model 2: Better during day
        'model2': pd.DataFrame({
            'pred_h0': 1000 + (dates.hour >= 12) * 50 + np.random.randn(200) * 20,
            'pred_h1': 1000 + (dates.hour >= 12) * 50 + np.random.randn(200) * 20
        }, index=dates)
    }
    
    # Targets follow hour-dependent pattern
    targets = pd.DataFrame({
        'target_h0': 1000 + (dates.hour < 12) * 30 + np.random.randn(200) * 10,
        'target_h1': 1000 + (dates.hour < 12) * 30 + np.random.randn(200) * 10
    }, index=dates)
    
    return predictions, targets


def test_context_aware_training(sample_context_data):
    """Test context-aware adaptation training."""
    predictions, targets = sample_context_data
    
    ca_combiner = ContextAwareCombiner(config={'adaptation_method': 'ridge'})
    ca_combiner.fit(predictions, targets)
    
    assert ca_combiner.is_fitted
    assert len(ca_combiner.adaptation_models) == 2


def test_feature_extraction(sample_context_data):
    """Test context feature extraction."""
    predictions, targets = sample_context_data
    
    ca_combiner = ContextAwareCombiner()
    
    first_pred = next(iter(predictions.values()))
    features = ca_combiner.feature_extractor.extract(first_pred.index)
    
    assert 'hour' in features.columns
    assert 'day_of_week' in features.columns
    assert len(features) == len(first_pred)


def test_different_adaptation_methods(sample_context_data):
    """Test different adaptation strategies."""
    predictions, targets = sample_context_data
    
    for method in ['ridge', 'random_forest']:
        ca_combiner = ContextAwareCombiner(config={'adaptation_method': method})
        ca_combiner.fit(predictions, targets)
        
        assert ca_combiner.is_fitted
        
        result = ca_combiner.combine(predictions)
        assert 'pred_h0' in result.combined_predictions.columns


def test_feature_importance(sample_context_data):
    """Test feature importance calculation."""
    predictions, targets = sample_context_data
    
    ca_combiner = ContextAwareCombiner(config={'adaptation_method': 'ridge'})
    ca_combiner.fit(predictions, targets)
    
    importance = ca_combiner.get_feature_importance()
    
    assert len(importance) == 2  # Two models
    assert 'model1' in importance
```

---

## 📝 Technical Notes

### Context Features
- **Temporal:** Cyclic encoding for hour/day
- **Load:** Level, volatility, trend
- **Meteorological:** Temperature (optional)

### Adaptation Strategies
- **Ridge:** Fast, interpretable, linear
- **RF:** Non-linear, handles interactions
- **NN:** Deep patterns (optional)

---

## 🔗 Dependencies

**Depends On:**
- PC-036-05A: Base Combiner Interface
- PC-043-05B: Markov Chain Combiner

**Blocks:**
- PC-045-05B: Advanced Bias Correction
- Epic-06A: Hierarchical reconciliation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Context feature extraction working
- [ ] Multiple adaptation strategies supported
- [ ] Weight constraints enforced
- [ ] Feature importance analysis functional
- [ ] Context-weight relationships interpretable
- [ ] Performance improvement over static weights
- [ ] Unit tests >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-05B: Advanced Ensemble Methods](../epics/Epic-05B.md)  
**Previous:** [PC-043-05B: Markov Chain Combiner](PC-043-05B-markov-chain-combiner.md)  
**Next:** [PC-045-05B: Advanced Bias Correction](PC-045-05B-advanced-bias-correction.md)
