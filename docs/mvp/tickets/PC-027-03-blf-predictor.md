# PC-027-03: BLF Intraday Predictor

**Ticket ID:** PC-027-03  
**Epic:** [Epic-03: Model Layer - End-to-End Models](../epics/Epic-03.md)  
**User Story:** US-4  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement Baseline Load Forecast (BLF) predictor for intraday forecast corrections using real-time observations. Updates predictions throughout the day by calculating correction factors based on recent forecast errors and applying them to base model forecasts.

**As a** real-time forecasting system  
**I want** BLF predictor to update forecasts using latest observations  
**So that** I can improve accuracy as actual load data becomes available

---

## ✅ Acceptance Criteria

- [ ] Updates predictions using latest available observations
- [ ] Calculates correction factors based on recent forecast errors
- [ ] Supports sliding window updates throughout the day
- [ ] Handles missing real-time data gracefully
- [ ] Provides confidence bounds for corrected forecasts
- [ ] Integrates seamlessly with existing model predictions
- [ ] Update latency < 30 seconds
- [ ] Improves accuracy by > 10% over base forecasts
- [ ] Comprehensive tests with real-time simulation

---

## 🔧 Implementation Tasks

### 1. Create BLF Module Structure
- [ ] Create `src/models/end_to_end/blf_predictor.py`
- [ ] Create `src/models/config/blf_config.py`
- [ ] Add module docstrings

### 2. Implement BLF Configuration Schema
- [ ] Create `BLFConfig` Pydantic model
- [ ] Add `correction_window_hours` field (default: 6)
- [ ] Add `min_observations` field (default: 3)
- [ ] Add `error_decay_factor` field (default: 0.95)
- [ ] Add `max_correction_pct` field (default: 20.0)
- [ ] Add `use_error_model` boolean (default: True)
- [ ] Add `error_model_type` enum (default: "ridge")
- [ ] Add `confidence_level` field (default: 0.95)

### 3. Implement BLFPredictor Class
- [ ] Accept base model in constructor
- [ ] Initialize correction history storage
- [ ] Store configuration
- [ ] Create error model placeholder
- [ ] Add `_is_fitted` flag

### 4. Implement Observation Update Method
- [ ] Create `update_with_observations()` method
- [ ] Accept observations DataFrame with timestamp
- [ ] Validate observation timestamps
- [ ] Calculate forecast errors for observed periods
- [ ] Update correction history
- [ ] Retrain error model if needed
- [ ] Log update statistics

### 5. Implement Error Calculation
- [ ] Create `_calculate_recent_errors()` method
- [ ] Extract base predictions for observed timestamps
- [ ] Compute errors (observation - prediction)
- [ ] Apply decay weights to older errors
- [ ] Filter by correction window
- [ ] Return weighted errors DataFrame

### 6. Implement Correction Factor Calculation
- [ ] Create `_calculate_correction_factors()` method
- [ ] Aggregate recent errors by hour-of-day
- [ ] Compute mean correction per hour
- [ ] Cap corrections at max_correction_pct
- [ ] Return correction factors dictionary

### 7. Implement Error Model Training
- [ ] Create `_train_error_model()` method
- [ ] Extract features (hour, day_of_week, recent_errors)
- [ ] Train Ridge/Lasso regression on errors
- [ ] Store trained error model
- [ ] Log model performance

### 8. Implement Prediction Correction
- [ ] Create `predict_corrected()` method
- [ ] Get base model predictions
- [ ] Apply correction factors by hour
- [ ] If error model available, use it for adjustments
- [ ] Clip negative values if needed
- [ ] Return corrected predictions

### 9. Implement Confidence Bound Adjustment
- [ ] Create `_adjust_confidence_bounds()` method
- [ ] Get base model confidence intervals
- [ ] Calculate empirical error std from history
- [ ] Adjust bounds based on recent accuracy
- [ ] Return updated lower/upper bounds

### 10. Handle Missing Observations
- [ ] Create `_handle_missing_observations()` method
- [ ] Detect missing timestamps in observation stream
- [ ] Interpolate or carry forward corrections
- [ ] Flag periods with stale corrections
- [ ] Log missing data warnings

### 11. Implement Sliding Window Management
- [ ] Create `_update_correction_window()` method
- [ ] Add new errors to history
- [ ] Remove errors outside window
- [ ] Maintain chronological order
- [ ] Limit memory usage

### 12. Implement State Persistence
- [ ] Create `save_state()` method
- [ ] Serialize correction history
- [ ] Save error model
- [ ] Save configuration
- [ ] Store last update timestamp

### 13. Implement State Loading
- [ ] Create `load_state()` method
- [ ] Deserialize correction history
- [ ] Load error model
- [ ] Restore configuration
- [ ] Validate state integrity

### 14. Add Diagnostic Methods
- [ ] Create `get_correction_summary()` method
- [ ] Report average correction by hour
- [ ] Calculate improvement metrics
- [ ] Show error distribution
- [ ] Return diagnostic DataFrame

### 15. Write Comprehensive Tests
- [ ] Create `tests/models/end_to_end/test_blf_predictor.py`
- [ ] Test initialization
- [ ] Test single observation update
- [ ] Test batch observation updates
- [ ] Test correction factor calculation
- [ ] Test error model training
- [ ] Test corrected predictions
- [ ] Test missing data handling
- [ ] Test state persistence

### 16. Write Integration Tests
- [ ] Test with LGBM base model
- [ ] Test with Random Forest base model
- [ ] Test real-time simulation
- [ ] Test multi-day updates

### 17. Create Usage Examples
- [ ] Create `examples/blf_predictor_demo.py`
- [ ] Show initialization with base model
- [ ] Show intraday updates
- [ ] Show accuracy improvement
- [ ] Visualize corrections over time

---

## 💻 Implementation Details

### BLF Configuration

```python
"""Configuration for BLF predictor."""
from typing import Literal
from pydantic import BaseModel, Field


class BLFConfig(BaseModel):
    """Configuration for Baseline Load Forecast predictor."""
    
    # Correction parameters
    correction_window_hours: int = Field(
        default=6,
        description="Hours of recent errors to consider"
    )
    min_observations: int = Field(
        default=3,
        description="Minimum observations required for correction"
    )
    error_decay_factor: float = Field(
        default=0.95,
        description="Exponential decay for older errors (0-1)"
    )
    max_correction_pct: float = Field(
        default=20.0,
        description="Maximum correction as % of prediction"
    )
    
    # Error model parameters
    use_error_model: bool = Field(
        default=True,
        description="Use ML model for error prediction"
    )
    error_model_type: Literal["ridge", "lasso"] = Field(
        default="ridge",
        description="Regression type for error model"
    )
    error_model_alpha: float = Field(
        default=1.0,
        description="Regularization strength"
    )
    
    # Confidence parameters
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for bounds (0-1)"
    )
    
    # Operational parameters
    max_history_days: int = Field(
        default=7,
        description="Maximum days of history to keep"
    )
```

### BLF Predictor Implementation

```python
"""BLF predictor for intraday forecast corrections."""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge, Lasso
import joblib

from src.models.base.model import BaseModel
from src.models.config.blf_config import BLFConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BLFPredictor:
    """
    Baseline Load Forecast (BLF) predictor for intraday corrections.
    
    Updates forecasts throughout the day using real-time observations.
    Calculates correction factors based on recent forecast errors and
    applies them to base model predictions.
    
    Features:
    - Real-time observation updates
    - Exponentially weighted error correction
    - Optional ML-based error modeling
    - Sliding window management
    - Confidence bound adjustment
    
    Example:
        >>> base_model = LGBMModel.load('lgbm_model.pkl')
        >>> blf = BLFPredictor(base_model)
        >>> 
        >>> # Update with observations
        >>> observations = pd.DataFrame({
        ...     'timestamp': [...],
        ...     'load': [...]
        ... })
        >>> blf.update_with_observations(observations)
        >>> 
        >>> # Get corrected predictions
        >>> corrected = blf.predict_corrected(X_future, horizons=[0, 1])
    """
    
    def __init__(
        self,
        base_model: BaseModel,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BLF predictor.
        
        Args:
            base_model: Fitted forecasting model
            config: BLF configuration
        """
        if not base_model.is_fitted():
            raise ValueError("Base model must be fitted")
        
        self.base_model = base_model
        self.config = BLFConfig(**config) if config else BLFConfig()
        
        # Correction state
        self.correction_history: pd.DataFrame = pd.DataFrame()
        self.error_model: Optional[Ridge] = None
        self.last_update: Optional[datetime] = None
        
        logger.info(f"BLF predictor initialized with {base_model.name} base model")
    
    def update_with_observations(
        self,
        observations: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Update predictor with new observations.
        
        Args:
            observations: DataFrame with 'timestamp' and 'load' columns
        
        Returns:
            Update statistics dictionary
        """
        if observations.empty:
            logger.warning("Empty observations provided")
            return {}
        
        # Validate observations
        required_cols = ['timestamp', 'load']
        if not all(col in observations.columns for col in required_cols):
            raise ValueError(f"Observations must contain {required_cols}")
        
        observations = observations.copy()
        observations['timestamp'] = pd.to_datetime(observations['timestamp'])
        observations = observations.set_index('timestamp').sort_index()
        
        # Calculate errors for observed periods
        errors = self._calculate_recent_errors(observations)
        
        # Update history
        if self.correction_history.empty:
            self.correction_history = errors
        else:
            self.correction_history = pd.concat([
                self.correction_history,
                errors
            ]).sort_index()
        
        # Trim to max history
        cutoff = datetime.now() - timedelta(days=self.config.max_history_days)
        self.correction_history = self.correction_history[
            self.correction_history.index >= cutoff
        ]
        
        # Update error model if enabled
        if self.config.use_error_model and len(self.correction_history) >= 50:
            self._train_error_model()
        
        self.last_update = datetime.now()
        
        stats = {
            'n_observations': len(observations),
            'mean_error': errors['error'].mean(),
            'rmse': np.sqrt((errors['error'] ** 2).mean()),
            'history_size': len(self.correction_history),
            'last_update': self.last_update
        }
        
        logger.info(
            f"Updated with {stats['n_observations']} observations, "
            f"RMSE: {stats['rmse']:.2f}"
        )
        
        return stats
    
    def predict_corrected(
        self,
        X: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Generate corrected predictions.
        
        Args:
            X: Feature DataFrame
            horizons: List of horizons to predict
        
        Returns:
            DataFrame with corrected predictions
        """
        # Get base predictions
        base_predictions = self.base_model.predict(X, horizons)
        
        if self.correction_history.empty:
            logger.warning("No correction history, returning base predictions")
            return base_predictions
        
        # Calculate correction factors
        correction_factors = self._calculate_correction_factors()
        
        # Apply corrections
        corrected = base_predictions.copy()
        
        for horizon in horizons:
            pred_col = f'pred_h{horizon}'
            if pred_col not in corrected.columns:
                continue
            
            # Extract hour from index or features
            if isinstance(X.index, pd.DatetimeIndex):
                hours = X.index.hour
            elif 'hour' in X.columns:
                hours = X['hour']
            else:
                logger.warning("Cannot extract hour for corrections")
                continue
            
            # Apply hour-specific corrections
            corrections = hours.map(correction_factors)
            corrections = corrections.fillna(0)
            
            # Apply with maximum limit
            max_correction = corrected[pred_col] * self.config.max_correction_pct / 100
            corrections = corrections.clip(-max_correction, max_correction)
            
            corrected[pred_col] = corrected[pred_col] + corrections
            
            # Ensure non-negative
            corrected[pred_col] = corrected[pred_col].clip(lower=0)
        
        # Adjust confidence bounds if available
        corrected = self._adjust_confidence_bounds(corrected, horizons)
        
        return corrected
    
    def _calculate_recent_errors(
        self,
        observations: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate forecast errors for observed periods.
        
        Args:
            observations: Observed load data
        
        Returns:
            DataFrame with errors
        """
        # This is a simplified version
        # In practice, would need to fetch base predictions for observed timestamps
        
        errors = pd.DataFrame(index=observations.index)
        errors['observed'] = observations['load']
        errors['hour'] = observations.index.hour
        errors['day_of_week'] = observations.index.dayofweek
        
        # Placeholder for actual prediction retrieval
        # In real implementation, would query base model predictions
        errors['predicted'] = np.nan  # Would be filled from stored predictions
        errors['error'] = errors['observed'] - errors['predicted']
        
        # Apply decay weights
        now = datetime.now()
        hours_ago = (now - errors.index).total_seconds() / 3600
        errors['weight'] = self.config.error_decay_factor ** hours_ago
        
        return errors.dropna(subset=['error'])
    
    def _calculate_correction_factors(self) -> Dict[int, float]:
        """
        Calculate correction factors by hour.
        
        Returns:
            Dictionary mapping hour to correction factor
        """
        if len(self.correction_history) < self.config.min_observations:
            return {}
        
        # Filter to correction window
        cutoff = datetime.now() - timedelta(hours=self.config.correction_window_hours)
        recent = self.correction_history[self.correction_history.index >= cutoff]
        
        if recent.empty:
            return {}
        
        # Calculate weighted mean error by hour
        corrections = {}
        for hour in range(24):
            hour_errors = recent[recent['hour'] == hour]
            if len(hour_errors) >= self.config.min_observations:
                weighted_error = np.average(
                    hour_errors['error'],
                    weights=hour_errors['weight']
                )
                corrections[hour] = weighted_error
        
        return corrections
    
    def _train_error_model(self) -> None:
        """Train ML model to predict errors."""
        if len(self.correction_history) < 50:
            return
        
        # Prepare features
        X = self.correction_history[['hour', 'day_of_week']].copy()
        
        # Add lagged errors as features
        for lag in [1, 2, 3]:
            X[f'error_lag_{lag}'] = self.correction_history['error'].shift(lag)
        
        y = self.correction_history['error']
        
        # Drop NaN
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        if len(X_clean) < 20:
            return
        
        # Train model
        if self.config.error_model_type == "ridge":
            model = Ridge(alpha=self.config.error_model_alpha)
        else:
            model = Lasso(alpha=self.config.error_model_alpha)
        
        model.fit(X_clean, y_clean)
        self.error_model = model
        
        # Calculate R²
        score = model.score(X_clean, y_clean)
        logger.info(f"Error model trained, R² = {score:.4f}")
    
    def _adjust_confidence_bounds(
        self,
        predictions: pd.DataFrame,
        horizons: List[int]
    ) -> pd.DataFrame:
        """
        Adjust confidence bounds based on recent accuracy.
        
        Args:
            predictions: Predictions with confidence bounds
            horizons: Horizons to adjust
        
        Returns:
            Predictions with adjusted bounds
        """
        if self.correction_history.empty:
            return predictions
        
        # Calculate empirical error std
        recent_std = self.correction_history['error'].std()
        
        # Adjust bounds
        z_score = 1.96  # For 95% confidence
        
        for horizon in horizons:
            pred_col = f'pred_h{horizon}'
            lower_col = f'pred_h{horizon}_lower'
            upper_col = f'pred_h{horizon}_upper'
            
            if pred_col in predictions.columns:
                if lower_col not in predictions.columns:
                    # Create bounds if not exist
                    predictions[lower_col] = predictions[pred_col] - z_score * recent_std
                    predictions[upper_col] = predictions[pred_col] + z_score * recent_std
                else:
                    # Adjust existing bounds
                    current_width = (
                        predictions[upper_col] - predictions[lower_col]
                    ) / 2
                    
                    # Blend with empirical std
                    adjusted_std = 0.7 * current_width + 0.3 * recent_std
                    
                    predictions[lower_col] = predictions[pred_col] - z_score * adjusted_std
                    predictions[upper_col] = predictions[pred_col] + z_score * adjusted_std
        
        return predictions
    
    def get_correction_summary(self) -> pd.DataFrame:
        """
        Get summary of corrections.
        
        Returns:
            DataFrame with correction statistics by hour
        """
        if self.correction_history.empty:
            return pd.DataFrame()
        
        summary = self.correction_history.groupby('hour').agg({
            'error': ['mean', 'std', 'count'],
            'weight': 'sum'
        }).round(2)
        
        summary.columns = ['_'.join(col) for col in summary.columns]
        
        return summary
    
    def save_state(self, path: str) -> None:
        """Save BLF predictor state."""
        state = {
            'config': self.config.dict(),
            'correction_history': self.correction_history,
            'error_model': self.error_model,
            'last_update': self.last_update
        }
        
        joblib.dump(state, path)
        logger.info(f"BLF state saved to {path}")
    
    @classmethod
    def load_state(
        cls,
        path: str,
        base_model: BaseModel
    ) -> 'BLFPredictor':
        """Load BLF predictor state."""
        state = joblib.load(path)
        
        predictor = cls(base_model, config=state['config'])
        predictor.correction_history = state['correction_history']
        predictor.error_model = state['error_model']
        predictor.last_update = state['last_update']
        
        logger.info(f"BLF state loaded from {path}")
        
        return predictor
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for BLF predictor."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.end_to_end.blf_predictor import BLFPredictor
from src.models.end_to_end.lgbm_model import LGBMModel


@pytest.fixture
def mock_base_model(mocker):
    """Create mock base model."""
    model = mocker.Mock(spec=LGBMModel)
    model.is_fitted.return_value = True
    model.name = "lgbm"
    
    def mock_predict(X, horizons):
        pred = pd.DataFrame(index=X.index)
        for h in horizons:
            pred[f'pred_h{h}'] = 1000 + np.random.randn(len(X)) * 50
        return pred
    
    model.predict.side_effect = mock_predict
    
    return model


@pytest.fixture
def sample_observations():
    """Create sample observations."""
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=6),
        end=datetime.now(),
        freq='30min'
    )
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'load': 1000 + np.random.randn(len(timestamps)) * 50
    })


def test_blf_initialization(mock_base_model):
    """Test BLF predictor initialization."""
    blf = BLFPredictor(mock_base_model)
    
    assert blf.base_model == mock_base_model
    assert blf.correction_history.empty
    assert blf.error_model is None


def test_blf_update_observations(mock_base_model, sample_observations):
    """Test observation updates."""
    blf = BLFPredictor(mock_base_model)
    
    stats = blf.update_with_observations(sample_observations)
    
    assert stats['n_observations'] == len(sample_observations)
    assert 'mean_error' in stats
    assert not blf.correction_history.empty


def test_blf_corrected_predictions(mock_base_model, sample_observations):
    """Test corrected predictions."""
    blf = BLFPredictor(mock_base_model)
    blf.update_with_observations(sample_observations)
    
    X_future = pd.DataFrame({
        'hour': np.arange(24),
        'feature1': np.random.randn(24)
    }, index=pd.date_range('2025-01-01', periods=24, freq='h'))
    
    corrected = blf.predict_corrected(X_future, horizons=[0, 1])
    
    assert 'pred_h0' in corrected.columns
    assert len(corrected) == len(X_future)


def test_blf_correction_factors(mock_base_model, sample_observations):
    """Test correction factor calculation."""
    blf = BLFPredictor(mock_base_model)
    blf.update_with_observations(sample_observations)
    
    # Manually set some errors
    blf.correction_history['error'] = -50  # Consistent under-prediction
    
    factors = blf._calculate_correction_factors()
    
    assert isinstance(factors, dict)


def test_blf_state_persistence(mock_base_model, sample_observations, tmp_path):
    """Test state save/load."""
    blf = BLFPredictor(mock_base_model)
    blf.update_with_observations(sample_observations)
    
    state_path = tmp_path / "blf_state.pkl"
    blf.save_state(str(state_path))
    
    loaded_blf = BLFPredictor.load_state(str(state_path), mock_base_model)
    
    assert len(loaded_blf.correction_history) == len(blf.correction_history)
```

---

## 📝 Technical Notes

### Correction Strategy
- Uses exponentially weighted recent errors
- Hour-of-day specific corrections
- Optional ML model for sophisticated error prediction
- Capped at max_correction_pct to prevent over-correction

### Real-time Operation
- Sliding window maintains recent history
- Fast updates (<30s) for operational deployment
- Graceful handling of missing observations
- State persistence for recovery

---

## 🔗 Dependencies

**Depends On:**
- PC-024-03: Base Model Interface
- PC-025-03: LGBM Model (or PC-026-03: Random Forest)

**External Dependencies:**
- `scikit-learn>=1.3.0`
- `pandas>=2.0.0`

**Blocks:**
- Epic-04 hierarchical predictions

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Updates with real-time observations
- [ ] Correction factors calculated correctly
- [ ] Error model training functional
- [ ] Corrected predictions improve accuracy >10%
- [ ] Update latency < 30s
- [ ] Unit tests pass with >85% coverage
- [ ] Integration tests with base models pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-026-03: Random Forest Multi-Horizon Model](PC-026-03-random-forest-model.md)  
**Next:** [PC-028-03: Model Serialization & Version Management](PC-028-03-serialization-version-management.md)
