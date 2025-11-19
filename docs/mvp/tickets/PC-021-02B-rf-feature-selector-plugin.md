# PC-021-02B: RF Feature Selector Plugin

**Ticket ID:** PC-021-02B  
**Epic:** [Epic-02B: Advanced Feature Transformations](../epics/Epic-02B.md)  
**User Story:** US-4  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a Random Forest-based feature selection plugin that performs horizon-specific selection for multi-horizon forecasting. Prevents temporal data leakage by enforcing horizon-aware feature availability and uses advanced importance metrics (permutation importance, SHAP values) for robust selection.

**As an** ML engineer  
**I want** horizon-specific feature selection for Random Forest  
**So that** multi-horizon models avoid data leakage and optimize per-horizon accuracy

---

## ✅ Acceptance Criteria

- [ ] `RFFeatureSelectorPlugin` selects optimal features per horizon (D+0 to D+8)
- [ ] Prevents temporal leakage by respecting horizon-specific data availability
- [ ] Uses permutation importance and SHAP values for selection
- [ ] Supports RFECV (Recursive Feature Elimination with Cross-Validation)
- [ ] Generates feature importance rankings and selection reports
- [ ] Optimizes for model size vs accuracy trade-offs
- [ ] Configurable selection criteria (threshold, max_features, min_importance)
- [ ] Integration tests with actual feature pipelines
- [ ] Performance: <5 minutes per area for all horizons

---

## 🔧 Implementation Tasks

### 1. Create Feature Selection Module Structure
- [ ] Create `src/features/plugins/rf_selector.py`
- [ ] Create `src/features/advanced/feature_selection.py`
- [ ] Create `src/features/evaluation/leakage_detector.py`
- [ ] Add module docstrings

### 2. Implement Leakage Detection Utilities
- [ ] Create `DataLeakageDetector` class
- [ ] Implement `check_temporal_leakage()` method
- [ ] Detect future-looking features based on naming patterns
- [ ] Validate lag features have sufficient lag for horizon
- [ ] Create `get_horizon_safe_features()` method
- [ ] Document leakage detection rules

### 3. Implement Configuration Schema
- [ ] Create `RFFeatureSelectorConfig` Pydantic model
- [ ] Add `target_column` string field
- [ ] Add `horizons` list of integers (e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8])
- [ ] Add `max_features_per_horizon` integer (default: 50)
- [ ] Add `selection_method` enum: "rfecv", "permutation", "shap", "combined"
- [ ] Add `cv_folds` integer (default: 5)
- [ ] Add `importance_threshold` float (default: 0.01)
- [ ] Add `n_estimators` integer for RF (default: 50 for speed)
- [ ] Add `random_state` integer (default: 42)
- [ ] Add `n_jobs` integer (default: -1)
- [ ] Validate horizons are non-negative

### 4. Implement RFFeatureSelectorPlugin Class
- [ ] Inherit from `AdvancedFeaturePlugin`
- [ ] Implement `name` property returning "rf_feature_selector"
- [ ] Implement `version` property
- [ ] Implement `computational_complexity` returning "high"
- [ ] Add class docstring with methodology explanation

### 5. Implement Horizon-Safe Feature Filtering
- [ ] Create `_get_horizon_safe_features()` method
- [ ] Filter out target variable and its direct derivatives
- [ ] Check lag features: lag must be >= horizon * periods_per_day
- [ ] Exclude features with "future" or "forward" in name
- [ ] Include temporal, calendar, and auxiliary features
- [ ] Log filtered feature count per horizon
- [ ] Return list of safe features

### 6. Implement Target Variable Creation
- [ ] Create `_create_horizon_target()` method
- [ ] Shift target by horizon * periods (e.g., 48 for daily semi-hourly)
- [ ] Handle timezone-aware indices
- [ ] Align target with available feature data
- [ ] Return series with proper index alignment

### 7. Implement RFECV Selection
- [ ] Create `_select_features_rfecv()` method
- [ ] Use sklearn's RFECV with RandomForestRegressor
- [ ] Configure cross-validation strategy
- [ ] Use MAE scoring metric
- [ ] Extract selected feature names
- [ ] Return selection results with scores

### 8. Implement Permutation Importance Selection
- [ ] Create `_select_features_permutation()` method
- [ ] Train RandomForestRegressor on all features
- [ ] Calculate permutation importance
- [ ] Sort features by importance
- [ ] Select top-k or threshold-based
- [ ] Return importance scores and selected features

### 9. Implement SHAP-Based Selection
- [ ] Create `_select_features_shap()` method
- [ ] Import shap library
- [ ] Create TreeExplainer for trained RF
- [ ] Calculate SHAP values for feature set
- [ ] Aggregate absolute SHAP values per feature
- [ ] Select features by mean |SHAP| value
- [ ] Return SHAP-based rankings

### 10. Implement Combined Selection Strategy
- [ ] Create `_select_features_combined()` method
- [ ] Run multiple selection methods
- [ ] Aggregate feature scores using voting or ranking
- [ ] Compute consensus features (selected by multiple methods)
- [ ] Weight different methods based on configuration
- [ ] Return robust feature set

### 11. Implement Feature Importance Ranking
- [ ] Create `_rank_features_by_importance()` method
- [ ] Train final RF on selected features
- [ ] Extract feature_importances_ attribute
- [ ] Create ranking DataFrame
- [ ] Include multiple importance metrics if available
- [ ] Return ranked DataFrame

### 12. Implement generate_features() Method
- [ ] Validate input DataFrame structure
- [ ] Check target column exists
- [ ] Initialize leakage detector
- [ ] Loop through configured horizons
- [ ] Get horizon-safe features for each
- [ ] Create horizon-specific target
- [ ] Align features and target (dropna)
- [ ] Perform feature selection
- [ ] Store selected features per horizon
- [ ] Generate importance score features (optional)
- [ ] Return metadata about selections
- [ ] Log selection summary per horizon

### 13. Store Selection Metadata
- [ ] Create `selection_metadata` attribute
- [ ] Store selected features per horizon as dictionary
- [ ] Store importance scores per horizon
- [ ] Store selection method used
- [ ] Store cross-validation scores
- [ ] Provide `get_selected_features()` accessor method
- [ ] Provide `get_importance_scores()` accessor method

### 14. Implement Feature Selection Report
- [ ] Create `generate_selection_report()` method
- [ ] Generate markdown or DataFrame report
- [ ] Include features per horizon
- [ ] Include importance rankings
- [ ] Include cross-validation performance
- [ ] Include leakage detection results
- [ ] Save report to file if path provided

### 15. Write Comprehensive Tests
- [ ] Create `tests/features/plugins/test_rf_selector.py`
- [ ] Test basic feature selection for single horizon
- [ ] Test multi-horizon selection
- [ ] Test leakage detection (should reject future features)
- [ ] Test with different selection methods
- [ ] Test that selected features improve performance
- [ ] Test insufficient data handling
- [ ] Test edge case: more features than samples
- [ ] Mock SHAP calculations for speed
- [ ] Test performance benchmark

### 16. Create Leakage Detection Tests
- [ ] Create `tests/features/evaluation/test_leakage_detector.py`
- [ ] Test detection of future-looking features
- [ ] Test lag feature validation
- [ ] Test horizon-specific safety checks
- [ ] Test with known leakage scenarios
- [ ] Validate safe features pass checks

### 17. Create Usage Examples
- [ ] Create `examples/rf_feature_selection_demo.py`
- [ ] Show single horizon selection
- [ ] Show multi-horizon selection
- [ ] Visualize feature importance across horizons
- [ ] Compare different selection methods
- [ ] Show before/after model performance

---

## 💻 Implementation Details

### Configuration Schema

```python
"""Configuration for RF feature selector plugin."""
from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class RFFeatureSelectorConfig(BaseModel):
    """Configuration for Random Forest feature selector."""
    
    target_column: str = Field(
        default="carga",
        description="Target column for feature selection"
    )
    horizons: List[int] = Field(
        default_factory=lambda: list(range(0, 9)),  # D+0 to D+8
        description="Forecast horizons (in days) for selection"
    )
    max_features_per_horizon: int = Field(
        default=50,
        description="Maximum features to select per horizon"
    )
    selection_method: Literal["rfecv", "permutation", "shap", "combined"] = Field(
        default="rfecv",
        description="Feature selection method"
    )
    cv_folds: int = Field(
        default=5,
        description="Cross-validation folds for RFECV"
    )
    importance_threshold: float = Field(
        default=0.01,
        description="Minimum importance threshold for selection"
    )
    n_estimators: int = Field(
        default=50,
        description="Number of trees in Random Forest"
    )
    random_state: int = Field(
        default=42,
        description="Random state for reproducibility"
    )
    n_jobs: int = Field(
        default=-1,
        description="Parallel jobs (-1 for all cores)"
    )
    periods_per_day: int = Field(
        default=48,
        description="Periods per day (48 for semi-hourly)"
    )
    
    @field_validator('horizons')
    @classmethod
    def validate_horizons(cls, v):
        """Validate horizons are non-negative."""
        if not all(h >= 0 for h in v):
            raise ValueError("All horizons must be non-negative")
        return sorted(v)
    
    @field_validator('max_features_per_horizon')
    @classmethod
    def validate_max_features(cls, v):
        """Validate max features is positive."""
        if v < 1:
            raise ValueError("max_features_per_horizon must be >= 1")
        return v
```

### Data Leakage Detector

```python
"""Data leakage detection utilities."""
from typing import List, Set
import re
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLeakageDetector:
    """
    Detector for temporal data leakage in feature sets.
    
    Identifies features that would not be available at forecast time,
    preventing future information from leaking into the model.
    """
    
    def __init__(self, periods_per_day: int = 48):
        """
        Initialize detector.
        
        Args:
            periods_per_day: Number of periods per day (48 for semi-hourly)
        """
        self.periods_per_day = periods_per_day
        self.leakage_patterns = [
            r'.*future.*',
            r'.*forward.*',
            r'.*actual.*',  # Avoid actual values
            r'.*target.*',  # Avoid target derivatives
        ]
    
    def get_horizon_safe_features(
        self,
        features: List[str],
        horizon: int,
        target_column: str = "carga"
    ) -> List[str]:
        """
        Get features that are safe for given forecast horizon.
        
        Args:
            features: List of feature names
            horizon: Forecast horizon in days
            target_column: Name of target column
        
        Returns:
            List of safe feature names
        """
        safe_features = []
        
        for feature in features:
            # Skip target column
            if feature == target_column:
                continue
            
            # Check leakage patterns
            if self._matches_leakage_pattern(feature):
                logger.debug(f"Rejecting '{feature}': matches leakage pattern")
                continue
            
            # Validate lag features
            if 'lag' in feature.lower():
                if not self._is_lag_safe_for_horizon(feature, horizon):
                    logger.debug(
                        f"Rejecting '{feature}': insufficient lag for horizon {horizon}"
                    )
                    continue
            
            safe_features.append(feature)
        
        logger.info(
            f"Horizon {horizon}: {len(safe_features)}/{len(features)} features safe"
        )
        
        return safe_features
    
    def _matches_leakage_pattern(self, feature_name: str) -> bool:
        """Check if feature name matches leakage patterns."""
        feature_lower = feature_name.lower()
        
        for pattern in self.leakage_patterns:
            if re.match(pattern, feature_lower):
                return True
        
        return False
    
    def _is_lag_safe_for_horizon(self, feature_name: str, horizon: int) -> bool:
        """
        Check if lag feature has sufficient lag for horizon.
        
        Args:
            feature_name: Feature name (e.g., "carga_lag_24")
            horizon: Forecast horizon in days
        
        Returns:
            True if lag is sufficient
        """
        # Extract lag value from feature name
        lag_match = re.search(r'lag[_-](\d+)', feature_name, re.IGNORECASE)
        
        if not lag_match:
            # Not a standard lag feature, allow it
            return True
        
        lag_value = int(lag_match.group(1))
        
        # Required lag for horizon (in periods)
        required_lag = horizon * self.periods_per_day
        
        # Lag must be at least as large as horizon
        return lag_value >= required_lag
    
    def check_temporal_leakage(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        horizon: int
    ) -> dict:
        """
        Comprehensive temporal leakage check.
        
        Args:
            df: DataFrame with features and target
            feature_cols: Feature column names
            target_col: Target column name
            horizon: Forecast horizon
        
        Returns:
            Dictionary with leakage analysis results
        """
        results = {
            'has_leakage': False,
            'unsafe_features': [],
            'warnings': []
        }
        
        # Get safe features
        safe_features = self.get_horizon_safe_features(
            feature_cols,
            horizon,
            target_col
        )
        
        # Identify unsafe features
        unsafe = set(feature_cols) - set(safe_features)
        
        if unsafe:
            results['has_leakage'] = True
            results['unsafe_features'] = list(unsafe)
            results['warnings'].append(
                f"Found {len(unsafe)} potentially unsafe features for horizon {horizon}"
            )
        
        return results
```

### RFFeatureSelectorPlugin Implementation

```python
"""Random Forest feature selector plugin with horizon awareness."""
from typing import Dict, Any, List, Literal, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from src.features.base.plugin import AdvancedFeaturePlugin
from src.features.evaluation.leakage_detector import DataLeakageDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RFFeatureSelectorPlugin(AdvancedFeaturePlugin):
    """
    Random Forest-based horizon-aware feature selector.
    
    Performs feature selection independently for each forecast horizon,
    ensuring that selected features are available at forecast time and
    do not introduce temporal data leakage.
    
    Selection Methods:
    1. RFECV: Recursive Feature Elimination with Cross-Validation
       - Removes least important features iteratively
       - Selects optimal subset via CV performance
    
    2. Permutation Importance: Post-hoc importance via shuffling
       - Measures impact of each feature on model performance
       - More reliable than tree-based importance
    
    3. SHAP: SHapley Additive exPlanations
       - Game-theoretic feature importance
       - Consistent and accurate attribution
    
    4. Combined: Ensemble of multiple methods
       - More robust feature selection
       - Consensus across methods
    
    Example:
        >>> plugin = RFFeatureSelectorPlugin()
        >>> config = {
        ...     "target_column": "carga",
        ...     "horizons": [0, 1, 2, 3],
        ...     "selection_method": "rfecv",
        ...     "max_features_per_horizon": 30
        ... }
        >>> # Feature selection doesn't return features directly
        >>> # It returns selection metadata
        >>> result = plugin.generate_features(df, config)
        >>> selected = plugin.get_selected_features(horizon=1)
    """
    
    def __init__(self):
        """Initialize plugin with storage for selection results."""
        self.selection_metadata = {}
        self.leakage_detector = None
    
    @property
    def name(self) -> str:
        return "rf_feature_selector"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def computational_complexity(self) -> Literal["low", "medium", "high"]:
        return "high"
    
    def estimate_compute_time(self, data_size: int) -> float:
        """
        Estimate selection time.
        
        RFECV is expensive: multiple CV folds × feature iterations × RF training
        """
        # Empirical: ~60 seconds per horizon for 1 year of data
        return 60.0 * (data_size / 8760)
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration."""
        RFFeatureSelectorConfig(**config)
        logger.debug("RF feature selector config validated")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Perform horizon-aware feature selection.
        
        Note: This plugin doesn't generate new features in the traditional sense.
        Instead, it performs feature selection and stores the results in
        self.selection_metadata. Use get_selected_features() to retrieve results.
        
        Args:
            df: Input DataFrame with all features
            config: Plugin configuration
        
        Returns:
            Empty DataFrame (selection results stored in metadata)
        """
        # Validate config
        validated_config = RFFeatureSelectorConfig(**config)
        
        # Initialize leakage detector
        self.leakage_detector = DataLeakageDetector(
            periods_per_day=validated_config.periods_per_day
        )
        
        # Validate target column
        if validated_config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{validated_config.target_column}' not found"
            )
        
        logger.info(
            f"Starting feature selection for {len(validated_config.horizons)} horizons "
            f"using {validated_config.selection_method} method"
        )
        
        # Get all feature columns (excluding target)
        feature_cols = [
            col for col in df.columns
            if col != validated_config.target_column
        ]
        
        # Perform selection for each horizon
        for horizon in validated_config.horizons:
            logger.info(f"Selecting features for horizon D+{horizon}")
            
            try:
                selected_features = self._select_for_horizon(
                    df,
                    feature_cols,
                    validated_config.target_column,
                    horizon,
                    validated_config
                )
                
                self.selection_metadata[f'horizon_{horizon}'] = selected_features
                
                logger.info(
                    f"Horizon D+{horizon}: selected {len(selected_features['features'])} features"
                )
                
            except Exception as e:
                logger.error(f"Feature selection failed for horizon {horizon}: {e}")
                self.selection_metadata[f'horizon_{horizon}'] = {
                    'features': [],
                    'importance': {},
                    'error': str(e)
                }
        
        # Return empty DataFrame (results in metadata)
        logger.info("Feature selection complete. Use get_selected_features() to retrieve results.")
        return pd.DataFrame(index=df.index)
    
    def _select_for_horizon(
        self,
        df: pd.DataFrame,
        all_features: List[str],
        target_col: str,
        horizon: int,
        config: RFFeatureSelectorConfig
    ) -> Dict[str, Any]:
        """
        Select features for specific horizon.
        
        Args:
            df: Input DataFrame
            all_features: All available feature names
            target_col: Target column name
            horizon: Forecast horizon (days)
            config: Configuration object
        
        Returns:
            Dictionary with selected features and metadata
        """
        # Get horizon-safe features
        safe_features = self.leakage_detector.get_horizon_safe_features(
            all_features,
            horizon,
            target_col
        )
        
        if len(safe_features) == 0:
            logger.warning(f"No safe features for horizon {horizon}")
            return {'features': [], 'importance': {}}
        
        # Create horizon-specific target
        target = self._create_horizon_target(
            df[target_col],
            horizon,
            config.periods_per_day
        )
        
        # Prepare feature matrix
        X = df[safe_features].copy()
        y = target.copy()
        
        # Align and drop NaN
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Drop rows with any NaN
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X.loc[valid_idx]
        y_clean = y.loc[valid_idx]
        
        if len(X_clean) < 100:
            logger.warning(
                f"Insufficient samples for horizon {horizon}: {len(X_clean)}"
            )
            return {'features': safe_features[:config.max_features_per_horizon], 'importance': {}}
        
        logger.info(
            f"Horizon {horizon}: {len(X_clean)} samples, {len(safe_features)} features"
        )
        
        # Perform selection based on method
        if config.selection_method == "rfecv":
            selected = self._select_features_rfecv(
                X_clean, y_clean, config
            )
        elif config.selection_method == "permutation":
            selected = self._select_features_permutation(
                X_clean, y_clean, config
            )
        elif config.selection_method == "shap":
            selected = self._select_features_shap(
                X_clean, y_clean, config
            )
        elif config.selection_method == "combined":
            selected = self._select_features_combined(
                X_clean, y_clean, config
            )
        else:
            raise ValueError(f"Unknown selection method: {config.selection_method}")
        
        # Limit to max features
        if len(selected['features']) > config.max_features_per_horizon:
            # Sort by importance and take top-k
            sorted_features = sorted(
                selected['features'],
                key=lambda f: selected['importance'].get(f, 0),
                reverse=True
            )
            selected['features'] = sorted_features[:config.max_features_per_horizon]
        
        return selected
    
    def _create_horizon_target(
        self,
        target: pd.Series,
        horizon: int,
        periods_per_day: int
    ) -> pd.Series:
        """Create horizon-shifted target variable."""
        shift_periods = horizon * periods_per_day
        return target.shift(-shift_periods)
    
    def _select_features_rfecv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: RFFeatureSelectorConfig
    ) -> Dict[str, Any]:
        """Select features using RFECV."""
        logger.info("Running RFECV feature selection")
        
        # Create Random Forest
        rf = RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=config.n_jobs
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=config.cv_folds)
        
        # RFECV
        selector = RFECV(
            estimator=rf,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=config.n_jobs,
            min_features_to_select=min(5, len(X.columns))
        )
        
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.support_].tolist()
        
        # Get feature importances
        rf.fit(X[selected_features], y)
        importance_dict = dict(zip(selected_features, rf.feature_importances_))
        
        return {
            'features': selected_features,
            'importance': importance_dict,
            'cv_scores': selector.cv_results_,
            'optimal_n_features': selector.n_features_
        }
    
    def _select_features_permutation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: RFFeatureSelectorConfig
    ) -> Dict[str, Any]:
        """Select features using permutation importance."""
        logger.info("Running permutation importance feature selection")
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=config.n_jobs
        )
        
        rf.fit(X, y)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            rf, X, y,
            n_repeats=10,
            random_state=config.random_state,
            n_jobs=config.n_jobs
        )
        
        # Create importance dictionary
        importance_dict = dict(zip(X.columns, perm_importance.importances_mean))
        
        # Select features above threshold
        selected_features = [
            feature for feature, importance in importance_dict.items()
            if importance >= config.importance_threshold
        ]
        
        return {
            'features': selected_features,
            'importance': importance_dict,
            'importance_std': dict(zip(X.columns, perm_importance.importances_std))
        }
    
    def _select_features_shap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: RFFeatureSelectorConfig
    ) -> Dict[str, Any]:
        """Select features using SHAP values."""
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not available, falling back to permutation importance")
            return self._select_features_permutation(X, y, config)
        
        logger.info("Running SHAP-based feature selection")
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=config.n_jobs
        )
        
        rf.fit(X, y)
        
        # Calculate SHAP values (use subset for speed)
        sample_size = min(500, len(X))
        X_sample = X.sample(n=sample_size, random_state=config.random_state)
        
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)
        
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_dict = dict(zip(X.columns, mean_abs_shap))
        
        # Select features above threshold
        threshold = np.percentile(mean_abs_shap, 100 * (1 - config.max_features_per_horizon / len(X.columns)))
        selected_features = [
            feature for feature, shap_val in importance_dict.items()
            if shap_val >= threshold
        ]
        
        return {
            'features': selected_features,
            'importance': importance_dict,
            'method': 'shap'
        }
    
    def _select_features_combined(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: RFFeatureSelectorConfig
    ) -> Dict[str, Any]:
        """Combine multiple selection methods."""
        logger.info("Running combined feature selection")
        
        # Run RFECV
        rfecv_results = self._select_features_rfecv(X, y, config)
        
        # Run permutation
        perm_results = self._select_features_permutation(X, y, config)
        
        # Consensus: features selected by both methods
        consensus_features = list(
            set(rfecv_results['features']) & set(perm_results['features'])
        )
        
        # If too few, add high-ranking features from either method
        if len(consensus_features) < config.max_features_per_horizon // 2:
            all_features = set(rfecv_results['features']) | set(perm_results['features'])
            consensus_features = list(all_features)[:config.max_features_per_horizon]
        
        # Combine importance scores (average)
        combined_importance = {}
        for feature in consensus_features:
            rfecv_imp = rfecv_results['importance'].get(feature, 0)
            perm_imp = perm_results['importance'].get(feature, 0)
            combined_importance[feature] = (rfecv_imp + perm_imp) / 2
        
        return {
            'features': consensus_features,
            'importance': combined_importance,
            'rfecv_selected': rfecv_results['features'],
            'perm_selected': perm_results['features']
        }
    
    def get_selected_features(self, horizon: int) -> List[str]:
        """
        Get selected features for specific horizon.
        
        Args:
            horizon: Forecast horizon (days)
        
        Returns:
            List of selected feature names
        """
        key = f'horizon_{horizon}'
        if key not in self.selection_metadata:
            logger.warning(f"No selection results for horizon {horizon}")
            return []
        
        return self.selection_metadata[key].get('features', [])
    
    def get_importance_scores(self, horizon: int) -> Dict[str, float]:
        """
        Get feature importance scores for horizon.
        
        Args:
            horizon: Forecast horizon (days)
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        key = f'horizon_{horizon}'
        if key not in self.selection_metadata:
            return {}
        
        return self.selection_metadata[key].get('importance', {})
    
    def generate_selection_report(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate feature selection report.
        
        Args:
            output_path: Optional path to save report
        
        Returns:
            DataFrame with selection summary
        """
        report_data = []
        
        for horizon_key, metadata in self.selection_metadata.items():
            horizon = int(horizon_key.split('_')[1])
            
            report_data.append({
                'horizon': horizon,
                'n_features': len(metadata.get('features', [])),
                'top_feature': (
                    max(metadata.get('importance', {}), 
                        key=metadata.get('importance', {}).get, default='N/A')
                    if metadata.get('importance') else 'N/A'
                ),
                'mean_importance': (
                    np.mean(list(metadata.get('importance', {}).values()))
                    if metadata.get('importance') else 0
                )
            })
        
        report_df = pd.DataFrame(report_data).sort_values('horizon')
        
        if output_path:
            report_df.to_csv(output_path, index=False)
            logger.info(f"Selection report saved to {output_path}")
        
        return report_df
    
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """This plugin doesn't generate features directly."""
        return []
```

---

## 🧪 Testing & Validation

### Unit Tests

```python
"""Tests for RF feature selector plugin."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.features.plugins.rf_selector import RFFeatureSelectorPlugin, RFFeatureSelectorConfig
from src.features.evaluation.leakage_detector import DataLeakageDetector


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return RFFeatureSelectorPlugin()


@pytest.fixture
def sample_df():
    """Create sample DataFrame with features."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=500, freq="h")
    
    # Target
    target = 1000 + np.random.randn(500) * 50
    
    # Safe features (temporal, calendar)
    hour = dates.hour
    day_of_week = dates.dayofweek
    
    # Lag features
    lag_24 = pd.Series(target).shift(24).values
    lag_48 = pd.Series(target).shift(48).values
    
    # Unsafe features (future looking)
    future_value = pd.Series(target).shift(-24).values
    
    return pd.DataFrame({
        'carga': target,
        'hour': hour,
        'day_of_week': day_of_week,
        'carga_lag_24': lag_24,
        'carga_lag_48': lag_48,
        'future_value': future_value,  # Should be filtered
        'temperature': np.random.randn(500) + 25
    }, index=dates)


def test_basic_feature_selection(plugin, sample_df):
    """Test basic feature selection for single horizon."""
    config = {
        'target_column': 'carga',
        'horizons': [0],
        'max_features_per_horizon': 5,
        'selection_method': 'permutation',
        'n_estimators': 10  # Fast
    }
    
    result = plugin.generate_features(sample_df, config)
    
    # Should have selection metadata
    selected = plugin.get_selected_features(horizon=0)
    assert len(selected) > 0
    assert len(selected) <= 5


def test_leakage_detection(plugin, sample_df):
    """Test that future-looking features are filtered."""
    config = {
        'target_column': 'carga',
        'horizons': [1],
        'max_features_per_horizon': 10
    }
    
    plugin.generate_features(sample_df, config)
    selected = plugin.get_selected_features(horizon=1)
    
    # 'future_value' should not be selected
    assert 'future_value' not in selected


def test_lag_feature_validation():
    """Test lag feature validation for horizons."""
    detector = DataLeakageDetector(periods_per_day=24)
    
    # Lag 24 is safe for horizon 0
    assert detector._is_lag_safe_for_horizon('carga_lag_24', horizon=0)
    
    # Lag 24 is NOT safe for horizon 2 (requires lag >= 48)
    assert not detector._is_lag_safe_for_horizon('carga_lag_24', horizon=2)
    
    # Lag 48 is safe for horizon 2
    assert detector._is_lag_safe_for_horizon('carga_lag_48', horizon=2)


def test_multi_horizon_selection(plugin, sample_df):
    """Test selection for multiple horizons."""
    config = {
        'target_column': 'carga',
        'horizons': [0, 1, 2],
        'max_features_per_horizon': 5,
        'n_estimators': 10
    }
    
    plugin.generate_features(sample_df, config)
    
    # Should have selections for all horizons
    for horizon in [0, 1, 2]:
        selected = plugin.get_selected_features(horizon)
        assert len(selected) > 0


def test_rfecv_selection(plugin, sample_df):
    """Test RFECV selection method."""
    config = {
        'target_column': 'carga',
        'horizons': [0],
        'selection_method': 'rfecv',
        'n_estimators': 10,
        'cv_folds': 3
    }
    
    plugin.generate_features(sample_df, config)
    selected = plugin.get_selected_features(horizon=0)
    
    assert len(selected) > 0


@patch('shap.TreeExplainer')
def test_shap_selection(mock_shap, plugin, sample_df):
    """Test SHAP selection method (mocked for speed)."""
    config = {
        'target_column': 'carga',
        'horizons': [0],
        'selection_method': 'shap',
        'n_estimators': 10
    }
    
    # Mock SHAP values
    mock_explainer = mock_shap.return_value
    mock_explainer.shap_values.return_value = np.random.randn(100, 5)
    
    plugin.generate_features(sample_df, config)
    selected = plugin.get_selected_features(horizon=0)
    
    assert len(selected) > 0


def test_importance_scores(plugin, sample_df):
    """Test feature importance score retrieval."""
    config = {
        'target_column': 'carga',
        'horizons': [0],
        'max_features_per_horizon': 5
    }
    
    plugin.generate_features(sample_df, config)
    importance = plugin.get_importance_scores(horizon=0)
    
    assert isinstance(importance, dict)
    assert len(importance) > 0
    # All importance values should be non-negative
    assert all(v >= 0 for v in importance.values())


def test_selection_report(plugin, sample_df):
    """Test selection report generation."""
    config = {
        'target_column': 'carga',
        'horizons': [0, 1],
        'max_features_per_horizon': 5
    }
    
    plugin.generate_features(sample_df, config)
    report = plugin.generate_selection_report()
    
    assert isinstance(report, pd.DataFrame)
    assert len(report) == 2  # Two horizons
    assert 'horizon' in report.columns
    assert 'n_features' in report.columns


def test_insufficient_data_handling(plugin):
    """Test handling of insufficient data."""
    small_df = pd.DataFrame({
        'carga': np.random.randn(50),
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50)
    }, index=pd.date_range("2024-01-01", periods=50, freq="h"))
    
    config = {
        'target_column': 'carga',
        'horizons': [0],
        'max_features_per_horizon': 3
    }
    
    # Should not crash
    plugin.generate_features(small_df, config)


def test_config_validation():
    """Test configuration validation."""
    # Negative horizon
    with pytest.raises(ValueError, match="non-negative"):
        RFFeatureSelectorConfig(horizons=[-1, 0, 1])
    
    # Invalid max features
    with pytest.raises(ValueError, match="must be >= 1"):
        RFFeatureSelectorConfig(max_features_per_horizon=0)


def test_computational_complexity(plugin):
    """Test complexity reporting."""
    assert plugin.computational_complexity == "high"


def test_estimate_compute_time(plugin):
    """Test time estimation."""
    time_estimate = plugin.estimate_compute_time(8760)  # 1 year
    
    # Should estimate reasonable time
    assert 30 < time_estimate < 300  # 30s to 5min per horizon
```

---

## 📝 Technical Notes

### Temporal Leakage Prevention
- **Lag Validation**: Lag features must have lag >= horizon × periods_per_day
- **Pattern Matching**: Reject features with "future", "forward", "actual" in name
- **Target Exclusion**: Never use target variable or its direct transformations

### Feature Selection Methods
- **RFECV**: Best for finding optimal subset, but computationally expensive
- **Permutation**: Fast and reliable importance, good for large feature sets
- **SHAP**: Most accurate attribution, slower but provides interpretability
- **Combined**: Most robust, uses consensus across methods

### Performance Considerations
- RFECV: O(n_features × n_cv_folds × n_estimators × n_samples × log(n_samples))
- Permutation: O(n_repeats × n_features × prediction_time)
- SHAP: O(n_samples × n_estimators × 2^n_features) (approximate with sampling)

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation
- PC-018-02B: LOESS Smoothing Plugin (AdvancedFeaturePlugin interface)
- Epic-02A: Core feature pipeline (provides features to select from)

**External Dependencies:**
- `scikit-learn>=1.3.0` (RandomForest, RFECV, permutation_importance)
- `shap>=0.42.0` (optional, for SHAP-based selection)

**Blocks:**
- Model training pipelines (provides optimal feature sets)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] RFFeatureSelectorPlugin implemented with horizon-aware selection
- [ ] Data leakage detection working correctly
- [ ] All 4 selection methods (RFECV, permutation, SHAP, combined) functional
- [ ] Temporal leakage prevention validated
- [ ] Selection metadata storage and retrieval working
- [ ] Selection report generation functional
- [ ] Unit tests pass with >85% coverage
- [ ] Integration test with full feature pipeline passes
- [ ] Performance benchmark met (<5min per area for all horizons)
- [ ] Documentation complete with methodology explanation
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-020-02B: BLF Strategy Plugin](PC-020-02B-blf-strategy-plugin.md)  
**Next:** [PC-022-02B: Seasonality Calculation Plugin](PC-022-02B-seasonality-calculation-plugin.md)
