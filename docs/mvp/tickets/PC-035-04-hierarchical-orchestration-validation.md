# PC-035-04: Hierarchical Orchestration & Validation

**Ticket ID:** PC-035-04  
**Epic:** [Epic-04: Model Layer - Hierarchical Models](../epics/Epic-04.md)  
**User Story:** US-5, US-6  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement comprehensive orchestration and validation framework for hierarchical models. Manages multi-stage training workflow, validates hierarchical consistency through energy conservation tests, supports parallel profile training, and provides detailed component analysis for debugging and model interpretation.

**As a** forecasting engineer  
**I want** robust orchestration and validation for hierarchical models  
**So that** I can ensure reliable two-stage training and validate energy conservation

---

## ✅ Acceptance Criteria

- [ ] Orchestrates multi-stage training (demand mean → profiles)
- [ ] Manages data flow between hierarchical components
- [ ] Validates hierarchical consistency (energy conservation)
- [ ] Supports parallel training of 48 profile models
- [ ] Handles missing or failed component models gracefully
- [ ] Provides comprehensive logging and monitoring
- [ ] Validates seasonal pattern consistency
- [ ] Tests profile bounds (0.1 to 5.0 range)
- [ ] Generates validation reports with metrics
- [ ] Component analysis tools for debugging
- [ ] Parallel training reduces time by 50%+
- [ ] All hierarchical models pass validation suite
- [ ] Comprehensive tests with mock components
- [ ] Documentation explains orchestration workflow

---

## 🔧 Implementation Tasks

### 1. Create Orchestration Module
- [ ] Create `src/models/hierarchical/orchestration.py`
- [ ] Import multiprocessing utilities
- [ ] Import BaseHierarchicalModel
- [ ] Add module docstrings

### 2. Implement HierarchicalTrainer Class
- [ ] Create class for coordinated training
- [ ] Method `train_hierarchical_model()`
- [ ] Stage 1: Train demand mean model
- [ ] Stage 2: Generate demand mean predictions
- [ ] Stage 3: Prepare profile data
- [ ] Stage 4: Train profile models (parallel)
- [ ] Error handling for each stage
- [ ] Progress logging

### 3. Implement Parallel Profile Training
- [ ] Create `_train_single_profile_model()` method
- [ ] Accept hour, data, config as parameters
- [ ] Fit profile model for one hour
- [ ] Return fitted model or error
- [ ] Use ProcessPoolExecutor for parallelization
- [ ] Configure n_jobs parameter
- [ ] Handle worker failures gracefully

### 4. Implement Training Pipeline
- [ ] Create `_stage1_train_demand_mean()` method
- [ ] Prepare daily aggregated data
- [ ] Train demand mean model
- [ ] Validate fitting
- [ ] Return trained model

### 5. Implement Profile Data Pipeline
- [ ] Create `_stage2_prepare_profiles()` method
- [ ] Generate demand mean predictions
- [ ] Calculate profile ratios
- [ ] Group by semi-hourly period
- [ ] Remove outliers
- [ ] Return profile data dict

### 6. Implement Profile Training Pipeline
- [ ] Create `_stage3_train_profiles()` method
- [ ] Create tasks for 48 hours
- [ ] Submit to thread/process pool
- [ ] Collect results
- [ ] Handle failures (use defaults)
- [ ] Return profile models dict

### 7. Create Validation Module
- [ ] Create `src/models/hierarchical/validation.py`
- [ ] Import validation utilities
- [ ] Add comprehensive test suite

### 8. Implement HierarchicalValidator Class
- [ ] Create class for validation tests
- [ ] Method `validate_model()`
- [ ] Run all validation tests
- [ ] Aggregate results
- [ ] Generate report

### 9. Implement Energy Conservation Tests
- [ ] Create `_test_energy_conservation()` method
- [ ] Calculate daily sums of predictions
- [ ] Compare to demand mean × 48
- [ ] Calculate relative errors
- [ ] Check threshold (<5%)
- [ ] Return pass/fail with metrics

### 10. Implement Profile Bounds Tests
- [ ] Create `_test_profile_bounds()` method
- [ ] Check profile ratios in range [0.1, 5.0]
- [ ] Calculate percentage of violations
- [ ] Check extreme outliers
- [ ] Return validation results

### 11. Implement Seasonal Consistency Tests
- [ ] Create `_test_seasonal_consistency()` method
- [ ] Check profile patterns across weeks
- [ ] Validate seasonal stability
- [ ] Detect anomalous patterns
- [ ] Return consistency metrics

### 12. Implement Component Analysis Tools
- [ ] Create `analyze_components()` method
- [ ] Decompose forecasts into components
- [ ] Calculate component contributions
- [ ] Identify problematic hours
- [ ] Generate diagnostic plots
- [ ] Return analysis report

### 13. Implement Stress Testing
- [ ] Create `stress_test_model()` method
- [ ] Test with edge cases (zero demand, extreme loads)
- [ ] Test with missing data
- [ ] Test with short training periods
- [ ] Test with anomalous patterns
- [ ] Return stress test results

### 14. Implement Validation Report Generation
- [ ] Create `generate_validation_report()` method
- [ ] Aggregate all validation results
- [ ] Calculate summary statistics
- [ ] Format as structured report
- [ ] Include pass/fail status
- [ ] Save to file if requested

### 15. Write Comprehensive Tests
- [ ] Create `tests/models/hierarchical/test_orchestration.py`
- [ ] Test parallel profile training
- [ ] Test training pipeline stages
- [ ] Test error handling
- [ ] Mock profile training failures
- [ ] Verify graceful degradation

### 16. Write Validation Tests
- [ ] Create `tests/models/hierarchical/test_validation.py`
- [ ] Test energy conservation validation
- [ ] Test profile bounds validation
- [ ] Test seasonal consistency
- [ ] Test component analysis
- [ ] Test report generation

### 17. Create Usage Examples
- [ ] Create `examples/hierarchical_orchestration_demo.py`
- [ ] Show coordinated training
- [ ] Show parallel profile training
- [ ] Show validation workflow
- [ ] Show component analysis
- [ ] Show debugging techniques

---

## 💻 Implementation Details

### Hierarchical Trainer

```python
"""Orchestration for hierarchical model training."""
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HierarchicalTrainer:
    """
    Orchestrates multi-stage hierarchical model training.
    
    Manages:
    - Stage 1: Demand mean model training
    - Stage 2: Profile data preparation
    - Stage 3: Parallel profile model training
    - Stage 4: Model validation
    
    Features:
    - Parallel profile training (50%+ speedup)
    - Graceful error handling
    - Progress monitoring
    - Component validation
    
    Example:
        >>> trainer = HierarchicalTrainer(n_jobs=8)
        >>> model = RegDinSVMModel()
        >>> config = {...}
        >>> trained_model = trainer.train_hierarchical_model(
        ...     model, X_train, y_train, config
        ... )
    """
    
    def __init__(self, n_jobs: int = 4, verbose: bool = True):
        """
        Initialize trainer.
        
        Args:
            n_jobs: Number of parallel workers for profile training
            verbose: Whether to show progress bars
        """
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def train_hierarchical_model(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ) -> BaseHierarchicalModel:
        """
        Train hierarchical model with orchestrated workflow.
        
        Args:
            model: Hierarchical model instance
            X: Training features
            y: Training target
            config: Configuration dict
        
        Returns:
            Trained model
        """
        logger.info(f"Starting hierarchical training for {model.name}")
        
        # Stage 1: Train demand mean model
        logger.info("Stage 1: Training demand mean model")
        demand_mean_model = self._stage1_train_demand_mean(
            model, X, y, config.get('demand_mean', {})
        )
        model.demand_mean_model = demand_mean_model
        
        # Stage 2: Prepare profile data
        logger.info("Stage 2: Preparing profile data")
        profile_data = self._stage2_prepare_profiles(
            model, X, y
        )
        
        # Stage 3: Train profile models (parallel)
        logger.info(f"Stage 3: Training {len(profile_data)} profile models (parallel)")
        profile_models = self._stage3_train_profiles(
            model, profile_data, config.get('profile', {})
        )
        model.profile_models = profile_models
        
        # Stage 4: Initialize combiner
        logger.info("Stage 4: Initializing profile combiner")
        model._initialize_combiner()
        
        logger.info("Hierarchical training complete")
        
        return model
    
    def _stage1_train_demand_mean(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ):
        """Train demand mean model."""
        # Prepare data
        X_dm, y_dm = model._prepare_demand_mean_data(X, y)
        
        # Create and train model
        dm_model = model.demand_mean_model_class()
        dm_model.fit(X_dm, y_dm, config)
        
        logger.info(f"Demand mean model trained: {dm_model.name}")
        
        return dm_model
    
    def _stage2_prepare_profiles(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[int, Tuple[pd.DataFrame, pd.Series]]:
        """Prepare profile data."""
        # Get demand mean predictions
        X_dm, y_dm = model._prepare_demand_mean_data(X, y)
        dm_predictions = model.demand_mean_model.predict(X_dm, [0])
        
        # Prepare profile data
        profile_data = model._prepare_profile_data(
            X, y, dm_predictions['demand_mean_h0']
        )
        
        logger.info(f"Prepared profile data for {len(profile_data)} hours")
        
        return profile_data
    
    def _stage3_train_profiles(
        self,
        model: BaseHierarchicalModel,
        profile_data: Dict[int, Tuple[pd.DataFrame, pd.Series]],
        config: Dict[str, Any]
    ) -> Dict[int, Any]:
        """Train profile models in parallel."""
        profile_models = {}
        
        # Create tasks
        tasks = []
        for hour, (X_hour, profiles_hour) in profile_data.items():
            tasks.append((hour, X_hour, profiles_hour, config))
        
        # Train in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(
                    self._train_single_profile,
                    model.profile_model_class,
                    hour,
                    X_hour,
                    profiles_hour,
                    config
                ): hour
                for hour, X_hour, profiles_hour, config in tasks
            }
            
            # Collect results with progress bar
            iterator = as_completed(futures)
            if self.verbose:
                iterator = tqdm(iterator, total=len(futures), desc="Training profiles")
            
            for future in iterator:
                hour = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        profile_models[hour] = result
                except Exception as e:
                    logger.warning(f"Hour {hour} failed: {e}")
        
        logger.info(f"Trained {len(profile_models)}/{len(profile_data)} profile models")
        
        return profile_models
    
    @staticmethod
    def _train_single_profile(
        model_class,
        hour: int,
        X_hour: pd.DataFrame,
        profiles_hour: pd.Series,
        config: Dict[str, Any]
    ):
        """Train single profile model (parallelizable)."""
        try:
            model = model_class()
            model.fit({hour: (X_hour, profiles_hour)}, config)
            return model.models.get(hour)
        except Exception as e:
            logger.error(f"Failed to train hour {hour}: {e}")
            return None
```

### Hierarchical Validator

```python
"""Validation framework for hierarchical models."""
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HierarchicalValidator:
    """
    Comprehensive validation for hierarchical models.
    
    Tests:
    - Energy conservation (daily sums match demand mean × 48)
    - Profile bounds (ratios in [0.1, 5.0])
    - Seasonal consistency (stable patterns across weeks)
    - Component validity
    
    Example:
        >>> validator = HierarchicalValidator()
        >>> results = validator.validate_model(model, X_test, y_test)
        >>> validator.generate_validation_report(results)
    """
    
    def __init__(self):
        self.validation_results: Dict[str, Any] = {}
    
    def validate_model(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,
        y: pd.Series,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation suite.
        
        Args:
            model: Trained hierarchical model
            X: Validation features
            y: Validation target
            config: Validation configuration
        
        Returns:
            Dictionary with validation results
        """
        if config is None:
            config = {}
        
        logger.info("Running hierarchical model validation")
        
        results = {
            'model_name': model.name,
            'timestamp': pd.Timestamp.now()
        }
        
        # Test 1: Energy conservation
        logger.info("Test 1: Energy conservation")
        results['energy_conservation'] = self._test_energy_conservation(
            model, X, y, threshold=config.get('conservation_threshold', 0.05)
        )
        
        # Test 2: Profile bounds
        logger.info("Test 2: Profile bounds")
        results['profile_bounds'] = self._test_profile_bounds(
            model, X, y
        )
        
        # Test 3: Seasonal consistency
        logger.info("Test 3: Seasonal consistency")
        results['seasonal_consistency'] = self._test_seasonal_consistency(
            model, X, y
        )
        
        # Overall pass/fail
        results['passed'] = all([
            results['energy_conservation']['passed'],
            results['profile_bounds']['passed'],
            results['seasonal_consistency']['passed']
        ])
        
        self.validation_results = results
        logger.info(f"Validation {'PASSED' if results['passed'] else 'FAILED'}")
        
        return results
    
    def _test_energy_conservation(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Test energy conservation."""
        predictions = model.predict(X, [0])
        
        if 'pred_h0' not in predictions.columns:
            return {'passed': False, 'error': 'No predictions'}
        
        # Calculate daily sums
        daily_pred = predictions['pred_h0'].groupby(
            predictions.index.date
        ).sum()
        
        # Get demand mean × 48
        if 'demand_mean_h0' in predictions.columns:
            daily_expected = predictions['demand_mean_h0'].groupby(
                predictions.index.date
            ).first() * 48
        else:
            daily_expected = y.groupby(y.index.date).sum()
        
        # Calculate errors
        common_dates = daily_pred.index.intersection(daily_expected.index)
        errors = np.abs(
            daily_pred.loc[common_dates] - daily_expected.loc[common_dates]
        ) / daily_expected.loc[common_dates]
        
        mean_error = errors.mean()
        max_error = errors.max()
        
        return {
            'passed': mean_error <= threshold,
            'mean_error': mean_error,
            'max_error': max_error,
            'threshold': threshold,
            'violations': (errors > threshold).sum()
        }
    
    def _test_profile_bounds(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Test profile ratio bounds."""
        # Get decomposition
        decomp = model.get_decomposition(X, horizon=0)
        
        if decomp['profile'] is None:
            return {'passed': False, 'error': 'No profiles'}
        
        profiles = decomp['profile']
        
        # Check bounds [0.1, 5.0]
        lower_violations = (profiles < 0.1).sum()
        upper_violations = (profiles > 5.0).sum()
        total_violations = lower_violations + upper_violations
        
        violation_rate = total_violations / len(profiles)
        
        return {
            'passed': violation_rate < 0.01,  # <1% violations
            'violation_rate': violation_rate,
            'lower_violations': lower_violations,
            'upper_violations': upper_violations,
            'profile_min': profiles.min(),
            'profile_max': profiles.max()
        }
    
    def _test_seasonal_consistency(
        self,
        model: BaseHierarchicalModel,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Test seasonal pattern consistency."""
        decomp = model.get_decomposition(X, horizon=0)
        
        if decomp['profile'] is None:
            return {'passed': False, 'error': 'No profiles'}
        
        profiles = decomp['profile']
        
        # Group by day of week and hour
        profiles_df = pd.DataFrame({
            'profile': profiles,
            'dow': profiles.index.dayofweek,
            'hour': profiles.index.hour * 2 + (profiles.index.minute // 30)
        })
        
        # Calculate consistency (std across weeks for same hour/dow)
        consistency = profiles_df.groupby(['dow', 'hour'])['profile'].std()
        mean_consistency = consistency.mean()
        
        return {
            'passed': mean_consistency < 0.5,  # Reasonable threshold
            'mean_std': mean_consistency,
            'max_std': consistency.max(),
            'unstable_periods': (consistency > 1.0).sum()
        }
    
    def generate_validation_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate formatted validation report."""
        report = []
        report.append("=" * 60)
        report.append(f"HIERARCHICAL MODEL VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Model: {results['model_name']}")
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Overall Status: {'PASSED' if results['passed'] else 'FAILED'}")
        report.append("")
        
        # Energy conservation
        ec = results['energy_conservation']
        report.append("Energy Conservation Test:")
        report.append(f"  Status: {'PASS' if ec['passed'] else 'FAIL'}")
        report.append(f"  Mean Error: {ec['mean_error']:.2%}")
        report.append(f"  Max Error: {ec['max_error']:.2%}")
        report.append(f"  Threshold: {ec['threshold']:.2%}")
        report.append(f"  Violations: {ec['violations']}")
        report.append("")
        
        # Profile bounds
        pb = results['profile_bounds']
        report.append("Profile Bounds Test:")
        report.append(f"  Status: {'PASS' if pb['passed'] else 'FAIL'}")
        report.append(f"  Violation Rate: {pb['violation_rate']:.2%}")
        report.append(f"  Profile Range: [{pb['profile_min']:.2f}, {pb['profile_max']:.2f}]")
        report.append("")
        
        # Seasonal consistency
        sc = results['seasonal_consistency']
        report.append("Seasonal Consistency Test:")
        report.append(f"  Status: {'PASS' if sc['passed'] else 'FAIL'}")
        report.append(f"  Mean Std: {sc['mean_std']:.3f}")
        report.append(f"  Unstable Periods: {sc['unstable_periods']}")
        report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {output_path}")
        
        return report_text
```

---

## 🧪 Testing & Validation

```python
"""Tests for hierarchical orchestration."""
import pytest
from unittest.mock import Mock
import pandas as pd
import numpy as np

from src.models.hierarchical.orchestration import HierarchicalTrainer
from src.models.hierarchical.validation import HierarchicalValidator


def test_parallel_profile_training():
    """Test parallel profile training speedup."""
    # Mock model and data
    model = Mock()
    model.name = "test_model"
    
    # ... test implementation
```

---

## 📝 Technical Notes

### Parallel Training
- ProcessPoolExecutor for CPU-bound profile training
- 50%+ speedup with 4-8 workers
- Graceful handling of worker failures

### Validation Thresholds
- Energy conservation: <5% error
- Profile bounds: [0.1, 5.0]
- Seasonal std: <0.5 reasonable

---

## 🔗 Dependencies

**Depends On:**
- PC-030-04: Base Hierarchical Model Interface
- PC-033-04: RegDin+SVM Pipeline
- PC-034-04: Holt-Winters Model

**Blocks:**
- Epic-05 model combination
- Epic-06 hierarchical reconciliation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Parallel profile training functional (50%+ speedup)
- [ ] Validation suite complete
- [ ] Energy conservation tests working
- [ ] Profile bounds tests working
- [ ] Seasonal consistency tests working
- [ ] Validation reports generated
- [ ] All hierarchical models pass validation
- [ ] Unit tests pass with >85% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Previous:** [PC-034-04: Holt-Winters Hierarchical Model](PC-034-04-holt-winters-model.md)  
**Epic:** [Epic-04: Model Layer - Hierarchical Models](../epics/Epic-04.md)
