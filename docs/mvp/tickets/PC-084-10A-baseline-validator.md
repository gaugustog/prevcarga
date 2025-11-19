# PC-084-10A: Baseline Validator Implementation

**Ticket ID:** PC-084-10A  
**Epic:** Epic-10A (Baseline Validation & Comprehensive Backtesting)  
**Story:** User Story 1 - Baseline Reproduction and Validation  
**Assignee:** TBD  
**Status:** Not Started  
**Priority:** Critical  
**Estimated Effort:** 5 days  
**Sprint:** Week 27

---

## 📋 Description

Implement the baseline validation framework to reproduce PrevCargaDESSEM R system results and establish reference standards for system accuracy validation. This includes baseline data extraction, statistical validation, and comparison framework for all 26 time series across 5 forecasting models.

---

## 🎯 Acceptance Criteria

- [ ] `BaselineValidator` class implemented with configuration management
- [ ] `BaselineDataLoader` loads historical PrevCargaDESSEM predictions and metrics
- [ ] `StatisticalTestSuite` performs normality, consistency, and significance tests
- [ ] Baseline reproduction within ±5% MAPE tolerance for all model-area combinations
- [ ] Historical data alignment between systems validated (2023-2024 period)
- [ ] Model-by-model comparison framework operational
- [ ] Tolerance thresholds documented (MAPE: ±5%, MAE: ±100MW, RMSE: ±150MW)
- [ ] Reference metrics archived for all 26 time series
- [ ] Unit tests achieve >80% coverage for validation components
- [ ] Integration tests validate end-to-end baseline comparison workflow

---

## 🏗️ Technical Implementation

### **1. BaselineValidator Class**

**Location:** `src/validation/baseline_validator.py`

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from scipy import stats

@dataclass
class ValidationConfig:
    """Configuration for baseline validation."""
    baseline_path: str
    areas: List[str]
    models: List[str]
    mape_tolerance: float = 0.05  # ±5%
    mae_tolerance: float = 100.0  # MW
    rmse_tolerance: float = 150.0  # MW
    confidence_level: float = 0.95

@dataclass
class BaselineReproductionResult:
    """Results from baseline reproduction."""
    baseline_metrics: Dict[str, float]
    validation_result: 'ValidationResult'
    reference_dataset: Dict
    reproduction_timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'baseline_metrics': self.baseline_metrics,
            'validation_result': self.validation_result.to_dict(),
            'reference_dataset': self.reference_dataset,
            'timestamp': self.reproduction_timestamp.isoformat()
        }

@dataclass
class ComparisonResult:
    """Results from baseline comparison."""
    metrics: Dict[str, Dict[str, float]]
    summary: Dict[str, float]
    violations: List[str]
    
    def passes_validation(self) -> bool:
        """Check if all comparisons pass tolerance thresholds."""
        return len(self.violations) == 0

class BaselineValidator:
    """Validates system accuracy against PrevCargaDESSEM baseline."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.baseline_data = BaselineDataLoader(config.baseline_path)
        self.statistical_tests = StatisticalTestSuite()
        self.logger = logging.getLogger(__name__)
    
    def reproduce_baseline(
        self,
        start_date: datetime,
        end_date: datetime,
        models: List[str] = None
    ) -> BaselineReproductionResult:
        """Reproduce PrevCargaDESSEM baseline for validation period.
        
        Args:
            start_date: Start of validation period
            end_date: End of validation period
            models: Models to validate (default: all configured models)
            
        Returns:
            BaselineReproductionResult with metrics and validation status
        """
        self.logger.info(f"Reproducing baseline from {start_date} to {end_date}")
        
        # Calculate baseline metrics
        baseline_metrics = self.baseline_data.calculate_metrics(
            start_date, end_date, models or self.config.models
        )
        
        # Validate baseline reproduction
        validation_result = self.statistical_tests.validate_baseline(
            baseline_metrics, self.config.tolerance_thresholds
        )
        
        # Create reference dataset
        reference_dataset = baseline_metrics.to_reference_format()
        
        result = BaselineReproductionResult(
            baseline_metrics=baseline_metrics,
            validation_result=validation_result,
            reference_dataset=reference_dataset,
            reproduction_timestamp=datetime.now()
        )
        
        self.logger.info(f"Baseline reproduction complete. Valid: {validation_result.is_valid}")
        return result
    
    def compare_with_baseline(
        self,
        system_predictions: 'PredictionDataset',
        baseline_predictions: 'BaselineDataset'
    ) -> ComparisonResult:
        """Compare system predictions with baseline across all metrics.
        
        Args:
            system_predictions: Predictions from PrevCarga system
            baseline_predictions: Predictions from PrevCargaDESSEM
            
        Returns:
            ComparisonResult with detailed comparison metrics
        """
        self.logger.info("Comparing system predictions with baseline")
        
        comparison_metrics = {}
        violations = []
        
        for area in self.config.areas:
            for model in self.config.models:
                # Calculate system metrics
                system_mape = self._calculate_mape(
                    system_predictions.get_predictions(area, model),
                    system_predictions.get_actuals(area)
                )
                
                # Get baseline metrics
                baseline_mape = baseline_predictions.get_mape(area, model)
                
                # Calculate difference
                difference = system_mape - baseline_mape
                within_tolerance = abs(difference) <= self.config.mape_tolerance
                
                key = f"{area}_{model}"
                comparison_metrics[key] = {
                    'system_mape': system_mape,
                    'baseline_mape': baseline_mape,
                    'difference': difference,
                    'difference_percentage': (difference / baseline_mape) * 100,
                    'within_tolerance': within_tolerance
                }
                
                # Track violations
                if not within_tolerance:
                    violations.append(
                        f"{key}: difference {difference:.3f} exceeds tolerance "
                        f"{self.config.mape_tolerance}"
                    )
        
        # Calculate summary statistics
        summary = self._calculate_summary(comparison_metrics)
        
        result = ComparisonResult(
            metrics=comparison_metrics,
            summary=summary,
            violations=violations
        )
        
        self.logger.info(f"Comparison complete. Violations: {len(violations)}")
        return result
    
    def _calculate_mape(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    def _calculate_summary(
        self,
        comparison_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate summary statistics from comparison metrics."""
        differences = [m['difference'] for m in comparison_metrics.values()]
        within_tolerance = [m['within_tolerance'] for m in comparison_metrics.values()]
        
        return {
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'max_difference': np.max(differences),
            'min_difference': np.min(differences),
            'pass_rate': sum(within_tolerance) / len(within_tolerance)
        }
```

### **2. BaselineDataLoader Class**

**Location:** `src/validation/baseline_data_loader.py`

```python
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class BaselineDataLoader:
    """Loads and processes PrevCargaDESSEM baseline data."""
    
    def __init__(self, baseline_path: str):
        self.baseline_path = Path(baseline_path)
        self.cache = {}
        self.logger = logging.getLogger(__name__)
    
    def load_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load historical predictions from PrevCargaDESSEM.
        
        Args:
            start_date: Start of period
            end_date: End of period
            
        Returns:
            DataFrame with historical predictions
        """
        self.logger.info(f"Loading baseline data from {start_date} to {end_date}")
        
        # Load from R system output files
        baseline_files = self._discover_baseline_files(start_date, end_date)
        
        dataframes = []
        for file_path in baseline_files:
            df = self._load_baseline_file(file_path)
            dataframes.append(df)
        
        # Concatenate and filter
        baseline_data = pd.concat(dataframes, ignore_index=True)
        baseline_data = baseline_data[
            (baseline_data['date'] >= start_date) &
            (baseline_data['date'] <= end_date)
        ]
        
        self.logger.info(f"Loaded {len(baseline_data)} baseline records")
        return baseline_data
    
    def calculate_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        models: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate baseline metrics for specified period and models.
        
        Args:
            start_date: Start of period
            end_date: End of period
            models: List of model names
            
        Returns:
            Dictionary of metrics by area and model
        """
        baseline_data = self.load_historical_data(start_date, end_date)
        actuals = self._load_actuals(start_date, end_date)
        
        metrics = {}
        for model in models:
            for area in baseline_data['area'].unique():
                predictions = baseline_data[
                    (baseline_data['model'] == model) &
                    (baseline_data['area'] == area)
                ]['prediction'].values
                
                actual_values = actuals[
                    actuals['area'] == area
                ]['load'].values
                
                key = f"{area}_{model}"
                metrics[key] = {
                    'mape': self._calculate_mape(predictions, actual_values),
                    'mae': self._calculate_mae(predictions, actual_values),
                    'rmse': self._calculate_rmse(predictions, actual_values)
                }
        
        return metrics
    
    def _discover_baseline_files(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Path]:
        """Discover baseline files for date range."""
        files = []
        for file_path in self.baseline_path.glob("*.csv"):
            # Parse date from filename
            file_date = self._extract_date_from_filename(file_path.name)
            if start_date <= file_date <= end_date:
                files.append(file_path)
        return sorted(files)
    
    def _load_baseline_file(self, file_path: Path) -> pd.DataFrame:
        """Load single baseline file."""
        return pd.read_csv(file_path, parse_dates=['date'])
    
    def _load_actuals(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load actual load data for period."""
        # Load from actuals database/files
        actuals_path = self.baseline_path.parent / "actuals"
        return pd.read_csv(
            actuals_path / "actuals.csv",
            parse_dates=['date']
        )
    
    @staticmethod
    def _calculate_mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate MAPE."""
        return np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    @staticmethod
    def _calculate_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate MAE."""
        return np.mean(np.abs(actuals - predictions))
    
    @staticmethod
    def _calculate_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate RMSE."""
        return np.sqrt(np.mean((actuals - predictions) ** 2))
```

### **3. StatisticalTestSuite Class**

**Location:** `src/validation/statistical_tests.py`

```python
from scipy import stats
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class StatisticalTestResult:
    """Results from statistical test."""
    test_type: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    confidence_interval: tuple
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_type': self.test_type,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'effect_size': self.effect_size,
            'confidence_interval': self.confidence_interval
        }

class StatisticalTestSuite:
    """Suite of statistical tests for validation."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.logger = logging.getLogger(__name__)
    
    def validate_baseline(
        self,
        baseline_metrics: Dict[str, Dict[str, float]],
        tolerance_thresholds: Dict[str, float]
    ) -> 'ValidationResult':
        """Validate baseline reproduction against tolerance thresholds.
        
        Args:
            baseline_metrics: Calculated metrics from baseline
            tolerance_thresholds: Acceptable tolerance thresholds
            
        Returns:
            ValidationResult with overall validation status
        """
        validations = {}
        
        for key, metrics in baseline_metrics.items():
            mape_valid = metrics['mape'] <= tolerance_thresholds.get('mape', 0.15)
            mae_valid = metrics['mae'] <= tolerance_thresholds.get('mae', 200.0)
            rmse_valid = metrics['rmse'] <= tolerance_thresholds.get('rmse', 300.0)
            
            validations[key] = {
                'mape_valid': mape_valid,
                'mae_valid': mae_valid,
                'rmse_valid': rmse_valid,
                'all_valid': mape_valid and mae_valid and rmse_valid
            }
        
        overall_valid = all(v['all_valid'] for v in validations.values())
        
        return ValidationResult(
            is_valid=overall_valid,
            validations=validations,
            summary=self._create_validation_summary(validations)
        )
    
    def test_significance(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        test_type: str = 'paired_t_test'
    ) -> StatisticalTestResult:
        """Perform statistical significance testing.
        
        Args:
            sample1: First sample
            sample2: Second sample
            test_type: Type of test ('paired_t_test', 'wilcoxon', etc.)
            
        Returns:
            StatisticalTestResult with test outcomes
        """
        if test_type == 'paired_t_test':
            statistic, p_value = stats.ttest_rel(sample1, sample2)
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(sample1, sample2)
        elif test_type == 'mann_whitney':
            statistic, p_value = stats.mannwhitneyu(sample1, sample2)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        effect_size = self._calculate_effect_size(sample1, sample2)
        confidence_interval = self._calculate_confidence_interval(
            sample1 - sample2 if test_type == 'paired_t_test' else sample1
        )
        
        return StatisticalTestResult(
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            effect_size=effect_size,
            confidence_interval=confidence_interval
        )
    
    def test_normality(self, sample: np.ndarray) -> bool:
        """Test if sample follows normal distribution."""
        statistic, p_value = stats.shapiro(sample)
        return p_value > self.alpha
    
    def _calculate_effect_size(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray
    ) -> float:
        """Calculate Cohen's d effect size."""
        mean_diff = np.mean(sample1) - np.mean(sample2)
        pooled_std = np.sqrt(
            (np.var(sample1) + np.var(sample2)) / 2
        )
        return mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    def _calculate_confidence_interval(
        self,
        sample: np.ndarray
    ) -> tuple:
        """Calculate confidence interval for sample mean."""
        mean = np.mean(sample)
        se = stats.sem(sample)
        ci = stats.t.interval(
            self.confidence_level,
            len(sample) - 1,
            loc=mean,
            scale=se
        )
        return ci
    
    def _create_validation_summary(
        self,
        validations: Dict
    ) -> Dict[str, float]:
        """Create summary statistics from validations."""
        total = len(validations)
        valid = sum(1 for v in validations.values() if v['all_valid'])
        
        return {
            'total_combinations': total,
            'valid_combinations': valid,
            'pass_rate': valid / total if total > 0 else 0.0
        }
```

---

## 🧪 Testing Requirements

### **Unit Tests**

**Location:** `tests/unit/validation/test_baseline_validator.py`

```python
import pytest
from datetime import datetime
from src.validation.baseline_validator import (
    BaselineValidator, ValidationConfig
)

class TestBaselineValidator:
    @pytest.fixture
    def config(self):
        return ValidationConfig(
            baseline_path="/data/baseline",
            areas=["SE", "S", "NE", "N"],
            models=["lgbm", "rf", "regdin_svm", "holt_winters"],
            mape_tolerance=0.05
        )
    
    @pytest.fixture
    def validator(self, config):
        return BaselineValidator(config)
    
    def test_reproduce_baseline_success(self, validator):
        """Test baseline reproduction with valid data."""
        result = validator.reproduce_baseline(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        assert result.validation_result.is_valid
        assert len(result.baseline_metrics) > 0
    
    def test_compare_with_baseline_within_tolerance(self, validator):
        """Test comparison when predictions are within tolerance."""
        system_predictions = create_mock_predictions()
        baseline_predictions = create_mock_baseline()
        
        result = validator.compare_with_baseline(
            system_predictions,
            baseline_predictions
        )
        
        assert result.passes_validation()
        assert len(result.violations) == 0
    
    def test_compare_with_baseline_exceeds_tolerance(self, validator):
        """Test comparison when predictions exceed tolerance."""
        system_predictions = create_mock_predictions_outside_tolerance()
        baseline_predictions = create_mock_baseline()
        
        result = validator.compare_with_baseline(
            system_predictions,
            baseline_predictions
        )
        
        assert not result.passes_validation()
        assert len(result.violations) > 0
```

### **Integration Tests**

**Location:** `tests/integration/validation/test_baseline_integration.py`

```python
import pytest
from datetime import datetime
from src.validation.baseline_validator import BaselineValidator, ValidationConfig

class TestBaselineValidationIntegration:
    def test_end_to_end_baseline_validation(self):
        """Test complete baseline validation workflow."""
        config = ValidationConfig(
            baseline_path="/data/baseline",
            areas=["SE", "S", "NE", "N"],
            models=["lgbm", "rf"]
        )
        
        validator = BaselineValidator(config)
        
        # Reproduce baseline
        baseline_result = validator.reproduce_baseline(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        assert baseline_result.validation_result.is_valid
        
        # Compare with system
        system_predictions = load_system_predictions()
        baseline_predictions = load_baseline_predictions()
        
        comparison = validator.compare_with_baseline(
            system_predictions,
            baseline_predictions
        )
        
        assert comparison.summary['pass_rate'] >= 0.95
```

---

## 📊 Success Metrics

- [ ] Baseline reproduction accuracy: 100% of model-area combinations within ±5% MAPE
- [ ] Data alignment validation: 100% successful for 2023-2024 period
- [ ] Statistical test coverage: >80% code coverage
- [ ] Performance: Baseline validation completes in <30 minutes
- [ ] Reference dataset: All 26 time series documented with metrics

---

## 🔗 Dependencies

### **Upstream**
- Epic-07 (Evaluation): MetricsCalculator for metric computation
- Epic-08 (Orchestration): Access to historical predictions
- Data infrastructure: Access to PrevCargaDESSEM historical data

### **Downstream**
- PC-085-10A (Comprehensive Backtester): Uses baseline as reference
- PC-086-10A (Performance Analyzer): Uses baseline for comparisons

---

## 📚 Documentation

- [ ] Baseline validation methodology documented in `/docs/validation/baseline_validation.md`
- [ ] PrevCargaDESSEM data format specification in `/docs/data/baseline_format.md`
- [ ] Statistical test rationale documented in `/docs/validation/statistical_tests.md`
- [ ] Tolerance threshold justification in `/docs/validation/tolerance_analysis.md`
- [ ] API documentation generated with complete docstrings

---

## ✅ Definition of Done

- [ ] All code implemented and peer-reviewed
- [ ] Unit tests passing with >80% coverage
- [ ] Integration tests validate end-to-end workflow
- [ ] Baseline reproduction achieves statistical consistency
- [ ] Reference metrics established for all 130 model-area combinations (26 areas × 5 models)
- [ ] Documentation complete and reviewed
- [ ] Code merged to main branch
- [ ] Validation results archived and certified

---

**Notes:**
- Coordinate with data team for PrevCargaDESSEM data access
- Validate data format compatibility early in implementation
- Consider parallel processing for multiple model-area combinations
- Ensure proper error handling for missing baseline data
