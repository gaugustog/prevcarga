# PC-088-10B: System Robustness Validator Implementation

**Ticket ID:** PC-088-10B  
**Epic:** Epic-10B (Quality Assurance & Acceptance Validation)  
**Story:** User Story 1 - System Performance and Quality Validation  
**Assignee:** TBD  
**Status:** Not Started  
**Priority:** High  
**Estimated Effort:** 4 days  
**Sprint:** Week 29

---

## 📋 Description

Implement comprehensive system robustness validation framework to test system resilience under various failure scenarios including data quality issues, infrastructure failures, and model failures. The validator must inject errors systematically and verify graceful degradation and recovery mechanisms.

---

## 🎯 Acceptance Criteria

- [ ] `SystemRobustnessValidator` class with error injection capabilities
- [ ] Data quality resilience testing (missing values, outliers, temporal gaps, schema violations)
- [ ] Infrastructure failure resilience testing (S3 unavailability, network latency, resource pressure)
- [ ] Model failure handling validation (single/multiple model failures, training interruptions)
- [ ] Graceful degradation verification for all failure scenarios
- [ ] Recovery mechanism validation (automatic retries, fallback strategies)
- [ ] Error injection framework with configurable failure scenarios
- [ ] Robustness scoring and reporting
- [ ] Integration with monitoring and alerting systems
- [ ] Unit tests achieve >80% coverage
- [ ] Integration tests validate end-to-end robustness scenarios

---

## 🏗️ Technical Implementation

### **1. SystemRobustnessValidator Class**

**Location:** `src/validation/robustness_validator.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import logging
import numpy as np
import pandas as pd
from enum import Enum

class FailureScenario(Enum):
    """Types of failure scenarios to test."""
    MISSING_VALUES_30_PERCENT = "missing_values_30_percent"
    OUTLIER_INJECTION = "outlier_injection"
    TEMPORAL_GAPS = "temporal_gaps"
    SCHEMA_VIOLATIONS = "schema_violations"
    S3_TEMPORARY_UNAVAILABILITY = "s3_temporary_unavailability"
    NETWORK_LATENCY_SPIKES = "network_latency_spikes"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_THROTTLING = "cpu_throttling"
    SINGLE_MODEL_FAILURE = "single_model_failure"
    MULTIPLE_MODEL_FAILURE = "multiple_model_failure"
    TRAINING_INTERRUPTION = "training_interruption"
    PREDICTION_TIMEOUT = "prediction_timeout"

@dataclass
class RobustnessTestResult:
    """Result from a single robustness test."""
    scenario: str
    success: bool
    graceful_degradation: bool
    recovery_successful: bool
    error_details: Optional[str]
    execution_time: float
    resource_impact: Dict[str, float]

@dataclass
class RobustnessValidationResult:
    """Results from robustness validation."""
    tests: Dict[str, List[RobustnessTestResult]]
    overall_score: float
    critical_failures: List[str]
    
    def passes_validation(self) -> bool:
        """Check if robustness validation passes."""
        return len(self.critical_failures) == 0 and self.overall_score >= 0.8

class SystemRobustnessValidator:
    """Validates system robustness through error injection and stress testing."""
    
    def __init__(self, config: 'RobustnessConfig'):
        """Initialize robustness validator.
        
        Args:
            config: Configuration for robustness testing
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_injector = ErrorInjector()
    
    def validate_system_robustness(self) -> RobustnessValidationResult:
        """Validate system robustness through error injection and stress testing.
        
        Returns:
            RobustnessValidationResult with all test results
        """
        self.logger.info("Starting system robustness validation")
        robustness_tests = {}
        
        # Data quality robustness
        self.logger.info("Testing data quality resilience")
        data_quality_test = self.test_data_quality_resilience(
            scenarios=[
                FailureScenario.MISSING_VALUES_30_PERCENT,
                FailureScenario.OUTLIER_INJECTION,
                FailureScenario.TEMPORAL_GAPS,
                FailureScenario.SCHEMA_VIOLATIONS
            ]
        )
        robustness_tests['data_quality'] = data_quality_test
        
        # Infrastructure failure resilience
        self.logger.info("Testing infrastructure resilience")
        infrastructure_test = self.test_infrastructure_resilience(
            scenarios=[
                FailureScenario.S3_TEMPORARY_UNAVAILABILITY,
                FailureScenario.NETWORK_LATENCY_SPIKES,
                FailureScenario.MEMORY_PRESSURE,
                FailureScenario.CPU_THROTTLING
            ]
        )
        robustness_tests['infrastructure'] = infrastructure_test
        
        # Model failure handling
        self.logger.info("Testing model failure handling")
        model_failure_test = self.test_model_failure_handling(
            scenarios=[
                FailureScenario.SINGLE_MODEL_FAILURE,
                FailureScenario.MULTIPLE_MODEL_FAILURE,
                FailureScenario.TRAINING_INTERRUPTION,
                FailureScenario.PREDICTION_TIMEOUT
            ]
        )
        robustness_tests['model_failure'] = model_failure_test
        
        # Calculate overall robustness score
        overall_score = self._calculate_robustness_score(robustness_tests)
        critical_failures = self._identify_critical_failures(robustness_tests)
        
        return RobustnessValidationResult(
            tests=robustness_tests,
            overall_score=overall_score,
            critical_failures=critical_failures
        )
    
    def test_data_quality_resilience(
        self,
        scenarios: List[FailureScenario]
    ) -> List[RobustnessTestResult]:
        """Test system resilience to data quality issues.
        
        Args:
            scenarios: List of data quality failure scenarios to test
            
        Returns:
            List of test results for each scenario
        """
        results = []
        
        for scenario in scenarios:
            self.logger.info(f"Testing scenario: {scenario.value}")
            
            if scenario == FailureScenario.MISSING_VALUES_30_PERCENT:
                result = self._test_missing_values()
            elif scenario == FailureScenario.OUTLIER_INJECTION:
                result = self._test_outlier_injection()
            elif scenario == FailureScenario.TEMPORAL_GAPS:
                result = self._test_temporal_gaps()
            elif scenario == FailureScenario.SCHEMA_VIOLATIONS:
                result = self._test_schema_violations()
            else:
                continue
            
            results.append(result)
        
        return results
    
    def test_infrastructure_resilience(
        self,
        scenarios: List[FailureScenario]
    ) -> List[RobustnessTestResult]:
        """Test system resilience to infrastructure failures.
        
        Args:
            scenarios: List of infrastructure failure scenarios to test
            
        Returns:
            List of test results for each scenario
        """
        results = []
        
        for scenario in scenarios:
            self.logger.info(f"Testing scenario: {scenario.value}")
            
            if scenario == FailureScenario.S3_TEMPORARY_UNAVAILABILITY:
                result = self._test_s3_unavailability()
            elif scenario == FailureScenario.NETWORK_LATENCY_SPIKES:
                result = self._test_network_latency()
            elif scenario == FailureScenario.MEMORY_PRESSURE:
                result = self._test_memory_pressure()
            elif scenario == FailureScenario.CPU_THROTTLING:
                result = self._test_cpu_throttling()
            else:
                continue
            
            results.append(result)
        
        return results
    
    def test_model_failure_handling(
        self,
        scenarios: List[FailureScenario]
    ) -> List[RobustnessTestResult]:
        """Test system handling of model failures.
        
        Args:
            scenarios: List of model failure scenarios to test
            
        Returns:
            List of test results for each scenario
        """
        results = []
        
        for scenario in scenarios:
            self.logger.info(f"Testing scenario: {scenario.value}")
            
            if scenario == FailureScenario.SINGLE_MODEL_FAILURE:
                result = self._test_single_model_failure()
            elif scenario == FailureScenario.MULTIPLE_MODEL_FAILURE:
                result = self._test_multiple_model_failure()
            elif scenario == FailureScenario.TRAINING_INTERRUPTION:
                result = self._test_training_interruption()
            elif scenario == FailureScenario.PREDICTION_TIMEOUT:
                result = self._test_prediction_timeout()
            else:
                continue
            
            results.append(result)
        
        return results
    
    def _test_missing_values(self) -> RobustnessTestResult:
        """Test handling of 30% missing values in data."""
        import time
        from prevcarga.data.preprocessing import DataPreprocessor
        
        start_time = time.time()
        
        try:
            # Load test data and inject 30% missing values
            corrupted_data = self.error_injector.inject_missing_values(
                self._load_test_data(),
                missing_rate=0.3
            )
            
            # Attempt preprocessing
            preprocessor = DataPreprocessor()
            processed_data = preprocessor.preprocess(corrupted_data)
            
            # Verify data quality after processing
            missing_rate = processed_data.isna().sum().sum() / processed_data.size
            
            graceful_degradation = missing_rate < 0.05  # Should impute most values
            recovery_successful = True
            
            return RobustnessTestResult(
                scenario="missing_values_30_percent",
                success=True,
                graceful_degradation=graceful_degradation,
                recovery_successful=recovery_successful,
                error_details=None,
                execution_time=time.time() - start_time,
                resource_impact=self._measure_resource_impact()
            )
            
        except Exception as e:
            return RobustnessTestResult(
                scenario="missing_values_30_percent",
                success=False,
                graceful_degradation=False,
                recovery_successful=False,
                error_details=str(e),
                execution_time=time.time() - start_time,
                resource_impact=self._measure_resource_impact()
            )
    
    def _test_storage_unavailability(self) -> RobustnessTestResult:
        """Test handling of temporary storage backend unavailability."""
        import time
        from prevcarga.storage.factory import StorageFactory
        from prevcarga.data.loaders import DataLoader
        
        start_time = time.time()
        
        try:
            backend = StorageFactory.from_config()
            loader = DataLoader(storage_backend=backend)
            
            # Inject storage unavailability (S3 or local access failure)
            with self.error_injector.inject_storage_failure(duration=5):
                # Attempt to load data (should retry)
                data = loader.load_carga(
                    areas=['01'],
                    start_date='2024-01-01',
                    end_date='2024-01-31'
                )
            
            # Verify data was eventually loaded
            success = data is not None and not data.empty
            graceful_degradation = success  # Should retry and succeed
            recovery_successful = success
            
            return RobustnessTestResult(
                scenario="storage_backend_temporary_unavailability",
                success=success,
                graceful_degradation=graceful_degradation,
                recovery_successful=recovery_successful,
                error_details=None if success else "Failed to recover from storage failure",
                execution_time=time.time() - start_time,
                resource_impact=self._measure_resource_impact()
            )
            
        except Exception as e:
            return RobustnessTestResult(
                scenario="storage_backend_temporary_unavailability",
                success=False,
                graceful_degradation=False,
                recovery_successful=False,
                error_details=str(e),
                execution_time=time.time() - start_time,
                resource_impact=self._measure_resource_impact()
            )
    
    def _test_single_model_failure(self) -> RobustnessTestResult:
        """Test handling when a single model fails."""
        import time
        from prevcarga.workflows.prediction import PredictionWorkflow
        
        start_time = time.time()
        
        try:
            workflow = PredictionWorkflow()
            
            # Inject failure for one model
            with self.error_injector.inject_model_failure(['lgbm']):
                # Attempt ensemble prediction
                predictions = workflow.predict_ensemble(
                    areas=['01'],
                    prediction_date='2024-12-01',
                    models=['lgbm', 'rf', 'regdin_svm']
                )
            
            # Verify predictions were generated despite single model failure
            success = predictions is not None and not predictions.empty
            graceful_degradation = success  # Should use other models
            recovery_successful = success
            
            return RobustnessTestResult(
                scenario="single_model_failure",
                success=success,
                graceful_degradation=graceful_degradation,
                recovery_successful=recovery_successful,
                error_details=None if success else "Failed to handle single model failure",
                execution_time=time.time() - start_time,
                resource_impact=self._measure_resource_impact()
            )
            
        except Exception as e:
            return RobustnessTestResult(
                scenario="single_model_failure",
                success=False,
                graceful_degradation=False,
                recovery_successful=False,
                error_details=str(e),
                execution_time=time.time() - start_time,
                resource_impact=self._measure_resource_impact()
            )
    
    def _calculate_robustness_score(
        self,
        tests: Dict[str, List[RobustnessTestResult]]
    ) -> float:
        """Calculate overall robustness score."""
        all_results = []
        for category_results in tests.values():
            all_results.extend(category_results)
        
        if not all_results:
            return 0.0
        
        # Score based on success, graceful degradation, and recovery
        scores = []
        for result in all_results:
            score = 0.0
            if result.success:
                score += 0.4
            if result.graceful_degradation:
                score += 0.3
            if result.recovery_successful:
                score += 0.3
            scores.append(score)
        
        return np.mean(scores)
    
    def _identify_critical_failures(
        self,
        tests: Dict[str, List[RobustnessTestResult]]
    ) -> List[str]:
        """Identify critical failures that block production."""
        critical_failures = []
        
        for category, results in tests.items():
            for result in results:
                # Critical if it fails and doesn't degrade gracefully
                if not result.success and not result.graceful_degradation:
                    critical_failures.append(
                        f"{category}.{result.scenario}: {result.error_details}"
                    )
        
        return critical_failures
    
    def _load_test_data(self) -> pd.DataFrame:
        """Load test data for robustness testing."""
        from prevcarga.storage.factory import StorageFactory
        from prevcarga.data.loaders import DataLoader
        
        backend = StorageFactory.from_config()
        loader = DataLoader(storage_backend=backend)
        return loader.load_carga(
            areas=['01'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
    
    def _measure_resource_impact(self) -> Dict[str, float]:
        """Measure resource impact during test."""
        import psutil
        
        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024
        }

class ErrorInjector:
    """Inject various types of errors for robustness testing."""
    
    def inject_missing_values(
        self,
        data: pd.DataFrame,
        missing_rate: float
    ) -> pd.DataFrame:
        """Inject missing values into data."""
        corrupted = data.copy()
        mask = np.random.random(corrupted.shape) < missing_rate
        corrupted = corrupted.mask(mask)
        return corrupted
    
    def inject_s3_failure(self, duration: int):
        """Context manager to inject S3 failures."""
        return S3FailureInjector(duration)
    
    def inject_model_failure(self, models: List[str]):
        """Context manager to inject model failures."""
        return ModelFailureInjector(models)

class S3FailureInjector:
    """Context manager for S3 failure injection."""
    
    def __init__(self, duration: int):
        self.duration = duration
    
    def __enter__(self):
        # Mock S3 to fail temporarily
        import boto3
        from unittest.mock import patch
        self.patcher = patch('boto3.client')
        self.mock_client = self.patcher.start()
        self.mock_client.side_effect = Exception("S3 temporarily unavailable")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.stop()

class ModelFailureInjector:
    """Context manager for model failure injection."""
    
    def __init__(self, models: List[str]):
        self.models = models
    
    def __enter__(self):
        # Mock models to fail
        from unittest.mock import patch
        self.patchers = []
        for model in self.models:
            patcher = patch(f'prevcarga.models.{model}.predict')
            mock_predict = patcher.start()
            mock_predict.side_effect = Exception(f"{model} failed")
            self.patchers.append(patcher)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patcher in self.patchers:
            patcher.stop()

@dataclass
class RobustnessConfig:
    """Configuration for robustness testing."""
    test_data_path: str
    failure_duration: int = 5
    retry_attempts: int = 3
    timeout_seconds: int = 300
```

---

## 🧪 Testing Requirements

### **Unit Tests**

**Location:** `tests/validation/test_robustness_validator.py`

```python
import pytest
from unittest.mock import Mock, patch
from prevcarga.validation.robustness_validator import (
    SystemRobustnessValidator,
    FailureScenario,
    ErrorInjector,
    RobustnessConfig
)

class TestSystemRobustnessValidator:
    """Test suite for SystemRobustnessValidator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RobustnessConfig(
            test_data_path='/test/data',
            failure_duration=1,
            retry_attempts=2
        )
    
    @pytest.fixture
    def validator(self, config):
        """Create validator instance."""
        return SystemRobustnessValidator(config)
    
    def test_validate_system_robustness(self, validator):
        """Test complete robustness validation."""
        with patch.object(validator, 'test_data_quality_resilience') as mock_dq:
            with patch.object(validator, 'test_infrastructure_resilience') as mock_infra:
                with patch.object(validator, 'test_model_failure_handling') as mock_model:
                    mock_dq.return_value = []
                    mock_infra.return_value = []
                    mock_model.return_value = []
                    
                    result = validator.validate_system_robustness()
                    
                    assert 'data_quality' in result.tests
                    assert 'infrastructure' in result.tests
                    assert 'model_failure' in result.tests
    
    def test_calculate_robustness_score(self, validator):
        """Test robustness score calculation."""
        from prevcarga.validation.robustness_validator import RobustnessTestResult
        
        tests = {
            'category1': [
                RobustnessTestResult(
                    scenario='test1',
                    success=True,
                    graceful_degradation=True,
                    recovery_successful=True,
                    error_details=None,
                    execution_time=1.0,
                    resource_impact={}
                )
            ]
        }
        
        score = validator._calculate_robustness_score(tests)
        
        assert score == 1.0  # Perfect score
    
    def test_identify_critical_failures(self, validator):
        """Test identification of critical failures."""
        from prevcarga.validation.robustness_validator import RobustnessTestResult
        
        tests = {
            'category1': [
                RobustnessTestResult(
                    scenario='critical_test',
                    success=False,
                    graceful_degradation=False,
                    recovery_successful=False,
                    error_details='Critical error',
                    execution_time=1.0,
                    resource_impact={}
                )
            ]
        }
        
        failures = validator._identify_critical_failures(tests)
        
        assert len(failures) == 1
        assert 'critical_test' in failures[0]

class TestErrorInjector:
    """Test suite for ErrorInjector."""
    
    @pytest.fixture
    def injector(self):
        """Create error injector instance."""
        return ErrorInjector()
    
    def test_inject_missing_values(self, injector):
        """Test missing value injection."""
        import pandas as pd
        
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [6, 7, 8, 9, 10]
        })
        
        corrupted = injector.inject_missing_values(data, missing_rate=0.5)
        
        missing_count = corrupted.isna().sum().sum()
        assert missing_count > 0
```

---

## 📊 Success Metrics

- [ ] All data quality scenarios pass with graceful degradation
- [ ] Infrastructure failure scenarios recover successfully
- [ ] Model failure handling maintains service availability
- [ ] Overall robustness score >0.8
- [ ] Zero critical failures identified

---

## 🔗 Dependencies

**Requires:**
- PC-087-10B (Performance Validator - baseline metrics)
- Epic-08A (Workflows for testing)
- Epic-01 (Data pipeline)

**Blocks:**
- PC-091-10B (Acceptance Criteria Validator)

---

## 📚 Documentation

- [ ] Robustness testing framework documented
- [ ] Failure scenarios and expected behaviors documented
- [ ] Error injection patterns documented
- [ ] Recovery mechanisms guide created

---

## 🔄 Implementation Steps

1. **Day 1: Core Robustness Framework**
   - Implement `SystemRobustnessValidator` class
   - Implement `ErrorInjector` class
   - Create failure scenario definitions

2. **Day 2: Data Quality Tests**
   - Implement data quality resilience tests
   - Create data corruption scenarios
   - Test recovery mechanisms

3. **Day 3: Infrastructure Tests**
   - Implement infrastructure failure tests
   - Create S3 and network failure injection
   - Test retry and fallback logic

4. **Day 4: Testing and Documentation**
   - Complete unit test suite
   - Complete integration tests
   - Write documentation

---

**Estimated Completion:** Week 29, Day 4  
**Review Required:** QA Team, Infrastructure Team
