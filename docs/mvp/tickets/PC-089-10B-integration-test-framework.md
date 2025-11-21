# PC-089-10B: Integration Test Framework Implementation

**Ticket ID:** PC-089-10B  
**Epic:** Epic-10B (Quality Assurance & Acceptance Validation)  
**Story:** User Story 1 - System Performance and Quality Validation  
**Assignee:** TBD  
**Status:** Not Started  
**Priority:** High  
**Estimated Effort:** 3 days  
**Sprint:** Week 29

---

## 📋 Description

Implement comprehensive integration test framework to validate end-to-end workflows across all system components including CLI to workflow integration, data pipeline integration, model pipeline integration, and evaluation pipeline integration. Tests must cover complete user journeys and validate component interactions.

---

## 🎯 Acceptance Criteria

- [ ] `IntegrationTestFramework` class with test orchestration capabilities
- [ ] CLI to workflow integration tests for all commands
- [ ] Data pipeline integration tests (load, validate, preprocess, filter)
- [ ] Model pipeline integration tests (training, prediction, combination)
- [ ] Evaluation pipeline integration tests (evaluation, backtesting, validation)
- [ ] End-to-end user journey tests (complete workflows)
- [ ] Component interaction validation across all integration points
- [ ] Test data fixtures and factories for reproducible tests
- [ ] Parallel test execution for performance
- [ ] Integration test reporting with failure analysis
- [ ] Unit tests achieve >75% coverage
- [ ] All critical integration paths validated

---

## 🏗️ Technical Implementation

### **1. IntegrationTestFramework Class**

**Location:** `src/testing/integration_test_framework.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import time
from pathlib import Path

@dataclass
class IntegrationTestResult:
    """Result from a single integration test."""
    test_name: str
    category: str
    success: bool
    execution_time: float
    error_details: Optional[str]
    components_tested: List[str]

@dataclass
class IntegrationValidationResult:
    """Results from integration validation."""
    tests: Dict[str, List[IntegrationTestResult]]
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_success: bool
    
    def get_success_rate(self) -> float:
        """Calculate test success rate."""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

class IntegrationTestFramework:
    """Framework for orchestrating integration tests."""
    
    def __init__(self, config: 'IntegrationTestConfig'):
        """Initialize integration test framework.
        
        Args:
            config: Configuration for integration testing
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.test_data_manager = TestDataManager()
        self.results: List[IntegrationTestResult] = []
    
    def validate_integration_completeness(self) -> IntegrationValidationResult:
        """Validate end-to-end integration across all system components.
        
        Returns:
            IntegrationValidationResult with all test results
        """
        self.logger.info("Starting integration completeness validation")
        integration_tests = {}
        
        # CLI to workflow integration
        self.logger.info("Testing CLI to workflow integration")
        cli_integration = self.test_cli_workflow_integration()
        integration_tests['cli_workflow'] = cli_integration
        
        # Data pipeline integration
        self.logger.info("Testing data pipeline integration")
        data_integration = self.test_data_pipeline_integration()
        integration_tests['data_pipeline'] = data_integration
        
        # Model pipeline integration
        self.logger.info("Testing model pipeline integration")
        model_integration = self.test_model_pipeline_integration()
        integration_tests['model_pipeline'] = model_integration
        
        # Evaluation pipeline integration
        self.logger.info("Testing evaluation pipeline integration")
        eval_integration = self.test_evaluation_pipeline_integration()
        integration_tests['evaluation_pipeline'] = eval_integration
        
        # End-to-end user journeys
        self.logger.info("Testing end-to-end user journeys")
        journey_tests = self.test_end_to_end_journeys()
        integration_tests['user_journeys'] = journey_tests
        
        # Calculate overall results
        total_tests = sum(len(tests) for tests in integration_tests.values())
        passed_tests = sum(
            sum(1 for t in tests if t.success) 
            for tests in integration_tests.values()
        )
        failed_tests = total_tests - passed_tests
        
        return IntegrationValidationResult(
            tests=integration_tests,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_success=failed_tests == 0
        )
    
    def test_cli_workflow_integration(self) -> List[IntegrationTestResult]:
        """Test CLI to workflow integration for all commands.
        
        Returns:
            List of test results for CLI integration
        """
        tests = []
        
        # Test data commands integration
        tests.append(self._test_cli_data_load())
        tests.append(self._test_cli_data_validate())
        tests.append(self._test_cli_data_preprocess())
        
        # Test training commands integration
        tests.append(self._test_cli_train_model())
        tests.append(self._test_cli_train_batch())
        
        # Test prediction commands integration
        tests.append(self._test_cli_predict_single())
        tests.append(self._test_cli_predict_batch())
        
        # Test evaluation commands integration
        tests.append(self._test_cli_evaluate_model())
        tests.append(self._test_cli_backtest())
        
        return tests
    
    def test_data_pipeline_integration(self) -> List[IntegrationTestResult]:
        """Test data pipeline integration.
        
        Returns:
            List of test results for data pipeline
        """
        tests = []
        
        # Test load -> validate -> preprocess flow
        tests.append(self._test_data_load_validate_preprocess())
        
        # Test complete days filtering integration
        tests.append(self._test_complete_days_filtering())
        
        # Test auxiliary data integration
        tests.append(self._test_auxiliary_data_integration())
        
        # Test data catalog integration
        tests.append(self._test_data_catalog_integration())
        
        return tests
    
    def test_model_pipeline_integration(self) -> List[IntegrationTestResult]:
        """Test model pipeline integration.
        
        Returns:
            List of test results for model pipeline
        """
        tests = []
        
        # Test feature engineering -> training flow
        tests.append(self._test_feature_engineering_training())
        
        # Test training -> prediction flow
        tests.append(self._test_training_prediction_flow())
        
        # Test model combination integration
        tests.append(self._test_model_combination_integration())
        
        # Test hierarchical reconciliation integration
        tests.append(self._test_hierarchical_reconciliation())
        
        # Test model serialization integration
        tests.append(self._test_model_serialization())
        
        return tests
    
    def test_evaluation_pipeline_integration(self) -> List[IntegrationTestResult]:
        """Test evaluation pipeline integration.
        
        Returns:
            List of test results for evaluation pipeline
        """
        tests = []
        
        # Test prediction -> evaluation flow
        tests.append(self._test_prediction_evaluation_flow())
        
        # Test backtesting integration
        tests.append(self._test_backtesting_integration())
        
        # Test baseline comparison integration
        tests.append(self._test_baseline_comparison())
        
        return tests
    
    def test_end_to_end_journeys(self) -> List[IntegrationTestResult]:
        """Test complete end-to-end user journeys.
        
        Returns:
            List of test results for user journeys
        """
        tests = []
        
        # Journey 1: New area onboarding
        tests.append(self._test_new_area_onboarding_journey())
        
        # Journey 2: Daily prediction workflow
        tests.append(self._test_daily_prediction_journey())
        
        # Journey 3: Model retraining workflow
        tests.append(self._test_model_retraining_journey())
        
        # Journey 4: Performance evaluation workflow
        tests.append(self._test_performance_evaluation_journey())
        
        return tests
    
    def _test_cli_train_model(self) -> IntegrationTestResult:
        """Test CLI train model command integration."""
        start_time = time.time()
        
        try:
            from click.testing import CliRunner
            from prevcarga.cli.commands.train import train_model
            
            runner = CliRunner()
            result = runner.invoke(train_model, [
                '--model-type', 'lgbm',
                '--area', '01',
                '--start-date', '2024-01-01',
                '--end-date', '2024-12-31'
            ])
            
            success = result.exit_code == 0
            
            return IntegrationTestResult(
                test_name='cli_train_model',
                category='cli_workflow',
                success=success,
                execution_time=time.time() - start_time,
                error_details=None if success else result.output,
                components_tested=['CLI', 'TrainingWorkflow', 'Model']
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name='cli_train_model',
                category='cli_workflow',
                success=False,
                execution_time=time.time() - start_time,
                error_details=str(e),
                components_tested=['CLI', 'TrainingWorkflow', 'Model']
            )
    
    def _test_data_load_validate_preprocess(self) -> IntegrationTestResult:
        """Test complete data pipeline flow."""
        start_time = time.time()
        
        try:
            from prevcarga.storage.factory import StorageFactory
            from prevcarga.data.loaders import DataLoader
            from prevcarga.data.validation import DataValidator
            from prevcarga.data.preprocessing import DataPreprocessor
            
            # Load data
            backend = StorageFactory.from_config()
            loader = DataLoader(storage_backend=backend)
            raw_data = loader.load_carga(
                areas=['01'],
                start_date='2024-01-01',
                end_date='2024-01-31'
            )
            
            # Validate data
            validator = DataValidator()
            validation_result = validator.validate_schema(raw_data)
            
            if not validation_result.is_valid:
                raise ValueError("Data validation failed")
            
            # Preprocess data
            preprocessor = DataPreprocessor()
            processed_data = preprocessor.preprocess(raw_data)
            
            success = processed_data is not None and not processed_data.empty
            
            return IntegrationTestResult(
                test_name='data_load_validate_preprocess',
                category='data_pipeline',
                success=success,
                execution_time=time.time() - start_time,
                error_details=None,
                components_tested=['StorageBackend', 'DataLoader', 'DataValidator', 'DataPreprocessor']
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name='data_load_validate_preprocess',
                category='data_pipeline',
                success=False,
                execution_time=time.time() - start_time,
                error_details=str(e),
                components_tested=['StorageBackend', 'DataLoader', 'DataValidator', 'DataPreprocessor']
            )
    
    def _test_feature_engineering_training(self) -> IntegrationTestResult:
        """Test feature engineering to training flow."""
        start_time = time.time()
        
        try:
            from prevcarga.features.pipeline import FeaturePipeline
            from prevcarga.models.lgbm import LGBMModel
            
            # Setup test data
            test_data = self.test_data_manager.get_test_data('01')
            
            # Apply feature engineering
            feature_pipeline = FeaturePipeline()
            features = feature_pipeline.transform(test_data)
            
            # Train model
            model = LGBMModel()
            model.train(features)
            
            # Verify model trained successfully
            success = model.is_trained()
            
            return IntegrationTestResult(
                test_name='feature_engineering_training',
                category='model_pipeline',
                success=success,
                execution_time=time.time() - start_time,
                error_details=None,
                components_tested=['FeaturePipeline', 'LGBMModel']
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name='feature_engineering_training',
                category='model_pipeline',
                success=False,
                execution_time=time.time() - start_time,
                error_details=str(e),
                components_tested=['FeaturePipeline', 'LGBMModel']
            )
    
    def _test_new_area_onboarding_journey(self) -> IntegrationTestResult:
        """Test complete new area onboarding journey."""
        start_time = time.time()
        
        try:
            from click.testing import CliRunner
            from prevcarga.cli.main import cli
            
            runner = CliRunner()
            
            # Step 1: Load data for new area
            result = runner.invoke(cli, [
                'data', 'load',
                '--area', '99',
                '--start-date', '2024-01-01',
                '--end-date', '2024-12-31'
            ])
            if result.exit_code != 0:
                raise ValueError(f"Data load failed: {result.output}")
            
            # Step 2: Validate data
            result = runner.invoke(cli, [
                'data', 'validate',
                '--area', '99'
            ])
            if result.exit_code != 0:
                raise ValueError(f"Data validation failed: {result.output}")
            
            # Step 3: Train all models
            result = runner.invoke(cli, [
                'train', 'batch',
                '--config', 'test_config.yaml',
                '--areas', '99'
            ])
            if result.exit_code != 0:
                raise ValueError(f"Training failed: {result.output}")
            
            # Step 4: Generate predictions
            result = runner.invoke(cli, [
                'predict', 'batch',
                '--area', '99',
                '--prediction-date', '2024-12-15'
            ])
            if result.exit_code != 0:
                raise ValueError(f"Prediction failed: {result.output}")
            
            # Step 5: Evaluate performance
            result = runner.invoke(cli, [
                'evaluate', 'model',
                '--model-type', 'all',
                '--area', '99'
            ])
            success = result.exit_code == 0
            
            return IntegrationTestResult(
                test_name='new_area_onboarding_journey',
                category='user_journeys',
                success=success,
                execution_time=time.time() - start_time,
                error_details=None if success else result.output,
                components_tested=['CLI', 'Data', 'Training', 'Prediction', 'Evaluation']
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name='new_area_onboarding_journey',
                category='user_journeys',
                success=False,
                execution_time=time.time() - start_time,
                error_details=str(e),
                components_tested=['CLI', 'Data', 'Training', 'Prediction', 'Evaluation']
            )

class TestDataManager:
    """Manages test data fixtures and factories."""
    
    def __init__(self):
        self.test_data_cache = {}
    
    def get_test_data(self, area: str) -> Any:
        """Get test data for specified area."""
        if area not in self.test_data_cache:
            self.test_data_cache[area] = self._generate_test_data(area)
        return self.test_data_cache[area]
    
    def _generate_test_data(self, area: str) -> Any:
        """Generate test data for area."""
        import pandas as pd
        import numpy as np
        
        # Generate synthetic test data
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='H')
        data = pd.DataFrame({
            'datetime': dates,
            'area': area,
            'load': np.random.randint(1000, 5000, len(dates)),
            'temperature': np.random.uniform(10, 35, len(dates)),
            'holiday': np.random.choice([0, 1], len(dates), p=[0.95, 0.05])
        })
        return data

@dataclass
class IntegrationTestConfig:
    """Configuration for integration testing."""
    test_data_path: str
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_seconds: int = 600
```

---

## 🧪 Testing Requirements

### **Unit Tests**

**Location:** `tests/testing/test_integration_framework.py`

```python
import pytest
from unittest.mock import Mock, patch
from prevcarga.testing.integration_test_framework import (
    IntegrationTestFramework,
    IntegrationTestConfig,
    TestDataManager
)

class TestIntegrationTestFramework:
    """Test suite for IntegrationTestFramework."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return IntegrationTestConfig(
            test_data_path='/test/data',
            parallel_execution=False
        )
    
    @pytest.fixture
    def framework(self, config):
        """Create framework instance."""
        return IntegrationTestFramework(config)
    
    def test_validate_integration_completeness(self, framework):
        """Test complete integration validation."""
        with patch.object(framework, 'test_cli_workflow_integration') as mock_cli:
            with patch.object(framework, 'test_data_pipeline_integration') as mock_data:
                with patch.object(framework, 'test_model_pipeline_integration') as mock_model:
                    with patch.object(framework, 'test_evaluation_pipeline_integration') as mock_eval:
                        with patch.object(framework, 'test_end_to_end_journeys') as mock_journey:
                            mock_cli.return_value = []
                            mock_data.return_value = []
                            mock_model.return_value = []
                            mock_eval.return_value = []
                            mock_journey.return_value = []
                            
                            result = framework.validate_integration_completeness()
                            
                            assert result.total_tests == 0
                            assert result.overall_success is True
    
    def test_test_cli_workflow_integration(self, framework):
        """Test CLI workflow integration tests."""
        with patch('prevcarga.cli.commands.train.train_model'):
            tests = framework.test_cli_workflow_integration()
            
            assert len(tests) > 0
            assert all(hasattr(t, 'test_name') for t in tests)

class TestTestDataManager:
    """Test suite for TestDataManager."""
    
    @pytest.fixture
    def manager(self):
        """Create test data manager instance."""
        return TestDataManager()
    
    def test_get_test_data(self, manager):
        """Test test data retrieval."""
        data = manager.get_test_data('01')
        
        assert data is not None
        assert not data.empty
    
    def test_test_data_caching(self, manager):
        """Test that test data is cached."""
        data1 = manager.get_test_data('01')
        data2 = manager.get_test_data('01')
        
        assert data1 is data2  # Same object reference
```

---

## 📊 Success Metrics

- [ ] All CLI command integrations tested successfully
- [ ] All data pipeline flows validated
- [ ] All model pipeline flows validated
- [ ] All user journeys complete successfully
- [ ] Integration test coverage >75%

---

## 🔗 Dependencies

**Requires:**
- Epic-09A (CLI Commands)
- Epic-08A (Workflows)
- Epic-01-07 (All core components)

**Blocks:**
- PC-091-10B (Acceptance Validator - needs integration tests)

---

## 📚 Documentation

- [ ] Integration testing framework documented
- [ ] Test scenario definitions documented
- [ ] User journey test cases documented
- [ ] Test data fixtures documented

---

## 🔄 Implementation Steps

1. **Day 1: Core Integration Framework**
   - Implement `IntegrationTestFramework` class
   - Implement `TestDataManager` class
   - Create test result tracking

2. **Day 2: Pipeline Integration Tests**
   - Implement CLI integration tests
   - Implement data pipeline tests
   - Implement model pipeline tests

3. **Day 3: Journey Tests and Documentation**
   - Implement end-to-end journey tests
   - Complete unit tests
   - Write documentation

---

**Estimated Completion:** Week 29, Day 3  
**Review Required:** QA Team, Development Team
