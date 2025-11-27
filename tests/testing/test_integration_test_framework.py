"""Tests for integration test framework.

This module tests the IntegrationTestFramework class and related components
for orchestrating integration tests across all system components.
"""

from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.testing.integration_test_framework import (
    FixtureDataManager,
    IntegrationCategory,
    IntegrationTestConfig,
    IntegrationTestFramework,
    IntegrationTestResult,
    IntegrationValidationResult,
    SyntheticDataFactory,
)

# Use shorter aliases in tests
TestCategory = IntegrationCategory
TestDataFactory = SyntheticDataFactory
TestDataManager = FixtureDataManager


class TestTestCategory:
    """Tests for TestCategory enum."""

    def test_cli_workflow_category(self) -> None:
        """Test CLI workflow category value."""
        assert TestCategory.CLI_WORKFLOW.value == "cli_workflow"

    def test_data_pipeline_category(self) -> None:
        """Test data pipeline category value."""
        assert TestCategory.DATA_PIPELINE.value == "data_pipeline"

    def test_model_pipeline_category(self) -> None:
        """Test model pipeline category value."""
        assert TestCategory.MODEL_PIPELINE.value == "model_pipeline"

    def test_evaluation_pipeline_category(self) -> None:
        """Test evaluation pipeline category value."""
        assert TestCategory.EVALUATION_PIPELINE.value == "evaluation_pipeline"

    def test_user_journeys_category(self) -> None:
        """Test user journeys category value."""
        assert TestCategory.USER_JOURNEYS.value == "user_journeys"

    def test_all_categories_count(self) -> None:
        """Test total number of categories."""
        assert len(list(TestCategory)) == 5


class TestIntegrationTestConfig:
    """Tests for IntegrationTestConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = IntegrationTestConfig()
        assert config.test_data_path == "/tmp/test_data"
        assert config.parallel_execution is True
        assert config.max_workers == 4
        assert config.timeout_seconds == 600
        assert config.fail_fast is False
        assert config.categories is None

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = IntegrationTestConfig(
            test_data_path="/custom/path",
            parallel_execution=False,
            max_workers=8,
            timeout_seconds=300,
            fail_fast=True,
            categories=[TestCategory.CLI_WORKFLOW],
        )
        assert config.test_data_path == "/custom/path"
        assert config.parallel_execution is False
        assert config.max_workers == 8
        assert config.timeout_seconds == 300
        assert config.fail_fast is True
        assert config.categories == [TestCategory.CLI_WORKFLOW]

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = IntegrationTestConfig(
            categories=[TestCategory.CLI_WORKFLOW, TestCategory.DATA_PIPELINE]
        )
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["test_data_path"] == "/tmp/test_data"
        assert result["parallel_execution"] is True
        assert result["categories"] == ["cli_workflow", "data_pipeline"]

    def test_to_dict_no_categories(self) -> None:
        """Test to_dict with no categories."""
        config = IntegrationTestConfig()
        result = config.to_dict()

        assert result["categories"] is None


class TestIntegrationTestResult:
    """Tests for IntegrationTestResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> IntegrationTestResult:
        """Create a sample test result."""
        return IntegrationTestResult(
            test_name="test_example",
            category="cli_workflow",
            success=True,
            execution_time=1.5,
            error_details=None,
            components_tested=["CLI", "DataLoader"],
        )

    def test_successful_result(self, sample_result: IntegrationTestResult) -> None:
        """Test successful result attributes."""
        assert sample_result.test_name == "test_example"
        assert sample_result.category == "cli_workflow"
        assert sample_result.success is True
        assert sample_result.execution_time == 1.5
        assert sample_result.error_details is None
        assert sample_result.components_tested == ["CLI", "DataLoader"]
        assert sample_result.timestamp is not None

    def test_failed_result(self) -> None:
        """Test failed result attributes."""
        result = IntegrationTestResult(
            test_name="test_failed",
            category="data_pipeline",
            success=False,
            execution_time=0.5,
            error_details="Connection failed",
            components_tested=["DataLoader"],
        )
        assert result.success is False
        assert result.error_details == "Connection failed"

    def test_to_dict(self, sample_result: IntegrationTestResult) -> None:
        """Test conversion to dictionary."""
        result = sample_result.to_dict()

        assert result["test_name"] == "test_example"
        assert result["category"] == "cli_workflow"
        assert result["success"] is True
        assert result["execution_time"] == 1.5
        assert "timestamp" in result
        assert result["components_tested"] == ["CLI", "DataLoader"]


class TestIntegrationValidationResult:
    """Tests for IntegrationValidationResult dataclass."""

    @pytest.fixture
    def sample_validation_result(self) -> IntegrationValidationResult:
        """Create a sample validation result."""
        test_results = [
            IntegrationTestResult(
                test_name="test1",
                category="cli_workflow",
                success=True,
                execution_time=1.0,
                error_details=None,
                components_tested=["CLI"],
            ),
            IntegrationTestResult(
                test_name="test2",
                category="cli_workflow",
                success=True,
                execution_time=2.0,
                error_details=None,
                components_tested=["DataLoader"],
            ),
            IntegrationTestResult(
                test_name="test3",
                category="cli_workflow",
                success=False,
                execution_time=0.5,
                error_details="Test failed",
                components_tested=["Validator"],
            ),
        ]
        return IntegrationValidationResult(
            tests={"cli_workflow": test_results},
            total_tests=3,
            passed_tests=2,
            failed_tests=1,
            overall_success=False,
            execution_time=3.5,
        )

    def test_get_success_rate(self, sample_validation_result: IntegrationValidationResult) -> None:
        """Test success rate calculation."""
        rate = sample_validation_result.get_success_rate()
        assert abs(rate - 2/3) < 0.001

    def test_get_success_rate_no_tests(self) -> None:
        """Test success rate with no tests."""
        result = IntegrationValidationResult(
            tests={},
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            overall_success=True,
        )
        assert result.get_success_rate() == 0.0

    def test_get_failed_tests(self, sample_validation_result: IntegrationValidationResult) -> None:
        """Test getting failed tests."""
        failed = sample_validation_result.get_failed_tests()
        assert len(failed) == 1
        assert failed[0].test_name == "test3"

    def test_get_tests_by_category(self, sample_validation_result: IntegrationValidationResult) -> None:
        """Test getting tests by category."""
        tests = sample_validation_result.get_tests_by_category("cli_workflow")
        assert len(tests) == 3

    def test_get_tests_by_category_not_found(self, sample_validation_result: IntegrationValidationResult) -> None:
        """Test getting tests by non-existent category."""
        tests = sample_validation_result.get_tests_by_category("nonexistent")
        assert len(tests) == 0

    def test_to_dict(self, sample_validation_result: IntegrationValidationResult) -> None:
        """Test conversion to dictionary."""
        result = sample_validation_result.to_dict()

        assert "tests" in result
        assert result["total_tests"] == 3
        assert result["passed_tests"] == 2
        assert result["failed_tests"] == 1
        assert result["overall_success"] is False
        assert "success_rate" in result
        assert "timestamp" in result


class TestTestDataFactory:
    """Tests for TestDataFactory class."""

    @pytest.fixture
    def factory(self) -> TestDataFactory:
        """Create a test data factory."""
        return TestDataFactory(seed=42)

    def test_create_load_data(self, factory: TestDataFactory) -> None:
        """Test load data creation."""
        data = factory.create_load_data(
            area="SECO",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        assert isinstance(data, pd.DataFrame)
        assert "datetime" in data.columns
        assert "area" in data.columns
        assert "load_mw" in data.columns
        assert "temperature" in data.columns
        assert "humidity" in data.columns
        assert "holiday" in data.columns
        assert (data["area"] == "SECO").all()

    def test_create_load_data_reproducible(self, factory: TestDataFactory) -> None:
        """Test that load data creation is reproducible."""
        data1 = factory.create_load_data(area="SECO", start_date="2024-01-01", end_date="2024-01-07")
        data2 = factory.create_load_data(area="SECO", start_date="2024-01-01", end_date="2024-01-07")

        pd.testing.assert_frame_equal(data1, data2)

    def test_create_prediction_data(self, factory: TestDataFactory) -> None:
        """Test prediction data creation."""
        data = factory.create_prediction_data(
            area="SECO",
            prediction_date="2024-12-15",
            horizons=8,
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 8
        assert "prediction_date" in data.columns
        assert "horizon" in data.columns
        assert "area" in data.columns
        assert "predicted_load" in data.columns
        assert "actual_load" in data.columns
        assert "model" in data.columns

    def test_create_model_config_lgbm(self, factory: TestDataFactory) -> None:
        """Test LGBM model config creation."""
        config = factory.create_model_config("lgbm")

        assert "n_estimators" in config
        assert "learning_rate" in config
        assert "max_depth" in config

    def test_create_model_config_rf(self, factory: TestDataFactory) -> None:
        """Test Random Forest model config creation."""
        config = factory.create_model_config("rf")

        assert "n_estimators" in config
        assert "max_depth" in config

    def test_create_model_config_unknown(self, factory: TestDataFactory) -> None:
        """Test unknown model config defaults to LGBM."""
        config = factory.create_model_config("unknown")

        assert "n_estimators" in config
        assert "learning_rate" in config


class TestTestDataManager:
    """Tests for TestDataManager class."""

    @pytest.fixture
    def manager(self) -> TestDataManager:
        """Create a test data manager."""
        return TestDataManager()

    def test_get_test_data_load(self, manager: TestDataManager) -> None:
        """Test getting load test data."""
        data = manager.get_test_data("SECO", "load")

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "load_mw" in data.columns

    def test_get_test_data_prediction(self, manager: TestDataManager) -> None:
        """Test getting prediction test data."""
        data = manager.get_test_data("SECO", "prediction")

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "predicted_load" in data.columns

    def test_data_caching(self, manager: TestDataManager) -> None:
        """Test that data is cached."""
        data1 = manager.get_test_data("SECO", "load")
        data2 = manager.get_test_data("SECO", "load")

        assert data1 is data2  # Same object reference

    def test_different_areas_not_cached_together(self, manager: TestDataManager) -> None:
        """Test that different areas have separate cache entries."""
        data_seco = manager.get_test_data("SECO", "load")
        data_s = manager.get_test_data("S", "load")

        assert data_seco is not data_s
        assert (data_seco["area"] == "SECO").all()
        assert (data_s["area"] == "S").all()

    def test_clear_cache(self, manager: TestDataManager) -> None:
        """Test cache clearing."""
        _ = manager.get_test_data("SECO", "load")
        assert manager.get_cache_size() > 0

        manager.clear_cache()
        assert manager.get_cache_size() == 0

    def test_get_cache_size(self, manager: TestDataManager) -> None:
        """Test getting cache size."""
        assert manager.get_cache_size() == 0

        _ = manager.get_test_data("SECO", "load")
        assert manager.get_cache_size() == 1

        _ = manager.get_test_data("S", "load")
        assert manager.get_cache_size() == 2


class TestIntegrationTestFramework:
    """Tests for IntegrationTestFramework class."""

    @pytest.fixture
    def framework(self) -> IntegrationTestFramework:
        """Create a framework with default config."""
        return IntegrationTestFramework()

    @pytest.fixture
    def custom_framework(self) -> IntegrationTestFramework:
        """Create a framework with custom config."""
        config = IntegrationTestConfig(
            parallel_execution=False,
            fail_fast=True,
        )
        return IntegrationTestFramework(config)

    def test_initialization_default(self, framework: IntegrationTestFramework) -> None:
        """Test initialization with default config."""
        assert framework.config is not None
        assert framework.test_data_manager is not None
        assert framework.results == []

    def test_initialization_custom(self, custom_framework: IntegrationTestFramework) -> None:
        """Test initialization with custom config."""
        assert custom_framework.config.parallel_execution is False
        assert custom_framework.config.fail_fast is True

    def test_validate_integration_completeness(self, framework: IntegrationTestFramework) -> None:
        """Test complete integration validation."""
        result = framework.validate_integration_completeness()

        assert isinstance(result, IntegrationValidationResult)
        assert result.total_tests > 0
        assert result.config == framework.config

    def test_validate_integration_specific_categories(self, framework: IntegrationTestFramework) -> None:
        """Test validation with specific categories."""
        result = framework.validate_integration_completeness(
            categories=[TestCategory.CLI_WORKFLOW]
        )

        assert TestCategory.CLI_WORKFLOW.value in result.tests
        assert len(result.tests) == 1

    def test_test_cli_workflow_integration(self, framework: IntegrationTestFramework) -> None:
        """Test CLI workflow integration tests."""
        tests = framework.test_cli_workflow_integration()

        assert len(tests) == 9  # 9 CLI tests
        assert all(isinstance(t, IntegrationTestResult) for t in tests)
        assert all(t.category == TestCategory.CLI_WORKFLOW.value for t in tests)

    def test_test_data_pipeline_integration(self, framework: IntegrationTestFramework) -> None:
        """Test data pipeline integration tests."""
        tests = framework.test_data_pipeline_integration()

        assert len(tests) == 4  # 4 data pipeline tests
        assert all(isinstance(t, IntegrationTestResult) for t in tests)
        assert all(t.category == TestCategory.DATA_PIPELINE.value for t in tests)

    def test_test_model_pipeline_integration(self, framework: IntegrationTestFramework) -> None:
        """Test model pipeline integration tests."""
        tests = framework.test_model_pipeline_integration()

        assert len(tests) == 5  # 5 model pipeline tests
        assert all(isinstance(t, IntegrationTestResult) for t in tests)
        assert all(t.category == TestCategory.MODEL_PIPELINE.value for t in tests)

    def test_test_evaluation_pipeline_integration(self, framework: IntegrationTestFramework) -> None:
        """Test evaluation pipeline integration tests."""
        tests = framework.test_evaluation_pipeline_integration()

        assert len(tests) == 3  # 3 evaluation pipeline tests
        assert all(isinstance(t, IntegrationTestResult) for t in tests)
        assert all(t.category == TestCategory.EVALUATION_PIPELINE.value for t in tests)

    def test_test_end_to_end_journeys(self, framework: IntegrationTestFramework) -> None:
        """Test end-to-end journey tests."""
        tests = framework.test_end_to_end_journeys()

        assert len(tests) == 4  # 4 journey tests
        assert all(isinstance(t, IntegrationTestResult) for t in tests)
        assert all(t.category == TestCategory.USER_JOURNEYS.value for t in tests)

    def test_run_test_success(self, framework: IntegrationTestFramework) -> None:
        """Test running a successful test."""
        def success_test() -> IntegrationTestResult:
            return IntegrationTestResult(
                test_name="success_test",
                category="test",
                success=True,
                execution_time=0.1,
                error_details=None,
                components_tested=["Test"],
            )

        result = framework.run_test(success_test)
        assert result.success is True

    def test_run_test_exception(self, framework: IntegrationTestFramework) -> None:
        """Test running a test that raises exception."""
        def exception_test() -> IntegrationTestResult:
            raise ValueError("Test error")

        result = framework.run_test(exception_test)
        assert result.success is False
        assert "Test error" in result.error_details

    def test_run_tests_parallel(self, framework: IntegrationTestFramework) -> None:
        """Test running tests in parallel."""
        def make_test(name: str) -> IntegrationTestResult:
            time.sleep(0.01)
            return IntegrationTestResult(
                test_name=name,
                category="test",
                success=True,
                execution_time=0.01,
                error_details=None,
                components_tested=["Test"],
            )

        test_funcs = [lambda n=i: make_test(f"test_{n}") for i in range(4)]
        results = framework.run_tests_parallel(test_funcs)

        assert len(results) == 4
        assert all(r.success for r in results)

    def test_generate_report(self, framework: IntegrationTestFramework) -> None:
        """Test report generation."""
        validation_result = IntegrationValidationResult(
            tests={
                "cli_workflow": [
                    IntegrationTestResult(
                        test_name="test1",
                        category="cli_workflow",
                        success=True,
                        execution_time=1.0,
                        error_details=None,
                        components_tested=["CLI"],
                    ),
                    IntegrationTestResult(
                        test_name="test2",
                        category="cli_workflow",
                        success=False,
                        execution_time=0.5,
                        error_details="Error message",
                        components_tested=["DataLoader"],
                    ),
                ]
            },
            total_tests=2,
            passed_tests=1,
            failed_tests=1,
            overall_success=False,
            execution_time=1.5,
        )

        report = framework.generate_report(validation_result)

        assert "INTEGRATION TEST REPORT" in report
        assert "Total Tests: 2" in report
        assert "Passed: 1" in report
        assert "Failed: 1" in report
        assert "test1" in report
        assert "test2" in report
        assert "FAILED TESTS SUMMARY" in report


class TestCLIIntegrationTests:
    """Tests for CLI integration test methods."""

    @pytest.fixture
    def framework(self) -> IntegrationTestFramework:
        """Create a framework instance."""
        return IntegrationTestFramework()

    def test_cli_data_load(self, framework: IntegrationTestFramework) -> None:
        """Test CLI data load integration."""
        result = framework._test_cli_data_load()

        assert result.test_name == "cli_data_load"
        assert result.category == TestCategory.CLI_WORKFLOW.value
        assert "DataLoader" in result.components_tested

    def test_cli_data_validate(self, framework: IntegrationTestFramework) -> None:
        """Test CLI data validate integration."""
        result = framework._test_cli_data_validate()

        assert result.test_name == "cli_data_validate"
        assert "DataValidator" in result.components_tested

    def test_cli_data_preprocess(self, framework: IntegrationTestFramework) -> None:
        """Test CLI data preprocess integration."""
        result = framework._test_cli_data_preprocess()

        assert result.test_name == "cli_data_preprocess"
        assert "DataPreprocessor" in result.components_tested

    def test_cli_train_model(self, framework: IntegrationTestFramework) -> None:
        """Test CLI train model integration."""
        result = framework._test_cli_train_model()

        assert result.test_name == "cli_train_model"
        assert "TrainingWorkflow" in result.components_tested

    def test_cli_predict_single(self, framework: IntegrationTestFramework) -> None:
        """Test CLI predict single integration."""
        result = framework._test_cli_predict_single()

        assert result.test_name == "cli_predict_single"
        assert "PredictionWorkflow" in result.components_tested

    def test_cli_evaluate_model(self, framework: IntegrationTestFramework) -> None:
        """Test CLI evaluate model integration."""
        result = framework._test_cli_evaluate_model()

        assert result.test_name == "cli_evaluate_model"
        assert "MetricsCalculator" in result.components_tested

    def test_cli_backtest(self, framework: IntegrationTestFramework) -> None:
        """Test CLI backtest integration."""
        result = framework._test_cli_backtest()

        assert result.test_name == "cli_backtest"
        assert "BacktestingWorkflow" in result.components_tested


class TestDataPipelineIntegrationTests:
    """Tests for data pipeline integration test methods."""

    @pytest.fixture
    def framework(self) -> IntegrationTestFramework:
        """Create a framework instance."""
        return IntegrationTestFramework()

    def test_data_load_validate_preprocess(self, framework: IntegrationTestFramework) -> None:
        """Test data load, validate, preprocess flow."""
        result = framework._test_data_load_validate_preprocess()

        assert result.test_name == "data_load_validate_preprocess"
        assert result.category == TestCategory.DATA_PIPELINE.value
        assert "DataLoader" in result.components_tested
        assert "DataValidator" in result.components_tested
        assert "DataPreprocessor" in result.components_tested

    def test_complete_days_filtering(self, framework: IntegrationTestFramework) -> None:
        """Test complete days filtering integration."""
        result = framework._test_complete_days_filtering()

        assert result.test_name == "complete_days_filtering"
        assert "CompleteDaysFilter" in result.components_tested

    def test_auxiliary_data_integration(self, framework: IntegrationTestFramework) -> None:
        """Test auxiliary data integration."""
        result = framework._test_auxiliary_data_integration()

        assert result.test_name == "auxiliary_data_integration"
        assert "AuxiliaryDataLoader" in result.components_tested

    def test_data_catalog_integration(self, framework: IntegrationTestFramework) -> None:
        """Test data catalog integration."""
        result = framework._test_data_catalog_integration()

        assert result.test_name == "data_catalog_integration"
        assert "DataCatalog" in result.components_tested


class TestModelPipelineIntegrationTests:
    """Tests for model pipeline integration test methods."""

    @pytest.fixture
    def framework(self) -> IntegrationTestFramework:
        """Create a framework instance."""
        return IntegrationTestFramework()

    def test_feature_engineering_training(self, framework: IntegrationTestFramework) -> None:
        """Test feature engineering to training flow."""
        result = framework._test_feature_engineering_training()

        assert result.test_name == "feature_engineering_training"
        assert result.category == TestCategory.MODEL_PIPELINE.value
        assert "FeaturePipeline" in result.components_tested
        assert "LGBMModel" in result.components_tested

    def test_training_prediction_flow(self, framework: IntegrationTestFramework) -> None:
        """Test training to prediction flow."""
        result = framework._test_training_prediction_flow()

        assert result.test_name == "training_prediction_flow"
        assert "TrainingWorkflow" in result.components_tested
        assert "PredictionWorkflow" in result.components_tested

    def test_model_combination_integration(self, framework: IntegrationTestFramework) -> None:
        """Test model combination integration."""
        result = framework._test_model_combination_integration()

        assert result.test_name == "model_combination_integration"
        assert "BaseCombiner" in result.components_tested

    def test_hierarchical_reconciliation(self, framework: IntegrationTestFramework) -> None:
        """Test hierarchical reconciliation integration."""
        result = framework._test_hierarchical_reconciliation()

        assert result.test_name == "hierarchical_reconciliation"
        assert "HierarchyDefinition" in result.components_tested
        assert "MinTReconciler" in result.components_tested

    def test_model_serialization(self, framework: IntegrationTestFramework) -> None:
        """Test model serialization integration."""
        result = framework._test_model_serialization()

        assert result.test_name == "model_serialization"
        assert "ModelSerializer" in result.components_tested


class TestEvaluationPipelineIntegrationTests:
    """Tests for evaluation pipeline integration test methods."""

    @pytest.fixture
    def framework(self) -> IntegrationTestFramework:
        """Create a framework instance."""
        return IntegrationTestFramework()

    def test_prediction_evaluation_flow(self, framework: IntegrationTestFramework) -> None:
        """Test prediction to evaluation flow."""
        result = framework._test_prediction_evaluation_flow()

        assert result.test_name == "prediction_evaluation_flow"
        assert result.category == TestCategory.EVALUATION_PIPELINE.value
        assert "PredictionWorkflow" in result.components_tested
        assert "MetricsCalculator" in result.components_tested

    def test_backtesting_integration(self, framework: IntegrationTestFramework) -> None:
        """Test backtesting integration."""
        result = framework._test_backtesting_integration()

        assert result.test_name == "backtesting_integration"
        assert "BacktestingWorkflow" in result.components_tested
        assert "ComprehensiveBacktester" in result.components_tested

    def test_baseline_comparison(self, framework: IntegrationTestFramework) -> None:
        """Test baseline comparison integration."""
        result = framework._test_baseline_comparison()

        assert result.test_name == "baseline_comparison"
        assert "BaselineValidator" in result.components_tested


class TestUserJourneyIntegrationTests:
    """Tests for user journey integration test methods."""

    @pytest.fixture
    def framework(self) -> IntegrationTestFramework:
        """Create a framework instance."""
        return IntegrationTestFramework()

    def test_new_area_onboarding_journey(self, framework: IntegrationTestFramework) -> None:
        """Test new area onboarding journey."""
        result = framework._test_new_area_onboarding_journey()

        assert result.test_name == "new_area_onboarding_journey"
        assert result.category == TestCategory.USER_JOURNEYS.value
        assert "DataLoader" in result.components_tested
        assert "TrainingWorkflow" in result.components_tested
        assert "PredictionWorkflow" in result.components_tested

    def test_daily_prediction_journey(self, framework: IntegrationTestFramework) -> None:
        """Test daily prediction journey."""
        result = framework._test_daily_prediction_journey()

        assert result.test_name == "daily_prediction_journey"
        assert "PredictionWorkflow" in result.components_tested
        assert "BaseCombiner" in result.components_tested

    def test_model_retraining_journey(self, framework: IntegrationTestFramework) -> None:
        """Test model retraining journey."""
        result = framework._test_model_retraining_journey()

        assert result.test_name == "model_retraining_journey"
        assert "TrainingWorkflow" in result.components_tested
        assert "DriftDetector" in result.components_tested

    def test_performance_evaluation_journey(self, framework: IntegrationTestFramework) -> None:
        """Test performance evaluation journey."""
        result = framework._test_performance_evaluation_journey()

        assert result.test_name == "performance_evaluation_journey"
        assert "MetricsCalculator" in result.components_tested
        assert "ReportGenerator" in result.components_tested
        assert "ModelPerformanceAnalyzer" in result.components_tested


class TestIntegrationFrameworkIntegration:
    """Integration tests for the IntegrationTestFramework itself."""

    def test_full_validation_workflow(self) -> None:
        """Test complete validation workflow."""
        config = IntegrationTestConfig(
            parallel_execution=False,
            categories=[TestCategory.CLI_WORKFLOW],
        )
        framework = IntegrationTestFramework(config)

        result = framework.validate_integration_completeness()

        assert isinstance(result, IntegrationValidationResult)
        assert result.total_tests == 9  # CLI has 9 tests
        assert result.execution_time > 0
        assert result.config == config

    def test_validation_result_serialization(self) -> None:
        """Test validation result can be serialized."""
        framework = IntegrationTestFramework()
        result = framework.validate_integration_completeness(
            categories=[TestCategory.CLI_WORKFLOW]
        )

        result_dict = result.to_dict()

        assert "tests" in result_dict
        assert "total_tests" in result_dict
        assert "passed_tests" in result_dict
        assert "success_rate" in result_dict
        assert "config" in result_dict

    def test_test_data_manager_in_framework(self) -> None:
        """Test that framework has working test data manager."""
        framework = IntegrationTestFramework()

        data = framework.test_data_manager.get_test_data("SECO", "load")

        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    def test_report_generation_from_validation(self) -> None:
        """Test generating report from validation results."""
        framework = IntegrationTestFramework()
        result = framework.validate_integration_completeness(
            categories=[TestCategory.CLI_WORKFLOW]
        )

        report = framework.generate_report(result)

        assert "INTEGRATION TEST REPORT" in report
        assert "cli_workflow" in report
        assert str(result.total_tests) in report
