"""Tests for TrainingWorkflow orchestration.

Tests cover:
- Workflow initialization
- Stage execution
- Complete workflow run
- Workflow control (pause, resume, cancel)
- Progress tracking
- Error handling
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.orchestration.config_manager import ConfigManager
from src.orchestration.training_workflow import (
    StageResult,
    TrainingWorkflow,
    WorkflowResult,
    WorkflowStage,
    WorkflowStatus,
)


class TestEnums:
    """Tests for workflow enums."""

    def test_workflow_stage_values(self):
        """Test WorkflowStage enum values."""
        assert WorkflowStage.INITIALIZED.value == "initialized"
        assert WorkflowStage.LOADING_DATA.value == "loading_data"
        assert WorkflowStage.FEATURE_ENGINEERING.value == "feature_engineering"
        assert WorkflowStage.TRAINING.value == "training"
        assert WorkflowStage.VALIDATING.value == "validating"
        assert WorkflowStage.SAVING.value == "saving"
        assert WorkflowStage.COMPLETED.value == "completed"
        assert WorkflowStage.FAILED.value == "failed"
        assert WorkflowStage.CANCELLED.value == "cancelled"

    def test_workflow_status_values(self):
        """Test WorkflowStatus enum values."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.PAUSED.value == "paused"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_stage_result_creation(self):
        """Test creating a stage result."""
        result = StageResult(
            stage=WorkflowStage.TRAINING,
            success=True,
            started_at=datetime.now(),
        )

        assert result.stage == WorkflowStage.TRAINING
        assert result.success is True
        assert result.error is None

    def test_stage_result_with_error(self):
        """Test stage result with error."""
        result = StageResult(
            stage=WorkflowStage.LOADING_DATA,
            success=False,
            error="Data not found",
        )

        assert result.success is False
        assert result.error == "Data not found"

    def test_stage_result_to_dict(self):
        """Test conversion to dictionary."""
        result = StageResult(
            stage=WorkflowStage.TRAINING,
            success=True,
            duration_seconds=120.5,
            metrics={"mape": 5.2},
        )
        d = result.to_dict()

        assert d["stage"] == "training"
        assert d["success"] is True
        assert d["duration_seconds"] == 120.5
        assert d["metrics"]["mape"] == 5.2


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_workflow_result_creation(self):
        """Test creating a workflow result."""
        result = WorkflowResult()

        assert result.success is False
        assert result.status == WorkflowStatus.PENDING
        assert result.trained_models == {}
        assert result.error is None

    def test_workflow_result_to_dict(self):
        """Test conversion to dictionary."""
        result = WorkflowResult(
            success=True,
            status=WorkflowStatus.COMPLETED,
            duration_seconds=300.0,
            trained_models={"SECO": MagicMock()},
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["status"] == "completed"
        assert d["duration_seconds"] == 300.0
        assert "SECO" in d["trained_models"]

    def test_workflow_result_get_summary(self):
        """Test getting human-readable summary."""
        stage_result = StageResult(
            stage=WorkflowStage.TRAINING,
            success=True,
            duration_seconds=120.0,
        )
        result = WorkflowResult(
            success=True,
            status=WorkflowStatus.COMPLETED,
            duration_seconds=300.0,
            stage_results=[stage_result],
            trained_models={"SECO": MagicMock()},
            validation_results={"SECO": True},
        )

        summary = result.get_summary()
        assert "Training Workflow Summary" in summary
        assert "COMPLETED" in summary
        assert "300.00" in summary
        assert "training: OK" in summary


class TestTrainingWorkflow:
    """Tests for TrainingWorkflow class."""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager for testing."""
        return ConfigManager.default()

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 5

        X = pd.DataFrame(
            rng.standard_normal((n_samples, n_features)),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y = pd.Series(X.sum(axis=1) + rng.standard_normal(n_samples) * 0.1)

        return {"SECO": (X, y)}

    def test_workflow_initialization(self, config_manager):
        """Test workflow initialization."""
        workflow = TrainingWorkflow(config_manager)

        assert workflow.status == WorkflowStatus.PENDING
        assert workflow.current_stage == WorkflowStage.INITIALIZED

    def test_workflow_status_property(self, config_manager):
        """Test workflow status property."""
        workflow = TrainingWorkflow(config_manager)
        assert workflow.status == WorkflowStatus.PENDING

    def test_workflow_current_stage_property(self, config_manager):
        """Test current stage property."""
        workflow = TrainingWorkflow(config_manager)
        assert workflow.current_stage == WorkflowStage.INITIALIZED

    @patch("src.orchestration.training_workflow.UniversalTrainer")
    def test_run_with_data(self, mock_trainer_cls, config_manager, sample_data):
        """Test running workflow with provided data."""
        # Setup mock trainer
        mock_trainer = MagicMock()
        mock_trainer.train_multiple_areas.return_value = {"SECO": MagicMock()}
        mock_trainer.validate_trained_models.return_value = {"SECO": True}
        mock_trainer.save_all_models.return_value = {"SECO": "/path/model.pkl"}
        mock_trainer_cls.return_value = mock_trainer

        workflow = TrainingWorkflow(config_manager)
        result = workflow.run(training_data=sample_data)

        assert result.success is True
        assert result.status == WorkflowStatus.COMPLETED
        assert "SECO" in result.trained_models

    @patch("src.orchestration.training_workflow.UniversalTrainer")
    def test_run_with_data_single_area(self, mock_trainer_cls, config_manager):
        """Test running workflow with run_with_data method."""
        mock_trainer = MagicMock()
        mock_trainer.train_multiple_areas.return_value = {"SECO": MagicMock()}
        mock_trainer.validate_trained_models.return_value = {"SECO": True}
        mock_trainer.save_all_models.return_value = {"SECO": "/path/model.pkl"}
        mock_trainer_cls.return_value = mock_trainer

        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 5)))
        y = pd.Series(rng.standard_normal(50))

        workflow = TrainingWorkflow(config_manager)
        result = workflow.run_with_data(X, y, area="SECO")

        assert result.success is True
        assert "SECO" in result.trained_models

    @patch("src.orchestration.training_workflow.UniversalTrainer")
    def test_run_generates_synthetic_data(self, mock_trainer_cls, config_manager):
        """Test workflow generates synthetic data when no data provided."""
        mock_trainer = MagicMock()
        mock_trainer.train_multiple_areas.return_value = {"SECO": MagicMock()}
        mock_trainer.validate_trained_models.return_value = {"SECO": True}
        mock_trainer.save_all_models.return_value = {}
        mock_trainer_cls.return_value = mock_trainer

        workflow = TrainingWorkflow(config_manager)
        result = workflow.run()

        # Should generate synthetic data and train
        assert mock_trainer.train_multiple_areas.called

    def test_workflow_cancel(self, config_manager):
        """Test cancelling workflow."""
        workflow = TrainingWorkflow(config_manager)
        workflow.cancel()

        assert workflow.status == WorkflowStatus.CANCELLED

    def test_workflow_pause_resume(self, config_manager):
        """Test pausing and resuming workflow."""
        workflow = TrainingWorkflow(config_manager)

        workflow.pause()
        assert workflow.status == WorkflowStatus.PAUSED

        workflow._status = WorkflowStatus.PAUSED  # Simulate running state
        workflow.resume()
        assert workflow.status == WorkflowStatus.RUNNING

    def test_get_progress(self, config_manager):
        """Test getting workflow progress."""
        workflow = TrainingWorkflow(config_manager)
        progress = workflow.get_progress()

        assert "status" in progress
        assert "current_stage" in progress
        assert "progress_percent" in progress
        assert progress["status"] == "pending"

    def test_get_result(self, config_manager):
        """Test getting workflow result."""
        workflow = TrainingWorkflow(config_manager)
        result = workflow.get_result()

        assert isinstance(result, WorkflowResult)

    def test_workflow_repr(self, config_manager):
        """Test string representation."""
        workflow = TrainingWorkflow(config_manager)
        repr_str = repr(workflow)

        assert "TrainingWorkflow" in repr_str
        assert "pending" in repr_str


class TestWorkflowStages:
    """Tests for individual workflow stages."""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager."""
        return ConfigManager.default()

    @pytest.fixture
    def workflow(self, config_manager):
        """Create a workflow instance."""
        return TrainingWorkflow(config_manager)

    def test_load_data_with_provided_data(self, workflow):
        """Test loading data with provided data."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 5)))
        y = pd.Series(rng.standard_normal(50))
        data = {"test": (X, y)}

        workflow._load_data(data)
        assert "test" in workflow._training_data

    def test_load_data_with_data_loader(self, workflow):
        """Test loading data with data loader."""
        mock_loader = MagicMock()
        mock_loader.load_training_data.return_value = {
            "area1": (pd.DataFrame(), pd.Series(dtype=float))
        }
        workflow._data_loader = mock_loader

        workflow._load_data(None)
        assert mock_loader.load_training_data.called

    def test_load_data_generates_synthetic(self, workflow):
        """Test synthetic data generation."""
        workflow._load_data(None)
        # Should have generated synthetic data for default areas
        assert len(workflow._training_data) > 0

    def test_generate_features_no_pipeline(self, workflow):
        """Test feature generation without pipeline."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 5)))
        y = pd.Series(rng.standard_normal(50))
        workflow._training_data = {"test": (X, y)}

        # Should not fail without pipeline
        workflow._generate_features()
        assert "test" in workflow._training_data

    def test_generate_features_with_pipeline(self, workflow):
        """Test feature generation with pipeline."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 5)))
        y = pd.Series(rng.standard_normal(50))
        workflow._training_data = {"test": (X, y)}

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = pd.DataFrame(rng.standard_normal((50, 10)))
        workflow._feature_pipeline = mock_pipeline

        workflow._generate_features()
        assert mock_pipeline.transform.called


class TestWorkflowErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager."""
        return ConfigManager.default()

    @patch("src.orchestration.training_workflow.UniversalTrainer")
    def test_training_failure(self, mock_trainer_cls, config_manager):
        """Test handling of training failure."""
        mock_trainer = MagicMock()
        mock_trainer.train_multiple_areas.side_effect = Exception("Training failed")
        mock_trainer_cls.return_value = mock_trainer

        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 5)))
        y = pd.Series(rng.standard_normal(50))

        workflow = TrainingWorkflow(config_manager)
        result = workflow.run(training_data={"SECO": (X, y)})

        assert result.success is False
        assert result.status == WorkflowStatus.FAILED
        assert "Training failed" in result.error

    def test_stage_result_captures_error(self, config_manager):
        """Test that stage results capture errors."""
        workflow = TrainingWorkflow(config_manager)

        # Force an error in a stage
        def failing_stage():
            raise ValueError("Test error")

        workflow._execute_stage(WorkflowStage.LOADING_DATA, failing_stage)

        assert len(workflow._result.stage_results) == 1
        assert workflow._result.stage_results[0].success is False
        assert "Test error" in workflow._result.stage_results[0].error


class TestWorkflowIntegration:
    """Integration tests for training workflow."""

    @pytest.fixture
    def config_manager(self):
        """Create ConfigManager with custom settings."""
        manager = ConfigManager.default()
        manager.update_training_config(
            parallel_workers=1,
            save_models=False,
        )
        return manager

    @patch("src.orchestration.training_workflow.UniversalTrainer")
    def test_complete_workflow_execution(self, mock_trainer_cls, config_manager):
        """Test complete workflow execution."""
        # Setup comprehensive mock
        mock_trainer = MagicMock()
        mock_trainer.train_multiple_areas.return_value = {
            "SECO": MagicMock(),
            "S": MagicMock(),
        }
        mock_trainer.validate_trained_models.return_value = {
            "SECO": True,
            "S": True,
        }
        mock_trainer.save_all_models.return_value = {}
        mock_trainer_cls.return_value = mock_trainer

        rng = np.random.default_rng(42)
        training_data = {}
        for area in ["SECO", "S"]:
            X = pd.DataFrame(rng.standard_normal((100, 10)))
            y = pd.Series(rng.standard_normal(100))
            training_data[area] = (X, y)

        workflow = TrainingWorkflow(config_manager)
        result = workflow.run(training_data=training_data)

        # Verify result
        assert result.success is True
        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.trained_models) == 2
        assert result.validation_results["SECO"] is True
        assert result.validation_results["S"] is True

        # Verify stages completed
        assert len(result.stage_results) == 5  # All 5 stages
        for stage_result in result.stage_results:
            assert stage_result.success is True

    @patch("src.orchestration.training_workflow.UniversalTrainer")
    def test_workflow_timing(self, mock_trainer_cls, config_manager):
        """Test workflow timing is recorded."""
        mock_trainer = MagicMock()
        mock_trainer.train_multiple_areas.return_value = {"SECO": MagicMock()}
        mock_trainer.validate_trained_models.return_value = {"SECO": True}
        mock_trainer.save_all_models.return_value = {}
        mock_trainer_cls.return_value = mock_trainer

        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 5)))
        y = pd.Series(rng.standard_normal(50))

        workflow = TrainingWorkflow(config_manager)
        result = workflow.run(training_data={"SECO": (X, y)})

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.duration_seconds >= 0

    @patch("src.orchestration.training_workflow.UniversalTrainer")
    def test_generate_report(self, mock_trainer_cls, config_manager):
        """Test report generation."""
        mock_trainer = MagicMock()
        mock_trainer.train_multiple_areas.return_value = {"SECO": MagicMock()}
        mock_trainer.validate_trained_models.return_value = {"SECO": True}
        mock_trainer.save_all_models.return_value = {}
        mock_trainer.generate_training_report.return_value = "Training Report Content"
        mock_trainer_cls.return_value = mock_trainer

        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((50, 5)))
        y = pd.Series(rng.standard_normal(50))

        workflow = TrainingWorkflow(config_manager)
        workflow.run(training_data={"SECO": (X, y)})

        report = workflow.generate_report()
        assert "Training Report" in report


class TestBrazilianLoadForecasting:
    """Tests specific to Brazilian load forecasting scenario."""

    @pytest.fixture
    def brazil_config_manager(self):
        """Create ConfigManager for Brazil scenario."""
        manager = ConfigManager.default()
        manager.update_training_config(
            areas=["SECO", "S", "NE", "N"],
            horizons=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            parallel_workers=2,
            save_models=False,
        )
        return manager

    @patch("src.orchestration.training_workflow.UniversalTrainer")
    def test_brazil_subsystems_training(self, mock_trainer_cls, brazil_config_manager):
        """Test training for Brazilian subsystems."""
        mock_trainer = MagicMock()
        mock_trainer.train_multiple_areas.return_value = {
            "SECO": MagicMock(),
            "S": MagicMock(),
            "NE": MagicMock(),
            "N": MagicMock(),
        }
        mock_trainer.validate_trained_models.return_value = {
            "SECO": True,
            "S": True,
            "NE": True,
            "N": True,
        }
        mock_trainer.save_all_models.return_value = {}
        mock_trainer_cls.return_value = mock_trainer

        # Generate training data for each subsystem
        rng = np.random.default_rng(42)
        training_data = {}
        for area in ["SECO", "S", "NE", "N"]:
            X = pd.DataFrame(rng.standard_normal((200, 15)))
            y = pd.Series(rng.standard_normal(200))
            training_data[area] = (X, y)

        workflow = TrainingWorkflow(brazil_config_manager)
        result = workflow.run(training_data=training_data)

        assert result.success is True
        assert len(result.trained_models) == 4
        assert all(result.validation_results.values())
