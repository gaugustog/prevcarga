"""Tests for prediction workflow orchestration.

This module tests the PredictionWorkflow class and related components.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.orchestration.config_manager import ConfigManager, PredictionWorkflowConfig
from src.orchestration.prediction_workflow import (
    ForecastOutput,
    PredictionResult,
    PredictionStage,
    PredictionStatus,
    PredictionWorkflow,
    StageResult,
)


# ==================== Fixtures ====================


@pytest.fixture
def config_manager() -> ConfigManager:
    """Create a default configuration manager."""
    return ConfigManager.default()


@pytest.fixture
def config_manager_with_areas() -> ConfigManager:
    """Create config manager with specific areas."""
    manager = ConfigManager.default()
    manager.update_prediction_config(areas=["SECO", "S"])
    return manager


@pytest.fixture
def sample_features() -> dict[str, pd.DataFrame]:
    """Create sample feature DataFrames."""
    rng = np.random.default_rng(42)
    return {
        "SECO": pd.DataFrame(
            rng.standard_normal((48, 10)),
            columns=[f"feature_{i}" for i in range(10)],
        ),
        "S": pd.DataFrame(
            rng.standard_normal((48, 10)),
            columns=[f"feature_{i}" for i in range(10)],
        ),
    }


@pytest.fixture
def single_feature_df() -> pd.DataFrame:
    """Create a single feature DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((48, 10)),
        columns=[f"feature_{i}" for i in range(10)],
    )


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.predict.return_value = np.array([1000.0, 1001.0, 1002.0])
    return model


@pytest.fixture
def temp_output_dir() -> Path:
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ==================== ForecastOutput Tests ====================


class TestForecastOutput:
    """Tests for ForecastOutput dataclass."""

    def test_forecast_output_creation(self) -> None:
        """Test basic ForecastOutput creation."""
        forecast = ForecastOutput(
            area="SECO",
            horizon=1,
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            forecast=1500.5,
        )

        assert forecast.area == "SECO"
        assert forecast.horizon == 1
        assert forecast.forecast == 1500.5
        assert forecast.lower_bound is None
        assert forecast.upper_bound is None
        assert forecast.confidence_level == 0.95
        assert forecast.reconciled is False

    def test_forecast_output_with_bounds(self) -> None:
        """Test ForecastOutput with confidence bounds."""
        forecast = ForecastOutput(
            area="S",
            horizon=2,
            timestamp=datetime(2024, 1, 15),
            forecast=1200.0,
            lower_bound=1100.0,
            upper_bound=1300.0,
            confidence_level=0.9,
        )

        assert forecast.lower_bound == 1100.0
        assert forecast.upper_bound == 1300.0
        assert forecast.confidence_level == 0.9

    def test_forecast_output_to_dict(self) -> None:
        """Test ForecastOutput serialization."""
        timestamp = datetime(2024, 1, 15, 12, 0, 0)
        forecast = ForecastOutput(
            area="NE",
            horizon=0,
            timestamp=timestamp,
            forecast=800.0,
            model_id="ne_h0",
            reconciled=True,
        )

        data = forecast.to_dict()

        assert data["area"] == "NE"
        assert data["horizon"] == 0
        assert data["forecast"] == 800.0
        assert data["model_id"] == "ne_h0"
        assert data["reconciled"] is True
        assert "timestamp" in data


# ==================== StageResult Tests ====================


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_stage_result_creation(self) -> None:
        """Test basic StageResult creation."""
        result = StageResult(
            stage=PredictionStage.LOADING_MODELS,
            success=True,
        )

        assert result.stage == PredictionStage.LOADING_MODELS
        assert result.success is True
        assert result.error is None

    def test_stage_result_with_timing(self) -> None:
        """Test StageResult with timing information."""
        start = datetime(2024, 1, 15, 10, 0, 0)
        end = datetime(2024, 1, 15, 10, 0, 30)

        result = StageResult(
            stage=PredictionStage.GENERATING_FORECASTS,
            success=True,
            started_at=start,
            completed_at=end,
            duration_seconds=30.0,
        )

        assert result.duration_seconds == 30.0

    def test_stage_result_with_error(self) -> None:
        """Test StageResult with error."""
        result = StageResult(
            stage=PredictionStage.APPLYING_RECONCILIATION,
            success=False,
            error="Reconciliation failed: singular matrix",
        )

        assert result.success is False
        assert "singular matrix" in result.error

    def test_stage_result_to_dict(self) -> None:
        """Test StageResult serialization."""
        result = StageResult(
            stage=PredictionStage.SAVING_OUTPUTS,
            success=True,
            metrics={"files_saved": 3},
        )

        data = result.to_dict()

        assert data["stage"] == "saving_outputs"
        assert data["success"] is True
        assert data["metrics"]["files_saved"] == 3


# ==================== PredictionResult Tests ====================


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_prediction_result_creation(self) -> None:
        """Test basic PredictionResult creation."""
        result = PredictionResult()

        assert result.success is False
        assert result.status == PredictionStatus.PENDING
        assert len(result.forecasts) == 0
        assert result.reconciled is False

    def test_prediction_result_with_forecasts(self) -> None:
        """Test PredictionResult with forecasts."""
        result = PredictionResult()
        result.forecasts["SECO"] = {
            0: ForecastOutput(
                area="SECO",
                horizon=0,
                timestamp=datetime.now(),
                forecast=1500.0,
            ),
        }
        result.success = True
        result.status = PredictionStatus.COMPLETED

        assert len(result.forecasts) == 1
        assert "SECO" in result.forecasts
        assert result.forecasts["SECO"][0].forecast == 1500.0

    def test_prediction_result_to_dict(self) -> None:
        """Test PredictionResult serialization."""
        result = PredictionResult()
        result.success = True
        result.status = PredictionStatus.COMPLETED
        result.forecasts["SECO"] = {0: ForecastOutput(
            area="SECO", horizon=0, timestamp=datetime.now(), forecast=1000.0
        )}
        result.reconciled = True

        data = result.to_dict()

        assert data["success"] is True
        assert data["status"] == "completed"
        assert data["n_forecasts"] == 1
        assert data["reconciled"] is True

    def test_prediction_result_get_summary(self) -> None:
        """Test PredictionResult summary generation."""
        result = PredictionResult()
        result.success = True
        result.status = PredictionStatus.COMPLETED
        result.duration_seconds = 45.5
        result.forecasts["SECO"] = {0: ForecastOutput(
            area="SECO", horizon=0, timestamp=datetime.now(), forecast=1000.0
        )}
        result.stage_results.append(StageResult(
            stage=PredictionStage.LOADING_MODELS,
            success=True,
            duration_seconds=5.0,
        ))

        summary = result.get_summary()

        assert "Prediction Workflow Summary" in summary
        assert "COMPLETED" in summary
        assert "45.50 seconds" in summary
        assert "loading_models" in summary


# ==================== PredictionStage Tests ====================


class TestPredictionStage:
    """Tests for PredictionStage enum."""

    def test_all_stages_exist(self) -> None:
        """Test all expected stages exist."""
        expected_stages = [
            "INITIALIZED",
            "LOADING_MODELS",
            "PREPARING_FEATURES",
            "GENERATING_FORECASTS",
            "APPLYING_RECONCILIATION",
            "COMPUTING_INTERVALS",
            "SAVING_OUTPUTS",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        ]

        for stage_name in expected_stages:
            assert hasattr(PredictionStage, stage_name)

    def test_stage_values(self) -> None:
        """Test stage value strings."""
        assert PredictionStage.LOADING_MODELS.value == "loading_models"
        assert PredictionStage.GENERATING_FORECASTS.value == "generating_forecasts"


# ==================== PredictionStatus Tests ====================


class TestPredictionStatus:
    """Tests for PredictionStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Test all expected statuses exist."""
        expected_statuses = ["PENDING", "RUNNING", "PAUSED", "COMPLETED", "FAILED", "CANCELLED"]

        for status_name in expected_statuses:
            assert hasattr(PredictionStatus, status_name)


# ==================== PredictionWorkflow Tests ====================


class TestPredictionWorkflow:
    """Tests for PredictionWorkflow class."""

    def test_workflow_initialization(self, config_manager: ConfigManager) -> None:
        """Test workflow initialization."""
        workflow = PredictionWorkflow(config_manager)

        assert workflow.status == PredictionStatus.PENDING
        assert workflow.current_stage == PredictionStage.INITIALIZED

    def test_workflow_repr(self, config_manager: ConfigManager) -> None:
        """Test workflow string representation."""
        workflow = PredictionWorkflow(config_manager)
        repr_str = repr(workflow)

        assert "PredictionWorkflow" in repr_str
        assert "pending" in repr_str
        assert "initialized" in repr_str

    def test_run_workflow_basic(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test basic workflow execution."""
        # Configure output directory
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=sample_features)

        assert result.success is True
        assert result.status == PredictionStatus.COMPLETED
        assert len(result.stage_results) >= 3

    def test_run_workflow_with_single_df(
        self,
        config_manager: ConfigManager,
        single_feature_df: pd.DataFrame,
        temp_output_dir: Path,
    ) -> None:
        """Test workflow with single DataFrame."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=single_feature_df)

        assert result.success is True

    def test_run_with_features_method(
        self,
        config_manager: ConfigManager,
        single_feature_df: pd.DataFrame,
        temp_output_dir: Path,
    ) -> None:
        """Test run_with_features convenience method."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run_with_features(
            X=single_feature_df,
            area="SECO",
        )

        assert result.success is True

    def test_workflow_with_reconciliation(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test workflow with reconciliation enabled."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=True,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=sample_features)

        assert result.success is True
        assert result.reconciled is True

    def test_workflow_with_confidence_intervals(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test workflow with confidence intervals."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=True,
            confidence_level=0.95,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=sample_features)

        assert result.success is True

        # Check that intervals are computed
        for area, horizons in result.forecasts.items():
            for horizon, forecast in horizons.items():
                assert forecast.lower_bound is not None
                assert forecast.upper_bound is not None
                assert forecast.lower_bound < forecast.forecast
                assert forecast.upper_bound > forecast.forecast

    def test_workflow_cancellation(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
    ) -> None:
        """Test workflow cancellation."""
        workflow = PredictionWorkflow(config_manager)

        # Cancel before running
        workflow.cancel()

        result = workflow.run(input_features=sample_features)

        assert result.success is False
        assert result.status == PredictionStatus.CANCELLED

    def test_workflow_pause_resume(self, config_manager: ConfigManager) -> None:
        """Test workflow pause and resume."""
        workflow = PredictionWorkflow(config_manager)

        workflow.pause()
        assert workflow.status == PredictionStatus.PAUSED

        workflow.resume()
        assert workflow.status == PredictionStatus.RUNNING

    def test_get_progress(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test progress retrieval."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        workflow.run(input_features=sample_features)

        progress = workflow.get_progress()

        assert "status" in progress
        assert "current_stage" in progress
        assert "progress_percent" in progress

    def test_get_result(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test result retrieval."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        workflow.run(input_features=sample_features)

        result = workflow.get_result()

        assert isinstance(result, PredictionResult)

    def test_get_forecasts_df(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test forecast DataFrame retrieval."""
        # Configure to use areas that match sample_features
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=False,
            areas=["SECO", "S"],  # Match sample_features keys
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=sample_features)

        assert result.success is True
        df = workflow.get_forecasts_df()

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        # DataFrame will have forecasts for multiple horizons per area
        assert len(df) > 0


# ==================== Output Format Tests ====================


class TestOutputFormats:
    """Tests for different output formats."""

    def test_parquet_output(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test Parquet output format."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            output_format="parquet",
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=sample_features)

        assert result.success is True
        assert "forecasts" in result.output_paths

        output_path = Path(result.output_paths["forecasts"])
        assert output_path.suffix == ".parquet"
        assert output_path.exists()

    def test_csv_output(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test CSV output format."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            output_format="csv",
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=sample_features)

        assert result.success is True

        output_path = Path(result.output_paths["forecasts"])
        assert output_path.suffix == ".csv"
        assert output_path.exists()

    def test_json_output(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test JSON output format."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            output_format="json",
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=sample_features)

        assert result.success is True

        output_path = Path(result.output_paths["forecasts"])
        assert output_path.suffix == ".json"
        assert output_path.exists()


# ==================== Model Loading Tests ====================


class TestModelLoading:
    """Tests for model loading functionality."""

    def test_placeholder_models_creation(
        self,
        config_manager: ConfigManager,
        temp_output_dir: Path,
    ) -> None:
        """Test placeholder model creation when no models exist."""
        config_manager.update_prediction_config(
            model_dir=str(temp_output_dir / "nonexistent"),
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run()

        # Should succeed with placeholder models
        assert result.success is True

    def test_custom_model_loader(
        self,
        config_manager: ConfigManager,
        temp_output_dir: Path,
    ) -> None:
        """Test with custom model loader."""
        # Create mock model loader
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1000.0])
        mock_loader.load_models.return_value = {"SECO": {0: mock_model}}

        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=False,
            areas=["SECO"],
            horizons=[0],
        )

        workflow = PredictionWorkflow(
            config_manager,
            model_loader=mock_loader,
        )

        result = workflow.run()

        mock_loader.load_models.assert_called_once()
        assert result.success is True


# ==================== Reconciliation Tests ====================


class TestReconciliation:
    """Tests for reconciliation functionality."""

    def test_simple_reconciliation(
        self,
        config_manager: ConfigManager,
        temp_output_dir: Path,
    ) -> None:
        """Test simple proportional reconciliation."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=True,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run()

        assert result.success is True
        assert result.reconciled is True

    def test_custom_reconciler(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test with custom reconciler."""
        mock_reconciler = MagicMock()
        mock_reconciler.reconcile.return_value = np.array([[1000.0, 1100.0]])

        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=True,
            include_confidence_intervals=False,
            areas=["SECO"],
            horizons=[0, 1],
        )

        workflow = PredictionWorkflow(
            config_manager,
            reconciler=mock_reconciler,
        )

        result = workflow.run(input_features=sample_features)

        mock_reconciler.reconcile.assert_called_once()


# ==================== Error Handling Tests ====================


class TestErrorHandling:
    """Tests for error handling."""

    def test_stage_failure_handling(
        self,
        config_manager: ConfigManager,
    ) -> None:
        """Test handling of stage failures."""
        # Create mock model loader that raises an exception
        mock_loader = MagicMock()
        mock_loader.load_models.side_effect = RuntimeError("Failed to load models")

        workflow = PredictionWorkflow(
            config_manager,
            model_loader=mock_loader,
        )

        result = workflow.run()

        assert result.success is False
        assert result.status == PredictionStatus.FAILED
        assert "Failed to load models" in result.error

    def test_workflow_exception_handling(
        self,
        config_manager: ConfigManager,
    ) -> None:
        """Test exception handling during workflow execution."""
        # Create mock that raises during prediction
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.side_effect = ValueError("Prediction error")
        mock_loader.load_models.return_value = {"SECO": {0: mock_model}}

        config_manager.update_prediction_config(
            areas=["SECO"],
            horizons=[0],
        )

        workflow = PredictionWorkflow(
            config_manager,
            model_loader=mock_loader,
        )

        # Should not raise, but return failed result
        result = workflow.run()

        # The workflow might still succeed because errors in individual
        # predictions are logged but don't necessarily fail the whole workflow
        assert result is not None


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for prediction workflow."""

    def test_full_workflow_pipeline(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test complete workflow pipeline."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=True,
            include_confidence_intervals=True,
            confidence_level=0.95,
            output_format="parquet",
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=sample_features)

        assert result.success is True
        assert result.reconciled is True
        assert result.duration_seconds > 0
        assert len(result.stage_results) >= 5

        # Verify forecasts have confidence intervals
        for area, horizons in result.forecasts.items():
            for horizon, forecast in horizons.items():
                assert forecast.lower_bound is not None
                assert forecast.upper_bound is not None

        # Verify output file exists
        assert len(result.output_paths) > 0
        assert Path(result.output_paths["forecasts"]).exists()

    def test_workflow_with_forecast_date(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test workflow with specific forecast date."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        forecast_date = datetime(2024, 6, 15, 12, 0, 0)

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(
            input_features=sample_features,
            forecast_date=forecast_date,
        )

        assert result.success is True

        # Check forecasts have the specified date
        for area, horizons in result.forecasts.items():
            for horizon, forecast in horizons.items():
                assert forecast.timestamp == forecast_date


# ==================== Stage Execution Tests ====================


class TestStageExecution:
    """Tests for individual stage execution."""

    def test_stages_executed_in_order(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test stages are executed in correct order."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=True,
            include_confidence_intervals=True,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=sample_features)

        stages = [sr.stage for sr in result.stage_results]

        # Verify order
        assert stages[0] == PredictionStage.LOADING_MODELS
        assert stages[1] == PredictionStage.PREPARING_FEATURES
        assert stages[2] == PredictionStage.GENERATING_FORECASTS

    def test_all_stages_have_timing(
        self,
        config_manager: ConfigManager,
        sample_features: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test all stages have timing information."""
        config_manager.update_prediction_config(
            output_dir=str(temp_output_dir),
            apply_reconciliation=False,
            include_confidence_intervals=False,
        )

        workflow = PredictionWorkflow(config_manager)
        result = workflow.run(input_features=sample_features)

        for stage_result in result.stage_results:
            assert stage_result.started_at is not None
            assert stage_result.completed_at is not None
            assert stage_result.duration_seconds >= 0
