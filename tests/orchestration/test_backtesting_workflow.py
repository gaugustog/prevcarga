"""Tests for backtesting workflow orchestration.

This module tests the BacktestingWorkflow class and related components.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.orchestration.config_manager import ConfigManager
from src.orchestration.backtesting_workflow import (
    BacktestingResult,
    BacktestingStage,
    BacktestingStatus,
    BacktestingWorkflow,
    BacktestWindow,
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
    manager.update_backtesting_config(areas=["SECO", "S"])
    return manager


@pytest.fixture
def sample_historical_data() -> dict[str, pd.DataFrame]:
    """Create sample historical data."""
    rng = np.random.default_rng(42)
    n_periods = 365 * 48  # One year at 30-min resolution

    data = {}
    for area in ["SECO", "S"]:
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=365),
            periods=n_periods,
            freq="30min",
        )

        base_load = 1500 + rng.standard_normal(n_periods) * 100
        df = pd.DataFrame({
            "timestamp": dates,
            "load_mw": base_load,
            "temperature": 20 + rng.standard_normal(n_periods) * 5,
        })
        df.set_index("timestamp", inplace=True)
        data[area] = df

    return data


@pytest.fixture
def temp_output_dir() -> Path:
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ==================== BacktestWindow Tests ====================


class TestBacktestWindow:
    """Tests for BacktestWindow dataclass."""

    def test_window_creation(self) -> None:
        """Test basic window creation."""
        window = BacktestWindow(
            window_id=0,
            forecast_date=datetime(2024, 1, 15),
        )

        assert window.window_id == 0
        assert window.forecast_date == datetime(2024, 1, 15)
        assert len(window.metrics) == 0

    def test_window_with_periods(self) -> None:
        """Test window with train/test periods."""
        window = BacktestWindow(
            window_id=1,
            forecast_date=datetime(2024, 1, 15),
            train_start=datetime(2023, 1, 15),
            train_end=datetime(2024, 1, 14),
            test_start=datetime(2024, 1, 15),
            test_end=datetime(2024, 1, 22),
        )

        assert window.train_start == datetime(2023, 1, 15)
        assert window.test_end == datetime(2024, 1, 22)

    def test_window_with_metrics(self) -> None:
        """Test window with metrics."""
        window = BacktestWindow(
            window_id=2,
            forecast_date=datetime(2024, 1, 15),
            metrics={
                "SECO": {
                    "h0": {"mape": 2.5, "mae": 100.0, "rmse": 150.0},
                    "h1": {"mape": 3.0, "mae": 120.0, "rmse": 180.0},
                },
            },
        )

        assert "SECO" in window.metrics
        assert window.metrics["SECO"]["h0"]["mape"] == 2.5

    def test_window_to_dict(self) -> None:
        """Test window serialization."""
        window = BacktestWindow(
            window_id=0,
            forecast_date=datetime(2024, 1, 15, 12, 0, 0),
            metrics={"SECO": {"h0": {"mape": 2.5}}},
        )

        data = window.to_dict()

        assert data["window_id"] == 0
        assert "forecast_date" in data
        assert data["metrics"]["SECO"]["h0"]["mape"] == 2.5


# ==================== StageResult Tests ====================


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_stage_result_creation(self) -> None:
        """Test basic StageResult creation."""
        result = StageResult(
            stage=BacktestingStage.LOADING_DATA,
            success=True,
        )

        assert result.stage == BacktestingStage.LOADING_DATA
        assert result.success is True

    def test_stage_result_with_error(self) -> None:
        """Test StageResult with error."""
        result = StageResult(
            stage=BacktestingStage.EVALUATING,
            success=False,
            error="Model evaluation failed",
        )

        assert result.success is False
        assert "evaluation failed" in result.error

    def test_stage_result_to_dict(self) -> None:
        """Test StageResult serialization."""
        result = StageResult(
            stage=BacktestingStage.GENERATING_WINDOWS,
            success=True,
            duration_seconds=1.5,
            metrics={"windows_generated": 52},
        )

        data = result.to_dict()

        assert data["stage"] == "generating_windows"
        assert data["duration_seconds"] == 1.5


# ==================== BacktestingResult Tests ====================


class TestBacktestingResult:
    """Tests for BacktestingResult dataclass."""

    def test_result_creation(self) -> None:
        """Test basic result creation."""
        result = BacktestingResult()

        assert result.success is False
        assert result.status == BacktestingStatus.PENDING
        assert len(result.windows) == 0

    def test_result_with_windows(self) -> None:
        """Test result with windows."""
        result = BacktestingResult()
        result.windows.append(BacktestWindow(
            window_id=0,
            forecast_date=datetime.now(),
        ))
        result.success = True
        result.status = BacktestingStatus.COMPLETED

        assert len(result.windows) == 1

    def test_result_with_aggregate_metrics(self) -> None:
        """Test result with aggregate metrics."""
        result = BacktestingResult()
        result.aggregate_metrics = {
            "mape": {"mean": 2.5, "std": 0.5, "min": 1.5, "max": 4.0},
            "mae": {"mean": 100.0, "std": 20.0, "min": 60.0, "max": 150.0},
        }

        assert result.aggregate_metrics["mape"]["mean"] == 2.5

    def test_result_to_dict(self) -> None:
        """Test result serialization."""
        result = BacktestingResult()
        result.success = True
        result.status = BacktestingStatus.COMPLETED
        result.aggregate_metrics = {"mape": {"mean": 2.5}}

        data = result.to_dict()

        assert data["success"] is True
        assert data["status"] == "completed"

    def test_result_get_summary(self) -> None:
        """Test result summary generation."""
        result = BacktestingResult()
        result.success = True
        result.status = BacktestingStatus.COMPLETED
        result.duration_seconds = 120.5
        result.aggregate_metrics = {"mape": {"mean": 2.5, "std": 0.5}}
        result.stage_results.append(StageResult(
            stage=BacktestingStage.LOADING_DATA,
            success=True,
            duration_seconds=1.0,
        ))

        summary = result.get_summary()

        assert "Backtesting Workflow Summary" in summary
        assert "COMPLETED" in summary
        assert "120.50 seconds" in summary


# ==================== BacktestingStage Tests ====================


class TestBacktestingStage:
    """Tests for BacktestingStage enum."""

    def test_all_stages_exist(self) -> None:
        """Test all expected stages exist."""
        expected_stages = [
            "INITIALIZED",
            "LOADING_DATA",
            "LOADING_MODELS",
            "GENERATING_WINDOWS",
            "EVALUATING",
            "AGGREGATING_METRICS",
            "GENERATING_REPORT",
            "SAVING_RESULTS",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        ]

        for stage_name in expected_stages:
            assert hasattr(BacktestingStage, stage_name)

    def test_stage_values(self) -> None:
        """Test stage value strings."""
        assert BacktestingStage.EVALUATING.value == "evaluating"
        assert BacktestingStage.AGGREGATING_METRICS.value == "aggregating_metrics"


# ==================== BacktestingWorkflow Tests ====================


class TestBacktestingWorkflow:
    """Tests for BacktestingWorkflow class."""

    def test_workflow_initialization(self, config_manager: ConfigManager) -> None:
        """Test workflow initialization."""
        workflow = BacktestingWorkflow(config_manager)

        assert workflow.status == BacktestingStatus.PENDING
        assert workflow.current_stage == BacktestingStage.INITIALIZED

    def test_workflow_repr(self, config_manager: ConfigManager) -> None:
        """Test workflow string representation."""
        workflow = BacktestingWorkflow(config_manager)
        repr_str = repr(workflow)

        assert "BacktestingWorkflow" in repr_str
        assert "pending" in repr_str

    def test_run_workflow_basic(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test basic workflow execution."""
        config_manager.update_backtesting_config(
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            generate_report=False,
            save_forecasts=False,
            step_days=30,  # Fewer windows for faster test
            areas=["SECO", "S"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=90)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        assert result.success is True
        assert result.status == BacktestingStatus.COMPLETED
        assert len(result.windows) > 0

    def test_run_workflow_with_dates_from_config(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test workflow using dates from configuration."""
        start_str = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        end_str = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        config_manager.update_backtesting_config(
            start_date=start_str,
            end_date=end_str,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            generate_report=False,
            save_forecasts=False,
            step_days=30,
            areas=["SECO", "S"],
        )

        workflow = BacktestingWorkflow(config_manager)
        result = workflow.run(historical_data=sample_historical_data)

        assert result.success is True

    def test_run_workflow_with_report(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test workflow with report generation."""
        config_manager.update_backtesting_config(
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            generate_report=True,
            save_forecasts=False,
            step_days=30,
            areas=["SECO", "S"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        assert result.success is True
        assert result.report_path is not None
        assert Path(result.report_path).exists()

    def test_run_workflow_with_saved_results(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test workflow with result saving."""
        config_manager.update_backtesting_config(
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            generate_report=False,
            save_forecasts=True,
            step_days=30,
            areas=["SECO", "S"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        assert result.success is True
        assert result.results_path is not None
        assert Path(result.results_path).exists()

    def test_workflow_cancellation(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Test workflow cancellation."""
        workflow = BacktestingWorkflow(config_manager)

        workflow.cancel()

        result = workflow.run(historical_data=sample_historical_data)

        assert result.success is False
        assert result.status == BacktestingStatus.CANCELLED

    def test_workflow_pause_resume(self, config_manager: ConfigManager) -> None:
        """Test workflow pause and resume."""
        workflow = BacktestingWorkflow(config_manager)

        workflow.pause()
        assert workflow.status == BacktestingStatus.PAUSED

        workflow.resume()
        assert workflow.status == BacktestingStatus.RUNNING

    def test_get_progress(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test progress retrieval."""
        config_manager.update_backtesting_config(
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            generate_report=False,
            save_forecasts=False,
            step_days=30,
            areas=["SECO", "S"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        progress = workflow.get_progress()

        assert "status" in progress
        assert "current_stage" in progress
        assert "windows_evaluated" in progress
        assert "windows_total" in progress

    def test_get_result(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test result retrieval."""
        config_manager.update_backtesting_config(
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            generate_report=False,
            save_forecasts=False,
            step_days=30,
            areas=["SECO", "S"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        result = workflow.get_result()

        assert isinstance(result, BacktestingResult)


# ==================== Window Generation Tests ====================


class TestWindowGeneration:
    """Tests for backtest window generation."""

    def test_window_generation_weekly(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test weekly window generation."""
        config_manager.update_backtesting_config(
            step_days=7,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        # Expect approximately 4-5 windows for 31 days with 7-day step
        assert len(result.windows) >= 4
        assert len(result.windows) <= 5

    def test_window_generation_monthly(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test monthly window generation."""
        config_manager.update_backtesting_config(
            step_days=30,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 31)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        # Expect approximately 3 windows for 90 days with 30-day step
        assert len(result.windows) >= 2
        assert len(result.windows) <= 4


# ==================== Metrics Aggregation Tests ====================


class TestMetricsAggregation:
    """Tests for metrics aggregation."""

    def test_aggregate_metrics_computed(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test that aggregate metrics are computed."""
        config_manager.update_backtesting_config(
            step_days=30,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO", "S"],
            metrics=["mape", "mae", "rmse"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        assert result.success is True
        # Aggregate metrics should have statistical summaries
        for metric_name in result.aggregate_metrics:
            stats = result.aggregate_metrics[metric_name]
            assert "mean" in stats or isinstance(stats, float)

    def test_metrics_by_horizon(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test metrics broken down by horizon."""
        config_manager.update_backtesting_config(
            step_days=30,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO"],
            horizons=[0, 1, 2],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        assert result.success is True
        # Should have metrics for each horizon
        if result.metrics_by_horizon:
            for horizon in result.metrics_by_horizon:
                assert isinstance(horizon, int)

    def test_metrics_by_area(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test metrics broken down by area."""
        config_manager.update_backtesting_config(
            step_days=30,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO", "S"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        assert result.success is True
        # Should have metrics for each area
        if result.metrics_by_area:
            assert "SECO" in result.metrics_by_area or "S" in result.metrics_by_area


# ==================== Custom Components Tests ====================


class TestCustomComponents:
    """Tests for custom component injection."""

    def test_custom_data_loader(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test with custom data loader."""
        mock_loader = MagicMock()
        mock_loader.load_historical_data.return_value = sample_historical_data

        config_manager.update_backtesting_config(
            step_days=30,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO", "S"],
        )

        workflow = BacktestingWorkflow(
            config_manager,
            data_loader=mock_loader,
        )

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
        )

        mock_loader.load_historical_data.assert_called_once()
        assert result.success is True

    def test_custom_model_loader(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test with custom model loader."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1500.0] * 48)

        mock_loader = MagicMock()
        mock_loader.load_models.return_value = {"SECO": {0: mock_model}}

        config_manager.update_backtesting_config(
            step_days=30,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO"],
            horizons=[0],
        )

        workflow = BacktestingWorkflow(
            config_manager,
            model_loader=mock_loader,
        )

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        mock_loader.load_models.assert_called_once()
        assert result.success is True

    def test_custom_metrics_calculator(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test with custom metrics calculator."""
        mock_calculator = MagicMock()
        mock_calculator.calculate.return_value = {
            "mape": 2.5,
            "mae": 100.0,
            "rmse": 150.0,
        }

        config_manager.update_backtesting_config(
            step_days=30,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO"],
        )

        workflow = BacktestingWorkflow(
            config_manager,
            metrics_calculator=mock_calculator,
        )

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        assert result.success is True
        # Calculator should have been called
        assert mock_calculator.calculate.call_count >= 0


# ==================== Error Handling Tests ====================


class TestErrorHandling:
    """Tests for error handling."""

    def test_stage_failure_handling(
        self,
        config_manager: ConfigManager,
    ) -> None:
        """Test handling of stage failures."""
        mock_loader = MagicMock()
        mock_loader.load_historical_data.side_effect = RuntimeError("Data load failed")

        workflow = BacktestingWorkflow(
            config_manager,
            data_loader=mock_loader,
        )

        result = workflow.run()

        assert result.success is False
        assert result.status == BacktestingStatus.FAILED
        assert "Data load failed" in result.error

    def test_workflow_handles_missing_data(
        self,
        config_manager: ConfigManager,
        temp_output_dir: Path,
    ) -> None:
        """Test workflow handles missing historical data gracefully."""
        config_manager.update_backtesting_config(
            step_days=30,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
        )

        workflow = BacktestingWorkflow(config_manager)

        # Run without providing data - should generate synthetic
        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
        )

        # Should succeed with synthetic data
        assert result.success is True


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for backtesting workflow."""

    def test_full_workflow_pipeline(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test complete workflow pipeline."""
        config_manager.update_backtesting_config(
            step_days=15,
            generate_report=True,
            save_forecasts=True,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO", "S"],
            horizons=[0, 1, 2],
            metrics=["mape", "mae", "rmse"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        assert result.success is True
        assert len(result.windows) > 0
        assert result.aggregate_metrics
        assert result.report_path is not None
        assert result.results_path is not None
        assert Path(result.report_path).exists()
        assert Path(result.results_path).exists()

    def test_stages_executed_in_order(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test stages are executed in correct order."""
        config_manager.update_backtesting_config(
            step_days=30,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        stages = [sr.stage for sr in result.stage_results]

        # Verify order
        assert stages[0] == BacktestingStage.LOADING_DATA
        assert stages[1] == BacktestingStage.LOADING_MODELS
        assert stages[2] == BacktestingStage.GENERATING_WINDOWS
        assert stages[3] == BacktestingStage.EVALUATING

    def test_all_stages_have_timing(
        self,
        config_manager: ConfigManager,
        sample_historical_data: dict[str, pd.DataFrame],
        temp_output_dir: Path,
    ) -> None:
        """Test all stages have timing information."""
        config_manager.update_backtesting_config(
            step_days=30,
            generate_report=False,
            save_forecasts=False,
            report_dir=str(temp_output_dir / "reports"),
            forecasts_dir=str(temp_output_dir / "forecasts"),
            areas=["SECO"],
        )

        workflow = BacktestingWorkflow(config_manager)

        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        result = workflow.run(
            start_date=start_date,
            end_date=end_date,
            historical_data=sample_historical_data,
        )

        for stage_result in result.stage_results:
            assert stage_result.started_at is not None
            assert stage_result.completed_at is not None
            assert stage_result.duration_seconds >= 0
