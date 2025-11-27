"""Tests for training CLI commands.

This module tests the train command group including train model
and train batch subcommands.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from src.cli.commands.train import (
    VALID_AREAS,
    VALID_MODEL_TYPES,
    TrainingProgressBar,
    _validate_training_config,
    execute_training,
    train,
    train_batch,
    train_model,
    validate_area_selection,
    validate_date_range,
    validate_model_selection,
)
from src.cli.commands.training_display import (
    TrainedModelInfo,
    TrainingResult,
    TrainingResultDisplay,
    display_batch_results,
    display_batch_training_plan,
    display_training_plan,
    display_training_results,
)


# ==================== Fixtures ====================


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_config() -> MagicMock:
    """Create mock configuration manager."""
    config = MagicMock()
    config.update_training_config = MagicMock()
    return config


@pytest.fixture
def temp_training_config(tmp_path: Path) -> Path:
    """Create a temporary training configuration file."""
    config = {
        "models": ["lgbm", "rf"],
        "areas": ["SECO", "S"],
        "training_periods": [
            {"start_date": "2023-01-01", "end_date": "2023-06-30"},
            {"start_date": "2023-07-01", "end_date": "2023-12-31"},
        ],
        "parallel_workers": 4,
        "force_retrain": False,
        "horizons": [0, 1, 2],
    }
    config_file = tmp_path / "training-config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def sample_training_result() -> TrainingResult:
    """Create a sample training result."""
    return TrainingResult(
        trained_models=[
            TrainedModelInfo(
                model_type="lgbm",
                area="SECO",
                mape=2.5,
                training_time=120.0,
            ),
            TrainedModelInfo(
                model_type="rf",
                area="S",
                mape=2.8,
                training_time=150.0,
            ),
        ],
        failed_models=[
            TrainedModelInfo(
                model_type="lgbm",
                area="N",
                error="Insufficient data",
            ),
        ],
        training_time=300.0,
        avg_time_per_model=150.0,
        success=False,
    )


# ==================== Validation Tests ====================


class TestValidateDateRange:
    """Tests for date range validation."""

    def test_valid_date_range(self) -> None:
        """Test valid date range passes."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        # Should not raise
        validate_date_range(start, end)

    def test_start_after_end_raises(self) -> None:
        """Test start after end raises exception."""
        start = datetime(2023, 12, 31)
        end = datetime(2023, 1, 1)
        with pytest.raises(Exception) as exc_info:
            validate_date_range(start, end)
        assert "must be before" in str(exc_info.value)

    def test_same_date_raises(self) -> None:
        """Test same start and end date raises exception."""
        date = datetime(2023, 6, 15)
        with pytest.raises(Exception) as exc_info:
            validate_date_range(date, date)
        assert "must be before" in str(exc_info.value)

    def test_insufficient_days_raises(self) -> None:
        """Test period less than 30 days raises exception."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 15)  # Only 14 days
        with pytest.raises(Exception) as exc_info:
            validate_date_range(start, end)
        assert "at least 30 days" in str(exc_info.value)

    def test_exactly_30_days_passes(self) -> None:
        """Test exactly 30 days passes."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)
        # Should not raise
        validate_date_range(start, end)


class TestValidateAreaSelection:
    """Tests for area selection validation."""

    def test_valid_areas(self) -> None:
        """Test valid areas pass."""
        validate_area_selection(["SECO", "S", "NE", "N"])

    def test_single_valid_area(self) -> None:
        """Test single valid area passes."""
        validate_area_selection(["SECO"])

    def test_all_valid_areas(self) -> None:
        """Test all valid areas pass."""
        validate_area_selection(VALID_AREAS)

    def test_invalid_area_raises(self) -> None:
        """Test invalid area raises exception."""
        with pytest.raises(Exception) as exc_info:
            validate_area_selection(["INVALID"])
        assert "Invalid areas" in str(exc_info.value)

    def test_mixed_valid_invalid_raises(self) -> None:
        """Test mix of valid and invalid areas raises exception."""
        with pytest.raises(Exception) as exc_info:
            validate_area_selection(["SECO", "INVALID", "S"])
        assert "Invalid areas" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)


class TestValidateModelSelection:
    """Tests for model selection validation."""

    def test_valid_models(self) -> None:
        """Test valid models pass."""
        validate_model_selection(["lgbm", "rf"])

    def test_single_valid_model(self) -> None:
        """Test single valid model passes."""
        validate_model_selection(["lgbm"])

    def test_all_valid_models(self) -> None:
        """Test all valid models pass."""
        validate_model_selection(VALID_MODEL_TYPES)

    def test_invalid_model_raises(self) -> None:
        """Test invalid model raises exception."""
        with pytest.raises(Exception) as exc_info:
            validate_model_selection(["invalid_model"])
        assert "Invalid model types" in str(exc_info.value)


# ==================== TrainingProgressBar Tests ====================


class TestTrainingProgressBar:
    """Tests for TrainingProgressBar."""

    def test_progress_bar_creation(self) -> None:
        """Test progress bar creation."""
        progress = TrainingProgressBar(
            models=["lgbm", "rf"],
            areas=["SECO", "S"],
        )
        assert progress.total_tasks == 4
        assert progress.completed_tasks == 0

    def test_progress_bar_update(self) -> None:
        """Test progress bar update."""
        progress = TrainingProgressBar(
            models=["lgbm"],
            areas=["SECO"],
        )
        progress.update(model="lgbm", area="SECO")
        assert progress.completed_tasks == 1

    def test_progress_bar_context_manager(self) -> None:
        """Test progress bar as context manager."""
        with TrainingProgressBar(
            models=["lgbm"],
            areas=["SECO"],
        ) as progress:
            progress.update()
            assert progress.completed_tasks == 1


# ==================== TrainingResultDisplay Tests ====================


class TestTrainingResultDisplay:
    """Tests for TrainingResultDisplay."""

    def test_display_creation(self) -> None:
        """Test display creation."""
        display = TrainingResultDisplay()
        assert display.use_colors is True
        assert display.verbose is False

    def test_display_no_colors(self) -> None:
        """Test display without colors."""
        display = TrainingResultDisplay(use_colors=False)
        assert display.use_colors is False

    def test_header_output(self, capsys) -> None:
        """Test header output."""
        display = TrainingResultDisplay(use_colors=False)
        display.header("Test Header")
        captured = capsys.readouterr()
        assert "Test Header" in captured.out

    def test_key_value_output(self, capsys) -> None:
        """Test key-value output."""
        display = TrainingResultDisplay(use_colors=False)
        display.key_value("Key:", "Value")
        captured = capsys.readouterr()
        assert "Key:" in captured.out
        assert "Value" in captured.out


# ==================== Display Function Tests ====================


class TestDisplayFunctions:
    """Tests for display functions."""

    def test_display_training_plan(self, capsys) -> None:
        """Test training plan display."""
        display_training_plan(
            models=["lgbm", "rf"],
            areas=["SECO", "S"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )
        captured = capsys.readouterr()
        assert "lgbm" in captured.out
        assert "rf" in captured.out
        assert "SECO" in captured.out
        assert "S" in captured.out

    def test_display_training_plan_with_horizons(self, capsys) -> None:
        """Test training plan display with horizons."""
        display_training_plan(
            models=["lgbm"],
            areas=["SECO"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            horizons=[0, 1, 2],
        )
        captured = capsys.readouterr()
        assert "D+0" in captured.out

    def test_display_training_results(self, sample_training_result: TrainingResult, capsys) -> None:
        """Test training results display."""
        display_training_results(sample_training_result)
        captured = capsys.readouterr()
        assert "Training Results" in captured.out
        assert "2" in captured.out  # 2 trained models
        assert "1" in captured.out  # 1 failed model

    def test_display_batch_training_plan(self, capsys) -> None:
        """Test batch training plan display."""
        config = {
            "models": ["lgbm"],
            "areas": ["SECO"],
            "training_periods": [{"start_date": "2023-01-01", "end_date": "2023-12-31"}],
            "parallel_workers": 4,
        }
        display_batch_training_plan(config)
        captured = capsys.readouterr()
        assert "Batch Training Configuration" in captured.out
        assert "lgbm" in captured.out

    def test_display_batch_results(self, sample_training_result: TrainingResult, capsys) -> None:
        """Test batch results display."""
        display_batch_results([sample_training_result])
        captured = capsys.readouterr()
        assert "Batch Training Results" in captured.out


# ==================== Configuration Validation Tests ====================


class TestValidateTrainingConfig:
    """Tests for training configuration validation."""

    def test_valid_config(self) -> None:
        """Test valid configuration passes."""
        config = {
            "models": ["lgbm"],
            "areas": ["SECO"],
            "training_periods": [{"start_date": "2023-01-01", "end_date": "2023-12-31"}],
        }
        _validate_training_config(config)

    def test_missing_models_raises(self) -> None:
        """Test missing models field raises exception."""
        config = {
            "areas": ["SECO"],
            "training_periods": [{"start_date": "2023-01-01", "end_date": "2023-12-31"}],
        }
        with pytest.raises(Exception) as exc_info:
            _validate_training_config(config)
        assert "models" in str(exc_info.value)

    def test_missing_areas_raises(self) -> None:
        """Test missing areas field raises exception."""
        config = {
            "models": ["lgbm"],
            "training_periods": [{"start_date": "2023-01-01", "end_date": "2023-12-31"}],
        }
        with pytest.raises(Exception) as exc_info:
            _validate_training_config(config)
        assert "areas" in str(exc_info.value)

    def test_missing_periods_raises(self) -> None:
        """Test missing training_periods field raises exception."""
        config = {
            "models": ["lgbm"],
            "areas": ["SECO"],
        }
        with pytest.raises(Exception) as exc_info:
            _validate_training_config(config)
        assert "training_periods" in str(exc_info.value)

    def test_empty_periods_raises(self) -> None:
        """Test empty training periods raises exception."""
        config = {
            "models": ["lgbm"],
            "areas": ["SECO"],
            "training_periods": [],
        }
        with pytest.raises(Exception) as exc_info:
            _validate_training_config(config)
        assert "At least one training period" in str(exc_info.value)

    def test_invalid_period_format_raises(self) -> None:
        """Test invalid period format raises exception."""
        config = {
            "models": ["lgbm"],
            "areas": ["SECO"],
            "training_periods": [{"start_date": "invalid-date", "end_date": "2023-12-31"}],
        }
        with pytest.raises(Exception) as exc_info:
            _validate_training_config(config)
        assert "Invalid date format" in str(exc_info.value)


# ==================== Command Tests ====================


class TestTrainModelCommand:
    """Tests for train model command."""

    def test_train_model_help(self, runner: CliRunner) -> None:
        """Test train model help."""
        result = runner.invoke(train, ["model", "--help"])
        assert result.exit_code == 0
        assert "Train individual or all models" in result.output

    def test_train_model_requires_model(self, runner: CliRunner) -> None:
        """Test train model requires model option."""
        result = runner.invoke(
            train,
            [
                "model",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
            ],
        )
        assert result.exit_code != 0
        assert "Missing option" in result.output or "--model" in result.output

    def test_train_model_requires_dates(self, runner: CliRunner) -> None:
        """Test train model requires dates."""
        result = runner.invoke(train, ["model", "--model", "lgbm"])
        assert result.exit_code != 0

    @patch("src.cli.commands.train.execute_training")
    def test_train_model_dry_run(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test train model dry run."""
        result = runner.invoke(
            train,
            [
                "model",
                "--model",
                "lgbm",
                "--area",
                "SECO",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--dry-run",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        mock_execute.assert_not_called()

    @patch("src.cli.commands.train.execute_training")
    def test_train_model_all(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test training all models."""
        mock_execute.return_value = TrainingResult(
            trained_models=[TrainedModelInfo(model_type="lgbm", area="SECO", mape=2.5)],
            success=True,
            training_time=60.0,
        )
        result = runner.invoke(
            train,
            [
                "model",
                "--model",
                "all",
                "--area",
                "SECO",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
            ],
            obj=mock_config,
        )
        # Will prompt for confirmation because >20 tasks
        assert "Training Plan" in result.output

    @patch("src.cli.commands.train.execute_training")
    def test_train_model_single_area(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test training single model for single area."""
        mock_execute.return_value = TrainingResult(
            trained_models=[TrainedModelInfo(model_type="lgbm", area="SECO", mape=2.5)],
            success=True,
            training_time=60.0,
            avg_time_per_model=60.0,
        )
        result = runner.invoke(
            train,
            [
                "model",
                "--model",
                "lgbm",
                "--area",
                "SECO",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "Training completed successfully" in result.output

    def test_train_model_invalid_area(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test training with invalid area."""
        result = runner.invoke(
            train,
            [
                "model",
                "--model",
                "lgbm",
                "--area",
                "INVALID",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
            ],
            obj=mock_config,
        )
        assert result.exit_code != 0
        assert "Invalid areas" in result.output

    def test_train_model_with_horizons(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test training with specific horizons."""
        result = runner.invoke(
            train,
            [
                "model",
                "--model",
                "lgbm",
                "--area",
                "SECO",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--horizons",
                "0,1,2",
                "--dry-run",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "D+0" in result.output


class TestTrainBatchCommand:
    """Tests for train batch command."""

    def test_train_batch_help(self, runner: CliRunner) -> None:
        """Test train batch help."""
        result = runner.invoke(train, ["batch", "--help"])
        assert result.exit_code == 0
        assert "batch training" in result.output.lower()

    def test_train_batch_requires_config(self, runner: CliRunner) -> None:
        """Test train batch requires config file."""
        result = runner.invoke(train, ["batch"])
        assert result.exit_code != 0

    def test_train_batch_config_not_found(self, runner: CliRunner) -> None:
        """Test train batch with nonexistent config file."""
        result = runner.invoke(train, ["batch", "--config-file", "/nonexistent/file.yaml"])
        assert result.exit_code != 0

    def test_train_batch_dry_run(
        self, runner: CliRunner, temp_training_config: Path, mock_config: MagicMock
    ) -> None:
        """Test train batch dry run."""
        result = runner.invoke(
            train,
            ["batch", "--config-file", str(temp_training_config), "--dry-run"],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "lgbm" in result.output
        assert "SECO" in result.output

    @patch("src.cli.commands.train.execute_training")
    def test_train_batch_execution(
        self,
        mock_execute: MagicMock,
        runner: CliRunner,
        temp_training_config: Path,
        mock_config: MagicMock,
    ) -> None:
        """Test train batch execution."""
        mock_execute.return_value = TrainingResult(
            trained_models=[TrainedModelInfo(model_type="lgbm", area="SECO", mape=2.5)],
            success=True,
            training_time=60.0,
        )
        result = runner.invoke(
            train,
            ["batch", "--config-file", str(temp_training_config)],
            obj=mock_config,
            input="y\n",  # Confirm execution
        )
        assert "Batch Training Configuration" in result.output


# ==================== TrainedModelInfo Tests ====================


class TestTrainedModelInfo:
    """Tests for TrainedModelInfo dataclass."""

    def test_creation(self) -> None:
        """Test model info creation."""
        info = TrainedModelInfo(
            model_type="lgbm",
            area="SECO",
            horizon=0,
            mape=2.5,
            training_time=120.0,
        )
        assert info.model_type == "lgbm"
        assert info.area == "SECO"
        assert info.horizon == 0
        assert info.mape == 2.5
        assert info.training_time == 120.0

    def test_defaults(self) -> None:
        """Test model info defaults."""
        info = TrainedModelInfo(model_type="lgbm", area="SECO")
        assert info.horizon == 0
        assert info.mape == 0.0
        assert info.training_time == 0.0
        assert info.error is None

    def test_failed_model(self) -> None:
        """Test failed model info."""
        info = TrainedModelInfo(model_type="lgbm", area="SECO", error="Training failed")
        assert info.error == "Training failed"


# ==================== TrainingResult Tests ====================


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_creation(self) -> None:
        """Test result creation."""
        result = TrainingResult(
            trained_models=[TrainedModelInfo(model_type="lgbm", area="SECO")],
            training_time=60.0,
            success=True,
        )
        assert len(result.trained_models) == 1
        assert result.training_time == 60.0
        assert result.success is True

    def test_defaults(self) -> None:
        """Test result defaults."""
        result = TrainingResult()
        assert result.trained_models == []
        assert result.failed_models == []
        assert result.training_time == 0.0
        assert result.success is True

    def test_failed_result(self) -> None:
        """Test failed result."""
        result = TrainingResult(
            failed_models=[TrainedModelInfo(model_type="lgbm", area="SECO", error="Failed")],
            success=False,
            error="Training failed",
        )
        assert len(result.failed_models) == 1
        assert result.success is False
        assert result.error == "Training failed"


# ==================== Integration Tests ====================


class TestTrainIntegration:
    """Integration tests for train commands."""

    def test_train_group_available(self, runner: CliRunner) -> None:
        """Test train group is available."""
        result = runner.invoke(train, ["--help"])
        assert result.exit_code == 0
        assert "model" in result.output
        assert "batch" in result.output

    def test_train_model_choices(self, runner: CliRunner) -> None:
        """Test train model choices."""
        result = runner.invoke(train, ["model", "--help"])
        assert "lgbm" in result.output
        assert "rf" in result.output
        assert "regdin_svm" in result.output
        assert "holt_winters" in result.output
        assert "all" in result.output

    @patch("src.cli.commands.train.execute_training")
    def test_full_training_flow(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test full training flow."""
        mock_execute.return_value = TrainingResult(
            trained_models=[
                TrainedModelInfo(model_type="lgbm", area="SECO", mape=2.5, training_time=60.0),
                TrainedModelInfo(model_type="lgbm", area="S", mape=2.8, training_time=70.0),
            ],
            success=True,
            training_time=130.0,
            avg_time_per_model=65.0,
        )

        result = runner.invoke(
            train,
            [
                "model",
                "--model",
                "lgbm",
                "--area",
                "SECO",
                "--area",
                "S",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--parallel",
                "2",
            ],
            obj=mock_config,
        )

        assert result.exit_code == 0
        assert "Training completed successfully" in result.output
        assert "SECO" in result.output


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_areas_uses_default(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test empty areas uses default subsystems."""
        result = runner.invoke(
            train,
            [
                "model",
                "--model",
                "lgbm",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--dry-run",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        # Default areas should be shown
        assert "SECO" in result.output or "Areas:" in result.output

    def test_parallel_workers_option(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test parallel workers option."""
        result = runner.invoke(
            train,
            [
                "model",
                "--model",
                "lgbm",
                "--area",
                "SECO",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--parallel",
                "8",
                "--dry-run",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "8" in result.output

    def test_force_flag(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test force retrain flag."""
        result = runner.invoke(
            train,
            [
                "model",
                "--model",
                "lgbm",
                "--area",
                "SECO",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--force",
                "--dry-run",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "Yes" in result.output  # Force retrain: Yes
