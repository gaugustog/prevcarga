"""Tests for prediction CLI commands.

This module tests the predict command group including predict batch
and predict intraday subcommands.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from src.cli.commands.predict import (
    DEFAULT_AREAS,
    VALID_COMBINATIONS,
    VALID_MODELS,
    VALID_OUTPUT_FORMATS,
    VALID_RECONCILIATIONS,
    PredictionProgressBar,
    batch_predict,
    execute_batch_prediction,
    execute_intraday_prediction,
    intraday_predict,
    load_intraday_data,
    predict,
    validate_horizons,
    validate_output_path,
)
from src.cli.commands.prediction_display import (
    BatchPredictionResult,
    BLFWeights,
    ImprovementInfo,
    IntradayPredictionResult,
    PredictionResultDisplay,
    PredictionStatistics,
    UncertaintyInfo,
    display_intraday_summary,
    display_prediction_plan,
    display_prediction_summary,
    save_predictions,
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
    config.update_prediction_config = MagicMock()
    return config


@pytest.fixture
def sample_batch_result() -> BatchPredictionResult:
    """Create a sample batch prediction result."""
    return BatchPredictionResult(
        horizons=[0, 1, 2],
        areas=["SECO", "S", "NE", "N"],
        total_forecasts=576,  # 3 horizons * 4 areas * 48 periods
        execution_time=30.0,
        statistics=PredictionStatistics(
            mean_load=45000.0,
            max_load=65000.0,
            min_load=30000.0,
            std_load=8000.0,
        ),
        uncertainty=UncertaintyInfo(
            avg_pi_width=5000.0,
            coverage=95.0,
        ),
        success=True,
    )


@pytest.fixture
def sample_intraday_result() -> IntradayPredictionResult:
    """Create a sample intraday prediction result."""
    return IntradayPredictionResult(
        update_datetime=datetime(2024, 1, 15, 14, 30),
        areas=["SECO", "S", "NE", "N"],
        execution_time=5.0,
        blf_weights=BLFWeights(model=0.7, observed=0.3),
        improvement=ImprovementInfo(mape_reduction=0.5, rmse_reduction=150.0),
        success=True,
    )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    return tmp_path


@pytest.fixture
def temp_data_file(tmp_path: Path) -> Path:
    """Create a temporary intraday data file."""
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-15 12:00", periods=4, freq="30min"),
            "area": ["SECO", "S", "NE", "N"],
            "load": [50000, 30000, 20000, 10000],
        }
    )
    data_file = tmp_path / "intraday_data.csv"
    data.to_csv(data_file, index=False)
    return data_file


# ==================== Validation Tests ====================


class TestValidateHorizons:
    """Tests for horizon validation."""

    def test_validate_all_horizons_default(self) -> None:
        """Test default returns all horizons."""
        horizons = validate_horizons(())
        assert horizons == list(range(9))

    def test_validate_specific_horizons(self) -> None:
        """Test specific horizons."""
        horizons = validate_horizons((0, 2, 4))
        assert horizons == [0, 2, 4]

    def test_validate_single_horizon(self) -> None:
        """Test single horizon."""
        horizons = validate_horizons((3,))
        assert horizons == [3]

    def test_validate_duplicate_horizons_deduplicated(self) -> None:
        """Test duplicates are removed."""
        horizons = validate_horizons((1, 1, 2, 2))
        assert horizons == [1, 2]

    def test_validate_horizons_sorted(self) -> None:
        """Test horizons are sorted."""
        horizons = validate_horizons((5, 2, 8, 1))
        assert horizons == [1, 2, 5, 8]


class TestValidateOutputPath:
    """Tests for output path validation."""

    def test_valid_output_path(self, temp_output_dir: Path) -> None:
        """Test valid output path."""
        output = temp_output_dir / "predictions.csv"
        # Should not raise
        validate_output_path(output, "csv")

    def test_nonexistent_directory_raises(self) -> None:
        """Test nonexistent directory raises exception."""
        output = Path("/nonexistent/dir/predictions.csv")
        with pytest.raises(Exception) as exc_info:
            validate_output_path(output, "csv")
        assert "does not exist" in str(exc_info.value)

    def test_extension_mismatch_warning(self, temp_output_dir: Path, capsys) -> None:
        """Test mismatched extension shows warning."""
        output = temp_output_dir / "predictions.json"
        validate_output_path(output, "csv")  # Format doesn't match extension
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# ==================== PredictionProgressBar Tests ====================


class TestPredictionProgressBar:
    """Tests for PredictionProgressBar."""

    def test_progress_bar_creation(self) -> None:
        """Test progress bar creation."""
        progress = PredictionProgressBar(
            horizons=[0, 1, 2],
            areas=["SECO", "S"],
        )
        assert progress.total_tasks == 6  # 3 horizons * 2 areas
        assert progress.completed_tasks == 0

    def test_progress_bar_update(self) -> None:
        """Test progress bar update."""
        progress = PredictionProgressBar(
            horizons=[0],
            areas=["SECO"],
        )
        progress.update(horizon=0, area="SECO")
        assert progress.completed_tasks == 1

    def test_progress_bar_context_manager(self) -> None:
        """Test progress bar as context manager."""
        with PredictionProgressBar(
            horizons=[0],
            areas=["SECO"],
        ) as progress:
            progress.update()
            assert progress.completed_tasks == 1


# ==================== PredictionResultDisplay Tests ====================


class TestPredictionResultDisplay:
    """Tests for PredictionResultDisplay."""

    def test_display_creation(self) -> None:
        """Test display creation."""
        display = PredictionResultDisplay()
        assert display.use_colors is True
        assert display.verbose is False

    def test_display_no_colors(self) -> None:
        """Test display without colors."""
        display = PredictionResultDisplay(use_colors=False)
        assert display.use_colors is False

    def test_header_output(self, capsys) -> None:
        """Test header output."""
        display = PredictionResultDisplay(use_colors=False)
        display.header("Test Header")
        captured = capsys.readouterr()
        assert "Test Header" in captured.out

    def test_key_value_output(self, capsys) -> None:
        """Test key-value output."""
        display = PredictionResultDisplay(use_colors=False)
        display.key_value("Key:", "Value")
        captured = capsys.readouterr()
        assert "Key:" in captured.out
        assert "Value" in captured.out


# ==================== Display Function Tests ====================


class TestDisplayFunctions:
    """Tests for display functions."""

    def test_display_prediction_plan(self, capsys) -> None:
        """Test prediction plan display."""
        display_prediction_plan(
            date=datetime(2024, 1, 15),
            horizons=[0, 1, 2],
            areas=["SECO", "S"],
            models=["lgbm", "rf"],
            combination="weighted_avg",
            reconciliation="mint",
            output_format="csv",
        )
        captured = capsys.readouterr()
        assert "Prediction Plan" in captured.out
        assert "2024-01-15" in captured.out
        assert "D+0 to D+2" in captured.out

    def test_display_prediction_summary(
        self, sample_batch_result: BatchPredictionResult, capsys
    ) -> None:
        """Test prediction summary display."""
        display_prediction_summary(sample_batch_result)
        captured = capsys.readouterr()
        assert "Batch Prediction Results" in captured.out
        assert "576" in captured.out  # total forecasts

    def test_display_prediction_summary_with_statistics(
        self, sample_batch_result: BatchPredictionResult, capsys
    ) -> None:
        """Test prediction summary with statistics."""
        display_prediction_summary(sample_batch_result)
        captured = capsys.readouterr()
        assert "Forecast Statistics" in captured.out
        assert "Mean load" in captured.out

    def test_display_prediction_summary_with_uncertainty(
        self, sample_batch_result: BatchPredictionResult, capsys
    ) -> None:
        """Test prediction summary with uncertainty."""
        display_prediction_summary(sample_batch_result)
        captured = capsys.readouterr()
        assert "Uncertainty" in captured.out

    def test_display_intraday_summary(
        self, sample_intraday_result: IntradayPredictionResult, capsys
    ) -> None:
        """Test intraday summary display."""
        display_intraday_summary(sample_intraday_result)
        captured = capsys.readouterr()
        assert "Intraday Prediction Results" in captured.out
        assert "LGBM BLF" in captured.out

    def test_display_intraday_summary_with_blf_weights(
        self, sample_intraday_result: IntradayPredictionResult, capsys
    ) -> None:
        """Test intraday summary with BLF weights."""
        display_intraday_summary(sample_intraday_result)
        captured = capsys.readouterr()
        assert "BLF Blending Weights" in captured.out


# ==================== Save Predictions Tests ====================


class TestSavePredictions:
    """Tests for save_predictions function."""

    def test_save_csv(
        self, sample_batch_result: BatchPredictionResult, temp_output_dir: Path
    ) -> None:
        """Test saving as CSV."""
        output = temp_output_dir / "predictions.csv"
        save_predictions(sample_batch_result, output, "csv")
        assert output.exists()

    def test_save_json(
        self, sample_batch_result: BatchPredictionResult, temp_output_dir: Path
    ) -> None:
        """Test saving as JSON."""
        output = temp_output_dir / "predictions.json"
        save_predictions(sample_batch_result, output, "json")
        assert output.exists()

    def test_save_parquet(
        self, sample_batch_result: BatchPredictionResult, temp_output_dir: Path
    ) -> None:
        """Test saving as Parquet."""
        output = temp_output_dir / "predictions.parquet"
        save_predictions(sample_batch_result, output, "parquet")
        assert output.exists()

    def test_save_invalid_format_raises(
        self, sample_batch_result: BatchPredictionResult, temp_output_dir: Path
    ) -> None:
        """Test invalid format raises exception."""
        output = temp_output_dir / "predictions.txt"
        with pytest.raises(ValueError) as exc_info:
            save_predictions(sample_batch_result, output, "txt")
        assert "Unsupported output format" in str(exc_info.value)


# ==================== Load Intraday Data Tests ====================


class TestLoadIntradayData:
    """Tests for load_intraday_data function."""

    def test_load_valid_data(self, temp_data_file: Path) -> None:
        """Test loading valid data file."""
        data = load_intraday_data(temp_data_file)
        assert len(data) == 4
        assert "timestamp" in data.columns
        assert "area" in data.columns
        assert "load" in data.columns

    def test_load_missing_columns_raises(self, tmp_path: Path) -> None:
        """Test missing columns raises exception."""
        data = pd.DataFrame({"timestamp": [datetime.now()], "area": ["SECO"]})
        data_file = tmp_path / "bad_data.csv"
        data.to_csv(data_file, index=False)

        with pytest.raises(Exception) as exc_info:
            load_intraday_data(data_file)
        assert "Missing required columns" in str(exc_info.value)


# ==================== Command Tests ====================


class TestBatchPredictCommand:
    """Tests for predict batch command."""

    def test_batch_predict_help(self, runner: CliRunner) -> None:
        """Test predict batch help."""
        result = runner.invoke(predict, ["batch", "--help"])
        assert result.exit_code == 0
        assert "batch predictions" in result.output.lower()

    def test_batch_predict_requires_date(self, runner: CliRunner) -> None:
        """Test predict batch requires date option."""
        result = runner.invoke(predict, ["batch"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "--date" in result.output

    @patch("src.cli.commands.predict.execute_batch_prediction")
    def test_batch_predict_dry_run(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test predict batch dry run."""
        result = runner.invoke(
            predict,
            [
                "batch",
                "--date",
                "2024-01-15",
                "--dry-run",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        mock_execute.assert_not_called()

    @patch("src.cli.commands.predict.execute_batch_prediction")
    def test_batch_predict_specific_horizons(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test predict batch with specific horizons."""
        mock_execute.return_value = BatchPredictionResult(
            horizons=[0, 1],
            areas=DEFAULT_AREAS,
            total_forecasts=384,
            execution_time=10.0,
            success=True,
        )
        result = runner.invoke(
            predict,
            [
                "batch",
                "--date",
                "2024-01-15",
                "--horizon",
                "0",
                "--horizon",
                "1",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "D+0 to D+1" in result.output

    @patch("src.cli.commands.predict.execute_batch_prediction")
    def test_batch_predict_with_combination(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test predict batch with combination method."""
        mock_execute.return_value = BatchPredictionResult(
            horizons=list(range(9)),
            areas=DEFAULT_AREAS,
            total_forecasts=1728,
            execution_time=30.0,
            success=True,
        )
        result = runner.invoke(
            predict,
            [
                "batch",
                "--date",
                "2024-01-15",
                "--combination",
                "stacking",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "stacking" in result.output

    @patch("src.cli.commands.predict.execute_batch_prediction")
    def test_batch_predict_with_reconciliation(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test predict batch with reconciliation method."""
        mock_execute.return_value = BatchPredictionResult(
            horizons=list(range(9)),
            areas=DEFAULT_AREAS,
            total_forecasts=1728,
            execution_time=30.0,
            success=True,
        )
        result = runner.invoke(
            predict,
            [
                "batch",
                "--date",
                "2024-01-15",
                "--reconciliation",
                "wls",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "wls" in result.output


class TestIntradayPredictCommand:
    """Tests for predict intraday command."""

    def test_intraday_predict_help(self, runner: CliRunner) -> None:
        """Test predict intraday help."""
        result = runner.invoke(predict, ["intraday", "--help"])
        assert result.exit_code == 0
        assert "intraday" in result.output.lower()

    def test_intraday_predict_requires_datetime(self, runner: CliRunner) -> None:
        """Test predict intraday requires datetime option."""
        result = runner.invoke(predict, ["intraday"])
        assert result.exit_code != 0

    @patch("src.cli.commands.predict.execute_intraday_prediction")
    def test_intraday_predict_dry_run(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test predict intraday dry run."""
        result = runner.invoke(
            predict,
            [
                "intraday",
                "--datetime",
                "2024-01-15 14:30",
                "--dry-run",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        mock_execute.assert_not_called()

    @patch("src.cli.commands.predict.execute_intraday_prediction")
    def test_intraday_predict_basic(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test basic intraday prediction."""
        mock_execute.return_value = IntradayPredictionResult(
            update_datetime=datetime(2024, 1, 15, 14, 30),
            areas=DEFAULT_AREAS,
            execution_time=5.0,
            success=True,
        )
        result = runner.invoke(
            predict,
            [
                "intraday",
                "--datetime",
                "2024-01-15 14:30",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "LGBM BLF" in result.output

    @patch("src.cli.commands.predict.execute_intraday_prediction")
    def test_intraday_predict_with_data_file(
        self,
        mock_execute: MagicMock,
        runner: CliRunner,
        mock_config: MagicMock,
        temp_data_file: Path,
    ) -> None:
        """Test intraday prediction with data file."""
        mock_execute.return_value = IntradayPredictionResult(
            update_datetime=datetime(2024, 1, 15, 14, 30),
            areas=DEFAULT_AREAS,
            execution_time=5.0,
            success=True,
        )
        result = runner.invoke(
            predict,
            [
                "intraday",
                "--datetime",
                "2024-01-15 14:30",
                "--data-file",
                str(temp_data_file),
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "Loaded updated data" in result.output


# ==================== Data Classes Tests ====================


class TestBatchPredictionResult:
    """Tests for BatchPredictionResult dataclass."""

    def test_creation(self) -> None:
        """Test result creation."""
        result = BatchPredictionResult(
            horizons=[0, 1, 2],
            areas=["SECO", "S"],
            total_forecasts=288,
            execution_time=15.0,
            success=True,
        )
        assert result.horizons == [0, 1, 2]
        assert len(result.areas) == 2
        assert result.success is True

    def test_defaults(self) -> None:
        """Test result defaults."""
        result = BatchPredictionResult()
        assert result.horizons == []
        assert result.areas == []
        assert result.total_forecasts == 0
        assert result.success is True

    def test_to_dataframe(self) -> None:
        """Test conversion to DataFrame."""
        result = BatchPredictionResult()
        df = result.to_dataframe()
        assert "timestamp" in df.columns
        assert "forecast" in df.columns


class TestIntradayPredictionResult:
    """Tests for IntradayPredictionResult dataclass."""

    def test_creation(self) -> None:
        """Test result creation."""
        result = IntradayPredictionResult(
            update_datetime=datetime(2024, 1, 15, 14, 30),
            areas=["SECO"],
            execution_time=5.0,
            success=True,
        )
        assert result.update_datetime == datetime(2024, 1, 15, 14, 30)
        assert result.success is True

    def test_defaults(self) -> None:
        """Test result defaults."""
        result = IntradayPredictionResult()
        assert result.update_datetime is None
        assert result.areas == []
        assert result.success is True


class TestPredictionStatistics:
    """Tests for PredictionStatistics dataclass."""

    def test_creation(self) -> None:
        """Test statistics creation."""
        stats = PredictionStatistics(
            mean_load=45000.0,
            max_load=65000.0,
            min_load=30000.0,
            std_load=8000.0,
        )
        assert stats.mean_load == 45000.0
        assert stats.max_load == 65000.0


class TestBLFWeights:
    """Tests for BLFWeights dataclass."""

    def test_creation(self) -> None:
        """Test weights creation."""
        weights = BLFWeights(model=0.7, observed=0.3)
        assert weights.model == 0.7
        assert weights.observed == 0.3

    def test_defaults(self) -> None:
        """Test weight defaults."""
        weights = BLFWeights()
        assert weights.model == 0.5
        assert weights.observed == 0.5


# ==================== Integration Tests ====================


class TestPredictIntegration:
    """Integration tests for predict commands."""

    def test_predict_group_available(self, runner: CliRunner) -> None:
        """Test predict group is available."""
        result = runner.invoke(predict, ["--help"])
        assert result.exit_code == 0
        assert "batch" in result.output
        assert "intraday" in result.output

    def test_predict_batch_choices(self, runner: CliRunner) -> None:
        """Test predict batch choices."""
        result = runner.invoke(predict, ["batch", "--help"])
        assert "lgbm" in result.output
        assert "rf" in result.output
        assert "simple_avg" in result.output
        assert "mint" in result.output

    @patch("src.cli.commands.predict.execute_batch_prediction")
    def test_full_batch_prediction_flow(
        self, mock_execute: MagicMock, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test full batch prediction flow."""
        mock_execute.return_value = BatchPredictionResult(
            horizons=[0, 1, 2],
            areas=DEFAULT_AREAS,
            total_forecasts=576,
            execution_time=15.0,
            statistics=PredictionStatistics(
                mean_load=45000.0,
                max_load=65000.0,
                min_load=30000.0,
            ),
            success=True,
        )

        result = runner.invoke(
            predict,
            [
                "batch",
                "--date",
                "2024-01-15",
                "--horizon",
                "0",
                "--horizon",
                "1",
                "--horizon",
                "2",
                "--combination",
                "weighted_avg",
                "--reconciliation",
                "mint",
            ],
            obj=mock_config,
        )

        assert result.exit_code == 0
        assert "Batch prediction completed successfully" in result.output


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_horizons_uses_default(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test empty horizons uses all horizons."""
        result = runner.invoke(
            predict,
            [
                "batch",
                "--date",
                "2024-01-15",
                "--dry-run",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0
        assert "D+0 to D+8" in result.output

    def test_include_uncertainty_flag(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test include uncertainty flag."""
        result = runner.invoke(
            predict,
            [
                "batch",
                "--date",
                "2024-01-15",
                "--include-uncertainty",
                "--dry-run",
            ],
            obj=mock_config,
        )
        assert result.exit_code == 0

    def test_output_format_options(self, runner: CliRunner) -> None:
        """Test all output format options are available."""
        result = runner.invoke(predict, ["batch", "--help"])
        for fmt in VALID_OUTPUT_FORMATS:
            assert fmt in result.output

    def test_combination_options(self, runner: CliRunner) -> None:
        """Test all combination options are available."""
        result = runner.invoke(predict, ["batch", "--help"])
        for combo in VALID_COMBINATIONS:
            assert combo in result.output

    def test_reconciliation_options(self, runner: CliRunner) -> None:
        """Test all reconciliation options are available."""
        result = runner.invoke(predict, ["batch", "--help"])
        for recon in VALID_RECONCILIATIONS:
            assert recon in result.output
