"""Tests for CLI core module.

This module tests the main CLI application and commands.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.core import (
    CLIContext,
    OutputFormat,
    OutputHelper,
    Verbosity,
    cli,
)


# ==================== Fixtures ====================


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_config() -> Path:
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
system:
  environment: development
  log_level: INFO

training:
  areas:
    - SECO
    - S
  horizons:
    - 0
    - 1
  model_types:
    - lgbm
  parallel_workers: 2

prediction:
  areas:
    - SECO
    - S
  horizons:
    - 0
    - 1

backtesting:
  step_days: 7
  areas:
    - SECO
""")
        return Path(f.name)


# ==================== OutputFormat Tests ====================


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_all_formats_exist(self) -> None:
        """Test all expected formats exist."""
        expected = ["TEXT", "JSON", "TABLE", "QUIET"]
        for fmt_name in expected:
            assert hasattr(OutputFormat, fmt_name)

    def test_format_values(self) -> None:
        """Test format value strings."""
        assert OutputFormat.TEXT.value == "text"
        assert OutputFormat.JSON.value == "json"


# ==================== Verbosity Tests ====================


class TestVerbosity:
    """Tests for Verbosity enum."""

    def test_all_levels_exist(self) -> None:
        """Test all expected levels exist."""
        expected = ["QUIET", "NORMAL", "VERBOSE", "DEBUG"]
        for level_name in expected:
            assert hasattr(Verbosity, level_name)

    def test_level_values(self) -> None:
        """Test verbosity level values."""
        assert Verbosity.QUIET.value == 0
        assert Verbosity.NORMAL.value == 1
        assert Verbosity.VERBOSE.value == 2
        assert Verbosity.DEBUG.value == 3


# ==================== CLIContext Tests ====================


class TestCLIContext:
    """Tests for CLIContext class."""

    def test_context_creation(self) -> None:
        """Test context creation."""
        ctx = CLIContext()

        assert ctx.config_manager is None
        assert ctx.verbosity == Verbosity.NORMAL
        assert ctx.output_format == OutputFormat.TEXT

    def test_load_config_from_file(self, temp_config: Path) -> None:
        """Test loading config from file."""
        ctx = CLIContext()
        ctx.load_config(temp_config)

        assert ctx.config_manager is not None

    def test_load_config_not_found(self) -> None:
        """Test loading nonexistent config."""
        ctx = CLIContext()
        ctx.load_config("nonexistent.yaml")

        # Should use defaults
        assert ctx.config_manager is not None

    def test_setup_logger(self) -> None:
        """Test logger setup."""
        ctx = CLIContext()
        logger = ctx.setup_logger("test_workflow")

        assert logger is not None
        assert ctx.logger is logger


# ==================== OutputHelper Tests ====================


class TestOutputHelper:
    """Tests for OutputHelper class."""

    def test_helper_creation(self) -> None:
        """Test helper creation."""
        helper = OutputHelper(fmt=OutputFormat.TEXT, quiet=False)

        assert helper.fmt == OutputFormat.TEXT
        assert helper.quiet is False

    def test_success_text(self, capsys) -> None:
        """Test success message in text format."""
        helper = OutputHelper(fmt=OutputFormat.TEXT)
        helper.success("Operation completed")

        captured = capsys.readouterr()
        assert "Operation completed" in captured.out

    def test_success_json(self, capsys) -> None:
        """Test success message in JSON format."""
        helper = OutputHelper(fmt=OutputFormat.JSON)
        helper.success("Operation completed")

        captured = capsys.readouterr()
        assert '"status": "success"' in captured.out

    def test_success_quiet(self, capsys) -> None:
        """Test success message when quiet."""
        helper = OutputHelper(fmt=OutputFormat.TEXT, quiet=True)
        helper.success("Operation completed")

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_error_text(self, capsys) -> None:
        """Test error message in text format."""
        helper = OutputHelper(fmt=OutputFormat.TEXT)
        helper.error("Something went wrong")

        captured = capsys.readouterr()
        assert "Something went wrong" in captured.err

    def test_info_text(self, capsys) -> None:
        """Test info message in text format."""
        helper = OutputHelper(fmt=OutputFormat.TEXT)
        helper.info("Information")

        captured = capsys.readouterr()
        assert "Information" in captured.out

    def test_warning_text(self, capsys) -> None:
        """Test warning message in text format."""
        helper = OutputHelper(fmt=OutputFormat.TEXT)
        helper.warning("Warning message")

        captured = capsys.readouterr()
        assert "Warning message" in captured.out

    def test_header_text(self, capsys) -> None:
        """Test header in text format."""
        helper = OutputHelper(fmt=OutputFormat.TEXT)
        helper.header("Section Title")

        captured = capsys.readouterr()
        assert "Section Title" in captured.out

    def test_table_text(self, capsys) -> None:
        """Test table in text format."""
        helper = OutputHelper(fmt=OutputFormat.TEXT)
        headers = ["Name", "Value"]
        rows = [["area", "SECO"], ["horizon", "0"]]
        helper.table(headers, rows)

        captured = capsys.readouterr()
        assert "Name" in captured.out
        assert "SECO" in captured.out

    def test_table_json(self, capsys) -> None:
        """Test table in JSON format."""
        helper = OutputHelper(fmt=OutputFormat.JSON)
        headers = ["name", "value"]
        rows = [["area", "SECO"]]
        helper.table(headers, rows)

        captured = capsys.readouterr()
        assert '"name": "area"' in captured.out

    def test_progress_text(self, capsys) -> None:
        """Test progress indicator."""
        helper = OutputHelper(fmt=OutputFormat.TEXT)
        helper.progress(5, 10, "Processing")

        captured = capsys.readouterr()
        assert "50" in captured.out  # 50%


# ==================== CLI Main Group Tests ====================


class TestCLIMainGroup:
    """Tests for main CLI group."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test CLI help."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "PrevCarga" in result.output
        assert "train" in result.output
        assert "predict" in result.output
        assert "backtest" in result.output

    def test_cli_version(self, runner: CliRunner) -> None:
        """Test CLI version."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_verbose_flag(self, runner: CliRunner) -> None:
        """Test verbose flags."""
        result = runner.invoke(cli, ["-v", "status"])

        # Should not fail
        assert result.exit_code == 0

    def test_cli_quiet_flag(self, runner: CliRunner) -> None:
        """Test quiet flag."""
        result = runner.invoke(cli, ["-q", "status"])

        # Should have minimal output
        assert result.exit_code == 0

    def test_cli_output_format(self, runner: CliRunner) -> None:
        """Test output format option."""
        result = runner.invoke(cli, ["-o", "json", "status"])

        assert result.exit_code == 0


# ==================== Train Command Tests ====================


class TestTrainCommand:
    """Tests for train command."""

    def test_train_help(self, runner: CliRunner) -> None:
        """Test train command help."""
        result = runner.invoke(cli, ["train", "--help"])

        assert result.exit_code == 0
        assert "Train forecasting models" in result.output
        assert "--areas" in result.output
        assert "--horizons" in result.output

    def test_train_dry_run(self, runner: CliRunner, temp_config: Path) -> None:
        """Test train dry run."""
        result = runner.invoke(cli, [
            "-c", str(temp_config),
            "train",
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_train_with_areas(self, runner: CliRunner, temp_config: Path) -> None:
        """Test train with specific areas."""
        result = runner.invoke(cli, [
            "-c", str(temp_config),
            "train",
            "--areas", "SECO,S",
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "SECO" in result.output


# ==================== Predict Command Tests ====================


class TestPredictCommand:
    """Tests for predict command."""

    def test_predict_help(self, runner: CliRunner) -> None:
        """Test predict command help."""
        result = runner.invoke(cli, ["predict", "--help"])

        assert result.exit_code == 0
        assert "Generate load forecasts" in result.output
        assert "--areas" in result.output
        assert "--format" in result.output

    @patch("src.cli.core.PredictionWorkflow")
    def test_predict_basic(
        self,
        mock_workflow_class: MagicMock,
        runner: CliRunner,
        temp_config: Path,
    ) -> None:
        """Test basic predict execution."""
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.duration_seconds = 1.5
        mock_result.forecasts = {"SECO": {0: MagicMock()}}
        mock_result.output_paths = {"forecasts": "/tmp/forecasts.parquet"}
        mock_workflow.run.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        result = runner.invoke(cli, [
            "-c", str(temp_config),
            "predict",
        ])

        assert result.exit_code == 0


# ==================== Backtest Command Tests ====================


class TestBacktestCommand:
    """Tests for backtest command."""

    def test_backtest_help(self, runner: CliRunner) -> None:
        """Test backtest command help."""
        result = runner.invoke(cli, ["backtest", "--help"])

        assert result.exit_code == 0
        assert "Run backtesting evaluation" in result.output
        assert "--start" in result.output
        assert "--end" in result.output

    @patch("src.cli.core.BacktestingWorkflow")
    def test_backtest_basic(
        self,
        mock_workflow_class: MagicMock,
        runner: CliRunner,
        temp_config: Path,
    ) -> None:
        """Test basic backtest execution."""
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.duration_seconds = 10.0
        mock_result.windows = [MagicMock()]
        mock_result.aggregate_metrics = {"mape": {"mean": 2.5, "std": 0.5}}
        mock_result.report_path = None
        mock_workflow.run.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        result = runner.invoke(cli, [
            "-c", str(temp_config),
            "backtest",
        ])

        assert result.exit_code == 0


# ==================== Status Command Tests ====================


class TestStatusCommand:
    """Tests for status command."""

    def test_status_help(self, runner: CliRunner) -> None:
        """Test status command help."""
        result = runner.invoke(cli, ["status", "--help"])

        assert result.exit_code == 0
        assert "Show system status" in result.output

    def test_status_basic(self, runner: CliRunner, temp_config: Path) -> None:
        """Test basic status output."""
        result = runner.invoke(cli, [
            "-c", str(temp_config),
            "status",
        ])

        assert result.exit_code == 0
        assert "Environment" in result.output

    def test_status_json(self, runner: CliRunner, temp_config: Path) -> None:
        """Test status in JSON format."""
        result = runner.invoke(cli, [
            "-c", str(temp_config),
            "-o", "json",
            "status",
        ])

        # Should contain JSON-formatted output
        assert result.exit_code == 0


# ==================== Config Commands Tests ====================


class TestConfigCommands:
    """Tests for config subcommands."""

    def test_config_help(self, runner: CliRunner) -> None:
        """Test config group help."""
        result = runner.invoke(cli, ["config", "--help"])

        assert result.exit_code == 0
        assert "Configuration management" in result.output

    def test_config_show(self, runner: CliRunner, temp_config: Path) -> None:
        """Test config show command."""
        result = runner.invoke(cli, [
            "-c", str(temp_config),
            "config", "show",
        ])

        assert result.exit_code == 0

    def test_config_show_section(self, runner: CliRunner, temp_config: Path) -> None:
        """Test config show with section."""
        result = runner.invoke(cli, [
            "-c", str(temp_config),
            "config", "show",
            "--section", "training",
        ])

        assert result.exit_code == 0
        assert "Training" in result.output or "areas" in result.output

    def test_config_validate(self, runner: CliRunner, temp_config: Path) -> None:
        """Test config validate command."""
        result = runner.invoke(cli, [
            "-c", str(temp_config),
            "config", "validate",
        ])

        assert result.exit_code == 0


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for CLI."""

    def test_workflow_with_all_options(
        self,
        runner: CliRunner,
        temp_config: Path,
    ) -> None:
        """Test command with all options."""
        result = runner.invoke(cli, [
            "-c", str(temp_config),
            "-v", "-v",  # Verbose
            "train",
            "--areas", "SECO",
            "--horizons", "0,1",
            "--parallel-workers", "2",
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "SECO" in result.output

    def test_multiple_commands_sequence(
        self,
        runner: CliRunner,
        temp_config: Path,
    ) -> None:
        """Test running multiple commands in sequence."""
        # First check status
        result1 = runner.invoke(cli, ["-c", str(temp_config), "status"])
        assert result1.exit_code == 0

        # Then validate config
        result2 = runner.invoke(cli, ["-c", str(temp_config), "config", "validate"])
        assert result2.exit_code == 0

        # Then dry-run train
        result3 = runner.invoke(cli, ["-c", str(temp_config), "train", "--dry-run"])
        assert result3.exit_code == 0
