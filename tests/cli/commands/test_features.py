"""Tests for feature engineering CLI commands.

This module contains comprehensive tests for the features CLI commands
including generation, evaluation, and validation functionality.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pandas as pd
import pytest
from click.testing import CliRunner

from src.cli.commands.feature_display import (
    FeatureEvaluationResult,
    FeatureGenerationResult,
    FeatureProgressBar,
    FeatureResultDisplay,
    FeatureValidationResult,
    display_evaluation_results,
    display_generation_plan,
    display_generation_summary,
    display_validation_results,
    save_evaluation_report,
)
from src.cli.commands.features import (
    VALID_AREAS,
    VALID_EVAL_MODELS,
    VALID_OUTPUT_FORMATS,
    VALID_PLUGINS,
    evaluate_features,
    execute_feature_evaluation,
    execute_feature_generation,
    execute_feature_validation,
    features,
    generate_features,
    list_plugins,
    validate_areas,
    validate_date_range,
    validate_output_format,
    validate_plugins,
    validate_features,
)


class TestValidationFunctions:
    """Tests for input validation functions."""

    def test_validate_plugins_empty_returns_all(self) -> None:
        """Test that empty plugins returns all valid plugins."""
        result = validate_plugins(())
        assert result == list(VALID_PLUGINS)

    def test_validate_plugins_valid(self) -> None:
        """Test validation of valid plugins."""
        result = validate_plugins(("temporal", "calendar"))
        assert result == ["temporal", "calendar"]

    def test_validate_plugins_invalid_raises(self) -> None:
        """Test that invalid plugins raise BadParameter."""
        with pytest.raises(click.BadParameter) as exc_info:
            validate_plugins(("invalid_plugin",))
        assert "Invalid plugin" in str(exc_info.value)

    def test_validate_plugins_partial_invalid_raises(self) -> None:
        """Test that partially invalid plugins raise BadParameter."""
        with pytest.raises(click.BadParameter):
            validate_plugins(("temporal", "invalid"))

    def test_validate_areas_empty_returns_nacional(self) -> None:
        """Test that empty areas returns nacional."""
        result = validate_areas(())
        assert result == ["nacional"]

    def test_validate_areas_valid(self) -> None:
        """Test validation of valid areas."""
        result = validate_areas(("SE", "S"))
        assert result == ["SE", "S"]

    def test_validate_areas_invalid_raises(self) -> None:
        """Test that invalid areas raise BadParameter."""
        with pytest.raises(click.BadParameter) as exc_info:
            validate_areas(("invalid_area",))
        assert "Invalid area" in str(exc_info.value)

    def test_validate_date_range_valid(self) -> None:
        """Test validation of valid date range."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        # Should not raise
        validate_date_range(start, end)

    def test_validate_date_range_start_after_end_raises(self) -> None:
        """Test that start after end raises BadParameter."""
        start = datetime(2023, 12, 31)
        end = datetime(2023, 1, 1)
        with pytest.raises(click.BadParameter) as exc_info:
            validate_date_range(start, end)
        assert "must be before" in str(exc_info.value)

    def test_validate_date_range_same_dates_raises(self) -> None:
        """Test that same dates raise BadParameter."""
        date = datetime(2023, 6, 15)
        with pytest.raises(click.BadParameter):
            validate_date_range(date, date)

    def test_validate_date_range_too_large_raises(self) -> None:
        """Test that too large date range raises BadParameter."""
        start = datetime(2018, 1, 1)
        end = datetime(2024, 1, 1)  # > 5 years
        with pytest.raises(click.BadParameter) as exc_info:
            validate_date_range(start, end)
        assert "too large" in str(exc_info.value)

    def test_validate_output_format_valid(self) -> None:
        """Test validation of valid output formats."""
        assert validate_output_format("parquet") == "parquet"
        assert validate_output_format("csv") == "csv"
        assert validate_output_format("feather") == "feather"

    def test_validate_output_format_invalid_raises(self) -> None:
        """Test that invalid format raises BadParameter."""
        with pytest.raises(click.BadParameter) as exc_info:
            validate_output_format("invalid")
        assert "Invalid format" in str(exc_info.value)


class TestFeatureProgressBar:
    """Tests for FeatureProgressBar class."""

    def test_progress_bar_initialization(self) -> None:
        """Test progress bar initialization."""
        bar = FeatureProgressBar(
            plugins=["temporal", "lag"],
            areas=["SE", "S"],
            show_progress=True,
        )
        assert bar.plugins == ["temporal", "lag"]
        assert bar.areas == ["SE", "S"]
        assert bar.total_steps == 4
        assert bar.current_step == 0

    def test_progress_bar_context_manager(self) -> None:
        """Test progress bar as context manager."""
        bar = FeatureProgressBar(
            plugins=["temporal"],
            areas=["SE"],
            show_progress=False,
        )
        with bar as pb:
            assert pb is bar

    def test_progress_bar_update(self) -> None:
        """Test progress bar update."""
        bar = FeatureProgressBar(
            plugins=["temporal"],
            areas=["SE"],
            show_progress=False,
        )
        bar.update("temporal", "SE", "processing")
        assert bar.current_step == 1

    def test_progress_bar_no_steps(self) -> None:
        """Test progress bar with no steps."""
        bar = FeatureProgressBar(plugins=[], areas=[], show_progress=True)
        assert bar.total_steps == 0


class TestFeatureResultDisplay:
    """Tests for FeatureResultDisplay class."""

    def test_format_importance_high(self) -> None:
        """Test formatting high importance values."""
        result = FeatureResultDisplay.format_importance(0.5)
        assert "0.5000" in result

    def test_format_importance_medium(self) -> None:
        """Test formatting medium importance values."""
        result = FeatureResultDisplay.format_importance(0.05)
        assert "0.0500" in result

    def test_format_importance_low(self) -> None:
        """Test formatting low importance values."""
        result = FeatureResultDisplay.format_importance(0.005)
        assert "0.0050" in result

    def test_format_validation_status_pass(self) -> None:
        """Test formatting pass status."""
        result = FeatureResultDisplay.format_validation_status(True)
        assert "PASS" in result

    def test_format_validation_status_fail(self) -> None:
        """Test formatting fail status."""
        result = FeatureResultDisplay.format_validation_status(False)
        assert "FAIL" in result


class TestFeatureGenerationResult:
    """Tests for FeatureGenerationResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = FeatureGenerationResult()
        assert result.plugins_used == []
        assert result.areas_processed == []
        assert result.features_generated == 0
        assert result.success is True
        assert result.error_message is None

    def test_with_values(self) -> None:
        """Test with custom values."""
        result = FeatureGenerationResult(
            plugins_used=["temporal", "lag"],
            areas_processed=["SE"],
            features_generated=50,
            rows_processed=1000,
            output_path=Path("features/"),
            execution_time_seconds=5.0,
        )
        assert result.plugins_used == ["temporal", "lag"]
        assert result.areas_processed == ["SE"]
        assert result.features_generated == 50
        assert result.rows_processed == 1000


class TestFeatureEvaluationResult:
    """Tests for FeatureEvaluationResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = FeatureEvaluationResult()
        assert result.model_type == "lgbm"
        assert result.area == "nacional"
        assert result.total_features == 0
        assert result.top_features == []
        assert result.success is True

    def test_with_top_features(self) -> None:
        """Test with top features."""
        result = FeatureEvaluationResult(
            top_features=[
                ("lag_24", 0.5),
                ("hour", 0.3),
                ("day_of_week", 0.2),
            ]
        )
        assert len(result.top_features) == 3
        assert result.top_features[0][0] == "lag_24"


class TestFeatureValidationResult:
    """Tests for FeatureValidationResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = FeatureValidationResult()
        assert result.is_valid is True
        assert result.total_files == 0
        assert result.valid_files == 0
        assert result.issues == []

    def test_with_issues(self) -> None:
        """Test with validation issues."""
        result = FeatureValidationResult(
            is_valid=False,
            issues=[
                {"severity": "error", "message": "Missing column"},
                {"severity": "warning", "message": "High null percentage"},
            ],
        )
        assert not result.is_valid
        assert len(result.issues) == 2


class TestDisplayFunctions:
    """Tests for display functions."""

    def test_display_generation_plan(self, capsys: pytest.CaptureFixture) -> None:
        """Test generation plan display."""
        display_generation_plan(
            plugins=["temporal", "lag"],
            areas=["SE", "S"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            output_dir=Path("features/"),
            output_format="parquet",
            parallel_workers=4,
        )
        captured = capsys.readouterr()
        assert "Feature Generation Plan" in captured.out
        assert "temporal" in captured.out
        assert "parquet" in captured.out

    def test_display_generation_summary_success(self, capsys: pytest.CaptureFixture) -> None:
        """Test generation summary display for success."""
        result = FeatureGenerationResult(
            plugins_used=["temporal"],
            areas_processed=["SE"],
            features_generated=10,
            rows_processed=1000,
            execution_time_seconds=5.0,
            success=True,
        )
        display_generation_summary(result)
        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out
        assert "10" in captured.out

    def test_display_generation_summary_failure(self, capsys: pytest.CaptureFixture) -> None:
        """Test generation summary display for failure."""
        result = FeatureGenerationResult(
            success=False,
            error_message="Test error",
        )
        display_generation_summary(result)
        captured = capsys.readouterr()
        assert "FAILED" in captured.out
        assert "Test error" in captured.out

    def test_display_evaluation_results(self, capsys: pytest.CaptureFixture) -> None:
        """Test evaluation results display."""
        result = FeatureEvaluationResult(
            model_type="lgbm",
            area="SE",
            total_features=50,
            top_features=[
                ("lag_24", 0.5),
                ("hour", 0.3),
            ],
            execution_time_seconds=2.0,
        )
        display_evaluation_results(result, top_n=10)
        captured = capsys.readouterr()
        assert "Feature Evaluation Results" in captured.out
        assert "lgbm" in captured.out
        assert "lag_24" in captured.out

    def test_display_validation_results_pass(self, capsys: pytest.CaptureFixture) -> None:
        """Test validation results display for pass."""
        result = FeatureValidationResult(
            feature_dir=Path("features/"),
            is_valid=True,
            total_files=5,
            valid_files=5,
        )
        display_validation_results(result)
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_display_validation_results_fail(self, capsys: pytest.CaptureFixture) -> None:
        """Test validation results display for fail."""
        result = FeatureValidationResult(
            feature_dir=Path("features/"),
            is_valid=False,
            total_files=5,
            valid_files=3,
            issues=[
                {"severity": "error", "message": "Missing column", "file": "test.parquet"},
            ],
        )
        display_validation_results(result)
        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "Missing column" in captured.out


class TestSaveEvaluationReport:
    """Tests for save_evaluation_report function."""

    def test_save_json_report(self, tmp_path: Path) -> None:
        """Test saving JSON report."""
        result = FeatureEvaluationResult(
            model_type="lgbm",
            area="SE",
            total_features=10,
            top_features=[("feature1", 0.5), ("feature2", 0.3)],
        )
        output_path = tmp_path / "report.json"
        save_evaluation_report(result, output_path, format="json")

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["model_type"] == "lgbm"
        assert len(data["top_features"]) == 2

    def test_save_csv_report(self, tmp_path: Path) -> None:
        """Test saving CSV report."""
        result = FeatureEvaluationResult(
            top_features=[("feature1", 0.5), ("feature2", 0.3)],
        )
        output_path = tmp_path / "report.csv"
        save_evaluation_report(result, output_path, format="csv")

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert "feature_name" in df.columns

    def test_save_html_report(self, tmp_path: Path) -> None:
        """Test saving HTML report."""
        result = FeatureEvaluationResult(
            model_type="rf",
            top_features=[("feature1", 0.5)],
        )
        output_path = tmp_path / "report.html"
        save_evaluation_report(result, output_path, format="html")

        assert output_path.exists()
        content = output_path.read_text()
        assert "<html>" in content
        assert "rf" in content


class TestExecuteFeatureGeneration:
    """Tests for execute_feature_generation function."""

    def test_dry_run(self) -> None:
        """Test dry run mode."""
        result = execute_feature_generation(
            plugins=["temporal", "lag"],
            areas=["SE"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            output_dir=Path("features/"),
            output_format="parquet",
            parallel_workers=4,
            dry_run=True,
        )
        assert result.success
        assert len(result.plugins_used) == 2
        assert result.features_generated > 0

    @patch("src.features.PluginRegistry")
    @patch("src.features.FeaturePipeline")
    def test_with_mocked_pipeline(
        self,
        mock_pipeline_class: MagicMock,
        mock_registry_class: MagicMock,
    ) -> None:
        """Test with mocked pipeline."""
        # Setup mock registry
        mock_registry = MagicMock()
        mock_plugin = MagicMock()
        mock_plugin.name = "temporal_features"
        mock_plugin.get_feature_names.return_value = ["hour", "day"]
        mock_registry.get_plugin.return_value = mock_plugin
        mock_registry_class.return_value = mock_registry

        # Setup mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.get_all_feature_names.return_value = ["hour", "day"]
        mock_pipeline_class.return_value = mock_pipeline

        result = execute_feature_generation(
            plugins=["temporal"],
            areas=["SE"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            output_dir=Path("features/"),
            output_format="parquet",
            parallel_workers=2,
            dry_run=False,
        )

        assert result.success


class TestExecuteFeatureEvaluation:
    """Tests for execute_feature_evaluation function."""

    @patch("src.features.performance.importance_tracker.FeatureImportanceTracker")
    def test_with_tracker(self, mock_tracker_class: MagicMock) -> None:
        """Test evaluation with importance tracker."""
        mock_tracker = MagicMock()
        mock_tracker.get_aggregated_importance.return_value = pd.DataFrame({
            "feature_name": ["feature1", "feature2"],
            "aggregated_importance": [0.5, 0.3],
        })
        mock_tracker.get_feature_stability.return_value = pd.DataFrame({
            "feature_name": ["feature1"],
            "cv": [0.1],
            "mean": [0.5],
        })
        mock_tracker_class.return_value = mock_tracker

        result = execute_feature_evaluation(
            feature_path=Path("features/"),
            model_type="lgbm",
            area="SE",
            top_n=10,
        )

        assert result.success
        assert result.total_features == 2
        assert len(result.top_features) == 2

    @patch("src.features.performance.importance_tracker.FeatureImportanceTracker")
    def test_empty_tracker(self, mock_tracker_class: MagicMock) -> None:
        """Test evaluation with empty tracker."""
        mock_tracker = MagicMock()
        mock_tracker.get_aggregated_importance.return_value = pd.DataFrame()
        mock_tracker_class.return_value = mock_tracker

        result = execute_feature_evaluation(
            feature_path=Path("features/"),
            model_type="lgbm",
            area="SE",
            top_n=10,
        )

        # Should fallback to synthetic features
        assert result.success
        assert len(result.top_features) > 0


class TestExecuteFeatureValidation:
    """Tests for execute_feature_validation function."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test validation of empty directory."""
        result = execute_feature_validation(
            feature_dir=tmp_path,
            schema_file=None,
            strict=False,
        )
        assert not result.is_valid
        assert result.total_files == 0
        assert len(result.issues) > 0

    def test_valid_parquet_files(self, tmp_path: Path) -> None:
        """Test validation of valid parquet files."""
        # Create valid parquet file
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
        })
        df.to_parquet(tmp_path / "features.parquet")

        result = execute_feature_validation(
            feature_dir=tmp_path,
            schema_file=None,
            strict=False,
        )
        assert result.is_valid
        assert result.total_files == 1
        assert result.valid_files == 1

    def test_files_with_missing_values(self, tmp_path: Path) -> None:
        """Test validation of files with missing values."""
        df = pd.DataFrame({
            "feature1": [1.0, None, 3.0],
            "feature2": [None, None, 6.0],
        })
        df.to_parquet(tmp_path / "features.parquet")

        result = execute_feature_validation(
            feature_dir=tmp_path,
            schema_file=None,
            strict=False,
        )
        assert result.missing_value_summary is not None
        assert "feature2" in result.missing_value_summary

    def test_strict_mode_with_high_missing(self, tmp_path: Path) -> None:
        """Test strict validation with high missing values."""
        df = pd.DataFrame({
            "feature1": [1.0, None, None, None, None],  # 80% missing
        })
        df.to_parquet(tmp_path / "features.parquet")

        result = execute_feature_validation(
            feature_dir=tmp_path,
            schema_file=None,
            strict=True,
        )
        assert not result.is_valid
        # Should have error for high missing values
        errors = [i for i in result.issues if i["severity"] == "error"]
        assert len(errors) > 0


class TestFeaturesCommandGroup:
    """Tests for features command group."""

    def test_features_group_help(self) -> None:
        """Test features group help."""
        runner = CliRunner()
        result = runner.invoke(features, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.output
        assert "evaluate" in result.output
        assert "validate" in result.output


class TestGenerateFeaturesCommand:
    """Tests for generate features command."""

    def test_generate_dry_run(self) -> None:
        """Test generate with dry run."""
        runner = CliRunner()
        result = runner.invoke(
            generate_features,
            [
                "--start-date", "2023-01-01",
                "--end-date", "2023-06-30",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_generate_with_plugins(self, tmp_path: Path) -> None:
        """Test generate with specific plugins."""
        runner = CliRunner()
        result = runner.invoke(
            generate_features,
            [
                "--plugin", "temporal",
                "--plugin", "calendar",
                "--start-date", "2023-01-01",
                "--end-date", "2023-06-30",
                "--output-dir", str(tmp_path),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "temporal" in result.output
        assert "calendar" in result.output

    def test_generate_with_areas(self) -> None:
        """Test generate with specific areas."""
        runner = CliRunner()
        result = runner.invoke(
            generate_features,
            [
                "--area", "SE",
                "--area", "S",
                "--start-date", "2023-01-01",
                "--end-date", "2023-03-31",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "2" in result.output  # 2 areas

    def test_generate_invalid_plugin_fails(self) -> None:
        """Test that invalid plugin fails."""
        runner = CliRunner()
        result = runner.invoke(
            generate_features,
            [
                "--plugin", "invalid_plugin",
                "--start-date", "2023-01-01",
                "--end-date", "2023-06-30",
            ],
        )
        assert result.exit_code != 0

    def test_generate_invalid_date_range(self) -> None:
        """Test that invalid date range fails."""
        runner = CliRunner()
        result = runner.invoke(
            generate_features,
            [
                "--start-date", "2023-06-30",
                "--end-date", "2023-01-01",
            ],
        )
        assert result.exit_code != 0

    def test_generate_output_format(self) -> None:
        """Test generate with different output formats."""
        runner = CliRunner()
        for fmt in ["parquet", "csv", "feather"]:
            result = runner.invoke(
                generate_features,
                [
                    "--start-date", "2023-01-01",
                    "--end-date", "2023-03-31",
                    "--format", fmt,
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert fmt in result.output

    def test_generate_parallel_workers(self) -> None:
        """Test generate with parallel workers."""
        runner = CliRunner()
        result = runner.invoke(
            generate_features,
            [
                "--start-date", "2023-01-01",
                "--end-date", "2023-03-31",
                "--parallel", "8",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "8" in result.output


class TestEvaluateFeaturesCommand:
    """Tests for evaluate features command."""

    def test_evaluate_basic(self, tmp_path: Path) -> None:
        """Test basic evaluation."""
        # Create a dummy feature file
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        df.to_parquet(tmp_path / "features.parquet")

        runner = CliRunner()
        result = runner.invoke(
            evaluate_features,
            [
                "--feature-set", str(tmp_path),
                "--model", "lgbm",
            ],
        )
        assert result.exit_code == 0
        assert "Feature Evaluation Results" in result.output

    def test_evaluate_with_output(self, tmp_path: Path) -> None:
        """Test evaluation with output file."""
        # Create a dummy feature file
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        df.to_parquet(tmp_path / "features.parquet")

        output_file = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(
            evaluate_features,
            [
                "--feature-set", str(tmp_path),
                "--output", str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_evaluate_different_models(self, tmp_path: Path) -> None:
        """Test evaluation with different models."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        df.to_parquet(tmp_path / "features.parquet")

        runner = CliRunner()
        for model in ["lgbm", "rf"]:
            result = runner.invoke(
                evaluate_features,
                [
                    "--feature-set", str(tmp_path),
                    "--model", model,
                ],
            )
            assert result.exit_code == 0
            assert model in result.output

    def test_evaluate_top_n(self, tmp_path: Path) -> None:
        """Test evaluation with custom top-n."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        df.to_parquet(tmp_path / "features.parquet")

        runner = CliRunner()
        result = runner.invoke(
            evaluate_features,
            [
                "--feature-set", str(tmp_path),
                "--top-n", "30",
            ],
        )
        assert result.exit_code == 0

    def test_evaluate_nonexistent_path_fails(self) -> None:
        """Test that nonexistent path fails."""
        runner = CliRunner()
        result = runner.invoke(
            evaluate_features,
            [
                "--feature-set", "/nonexistent/path",
            ],
        )
        assert result.exit_code != 0


class TestValidateFeaturesCommand:
    """Tests for validate features command."""

    def test_validate_basic(self, tmp_path: Path) -> None:
        """Test basic validation."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        df.to_parquet(tmp_path / "features.parquet")

        runner = CliRunner()
        result = runner.invoke(
            validate_features,
            [
                "--feature-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_validate_strict_mode(self, tmp_path: Path) -> None:
        """Test strict validation mode."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        df.to_parquet(tmp_path / "features.parquet")

        runner = CliRunner()
        result = runner.invoke(
            validate_features,
            [
                "--feature-dir", str(tmp_path),
                "--strict",
            ],
        )
        assert result.exit_code == 0
        assert "Strict mode: Enabled" in result.output

    def test_validate_empty_directory_fails(self, tmp_path: Path) -> None:
        """Test that empty directory fails validation."""
        runner = CliRunner()
        result = runner.invoke(
            validate_features,
            [
                "--feature-dir", str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        assert "FAIL" in result.output

    def test_validate_nonexistent_path_fails(self) -> None:
        """Test that nonexistent path fails."""
        runner = CliRunner()
        result = runner.invoke(
            validate_features,
            [
                "--feature-dir", "/nonexistent/path",
            ],
        )
        assert result.exit_code != 0


class TestListPluginsCommand:
    """Tests for list plugins command."""

    def test_list_plugins(self) -> None:
        """Test list plugins command."""
        runner = CliRunner()
        result = runner.invoke(list_plugins, [])
        assert result.exit_code == 0
        assert "temporal" in result.output
        assert "calendar" in result.output
        assert "lag" in result.output
        assert "wavelet" in result.output


class TestValidConstants:
    """Tests for valid constant values."""

    def test_valid_plugins_not_empty(self) -> None:
        """Test VALID_PLUGINS is not empty."""
        assert len(VALID_PLUGINS) > 0
        assert "temporal" in VALID_PLUGINS

    def test_valid_output_formats_not_empty(self) -> None:
        """Test VALID_OUTPUT_FORMATS is not empty."""
        assert len(VALID_OUTPUT_FORMATS) > 0
        assert "parquet" in VALID_OUTPUT_FORMATS

    def test_valid_eval_models_not_empty(self) -> None:
        """Test VALID_EVAL_MODELS is not empty."""
        assert len(VALID_EVAL_MODELS) > 0
        assert "lgbm" in VALID_EVAL_MODELS

    def test_valid_areas_not_empty(self) -> None:
        """Test VALID_AREAS is not empty."""
        assert len(VALID_AREAS) > 0
        assert "nacional" in VALID_AREAS
        assert "SE" in VALID_AREAS


class TestIntegration:
    """Integration tests for feature commands."""

    def test_full_generation_workflow(self, tmp_path: Path) -> None:
        """Test full generation workflow."""
        runner = CliRunner()

        # Generate features (dry run)
        result = runner.invoke(
            generate_features,
            [
                "--plugin", "temporal",
                "--area", "SE",
                "--start-date", "2023-01-01",
                "--end-date", "2023-03-31",
                "--output-dir", str(tmp_path),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Feature Generation Plan" in result.output
        assert "Feature Generation Summary" in result.output

    def test_full_validation_workflow(self, tmp_path: Path) -> None:
        """Test full validation workflow."""
        # Create test files
        df1 = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"feature2": [4.0, 5.0, 6.0]})
        df1.to_parquet(tmp_path / "features1.parquet")
        df2.to_csv(tmp_path / "features2.csv", index=False)

        runner = CliRunner()
        result = runner.invoke(
            validate_features,
            [
                "--feature-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert "Total files:" in result.output


class TestEdgeCases:
    """Tests for edge cases."""

    def test_generate_minimum_date_range(self) -> None:
        """Test generate with minimum date range."""
        runner = CliRunner()
        result = runner.invoke(
            generate_features,
            [
                "--start-date", "2023-01-01",
                "--end-date", "2023-01-02",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_generate_single_plugin(self) -> None:
        """Test generate with single plugin."""
        runner = CliRunner()
        result = runner.invoke(
            generate_features,
            [
                "--plugin", "temporal",
                "--start-date", "2023-01-01",
                "--end-date", "2023-03-31",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_generate_all_plugins(self) -> None:
        """Test generate with all plugins."""
        runner = CliRunner()
        args = ["--start-date", "2023-01-01", "--end-date", "2023-03-31", "--dry-run"]
        for plugin in VALID_PLUGINS:
            args.extend(["--plugin", plugin])

        result = runner.invoke(generate_features, args)
        assert result.exit_code == 0

    def test_evaluate_minimum_top_n(self, tmp_path: Path) -> None:
        """Test evaluate with minimum top-n."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        df.to_parquet(tmp_path / "features.parquet")

        runner = CliRunner()
        result = runner.invoke(
            evaluate_features,
            [
                "--feature-set", str(tmp_path),
                "--top-n", "1",
            ],
        )
        assert result.exit_code == 0

    def test_validate_nested_directory(self, tmp_path: Path) -> None:
        """Test validate with nested directory structure."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        df.to_parquet(subdir / "features.parquet")

        runner = CliRunner()
        result = runner.invoke(
            validate_features,
            [
                "--feature-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert "1" in result.output  # Should find the nested file
