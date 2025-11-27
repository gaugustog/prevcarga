"""Tests for evaluation CLI commands.

This module contains comprehensive tests for the evaluation commands
including metrics, drift detection, model comparison, and report generation.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.commands.evaluation import (
    VALID_AREAS,
    VALID_COMPARISON_METHODS,
    VALID_DRIFT_METHODS,
    VALID_METRICS,
    VALID_MODELS,
    VALID_OUTPUT_FORMATS,
    compare_command,
    drift_command,
    evaluate,
    metrics_command,
    ranking_command,
    report_command,
)
from src.cli.commands.evaluation_display import (
    ComparisonDisplay,
    ComparisonDisplayResult,
    DisplayTheme,
    DriftDisplay,
    DriftDisplayResult,
    EvaluationDisplayConfig,
    MetricsDisplay,
    MetricsDisplayResult,
    display_evaluation_summary,
    display_metrics_plan,
    save_evaluation_report,
)
from src.cli.core import OutputFormat


class TestConstants:
    """Tests for module constants."""

    def test_valid_models(self) -> None:
        """Test valid models list."""
        assert len(VALID_MODELS) >= 4
        assert "lgbm" in VALID_MODELS
        assert "random_forest" in VALID_MODELS
        assert "arima" in VALID_MODELS

    def test_valid_areas(self) -> None:
        """Test valid areas list."""
        assert "SECO" in VALID_AREAS
        assert "S" in VALID_AREAS
        assert "NE" in VALID_AREAS
        assert "N" in VALID_AREAS
        assert "SIN" in VALID_AREAS

    def test_valid_metrics(self) -> None:
        """Test valid metrics list."""
        assert "mape" in VALID_METRICS
        assert "smape" in VALID_METRICS
        assert "mae" in VALID_METRICS
        assert "rmse" in VALID_METRICS
        assert "r2" in VALID_METRICS

    def test_valid_drift_methods(self) -> None:
        """Test valid drift methods list."""
        assert "ks" in VALID_DRIFT_METHODS
        assert "psi" in VALID_DRIFT_METHODS
        assert "wasserstein" in VALID_DRIFT_METHODS

    def test_valid_comparison_methods(self) -> None:
        """Test valid comparison methods list."""
        assert "paired_ttest" in VALID_COMPARISON_METHODS
        assert "wilcoxon" in VALID_COMPARISON_METHODS
        assert "diebold_mariano" in VALID_COMPARISON_METHODS

    def test_valid_output_formats(self) -> None:
        """Test valid output formats list."""
        assert "html" in VALID_OUTPUT_FORMATS
        assert "json" in VALID_OUTPUT_FORMATS
        assert "markdown" in VALID_OUTPUT_FORMATS


class TestEvaluateGroup:
    """Tests for the evaluate command group."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()

    def test_evaluate_help(self, runner: CliRunner) -> None:
        """Test evaluate group help."""
        result = runner.invoke(evaluate, ["--help"])
        assert result.exit_code == 0
        assert "Model evaluation" in result.output

    def test_evaluate_subcommands_listed(self, runner: CliRunner) -> None:
        """Test that subcommands are listed in help."""
        result = runner.invoke(evaluate, ["--help"])
        assert "metrics" in result.output
        assert "drift" in result.output
        assert "compare" in result.output
        assert "report" in result.output
        assert "ranking" in result.output


class TestMetricsCommand:
    """Tests for the metrics command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_context(self) -> MagicMock:
        """Create mock CLI context."""
        ctx = MagicMock()
        ctx.output_format = OutputFormat.TEXT
        ctx.verbosity = MagicMock()
        ctx.verbosity.value = 1
        return ctx

    def test_metrics_help(self, runner: CliRunner) -> None:
        """Test metrics command help."""
        result = runner.invoke(evaluate, ["metrics", "--help"])
        assert result.exit_code == 0
        assert "Calculate" in result.output or "metrics" in result.output.lower()

    def test_metrics_dry_run(self, runner: CliRunner) -> None:
        """Test metrics command dry run."""
        result = runner.invoke(
            evaluate,
            ["metrics", "--dry-run", "--models", "lgbm"],
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_metrics_invalid_model(self, runner: CliRunner) -> None:
        """Test metrics command with invalid model."""
        result = runner.invoke(
            evaluate,
            ["metrics", "--models", "invalid_model"],
        )
        assert result.exit_code == 1
        assert "Invalid model" in result.output

    def test_metrics_invalid_area(self, runner: CliRunner) -> None:
        """Test metrics command with invalid area."""
        result = runner.invoke(
            evaluate,
            ["metrics", "--areas", "INVALID"],
        )
        assert result.exit_code == 1
        assert "Invalid area" in result.output

    def test_metrics_invalid_metric(self, runner: CliRunner) -> None:
        """Test metrics command with invalid metric."""
        result = runner.invoke(
            evaluate,
            ["metrics", "--metrics", "invalid_metric"],
        )
        assert result.exit_code == 1
        assert "Invalid metric" in result.output

    def test_metrics_with_date_range(self, runner: CliRunner) -> None:
        """Test metrics command with date range."""
        result = runner.invoke(
            evaluate,
            [
                "metrics",
                "--dry-run",
                "--start-date", "2024-01-01",
                "--end-date", "2024-06-30",
            ],
        )
        assert result.exit_code == 0

    def test_metrics_with_horizons(self, runner: CliRunner) -> None:
        """Test metrics command with horizons."""
        result = runner.invoke(
            evaluate,
            ["metrics", "--dry-run", "--horizons", "0,1,2"],
        )
        assert result.exit_code == 0

    def test_metrics_with_bootstrap(self, runner: CliRunner) -> None:
        """Test metrics command with bootstrap option."""
        result = runner.invoke(
            evaluate,
            ["metrics", "--dry-run", "--bootstrap"],
        )
        assert result.exit_code == 0

    def test_metrics_without_bootstrap(self, runner: CliRunner) -> None:
        """Test metrics command without bootstrap."""
        result = runner.invoke(
            evaluate,
            ["metrics", "--dry-run", "--no-bootstrap"],
        )
        assert result.exit_code == 0

    def test_metrics_confidence_level(self, runner: CliRunner) -> None:
        """Test metrics command with confidence level."""
        result = runner.invoke(
            evaluate,
            ["metrics", "--dry-run", "--confidence-level", "0.99"],
        )
        assert result.exit_code == 0


class TestDriftCommand:
    """Tests for the drift command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()

    def test_drift_help(self, runner: CliRunner) -> None:
        """Test drift command help."""
        result = runner.invoke(evaluate, ["drift", "--help"])
        assert result.exit_code == 0
        assert "drift" in result.output.lower()

    def test_drift_dry_run(self, runner: CliRunner) -> None:
        """Test drift command dry run."""
        result = runner.invoke(
            evaluate,
            ["drift", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_drift_with_features(self, runner: CliRunner) -> None:
        """Test drift command with specific features."""
        result = runner.invoke(
            evaluate,
            ["drift", "--dry-run", "--features", "load,temperature"],
        )
        assert result.exit_code == 0
        assert "load" in result.output
        assert "temperature" in result.output

    def test_drift_methods(self, runner: CliRunner) -> None:
        """Test drift command with different methods."""
        for method in VALID_DRIFT_METHODS:
            result = runner.invoke(
                evaluate,
                ["drift", "--dry-run", "--method", method],
            )
            assert result.exit_code == 0
            assert method.upper() in result.output

    def test_drift_threshold(self, runner: CliRunner) -> None:
        """Test drift command with custom threshold."""
        result = runner.invoke(
            evaluate,
            ["drift", "--dry-run", "--threshold", "0.01"],
        )
        assert result.exit_code == 0
        assert "0.01" in result.output

    def test_drift_alert_severity(self, runner: CliRunner) -> None:
        """Test drift command with alert severity."""
        for severity in ["low", "medium", "high"]:
            result = runner.invoke(
                evaluate,
                ["drift", "--dry-run", "--alert-severity", severity],
            )
            assert result.exit_code == 0


class TestCompareCommand:
    """Tests for the compare command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()

    def test_compare_help(self, runner: CliRunner) -> None:
        """Test compare command help."""
        result = runner.invoke(evaluate, ["compare", "--help"])
        assert result.exit_code == 0
        assert "Compare" in result.output or "model" in result.output.lower()

    def test_compare_requires_model_a(self, runner: CliRunner) -> None:
        """Test compare command requires model-a."""
        result = runner.invoke(
            evaluate,
            ["compare", "--model-b", "random_forest"],
        )
        assert result.exit_code != 0

    def test_compare_requires_model_b(self, runner: CliRunner) -> None:
        """Test compare command requires model-b."""
        result = runner.invoke(
            evaluate,
            ["compare", "--model-a", "lgbm"],
        )
        assert result.exit_code != 0

    def test_compare_dry_run(self, runner: CliRunner) -> None:
        """Test compare command dry run."""
        result = runner.invoke(
            evaluate,
            ["compare", "--model-a", "lgbm", "--model-b", "random_forest", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_compare_invalid_model_a(self, runner: CliRunner) -> None:
        """Test compare command with invalid model-a."""
        result = runner.invoke(
            evaluate,
            ["compare", "--model-a", "invalid", "--model-b", "lgbm"],
        )
        assert result.exit_code == 1
        assert "Invalid" in result.output

    def test_compare_invalid_model_b(self, runner: CliRunner) -> None:
        """Test compare command with invalid model-b."""
        result = runner.invoke(
            evaluate,
            ["compare", "--model-a", "lgbm", "--model-b", "invalid"],
        )
        assert result.exit_code == 1
        assert "Invalid" in result.output

    def test_compare_metrics(self, runner: CliRunner) -> None:
        """Test compare command with different metrics."""
        for metric in VALID_METRICS:
            result = runner.invoke(
                evaluate,
                [
                    "compare",
                    "--model-a", "lgbm",
                    "--model-b", "random_forest",
                    "--metric", metric,
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0
            assert metric.upper() in result.output

    def test_compare_tests(self, runner: CliRunner) -> None:
        """Test compare command with different statistical tests."""
        for test in VALID_COMPARISON_METHODS:
            result = runner.invoke(
                evaluate,
                [
                    "compare",
                    "--model-a", "lgbm",
                    "--model-b", "random_forest",
                    "--test", test,
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0

    def test_compare_significance_level(self, runner: CliRunner) -> None:
        """Test compare command with custom significance level."""
        result = runner.invoke(
            evaluate,
            [
                "compare",
                "--model-a", "lgbm",
                "--model-b", "random_forest",
                "--significance-level", "0.01",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "0.01" in result.output


class TestReportCommand:
    """Tests for the report command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()

    def test_report_help(self, runner: CliRunner) -> None:
        """Test report command help."""
        result = runner.invoke(evaluate, ["report", "--help"])
        assert result.exit_code == 0
        assert "report" in result.output.lower()

    def test_report_dry_run(self, runner: CliRunner) -> None:
        """Test report command dry run."""
        result = runner.invoke(
            evaluate,
            ["report", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_report_formats(self, runner: CliRunner) -> None:
        """Test report command with different formats."""
        for fmt in VALID_OUTPUT_FORMATS:
            result = runner.invoke(
                evaluate,
                ["report", "--dry-run", "--format", fmt],
            )
            assert result.exit_code == 0
            assert fmt in result.output

    def test_report_with_output_path(self, runner: CliRunner) -> None:
        """Test report command with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            result = runner.invoke(
                evaluate,
                ["report", "--dry-run", "--output", str(output_path)],
            )
            assert result.exit_code == 0

    def test_report_include_options(self, runner: CliRunner) -> None:
        """Test report command with include options."""
        result = runner.invoke(
            evaluate,
            [
                "report",
                "--dry-run",
                "--include-drift",
                "--include-comparison",
                "--include-percentiles",
            ],
        )
        assert result.exit_code == 0

    def test_report_exclude_options(self, runner: CliRunner) -> None:
        """Test report command with exclude options."""
        result = runner.invoke(
            evaluate,
            [
                "report",
                "--dry-run",
                "--no-drift",
                "--no-comparison",
            ],
        )
        assert result.exit_code == 0


class TestRankingCommand:
    """Tests for the ranking command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()

    def test_ranking_help(self, runner: CliRunner) -> None:
        """Test ranking command help."""
        result = runner.invoke(evaluate, ["ranking", "--help"])
        assert result.exit_code == 0
        assert "Rank" in result.output or "ranking" in result.output.lower()

    def test_ranking_default(self, runner: CliRunner) -> None:
        """Test ranking command with defaults."""
        result = runner.invoke(evaluate, ["ranking"])
        assert result.exit_code == 0
        assert "Ranking" in result.output

    def test_ranking_methods(self, runner: CliRunner) -> None:
        """Test ranking command with different methods."""
        for method in ["single_metric", "weighted_average", "borda_count", "pareto"]:
            result = runner.invoke(
                evaluate,
                ["ranking", "--method", method],
            )
            assert result.exit_code == 0

    def test_ranking_with_models(self, runner: CliRunner) -> None:
        """Test ranking command with specific models."""
        result = runner.invoke(
            evaluate,
            ["ranking", "--models", "lgbm,random_forest,arima"],
        )
        assert result.exit_code == 0

    def test_ranking_with_weights(self, runner: CliRunner) -> None:
        """Test ranking command with custom weights."""
        result = runner.invoke(
            evaluate,
            ["ranking", "--method", "weighted_average", "--weights", "0.5,0.3,0.2"],
        )
        assert result.exit_code == 0


class TestEvaluationDisplayConfig:
    """Tests for EvaluationDisplayConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = EvaluationDisplayConfig()
        assert config.output_format == OutputFormat.TEXT
        assert config.theme == DisplayTheme.DEFAULT
        assert config.show_timestamps is True
        assert config.decimal_places == 4
        assert config.show_confidence_intervals is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = EvaluationDisplayConfig(
            output_format=OutputFormat.JSON,
            theme=DisplayTheme.MINIMAL,
            show_timestamps=False,
            decimal_places=2,
            show_confidence_intervals=False,
        )
        assert config.output_format == OutputFormat.JSON
        assert config.theme == DisplayTheme.MINIMAL
        assert config.show_timestamps is False
        assert config.decimal_places == 2
        assert config.show_confidence_intervals is False


class TestMetricsDisplayResult:
    """Tests for MetricsDisplayResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic result creation."""
        result = MetricsDisplayResult(
            model_name="lgbm_SECO",
            mape=3.5,
            smape=3.3,
            mae=100.0,
            rmse=150.0,
            r2=0.95,
        )
        assert result.model_name == "lgbm_SECO"
        assert result.mape == 3.5
        assert result.smape == 3.3
        assert result.mae == 100.0
        assert result.rmse == 150.0
        assert result.r2 == 0.95
        assert result.mape_ci is None
        assert result.rmse_ci is None

    def test_result_with_ci(self) -> None:
        """Test result with confidence intervals."""
        result = MetricsDisplayResult(
            model_name="lgbm_SECO",
            mape=3.5,
            smape=3.3,
            mae=100.0,
            rmse=150.0,
            r2=0.95,
            mape_ci=(3.0, 4.0),
            rmse_ci=(140.0, 160.0),
        )
        assert result.mape_ci == (3.0, 4.0)
        assert result.rmse_ci == (140.0, 160.0)

    def test_result_has_timestamp(self) -> None:
        """Test result has timestamp."""
        result = MetricsDisplayResult(
            model_name="test",
            mape=1.0,
            smape=1.0,
            mae=1.0,
            rmse=1.0,
            r2=0.9,
        )
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class TestDriftDisplayResult:
    """Tests for DriftDisplayResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic drift result creation."""
        result = DriftDisplayResult(
            feature_name="load",
            drift_detected=True,
            drift_score=0.25,
            p_value=0.01,
            method="KS",
        )
        assert result.feature_name == "load"
        assert result.drift_detected is True
        assert result.drift_score == 0.25
        assert result.p_value == 0.01
        assert result.method == "KS"
        assert result.severity == "low"
        assert result.recommendation == ""

    def test_result_with_severity(self) -> None:
        """Test drift result with severity."""
        result = DriftDisplayResult(
            feature_name="temperature",
            drift_detected=True,
            drift_score=0.5,
            p_value=0.001,
            method="PSI",
            severity="high",
            recommendation="Retrain model",
        )
        assert result.severity == "high"
        assert result.recommendation == "Retrain model"

    def test_no_drift_result(self) -> None:
        """Test result when no drift detected."""
        result = DriftDisplayResult(
            feature_name="hour_sin",
            drift_detected=False,
            drift_score=0.02,
            p_value=0.5,
            method="KS",
        )
        assert result.drift_detected is False


class TestComparisonDisplayResult:
    """Tests for ComparisonDisplayResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic comparison result creation."""
        result = ComparisonDisplayResult(
            model_a="lgbm",
            model_b="rf",
            metric="MAPE",
            model_a_value=3.5,
            model_b_value=4.0,
            statistically_significant=True,
            p_value=0.02,
            winner="lgbm",
        )
        assert result.model_a == "lgbm"
        assert result.model_b == "rf"
        assert result.metric == "MAPE"
        assert result.model_a_value == 3.5
        assert result.model_b_value == 4.0
        assert result.statistically_significant is True
        assert result.p_value == 0.02
        assert result.winner == "lgbm"

    def test_tie_result(self) -> None:
        """Test comparison result with no winner (tie)."""
        result = ComparisonDisplayResult(
            model_a="lgbm",
            model_b="rf",
            metric="MAPE",
            model_a_value=3.5,
            model_b_value=3.6,
            statistically_significant=False,
            p_value=0.15,
            winner=None,
        )
        assert result.winner is None
        assert result.statistically_significant is False

    def test_result_with_effect_size(self) -> None:
        """Test comparison result with effect size."""
        result = ComparisonDisplayResult(
            model_a="lgbm",
            model_b="rf",
            metric="MAPE",
            model_a_value=3.5,
            model_b_value=4.5,
            statistically_significant=True,
            p_value=0.001,
            winner="lgbm",
            effect_size=0.8,
        )
        assert result.effect_size == 0.8


class TestMetricsDisplay:
    """Tests for MetricsDisplay class."""

    @pytest.fixture
    def display(self) -> MetricsDisplay:
        """Create metrics display instance."""
        return MetricsDisplay()

    @pytest.fixture
    def sample_results(self) -> list[MetricsDisplayResult]:
        """Create sample metrics results."""
        return [
            MetricsDisplayResult(
                model_name="lgbm_SECO",
                mape=3.5,
                smape=3.3,
                mae=100.0,
                rmse=150.0,
                r2=0.95,
                mape_ci=(3.0, 4.0),
            ),
            MetricsDisplayResult(
                model_name="rf_SECO",
                mape=4.0,
                smape=3.8,
                mae=120.0,
                rmse=170.0,
                r2=0.93,
                mape_ci=(3.5, 4.5),
            ),
        ]

    def test_format_metric(self, display: MetricsDisplay) -> None:
        """Test metric formatting."""
        result = display._format_metric(3.1415926)
        assert "3.1416" in result

    def test_format_metric_with_suffix(self, display: MetricsDisplay) -> None:
        """Test metric formatting with suffix."""
        result = display._format_metric(3.5, "%")
        assert "%" in result

    def test_format_ci(self, display: MetricsDisplay) -> None:
        """Test confidence interval formatting."""
        result = display._format_ci((3.0, 4.0))
        assert "3.0" in result
        assert "4.0" in result
        assert "[" in result
        assert "]" in result

    def test_format_ci_none(self, display: MetricsDisplay) -> None:
        """Test confidence interval formatting with None."""
        result = display._format_ci(None)
        assert result == "N/A"

    def test_show_metrics_table_text(
        self,
        display: MetricsDisplay,
        sample_results: list[MetricsDisplayResult],
    ) -> None:
        """Test metrics table display in text format."""
        # Should not raise
        display.show_metrics_table(sample_results)

    def test_show_metrics_table_json(
        self,
        sample_results: list[MetricsDisplayResult],
    ) -> None:
        """Test metrics table display in JSON format."""
        config = EvaluationDisplayConfig(output_format=OutputFormat.JSON)
        display = MetricsDisplay(config)
        # Should not raise
        display.show_metrics_table(sample_results)

    def test_show_metrics_table_quiet(
        self,
        sample_results: list[MetricsDisplayResult],
    ) -> None:
        """Test metrics table display in quiet mode."""
        config = EvaluationDisplayConfig(output_format=OutputFormat.QUIET)
        display = MetricsDisplay(config)
        # Should not raise, but produce no output
        display.show_metrics_table(sample_results)

    def test_show_single_model_metrics(
        self,
        display: MetricsDisplay,
        sample_results: list[MetricsDisplayResult],
    ) -> None:
        """Test single model metrics display."""
        display.show_single_model_metrics(sample_results[0])


class TestDriftDisplay:
    """Tests for DriftDisplay class."""

    @pytest.fixture
    def display(self) -> DriftDisplay:
        """Create drift display instance."""
        return DriftDisplay()

    @pytest.fixture
    def sample_results(self) -> list[DriftDisplayResult]:
        """Create sample drift results."""
        return [
            DriftDisplayResult(
                feature_name="load",
                drift_detected=True,
                drift_score=0.3,
                p_value=0.01,
                method="KS",
                severity="medium",
                recommendation="Monitor closely",
            ),
            DriftDisplayResult(
                feature_name="temperature",
                drift_detected=False,
                drift_score=0.02,
                p_value=0.8,
                method="KS",
            ),
        ]

    def test_show_drift_summary(
        self,
        display: DriftDisplay,
        sample_results: list[DriftDisplayResult],
    ) -> None:
        """Test drift summary display."""
        display.show_drift_summary(sample_results)

    def test_show_drift_summary_json(
        self,
        sample_results: list[DriftDisplayResult],
    ) -> None:
        """Test drift summary in JSON format."""
        config = EvaluationDisplayConfig(output_format=OutputFormat.JSON)
        display = DriftDisplay(config)
        display.show_drift_summary(sample_results)

    def test_show_drift_alert(self, display: DriftDisplay) -> None:
        """Test drift alert display."""
        result = DriftDisplayResult(
            feature_name="load",
            drift_detected=True,
            drift_score=0.5,
            p_value=0.001,
            method="KS",
            severity="high",
        )
        display.show_drift_alert(result)

    def test_show_drift_alert_no_drift(self, display: DriftDisplay) -> None:
        """Test drift alert when no drift."""
        result = DriftDisplayResult(
            feature_name="load",
            drift_detected=False,
            drift_score=0.01,
            p_value=0.9,
            method="KS",
        )
        # Should not display anything
        display.show_drift_alert(result)


class TestComparisonDisplay:
    """Tests for ComparisonDisplay class."""

    @pytest.fixture
    def display(self) -> ComparisonDisplay:
        """Create comparison display instance."""
        return ComparisonDisplay()

    @pytest.fixture
    def sample_results(self) -> list[ComparisonDisplayResult]:
        """Create sample comparison results."""
        return [
            ComparisonDisplayResult(
                model_a="lgbm_SECO",
                model_b="rf_SECO",
                metric="MAPE",
                model_a_value=3.5,
                model_b_value=4.0,
                statistically_significant=True,
                p_value=0.02,
                winner="lgbm_SECO",
                effect_size=0.5,
            ),
            ComparisonDisplayResult(
                model_a="lgbm_S",
                model_b="rf_S",
                metric="MAPE",
                model_a_value=3.8,
                model_b_value=3.9,
                statistically_significant=False,
                p_value=0.15,
                winner=None,
            ),
        ]

    def test_show_comparison_table(
        self,
        display: ComparisonDisplay,
        sample_results: list[ComparisonDisplayResult],
    ) -> None:
        """Test comparison table display."""
        display.show_comparison_table(sample_results)

    def test_show_comparison_table_json(
        self,
        sample_results: list[ComparisonDisplayResult],
    ) -> None:
        """Test comparison table in JSON format."""
        config = EvaluationDisplayConfig(output_format=OutputFormat.JSON)
        display = ComparisonDisplay(config)
        display.show_comparison_table(sample_results)

    def test_show_ranking(self, display: ComparisonDisplay) -> None:
        """Test ranking display."""
        rankings = [
            ("lgbm", 92.5, 1),
            ("rf", 89.0, 2),
            ("arima", 85.0, 3),
        ]
        display.show_ranking(rankings)

    def test_show_ranking_json(self) -> None:
        """Test ranking display in JSON format."""
        config = EvaluationDisplayConfig(output_format=OutputFormat.JSON)
        display = ComparisonDisplay(config)
        rankings = [
            ("lgbm", 92.5, 1),
            ("rf", 89.0, 2),
        ]
        display.show_ranking(rankings)


class TestDisplayFunctions:
    """Tests for standalone display functions."""

    def test_display_metrics_plan_text(self) -> None:
        """Test metrics plan display in text format."""
        display_metrics_plan(
            models=["lgbm", "rf"],
            areas=["SECO", "S"],
            horizons=[0, 1, 2],
            metrics=["mape", "rmse"],
            output_format=OutputFormat.TEXT,
        )

    def test_display_metrics_plan_json(self) -> None:
        """Test metrics plan display in JSON format."""
        display_metrics_plan(
            models=["lgbm"],
            areas=["SECO"],
            horizons=[0],
            metrics=["mape"],
            output_format=OutputFormat.JSON,
        )

    def test_display_metrics_plan_quiet(self) -> None:
        """Test metrics plan display in quiet mode."""
        display_metrics_plan(
            models=["lgbm"],
            areas=["SECO"],
            horizons=[0],
            metrics=["mape"],
            output_format=OutputFormat.QUIET,
        )

    def test_display_evaluation_summary_text(self) -> None:
        """Test evaluation summary display in text format."""
        display_evaluation_summary(
            total_evaluations=10,
            successful=8,
            failed=2,
            duration_seconds=15.5,
            output_format=OutputFormat.TEXT,
        )

    def test_display_evaluation_summary_json(self) -> None:
        """Test evaluation summary display in JSON format."""
        display_evaluation_summary(
            total_evaluations=10,
            successful=10,
            failed=0,
            duration_seconds=12.0,
            output_format=OutputFormat.JSON,
        )


class TestSaveEvaluationReport:
    """Tests for save_evaluation_report function."""

    @pytest.fixture
    def sample_metrics(self) -> list[MetricsDisplayResult]:
        """Create sample metrics results."""
        return [
            MetricsDisplayResult(
                model_name="lgbm_SECO",
                mape=3.5,
                smape=3.3,
                mae=100.0,
                rmse=150.0,
                r2=0.95,
            ),
        ]

    @pytest.fixture
    def sample_drift(self) -> list[DriftDisplayResult]:
        """Create sample drift results."""
        return [
            DriftDisplayResult(
                feature_name="load",
                drift_detected=True,
                drift_score=0.3,
                p_value=0.01,
                method="KS",
                severity="medium",
            ),
        ]

    @pytest.fixture
    def sample_comparison(self) -> list[ComparisonDisplayResult]:
        """Create sample comparison results."""
        return [
            ComparisonDisplayResult(
                model_a="lgbm",
                model_b="rf",
                metric="MAPE",
                model_a_value=3.5,
                model_b_value=4.0,
                statistically_significant=True,
                p_value=0.02,
                winner="lgbm",
            ),
        ]

    def test_save_html_report(
        self,
        sample_metrics: list[MetricsDisplayResult],
        sample_drift: list[DriftDisplayResult],
        sample_comparison: list[ComparisonDisplayResult],
    ) -> None:
        """Test saving HTML report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.html"
            result_path = save_evaluation_report(
                report_path=report_path,
                metrics_results=sample_metrics,
                drift_results=sample_drift,
                comparison_results=sample_comparison,
                format="html",
            )
            assert result_path.exists()
            content = result_path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "lgbm_SECO" in content
            assert "Metrics" in content

    def test_save_json_report(
        self,
        sample_metrics: list[MetricsDisplayResult],
    ) -> None:
        """Test saving JSON report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            result_path = save_evaluation_report(
                report_path=report_path,
                metrics_results=sample_metrics,
                format="json",
            )
            assert result_path.exists()
            content = json.loads(result_path.read_text())
            assert "metrics" in content
            assert len(content["metrics"]) == 1

    def test_save_markdown_report(
        self,
        sample_metrics: list[MetricsDisplayResult],
        sample_drift: list[DriftDisplayResult],
    ) -> None:
        """Test saving Markdown report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            result_path = save_evaluation_report(
                report_path=report_path,
                metrics_results=sample_metrics,
                drift_results=sample_drift,
                format="markdown",
            )
            assert result_path.exists()
            content = result_path.read_text()
            assert "# Evaluation Report" in content
            assert "| Model |" in content

    def test_save_creates_parent_directories(
        self,
        sample_metrics: list[MetricsDisplayResult],
    ) -> None:
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "subdir" / "nested" / "report.html"
            result_path = save_evaluation_report(
                report_path=report_path,
                metrics_results=sample_metrics,
                format="html",
            )
            assert result_path.exists()
            assert report_path.parent.exists()


class TestDisplayTheme:
    """Tests for DisplayTheme enum."""

    def test_theme_values(self) -> None:
        """Test theme enum values."""
        assert DisplayTheme.DEFAULT.value == "default"
        assert DisplayTheme.MINIMAL.value == "minimal"
        assert DisplayTheme.DETAILED.value == "detailed"

    def test_theme_from_string(self) -> None:
        """Test creating theme from string."""
        theme = DisplayTheme("default")
        assert theme == DisplayTheme.DEFAULT
