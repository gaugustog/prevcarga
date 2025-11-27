"""Tests for report generation framework.

Tests cover:
- Report configuration and validation
- Section generation for all section types
- Multiple output formats (HTML, Markdown, JSON, CSV, Text)
- Integration with metrics, comparison, and drift results
- Edge cases and error handling
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evaluation.report_generator import (
    EvaluationReport,
    ReportConfig,
    ReportFormat,
    ReportGenerator,
    ReportSection,
    SectionType,
)


class TestReportConfig:
    """Tests for ReportConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReportConfig()

        assert config.format == "html"
        assert config.title == "Forecast Evaluation Report"
        assert config.include_executive_summary is True
        assert config.include_metrics_table is True
        assert config.include_model_comparison is True
        assert config.include_drift_analysis is True
        assert config.include_time_analysis is True
        assert config.include_recommendations is True
        assert config.include_raw_data is False
        assert config.include_visualizations is True
        assert config.decimal_places == 2
        assert config.timezone == "America/Sao_Paulo"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReportConfig(
            format="markdown",
            title="Custom Report",
            include_raw_data=True,
            decimal_places=4,
        )

        assert config.format == "markdown"
        assert config.title == "Custom Report"
        assert config.include_raw_data is True
        assert config.decimal_places == 4

    def test_invalid_format(self):
        """Test validation rejects invalid format."""
        with pytest.raises(ValueError, match="Invalid format"):
            ReportConfig(format="invalid")

    def test_invalid_decimal_places(self):
        """Test validation rejects negative decimal places."""
        with pytest.raises(ValueError, match="decimal_places must be non-negative"):
            ReportConfig(decimal_places=-1)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ReportConfig(format="json", title="Test Report")
        result = config.to_dict()

        assert result["format"] == "json"
        assert result["title"] == "Test Report"
        assert "include_executive_summary" in result


class TestReportSection:
    """Tests for ReportSection."""

    def test_section_creation(self):
        """Test section creation."""
        section = ReportSection(
            section_type=SectionType.METRICS_TABLE,
            title="Performance Metrics",
            content="Test content",
        )

        assert section.section_type == SectionType.METRICS_TABLE
        assert section.title == "Performance Metrics"
        assert section.content == "Test content"
        assert section.tables == []
        assert section.charts == []

    def test_section_with_tables(self):
        """Test section with table data."""
        table = {
            "name": "metrics",
            "headers": ["Model", "MAPE"],
            "rows": [{"Model": "A", "MAPE": "5.2"}],
        }
        section = ReportSection(
            section_type=SectionType.METRICS_TABLE,
            title="Metrics",
            tables=[table],
        )

        assert len(section.tables) == 1
        assert section.tables[0]["name"] == "metrics"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        section = ReportSection(
            section_type=SectionType.EXECUTIVE_SUMMARY,
            title="Summary",
            content="Test",
            metadata={"key": "value"},
        )
        result = section.to_dict()

        assert result["section_type"] == "executive_summary"
        assert result["title"] == "Summary"
        assert result["metadata"]["key"] == "value"


class TestEvaluationReport:
    """Tests for EvaluationReport."""

    def test_report_creation(self):
        """Test report creation."""
        report = EvaluationReport(
            title="Test Report",
            format=ReportFormat.HTML,
        )

        assert report.title == "Test Report"
        assert report.format == ReportFormat.HTML
        assert report.sections == []
        assert isinstance(report.generated_at, datetime)

    def test_save_report(self):
        """Test saving report to file."""
        report = EvaluationReport(
            title="Test",
            raw_content="<html><body>Test</body></html>",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.html"
            report.save(filepath)

            assert filepath.exists()
            with open(filepath) as f:
                assert f.read() == "<html><body>Test</body></html>"

    def test_save_creates_directory(self):
        """Test saving creates parent directories."""
        report = EvaluationReport(title="Test", raw_content="content")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "report.html"
            report.save(filepath)

            assert filepath.exists()

    def test_get_section(self):
        """Test retrieving specific section."""
        section = ReportSection(
            section_type=SectionType.METRICS_TABLE,
            title="Metrics",
        )
        report = EvaluationReport(title="Test", sections=[section])

        found = report.get_section(SectionType.METRICS_TABLE)
        assert found is not None
        assert found.title == "Metrics"

        not_found = report.get_section(SectionType.DRIFT_ANALYSIS)
        assert not_found is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = EvaluationReport(
            title="Test",
            format=ReportFormat.JSON,
            metadata={"version": "1.0"},
        )
        result = report.to_dict()

        assert result["title"] == "Test"
        assert result["format"] == "json"
        assert "generated_at" in result
        assert result["metadata"]["version"] == "1.0"

    def test_repr(self):
        """Test string representation."""
        report = EvaluationReport(
            title="My Report",
            format=ReportFormat.MARKDOWN,
            sections=[
                ReportSection(SectionType.EXECUTIVE_SUMMARY, "Summary"),
                ReportSection(SectionType.METRICS_TABLE, "Metrics"),
            ],
        )

        repr_str = repr(report)
        assert "My Report" in repr_str
        assert "markdown" in repr_str
        assert "n_sections=2" in repr_str


class TestReportGenerator:
    """Tests for ReportGenerator."""

    def test_default_generator(self):
        """Test generator with default config."""
        generator = ReportGenerator()

        assert generator.config.format == "html"

    def test_custom_config_generator(self):
        """Test generator with custom config."""
        config = ReportConfig(format="markdown", title="Custom")
        generator = ReportGenerator(config)

        assert generator.config.format == "markdown"
        assert generator.config.title == "Custom"

    def test_generate_empty_report(self):
        """Test generating report with no data."""
        generator = ReportGenerator()
        report = generator.generate()

        assert report.title == "Forecast Evaluation Report"
        assert len(report.sections) >= 1  # At least executive summary

    def test_generate_with_metrics(self):
        """Test generating report with metrics data."""
        metrics_results = {
            "model_a": {"mape": 5.2, "mae": 100, "rmse": 150, "r2": 0.95},
            "model_b": {"mape": 6.1, "mae": 120, "rmse": 180, "r2": 0.92},
        }

        generator = ReportGenerator()
        report = generator.generate(metrics_results=metrics_results)

        # Find metrics section
        metrics_section = report.get_section(SectionType.METRICS_TABLE)
        assert metrics_section is not None
        assert "model_a" in metrics_section.content
        assert "model_b" in metrics_section.content

    def test_generate_with_metrics_objects(self):
        """Test generating report with metrics result objects."""
        # Create mock metrics result objects
        mock_result_a = MagicMock()
        mock_result_a.mape = 5.2
        mock_result_a.to_dict.return_value = {"mape": 5.2, "mae": 100, "rmse": 150, "r2": 0.95}

        mock_result_b = MagicMock()
        mock_result_b.mape = 6.1
        mock_result_b.to_dict.return_value = {"mape": 6.1, "mae": 120, "rmse": 180, "r2": 0.92}

        metrics_results = {"model_a": mock_result_a, "model_b": mock_result_b}

        generator = ReportGenerator()
        report = generator.generate(metrics_results=metrics_results)

        metrics_section = report.get_section(SectionType.METRICS_TABLE)
        assert metrics_section is not None
        assert len(metrics_section.tables) == 1

    def test_generate_with_comparison(self):
        """Test generating report with comparison data."""
        # Create mock comparison result
        mock_ranking = MagicMock()
        mock_ranking.model_name = "model_a"
        mock_ranking.rank = 1
        mock_ranking.metrics = {"mape": 5.2}
        mock_ranking.score = 0.95

        mock_comparison = MagicMock()
        mock_comparison.best_model = "model_a"
        mock_comparison.rankings = [mock_ranking]
        mock_comparison.pairwise_comparisons = []
        mock_comparison.recommendation = "Use model_a"

        generator = ReportGenerator()
        report = generator.generate(comparison_result=mock_comparison)

        comparison_section = report.get_section(SectionType.MODEL_COMPARISON)
        assert comparison_section is not None
        assert "model_a" in comparison_section.content
        assert "Best Model" in comparison_section.content

    def test_generate_with_pairwise_comparisons(self):
        """Test report with pairwise statistical comparisons."""
        mock_pairwise = MagicMock()
        mock_pairwise.model_a = "model_a"
        mock_pairwise.model_b = "model_b"
        mock_pairwise.is_significant = True
        mock_pairwise.better_model = "model_a"

        mock_comparison = MagicMock()
        mock_comparison.best_model = "model_a"
        mock_comparison.rankings = []
        mock_comparison.pairwise_comparisons = [mock_pairwise]
        mock_comparison.recommendation = None

        generator = ReportGenerator()
        report = generator.generate(comparison_result=mock_comparison)

        comparison_section = report.get_section(SectionType.MODEL_COMPARISON)
        assert comparison_section is not None
        assert "SIGNIFICANT" in comparison_section.content
        assert "model_a vs model_b" in comparison_section.content

    def test_generate_with_drift_results(self):
        """Test generating report with drift detection data."""
        # Create mock drift results
        mock_drift = MagicMock()
        mock_drift.drift_detected = True
        mock_drift.drift_score = 0.15
        mock_drift.method = MagicMock()
        mock_drift.method.value = "ks_test"

        drift_results = {"feature_1": mock_drift}

        generator = ReportGenerator()
        report = generator.generate(drift_results=drift_results)

        drift_section = report.get_section(SectionType.DRIFT_ANALYSIS)
        assert drift_section is not None
        assert "feature_1" in drift_section.content
        assert "DETECTED" in drift_section.content

    def test_generate_with_drift_dict_results(self):
        """Test generating report with drift results as dicts."""
        drift_results = {
            "feature_1": {
                "drift_detected": True,
                "drift_score": 0.15,
                "method": "ks_test",
            },
            "feature_2": {
                "drift_detected": False,
                "drift_score": 0.02,
                "method": "psi",
            },
        }

        generator = ReportGenerator()
        report = generator.generate(drift_results=drift_results)

        drift_section = report.get_section(SectionType.DRIFT_ANALYSIS)
        assert drift_section is not None
        assert "feature_1" in drift_section.content
        assert "feature_2" in drift_section.content

    def test_generate_with_time_analysis(self):
        """Test generating report with time period analysis."""
        time_analysis = {
            "hourly": {
                "morning": {"mape": 4.5},
                "afternoon": {"mape": 5.2},
                "evening": {"mape": 6.1},
            },
            "daily": {
                "weekday": {"mape": 5.0},
                "weekend": {"mape": 7.2},
            },
        }

        generator = ReportGenerator()
        report = generator.generate(time_analysis=time_analysis)

        time_section = report.get_section(SectionType.TIME_ANALYSIS)
        assert time_section is not None
        assert "hourly" in time_section.content
        assert "morning" in time_section.content

    def test_generate_with_raw_data(self):
        """Test generating report with raw data section."""
        config = ReportConfig(include_raw_data=True)
        generator = ReportGenerator(config)

        metrics_results = {"model_a": {"mape": 5.2}}
        report = generator.generate(
            metrics_results=metrics_results,
            custom_data={"custom_key": "custom_value"},
        )

        raw_section = report.get_section(SectionType.RAW_DATA)
        assert raw_section is not None
        assert "custom_key" in raw_section.content


class TestReportFormats:
    """Tests for different output formats."""

    @pytest.fixture
    def sample_metrics(self):
        """Sample metrics for testing."""
        return {
            "model_a": {"mape": 5.2, "mae": 100, "rmse": 150, "r2": 0.95},
            "model_b": {"mape": 6.1, "mae": 120, "rmse": 180, "r2": 0.92},
        }

    def test_html_format(self, sample_metrics):
        """Test HTML format output."""
        config = ReportConfig(format="html", title="HTML Report")
        generator = ReportGenerator(config)
        report = generator.generate(metrics_results=sample_metrics)

        assert report.format == ReportFormat.HTML
        assert "<!DOCTYPE html>" in report.raw_content
        assert "<html>" in report.raw_content
        assert "HTML Report" in report.raw_content
        assert "<table>" in report.raw_content
        assert "</html>" in report.raw_content

    def test_html_custom_css(self, sample_metrics):
        """Test HTML format with custom CSS."""
        custom_css = "body { background: red; }"
        config = ReportConfig(format="html", custom_css=custom_css)
        generator = ReportGenerator(config)
        report = generator.generate(metrics_results=sample_metrics)

        assert custom_css in report.raw_content

    def test_markdown_format(self, sample_metrics):
        """Test Markdown format output."""
        config = ReportConfig(format="markdown", title="MD Report")
        generator = ReportGenerator(config)
        report = generator.generate(metrics_results=sample_metrics)

        assert report.format == ReportFormat.MARKDOWN
        assert "# MD Report" in report.raw_content
        assert "## " in report.raw_content  # Section headers
        assert "|" in report.raw_content  # Table syntax
        assert "---" in report.raw_content  # Table separator

    def test_json_format(self, sample_metrics):
        """Test JSON format output."""
        config = ReportConfig(format="json", title="JSON Report")
        generator = ReportGenerator(config)
        report = generator.generate(metrics_results=sample_metrics)

        assert report.format == ReportFormat.JSON

        # Verify it's valid JSON
        parsed = json.loads(report.raw_content)
        assert parsed["title"] == "JSON Report"
        assert "sections" in parsed
        assert "generated_at" in parsed

    def test_csv_format(self, sample_metrics):
        """Test CSV format output."""
        config = ReportConfig(format="csv", title="CSV Report")
        generator = ReportGenerator(config)
        report = generator.generate(metrics_results=sample_metrics)

        assert report.format == ReportFormat.CSV
        assert "," in report.raw_content  # CSV separator
        assert "Model" in report.raw_content  # Table header

    def test_text_format(self, sample_metrics):
        """Test plain text format output."""
        config = ReportConfig(format="text", title="Text Report")
        generator = ReportGenerator(config)
        report = generator.generate(metrics_results=sample_metrics)

        assert report.format == ReportFormat.TEXT
        assert "Text Report" in report.raw_content
        assert "=" * 60 in report.raw_content  # Header separator
        assert "-" * 60 in report.raw_content  # Section separator


class TestRecommendations:
    """Tests for recommendations generation."""

    def test_recommendations_with_best_model(self):
        """Test recommendations with best model identified."""
        mock_comparison = MagicMock()
        mock_comparison.best_model = "model_a"
        mock_comparison.rankings = []

        generator = ReportGenerator()
        report = generator.generate(comparison_result=mock_comparison)

        recommendations = report.get_section(SectionType.RECOMMENDATIONS)
        assert recommendations is not None
        assert "model_a" in recommendations.content
        assert "Deploy" in recommendations.content

    def test_recommendations_with_drift(self):
        """Test recommendations when drift is detected."""
        drift_results = {
            "feature_1": {"drift_detected": True, "drift_score": 0.2},
        }

        generator = ReportGenerator()
        report = generator.generate(drift_results=drift_results)

        recommendations = report.get_section(SectionType.RECOMMENDATIONS)
        assert recommendations is not None
        assert "drift" in recommendations.content.lower()
        assert "retrain" in recommendations.content.lower()

    def test_recommendations_with_high_mape(self):
        """Test recommendations when MAPE is high."""
        metrics_results = {
            "model_a": {"mape": 15.0},  # High MAPE
            "model_b": {"mape": 12.0},
        }

        generator = ReportGenerator()
        report = generator.generate(metrics_results=metrics_results)

        recommendations = report.get_section(SectionType.RECOMMENDATIONS)
        assert recommendations is not None
        assert "10%" in recommendations.content  # Threshold mention

    def test_general_recommendations_always_present(self):
        """Test that general recommendations are always included."""
        generator = ReportGenerator()
        report = generator.generate()

        recommendations = report.get_section(SectionType.RECOMMENDATIONS)
        assert recommendations is not None
        assert "General Recommendations" in recommendations.content


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_metrics_results(self):
        """Test with empty metrics dictionary."""
        generator = ReportGenerator()
        report = generator.generate(metrics_results={})

        # Should not include metrics section with empty data
        assert report is not None

    def test_none_values_in_metrics(self):
        """Test handling of None values in metrics."""
        metrics_results = {
            "model_a": {"mape": None, "mae": 100, "rmse": None, "r2": 0.95},
        }

        generator = ReportGenerator()
        report = generator.generate(metrics_results=metrics_results)

        metrics_section = report.get_section(SectionType.METRICS_TABLE)
        assert metrics_section is not None
        assert "N/A" in metrics_section.content

    def test_format_number_edge_cases(self):
        """Test number formatting edge cases."""
        generator = ReportGenerator()

        # Test N/A handling
        assert generator._format_number("N/A") == "N/A"
        assert generator._format_number(None) == "N/A"

        # Test numeric formatting
        assert generator._format_number(5.123) == "5.12"
        assert generator._format_number(5) == "5.00"

        # Test invalid values
        assert generator._format_number("invalid") == "invalid"

    def test_custom_decimal_places(self):
        """Test custom decimal places formatting."""
        config = ReportConfig(decimal_places=4)
        generator = ReportGenerator(config)

        assert generator._format_number(5.123456) == "5.1235"

    def test_section_type_enum_values(self):
        """Test section type enum values."""
        assert SectionType.EXECUTIVE_SUMMARY.value == "executive_summary"
        assert SectionType.METRICS_TABLE.value == "metrics_table"
        assert SectionType.MODEL_COMPARISON.value == "model_comparison"
        assert SectionType.DRIFT_ANALYSIS.value == "drift_analysis"
        assert SectionType.TIME_ANALYSIS.value == "time_analysis"
        assert SectionType.RECOMMENDATIONS.value == "recommendations"
        assert SectionType.RAW_DATA.value == "raw_data"

    def test_report_format_enum_values(self):
        """Test report format enum values."""
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.CSV.value == "csv"
        assert ReportFormat.TEXT.value == "text"

    def test_generator_repr(self):
        """Test generator string representation."""
        config = ReportConfig(format="markdown")
        generator = ReportGenerator(config)

        repr_str = repr(generator)
        assert "ReportGenerator" in repr_str
        assert "markdown" in repr_str


class TestConfiguredSections:
    """Tests for section inclusion configuration."""

    def test_disable_executive_summary(self):
        """Test disabling executive summary."""
        config = ReportConfig(include_executive_summary=False)
        generator = ReportGenerator(config)
        report = generator.generate()

        summary = report.get_section(SectionType.EXECUTIVE_SUMMARY)
        assert summary is None

    def test_disable_metrics_table(self):
        """Test disabling metrics table."""
        config = ReportConfig(include_metrics_table=False)
        generator = ReportGenerator(config)

        metrics_results = {"model_a": {"mape": 5.2}}
        report = generator.generate(metrics_results=metrics_results)

        metrics = report.get_section(SectionType.METRICS_TABLE)
        assert metrics is None

    def test_disable_model_comparison(self):
        """Test disabling model comparison."""
        config = ReportConfig(include_model_comparison=False)
        generator = ReportGenerator(config)

        mock_comparison = MagicMock()
        mock_comparison.best_model = "model_a"
        report = generator.generate(comparison_result=mock_comparison)

        comparison = report.get_section(SectionType.MODEL_COMPARISON)
        assert comparison is None

    def test_disable_drift_analysis(self):
        """Test disabling drift analysis."""
        config = ReportConfig(include_drift_analysis=False)
        generator = ReportGenerator(config)

        drift_results = {"feature_1": {"drift_detected": True}}
        report = generator.generate(drift_results=drift_results)

        drift = report.get_section(SectionType.DRIFT_ANALYSIS)
        assert drift is None

    def test_disable_time_analysis(self):
        """Test disabling time analysis."""
        config = ReportConfig(include_time_analysis=False)
        generator = ReportGenerator(config)

        time_analysis = {"hourly": {"morning": {"mape": 5.0}}}
        report = generator.generate(time_analysis=time_analysis)

        time_section = report.get_section(SectionType.TIME_ANALYSIS)
        assert time_section is None

    def test_disable_recommendations(self):
        """Test disabling recommendations."""
        config = ReportConfig(include_recommendations=False)
        generator = ReportGenerator(config)
        report = generator.generate()

        recommendations = report.get_section(SectionType.RECOMMENDATIONS)
        assert recommendations is None

    def test_minimal_report(self):
        """Test generating minimal report with all sections disabled."""
        config = ReportConfig(
            include_executive_summary=False,
            include_metrics_table=False,
            include_model_comparison=False,
            include_drift_analysis=False,
            include_time_analysis=False,
            include_recommendations=False,
            include_raw_data=False,
        )
        generator = ReportGenerator(config)
        report = generator.generate()

        assert len(report.sections) == 0


class TestIntegration:
    """Integration tests with real-like data."""

    def test_complete_report_generation(self):
        """Test generating complete report with all data types."""
        # Metrics
        metrics_results = {
            "lightgbm": {"mape": 4.5, "mae": 95, "rmse": 140, "r2": 0.96},
            "xgboost": {"mape": 5.1, "mae": 110, "rmse": 160, "r2": 0.94},
            "linear": {"mape": 7.2, "mae": 150, "rmse": 200, "r2": 0.89},
        }

        # Comparison
        mock_ranking1 = MagicMock()
        mock_ranking1.model_name = "lightgbm"
        mock_ranking1.rank = 1
        mock_ranking1.metrics = {"mape": 4.5}
        mock_ranking1.score = 0.96

        mock_ranking2 = MagicMock()
        mock_ranking2.model_name = "xgboost"
        mock_ranking2.rank = 2
        mock_ranking2.metrics = {"mape": 5.1}
        mock_ranking2.score = 0.94

        mock_comparison = MagicMock()
        mock_comparison.best_model = "lightgbm"
        mock_comparison.rankings = [mock_ranking1, mock_ranking2]
        mock_comparison.pairwise_comparisons = []
        mock_comparison.recommendation = "Deploy lightgbm"

        # Drift
        drift_results = {
            "load": {"drift_detected": False, "drift_score": 0.02, "method": "ks_test"},
            "temperature": {"drift_detected": True, "drift_score": 0.18, "method": "psi"},
        }

        # Time analysis
        time_analysis = {
            "hourly": {
                "peak": {"mape": 3.8},
                "off_peak": {"mape": 5.2},
            },
        }

        # Generate report
        config = ReportConfig(
            format="html",
            title="Complete Forecast Evaluation Report",
            include_raw_data=True,
        )
        generator = ReportGenerator(config)
        report = generator.generate(
            metrics_results=metrics_results,
            comparison_result=mock_comparison,
            drift_results=drift_results,
            time_analysis=time_analysis,
            custom_data={"evaluation_date": "2024-01-15"},
        )

        # Verify all sections present
        assert report.get_section(SectionType.EXECUTIVE_SUMMARY) is not None
        assert report.get_section(SectionType.METRICS_TABLE) is not None
        assert report.get_section(SectionType.MODEL_COMPARISON) is not None
        assert report.get_section(SectionType.DRIFT_ANALYSIS) is not None
        assert report.get_section(SectionType.TIME_ANALYSIS) is not None
        assert report.get_section(SectionType.RECOMMENDATIONS) is not None
        assert report.get_section(SectionType.RAW_DATA) is not None

        # Verify content
        assert "lightgbm" in report.raw_content
        assert "DETECTED" in report.raw_content  # Drift detected
        assert len(report.sections) == 7

    def test_save_and_load_json_report(self):
        """Test saving JSON report and verifying structure."""
        metrics_results = {
            "model_a": {"mape": 5.2, "mae": 100, "rmse": 150, "r2": 0.95},
        }

        config = ReportConfig(format="json")
        generator = ReportGenerator(config)
        report = generator.generate(metrics_results=metrics_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            report.save(filepath)

            # Load and verify
            with open(filepath) as f:
                loaded = json.load(f)

            assert loaded["title"] == "Forecast Evaluation Report"
            assert len(loaded["sections"]) >= 2  # Summary + Metrics + Recommendations

    def test_brazilian_load_forecasting_scenario(self):
        """Test report for Brazilian load forecasting scenario."""
        # Metrics by subsystem
        metrics_results = {
            "SECO": {"mape": 3.8, "mae": 450, "rmse": 580, "r2": 0.97},
            "S": {"mape": 4.2, "mae": 220, "rmse": 290, "r2": 0.96},
            "NE": {"mape": 5.1, "mae": 180, "rmse": 240, "r2": 0.94},
            "N": {"mape": 6.3, "mae": 95, "rmse": 130, "r2": 0.92},
        }

        # Time analysis by hour of day
        time_analysis = {
            "hour_of_day": {
                "00": {"mape": 3.5},
                "06": {"mape": 4.2},
                "12": {"mape": 5.8},
                "18": {"mape": 6.1},
            },
        }

        config = ReportConfig(
            format="markdown",
            title="PrevCarga Load Forecast Evaluation - SIN Subsystems",
        )
        generator = ReportGenerator(config)
        report = generator.generate(
            metrics_results=metrics_results,
            time_analysis=time_analysis,
        )

        # Verify subsystem names present
        assert "SECO" in report.raw_content
        assert "## " in report.raw_content  # Markdown headers
