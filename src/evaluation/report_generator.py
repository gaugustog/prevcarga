"""Report generation framework for forecast evaluation results.

This module provides comprehensive report generation capabilities for
summarizing and presenting forecast model evaluation results in various
formats (HTML, Markdown, JSON, CSV).

Key Components:
- ReportGenerator: Main class for generating evaluation reports
- ReportConfig: Configuration for report generation
- ReportSection: Container for individual report sections
- EvaluationReport: Complete evaluation report

Supported Features:
- Multiple output formats (HTML, Markdown, JSON, CSV)
- Executive summary generation
- Detailed metrics tables
- Model comparison summaries
- Drift detection summaries
- Visualization placeholders
- Template-based customization

Example:
    ```python
    from src.evaluation.report_generator import ReportGenerator, ReportConfig

    # Configure generator
    config = ReportConfig(
        format="html",
        include_visualizations=True,
        include_recommendations=True,
    )

    # Create generator
    generator = ReportGenerator(config)

    # Generate report
    report = generator.generate(
        metrics_results=metrics_dict,
        comparison_result=comparison_result,
        drift_results=drift_results,
    )

    # Save to file
    report.save("evaluation_report.html")
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Supported report formats.

    Attributes:
        HTML: HTML format with styling.
        MARKDOWN: Markdown format.
        JSON: JSON format for programmatic use.
        CSV: CSV format for tabular data.
        TEXT: Plain text format.
    """

    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    TEXT = "text"


class SectionType(Enum):
    """Types of report sections.

    Attributes:
        EXECUTIVE_SUMMARY: High-level summary.
        METRICS_TABLE: Detailed metrics table.
        MODEL_COMPARISON: Model comparison results.
        DRIFT_ANALYSIS: Drift detection results.
        TIME_ANALYSIS: Time period analysis.
        RECOMMENDATIONS: Actionable recommendations.
        RAW_DATA: Raw data dump.
    """

    EXECUTIVE_SUMMARY = "executive_summary"
    METRICS_TABLE = "metrics_table"
    MODEL_COMPARISON = "model_comparison"
    DRIFT_ANALYSIS = "drift_analysis"
    TIME_ANALYSIS = "time_analysis"
    RECOMMENDATIONS = "recommendations"
    RAW_DATA = "raw_data"


@dataclass
class ReportConfig:
    """Configuration for report generation.

    Attributes:
        format: Output format (html, markdown, json, csv, text).
        title: Report title.
        include_executive_summary: Include executive summary section.
        include_metrics_table: Include detailed metrics table.
        include_model_comparison: Include model comparison section.
        include_drift_analysis: Include drift analysis section.
        include_time_analysis: Include time period analysis.
        include_recommendations: Include recommendations section.
        include_raw_data: Include raw data dump.
        include_visualizations: Include visualization placeholders.
        decimal_places: Number of decimal places for numeric values.
        date_format: Format for dates.
        timezone: Timezone for timestamps.
        custom_css: Custom CSS for HTML reports.
        template_dir: Directory for custom templates.

    Example:
        >>> config = ReportConfig(
        ...     format="html",
        ...     title="Model Evaluation Report",
        ...     include_recommendations=True
        ... )
    """

    format: str = "html"
    title: str = "Forecast Evaluation Report"
    include_executive_summary: bool = True
    include_metrics_table: bool = True
    include_model_comparison: bool = True
    include_drift_analysis: bool = True
    include_time_analysis: bool = True
    include_recommendations: bool = True
    include_raw_data: bool = False
    include_visualizations: bool = True
    decimal_places: int = 2
    date_format: str = "%Y-%m-%d %H:%M:%S"
    timezone: str = "America/Sao_Paulo"
    custom_css: str | None = None
    template_dir: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_formats = {"html", "markdown", "json", "csv", "text"}
        if self.format not in valid_formats:
            msg = f"Invalid format '{self.format}'. Must be one of {valid_formats}"
            raise ValueError(msg)

        if self.decimal_places < 0:
            msg = f"decimal_places must be non-negative, got {self.decimal_places}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "format": self.format,
            "title": self.title,
            "include_executive_summary": self.include_executive_summary,
            "include_metrics_table": self.include_metrics_table,
            "include_model_comparison": self.include_model_comparison,
            "include_drift_analysis": self.include_drift_analysis,
            "include_time_analysis": self.include_time_analysis,
            "include_recommendations": self.include_recommendations,
            "include_raw_data": self.include_raw_data,
            "include_visualizations": self.include_visualizations,
            "decimal_places": self.decimal_places,
        }


@dataclass
class ReportSection:
    """Container for a single report section.

    Attributes:
        section_type: Type of section.
        title: Section title.
        content: Section content (format-specific).
        tables: List of tables in the section.
        charts: List of chart specifications.
        metadata: Section metadata.

    Example:
        >>> section = ReportSection(
        ...     section_type=SectionType.METRICS_TABLE,
        ...     title="Performance Metrics",
        ...     content="Detailed metrics..."
        ... )
    """

    section_type: SectionType
    title: str
    content: str = ""
    tables: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "section_type": self.section_type.value,
            "title": self.title,
            "content": self.content,
            "tables": self.tables,
            "charts": self.charts,
            "metadata": self.metadata.copy(),
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report.

    Attributes:
        title: Report title.
        generated_at: When the report was generated.
        sections: List of report sections.
        format: Report format.
        raw_content: Raw formatted content.
        metadata: Report metadata.

    Example:
        >>> report = EvaluationReport(
        ...     title="Model Evaluation",
        ...     sections=[section1, section2],
        ...     format=ReportFormat.HTML
        ... )
        >>> report.save("report.html")
    """

    title: str
    generated_at: datetime = field(default_factory=datetime.now)
    sections: list[ReportSection] = field(default_factory=list)
    format: ReportFormat = ReportFormat.HTML
    raw_content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, filepath: str | Path) -> None:
        """Save report to file.

        Args:
            filepath: Path to save the report.

        Raises:
            IOError: If file cannot be written.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.raw_content)

        logger.info("Report saved to %s", filepath)

    def get_section(self, section_type: SectionType) -> ReportSection | None:
        """Get a specific section by type.

        Args:
            section_type: Type of section to retrieve.

        Returns:
            ReportSection or None if not found.
        """
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "format": self.format.value,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata.copy(),
        }

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return f"EvaluationReport(title={self.title!r}, format={self.format.value}, n_sections={len(self.sections)})"


class ReportGenerator:
    """Report generation framework for forecast evaluation.

    Generates comprehensive evaluation reports in multiple formats,
    including executive summaries, metrics tables, model comparisons,
    and recommendations.

    Example:
        >>> generator = ReportGenerator()

        >>> # Generate report from evaluation results
        >>> report = generator.generate(
        ...     metrics_results={"model_a": metrics_a},
        ...     comparison_result=comparison,
        ... )

        >>> # Save report
        >>> report.save("evaluation_report.html")

        >>> # Generate specific sections
        >>> summary = generator.generate_executive_summary(metrics_dict)
    """

    def __init__(self, config: ReportConfig | None = None) -> None:
        """Initialize generator.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or ReportConfig()

        logger.debug(
            "Initialized ReportGenerator with format=%s",
            self.config.format,
        )

    def generate(
        self,
        metrics_results: dict[str, Any] | None = None,
        comparison_result: Any | None = None,
        drift_results: dict[str, Any] | None = None,
        time_analysis: dict[str, Any] | None = None,
        custom_data: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Generate complete evaluation report.

        Args:
            metrics_results: Metrics calculation results by model.
            comparison_result: Model comparison results.
            drift_results: Drift detection results.
            time_analysis: Time period analysis results.
            custom_data: Additional custom data to include.

        Returns:
            EvaluationReport with all requested sections.
        """
        sections = []

        # Executive Summary
        if self.config.include_executive_summary:
            summary_section = self._generate_executive_summary(
                metrics_results, comparison_result, drift_results
            )
            sections.append(summary_section)

        # Metrics Table
        if self.config.include_metrics_table and metrics_results:
            metrics_section = self._generate_metrics_section(metrics_results)
            sections.append(metrics_section)

        # Model Comparison
        if self.config.include_model_comparison and comparison_result:
            comparison_section = self._generate_comparison_section(comparison_result)
            sections.append(comparison_section)

        # Drift Analysis
        if self.config.include_drift_analysis and drift_results:
            drift_section = self._generate_drift_section(drift_results)
            sections.append(drift_section)

        # Time Analysis
        if self.config.include_time_analysis and time_analysis:
            time_section = self._generate_time_section(time_analysis)
            sections.append(time_section)

        # Recommendations
        if self.config.include_recommendations:
            recommendations_section = self._generate_recommendations(
                metrics_results, comparison_result, drift_results
            )
            sections.append(recommendations_section)

        # Raw Data
        if self.config.include_raw_data:
            raw_section = self._generate_raw_data_section(
                metrics_results, comparison_result, drift_results, custom_data
            )
            sections.append(raw_section)

        # Format the report
        report_format = ReportFormat(self.config.format)
        raw_content = self._format_report(sections, report_format)

        return EvaluationReport(
            title=self.config.title,
            generated_at=datetime.now(),
            sections=sections,
            format=report_format,
            raw_content=raw_content,
            metadata={
                "config": self.config.to_dict(),
                "generated_by": "ReportGenerator",
            },
        )

    def _generate_executive_summary(
        self,
        metrics_results: dict[str, Any] | None,
        comparison_result: Any | None,
        drift_results: dict[str, Any] | None,
    ) -> ReportSection:
        """Generate executive summary section.

        Args:
            metrics_results: Metrics calculation results.
            comparison_result: Comparison results.
            drift_results: Drift detection results.

        Returns:
            ReportSection for executive summary.
        """
        summary_lines = [
            f"Report generated on {datetime.now().strftime(self.config.date_format)}",
            "",
        ]

        # Model count
        if metrics_results:
            n_models = len(metrics_results)
            summary_lines.append(f"Models evaluated: {n_models}")

        # Best model
        if comparison_result and hasattr(comparison_result, "best_model"):
            summary_lines.append(f"Best performing model: {comparison_result.best_model}")

        # Drift status
        if drift_results:
            drift_detected = any(
                r.get("drift_detected", False) if isinstance(r, dict) else getattr(r, "drift_detected", False)
                for r in drift_results.values()
            )
            status = "DETECTED" if drift_detected else "Not detected"
            summary_lines.append(f"Drift status: {status}")

        # Key metrics
        if metrics_results:
            summary_lines.append("")
            summary_lines.append("Key Metrics Summary:")
            for model_name, result in metrics_results.items():
                if hasattr(result, "mape"):
                    mape = result.mape
                elif isinstance(result, dict):
                    mape = result.get("mape", "N/A")
                else:
                    mape = "N/A"
                summary_lines.append(f"  - {model_name}: MAPE = {mape}")

        return ReportSection(
            section_type=SectionType.EXECUTIVE_SUMMARY,
            title="Executive Summary",
            content="\n".join(summary_lines),
            metadata={"generated_at": datetime.now().isoformat()},
        )

    def _generate_metrics_section(
        self,
        metrics_results: dict[str, Any],
    ) -> ReportSection:
        """Generate metrics table section.

        Args:
            metrics_results: Metrics calculation results.

        Returns:
            ReportSection for metrics table.
        """
        # Build table data
        table_rows = []
        for model_name, result in metrics_results.items():
            if hasattr(result, "to_dict"):
                metrics = result.to_dict()
            elif isinstance(result, dict):
                metrics = result
            else:
                metrics = {}

            row = {
                "Model": model_name,
                "MAPE (%)": self._format_number(metrics.get("mape", "N/A")),
                "MAE": self._format_number(metrics.get("mae", "N/A")),
                "RMSE": self._format_number(metrics.get("rmse", "N/A")),
                "R²": self._format_number(metrics.get("r2", "N/A")),
            }
            table_rows.append(row)

        table = {
            "name": "model_metrics",
            "headers": ["Model", "MAPE (%)", "MAE", "RMSE", "R²"],
            "rows": table_rows,
        }

        content_lines = ["Performance metrics for all evaluated models:", ""]
        for row in table_rows:
            content_lines.append(
                f"{row['Model']}: MAPE={row['MAPE (%)']}, MAE={row['MAE']}, RMSE={row['RMSE']}, R²={row['R²']}"
            )

        return ReportSection(
            section_type=SectionType.METRICS_TABLE,
            title="Performance Metrics",
            content="\n".join(content_lines),
            tables=[table],
        )

    def _generate_comparison_section(
        self,
        comparison_result: Any,
    ) -> ReportSection:
        """Generate model comparison section.

        Args:
            comparison_result: Comparison results.

        Returns:
            ReportSection for model comparison.
        """
        content_lines = []

        # Best model
        if hasattr(comparison_result, "best_model"):
            content_lines.append(f"Best Model: {comparison_result.best_model}")
            content_lines.append("")

        # Rankings
        if hasattr(comparison_result, "rankings"):
            content_lines.append("Model Rankings:")
            for ranking in comparison_result.rankings:
                if hasattr(ranking, "model_name") and hasattr(ranking, "rank"):
                    metrics_str = ""
                    if hasattr(ranking, "metrics") and ranking.metrics:
                        mape = ranking.metrics.get("mape", "N/A")
                        metrics_str = f" (MAPE: {self._format_number(mape)})"
                    content_lines.append(
                        f"  {ranking.rank}. {ranking.model_name}{metrics_str}"
                    )
            content_lines.append("")

        # Pairwise comparisons
        if hasattr(comparison_result, "pairwise_comparisons"):
            content_lines.append("Pairwise Statistical Tests:")
            for comp in comparison_result.pairwise_comparisons:
                if hasattr(comp, "model_a") and hasattr(comp, "model_b"):
                    sig = "SIGNIFICANT" if comp.is_significant else "not significant"
                    better = f", better: {comp.better_model}" if comp.better_model else ""
                    content_lines.append(
                        f"  - {comp.model_a} vs {comp.model_b}: {sig}{better}"
                    )

        # Recommendation
        if hasattr(comparison_result, "recommendation") and comparison_result.recommendation:
            content_lines.append("")
            content_lines.append(f"Recommendation: {comparison_result.recommendation}")

        # Build rankings table
        rankings_table = None
        if hasattr(comparison_result, "rankings"):
            table_rows = []
            for ranking in comparison_result.rankings:
                row = {
                    "Rank": str(ranking.rank),
                    "Model": ranking.model_name,
                    "MAPE (%)": self._format_number(ranking.metrics.get("mape", "N/A")) if ranking.metrics else "N/A",
                    "Score": self._format_number(ranking.score) if hasattr(ranking, "score") else "N/A",
                }
                table_rows.append(row)

            rankings_table = {
                "name": "model_rankings",
                "headers": ["Rank", "Model", "MAPE (%)", "Score"],
                "rows": table_rows,
            }

        tables = [rankings_table] if rankings_table else []

        return ReportSection(
            section_type=SectionType.MODEL_COMPARISON,
            title="Model Comparison",
            content="\n".join(content_lines),
            tables=tables,
        )

    def _generate_drift_section(
        self,
        drift_results: dict[str, Any],
    ) -> ReportSection:
        """Generate drift analysis section.

        Args:
            drift_results: Drift detection results.

        Returns:
            ReportSection for drift analysis.
        """
        content_lines = ["Drift Detection Analysis:", ""]

        table_rows = []
        for name, result in drift_results.items():
            if hasattr(result, "drift_detected"):
                detected = result.drift_detected
                score = result.drift_score if hasattr(result, "drift_score") else "N/A"
                method = result.method.value if hasattr(result, "method") else "N/A"
            elif isinstance(result, dict):
                detected = result.get("drift_detected", False)
                score = result.get("drift_score", "N/A")
                method = result.get("method", "N/A")
            else:
                detected = False
                score = "N/A"
                method = "N/A"

            status = "DETECTED" if detected else "Not detected"
            content_lines.append(
                f"  - {name}: {status} (score: {self._format_number(score)}, method: {method})"
            )

            table_rows.append({
                "Feature/Model": name,
                "Status": status,
                "Score": self._format_number(score),
                "Method": str(method),
            })

        # Overall assessment
        any_drift = any(
            r.get("drift_detected", False) if isinstance(r, dict) else getattr(r, "drift_detected", False)
            for r in drift_results.values()
        )
        content_lines.append("")
        if any_drift:
            content_lines.append("ALERT: Drift detected in one or more features/models.")
            content_lines.append("Recommended: Review data pipeline and consider model retraining.")
        else:
            content_lines.append("No significant drift detected. Models appear stable.")

        table = {
            "name": "drift_analysis",
            "headers": ["Feature/Model", "Status", "Score", "Method"],
            "rows": table_rows,
        }

        return ReportSection(
            section_type=SectionType.DRIFT_ANALYSIS,
            title="Drift Detection Analysis",
            content="\n".join(content_lines),
            tables=[table],
        )

    def _generate_time_section(
        self,
        time_analysis: dict[str, Any],
    ) -> ReportSection:
        """Generate time period analysis section.

        Args:
            time_analysis: Time period analysis results.

        Returns:
            ReportSection for time analysis.
        """
        content_lines = ["Time Period Performance Analysis:", ""]

        for period_type, result in time_analysis.items():
            content_lines.append(f"By {period_type}:")

            if hasattr(result, "metrics_by_period"):
                for period_name, metrics in result.metrics_by_period.items():
                    mape = metrics.mape if hasattr(metrics, "mape") else metrics.get("mape", "N/A")
                    content_lines.append(
                        f"  - {period_name}: MAPE = {self._format_number(mape)}"
                    )
            elif isinstance(result, dict):
                for period_name, metrics in result.items():
                    if isinstance(metrics, dict):
                        mape = metrics.get("mape", "N/A")
                        content_lines.append(
                            f"  - {period_name}: MAPE = {self._format_number(mape)}"
                        )

            content_lines.append("")

        return ReportSection(
            section_type=SectionType.TIME_ANALYSIS,
            title="Time Period Analysis",
            content="\n".join(content_lines),
        )

    def _generate_recommendations(
        self,
        metrics_results: dict[str, Any] | None,
        comparison_result: Any | None,
        drift_results: dict[str, Any] | None,
    ) -> ReportSection:
        """Generate recommendations section.

        Args:
            metrics_results: Metrics results.
            comparison_result: Comparison results.
            drift_results: Drift results.

        Returns:
            ReportSection for recommendations.
        """
        recommendations = []

        # Model selection recommendation
        if comparison_result and hasattr(comparison_result, "best_model"):
            best = comparison_result.best_model
            recommendations.append(f"1. Deploy '{best}' as the primary model for production.")

            # Check if significantly better
            if hasattr(comparison_result, "rankings") and len(comparison_result.rankings) > 0:
                top = comparison_result.rankings[0]
                if hasattr(top, "is_significantly_best") and top.is_significantly_best:
                    recommendations.append(
                        f"   ('{best}' is statistically significantly better than alternatives)"
                    )

        # Drift-related recommendations
        if drift_results:
            any_drift = any(
                r.get("drift_detected", False) if isinstance(r, dict) else getattr(r, "drift_detected", False)
                for r in drift_results.values()
            )
            if any_drift:
                recommendations.append("")
                recommendations.append("2. Address detected drift:")
                recommendations.append("   - Review recent data quality changes")
                recommendations.append("   - Consider retraining affected models")
                recommendations.append("   - Set up automated drift monitoring")

        # Performance improvement recommendations
        if metrics_results:
            mapes = []
            for name, result in metrics_results.items():
                if hasattr(result, "mape") and result.mape is not None:
                    mapes.append((name, result.mape))
                elif isinstance(result, dict) and result.get("mape") is not None:
                    mapes.append((name, result["mape"]))

            if mapes:
                avg_mape = np.mean([m[1] for m in mapes])
                if avg_mape > 10:
                    recommendations.append("")
                    recommendations.append("3. Performance Improvement:")
                    recommendations.append(
                        f"   - Average MAPE ({self._format_number(avg_mape)}%) exceeds 10% target"
                    )
                    recommendations.append("   - Consider feature engineering improvements")
                    recommendations.append("   - Explore additional model architectures")

        # General recommendations
        recommendations.append("")
        recommendations.append("General Recommendations:")
        recommendations.append("   - Schedule regular model performance reviews")
        recommendations.append("   - Maintain model version documentation")
        recommendations.append("   - Implement A/B testing for model updates")

        return ReportSection(
            section_type=SectionType.RECOMMENDATIONS,
            title="Recommendations",
            content="\n".join(recommendations),
        )

    def _generate_raw_data_section(
        self,
        metrics_results: dict[str, Any] | None,
        comparison_result: Any | None,
        drift_results: dict[str, Any] | None,
        custom_data: dict[str, Any] | None,
    ) -> ReportSection:
        """Generate raw data section.

        Args:
            metrics_results: Metrics results.
            comparison_result: Comparison results.
            drift_results: Drift results.
            custom_data: Custom data.

        Returns:
            ReportSection for raw data.
        """
        import json

        raw_data = {}

        if metrics_results:
            raw_data["metrics_results"] = {
                name: result.to_dict() if hasattr(result, "to_dict") else result
                for name, result in metrics_results.items()
            }

        if comparison_result:
            raw_data["comparison_result"] = (
                comparison_result.to_dict()
                if hasattr(comparison_result, "to_dict")
                else str(comparison_result)
            )

        if drift_results:
            raw_data["drift_results"] = {
                name: result.to_dict() if hasattr(result, "to_dict") else result
                for name, result in drift_results.items()
            }

        if custom_data:
            raw_data["custom_data"] = custom_data

        content = json.dumps(raw_data, indent=2, default=str)

        return ReportSection(
            section_type=SectionType.RAW_DATA,
            title="Raw Data",
            content=content,
        )

    def _format_report(
        self,
        sections: list[ReportSection],
        report_format: ReportFormat,
    ) -> str:
        """Format report content based on format type.

        Args:
            sections: List of report sections.
            report_format: Output format.

        Returns:
            Formatted report content.
        """
        if report_format == ReportFormat.HTML:
            return self._format_html(sections)
        elif report_format == ReportFormat.MARKDOWN:
            return self._format_markdown(sections)
        elif report_format == ReportFormat.JSON:
            return self._format_json(sections)
        elif report_format == ReportFormat.CSV:
            return self._format_csv(sections)
        else:  # TEXT
            return self._format_text(sections)

    def _format_html(self, sections: list[ReportSection]) -> str:
        """Format report as HTML.

        Args:
            sections: List of report sections.

        Returns:
            HTML formatted content.
        """
        css = self.config.custom_css or self._get_default_css()

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{self.config.title}</title>",
            "<style>",
            css,
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{self.config.title}</h1>",
        ]

        for section in sections:
            html_parts.append(f"<section id='{section.section_type.value}'>")
            html_parts.append(f"<h2>{section.title}</h2>")
            html_parts.append(f"<pre>{section.content}</pre>")

            # Add tables
            for table in section.tables:
                html_parts.append("<table>")
                html_parts.append("<thead><tr>")
                for header in table.get("headers", []):
                    html_parts.append(f"<th>{header}</th>")
                html_parts.append("</tr></thead>")
                html_parts.append("<tbody>")
                for row in table.get("rows", []):
                    html_parts.append("<tr>")
                    for header in table.get("headers", []):
                        html_parts.append(f"<td>{row.get(header, '')}</td>")
                    html_parts.append("</tr>")
                html_parts.append("</tbody></table>")

            html_parts.append("</section>")

        html_parts.extend([
            f"<footer>Generated on {datetime.now().strftime(self.config.date_format)}</footer>",
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def _format_markdown(self, sections: list[ReportSection]) -> str:
        """Format report as Markdown.

        Args:
            sections: List of report sections.

        Returns:
            Markdown formatted content.
        """
        md_parts = [f"# {self.config.title}", ""]

        for section in sections:
            md_parts.append(f"## {section.title}")
            md_parts.append("")
            md_parts.append(section.content)
            md_parts.append("")

            # Add tables
            for table in section.tables:
                headers = table.get("headers", [])
                if headers:
                    md_parts.append("| " + " | ".join(headers) + " |")
                    md_parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for row in table.get("rows", []):
                        values = [str(row.get(h, "")) for h in headers]
                        md_parts.append("| " + " | ".join(values) + " |")
                    md_parts.append("")

        md_parts.append(f"---\n*Generated on {datetime.now().strftime(self.config.date_format)}*")

        return "\n".join(md_parts)

    def _format_json(self, sections: list[ReportSection]) -> str:
        """Format report as JSON.

        Args:
            sections: List of report sections.

        Returns:
            JSON formatted content.
        """
        import json

        report_data = {
            "title": self.config.title,
            "generated_at": datetime.now().isoformat(),
            "sections": [s.to_dict() for s in sections],
        }

        return json.dumps(report_data, indent=2, default=str)

    def _format_csv(self, sections: list[ReportSection]) -> str:
        """Format report as CSV.

        Exports table data as CSV format.

        Args:
            sections: List of report sections.

        Returns:
            CSV formatted content.
        """
        csv_parts = []

        for section in sections:
            for table in section.tables:
                headers = table.get("headers", [])
                if headers:
                    csv_parts.append(f"# {section.title} - {table.get('name', 'table')}")
                    csv_parts.append(",".join(headers))
                    for row in table.get("rows", []):
                        values = [str(row.get(h, "")).replace(",", ";") for h in headers]
                        csv_parts.append(",".join(values))
                    csv_parts.append("")

        return "\n".join(csv_parts)

    def _format_text(self, sections: list[ReportSection]) -> str:
        """Format report as plain text.

        Args:
            sections: List of report sections.

        Returns:
            Plain text formatted content.
        """
        text_parts = [
            "=" * 60,
            self.config.title.center(60),
            "=" * 60,
            "",
        ]

        for section in sections:
            text_parts.append("-" * 60)
            text_parts.append(section.title)
            text_parts.append("-" * 60)
            text_parts.append(section.content)
            text_parts.append("")

        text_parts.append("=" * 60)
        text_parts.append(f"Generated on {datetime.now().strftime(self.config.date_format)}")

        return "\n".join(text_parts)

    def _get_default_css(self) -> str:
        """Get default CSS for HTML reports.

        Returns:
            CSS string.
        """
        return """
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        pre {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        footer {
            margin-top: 40px;
            color: #777;
            font-size: 0.9em;
        }
        section {
            margin-bottom: 30px;
        }
        """

    def _format_number(self, value: Any) -> str:
        """Format a numeric value.

        Args:
            value: Value to format.

        Returns:
            Formatted string.
        """
        if value is None or value == "N/A":
            return "N/A"
        try:
            return f"{float(value):.{self.config.decimal_places}f}"
        except (ValueError, TypeError):
            return str(value)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return f"ReportGenerator(format={self.config.format!r})"
