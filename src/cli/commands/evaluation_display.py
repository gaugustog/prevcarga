"""Evaluation result display utilities for CLI.

This module provides display utilities for evaluation commands,
including metrics display, drift detection results, and report generation.

Key Components:
- EvaluationDisplayConfig: Configuration for display options
- MetricsDisplay: Display for metrics results
- DriftDisplay: Display for drift detection results
- ComparisonDisplay: Display for model comparison results

Example:
    ```python
    from src.cli.commands.evaluation_display import MetricsDisplay

    display = MetricsDisplay(output_format=OutputFormat.TEXT)
    display.show_metrics(metrics_result)
    display.show_summary(summary_data)
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import click

from src.cli.core import OutputFormat


class DisplayTheme(Enum):
    """Display theme options.

    Attributes:
        DEFAULT: Default theme with colors.
        MINIMAL: Minimal output without decorations.
        DETAILED: Detailed output with all information.
    """

    DEFAULT = "default"
    MINIMAL = "minimal"
    DETAILED = "detailed"


@dataclass
class EvaluationDisplayConfig:
    """Configuration for evaluation display.

    Attributes:
        output_format: Output format (text, json, table).
        theme: Display theme.
        show_timestamps: Whether to show timestamps.
        decimal_places: Number of decimal places for metrics.
        show_confidence_intervals: Whether to show confidence intervals.
    """

    output_format: OutputFormat = OutputFormat.TEXT
    theme: DisplayTheme = DisplayTheme.DEFAULT
    show_timestamps: bool = True
    decimal_places: int = 4
    show_confidence_intervals: bool = True


@dataclass
class MetricsDisplayResult:
    """Result container for metrics display.

    Attributes:
        model_name: Name of the model.
        mape: Mean Absolute Percentage Error.
        smape: Symmetric MAPE.
        mae: Mean Absolute Error.
        rmse: Root Mean Square Error.
        r2: R-squared score.
        mape_ci: MAPE confidence interval (lower, upper).
        rmse_ci: RMSE confidence interval (lower, upper).
        timestamp: When metrics were calculated.
    """

    model_name: str
    mape: float
    smape: float
    mae: float
    rmse: float
    r2: float
    mape_ci: tuple[float, float] | None = None
    rmse_ci: tuple[float, float] | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class DriftDisplayResult:
    """Result container for drift detection display.

    Attributes:
        feature_name: Name of the feature.
        drift_detected: Whether drift was detected.
        drift_score: Drift score/statistic.
        p_value: Statistical p-value.
        method: Detection method used.
        severity: Drift severity level.
        recommendation: Recommended action.
    """

    feature_name: str
    drift_detected: bool
    drift_score: float
    p_value: float | None
    method: str
    severity: str = "low"
    recommendation: str = ""


@dataclass
class ComparisonDisplayResult:
    """Result container for model comparison display.

    Attributes:
        model_a: First model name.
        model_b: Second model name.
        metric: Comparison metric.
        model_a_value: Metric value for model A.
        model_b_value: Metric value for model B.
        statistically_significant: Whether difference is significant.
        p_value: Statistical p-value.
        winner: Name of the winning model.
        effect_size: Cohen's d effect size.
    """

    model_a: str
    model_b: str
    metric: str
    model_a_value: float
    model_b_value: float
    statistically_significant: bool
    p_value: float | None
    winner: str | None
    effect_size: float | None = None


class MetricsDisplay:
    """Display helper for metrics results."""

    def __init__(self, config: EvaluationDisplayConfig | None = None) -> None:
        """Initialize metrics display.

        Args:
            config: Display configuration.
        """
        self.config = config or EvaluationDisplayConfig()
        self._dp = self.config.decimal_places

    def _format_metric(self, value: float, suffix: str = "") -> str:
        """Format a metric value.

        Args:
            value: Metric value.
            suffix: Optional suffix (e.g., '%').

        Returns:
            Formatted string.
        """
        return f"{value:.{self._dp}f}{suffix}"

    def _format_ci(self, ci: tuple[float, float] | None, suffix: str = "") -> str:
        """Format confidence interval.

        Args:
            ci: Confidence interval (lower, upper).
            suffix: Optional suffix.

        Returns:
            Formatted string.
        """
        if ci is None:
            return "N/A"
        return f"[{ci[0]:.{self._dp}f}, {ci[1]:.{self._dp}f}]{suffix}"

    def show_header(self, title: str) -> None:
        """Display section header.

        Args:
            title: Header title.
        """
        if self.config.output_format == OutputFormat.TEXT:
            click.echo()
            click.echo(click.style(f"{'=' * 60}", fg="cyan"))
            click.echo(click.style(f"  {title}", fg="cyan", bold=True))
            click.echo(click.style(f"{'=' * 60}", fg="cyan"))
            click.echo()

    def show_metrics_table(
        self,
        results: list[MetricsDisplayResult],
        title: str = "Metrics Summary",
    ) -> None:
        """Display metrics in a table format.

        Args:
            results: List of metrics results.
            title: Table title.
        """
        if self.config.output_format == OutputFormat.QUIET:
            return

        if self.config.output_format == OutputFormat.JSON:
            import json
            data = [
                {
                    "model": r.model_name,
                    "mape": r.mape,
                    "smape": r.smape,
                    "mae": r.mae,
                    "rmse": r.rmse,
                    "r2": r.r2,
                    "mape_ci": r.mape_ci,
                    "rmse_ci": r.rmse_ci,
                }
                for r in results
            ]
            click.echo(json.dumps({"title": title, "metrics": data}, indent=2))
            return

        self.show_header(title)

        # Build header row
        headers = ["Model", "MAPE", "sMAPE", "MAE", "RMSE", "R2"]
        if self.config.show_confidence_intervals:
            headers.extend(["MAPE CI", "RMSE CI"])

        # Calculate column widths
        widths = [len(h) for h in headers]
        rows: list[list[str]] = []

        for result in results:
            row = [
                result.model_name[:20],
                self._format_metric(result.mape, "%"),
                self._format_metric(result.smape, "%"),
                self._format_metric(result.mae),
                self._format_metric(result.rmse),
                self._format_metric(result.r2),
            ]
            if self.config.show_confidence_intervals:
                row.extend([
                    self._format_ci(result.mape_ci, "%"),
                    self._format_ci(result.rmse_ci),
                ])
            rows.append(row)
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        # Print header
        header_str = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        click.echo(click.style(header_str, bold=True))
        click.echo("-" * len(header_str))

        # Print rows
        for row in rows:
            row_str = " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            click.echo(row_str)

        click.echo()

    def show_single_model_metrics(self, result: MetricsDisplayResult) -> None:
        """Display metrics for a single model.

        Args:
            result: Metrics result.
        """
        if self.config.output_format == OutputFormat.QUIET:
            return

        if self.config.output_format == OutputFormat.JSON:
            import json
            data = {
                "model": result.model_name,
                "mape": result.mape,
                "smape": result.smape,
                "mae": result.mae,
                "rmse": result.rmse,
                "r2": result.r2,
                "mape_ci": result.mape_ci,
                "rmse_ci": result.rmse_ci,
            }
            click.echo(json.dumps(data, indent=2))
            return

        click.echo(click.style(f"Model: {result.model_name}", bold=True))
        click.echo(f"  MAPE:  {self._format_metric(result.mape, '%')}", nl=False)
        if self.config.show_confidence_intervals and result.mape_ci:
            click.echo(f"  CI: {self._format_ci(result.mape_ci, '%')}")
        else:
            click.echo()

        click.echo(f"  sMAPE: {self._format_metric(result.smape, '%')}")
        click.echo(f"  MAE:   {self._format_metric(result.mae)}")
        click.echo(f"  RMSE:  {self._format_metric(result.rmse)}", nl=False)
        if self.config.show_confidence_intervals and result.rmse_ci:
            click.echo(f"  CI: {self._format_ci(result.rmse_ci)}")
        else:
            click.echo()
        click.echo(f"  R2:    {self._format_metric(result.r2)}")
        click.echo()


class DriftDisplay:
    """Display helper for drift detection results."""

    def __init__(self, config: EvaluationDisplayConfig | None = None) -> None:
        """Initialize drift display.

        Args:
            config: Display configuration.
        """
        self.config = config or EvaluationDisplayConfig()

    def show_drift_summary(
        self,
        results: list[DriftDisplayResult],
        title: str = "Drift Detection Results",
    ) -> None:
        """Display drift detection summary.

        Args:
            results: List of drift results.
            title: Summary title.
        """
        if self.config.output_format == OutputFormat.QUIET:
            return

        if self.config.output_format == OutputFormat.JSON:
            import json
            data = [
                {
                    "feature": r.feature_name,
                    "drift_detected": r.drift_detected,
                    "drift_score": r.drift_score,
                    "p_value": r.p_value,
                    "method": r.method,
                    "severity": r.severity,
                    "recommendation": r.recommendation,
                }
                for r in results
            ]
            click.echo(json.dumps({"title": title, "drift_results": data}, indent=2))
            return

        # Text format
        click.echo()
        click.echo(click.style(f"{'=' * 60}", fg="yellow"))
        click.echo(click.style(f"  {title}", fg="yellow", bold=True))
        click.echo(click.style(f"{'=' * 60}", fg="yellow"))
        click.echo()

        drift_count = sum(1 for r in results if r.drift_detected)
        click.echo(f"Features analyzed: {len(results)}")
        click.echo(f"Drift detected in: {drift_count} features")
        click.echo()

        if drift_count > 0:
            click.echo(click.style("Features with drift:", bold=True))
            for result in results:
                if result.drift_detected:
                    severity_color = {
                        "low": "yellow",
                        "medium": "bright_yellow",
                        "high": "red",
                        "critical": "bright_red",
                    }.get(result.severity, "yellow")

                    click.echo(f"  {click.style('!', fg=severity_color)} {result.feature_name}")
                    click.echo(f"    Method: {result.method}")
                    click.echo(f"    Score: {result.drift_score:.4f}")
                    if result.p_value is not None:
                        click.echo(f"    P-value: {result.p_value:.4f}")
                    click.echo(f"    Severity: {click.style(result.severity, fg=severity_color)}")
                    if result.recommendation:
                        click.echo(f"    Recommendation: {result.recommendation}")
                    click.echo()

    def show_drift_alert(self, result: DriftDisplayResult) -> None:
        """Display a single drift alert.

        Args:
            result: Drift result to display.
        """
        if not result.drift_detected:
            return

        severity_color = {
            "low": "yellow",
            "medium": "bright_yellow",
            "high": "red",
            "critical": "bright_red",
        }.get(result.severity, "yellow")

        icon = "!" if result.severity in ["low", "medium"] else "!!"
        click.echo(
            click.style(
                f"{icon} Drift Alert: {result.feature_name} ({result.severity})",
                fg=severity_color,
            )
        )


class ComparisonDisplay:
    """Display helper for model comparison results."""

    def __init__(self, config: EvaluationDisplayConfig | None = None) -> None:
        """Initialize comparison display.

        Args:
            config: Display configuration.
        """
        self.config = config or EvaluationDisplayConfig()

    def show_comparison_table(
        self,
        results: list[ComparisonDisplayResult],
        title: str = "Model Comparison",
    ) -> None:
        """Display model comparison table.

        Args:
            results: List of comparison results.
            title: Table title.
        """
        if self.config.output_format == OutputFormat.QUIET:
            return

        if self.config.output_format == OutputFormat.JSON:
            import json
            data = [
                {
                    "model_a": r.model_a,
                    "model_b": r.model_b,
                    "metric": r.metric,
                    "model_a_value": r.model_a_value,
                    "model_b_value": r.model_b_value,
                    "statistically_significant": r.statistically_significant,
                    "p_value": r.p_value,
                    "winner": r.winner,
                    "effect_size": r.effect_size,
                }
                for r in results
            ]
            click.echo(json.dumps({"title": title, "comparisons": data}, indent=2))
            return

        click.echo()
        click.echo(click.style(f"{'=' * 70}", fg="magenta"))
        click.echo(click.style(f"  {title}", fg="magenta", bold=True))
        click.echo(click.style(f"{'=' * 70}", fg="magenta"))
        click.echo()

        headers = ["Models", "Metric", "A Value", "B Value", "Winner", "Sig.", "P-Value"]
        widths = [len(h) for h in headers]
        rows: list[list[str]] = []

        for result in results:
            sig_str = "Yes" if result.statistically_significant else "No"
            p_str = f"{result.p_value:.4f}" if result.p_value is not None else "N/A"
            winner_str = result.winner or "Tie"

            row = [
                f"{result.model_a} vs {result.model_b}",
                result.metric,
                f"{result.model_a_value:.4f}",
                f"{result.model_b_value:.4f}",
                winner_str,
                sig_str,
                p_str,
            ]
            rows.append(row)
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        # Print header
        header_str = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        click.echo(click.style(header_str, bold=True))
        click.echo("-" * len(header_str))

        # Print rows with color for winner
        for row_idx, row in enumerate(rows):
            result = results[row_idx]
            row_parts = []
            for i, cell in enumerate(row):
                if i == 4 and result.winner:  # Winner column
                    row_parts.append(click.style(cell.ljust(widths[i]), fg="green"))
                else:
                    row_parts.append(cell.ljust(widths[i]))
            click.echo(" | ".join(row_parts))

        click.echo()

    def show_ranking(
        self,
        rankings: list[tuple[str, float, int]],
        title: str = "Model Rankings",
    ) -> None:
        """Display model rankings.

        Args:
            rankings: List of (model_name, score, rank) tuples.
            title: Rankings title.
        """
        if self.config.output_format == OutputFormat.QUIET:
            return

        if self.config.output_format == OutputFormat.JSON:
            import json
            data = [
                {"model": name, "score": score, "rank": rank}
                for name, score, rank in rankings
            ]
            click.echo(json.dumps({"title": title, "rankings": data}, indent=2))
            return

        click.echo()
        click.echo(click.style(f"  {title}", bold=True))
        click.echo("-" * 40)

        for model, score, rank in rankings:
            rank_color = {1: "green", 2: "yellow", 3: "blue"}.get(rank, "white")
            medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
            click.echo(
                f"  {click.style(medal.ljust(4), fg=rank_color)} "
                f"{model}: {score:.4f}"
            )
        click.echo()


def display_metrics_plan(
    models: list[str],
    areas: list[str],
    horizons: list[int],
    metrics: list[str],
    output_format: OutputFormat = OutputFormat.TEXT,
) -> None:
    """Display evaluation metrics plan.

    Args:
        models: List of model names.
        areas: List of area codes.
        horizons: List of forecast horizons.
        metrics: List of metrics to calculate.
        output_format: Output format.
    """
    if output_format == OutputFormat.QUIET:
        return

    if output_format == OutputFormat.JSON:
        import json
        plan = {
            "models": models,
            "areas": areas,
            "horizons": horizons,
            "metrics": metrics,
        }
        click.echo(json.dumps({"evaluation_plan": plan}, indent=2))
        return

    click.echo()
    click.echo(click.style("Evaluation Plan", bold=True, fg="cyan"))
    click.echo("-" * 40)
    click.echo(f"  Models:   {', '.join(models)}")
    click.echo(f"  Areas:    {', '.join(areas)}")
    click.echo(f"  Horizons: {', '.join(f'D+{h}' for h in horizons)}")
    click.echo(f"  Metrics:  {', '.join(metrics)}")
    click.echo()


def display_evaluation_summary(
    total_evaluations: int,
    successful: int,
    failed: int,
    duration_seconds: float,
    output_format: OutputFormat = OutputFormat.TEXT,
) -> None:
    """Display evaluation summary.

    Args:
        total_evaluations: Total number of evaluations.
        successful: Number of successful evaluations.
        failed: Number of failed evaluations.
        duration_seconds: Total duration in seconds.
        output_format: Output format.
    """
    if output_format == OutputFormat.QUIET:
        return

    if output_format == OutputFormat.JSON:
        import json
        summary = {
            "total_evaluations": total_evaluations,
            "successful": successful,
            "failed": failed,
            "duration_seconds": duration_seconds,
        }
        click.echo(json.dumps({"summary": summary}, indent=2))
        return

    click.echo()
    click.echo(click.style("Evaluation Summary", bold=True))
    click.echo("-" * 40)
    click.echo(f"  Total Evaluations: {total_evaluations}")
    click.echo(f"  Successful:        {click.style(str(successful), fg='green')}")
    if failed > 0:
        click.echo(f"  Failed:            {click.style(str(failed), fg='red')}")
    else:
        click.echo(f"  Failed:            {failed}")
    click.echo(f"  Duration:          {duration_seconds:.2f}s")
    click.echo()


def save_evaluation_report(
    report_path: Path,
    metrics_results: list[MetricsDisplayResult],
    drift_results: list[DriftDisplayResult] | None = None,
    comparison_results: list[ComparisonDisplayResult] | None = None,
    format: str = "html",
) -> Path:
    """Save evaluation report to file.

    Args:
        report_path: Path to save report.
        metrics_results: List of metrics results.
        drift_results: Optional list of drift results.
        comparison_results: Optional list of comparison results.
        format: Report format ('html', 'json', 'markdown').

    Returns:
        Path to saved report.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        import json
        report_data: dict[str, Any] = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "metrics": [
                {
                    "model": r.model_name,
                    "mape": r.mape,
                    "smape": r.smape,
                    "mae": r.mae,
                    "rmse": r.rmse,
                    "r2": r.r2,
                }
                for r in metrics_results
            ],
        }
        if drift_results:
            report_data["drift"] = [
                {
                    "feature": r.feature_name,
                    "drift_detected": r.drift_detected,
                    "drift_score": r.drift_score,
                    "p_value": r.p_value,
                    "method": r.method,
                    "severity": r.severity,
                }
                for r in drift_results
            ]
        if comparison_results:
            report_data["comparisons"] = [
                {
                    "model_a": r.model_a,
                    "model_b": r.model_b,
                    "metric": r.metric,
                    "winner": r.winner,
                    "p_value": r.p_value,
                }
                for r in comparison_results
            ]

        with report_path.open("w") as f:
            json.dump(report_data, f, indent=2)

    elif format == "markdown":
        lines = [
            "# Evaluation Report",
            f"",
            f"Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Metrics Summary",
            "",
            "| Model | MAPE | sMAPE | MAE | RMSE | R2 |",
            "|-------|------|-------|-----|------|-----|",
        ]
        for r in metrics_results:
            lines.append(
                f"| {r.model_name} | {r.mape:.4f}% | {r.smape:.4f}% | "
                f"{r.mae:.4f} | {r.rmse:.4f} | {r.r2:.4f} |"
            )

        if drift_results:
            lines.extend([
                "",
                "## Drift Detection",
                "",
                "| Feature | Drift | Score | Method | Severity |",
                "|---------|-------|-------|--------|----------|",
            ])
            for r in drift_results:
                drift_str = "Yes" if r.drift_detected else "No"
                lines.append(
                    f"| {r.feature_name} | {drift_str} | {r.drift_score:.4f} | "
                    f"{r.method} | {r.severity} |"
                )

        with report_path.open("w") as f:
            f.write("\n".join(lines))

    else:  # HTML format
        html_content = _generate_html_report(
            metrics_results, drift_results, comparison_results
        )
        with report_path.open("w") as f:
            f.write(html_content)

    return report_path


def _generate_html_report(
    metrics_results: list[MetricsDisplayResult],
    drift_results: list[DriftDisplayResult] | None = None,
    comparison_results: list[ComparisonDisplayResult] | None = None,
) -> str:
    """Generate HTML report content.

    Args:
        metrics_results: List of metrics results.
        drift_results: Optional list of drift results.
        comparison_results: Optional list of comparison results.

    Returns:
        HTML content as string.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build metrics table
    metrics_rows = ""
    for r in metrics_results:
        metrics_rows += f"""
        <tr>
            <td>{r.model_name}</td>
            <td>{r.mape:.4f}%</td>
            <td>{r.smape:.4f}%</td>
            <td>{r.mae:.4f}</td>
            <td>{r.rmse:.4f}</td>
            <td>{r.r2:.4f}</td>
        </tr>"""

    # Build drift section
    drift_section = ""
    if drift_results:
        drift_rows = ""
        for r in drift_results:
            drift_class = "drift-yes" if r.drift_detected else "drift-no"
            drift_str = "Yes" if r.drift_detected else "No"
            drift_rows += f"""
            <tr class="{drift_class}">
                <td>{r.feature_name}</td>
                <td>{drift_str}</td>
                <td>{r.drift_score:.4f}</td>
                <td>{r.method}</td>
                <td>{r.severity}</td>
            </tr>"""

        drift_section = f"""
        <h2>Drift Detection</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Drift</th>
                    <th>Score</th>
                    <th>Method</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
                {drift_rows}
            </tbody>
        </table>"""

    # Build comparison section
    comparison_section = ""
    if comparison_results:
        comp_rows = ""
        for r in comparison_results:
            sig_str = "Yes" if r.statistically_significant else "No"
            p_str = f"{r.p_value:.4f}" if r.p_value is not None else "N/A"
            winner_str = r.winner or "Tie"
            comp_rows += f"""
            <tr>
                <td>{r.model_a} vs {r.model_b}</td>
                <td>{r.metric}</td>
                <td>{r.model_a_value:.4f}</td>
                <td>{r.model_b_value:.4f}</td>
                <td class="winner">{winner_str}</td>
                <td>{sig_str}</td>
                <td>{p_str}</td>
            </tr>"""

        comparison_section = f"""
        <h2>Model Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Models</th>
                    <th>Metric</th>
                    <th>A Value</th>
                    <th>B Value</th>
                    <th>Winner</th>
                    <th>Significant</th>
                    <th>P-Value</th>
                </tr>
            </thead>
            <tbody>
                {comp_rows}
            </tbody>
        </table>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PrevCarga Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .timestamp {{ color: #777; font-size: 0.9em; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #007bff; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #f0f0f0; }}
        .drift-yes {{ background: #fff3cd !important; }}
        .drift-no {{ background: #d4edda !important; }}
        .winner {{ color: #28a745; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>PrevCarga Evaluation Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>

    <h2>Metrics Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>MAPE</th>
                <th>sMAPE</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R2</th>
            </tr>
        </thead>
        <tbody>
            {metrics_rows}
        </tbody>
    </table>

    {drift_section}
    {comparison_section}
</body>
</html>"""

    return html
