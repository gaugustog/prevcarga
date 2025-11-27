"""Display utilities for feature engineering commands.

This module provides display utilities for the feature CLI commands,
including progress bars, result formatting, and summary displays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click


@dataclass
class FeatureGenerationResult:
    """Result of feature generation operation.

    Attributes:
        plugins_used: List of plugin names that were executed.
        areas_processed: List of areas that features were generated for.
        features_generated: Total number of features generated.
        rows_processed: Number of data rows processed.
        output_path: Path where features were saved.
        output_format: Format of saved features.
        execution_time_seconds: Total execution time.
        plugin_details: Per-plugin execution details.
        success: Whether generation succeeded.
        error_message: Error message if failed.
    """

    plugins_used: list[str] = field(default_factory=list)
    areas_processed: list[str] = field(default_factory=list)
    features_generated: int = 0
    rows_processed: int = 0
    output_path: Path | None = None
    output_format: str = "parquet"
    execution_time_seconds: float = 0.0
    plugin_details: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None


@dataclass
class FeatureEvaluationResult:
    """Result of feature evaluation operation.

    Attributes:
        model_type: Model used for evaluation.
        area: Area evaluated.
        total_features: Total number of features evaluated.
        top_features: List of top feature names with importance.
        correlation_matrix: Correlation summary if computed.
        stability_metrics: Feature stability metrics.
        execution_time_seconds: Total execution time.
        success: Whether evaluation succeeded.
        error_message: Error message if failed.
    """

    model_type: str = "lgbm"
    area: str = "nacional"
    total_features: int = 0
    top_features: list[tuple[str, float]] = field(default_factory=list)
    correlation_matrix: dict[str, Any] | None = None
    stability_metrics: dict[str, float] | None = None
    execution_time_seconds: float = 0.0
    success: bool = True
    error_message: str | None = None


@dataclass
class FeatureValidationResult:
    """Result of feature validation operation.

    Attributes:
        feature_dir: Directory that was validated.
        is_valid: Overall validation result.
        total_files: Number of files checked.
        valid_files: Number of valid files.
        issues: List of validation issues found.
        schema_compliance: Schema compliance status.
        missing_value_summary: Summary of missing values.
        type_check_results: Data type validation results.
        execution_time_seconds: Total execution time.
    """

    feature_dir: Path | None = None
    is_valid: bool = True
    total_files: int = 0
    valid_files: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)
    schema_compliance: dict[str, bool] | None = None
    missing_value_summary: dict[str, float] | None = None
    type_check_results: dict[str, bool] | None = None
    execution_time_seconds: float = 0.0


class FeatureProgressBar:
    """Progress bar for feature generation operations.

    Provides visual feedback during feature generation with
    plugin-level and area-level progress tracking.
    """

    def __init__(
        self,
        plugins: list[str],
        areas: list[str],
        show_progress: bool = True,
    ) -> None:
        """Initialize feature progress bar.

        Args:
            plugins: List of plugin names to process.
            areas: List of areas to process.
            show_progress: Whether to show progress bar.
        """
        self.plugins = plugins
        self.areas = areas
        self.show_progress = show_progress
        self.total_steps = len(plugins) * len(areas)
        self.current_step = 0
        self._progress_bar: Any = None

    def __enter__(self) -> FeatureProgressBar:
        """Enter context manager."""
        if self.show_progress and self.total_steps > 0:
            self._progress_bar = click.progressbar(
                length=self.total_steps,
                label="Generating features",
                show_eta=True,
                show_percent=True,
                item_show_func=lambda x: x if x else "",
            )
            self._progress_bar.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        if self._progress_bar is not None:
            self._progress_bar.__exit__(exc_type, exc_val, exc_tb)

    def update(
        self,
        plugin: str,
        area: str,
        status: str = "processing",
    ) -> None:
        """Update progress bar.

        Args:
            plugin: Current plugin being processed.
            area: Current area being processed.
            status: Current status message.
        """
        del status  # Unused but part of callback interface
        self.current_step += 1
        if self._progress_bar is not None:
            self._progress_bar.update(1, f"{plugin}/{area}")


class FeatureResultDisplay:
    """Helper class for displaying feature operation results."""

    # Importance thresholds for display coloring
    HIGH_IMPORTANCE_THRESHOLD = 0.1
    MEDIUM_IMPORTANCE_THRESHOLD = 0.01

    @staticmethod
    def format_importance(importance: float) -> str:
        """Format importance score for display.

        Args:
            importance: Importance score (0-1 typically).

        Returns:
            Formatted importance string.
        """
        if importance >= FeatureResultDisplay.HIGH_IMPORTANCE_THRESHOLD:
            return click.style(f"{importance:.4f}", fg="green", bold=True)
        if importance >= FeatureResultDisplay.MEDIUM_IMPORTANCE_THRESHOLD:
            return click.style(f"{importance:.4f}", fg="yellow")
        return click.style(f"{importance:.4f}", fg="white")

    @staticmethod
    def format_validation_status(is_valid: bool) -> str:
        """Format validation status for display.

        Args:
            is_valid: Whether validation passed.

        Returns:
            Formatted status string.
        """
        if is_valid:
            return click.style("✓ PASS", fg="green", bold=True)
        return click.style("✗ FAIL", fg="red", bold=True)


def display_generation_plan(
    plugins: list[str],
    areas: list[str],
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    output_format: str,
    parallel_workers: int,
) -> None:
    """Display the feature generation plan.

    Args:
        plugins: List of plugins to use.
        areas: List of areas to process.
        start_date: Start date for feature generation.
        end_date: End date for feature generation.
        output_dir: Output directory path.
        output_format: Output file format.
        parallel_workers: Number of parallel workers.
    """
    click.echo()
    click.echo("╔════════════════════════════════════════════════════════════╗")
    click.echo("║              Feature Generation Plan                        ║")
    click.echo("╚════════════════════════════════════════════════════════════╝")
    click.echo()
    click.echo(f"  Plugins:         {', '.join(plugins)}")
    click.echo(
        f"  Areas:           {len(areas)} ({', '.join(areas[:3])}{'...' if len(areas) > 3 else ''})"
    )
    click.echo(
        f"  Date range:      {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    click.echo(f"  Output dir:      {output_dir}")
    click.echo(f"  Output format:   {output_format}")
    click.echo(f"  Parallel workers: {parallel_workers}")
    click.echo()


def display_generation_summary(result: FeatureGenerationResult) -> None:
    """Display feature generation summary.

    Args:
        result: Feature generation result to display.
    """
    click.echo()
    click.echo("╔════════════════════════════════════════════════════════════╗")
    click.echo("║              Feature Generation Summary                     ║")
    click.echo("╚════════════════════════════════════════════════════════════╝")
    click.echo()

    if result.success:
        click.secho("  Status: SUCCESS", fg="green", bold=True)
    else:
        click.secho("  Status: FAILED", fg="red", bold=True)
        if result.error_message:
            click.echo(f"  Error: {result.error_message}")

    click.echo()
    click.echo(f"  Plugins executed:    {len(result.plugins_used)}")
    click.echo(f"  Areas processed:     {len(result.areas_processed)}")
    click.echo(f"  Features generated:  {result.features_generated}")
    click.echo(f"  Rows processed:      {result.rows_processed:,}")
    click.echo(f"  Execution time:      {result.execution_time_seconds:.2f}s")

    if result.output_path:
        click.echo(f"  Output saved to:     {result.output_path}")

    if result.plugin_details:
        click.echo()
        click.echo("  Plugin Details:")
        click.echo("  " + "-" * 56)
        for detail in result.plugin_details:
            name = detail.get("name", "unknown")
            features = detail.get("features", 0)
            time_ms = detail.get("time_ms", 0)
            status = "✓" if detail.get("success", True) else "✗"
            click.echo(f"    {status} {name}: {features} features ({time_ms:.1f}ms)")

    click.echo()


def display_evaluation_results(
    result: FeatureEvaluationResult,
    top_n: int = 20,
) -> None:
    """Display feature evaluation results.

    Args:
        result: Feature evaluation result to display.
        top_n: Number of top features to show.
    """
    click.echo()
    click.echo("╔════════════════════════════════════════════════════════════╗")
    click.echo("║              Feature Evaluation Results                     ║")
    click.echo("╚════════════════════════════════════════════════════════════╝")
    click.echo()

    click.echo(f"  Model type:       {result.model_type}")
    click.echo(f"  Area:             {result.area}")
    click.echo(f"  Total features:   {result.total_features}")
    click.echo(f"  Execution time:   {result.execution_time_seconds:.2f}s")
    click.echo()

    if result.top_features:
        click.echo(f"  Top {min(top_n, len(result.top_features))} Features by Importance:")
        click.echo("  " + "-" * 56)
        click.echo(f"  {'Rank':<6} {'Feature Name':<35} {'Importance':<12}")
        click.echo("  " + "-" * 56)

        display = FeatureResultDisplay()
        for i, (name, importance) in enumerate(result.top_features[:top_n], 1):
            formatted_imp = display.format_importance(importance)
            click.echo(f"  {i:<6} {name:<35} {formatted_imp}")

    if result.stability_metrics:
        click.echo()
        click.echo("  Stability Metrics:")
        click.echo("  " + "-" * 56)
        for metric, value in result.stability_metrics.items():
            click.echo(f"    {metric}: {value:.4f}")

    click.echo()


def display_validation_results(result: FeatureValidationResult) -> None:
    """Display feature validation results.

    Args:
        result: Feature validation result to display.
    """
    click.echo()
    click.echo("╔════════════════════════════════════════════════════════════╗")
    click.echo("║              Feature Validation Results                     ║")
    click.echo("╚════════════════════════════════════════════════════════════╝")
    click.echo()

    display = FeatureResultDisplay()
    status = display.format_validation_status(result.is_valid)
    click.echo(f"  Overall Status:   {status}")
    click.echo()

    click.echo(f"  Directory:        {result.feature_dir}")
    click.echo(f"  Total files:      {result.total_files}")
    click.echo(f"  Valid files:      {result.valid_files}")
    click.echo(f"  Execution time:   {result.execution_time_seconds:.2f}s")
    click.echo()

    if result.schema_compliance:
        click.echo("  Schema Compliance:")
        click.echo("  " + "-" * 56)
        for check, passed in result.schema_compliance.items():
            status_icon = "✓" if passed else "✗"
            color = "green" if passed else "red"
            click.secho(f"    {status_icon} {check}", fg=color)

    if result.missing_value_summary:
        click.echo()
        click.echo("  Missing Value Summary:")
        click.echo("  " + "-" * 56)
        for col, pct in result.missing_value_summary.items():
            if pct > 0:
                color = "red" if pct > 10 else "yellow" if pct > 1 else "white"
                click.secho(f"    {col}: {pct:.2f}%", fg=color)

    if result.issues:
        click.echo()
        click.echo("  Issues Found:")
        click.echo("  " + "-" * 56)
        for issue in result.issues[:10]:
            severity = issue.get("severity", "warning")
            message = issue.get("message", "Unknown issue")
            file_name = issue.get("file", "")
            color = "red" if severity == "error" else "yellow"
            click.secho(f"    [{severity.upper()}] {file_name}: {message}", fg=color)

        if len(result.issues) > 10:
            click.echo(f"    ... and {len(result.issues) - 10} more issues")

    click.echo()


def save_evaluation_report(
    result: FeatureEvaluationResult,
    output_path: Path,
    format: str = "json",
) -> None:
    """Save evaluation report to file.

    Args:
        result: Evaluation result to save.
        output_path: Output file path.
        format: Output format (json, csv, html).
    """
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = {
            "model_type": result.model_type,
            "area": result.area,
            "total_features": result.total_features,
            "top_features": [
                {"name": name, "importance": imp} for name, imp in result.top_features
            ],
            "stability_metrics": result.stability_metrics,
            "execution_time_seconds": result.execution_time_seconds,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "csv":
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "feature_name", "importance"])
            for i, (name, imp) in enumerate(result.top_features, 1):
                writer.writerow([i, name, imp])

    elif format == "html":
        html_content = _generate_evaluation_html(result)
        with open(output_path, "w") as f:
            f.write(html_content)


def _generate_evaluation_html(result: FeatureEvaluationResult) -> str:
    """Generate HTML report for evaluation results.

    Args:
        result: Evaluation result to convert to HTML.

    Returns:
        HTML string.
    """
    rows = "\n".join(
        f"<tr><td>{i}</td><td>{name}</td><td>{imp:.6f}</td></tr>"
        for i, (name, imp) in enumerate(result.top_features, 1)
    )

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Feature Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Feature Evaluation Report</h1>
    <div class="summary">
        <p><strong>Model:</strong> {result.model_type}</p>
        <p><strong>Area:</strong> {result.area}</p>
        <p><strong>Total Features:</strong> {result.total_features}</p>
        <p><strong>Generated:</strong> {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
    <h2>Feature Importance Rankings</h2>
    <table>
        <tr><th>Rank</th><th>Feature Name</th><th>Importance</th></tr>
        {rows}
    </table>
</body>
</html>"""
