"""Core CLI structure for PrevCarga.

This module provides the main command-line interface for the PrevCarga
electric load forecasting system, with commands for training, prediction,
backtesting, and system management.

Key Components:
- Main CLI application with Click
- Command groups for different functionality
- Common options and utilities
- Output formatting helpers

Example:
    ```bash
    # Show help
    prevcarga --help

    # Train models
    prevcarga train --config config/prevcarga.yaml

    # Generate predictions
    prevcarga predict --areas SECO,S --horizons 0,1,2

    # Run backtesting
    prevcarga backtest --start 2023-01-01 --end 2023-12-31
    ```
"""

import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import click

from src.orchestration import (
    BacktestingWorkflow,
    ConfigManager,
    PredictionWorkflow,
    TrainingWorkflow,
)
from src.orchestration.structured_logger import (
    LogFormat,
    LogLevel,
    StructuredLogger,
    create_workflow_logger,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OutputFormat(Enum):
    """CLI output format options.

    Attributes:
        TEXT: Human-readable text format.
        JSON: Machine-readable JSON format.
        TABLE: Tabular format with headers.
        QUIET: Minimal output.
    """

    TEXT = "text"
    JSON = "json"
    TABLE = "table"
    QUIET = "quiet"


class Verbosity(Enum):
    """CLI verbosity levels.

    Attributes:
        QUIET: Minimal output.
        NORMAL: Standard output.
        VERBOSE: Detailed output.
        DEBUG: Debug-level output.
    """

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


# Common option decorators
def common_options(func: Callable) -> Callable:
    """Decorator to add common options to commands.

    Args:
        func: Command function to decorate.

    Returns:
        Decorated function.
    """
    func = click.option(
        "-c", "--config",
        type=click.Path(exists=True, path_type=Path),
        default="config/prevcarga.yaml",
        help="Path to configuration file",
    )(func)
    func = click.option(
        "-v", "--verbose",
        count=True,
        help="Increase verbosity (-v, -vv, -vvv)",
    )(func)
    func = click.option(
        "-q", "--quiet",
        is_flag=True,
        help="Suppress non-essential output",
    )(func)
    func = click.option(
        "-o", "--output-format",
        type=click.Choice(["text", "json", "table", "quiet"]),
        default="text",
        help="Output format",
    )(func)
    return func


def workflow_options(func: Callable) -> Callable:
    """Decorator to add workflow-specific options.

    Args:
        func: Command function to decorate.

    Returns:
        Decorated function.
    """
    func = click.option(
        "--areas",
        type=str,
        default=None,
        help="Comma-separated list of areas (e.g., SECO,S,NE,N)",
    )(func)
    func = click.option(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated list of horizons (e.g., 0,1,2,3)",
    )(func)
    func = click.option(
        "--parallel-workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )(func)
    return func


class CLIContext:
    """Context object passed through CLI commands.

    Stores shared state like configuration, verbosity, and output format.

    Attributes:
        config_manager: Configuration manager instance.
        verbosity: Current verbosity level.
        output_format: Output format.
        logger: Structured logger instance.
    """

    def __init__(self) -> None:
        """Initialize CLI context."""
        self.config_manager: ConfigManager | None = None
        self.verbosity: Verbosity = Verbosity.NORMAL
        self.output_format: OutputFormat = OutputFormat.TEXT
        self.logger: StructuredLogger | None = None

    def load_config(self, config_path: Path | str) -> None:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file.
        """
        path = Path(config_path)
        if path.exists():
            self.config_manager = ConfigManager.from_yaml_with_env_overrides(path)
        else:
            click.echo(f"Config file not found: {path}, using defaults", err=True)
            self.config_manager = ConfigManager.default()

    def setup_logger(self, workflow_name: str) -> StructuredLogger:
        """Set up structured logger for a workflow.

        Args:
            workflow_name: Name of the workflow.

        Returns:
            Configured StructuredLogger.
        """
        level = LogLevel.DEBUG if self.verbosity == Verbosity.DEBUG else LogLevel.INFO
        fmt = LogFormat.JSON if self.output_format == OutputFormat.JSON else LogFormat.TEXT

        self.logger = create_workflow_logger(
            workflow_name=workflow_name,
            level=level,
            fmt=fmt,
        )
        return self.logger


pass_context = click.make_pass_decorator(CLIContext, ensure=True)


# ==================== Output Helpers ====================


class OutputHelper:
    """Helper class for formatted CLI output."""

    def __init__(
        self,
        fmt: OutputFormat = OutputFormat.TEXT,
        quiet: bool = False,
    ) -> None:
        """Initialize output helper.

        Args:
            fmt: Output format.
            quiet: Whether to suppress output.
        """
        self.fmt = fmt
        self.quiet = quiet

    def success(self, message: str) -> None:
        """Print success message.

        Args:
            message: Success message.
        """
        if not self.quiet:
            if self.fmt == OutputFormat.TEXT:
                click.echo(click.style(f"✓ {message}", fg="green"))
            elif self.fmt == OutputFormat.JSON:
                click.echo(f'{{"status": "success", "message": "{message}"}}')
            else:
                click.echo(message)

    def error(self, message: str) -> None:
        """Print error message.

        Args:
            message: Error message.
        """
        if self.fmt == OutputFormat.TEXT:
            click.echo(click.style(f"✗ {message}", fg="red"), err=True)
        elif self.fmt == OutputFormat.JSON:
            click.echo(f'{{"status": "error", "message": "{message}"}}', err=True)
        else:
            click.echo(f"Error: {message}", err=True)

    def info(self, message: str) -> None:
        """Print info message.

        Args:
            message: Info message.
        """
        if not self.quiet:
            if self.fmt == OutputFormat.TEXT:
                click.echo(click.style(f"ℹ {message}", fg="blue"))
            elif self.fmt == OutputFormat.JSON:
                click.echo(f'{{"status": "info", "message": "{message}"}}')
            else:
                click.echo(message)

    def warning(self, message: str) -> None:
        """Print warning message.

        Args:
            message: Warning message.
        """
        if not self.quiet:
            if self.fmt == OutputFormat.TEXT:
                click.echo(click.style(f"⚠ {message}", fg="yellow"))
            elif self.fmt == OutputFormat.JSON:
                click.echo(f'{{"status": "warning", "message": "{message}"}}')
            else:
                click.echo(f"Warning: {message}")

    def header(self, title: str) -> None:
        """Print section header.

        Args:
            title: Header title.
        """
        if not self.quiet and self.fmt == OutputFormat.TEXT:
            click.echo()
            click.echo(click.style(f"═══ {title} ═══", fg="cyan", bold=True))
            click.echo()

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Print formatted table.

        Args:
            headers: Table headers.
            rows: Table data rows.
        """
        if self.quiet:
            return

        if self.fmt == OutputFormat.JSON:
            import json
            data = [dict(zip(headers, row)) for row in rows]
            click.echo(json.dumps(data, indent=2))
            return

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        # Print header
        header_row = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        click.echo(header_row)
        click.echo("-" * len(header_row))

        # Print rows
        for row in rows:
            row_str = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            click.echo(row_str)

    def progress(self, current: int, total: int, label: str = "") -> None:
        """Print progress indicator.

        Args:
            current: Current progress.
            total: Total items.
            label: Optional label.
        """
        if self.quiet or self.fmt != OutputFormat.TEXT:
            return

        percent = (current / total) * 100 if total > 0 else 0
        bar_width = 30
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        label_str = f" {label}" if label else ""
        click.echo(f"\r[{bar}] {percent:5.1f}% ({current}/{total}){label_str}", nl=False)

        if current >= total:
            click.echo()


# ==================== Main CLI Group ====================


@click.group()
@click.version_option(version="0.1.0", prog_name="prevcarga")
@click.option(
    "-c", "--config",
    type=click.Path(path_type=Path),
    default="config/prevcarga.yaml",
    help="Path to configuration file",
)
@click.option(
    "-v", "--verbose",
    count=True,
    help="Increase verbosity (-v, -vv, -vvv)",
)
@click.option(
    "-q", "--quiet",
    is_flag=True,
    help="Suppress non-essential output",
)
@click.option(
    "-o", "--output-format",
    type=click.Choice(["text", "json", "table", "quiet"]),
    default="text",
    help="Output format",
)
@pass_context
def cli(
    ctx: CLIContext,
    config: Path,
    verbose: int,
    quiet: bool,
    output_format: str,
) -> None:
    """PrevCarga - Brazilian Electric Load Forecasting System.

    A comprehensive command-line tool for training, prediction, and
    evaluation of electric load forecasting models for the Brazilian
    interconnected power system (SIN).

    Example usage:

        # Train models for all areas
        prevcarga train --config config/prevcarga.yaml

        # Generate predictions
        prevcarga predict --areas SECO,S --horizons 0,1,2

        # Run backtesting
        prevcarga backtest --start 2023-01-01 --end 2023-12-31

        # Show system status
        prevcarga status
    """
    # Set verbosity
    if quiet:
        ctx.verbosity = Verbosity.QUIET
    elif verbose >= 3:
        ctx.verbosity = Verbosity.DEBUG
    elif verbose >= 2:
        ctx.verbosity = Verbosity.VERBOSE
    elif verbose >= 1:
        ctx.verbosity = Verbosity.NORMAL
    else:
        ctx.verbosity = Verbosity.NORMAL

    # Set output format
    ctx.output_format = OutputFormat(output_format)

    # Load configuration
    ctx.load_config(config)


# ==================== Train Command ====================


@cli.command()
@click.option(
    "--areas",
    type=str,
    default=None,
    help="Comma-separated list of areas (e.g., SECO,S,NE,N)",
)
@click.option(
    "--horizons",
    type=str,
    default=None,
    help="Comma-separated list of horizons (e.g., 0,1,2,3)",
)
@click.option(
    "--model-types",
    type=str,
    default=None,
    help="Comma-separated list of model types (e.g., lgbm,random_forest)",
)
@click.option(
    "--optimize/--no-optimize",
    default=None,
    help="Enable/disable hyperparameter optimization",
)
@click.option(
    "--parallel-workers",
    type=int,
    default=None,
    help="Number of parallel workers",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without executing",
)
@pass_context
def train(
    ctx: CLIContext,
    areas: str | None,
    horizons: str | None,
    model_types: str | None,
    optimize: bool | None,
    parallel_workers: int | None,
    dry_run: bool,
) -> None:
    """Train forecasting models.

    Train machine learning models for electric load forecasting across
    specified areas and forecast horizons.

    Examples:

        # Train with default settings
        prevcarga train

        # Train specific areas and horizons
        prevcarga train --areas SECO,S --horizons 0,1,2

        # Train with hyperparameter optimization
        prevcarga train --optimize

        # Dry run to see configuration
        prevcarga train --dry-run
    """
    output = OutputHelper(ctx.output_format, ctx.verbosity == Verbosity.QUIET)
    output.header("Training Workflow")

    if ctx.config_manager is None:
        output.error("Configuration not loaded")
        sys.exit(1)

    # Apply command-line overrides
    if areas:
        ctx.config_manager.update_training_config(areas=areas.split(","))
    if horizons:
        ctx.config_manager.update_training_config(horizons=[int(h) for h in horizons.split(",")])
    if model_types:
        ctx.config_manager.update_training_config(model_types=model_types.split(","))
    if optimize is not None:
        ctx.config_manager.update_training_config(optimize_hyperparameters=optimize)
    if parallel_workers is not None:
        ctx.config_manager.update_training_config(parallel_workers=parallel_workers)

    config = ctx.config_manager.get_training_config()

    output.info(f"Areas: {', '.join(config.areas)}")
    output.info(f"Horizons: {', '.join(f'D+{h}' for h in config.horizons)}")
    output.info(f"Model Types: {', '.join(config.model_types)}")
    output.info(f"Parallel Workers: {config.parallel_workers}")

    if dry_run:
        output.info("Dry run - no training performed")
        return

    # Run training workflow
    workflow = TrainingWorkflow(ctx.config_manager)

    output.info("Starting training...")
    result = workflow.run()

    if result.success:
        output.success(f"Training completed in {result.duration_seconds:.1f}s")
        output.info(f"Models trained: {len(result.trained_models)}")
    else:
        output.error(f"Training failed: {result.error}")
        sys.exit(1)


# ==================== Predict Command ====================


@cli.command()
@click.option(
    "--areas",
    type=str,
    default=None,
    help="Comma-separated list of areas",
)
@click.option(
    "--horizons",
    type=str,
    default=None,
    help="Comma-separated list of horizons",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for forecasts",
)
@click.option(
    "--format",
    "output_file_format",
    type=click.Choice(["parquet", "csv", "json"]),
    default=None,
    help="Output file format",
)
@click.option(
    "--reconcile/--no-reconcile",
    default=None,
    help="Apply hierarchical reconciliation",
)
@pass_context
def predict(
    ctx: CLIContext,
    areas: str | None,
    horizons: str | None,
    output_dir: Path | None,
    output_file_format: str | None,
    reconcile: bool | None,
) -> None:
    """Generate load forecasts.

    Generate electric load predictions for specified areas and horizons
    using trained models.

    Examples:

        # Generate predictions with defaults
        prevcarga predict

        # Predict specific areas
        prevcarga predict --areas SECO,S --horizons 0,1,2

        # Save as CSV
        prevcarga predict --format csv --output-dir ./forecasts
    """
    output = OutputHelper(ctx.output_format, ctx.verbosity == Verbosity.QUIET)
    output.header("Prediction Workflow")

    if ctx.config_manager is None:
        output.error("Configuration not loaded")
        sys.exit(1)

    # Apply overrides
    if areas:
        ctx.config_manager.update_prediction_config(areas=areas.split(","))
    if horizons:
        ctx.config_manager.update_prediction_config(horizons=[int(h) for h in horizons.split(",")])
    if output_dir:
        ctx.config_manager.update_prediction_config(output_dir=str(output_dir))
    if output_file_format:
        ctx.config_manager.update_prediction_config(output_format=output_file_format)
    if reconcile is not None:
        ctx.config_manager.update_prediction_config(apply_reconciliation=reconcile)

    config = ctx.config_manager.get_prediction_config()

    output.info(f"Areas: {', '.join(config.areas)}")
    output.info(f"Horizons: {', '.join(f'D+{h}' for h in config.horizons)}")
    output.info(f"Output Format: {config.output_format}")

    # Run prediction workflow
    workflow = PredictionWorkflow(ctx.config_manager)

    output.info("Generating predictions...")
    result = workflow.run()

    if result.success:
        n_forecasts = sum(len(h) for h in result.forecasts.values())
        output.success(f"Prediction completed in {result.duration_seconds:.1f}s")
        output.info(f"Forecasts generated: {n_forecasts}")
        if result.output_paths:
            output.info(f"Output: {result.output_paths.get('forecasts', 'N/A')}")
    else:
        output.error(f"Prediction failed: {result.error}")
        sys.exit(1)


# ==================== Backtest Command ====================


@cli.command()
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="End date (YYYY-MM-DD)",
)
@click.option(
    "--step-days",
    type=int,
    default=None,
    help="Days between backtests",
)
@click.option(
    "--areas",
    type=str,
    default=None,
    help="Comma-separated list of areas",
)
@click.option(
    "--report/--no-report",
    default=None,
    help="Generate HTML report",
)
@pass_context
def backtest(
    ctx: CLIContext,
    start: datetime | None,
    end: datetime | None,
    step_days: int | None,
    areas: str | None,
    report: bool | None,
) -> None:
    """Run backtesting evaluation.

    Evaluate forecast model performance over historical periods using
    time-series cross-validation.

    Examples:

        # Backtest with defaults
        prevcarga backtest

        # Backtest specific period
        prevcarga backtest --start 2023-01-01 --end 2023-12-31

        # Weekly backtests with report
        prevcarga backtest --step-days 7 --report
    """
    output = OutputHelper(ctx.output_format, ctx.verbosity == Verbosity.QUIET)
    output.header("Backtesting Workflow")

    if ctx.config_manager is None:
        output.error("Configuration not loaded")
        sys.exit(1)

    # Apply overrides
    if start:
        ctx.config_manager.update_backtesting_config(start_date=start.strftime("%Y-%m-%d"))
    if end:
        ctx.config_manager.update_backtesting_config(end_date=end.strftime("%Y-%m-%d"))
    if step_days:
        ctx.config_manager.update_backtesting_config(step_days=step_days)
    if areas:
        ctx.config_manager.update_backtesting_config(areas=areas.split(","))
    if report is not None:
        ctx.config_manager.update_backtesting_config(generate_report=report)

    config = ctx.config_manager.get_backtesting_config()

    output.info(f"Start Date: {config.start_date or 'auto'}")
    output.info(f"End Date: {config.end_date or 'auto'}")
    output.info(f"Step Days: {config.step_days}")
    output.info(f"Areas: {', '.join(config.areas)}")

    # Run backtesting workflow
    workflow = BacktestingWorkflow(ctx.config_manager)

    output.info("Running backtests...")
    result = workflow.run(
        start_date=start,
        end_date=end,
    )

    if result.success:
        output.success(f"Backtesting completed in {result.duration_seconds:.1f}s")
        output.info(f"Windows evaluated: {len(result.windows)}")

        # Show aggregate metrics
        if result.aggregate_metrics:
            output.header("Aggregate Metrics")
            headers = ["Metric", "Mean", "Std", "Min", "Max"]
            rows = []
            for metric, stats in result.aggregate_metrics.items():
                if isinstance(stats, dict):
                    rows.append([
                        metric.upper(),
                        f"{stats.get('mean', 0):.4f}",
                        f"{stats.get('std', 0):.4f}",
                        f"{stats.get('min', 0):.4f}",
                        f"{stats.get('max', 0):.4f}",
                    ])
            output.table(headers, rows)

        if result.report_path:
            output.info(f"Report: {result.report_path}")
    else:
        output.error(f"Backtesting failed: {result.error}")
        sys.exit(1)


# ==================== Status Command ====================


@cli.command()
@pass_context
def status(ctx: CLIContext) -> None:
    """Show system status.

    Display information about the current configuration, available models,
    and system health.

    Examples:

        # Show status
        prevcarga status

        # Show status in JSON format
        prevcarga -o json status
    """
    output = OutputHelper(ctx.output_format, ctx.verbosity == Verbosity.QUIET)
    output.header("System Status")

    if ctx.config_manager is None:
        output.error("Configuration not loaded")
        sys.exit(1)

    config = ctx.config_manager.get_config()

    # System info
    output.info(f"Environment: {config.system.environment.value}")
    output.info(f"Log Level: {config.system.log_level.value}")
    output.info(f"Storage Backend: {config.system.storage.backend.value}")

    # Training config
    output.header("Training Configuration")
    training = config.training
    output.info(f"Enabled: {training.enabled}")
    output.info(f"Areas: {', '.join(training.areas)}")
    output.info(f"Horizons: {', '.join(f'D+{h}' for h in training.horizons)}")
    output.info(f"Model Types: {', '.join(training.model_types)}")
    output.info(f"Parallel Workers: {training.parallel_workers}")

    # Prediction config
    output.header("Prediction Configuration")
    prediction = config.prediction
    output.info(f"Enabled: {prediction.enabled}")
    output.info(f"Output Format: {prediction.output_format}")
    output.info(f"Reconciliation: {prediction.apply_reconciliation}")

    # Validate config
    warnings = ctx.config_manager.validate()
    if warnings:
        output.header("Configuration Warnings")
        for warning in warnings:
            output.warning(warning)


# ==================== Config Commands ====================


@cli.group()
def config() -> None:
    """Configuration management commands.

    View and manage PrevCarga configuration settings.
    """
    pass


@config.command("show")
@click.option(
    "--section",
    type=click.Choice(["system", "training", "prediction", "backtesting", "features", "reconciliation"]),
    default=None,
    help="Show specific section",
)
@pass_context
def config_show(ctx: CLIContext, section: str | None) -> None:
    """Show current configuration.

    Display the current configuration settings.

    Examples:

        # Show all configuration
        prevcarga config show

        # Show training configuration
        prevcarga config show --section training
    """
    output = OutputHelper(ctx.output_format, ctx.verbosity == Verbosity.QUIET)

    if ctx.config_manager is None:
        output.error("Configuration not loaded")
        sys.exit(1)

    config = ctx.config_manager.get_config()

    if ctx.output_format == OutputFormat.JSON:
        import json
        if section:
            data = getattr(config, section).model_dump()
        else:
            data = config.model_dump()
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        if section:
            output.header(f"{section.title()} Configuration")
            section_config = getattr(config, section)
            for key, value in section_config.model_dump().items():
                click.echo(f"  {key}: {value}")
        else:
            for sect in ["system", "training", "prediction", "backtesting"]:
                output.header(f"{sect.title()}")
                section_config = getattr(config, sect)
                for key, value in section_config.model_dump().items():
                    click.echo(f"  {key}: {value}")


@config.command("validate")
@pass_context
def config_validate(ctx: CLIContext) -> None:
    """Validate current configuration.

    Check configuration for errors and warnings.
    """
    output = OutputHelper(ctx.output_format, ctx.verbosity == Verbosity.QUIET)

    if ctx.config_manager is None:
        output.error("Configuration not loaded")
        sys.exit(1)

    warnings = ctx.config_manager.validate()

    if warnings:
        output.warning(f"Found {len(warnings)} warning(s):")
        for warning in warnings:
            output.warning(f"  - {warning}")
    else:
        output.success("Configuration is valid")


# ==================== Entry Point ====================


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
