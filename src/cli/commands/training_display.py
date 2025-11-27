"""Training result display utilities for CLI.

This module provides utilities for displaying training plans, results,
and progress information in the CLI.

Key Components:
- TrainingResultDisplay: Class for formatted training result display
- display_training_plan: Show detailed training plan
- display_training_results: Show training results summary
- display_batch_training_plan: Show batch training configuration
- display_batch_results: Show aggregate batch results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import click


@dataclass
class TrainedModelInfo:
    """Information about a trained model.

    Attributes:
        model_type: Type of the model (lgbm, rf, etc.).
        area: Area the model was trained for.
        horizon: Forecast horizon.
        mape: Mean Absolute Percentage Error.
        training_time: Time taken to train in seconds.
        error: Error message if training failed.
    """

    model_type: str
    area: str
    horizon: int = 0
    mape: float = 0.0
    training_time: float = 0.0
    error: str | None = None


@dataclass
class TrainingResult:
    """Result of a training run.

    Attributes:
        trained_models: List of successfully trained models.
        failed_models: List of models that failed to train.
        training_time: Total training time in seconds.
        avg_time_per_model: Average time per model in seconds.
        success: Whether the training run was successful.
        error: Error message if training failed.
    """

    trained_models: list[TrainedModelInfo] = field(default_factory=list)
    failed_models: list[TrainedModelInfo] = field(default_factory=list)
    training_time: float = 0.0
    avg_time_per_model: float = 0.0
    success: bool = True
    error: str | None = None


class TrainingResultDisplay:
    """Helper class for displaying training results.

    Provides formatted output for training plans, progress, and results.

    Attributes:
        use_colors: Whether to use colored output.
        verbose: Whether to show verbose output.
    """

    def __init__(self, use_colors: bool = True, verbose: bool = False) -> None:
        """Initialize training result display.

        Args:
            use_colors: Whether to use colored output.
            verbose: Whether to show verbose output.
        """
        self.use_colors = use_colors
        self.verbose = verbose

    def header(self, title: str, char: str = "═") -> None:
        """Print a header with border.

        Args:
            title: Header title.
            char: Character for border (default: ═).
        """
        width = 50
        click.echo(f"╔{char * (width - 2)}╗")
        click.echo(f"║{title:^{width - 2}}║")
        click.echo(f"╚{char * (width - 2)}╝")

    def section(self, title: str) -> None:
        """Print a section header.

        Args:
            title: Section title.
        """
        click.echo()
        if self.use_colors:
            click.secho(f"──── {title} ────", fg="cyan", bold=True)
        else:
            click.echo(f"──── {title} ────")
        click.echo()

    def success(self, message: str) -> None:
        """Print a success message.

        Args:
            message: Success message.
        """
        if self.use_colors:
            click.secho(f"✓ {message}", fg="green", bold=True)
        else:
            click.echo(f"OK: {message}")

    def error(self, message: str) -> None:
        """Print an error message.

        Args:
            message: Error message.
        """
        if self.use_colors:
            click.secho(f"✗ {message}", fg="red", bold=True)
        else:
            click.echo(f"ERROR: {message}")

    def warning(self, message: str) -> None:
        """Print a warning message.

        Args:
            message: Warning message.
        """
        if self.use_colors:
            click.secho(f"⚠ {message}", fg="yellow", bold=True)
        else:
            click.echo(f"WARNING: {message}")

    def info(self, message: str) -> None:
        """Print an info message.

        Args:
            message: Info message.
        """
        if self.use_colors:
            click.secho(f"ℹ {message}", fg="blue")
        else:
            click.echo(f"INFO: {message}")

    def key_value(self, key: str, value: Any, indent: int = 0) -> None:
        """Print a key-value pair.

        Args:
            key: Key name.
            value: Value to display.
            indent: Indentation level.
        """
        prefix = "  " * indent
        click.echo(f"{prefix}{key:20} {value}")

    def bullet(self, message: str, indent: int = 0) -> None:
        """Print a bulleted item.

        Args:
            message: Message to display.
            indent: Indentation level.
        """
        prefix = "  " * indent
        click.echo(f"{prefix}• {message}")


def display_training_plan(
    models: list[str],
    areas: list[str],
    start_date: datetime,
    end_date: datetime,
    horizons: list[int] | None = None,
) -> None:
    """Display detailed training plan.

    Args:
        models: List of model types to train.
        areas: List of areas to train for.
        start_date: Training data start date.
        end_date: Training data end date.
        horizons: Forecast horizons (optional).
    """
    display = TrainingResultDisplay()
    display.section("Detailed Training Plan")

    horizons_str = ", ".join(f"D+{h}" for h in (horizons or list(range(9))))

    for model in models:
        click.echo(f"\nModel: {model}")
        click.echo("-" * 60)
        for area in areas:
            click.echo(f"  • Area: {area:10} | Period: {start_date.date()} to {end_date.date()}")
            if horizons:
                click.echo(f"                       | Horizons: {horizons_str}")

    total_tasks = len(models) * len(areas)
    click.echo()
    click.echo(f"Total training tasks: {total_tasks}")


def display_training_results(result: TrainingResult) -> None:
    """Display training results summary.

    Args:
        result: Training result to display.
    """
    display = TrainingResultDisplay()
    display.header("Training Results")

    click.echo()
    display.key_value("Models trained:", len(result.trained_models))
    display.key_value("Models failed:", len(result.failed_models))
    display.key_value("Training time:", f"{result.training_time:.1f}s")
    if result.trained_models:
        display.key_value("Avg time/model:", f"{result.avg_time_per_model:.1f}s")

    if result.trained_models:
        display.section("Successfully Trained Models")
        # Show first 10 models
        for i, model_info in enumerate(result.trained_models[:10]):
            mape_str = f"MAPE: {model_info.mape:.2f}%" if model_info.mape else ""
            display.bullet(f"{model_info.model_type} - {model_info.area} {mape_str}")

        if len(result.trained_models) > 10:
            remaining = len(result.trained_models) - 10
            click.echo(f"  ... and {remaining} more")

    if result.failed_models:
        display.section("Failed Models")
        for model_info in result.failed_models:
            error_msg = model_info.error or "Unknown error"
            display.bullet(f"{model_info.model_type} - {model_info.area}: {error_msg}")


def display_batch_training_plan(training_config: dict[str, Any]) -> None:
    """Display batch training plan.

    Args:
        training_config: Batch training configuration dictionary.
    """
    display = TrainingResultDisplay()
    display.header("Batch Training Configuration")

    click.echo()
    models = training_config.get("models", [])
    areas = training_config.get("areas", [])
    periods = training_config.get("training_periods", [])
    parallel = training_config.get("parallel_workers", 4)
    force = training_config.get("force_retrain", False)

    display.key_value("Models:", ", ".join(models))
    display.key_value("Areas:", ", ".join(areas))
    display.key_value("Periods:", len(periods))
    display.key_value("Parallel workers:", parallel)
    display.key_value("Force retrain:", "Yes" if force else "No")

    if periods:
        display.section("Training Periods")
        for i, period in enumerate(periods, 1):
            start = period.get("start_date", "N/A")
            end = period.get("end_date", "N/A")
            click.echo(f"  {i}. {start} to {end}")

    total_tasks = len(models) * len(areas) * len(periods)
    click.echo()
    click.echo(f"Total training tasks: {total_tasks}")


def display_batch_results(results: list[TrainingResult]) -> None:
    """Display aggregate batch training results.

    Args:
        results: List of training results from each period.
    """
    display = TrainingResultDisplay()
    display.header("Batch Training Results")

    click.echo()
    total_trained = sum(len(r.trained_models) for r in results)
    total_failed = sum(len(r.failed_models) for r in results)
    total_time = sum(r.training_time for r in results)

    display.key_value("Total periods:", len(results))
    display.key_value("Models trained:", total_trained)
    display.key_value("Models failed:", total_failed)
    display.key_value("Total time:", f"{total_time:.1f}s")

    if total_trained > 0:
        avg_time = total_time / total_trained
        display.key_value("Avg time/model:", f"{avg_time:.1f}s")

    # Show per-period summary
    if len(results) > 1:
        display.section("Per-Period Summary")
        for i, result in enumerate(results, 1):
            trained = len(result.trained_models)
            failed = len(result.failed_models)
            status = "✓" if failed == 0 else "⚠"
            click.echo(
                f"  {i}. {status} Trained: {trained}, Failed: {failed}, "
                f"Time: {result.training_time:.1f}s"
            )

    # Show any failures
    all_failures = []
    for result in results:
        all_failures.extend(result.failed_models)

    if all_failures:
        display.section("All Failed Models")
        for model_info in all_failures:
            error_msg = model_info.error or "Unknown error"
            display.bullet(f"{model_info.model_type} - {model_info.area}: {error_msg}")
