"""Prediction result display utilities for CLI.

This module provides utilities for displaying prediction plans, results,
and progress information in the CLI.

Key Components:
- PredictionResultDisplay: Class for formatted prediction result display
- display_prediction_summary: Show batch prediction summary
- display_intraday_summary: Show intraday prediction summary
- save_predictions: Save predictions to file in various formats
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import click


@dataclass
class PredictionStatistics:
    """Statistics about generated predictions.

    Attributes:
        mean_load: Average forecasted load in MW.
        max_load: Maximum forecasted load in MW.
        min_load: Minimum forecasted load in MW.
        std_load: Standard deviation of forecasted load.
    """

    mean_load: float = 0.0
    max_load: float = 0.0
    min_load: float = 0.0
    std_load: float = 0.0


@dataclass
class UncertaintyInfo:
    """Uncertainty quantification information.

    Attributes:
        avg_pi_width: Average prediction interval width in MW.
        coverage: Coverage percentage of prediction intervals.
        lower_bound: Lower bound percentile.
        upper_bound: Upper bound percentile.
    """

    avg_pi_width: float = 0.0
    coverage: float = 95.0
    lower_bound: float = 2.5
    upper_bound: float = 97.5


@dataclass
class BLFWeights:
    """BLF (Best Linear Forecast) blending weights.

    Attributes:
        model: Weight for model predictions.
        observed: Weight for observed values.
    """

    model: float = 0.5
    observed: float = 0.5


@dataclass
class ImprovementInfo:
    """Forecast improvement information.

    Attributes:
        mape_reduction: Reduction in MAPE percentage.
        rmse_reduction: Reduction in RMSE.
    """

    mape_reduction: float = 0.0
    rmse_reduction: float = 0.0


@dataclass
class BatchPredictionResult:
    """Result of a batch prediction run.

    Attributes:
        horizons: List of forecast horizons.
        areas: List of areas forecasted.
        total_forecasts: Total number of forecasts generated.
        execution_time: Time taken in seconds.
        statistics: Forecast statistics.
        uncertainty: Uncertainty quantification info.
        predictions: DataFrame of predictions.
        success: Whether prediction was successful.
        error: Error message if failed.
    """

    horizons: list[int] = field(default_factory=list)
    areas: list[str] = field(default_factory=list)
    total_forecasts: int = 0
    execution_time: float = 0.0
    statistics: PredictionStatistics | None = None
    uncertainty: UncertaintyInfo | None = None
    predictions: Any = None
    success: bool = True
    error: str | None = None

    def to_dataframe(self) -> Any:
        """Convert predictions to DataFrame.

        Returns:
            DataFrame with predictions.
        """
        if self.predictions is not None:
            return self.predictions

        import pandas as pd

        # Create empty DataFrame if no predictions
        return pd.DataFrame(columns=["timestamp", "area", "horizon", "forecast", "lower", "upper"])


@dataclass
class IntradayPredictionResult:
    """Result of an intraday prediction run.

    Attributes:
        update_datetime: Time of the update.
        areas: List of areas updated.
        execution_time: Time taken in seconds.
        blf_weights: BLF blending weights used.
        improvement: Forecast improvement metrics.
        predictions: DataFrame of predictions.
        success: Whether prediction was successful.
        error: Error message if failed.
    """

    update_datetime: datetime | None = None
    areas: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    blf_weights: BLFWeights | None = None
    improvement: ImprovementInfo | None = None
    predictions: Any = None
    success: bool = True
    error: str | None = None

    def to_dataframe(self) -> Any:
        """Convert predictions to DataFrame.

        Returns:
            DataFrame with predictions.
        """
        if self.predictions is not None:
            return self.predictions

        import pandas as pd

        return pd.DataFrame(columns=["timestamp", "area", "forecast", "lower", "upper"])


class PredictionResultDisplay:
    """Helper class for displaying prediction results.

    Provides formatted output for prediction plans, progress, and results.

    Attributes:
        use_colors: Whether to use colored output.
        verbose: Whether to show verbose output.
    """

    def __init__(self, use_colors: bool = True, verbose: bool = False) -> None:
        """Initialize prediction result display.

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


def display_prediction_summary(result: BatchPredictionResult) -> None:
    """Display batch prediction summary.

    Args:
        result: Batch prediction result to display.
    """
    display = PredictionResultDisplay()
    display.header("Batch Prediction Results")

    click.echo()
    display.key_value("Horizons:", len(result.horizons))
    display.key_value("Areas:", len(result.areas))
    display.key_value("Total forecasts:", result.total_forecasts)
    display.key_value("Execution time:", f"{result.execution_time:.1f}s")

    # Summary statistics
    if result.statistics:
        display.section("Forecast Statistics")
        display.key_value("Mean load:", f"{result.statistics.mean_load:.2f} MW")
        display.key_value("Max load:", f"{result.statistics.max_load:.2f} MW")
        display.key_value("Min load:", f"{result.statistics.min_load:.2f} MW")

    # Uncertainty info
    if result.uncertainty:
        display.section("Uncertainty Quantification")
        display.key_value("Avg PI width:", f"{result.uncertainty.avg_pi_width:.2f} MW")
        display.key_value("Coverage:", f"{result.uncertainty.coverage:.1f}%")


def display_intraday_summary(result: IntradayPredictionResult) -> None:
    """Display intraday prediction summary.

    Args:
        result: Intraday prediction result to display.
    """
    display = PredictionResultDisplay()
    display.header("Intraday Prediction Results")

    click.echo()
    display.key_value("Update time:", str(result.update_datetime))
    display.key_value("Strategy:", "LGBM BLF")
    display.key_value("Areas updated:", len(result.areas))
    display.key_value("Execution time:", f"{result.execution_time:.1f}s")

    # BLF blending info
    if result.blf_weights:
        display.section("BLF Blending Weights")
        display.key_value("Model weight:", f"{result.blf_weights.model:.3f}")
        display.key_value("Observed weight:", f"{result.blf_weights.observed:.3f}")

    # Forecast improvement
    if result.improvement:
        display.section("Forecast Improvement")
        display.key_value("MAPE reduction:", f"{result.improvement.mape_reduction:.2f}%")


def display_prediction_plan(
    date: datetime,
    horizons: list[int],
    areas: list[str],
    models: list[str],
    combination: str,
    reconciliation: str,
    output_format: str,
) -> None:
    """Display prediction plan.

    Args:
        date: Prediction date.
        horizons: List of forecast horizons.
        areas: List of areas.
        models: List of models to use.
        combination: Combination method.
        reconciliation: Reconciliation method.
        output_format: Output format.
    """
    display = PredictionResultDisplay()
    display.header("Prediction Plan")

    click.echo()
    display.key_value("Date:", str(date.date()))
    display.key_value("Horizons:", f"D+{min(horizons)} to D+{max(horizons)}")
    display.key_value("Areas:", len(areas))
    display.key_value("Models:", ", ".join(models))
    display.key_value("Combination:", combination)
    display.key_value("Reconciliation:", reconciliation)
    display.key_value("Output format:", output_format)


def save_predictions(
    result: BatchPredictionResult | IntradayPredictionResult,
    output_path: Path,
    output_format: str,
) -> None:
    """Save predictions to file.

    Args:
        result: Prediction result to save.
        output_path: Path to output file.
        output_format: Output format (csv, json, parquet).
    """
    df = result.to_dataframe()

    if output_format == "csv":
        df.to_csv(output_path, index=False)
    elif output_format == "json":
        df.to_json(output_path, orient="records", date_format="iso")
    elif output_format == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
