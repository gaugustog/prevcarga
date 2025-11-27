"""Progress monitoring system for CLI commands.

This module provides comprehensive progress bar management for long-running
CLI operations including training, prediction, feature generation, and
evaluation tasks.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from collections.abc import Callable


class ProgressBarManager:
    """Centralized progress bar management for CLI commands.

    Provides a unified interface for creating and managing progress bars
    across different CLI operations.

    Attributes:
        progress_bars: Dictionary of active progress bars.
        active_bar: Currently active progress bar ID.

    Example:
        >>> manager = ProgressBarManager()
        >>> with manager.create_progress_bar("Processing", 100) as bar:
        ...     for i in range(100):
        ...         # do work
        ...         bar.update(1)
    """

    # Default styling options
    DEFAULT_FILL_CHAR = "█"
    DEFAULT_EMPTY_CHAR = "░"

    def __init__(self) -> None:
        """Initialize progress bar manager."""
        self.progress_bars: dict[str, Any] = {}
        self.active_bar: str | None = None
        self._bar_counter = 0

    def create_progress_bar(
        self,
        label: str,
        length: int,
        show_eta: bool = True,
        show_percent: bool = True,
        show_pos: bool = False,
        item_show_func: Callable[[Any], str | None] | None = None,
        fill_char: str | None = None,
        empty_char: str | None = None,
    ) -> Any:
        """Create a new progress bar.

        Args:
            label: Progress bar label.
            length: Total number of steps.
            show_eta: Whether to show estimated time remaining.
            show_percent: Whether to show percentage complete.
            show_pos: Whether to show current position.
            item_show_func: Function to format current item display.
            fill_char: Character for filled portion of bar.
            empty_char: Character for empty portion of bar.

        Returns:
            Click progress bar object.
        """
        return click.progressbar(
            length=length,
            label=label,
            show_eta=show_eta,
            show_percent=show_percent,
            show_pos=show_pos,
            item_show_func=item_show_func,
            fill_char=fill_char or self.DEFAULT_FILL_CHAR,
            empty_char=empty_char or self.DEFAULT_EMPTY_CHAR,
        )

    def register_bar(self, bar_id: str, bar: Any) -> None:
        """Register a progress bar.

        Args:
            bar_id: Unique identifier for the bar.
            bar: Progress bar object.
        """
        self.progress_bars[bar_id] = bar
        self.active_bar = bar_id

    def unregister_bar(self, bar_id: str) -> None:
        """Unregister a progress bar.

        Args:
            bar_id: Identifier of bar to remove.
        """
        if bar_id in self.progress_bars:
            del self.progress_bars[bar_id]
        if self.active_bar == bar_id:
            self.active_bar = None

    def get_unique_id(self) -> str:
        """Generate a unique progress bar ID.

        Returns:
            Unique identifier string.
        """
        self._bar_counter += 1
        return f"bar_{self._bar_counter}"


class BaseProgressBar:
    """Base class for progress bar implementations.

    Provides common functionality for all progress bar types including
    context manager support, timing, and completion statistics.

    Attributes:
        total_tasks: Total number of tasks to complete.
        current_task: Number of completed tasks.
        start_time: Timestamp when progress started.
        task_times: List of individual task completion times.
    """

    def __init__(self, total_tasks: int, label: str = "Processing") -> None:
        """Initialize base progress bar.

        Args:
            total_tasks: Total number of tasks.
            label: Progress bar label.
        """
        self.total_tasks = total_tasks
        self.label = label
        self.current_task = 0
        self.start_time: float | None = None
        self.task_times: list[float] = []
        self._progress_bar: Any = None
        self._current_info: dict[str, Any] = {}
        self._show_progress = True

    def __enter__(self) -> BaseProgressBar:
        """Enter context manager."""
        self.start_time = time.time()
        if self._show_progress and self.total_tasks > 0:
            self._progress_bar = click.progressbar(
                length=self.total_tasks,
                label=self.label,
                show_eta=True,
                show_percent=True,
                fill_char="█",
                empty_char="░",
                item_show_func=self._item_display,
            )
            self._progress_bar.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        if self._progress_bar is not None:
            self._progress_bar.__exit__(exc_type, exc_val, exc_tb)

        if not exc_type and self.start_time:
            elapsed = time.time() - self.start_time
            click.echo(f"\nCompleted {self.current_task} tasks in {format_time(elapsed)}")

    def _item_display(self, item: Any) -> str:
        """Format current item display.

        Override in subclasses for custom display.

        Args:
            item: Current item (unused in base implementation).

        Returns:
            Formatted display string.
        """
        del item  # Unused in base implementation
        return ""

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def average_task_time(self) -> float:
        """Get average time per task in seconds."""
        if not self.task_times:
            return 0.0
        return sum(self.task_times) / len(self.task_times)

    @property
    def estimated_remaining(self) -> float:
        """Get estimated remaining time in seconds."""
        remaining_tasks = self.total_tasks - self.current_task
        return remaining_tasks * self.average_task_time

    @property
    def progress_percent(self) -> float:
        """Get completion percentage."""
        if self.total_tasks == 0:
            return 100.0
        return (self.current_task / self.total_tasks) * 100


class TrainingProgressBar(BaseProgressBar):
    """Progress bar for model training operations.

    Tracks training progress across multiple models and areas with
    status indicators for successful, failed, and skipped tasks.

    Attributes:
        models: List of model types being trained.
        areas: List of areas being trained.

    Example:
        >>> with TrainingProgressBar(["lgbm", "rf"], ["SE", "S"]) as progress:
        ...     for model in ["lgbm", "rf"]:
        ...         for area in ["SE", "S"]:
        ...             # train model
        ...             progress.update(model, area, "completed")
    """

    def __init__(
        self,
        models: list[str],
        areas: list[str],
        show_progress: bool = True,
    ) -> None:
        """Initialize training progress bar.

        Args:
            models: List of model types being trained.
            areas: List of areas being trained.
            show_progress: Whether to show progress bar.
        """
        self.models = models
        self.areas = areas
        total = len(models) * len(areas)
        super().__init__(total, "Training models")
        self._show_progress = show_progress

    def update(
        self,
        model: str,
        area: str,
        status: str = "completed",
    ) -> None:
        """Update progress bar.

        Args:
            model: Model type that was trained.
            area: Area that was trained.
            status: Task status ('completed', 'failed', 'skipped').
        """
        if self.start_time:
            task_time = time.time() - self.start_time - sum(self.task_times)
            self.task_times.append(task_time)

        self.current_task += 1
        self._current_info = {
            "model": model,
            "area": area,
            "status": status,
        }

        if self._progress_bar is not None:
            self._progress_bar.update(1)

    def _item_display(self, item: Any) -> str:
        """Format current item display."""
        del item  # Unused
        if self._current_info:
            info = self._current_info
            status_icon = "✓" if info["status"] == "completed" else "✗"
            return f"{status_icon} {info['model']} - {info['area']}"
        return ""


class PredictionProgressBar(BaseProgressBar):
    """Progress bar for prediction operations.

    Tracks prediction progress across multiple horizons and areas.

    Attributes:
        horizons: List of forecast horizons.
        areas: List of areas for prediction.

    Example:
        >>> with PredictionProgressBar([0, 1, 2], ["SE", "S"]) as progress:
        ...     for horizon in [0, 1, 2]:
        ...         for area in ["SE", "S"]:
        ...             # generate predictions
        ...             progress.update(horizon, area)
    """

    def __init__(
        self,
        horizons: list[int],
        areas: list[str],
        show_progress: bool = True,
    ) -> None:
        """Initialize prediction progress bar.

        Args:
            horizons: List of forecast horizons.
            areas: List of areas for prediction.
            show_progress: Whether to show progress bar.
        """
        self.horizons = horizons
        self.areas = areas
        total = len(horizons) * len(areas)
        super().__init__(total, "Generating predictions")
        self._show_progress = show_progress

    def update(self, horizon: int, area: str) -> None:
        """Update progress bar.

        Args:
            horizon: Horizon that was predicted.
            area: Area that was predicted.
        """
        if self.start_time:
            task_time = time.time() - self.start_time - sum(self.task_times)
            self.task_times.append(task_time)

        self.current_task += 1
        self._current_info = {
            "horizon": horizon,
            "area": area,
        }

        if self._progress_bar is not None:
            self._progress_bar.update(1)

    def _item_display(self, item: Any) -> str:
        """Format current item display."""
        del item  # Unused
        if self._current_info:
            info = self._current_info
            return f"D+{info['horizon']} - {info['area']}"
        return ""


class FeatureProgressBar(BaseProgressBar):
    """Progress bar for feature generation operations.

    Tracks feature generation progress across multiple plugins and areas.

    Attributes:
        plugins: List of feature plugins.
        areas: List of areas.

    Example:
        >>> with FeatureProgressBar(["temporal", "calendar"], ["SE"]) as progress:
        ...     for plugin in ["temporal", "calendar"]:
        ...         for area in ["SE"]:
        ...             # generate features
        ...             progress.update(plugin, area)
    """

    def __init__(
        self,
        plugins: list[str],
        areas: list[str],
        show_progress: bool = True,
    ) -> None:
        """Initialize feature generation progress bar.

        Args:
            plugins: List of feature plugins.
            areas: List of areas.
            show_progress: Whether to show progress bar.
        """
        self.plugins = plugins
        self.areas = areas
        total = len(plugins) * len(areas)
        super().__init__(total, "Generating features")
        self._show_progress = show_progress

    def update(self, plugin: str, area: str) -> None:
        """Update progress bar.

        Args:
            plugin: Plugin that generated features.
            area: Area for which features were generated.
        """
        if self.start_time:
            task_time = time.time() - self.start_time - sum(self.task_times)
            self.task_times.append(task_time)

        self.current_task += 1
        self._current_info = {
            "plugin": plugin,
            "area": area,
        }

        if self._progress_bar is not None:
            self._progress_bar.update(1)

    def _item_display(self, item: Any) -> str:
        """Format current item display."""
        del item  # Unused
        if self._current_info:
            info = self._current_info
            return f"{info['plugin']} - {info['area']}"
        return ""


class EvaluationProgressBar(BaseProgressBar):
    """Progress bar for evaluation operations.

    Tracks evaluation progress across multiple areas and metrics.

    Attributes:
        areas: List of areas to evaluate.
        metrics: List of metrics to calculate.

    Example:
        >>> with EvaluationProgressBar(["SE", "S"], ["mape", "rmse"]) as progress:
        ...     for area in ["SE", "S"]:
        ...         for metric in ["mape", "rmse"]:
        ...             # calculate metric
        ...             progress.update(area, metric)
    """

    def __init__(
        self,
        areas: list[str],
        metrics: list[str],
        show_progress: bool = True,
    ) -> None:
        """Initialize evaluation progress bar.

        Args:
            areas: List of areas to evaluate.
            metrics: List of metrics to calculate.
            show_progress: Whether to show progress bar.
        """
        self.areas = areas
        self.metrics = metrics
        total = len(areas) * len(metrics)
        super().__init__(total, "Evaluating model")
        self._show_progress = show_progress

    def update(self, area: str, metric: str) -> None:
        """Update progress bar.

        Args:
            area: Area being evaluated.
            metric: Metric being calculated.
        """
        if self.start_time:
            task_time = time.time() - self.start_time - sum(self.task_times)
            self.task_times.append(task_time)

        self.current_task += 1
        self._current_info = {
            "area": area,
            "metric": metric,
        }

        if self._progress_bar is not None:
            self._progress_bar.update(1)

    def _item_display(self, item: Any) -> str:
        """Format current item display."""
        del item  # Unused
        if self._current_info:
            info = self._current_info
            return f"{info['area']} - {info['metric']}"
        return ""


class BacktestProgressBar(BaseProgressBar):
    """Progress bar for backtesting operations.

    Tracks backtest progress across days and retraining intervals.

    Attributes:
        days: Total days in backtest period.
        intervals: Retraining intervals being tested.

    Example:
        >>> with BacktestProgressBar(30, [7, 14]) as progress:
        ...     for day in range(30):
        ...         for interval in [7, 14]:
        ...             # run backtest
        ...             progress.update(day, interval)
    """

    def __init__(
        self,
        days: int,
        intervals: list[int],
        show_progress: bool = True,
    ) -> None:
        """Initialize backtest progress bar.

        Args:
            days: Total days in backtest period.
            intervals: Retraining intervals being tested.
            show_progress: Whether to show progress bar.
        """
        self.days = days
        self.intervals = intervals
        total = days * len(intervals)
        super().__init__(total, "Running backtest")
        self._show_progress = show_progress

    def update(self, day: int, interval: int) -> None:
        """Update progress bar.

        Args:
            day: Current backtest day.
            interval: Current retraining interval.
        """
        if self.start_time:
            task_time = time.time() - self.start_time - sum(self.task_times)
            self.task_times.append(task_time)

        self.current_task += 1
        self._current_info = {
            "day": day,
            "interval": interval,
        }

        if self._progress_bar is not None:
            self._progress_bar.update(1)

    def _item_display(self, item: Any) -> str:
        """Format current item display."""
        del item  # Unused
        if self._current_info:
            info = self._current_info
            return f"Day {info['day']} - {info['interval']}d interval"
        return ""


class MultiLevelProgressBar:
    """Multi-level progress bar for nested operations.

    Supports tracking progress at multiple levels (e.g., overall and
    per-task progress).

    Attributes:
        levels: Number of progress levels.
        labels: Labels for each level.

    Example:
        >>> with MultiLevelProgressBar(["Models", "Areas"], [3, 5]) as progress:
        ...     for model_idx in range(3):
        ...         for area_idx in range(5):
        ...             progress.update(0, model_idx)  # Update model level
        ...             progress.update(1, area_idx)   # Update area level
    """

    def __init__(
        self,
        labels: list[str],
        totals: list[int],
        show_progress: bool = True,
    ) -> None:
        """Initialize multi-level progress bar.

        Args:
            labels: Labels for each level.
            totals: Total counts for each level.
            show_progress: Whether to show progress bars.
        """
        if len(labels) != len(totals):
            msg = "labels and totals must have the same length"
            raise ValueError(msg)

        self.labels = labels
        self.totals = totals
        self.levels = len(labels)
        self.current_counts = [0] * self.levels
        self._progress_bars: list[Any] = []
        self._show_progress = show_progress
        self.start_time: float | None = None

    def __enter__(self) -> MultiLevelProgressBar:
        """Enter context manager."""
        self.start_time = time.time()
        if self._show_progress:
            for label, total in zip(self.labels, self.totals, strict=True):
                if total > 0:
                    bar = click.progressbar(
                        length=total,
                        label=label,
                        show_eta=True,
                        show_percent=True,
                        fill_char="█",
                        empty_char="░",
                    )
                    bar.__enter__()
                    self._progress_bars.append(bar)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        for bar in reversed(self._progress_bars):
            bar.__exit__(exc_type, exc_val, exc_tb)

        if not exc_type and self.start_time:
            elapsed = time.time() - self.start_time
            click.echo(f"\nCompleted all tasks in {format_time(elapsed)}")

    def update(self, level: int, increment: int = 1) -> None:
        """Update progress at specified level.

        Args:
            level: Progress level to update (0-indexed).
            increment: Amount to increment by.

        Raises:
            IndexError: If level is out of range.
        """
        if level < 0 or level >= self.levels:
            msg = f"Level {level} out of range (0-{self.levels - 1})"
            raise IndexError(msg)

        self.current_counts[level] += increment
        if self._show_progress and level < len(self._progress_bars):
            self._progress_bars[level].update(increment)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string (e.g., "5.2s", "2.3m", "1.5h").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    hours = seconds / 3600
    return f"{hours:.1f}h"


def format_time_detailed(seconds: float) -> str:
    """Format seconds into detailed human-readable time.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string (e.g., "5s", "2m 30s", "1h 15m").
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"


def create_simple_progress(
    label: str,
    length: int,
    show_eta: bool = False,
    show_percent: bool = False,
) -> Any:
    """Create a simple progress bar for quick tasks.

    Args:
        label: Progress bar label.
        length: Total number of steps.
        show_eta: Whether to show ETA.
        show_percent: Whether to show percentage.

    Returns:
        Click progress bar object.
    """
    return click.progressbar(
        length=length,
        label=label,
        show_eta=show_eta,
        show_percent=show_percent,
    )


def display_progress_summary(
    completed: int,
    total: int,
    elapsed: float,
    success_count: int | None = None,
    failed_count: int | None = None,
) -> None:
    """Display a summary of progress completion.

    Args:
        completed: Number of completed tasks.
        total: Total number of tasks.
        elapsed: Elapsed time in seconds.
        success_count: Number of successful tasks.
        failed_count: Number of failed tasks.
    """
    click.echo()
    click.echo("═" * 60)
    click.echo("                    Progress Summary")
    click.echo("═" * 60)
    click.echo(f"  Completed:    {completed}/{total} tasks")
    click.echo(f"  Elapsed:      {format_time_detailed(elapsed)}")

    if total > 0:
        avg_time = elapsed / total
        click.echo(f"  Avg per task: {format_time(avg_time)}")

    if success_count is not None:
        click.secho(f"  Succeeded:    {success_count}", fg="green")

    if failed_count is not None and failed_count > 0:
        click.secho(f"  Failed:       {failed_count}", fg="red")

    click.echo("═" * 60)
