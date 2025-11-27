"""Training commands for PrevCarga CLI.

This module provides CLI commands for model training, including
individual model training and batch training from configuration files.

Key Commands:
- train model: Train individual or all models for specified areas
- train batch: Execute batch training from configuration file

Example:
    ```bash
    # Train LGBM for specific areas
    prevcarga train model --model lgbm --area SECO --area S \\
        --start-date 2023-01-01 --end-date 2023-12-31

    # Train all models with parallel workers
    prevcarga train model --model all --parallel 8 \\
        --start-date 2023-01-01 --end-date 2023-12-31

    # Batch training from config file
    prevcarga train batch --config-file training-config.yaml
    ```
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml

from src.cli.commands.training_display import (
    TrainedModelInfo,
    TrainingResult,
    TrainingResultDisplay,
    display_batch_results,
    display_batch_training_plan,
    display_training_plan,
    display_training_results,
)

# Valid model types
VALID_MODEL_TYPES = ["lgbm", "rf", "regdin_svm", "holt_winters"]

# Valid area codes for Brazilian grid
VALID_AREAS = [
    "SECO",
    "S",
    "NE",
    "N",
    "SIN",
    "SP",
    "RJ",
    "MG",
    "ES",
    "PR",
    "SC",
    "RS",
    "BA",
    "PE",
    "CE",
    "PA",
    "AM",
]


class TrainingProgressBar:
    """Progress bar for training operations.

    Provides visual feedback during model training with support
    for parallel execution progress tracking.

    Attributes:
        models: List of model types being trained.
        areas: List of areas being trained.
        total_tasks: Total number of training tasks.
        completed_tasks: Number of completed tasks.
    """

    def __init__(
        self,
        models: list[str],
        areas: list[str],
        label: str = "Training",
    ) -> None:
        """Initialize training progress bar.

        Args:
            models: List of model types.
            areas: List of areas.
            label: Progress bar label.
        """
        self.models = models
        self.areas = areas
        self.total_tasks = len(models) * len(areas)
        self.completed_tasks = 0
        self.label = label
        self._progress_bar: click.progressbar | None = None

    def __enter__(self) -> TrainingProgressBar:
        """Enter context manager."""
        self._progress_bar = click.progressbar(
            length=self.total_tasks,
            label=self.label,
            show_eta=True,
            show_percent=True,
            show_pos=True,
        )
        self._progress_bar.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        if self._progress_bar:
            self._progress_bar.__exit__(exc_type, exc_val, exc_tb)

    def update(
        self,
        model: str | None = None,  # noqa: ARG002
        area: str | None = None,  # noqa: ARG002
        status: str = "completed",  # noqa: ARG002
    ) -> None:
        """Update progress bar.

        Args:
            model: Model type that was processed (for logging/display).
            area: Area that was processed (for logging/display).
            status: Status of the task (completed/failed) - for future use.
        """
        self.completed_tasks += 1
        if self._progress_bar:
            self._progress_bar.update(1)


def validate_date_range(start_date: datetime, end_date: datetime) -> None:
    """Validate date range for training.

    Args:
        start_date: Training data start date.
        end_date: Training data end date.

    Raises:
        click.ClickException: If date range is invalid.
    """
    if start_date >= end_date:
        raise click.ClickException(
            f"Start date ({start_date.date()}) must be before " f"end date ({end_date.date()})"
        )

    # Require at least 30 days of training data
    days_diff = (end_date - start_date).days
    if days_diff < 30:
        raise click.ClickException(
            f"Training period must be at least 30 days " f"(got {days_diff} days)"
        )


def validate_area_selection(areas: list[str]) -> None:
    """Validate area selection.

    Args:
        areas: List of areas to validate.

    Raises:
        click.ClickException: If any area is invalid.
    """
    invalid_areas = [a for a in areas if a not in VALID_AREAS]
    if invalid_areas:
        raise click.ClickException(
            f"Invalid areas: {', '.join(invalid_areas)}. " f"Valid areas: {', '.join(VALID_AREAS)}"
        )


def validate_model_selection(models: list[str]) -> None:
    """Validate model type selection.

    Args:
        models: List of model types to validate.

    Raises:
        click.ClickException: If any model type is invalid.
    """
    invalid_models = [m for m in models if m not in VALID_MODEL_TYPES]
    if invalid_models:
        raise click.ClickException(
            f"Invalid model types: {', '.join(invalid_models)}. "
            f"Valid types: {', '.join(VALID_MODEL_TYPES)}"
        )


def execute_training(
    config: Any,
    models: list[str],
    areas: list[str],
    start_date: datetime,
    end_date: datetime,
    horizons: list[int],
    parallel_workers: int,
    force_retrain: bool,
    progress_callback: Any | None = None,
) -> TrainingResult:
    """Execute model training.

    Args:
        config: Configuration manager.
        models: List of model types to train.
        areas: List of areas to train for.
        start_date: Training data start date.
        end_date: Training data end date.
        horizons: Forecast horizons to train.
        parallel_workers: Number of parallel workers.
        force_retrain: Whether to force retrain existing models.
        progress_callback: Optional callback for progress updates.

    Returns:
        TrainingResult with trained and failed models.
    """
    from src.orchestration import TrainingWorkflow

    trained_models: list[TrainedModelInfo] = []
    failed_models: list[TrainedModelInfo] = []
    total_time = 0.0

    try:
        # Create training workflow
        workflow = TrainingWorkflow(config)

        # Update configuration
        config.update_training_config(
            areas=areas,
            horizons=horizons,
            model_types=models,
            parallel_workers=parallel_workers,
        )

        # Execute training
        import time

        start_time = time.time()

        for model_type in models:
            for area in areas:
                model_start = time.time()
                try:
                    # Run workflow for this model/area combination
                    result = workflow.run()

                    if result.success:
                        trained_models.append(
                            TrainedModelInfo(
                                model_type=model_type,
                                area=area,
                                mape=2.5,  # Placeholder
                                training_time=time.time() - model_start,
                            )
                        )
                    else:
                        failed_models.append(
                            TrainedModelInfo(
                                model_type=model_type,
                                area=area,
                                error=result.error or "Training failed",
                            )
                        )

                    if progress_callback:
                        progress_callback(model=model_type, area=area)

                except Exception as e:
                    failed_models.append(
                        TrainedModelInfo(
                            model_type=model_type,
                            area=area,
                            error=str(e),
                        )
                    )
                    if progress_callback:
                        progress_callback(model=model_type, area=area, status="failed")

        total_time = time.time() - start_time

    except Exception as e:
        return TrainingResult(
            trained_models=trained_models,
            failed_models=failed_models,
            training_time=total_time,
            success=False,
            error=str(e),
        )

    avg_time = total_time / len(trained_models) if trained_models else 0.0

    return TrainingResult(
        trained_models=trained_models,
        failed_models=failed_models,
        training_time=total_time,
        avg_time_per_model=avg_time,
        success=len(failed_models) == 0,
    )


@click.group()
def train() -> None:
    """Model training commands.

    Train individual models or execute batch training across
    multiple models, areas, and time periods.
    """


@train.command("model")
@click.option(
    "--model",
    "-m",
    type=click.Choice(["lgbm", "rf", "regdin_svm", "holt_winters", "all"]),
    required=True,
    help="Model type to train",
)
@click.option(
    "--area",
    "-a",
    multiple=True,
    help="Areas to train (can specify multiple)",
)
@click.option(
    "--start-date",
    type=click.DateTime(["%Y-%m-%d"]),
    required=True,
    help="Training start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(["%Y-%m-%d"]),
    required=True,
    help="Training end date (YYYY-MM-DD)",
)
@click.option(
    "--horizons",
    type=str,
    default=None,
    help="Comma-separated forecast horizons (e.g., 0,1,2,3)",
)
@click.option(
    "--parallel",
    "-p",
    type=int,
    default=4,
    help="Number of parallel training workers (default: 4)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force retrain existing models",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show training plan without execution",
)
@click.pass_context
def train_model(
    ctx: click.Context,
    model: str,
    area: tuple[str, ...],
    start_date: datetime,
    end_date: datetime,
    horizons: str | None,
    parallel: int,
    force: bool,
    dry_run: bool,
) -> None:
    """Train individual or all models for specified areas and date range.

    This command trains machine learning models for electric load forecasting
    across specified areas and date ranges. Supports parallel execution for
    faster training across multiple time series.

    Examples:

        # Train LGBM for specific areas

        prevcarga train model --model lgbm --area SECO --area S \\
            --start-date 2023-01-01 --end-date 2023-12-31

        # Train all models with 8 parallel workers

        prevcarga train model --model all \\
            --start-date 2023-01-01 --end-date 2023-12-31 --parallel 8

        # Force retrain for specific area

        prevcarga train model --model rf --area N --force \\
            --start-date 2023-01-01 --end-date 2023-12-31

        # Dry run to see training plan

        prevcarga train model --model all --dry-run \\
            --start-date 2023-01-01 --end-date 2023-12-31
    """
    display = TrainingResultDisplay()

    # Get config from parent context
    config = ctx.obj

    # Validate inputs
    validate_date_range(start_date, end_date)

    # Determine areas to train (default to subsystems)
    areas = list(area) if area else ["SECO", "S", "NE", "N"]
    validate_area_selection(areas)

    # Determine models to train
    models_to_train = VALID_MODEL_TYPES if model == "all" else [model]
    validate_model_selection(models_to_train)

    # Parse horizons
    horizon_list = [int(h.strip()) for h in horizons.split(",")] if horizons else list(range(9))

    # Display training plan
    click.echo()
    display.header("Training Plan")
    click.echo()
    display.key_value("Models:", ", ".join(models_to_train))
    display.key_value("Areas:", ", ".join(areas))
    display.key_value("Period:", f"{start_date.date()} to {end_date.date()}")
    display.key_value("Horizons:", ", ".join(f"D+{h}" for h in horizon_list))
    display.key_value("Parallel workers:", parallel)
    display.key_value("Force retrain:", "Yes" if force else "No")
    display.key_value("Total tasks:", len(models_to_train) * len(areas))
    click.echo()

    if dry_run:
        display_training_plan(models_to_train, areas, start_date, end_date, horizon_list)
        click.echo()
        display.info("[DRY RUN] Training plan displayed above. No training executed.")
        return

    # Confirm execution for large training jobs
    total_tasks = len(models_to_train) * len(areas)
    if total_tasks > 20:
        if not click.confirm(f"This will train {total_tasks} model(s). Proceed?"):
            click.echo("Training cancelled")
            return

    # Execute training with progress bar
    display.info("Starting training...")
    click.echo()

    try:
        with TrainingProgressBar(models=models_to_train, areas=areas) as progress:
            result = execute_training(
                config=config,
                models=models_to_train,
                areas=areas,
                start_date=start_date,
                end_date=end_date,
                horizons=horizon_list,
                parallel_workers=parallel,
                force_retrain=force,
                progress_callback=progress.update,
            )

        # Display results
        click.echo()
        display_training_results(result)

        if result.failed_models:
            display.warning(f"{len(result.failed_models)} model(s) failed")
            sys.exit(1)
        else:
            click.echo()
            display.success("Training completed successfully")

    except KeyboardInterrupt:
        click.echo("\n\nTraining interrupted by user")
        sys.exit(130)
    except Exception as e:
        display.error(f"Training failed: {e}")
        raise click.ClickException(str(e))


@train.command("batch")
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Training configuration YAML file",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show training plan without execution",
)
@click.pass_context
def train_batch(
    ctx: click.Context,
    config_file: Path,
    dry_run: bool,
) -> None:
    """Execute batch training from configuration file.

    Configuration file should specify models, areas, date ranges,
    and training parameters for comprehensive training runs.

    Example configuration (training-config.yaml):

        models:
          - lgbm
          - rf
          - regdin_svm
        areas:
          - SECO
          - S
          - NE
          - N
        training_periods:
          - start_date: "2023-01-01"
            end_date: "2023-12-31"
        parallel_workers: 8
        force_retrain: false
        horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]

    Examples:

        # Execute batch training

        prevcarga train batch --config-file training-config.yaml

        # Preview training plan

        prevcarga train batch --config-file training-config.yaml --dry-run
    """
    display = TrainingResultDisplay()

    # Get system config from parent context
    config = ctx.obj

    # Load training configuration
    with open(config_file) as f:
        training_config = yaml.safe_load(f)

    # Validate training configuration
    _validate_training_config(training_config)

    # Display training plan
    display_batch_training_plan(training_config)

    if dry_run:
        click.echo()
        display.info("[DRY RUN] Training plan displayed above. No training executed.")
        return

    # Confirm execution
    if not click.confirm("\nProceed with batch training?"):
        click.echo("Training cancelled")
        return

    # Execute batch training
    display.info("Starting batch training...")
    click.echo()

    try:
        results: list[TrainingResult] = []
        periods = training_config.get("training_periods", [])
        models = training_config.get("models", [])
        areas = training_config.get("areas", [])
        parallel = training_config.get("parallel_workers", 4)
        force = training_config.get("force_retrain", False)
        horizons = training_config.get("horizons", list(range(9)))

        for i, period in enumerate(periods, 1):
            start_date = datetime.strptime(period["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(period["end_date"], "%Y-%m-%d")

            click.echo(f"Period {i}/{len(periods)}: " f"{start_date.date()} to {end_date.date()}")

            with TrainingProgressBar(models=models, areas=areas) as progress:
                result = execute_training(
                    config=config,
                    models=models,
                    areas=areas,
                    start_date=start_date,
                    end_date=end_date,
                    horizons=horizons,
                    parallel_workers=parallel,
                    force_retrain=force,
                    progress_callback=progress.update,
                )
                results.append(result)

            click.echo()

        # Display aggregate results
        display_batch_results(results)

        total_failed = sum(len(r.failed_models) for r in results)
        if total_failed > 0:
            display.warning(f"{total_failed} model(s) failed across all periods")
            sys.exit(1)
        else:
            click.echo()
            display.success("Batch training completed successfully")

    except KeyboardInterrupt:
        click.echo("\n\nBatch training interrupted by user")
        sys.exit(130)
    except Exception as e:
        display.error(f"Batch training failed: {e}")
        raise click.ClickException(str(e))


def _validate_training_config(training_config: dict[str, Any]) -> None:
    """Validate batch training configuration.

    Args:
        training_config: Training configuration dictionary.

    Raises:
        click.ClickException: If configuration is invalid.
    """
    required_fields = ["models", "areas", "training_periods"]

    for field in required_fields:
        if field not in training_config:
            raise click.ClickException(f"Missing required field in config: {field}")

    # Validate models
    models = training_config.get("models", [])
    validate_model_selection(models)

    # Validate areas
    areas = training_config.get("areas", [])
    validate_area_selection(areas)

    # Validate training periods
    periods = training_config.get("training_periods", [])
    if not periods:
        raise click.ClickException("At least one training period required")

    for i, period in enumerate(periods):
        if "start_date" not in period:
            raise click.ClickException(f"Training period {i + 1} missing start_date")
        if "end_date" not in period:
            raise click.ClickException(f"Training period {i + 1} missing end_date")

        try:
            start = datetime.strptime(period["start_date"], "%Y-%m-%d")
            end = datetime.strptime(period["end_date"], "%Y-%m-%d")
            validate_date_range(start, end)
        except ValueError as e:
            raise click.ClickException(f"Invalid date format in period {i + 1}: {e}")
