"""Backtesting workflow orchestration for PrevCarga.

This module provides a high-level backtesting workflow that orchestrates
time-series cross-validation, rolling window evaluation, and comprehensive
performance analysis for forecast models.

Key Components:
- BacktestingWorkflow: Main workflow orchestration class
- BacktestingStage: Backtesting workflow stages
- BacktestingResult: Backtesting workflow results
- BacktestWindow: Individual backtest window result

Example:
    ```python
    from src.orchestration import ConfigManager
    from src.orchestration.backtesting_workflow import BacktestingWorkflow

    # Load configuration
    config_manager = ConfigManager.from_yaml("config/prevcarga.yaml")

    # Create and run workflow
    workflow = BacktestingWorkflow(config_manager)
    result = workflow.run(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
    )

    # Check results
    print(f"Success: {result.success}")
    print(f"Mean MAPE: {result.aggregate_metrics.get('mape', {}).get('mean', 'N/A')}")
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.orchestration.config_manager import (
    BacktestingWorkflowConfig,
    ConfigManager,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestingStage(Enum):
    """Backtesting workflow stages.

    Attributes:
        INITIALIZED: Workflow initialized but not started.
        LOADING_DATA: Loading historical data.
        LOADING_MODELS: Loading trained models.
        GENERATING_WINDOWS: Generating backtest windows.
        EVALUATING: Evaluating each backtest window.
        AGGREGATING_METRICS: Aggregating metrics across windows.
        GENERATING_REPORT: Generating analysis report.
        SAVING_RESULTS: Saving backtest results.
        COMPLETED: Workflow completed successfully.
        FAILED: Workflow failed.
        CANCELLED: Workflow was cancelled.
    """

    INITIALIZED = "initialized"
    LOADING_DATA = "loading_data"
    LOADING_MODELS = "loading_models"
    GENERATING_WINDOWS = "generating_windows"
    EVALUATING = "evaluating"
    AGGREGATING_METRICS = "aggregating_metrics"
    GENERATING_REPORT = "generating_report"
    SAVING_RESULTS = "saving_results"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BacktestingStatus(Enum):
    """Backtesting workflow execution status.

    Attributes:
        PENDING: Workflow is pending start.
        RUNNING: Workflow is currently running.
        PAUSED: Workflow is paused.
        COMPLETED: Workflow completed successfully.
        FAILED: Workflow failed.
        CANCELLED: Workflow was cancelled.
    """

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StageResult:
    """Result of a workflow stage.

    Attributes:
        stage: The workflow stage.
        success: Whether the stage succeeded.
        started_at: When the stage started.
        completed_at: When the stage completed.
        duration_seconds: Duration in seconds.
        error: Error message if failed.
        metrics: Stage-specific metrics.
    """

    stage: BacktestingStage
    success: bool = True
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "stage": self.stage.value,
            "success": self.success,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "metrics": self.metrics,
        }


@dataclass
class BacktestWindow:
    """Result from a single backtest window.

    Attributes:
        window_id: Unique window identifier.
        forecast_date: Date for which forecast was made.
        train_start: Training period start.
        train_end: Training period end.
        test_start: Test period start.
        test_end: Test period end.
        metrics: Performance metrics for this window.
        forecasts: Forecast values.
        actuals: Actual values.
        errors: Forecast errors.
    """

    window_id: int
    forecast_date: datetime
    train_start: datetime | None = None
    train_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    forecasts: dict[str, np.ndarray] = field(default_factory=dict)
    actuals: dict[str, np.ndarray] = field(default_factory=dict)
    errors: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "window_id": self.window_id,
            "forecast_date": self.forecast_date.isoformat(),
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "test_start": self.test_start.isoformat() if self.test_start else None,
            "test_end": self.test_end.isoformat() if self.test_end else None,
            "metrics": self.metrics,
        }


@dataclass
class BacktestingResult:
    """Result of backtesting workflow execution.

    Attributes:
        success: Whether workflow completed successfully.
        status: Final workflow status.
        started_at: When workflow started.
        completed_at: When workflow completed.
        duration_seconds: Total duration in seconds.
        stage_results: Results for each stage.
        windows: Individual backtest window results.
        aggregate_metrics: Aggregated metrics across all windows.
        metrics_by_horizon: Metrics broken down by forecast horizon.
        metrics_by_area: Metrics broken down by area.
        report_path: Path to generated report.
        results_path: Path to saved results.
        error: Error message if failed.
    """

    success: bool = False
    status: BacktestingStatus = BacktestingStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    stage_results: list[StageResult] = field(default_factory=list)
    windows: list[BacktestWindow] = field(default_factory=list)
    aggregate_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    metrics_by_horizon: dict[int, dict[str, float]] = field(default_factory=dict)
    metrics_by_area: dict[str, dict[str, float]] = field(default_factory=dict)
    report_path: str | None = None
    results_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "success": self.success,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "stage_results": [s.to_dict() for s in self.stage_results],
            "n_windows": len(self.windows),
            "aggregate_metrics": self.aggregate_metrics,
            "metrics_by_horizon": self.metrics_by_horizon,
            "metrics_by_area": self.metrics_by_area,
            "report_path": self.report_path,
            "results_path": self.results_path,
            "error": self.error,
        }

    def get_summary(self) -> str:
        """Get human-readable summary.

        Returns:
            Summary string.
        """
        lines = [
            "=" * 60,
            "Backtesting Workflow Summary",
            "=" * 60,
            f"Status: {self.status.value.upper()}",
            f"Success: {self.success}",
            f"Duration: {self.duration_seconds:.2f} seconds",
            f"Windows Evaluated: {len(self.windows)}",
            "",
            "Stage Results:",
        ]

        for stage in self.stage_results:
            status = "OK" if stage.success else "FAILED"
            lines.append(f"  - {stage.stage.value}: {status} ({stage.duration_seconds:.1f}s)")

        if self.aggregate_metrics:
            lines.append("")
            lines.append("Aggregate Metrics:")
            for metric, values in self.aggregate_metrics.items():
                if isinstance(values, dict) and "mean" in values:
                    lines.append(f"  - {metric}: {values['mean']:.4f} (std: {values.get('std', 0):.4f})")

        if self.report_path:
            lines.append("")
            lines.append(f"Report: {self.report_path}")

        if self.error:
            lines.append("")
            lines.append(f"Error: {self.error}")

        lines.append("=" * 60)
        return "\n".join(lines)


class BacktestingWorkflow:
    """High-level backtesting workflow orchestration.

    Orchestrates comprehensive backtesting including:
    - Rolling window generation
    - Time series cross-validation
    - Multi-horizon evaluation
    - Metrics aggregation
    - Performance reporting

    Example:
        >>> config_manager = ConfigManager.from_yaml("config/prevcarga.yaml")
        >>> workflow = BacktestingWorkflow(config_manager)
        >>> result = workflow.run(
        ...     start_date=datetime(2023, 1, 1),
        ...     end_date=datetime(2023, 12, 31),
        ... )
        >>> print(result.get_summary())
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        data_loader: Any | None = None,
        model_loader: Any | None = None,
        metrics_calculator: Any | None = None,
    ) -> None:
        """Initialize backtesting workflow.

        Args:
            config_manager: Configuration manager instance.
            data_loader: Optional data loader instance.
            model_loader: Optional model loader instance.
            metrics_calculator: Optional metrics calculator instance.
        """
        self._config_manager = config_manager
        self._data_loader = data_loader
        self._model_loader = model_loader
        self._metrics_calculator = metrics_calculator

        self._status = BacktestingStatus.PENDING
        self._current_stage = BacktestingStage.INITIALIZED
        self._result = BacktestingResult()
        self._cancelled = False
        self._paused = False

        self._historical_data: dict[str, pd.DataFrame] = {}
        self._models: dict[str, dict[int, Any]] = {}
        self._backtest_dates: list[datetime] = []

        logger.info("Initialized BacktestingWorkflow")

    @property
    def status(self) -> BacktestingStatus:
        """Get current workflow status."""
        return self._status

    @property
    def current_stage(self) -> BacktestingStage:
        """Get current workflow stage."""
        return self._current_stage

    def run(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        historical_data: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestingResult:
        """Run the complete backtesting workflow.

        Args:
            start_date: Start date for backtesting period.
            end_date: End date for backtesting period.
            historical_data: Pre-loaded historical data by area.

        Returns:
            BacktestingResult with evaluation outcomes.
        """
        self._result = BacktestingResult()
        self._result.started_at = datetime.now()
        self._status = BacktestingStatus.RUNNING

        # Check if already cancelled before starting
        if self._cancelled:
            return self._finalize_result(success=False)

        config = self._config_manager.get_backtesting_config()

        # Use config dates if not provided
        if start_date is None:
            start_date = self._parse_date(config.start_date) if config.start_date else None
        if end_date is None:
            end_date = self._parse_date(config.end_date) if config.end_date else None

        # Default dates if still None
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now() - timedelta(days=1)

        try:
            # Stage 1: Load Historical Data
            if not self._execute_stage(
                BacktestingStage.LOADING_DATA,
                self._load_historical_data,
                historical_data,
            ):
                return self._finalize_result(success=False)

            # Stage 2: Load Models
            if not self._execute_stage(
                BacktestingStage.LOADING_MODELS,
                self._load_models,
            ):
                return self._finalize_result(success=False)

            # Stage 3: Generate Backtest Windows
            if not self._execute_stage(
                BacktestingStage.GENERATING_WINDOWS,
                self._generate_windows,
                start_date,
                end_date,
            ):
                return self._finalize_result(success=False)

            # Stage 4: Evaluate Each Window
            if not self._execute_stage(
                BacktestingStage.EVALUATING,
                self._evaluate_windows,
            ):
                return self._finalize_result(success=False)

            # Stage 5: Aggregate Metrics
            if not self._execute_stage(
                BacktestingStage.AGGREGATING_METRICS,
                self._aggregate_metrics,
            ):
                return self._finalize_result(success=False)

            # Stage 6: Generate Report (optional)
            if config.generate_report:
                if not self._execute_stage(
                    BacktestingStage.GENERATING_REPORT,
                    self._generate_report,
                ):
                    return self._finalize_result(success=False)

            # Stage 7: Save Results
            if not self._execute_stage(
                BacktestingStage.SAVING_RESULTS,
                self._save_results,
            ):
                return self._finalize_result(success=False)

            return self._finalize_result(success=True)

        except Exception as e:
            logger.exception("Backtesting workflow failed with exception")
            self._result.error = str(e)
            return self._finalize_result(success=False)

    def cancel(self) -> None:
        """Cancel the running workflow."""
        logger.info("Cancelling backtesting workflow")
        self._cancelled = True
        self._status = BacktestingStatus.CANCELLED

    def pause(self) -> None:
        """Pause the running workflow."""
        logger.info("Pausing backtesting workflow")
        self._paused = True
        self._status = BacktestingStatus.PAUSED

    def resume(self) -> None:
        """Resume a paused workflow."""
        if self._status == BacktestingStatus.PAUSED:
            logger.info("Resuming backtesting workflow")
            self._paused = False
            self._status = BacktestingStatus.RUNNING

    def _execute_stage(
        self,
        stage: BacktestingStage,
        stage_func: callable,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Execute a workflow stage with timing and error handling.

        Args:
            stage: The stage to execute.
            stage_func: The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            True if stage succeeded, False otherwise.
        """
        if self._cancelled:
            return False

        while self._paused:
            import time
            time.sleep(0.1)
            if self._cancelled:
                return False

        self._current_stage = stage
        stage_result = StageResult(stage=stage)
        stage_result.started_at = datetime.now()

        logger.info("Starting stage: %s", stage.value)

        try:
            stage_func(*args, **kwargs)
            stage_result.success = True
        except Exception as e:
            logger.exception("Stage %s failed", stage.value)
            stage_result.success = False
            stage_result.error = str(e)
            self._result.error = f"Stage {stage.value} failed: {e}"

        stage_result.completed_at = datetime.now()
        stage_result.duration_seconds = (
            stage_result.completed_at - stage_result.started_at
        ).total_seconds()

        self._result.stage_results.append(stage_result)

        logger.info(
            "Stage %s completed: success=%s, duration=%.2fs",
            stage.value,
            stage_result.success,
            stage_result.duration_seconds,
        )

        return stage_result.success

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime.

        Args:
            date_str: Date string in YYYY-MM-DD format.

        Returns:
            Parsed datetime.
        """
        return datetime.strptime(date_str, "%Y-%m-%d")

    def _load_historical_data(
        self,
        historical_data: dict[str, pd.DataFrame] | None,
    ) -> None:
        """Load historical data for backtesting.

        Args:
            historical_data: Pre-loaded data or None.
        """
        config = self._config_manager.get_backtesting_config()

        if historical_data is not None:
            self._historical_data = historical_data
            logger.info("Using provided historical data for %d areas", len(historical_data))
            return

        if self._data_loader is not None:
            self._historical_data = self._data_loader.load_historical_data(
                areas=config.areas,
            )
            logger.info("Loaded historical data for %d areas", len(self._historical_data))
            return

        # Generate synthetic data for testing
        logger.warning("No data loader, generating synthetic historical data")
        self._historical_data = self._generate_synthetic_data(config)

    def _generate_synthetic_data(
        self,
        config: BacktestingWorkflowConfig,
    ) -> dict[str, pd.DataFrame]:
        """Generate synthetic historical data for testing.

        Args:
            config: Backtesting configuration.

        Returns:
            Dictionary of synthetic DataFrames.
        """
        rng = np.random.default_rng(42)
        n_days = 365
        n_periods = n_days * 48  # 48 half-hours per day

        data = {}
        for area in config.areas:
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=n_days),
                periods=n_periods,
                freq="30min",
            )

            # Generate synthetic load with daily and weekly patterns
            base_load = 1000 + rng.standard_normal(n_periods) * 100
            hour_pattern = np.sin(2 * np.pi * np.arange(n_periods) / 48) * 200
            weekly_pattern = np.sin(2 * np.pi * np.arange(n_periods) / (48 * 7)) * 100

            load = base_load + hour_pattern + weekly_pattern + 1500

            df = pd.DataFrame({
                "timestamp": dates,
                "load_mw": load,
                "temperature": 20 + rng.standard_normal(n_periods) * 5,
            })
            df.set_index("timestamp", inplace=True)
            data[area] = df

        return data

    def _load_models(self) -> None:
        """Load models for backtesting."""
        config = self._config_manager.get_backtesting_config()

        if self._model_loader is not None:
            self._models = self._model_loader.load_models(
                areas=config.areas,
                horizons=config.horizons,
            )
            logger.info("Loaded models for %d areas", len(self._models))
            return

        # Create placeholder models for testing
        logger.warning("No model loader, creating placeholder models")
        self._models = self._create_placeholder_models(config)

    def _create_placeholder_models(
        self,
        config: BacktestingWorkflowConfig,
    ) -> dict[str, dict[int, Any]]:
        """Create placeholder models for testing.

        Args:
            config: Backtesting configuration.

        Returns:
            Dictionary of placeholder models.
        """

        class PlaceholderModel:
            """Simple placeholder model."""

            def __init__(self, area: str, horizon: int) -> None:
                self.area = area
                self.horizon = horizon
                self._rng = np.random.default_rng(42 + horizon)

            def predict(self, X: pd.DataFrame) -> np.ndarray:
                """Generate predictions with some noise."""
                if "load_mw" in X.columns:
                    base = X["load_mw"].values
                else:
                    base = np.ones(len(X)) * 1500

                noise = self._rng.standard_normal(len(X)) * 50 * (1 + self.horizon * 0.1)
                return base + noise

        models: dict[str, dict[int, Any]] = {}
        for area in config.areas:
            models[area] = {}
            for horizon in config.horizons:
                models[area][horizon] = PlaceholderModel(area, horizon)

        return models

    def _generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """Generate backtest windows.

        Args:
            start_date: Start date.
            end_date: End date.
        """
        config = self._config_manager.get_backtesting_config()
        step = timedelta(days=config.step_days)

        self._backtest_dates = []
        current = start_date

        while current <= end_date:
            self._backtest_dates.append(current)
            current += step

        logger.info(
            "Generated %d backtest windows from %s to %s (step: %d days)",
            len(self._backtest_dates),
            start_date.date(),
            end_date.date(),
            config.step_days,
        )

    def _evaluate_windows(self) -> None:
        """Evaluate all backtest windows."""
        config = self._config_manager.get_backtesting_config()

        for i, forecast_date in enumerate(self._backtest_dates):
            if self._cancelled:
                break

            while self._paused:
                import time
                time.sleep(0.1)
                if self._cancelled:
                    break

            window = self._evaluate_single_window(
                window_id=i,
                forecast_date=forecast_date,
                config=config,
            )
            self._result.windows.append(window)

            if (i + 1) % 10 == 0:
                logger.info("Evaluated %d/%d windows", i + 1, len(self._backtest_dates))

        logger.info("Completed evaluation of %d windows", len(self._result.windows))

    def _evaluate_single_window(
        self,
        window_id: int,
        forecast_date: datetime,
        config: BacktestingWorkflowConfig,
    ) -> BacktestWindow:
        """Evaluate a single backtest window.

        Args:
            window_id: Window identifier.
            forecast_date: Date for forecast.
            config: Backtesting configuration.

        Returns:
            BacktestWindow with evaluation results.
        """
        window = BacktestWindow(
            window_id=window_id,
            forecast_date=forecast_date,
        )

        # Define train/test periods
        window.train_end = forecast_date - timedelta(days=1)
        window.train_start = window.train_end - timedelta(days=365)
        window.test_start = forecast_date
        window.test_end = forecast_date + timedelta(days=max(config.horizons))

        # Evaluate for each area
        for area in config.areas:
            if area not in self._historical_data:
                continue

            area_data = self._historical_data[area]

            # Get actual values for test period
            test_mask = (
                (area_data.index >= window.test_start) &
                (area_data.index <= window.test_end)
            )
            if test_mask.sum() == 0:
                continue

            test_data = area_data[test_mask]
            if "load_mw" not in test_data.columns:
                continue

            actuals = test_data["load_mw"].values

            # Generate forecasts for each horizon
            area_metrics = {}
            for horizon in config.horizons:
                if area not in self._models or horizon not in self._models[area]:
                    continue

                model = self._models[area][horizon]

                try:
                    # Get training features
                    train_mask = (
                        (area_data.index >= window.train_start) &
                        (area_data.index < window.train_end)
                    )
                    train_data = area_data[train_mask].tail(48)  # Last day of training

                    if len(train_data) == 0:
                        continue

                    predictions = model.predict(train_data)

                    # Align predictions with actuals
                    n_compare = min(len(predictions), len(actuals))
                    if n_compare == 0:
                        continue

                    pred_slice = predictions[:n_compare]
                    actual_slice = actuals[:n_compare]

                    # Calculate metrics
                    horizon_metrics = self._calculate_metrics(actual_slice, pred_slice)
                    area_metrics[f"h{horizon}"] = horizon_metrics

                except Exception as e:
                    logger.warning(
                        "Error evaluating window %d, area %s, horizon %d: %s",
                        window_id, area, horizon, e
                    )

            if area_metrics:
                window.metrics[area] = area_metrics

        return window

    def _calculate_metrics(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        """Calculate forecast evaluation metrics.

        Args:
            actuals: Actual values.
            predictions: Predicted values.

        Returns:
            Dictionary of metric values.
        """
        if self._metrics_calculator is not None:
            return self._metrics_calculator.calculate(actuals, predictions)

        # Basic metric calculations
        errors = actuals - predictions
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2

        # Avoid division by zero
        nonzero_actuals = actuals[actuals != 0]
        nonzero_predictions = predictions[actuals != 0]

        metrics = {
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(squared_errors))),
            "bias": float(np.mean(errors)),
        }

        if len(nonzero_actuals) > 0:
            mape = np.mean(np.abs(actuals[actuals != 0] - predictions[actuals != 0]) / np.abs(nonzero_actuals)) * 100
            metrics["mape"] = float(mape)

        return metrics

    def _aggregate_metrics(self) -> None:
        """Aggregate metrics across all windows."""
        config = self._config_manager.get_backtesting_config()

        # Collect all metrics
        all_metrics: dict[str, list[float]] = {metric: [] for metric in config.metrics}
        metrics_by_horizon: dict[int, dict[str, list[float]]] = {}
        metrics_by_area: dict[str, dict[str, list[float]]] = {}

        for window in self._result.windows:
            for area, area_metrics in window.metrics.items():
                if area not in metrics_by_area:
                    metrics_by_area[area] = {m: [] for m in config.metrics}

                for horizon_key, horizon_metrics in area_metrics.items():
                    # Extract horizon number
                    horizon = int(horizon_key[1:]) if horizon_key.startswith("h") else 0

                    if horizon not in metrics_by_horizon:
                        metrics_by_horizon[horizon] = {m: [] for m in config.metrics}

                    for metric_name in config.metrics:
                        if metric_name in horizon_metrics:
                            value = horizon_metrics[metric_name]
                            all_metrics[metric_name].append(value)
                            metrics_by_horizon[horizon][metric_name].append(value)
                            metrics_by_area[area][metric_name].append(value)

        # Calculate aggregate statistics
        for metric_name, values in all_metrics.items():
            if values:
                self._result.aggregate_metrics[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "count": len(values),
                }

        # Metrics by horizon
        for horizon, horizon_metrics in metrics_by_horizon.items():
            self._result.metrics_by_horizon[horizon] = {}
            for metric_name, values in horizon_metrics.items():
                if values:
                    self._result.metrics_by_horizon[horizon][metric_name] = float(np.mean(values))

        # Metrics by area
        for area, area_metrics in metrics_by_area.items():
            self._result.metrics_by_area[area] = {}
            for metric_name, values in area_metrics.items():
                if values:
                    self._result.metrics_by_area[area][metric_name] = float(np.mean(values))

        logger.info("Aggregated metrics across %d windows", len(self._result.windows))

    def _generate_report(self) -> None:
        """Generate backtesting analysis report."""
        config = self._config_manager.get_backtesting_config()
        report_dir = Path(config.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"backtest_report_{timestamp}.html"

        # Generate HTML report
        html = self._generate_html_report()

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        self._result.report_path = str(report_path)
        logger.info("Generated report: %s", report_path)

    def _generate_html_report(self) -> str:
        """Generate HTML report content.

        Returns:
            HTML string.
        """
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>PrevCarga Backtesting Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1 { color: #333; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            ".summary { background-color: #e7f3fe; padding: 15px; border-radius: 5px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>PrevCarga Backtesting Report</h1>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "",
            "<div class='summary'>",
            f"<p><strong>Windows Evaluated:</strong> {len(self._result.windows)}</p>",
            f"<p><strong>Duration:</strong> {self._result.duration_seconds:.2f} seconds</p>",
            "</div>",
            "",
            "<h2>Aggregate Metrics</h2>",
            "<table>",
            "<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>",
        ]

        for metric_name, stats in self._result.aggregate_metrics.items():
            lines.append(
                f"<tr><td>{metric_name.upper()}</td>"
                f"<td>{stats.get('mean', 'N/A'):.4f}</td>"
                f"<td>{stats.get('std', 'N/A'):.4f}</td>"
                f"<td>{stats.get('min', 'N/A'):.4f}</td>"
                f"<td>{stats.get('max', 'N/A'):.4f}</td></tr>"
            )

        lines.extend([
            "</table>",
            "",
            "<h2>Metrics by Horizon</h2>",
            "<table>",
            "<tr><th>Horizon</th><th>MAPE</th><th>MAE</th><th>RMSE</th></tr>",
        ])

        for horizon in sorted(self._result.metrics_by_horizon.keys()):
            metrics = self._result.metrics_by_horizon[horizon]
            lines.append(
                f"<tr><td>D+{horizon}</td>"
                f"<td>{metrics.get('mape', 'N/A'):.2f}%</td>"
                f"<td>{metrics.get('mae', 'N/A'):.2f}</td>"
                f"<td>{metrics.get('rmse', 'N/A'):.2f}</td></tr>"
            )

        lines.extend([
            "</table>",
            "",
            "<h2>Metrics by Area</h2>",
            "<table>",
            "<tr><th>Area</th><th>MAPE</th><th>MAE</th><th>RMSE</th></tr>",
        ])

        for area in sorted(self._result.metrics_by_area.keys()):
            metrics = self._result.metrics_by_area[area]
            lines.append(
                f"<tr><td>{area}</td>"
                f"<td>{metrics.get('mape', 'N/A'):.2f}%</td>"
                f"<td>{metrics.get('mae', 'N/A'):.2f}</td>"
                f"<td>{metrics.get('rmse', 'N/A'):.2f}</td></tr>"
            )

        lines.extend([
            "</table>",
            "</body>",
            "</html>",
        ])

        return "\n".join(lines)

    def _save_results(self) -> None:
        """Save backtesting results."""
        config = self._config_manager.get_backtesting_config()

        if not config.save_forecasts:
            logger.info("Forecast saving disabled in configuration")
            return

        forecasts_dir = Path(config.forecasts_dir)
        forecasts_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save aggregate results as JSON
        results_path = forecasts_dir / f"backtest_results_{timestamp}.json"
        import json

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self._result.to_dict(), f, indent=2, default=str)

        self._result.results_path = str(results_path)

        # Optionally save window details as CSV
        if self._result.windows:
            windows_data = []
            for window in self._result.windows:
                for area, area_metrics in window.metrics.items():
                    for horizon_key, metrics in area_metrics.items():
                        row = {
                            "window_id": window.window_id,
                            "forecast_date": window.forecast_date,
                            "area": area,
                            "horizon": horizon_key,
                            **metrics,
                        }
                        windows_data.append(row)

            if windows_data:
                df = pd.DataFrame(windows_data)
                csv_path = forecasts_dir / f"backtest_details_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                logger.info("Saved detailed results to %s", csv_path)

        logger.info("Saved results to %s", results_path)

    def _finalize_result(self, success: bool) -> BacktestingResult:
        """Finalize workflow result.

        Args:
            success: Whether workflow succeeded.

        Returns:
            Finalized BacktestingResult.
        """
        self._result.completed_at = datetime.now()
        self._result.duration_seconds = (
            self._result.completed_at - self._result.started_at
        ).total_seconds()
        self._result.success = success

        if success:
            self._status = BacktestingStatus.COMPLETED
            self._current_stage = BacktestingStage.COMPLETED
            self._result.status = BacktestingStatus.COMPLETED
        elif self._cancelled:
            self._status = BacktestingStatus.CANCELLED
            self._current_stage = BacktestingStage.CANCELLED
            self._result.status = BacktestingStatus.CANCELLED
        else:
            self._status = BacktestingStatus.FAILED
            self._current_stage = BacktestingStage.FAILED
            self._result.status = BacktestingStatus.FAILED

        logger.info(
            "Backtesting workflow completed: success=%s, duration=%.2fs, windows=%d",
            success,
            self._result.duration_seconds,
            len(self._result.windows),
        )

        return self._result

    def get_progress(self) -> dict[str, Any]:
        """Get current workflow progress.

        Returns:
            Progress information dictionary.
        """
        stages = list(BacktestingStage)
        current_idx = stages.index(self._current_stage)
        total_stages = len([s for s in stages if s not in {
            BacktestingStage.COMPLETED, BacktestingStage.FAILED, BacktestingStage.CANCELLED
        }])

        windows_progress = 0
        if self._backtest_dates:
            windows_progress = len(self._result.windows) / len(self._backtest_dates) * 100

        return {
            "status": self._status.value,
            "current_stage": self._current_stage.value,
            "stage_index": current_idx,
            "total_stages": total_stages,
            "stage_progress_percent": (current_idx / total_stages) * 100 if total_stages > 0 else 0,
            "windows_evaluated": len(self._result.windows),
            "windows_total": len(self._backtest_dates),
            "windows_progress_percent": windows_progress,
            "completed_stages": [s.to_dict() for s in self._result.stage_results],
        }

    def get_result(self) -> BacktestingResult:
        """Get workflow result.

        Returns:
            Current workflow result.
        """
        return self._result

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BacktestingWorkflow(status={self._status.value}, stage={self._current_stage.value})"
