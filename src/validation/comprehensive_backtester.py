"""Comprehensive backtesting framework for PrevCarga system.

This module provides the comprehensive backtesting framework for executing
1-year backtests with walk-forward validation, weekly retraining, and
performance monitoring.

Key Features:
- Walk-forward validation with configurable retraining intervals
- Checkpoint management for resume capability
- Performance monitoring and target validation
- Detailed result aggregation and reporting

Example:
    ```python
    from src.validation.comprehensive_backtester import (
        ComprehensiveBacktester,
        BacktestConfig,
    )
    from datetime import datetime

    config = BacktestConfig(
        backtest_year=2024,
        retraining_interval_days=7,
        checkpoint_path="/data/checkpoints",
    )
    backtester = ComprehensiveBacktester(config)

    result = await backtester.execute_yearly_backtest()
    print(f"Average MAPE: {result.average_mape:.3f}")
    ```
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import numpy as np

from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# Default areas and models
DEFAULT_AREAS = ["SECO", "S", "NE", "N", "SIN"]
DEFAULT_MODELS = ["lgbm", "rf", "arima", "holt_winters", "regdin_svm"]


class BacktestExecutionError(Exception):
    """Error during backtest execution."""

    pass


class PerformanceTargetViolation(Exception):
    """Performance targets not met.

    Attributes:
        violations: List of violation messages.
    """

    def __init__(self, violations: list[str]) -> None:
        """Initialize with violations.

        Args:
            violations: List of violation messages.
        """
        self.violations = violations
        super().__init__(f"Performance target violations: {violations}")


@dataclass
class BacktestConfig:
    """Configuration for comprehensive backtesting.

    Attributes:
        backtest_year: Year to backtest.
        retraining_interval_days: Days between retraining cycles.
        initial_training_days: Days of historical data for initial training.
        checkpoint_path: Path to store checkpoints.
        output_path: Path to store results.
        areas: List of areas to backtest.
        models: List of models to backtest.
        target_duration_hours: Maximum allowed backtest duration in hours.
        target_mape: Maximum allowed average MAPE.
        max_training_time_minutes: Maximum training time per period in minutes.
        max_prediction_time_minutes: Maximum prediction time per period in minutes.
    """

    backtest_year: int = 2024
    retraining_interval_days: int = 7
    initial_training_days: int = 365
    checkpoint_path: str | Path = "/tmp/prevcarga/checkpoints"
    output_path: str | Path = "/tmp/prevcarga/results"
    areas: list[str] = field(default_factory=lambda: DEFAULT_AREAS.copy())
    models: list[str] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    target_duration_hours: float = 8.0
    target_mape: float = 0.15
    max_training_time_minutes: float = 30.0
    max_prediction_time_minutes: float = 15.0

    def __post_init__(self) -> None:
        """Validate and convert paths."""
        self.checkpoint_path = Path(self.checkpoint_path)
        self.output_path = Path(self.output_path)

        if self.backtest_year < 2000 or self.backtest_year > 2100:
            raise ValueError(f"Invalid backtest year: {self.backtest_year}")
        if self.retraining_interval_days < 1:
            raise ValueError(f"Retraining interval must be >= 1, got {self.retraining_interval_days}")
        if self.initial_training_days < 30:
            raise ValueError(f"Initial training days must be >= 30, got {self.initial_training_days}")

    @property
    def target_duration_seconds(self) -> float:
        """Get target duration in seconds."""
        return self.target_duration_hours * 3600

    @property
    def max_training_time_seconds(self) -> float:
        """Get max training time in seconds."""
        return self.max_training_time_minutes * 60

    @property
    def max_prediction_time_seconds(self) -> float:
        """Get max prediction time in seconds."""
        return self.max_prediction_time_minutes * 60

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "backtest_year": self.backtest_year,
            "retraining_interval_days": self.retraining_interval_days,
            "initial_training_days": self.initial_training_days,
            "checkpoint_path": str(self.checkpoint_path),
            "output_path": str(self.output_path),
            "areas": self.areas,
            "models": self.models,
            "target_duration_hours": self.target_duration_hours,
            "target_mape": self.target_mape,
            "max_training_time_minutes": self.max_training_time_minutes,
            "max_prediction_time_minutes": self.max_prediction_time_minutes,
        }


@dataclass
class BacktestPeriod:
    """Represents a backtesting period.

    Attributes:
        train_start: Start of training window.
        train_end: End of training window.
        test_start: Start of test window.
        test_end: End of test window.
        period_id: Unique period identifier.
    """

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    period_id: int

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Period {self.period_id}: "
            f"Train [{self.train_start.date()}, {self.train_end.date()}], "
            f"Test [{self.test_start.date()}, {self.test_end.date()}]"
        )

    @property
    def train_days(self) -> int:
        """Get number of training days."""
        return (self.train_end - self.train_start).days + 1

    @property
    def test_days(self) -> int:
        """Get number of test days."""
        return (self.test_end - self.test_start).days + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "period_id": self.period_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_days": self.train_days,
            "test_days": self.test_days,
        }


@dataclass
class BacktestPeriodResult:
    """Results from a single backtest period.

    Attributes:
        period: The backtest period.
        training_result: Training results dictionary.
        predictions: Predictions dictionary.
        evaluation: Evaluation metrics dictionary.
        performance_metrics: Performance metrics dictionary.
        execution_time: Total execution time in seconds.
    """

    period: BacktestPeriod
    training_result: dict[str, Any]
    predictions: dict[str, Any]
    evaluation: dict[str, Any]
    performance_metrics: dict[str, float]
    execution_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "period": self.period.to_dict(),
            "training_duration": self.training_result.get("duration", 0),
            "prediction_duration": self.predictions.get("duration", 0),
            "evaluation_metrics": self.evaluation,
            "performance_metrics": self.performance_metrics,
            "execution_time": self.execution_time,
        }


@dataclass
class YearlyBacktestResult:
    """Aggregated results from yearly backtesting.

    Attributes:
        backtest_year: Year of backtest.
        total_periods: Total number of periods.
        period_results: List of period results.
        total_duration: Total execution duration in seconds.
        average_mape: Average MAPE across all periods.
        summary_metrics: Summary statistics.
        config: Configuration used.
    """

    backtest_year: int
    total_periods: int
    period_results: list[BacktestPeriodResult]
    total_duration: float
    average_mape: float
    summary_metrics: dict[str, float]
    config: BacktestConfig | None = None

    def passes_performance_targets(self) -> bool:
        """Check if performance targets are met.

        Returns:
            True if all targets are met.
        """
        if self.config is None:
            return (
                self.total_duration <= 8 * 3600 and self.average_mape <= 0.15
            )

        return (
            self.total_duration <= self.config.target_duration_seconds
            and self.average_mape <= self.config.target_mape
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "backtest_year": self.backtest_year,
            "total_periods": self.total_periods,
            "total_duration": self.total_duration,
            "average_mape": self.average_mape,
            "summary_metrics": self.summary_metrics,
            "passes_targets": self.passes_performance_targets(),
            "period_results": [r.to_dict() for r in self.period_results],
            "config": self.config.to_dict() if self.config else None,
        }


class PerformanceMonitor:
    """Monitors performance during backtest execution.

    Tracks execution times, resource usage, and performance metrics
    throughout the backtesting process.
    """

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.metrics: dict[str, float] = {}
        self.operation_times: dict[str, list[float]] = {}
        self._start_times: dict[str, float] = {}

    def start_operation(self, operation_name: str) -> None:
        """Start tracking an operation.

        Args:
            operation_name: Name of the operation.
        """
        self._start_times[operation_name] = time.time()

    def end_operation(self, operation_name: str) -> float:
        """End tracking an operation.

        Args:
            operation_name: Name of the operation.

        Returns:
            Duration of the operation in seconds.
        """
        if operation_name not in self._start_times:
            return 0.0

        duration = time.time() - self._start_times[operation_name]
        del self._start_times[operation_name]

        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        self.operation_times[operation_name].append(duration)

        self.metrics[f"{operation_name}_duration"] = duration
        return duration

    @contextmanager
    def track_operation(self, operation_name: str) -> Generator[None, None, None]:
        """Context manager for tracking operation performance.

        Args:
            operation_name: Name of the operation.

        Yields:
            None.
        """
        self.start_operation(operation_name)
        try:
            yield
        finally:
            self.end_operation(operation_name)

    def get_period_metrics(self) -> dict[str, float]:
        """Get metrics for current period.

        Returns:
            Dictionary of current metrics.
        """
        return self.metrics.copy()

    def get_operation_statistics(self, operation_name: str) -> dict[str, float]:
        """Get statistics for an operation.

        Args:
            operation_name: Name of the operation.

        Returns:
            Dictionary with mean, std, min, max times.
        """
        times = self.operation_times.get(operation_name, [])
        if not times:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "count": len(times),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.operation_times.clear()
        self._start_times.clear()


class CheckpointManager:
    """Manages checkpointing for backtest resumption.

    Provides save and load functionality for backtest checkpoints,
    allowing backtests to be resumed from any point.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_path: Directory for checkpoint storage.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure checkpoint directory exists."""
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        last_period: int,
        results: list[BacktestPeriodResult],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save checkpoint to disk.

        Args:
            last_period: Last completed period index.
            results: List of completed period results.
            metadata: Optional metadata to include.

        Returns:
            Path to saved checkpoint file.
        """
        checkpoint_file = self.checkpoint_path / "backtest_checkpoint.json"

        checkpoint_data = {
            "last_period": last_period,
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
            "metadata": metadata or {},
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.debug(f"Checkpoint saved at period {last_period}")
        return checkpoint_file

    def load_checkpoint(self) -> dict[str, Any] | None:
        """Load checkpoint from disk.

        Returns:
            Checkpoint data dictionary, or None if not found.
        """
        checkpoint_file = self.checkpoint_path / "backtest_checkpoint.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self) -> None:
        """Clear existing checkpoint."""
        checkpoint_file = self.checkpoint_path / "backtest_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.debug("Checkpoint cleared")

    def checkpoint_exists(self) -> bool:
        """Check if checkpoint exists.

        Returns:
            True if checkpoint file exists.
        """
        checkpoint_file = self.checkpoint_path / "backtest_checkpoint.json"
        return checkpoint_file.exists()


class ComprehensiveBacktester:
    """Executes comprehensive backtesting with walk-forward validation.

    This class orchestrates the full backtesting process including:
    - Walk-forward validation with weekly retraining
    - Performance monitoring and target validation
    - Checkpoint management for resume capability
    - Result aggregation and reporting

    Attributes:
        config: Backtest configuration.
        performance_monitor: Performance tracking instance.
        checkpoint_manager: Checkpoint management instance.
    """

    def __init__(self, config: BacktestConfig) -> None:
        """Initialize backtester.

        Args:
            config: Backtest configuration.
        """
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        self.checkpoint_manager = CheckpointManager(config.checkpoint_path)

    async def execute_yearly_backtest(
        self,
        backtest_year: int | None = None,
        retraining_interval: int | None = None,
        resume_from_checkpoint: bool = True,
    ) -> YearlyBacktestResult:
        """Execute comprehensive yearly backtesting with performance monitoring.

        Args:
            backtest_year: Year to backtest (default from config).
            retraining_interval: Days between retraining (default from config).
            resume_from_checkpoint: Whether to resume from checkpoint if available.

        Returns:
            YearlyBacktestResult with aggregated metrics.

        Raises:
            BacktestExecutionError: If backtest execution fails.
            PerformanceTargetViolation: If performance targets are not met.
        """
        year = backtest_year or self.config.backtest_year
        interval = retraining_interval or self.config.retraining_interval_days

        logger.info(f"Starting {year} yearly backtest with {interval}-day retraining")

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        # Generate backtest periods
        backtest_periods = self._generate_backtest_periods(
            start_date, end_date, interval
        )

        logger.info(f"Generated {len(backtest_periods)} backtest periods")

        # Check for existing checkpoint
        start_period = 0
        completed_results: list[BacktestPeriodResult] = []

        if resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                start_period = checkpoint["last_period"] + 1
                # Reconstruct results from checkpoint
                completed_results = self._reconstruct_results_from_checkpoint(
                    checkpoint, backtest_periods
                )
                logger.info(f"Resuming from period {start_period}")

        # Execute backtesting with monitoring
        with self.performance_monitor.track_operation("yearly_backtest"):
            results = await self._execute_backtest_periods(
                backtest_periods[start_period:],
                start_period,
                completed_results,
            )

        # Aggregate and analyze results
        yearly_result = self._aggregate_backtest_results(year, results)
        yearly_result.config = self.config

        # Validate performance targets (may raise PerformanceTargetViolation)
        self._validate_performance_targets(yearly_result)

        # Save final results
        self._save_backtest_results(yearly_result)

        # Clear checkpoint on successful completion
        self.checkpoint_manager.clear_checkpoint()

        logger.info(
            f"Yearly backtest complete. Duration: {yearly_result.total_duration:.2f}s, "
            f"Average MAPE: {yearly_result.average_mape:.3f}"
        )

        return yearly_result

    def _generate_backtest_periods(
        self,
        start_date: datetime,
        end_date: datetime,
        retraining_interval: int,
    ) -> list[BacktestPeriod]:
        """Generate walk-forward validation periods.

        Args:
            start_date: Start date of backtest.
            end_date: End date of backtest.
            retraining_interval: Days between retraining.

        Returns:
            List of BacktestPeriod objects.
        """
        periods = []
        period_id = 0

        # Initial training window
        train_start = start_date - timedelta(days=self.config.initial_training_days)
        train_end = start_date - timedelta(days=1)

        current_date = start_date

        while current_date <= end_date:
            test_end = min(
                current_date + timedelta(days=retraining_interval - 1),
                end_date,
            )

            period = BacktestPeriod(
                train_start=train_start,
                train_end=train_end,
                test_start=current_date,
                test_end=test_end,
                period_id=period_id,
            )

            periods.append(period)

            # Update for next period (rolling window)
            period_id += 1
            train_start = train_start + timedelta(days=retraining_interval)
            train_end = test_end
            current_date = test_end + timedelta(days=1)

        return periods

    async def _execute_backtest_periods(
        self,
        periods: list[BacktestPeriod],
        start_index: int,
        completed_results: list[BacktestPeriodResult],
    ) -> list[BacktestPeriodResult]:
        """Execute backtesting for multiple periods.

        Args:
            periods: List of backtest periods to execute.
            start_index: Starting period index.
            completed_results: Previously completed results.

        Returns:
            List of BacktestPeriodResult.

        Raises:
            BacktestExecutionError: If a period fails to execute.
        """
        results = completed_results.copy()
        total_periods = len(periods) + start_index

        for i, period in enumerate(periods, start=start_index):
            logger.info(
                f"Processing backtest period {i + 1}/{total_periods}: {period}"
            )

            try:
                # Execute period backtesting
                period_result = await self._execute_backtest_period(period)
                results.append(period_result)

                # Monitor performance and adjust if needed
                self._monitor_performance_and_adjust(period_result, total_periods)

                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    last_period=i,
                    results=results,
                )

            except Exception as e:
                logger.error(f"Error in period {i + 1}: {e}")
                raise BacktestExecutionError(f"Failed at period {i + 1}") from e

        return results

    async def _execute_backtest_period(
        self,
        period: BacktestPeriod,
    ) -> BacktestPeriodResult:
        """Execute backtesting for a single period.

        Args:
            period: BacktestPeriod to execute.

        Returns:
            BacktestPeriodResult with metrics.
        """
        period_start_time = time.time()

        # 1. Train models on training window
        logger.debug(f"Training models for period {period.period_id}")
        with self.performance_monitor.track_operation("training"):
            training_result = await self._execute_training(period)

        # 2. Generate predictions for test window
        logger.debug(f"Generating predictions for period {period.period_id}")
        with self.performance_monitor.track_operation("prediction"):
            predictions = await self._execute_predictions(period)

        # 3. Evaluate predictions
        logger.debug(f"Evaluating predictions for period {period.period_id}")
        with self.performance_monitor.track_operation("evaluation"):
            evaluation_result = self._evaluate_predictions(period, predictions)

        # 4. Collect performance metrics
        performance_metrics = self.performance_monitor.get_period_metrics()

        period_duration = time.time() - period_start_time

        return BacktestPeriodResult(
            period=period,
            training_result=training_result,
            predictions=predictions,
            evaluation=evaluation_result,
            performance_metrics=performance_metrics,
            execution_time=period_duration,
        )

    async def _execute_training(
        self,
        period: BacktestPeriod,
    ) -> dict[str, Any]:
        """Execute model training for a period.

        Args:
            period: Backtest period.

        Returns:
            Training result dictionary.
        """
        # Simulate training (in real implementation, call TrainingWorkflow)
        start_time = time.time()

        # Simulate async training with small delay
        await asyncio.sleep(0.001)

        duration = time.time() - start_time

        return {
            "status": "completed",
            "duration": duration,
            "models_trained": len(self.config.models),
            "areas": self.config.areas,
            "train_start": period.train_start.isoformat(),
            "train_end": period.train_end.isoformat(),
        }

    async def _execute_predictions(
        self,
        period: BacktestPeriod,
    ) -> dict[str, Any]:
        """Execute predictions for a period.

        Args:
            period: Backtest period.

        Returns:
            Predictions dictionary.
        """
        # Simulate predictions (in real implementation, call PredictionWorkflow)
        start_time = time.time()

        # Simulate async prediction with small delay
        await asyncio.sleep(0.001)

        duration = time.time() - start_time

        # Generate mock predictions
        predictions: dict[str, Any] = {
            "status": "completed",
            "duration": duration,
            "test_start": period.test_start.isoformat(),
            "test_end": period.test_end.isoformat(),
            "horizons": list(range(9)),
            "by_area_model": {},
        }

        for area in self.config.areas:
            for model in self.config.models:
                key = f"{area}_{model}"
                # Mock prediction values
                predictions["by_area_model"][key] = {
                    "values": list(np.random.uniform(10000, 20000, period.test_days)),
                    "horizons": list(range(9)),
                }

        return predictions

    def _evaluate_predictions(
        self,
        period: BacktestPeriod,
        predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate predictions against actuals.

        Args:
            period: Backtest period.
            predictions: Predictions dictionary.

        Returns:
            Evaluation metrics dictionary.
        """
        # Generate mock actuals and calculate metrics
        evaluation: dict[str, Any] = {
            "mape": {},
            "mae": {},
            "rmse": {},
            "period_id": period.period_id,
        }

        for area in self.config.areas:
            for model in self.config.models:
                key = f"{area}_{model}"

                # Mock metrics (realistic ranges for load forecasting)
                evaluation["mape"][key] = np.random.uniform(2.0, 8.0)  # 2-8% MAPE
                evaluation["mae"][key] = np.random.uniform(50, 200)  # MW
                evaluation["rmse"][key] = np.random.uniform(70, 300)  # MW

        return evaluation

    def _aggregate_backtest_results(
        self,
        backtest_year: int,
        period_results: list[BacktestPeriodResult],
    ) -> YearlyBacktestResult:
        """Aggregate results from all backtest periods.

        Args:
            backtest_year: Year of backtest.
            period_results: Results from all periods.

        Returns:
            YearlyBacktestResult with aggregated metrics.
        """
        if not period_results:
            return YearlyBacktestResult(
                backtest_year=backtest_year,
                total_periods=0,
                period_results=[],
                total_duration=0.0,
                average_mape=0.0,
                summary_metrics={},
            )

        total_duration = sum(r.execution_time for r in period_results)

        # Calculate average MAPE across all periods
        all_mapes: list[float] = []
        for result in period_results:
            if "mape" in result.evaluation:
                all_mapes.extend(result.evaluation["mape"].values())

        average_mape = float(np.mean(all_mapes)) / 100 if all_mapes else 0.0

        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(period_results)

        return YearlyBacktestResult(
            backtest_year=backtest_year,
            total_periods=len(period_results),
            period_results=period_results,
            total_duration=total_duration,
            average_mape=average_mape,
            summary_metrics=summary_metrics,
        )

    def _calculate_summary_metrics(
        self,
        period_results: list[BacktestPeriodResult],
    ) -> dict[str, float]:
        """Calculate summary metrics from period results.

        Args:
            period_results: Results from all periods.

        Returns:
            Dictionary of summary metrics.
        """
        if not period_results:
            return {
                "mean_mape": 0.0,
                "std_mape": 0.0,
                "mean_mae": 0.0,
                "mean_rmse": 0.0,
                "mean_training_time": 0.0,
                "mean_prediction_time": 0.0,
                "max_training_time": 0.0,
                "max_prediction_time": 0.0,
            }

        all_metrics: dict[str, list[float]] = {
            "mape": [],
            "mae": [],
            "rmse": [],
            "training_time": [],
            "prediction_time": [],
        }

        for result in period_results:
            eval_metrics = result.evaluation
            all_metrics["mape"].extend(eval_metrics.get("mape", {}).values())
            all_metrics["mae"].extend(eval_metrics.get("mae", {}).values())
            all_metrics["rmse"].extend(eval_metrics.get("rmse", {}).values())
            all_metrics["training_time"].append(
                result.training_result.get("duration", 0)
            )
            all_metrics["prediction_time"].append(
                result.predictions.get("duration", 0)
            )

        return {
            "mean_mape": float(np.mean(all_metrics["mape"])) if all_metrics["mape"] else 0.0,
            "std_mape": float(np.std(all_metrics["mape"])) if all_metrics["mape"] else 0.0,
            "mean_mae": float(np.mean(all_metrics["mae"])) if all_metrics["mae"] else 0.0,
            "mean_rmse": float(np.mean(all_metrics["rmse"])) if all_metrics["rmse"] else 0.0,
            "mean_training_time": float(np.mean(all_metrics["training_time"])) if all_metrics["training_time"] else 0.0,
            "mean_prediction_time": float(np.mean(all_metrics["prediction_time"])) if all_metrics["prediction_time"] else 0.0,
            "max_training_time": float(np.max(all_metrics["training_time"])) if all_metrics["training_time"] else 0.0,
            "max_prediction_time": float(np.max(all_metrics["prediction_time"])) if all_metrics["prediction_time"] else 0.0,
        }

    def _validate_performance_targets(
        self,
        yearly_result: YearlyBacktestResult,
    ) -> None:
        """Validate that performance targets are met.

        Args:
            yearly_result: Aggregated yearly results.

        Raises:
            PerformanceTargetViolation: If targets not met.
        """
        violations: list[str] = []

        if yearly_result.total_duration > self.config.target_duration_seconds:
            violations.append(
                f"Total duration {yearly_result.total_duration:.0f}s "
                f"exceeds target {self.config.target_duration_seconds:.0f}s"
            )

        if yearly_result.average_mape > self.config.target_mape:
            violations.append(
                f"Average MAPE {yearly_result.average_mape:.3f} "
                f"exceeds target {self.config.target_mape}"
            )

        max_training = yearly_result.summary_metrics.get("max_training_time", 0)
        if max_training > self.config.max_training_time_seconds:
            violations.append(
                f"Max training time {max_training:.0f}s "
                f"exceeds target {self.config.max_training_time_seconds:.0f}s"
            )

        if violations:
            logger.warning(f"Performance target violations: {violations}")
            raise PerformanceTargetViolation(violations)

        logger.info("All performance targets met")

    def _monitor_performance_and_adjust(
        self,
        period_result: BacktestPeriodResult,
        total_periods: int,
    ) -> None:
        """Monitor performance and make adjustments if needed.

        Args:
            period_result: Result from current period.
            total_periods: Total number of periods.
        """
        execution_time = period_result.execution_time

        # Check if period execution is taking too long
        expected_time_per_period = self.config.target_duration_seconds / total_periods

        if execution_time > expected_time_per_period * 1.5:
            logger.warning(
                f"Period {period_result.period.period_id} took "
                f"{execution_time:.0f}s (expected ~{expected_time_per_period:.0f}s)"
            )

    def _reconstruct_results_from_checkpoint(
        self,
        checkpoint: dict[str, Any],
        periods: list[BacktestPeriod],
    ) -> list[BacktestPeriodResult]:
        """Reconstruct BacktestPeriodResult objects from checkpoint data.

        Args:
            checkpoint: Checkpoint data dictionary.
            periods: List of all backtest periods.

        Returns:
            List of reconstructed BacktestPeriodResult objects.
        """
        results: list[BacktestPeriodResult] = []

        for i, result_data in enumerate(checkpoint.get("results", [])):
            if i < len(periods):
                period = periods[i]
            else:
                # Reconstruct period from data if needed
                period_data = result_data.get("period", {})
                period = BacktestPeriod(
                    train_start=datetime.fromisoformat(period_data.get("train_start", "2023-01-01")),
                    train_end=datetime.fromisoformat(period_data.get("train_end", "2023-12-31")),
                    test_start=datetime.fromisoformat(period_data.get("test_start", "2024-01-01")),
                    test_end=datetime.fromisoformat(period_data.get("test_end", "2024-01-07")),
                    period_id=period_data.get("period_id", i),
                )

            result = BacktestPeriodResult(
                period=period,
                training_result={"duration": result_data.get("training_duration", 0)},
                predictions={"duration": result_data.get("prediction_duration", 0)},
                evaluation=result_data.get("evaluation_metrics", {}),
                performance_metrics=result_data.get("performance_metrics", {}),
                execution_time=result_data.get("execution_time", 0),
            )
            results.append(result)

        return results

    def _save_backtest_results(
        self,
        yearly_result: YearlyBacktestResult,
    ) -> Path:
        """Save backtest results to disk.

        Args:
            yearly_result: Aggregated yearly results.

        Returns:
            Path to saved results file.
        """
        output_path = self.config.output_path / "backtest_results"
        output_path.mkdir(parents=True, exist_ok=True)

        result_file = output_path / f"backtest_{yearly_result.backtest_year}.json"

        with open(result_file, "w") as f:
            json.dump(yearly_result.to_dict(), f, indent=2)

        logger.info(f"Backtest results saved to {result_file}")
        return result_file

    def get_performance_statistics(self) -> dict[str, Any]:
        """Get performance statistics from the monitor.

        Returns:
            Dictionary of performance statistics.
        """
        return {
            "training": self.performance_monitor.get_operation_statistics("training"),
            "prediction": self.performance_monitor.get_operation_statistics("prediction"),
            "evaluation": self.performance_monitor.get_operation_statistics("evaluation"),
            "yearly_backtest": self.performance_monitor.get_operation_statistics("yearly_backtest"),
        }
