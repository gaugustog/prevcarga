"""Tests for the comprehensive backtester module.

Tests cover:
- BacktestConfig validation and defaults
- BacktestPeriod generation and properties
- BacktestPeriodResult serialization
- YearlyBacktestResult aggregation
- PerformanceMonitor tracking
- CheckpointManager save/load/clear
- ComprehensiveBacktester execution
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.validation.comprehensive_backtester import (
    BacktestConfig,
    BacktestExecutionError,
    BacktestPeriod,
    BacktestPeriodResult,
    CheckpointManager,
    ComprehensiveBacktester,
    PerformanceMonitor,
    PerformanceTargetViolation,
    YearlyBacktestResult,
)


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.backtest_year == 2024
        assert config.retraining_interval_days == 7
        assert config.initial_training_days == 365
        assert config.target_duration_hours == 8.0
        assert config.target_mape == 0.15
        assert len(config.areas) == 5
        assert len(config.models) == 5

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BacktestConfig(
            backtest_year=2023,
            retraining_interval_days=14,
            initial_training_days=180,
            target_duration_hours=4.0,
            target_mape=0.10,
            areas=["SECO", "S"],
            models=["lgbm", "rf"],
        )

        assert config.backtest_year == 2023
        assert config.retraining_interval_days == 14
        assert config.initial_training_days == 180
        assert config.target_duration_hours == 4.0
        assert config.target_mape == 0.10
        assert config.areas == ["SECO", "S"]
        assert config.models == ["lgbm", "rf"]

    def test_path_conversion(self) -> None:
        """Test path string to Path conversion."""
        config = BacktestConfig(
            checkpoint_path="/tmp/checkpoints",
            output_path="/tmp/output",
        )

        assert isinstance(config.checkpoint_path, Path)
        assert isinstance(config.output_path, Path)

    def test_invalid_year(self) -> None:
        """Test validation of invalid year."""
        with pytest.raises(ValueError, match="Invalid backtest year"):
            BacktestConfig(backtest_year=1900)

    def test_invalid_retraining_interval(self) -> None:
        """Test validation of invalid retraining interval."""
        with pytest.raises(ValueError, match="Retraining interval"):
            BacktestConfig(retraining_interval_days=0)

    def test_invalid_initial_training_days(self) -> None:
        """Test validation of invalid initial training days."""
        with pytest.raises(ValueError, match="Initial training days"):
            BacktestConfig(initial_training_days=10)

    def test_target_duration_seconds(self) -> None:
        """Test target duration conversion to seconds."""
        config = BacktestConfig(target_duration_hours=8.0)
        assert config.target_duration_seconds == 28800.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = BacktestConfig()
        result = config.to_dict()

        assert "backtest_year" in result
        assert "retraining_interval_days" in result
        assert "areas" in result
        assert "models" in result
        assert isinstance(result["checkpoint_path"], str)


class TestBacktestPeriod:
    """Tests for BacktestPeriod dataclass."""

    @pytest.fixture
    def sample_period(self) -> BacktestPeriod:
        """Create sample backtest period."""
        return BacktestPeriod(
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 12, 31),
            test_start=datetime(2024, 1, 1),
            test_end=datetime(2024, 1, 7),
            period_id=0,
        )

    def test_str_representation(self, sample_period: BacktestPeriod) -> None:
        """Test string representation."""
        result = str(sample_period)
        assert "Period 0" in result
        assert "Train" in result
        assert "Test" in result

    def test_train_days(self, sample_period: BacktestPeriod) -> None:
        """Test train days calculation."""
        assert sample_period.train_days == 365

    def test_test_days(self, sample_period: BacktestPeriod) -> None:
        """Test test days calculation."""
        assert sample_period.test_days == 7

    def test_to_dict(self, sample_period: BacktestPeriod) -> None:
        """Test conversion to dictionary."""
        result = sample_period.to_dict()

        assert result["period_id"] == 0
        assert "train_start" in result
        assert "train_end" in result
        assert "test_start" in result
        assert "test_end" in result
        assert result["train_days"] == 365
        assert result["test_days"] == 7


class TestBacktestPeriodResult:
    """Tests for BacktestPeriodResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> BacktestPeriodResult:
        """Create sample period result."""
        period = BacktestPeriod(
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 12, 31),
            test_start=datetime(2024, 1, 1),
            test_end=datetime(2024, 1, 7),
            period_id=0,
        )
        return BacktestPeriodResult(
            period=period,
            training_result={"duration": 10.5, "status": "completed"},
            predictions={"duration": 5.2},
            evaluation={"mape": {"SECO_lgbm": 3.5}},
            performance_metrics={"training_duration": 10.5},
            execution_time=20.0,
        )

    def test_to_dict(self, sample_result: BacktestPeriodResult) -> None:
        """Test conversion to dictionary."""
        result = sample_result.to_dict()

        assert "period" in result
        assert result["training_duration"] == 10.5
        assert result["prediction_duration"] == 5.2
        assert "evaluation_metrics" in result
        assert result["execution_time"] == 20.0


class TestYearlyBacktestResult:
    """Tests for YearlyBacktestResult dataclass."""

    @pytest.fixture
    def sample_yearly_result(self) -> YearlyBacktestResult:
        """Create sample yearly result."""
        return YearlyBacktestResult(
            backtest_year=2024,
            total_periods=52,
            period_results=[],
            total_duration=25000.0,  # ~7 hours
            average_mape=0.05,
            summary_metrics={"mean_mape": 5.0, "std_mape": 1.0},
        )

    def test_passes_performance_targets_default(
        self, sample_yearly_result: YearlyBacktestResult
    ) -> None:
        """Test performance target check with defaults."""
        assert sample_yearly_result.passes_performance_targets()

    def test_passes_performance_targets_with_config(self) -> None:
        """Test performance target check with config."""
        config = BacktestConfig(
            target_duration_hours=6.0,  # 21600 seconds
            target_mape=0.10,
        )
        result = YearlyBacktestResult(
            backtest_year=2024,
            total_periods=52,
            period_results=[],
            total_duration=20000.0,  # Under 6 hours
            average_mape=0.05,  # Under 10%
            summary_metrics={},
            config=config,
        )
        assert result.passes_performance_targets()

    def test_fails_duration_target(self) -> None:
        """Test failure when duration target exceeded."""
        result = YearlyBacktestResult(
            backtest_year=2024,
            total_periods=52,
            period_results=[],
            total_duration=40000.0,  # >8 hours
            average_mape=0.05,
            summary_metrics={},
        )
        assert not result.passes_performance_targets()

    def test_fails_mape_target(self) -> None:
        """Test failure when MAPE target exceeded."""
        result = YearlyBacktestResult(
            backtest_year=2024,
            total_periods=52,
            period_results=[],
            total_duration=25000.0,
            average_mape=0.20,  # >15%
            summary_metrics={},
        )
        assert not result.passes_performance_targets()

    def test_to_dict(self, sample_yearly_result: YearlyBacktestResult) -> None:
        """Test conversion to dictionary."""
        result = sample_yearly_result.to_dict()

        assert result["backtest_year"] == 2024
        assert result["total_periods"] == 52
        assert result["total_duration"] == 25000.0
        assert result["average_mape"] == 0.05
        assert "summary_metrics" in result
        assert "passes_targets" in result


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    @pytest.fixture
    def monitor(self) -> PerformanceMonitor:
        """Create performance monitor instance."""
        return PerformanceMonitor()

    def test_start_end_operation(self, monitor: PerformanceMonitor) -> None:
        """Test start and end operation tracking."""
        monitor.start_operation("test_op")
        import time

        time.sleep(0.01)
        duration = monitor.end_operation("test_op")

        assert duration > 0
        assert "test_op_duration" in monitor.metrics

    def test_track_operation_context(self, monitor: PerformanceMonitor) -> None:
        """Test context manager for operation tracking."""
        with monitor.track_operation("test_context"):
            import time

            time.sleep(0.01)

        assert "test_context_duration" in monitor.metrics
        assert monitor.metrics["test_context_duration"] > 0

    def test_get_period_metrics(self, monitor: PerformanceMonitor) -> None:
        """Test getting period metrics."""
        with monitor.track_operation("op1"):
            pass
        with monitor.track_operation("op2"):
            pass

        metrics = monitor.get_period_metrics()

        assert "op1_duration" in metrics
        assert "op2_duration" in metrics

    def test_get_operation_statistics(self, monitor: PerformanceMonitor) -> None:
        """Test getting operation statistics."""
        for _ in range(5):
            with monitor.track_operation("repeated_op"):
                import time

                time.sleep(0.001)

        stats = monitor.get_operation_statistics("repeated_op")

        assert stats["count"] == 5
        assert stats["mean"] > 0
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_get_statistics_empty(self, monitor: PerformanceMonitor) -> None:
        """Test statistics for non-existent operation."""
        stats = monitor.get_operation_statistics("nonexistent")

        assert stats["count"] == 0
        assert stats["mean"] == 0.0

    def test_reset(self, monitor: PerformanceMonitor) -> None:
        """Test resetting monitor."""
        with monitor.track_operation("test"):
            pass

        monitor.reset()

        assert len(monitor.metrics) == 0
        assert len(monitor.operation_times) == 0


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir: Path) -> CheckpointManager:
        """Create checkpoint manager."""
        return CheckpointManager(temp_dir / "checkpoints")

    @pytest.fixture
    def sample_results(self) -> list[BacktestPeriodResult]:
        """Create sample results list."""
        period = BacktestPeriod(
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 12, 31),
            test_start=datetime(2024, 1, 1),
            test_end=datetime(2024, 1, 7),
            period_id=0,
        )
        return [
            BacktestPeriodResult(
                period=period,
                training_result={"duration": 10.0},
                predictions={"duration": 5.0},
                evaluation={"mape": {"SECO_lgbm": 3.5}},
                performance_metrics={},
                execution_time=20.0,
            )
        ]

    def test_save_checkpoint(
        self, manager: CheckpointManager, sample_results: list[BacktestPeriodResult]
    ) -> None:
        """Test saving checkpoint."""
        path = manager.save_checkpoint(last_period=0, results=sample_results)

        assert path.exists()
        assert manager.checkpoint_exists()

    def test_load_checkpoint(
        self, manager: CheckpointManager, sample_results: list[BacktestPeriodResult]
    ) -> None:
        """Test loading checkpoint."""
        manager.save_checkpoint(last_period=5, results=sample_results)
        checkpoint = manager.load_checkpoint()

        assert checkpoint is not None
        assert checkpoint["last_period"] == 5
        assert "timestamp" in checkpoint
        assert len(checkpoint["results"]) == 1

    def test_load_checkpoint_not_exists(self, manager: CheckpointManager) -> None:
        """Test loading when no checkpoint exists."""
        checkpoint = manager.load_checkpoint()
        assert checkpoint is None

    def test_clear_checkpoint(
        self, manager: CheckpointManager, sample_results: list[BacktestPeriodResult]
    ) -> None:
        """Test clearing checkpoint."""
        manager.save_checkpoint(last_period=0, results=sample_results)
        assert manager.checkpoint_exists()

        manager.clear_checkpoint()
        assert not manager.checkpoint_exists()

    def test_save_with_metadata(
        self, manager: CheckpointManager, sample_results: list[BacktestPeriodResult]
    ) -> None:
        """Test saving checkpoint with metadata."""
        metadata = {"config": "test", "version": 1}
        manager.save_checkpoint(
            last_period=0, results=sample_results, metadata=metadata
        )

        checkpoint = manager.load_checkpoint()
        assert checkpoint["metadata"] == metadata


class TestComprehensiveBacktester:
    """Tests for ComprehensiveBacktester class."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_dir: Path) -> BacktestConfig:
        """Create test configuration."""
        return BacktestConfig(
            backtest_year=2024,
            retraining_interval_days=7,
            checkpoint_path=temp_dir / "checkpoints",
            output_path=temp_dir / "output",
            areas=["SECO", "S"],
            models=["lgbm"],
            target_duration_hours=1.0,
            target_mape=0.20,
        )

    @pytest.fixture
    def backtester(self, config: BacktestConfig) -> ComprehensiveBacktester:
        """Create backtester instance."""
        return ComprehensiveBacktester(config)

    def test_init(self, backtester: ComprehensiveBacktester) -> None:
        """Test backtester initialization."""
        assert backtester.config is not None
        assert backtester.performance_monitor is not None
        assert backtester.checkpoint_manager is not None

    def test_generate_backtest_periods(
        self, backtester: ComprehensiveBacktester
    ) -> None:
        """Test backtest period generation."""
        periods = backtester._generate_backtest_periods(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            retraining_interval=7,
        )

        # 52 weeks in a year (approximately, may be 52 or 53 depending on dates)
        assert len(periods) >= 52
        assert len(periods) <= 53
        assert periods[0].period_id == 0
        assert periods[-1].period_id == len(periods) - 1

    def test_generate_periods_monthly(
        self, backtester: ComprehensiveBacktester
    ) -> None:
        """Test period generation with monthly interval."""
        periods = backtester._generate_backtest_periods(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            retraining_interval=30,
        )

        # ~12 periods for monthly retraining
        assert len(periods) >= 12
        assert len(periods) <= 13

    def test_generate_periods_short_range(
        self, backtester: ComprehensiveBacktester
    ) -> None:
        """Test period generation for short date range."""
        periods = backtester._generate_backtest_periods(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 14),
            retraining_interval=7,
        )

        assert len(periods) == 2

    @pytest.mark.asyncio
    async def test_execute_backtest_period(
        self, backtester: ComprehensiveBacktester
    ) -> None:
        """Test single period execution."""
        period = BacktestPeriod(
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 12, 31),
            test_start=datetime(2024, 1, 1),
            test_end=datetime(2024, 1, 7),
            period_id=0,
        )

        result = await backtester._execute_backtest_period(period)

        assert result.period == period
        assert "mape" in result.evaluation
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_execute_yearly_backtest_small(
        self, temp_dir: Path
    ) -> None:
        """Test yearly backtest with small config."""
        config = BacktestConfig(
            backtest_year=2024,
            retraining_interval_days=90,  # Quarterly for faster test
            checkpoint_path=temp_dir / "checkpoints",
            output_path=temp_dir / "output",
            areas=["SECO"],
            models=["lgbm"],
            target_duration_hours=1.0,
            target_mape=0.50,  # High threshold for test
        )
        backtester = ComprehensiveBacktester(config)

        result = await backtester.execute_yearly_backtest()

        assert result.backtest_year == 2024
        assert result.total_periods >= 4  # At least quarterly
        assert result.total_duration > 0

    @pytest.mark.asyncio
    async def test_checkpoint_resume(self, temp_dir: Path) -> None:
        """Test checkpoint and resume functionality."""
        config = BacktestConfig(
            backtest_year=2024,
            retraining_interval_days=90,
            checkpoint_path=temp_dir / "checkpoints",
            output_path=temp_dir / "output",
            areas=["SECO"],
            models=["lgbm"],
            target_duration_hours=1.0,
            target_mape=0.50,
        )

        # Run first backtest
        backtester1 = ComprehensiveBacktester(config)
        result1 = await backtester1.execute_yearly_backtest()

        # Checkpoint should be cleared after successful completion
        assert not backtester1.checkpoint_manager.checkpoint_exists()

        # Results should be saved
        result_file = config.output_path / "backtest_results" / "backtest_2024.json"
        assert result_file.exists()

    def test_aggregate_backtest_results_empty(
        self, backtester: ComprehensiveBacktester
    ) -> None:
        """Test aggregation with empty results."""
        result = backtester._aggregate_backtest_results(2024, [])

        assert result.backtest_year == 2024
        assert result.total_periods == 0
        assert result.total_duration == 0.0
        assert result.average_mape == 0.0

    def test_calculate_summary_metrics_empty(
        self, backtester: ComprehensiveBacktester
    ) -> None:
        """Test summary metrics with empty results."""
        metrics = backtester._calculate_summary_metrics([])

        assert metrics["mean_mape"] == 0.0
        assert metrics["mean_mae"] == 0.0
        assert metrics["max_training_time"] == 0.0

    def test_get_performance_statistics(
        self, backtester: ComprehensiveBacktester
    ) -> None:
        """Test getting performance statistics."""
        with backtester.performance_monitor.track_operation("training"):
            pass

        stats = backtester.get_performance_statistics()

        assert "training" in stats
        assert "prediction" in stats


class TestPerformanceTargetViolation:
    """Tests for PerformanceTargetViolation exception."""

    def test_exception_message(self) -> None:
        """Test exception message."""
        violations = ["Duration exceeded", "MAPE exceeded"]
        exc = PerformanceTargetViolation(violations)

        assert exc.violations == violations
        assert "Duration exceeded" in str(exc)
        assert "MAPE exceeded" in str(exc)


class TestBacktestExecutionError:
    """Tests for BacktestExecutionError exception."""

    def test_exception_message(self) -> None:
        """Test exception message."""
        exc = BacktestExecutionError("Failed at period 5")
        assert "period 5" in str(exc)


class TestEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_single_day_backtest(self, temp_dir: Path) -> None:
        """Test backtest with single day."""
        config = BacktestConfig(
            backtest_year=2024,
            retraining_interval_days=1,
            checkpoint_path=temp_dir / "checkpoints",
            output_path=temp_dir / "output",
            areas=["SECO"],
            models=["lgbm"],
        )
        backtester = ComprehensiveBacktester(config)

        periods = backtester._generate_backtest_periods(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1),
            retraining_interval=1,
        )

        assert len(periods) == 1

    def test_checkpoint_manager_directory_creation(self, temp_dir: Path) -> None:
        """Test checkpoint manager creates directory."""
        nested_path = temp_dir / "a" / "b" / "c" / "checkpoints"
        manager = CheckpointManager(nested_path)

        assert nested_path.exists()

    @pytest.mark.asyncio
    async def test_backtest_with_high_mape_target(self, temp_dir: Path) -> None:
        """Test backtest passes with high MAPE target."""
        config = BacktestConfig(
            backtest_year=2024,
            retraining_interval_days=180,  # Semi-annual
            checkpoint_path=temp_dir / "checkpoints",
            output_path=temp_dir / "output",
            areas=["SECO"],
            models=["lgbm"],
            target_duration_hours=10.0,
            target_mape=1.0,  # 100% MAPE allowed
        )
        backtester = ComprehensiveBacktester(config)

        result = await backtester.execute_yearly_backtest()

        assert result.passes_performance_targets()

    def test_reconstruct_results_from_checkpoint(self, temp_dir: Path) -> None:
        """Test reconstructing results from checkpoint data."""
        config = BacktestConfig(
            checkpoint_path=temp_dir / "checkpoints",
            output_path=temp_dir / "output",
        )
        backtester = ComprehensiveBacktester(config)

        checkpoint = {
            "last_period": 1,
            "results": [
                {
                    "period": {
                        "period_id": 0,
                        "train_start": "2023-01-01T00:00:00",
                        "train_end": "2023-12-31T00:00:00",
                        "test_start": "2024-01-01T00:00:00",
                        "test_end": "2024-01-07T00:00:00",
                    },
                    "training_duration": 10.0,
                    "prediction_duration": 5.0,
                    "evaluation_metrics": {"mape": {"SECO_lgbm": 3.5}},
                    "performance_metrics": {},
                    "execution_time": 20.0,
                }
            ],
        }

        periods = backtester._generate_backtest_periods(
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
            7,
        )

        results = backtester._reconstruct_results_from_checkpoint(checkpoint, periods)

        assert len(results) == 1
        assert results[0].period.period_id == 0
        assert results[0].execution_time == 20.0
