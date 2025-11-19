# PC-085-10A: Comprehensive Backtester Implementation

**Ticket ID:** PC-085-10A  
**Epic:** Epic-10A (Baseline Validation & Comprehensive Backtesting)  
**Story:** User Story 2 - Comprehensive 1-Year Backtesting  
**Assignee:** TBD  
**Status:** Not Started  
**Priority:** Critical  
**Estimated Effort:** 5 days  
**Sprint:** Week 27-28

---

## 📋 Description

Implement comprehensive backtesting framework to execute 1-year backtest (365 days, 2024) with walk-forward validation, weekly retraining (52 cycles), and performance monitoring. The system must validate all 5 forecasting models across 26 time series while meeting the 8-hour execution target.

---

## 🎯 Acceptance Criteria

- [ ] `ComprehensiveBacktester` class with async execution support
- [ ] Walk-forward validation with weekly retraining (52 cycles)
- [ ] Full 2024 backtesting execution (365 days)
- [ ] All 5 models tested across all 26 time series (130 combinations)
- [ ] Model combination and reconciliation integrated in backtest
- [ ] Performance monitoring tracks execution time and resource usage
- [ ] Backtest completes within 8-hour target
- [ ] Progress tracking with checkpointing for resume capability
- [ ] Detailed result aggregation and reporting
- [ ] Unit tests achieve >80% coverage
- [ ] Integration tests validate end-to-end backtesting workflow

---

## 🏗️ Technical Implementation

### **1. ComprehensiveBacktester Class**

**Location:** `src/validation/comprehensive_backtester.py`

```python
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import numpy as np
from pathlib import Path

@dataclass
class BacktestPeriod:
    """Represents a backtesting period."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    period_id: int
    
    def __str__(self) -> str:
        return (
            f"Period {self.period_id}: "
            f"Train [{self.train_start.date()}, {self.train_end.date()}], "
            f"Test [{self.test_start.date()}, {self.test_end.date()}]"
        )

@dataclass
class BacktestPeriodResult:
    """Results from a single backtest period."""
    period: BacktestPeriod
    training_result: Dict
    predictions: Dict
    evaluation: Dict
    performance_metrics: Dict
    execution_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'period': str(self.period),
            'training_duration': self.training_result.get('duration', 0),
            'prediction_duration': self.predictions.get('duration', 0),
            'evaluation_metrics': self.evaluation,
            'performance_metrics': self.performance_metrics,
            'execution_time': self.execution_time
        }

@dataclass
class YearlyBacktestResult:
    """Aggregated results from yearly backtesting."""
    backtest_year: int
    total_periods: int
    period_results: List[BacktestPeriodResult]
    total_duration: float
    average_mape: float
    summary_metrics: Dict[str, float]
    
    def passes_performance_targets(self) -> bool:
        """Check if performance targets are met."""
        return (
            self.total_duration <= 8 * 3600 and  # 8 hours
            self.average_mape <= 0.15  # 15%
        )

class ComprehensiveBacktester:
    """Executes comprehensive backtesting with walk-forward validation."""
    
    def __init__(self, config: 'SystemConfig'):
        self.config = config
        self.training_workflow = TrainingWorkflow(config)
        self.prediction_workflow = PredictionWorkflow(config)
        self.evaluator = MetricsCalculator(config)
        self.performance_monitor = PerformanceMonitor()
        self.checkpoint_manager = CheckpointManager(config.checkpoint_path)
        self.logger = logging.getLogger(__name__)
    
    async def execute_yearly_backtest(
        self,
        backtest_year: int = 2024,
        retraining_interval: int = 7,  # days
        resume_from_checkpoint: bool = True
    ) -> YearlyBacktestResult:
        """Execute comprehensive yearly backtesting with performance monitoring.
        
        Args:
            backtest_year: Year to backtest (default: 2024)
            retraining_interval: Days between retraining (default: 7)
            resume_from_checkpoint: Resume from last checkpoint if available
            
        Returns:
            YearlyBacktestResult with aggregated metrics
        """
        self.logger.info(f"Starting {backtest_year} yearly backtest")
        
        start_date = datetime(backtest_year, 1, 1)
        end_date = datetime(backtest_year, 12, 31)
        
        # Generate backtest periods
        backtest_periods = self._generate_backtest_periods(
            start_date, end_date, retraining_interval
        )
        
        self.logger.info(f"Generated {len(backtest_periods)} backtest periods")
        
        # Check for existing checkpoint
        start_period = 0
        completed_results = []
        
        if resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                start_period = checkpoint['last_period'] + 1
                completed_results = checkpoint['results']
                self.logger.info(f"Resuming from period {start_period}")
        
        # Execute backtesting with monitoring
        with self.performance_monitor.track_operation("yearly_backtest"):
            results = await self._execute_backtest_periods(
                backtest_periods[start_period:],
                start_period,
                completed_results
            )
        
        # Aggregate and analyze results
        yearly_result = self._aggregate_backtest_results(
            backtest_year, results
        )
        
        # Validate performance targets
        self._validate_performance_targets(yearly_result)
        
        # Save final results
        self._save_backtest_results(yearly_result)
        
        self.logger.info(
            f"Yearly backtest complete. Duration: {yearly_result.total_duration:.2f}s, "
            f"Average MAPE: {yearly_result.average_mape:.3f}"
        )
        
        return yearly_result
    
    async def _execute_backtest_periods(
        self,
        periods: List[BacktestPeriod],
        start_index: int,
        completed_results: List[BacktestPeriodResult]
    ) -> List[BacktestPeriodResult]:
        """Execute backtesting for multiple periods.
        
        Args:
            periods: List of backtest periods to execute
            start_index: Starting period index
            completed_results: Previously completed results
            
        Returns:
            List of BacktestPeriodResult
        """
        results = completed_results.copy()
        total_periods = len(periods) + start_index
        
        for i, period in enumerate(periods, start=start_index):
            self.logger.info(
                f"Processing backtest period {i+1}/{total_periods}: {period}"
            )
            
            try:
                # Execute period backtesting
                period_result = await self._execute_backtest_period(period)
                results.append(period_result)
                
                # Monitor performance and adjust if needed
                self._monitor_performance_and_adjust(period_result)
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    last_period=i,
                    results=results
                )
                
            except Exception as e:
                self.logger.error(f"Error in period {i+1}: {e}")
                raise BacktestExecutionError(f"Failed at period {i+1}") from e
        
        return results
    
    async def _execute_backtest_period(
        self,
        period: BacktestPeriod
    ) -> BacktestPeriodResult:
        """Execute backtesting for a single period.
        
        Args:
            period: BacktestPeriod to execute
            
        Returns:
            BacktestPeriodResult with metrics
        """
        period_start_time = datetime.now()
        
        # 1. Train models on training window
        self.logger.info(f"Training models for period {period.period_id}")
        training_result = await self.training_workflow.execute_training(
            areas=self.config.data.areas,
            models=self.config.models.types,
            start_date=period.train_start,
            end_date=period.train_end
        )
        
        # 2. Generate predictions for test window
        self.logger.info(f"Generating predictions for period {period.period_id}")
        predictions = await self.prediction_workflow.execute_batch_prediction(
            prediction_date=period.test_start,
            horizons=list(range(9))  # D+0 to D+8
        )
        
        # 3. Evaluate predictions
        self.logger.info(f"Evaluating predictions for period {period.period_id}")
        actuals = self._load_actuals(period.test_start, period.test_end)
        evaluation_result = self.evaluator.calculate_metrics(
            predictions=predictions,
            actuals=actuals,
            metrics=['mape', 'mae', 'rmse', 'percentiles']
        )
        
        # 4. Collect performance metrics
        performance_metrics = self.performance_monitor.get_period_metrics()
        
        period_duration = (datetime.now() - period_start_time).total_seconds()
        
        return BacktestPeriodResult(
            period=period,
            training_result=training_result,
            predictions=predictions,
            evaluation=evaluation_result,
            performance_metrics=performance_metrics,
            execution_time=period_duration
        )
    
    def _generate_backtest_periods(
        self,
        start_date: datetime,
        end_date: datetime,
        retraining_interval: int
    ) -> List[BacktestPeriod]:
        """Generate walk-forward validation periods.
        
        Args:
            start_date: Start date of backtest
            end_date: End date of backtest
            retraining_interval: Days between retraining
            
        Returns:
            List of BacktestPeriod
        """
        periods = []
        period_id = 0
        
        # Initial training window (e.g., 1 year of historical data)
        initial_train_days = 365
        train_start = start_date - timedelta(days=initial_train_days)
        train_end = start_date - timedelta(days=1)
        
        current_date = start_date
        
        while current_date <= end_date:
            test_end = min(
                current_date + timedelta(days=retraining_interval - 1),
                end_date
            )
            
            period = BacktestPeriod(
                train_start=train_start,
                train_end=train_end,
                test_start=current_date,
                test_end=test_end,
                period_id=period_id
            )
            
            periods.append(period)
            
            # Update for next period
            period_id += 1
            train_start = train_start + timedelta(days=retraining_interval)
            train_end = test_end
            current_date = test_end + timedelta(days=1)
        
        return periods
    
    def _aggregate_backtest_results(
        self,
        backtest_year: int,
        period_results: List[BacktestPeriodResult]
    ) -> YearlyBacktestResult:
        """Aggregate results from all backtest periods.
        
        Args:
            backtest_year: Year of backtest
            period_results: Results from all periods
            
        Returns:
            YearlyBacktestResult with aggregated metrics
        """
        total_duration = sum(r.execution_time for r in period_results)
        
        # Calculate average MAPE across all periods
        all_mapes = []
        for result in period_results:
            if 'mape' in result.evaluation:
                all_mapes.extend(result.evaluation['mape'].values())
        
        average_mape = np.mean(all_mapes) if all_mapes else 0.0
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(period_results)
        
        return YearlyBacktestResult(
            backtest_year=backtest_year,
            total_periods=len(period_results),
            period_results=period_results,
            total_duration=total_duration,
            average_mape=average_mape,
            summary_metrics=summary_metrics
        )
    
    def _calculate_summary_metrics(
        self,
        period_results: List[BacktestPeriodResult]
    ) -> Dict[str, float]:
        """Calculate summary metrics from period results.
        
        Args:
            period_results: Results from all periods
            
        Returns:
            Dictionary of summary metrics
        """
        all_metrics = {
            'mape': [],
            'mae': [],
            'rmse': [],
            'training_time': [],
            'prediction_time': []
        }
        
        for result in period_results:
            eval_metrics = result.evaluation
            all_metrics['mape'].extend(eval_metrics.get('mape', {}).values())
            all_metrics['mae'].extend(eval_metrics.get('mae', {}).values())
            all_metrics['rmse'].extend(eval_metrics.get('rmse', {}).values())
            all_metrics['training_time'].append(
                result.training_result.get('duration', 0)
            )
            all_metrics['prediction_time'].append(
                result.predictions.get('duration', 0)
            )
        
        return {
            'mean_mape': np.mean(all_metrics['mape']),
            'std_mape': np.std(all_metrics['mape']),
            'mean_mae': np.mean(all_metrics['mae']),
            'mean_rmse': np.mean(all_metrics['rmse']),
            'mean_training_time': np.mean(all_metrics['training_time']),
            'mean_prediction_time': np.mean(all_metrics['prediction_time']),
            'max_training_time': np.max(all_metrics['training_time']),
            'max_prediction_time': np.max(all_metrics['prediction_time'])
        }
    
    def _validate_performance_targets(
        self,
        yearly_result: YearlyBacktestResult
    ):
        """Validate that performance targets are met.
        
        Args:
            yearly_result: Aggregated yearly results
            
        Raises:
            PerformanceTargetViolation: If targets not met
        """
        targets = {
            'total_duration': 8 * 3600,  # 8 hours in seconds
            'average_mape': 0.15,  # 15% MAPE threshold
            'max_training_time': 30 * 60,  # 30 minutes per model-area
            'prediction_latency_p95': 5 * 60  # 5 minutes P95
        }
        
        violations = []
        
        if yearly_result.total_duration > targets['total_duration']:
            violations.append(
                f"Total duration {yearly_result.total_duration:.0f}s "
                f"exceeds target {targets['total_duration']:.0f}s"
            )
        
        if yearly_result.average_mape > targets['average_mape']:
            violations.append(
                f"Average MAPE {yearly_result.average_mape:.3f} "
                f"exceeds target {targets['average_mape']}"
            )
        
        max_training = yearly_result.summary_metrics.get('max_training_time', 0)
        if max_training > targets['max_training_time']:
            violations.append(
                f"Max training time {max_training:.0f}s "
                f"exceeds target {targets['max_training_time']:.0f}s"
            )
        
        if violations:
            self.logger.warning(f"Performance target violations: {violations}")
            raise PerformanceTargetViolation(violations)
        
        self.logger.info("All performance targets met")
    
    def _monitor_performance_and_adjust(
        self,
        period_result: BacktestPeriodResult
    ):
        """Monitor performance and make adjustments if needed.
        
        Args:
            period_result: Result from current period
        """
        execution_time = period_result.execution_time
        
        # Check if period execution is taking too long
        expected_time_per_period = (8 * 3600) / 52  # ~553 seconds
        
        if execution_time > expected_time_per_period * 1.5:
            self.logger.warning(
                f"Period {period_result.period.period_id} took "
                f"{execution_time:.0f}s (expected ~{expected_time_per_period:.0f}s)"
            )
            
            # Could adjust parallelism, reduce batch sizes, etc.
            self._adjust_performance_settings()
    
    def _adjust_performance_settings(self):
        """Adjust performance settings to meet targets."""
        # Placeholder for dynamic performance adjustments
        self.logger.info("Adjusting performance settings...")
    
    def _load_actuals(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Load actual load data for period.
        
        Args:
            start_date: Start of period
            end_date: End of period
            
        Returns:
            Dictionary of actual values by area
        """
        # Load from actuals database
        actuals_loader = ActualsLoader(self.config.data.actuals_path)
        return actuals_loader.load(start_date, end_date)
    
    def _save_backtest_results(
        self,
        yearly_result: YearlyBacktestResult
    ):
        """Save backtest results to disk.
        
        Args:
            yearly_result: Aggregated yearly results
        """
        output_path = Path(self.config.output_path) / "backtest_results"
        output_path.mkdir(parents=True, exist_ok=True)
        
        result_file = output_path / f"backtest_{yearly_result.backtest_year}.json"
        
        import json
        with open(result_file, 'w') as f:
            json.dump({
                'backtest_year': yearly_result.backtest_year,
                'total_periods': yearly_result.total_periods,
                'total_duration': yearly_result.total_duration,
                'average_mape': yearly_result.average_mape,
                'summary_metrics': yearly_result.summary_metrics,
                'period_results': [r.to_dict() for r in yearly_result.period_results]
            }, f, indent=2)
        
        self.logger.info(f"Backtest results saved to {result_file}")


class CheckpointManager:
    """Manages checkpointing for backtest resumption."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        last_period: int,
        results: List[BacktestPeriodResult]
    ):
        """Save checkpoint to disk."""
        checkpoint_file = self.checkpoint_path / "backtest_checkpoint.json"
        
        import json
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'last_period': last_period,
                'timestamp': datetime.now().isoformat(),
                'results': [r.to_dict() for r in results]
            }, f)
        
        self.logger.debug(f"Checkpoint saved at period {last_period}")
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint from disk."""
        checkpoint_file = self.checkpoint_path / "backtest_checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
        
        import json
        with open(checkpoint_file, 'r') as f:
            return json.load(f)


class PerformanceMonitor:
    """Monitors performance during backtest execution."""
    
    def __init__(self):
        self.metrics = {}
        self.current_operation = None
        self.logger = logging.getLogger(__name__)
    
    def track_operation(self, operation_name: str):
        """Context manager for tracking operation performance."""
        return OperationTracker(self, operation_name)
    
    def get_period_metrics(self) -> Dict[str, float]:
        """Get metrics for current period."""
        return self.metrics.copy()


class OperationTracker:
    """Context manager for operation tracking."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        self.monitor.metrics[f"{self.operation_name}_duration"] = duration


class BacktestExecutionError(Exception):
    """Error during backtest execution."""
    pass


class PerformanceTargetViolation(Exception):
    """Performance targets not met."""
    pass
```

---

## 🧪 Testing Requirements

### **Unit Tests**

**Location:** `tests/unit/validation/test_comprehensive_backtester.py`

```python
import pytest
from datetime import datetime
from src.validation.comprehensive_backtester import (
    ComprehensiveBacktester, BacktestPeriod
)

class TestComprehensiveBacktester:
    @pytest.fixture
    def backtester(self, mock_config):
        return ComprehensiveBacktester(mock_config)
    
    def test_generate_backtest_periods(self, backtester):
        """Test backtest period generation."""
        periods = backtester._generate_backtest_periods(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            retraining_interval=7
        )
        
        assert len(periods) == 52  # Weekly retraining
        assert periods[0].period_id == 0
        assert periods[-1].period_id == 51
    
    @pytest.mark.asyncio
    async def test_execute_backtest_period(self, backtester):
        """Test single period execution."""
        period = BacktestPeriod(
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 12, 31),
            test_start=datetime(2024, 1, 1),
            test_end=datetime(2024, 1, 7),
            period_id=0
        )
        
        result = await backtester._execute_backtest_period(period)
        
        assert result.period == period
        assert 'mape' in result.evaluation
        assert result.execution_time > 0
    
    def test_aggregate_backtest_results(self, backtester, mock_period_results):
        """Test result aggregation."""
        yearly_result = backtester._aggregate_backtest_results(
            backtest_year=2024,
            period_results=mock_period_results
        )
        
        assert yearly_result.backtest_year == 2024
        assert yearly_result.total_periods == len(mock_period_results)
        assert yearly_result.average_mape > 0
```

### **Integration Tests**

**Location:** `tests/integration/validation/test_backtesting_integration.py`

```python
import pytest
from datetime import datetime
from src.validation.comprehensive_backtester import ComprehensiveBacktester

class TestBacktestingIntegration:
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_year_backtest(self, system_config):
        """Test complete yearly backtest execution."""
        backtester = ComprehensiveBacktester(system_config)
        
        result = await backtester.execute_yearly_backtest(
            backtest_year=2024,
            retraining_interval=7
        )
        
        assert result.backtest_year == 2024
        assert result.total_periods == 52
        assert result.total_duration < 8 * 3600  # <8 hours
        assert result.average_mape < 0.15  # <15%
        assert result.passes_performance_targets()
```

---

## 📊 Success Metrics

- [ ] 1-year backtest execution: 100% completion within 8-hour target
- [ ] All 52 periods: Successfully executed with walk-forward validation
- [ ] Model coverage: All 5 models × 26 areas = 130 combinations validated
- [ ] Performance targets: Training <30min, prediction <15min per period
- [ ] Checkpoint reliability: 100% successful resume from any period

---

## 🔗 Dependencies

### **Upstream**
- PC-084-10A (Baseline Validator): Reference baseline for comparison
- Epic-08A (Training Workflow): Model training execution
- Epic-08B (Prediction Workflow): Batch prediction generation
- Epic-07 (Evaluation): Metrics calculation

### **Downstream**
- PC-086-10A (Performance Analyzer): Uses backtest results for analysis
- Epic-10B (Quality): Backtest results for quality validation

---

## 📚 Documentation

- [ ] Backtesting methodology documented in `/docs/validation/backtesting.md`
- [ ] Walk-forward validation strategy in `/docs/validation/walk_forward.md`
- [ ] Performance optimization guide in `/docs/validation/performance_tuning.md`
- [ ] Checkpoint and resume procedure in `/docs/validation/checkpoint_management.md`
- [ ] API documentation with complete docstrings

---

## ✅ Definition of Done

- [ ] All code implemented and peer-reviewed
- [ ] Unit tests passing with >80% coverage
- [ ] Integration tests validate full backtest workflow
- [ ] 1-year backtest completes in <8 hours
- [ ] All 52 periods execute successfully with walk-forward validation
- [ ] Checkpoint/resume functionality tested and working
- [ ] Performance monitoring captures all key metrics
- [ ] Documentation complete and reviewed
- [ ] Code merged to main branch

---

**Notes:**
- Consider parallel execution across areas to meet 8-hour target
- Implement progress bar/dashboard for long-running backtests
- Monitor memory usage during execution
- Ensure proper cleanup of temporary files between periods
- Test checkpoint/resume thoroughly to handle failures gracefully
