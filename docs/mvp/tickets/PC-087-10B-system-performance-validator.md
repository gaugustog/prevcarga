# PC-087-10B: System Performance Validator Implementation

**Ticket ID:** PC-087-10B  
**Epic:** Epic-10B (Quality Assurance & Acceptance Validation)  
**Story:** User Story 1 - System Performance and Quality Validation  
**Assignee:** TBD  
**Status:** Not Started  
**Priority:** Critical  
**Estimated Effort:** 5 days  
**Sprint:** Week 29

---

## 📋 Description

Implement comprehensive system performance validation framework to benchmark training and prediction performance, validate system scalability, test concurrent operations, and ensure the system meets all operational requirements. The validator must verify performance against defined thresholds and validate system robustness under various load conditions.

---

## 🎯 Acceptance Criteria

- [ ] `SystemPerformanceValidator` class with comprehensive benchmarking capabilities
- [ ] Training performance benchmarking (model training time, parallel efficiency, memory usage)
- [ ] Prediction performance benchmarking (batch, intraday, latency P95)
- [ ] System scalability validation with concurrent user simulation
- [ ] Resource consumption monitoring (CPU, memory, I/O)
- [ ] Performance threshold validation against targets
- [ ] Integration testing across all system components
- [ ] Performance degradation detection and alerting
- [ ] Benchmark result persistence and trend tracking
- [ ] Comprehensive performance reporting with visualizations
- [ ] Unit tests achieve >80% coverage
- [ ] Integration tests validate end-to-end performance workflows

---

## 🏗️ Technical Implementation

### **1. SystemPerformanceValidator Class**

**Location:** `src/validation/system_performance_validator.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import psutil
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class PerformanceBenchmarkResult:
    """Results from performance benchmarking."""
    results: Dict[str, Dict]
    timestamp: float
    system_info: Dict[str, str]
    
    def passes_all_thresholds(self) -> bool:
        """Check if all benchmarks pass their thresholds."""
        return all(
            r['passes_threshold'] 
            for r in self.results.values()
        )
    
    def get_failing_benchmarks(self) -> List[str]:
        """Get list of benchmarks that failed thresholds."""
        return [
            name for name, result in self.results.items()
            if not result['passes_threshold']
        ]

@dataclass
class ValidationConfig:
    """Configuration for performance validation."""
    training_time_threshold: int = 30 * 60  # 30 minutes
    parallel_efficiency_threshold: float = 0.7  # 70%
    memory_threshold: int = 16 * 1024 * 1024 * 1024  # 16GB
    batch_prediction_threshold: int = 15 * 60  # 15 minutes
    intraday_prediction_threshold: int = 5 * 60  # 5 minutes
    prediction_latency_p95_threshold: int = 3 * 60  # 3 minutes
    concurrent_users: int = 10
    throughput_degradation_threshold: float = 0.2  # 20%
    error_rate_threshold: float = 0.01  # 1%

class SystemPerformanceValidator:
    """Validates system performance against operational requirements."""
    
    def __init__(self, config: ValidationConfig):
        """Initialize performance validator.
        
        Args:
            config: Validation configuration with thresholds
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PerformanceMonitor()
        self.load_tester = SystemLoadTester()
    
    def validate_performance_benchmarks(self) -> PerformanceBenchmarkResult:
        """Validate system performance against all benchmarks.
        
        Returns:
            PerformanceBenchmarkResult with all benchmark results
        """
        self.logger.info("Starting performance benchmark validation")
        benchmark_results = {}
        
        # Training performance
        self.logger.info("Benchmarking training performance")
        training_benchmark = self._benchmark_training_performance()
        benchmark_results['training'] = self._validate_against_targets(
            training_benchmark,
            targets={
                'model_training_time': self.config.training_time_threshold,
                'parallel_efficiency': self.config.parallel_efficiency_threshold,
                'memory_usage': self.config.memory_threshold
            }
        )
        
        # Prediction performance
        self.logger.info("Benchmarking prediction performance")
        prediction_benchmark = self._benchmark_prediction_performance()
        benchmark_results['prediction'] = self._validate_against_targets(
            prediction_benchmark,
            targets={
                'batch_prediction_time': self.config.batch_prediction_threshold,
                'intraday_prediction_time': self.config.intraday_prediction_threshold,
                'prediction_latency_p95': self.config.prediction_latency_p95_threshold
            }
        )
        
        # System scalability
        self.logger.info("Benchmarking system scalability")
        scalability_benchmark = self._benchmark_scalability()
        benchmark_results['scalability'] = self._validate_against_targets(
            scalability_benchmark,
            targets={
                'concurrent_users': self.config.concurrent_users,
                'throughput_degradation': self.config.throughput_degradation_threshold,
                'error_rate': self.config.error_rate_threshold
            }
        )
        
        return PerformanceBenchmarkResult(
            results=benchmark_results,
            timestamp=time.time(),
            system_info=self._get_system_info()
        )
    
    def _benchmark_training_performance(self) -> Dict:
        """Benchmark model training performance."""
        from prevcarga.workflows.training import TrainingWorkflow
        
        results = {
            'training_times': [],
            'memory_usage': [],
            'parallel_efficiency': []
        }
        
        workflow = TrainingWorkflow()
        
        # Test training for each model type across sample areas
        test_models = ['lgbm', 'rf', 'regdin_svm']
        test_areas = ['01', '02', '03']
        
        for model_type in test_models:
            for area in test_areas:
                with self.performance_monitor.track_execution() as monitor:
                    workflow.train_model(
                        model_type=model_type,
                        area=area,
                        start_date='2024-01-01',
                        end_date='2024-12-31'
                    )
                
                results['training_times'].append(monitor.elapsed_time)
                results['memory_usage'].append(monitor.peak_memory)
        
        # Calculate parallel efficiency
        sequential_time = sum(results['training_times'])
        with self.performance_monitor.track_execution() as monitor:
            workflow.train_batch(
                model_types=test_models,
                areas=test_areas,
                start_date='2024-01-01',
                end_date='2024-12-31',
                parallel=True,
                n_workers=4
            )
        parallel_time = monitor.elapsed_time
        
        results['parallel_efficiency'].append(
            sequential_time / (parallel_time * 4)
        )
        
        return {
            'avg_training_time': np.mean(results['training_times']),
            'max_training_time': np.max(results['training_times']),
            'peak_memory': np.max(results['memory_usage']),
            'parallel_efficiency': np.mean(results['parallel_efficiency'])
        }
    
    def _benchmark_prediction_performance(self) -> Dict:
        """Benchmark prediction performance."""
        from prevcarga.workflows.prediction import PredictionWorkflow
        
        workflow = PredictionWorkflow()
        results = {
            'batch_times': [],
            'intraday_times': [],
            'latencies': []
        }
        
        # Batch prediction benchmark
        with self.performance_monitor.track_execution() as monitor:
            workflow.predict_batch(
                model_type='all',
                areas=['01', '02', '03', '04', '05'],
                prediction_date='2024-12-01'
            )
        results['batch_times'].append(monitor.elapsed_time)
        
        # Intraday prediction benchmark
        for _ in range(10):
            with self.performance_monitor.track_execution() as monitor:
                workflow.predict_intraday(
                    areas=['01', '02'],
                    current_datetime='2024-12-01 14:00:00'
                )
            results['intraday_times'].append(monitor.elapsed_time)
        
        # Latency benchmark (single predictions)
        for _ in range(100):
            with self.performance_monitor.track_execution() as monitor:
                workflow.predict_single(
                    model_type='lgbm',
                    area='01',
                    prediction_datetime='2024-12-01 15:00:00'
                )
            results['latencies'].append(monitor.elapsed_time)
        
        return {
            'avg_batch_time': np.mean(results['batch_times']),
            'avg_intraday_time': np.mean(results['intraday_times']),
            'latency_p50': np.percentile(results['latencies'], 50),
            'latency_p95': np.percentile(results['latencies'], 95),
            'latency_p99': np.percentile(results['latencies'], 99)
        }
    
    def _benchmark_scalability(self) -> Dict:
        """Benchmark system scalability with concurrent operations."""
        baseline_throughput = self.load_tester.measure_baseline_throughput()
        
        load_test_result = self.load_tester.execute_concurrent_load_test(
            concurrent_users=self.config.concurrent_users,
            test_duration=60,
            scenarios=['training', 'prediction', 'evaluation']
        )
        
        throughput_degradation = (
            (baseline_throughput - load_test_result.throughput) / 
            baseline_throughput
        )
        
        return {
            'baseline_throughput': baseline_throughput,
            'concurrent_throughput': load_test_result.throughput,
            'throughput_degradation': throughput_degradation,
            'error_rate': load_test_result.error_rate,
            'avg_latency': load_test_result.average_latency,
            'p95_latency': load_test_result.p95_latency
        }
    
    def _validate_against_targets(
        self,
        benchmark: Dict,
        targets: Dict
    ) -> Dict:
        """Validate benchmark results against target thresholds."""
        validation = {
            'benchmark_values': benchmark,
            'targets': targets,
            'validations': {},
            'passes_threshold': True
        }
        
        for metric, target in targets.items():
            actual = benchmark.get(metric)
            
            # Determine if metric should be <= or >= target
            if metric in ['training_time', 'prediction_time', 'latency', 
                         'memory_usage', 'throughput_degradation', 'error_rate']:
                passes = actual <= target
                comparison = '<='
            else:
                passes = actual >= target
                comparison = '>='
            
            validation['validations'][metric] = {
                'actual': actual,
                'target': target,
                'passes': passes,
                'comparison': comparison,
                'deviation_percent': abs((actual - target) / target * 100)
            }
            
            if not passes:
                validation['passes_threshold'] = False
                self.logger.warning(
                    f"Metric {metric} failed: {actual} {comparison} {target}"
                )
        
        return validation
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for benchmark context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current,
            'total_memory': psutil.virtual_memory().total,
            'platform': psutil.LINUX or psutil.WINDOWS or 'unknown'
        }

class PerformanceMonitor:
    """Monitor performance metrics during execution."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.process = psutil.Process()
    
    def track_execution(self):
        """Context manager for tracking execution performance."""
        return self
    
    def __enter__(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.peak_memory = self.process.memory_info().rss
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop performance monitoring."""
        self.end_time = time.time()
        current_memory = self.process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

class SystemLoadTester:
    """Execute load testing scenarios."""
    
    def measure_baseline_throughput(self) -> float:
        """Measure baseline throughput with single user."""
        from prevcarga.workflows.prediction import PredictionWorkflow
        
        workflow = PredictionWorkflow()
        operations = 0
        start_time = time.time()
        duration = 60  # 1 minute baseline
        
        while time.time() - start_time < duration:
            workflow.predict_single(
                model_type='lgbm',
                area='01',
                prediction_datetime='2024-12-01 15:00:00'
            )
            operations += 1
        
        return operations / duration
    
    def execute_concurrent_load_test(
        self,
        concurrent_users: int,
        test_duration: int,
        scenarios: List[str]
    ) -> 'LoadTestResult':
        """Execute concurrent load testing with multiple user scenarios.
        
        Args:
            concurrent_users: Number of concurrent users to simulate
            test_duration: Duration of test in seconds
            scenarios: List of scenarios to test
            
        Returns:
            LoadTestResult with test results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            for user_id in range(concurrent_users):
                for scenario in scenarios:
                    future = executor.submit(
                        self._execute_user_scenario, user_id, scenario
                    )
                    futures.append(future)
            
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(ScenarioResult(
                        success=False,
                        latency=0.0,
                        error=str(e)
                    ))
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        return LoadTestResult(
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            average_latency=np.mean([r.latency for r in successful]) if successful else 0,
            p95_latency=np.percentile([r.latency for r in successful], 95) if successful else 0,
            throughput=len(successful) / test_duration,
            error_rate=len(failed) / len(results)
        )
    
    def _execute_user_scenario(
        self,
        user_id: int,
        scenario: str
    ) -> 'ScenarioResult':
        """Execute a single user scenario."""
        start_time = time.time()
        
        try:
            if scenario == 'training':
                from prevcarga.workflows.training import TrainingWorkflow
                workflow = TrainingWorkflow()
                workflow.train_model(
                    model_type='lgbm',
                    area='01',
                    start_date='2024-01-01',
                    end_date='2024-12-31'
                )
            elif scenario == 'prediction':
                from prevcarga.workflows.prediction import PredictionWorkflow
                workflow = PredictionWorkflow()
                workflow.predict_single(
                    model_type='lgbm',
                    area='01',
                    prediction_datetime='2024-12-01 15:00:00'
                )
            elif scenario == 'evaluation':
                from prevcarga.workflows.evaluation import EvaluationWorkflow
                workflow = EvaluationWorkflow()
                workflow.evaluate_model(
                    model_type='lgbm',
                    area='01',
                    start_date='2024-01-01',
                    end_date='2024-12-31'
                )
            
            latency = time.time() - start_time
            return ScenarioResult(success=True, latency=latency)
            
        except Exception as e:
            latency = time.time() - start_time
            return ScenarioResult(success=False, latency=latency, error=str(e))

@dataclass
class LoadTestResult:
    """Results from load testing."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency: float
    p95_latency: float
    throughput: float
    error_rate: float

@dataclass
class ScenarioResult:
    """Result from executing a single scenario."""
    success: bool
    latency: float
    error: Optional[str] = None
```

---

## 🧪 Testing Requirements

### **Unit Tests**

**Location:** `tests/validation/test_system_performance_validator.py`

```python
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from prevcarga.validation.system_performance_validator import (
    SystemPerformanceValidator,
    ValidationConfig,
    PerformanceMonitor,
    SystemLoadTester
)

class TestSystemPerformanceValidator:
    """Test suite for SystemPerformanceValidator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ValidationConfig(
            training_time_threshold=1800,
            memory_threshold=16 * 1024 * 1024 * 1024
        )
    
    @pytest.fixture
    def validator(self, config):
        """Create validator instance."""
        return SystemPerformanceValidator(config)
    
    def test_validate_performance_benchmarks(self, validator):
        """Test complete performance benchmark validation."""
        with patch.object(validator, '_benchmark_training_performance') as mock_train:
            with patch.object(validator, '_benchmark_prediction_performance') as mock_pred:
                with patch.object(validator, '_benchmark_scalability') as mock_scale:
                    mock_train.return_value = {
                        'avg_training_time': 1500,
                        'parallel_efficiency': 0.75,
                        'peak_memory': 14 * 1024 * 1024 * 1024
                    }
                    mock_pred.return_value = {
                        'avg_batch_time': 800,
                        'latency_p95': 150
                    }
                    mock_scale.return_value = {
                        'throughput_degradation': 0.15,
                        'error_rate': 0.005
                    }
                    
                    result = validator.validate_performance_benchmarks()
                    
                    assert 'training' in result.results
                    assert 'prediction' in result.results
                    assert 'scalability' in result.results
    
    def test_benchmark_training_performance(self, validator):
        """Test training performance benchmarking."""
        with patch('prevcarga.workflows.training.TrainingWorkflow') as mock_workflow:
            mock_instance = Mock()
            mock_workflow.return_value = mock_instance
            
            result = validator._benchmark_training_performance()
            
            assert 'avg_training_time' in result
            assert 'parallel_efficiency' in result
            assert 'peak_memory' in result
    
    def test_validate_against_targets_pass(self, validator):
        """Test validation when all metrics pass thresholds."""
        benchmark = {
            'model_training_time': 1500,
            'parallel_efficiency': 0.75
        }
        targets = {
            'model_training_time': 1800,
            'parallel_efficiency': 0.7
        }
        
        result = validator._validate_against_targets(benchmark, targets)
        
        assert result['passes_threshold'] is True
        assert all(v['passes'] for v in result['validations'].values())
    
    def test_validate_against_targets_fail(self, validator):
        """Test validation when metrics fail thresholds."""
        benchmark = {
            'model_training_time': 2000,
            'parallel_efficiency': 0.6
        }
        targets = {
            'model_training_time': 1800,
            'parallel_efficiency': 0.7
        }
        
        result = validator._validate_against_targets(benchmark, targets)
        
        assert result['passes_threshold'] is False

class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor."""
    
    def test_track_execution(self):
        """Test execution tracking."""
        monitor = PerformanceMonitor()
        
        with monitor.track_execution():
            time.sleep(0.1)
        
        assert monitor.elapsed_time >= 0.1
        assert monitor.peak_memory > 0

class TestSystemLoadTester:
    """Test suite for SystemLoadTester."""
    
    @pytest.fixture
    def load_tester(self):
        """Create load tester instance."""
        return SystemLoadTester()
    
    def test_execute_concurrent_load_test(self, load_tester):
        """Test concurrent load testing."""
        with patch.object(load_tester, '_execute_user_scenario') as mock_scenario:
            from prevcarga.validation.system_performance_validator import ScenarioResult
            mock_scenario.return_value = ScenarioResult(
                success=True,
                latency=0.5
            )
            
            result = load_tester.execute_concurrent_load_test(
                concurrent_users=2,
                test_duration=1,
                scenarios=['prediction']
            )
            
            assert result.total_requests > 0
            assert result.successful_requests <= result.total_requests
            assert result.error_rate <= 1.0
```

---

## 📊 Success Metrics

- [ ] Training benchmarks complete in <30 minutes per model-area
- [ ] Prediction latency P95 <3 minutes
- [ ] System handles 10 concurrent users with <20% degradation
- [ ] Memory usage remains <16GB under load
- [ ] All performance thresholds validated and documented

---

## 🔗 Dependencies

**Requires:**
- Epic-08A (TrainingWorkflow, PredictionWorkflow, EvaluationWorkflow)
- Epic-03 (Model implementations for benchmarking)
- Epic-01 (Data pipeline for test data)

**Blocks:**
- PC-088-10B (Robustness Validator - needs performance baselines)
- PC-090-10B (Code Quality Validator - needs performance regression tests)

---

## 📚 Documentation

- [ ] Performance validation framework architecture documented
- [ ] Benchmark configuration and thresholds documented
- [ ] Performance monitoring usage guide created
- [ ] Load testing scenarios and results documented
- [ ] API documentation with examples

---

## 🔄 Implementation Steps

1. **Day 1: Core Performance Validator**
   - Implement `SystemPerformanceValidator` class
   - Implement `PerformanceMonitor` context manager
   - Create validation configuration system

2. **Day 2: Training and Prediction Benchmarks**
   - Implement training performance benchmarking
   - Implement prediction performance benchmarking
   - Create threshold validation logic

3. **Day 3: Load Testing Framework**
   - Implement `SystemLoadTester` class
   - Create concurrent user simulation
   - Implement scenario execution

4. **Day 4: Scalability and Integration**
   - Implement scalability benchmarking
   - Create integration test scenarios
   - Implement result persistence

5. **Day 5: Testing and Documentation**
   - Complete unit test suite
   - Complete integration tests
   - Write documentation and usage guides

---

**Estimated Completion:** Week 29, Day 5  
**Review Required:** System Architecture Team, QA Team
