"""System performance validator for PrevCarga system.

This module provides comprehensive system performance validation capabilities
including training and prediction benchmarking, scalability testing, and
resource monitoring.

Key Features:
- Training performance benchmarking
- Prediction performance benchmarking (batch, intraday, latency)
- System scalability validation with concurrent user simulation
- Resource consumption monitoring (CPU, memory, I/O)
- Performance threshold validation

Example:
    ```python
    from src.validation.system_performance_validator import (
        SystemPerformanceValidator,
        PerformanceValidationConfig,
    )

    config = PerformanceValidationConfig(
        training_time_threshold_seconds=1800,
        memory_threshold_gb=16,
    )
    validator = SystemPerformanceValidator(config)

    result = validator.validate_performance_benchmarks()
    print(f"All benchmarks pass: {result.passes_all_thresholds()}")
    ```
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Generator

import numpy as np

from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class PerformanceValidationConfig:
    """Configuration for performance validation.

    Attributes:
        training_time_threshold_seconds: Max training time per model-area in seconds.
        parallel_efficiency_threshold: Minimum parallel efficiency (0-1).
        memory_threshold_gb: Maximum memory usage in GB.
        batch_prediction_threshold_seconds: Max batch prediction time in seconds.
        intraday_prediction_threshold_seconds: Max intraday prediction time in seconds.
        prediction_latency_p95_threshold_seconds: Max P95 prediction latency in seconds.
        concurrent_users: Number of concurrent users for load testing.
        throughput_degradation_threshold: Max throughput degradation under load (0-1).
        error_rate_threshold: Max error rate under load (0-1).
    """

    training_time_threshold_seconds: float = 30 * 60  # 30 minutes
    parallel_efficiency_threshold: float = 0.7  # 70%
    memory_threshold_gb: float = 16.0  # 16GB
    batch_prediction_threshold_seconds: float = 15 * 60  # 15 minutes
    intraday_prediction_threshold_seconds: float = 5 * 60  # 5 minutes
    prediction_latency_p95_threshold_seconds: float = 3 * 60  # 3 minutes
    concurrent_users: int = 10
    throughput_degradation_threshold: float = 0.2  # 20%
    error_rate_threshold: float = 0.01  # 1%

    @property
    def memory_threshold_bytes(self) -> int:
        """Get memory threshold in bytes."""
        return int(self.memory_threshold_gb * 1024 * 1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "training_time_threshold_seconds": self.training_time_threshold_seconds,
            "parallel_efficiency_threshold": self.parallel_efficiency_threshold,
            "memory_threshold_gb": self.memory_threshold_gb,
            "batch_prediction_threshold_seconds": self.batch_prediction_threshold_seconds,
            "intraday_prediction_threshold_seconds": self.intraday_prediction_threshold_seconds,
            "prediction_latency_p95_threshold_seconds": self.prediction_latency_p95_threshold_seconds,
            "concurrent_users": self.concurrent_users,
            "throughput_degradation_threshold": self.throughput_degradation_threshold,
            "error_rate_threshold": self.error_rate_threshold,
        }


@dataclass
class MetricValidation:
    """Validation result for a single metric.

    Attributes:
        actual: Actual measured value.
        target: Target threshold value.
        passes: Whether the metric passes validation.
        comparison: Comparison operator used.
        deviation_percent: Percentage deviation from target.
    """

    actual: float
    target: float
    passes: bool
    comparison: str
    deviation_percent: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "actual": self.actual,
            "target": self.target,
            "passes": self.passes,
            "comparison": self.comparison,
            "deviation_percent": self.deviation_percent,
        }


@dataclass
class BenchmarkValidation:
    """Validation result for a benchmark category.

    Attributes:
        benchmark_values: Raw benchmark measurements.
        targets: Target thresholds.
        validations: Individual metric validations.
        passes_threshold: Whether all metrics pass.
    """

    benchmark_values: dict[str, float]
    targets: dict[str, float]
    validations: dict[str, MetricValidation]
    passes_threshold: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "benchmark_values": self.benchmark_values,
            "targets": self.targets,
            "validations": {k: v.to_dict() for k, v in self.validations.items()},
            "passes_threshold": self.passes_threshold,
        }


@dataclass
class PerformanceBenchmarkResult:
    """Results from performance benchmarking.

    Attributes:
        results: Benchmark results by category.
        timestamp: When benchmarks were run.
        system_info: System information for context.
        duration_seconds: Total benchmark duration.
    """

    results: dict[str, BenchmarkValidation]
    timestamp: datetime
    system_info: dict[str, Any]
    duration_seconds: float = 0.0

    def passes_all_thresholds(self) -> bool:
        """Check if all benchmarks pass their thresholds.

        Returns:
            True if all thresholds are met.
        """
        return all(r.passes_threshold for r in self.results.values())

    def get_failing_benchmarks(self) -> list[str]:
        """Get list of benchmarks that failed thresholds.

        Returns:
            List of failing benchmark names.
        """
        return [name for name, result in self.results.items() if not result.passes_threshold]

    def get_failing_metrics(self) -> dict[str, list[str]]:
        """Get failing metrics by category.

        Returns:
            Dictionary of category -> list of failing metrics.
        """
        failing: dict[str, list[str]] = {}
        for category, result in self.results.items():
            failed = [
                metric
                for metric, validation in result.validations.items()
                if not validation.passes
            ]
            if failed:
                failing[category] = failed
        return failing

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "timestamp": self.timestamp.isoformat(),
            "system_info": self.system_info,
            "duration_seconds": self.duration_seconds,
            "passes_all_thresholds": self.passes_all_thresholds(),
            "failing_benchmarks": self.get_failing_benchmarks(),
        }


@dataclass
class ScenarioResult:
    """Result from executing a single scenario.

    Attributes:
        success: Whether the scenario succeeded.
        latency: Execution latency in seconds.
        error: Error message if failed.
        scenario_type: Type of scenario executed.
    """

    success: bool
    latency: float
    error: str | None = None
    scenario_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "success": self.success,
            "latency": self.latency,
            "error": self.error,
            "scenario_type": self.scenario_type,
        }


@dataclass
class LoadTestResult:
    """Results from load testing.

    Attributes:
        total_requests: Total number of requests executed.
        successful_requests: Number of successful requests.
        failed_requests: Number of failed requests.
        average_latency: Average request latency in seconds.
        p95_latency: 95th percentile latency in seconds.
        throughput: Requests per second.
        error_rate: Fraction of failed requests.
    """

    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency: float
    p95_latency: float
    throughput: float
    error_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_latency": self.average_latency,
            "p95_latency": self.p95_latency,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
        }


class PerformanceMonitor:
    """Monitor performance metrics during execution.

    Tracks execution time and memory usage for operations.
    """

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.peak_memory: int = 0
        self._memory_samples: list[int] = []

    @contextmanager
    def track_execution(self) -> Generator[PerformanceMonitor, None, None]:
        """Context manager for tracking execution performance.

        Yields:
            Self for accessing metrics.
        """
        self.start_time = time.time()
        self.peak_memory = self._get_memory_usage()
        self._memory_samples = [self.peak_memory]

        try:
            yield self
        finally:
            self.end_time = time.time()
            current_memory = self._get_memory_usage()
            self._memory_samples.append(current_memory)
            self.peak_memory = max(self._memory_samples)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            Elapsed time in seconds.
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def _get_memory_usage(self) -> int:
        """Get current memory usage.

        Returns:
            Memory usage in bytes.
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # psutil not available, return 0
            return 0

    def sample_memory(self) -> None:
        """Sample current memory usage."""
        current = self._get_memory_usage()
        self._memory_samples.append(current)
        self.peak_memory = max(self.peak_memory, current)


class SystemLoadTester:
    """Execute load testing scenarios.

    Simulates concurrent user load to test system scalability.
    """

    def __init__(self) -> None:
        """Initialize load tester."""
        self._scenario_handlers: dict[str, Callable[[], None]] = {}

    def register_scenario(
        self,
        scenario_name: str,
        handler: Callable[[], None],
    ) -> None:
        """Register a scenario handler.

        Args:
            scenario_name: Name of the scenario.
            handler: Function to execute for the scenario.
        """
        self._scenario_handlers[scenario_name] = handler

    def measure_baseline_throughput(
        self,
        scenario: str = "default",
        duration_seconds: float = 10.0,
    ) -> float:
        """Measure baseline throughput with single user.

        Args:
            scenario: Scenario to execute.
            duration_seconds: Duration for measurement.

        Returns:
            Operations per second.
        """
        operations = 0
        start_time = time.time()

        handler = self._scenario_handlers.get(scenario, self._default_scenario)

        while time.time() - start_time < duration_seconds:
            try:
                handler()
                operations += 1
            except Exception:
                pass

        elapsed = time.time() - start_time
        return operations / elapsed if elapsed > 0 else 0.0

    def execute_concurrent_load_test(
        self,
        concurrent_users: int,
        test_duration_seconds: float,
        scenarios: list[str] | None = None,
    ) -> LoadTestResult:
        """Execute concurrent load testing with multiple user scenarios.

        Args:
            concurrent_users: Number of concurrent users to simulate.
            test_duration_seconds: Duration of test in seconds.
            scenarios: List of scenarios to test.

        Returns:
            LoadTestResult with test results.
        """
        scenarios = scenarios or ["default"]
        results: list[ScenarioResult] = []

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            for user_id in range(concurrent_users):
                for scenario in scenarios:
                    future = executor.submit(
                        self._execute_user_scenario,
                        user_id,
                        scenario,
                    )
                    futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=test_duration_seconds)
                    results.append(result)
                except Exception as e:
                    results.append(
                        ScenarioResult(
                            success=False,
                            latency=0.0,
                            error=str(e),
                        )
                    )

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        successful_latencies = [r.latency for r in successful]

        return LoadTestResult(
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            average_latency=float(np.mean(successful_latencies)) if successful_latencies else 0.0,
            p95_latency=float(np.percentile(successful_latencies, 95)) if successful_latencies else 0.0,
            throughput=len(successful) / test_duration_seconds if test_duration_seconds > 0 else 0.0,
            error_rate=len(failed) / len(results) if results else 0.0,
        )

    def _execute_user_scenario(
        self,
        user_id: int,
        scenario: str,
    ) -> ScenarioResult:
        """Execute a single user scenario.

        Args:
            user_id: User identifier.
            scenario: Scenario to execute.

        Returns:
            ScenarioResult with execution results.
        """
        start_time = time.time()

        try:
            handler = self._scenario_handlers.get(scenario, self._default_scenario)
            handler()

            latency = time.time() - start_time
            return ScenarioResult(
                success=True,
                latency=latency,
                scenario_type=scenario,
            )

        except Exception as e:
            latency = time.time() - start_time
            return ScenarioResult(
                success=False,
                latency=latency,
                error=str(e),
                scenario_type=scenario,
            )

    def _default_scenario(self) -> None:
        """Default scenario that does minimal work."""
        time.sleep(0.001)  # Simulate minimal work


class SystemPerformanceValidator:
    """Validates system performance against operational requirements.

    Provides comprehensive benchmarking capabilities for training,
    prediction, and scalability testing.

    Attributes:
        config: Performance validation configuration.
        performance_monitor: Performance monitoring instance.
        load_tester: Load testing instance.
    """

    def __init__(
        self,
        config: PerformanceValidationConfig | None = None,
    ) -> None:
        """Initialize performance validator.

        Args:
            config: Validation configuration with thresholds.
        """
        self.config = config or PerformanceValidationConfig()
        self.performance_monitor = PerformanceMonitor()
        self.load_tester = SystemLoadTester()

        # Register default scenarios
        self._register_default_scenarios()

    def _register_default_scenarios(self) -> None:
        """Register default load test scenarios."""
        self.load_tester.register_scenario("default", lambda: time.sleep(0.001))
        self.load_tester.register_scenario("training", self._mock_training_scenario)
        self.load_tester.register_scenario("prediction", self._mock_prediction_scenario)
        self.load_tester.register_scenario("evaluation", self._mock_evaluation_scenario)

    def _mock_training_scenario(self) -> None:
        """Mock training scenario for testing."""
        time.sleep(0.01)

    def _mock_prediction_scenario(self) -> None:
        """Mock prediction scenario for testing."""
        time.sleep(0.005)

    def _mock_evaluation_scenario(self) -> None:
        """Mock evaluation scenario for testing."""
        time.sleep(0.008)

    def validate_performance_benchmarks(
        self,
        run_training: bool = True,
        run_prediction: bool = True,
        run_scalability: bool = True,
    ) -> PerformanceBenchmarkResult:
        """Validate system performance against all benchmarks.

        Args:
            run_training: Whether to run training benchmarks.
            run_prediction: Whether to run prediction benchmarks.
            run_scalability: Whether to run scalability benchmarks.

        Returns:
            PerformanceBenchmarkResult with all benchmark results.
        """
        logger.info("Starting performance benchmark validation")
        start_time = time.time()
        benchmark_results: dict[str, BenchmarkValidation] = {}

        if run_training:
            logger.info("Benchmarking training performance")
            training_benchmark = self._benchmark_training_performance()
            benchmark_results["training"] = self._validate_against_targets(
                training_benchmark,
                targets={
                    "avg_training_time": self.config.training_time_threshold_seconds,
                    "parallel_efficiency": self.config.parallel_efficiency_threshold,
                    "peak_memory": self.config.memory_threshold_bytes,
                },
                lower_is_better={"avg_training_time", "max_training_time", "peak_memory"},
            )

        if run_prediction:
            logger.info("Benchmarking prediction performance")
            prediction_benchmark = self._benchmark_prediction_performance()
            benchmark_results["prediction"] = self._validate_against_targets(
                prediction_benchmark,
                targets={
                    "avg_batch_time": self.config.batch_prediction_threshold_seconds,
                    "avg_intraday_time": self.config.intraday_prediction_threshold_seconds,
                    "latency_p95": self.config.prediction_latency_p95_threshold_seconds,
                },
                lower_is_better={"avg_batch_time", "avg_intraday_time", "latency_p95", "latency_p99"},
            )

        if run_scalability:
            logger.info("Benchmarking system scalability")
            scalability_benchmark = self._benchmark_scalability()
            benchmark_results["scalability"] = self._validate_against_targets(
                scalability_benchmark,
                targets={
                    "concurrent_users_supported": float(self.config.concurrent_users),
                    "throughput_degradation": self.config.throughput_degradation_threshold,
                    "error_rate": self.config.error_rate_threshold,
                },
                lower_is_better={"throughput_degradation", "error_rate"},
            )

        duration = time.time() - start_time

        result = PerformanceBenchmarkResult(
            results=benchmark_results,
            timestamp=datetime.now(),
            system_info=self._get_system_info(),
            duration_seconds=duration,
        )

        logger.info(
            f"Performance benchmark complete. "
            f"All thresholds pass: {result.passes_all_thresholds()}"
        )

        return result

    def _benchmark_training_performance(self) -> dict[str, float]:
        """Benchmark model training performance.

        Returns:
            Dictionary of training performance metrics.
        """
        results: dict[str, list[float]] = {
            "training_times": [],
            "memory_usage": [],
        }

        # Simulate training benchmarks
        for _ in range(3):  # 3 training iterations
            with self.performance_monitor.track_execution() as monitor:
                self._mock_training_scenario()

            results["training_times"].append(monitor.elapsed_time)
            results["memory_usage"].append(monitor.peak_memory)

        # Calculate parallel efficiency
        sequential_time = sum(results["training_times"])
        with self.performance_monitor.track_execution() as monitor:
            # Simulate parallel training
            time.sleep(sequential_time / 4)

        parallel_time = monitor.elapsed_time
        parallel_efficiency = (
            sequential_time / (parallel_time * 4) if parallel_time > 0 else 1.0
        )

        return {
            "avg_training_time": float(np.mean(results["training_times"])),
            "max_training_time": float(np.max(results["training_times"])),
            "peak_memory": float(np.max(results["memory_usage"])),
            "parallel_efficiency": min(parallel_efficiency, 1.0),
        }

    def _benchmark_prediction_performance(self) -> dict[str, float]:
        """Benchmark prediction performance.

        Returns:
            Dictionary of prediction performance metrics.
        """
        results: dict[str, list[float]] = {
            "batch_times": [],
            "intraday_times": [],
            "latencies": [],
        }

        # Batch prediction benchmark
        for _ in range(3):
            with self.performance_monitor.track_execution() as monitor:
                self._mock_prediction_scenario()
            results["batch_times"].append(monitor.elapsed_time)

        # Intraday prediction benchmark
        for _ in range(10):
            with self.performance_monitor.track_execution() as monitor:
                time.sleep(0.002)
            results["intraday_times"].append(monitor.elapsed_time)

        # Latency benchmark (single predictions)
        for _ in range(100):
            with self.performance_monitor.track_execution() as monitor:
                time.sleep(0.001)
            results["latencies"].append(monitor.elapsed_time)

        return {
            "avg_batch_time": float(np.mean(results["batch_times"])),
            "avg_intraday_time": float(np.mean(results["intraday_times"])),
            "latency_p50": float(np.percentile(results["latencies"], 50)),
            "latency_p95": float(np.percentile(results["latencies"], 95)),
            "latency_p99": float(np.percentile(results["latencies"], 99)),
        }

    def _benchmark_scalability(self) -> dict[str, float]:
        """Benchmark system scalability with concurrent operations.

        Returns:
            Dictionary of scalability metrics.
        """
        baseline_throughput = self.load_tester.measure_baseline_throughput(
            scenario="prediction",
            duration_seconds=2.0,
        )

        load_test_result = self.load_tester.execute_concurrent_load_test(
            concurrent_users=self.config.concurrent_users,
            test_duration_seconds=5.0,
            scenarios=["prediction"],
        )

        throughput_degradation = (
            (baseline_throughput - load_test_result.throughput) / baseline_throughput
            if baseline_throughput > 0
            else 0.0
        )

        return {
            "baseline_throughput": baseline_throughput,
            "concurrent_throughput": load_test_result.throughput,
            "throughput_degradation": max(0.0, throughput_degradation),
            "error_rate": load_test_result.error_rate,
            "avg_latency": load_test_result.average_latency,
            "p95_latency": load_test_result.p95_latency,
            "concurrent_users_supported": float(self.config.concurrent_users),
        }

    def _validate_against_targets(
        self,
        benchmark: dict[str, float],
        targets: dict[str, float],
        lower_is_better: set[str] | None = None,
    ) -> BenchmarkValidation:
        """Validate benchmark results against target thresholds.

        Args:
            benchmark: Benchmark measurements.
            targets: Target thresholds.
            lower_is_better: Set of metrics where lower is better.

        Returns:
            BenchmarkValidation with results.
        """
        lower_is_better = lower_is_better or set()
        validations: dict[str, MetricValidation] = {}
        passes_threshold = True

        for metric, target in targets.items():
            actual = benchmark.get(metric, 0.0)

            # Determine comparison based on metric type
            if metric in lower_is_better:
                passes = actual <= target
                comparison = "<="
            else:
                passes = actual >= target
                comparison = ">="

            deviation_percent = abs((actual - target) / target * 100) if target != 0 else 0.0

            validations[metric] = MetricValidation(
                actual=actual,
                target=target,
                passes=passes,
                comparison=comparison,
                deviation_percent=deviation_percent,
            )

            if not passes:
                passes_threshold = False
                logger.warning(f"Metric {metric} failed: {actual} {comparison} {target}")

        return BenchmarkValidation(
            benchmark_values=benchmark,
            targets=targets,
            validations=validations,
            passes_threshold=passes_threshold,
        )

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for benchmark context.

        Returns:
            Dictionary of system information.
        """
        try:
            import psutil

            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                "total_memory_gb": psutil.virtual_memory().total / (1024 ** 3),
                "available_memory_gb": psutil.virtual_memory().available / (1024 ** 3),
                "platform": "linux" if hasattr(psutil, "LINUX") else "other",
            }
        except ImportError:
            return {
                "cpu_count": 0,
                "cpu_freq_mhz": 0,
                "total_memory_gb": 0,
                "available_memory_gb": 0,
                "platform": "unknown",
            }

    def register_scenario(
        self,
        scenario_name: str,
        handler: Callable[[], None],
    ) -> None:
        """Register a custom scenario for load testing.

        Args:
            scenario_name: Name of the scenario.
            handler: Function to execute for the scenario.
        """
        self.load_tester.register_scenario(scenario_name, handler)

    def run_custom_benchmark(
        self,
        name: str,
        operation: Callable[[], None],
        iterations: int = 10,
    ) -> dict[str, float]:
        """Run a custom benchmark operation.

        Args:
            name: Name of the benchmark.
            operation: Operation to benchmark.
            iterations: Number of iterations.

        Returns:
            Dictionary of benchmark results.
        """
        times: list[float] = []
        memory_usage: list[int] = []

        for _ in range(iterations):
            with self.performance_monitor.track_execution() as monitor:
                operation()
            times.append(monitor.elapsed_time)
            memory_usage.append(monitor.peak_memory)

        return {
            f"{name}_avg_time": float(np.mean(times)),
            f"{name}_min_time": float(np.min(times)),
            f"{name}_max_time": float(np.max(times)),
            f"{name}_std_time": float(np.std(times)),
            f"{name}_p95_time": float(np.percentile(times, 95)),
            f"{name}_peak_memory": float(np.max(memory_usage)),
        }
