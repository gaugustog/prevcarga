"""Tests for the system performance validator module.

Tests cover:
- PerformanceValidationConfig validation and defaults
- PerformanceMonitor tracking
- SystemLoadTester concurrent execution
- SystemPerformanceValidator benchmarking
- Benchmark result validation
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.validation.system_performance_validator import (
    BenchmarkValidation,
    LoadTestResult,
    MetricValidation,
    PerformanceBenchmarkResult,
    PerformanceMonitor,
    PerformanceValidationConfig,
    ScenarioResult,
    SystemLoadTester,
    SystemPerformanceValidator,
)


class TestPerformanceValidationConfig:
    """Tests for PerformanceValidationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PerformanceValidationConfig()

        assert config.training_time_threshold_seconds == 30 * 60
        assert config.parallel_efficiency_threshold == 0.7
        assert config.memory_threshold_gb == 16.0
        assert config.concurrent_users == 10
        assert config.error_rate_threshold == 0.01

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PerformanceValidationConfig(
            training_time_threshold_seconds=1800,
            parallel_efficiency_threshold=0.8,
            memory_threshold_gb=32.0,
            concurrent_users=20,
        )

        assert config.training_time_threshold_seconds == 1800
        assert config.parallel_efficiency_threshold == 0.8
        assert config.memory_threshold_gb == 32.0
        assert config.concurrent_users == 20

    def test_memory_threshold_bytes(self) -> None:
        """Test memory threshold conversion to bytes."""
        config = PerformanceValidationConfig(memory_threshold_gb=16.0)
        expected = 16 * 1024 * 1024 * 1024
        assert config.memory_threshold_bytes == expected

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = PerformanceValidationConfig()
        result = config.to_dict()

        assert "training_time_threshold_seconds" in result
        assert "parallel_efficiency_threshold" in result
        assert "memory_threshold_gb" in result
        assert "concurrent_users" in result


class TestMetricValidation:
    """Tests for MetricValidation dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        validation = MetricValidation(
            actual=1500.0,
            target=1800.0,
            passes=True,
            comparison="<=",
            deviation_percent=16.67,
        )

        result = validation.to_dict()

        assert result["actual"] == 1500.0
        assert result["target"] == 1800.0
        assert result["passes"] is True
        assert result["comparison"] == "<="


class TestBenchmarkValidation:
    """Tests for BenchmarkValidation dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        validation = BenchmarkValidation(
            benchmark_values={"metric1": 100.0},
            targets={"metric1": 150.0},
            validations={
                "metric1": MetricValidation(
                    actual=100.0,
                    target=150.0,
                    passes=True,
                    comparison="<=",
                    deviation_percent=33.33,
                )
            },
            passes_threshold=True,
        )

        result = validation.to_dict()

        assert "benchmark_values" in result
        assert "targets" in result
        assert "validations" in result
        assert result["passes_threshold"] is True


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful scenario result."""
        result = ScenarioResult(
            success=True,
            latency=0.5,
            scenario_type="prediction",
        )

        assert result.success is True
        assert result.latency == 0.5
        assert result.error is None

    def test_failure_result(self) -> None:
        """Test failed scenario result."""
        result = ScenarioResult(
            success=False,
            latency=1.0,
            error="Connection timeout",
            scenario_type="training",
        )

        assert result.success is False
        assert result.error == "Connection timeout"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = ScenarioResult(success=True, latency=0.5)
        d = result.to_dict()

        assert "success" in d
        assert "latency" in d
        assert "error" in d


class TestLoadTestResult:
    """Tests for LoadTestResult dataclass."""

    def test_all_successful(self) -> None:
        """Test result with all successful requests."""
        result = LoadTestResult(
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            average_latency=0.1,
            p95_latency=0.15,
            throughput=50.0,
            error_rate=0.0,
        )

        assert result.error_rate == 0.0
        assert result.successful_requests == result.total_requests

    def test_some_failures(self) -> None:
        """Test result with some failures."""
        result = LoadTestResult(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            average_latency=0.1,
            p95_latency=0.15,
            throughput=47.5,
            error_rate=0.05,
        )

        assert result.error_rate == 0.05
        assert result.failed_requests == 5

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = LoadTestResult(
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            average_latency=0.1,
            p95_latency=0.15,
            throughput=50.0,
            error_rate=0.0,
        )

        d = result.to_dict()

        assert "total_requests" in d
        assert "throughput" in d
        assert "error_rate" in d


class TestPerformanceBenchmarkResult:
    """Tests for PerformanceBenchmarkResult dataclass."""

    @pytest.fixture
    def passing_result(self) -> PerformanceBenchmarkResult:
        """Create passing benchmark result."""
        from datetime import datetime

        return PerformanceBenchmarkResult(
            results={
                "training": BenchmarkValidation(
                    benchmark_values={},
                    targets={},
                    validations={},
                    passes_threshold=True,
                ),
                "prediction": BenchmarkValidation(
                    benchmark_values={},
                    targets={},
                    validations={},
                    passes_threshold=True,
                ),
            },
            timestamp=datetime.now(),
            system_info={"cpu_count": 8},
            duration_seconds=60.0,
        )

    @pytest.fixture
    def failing_result(self) -> PerformanceBenchmarkResult:
        """Create failing benchmark result."""
        from datetime import datetime

        return PerformanceBenchmarkResult(
            results={
                "training": BenchmarkValidation(
                    benchmark_values={},
                    targets={},
                    validations={},
                    passes_threshold=False,
                ),
                "prediction": BenchmarkValidation(
                    benchmark_values={},
                    targets={},
                    validations={},
                    passes_threshold=True,
                ),
            },
            timestamp=datetime.now(),
            system_info={"cpu_count": 8},
            duration_seconds=60.0,
        )

    def test_passes_all_thresholds_true(
        self, passing_result: PerformanceBenchmarkResult
    ) -> None:
        """Test when all thresholds pass."""
        assert passing_result.passes_all_thresholds()

    def test_passes_all_thresholds_false(
        self, failing_result: PerformanceBenchmarkResult
    ) -> None:
        """Test when some thresholds fail."""
        assert not failing_result.passes_all_thresholds()

    def test_get_failing_benchmarks(
        self, failing_result: PerformanceBenchmarkResult
    ) -> None:
        """Test getting failing benchmarks."""
        failing = failing_result.get_failing_benchmarks()
        assert "training" in failing
        assert "prediction" not in failing

    def test_to_dict(self, passing_result: PerformanceBenchmarkResult) -> None:
        """Test conversion to dictionary."""
        result = passing_result.to_dict()

        assert "results" in result
        assert "timestamp" in result
        assert "system_info" in result
        assert "passes_all_thresholds" in result


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    @pytest.fixture
    def monitor(self) -> PerformanceMonitor:
        """Create monitor instance."""
        return PerformanceMonitor()

    def test_track_execution(self, monitor: PerformanceMonitor) -> None:
        """Test execution tracking."""
        with monitor.track_execution():
            time.sleep(0.05)

        assert monitor.elapsed_time >= 0.05

    def test_track_execution_multiple(self, monitor: PerformanceMonitor) -> None:
        """Test multiple execution tracking."""
        with monitor.track_execution():
            time.sleep(0.02)

        first_time = monitor.elapsed_time

        with monitor.track_execution():
            time.sleep(0.03)

        second_time = monitor.elapsed_time

        assert first_time >= 0.02
        assert second_time >= 0.03

    def test_elapsed_time_before_tracking(self, monitor: PerformanceMonitor) -> None:
        """Test elapsed time before any tracking."""
        assert monitor.elapsed_time == 0.0

    def test_sample_memory(self, monitor: PerformanceMonitor) -> None:
        """Test memory sampling."""
        monitor.sample_memory()
        # May be 0 if psutil not available, but should not raise


class TestSystemLoadTester:
    """Tests for SystemLoadTester class."""

    @pytest.fixture
    def load_tester(self) -> SystemLoadTester:
        """Create load tester instance."""
        return SystemLoadTester()

    def test_register_scenario(self, load_tester: SystemLoadTester) -> None:
        """Test registering a scenario."""
        executed = []

        def custom_scenario():
            executed.append(True)

        load_tester.register_scenario("custom", custom_scenario)
        load_tester._execute_user_scenario(0, "custom")

        assert len(executed) == 1

    def test_measure_baseline_throughput(self, load_tester: SystemLoadTester) -> None:
        """Test baseline throughput measurement."""
        # Register fast scenario
        load_tester.register_scenario("fast", lambda: time.sleep(0.001))

        throughput = load_tester.measure_baseline_throughput(
            scenario="fast",
            duration_seconds=0.5,
        )

        assert throughput > 0

    def test_execute_concurrent_load_test(self, load_tester: SystemLoadTester) -> None:
        """Test concurrent load testing."""
        # Register fast scenario
        load_tester.register_scenario("fast", lambda: time.sleep(0.001))

        result = load_tester.execute_concurrent_load_test(
            concurrent_users=3,
            test_duration_seconds=1.0,
            scenarios=["fast"],
        )

        assert result.total_requests > 0
        assert result.successful_requests <= result.total_requests
        assert result.error_rate <= 1.0

    def test_execute_user_scenario_success(
        self, load_tester: SystemLoadTester
    ) -> None:
        """Test successful scenario execution."""
        load_tester.register_scenario("success", lambda: time.sleep(0.001))

        result = load_tester._execute_user_scenario(0, "success")

        assert result.success is True
        assert result.latency > 0
        assert result.error is None

    def test_execute_user_scenario_failure(
        self, load_tester: SystemLoadTester
    ) -> None:
        """Test failed scenario execution."""

        def failing_scenario():
            raise RuntimeError("Test error")

        load_tester.register_scenario("failing", failing_scenario)

        result = load_tester._execute_user_scenario(0, "failing")

        assert result.success is False
        assert result.error is not None
        assert "Test error" in result.error

    def test_default_scenario(self, load_tester: SystemLoadTester) -> None:
        """Test default scenario execution."""
        result = load_tester._execute_user_scenario(0, "nonexistent")

        # Should fall back to default scenario
        assert result.success is True


class TestSystemPerformanceValidator:
    """Tests for SystemPerformanceValidator class."""

    @pytest.fixture
    def config(self) -> PerformanceValidationConfig:
        """Create test configuration."""
        return PerformanceValidationConfig(
            training_time_threshold_seconds=100.0,
            parallel_efficiency_threshold=0.5,
            memory_threshold_gb=32.0,
            batch_prediction_threshold_seconds=100.0,
            intraday_prediction_threshold_seconds=100.0,
            prediction_latency_p95_threshold_seconds=100.0,
            concurrent_users=3,
            throughput_degradation_threshold=0.5,
            error_rate_threshold=0.1,
        )

    @pytest.fixture
    def validator(
        self, config: PerformanceValidationConfig
    ) -> SystemPerformanceValidator:
        """Create validator instance."""
        return SystemPerformanceValidator(config)

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        validator = SystemPerformanceValidator()
        assert validator.config is not None

    def test_init_custom_config(
        self, config: PerformanceValidationConfig
    ) -> None:
        """Test initialization with custom config."""
        validator = SystemPerformanceValidator(config)
        assert validator.config.concurrent_users == 3

    def test_validate_performance_benchmarks(
        self, validator: SystemPerformanceValidator
    ) -> None:
        """Test complete performance benchmark validation."""
        result = validator.validate_performance_benchmarks(
            run_training=True,
            run_prediction=True,
            run_scalability=True,
        )

        assert isinstance(result, PerformanceBenchmarkResult)
        assert "training" in result.results
        assert "prediction" in result.results
        assert "scalability" in result.results

    def test_validate_training_only(
        self, validator: SystemPerformanceValidator
    ) -> None:
        """Test training-only benchmark."""
        result = validator.validate_performance_benchmarks(
            run_training=True,
            run_prediction=False,
            run_scalability=False,
        )

        assert "training" in result.results
        assert "prediction" not in result.results
        assert "scalability" not in result.results

    def test_benchmark_training_performance(
        self, validator: SystemPerformanceValidator
    ) -> None:
        """Test training performance benchmarking."""
        result = validator._benchmark_training_performance()

        assert "avg_training_time" in result
        assert "max_training_time" in result
        assert "parallel_efficiency" in result
        assert "peak_memory" in result

    def test_benchmark_prediction_performance(
        self, validator: SystemPerformanceValidator
    ) -> None:
        """Test prediction performance benchmarking."""
        result = validator._benchmark_prediction_performance()

        assert "avg_batch_time" in result
        assert "avg_intraday_time" in result
        assert "latency_p50" in result
        assert "latency_p95" in result
        assert "latency_p99" in result

    def test_benchmark_scalability(
        self, validator: SystemPerformanceValidator
    ) -> None:
        """Test scalability benchmarking."""
        result = validator._benchmark_scalability()

        assert "baseline_throughput" in result
        assert "concurrent_throughput" in result
        assert "throughput_degradation" in result
        assert "error_rate" in result

    def test_validate_against_targets_pass(
        self, validator: SystemPerformanceValidator
    ) -> None:
        """Test validation when metrics pass."""
        benchmark = {
            "avg_training_time": 1500.0,
            "parallel_efficiency": 0.75,
        }
        targets = {
            "avg_training_time": 1800.0,
            "parallel_efficiency": 0.7,
        }

        result = validator._validate_against_targets(
            benchmark,
            targets,
            lower_is_better={"avg_training_time"},
        )

        assert result.passes_threshold is True
        assert result.validations["avg_training_time"].passes is True
        assert result.validations["parallel_efficiency"].passes is True

    def test_validate_against_targets_fail(
        self, validator: SystemPerformanceValidator
    ) -> None:
        """Test validation when metrics fail."""
        benchmark = {
            "avg_training_time": 2000.0,
            "parallel_efficiency": 0.6,
        }
        targets = {
            "avg_training_time": 1800.0,
            "parallel_efficiency": 0.7,
        }

        result = validator._validate_against_targets(
            benchmark,
            targets,
            lower_is_better={"avg_training_time"},
        )

        assert result.passes_threshold is False

    def test_get_system_info(self, validator: SystemPerformanceValidator) -> None:
        """Test getting system information."""
        info = validator._get_system_info()

        assert "cpu_count" in info
        assert "platform" in info

    def test_register_scenario(self, validator: SystemPerformanceValidator) -> None:
        """Test registering custom scenario."""
        executed = []

        def custom():
            executed.append(True)

        validator.register_scenario("custom", custom)

        # Verify it's registered
        result = validator.load_tester._execute_user_scenario(0, "custom")
        assert result.success is True
        assert len(executed) == 1

    def test_run_custom_benchmark(
        self, validator: SystemPerformanceValidator
    ) -> None:
        """Test running custom benchmark."""

        def custom_operation():
            time.sleep(0.01)

        result = validator.run_custom_benchmark(
            name="custom",
            operation=custom_operation,
            iterations=5,
        )

        assert "custom_avg_time" in result
        assert "custom_min_time" in result
        assert "custom_max_time" in result
        assert "custom_p95_time" in result
        assert result["custom_avg_time"] >= 0.01


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_duration_throughput(self) -> None:
        """Test throughput calculation with zero duration."""
        load_tester = SystemLoadTester()

        # Should handle gracefully
        result = load_tester.execute_concurrent_load_test(
            concurrent_users=1,
            test_duration_seconds=0.1,
            scenarios=["default"],
        )

        assert result.total_requests >= 0

    def test_empty_scenarios(self) -> None:
        """Test load test with default scenarios."""
        load_tester = SystemLoadTester()

        result = load_tester.execute_concurrent_load_test(
            concurrent_users=1,
            test_duration_seconds=0.5,
            scenarios=None,  # Use default
        )

        assert result.total_requests >= 0

    def test_validation_zero_target(self) -> None:
        """Test validation with zero target."""
        validator = SystemPerformanceValidator()

        benchmark = {"metric": 100.0}
        targets = {"metric": 0.0}

        result = validator._validate_against_targets(
            benchmark,
            targets,
            lower_is_better=set(),
        )

        # Should handle division by zero gracefully
        assert "metric" in result.validations

    def test_failing_metrics_report(self) -> None:
        """Test getting failing metrics report."""
        from datetime import datetime

        result = PerformanceBenchmarkResult(
            results={
                "training": BenchmarkValidation(
                    benchmark_values={},
                    targets={},
                    validations={
                        "time": MetricValidation(
                            actual=2000,
                            target=1800,
                            passes=False,
                            comparison="<=",
                            deviation_percent=11.11,
                        )
                    },
                    passes_threshold=False,
                ),
            },
            timestamp=datetime.now(),
            system_info={},
            duration_seconds=60.0,
        )

        failing = result.get_failing_metrics()

        assert "training" in failing
        assert "time" in failing["training"]
