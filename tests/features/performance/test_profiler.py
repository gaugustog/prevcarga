"""Tests for execution profiler."""

import time

import pandas as pd
import pytest

from src.features.performance.profiler import (
    ExecutionProfiler,
    profile_execution,
)


def test_profiler_context_manager():
    """Test profiler context manager."""
    profiler = ExecutionProfiler(enable_memory=False)

    with profiler.profile("test_operation"):
        time.sleep(0.01)

    results = profiler.get_results_df()
    assert len(results) == 1
    assert results.iloc[0]["operation"] == "test_operation"
    assert results.iloc[0]["execution_time_seconds"] >= 0.01


def test_profiler_multiple_operations():
    """Test profiling multiple operations."""
    profiler = ExecutionProfiler(enable_memory=False)

    with profiler.profile("op1"):
        time.sleep(0.01)

    with profiler.profile("op2"):
        time.sleep(0.02)

    results = profiler.get_results_df()
    assert len(results) == 2
    assert results.iloc[0]["operation"] == "op1"
    assert results.iloc[1]["operation"] == "op2"


def test_profiler_with_metadata():
    """Test profiler with metadata."""
    profiler = ExecutionProfiler(enable_memory=False)

    with profiler.profile("test_op", metadata={"plugin": "lag", "horizon": 1}):
        time.sleep(0.01)

    results = profiler.get_results_df()
    assert len(results) == 1
    assert results.iloc[0]["plugin"] == "lag"
    assert results.iloc[0]["horizon"] == 1


def test_profiler_summary():
    """Test profiler summary generation."""
    profiler = ExecutionProfiler(enable_memory=False)

    # Profile same operation multiple times
    for _ in range(3):
        with profiler.profile("repeated_op"):
            time.sleep(0.01)

    summary = profiler.get_summary()
    assert len(summary) == 1
    assert summary.iloc[0]["call_count"] == 3
    assert summary.iloc[0]["mean_time_seconds"] >= 0.01


def test_profiler_report():
    """Test profiler report generation."""
    profiler = ExecutionProfiler(enable_memory=False)

    with profiler.profile("op1"):
        time.sleep(0.01)

    with profiler.profile("op2"):
        time.sleep(0.02)

    report = profiler.get_report(top_n=2)
    assert "EXECUTION PROFILING REPORT" in report
    assert "op1" in report or "op2" in report


def test_profiler_reset():
    """Test profiler reset."""
    profiler = ExecutionProfiler(enable_memory=False)

    with profiler.profile("test_op"):
        time.sleep(0.01)

    profiler.reset()

    results = profiler.get_results_df()
    assert len(results) == 0


def test_profiler_decorator():
    """Test profiler decorator."""
    profiler = ExecutionProfiler(enable_memory=False)

    @profile_execution(profiler, "decorated_func")
    def test_function():
        time.sleep(0.01)
        return 42

    result = test_function()

    assert result == 42

    results = profiler.get_results_df()
    assert len(results) == 1
    assert results.iloc[0]["operation"] == "decorated_func"


def test_profiler_decorator_with_args():
    """Test profiler decorator with function arguments."""
    profiler = ExecutionProfiler(enable_memory=False)

    @profile_execution(profiler, "func_with_args")
    def test_function(x, y):
        time.sleep(0.01)
        return x + y

    result = test_function(2, 3)

    assert result == 5

    results = profiler.get_results_df()
    assert len(results) == 1


def test_profiler_decorator_auto_name():
    """Test profiler decorator with automatic naming."""
    profiler = ExecutionProfiler(enable_memory=False)

    @profile_execution(profiler)
    def my_test_function():
        time.sleep(0.01)
        return 42

    my_test_function()

    results = profiler.get_results_df()
    assert len(results) == 1
    assert results.iloc[0]["operation"] == "my_test_function"


def test_profiler_memory_tracking():
    """Test memory profiling."""
    profiler = ExecutionProfiler(enable_memory=True)

    if profiler.enable_memory_profiling:
        with profiler.profile("memory_test"):
            # Allocate some memory
            data = [i for i in range(100000)]

        results = profiler.get_results_df()
        assert len(results) == 1
        assert "start_memory_mb" in results.columns
        assert "end_memory_mb" in results.columns
        assert "memory_delta_mb" in results.columns
    else:
        pytest.skip("psutil not available for memory profiling")


def test_profiler_save_results(tmp_path):
    """Test saving profiler results."""
    profiler = ExecutionProfiler(enable_memory=False)

    with profiler.profile("test_op"):
        time.sleep(0.01)

    # Test CSV
    csv_path = tmp_path / "results.csv"
    profiler.save_results(str(csv_path))
    assert csv_path.exists()

    # Test Parquet
    parquet_path = tmp_path / "results.parquet"
    profiler.save_results(str(parquet_path))
    assert parquet_path.exists()

    # Test JSON
    json_path = tmp_path / "results.json"
    profiler.save_results(str(json_path))
    assert json_path.exists()


def test_profiler_empty_results():
    """Test profiler with no results."""
    profiler = ExecutionProfiler(enable_memory=False)

    results = profiler.get_results_df()
    assert len(results) == 0

    summary = profiler.get_summary()
    assert len(summary) == 0

    report = profiler.get_report()
    assert "No profiling results" in report


def test_profiler_exception_handling():
    """Test profiler handles exceptions in profiled code."""
    profiler = ExecutionProfiler(enable_memory=False)

    try:
        with profiler.profile("failing_op"):
            raise ValueError("Test error")
    except ValueError:
        pass

    # Should still record the operation
    results = profiler.get_results_df()
    assert len(results) == 1
    assert results.iloc[0]["operation"] == "failing_op"
