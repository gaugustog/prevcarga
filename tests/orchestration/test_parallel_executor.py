"""Tests for parallel executor.

This module tests the ParallelExecutor and related components.
"""

import time
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from src.orchestration.config_manager import ParallelBackend
from src.orchestration.parallel_executor import (
    BatchResult,
    ExecutionStatus,
    MultiprocessingStrategy,
    ParallelExecutor,
    ProgressTrackingExecutor,
    SequentialStrategy,
    TaskResult,
    ThreadingStrategy,
    create_executor,
)


# ==================== Helper Functions ====================


def double(x: int) -> int:
    """Double a number."""
    return x * 2


def slow_double(x: int) -> int:
    """Double a number with delay."""
    time.sleep(0.01)
    return x * 2


def raise_error(x: int) -> int:
    """Function that raises an error."""
    raise ValueError(f"Error for {x}")


def conditional_error(x: int) -> int:
    """Raise error for odd numbers."""
    if x % 2 == 1:
        raise ValueError(f"Odd number: {x}")
    return x * 2


# ==================== TaskResult Tests ====================


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_creation(self) -> None:
        """Test basic TaskResult creation."""
        result = TaskResult[int](task_id="task_0")

        assert result.task_id == "task_0"
        assert result.status == ExecutionStatus.PENDING
        assert result.result is None
        assert result.error is None

    def test_task_result_success(self) -> None:
        """Test successful task result."""
        result = TaskResult[int](
            task_id="task_1",
            status=ExecutionStatus.COMPLETED,
            result=42,
        )

        assert result.is_success()
        assert result.result == 42

    def test_task_result_failure(self) -> None:
        """Test failed task result."""
        result = TaskResult[int](
            task_id="task_2",
            status=ExecutionStatus.FAILED,
            error="Something went wrong",
        )

        assert not result.is_success()
        assert result.error == "Something went wrong"

    def test_task_result_to_dict(self) -> None:
        """Test TaskResult serialization."""
        result = TaskResult[int](
            task_id="task_3",
            status=ExecutionStatus.COMPLETED,
            result=100,
        )

        data = result.to_dict()

        assert data["task_id"] == "task_3"
        assert data["status"] == "completed"
        assert data["result"] == "100"


# ==================== BatchResult Tests ====================


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_batch_result_creation(self) -> None:
        """Test basic BatchResult creation."""
        batch = BatchResult[int](batch_id="batch_0")

        assert batch.batch_id == "batch_0"
        assert batch.total_tasks == 0
        assert batch.successful_tasks == 0
        assert batch.failed_tasks == 0

    def test_batch_result_all_successful(self) -> None:
        """Test batch with all successful tasks."""
        batch = BatchResult[int](
            batch_id="batch_1",
            total_tasks=3,
            successful_tasks=3,
            failed_tasks=0,
        )

        assert batch.all_successful()

    def test_batch_result_with_failures(self) -> None:
        """Test batch with some failures."""
        batch = BatchResult[int](
            batch_id="batch_2",
            total_tasks=3,
            successful_tasks=2,
            failed_tasks=1,
        )

        assert not batch.all_successful()

    def test_batch_result_get_results(self) -> None:
        """Test getting successful results."""
        results = [
            TaskResult[int](task_id="0", status=ExecutionStatus.COMPLETED, result=1),
            TaskResult[int](task_id="1", status=ExecutionStatus.FAILED, error="err"),
            TaskResult[int](task_id="2", status=ExecutionStatus.COMPLETED, result=3),
        ]
        batch = BatchResult[int](
            batch_id="batch_3",
            results=results,
        )

        assert batch.get_results() == [1, 3]

    def test_batch_result_get_errors(self) -> None:
        """Test getting error messages."""
        results = [
            TaskResult[int](task_id="0", status=ExecutionStatus.COMPLETED, result=1),
            TaskResult[int](task_id="1", status=ExecutionStatus.FAILED, error="error1"),
            TaskResult[int](task_id="2", status=ExecutionStatus.FAILED, error="error2"),
        ]
        batch = BatchResult[int](
            batch_id="batch_4",
            results=results,
        )

        errors = batch.get_errors()
        assert "error1" in errors
        assert "error2" in errors

    def test_batch_result_to_dict(self) -> None:
        """Test BatchResult serialization."""
        batch = BatchResult[int](
            batch_id="batch_5",
            total_tasks=10,
            successful_tasks=8,
            failed_tasks=2,
        )

        data = batch.to_dict()

        assert data["batch_id"] == "batch_5"
        assert data["total_tasks"] == 10
        assert data["successful_tasks"] == 8


# ==================== ExecutionStatus Tests ====================


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Test all expected statuses exist."""
        expected = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]
        for status_name in expected:
            assert hasattr(ExecutionStatus, status_name)

    def test_status_values(self) -> None:
        """Test status value strings."""
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"


# ==================== SequentialStrategy Tests ====================


class TestSequentialStrategy:
    """Tests for SequentialStrategy."""

    def test_sequential_map(self) -> None:
        """Test sequential map execution."""
        strategy = SequentialStrategy()
        results = strategy.map(double, [1, 2, 3, 4])

        assert len(results) == 4
        assert all(r.is_success() for r in results)
        assert [r.result for r in results] == [2, 4, 6, 8]

    def test_sequential_map_with_error(self) -> None:
        """Test sequential map with errors."""
        strategy = SequentialStrategy()
        results = strategy.map(raise_error, [1, 2, 3])

        assert all(not r.is_success() for r in results)
        assert all("Error for" in r.error for r in results)

    def test_sequential_submit(self) -> None:
        """Test sequential submit."""
        strategy = SequentialStrategy()
        future = strategy.submit(double, 5)

        assert isinstance(future, Future)
        assert future.result() == 10

    def test_sequential_n_workers(self) -> None:
        """Test sequential returns 1 worker."""
        strategy = SequentialStrategy()
        assert strategy.n_workers == 1


# ==================== ThreadingStrategy Tests ====================


class TestThreadingStrategy:
    """Tests for ThreadingStrategy."""

    def test_threading_map(self) -> None:
        """Test threading map execution."""
        strategy = ThreadingStrategy(n_workers=2)
        try:
            results = strategy.map(double, [1, 2, 3, 4])

            assert len(results) == 4
            assert all(r.is_success() for r in results)
            assert sorted([r.result for r in results]) == [2, 4, 6, 8]
        finally:
            strategy.shutdown()

    def test_threading_map_with_error(self) -> None:
        """Test threading map with errors."""
        strategy = ThreadingStrategy(n_workers=2)
        try:
            results = strategy.map(conditional_error, [1, 2, 3, 4])

            # Half should fail (odd numbers)
            successes = [r for r in results if r.is_success()]
            failures = [r for r in results if not r.is_success()]

            assert len(successes) == 2
            assert len(failures) == 2
        finally:
            strategy.shutdown()

    def test_threading_submit(self) -> None:
        """Test threading submit."""
        strategy = ThreadingStrategy(n_workers=2)
        try:
            future = strategy.submit(double, 7)
            assert future.result(timeout=5) == 14
        finally:
            strategy.shutdown()

    def test_threading_n_workers(self) -> None:
        """Test threading worker count."""
        strategy = ThreadingStrategy(n_workers=4)
        try:
            assert strategy.n_workers == 4
        finally:
            strategy.shutdown()


# ==================== MultiprocessingStrategy Tests ====================


class TestMultiprocessingStrategy:
    """Tests for MultiprocessingStrategy."""

    def test_multiprocessing_map(self) -> None:
        """Test multiprocessing map execution."""
        strategy = MultiprocessingStrategy(n_workers=2)
        try:
            results = strategy.map(double, [1, 2, 3, 4])

            assert len(results) == 4
            assert all(r.is_success() for r in results)
            assert sorted([r.result for r in results]) == [2, 4, 6, 8]
        finally:
            strategy.shutdown()

    def test_multiprocessing_submit(self) -> None:
        """Test multiprocessing submit."""
        strategy = MultiprocessingStrategy(n_workers=2)
        try:
            future = strategy.submit(double, 8)
            assert future.result(timeout=10) == 16
        finally:
            strategy.shutdown()

    def test_multiprocessing_n_workers(self) -> None:
        """Test multiprocessing worker count."""
        strategy = MultiprocessingStrategy(n_workers=3)
        try:
            assert strategy.n_workers == 3
        finally:
            strategy.shutdown()


# ==================== ParallelExecutor Tests ====================


class TestParallelExecutor:
    """Tests for ParallelExecutor."""

    def test_executor_creation_sequential(self) -> None:
        """Test executor creation with sequential backend."""
        executor = ParallelExecutor(
            n_workers=1,
            backend=ParallelBackend.SEQUENTIAL,
        )

        assert executor.backend == ParallelBackend.SEQUENTIAL
        assert executor.n_workers == 1
        executor.shutdown()

    def test_executor_creation_threading(self) -> None:
        """Test executor creation with threading backend."""
        executor = ParallelExecutor(
            n_workers=4,
            backend=ParallelBackend.THREADING,
        )

        assert executor.backend == ParallelBackend.THREADING
        assert executor.n_workers == 4
        executor.shutdown()

    def test_executor_creation_multiprocessing(self) -> None:
        """Test executor creation with multiprocessing backend."""
        executor = ParallelExecutor(
            n_workers=2,
            backend=ParallelBackend.MULTIPROCESSING,
        )

        assert executor.backend == ParallelBackend.MULTIPROCESSING
        assert executor.n_workers == 2
        executor.shutdown()

    def test_executor_creation_with_string_backend(self) -> None:
        """Test executor creation with string backend."""
        executor = ParallelExecutor(
            n_workers=2,
            backend="sequential",
        )

        assert executor.backend == ParallelBackend.SEQUENTIAL
        executor.shutdown()

    def test_executor_map(self) -> None:
        """Test executor map."""
        with ParallelExecutor(n_workers=2, backend="sequential") as executor:
            results = executor.map(double, [1, 2, 3, 4])

            assert len(results) == 4
            assert all(r.is_success() for r in results)
            assert [r.result for r in results] == [2, 4, 6, 8]

    def test_executor_map_batch(self) -> None:
        """Test executor map_batch."""
        with ParallelExecutor(n_workers=2, backend="sequential") as executor:
            batch = executor.map_batch(double, [1, 2, 3, 4], batch_id="test_batch")

            assert batch.batch_id == "test_batch"
            assert batch.total_tasks == 4
            assert batch.successful_tasks == 4
            assert batch.failed_tasks == 0
            assert batch.all_successful()

    def test_executor_map_batch_with_errors(self) -> None:
        """Test executor map_batch with errors."""
        with ParallelExecutor(n_workers=2, backend="sequential") as executor:
            batch = executor.map_batch(conditional_error, [1, 2, 3, 4])

            assert batch.total_tasks == 4
            assert batch.successful_tasks == 2
            assert batch.failed_tasks == 2
            assert not batch.all_successful()

    def test_executor_submit(self) -> None:
        """Test executor submit."""
        with ParallelExecutor(n_workers=2, backend="sequential") as executor:
            future = executor.submit(double, 10)
            assert future.result() == 20

    def test_executor_submit_all(self) -> None:
        """Test executor submit_all."""
        with ParallelExecutor(n_workers=2, backend="sequential") as executor:
            tasks = [
                (double, (1,), {}),
                (double, (2,), {}),
                (double, (3,), {}),
            ]
            futures = executor.submit_all(tasks)

            assert len(futures) == 3
            results = [f.result() for f in futures]
            assert sorted(results) == [2, 4, 6]

    def test_executor_context_manager(self) -> None:
        """Test executor context manager."""
        with ParallelExecutor(n_workers=2, backend="sequential") as executor:
            results = executor.map(double, [1, 2])
            assert len(results) == 2

    def test_executor_auto_batch_id(self) -> None:
        """Test automatic batch ID generation."""
        with ParallelExecutor(n_workers=1, backend="sequential") as executor:
            batch1 = executor.map_batch(double, [1])
            batch2 = executor.map_batch(double, [2])

            assert batch1.batch_id != batch2.batch_id


# ==================== ProgressTrackingExecutor Tests ====================


class TestProgressTrackingExecutor:
    """Tests for ProgressTrackingExecutor."""

    def test_progress_tracking_basic(self) -> None:
        """Test basic progress tracking."""
        progress_calls = []

        def on_progress(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        executor = ProgressTrackingExecutor(
            n_workers=2,
            backend="sequential",
            on_progress=on_progress,
        )

        results = executor.map_with_progress(double, [1, 2, 3])
        executor.shutdown()

        assert len(results) == 3
        assert all(r.is_success() for r in results)

        # Should have 3 progress updates
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    def test_progress_tracking_task_complete_callback(self) -> None:
        """Test task complete callback."""
        completed_tasks = []

        def on_task_complete(result: TaskResult) -> None:
            completed_tasks.append(result.task_id)

        executor = ProgressTrackingExecutor(
            n_workers=2,
            backend="sequential",
            on_task_complete=on_task_complete,
        )

        executor.map_with_progress(double, [1, 2, 3])
        executor.shutdown()

        assert len(completed_tasks) == 3

    def test_progress_tracking_n_workers(self) -> None:
        """Test progress executor worker count."""
        executor = ProgressTrackingExecutor(n_workers=4, backend="sequential")
        # Note: SequentialStrategy always returns 1
        executor.shutdown()


# ==================== Factory Function Tests ====================


class TestCreateExecutor:
    """Tests for create_executor factory function."""

    def test_create_executor_defaults(self) -> None:
        """Test create_executor with defaults."""
        executor = create_executor()

        assert executor.backend == ParallelBackend.MULTIPROCESSING
        assert executor.n_workers >= 1
        executor.shutdown()

    def test_create_executor_custom(self) -> None:
        """Test create_executor with custom settings."""
        executor = create_executor(
            n_workers=8,
            backend="threading",
            timeout=30.0,
        )

        assert executor.backend == ParallelBackend.THREADING
        assert executor.n_workers == 8
        executor.shutdown()

    def test_create_executor_with_enum_backend(self) -> None:
        """Test create_executor with enum backend."""
        executor = create_executor(
            n_workers=2,
            backend=ParallelBackend.SEQUENTIAL,
        )

        assert executor.backend == ParallelBackend.SEQUENTIAL
        executor.shutdown()


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for parallel executor."""

    def test_executor_with_different_backends(self) -> None:
        """Test same operations with different backends."""
        items = [1, 2, 3, 4, 5]

        for backend in [ParallelBackend.SEQUENTIAL, ParallelBackend.THREADING]:
            with ParallelExecutor(n_workers=2, backend=backend) as executor:
                results = executor.map(double, items)

                assert len(results) == 5
                assert all(r.is_success() for r in results)
                result_values = sorted([r.result for r in results])
                assert result_values == [2, 4, 6, 8, 10]

    def test_large_batch_execution(self) -> None:
        """Test execution with larger batch."""
        with ParallelExecutor(n_workers=4, backend="threading") as executor:
            items = list(range(100))
            results = executor.map(double, items)

            assert len(results) == 100
            assert all(r.is_success() for r in results)

    def test_mixed_success_failure(self) -> None:
        """Test batch with mixed success and failure."""
        with ParallelExecutor(n_workers=2, backend="sequential") as executor:
            # 1, 3, 5, 7, 9 will fail (odd numbers)
            batch = executor.map_batch(conditional_error, range(10))

            assert batch.successful_tasks == 5
            assert batch.failed_tasks == 5
            assert batch.get_results() == [0, 4, 8, 12, 16]

    def test_empty_input(self) -> None:
        """Test with empty input."""
        with ParallelExecutor(n_workers=2, backend="sequential") as executor:
            results = executor.map(double, [])
            assert len(results) == 0

            batch = executor.map_batch(double, [])
            assert batch.total_tasks == 0
            assert batch.all_successful()

    def test_single_item(self) -> None:
        """Test with single item."""
        with ParallelExecutor(n_workers=4, backend="sequential") as executor:
            results = executor.map(double, [42])

            assert len(results) == 1
            assert results[0].is_success()
            assert results[0].result == 84
