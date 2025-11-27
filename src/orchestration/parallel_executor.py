"""Parallel executor for PrevCarga workflow orchestration.

This module provides a unified parallel execution framework supporting
multiple backends (multiprocessing, threading, sequential) for scalable
task execution in forecasting workflows.

Key Components:
- ParallelExecutor: Main executor class with configurable backends
- TaskResult: Result container for parallel tasks
- TaskBatch: Batch of tasks for parallel execution
- ExecutionStrategy: Strategy pattern for different execution modes

Example:
    ```python
    from src.orchestration.parallel_executor import ParallelExecutor, TaskBatch

    # Create executor with multiprocessing backend
    executor = ParallelExecutor(n_workers=4, backend="multiprocessing")

    # Define tasks
    def train_model(area: str) -> dict:
        return {"area": area, "status": "trained"}

    tasks = [
        {"func": train_model, "args": ("SECO",)},
        {"func": train_model, "args": ("S",)},
        {"func": train_model, "args": ("NE",)},
        {"func": train_model, "args": ("N",)},
    ]

    # Execute in parallel
    results = executor.map(train_model, ["SECO", "S", "NE", "N"])
    ```
"""

import multiprocessing as mp
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar

from src.orchestration.config_manager import ParallelBackend
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ExecutionStatus(Enum):
    """Status of task execution.

    Attributes:
        PENDING: Task is pending execution.
        RUNNING: Task is currently running.
        COMPLETED: Task completed successfully.
        FAILED: Task failed with an error.
        CANCELLED: Task was cancelled.
        TIMEOUT: Task timed out.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskResult(Generic[T]):
    """Result container for a parallel task.

    Attributes:
        task_id: Unique task identifier.
        status: Execution status.
        result: Task result if successful.
        error: Error message if failed.
        started_at: When task started.
        completed_at: When task completed.
        duration_seconds: Execution duration.
        worker_id: ID of worker that executed task.
    """

    task_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: T | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    worker_id: str | None = None

    def is_success(self) -> bool:
        """Check if task completed successfully.

        Returns:
            True if task completed successfully.
        """
        return self.status == ExecutionStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "worker_id": self.worker_id,
        }


@dataclass
class BatchResult(Generic[T]):
    """Result container for a batch of parallel tasks.

    Attributes:
        batch_id: Unique batch identifier.
        results: Individual task results.
        total_tasks: Total number of tasks.
        successful_tasks: Number of successful tasks.
        failed_tasks: Number of failed tasks.
        started_at: When batch started.
        completed_at: When batch completed.
        duration_seconds: Total execution duration.
    """

    batch_id: str
    results: list[TaskResult[T]] = field(default_factory=list)
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0

    def all_successful(self) -> bool:
        """Check if all tasks completed successfully.

        Returns:
            True if all tasks succeeded.
        """
        return self.failed_tasks == 0

    def get_results(self) -> list[T]:
        """Get list of successful results.

        Returns:
            List of results from successful tasks.
        """
        return [r.result for r in self.results if r.is_success() and r.result is not None]

    def get_errors(self) -> list[str]:
        """Get list of error messages.

        Returns:
            List of error messages from failed tasks.
        """
        return [r.error for r in self.results if r.error is not None]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "batch_id": self.batch_id,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "results": [r.to_dict() for r in self.results],
        }


class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies.

    Defines the interface for different parallel execution backends.
    """

    @abstractmethod
    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        timeout: float | None = None,
    ) -> list[TaskResult[R]]:
        """Map a function over items in parallel.

        Args:
            func: Function to apply.
            items: Items to process.
            timeout: Optional timeout per task in seconds.

        Returns:
            List of task results.
        """
        pass

    @abstractmethod
    def submit(
        self,
        func: Callable[..., R],
        *args: Any,
        **kwargs: Any,
    ) -> Future[R]:
        """Submit a single task for execution.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Future representing the pending result.
        """
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete.
        """
        pass

    @property
    @abstractmethod
    def n_workers(self) -> int:
        """Get number of workers."""
        pass


class SequentialStrategy(ExecutionStrategy):
    """Sequential (single-threaded) execution strategy.

    Useful for debugging and testing.
    """

    def __init__(self) -> None:
        """Initialize sequential strategy."""
        self._shutdown = False

    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        timeout: float | None = None,
    ) -> list[TaskResult[R]]:
        """Execute function sequentially over items.

        Args:
            func: Function to apply.
            items: Items to process.
            timeout: Optional timeout (not enforced in sequential mode).

        Returns:
            List of task results.
        """
        results = []
        for i, item in enumerate(items):
            task_id = f"task_{i}"
            result = TaskResult[R](task_id=task_id)
            result.started_at = datetime.now()

            try:
                result.result = func(item)
                result.status = ExecutionStatus.COMPLETED
            except Exception as e:
                result.error = str(e)
                result.status = ExecutionStatus.FAILED

            result.completed_at = datetime.now()
            result.duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()
            result.worker_id = "main"

            results.append(result)

        return results

    def submit(
        self,
        func: Callable[..., R],
        *args: Any,
        **kwargs: Any,
    ) -> Future[R]:
        """Submit a task for sequential execution.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Completed future.
        """
        future: Future[R] = Future()
        try:
            result = func(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown (no-op for sequential)."""
        self._shutdown = True

    @property
    def n_workers(self) -> int:
        """Return 1 for sequential execution."""
        return 1


class ThreadingStrategy(ExecutionStrategy):
    """Thread-based parallel execution strategy.

    Good for I/O-bound tasks.
    """

    def __init__(self, n_workers: int = 4) -> None:
        """Initialize threading strategy.

        Args:
            n_workers: Number of worker threads.
        """
        self._n_workers = n_workers
        self._executor = ThreadPoolExecutor(max_workers=n_workers)

    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        timeout: float | None = None,
    ) -> list[TaskResult[R]]:
        """Execute function using thread pool.

        Args:
            func: Function to apply.
            items: Items to process.
            timeout: Optional timeout per task in seconds.

        Returns:
            List of task results.
        """
        items_list = list(items)
        results = []
        futures: dict[Future[R], tuple[int, T]] = {}

        # Submit all tasks
        for i, item in enumerate(items_list):
            future = self._executor.submit(func, item)
            futures[future] = (i, item)

        # Collect results
        for future in as_completed(futures, timeout=timeout):
            i, item = futures[future]
            task_id = f"task_{i}"
            result = TaskResult[R](task_id=task_id)
            result.started_at = datetime.now()

            try:
                result.result = future.result(timeout=0)
                result.status = ExecutionStatus.COMPLETED
            except Exception as e:
                result.error = str(e)
                result.status = ExecutionStatus.FAILED

            result.completed_at = datetime.now()
            results.append(result)

        # Sort by task_id to maintain order
        results.sort(key=lambda r: int(r.task_id.split("_")[1]))
        return results

    def submit(
        self,
        func: Callable[..., R],
        *args: Any,
        **kwargs: Any,
    ) -> Future[R]:
        """Submit a task to thread pool.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Future representing pending result.
        """
        return self._executor.submit(func, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown thread pool.

        Args:
            wait: Whether to wait for pending tasks.
        """
        self._executor.shutdown(wait=wait)

    @property
    def n_workers(self) -> int:
        """Return number of worker threads."""
        return self._n_workers


class MultiprocessingStrategy(ExecutionStrategy):
    """Process-based parallel execution strategy.

    Good for CPU-bound tasks.
    """

    def __init__(self, n_workers: int = 4) -> None:
        """Initialize multiprocessing strategy.

        Args:
            n_workers: Number of worker processes.
        """
        self._n_workers = n_workers
        # Use spawn context for better compatibility
        ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
        )

    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        timeout: float | None = None,
    ) -> list[TaskResult[R]]:
        """Execute function using process pool.

        Args:
            func: Function to apply.
            items: Items to process.
            timeout: Optional timeout per task in seconds.

        Returns:
            List of task results.
        """
        items_list = list(items)
        results = []
        futures: dict[Future[R], tuple[int, T]] = {}

        # Submit all tasks
        for i, item in enumerate(items_list):
            future = self._executor.submit(func, item)
            futures[future] = (i, item)

        # Collect results
        for future in as_completed(futures, timeout=timeout):
            i, item = futures[future]
            task_id = f"task_{i}"
            result = TaskResult[R](task_id=task_id)
            result.started_at = datetime.now()
            result.worker_id = f"process_{os.getpid()}"

            try:
                result.result = future.result(timeout=0)
                result.status = ExecutionStatus.COMPLETED
            except Exception as e:
                result.error = str(e)
                result.status = ExecutionStatus.FAILED

            result.completed_at = datetime.now()
            results.append(result)

        # Sort by task_id to maintain order
        results.sort(key=lambda r: int(r.task_id.split("_")[1]))
        return results

    def submit(
        self,
        func: Callable[..., R],
        *args: Any,
        **kwargs: Any,
    ) -> Future[R]:
        """Submit a task to process pool.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Future representing pending result.
        """
        return self._executor.submit(func, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown process pool.

        Args:
            wait: Whether to wait for pending tasks.
        """
        self._executor.shutdown(wait=wait)

    @property
    def n_workers(self) -> int:
        """Return number of worker processes."""
        return self._n_workers


class ParallelExecutor:
    """Unified parallel executor with configurable backends.

    Provides a consistent interface for parallel execution across
    different backends (multiprocessing, threading, sequential).

    Example:
        >>> executor = ParallelExecutor(n_workers=4, backend="multiprocessing")
        >>> results = executor.map(lambda x: x * 2, [1, 2, 3, 4])
        >>> print([r.result for r in results if r.is_success()])
        [2, 4, 6, 8]
    """

    def __init__(
        self,
        n_workers: int = 4,
        backend: str | ParallelBackend = ParallelBackend.MULTIPROCESSING,
        timeout: float | None = None,
    ) -> None:
        """Initialize parallel executor.

        Args:
            n_workers: Number of parallel workers.
            backend: Execution backend (multiprocessing, threading, sequential).
            timeout: Default timeout per task in seconds.
        """
        self._n_workers = n_workers
        self._timeout = timeout
        self._backend = backend if isinstance(backend, ParallelBackend) else ParallelBackend(backend)
        self._strategy = self._create_strategy()
        self._batch_counter = 0

        logger.info(
            "Initialized ParallelExecutor with backend=%s, workers=%d",
            self._backend.value,
            self._n_workers,
        )

    def _create_strategy(self) -> ExecutionStrategy:
        """Create execution strategy based on backend.

        Returns:
            ExecutionStrategy instance.
        """
        if self._backend == ParallelBackend.SEQUENTIAL:
            return SequentialStrategy()
        elif self._backend == ParallelBackend.THREADING:
            return ThreadingStrategy(self._n_workers)
        elif self._backend == ParallelBackend.MULTIPROCESSING:
            return MultiprocessingStrategy(self._n_workers)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    @property
    def n_workers(self) -> int:
        """Get number of workers."""
        return self._strategy.n_workers

    @property
    def backend(self) -> ParallelBackend:
        """Get execution backend."""
        return self._backend

    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        timeout: float | None = None,
    ) -> list[TaskResult[R]]:
        """Map a function over items in parallel.

        Args:
            func: Function to apply to each item.
            items: Items to process.
            timeout: Optional timeout per task in seconds.

        Returns:
            List of TaskResult objects.
        """
        timeout = timeout or self._timeout
        return self._strategy.map(func, items, timeout=timeout)

    def map_batch(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        batch_id: str | None = None,
        timeout: float | None = None,
    ) -> BatchResult[R]:
        """Map a function over items and return batch result.

        Args:
            func: Function to apply to each item.
            items: Items to process.
            batch_id: Optional batch identifier.
            timeout: Optional timeout per task in seconds.

        Returns:
            BatchResult containing all task results.
        """
        if batch_id is None:
            self._batch_counter += 1
            batch_id = f"batch_{self._batch_counter}"

        batch = BatchResult[R](batch_id=batch_id)
        batch.started_at = datetime.now()

        items_list = list(items)
        batch.total_tasks = len(items_list)

        batch.results = self.map(func, items_list, timeout=timeout)

        batch.successful_tasks = sum(1 for r in batch.results if r.is_success())
        batch.failed_tasks = batch.total_tasks - batch.successful_tasks

        batch.completed_at = datetime.now()
        batch.duration_seconds = (
            batch.completed_at - batch.started_at
        ).total_seconds()

        logger.info(
            "Batch %s completed: %d/%d successful (%.2fs)",
            batch_id,
            batch.successful_tasks,
            batch.total_tasks,
            batch.duration_seconds,
        )

        return batch

    def submit(
        self,
        func: Callable[..., R],
        *args: Any,
        **kwargs: Any,
    ) -> Future[R]:
        """Submit a single task for execution.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Future representing the pending result.
        """
        return self._strategy.submit(func, *args, **kwargs)

    def submit_all(
        self,
        tasks: list[tuple[Callable[..., R], tuple[Any, ...], dict[str, Any]]],
    ) -> list[Future[R]]:
        """Submit multiple tasks for execution.

        Args:
            tasks: List of (func, args, kwargs) tuples.

        Returns:
            List of futures.
        """
        futures = []
        for func, args, kwargs in tasks:
            future = self.submit(func, *args, **kwargs)
            futures.append(future)
        return futures

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete.
        """
        self._strategy.shutdown(wait=wait)
        logger.info("ParallelExecutor shutdown complete")

    def __enter__(self) -> "ParallelExecutor":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.shutdown(wait=True)


class ProgressTrackingExecutor:
    """Executor with progress tracking capabilities.

    Extends ParallelExecutor with progress callbacks and status reporting.

    Example:
        >>> def on_progress(completed, total):
        ...     print(f"Progress: {completed}/{total}")
        >>> executor = ProgressTrackingExecutor(
        ...     n_workers=4,
        ...     on_progress=on_progress,
        ... )
        >>> results = executor.map_with_progress(func, items)
    """

    def __init__(
        self,
        n_workers: int = 4,
        backend: str | ParallelBackend = ParallelBackend.MULTIPROCESSING,
        on_progress: Callable[[int, int], None] | None = None,
        on_task_complete: Callable[[TaskResult[Any]], None] | None = None,
    ) -> None:
        """Initialize progress tracking executor.

        Args:
            n_workers: Number of parallel workers.
            backend: Execution backend.
            on_progress: Callback for progress updates (completed, total).
            on_task_complete: Callback when each task completes.
        """
        self._executor = ParallelExecutor(n_workers=n_workers, backend=backend)
        self._on_progress = on_progress
        self._on_task_complete = on_task_complete

    def map_with_progress(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        timeout: float | None = None,
    ) -> list[TaskResult[R]]:
        """Map function with progress tracking.

        Args:
            func: Function to apply.
            items: Items to process.
            timeout: Optional timeout per task.

        Returns:
            List of task results.
        """
        items_list = list(items)
        total = len(items_list)
        completed = 0

        # Use sequential tracking wrapper
        results = []
        for i, item in enumerate(items_list):
            task_id = f"task_{i}"
            result = TaskResult[R](task_id=task_id)
            result.started_at = datetime.now()

            try:
                result.result = func(item)
                result.status = ExecutionStatus.COMPLETED
            except Exception as e:
                result.error = str(e)
                result.status = ExecutionStatus.FAILED

            result.completed_at = datetime.now()
            result.duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()

            results.append(result)
            completed += 1

            if self._on_task_complete:
                self._on_task_complete(result)

            if self._on_progress:
                self._on_progress(completed, total)

        return results

    @property
    def n_workers(self) -> int:
        """Get number of workers."""
        return self._executor.n_workers

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=wait)


def create_executor(
    n_workers: int | None = None,
    backend: str | ParallelBackend | None = None,
    timeout: float | None = None,
) -> ParallelExecutor:
    """Factory function to create a parallel executor.

    Args:
        n_workers: Number of workers. Defaults to CPU count.
        backend: Execution backend. Defaults to multiprocessing.
        timeout: Default task timeout.

    Returns:
        Configured ParallelExecutor instance.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    if backend is None:
        backend = ParallelBackend.MULTIPROCESSING

    return ParallelExecutor(
        n_workers=n_workers,
        backend=backend,
        timeout=timeout,
    )
