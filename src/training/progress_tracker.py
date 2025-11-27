"""Progress tracking for multi-area training jobs.

This module provides the ProgressTracker class for monitoring training
progress, estimating completion time, and supporting external callbacks
for progress updates.

Example:
    ```python
    from src.training.progress_tracker import ProgressTracker

    # Create tracker
    tracker = ProgressTracker(total_tasks=10, task_description="areas")

    # Define callback for external monitoring
    def on_progress(tracker):
        print(f"Progress: {tracker.get_progress_percentage():.1f}%")
        print(f"ETA: {tracker.get_estimated_time_remaining()}")

    tracker.add_callback(on_progress)

    # Update progress
    for i in range(10):
        # Do work...
        tracker.mark_completed("area_" + str(i))

    # Get summary
    summary = tracker.get_summary()
    print(f"Total time: {summary['elapsed_time']}")
    ```
"""

import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProgressTracker:
    """Tracks progress of multi-task training jobs.

    This class provides progress tracking with:
    - Completed/total task counting
    - Elapsed and estimated remaining time
    - Progress percentage calculation
    - Callback support for external monitoring
    - Task-level metadata storage

    Attributes:
        total_tasks: Total number of tasks to complete.
        task_description: Human-readable description of task units (e.g., "areas").
        completed_tasks: Number of tasks completed so far.
        start_time: When tracking started (Unix timestamp).
        task_completion_times: Dictionary mapping task ID -> completion timestamp.
        callbacks: List of callback functions called on task completion.
    """

    def __init__(
        self,
        total_tasks: int,
        task_description: str = "tasks",
    ) -> None:
        """Initialize progress tracker.

        Args:
            total_tasks: Total number of tasks to complete.
            task_description: Human-readable description of task units.

        Raises:
            ValueError: If total_tasks is not positive.
        """
        if total_tasks < 1:
            msg = f"total_tasks must be >= 1, got {total_tasks}"
            raise ValueError(msg)

        self.total_tasks = total_tasks
        self.task_description = task_description
        self.completed_tasks = 0
        self.start_time = time.time()

        # Track individual task completion times
        self.task_completion_times: dict[str, float] = {}

        # Callbacks for external monitoring
        self.callbacks: list[Callable[[ProgressTracker], None]] = []

        logger.debug(
            "Initialized ProgressTracker: %d %s to complete", total_tasks, task_description
        )

    def mark_completed(self, task_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Mark a task as completed and update progress.

        This method increments the completed task counter, records the
        completion time, and triggers all registered callbacks.

        Args:
            task_id: Unique identifier for the completed task.
            metadata: Optional metadata about the task (e.g., training metrics).

        Example:
            ```python
            tracker.mark_completed(
                "SP",
                metadata={"train_time": 45.3, "mae": 125.4}
            )
            ```
        """
        if task_id in self.task_completion_times:
            logger.warning("Task %s already marked as completed, ignoring", task_id)
            return

        self.completed_tasks += 1
        self.task_completion_times[task_id] = time.time()

        # Store metadata if provided
        if metadata is not None:
            self.task_completion_times[f"{task_id}_metadata"] = metadata  # type: ignore[assignment]

        logger.debug(
            "Task completed: %s (%d/%d %s)",
            task_id,
            self.completed_tasks,
            self.total_tasks,
            self.task_description,
        )

        # Trigger callbacks
        self._trigger_callbacks()

    def get_progress_percentage(self) -> float:
        """Get current progress as percentage.

        Returns:
            Progress percentage (0.0 to 100.0).

        Example:
            ```python
            progress = tracker.get_progress_percentage()
            print(f"Progress: {progress:.1f}%")
            ```
        """
        return (self.completed_tasks / self.total_tasks) * 100.0

    def get_elapsed_time(self) -> timedelta:
        """Get elapsed time since tracking started.

        Returns:
            Elapsed time as timedelta.

        Example:
            ```python
            elapsed = tracker.get_elapsed_time()
            print(f"Elapsed: {elapsed}")
            ```
        """
        elapsed_seconds = time.time() - self.start_time
        return timedelta(seconds=elapsed_seconds)

    def get_estimated_time_remaining(self) -> timedelta | None:
        """Estimate time remaining until completion.

        This uses the average time per completed task to estimate
        the remaining time. Returns None if no tasks completed yet.

        Returns:
            Estimated time remaining as timedelta, or None if no data.

        Example:
            ```python
            eta = tracker.get_estimated_time_remaining()
            if eta:
                print(f"ETA: {eta}")
            else:
                print("ETA: calculating...")
            ```
        """
        if self.completed_tasks == 0:
            return None

        elapsed_seconds = time.time() - self.start_time
        avg_time_per_task = elapsed_seconds / self.completed_tasks
        remaining_tasks = self.total_tasks - self.completed_tasks
        remaining_seconds = avg_time_per_task * remaining_tasks

        return timedelta(seconds=remaining_seconds)

    def get_estimated_completion_time(self) -> datetime | None:
        """Estimate when all tasks will be completed.

        Returns:
            Estimated completion datetime, or None if no data.

        Example:
            ```python
            completion_time = tracker.get_estimated_completion_time()
            if completion_time:
                print(f"Expected completion: {completion_time}")
            ```
        """
        eta = self.get_estimated_time_remaining()
        if eta is None:
            return None

        return datetime.now() + eta

    def get_average_task_time(self) -> timedelta | None:
        """Get average time per completed task.

        Returns:
            Average time per task as timedelta, or None if no tasks completed.

        Example:
            ```python
            avg_time = tracker.get_average_task_time()
            if avg_time:
                print(f"Average time per area: {avg_time}")
            ```
        """
        if self.completed_tasks == 0:
            return None

        elapsed_seconds = time.time() - self.start_time
        avg_seconds = elapsed_seconds / self.completed_tasks

        return timedelta(seconds=avg_seconds)

    def is_completed(self) -> bool:
        """Check if all tasks are completed.

        Returns:
            True if all tasks completed, False otherwise.

        Example:
            ```python
            if tracker.is_completed():
                print("All training completed!")
            ```
        """
        return self.completed_tasks >= self.total_tasks

    def get_remaining_tasks(self) -> int:
        """Get number of tasks remaining.

        Returns:
            Number of tasks not yet completed.

        Example:
            ```python
            remaining = tracker.get_remaining_tasks()
            print(f"{remaining} areas left to train")
            ```
        """
        return max(0, self.total_tasks - self.completed_tasks)

    def add_callback(self, callback: Callable[["ProgressTracker"], None]) -> None:
        """Add a callback function to be called on task completion.

        Callbacks receive the ProgressTracker instance as an argument
        and can query progress, time estimates, etc.

        Args:
            callback: Function to call on each task completion.
                     Signature: callback(tracker: ProgressTracker) -> None

        Example:
            ```python
            def log_progress(tracker):
                print(f"Progress: {tracker.get_progress_percentage():.1f}%")

            tracker.add_callback(log_progress)
            ```
        """
        self.callbacks.append(callback)
        logger.debug("Added progress callback: %s", callback.__name__)

    def remove_callback(self, callback: Callable[["ProgressTracker"], None]) -> None:
        """Remove a previously registered callback.

        Args:
            callback: Callback function to remove.

        Example:
            ```python
            tracker.remove_callback(log_progress)
            ```
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug("Removed progress callback: %s", callback.__name__)

    def _trigger_callbacks(self) -> None:
        """Trigger all registered callbacks.

        This is called internally after each task completion.
        """
        for callback in self.callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(
                    "Error in progress callback %s: %s", callback.__name__, e, exc_info=True
                )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of progress tracking.

        Returns:
            Dictionary with progress statistics including:
            - total_tasks: Total number of tasks
            - completed_tasks: Number completed
            - progress_percentage: Completion percentage
            - elapsed_time: Time since start
            - estimated_time_remaining: Estimated time left
            - average_task_time: Average time per task
            - is_completed: Whether all tasks done

        Example:
            ```python
            summary = tracker.get_summary()
            print(f"Progress: {summary['progress_percentage']:.1f}%")
            print(f"Elapsed: {summary['elapsed_time']}")
            print(f"ETA: {summary['estimated_time_remaining']}")
            ```
        """
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "remaining_tasks": self.get_remaining_tasks(),
            "progress_percentage": self.get_progress_percentage(),
            "elapsed_time": self.get_elapsed_time(),
            "estimated_time_remaining": self.get_estimated_time_remaining(),
            "average_task_time": self.get_average_task_time(),
            "estimated_completion_time": self.get_estimated_completion_time(),
            "is_completed": self.is_completed(),
            "task_description": self.task_description,
        }

    def get_task_metadata(self, task_id: str) -> dict[str, Any] | None:
        """Get metadata for a specific task.

        Args:
            task_id: Task identifier.

        Returns:
            Task metadata dictionary if found, None otherwise.

        Example:
            ```python
            metadata = tracker.get_task_metadata("SP")
            if metadata:
                print(f"Training time: {metadata['train_time']}")
            ```
        """
        metadata_key = f"{task_id}_metadata"
        return self.task_completion_times.get(metadata_key)  # type: ignore[return-value]

    def reset(self) -> None:
        """Reset tracker to initial state.

        This clears all completed tasks and resets the start time.
        Useful for restarting tracking.

        Example:
            ```python
            # Reset for a new training run
            tracker.reset()
            ```
        """
        self.completed_tasks = 0
        self.start_time = time.time()
        self.task_completion_times.clear()

        logger.debug("Reset ProgressTracker")

    def __repr__(self) -> str:
        """Return string representation of progress tracker.

        Returns:
            String with completion status and progress percentage.
        """
        return (
            f"ProgressTracker({self.completed_tasks}/{self.total_tasks} "
            f"{self.task_description}, {self.get_progress_percentage():.1f}%)"
        )

    def __str__(self) -> str:
        """Return human-readable string representation.

        Returns:
            String with progress information.
        """
        elapsed = self.get_elapsed_time()
        eta = self.get_estimated_time_remaining()

        eta_str = str(eta) if eta else "calculating..."

        return (
            f"Progress: {self.completed_tasks}/{self.total_tasks} "
            f"{self.task_description} ({self.get_progress_percentage():.1f}%) | "
            f"Elapsed: {elapsed} | ETA: {eta_str}"
        )
