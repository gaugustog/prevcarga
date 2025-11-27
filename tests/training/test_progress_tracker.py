"""Tests for ProgressTracker."""

import time
from datetime import timedelta

import pytest

from src.training.progress_tracker import ProgressTracker


class TestProgressTracker:
    """Test suite for ProgressTracker."""

    def test_initialization(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker(total_tasks=10, task_description="areas")

        assert tracker.total_tasks == 10
        assert tracker.task_description == "areas"
        assert tracker.completed_tasks == 0
        assert len(tracker.task_completion_times) == 0

    def test_initialization_invalid_total(self):
        """Test that invalid total_tasks raises error."""
        with pytest.raises(ValueError) as exc_info:
            ProgressTracker(total_tasks=0)

        assert "total_tasks must be >= 1" in str(exc_info.value)

    def test_mark_completed(self):
        """Test marking a task as completed."""
        tracker = ProgressTracker(total_tasks=5)

        tracker.mark_completed("task_1")

        assert tracker.completed_tasks == 1
        assert "task_1" in tracker.task_completion_times

    def test_mark_completed_with_metadata(self):
        """Test marking task completed with metadata."""
        tracker = ProgressTracker(total_tasks=5)

        metadata = {"training_time": 45.3, "mae": 125.4}
        tracker.mark_completed("task_1", metadata=metadata)

        assert tracker.completed_tasks == 1
        retrieved_metadata = tracker.get_task_metadata("task_1")
        assert retrieved_metadata == metadata

    def test_mark_completed_duplicate(self):
        """Test marking same task completed twice (should warn, not error)."""
        tracker = ProgressTracker(total_tasks=5)

        tracker.mark_completed("task_1")
        tracker.mark_completed("task_1")  # Duplicate

        # Should still be 1, not 2
        assert tracker.completed_tasks == 1

    def test_get_progress_percentage(self):
        """Test progress percentage calculation."""
        tracker = ProgressTracker(total_tasks=10)

        assert tracker.get_progress_percentage() == 0.0

        tracker.mark_completed("task_1")
        assert tracker.get_progress_percentage() == 10.0

        tracker.mark_completed("task_2")
        assert tracker.get_progress_percentage() == 20.0

        for i in range(3, 11):
            tracker.mark_completed(f"task_{i}")

        assert tracker.get_progress_percentage() == 100.0

    def test_get_elapsed_time(self):
        """Test elapsed time tracking."""
        tracker = ProgressTracker(total_tasks=5)

        # Small delay
        time.sleep(0.1)

        elapsed = tracker.get_elapsed_time()
        assert isinstance(elapsed, timedelta)
        assert elapsed.total_seconds() >= 0.1

    def test_get_estimated_time_remaining_no_data(self):
        """Test ETA returns None when no tasks completed."""
        tracker = ProgressTracker(total_tasks=5)

        eta = tracker.get_estimated_time_remaining()
        assert eta is None

    def test_get_estimated_time_remaining_with_data(self):
        """Test ETA calculation with completed tasks."""
        tracker = ProgressTracker(total_tasks=4)

        # Complete 2 tasks with small delay
        tracker.mark_completed("task_1")
        time.sleep(0.1)
        tracker.mark_completed("task_2")

        eta = tracker.get_estimated_time_remaining()
        assert eta is not None
        assert isinstance(eta, timedelta)
        # Should estimate roughly 0.1 seconds for remaining 2 tasks
        assert eta.total_seconds() > 0

    def test_get_average_task_time(self):
        """Test average task time calculation."""
        tracker = ProgressTracker(total_tasks=5)

        avg = tracker.get_average_task_time()
        assert avg is None  # No tasks completed

        tracker.mark_completed("task_1")
        time.sleep(0.1)
        tracker.mark_completed("task_2")

        avg = tracker.get_average_task_time()
        assert avg is not None
        assert isinstance(avg, timedelta)
        assert avg.total_seconds() > 0

    def test_get_estimated_completion_time(self):
        """Test estimated completion time calculation."""
        tracker = ProgressTracker(total_tasks=4)

        completion = tracker.get_estimated_completion_time()
        assert completion is None  # No data yet

        tracker.mark_completed("task_1")
        time.sleep(0.05)
        tracker.mark_completed("task_2")

        completion = tracker.get_estimated_completion_time()
        assert completion is not None
        # Should be in the future
        from datetime import datetime

        assert completion > datetime.now()

    def test_is_completed(self):
        """Test completion status checking."""
        tracker = ProgressTracker(total_tasks=3)

        assert not tracker.is_completed()

        tracker.mark_completed("task_1")
        assert not tracker.is_completed()

        tracker.mark_completed("task_2")
        assert not tracker.is_completed()

        tracker.mark_completed("task_3")
        assert tracker.is_completed()

    def test_get_remaining_tasks(self):
        """Test remaining tasks calculation."""
        tracker = ProgressTracker(total_tasks=5)

        assert tracker.get_remaining_tasks() == 5

        tracker.mark_completed("task_1")
        assert tracker.get_remaining_tasks() == 4

        tracker.mark_completed("task_2")
        tracker.mark_completed("task_3")
        assert tracker.get_remaining_tasks() == 2

        tracker.mark_completed("task_4")
        tracker.mark_completed("task_5")
        assert tracker.get_remaining_tasks() == 0

    def test_add_callback(self):
        """Test adding progress callback."""
        tracker = ProgressTracker(total_tasks=3)

        callback_calls = []

        def test_callback(t):
            callback_calls.append(t.completed_tasks)

        tracker.add_callback(test_callback)
        assert len(tracker.callbacks) == 1

        tracker.mark_completed("task_1")
        assert len(callback_calls) == 1
        assert callback_calls[0] == 1

        tracker.mark_completed("task_2")
        assert len(callback_calls) == 2
        assert callback_calls[1] == 2

    def test_remove_callback(self):
        """Test removing progress callback."""
        tracker = ProgressTracker(total_tasks=3)

        callback_calls = []

        def test_callback(t):
            callback_calls.append(t.completed_tasks)

        tracker.add_callback(test_callback)
        tracker.mark_completed("task_1")
        assert len(callback_calls) == 1

        tracker.remove_callback(test_callback)
        tracker.mark_completed("task_2")
        # Callback should not be called after removal
        assert len(callback_calls) == 1

    def test_callback_exception_handling(self):
        """Test that exceptions in callbacks don't break tracking."""
        tracker = ProgressTracker(total_tasks=3)

        def bad_callback(t):
            raise RuntimeError("Callback error")

        tracker.add_callback(bad_callback)

        # Should not raise exception
        tracker.mark_completed("task_1")

        # Tracker should still work
        assert tracker.completed_tasks == 1

    def test_get_summary(self):
        """Test getting progress summary."""
        tracker = ProgressTracker(total_tasks=5, task_description="models")

        summary = tracker.get_summary()

        assert summary["total_tasks"] == 5
        assert summary["completed_tasks"] == 0
        assert summary["remaining_tasks"] == 5
        assert summary["progress_percentage"] == 0.0
        assert summary["is_completed"] is False
        assert summary["task_description"] == "models"
        assert summary["estimated_time_remaining"] is None

        # Complete some tasks
        tracker.mark_completed("task_1")
        tracker.mark_completed("task_2")

        summary = tracker.get_summary()
        assert summary["completed_tasks"] == 2
        assert summary["remaining_tasks"] == 3
        assert summary["progress_percentage"] == 40.0
        assert summary["estimated_time_remaining"] is not None

    def test_get_task_metadata(self):
        """Test retrieving task metadata."""
        tracker = ProgressTracker(total_tasks=5)

        metadata = {"time": 30.5, "metric": 0.95}
        tracker.mark_completed("task_1", metadata=metadata)

        retrieved = tracker.get_task_metadata("task_1")
        assert retrieved == metadata

        # Non-existent task
        assert tracker.get_task_metadata("nonexistent") is None

    def test_reset(self):
        """Test resetting tracker."""
        tracker = ProgressTracker(total_tasks=5)

        tracker.mark_completed("task_1")
        tracker.mark_completed("task_2")

        assert tracker.completed_tasks == 2
        assert len(tracker.task_completion_times) == 2

        tracker.reset()

        assert tracker.completed_tasks == 0
        assert len(tracker.task_completion_times) == 0
        assert tracker.get_progress_percentage() == 0.0

    def test_repr(self):
        """Test string representation."""
        tracker = ProgressTracker(total_tasks=10, task_description="areas")

        repr_str = repr(tracker)
        assert "ProgressTracker" in repr_str
        assert "0/10" in repr_str
        assert "areas" in repr_str
        assert "0.0%" in repr_str

        tracker.mark_completed("area_1")
        repr_str = repr(tracker)
        assert "1/10" in repr_str

    def test_str(self):
        """Test human-readable string."""
        tracker = ProgressTracker(total_tasks=5, task_description="models")

        str_repr = str(tracker)
        assert "0/5 models" in str_repr
        assert "0.0%" in str_repr
        assert "Elapsed:" in str_repr
        assert "ETA:" in str_repr

    def test_multiple_callbacks(self):
        """Test multiple callbacks working together."""
        tracker = ProgressTracker(total_tasks=3)

        calls_1 = []
        calls_2 = []

        def callback_1(t):
            calls_1.append(t.completed_tasks)

        def callback_2(t):
            calls_2.append(t.get_progress_percentage())

        tracker.add_callback(callback_1)
        tracker.add_callback(callback_2)

        tracker.mark_completed("task_1")

        assert calls_1 == [1]
        assert len(calls_2) == 1
        assert calls_2[0] > 0

    def test_progress_percentage_precision(self):
        """Test progress percentage calculation precision."""
        tracker = ProgressTracker(total_tasks=3)

        tracker.mark_completed("task_1")
        # 1/3 = 33.333...%
        percentage = tracker.get_progress_percentage()
        assert abs(percentage - 33.333333333333336) < 0.001

        tracker.mark_completed("task_2")
        # 2/3 = 66.666...%
        percentage = tracker.get_progress_percentage()
        assert abs(percentage - 66.66666666666667) < 0.001
