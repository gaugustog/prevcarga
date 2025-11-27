"""Tests for CLI progress monitoring system.

This module contains comprehensive tests for progress bar classes
and utility functions.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.cli.progress import (
    BacktestProgressBar,
    BaseProgressBar,
    EvaluationProgressBar,
    FeatureProgressBar,
    MultiLevelProgressBar,
    PredictionProgressBar,
    ProgressBarManager,
    TrainingProgressBar,
    create_simple_progress,
    display_progress_summary,
    format_time,
    format_time_detailed,
)


class TestProgressBarManager:
    """Tests for ProgressBarManager class."""

    def test_initialization(self) -> None:
        """Test manager initialization."""
        manager = ProgressBarManager()
        assert manager.progress_bars == {}
        assert manager.active_bar is None

    def test_register_bar(self) -> None:
        """Test registering a progress bar."""
        manager = ProgressBarManager()
        mock_bar = MagicMock()

        manager.register_bar("test_bar", mock_bar)

        assert "test_bar" in manager.progress_bars
        assert manager.active_bar == "test_bar"

    def test_unregister_bar(self) -> None:
        """Test unregistering a progress bar."""
        manager = ProgressBarManager()
        mock_bar = MagicMock()
        manager.register_bar("test_bar", mock_bar)

        manager.unregister_bar("test_bar")

        assert "test_bar" not in manager.progress_bars
        assert manager.active_bar is None

    def test_unregister_nonexistent_bar(self) -> None:
        """Test unregistering nonexistent bar doesn't raise."""
        manager = ProgressBarManager()
        # Should not raise
        manager.unregister_bar("nonexistent")

    def test_get_unique_id(self) -> None:
        """Test generating unique IDs."""
        manager = ProgressBarManager()

        id1 = manager.get_unique_id()
        id2 = manager.get_unique_id()

        assert id1 != id2
        assert id1.startswith("bar_")
        assert id2.startswith("bar_")

    @patch("click.progressbar")
    def test_create_progress_bar(self, mock_progressbar: MagicMock) -> None:
        """Test creating a progress bar."""
        manager = ProgressBarManager()

        manager.create_progress_bar("Test", 100)

        mock_progressbar.assert_called_once()
        call_kwargs = mock_progressbar.call_args.kwargs
        assert call_kwargs["length"] == 100
        assert call_kwargs["label"] == "Test"


class TestBaseProgressBar:
    """Tests for BaseProgressBar class."""

    def test_initialization(self) -> None:
        """Test base progress bar initialization."""
        bar = BaseProgressBar(100, "Testing")

        assert bar.total_tasks == 100
        assert bar.label == "Testing"
        assert bar.current_task == 0
        assert bar.start_time is None

    def test_progress_percent_empty(self) -> None:
        """Test progress percent with no tasks."""
        bar = BaseProgressBar(0)
        assert bar.progress_percent == 100.0

    def test_progress_percent_partial(self) -> None:
        """Test progress percent with partial completion."""
        bar = BaseProgressBar(100)
        bar.current_task = 25
        assert bar.progress_percent == 25.0

    def test_elapsed_time_not_started(self) -> None:
        """Test elapsed time before starting."""
        bar = BaseProgressBar(100)
        assert bar.elapsed_time == 0.0

    def test_average_task_time_empty(self) -> None:
        """Test average task time with no tasks."""
        bar = BaseProgressBar(100)
        assert bar.average_task_time == 0.0

    def test_estimated_remaining(self) -> None:
        """Test estimated remaining time."""
        bar = BaseProgressBar(100)
        bar.current_task = 50
        bar.task_times = [1.0, 1.0, 1.0]  # Average 1 second per task
        # 50 remaining * 1 second average = 50 seconds
        assert bar.estimated_remaining == 50.0


class TestTrainingProgressBar:
    """Tests for TrainingProgressBar class."""

    def test_initialization(self) -> None:
        """Test training progress bar initialization."""
        bar = TrainingProgressBar(["lgbm", "rf"], ["SE", "S"])

        assert bar.models == ["lgbm", "rf"]
        assert bar.areas == ["SE", "S"]
        assert bar.total_tasks == 4
        assert bar.current_task == 0

    def test_update_increments_task(self) -> None:
        """Test update increments current task."""
        bar = TrainingProgressBar(["lgbm"], ["SE"], show_progress=False)
        bar.start_time = time.time()

        bar.update("lgbm", "SE", "completed")

        assert bar.current_task == 1

    def test_update_stores_info(self) -> None:
        """Test update stores current info."""
        bar = TrainingProgressBar(["lgbm"], ["SE"], show_progress=False)
        bar.start_time = time.time()

        bar.update("lgbm", "SE", "failed")

        assert bar._current_info["model"] == "lgbm"
        assert bar._current_info["area"] == "SE"
        assert bar._current_info["status"] == "failed"

    def test_item_display_completed(self) -> None:
        """Test item display for completed task."""
        bar = TrainingProgressBar(["lgbm"], ["SE"], show_progress=False)
        bar._current_info = {"model": "lgbm", "area": "SE", "status": "completed"}

        display = bar._item_display(None)

        assert "✓" in display
        assert "lgbm" in display
        assert "SE" in display

    def test_item_display_failed(self) -> None:
        """Test item display for failed task."""
        bar = TrainingProgressBar(["lgbm"], ["SE"], show_progress=False)
        bar._current_info = {"model": "lgbm", "area": "SE", "status": "failed"}

        display = bar._item_display(None)

        assert "✗" in display

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        with TrainingProgressBar(["lgbm"], ["SE"], show_progress=False) as bar:
            assert bar.start_time is not None
            bar.update("lgbm", "SE")
            assert bar.current_task == 1


class TestPredictionProgressBar:
    """Tests for PredictionProgressBar class."""

    def test_initialization(self) -> None:
        """Test prediction progress bar initialization."""
        bar = PredictionProgressBar([0, 1, 2], ["SE", "S"])

        assert bar.horizons == [0, 1, 2]
        assert bar.areas == ["SE", "S"]
        assert bar.total_tasks == 6

    def test_update(self) -> None:
        """Test update method."""
        bar = PredictionProgressBar([0], ["SE"], show_progress=False)
        bar.start_time = time.time()

        bar.update(0, "SE")

        assert bar.current_task == 1
        assert bar._current_info["horizon"] == 0
        assert bar._current_info["area"] == "SE"

    def test_item_display(self) -> None:
        """Test item display formatting."""
        bar = PredictionProgressBar([0], ["SE"], show_progress=False)
        bar._current_info = {"horizon": 1, "area": "NE"}

        display = bar._item_display(None)

        assert "D+1" in display
        assert "NE" in display

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        with PredictionProgressBar([0, 1], ["SE"], show_progress=False) as bar:
            bar.update(0, "SE")
            bar.update(1, "SE")
            assert bar.current_task == 2


class TestFeatureProgressBar:
    """Tests for FeatureProgressBar class."""

    def test_initialization(self) -> None:
        """Test feature progress bar initialization."""
        bar = FeatureProgressBar(["temporal", "calendar"], ["SE"])

        assert bar.plugins == ["temporal", "calendar"]
        assert bar.areas == ["SE"]
        assert bar.total_tasks == 2

    def test_update(self) -> None:
        """Test update method."""
        bar = FeatureProgressBar(["temporal"], ["SE"], show_progress=False)
        bar.start_time = time.time()

        bar.update("temporal", "SE")

        assert bar.current_task == 1
        assert bar._current_info["plugin"] == "temporal"

    def test_item_display(self) -> None:
        """Test item display formatting."""
        bar = FeatureProgressBar(["temporal"], ["SE"], show_progress=False)
        bar._current_info = {"plugin": "lag", "area": "S"}

        display = bar._item_display(None)

        assert "lag" in display
        assert "S" in display


class TestEvaluationProgressBar:
    """Tests for EvaluationProgressBar class."""

    def test_initialization(self) -> None:
        """Test evaluation progress bar initialization."""
        bar = EvaluationProgressBar(["SE", "S"], ["mape", "rmse"])

        assert bar.areas == ["SE", "S"]
        assert bar.metrics == ["mape", "rmse"]
        assert bar.total_tasks == 4

    def test_update(self) -> None:
        """Test update method."""
        bar = EvaluationProgressBar(["SE"], ["mape"], show_progress=False)
        bar.start_time = time.time()

        bar.update("SE", "mape")

        assert bar.current_task == 1
        assert bar._current_info["area"] == "SE"
        assert bar._current_info["metric"] == "mape"

    def test_item_display(self) -> None:
        """Test item display formatting."""
        bar = EvaluationProgressBar(["SE"], ["mape"], show_progress=False)
        bar._current_info = {"area": "NE", "metric": "rmse"}

        display = bar._item_display(None)

        assert "NE" in display
        assert "rmse" in display


class TestBacktestProgressBar:
    """Tests for BacktestProgressBar class."""

    def test_initialization(self) -> None:
        """Test backtest progress bar initialization."""
        bar = BacktestProgressBar(30, [7, 14])

        assert bar.days == 30
        assert bar.intervals == [7, 14]
        assert bar.total_tasks == 60

    def test_update(self) -> None:
        """Test update method."""
        bar = BacktestProgressBar(10, [7], show_progress=False)
        bar.start_time = time.time()

        bar.update(5, 7)

        assert bar.current_task == 1
        assert bar._current_info["day"] == 5
        assert bar._current_info["interval"] == 7

    def test_item_display(self) -> None:
        """Test item display formatting."""
        bar = BacktestProgressBar(10, [7], show_progress=False)
        bar._current_info = {"day": 10, "interval": 14}

        display = bar._item_display(None)

        assert "Day 10" in display
        assert "14d interval" in display


class TestMultiLevelProgressBar:
    """Tests for MultiLevelProgressBar class."""

    def test_initialization(self) -> None:
        """Test multi-level progress bar initialization."""
        bar = MultiLevelProgressBar(["Level 1", "Level 2"], [10, 20])

        assert bar.levels == 2
        assert bar.labels == ["Level 1", "Level 2"]
        assert bar.totals == [10, 20]
        assert bar.current_counts == [0, 0]

    def test_mismatched_lengths_raises(self) -> None:
        """Test that mismatched labels and totals raises error."""
        with pytest.raises(ValueError, match="same length"):
            MultiLevelProgressBar(["Level 1"], [10, 20])

    def test_update(self) -> None:
        """Test update method."""
        bar = MultiLevelProgressBar(["L1", "L2"], [5, 5], show_progress=False)

        bar.update(0, 1)
        bar.update(1, 2)

        assert bar.current_counts[0] == 1
        assert bar.current_counts[1] == 2

    def test_update_invalid_level_raises(self) -> None:
        """Test update with invalid level raises error."""
        bar = MultiLevelProgressBar(["L1"], [5], show_progress=False)

        with pytest.raises(IndexError, match="out of range"):
            bar.update(5, 1)

    def test_update_negative_level_raises(self) -> None:
        """Test update with negative level raises error."""
        bar = MultiLevelProgressBar(["L1"], [5], show_progress=False)

        with pytest.raises(IndexError, match="out of range"):
            bar.update(-1, 1)


class TestFormatTime:
    """Tests for format_time function."""

    def test_seconds(self) -> None:
        """Test formatting seconds."""
        assert format_time(5.5) == "5.5s"
        assert format_time(30.0) == "30.0s"
        assert format_time(59.9) == "59.9s"

    def test_minutes(self) -> None:
        """Test formatting minutes."""
        assert format_time(60) == "1.0m"
        assert format_time(120) == "2.0m"
        assert format_time(90) == "1.5m"

    def test_hours(self) -> None:
        """Test formatting hours."""
        assert format_time(3600) == "1.0h"
        assert format_time(7200) == "2.0h"
        assert format_time(5400) == "1.5h"


class TestFormatTimeDetailed:
    """Tests for format_time_detailed function."""

    def test_seconds_only(self) -> None:
        """Test formatting seconds only."""
        assert format_time_detailed(45) == "45s"

    def test_minutes_and_seconds(self) -> None:
        """Test formatting minutes and seconds."""
        assert format_time_detailed(90) == "1m 30s"
        assert format_time_detailed(60) == "1m"

    def test_hours_and_minutes(self) -> None:
        """Test formatting hours and minutes."""
        assert format_time_detailed(3660) == "1h 1m"
        assert format_time_detailed(3600) == "1h"


class TestCreateSimpleProgress:
    """Tests for create_simple_progress function."""

    @patch("click.progressbar")
    def test_creates_progress_bar(self, mock_progressbar: MagicMock) -> None:
        """Test creating simple progress bar."""
        create_simple_progress("Test", 100)

        mock_progressbar.assert_called_once()
        call_kwargs = mock_progressbar.call_args.kwargs
        assert call_kwargs["label"] == "Test"
        assert call_kwargs["length"] == 100
        assert call_kwargs["show_eta"] is False
        assert call_kwargs["show_percent"] is False

    @patch("click.progressbar")
    def test_custom_options(self, mock_progressbar: MagicMock) -> None:
        """Test creating progress bar with custom options."""
        create_simple_progress("Test", 50, show_eta=True, show_percent=True)

        call_kwargs = mock_progressbar.call_args.kwargs
        assert call_kwargs["show_eta"] is True
        assert call_kwargs["show_percent"] is True


class TestDisplayProgressSummary:
    """Tests for display_progress_summary function."""

    def test_basic_summary(self, capsys: pytest.CaptureFixture) -> None:
        """Test basic progress summary display."""
        display_progress_summary(10, 20, 60.0)

        captured = capsys.readouterr()
        assert "10/20" in captured.out
        assert "Progress Summary" in captured.out

    def test_with_success_count(self, capsys: pytest.CaptureFixture) -> None:
        """Test summary with success count."""
        display_progress_summary(10, 10, 30.0, success_count=8)

        captured = capsys.readouterr()
        assert "Succeeded" in captured.out
        assert "8" in captured.out

    def test_with_failed_count(self, capsys: pytest.CaptureFixture) -> None:
        """Test summary with failed count."""
        display_progress_summary(10, 10, 30.0, failed_count=2)

        captured = capsys.readouterr()
        assert "Failed" in captured.out
        assert "2" in captured.out

    def test_no_failed_when_zero(self, capsys: pytest.CaptureFixture) -> None:
        """Test no failed line when count is zero."""
        display_progress_summary(10, 10, 30.0, failed_count=0)

        captured = capsys.readouterr()
        assert "Failed" not in captured.out


class TestProgressBarIntegration:
    """Integration tests for progress bars."""

    def test_training_full_workflow(self) -> None:
        """Test full training progress workflow."""
        models = ["lgbm", "rf"]
        areas = ["SE", "S"]

        with TrainingProgressBar(models, areas, show_progress=False) as progress:
            for model in models:
                for area in areas:
                    progress.update(model, area, "completed")

            assert progress.current_task == 4
            assert len(progress.task_times) == 4

    def test_prediction_full_workflow(self) -> None:
        """Test full prediction progress workflow."""
        horizons = [0, 1, 2]
        areas = ["SE", "S"]

        with PredictionProgressBar(horizons, areas, show_progress=False) as progress:
            for horizon in horizons:
                for area in areas:
                    progress.update(horizon, area)

            assert progress.current_task == 6

    def test_feature_full_workflow(self) -> None:
        """Test full feature generation progress workflow."""
        plugins = ["temporal", "calendar", "lag"]
        areas = ["SE"]

        with FeatureProgressBar(plugins, areas, show_progress=False) as progress:
            for plugin in plugins:
                for area in areas:
                    progress.update(plugin, area)

            assert progress.current_task == 3

    def test_nested_progress_bars(self) -> None:
        """Test nested progress bar usage."""
        # Outer progress bar for models
        with TrainingProgressBar(["lgbm"], ["SE"], show_progress=False) as outer:
            # Could have inner progress bar for epochs (simulated)
            outer.update("lgbm", "SE", "completed")

            assert outer.current_task == 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_task_list(self) -> None:
        """Test progress bar with empty task list."""
        bar = TrainingProgressBar([], [], show_progress=False)
        assert bar.total_tasks == 0

    def test_single_task(self) -> None:
        """Test progress bar with single task."""
        with TrainingProgressBar(["lgbm"], ["SE"], show_progress=False) as bar:
            bar.update("lgbm", "SE")
            assert bar.progress_percent == 100.0

    def test_item_display_without_info(self) -> None:
        """Test item display when no info set."""
        bar = TrainingProgressBar(["lgbm"], ["SE"], show_progress=False)
        # Don't set _current_info
        bar._current_info = {}
        display = bar._item_display(None)
        assert display == ""

    def test_multiple_updates_same_task(self) -> None:
        """Test multiple updates track correctly."""
        bar = TrainingProgressBar(["a", "b"], ["X", "Y"], show_progress=False)
        bar.start_time = time.time()

        bar.update("a", "X")
        bar.update("a", "Y")
        bar.update("b", "X")
        bar.update("b", "Y")

        assert bar.current_task == 4
        assert len(bar.task_times) == 4
