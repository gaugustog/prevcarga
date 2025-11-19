# PC-078-09A: Progress Monitoring System

**Ticket ID:** PC-078-09A  
**Epic:** Epic-09A - Core CLI Commands  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 5  
**Priority:** High  
**Status:** Not Started  

---

## 📋 Description

Implement comprehensive progress monitoring system (ProgressBarManager class) with real-time progress bars for all long-running CLI operations.

---

## 🎯 Acceptance Criteria

- [ ] `ProgressBarManager` class for centralized progress management
- [ ] `TrainingProgressBar` for training operations
- [ ] `PredictionProgressBar` for prediction operations
- [ ] `FeatureProgressBar` for feature generation
- [ ] `EvaluationProgressBar` for evaluation operations
- [ ] Real-time ETA calculation
- [ ] Percentage completion display
- [ ] Task-specific progress information
- [ ] Integration with workflow callbacks from Epic-08A/B
- [ ] Context manager support for automatic cleanup
- [ ] Multi-level progress for nested operations
- [ ] Progress bar styling and formatting

---

## 🔧 Technical Implementation

```python
# prevcarga/cli/progress.py
import click
from typing import List, Optional, Callable
from contextlib import contextmanager
import time


class ProgressBarManager:
    """Centralized progress bar management for CLI commands."""
    
    def __init__(self):
        self.progress_bars = {}
        self.active_bar = None
    
    def create_progress_bar(self, label: str, length: int, **kwargs):
        """Create a new progress bar.
        
        Args:
            label: Progress bar label
            length: Total number of steps
            **kwargs: Additional click.progressbar arguments
        
        Returns:
            Progress bar object
        """
        return click.progressbar(
            length=length,
            label=label,
            show_eta=kwargs.get('show_eta', True),
            show_percent=kwargs.get('show_percent', True),
            show_pos=kwargs.get('show_pos', False),
            item_show_func=kwargs.get('item_show_func'),
            fill_char=kwargs.get('fill_char', '█'),
            empty_char=kwargs.get('empty_char', '░')
        )


class TrainingProgressBar:
    """Progress bar for model training operations."""
    
    def __init__(self, models: List[str], areas: List[str]):
        """Initialize training progress bar.
        
        Args:
            models: List of model types being trained
            areas: List of areas being trained
        """
        self.models = models
        self.areas = areas
        self.total_tasks = len(models) * len(areas)
        self.current_task = 0
        self.progress_bar = None
        self.start_time = None
        self.task_times = []
    
    def __enter__(self):
        """Enter context manager."""
        self.start_time = time.time()
        self.progress_bar = click.progressbar(
            length=self.total_tasks,
            label='Training models',
            show_eta=True,
            show_percent=True,
            fill_char='█',
            empty_char='░',
            item_show_func=self._item_display
        )
        self.progress_bar.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.progress_bar:
            self.progress_bar.__exit__(exc_type, exc_val, exc_tb)
        
        if not exc_type:
            elapsed = time.time() - self.start_time
            click.echo(f"\nCompleted {self.total_tasks} tasks in {elapsed:.1f}s")
    
    def update(self, model: str, area: str, status: str = 'completed'):
        """Update progress bar.
        
        Args:
            model: Model type that was trained
            area: Area that was trained
            status: Task status ('completed', 'failed', 'skipped')
        """
        task_time = time.time() - self.start_time
        self.task_times.append(task_time)
        
        self.current_task += 1
        self.current_info = {
            'model': model,
            'area': area,
            'status': status
        }
        
        if self.progress_bar:
            self.progress_bar.update(1)
    
    def _item_display(self, item):
        """Format current item display."""
        if hasattr(self, 'current_info'):
            info = self.current_info
            status_icon = '✓' if info['status'] == 'completed' else '✗'
            return f"{status_icon} {info['model']} - {info['area']}"
        return ""


class PredictionProgressBar:
    """Progress bar for prediction operations."""
    
    def __init__(self, horizons: List[int], areas: List[str]):
        """Initialize prediction progress bar.
        
        Args:
            horizons: List of forecast horizons
            areas: List of areas for prediction
        """
        self.horizons = horizons
        self.areas = areas
        self.total_tasks = len(horizons) * len(areas)
        self.current_task = 0
        self.progress_bar = None
        self.start_time = None
    
    def __enter__(self):
        """Enter context manager."""
        self.start_time = time.time()
        self.progress_bar = click.progressbar(
            length=self.total_tasks,
            label='Generating predictions',
            show_eta=True,
            show_percent=True,
            fill_char='█',
            empty_char='░',
            item_show_func=self._item_display
        )
        self.progress_bar.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.progress_bar:
            self.progress_bar.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, horizon: int, area: str):
        """Update progress bar.
        
        Args:
            horizon: Horizon that was predicted
            area: Area that was predicted
        """
        self.current_task += 1
        self.current_info = {
            'horizon': horizon,
            'area': area
        }
        
        if self.progress_bar:
            self.progress_bar.update(1)
    
    def _item_display(self, item):
        """Format current item display."""
        if hasattr(self, 'current_info'):
            info = self.current_info
            return f"D+{info['horizon']} - {info['area']}"
        return ""


class FeatureProgressBar:
    """Progress bar for feature generation operations."""
    
    def __init__(self, plugins: List[str], areas: List[str]):
        """Initialize feature generation progress bar.
        
        Args:
            plugins: List of feature plugins
            areas: List of areas
        """
        self.plugins = plugins
        self.areas = areas
        self.total_tasks = len(plugins) * len(areas)
        self.current_task = 0
        self.progress_bar = None
        self.start_time = None
    
    def __enter__(self):
        """Enter context manager."""
        self.start_time = time.time()
        self.progress_bar = click.progressbar(
            length=self.total_tasks,
            label='Generating features',
            show_eta=True,
            show_percent=True,
            fill_char='█',
            empty_char='░',
            item_show_func=self._item_display
        )
        self.progress_bar.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.progress_bar:
            self.progress_bar.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, plugin: str, area: str):
        """Update progress bar.
        
        Args:
            plugin: Plugin that generated features
            area: Area for which features were generated
        """
        self.current_task += 1
        self.current_info = {
            'plugin': plugin,
            'area': area
        }
        
        if self.progress_bar:
            self.progress_bar.update(1)
    
    def _item_display(self, item):
        """Format current item display."""
        if hasattr(self, 'current_info'):
            info = self.current_info
            return f"{info['plugin']} - {info['area']}"
        return ""


class EvaluationProgressBar:
    """Progress bar for evaluation operations."""
    
    def __init__(self, areas: List[str], metrics: List[str]):
        """Initialize evaluation progress bar.
        
        Args:
            areas: List of areas to evaluate
            metrics: List of metrics to calculate
        """
        self.areas = areas
        self.metrics = metrics
        self.total_tasks = len(areas) * len(metrics)
        self.current_task = 0
        self.progress_bar = None
    
    def __enter__(self):
        """Enter context manager."""
        self.progress_bar = click.progressbar(
            length=self.total_tasks,
            label='Evaluating model',
            show_eta=True,
            show_percent=True,
            fill_char='█',
            empty_char='░'
        )
        self.progress_bar.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.progress_bar:
            self.progress_bar.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, area: str, metric: str):
        """Update progress bar."""
        self.current_task += 1
        if self.progress_bar:
            self.progress_bar.update(1)


class BacktestProgressBar:
    """Progress bar for backtesting operations."""
    
    def __init__(self, days: int, intervals: List[int]):
        """Initialize backtest progress bar.
        
        Args:
            days: Total days in backtest period
            intervals: Retraining intervals being tested
        """
        self.days = days
        self.intervals = intervals
        self.total_tasks = days * len(intervals)
        self.current_day = 0
        self.progress_bar = None
    
    def __enter__(self):
        """Enter context manager."""
        self.progress_bar = click.progressbar(
            length=self.total_tasks,
            label='Running backtest',
            show_eta=True,
            show_percent=True,
            fill_char='█',
            empty_char='░'
        )
        self.progress_bar.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.progress_bar:
            self.progress_bar.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, day: int, interval: int):
        """Update progress bar."""
        self.current_day = day
        if self.progress_bar:
            self.progress_bar.update(1)


# Utility functions for progress display
def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_simple_progress(label: str, length: int):
    """Create a simple progress bar for quick tasks."""
    return click.progressbar(
        length=length,
        label=label,
        show_eta=False,
        show_percent=False
    )
```

---

## 🧪 Testing Requirements

```python
# tests/cli/test_progress.py
import pytest
from prevcarga.cli.progress import (
    TrainingProgressBar,
    PredictionProgressBar,
    FeatureProgressBar
)


def test_training_progress_bar():
    """Test training progress bar."""
    models = ['lgbm', 'rf']
    areas = ['SE', 'S']
    
    with TrainingProgressBar(models, areas) as progress:
        assert progress.total_tasks == 4
        
        progress.update('lgbm', 'SE', 'completed')
        assert progress.current_task == 1
        
        progress.update('lgbm', 'S', 'completed')
        assert progress.current_task == 2


def test_prediction_progress_bar():
    """Test prediction progress bar."""
    horizons = [0, 1, 2]
    areas = ['SE', 'S']
    
    with PredictionProgressBar(horizons, areas) as progress:
        assert progress.total_tasks == 6
        
        progress.update(0, 'SE')
        assert progress.current_task == 1


def test_feature_progress_bar():
    """Test feature generation progress bar."""
    plugins = ['temporal', 'calendar']
    areas = ['SE']
    
    with FeatureProgressBar(plugins, areas) as progress:
        assert progress.total_tasks == 2
        
        progress.update('temporal', 'SE')
        assert progress.current_task == 1
```

---

## ✅ Definition of Done

- [ ] All progress bar classes implemented
- [ ] Context manager support working
- [ ] Real-time updates functional
- [ ] ETA calculation accurate
- [ ] Visual styling consistent
- [ ] Unit tests pass (>85% coverage)
- [ ] Integration with commands tested
- [ ] Documentation complete

---

**Assignee:** Backend Team  
**Estimated Hours:** 8-10 hours  
**Target Completion:** Week 24, Day 5
