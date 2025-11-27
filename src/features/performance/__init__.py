"""Performance optimization and profiling for feature engineering.

This module provides tools for optimizing feature computation performance:
- Intelligent caching with multiple backends (memory, disk, Redis)
- Execution time and memory profiling
- Memory usage optimization
- Feature importance tracking and aggregation

Example:
    ```python
    from src.features.performance import (
        FeatureCache,
        ExecutionProfiler,
        MemoryOptimizer,
        FeatureImportanceTracker,
        CacheConfig,
        ProfilerConfig
    )

    # Setup caching
    cache_config = CacheConfig(cache_backend="disk", ttl_hours=48)
    cache = FeatureCache(backend="disk", config=cache_config)

    # Setup profiling
    profiler = ExecutionProfiler(enable_memory=True)

    # Memory optimization
    optimizer = MemoryOptimizer(aggressive=False)

    # Feature importance tracking
    tracker = FeatureImportanceTracker()
    ```
"""

from src.features.performance.cache import FeatureCache
from src.features.performance.config import CacheConfig, ProfilerConfig
from src.features.performance.importance_tracker import FeatureImportanceTracker
from src.features.performance.optimizer import MemoryOptimizer
from src.features.performance.profiler import (
    ExecutionProfiler,
    profile_execution,
)

__all__ = [
    "FeatureCache",
    "CacheConfig",
    "ProfilerConfig",
    "ExecutionProfiler",
    "profile_execution",
    "MemoryOptimizer",
    "FeatureImportanceTracker",
]
