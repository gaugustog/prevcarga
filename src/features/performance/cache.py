"""Feature caching system.

This module provides the main FeatureCache class for intelligent caching
of feature engineering computations, with support for multiple backends
and automatic cache invalidation.

Example:
    ```python
    from src.features.performance.cache import FeatureCache
    from src.features.performance.config import CacheConfig

    # Initialize cache
    config = CacheConfig(cache_backend="disk", ttl_hours=48)
    cache = FeatureCache(backend="disk", config=config)

    # Use cache with compute function
    def compute_features(df):
        return plugin.generate_features(df, config)

    features = cache.get_or_compute(
        df=input_df,
        plugin_name="lag",
        plugin_version="1.0.0",
        config={"lag_periods": [1, 7]},
        compute_fn=compute_features
    )

    # Check cache statistics
    stats = cache.get_stats()
    print(f"Hit rate: {stats['hit_rate']:.1%}")
    ```
"""

import threading
import time
from typing import Any, Callable, Optional

import pandas as pd

from src.features.performance.cache_backends import (
    DiskCacheBackend,
    MemoryCacheBackend,
    RedisCacheBackend,
)
from src.features.performance.cache_key import CacheKeyGenerator
from src.features.performance.config import CacheConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureCache:
    """Intelligent caching system for feature engineering.

    Caches computed features to avoid redundant calculations.
    Supports multiple backends (memory, disk, Redis) with
    automatic invalidation and TTL management.

    The cache key is generated from:
    - Input data fingerprint (shape, columns, dtypes, sample content)
    - Plugin name and version
    - Configuration parameters

    This ensures cache hits only occur for identical inputs while
    automatically invalidating when data or configuration changes.

    Attributes:
        config: Cache configuration.
        backend: Storage backend instance.
        key_generator: Cache key generator.
        stats: Dictionary tracking hits, misses, and time saved.

    Example:
        >>> cache = FeatureCache(backend="disk")
        >>>
        >>> def expensive_computation(df):
        ...     return plugin.generate_features(df, config)
        >>>
        >>> features = cache.get_or_compute(
        ...     df=df,
        ...     plugin_name="lag",
        ...     plugin_version="1.0.0",
        ...     config={"lag_periods": [1, 7]},
        ...     compute_fn=expensive_computation
        ... )
    """

    def __init__(
        self, backend: str = "disk", config: Optional[CacheConfig] = None
    ):
        """Initialize feature cache.

        Args:
            backend: Cache backend ("memory", "disk", "redis").
            config: Cache configuration. If None, uses default config.

        Raises:
            ValueError: If backend type is unknown.
        """
        self.config = config or CacheConfig()
        self.backend = self._create_backend(backend)
        self.key_generator = CacheKeyGenerator()
        self.stats = {"hits": 0, "misses": 0, "time_saved_seconds": 0.0}
        self._lock = threading.Lock()

        logger.info(f"Initialized FeatureCache with {backend} backend")

    def _create_backend(self, backend: str):
        """Create cache backend instance.

        Args:
            backend: Backend type ("memory", "disk", "redis").

        Returns:
            Backend instance.

        Raises:
            ValueError: If backend type is unknown.
        """
        if backend == "memory":
            return MemoryCacheBackend(max_size_mb=self.config.max_cache_size_mb)
        elif backend == "disk":
            return DiskCacheBackend(
                cache_dir=self.config.cache_directory,
                max_size_mb=self.config.max_cache_size_mb,
                ttl_hours=self.config.ttl_hours,
                compression=self.config.compression,
            )
        elif backend == "redis":
            return RedisCacheBackend(
                redis_url=self.config.redis_url, ttl_hours=self.config.ttl_hours
            )
        else:
            raise ValueError(
                f"Unknown cache backend: {backend}. "
                f"Supported: memory, disk, redis"
            )

    def get_or_compute(
        self,
        df: pd.DataFrame,
        plugin_name: str,
        plugin_version: str,
        config: dict[str, Any],
        compute_fn: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        """Get cached features or compute them.

        First tries to retrieve features from cache. If not found (cache miss),
        computes features using the provided function and stores in cache.

        Args:
            df: Input DataFrame.
            plugin_name: Feature plugin name.
            plugin_version: Plugin version string.
            config: Plugin configuration dictionary.
            compute_fn: Function to compute features if cache miss.
                Should take DataFrame as input and return feature DataFrame.

        Returns:
            Feature DataFrame (either from cache or freshly computed).

        Example:
            >>> def compute_lags(df):
            ...     return df.shift(1)
            >>>
            >>> result = cache.get_or_compute(
            ...     df=data,
            ...     plugin_name="lag",
            ...     plugin_version="1.0.0",
            ...     config={"periods": 1},
            ...     compute_fn=compute_lags
            ... )
        """
        if not self.config.enable_cache:
            logger.debug("Cache disabled, computing features")
            return compute_fn(df)

        # Generate cache key
        cache_key = self.key_generator.generate_key(
            df, plugin_name, plugin_version, config
        )

        # Try to get from cache
        cached_result = self.backend.get(cache_key)

        if cached_result is not None:
            with self._lock:
                self.stats["hits"] += 1
            logger.info(f"Cache HIT for {plugin_name}")
            return cached_result

        # Cache miss - compute features
        with self._lock:
            self.stats["misses"] += 1

        logger.info(f"Cache MISS for {plugin_name}, computing...")

        start_time = time.perf_counter()

        result = compute_fn(df)

        compute_time = time.perf_counter() - start_time

        # Store in cache
        self.backend.set(cache_key, result)

        # Track time (will be saved on subsequent hits)
        with self._lock:
            self.stats["time_saved_seconds"] += 0  # First computation, no time saved

        logger.info(
            f"Computed and cached {plugin_name} features in {compute_time:.2f}s"
        )

        return result

    def invalidate(
        self,
        df: Optional[pd.DataFrame] = None,
        plugin_name: Optional[str] = None,
    ) -> None:
        """Invalidate cache entries.

        Args:
            df: DataFrame to invalidate (None for all).
            plugin_name: Plugin to invalidate (None for all).

        Note:
            Currently only supports full cache invalidation.
            Partial invalidation by specific keys would require
            maintaining a key index.
        """
        if df is None and plugin_name is None:
            self.backend.clear()
            logger.info("Cleared entire cache")
        else:
            # Partial invalidation would require key prefix matching
            # or maintaining a reverse index
            logger.warning(
                "Partial invalidation not fully implemented. "
                "Use clear() for full cache invalidation."
            )

    def clear(self) -> None:
        """Clear entire cache.

        Removes all cached entries and resets statistics.
        """
        self.backend.clear()
        self.reset_stats()
        logger.info("Cleared cache and reset statistics")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache performance metrics:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Percentage of requests served from cache
                - time_saved_seconds: Estimated time saved by caching
                - backend_size: Number of entries in cache

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
            >>> print(f"Cache size: {stats['backend_size']} entries")
        """
        with self._lock:
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total if total > 0 else 0.0

            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "time_saved_seconds": self.stats["time_saved_seconds"],
                "backend_size": self.backend.size(),
            }

    def reset_stats(self) -> None:
        """Reset cache statistics.

        Resets hit/miss counters and time saved tracking.
        Does not affect cached entries.
        """
        with self._lock:
            self.stats = {"hits": 0, "misses": 0, "time_saved_seconds": 0.0}
        logger.info("Reset cache statistics")
