"""Cache backend implementations.

This module provides different storage backends for the feature caching system:
- MemoryCacheBackend: In-memory cache with LRU eviction
- DiskCacheBackend: Persistent disk-based cache with Parquet format
- RedisCacheBackend: Distributed Redis-based cache (optional)

Example:
    ```python
    from src.features.performance.cache_backends import (
        MemoryCacheBackend,
        DiskCacheBackend
    )

    # Memory cache
    memory_cache = MemoryCacheBackend(max_size_mb=500)
    memory_cache.set("key1", df)
    cached_df = memory_cache.get("key1")

    # Disk cache
    disk_cache = DiskCacheBackend(
        cache_dir=Path(".cache"),
        max_size_mb=1000,
        ttl_hours=24
    )
    ```
"""

import json
import shutil
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryCacheBackend:
    """In-memory cache storage with LRU eviction.

    Features:
    - Fast access (no I/O)
    - LRU eviction policy
    - Thread-safe operations
    - Size-limited
    - Not persistent across sessions

    Attributes:
        max_size_mb: Maximum cache size in megabytes.
        cache: Ordered dictionary storing cached DataFrames.
        lock: Threading lock for thread-safe operations.
        stats: Dictionary tracking hits, misses, and evictions.
    """

    def __init__(self, max_size_mb: int = 1000):
        """Initialize memory cache.

        Args:
            max_size_mb: Maximum cache size in megabytes.
        """
        self.max_size_mb = max_size_mb
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

        logger.info(f"Initialized MemoryCacheBackend (max: {max_size_mb}MB)")

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame.

        Args:
            key: Cache key.

        Returns:
            Cached DataFrame if found, None otherwise.
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                logger.debug(f"Memory cache HIT: {key[:16]}...")
                return self.cache[key].copy()
            else:
                self.stats["misses"] += 1
                logger.debug(f"Memory cache MISS: {key[:16]}...")
                return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        """Store DataFrame in cache.

        Args:
            key: Cache key.
            df: DataFrame to cache.
        """
        with self.lock:
            # Store copy to avoid external modifications
            self.cache[key] = df.copy()
            self.cache.move_to_end(key)

            # Enforce size limit
            self._enforce_size_limit()

            logger.debug(f"Memory cache SET: {key[:16]}...")

    def delete(self, key: str) -> None:
        """Delete cache entry.

        Args:
            key: Cache key to delete.
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"Memory cache DELETE: {key[:16]}...")

    def clear(self) -> None:
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            logger.info("Cleared memory cache")

    def size(self) -> int:
        """Get number of cache entries.

        Returns:
            Number of cached items.
        """
        with self.lock:
            return len(self.cache)

    def _enforce_size_limit(self) -> None:
        """Enforce maximum cache size using LRU eviction."""
        # Estimate memory usage (rough approximation)
        total_size_mb = sum(
            df.memory_usage(deep=True).sum() for df in self.cache.values()
        ) / (1024 * 1024)

        if total_size_mb > self.max_size_mb:
            # Evict oldest entries until under limit
            while total_size_mb > self.max_size_mb * 0.8 and self.cache:
                oldest_key = next(iter(self.cache))
                evicted_df = self.cache.pop(oldest_key)
                evicted_size_mb = (
                    evicted_df.memory_usage(deep=True).sum() / (1024 * 1024)
                )
                total_size_mb -= evicted_size_mb
                self.stats["evictions"] += 1
                logger.debug(
                    f"Evicted cache entry: {oldest_key[:16]}... "
                    f"({evicted_size_mb:.1f}MB)"
                )


class DiskCacheBackend:
    """Disk-based cache storage using Parquet format.

    Features:
    - Persistent across sessions
    - Compressed storage
    - TTL-based expiration
    - Automatic cleanup
    - Thread-safe file operations

    Attributes:
        cache_dir: Directory for cache files.
        max_size_mb: Maximum cache size in megabytes.
        ttl_hours: Time-to-live for cache entries.
        compression: Enable compression.
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: int = 1000,
        ttl_hours: int = 24,
        compression: bool = True,
    ):
        """Initialize disk cache.

        Args:
            cache_dir: Directory for cache files.
            max_size_mb: Maximum cache size in MB.
            ttl_hours: Time-to-live for cache entries.
            compression: Enable compression.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.ttl_hours = ttl_hours
        self.compression = compression
        self.lock = threading.Lock()

        logger.info(f"Initialized DiskCacheBackend at {self.cache_dir}")

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame.

        Args:
            key: Cache key.

        Returns:
            Cached DataFrame if found and not expired, None otherwise.
        """
        cache_file = self._get_cache_path(key)
        metadata_file = self._get_metadata_path(key)

        if not cache_file.exists():
            logger.debug(f"Disk cache MISS: {key[:16]}...")
            return None

        # Check TTL
        if self._is_expired(metadata_file):
            logger.debug(f"Cache entry expired: {key[:16]}...")
            self._delete_entry(key)
            return None

        try:
            df = pd.read_parquet(cache_file)
            logger.debug(f"Disk cache HIT: {key[:16]}...")
            return df
        except Exception as e:
            logger.error(f"Failed to load cache entry: {e}")
            self._delete_entry(key)
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        """Store DataFrame in cache.

        Args:
            key: Cache key.
            df: DataFrame to cache.
        """
        cache_file = self._get_cache_path(key)
        metadata_file = self._get_metadata_path(key)

        try:
            with self.lock:
                # Save DataFrame
                df.to_parquet(
                    cache_file, compression="gzip" if self.compression else None
                )

                # Save metadata
                metadata = {
                    "created_at": datetime.now().isoformat(),
                    "expires_at": (
                        datetime.now() + timedelta(hours=self.ttl_hours)
                    ).isoformat(),
                    "shape": df.shape,
                    "size_bytes": cache_file.stat().st_size,
                }

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f)

                logger.debug(f"Disk cache SET: {key[:16]}...")

                # Check cache size and cleanup if needed
                self._enforce_size_limit()

        except Exception as e:
            logger.error(f"Failed to store cache entry: {e}")

    def delete(self, key: str) -> None:
        """Delete cache entry.

        Args:
            key: Cache key to delete.
        """
        self._delete_entry(key)

    def clear(self) -> None:
        """Clear entire cache."""
        with self.lock:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared disk cache")

    def size(self) -> int:
        """Get number of cache entries.

        Returns:
            Number of cached files.
        """
        if not self.cache_dir.exists():
            return 0
        return len(list(self.cache_dir.glob("*.parquet")))

    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache file.

        Args:
            key: Cache key.

        Returns:
            Path to cache file.
        """
        return self.cache_dir / f"{key}.parquet"

    def _get_metadata_path(self, key: str) -> Path:
        """Get path for metadata file.

        Args:
            key: Cache key.

        Returns:
            Path to metadata file.
        """
        return self.cache_dir / f"{key}.meta.json"

    def _is_expired(self, metadata_file: Path) -> bool:
        """Check if cache entry is expired.

        Args:
            metadata_file: Path to metadata file.

        Returns:
            True if expired, False otherwise.
        """
        if not metadata_file.exists():
            return True

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            expires_at = datetime.fromisoformat(metadata["expires_at"])
            return datetime.now() > expires_at

        except Exception:
            return True

    def _delete_entry(self, key: str) -> None:
        """Delete cache entry and metadata.

        Args:
            key: Cache key.
        """
        cache_file = self._get_cache_path(key)
        metadata_file = self._get_metadata_path(key)

        with self.lock:
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)

    def _enforce_size_limit(self) -> None:
        """Enforce maximum cache size."""
        total_size_mb = (
            sum(f.stat().st_size for f in self.cache_dir.glob("*.parquet"))
            / (1024 * 1024)
        )

        if total_size_mb > self.max_size_mb:
            logger.warning(
                f"Cache size ({total_size_mb:.1f}MB) exceeds limit "
                f"({self.max_size_mb}MB)"
            )
            # Delete oldest entries
            self._cleanup_old_entries()

    def _cleanup_old_entries(self) -> None:
        """Delete oldest cache entries."""
        entries = []

        for cache_file in self.cache_dir.glob("*.parquet"):
            key = cache_file.stem
            metadata_file = self._get_metadata_path(key)

            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    entries.append(
                        (key, datetime.fromisoformat(metadata["created_at"]))
                    )
                except Exception:
                    # If metadata is corrupted, delete the entry
                    self._delete_entry(key)

        # Sort by creation time (oldest first)
        entries.sort(key=lambda x: x[1])

        # Delete oldest 25%
        n_to_delete = max(1, len(entries) // 4)

        for key, _ in entries[:n_to_delete]:
            self._delete_entry(key)
            logger.debug(f"Deleted old cache entry: {key[:16]}...")


class RedisCacheBackend:
    """Redis-based distributed cache storage.

    Features:
    - Distributed cache across workers
    - Built-in TTL support
    - High performance
    - Requires Redis server

    Note:
        This backend requires redis-py package to be installed.
        Install with: pip install redis

    Attributes:
        redis_url: Redis connection URL.
        ttl_hours: Time-to-live for cache entries.
        client: Redis client instance.
    """

    def __init__(self, redis_url: Optional[str] = None, ttl_hours: int = 24):
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0").
            ttl_hours: Time-to-live for cache entries.

        Raises:
            ImportError: If redis package is not installed.
            ConnectionError: If cannot connect to Redis server.
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package required for RedisCacheBackend. "
                "Install with: pip install redis"
            )

        self.redis_url = redis_url or "redis://localhost:6379/0"
        self.ttl_hours = ttl_hours
        self.ttl_seconds = ttl_hours * 3600

        try:
            self.client = redis.from_url(self.redis_url, decode_responses=False)
            self.client.ping()
            logger.info(f"Connected to Redis: {self.redis_url}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame.

        Args:
            key: Cache key.

        Returns:
            Cached DataFrame if found, None otherwise.
        """
        try:
            data = self.client.get(key)
            if data is None:
                logger.debug(f"Redis cache MISS: {key[:16]}...")
                return None

            # Deserialize using pickle
            import pickle

            df = pickle.loads(data)
            logger.debug(f"Redis cache HIT: {key[:16]}...")
            return df

        except Exception as e:
            logger.error(f"Failed to get from Redis cache: {e}")
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        """Store DataFrame in cache.

        Args:
            key: Cache key.
            df: DataFrame to cache.
        """
        try:
            # Serialize using pickle
            import pickle

            data = pickle.dumps(df)

            # Store with TTL
            self.client.setex(key, self.ttl_seconds, data)
            logger.debug(f"Redis cache SET: {key[:16]}...")

        except Exception as e:
            logger.error(f"Failed to set Redis cache: {e}")

    def delete(self, key: str) -> None:
        """Delete cache entry.

        Args:
            key: Cache key to delete.
        """
        try:
            self.client.delete(key)
            logger.debug(f"Redis cache DELETE: {key[:16]}...")
        except Exception as e:
            logger.error(f"Failed to delete from Redis cache: {e}")

    def clear(self) -> None:
        """Clear entire cache.

        Warning:
            This will flush the entire Redis database, not just
            feature cache entries.
        """
        try:
            self.client.flushdb()
            logger.warning("Flushed entire Redis database")
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")

    def size(self) -> int:
        """Get number of cache entries.

        Returns:
            Approximate number of keys in Redis database.

        Note:
            This returns the total number of keys in the database,
            not just feature cache entries.
        """
        try:
            return self.client.dbsize()
        except Exception as e:
            logger.error(f"Failed to get Redis cache size: {e}")
            return 0
