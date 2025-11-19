# PC-023-02B: Performance Optimization & Feature Importance Tracking

**Ticket ID:** PC-023-02B  
**Epic:** [Epic-02B: Advanced Feature Transformations](../epics/Epic-02B.md)  
**User Story:** US-6  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement system-level performance optimization and feature importance tracking infrastructure for the feature engineering pipeline. Includes intelligent caching, execution profiling, memory optimization, and persistent feature importance metadata to accelerate model development and reduce computational costs.

**As a** ML engineer  
**I want** optimized feature computation with caching and importance tracking  
**So that** I can iterate faster and understand which features drive model performance

---

## ✅ Acceptance Criteria

- [ ] Feature cache system reduces redundant computation by >70%
- [ ] Execution time profiler identifies bottlenecks
- [ ] Memory profiler tracks and optimizes memory usage
- [ ] Feature importance tracker persists across runs
- [ ] Cache invalidation detects data/config changes
- [ ] Configurable cache storage (disk, memory, Redis)
- [ ] Profiling reports with visualization
- [ ] Importance aggregation across multiple models
- [ ] Performance benchmarks show >50% speedup with caching
- [ ] Thread-safe cache operations
- [ ] Comprehensive tests for cache correctness

---

## 🔧 Implementation Tasks

### 1. Create Performance Module Structure
- [ ] Create `src/features/performance/` directory
- [ ] Create `src/features/performance/cache.py`
- [ ] Create `src/features/performance/profiler.py`
- [ ] Create `src/features/performance/importance_tracker.py`
- [ ] Create `src/features/performance/optimizer.py`
- [ ] Add module docstrings

### 2. Implement Cache Configuration
- [ ] Create `CacheConfig` Pydantic model
- [ ] Add `cache_backend` enum: "memory", "disk", "redis"
- [ ] Add `cache_directory` path field
- [ ] Add `max_cache_size_mb` integer (default: 1000)
- [ ] Add `ttl_hours` integer (time-to-live, default: 24)
- [ ] Add `compression` boolean (default: True)
- [ ] Add `enable_cache` boolean (default: True)
- [ ] Add `cache_key_prefix` string
- [ ] Validate cache directory is writable

### 3. Implement Cache Key Generation
- [ ] Create `CacheKeyGenerator` class
- [ ] Generate hash from DataFrame shape and sample
- [ ] Include plugin name and version in key
- [ ] Include configuration parameters in key
- [ ] Hash feature names and dtypes
- [ ] Support content-based hashing (checksums)
- [ ] Ensure deterministic key generation
- [ ] Handle timezone-aware indices

### 4. Implement Memory Cache Backend
- [ ] Create `MemoryCacheBackend` class
- [ ] Use dict for storage
- [ ] Implement LRU eviction policy
- [ ] Track cache size and enforce limits
- [ ] Thread-safe operations with locks
- [ ] Implement `get()`, `set()`, `delete()`, `clear()`
- [ ] Track hit/miss statistics

### 5. Implement Disk Cache Backend
- [ ] Create `DiskCacheBackend` class
- [ ] Use Parquet format for DataFrames
- [ ] Store metadata in JSON sidecar files
- [ ] Implement compression (gzip)
- [ ] Check TTL on read
- [ ] Clean expired entries periodically
- [ ] Handle file system errors gracefully

### 6. Implement Redis Cache Backend (Optional)
- [ ] Create `RedisCacheBackend` class
- [ ] Connect to Redis server
- [ ] Serialize DataFrames to bytes (pickle or arrow)
- [ ] Use Redis TTL for expiration
- [ ] Handle connection failures
- [ ] Support Redis Cluster
- [ ] Add redis-py dependency

### 7. Implement FeatureCache Main Class
- [ ] Create `FeatureCache` class
- [ ] Initialize with backend (memory/disk/redis)
- [ ] Implement `get_or_compute()` method
- [ ] Check cache before computation
- [ ] Store results after computation
- [ ] Log cache hits/misses
- [ ] Provide cache statistics
- [ ] Support cache warming

### 8. Implement Cache Invalidation
- [ ] Detect data changes (hash comparison)
- [ ] Detect config changes
- [ ] Detect plugin version changes
- [ ] Implement manual invalidation API
- [ ] Support partial invalidation (specific plugins)
- [ ] Log invalidation events

### 9. Implement Execution Profiler
- [ ] Create `ExecutionProfiler` class
- [ ] Track execution time per plugin
- [ ] Track memory usage per plugin
- [ ] Use `time.perf_counter()` for timing
- [ ] Use `memory_profiler` for memory tracking
- [ ] Store profiling results in DataFrame
- [ ] Support nested profiling (plugin methods)

### 10. Implement Profiling Context Manager
- [ ] Create `@profile_execution` decorator
- [ ] Automatically track decorated functions
- [ ] Capture start/end timestamps
- [ ] Measure peak memory usage
- [ ] Store results in profiler instance
- [ ] Support async functions

### 11. Implement Memory Optimizer
- [ ] Create `MemoryOptimizer` class
- [ ] Downcast numeric types (int64 → int32 where safe)
- [ ] Convert object columns to category
- [ ] Remove duplicate columns
- [ ] Identify and report memory savings
- [ ] Optional in-place optimization

### 12. Implement Feature Importance Tracker
- [ ] Create `FeatureImportanceTracker` class
- [ ] Store importance scores per feature
- [ ] Support multiple models (aggregate scores)
- [ ] Track importance source (model type, date)
- [ ] Persist to disk (JSON or Parquet)
- [ ] Load historical importance data
- [ ] Calculate aggregated importance (mean, median, max)

### 13. Implement Importance Storage Format
- [ ] Create importance schema
- [ ] Fields: feature_name, importance_score, model_id, timestamp
- [ ] Support multiple importance types (gain, split, permutation)
- [ ] Store metadata (model hyperparameters)
- [ ] Append-only for historical tracking

### 14. Implement Importance Aggregation
- [ ] Create `aggregate_importance()` method
- [ ] Combine scores from multiple models
- [ ] Weight by model performance (optional)
- [ ] Calculate temporal trends in importance
- [ ] Identify stable vs volatile features
- [ ] Generate importance rankings

### 15. Implement Profiling Report Generator
- [ ] Create `ProfilingReport` class
- [ ] Generate execution time summary
- [ ] Generate memory usage summary
- [ ] Identify top bottlenecks
- [ ] Compare runs (before/after optimization)
- [ ] Export to markdown/HTML/JSON

### 16. Implement Performance Visualization
- [ ] Create `plot_execution_times()` method
- [ ] Bar chart of plugin execution times
- [ ] Waterfall chart showing cumulative time
- [ ] Memory usage over time
- [ ] Cache hit rate visualization
- [ ] Save plots to files

### 17. Implement Cache Statistics
- [ ] Track total cache hits/misses
- [ ] Calculate hit rate percentage
- [ ] Track cache size (entries, bytes)
- [ ] Track time saved by caching
- [ ] Provide `get_stats()` method
- [ ] Reset statistics on demand

### 18. Integrate Cache with Plugin System
- [ ] Modify `BaseFeaturePlugin` to support caching
- [ ] Add `cache_features()` method
- [ ] Wrap `generate_features()` with cache check
- [ ] Make caching opt-in per plugin
- [ ] Support cache bypass flag
- [ ] Log cache operations

### 19. Write Comprehensive Tests
- [ ] Create `tests/features/performance/test_cache.py`
- [ ] Test memory cache operations
- [ ] Test disk cache persistence
- [ ] Test cache key generation
- [ ] Test cache invalidation
- [ ] Test TTL expiration
- [ ] Test concurrent access
- [ ] Test cache size limits
- [ ] Mock Redis for testing

### 20. Write Profiler Tests
- [ ] Create `tests/features/performance/test_profiler.py`
- [ ] Test execution time tracking
- [ ] Test memory profiling
- [ ] Test profiling decorator
- [ ] Test report generation
- [ ] Test nested profiling

### 21. Write Importance Tracker Tests
- [ ] Create `tests/features/performance/test_importance_tracker.py`
- [ ] Test importance storage
- [ ] Test importance loading
- [ ] Test aggregation logic
- [ ] Test multi-model tracking
- [ ] Test temporal analysis

### 22. Create Performance Optimization Guide
- [ ] Create `docs/mvp/guides/performance-optimization.md`
- [ ] Document caching best practices
- [ ] Profiling workflow
- [ ] Memory optimization techniques
- [ ] Cache configuration guidelines
- [ ] Troubleshooting common issues

### 23. Create Usage Examples
- [ ] Create `examples/performance_optimization_demo.py`
- [ ] Show cache setup and usage
- [ ] Show profiling execution
- [ ] Show importance tracking
- [ ] Compare performance with/without cache
- [ ] Demonstrate memory optimization

---

## 💻 Implementation Details

### Cache Configuration Schema

```python
"""Configuration for performance optimization."""
from typing import Literal, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class CacheConfig(BaseModel):
    """Configuration for feature caching system."""
    
    enable_cache: bool = Field(
        default=True,
        description="Enable feature caching"
    )
    cache_backend: Literal["memory", "disk", "redis"] = Field(
        default="disk",
        description="Cache storage backend"
    )
    cache_directory: Path = Field(
        default=Path(".cache/features"),
        description="Directory for disk cache"
    )
    max_cache_size_mb: int = Field(
        default=1000,
        description="Maximum cache size in MB"
    )
    ttl_hours: int = Field(
        default=24,
        description="Cache entry time-to-live in hours"
    )
    compression: bool = Field(
        default=True,
        description="Enable cache compression"
    )
    cache_key_prefix: str = Field(
        default="feature_cache",
        description="Prefix for cache keys"
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL (for redis backend)"
    )
    
    @field_validator('max_cache_size_mb')
    @classmethod
    def validate_cache_size(cls, v):
        """Validate cache size is positive."""
        if v <= 0:
            raise ValueError("max_cache_size_mb must be positive")
        return v
    
    @field_validator('ttl_hours')
    @classmethod
    def validate_ttl(cls, v):
        """Validate TTL is positive."""
        if v <= 0:
            raise ValueError("ttl_hours must be positive")
        return v


class ProfilerConfig(BaseModel):
    """Configuration for execution profiling."""
    
    enable_profiling: bool = Field(
        default=False,
        description="Enable execution profiling"
    )
    profile_memory: bool = Field(
        default=True,
        description="Include memory profiling"
    )
    profile_directory: Path = Field(
        default=Path(".profiles"),
        description="Directory for profiling results"
    )
    save_reports: bool = Field(
        default=True,
        description="Save profiling reports to disk"
    )
```

### Cache Key Generation

```python
"""Cache key generation utilities."""
import hashlib
import json
from typing import Any, Dict
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheKeyGenerator:
    """
    Generate deterministic cache keys for feature computations.
    
    Keys are based on:
    - Data shape and sample content
    - Plugin name and version
    - Configuration parameters
    - Feature column names and types
    """
    
    @staticmethod
    def generate_key(
        df: pd.DataFrame,
        plugin_name: str,
        plugin_version: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Generate cache key for feature computation.
        
        Args:
            df: Input DataFrame
            plugin_name: Name of feature plugin
            plugin_version: Plugin version
            config: Plugin configuration
        
        Returns:
            Hexadecimal cache key
        """
        key_components = []
        
        # Data fingerprint
        data_fingerprint = CacheKeyGenerator._create_data_fingerprint(df)
        key_components.append(data_fingerprint)
        
        # Plugin identifier
        plugin_id = f"{plugin_name}:v{plugin_version}"
        key_components.append(plugin_id)
        
        # Config hash
        config_hash = CacheKeyGenerator._hash_config(config)
        key_components.append(config_hash)
        
        # Combine and hash
        combined = "|".join(key_components)
        cache_key = hashlib.sha256(combined.encode()).hexdigest()
        
        logger.debug(f"Generated cache key: {cache_key[:16]}...")
        
        return cache_key
    
    @staticmethod
    def _create_data_fingerprint(df: pd.DataFrame) -> str:
        """Create fingerprint of DataFrame."""
        components = [
            f"shape:{df.shape}",
            f"columns:{','.join(sorted(df.columns))}",
            f"dtypes:{','.join(str(dt) for dt in df.dtypes)}",
            f"index_type:{type(df.index).__name__}"
        ]
        
        # Sample first and last few rows for content hash
        if len(df) > 0:
            sample_size = min(10, len(df))
            head_hash = pd.util.hash_pandas_object(df.head(sample_size)).sum()
            tail_hash = pd.util.hash_pandas_object(df.tail(sample_size)).sum()
            components.append(f"content:{head_hash}:{tail_hash}")
        
        return "|".join(components)
    
    @staticmethod
    def _hash_config(config: Dict[str, Any]) -> str:
        """Hash configuration dictionary."""
        # Serialize config to deterministic JSON
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        return config_hash
```

### Feature Cache Implementation

```python
"""Feature caching system."""
from typing import Optional, Callable, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import threading

from src.features.performance.cache_backends import (
    MemoryCacheBackend,
    DiskCacheBackend
)
from src.features.performance.cache_key import CacheKeyGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureCache:
    """
    Intelligent caching system for feature engineering.
    
    Caches computed features to avoid redundant calculations.
    Supports multiple backends (memory, disk, Redis) with
    automatic invalidation and TTL management.
    
    Example:
        >>> cache = FeatureCache(backend="disk")
        >>> 
        >>> def expensive_computation(df):
        ...     return plugin.generate_features(df, config)
        >>> 
        >>> features = cache.get_or_compute(
        ...     key=cache_key,
        ...     compute_fn=expensive_computation,
        ...     df=df
        ... )
    """
    
    def __init__(
        self,
        backend: str = "disk",
        config: Optional[CacheConfig] = None
    ):
        """
        Initialize feature cache.
        
        Args:
            backend: Cache backend ("memory", "disk", "redis")
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.backend = self._create_backend(backend)
        self.key_generator = CacheKeyGenerator()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'time_saved_seconds': 0.0
        }
        self._lock = threading.Lock()
        
        logger.info(f"Initialized FeatureCache with {backend} backend")
    
    def _create_backend(self, backend: str):
        """Create cache backend instance."""
        if backend == "memory":
            return MemoryCacheBackend(
                max_size_mb=self.config.max_cache_size_mb
            )
        elif backend == "disk":
            return DiskCacheBackend(
                cache_dir=self.config.cache_directory,
                max_size_mb=self.config.max_cache_size_mb,
                ttl_hours=self.config.ttl_hours,
                compression=self.config.compression
            )
        elif backend == "redis":
            # Import here to make redis optional
            from src.features.performance.cache_backends import RedisCacheBackend
            return RedisCacheBackend(
                redis_url=self.config.redis_url,
                ttl_hours=self.config.ttl_hours
            )
        else:
            raise ValueError(f"Unknown cache backend: {backend}")
    
    def get_or_compute(
        self,
        df: pd.DataFrame,
        plugin_name: str,
        plugin_version: str,
        config: Dict[str, Any],
        compute_fn: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Get cached features or compute them.
        
        Args:
            df: Input DataFrame
            plugin_name: Feature plugin name
            plugin_version: Plugin version
            config: Plugin configuration
            compute_fn: Function to compute features if cache miss
        
        Returns:
            Feature DataFrame
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
                self.stats['hits'] += 1
            logger.info(f"Cache HIT for {plugin_name}")
            return cached_result
        
        # Cache miss - compute features
        with self._lock:
            self.stats['misses'] += 1
        
        logger.info(f"Cache MISS for {plugin_name}, computing...")
        
        import time
        start_time = time.perf_counter()
        
        result = compute_fn(df)
        
        compute_time = time.perf_counter() - start_time
        
        # Store in cache
        self.backend.set(cache_key, result)
        
        logger.info(
            f"Computed and cached {plugin_name} features in {compute_time:.2f}s"
        )
        
        return result
    
    def invalidate(
        self,
        df: Optional[pd.DataFrame] = None,
        plugin_name: Optional[str] = None
    ) -> None:
        """
        Invalidate cache entries.
        
        Args:
            df: DataFrame to invalidate (None for all)
            plugin_name: Plugin to invalidate (None for all)
        """
        if df is None and plugin_name is None:
            self.backend.clear()
            logger.info("Cleared entire cache")
        else:
            # Partial invalidation would require key prefix matching
            logger.warning("Partial invalidation not fully implemented")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hit rate and other metrics
        """
        with self._lock:
            total = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total if total > 0 else 0.0
            
            return {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'time_saved_seconds': self.stats['time_saved_seconds'],
                'backend_size': self.backend.size()
            }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self.stats = {
                'hits': 0,
                'misses': 0,
                'time_saved_seconds': 0.0
            }
        logger.info("Reset cache statistics")
```

### Disk Cache Backend

```python
"""Disk-based cache backend."""
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import json
import shutil

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiskCacheBackend:
    """
    Disk-based cache storage using Parquet format.
    
    Features:
    - Persistent across sessions
    - Compressed storage
    - TTL-based expiration
    - Automatic cleanup
    """
    
    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: int = 1000,
        ttl_hours: int = 24,
        compression: bool = True
    ):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            ttl_hours: Time-to-live for cache entries
            compression: Enable compression
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.ttl_hours = ttl_hours
        self.compression = compression
        
        logger.info(f"Initialized disk cache at {self.cache_dir}")
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached DataFrame."""
        cache_file = self._get_cache_path(key)
        metadata_file = self._get_metadata_path(key)
        
        if not cache_file.exists():
            return None
        
        # Check TTL
        if self._is_expired(metadata_file):
            logger.debug(f"Cache entry expired: {key[:16]}...")
            self._delete_entry(key)
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            logger.debug(f"Loaded cache entry: {key[:16]}...")
            return df
        except Exception as e:
            logger.error(f"Failed to load cache entry: {e}")
            self._delete_entry(key)
            return None
    
    def set(self, key: str, df: pd.DataFrame) -> None:
        """Store DataFrame in cache."""
        cache_file = self._get_cache_path(key)
        metadata_file = self._get_metadata_path(key)
        
        try:
            # Save DataFrame
            df.to_parquet(
                cache_file,
                compression='gzip' if self.compression else None
            )
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=self.ttl_hours)).isoformat(),
                'shape': df.shape,
                'size_bytes': cache_file.stat().st_size
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            logger.debug(f"Stored cache entry: {key[:16]}...")
            
            # Check cache size and cleanup if needed
            self._enforce_size_limit()
            
        except Exception as e:
            logger.error(f"Failed to store cache entry: {e}")
    
    def delete(self, key: str) -> None:
        """Delete cache entry."""
        self._delete_entry(key)
    
    def clear(self) -> None:
        """Clear entire cache."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared disk cache")
    
    def size(self) -> int:
        """Get number of cache entries."""
        if not self.cache_dir.exists():
            return 0
        return len(list(self.cache_dir.glob("*.parquet")))
    
    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache file."""
        return self.cache_dir / f"{key}.parquet"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get path for metadata file."""
        return self.cache_dir / f"{key}.meta.json"
    
    def _is_expired(self, metadata_file: Path) -> bool:
        """Check if cache entry is expired."""
        if not metadata_file.exists():
            return True
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            expires_at = datetime.fromisoformat(metadata['expires_at'])
            return datetime.now() > expires_at
            
        except Exception:
            return True
    
    def _delete_entry(self, key: str) -> None:
        """Delete cache entry and metadata."""
        cache_file = self._get_cache_path(key)
        metadata_file = self._get_metadata_path(key)
        
        cache_file.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)
    
    def _enforce_size_limit(self) -> None:
        """Enforce maximum cache size."""
        total_size_mb = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.parquet")
        ) / (1024 * 1024)
        
        if total_size_mb > self.max_size_mb:
            logger.warning(
                f"Cache size ({total_size_mb:.1f}MB) exceeds limit ({self.max_size_mb}MB)"
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
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                entries.append((
                    key,
                    datetime.fromisoformat(metadata['created_at'])
                ))
        
        # Sort by creation time (oldest first)
        entries.sort(key=lambda x: x[1])
        
        # Delete oldest 25%
        n_to_delete = max(1, len(entries) // 4)
        
        for key, _ in entries[:n_to_delete]:
            self._delete_entry(key)
            logger.debug(f"Deleted old cache entry: {key[:16]}...")
```

### Feature Importance Tracker

```python
"""Feature importance tracking and aggregation."""
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureImportanceTracker:
    """
    Track and aggregate feature importance across multiple models.
    
    Maintains historical record of feature importance scores,
    enabling analysis of feature stability and impact over time.
    
    Example:
        >>> tracker = FeatureImportanceTracker()
        >>> 
        >>> # Record importance from model
        >>> tracker.record_importance(
        ...     feature_importance={'feature1': 0.5, 'feature2': 0.3},
        ...     model_id='lgbm_v1',
        ...     metadata={'horizon': 1, 'area': 'SE'}
        ... )
        >>> 
        >>> # Get aggregated importance
        >>> agg_importance = tracker.get_aggregated_importance()
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize importance tracker.
        
        Args:
            storage_path: Path to store importance data
        """
        self.storage_path = storage_path or Path(".importance/feature_importance.parquet")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing importance data
        self.importance_records = self._load_records()
        
        logger.info(f"Initialized FeatureImportanceTracker with {len(self.importance_records)} records")
    
    def record_importance(
        self,
        feature_importance: Dict[str, float],
        model_id: str,
        importance_type: str = "gain",
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record feature importance from a model.
        
        Args:
            feature_importance: Dictionary mapping features to importance scores
            model_id: Identifier for the model
            importance_type: Type of importance (gain, split, permutation, shap)
            metadata: Additional metadata (horizon, area, hyperparameters, etc.)
        """
        timestamp = datetime.now()
        
        records = []
        for feature_name, importance_score in feature_importance.items():
            record = {
                'feature_name': feature_name,
                'importance_score': importance_score,
                'importance_type': importance_type,
                'model_id': model_id,
                'timestamp': timestamp,
                **(metadata or {})
            }
            records.append(record)
        
        # Append to existing records
        new_df = pd.DataFrame(records)
        
        if self.importance_records.empty:
            self.importance_records = new_df
        else:
            self.importance_records = pd.concat(
                [self.importance_records, new_df],
                ignore_index=True
            )
        
        logger.info(
            f"Recorded importance for {len(feature_importance)} features from {model_id}"
        )
        
        # Save to disk
        self._save_records()
    
    def get_aggregated_importance(
        self,
        method: str = "mean",
        importance_type: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get aggregated feature importance.
        
        Args:
            method: Aggregation method ("mean", "median", "max")
            importance_type: Filter by importance type
            top_k: Return only top-k features
        
        Returns:
            DataFrame with aggregated importance scores
        """
        if self.importance_records.empty:
            logger.warning("No importance records available")
            return pd.DataFrame()
        
        df = self.importance_records.copy()
        
        # Filter by importance type
        if importance_type:
            df = df[df['importance_type'] == importance_type]
        
        # Aggregate by feature
        if method == "mean":
            agg_df = df.groupby('feature_name')['importance_score'].mean()
        elif method == "median":
            agg_df = df.groupby('feature_name')['importance_score'].median()
        elif method == "max":
            agg_df = df.groupby('feature_name')['importance_score'].max()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Convert to DataFrame and sort
        result = agg_df.reset_index()
        result.columns = ['feature_name', 'aggregated_importance']
        result = result.sort_values('aggregated_importance', ascending=False)
        
        # Top-k filtering
        if top_k:
            result = result.head(top_k)
        
        logger.info(f"Aggregated importance for {len(result)} features using {method}")
        
        return result
    
    def get_importance_trends(
        self,
        feature_name: str
    ) -> pd.DataFrame:
        """
        Get importance trend for specific feature.
        
        Args:
            feature_name: Feature to analyze
        
        Returns:
            DataFrame with importance over time
        """
        if self.importance_records.empty:
            return pd.DataFrame()
        
        trend = self.importance_records[
            self.importance_records['feature_name'] == feature_name
        ].sort_values('timestamp')
        
        return trend[['timestamp', 'importance_score', 'model_id']]
    
    def get_top_features(
        self,
        n: int = 20,
        importance_type: Optional[str] = None
    ) -> List[str]:
        """
        Get list of top-n most important features.
        
        Args:
            n: Number of features to return
            importance_type: Filter by importance type
        
        Returns:
            List of feature names
        """
        agg_importance = self.get_aggregated_importance(
            method="mean",
            importance_type=importance_type,
            top_k=n
        )
        
        return agg_importance['feature_name'].tolist()
    
    def _load_records(self) -> pd.DataFrame:
        """Load importance records from disk."""
        if self.storage_path.exists():
            try:
                return pd.read_parquet(self.storage_path)
            except Exception as e:
                logger.error(f"Failed to load importance records: {e}")
        
        return pd.DataFrame()
    
    def _save_records(self) -> None:
        """Save importance records to disk."""
        try:
            self.importance_records.to_parquet(
                self.storage_path,
                compression='gzip'
            )
            logger.debug(f"Saved importance records to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save importance records: {e}")
```

---

## 🧪 Testing & Validation

### Cache Tests

```python
"""Tests for feature caching system."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import time

from src.features.performance.cache import FeatureCache, CacheConfig
from src.features.performance.cache_key import CacheKeyGenerator


@pytest.fixture
def sample_df():
    """Create sample DataFrame."""
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })


@pytest.fixture
def cache_dir(tmp_path):
    """Create temporary cache directory."""
    return tmp_path / "test_cache"


def test_memory_cache_hit(sample_df):
    """Test memory cache hit."""
    cache = FeatureCache(backend="memory")
    
    def compute_fn(df):
        return df * 2
    
    # First call - cache miss
    result1 = cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn
    )
    
    # Second call - cache hit
    result2 = cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn
    )
    
    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)
    
    # Check statistics
    stats = cache.get_stats()
    assert stats['hits'] == 1
    assert stats['misses'] == 1


def test_disk_cache_persistence(sample_df, cache_dir):
    """Test disk cache persists across instances."""
    config = CacheConfig(cache_directory=cache_dir)
    
    # First cache instance
    cache1 = FeatureCache(backend="disk", config=config)
    
    def compute_fn(df):
        return df * 2
    
    result1 = cache1.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn
    )
    
    # Create new cache instance (simulating restart)
    cache2 = FeatureCache(backend="disk", config=config)
    
    result2 = cache2.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn
    )
    
    # Should load from persisted cache
    pd.testing.assert_frame_equal(result1, result2)
    
    stats = cache2.get_stats()
    assert stats['hits'] == 1


def test_cache_invalidation_on_data_change(sample_df):
    """Test cache invalidates when data changes."""
    cache = FeatureCache(backend="memory")
    
    def compute_fn(df):
        return df * 2
    
    # First computation
    result1 = cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn
    )
    
    # Change data
    sample_df['feature1'] = sample_df['feature1'] + 100
    
    # Should compute again (different data hash)
    result2 = cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn
    )
    
    # Results should be different
    assert not result1.equals(result2)


def test_cache_invalidation_on_config_change(sample_df):
    """Test cache invalidates when config changes."""
    cache = FeatureCache(backend="memory")
    
    def compute_fn(df):
        return df * 2
    
    # First computation
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={'param': 1},
        compute_fn=compute_fn
    )
    
    # Different config
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={'param': 2},
        compute_fn=compute_fn
    )
    
    # Should have 2 misses (different configs)
    stats = cache.get_stats()
    assert stats['misses'] == 2


def test_cache_key_generation_deterministic(sample_df):
    """Test cache key generation is deterministic."""
    generator = CacheKeyGenerator()
    
    key1 = generator.generate_key(
        df=sample_df,
        plugin_name="test",
        plugin_version="1.0.0",
        config={'a': 1}
    )
    
    key2 = generator.generate_key(
        df=sample_df,
        plugin_name="test",
        plugin_version="1.0.0",
        config={'a': 1}
    )
    
    assert key1 == key2


def test_cache_performance_improvement(sample_df):
    """Test cache provides performance improvement."""
    cache = FeatureCache(backend="memory")
    
    def slow_compute_fn(df):
        time.sleep(0.1)  # Simulate slow computation
        return df * 2
    
    # First call - slow
    start = time.perf_counter()
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=slow_compute_fn
    )
    first_time = time.perf_counter() - start
    
    # Second call - fast (cached)
    start = time.perf_counter()
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=slow_compute_fn
    )
    cached_time = time.perf_counter() - start
    
    # Cached should be much faster
    assert cached_time < first_time / 10
```

### Importance Tracker Tests

```python
"""Tests for feature importance tracker."""
import pytest
import pandas as pd
from pathlib import Path

from src.features.performance.importance_tracker import FeatureImportanceTracker


@pytest.fixture
def storage_path(tmp_path):
    """Create temporary storage path."""
    return tmp_path / "importance.parquet"


def test_record_and_retrieve_importance(storage_path):
    """Test recording and retrieving importance."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)
    
    importance = {
        'feature1': 0.5,
        'feature2': 0.3,
        'feature3': 0.2
    }
    
    tracker.record_importance(
        feature_importance=importance,
        model_id='model1',
        importance_type='gain'
    )
    
    # Get aggregated importance
    agg = tracker.get_aggregated_importance()
    
    assert len(agg) == 3
    assert agg.iloc[0]['feature_name'] == 'feature1'
    assert agg.iloc[0]['aggregated_importance'] == 0.5


def test_aggregate_multiple_models(storage_path):
    """Test aggregating importance across models."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)
    
    # Record from model 1
    tracker.record_importance(
        feature_importance={'feature1': 0.5, 'feature2': 0.3},
        model_id='model1'
    )
    
    # Record from model 2
    tracker.record_importance(
        feature_importance={'feature1': 0.7, 'feature2': 0.2},
        model_id='model2'
    )
    
    # Aggregate (mean)
    agg = tracker.get_aggregated_importance(method='mean')
    
    # feature1 should have mean of 0.6
    feature1_importance = agg[agg['feature_name'] == 'feature1']['aggregated_importance'].values[0]
    assert abs(feature1_importance - 0.6) < 0.01


def test_get_top_features(storage_path):
    """Test getting top features."""
    tracker = FeatureImportanceTracker(storage_path=storage_path)
    
    tracker.record_importance(
        feature_importance={
            'feature1': 0.5,
            'feature2': 0.3,
            'feature3': 0.2,
            'feature4': 0.1
        },
        model_id='model1'
    )
    
    top_features = tracker.get_top_features(n=2)
    
    assert len(top_features) == 2
    assert top_features[0] == 'feature1'
    assert top_features[1] == 'feature2'


def test_persistence(storage_path):
    """Test importance data persists."""
    # First tracker instance
    tracker1 = FeatureImportanceTracker(storage_path=storage_path)
    tracker1.record_importance(
        feature_importance={'feature1': 0.5},
        model_id='model1'
    )
    
    # Create new instance (simulating restart)
    tracker2 = FeatureImportanceTracker(storage_path=storage_path)
    
    # Should load previous data
    agg = tracker2.get_aggregated_importance()
    assert len(agg) == 1
    assert agg.iloc[0]['feature_name'] == 'feature1'
```

---

## 📝 Technical Notes

### Cache Backends Comparison
- **Memory**: Fastest, but not persistent. Use for single-run optimization
- **Disk**: Persistent, good balance. Recommended for most use cases
- **Redis**: Distributed, supports multiple workers. Use for production clusters

### Cache Invalidation Strategy
- Content-based: Hash data shape and sample
- Version-based: Include plugin version in key
- Config-based: Hash configuration parameters
- Time-based: TTL expiration

### Memory Optimization Techniques
- Downcast numerics: int64 → int32 saves 50% memory
- Categorize strings: Convert to category dtype
- Drop duplicates: Remove redundant columns
- Sparse matrices: Use sparse format for sparse features

---

## 🔗 Dependencies

**Depends On:**
- PC-012-02A: Plugin Architecture Foundation
- All Epic-02B plugins (provides performance optimizations for them)

**External Dependencies:**
- `pandas>=2.0.0`
- `pyarrow>=12.0.0` (for Parquet)
- `redis>=4.5.0` (optional, for Redis backend)
- `memory_profiler>=0.61.0` (optional, for memory profiling)

**Blocks:**
- Production deployment (optimization is critical)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] FeatureCache implemented with memory/disk/Redis backends
- [ ] Cache key generation deterministic and robust
- [ ] Cache invalidation working correctly
- [ ] ExecutionProfiler tracking time and memory
- [ ] FeatureImportanceTracker persisting and aggregating
- [ ] Memory optimizer reducing memory footprint
- [ ] Unit tests pass with >85% coverage
- [ ] Performance benchmarks show >50% speedup with caching
- [ ] Thread-safety tests pass
- [ ] Documentation complete with usage examples
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Previous:** [PC-022-02B: Seasonality Calculation Plugin](PC-022-02B-seasonality-calculation-plugin.md)  
**Epic:** [Epic-02B: Advanced Feature Transformations](../epics/Epic-02B.md)
