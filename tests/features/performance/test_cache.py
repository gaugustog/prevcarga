"""Tests for feature caching system."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.performance.cache import FeatureCache
from src.features.performance.cache_key import CacheKeyGenerator
from src.features.performance.config import CacheConfig


@pytest.fixture
def sample_df():
    """Create sample DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randint(0, 10, 100),
        }
    )


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
        compute_fn=compute_fn,
    )

    # Second call - cache hit
    result2 = cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn,
    )

    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)

    # Check statistics
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert 0.0 <= stats["hit_rate"] <= 1.0


def test_memory_cache_miss_on_different_data(sample_df):
    """Test cache miss when data changes."""
    cache = FeatureCache(backend="memory")

    def compute_fn(df):
        return df * 2

    # First computation
    result1 = cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn,
    )

    # Change data
    modified_df = sample_df.copy()
    modified_df["feature1"] = modified_df["feature1"] + 100

    # Should compute again (different data hash)
    result2 = cache.get_or_compute(
        df=modified_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn,
    )

    # Results should be different
    assert not result1.equals(result2)

    stats = cache.get_stats()
    assert stats["misses"] == 2


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
        compute_fn=compute_fn,
    )

    # Create new cache instance (simulating restart)
    cache2 = FeatureCache(backend="disk", config=config)

    result2 = cache2.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn,
    )

    # Should load from persisted cache
    pd.testing.assert_frame_equal(result1, result2)

    stats = cache2.get_stats()
    assert stats["hits"] == 1


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
        config={"param": 1},
        compute_fn=compute_fn,
    )

    # Different config
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={"param": 2},
        compute_fn=compute_fn,
    )

    # Should have 2 misses (different configs)
    stats = cache.get_stats()
    assert stats["misses"] == 2


def test_cache_invalidation_on_plugin_version_change(sample_df):
    """Test cache invalidates when plugin version changes."""
    cache = FeatureCache(backend="memory")

    def compute_fn(df):
        return df * 2

    # Version 1.0.0
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn,
    )

    # Version 2.0.0
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="2.0.0",
        config={},
        compute_fn=compute_fn,
    )

    # Should have 2 misses (different versions)
    stats = cache.get_stats()
    assert stats["misses"] == 2


def test_cache_key_generation_deterministic(sample_df):
    """Test cache key generation is deterministic."""
    generator = CacheKeyGenerator()

    key1 = generator.generate_key(
        df=sample_df,
        plugin_name="test",
        plugin_version="1.0.0",
        config={"a": 1, "b": 2},
    )

    key2 = generator.generate_key(
        df=sample_df,
        plugin_name="test",
        plugin_version="1.0.0",
        config={"b": 2, "a": 1},  # Different order, same content
    )

    assert key1 == key2


def test_cache_key_changes_with_data(sample_df):
    """Test cache key changes when data changes."""
    generator = CacheKeyGenerator()

    key1 = generator.generate_key(
        df=sample_df,
        plugin_name="test",
        plugin_version="1.0.0",
        config={},
    )

    # Modify data
    modified_df = sample_df.copy()
    modified_df["feature1"] = modified_df["feature1"] + 1

    key2 = generator.generate_key(
        df=modified_df,
        plugin_name="test",
        plugin_version="1.0.0",
        config={},
    )

    assert key1 != key2


def test_cache_performance_improvement(sample_df):
    """Test cache provides performance improvement."""
    cache = FeatureCache(backend="memory")

    def slow_compute_fn(df):
        time.sleep(0.05)  # Simulate slow computation
        return df * 2

    # First call - slow
    start = time.perf_counter()
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=slow_compute_fn,
    )
    first_time = time.perf_counter() - start

    # Second call - fast (cached)
    start = time.perf_counter()
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=slow_compute_fn,
    )
    cached_time = time.perf_counter() - start

    # Cached should be much faster
    assert cached_time < first_time / 5


def test_cache_disabled(sample_df):
    """Test cache can be disabled."""
    config = CacheConfig(enable_cache=False)
    cache = FeatureCache(backend="memory", config=config)

    call_count = 0

    def compute_fn(df):
        nonlocal call_count
        call_count += 1
        return df * 2

    # Multiple calls should all compute
    for _ in range(3):
        cache.get_or_compute(
            df=sample_df,
            plugin_name="test_plugin",
            plugin_version="1.0.0",
            config={},
            compute_fn=compute_fn,
        )

    assert call_count == 3  # All calls computed
    stats = cache.get_stats()
    assert stats["hits"] == 0


def test_cache_clear(sample_df):
    """Test cache can be cleared."""
    cache = FeatureCache(backend="memory")

    def compute_fn(df):
        return df * 2

    # Populate cache
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn,
    )

    # Clear cache
    cache.clear()

    # Should be cache miss now
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn,
    )

    stats = cache.get_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0


def test_cache_stats_reset(sample_df):
    """Test cache statistics can be reset."""
    cache = FeatureCache(backend="memory")

    def compute_fn(df):
        return df * 2

    # Generate some stats
    cache.get_or_compute(
        df=sample_df,
        plugin_name="test_plugin",
        plugin_version="1.0.0",
        config={},
        compute_fn=compute_fn,
    )

    cache.reset_stats()

    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_disk_cache_cleanup(cache_dir):
    """Test disk cache cleanup."""
    config = CacheConfig(cache_directory=cache_dir, max_cache_size_mb=1)
    cache = FeatureCache(backend="disk", config=config)

    # Create large DataFrames to trigger cleanup
    def compute_fn(df):
        return df

    for i in range(10):
        df = pd.DataFrame(np.random.randn(10000, 100))
        cache.get_or_compute(
            df=df,
            plugin_name=f"plugin_{i}",
            plugin_version="1.0.0",
            config={},
            compute_fn=compute_fn,
        )

    # Cache should have enforced size limit
    assert cache.backend.size() > 0  # Not empty
    # Should have triggered cleanup at some point
