"""Configuration models for performance optimization.

This module defines Pydantic configuration models for the performance
optimization system, including caching and profiling configurations.

Example:
    ```python
    from src.features.performance.config import CacheConfig, ProfilerConfig

    cache_config = CacheConfig(
        enable_cache=True,
        cache_backend="disk",
        max_cache_size_mb=2000,
        ttl_hours=48
    )

    profiler_config = ProfilerConfig(
        enable_profiling=True,
        profile_memory=True
    )
    ```
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class CacheConfig(BaseModel):
    """Configuration for feature caching system.

    Controls cache behavior including backend selection, size limits,
    TTL expiration, and compression settings.

    Attributes:
        enable_cache: Enable or disable feature caching globally.
        cache_backend: Storage backend ("memory", "disk", or "redis").
        cache_directory: Directory for disk cache storage.
        max_cache_size_mb: Maximum cache size in megabytes.
        ttl_hours: Time-to-live for cache entries in hours.
        compression: Enable cache compression (disk backend only).
        cache_key_prefix: Prefix for all cache keys.
        redis_url: Redis connection URL (for redis backend).
    """

    enable_cache: bool = Field(
        default=True, description="Enable feature caching"
    )
    cache_backend: Literal["memory", "disk", "redis"] = Field(
        default="disk", description="Cache storage backend"
    )
    cache_directory: Path = Field(
        default=Path(".cache/features"),
        description="Directory for disk cache",
    )
    max_cache_size_mb: int = Field(
        default=1000, description="Maximum cache size in MB", ge=1
    )
    ttl_hours: int = Field(
        default=24, description="Cache entry time-to-live in hours", ge=1
    )
    compression: bool = Field(
        default=True, description="Enable cache compression"
    )
    cache_key_prefix: str = Field(
        default="feature_cache", description="Prefix for cache keys"
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL (for redis backend)",
    )

    @field_validator("max_cache_size_mb")
    @classmethod
    def validate_cache_size(cls, v: int) -> int:
        """Validate cache size is positive.

        Args:
            v: Cache size value to validate.

        Returns:
            Validated cache size.

        Raises:
            ValueError: If cache size is not positive.
        """
        if v <= 0:
            raise ValueError("max_cache_size_mb must be positive")
        return v

    @field_validator("ttl_hours")
    @classmethod
    def validate_ttl(cls, v: int) -> int:
        """Validate TTL is positive.

        Args:
            v: TTL value to validate.

        Returns:
            Validated TTL.

        Raises:
            ValueError: If TTL is not positive.
        """
        if v <= 0:
            raise ValueError("ttl_hours must be positive")
        return v


class ProfilerConfig(BaseModel):
    """Configuration for execution profiling.

    Controls profiling behavior including memory tracking and report
    generation settings.

    Attributes:
        enable_profiling: Enable or disable execution profiling.
        profile_memory: Include memory usage in profiling.
        profile_directory: Directory for profiling results.
        save_reports: Save profiling reports to disk.
    """

    enable_profiling: bool = Field(
        default=False, description="Enable execution profiling"
    )
    profile_memory: bool = Field(
        default=True, description="Include memory profiling"
    )
    profile_directory: Path = Field(
        default=Path(".profiles"), description="Directory for profiling results"
    )
    save_reports: bool = Field(
        default=True, description="Save profiling reports to disk"
    )
