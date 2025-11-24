"""Storage configuration models using Pydantic.

This module provides Pydantic models for configuring storage backends
(S3 and local filesystem) with validation and sensible defaults.
"""

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class S3Timeouts(BaseModel):
    """Timeout configuration for S3 operations.

    Attributes:
        connect_timeout: Timeout for establishing connection in seconds.
        read_timeout: Timeout for read operations in seconds.
    """

    connect_timeout: float = Field(default=10.0, ge=1.0, le=300.0)
    read_timeout: float = Field(default=30.0, ge=1.0, le=600.0)


class S3Retry(BaseModel):
    """Retry configuration for S3 operations.

    Attributes:
        max_attempts: Maximum number of retry attempts.
        mode: Retry mode - 'standard' or 'adaptive'.
    """

    max_attempts: int = Field(default=3, ge=1, le=10)
    mode: Literal["standard", "adaptive"] = Field(default="standard")


class S3Config(BaseModel):
    """Configuration for S3 storage backend.

    Attributes:
        bucket: S3 bucket name.
        region: AWS region name.
        prefix: Key prefix for all objects.
        endpoint_url: Custom endpoint URL (for MinIO, LocalStack, etc.).
        timeouts: Timeout configuration.
        retry: Retry configuration.
    """

    bucket: str = Field(default="prevcarga-data", min_length=3, max_length=63)
    region: str = Field(default="sa-east-1")
    prefix: str = Field(default="")
    endpoint_url: str | None = Field(default=None)
    timeouts: S3Timeouts = Field(default_factory=S3Timeouts)
    retry: S3Retry = Field(default_factory=S3Retry)

    @field_validator("bucket")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate S3 bucket name follows AWS naming rules.

        Args:
            v: The bucket name to validate.

        Returns:
            The validated bucket name.

        Raises:
            ValueError: If bucket name is invalid.
        """
        # Basic S3 bucket naming rules
        if not v[0].isalnum():
            msg = "Bucket name must start with alphanumeric character"
            raise ValueError(msg)
        if not all(c.isalnum() or c in "-." for c in v):
            msg = "Bucket name can only contain alphanumeric characters, hyphens, and periods"
            raise ValueError(msg)
        return v.lower()

    @field_validator("prefix")
    @classmethod
    def normalize_prefix(cls, v: str) -> str:
        """Normalize S3 prefix by removing leading slash and ensuring trailing slash.

        Args:
            v: The prefix to normalize.

        Returns:
            The normalized prefix.
        """
        if not v:
            return ""
        v = v.lstrip("/")
        if v and not v.endswith("/"):
            v = f"{v}/"
        return v


class LocalConfig(BaseModel):
    """Configuration for local filesystem storage backend.

    Attributes:
        base_path: Base directory for all storage operations.
        create_dirs: Whether to create directories if they don't exist.
    """

    base_path: Path = Field(default=Path("./data"))
    create_dirs: bool = Field(default=True)

    @field_validator("base_path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """Expand environment variables and user home in path.

        Args:
            v: The path to expand.

        Returns:
            The expanded path.
        """
        if isinstance(v, str):
            v = os.path.expandvars(v)
            v = Path(v).expanduser()
        elif isinstance(v, Path):
            v = v.expanduser()
        return v


class StoragePaths(BaseModel):
    """Standard paths for different data types.

    Attributes:
        raw_data: Path for raw input data.
        features: Path for generated features.
        models: Path for trained models.
        results: Path for forecast results.
        cache: Path for cached data.
    """

    raw_data: str = Field(default="raw_data/")
    features: str = Field(default="features/")
    models: str = Field(default="models/")
    results: str = Field(default="results/")
    cache: str = Field(default="cache/")

    @field_validator("*", mode="before")
    @classmethod
    def ensure_trailing_slash(cls, v: str) -> str:
        """Ensure all paths have trailing slash.

        Args:
            v: The path to normalize.

        Returns:
            The path with trailing slash.
        """
        if v and not v.endswith("/"):
            return f"{v}/"
        return v


class StorageConfig(BaseModel):
    """Main storage configuration.

    Attributes:
        backend: Storage backend type ('s3' or 'local').
        paths: Standard paths configuration.
        s3: S3-specific configuration.
        local: Local filesystem configuration.
    """

    backend: Literal["s3", "local"] = Field(default="local")
    paths: StoragePaths = Field(default_factory=StoragePaths)
    s3: S3Config = Field(default_factory=S3Config)
    local: LocalConfig = Field(default_factory=LocalConfig)

    @model_validator(mode="after")
    def validate_backend_config(self) -> "StorageConfig":
        """Validate that the selected backend has proper configuration.

        Returns:
            The validated configuration.
        """
        if self.backend == "s3":
            logger.debug(
                "S3 backend configured with bucket '%s' in region '%s'",
                self.s3.bucket,
                self.s3.region,
            )
        else:
            logger.debug("Local backend configured with base path '%s'", self.local.base_path)
        return self


def load_storage_config(config_path: str | Path | None = None) -> StorageConfig:
    """Load storage configuration from a YAML file.

    If no path is provided, attempts to load from default locations in order:
    1. PREVCARGA_STORAGE_CONFIG environment variable
    2. ./config/storage.yaml
    3. ~/.config/prevcarga/storage.yaml

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        StorageConfig instance with loaded or default configuration.

    Raises:
        FileNotFoundError: If the specified config file doesn't exist.
        yaml.YAMLError: If the config file has invalid YAML.
        ValidationError: If the configuration is invalid.
    """
    # Determine config path
    if config_path is None:
        env_path = os.environ.get("PREVCARGA_STORAGE_CONFIG")
        if env_path:
            config_path = Path(env_path)
        else:
            default_paths = [
                Path("./config/storage.yaml"),
                Path.home() / ".config" / "prevcarga" / "storage.yaml",
            ]
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break

    # If no config found, return defaults
    if config_path is None:
        logger.info("No storage configuration found, using defaults")
        return StorageConfig()

    config_path = Path(config_path)

    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading storage configuration from %s", config_path)

    with config_path.open() as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        logger.warning("Empty configuration file, using defaults")
        return StorageConfig()

    return StorageConfig.model_validate(raw_config)
