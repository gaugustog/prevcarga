"""Storage backend factory.

This module provides the StorageFactory class for creating storage backend
instances based on configuration or path prefixes.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

from src.storage.backend import StorageBackend
from src.storage.config import StorageConfig, load_storage_config
from src.storage.local_backend import LocalStorageBackend
from src.storage.s3_backend import S3StorageBackend

logger = logging.getLogger(__name__)


class StorageFactory:
    """Factory for creating storage backend instances.

    This factory provides multiple ways to create storage backends:
    - Explicit backend type with kwargs
    - From a YAML configuration file
    - Auto-detection from path prefix (s3://, file://, or relative)

    Example:
        ```python
        # Create from explicit type
        s3_backend = StorageFactory.create("s3", bucket="my-bucket", region="us-east-1")

        # Create from configuration file
        backend = StorageFactory.from_config("config/storage.yaml")

        # Auto-detect from path
        backend = StorageFactory.from_path("s3://my-bucket/data/")
        backend = StorageFactory.from_path("./local_data/")
        backend = StorageFactory.from_path("file:///absolute/path/")
        ```
    """

    # Mapping of backend types to their classes
    _backends: ClassVar[dict[str, type[StorageBackend]]] = {
        "s3": S3StorageBackend,
        "local": LocalStorageBackend,
    }

    @classmethod
    def register_backend(cls, name: str, backend_class: type[StorageBackend]) -> None:
        """Register a new backend type.

        Args:
            name: The name to register the backend under.
            backend_class: The backend class to register.
        """
        cls._backends[name] = backend_class
        logger.info("Registered storage backend: %s", name)

    @classmethod
    def available_backends(cls) -> list[str]:
        """Get list of available backend types.

        Returns:
            List of registered backend type names.
        """
        return list(cls._backends.keys())

    @classmethod
    def create(cls, backend_type: str, **kwargs: Any) -> StorageBackend:
        """Create a storage backend instance.

        Args:
            backend_type: Type of backend to create ('s3' or 'local').
            **kwargs: Arguments passed to the backend constructor.

        Returns:
            Configured storage backend instance.

        Raises:
            ValueError: If the backend type is unknown.

        Example:
            ```python
            # Create S3 backend
            backend = StorageFactory.create(
                "s3",
                bucket="my-bucket",
                region="sa-east-1",
                prefix="prevcarga/"
            )

            # Create local backend
            backend = StorageFactory.create(
                "local",
                base_path="./data",
                create_dirs=True
            )
            ```
        """
        if backend_type not in cls._backends:
            available = ", ".join(cls._backends.keys())
            msg = f"Unknown backend type: '{backend_type}'. Available: {available}"
            raise ValueError(msg)

        backend_class = cls._backends[backend_type]
        backend = backend_class(**kwargs)

        logger.info("Created %s storage backend", backend_type)
        return backend

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> StorageBackend:
        """Create a storage backend from a configuration file.

        Args:
            config_path: Path to YAML configuration file. If None, uses default
                        locations (see load_storage_config).

        Returns:
            Configured storage backend instance.

        Raises:
            FileNotFoundError: If the specified config file doesn't exist.
            ValueError: If the configuration is invalid.

        Example:
            ```python
            # Load from explicit path
            backend = StorageFactory.from_config("config/storage.yaml")

            # Load from default location
            backend = StorageFactory.from_config()
            ```
        """
        config = load_storage_config(config_path)
        return cls.from_storage_config(config)

    @classmethod
    def from_storage_config(cls, config: StorageConfig) -> StorageBackend:
        """Create a storage backend from a StorageConfig instance.

        Args:
            config: StorageConfig instance with configuration.

        Returns:
            Configured storage backend instance.

        Raises:
            ValueError: If the backend type in config is unknown.
        """
        backend: StorageBackend
        if config.backend == "s3":
            backend = S3StorageBackend.from_config(config.s3)
        elif config.backend == "local":
            backend = LocalStorageBackend.from_config(config.local)
        else:
            msg = f"Unknown backend type in config: '{config.backend}'"
            raise ValueError(msg)

        logger.info("Created %s storage backend from config", config.backend)
        return backend

    @classmethod
    def from_path(cls, path: str) -> StorageBackend:
        """Create a storage backend by auto-detecting from path prefix.

        Supports the following path formats:
        - s3://bucket/prefix/ -> S3StorageBackend
        - file:///absolute/path/ -> LocalStorageBackend
        - ./relative/path/ -> LocalStorageBackend
        - /absolute/path/ -> LocalStorageBackend

        Args:
            path: Path string with optional prefix.

        Returns:
            Configured storage backend instance.

        Raises:
            ValueError: If the path format is invalid.

        Example:
            ```python
            # S3 path
            backend = StorageFactory.from_path("s3://my-bucket/data/")

            # Local paths
            backend = StorageFactory.from_path("./local_data/")
            backend = StorageFactory.from_path("file:///home/user/data/")
            backend = StorageFactory.from_path("/absolute/path/to/data/")
            ```
        """
        if path.startswith("s3://"):
            return cls._create_s3_from_path(path)
        if path.startswith("file://"):
            return cls._create_local_from_file_uri(path)
        return cls._create_local_from_path(path)

    @classmethod
    def _create_s3_from_path(cls, path: str) -> S3StorageBackend:
        """Create S3 backend from s3:// URI.

        Args:
            path: S3 URI in format s3://bucket/prefix/

        Returns:
            Configured S3StorageBackend.

        Raises:
            ValueError: If the S3 URI format is invalid.
        """
        # Remove s3:// prefix
        path = path[5:]

        # Split into bucket and prefix
        parts = path.split("/", 1)
        bucket = parts[0]

        if not bucket:
            msg = "Invalid S3 URI: bucket name is required"
            raise ValueError(msg)

        prefix = parts[1] if len(parts) > 1 else ""

        logger.debug("Parsed S3 path: bucket=%s, prefix=%s", bucket, prefix)

        return S3StorageBackend(bucket=bucket, prefix=prefix)

    @classmethod
    def _create_local_from_file_uri(cls, path: str) -> LocalStorageBackend:
        """Create local backend from file:// URI.

        Args:
            path: File URI in format file:///absolute/path/

        Returns:
            Configured LocalStorageBackend.
        """
        # Remove file:// prefix
        local_path = path[7:]

        logger.debug("Parsed file URI: path=%s", local_path)

        return LocalStorageBackend(base_path=local_path)

    @classmethod
    def _create_local_from_path(cls, path: str) -> LocalStorageBackend:
        """Create local backend from regular path.

        Args:
            path: Local filesystem path (relative or absolute).

        Returns:
            Configured LocalStorageBackend.
        """
        logger.debug("Using local path: %s", path)

        return LocalStorageBackend(base_path=path)


def get_storage_backend(
    config_path: str | Path | None = None,
    path: str | None = None,
) -> StorageBackend:
    """Convenience function to get a storage backend.

    Creates a storage backend using either a config file or a path URI.
    If both are provided, path takes precedence. If neither is provided,
    loads from the default configuration.

    Args:
        config_path: Path to YAML configuration file.
        path: Path URI for auto-detection (s3://, file://, or local path).

    Returns:
        Configured storage backend instance.

    Example:
        ```python
        # From path
        backend = get_storage_backend(path="s3://my-bucket/data/")

        # From config
        backend = get_storage_backend(config_path="config/storage.yaml")

        # From default config
        backend = get_storage_backend()
        ```
    """
    if path:
        return StorageFactory.from_path(path)
    return StorageFactory.from_config(config_path)
