"""Storage backends for data access (S3 and local filesystem).

This module provides a unified interface for storage operations across
different backend types, including AWS S3 and local filesystem storage.

Example:
    ```python
    from src.storage import StorageFactory, get_storage_backend

    # Create backend from path auto-detection
    backend = StorageFactory.from_path("s3://my-bucket/data/")

    # Or use the convenience function
    backend = get_storage_backend(path="./local_data/")

    # Store and retrieve data
    backend.put_object("test.txt", b"Hello, World!")
    data = backend.get_object("test.txt")

    # Work with DataFrames
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    backend.put_parquet("data.parquet", df)
    loaded_df = backend.get_parquet("data.parquet")
    ```
"""

from src.storage.backend import (
    ObjectNotFoundError,
    StorageBackend,
    StorageError,
)
from src.storage.config import (
    LocalConfig,
    S3Config,
    S3Retry,
    S3Timeouts,
    StorageConfig,
    StoragePaths,
    load_storage_config,
)
from src.storage.factory import (
    StorageFactory,
    get_storage_backend,
)
from src.storage.local_backend import LocalStorageBackend
from src.storage.s3_backend import S3StorageBackend

__all__ = [
    "LocalConfig",
    "LocalStorageBackend",
    "ObjectNotFoundError",
    "S3Config",
    "S3Retry",
    "S3StorageBackend",
    "S3Timeouts",
    "StorageBackend",
    "StorageConfig",
    "StorageError",
    "StorageFactory",
    "StoragePaths",
    "get_storage_backend",
    "load_storage_config",
]
