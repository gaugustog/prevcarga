"""Local filesystem storage backend implementation.

This module provides the LocalStorageBackend class that implements
storage operations using the local filesystem via pathlib.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.storage.backend import ObjectNotFoundError, StorageBackend, StorageError
from src.storage.config import LocalConfig

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend.

    This backend stores data on the local filesystem. It supports both raw bytes
    and Parquet DataFrames, mirroring the S3 backend API for seamless switching.

    Attributes:
        base_path: Base directory for all storage operations.

    Example:
        ```python
        backend = LocalStorageBackend(base_path="./data")

        # Store and retrieve data
        backend.put_object("raw/test.txt", b"Hello!")
        data = backend.get_object("raw/test.txt")

        # Work with DataFrames
        df = pd.DataFrame({"a": [1, 2, 3]})
        backend.put_parquet("features/test.parquet", df)
        ```
    """

    def __init__(self, base_path: str | Path, create_dirs: bool = True) -> None:
        """Initialize LocalStorageBackend.

        Args:
            base_path: Base directory for all storage operations.
            create_dirs: If True, create directories as needed during put operations.
        """
        self.base_path = Path(base_path).resolve()
        self._create_dirs = create_dirs

        if create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized LocalStorageBackend: base_path=%s, create_dirs=%s",
            self.base_path,
            create_dirs,
        )

    @classmethod
    def from_config(cls, config: LocalConfig) -> "LocalStorageBackend":
        """Create LocalStorageBackend from configuration object.

        Args:
            config: LocalConfig instance with configuration.

        Returns:
            Configured LocalStorageBackend instance.
        """
        return cls(
            base_path=config.base_path,
            create_dirs=config.create_dirs,
        )

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier.

        Returns:
            String 'local'.
        """
        return "local"

    def _full_path(self, key: str) -> Path:
        """Construct full filesystem path from key.

        Args:
            key: The relative object key.

        Returns:
            Full filesystem path.

        Raises:
            StorageError: If the resolved path is outside base_path (security check).
        """
        key = key.lstrip("/")
        full_path = (self.base_path / key).resolve()

        # Security check: ensure path is within base_path
        try:
            full_path.relative_to(self.base_path)
        except ValueError as e:
            msg = f"Path traversal attempt detected: {key}"
            logger.error(msg)
            raise StorageError(msg, key=key, cause=e) from e

        return full_path

    def list_objects(self, prefix: str = "") -> list[str]:
        """List objects with the given prefix.

        Args:
            prefix: Key prefix to filter objects. Empty string lists all objects.

        Returns:
            List of object keys matching the prefix (relative to base_path).

        Raises:
            StorageError: If the listing operation fails.
        """
        try:
            search_path = self._full_path(prefix) if prefix else self.base_path

            if not search_path.exists():
                logger.debug("Path does not exist for listing: %s", search_path)
                return []

            keys: list[str] = []

            if search_path.is_file():
                # If prefix points to a file, return just that file
                rel_path = search_path.relative_to(self.base_path)
                keys.append(str(rel_path))
            else:
                # Recursively list all files in directory
                for path in search_path.rglob("*"):
                    if path.is_file():
                        rel_path = path.relative_to(self.base_path)
                        keys.append(str(rel_path))

            logger.debug("Listed %d objects with prefix '%s'", len(keys), prefix)
            return sorted(keys)

        except StorageError:
            raise
        except Exception as e:
            logger.error("Failed to list objects with prefix '%s': %s", prefix, e)
            msg = f"Failed to list objects with prefix '{prefix}'"
            raise StorageError(msg, key=prefix, cause=e) from e

    def get_object(self, key: str) -> bytes:
        """Retrieve an object's contents as bytes.

        Args:
            key: The object key to retrieve.

        Returns:
            The object contents as bytes.

        Raises:
            ObjectNotFoundError: If the object doesn't exist.
            StorageError: If the retrieval operation fails.
        """
        full_path = self._full_path(key)

        if not full_path.exists():
            logger.warning("Object not found: %s", key)
            raise ObjectNotFoundError(key)

        if not full_path.is_file():
            msg = f"Path is not a file: {key}"
            logger.error(msg)
            raise StorageError(msg, key=key)

        try:
            data = full_path.read_bytes()
            logger.debug("Retrieved object '%s' (%d bytes)", key, len(data))
            return data

        except PermissionError as e:
            logger.error("Permission denied reading '%s': %s", key, e)
            msg = f"Permission denied reading '{key}'"
            raise StorageError(msg, key=key, cause=e) from e
        except OSError as e:
            logger.error("Failed to get object '%s': %s", key, e)
            msg = f"Failed to get object '{key}'"
            raise StorageError(msg, key=key, cause=e) from e

    def put_object(self, key: str, data: bytes) -> None:
        """Store data as an object.

        Args:
            key: The object key to store.
            data: The bytes to store.

        Raises:
            StorageError: If the storage operation fails.
        """
        full_path = self._full_path(key)

        try:
            if self._create_dirs:
                full_path.parent.mkdir(parents=True, exist_ok=True)

            full_path.write_bytes(data)
            logger.debug("Stored object '%s' (%d bytes)", key, len(data))

        except PermissionError as e:
            logger.error("Permission denied writing '%s': %s", key, e)
            msg = f"Permission denied writing '{key}'"
            raise StorageError(msg, key=key, cause=e) from e
        except OSError as e:
            logger.error("Failed to put object '%s': %s", key, e)
            msg = f"Failed to put object '{key}'"
            raise StorageError(msg, key=key, cause=e) from e

    def delete_object(self, key: str) -> None:
        """Delete an object.

        Args:
            key: The object key to delete.

        Raises:
            ObjectNotFoundError: If the object doesn't exist.
            StorageError: If the deletion operation fails.
        """
        full_path = self._full_path(key)

        if not full_path.exists():
            raise ObjectNotFoundError(key)

        if not full_path.is_file():
            msg = f"Path is not a file: {key}"
            logger.error(msg)
            raise StorageError(msg, key=key)

        try:
            full_path.unlink()
            logger.debug("Deleted object '%s'", key)

        except PermissionError as e:
            logger.error("Permission denied deleting '%s': %s", key, e)
            msg = f"Permission denied deleting '{key}'"
            raise StorageError(msg, key=key, cause=e) from e
        except OSError as e:
            logger.error("Failed to delete object '%s': %s", key, e)
            msg = f"Failed to delete object '{key}'"
            raise StorageError(msg, key=key, cause=e) from e

    def object_exists(self, key: str) -> bool:
        """Check if an object exists.

        Args:
            key: The object key to check.

        Returns:
            True if the object exists and is a file, False otherwise.

        Raises:
            StorageError: If the check operation fails (e.g., path traversal).
        """
        try:
            full_path = self._full_path(key)
            return full_path.is_file()
        except StorageError:
            raise

    def get_parquet(self, key: str) -> pd.DataFrame:
        """Read a Parquet file and return as DataFrame.

        Args:
            key: The object key of the Parquet file.

        Returns:
            DataFrame containing the Parquet data.

        Raises:
            ObjectNotFoundError: If the object doesn't exist.
            StorageError: If the read operation fails.
        """
        full_path = self._full_path(key)

        if not full_path.exists():
            logger.warning("Parquet file not found: %s", key)
            raise ObjectNotFoundError(key)

        try:
            df = pd.read_parquet(full_path)
            logger.debug(
                "Read Parquet '%s': %d rows, %d columns",
                key,
                len(df),
                len(df.columns),
            )
            return df

        except Exception as e:
            logger.error("Failed to read Parquet '%s': %s", key, e)
            msg = f"Failed to read Parquet '{key}'"
            raise StorageError(msg, key=key, cause=e) from e

    def put_parquet(self, key: str, df: pd.DataFrame, **kwargs: Any) -> None:
        """Write a DataFrame as a Parquet file.

        Args:
            key: The object key to store the Parquet file.
            df: The DataFrame to store.
            **kwargs: Additional arguments passed to DataFrame.to_parquet().

        Raises:
            StorageError: If the write operation fails.
        """
        full_path = self._full_path(key)

        try:
            if self._create_dirs:
                full_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_parquet(full_path, index=False, **kwargs)
            logger.debug(
                "Wrote Parquet '%s': %d rows, %d columns",
                key,
                len(df),
                len(df.columns),
            )

        except Exception as e:
            logger.error("Failed to write Parquet '%s': %s", key, e)
            msg = f"Failed to write Parquet '{key}'"
            raise StorageError(msg, key=key, cause=e) from e
