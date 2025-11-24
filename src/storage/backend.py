"""Abstract base class for storage backends.

This module defines the interface that all storage backends must implement,
providing a unified API for data storage operations regardless of the
underlying storage technology.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    This class defines the interface for all storage operations. Concrete
    implementations must provide functionality for both raw bytes and
    Parquet DataFrames.

    Example:
        ```python
        backend = S3StorageBackend(bucket="my-bucket", region="us-east-1")

        # Store data
        backend.put_object("data/test.txt", b"Hello, World!")

        # Retrieve data
        data = backend.get_object("data/test.txt")

        # Work with DataFrames
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        backend.put_parquet("data/test.parquet", df)
        loaded_df = backend.get_parquet("data/test.parquet")
        ```
    """

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return the backend type identifier.

        Returns:
            String identifier for this backend type (e.g., 's3', 'local').
        """
        ...

    @abstractmethod
    def list_objects(self, prefix: str = "") -> list[str]:
        """List objects with the given prefix.

        Args:
            prefix: Key prefix to filter objects. Empty string lists all objects.

        Returns:
            List of object keys matching the prefix.

        Raises:
            StorageError: If the listing operation fails.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def put_object(self, key: str, data: bytes) -> None:
        """Store data as an object.

        Args:
            key: The object key to store.
            data: The bytes to store.

        Raises:
            StorageError: If the storage operation fails.
        """
        ...

    @abstractmethod
    def delete_object(self, key: str) -> None:
        """Delete an object.

        Args:
            key: The object key to delete.

        Raises:
            ObjectNotFoundError: If the object doesn't exist.
            StorageError: If the deletion operation fails.
        """
        ...

    @abstractmethod
    def object_exists(self, key: str) -> bool:
        """Check if an object exists.

        Args:
            key: The object key to check.

        Returns:
            True if the object exists, False otherwise.

        Raises:
            StorageError: If the check operation fails.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def put_parquet(self, key: str, df: pd.DataFrame, **kwargs: Any) -> None:
        """Write a DataFrame as a Parquet file.

        Args:
            key: The object key to store the Parquet file.
            df: The DataFrame to store.
            **kwargs: Additional arguments passed to pyarrow.parquet.write_table.

        Raises:
            StorageError: If the write operation fails.
        """
        ...


class StorageError(Exception):
    """Base exception for storage operations.

    Attributes:
        message: Error message.
        key: The object key involved in the error (if applicable).
        cause: The underlying exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        key: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize StorageError.

        Args:
            message: Error message.
            key: The object key involved in the error.
            cause: The underlying exception that caused this error.
        """
        self.key = key
        self.cause = cause
        super().__init__(message)


class ObjectNotFoundError(StorageError):
    """Exception raised when an object is not found.

    Attributes:
        key: The object key that was not found.
    """

    def __init__(self, key: str, cause: Exception | None = None) -> None:
        """Initialize ObjectNotFoundError.

        Args:
            key: The object key that was not found.
            cause: The underlying exception that caused this error.
        """
        super().__init__(f"Object not found: {key}", key=key, cause=cause)
