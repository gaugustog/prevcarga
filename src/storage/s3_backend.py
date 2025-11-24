"""S3 storage backend implementation.

This module provides the S3StorageBackend class that implements
storage operations using AWS S3 via the boto3 library.
"""

import io
import logging
from typing import Any

import boto3
import pandas as pd
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from src.storage.backend import ObjectNotFoundError, StorageBackend, StorageError
from src.storage.config import S3Config

logger = logging.getLogger(__name__)


class S3StorageBackend(StorageBackend):
    """S3 storage backend using boto3.

    This backend stores data in AWS S3 buckets. It supports both raw bytes
    and Parquet DataFrames.

    Attributes:
        bucket: S3 bucket name.
        prefix: Key prefix for all operations.
        region: AWS region.

    Example:
        ```python
        backend = S3StorageBackend(
            bucket="my-bucket",
            region="sa-east-1",
            prefix="prevcarga/"
        )

        # Store and retrieve data
        backend.put_object("data/test.txt", b"Hello!")
        data = backend.get_object("data/test.txt")
        ```
    """

    def __init__(
        self,
        bucket: str,
        region: str = "sa-east-1",
        prefix: str = "",
        endpoint_url: str | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        max_retry_attempts: int = 3,
        retry_mode: str = "standard",
    ) -> None:
        """Initialize S3StorageBackend.

        Args:
            bucket: S3 bucket name.
            region: AWS region name.
            prefix: Key prefix for all operations.
            endpoint_url: Custom endpoint URL (for MinIO, LocalStack, etc.).
            connect_timeout: Connection timeout in seconds.
            read_timeout: Read timeout in seconds.
            max_retry_attempts: Maximum retry attempts.
            retry_mode: Retry mode ('standard' or 'adaptive').
        """
        self.bucket = bucket
        self.region = region
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._endpoint_url = endpoint_url

        # Configure boto3 client
        boto_config = BotoConfig(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries={
                "max_attempts": max_retry_attempts,
                "mode": retry_mode,
            },
        )

        client_kwargs: dict[str, Any] = {
            "region_name": region,
            "config": boto_config,
        }
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        self._client = boto3.client("s3", **client_kwargs)
        logger.info(
            "Initialized S3StorageBackend: bucket=%s, region=%s, prefix=%s",
            bucket,
            region,
            prefix,
        )

    @classmethod
    def from_config(cls, config: S3Config) -> "S3StorageBackend":
        """Create S3StorageBackend from configuration object.

        Args:
            config: S3Config instance with configuration.

        Returns:
            Configured S3StorageBackend instance.
        """
        return cls(
            bucket=config.bucket,
            region=config.region,
            prefix=config.prefix,
            endpoint_url=config.endpoint_url,
            connect_timeout=config.timeouts.connect_timeout,
            read_timeout=config.timeouts.read_timeout,
            max_retry_attempts=config.retry.max_attempts,
            retry_mode=config.retry.mode,
        )

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier.

        Returns:
            String 's3'.
        """
        return "s3"

    def _full_key(self, key: str) -> str:
        """Construct full S3 key with prefix.

        Args:
            key: The relative object key.

        Returns:
            Full S3 key including prefix.
        """
        key = key.lstrip("/")
        return f"{self.prefix}{key}"

    def list_objects(self, prefix: str = "") -> list[str]:
        """List objects with the given prefix.

        Args:
            prefix: Key prefix to filter objects.

        Returns:
            List of object keys matching the prefix (without backend prefix).

        Raises:
            StorageError: If the listing operation fails.
        """
        full_prefix = self._full_key(prefix)
        keys: list[str] = []

        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    # Remove the backend prefix from returned keys
                    key = obj["Key"]
                    if self.prefix and key.startswith(self.prefix):
                        key = key[len(self.prefix) :]
                    keys.append(key)

            logger.debug("Listed %d objects with prefix '%s'", len(keys), prefix)
            return keys

        except ClientError as e:
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
        full_key = self._full_key(key)

        try:
            response = self._client.get_object(Bucket=self.bucket, Key=full_key)
            data = response["Body"].read()
            logger.debug("Retrieved object '%s' (%d bytes)", key, len(data))
            return data

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                logger.warning("Object not found: %s", key)
                raise ObjectNotFoundError(key, cause=e) from e

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
        full_key = self._full_key(key)

        try:
            self._client.put_object(Bucket=self.bucket, Key=full_key, Body=data)
            logger.debug("Stored object '%s' (%d bytes)", key, len(data))

        except ClientError as e:
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
        full_key = self._full_key(key)

        # First check if object exists
        if not self.object_exists(key):
            raise ObjectNotFoundError(key)

        try:
            self._client.delete_object(Bucket=self.bucket, Key=full_key)
            logger.debug("Deleted object '%s'", key)

        except ClientError as e:
            logger.error("Failed to delete object '%s': %s", key, e)
            msg = f"Failed to delete object '{key}'"
            raise StorageError(msg, key=key, cause=e) from e

    def object_exists(self, key: str) -> bool:
        """Check if an object exists.

        Args:
            key: The object key to check.

        Returns:
            True if the object exists, False otherwise.

        Raises:
            StorageError: If the check operation fails.
        """
        full_key = self._full_key(key)

        try:
            self._client.head_object(Bucket=self.bucket, Key=full_key)
            return True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False

            logger.error("Failed to check object existence '%s': %s", key, e)
            msg = f"Failed to check object existence '{key}'"
            raise StorageError(msg, key=key, cause=e) from e

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
        try:
            data = self.get_object(key)
            buffer = io.BytesIO(data)
            df = pd.read_parquet(buffer)
            logger.debug(
                "Read Parquet '%s': %d rows, %d columns",
                key,
                len(df),
                len(df.columns),
            )
            return df

        except (ObjectNotFoundError, StorageError):
            raise
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
        try:
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False, **kwargs)
            buffer.seek(0)
            self.put_object(key, buffer.getvalue())
            logger.debug(
                "Wrote Parquet '%s': %d rows, %d columns",
                key,
                len(df),
                len(df.columns),
            )

        except StorageError:
            raise
        except Exception as e:
            logger.error("Failed to write Parquet '%s': %s", key, e)
            msg = f"Failed to write Parquet '{key}'"
            raise StorageError(msg, key=key, cause=e) from e
