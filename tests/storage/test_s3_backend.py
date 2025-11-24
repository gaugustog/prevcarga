"""Tests for S3 storage backend using moto for AWS mocking.

This module contains comprehensive tests for the S3StorageBackend class,
including all CRUD operations, parquet handling, and error cases.
"""

import os

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from src.storage.backend import ObjectNotFoundError
from src.storage.config import S3Config, S3Retry, S3Timeouts
from src.storage.s3_backend import S3StorageBackend

# Test constants
TEST_BUCKET = "test-bucket"
TEST_REGION = "us-east-1"
TEST_PREFIX = "test-prefix/"

# Mock credentials for moto - these are intentionally fake test values
_MOCK_AWS_CREDENTIAL = "testing"


@pytest.fixture
def aws_credentials():
    """Mock AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = _MOCK_AWS_CREDENTIAL
    os.environ["AWS_SECRET_ACCESS_KEY"] = _MOCK_AWS_CREDENTIAL
    os.environ["AWS_SECURITY_TOKEN"] = _MOCK_AWS_CREDENTIAL
    os.environ["AWS_SESSION_TOKEN"] = _MOCK_AWS_CREDENTIAL
    os.environ["AWS_DEFAULT_REGION"] = TEST_REGION


@pytest.fixture
def s3_client(aws_credentials):
    """Create a mocked S3 client and bucket."""
    with mock_aws():
        client = boto3.client("s3", region_name=TEST_REGION)
        client.create_bucket(Bucket=TEST_BUCKET)
        yield client


@pytest.fixture
def s3_backend(s3_client):
    """Create an S3StorageBackend for testing."""
    return S3StorageBackend(
        bucket=TEST_BUCKET,
        region=TEST_REGION,
        prefix="",
    )


@pytest.fixture
def s3_backend_with_prefix(s3_client):
    """Create an S3StorageBackend with prefix for testing."""
    return S3StorageBackend(
        bucket=TEST_BUCKET,
        region=TEST_REGION,
        prefix=TEST_PREFIX,
    )


class TestS3StorageBackendInit:
    """Tests for S3StorageBackend initialization."""

    def test_init_basic(self, s3_client):
        """Test basic initialization."""
        backend = S3StorageBackend(bucket=TEST_BUCKET, region=TEST_REGION)

        assert backend.bucket == TEST_BUCKET
        assert backend.region == TEST_REGION
        assert backend.prefix == ""
        assert backend.backend_type == "s3"

    def test_init_with_prefix(self, s3_client):
        """Test initialization with prefix."""
        backend = S3StorageBackend(
            bucket=TEST_BUCKET,
            region=TEST_REGION,
            prefix="my/prefix",
        )

        # Prefix should have trailing slash
        assert backend.prefix == "my/prefix/"

    def test_init_with_trailing_slash_prefix(self, s3_client):
        """Test prefix normalization when already has trailing slash."""
        backend = S3StorageBackend(
            bucket=TEST_BUCKET,
            region=TEST_REGION,
            prefix="my/prefix/",
        )

        assert backend.prefix == "my/prefix/"

    def test_from_config(self, s3_client):
        """Test creation from S3Config."""
        config = S3Config(
            bucket="config-bucket",
            region="eu-west-1",
            prefix="data/",
            timeouts=S3Timeouts(connect_timeout=15.0, read_timeout=45.0),
            retry=S3Retry(max_attempts=5, mode="adaptive"),
        )

        # Create bucket for this test
        s3_client.create_bucket(
            Bucket="config-bucket",
            CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
        )

        backend = S3StorageBackend.from_config(config)

        assert backend.bucket == "config-bucket"
        assert backend.region == "eu-west-1"
        assert backend.prefix == "data/"


class TestS3StorageBackendOperations:
    """Tests for S3StorageBackend CRUD operations."""

    def test_put_and_get_object(self, s3_backend):
        """Test storing and retrieving an object."""
        test_data = b"Hello, World!"
        key = "test/file.txt"

        s3_backend.put_object(key, test_data)
        result = s3_backend.get_object(key)

        assert result == test_data

    def test_put_and_get_object_with_prefix(self, s3_backend_with_prefix):
        """Test storing and retrieving with prefix."""
        test_data = b"Prefixed data"
        key = "nested/file.txt"

        s3_backend_with_prefix.put_object(key, test_data)
        result = s3_backend_with_prefix.get_object(key)

        assert result == test_data

    def test_get_nonexistent_object(self, s3_backend):
        """Test getting an object that doesn't exist."""
        with pytest.raises(ObjectNotFoundError) as exc_info:
            s3_backend.get_object("nonexistent/file.txt")

        assert "nonexistent/file.txt" in str(exc_info.value)

    def test_object_exists_true(self, s3_backend):
        """Test object_exists returns True for existing object."""
        s3_backend.put_object("exists.txt", b"data")

        assert s3_backend.object_exists("exists.txt") is True

    def test_object_exists_false(self, s3_backend):
        """Test object_exists returns False for nonexistent object."""
        assert s3_backend.object_exists("nonexistent.txt") is False

    def test_delete_object(self, s3_backend):
        """Test deleting an object."""
        key = "to_delete.txt"
        s3_backend.put_object(key, b"delete me")

        assert s3_backend.object_exists(key) is True

        s3_backend.delete_object(key)

        assert s3_backend.object_exists(key) is False

    def test_delete_nonexistent_object(self, s3_backend):
        """Test deleting a nonexistent object raises error."""
        with pytest.raises(ObjectNotFoundError):
            s3_backend.delete_object("nonexistent.txt")

    def test_list_objects_empty(self, s3_backend):
        """Test listing objects from empty bucket."""
        result = s3_backend.list_objects()

        assert result == []

    def test_list_objects(self, s3_backend):
        """Test listing objects."""
        # Store some test objects
        s3_backend.put_object("dir1/file1.txt", b"data1")
        s3_backend.put_object("dir1/file2.txt", b"data2")
        s3_backend.put_object("dir2/file3.txt", b"data3")

        result = s3_backend.list_objects()

        assert len(result) == 3
        assert "dir1/file1.txt" in result
        assert "dir1/file2.txt" in result
        assert "dir2/file3.txt" in result

    def test_list_objects_with_prefix(self, s3_backend):
        """Test listing objects with prefix filter."""
        s3_backend.put_object("dir1/file1.txt", b"data1")
        s3_backend.put_object("dir1/file2.txt", b"data2")
        s3_backend.put_object("dir2/file3.txt", b"data3")

        result = s3_backend.list_objects("dir1/")

        assert len(result) == 2
        assert "dir1/file1.txt" in result
        assert "dir1/file2.txt" in result
        assert "dir2/file3.txt" not in result

    def test_list_objects_with_backend_prefix(self, s3_backend_with_prefix):
        """Test listing objects respects backend prefix."""
        s3_backend_with_prefix.put_object("file1.txt", b"data1")
        s3_backend_with_prefix.put_object("file2.txt", b"data2")

        result = s3_backend_with_prefix.list_objects()

        # Keys should be returned without the backend prefix
        assert len(result) == 2
        assert "file1.txt" in result
        assert "file2.txt" in result


class TestS3StorageBackendParquet:
    """Tests for S3StorageBackend parquet operations."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
                "load_mw": [100.0 + i * 0.5 for i in range(100)],
                "area": ["SP"] * 100,
            }
        )

    def test_put_and_get_parquet(self, s3_backend, sample_dataframe):
        """Test storing and retrieving a DataFrame as parquet."""
        key = "data/test.parquet"

        s3_backend.put_parquet(key, sample_dataframe)
        result = s3_backend.get_parquet(key)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        assert list(result.columns) == list(sample_dataframe.columns)
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_put_parquet_with_kwargs(self, s3_backend, sample_dataframe):
        """Test parquet with compression option."""
        key = "data/compressed.parquet"

        s3_backend.put_parquet(key, sample_dataframe, compression="gzip")
        result = s3_backend.get_parquet(key)

        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_get_parquet_nonexistent(self, s3_backend):
        """Test getting nonexistent parquet raises error."""
        with pytest.raises(ObjectNotFoundError):
            s3_backend.get_parquet("nonexistent.parquet")

    def test_put_parquet_empty_dataframe(self, s3_backend):
        """Test storing an empty DataFrame."""
        key = "data/empty.parquet"
        empty_df = pd.DataFrame()

        s3_backend.put_parquet(key, empty_df)
        result = s3_backend.get_parquet(key)

        assert len(result) == 0


class TestS3StorageBackendEdgeCases:
    """Tests for S3StorageBackend edge cases and error handling."""

    def test_put_empty_object(self, s3_backend):
        """Test storing empty bytes."""
        key = "empty.txt"
        s3_backend.put_object(key, b"")

        result = s3_backend.get_object(key)
        assert result == b""

    def test_large_object(self, s3_backend):
        """Test storing larger data."""
        key = "large.bin"
        # Create ~1MB of data
        large_data = b"x" * (1024 * 1024)

        s3_backend.put_object(key, large_data)
        result = s3_backend.get_object(key)

        assert len(result) == len(large_data)
        assert result == large_data

    def test_binary_data(self, s3_backend):
        """Test storing binary data with special characters."""
        key = "binary.bin"
        binary_data = bytes(range(256))

        s3_backend.put_object(key, binary_data)
        result = s3_backend.get_object(key)

        assert result == binary_data

    def test_unicode_key(self, s3_backend):
        """Test object key with special characters."""
        # Note: S3 supports most UTF-8 characters in keys
        key = "data/file-2024-01-01_v1.txt"
        data = b"test data"

        s3_backend.put_object(key, data)
        result = s3_backend.get_object(key)

        assert result == data

    def test_overwrite_object(self, s3_backend):
        """Test overwriting an existing object."""
        key = "overwrite.txt"

        s3_backend.put_object(key, b"original")
        s3_backend.put_object(key, b"updated")

        result = s3_backend.get_object(key)
        assert result == b"updated"

    def test_nested_keys(self, s3_backend):
        """Test deeply nested key paths."""
        key = "a/b/c/d/e/f/file.txt"
        data = b"nested data"

        s3_backend.put_object(key, data)
        result = s3_backend.get_object(key)

        assert result == data

    def test_key_with_leading_slash(self, s3_backend):
        """Test that leading slash is stripped from key."""
        s3_backend.put_object("/leading/slash.txt", b"data")

        # Should be accessible without the leading slash
        result = s3_backend.get_object("leading/slash.txt")
        assert result == b"data"
