"""Tests for StorageFactory.

This module contains tests for the StorageFactory class, including
backend creation from explicit types, configuration files, and path URIs.
"""

import os
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from src.storage.backend import StorageBackend
from src.storage.config import LocalConfig, S3Config, StorageConfig
from src.storage.factory import StorageFactory, get_storage_backend
from src.storage.local_backend import LocalStorageBackend
from src.storage.s3_backend import S3StorageBackend

# Test constants
TEST_BUCKET = "test-bucket"
TEST_REGION = "us-east-1"

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


class TestStorageFactoryCreate:
    """Tests for StorageFactory.create() method."""

    def test_create_local_backend(self, tmp_path: Path):
        """Test creating a local backend."""
        backend = StorageFactory.create(
            "local",
            base_path=tmp_path,
            create_dirs=True,
        )

        assert isinstance(backend, LocalStorageBackend)
        assert backend.backend_type == "local"
        assert backend.base_path == tmp_path.resolve()

    def test_create_s3_backend(self, s3_client):
        """Test creating an S3 backend."""
        backend = StorageFactory.create(
            "s3",
            bucket=TEST_BUCKET,
            region=TEST_REGION,
        )

        assert isinstance(backend, S3StorageBackend)
        assert backend.backend_type == "s3"
        assert backend.bucket == TEST_BUCKET

    def test_create_unknown_backend_raises_error(self):
        """Test that unknown backend type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            StorageFactory.create("unknown_backend")

        assert "Unknown backend type" in str(exc_info.value)
        assert "unknown_backend" in str(exc_info.value)

    def test_available_backends(self):
        """Test listing available backends."""
        backends = StorageFactory.available_backends()

        assert "s3" in backends
        assert "local" in backends

    def test_register_custom_backend(self, tmp_path: Path):
        """Test registering a custom backend."""

        class CustomBackend(StorageBackend):
            def __init__(self, **kwargs):
                pass

            @property
            def backend_type(self) -> str:
                return "custom"

            def list_objects(self, prefix: str = "") -> list[str]:
                return []

            def get_object(self, key: str) -> bytes:
                return b""

            def put_object(self, key: str, data: bytes) -> None:
                pass

            def delete_object(self, key: str) -> None:
                pass

            def object_exists(self, key: str) -> bool:
                return False

            def get_parquet(self, key: str):
                import pandas as pd

                return pd.DataFrame()

            def put_parquet(self, key: str, df, **kwargs) -> None:
                pass

        # Register the custom backend
        StorageFactory.register_backend("custom", CustomBackend)

        assert "custom" in StorageFactory.available_backends()

        backend = StorageFactory.create("custom")
        assert backend.backend_type == "custom"


class TestStorageFactoryFromConfig:
    """Tests for StorageFactory.from_config() method."""

    @pytest.fixture
    def local_config_file(self, tmp_path: Path) -> Path:
        """Create a local storage configuration file."""
        config_content = f"""
backend: local
local:
  base_path: {tmp_path}/data
  create_dirs: true
paths:
  raw_data: raw/
  features: features/
"""
        config_file = tmp_path / "storage.yaml"
        config_file.write_text(config_content)
        return config_file

    @pytest.fixture
    def s3_config_file(self, tmp_path: Path) -> Path:
        """Create an S3 storage configuration file."""
        config_content = f"""
backend: s3
s3:
  bucket: {TEST_BUCKET}
  region: {TEST_REGION}
  prefix: data/
  timeouts:
    connect_timeout: 15.0
    read_timeout: 45.0
  retry:
    max_attempts: 5
    mode: adaptive
"""
        config_file = tmp_path / "s3_storage.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_from_config_local(self, local_config_file: Path, tmp_path: Path):
        """Test creating backend from local config file."""
        backend = StorageFactory.from_config(local_config_file)

        assert isinstance(backend, LocalStorageBackend)
        assert backend.backend_type == "local"

    def test_from_config_s3(self, s3_config_file: Path, s3_client):
        """Test creating backend from S3 config file."""
        backend = StorageFactory.from_config(s3_config_file)

        assert isinstance(backend, S3StorageBackend)
        assert backend.bucket == TEST_BUCKET
        assert backend.region == TEST_REGION
        assert backend.prefix == "data/"

    def test_from_config_nonexistent_file(self, tmp_path: Path):
        """Test that nonexistent config file raises error."""
        with pytest.raises(FileNotFoundError):
            StorageFactory.from_config(tmp_path / "nonexistent.yaml")

    def test_from_storage_config_local(self, tmp_path: Path):
        """Test creating backend from StorageConfig object."""
        config = StorageConfig(
            backend="local",
            local=LocalConfig(
                base_path=tmp_path / "data",
                create_dirs=True,
            ),
        )

        backend = StorageFactory.from_storage_config(config)

        assert isinstance(backend, LocalStorageBackend)

    def test_from_storage_config_s3(self, s3_client):
        """Test creating S3 backend from StorageConfig object."""
        config = StorageConfig(
            backend="s3",
            s3=S3Config(
                bucket=TEST_BUCKET,
                region=TEST_REGION,
            ),
        )

        backend = StorageFactory.from_storage_config(config)

        assert isinstance(backend, S3StorageBackend)


class TestStorageFactoryFromPath:
    """Tests for StorageFactory.from_path() method."""

    def test_from_path_s3_basic(self, s3_client):
        """Test creating S3 backend from s3:// URI."""
        backend = StorageFactory.from_path(f"s3://{TEST_BUCKET}/")

        assert isinstance(backend, S3StorageBackend)
        assert backend.bucket == TEST_BUCKET
        assert backend.prefix == ""

    def test_from_path_s3_with_prefix(self, s3_client):
        """Test creating S3 backend from s3:// URI with prefix."""
        backend = StorageFactory.from_path(f"s3://{TEST_BUCKET}/data/prevcarga/")

        assert isinstance(backend, S3StorageBackend)
        assert backend.bucket == TEST_BUCKET
        assert backend.prefix == "data/prevcarga/"

    def test_from_path_s3_invalid_no_bucket(self, s3_client):
        """Test that s3:// without bucket raises error."""
        with pytest.raises(ValueError) as exc_info:
            StorageFactory.from_path("s3:///")

        assert "bucket name is required" in str(exc_info.value)

    def test_from_path_file_uri(self, tmp_path: Path):
        """Test creating local backend from file:// URI."""
        backend = StorageFactory.from_path(f"file://{tmp_path}")

        assert isinstance(backend, LocalStorageBackend)
        assert backend.base_path == tmp_path.resolve()

    def test_from_path_relative(self, tmp_path: Path):
        """Test creating local backend from relative path."""
        # Change to tmp_path and use relative
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            backend = StorageFactory.from_path("./data")

            assert isinstance(backend, LocalStorageBackend)
        finally:
            os.chdir(original_cwd)

    def test_from_path_absolute(self, tmp_path: Path):
        """Test creating local backend from absolute path."""
        backend = StorageFactory.from_path(str(tmp_path / "absolute"))

        assert isinstance(backend, LocalStorageBackend)

    def test_from_path_with_trailing_slash(self, s3_client):
        """Test that paths with/without trailing slashes are handled."""
        backend1 = StorageFactory.from_path(f"s3://{TEST_BUCKET}/prefix")
        backend2 = StorageFactory.from_path(f"s3://{TEST_BUCKET}/prefix/")

        # Both should work
        assert isinstance(backend1, S3StorageBackend)
        assert isinstance(backend2, S3StorageBackend)


class TestGetStorageBackend:
    """Tests for get_storage_backend convenience function."""

    def test_with_path_s3(self, s3_client):
        """Test get_storage_backend with S3 path."""
        backend = get_storage_backend(path=f"s3://{TEST_BUCKET}/data/")

        assert isinstance(backend, S3StorageBackend)

    def test_with_path_local(self, tmp_path: Path):
        """Test get_storage_backend with local path."""
        backend = get_storage_backend(path=str(tmp_path))

        assert isinstance(backend, LocalStorageBackend)

    def test_with_config_path(self, tmp_path: Path):
        """Test get_storage_backend with config path."""
        config_content = f"""
backend: local
local:
  base_path: {tmp_path}/data
  create_dirs: true
"""
        config_file = tmp_path / "storage.yaml"
        config_file.write_text(config_content)

        backend = get_storage_backend(config_path=config_file)

        assert isinstance(backend, LocalStorageBackend)

    def test_path_takes_precedence_over_config(self, tmp_path: Path, s3_client):
        """Test that path takes precedence over config_path."""
        # Create a config that would use local
        config_content = f"""
backend: local
local:
  base_path: {tmp_path}/data
"""
        config_file = tmp_path / "storage.yaml"
        config_file.write_text(config_content)

        # But provide an S3 path
        backend = get_storage_backend(
            config_path=config_file,
            path=f"s3://{TEST_BUCKET}/",
        )

        # Should use S3 from path, not local from config
        assert isinstance(backend, S3StorageBackend)


class TestStorageFactoryIntegration:
    """Integration tests verifying factory creates working backends."""

    def test_local_backend_works(self, tmp_path: Path):
        """Test that created local backend can perform operations."""
        backend = StorageFactory.create("local", base_path=tmp_path)

        # Should be able to use the backend
        backend.put_object("test.txt", b"Hello, World!")
        data = backend.get_object("test.txt")

        assert data == b"Hello, World!"
        assert backend.object_exists("test.txt")

    def test_s3_backend_works(self, s3_client):
        """Test that created S3 backend can perform operations."""
        backend = StorageFactory.create(
            "s3",
            bucket=TEST_BUCKET,
            region=TEST_REGION,
        )

        # Should be able to use the backend
        backend.put_object("test.txt", b"Hello, S3!")
        data = backend.get_object("test.txt")

        assert data == b"Hello, S3!"
        assert backend.object_exists("test.txt")

    def test_from_path_backend_works(self, tmp_path: Path):
        """Test that backend from path can perform operations."""
        backend = StorageFactory.from_path(str(tmp_path))

        backend.put_object("path_test.txt", b"Path test data")
        assert backend.object_exists("path_test.txt")
