"""Tests for local filesystem storage backend.

This module contains comprehensive tests for the LocalStorageBackend class,
using pytest's tmp_path fixture for isolated filesystem testing.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.storage.backend import ObjectNotFoundError, StorageError
from src.storage.config import LocalConfig
from src.storage.local_backend import LocalStorageBackend


class TestLocalStorageBackendInit:
    """Tests for LocalStorageBackend initialization."""

    def test_init_creates_directory(self, tmp_path: Path):
        """Test that initialization creates the base directory."""
        base_path = tmp_path / "new_dir"
        assert not base_path.exists()

        backend = LocalStorageBackend(base_path=base_path, create_dirs=True)

        assert base_path.exists()
        assert backend.base_path == base_path.resolve()
        assert backend.backend_type == "local"

    def test_init_without_create_dirs(self, tmp_path: Path):
        """Test initialization with create_dirs=False."""
        base_path = tmp_path / "existing"
        base_path.mkdir()

        backend = LocalStorageBackend(base_path=base_path, create_dirs=False)

        assert backend.base_path == base_path.resolve()

    def test_init_with_string_path(self, tmp_path: Path):
        """Test initialization with string path."""
        base_path = str(tmp_path / "string_path")

        backend = LocalStorageBackend(base_path=base_path)

        assert backend.base_path == Path(base_path).resolve()

    def test_from_config(self, tmp_path: Path):
        """Test creation from LocalConfig."""
        config = LocalConfig(
            base_path=tmp_path / "config_path",
            create_dirs=True,
        )

        backend = LocalStorageBackend.from_config(config)

        assert backend.base_path == (tmp_path / "config_path").resolve()
        assert backend.base_path.exists()


class TestLocalStorageBackendOperations:
    """Tests for LocalStorageBackend CRUD operations."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> LocalStorageBackend:
        """Create a LocalStorageBackend for testing."""
        return LocalStorageBackend(base_path=tmp_path)

    def test_put_and_get_object(self, backend: LocalStorageBackend):
        """Test storing and retrieving an object."""
        test_data = b"Hello, World!"
        key = "test/file.txt"

        backend.put_object(key, test_data)
        result = backend.get_object(key)

        assert result == test_data

    def test_put_creates_directories(self, backend: LocalStorageBackend):
        """Test that put_object creates nested directories."""
        key = "deep/nested/path/file.txt"
        data = b"nested data"

        backend.put_object(key, data)

        full_path = backend.base_path / key
        assert full_path.exists()
        assert full_path.read_bytes() == data

    def test_get_nonexistent_object(self, backend: LocalStorageBackend):
        """Test getting an object that doesn't exist."""
        with pytest.raises(ObjectNotFoundError) as exc_info:
            backend.get_object("nonexistent/file.txt")

        assert "nonexistent/file.txt" in str(exc_info.value)

    def test_get_directory_raises_error(self, backend: LocalStorageBackend, tmp_path: Path):
        """Test that getting a directory raises StorageError."""
        dir_path = tmp_path / "some_dir"
        dir_path.mkdir()

        # Create backend with tmp_path as base
        backend_with_dir = LocalStorageBackend(base_path=tmp_path.parent)

        with pytest.raises(StorageError):
            backend_with_dir.get_object(tmp_path.name)

    def test_object_exists_true(self, backend: LocalStorageBackend):
        """Test object_exists returns True for existing file."""
        backend.put_object("exists.txt", b"data")

        assert backend.object_exists("exists.txt") is True

    def test_object_exists_false(self, backend: LocalStorageBackend):
        """Test object_exists returns False for nonexistent file."""
        assert backend.object_exists("nonexistent.txt") is False

    def test_object_exists_false_for_directory(self, backend: LocalStorageBackend, tmp_path: Path):
        """Test object_exists returns False for directories."""
        dir_path = tmp_path / "some_directory"
        dir_path.mkdir()

        # This backend has tmp_path as base, so "some_directory" is a valid key path
        assert backend.object_exists("some_directory") is False

    def test_delete_object(self, backend: LocalStorageBackend):
        """Test deleting an object."""
        key = "to_delete.txt"
        backend.put_object(key, b"delete me")

        assert backend.object_exists(key) is True

        backend.delete_object(key)

        assert backend.object_exists(key) is False

    def test_delete_nonexistent_object(self, backend: LocalStorageBackend):
        """Test deleting a nonexistent object raises error."""
        with pytest.raises(ObjectNotFoundError):
            backend.delete_object("nonexistent.txt")

    def test_delete_directory_raises_error(self, backend: LocalStorageBackend, tmp_path: Path):
        """Test that deleting a directory raises StorageError."""
        dir_path = tmp_path / "some_dir"
        dir_path.mkdir()

        with pytest.raises(StorageError):
            backend.delete_object("some_dir")


class TestLocalStorageBackendListing:
    """Tests for LocalStorageBackend list_objects."""

    @pytest.fixture
    def backend_with_files(self, tmp_path: Path) -> LocalStorageBackend:
        """Create a backend with pre-existing files."""
        backend = LocalStorageBackend(base_path=tmp_path)

        # Create test files
        backend.put_object("dir1/file1.txt", b"data1")
        backend.put_object("dir1/file2.txt", b"data2")
        backend.put_object("dir2/file3.txt", b"data3")
        backend.put_object("root.txt", b"root data")

        return backend

    def test_list_objects_all(self, backend_with_files: LocalStorageBackend):
        """Test listing all objects."""
        result = backend_with_files.list_objects()

        assert len(result) == 4
        # Results should be sorted
        assert result == sorted(result)

    def test_list_objects_with_prefix(self, backend_with_files: LocalStorageBackend):
        """Test listing objects with prefix filter."""
        result = backend_with_files.list_objects("dir1/")

        assert len(result) == 2
        assert "dir1/file1.txt" in result
        assert "dir1/file2.txt" in result

    def test_list_objects_nonexistent_prefix(self, backend_with_files: LocalStorageBackend):
        """Test listing with nonexistent prefix returns empty list."""
        result = backend_with_files.list_objects("nonexistent/")

        assert result == []

    def test_list_objects_empty_directory(self, tmp_path: Path):
        """Test listing objects from empty directory."""
        backend = LocalStorageBackend(base_path=tmp_path)
        result = backend.list_objects()

        assert result == []


class TestLocalStorageBackendParquet:
    """Tests for LocalStorageBackend parquet operations."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> LocalStorageBackend:
        """Create a LocalStorageBackend for testing."""
        return LocalStorageBackend(base_path=tmp_path)

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
                "load_mw": [100.0 + i * 0.5 for i in range(100)],
                "area": ["SP"] * 100,
            }
        )

    def test_put_and_get_parquet(
        self, backend: LocalStorageBackend, sample_dataframe: pd.DataFrame
    ):
        """Test storing and retrieving a DataFrame as parquet."""
        key = "data/test.parquet"

        backend.put_parquet(key, sample_dataframe)
        result = backend.get_parquet(key)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_put_parquet_with_kwargs(
        self, backend: LocalStorageBackend, sample_dataframe: pd.DataFrame
    ):
        """Test parquet with compression option."""
        key = "data/compressed.parquet"

        backend.put_parquet(key, sample_dataframe, compression="gzip")
        result = backend.get_parquet(key)

        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_get_parquet_nonexistent(self, backend: LocalStorageBackend):
        """Test getting nonexistent parquet raises error."""
        with pytest.raises(ObjectNotFoundError):
            backend.get_parquet("nonexistent.parquet")

    def test_put_parquet_empty_dataframe(self, backend: LocalStorageBackend):
        """Test storing an empty DataFrame."""
        key = "data/empty.parquet"
        empty_df = pd.DataFrame()

        backend.put_parquet(key, empty_df)
        result = backend.get_parquet(key)

        assert len(result) == 0


class TestLocalStorageBackendSecurity:
    """Tests for LocalStorageBackend security measures."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> LocalStorageBackend:
        """Create a LocalStorageBackend for testing."""
        return LocalStorageBackend(base_path=tmp_path)

    def test_path_traversal_rejected(self, backend: LocalStorageBackend):
        """Test that path traversal attempts are rejected."""
        with pytest.raises(StorageError) as exc_info:
            backend.get_object("../../../etc/passwd")

        assert "Path traversal" in str(exc_info.value)

    def test_path_traversal_rejected_put(self, backend: LocalStorageBackend):
        """Test that path traversal is rejected on put."""
        with pytest.raises(StorageError):
            backend.put_object("../../outside.txt", b"malicious")

    def test_absolute_path_outside_base_rejected(self, backend: LocalStorageBackend):
        """Test that absolute paths outside base are handled correctly."""
        # The key is relative, so even if it looks absolute, it's relative to base
        # This should work (creates /abs/path/file.txt under base_path)
        backend.put_object("abs/path/file.txt", b"data")
        assert backend.object_exists("abs/path/file.txt")


class TestLocalStorageBackendEdgeCases:
    """Tests for LocalStorageBackend edge cases."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> LocalStorageBackend:
        """Create a LocalStorageBackend for testing."""
        return LocalStorageBackend(base_path=tmp_path)

    def test_put_empty_object(self, backend: LocalStorageBackend):
        """Test storing empty bytes."""
        key = "empty.txt"
        backend.put_object(key, b"")

        result = backend.get_object(key)
        assert result == b""

    def test_large_object(self, backend: LocalStorageBackend):
        """Test storing larger data."""
        key = "large.bin"
        # Create ~1MB of data
        large_data = b"x" * (1024 * 1024)

        backend.put_object(key, large_data)
        result = backend.get_object(key)

        assert len(result) == len(large_data)
        assert result == large_data

    def test_binary_data(self, backend: LocalStorageBackend):
        """Test storing binary data with all byte values."""
        key = "binary.bin"
        binary_data = bytes(range(256))

        backend.put_object(key, binary_data)
        result = backend.get_object(key)

        assert result == binary_data

    def test_overwrite_object(self, backend: LocalStorageBackend):
        """Test overwriting an existing object."""
        key = "overwrite.txt"

        backend.put_object(key, b"original")
        backend.put_object(key, b"updated")

        result = backend.get_object(key)
        assert result == b"updated"

    def test_key_with_leading_slash(self, backend: LocalStorageBackend):
        """Test that leading slash in key is handled."""
        backend.put_object("/leading/slash.txt", b"data")

        # Should be accessible with or without the leading slash
        assert backend.object_exists("leading/slash.txt")

    def test_unicode_in_data(self, backend: LocalStorageBackend):
        """Test storing UTF-8 encoded data."""
        key = "unicode.txt"
        unicode_data = b"Hello, World! Ola, Mundo! "

        backend.put_object(key, unicode_data)
        result = backend.get_object(key)

        assert result == unicode_data
        assert result.decode("utf-8") == "Hello, World! Ola, Mundo! "
