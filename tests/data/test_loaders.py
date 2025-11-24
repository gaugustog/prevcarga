"""Tests for data loaders module.

This module contains comprehensive tests for the DataLoader class and
DateRangeGenerator utility, including tests for both local filesystem
and S3 storage backends using moto for AWS mocking.
"""

import os
from datetime import date
from pathlib import Path

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from src.data.loaders import OUTPUT_COLUMNS, DataLoader
from src.data.utils import DateRangeGenerator
from src.storage import LocalStorageBackend, S3StorageBackend

# Test constants
TEST_BUCKET = "test-data-bucket"
TEST_REGION = "us-east-1"

# Mock credentials for moto - these are intentionally fake test values
_MOCK_AWS_CREDENTIAL = "testing"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def aws_credentials():
    """Mock AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = _MOCK_AWS_CREDENTIAL
    os.environ["AWS_SECRET_ACCESS_KEY"] = _MOCK_AWS_CREDENTIAL
    os.environ["AWS_SECURITY_TOKEN"] = _MOCK_AWS_CREDENTIAL
    os.environ["AWS_SESSION_TOKEN"] = _MOCK_AWS_CREDENTIAL
    os.environ["AWS_DEFAULT_REGION"] = TEST_REGION


@pytest.fixture
def sample_carga_dataframe() -> pd.DataFrame:
    """Create a sample carga DataFrame for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
            "cod_area": ["SP"] * 24,
            "carga_mwh": [100.0 + i * 0.5 for i in range(24)],
        }
    )


@pytest.fixture
def local_backend(tmp_path: Path) -> LocalStorageBackend:
    """Create a LocalStorageBackend for testing."""
    return LocalStorageBackend(base_path=tmp_path, create_dirs=True)


@pytest.fixture
def local_backend_with_data(
    local_backend: LocalStorageBackend, sample_carga_dataframe: pd.DataFrame
) -> LocalStorageBackend:
    """Create a LocalStorageBackend with sample data."""
    # Store sample data for SP on 2024-01-01
    path = "raw_data/2024/01/carga_SP_20240101.parquet"
    local_backend.put_parquet(path, sample_carga_dataframe)
    return local_backend


# =============================================================================
# DateRangeGenerator Tests
# =============================================================================


class TestDateRangeGenerator:
    """Tests for DateRangeGenerator utility class."""

    def test_date_range_generator_single_day(self):
        """Test generator with same start and end date."""
        date_range = DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 1))
        dates = list(date_range)

        assert len(dates) == 1
        assert dates[0] == date(2024, 1, 1)

    def test_date_range_generator_multiple_days(self):
        """Test generator with a range of dates."""
        date_range = DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 5))
        dates = list(date_range)

        assert len(dates) == 5
        assert dates[0] == date(2024, 1, 1)
        assert dates[-1] == date(2024, 1, 5)

    def test_date_range_generator_month_boundary(self):
        """Test generator crossing month boundary."""
        date_range = DateRangeGenerator(date(2024, 1, 30), date(2024, 2, 2))
        dates = list(date_range)

        assert len(dates) == 4
        assert dates == [
            date(2024, 1, 30),
            date(2024, 1, 31),
            date(2024, 2, 1),
            date(2024, 2, 2),
        ]

    def test_date_range_generator_year_boundary(self):
        """Test generator crossing year boundary."""
        date_range = DateRangeGenerator(date(2023, 12, 30), date(2024, 1, 2))
        dates = list(date_range)

        assert len(dates) == 4
        assert dates[0] == date(2023, 12, 30)
        assert dates[-1] == date(2024, 1, 2)

    def test_date_range_generator_invalid_range(self):
        """Test generator raises error when start > end."""
        with pytest.raises(ValueError) as exc_info:
            DateRangeGenerator(date(2024, 1, 5), date(2024, 1, 1))

        assert "cannot be after" in str(exc_info.value)

    def test_date_range_generator_len(self):
        """Test __len__ method."""
        date_range = DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 10))
        assert len(date_range) == 10

    def test_date_range_generator_repr(self):
        """Test __repr__ method."""
        date_range = DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 5))
        repr_str = repr(date_range)

        assert "DateRangeGenerator" in repr_str
        assert "2024-01-01" in repr_str
        assert "2024-01-05" in repr_str

    def test_date_range_generator_iterable_multiple_times(self):
        """Test that generator can be iterated multiple times."""
        date_range = DateRangeGenerator(date(2024, 1, 1), date(2024, 1, 3))

        first_iteration = list(date_range)
        second_iteration = list(date_range)

        assert first_iteration == second_iteration


# =============================================================================
# DataLoader Initialization Tests
# =============================================================================


class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_data_loader_init_with_backend(self, local_backend: LocalStorageBackend):
        """Test initialization with explicit backend."""
        loader = DataLoader(storage_backend=local_backend, max_workers=2)

        assert loader.storage_backend is local_backend
        assert loader.max_workers == 2

    def test_data_loader_init_default_max_workers(
        self, local_backend: LocalStorageBackend
    ):
        """Test default max_workers value."""
        loader = DataLoader(storage_backend=local_backend)

        assert loader.max_workers == 4

    def test_data_loader_init_default_backend(self, tmp_path: Path, monkeypatch):
        """Test initialization with default backend from config."""
        # Create a minimal config file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "storage.yaml"
        config_file.write_text(
            """
backend: local
local:
  base_path: /tmp/test_data
  create_dirs: true
"""
        )

        # Change to tmp_path so config can be found
        monkeypatch.chdir(tmp_path)

        loader = DataLoader()

        assert loader.storage_backend is not None
        assert loader.storage_backend.backend_type == "local"


# =============================================================================
# Path Generation Tests
# =============================================================================


class TestGeneratePaths:
    """Tests for DataLoader._generate_paths method."""

    def test_generate_paths_single_area_single_day(
        self, local_backend: LocalStorageBackend
    ):
        """Test path generation for single area and day."""
        loader = DataLoader(storage_backend=local_backend)
        paths = loader._generate_paths(
            prefix="carga",
            areas=["SP"],
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        assert len(paths) == 1
        assert paths[0] == "raw_data/2024/01/carga_SP_20240115.parquet"

    def test_generate_paths_multiple_areas(self, local_backend: LocalStorageBackend):
        """Test path generation for multiple areas."""
        loader = DataLoader(storage_backend=local_backend)
        paths = loader._generate_paths(
            prefix="carga",
            areas=["SP", "RJ", "MG"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert len(paths) == 3
        assert "raw_data/2024/01/carga_SP_20240101.parquet" in paths
        assert "raw_data/2024/01/carga_RJ_20240101.parquet" in paths
        assert "raw_data/2024/01/carga_MG_20240101.parquet" in paths

    def test_generate_paths_date_range(self, local_backend: LocalStorageBackend):
        """Test path generation for date range."""
        loader = DataLoader(storage_backend=local_backend)
        paths = loader._generate_paths(
            prefix="carga",
            areas=["SP"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
        )

        assert len(paths) == 3
        assert "raw_data/2024/01/carga_SP_20240101.parquet" in paths
        assert "raw_data/2024/01/carga_SP_20240102.parquet" in paths
        assert "raw_data/2024/01/carga_SP_20240103.parquet" in paths

    def test_generate_paths_crossing_months(self, local_backend: LocalStorageBackend):
        """Test path generation crossing month boundary."""
        loader = DataLoader(storage_backend=local_backend)
        paths = loader._generate_paths(
            prefix="carga",
            areas=["SP"],
            start_date=date(2024, 1, 31),
            end_date=date(2024, 2, 1),
        )

        assert len(paths) == 2
        assert "raw_data/2024/01/carga_SP_20240131.parquet" in paths
        assert "raw_data/2024/02/carga_SP_20240201.parquet" in paths

    def test_generate_paths_empty_areas(self, local_backend: LocalStorageBackend):
        """Test path generation with empty areas list."""
        loader = DataLoader(storage_backend=local_backend)
        paths = loader._generate_paths(
            prefix="carga",
            areas=[],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert paths == []


# =============================================================================
# Load Carga Tests - Local Backend
# =============================================================================


class TestLoadCargaLocal:
    """Tests for DataLoader.load_carga with local backend."""

    def test_load_carga_single_file_local(
        self, local_backend_with_data: LocalStorageBackend
    ):
        """Test loading single file from local storage."""
        loader = DataLoader(storage_backend=local_backend_with_data)
        df = loader.load_carga(
            areas=["SP"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert list(df.columns) == OUTPUT_COLUMNS
        assert all(df["cod_area"] == "SP")

    def test_load_carga_missing_file_returns_empty(
        self, local_backend: LocalStorageBackend
    ):
        """Test loading when file doesn't exist returns empty DataFrame."""
        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_carga(
            areas=["NONEXISTENT"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == OUTPUT_COLUMNS

    def test_load_carga_multiple_areas_parallel(
        self, local_backend: LocalStorageBackend, sample_carga_dataframe: pd.DataFrame
    ):
        """Test loading multiple areas in parallel."""
        # Store data for multiple areas
        for area in ["SP", "RJ", "MG"]:
            df = sample_carga_dataframe.copy()
            df["cod_area"] = area
            path = f"raw_data/2024/01/carga_{area}_20240101.parquet"
            local_backend.put_parquet(path, df)

        loader = DataLoader(storage_backend=local_backend, max_workers=3)
        df = loader.load_carga(
            areas=["SP", "RJ", "MG"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 72  # 24 hours * 3 areas
        assert set(df["cod_area"].unique()) == {"SP", "RJ", "MG"}

    def test_load_carga_date_range(
        self, local_backend: LocalStorageBackend, sample_carga_dataframe: pd.DataFrame
    ):
        """Test loading data for a date range."""
        # Store data for multiple days
        for day in [1, 2, 3]:
            df = sample_carga_dataframe.copy()
            path = f"raw_data/2024/01/carga_SP_2024010{day}.parquet"
            local_backend.put_parquet(path, df)

        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_carga(
            areas=["SP"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 72  # 24 hours * 3 days

    def test_load_carga_partial_data(
        self, local_backend: LocalStorageBackend, sample_carga_dataframe: pd.DataFrame
    ):
        """Test loading when only some files exist."""
        # Only store day 1 and 3, not day 2
        for day in [1, 3]:
            df = sample_carga_dataframe.copy()
            path = f"raw_data/2024/01/carga_SP_2024010{day}.parquet"
            local_backend.put_parquet(path, df)

        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_carga(
            areas=["SP"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 48  # 24 hours * 2 days (missing day 2)

    def test_load_carga_empty_areas(self, local_backend: LocalStorageBackend):
        """Test loading with empty areas list returns empty DataFrame."""
        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_carga(
            areas=[],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == OUTPUT_COLUMNS


# =============================================================================
# Load Carga Tests - S3 Backend
# =============================================================================


class TestLoadCargaS3:
    """Tests for DataLoader.load_carga with S3 backend using moto."""

    @pytest.fixture
    def s3_client(self, aws_credentials):
        """Create a mocked S3 client and bucket."""
        with mock_aws():
            client = boto3.client("s3", region_name=TEST_REGION)
            client.create_bucket(Bucket=TEST_BUCKET)
            yield client

    @pytest.fixture
    def s3_backend(self, s3_client) -> S3StorageBackend:
        """Create an S3StorageBackend for testing."""
        return S3StorageBackend(
            bucket=TEST_BUCKET,
            region=TEST_REGION,
            prefix="",
        )

    @pytest.fixture
    def s3_backend_with_data(
        self, s3_backend: S3StorageBackend, sample_carga_dataframe: pd.DataFrame
    ) -> S3StorageBackend:
        """Create S3 backend with sample data."""
        path = "raw_data/2024/01/carga_SP_20240101.parquet"
        s3_backend.put_parquet(path, sample_carga_dataframe)
        return s3_backend

    def test_load_carga_single_file_s3(
        self, s3_backend_with_data: S3StorageBackend
    ):
        """Test loading single file from S3 storage."""
        loader = DataLoader(storage_backend=s3_backend_with_data)
        df = loader.load_carga(
            areas=["SP"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert list(df.columns) == OUTPUT_COLUMNS
        assert all(df["cod_area"] == "SP")

    def test_load_carga_s3_missing_file(self, s3_backend: S3StorageBackend):
        """Test loading missing file from S3 returns empty DataFrame."""
        loader = DataLoader(storage_backend=s3_backend)
        df = loader.load_carga(
            areas=["NONEXISTENT"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == OUTPUT_COLUMNS

    def test_load_carga_s3_multiple_areas(
        self, s3_backend: S3StorageBackend, sample_carga_dataframe: pd.DataFrame
    ):
        """Test loading multiple areas from S3."""
        # Store data for multiple areas
        for area in ["SP", "RJ"]:
            df = sample_carga_dataframe.copy()
            df["cod_area"] = area
            path = f"raw_data/2024/01/carga_{area}_20240101.parquet"
            s3_backend.put_parquet(path, df)

        loader = DataLoader(storage_backend=s3_backend, max_workers=2)
        df = loader.load_carga(
            areas=["SP", "RJ"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 48  # 24 hours * 2 areas
        assert set(df["cod_area"].unique()) == {"SP", "RJ"}


# =============================================================================
# Single File Loading Tests
# =============================================================================


class TestLoadSingleFile:
    """Tests for DataLoader._load_single_file method."""

    def test_load_single_file_success(
        self, local_backend_with_data: LocalStorageBackend
    ):
        """Test successful single file loading."""
        loader = DataLoader(storage_backend=local_backend_with_data)
        df = loader._load_single_file("raw_data/2024/01/carga_SP_20240101.parquet")

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24

    def test_load_single_file_not_found(self, local_backend: LocalStorageBackend):
        """Test loading non-existent file returns None."""
        loader = DataLoader(storage_backend=local_backend)
        df = loader._load_single_file("nonexistent/path.parquet")

        assert df is None

    def test_load_single_file_handles_exception(
        self, local_backend: LocalStorageBackend
    ):
        """Test that exceptions during loading are handled gracefully."""
        # Create a file with invalid parquet data
        invalid_path = "raw_data/2024/01/invalid.parquet"
        local_backend.put_object(invalid_path, b"not valid parquet data")

        loader = DataLoader(storage_backend=local_backend)
        df = loader._load_single_file(invalid_path)

        assert df is None
