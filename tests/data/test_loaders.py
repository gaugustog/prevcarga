"""Tests for data loaders module.

This module contains comprehensive tests for the DataLoader class,
DateRangeGenerator utility, MultiFormatLoader, and DataJoiner classes,
including tests for both local filesystem and S3 storage backends using
moto for AWS mocking.
"""

import os
from datetime import date
from pathlib import Path

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from src.data.loaders import (
    HEAT_INDEX_OUTPUT_COLUMNS,
    HOLIDAY_OUTPUT_COLUMNS,
    OUTPUT_COLUMNS,
    TEMPERATURE_OUTPUT_COLUMNS,
    DataJoiner,
    DataLoader,
    MultiFormatLoader,
)
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


# =============================================================================
# Temperature Loading Tests
# =============================================================================


class TestLoadTemperatura:
    """Tests for DataLoader temperature loading methods."""

    @pytest.fixture
    def sample_temperatura_dataframe(self) -> pd.DataFrame:
        """Create a sample temperature DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "cod_area": ["SP"] * 24,
                "temp_celsius": [25.0 + i * 0.5 for i in range(24)],
            }
        )

    @pytest.fixture
    def local_backend_with_temp_data(
        self, local_backend: LocalStorageBackend, sample_temperatura_dataframe: pd.DataFrame
    ) -> LocalStorageBackend:
        """Create a LocalStorageBackend with sample temperature data."""
        # Store D+1 temperature data
        path_d1 = "raw_data/2024/01/temperatura_d1_SP_20240101.parquet"
        local_backend.put_parquet(path_d1, sample_temperatura_dataframe)

        # Store D+2 temperature data
        path_d2 = "raw_data/2024/01/temperatura_d2_SP_20240101.parquet"
        local_backend.put_parquet(path_d2, sample_temperatura_dataframe)

        return local_backend

    def test_load_temperatura_d1_single_file(
        self, local_backend_with_temp_data: LocalStorageBackend
    ):
        """Test loading D+1 temperature data from single file."""
        loader = DataLoader(storage_backend=local_backend_with_temp_data)
        df = loader.load_temperatura_d1(
            areas=["SP"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert "source" in df.columns
        assert all(df["source"] == "D+1")
        # Check standard columns exist
        for col in ["timestamp", "cod_area", "temp_celsius", "source"]:
            assert col in df.columns

    def test_load_temperatura_d2_single_file(
        self, local_backend_with_temp_data: LocalStorageBackend
    ):
        """Test loading D+2 temperature data from single file."""
        loader = DataLoader(storage_backend=local_backend_with_temp_data)
        df = loader.load_temperatura_d2(
            areas=["SP"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert "source" in df.columns
        assert all(df["source"] == "D+2")

    def test_load_temperatura_missing_file(
        self, local_backend: LocalStorageBackend
    ):
        """Test loading temperature when file doesn't exist returns empty DataFrame."""
        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_temperatura_d1(
            areas=["NONEXISTENT"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == TEMPERATURE_OUTPUT_COLUMNS

    def test_load_temperatura_empty_areas(
        self, local_backend: LocalStorageBackend
    ):
        """Test loading temperature with empty areas list."""
        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_temperatura_d1(
            areas=[],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == TEMPERATURE_OUTPUT_COLUMNS

    def test_generate_temperatura_paths(self, local_backend: LocalStorageBackend):
        """Test temperature path generation."""
        loader = DataLoader(storage_backend=local_backend)
        paths = loader._generate_temperatura_paths(
            areas=["SP", "RJ"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            horizon=1,
        )

        assert len(paths) == 4  # 2 areas * 2 days
        assert "raw_data/2024/01/temperatura_d1_SP_20240101.parquet" in paths
        assert "raw_data/2024/01/temperatura_d1_RJ_20240101.parquet" in paths
        assert "raw_data/2024/01/temperatura_d1_SP_20240102.parquet" in paths
        assert "raw_data/2024/01/temperatura_d1_RJ_20240102.parquet" in paths


# =============================================================================
# Heat Index Loading Tests
# =============================================================================


class TestLoadHeatIndex:
    """Tests for DataLoader heat index loading methods."""

    @pytest.fixture
    def sample_heat_index_dataframe(self) -> pd.DataFrame:
        """Create a sample heat index DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "heat_index": [30.0 + i * 0.5 for i in range(24)],
            }
        )

    @pytest.fixture
    def local_backend_with_heat_data(
        self, local_backend: LocalStorageBackend, sample_heat_index_dataframe: pd.DataFrame
    ) -> LocalStorageBackend:
        """Create a LocalStorageBackend with sample heat index data."""
        # Store CSV heat index data
        csv_data = sample_heat_index_dataframe.to_csv(index=False).encode("utf-8")
        local_backend.put_object("raw_data/heatindex_SP.csv", csv_data)

        # Store Parquet heat index data
        local_backend.put_parquet("raw_data/heatindex_RJ.parquet", sample_heat_index_dataframe)

        return local_backend

    def test_load_heat_index_csv(
        self, local_backend_with_heat_data: LocalStorageBackend
    ):
        """Test loading heat index from CSV file."""
        loader = DataLoader(storage_backend=local_backend_with_heat_data)
        df = loader.load_heat_index(areas=["SP"], file_format="csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert "cod_area" in df.columns
        assert all(df["cod_area"] == "SP")

    def test_load_heat_index_parquet(
        self, local_backend_with_heat_data: LocalStorageBackend
    ):
        """Test loading heat index from Parquet file."""
        loader = DataLoader(storage_backend=local_backend_with_heat_data)
        df = loader.load_heat_index(areas=["RJ"], file_format="parquet")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert "cod_area" in df.columns
        assert all(df["cod_area"] == "RJ")

    def test_load_heat_index_missing_file(
        self, local_backend: LocalStorageBackend
    ):
        """Test loading heat index when file doesn't exist."""
        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_heat_index(areas=["NONEXISTENT"], file_format="csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == HEAT_INDEX_OUTPUT_COLUMNS

    def test_load_heat_index_empty_areas(
        self, local_backend: LocalStorageBackend
    ):
        """Test loading heat index with empty areas list."""
        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_heat_index(areas=[], file_format="csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == HEAT_INDEX_OUTPUT_COLUMNS


# =============================================================================
# ECMWF Temperature Loading Tests
# =============================================================================


class TestLoadTemperaturaEcmwf:
    """Tests for DataLoader ECMWF temperature loading methods."""

    @pytest.fixture
    def sample_ecmwf_dataframe(self) -> pd.DataFrame:
        """Create a sample ECMWF temperature DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "temp_celsius": [26.0 + i * 0.3 for i in range(24)],
            }
        )

    @pytest.fixture
    def local_backend_with_ecmwf_data(
        self, local_backend: LocalStorageBackend, sample_ecmwf_dataframe: pd.DataFrame
    ) -> LocalStorageBackend:
        """Create a LocalStorageBackend with sample ECMWF data."""
        csv_data = sample_ecmwf_dataframe.to_csv(index=False).encode("utf-8")
        local_backend.put_object("raw_data/heatindex_SP.csv", csv_data)
        return local_backend

    def test_load_temperatura_ecmwf(
        self, local_backend_with_ecmwf_data: LocalStorageBackend
    ):
        """Test loading ECMWF temperature data."""
        loader = DataLoader(storage_backend=local_backend_with_ecmwf_data)
        df = loader.load_temperatura_ecmwf(areas=["SP"], file_format="csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert "source" in df.columns
        assert all(df["source"] == "ECMWF")
        assert "cod_area" in df.columns
        assert all(df["cod_area"] == "SP")

    def test_load_temperatura_ecmwf_empty_areas(
        self, local_backend: LocalStorageBackend
    ):
        """Test loading ECMWF temperature with empty areas list."""
        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_temperatura_ecmwf(areas=[], file_format="csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == TEMPERATURE_OUTPUT_COLUMNS


# =============================================================================
# Holiday Loading Tests
# =============================================================================


class TestLoadFeriados:
    """Tests for DataLoader holiday loading methods."""

    @pytest.fixture
    def sample_feriados_dataframe(self) -> pd.DataFrame:
        """Create a sample holidays DataFrame for testing."""
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-04-21", "2024-12-25"]),
                "holiday_name": ["Ano Novo", "Tiradentes", "Natal"],
                "holiday_type": ["National", "National", "National"],
                "id_tipodiaespecial": [1, 2, 1],
            }
        )

    @pytest.fixture
    def local_backend_with_feriados(
        self, local_backend: LocalStorageBackend, sample_feriados_dataframe: pd.DataFrame
    ) -> LocalStorageBackend:
        """Create a LocalStorageBackend with sample holiday data."""
        local_backend.put_parquet("raw_data/feriados_2024.parquet", sample_feriados_dataframe)
        return local_backend

    def test_load_feriados_single_year(
        self, local_backend_with_feriados: LocalStorageBackend
    ):
        """Test loading holidays for a single year."""
        loader = DataLoader(storage_backend=local_backend_with_feriados)
        df = loader.load_feriados(years=[2024])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "holiday_name" in df.columns
        assert "Ano Novo" in df["holiday_name"].values

    def test_load_feriados_multiple_years(
        self, local_backend: LocalStorageBackend, sample_feriados_dataframe: pd.DataFrame
    ):
        """Test loading holidays for multiple years."""
        # Store data for multiple years
        for year in [2023, 2024]:
            df = sample_feriados_dataframe.copy()
            local_backend.put_parquet(f"raw_data/feriados_{year}.parquet", df)

        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_feriados(years=[2023, 2024])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # 3 holidays per year * 2 years

    def test_load_feriados_missing_year(
        self, local_backend: LocalStorageBackend
    ):
        """Test loading holidays when year file doesn't exist."""
        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_feriados(years=[2099])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == HOLIDAY_OUTPUT_COLUMNS

    def test_load_feriados_empty_years(
        self, local_backend: LocalStorageBackend
    ):
        """Test loading holidays with empty years list."""
        loader = DataLoader(storage_backend=local_backend)
        df = loader.load_feriados(years=[])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == HOLIDAY_OUTPUT_COLUMNS


# =============================================================================
# MultiFormatLoader Tests
# =============================================================================


class TestMultiFormatLoader:
    """Tests for MultiFormatLoader class."""

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "col_a": [1, 2, 3],
                "col_b": ["x", "y", "z"],
            }
        )

    def test_load_parquet_format(
        self, local_backend: LocalStorageBackend, sample_dataframe: pd.DataFrame
    ):
        """Test loading Parquet files."""
        local_backend.put_parquet("data/test.parquet", sample_dataframe)
        loader = MultiFormatLoader(local_backend)

        df = loader.load("data/test.parquet")

        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["col_a", "col_b"]

    def test_load_csv_format(
        self, local_backend: LocalStorageBackend, sample_dataframe: pd.DataFrame
    ):
        """Test loading CSV files."""
        csv_data = sample_dataframe.to_csv(index=False).encode("utf-8")
        local_backend.put_object("data/test.csv", csv_data)
        loader = MultiFormatLoader(local_backend)

        df = loader.load("data/test.csv")

        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["col_a", "col_b"]

    def test_load_csv_with_explicit_format(
        self, local_backend: LocalStorageBackend, sample_dataframe: pd.DataFrame
    ):
        """Test loading CSV with explicit format parameter."""
        csv_data = sample_dataframe.to_csv(index=False).encode("utf-8")
        local_backend.put_object("data/test.csv", csv_data)
        loader = MultiFormatLoader(local_backend)

        df = loader.load("data/test.csv", file_format="csv")

        assert df is not None
        assert len(df) == 3

    def test_load_csv_latin1_encoding(
        self, local_backend: LocalStorageBackend
    ):
        """Test loading CSV with Latin-1 encoding."""
        # Create CSV with Latin-1 characters
        df = pd.DataFrame({"nome": ["Sao Paulo", "Rio de Janeiro"]})
        csv_data = df.to_csv(index=False).encode("latin-1")
        local_backend.put_object("data/test.csv", csv_data)
        loader = MultiFormatLoader(local_backend)

        result = loader.load("data/test.csv", encoding="latin-1")

        assert result is not None
        assert len(result) == 2

    def test_load_auto_detect_format_parquet(
        self, local_backend: LocalStorageBackend, sample_dataframe: pd.DataFrame
    ):
        """Test auto-detection of Parquet format from extension."""
        local_backend.put_parquet("data/file.parquet", sample_dataframe)
        loader = MultiFormatLoader(local_backend)

        df = loader.load("data/file.parquet")

        assert df is not None
        assert len(df) == 3

    def test_load_auto_detect_format_csv(
        self, local_backend: LocalStorageBackend, sample_dataframe: pd.DataFrame
    ):
        """Test auto-detection of CSV format from extension."""
        csv_data = sample_dataframe.to_csv(index=False).encode("utf-8")
        local_backend.put_object("data/file.csv", csv_data)
        loader = MultiFormatLoader(local_backend)

        df = loader.load("data/file.csv")

        assert df is not None
        assert len(df) == 3

    def test_load_unsupported_format(
        self, local_backend: LocalStorageBackend
    ):
        """Test error on unsupported file format."""
        loader = MultiFormatLoader(local_backend)

        with pytest.raises(ValueError) as exc_info:
            loader.load("data/file.json")

        assert "Cannot detect file format" in str(exc_info.value)

    def test_load_file_not_found(
        self, local_backend: LocalStorageBackend
    ):
        """Test loading non-existent file returns None."""
        loader = MultiFormatLoader(local_backend)

        df = loader.load("nonexistent/file.parquet")

        assert df is None

    def test_detect_format_parquet(
        self, local_backend: LocalStorageBackend
    ):
        """Test format detection for parquet extension."""
        loader = MultiFormatLoader(local_backend)

        fmt = loader._detect_format("path/to/file.parquet")
        assert fmt == "parquet"

        fmt = loader._detect_format("path/to/file.PARQUET")
        assert fmt == "parquet"

    def test_detect_format_csv(
        self, local_backend: LocalStorageBackend
    ):
        """Test format detection for CSV extension."""
        loader = MultiFormatLoader(local_backend)

        fmt = loader._detect_format("path/to/file.csv")
        assert fmt == "csv"

        fmt = loader._detect_format("path/to/file.CSV")
        assert fmt == "csv"


# =============================================================================
# MultiFormatLoader S3 Tests
# =============================================================================


class TestMultiFormatLoaderS3:
    """Tests for MultiFormatLoader with S3 backend using moto."""

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
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "col_a": [1, 2, 3],
                "col_b": ["x", "y", "z"],
            }
        )

    def test_load_csv_from_s3(
        self, s3_backend: S3StorageBackend, sample_dataframe: pd.DataFrame
    ):
        """Test loading CSV from S3."""
        csv_data = sample_dataframe.to_csv(index=False).encode("utf-8")
        s3_backend.put_object("data/test.csv", csv_data)
        loader = MultiFormatLoader(s3_backend)

        df = loader.load("data/test.csv")

        assert df is not None
        assert len(df) == 3

    def test_load_parquet_from_s3(
        self, s3_backend: S3StorageBackend, sample_dataframe: pd.DataFrame
    ):
        """Test loading Parquet from S3."""
        s3_backend.put_parquet("data/test.parquet", sample_dataframe)
        loader = MultiFormatLoader(s3_backend)

        df = loader.load("data/test.parquet")

        assert df is not None
        assert len(df) == 3


# =============================================================================
# DataJoiner Tests
# =============================================================================


class TestDataJoiner:
    """Tests for DataJoiner class."""

    @pytest.fixture
    def sample_load_dataframe(self) -> pd.DataFrame:
        """Create a sample load DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "cod_area": ["SP"] * 24,
                "carga_mwh": [100.0 + i for i in range(24)],
            }
        )

    @pytest.fixture
    def sample_temp_dataframe(self) -> pd.DataFrame:
        """Create a sample temperature DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "cod_area": ["SP"] * 24,
                "temp_celsius": [25.0 + i * 0.5 for i in range(24)],
                "source": ["D+1"] * 24,
            }
        )

    @pytest.fixture
    def sample_holiday_dataframe(self) -> pd.DataFrame:
        """Create a sample holiday DataFrame for testing."""
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "holiday_name": ["Ano Novo"],
                "holiday_type": ["National"],
                "id_tipodiaespecial": [1],
            }
        )

    def test_join_load_and_temperature_left(
        self, sample_load_dataframe: pd.DataFrame, sample_temp_dataframe: pd.DataFrame
    ):
        """Test left join of load and temperature data."""
        result = DataJoiner.join_load_and_temperature(
            sample_load_dataframe,
            sample_temp_dataframe,
            how="left",
        )

        assert len(result) == 24
        assert "carga_mwh" in result.columns
        assert "temp_celsius" in result.columns
        assert "source" in result.columns

    def test_join_load_and_temperature_inner(
        self, sample_load_dataframe: pd.DataFrame, sample_temp_dataframe: pd.DataFrame
    ):
        """Test inner join of load and temperature data."""
        result = DataJoiner.join_load_and_temperature(
            sample_load_dataframe,
            sample_temp_dataframe,
            how="inner",
        )

        assert len(result) == 24
        assert "carga_mwh" in result.columns
        assert "temp_celsius" in result.columns

    def test_join_load_and_temperature_empty_load(
        self, sample_temp_dataframe: pd.DataFrame
    ):
        """Test join with empty load DataFrame."""
        empty_load = pd.DataFrame(columns=["timestamp", "cod_area", "carga_mwh"])
        result = DataJoiner.join_load_and_temperature(empty_load, sample_temp_dataframe)

        assert len(result) == 0

    def test_join_load_and_temperature_empty_temp(
        self, sample_load_dataframe: pd.DataFrame
    ):
        """Test join with empty temperature DataFrame."""
        empty_temp = pd.DataFrame(
            columns=["timestamp", "cod_area", "temp_celsius", "source"]
        )
        result = DataJoiner.join_load_and_temperature(sample_load_dataframe, empty_temp)

        assert len(result) == 24
        assert "carga_mwh" in result.columns

    def test_join_with_holidays(
        self, sample_load_dataframe: pd.DataFrame, sample_holiday_dataframe: pd.DataFrame
    ):
        """Test joining with holiday data."""
        result = DataJoiner.join_with_holidays(
            sample_load_dataframe,
            sample_holiday_dataframe,
        )

        assert len(result) == 24
        assert "holiday_name" in result.columns
        # First 24 hours are on Jan 1, so they should all have the holiday
        jan_1_rows = result[pd.to_datetime(result["timestamp"]).dt.date == date(2024, 1, 1)]
        assert all(jan_1_rows["holiday_name"] == "Ano Novo")

    def test_join_with_holidays_empty_input(
        self, sample_holiday_dataframe: pd.DataFrame
    ):
        """Test join_with_holidays with empty input DataFrame."""
        empty_df = pd.DataFrame(columns=["timestamp", "cod_area", "carga_mwh"])
        result = DataJoiner.join_with_holidays(empty_df, sample_holiday_dataframe)

        assert len(result) == 0

    def test_join_with_holidays_empty_holidays(
        self, sample_load_dataframe: pd.DataFrame
    ):
        """Test join_with_holidays with empty holiday DataFrame."""
        empty_holidays = pd.DataFrame(columns=["date", "holiday_name", "holiday_type"])
        result = DataJoiner.join_with_holidays(sample_load_dataframe, empty_holidays)

        assert len(result) == 24
        assert "carga_mwh" in result.columns

    def test_join_with_holidays_custom_timestamp_col(
        self, sample_holiday_dataframe: pd.DataFrame
    ):
        """Test join_with_holidays with custom timestamp column name."""
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=24, freq="h"),
                "cod_area": ["SP"] * 24,
                "carga_mwh": [100.0] * 24,
            }
        )

        result = DataJoiner.join_with_holidays(
            df,
            sample_holiday_dataframe,
            timestamp_col="datetime",
        )

        assert len(result) == 24
        assert "holiday_name" in result.columns

    def test_join_all_sources(
        self,
        sample_load_dataframe: pd.DataFrame,
        sample_temp_dataframe: pd.DataFrame,
        sample_holiday_dataframe: pd.DataFrame,
    ):
        """Test joining all data sources at once."""
        # Create heat index DataFrame
        df_heat = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "cod_area": ["SP"] * 24,
                "heat_index": [30.0 + i * 0.5 for i in range(24)],
            }
        )

        result = DataJoiner.join_all_sources(
            df_load=sample_load_dataframe,
            df_temps=[sample_temp_dataframe],
            df_heat=df_heat,
            df_holidays=sample_holiday_dataframe,
        )

        assert len(result) == 24
        assert "carga_mwh" in result.columns
        assert "temp_celsius" in result.columns
        assert "heat_index" in result.columns
        assert "holiday_name" in result.columns

    def test_join_all_sources_empty_load(self):
        """Test join_all_sources with empty load DataFrame."""
        empty_load = pd.DataFrame(columns=["timestamp", "cod_area", "carga_mwh"])
        result = DataJoiner.join_all_sources(df_load=empty_load)

        assert len(result) == 0

    def test_join_all_sources_no_auxiliary_data(
        self, sample_load_dataframe: pd.DataFrame
    ):
        """Test join_all_sources with no auxiliary data."""
        result = DataJoiner.join_all_sources(df_load=sample_load_dataframe)

        assert len(result) == 24
        assert "carga_mwh" in result.columns

    def test_join_all_sources_multiple_temp_sources(
        self, sample_load_dataframe: pd.DataFrame
    ):
        """Test joining with multiple temperature sources."""
        df_temp_d1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "cod_area": ["SP"] * 24,
                "temp_celsius": [25.0] * 24,
                "source": ["D+1"] * 24,
            }
        )
        df_temp_d2 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "cod_area": ["SP"] * 24,
                "temp_celsius": [26.0] * 24,
                "source": ["D+2"] * 24,
            }
        )

        result = DataJoiner.join_all_sources(
            df_load=sample_load_dataframe,
            df_temps=[df_temp_d1, df_temp_d2],
        )

        assert len(result) == 24
        assert "carga_mwh" in result.columns
        # Should have temp data from both sources


# =============================================================================
# Column Standardization Tests
# =============================================================================


class TestColumnStandardization:
    """Tests for column standardization methods."""

    def test_standardize_temperatura_columns_with_mapping(
        self, local_backend: LocalStorageBackend
    ):
        """Test standardization of temperature columns with various names."""
        loader = DataLoader(storage_backend=local_backend)

        # DataFrame with non-standard column names
        df = pd.DataFrame(
            {
                "data_hora": pd.date_range("2024-01-01", periods=3, freq="h"),
                "area": ["SP"] * 3,
                "temperatura": [25.0, 26.0, 27.0],
            }
        )

        result = loader._standardize_temperatura_columns(df, horizon=1)

        assert "timestamp" in result.columns
        assert "cod_area" in result.columns
        assert "temp_celsius" in result.columns
        assert "source" in result.columns
        assert all(result["source"] == "D+1")

    def test_standardize_feriados_columns_with_mapping(
        self, local_backend: LocalStorageBackend
    ):
        """Test standardization of holiday columns with various names."""
        loader = DataLoader(storage_backend=local_backend)

        # DataFrame with non-standard column names
        df = pd.DataFrame(
            {
                "data": pd.to_datetime(["2024-01-01"]),
                "nome": ["Ano Novo"],
                "tipo": ["National"],
                "id_tipo": [1],
            }
        )

        result = loader._standardize_feriados_columns(df)

        assert "date" in result.columns
        assert "holiday_name" in result.columns
        assert "holiday_type" in result.columns
        assert "id_tipodiaespecial" in result.columns
