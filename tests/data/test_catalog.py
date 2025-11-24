"""Tests for data catalog module.

This module contains comprehensive tests for the DataCatalog class and
DatasetMetadata model, including tests for thread safety, persistence,
filtering, and version management.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import boto3
import pytest
from moto import mock_aws
from pydantic import ValidationError

from src.data.catalog import (
    CatalogError,
    DataCatalog,
    DatasetMetadata,
    DatasetNotFoundError,
)
from src.storage import LocalStorageBackend, S3StorageBackend

# Test constants
TEST_BUCKET = "test-catalog-bucket"
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
def local_backend(tmp_path: Path) -> LocalStorageBackend:
    """Create a LocalStorageBackend for testing."""
    return LocalStorageBackend(base_path=tmp_path, create_dirs=True)


@pytest.fixture
def sample_metadata() -> DatasetMetadata:
    """Create sample dataset metadata for testing."""
    return DatasetMetadata(
        name="carga_sp",
        version="1.0.0",
        storage_path="raw_data/2024/01/carga_SP.parquet",
        schema_type="carga",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        row_count=744,
        areas=["SP"],
        created_by="test_pipeline",
    )


@pytest.fixture
def sample_metadata_v2() -> DatasetMetadata:
    """Create sample dataset metadata version 2 for testing."""
    return DatasetMetadata(
        name="carga_sp",
        version="2.0.0",
        storage_path="raw_data/2024/02/carga_SP.parquet",
        schema_type="carga",
        start_date=date(2024, 2, 1),
        end_date=date(2024, 2, 29),
        row_count=696,
        areas=["SP"],
        created_by="test_pipeline",
        created_at=datetime.now(UTC) + timedelta(days=1),
    )


@pytest.fixture
def catalog(local_backend: LocalStorageBackend) -> DataCatalog:
    """Create a DataCatalog for testing."""
    return DataCatalog(storage_backend=local_backend)


# =============================================================================
# DatasetMetadata Tests
# =============================================================================


class TestDatasetMetadata:
    """Tests for DatasetMetadata Pydantic model."""

    def test_valid_metadata(self, sample_metadata: DatasetMetadata):
        """Test creating valid metadata."""
        assert sample_metadata.name == "carga_sp"
        assert sample_metadata.version == "1.0.0"
        assert sample_metadata.storage_path == "raw_data/2024/01/carga_SP.parquet"
        assert sample_metadata.schema_type == "carga"
        assert sample_metadata.start_date == date(2024, 1, 1)
        assert sample_metadata.end_date == date(2024, 1, 31)
        assert sample_metadata.row_count == 744
        assert sample_metadata.areas == ["SP"]
        assert sample_metadata.created_by == "test_pipeline"
        assert isinstance(sample_metadata.created_at, datetime)

    def test_version_validation_valid(self):
        """Test valid semantic version formats."""
        valid_versions = ["0.0.1", "1.0.0", "1.2.3", "10.20.30", "100.200.300"]
        for version in valid_versions:
            metadata = DatasetMetadata(
                name="test",
                version=version,
                storage_path="test.parquet",
                schema_type="test",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1),
                row_count=0,
                areas=[],
                created_by="test",
            )
            assert metadata.version == version

    def test_version_validation_invalid(self):
        """Test invalid version formats raise ValidationError."""
        invalid_versions = [
            "1",
            "1.0",
            "v1.0.0",
            "1.0.0-alpha",
            "1.0.0.0",
            "a.b.c",
            "",
            "latest",
        ]
        for version in invalid_versions:
            with pytest.raises(ValidationError) as exc_info:
                DatasetMetadata(
                    name="test",
                    version=version,
                    storage_path="test.parquet",
                    schema_type="test",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 1),
                    row_count=0,
                    areas=[],
                    created_by="test",
                )
            assert "version" in str(exc_info.value).lower()

    def test_date_range_validation_valid(self):
        """Test valid date ranges."""
        # Same start and end date
        metadata = DatasetMetadata(
            name="test",
            version="1.0.0",
            storage_path="test.parquet",
            schema_type="test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            row_count=0,
            areas=[],
            created_by="test",
        )
        assert metadata.start_date == metadata.end_date

    def test_date_range_validation_invalid(self):
        """Test invalid date range raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(
                name="test",
                version="1.0.0",
                storage_path="test.parquet",
                schema_type="test",
                start_date=date(2024, 1, 31),
                end_date=date(2024, 1, 1),  # Before start_date
                row_count=0,
                areas=[],
                created_by="test",
            )
        assert "end_date" in str(exc_info.value).lower()

    def test_row_count_validation(self):
        """Test row_count must be non-negative."""
        with pytest.raises(ValidationError):
            DatasetMetadata(
                name="test",
                version="1.0.0",
                storage_path="test.parquet",
                schema_type="test",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1),
                row_count=-1,  # Negative
                areas=[],
                created_by="test",
            )

    def test_required_fields(self):
        """Test that required fields must be provided."""
        with pytest.raises(ValidationError):
            DatasetMetadata(
                name="test",
                version="1.0.0",
                # Missing required fields
            )

    def test_model_dump_json(self, sample_metadata: DatasetMetadata):
        """Test JSON serialization."""
        data = sample_metadata.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["name"] == "carga_sp"
        assert data["version"] == "1.0.0"
        assert isinstance(data["start_date"], str)
        assert isinstance(data["created_at"], str)

    def test_model_validate_from_dict(self, sample_metadata: DatasetMetadata):
        """Test deserialization from dict."""
        data = sample_metadata.model_dump(mode="json")
        restored = DatasetMetadata.model_validate(data)
        assert restored.name == sample_metadata.name
        assert restored.version == sample_metadata.version
        assert restored.start_date == sample_metadata.start_date

    def test_frozen_model(self, sample_metadata: DatasetMetadata):
        """Test that the model is immutable (frozen)."""
        with pytest.raises(ValidationError):
            sample_metadata.name = "new_name"


# =============================================================================
# DataCatalog Initialization Tests
# =============================================================================


class TestDataCatalogInit:
    """Tests for DataCatalog initialization."""

    def test_init_with_empty_storage(self, local_backend: LocalStorageBackend):
        """Test initialization with empty storage."""
        catalog = DataCatalog(storage_backend=local_backend)
        assert catalog.list() == []

    def test_init_with_custom_path(self, local_backend: LocalStorageBackend):
        """Test initialization with custom catalog path."""
        custom_path = "custom/catalog.json"
        catalog = DataCatalog(storage_backend=local_backend, catalog_path=custom_path)
        catalog.register(
            DatasetMetadata(
                name="test",
                version="1.0.0",
                storage_path="test.parquet",
                schema_type="test",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1),
                row_count=0,
                areas=[],
                created_by="test",
            )
        )
        # Verify file was created at custom path
        assert local_backend.object_exists(custom_path)

    def test_init_loads_existing_catalog(self, local_backend: LocalStorageBackend):
        """Test that initialization loads existing catalog data."""
        # Create and populate a catalog
        catalog1 = DataCatalog(storage_backend=local_backend)
        metadata = DatasetMetadata(
            name="persistent",
            version="1.0.0",
            storage_path="test.parquet",
            schema_type="test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            row_count=100,
            areas=["SP"],
            created_by="test",
        )
        catalog1.register(metadata)

        # Create new catalog instance - should load existing data
        catalog2 = DataCatalog(storage_backend=local_backend)
        loaded = catalog2.get("persistent")
        assert loaded.name == "persistent"
        assert loaded.version == "1.0.0"

    def test_init_handles_corrupted_catalog(self, local_backend: LocalStorageBackend):
        """Test initialization handles corrupted catalog file gracefully."""
        # Write invalid JSON to catalog path
        local_backend.put_object("catalog/datasets.json", b"not valid json{")
        # Should not raise, starts with empty cache
        catalog = DataCatalog(storage_backend=local_backend)
        assert catalog.list() == []


# =============================================================================
# Register Tests
# =============================================================================


class TestRegister:
    """Tests for DataCatalog.register method."""

    def test_register_new_dataset(self, catalog: DataCatalog, sample_metadata: DatasetMetadata):
        """Test registering a new dataset."""
        catalog.register(sample_metadata)

        retrieved = catalog.get(sample_metadata.name)
        assert retrieved.name == sample_metadata.name
        assert retrieved.version == sample_metadata.version

    def test_register_multiple_versions(self, catalog: DataCatalog):
        """Test registering multiple versions of the same dataset."""
        v1 = DatasetMetadata(
            name="multi_version",
            version="1.0.0",
            storage_path="v1.parquet",
            schema_type="test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            row_count=100,
            areas=["SP"],
            created_by="test",
        )
        v2 = DatasetMetadata(
            name="multi_version",
            version="2.0.0",
            storage_path="v2.parquet",
            schema_type="test",
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 29),
            row_count=200,
            areas=["SP"],
            created_by="test",
        )

        catalog.register(v1)
        catalog.register(v2)

        assert catalog.get("multi_version", version="1.0.0").row_count == 100
        assert catalog.get("multi_version", version="2.0.0").row_count == 200

    def test_register_duplicate_raises_error(
        self, catalog: DataCatalog, sample_metadata: DatasetMetadata
    ):
        """Test registering duplicate dataset version raises error."""
        catalog.register(sample_metadata)

        with pytest.raises(CatalogError) as exc_info:
            catalog.register(sample_metadata)
        assert "already exists" in str(exc_info.value)
        assert exc_info.value.dataset_name == sample_metadata.name
        assert exc_info.value.version == sample_metadata.version

    def test_register_persists_to_storage(
        self, local_backend: LocalStorageBackend, sample_metadata: DatasetMetadata
    ):
        """Test that registration persists to storage."""
        catalog1 = DataCatalog(storage_backend=local_backend)
        catalog1.register(sample_metadata)

        # Create new catalog - should see the registered dataset
        catalog2 = DataCatalog(storage_backend=local_backend)
        assert catalog2.exists(sample_metadata.name)


# =============================================================================
# Get Tests
# =============================================================================


class TestGet:
    """Tests for DataCatalog.get method."""

    def test_get_specific_version(self, catalog: DataCatalog, sample_metadata: DatasetMetadata):
        """Test getting a specific version."""
        catalog.register(sample_metadata)
        retrieved = catalog.get(sample_metadata.name, version="1.0.0")
        assert retrieved.version == "1.0.0"

    def test_get_latest_version(
        self,
        catalog: DataCatalog,
        sample_metadata: DatasetMetadata,
        sample_metadata_v2: DatasetMetadata,
    ):
        """Test getting latest version by created_at."""
        catalog.register(sample_metadata)
        catalog.register(sample_metadata_v2)

        latest = catalog.get(sample_metadata.name, version="latest")
        # v2 was created later
        assert latest.version == "2.0.0"

    def test_get_latest_defaults(self, catalog: DataCatalog, sample_metadata: DatasetMetadata):
        """Test that default version is 'latest'."""
        catalog.register(sample_metadata)
        retrieved = catalog.get(sample_metadata.name)
        assert retrieved.version == sample_metadata.version

    def test_get_nonexistent_dataset_raises(self, catalog: DataCatalog):
        """Test getting non-existent dataset raises error."""
        with pytest.raises(DatasetNotFoundError) as exc_info:
            catalog.get("nonexistent")
        assert exc_info.value.dataset_name == "nonexistent"

    def test_get_nonexistent_version_raises(
        self, catalog: DataCatalog, sample_metadata: DatasetMetadata
    ):
        """Test getting non-existent version raises error."""
        catalog.register(sample_metadata)

        with pytest.raises(DatasetNotFoundError) as exc_info:
            catalog.get(sample_metadata.name, version="9.9.9")
        assert exc_info.value.version == "9.9.9"


# =============================================================================
# List Tests
# =============================================================================


class TestList:
    """Tests for DataCatalog.list method."""

    def test_list_all_datasets(self, catalog: DataCatalog):
        """Test listing all datasets."""
        datasets = [
            DatasetMetadata(
                name=f"dataset_{i}",
                version="1.0.0",
                storage_path=f"data_{i}.parquet",
                schema_type="test",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                row_count=i * 100,
                areas=["SP"],
                created_by="test",
            )
            for i in range(5)
        ]
        for dataset in datasets:
            catalog.register(dataset)

        listed = catalog.list()
        assert len(listed) == 5

    def test_list_empty_catalog(self, catalog: DataCatalog):
        """Test listing empty catalog returns empty list."""
        assert catalog.list() == []

    def test_list_filter_by_name(self, catalog: DataCatalog):
        """Test filtering by dataset name."""
        for name in ["alpha", "beta", "gamma"]:
            catalog.register(
                DatasetMetadata(
                    name=name,
                    version="1.0.0",
                    storage_path=f"{name}.parquet",
                    schema_type="test",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 31),
                    row_count=100,
                    areas=["SP"],
                    created_by="test",
                )
            )

        listed = catalog.list(name="beta")
        assert len(listed) == 1
        assert listed[0].name == "beta"

    def test_list_filter_by_schema_type(self, catalog: DataCatalog):
        """Test filtering by schema type."""
        for schema in ["carga", "temperatura", "carga"]:
            catalog.register(
                DatasetMetadata(
                    name=f"ds_{schema}_{time.time()}",
                    version="1.0.0",
                    storage_path="test.parquet",
                    schema_type=schema,
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 31),
                    row_count=100,
                    areas=["SP"],
                    created_by="test",
                )
            )
            time.sleep(0.001)  # Ensure unique names

        listed = catalog.list(schema_type="carga")
        assert len(listed) == 2
        assert all(d.schema_type == "carga" for d in listed)

    def test_list_filter_by_areas(self, catalog: DataCatalog):
        """Test filtering by areas."""
        areas_list = [["SP"], ["RJ"], ["SP", "RJ"], ["MG"]]
        for i, areas in enumerate(areas_list):
            catalog.register(
                DatasetMetadata(
                    name=f"area_ds_{i}",
                    version="1.0.0",
                    storage_path="test.parquet",
                    schema_type="test",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 31),
                    row_count=100,
                    areas=areas,
                    created_by="test",
                )
            )

        # Filter for SP should match datasets 0, 2
        listed = catalog.list(areas=["SP"])
        assert len(listed) == 2

        # Filter for RJ should match datasets 1, 2
        listed = catalog.list(areas=["RJ"])
        assert len(listed) == 2

        # Filter for both SP and RJ should match all with either
        listed = catalog.list(areas=["SP", "RJ"])
        assert len(listed) == 3  # 0, 1, 2

    def test_list_filter_by_date_range(self, catalog: DataCatalog):
        """Test filtering by date range."""
        date_ranges = [
            (date(2024, 1, 1), date(2024, 1, 31)),
            (date(2024, 2, 1), date(2024, 2, 29)),
            (date(2024, 3, 1), date(2024, 3, 31)),
        ]
        for i, (start, end) in enumerate(date_ranges):
            catalog.register(
                DatasetMetadata(
                    name=f"date_ds_{i}",
                    version="1.0.0",
                    storage_path="test.parquet",
                    schema_type="test",
                    start_date=start,
                    end_date=end,
                    row_count=100,
                    areas=["SP"],
                    created_by="test",
                )
            )

        # Filter by start_date
        listed = catalog.list(start_date=date(2024, 2, 1))
        assert len(listed) == 2  # Feb and Mar

        # Filter by end_date
        listed = catalog.list(end_date=date(2024, 2, 29))
        assert len(listed) == 2  # Jan and Feb

        # Filter by both
        listed = catalog.list(start_date=date(2024, 2, 1), end_date=date(2024, 2, 29))
        assert len(listed) == 1  # Only Feb

    def test_list_multiple_filters(self, catalog: DataCatalog):
        """Test combining multiple filters."""
        datasets = [
            DatasetMetadata(
                name="carga_sp",
                version="1.0.0",
                storage_path="test.parquet",
                schema_type="carga",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                row_count=100,
                areas=["SP"],
                created_by="test",
            ),
            DatasetMetadata(
                name="carga_rj",
                version="1.0.0",
                storage_path="test.parquet",
                schema_type="carga",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                row_count=100,
                areas=["RJ"],
                created_by="test",
            ),
            DatasetMetadata(
                name="temp_sp",
                version="1.0.0",
                storage_path="test.parquet",
                schema_type="temperatura",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                row_count=100,
                areas=["SP"],
                created_by="test",
            ),
        ]
        for ds in datasets:
            catalog.register(ds)

        # Filter by schema_type AND areas
        listed = catalog.list(schema_type="carga", areas=["SP"])
        assert len(listed) == 1
        assert listed[0].name == "carga_sp"


# =============================================================================
# Delete Tests
# =============================================================================


class TestDelete:
    """Tests for DataCatalog.delete method."""

    def test_delete_specific_version(self, catalog: DataCatalog, sample_metadata: DatasetMetadata):
        """Test deleting a specific version."""
        catalog.register(sample_metadata)
        assert catalog.exists(sample_metadata.name)

        catalog.delete(sample_metadata.name, sample_metadata.version)
        assert not catalog.exists(sample_metadata.name)

    def test_delete_one_version_keeps_others(
        self,
        catalog: DataCatalog,
        sample_metadata: DatasetMetadata,
        sample_metadata_v2: DatasetMetadata,
    ):
        """Test deleting one version keeps other versions."""
        catalog.register(sample_metadata)
        catalog.register(sample_metadata_v2)

        catalog.delete(sample_metadata.name, "1.0.0")

        assert not catalog.exists(sample_metadata.name, version="1.0.0")
        assert catalog.exists(sample_metadata.name, version="2.0.0")

    def test_delete_nonexistent_raises(self, catalog: DataCatalog):
        """Test deleting non-existent dataset raises error."""
        with pytest.raises(DatasetNotFoundError):
            catalog.delete("nonexistent", "1.0.0")

    def test_delete_nonexistent_version_raises(
        self, catalog: DataCatalog, sample_metadata: DatasetMetadata
    ):
        """Test deleting non-existent version raises error."""
        catalog.register(sample_metadata)

        with pytest.raises(DatasetNotFoundError):
            catalog.delete(sample_metadata.name, "9.9.9")

    def test_delete_persists(
        self, local_backend: LocalStorageBackend, sample_metadata: DatasetMetadata
    ):
        """Test that deletion persists to storage."""
        catalog1 = DataCatalog(storage_backend=local_backend)
        catalog1.register(sample_metadata)
        catalog1.delete(sample_metadata.name, sample_metadata.version)

        catalog2 = DataCatalog(storage_backend=local_backend)
        assert not catalog2.exists(sample_metadata.name)


# =============================================================================
# Exists Tests
# =============================================================================


class TestExists:
    """Tests for DataCatalog.exists method."""

    def test_exists_by_name(self, catalog: DataCatalog, sample_metadata: DatasetMetadata):
        """Test checking existence by name only."""
        assert not catalog.exists(sample_metadata.name)
        catalog.register(sample_metadata)
        assert catalog.exists(sample_metadata.name)

    def test_exists_by_name_and_version(
        self, catalog: DataCatalog, sample_metadata: DatasetMetadata
    ):
        """Test checking existence by name and version."""
        catalog.register(sample_metadata)
        assert catalog.exists(sample_metadata.name, version="1.0.0")
        assert not catalog.exists(sample_metadata.name, version="2.0.0")


# =============================================================================
# Get Versions Tests
# =============================================================================


class TestGetVersions:
    """Tests for DataCatalog.get_versions method."""

    def test_get_versions_single(self, catalog: DataCatalog, sample_metadata: DatasetMetadata):
        """Test getting versions for single-version dataset."""
        catalog.register(sample_metadata)
        versions = catalog.get_versions(sample_metadata.name)
        assert versions == ["1.0.0"]

    def test_get_versions_multiple(
        self,
        catalog: DataCatalog,
        sample_metadata: DatasetMetadata,
        sample_metadata_v2: DatasetMetadata,
    ):
        """Test getting versions sorted by created_at."""
        catalog.register(sample_metadata)
        catalog.register(sample_metadata_v2)

        versions = catalog.get_versions(sample_metadata.name)
        # v2 was created later, should be first
        assert versions == ["2.0.0", "1.0.0"]

    def test_get_versions_nonexistent_raises(self, catalog: DataCatalog):
        """Test getting versions for non-existent dataset raises error."""
        with pytest.raises(DatasetNotFoundError):
            catalog.get_versions("nonexistent")


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_registrations(self, local_backend: LocalStorageBackend):
        """Test concurrent registrations are thread-safe."""
        catalog = DataCatalog(storage_backend=local_backend)
        num_threads = 10
        datasets_per_thread = 5

        def register_datasets(thread_id: int):
            for i in range(datasets_per_thread):
                metadata = DatasetMetadata(
                    name=f"thread_{thread_id}_ds_{i}",
                    version="1.0.0",
                    storage_path=f"thread_{thread_id}_{i}.parquet",
                    schema_type="test",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 31),
                    row_count=100,
                    areas=["SP"],
                    created_by=f"thread_{thread_id}",
                )
                catalog.register(metadata)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_datasets, i) for i in range(num_threads)]
            for future in futures:
                future.result()  # Raise any exceptions

        # Verify all datasets were registered
        all_datasets = catalog.list()
        assert len(all_datasets) == num_threads * datasets_per_thread

    def test_concurrent_read_write(self, local_backend: LocalStorageBackend):
        """Test concurrent reads and writes are thread-safe."""
        catalog = DataCatalog(storage_backend=local_backend)
        errors: list[Exception] = []

        # Pre-register some datasets
        for i in range(5):
            catalog.register(
                DatasetMetadata(
                    name=f"initial_ds_{i}",
                    version="1.0.0",
                    storage_path=f"initial_{i}.parquet",
                    schema_type="test",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 31),
                    row_count=100,
                    areas=["SP"],
                    created_by="setup",
                )
            )

        def reader(iterations: int):
            try:
                for _ in range(iterations):
                    catalog.list()
                    if catalog.exists("initial_ds_0"):
                        catalog.get("initial_ds_0")
            except Exception as e:
                errors.append(e)

        def writer(thread_id: int, iterations: int):
            try:
                for i in range(iterations):
                    catalog.register(
                        DatasetMetadata(
                            name=f"writer_{thread_id}_ds_{i}",
                            version="1.0.0",
                            storage_path=f"writer_{thread_id}_{i}.parquet",
                            schema_type="test",
                            start_date=date(2024, 1, 1),
                            end_date=date(2024, 1, 31),
                            row_count=100,
                            areas=["SP"],
                            created_by=f"writer_{thread_id}",
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        # Start reader threads
        for _ in range(3):
            t = threading.Thread(target=reader, args=(20,))
            threads.append(t)
            t.start()

        # Start writer threads
        for i in range(3):
            t = threading.Thread(target=writer, args=(i, 5))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"


# =============================================================================
# S3 Backend Tests
# =============================================================================


class TestS3Backend:
    """Tests for DataCatalog with S3 backend using moto."""

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
    def s3_catalog(self, s3_backend: S3StorageBackend) -> DataCatalog:
        """Create a DataCatalog with S3 backend."""
        return DataCatalog(storage_backend=s3_backend)

    def test_register_and_get_s3(self, s3_catalog: DataCatalog, sample_metadata: DatasetMetadata):
        """Test registering and retrieving with S3 backend."""
        s3_catalog.register(sample_metadata)
        retrieved = s3_catalog.get(sample_metadata.name)
        assert retrieved.name == sample_metadata.name
        assert retrieved.version == sample_metadata.version

    def test_persistence_s3(self, s3_backend: S3StorageBackend, sample_metadata: DatasetMetadata):
        """Test that catalog persists across instances with S3."""
        catalog1 = DataCatalog(storage_backend=s3_backend)
        catalog1.register(sample_metadata)

        catalog2 = DataCatalog(storage_backend=s3_backend)
        assert catalog2.exists(sample_metadata.name)

    def test_list_filter_s3(self, s3_catalog: DataCatalog):
        """Test filtering with S3 backend."""
        datasets = [
            DatasetMetadata(
                name="s3_carga",
                version="1.0.0",
                storage_path="carga.parquet",
                schema_type="carga",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                row_count=100,
                areas=["SP"],
                created_by="test",
            ),
            DatasetMetadata(
                name="s3_temp",
                version="1.0.0",
                storage_path="temp.parquet",
                schema_type="temperatura",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                row_count=100,
                areas=["RJ"],
                created_by="test",
            ),
        ]
        for ds in datasets:
            s3_catalog.register(ds)

        listed = s3_catalog.list(schema_type="carga")
        assert len(listed) == 1
        assert listed[0].name == "s3_carga"

    def test_delete_s3(self, s3_catalog: DataCatalog, sample_metadata: DatasetMetadata):
        """Test deletion with S3 backend."""
        s3_catalog.register(sample_metadata)
        s3_catalog.delete(sample_metadata.name, sample_metadata.version)
        assert not s3_catalog.exists(sample_metadata.name)


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for catalog persistence."""

    def test_reload_after_multiple_operations(self, local_backend: LocalStorageBackend):
        """Test catalog reload after multiple operations."""
        catalog1 = DataCatalog(storage_backend=local_backend)

        # Register multiple datasets
        for i in range(3):
            catalog1.register(
                DatasetMetadata(
                    name=f"persist_ds_{i}",
                    version="1.0.0",
                    storage_path=f"persist_{i}.parquet",
                    schema_type="test",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 31),
                    row_count=100,
                    areas=["SP"],
                    created_by="test",
                )
            )

        # Delete one
        catalog1.delete("persist_ds_1", "1.0.0")

        # Reload and verify
        catalog2 = DataCatalog(storage_backend=local_backend)
        assert catalog2.exists("persist_ds_0")
        assert not catalog2.exists("persist_ds_1")
        assert catalog2.exists("persist_ds_2")

    def test_catalog_json_format(self, local_backend: LocalStorageBackend):
        """Test that catalog is stored as valid JSON."""
        catalog = DataCatalog(storage_backend=local_backend)
        catalog.register(
            DatasetMetadata(
                name="json_test",
                version="1.0.0",
                storage_path="test.parquet",
                schema_type="test",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                row_count=100,
                areas=["SP"],
                created_by="test",
            )
        )

        # Read raw catalog file
        import json

        data = local_backend.get_object("catalog/datasets.json")
        catalog_json = json.loads(data.decode("utf-8"))

        assert "json_test" in catalog_json
        assert len(catalog_json["json_test"]) == 1
        assert catalog_json["json_test"][0]["version"] == "1.0.0"
