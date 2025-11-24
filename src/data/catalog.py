"""Data catalog for registering, discovering, and tracking datasets.

This module provides the DataCatalog class for managing dataset metadata,
including versioning, filtering, and persistence to storage backends.

Example:
    ```python
    from datetime import date, datetime
    from src.data.catalog import DataCatalog, DatasetMetadata
    from src.storage import LocalStorageBackend

    # Create catalog with local backend
    backend = LocalStorageBackend(base_path="./data")
    catalog = DataCatalog(storage_backend=backend)

    # Register a dataset
    metadata = DatasetMetadata(
        name="carga_sp",
        version="1.0.0",
        storage_path="raw_data/2024/01/carga_SP.parquet",
        schema_type="carga",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        row_count=744,
        areas=["SP"],
        created_by="data_pipeline",
    )
    catalog.register(metadata)

    # Get latest version
    latest = catalog.get("carga_sp", version="latest")

    # List datasets with filtering
    datasets = catalog.list(schema_type="carga", areas=["SP"])
    ```
"""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.storage.backend import ObjectNotFoundError, StorageBackend
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _utc_now() -> datetime:
    """Return current UTC datetime in a timezone-aware format."""
    return datetime.now(UTC)


# Semantic version pattern: MAJOR.MINOR.PATCH
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")

# Default catalog file path
DEFAULT_CATALOG_PATH = "catalog/datasets.json"


class DatasetMetadata(BaseModel):
    """Metadata for a registered dataset.

    Attributes:
        name: Unique identifier for the dataset.
        version: Semantic version string (e.g., "1.0.0").
        storage_path: Path to the dataset in storage.
        schema_type: Type of data schema (e.g., "carga", "temperatura").
        start_date: Start date of the data coverage.
        end_date: End date of the data coverage.
        row_count: Number of rows in the dataset.
        areas: List of geographic area codes.
        created_at: Timestamp when the metadata was created.
        created_by: Identifier of the entity that created the dataset.

    Example:
        ```python
        metadata = DatasetMetadata(
            name="carga_sp_202401",
            version="1.0.0",
            storage_path="raw_data/2024/01/carga_SP.parquet",
            schema_type="carga",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            row_count=744,
            areas=["SP"],
            created_by="data_pipeline",
        )
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
    )

    name: str = Field(..., min_length=1, description="Dataset name identifier")
    version: str = Field(..., description="Semantic version (MAJOR.MINOR.PATCH)")
    storage_path: str = Field(..., min_length=1, description="Path to dataset in storage")
    schema_type: str = Field(..., min_length=1, description="Data schema type")
    start_date: date = Field(..., description="Start date of data coverage")
    end_date: date = Field(..., description="End date of data coverage")
    row_count: int = Field(..., ge=0, description="Number of rows in dataset")
    areas: list[str] = Field(default_factory=list, description="Geographic area codes")
    created_at: datetime = Field(default_factory=_utc_now, description="Creation timestamp")
    created_by: str = Field(..., min_length=1, description="Creator identifier")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate that version follows semantic versioning pattern.

        Args:
            v: Version string to validate.

        Returns:
            The validated version string.

        Raises:
            ValueError: If version doesn't match semantic versioning pattern.
        """
        if not VERSION_PATTERN.match(v):
            msg = f"Version must follow semantic versioning (MAJOR.MINOR.PATCH), got: {v}"
            raise ValueError(msg)
        return v

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: date, info: Any) -> date:
        """Validate that end_date is not before start_date.

        Args:
            v: End date to validate.
            info: Validation context with other field values.

        Returns:
            The validated end date.

        Raises:
            ValueError: If end_date is before start_date.
        """
        if "start_date" in info.data and v < info.data["start_date"]:
            msg = f"end_date ({v}) cannot be before start_date ({info.data['start_date']})"
            raise ValueError(msg)
        return v


class CatalogError(Exception):
    """Exception raised for catalog operations errors.

    Attributes:
        message: Error message.
        dataset_name: Name of the dataset involved (if applicable).
        version: Version of the dataset involved (if applicable).
    """

    def __init__(
        self,
        message: str,
        dataset_name: str | None = None,
        version: str | None = None,
    ) -> None:
        """Initialize CatalogError.

        Args:
            message: Error message.
            dataset_name: Name of the dataset involved.
            version: Version of the dataset involved.
        """
        self.dataset_name = dataset_name
        self.version = version
        super().__init__(message)


class DatasetNotFoundError(CatalogError):
    """Exception raised when a dataset is not found in the catalog."""

    def __init__(self, name: str, version: str | None = None) -> None:
        """Initialize DatasetNotFoundError.

        Args:
            name: Name of the dataset that was not found.
            version: Version of the dataset that was not found.
        """
        version_str = f" version {version}" if version else ""
        super().__init__(
            f"Dataset not found: {name}{version_str}",
            dataset_name=name,
            version=version,
        )


class DataCatalog:
    """Catalog for registering, discovering, and tracking datasets.

    The DataCatalog provides thread-safe operations for managing dataset metadata,
    including version management, filtering, and persistence to storage backends.

    Attributes:
        storage_backend: Storage backend for persistence.
        catalog_path: Path to the catalog JSON file in storage.

    Example:
        ```python
        from src.data.catalog import DataCatalog
        from src.storage import LocalStorageBackend

        backend = LocalStorageBackend(base_path="./data")
        catalog = DataCatalog(storage_backend=backend)

        # Register a dataset
        catalog.register(metadata)

        # Get specific version
        meta = catalog.get("my_dataset", version="1.0.0")

        # Get latest version
        latest = catalog.get("my_dataset", version="latest")

        # List with filters
        datasets = catalog.list(schema_type="carga", areas=["SP", "RJ"])
        ```
    """

    def __init__(
        self,
        storage_backend: StorageBackend,
        catalog_path: str = DEFAULT_CATALOG_PATH,
    ) -> None:
        """Initialize DataCatalog.

        Args:
            storage_backend: Storage backend for persistence.
            catalog_path: Path to store the catalog JSON file.
        """
        self._storage = storage_backend
        self._catalog_path = catalog_path
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, DatasetMetadata]] = {}

        # Load existing catalog from storage
        self._load_from_storage()

        logger.info(
            "Initialized DataCatalog with %d datasets, catalog_path=%s",
            sum(len(versions) for versions in self._cache.values()),
            catalog_path,
        )

    def _load_from_storage(self) -> None:
        """Load catalog data from storage backend.

        This method attempts to load the catalog JSON file from storage.
        If the file doesn't exist, an empty cache is initialized.
        """
        try:
            data = self._storage.get_object(self._catalog_path)
            catalog_data = json.loads(data.decode("utf-8"))
            self._deserialize_cache(catalog_data)
            logger.debug("Loaded catalog with %d datasets from storage", len(self._cache))
        except ObjectNotFoundError:
            logger.debug("No existing catalog found, starting with empty cache")
            self._cache = {}
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse catalog file, starting fresh: %s", e)
            self._cache = {}

    def _save_to_storage(self) -> None:
        """Save catalog data to storage backend.

        This method serializes the current cache to JSON and persists it
        to the storage backend.
        """
        try:
            catalog_data = self._serialize_cache()
            json_data = json.dumps(catalog_data, indent=2)
            self._storage.put_object(self._catalog_path, json_data.encode("utf-8"))
            logger.debug("Saved catalog with %d datasets to storage", len(self._cache))
        except Exception as e:
            logger.error("Failed to save catalog to storage: %s", e)
            msg = f"Failed to save catalog: {e}"
            raise CatalogError(msg) from e

    def _serialize_cache(self) -> dict[str, list[dict[str, Any]]]:
        """Serialize the cache to a JSON-compatible dictionary.

        Returns:
            Dictionary with dataset names as keys and lists of version metadata.
        """
        result: dict[str, list[dict[str, Any]]] = {}
        for name, versions in self._cache.items():
            result[name] = [metadata.model_dump(mode="json") for metadata in versions.values()]
        return result

    def _deserialize_cache(self, data: dict[str, list[dict[str, Any]]]) -> None:
        """Deserialize catalog data into the cache.

        Args:
            data: Dictionary with dataset names and version metadata lists.
        """
        self._cache = {}
        for name, versions in data.items():
            self._cache[name] = {}
            for version_data in versions:
                metadata = DatasetMetadata.model_validate(version_data)
                self._cache[name][metadata.version] = metadata

    def register(self, metadata: DatasetMetadata) -> None:
        """Register a dataset in the catalog.

        Args:
            metadata: Dataset metadata to register.

        Raises:
            CatalogError: If a dataset with the same name and version exists.

        Example:
            ```python
            metadata = DatasetMetadata(
                name="carga_sp",
                version="1.0.0",
                storage_path="raw_data/carga_SP.parquet",
                schema_type="carga",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                row_count=744,
                areas=["SP"],
                created_by="pipeline",
            )
            catalog.register(metadata)
            ```
        """
        with self._lock:
            if metadata.name not in self._cache:
                self._cache[metadata.name] = {}

            if metadata.version in self._cache[metadata.name]:
                msg = f"Dataset '{metadata.name}' version '{metadata.version}' already exists"
                raise CatalogError(msg, metadata.name, metadata.version)

            self._cache[metadata.name][metadata.version] = metadata
            self._save_to_storage()

        logger.info(
            "Registered dataset '%s' version '%s'",
            metadata.name,
            metadata.version,
        )

    def get(
        self,
        name: str,
        version: str = "latest",
    ) -> DatasetMetadata:
        """Get dataset metadata by name and version.

        Args:
            name: Dataset name to retrieve.
            version: Version to retrieve. Use "latest" for the most recent version.

        Returns:
            The requested DatasetMetadata.

        Raises:
            DatasetNotFoundError: If the dataset or version is not found.

        Example:
            ```python
            # Get specific version
            meta = catalog.get("carga_sp", version="1.0.0")

            # Get latest version
            latest = catalog.get("carga_sp", version="latest")
            ```
        """
        with self._lock:
            if name not in self._cache:
                raise DatasetNotFoundError(name)

            versions = self._cache[name]

            if version == "latest":
                return self._get_latest_version(name, versions)

            if version not in versions:
                raise DatasetNotFoundError(name, version)

            return versions[version]

    def _get_latest_version(
        self,
        name: str,
        versions: dict[str, DatasetMetadata],
    ) -> DatasetMetadata:
        """Get the latest version of a dataset based on created_at timestamp.

        Args:
            name: Dataset name.
            versions: Dictionary of version -> metadata.

        Returns:
            The metadata with the most recent created_at timestamp.

        Raises:
            DatasetNotFoundError: If no versions exist for the dataset.
        """
        if not versions:
            raise DatasetNotFoundError(name, "latest")

        # Sort by created_at timestamp descending
        sorted_versions = sorted(
            versions.values(),
            key=lambda m: m.created_at,
            reverse=True,
        )
        return sorted_versions[0]

    def list(
        self,
        name: str | None = None,
        schema_type: str | None = None,
        areas: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[DatasetMetadata]:
        """List datasets matching the specified filters.

        Args:
            name: Filter by exact dataset name.
            schema_type: Filter by schema type.
            areas: Filter by areas (datasets must contain at least one).
            start_date: Filter datasets with start_date >= this date.
            end_date: Filter datasets with end_date <= this date.

        Returns:
            List of DatasetMetadata matching all specified filters.

        Example:
            ```python
            # List all datasets
            all_datasets = catalog.list()

            # Filter by schema type
            carga_datasets = catalog.list(schema_type="carga")

            # Filter by areas
            sp_datasets = catalog.list(areas=["SP"])

            # Filter by date range
            jan_datasets = catalog.list(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )
            ```
        """
        with self._lock:
            results: list[DatasetMetadata] = []

            for dataset_name, versions in self._cache.items():
                # Filter by name if specified
                if name is not None and dataset_name != name:
                    continue

                for metadata in versions.values():
                    if self._matches_filters(
                        metadata,
                        schema_type=schema_type,
                        areas=areas,
                        start_date=start_date,
                        end_date=end_date,
                    ):
                        results.append(metadata)

            # Sort by name, then version
            results.sort(key=lambda m: (m.name, m.version))
            return results

    def _matches_filters(
        self,
        metadata: DatasetMetadata,
        schema_type: str | None = None,
        areas: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> bool:
        """Check if metadata matches all specified filters.

        Args:
            metadata: Dataset metadata to check.
            schema_type: Filter by schema type.
            areas: Filter by areas (must contain at least one).
            start_date: Filter by minimum start date.
            end_date: Filter by maximum end date.

        Returns:
            True if metadata matches all filters, False otherwise.
        """
        if schema_type is not None and metadata.schema_type != schema_type:
            return False

        if areas is not None and not any(area in metadata.areas for area in areas):
            return False

        if start_date is not None and metadata.start_date < start_date:
            return False

        return not (end_date is not None and metadata.end_date > end_date)

    def delete(self, name: str, version: str) -> None:
        """Delete a dataset from the catalog.

        Args:
            name: Dataset name to delete.
            version: Version to delete.

        Raises:
            DatasetNotFoundError: If the dataset or version is not found.

        Example:
            ```python
            catalog.delete("carga_sp", version="1.0.0")
            ```
        """
        with self._lock:
            if name not in self._cache:
                raise DatasetNotFoundError(name)

            if version not in self._cache[name]:
                raise DatasetNotFoundError(name, version)

            del self._cache[name][version]

            # Remove the dataset entry if no versions remain
            if not self._cache[name]:
                del self._cache[name]

            self._save_to_storage()

        logger.info("Deleted dataset '%s' version '%s'", name, version)

    def exists(self, name: str, version: str | None = None) -> bool:
        """Check if a dataset exists in the catalog.

        Args:
            name: Dataset name to check.
            version: Optional version to check. If None, checks if any version exists.

        Returns:
            True if the dataset (and version, if specified) exists.

        Example:
            ```python
            # Check if any version exists
            if catalog.exists("carga_sp"):
                print("Dataset exists")

            # Check specific version
            if catalog.exists("carga_sp", version="1.0.0"):
                print("Version 1.0.0 exists")
            ```
        """
        with self._lock:
            if name not in self._cache:
                return False

            if version is None:
                return True

            return version in self._cache[name]

    def get_versions(self, name: str) -> list[str]:
        """Get all versions for a dataset.

        Args:
            name: Dataset name.

        Returns:
            List of version strings, sorted by created_at timestamp (newest first).

        Raises:
            DatasetNotFoundError: If the dataset is not found.

        Example:
            ```python
            versions = catalog.get_versions("carga_sp")
            # Returns: ["1.2.0", "1.1.0", "1.0.0"]
            ```
        """
        with self._lock:
            if name not in self._cache:
                raise DatasetNotFoundError(name)

            # Sort versions by created_at descending
            sorted_metadata = sorted(
                self._cache[name].values(),
                key=lambda m: m.created_at,
                reverse=True,
            )
            return [m.version for m in sorted_metadata]
