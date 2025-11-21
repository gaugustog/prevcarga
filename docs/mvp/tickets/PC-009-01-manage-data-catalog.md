# PC-009-01: Manage Data Catalog

**Ticket ID:** PC-009-01  
**Epic:** [Epic-01: Data Infrastructure Layer](../epics/Epic-01.md)  
**User Story:** US-1.4  
**Story Points:** 3  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement DataCatalog system for registering, discovering, and tracking datasets with metadata, versioning, and S3 persistence to enable data lineage and version management across the pipeline.

**As a** ML engineer  
**I want to** register and discover datasets with metadata  
**So that** I can track data versions and lineage

---

## ✅ Acceptance Criteria

- [ ] `DataCatalog` class for dataset registration and discovery
- [ ] Metadata stored includes: name, version, storage path (unified), schema, date range, row count, creation timestamp
- [ ] Support searching datasets by area, date range, version
- [ ] Persist catalog to configured storage backend (S3 or local) as JSON or Parquet
- [ ] Support dataset versioning with semantic versioning
- [ ] API methods: `register()`, `get()`, `list()`, `delete()`
- [ ] Thread-safe for concurrent registrations
- [ ] Storage backend configurable via DataCatalog initialization

---

## 🔧 Implementation Tasks

### 1. Define DatasetMetadata Schema
- [ ] Create Pydantic model for dataset metadata
- [ ] Fields: name, version, s3_path, schema_type, start_date, end_date, row_count, areas, created_at, created_by
- [ ] Add validation for version format (semantic versioning)
- [ ] Add JSON serialization support

### 2. Implement DataCatalog Class
- [ ] Initialize with S3 bucket and catalog key
- [ ] Implement in-memory cache (dict)
- [ ] Add thread lock for concurrent access
- [ ] Load catalog from S3 on initialization
- [ ] Implement `register()` method
- [ ] Implement `get()` method with version support ("latest")
- [ ] Implement `list()` method with filtering
- [ ] Implement `delete()` method
- [ ] Implement `_load_from_s3()` private method
- [ ] Implement `_persist_to_s3()` private method

### 3. Add Search and Filter Capabilities
- [ ] Filter by dataset name
- [ ] Filter by area codes
- [ ] Filter by date range
- [ ] Filter by schema type
- [ ] Support combining filters
- [ ] Return sorted results (by creation date)

### 4. Implement Versioning Support
- [ ] Parse semantic versions (major.minor.patch)
- [ ] Find latest version for dataset name
- [ ] Support version comparison
- [ ] Validate version format
- [ ] Auto-increment patch version option

### 5. Add Storage Backend Persistence
- [ ] Serialize catalog to JSON
- [ ] Handle datetime serialization
- [ ] Upload to storage backend after each modification
- [ ] Download from storage backend on initialization
- [ ] Handle missing catalog (create new)
- [ ] Add error handling for storage operations (both S3 and local)
- [ ] Support StorageBackend interface for flexibility

### 6. Write Tests
- [ ] Test dataset registration
- [ ] Test retrieval by name and version
- [ ] Test "latest" version resolution
- [ ] Test filtering (by area, date, schema)
- [ ] Test persistence to/from S3 (moto)
- [ ] Test concurrent access (threading)
- [ ] Test version comparison

---

## 💻 Implementation Details

```python
"""Data catalog for dataset management and versioning."""
import json
import threading
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from src.storage.backend import StorageBackend
from src.storage.factory import StorageFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetMetadata(BaseModel):
    """Metadata for cataloged dataset."""
    
    name: str = Field(..., description="Dataset name")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version")
    storage_path: str = Field(..., description="Storage path to dataset (S3 or local)")
    schema_type: str = Field(..., description="Schema class name")
    start_date: date = Field(..., description="First date in dataset")
    end_date: date = Field(..., description="Last date in dataset")
    row_count: int = Field(..., ge=0, description="Number of records")
    areas: List[str] = Field(default_factory=list, description="Area codes in dataset")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    created_by: str = Field(default="system", description="Creator identifier")
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat(),
            datetime: lambda v: v.isoformat(),
        }


class DataCatalog:
    """Registry for dataset metadata with unified storage persistence."""
    
    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        catalog_path: str = "catalog/datasets.json"
    ) -> None:
        """
        Initialize data catalog.
        
        Args:
            storage_backend: Storage backend for catalog persistence. If None, uses config.
            catalog_path: Storage path for catalog JSON file
        """
        self.storage = storage_backend or StorageFactory.from_config()
        self.catalog_path = catalog_path
        self._cache: Dict[str, DatasetMetadata] = {}
        self._lock = threading.Lock()
        
        # Load existing catalog
        self._load_from_storage()
        logger.info(
            f"DataCatalog initialized with {len(self._cache)} datasets "
            f"(backend: {self.storage.__class__.__name__})"
        )
    
    def register(self, metadata: DatasetMetadata) -> None:
        """
        Register new dataset in catalog.
        
        Args:
            metadata: Dataset metadata to register
        """
        with self._lock:
            key = f"{metadata.name}:{metadata.version}"
            
            if key in self._cache:
                logger.warning(f"Overwriting existing dataset: {key}")
            
            self._cache[key] = metadata
            self._persist_to_storage()
            logger.info(f"Registered dataset: {key} ({metadata.row_count} rows)")
    
    def get(self, name: str, version: str = "latest") -> Optional[DatasetMetadata]:
        """
        Retrieve dataset metadata.
        
        Args:
            name: Dataset name
            version: Version or "latest" for most recent
        
        Returns:
            Dataset metadata or None if not found
        """
        with self._lock:
            if version == "latest":
                matches = [m for k, m in self._cache.items() if m.name == name]
                if not matches:
                    logger.warning(f"Dataset not found: {name}")
                    return None
                return max(matches, key=lambda m: m.created_at)
            
            key = f"{name}:{version}"
            return self._cache.get(key)
    
    def list(
        self,
        name: Optional[str] = None,
        areas: Optional[List[str]] = None,
        schema_type: Optional[str] = None
    ) -> List[DatasetMetadata]:
        """
        List datasets with optional filtering.
        
        Args:
            name: Filter by dataset name
            areas: Filter by areas (any match)
            schema_type: Filter by schema type
        
        Returns:
            List of matching datasets, sorted by creation date (descending)
        """
        with self._lock:
            datasets = list(self._cache.values())
            
            if name:
                datasets = [d for d in datasets if d.name == name]
            
            if areas:
                datasets = [d for d in datasets if any(a in d.areas for a in areas)]
            
            if schema_type:
                datasets = [d for d in datasets if d.schema_type == schema_type]
            
            return sorted(datasets, key=lambda d: d.created_at, reverse=True)
    
    def delete(self, name: str, version: str) -> bool:
        """
        Delete dataset from catalog.
        
        Args:
            name: Dataset name
            version: Dataset version
        
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            key = f"{name}:{version}"
            if key in self._cache:
                del self._cache[key]
                self._persist_to_s3()
                logger.info(f"Deleted dataset: {key}")
                return True
            return False
    
    def _load_from_s3(self) -> None:
        """Load catalog from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.catalog_key)
            data = json.loads(response["Body"].read())
            self._cache = {k: DatasetMetadata(**v) for k, v in data.items()}
            logger.info(f"Loaded {len(self._cache)} datasets from S3")
        except self.s3_client.exceptions.NoSuchKey:
            logger.info("Catalog not found in S3, starting fresh")
            self._cache = {}
        except Exception as e:
            logger.error(f"Error loading catalog from S3: {e}")
            self._cache = {}
    
    def _persist_to_s3(self) -> None:
        """Persist catalog to S3."""
        try:
            data = {k: v.model_dump(mode="json") for k, v in self._cache.items()}
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=self.catalog_key,
                Body=json.dumps(data, default=str, indent=2),
                ContentType="application/json"
            )
            logger.debug(f"Persisted catalog to S3: {len(self._cache)} datasets")
        except Exception as e:
            logger.error(f"Error persisting catalog to S3: {e}")
            raise
```

---

## 🧪 Testing & Validation

```python
"""Tests for DataCatalog."""
import pytest
from datetime import date, datetime
from moto import mock_s3
import boto3

from src.data.catalog import DataCatalog, DatasetMetadata


@mock_s3
def test_catalog_with_s3_backend():
    """Test dataset registration and retrieval with S3 backend."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    
    from src.storage.factory import StorageFactory
    backend = StorageFactory.create("s3", bucket="test-bucket")
    catalog = DataCatalog(storage_backend=backend)
    
    metadata = DatasetMetadata(
        name="carga_rj",
        version="1.0.0",
        storage_path="raw_data/2024/01/carga_rj.parquet",
        schema_type="CargaSchema",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        row_count=1488,
        areas=["RJ"]
    )
    
    catalog.register(metadata)
    
    result = catalog.get("carga_rj", "1.0.0")
    assert result is not None
    assert result.name == "carga_rj"
    assert result.row_count == 1488


def test_catalog_with_local_backend(tmp_path):
    """Test dataset registration and retrieval with local backend."""
    from src.storage.factory import StorageFactory
    backend = StorageFactory.create("local", base_path=str(tmp_path))
    catalog = DataCatalog(storage_backend=backend)
    
    metadata = DatasetMetadata(
        name="carga_rj",
        version="1.0.0",
        storage_path="raw_data/2024/01/carga_rj.parquet",
        schema_type="CargaSchema",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        row_count=1488,
        areas=["RJ"]
    )
    
    catalog.register(metadata)
    
    result = catalog.get("carga_rj", "1.0.0")
    assert result is not None
    assert result.name == "carga_rj"
    assert result.row_count == 1488


@mock_s3
def test_catalog_latest_version():
    """Test retrieving latest version."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    
    catalog = DataCatalog("test-bucket")
    
    # Register multiple versions
    for version in ["1.0.0", "1.1.0", "2.0.0"]:
        metadata = DatasetMetadata(
            name="carga_rj",
            version=version,
            s3_path=f"s3://bucket/{version}",
            schema_type="CargaSchema",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            row_count=1488,
            areas=["RJ"]
        )
        catalog.register(metadata)
    
    latest = catalog.get("carga_rj", "latest")
    assert latest.version == "2.0.0"
```

---

## 🔗 Dependencies

**Depends On:**
- PC-003-00: S3 Storage Configuration
- PC-006-01: Load Raw Load Data from S3

**Blocks:**
- Epic-08: Orchestration (uses catalog for dataset discovery)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] DataCatalog class implemented
- [ ] S3 persistence working
- [ ] Versioning support implemented
- [ ] Unit tests pass with >80% coverage
- [ ] Thread-safety verified
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next Ticket:** [PC-010-01: Filter Complete Days and Validate Continuity](PC-010-01-filter-complete-days.md)
