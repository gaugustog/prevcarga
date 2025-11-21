# PC-003-00: Unified Storage Configuration (S3 + Local)

**Ticket ID:** PC-003-00  
**Epic:** [Epic-00: Project Foundation & Setup](../epics/Epic-00.md)  
**User Story:** US-00.3  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement a unified storage abstraction layer supporting both AWS S3 (production) and local filesystem (development) with a common interface, enabling seamless switching between storage backends without code changes.

**As a** developer  
**I want** a unified storage interface that works with both S3 and local filesystem  
**So that** I can develop locally and deploy to production without code changes

---

## ✅ Acceptance Criteria

- [ ] Abstract `StorageBackend` interface defined with operations: list, get, put, delete, exists
- [ ] `S3StorageBackend` implementation with boto3 integration
- [ ] `LocalStorageBackend` implementation for local filesystem
- [ ] `StorageFactory` for backend instantiation and configuration-based selection
- [ ] Storage backend auto-detection based on configuration or path prefix (s3://, file://, relative)
- [ ] AWS credentials configured (via environment variables or AWS CLI) for S3 backend
- [ ] S3 bucket created or identified: `prevcarga-bucket-sandbox`
- [ ] Local storage base path configurable (default: `./data/`)
- [ ] Storage structure documented (raw_data/, features/, models/, results/, cache/) for both backends
- [ ] Connection test script works for both backends
- [ ] Storage configuration module created in `src/storage/`
- [ ] CLI can specify storage backend via `--storage-backend` flag or config

---

## 🔧 Implementation Tasks

### 1. AWS Credentials Setup
- [ ] Document AWS credential options:
  - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
  - AWS CLI configuration (~/.aws/credentials)
  - IAM roles (for EC2/Fargate)
- [ ] Create `.env.template` file with required variables
- [ ] Add `.env` to `.gitignore`
- [ ] Document credential setup in README.md

### 2. S3 Bucket Setup
- [ ] Identify or create S3 bucket: `prevcarga-bucket-sandbox`
- [ ] Document bucket structure:
  - `raw_data/` - Raw parquet files (load, temperature, holidays)
  - `features/` - Engineered feature datasets
  - `models/` - Trained model artifacts
  - `results/` - Prediction outputs and evaluation results
  - `cache/` - Temporary cached data
- [ ] Verify bucket permissions (read, write, list, delete)
- [ ] Configure bucket region (us-east-1)

### 3. Create Storage Configuration Module
- [ ] Create `config/storage.yaml` with:
  - Backend selector (s3 or local)
  - Common path prefixes (raw_data/, features/, models/, results/, cache/)
  - S3-specific configuration (bucket, region, timeouts, retry)
  - Local-specific configuration (base_path, create_dirs)
- [ ] Create `src/storage/config.py` to load configuration
- [ ] Implement configuration validation using Pydantic
- [ ] Add type hints for all configuration fields
- [ ] Support environment variable override for backend selection

### 4. Implement Abstract Storage Backend
- [ ] Create `src/storage/backend.py` with `StorageBackend` abstract base class
- [ ] Define interface methods:
  - `list_objects(prefix: str) -> List[str]` - List objects with prefix
  - `get_object(key: str) -> bytes` - Download object
  - `put_object(key: str, data: bytes) -> None` - Upload object
  - `delete_object(key: str) -> None` - Delete object
  - `object_exists(key: str) -> bool` - Check if object exists
  - `get_parquet(key: str) -> pd.DataFrame` - Load parquet file
  - `put_parquet(key: str, df: pd.DataFrame) -> None` - Save parquet file
- [ ] Add type hints and docstrings for interface

### 5. Implement S3 Storage Backend
- [ ] Create `src/storage/s3_backend.py`
- [ ] Implement `S3StorageBackend` class extending `StorageBackend`
- [ ] Initialize with boto3 client and retry configuration
- [ ] Implement all interface methods using boto3
- [ ] Add error handling with exponential backoff
- [ ] Add logging for all S3 operations
- [ ] Handle S3-specific errors (NoSuchBucket, AccessDenied, NoSuchKey)

### 6. Implement Local Storage Backend
- [ ] Create `src/storage/local_backend.py`
- [ ] Implement `LocalStorageBackend` class extending `StorageBackend`
- [ ] Initialize with base path and auto-create directories option
- [ ] Implement all interface methods using pathlib
- [ ] Add error handling for filesystem operations
- [ ] Add logging for all file operations
- [ ] Ensure thread-safe file operations

### 7. Implement Storage Factory
- [ ] Create `src/storage/factory.py`
- [ ] Implement `StorageFactory` class with:
  - `create(backend_type: str, **kwargs) -> StorageBackend` - Manual backend creation
  - `from_config(config_path: str = "config/storage.yaml") -> StorageBackend` - Config-based creation
  - `from_path(path: str) -> StorageBackend` - Auto-detect from path prefix
- [ ] Add backend registry for extensibility
- [ ] Add validation for backend types

### 5. Create Connection Test Script
- [ ] Create `scripts/test_s3_connection.py`
- [ ] Test S3 client initialization
- [ ] Test list_objects operation
- [ ] Test put_object operation (upload test file)
- [ ] Test get_object operation (download test file)
- [ ] Test delete_object operation (cleanup)
- [ ] Print success/failure messages

### 8. Write Unit Tests
- [ ] Create `tests/storage/test_backend_interface.py`
- [ ] Test `StorageBackend` interface definition
- [ ] Create `tests/storage/test_s3_backend.py`
- [ ] Test `S3StorageBackend` initialization
- [ ] Test S3 operations with moto (list, get, put, delete, exists)
- [ ] Test S3 parquet operations with moto
- [ ] Test S3 error handling (bucket not found, permission denied)
- [ ] Create `tests/storage/test_local_backend.py`
- [ ] Test `LocalStorageBackend` initialization
- [ ] Test local operations with tmp_path fixture
- [ ] Test local parquet operations
- [ ] Test local error handling (permission denied, path not found)
- [ ] Create `tests/storage/test_factory.py`
- [ ] Test `StorageFactory.create()` for both backends
- [ ] Test `StorageFactory.from_config()`
- [ ] Test `StorageFactory.from_path()` with different prefixes
- [ ] Test environment variable override
- [ ] Add pytest fixtures for mocked S3 and temporary directories

---

## 📄 Configuration Files

### config/storage.yaml

```yaml
# Storage backend: 's3' or 'local'
# Can be overridden via environment variable STORAGE_BACKEND
backend: local  # Use 'local' for development, 's3' for production

# Common paths for all backends
paths:
  raw_data: raw_data/
  features: features/
  models: models/
  results: results/
  cache: cache/

# S3-specific configuration
s3:
  bucket: prevcarga-bucket-sandbox
  region: us-east-1
  prefix: ""  # Optional prefix for all S3 keys
  
  timeouts:
    connect: 5
    read: 30
  
  retry:
    max_attempts: 3
    mode: adaptive

# Local filesystem configuration
local:
  base_path: ./data  # Base directory for local storage
  create_dirs: true  # Auto-create directories if missing
```

### .env.template

```bash
# AWS Credentials (optional if using AWS CLI or IAM roles)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# S3 Configuration
S3_BUCKET_NAME=prevcarga-bucket-sandbox
```

---

## 💻 Code Structure

### src/storage/config.py

```python
"""S3 storage configuration module."""
from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel, Field


class S3Timeouts(BaseModel):
    """S3 timeout configuration."""
    connect: int = Field(default=5, description="Connection timeout in seconds")
    read: int = Field(default=30, description="Read timeout in seconds")


class S3Retry(BaseModel):
    """S3 retry configuration."""
    max_attempts: int = Field(default=3, description="Maximum retry attempts")
    mode: str = Field(default="adaptive", description="Retry mode")


class S3Paths(BaseModel):
    """S3 path configuration."""
    raw_data: str = Field(default="raw_data/", description="Raw data prefix")
    features: str = Field(default="features/", description="Features prefix")
    models: str = Field(default="models/", description="Models prefix")
    results: str = Field(default="results/", description="Results prefix")
    cache: str = Field(default="cache/", description="Cache prefix")


class S3Config(BaseModel):
    """S3 storage configuration."""
    bucket: str = Field(..., description="S3 bucket name")
    region: str = Field(default="us-east-1", description="AWS region")
    paths: S3Paths = Field(default_factory=S3Paths)
    timeouts: S3Timeouts = Field(default_factory=S3Timeouts)
    retry: S3Retry = Field(default_factory=S3Retry)


class LocalConfig(BaseModel):
    """Local storage configuration."""
    base_path: str = Field(default="./data", description="Base directory path")
    create_dirs: bool = Field(default=True, description="Auto-create directories")


class StorageConfig(BaseModel):
    """Storage configuration."""
    backend: str = Field(default="local", description="Storage backend type: 's3' or 'local'")
    paths: S3Paths
    s3: S3Config
    local: LocalConfig = Field(default_factory=LocalConfig)


def load_storage_config(config_path: str = "config/storage.yaml") -> StorageConfig:
    """Load storage configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return StorageConfig(**config_dict)
```

### src/storage/backend.py (Abstract Interface)

```python
"""Abstract storage backend interface."""
from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects with given prefix."""
        pass
    
    @abstractmethod
    def get_object(self, key: str) -> bytes:
        """Get object from storage."""
        pass
    
    @abstractmethod
    def put_object(self, key: str, data: bytes) -> None:
        """Put object to storage."""
        pass
    
    @abstractmethod
    def delete_object(self, key: str) -> None:
        """Delete object from storage."""
        pass
    
    @abstractmethod
    def object_exists(self, key: str) -> bool:
        """Check if object exists."""
        pass
    
    @abstractmethod
    def get_parquet(self, key: str) -> pd.DataFrame:
        """Load parquet file into DataFrame."""
        pass
    
    @abstractmethod
    def put_parquet(self, key: str, df: pd.DataFrame) -> None:
        """Save DataFrame as parquet file."""
        pass
```

### src/storage/s3_backend.py (S3 Implementation)

```python
"""S3 storage backend implementation."""
import logging
from typing import List, Optional
import io

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError

from src.storage.backend import StorageBackend
from src.storage.config import load_storage_config

logger = logging.getLogger(__name__)


class S3StorageBackend(StorageBackend):
    """S3 storage backend implementation."""
    
    def __init__(self, bucket: Optional[str] = None, region: Optional[str] = None) -> None:
        """Initialize S3 storage backend."""
        config = load_storage_config()
        self.bucket = bucket or config.s3.bucket
        self.region = region or config.s3.region
        
        # Configure boto3 client with retries
        boto_config = Config(
            region_name=self.region,
            connect_timeout=config.s3.timeouts.connect,
            read_timeout=config.s3.timeouts.read,
            retries={
                "max_attempts": config.s3.retry.max_attempts,
                "mode": config.s3.retry.mode,
            },
        )
        
        self.s3_client = boto3.client("s3", config=boto_config)
        logger.info(f"S3StorageBackend initialized for bucket: {self.bucket}")
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in bucket with given prefix."""
        # Implementation using s3_client.list_objects_v2
        pass
    
    def get_object(self, key: str) -> bytes:
        """Get object from S3."""
        # Implementation using s3_client.get_object
        pass
    
    def put_object(self, key: str, data: bytes) -> None:
        """Put object to S3."""
        # Implementation using s3_client.put_object
        pass
    
    def delete_object(self, key: str) -> None:
        """Delete object from S3."""
        # Implementation using s3_client.delete_object
        pass
    
    def object_exists(self, key: str) -> bool:
        """Check if object exists in S3."""
        # Implementation using s3_client.head_object
        pass
    
    def get_parquet(self, key: str) -> pd.DataFrame:
        """Load parquet file from S3 into DataFrame."""
        data = self.get_object(key)
        return pd.read_parquet(io.BytesIO(data))
    
    def put_parquet(self, key: str, df: pd.DataFrame) -> None:
        """Save DataFrame as parquet to S3."""
        buffer = io.BytesIO()
        df.to_parquet(buffer)
        self.put_object(key, buffer.getvalue())
```

### src/storage/local_backend.py (Local Implementation)

```python
"""Local filesystem storage backend implementation."""
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend implementation."""
    
    def __init__(self, base_path: str = "./data", create_dirs: bool = True) -> None:
        """Initialize local storage backend."""
        self.base_path = Path(base_path)
        if create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorageBackend initialized at: {self.base_path}")
    
    def _resolve_path(self, key: str) -> Path:
        """Resolve key to full path."""
        return self.base_path / key
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List files with given prefix."""
        prefix_path = self._resolve_path(prefix)
        if prefix_path.is_file():
            return [prefix]
        if not prefix_path.exists():
            return []
        # Implementation using pathlib.glob
        pass
    
    def get_object(self, key: str) -> bytes:
        """Get file from local storage."""
        path = self._resolve_path(key)
        return path.read_bytes()
    
    def put_object(self, key: str, data: bytes) -> None:
        """Put file to local storage."""
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    
    def delete_object(self, key: str) -> None:
        """Delete file from local storage."""
        path = self._resolve_path(key)
        if path.exists():
            path.unlink()
    
    def object_exists(self, key: str) -> bool:
        """Check if file exists."""
        return self._resolve_path(key).exists()
    
    def get_parquet(self, key: str) -> pd.DataFrame:
        """Load parquet file into DataFrame."""
        path = self._resolve_path(key)
        return pd.read_parquet(path)
    
    def put_parquet(self, key: str, df: pd.DataFrame) -> None:
        """Save DataFrame as parquet file."""
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
```

### src/storage/factory.py (Backend Factory)

```python
"""Storage backend factory."""
import logging
import os
from typing import Optional

from src.storage.backend import StorageBackend
from src.storage.config import load_storage_config
from src.storage.s3_backend import S3StorageBackend
from src.storage.local_backend import LocalStorageBackend

logger = logging.getLogger(__name__)


class StorageFactory:
    """Factory for creating storage backends."""
    
    @staticmethod
    def create(backend_type: str, **kwargs) -> StorageBackend:
        """Create storage backend by type."""
        if backend_type == "s3":
            return S3StorageBackend(**kwargs)
        elif backend_type == "local":
            return LocalStorageBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    @staticmethod
    def from_config(config_path: str = "config/storage.yaml") -> StorageBackend:
        """Create storage backend from configuration file."""
        config = load_storage_config(config_path)
        
        # Allow environment variable override
        backend_type = os.getenv("STORAGE_BACKEND", config.backend)
        
        logger.info(f"Creating {backend_type} storage backend from config")
        
        if backend_type == "s3":
            return S3StorageBackend(
                bucket=config.s3.bucket,
                region=config.s3.region
            )
        elif backend_type == "local":
            return LocalStorageBackend(
                base_path=config.local.base_path,
                create_dirs=config.local.create_dirs
            )
        else:
            raise ValueError(f"Unknown backend type in config: {backend_type}")
    
    @staticmethod
    def from_path(path: str) -> StorageBackend:
        """Auto-detect backend from path prefix."""
        if path.startswith("s3://"):
            # Extract bucket from s3://bucket/key
            bucket = path.split("/")[2]
            return S3StorageBackend(bucket=bucket)
        elif path.startswith("file://"):
            # Extract path from file:///path
            local_path = path[7:]
            return LocalStorageBackend(base_path=local_path)
        else:
            # Assume local relative path
            return LocalStorageBackend()
```

---

## 🧪 Testing & Validation

### Validation Commands

```bash
# Install moto for S3 mocking in tests
uv add --dev moto[s3]>=4.2.0

# Run connection test script for both backends
python scripts/test_storage_connection.py

# Run unit tests for all backends
pytest tests/storage/test_backends.py -v

# Test storage backends in Python REPL
python -c "from src.storage.factory import StorageFactory; backend = StorageFactory.create('local'); print('Local storage OK')"
python -c "from src.storage.factory import StorageFactory; backend = StorageFactory.create('s3', bucket='test'); print('S3 storage OK')"
python -c "from src.storage.factory import StorageFactory; backend = StorageFactory.from_config(); print('Config-based storage OK')"
```

### Success Criteria
- [ ] `StorageBackend` interface defined and documented
- [ ] `S3StorageBackend` implemented and tested
- [ ] `LocalStorageBackend` implemented and tested
- [ ] `StorageFactory` creates correct backends
- [ ] AWS credentials configured correctly for S3
- [ ] S3 bucket accessible
- [ ] Local storage directory created automatically
- [ ] Connection test script passes for both backends
- [ ] All unit tests pass (S3 with moto mocking, local with temp directories)
- [ ] Both backends support list, get, put, delete operations
- [ ] Parquet operations work correctly on both backends
- [ ] Backend selection via config works
- [ ] Backend selection via environment variable works
- [ ] Path-based auto-detection works
- [ ] Error handling works as expected for both backends

---

## 📝 Technical Notes

- Use moto library for mocking S3 in unit tests
- Use tempfile/tmp_path fixtures for local backend tests
- Never commit AWS credentials to repository
- Use structured logging for all storage operations
- Implement exponential backoff for S3 retries only
- Handle common S3 errors (NoSuchBucket, AccessDenied, NoSuchKey)
- Handle filesystem errors gracefully (PermissionError, FileNotFoundError)
- Consider using presigned URLs for large file transfers (future)
- boto3 client is thread-safe and can be reused
- Abstract interface enables easy addition of new backends (Azure, GCS)
- Local backend is faster for development and testing
- Path prefixes (s3://, file://) enable transparent backend selection

---

## 🔗 Dependencies

**Depends On:**
- PC-001-00: Repository Structure Setup
- PC-002-00: Python Environment with uv

**Blocks:**
- Epic-01: Data Infrastructure Layer (all tickets)

**Related:**
- PC-004-00: Structured Logging Framework (for logging S3 operations)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] All implementation tasks completed
- [ ] Configuration files created and documented (storage.yaml with both backends)
- [ ] `StorageBackend` interface defined
- [ ] `S3StorageBackend` implementation complete with all methods
- [ ] `LocalStorageBackend` implementation complete with all methods
- [ ] `StorageFactory` implementation complete
- [ ] Connection test script passes for both backends
- [ ] Unit tests pass with >80% coverage for all backends
- [ ] Integration tests verify backend switching
- [ ] Code reviewed and approved
- [ ] Documentation updated in README.md (both S3 and local setup)
- [ ] CLI integration documented (--storage-backend flag)
- [ ] Ready for Epic-01 data loading with unified interface

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next Ticket:** [PC-004-00: Structured Logging Framework](PC-004-00-structured-logging-framework.md)
