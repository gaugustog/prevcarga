# PC-003-00: S3 Storage Configuration

**Ticket ID:** PC-003-00  
**Epic:** [Epic-00: Project Foundation & Setup](../epics/Epic-00.md)  
**User Story:** US-00.3  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Configure AWS S3 storage infrastructure with bucket setup, access configuration, and a boto3 client wrapper to enable reliable storage and retrieval of raw data, models, and results.

**As a** developer  
**I want** S3 storage configured and accessible  
**So that** I can store and retrieve raw data, models, and results

---

## ✅ Acceptance Criteria

- [ ] AWS credentials configured (via environment variables or AWS CLI)
- [ ] S3 bucket created or identified: `prevcarga-bucket-sandbox`
- [ ] Bucket structure documented (raw_data/, features/, models/, results/, cache/)
- [ ] boto3 S3 client wrapper created with basic operations (list, get, put, delete)
- [ ] Connection test script created and passing
- [ ] S3 configuration module created in `src/storage/`

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
  - Bucket name
  - Region
  - Path prefixes
  - Timeout settings
  - Retry configuration
- [ ] Create `src/storage/config.py` to load configuration
- [ ] Implement configuration validation using Pydantic
- [ ] Add type hints for all configuration fields

### 4. Implement S3 Client Wrapper
- [ ] Create `src/storage/s3_client.py`
- [ ] Implement `S3Client` class with:
  - `__init__(bucket_name: str, region: str = "us-east-1")`
  - `list_objects(prefix: str) -> List[str]` - List objects with prefix
  - `get_object(key: str) -> bytes` - Download object
  - `put_object(key: str, data: bytes) -> None` - Upload object
  - `delete_object(key: str) -> None` - Delete object
  - `object_exists(key: str) -> bool` - Check if object exists
  - `get_parquet(key: str) -> pd.DataFrame` - Load parquet file
  - `put_parquet(key: str, df: pd.DataFrame) -> None` - Save parquet file
- [ ] Implement error handling with retries
- [ ] Add logging for all operations
- [ ] Add type hints and docstrings

### 5. Create Connection Test Script
- [ ] Create `scripts/test_s3_connection.py`
- [ ] Test S3 client initialization
- [ ] Test list_objects operation
- [ ] Test put_object operation (upload test file)
- [ ] Test get_object operation (download test file)
- [ ] Test delete_object operation (cleanup)
- [ ] Print success/failure messages

### 6. Write Unit Tests
- [ ] Create `tests/storage/test_s3_client.py`
- [ ] Test S3Client initialization
- [ ] Test list_objects with moto (S3 mocking)
- [ ] Test get_object with moto
- [ ] Test put_object with moto
- [ ] Test delete_object with moto
- [ ] Test object_exists with moto
- [ ] Test error handling (bucket not found, permission denied)
- [ ] Add pytest fixtures for mocked S3

---

## 📄 Configuration Files

### config/storage.yaml

```yaml
s3:
  bucket: prevcarga-bucket-sandbox
  region: us-east-1
  
  paths:
    raw_data: raw_data/
    features: features/
    models: models/
    results: results/
    cache: cache/
  
  timeouts:
    connect: 5
    read: 30
  
  retry:
    max_attempts: 3
    mode: adaptive
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


class StorageConfig(BaseModel):
    """Storage configuration."""
    s3: S3Config


def load_storage_config(config_path: str = "config/storage.yaml") -> StorageConfig:
    """Load storage configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return StorageConfig(**config_dict)
```

### src/storage/s3_client.py (Core Structure)

```python
"""S3 client wrapper with basic operations."""
import logging
from typing import List, Optional

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError

from src.storage.config import load_storage_config

logger = logging.getLogger(__name__)


class S3Client:
    """S3 client wrapper for common operations."""
    
    def __init__(self, bucket_name: Optional[str] = None, region: Optional[str] = None) -> None:
        """Initialize S3 client."""
        config = load_storage_config()
        self.bucket_name = bucket_name or config.s3.bucket
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
        logger.info(f"S3Client initialized for bucket: {self.bucket_name}")
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in bucket with given prefix."""
        # Implementation here
        pass
    
    def get_object(self, key: str) -> bytes:
        """Get object from S3."""
        # Implementation here
        pass
    
    def put_object(self, key: str, data: bytes) -> None:
        """Put object to S3."""
        # Implementation here
        pass
    
    def delete_object(self, key: str) -> None:
        """Delete object from S3."""
        # Implementation here
        pass
    
    def object_exists(self, key: str) -> bool:
        """Check if object exists in S3."""
        # Implementation here
        pass
    
    def get_parquet(self, key: str) -> pd.DataFrame:
        """Load parquet file from S3 into DataFrame."""
        # Implementation here
        pass
    
    def put_parquet(self, key: str, df: pd.DataFrame) -> None:
        """Save DataFrame as parquet to S3."""
        # Implementation here
        pass
```

---

## 🧪 Testing & Validation

### Validation Commands

```bash
# Install moto for S3 mocking in tests
uv add --dev moto[s3]>=4.2.0

# Run connection test script
python scripts/test_s3_connection.py

# Run unit tests
pytest tests/storage/test_s3_client.py -v

# Test S3 client in Python REPL
python -c "from src.storage.s3_client import S3Client; client = S3Client(); print('S3 OK')"
```

### Success Criteria
- [ ] AWS credentials configured correctly
- [ ] S3 bucket accessible
- [ ] Connection test script passes
- [ ] All unit tests pass (with moto mocking)
- [ ] S3Client can list, get, put, delete objects
- [ ] Parquet operations work correctly
- [ ] Error handling works as expected

---

## 📝 Technical Notes

- Use moto library for mocking S3 in unit tests
- Never commit AWS credentials to repository
- Use structured logging for all S3 operations
- Implement exponential backoff for retries
- Handle common S3 errors (NoSuchBucket, AccessDenied, NoSuchKey)
- Consider using presigned URLs for large file transfers (future)
- boto3 client is thread-safe and can be reused

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
- [ ] Configuration files created and documented
- [ ] S3Client implementation complete with all methods
- [ ] Connection test script passes
- [ ] Unit tests pass with >80% coverage
- [ ] Code reviewed and approved
- [ ] Documentation updated in README.md
- [ ] Ready for Epic-01 data loading

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next Ticket:** [PC-004-00: Structured Logging Framework](PC-004-00-structured-logging-framework.md)
