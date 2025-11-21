# PC-006-01: Load Raw Load Data from Storage

**Ticket ID:** PC-006-01  
**Epic:** [Epic-01: Data Infrastructure Layer](../epics/Epic-01.md)  
**User Story:** US-1.1  
**Story Points:** 6  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `DataLoader` class with unified storage backend support to load historical electric load data from either S3 (production) or local filesystem (development) with parallel loading, date/area filtering, and retry logic. This loader serves as the foundation for all data loading operations.

**As a** data scientist  
**I want to** load historical electric load data from any storage backend (S3 or local)  
**So that** I can develop locally and deploy to production without code changes

---

## ✅ Acceptance Criteria

- [ ] Load Parquet files from unified paths following pattern: `raw_data/{year}/{month}/carga_horaria_{cod_area}_{YYYYMMDD}.parquet`
- [ ] Auto-detect storage backend from configuration or path prefix (s3://, file://, or relative)
- [ ] Support S3 backend: `s3://bucket/raw_data/...`
- [ ] Support local backend: `./data/raw_data/...` or `/absolute/path/raw_data/...`
- [ ] Support filtering by area codes (e.g., "RJ", "SP", "SECO")
- [ ] Support filtering by date ranges (start_date, end_date)
- [ ] Handle missing files gracefully with informative error messages
- [ ] Return pandas DataFrame with standardized columns: `[timestamp, cod_area, carga_mwh]`
- [ ] Support batch loading of multiple areas/dates in parallel (both backends)
- [ ] Load performance: <5s for 1 year of data (1 area) on both backends
- [ ] CLI supports `--storage-backend` flag to override config

---

## 🔧 Implementation Tasks

### 1. Create DataLoader Base Class
- [ ] Create `src/data/loaders.py`
- [ ] Implement `DataLoader` class using `StorageBackend` abstraction from Epic-00
- [ ] Add initialization with optional storage_backend parameter
- [ ] Default to StorageFactory.from_config() if no backend provided
- [ ] Configure max_workers for parallel loading
- [ ] Add logging for all storage operations

### 2. Implement load_carga() Method
- [ ] Create method signature: `load_carga(areas: List[str], start_date: date, end_date: date, validate: bool = True) -> pd.DataFrame`
- [ ] Implement path generation for date range and areas (backend-agnostic)
- [ ] Add date range iteration logic
- [ ] Build path list following naming convention (works for both S3 and local)
- [ ] Sort and deduplicate paths

### 3. Implement Parallel Loading
- [ ] Use ThreadPoolExecutor for concurrent loading (both backends)
- [ ] Implement `_load_single_file()` helper method using StorageBackend
- [ ] Use storage_backend.get_parquet() for retrieval
- [ ] Handle both S3 and local filesystem transparently
- [ ] Concatenate DataFrames from multiple files
- [ ] Sort by area and timestamp

### 4. Add Error Handling and Retries
- [ ] Implement exponential backoff for retries (S3 backend, max 3 attempts)
- [ ] Handle file not found gracefully for both backends (log warning, continue)
- [ ] Handle permission errors (S3: AccessDenied, Local: PermissionError)
- [ ] Handle network errors with retry logic (S3 only)
- [ ] Handle filesystem errors for local backend (FileNotFoundError, OSError)
- [ ] Log all errors with context (file path, backend type, attempt number)

### 5. Implement Path Generation Utility
- [ ] Create `_generate_paths()` method (backend-agnostic)
- [ ] Support multiple areas
- [ ] Support date range iteration
- [ ] Format paths with year/month subdirectories (relative paths)
- [ ] Handle area code normalization (uppercase)
- [ ] Paths work for both S3 keys and local filesystem

### 6. Add Performance Optimization
- [ ] Configure optimal max_workers (default: 4 for both backends)
- [ ] S3 backend: connection pooling via StorageBackend
- [ ] Local backend: optimize file I/O
- [ ] Implement batch size configuration
- [ ] Add progress logging for large loads
- [ ] Profile and optimize memory usage for both backends

### 7. Write Unit Tests
- [ ] Create `tests/data/test_loaders.py`
- [ ] Test successful single file load with S3 backend (moto)
- [ ] Test successful single file load with local backend (tmp_path fixture)
- [ ] Test multiple files parallel load (both backends)
- [ ] Test missing file handling (both backends)
- [ ] Test date range filtering (both backends)
- [ ] Test area filtering (both backends)
- [ ] Test retry logic (S3 backend)
- [ ] Test error scenarios (S3: invalid bucket/permissions, Local: path not found/permissions)
- [ ] Test StorageBackend integration

### 8. Write Performance Tests
- [ ] Create benchmark for 1 year load with S3 backend (<5s target)
- [ ] Create benchmark for 1 year load with local backend (<3s target)
- [ ] Test concurrent loading efficiency (both backends)
- [ ] Profile memory usage (both backends)
- [ ] Compare performance between backends
- [ ] Document performance characteristics

---

## 💻 Implementation Details

### DataLoader Class Structure (Unified Storage)

```python
"""Unified data loader supporting multiple storage backends."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import List, Optional

import pandas as pd

from src.storage.backend import StorageBackend
from src.storage.factory import StorageFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load data from any storage backend (S3 or local) with parallel support."""
    
    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        max_workers: int = 4
    ) -> None:
        """
        Initialize data loader with storage backend.
        
        Args:
            storage_backend: Storage backend instance. If None, creates from config.
            max_workers: Maximum parallel loading threads
        """
        self.storage = storage_backend or StorageFactory.from_config()
        self.max_workers = max_workers
        
        logger.info(
            f"DataLoader initialized: backend={self.storage.__class__.__name__}, "
            f"max_workers={self.max_workers}"
        )
    
    def load_carga(
        self,
        areas: List[str],
        start_date: date,
        end_date: date,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load electric load data for specified areas and date range.
        
        Args:
            areas: List of area codes (e.g., ["RJ", "SP", "SECO"])
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            validate: Whether to validate schema after loading
        
        Returns:
            DataFrame with columns: [timestamp, cod_area, carga_mwh]
        
        Example:
            >>> from src.storage.factory import StorageFactory
            >>> # Use local backend for development
            >>> backend = StorageFactory.create("local", base_path="./data")
            >>> loader = DataLoader(storage_backend=backend)
            >>> df = loader.load_carga(["RJ"], date(2024, 1, 1), date(2024, 1, 31))
            >>> print(df.shape)
            (1488, 3)  # 31 days * 48 semi-hourly records
            
            >>> # Use S3 backend for production (from config)
            >>> loader = DataLoader()  # Uses StorageFactory.from_config()
            >>> df = loader.load_carga(["RJ"], date(2024, 1, 1), date(2024, 1, 31))
        """
        logger.info(
            f"Loading carga data: areas={areas}, "
            f"date_range={start_date} to {end_date}, "
            f"backend={self.storage.__class__.__name__}"
        )
        
        # Generate storage paths (backend-agnostic)
        paths = self._generate_paths("carga_horaria", areas, start_date, end_date)
        logger.debug(f"Generated {len(paths)} paths to load")
        
        # Load in parallel using storage backend
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            dfs = list(executor.map(self._load_single_file, paths))
        
        # Filter out None (missing files)
        dfs = [df for df in dfs if df is not None]
        
        if not dfs:
            logger.warning("No data loaded - all files missing or empty")
            return pd.DataFrame(columns=["timestamp", "cod_area", "carga_mwh"])
        
        # Concatenate and sort
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values(["cod_area", "timestamp"]).reset_index(drop=True)
        
        elapsed = time.time() - start_time
        logger.info(
            f"Loaded {len(df)} records in {elapsed:.2f}s "
            f"({len(df)/elapsed:.0f} records/s)"
        )
        
        # Optional validation
        if validate:
            from src.data.validators import DataValidator, CargaSchema
            validator = DataValidator(CargaSchema)
            df = validator.validate(df)
        
        return df
    
    def _load_single_file(self, path: str) -> Optional[pd.DataFrame]:
        """
        Load single Parquet file using storage backend.
        
        Args:
            path: Storage path (works for both S3 and local)
        
        Returns:
            DataFrame or None if file not found
        """
        try:
            logger.debug(f"Loading {path}")
            
            # Use storage backend's get_parquet method
            # Handles retries internally for S3, direct read for local
            df = self.storage.get_parquet(path)
            
            logger.debug(f"Loaded {len(df)} records from {path}")
            return df
            
        except FileNotFoundError:
            # Local backend: file not found
            logger.warning(f"File not found: {path}")
            return None
            
        except Exception as e:
            # Check if it's an S3 "not found" error
            if "NoSuchKey" in str(e) or "404" in str(e):
                logger.warning(f"File not found: {path}")
                return None
            
            # For other errors, log and re-raise
            logger.error(f"Error loading {path}: {e}")
            raise
    
    def _generate_paths(
        self,
        prefix: str,
        areas: List[str],
        start_date: date,
        end_date: date
    ) -> List[str]:
        """
        Generate S3 paths for date range and areas.
        
        Args:
            prefix: File prefix (e.g., "carga_horaria")
            areas: List of area codes
            start_date: Start date
            end_date: End date
        
        Returns:
            List of S3 keys
        """
        from src.data.utils import DateRangeGenerator
        
        paths = []
        for area in areas:
            area_upper = area.upper()
            for dt in DateRangeGenerator(start_date, end_date):
                path = (
                    f"raw_data/{dt.year}/{dt.month:02d}/"
                    f"{prefix}_{area_upper}_{dt.strftime('%Y%m%d')}.parquet"
                )
                paths.append(path)
        
        return paths
```

### DateRangeGenerator Utility

```python
"""Utility classes for data operations."""
from datetime import date, timedelta
from typing import Iterator


class DateRangeGenerator:
    """Generate date range for iteration."""
    
    def __init__(self, start_date: date, end_date: date) -> None:
        """
        Initialize date range generator.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
        """
        self.start_date = start_date
        self.end_date = end_date
    
    def __iter__(self) -> Iterator[date]:
        """Iterate through date range."""
        current = self.start_date
        while current <= self.end_date:
            yield current
            current += timedelta(days=1)
```

---

## 🧪 Testing & Validation

### Unit Tests with moto

```python
"""Unit tests for S3ParquetLoader."""
import pytest
from datetime import date
from moto import mock_s3
import boto3
import pandas as pd

from src.data.loaders import S3ParquetLoader


@mock_s3
def test_load_carga_with_s3_backend():
    """Test successful load of carga data with S3 backend."""
    # Setup mock S3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    
    # Create test data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=48, freq="30T"),
        "cod_area": "RJ",
        "carga_mwh": range(1000, 1048)
    })
    
    # Upload to S3
    parquet_bytes = df.to_parquet()
    s3.put_object(
        Bucket="test-bucket",
        Key="raw_data/2024/01/carga_horaria_RJ_20240101.parquet",
        Body=parquet_bytes
    )
    
    # Test loader with S3 backend
    from src.storage.factory import StorageFactory
    backend = StorageFactory.create("s3", bucket="test-bucket")
    loader = DataLoader(storage_backend=backend)
    result = loader.load_carga(
        areas=["RJ"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        validate=False
    )
    
    # Assertions
    assert len(result) == 48
    assert result["cod_area"].unique()[0] == "RJ"
    assert result["carga_mwh"].min() == 1000
    assert result["carga_mwh"].max() == 1047


def test_load_carga_with_local_backend(tmp_path):
    """Test successful load of carga data with local backend."""
    # Create test data directory
    data_dir = tmp_path / "raw_data" / "2024" / "01"
    data_dir.mkdir(parents=True)
    
    # Create test data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=48, freq="30T"),
        "cod_area": "RJ",
        "carga_mwh": range(1000, 1048)
    })
    
    # Save to local file
    file_path = data_dir / "carga_horaria_RJ_20240101.parquet"
    df.to_parquet(file_path)
    
    # Test loader with local backend
    from src.storage.factory import StorageFactory
    backend = StorageFactory.create("local", base_path=str(tmp_path))
    loader = DataLoader(storage_backend=backend)
    result = loader.load_carga(
        areas=["RJ"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        validate=False
    )
    
    # Assertions
    assert len(result) == 48
    assert result["cod_area"].unique()[0] == "RJ"
    assert result["carga_mwh"].min() == 1000


@mock_s3
def test_load_missing_file():
    """Test handling of missing S3 file."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    
    loader = S3ParquetLoader(bucket="test-bucket")
    result = loader.load_carga(
        areas=["SP"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        validate=False
    )
    
    # Should return empty DataFrame, not raise error
    assert len(result) == 0
    assert list(result.columns) == ["timestamp", "cod_area", "carga_mwh"]


@mock_s3
def test_load_multiple_areas_parallel():
    """Test parallel loading of multiple areas."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    
    # Upload data for multiple areas
    for area in ["RJ", "SP", "MG"]:
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=48, freq="30T"),
            "cod_area": area,
            "carga_mwh": range(1000, 1048)
        })
        s3.put_object(
            Bucket="test-bucket",
            Key=f"raw_data/2024/01/carga_horaria_{area}_20240101.parquet",
            Body=df.to_parquet()
        )
    
    loader = S3ParquetLoader(bucket="test-bucket", max_workers=3)
    result = loader.load_carga(
        areas=["RJ", "SP", "MG"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        validate=False
    )
    
    assert len(result) == 144  # 48 records * 3 areas
    assert set(result["cod_area"].unique()) == {"RJ", "SP", "MG"}


@mock_s3
def test_load_date_range():
    """Test loading data across date range."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    
    # Upload data for 3 days
    for day in range(1, 4):
        df = pd.DataFrame({
            "timestamp": pd.date_range(f"2024-01-{day:02d}", periods=48, freq="30T"),
            "cod_area": "RJ",
            "carga_mwh": range(1000, 1048)
        })
        s3.put_object(
            Bucket="test-bucket",
            Key=f"raw_data/2024/01/carga_horaria_RJ_2024010{day}.parquet",
            Body=df.to_parquet()
        )
    
    loader = S3ParquetLoader(bucket="test-bucket")
    result = loader.load_carga(
        areas=["RJ"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        validate=False
    )
    
    assert len(result) == 144  # 48 records * 3 days
```

### Performance Test

```python
"""Performance tests for data loading."""
import time
import pytest
from datetime import date


@pytest.mark.performance
def test_load_one_year_performance():
    """Test load performance for 1 year of data."""
    loader = S3ParquetLoader(bucket="test-bucket", max_workers=8)
    
    start_time = time.time()
    df = loader.load_carga(
        areas=["SECO"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        validate=False
    )
    elapsed = time.time() - start_time
    
    # Must complete in under 5 seconds
    assert elapsed < 5.0, f"Load took {elapsed:.2f}s (target: <5s)"
    assert len(df) > 0
```

---

## 📝 Technical Notes

- Use ThreadPoolExecutor for I/O-bound parallel S3 downloads
- Exponential backoff prevents S3 throttling
- Connection pooling improves performance for multiple requests
- Missing files logged but don't stop pipeline (graceful degradation)
- Area codes normalized to uppercase for consistency
- S3 path format: `raw_data/{year}/{month}/{prefix}_{area}_{YYYYMMDD}.parquet`

---

## 🔗 Dependencies

**Depends On:**
- PC-001-00: Repository Structure Setup
- PC-002-00: Python Environment with uv
- PC-003-00: S3 Storage Configuration
- PC-004-00: Structured Logging Framework

**Blocks:**
- PC-007-01: Validate Data Schemas
- PC-011-01: Load Temperature and Holiday Data

**Related:**
- Epic-02: Feature Engineering (consumer of loaded data)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] All implementation tasks completed
- [ ] S3ParquetLoader class implemented with all methods
- [ ] Parallel loading working with ThreadPoolExecutor
- [ ] Retry logic with exponential backoff working
- [ ] Unit tests pass with >80% coverage
- [ ] Performance test passes (<5s for 1 year)
- [ ] Error handling tested (missing files, permissions, network)
- [ ] Code reviewed and approved
- [ ] Documentation complete with examples
- [ ] Integration with logging framework verified
- [ ] Ready for PC-007-01 (validation layer)

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next Ticket:** [PC-007-01: Validate Data Schemas](PC-007-01-validate-data-schemas.md)
