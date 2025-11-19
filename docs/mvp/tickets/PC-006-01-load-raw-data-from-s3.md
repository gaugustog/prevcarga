# PC-006-01: Load Raw Load Data from S3

**Ticket ID:** PC-006-01  
**Epic:** [Epic-01: Data Infrastructure Layer](../epics/Epic-01.md)  
**User Story:** US-1.1  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `S3ParquetLoader` class to load historical electric load data from S3 with support for parallel loading, date/area filtering, and retry logic. This loader serves as the foundation for all data loading operations.

**As a** data scientist  
**I want to** load historical electric load data from S3  
**So that** I can use it for model training and evaluation

---

## ✅ Acceptance Criteria

- [ ] Load Parquet files from S3 paths following pattern: `s3://bucket/raw_data/{year}/{month}/carga_horaria_{cod_area}_{YYYYMMDD}.parquet`
- [ ] Support filtering by area codes (e.g., "RJ", "SP", "SECO")
- [ ] Support filtering by date ranges (start_date, end_date)
- [ ] Handle missing files gracefully with informative error messages
- [ ] Return pandas DataFrame with standardized columns: `[timestamp, cod_area, carga_mwh]`
- [ ] Support batch loading of multiple areas/dates in parallel
- [ ] Load performance: <5s for 1 year of data (1 area)

---

## 🔧 Implementation Tasks

### 1. Create S3ParquetLoader Base Class
- [ ] Create `src/data/loaders.py`
- [ ] Implement `S3ParquetLoader` class with boto3 integration
- [ ] Add initialization with bucket name, region, max_workers
- [ ] Configure S3 client with connection pooling
- [ ] Add logging for all S3 operations

### 2. Implement load_carga() Method
- [ ] Create method signature: `load_carga(areas: List[str], start_date: date, end_date: date, validate: bool = True) -> pd.DataFrame`
- [ ] Implement path generation for date range and areas
- [ ] Add date range iteration logic
- [ ] Build S3 key list following naming convention
- [ ] Sort and deduplicate paths

### 3. Implement Parallel Loading
- [ ] Use ThreadPoolExecutor for concurrent S3 downloads
- [ ] Implement `_load_single_parquet()` helper method
- [ ] Add S3 object retrieval with boto3
- [ ] Parse Parquet from bytes stream
- [ ] Concatenate DataFrames from multiple files
- [ ] Sort by area and timestamp

### 4. Add Error Handling and Retries
- [ ] Implement exponential backoff for retries (max 3 attempts)
- [ ] Handle `NoSuchKey` error gracefully (log warning, continue)
- [ ] Handle `AccessDenied` error (raise with clear message)
- [ ] Handle network errors with retry logic
- [ ] Log all errors with context (file path, attempt number)

### 5. Implement Path Generation Utility
- [ ] Create `_generate_paths()` method
- [ ] Support multiple areas
- [ ] Support date range iteration
- [ ] Format paths with year/month subdirectories
- [ ] Handle area code normalization (uppercase)

### 6. Add Performance Optimization
- [ ] Configure optimal max_workers (default: 4)
- [ ] Add connection pooling for S3 client
- [ ] Implement batch size configuration
- [ ] Add progress logging for large loads
- [ ] Profile and optimize memory usage

### 7. Write Unit Tests
- [ ] Create `tests/data/test_loaders.py`
- [ ] Test successful single file load (moto)
- [ ] Test multiple files parallel load
- [ ] Test missing file handling
- [ ] Test date range filtering
- [ ] Test area filtering
- [ ] Test retry logic
- [ ] Test error scenarios (invalid bucket, permissions)

### 8. Write Performance Tests
- [ ] Create benchmark for 1 year load (<5s target)
- [ ] Test concurrent loading efficiency
- [ ] Profile memory usage
- [ ] Document performance characteristics

---

## 💻 Implementation Details

### S3ParquetLoader Class Structure

```python
"""S3 Parquet data loader with parallel support."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import List, Optional

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError

from src.storage.config import load_storage_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class S3ParquetLoader:
    """Load Parquet data from S3 with parallel support."""
    
    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        max_workers: int = 4
    ) -> None:
        """
        Initialize S3 Parquet loader.
        
        Args:
            bucket: S3 bucket name (uses config if not provided)
            region: AWS region (uses config if not provided)
            max_workers: Maximum parallel download threads
        """
        config = load_storage_config()
        self.bucket = bucket or config.s3.bucket
        self.region = region or config.s3.region
        self.max_workers = max_workers
        
        # Configure boto3 client
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
        logger.info(
            f"S3ParquetLoader initialized: bucket={self.bucket}, "
            f"region={self.region}, max_workers={self.max_workers}"
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
            >>> loader = S3ParquetLoader()
            >>> df = loader.load_carga(["RJ"], date(2024, 1, 1), date(2024, 1, 31))
            >>> print(df.shape)
            (1488, 3)  # 31 days * 48 semi-hourly records
        """
        logger.info(
            f"Loading carga data: areas={areas}, "
            f"date_range={start_date} to {end_date}"
        )
        
        # Generate S3 paths
        paths = self._generate_paths("carga_horaria", areas, start_date, end_date)
        logger.debug(f"Generated {len(paths)} S3 paths to load")
        
        # Load in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            dfs = list(executor.map(self._load_single_parquet, paths))
        
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
    
    def _load_single_parquet(self, s3_key: str) -> Optional[pd.DataFrame]:
        """
        Load single Parquet file with retry logic.
        
        Args:
            s3_key: S3 object key
        
        Returns:
            DataFrame or None if file not found
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Loading s3://{self.bucket}/{s3_key} (attempt {attempt + 1})")
                
                response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                df = pd.read_parquet(response["Body"])
                
                logger.debug(f"Loaded {len(df)} records from {s3_key}")
                return df
                
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                
                if error_code == "NoSuchKey":
                    logger.warning(f"File not found: s3://{self.bucket}/{s3_key}")
                    return None
                
                if error_code == "AccessDenied":
                    logger.error(f"Access denied: s3://{self.bucket}/{s3_key}")
                    raise
                
                # Retry on other errors
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Error loading {s3_key}: {e}. "
                        f"Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"Failed to load {s3_key} after {max_retries} attempts"
                    )
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
def test_load_carga_success():
    """Test successful load of carga data."""
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
    
    # Test loader
    loader = S3ParquetLoader(bucket="test-bucket")
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
