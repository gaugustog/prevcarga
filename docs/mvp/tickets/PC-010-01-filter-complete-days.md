# PC-010-01: Filter Complete Days and Validate Continuity

**Ticket ID:** PC-010-01  
**Epic:** [Epic-01: Data Infrastructure Layer](../epics/Epic-01.md)  
**User Story:** US-1.5  
**Story Points:** 3  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement filtering and validation to ensure only complete days with proper temporal continuity are used for model training, preventing bias from incomplete or irregular data.

**As a** data quality engineer  
**I want to** ensure only complete days with proper temporal continuity are used  
**So that** models are trained on consistent, high-quality data

---

## ✅ Acceptance Criteria

- [ ] Filter datasets to include only complete days (48 semi-hourly or 24 hourly records)
- [ ] Validate timestamp intervals are consistent (30-min or 60-min)
- [ ] Detect and flag duplicate timestamps within same area
- [ ] Verify chronological ordering per area
- [ ] Remove days with excessive missing values (>10% after imputation)
- [ ] Log statistics on filtered days (count removed, reasons)
- [ ] Performance: <1s for filtering 1 year of data

---

## 🔧 Implementation Tasks

### 1. Create CompleteDayFilter Class
- [ ] Create filter with configurable thresholds
- [ ] Support semi-hourly (48 records) and hourly (24 records) detection
- [ ] Add date column extraction from timestamp
- [ ] Group by area and date
- [ ] Count records per day
- [ ] Filter to complete days only

### 2. Implement Timestamp Continuity Validator
- [ ] Check intervals are consistent (30-min or 60-min)
- [ ] Detect gaps (missing intervals)
- [ ] Detect overlaps (duplicate intervals)
- [ ] Group validation by area
- [ ] Return validation report

### 3. Add Duplicate Detection
- [ ] Find duplicate (timestamp, cod_area) pairs
- [ ] Keep first occurrence, remove duplicates
- [ ] Log duplicate details
- [ ] Add option to raise error vs. filter

### 4. Create Chronological Order Validator
- [ ] Verify timestamps sorted within area
- [ ] Detect out-of-order records
- [ ] Option to auto-sort vs. raise error
- [ ] Log ordering issues

### 5. Implement Quality Threshold Filter
- [ ] Calculate missing value percentage per day
- [ ] Remove days exceeding threshold (default: 10%)
- [ ] Log filtered days with reasons
- [ ] Track statistics before/after

### 6. Create FilteringReport
- [ ] Total days processed
- [ ] Complete days kept
- [ ] Incomplete days removed
- [ ] Reasons breakdown (missing records, duplicates, NAs)
- [ ] Affected areas
- [ ] Date ranges impacted

### 7. Write Tests
- [ ] Test complete day detection (48/24 records)
- [ ] Test incomplete day filtering
- [ ] Test duplicate detection and removal
- [ ] Test chronological validation
- [ ] Test edge cases (DST transitions, leap years)
- [ ] Test performance with 1 year data

---

## 💻 Implementation Details

```python
"""Complete day filtering and data quality validation."""
import pandas as pd
from typing import Dict, List, Tuple
from datetime import date

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FilteringReport:
    """Report on filtering operations."""
    
    def __init__(self) -> None:
        self.total_days: int = 0
        self.complete_days: int = 0
        self.incomplete_days: int = 0
        self.reasons: Dict[str, int] = {}
        self.affected_areas: List[str] = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_days": self.total_days,
            "complete_days": self.complete_days,
            "incomplete_days": self.incomplete_days,
            "retention_rate": self.complete_days / self.total_days if self.total_days > 0 else 0,
            "reasons": self.reasons,
            "affected_areas": self.affected_areas
        }
    
    def log_summary(self) -> None:
        """Log filtering summary."""
        logger.info(
            f"Filtering complete: {self.complete_days}/{self.total_days} days kept "
            f"({self.retention_rate:.1%}), {self.incomplete_days} removed"
        )
        if self.reasons:
            logger.info(f"Removal reasons: {self.reasons}")


class CompleteDayFilter:
    """Filter datasets to complete days only."""
    
    def __init__(
        self,
        semi_hourly_records: int = 48,
        hourly_records: int = 24,
        max_missing_pct: float = 10.0
    ) -> None:
        """
        Initialize complete day filter.
        
        Args:
            semi_hourly_records: Expected records for semi-hourly data
            hourly_records: Expected records for hourly data
            max_missing_pct: Maximum missing % after imputation
        """
        self.semi_hourly_records = semi_hourly_records
        self.hourly_records = hourly_records
        self.max_missing_pct = max_missing_pct
    
    def filter(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        area_col: str = "cod_area",
        value_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, FilteringReport]:
        """
        Filter to complete days.
        
        Args:
            df: DataFrame to filter
            timestamp_col: Timestamp column name
            area_col: Area code column name
            value_cols: Columns to check for missing values
        
        Returns:
            (filtered_df, report)
        """
        df = df.copy()
        report = FilteringReport()
        
        # Extract date
        df["_date"] = pd.to_datetime(df[timestamp_col]).dt.date
        
        # Detect frequency
        intervals = df[timestamp_col].diff().dropna()
        avg_interval = intervals.mode()[0] if len(intervals) > 0 else pd.Timedelta("30T")
        
        if avg_interval == pd.Timedelta("30T"):
            expected_records = self.semi_hourly_records
            freq_name = "semi-hourly"
        else:
            expected_records = self.hourly_records
            freq_name = "hourly"
        
        logger.info(f"Detected {freq_name} data (expected {expected_records} records/day)")
        
        # Group by area and date
        grouped = df.groupby([area_col, "_date"])
        
        # Track statistics
        report.total_days = len(grouped)
        report.affected_areas = df[area_col].unique().tolist()
        
        valid_groups = []
        
        for (area, dt), group in grouped:
            record_count = len(group)
            
            # Check record count
            if record_count != expected_records:
                report.incomplete_days += 1
                report.reasons["incomplete_records"] = report.reasons.get("incomplete_records", 0) + 1
                logger.debug(f"Filtered {area} {dt}: {record_count} records (expected {expected_records})")
                continue
            
            # Check missing values
            if value_cols:
                missing_pct = (group[value_cols].isna().sum().sum() / 
                              (len(group) * len(value_cols)) * 100)
                
                if missing_pct > self.max_missing_pct:
                    report.incomplete_days += 1
                    report.reasons["excessive_missing"] = report.reasons.get("excessive_missing", 0) + 1
                    logger.debug(f"Filtered {area} {dt}: {missing_pct:.1f}% missing")
                    continue
            
            # Day is valid
            valid_groups.append(group)
            report.complete_days += 1
        
        # Combine valid groups
        if valid_groups:
            df_filtered = pd.concat(valid_groups, ignore_index=True)
            df_filtered = df_filtered.drop(columns=["_date"])
        else:
            logger.warning("No complete days found after filtering!")
            df_filtered = df.drop(columns=["_date"]).iloc[:0]  # Empty with columns
        
        report.log_summary()
        return df_filtered, report


class DuplicateHandler:
    """Handle duplicate timestamps."""
    
    @staticmethod
    def detect_and_remove(
        df: pd.DataFrame,
        key_columns: List[str] = ["timestamp", "cod_area"]
    ) -> Tuple[pd.DataFrame, int]:
        """
        Detect and remove duplicate records.
        
        Args:
            df: DataFrame
            key_columns: Columns defining uniqueness
        
        Returns:
            (deduplicated_df, num_duplicates_removed)
        """
        original_len = len(df)
        df_dedup = df.drop_duplicates(subset=key_columns, keep="first")
        num_removed = original_len - len(df_dedup)
        
        if num_removed > 0:
            logger.warning(f"Removed {num_removed} duplicate records on {key_columns}")
        
        return df_dedup, num_removed


class ChronologicalValidator:
    """Validate chronological ordering."""
    
    @staticmethod
    def validate_and_sort(
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        area_col: str = "cod_area",
        auto_sort: bool = True
    ) -> pd.DataFrame:
        """
        Validate and optionally sort chronologically.
        
        Args:
            df: DataFrame
            timestamp_col: Timestamp column
            area_col: Area column
            auto_sort: If True, auto-sort. If False, raise on disorder.
        
        Returns:
            Sorted DataFrame
        """
        # Check if already sorted
        is_sorted = df.groupby(area_col)[timestamp_col].apply(
            lambda x: x.is_monotonic_increasing
        ).all()
        
        if not is_sorted:
            if auto_sort:
                logger.warning("Data not chronologically ordered, sorting...")
                df = df.sort_values([area_col, timestamp_col]).reset_index(drop=True)
            else:
                raise ValueError("Data not in chronological order")
        
        return df
```

---

## 🧪 Testing & Validation

```python
"""Tests for complete day filtering."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.preprocessors import CompleteDayFilter, DuplicateHandler, ChronologicalValidator


def test_complete_day_filter_semi_hourly():
    """Test filtering with complete semi-hourly days."""
    # Day 1: Complete (48 records)
    day1 = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=48, freq="30T"),
        "cod_area": "RJ",
        "value": range(48)
    })
    
    # Day 2: Incomplete (40 records)
    day2 = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-02", periods=40, freq="30T"),
        "cod_area": "RJ",
        "value": range(40)
    })
    
    df = pd.concat([day1, day2], ignore_index=True)
    
    filter = CompleteDayFilter()
    result, report = filter.filter(df, value_cols=["value"])
    
    # Only day 1 should remain
    assert len(result) == 48
    assert report.complete_days == 1
    assert report.incomplete_days == 1


def test_duplicate_detection():
    """Test duplicate detection and removal."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="30T").tolist() + 
                     [pd.Timestamp("2024-01-01 01:00:00")],  # Duplicate
        "cod_area": ["RJ"] * 6,
        "value": [1, 2, 3, 4, 5, 999]  # Last is duplicate
    })
    
    result, num_removed = DuplicateHandler.detect_and_remove(df)
    
    assert len(result) == 5
    assert num_removed == 1
    assert result["value"].tolist() == [1, 2, 3, 4, 5]


def test_chronological_validation():
    """Test chronological ordering."""
    df = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 02:00"),  # Out of order
            pd.Timestamp("2024-01-01 01:00"),
        ],
        "cod_area": ["RJ", "RJ", "RJ"],
        "value": [1, 2, 3]
    })
    
    result = ChronologicalValidator.validate_and_sort(df, auto_sort=True)
    
    assert result["timestamp"].iloc[1] == pd.Timestamp("2024-01-01 01:00")
    assert result["timestamp"].iloc[2] == pd.Timestamp("2024-01-01 02:00")
```

---

## 🔗 Dependencies

**Depends On:**
- PC-006-01: Load Raw Load Data from Storage
- PC-008-01: Preprocess and Impute Missing Data

**Blocks:**
- Epic-02: Feature Engineering (requires clean, complete days)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] CompleteDayFilter implemented
- [ ] Duplicate handling implemented
- [ ] Chronological validation implemented
- [ ] FilteringReport with statistics
- [ ] Unit tests pass with >80% coverage
- [ ] Performance test passes (<1s for 1 year)
- [ ] Edge cases tested (DST, leap years)
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next Ticket:** [PC-011-01: Load Temperature and Holiday Data](PC-011-01-load-auxiliary-data.md)
