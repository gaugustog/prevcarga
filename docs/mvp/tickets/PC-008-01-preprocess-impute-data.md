# PC-008-01: Preprocess and Impute Missing Data

**Ticket ID:** PC-008-01  
**Epic:** [Epic-01: Data Infrastructure Layer](../epics/Epic-01.md)  
**User Story:** US-1.3  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement sophisticated data preprocessing pipeline with triple-pass imputation strategy (proven from R implementation), timezone handling for Brazil (America/Sao_Paulo), and bidirectional resampling (hourly ↔ semi-hourly) to ensure models receive complete, uniform datasets.

**As a** data engineer  
**I want to** handle missing values and resample data to required frequency  
**So that** models receive complete, uniform datasets

---

## ✅ Acceptance Criteria

- [ ] Support multiple imputation strategies:
  - Forward-fill (within-day and cross-day)
  - Backward-fill (within-day)
  - Lag fill (24h/48h same-hour previous day)
  - Interpolation (linear, average of forward/backward)
  - Mean/median (rolling window)
- [ ] **Triple-pass imputation** implemented per proven R approach:
  - Pass 1: Forward fill within same day (don't cross day boundaries)
  - Pass 2: Backward fill within same day
  - Pass 3: Interpolate using average of forward/backward
  - Pass 4: Final forward fill across days for remaining NAs
- [ ] Strategy configurable per column (load vs temperature vs heat index)
- [ ] Resample hourly data to semi-hourly (30min intervals) using:
  - Create complete time sequence
  - Lag fill with same-hour previous day (24h/48h)
  - Linear interpolation between filled hourly points
- [ ] Resample semi-hourly data to hourly using aggregation (mean)
- [ ] Preserve timezone information throughout pipeline (`America/Sao_Paulo`)
- [ ] Timestamp adjustment: Backward by 30 minutes for load data alignment
- [ ] Log all imputation actions (how many values, which method, which pass)
- [ ] Performance: <3s to preprocess 1 year hourly data

---

## 🔧 Implementation Tasks

### 1. Create BaseImputer Abstract Class
- [ ] Create `src/data/preprocessors.py`
- [ ] Define `BaseImputer` ABC with `impute()` method
- [ ] Add configuration parameters base class
- [ ] Add logging support
- [ ] Add type hints for all methods

### 2. Implement Basic Imputers
- [ ] `ForwardFillImputer`:
  - Support limit parameter
  - Support groupby for within-day filling
- [ ] `BackwardFillImputer`:
  - Support limit parameter
  - Support groupby for within-day filling
- [ ] `InterpolationImputer`:
  - Support linear, spline, polynomial methods
  - Handle edge cases (start/end NAs)

### 3. Implement Advanced Imputers
- [ ] `LagFillImputer`:
  - Implement 24h lag (same hour previous day)
  - Implement 48h lag (same hour 2 days before)
  - Group by hour for same-hour matching
- [ ] `MeanImputer`:
  - Rolling window mean
  - Configurable window size
- [ ] `MedianImputer`:
  - Rolling window median
  - Robust to outliers

### 4. Implement TriplePassImputer (Critical!)
- [ ] Pass 1: Forward fill within day (groupby date)
- [ ] Pass 2: Backward fill within day (groupby date)
- [ ] Pass 3: Average interpolation (forward + backward) / 2
- [ ] Pass 4: Final cross-day forward fill
- [ ] Add detailed logging for each pass
- [ ] Track imputation statistics per pass
- [ ] Match R implementation exactly

### 5. Create ImputerChain
- [ ] Support chaining multiple imputation strategies
- [ ] Execute imputers in sequence
- [ ] Track cumulative imputation statistics
- [ ] Log progress after each imputer
- [ ] Return imputation report

### 6. Implement ResamplerMixin
- [ ] `resample_to_30min()`:
  - Create complete time range (30-min intervals)
  - Lag fill missing hours (24h/48h same-hour)
  - Linear interpolation between filled points
  - Preserve timezone
- [ ] `resample_to_hourly()`:
  - Aggregate 30-min to hourly using mean
  - Handle partial hours
  - Preserve timezone

### 7. Implement TimezoneHandler
- [ ] Convert timestamps to `America/Sao_Paulo`
- [ ] Handle DST transitions (spring forward, fall back)
- [ ] Implement 30-minute backward adjustment for load data
- [ ] Validate timezone-aware timestamps
- [ ] Add DST transition detection

### 8. Create Preprocessing Pipeline
- [ ] Combine imputation, resampling, timezone handling
- [ ] Configurable via YAML
- [ ] Support different strategies per column
- [ ] Generate preprocessing report
- [ ] Log all transformations

### 9. Write Comprehensive Tests
- [ ] Test each imputer individually
- [ ] Test triple-pass imputation end-to-end
- [ ] Test resampling hourly→30min
- [ ] Test resampling 30min→hourly
- [ ] Test timezone handling and DST transitions
- [ ] Test lag fill with different lags
- [ ] Test ImputerChain with multiple strategies
- [ ] Test edge cases (all NAs, no NAs, partial days)
- [ ] Performance test with 1 year data

---

## 💻 Implementation Details

### Triple-Pass Imputation (Critical Component)

```python
"""Triple-pass imputation strategy from proven R implementation."""
import pandas as pd
import numpy as np
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TriplePassImputer:
    """
    Implement sophisticated triple-pass imputation from R production code.
    
    This strategy has been proven effective over 2+ years in production:
    - Pass 1: Forward fill within same day (preserves intraday patterns)
    - Pass 2: Backward fill within same day (fills gaps forward couldn't reach)
    - Pass 3: Average interpolation (smoother than simple fill)
    - Pass 4: Final cross-day forward fill (pragmatic for longer gaps)
    """
    
    def __init__(self, date_col: str = "Data") -> None:
        """
        Initialize triple-pass imputer.
        
        Args:
            date_col: Column name for date grouping (default: "Data")
        """
        self.date_col = date_col
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        value_col: str,
        add_date_if_missing: bool = True
    ) -> pd.DataFrame:
        """
        Apply triple-pass imputation to value column.
        
        Args:
            df: DataFrame with timestamp and value columns
            value_col: Column to impute
            add_date_if_missing: If True, extract date from timestamp
        
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        # Extract date if not present
        if self.date_col not in df.columns and add_date_if_missing:
            if "timestamp" in df.columns:
                df[self.date_col] = pd.to_datetime(df["timestamp"]).dt.date
            else:
                raise ValueError(f"Column '{self.date_col}' not found and cannot be derived")
        
        original_nas = df[value_col].isna().sum()
        logger.info(f"Triple-pass imputation starting: {original_nas} NAs in {value_col}")
        
        # Pass 1: Forward fill within same day
        df["_prev"] = df.groupby(self.date_col)[value_col].ffill()
        nas_after_pass1 = df[value_col].isna().sum()
        logger.debug(f"Pass 1 (forward fill within day): {original_nas - nas_after_pass1} values filled")
        
        # Pass 2: Backward fill within same day
        df["_next"] = df.groupby(self.date_col)[value_col].bfill()
        
        # Pass 3: Interpolate with average of forward and backward
        mask = (
            df[value_col].isna() &
            df["_prev"].notna() &
            df["_next"].notna()
        )
        df.loc[mask, value_col] = (df.loc[mask, "_prev"] + df.loc[mask, "_next"]) / 2
        nas_after_pass3 = df[value_col].isna().sum()
        logger.debug(f"Pass 3 (average interpolation): {nas_after_pass1 - nas_after_pass3} values filled")
        
        # Pass 4: Final forward fill across days
        df[value_col] = df[value_col].ffill()
        final_nas = df[value_col].isna().sum()
        logger.debug(f"Pass 4 (cross-day forward fill): {nas_after_pass3 - final_nas} values filled")
        
        # Cleanup temporary columns
        df.drop(columns=["_prev", "_next"], inplace=True)
        
        # Summary
        total_filled = original_nas - final_nas
        logger.info(
            f"Triple-pass imputation complete: {total_filled}/{original_nas} NAs filled "
            f"({final_nas} remaining)"
        )
        
        return df
```

### Lag Fill Imputer (for Temperature)

```python
"""Lag fill imputation for same-hour previous day."""


class LagFillImputer:
    """
    Impute using same-hour previous day(s).
    
    Useful for temperature data where diurnal patterns are strong.
    """
    
    def __init__(self, lag_hours: List[int] = [24, 48]) -> None:
        """
        Initialize lag fill imputer.
        
        Args:
            lag_hours: List of lag hours to try (default: 24h, then 48h)
        """
        self.lag_hours = lag_hours
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        value_col: str,
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Apply lag fill imputation.
        
        Args:
            df: DataFrame with timestamp and value columns
            value_col: Column to impute
            timestamp_col: Timestamp column name
        
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        original_nas = df[value_col].isna().sum()
        logger.info(f"Lag fill imputation starting: {original_nas} NAs")
        
        # Try each lag in order
        for lag_h in self.lag_hours:
            if df[value_col].isna().sum() == 0:
                break  # All filled
            
            # Shift by lag hours
            df[f"_lag_{lag_h}"] = df[value_col].shift(lag_h)
            
            # Fill NAs with lagged values
            mask = df[value_col].isna() & df[f"_lag_{lag_h}"].notna()
            filled_count = mask.sum()
            df.loc[mask, value_col] = df.loc[mask, f"_lag_{lag_h}"]
            
            logger.debug(f"Lag {lag_h}h filled {filled_count} values")
            df.drop(columns=[f"_lag_{lag_h}"], inplace=True)
        
        final_nas = df[value_col].isna().sum()
        logger.info(f"Lag fill complete: {original_nas - final_nas} values filled")
        
        return df
```

### Resampler with Timezone Support

```python
"""Resampling utilities with timezone support."""
import pytz


class ResamplerMixin:
    """Mixin for data resampling operations."""
    
    BRAZIL_TZ = pytz.timezone("America/Sao_Paulo")
    
    @staticmethod
    def resample_to_30min(
        df: pd.DataFrame,
        value_col: str = "carga_mwh",
        timestamp_col: str = "timestamp",
        use_lag_fill: bool = True
    ) -> pd.DataFrame:
        """
        Resample hourly to 30-minute using lag fill + interpolation.
        
        Args:
            df: DataFrame with hourly data
            value_col: Column to resample
            timestamp_col: Timestamp column
            use_lag_fill: If True, use 24h lag fill before interpolation
        
        Returns:
            DataFrame with 30-minute frequency
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Ensure timezone-aware
        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(ResamplerMixin.BRAZIL_TZ)
        
        # Create complete time range
        start = df[timestamp_col].min()
        end = df[timestamp_col].max()
        complete_range = pd.date_range(start=start, end=end, freq="1H", tz=ResamplerMixin.BRAZIL_TZ)
        
        # Reindex to complete hourly sequence
        df_indexed = df.set_index(timestamp_col).reindex(complete_range)
        
        # Lag fill missing hours (24h same-hour previous day)
        if use_lag_fill:
            df_indexed[value_col] = df_indexed.groupby(df_indexed.index.hour)[value_col].transform(
                lambda x: x.fillna(x.shift(24))
            )
        
        # Resample to 30-min with linear interpolation
        df_30min = df_indexed.resample("30T").interpolate(method="linear")
        
        return df_30min.reset_index().rename(columns={"index": timestamp_col})
    
    @staticmethod
    def resample_to_hourly(
        df: pd.DataFrame,
        value_col: str = "carga_mwh",
        timestamp_col: str = "timestamp",
        agg_method: str = "mean"
    ) -> pd.DataFrame:
        """
        Resample 30-minute to hourly using aggregation.
        
        Args:
            df: DataFrame with 30-minute data
            value_col: Column to resample
            timestamp_col: Timestamp column
            agg_method: Aggregation method ("mean", "sum", "max")
        
        Returns:
            DataFrame with hourly frequency
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Ensure timezone-aware
        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(ResamplerMixin.BRAZIL_TZ)
        
        # Set index and resample
        df_indexed = df.set_index(timestamp_col)
        
        if agg_method == "mean":
            df_hourly = df_indexed.resample("1H").mean()
        elif agg_method == "sum":
            df_hourly = df_indexed.resample("1H").sum()
        elif agg_method == "max":
            df_hourly = df_indexed.resample("1H").max()
        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")
        
        return df_hourly.reset_index()


class TimezoneHandler:
    """Handle timezone conversions and adjustments."""
    
    BRAZIL_TZ = pytz.timezone("America/Sao_Paulo")
    
    @staticmethod
    def convert_to_brazil_tz(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Convert timestamps to Brazil timezone."""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
        
        df[timestamp_col] = df[timestamp_col].dt.tz_convert(TimezoneHandler.BRAZIL_TZ)
        return df
    
    @staticmethod
    def adjust_timestamp_backward(
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        minutes: int = 30
    ) -> pd.DataFrame:
        """
        Adjust timestamps backward (for load data alignment).
        
        Args:
            df: DataFrame
            timestamp_col: Timestamp column
            minutes: Minutes to subtract
        
        Returns:
            DataFrame with adjusted timestamps
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col]) - pd.Timedelta(minutes=minutes)
        return df
```

---

## 🧪 Testing & Validation

```python
"""Tests for preprocessing and imputation."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.preprocessors import TriplePassImputer, LagFillImputer, ResamplerMixin


def test_triple_pass_imputation():
    """Test triple-pass imputation fills all NAs."""
    df = pd.DataFrame({
        "Data": pd.date_range("2024-01-01", periods=96, freq="30T").date,
        "timestamp": pd.date_range("2024-01-01", periods=96, freq="30T"),
        "value": [1.0] * 20 + [np.nan] * 10 + [2.0] * 20 + [np.nan] * 10 + [3.0] * 36
    })
    
    imputer = TriplePassImputer(date_col="Data")
    result = imputer.fit_transform(df, "value")
    
    # All NAs should be filled
    assert result["value"].isna().sum() == 0
    
    # Check that within-day gaps were filled with interpolation
    assert result["value"].iloc[20] > 1.0
    assert result["value"].iloc[20] < 2.0


def test_lag_fill_imputation():
    """Test lag fill uses previous day same hour."""
    # Create 3 days of hourly data with pattern
    timestamps = pd.date_range("2024-01-01", periods=72, freq="1H")
    values = [float(i % 24) for i in range(72)]  # Repeating daily pattern
    
    # Add NAs on day 2
    values[24:28] = [np.nan] * 4
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temp": values
    })
    
    imputer = LagFillImputer(lag_hours=[24])
    result = imputer.fit_transform(df, "temp")
    
    # NAs should be filled with day 1 values
    assert result["temp"].iloc[24] == 0.0
    assert result["temp"].iloc[27] == 3.0


def test_resample_hourly_to_30min():
    """Test resampling from hourly to 30-minute."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=24, freq="1H", tz="America/Sao_Paulo"),
        "temp": range(24)
    })
    
    result = ResamplerMixin.resample_to_30min(df, "temp", use_lag_fill=False)
    
    # Should have 47 records (24 hours = 47 30-min intervals)
    assert len(result) == 47
    
    # Check interpolation
    assert result["temp"].iloc[1] == 0.5  # Between 0 and 1


def test_timezone_conversion():
    """Test timezone conversion to Brazil."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1H", tz="UTC"),
        "value": range(10)
    })
    
    result = TimezoneHandler.convert_to_brazil_tz(df)
    
    assert result["timestamp"].dt.tz.zone == "America/Sao_Paulo"
```

---

## 📝 Technical Notes

- Triple-pass imputation is critical - matches proven R implementation
- Lag fill preserves diurnal patterns better than interpolation
- DST transitions require careful handling (spring: skip hour, fall: repeat hour)
- 30-minute backward adjustment for load data is non-obvious but required
- Use `pytz` for robust timezone handling
- Interpolation between lag-filled points provides smooth curves

---

## 🔗 Dependencies

**Depends On:**
- PC-006-01: Load Raw Load Data from S3
- PC-007-01: Validate Data Schemas

**Blocks:**
- PC-010-01: Filter Complete Days
- Epic-02: Feature Engineering

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Triple-pass imputation implemented exactly per R code
- [ ] All imputer classes implemented
- [ ] Resampling (hourly ↔ 30min) working
- [ ] Timezone handling with DST support
- [ ] Unit tests pass with >80% coverage
- [ ] Performance test passes (<3s for 1 year)
- [ ] Integration test with real missing data patterns
- [ ] Documentation complete with strategy guide
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next Ticket:** [PC-009-01: Manage Data Catalog](PC-009-01-manage-data-catalog.md)
