# PC-011-01: Load Temperature and Holiday Data

**Ticket ID:** PC-011-01  
**Epic:** [Epic-01: Data Infrastructure Layer](../epics/Epic-01.md)  
**User Story:** US-1.6  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Extend DataLoader to support loading auxiliary data (temperature forecasts from 4 sources, heat index, and holidays) with multi-format support (Parquet + CSV), unified storage backend (S3 or local), and join utilities for combining all data sources.

**As a** data pipeline  
**I want to** load auxiliary data (temperature forecasts, heat index, holidays)  
**So that** feature engineering can access all required inputs

---

## ✅ Acceptance Criteria

- [ ] Load temperature forecast D+1 data from: `temperatura_d1_{cod_area}_{YYYYMMDD}.parquet`
- [ ] Load temperature forecast D+2 data from: `temperatura_d2_{cod_area}_{YYYYMMDD}.parquet`
- [ ] Load ECMWF temperature from: `heatindex_{cod_area}.csv` (Temperatura_ECMWF field)
- [ ] Load heat index from: `heatindex_{cod_area}.csv` (Heatindex field)
- [ ] Load holiday calendar from: `feriados_{year}.parquet`
- [ ] Temperature data returns DataFrame: `[timestamp, cod_area, temp_celsius, source]`
- [ ] Heat index returns DataFrame: `[timestamp, cod_area, heat_index]`
- [ ] Holiday data returns DataFrame: `[date, holiday_name, holiday_type, id_tipodiaespecial, affected_areas]`
- [ ] Support joining load, temperature, heat index, and holiday data on timestamp/area
- [ ] Handle missing temperature forecasts with lag fill (24/48h) + interpolation
- [ ] Support both CSV and Parquet formats for heat index data
- [ ] Performance: <3s to load all auxiliary data for 1 year

---

## 🔧 Implementation Tasks

### 1. Extend DataLoader for Temperature D+1/D+2
- [ ] Add `load_temperatura_d1()` method to DataLoader:
  - Generate paths: `temperatura_d1_{area}_{YYYYMMDD}.parquet`
  - Load in parallel using storage backend
  - Add source column: "D+1"
- [ ] Add `load_temperatura_d2()` method:
  - Generate paths: `temperatura_d2_{area}_{YYYYMMDD}.parquet`
  - Load in parallel using storage backend
  - Add source column: "D+2"
- [ ] Standardize columns: `[timestamp, cod_area, temp_celsius, source]`
- [ ] Works with both S3 and local backends

### 2. Add MultiFormatLoader for CSV Support
- [ ] Create `MultiFormatLoader` class using StorageBackend
- [ ] Support loading CSV from any backend (S3 or local)
- [ ] Support loading Parquet from any backend
- [ ] Auto-detect format from file extension
- [ ] Parse CSV with pandas
- [ ] Add encoding support (UTF-8, Latin-1)
- [ ] Use storage_backend.get_object() for file retrieval

### 3. Implement Heat Index and ECMWF Temperature Loading
- [ ] Add `load_heat_index()` method:
  - Path: `heatindex_{area}.csv`
  - Extract Heatindex column
  - Return: `[timestamp, cod_area, heat_index]`
- [ ] Add `load_temperatura_ecmwf()` method:
  - Path: `heatindex_{area}.csv` (same file)
  - Extract Temperatura_ECMWF column
  - Return: `[timestamp, cod_area, temp_celsius, source="ECMWF"]`
- [ ] Support both CSV and Parquet formats

### 4. Implement Holiday Calendar Loading
- [ ] Add `load_feriados()` method:
  - Path: `feriados_{year}.parquet`
  - Load multiple years if needed
  - Return: `[date, holiday_name, holiday_type, id_tipodiaespecial]`
- [ ] Add `_expand_holidays_to_areas()` helper:
  - Map holidays to affected areas
  - Handle national (all areas) vs regional
- [ ] Cache holidays per year

### 5. Create DataJoiner Utility
- [ ] Implement `join_load_and_temperature()`:
  - Left join on (timestamp, cod_area)
  - Handle multiple temperature sources
  - Fill missing with lag fill (24/48h)
- [ ] Implement `join_with_holidays()`:
  - Join on date (extract from timestamp)
  - Create holiday flag columns
  - Add days_to_holiday feature
- [ ] Implement `join_all_sources()`:
  - Combine load, all temps, heat index, holidays
  - Handle missing data gracefully
  - Return unified DataFrame

### 6. Add Missing Data Fallbacks
- [ ] Implement lag fill for missing temperature files:
  - Use 24h same-hour previous day
  - Use 48h if 24h unavailable
- [ ] Add interpolation fallback
- [ ] Log all fallback usage
- [ ] Track data quality metrics

### 7. Create Temperature Wide-to-Long Transformer
- [ ] For Random Forest weighted temperature
- [ ] Input: `[dataOrigem, temp_max_1-9, temp_min_1-9]`
- [ ] Output: `[timestamp, lead_hour, tmax, tmin]`
- [ ] Expand 9 days to 216 hourly records
- [ ] Calculate lead_hour (1-216)

### 8. Write Comprehensive Tests
- [ ] Test each temperature source loading
- [ ] Test heat index loading (CSV + Parquet)
- [ ] Test holiday loading
- [ ] Test multi-format support
- [ ] Test DataJoiner with all sources
- [ ] Test missing data fallbacks
- [ ] Test wide-to-long transformation
- [ ] Test performance with 1 year multi-source load

---

## 💻 Implementation Details

### Extended DataLoader

```python
"""Extended loader for auxiliary data sources."""


class DataLoader:
    # ... existing code from PC-006-01 ...
    
    def load_temperatura_d1(
        self,
        areas: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Load temperature forecast D+1 data.
        
        Returns:
            DataFrame: [timestamp, cod_area, temp_celsius, source]
        """
        paths = self._generate_paths("temperatura_d1", areas, start_date, end_date)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            dfs = list(executor.map(self._load_single_parquet, paths))
        
        dfs = [df for df in dfs if df is not None]
        if not dfs:
            return pd.DataFrame(columns=["timestamp", "cod_area", "temp_celsius", "source"])
        
        df = pd.concat(dfs, ignore_index=True)
        df["source"] = "D+1"
        
        # Standardize column names
        df = df.rename(columns={"ValItemserieoriginal": "temp_celsius", "DataHora": "timestamp"})
        
        return df[["timestamp", "cod_area", "temp_celsius", "source"]]
    
    def load_temperatura_d2(
        self,
        areas: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Load temperature forecast D+2 data."""
        # Similar to D+1 but with different prefix
        paths = self._generate_paths("temperatura_d2", areas, start_date, end_date)
        # ... rest similar to load_temperatura_d1
        pass
    
    def load_heat_index(
        self,
        areas: List[str],
        format: str = "csv"
    ) -> pd.DataFrame:
        """
        Load heat index data.
        
        Args:
            areas: Area codes
            format: "csv" or "parquet"
        
        Returns:
            DataFrame: [timestamp, cod_area, heat_index]
        """
        loader = MultiFormatLoader(self.bucket, self.region)
        dfs = []
        
        for area in areas:
            path = f"raw_data/heatindex_{area}.{format}"
            df = loader.load(path, format=format)
            
            if df is not None and "Heatindex" in df.columns:
                df = df.rename(columns={"Heatindex": "heat_index", "timestamp": "timestamp"})
                df["cod_area"] = area
                dfs.append(df[["timestamp", "cod_area", "heat_index"]])
        
        if not dfs:
            return pd.DataFrame(columns=["timestamp", "cod_area", "heat_index"])
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_temperatura_ecmwf(
        self,
        areas: List[str],
        format: str = "csv"
    ) -> pd.DataFrame:
        """
        Load ECMWF temperature from heat index files.
        
        Returns:
            DataFrame: [timestamp, cod_area, temp_celsius, source]
        """
        loader = MultiFormatLoader(self.bucket, self.region)
        dfs = []
        
        for area in areas:
            path = f"raw_data/heatindex_{area}.{format}"
            df = loader.load(path, format=format)
            
            if df is not None and "Temperatura_ECMWF" in df.columns:
                df = df.rename(columns={"Temperatura_ECMWF": "temp_celsius"})
                df["cod_area"] = area
                df["source"] = "ECMWF"
                dfs.append(df[["timestamp", "cod_area", "temp_celsius", "source"]])
        
        if not dfs:
            return pd.DataFrame(columns=["timestamp", "cod_area", "temp_celsius", "source"])
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_feriados(self, years: List[int]) -> pd.DataFrame:
        """
        Load holiday calendar for specified years.
        
        Args:
            years: List of years
        
        Returns:
            DataFrame: [date, holiday_name, holiday_type, id_tipodiaespecial]
        """
        dfs = []
        
        for year in years:
            path = f"raw_data/feriados_{year}.parquet"
            df = self._load_single_parquet(path)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame(columns=["date", "holiday_name", "holiday_type", "id_tipodiaespecial"])
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Standardize columns
        df = df.rename(columns={"dat_diaespecial": "date"})
        
        return df


class MultiFormatLoader:
    """Load data from multiple formats (CSV, Parquet) using unified storage backend."""
    
    def __init__(self, storage_backend: StorageBackend) -> None:
        self.storage = storage_backend
    
    def load(self, storage_path: str, format: str = None) -> Optional[pd.DataFrame]:
        """
        Load file from any storage backend in specified format.
        
        Args:
            storage_path: Path to file (works with both S3 and local)
            format: "csv" or "parquet" (auto-detect if None)
        
        Returns:
            DataFrame or None if not found
        """
        if format is None:
            format = "parquet" if storage_path.endswith(".parquet") else "csv"
        
        try:
            if format == "csv":
                # For CSV, read directly from storage backend
                df = pd.read_csv(io.BytesIO(self.storage.get_object(storage_path)), encoding="utf-8")
            elif format == "parquet":
                df = self.storage.get_parquet(storage_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.debug(f"Loaded {len(df)} records from {storage_path} ({format})")
            return df
            
        except FileNotFoundError:
            logger.warning(f"File not found: {storage_path}")
            return None


class DataJoiner:
    """Join multiple data sources."""
    
    @staticmethod
    def join_load_and_temperature(
        df_load: pd.DataFrame,
        df_temp: pd.DataFrame,
        how: str = "left"
    ) -> pd.DataFrame:
        """
        Join load and temperature data.
        
        Args:
            df_load: Load DataFrame
            df_temp: Temperature DataFrame (may have multiple sources)
            how: Join type
        
        Returns:
            Joined DataFrame
        """
        return df_load.merge(
            df_temp,
            on=["timestamp", "cod_area"],
            how=how,
            suffixes=("", "_temp")
        )
    
    @staticmethod
    def join_with_holidays(
        df: pd.DataFrame,
        df_holidays: pd.DataFrame,
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Join with holiday calendar.
        
        Args:
            df: Main DataFrame
            df_holidays: Holiday DataFrame
            timestamp_col: Timestamp column name
        
        Returns:
            DataFrame with holiday columns
        """
        df = df.copy()
        df["_date"] = pd.to_datetime(df[timestamp_col]).dt.date
        
        df_holidays = df_holidays.copy()
        df_holidays["_date"] = pd.to_datetime(df_holidays["date"]).dt.date
        
        # Left join on date
        result = df.merge(
            df_holidays[["_date", "holiday_name", "holiday_type", "id_tipodiaespecial"]],
            on="_date",
            how="left"
        )
        
        # Fill NAs (no holiday)
        result["id_tipodiaespecial"] = result["id_tipodiaespecial"].fillna(0).astype(int)
        result["holiday_name"] = result["holiday_name"].fillna("")
        result["holiday_type"] = result["holiday_type"].fillna("")
        
        result = result.drop(columns=["_date"])
        return result
```

---

## 🧪 Testing & Validation

```python
"""Tests for auxiliary data loading."""
import pytest
from datetime import date


@mock_s3
def test_load_temperatura_d1():
    """Test temperature D+1 loading."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    
    # Upload test data
    df = pd.DataFrame({
        "DataHora": pd.date_range("2024-01-01", periods=24, freq="1H"),
        "cod_area": "RJ",
        "ValItemserieoriginal": range(20, 44)
    })
    s3.put_object(
        Bucket="test-bucket",
        Key="raw_data/2024/01/temperatura_d1_RJ_20240101.parquet",
        Body=df.to_parquet()
    )
    
    loader = DataLoader(storage_backend=backend)
    result = loader.load_temperatura_d1(["RJ"], date(2024, 1, 1), date(2024, 1, 1))
    
    assert len(result) == 24
    assert "source" in result.columns
    assert result["source"].iloc[0] == "D+1"
    assert "temp_celsius" in result.columns


@mock_s3
def test_load_heat_index_csv():
    """Test heat index loading from CSV."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    
    # Upload CSV
    csv_data = "timestamp,cod_area,Heatindex\n2024-01-01 00:00,RJ,25.5\n"
    s3.put_object(
        Bucket="test-bucket",
        Key="raw_data/heatindex_RJ.csv",
        Body=csv_data.encode()
    )
    
    backend = StorageFactory.create("s3", bucket="test-bucket")
    loader = DataLoader(storage_backend=backend)
    result = loader.load_heat_index(["RJ"], format="csv")
    
    assert len(result) > 0
    assert "heat_index" in result.columns


def test_data_joiner():
    """Test joining load and temperature."""
    df_load = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1H"),
        "cod_area": "RJ",
        "carga_mwh": [1000, 1100, 1200, 1300, 1400]
    })
    
    df_temp = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1H"),
        "cod_area": "RJ",
        "temp_celsius": [25, 26, 27, 28, 29],
        "source": "D+1"
    })
    
    result = DataJoiner.join_load_and_temperature(df_load, df_temp)
    
    assert len(result) == 5
    assert "carga_mwh" in result.columns
    assert "temp_celsius" in result.columns
```

---

## 🔗 Dependencies

**Depends On:**
- PC-006-01: Load Raw Load Data from Storage
- PC-008-01: Preprocess and Impute Missing Data

**Blocks:**
- Epic-02: Feature Engineering (needs all data sources)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] All 6 data sources loadable (D+1, D+2, ECMWF, heat index, weighted, holidays)
- [ ] Multi-format support (CSV + Parquet) working with both backends
- [ ] DataJoiner utilities implemented
- [ ] Missing data fallbacks working
- [ ] Unit tests pass with >80% coverage for both S3 and local backends
- [ ] Performance test passes (<3s for 1 year all sources) on both backends
- [ ] Integration test with all sources (both backends)
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Epic Complete:** This is the final ticket for Epic-01. Upon completion, all data infrastructure will be ready for [Epic-02: Core Feature Engineering](../epics/Epic-02A.md).
