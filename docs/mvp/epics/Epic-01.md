# 📦 Epic-01: Data Infrastructure Layer

**Epic ID:** Epic-01  
**Epic Name:** Data Infrastructure Layer  
**Phase:** Phase 1  
**Duration:** 2 weeks (Weeks 2-3)  
**Priority:** Critical  
**Dependencies:** Epic-00 (Project Foundation)

---

## 🎯 Epic Goal

Build a robust, scalable data infrastructure layer capable of loading, validating, preprocessing, and cataloging electric load forecasting data from multiple storage backends (AWS S3 and local filesystem). This layer serves as the foundation for all subsequent model training and prediction workflows, enabling seamless local development and cloud deployment.

---

## 📋 Epic Overview

### Scope

This epic encompasses the complete Data Layer implementation, including:
- **Multi-source unified loaders** supporting both S3 and local filesystem for:
  - Electric load data (semi-hourly granularity)
  - Temperature forecasts (D+1, D+2, ECMWF, weighted)
  - Heat index data (apparent temperature)
  - Holiday calendars (national and regional)
- **Pydantic-based schema validation** ensuring data quality
- **Advanced preprocessing pipeline** with:
  - Triple-pass missing value imputation (proven R implementation)
  - Timezone-aware datetime handling (America/Sao_Paulo)
  - Bidirectional resampling (hourly ↔ semi-hourly)
  - Complete day filtering (48 semi-hourly records)
- **Data catalog system** for dataset management and versioning
- **Support for 26 time series** (17 areas + 4 subsystems + 4 losses + 1 national)

### Key Implementation Insights from R Analysis

Based on comprehensive analysis of both LGBM and Random Forest implementations:

**Data Sources (5 primary streams):**
1. Load: Semi-hourly (30min), `val_cargaglobalcons`, `val_cargammgd`
2. Temperature D+1/D+2: Hourly → resampled to semi-hourly
3. ECMWF Temperature: Semi-hourly, long-range forecast
4. Heat Index: Semi-hourly, apparent temperature calculation
5. Holidays: Daily calendar with special period detection

**Critical Processing Requirements:**
- **Timezone:** All timestamps to `America/Sao_Paulo` with 30-min backward adjustment
- **Imputation:** Triple-pass strategy (forward fill within day → backward fill within day → interpolation → cross-day forward fill)
- **Resampling:** Temperature hourly→semi-hourly using lag fill (24/48h) + interpolation
- **Quality:** Filter to complete days only (48 records for semi-hourly)

### Business Value

- **Data Quality**: Ensures data integrity through automated validation and proven triple-pass imputation
- **Scalability**: Handles large volumes of time series data efficiently with parallel loading (7 data sources)
- **Reliability**: Robust imputation strategies (proven in R production) prevent training failures
- **Flexibility**: Supports multiple data sources, formats (Parquet + CSV), and model approaches (LGBM + Random Forest)
- **Maintainability**: Clean abstractions enable easy extension to new data sources
- **Production-Ready**: Based on 2+ years of proven R implementation patterns

### Technical Approach

- **Storage**: Unified backend abstraction supporting AWS S3 (production) and local filesystem (development) with Parquet format for efficient columnar storage (+ CSV fallback)
- **Validation**: Pydantic V2 schemas with type checking, business rule validation, and continuity checks
- **Processing**: Pandas/NumPy for data manipulation with optimized memory usage and chunked processing
- **Architecture**: Interface-based design with loader, validator, preprocessor, and transformer components
- **Testing**: Comprehensive unit tests with mocked S3 (moto), property-based tests (hypothesis), and integration tests covering edge cases (DST, leap years)
- **Proven Patterns**: Triple-pass imputation, lag fill resampling, complete day filtering from R implementation

### Key Differentiators from Standard Data Pipelines

1. **Triple-Pass Imputation**: Sophisticated within-day and cross-day strategy proven in production
2. **Multi-Source Temperature**: Handles 4 different temperature sources (D+1, D+2, ECMWF, weighted)
3. **Timezone Complexity**: Brazil-specific handling with 30-min backward adjustment for load data
4. **Complete Day Filtering**: Ensures training quality by rejecting incomplete days (48 records)
5. **Model-Specific Formats**: Supports both long (LGBM) and wide (Random Forest) temperature formats
6. **Lag Fill Resampling**: Preserves diurnal patterns when expanding hourly to semi-hourly

---

## 👥 User Stories

### **User Story 1.1: Load Raw Load Data from Storage**

**As a** data scientist  
**I want to** load historical electric load data from any storage backend (S3 or local)  
**So that** I can develop locally and deploy to production without code changes

**Acceptance Criteria:**
- [ ] Load Parquet files from unified paths following pattern: `raw_data/load/{area}/{YYYY}/{MM}/{YYYYMMDD}.parquet`
- [ ] Support ML-optimized consolidated files: `processed/load/area_code={area}/year={YYYY}/data.parquet`
- [ ] Auto-detect storage backend from configuration or path prefix (s3://, file://, or relative)
- [ ] Support S3 backend: `s3://bucket/raw_data/...`
- [ ] Support local backend: `./data/raw_data/...` or `/absolute/path/raw_data/...`
- [ ] Support filtering by area codes (e.g., "RJ", "SP", "SECO")
- [ ] Support filtering by date ranges (start_date, end_date)
- [ ] Handle missing files gracefully with informative error messages
- [ ] Return pandas DataFrame with standardized columns: `[timestamp, area_code, load_mwh]`
- [ ] Support batch loading of multiple areas/dates in parallel
- [ ] Load performance: <5s for 1 year of data (1 area)
- [ ] CLI supports `--storage-backend` flag to override config

**Tasks:**
1. Create `DataLoader` class using `StorageBackend` abstraction from Epic-00
2. Implement `load_load()` method with date/area filtering, backend-agnostic
3. Add connection pooling for S3 backend
4. Implement parallel loading using ThreadPoolExecutor (for both backends)
5. Add retry logic with exponential backoff (S3 backend)
6. Create path resolver to handle s3://, file://, and relative paths
7. Create unit tests with moto (S3 mocking) and local filesystem fixtures

**Definition of Done:**
- Code implemented and committed
- Unit tests passing with >80% coverage
- Can load 1 year of data for SECO subsystem in <30s
- Documentation with usage examples

---

### **User Story 1.2: Validate Data Schemas**

**As a** system  
**I want to** validate all incoming data against predefined schemas  
**So that** downstream components can assume data quality

**Acceptance Criteria:**
- [ ] Pydantic schemas defined for:
  - Load (`LoadSchema`): `timestamp`, `area_code`, `load_mwh`
  - Weather Observed (`WeatherObservedSchema`): `timestamp`, `area_code`, `temperature`, `heat_index`
  - Weather Forecast (`WeatherForecastSchema`): `issue_time`, `valid_time`, `area_code`, `temperature`
  - Holidays (`HolidaySchema`): `date`, `area_code`, `holiday_type_id`, `special_day_code`
  - Prediction Results (`PredictionResultSchema`): `run_time`, `step`, `target_time`, `predicted_load`
- [ ] Validation checks include:
  - Data types and required columns
  - Value ranges: load > 0, temperature -10°C to 50°C, heat index 15°C to 55°C
  - Timestamp continuity (30-min or 60-min intervals)
  - Complete day validation (48 semi-hourly or 24 hourly records per day)
- [ ] Business rules validated:
  - Load values non-negative
  - Temperatures within Brazil climate range
  - No duplicate (timestamp, area_code) pairs
  - Chronological ordering within area
  - Forecast issue_time before valid_time
- [ ] Invalid records logged with details but don't stop pipeline
- [ ] Validation summary report generated (counts of valid/invalid records)
- [ ] Performance: <2s validation overhead for 100k records

**Tasks:**
1. Define `LoadSchema` with fields: timestamp (datetime), area_code (str), load_mwh (float > 0)
2. Define `WeatherObservedSchema` with temperature range validation
3. Define `WeatherForecastSchema` with issue_time/valid_time provenance
4. Define `HolidaySchema` with holiday type enumeration
5. Define `PredictionResultSchema` for operational outputs
6. Implement `DataValidator` class with bulk validation
7. Add custom validators for timestamp continuity detection (30-min intervals)
8. Add complete day validator (48 records check)
9. Create validation report generator with statistics
10. Write comprehensive schema tests with valid/invalid samples

**Definition of Done:**
- All schemas implemented with Pydantic V2
- Validation detects common data quality issues (nulls, outliers, gaps)
- Unit tests with valid/invalid data samples
- Performance benchmark showing <1% overhead

---

### **User Story 1.3: Preprocess and Impute Missing Data**

**As a** data engineer  
**I want to** handle missing values and resample data to required frequency  
**So that** models receive complete, uniform datasets

**Acceptance Criteria:**
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

**Tasks:**
1. Create `ImputerChain` class supporting multiple strategies
2. Implement strategy pattern for imputation methods: 
   - `ForwardFillImputer` (with groupby support)
   - `BackwardFillImputer` (with groupby support)
   - `LagFillImputer` (24h/48h same-hour)
   - `InterpolationImputer` (linear, average)
   - `MeanImputer` (rolling window)
3. Create `TriplePassImputer` implementing proven R logic
4. Create `ResamplerMixin` for frequency conversion
5. Add `TimezoneHandler` for America/Sao_Paulo with timestamp adjustment
6. Add timezone-aware datetime handling
7. Implement validation of resampled data (no duplicates, proper intervals)
8. Add logging with detailed imputation statistics per pass
9. Write integration tests with real-world missing data patterns
10. Add tests for DST transitions and timestamp adjustment

**Definition of Done:**
- All imputation strategies implemented and tested
- Resampling works bidirectionally (hourly↔30min)
- Edge cases handled: DST transitions, leap years
- Performance tests passing
- Documentation with imputation strategy selection guide

---

### **User Story 1.4: Manage Data Catalog**

**As a** ML engineer  
**I want to** register and discover datasets with metadata  
**So that** I can track data versions and lineage

**Acceptance Criteria:**
- [ ] `DataCatalog` class for dataset registration and discovery
- [ ] Metadata stored includes: name, version, storage path (unified), schema, date range, row count, creation timestamp
- [ ] Support searching datasets by area, date range, version
- [ ] Persist catalog to configured storage backend (S3 or local) as JSON or Parquet
- [ ] Support dataset versioning with semantic versioning
- [ ] API methods: `register()`, `get()`, `list()`, `delete()`
- [ ] Thread-safe for concurrent registrations
- [ ] Storage backend configurable via DataCatalog initialization

**Tasks:**
1. Design catalog schema with Pydantic: `DatasetMetadata`
2. Implement `DataCatalog` class with in-memory cache and `StorageBackend` dependency
3. Add persistence layer using `StorageBackend` abstraction (JSON storage)
4. Implement search and filter methods
5. Add versioning support with `SemanticVersion` integration
6. Create thread-safe lock mechanism
7. Write tests for concurrent access patterns (both S3 and local backends)

**Definition of Done:**
- Catalog can store/retrieve 1000+ dataset entries
- Persistence working with S3 backend
- Search operations <100ms for typical queries
- Unit and integration tests passing
- API documentation generated

---

### **User Story 1.5: Filter Complete Days and Validate Continuity**

**As a** data quality engineer  
**I want to** ensure only complete days with proper temporal continuity are used  
**So that** models are trained on consistent, high-quality data

**Acceptance Criteria:**
- [ ] Filter datasets to include only complete days (48 semi-hourly or 24 hourly records)
- [ ] Validate timestamp intervals are consistent (30-min or 60-min)
- [ ] Detect and flag duplicate timestamps within same area
- [ ] Verify chronological ordering per area
- [ ] Remove days with excessive missing values (>10% after imputation)
- [ ] Log statistics on filtered days (count removed, reasons)
- [ ] Performance: <1s for filtering 1 year of data

**Tasks:**
1. Create `CompleteDayFilter` class with configurable thresholds
2. Implement timestamp continuity validator
3. Add duplicate detection with area-grouping
4. Create chronological order validator
5. Add summary statistics generator
6. Write tests with edge cases (DST transitions, leap years)
7. Document filtering rules and thresholds

**Definition of Done:**
- Complete day filter working correctly
- Handles edge cases (DST, partial days at boundaries)
- Statistics logged with filtering details
- Tests covering normal and edge cases
- Documentation with examples

---

### **User Story 1.6: Load Weather and Holiday Data**

**As a** data pipeline
**I want to** load auxiliary data (observed weather, forecast weather, holidays)
**So that** feature engineering can access all required inputs

**Acceptance Criteria:**
- [ ] Load observed weather from: `raw_data/weather/observed/{area}/{YYYY}/{MM}/{YYYYMMDD}.parquet`
- [ ] Support ML-optimized: `processed/weather/observed/area_code={area}/year={YYYY}/data.parquet`
- [ ] Load forecast weather from: `raw_data/weather/forecast/{area}/{YYYY}/{MM}/{YYYYMMDD}.parquet`
- [ ] Support ML-optimized: `processed/weather/forecast/area_code={area}/year={YYYY}/data.parquet`
- [ ] Load holiday calendar from: `raw_data/auxiliary/holidays/holidays_{year}.parquet`
- [ ] All weather data at semi-hourly resolution (interpolated via cubic spline during migration)
- [ ] Observed weather returns DataFrame: `[timestamp, area_code, temperature, heat_index]`
- [ ] Forecast weather returns DataFrame: `[issue_time, valid_time, area_code, temperature, heat_index]`
- [ ] Holiday data returns DataFrame: `[date, area_code, holiday_type_id, prevcarga_code, simplified_code]`
- [ ] Support `as_of` parameter for backtesting (only use forecasts available at that time)
- [ ] Performance: <3s to load all auxiliary data for 1 year (from consolidated files)

**Tasks:**
1. Implement `load_weather_observed()` method for training data
2. Implement `load_weather_forecast()` method with `as_of` support for backtesting
3. Add `load_holidays()` method with year filtering
4. Implement `DataJoiner` utility for multi-source joins
5. Add lag fill fallback mechanism for missing weather data (24/48h)
6. Create holiday type enumeration and special period detection
7. Write integration tests with all data types (S3 mocked, local filesystem)
8. Add performance benchmarks for multi-source loading (both backends)
9. Ensure path resolution works for both backends transparently

**Definition of Done:**
- All auxiliary data loading functions implemented (weather observed, weather forecast, holidays)
- `as_of` parameter enables proper backtesting without look-ahead bias
- Join operations preserve data integrity across all sources
- Missing data fallbacks working (lag fill for weather, defaults for holidays)
- Multi-format support (CSV + Parquet) working for legacy migration
- Tests covering edge cases (missing files, partial data, format mismatches)
- Documentation with data format specifications for each source
- Integration test loading all data types simultaneously

---

## 🏗️ Technical Architecture

### Unified Data Structure

The data infrastructure follows a unified hierarchy supporting both S3 and local filesystem backends.
**All time series data is standardized to semi-hourly (30-minute) resolution.**

```
data/
│
├── raw_data/                                    # Daily files for ingestion/updates
│   │
│   ├── load/                                   # Electric load data (semi-hourly only)
│   │   └── {area}/
│   │       └── {YYYY}/{MM}/
│   │           └── {YYYYMMDD}.parquet          # 48 rows per day
│   │           # Schema: timestamp, area_code, load_mwh
│   │
│   ├── weather/                                # Meteorological data (semi-hourly, interpolated)
│   │   ├── observed/                           # Actual measurements
│   │   │   └── {area}/
│   │   │       └── {YYYY}/{MM}/
│   │   │           └── {YYYYMMDD}.parquet      # ~47 rows (interpolated)
│   │   │           # Schema: timestamp, area_code, temperature, heat_index
│   │   │
│   │   └── forecast/                           # Forecasted values
│   │       └── {area}/
│   │           └── {YYYY}/{MM}/
│   │               └── {YYYYMMDD}.parquet
│   │               # Schema: issue_time, valid_time, area_code, temperature, heat_index
│   │
│   └── auxiliary/                              # Reference data
│       ├── holidays/
│       │   └── holidays_{year}.parquet
│       │   # Schema: date, area_code, holiday_type_id, prevcarga_code, simplified_code
│       │
│       ├── load_tiers/                         # Patamares
│       │   └── load_tiers_{area}.parquet
│       │   # Schema: hour, weekday_winter, weekend_winter, ...
│       │
│       └── dst_periods/                        # Daylight saving time
│           └── dst_periods.parquet
│           # Schema: start_date, end_date
│
├── processed/                                   # ML-ready, consolidated yearly files
│   │                                           # (Hive-style partitioning for efficient reads)
│   ├── load/
│   │   └── area_code={area}/
│   │       └── year={YYYY}/
│   │           └── data.parquet                # ~17,520 rows per year
│   │
│   └── weather/
│       ├── observed/
│       │   └── area_code={area}/
│       │       └── year={YYYY}/
│       │           └── data.parquet
│       │
│       └── forecast/
│           └── area_code={area}/
│               └── year={YYYY}/
│                   └── data.parquet
│
├── features/                                   # Engineered features
│   └── {model_type}/                           # lgbm, random_forest
│       └── {version}/                          # v1.0.0, v1.1.0
│           └── {area}/
│               ├── train_{YYYYMMDD}.parquet
│               └── inference_{YYYYMMDD}.parquet
│
├── models/                                     # Trained model artifacts
│   └── {model_type}/
│       └── {version}/
│           ├── metadata.yaml                   # Training params, metrics
│           ├── model.pkl                       # Serialized model
│           ├── scaler.pkl                      # Feature scaler
│           ├── feature_names.json              # Feature list
│           └── latest -> {version}/            # Symlink to latest
│
├── results/                                    # Model outputs
│   │
│   ├── predictions/                            # Operational forecasts
│   │   └── {area}/
│   │       └── {model_type}/
│   │           └── {YYYY}/{MM}/
│   │               └── pred_{YYYYMMDD}.parquet
│   │               # Schema:
│   │               #   - run_time (datetime, 30min granularity)
│   │               #   - step (int, 1-10: D+0 to D+9)
│   │               #   - target_time (datetime)
│   │               #   - predicted_load (float)
│   │               #   - updated_at (datetime)
│   │               # Key: (area, model, run_time, step, target_time) - latest wins
│   │
│   ├── backtests/                              # Historical evaluations
│   │   └── {backtest_id}/
│   │       ├── config.yaml                     # Backtest parameters
│   │       ├── metrics_summary.csv             # Aggregated metrics
│   │       ├── predictions_full.parquet        # All predictions
│   │       └── report.html                     # Visual report
│   │
│   └── reconciled/                             # Hierarchical reconciliation
│       └── {execution_date}/
│           ├── individual/                     # Per-area forecasts
│           ├── combined/                       # Aggregated forecasts
│           └── reconciled/                     # After reconciliation
│
├── cache/                                      # Temporary working data
│   └── {execution_id}/
│       ├── intermediate_features/
│       └── temp_predictions/
│
└── catalog/                                    # Metadata & registry
    ├── datasets.json                           # Dataset registry
    └── models.parquet                          # Model catalog
```

---

### Data Sources and Loading Patterns

#### **Data Source Catalog**

Based on comprehensive analysis of LGBM and Random Forest implementations:

| Data Source | Frequency | Storage | Key Fields (English) | Missing Value Strategy | Model Usage |
|-------------|-----------|---------|---------------------|----------------------|-------------|
| **Load** | Semi-hourly (30min) | Parquet | `timestamp`, `area_code`, `load_mwh` | Triple-pass imputation | Both |
| **Weather Observed** | Semi-hourly (interpolated from hourly) | Parquet | `timestamp`, `area_code`, `temperature`, `heat_index` | Cubic spline interpolation | Training |
| **Weather Forecast** | Semi-hourly (interpolated from hourly) | Parquet | `issue_time`, `valid_time`, `area_code`, `temperature`, `heat_index` | Cubic spline interpolation | Inference |
| **Holidays** | Daily | Parquet | `date`, `area_code`, `holiday_type_id`, `prevcarga_code`, `simplified_code` | Default to 0 (no holiday) | Both |
| **Load Tiers** | Hourly config | Parquet | `hour`, `weekday_winter`, `weekend_winter`, ... | N/A (reference data) | Both |
| **DST Periods** | Periods | Parquet | `start_date`, `end_date` | N/A (reference data) | Both |

#### **Field Name Mapping (Legacy → New)**

| Legacy (Portuguese) | New (English) | Description |
|---------------------|---------------|-------------|
| `cod_areacarga` | `area_code` | Area identifier (SP, RJ, SECO, etc.) |
| `dat_referencia` | `reference_date` | Reference date |
| `din_referencia` | `timestamp` | Reference datetime |
| `din_origemprevisaoutc` | `issue_time` | When forecast was issued |
| `val_carga` | `load_mwh` | Load value in MWh |
| `val_tmp` | `temperature` | Temperature in °C |
| `val_previsaocarga` | `predicted_load` | Predicted load value |
| `passo` | `step` | Forecast horizon (1-10: D+0 to D+9) |
| `din_alvo` | `target_time` | Target datetime being predicted |
| `din_atualizacao` | `updated_at` | Last update timestamp |
| `id_tipodiaespecial` | `holiday_type_id` | Holiday type identifier |
| `cod_prevcarga` | `prevcarga_code` | PrevCarga internal code |
| `dat_diaespecial` | `date` | Special day date |

#### **Critical Implementation Details**

**1. Timezone Handling (America/Sao_Paulo)**
```python
# All timestamps must be converted to Brazil timezone
df['DataHora'] = pd.to_datetime(df['DataHora']).dt.tz_convert('America/Sao_Paulo')

# Load data: Backward adjustment by 30 minutes for alignment
# Ensures semi-hourly intervals start at 00:30, 01:00, 01:30, etc.
df_carga['DataHora'] = df_carga['DataHora'] - pd.Timedelta(minutes=30)
```

**2. Resampling: Hourly → Semi-hourly**
```python
# Temperature forecast arrives hourly, must expand to 30-min
# Step 1: Create complete time sequence
time_range = pd.date_range(start=df['DataHora'].min(), 
                            end=df['DataHora'].max(), 
                            freq='1H')

# Step 2: Fill missing hours with same-hour previous day
df_complete = df.set_index('DataHora').reindex(time_range)
df_complete['Value'] = df_complete.groupby(df_complete.index.hour)['Value'].transform(
    lambda x: x.fillna(x.shift(24))  # 24-hour lag
)

# Step 3: Resample to 30-min with linear interpolation
df_30min = df_complete.resample('30T').interpolate(method='linear')
```

**3. Triple-Pass Imputation (Proven R Pattern)**
```python
class TriplePassImputer:
    def fit_transform(self, df: pd.DataFrame, value_col: str, date_col: str = 'Data') -> pd.DataFrame:
        \"\"\"
        Implements sophisticated imputation from R:
        - Pass 1: Forward fill within same day
        - Pass 2: Backward fill within same day  
        - Pass 3: Average interpolation using both directions
        - Pass 4: Final cross-day forward fill
        \"\"\"
        # Pass 1: Forward fill (locf) within day
        df['_prev'] = df.groupby(date_col)[value_col].ffill()
        
        # Pass 2: Backward fill (nocb) within day
        df['_next'] = df.groupby(date_col)[value_col].bfill()
        
        # Pass 3: Interpolate with average
        mask = df[value_col].isna() & df['_prev'].notna() & df['_next'].notna()
        df.loc[mask, value_col] = (df.loc[mask, '_prev'] + df.loc[mask, '_next']) / 2
        
        # Pass 4: Final forward fill across days
        df[value_col] = df[value_col].ffill()
        
        # Cleanup
        df.drop(columns=['_prev', '_next'], inplace=True)
        return df
```

**4. Complete Day Filtering**
```python
# Filter to days with complete 48 semi-hourly records
df_complete = df.groupby('Data').filter(lambda x: len(x) == 48)

# Validate chronological ordering
df_complete = df_complete.sort_values(['Data', 'Hora', 'Minuto'])
```

**5. Temperature Transformation (Random Forest)**
```python
# Wide format (daily) → Long format (hourly with lead_hour)
def transform_temperature_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Expands daily temperature forecasts to hourly records.
    Input:  dataOrigem | temp_max_1 | temp_min_1 | ... | temp_max_9 | temp_min_9
    Output: dataOrigem | day_of_horizon | hora | tmax | tmin | lead_hour
    \"\"\"
    rows = []
    for _, row in df.iterrows():
        for day in range(1, 10):  # 9 days ahead
            tmax = row[f'temp_max_{day}']
            tmin = row[f'temp_min_{day}']
            for hour in range(24):
                lead_hour = (day - 1) * 24 + (hour + 1)  # 1-216
                target_date = row['dataOrigem'] + pd.Timedelta(days=day)
                rows.append({
                    'dataOrigem': row['dataOrigem'],
                    'day_of_horizon': day,
                    'target_date': target_date,
                    'hora': hour,
                    'tmax': tmax,
                    'tmin': tmin,
                    'lead_hour': lead_hour,
                    'timestamp': target_date + pd.Timedelta(hours=hour)
                })
    return pd.DataFrame(rows)
```

---

### Component Diagram

```
┌─────────────────────────────────────────────────┐
│           Data Layer (src/data/)                │
│                                                 │
│  ┌────────────────┐      ┌──────────────────┐ │
│  │   DataLoader   │──────│  DataValidator   │ │
│  │   (Unified)    │      │  (Pydantic)      │ │
│  │ - load_carga() │      │                  │ │
│  │ - load_temp()  │      │ - CargaSchema    │ │
│  │ - load_feriado │      │ - TempSchema     │ │
│  │                │      │                  │ │
│  │ Uses:          │      │                  │ │
│  │ StorageBackend │      │                  │ │
│  └────────────────┘      └──────────────────┘ │
│           │                       │            │
│           ▼                       ▼            │
│  ┌────────────────────────────────────────┐   │
│  │      Preprocessor Pipeline             │   │
│  │                                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ImputerChain  │  │ Resampler    │  │   │
│  │  │              │→ │              │  │   │
│  │  │- forward_fill│  │- hourly→30min│  │   │
│  │  │- interpolate │  │- 30min→hourly│  │   │
│  │  └──────────────┘  └──────────────┘  │   │
│  └────────────────────────────────────────┘   │
│           │                                    │
│           ▼                                    │
│  ┌────────────────────────────────────────┐   │
│  │         DataCatalog                    │   │
│  │                                        │   │
│  │  - register_dataset()                  │   │
│  │  - get_dataset()                       │   │
│  │  - list_datasets()                     │   │
│  │  - S3 persistence                      │   │
│  └────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Data Flow

```mermaid
graph TD
    A[Storage Backend<br/>S3 or Local] -->|StorageBackend| B[DataLoader]
    A1[Multi-Source Temps] -->|D+1, D+2, ECMWF| B
    A2[Heat Index CSV] -->|CSV/Parquet| B
    A3[Holidays API/File] --> B
    
    B --> C{DataValidator}
    C -->|Valid| D[Preprocessor Pipeline]
    C -->|Invalid| E[Log & Continue]
    
    D --> F{TimezoneHandler}
    F -->|America/Sao_Paulo| G{ResamplerMixin}
    
    G -->|Hourly→30min| H{TriplePassImputer}
    G -->|30min→Hourly| H
    
    H -->|Pass 1: Forward Fill| I[Within-Day Fill]
    I -->|Pass 2: Backward Fill| J[Within-Day Fill]
    J -->|Pass 3: Interpolate| K[Average Forward+Backward]
    K -->|Pass 4: Cross-Day Fill| L[Final Forward Fill]
    
    L --> M{CompleteDayFilter}
    M -->|48 records/day| N[Clean DataFrame]
    M -->|Incomplete| E
    
    N --> O[DataCatalog]
    O -->|Register Metadata| P[Catalog S3]
    
    N --> Q[Epic-02: Feature Engineering Layer]
    
    style D fill:#e1f5ff
    style H fill:#ffe1e1
    style M fill:#fff4e1
    style Q fill:#e1ffe1
```

**Pipeline Stages:**

1. **Loading (Multi-Source):** DataLoader with unified StorageBackend handles 7 data sources (S3 or local)
2. **Validation (Schema):** Pydantic validation with continuity checks
3. **Timezone (Standardization):** Convert to America/Sao_Paulo, adjust timestamps
4. **Resampling (Frequency):** Hourly↔Semi-hourly conversion
5. **Imputation (Quality):** Triple-pass strategy proven in R
6. **Filtering (Completeness):** Only complete days (48/24 records)
7. **Cataloging (Tracking):** Metadata registration for lineage
8. **Handoff (Epic-02):** Clean DataFrames ready for feature engineering

---

### Integration Points with Epic-02

**Epic-01 Outputs → Epic-02 Inputs:**

| Output Dataset | Columns | Frequency | Epic-02 Usage |
|----------------|---------|-----------|---------------|
| `df_carga_clean` | `[timestamp, cod_area, val_cargaglobalcons, val_cargammgd]` | Semi-hourly | Lag features, seasonality, wavelets |
| `df_temp_d1` | `[timestamp, cod_area, temp_celsius]` | Semi-hourly | LGBM LOESS smoothing, BLF features |
| `df_temp_d2` | `[timestamp, cod_area, temp_celsius]` | Semi-hourly | LGBM LOESS smoothing |
| `df_temp_ecmwf` | `[timestamp, cod_area, temp_celsius]` | Semi-hourly | LGBM LOESS smoothing, BLF features |
| `df_heat_index` | `[timestamp, cod_area, heat_index]` | Semi-hourly | LGBM BLF features (optional) |
| `df_temp_weighted` | `[dataOrigem, temp_max_1-9, temp_min_1-9]` | Daily | Random Forest temperature features |
| `df_feriados` | `[date, id_tipodiaespecial, holiday_name, holiday_type]` | Daily | Holiday flags, proximity features |

**Data Quality Guarantees:**

✅ No missing values after triple-pass imputation  
✅ All timestamps timezone-aware (America/Sao_Paulo)  
✅ Only complete days (prevents training bias)  
✅ Consistent intervals (30-min or 60-min)  
✅ No duplicates within (timestamp, cod_area)  
✅ Chronologically ordered  
✅ Schema-validated (types, ranges, business rules)

---

### Component Diagram

```
src/data/
├── __init__.py
├── loaders.py              # DataLoader (unified), DataJoiner, MultiFormatLoader
├── schemas.py              # LoadSchema, WeatherObservedSchema, WeatherForecastSchema,
│                           # HolidaySchema, PredictionResultSchema, VALID_AREA_CODES
├── validators.py           # DataValidator, bulk validation utilities
├── preprocessors.py        # ImputerChain, TriplePassImputer, LagFillImputer,
│                           # ResamplerMixin, CompleteDayFilter
├── catalog.py              # DataCatalog, DatasetMetadata
├── results.py              # PredictionResultWriter, BacktestManager
└── utils.py                # PathResolver, DateRangeGenerator, TimezoneHandler

src/data/migration/          # Legacy data migration utilities
├── __init__.py
├── legacy_loader.py        # Load from old CARGAHIST, TEMPHIST format
└── converter.py            # Convert legacy → new format

tests/data/
├── __init__.py
├── test_loaders.py         # Unit tests for all loaders
├── test_schemas.py         # Schema validation tests (English names)
├── test_preprocessors.py   # Imputation and resampling tests (including triple-pass)
├── test_catalog.py         # Catalog operations tests
├── test_results.py         # Prediction result storage tests
├── test_integration.py     # End-to-end data pipeline tests
├── test_complete_day_filter.py  # Complete day filtering tests
├── test_migration.py       # Legacy migration tests
└── fixtures/
    ├── sample_load.parquet
    ├── sample_weather_obs.parquet
    ├── sample_weather_fcst.parquet
    ├── sample_holidays.parquet
    ├── sample_predictions.parquet
    ├── incomplete_days.parquet
    └── legacy/              # Legacy format samples for migration tests
        ├── CARGAHIST.csv.gz
        ├── TEMPHIST.csv.gz
        └── TEMPPREVHIST.csv.gz
```

---

### Legacy Data Migration

The system supports migration from the legacy format (per-area CSV.gz files) to the new unified structure.
**All data is standardized to semi-hourly resolution, with cubic spline interpolation for weather data.**

#### Legacy Format (old-data-structure/)
```
{area}/
├── CARGAHIST.csv.gz        → (ignored, use semi-hourly as source of truth)
├── CARGASHHIST.csv.gz      → raw_data/load/{area}/{YYYY}/{MM}/{YYYYMMDD}.parquet
├── TEMPHIST.csv.gz         → raw_data/weather/observed/{area}/{YYYY}/{MM}/{YYYYMMDD}.parquet (interpolated)
├── TEMPPREVHIST.csv.gz     → raw_data/weather/forecast/{area}/{YYYY}/{MM}/{YYYYMMDD}.parquet (interpolated)
├── FERIADOS.csv.gz         → raw_data/auxiliary/holidays/holidays_{year}.parquet
├── PATAMARES.csv.gz        → raw_data/auxiliary/load_tiers/load_tiers_{area}.parquet
├── HORAVERAO.csv.gz        → raw_data/auxiliary/dst_periods/dst_periods.parquet
└── MDLHISTP{1-10}.csv.gz   → results/predictions/{area}/.../pred_{date}.parquet (historical)
```

#### Migration Scripts

**Step 1: Migrate legacy data to daily raw files**
```bash
# Migrate a single area
python scripts/migrate_legacy_data.py --source ./old-data-structure --dest ./data --areas SP

# Migrate all areas
python scripts/migrate_legacy_data.py --source ./old-data-structure --dest ./data --all-areas
```

**Step 2: Consolidate into ML-ready yearly partitions**
```bash
# Consolidate a single area
python scripts/consolidate_for_ml.py --source ./data --areas SP

# Consolidate all areas
python scripts/consolidate_for_ml.py --source ./data --all-areas

# Consolidate specific years
python scripts/consolidate_for_ml.py --source ./data --all-areas --years 2023,2024
```

#### Weather Data Interpolation

Weather data is originally hourly and is interpolated to semi-hourly using **cubic spline interpolation**:
- Preserves original hourly values at :00
- Smoothly interpolates values at :30
- Maintains physically realistic temperature transitions

```python
# Example: Interpolated temperature values
# Original: 22.6°C (00:00), 22.3°C (01:00)
# After interpolation:
#   00:00 → 22.600°C (original)
#   00:30 → 22.308°C (interpolated via cubic spline)
#   01:00 → 22.300°C (original)
```

#### ML-Ready Data Loading

After consolidation, load data efficiently using predicate pushdown:
```python
import pandas as pd

# Load all SP data from 2022 onwards (reads only relevant partitions)
df = pd.read_parquet(
    "data/processed/load",
    filters=[("area_code", "==", "SP"), ("year", ">=", 2022)]
)
```

---

## 🔧 Implementation Details

### Key Classes

#### 1. DataLoader (Unified Storage)

```python
from typing import List, Optional
import pandas as pd
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor
from src.storage.backend import StorageBackend
from src.storage.factory import StorageFactory

class DataLoader:
    """Load data from any storage backend (S3 or local) with parallel support."""
    
    def __init__(self, storage_backend: Optional[StorageBackend] = None, max_workers: int = 4):
        """Initialize loader with storage backend.
        
        Args:
            storage_backend: Storage backend instance. If None, creates from config.
            max_workers: Number of parallel workers for loading.
        """
        self.storage = storage_backend or StorageFactory.from_config()
        self.max_workers = max_workers
    
    def load_load(
        self,
        areas: List[str],
        start_date: date,
        end_date: date,
        frequency: str = "semihourly",  # "hourly" or "semihourly"
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load electric load data for specified areas and date range.
        Works with any storage backend (S3 or local).

        Args:
            areas: List of area codes (e.g., ["RJ", "SP"])
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            frequency: "hourly" or "semihourly"
            validate: Whether to validate schema

        Returns:
            DataFrame with columns: [timestamp, area_code, load_mwh]
        """
        prefix = f"load_{frequency}"
        paths = self._generate_paths(prefix, areas, start_date, end_date)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            dfs = list(executor.map(self._load_single_file, paths))

        df = pd.concat([d for d in dfs if d is not None], ignore_index=True)
        df = df.sort_values(['area_code', 'timestamp']).reset_index(drop=True)

        if validate:
            from .validators import DataValidator, LoadSchema
            validator = DataValidator(LoadSchema)
            df = validator.validate(df)

        return df

    def load_weather_observed(
        self,
        areas: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Load observed (actual) meteorological data for training."""
        paths = self._generate_paths("weather_obs", areas, start_date, end_date,
                                      base_path="raw_data/weather/observed")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            dfs = list(executor.map(self._load_single_parquet, paths))

        df = pd.concat([d for d in dfs if d is not None], ignore_index=True)
        return df.sort_values(['area_code', 'timestamp']).reset_index(drop=True)

    def load_weather_forecast(
        self,
        areas: List[str],
        start_date: date,
        end_date: date,
        as_of: Optional[datetime] = None  # For backtesting: only forecasts available at this time
    ) -> pd.DataFrame:
        """Load forecasted meteorological data for inference/backtesting."""
        paths = self._generate_paths("weather_fcst", areas, start_date, end_date,
                                      base_path="raw_data/weather/forecast")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            dfs = list(executor.map(self._load_single_parquet, paths))

        df = pd.concat([d for d in dfs if d is not None], ignore_index=True)

        # Filter by as_of for proper backtesting (no look-ahead bias)
        if as_of is not None:
            df = df[df['issue_time'] <= as_of]

        return df.sort_values(['area_code', 'issue_time', 'valid_time']).reset_index(drop=True)

    def load_holidays(self, year: int) -> pd.DataFrame:
        """Load holiday calendar for specified year."""
        path = f"raw_data/auxiliary/holidays/holidays_{year}.parquet"
        return self._load_single_parquet(path)

    def save_prediction(
        self,
        df: pd.DataFrame,
        area: str,
        model_type: str,
        run_date: date
    ) -> str:
        """
        Save prediction results (upsert by run_time - latest wins).

        Args:
            df: DataFrame with PredictionResultSchema columns
            area: Area code
            model_type: Model type (lgbm, random_forest)
            run_date: Date of the run

        Returns:
            Path where data was saved
        """
        path = f"results/predictions/{area}/{model_type}/{run_date.year}/{run_date.month:02d}/pred_{run_date.strftime('%Y%m%d')}.parquet"

        # Load existing, merge (latest wins), save
        existing = self._load_single_parquet(path)
        if existing is not None:
            combined = pd.concat([existing, df], ignore_index=True)
            # Keep latest by (area_code, run_time, step, target_time)
            combined = combined.sort_values('updated_at').drop_duplicates(
                subset=['area_code', 'run_time', 'step', 'target_time'],
                keep='last'
            )
            df = combined

        self.storage.put(path, df.to_parquet())
        return path
    
    def _load_single_parquet(self, s3_key: str) -> Optional[pd.DataFrame]:
        """Load single Parquet file with retry logic."""
        import time
        from botocore.exceptions import ClientError
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                obj = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
                return pd.read_parquet(obj['Body'])
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    logger.warning(f"File not found: s3://{self.bucket}/{s3_key}")
                    return None
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to load {s3_key} after {max_retries} attempts")
                    raise
    
    def _generate_paths(
        self,
        prefix: str,
        areas: List[str],
        start_date: date,
        end_date: date
    ) -> List[str]:
        """Generate S3 paths for date range."""
        from .utils import DateRangeGenerator
        paths = []
        for area in areas:
            for dt in DateRangeGenerator(start_date, end_date):
                path = f"raw_data/{dt.year}/{dt.month:02d}/{prefix}_{area}_{dt.strftime('%Y%m%d')}.parquet"
                paths.append(path)
        return paths
```

#### 2. DataValidator & Schemas (English)

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import pandas as pd
from datetime import datetime, date

# Valid area codes (17 areas + 4 subsystems + losses)
VALID_AREA_CODES = [
    # Individual areas
    'RJ', 'SP', 'MG', 'ES', 'MT', 'MS', 'AC', 'RO', 'DF', 'GO',
    'PR', 'SC', 'RS', 'ALPE', 'PBRN', 'BASE', 'CE', 'PI', 'BAOE',
    'AM', 'PA', 'MA', 'TON', 'RR', 'AP',
    # Subsystems
    'SECO', 'S', 'NE', 'N',
    # Losses
    'PESE', 'PES', 'PENE', 'PEN'
]

class LoadSchema(BaseModel):
    """Schema for electric load data."""
    timestamp: datetime
    area_code: str = Field(pattern=r'^[A-Z]{2,6}$')
    load_mwh: float = Field(gt=0)  # Load must be positive

    @field_validator('area_code')
    def validate_area_code(cls, v):
        if v not in VALID_AREA_CODES:
            raise ValueError(f"Invalid area code: {v}")
        return v

class WeatherObservedSchema(BaseModel):
    """Schema for observed (actual) meteorological data."""
    timestamp: datetime
    area_code: str = Field(pattern=r'^[A-Z]{2,6}$')
    temperature: float = Field(ge=-10, le=50)  # °C, Brazil climate range
    heat_index: Optional[float] = Field(default=None, ge=15, le=55)  # °C, apparent temperature

class WeatherForecastSchema(BaseModel):
    """Schema for forecasted meteorological data with provenance."""
    issue_time: datetime       # When the forecast was made
    valid_time: datetime       # When the forecast is valid for
    area_code: str = Field(pattern=r'^[A-Z]{2,6}$')
    temperature: float = Field(ge=-10, le=50)  # °C
    heat_index: Optional[float] = Field(default=None, ge=15, le=55)  # °C

class HolidaySchema(BaseModel):
    """Schema for holiday/special day data."""
    date: date
    area_code: str = Field(pattern=r'^[A-Z]{2,6}$')
    holiday_type_id: int       # id_tipodiaespecial
    prevcarga_code: int        # cod_prevcarga
    simplified_code: int       # cod_simplificado (0, 1, 2)

class PredictionResultSchema(BaseModel):
    """Schema for model prediction results (operational outputs)."""
    area_code: str = Field(pattern=r'^[A-Z]{2,6}$')
    model_type: str            # lgbm, random_forest, etc.
    run_time: datetime         # Origin time (30min granularity)
    step: int = Field(ge=1, le=10)  # 1=D+0, 2=D+1, ..., 10=D+9
    target_time: datetime      # Target datetime being predicted
    predicted_load: float      # MWh
    updated_at: datetime       # Last update timestamp

class DataValidator:
    """Validate DataFrames against Pydantic schemas."""

    def __init__(self, schema: type[BaseModel]):
        self.schema = schema

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate DataFrame rows against schema.

        Returns validated DataFrame with invalid rows removed.
        Logs validation errors.
        """
        valid_rows = []
        invalid_count = 0

        for idx, row in df.iterrows():
            try:
                self.schema(**row.to_dict())
                valid_rows.append(idx)
            except Exception as e:
                invalid_count += 1
                logger.warning(f"Invalid row {idx}: {e}")

        logger.info(f"Validation: {len(valid_rows)} valid, {invalid_count} invalid rows")
        return df.loc[valid_rows].reset_index(drop=True)
```

#### 3. ImputerChain & Preprocessors

```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseImputer(ABC):
    """Base class for imputation strategies."""
    
    @abstractmethod
    def impute(self, series: pd.Series) -> pd.Series:
        """Impute missing values in series."""
        pass

class ForwardFillImputer(BaseImputer):
    """Forward-fill imputation."""
    
    def __init__(self, limit: int = 3):
        self.limit = limit  # Max consecutive fills
    
    def impute(self, series: pd.Series) -> pd.Series:
        return series.fillna(method='ffill', limit=self.limit)

class InterpolationImputer(BaseImputer):
    """Interpolation imputation."""
    
    def __init__(self, method: str = 'linear'):
        self.method = method  # linear, spline, polynomial
    
    def impute(self, series: pd.Series) -> pd.Series:
        return series.interpolate(method=self.method)

class ImputerChain:
    """Chain multiple imputation strategies."""
    
    def __init__(self, imputers: List[BaseImputer]):
        self.imputers = imputers
    
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply imputation chain to specified columns."""
        df_imputed = df.copy()
        
        for col in columns:
            original_nulls = df_imputed[col].isna().sum()
            
            for imputer in self.imputers:
                df_imputed[col] = imputer.impute(df_imputed[col])
            
            remaining_nulls = df_imputed[col].isna().sum()
            logger.info(f"Column {col}: {original_nulls} nulls → {remaining_nulls} nulls")
        
        return df_imputed

class ResamplerMixin:
    """Mixin for data resampling."""
    
    @staticmethod
    def resample_to_30min(df: pd.DataFrame, value_col: str = 'carga_mwh') -> pd.DataFrame:
        """Resample hourly to 30-minute using interpolation."""
        df = df.set_index('timestamp')
        df_resampled = df.resample('30T').interpolate(method='linear')
        return df_resampled.reset_index()
    
    @staticmethod
    def resample_to_hourly(df: pd.DataFrame, value_col: str = 'carga_mwh') -> pd.DataFrame:
        """Resample 30-minute to hourly using mean."""
        df = df.set_index('timestamp')
        df_resampled = df.resample('1H').mean()
        return df_resampled.reset_index()
```

#### 4. DataCatalog

```python
from typing import Optional, List, Dict
from pydantic import BaseModel
from datetime import datetime
import json
import threading

class DatasetMetadata(BaseModel):
    """Metadata for cataloged dataset."""
    name: str
    version: str
    s3_path: str
    schema_type: str  # "CargaSchema", "TemperaturaSchema", etc.
    start_date: datetime
    end_date: datetime
    row_count: int
    areas: List[str]
    created_at: datetime
    created_by: str

class DataCatalog:
    """Registry for dataset metadata."""
    
    def __init__(self, s3_bucket: str, catalog_key: str = "catalog/datasets.json"):
        self.s3_bucket = s3_bucket
        self.catalog_key = catalog_key
        self.s3_client = boto3.client('s3')
        self._cache: Dict[str, DatasetMetadata] = {}
        self._lock = threading.Lock()
        self._load_from_s3()
    
    def register(self, metadata: DatasetMetadata) -> None:
        """Register new dataset in catalog."""
        with self._lock:
            key = f"{metadata.name}:{metadata.version}"
            self._cache[key] = metadata
            self._persist_to_s3()
            logger.info(f"Registered dataset: {key}")
    
    def get(self, name: str, version: str = "latest") -> Optional[DatasetMetadata]:
        """Retrieve dataset metadata."""
        with self._lock:
            if version == "latest":
                # Find most recent version
                matches = [m for k, m in self._cache.items() if m.name == name]
                if not matches:
                    return None
                return max(matches, key=lambda m: m.created_at)
            
            key = f"{name}:{version}"
            return self._cache.get(key)
    
    def list(self, filters: Optional[Dict] = None) -> List[DatasetMetadata]:
        """List datasets with optional filters."""
        with self._lock:
            datasets = list(self._cache.values())
            
            if filters:
                if 'name' in filters:
                    datasets = [d for d in datasets if d.name == filters['name']]
                if 'areas' in filters:
                    datasets = [d for d in datasets if any(a in d.areas for a in filters['areas'])]
            
            return sorted(datasets, key=lambda d: d.created_at, reverse=True)
    
    def _load_from_s3(self) -> None:
        """Load catalog from S3."""
        try:
            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.catalog_key)
            data = json.loads(obj['Body'].read())
            self._cache = {k: DatasetMetadata(**v) for k, v in data.items()}
            logger.info(f"Loaded {len(self._cache)} datasets from catalog")
        except self.s3_client.exceptions.NoSuchKey:
            logger.info("Catalog not found, starting fresh")
    
    def _persist_to_s3(self) -> None:
        """Persist catalog to S3."""
        data = {k: v.model_dump(mode='json') for k, v in self._cache.items()}
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=self.catalog_key,
            Body=json.dumps(data, default=str),
            ContentType='application/json'
        )
```

---

## 🧪 Testing Strategy

### Unit Tests

```python
# tests/data/test_loaders.py
import pytest
from moto import mock_s3
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from src.data.loaders import DataLoader
from src.storage.factory import StorageFactory

@mock_s3
def test_load_carga_from_s3():
    """Test successful load of carga data from S3."""
    storage_backend = StorageFactory.create(backend_type="s3", bucket="test-bucket")
    loader = DataLoader(storage_backend=storage_backend)
    # Setup mock S3
    s3 = boto3.client('s3', region_name='us-east-1')
    s3.create_bucket(Bucket='test-bucket')
    
    # Upload test data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
        'cod_area': 'RJ',
        'carga_mwh': range(1000, 1024)
    })
    s3.put_object(
        Bucket='test-bucket',
        Key='raw_data/2024/01/carga_horaria_RJ_20240101.parquet',
        Body=df.to_parquet()
    )
    
    # Test loader
    loader = S3ParquetLoader('test-bucket')
    result = loader.load_carga(['RJ'], date(2024, 1, 1), date(2024, 1, 1))
    
    assert len(result) == 24
    assert result['cod_area'].unique()[0] == 'RJ'
    assert result['carga_mwh'].min() == 1000

def test_load_carga_from_local():
    """Test successful load of carga data from local filesystem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup test data
        test_path = Path(tmpdir) / "raw_data" / "2024" / "01"
        test_path.mkdir(parents=True)
        # Create test parquet file...
        
        storage_backend = StorageFactory.create(backend_type="local", base_path=tmpdir)
        loader = DataLoader(storage_backend=storage_backend)
        # Test loading...

@mock_s3
def test_load_missing_file_s3():
    """Test handling of missing S3 file."""
    s3 = boto3.client('s3', region_name='us-east-1')
    s3.create_bucket(Bucket='test-bucket')
    
    loader = S3ParquetLoader('test-bucket')
    result = loader.load_carga(['SP'], date(2024, 1, 1), date(2024, 1, 1))
    
    assert len(result) == 0  # Should return empty DataFrame, not error

# tests/data/test_preprocessors.py
def test_forward_fill_imputation():
    """Test forward-fill imputation."""
    series = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan])
    imputer = ForwardFillImputer(limit=2)
    result = imputer.impute(series)
    
    expected = pd.Series([1.0, 1.0, 1.0, 4.0, 4.0])
    pd.testing.assert_series_equal(result, expected)

def test_resample_to_30min():
    """Test hourly to 30-minute resampling."""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=3, freq='H'),
        'carga_mwh': [100.0, 200.0, 300.0]
    })
    
    result = ResamplerMixin.resample_to_30min(df)
    
    assert len(result) == 5  # 3 hours = 5 intervals (0:00, 0:30, 1:00, 1:30, 2:00)
    assert result['carga_mwh'].iloc[1] == 150.0  # Interpolated value
```

### Integration Tests

```python
# tests/data/test_integration.py
@mock_s3
def test_full_data_pipeline():
    """Test complete data loading and preprocessing pipeline."""
    # Setup S3 with test data
    setup_test_s3_data()
    
    # Load data
    loader = S3ParquetLoader('test-bucket')
    df = loader.load_carga(['RJ', 'SP'], date(2024, 1, 1), date(2024, 1, 31))
    
    # Validate
    validator = DataValidator(CargaSchema)
    df = validator.validate(df)
    
    # Preprocess
    imputer = ImputerChain([ForwardFillImputer(), InterpolationImputer()])
    df = imputer.fit_transform(df, ['carga_mwh'])
    
    # Resample
    df = ResamplerMixin.resample_to_30min(df)
    
    # Assertions
    assert df['carga_mwh'].isna().sum() == 0
    assert len(df) > 0
    assert df['timestamp'].diff().mode()[0] == pd.Timedelta('30min')
```

### Performance Tests

```python
# tests/data/test_performance.py
def test_load_performance_one_year():
    """Verify load time for 1 year of data."""
    import time
    
    start = time.time()
    loader = S3ParquetLoader('test-bucket', max_workers=8)
    df = loader.load_carga(['SECO'], date(2023, 1, 1), date(2023, 12, 31))
    elapsed = time.time() - start
    
    assert elapsed < 30.0  # Must complete in under 30 seconds
    assert len(df) > 0
```

---

## 📊 Configuration Examples

### config/storage.yaml

```yaml
storage:
  type: s3
  bucket: prevcarga-bucket-sandbox
  region: us-east-1
  paths:
    raw_data: raw_data/
    catalog: catalog/datasets.json
  
  loader:
    max_workers: 8
    retry_attempts: 3
    retry_backoff: 2  # exponential backoff base

preprocessing:
  imputation:
    # Triple-pass imputation for load data (LGBM approach)
    carga_mwh:
      - type: forward_fill
        limit: null  # Within same day
        groupby: ['Data']  # Don't cross day boundaries
      - type: backward_fill
        limit: null
        groupby: ['Data']
      - type: average_interpolation
        use_forward_backward: true
      - type: forward_fill
        limit: null  # Final cross-day fill
    
    # Temperature: Lag fill + interpolation (LGBM approach)
    temp_celsius:
      - type: lag_fill
        lag_hours: [24, 48]  # Same hour previous day(s)
        groupby: ['Hora']
      - type: interpolation
        method: linear
    
    # Heat index: Simple forward/backward + interpolation
    heat_index:
      - type: forward_fill
        limit: 4  # Max 2 hours
      - type: backward_fill
        limit: 4
      - type: interpolation
        method: linear
  
  resampling:
    target_frequency: 30min
    method: interpolation

validation:
  strict_mode: false  # If true, fail on any invalid data
  log_invalid_rows: true
  max_invalid_percentage: 5.0  # Fail if >5% rows invalid
```

---

## ✅ Definition of Done

### Code Completeness
- [ ] All 6 user stories implemented
- [ ] `DataLoader` with unified storage backend (S3 and local) with parallel loading and multi-source support
- [ ] Integration with Epic-00 `StorageBackend` abstraction
- [ ] Pydantic schemas for all data types (load, temperature, heat index, holidays)
- [ ] `ImputerChain` with 4+ strategies (forward fill, backward fill, lag fill, interpolation)
- [ ] Triple-pass imputation implemented per LGBM analysis
- [ ] `ResamplerMixin` for bidirectional frequency conversion
- [ ] `CompleteDayFilter` for data quality enforcement
- [ ] `DataCatalog` with S3 persistence
- [ ] Timezone handling utilities (America/Sao_Paulo)

### Testing
- [ ] Unit test coverage >80%
- [ ] Integration tests for full pipeline
- [ ] Performance tests passing (<30s for 1 year)
- [ ] Edge case tests (missing data, invalid schemas, DST transitions)

### Performance
- [ ] Load 1 year of 1 area: <5s
- [ ] Load 1 year of SECO subsystem (11 areas): <30s
- [ ] Validation overhead: <1%
- [ ] Preprocessing 1 year: <3s

### Documentation
- [ ] API documentation (docstrings)
- [ ] Usage examples in `docs/user-guide/data-loading.md`
- [ ] Data format specifications documented
- [ ] Configuration guide with examples

### Integration
- [ ] Works with Epic-00 logging configuration
- [ ] S3 credentials via boto3 standard methods
- [ ] Compatible with Phase 2 Feature Engineering interface
- [ ] Local test scripts run successfully

---

## 📝 Validation Commands

```bash
# Run data layer tests
./scripts/test.sh tests/data/

# Run with coverage
pytest tests/data/ --cov=src/data --cov-report=html

# Performance benchmarks
pytest tests/data/test_performance.py -v

# Integration test with local storage (default)
pytest tests/data/test_integration.py

# Integration test with real S3 (requires credentials)
pytest tests/data/test_integration.py --storage-backend=s3 --s3-bucket=prevcarga-test

# Lint data layer code
./scripts/lint.sh src/data/

# Type checking
mypy src/data/
```

---

## 🎯 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Load Time (1 year, 1 area, single source) | <5s | Performance test |
| Load Time (1 year, SECO, all sources) | <30s | Performance test |
| Load Time (heat index + temperature multi-source) | <10s | Performance test |
| Validation Accuracy | >99% | Schema test suite |
| Imputation Coverage | 100% | No nulls after triple-pass preprocessing |
| Complete Day Filter Accuracy | 100% | No incomplete days in output |
| Test Coverage | >80% | pytest-cov |
| Documentation Coverage | 100% | All public APIs documented |
| Multi-source Join Integrity | 100% | No data loss in joins |

---

## 🔗 Dependencies

### Upstream Dependencies
- **Epic-00**: Requires unified `StorageBackend` abstraction and logging setup

### Downstream Dependencies
- **Epic-02**: Feature Engineering depends on clean DataFrames from this layer
- **Epic-03/04**: Models depend on validated, preprocessed data
- **Epic-08**: Orchestrator uses DataCatalog for dataset discovery
- **Epic-09A**: CLI commands will use DataLoader for all data operations

### CLI Integration
The unified storage approach enables CLI commands to work seamlessly with both local and cloud storage:

**Example CLI Usage:**
```bash
# Load data from local filesystem
prevcarga data load --area RJ --start-date 2024-01-01 --end-date 2024-12-31 --storage-backend local

# Load data from S3
prevcarga data load --area RJ --start-date 2024-01-01 --end-date 2024-12-31 --storage-backend s3

# Validate data (works with any backend)
prevcarga data validate --path raw_data/2024/01/carga_horaria_RJ_20240101.parquet

# List catalog (from configured backend)
prevcarga catalog list --filter area=RJ

# Train model with local data
prevcarga train --model lgbm --data-path ./data/raw_data/ --storage-backend local

# Train model with S3 data
prevcarga train --model lgbm --data-path s3://bucket/raw_data/ --storage-backend s3
```

**Key CLI Features Enabled by Unified Storage:**
- ✅ **Path Transparency**: CLI accepts paths without knowing backend (auto-detected from prefix)
- ✅ **Config Override**: `--storage-backend` flag overrides config file
- ✅ **Consistent Interface**: Same command syntax for local and S3
- ✅ **Development Workflow**: Develop locally, deploy to production without changes
- ✅ **Testing**: Easy integration testing with local fixtures

---

## 🚨 Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| S3 performance issues | High | Low | Implement caching, parallel loading |
| Invalid data quality | High | Medium | Strict validation, detailed logging |
| Missing historical data | Medium | Medium | Graceful handling, fallback strategies |
| Timezone handling complexity | Medium | Low | Use timezone-aware datetime throughout |
| Memory issues with large loads | Medium | Low | Chunked processing, streaming where possible |

---

## 📅 Timeline

### Week 2
- **Days 1-2**: Implement `S3ParquetLoader` with multi-source support (load, temperature D+1/D+2, ECMWF, heat index, holidays)
- **Days 3-4**: Create Pydantic schemas (`CargaSchema`, `TemperaturaSchema`, `HeatIndexSchema`, `FeriadoSchema`) and `DataValidator` with continuity checks
- **Day 5**: Unit tests for loaders and validators (including multi-source tests)

### Week 3
- **Days 1-2**: Implement preprocessing:
  - `TriplePassImputer` (proven R pattern)
  - `LagFillImputer` (24h/48h same-hour)
  - `ResamplerMixin` (hourly ↔ semi-hourly)
  - `TimezoneHandler` (America/Sao_Paulo + timestamp adjustment)
  - `CompleteDayFilter` (48/24 record validation)
- **Day 3**: Create `DataCatalog` with S3 persistence and `TemperatureWideToLongTransformer` (Random Forest)
- **Days 4-5**: Integration tests (complete pipeline with all sources), performance optimization, comprehensive documentation

---

## 🎉 Completion Checklist

- [ ] All 6 user stories accepted by stakeholders
- [ ] Code reviewed and merged to main branch
- [ ] Test coverage >80% verified (target >85% for critical components)
- [ ] Performance benchmarks passing:
  - [ ] Load 1 year, 1 area: <5s
  - [ ] Load 1 year, SECO (11 areas): <30s
  - [ ] Load all auxiliary sources (1 year): <10s
  - [ ] Triple-pass imputation (1 year): <3s
  - [ ] Complete day filter (1 year): <1s
- [ ] Documentation complete and published
- [ ] Integration with Epic-00 validated (logging, S3 config)
- [ ] Demo prepared showing:
  - [ ] Load 1 year of multi-area, multi-source data (7 sources)
  - [ ] Schema validation catching errors (invalid ranges, missing fields, duplicates)
  - [ ] Triple-pass imputation handling missing values (show before/after stats)
  - [ ] Complete day filter removing incomplete days (show filtering report)
  - [ ] Timezone handling and timestamp adjustment (America/Sao_Paulo)
  - [ ] Temperature transformation: wide→long (Random Forest use case)
  - [ ] Catalog registration and discovery (metadata tracking)
- [ ] Handoff to Epic-02 validated:
  - [ ] Output DataFrames meet Epic-02 input requirements
  - [ ] Data quality guarantees documented
  - [ ] Sample datasets provided for feature engineering testing

---

## 📚 Appendix: Key Learnings from R Implementation Analysis

### A.1 Critical Success Factors

Based on analyzing both LGBM and Random Forest R implementations:

1. **Triple-Pass Imputation is Essential**
   - R implementation proven effective over 2+ years production use
   - Within-day fills preserve daily patterns
   - Cross-day fills pragmatic for longer gaps
   - Average interpolation smoother than simple forward/backward

2. **Complete Day Filtering Prevents Bias**
   - Incomplete days skew daily aggregations
   - Prevents edge effects in seasonal features
   - Critical for models using daily statistics

3. **Timezone and Timestamp Alignment**
   - 30-minute backward adjustment for load data non-obvious but critical
   - America/Sao_Paulo timezone required for DST handling
   - Chronological ordering must be validated, not assumed

4. **Multi-Source Temperature Complexity**
   - LGBM uses 4 temperature sources (D+1, D+2, ECMWF, Heat Index)
   - Random Forest uses weighted daily forecasts (wide format)
   - Both patterns must be supported for model flexibility

5. **Lag Fill for Temperature Resampling**
   - Simple interpolation insufficient for hourly→semi-hourly
   - Same-hour previous day preserves diurnal patterns
   - 24h/48h fallback provides robustness

### A.2 Python Migration Considerations

**Pandas vs Polars:**
- Start with Pandas (proven ecosystem, easier debugging)
- Consider Polars later for performance (lazy evaluation, multi-threading)
- Abstract interface allows swapping implementations

**Memory Management:**
- Use chunked processing for multi-year backtests
- Leverage Parquet column selection (read only needed columns)
- Consider Dask for distributed processing (future optimization)

**Testing Priorities:**
1. DST transitions (spring forward: missing hour, fall back: duplicate hour)
2. Leap years (Feb 29 handling)
3. Dataset boundaries (first/last day completeness)
4. Multi-source joins (ensure no data loss)
5. Imputation edge cases (all-NA days, long gaps)

### A.3 Integration with Epic-02 Feature Engineering

**Epic-01 Deliverables → Epic-02 Inputs:**

| Epic-01 Output | Format | Frequency | Epic-02 Plugin Using It |
|----------------|--------|-----------|-------------------------|
| `df_carga_clean` | Long (timestamp-based) | Semi-hourly | `LagFeaturePlugin`, `SeasonalityPlugin`, `WaveletPlugin` (LGBM) |
| `df_temp_d1_smoothed` | Long | Semi-hourly | `LoessSmoothingPlugin`, `BLFStrategyPlugin` (LGBM) |
| `df_temp_weighted` | Wide (daily) | Daily (9 days) | `TemperatureFeaturePlugin` (Random Forest) |
| `df_feriados` | Daily calendar | Daily | `HolidayProximityPlugin`, `SpecialPeriodPlugin` (Both) |

**Data Quality Contract:**
- ✅ No NULLs (100% imputation coverage)
- ✅ Consistent intervals (30-min or 60-min)
- ✅ Complete days only (48 or 24 records)
- ✅ Timezone-aware timestamps
- ✅ No duplicates
- ✅ Schema-validated
- ✅ Chronologically ordered

This contract ensures Epic-02 can focus on feature engineering logic without data quality concerns.

---

**Epic Owner:** Data Engineering Team
**Stakeholders:** ML Engineering, Data Science, Platform Engineering
**Status:** Ready for Implementation
**Last Updated:** November 21, 2025 (Updated with semi-hourly only structure, cubic spline interpolation for weather, Hive-style yearly partitions for ML efficiency, and migration/consolidation scripts)
