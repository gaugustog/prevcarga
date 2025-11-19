# Feature Engineering Analysis - Random Forest Model

**Project:** PrevCarga - Electric Load Forecasting  
**Model:** Random Forest Multi-Output (216 hours ahead)  
**Date:** November 17, 2025  
**Purpose:** Inform Epic-01 (Data Infrastructure) and Epic-02 (Feature Engineering)

---

## Table of Contents

1. [Overview](#overview)
2. [Data Infrastructure (Epic-01 Foundation)](#data-infrastructure-epic-01-foundation)
3. [Feature Engineering System (Epic-02 Foundation)](#feature-engineering-system-epic-02-foundation)
4. [Model-Specific Feature Selection](#model-specific-feature-selection)
5. [Key Insights for Python Implementation](#key-insights-for-python-implementation)

---

## Overview

The existing R implementation provides a comprehensive framework for electric load forecasting using Random Forest with a **multi-output architecture**. The system forecasts load for **216 hours ahead (9 days)** using one independent model per lead hour, with intelligent feature selection based on the forecast horizon.

### Key Architecture Characteristics

- **Multi-output approach:** 216 independent Random Forest models (one per hour)
- **Forecast horizon:** D+0 to D+8 (9 days, 24 hours each)
- **Training period:** Rolling 13-month window
- **Test period:** 30 days per evaluation cycle
- **Feature selection:** Dynamic per lead hour (not all features used for all horizons)

---

## Data Infrastructure (Epic-01 Foundation)

### 2.1 Data Sources

The system loads three primary data sources from external APIs:

#### **Load Data** (`get_carga`)
- **Source:** `CGPeriodo()` API function
- **Raw granularity:** Semi-hourly (30-minute intervals)
- **Preprocessing:** Aggregated to hourly mean values
- **Key field:** `mean_val_cargaglobalcons` (hourly load in MW)
- **Structure:**
  ```
  dia (date) | hora (0-23) | mean_val_cargaglobalcons (MW)
  ```

#### **Temperature Data** (`get_temp_data`)
- **Source:** `TemperaturaPonderada()` API function
- **Type:** Forecast temperature (not observed)
- **Horizon:** 10 days ahead from each forecast origin
- **Variables:**
  - `temp_max_[1-9]`: Maximum daily temperature for day D+n
  - `temp_min_[1-9]`: Minimum daily temperature for day D+n
- **Structure:**
  ```
  dataOrigem (date) | temp_max_1 | temp_min_1 | ... | temp_max_9 | temp_min_9
  ```
- **Handling missing data:** Falls back to previous day's forecast if API returns empty

#### **Holiday Data** (`get_feriado`)
- **Source:** `GetDiasEspeciaisAssociados()` API + hardcoded fallback list
- **Coverage:** 2022-2025 Brazilian national/regional holidays
- **Special periods:**
  - Carnival (3 days)
  - Christmas period (Dec 23-29)
  - New Year period (Dec 30 - Jan 4)
- **Structure:** Vector of dates

### 2.2 Data Preprocessing Pipeline

#### **Temperature Adjustment** (`ajusta_dt_temp`)
Transforms wide-format daily temperature into long-format hourly:
```
Input:  dataOrigem | temp_max_1 | temp_min_1 | ... | temp_max_9 | temp_min_9
Output: dataOrigem | day_of_horizon | target_date | hora (0-23) | tmax | tmin | ts | lead_hour
```
- Expands each daily forecast to 24 hourly records
- Calculates `lead_hour = (day_of_horizon - 1) * 24 + (hour + 1)` (1-216)
- Creates timestamp for joining with load data

#### **Data Union** (`uniao_dts`)
Joins temperature and load data on timestamp:
```sql
-- Pseudo-SQL representation
SELECT temp.*, load.mean_val_cargaglobalcons
FROM dt_temp AS temp
INNER JOIN dt_load AS load
  ON temp.ts = load.ts
```
- Inner join ensures only complete records (no missing load or temp)
- Results in long-format table with one row per (dataOrigem, lead_hour) combination

### 2.3 Data Quality Considerations

**For Epic-01 Implementation:**

1. **Schema Validation (Pydantic)**
   - Load data: validate hour range (0-23), non-negative MW values
   - Temperature: validate temp_max > temp_min, reasonable ranges (-10 to 50°C)
   - Holiday: validate date format, no duplicates

2. **Missing Value Handling**
   - Load data: Use interpolation for short gaps (<2 hours), flag longer gaps
   - Temperature: Implement fallback strategy (previous day or climatology)
   - Holiday: Maintain comprehensive static list + API fallback

3. **Resampling Strategy**
   - Current: Semi-hourly → hourly (mean aggregation)
   - Future: Support hourly → semi-hourly expansion for finer granularity

4. **Data Catalog**
   - Track data versions (API call timestamps)
   - Log data quality metrics (completeness %, anomaly counts)
   - Store metadata (subsystem codes, date ranges, feature versions)

---

## Feature Engineering System (Epic-02 Foundation)

The R implementation creates **10 feature groups** with varying granularity (origin-level, day-level, hour-level). The system is implemented in `multioutput_dt()` function.

### 3.1 Feature Categories Overview

| Feature Group | Granularity | Count (approx) | Selection Logic |
|---------------|-------------|----------------|-----------------|
| Origin Calendar | Origin | 4 | Always included |
| Temperature | Day | 18 (9 days × 2) | Only up to target day |
| Day-of-Week | Day | 63 (9 days × 7) | Only target day |
| Hour-Specific | Hour | 1,296 (216 hours × 6) | Only target hour |
| Seasonal | Day | 36 (9 days × 4) | Only target day |
| Christmas/New Year | Day | 20 (10 × 2) | All days |
| Holiday Proximity | Origin | 3 | Always included |
| Previous Day Load | Origin | 8 | Always included |
| Previous 2-Day Load | Origin | 8 | Always included |
| Recent Load Level | Origin | 3 | Always included |

**Total feature space:** ~1,400+ features  
**Average features per model:** ~50-100 (through dynamic selection)

---

### 3.2 Detailed Feature Specifications

#### **3.2.1 Origin Calendar Features**
Features describing the forecast origin date:

```python
# Pseudo-code structure
origin_dow: int              # Day of week (1=Monday, 7=Sunday)
origin_month: int            # Month (1-12)
origin_is_weekend: bool      # True if Saturday or Sunday
origin_is_feriado: bool      # True if holiday
```

**Rationale:** Captures baseline temporal patterns and special day effects at forecast time.

---

#### **3.2.2 Temperature Features**
Daily max/min forecasts for each horizon day:

```python
# For each day d in [1, 9]:
tmax_d: float  # Maximum temperature forecast for day D+d
tmin_d: float  # Minimum temperature forecast for day D+d
```

**Selection logic in training:**
- For lead_hour in day D+3: Use `tmax_1`, `tmin_1`, `tmax_2`, `tmin_2`, `tmax_3`, `tmin_3`
- For lead_hour in day D+7: Use all temperature features up to day 7

**Rationale:** Temperature directly affects cooling/heating loads. Only future temperatures (up to target day) are available at forecast time.

**Python implementation note:**
- Consider derived features: `tmean = (tmax + tmin) / 2`
- Cooling degree days: `cdd = max(tmean - 20, 0)`
- Heating degree days: `hdd = max(18 - tmean, 0)`

---

#### **3.2.3 Day-of-Week Features**
Calendar features for each target day in horizon:

```python
# For each day d in [1, 9]:
day{d}_dow: int                  # Day of week (1-7)
day{d}_is_weekend: bool          # Weekend flag
day{d}_is_monday: bool           # One-hot encoding
day{d}_is_tuesday: bool          # One-hot encoding
day{d}_is_wednesday: bool        # ...
day{d}_is_thursday: bool
day{d}_is_friday: bool
day{d}_is_saturday: bool
day{d}_is_sunday: bool
day{d}_is_feriado: bool          # Holiday flag
```

**Selection logic:** Only features for the target day are included.
- Example: For lead_hour 73 (day 4, hour 1), only `day4_*` features are used.

**Rationale:** Weekly patterns are strong in electric load (workdays vs weekends), holidays disrupt patterns.

---

#### **3.2.4 Hour-Specific Features**
Fine-grained temporal encoding for each lead hour:

```python
# For each lead_hour h in [1, 216]:
h{h:03d}_is_daily_peak: bool           # Hour in 9-18 range
h{h:03d}_is_evening_peak: bool         # Hour in 18-23 range
h{h:03d}_is_minimum: bool              # Hour in 0-8 range
h{h:03d}_hour_sin: float               # sin(2π × hour / 24)
h{h:03d}_hour_cos: float               # cos(2π × hour / 24)
h{h:03d}_hour_month_interaction: float # hour × target_month
```

**Selection logic:** Only features for the specific target lead hour are used.
- Example: For lead_hour 001, only `h001_*` features are included.

**Rationale:**
- Peak periods have different load dynamics (commercial/industrial activity)
- Cyclical encoding captures smooth transitions across midnight
- Hour×Month interaction captures seasonal shifts in daily patterns

**Critical periods defined:**
- Daily peak: 9:00-18:00 (business hours)
- Evening peak: 18:00-23:00 (residential peak)
- Minimum load: 0:00-8:00 (overnight)

---

#### **3.2.5 Seasonal Features**
Seasonal context for each target day:

```python
# For each day d in [1, 9]:
day{d}_month: int           # Month of target day (1-12)
day{d}_quarter: int         # Quarter (1-4)
day{d}_is_summer: bool      # Dec-Feb (Southern Hemisphere)
day{d}_is_winter: bool      # Jun-Aug
```

**Selection logic:** Only features for target day are included.

**Rationale:** Brazil has distinct seasonal load patterns:
- Summer (Dec-Feb): High cooling loads (especially in SE/CO)
- Winter (Jun-Aug): Higher evening peaks (heating, lighting)

---

#### **3.2.6 Christmas/New Year Period Features**
Special period indicators with extended influence windows:

```python
# Origin features:
origin_is_christmas: bool    # Origin in Dec 23-29
origin_is_newyear: bool      # Origin in Dec 30 - Jan 4

# For each day d in [1, 9]:
day{d}_is_christmas: bool    # Target day in Dec 23-29
day{d}_is_newyear: bool      # Target day in Dec 30 - Jan 4
```

**Selection logic:** All features included (not filtered by horizon).

**Rationale:** These periods have unique load patterns:
- Reduced industrial/commercial loads
- Extended multi-day effect (not just single holiday)
- Different from regular holidays due to vacation schedules

**Implementation note:** Separate from regular holiday flags to capture multi-day effects.

---

#### **3.2.7 Holiday Proximity Features**
Context about holidays relative to forecast origin:

```python
has_holiday_in_horizon: bool       # Any holiday in next 9 days
days_to_next_holiday: int          # Days until next holiday (max 365)
days_since_last_holiday: int       # Days since last holiday (max 365)
```

**Selection logic:** Always included (origin-level features).

**Rationale:**
- Pre-holiday days often have different load patterns (early closures)
- Post-holiday days have ramp-up effects
- Captures anticipation effects not visible in binary holiday flags

---

#### **3.2.8 Previous Day Load Features**
Recent historical load statistics (D-1):

```python
prev_day_mean: float           # Average load of previous day
prev_day_peak: float           # Maximum load of previous day
prev_day_min: float            # Minimum load of previous day
prev_day_daily_peak: float     # Peak during 9-17 hours
prev_day_evening_peak: float   # Peak during 18-23 hours
prev_day_dow: int              # Day of week of D-1
prev_day_is_weekend: bool      # Weekend flag for D-1
prev_day_is_feriado: bool      # Holiday flag for D-1
```

**Selection logic:** Always included (origin-level features).

**Rationale:**
- Strong autoregressive patterns in load
- Recent history provides baseline for forecast
- Calendar context of previous day helps interpret the values

**Join logic:** `dataOrigem` joins with `dia + 1` from daily load statistics.

---

#### **3.2.9 Previous 2-Day Load Features**
Extended recent history (D-2):

```python
prev_2day_mean: float
prev_2day_peak: float
prev_2day_min: float
prev_2day_daily_peak: float
prev_2day_evening_peak: float
prev_2day_dow: int
prev_2day_is_weekend: bool
prev_2day_is_feriado: bool
```

**Selection logic:** Always included (origin-level features).

**Rationale:** Provides additional context for trend detection and weekend transitions.

---

#### **3.2.10 Recent Load Level Features**
Adaptive baseline from longer history:

```python
prev_7day_mean: float            # Average daily load over D-7 to D-1
prev_3day_mean: float            # Average daily load over D-3 to D-1
recent_vs_historical: float      # Ratio: prev_3day_mean / prev_7day_mean
```

**Selection logic:** Always included (origin-level features).

**Rationale:**
- Captures business cycle effects (economic activity changes)
- Detects trend shifts (unusual weeks due to events)
- `recent_vs_historical` is a relative measure that adapts to load level changes

**Calculation details:**
- Uses daily aggregated means (not hourly)
- Robust to missing values (NA if insufficient data)

---

### 3.3 Feature Selection Strategy

The R implementation uses a **horizon-aware feature selection** approach:

```r
# Pseudo-code from train_multioutput_rf
for (lead_hour in 1:216) {
  day_ahead = ceiling(lead_hour / 24)
  hour_of_day = ((lead_hour - 1) %% 24)
  
  selected_features = c(
    origin_features,                           # All origin features
    temperature_features[1:day_ahead],         # Only up to target day
    dow_features[day_ahead],                   # Only target day
    hour_features[lead_hour],                  # Only target hour
    seasonal_features[day_ahead],              # Only target day
    prev_day_features,                         # All
    prev_2day_features,                        # All
    recent_level_features                      # All
  )
  
  train_model(lead_hour, selected_features)
}
```

**Result:** Each of the 216 models sees only relevant features, reducing dimensionality and preventing data leakage.

---

### 3.4 Data Structure Transformation

The feature engineering pipeline performs a critical reshape operation:

#### **Input Format (Long)**
```
dataOrigem | lead_hour | tmax | tmin | mean_val_cargaglobalcons
2024-10-01 | 1         | 28.5 | 18.2 | 45000
2024-10-01 | 2         | 28.5 | 18.2 | 44500
...
2024-10-01 | 216       | 30.1 | 19.5 | 50000
2024-10-02 | 1         | 29.0 | 18.8 | 45200
```

#### **Output Format (Wide)**
```
dataOrigem | [1400+ features] | y_h001 | y_h002 | ... | y_h216
2024-10-01 | ...              | 45000  | 44500  | ... | 50000
2024-10-02 | ...              | 45200  | 44800  | ... | 50500
```

**Key operations:**
1. Temperature: `dcast(dataOrigem ~ day_of_horizon, value.var = c(tmax, tmin))`
2. Targets: `dcast(dataOrigem ~ lead_hour, value.var = mean_val_cargaglobalcons)`
3. Features: Calculated per `dataOrigem`, resulting in one row per forecast origin

---

## Model-Specific Feature Selection

### 4.1 Example: Lead Hour 25 (Day 2, Hour 1)

**Selected features:**
- Origin: `origin_dow`, `origin_month`, `origin_is_weekend`, `origin_is_feriado`
- Temperature: `tmax_1`, `tmin_1`, `tmax_2`, `tmin_2` (days 1-2 only)
- Day-of-week: `day2_dow`, `day2_is_weekend`, `day2_is_monday`, ..., `day2_is_feriado`
- Hour: `h025_is_daily_peak`, `h025_is_evening_peak`, `h025_is_minimum`, `h025_hour_sin`, `h025_hour_cos`, `h025_hour_month_interaction`
- Seasonal: `day2_month`, `day2_quarter`, `day2_is_summer`, `day2_is_winter`
- Previous day: All 8 features
- Previous 2-day: All 8 features
- Recent level: All 3 features

**Total:** ~50 features (vs 1400+ in full feature space)

### 4.2 Example: Lead Hour 168 (Day 7, Hour 24)

**Selected features:**
- Origin: Same 4 features
- Temperature: All features (days 1-7)
- Day-of-week: `day7_*` features
- Hour: `h168_*` features
- Seasonal: `day7_*` features
- Historical load: Same 19 features

**Total:** ~80 features

---

## Key Insights for Python Implementation

### 5.1 Epic-01: Data Infrastructure Layer

**Must-have capabilities:**

1. **S3 Parquet Loaders**
   - Load historical load data (hourly granularity)
   - Load temperature forecasts (daily, expand to hourly)
   - Load holiday calendar (static + API)

2. **Pydantic Schemas**
   ```python
   class LoadDataSchema(BaseModel):
       dia: date
       hora: int = Field(ge=0, le=23)
       mean_val_cargaglobalcons: float = Field(gt=0)
   
   class TemperatureSchema(BaseModel):
       dataOrigem: date
       day_of_horizon: int = Field(ge=1, le=9)
       temp_max: float = Field(ge=-10, le=50)
       temp_min: float = Field(ge=-10, le=50)
       
       @validator('temp_max')
       def check_max_gt_min(cls, v, values):
           if 'temp_min' in values and v <= values['temp_min']:
               raise ValueError('temp_max must be > temp_min')
           return v
   ```

3. **Missing Value Imputation**
   - Load: Linear interpolation for gaps <2 hours, flag longer gaps
   - Temperature: Fallback to previous day's forecast, then climatology
   - Holiday: Maintain comprehensive static list

4. **Hourly ↔ Semi-hourly Resampling**
   - Downsampling: Mean aggregation (current approach)
   - Upsampling: Forward fill or interpolation (for future use)

5. **Data Catalog System**
   ```python
   class DataCatalogEntry(BaseModel):
       dataset_name: str
       subsystem: str
       date_range: tuple[date, date]
       row_count: int
       completeness_pct: float
       loaded_at: datetime
       source_version: str
   ```

---

### 5.2 Epic-02: Feature Engineering System

**Plugin Architecture Design:**

```python
from abc import ABC, abstractmethod

class BaseFeaturePlugin(ABC):
    """Base class for all feature plugins."""
    
    @abstractmethod
    def compute(self, df: pd.DataFrame, context: dict) -> pd.DataFrame:
        """Compute features and return DataFrame with new columns."""
        pass
    
    @abstractmethod
    def get_feature_names(self, context: dict) -> list[str]:
        """Return list of feature names that will be generated."""
        pass
    
    @property
    @abstractmethod
    def plugin_name(self) -> str:
        pass
```

**Essential Plugins (based on R implementation):**

1. **CalendarFeaturePlugin**
   - Origin-level: dow, month, is_weekend, is_holiday
   - Day-level: target day dow, weekend flags, one-hot encoding

2. **TemperatureFeaturePlugin**
   - Daily max/min per horizon day
   - Optional derived features: mean, CDD, HDD

3. **CyclicalTimePlugin**
   - Hour-specific: sin/cos encoding
   - Period flags: daily_peak, evening_peak, minimum

4. **HolidayProximityPlugin**
   - has_holiday_in_horizon
   - days_to_next_holiday, days_since_last_holiday

5. **LagFeaturePlugin**
   - Previous day statistics (mean, peak, min, period-specific peaks)
   - Previous 2-day statistics
   - Recent load level (7-day, 3-day, ratio)

6. **SeasonalFeaturePlugin**
   - Month, quarter per target day
   - Season flags (summer/winter)

7. **SpecialPeriodPlugin**
   - Christmas period (Dec 23-29)
   - New Year period (Dec 30 - Jan 4)

8. **InteractionFeaturePlugin**
   - Hour × Month interaction
   - Custom interactions as needed

**Feature Pipeline Composer:**

```python
class FeaturePipeline:
    def __init__(self):
        self.plugins: list[BaseFeaturePlugin] = []
    
    def add_plugin(self, plugin: BaseFeaturePlugin):
        self.plugins.append(plugin)
        return self
    
    def fit(self, df: pd.DataFrame, context: dict):
        """Fit all plugins (e.g., for scaling, encoding)."""
        for plugin in self.plugins:
            plugin.fit(df, context)
        return self
    
    def transform(self, df: pd.DataFrame, context: dict) -> pd.DataFrame:
        """Apply all plugins sequentially."""
        for plugin in self.plugins:
            df = plugin.compute(df, context)
        return df
    
    def get_all_feature_names(self, context: dict) -> list[str]:
        """Get complete feature list from all plugins."""
        names = []
        for plugin in self.plugins:
            names.extend(plugin.get_feature_names(context))
        return names
```

---

### 5.3 Critical Implementation Considerations

#### **Wide vs Long Format**
- R implementation uses **wide format** for training (one row per forecast origin)
- Python should support both:
  - Long format for data loading and exploration
  - Wide format for model training
  - Efficient pivot operations (pandas `pivot_table` or polars)

#### **Feature Selection Logic**
Must be **model-aware**:
```python
def select_features_for_lead_hour(
    lead_hour: int,
    all_features: list[str],
    max_horizon_days: int = 9
) -> list[str]:
    """Select only relevant features for a specific lead hour."""
    day_ahead = (lead_hour - 1) // 24 + 1
    hour_of_day = (lead_hour - 1) % 24
    
    selected = []
    
    # Always include origin-level features
    selected.extend([f for f in all_features if f.startswith('origin_')])
    selected.extend([f for f in all_features if f.startswith('prev_day_')])
    selected.extend([f for f in all_features if f.startswith('prev_2day_')])
    selected.extend([f for f in all_features if f.startswith('prev_7day_')])
    selected.extend([f for f in all_features if 'holiday' in f.lower()])
    
    # Temperature: only up to target day
    for d in range(1, day_ahead + 1):
        selected.extend([f for f in all_features if f in [f'tmax_{d}', f'tmin_{d}']])
    
    # Day-specific: only target day
    selected.extend([f for f in all_features if f.startswith(f'day{day_ahead}_')])
    
    # Hour-specific: only target lead hour
    h_prefix = f'h{lead_hour:03d}_'
    selected.extend([f for f in all_features if f.startswith(h_prefix)])
    
    return selected
```

#### **Performance Optimization**
- **Caching:** Pre-compute features for all forecast origins, save to parquet
- **Parallelization:** Compute features for different subsystems in parallel
- **Incremental updates:** Only compute features for new forecast origins

#### **Validation**
- **Schema validation:** Use Pydantic at data loading
- **Feature validation:** Check for NaN, Inf, outliers after feature computation
- **Consistency checks:** Ensure temp_max > temp_min, day-of-week in 1-7, etc.

---

### 5.4 Differences from LGBM Approach

The Random Forest implementation provides insights for Epic-02, but note key differences:

| Aspect | Random Forest (R) | LGBM (Epic-03 Target) |
|--------|-------------------|----------------------|
| **Architecture** | 216 independent models | Single multi-output model or separate models |
| **Feature selection** | Dynamic per lead hour | Potentially use all features |
| **Training approach** | Sequential per model | Batch training with custom objective |
| **Intraday updates** | Not supported | BLF strategy for D+0 |
| **Forecast horizon** | D+0 to D+8 (216 hours) | D+0 to D+1 (48 hours initially) |

**Implication:** Epic-02 must design features flexible enough for both approaches:
- Features computed at dataOrigem level (wide format)
- Support for dynamic feature selection (RF approach)
- Support for using all features (LGBM approach with built-in selection)

---

## Summary: Epic-02 Deliverables

Based on this analysis, Epic-02 should deliver:

### Core Components

1. **BaseFeaturePlugin interface** with standardized methods
2. **8+ feature plugins:**
   - CalendarFeaturePlugin (origin + target days)
   - TemperatureFeaturePlugin (daily max/min)
   - CyclicalTimePlugin (hour sin/cos, period flags)
   - HolidayProximityPlugin (proximity metrics)
   - LagFeaturePlugin (prev day, 2-day, 7-day stats)
   - SeasonalFeaturePlugin (month, quarter, season)
   - SpecialPeriodPlugin (Christmas, New Year)
   - InteractionFeaturePlugin (hour×month)

3. **Feature pipeline composer** for chaining plugins
4. **Feature selection utilities:**
   - `select_features_for_lead_hour()` (RF-style)
   - `get_all_features()` (LGBM-style)
   - Feature importance tracking

5. **Wide↔Long transformation utilities**
6. **Feature validation and testing framework**

### Success Criteria

✅ Plugin system is extensible (new plugins can be added easily)  
✅ Pipeline composer combines plugins without conflicts  
✅ Feature selection logic prevents data leakage  
✅ Wide/long transformations are memory-efficient  
✅ Features validated for NaN, Inf, outliers  
✅ Performance benchmarks: <5 minutes for 1 year of data (single subsystem)  
✅ Documentation includes feature engineering rationale  
✅ 80%+ unit test coverage

---

## Appendix: Complete Feature List

### Origin-Level Features (47 features)

```python
# Calendar (4)
origin_dow, origin_month, origin_is_weekend, origin_is_feriado

# Holiday proximity (3)
has_holiday_in_horizon, days_to_next_holiday, days_since_last_holiday

# Special periods (2)
origin_is_christmas, origin_is_newyear

# Previous day (8)
prev_day_mean, prev_day_peak, prev_day_min, prev_day_daily_peak, 
prev_day_evening_peak, prev_day_dow, prev_day_is_weekend, prev_day_is_feriado

# Previous 2-day (8)
prev_2day_mean, prev_2day_peak, prev_2day_min, prev_2day_daily_peak,
prev_2day_evening_peak, prev_2day_dow, prev_2day_is_weekend, prev_2day_is_feriado

# Recent load level (3)
prev_7day_mean, prev_3day_mean, recent_vs_historical
```

### Day-Level Features (per day d in 1-9)

```python
# Temperature (2 per day = 18 total)
tmax_d, tmin_d

# Day-of-week (11 per day = 99 total)
day{d}_dow, day{d}_is_weekend, day{d}_is_monday, day{d}_is_tuesday,
day{d}_is_wednesday, day{d}_is_thursday, day{d}_is_friday,
day{d}_is_saturday, day{d}_is_sunday, day{d}_is_feriado

# Seasonal (4 per day = 36 total)
day{d}_month, day{d}_quarter, day{d}_is_summer, day{d}_is_winter

# Special periods (2 per day = 18 total)
day{d}_is_christmas, day{d}_is_newyear
```

### Hour-Level Features (per lead hour h in 1-216)

```python
# Hour-specific (6 per hour = 1,296 total)
h{h:03d}_is_daily_peak, h{h:03d}_is_evening_peak, h{h:03d}_is_minimum,
h{h:03d}_hour_sin, h{h:03d}_hour_cos, h{h:03d}_hour_month_interaction
```

### Total Feature Count

- Origin-level: 47
- Day-level: 171 (19 × 9)
- Hour-level: 1,296 (6 × 216)
- **Grand Total: ~1,514 features**

**Average per model:** 50-100 features (after horizon-aware selection)

---

## References

- **Source files:**
  - `0_config.R`: Configuration, holidays, training periods
  - `1_funcoes_obtencao_dados.R`: Data loading, temperature adjustment
  - `2_funcoes_dataset_modelo.R`: Feature engineering, train/test split
  - `3_treinamento_teste_modelo.R`: Model training with feature selection
  - `4_avalia_performance_modelo.R`: Evaluation metrics
  - `exec_script.R`: Orchestration workflow
  - `cria_salva_dados.R`: Data export utility

- **Related epics:**
  - Epic-01: Data Infrastructure Layer (informed by sections 2.1-2.3)
  - Epic-02: Feature Engineering System (informed by sections 3.1-3.4)
  - Epic-03: Model Layer - End-to-End Models (RF vs LGBM comparison in 5.4)

---

**Document Status:** Complete  
**Next Steps:** Use this analysis to detail user stories in Epic-01 and Epic-02  
**Review:** Validate feature specifications with domain experts before implementation
