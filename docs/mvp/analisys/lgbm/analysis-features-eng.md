# Feature Engineering Analysis - PrevCarga LGBM Model

**Document:** Analysis of Feature Engineering Implementation  
**Source Files:** `1_Executar.R`, `ANN_PVDS.R`, `funcoes_aux.R`, `ModeloBLF_lgbm_v4.R`  
**Purpose:** Inform Epic-01 (Data Infrastructure) and Epic-02 (Feature Engineering System)  
**Date:** November 17, 2025

---

## Executive Summary

The current R implementation demonstrates a sophisticated feature engineering pipeline for electric load forecasting using LGBM. The system processes multiple data sources (load, temperature, heat index, holidays) and generates ~100+ features through various transformation techniques. This analysis identifies the core patterns, transformations, and architectural decisions that must be replicated in the Python unified system.

---

## 1. Data Infrastructure Layer (Epic-01 Foundation)

### 1.1 Data Sources and Loading

**Primary Data Streams:**

| Data Source | Frequency | Source System | Key Variables | Missing Value Strategy |
|-------------|-----------|---------------|---------------|------------------------|
| Load (Carga) | Semi-hourly (30min) | `LoadServices::CGPeriodo()` | `val_cargaglobalcons`, `val_cargammgd` | N/A (assumed complete) |
| Temperature Forecast D+1 | Hourly → Semi-hourly | `LoadServices::TemperaturaPrevistaDelay()` (delay=0) | `ValItemserieoriginal` | Lag fill (24/48h) + interpolation |
| Temperature Forecast D+2 | Hourly → Semi-hourly | `LoadServices::TemperaturaPrevistaDelay()` (delay=1) | `ValItemserieoriginal` | Lag fill (24/48h) + interpolation |
| ECMWF Temperature | Semi-hourly | CSV file (`heatindex_{area}.csv`) | `Temperatura_ECMWF` | Forward/backward fill + interpolation |
| Heat Index | Semi-hourly | CSV file (`heatindex_{area}.csv`) | `Heatindex` | Forward/backward fill + interpolation |
| Holidays | Daily | `LoadServices::GetDiasEspeciaisAssociados()` | `id_tipodiaespecial`, `dat_diaespecial` | Default to 0 (no holiday) |

**Critical Implementation Details:**

1. **Timezone Handling:**
   - All timestamps converted to `America/Sao_Paulo` timezone
   - Carga timestamps adjusted backward by 30 minutes: `DataHora %m-% minutes(30)`
   - This ensures alignment with semi-hourly intervals starting at 00:30

2. **Resampling Strategy:**
   - Temperature data arrives hourly, must be resampled to semi-hourly
   - Uses data.table complete sequence: `seq(min, max, by = "1 hour")`
   - Missing values filled with same-hour previous day: `shift(n = 24, type = "lag")`

3. **Missing Value Imputation Pattern (Triple Pass):**
   ```r
   # Pass 1: Forward fill within day
   Total[, Value_prev := nafill(Value, type = "locf"), by = Data]
   
   # Pass 2: Backward fill within day
   Total[, Value_next := nafill(Value, type = "nocb"), by = Data]
   
   # Pass 3: Interpolate with average
   Total[is.na(Value) & !is.na(Value_prev) & !is.na(Value_next),
         Value := (Value_prev + Value_next) / 2, by = Data]
   
   # Pass 4: Final forward fill
   Total[, Value := nafill(Value, type = "locf"), by = Data]
   ```

4. **Data Validation:**
   - Filters to complete days only (48 semi-hourly records per day)
   - Removes rows with NA in critical lag features
   - Ensures chronological ordering: `setorder(Total, Data, Hora, Minuto)`

**Python Migration Requirements (Epic-01):**

- [ ] S3 Parquet loaders for all data sources
- [ ] Pydantic schemas with timezone-aware datetime validation
- [ ] Resampling utilities: hourly → semi-hourly with configurable strategies
- [ ] Triple-pass imputation logic as a composable function
- [ ] Complete-day filter utility
- [ ] Data catalog to track dataset versions and lineage

---

## 2. Feature Engineering System (Epic-02 Foundation)

### 2.1 Feature Categories and Transformations

The implementation generates **8 major feature categories**, totaling approximately **100-150 features** depending on the time period:

---

#### **Category 1: Temporal Features (Base)**

**Purpose:** Encode cyclical patterns and calendar information

| Feature | Type | Encoding Method | Rationale |
|---------|------|-----------------|-----------|
| `Hora` | Continuous | Hour + Minute/60 | Decimal time representation |
| `Hora_seno` | Continuous | sin(2π × Hora/24) | Cyclical hour encoding |
| `Hora_cosseno` | Continuous | cos(2π × Hora/24) | Cyclical hour encoding |
| `Minuto` | Integer | 0 or 30 | Semi-hourly indicator |
| `DiaAno` | Integer | 1-366 | Day of year |
| `SemanaAno` | Integer | 1-53 | Week of year |
| `Mes` | Factor | 1-12 | Month categorical |
| `Ano` | Integer | Year | Trend capture |
| `DiaSemana` | Text | "segunda-feira", etc. | Day name |
| `DiaSemanaNum` | Factor | 1-7 | Day numeric (custom order: Sun=1, Sat=2, Mon=3-Fri=7) |
| `FimDeSemana` | Binary | 0/1 | Weekend indicator |

**Implementation Note:** Custom day ordering prioritizes weekends (Sunday=1, Saturday=2) before weekdays.

**Python Requirements:**
- Datetime decomposition utilities
- Cyclical encoding functions (sin/cos transformations)
- Custom categorical ordering support

---

#### **Category 2: Holiday and Special Days Features**

**Purpose:** Capture behavioral changes during holidays and adjacent days

**2.1 National Holidays (Binary Encoding):**
```python
# Hardcoded national holidays list (2021-2025)
national_holidays = [
    "01-01",  # New Year
    "04-21",  # Tiradentes
    "05-01",  # Labor Day
    "09-07",  # Independence Day
    "10-12",  # Nossa Senhora Aparecida
    "11-02",  # All Souls' Day
    "11-15",  # Proclamation of Republic
    "11-20",  # Black Consciousness Day (2024+)
    "12-25",  # Christmas
    # + Movable holidays (Easter, Carnival, Corpus Christi)
]
```

**Derived Features:**
- `Feriado`: 1 if holiday, 0 otherwise (original `id_tipodiaespecial >= 1` mapped to 1)
- `PreFeriado`: 1 if day before holiday
- `PosFeriado`: 1 if day after holiday

**2.2 Special Days (Christmas/New Year Period):**

| Feature | Date Range | Label |
|---------|-----------|-------|
| `DiaEspecial_Natal_AnoNovo` | 12-25, 01-01 | Actual holiday |
| `DiaEspecial_Vespera` | 12-24, 12-31 | Eve of holiday |
| `DiaEspecial_PreVespera` | 12-23, 12-30 | Two days before |
| `DiaEspecial_PosNatalAnoNovo` | 12-26, 01-02 | Day after |
| `DiaEspecial_SemanaNatalAnoNovo` | 12-27 to 12-30 | Week between |
| `DiaEspecial_PrimeiraSemanaAno` | 01-03 to 01-07 | First week of year |

**All converted to binary dummy variables** (one-hot encoding).

**Python Requirements:**
- Holiday calendar management (regional + national)
- Automatic dummy variable generation for multi-class categoricals
- Date range matching utilities

---

#### **Category 3: Dummy Variables (One-Hot Encoding)**

**Purpose:** Categorical encoding for non-ordinal variables

**3.1 Day of Week Dummies:**
```python
# Creates columns: DiaSemana_domingo, DiaSemana_terça-feira, ..., DiaSemana_sábado
# Reference category: segunda-feira (excluded)
dias_dummy = [col for col in columns if col.startswith("DiaSemana_")]
# Count: 6 columns (7 days - 1 reference)
```

**3.2 Month Dummies:**
```python
# Creates columns: Mes_1, Mes_3, Mes_4, ..., Mes_12
# Reference category: Mes_2 (February - excluded)
mes_dummy = [col for col in columns if col.startswith("Mes_")]
# Count: 11 columns (12 months - 1 reference)
```

**3.3 Special Days Dummies:**
```python
# Creates columns: DiaEspecial_Natal_AnoNovo, DiaEspecial_Vespera, ...
# Count: 6 columns (see Category 2.2)
dias_especiais_dummy = [col for col in columns if col.startswith("DiaEspecial_")]
```

**Total Dummy Variables:** ~23 columns (6 + 11 + 6)

**Python Requirements:**
- Automatic one-hot encoding with reference category exclusion
- Dynamic column name generation
- Factor-to-integer conversion utilities

---

#### **Category 4: LOESS Smoothing Features**

**Purpose:** Noise reduction and trend extraction using local polynomial regression

**Algorithm:** LOESS (Locally Estimated Scatterplot Smoothing)
- **Implementation:** `loess(y ~ x, span = 0.2-0.3, degree = 2)`
- **Span:** 0.2-0.3 (20-30% of data points in local window)
- **Degree:** 2 (quadratic polynomial)
- **Grouping:** Applied per day (`by = Data`)

**Smoothed Variables:**

| Original Feature | Smoothed Feature | Span | Purpose |
|------------------|------------------|------|---------|
| `TemperaturaD1` | `TemperaturaD1amort` | 0.3 | Remove sensor noise |
| `TemperaturaD2` | `TemperaturaD2amort` | 0.3 | Remove sensor noise |
| `CargaGlobal` | `CargaGlobalamort` | 0.3 | Remove short-term spikes |
| `Temperatura_ECMWF` | `Temperatura_ECMWFamort` | 0.3 | Remove forecast artifacts |

**Implementation Pattern:**
```r
Total[, TemperaturaD1amort := predict(
  loess(TemperaturaD1 ~ HoraDecimal, span = 0.3, degree = 2)
), by = Data]
```

**HoraDecimal:** Continuous time variable = `Hora + Minuto / 60`

**Python Requirements:**
- Statsmodels LOWESS implementation or scipy.signal.savgol_filter
- Groupby apply pattern for per-day smoothing
- Configurable span/degree parameters
- Plugin interface: `LoessSmoothingPlugin(span=0.3, degree=2)`

---

#### **Category 5: Wavelet Transform Features**

**Purpose:** Multi-scale time series decomposition to capture patterns at different frequencies

**Algorithm:** Discrete Wavelet Transform (DWT)
- **Wavelet Filter:** Haar (default, simplest)
- **Window Size:** 32 samples (2^5 semi-hourly intervals = 16 hours)
- **Rolling Application:** Right-aligned with NA padding

**Generated Features:**

1. **Wavelet Coefficients (W):** Detail coefficients at each decomposition level
2. **Scaling Coefficients (V):** Approximation coefficients

**Implementation (from `funcoes_aux.R`):**

```r
build_wavelets <- function(ts, window = 2^5, filter = "haar") {
  # Rolling window application
  out <- zoo::rollapply(ts, width = window, 
                        build_wavelets_single_window,
                        na.pad = TRUE, align = "right")
  return(out)
}

build_wavelets_single_window <- function(v, filter = "haar") {
  wt <- dwt(v, filter = filter)          # Perform DWT
  ws <- w2dt(wt)                         # Extract detail coefficients
  vs <- v2dt(wt)                         # Extract approximation coefficients
  wavs <- cbind(ws, vs)                  # Combine
  return(wavs)
}
```

**Applied To:**
1. **Load Wavelets:** `CargaGlobal_anterior_BLF` (previous day load at same hour)
   - Column names: `CarWav_W1_1`, `CarWav_W1_2`, ..., `CarWav_V5_1`, ...
   - Temporal shift: +48 hours (aligned to prediction time)

2. **Temperature Wavelets:** `Temperatura_prevista_BLF` (forecast temperature D+1)
   - Column names: `TempWav_W1_1`, `TempWav_W1_2`, ..., `TempWav_V5_1`, ...

**Typical Output Dimensions:**
- 5 decomposition levels (W1-W5, V1-V5)
- ~16-32 columns per variable
- **Total:** ~32-64 wavelet features

**Python Requirements:**
- PyWavelets library (`pywt.dwt`)
- Rolling window utility with configurable alignment
- Automatic column naming convention
- Plugin interface: `WaveletTransformPlugin(window=32, wavelet='haar')`

---

#### **Category 6: Lag Features (BLF Strategy)**

**Purpose:** Incorporate previous day's information at same hour (day-ahead persistence)

**BLF Methodology (Best Linear Forecast):**
- Core assumption: Load at time `t` is related to load at time `t - 24h` (same hour previous day)
- Implementation: Self-join with 1-day offset

**Lag Features Created:**

| Feature | Source | Lag Period | Purpose |
|---------|--------|-----------|---------|
| `CargaGlobal_anterior_BLF` | `CargaGlobalamort` | +1 day | Previous day load (smoothed) |
| `TemperaturaECMWF_anterior_BLF` | `Temperatura_ECMWFamort` | +1 day | Previous day ECMWF temp (smoothed) |
| `Heatindex_anterior_BLF` | `Heatindex` | +1 day | Previous day heat index |

**Implementation Pattern:**
```r
# 1. Create lag table with shifted date
lag_BLF <- Total[, .(
  Data_lag_BLF = Data + days(+1),    # Shift forward 1 day
  Hora,
  CargaGlobal_lag_BLF = CargaGlobalamort
)]

# 2. Self-join to align current row with previous day's value
Total_BLF <- merge(
  Total,
  lag_BLF,
  by.x = c("Data", "Hora"),
  by.y = c("Data_lag_BLF", "Hora"),
  all.x = TRUE
)
```

**Key Insight:** The "lag" is actually a forward shift in the lookup table, creating a same-hour previous-day alignment.

**Python Requirements:**
- Efficient time-based join utilities (pandas merge_asof or polars join)
- Configurable lag periods (24h, 48h, 168h for weekly)
- Plugin interface: `LagFeaturePlugin(lags=[24, 48, 168], units='hours')`

---

#### **Category 7: Seasonality Feature (Intraday Pattern)**

**Purpose:** Capture typical load profile by day type, accounting for holidays and special days

**Methodology:**
1. **Group By:** `TipoDia + Hora + Minuto + Feriado + DiaEspecial`
2. **Aggregate:** Mean load across historical data
3. **Smooth:** LOESS with span=0.2, degree=2
4. **Join Back:** Merge to original dataset

**TipoDia Classification:**
```r
TipoDia := fcase(
  DiaSemana == "segunda-feira", "segunda-feira",
  DiaSemana == "sexta-feira",   "sexta-feira",
  DiaSemana == "sábado",        "sábado",
  DiaSemana == "domingo",       "domingo",
  default = "Dia útil intermediário"  # Tue-Thu
)
```

**Implementation:**
```r
# 1. Calculate mean load profile
sazonalidade_diaria_mes <- Total[, .(
  CargaGlobal_media = mean(CargaGlobal, na.rm = TRUE)
), by = .(TipoDia, Hora, Minuto, Feriado, DiaEspecial)]

# 2. Smooth profile
sazonalidade_diaria_mes[, Sazonalidade := predict(
  loess(CargaGlobal_media ~ Hora, span = 0.2, degree = 2)
), by = .(TipoDia, Feriado, DiaEspecial)]

# 3. Merge back
Total <- merge(Total, sazonalidade_diaria_mes[, .(TipoDia, Hora, Minuto, Feriado, DiaEspecial, Sazonalidade)],
               by = c("TipoDia", "Hora", "Minuto", "Feriado", "DiaEspecial"), all.x = TRUE)
```

**Result:** Single feature `Sazonalidade` representing expected load pattern

**Python Requirements:**
- Groupby aggregation with multiple keys
- LOESS smoothing within groups
- Merge with multi-key join
- Plugin interface: `SeasonalityPlugin(groupby_cols=[...], span=0.2)`

---

#### **Category 8: Exogenous Variables**

**Purpose:** External factors influencing load

| Feature | Description | Type | Source |
|---------|-------------|------|--------|
| `GerMMGD` | Distributed generation (small hydro/solar) | Continuous | LoadServices API |
| `Temperatura_prevista_BLF` | D+1 temperature forecast (smoothed) | Continuous | TemperaturaD1amort |

**Note:** Heat index considered but commented out in final model (`Heatindex_anterior_BLF` excluded from input features).

---

### 2.2 Feature Pipeline Architecture

**Execution Flow:**

```
Raw Data Loading (ANN_PVDS.R)
    ↓
Timezone Alignment & Resampling
    ↓
Missing Value Imputation (Triple Pass)
    ↓
Temporal Feature Extraction
    ↓
Holiday/Special Day Encoding → Dummy Variables
    ↓
LOESS Smoothing (per day)
    ↓
Seasonality Calculation
    ↓
BLF Lag Feature Creation (ModeloBLF_lgbm_v4.R)
    ↓
Wavelet Transform (32-sample rolling window)
    ↓
Feature Selection & Normalization
    ↓
Model Training/Prediction
```

**Critical Dependencies:**
- LOESS smoothing must occur **before** lag feature creation
- Wavelet transform applied **after** lag features created
- Dummy variables created **before** normalization
- Seasonality calculated **after** special day encoding

---

### 2.3 Normalization Strategy

**Min-Max Normalization:**
```python
normalized = (x - min) / (max - min)
denormalized = x * (max - min) + min
```

**Excluded from Normalization (Categorical Features):**
- `Feriado`, `PreFeriado`, `PosFeriado`, `Mes`
- All dummy variables (`dias_dummy`, `mes_dummy`, `dias_especiais_dummy`)

**Normalized Features:**
- All continuous variables (load, temperature, wavelets, lags, etc.)
- Target variable: `CargaGlobal_real_BLF`

**Normalization Scope:**
- Fit on training set only
- Store min/max values for each feature
- Apply same transformation to validation/test sets
- Denormalize predictions before evaluation

**Python Requirements:**
- Scikit-learn `MinMaxScaler` with feature-wise fit
- Exclude categorical features from scaling
- Serialize scaler parameters with model
- Plugin interface: `NormalizationPlugin(strategy='minmax', exclude=[...])`

---

### 2.4 Feature Selection

**Final Input Features (ModeloBLF_lgbm_v4.R):**

```python
input_features = [
    # Lag features (3)
    "CargaGlobal_anterior_BLF",
    "TemperaturaECMWF_anterior_BLF",
    
    # Exogenous (2)
    "Temperatura_prevista_BLF",
    "GerMMGD",
    
    # Temporal cyclical (1)
    "Hora_seno",
    
    # Holiday indicators (3)
    "Feriado",
    "PreFeriado", 
    "PosFeriado",
    
    # Seasonality (1)
    "Sazonalidade",
    
    # Wavelet transforms (~32-64)
    carga_wav,      # CarWav_W1_1, CarWav_W1_2, ..., CarWav_V5_X
    temp_wav,       # TempWav_W1_1, TempWav_W1_2, ..., TempWav_V5_X
    
    # Dummy variables (~23)
    dias_dummy,             # DiaSemana_domingo, DiaSemana_terça-feira, ...
    dias_especiais_dummy,   # DiaEspecial_Natal_AnoNovo, ...
    mes_dummy               # Mes_1, Mes_3, ..., Mes_12
]
```

**Total Feature Count:** ~75-105 features (depending on wavelet decomposition depth)

**Notable Exclusions:**
- `Hora_cosseno` (sine alone sufficient)
- `Heatindex_anterior_BLF` (redundant with temperature)
- `TemperaturaD2` (D+1 more reliable)

---

## 3. Model-Specific Considerations

### 3.1 LGBM Configuration

**Categorical Feature Handling:**
```r
parametros_BLF <- list(
  categorical_features = match(coluna_categorica_BLF, colnames(entradas_treino_BLF)),
  feature_pre_filter = FALSE
)
```

**Categorical Features List:**
- `Feriado`, `PreFeriado`, `PosFeriado`, `Mes`
- All dummy variables

**Important:** LGBM handles categorical features natively; no need for one-hot encoding internally, but current implementation still uses dummies for interpretability.

---

### 3.2 Custom Loss Function (ModeloBLF_lgbm_v4.R)

**Weighted MAPE with Underestimation Penalty:**

```r
objetivo_customizado <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  weights <- getinfo(dtrain, "weight")  # Higher weight for hours > 18
  
  error <- preds - labels
  
  # Penalize underestimation more heavily
  grad <- ifelse(error < 0, 
                 -weights * (1 + penalizacao) / labels,  # Underestimation
                 -weights / labels)                      # Overestimation
  
  hess <- weights / labels^2
  return(list(grad = grad, hess = hess))
}
```

**Weight Vector:**
- `fator_penalidade = 1` for hours ≤ 18
- Higher penalty for hours > 18 (peak demand period)

**Python Requirements:**
- Custom LGBM objective function implementation
- Weight vector passed to `lgb.Dataset(weight=...)`
- Configurable penalty factors

---

### 3.3 Post-Prediction LOESS Smoothing

**Purpose:** Remove short-term prediction noise while preserving daily trends

**Applied After Prediction:**
```r
AvaliacaoFinal[, CargaGlobalSuavizada := {
  ajuste <- loess(Previsto_denorm ~ HoraDecimal, span = 0.3, degree = 2)
  ajuste$fitted
}, by = Data]
```

**Final Output:** `CargaGlobalSuavizada` replaces `Previsto_denorm`

**Python Requirements:**
- Post-processing pipeline step
- Per-day groupby apply
- Optional toggle (may not be needed for all models)

---

## 4. Epic-02 Specifications from Analysis

### 4.1 Required Feature Plugins

Based on the R implementation, Epic-02 must implement these plugins:

| Plugin Class | Priority | Input | Output | Configuration |
|-------------|----------|-------|--------|---------------|
| `TemporalFeaturesPlugin` | Critical | datetime | Hora, Mes, DiaAno, Hora_seno, Hora_cosseno, etc. | cyclical_encoding=True |
| `HolidayFeaturesPlugin` | Critical | dates, holiday_calendar | Feriado, PreFeriado, PosFeriado | region="BR" |
| `SpecialDaysPlugin` | High | dates | DiaEspecial_* dummies | patterns=["christmas", "newyear"] |
| `DummyEncoderPlugin` | Critical | categorical columns | One-hot encoded columns | drop_first=True |
| `LoessSmoothingPlugin` | High | continuous time series | Smoothed variables | span=0.3, degree=2, groupby="Data" |
| `WaveletTransformPlugin` | High | time series | W1-W5, V1-V5 coefficients | window=32, wavelet="haar", align="right" |
| `LagFeaturesPlugin` | Critical | time series + datetime | Lagged variables | lags=[24], units="hours", groupby="Hora" |
| `SeasonalityPlugin` | High | load + grouping keys | Sazonalidade | groupby=["TipoDia","Hora","Minuto","Feriado"] |
| `NormalizationPlugin` | Critical | continuous features | Normalized features | strategy="minmax", exclude=["Feriado",...] |

### 4.2 Plugin Composition Pattern

**Sequential Pipeline:**
```python
from prevcarga.features import FeaturePipeline

pipeline = FeaturePipeline([
    TemporalFeaturesPlugin(cyclical=True),
    HolidayFeaturesPlugin(region="BR"),
    SpecialDaysPlugin(patterns=["christmas_week"]),
    DummyEncoderPlugin(drop_first=True, columns=["DiaSemana", "Mes"]),
    LoessSmoothingPlugin(
        columns=["CargaGlobal", "TemperaturaD1"],
        span=0.3,
        degree=2,
        groupby="Data"
    ),
    SeasonalityPlugin(
        target="CargaGlobal",
        groupby=["TipoDia", "Hora", "Minuto", "Feriado"],
        smooth=True
    ),
    LagFeaturesPlugin(
        columns=["CargaGlobal", "Temperatura_ECMWF"],
        lags=[24],
        units="hours"
    ),
    WaveletTransformPlugin(
        columns=["CargaGlobal_lag24", "Temperatura_prevista"],
        window=32,
        wavelet="haar"
    ),
    NormalizationPlugin(
        strategy="minmax",
        exclude_categorical=True
    )
])

features = pipeline.transform(data)
```

### 4.3 Feature Storage and Versioning

**Requirements:**
- Store intermediate features at each plugin stage (for debugging)
- Serialize pipeline configuration as YAML
- Version feature sets with semantic versioning
- Track feature importance after model training
- Cache expensive computations (wavelets, LOESS)

**Proposed Schema:**
```yaml
feature_version: "1.0.0"
created_at: "2025-11-17T10:30:00"
area: "SECO"
plugins:
  - name: TemporalFeaturesPlugin
    version: "1.0.0"
    params:
      cyclical: true
  - name: WaveletTransformPlugin
    version: "1.0.0"
    params:
      window: 32
      wavelet: "haar"
features_generated: 105
feature_names:
  - CargaGlobal_anterior_BLF
  - CarWav_W1_1
  - ...
```

---

## 5. Performance Considerations

### 5.1 Computational Bottlenecks (from R implementation)

| Operation | Estimated Time | Parallelizable | Optimization Strategy |
|-----------|---------------|----------------|----------------------|
| LOESS smoothing (per day) | ~5-10s per variable | Yes (by day) | Vectorize with scipy.signal.savgol_filter |
| Wavelet transform (rolling) | ~10-15s per variable | Yes (by chunks) | Use PyWavelets C backend |
| Seasonality calculation | ~2-3s | Yes (by group) | Use polars for groupby aggregations |
| LGBM training (per area) | ~10-30 min | No (but multiple areas parallel) | GPU acceleration |
| Holiday/dummy encoding | <1s | N/A | Negligible |

**Target Performance (Epic-02):**
- Feature generation for 1 area, 1 year: **< 2 minutes**
- Full 26 areas: **< 30 minutes** (with parallelization)

### 5.2 Memory Footprint

**Training Set Size (from code analysis):**
- Timespan: ~2 years (2022-10-01 to 2024-09-30)
- Frequency: Semi-hourly (48 rows/day)
- Total rows: ~35,000 rows
- Features: ~105 columns
- Memory: ~30 MB per area (uncompressed)

**Recommendation:** Use memory-efficient data structures (polars DataFrames) and chunked processing for backtesting.

---

## 6. Data Quality and Validation

### 6.1 Critical Data Quality Checks (from R)

1. **Complete Days Filter:**
   ```python
   day_counts = df.groupby("Data").size()
   complete_days = day_counts[day_counts == 48].index
   df_clean = df[df["Data"].isin(complete_days)]
   ```

2. **NA Handling:**
   - No NAs allowed in lag features after creation
   - Remove rows with NA in `CargaGlobal_anterior_BLF`

3. **Chronological Ordering:**
   - Always sort by `Data`, `Hora`, `Minuto` before feature engineering

4. **Timezone Consistency:**
   - All timestamps must be in `America/Sao_Paulo` timezone

### 6.2 Feature Validation (Epic-02 Requirement)

**Automated Tests:**
- [ ] Check feature distribution (no constant columns)
- [ ] Verify no data leakage (future information in lags)
- [ ] Confirm normalization range [0, 1] for continuous features
- [ ] Validate wavelet decomposition (no NaNs except expected padding)
- [ ] Check dummy variable sum (should equal 1 for one-hot groups)

---

## 7. Key Insights for Python Migration

### 7.1 Critical Success Factors

1. **LOESS Smoothing is Central:**
   - Applied to 4 key variables before all other transformations
   - Reduces noise and improves model stability
   - Must preserve per-day grouping

2. **BLF Lag Strategy is Non-Trivial:**
   - Not a simple pandas `.shift()` operation
   - Requires date-hour alignment with forward date shift
   - Smoothed variables used for lags, not raw data

3. **Wavelet Transform is Expensive but Valuable:**
   - 32-64 features generated per variable
   - Rolling window with right alignment
   - Consider caching for production

4. **Seasonality Feature Requires Historical Data:**
   - Cannot be computed for new areas without history
   - Needs at least 1 month of data per TipoDia/Feriado combination
   - Must be recalculated periodically (monthly?)

5. **Dummy Variables Must Exclude Reference Category:**
   - Prevents multicollinearity
   - LGBM handles this natively, but explicit encoding aids interpretability

### 7.2 Recommended Technology Stack (Epic-02)

- **Data Processing:** Polars (10x faster than pandas for groupby operations)
- **Smoothing:** Statsmodels LOWESS or scipy.signal.savgol_filter
- **Wavelets:** PyWavelets (C-optimized)
- **Normalization:** Scikit-learn StandardScaler/MinMaxScaler
- **Temporal Features:** Custom utilities + pandas datetime accessors
- **Holiday Calendars:** Workalendar (Brazilian holidays built-in)

### 7.3 Testing Strategy

**Unit Tests:**
- Each plugin independently tested with synthetic data
- Verify output shape, type, and range
- Test edge cases (missing data, single day, etc.)

**Integration Tests:**
- Full pipeline run on small dataset (1 month, 1 area)
- Compare Python output with R output (row-by-row validation)
- Tolerance: ±1e-6 for continuous features, exact match for categorical

**Regression Tests:**
- Freeze R output as "golden dataset"
- Run Python pipeline and compare final features
- Flag any deviations > 0.01% for investigation

---

## 8. Open Questions for Epic Refinement

1. **Seasonality Recalculation Frequency:**
   - R code calculates monthly (`sazonalidade_diaria_mes`)
   - Should this be updated daily, weekly, or monthly in production?
   - Impact on model drift?

2. **Wavelet Window Size Optimization:**
   - Current: 32 samples (16 hours)
   - Alternative: 48 samples (24 hours) for full daily cycle?
   - Trade-off: More features vs. more NAs from padding

3. **LOESS Span Parameter Tuning:**
   - Current: 0.2-0.3
   - Should this be hyperparameter optimized per area?
   - Or fixed globally for consistency?

4. **Heat Index Exclusion Rationale:**
   - Commented out in final model (`Heatindex_anterior_BLF`)
   - Why excluded? Multicollinearity with temperature?
   - Document decision for future reference

5. **Post-Prediction Smoothing:**
   - Is this necessary for all models or LGBM-specific?
   - Should Random Forest predictions also be smoothed?
   - Add as optional pipeline step?

---

## 9. Actionable Recommendations

### For Epic-01 (Data Infrastructure):

1. **Priority 1:** Implement triple-pass imputation logic as reusable utility
2. **Priority 1:** Build timezone-aware datetime handling with Pydantic
3. **Priority 2:** Create S3 Parquet loaders with schema validation
4. **Priority 2:** Implement complete-day filter and data quality checks
5. **Priority 3:** Add data catalog for lineage tracking

### For Epic-02 (Feature Engineering):

1. **Priority 1:** Implement plugin architecture with base class
2. **Priority 1:** Create core plugins: Temporal, LOESS, Lag, Normalization
3. **Priority 2:** Add Wavelet, Seasonality, Holiday, Dummy plugins
4. **Priority 2:** Build feature pipeline composer with YAML config
5. **Priority 3:** Add feature importance analysis utilities
6. **Priority 3:** Implement caching layer for expensive operations

### For Epic-03 (LGBM Model):

1. **Priority 1:** Port custom objective function (weighted MAPE)
2. **Priority 2:** Implement post-prediction LOESS smoothing
3. **Priority 3:** Add feature selection based on importance

---

## 10. Appendix: Feature Inventory

**Complete Feature List (Approximate - varies by date range):**

| Category | Count | Examples |
|----------|-------|----------|
| Temporal Base | 11 | Hora, Minuto, Mes, Ano, DiaAno, SemanaAno, DiaSemana, DiaSemanaNum, FimDeSemana, Hora_seno, Hora_cosseno |
| Holiday Indicators | 3 | Feriado, PreFeriado, PosFeriado |
| Day of Week Dummies | 6 | DiaSemana_domingo, DiaSemana_terça-feira, ... |
| Month Dummies | 11 | Mes_1, Mes_3, Mes_4, ..., Mes_12 |
| Special Day Dummies | 6 | DiaEspecial_Natal_AnoNovo, DiaEspecial_Vespera, ... |
| LOESS Smoothed | 4 | TemperaturaD1amort, TemperaturaD2amort, CargaGlobalamort, Temperatura_ECMWFamort |
| Lag Features | 3 | CargaGlobal_anterior_BLF, TemperaturaECMWF_anterior_BLF, Heatindex_anterior_BLF |
| Load Wavelets | 32 | CarWav_W1_1, CarWav_W1_2, ..., CarWav_V5_X |
| Temp Wavelets | 32 | TempWav_W1_1, TempWav_W1_2, ..., TempWav_V5_X |
| Seasonality | 1 | Sazonalidade |
| Exogenous | 2 | GerMMGD, Temperatura_prevista_BLF |
| **Total** | **~111** | (Varies by ±10 depending on date range and available months/days) |

---

**Document End**

*This analysis provides a comprehensive blueprint for Epic-01 and Epic-02 implementation. All code patterns, parameters, and transformation logic are derived directly from the R source files and reflect production-tested methods.*
