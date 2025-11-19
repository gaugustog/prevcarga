# Model Analysis - Random Forest Multi-Output Architecture

**Project:** PrevCarga - Electric Load Forecasting  
**Model:** Random Forest Multi-Output (216 independent models)  
**Date:** November 17, 2025  
**Purpose:** Inform Epic-03 (Model Layer - End-to-End Models)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Model Architecture](#model-architecture)
3. [Training Pipeline](#training-pipeline)
4. [Prediction Pipeline](#prediction-pipeline)
5. [Evaluation Framework](#evaluation-framework)
6. [Orchestration & Workflow](#orchestration--workflow)
7. [Key Insights for Epic-03 Implementation](#key-insights-for-epic-03-implementation)

---

## Executive Summary

### Model Overview

The Random Forest implementation uses a **multi-output architecture** where 216 independent models are trained, one for each hour in a 9-day forecast horizon (D+0 to D+8, 24 hours × 9 days).

**Key Characteristics:**
- **Architecture:** 216 independent Random Forest models (not a single multi-output model)
- **Horizon:** 216 hours (9 complete days)
- **Training data:** Rolling 13-month window
- **Test period:** 30 days per evaluation cycle
- **Feature selection:** Dynamic per lead hour (50-100 features per model)
- **Hyperparameters:** 100 trees per forest (default)

### Performance Context

The R implementation achieves competitive results against the baseline PrevCargaDESSEM system:
- Evaluation across 12 months (Oct 2024 - Sep 2025)
- Metrics: MAPE, MAE, RMSE, R² by day and hour
- Comparison shows improvements in certain horizons/periods

---

## Model Architecture

### 2.1 Multi-Output Strategy

The implementation uses **one model per lead hour** rather than a single multi-output model:

```r
# From 3_treinamento_teste_modelo.R
for (i in 1:length(target_cols)) {  # length(target_cols) = 216
  target_name <- target_cols[i]     # e.g., "y_h001", "y_h002", ...
  lead_hour <- as.numeric(gsub("y_h", "", target_name))
  
  # Train independent model for this specific lead_hour
  rf_model <- randomForest(
    x = X_train_clean,
    y = Y_train_clean,
    ntree = ntree,
    importance = TRUE,
    do.trace = FALSE
  )
  
  models[[target_name]] <- list(
    model = rf_model,
    features = selected_features,
    lead_hour = lead_hour,
    day_ahead = day_ahead,
    hour_of_day = hour_of_day,
    n_obs = nrow(X_train_clean)
  )
}
```

**Rationale for this approach:**
1. **Feature flexibility:** Each model can use different features based on horizon
2. **Computational parallelization:** Models can be trained independently
3. **Debugging simplicity:** Easy to isolate issues in specific horizons
4. **Memory efficiency:** Can train/predict in batches

**Trade-offs:**
- ✅ **Pros:** Flexible feature selection, parallel training, easy debugging
- ❌ **Cons:** No cross-horizon learning, 216× storage overhead, longer total training time (if sequential)

### 2.2 Model Structure per Lead Hour

Each of the 216 models is a standard Random Forest regressor:

```python
# Python equivalent structure
class RandomForestModel:
    """Represents one model in the 216-model ensemble."""
    
    def __init__(self, lead_hour: int):
        self.lead_hour = lead_hour
        self.day_ahead = (lead_hour - 1) // 24 + 1
        self.hour_of_day = (lead_hour - 1) % 24
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=123,
            n_jobs=-1
        )
        self.features: List[str] = []
        self.n_obs: int = 0
        self.trained_at: datetime = None
```

**Key metadata stored with each model:**
- `lead_hour`: Target hour (1-216)
- `day_ahead`: Target day (1-9)
- `hour_of_day`: Hour within day (0-23)
- `features`: List of feature names used
- `n_obs`: Number of training observations
- `model`: The actual RandomForest object

### 2.3 Hyperparameters

Current implementation uses **default/minimal hyperparameters**:

```r
rf_model <- randomForest(
  x = X_train_clean,
  y = Y_train_clean,
  ntree = 100,              # Number of trees
  importance = TRUE,        # Calculate feature importance
  do.trace = FALSE          # Suppress progress messages
)
```

**Default RandomForest parameters** (not explicitly set):
- `mtry`: sqrt(n_features) for regression
- `nodesize`: 5 (minimum size of terminal nodes)
- `maxnodes`: NULL (trees grown to maximum depth)
- `replace`: TRUE (bootstrap sampling with replacement)

**For Epic-03 implementation:**
- These hyperparameters should be **configurable** via YAML
- Consider adding: `min_samples_split`, `min_samples_leaf`, `max_depth`
- Grid search or Bayesian optimization for tuning

---

## Training Pipeline

### 3.1 Data Preparation

#### **Train/Test Split** (`split_train_test`)

```r
# From 2_funcoes_dataset_modelo.R
setorder(dt_features, dataOrigem)
dt_features <- na.omit(dt_features)

n_total <- nrow(dt_features)
n_train <- floor(n_total - 30)  # Last 30 days reserved for testing

train_data <- dt_features[1:n_train]
test_data  <- dt_features[(n_train + 1):n_total]
```

**Key characteristics:**
- **Temporal ordering:** Data sorted by `dataOrigem` (forecast origin date)
- **Fixed test size:** Last 30 days (not percentage-based)
- **No shuffling:** Preserves temporal structure
- **Missing values:** Removed with `na.omit()` (potentially aggressive)

**Training window:**
- 13 months of data total
- ~12 months for training (~360 forecast origins)
- 1 month for testing (30 forecast origins)

#### **Feature Selection per Model**

The most critical aspect is **dynamic feature selection** based on lead hour:

```r
# From 3_treinamento_teste_modelo.R
day_ahead = ceiling(lead_hour / 24)
hour_of_day = ((lead_hour - 1) %% 24)

# 1. Origin calendar features (always included)
origin_features <- grep("^origin_", feature_cols, value = TRUE)

# 2. Temperature features for TARGET DAY and ALL PRECEDING DAYS
temp_features <- grep(paste0("^(tmax|tmin)_[1-", day_ahead, "]$"), 
                      feature_cols, value = TRUE)

# 3. Day-of-week features for TARGET DAY only
dow_features <- grep(paste0("^day", day_ahead, "_"), feature_cols, value = TRUE)

# 4. Hour-specific features for THIS LEAD HOUR only
hour_features <- grep(paste0("^h", sprintf("%03d", lead_hour), "_"), 
                      feature_cols, value = TRUE)

# 5. Seasonal features for TARGET DAY only
seasonal_features <- grep(paste0("^day", day_ahead, "_(month|quarter|is_summer|is_winter)"), 
                          feature_cols, value = TRUE)

# 6-10. Always include historical load features
prev_day_features <- grep("^prev_day_", feature_cols, value = TRUE)
prev_2day_features <- grep("^prev_2day_", feature_cols, value = TRUE)
recent_level_features <- grep("^(prev_7day_mean|prev_3day_mean|recent_vs_historical)$", 
                              feature_cols, value = TRUE)
holiday_features <- grep("^(has_holiday_in_horizon|days_to_next_holiday|days_since_last_holiday)$", 
                         feature_cols, value = TRUE)

# Combine all relevant features
selected_features <- unique(c(
  origin_features,
  temp_features,
  dow_features,
  hour_features,
  seasonal_features,
  prev_day_features,
  prev_2day_features,
  recent_level_features,
  holiday_features
))
```

**Feature selection logic:**
- **Origin features:** Always included (4 features)
- **Temperature:** Only up to target day (e.g., day 3 uses tmax_1, tmin_1, tmax_2, tmin_2, tmax_3, tmin_3)
- **Day-of-week:** Only target day (e.g., day 3 uses only day3_* features)
- **Hour-specific:** Only target hour (e.g., h073_* for lead_hour 73)
- **Historical load:** Always included (19 features total)

**Result:** Average of 50-100 features per model (vs 1,400+ total features)

### 3.2 Training Process

#### **Sequential Training Loop**

```r
# From 3_treinamento_teste_modelo.R
models <- list()
feature_usage_summary <- data.frame(
  lead_hour = integer(),
  n_features = integer(),
  stringsAsFactors = FALSE
)

for (i in 1:length(target_cols)) {
  # Progress reporting every 24 hours (one day)
  if (i %% 24 == 1) {
    cat(sprintf("Treinando os modelos para o dia %d of 9 (lead hours %d-%d)...\n", 
                day_ahead, i, min(i+23, length(target_cols))))
  }
  
  # Select features for this lead hour
  selected_features <- ... # (see above)
  
  # Prepare training data
  X_train <- as.matrix(train_data[, selected_features, with = FALSE])
  Y_train <- train_data[[target_name]]
  
  # Remove incomplete cases
  complete_rows <- complete.cases(X_train) & !is.na(Y_train)
  X_train_clean <- X_train[complete_rows, , drop = FALSE]
  Y_train_clean <- Y_train[complete_rows]
  
  # Train model
  set.seed(123)  # Reproducibility
  rf_model <- randomForest(
    x = X_train_clean,
    y = Y_train_clean,
    ntree = 100,
    importance = TRUE,
    do.trace = FALSE
  )
  
  # Store model with metadata
  models[[target_name]] <- list(
    model = rf_model,
    features = selected_features,
    lead_hour = lead_hour,
    day_ahead = day_ahead,
    hour_of_day = hour_of_day,
    n_obs = nrow(X_train_clean)
  )
  
  # Track feature usage
  feature_usage_summary <- rbind(feature_usage_summary, 
                                 data.frame(lead_hour = lead_hour, 
                                            n_features = length(selected_features)))
}

cat("\n=== TRAINING COMPLETE ===\n")
cat(length(models), "modelos individuais treinados\n")
cat(sprintf("Media de features por modelo: %.1f (vs %d total)\n", 
            mean(feature_usage_summary$n_features), length(feature_cols)))
```

**Training characteristics:**
- **Sequential execution:** Models trained one at a time (could be parallelized)
- **Fixed seed:** `set.seed(123)` ensures reproducibility
- **Progress reporting:** Every 24 models (one complete day)
- **Feature tracking:** Records number of features per model for analysis

#### **Training Time Estimates**

Based on the R implementation:
- **Per model:** ~5-30 seconds (depending on data size, features, ntree)
- **Total (216 models, sequential):** ~18-108 minutes
- **With parallelization (8 cores):** ~2.5-14 minutes

**For Epic-03:**
- Implement parallel training with joblib or multiprocessing
- Target: <10 minutes for 216 models on modern hardware

### 3.3 Model Storage

The R implementation stores models in memory as a list structure:

```r
models <- list(
  "y_h001" = list(model=<RF>, features=c(...), lead_hour=1, ...),
  "y_h002" = list(model=<RF>, features=c(...), lead_hour=2, ...),
  ...
  "y_h216" = list(model=<RF>, features=c(...), lead_hour=216, ...)
)
```

**For Epic-03 Python implementation:**
- **Serialization:** Use pickle or joblib for model objects
- **Metadata:** Store as separate JSON file
- **Versioning:** Include training date, data version, hyperparameters
- **Compression:** Use joblib's compress parameter to reduce size

**Example structure:**
```
models/
  random_forest/
    SECO/
      v1.0.0_20251117/
        metadata.json
        model_h001.pkl
        model_h002.pkl
        ...
        model_h216.pkl
```

---

## Prediction Pipeline

### 4.1 Prediction Process

```r
# From 3_treinamento_teste_modelo.R
predict_multioutput_rf <- function(models, test_data, feature_cols, target_cols) {
  # Initialize prediction matrix
  predictions <- matrix(NA, nrow = nrow(test_data), ncol = length(target_cols))
  colnames(predictions) <- target_cols
  
  # Predict for each lead hour
  for (i in 1:length(target_cols)) {
    target_name <- target_cols[i]
    model_info <- models[[target_name]]
    
    # Select features used by this model
    selected_features <- model_info$features
    
    # Prepare test data
    X_test <- as.matrix(test_data[, selected_features, with = FALSE])
    
    # Make predictions
    predictions[, i] <- predict(model_info$model, X_test)
  }
  
  return(predictions)
}
```

**Prediction characteristics:**
- **Input:** One row per forecast origin (30 rows for 30-day test)
- **Output:** Matrix of shape (n_origins, 216) with all hourly predictions
- **Feature alignment:** Each model uses its specific feature subset
- **No post-processing:** Raw predictions returned

**Prediction time:**
- **Per model:** <1 second
- **Total (216 models, sequential):** ~3-5 minutes
- **With parallelization:** <1 minute

### 4.2 Output Format

```python
# Python equivalent output structure
predictions = {
    'dataOrigem': ['2024-10-01', '2024-10-02', ...],  # Forecast origins
    'y_h001': [45000.2, 45200.5, ...],  # Predictions for hour 1
    'y_h002': [44500.8, 44800.3, ...],  # Predictions for hour 2
    ...
    'y_h216': [50000.1, 50500.7, ...]   # Predictions for hour 216
}
```

**Conversion to long format** (for downstream use):

```r
# R code to reshape predictions
pred_long <- melt(
  as.data.table(predictions),
  id.vars = "dataOrigem",
  measure.vars = target_cols,
  variable.name = "lead_hour",
  value.name = "prediction"
)

pred_long[, lead_hour := as.integer(gsub("y_h", "", lead_hour))]
pred_long[, timestamp := dataOrigem + hours(lead_hour)]
```

---

## Evaluation Framework

### 5.1 Metrics Calculation

The evaluation function computes **4 primary metrics per lead hour**:

```r
# From 4_avalia_performance_modelo.R
evaluate_multioutput_model <- function(y_true, y_pred, target_cols) {
  
  metrics <- data.table(
    lead_hour = as.numeric(gsub("y_h", "", target_cols)),
    day = ceiling(as.numeric(gsub("y_h", "", target_cols)) / 24),
    hour_of_day = ((as.numeric(gsub("y_h", "", target_cols)) - 1) %% 24) + 1,
    mae = NA_real_,
    rmse = NA_real_,
    mape = NA_real_,
    r2 = NA_real_
  )
  
  for (i in 1:length(target_cols)) {
    true_vals <- y_true[, i]
    pred_vals <- y_pred[, i]
    
    # Remove NAs
    valid_idx <- !is.na(true_vals) & !is.na(pred_vals)
    true_vals <- true_vals[valid_idx]
    pred_vals <- pred_vals[valid_idx]
    
    if (length(true_vals) > 0) {
      # MAE: Mean Absolute Error
      metrics$mae[i] <- mean(abs(true_vals - pred_vals))
      
      # RMSE: Root Mean Squared Error
      metrics$rmse[i] <- sqrt(mean((true_vals - pred_vals)^2))
      
      # MAPE: Mean Absolute Percentage Error
      metrics$mape[i] <- mean(abs((true_vals - pred_vals) / true_vals)) * 100
      
      # R²: Coefficient of Determination
      ss_res <- sum((true_vals - pred_vals)^2)
      ss_tot <- sum((true_vals - mean(true_vals))^2)
      metrics$r2[i] <- 1 - (ss_res / ss_tot)
    }
  }
  
  return(metrics)
}
```

**Metrics definitions:**

1. **MAE (Mean Absolute Error)**
   - Formula: $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
   - Unit: MW (same as target variable)
   - Interpretation: Average absolute deviation

2. **RMSE (Root Mean Squared Error)**
   - Formula: $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
   - Unit: MW
   - Interpretation: Standard deviation of residuals (penalizes large errors)

3. **MAPE (Mean Absolute Percentage Error)**
   - Formula: $\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$
   - Unit: %
   - Interpretation: Average percentage deviation (scale-independent)

4. **R² (Coefficient of Determination)**
   - Formula: $R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$
   - Unit: Dimensionless (0-1, can be negative)
   - Interpretation: Proportion of variance explained

**Output structure:**
```
lead_hour | day | hour_of_day | mae    | rmse   | mape  | r2
1         | 1   | 1           | 1250.5 | 1580.2 | 2.8   | 0.95
2         | 1   | 2           | 1180.3 | 1490.7 | 2.6   | 0.96
...
216       | 9   | 24          | 2100.8 | 2650.4 | 4.5   | 0.88
```

### 5.2 Aggregated Metrics

The implementation computes **overall performance** across horizons:

```r
# From exec_script.R
summary_rf <- data.table(
  model = "Random Forest",
  mae = mean(performance_metrics_rf$mae, na.rm = TRUE),
  rmse = mean(performance_metrics_rf$rmse, na.rm = TRUE),
  mape = mean(performance_metrics_rf$mape, na.rm = TRUE),
  r2 = mean(performance_metrics_rf$r2, na.rm = TRUE)
)
```

**Aggregation strategies:**
- **Overall:** Mean across all 216 lead hours
- **By day:** Mean across 24 hours per day (9 values)
- **By hour:** Mean across 9 days per hour (24 values)
- **By period:** Peak hours vs off-peak hours

### 5.3 Visualization Tools

The R implementation includes **3 visualization functions**:

#### **1. Time Series Comparison** (`plot_load_comparison`)

```r
p1 <- ggplot(plot_data, aes(x = timestamp)) +
  geom_line(aes(y = actual, color = "Actual"), size = 0.8) +
  geom_line(aes(y = predicted, color = "Predicted"), size = 0.8, alpha = 0.8) +
  facet_wrap(~ paste("Origin:", forecast_origin), scales = "free_x", ncol = 1) +
  labs(title = "Actual vs Predicted Load - Time Series View",
       x = "Forecast Timestamp", y = "Load (MW)")
```

**Shows:** Predicted vs actual load over time for selected forecast origins

#### **2. Scatter Plot by Day** 

```r
p2 <- ggplot(plot_data, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.6, size = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  facet_wrap(~ paste("Day", day), ncol = 3) +
  labs(title = "Actual vs Predicted by Forecast Day")
```

**Shows:** Prediction accuracy scatter plots, one panel per day (D+1 to D+9)

#### **3. Daily Profile Comparison**

```r
# Average profiles by day
daily_profiles <- plot_data[, .(
  actual = mean(actual, na.rm = TRUE),
  predicted = mean(predicted, na.rm = TRUE)
), by = .(day, hour_of_day)]

p3 <- ggplot(daily_profiles_long, aes(x = hour_of_day, y = load, color = type)) +
  geom_line(size = 1) +
  facet_wrap(~ paste("Day", day), ncol = 3) +
  labs(title = "Average Daily Load Profiles by Forecast Day")
```

**Shows:** Average hourly profiles, comparing actual vs predicted patterns

**For Epic-03 Python implementation:**
- Use matplotlib or plotly for visualizations
- Generate HTML reports with Jinja2 templates
- Include metrics tables alongside plots

---

## Orchestration & Workflow

### 6.1 Execution Flow

The main execution script (`exec_script.R`) orchestrates the entire pipeline:

```r
# High-level workflow
for (m in names(periodos_treino)) {  # Loop through test months
  
  # 1. DATA LOADING
  for (s in names(subsistemas)) {
    dt_load_s <- get_carga(s, DataIni, DataFin)
    dt_feriados_s <- get_feriado(s, DataIni, vetor_feriados)
    dt_temp_s <- get_temp_data(subsistemas[[s]], n=10, DataIni, DataFin)
  }
  
  # 2. DATA PREPROCESSING
  for (s in names(subsistemas)) {
    dt_temp_ajustado_s <- ajusta_dt_temp(s)
    dt_s <- uniao_dts(s)
  }
  
  # 3. FEATURE ENGINEERING
  for (s in names(subsistemas)) {
    dt_features_s <- multioutput_dt(s)
  }
  
  # 4. TRAIN/TEST SPLIT
  for (s in names(subsistemas)) {
    split_result <- split_train_test(s)
    train_data <- split_result$train
    test_data <- split_result$test
  }
  
  # 5. MODEL TRAINING
  for (s in names(subsistemas)) {
    models <- train_multioutput_rf(train_data, feature_cols, target_cols, ntree=100)
  }
  
  # 6. PREDICTION
  for (s in names(subsistemas)) {
    predictions <- predict_multioutput_rf(models, test_data, feature_cols, target_cols)
  }
  
  # 7. EVALUATION
  for (s in names(subsistemas)) {
    metrics <- evaluate_multioutput_model(y_true, y_pred, target_cols)
  }
  
  # 8. COMPARISON WITH BASELINE
  # Load PrevCargaDESSEM results
  dt_prevcarga <- fread("path/to/prevcarga_results.csv")
  # Compare metrics
  comparacao_geral <- rbind(summary_rf, summary_prevcarga)
  
  # 9. SAVE RESULTS
  fwrite(metrics, "resultados/metrics_month.csv")
  fwrite(comparacao_geral, "resultados/comparison_month.csv")
}

# 10. CONSOLIDATE ACROSS MONTHS
todas_comparacoes_gerais <- rbindlist(lapply(resultados_todos_meses, 
                                              function(x) x$comparacao_geral))
```

**Workflow characteristics:**
- **Monthly rolling evaluation:** 12 test months (Oct 2024 - Sep 2025)
- **13-month training window:** Each test month uses data from 13 months prior
- **Per-subsystem processing:** Models trained separately for each subsystem
- **Result consolidation:** Aggregate metrics across all test months

### 6.2 Configuration Management

**Training periods defined programmatically:**

```r
# From 0_config.R
meses_teste <- seq(ym("2024-10"), ym("2025-09"), by = "1 month")

periodos_treino <- lapply(meses_teste, function(m) {
  DataIni <- (m %m-% months(13)) %>% floor_date("month")
  DataFin <- m + days(8)
  list(
    DataIni = as.character(DataIni),
    DataFin = as.character(DataFin)
  )
})

names(periodos_treino) <- format(meses_teste, "%Y-%m")
```

**Results:**
```
periodos_treino = {
  "2024-10": {DataIni: "2023-09-01", DataFin: "2024-10-09"},
  "2024-11": {DataIni: "2023-10-01", DataFin: "2024-11-09"},
  ...
  "2025-09": {DataIni: "2024-08-01", DataFin: "2025-09-09"}
}
```

**For Epic-03:**
- Store configuration in YAML files
- Support flexible training periods (not just 13 months)
- Allow multiple subsystems in single configuration

**Example YAML:**
```yaml
training:
  model_type: random_forest
  horizons:
    - start: 1
      end: 216
  hyperparameters:
    n_estimators: 100
    random_state: 123
    n_jobs: -1
  
evaluation:
  test_size_days: 30
  metrics:
    - mae
    - rmse
    - mape
    - r2
  
rolling_window:
  train_months: 13
  test_months: 
    - "2024-10"
    - "2024-11"
    # ... etc
```

### 6.3 Result Storage

**Per-month results:**
```
resultados/
  2024-10/
    metrics_SECO.csv
    predictions_SECO.csv
    comparison_SECO.csv
  2024-11/
    metrics_SECO.csv
    predictions_SECO.csv
    comparison_SECO.csv
  ...
```

**Consolidated results:**
```
resultados/
  consolidacao_geral_todos_meses.csv
  consolidacao_diaria_todos_meses.csv
```

**For Epic-03:**
- Store in Parquet format (more efficient than CSV)
- Include model metadata (version, hyperparameters, training date)
- Implement versioning for reproducibility

---

## Key Insights for Epic-03 Implementation

### 7.1 Architecture Decisions

#### **Multi-Output vs Single Model**

The R implementation uses **216 independent models**. For Epic-03, consider:

**Option A: Keep 216 independent models (RF approach)**
- ✅ Pros: Feature flexibility, parallel training, proven approach
- ❌ Cons: 216× storage, no cross-horizon learning
- **Use case:** Random Forest implementation in Epic-03

**Option B: Single multi-output model (LGBM approach)**
- ✅ Pros: Cross-horizon learning, compact storage, faster inference
- ❌ Cons: Less flexible feature selection, more complex training
- **Use case:** LGBM implementation in Epic-03

**Recommendation:** Support **both architectures** with a unified interface:

```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame, context: dict):
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, context: dict) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def save(self, path: str):
        pass
    
    @abstractmethod
    def load(self, path: str):
        pass
```

### 7.2 Training Pipeline Requirements

**Must-have capabilities:**

1. **Feature Selection Module**
   ```python
   def select_features_for_lead_hour(
       lead_hour: int,
       all_features: List[str],
       strategy: str = "horizon_aware"
   ) -> List[str]:
       """Select relevant features based on forecast horizon."""
       pass
   ```

2. **Parallel Training Support**
   ```python
   from joblib import Parallel, delayed
   
   def train_parallel(
       train_data: pd.DataFrame,
       target_cols: List[str],
       n_jobs: int = -1
   ) -> Dict[str, Model]:
       """Train multiple models in parallel."""
       models = Parallel(n_jobs=n_jobs)(
           delayed(train_single_model)(train_data, target_col)
           for target_col in target_cols
       )
       return dict(zip(target_cols, models))
   ```

3. **Model Registry**
   ```python
   class ModelRegistry:
       """Manage model versions and metadata."""
       
       def register(self, model: BaseModel, metadata: dict):
           """Save model with versioning."""
           pass
       
       def load(self, model_id: str, version: str = "latest") -> BaseModel:
           """Load specific model version."""
           pass
       
       def list_versions(self, model_id: str) -> List[str]:
           """List available versions."""
           pass
   ```

4. **Training Progress Tracking**
   ```python
   from tqdm import tqdm
   
   for i, target_col in enumerate(tqdm(target_cols, desc="Training RF models")):
       if i % 24 == 0:
           logger.info(f"Training day {i//24 + 1}/9 (hours {i+1}-{min(i+24, 216)})")
       model = train_single_model(train_data, target_col)
   ```

### 7.3 Prediction Pipeline Requirements

1. **Batch Prediction**
   ```python
   def predict_batch(
       models: Dict[str, Model],
       test_data: pd.DataFrame,
       batch_size: int = 30
   ) -> pd.DataFrame:
       """Predict for multiple forecast origins."""
       pass
   ```

2. **Feature Alignment**
   ```python
   def align_features(
       data: pd.DataFrame,
       required_features: List[str]
   ) -> pd.DataFrame:
       """Ensure data has required features in correct order."""
       missing = set(required_features) - set(data.columns)
       if missing:
           raise ValueError(f"Missing features: {missing}")
       return data[required_features]
   ```

3. **Output Formatting**
   ```python
   def format_predictions(
       predictions: np.ndarray,
       origins: List[str],
       target_cols: List[str]
   ) -> pd.DataFrame:
       """Convert prediction matrix to long format."""
       df = pd.DataFrame(predictions, columns=target_cols)
       df['dataOrigem'] = origins
       return df.melt(id_vars='dataOrigem', var_name='lead_hour', 
                      value_name='prediction')
   ```

### 7.4 Evaluation Framework Requirements

**Epic-07 will handle comprehensive evaluation, but Epic-03 should include:**

1. **Basic Metrics Calculation**
   ```python
   class MetricsCalculator:
       @staticmethod
       def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
           return np.mean(np.abs(y_true - y_pred))
       
       @staticmethod
       def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
           return np.sqrt(np.mean((y_true - y_pred) ** 2))
       
       @staticmethod
       def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
           return 100 * np.mean(np.abs((y_true - y_pred) / y_true))
       
       @staticmethod
       def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
           ss_res = np.sum((y_true - y_pred) ** 2)
           ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
           return 1 - (ss_res / ss_tot)
   ```

2. **Per-Horizon Metrics**
   ```python
   def evaluate_by_horizon(
       y_true: pd.DataFrame,
       y_pred: pd.DataFrame
   ) -> pd.DataFrame:
       """Compute metrics for each lead hour."""
       metrics = []
       for col in y_true.columns:
           metrics.append({
               'lead_hour': int(col.replace('y_h', '')),
               'mae': MetricsCalculator.mae(y_true[col], y_pred[col]),
               'rmse': MetricsCalculator.rmse(y_true[col], y_pred[col]),
               'mape': MetricsCalculator.mape(y_true[col], y_pred[col]),
               'r2': MetricsCalculator.r2(y_true[col], y_pred[col])
           })
       return pd.DataFrame(metrics)
   ```

### 7.5 Model Serialization

**Requirements for model persistence:**

1. **Serialization Format**
   ```python
   import joblib
   from pathlib import Path
   
   def save_model(
       model: BaseModel,
       path: str,
       metadata: dict,
       compress: int = 3
   ):
       """Save model with metadata."""
       path = Path(path)
       path.mkdir(parents=True, exist_ok=True)
       
       # Save model
       joblib.dump(model, path / "model.pkl", compress=compress)
       
       # Save metadata
       with open(path / "metadata.json", "w") as f:
           json.dump(metadata, f, indent=2)
   
   def load_model(path: str) -> Tuple[BaseModel, dict]:
       """Load model and metadata."""
       path = Path(path)
       model = joblib.load(path / "model.pkl")
       with open(path / "metadata.json", "r") as f:
           metadata = json.load(f)
       return model, metadata
   ```

2. **Metadata Schema**
   ```python
   metadata = {
       "model_type": "random_forest",
       "model_id": "rf_SECO_v1.0.0",
       "subsystem": "SECO",
       "horizon": {"start": 1, "end": 216},
       "n_models": 216,
       "training": {
           "date": "2025-11-17T10:30:00Z",
           "data_version": "v2.3.1",
           "train_start": "2023-09-01",
           "train_end": "2024-09-30",
           "n_samples": 360,
           "hyperparameters": {
               "n_estimators": 100,
               "random_state": 123
           }
       },
       "features": {
           "total": 1514,
           "avg_per_model": 75.2,
           "selection_strategy": "horizon_aware"
       },
       "performance": {
           "mae": 1450.3,
           "rmse": 1850.7,
           "mape": 3.2,
           "r2": 0.94
       }
   }
   ```

### 7.6 Comparison with LGBM (for Epic-03)

| Aspect | Random Forest | LGBM (Target) |
|--------|---------------|---------------|
| **Architecture** | 216 independent models | Single multi-output or fewer models |
| **Feature selection** | Dynamic per hour | Use all features + built-in importance |
| **Training speed** | Moderate (parallelizable) | Fast (gradient boosting) |
| **Hyperparameters** | Fewer (ntree, mtry, nodesize) | Many (learning_rate, max_depth, etc.) |
| **Interpretability** | Feature importance per model | Global feature importance |
| **Memory usage** | High (216 models) | Low (single model) |
| **Inference speed** | Moderate | Fast |
| **Overfitting risk** | Lower (ensemble) | Higher (requires tuning) |

**Epic-03 should implement both:**
1. **Random Forest:** Baseline model, proven approach
2. **LGBM:** Advanced model, better performance potential

### 7.7 Testing Strategy

**Unit tests required:**

1. **Model Training**
   ```python
   def test_train_single_model():
       """Test training a single RF model."""
       model = RandomForestModel(lead_hour=1)
       model.train(X_train, y_train)
       assert model.is_fitted()
       assert len(model.features) > 0
   
   def test_train_all_models():
       """Test training all 216 models."""
       models = train_multioutput_rf(train_data, target_cols)
       assert len(models) == 216
       assert all(m.is_fitted() for m in models.values())
   ```

2. **Feature Selection**
   ```python
   def test_feature_selection_horizon_aware():
       """Test horizon-aware feature selection."""
       features_h1 = select_features_for_lead_hour(1, all_features)
       features_h216 = select_features_for_lead_hour(216, all_features)
       
       # Hour 1 should have fewer temperature features than hour 216
       temp_h1 = [f for f in features_h1 if f.startswith('tmax') or f.startswith('tmin')]
       temp_h216 = [f for f in features_h216 if f.startswith('tmax') or f.startswith('tmin')]
       assert len(temp_h1) < len(temp_h216)
   ```

3. **Prediction**
   ```python
   def test_predict_shape():
       """Test prediction output shape."""
       predictions = predict_multioutput_rf(models, test_data)
       assert predictions.shape == (len(test_data), 216)
   
   def test_predict_no_nan():
       """Test predictions have no NaN values."""
       predictions = predict_multioutput_rf(models, test_data)
       assert not np.isnan(predictions).any()
   ```

4. **Serialization**
   ```python
   def test_model_save_load():
       """Test model can be saved and loaded."""
       model.save("test_model.pkl")
       loaded_model = RandomForestModel.load("test_model.pkl")
       
       # Predictions should be identical
       pred1 = model.predict(X_test)
       pred2 = loaded_model.predict(X_test)
       np.testing.assert_array_almost_equal(pred1, pred2)
   ```

### 7.8 Performance Benchmarks

**Target performance for Epic-03:**

| Operation | Target Time | Current R Time |
|-----------|-------------|----------------|
| Data loading (1 subsystem, 13 months) | <2 min | ~3 min |
| Feature engineering (1 subsystem) | <5 min | ~7 min |
| Training (216 models, parallel) | <10 min | ~30 min (sequential) |
| Prediction (30 days) | <1 min | ~3 min |
| Evaluation (all metrics) | <30 sec | ~1 min |
| **Total pipeline** | **<20 min** | **~45 min** |

**Optimization strategies:**
- Use polars for faster data manipulation
- Parallel training with joblib (n_jobs=-1)
- Batch prediction
- Parquet for I/O (faster than CSV)
- Caching intermediate results

---

## Summary: Epic-03 Deliverables

Based on this analysis, Epic-03 should deliver:

### Core Components

1. **Base Model Interface** (`BaseModel` abstract class)
2. **Random Forest Model Implementation:**
   - `RandomForestModel` class
   - `train_multioutput_rf()` function
   - `predict_multioutput_rf()` function
   - Feature selection logic per lead hour

3. **Model Registry System:**
   - Save/load with versioning
   - Metadata management
   - Model tracking

4. **Training Pipeline:**
   - Train/test split (temporal)
   - Feature selection per model
   - Parallel training support
   - Progress tracking

5. **Prediction Pipeline:**
   - Batch prediction
   - Feature alignment
   - Output formatting (wide & long)

6. **Serialization:**
   - Pickle/joblib for models
   - JSON for metadata
   - Compression support

7. **Basic Evaluation:**
   - MAE, RMSE, MAPE, R² calculation
   - Per-horizon metrics
   - Aggregated metrics

8. **Universal Trainer:**
   - Configurable via YAML
   - Support multiple model types
   - Rolling window evaluation

### Success Criteria

✅ RF trains and predicts for D+0 to D+8 (216 hours)  
✅ Feature selection prevents data leakage  
✅ Parallel training reduces time by 5-10×  
✅ Models save/load with versioning  
✅ Predictions match expected shape and quality  
✅ Basic evaluation metrics computed correctly  
✅ Memory usage acceptable (<8GB for 216 models)  
✅ Training pipeline completes in <20 minutes  
✅ 80%+ unit test coverage

### Out of Scope (for other Epics)

- ❌ LGBM BLF strategy (intraday updates) → Part of Epic-03 but separate story
- ❌ Comprehensive evaluation dashboard → Epic-07
- ❌ Model combination → Epic-05
- ❌ Hierarchical reconciliation → Epic-06
- ❌ CLI interface → Epic-09
- ❌ Production deployment → Epic-11

---

## Appendix A: R vs Python Implementation Guide

### A.1 Data Structures

| R | Python (pandas) | Notes |
|---|-----------------|-------|
| `data.table` | `pd.DataFrame` | Use polars for better performance |
| `matrix` | `np.ndarray` | For model training |
| `list()` | `dict` or `list` | For model storage |
| `NA_real_` | `np.nan` | Missing values |

### A.2 Key Function Translations

**Train/Test Split:**
```python
# R: Manual slicing
n_train <- floor(n_total - 30)
train_data <- dt_features[1:n_train]
test_data <- dt_features[(n_train + 1):n_total]

# Python: sklearn
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(
    df, test_size=30, shuffle=False
)
```

**Random Forest:**
```python
# R: randomForest package
library(randomForest)
rf_model <- randomForest(x=X, y=Y, ntree=100, importance=TRUE)

# Python: sklearn
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=123, n_jobs=-1)
rf_model.fit(X, y)
```

**Feature Selection (regex):**
```python
# R: grep
origin_features <- grep("^origin_", feature_cols, value=TRUE)

# Python: list comprehension
origin_features = [f for f in feature_cols if f.startswith('origin_')]

# Or: regex
import re
origin_features = [f for f in feature_cols if re.match(r'^origin_', f)]
```

**Parallel Processing:**
```python
# R: parallel package (not used in current implementation)
library(parallel)
models <- mclapply(target_cols, train_single_model, mc.cores=8)

# Python: joblib
from joblib import Parallel, delayed
models = Parallel(n_jobs=8)(
    delayed(train_single_model)(target_col) for target_col in target_cols
)
```

### A.3 Performance Considerations

| Aspect | R (current) | Python (target) |
|--------|-------------|-----------------|
| Data I/O | `fread/fwrite` (fast) | `pd.read_parquet` (faster) |
| Data manipulation | `data.table` (fast) | `polars` (faster) |
| Model training | Sequential | Parallel (joblib) |
| Memory usage | ~6GB | Target <8GB |
| Total runtime | ~45 min | Target <20 min |

---

## Appendix B: Configuration Examples

### B.1 Model Configuration (YAML)

```yaml
model:
  type: random_forest
  subsystem: SECO
  horizon:
    start: 1
    end: 216
  
  hyperparameters:
    n_estimators: 100
    random_state: 123
    n_jobs: -1
    max_depth: null
    min_samples_split: 2
    min_samples_leaf: 1
  
  feature_selection:
    strategy: horizon_aware
    include_all_historical: true
  
  training:
    test_size_days: 30
    validation_split: 0.1
    remove_incomplete: true
  
  output:
    save_predictions: true
    save_feature_importance: true
    format: parquet
```

### B.2 Training Period Configuration

```yaml
rolling_evaluation:
  train_window_months: 13
  test_periods:
    - year_month: "2024-10"
      data_start: "2023-09-01"
      data_end: "2024-10-09"
    - year_month: "2024-11"
      data_start: "2023-10-01"
      data_end: "2024-11-09"
    # ... etc
  
  subsystems:
    - code: SECO
      id: 11
    - code: S
      id: 10
    - code: NE
      id: 9
    - code: N
      id: 8
```

---

**Document Status:** Complete  
**Next Steps:** Use this analysis to create detailed user stories for Epic-03  
**Focus Areas:**
1. Base model interface design
2. Random Forest multi-output implementation
3. Feature selection per lead hour
4. Model registry and versioning
5. Parallel training optimization

**Review:** Validate architecture decisions and performance targets with team
