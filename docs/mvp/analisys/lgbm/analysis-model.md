# LGBM Model Analysis - PrevCarga BLF Strategy

**Document:** Analysis of LGBM Model Implementation  
**Source Files:** `1_Executar.R`, `ANN_PVDS.R`, `funcoes_aux.R`, `ModeloBLF_lgbm_v4.R`  
**Purpose:** Inform Epic-03 (Model Layer - End-to-End Models)  
**Date:** November 17, 2025

---

## Executive Summary

The current R implementation demonstrates a sophisticated LGBM-based forecasting system with a unique **BLF (Best Linear Forecast)** intraday prediction strategy. The model uses custom loss functions, weighted training to prioritize peak hours, and iterative refinement for day-ahead predictions. This analysis documents the architecture, training methodology, prediction strategies, and evaluation framework to guide Epic-03 implementation in Python.

**Key Characteristics:**
- **Model Type:** LightGBM with custom objective function
- **Prediction Horizons:** D+0 (intraday from 8h) and D+1 (full day)
- **Training Strategy:** Weighted loss with peak hour emphasis
- **BLF Strategy:** Proportional adjustment using verified early-day load
- **Evaluation Focus:** Peak period accuracy (18h-22h)

---

## 1. Model Architecture Overview

### 1.1 Model Type and Configuration

**Algorithm:** LightGBM (Gradient Boosting Decision Trees)

**Core Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_leaves` | 16 | Control tree complexity to prevent overfitting |
| `min_data_in_leaf` | 50 | Minimum samples per leaf for generalization |
| `learning_rate` | 0.1 | Standard learning rate for stable convergence |
| `num_iterations` | 5000 | Maximum iterations with early stopping |
| `early_stopping_rounds` | 50 | Stop if no improvement for 50 iterations |
| `objective` | Custom (weighted L1) | Penalize underestimation during peak hours |
| `metric` | Custom weighted MAPE | Focus on peak period accuracy |

**Feature Configuration:**

```r
parametros_BLF <- list(
  categorical_features = match(coluna_categorica_BLF, colnames(entradas_treino_BLF)),
  feature_pre_filter = FALSE
)
```

**Categorical Features Treated Natively:**
- `Feriado`, `PreFeriado`, `PosFeriado`, `Mes`
- All dummy variables (day of week, month, special days)

**Note:** Current implementation creates one-hot encoded dummies, but LGBM can handle categorical features directly. Python implementation should leverage native categorical support for efficiency.

---

### 1.2 Data Splits

**Training-Validation-Test Strategy:**

```python
# Dynamic split based on evaluation month
evaluation_month = f"{ANO}-{MES}-01"
first_day_eval_month = date(ANO, MES, 1)

# Validation set: Month before evaluation
validation_end = first_day_eval_month - timedelta(days=1)  # Last day of previous month
validation_start = date(validation_end.year, validation_end.month, 1)

# Training set: All data before validation month
training_end = validation_start - timedelta(days=1)
training_start = "2022-10-01"  # Fixed start

# Test set: Evaluation month
test_start = first_day_eval_month
test_end = current_date  # Up to present
```

**Example for ANO=2024, MES=5:**
- **Training:** 2022-10-01 to 2024-03-31
- **Validation:** 2024-04-01 to 2024-04-30
- **Test:** 2024-05-01 to 2024-05-31

**Rationale:** Rolling window ensures model sees most recent patterns while maintaining temporal separation.

---

## 2. Custom Loss Function and Weighting

### 2.1 Time-Based Sample Weighting

**Purpose:** Prioritize accuracy during peak demand hours (18h-22h)

**Implementation:**

```r
# Weight calculation
fator_penalidade <- 1  # Can be increased (e.g., 2) for stronger emphasis
vetor_penalidade_treino <- ifelse(hour(dados_treino_BLF$DataHora) > 18, 
                                   fator_penalidade, 
                                   1)
```

**Weight Assignment:**
- **Hours 00:00-18:00:** Weight = 1.0 (standard)
- **Hours 18:30-23:30:** Weight = `fator_penalidade` (default 1.0, configurable)

**Effect:** Higher weights during peak hours cause LGBM to:
1. Prioritize reducing errors during these periods
2. Accept slightly larger errors during off-peak if it improves peak accuracy
3. Focus gradient descent on critical business hours

**Python Requirements:**
- Configurable `peak_weight_factor` parameter
- Time-based weight vector generation utility
- Pass to `lgb.Dataset(weight=...)`

---

### 2.2 Custom Objective Function (Asymmetric Loss)

**Purpose:** Penalize underestimation more heavily than overestimation (conservative forecasting)

**Implementation:**

```r
objetivo_customizado <- function(preds, dtrain) {
  labels <- rotulos_treino_global
  pesos <- attr(dtrain, "weight")
  if (is.null(pesos)) {
    pesos <- rep(1, length(labels))
  }
  
  res <- preds - labels  # Residual: positive if overestimation, negative if underestimation
  
  # Gradient: Apply penalty factor to underestimation errors
  grad <- ifelse((res < 0) & (pesos > 1), 
                 pesos * res,      # Amplify gradient for weighted underestimation
                 res)              # Standard gradient otherwise
  
  # Hessian: Second derivative for Newton step
  hess <- ifelse((res < 0) & (pesos > 1), 
                 pesos,            # Amplify curvature for weighted underestimation
                 1)                # Standard curvature otherwise
  
  return(list(grad = grad, hess = hess))
}
```

**Mathematical Interpretation:**

For observation $i$ with prediction $\hat{y}_i$, actual $y_i$, and weight $w_i$:

$$
\text{residual}_i = \hat{y}_i - y_i
$$

**Gradient (first derivative of loss):**
$$
\frac{\partial L}{\partial \hat{y}_i} = 
\begin{cases}
w_i \cdot (\hat{y}_i - y_i) & \text{if } \hat{y}_i < y_i \text{ and } w_i > 1 \\
(\hat{y}_i - y_i) & \text{otherwise}
\end{cases}
$$

**Hessian (second derivative of loss):**
$$
\frac{\partial^2 L}{\partial \hat{y}_i^2} = 
\begin{cases}
w_i & \text{if } \hat{y}_i < y_i \text{ and } w_i > 1 \\
1 & \text{otherwise}
\end{cases}
$$

**Effect:**
- **Underestimation during peak hours:** Gradient and Hessian scaled by weight → model learns faster to avoid these errors
- **Overestimation or off-peak errors:** Standard L1-like behavior
- **Result:** Model becomes more conservative during critical periods

**Python Requirements:**
- Custom objective function compatible with `lightgbm.train(obj=...)`
- Access to training labels via closure or global variable
- Return gradient and Hessian as numpy arrays

---

### 2.3 Custom Evaluation Metric (Weighted MAPE)

**Purpose:** Monitor validation performance with same weighting as training

**Implementation:**

```r
avaliacao_customizada <- function(preds, dtrain) {
  labels <- attr(dtrain, "label")
  if (is.null(labels)) {
    labels <- saidas_validacao_BLF
  }
  
  pesos_validacao <- vetor_penalidade_validacao
  
  # Absolute Percentage Error per observation
  errors <- abs(preds - labels) / pmax(abs(labels), 1e-6)
  
  # Weighted MAPE: sum of weighted errors / total observations
  weighted_mape <- (sum(pesos_validacao * errors) / length(errors)) * 100
  
  return(list(name = "weighted_mape_new", 
              value = weighted_mape, 
              higher_better = FALSE))
}
```

**Formula:**

$$
\text{Weighted MAPE} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100
$$

Where:
- $N$ = total observations
- $w_i$ = weight for observation $i$ (1.0 or `fator_penalidade`)
- $y_i$ = actual load
- $\hat{y}_i$ = predicted load

**Note:** Division by $N$ (not $\sum w_i$) keeps metric interpretable as percentage.

**Python Requirements:**
- Custom evaluation function compatible with `lightgbm.train(feval=...)`
- Return tuple: `(metric_name, metric_value, is_higher_better)`
- Use same weight vector as training

---

## 3. Training Process

### 3.1 Hyperparameter Optimization

**Grid Search Configuration:**

```r
grid_params <- expand.grid(
  max_leaves = c(16),
  min_data_in_leaf = c(50),
  learning_rate = c(0.1),
  num_iterations = c(5000),
  stringsAsFactors = FALSE
)
```

**Current State:** Grid contains only single values (no actual tuning).

**Recommendation for Python Implementation:**

```python
param_grid = {
    'max_leaves': [8, 16, 31, 63],
    'min_data_in_leaf': [20, 50, 100],
    'learning_rate': [0.05, 0.1, 0.2],
    'num_iterations': [5000],  # Fixed, use early stopping
    'feature_fraction': [0.8, 1.0],  # Column subsampling
    'bagging_fraction': [0.8, 1.0],  # Row subsampling
    'bagging_freq': [0, 5]
}
```

**Search Strategy:** Use Optuna or Hyperopt for Bayesian optimization instead of exhaustive grid search.

**Evaluation:** Select best model based on validation weighted MAPE.

---

### 3.2 Training Loop

**Pseudo-code:**

```python
best_mape = float('inf')
best_model = None

for params in param_grid:
    # Configure parameters
    lgb_params = {
        'objective': custom_objective,
        'metric': 'mape',
        'max_leaves': params['max_leaves'],
        'min_data_in_leaf': params['min_data_in_leaf'],
        'learning_rate': params['learning_rate'],
        'num_iterations': params['num_iterations'],
        'early_stopping_rounds': 50,
        'verbosity': -1
    }
    
    # Train model
    model = lgb.train(
        lgb_params,
        train_set=train_data,
        valid_sets=[valid_data],
        feval=custom_weighted_mape,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # Evaluate on validation
    preds = model.predict(valid_features)
    weighted_mape = calculate_weighted_mape(valid_labels, preds, valid_weights)
    
    # Track best
    if weighted_mape < best_mape:
        best_mape = weighted_mape
        best_model = model
        best_params = params

print(f"Best Weighted MAPE: {best_mape:.2f}%")
print(f"Best Parameters: {best_params}")
```

**Key Features:**
- Early stopping prevents overfitting (monitors `weighted_mape_new` on validation)
- Best model selected by minimum validation weighted MAPE
- Training history plotted for diagnostics

---

## 4. BLF Prediction Strategy (Intraday D+0 and D+1)

### 4.1 Two-Stage Prediction Workflow

**Stage 1: Intraday Update (D+0)**
1. Model predicts full day D+0 using previous day (D-1) load at same hours
2. At 8:00 AM, verified load from 00:30-08:00 (16 observations) becomes available
3. Calculate hourly proportion factors from model prediction
4. Apply proportions to verified data to forecast 08:30-23:30

**Stage 2: Day-Ahead Forecast (D+1)**
1. Use corrected D+0 forecast (after 8:00 AM update) as lag feature
2. Update temperature forecast to D+1 values
3. Predict full D+1 (48 semi-hourly values)

---

### 4.2 BLF Algorithm Details

**Mathematical Formulation:**

Let:
- $\hat{y}_h^{(0)}$ = initial model prediction for hour $h$ of day $D$
- $y_h^{v}$ = verified load for hour $h$ (available for $h \in [0.5, 8.0]$)
- $\hat{y}_h^{(BLF)}$ = BLF-adjusted prediction for hour $h$

**Step 1: Calculate Proportional Factors**

For each semi-hour $h$:
$$
r_h = \frac{\hat{y}_h^{(0)}}{\hat{y}_{h-0.5}^{(0)}}
$$

**Step 2: Iterative Propagation**

For $h > 8.0$:
$$
\hat{y}_h^{(BLF)} = \hat{y}_{h-0.5}^{(BLF)} \cdot r_h
$$

With initialization:
$$
\hat{y}_h^{(BLF)} = y_h^{v} \quad \text{for } h \leq 8.0
$$

**Implementation (from R code):**

```r
# 1. Calculate hourly differences (proportions)
diferencas_horarias <- c(1, round(previsao_dia_atual[-1] / previsao_dia_atual[-length(previsao_dia_atual)], 7))

# 2. Start with verified values (00:30 to 08:00)
valores_verificados <- teste[hour(DataHora) <= 8 & Data == datas_teste[i], 
                             .(DataHora, CargaGlobal_real_BLF)]
valores_completos <- copy(valores_verificados)

# 3. Propagate using proportions
for (j in seq_len(length(previsao_dia_atual))[-seq_len(nrow(valores_verificados))]) {
  proximo_valor <- valores_completos$CargaGlobal_real_BLF[nrow(valores_completos)] * diferencas_horarias[j]
  valores_completos <- rbind(
    valores_completos,
    data.table(
      DataHora = teste[Data == datas_teste[i] & Hora > 8]$DataHora[j - nrow(valores_verificados)],
      CargaGlobal_real_BLF = proximo_valor
    )
  )
}
```

**Advantages:**
1. **Self-correcting:** Incorporates actual observed load trend
2. **Captures unexpected events:** Early-morning patterns propagate forward
3. **Reduces bias:** Anchors to verified data rather than purely model-based
4. **Smooth transitions:** Proportional adjustments preserve intraday shape

**Limitations:**
1. **Error accumulation:** Early errors propagate through the day
2. **Assumes stable patterns:** May fail if afternoon significantly deviates from morning
3. **Limited to D+0:** Requires verified data (not applicable to D+2+)

---

### 4.3 Temperature BLF Adjustment (Same Logic)

**Application:** Apply same proportional logic to temperature lags for D+1 prediction

```r
# Calculate temperature proportions from forecast
diferencas_horarias_temp <- c(1, round(TemperaturaD2$TemperaturaD1[-1] / 
                                       TemperaturaD2$TemperaturaD1[-length(TemperaturaD2$TemperaturaD1)], 7))

# Propagate from verified temperatures
valores_verificados_temp <- teste[hour(DataHora) <= 8 & Data == datas_teste[i], 
                                  .(DataHora, TemperaturaECMWF_anterior_BLF)]
# ... (same propagation loop as load)
```

**Rationale:** Temperature forecast errors at 8:00 AM can be corrected similarly, improving D+1 accuracy.

---

### 4.4 D+1 Prediction with Updated Lags

**After BLF correction of D+0:**

```r
if (nrow(entradas_dia_seguinte) > 0) {
  entradas_dia_seguinte[, `:=`(
    CargaGlobal_anterior_BLF = valores_completos$CargaGlobal_real_BLF,  # Updated with BLF
    TemperaturaECMWF_anterior_BLF = valores_completos_temp$TemperaturaECMWF_anterior_BLF,  # Updated
    Temperatura_prevista_BLF = TemperaturaD2$TemperaturaD2amort  # D+2 forecast
  )]
  
  # Predict full D+1
  previsao_dia_seguinte <- predict(
    melhor_modelo,
    data.matrix(entradas_dia_seguinte[, c(coluna_numerica_BLF, coluna_categorica_BLF), with = FALSE])
  )
}
```

**Key Insight:** D+1 prediction uses **corrected D+0 values** as lag features, not original model predictions. This creates a cascade effect where intraday corrections improve next-day forecasts.

---

## 5. Post-Processing and Smoothing

### 5.1 Post-Prediction LOESS Smoothing

**Purpose:** Remove high-frequency noise from raw predictions while preserving daily patterns

**Application:** Two-stage smoothing

**Stage 1: Normalized Space (before denormalization)**

```r
resultado_previsao[, CargaGlobalSuavizada := loess(
  CargaGlobal_prevista ~ as.numeric(DataHora), 
  data = .SD, 
  span = 0.1
)$fitted]
```

**Stage 2: After Denormalization (final output)**

```r
AvaliacaoFinal[, CargaGlobalSuavizada := {
  ajuste <- loess(Previsto_denorm ~ as.numeric(DataHora), 
                  data = .SD, 
                  span = 0.15)
  ajuste$fitted
}, by = Data]
```

**Parameters:**
- **Span:** 0.1 (stage 1) and 0.15 (stage 2) = 10-15% of data points in local window
- **Grouping:** Per day (`by = Data`)
- **Independent Variable:** Numeric timestamp for smooth interpolation

**Effect:**
- Reduces semi-hourly oscillations
- Preserves peak magnitudes and timing
- Creates visually smooth forecast curves

**Python Implementation:**

```python
from statsmodels.nonparametric.smoothers_lowess import lowess

def smooth_predictions(df, value_col='prediction', time_col='datetime', span=0.15):
    """Apply LOESS smoothing per day."""
    smoothed = []
    for date, group in df.groupby(df[time_col].dt.date):
        x = group[time_col].astype('int64').values  # Convert to numeric
        y = group[value_col].values
        
        # LOWESS smoothing
        frac = span  # Fraction of data used for smoothing
        smoothed_y = lowess(y, x, frac=frac, return_sorted=False)
        
        smoothed.append(pd.DataFrame({
            time_col: group[time_col],
            f'{value_col}_smoothed': smoothed_y
        }))
    
    return pd.concat(smoothed).reset_index(drop=True)
```

**Recommendation:** Make smoothing **optional and configurable** in Python implementation (not all models may benefit).

---

## 6. Normalization and Denormalization

### 6.1 Feature Normalization (Min-Max Scaling)

**Applied Features:** All continuous variables except categorical/dummies

**Fitting:** Only on training set

```r
# Calculate min/max per feature from training data
min_max_BLF <- lapply(
  Total_completos_BLF[, ..colunas_normalizar_BLF],
  function(x) {
    c(min = min(x, na.rm = TRUE), max = max(x, na.rm = TRUE))
  }
)

# Normalize
normalizar_BLF <- function(x, min_val, max_val) {
  return((x - min_val) / (max_val - min_val))
}

for (i in 1:length(colunas_normalizar_BLF)) {
  col <- colunas_normalizar_BLF[i]
  min_val <- min_max_BLF[[col]]["min"]
  max_val <- min_max_BLF[[col]]["max"]
  Total_completos_BLF[, (col) := normalizar_BLF(get(col), min_val, max_val)]
}
```

**Excluded Features:**
- `Feriado`, `PreFeriado`, `PosFeriado`, `Mes`
- All dummy variables (`dias_dummy`, `mes_dummy`, `dias_especiais_dummy`)

**Result:** All normalized features in range [0, 1]

---

### 6.2 Prediction Denormalization

**Target Variable Only:** `CargaGlobal_real_BLF`

```r
# Denormalize predictions back to MW
AvaliacaoFinal[, Previsto_denorm := denormalizar_BLF(
  CargaGlobalSuavizada,  # Smoothed normalized prediction
  min_val = min_max_BLF$CargaGlobal_real_BLF["min"], 
  max_val = min_max_BLF$CargaGlobal_real_BLF["max"]
)]
```

**Formula:**

$$
\hat{y}_{\text{MW}} = \hat{y}_{\text{norm}} \cdot (\text{max} - \text{min}) + \text{min}
$$

**Python Requirements:**
- Store `MinMaxScaler` fitted on training data
- Serialize scaler with model
- Apply same transformation to validation/test/production data
- Inverse transform predictions before evaluation/output

---

## 7. Evaluation Framework

### 7.1 Evaluation Metrics

**Primary Metric: MAPE (Mean Absolute Percentage Error)**

```r
mape <- function(Real, Previsto) {
  return(mean(abs((Real - Previsto) / Real)) * 100)
}

# Overall MAPE
mape_diatodo <- mape(
  Real = avaliacao$CargaGlobal, 
  Previsto = avaliacao$Previsto_denorm
)

# Peak Period MAPE (18h-22h)
mape_noite <- mape(
  Real = PontaNoturna$CargaGlobal, 
  Previsto = PontaNoturna$Previsto_denorm
)
```

**Formula:**

$$
\text{MAPE} = \frac{100}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

---

### 7.2 Business-Specific Metrics

**1. Peak Period Bias (Viés Médio da Ponta)**

```r
viés_medio_ponta <- mean(PontaNoturna$Previsto_denorm - PontaNoturna$CargaGlobal, na.rm = TRUE)
```

**Interpretation:**
- **Positive bias:** Systematic overestimation during peak → acceptable (conservative)
- **Negative bias:** Systematic underestimation during peak → **critical problem**

**Target:** Bias ≈ 0 or slightly positive

---

**2. Days with Underestimation**

```r
desvio_medio_por_dia <- PontaNoturna[, .(
  DesvioMedio_MW = mean(CargaGlobal - Previsto_denorm, na.rm = TRUE)
), by = as.Date(DataHora)]

dias_subestimativa <- desvio_medio_por_dia[DesvioMedio_MW < 0, .N]
```

**Interpretation:** Count of days where average peak period prediction is below actual load.

**Target:** Minimize this count (ideally < 10% of days)

---

**3. Maximum Peak Underestimation**

```r
subestimativa_maxima <- max(PontaNoturna[Previsto_denorm < CargaGlobal, 
                                         abs(CargaGlobal - Previsto_denorm)], 
                            na.rm = TRUE)
```

**Interpretation:** Worst-case underestimation during peak hours (in MW).

**Target:** Keep below operational safety margin (area-specific, typically < 500 MW for large systems)

---

### 7.3 Evaluation Report Structure

**Console Output:**

```
MAPE: 2.45 %
MAPE Noite: 3.12 %
Maior Desvio Ponta: 287.5 MW
Viés médio no horário crítico (MW): -45.2
Número de dias com subestimativa: 4
Subestimativa máxima no horário crítico (MW): 287.5
```

**Visual Output:**

```r
grafico <- ggplot() +
  geom_line(data = avaliacao, aes(DataHora, CargaGlobal, color = "Verificado")) +
  geom_line(data = avaliacao, aes(DataHora, Previsto_denorm, color = "Previsto")) +
  scale_x_datetime(date_labels = "%H:%M") +
  facet_wrap(~ Data, scales = "free_x") +
  labs(
    title = paste0(ANO, "/", MES),
    subtitle = paste0("MAPE: ", round(mape_diatodo, 2), "% | MAPE Noite: ", round(mape_noite, 2), "%")
  ) +
  theme_bw()

ggsave(filename = paste0("Resultados/", Area, "_", ANO, MES, "_v3.png"), 
       plot = grafico, width = 10, height = 8, dpi = 600)
```

**Features:**
- Faceted by day for easy daily pattern inspection
- Dual-line plot (actual vs. predicted)
- Prominent MAPE display
- High-resolution export for presentations

**Python Requirements:**
- Matplotlib/Plotly for interactive visualizations
- Faceted plots with daily breakdowns
- Export to PNG/HTML with configurable DPI

---

## 8. Model Persistence and Versioning

### 8.1 Model Serialization

**Current R Implementation:** No explicit serialization in provided code.

**Required for Python:**

```python
import joblib
import json
from pathlib import Path

class LGBMModelArtifact:
    """Container for model and metadata."""
    
    def __init__(self, model, scaler, metadata):
        self.model = model  # LightGBM Booster
        self.scaler = scaler  # MinMaxScaler
        self.metadata = metadata  # Dict with training info
    
    def save(self, path: Path):
        """Save model, scaler, and metadata."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(path / "model.txt"))
        
        # Save scaler
        joblib.dump(self.scaler, path / "scaler.pkl")
        
        # Save metadata
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load model from disk."""
        import lightgbm as lgb
        
        model = lgb.Booster(model_file=str(path / "model.txt"))
        scaler = joblib.load(path / "scaler.pkl")
        
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        return cls(model, scaler, metadata)
```

---

### 8.2 Metadata Schema

**Required Information:**

```python
metadata = {
    "model_version": "1.0.0",
    "area": "SECO",
    "trained_at": "2024-11-17T10:30:00",
    "training_period": {
        "start": "2022-10-01",
        "end": "2024-03-31"
    },
    "validation_period": {
        "start": "2024-04-01",
        "end": "2024-04-30"
    },
    "hyperparameters": {
        "max_leaves": 16,
        "min_data_in_leaf": 50,
        "learning_rate": 0.1,
        "num_iterations": 5000,
        "peak_weight_factor": 1.0
    },
    "performance": {
        "validation_weighted_mape": 2.45,
        "validation_mape_overall": 2.12,
        "validation_mape_peak": 3.10
    },
    "features": {
        "count": 105,
        "names": ["CargaGlobal_anterior_BLF", "CarWav_W1_1", ...],
        "categorical": ["Feriado", "PreFeriado", ...]
    },
    "normalization": {
        "method": "minmax",
        "target_min": 12345.67,
        "target_max": 67890.12
    }
}
```

---

### 8.3 Model Registry Pattern

**Directory Structure:**

```
models/
├── lgbm/
│   ├── SECO/
│   │   ├── v1.0.0_2024-11-17/
│   │   │   ├── model.txt
│   │   │   ├── scaler.pkl
│   │   │   ├── metadata.json
│   │   │   └── training_history.csv
│   │   ├── v1.1.0_2024-12-01/
│   │   └── latest -> v1.1.0_2024-12-01/
│   ├── SP/
│   └── ...
```

**Semantic Versioning:**
- **Major (v1.0.0 → v2.0.0):** Breaking changes (feature set change, algorithm change)
- **Minor (v1.0.0 → v1.1.0):** Backward-compatible improvements (hyperparameter tuning, more training data)
- **Patch (v1.0.0 → v1.0.1):** Bug fixes (no retraining, code fixes only)

---

## 9. Key Implementation Decisions for Epic-03

### 9.1 Critical Success Factors

**1. Custom Objective Function Must Be Preserved**
- Asymmetric loss is **core to model behavior**
- Without it, model tends to underestimate during peak (operationally dangerous)
- Python implementation must match R gradient/Hessian logic exactly

**2. BLF Strategy Requires Careful Implementation**
- Proportional adjustment logic is non-trivial
- Must handle edge cases (division by zero, negative predictions)
- Temperature BLF is secondary but improves D+1 accuracy

**3. Weighted Training Is Essential**
- Peak hour weighting is what makes the model operationally useful
- Current implementation uses `fator_penalidade=1` (no actual weighting)
- **Recommendation:** Enable `fator_penalidade=2` for production

**4. Post-Processing Smoothing May Be Optional**
- Two-stage LOESS adds complexity
- Should be configurable (some areas may not need it)
- Consider simpler alternatives (moving average, Savitzky-Golay filter)

**5. Evaluation Must Focus on Peak Period**
- Overall MAPE is secondary
- Peak MAPE and underestimation metrics are primary business KPIs
- Reporting framework must highlight these

---

### 9.2 Recommended Architecture (Python)

**Class Structure:**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import lightgbm as lgb

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    max_leaves: int = 16
    min_data_in_leaf: int = 50
    learning_rate: float = 0.1
    num_iterations: int = 5000
    early_stopping_rounds: int = 50
    peak_weight_factor: float = 1.0

class BaseModel(ABC):
    """Abstract base for all models."""
    
    @abstractmethod
    def train(self, X_train, y_train, X_valid, y_valid):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass

class LGBMBLFModel(BaseModel):
    """LGBM with BLF strategy and custom objective."""
    
    def __init__(self, config: TrainingConfig, scaler=None):
        self.config = config
        self.scaler = scaler or MinMaxScaler()
        self.model = None
        self.metadata = {}
    
    def _custom_objective(self, preds, train_data):
        """Asymmetric loss favoring overestimation during peak."""
        labels = train_data.get_label()
        weights = train_data.get_weight()
        
        residuals = preds - labels
        
        # Apply asymmetric penalty
        grad = np.where(
            (residuals < 0) & (weights > 1),
            weights * residuals,
            residuals
        )
        
        hess = np.where(
            (residuals < 0) & (weights > 1),
            weights,
            np.ones_like(residuals)
        )
        
        return grad, hess
    
    def _weighted_mape(self, preds, valid_data):
        """Evaluation metric matching training objective."""
        labels = valid_data.get_label()
        weights = valid_data.get_weight()
        
        errors = np.abs((labels - preds) / np.maximum(np.abs(labels), 1e-6))
        weighted_mape = np.sum(weights * errors) / len(errors) * 100
        
        return 'weighted_mape', weighted_mape, False
    
    def train(self, X_train, y_train, X_valid, y_valid, 
              train_weights=None, valid_weights=None):
        """Train LGBM with custom objective."""
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train_scaled, 
            label=y_train, 
            weight=train_weights
        )
        
        valid_data = lgb.Dataset(
            X_valid_scaled, 
            label=y_valid, 
            weight=valid_weights,
            reference=train_data
        )
        
        # Train
        params = {
            'max_leaves': self.config.max_leaves,
            'min_data_in_leaf': self.config.min_data_in_leaf,
            'learning_rate': self.config.learning_rate,
            'verbose': -1
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.num_iterations,
            valid_sets=[valid_data],
            fobj=self._custom_objective,
            feval=self._weighted_mape,
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds),
                lgb.log_evaluation(100)
            ]
        )
        
        return self
    
    def predict(self, X):
        """Predict with normalization."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_blf(self, X, verified_load, verified_hours=8):
        """Predict with BLF intraday correction."""
        # Get raw prediction
        raw_pred = self.predict(X)
        
        # Calculate proportions
        proportions = np.concatenate([[1.0], raw_pred[1:] / raw_pred[:-1]])
        
        # Propagate from verified values
        n_verified = verified_hours * 2  # Semi-hourly
        blf_pred = np.zeros_like(raw_pred)
        blf_pred[:n_verified] = verified_load
        
        for i in range(n_verified, len(raw_pred)):
            blf_pred[i] = blf_pred[i-1] * proportions[i]
        
        return blf_pred
    
    def save(self, path):
        """Persist model artifacts."""
        artifact = LGBMModelArtifact(self.model, self.scaler, self.metadata)
        artifact.save(Path(path))
    
    def load(self, path):
        """Load model from disk."""
        artifact = LGBMModelArtifact.load(Path(path))
        self.model = artifact.model
        self.scaler = artifact.scaler
        self.metadata = artifact.metadata
        return self
```

---

### 9.3 Testing Strategy

**Unit Tests:**
- Custom objective gradient/Hessian correctness
- Weighted MAPE calculation matches formula
- BLF propagation logic with synthetic data
- Normalization/denormalization round-trip

**Integration Tests:**
- Full training pipeline on small dataset (1 month)
- Prediction pipeline with and without BLF
- Model save/load preserves predictions
- Comparison with R output (tolerance ±0.1%)

**Regression Tests:**
- Freeze R model predictions as "golden dataset"
- Run Python model on same input features
- Assert predictions within ±0.5% MAPE difference

---

## 10. Open Questions and Recommendations

### 10.1 Open Questions for Epic Refinement

**1. Hyperparameter Tuning Scope**
- Current R code has no actual tuning (single grid point)
- Should Python implementation include Optuna-based HPO?
- Budget: How many trials? (Recommendation: 50-100)

**2. Peak Weight Factor**
- Current: `fator_penalidade = 1` (no effect)
- Should default be 1.5 or 2.0 for production?
- Make configurable per area?

**3. BLF Verification Hour**
- Current: 8:00 AM cutoff
- Should this be configurable (e.g., 6:00 or 10:00)?
- Impact on operational workflow?

**4. Post-Processing Smoothing**
- Two-stage LOESS adds 10-15% runtime
- Is it necessary for all areas?
- Alternative: Single-stage with span=0.2?

**5. Model Retraining Frequency**
- How often should models be retrained? (Weekly? Monthly?)
- Incremental training or full retrain?
- Drift detection triggers?

---

### 10.2 Recommendations for Epic-03

**Priority 1 (Must Have):**
1. ✅ Implement custom objective function with exact R logic
2. ✅ Implement weighted MAPE evaluation metric
3. ✅ Implement BLF strategy for D+0 and D+1
4. ✅ Implement training/validation/test split with dynamic dates
5. ✅ Implement normalization pipeline with serialization
6. ✅ Implement model persistence with versioning

**Priority 2 (Should Have):**
7. ✅ Add hyperparameter optimization (Optuna)
8. ✅ Add post-prediction LOESS smoothing (configurable)
9. ✅ Add comprehensive evaluation metrics (peak focus)
10. ✅ Add visualization utilities (faceted plots)
11. ✅ Add feature importance analysis

**Priority 3 (Nice to Have):**
12. ⚪ Add SHAP explanations for predictions
13. ⚪ Add online learning / incremental training
14. ⚪ Add prediction confidence intervals
15. ⚪ Add automatic drift detection

---

## 11. Performance Requirements

### 11.1 Training Performance

**Target Benchmarks (per area, single model):**

| Metric | Target | Notes |
|--------|--------|-------|
| Training time | < 30 min | ~2 years of data, 35K rows, 105 features |
| Memory usage | < 8 GB | Training + validation sets in memory |
| Model size | < 50 MB | Serialized booster |

**Optimization Strategies:**
- Use LightGBM native categorical feature handling (avoid one-hot)
- Enable GPU acceleration if available (`device='gpu'`)
- Use early stopping to prevent overtraining
- Parallelize hyperparameter search across trials

---

### 11.2 Prediction Performance

**Target Benchmarks:**

| Operation | Target | Notes |
|-----------|--------|-------|
| Predict 48 values (1 day) | < 100 ms | Single area |
| BLF adjustment | < 50 ms | Proportional propagation |
| LOESS smoothing | < 200 ms | Per-day LOWESS |
| **Total (intraday update)** | **< 5 min** | All 26 areas in parallel |

**Optimization Strategies:**
- Vectorize BLF calculations (avoid Python loops)
- Cache LOWESS computations where possible
- Parallelize across areas using multiprocessing/Dask
- Use compiled libraries (NumPy, SciPy) for smoothing

---

## 12. Integration with Epic-02 (Feature Engineering)

### 12.1 Feature Dependencies

**Required Features from Epic-02 Pipeline:**

| Feature | Plugin | Priority | Notes |
|---------|--------|----------|-------|
| `CargaGlobal_anterior_BLF` | LagFeaturesPlugin | Critical | 24h lag after LOESS smoothing |
| `TemperaturaECMWF_anterior_BLF` | LagFeaturesPlugin | Critical | 24h lag after LOESS smoothing |
| `Temperatura_prevista_BLF` | - | Critical | D+1 forecast (external source) |
| Wavelet features | WaveletTransformPlugin | High | ~64 features total |
| `Sazonalidade` | SeasonalityPlugin | High | Intraday pattern by day type |
| `Hora_seno` | TemporalFeaturesPlugin | Medium | Cyclical hour encoding |
| Holiday dummies | HolidayFeaturesPlugin | Medium | 3 features |
| Day/Month dummies | DummyEncoderPlugin | Medium | ~17 features |
| Special day dummies | SpecialDaysPlugin | Low | 6 features |
| `GerMMGD` | - | Medium | Distributed generation (external) |

**Total:** ~105 features (matches current R implementation)

---

### 12.2 Feature Pipeline Contract

**Interface:**

```python
from prevcarga.features import FeaturePipeline

# Epic-02 provides
pipeline = FeaturePipeline.from_config("configs/features_lgbm.yaml")

# Epic-03 uses
def train_lgbm(area: str, start_date: str, end_date: str):
    # Load raw data
    raw_data = load_raw_data(area, start_date, end_date)
    
    # Generate features
    features = pipeline.transform(raw_data)
    
    # Separate target
    X = features.drop(columns=['CargaGlobal', 'DataHora', 'Data'])
    y = features['CargaGlobal']
    
    # Train model
    model = LGBMBLFModel(config)
    model.train(X_train, y_train, X_valid, y_valid)
    
    return model
```

**Epic-02 Deliverables Required:**
1. ✅ `FeaturePipeline` class with `.transform()` method
2. ✅ YAML configuration for feature plugins
3. ✅ Feature name consistency (exact column names)
4. ✅ Categorical feature list for LGBM
5. ✅ Feature metadata (types, descriptions)

---

## 13. Acceptance Criteria for Epic-03

### 13.1 Functional Requirements

**FR-1: Training**
- ✅ Train LGBM model on historical data (2+ years)
- ✅ Use custom weighted objective function
- ✅ Perform hyperparameter optimization (manual or automated)
- ✅ Achieve validation MAPE within ±10% of R baseline

**FR-2: Prediction**
- ✅ Predict D+0 (intraday from 8:00 AM) with BLF strategy
- ✅ Predict D+1 (full day) using corrected D+0 lags
- ✅ Denormalize predictions to MW scale
- ✅ (Optional) Apply post-prediction smoothing

**FR-3: Evaluation**
- ✅ Calculate overall MAPE
- ✅ Calculate peak period MAPE (18h-22h)
- ✅ Calculate bias and underestimation metrics
- ✅ Generate faceted visualization plots

**FR-4: Persistence**
- ✅ Save trained model with metadata
- ✅ Load model for prediction
- ✅ Version models semantically
- ✅ Track model lineage

---

### 13.2 Non-Functional Requirements

**NFR-1: Performance**
- ✅ Training: < 30 min per area
- ✅ Prediction: < 5 min for all 26 areas
- ✅ Memory: < 8 GB during training

**NFR-2: Accuracy**
- ✅ Validation MAPE within ±5% of R baseline
- ✅ Peak period MAPE < 4% (stretch goal: < 3%)
- ✅ Peak underestimation bias > -50 MW

**NFR-3: Code Quality**
- ✅ Unit test coverage > 80%
- ✅ Integration tests for full pipeline
- ✅ Type hints on all public functions
- ✅ Docstrings with examples

**NFR-4: Documentation**
- ✅ API documentation (Sphinx)
- ✅ Training guide with examples
- ✅ Model card template
- ✅ Troubleshooting guide

---

## 14. Appendix: Code Comparison (R vs Python)

### 14.1 Custom Objective Function

**R Implementation:**

```r
objetivo_customizado <- function(preds, dtrain) {
  labels <- rotulos_treino_global
  pesos <- attr(dtrain, "weight")
  if (is.null(pesos)) {
    pesos <- rep(1, length(labels))
  }
  
  res <- preds - labels
  grad <- ifelse((res < 0) & (pesos > 1), pesos * res, res)
  hess <- ifelse((res < 0) & (pesos > 1), pesos, 1)
  
  return(list(grad = grad, hess = hess))
}
```

**Python Equivalent:**

```python
def custom_objective(preds: np.ndarray, train_data: lgb.Dataset):
    """Asymmetric loss for underestimation penalty."""
    labels = train_data.get_label()
    weights = train_data.get_weight()
    
    if weights is None:
        weights = np.ones(len(labels))
    
    residuals = preds - labels
    
    # Gradient (first derivative)
    grad = np.where(
        (residuals < 0) & (weights > 1),
        weights * residuals,
        residuals
    )
    
    # Hessian (second derivative)
    hess = np.where(
        (residuals < 0) & (weights > 1),
        weights,
        np.ones_like(residuals)
    )
    
    return grad, hess
```

---

### 14.2 BLF Strategy

**R Implementation:**

```r
diferencas_horarias <- c(1, round(previsao_dia_atual[-1] / previsao_dia_atual[-length(previsao_dia_atual)], 7))

for (j in seq_len(length(previsao_dia_atual))[-seq_len(nrow(valores_verificados))]) {
  proximo_valor <- valores_completos$CargaGlobal_real_BLF[nrow(valores_completos)] * diferencas_horarias[j]
  valores_completos <- rbind(valores_completos, data.table(...))
}
```

**Python Equivalent:**

```python
def apply_blf(raw_prediction: np.ndarray, 
              verified_load: np.ndarray,
              verified_hours: int = 8) -> np.ndarray:
    """Apply Best Linear Forecast intraday correction."""
    # Calculate proportions
    proportions = np.concatenate([[1.0], raw_prediction[1:] / raw_prediction[:-1]])
    
    # Initialize with verified values
    n_verified = verified_hours * 2  # Semi-hourly
    blf_forecast = np.zeros(len(raw_prediction))
    blf_forecast[:n_verified] = verified_load
    
    # Propagate using proportions
    for i in range(n_verified, len(raw_prediction)):
        blf_forecast[i] = blf_forecast[i-1] * proportions[i]
    
    return blf_forecast
```

---

**Document End**

*This analysis provides a complete blueprint for Epic-03 implementation of the LGBM model with BLF strategy. All algorithms, configurations, and business logic are documented from the R production system.*
