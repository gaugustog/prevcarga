# Epic-04: Model Layer - Hierarchical Models

**Epic ID:** Epic-04  
**Epic Name:** Model Layer - Hierarchical Models  
**Phase:** Phase 4  
**Duration:** 3 weeks  
**Dependencies:** Epic-03 (Model Layer - End-to-End Models)  
**Priority:** High  

---

## 🎯 Epic Overview

### **Business Goal**
Implement hierarchical forecasting models (RegDin+SVM and Holt-Winters) that decompose electric load forecasting into demand mean and profile components, enabling better capture of seasonal patterns and operational constraints while maintaining forecast accuracy across all horizons D+0 to D+8.

### **Technical Goal**
Build a two-stage hierarchical modeling system where demand mean forecasting uses time series models (ARIMA, Holt-Winters) and profile modeling uses machine learning approaches (SVM), with proper orchestration to combine components into final load forecasts that respect temporal dependencies and seasonal patterns.

### **Success Metrics**
- ✅ RegDin+SVM pipeline working
- ✅ Holt-Winters pipeline working
- ✅ Both models forecast D+0 to D+8
- ✅ Hierarchical structure respected
- ✅ Integration tests pass

---

## 🏗️ Technical Architecture

### **Hierarchical Model Components**

```
prevcarga/
├── models/
│   ├── hierarchical/
│   │   ├── __init__.py
│   │   ├── base_hierarchical.py    # BaseHierarchicalModel
│   │   ├── regdin_svm.py          # RegDinSVMModel
│   │   ├── holt_winters.py        # HoltWintersModel
│   │   └── profile_combiner.py    # DemandMean → Profile combination
│   ├── demand_mean/
│   │   ├── __init__.py
│   │   ├── arima_model.py         # ARIMA demand mean forecasting
│   │   ├── holt_winters_dm.py     # Holt-Winters demand mean
│   │   └── demand_processor.py    # Demand mean calculation utilities
│   ├── profile/
│   │   ├── __init__.py
│   │   ├── svm_profile.py         # SVM profile models (48 models)
│   │   ├── hw_profile.py          # Holt-Winters profile models
│   │   └── profile_normalizer.py  # Profile normalization utilities
│   └── orchestration/
│       ├── __init__.py
│       ├── hierarchical_trainer.py   # Coordinated training
│       ├── hierarchical_predictor.py # Two-stage prediction
│       └── pipeline_validator.py     # Hierarchical validation
└── tests/
    └── models/
        └── hierarchical/
            ├── test_regdin_svm.py
            ├── test_holt_winters.py
            ├── test_demand_mean.py
            └── test_profile_models.py
```

### **Hierarchical Architecture Design**

```python
# Abstract base for hierarchical models
class BaseHierarchicalModel(BaseModel):
    """Base class for hierarchical forecasting models."""
    
    def __init__(self):
        self.demand_mean_model = None
        self.profile_models = {}  # {hour: model}
        self.combiner = None
        self._is_fitted = False
    
    @property
    @abstractmethod
    def demand_mean_model_class(self) -> Type:
        """Class for demand mean forecasting."""
        pass
    
    @property
    @abstractmethod
    def profile_model_class(self) -> Type:
        """Class for profile forecasting."""
        pass
    
    @abstractmethod
    def _prepare_demand_mean_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for demand mean modeling."""
        pass
    
    @abstractmethod
    def _prepare_profile_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        demand_mean: pd.Series
    ) -> Dict[int, Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for profile modeling by hour."""
        pass
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: Dict[str, Any]
    ) -> 'BaseHierarchicalModel':
        """Train hierarchical model with demand mean and profiles."""
        
        # Step 1: Train demand mean model
        dm_X, dm_y = self._prepare_demand_mean_data(X, y)
        self.demand_mean_model = self.demand_mean_model_class()
        self.demand_mean_model.fit(dm_X, dm_y, config.get('demand_mean', {}))
        
        # Step 2: Generate demand mean predictions for training
        dm_pred = self.demand_mean_model.predict(dm_X, [0])['pred_h0']
        
        # Step 3: Train profile models
        profile_data = self._prepare_profile_data(X, y, dm_pred)
        
        for hour, (prof_X, prof_y) in profile_data.items():
            if len(prof_y) < 10:  # Minimum samples
                continue
                
            profile_model = self.profile_model_class()
            profile_model.fit(prof_X, prof_y, config.get('profile', {}))
            self.profile_models[hour] = profile_model
        
        # Step 4: Initialize combiner
        self.combiner = ProfileCombiner()
        
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        horizons: List[int]
    ) -> pd.DataFrame:
        """Generate hierarchical predictions."""
        
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Step 1: Predict demand mean
        dm_X, _ = self._prepare_demand_mean_data(X, pd.Series(index=X.index))
        dm_predictions = self.demand_mean_model.predict(dm_X, horizons)
        
        # Step 2: Predict profiles for each horizon
        final_predictions = {}
        
        for horizon in horizons:
            dm_col = f'pred_h{horizon}'
            if dm_col not in dm_predictions.columns:
                continue
            
            # Get demand mean for this horizon
            demand_mean = dm_predictions[dm_col]
            
            # Predict profiles for each semi-hourly period
            hourly_profiles = {}
            
            for hour in range(48):  # 48 semi-hourly periods
                if hour in self.profile_models:
                    # Prepare profile features for this hour
                    prof_X = self._prepare_profile_features(X, hour, horizon)
                    
                    # Predict profile
                    profile_pred = self.profile_models[hour].predict(prof_X, [0])
                    hourly_profiles[hour] = profile_pred['pred_h0']
            
            # Combine demand mean with profiles
            combined_forecast = self.combiner.combine(
                demand_mean=demand_mean,
                profiles=hourly_profiles,
                horizon=horizon
            )
            
            final_predictions[f'pred_h{horizon}'] = combined_forecast
        
        return pd.DataFrame(final_predictions, index=X.index)
```

---

## 📋 User Stories

### **User Story 1: ARIMA Demand Mean Model**
**As a** time series analyst  
**I want** ARIMA model for demand mean forecasting  
**So that** I can capture long-term trends and seasonal patterns in daily energy demand

#### **Acceptance Criteria:**
- [ ] Implements ARIMA using statsforecast for robust time series modeling
- [ ] Automatically selects optimal (p,d,q) parameters using information criteria
- [ ] Handles seasonal ARIMA (SARIMA) for weekly and yearly patterns
- [ ] Processes daily aggregated load data (24-hour means)
- [ ] Generates forecasts for D+0 to D+8 horizons
- [ ] Provides prediction intervals with confidence bounds

#### **Technical Implementation:**
```python
class ARIMADemandMeanModel(BaseModel):
    """ARIMA model for demand mean forecasting using statsforecast."""
    
    def __init__(self):
        self.model = None
        self.seasonal_periods = [7, 365.25]  # Weekly, yearly
        self.fitted_params = {}
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "arima_demand_mean"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return list(range(0, 9))  # D+0 to D+8
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: Dict[str, Any]
    ) -> 'ARIMADemandMeanModel':
        """Train ARIMA model on daily demand mean data."""
        
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA
        
        # Convert to daily demand mean
        daily_demand = self._convert_to_daily_mean(y)
        
        # Prepare data for statsforecast
        sf_data = pd.DataFrame({
            'unique_id': 'demand_mean',
            'ds': daily_demand.index,
            'y': daily_demand.values
        })
        
        # Configure AutoARIMA
        arima_config = config.get('arima', {})
        model = AutoARIMA(
            season_length=arima_config.get('season_length', 7),
            stepwise=arima_config.get('stepwise', True),
            approximation=arima_config.get('approximation', False),
            max_p=arima_config.get('max_p', 5),
            max_q=arima_config.get('max_q', 5),
            max_d=arima_config.get('max_d', 2),
            max_P=arima_config.get('max_P', 2),
            max_Q=arima_config.get('max_Q', 2),
            max_D=arima_config.get('max_D', 1),
        )
        
        # Create StatsForecast instance
        sf = StatsForecast(
            models=[model],
            freq='D',  # Daily frequency
            n_jobs=1
        )
        
        # Fit model
        sf.fit(sf_data)
        self.model = sf
        
        # Store fitted parameters for inspection
        self.fitted_params = {
            'model_type': 'AutoARIMA',
            'training_periods': len(daily_demand),
            'last_training_date': daily_demand.index[-1],
            'seasonal_patterns': self.seasonal_periods
        }
        
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        horizons: List[int]
    ) -> pd.DataFrame:
        """Generate ARIMA demand mean forecasts."""
        
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Determine maximum horizon for forecasting
        max_horizon = max(horizons)
        
        # Generate forecasts
        forecasts = self.model.forecast(h=max_horizon + 1, level=[80, 95])
        
        # Convert back to semi-hourly resolution and align with request
        predictions = {}
        
        for horizon in horizons:
            if horizon <= len(forecasts) - 1:
                # Get daily forecast for this horizon
                daily_forecast = forecasts.iloc[horizon]['AutoARIMA']
                
                # Replicate to semi-hourly (assuming uniform distribution)
                semi_hourly_forecast = pd.Series(
                    [daily_forecast] * len(X), 
                    index=X.index
                )
                
                predictions[f'pred_h{horizon}'] = semi_hourly_forecast
                
                # Add confidence intervals if available
                if 'AutoARIMA-lo-80' in forecasts.columns:
                    predictions[f'pred_h{horizon}_lo80'] = pd.Series(
                        [forecasts.iloc[horizon]['AutoARIMA-lo-80']] * len(X),
                        index=X.index
                    )
                    predictions[f'pred_h{horizon}_hi80'] = pd.Series(
                        [forecasts.iloc[horizon]['AutoARIMA-hi-80']] * len(X),
                        index=X.index
                    )
        
        return pd.DataFrame(predictions, index=X.index)
    
    def _convert_to_daily_mean(self, semi_hourly_data: pd.Series) -> pd.Series:
        """Convert semi-hourly data to daily means."""
        
        # Group by date and calculate mean
        daily_mean = semi_hourly_data.groupby(semi_hourly_data.index.date).mean()
        
        # Convert index back to datetime
        daily_mean.index = pd.to_datetime(daily_mean.index)
        
        return daily_mean
```

#### **Definition of Done:**
- [ ] ARIMA model trains successfully on daily aggregated data
- [ ] Automatic parameter selection works for different seasonal patterns
- [ ] Forecasts provide reasonable demand mean estimates for all horizons
- [ ] Confidence intervals accurately reflect forecast uncertainty

---

### **User Story 2: SVM Profile Models**
**As a** ML engineer  
**I want** SVM models for semi-hourly profile forecasting  
**So that** I can capture intraday patterns and their variations around demand mean

#### **Acceptance Criteria:**
- [ ] Implements 48 SVM models (one per semi-hourly period)
- [ ] Each model predicts profile ratio (load / demand_mean)
- [ ] Uses RBF kernel with hyperparameter optimization
- [ ] Handles categorical features (weekday, holiday, season)
- [ ] Includes temporal features (hour, day type, calendar effects)
- [ ] Ensures profile ratios sum to reasonable daily totals

#### **Technical Implementation:**
```python
class SVMProfileModel(BaseModel):
    """SVM model for semi-hourly profile forecasting."""
    
    def __init__(self):
        self.models = {}  # {hour: SVM model}
        self.scalers = {}  # {hour: StandardScaler}
        self.profile_statistics = {}  # {hour: {'mean', 'std', 'bounds'}}
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "svm_profile"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return [0]  # SVM profiles predict current period only
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: Dict[str, Any]
    ) -> 'SVMProfileModel':
        """Train SVM models for each semi-hourly period."""
        
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV
        
        # SVM configuration
        svm_config = config.get('svm', {})
        optimize_hyperparams = svm_config.get('optimize_hyperparams', True)
        
        # Prepare profile data grouped by hour
        profile_data = self._prepare_profile_data(X, y)
        
        for hour in range(48):  # 48 semi-hourly periods
            if hour not in profile_data:
                continue
            
            hour_X, hour_y = profile_data[hour]
            
            if len(hour_y) < 50:  # Minimum samples for SVM
                continue
            
            # Feature scaling
            scaler = StandardScaler()
            hour_X_scaled = scaler.fit_transform(hour_X)
            
            # SVM model
            if optimize_hyperparams:
                # Hyperparameter optimization
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'epsilon': [0.01, 0.1, 0.2]
                }
                
                svm = SVR(kernel='rbf')
                
                grid_search = GridSearchCV(
                    svm,
                    param_grid,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                grid_search.fit(hour_X_scaled, hour_y)
                model = grid_search.best_estimator_
                
            else:
                # Use default parameters
                model = SVR(
                    kernel='rbf',
                    C=svm_config.get('C', 1.0),
                    gamma=svm_config.get('gamma', 'scale'),
                    epsilon=svm_config.get('epsilon', 0.1)
                )
                model.fit(hour_X_scaled, hour_y)
            
            # Store model and scaler
            self.models[hour] = model
            self.scalers[hour] = scaler
            
            # Store profile statistics for validation
            self.profile_statistics[hour] = {
                'mean': hour_y.mean(),
                'std': hour_y.std(),
                'min': hour_y.min(),
                'max': hour_y.max(),
                'q25': hour_y.quantile(0.25),
                'q75': hour_y.quantile(0.75)
            }
        
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        horizons: List[int]
    ) -> pd.DataFrame:
        """Generate SVM profile predictions."""
        
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        
        for horizon in horizons:
            if horizon != 0:  # SVM only predicts current period
                continue
            
            # Predict profile for each semi-hourly period
            profile_preds = []
            
            for i, timestamp in enumerate(X.index):
                hour = timestamp.hour * 2 + (timestamp.minute // 30)
                
                if hour in self.models:
                    # Prepare features for this timestamp
                    hour_X = self._prepare_timestamp_features(X.iloc[[i]], timestamp)
                    hour_X_scaled = self.scalers[hour].transform(hour_X)
                    
                    # Predict profile ratio
                    profile_pred = self.models[hour].predict(hour_X_scaled)[0]
                    
                    # Apply bounds based on training statistics
                    stats = self.profile_statistics[hour]
                    profile_pred = np.clip(
                        profile_pred,
                        stats['q25'] - 2 * stats['std'],  # Lower bound
                        stats['q75'] + 2 * stats['std']   # Upper bound
                    )
                    
                    profile_preds.append(profile_pred)
                else:
                    # Use average profile if no model for this hour
                    avg_profile = np.mean([
                        stats['mean'] for stats in self.profile_statistics.values()
                    ])
                    profile_preds.append(avg_profile)
            
            predictions[f'pred_h{horizon}'] = profile_preds
        
        return pd.DataFrame(predictions, index=X.index)
    
    def _prepare_profile_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Prepare training data grouped by semi-hourly periods."""
        
        profile_data = {}
        
        for hour in range(48):
            # Filter data for this semi-hourly period
            mask = (X.index.hour * 2 + (X.index.minute // 30)) == hour
            
            if mask.sum() == 0:
                continue
            
            hour_X = X[mask]
            hour_y = y[mask]
            
            # Prepare features for this hour
            features = self._prepare_hour_features(hour_X)
            
            profile_data[hour] = (features, hour_y.values)
        
        return profile_data
    
    def _prepare_hour_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare features for a specific hour."""
        
        features = []
        
        for idx, row in X.iterrows():
            feature_vector = [
                idx.dayofweek,  # Day of week
                idx.day,        # Day of month
                idx.month,      # Month
                idx.quarter,    # Quarter
                int(idx.dayofweek >= 5),  # Weekend flag
                # Add weather features if available
                row.get('temperature', 20),  # Default temperature
                row.get('humidity', 50),     # Default humidity
                # Add calendar features if available
                row.get('is_holiday', 0),    # Holiday flag
                row.get('is_bridge_day', 0), # Bridge day flag
            ]
            features.append(feature_vector)
        
        return np.array(features)
```

#### **Definition of Done:**
- [ ] 48 SVM models train successfully for all semi-hourly periods
- [ ] Hyperparameter optimization improves profile prediction accuracy
- [ ] Profile ratios are bounded and realistic (e.g., 0.5-2.0 range)
- [ ] Models handle seasonal and calendar variations correctly

---

### **User Story 3: RegDin+SVM Hierarchical Pipeline**
**As a** forecasting engineer  
**I want** complete RegDin+SVM hierarchical model  
**So that** I can combine ARIMA demand mean with SVM profiles for accurate load forecasting

#### **Acceptance Criteria:**
- [ ] Integrates ARIMA demand mean with SVM profile models
- [ ] Orchestrates two-stage training (demand mean → profiles)
- [ ] Combines predictions maintaining energy conservation
- [ ] Handles horizon-specific demand mean forecasting
- [ ] Provides interpretable decomposition (mean vs profile components)
- [ ] Validates hierarchical consistency

#### **Technical Implementation:**
```python
class RegDinSVMModel(BaseHierarchicalModel):
    """RegDin+SVM hierarchical model combining ARIMA and SVM."""
    
    @property
    def name(self) -> str:
        return "regdin_svm"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return list(range(0, 9))  # D+0 to D+8
    
    @property
    def demand_mean_model_class(self) -> Type:
        return ARIMADemandMeanModel
    
    @property
    def profile_model_class(self) -> Type:
        return SVMProfileModel
    
    def _prepare_demand_mean_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare daily aggregated data for ARIMA demand mean."""
        
        # Aggregate to daily means
        daily_X = X.groupby(X.index.date).first()
        daily_y = y.groupby(y.index.date).mean()
        
        # Align indices
        daily_X.index = pd.to_datetime(daily_X.index)
        daily_y.index = pd.to_datetime(daily_y.index)
        
        return daily_X, daily_y
    
    def _prepare_profile_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        demand_mean: pd.Series
    ) -> Dict[int, Tuple[pd.DataFrame, pd.Series]]:
        """Prepare profile ratios (load / demand_mean) by hour."""
        
        # Calculate daily demand mean for each timestamp
        daily_demand_mean = demand_mean.reindex(
            y.index.date, method='ffill'
        )
        daily_demand_mean.index = y.index
        
        # Calculate profile ratios
        profile_ratios = y / (daily_demand_mean + 1e-8)  # Avoid division by zero
        
        # Group by semi-hourly periods
        profile_data = {}
        
        for hour in range(48):
            # Filter for this semi-hourly period
            mask = (X.index.hour * 2 + (X.index.minute // 30)) == hour
            
            if mask.sum() == 0:
                continue
            
            hour_X = X[mask]
            hour_profiles = profile_ratios[mask]
            
            # Remove outliers (beyond 3 std from mean)
            mean_ratio = hour_profiles.mean()
            std_ratio = hour_profiles.std()
            
            outlier_mask = np.abs(hour_profiles - mean_ratio) <= 3 * std_ratio
            
            clean_X = hour_X[outlier_mask]
            clean_profiles = hour_profiles[outlier_mask]
            
            if len(clean_profiles) > 10:  # Minimum samples
                profile_data[hour] = (clean_X, clean_profiles)
        
        return profile_data
    
    def predict(
        self, 
        X: pd.DataFrame, 
        horizons: List[int]
    ) -> pd.DataFrame:
        """Generate RegDin+SVM hierarchical predictions."""
        
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        
        # Step 1: Get demand mean forecasts
        dm_X, _ = self._prepare_demand_mean_data(X, pd.Series(index=X.index))
        dm_forecasts = self.demand_mean_model.predict(dm_X, horizons)
        
        # Step 2: Generate profile predictions and combine
        for horizon in horizons:
            dm_col = f'pred_h{horizon}'
            
            if dm_col not in dm_forecasts.columns:
                continue
            
            # Get demand mean forecast for this horizon (daily value)
            demand_mean_daily = dm_forecasts[dm_col].iloc[0]  # Same for all hours
            
            # Get profile predictions for each timestamp
            load_forecasts = []
            
            for timestamp in X.index:
                hour = timestamp.hour * 2 + (timestamp.minute // 30)
                
                # Get profile prediction for this hour
                if hour in self.profile_models:
                    # Prepare features for profile prediction
                    hour_X = X.loc[[timestamp]]
                    
                    # Get profile ratio prediction
                    profile_pred = self.profile_models[hour].predict(
                        hour_X, [0]
                    )['pred_h0'].iloc[0]
                    
                    # Combine: load = demand_mean * profile_ratio
                    load_forecast = demand_mean_daily * profile_pred
                else:
                    # Use average profile if no model available
                    avg_profile = 1.0  # Neutral profile
                    load_forecast = demand_mean_daily * avg_profile
                
                load_forecasts.append(load_forecast)
            
            predictions[f'pred_h{horizon}'] = load_forecasts
            
            # Add decomposition components for interpretability
            predictions[f'demand_mean_h{horizon}'] = [demand_mean_daily] * len(X)
        
        return pd.DataFrame(predictions, index=X.index)
```

#### **Definition of Done:**
- [ ] RegDin+SVM pipeline trains and predicts for all horizons
- [ ] Energy conservation maintained (daily sum ≈ demand mean × 48)
- [ ] Decomposition provides interpretable mean vs profile components
- [ ] Performance comparable to end-to-end models for accuracy baseline

---

### **User Story 4: Holt-Winters Hierarchical Model**
**As a** time series modeler  
**I want** Holt-Winters hierarchical model for seasonal forecasting  
**So that** I can capture trend, seasonal, and irregular components separately

#### **Acceptance Criteria:**
- [ ] Implements Holt-Winters for both demand mean and profiles
- [ ] Handles multiple seasonal patterns (daily, weekly, yearly)
- [ ] Uses exponential smoothing for trend and seasonal components
- [ ] Provides additive and multiplicative seasonal options
- [ ] Generates prediction intervals using simulation
- [ ] Validates seasonal pattern consistency

#### **Technical Implementation:**
```python
class HoltWintersModel(BaseHierarchicalModel):
    """Holt-Winters hierarchical model for seasonal forecasting."""
    
    @property
    def name(self) -> str:
        return "holt_winters"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return list(range(0, 9))  # D+0 to D+8
    
    @property
    def demand_mean_model_class(self) -> Type:
        return HoltWintersDemandMeanModel
    
    @property
    def profile_model_class(self) -> Type:
        return HoltWintersProfileModel

class HoltWintersDemandMeanModel(BaseModel):
    """Holt-Winters model for demand mean forecasting."""
    
    def __init__(self):
        self.model = None
        self.seasonal_periods = 7  # Weekly seasonality for daily data
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "holt_winters_demand_mean"
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: Dict[str, Any]
    ) -> 'HoltWintersDemandMeanModel':
        """Train Holt-Winters model on daily demand mean."""
        
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Convert to daily means
        daily_demand = self._convert_to_daily_mean(y)
        
        # Holt-Winters configuration
        hw_config = config.get('holt_winters', {})
        
        # Fit Holt-Winters model
        self.model = ExponentialSmoothing(
            daily_demand,
            trend=hw_config.get('trend', 'add'),
            seasonal=hw_config.get('seasonal', 'add'),
            seasonal_periods=hw_config.get('seasonal_periods', self.seasonal_periods),
            damped_trend=hw_config.get('damped_trend', False)
        ).fit(
            smoothing_level=hw_config.get('alpha'),
            smoothing_trend=hw_config.get('beta'),
            smoothing_seasonal=hw_config.get('gamma'),
            damping_trend=hw_config.get('phi')
        )
        
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        horizons: List[int]
    ) -> pd.DataFrame:
        """Generate Holt-Winters demand mean forecasts."""
        
        max_horizon = max(horizons)
        
        # Generate forecasts with prediction intervals
        forecasts = self.model.forecast(steps=max_horizon + 1)
        prediction_intervals = self.model.get_prediction(
            start=len(self.model.fittedvalues),
            end=len(self.model.fittedvalues) + max_horizon
        ).conf_int()
        
        predictions = {}
        
        for horizon in horizons:
            if horizon < len(forecasts):
                # Point forecast
                predictions[f'pred_h{horizon}'] = pd.Series(
                    [forecasts.iloc[horizon]] * len(X), 
                    index=X.index
                )
                
                # Prediction intervals
                predictions[f'pred_h{horizon}_lower'] = pd.Series(
                    [prediction_intervals.iloc[horizon, 0]] * len(X),
                    index=X.index
                )
                predictions[f'pred_h{horizon}_upper'] = pd.Series(
                    [prediction_intervals.iloc[horizon, 1]] * len(X),
                    index=X.index
                )
        
        return pd.DataFrame(predictions, index=X.index)

class HoltWintersProfileModel(BaseModel):
    """Holt-Winters model for profile forecasting."""
    
    def __init__(self):
        self.models = {}  # {hour: HW model}
        self.seasonal_periods = 7  # Weekly seasonality
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "holt_winters_profile"
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: Dict[str, Any]
    ) -> 'HoltWintersProfileModel':
        """Train Holt-Winters models for each semi-hourly period."""
        
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Group data by semi-hourly periods
        profile_data = self._prepare_profile_data(X, y)
        
        hw_config = config.get('holt_winters_profile', {})
        
        for hour, hour_data in profile_data.items():
            if len(hour_data) < 3 * self.seasonal_periods:  # Need multiple seasons
                continue
            
            try:
                # Fit Holt-Winters model for this hour
                model = ExponentialSmoothing(
                    hour_data,
                    trend=hw_config.get('trend', 'add'),
                    seasonal=hw_config.get('seasonal', 'add'),
                    seasonal_periods=self.seasonal_periods
                ).fit()
                
                self.models[hour] = model
                
            except Exception as e:
                # Skip this hour if fitting fails
                continue
        
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        horizons: List[int]
    ) -> pd.DataFrame:
        """Generate Holt-Winters profile predictions."""
        
        predictions = {}
        
        for horizon in horizons:
            if horizon != 0:  # Profiles only predict current period
                continue
            
            profile_preds = []
            
            for timestamp in X.index:
                hour = timestamp.hour * 2 + (timestamp.minute // 30)
                
                if hour in self.models:
                    # Predict next value for this hour
                    pred = self.models[hour].forecast(steps=1)[0]
                    profile_preds.append(pred)
                else:
                    # Use overall mean if no model
                    profile_preds.append(1.0)
            
            predictions[f'pred_h{horizon}'] = profile_preds
        
        return pd.DataFrame(predictions, index=X.index)
```

#### **Definition of Done:**
- [ ] Holt-Winters models capture seasonal patterns for demand mean and profiles
- [ ] Multiple seasonal periods (daily, weekly) handled appropriately
- [ ] Prediction intervals provide uncertainty quantification
- [ ] Seasonal components remain consistent and interpretable

---

### **User Story 5: Hierarchical Pipeline Orchestration**
**As a** ML engineer  
**I want** coordinated training and prediction orchestration  
**So that** hierarchical models train efficiently and predictions maintain consistency

#### **Acceptance Criteria:**
- [ ] Orchestrates multi-stage training (demand mean first, then profiles)
- [ ] Manages data flow between hierarchical components
- [ ] Validates hierarchical consistency (energy conservation)
- [ ] Supports parallel training of profile models
- [ ] Handles missing or failed component models gracefully
- [ ] Provides comprehensive logging and monitoring

#### **Technical Implementation:**
```python
class HierarchicalTrainer:
    """Orchestrate hierarchical model training."""
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs
        
    def train_hierarchical_model(
        self,
        model_class: Type[BaseHierarchicalModel],
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any]
    ) -> BaseHierarchicalModel:
        """Train complete hierarchical model with orchestration."""
        
        logger.info(f"Starting hierarchical training for {model_class.__name__}")
        
        # Create model instance
        model = model_class()
        
        try:
            # Stage 1: Train demand mean model
            logger.info("Stage 1: Training demand mean model")
            dm_X, dm_y = model._prepare_demand_mean_data(X, y)
            
            model.demand_mean_model = model.demand_mean_model_class()
            model.demand_mean_model.fit(
                dm_X, dm_y, config.get('demand_mean', {})
            )
            
            # Generate demand mean predictions for profile training
            dm_predictions = model.demand_mean_model.predict(dm_X, [0])
            dm_pred = dm_predictions['pred_h0']
            
            # Stage 2: Train profile models
            logger.info("Stage 2: Training profile models")
            profile_data = model._prepare_profile_data(X, y, dm_pred)
            
            if self.n_jobs == 1:
                # Sequential training
                for hour, (prof_X, prof_y) in profile_data.items():
                    model.profile_models[hour] = self._train_single_profile_model(
                        model.profile_model_class, prof_X, prof_y, 
                        config.get('profile', {}), hour
                    )
            else:
                # Parallel training
                from concurrent.futures import ProcessPoolExecutor
                
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = {}
                    
                    for hour, (prof_X, prof_y) in profile_data.items():
                        future = executor.submit(
                            self._train_single_profile_model,
                            model.profile_model_class, prof_X, prof_y,
                            config.get('profile', {}), hour
                        )
                        futures[future] = hour
                    
                    # Collect results
                    for future in futures:
                        hour = futures[future]
                        try:
                            profile_model = future.result()
                            model.profile_models[hour] = profile_model
                        except Exception as e:
                            logger.warning(f"Profile model training failed for hour {hour}: {e}")
            
            # Stage 3: Initialize combiner
            model.combiner = ProfileCombiner()
            model._is_fitted = True
            
            # Stage 4: Validate hierarchical consistency
            self._validate_hierarchical_model(model, X, y)
            
            logger.info("Hierarchical training completed successfully")
            
        except Exception as e:
            logger.error(f"Hierarchical training failed: {e}")
            raise
        
        return model
    
    @staticmethod
    def _train_single_profile_model(
        model_class: Type[BaseModel],
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any],
        hour: int
    ) -> BaseModel:
        """Train single profile model (for parallel execution)."""
        
        if len(y) < 10:  # Minimum samples
            raise ValueError(f"Insufficient data for hour {hour}")
        
        model = model_class()
        model.fit(X, y, config)
        return model
    
    def _validate_hierarchical_model(
        self,
        model: BaseHierarchicalModel,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
        """Validate hierarchical model consistency."""
        
        # Generate predictions
        predictions = model.predict(X_val, [0])
        
        if 'pred_h0' not in predictions.columns:
            raise ValueError("Model failed to generate predictions")
        
        # Check energy conservation (daily sums)
        daily_actual = y_val.groupby(y_val.index.date).sum()
        daily_predicted = predictions['pred_h0'].groupby(
            predictions.index.date
        ).sum()
        
        # Calculate relative error in daily totals
        daily_error = np.abs(daily_predicted - daily_actual) / daily_actual
        mean_daily_error = daily_error.mean()
        
        if mean_daily_error > 0.1:  # 10% threshold
            logger.warning(
                f"High daily energy conservation error: {mean_daily_error:.2%}"
            )
        
        # Check profile consistency
        if hasattr(model, 'profile_models'):
            profile_count = len(model.profile_models)
            logger.info(f"Successfully trained {profile_count}/48 profile models")
            
            if profile_count < 24:  # Less than half
                logger.warning("Low profile model coverage - may affect accuracy")

class ProfileCombiner:
    """Combine demand mean forecasts with profile predictions."""
    
    def combine(
        self,
        demand_mean: pd.Series,
        profiles: Dict[int, pd.Series],
        horizon: int
    ) -> pd.Series:
        """Combine demand mean with profiles to generate load forecast."""
        
        combined_forecast = []
        
        for i, timestamp in enumerate(demand_mean.index):
            hour = timestamp.hour * 2 + (timestamp.minute // 30)
            
            # Get demand mean for this timestamp
            dm_value = demand_mean.iloc[i] if i < len(demand_mean) else demand_mean.iloc[-1]
            
            # Get profile ratio
            if hour in profiles and i < len(profiles[hour]):
                profile_ratio = profiles[hour].iloc[i]
            else:
                # Use neutral profile if not available
                profile_ratio = 1.0
            
            # Combine: load = demand_mean * profile_ratio
            load_forecast = dm_value * profile_ratio
            combined_forecast.append(load_forecast)
        
        return pd.Series(combined_forecast, index=demand_mean.index)
```

#### **Definition of Done:**
- [ ] Orchestration coordinates multi-stage training successfully
- [ ] Parallel training reduces total training time by >40%
- [ ] Hierarchical validation detects consistency issues
- [ ] Error handling gracefully manages component failures

---

### **User Story 6: Hierarchical Model Validation and Testing**
**As a** QA engineer  
**I want** comprehensive validation for hierarchical models  
**So that** I can ensure model quality and hierarchical consistency

#### **Acceptance Criteria:**
- [ ] Validates energy conservation (daily sums match)
- [ ] Tests seasonal pattern consistency across components
- [ ] Verifies profile ratios are within reasonable bounds
- [ ] Compares hierarchical vs end-to-end model accuracy
- [ ] Provides decomposition analysis tools
- [ ] Includes stress testing with edge cases

#### **Technical Implementation:**
```python
class HierarchicalValidator:
    """Comprehensive validation for hierarchical models."""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_model(
        self,
        model: BaseHierarchicalModel,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Run comprehensive validation suite."""
        
        results = {}
        
        # Generate predictions
        predictions = model.predict(X_test, [0, 1, 2])
        
        # 1. Energy Conservation Test
        results['energy_conservation'] = self._test_energy_conservation(
            predictions, y_test
        )
        
        # 2. Profile Bounds Test
        results['profile_bounds'] = self._test_profile_bounds(model, X_test)
        
        # 3. Seasonal Consistency Test
        results['seasonal_consistency'] = self._test_seasonal_consistency(
            model, X_test
        )
        
        # 4. Accuracy Comparison
        results['accuracy_metrics'] = self._calculate_accuracy_metrics(
            predictions, y_test
        )
        
        # 5. Component Analysis
        results['component_analysis'] = self._analyze_components(
            model, X_test, y_test
        )
        
        # 6. Stress Testing
        results['stress_tests'] = self._run_stress_tests(model, X_test)
        
        self.validation_results = results
        return results
    
    def _test_energy_conservation(
        self,
        predictions: pd.DataFrame,
        actual: pd.Series
    ) -> Dict[str, float]:
        """Test daily energy conservation."""
        
        results = {}
        
        for col in predictions.columns:
            if col.startswith('pred_h'):
                # Daily sums
                daily_pred = predictions[col].groupby(
                    predictions.index.date
                ).sum()
                daily_actual = actual.groupby(actual.index.date).sum()
                
                # Align dates
                common_dates = daily_pred.index.intersection(daily_actual.index)
                
                if len(common_dates) > 0:
                    pred_aligned = daily_pred.loc[common_dates]
                    actual_aligned = daily_actual.loc[common_dates]
                    
                    # Calculate conservation error
                    relative_error = np.abs(
                        pred_aligned - actual_aligned
                    ) / actual_aligned
                    
                    results[f'{col}_conservation_error'] = relative_error.mean()
        
        return results
    
    def _test_profile_bounds(
        self,
        model: BaseHierarchicalModel,
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Test profile ratio bounds."""
        
        results = {}
        
        if hasattr(model, 'profile_models'):
            for hour, profile_model in model.profile_models.items():
                # Generate profile predictions
                hour_X = X_test[
                    (X_test.index.hour * 2 + (X_test.index.minute // 30)) == hour
                ]
                
                if len(hour_X) > 0:
                    prof_pred = profile_model.predict(hour_X, [0])['pred_h0']
                    
                    results[f'hour_{hour}'] = {
                        'min_profile': prof_pred.min(),
                        'max_profile': prof_pred.max(),
                        'mean_profile': prof_pred.mean(),
                        'std_profile': prof_pred.std(),
                        'outliers_count': (
                            (prof_pred < 0.1) | (prof_pred > 5.0)
                        ).sum()
                    }
        
        return results
    
    def _test_seasonal_consistency(
        self,
        model: BaseHierarchicalModel,
        X_test: pd.DataFrame
    ) -> Dict[str, float]:
        """Test seasonal pattern consistency."""
        
        results = {}
        
        # Test demand mean seasonality
        if hasattr(model, 'demand_mean_model'):
            dm_pred = model.demand_mean_model.predict(X_test, [0])['pred_h0']
            
            # Weekly pattern consistency
            weekly_pattern = dm_pred.groupby(dm_pred.index.dayofweek).mean()
            weekly_std = dm_pred.groupby(dm_pred.index.dayofweek).std()
            
            results['dm_weekly_consistency'] = (weekly_std / weekly_pattern).mean()
        
        # Test profile seasonality
        if hasattr(model, 'profile_models'):
            profile_consistency = []
            
            for hour in range(0, 48, 12):  # Sample every 6 hours
                if hour in model.profile_models:
                    hour_X = X_test[
                        (X_test.index.hour * 2 + (X_test.index.minute // 30)) == hour
                    ]
                    
                    if len(hour_X) > 14:  # At least 2 weeks
                        prof_pred = model.profile_models[hour].predict(
                            hour_X, [0]
                        )['pred_h0']
                        
                        weekly_prof_pattern = prof_pred.groupby(
                            prof_pred.index.dayofweek
                        ).mean()
                        weekly_prof_std = prof_pred.groupby(
                            prof_pred.index.dayofweek
                        ).std()
                        
                        consistency = (weekly_prof_std / weekly_prof_pattern).mean()
                        profile_consistency.append(consistency)
            
            if profile_consistency:
                results['profile_weekly_consistency'] = np.mean(profile_consistency)
        
        return results
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        
        report = ["=== Hierarchical Model Validation Report ===\n"]
        
        if not self.validation_results:
            return "No validation results available"
        
        # Energy Conservation
        report.append("1. Energy Conservation:")
        conservation = self.validation_results.get('energy_conservation', {})
        for horizon, error in conservation.items():
            report.append(f"   {horizon}: {error:.2%} daily error")
        
        # Profile Bounds
        report.append("\n2. Profile Bounds:")
        bounds = self.validation_results.get('profile_bounds', {})
        outlier_hours = [
            hour for hour, stats in bounds.items()
            if stats.get('outliers_count', 0) > 0
        ]
        report.append(f"   Hours with profile outliers: {len(outlier_hours)}/48")
        
        # Seasonal Consistency
        report.append("\n3. Seasonal Consistency:")
        seasonal = self.validation_results.get('seasonal_consistency', {})
        for pattern, consistency in seasonal.items():
            report.append(f"   {pattern}: {consistency:.3f} (lower is better)")
        
        # Accuracy Metrics
        report.append("\n4. Accuracy Metrics:")
        accuracy = self.validation_results.get('accuracy_metrics', {})
        for metric, value in accuracy.items():
            report.append(f"   {metric}: {value:.3f}")
        
        return "\n".join(report)
```

#### **Definition of Done:**
- [ ] Validation suite detects energy conservation violations
- [ ] Profile bounds testing identifies unrealistic predictions
- [ ] Seasonal consistency validation ensures stable patterns
- [ ] Comprehensive reporting enables model debugging

---

## 🔧 Technical Requirements

### **Performance Requirements**
- Training demand mean model: <10 minutes per area
- Training all profile models: <20 minutes per area
- Hierarchical prediction: <10 seconds for all horizons
- Memory usage: <3GB for complete hierarchical model
- Profile model coverage: >80% of semi-hourly periods

### **Data Requirements**
- Input: Feature matrices from Epic-02A and base forecasts from Epic-03
- Training data: Minimum 2 years historical data for seasonal patterns
- Demand mean data: Daily aggregated values with weekly/yearly seasonality
- Profile data: Semi-hourly ratios bounded between 0.1 and 5.0

### **Integration Requirements**
- Uses end-to-end model predictions as baseline comparison
- Compatible with Epic-02A feature engineering pipeline
- Feeds into Epic-05 model combination system
- Supports Epic-06 hierarchical reconciliation

---

## 🧪 Testing Strategy

### **Unit Tests**
- [ ] Demand mean model accuracy on synthetic seasonal data
- [ ] Profile model bounds and ratio validation
- [ ] Hierarchical orchestration component integration
- [ ] Energy conservation mathematical properties

### **Integration Tests**
- [ ] End-to-end hierarchical training with real data
- [ ] Multi-horizon prediction consistency
- [ ] Performance benchmarks vs end-to-end models
- [ ] Robustness testing with missing profile models

### **Validation Tests**
- [ ] Seasonal pattern reproduction accuracy
- [ ] Profile decomposition interpretability
- [ ] Hierarchical consistency across all time series
- [ ] Edge case handling (holidays, extreme weather)

---

## 📊 Quality Gates

### **Code Quality**
- [ ] 80%+ unit test coverage for hierarchical components
- [ ] All models validate energy conservation within 5%
- [ ] Profile models achieve >80% coverage of semi-hourly periods
- [ ] Seasonal patterns remain stable across validation periods

### **Documentation**
- [ ] Hierarchical modeling theory and implementation guide
- [ ] Demand mean vs profile decomposition explanation
- [ ] Validation methodology and interpretation guide
- [ ] Troubleshooting guide for component failures

### **Production Readiness**
- [ ] Hierarchical models handle missing components gracefully
- [ ] Training orchestration scales to multiple areas
- [ ] Model serialization preserves hierarchical structure
- [ ] Integration with monitoring and alerting systems

---

## 🚀 Delivery Plan

### **Week 1: Demand Mean Models (Days 1-5)**
- **Day 1:** ARIMA demand mean model implementation
- **Day 2:** Holt-Winters demand mean model
- **Day 3:** Demand mean data preparation and validation
- **Day 4:** Unit tests and seasonal pattern validation
- **Day 5:** Performance optimization and integration testing

### **Week 2: Profile Models (Days 6-10)**
- **Day 6:** SVM profile models implementation
- **Day 7:** Holt-Winters profile models
- **Day 8:** Profile data preparation and ratio calculation
- **Day 9:** Parallel training optimization
- **Day 10:** Profile bounds validation and testing

### **Week 3: Hierarchical Integration (Days 11-15)**
- **Day 11:** Hierarchical orchestration framework
- **Day 12:** RegDin+SVM complete pipeline
- **Day 13:** Holt-Winters complete pipeline
- **Day 14:** Hierarchical validation suite
- **Day 15:** Integration testing and epic handoff

---

## 🔗 Dependencies and Interfaces

### **Input Dependencies (Epic-03)**
- End-to-end model predictions for baseline comparison
- Model registry and training infrastructure
- Feature engineering pipeline from Epic-02A

### **Output Interfaces (Epic-05, Epic-06)**
- Hierarchical model predictions with decomposition components
- Demand mean and profile forecasts for combination strategies
- Model metadata including seasonal patterns and accuracy metrics

### **External Dependencies**
```yaml
dependencies:
  - statsforecast>=1.5.0     # ARIMA and seasonal modeling
  - statsmodels>=0.14.0      # Holt-Winters exponential smoothing
  - scikit-learn>=1.3.0      # SVM profile models
  - numpy>=1.24.0            # Numerical computations
  - pandas>=2.0.0            # Time series data handling
```

---

## 📈 Success Criteria

### **Functional Success Criteria**
- ✅ RegDin+SVM pipeline working end-to-end for all horizons
- ✅ Holt-Winters pipeline working with seasonal decomposition
- ✅ Both models forecast D+0 to D+8 with hierarchical consistency
- ✅ Hierarchical structure respected (energy conservation within 5%)
- ✅ Integration tests pass with Epic-02A and Epic-03 components

### **Performance Success Criteria**
- ✅ Training time <30 minutes per area for complete hierarchical model
- ✅ Prediction accuracy within 10% of end-to-end models (MAPE)
- ✅ Profile model coverage >80% of semi-hourly periods
- ✅ Memory usage <3GB for complete model with all components

### **Quality Success Criteria**
- ✅ 80%+ test coverage with hierarchical validation
- ✅ Energy conservation validated across all test scenarios
- ✅ Seasonal patterns remain interpretable and stable
- ✅ Documentation enables hierarchical model extension

---

## 🎯 Epic Completion Definition

Epic-04 is considered complete when:

1. **All User Stories Delivered:** 6 user stories implemented with ARIMA, SVM, and Holt-Winters components
2. **Hierarchical Models Functional:** RegDin+SVM and Holt-Winters pipelines working end-to-end
3. **Validation Successful:** Energy conservation and seasonal consistency validated
4. **Performance Targets Met:** Training and prediction performance within acceptable limits
5. **Integration Ready:** Models compatible with combination and reconciliation systems

**Handoff to Epic-05:** Hierarchical models provide alternative forecasting approaches with decomposable components for ensemble combination strategies.

---

## 📋 Related Epics

- **Epic-02A (Core Features):** Provides temporal and seasonal features for profile modeling
- **Epic-03 (End-to-End Models):** Provides baseline predictions and model infrastructure
- **Epic-05 (Model Combination):** Combines hierarchical with end-to-end predictions
- **Epic-06 (Hierarchical Reconciliation):** Uses hierarchical structure for forecast reconciliation
- **Epic-07 (Evaluation):** Evaluates hierarchical vs end-to-end model performance

---

**Epic Owner:** ML Engineering Team (Time Series Specialization)  
**Stakeholders:** Data Science Team, Operations Team, Model Development Team  
**Review Date:** End of Week 3 (Complete hierarchical model demonstration with seasonal validation)