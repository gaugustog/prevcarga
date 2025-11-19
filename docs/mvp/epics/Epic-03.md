# Epic-03: Model Layer - End-to-End Models

**Epic ID:** Epic-03  
**Epic Name:** Model Layer - End-to-End Models  
**Phase:** Phase 3  
**Duration:** 4 weeks  
**Dependencies:** Epic-02A (Core Feature Engineering)  
**Priority:** High  

---

## 🎯 Epic Overview

### **Business Goal**
Implement LGBM and Random Forest models with complete training and prediction pipelines, enabling accurate electric load forecasting for horizons D+0 to D+8 with production-ready model management, versioning, and intraday update capabilities.

### **Technical Goal**
Build a robust model layer that supports end-to-end forecasting workflows, including model training, prediction, serialization, and intraday baseline load forecast (BLF) updates, with a focus on LGBM for short-term horizons and Random Forest for multi-horizon forecasting.

### **Success Metrics**
- ✅ LGBM trains and predicts (D+0, D+1)
- ✅ RF trains and predicts (D+0 to D+8)
- ✅ Intraday BLF strategy works
- ✅ Models save/load with versioning
- ✅ Training parallelization functional

---

## 🏗️ Technical Architecture

### **Core Model Components**

```
prevcarga/
├── models/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── model.py               # BaseModel abstract class
│   │   ├── trainer.py             # UniversalTrainer
│   │   ├── predictor.py           # BasePredictior
│   │   └── registry.py            # ModelRegistry
│   ├── end_to_end/
│   │   ├── __init__.py
│   │   ├── lgbm_model.py          # LGBMModel implementation
│   │   ├── random_forest.py       # RandomForestModel implementation
│   │   └── blf_predictor.py       # BLFPredictor for intraday
│   ├── serialization/
│   │   ├── __init__.py
│   │   ├── model_serializer.py    # Model save/load with metadata
│   │   └── version_manager.py     # Semantic versioning system
│   └── training/
│       ├── __init__.py
│       ├── trainer_config.py      # Training configuration schemas
│       ├── parallel_trainer.py    # Multi-area parallel training
│       └── validation.py          # Model validation utilities
└── tests/
    └── models/
        ├── test_lgbm_model.py
        ├── test_random_forest.py
        ├── test_training.py
        └── test_serialization.py
```

### **Model Architecture Design**

```python
# Abstract base for all forecasting models
class BaseModel(ABC):
    """Base class for all forecasting models."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for identification."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Model version (semantic versioning)."""
        pass
    
    @property
    @abstractmethod
    def supported_horizons(self) -> List[int]:
        """List of supported forecast horizons."""
        pass
    
    @abstractmethod
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: Dict[str, Any]
    ) -> 'BaseModel':
        """Train the model on provided data."""
        pass
    
    @abstractmethod
    def predict(
        self, 
        X: pd.DataFrame, 
        horizons: List[int]
    ) -> pd.DataFrame:
        """Generate predictions for specified horizons."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass
```

---

## 📋 User Stories

### **User Story 1: Base Model Interface and Registry**
**As a** ML engineer  
**I want** a standardized model interface with registry system  
**So that** I can manage multiple models consistently with versioning and metadata

#### **Acceptance Criteria:**
- [ ] `BaseModel` abstract class defines common interface for all models
- [ ] `ModelRegistry` manages model registration and discovery
- [ ] Semantic versioning system tracks model evolution
- [ ] Model metadata includes training timestamp, performance metrics, feature dependencies
- [ ] Registry supports model comparison and selection
- [ ] Thread-safe model loading and caching

#### **Technical Implementation:**
```python
class ModelRegistry:
    """Central registry for model management."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.loaded_models = {}
        self._lock = threading.Lock()
    
    def register_model(
        self, 
        model: BaseModel, 
        metadata: Dict[str, Any]
    ) -> str:
        """Register a trained model with metadata."""
        model_id = f"{model.name}_{model.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._lock:
            # Save model and metadata
            model_path = self.storage_path / model_id
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Serialize model
            model.save(model_path / "model.pkl")
            
            # Save metadata
            full_metadata = {
                **metadata,
                'model_id': model_id,
                'registered_at': datetime.now().isoformat(),
                'model_class': model.__class__.__name__,
                'supported_horizons': model.supported_horizons,
                'feature_dependencies': getattr(model, 'feature_dependencies', [])
            }
            
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(full_metadata, f, indent=2)
        
        return model_id
    
    def load_model(self, model_id: str) -> BaseModel:
        """Load model from registry."""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        model_path = self.storage_path / model_id
        
        # Load metadata
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Dynamically load model class
        model_class = self._get_model_class(metadata['model_class'])
        model = model_class.load(model_path / "model.pkl")
        
        with self._lock:
            self.loaded_models[model_id] = model
        
        return model
```

#### **Definition of Done:**
- [ ] Registry manages models with unique identifiers and metadata
- [ ] Model loading is thread-safe and cached for performance
- [ ] Versioning system prevents conflicts and enables rollback
- [ ] Metadata provides complete model lineage and dependencies

---

### **User Story 2: LGBM Model Implementation**
**As a** forecasting engineer  
**I want** LGBM model for short-term load forecasting (D+0, D+1)  
**So that** I can achieve high accuracy for critical near-term predictions

#### **Acceptance Criteria:**
- [ ] Implements LightGBM with custom objective function for load forecasting
- [ ] Supports D+0 and D+1 horizon predictions
- [ ] Includes hyperparameter optimization with Optuna
- [ ] Handles categorical features and missing values appropriately
- [ ] Provides feature importance analysis and SHAP values
- [ ] Integrates with BLF strategy for intraday updates

#### **Technical Implementation:**
```python
class LGBMModel(BaseModel):
    """LightGBM model for short-term electric load forecasting."""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.training_config = {}
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "lgbm"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_horizons(self) -> List[int]:
        return [0, 1]  # D+0, D+1
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: Dict[str, Any]
    ) -> 'LGBMModel':
        """Train LGBM model with hyperparameter optimization."""
        import lightgbm as lgb
        import optuna
        
        self.feature_names = X.columns.tolist()
        self.training_config = config
        
        # Prepare data
        train_data = lgb.Dataset(X, label=y)
        
        # Hyperparameter optimization
        if config.get('optimize_hyperparams', True):
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self._objective(trial, train_data, config),
                n_trials=config.get('n_trials', 100)
            )
            best_params = study.best_params
        else:
            best_params = config.get('lgbm_params', self._default_params())
        
        # Train final model
        self.model = lgb.train(
            params=best_params,
            train_set=train_data,
            num_boost_round=config.get('num_boost_round', 1000),
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        horizons: List[int]
    ) -> pd.DataFrame:
        """Generate predictions for D+0 and D+1."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Validate horizons
        invalid_horizons = [h for h in horizons if h not in self.supported_horizons]
        if invalid_horizons:
            raise ValueError(f"Unsupported horizons: {invalid_horizons}")
        
        # Generate predictions
        predictions = {}
        base_pred = self.model.predict(X)
        
        for horizon in horizons:
            if horizon == 0:
                predictions[f'pred_h{horizon}'] = base_pred
            elif horizon == 1:
                # Apply horizon-specific adjustment
                horizon_adjustment = self._calculate_horizon_adjustment(X, horizon)
                predictions[f'pred_h{horizon}'] = base_pred * horizon_adjustment
        
        return pd.DataFrame(predictions, index=X.index)
    
    def _objective(self, trial, train_data, config):
        """Optuna objective function for hyperparameter optimization."""
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        
        # Cross-validation
        cv_results = lgb.cv(
            params=params,
            train_set=train_data,
            num_boost_round=500,
            nfold=5,
            stratified=False,
            shuffle=True,
            seed=42,
            return_cvbooster=False,
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        return cv_results['valid mae-mean'][-1]
```

#### **Definition of Done:**
- [ ] LGBM model trains successfully with hyperparameter optimization
- [ ] D+0 and D+1 predictions achieve target accuracy (MAPE < 3%)
- [ ] Feature importance provides interpretable insights
- [ ] Model serialization and loading work correctly

---

### **User Story 3: Random Forest Multi-Horizon Model**
**As a** forecasting analyst  
**I want** Random Forest model for multi-horizon forecasting (D+0 to D+8)  
**So that** I can generate consistent predictions across all forecast horizons

#### **Acceptance Criteria:**
- [ ] Implements 216 Random Forest models (24 areas × 9 horizons)
- [ ] Each horizon uses appropriate feature selection to prevent leakage
- [ ] Supports parallel training across areas and horizons
- [ ] Includes out-of-bag error estimation and feature importance
- [ ] Handles seasonal patterns and calendar effects
- [ ] Provides prediction confidence intervals

#### **Technical Implementation:**
```python
class RandomForestModel(BaseModel):
    """Random Forest model for multi-horizon electric load forecasting."""
    
    def __init__(self):
        self.models = {}  # {horizon: sklearn.RandomForestRegressor}
        self.feature_selectors = {}  # {horizon: feature_selector}
        self.selected_features = {}  # {horizon: [feature_names]}
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        return "random_forest"
    
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
    ) -> 'RandomForestModel':
        """Train Random Forest models for each horizon."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import SelectFromModel
        
        n_jobs = config.get('n_jobs', -1)
        n_estimators = config.get('n_estimators', 100)
        
        for horizon in self.supported_horizons:
            # Create horizon-specific target
            y_horizon = y.shift(-horizon * 48)  # 48 semi-hourly periods per day
            
            # Get horizon-safe features (prevent data leakage)
            safe_features = self._get_horizon_safe_features(X.columns, horizon)
            X_horizon = X[safe_features]
            
            # Align data and remove NaN
            common_idx = X_horizon.dropna().index.intersection(y_horizon.dropna().index)
            X_clean = X_horizon.loc[common_idx]
            y_clean = y_horizon.loc[common_idx]
            
            if len(X_clean) < 100:  # Minimum samples for training
                continue
            
            # Feature selection for this horizon
            rf_selector = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                n_jobs=n_jobs
            )
            
            selector = SelectFromModel(
                estimator=rf_selector,
                max_features=config.get('max_features_per_horizon', 50)
            )
            
            X_selected = selector.fit_transform(X_clean, y_clean)
            selected_feature_names = X_clean.columns[selector.get_support()].tolist()
            
            # Train final model with selected features
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=config.get('max_depth', None),
                min_samples_split=config.get('min_samples_split', 2),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                max_features=config.get('max_features', 'sqrt'),
                random_state=42,
                n_jobs=n_jobs,
                oob_score=True
            )
            
            model.fit(X_selected, y_clean)
            
            # Store model and metadata
            self.models[horizon] = model
            self.feature_selectors[horizon] = selector
            self.selected_features[horizon] = selected_feature_names
        
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        horizons: List[int]
    ) -> pd.DataFrame:
        """Generate predictions for specified horizons."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        
        for horizon in horizons:
            if horizon not in self.models:
                continue
            
            # Get selected features for this horizon
            selected_features = self.selected_features[horizon]
            X_horizon = X[selected_features]
            
            # Apply feature selection transform
            selector = self.feature_selectors[horizon]
            X_transformed = selector.transform(X_horizon)
            
            # Generate predictions
            model = self.models[horizon]
            pred = model.predict(X_transformed)
            
            predictions[f'pred_h{horizon}'] = pred
            
            # Add confidence intervals using prediction std from trees
            if hasattr(model, 'estimators_'):
                tree_preds = np.array([
                    tree.predict(X_transformed) 
                    for tree in model.estimators_
                ])
                pred_std = np.std(tree_preds, axis=0)
                predictions[f'pred_h{horizon}_std'] = pred_std
        
        return pd.DataFrame(predictions, index=X.index)
    
    def _get_horizon_safe_features(self, all_features: List[str], horizon: int) -> List[str]:
        """Get features that don't cause data leakage for given horizon."""
        safe_features = []
        
        for feature in all_features:
            # Skip features that would cause leakage
            if any([
                'lag_' in feature and int(feature.split('lag_')[1].split('h')[0]) <= horizon * 24,
                'future_' in feature,
                f'h{horizon}' in feature and 'lag' not in feature
            ]):
                continue
            
            safe_features.append(feature)
        
        return safe_features
```

#### **Definition of Done:**
- [ ] Random Forest trains for all 9 horizons (D+0 to D+8)
- [ ] Feature selection prevents temporal leakage for each horizon
- [ ] Out-of-bag scores validate model performance
- [ ] Confidence intervals provide uncertainty quantification

---

### **User Story 4: BLF (Baseline Load Forecast) Intraday Predictor**
**As a** operations engineer  
**I want** intraday BLF updates using real-time observations  
**So that** I can improve forecast accuracy during the operational day

#### **Acceptance Criteria:**
- [ ] Updates predictions using latest available observations
- [ ] Calculates correction factors based on recent forecast errors
- [ ] Supports sliding window updates throughout the day
- [ ] Handles missing real-time data gracefully
- [ ] Provides confidence bounds for corrected forecasts
- [ ] Integrates seamlessly with existing model predictions

#### **Technical Implementation:**
```python
class BLFPredictor:
    """Baseline Load Forecast predictor for intraday updates."""
    
    def __init__(self, base_model: BaseModel):
        self.base_model = base_model
        self.correction_history = []
        self.error_model = None
    
    def update_with_observations(
        self,
        observations: pd.Series,
        base_predictions: pd.DataFrame,
        update_time: datetime
    ) -> pd.DataFrame:
        """Update predictions using latest observations."""
        
        # Calculate recent forecast errors
        recent_errors = self._calculate_recent_errors(
            observations, base_predictions, update_time
        )
        
        # Update error correction model
        self._update_error_model(recent_errors)
        
        # Generate corrected predictions
        corrected_predictions = self._apply_corrections(
            base_predictions, update_time
        )
        
        return corrected_predictions
    
    def _calculate_recent_errors(
        self,
        observations: pd.Series,
        predictions: pd.DataFrame,
        update_time: datetime
    ) -> pd.DataFrame:
        """Calculate forecast errors for recent time periods."""
        
        # Get observations up to current time
        obs_cutoff = observations.loc[:update_time]
        
        errors = {}
        for col in predictions.columns:
            if col.startswith('pred_h'):
                horizon = int(col.split('pred_h')[1])
                
                # Align predictions with observations (account for horizon)
                pred_times = obs_cutoff.index - pd.Timedelta(hours=horizon * 24)
                aligned_preds = predictions.loc[pred_times.intersection(predictions.index), col]
                aligned_obs = obs_cutoff.loc[aligned_preds.index + pd.Timedelta(hours=horizon * 24)]
                
                # Calculate errors
                error = aligned_obs - aligned_preds
                errors[f'error_h{horizon}'] = error
        
        return pd.DataFrame(errors)
    
    def _update_error_model(self, recent_errors: pd.DataFrame):
        """Update error correction model using recent errors."""
        from sklearn.linear_model import Ridge
        
        # Use simple ridge regression for error modeling
        for col in recent_errors.columns:
            if len(recent_errors[col].dropna()) < 10:  # Need minimum samples
                continue
            
            # Features: time-based patterns
            error_data = recent_errors[col].dropna()
            X_error = pd.DataFrame({
                'hour': error_data.index.hour,
                'dayofweek': error_data.index.dayofweek,
                'hour_sin': np.sin(2 * np.pi * error_data.index.hour / 24),
                'hour_cos': np.cos(2 * np.pi * error_data.index.hour / 24),
            })
            
            # Train error correction model
            if self.error_model is None:
                self.error_model = {}
            
            error_model = Ridge(alpha=1.0)
            error_model.fit(X_error, error_data)
            self.error_model[col] = error_model
    
    def _apply_corrections(
        self,
        base_predictions: pd.DataFrame,
        update_time: datetime
    ) -> pd.DataFrame:
        """Apply error corrections to base predictions."""
        
        corrected = base_predictions.copy()
        
        if self.error_model is None:
            return corrected
        
        for col in base_predictions.columns:
            error_col = col.replace('pred_', 'error_')
            
            if error_col not in self.error_model:
                continue
            
            # Generate correction features
            X_correction = pd.DataFrame({
                'hour': base_predictions.index.hour,
                'dayofweek': base_predictions.index.dayofweek,
                'hour_sin': np.sin(2 * np.pi * base_predictions.index.hour / 24),
                'hour_cos': np.cos(2 * np.pi * base_predictions.index.hour / 24),
            }, index=base_predictions.index)
            
            # Apply correction
            correction = self.error_model[error_col].predict(X_correction)
            corrected[col] = base_predictions[col] + correction
            
            # Add corrected flag
            corrected[f'{col}_blf_corrected'] = True
        
        return corrected
```

#### **Definition of Done:**
- [ ] BLF corrections improve intraday forecast accuracy by >10%
- [ ] Updates process in <30 seconds with real-time data
- [ ] Error correction model adapts to changing patterns
- [ ] Integration with operations workflow validated

---

### **User Story 5: Model Serialization and Version Management**
**As a** MLOps engineer  
**I want** robust model serialization with version control  
**So that** I can deploy, rollback, and manage models in production

#### **Acceptance Criteria:**
- [ ] Models serialize with complete metadata and dependencies
- [ ] Semantic versioning tracks model evolution and compatibility
- [ ] Serialization includes feature names, preprocessing steps, and performance metrics
- [ ] Supports model comparison and rollback capabilities
- [ ] Handles large models efficiently with compression
- [ ] Validates model integrity on loading

#### **Technical Implementation:**
```python
class ModelSerializer:
    """Handle model serialization with metadata and versioning."""
    
    @staticmethod
    def save_model(
        model: BaseModel,
        path: Path,
        metadata: Dict[str, Any]
    ):
        """Save model with complete metadata."""
        import joblib
        import hashlib
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model object
        model_path = path / "model.pkl"
        joblib.dump(model, model_path, compress=3)
        
        # Calculate model hash for integrity
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        
        # Prepare complete metadata
        complete_metadata = {
            **metadata,
            'model_class': model.__class__.__name__,
            'model_name': model.name,
            'model_version': model.version,
            'saved_at': datetime.now().isoformat(),
            'model_hash': model_hash,
            'supported_horizons': model.supported_horizons,
            'feature_names': getattr(model, 'feature_names', []),
            'training_config': getattr(model, 'training_config', {}),
            'serializer_version': "1.0.0"
        }
        
        # Save metadata
        with open(path / "metadata.json", 'w') as f:
            json.dump(complete_metadata, f, indent=2, default=str)
        
        # Save model-specific artifacts
        if hasattr(model, 'save_artifacts'):
            model.save_artifacts(path)
    
    @staticmethod
    def load_model(path: Path) -> Tuple[BaseModel, Dict[str, Any]]:
        """Load model with integrity validation."""
        import joblib
        import hashlib
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Validate model integrity
        model_path = path / "model.pkl"
        current_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        
        if current_hash != metadata['model_hash']:
            raise ValueError(f"Model integrity check failed: {path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load model-specific artifacts
        if hasattr(model, 'load_artifacts'):
            model.load_artifacts(path)
        
        return model, metadata

class VersionManager:
    """Manage model versions with semantic versioning."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        
    def create_version(
        self,
        model_name: str,
        version_type: str = "patch"  # major, minor, patch
    ) -> str:
        """Create new version number."""
        
        latest_version = self.get_latest_version(model_name)
        
        if latest_version is None:
            return "1.0.0"
        
        major, minor, patch = map(int, latest_version.split('.'))
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        elif version_type == "patch":
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get latest version for model."""
        
        model_dirs = [
            d for d in self.models_dir.iterdir()
            if d.is_dir() and d.name.startswith(model_name)
        ]
        
        if not model_dirs:
            return None
        
        versions = []
        for model_dir in model_dirs:
            try:
                with open(model_dir / "metadata.json", 'r') as f:
                    metadata = json.load(f)
                versions.append(metadata['model_version'])
            except:
                continue
        
        if not versions:
            return None
        
        # Sort versions semantically
        versions.sort(key=lambda v: tuple(map(int, v.split('.'))))
        return versions[-1]
```

#### **Definition of Done:**
- [ ] Model serialization preserves complete state and metadata
- [ ] Version management supports semantic versioning and rollback
- [ ] Integrity validation prevents corrupted model loading
- [ ] Serialization handles models >1GB efficiently

---

### **User Story 6: Universal Trainer with Parallelization**
**As a** training engineer  
**I want** a universal trainer that supports parallel model training  
**So that** I can efficiently train models across multiple areas and horizons

#### **Acceptance Criteria:**
- [ ] Supports training multiple areas in parallel
- [ ] Handles different model types with unified interface
- [ ] Implements checkpointing and resume capabilities
- [ ] Provides progress tracking and logging
- [ ] Manages memory efficiently during parallel training
- [ ] Includes automated hyperparameter optimization

#### **Technical Implementation:**
```python
class UniversalTrainer:
    """Universal trainer supporting parallel model training."""
    
    def __init__(
        self,
        n_jobs: int = -1,
        checkpoint_dir: Optional[Path] = None
    ):
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.checkpoint_dir = checkpoint_dir
        
    def train_multiple_areas(
        self,
        model_class: Type[BaseModel],
        training_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        config: Dict[str, Any]
    ) -> Dict[str, BaseModel]:
        """Train models for multiple areas in parallel."""
        
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # Prepare training jobs
        training_jobs = []
        for area_name, (X, y) in training_data.items():
            job = {
                'area_name': area_name,
                'model_class': model_class,
                'X': X,
                'y': y,
                'config': config,
                'checkpoint_dir': self.checkpoint_dir
            }
            training_jobs.append(job)
        
        # Execute parallel training
        trained_models = {}
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit jobs
            future_to_area = {
                executor.submit(self._train_single_area, job): job['area_name']
                for job in training_jobs
            }
            
            # Collect results
            for future in as_completed(future_to_area):
                area_name = future_to_area[future]
                
                try:
                    model = future.result()
                    trained_models[area_name] = model
                    logger.info(f"Training completed for area: {area_name}")
                    
                except Exception as e:
                    logger.error(f"Training failed for area {area_name}: {e}")
                    
        return trained_models
    
    @staticmethod
    def _train_single_area(job: Dict[str, Any]) -> BaseModel:
        """Train model for single area (used in parallel execution)."""
        
        area_name = job['area_name']
        model_class = job['model_class']
        X, y = job['X'], job['y']
        config = job['config']
        checkpoint_dir = job['checkpoint_dir']
        
        # Create model instance
        model = model_class()
        
        # Check for existing checkpoint
        if checkpoint_dir:
            checkpoint_path = checkpoint_dir / f"{area_name}_{model.name}_{model.version}.pkl"
            if checkpoint_path.exists():
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                model = joblib.load(checkpoint_path)
                if getattr(model, '_is_fitted', False):
                    return model
        
        # Train model
        logger.info(f"Starting training for {area_name} with {model.name}")
        
        try:
            model.fit(X, y, config)
            
            # Save checkpoint
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"{area_name}_{model.name}_{model.version}.pkl"
                joblib.dump(model, checkpoint_path)
                
        except Exception as e:
            logger.error(f"Training failed for {area_name}: {e}")
            raise
            
        return model
    
    def train_with_hyperopt(
        self,
        model_class: Type[BaseModel],
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        n_trials: int = 100
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """Train model with hyperparameter optimization."""
        
        import optuna
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # Train model with parameters
            model = model_class()
            config = {'optimize_hyperparams': False, **params}
            
            # Cross-validation
            cv_scores = self._cross_validate_model(model, X, y, config)
            return np.mean(cv_scores)
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Train final model with best parameters
        best_model = model_class()
        best_config = {'optimize_hyperparams': False, **study.best_params}
        best_model.fit(X, y, best_config)
        
        return best_model, study.best_params
    
    def _cross_validate_model(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        config: Dict[str, Any],
        cv_folds: int = 5
    ) -> List[float]:
        """Perform cross-validation for model evaluation."""
        
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_absolute_error
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and train model
            model_clone = model.__class__()
            model_clone.fit(X_train, y_train, config)
            
            # Predict and score
            predictions = model_clone.predict(X_val, [0])
            score = mean_absolute_error(y_val, predictions['pred_h0'])
            scores.append(score)
        
        return scores
```

#### **Definition of Done:**
- [ ] Parallel training reduces total training time by >50%
- [ ] Checkpoint and resume functionality works correctly
- [ ] Hyperparameter optimization improves model performance
- [ ] Memory usage remains stable during parallel execution

---

## 🔧 Technical Requirements

### **Performance Requirements**
- Training 1 area + 1 model: <30 minutes
- LGBM prediction: <1 second for 1 day ahead
- Random Forest prediction: <5 seconds for all horizons
- BLF update: <30 seconds with real-time data
- Model loading: <10 seconds from storage

### **Data Requirements**
- Input: Feature matrices from Epic-02A (core features)
- Training data: Minimum 1 year historical data per area
- Real-time data: Semi-hourly observations for BLF updates
- Missing data tolerance: Up to 2% for training, 0% for prediction

### **Integration Requirements**
- Compatible with Epic-02A feature pipeline output
- Feeds into Epic-04 hierarchical models as base forecasts
- Supports Epic-02B advanced features when available
- Integrates with production prediction workflows

---

## 🧪 Testing Strategy

### **Unit Tests**
- [ ] Model interface compliance for LGBM and Random Forest
- [ ] Prediction accuracy on synthetic data with known patterns
- [ ] Serialization/deserialization round-trip validation
- [ ] BLF correction accuracy under various error scenarios

### **Integration Tests**
- [ ] End-to-end training with Epic-02A features
- [ ] Multi-area parallel training performance
- [ ] Model registry operations (save, load, compare)
- [ ] Real-time BLF updates with simulated data

### **Performance Tests**
- [ ] Training time benchmarks for all model types
- [ ] Memory usage profiling during parallel training
- [ ] Prediction latency under production loads
- [ ] Model size and loading time optimization

---

## 📊 Quality Gates

### **Code Quality**
- [ ] 85%+ unit test coverage across all model components
- [ ] All model predictions validated against baseline accuracy
- [ ] Hyperparameter optimization converges consistently
- [ ] Error handling covers all failure modes

### **Documentation**
- [ ] Model development guide with examples
- [ ] Training configuration reference
- [ ] BLF strategy operational procedures
- [ ] Performance tuning recommendations

### **Production Readiness**
- [ ] Models handle production data volumes efficiently
- [ ] Parallel training scales to available compute resources
- [ ] Model versioning supports rollback scenarios
- [ ] Integration with monitoring and alerting systems

---

## 🚀 Delivery Plan

### **Week 1: Core Infrastructure (Days 1-5)**
- **Day 1:** Base model interface and registry system
- **Day 2:** Model serialization and version management
- **Day 3:** Universal trainer framework
- **Day 4:** LGBM model implementation
- **Day 5:** Unit tests and basic integration

### **Week 2: Random Forest Implementation (Days 6-10)**
- **Day 6:** Random Forest multi-horizon model
- **Day 7:** Feature selection and leakage prevention
- **Day 8:** Parallel training optimization
- **Day 9:** Performance testing and optimization
- **Day 10:** Integration with Epic-02A features

### **Week 3: BLF and Advanced Features (Days 11-15)**
- **Day 11:** BLF predictor implementation
- **Day 12:** Intraday update workflows
- **Day 13:** Hyperparameter optimization
- **Day 14:** Model comparison and selection tools
- **Day 15:** End-to-end testing with real data

### **Week 4: Production Readiness (Days 16-20)**
- **Day 16:** Performance optimization and caching
- **Day 17:** Error handling and recovery
- **Day 18:** Documentation and examples
- **Day 19:** Integration testing with downstream epics
- **Day 20:** Final validation and epic handoff

---

## 🔗 Dependencies and Interfaces

### **Input Dependencies (Epic-02A)**
- Feature matrices with temporal, calendar, lag, and cyclical features
- Feature pipeline composer for consistent feature generation
- Data validation framework ensuring data quality

### **Output Interfaces (Epic-04, Epic-05)**
- Trained model objects with standardized prediction interface
- Base forecast predictions for hierarchical model inputs
- Model performance metadata for combination strategies

### **External Dependencies**
```yaml
dependencies:
  - lightgbm>=4.0.0        # LGBM model implementation
  - scikit-learn>=1.3.0    # Random Forest and utilities
  - optuna>=3.0.0          # Hyperparameter optimization
  - joblib>=1.3.0          # Model serialization
  - shap>=0.42.0           # Model interpretability
```

---

## 📈 Success Criteria

### **Functional Success Criteria**
- ✅ LGBM trains and predicts for D+0, D+1 with MAPE <3%
- ✅ Random Forest trains and predicts for D+0 to D+8 with MAPE <5%
- ✅ Intraday BLF strategy improves forecast accuracy by >10%
- ✅ Models save/load with complete versioning and metadata
- ✅ Training parallelization reduces total time by >50%

### **Performance Success Criteria**
- ✅ Training 1 area + 1 model: <30 minutes
- ✅ Prediction latency: <5 seconds for all horizons
- ✅ Model loading: <10 seconds from storage
- ✅ BLF updates: <30 seconds with real-time data

### **Quality Success Criteria**
- ✅ 85%+ test coverage with comprehensive integration tests
- ✅ Model predictions validated against baseline accuracy
- ✅ Production deployment supports 26 time series
- ✅ Documentation enables model extension and maintenance

---

## 🎯 Epic Completion Definition

Epic-03 is considered complete when:

1. **All User Stories Delivered:** 6 user stories implemented with LGBM, Random Forest, and BLF capabilities
2. **Model Performance Validated:** Accuracy targets met for all supported horizons
3. **Production Infrastructure Ready:** Serialization, versioning, and parallel training operational
4. **Integration Successful:** Models consume Epic-02A features and provide inputs for Epic-04
5. **BLF Operations Functional:** Intraday updates improve forecast accuracy in production scenarios

**Handoff to Epic-04:** End-to-end models provide base forecasts for hierarchical model development, with established patterns for model training, versioning, and deployment.

---

## 📋 Related Epics

- **Epic-02A (Core Features):** Provides essential features for model training
- **Epic-02B (Advanced Features):** Enhances model performance with sophisticated transformations
- **Epic-04 (Hierarchical Models):** Uses end-to-end predictions as base forecasts
- **Epic-05 (Model Combination):** Combines predictions from multiple end-to-end models
- **Epic-08 (Orchestration):** Coordinates model training and prediction workflows

---

**Epic Owner:** ML Engineering Team (Model Development)  
**Stakeholders:** Data Science Team, Operations Team, Infrastructure Team  
**Review Date:** End of Week 4 (Full model demonstration with production data)