# Epic-02A: Core Feature Engineering

**Epic ID:** Epic-02A  
**Epic Name:** Core Feature Engineering  
**Phase:** Phase 2A  
**Duration:** 2 weeks  
**Dependencies:** Epic-01 (Data Infrastructure Layer)  
**Priority:** Critical  

---

## 🎯 Epic Overview

### **Business Goal**
Establish a flexible, plugin-based feature engineering system that provides essential temporal and calendar features needed for machine learning models in the electric load forecasting system.

### **Technical Goal**
Implement the foundational feature engineering architecture with core transformation plugins that can generate the basic features required by LGBM and other models, enabling the start of model development in Epic-03.

### **Success Metrics**
- ✅ Plugin system extensible and documented
- ✅ 4+ core feature plugins implemented  
- ✅ Pipeline composer combines plugins
- ✅ Feature validation working
- ✅ LGBM can use generated features

---

## 🏗️ Technical Architecture

### **Core Components**

```
prevcarga/
├── features/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── plugin.py              # BaseFeaturePlugin abstract class
│   │   ├── composer.py            # FeaturePipelineComposer
│   │   └── validator.py           # FeatureValidator
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── temporal.py            # TemporalFeaturePlugin
│   │   ├── calendar.py            # CalendarFeaturePlugin
│   │   ├── lags.py                # LagsFeaturePlugin
│   │   └── cyclical.py            # CyclicalFeaturePlugin
│   ├── registry.py                # PluginRegistry
│   └── utils.py                   # Feature utilities
└── tests/
    └── features/
        ├── test_plugins.py
        ├── test_composer.py
        └── test_validator.py
```

### **Plugin Architecture Design**

```python
# Abstract base for all feature plugins
class BaseFeaturePlugin(ABC):
    """Base class for feature engineering plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name for registration."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version (semantic versioning)."""
        pass
    
    @abstractmethod
    def generate_features(
        self, 
        df: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate features from input dataframe."""
        pass
    
    @abstractmethod
    def get_feature_names(self, config: Dict[str, Any]) -> List[str]:
        """Get list of feature names that will be generated."""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate plugin configuration."""
        pass
```

---

## 📋 User Stories

### **User Story 1: Plugin Architecture Foundation**
**As a** data scientist  
**I want** a flexible plugin system for feature engineering  
**So that** I can easily add new feature transformations without modifying core code

#### **Acceptance Criteria:**
- [ ] `BaseFeaturePlugin` abstract class defines plugin interface
- [ ] `PluginRegistry` manages plugin registration and discovery
- [ ] Plugin versioning system supports semantic versioning
- [ ] Plugin configuration validation prevents runtime errors
- [ ] Documentation explains how to create new plugins
- [ ] Unit tests cover plugin registration and validation

#### **Technical Implementation:**
- **Base Plugin Interface:** Abstract class with required methods
- **Registry Pattern:** Central plugin management with discovery
- **Config Validation:** Pydantic schemas for plugin configurations
- **Error Handling:** Graceful failures with informative messages

#### **Definition of Done:**
- [ ] Plugin system accepts new plugins via simple inheritance
- [ ] Registry can list all available plugins with versions
- [ ] Invalid configurations raise clear validation errors
- [ ] Developer documentation includes plugin creation guide

---

### **User Story 2: Temporal Features Plugin**
**As a** ML engineer  
**I want** temporal features (hour, day, month, weekday)  
**So that** models can capture time-based patterns in electric load

#### **Acceptance Criteria:**
- [ ] Generates hour of day (0-23) and hour cyclical encoding
- [ ] Generates day of week (0-6) and weekday vs weekend flags
- [ ] Generates day of month (1-31) and month of year (1-12)
- [ ] Generates quarter and season indicators
- [ ] Supports timezone-aware datetime handling
- [ ] Handles edge cases (leap years, DST transitions)

#### **Technical Implementation:**
```python
class TemporalFeaturePlugin(BaseFeaturePlugin):
    """Generate temporal features from datetime index."""
    
    def generate_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        # Extract temporal components
        features = {}
        
        # Hour features
        features['hour'] = df.index.hour
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Day features
        features['dayofweek'] = df.index.dayofweek
        features['is_weekend'] = (features['dayofweek'] >= 5).astype(int)
        
        # Monthly and seasonal
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        return pd.DataFrame(features, index=df.index)
```

#### **Definition of Done:**
- [ ] All temporal features generate correctly for sample data
- [ ] Timezone handling works for Brazilian timezone (America/Sao_Paulo)
- [ ] Cyclical encoding prevents boundary discontinuities
- [ ] Edge cases (DST, leap year) handled without errors

---

### **User Story 3: Calendar and Holiday Features**
**As a** forecasting analyst  
**I want** holiday and special day indicators  
**So that** models can account for demand changes on non-working days

#### **Acceptance Criteria:**
- [ ] Integrates with Brazilian holiday calendar
- [ ] Generates binary flags for national holidays
- [ ] Supports custom holiday definitions via configuration
- [ ] Includes bridge day detection (holidays adjacent to weekends)
- [ ] Handles regional holidays (configurable by area)
- [ ] Generates holiday proximity features (days until/since holiday)

#### **Technical Implementation:**
```python
class CalendarFeaturePlugin(BaseFeaturePlugin):
    """Generate calendar and holiday features."""
    
    def __init__(self):
        self.brazil_holidays = holidays.Brazil()
    
    def generate_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        features = {}
        dates = df.index.date
        
        # Holiday indicators
        features['is_holiday'] = [d in self.brazil_holidays for d in dates]
        features['is_bridge_day'] = self._detect_bridge_days(dates)
        
        # Holiday proximity
        features['days_to_next_holiday'] = self._days_to_next_holiday(dates)
        features['days_since_last_holiday'] = self._days_since_last_holiday(dates)
        
        return pd.DataFrame(features, index=df.index)
```

#### **Definition of Done:**
- [ ] Brazilian national holidays correctly identified for 2023-2025
- [ ] Bridge day detection works for common patterns
- [ ] Holiday proximity features calculated accurately
- [ ] Configuration allows custom holiday additions

---

### **User Story 4: Lag Features Plugin**
**As a** time series modeler  
**I want** lag features from historical load data  
**So that** models can use autoregressive patterns for prediction

#### **Acceptance Criteria:**
- [ ] Generates configurable lag features (1h, 24h, 48h, 168h)
- [ ] Supports multiple lag intervals simultaneously
- [ ] Handles missing values in lag calculation appropriately
- [ ] Provides rolling statistics (mean, std) for lag windows
- [ ] Prevents data leakage in multi-horizon forecasting
- [ ] Optimizes memory usage for large datasets

#### **Technical Implementation:**
```python
class LagsFeaturePlugin(BaseFeaturePlugin):
    """Generate lag features from target variable."""
    
    def generate_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        target_col = config['target_column']
        lag_hours = config['lag_hours']  # [1, 24, 48, 168]
        
        features = {}
        
        for lag in lag_hours:
            # Simple lag
            features[f'lag_{lag}h'] = df[target_col].shift(lag)
            
            # Rolling statistics
            window_size = min(lag, 24)  # Max 24-hour window
            rolling = df[target_col].shift(lag).rolling(window_size)
            features[f'lag_{lag}h_mean'] = rolling.mean()
            features[f'lag_{lag}h_std'] = rolling.std()
        
        return pd.DataFrame(features, index=df.index)
```

#### **Definition of Done:**
- [ ] Lag features calculated correctly for all specified intervals
- [ ] Rolling statistics provide meaningful aggregations
- [ ] Memory usage remains acceptable for full dataset
- [ ] No data leakage in lag feature generation

---

### **User Story 5: Cyclical Encoding Plugin**
**As a** ML practitioner  
**I want** proper cyclical encoding for periodic features  
**So that** models understand the continuous nature of time periods

#### **Acceptance Criteria:**
- [ ] Implements sin/cos encoding for cyclical variables
- [ ] Handles multiple periods (daily, weekly, monthly, yearly)
- [ ] Configurable phase shifts for different regions
- [ ] Maintains continuity at period boundaries
- [ ] Supports custom period definitions
- [ ] Validates period specifications in configuration

#### **Technical Implementation:**
```python
class CyclicalFeaturePlugin(BaseFeaturePlugin):
    """Generate cyclical encodings for periodic features."""
    
    def generate_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        features = {}
        
        cyclical_configs = config['cyclical_features']
        
        for feature_config in cyclical_configs:
            col_name = feature_config['column']
            period = feature_config['period']
            
            if col_name in df.columns:
                values = df[col_name]
                
                # Sin/Cos encoding
                features[f'{col_name}_sin'] = np.sin(2 * np.pi * values / period)
                features[f'{col_name}_cos'] = np.cos(2 * np.pi * values / period)
        
        return pd.DataFrame(features, index=df.index)
```

#### **Definition of Done:**
- [ ] Sin/cos encoding produces continuous cyclical features
- [ ] Multiple periods handled simultaneously
- [ ] Boundary conditions (0/period) transition smoothly
- [ ] Configuration validation prevents invalid periods

---

### **User Story 6: Feature Pipeline Composer**
**As a** feature engineer  
**I want** to combine multiple feature plugins in a pipeline  
**So that** I can generate all required features in a coordinated manner

#### **Acceptance Criteria:**
- [ ] Orchestrates multiple plugins in specified order
- [ ] Handles plugin dependencies and execution order
- [ ] Provides feature name collision detection
- [ ] Supports conditional plugin execution
- [ ] Includes feature validation and quality checks
- [ ] Generates feature metadata and lineage tracking

#### **Technical Implementation:**
```python
class FeaturePipelineComposer:
    """Compose and execute feature engineering pipelines."""
    
    def __init__(self, plugin_registry: PluginRegistry):
        self.registry = plugin_registry
        self.plugins = []
        
    def add_plugin(self, plugin_name: str, config: Dict) -> 'FeaturePipelineComposer':
        """Add plugin to pipeline with configuration."""
        plugin = self.registry.get_plugin(plugin_name)
        plugin.validate_config(config)
        self.plugins.append((plugin, config))
        return self
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute pipeline and generate all features."""
        result = df.copy()
        
        for plugin, config in self.plugins:
            features = plugin.generate_features(result, config)
            result = pd.concat([result, features], axis=1)
            
        return result
```

#### **Definition of Done:**
- [ ] Pipeline executes plugins in correct dependency order
- [ ] Feature name conflicts detected and reported clearly
- [ ] Feature validation runs after each plugin execution
- [ ] Pipeline configuration is serializable and reproducible

---

## 🔧 Technical Requirements

### **Performance Requirements**
- Feature generation for 1 area (8760 hours): <30 seconds
- Memory usage: <2GB for full year of data
- Pipeline execution: Linear scaling with number of features

### **Data Requirements**
- Input: Cleaned dataframes from Epic-01 data infrastructure
- Timezone: All datetime handling in America/Sao_Paulo
- Frequency: Semi-hourly (30-min intervals)
- Missing values: Handled gracefully with appropriate fill strategies

### **Integration Requirements**
- Compatible with Epic-01 data loaders and validation
- Feature output format consumable by LGBM models (Epic-03)
- Configuration via YAML files with Pydantic validation
- Logging integration with structured JSON output

---

## 🧪 Testing Strategy

### **Unit Tests**
- [ ] Plugin interface compliance for all implementations
- [ ] Feature generation accuracy for known inputs
- [ ] Configuration validation edge cases
- [ ] Memory and performance benchmarks

### **Integration Tests**
- [ ] End-to-end pipeline with real data from Epic-01
- [ ] Feature compatibility with downstream models
- [ ] Error handling and recovery scenarios
- [ ] Configuration serialization/deserialization

### **Property-Based Tests**
- [ ] Cyclical encoding continuity properties
- [ ] Lag feature temporal consistency
- [ ] Holiday calendar completeness

---

## 📊 Quality Gates

### **Code Quality**
- [ ] 90%+ unit test coverage
- [ ] All linting rules pass (ruff, black)
- [ ] Type hints for all public interfaces
- [ ] Comprehensive docstrings with examples

### **Documentation**
- [ ] Plugin development guide
- [ ] Configuration reference documentation
- [ ] Feature catalog with descriptions
- [ ] Performance tuning recommendations

### **Integration Readiness**
- [ ] Epic-01 integration tests pass
- [ ] Sample features generated for all 26 time series
- [ ] LGBM model can consume generated features
- [ ] Feature pipeline runs in <2 minutes for test data

---

## 🚀 Delivery Plan

### **Week 1: Foundation (Days 1-5)**
- **Day 1-2:** Plugin architecture and registry implementation
- **Day 3:** Temporal features plugin
- **Day 4:** Calendar and holiday features plugin
- **Day 5:** Unit tests and documentation

### **Week 2: Advanced Features (Days 6-10)**
- **Day 6:** Lag features plugin implementation
- **Day 7:** Cyclical encoding plugin
- **Day 8:** Feature pipeline composer
- **Day 9:** Integration testing with Epic-01
- **Day 10:** Performance optimization and Epic-03 integration validation

---

## 🔗 Dependencies and Interfaces

### **Input Dependencies (Epic-01)**
- `DataCatalog`: For accessing validated time series data
- `S3ParquetLoader`: For loading historical data for lag features
- Validated dataframes with proper datetime indices

### **Output Interfaces (Epic-03)**
- Feature matrices in pandas DataFrame format
- Feature metadata including names, types, and generation configs
- Serializable pipeline configurations for model training

### **Configuration Interface**
```yaml
# features_config.yaml
feature_pipeline:
  plugins:
    - name: "temporal"
      config:
        include_cyclical: true
        timezone: "America/Sao_Paulo"
    
    - name: "calendar"
      config:
        country: "Brazil"
        include_bridge_days: true
        
    - name: "lags"
      config:
        target_column: "load"
        lag_hours: [1, 24, 48, 168]
        
    - name: "cyclical"
      config:
        cyclical_features:
          - column: "hour"
            period: 24
          - column: "dayofweek"
            period: 7
```

---

## 📈 Success Criteria

### **Functional Success Criteria**
- ✅ Plugin system extensible and documented
- ✅ 4+ core feature plugins implemented (temporal, calendar, lags, cyclical)
- ✅ Pipeline composer combines plugins correctly
- ✅ Feature validation working and catching errors
- ✅ LGBM can use generated features for training

### **Performance Success Criteria**
- ✅ Feature generation: <30 seconds per area per year
- ✅ Memory usage: <2GB for full dataset
- ✅ Pipeline execution scales linearly with features

### **Quality Success Criteria**
- ✅ 90%+ test coverage across all components
- ✅ All integration tests pass with Epic-01 and Epic-03
- ✅ Documentation complete and developer-friendly
- ✅ Configuration validation prevents runtime errors

---

## 🎯 Epic Completion Definition

Epic-02A is considered complete when:

1. **All User Stories Delivered:** 6 user stories implemented and accepted
2. **Integration Validated:** Successfully generates features for LGBM model training
3. **Performance Targets Met:** All performance requirements satisfied
4. **Quality Gates Passed:** Code quality, testing, and documentation standards met
5. **Epic-03 Unblocked:** LGBM model development can begin with generated features

**Handoff to Epic-03:** Feature pipeline generates required features for LGBM model training, with clear documentation and stable APIs for advanced feature integration in Epic-02B.

---

## 📋 Related Epics

- **Epic-01 (Data Infrastructure):** Provides validated data inputs
- **Epic-02B (Advanced Features):** Extends with LOESS, wavelets, and BLF strategy
- **Epic-03 (LGBM Model):** First consumer of generated features
- **Epic-04 (Hierarchical Models):** Will use core features for demand mean forecasting

---

**Epic Owner:** ML Engineering Team  
**Stakeholders:** Data Science Team, Model Development Team  
**Review Date:** End of Week 2 (Feature pipeline demonstration)