# PC-067-08A: Configuration Management System

**Ticket ID:** PC-067-08A  
**Epic:** Epic-08A - Core Workflows & Configuration  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Sprint:** Week 21 (Days 1-2)

---

## 📋 Description

Implement comprehensive configuration management system with Pydantic-based validation, YAML configuration files, environment-specific overrides (dev/staging/prod), and hot-reloading capabilities. Provides type-safe, validated configuration for all system components including data sources, models, combination strategies, reconciliation, evaluation, and execution parameters.

---

## 🎯 Acceptance Criteria

- [ ] YAML configuration files with clear schema structure
- [ ] Pydantic models for all configuration sections with validation
- [ ] Environment-specific overrides (dev/staging/prod)
- [ ] Configuration hot-reloading without system restart
- [ ] Default configuration templates for quick setup
- [ ] Clear validation error messages with actionable guidance
- [ ] Configuration schema documentation
- [ ] Unit tests with 85%+ coverage

---

## 📐 Technical Specifications

### Core Components

#### 1. Configuration Schema (Pydantic Models)

```python
from pydantic import BaseModel, Field, validator

class DataConfig(BaseModel):
    s3_bucket: str = Field(..., description="S3 bucket for data storage")
    parquet_path: str = Field(..., description="Path to Parquet files")
    catalog_path: str = Field(..., description="Data catalog location")
    cache_dir: Optional[str] = Field(None, description="Local cache directory")
    
    @validator('s3_bucket')
    def validate_bucket(cls, v):
        if not v.startswith('s3://'):
            raise ValueError('S3 bucket must start with s3://')
        return v

class ModelConfig(BaseModel):
    enabled: bool = Field(True, description="Enable this model")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_window_days: int = Field(730, description="Training window in days")

class ModelsConfig(BaseModel):
    lgbm: ModelConfig
    random_forest: ModelConfig
    regdin_svm: ModelConfig
    holt_winters: ModelConfig
    registry_path: str = Field(..., description="Model registry location")

class CombinationConfig(BaseModel):
    strategy: str = Field("weighted_average", description="Combination strategy")
    weights_optimization: bool = Field(True, description="Optimize weights")
    bias_correction: bool = Field(True, description="Apply bias correction")

class ReconciliationConfig(BaseModel):
    method: str = Field("mint", description="Reconciliation method")
    hierarchy_path: str = Field(..., description="Hierarchy definition YAML")
    loss_calculation: str = Field("difference", description="Loss calculation method")

class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(["mape", "mae", "rmse"], description="Metrics to calculate")
    confidence_level: float = Field(0.95, description="Confidence level")
    drift_detection: bool = Field(True, description="Enable drift detection")

class ExecutionConfig(BaseModel):
    max_concurrent_areas: int = Field(4, description="Max parallel areas")
    max_concurrent_models: int = Field(2, description="Max parallel models")
    timeout_minutes: int = Field(120, description="Workflow timeout")
    retry_attempts: int = Field(3, description="Retry failed tasks")

class LoggingConfig(BaseModel):
    level: str = Field("INFO", description="Log level")
    format: str = Field("json", description="Log format")
    output_path: Optional[str] = Field(None, description="Log file path")
    console: bool = Field(True, description="Log to console")

class SystemConfig(BaseModel):
    environment: str = Field("dev", description="Deployment environment")
    data: DataConfig
    models: ModelsConfig
    combination: CombinationConfig
    reconciliation: ReconciliationConfig
    evaluation: EvaluationConfig
    execution: ExecutionConfig
    logging: LoggingConfig
    
    class Config:
        extra = 'forbid'  # Prevent unknown fields
```

#### 2. ConfigManager

```python
class ConfigManager:
    def __init__(self, config_path: str, env: str = "dev")
    
    def _load_config(self) -> SystemConfig
    def _merge_configs(self, base: dict, override: dict) -> dict
    def reload(self) -> SystemConfig
    def validate_config(self, config_dict: dict) -> tuple[bool, Optional[str]]
    def get_template(self, env: str = "dev") -> dict
```

### YAML Configuration Structure

```yaml
# config.yaml (base)
environment: dev

data:
  s3_bucket: s3://prevcarga-dev
  parquet_path: data/raw
  catalog_path: data/catalog
  cache_dir: /tmp/prevcarga

models:
  lgbm:
    enabled: true
    hyperparameters:
      num_leaves: 31
      learning_rate: 0.05
  random_forest:
    enabled: true
    hyperparameters:
      n_estimators: 100
  regdin_svm:
    enabled: true
  holt_winters:
    enabled: true
  registry_path: s3://prevcarga-dev/models

combination:
  strategy: weighted_average
  weights_optimization: true
  bias_correction: true

reconciliation:
  method: mint
  hierarchy_path: config/hierarchy.yaml
  loss_calculation: difference

evaluation:
  metrics:
    - mape
    - mae
    - rmse
  confidence_level: 0.95
  drift_detection: true

execution:
  max_concurrent_areas: 4
  max_concurrent_models: 2
  timeout_minutes: 120
  retry_attempts: 3

logging:
  level: INFO
  format: json
  console: true
```

```yaml
# config.prod.yaml (production overrides)
environment: prod

data:
  s3_bucket: s3://prevcarga-prod
  cache_dir: /var/cache/prevcarga

models:
  registry_path: s3://prevcarga-prod/models

execution:
  max_concurrent_areas: 8
  max_concurrent_models: 4
  timeout_minutes: 240

logging:
  level: WARNING
  output_path: /var/log/prevcarga/app.log
```

---

## 🔧 Implementation Tasks

### Task 1: Pydantic Models (4 hours)
- [ ] Implement all configuration Pydantic models
- [ ] Add field validators for complex validation
- [ ] Add field descriptions and defaults
- [ ] Implement `SystemConfig` with nested models
- [ ] Add `extra='forbid'` to prevent unknown fields

### Task 2: YAML Loading and Parsing (3 hours)
- [ ] Implement YAML file loading
- [ ] Add environment variable substitution
- [ ] Handle missing files gracefully
- [ ] Add error handling for malformed YAML

### Task 3: Configuration Merging (3 hours)
- [ ] Implement deep merge logic for overrides
- [ ] Handle nested dictionary merging
- [ ] Support list merging strategies
- [ ] Test merge precedence (base < env override)

### Task 4: ConfigManager Implementation (4 hours)
- [ ] Implement `ConfigManager` class
- [ ] Add configuration loading pipeline
- [ ] Implement validation error formatting
- [ ] Add configuration caching

### Task 5: Hot-Reload Mechanism (3 hours)
- [ ] Implement file modification detection
- [ ] Add `reload()` method
- [ ] Handle reload errors gracefully
- [ ] Add reload event notifications

### Task 6: Template Generation (2 hours)
- [ ] Implement `get_template()` for dev environment
- [ ] Add production template
- [ ] Add staging template
- [ ] Include sensible defaults

### Task 7: Error Handling and Validation (3 hours)
- [ ] Improve validation error messages
- [ ] Add field-level error context
- [ ] Create custom exception classes
- [ ] Add validation helper methods

### Task 8: Testing and Documentation (4 hours)
- [ ] Unit tests for each config section
- [ ] Test environment override precedence
- [ ] Test hot-reload functionality
- [ ] Test validation error messages
- [ ] Create configuration guide documentation

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_data_config_validation()
def test_data_config_s3_bucket_validation()
def test_models_config_all_models()
def test_system_config_full_validation()
def test_config_manager_load_base()
def test_config_manager_load_with_override()
def test_config_merge_nested_dicts()
def test_config_merge_override_precedence()
def test_hot_reload_on_file_change()
def test_hot_reload_no_change()
def test_validation_error_messages()
def test_template_generation_dev()
def test_template_generation_prod()
def test_unknown_field_rejection()
```

### Integration Tests
```python
def test_load_full_config_pipeline()
def test_multi_environment_configs()
def test_config_hot_reload_workflow()
def test_invalid_config_handling()
```

### Edge Case Tests
```python
def test_missing_config_file()
def test_malformed_yaml()
def test_missing_required_fields()
def test_invalid_field_types()
def test_invalid_field_values()
```

---

## 📊 Success Metrics

### Performance Targets
- Configuration loading: **<1 second**
- Configuration validation: **<200 ms**
- Hot-reload detection: **<500 ms**

### Quality Targets
- Unit test coverage: **≥85%**
- Validation error clarity: **100%** actionable messages
- Schema completeness: **100%** system parameters covered

---

## 📚 Dependencies

### Required Packages
```python
pydantic >= 2.0.0           # Configuration validation
pyyaml >= 6.0               # YAML parsing
watchdog >= 3.0.0           # File change detection (optional)
```

### Code Dependencies
- None (foundational component)

---

## 🔗 Related Tickets
- **Blocks:** PC-068-08A (Training Workflow)
- **Blocks:** PC-069-08A (Prediction Workflow)
- **Blocks:** PC-070-08A (Backtesting Workflow)

---

## 📖 Documentation Requirements

- [ ] Configuration schema reference with all fields
- [ ] Environment override examples
- [ ] Validation error troubleshooting guide
- [ ] Template usage instructions
- [ ] Hot-reload setup guide
- [ ] Best practices for production configuration

---

## 💡 Implementation Notes

### Pydantic Validation Best Practices
```python
# Use Field() for documentation
field: str = Field(..., description="Clear description")

# Add validators for complex logic
@validator('field_name')
def validate_field(cls, v):
    if not valid_condition(v):
        raise ValueError('Specific error message')
    return v

# Use root_validator for cross-field validation
@root_validator
def validate_combination(cls, values):
    if values.get('a') and not values.get('b'):
        raise ValueError('Field b required when a is set')
    return values
```

### Environment Variable Substitution
```yaml
data:
  s3_bucket: ${S3_BUCKET:s3://prevcarga-dev}  # Default if not set
  
# In code:
import os
config_str = config_str.replace('${S3_BUCKET}', os.getenv('S3_BUCKET', 's3://prevcarga-dev'))
```

### Hot-Reload Implementation
```python
import time
from pathlib import Path

class ConfigManager:
    def reload(self) -> SystemConfig:
        current_mtime = self.config_path.stat().st_mtime
        if current_mtime > self._config_mtime:
            self.config = self._load_config()
            self._config_mtime = current_mtime
            logger.info("Configuration reloaded")
        return self.config
```

### Clear Validation Errors
```python
try:
    config = SystemConfig(**config_dict)
except ValidationError as e:
    # Format errors clearly
    for error in e.errors():
        field = '.'.join(str(loc) for loc in error['loc'])
        message = error['msg']
        print(f"Configuration error in '{field}': {message}")
```

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥85% coverage
- [ ] All Pydantic models implemented with validation
- [ ] YAML loading with environment overrides working
- [ ] Configuration hot-reloading functional
- [ ] Template generation for all environments
- [ ] Clear validation error messages
- [ ] Code reviewed and approved
- [ ] Documentation complete with examples
- [ ] Ready for PC-068-08A (Training Workflow)
