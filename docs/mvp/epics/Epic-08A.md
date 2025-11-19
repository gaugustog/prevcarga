# Epic-08A: Core Workflows & Configuration

**Epic ID:** Epic-08A  
**Epic Name:** Core Workflows & Configuration  
**Phase:** 8A  
**Duration:** 2 weeks (Weeks 21-22)  
**Dependencies:** Epic-07B (Monitoring & Reporting)  
**Priority:** High  

---

## 🎯 Epic Overview

Implement core orchestration workflows and comprehensive configuration management for the unified electric load forecasting system. This epic establishes the foundational workflow infrastructure for training, prediction, and backtesting operations that coordinate across 26 time series, 5 models, combination strategies, and hierarchical reconciliation.

### Business Value
- **Systematic Orchestration:** Coordinated execution of complex multi-step workflows
- **Configuration Flexibility:** Environment-specific deployments with validation
- **Operational Efficiency:** Automated workflows reduce manual intervention
- **Quality Assurance:** Backtesting framework validates model performance
- **Production Readiness:** Workflow infrastructure for operational deployment

---

## 📋 User Stories

### **User Story 1: Configuration Management System**
**As a** system administrator  
**I want** a comprehensive configuration management system  
**So that** I can deploy the system across different environments with proper validation

**Acceptance Criteria:**
- [ ] YAML configuration files with Pydantic validation
- [ ] Environment-specific overrides (dev/staging/prod)
- [ ] Configuration schema validation with clear error messages
- [ ] Support for nested configurations (models, data sources, metrics)
- [ ] Configuration hot-reloading without system restart
- [ ] Default configuration templates for quick setup

**Technical Requirements:**
- Pydantic models for all configuration sections
- YAML parsing with environment variable substitution
- Schema validation with detailed error reporting
- Configuration inheritance (base + environment overrides)
- Hot-reload detection via file watchers
- Template generation for common deployment scenarios

**Definition of Done:**
- [ ] Configuration system handles all system parameters
- [ ] Validation provides clear, actionable error messages
- [ ] Environment-specific deployments work seamlessly
- [ ] Configuration hot-reloading tested and functional
- [ ] Unit tests cover all validation scenarios
- [ ] Documentation with configuration examples

---

### **User Story 2: Training Workflow Orchestrator**
**As a** ML engineer  
**I want** an orchestrated training workflow  
**So that** I can train all models systematically with proper dependency management

**Acceptance Criteria:**
- [ ] Train all 5 base models with proper sequencing
- [ ] Handle model dependencies (hierarchical models after end-to-end)
- [ ] Parallel training across areas when possible
- [ ] Model versioning and artifact management
- [ ] Training progress tracking with detailed logging
- [ ] Failure recovery and partial re-training capabilities

**Technical Requirements:**
- Workflow DAG for model dependencies
- Area-level parallelization for independent models
- Model registry integration for versioning
- Checkpoint mechanism for failure recovery
- Progress tracking with percentage completion
- Training metrics collection and logging

**Definition of Done:**
- [ ] Training workflow completes successfully for all 26 series
- [ ] Model dependencies correctly enforced
- [ ] Trained models versioned and registered
- [ ] Training progress visible through logs
- [ ] Failure recovery allows resuming from checkpoint
- [ ] Performance benchmark: <4 hours for full training

---

### **User Story 3: Prediction Workflow (Batch & Intraday)**
**As a** system operator  
**I want** flexible prediction workflows  
**So that** I can generate forecasts for both scheduled batch runs and real-time intraday updates

**Acceptance Criteria:**
- [ ] Batch prediction workflow for all horizons (D+0 to D+8)
- [ ] Intraday prediction workflow with BLF strategy
- [ ] Model combination and hierarchical reconciliation integration
- [ ] Prediction caching and incremental updates
- [ ] Output formatting for downstream systems
- [ ] Real-time monitoring and alerting

**Technical Requirements:**
- Batch workflow: load models → generate features → predict → combine → reconcile
- Intraday workflow: update features → LGBM BLF → partial reconciliation
- Integration with Epic-05A/05B combiners and Epic-06A/06B reconcilers
- Prediction result caching (Redis/in-memory)
- Standard output format (JSON/Parquet)
- Monitoring hooks for prediction latency and failures

**Definition of Done:**
- [ ] Batch predictions complete within 15 minutes for all series
- [ ] Intraday predictions complete within 5 minutes
- [ ] Prediction outputs compatible with downstream systems
- [ ] Monitoring alerts on prediction failures or delays
- [ ] Caching reduces redundant computation by >30%
- [ ] Integration tests with Epic-05/06 components successful

---

### **User Story 4: Backtesting Workflow with Retraining Evaluation**
**As a** data scientist  
**I want** a comprehensive backtesting framework  
**So that** I can evaluate model performance and optimize retraining intervals

**Acceptance Criteria:**
- [ ] Historical backtesting with walk-forward validation
- [ ] Retraining interval optimization (weekly, bi-weekly, monthly)
- [ ] Model performance tracking over time
- [ ] Drift detection integration
- [ ] Statistical significance testing for improvements
- [ ] Automated report generation with recommendations

**Technical Requirements:**
- Walk-forward validation with expanding/sliding window
- Multiple retraining intervals: 7, 14, 30 days
- Integration with Epic-07A metrics and Epic-07B drift detection
- Statistical comparison (paired t-tests) between intervals
- Performance degradation tracking
- Report generation with optimal interval recommendations

**Definition of Done:**
- [ ] Backtesting framework validates 1+ years of historical data
- [ ] Retraining interval optimization provides clear recommendations
- [ ] Statistical tests confirm significance of improvements
- [ ] Drift detection accuracy >90% in backtest scenarios
- [ ] Automated reports enable data-driven decisions
- [ ] Performance benchmark: 1-year backtest in <8 hours

---

## 🏗️ Technical Architecture

### Core Components

```python
# Configuration Management
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import yaml
from pathlib import Path

class DataConfig(BaseModel):
    """Data layer configuration."""
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
    """Individual model configuration."""
    enabled: bool = Field(True, description="Enable this model")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_window_days: int = Field(730, description="Training window in days")
    
class ModelsConfig(BaseModel):
    """All models configuration."""
    lgbm: ModelConfig
    random_forest: ModelConfig
    regdin_svm: ModelConfig
    holt_winters: ModelConfig
    registry_path: str = Field(..., description="Model registry location")

class CombinationConfig(BaseModel):
    """Model combination configuration."""
    strategy: str = Field("weighted_average", description="Combination strategy")
    weights_optimization: bool = Field(True, description="Optimize weights")
    bias_correction: bool = Field(True, description="Apply bias correction")

class ReconciliationConfig(BaseModel):
    """Hierarchical reconciliation configuration."""
    method: str = Field("mint", description="Reconciliation method")
    hierarchy_path: str = Field(..., description="Hierarchy definition YAML")
    loss_calculation: str = Field("difference", description="Loss calculation method")

class EvaluationConfig(BaseModel):
    """Evaluation and metrics configuration."""
    metrics: List[str] = Field(["mape", "mae", "rmse"], description="Metrics to calculate")
    confidence_level: float = Field(0.95, description="Confidence level")
    drift_detection: bool = Field(True, description="Enable drift detection")

class ExecutionConfig(BaseModel):
    """Execution and parallelization configuration."""
    max_concurrent_areas: int = Field(4, description="Max parallel areas")
    max_concurrent_models: int = Field(2, description="Max parallel models")
    timeout_minutes: int = Field(120, description="Workflow timeout")
    retry_attempts: int = Field(3, description="Retry failed tasks")

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field("INFO", description="Log level")
    format: str = Field("json", description="Log format")
    output_path: Optional[str] = Field(None, description="Log file path")
    console: bool = Field(True, description="Log to console")

class SystemConfig(BaseModel):
    """Complete system configuration."""
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

class ConfigManager:
    """Configuration management with validation and hot-reloading."""
    
    def __init__(self, config_path: str, env: str = "dev"):
        self.config_path = Path(config_path)
        self.env = env
        self.config = self._load_config()
        self._config_mtime = self.config_path.stat().st_mtime
    
    def _load_config(self) -> SystemConfig:
        """
        Load and validate configuration.
        
        Loads base config and applies environment-specific overrides.
        """
        # Load base configuration
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Apply environment-specific overrides
        env_path = self.config_path.parent / f"config.{self.env}.yaml"
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_overrides = yaml.safe_load(f)
            config_dict = self._merge_configs(config_dict, env_overrides)
        
        # Validate with Pydantic
        try:
            config = SystemConfig(**config_dict)
            return config
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    def _merge_configs(self, base: dict, override: dict) -> dict:
        """Deep merge configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def reload(self) -> SystemConfig:
        """Hot-reload configuration if file changed."""
        current_mtime = self.config_path.stat().st_mtime
        if current_mtime > self._config_mtime:
            self.config = self._load_config()
            self._config_mtime = current_mtime
        return self.config
    
    def validate_config(self, config_dict: dict) -> tuple[bool, Optional[str]]:
        """
        Validate configuration dictionary.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            SystemConfig(**config_dict)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_template(self, env: str = "dev") -> dict:
        """Generate configuration template for environment."""
        if env == "dev":
            return {
                'environment': 'dev',
                'data': {
                    's3_bucket': 's3://prevcarga-dev',
                    'parquet_path': 'data/raw',
                    'catalog_path': 'data/catalog',
                    'cache_dir': '/tmp/prevcarga'
                },
                'models': {
                    'lgbm': {'enabled': True, 'hyperparameters': {}},
                    'random_forest': {'enabled': True, 'hyperparameters': {}},
                    'regdin_svm': {'enabled': True, 'hyperparameters': {}},
                    'holt_winters': {'enabled': True, 'hyperparameters': {}},
                    'registry_path': 's3://prevcarga-dev/models'
                },
                'execution': {
                    'max_concurrent_areas': 2,
                    'max_concurrent_models': 1
                }
            }
        # Production template would have different defaults
        return {}

class ConfigurationError(Exception):
    """Configuration validation error."""
    pass


# Training Workflow Orchestrator
from enum import Enum
from datetime import datetime, timedelta
import asyncio

class ModelType(Enum):
    """Model types."""
    LGBM = "lgbm"
    RANDOM_FOREST = "random_forest"
    REGDIN_SVM = "regdin_svm"
    HOLT_WINTERS = "holt_winters"

class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingTask:
    """Training task specification."""
    model_type: ModelType
    area: str
    start_date: datetime
    end_date: datetime
    dependencies: List[str] = None  # Task IDs this depends on
    task_id: Optional[str] = None

@dataclass
class TrainingResult:
    """Training execution result."""
    task_id: str
    model_type: ModelType
    area: str
    status: WorkflowStatus
    model_version: Optional[str]
    metrics: Dict[str, float]
    duration_seconds: float
    error: Optional[str]

class TrainingWorkflow:
    """Orchestrator for model training workflows."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model_registry = ModelRegistry(config.models.registry_path)
        self.data_loader = DataLoader(config.data)
        self.checkpoint_manager = CheckpointManager()
    
    async def execute_training(
        self,
        areas: List[str],
        models: List[ModelType],
        start_date: datetime,
        end_date: datetime,
        resume_from_checkpoint: bool = False
    ) -> Dict[str, TrainingResult]:
        """
        Execute training workflow for specified areas and models.
        
        Args:
            areas: List of area identifiers to train
            models: List of model types to train
            start_date: Training period start
            end_date: Training period end
            resume_from_checkpoint: Resume from last checkpoint if available
            
        Returns:
            Dictionary mapping task_id to TrainingResult
        """
        # 1. Create training tasks with dependencies
        tasks = self._create_training_tasks(areas, models, start_date, end_date)
        
        # 2. Check for checkpoint and filter completed tasks
        if resume_from_checkpoint:
            completed_tasks = self.checkpoint_manager.get_completed_tasks()
            tasks = [t for t in tasks if t.task_id not in completed_tasks]
        
        # 3. Execute tasks respecting dependencies
        results = await self._execute_tasks_with_dependencies(tasks)
        
        # 4. Generate training summary
        summary = self._generate_training_summary(results)
        
        return results
    
    def _create_training_tasks(
        self,
        areas: List[str],
        models: List[ModelType],
        start_date: datetime,
        end_date: datetime
    ) -> List[TrainingTask]:
        """Create training tasks with proper dependency management."""
        tasks = []
        
        # End-to-end models (LGBM, RF) - no dependencies
        for area in areas:
            for model in [ModelType.LGBM, ModelType.RANDOM_FOREST]:
                if model in models and self._is_model_enabled(model):
                    task = TrainingTask(
                        model_type=model,
                        area=area,
                        start_date=start_date,
                        end_date=end_date,
                        dependencies=[],
                        task_id=f"{model.value}_{area}"
                    )
                    tasks.append(task)
        
        # Hierarchical models - depend on end-to-end models for same area
        for area in areas:
            e2e_task_ids = [f"{ModelType.LGBM.value}_{area}", 
                           f"{ModelType.RANDOM_FOREST.value}_{area}"]
            
            for model in [ModelType.REGDIN_SVM, ModelType.HOLT_WINTERS]:
                if model in models and self._is_model_enabled(model):
                    task = TrainingTask(
                        model_type=model,
                        area=area,
                        start_date=start_date,
                        end_date=end_date,
                        dependencies=e2e_task_ids,
                        task_id=f"{model.value}_{area}"
                    )
                    tasks.append(task)
        
        return tasks
    
    async def _execute_tasks_with_dependencies(
        self,
        tasks: List[TrainingTask]
    ) -> Dict[str, TrainingResult]:
        """Execute tasks respecting dependencies with parallelization."""
        results = {}
        pending_tasks = {t.task_id: t for t in tasks}
        completed_tasks = set()
        
        while pending_tasks:
            # Find tasks ready to execute (dependencies satisfied)
            ready_tasks = [
                task for task in pending_tasks.values()
                if all(dep in completed_tasks for dep in (task.dependencies or []))
            ]
            
            if not ready_tasks:
                raise ValueError("Circular dependency detected or no tasks ready")
            
            # Execute ready tasks in parallel (respecting concurrency limits)
            batch_results = await self._execute_task_batch(ready_tasks)
            
            # Update results and completed tasks
            for task_id, result in batch_results.items():
                results[task_id] = result
                if result.status == WorkflowStatus.COMPLETED:
                    completed_tasks.add(task_id)
                    self.checkpoint_manager.mark_completed(task_id, result)
                del pending_tasks[task_id]
        
        return results
    
    async def _execute_task_batch(
        self,
        tasks: List[TrainingTask]
    ) -> Dict[str, TrainingResult]:
        """Execute batch of tasks in parallel."""
        max_concurrent = self.config.execution.max_concurrent_areas
        
        # Group by area for better resource utilization
        area_tasks = {}
        for task in tasks:
            if task.area not in area_tasks:
                area_tasks[task.area] = []
            area_tasks[task.area].append(task)
        
        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_limit(task):
            async with semaphore:
                return await self._train_model_for_area(task)
        
        # Flatten tasks and execute
        all_tasks = [t for tasks_list in area_tasks.values() for t in tasks_list]
        task_results = await asyncio.gather(
            *[execute_with_limit(task) for task in all_tasks],
            return_exceptions=True
        )
        
        # Map results to task IDs
        results = {}
        for task, result in zip(all_tasks, task_results):
            if isinstance(result, Exception):
                results[task.task_id] = TrainingResult(
                    task_id=task.task_id,
                    model_type=task.model_type,
                    area=task.area,
                    status=WorkflowStatus.FAILED,
                    model_version=None,
                    metrics={},
                    duration_seconds=0,
                    error=str(result)
                )
            else:
                results[task.task_id] = result
        
        return results
    
    async def _train_model_for_area(self, task: TrainingTask) -> TrainingResult:
        """
        Train individual model for specific area.
        
        Returns:
            TrainingResult with model version and metrics
        """
        start_time = datetime.now()
        
        try:
            # 1. Load training data
            training_data = await self.data_loader.load_training_data(
                area=task.area,
                start_date=task.start_date,
                end_date=task.end_date
            )
            
            # 2. Get model trainer
            trainer = ModelTrainerFactory.create(
                task.model_type,
                self.config.models
            )
            
            # 3. Train model
            trained_model = await trainer.train(training_data)
            
            # 4. Calculate training metrics
            metrics = trainer.evaluate(trained_model, training_data)
            
            # 5. Version and register model
            model_version = self.model_registry.register_model(
                model=trained_model,
                model_type=task.model_type.value,
                area=task.area,
                training_period=(task.start_date, task.end_date),
                metrics=metrics
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TrainingResult(
                task_id=task.task_id,
                model_type=task.model_type,
                area=task.area,
                status=WorkflowStatus.COMPLETED,
                model_version=model_version,
                metrics=metrics,
                duration_seconds=duration,
                error=None
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TrainingResult(
                task_id=task.task_id,
                model_type=task.model_type,
                area=task.area,
                status=WorkflowStatus.FAILED,
                model_version=None,
                metrics={},
                duration_seconds=duration,
                error=str(e)
            )
    
    def _is_model_enabled(self, model_type: ModelType) -> bool:
        """Check if model is enabled in configuration."""
        model_config = getattr(self.config.models, model_type.value)
        return model_config.enabled
    
    def _generate_training_summary(
        self,
        results: Dict[str, TrainingResult]
    ) -> Dict[str, Any]:
        """Generate training execution summary."""
        total_tasks = len(results)
        completed = sum(1 for r in results.values() if r.status == WorkflowStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.status == WorkflowStatus.FAILED)
        total_duration = sum(r.duration_seconds for r in results.values())
        
        return {
            'total_tasks': total_tasks,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_tasks if total_tasks > 0 else 0,
            'total_duration_seconds': total_duration,
            'average_task_duration': total_duration / total_tasks if total_tasks > 0 else 0
        }


# Prediction Workflow
@dataclass
class PredictionTask:
    """Prediction task specification."""
    prediction_date: datetime
    horizons: List[int]  # D+0 to D+8
    areas: List[str]
    task_id: Optional[str] = None

@dataclass
class PredictionResult:
    """Prediction execution result."""
    task_id: str
    prediction_date: datetime
    predictions: Dict[str, np.ndarray]  # area -> predictions
    combined_predictions: Dict[str, np.ndarray]
    reconciled_predictions: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    duration_seconds: float
    status: WorkflowStatus
    error: Optional[str]

class PredictionWorkflow:
    """Orchestrator for batch and intraday prediction workflows."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model_registry = ModelRegistry(config.models.registry_path)
        self.feature_generator = FeatureGenerator(config.data)
        self.combiner = CombinerFactory.create(config.combination)
        self.reconciler = ReconcilerFactory.create(config.reconciliation)
        self.cache = PredictionCache()
    
    async def execute_batch_prediction(
        self,
        prediction_date: datetime,
        horizons: List[int] = list(range(9)),  # D+0 to D+8
        areas: Optional[List[str]] = None
    ) -> PredictionResult:
        """
        Execute batch prediction workflow.
        
        Generates forecasts for all specified horizons, applies combination
        and hierarchical reconciliation.
        
        Args:
            prediction_date: Date for which to generate predictions
            horizons: Forecast horizons (default D+0 to D+8)
            areas: Areas to predict (default all)
            
        Returns:
            PredictionResult with forecasts at all stages
        """
        start_time = datetime.now()
        task_id = f"batch_{prediction_date.strftime('%Y%m%d')}"
        
        try:
            # 1. Load latest trained models
            models = await self._load_models(areas)
            
            # 2. Generate features for prediction date
            features = await self.feature_generator.generate_features(
                prediction_date=prediction_date,
                horizons=horizons,
                areas=areas
            )
            
            # 3. Execute base model predictions in parallel
            base_predictions = await self._execute_base_predictions(
                models=models,
                features=features,
                prediction_date=prediction_date,
                horizons=horizons
            )
            
            # 4. Apply model combination strategies
            combined_predictions = self.combiner.combine(base_predictions)
            
            # 5. Perform hierarchical reconciliation
            reconciled_predictions = self.reconciler.reconcile(combined_predictions)
            
            # 6. Cache results
            self.cache.store(
                key=task_id,
                predictions=reconciled_predictions,
                ttl_seconds=3600
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return PredictionResult(
                task_id=task_id,
                prediction_date=prediction_date,
                predictions=base_predictions,
                combined_predictions=combined_predictions,
                reconciled_predictions=reconciled_predictions,
                metadata={
                    'models_used': list(models.keys()),
                    'combination_strategy': self.config.combination.strategy,
                    'reconciliation_method': self.config.reconciliation.method
                },
                duration_seconds=duration,
                status=WorkflowStatus.COMPLETED,
                error=None
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return PredictionResult(
                task_id=task_id,
                prediction_date=prediction_date,
                predictions={},
                combined_predictions={},
                reconciled_predictions={},
                metadata={},
                duration_seconds=duration,
                status=WorkflowStatus.FAILED,
                error=str(e)
            )
    
    async def execute_intraday_prediction(
        self,
        current_datetime: datetime,
        updated_data: 'IntradayData'
    ) -> PredictionResult:
        """
        Execute intraday prediction workflow with BLF strategy.
        
        Updates D+0 forecast using LGBM BLF based on latest observations.
        
        Args:
            current_datetime: Current timestamp
            updated_data: Latest intraday observations
            
        Returns:
            PredictionResult with updated D+0 forecast
        """
        start_time = datetime.now()
        task_id = f"intraday_{current_datetime.strftime('%Y%m%d_%H%M')}"
        
        try:
            # 1. Update features with latest intraday data
            features = await self.feature_generator.update_intraday_features(
                current_datetime=current_datetime,
                updated_data=updated_data
            )
            
            # 2. Execute LGBM BLF prediction (D+0 only)
            lgbm_model = await self.model_registry.load_latest_model(
                model_type='lgbm',
                area=updated_data.area
            )
            
            d0_prediction = lgbm_model.predict_intraday(
                features=features,
                current_datetime=current_datetime
            )
            
            # 3. Retrieve cached D+1 to D+8 forecasts
            cached_predictions = self.cache.retrieve_future_horizons(
                prediction_date=current_datetime.date(),
                area=updated_data.area,
                horizons=list(range(1, 9))
            )
            
            # 4. Combine D+0 with cached forecasts
            full_predictions = {
                updated_data.area: np.concatenate([
                    d0_prediction,
                    cached_predictions
                ])
            }
            
            # 5. Partial reconciliation if needed
            if self.config.reconciliation.intraday_reconcile:
                reconciled_predictions = self.reconciler.reconcile_partial(
                    full_predictions,
                    hierarchy_level=updated_data.area
                )
            else:
                reconciled_predictions = full_predictions
            
            # 6. Update cache
            self.cache.update_intraday(
                area=updated_data.area,
                datetime=current_datetime,
                prediction=d0_prediction
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return PredictionResult(
                task_id=task_id,
                prediction_date=current_datetime,
                predictions={'lgbm_blf': d0_prediction},
                combined_predictions=full_predictions,
                reconciled_predictions=reconciled_predictions,
                metadata={
                    'intraday_update': True,
                    'updated_area': updated_data.area,
                    'observation_time': current_datetime.isoformat()
                },
                duration_seconds=duration,
                status=WorkflowStatus.COMPLETED,
                error=None
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return PredictionResult(
                task_id=task_id,
                prediction_date=current_datetime,
                predictions={},
                combined_predictions={},
                reconciled_predictions={},
                metadata={'intraday_update': True},
                duration_seconds=duration,
                status=WorkflowStatus.FAILED,
                error=str(e)
            )
    
    async def _load_models(self, areas: Optional[List[str]]) -> Dict[str, Any]:
        """Load latest models for specified areas."""
        models = {}
        
        enabled_models = [
            model_type for model_type in ModelType
            if self._is_model_enabled(model_type)
        ]
        
        for model_type in enabled_models:
            for area in areas:
                model_key = f"{model_type.value}_{area}"
                model = await self.model_registry.load_latest_model(
                    model_type=model_type.value,
                    area=area
                )
                models[model_key] = model
        
        return models
    
    async def _execute_base_predictions(
        self,
        models: Dict[str, Any],
        features: Dict[str, np.ndarray],
        prediction_date: datetime,
        horizons: List[int]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute predictions for all base models in parallel."""
        # Implementation similar to training task batch execution
        pass
    
    def _is_model_enabled(self, model_type: ModelType) -> bool:
        """Check if model is enabled in configuration."""
        model_config = getattr(self.config.models, model_type.value)
        return model_config.enabled


# Backtesting Workflow
@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: datetime
    end_date: datetime
    retraining_intervals: List[int]  # days
    validation_strategy: str  # 'expanding' or 'sliding'
    initial_training_days: int = 730
    
@dataclass
class BacktestResult:
    """Backtesting execution result."""
    config: BacktestConfig
    interval_results: Dict[int, Dict[str, Any]]  # interval -> metrics
    optimal_interval: int
    recommendations: str
    statistical_tests: Dict[str, Any]
    duration_seconds: float

class BacktestingWorkflow:
    """Comprehensive backtesting framework with retraining evaluation."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.training_workflow = TrainingWorkflow(config)
        self.prediction_workflow = PredictionWorkflow(config)
        self.evaluator = MetricsCalculator(config.evaluation)
        self.drift_detector = DriftDetector(config.evaluation)
    
    async def execute_backtest(
        self,
        backtest_config: BacktestConfig,
        areas: Optional[List[str]] = None
    ) -> BacktestResult:
        """
        Execute comprehensive backtest with retraining interval optimization.
        
        Args:
            backtest_config: Backtesting configuration
            areas: Areas to backtest (default all)
            
        Returns:
            BacktestResult with optimal interval and recommendations
        """
        start_time = datetime.now()
        
        # Results for each retraining interval
        interval_results = {}
        
        for interval_days in backtest_config.retraining_intervals:
            # Execute backtest for this interval
            interval_result = await self._backtest_with_interval(
                start_date=backtest_config.start_date,
                end_date=backtest_config.end_date,
                retraining_interval_days=interval_days,
                validation_strategy=backtest_config.validation_strategy,
                initial_training_days=backtest_config.initial_training_days,
                areas=areas
            )
            
            interval_results[interval_days] = interval_result
        
        # Compare intervals and determine optimal
        optimal_interval = self._determine_optimal_interval(interval_results)
        
        # Statistical comparison
        statistical_tests = self._perform_statistical_comparison(interval_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            interval_results,
            optimal_interval,
            statistical_tests
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return BacktestResult(
            config=backtest_config,
            interval_results=interval_results,
            optimal_interval=optimal_interval,
            recommendations=recommendations,
            statistical_tests=statistical_tests,
            duration_seconds=duration
        )
    
    async def _backtest_with_interval(
        self,
        start_date: datetime,
        end_date: datetime,
        retraining_interval_days: int,
        validation_strategy: str,
        initial_training_days: int,
        areas: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Execute backtest with specific retraining interval."""
        
        # Define backtesting periods
        periods = self._define_backtest_periods(
            start_date=start_date,
            end_date=end_date,
            retraining_interval_days=retraining_interval_days,
            initial_training_days=initial_training_days,
            strategy=validation_strategy
        )
        
        all_predictions = []
        all_actuals = []
        drift_events = []
        
        for period in periods:
            # Train models
            training_result = await self.training_workflow.execute_training(
                areas=areas or self._get_all_areas(),
                models=[ModelType.LGBM, ModelType.RANDOM_FOREST],
                start_date=period['train_start'],
                end_date=period['train_end']
            )
            
            # Generate predictions for test period
            prediction_result = await self.prediction_workflow.execute_batch_prediction(
                prediction_date=period['test_start'],
                horizons=list(range(9))
            )
            
            # Load actuals
            actuals = await self._load_actuals(
                start_date=period['test_start'],
                end_date=period['test_end'],
                areas=areas
            )
            
            all_predictions.extend(prediction_result.reconciled_predictions)
            all_actuals.extend(actuals)
            
            # Check for drift
            metrics = self.evaluator.calculate_metrics(
                predictions=prediction_result.reconciled_predictions,
                actuals=actuals
            )
            
            drift_result = self.drift_detector.detect_drift(
                current_metric=metrics['mape'],
                method='ensemble'
            )
            
            if drift_result.drift_detected:
                drift_events.append({
                    'period': period,
                    'drift_type': drift_result.drift_type,
                    'degradation_pct': drift_result.degradation_pct
                })
        
        # Calculate overall metrics
        overall_metrics = self.evaluator.calculate_metrics(
            predictions=all_predictions,
            actuals=all_actuals
        )
        
        return {
            'retraining_interval_days': retraining_interval_days,
            'num_retraining_cycles': len(periods),
            'overall_metrics': overall_metrics,
            'drift_events': drift_events,
            'drift_count': len(drift_events)
        }
    
    def _define_backtest_periods(
        self,
        start_date: datetime,
        end_date: datetime,
        retraining_interval_days: int,
        initial_training_days: int,
        strategy: str
    ) -> List[Dict[str, datetime]]:
        """Define train/test periods for walk-forward validation."""
        periods = []
        
        current_train_start = start_date - timedelta(days=initial_training_days)
        current_train_end = start_date
        current_test_start = start_date
        
        while current_test_start < end_date:
            current_test_end = min(
                current_test_start + timedelta(days=retraining_interval_days),
                end_date
            )
            
            periods.append({
                'train_start': current_train_start,
                'train_end': current_train_end,
                'test_start': current_test_start,
                'test_end': current_test_end
            })
            
            # Move to next period
            if strategy == 'expanding':
                # Expanding window: train_start stays fixed
                current_train_end = current_test_end
            else:  # sliding
                # Sliding window: move both start and end
                current_train_start = current_train_start + timedelta(days=retraining_interval_days)
                current_train_end = current_test_end
            
            current_test_start = current_test_end
        
        return periods
    
    def _determine_optimal_interval(
        self,
        interval_results: Dict[int, Dict[str, Any]]
    ) -> int:
        """Determine optimal retraining interval based on metrics."""
        # Simple heuristic: best MAPE with fewest retraining cycles
        best_interval = None
        best_score = float('inf')
        
        for interval, result in interval_results.items():
            mape = result['overall_metrics']['mape']
            num_cycles = result['num_retraining_cycles']
            
            # Score: MAPE + penalty for frequent retraining
            score = mape + (0.1 * num_cycles)
            
            if score < best_score:
                best_score = score
                best_interval = interval
        
        return best_interval
    
    def _perform_statistical_comparison(
        self,
        interval_results: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform statistical tests comparing intervals."""
        # Paired t-tests between intervals
        tests = {}
        
        intervals = list(interval_results.keys())
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                interval_a = intervals[i]
                interval_b = intervals[j]
                
                # Compare MAPEs (would need period-level metrics)
                # Simplified version here
                mape_a = interval_results[interval_a]['overall_metrics']['mape']
                mape_b = interval_results[interval_b]['overall_metrics']['mape']
                
                tests[f"{interval_a}_vs_{interval_b}"] = {
                    'mape_difference': mape_a - mape_b,
                    'better_interval': interval_a if mape_a < mape_b else interval_b
                }
        
        return tests
    
    def _generate_recommendations(
        self,
        interval_results: Dict[int, Dict[str, Any]],
        optimal_interval: int,
        statistical_tests: Dict[str, Any]
    ) -> str:
        """Generate human-readable recommendations."""
        optimal_result = interval_results[optimal_interval]
        
        recommendation = f"""
Backtesting Recommendations:
----------------------------
Optimal Retraining Interval: {optimal_interval} days

Performance Summary:
- MAPE: {optimal_result['overall_metrics']['mape']:.2f}%
- MAE: {optimal_result['overall_metrics']['mae']:.2f}
- RMSE: {optimal_result['overall_metrics']['rmse']:.2f}

Retraining Cycles: {optimal_result['num_retraining_cycles']}
Drift Events Detected: {optimal_result['drift_count']}

Rationale:
The {optimal_interval}-day interval provides the best balance between forecast accuracy
and operational overhead. Statistical comparison shows this interval outperforms
alternatives with acceptable computational cost.

Next Steps:
1. Implement {optimal_interval}-day retraining schedule in production
2. Monitor drift detection to validate interval effectiveness
3. Re-evaluate interval quarterly or when significant drift detected
"""
        
        return recommendation
    
    async def _load_actuals(
        self,
        start_date: datetime,
        end_date: datetime,
        areas: List[str]
    ) -> Dict[str, np.ndarray]:
        """Load actual values for evaluation."""
        # Implementation would load from data layer
        pass
    
    def _get_all_areas(self) -> List[str]:
        """Get all area identifiers."""
        # Would load from configuration or data catalog
        return []
```

### Integration Points

```python
# Example usage combining all workflows
async def main():
    # 1. Load configuration
    config_manager = ConfigManager('config/config.yaml', env='prod')
    config = config_manager.config
    
    # 2. Execute training workflow
    training_workflow = TrainingWorkflow(config)
    training_results = await training_workflow.execute_training(
        areas=['area_1', 'area_2'],
        models=[ModelType.LGBM, ModelType.RANDOM_FOREST],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 12, 31)
    )
    
    # 3. Execute batch prediction
    prediction_workflow = PredictionWorkflow(config)
    prediction_result = await prediction_workflow.execute_batch_prediction(
        prediction_date=datetime(2025, 1, 1),
        horizons=list(range(9))
    )
    
    # 4. Execute backtesting
    backtesting_workflow = BacktestingWorkflow(config)
    backtest_config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        retraining_intervals=[7, 14, 30],
        validation_strategy='expanding'
    )
    
    backtest_result = await backtesting_workflow.execute_backtest(
        backtest_config=backtest_config
    )
    
    print(backtest_result.recommendations)
```

---

## 🔧 Implementation Plan

### Week 1: Configuration and Training Workflow (Days 1-5)

- **Day 1-2:** Implement configuration management system
  - Pydantic models for all config sections
  - YAML loading with environment overrides
  - Validation with detailed error messages
  - Hot-reload functionality
  
- **Day 3-4:** Implement training workflow orchestrator
  - Task creation with dependency management
  - Parallel execution with concurrency control
  - Model versioning and registry integration
  - Checkpoint mechanism for failure recovery
  
- **Day 5:** Integration testing
  - End-to-end training workflow test
  - Configuration validation scenarios
  - Model dependency enforcement validation

### Week 2: Prediction and Backtesting Workflows (Days 6-10)

- **Day 6-7:** Implement prediction workflows
  - Batch prediction workflow
  - Intraday prediction with BLF strategy
  - Integration with combination and reconciliation
  - Prediction caching mechanism
  
- **Day 8-9:** Implement backtesting workflow
  - Walk-forward validation framework
  - Retraining interval optimization
  - Integration with Epic-07A/07B components
  - Statistical comparison and recommendations
  
- **Day 10:** Comprehensive integration testing
  - End-to-end workflow testing
  - Performance benchmarking
  - Error handling and recovery scenarios

---

## 📊 Success Metrics

### Performance Targets
- **Configuration Loading:** <1 second for full config validation
- **Training Workflow:** Complete 26 series in <4 hours
- **Batch Prediction:** All forecasts in <15 minutes
- **Intraday Prediction:** Updates in <5 minutes
- **Backtesting:** 1-year backtest in <8 hours

### Quality Gates
- [ ] Configuration validation catches all schema errors
- [ ] Training workflow respects all model dependencies
- [ ] Prediction workflows integrate with Epic-05/06 successfully
- [ ] Backtesting provides statistically valid recommendations
- [ ] Checkpoint recovery works after failures
- [ ] 80%+ unit test coverage

### Acceptance Criteria
- [ ] All 4 user stories completed with comprehensive testing
- [ ] Integration with Epic-05A/05B/06A/06B successful
- [ ] Integration with Epic-07A/07B metrics and drift detection working
- [ ] Performance benchmarks met across all workflows
- [ ] Ready for Epic-08B (Execution Engine & Monitoring)

---

## 🧪 Testing Strategy

### Unit Tests
- Configuration loading and validation
- Workflow task creation and dependency management
- Model loading and versioning
- Prediction caching mechanism
- Backtest period generation

### Integration Tests
- End-to-end training workflow
- End-to-end prediction workflow (batch and intraday)
- End-to-end backtesting workflow
- Configuration hot-reload
- Cross-component data flow

### Performance Tests
- Training workflow with 26 series
- Prediction workflow with full hierarchy
- Backtesting with 1-year data
- Concurrent workflow execution
- Memory usage under load

### Validation Tests
- Model dependency enforcement
- Prediction output format validation
- Backtest statistical validity
- Configuration environment overrides

---

## 📚 Dependencies & Risks

### External Dependencies
- **Epic-07B Completion:** Metrics and reporting for backtesting
- **Epic-05A/05B:** Model combination integration
- **Epic-06A/06B:** Hierarchical reconciliation integration
- **Model Registry:** Model versioning and storage

### Python Dependencies
```
pydantic >= 2.0.0           # Configuration validation
pyyaml >= 6.0               # YAML parsing
asyncio (stdlib)            # Async workflow execution
numpy >= 1.24.0             # Numerical operations
pandas >= 2.0.0             # Data handling
```

### Technical Risks & Mitigation
1. **Workflow Complexity:** Modular design, comprehensive testing
2. **Dependency Management:** Clear DAG definition, validation
3. **Performance at Scale:** Parallel execution, caching, optimization
4. **Configuration Errors:** Strong validation, clear error messages

### Business Risks & Mitigation
1. **Workflow Failures:** Checkpoint mechanism, failure recovery
2. **Resource Exhaustion:** Dynamic concurrency limits, monitoring
3. **Backtesting Validity:** Statistical rigor, multiple validation strategies

---

## 🔄 Handoff Criteria

### Deliverables for Epic-08B
- [ ] Configuration management system operational
- [ ] All core workflows (training, prediction, backtesting) working
- [ ] Integration with Epic-05/06/07 components successful
- [ ] Workflow progress tracking implemented
- [ ] Checkpoint and recovery mechanisms tested

### Documentation Requirements
- [ ] Configuration schema reference
- [ ] Workflow architecture documentation
- [ ] Integration guide for Epic-08B
- [ ] Backtesting interpretation guide
- [ ] API reference for workflow components

---

## 📈 Success Definition

Epic-08A is successful when:
1. **Configuration system** handles all parameters with environment-specific validation
2. **Training workflow** orchestrates all 5 models with proper dependency management
3. **Prediction workflows** generate forecasts for batch and intraday scenarios
4. **Backtesting framework** provides statistically valid retraining recommendations
5. **Performance benchmarks** consistently met across all workflows
6. **Integration** with Epic-05/06/07 seamless and robust
7. **Foundation established** for Epic-08B execution engine and monitoring

**Ready for Epic-08B when:** All workflows operational, configuration validated, integration tested, performance benchmarks met, and comprehensive documentation complete.
