# PC-068-08A: Training Workflow Orchestrator

**Ticket ID:** PC-068-08A  
**Epic:** Epic-08A - Core Workflows & Configuration  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Sprint:** Week 21 (Days 3-4) & Week 22 (Day 1)

---

## 📋 Description

Implement orchestrated training workflow that coordinates training of all 5 base models across 26 time series with proper dependency management, parallel execution, model versioning, and checkpoint-based failure recovery. Manages hierarchical model dependencies (hierarchical models depend on end-to-end models) and optimizes execution through area-level parallelization.

---

## 🎯 Acceptance Criteria

- [ ] Train all 5 base models (LGBM, RF, RegDin+SVM, Holt-Winters) systematically
- [ ] Enforce model dependencies: hierarchical models after end-to-end
- [ ] Parallel training across areas with configurable concurrency
- [ ] Model versioning and registration in model registry
- [ ] Progress tracking with percentage completion and logging
- [ ] Checkpoint mechanism for failure recovery
- [ ] Resume training from last checkpoint after failure
- [ ] Complete 26 series training in <4 hours
- [ ] Unit tests with 80%+ coverage

---

## 📐 Technical Specifications

### Core Components

#### 1. Enums and Data Classes

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict

class ModelType(Enum):
    LGBM = "lgbm"
    RANDOM_FOREST = "random_forest"
    REGDIN_SVM = "regdin_svm"
    HOLT_WINTERS = "holt_winters"

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingTask:
    model_type: ModelType
    area: str
    start_date: datetime
    end_date: datetime
    dependencies: List[str] = None  # Task IDs this depends on
    task_id: Optional[str] = None

@dataclass
class TrainingResult:
    task_id: str
    model_type: ModelType
    area: str
    status: WorkflowStatus
    model_version: Optional[str]
    metrics: Dict[str, float]
    duration_seconds: float
    error: Optional[str]
```

#### 2. TrainingWorkflow

```python
class TrainingWorkflow:
    def __init__(self, config: SystemConfig)
    
    async def execute_training(
        self,
        areas: List[str],
        models: List[ModelType],
        start_date: datetime,
        end_date: datetime,
        resume_from_checkpoint: bool = False
    ) -> Dict[str, TrainingResult]
    
    def _create_training_tasks(
        self,
        areas: List[str],
        models: List[ModelType],
        start_date: datetime,
        end_date: datetime
    ) -> List[TrainingTask]
    
    async def _execute_tasks_with_dependencies(
        self,
        tasks: List[TrainingTask]
    ) -> Dict[str, TrainingResult]
    
    async def _execute_task_batch(
        self,
        tasks: List[TrainingTask]
    ) -> Dict[str, TrainingResult]
    
    async def _train_model_for_area(
        self,
        task: TrainingTask
    ) -> TrainingResult
    
    def _generate_training_summary(
        self,
        results: Dict[str, TrainingResult]
    ) -> Dict[str, Any]
```

### Model Dependency Graph

```
End-to-End Models (no dependencies):
- LGBM
- Random Forest

Hierarchical Models (depend on E2E models for same area):
- RegDin + SVM → depends on [LGBM, Random Forest]
- Holt-Winters → depends on [LGBM, Random Forest]

Execution Order:
1. Parallel: LGBM + Random Forest for all areas
2. Parallel: RegDin+SVM + Holt-Winters for all areas (after step 1)
```

### Task Dependency Management

```python
def _create_training_tasks(self, areas, models, start_date, end_date):
    tasks = []
    
    # End-to-end models - no dependencies
    for area in areas:
        for model in [ModelType.LGBM, ModelType.RANDOM_FOREST]:
            if model in models:
                task = TrainingTask(
                    model_type=model,
                    area=area,
                    start_date=start_date,
                    end_date=end_date,
                    dependencies=[],
                    task_id=f"{model.value}_{area}"
                )
                tasks.append(task)
    
    # Hierarchical models - depend on E2E for same area
    for area in areas:
        e2e_deps = [f"lgbm_{area}", f"random_forest_{area}"]
        
        for model in [ModelType.REGDIN_SVM, ModelType.HOLT_WINTERS]:
            if model in models:
                task = TrainingTask(
                    model_type=model,
                    area=area,
                    start_date=start_date,
                    end_date=end_date,
                    dependencies=e2e_deps,
                    task_id=f"{model.value}_{area}"
                )
                tasks.append(task)
    
    return tasks
```

### Checkpoint Mechanism

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: str = ".checkpoints")
    
    def mark_completed(self, task_id: str, result: TrainingResult)
    def get_completed_tasks(self) -> Set[str]
    def clear_checkpoint(self)
    def save_checkpoint(self, state: Dict[str, Any])
    def load_checkpoint(self) -> Optional[Dict[str, Any]]
```

---

## 🔧 Implementation Tasks

### Task 1: Core Data Structures (2 hours)
- [ ] Implement `ModelType` and `WorkflowStatus` enums
- [ ] Implement `TrainingTask` dataclass
- [ ] Implement `TrainingResult` dataclass
- [ ] Add validation and type hints

### Task 2: Task Creation with Dependencies (4 hours)
- [ ] Implement `_create_training_tasks()` method
- [ ] Define end-to-end models (LGBM, RF)
- [ ] Define hierarchical models with dependencies
- [ ] Generate unique task IDs
- [ ] Validate dependency references

### Task 3: Dependency Resolution Engine (5 hours)
- [ ] Implement `_execute_tasks_with_dependencies()`
- [ ] Build ready-task identification logic
- [ ] Detect circular dependencies
- [ ] Track completed tasks
- [ ] Handle failed task dependencies

### Task 4: Parallel Execution (4 hours)
- [ ] Implement `_execute_task_batch()` with asyncio
- [ ] Add semaphore-based concurrency control
- [ ] Group tasks by area for optimization
- [ ] Handle exceptions in parallel execution
- [ ] Map results back to task IDs

### Task 5: Individual Model Training (5 hours)
- [ ] Implement `_train_model_for_area()`
- [ ] Integrate with data loader
- [ ] Integrate with model trainers (factory pattern)
- [ ] Calculate training metrics
- [ ] Register trained models with versioning

### Task 6: Checkpoint Management (4 hours)
- [ ] Implement `CheckpointManager` class
- [ ] Save completed tasks to checkpoint file
- [ ] Load checkpoint on resume
- [ ] Filter out completed tasks when resuming
- [ ] Add checkpoint cleanup

### Task 7: Progress Tracking and Logging (3 hours)
- [ ] Add structured logging for task lifecycle
- [ ] Implement progress percentage calculation
- [ ] Log task start/completion/failure
- [ ] Generate training summary statistics
- [ ] Add duration tracking

### Task 8: Integration and Testing (5 hours)
- [ ] Unit tests for task creation
- [ ] Unit tests for dependency resolution
- [ ] Integration tests for full workflow
- [ ] Test checkpoint recovery
- [ ] Performance testing with 26 areas

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_training_task_creation()
def test_task_dependencies_e2e_models()
def test_task_dependencies_hierarchical_models()
def test_dependency_resolution_ready_tasks()
def test_dependency_resolution_circular_detection()
def test_parallel_execution_concurrency_limit()
def test_individual_model_training()
def test_model_registration()
def test_checkpoint_save()
def test_checkpoint_load()
def test_checkpoint_resume_filtering()
def test_training_summary_generation()
```

### Integration Tests
```python
def test_full_training_workflow_2_areas()
def test_full_training_workflow_26_areas()
def test_training_with_checkpoint_resume()
def test_training_with_partial_failure()
def test_dependency_enforcement()
```

### Performance Tests
```python
def test_training_duration_26_areas()
def test_parallel_speedup()
def test_memory_usage()
```

---

## 📊 Success Metrics

### Performance Targets
- Full training (26 areas, 5 models): **<4 hours**
- Single model training: **<10 minutes per area**
- Checkpoint overhead: **<1% of total time**
- Parallel speedup: **>50%** vs sequential

### Quality Targets
- Unit test coverage: **≥80%**
- Dependency enforcement: **100%** (no violations)
- Checkpoint recovery: **100%** success rate
- Training success rate: **>95%** (with retries)

---

## 📚 Dependencies

### Required Packages
```python
asyncio (stdlib)            # Async workflow execution
numpy >= 1.24.0
pandas >= 2.0.0
```

### Code Dependencies
- **Requires:** PC-067-08A (ConfigManager)
- **Requires:** Epic-04 model trainers (LGBM, RF, RegDin+SVM, HW)
- **Requires:** Model registry for versioning

---

## 🔗 Related Tickets
- **Depends on:** PC-067-08A (Configuration Manager)
- **Relates to:** Epic-04 (Model Training)
- **Blocks:** PC-069-08A (Prediction Workflow)

---

## 📖 Documentation Requirements

- [ ] Training workflow architecture diagram
- [ ] Model dependency graph documentation
- [ ] Checkpoint recovery guide
- [ ] Parallel execution configuration guide
- [ ] Model versioning conventions
- [ ] Troubleshooting guide for common failures

---

## 💡 Implementation Notes

### Asyncio Parallel Execution
```python
import asyncio

async def execute_with_concurrency_limit(tasks, max_concurrent):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_limit(task):
        async with semaphore:
            return await train_model(task)
    
    results = await asyncio.gather(
        *[execute_with_limit(task) for task in tasks],
        return_exceptions=True
    )
    return results
```

### Checkpoint File Format (JSON)
```json
{
  "timestamp": "2025-11-19T10:30:00",
  "workflow_id": "training_20251119",
  "completed_tasks": {
    "lgbm_area_01": {
      "status": "completed",
      "model_version": "v1.2.3",
      "duration_seconds": 450.2
    }
  }
}
```

### Task ID Convention
```python
task_id = f"{model_type.value}_{area}"
# Examples:
# - "lgbm_area_01"
# - "random_forest_area_02"
# - "regdin_svm_area_03"
```

### Progress Tracking
```python
def calculate_progress(completed, total):
    progress_pct = (completed / total) * 100
    logger.info(f"Training progress: {completed}/{total} ({progress_pct:.1f}%)")
```

### Model Trainer Factory
```python
class ModelTrainerFactory:
    @staticmethod
    def create(model_type: ModelType, config: ModelsConfig):
        if model_type == ModelType.LGBM:
            return LGBMTrainer(config.lgbm)
        elif model_type == ModelType.RANDOM_FOREST:
            return RandomForestTrainer(config.random_forest)
        # ... etc
```

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥80% coverage
- [ ] All 5 model types supported
- [ ] Dependency management working correctly
- [ ] Parallel execution with configurable concurrency
- [ ] Checkpoint mechanism functional
- [ ] Model versioning integrated
- [ ] Training completes in <4 hours for 26 areas
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] Integration tested with PC-067-08A
- [ ] Ready for PC-069-08A (Prediction Workflow)
