# PC-071-08B: Parallel Execution Engine

**Ticket ID:** PC-071-08B  
**Epic:** Epic-08B (Execution Engine & Monitoring)  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Sprint:** Week 23 (Days 1-3)  

---

## 📋 Description

Implement an advanced parallel execution engine that optimizes workflow performance through AsyncIO-based task parallelization, dynamic resource allocation, and fault tolerance. The engine will reduce workflow execution time by >60% while maintaining optimal resource utilization and graceful degradation on failures.

### Context
Building on Epic-08A's workflow foundation (TrainingWorkflow, PredictionWorkflow, BacktestingWorkflow), this ticket provides the execution infrastructure that enables concurrent processing across multiple dimensions (areas and models) with intelligent resource management and progress tracking.

### Business Value
- **Performance Optimization:** >60% reduction in workflow execution time through parallelization
- **Resource Efficiency:** CPU 70-85%, Memory <80% utilization through dynamic adjustment
- **Reliability:** >95% task success rate with fault tolerance and retry logic
- **Operational Visibility:** Real-time progress tracking for all parallel operations

---

## ✅ Acceptance Criteria

### Functional Requirements
- [ ] Parallel execution across 26 areas (time series)
- [ ] Parallel execution across 5 base models (LGBM, RF, RegDin+SVM, HW)
- [ ] Three parallelization strategies: AREA_FIRST, MODEL_FIRST, HYBRID
- [ ] Dynamic resource allocation based on system load
- [ ] Fault tolerance with graceful degradation on individual task failures
- [ ] Progress tracking with real-time percentage completion
- [ ] Configurable concurrency limits with semaphore control
- [ ] Task retry with exponential backoff (max 3 retries)
- [ ] Task timeout support (configurable per task)
- [ ] Task priority-based execution ordering

### Performance Requirements
- [ ] >60% reduction in sequential execution time
- [ ] Resource utilization: CPU 70-85%, Memory <80%
- [ ] >95% task success rate under normal conditions
- [ ] <30 seconds to adapt to load changes
- [ ] Optimal concurrency calculation based on system resources

### Quality Requirements
- [ ] 80%+ unit test coverage
- [ ] Integration with Epic-08A workflows (Training, Prediction, Backtesting)
- [ ] Comprehensive error handling and reporting
- [ ] Resource monitoring overhead <5% of total execution time

---

## 🔧 Technical Specifications

### Core Components

#### 1. ParallelStrategy Enum
```python
class ParallelStrategy(Enum):
    """Parallelization strategy."""
    AREA_FIRST = "area_first"      # Parallelize by areas, then models
    MODEL_FIRST = "model_first"    # Parallelize by models, then areas
    HYBRID = "hybrid"               # Mixed parallelization
```

**Purpose:** Define execution strategies for different use cases
- AREA_FIRST: Optimize for area-level parallelism (training, prediction)
- MODEL_FIRST: Optimize for model-level parallelism (comparative analysis)
- HYBRID: Balanced approach for complex workflows

#### 2. ExecutionTask Data Class
```python
@dataclass
class ExecutionTask:
    """Task specification for parallel execution."""
    task_id: str                           # Unique task identifier
    task_type: str                         # 'training', 'prediction', 'evaluation'
    callable: Callable                     # Async function to execute
    args: tuple                            # Positional arguments
    kwargs: dict                           # Keyword arguments
    priority: int = 0                      # Execution priority (higher = first)
    max_retries: int = 3                   # Maximum retry attempts
    timeout_seconds: Optional[int] = None  # Task timeout (None = no limit)
```

**Fields:**
- task_id: UUID or descriptive identifier (e.g., "train_area_01_lgbm")
- task_type: Categorizes task for logging and monitoring
- callable: Must be async function or coroutine
- priority: Used for sorting tasks within groups
- max_retries: Failed tasks retry with exponential backoff
- timeout_seconds: Prevents hanging tasks

#### 3. TaskResult Data Class
```python
@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str                           # Matches ExecutionTask.task_id
    status: str                            # 'success', 'failed', 'timeout'
    result: Optional[Any]                  # Task return value (if success)
    error: Optional[str]                   # Error message (if failed)
    duration_seconds: float                # Total execution time
    retries_attempted: int                 # Number of retries performed
    resource_usage: Dict[str, float]       # CPU, memory, I/O metrics
```

**Status Values:**
- success: Task completed successfully
- failed: Task failed after all retries
- timeout: Task exceeded timeout_seconds

#### 4. ResourceMonitor Class
```python
class ResourceMonitor:
    """Monitor system resource usage using psutil."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current resource usage.
        
        Returns:
            {
                'cpu_percent': float,      # Current CPU utilization (0-100)
                'memory_percent': float,   # Current memory utilization (0-100)
                'memory_mb': float,        # Process memory in MB
                'num_threads': int,        # Active threads
                'io_read_mb': float,       # Cumulative I/O reads in MB
                'io_write_mb': float       # Cumulative I/O writes in MB
            }
        """
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'num_threads': self.process.num_threads(),
            'io_read_mb': self.process.io_counters().read_bytes / 1024 / 1024,
            'io_write_mb': self.process.io_counters().write_bytes / 1024 / 1024
        }
    
    def is_overloaded(self, thresholds: Dict[str, float]) -> bool:
        """
        Check if system exceeds resource thresholds.
        
        Args:
            thresholds: {'cpu_percent': 85, 'memory_percent': 80}
            
        Returns:
            True if any threshold exceeded
        """
        usage = self.get_current_usage()
        
        if usage['cpu_percent'] > thresholds.get('cpu_percent', 90):
            return True
        if usage['memory_percent'] > thresholds.get('memory_percent', 85):
            return True
        
        return False
```

**Key Methods:**
- get_current_usage(): Real-time metrics using psutil
- is_overloaded(): Boolean check against configurable thresholds
- Interval=0.1 for cpu_percent balances accuracy and overhead

#### 5. ProgressTracker Class
```python
class ProgressTracker:
    """Track progress of parallel execution."""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.running_tasks = 0
        self.start_time = datetime.now()
    
    def task_started(self):
        """Mark task as started."""
        self.running_tasks += 1
    
    def task_completed(self, success: bool = True):
        """Mark task as completed or failed."""
        self.running_tasks -= 1
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress metrics.
        
        Returns:
            {
                'total_tasks': int,
                'completed': int,
                'failed': int,
                'running': int,
                'pending': int,
                'progress_percent': float,
                'elapsed_seconds': float,
                'eta_seconds': Optional[float]
            }
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        completion_rate = self.completed_tasks / elapsed if elapsed > 0 else 0
        eta_seconds = (self.total_tasks - self.completed_tasks) / completion_rate if completion_rate > 0 else None
        
        return {
            'total_tasks': self.total_tasks,
            'completed': self.completed_tasks,
            'failed': self.failed_tasks,
            'running': self.running_tasks,
            'pending': self.total_tasks - self.completed_tasks - self.failed_tasks - self.running_tasks,
            'progress_percent': (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0,
            'elapsed_seconds': elapsed,
            'eta_seconds': eta_seconds
        }
```

**Features:**
- Real-time tracking of task states
- Progress percentage calculation
- ETA estimation based on completion rate
- Thread-safe operations for async context

#### 6. ParallelExecutor Class (Main Engine)
```python
class ParallelExecutor:
    """Advanced parallel execution engine with resource optimization."""
    
    def __init__(
        self,
        max_concurrent_tasks: int = 4,
        resource_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize parallel executor.
        
        Args:
            max_concurrent_tasks: Maximum concurrent tasks (default: 4)
            resource_thresholds: {'cpu_percent': 85, 'memory_percent': 80}
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.resource_thresholds = resource_thresholds or {
            'cpu_percent': 85,
            'memory_percent': 80
        }
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.resource_monitor = ResourceMonitor()
        self.active_tasks = set()  # Track active task IDs
    
    async def execute_parallel_tasks(
        self,
        tasks: List[ExecutionTask],
        strategy: ParallelStrategy = ParallelStrategy.AREA_FIRST,
        adaptive_concurrency: bool = True
    ) -> List[TaskResult]:
        """
        Execute tasks in parallel with resource monitoring and fault tolerance.
        
        Args:
            tasks: List of ExecutionTask objects to execute
            strategy: Parallelization strategy (AREA_FIRST, MODEL_FIRST, HYBRID)
            adaptive_concurrency: Enable dynamic concurrency adjustment
            
        Returns:
            List of TaskResult objects (one per task)
        """
        # Initialize progress tracker
        progress_tracker = ProgressTracker(total_tasks=len(tasks))
        
        # Group tasks by strategy
        task_groups = self._group_tasks_by_strategy(tasks, strategy)
        
        # Execute task groups
        all_results = []
        
        for group in task_groups:
            # Adjust concurrency if adaptive mode enabled
            if adaptive_concurrency:
                await self._adjust_concurrency_dynamically()
            
            # Execute task group in parallel
            group_results = await self._execute_task_group(
                group,
                progress_tracker
            )
            
            all_results.extend(group_results)
        
        return all_results
    
    def _group_tasks_by_strategy(
        self,
        tasks: List[ExecutionTask],
        strategy: ParallelStrategy
    ) -> List[List[ExecutionTask]]:
        """
        Group tasks according to parallelization strategy.
        
        Args:
            tasks: List of ExecutionTask objects
            strategy: AREA_FIRST, MODEL_FIRST, or HYBRID
            
        Returns:
            List of task groups (each group executed in parallel)
        """
        if strategy == ParallelStrategy.AREA_FIRST:
            # Group by area, then by model
            # Example: All models for Area 01, then all models for Area 02, etc.
            area_groups = {}
            for task in tasks:
                area = task.kwargs.get('area', 'default')
                if area not in area_groups:
                    area_groups[area] = []
                area_groups[area].append(task)
            return list(area_groups.values())
        
        elif strategy == ParallelStrategy.MODEL_FIRST:
            # Group by model, then by area
            # Example: LGBM for all areas, then RF for all areas, etc.
            model_groups = {}
            for task in tasks:
                model = task.kwargs.get('model_type', 'default')
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append(task)
            return list(model_groups.values())
        
        else:  # HYBRID
            # Single group for hybrid execution
            # All tasks executed with maximum parallelism
            return [tasks]
    
    async def _execute_task_group(
        self,
        tasks: List[ExecutionTask],
        progress_tracker: ProgressTracker
    ) -> List[TaskResult]:
        """
        Execute a group of tasks in parallel.
        
        Args:
            tasks: Tasks in this group
            progress_tracker: Shared progress tracker
            
        Returns:
            List of TaskResult objects
        """
        # Sort by priority (higher priority first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Execute with semaphore limit
        task_coroutines = [
            self._execute_task_with_monitoring(task, progress_tracker)
            for task in sorted_tasks
        ]
        
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        # Convert exceptions to failed TaskResults
        processed_results = []
        for task, result in zip(sorted_tasks, results):
            if isinstance(result, Exception):
                processed_results.append(TaskResult(
                    task_id=task.task_id,
                    status='failed',
                    result=None,
                    error=str(result),
                    duration_seconds=0,
                    retries_attempted=task.max_retries,
                    resource_usage={}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_task_with_monitoring(
        self,
        task: ExecutionTask,
        progress_tracker: ProgressTracker
    ) -> TaskResult:
        """
        Execute single task with resource monitoring and retry logic.
        
        Args:
            task: ExecutionTask to execute
            progress_tracker: Shared progress tracker
            
        Returns:
            TaskResult with execution details
        """
        async with self.semaphore:
            progress_tracker.task_started()
            self.active_tasks.add(task.task_id)
            
            start_time = datetime.now()
            retries_attempted = 0
            last_error = None
            
            while retries_attempted < task.max_retries:
                try:
                    # Check resource availability
                    if self.resource_monitor.is_overloaded(self.resource_thresholds):
                        await asyncio.sleep(5)  # Wait for resources
                        continue
                    
                    # Execute task with timeout
                    if task.timeout_seconds:
                        result = await asyncio.wait_for(
                            task.callable(*task.args, **task.kwargs),
                            timeout=task.timeout_seconds
                        )
                    else:
                        result = await task.callable(*task.args, **task.kwargs)
                    
                    # Success
                    duration = (datetime.now() - start_time).total_seconds()
                    resource_usage = self.resource_monitor.get_current_usage()
                    
                    progress_tracker.task_completed(success=True)
                    self.active_tasks.remove(task.task_id)
                    
                    return TaskResult(
                        task_id=task.task_id,
                        status='success',
                        result=result,
                        error=None,
                        duration_seconds=duration,
                        retries_attempted=retries_attempted,
                        resource_usage=resource_usage
                    )
                
                except asyncio.TimeoutError:
                    last_error = f"Task timeout after {task.timeout_seconds}s"
                    retries_attempted += 1
                    if retries_attempted < task.max_retries:
                        await asyncio.sleep(2 ** retries_attempted)  # Exponential backoff
                
                except Exception as e:
                    last_error = str(e)
                    retries_attempted += 1
                    if retries_attempted < task.max_retries:
                        await asyncio.sleep(2 ** retries_attempted)  # Exponential backoff
            
            # All retries exhausted
            duration = (datetime.now() - start_time).total_seconds()
            progress_tracker.task_completed(success=False)
            self.active_tasks.remove(task.task_id)
            
            return TaskResult(
                task_id=task.task_id,
                status='failed',
                result=None,
                error=last_error,
                duration_seconds=duration,
                retries_attempted=retries_attempted,
                resource_usage=self.resource_monitor.get_current_usage()
            )
    
    async def _adjust_concurrency_dynamically(self):
        """
        Dynamically adjust concurrency based on resource usage.
        
        Increases concurrency if system underutilized.
        Decreases concurrency if system overloaded.
        """
        usage = self.resource_monitor.get_current_usage()
        
        current_limit = self.semaphore._value + len(self.active_tasks)
        
        # Reduce concurrency if overloaded
        if usage['cpu_percent'] > 85 or usage['memory_percent'] > 80:
            new_limit = max(1, current_limit - 1)
            self.semaphore = asyncio.Semaphore(new_limit)
        
        # Increase concurrency if underutilized
        elif usage['cpu_percent'] < 60 and usage['memory_percent'] < 60:
            new_limit = min(self.max_concurrent_tasks, current_limit + 1)
            self.semaphore = asyncio.Semaphore(new_limit)
    
    def calculate_optimal_concurrency(self) -> int:
        """
        Calculate optimal concurrency based on system resources.
        
        Returns:
            Recommended number of concurrent tasks
        """
        cpu_count = psutil.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        
        # Conservative estimate: 1 task per 2 cores, 2GB memory per task
        cpu_limit = max(1, cpu_count // 2)
        memory_limit = max(1, int(available_memory_gb // 2))
        
        return min(cpu_limit, memory_limit, self.max_concurrent_tasks)
```

**Key Methods:**
- execute_parallel_tasks(): Main entry point for parallel execution
- _group_tasks_by_strategy(): Implements three parallelization strategies
- _execute_task_group(): Executes one group with asyncio.gather()
- _execute_task_with_monitoring(): Single task execution with retry and timeout
- _adjust_concurrency_dynamically(): Real-time concurrency adjustment
- calculate_optimal_concurrency(): Static calculation based on hardware

**Design Patterns:**
- Semaphore pattern: Controls concurrency
- Retry pattern: Exponential backoff (2^n seconds)
- Circuit breaker pattern: Resource overload detection
- Observer pattern: Progress tracking

### Integration Points

#### Epic-08A Workflows
```python
# Integration with TrainingWorkflow
from src.workflows.training_workflow import TrainingWorkflow

async def train_with_parallel_execution(
    areas: List[str],
    models: List[str],
    start_date: datetime,
    end_date: datetime
):
    """Execute training workflow with parallel execution."""
    executor = ParallelExecutor(max_concurrent_tasks=4)
    
    # Create training tasks
    tasks = []
    for area in areas:
        for model in models:
            task = ExecutionTask(
                task_id=f"train_{area}_{model}",
                task_type='training',
                callable=train_single_model_async,
                args=(),
                kwargs={
                    'area': area,
                    'model_type': model,
                    'start_date': start_date,
                    'end_date': end_date
                },
                priority=0,
                max_retries=3,
                timeout_seconds=3600  # 1 hour per model
            )
            tasks.append(task)
    
    # Execute with AREA_FIRST strategy
    results = await executor.execute_parallel_tasks(
        tasks,
        strategy=ParallelStrategy.AREA_FIRST,
        adaptive_concurrency=True
    )
    
    return results


# Integration with PredictionWorkflow
async def predict_with_parallel_execution(
    areas: List[str],
    prediction_date: datetime
):
    """Execute prediction workflow with parallel execution."""
    executor = ParallelExecutor(max_concurrent_tasks=8)
    
    # Create prediction tasks
    tasks = []
    for area in areas:
        task = ExecutionTask(
            task_id=f"predict_{area}_{prediction_date.isoformat()}",
            task_type='prediction',
            callable=predict_single_area_async,
            args=(),
            kwargs={
                'area': area,
                'prediction_date': prediction_date
            },
            priority=1 if area in ['area_01', 'area_26'] else 0,  # Prioritize critical areas
            max_retries=2,
            timeout_seconds=300  # 5 minutes per area
        )
        tasks.append(task)
    
    # Execute with HYBRID strategy (maximum parallelism)
    results = await executor.execute_parallel_tasks(
        tasks,
        strategy=ParallelStrategy.HYBRID,
        adaptive_concurrency=True
    )
    
    return results
```

### Performance Characteristics

#### Expected Speedup
- **Sequential Execution:** ~4 hours for 26 areas × 5 models = 130 tasks
- **Parallel Execution (4 concurrent):** ~1.5 hours (60% reduction)
- **Parallel Execution (8 concurrent):** ~1 hour (75% reduction)

#### Resource Usage Patterns
- **CPU:** 70-85% during parallel execution (optimal range)
- **Memory:** Peak at model loading, <80% threshold
- **I/O:** Disk-bound during data loading, CPU-bound during training
- **Network:** Minimal (local execution)

#### Retry Logic
- **Backoff Schedule:** 2s, 4s, 8s (exponential)
- **Max Retries:** 3 per task
- **Total Time:** Original + 14s worst case per retry
- **Success Rate:** >95% after retries

---

## 📝 Implementation Tasks

### Task 1: Core Data Structures (Day 1 Morning)
**Estimated Time:** 2 hours

1. **Define ParallelStrategy enum:**
   - AREA_FIRST, MODEL_FIRST, HYBRID values
   - Docstrings explaining each strategy

2. **Implement ExecutionTask dataclass:**
   - Required fields: task_id, task_type, callable, args, kwargs
   - Optional fields: priority, max_retries, timeout_seconds
   - Validation: callable must be async function
   - Type hints for all fields

3. **Implement TaskResult dataclass:**
   - Fields: task_id, status, result, error, duration_seconds, retries_attempted, resource_usage
   - Status enum: success, failed, timeout
   - Type hints and default values

4. **Write unit tests:**
   - Test data class instantiation
   - Test field validation
   - Test default values

**Deliverables:**
- `src/execution/parallel_executor.py` with data structures
- `tests/execution/test_parallel_executor.py` with basic tests

**Acceptance:**
- All data classes defined with proper type hints
- Unit tests pass with 100% coverage

---

### Task 2: ResourceMonitor Implementation (Day 1 Afternoon)
**Estimated Time:** 3 hours

1. **Implement ResourceMonitor class:**
   - __init__: Initialize psutil.Process()
   - get_current_usage(): Return dict with CPU, memory, I/O metrics
   - is_overloaded(): Check against thresholds

2. **Add resource metrics:**
   - CPU percent (interval=0.1 for fast response)
   - Memory percent and MB
   - Thread count
   - I/O read/write MB (cumulative)

3. **Add threshold checking:**
   - Configurable CPU threshold (default 85%)
   - Configurable memory threshold (default 80%)
   - Boolean return value

4. **Write unit tests:**
   - Mock psutil calls
   - Test get_current_usage() format
   - Test is_overloaded() with various thresholds
   - Test edge cases (missing metrics)

**Deliverables:**
- Complete ResourceMonitor class
- Unit tests with mocked psutil

**Acceptance:**
- ResourceMonitor returns accurate metrics
- is_overloaded() correctly identifies threshold violations
- <5% overhead for monitoring

---

### Task 3: ProgressTracker Implementation (Day 1 Evening)
**Estimated Time:** 2 hours

1. **Implement ProgressTracker class:**
   - __init__: Initialize counters and start_time
   - task_started(): Increment running_tasks
   - task_completed(): Update completed/failed counters
   - get_progress(): Calculate progress metrics

2. **Add progress metrics:**
   - Total, completed, failed, running, pending counts
   - Progress percentage
   - Elapsed time
   - ETA calculation (completion rate based)

3. **Thread safety:**
   - Use asyncio.Lock if needed
   - Atomic operations for counters

4. **Write unit tests:**
   - Test task state transitions
   - Test progress percentage calculation
   - Test ETA estimation
   - Test edge cases (zero tasks, all failed)

**Deliverables:**
- Complete ProgressTracker class
- Comprehensive unit tests

**Acceptance:**
- ProgressTracker accurately tracks task states
- ETA estimation reasonable (±10% accuracy)
- Thread-safe for async context

---

### Task 4: Task Grouping and Strategy Logic (Day 2 Morning)
**Estimated Time:** 3 hours

1. **Implement _group_tasks_by_strategy():**
   - AREA_FIRST: Group by area from kwargs
   - MODEL_FIRST: Group by model_type from kwargs
   - HYBRID: Single group with all tasks
   - Handle missing area/model_type (default fallback)

2. **Add task sorting by priority:**
   - Within each group, sort by priority (descending)
   - Maintain stable sort for equal priorities

3. **Add validation:**
   - Ensure all tasks have valid callable
   - Check for duplicate task_ids (optional warning)

4. **Write unit tests:**
   - Test each strategy with sample tasks
   - Test priority sorting within groups
   - Test edge cases (empty tasks, no area/model info)

**Deliverables:**
- _group_tasks_by_strategy() implementation
- Strategy-specific unit tests

**Acceptance:**
- Tasks correctly grouped by strategy
- Priority sorting works within groups
- Edge cases handled gracefully

---

### Task 5: Single Task Execution with Retry (Day 2 Afternoon)
**Estimated Time:** 4 hours

1. **Implement _execute_task_with_monitoring():**
   - Semaphore acquisition (async with)
   - Progress tracker integration
   - Resource overload checking (wait if overloaded)
   - Task execution with timeout
   - Retry loop with exponential backoff
   - TaskResult creation

2. **Add timeout handling:**
   - Use asyncio.wait_for() if timeout_seconds set
   - Catch asyncio.TimeoutError
   - Include timeout in error message

3. **Add retry logic:**
   - Loop up to max_retries
   - Exponential backoff: 2^n seconds
   - Track retries_attempted
   - Log each retry attempt

4. **Add resource usage capture:**
   - Call resource_monitor.get_current_usage() on success/failure
   - Include in TaskResult

5. **Write unit tests:**
   - Test successful execution
   - Test timeout handling
   - Test retry with eventual success
   - Test retry with eventual failure
   - Test resource overload waiting
   - Mock async functions and time

**Deliverables:**
- Complete _execute_task_with_monitoring()
- Comprehensive unit tests with mocks

**Acceptance:**
- Tasks execute successfully in normal conditions
- Timeouts handled correctly
- Retry logic works with exponential backoff
- Resource usage captured accurately

---

### Task 6: Parallel Execution and Group Handling (Day 2 Evening + Day 3 Morning)
**Estimated Time:** 4 hours

1. **Implement _execute_task_group():**
   - Sort tasks by priority
   - Create coroutines for all tasks
   - Use asyncio.gather() with return_exceptions=True
   - Convert exceptions to failed TaskResults
   - Return processed results

2. **Implement execute_parallel_tasks():**
   - Initialize ProgressTracker
   - Group tasks by strategy
   - Iterate over groups
   - Call _adjust_concurrency_dynamically() if enabled
   - Execute each group
   - Aggregate results

3. **Add exception handling:**
   - Catch exceptions from asyncio.gather()
   - Create TaskResult for unexpected failures
   - Ensure all tasks accounted for in results

4. **Write integration tests:**
   - Test execution with multiple task groups
   - Test different strategies (AREA_FIRST, MODEL_FIRST, HYBRID)
   - Test with successful and failing tasks
   - Test progress tracking across groups

**Deliverables:**
- Complete execute_parallel_tasks() and _execute_task_group()
- Integration tests with real async functions

**Acceptance:**
- All tasks execute in correct order
- Strategies correctly affect grouping
- Progress tracked across all groups
- Exceptions handled gracefully

---

### Task 7: Dynamic Concurrency Adjustment (Day 3 Afternoon)
**Estimated Time:** 3 hours

1. **Implement _adjust_concurrency_dynamically():**
   - Get current resource usage
   - Calculate current concurrency limit
   - Reduce if CPU >85% or memory >80%
   - Increase if CPU <60% and memory <60%
   - Recreate semaphore with new limit
   - Respect max_concurrent_tasks limit

2. **Implement calculate_optimal_concurrency():**
   - Get CPU count from psutil
   - Get available memory from psutil
   - Calculate CPU limit (cpu_count // 2)
   - Calculate memory limit (available_gb // 2)
   - Return minimum of CPU, memory, max limits

3. **Add logging:**
   - Log concurrency adjustments (will integrate with PC-072)
   - Log resource usage at adjustment time

4. **Write unit tests:**
   - Mock resource usage at different levels
   - Test concurrency reduction when overloaded
   - Test concurrency increase when underutilized
   - Test calculate_optimal_concurrency() with various hardware
   - Test respect for max_concurrent_tasks limit

**Deliverables:**
- Complete dynamic concurrency methods
- Unit tests with mocked resource usage

**Acceptance:**
- Concurrency adjusts based on resource usage
- Optimal concurrency calculation reasonable
- Respects configured maximum
- <30 seconds to adapt to load changes

---

### Task 8: Integration with Epic-08A Workflows (Day 3 Evening)
**Estimated Time:** 2 hours

1. **Create integration helpers:**
   - train_with_parallel_execution() function
   - predict_with_parallel_execution() function
   - backtest_with_parallel_execution() function

2. **Update Epic-08A workflows:**
   - Modify TrainingWorkflow to use ParallelExecutor
   - Modify PredictionWorkflow to use ParallelExecutor
   - Modify BacktestingWorkflow to use ParallelExecutor

3. **Add configuration:**
   - Add parallel execution settings to SystemConfig
   - max_concurrent_tasks, resource_thresholds, adaptive_concurrency
   - Default values for production

4. **Write integration tests:**
   - End-to-end training with parallel execution
   - End-to-end prediction with parallel execution
   - Test actual speedup (measure sequential vs parallel)

**Deliverables:**
- Integration helpers and updated workflows
- Configuration additions
- End-to-end integration tests

**Acceptance:**
- Epic-08A workflows use ParallelExecutor
- Configuration properly integrated
- >60% speedup demonstrated in tests

---

### Task 9: Testing and Documentation (Day 3 Evening)
**Estimated Time:** 2 hours

1. **Performance testing:**
   - Benchmark sequential vs parallel execution
   - Measure resource utilization
   - Test with 1, 2, 4, 8, 16 concurrent tasks
   - Document optimal concurrency for system

2. **Stress testing:**
   - Test with 100+ tasks
   - Test with failing tasks (50% failure rate)
   - Test with resource-constrained system
   - Test adaptive concurrency under load

3. **Documentation:**
   - Add docstrings to all public methods
   - Create usage guide with examples
   - Document parallelization strategies
   - Document resource tuning recommendations

4. **Code review preparation:**
   - Run linters (flake8, mypy, black)
   - Ensure 80%+ test coverage
   - Add type hints to all functions
   - Clean up debug code

**Deliverables:**
- Performance benchmarks
- Stress test results
- Complete documentation
- Code review ready

**Acceptance:**
- 80%+ test coverage
- Performance targets met (>60% speedup)
- Documentation complete
- Code quality high (no linter errors)

---

## 🧪 Testing Requirements

### Unit Tests

#### 1. Data Structures Tests
```python
def test_execution_task_creation():
    """Test ExecutionTask dataclass creation."""
    async def dummy_func():
        return "result"
    
    task = ExecutionTask(
        task_id="test_001",
        task_type="training",
        callable=dummy_func,
        args=(),
        kwargs={'area': 'area_01'},
        priority=1
    )
    
    assert task.task_id == "test_001"
    assert task.priority == 1
    assert task.max_retries == 3  # Default


def test_task_result_creation():
    """Test TaskResult dataclass creation."""
    result = TaskResult(
        task_id="test_001",
        status="success",
        result="output",
        error=None,
        duration_seconds=1.5,
        retries_attempted=0,
        resource_usage={'cpu_percent': 50.0}
    )
    
    assert result.status == "success"
    assert result.duration_seconds == 1.5
```

#### 2. ResourceMonitor Tests
```python
def test_resource_monitor_get_usage(mocker):
    """Test resource usage retrieval."""
    mocker.patch('psutil.cpu_percent', return_value=75.0)
    mocker.patch('psutil.virtual_memory', return_value=MagicMock(percent=60.0))
    
    monitor = ResourceMonitor()
    usage = monitor.get_current_usage()
    
    assert usage['cpu_percent'] == 75.0
    assert usage['memory_percent'] == 60.0
    assert 'memory_mb' in usage


def test_resource_monitor_overload_detection(mocker):
    """Test overload detection."""
    mocker.patch('psutil.cpu_percent', return_value=90.0)
    mocker.patch('psutil.virtual_memory', return_value=MagicMock(percent=70.0))
    
    monitor = ResourceMonitor()
    thresholds = {'cpu_percent': 85, 'memory_percent': 80}
    
    assert monitor.is_overloaded(thresholds) is True
```

#### 3. ProgressTracker Tests
```python
def test_progress_tracker_basic():
    """Test basic progress tracking."""
    tracker = ProgressTracker(total_tasks=10)
    
    tracker.task_started()
    assert tracker.running_tasks == 1
    
    tracker.task_completed(success=True)
    assert tracker.completed_tasks == 1
    assert tracker.running_tasks == 0
    
    progress = tracker.get_progress()
    assert progress['progress_percent'] == 10.0


def test_progress_tracker_eta():
    """Test ETA estimation."""
    import time
    
    tracker = ProgressTracker(total_tasks=10)
    
    time.sleep(0.1)
    tracker.task_completed(success=True)
    
    progress = tracker.get_progress()
    assert progress['eta_seconds'] is not None
    assert progress['eta_seconds'] > 0
```

#### 4. Task Grouping Tests
```python
@pytest.mark.asyncio
async def test_group_tasks_area_first():
    """Test AREA_FIRST strategy."""
    async def dummy():
        pass
    
    tasks = [
        ExecutionTask("t1", "train", dummy, (), {'area': 'area_01', 'model_type': 'lgbm'}),
        ExecutionTask("t2", "train", dummy, (), {'area': 'area_01', 'model_type': 'rf'}),
        ExecutionTask("t3", "train", dummy, (), {'area': 'area_02', 'model_type': 'lgbm'}),
    ]
    
    executor = ParallelExecutor()
    groups = executor._group_tasks_by_strategy(tasks, ParallelStrategy.AREA_FIRST)
    
    assert len(groups) == 2  # Two areas
    assert len(groups[0]) == 2  # Area 01 has 2 models
    assert len(groups[1]) == 1  # Area 02 has 1 model


@pytest.mark.asyncio
async def test_group_tasks_model_first():
    """Test MODEL_FIRST strategy."""
    async def dummy():
        pass
    
    tasks = [
        ExecutionTask("t1", "train", dummy, (), {'area': 'area_01', 'model_type': 'lgbm'}),
        ExecutionTask("t2", "train", dummy, (), {'area': 'area_02', 'model_type': 'lgbm'}),
        ExecutionTask("t3", "train", dummy, (), {'area': 'area_01', 'model_type': 'rf'}),
    ]
    
    executor = ParallelExecutor()
    groups = executor._group_tasks_by_strategy(tasks, ParallelStrategy.MODEL_FIRST)
    
    assert len(groups) == 2  # Two models
    assert len(groups[0]) == 2  # LGBM for 2 areas
```

#### 5. Single Task Execution Tests
```python
@pytest.mark.asyncio
async def test_execute_task_success():
    """Test successful task execution."""
    async def successful_task():
        await asyncio.sleep(0.1)
        return "success"
    
    executor = ParallelExecutor()
    tracker = ProgressTracker(1)
    
    task = ExecutionTask("t1", "test", successful_task, (), {})
    result = await executor._execute_task_with_monitoring(task, tracker)
    
    assert result.status == "success"
    assert result.result == "success"
    assert result.retries_attempted == 0


@pytest.mark.asyncio
async def test_execute_task_timeout():
    """Test task timeout handling."""
    async def slow_task():
        await asyncio.sleep(10)
        return "never"
    
    executor = ParallelExecutor()
    tracker = ProgressTracker(1)
    
    task = ExecutionTask("t1", "test", slow_task, (), {}, timeout_seconds=0.1)
    result = await executor._execute_task_with_monitoring(task, tracker)
    
    assert result.status == "failed"
    assert "timeout" in result.error.lower()


@pytest.mark.asyncio
async def test_execute_task_retry():
    """Test task retry with eventual success."""
    call_count = 0
    
    async def flaky_task():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary failure")
        return "success"
    
    executor = ParallelExecutor()
    tracker = ProgressTracker(1)
    
    task = ExecutionTask("t1", "test", flaky_task, (), {}, max_retries=3)
    result = await executor._execute_task_with_monitoring(task, tracker)
    
    assert result.status == "success"
    assert result.retries_attempted == 2
```

### Integration Tests

#### 1. End-to-End Parallel Execution
```python
@pytest.mark.asyncio
async def test_parallel_execution_e2e():
    """Test end-to-end parallel execution."""
    async def sample_task(task_id: str):
        await asyncio.sleep(0.1)
        return f"result_{task_id}"
    
    executor = ParallelExecutor(max_concurrent_tasks=4)
    
    tasks = [
        ExecutionTask(f"t{i}", "test", sample_task, (), {'task_id': f"t{i}"})
        for i in range(20)
    ]
    
    start_time = asyncio.get_event_loop().time()
    results = await executor.execute_parallel_tasks(tasks, strategy=ParallelStrategy.HYBRID)
    end_time = asyncio.get_event_loop().time()
    
    assert len(results) == 20
    assert all(r.status == "success" for r in results)
    
    # With 4 concurrent tasks, should take ~0.5s (20 tasks / 4 = 5 batches × 0.1s)
    # Sequential would take ~2s (20 × 0.1s)
    elapsed = end_time - start_time
    assert elapsed < 1.0  # Significant speedup
```

#### 2. Integration with Epic-08A
```python
@pytest.mark.asyncio
async def test_training_workflow_integration():
    """Test integration with TrainingWorkflow."""
    from src.workflows.training_workflow import TrainingWorkflow
    
    config = SystemConfig(...)  # Load test config
    workflow = TrainingWorkflow(config)
    
    areas = ['area_01', 'area_02']
    models = ['lgbm', 'rf']
    
    results = await workflow.execute_training_parallel(
        areas=areas,
        models=models,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    assert len(results) == 4  # 2 areas × 2 models
    assert sum(1 for r in results if r.status == "success") >= 3  # At least 75% success
```

### Performance Tests

#### 1. Speedup Measurement
```python
@pytest.mark.performance
@pytest.mark.asyncio
async def test_parallel_speedup():
    """Measure parallel speedup vs sequential."""
    async def cpu_intensive_task():
        # Simulate CPU-intensive work
        result = sum(i**2 for i in range(1000000))
        return result
    
    # Sequential execution
    start_seq = time.time()
    for i in range(10):
        await cpu_intensive_task()
    sequential_time = time.time() - start_seq
    
    # Parallel execution
    executor = ParallelExecutor(max_concurrent_tasks=4)
    tasks = [
        ExecutionTask(f"t{i}", "test", cpu_intensive_task, (), {})
        for i in range(10)
    ]
    
    start_par = time.time()
    results = await executor.execute_parallel_tasks(tasks)
    parallel_time = time.time() - start_par
    
    speedup = sequential_time / parallel_time
    assert speedup > 1.6  # At least 60% reduction (1 / (1 - 0.6) = 2.5 ideal, 1.6 conservative)
```

#### 2. Resource Utilization
```python
@pytest.mark.performance
@pytest.mark.asyncio
async def test_resource_utilization():
    """Test resource utilization during parallel execution."""
    async def task():
        await asyncio.sleep(0.5)
        return "done"
    
    executor = ParallelExecutor(max_concurrent_tasks=4)
    tasks = [ExecutionTask(f"t{i}", "test", task, (), {}) for i in range(20)]
    
    # Monitor resources during execution
    max_cpu = 0
    max_memory = 0
    
    async def monitor():
        nonlocal max_cpu, max_memory
        for _ in range(10):
            usage = executor.resource_monitor.get_current_usage()
            max_cpu = max(max_cpu, usage['cpu_percent'])
            max_memory = max(max_memory, usage['memory_percent'])
            await asyncio.sleep(0.2)
    
    await asyncio.gather(
        executor.execute_parallel_tasks(tasks),
        monitor()
    )
    
    # Check utilization is in optimal range
    assert 70 <= max_cpu <= 90
    assert max_memory < 85
```

### Stress Tests

#### 1. High Task Count
```python
@pytest.mark.stress
@pytest.mark.asyncio
async def test_high_task_count():
    """Test with 100+ tasks."""
    async def task(i):
        await asyncio.sleep(0.01)
        return i
    
    executor = ParallelExecutor(max_concurrent_tasks=8)
    tasks = [
        ExecutionTask(f"t{i}", "test", task, (), {'i': i})
        for i in range(200)
    ]
    
    results = await executor.execute_parallel_tasks(tasks)
    
    assert len(results) == 200
    assert sum(1 for r in results if r.status == "success") > 190  # >95% success
```

#### 2. Failure Rate
```python
@pytest.mark.stress
@pytest.mark.asyncio
async def test_high_failure_rate():
    """Test with 50% task failure rate."""
    import random
    
    async def flaky_task(i):
        if random.random() < 0.5:
            raise ValueError("Random failure")
        return i
    
    executor = ParallelExecutor(max_concurrent_tasks=4)
    tasks = [
        ExecutionTask(f"t{i}", "test", flaky_task, (), {'i': i}, max_retries=1)
        for i in range(50)
    ]
    
    results = await executor.execute_parallel_tasks(tasks)
    
    # Executor should handle failures gracefully
    assert len(results) == 50
    assert all(r.status in ['success', 'failed'] for r in results)
```

---

## ✅ Success Metrics

### Functional Metrics
- [ ] >60% reduction in workflow execution time (parallel vs sequential)
- [ ] >95% task success rate under normal conditions
- [ ] All three parallelization strategies work correctly
- [ ] Dynamic concurrency adjusts within 30 seconds
- [ ] Optimal concurrency calculation accurate (±1 task)

### Performance Metrics
- [ ] CPU utilization: 70-85% during parallel execution
- [ ] Memory utilization: <80% peak
- [ ] Resource monitoring overhead: <5% of total time
- [ ] Task retry success rate: >80% after first retry

### Quality Metrics
- [ ] 80%+ unit test coverage
- [ ] All integration tests pass
- [ ] Performance tests demonstrate speedup
- [ ] Stress tests pass without crashes
- [ ] No memory leaks in long-running tests

---

## 📚 Dependencies

### Internal Dependencies
- **Epic-08A Components:**
  - SystemConfig (for configuration)
  - TrainingWorkflow (for integration)
  - PredictionWorkflow (for integration)
  - BacktestingWorkflow (for integration)

### External Dependencies
```python
# Standard library
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
from datetime import datetime

# Third-party
import psutil  # System resource monitoring

# Version requirements
# psutil >= 5.9.0
```

### System Requirements
- Python 3.9+
- Multi-core CPU (recommended 4+ cores)
- 8GB+ RAM (16GB recommended)
- Linux/macOS (Windows supported with limitations)

---

## 🔄 Related Tickets

### Depends On
- **PC-067-08A:** ConfigManager (configuration foundation)
- **PC-068-08A:** TrainingWorkflow (integration point)
- **PC-069-08A:** PredictionWorkflow (integration point)
- **PC-070-08A:** BacktestingWorkflow (integration point)

### Blocks
- **PC-072-08B:** StructuredLogger (will use progress tracking and resource monitoring)
- **Epic-09:** CLI (will use parallel execution for all commands)

### Related
- **PC-060-07A:** MetricsCalculator (used in parallel evaluation tasks)
- **PC-064-07B:** DriftDetector (used in parallel monitoring)

---

## 📖 Documentation Requirements

### Code Documentation
- [ ] Docstrings for all public classes and methods (Google style)
- [ ] Type hints for all function signatures
- [ ] Inline comments for complex logic (especially retry and concurrency adjustment)
- [ ] Usage examples in docstrings

### User Documentation
- [ ] **Parallelization Guide:**
  - When to use AREA_FIRST vs MODEL_FIRST vs HYBRID
  - How to choose optimal concurrency
  - Resource tuning recommendations
  
- [ ] **Configuration Guide:**
  - max_concurrent_tasks tuning
  - resource_thresholds configuration
  - adaptive_concurrency pros/cons
  
- [ ] **Integration Guide:**
  - How to wrap workflows with ParallelExecutor
  - How to create ExecutionTask objects
  - How to handle TaskResult objects
  
- [ ] **Performance Tuning Guide:**
  - Benchmarking methodology
  - Bottleneck identification
  - System-specific recommendations

### API Documentation
- [ ] ParallelExecutor API reference
- [ ] ExecutionTask and TaskResult specifications
- [ ] ResourceMonitor interface
- [ ] ProgressTracker interface

---

## 🎯 Definition of Done

### Code Complete
- [ ] All classes and methods implemented
- [ ] Type hints for all functions
- [ ] Docstrings for all public APIs
- [ ] No TODOs or FIXMEs in code

### Testing Complete
- [ ] 80%+ unit test coverage
- [ ] All integration tests pass
- [ ] Performance tests demonstrate >60% speedup
- [ ] Stress tests pass without failures

### Documentation Complete
- [ ] Code documentation complete
- [ ] User guides written
- [ ] API reference generated
- [ ] Performance tuning guide published

### Integration Complete
- [ ] Epic-08A workflows integrated
- [ ] Configuration properly wired
- [ ] End-to-end workflows tested
- [ ] Ready for PC-072-08B (StructuredLogger)

### Quality Gates
- [ ] Code review completed and approved
- [ ] No critical or high-severity bugs
- [ ] Performance benchmarks met
- [ ] Linters pass (flake8, mypy, black)

### Deployment Ready
- [ ] Works on Linux, macOS, Windows
- [ ] Dependencies documented
- [ ] Configuration examples provided
- [ ] Ready for production use

---

## 📝 Implementation Notes

### Design Decisions

1. **AsyncIO over Threading:**
   - AsyncIO chosen for I/O-bound tasks (data loading, API calls)
   - Better resource efficiency and control
   - Native support for timeouts and cancellation
   - Consider multiprocessing for CPU-bound tasks in future

2. **Semaphore for Concurrency Control:**
   - Simple and effective
   - Prevents resource exhaustion
   - Easy to adjust dynamically
   - Alternative considered: asyncio.Queue (more complex, unnecessary)

3. **Exponential Backoff for Retries:**
   - Standard pattern for distributed systems
   - Reduces thundering herd problem
   - 2^n formula balances retry speed and system load
   - Max 3 retries prevents infinite loops

4. **Three Parallelization Strategies:**
   - AREA_FIRST: Best for training (data locality)
   - MODEL_FIRST: Best for model comparison
   - HYBRID: Best for prediction (maximum parallelism)
   - User can choose based on workload characteristics

### Known Limitations

1. **GIL (Global Interpreter Lock):**
   - AsyncIO helps with I/O, but CPU-bound tasks limited by GIL
   - Future: Consider multiprocessing.Pool for CPU-intensive models
   - Current: Adequate for I/O-bound workflows

2. **Memory Pressure:**
   - Loading 26 models concurrently may exceed memory
   - Mitigation: Dynamic concurrency reduction
   - Recommendation: 2GB RAM per concurrent task

3. **Disk I/O Bottleneck:**
   - Parallel data loading may saturate disk bandwidth
   - Mitigation: SSD recommended, caching (Epic-05A)
   - Consider: NVMe or network storage for production

### Future Enhancements

1. **Adaptive Strategy Selection:**
   - Auto-select strategy based on task characteristics
   - Machine learning-based optimization

2. **Distributed Execution:**
   - Support for multi-machine parallelization
   - Celery or Dask integration

3. **GPU Support:**
   - Parallel GPU task scheduling
   - CUDA stream management

4. **Advanced Fault Tolerance:**
   - Circuit breaker pattern for cascading failures
   - Task dependency resolution (DAG-based)

---

## 🚀 Handoff Checklist

### For PC-072-08B (StructuredLogger)
- [ ] ParallelExecutor provides progress_tracker.get_progress()
- [ ] TaskResult includes resource_usage for logging
- [ ] ResourceMonitor.get_current_usage() available
- [ ] task_id propagates for correlation

### For Epic-09 (CLI)
- [ ] ParallelExecutor integrated in all workflows
- [ ] Configuration exposed via SystemConfig
- [ ] Performance benchmarks documented
- [ ] Usage examples provided

### For Production Deployment
- [ ] Resource requirements documented
- [ ] Tuning guide complete
- [ ] Monitoring integration points defined
- [ ] Failure scenarios tested
