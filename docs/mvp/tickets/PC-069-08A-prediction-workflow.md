# PC-069-08A: Prediction Workflows (Batch & Intraday)

**Ticket ID:** PC-069-08A  
**Epic:** Epic-08A - Core Workflows & Configuration  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Sprint:** Week 22 (Days 2-3)

---

## 📋 Description

Implement flexible prediction workflows for both scheduled batch predictions (D+0 to D+8 for all areas) and real-time intraday updates (LGBM BLF for D+0). Includes integration with Epic-05 combination strategies, Epic-06 hierarchical reconciliation, prediction caching for incremental updates, and standard output formatting for downstream systems.

---

## 🎯 Acceptance Criteria

- [ ] Batch prediction workflow generates forecasts for D+0 to D+8
- [ ] Intraday prediction workflow updates D+0 using LGBM BLF strategy
- [ ] Integration with Epic-05A/05B combination methods
- [ ] Integration with Epic-06A/06B hierarchical reconciliation
- [ ] Prediction caching (Redis/in-memory) for incremental updates
- [ ] Standard output formats (JSON/Parquet) for downstream systems
- [ ] Batch predictions complete within 15 minutes
- [ ] Intraday predictions complete within 5 minutes
- [ ] Caching reduces redundant computation by >30%
- [ ] Unit tests with 80%+ coverage

---

## 📐 Technical Specifications

### Core Components

#### 1. Data Structures

```python
@dataclass
class PredictionTask:
    prediction_date: datetime
    horizons: List[int]  # D+0 to D+8
    areas: List[str]
    task_id: Optional[str] = None

@dataclass
class PredictionResult:
    task_id: str
    prediction_date: datetime
    predictions: Dict[str, np.ndarray]  # area -> predictions
    combined_predictions: Dict[str, np.ndarray]
    reconciled_predictions: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    duration_seconds: float
    status: WorkflowStatus
    error: Optional[str]

@dataclass
class IntradayData:
    area: str
    timestamp: datetime
    observations: np.ndarray
    features: Dict[str, Any]
```

#### 2. PredictionWorkflow

```python
class PredictionWorkflow:
    def __init__(self, config: SystemConfig)
    
    async def execute_batch_prediction(
        self,
        prediction_date: datetime,
        horizons: List[int] = list(range(9)),  # D+0 to D+8
        areas: Optional[List[str]] = None
    ) -> PredictionResult
    
    async def execute_intraday_prediction(
        self,
        current_datetime: datetime,
        updated_data: IntradayData
    ) -> PredictionResult
    
    async def _load_models(self, areas: Optional[List[str]]) -> Dict[str, Any]
    
    async def _execute_base_predictions(
        self,
        models: Dict[str, Any],
        features: Dict[str, np.ndarray],
        prediction_date: datetime,
        horizons: List[int]
    ) -> Dict[str, Dict[str, np.ndarray]]
```

#### 3. PredictionCache

```python
class PredictionCache:
    def __init__(self, backend: str = "memory")  # or "redis"
    
    def store(
        self,
        key: str,
        predictions: Dict[str, np.ndarray],
        ttl_seconds: int = 3600
    )
    
    def retrieve(self, key: str) -> Optional[Dict[str, np.ndarray]]
    
    def retrieve_future_horizons(
        self,
        prediction_date: date,
        area: str,
        horizons: List[int]
    ) -> np.ndarray
    
    def update_intraday(
        self,
        area: str,
        datetime: datetime,
        prediction: np.ndarray
    )
    
    def invalidate(self, pattern: str)
```

### Batch Prediction Workflow

```
1. Load latest trained models from registry
   ↓
2. Generate features for prediction date and all horizons
   ↓
3. Execute base model predictions in parallel
   - LGBM predictions
   - Random Forest predictions
   - RegDin+SVM predictions
   - Holt-Winters predictions
   ↓
4. Apply model combination strategies (Epic-05A/05B)
   - Weighted average
   - Stacking
   - Bias correction
   ↓
5. Perform hierarchical reconciliation (Epic-06A/06B)
   - MinT reconciliation
   - Maintain coherence across hierarchy
   ↓
6. Cache results (key: batch_YYYYMMDD)
   ↓
7. Format output (JSON/Parquet)
   ↓
8. Return PredictionResult
```

### Intraday Prediction Workflow

```
1. Update features with latest intraday observations
   ↓
2. Load LGBM model (BLF strategy)
   ↓
3. Predict D+0 only with updated features
   ↓
4. Retrieve cached D+1 to D+8 from batch prediction
   ↓
5. Combine D+0 (updated) with D+1-D+8 (cached)
   ↓
6. Optional: Partial reconciliation for affected area
   ↓
7. Update cache with new D+0 prediction
   ↓
8. Return PredictionResult
```

---

## 🔧 Implementation Tasks

### Task 1: Core Data Structures (2 hours)
- [ ] Implement `PredictionTask` dataclass
- [ ] Implement `PredictionResult` dataclass
- [ ] Implement `IntradayData` dataclass
- [ ] Add validation and type hints

### Task 2: Model Loading (3 hours)
- [ ] Implement `_load_models()` method
- [ ] Integrate with model registry
- [ ] Load latest versions by model type and area
- [ ] Handle missing models gracefully
- [ ] Cache loaded models

### Task 3: Feature Generation Integration (3 hours)
- [ ] Integrate with feature generator (Epic-02)
- [ ] Generate features for all horizons (D+0-D+8)
- [ ] Support intraday feature updates
- [ ] Handle missing feature values

### Task 4: Base Model Predictions (4 hours)
- [ ] Implement `_execute_base_predictions()`
- [ ] Parallel prediction execution
- [ ] Handle prediction errors per model
- [ ] Collect predictions in structured format
- [ ] Log prediction latencies

### Task 5: Combination Integration (3 hours)
- [ ] Integrate with Epic-05A combiner
- [ ] Support multiple combination strategies
- [ ] Apply weights from configuration
- [ ] Handle bias correction

### Task 6: Reconciliation Integration (3 hours)
- [ ] Integrate with Epic-06A reconciler
- [ ] Apply MinT reconciliation
- [ ] Support partial reconciliation for intraday
- [ ] Validate hierarchical coherence

### Task 7: Batch Prediction Workflow (4 hours)
- [ ] Implement `execute_batch_prediction()`
- [ ] Orchestrate full pipeline
- [ ] Add error handling and retries
- [ ] Format output (JSON/Parquet)
- [ ] Performance optimization (<15 min)

### Task 8: Intraday Prediction Workflow (4 hours)
- [ ] Implement `execute_intraday_prediction()`
- [ ] LGBM BLF integration
- [ ] Cache integration for D+1-D+8 retrieval
- [ ] Partial reconciliation logic
- [ ] Performance optimization (<5 min)

### Task 9: Prediction Cache (4 hours)
- [ ] Implement `PredictionCache` class
- [ ] In-memory cache backend
- [ ] Redis cache backend (optional)
- [ ] TTL management
- [ ] Cache invalidation strategies
- [ ] Cache hit/miss metrics

### Task 10: Output Formatting (2 hours)
- [ ] JSON output formatter
- [ ] Parquet output formatter
- [ ] Include metadata (models, timestamps, versions)
- [ ] Schema validation

### Task 11: Testing (5 hours)
- [ ] Unit tests for each component
- [ ] Integration tests with Epic-05/06
- [ ] Performance benchmarking
- [ ] Cache effectiveness testing
- [ ] End-to-end workflow tests

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_prediction_task_creation()
def test_model_loading()
def test_feature_generation_batch()
def test_feature_generation_intraday()
def test_base_predictions_parallel()
def test_combination_integration()
def test_reconciliation_integration()
def test_cache_store_retrieve()
def test_cache_ttl_expiration()
def test_cache_invalidation()
def test_output_json_format()
def test_output_parquet_format()
```

### Integration Tests
```python
def test_batch_prediction_full_pipeline()
def test_intraday_prediction_full_pipeline()
def test_batch_with_all_combinations()
def test_intraday_cache_retrieval()
def test_partial_reconciliation()
def test_output_compatibility()
```

### Performance Tests
```python
def test_batch_prediction_duration()
def test_intraday_prediction_duration()
def test_cache_performance_improvement()
def test_parallel_prediction_speedup()
```

---

## 📊 Success Metrics

### Performance Targets
- Batch predictions (26 areas, D+0-D+8): **<15 minutes**
- Intraday predictions (single area, D+0): **<5 minutes**
- Cache hit rate: **>80%** for intraday
- Cache performance improvement: **>30%** reduction

### Quality Targets
- Unit test coverage: **≥80%**
- Prediction accuracy: Validated via Epic-07A metrics
- Cache consistency: **100%** (no stale data)
- Output schema compliance: **100%**

---

## 📚 Dependencies

### Required Packages
```python
asyncio (stdlib)
numpy >= 1.24.0
pandas >= 2.0.0
redis >= 4.5.0              # Optional, for Redis cache
pyarrow >= 12.0.0           # For Parquet output
```

### Code Dependencies
- **Requires:** PC-067-08A (ConfigManager)
- **Requires:** PC-068-08A (TrainingWorkflow for model registry)
- **Requires:** Epic-05A/05B (Model combination)
- **Requires:** Epic-06A/06B (Hierarchical reconciliation)
- **Requires:** Epic-02 (Feature engineering)

---

## 🔗 Related Tickets
- **Depends on:** PC-067-08A (Configuration Manager)
- **Depends on:** PC-068-08A (Training Workflow)
- **Integrates with:** Epic-05A/05B (Combination)
- **Integrates with:** Epic-06A/06B (Reconciliation)
- **Blocks:** PC-070-08A (Backtesting Workflow)

---

## 📖 Documentation Requirements

- [ ] Batch prediction workflow diagram
- [ ] Intraday prediction workflow diagram
- [ ] Cache architecture and strategies
- [ ] Output format specifications
- [ ] Integration guide for Epic-05/06
- [ ] Performance tuning guide
- [ ] Troubleshooting common issues

---

## 💡 Implementation Notes

### Cache Key Structure
```python
# Batch prediction cache
key = f"batch_{prediction_date.strftime('%Y%m%d')}_{area}"

# Intraday prediction cache
key = f"intraday_{area}_{current_datetime.strftime('%Y%m%d_%H%M')}"
```

### Redis Cache Backend
```python
import redis

class RedisPredictionCache(PredictionCache):
    def __init__(self, host='localhost', port=6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=False)
    
    def store(self, key, predictions, ttl_seconds=3600):
        serialized = pickle.dumps(predictions)
        self.client.setex(key, ttl_seconds, serialized)
    
    def retrieve(self, key):
        data = self.client.get(key)
        return pickle.loads(data) if data else None
```

### Intraday Cache Retrieval
```python
def retrieve_future_horizons(self, prediction_date, area, horizons):
    # Retrieve D+1 to D+8 from most recent batch prediction
    batch_key = f"batch_{prediction_date.strftime('%Y%m%d')}_{area}"
    cached = self.retrieve(batch_key)
    
    if cached and area in cached:
        # Extract D+1 to D+8 (skip D+0)
        return cached[area][1:]  # Horizons 1-8
    
    return None
```

### Output Format Example (JSON)
```json
{
  "prediction_date": "2025-11-19",
  "generation_time": "2025-11-19T10:30:00",
  "models_used": ["lgbm", "random_forest", "regdin_svm", "holt_winters"],
  "combination_strategy": "weighted_average",
  "reconciliation_method": "mint",
  "predictions": {
    "area_01": {
      "D+0": [100.5, 95.2, ...],  # 24 hours
      "D+1": [98.3, 93.1, ...],
      ...
      "D+8": [102.1, 97.8, ...]
    },
    "area_02": { ... }
  },
  "metadata": {
    "version": "1.0.0",
    "execution_time_seconds": 450.2
  }
}
```

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥80% coverage
- [ ] Batch prediction workflow complete
- [ ] Intraday prediction workflow complete
- [ ] Integration with Epic-05/06 successful
- [ ] Caching implemented and tested
- [ ] Output formatting working (JSON/Parquet)
- [ ] Performance benchmarks met (<15 min batch, <5 min intraday)
- [ ] Cache improvement >30%
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] Ready for PC-070-08A (Backtesting Workflow)
