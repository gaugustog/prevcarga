# PC-070-08A: Backtesting Workflow with Retraining Optimization

**Ticket ID:** PC-070-08A  
**Epic:** Epic-08A - Core Workflows & Configuration  
**Story Points:** 8  
**Priority:** Medium  
**Assignee:** TBD  
**Sprint:** Week 22 (Days 4-5)

---

## 📋 Description

Implement comprehensive backtesting framework with walk-forward validation to evaluate historical model performance and optimize retraining intervals. Tests multiple retraining strategies (7, 14, 30 days) with expanding/sliding window approaches, integrates with Epic-07A metrics and Epic-07B drift detection, and generates automated recommendations for optimal operational configuration.

---

## 🎯 Acceptance Criteria

- [ ] Walk-forward validation with expanding and sliding window strategies
- [ ] Multiple retraining intervals evaluation: 7, 14, 30 days
- [ ] Integration with Epic-07A MetricsCalculator for performance tracking
- [ ] Integration with Epic-07B DriftDetector for drift analysis
- [ ] Statistical significance testing between retraining intervals
- [ ] Automated report generation with optimal interval recommendation
- [ ] Complete 1-year backtest in <8 hours
- [ ] Actionable recommendations based on statistical analysis
- [ ] Unit tests with 75%+ coverage

---

## 📐 Technical Specifications

### Core Components

#### 1. Configuration and Results

```python
@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    retraining_intervals: List[int]  # days (e.g., [7, 14, 30])
    validation_strategy: str  # 'expanding' or 'sliding'
    initial_training_days: int = 730  # 2 years initial training
    test_period_days: int = 7  # Test period after each training

@dataclass
class BacktestResult:
    config: BacktestConfig
    interval_results: Dict[int, Dict[str, Any]]  # interval -> metrics
    optimal_interval: int
    recommendations: str
    statistical_tests: Dict[str, Any]
    duration_seconds: float
```

#### 2. BacktestingWorkflow

```python
class BacktestingWorkflow:
    def __init__(self, config: SystemConfig)
    
    async def execute_backtest(
        self,
        backtest_config: BacktestConfig,
        areas: Optional[List[str]] = None
    ) -> BacktestResult
    
    async def _backtest_with_interval(
        self,
        start_date: datetime,
        end_date: datetime,
        retraining_interval_days: int,
        validation_strategy: str,
        initial_training_days: int,
        areas: Optional[List[str]]
    ) -> Dict[str, Any]
    
    def _define_backtest_periods(
        self,
        start_date: datetime,
        end_date: datetime,
        retraining_interval_days: int,
        initial_training_days: int,
        strategy: str
    ) -> List[Dict[str, datetime]]
    
    def _determine_optimal_interval(
        self,
        interval_results: Dict[int, Dict[str, Any]]
    ) -> int
    
    def _perform_statistical_comparison(
        self,
        interval_results: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]
    
    def _generate_recommendations(
        self,
        interval_results: Dict[int, Dict[str, Any]],
        optimal_interval: int,
        statistical_tests: Dict[str, Any]
    ) -> str
```

### Walk-Forward Validation Strategies

#### Expanding Window
```
Initial Training: [----T1----]
Period 1:        [----T1----][P1]
Period 2:        [------T2------][P2]
Period 3:        [---------T3---------][P3]

Training window expands with each retraining cycle
```

#### Sliding Window
```
Period 1: [----T1----][P1]
Period 2:      [----T2----][P2]
Period 3:           [----T3----][P3]

Training window size stays constant, slides forward
```

### Retraining Period Definition

```python
def _define_backtest_periods(
    self, start_date, end_date, 
    retraining_interval_days, initial_training_days, strategy
):
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
        
        # Update for next period
        if strategy == 'expanding':
            current_train_end = current_test_end
        else:  # sliding
            current_train_start += timedelta(days=retraining_interval_days)
            current_train_end = current_test_end
        
        current_test_start = current_test_end
    
    return periods
```

### Optimal Interval Selection

```python
def _determine_optimal_interval(self, interval_results):
    # Scoring: MAPE + penalty for retraining frequency
    best_interval = None
    best_score = float('inf')
    
    for interval, result in interval_results.items():
        mape = result['overall_metrics']['mape']
        num_cycles = result['num_retraining_cycles']
        drift_count = result['drift_count']
        
        # Score formula
        score = (
            mape +                      # Forecast accuracy
            (0.1 * num_cycles) +        # Retraining cost penalty
            (0.05 * drift_count)        # Drift penalty
        )
        
        if score < best_score:
            best_score = score
            best_interval = interval
    
    return best_interval
```

---

## 🔧 Implementation Tasks

### Task 1: Core Data Structures (2 hours)
- [ ] Implement `BacktestConfig` dataclass
- [ ] Implement `BacktestResult` dataclass
- [ ] Add validation for config parameters
- [ ] Document configuration options

### Task 2: Period Generation Logic (4 hours)
- [ ] Implement `_define_backtest_periods()`
- [ ] Support expanding window strategy
- [ ] Support sliding window strategy
- [ ] Handle edge cases (end date, short periods)
- [ ] Validate period consistency

### Task 3: Single Interval Backtesting (6 hours)
- [ ] Implement `_backtest_with_interval()`
- [ ] Integrate with TrainingWorkflow for retraining
- [ ] Integrate with PredictionWorkflow for forecasts
- [ ] Load actuals for evaluation periods
- [ ] Calculate metrics for each test period
- [ ] Aggregate overall metrics

### Task 4: Drift Detection Integration (3 hours)
- [ ] Integrate Epic-07B DriftDetector
- [ ] Detect drift in each test period
- [ ] Track drift events and types
- [ ] Analyze drift patterns by interval
- [ ] Include drift metrics in results

### Task 5: Metrics Evaluation (3 hours)
- [ ] Integrate Epic-07A MetricsCalculator
- [ ] Calculate metrics for each test period
- [ ] Aggregate metrics across all periods
- [ ] Track metric evolution over time
- [ ] Identify performance degradation patterns

### Task 6: Statistical Comparison (4 hours)
- [ ] Implement `_perform_statistical_comparison()`
- [ ] Paired t-tests between intervals
- [ ] Calculate effect sizes (Cohen's d)
- [ ] Bonferroni correction for multiple comparisons
- [ ] Statistical significance interpretation

### Task 7: Optimal Interval Selection (3 hours)
- [ ] Implement `_determine_optimal_interval()`
- [ ] Design scoring function (accuracy + cost)
- [ ] Balance performance vs. retraining overhead
- [ ] Consider drift frequency in scoring
- [ ] Validate interval selection logic

### Task 8: Recommendation Generation (3 hours)
- [ ] Implement `_generate_recommendations()`
- [ ] Generate human-readable summary
- [ ] Include performance comparison
- [ ] Provide rationale for recommendation
- [ ] Add implementation guidance

### Task 9: Main Backtest Execution (4 hours)
- [ ] Implement `execute_backtest()` orchestrator
- [ ] Execute backtests for all intervals
- [ ] Collect and aggregate results
- [ ] Generate final report
- [ ] Handle execution errors

### Task 10: Testing and Optimization (4 hours)
- [ ] Unit tests for period generation
- [ ] Integration tests with workflows
- [ ] Performance optimization for 1-year backtest
- [ ] Memory management for large datasets
- [ ] Validate statistical methods

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_backtest_config_validation()
def test_period_generation_expanding()
def test_period_generation_sliding()
def test_period_count_calculation()
def test_optimal_interval_selection()
def test_statistical_comparison()
def test_recommendation_generation()
```

### Integration Tests
```python
def test_backtest_with_training_workflow()
def test_backtest_with_prediction_workflow()
def test_backtest_with_metrics_calculator()
def test_backtest_with_drift_detector()
def test_full_backtest_3_months()
def test_full_backtest_1_year()
```

### Validation Tests
```python
def test_expanding_window_correctness()
def test_sliding_window_correctness()
def test_statistical_test_validity()
def test_drift_detection_accuracy()
```

---

## 📊 Success Metrics

### Performance Targets
- 1-year backtest (26 areas, 3 intervals): **<8 hours**
- 3-month backtest: **<2 hours**
- Memory usage: **<4 GB peak**

### Quality Targets
- Unit test coverage: **≥75%**
- Statistical test validity: **α ≤ 0.05** (properly controlled)
- Recommendation accuracy: **>80%** (validated manually)

### Validation Targets
- Interval selection: Consistent with manual analysis
- Drift detection: Matches known degradation events
- Statistical tests: Proper Type I error control

---

## 📚 Dependencies

### Required Packages
```python
asyncio (stdlib)
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0             # Statistical tests
```

### Code Dependencies
- **Requires:** PC-067-08A (ConfigManager)
- **Requires:** PC-068-08A (TrainingWorkflow)
- **Requires:** PC-069-08A (PredictionWorkflow)
- **Requires:** PC-060-07A (MetricsCalculator)
- **Requires:** PC-064-07B (DriftDetector)

---

## 🔗 Related Tickets
- **Depends on:** PC-067-08A (Configuration Manager)
- **Depends on:** PC-068-08A (Training Workflow)
- **Depends on:** PC-069-08A (Prediction Workflow)
- **Integrates with:** Epic-07A (Metrics)
- **Integrates with:** Epic-07B (Drift Detection)
- **Completes:** Epic-08A

---

## 📖 Documentation Requirements

- [ ] Backtesting methodology documentation
- [ ] Walk-forward validation explanation
- [ ] Retraining interval selection guide
- [ ] Statistical test interpretation
- [ ] Recommendation implementation guide
- [ ] Performance optimization tips
- [ ] Troubleshooting common issues

---

## 💡 Implementation Notes

### Expanding vs. Sliding Window

**Expanding Window:**
- Advantages: More training data over time, stable baseline
- Disadvantages: Computational cost increases, older data may be less relevant
- Use when: Long-term stability important, concept drift minimal

**Sliding Window:**
- Advantages: Fixed computational cost, adapts to recent patterns
- Disadvantages: Less historical context, potential instability
- Use when: Concept drift significant, recent patterns more relevant

### Scoring Function Design
```python
def calculate_interval_score(mape, num_cycles, drift_count):
    # Base score: forecast accuracy (MAPE)
    score = mape
    
    # Penalty for frequent retraining (operational cost)
    # 10% penalty per additional cycle
    score += 0.1 * num_cycles
    
    # Penalty for drift events (model instability)
    # 5% penalty per drift event
    score += 0.05 * drift_count
    
    return score
```

### Recommendation Template
```python
recommendation = f"""
Backtesting Recommendations
============================

Optimal Retraining Interval: {optimal_interval} days

Performance Summary:
- MAPE: {result['overall_metrics']['mape']:.2f}%
- MAE: {result['overall_metrics']['mae']:.2f}
- RMSE: {result['overall_metrics']['rmse']:.2f}

Operational Metrics:
- Retraining Cycles: {result['num_retraining_cycles']}
- Drift Events: {result['drift_count']}
- Total Duration: {result['total_duration_hours']:.1f} hours

Statistical Comparison:
{comparison_summary}

Rationale:
The {optimal_interval}-day interval provides the best balance between:
1. Forecast accuracy (lowest MAPE)
2. Operational overhead (reasonable retraining frequency)
3. Model stability (minimal drift events)

Implementation Steps:
1. Configure production retraining schedule to {optimal_interval} days
2. Monitor drift detection alerts for early warning
3. Re-evaluate interval quarterly or when drift patterns change
4. Consider sliding window if concept drift increases

Next Review: {next_review_date}
"""
```

### Statistical Comparison Output
```python
{
    "7_vs_14": {
        "t_statistic": -2.45,
        "p_value": 0.018,
        "significant": True,
        "effect_size": 0.32,
        "interpretation": "14-day interval significantly better (p=0.018, d=0.32)"
    },
    "14_vs_30": {
        "t_statistic": -1.12,
        "p_value": 0.267,
        "significant": False,
        "effect_size": 0.15,
        "interpretation": "No significant difference (p=0.267)"
    }
}
```

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥75% coverage
- [ ] Both window strategies implemented
- [ ] All retraining intervals tested (7, 14, 30 days)
- [ ] Integration with Epic-07A/07B successful
- [ ] Statistical comparison working correctly
- [ ] Optimal interval selection validated
- [ ] Recommendations generated automatically
- [ ] 1-year backtest completes in <8 hours
- [ ] Code reviewed and approved
- [ ] Documentation complete with examples
- [ ] Epic-08A complete and ready for Epic-08B
