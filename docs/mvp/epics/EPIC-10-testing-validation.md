# EPIC-10: Testing & Validation

**Duration:** 3 weeks
**Dependencies:** EPIC-09
**Reference:** [MVP Plan R - Phase 10](../mvp-plan-r.md#phase-10-testing--validation-3-weeks)

---

## Objective

Validate the system against existing R model results, execute comprehensive backtests, compare model performance, and ensure test coverage meets quality standards.

---

## Scope

This epic covers:
- Reproduction of existing R model results
- Complete 2024 backtest (1 year)
- Model vs legacy comparison
- Combination evaluation
- Reconciliation validation
- Performance benchmarks
- Test coverage verification

**Out of Scope:**
- Plugin implementations (tested via integration)
- Documentation (EPIC-11)
- Deployment (EPIC-11)

---

## Tasks

### T-10.1: Create Baseline Validation Framework
- [ ] Create `tests/validation/baseline.R`
- [ ] Implement framework to compare against legacy results:
  ```r
  validate_against_baseline = function(model_results, baseline_path,
                                        tolerance = 0.05) {
    # Load baseline results
    baseline <- arrow::read_parquet(baseline_path)

    # Compare predictions
    merged <- merge(model_results, baseline, by = c("DataHora", "area_code"))

    # Calculate difference metrics
    mape_diff <- abs(model_results$mape - baseline$mape)

    list(
      passed = all(mape_diff < tolerance * baseline$mape),
      max_diff = max(mape_diff),
      mean_diff = mean(mape_diff),
      details = merged
    )
  }
  ```

### T-10.2: Reproduce Existing Model Results
- [ ] Run each contributed model plugin against historical data
- [ ] Compare MAPE within ±5% of baseline
- [ ] Document any discrepancies
- [ ] Create validation report

### T-10.3: Execute 1-Year Backtest
- [ ] Configure backtest for 2024 (Jan 1 - Dec 31)
- [ ] Run with all registered models
- [ ] Test retraining intervals: [1, 3, 7, 10, 15] days
- [ ] Generate metrics by:
  - Model
  - Area
  - Horizon (D+0 to D+8)
  - Patamar (peak/off-peak)

### T-10.4: Validate Combination Strategies
- [ ] Test each registered combiner
- [ ] Verify combined predictions outperform individuals
- [ ] Test weight optimization convergence
- [ ] Validate bias correction effectiveness

### T-10.5: Validate Reconciliation
- [ ] Verify hierarchical consistency:
  - SIN = Σ(Subsystems)
  - Subsystem = Σ(Areas) + Loss
- [ ] Test each registered reconciler
- [ ] Validate loss calculations

### T-10.6: Performance Benchmarking
- [ ] Measure and verify performance targets:
  ```r
  performance_targets <- list(
    training_single_area_model = 30 * 60,  # <30 minutes
    intraday_prediction = 5 * 60,           # <5 minutes
    backtest_1_year = 8 * 60 * 60           # <8 hours
  )
  ```
- [ ] Profile critical code paths
- [ ] Identify optimization opportunities

### T-10.7: Verify Test Coverage
- [ ] Run coverage analysis:
  ```r
  covr::package_coverage()
  ```
- [ ] Ensure ≥70% overall coverage
- [ ] Ensure ≥80% coverage for critical components:
  - Storage backends
  - Model infrastructure
  - Combination infrastructure
  - Reconciliation infrastructure

### T-10.8: Integration Test Suite
- [ ] Create `tests/integration/`
- [ ] Test complete workflows:
  - Train → Predict → Evaluate
  - Train → Combine → Reconcile → Evaluate
  - Backtest with retraining

### T-10.9: Edge Case Testing
- [ ] Test with missing data
- [ ] Test with extreme values
- [ ] Test with DST transitions
- [ ] Test with holidays
- [ ] Test failure recovery

### T-10.10: Generate Validation Report
- [ ] Create comprehensive validation report including:
  - Model accuracy comparison
  - Performance benchmarks
  - Test coverage summary
  - Recommendations

---

## Acceptance Criteria

- [ ] All registered models reproduce baseline within ±5% MAPE
- [ ] 1-year backtest completes in <8 hours
- [ ] All models validated across 21+ areas
- [ ] Model combinations outperform individuals (on average)
- [ ] Reconciliation maintains hierarchical consistency (100%)
- [ ] Test coverage ≥70% overall
- [ ] All performance benchmarks met
- [ ] Validation report generated and reviewed

---

## Definition of Done

- [ ] All tasks completed
- [ ] Validation report approved
- [ ] Test coverage verified
- [ ] Performance benchmarks documented
- [ ] Known issues documented

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `tests/validation/baseline.R` | Create | Baseline validation framework |
| `tests/validation/backtest_2024.R` | Create | 2024 backtest script |
| `tests/validation/reconciliation.R` | Create | Reconciliation validation |
| `tests/integration/train_predict.R` | Create | Integration test |
| `tests/integration/full_workflow.R` | Create | Full workflow test |
| `reports/validation_report.Rmd` | Create | Validation report template |

---

## Validation Targets

| Metric | Target | Description |
|--------|--------|-------------|
| MAPE Diff | <5% | Difference from baseline |
| Training Time | <30 min | Single area + single model |
| Intraday Prediction | <5 min | Full prediction cycle |
| Backtest Time | <8 hours | 1-year backtest |
| Test Coverage | ≥70% | Overall package coverage |
| Critical Coverage | ≥80% | Core infrastructure |

---

## Test Pyramid

```
          /\
         /E2E\        10% - End-to-End (CLI commands)
        /------\
       /  INT   \     30% - Integration (workflows)
      /----------\
     /   UNIT     \   60% - Unit (functions, R6 classes)
    /--------------\
```
