# Epic-05A: Base Combination Strategies

**Epic ID:** Epic-05A  
**Epic Name:** Base Combination Strategies  
**Phase:** 5A  
**Duration:** 2 weeks (Weeks 14-15)  
**Dependencies:** Epic-04 (Hierarchical Models)  
**Priority:** High  

---

## 🎯 Epic Overview

Implement foundational model combination approaches that merge predictions from multiple forecasting models (LGBM, Random Forest, RegDin+SVM, Holt-Winters) using fixed and learned weighting strategies. This epic establishes the core combination infrastructure that enables ensemble forecasting while maintaining interpretability and performance.

### Business Value
- **Improved Accuracy:** Combination typically outperforms individual models by 5-15%
- **Risk Mitigation:** Reduces dependency on single model performance
- **Robustness:** Ensemble approaches handle model failures gracefully
- **Foundation:** Enables advanced ensemble methods (Epic-05B)

---

## 📋 User Stories

### **User Story 1: Model Combination Interface**
**As a** ML engineer  
**I want** a standardized interface for combining model predictions  
**So that** I can easily integrate multiple combination strategies  

**Acceptance Criteria:**
- [ ] `BaseCombiner` interface defines standard combination contract
- [ ] Interface supports both point forecasts and prediction intervals
- [ ] Combination metadata (weights, performance) is tracked
- [ ] Interface handles missing predictions gracefully
- [ ] Extensible design allows new combination methods

**Technical Requirements:**
- Base interface with abstract `combine()` method
- Support for 26 time series (17 areas + 4 subsystems + 4 losses + 1 national)
- Handle horizon-specific combinations (D+0 to D+8)
- Input validation for prediction formats
- Logging and monitoring integration

**Definition of Done:**
- [ ] `BaseCombiner` class implemented with full interface
- [ ] Unit tests cover all interface methods
- [ ] Integration tests with mock predictions work
- [ ] Documentation includes usage examples
- [ ] Code review approved by tech lead

---

### **User Story 2: Simple Averaging Combiner**
**As a** forecasting analyst  
**I want** a simple averaging combination method  
**So that** I can quickly combine models without complex weight optimization  

**Acceptance Criteria:**
- [ ] `SimpleAveragingCombiner` provides equal-weight averaging
- [ ] Supports arithmetic and geometric mean combinations
- [ ] Handles missing predictions with available model subset
- [ ] Performance tracking compared to individual models
- [ ] Configurable model subset selection

**Technical Requirements:**
- Implement arithmetic mean: `y_combined = (1/n) * Σ(y_i)`
- Implement geometric mean: `y_combined = (Π(y_i))^(1/n)`
- Handle zero/negative values in geometric mean
- Missing value imputation strategies
- Performance metrics calculation (MAPE, MAE, RMSE)

**Definition of Done:**
- [ ] `SimpleAveragingCombiner` class implemented
- [ ] Both arithmetic and geometric mean working
- [ ] Missing value handling tested
- [ ] Performance comparison with baseline models
- [ ] Unit and integration tests pass

---

### **User Story 3: Weighted Voting Combiner**
**As a** ML engineer  
**I want** a weighted combination method with optimized weights  
**So that** I can leverage historical model performance for better combinations  

**Acceptance Criteria:**
- [ ] `WeightedVotingCombiner` with learned weights from validation data
- [ ] Weight optimization using multiple objective functions (MAPE, MAE, RMSE)
- [ ] Constraints ensure weights sum to 1 and are non-negative
- [ ] Weight stability analysis and validation
- [ ] Historical performance-based weight calculation

**Technical Requirements:**
- Implement constrained optimization: `min Σ|y_true - Σ(w_i * y_i)|` subject to `Σw_i = 1, w_i ≥ 0`
- Use scipy.optimize for weight optimization
- Multiple optimization objectives (MAPE-based, MAE-based, RMSE-based)
- Cross-validation for weight stability
- Weight persistence and versioning

**Definition of Done:**
- [ ] `WeightedVotingCombiner` class implemented
- [ ] Weight optimization converges reliably
- [ ] Multiple objective functions working
- [ ] Weight stability validation passes
- [ ] Performance improves over simple averaging

---

### **User Story 4: Best Model Selector**
**As a** forecasting analyst  
**I want** a dynamic model selection strategy  
**So that** I can automatically choose the best performing model per time series and horizon  

**Acceptance Criteria:**
- [ ] `BestModelSelector` chooses optimal model per context
- [ ] Selection criteria include recent performance, model confidence, and historical accuracy
- [ ] Fallback mechanisms when best model fails
- [ ] Selection logging for interpretability
- [ ] Performance tracking and model ranking

**Technical Requirements:**
- Implement rolling window performance evaluation
- Multiple selection criteria (MAPE, MAE, prediction confidence)
- Fallback hierarchy when primary model unavailable
- Model performance history tracking
- Decision logging and audit trail

**Definition of Done:**
- [ ] `BestModelSelector` class implemented
- [ ] Selection criteria configurable and tested
- [ ] Fallback mechanisms working reliably
- [ ] Performance tracking and logging functional
- [ ] Selection decisions are interpretable

---

### **User Story 5: Basic Bias Correction**
**As a** ML engineer  
**I want** bias correction for model combinations  
**So that** I can reduce systematic forecast errors and improve accuracy  

**Acceptance Criteria:**
- [ ] `BasicBiasCorrectionModule` identifies and corrects systematic biases
- [ ] Correction methods include additive and multiplicative bias
- [ ] Time-dependent bias correction (seasonal, weekly patterns)
- [ ] Validation prevents overfitting to correction data
- [ ] Integration with all combination methods

**Technical Requirements:**
- Additive bias correction: `y_corrected = y_forecast + bias_offset`
- Multiplicative bias correction: `y_corrected = y_forecast * bias_factor`
- Rolling window bias estimation
- Seasonal and weekly bias pattern detection
- Cross-validation for bias correction validation

**Definition of Done:**
- [ ] `BasicBiasCorrectionModule` class implemented
- [ ] Additive and multiplicative corrections working
- [ ] Time-dependent bias correction functional
- [ ] Validation framework prevents overfitting
- [ ] Integration tests with all combiners pass

---

### **User Story 6: Weight Validation Framework**
**As a** ML engineer  
**I want** comprehensive validation for combination weights and performance  
**So that** I can ensure combination quality and detect performance degradation  

**Acceptance Criteria:**
- [ ] `WeightValidationFramework` validates weight stability and reasonableness
- [ ] Performance validation against individual models and benchmarks
- [ ] Statistical significance testing for combination improvements
- [ ] Weight drift detection and alerting
- [ ] Combination diagnostics and reporting

**Technical Requirements:**
- Weight stability tests using bootstrapping
- Performance significance testing (t-tests, Wilcoxon)
- Weight drift detection using statistical process control
- Diagnostic plots and performance reports
- Alert system for significant performance changes

**Definition of Done:**
- [ ] `WeightValidationFramework` class implemented
- [ ] Statistical validation tests working
- [ ] Weight drift detection functional
- [ ] Performance reports generated successfully
- [ ] Alert system integrated and tested

---

## 🏗️ Technical Architecture

### Core Components

```python
# Base Interface
class BaseCombiner(ABC):
    """Base interface for model combination strategies."""
    
    @abstractmethod
    def combine(self, predictions: Dict[str, np.ndarray], 
                metadata: Dict[str, Any] = None) -> CombinationResult:
        """Combine predictions from multiple models."""
        pass
    
    @abstractmethod
    def fit(self, train_predictions: Dict[str, np.ndarray], 
            train_targets: np.ndarray) -> None:
        """Train/fit the combination strategy."""
        pass

# Core Implementations
class SimpleAveragingCombiner(BaseCombiner):
    """Equal-weight averaging combination."""
    pass

class WeightedVotingCombiner(BaseCombiner):
    """Optimized weighted combination."""
    pass

class BestModelSelector(BaseCombiner):
    """Dynamic model selection strategy."""
    pass

# Supporting Components
class BasicBiasCorrectionModule:
    """Bias detection and correction."""
    pass

class WeightValidationFramework:
    """Weight and performance validation."""
    pass
```

### Integration Points

1. **Model Registry Integration:** Fetch predictions from all trained models
2. **Feature Pipeline:** Access feature data for context-aware combinations
3. **Evaluation System:** Feed combination results to metrics calculation
4. **Configuration:** YAML-based combiner configuration and parameters
5. **Logging:** Structured logging of combination decisions and performance

### Data Flow

```
Model Predictions → BaseCombiner → Bias Correction → Combined Forecast
                 ↓                                        ↑
            Weight Optimization ← Validation Framework ←--┘
```

---

## 🔧 Implementation Plan

### Week 1: Core Infrastructure (Days 1-5)
- **Day 1-2:** Implement `BaseCombiner` interface and `SimpleAveragingCombiner`
- **Day 3-4:** Develop `WeightedVotingCombiner` with optimization
- **Day 5:** Integration testing and performance validation

### Week 2: Advanced Features (Days 6-10)
- **Day 6-7:** Implement `BestModelSelector` and selection logic
- **Day 8:** Develop `BasicBiasCorrectionModule`
- **Day 9:** Build `WeightValidationFramework`
- **Day 10:** End-to-end testing and performance benchmarking

### Key Deliverables
1. **Combiner Library:** Complete set of base combination strategies
2. **Weight Optimization:** Robust weight learning and validation
3. **Bias Correction:** Basic systematic error correction
4. **Validation Framework:** Comprehensive quality assurance
5. **Performance Reports:** Combination vs individual model analysis

---

## 📊 Success Metrics

### Performance Targets
- **Accuracy Improvement:** LGBM + RF combination achieves <3% MAPE (better than individuals)
- **Weight Stability:** Weight changes <10% between consecutive training periods
- **Processing Performance:** Combination adds <30s to prediction pipeline
- **Bias Reduction:** Systematic bias reduced by >20% with correction

### Quality Gates
- [ ] All combination methods outperform worst individual model
- [ ] Weight optimization converges within 100 iterations
- [ ] Bias correction improves accuracy on validation set
- [ ] Weight validation detects significant drift (p<0.05)
- [ ] Unit test coverage >85%

### Acceptance Criteria
- [ ] Integration with existing model registry successful
- [ ] All 6 user stories completed and tested
- [ ] Performance benchmarks meet or exceed targets
- [ ] Code review and documentation complete
- [ ] Ready for Epic-05B (Advanced Ensemble Methods)

---

## 🧪 Testing Strategy

### Unit Tests
- Individual combiner class functionality
- Weight optimization convergence
- Bias correction calculations
- Validation framework statistics

### Integration Tests
- End-to-end combination pipeline
- Integration with model registry
- Performance against baseline models
- Cross-validation stability

### Performance Tests
- Combination speed benchmarks
- Memory usage validation
- Scalability across 26 time series
- Weight optimization performance

---

## 📚 Dependencies & Risks

### External Dependencies
- **Model Outputs:** Requires Epic-04 (all 5 models) completion
- **Validation Data:** Historical predictions for weight optimization
- **Evaluation Framework:** Basic metrics for performance assessment

### Technical Risks & Mitigation
1. **Weight Instability:** Use regularization and cross-validation
2. **Overfitting:** Implement hold-out validation and early stopping
3. **Performance Degradation:** Comprehensive benchmarking and fallback strategies
4. **Integration Complexity:** Incremental integration with existing components

### Business Risks & Mitigation
1. **Limited Improvement:** Validate on multiple time series and periods
2. **Interpretability Loss:** Maintain weight transparency and logging
3. **Operational Complexity:** Provide clear configuration and monitoring

---

## 🔄 Handoff Criteria

### Deliverables for Epic-05B
- [ ] Complete combiner interface and base implementations
- [ ] Weight optimization framework ready for meta-learning
- [ ] Bias correction foundation for advanced methods
- [ ] Validation framework extensible for ensemble metrics

### Documentation Requirements
- [ ] API documentation for all combiner classes
- [ ] Configuration guide for combination parameters
- [ ] Performance analysis and benchmark results
- [ ] Integration guide for new combination methods

---

## 📈 Success Definition

Epic-05A is successful when:
1. **LGBM + RF combination consistently outperforms individual models** across multiple time series
2. **Weight optimization produces stable, interpretable weights** that improve with more training data
3. **Bias correction reduces systematic errors** without overfitting to correction data
4. **Validation framework reliably detects** weight drift and performance changes
5. **Foundation established** for advanced ensemble methods in Epic-05B

**Ready for Epic-05B when:** All base combination strategies working reliably, weight optimization stable, and bias correction improving accuracy consistently.