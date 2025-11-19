# Epic-06A: Core Hierarchical Reconciliation

**Epic ID:** Epic-06A  
**Epic Name:** Core Hierarchical Reconciliation  
**Phase:** 6A  
**Duration:** 1 week (Week 17)  
**Dependencies:** Epic-04 (Hierarchical Models)  
**Priority:** High  

---

## 🎯 Epic Overview

Implement fundamental hierarchical reconciliation infrastructure that ensures forecast consistency across the electric load forecasting hierarchy (National → Subsystems → Areas). This epic establishes the core reconciliation framework with MinT (Minimum Trace) method as the default approach, ensuring that forecasts satisfy aggregation constraints while maintaining forecast accuracy.

### Business Value
- **Forecast Consistency:** Ensures National = Σ(Subsystems) = Σ(Areas) automatically
- **Regulatory Compliance:** Meets energy balancing requirements for grid operations
- **Stakeholder Trust:** Provides coherent forecasts across all organizational levels
- **Foundation:** Enables advanced reconciliation methods (Epic-06B)

---

## 📋 User Stories

### **User Story 1: Hierarchy Definition System**
**As a** system architect  
**I want** a flexible hierarchy definition system that converts YAML configurations to computational graphs  
**So that** I can define and validate complex hierarchical structures for electric load forecasting  

**Acceptance Criteria:**
- [ ] `HierarchyDefinition` class loads hierarchy from YAML configuration
- [ ] Supports multi-level hierarchies (National → Subsystem → Area)
- [ ] Generates computational graph with aggregation relationships
- [ ] Validates hierarchy completeness and consistency
- [ ] Handles special cases like losses and interconnections

**Technical Requirements:**
- YAML schema: nodes, relationships, aggregation rules, constraints
- Graph representation using NetworkX for efficient traversal
- Validation: no cycles, complete coverage, proper aggregation
- Support for 4 subsystems, 17 areas, 4 losses, 1 national level
- Aggregation matrix S construction: `y_bottom = S * y_top`

**Definition of Done:**
- [ ] `HierarchyDefinition` class implemented with full YAML support
- [ ] Graph construction and validation working
- [ ] Aggregation matrix generation accurate
- [ ] Unit tests cover all hierarchy configurations
- [ ] Integration tests with real hierarchy structure pass

---

### **User Story 2: Base Reconciliation Interface**
**As a** ML engineer  
**I want** a standardized interface for hierarchical reconciliation methods  
**So that** I can easily integrate multiple reconciliation approaches  

**Acceptance Criteria:**
- [ ] `BaseReconciler` interface defines reconciliation contract
- [ ] Supports both point forecasts and prediction intervals
- [ ] Handles missing predictions gracefully
- [ ] Provides reconciliation diagnostics and metadata
- [ ] Extensible design for new reconciliation methods

**Technical Requirements:**
- Abstract base class with `reconcile()` method
- Input: base forecasts, hierarchy definition, optional variance matrix
- Output: reconciled forecasts + reconciliation metadata
- Error handling for incomplete forecast sets
- Performance monitoring and logging integration

**Definition of Done:**
- [ ] `BaseReconciler` abstract class implemented
- [ ] Interface supports all required forecast formats
- [ ] Error handling and validation working
- [ ] Diagnostic output comprehensive
- [ ] Documentation with usage examples complete

---

### **User Story 3: MinT Reconciler Implementation**
**As a** forecasting analyst  
**I want** a robust MinT reconciliation method implementation  
**So that** I can achieve optimal forecast reconciliation with minimum trace covariance  

**Acceptance Criteria:**
- [ ] `MinTReconciler` implements minimum trace reconciliation
- [ ] Handles both structural and sampling covariance estimation
- [ ] Optimized matrix operations for computational efficiency
- [ ] Numerical stability for ill-conditioned covariance matrices
- [ ] Configurable shrinkage parameters for robustness

**Technical Requirements:**
- MinT formula: `y_reconciled = S * (S'ΣS)^(-1) * S' * Σ * y_base`
- Covariance estimation: sample, structural, or shrinkage methods
- Matrix operations: Cholesky decomposition, pseudo-inverse fallback
- Regularization: ridge regularization for numerical stability
- Performance optimization: sparse matrices, efficient linear algebra

**Definition of Done:**
- [ ] `MinTReconciler` class implemented with full MinT algorithm
- [ ] Covariance estimation methods working correctly
- [ ] Matrix operations numerically stable
- [ ] Performance benchmarks meet requirements (<5s for 26 series)
- [ ] Integration tests with hierarchical models pass

---

### **User Story 4: Hierarchy Validation Framework**
**As a** system engineer  
**I want** comprehensive validation for hierarchy definitions and reconciliation results  
**So that** I can ensure data integrity and catch configuration errors early  

**Acceptance Criteria:**
- [ ] `HierarchyValidator` validates hierarchy structure and constraints
- [ ] Real-time validation during reconciliation process
- [ ] Detailed error reporting with specific constraint violations
- [ ] Performance monitoring for validation overhead
- [ ] Configurable validation levels (strict/lenient)

**Technical Requirements:**
- Structure validation: completeness, consistency, no cycles
- Constraint validation: aggregation rules, positive definiteness
- Runtime validation: forecast completeness, dimensional consistency
- Error reporting: specific violations with suggested fixes
- Performance: validation overhead <10% of reconciliation time

**Definition of Done:**
- [ ] `HierarchyValidator` class implemented
- [ ] All validation checks working correctly
- [ ] Error reporting clear and actionable
- [ ] Performance overhead within acceptable limits
- [ ] Unit tests cover all validation scenarios

---

### **User Story 5: Basic Aggregation Constraints**
**As a** forecasting analyst  
**I want** automatic enforcement of aggregation constraints  
**So that** I can ensure forecast consistency without manual intervention  

**Acceptance Criteria:**
- [ ] `ConstraintEnforcer` ensures Subsystem = Σ(Areas) automatically
- [ ] Handles National = Σ(Subsystems) constraint enforcement
- [ ] Manages loss calculations and energy balance
- [ ] Supports both hard and soft constraint enforcement
- [ ] Provides constraint violation reporting and diagnostics

**Technical Requirements:**
- Hard constraints: exact mathematical enforcement via reconciliation
- Soft constraints: penalty-based approaches with configurable weights
- Energy conservation: total generation = total consumption + losses
- Constraint matrices: linear equality constraints for aggregation
- Violation metrics: percentage errors, absolute differences

**Definition of Done:**
- [ ] `ConstraintEnforcer` class implemented
- [ ] Both hard and soft constraint modes working
- [ ] Energy balance calculations correct
- [ ] Constraint violation reporting functional
- [ ] Integration with MinT reconciler seamless

---

### **User Story 6: Loss Calculation by Difference**
**As a** grid operations analyst  
**I want** automatic loss calculation based on generation-consumption differences  
**So that** I can maintain energy balance without explicit loss forecasting  

**Acceptance Criteria:**
- [ ] `LossCalculator` computes transmission losses automatically
- [ ] Supports multiple loss calculation methods (percentage, difference-based)
- [ ] Handles negative losses and validation edge cases
- [ ] Integrates with reconciliation to maintain energy balance
- [ ] Provides loss analysis and reporting capabilities

**Technical Requirements:**
- Loss calculation: `Losses = Generation - Consumption - Exports + Imports`
- Percentage-based: `Loss_rate = Losses / Generation`
- Validation: losses within reasonable bounds (0-15% typical)
- Integration: losses as calculated series in reconciliation
- Reporting: loss statistics, trends, anomaly detection

**Definition of Done:**
- [ ] `LossCalculator` class implemented
- [ ] Multiple calculation methods working
- [ ] Validation and bounds checking functional
- [ ] Integration with reconciliation complete
- [ ] Loss reporting and analysis operational

---

## 🏗️ Technical Architecture

### Core Components

```python
# Hierarchy Management
@dataclass
class HierarchyNode:
    """Represents a single node in the hierarchy."""
    name: str
    level: int
    children: List['HierarchyNode']
    parent: Optional['HierarchyNode']
    aggregation_weights: Dict[str, float]

class HierarchyDefinition:
    """Manages hierarchy structure and relationships."""
    
    def __init__(self, config_path: str):
        self.nodes = {}
        self.aggregation_matrix = None
        self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str) -> None:
        """Load hierarchy from YAML configuration."""
        pass
    
    def build_aggregation_matrix(self) -> np.ndarray:
        """Construct aggregation matrix S."""
        pass
    
    def validate_structure(self) -> List[str]:
        """Validate hierarchy completeness and consistency."""
        pass

# Reconciliation Framework
class BaseReconciler(ABC):
    """Base interface for hierarchical reconciliation."""
    
    @abstractmethod
    def reconcile(self, base_forecasts: Dict[str, np.ndarray],
                 hierarchy: HierarchyDefinition,
                 covariance_matrix: Optional[np.ndarray] = None) -> ReconciliationResult:
        """Reconcile base forecasts to satisfy hierarchy constraints."""
        pass

class MinTReconciler(BaseReconciler):
    """Minimum Trace reconciliation implementation."""
    
    def __init__(self, shrinkage_param: float = 0.01):
        self.shrinkage_param = shrinkage_param
        self.covariance_estimator = CovarianceEstimator()
    
    def reconcile(self, base_forecasts: Dict, hierarchy: HierarchyDefinition,
                 covariance_matrix: Optional[np.ndarray] = None) -> ReconciliationResult:
        """Perform MinT reconciliation."""
        pass
    
    def estimate_covariance(self, residuals: np.ndarray, 
                          method: str = 'sample') -> np.ndarray:
        """Estimate forecast covariance matrix."""
        pass

# Validation and Constraints
class HierarchyValidator:
    """Validates hierarchy and reconciliation results."""
    
    def validate_hierarchy(self, hierarchy: HierarchyDefinition) -> ValidationResult:
        """Validate hierarchy structure."""
        pass
    
    def validate_forecasts(self, forecasts: Dict, hierarchy: HierarchyDefinition) -> ValidationResult:
        """Validate forecast completeness and consistency."""
        pass

class ConstraintEnforcer:
    """Enforces aggregation and energy balance constraints."""
    
    def enforce_aggregation_constraints(self, forecasts: Dict,
                                      hierarchy: HierarchyDefinition) -> Dict:
        """Enforce basic aggregation constraints."""
        pass

class LossCalculator:
    """Calculates transmission losses by difference."""
    
    def calculate_losses(self, generation: np.ndarray, consumption: np.ndarray,
                        method: str = 'difference') -> np.ndarray:
        """Calculate system losses."""
        pass
```

### Data Flow Architecture

```
Base Forecasts → Hierarchy Validation → Covariance Estimation → MinT Reconciliation
                                    ↓
Constraint Enforcement ← Loss Calculation ← Reconciled Forecasts
                                    ↓
Final Validated Forecasts + Diagnostics
```

### Hierarchy Structure (YAML Example)

```yaml
hierarchy:
  national:
    name: "BR_National"
    children: ["subsystem_se", "subsystem_s", "subsystem_ne", "subsystem_n"]
  
  subsystems:
    subsystem_se:
      name: "SE_Subsystem"
      children: ["area_1", "area_2", "area_3", "area_4", "area_5"]
    # ... other subsystems
  
  areas:
    area_1:
      name: "Area_1"
      type: "consumption"
    # ... other areas
  
  constraints:
    energy_balance: true
    loss_calculation: "difference"
    validation_tolerance: 0.001
```

---

## 🔧 Implementation Plan

### Week 1: Core Infrastructure (Days 1-5)
- **Day 1:** Implement `HierarchyDefinition` class with YAML loading
- **Day 2:** Develop `BaseReconciler` interface and `HierarchyValidator`
- **Day 3:** Build `MinTReconciler` with covariance estimation
- **Day 4:** Implement `ConstraintEnforcer` and `LossCalculator`
- **Day 5:** Integration testing and performance optimization

### Key Deliverables
1. **Hierarchy Framework:** Complete hierarchy definition and validation system
2. **MinT Reconciliation:** Robust MinT implementation with numerical stability
3. **Constraint System:** Automatic aggregation and energy balance enforcement
4. **Validation Suite:** Comprehensive validation and error reporting
5. **Loss Management:** Automatic loss calculation and integration

---

## 📊 Success Metrics

### Performance Targets
- **Reconciliation Speed:** <5 seconds for 26 time series reconciliation
- **Numerical Stability:** Successful reconciliation for condition numbers up to 1e12
- **Memory Efficiency:** <500MB RAM usage for full hierarchy reconciliation
- **Constraint Satisfaction:** 100% aggregation constraint compliance

### Quality Gates
- [ ] Hierarchy correctly defined and validated (4 subsystems, 17 areas)
- [ ] MinT reconciliation working with proper matrix operations
- [ ] Subsystem = Σ(Areas) constraint enforced automatically
- [ ] National = Σ(Subsystems) constraint enforced automatically
- [ ] Loss calculations accurate and efficient
- [ ] Reconciliation interface extensible for Epic-06B

### Acceptance Criteria
- [ ] All 6 user stories completed and tested
- [ ] Integration with Epic-04 hierarchical models successful
- [ ] Performance benchmarks meet requirements
- [ ] Numerical stability validated across scenarios
- [ ] Ready for Epic-06B (Advanced Reconciliation Methods)

---

## 🧪 Testing Strategy

### Unit Tests
- Hierarchy definition and validation
- MinT algorithm implementation
- Matrix operations and numerical stability
- Constraint enforcement logic
- Loss calculation methods

### Integration Tests
- End-to-end reconciliation pipeline
- Integration with hierarchical models
- Performance under various hierarchy sizes
- Edge case handling (missing data, singular matrices)

### Performance Tests
- Reconciliation speed benchmarks
- Memory usage validation
- Numerical stability stress tests
- Scalability across different hierarchy structures

---

## 📚 Dependencies & Risks

### External Dependencies
- **Epic-04 Completion:** Hierarchical models providing base forecasts
- **Forecast Data:** Historical residuals for covariance estimation
- **Hierarchy Definition:** Accurate system topology and relationships

### Technical Risks & Mitigation
1. **Numerical Instability:** Robust matrix operations with regularization
2. **Performance Issues:** Optimized linear algebra and sparse matrices
3. **Configuration Complexity:** Clear YAML schema and validation
4. **Integration Challenges:** Comprehensive interface testing

### Business Risks & Mitigation
1. **Forecast Accuracy Loss:** Benchmark against unreconciled forecasts
2. **Operational Complexity:** Clear documentation and error reporting
3. **System Reliability:** Fallback to unreconciled forecasts if needed

---

## 🔄 Handoff Criteria

### Deliverables for Epic-06B
- [ ] Complete reconciliation interface ready for advanced methods
- [ ] Hierarchy framework extensible for multiple reconciliation approaches
- [ ] Covariance estimation foundation for weighted reconciliation
- [ ] Performance baseline for advanced method comparison

### Documentation Requirements
- [ ] Hierarchy configuration guide and YAML schema
- [ ] MinT reconciliation methodology documentation
- [ ] API documentation for all reconciliation classes
- [ ] Integration guide with model outputs

---

## 📈 Success Definition

Epic-06A is successful when:
1. **Hierarchy correctly defined and validated** with 4 subsystems and 17 areas structure
2. **MinT reconciliation working reliably** with proper matrix operations and numerical stability
3. **Aggregation constraints automatically enforced** ensuring Subsystem = Σ(Areas) and National = Σ(Subsystems)
4. **Loss calculations accurate and integrated** maintaining energy balance
5. **Reconciliation interface established** providing foundation for Epic-06B advanced methods
6. **Performance targets achieved** with <5s reconciliation time for full hierarchy

**Ready for Epic-06B when:** Core reconciliation infrastructure working reliably, MinT method validated, hierarchy framework extensible, and performance benchmarks met.