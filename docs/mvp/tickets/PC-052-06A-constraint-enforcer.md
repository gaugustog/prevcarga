# PC-052-06A: Constraint Enforcement System

**Ticket ID:** PC-052-06A  
**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**User Story:** US-5  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement constraint enforcement system that ensures reconciled forecasts satisfy hierarchical aggregation rules, energy balance constraints, and business rules. Supports hard constraints (must satisfy) and soft constraints (penalized violations), configurable tolerance levels, and post-reconciliation adjustment methods. Provides detailed violation reports and automatic correction strategies.

**As a** forecasting system  
**I want** automatic constraint enforcement  
**So that** reconciled forecasts always satisfy business rules and physical constraints

---

## ✅ Acceptance Criteria

- [ ] Hard constraint enforcement (zero tolerance)
- [ ] Soft constraint enforcement (penalized violations)
- [ ] Aggregation constraint validation
- [ ] Energy balance constraint checking
- [ ] Non-negativity constraints (optional)
- [ ] Configurable tolerance levels
- [ ] Post-reconciliation adjustment methods
- [ ] Violation detection and reporting
- [ ] Performance impact <5%
- [ ] Integration with reconciliation pipeline

---

## 🔧 Implementation Tasks

### 1. Create Constraint Module Structure
- [ ] Create `src/models/reconciliation/constraints/__init__.py`
- [ ] Create `src/models/reconciliation/constraints/constraint_enforcer.py`
- [ ] Create `src/models/reconciliation/constraints/constraint_types.py`
- [ ] Import optimization libraries (scipy.optimize)
- [ ] Add module docstrings

### 2. Implement Constraint Dataclasses
- [ ] Create `Constraint` base dataclass
- [ ] Add constraint_type attribute
- [ ] Add severity attribute (HARD/SOFT)
- [ ] Add tolerance attribute
- [ ] Add penalty_weight attribute

### 3. Implement AggregationConstraint Class
- [ ] Inherit from Constraint
- [ ] Add parent_node attribute
- [ ] Add child_nodes list
- [ ] Add aggregation_weights dict
- [ ] Implement check() method
- [ ] Implement enforce() method

### 4. Implement EnergyBalanceConstraint Class
- [ ] Inherit from Constraint
- [ ] Define energy balance equation
- [ ] Add generation_nodes list
- [ ] Add consumption_nodes list
- [ ] Add loss_nodes list
- [ ] Implement check() and enforce()

### 5. Implement NonNegativityConstraint Class
- [ ] Inherit from Constraint
- [ ] Apply to specified nodes
- [ ] Enforce forecasts ≥ 0
- [ ] Optional floor value
- [ ] Implement check() and enforce()

### 6. Implement ConstraintEnforcer Class
- [ ] Create main `ConstraintEnforcer` class
- [ ] Initialize with constraint list
- [ ] Add tolerance configuration
- [ ] Add enforcement strategy
- [ ] Add logging

### 7. Implement Hard Constraint Enforcement
- [ ] Create `_enforce_hard_constraints()` method
- [ ] Iteratively adjust forecasts
- [ ] Ensure zero violation (within tolerance)
- [ ] Use projection methods
- [ ] Return adjusted forecasts

### 8. Implement Soft Constraint Enforcement
- [ ] Create `_enforce_soft_constraints()` method
- [ ] Formulate as optimization problem
- [ ] Minimize: ||y - y_reconciled||² + λΣ(violations²)
- [ ] Use scipy.optimize.minimize
- [ ] Return optimal forecasts

### 9. Implement Projection Methods
- [ ] Create `_project_onto_constraints()` method
- [ ] Project to nearest feasible point
- [ ] Use closed-form projections where possible
- [ ] Iterative projection for complex constraints
- [ ] Return projected forecasts

### 10. Implement Aggregation Constraint Adjustment
- [ ] Create `_adjust_aggregation()` method
- [ ] Proportional adjustment strategy
- [ ] Top-down or bottom-up adjustment
- [ ] Preserve relative relationships
- [ ] Return adjusted values

### 11. Implement Violation Detection
- [ ] Create `check_constraints()` method
- [ ] Evaluate all constraints
- [ ] Calculate violation magnitudes
- [ ] Classify by severity
- [ ] Return violation report

### 12. Implement Violation Report Generation
- [ ] Create `ConstraintViolationReport` dataclass
- [ ] List violations by constraint
- [ ] Calculate statistics (max, mean, count)
- [ ] Add severity classification
- [ ] Include suggested fixes

### 13. Implement Enforcement Strategies
- [ ] Strategy: PROJECTION (geometric projection)
- [ ] Strategy: OPTIMIZATION (constrained optimization)
- [ ] Strategy: ITERATIVE (successive adjustment)
- [ ] Strategy: HYBRID (combination)
- [ ] Configure via enforcement_strategy

### 14. Implement Integration with Reconciliation
- [ ] Create `apply_constraints()` method
- [ ] Hook into reconciliation pipeline
- [ ] Post-reconciliation enforcement
- [ ] Validate before returning results
- [ ] Return constrained forecasts

### 15. Handle Edge Cases
- [ ] Handle infeasible constraint sets
- [ ] Handle conflicting constraints
- [ ] Handle numerical instability
- [ ] Handle extreme forecast values
- [ ] Graceful degradation

### 16. Implement Performance Optimization
- [ ] Cache constraint evaluations
- [ ] Vectorize constraint checks
- [ ] Use sparse representations
- [ ] Parallelize independent constraints
- [ ] Target <5% performance impact

### 17. Write Comprehensive Tests
- [ ] Create `tests/models/reconciliation/constraints/test_constraint_enforcer.py`
- [ ] Test each constraint type
- [ ] Test hard constraint enforcement
- [ ] Test soft constraint enforcement
- [ ] Test violation detection
- [ ] Test edge cases
- [ ] Test performance impact

### 18. Create Constraint Documentation
- [ ] Create `docs/constraint_enforcement.md`
- [ ] Document constraint types
- [ ] Show enforcement strategies
- [ ] Provide examples
- [ ] Document configuration

---

## 💻 Implementation Details

### ConstraintEnforcer Implementation

```python
"""Constraint enforcement system for hierarchical reconciliation."""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np
from scipy.optimize import minimize

from src.models.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConstraintSeverity(Enum):
    """Constraint severity levels."""
    HARD = "HARD"    # Must be satisfied exactly (within tolerance)
    SOFT = "SOFT"    # Violations penalized but allowed


class EnforcementStrategy(Enum):
    """Constraint enforcement strategies."""
    PROJECTION = "PROJECTION"          # Geometric projection
    OPTIMIZATION = "OPTIMIZATION"      # Constrained optimization
    ITERATIVE = "ITERATIVE"           # Successive adjustment
    HYBRID = "HYBRID"                  # Combination of methods


@dataclass
class Constraint:
    """Base constraint class."""
    
    constraint_type: str
    severity: ConstraintSeverity
    tolerance: float = 1e-6
    penalty_weight: float = 1.0
    
    def check(self, forecasts: Dict[str, np.ndarray]) -> float:
        """
        Check constraint violation.
        
        Returns:
            Violation magnitude (0 if satisfied)
        """
        raise NotImplementedError
    
    def enforce(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Enforce constraint on forecasts.
        
        Returns:
            Adjusted forecasts
        """
        raise NotImplementedError


@dataclass
class AggregationConstraint(Constraint):
    """
    Aggregation constraint: parent = Σ(weighted children).
    
    Example:
        National = SE_Subsystem + S_Subsystem + NE_Subsystem + N_Subsystem
    """
    
    parent_node: str
    child_nodes: List[str]
    aggregation_weights: Dict[str, float]
    
    def __post_init__(self):
        """Initialize constraint type."""
        self.constraint_type = 'AGGREGATION'
    
    def check(self, forecasts: Dict[str, np.ndarray]) -> float:
        """Check aggregation constraint."""
        if self.parent_node not in forecasts:
            return 0.0
        
        parent_forecast = forecasts[self.parent_node]
        
        # Calculate expected value (sum of children)
        expected = np.zeros_like(parent_forecast)
        for child in self.child_nodes:
            if child in forecasts:
                weight = self.aggregation_weights.get(child, 1.0)
                expected += weight * forecasts[child]
        
        # Calculate violation
        violation = np.abs(parent_forecast - expected)
        return np.max(violation)
    
    def enforce(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Enforce aggregation constraint by adjusting parent."""
        adjusted = forecasts.copy()
        
        if self.parent_node not in adjusted:
            return adjusted
        
        # Calculate correct parent value
        correct_parent = np.zeros_like(adjusted[self.parent_node])
        for child in self.child_nodes:
            if child in adjusted:
                weight = self.aggregation_weights.get(child, 1.0)
                correct_parent += weight * adjusted[child]
        
        # Adjust parent
        adjusted[self.parent_node] = correct_parent
        
        return adjusted


@dataclass
class EnergyBalanceConstraint(Constraint):
    """
    Energy balance constraint: Generation = Consumption + Losses.
    
    Example:
        Total_Generation = Total_Consumption + Transmission_Losses
    """
    
    generation_nodes: List[str]
    consumption_nodes: List[str]
    loss_nodes: List[str]
    
    def __post_init__(self):
        """Initialize constraint type."""
        self.constraint_type = 'ENERGY_BALANCE'
    
    def check(self, forecasts: Dict[str, np.ndarray]) -> float:
        """Check energy balance constraint."""
        # Sum generation
        generation = np.zeros(len(next(iter(forecasts.values()))))
        for node in self.generation_nodes:
            if node in forecasts:
                generation += forecasts[node]
        
        # Sum consumption
        consumption = np.zeros_like(generation)
        for node in self.consumption_nodes:
            if node in forecasts:
                consumption += forecasts[node]
        
        # Sum losses
        losses = np.zeros_like(generation)
        for node in self.loss_nodes:
            if node in forecasts:
                losses += forecasts[node]
        
        # Check balance
        violation = np.abs(generation - (consumption + losses))
        return np.max(violation)
    
    def enforce(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Enforce energy balance by adjusting losses."""
        adjusted = forecasts.copy()
        
        # Calculate required losses
        generation = np.zeros(len(next(iter(forecasts.values()))))
        for node in self.generation_nodes:
            if node in adjusted:
                generation += adjusted[node]
        
        consumption = np.zeros_like(generation)
        for node in self.consumption_nodes:
            if node in adjusted:
                consumption += adjusted[node]
        
        required_losses = generation - consumption
        
        # Distribute losses proportionally
        if self.loss_nodes and all(node in adjusted for node in self.loss_nodes):
            total_current_losses = sum(adjusted[node] for node in self.loss_nodes)
            
            for node in self.loss_nodes:
                if total_current_losses > 0:
                    proportion = adjusted[node] / total_current_losses
                else:
                    proportion = 1.0 / len(self.loss_nodes)
                
                adjusted[node] = proportion * required_losses
        
        return adjusted


@dataclass
class NonNegativityConstraint(Constraint):
    """Non-negativity constraint: forecasts ≥ floor_value."""
    
    nodes: List[str]
    floor_value: float = 0.0
    
    def __post_init__(self):
        """Initialize constraint type."""
        self.constraint_type = 'NON_NEGATIVITY'
    
    def check(self, forecasts: Dict[str, np.ndarray]) -> float:
        """Check non-negativity constraint."""
        max_violation = 0.0
        
        for node in self.nodes:
            if node in forecasts:
                violations = self.floor_value - forecasts[node]
                violations = violations[violations > 0]  # Only negative values
                if len(violations) > 0:
                    max_violation = max(max_violation, np.max(violations))
        
        return max_violation
    
    def enforce(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Enforce non-negativity by clipping."""
        adjusted = forecasts.copy()
        
        for node in self.nodes:
            if node in adjusted:
                adjusted[node] = np.maximum(adjusted[node], self.floor_value)
        
        return adjusted


@dataclass
class ConstraintViolationReport:
    """Report of constraint violations."""
    
    violations: List[Dict[str, Any]]
    total_violations: int
    max_violation: float
    mean_violation: float
    hard_constraint_violations: int
    soft_constraint_violations: int


class ConstraintEnforcer:
    """
    Enforces constraints on hierarchical forecasts.
    
    Supports:
        - Hard constraints (must satisfy exactly)
        - Soft constraints (penalized violations)
        - Multiple enforcement strategies
        - Automatic violation detection
    
    Enforcement Strategies:
        - PROJECTION: Geometric projection onto constraint set
        - OPTIMIZATION: Solve constrained optimization problem
        - ITERATIVE: Successive constraint adjustment
        - HYBRID: Combine multiple methods
    
    Example:
        >>> # Define constraints
        >>> constraints = [
        ...     AggregationConstraint(
        ...         parent_node='National',
        ...         child_nodes=['SE', 'S', 'NE', 'N'],
        ...         aggregation_weights={'SE': 1.0, 'S': 1.0, 'NE': 1.0, 'N': 1.0},
        ...         severity=ConstraintSeverity.HARD,
        ...         tolerance=1e-6
        ...     ),
        ...     NonNegativityConstraint(
        ...         nodes=['National', 'SE', 'S'],
        ...         severity=ConstraintSeverity.HARD
        ...     )
        ... ]
        >>> 
        >>> # Enforce constraints
        >>> enforcer = ConstraintEnforcer(
        ...     constraints=constraints,
        ...     strategy=EnforcementStrategy.ITERATIVE
        ... )
        >>> adjusted_forecasts = enforcer.enforce(forecasts)
        >>> 
        >>> # Check violations
        >>> report = enforcer.check_violations(adjusted_forecasts)
        >>> print(f"Violations: {report.total_violations}")
    """
    
    def __init__(
        self,
        constraints: List[Constraint],
        strategy: EnforcementStrategy = EnforcementStrategy.ITERATIVE,
        max_iterations: int = 100,
        convergence_tolerance: float = 1e-8
    ):
        """
        Initialize constraint enforcer.
        
        Args:
            constraints: List of constraints to enforce
            strategy: Enforcement strategy
            max_iterations: Maximum iterations for iterative methods
            convergence_tolerance: Convergence criterion
        """
        self.constraints = constraints
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
    
    def enforce(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Enforce all constraints on forecasts.
        
        Args:
            forecasts: Forecast dictionary
        
        Returns:
            Constrained forecasts
        """
        logger.info(f"Enforcing {len(self.constraints)} constraints using {self.strategy.value}")
        
        if self.strategy == EnforcementStrategy.ITERATIVE:
            return self._enforce_iterative(forecasts)
        elif self.strategy == EnforcementStrategy.PROJECTION:
            return self._enforce_projection(forecasts)
        elif self.strategy == EnforcementStrategy.OPTIMIZATION:
            return self._enforce_optimization(forecasts)
        elif self.strategy == EnforcementStrategy.HYBRID:
            return self._enforce_hybrid(forecasts)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _enforce_iterative(
        self,
        forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Enforce constraints iteratively."""
        adjusted = forecasts.copy()
        
        for iteration in range(self.max_iterations):
            prev_adjusted = {k: v.copy() for k, v in adjusted.items()}
            
            # Apply each constraint
            for constraint in self.constraints:
                if constraint.severity == ConstraintSeverity.HARD:
                    adjusted = constraint.enforce(adjusted)
            
            # Check convergence
            max_change = max(
                np.max(np.abs(adjusted[k] - prev_adjusted[k]))
                for k in adjusted.keys()
            )
            
            if max_change < self.convergence_tolerance:
                logger.info(f"Converged in {iteration+1} iterations")
                break
        
        return adjusted
    
    def _enforce_projection(
        self,
        forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Enforce constraints via projection."""
        # Simplified projection method
        return self._enforce_iterative(forecasts)
    
    def _enforce_optimization(
        self,
        forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Enforce constraints via optimization."""
        # This would use scipy.optimize.minimize with constraints
        # For now, fallback to iterative
        logger.warning("Optimization method not fully implemented, using iterative")
        return self._enforce_iterative(forecasts)
    
    def _enforce_hybrid(
        self,
        forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Enforce constraints using hybrid approach."""
        # Hard constraints via iterative
        adjusted = self._enforce_iterative(forecasts)
        
        # Soft constraints via optimization (if needed)
        # ...
        
        return adjusted
    
    def check_violations(
        self,
        forecasts: Dict[str, np.ndarray]
    ) -> ConstraintViolationReport:
        """
        Check constraint violations.
        
        Args:
            forecasts: Forecasts to check
        
        Returns:
            Violation report
        """
        violations = []
        hard_violations = 0
        soft_violations = 0
        
        for constraint in self.constraints:
            violation_magnitude = constraint.check(forecasts)
            
            if violation_magnitude > constraint.tolerance:
                violations.append({
                    'constraint_type': constraint.constraint_type,
                    'severity': constraint.severity.value,
                    'magnitude': float(violation_magnitude),
                    'tolerance': constraint.tolerance
                })
                
                if constraint.severity == ConstraintSeverity.HARD:
                    hard_violations += 1
                else:
                    soft_violations += 1
        
        if violations:
            max_viol = max(v['magnitude'] for v in violations)
            mean_viol = np.mean([v['magnitude'] for v in violations])
        else:
            max_viol = 0.0
            mean_viol = 0.0
        
        return ConstraintViolationReport(
            violations=violations,
            total_violations=len(violations),
            max_violation=max_viol,
            mean_violation=mean_viol,
            hard_constraint_violations=hard_violations,
            soft_constraint_violations=soft_violations
        )
```

---

## 🧪 Testing & Validation

```python
"""Tests for constraint enforcer."""
import pytest
import numpy as np

from src.models.reconciliation.constraints.constraint_enforcer import (
    ConstraintEnforcer, AggregationConstraint, NonNegativityConstraint,
    ConstraintSeverity, EnforcementStrategy
)


def test_aggregation_constraint():
    """Test aggregation constraint enforcement."""
    constraint = AggregationConstraint(
        parent_node='National',
        child_nodes=['SE', 'S'],
        aggregation_weights={'SE': 1.0, 'S': 1.0},
        severity=ConstraintSeverity.HARD,
        tolerance=1e-6
    )
    
    forecasts = {
        'National': np.array([1000, 1100]),
        'SE': np.array([600, 650]),
        'S': np.array([400, 450])
    }
    
    # Violate constraint
    forecasts['National'] = np.array([900, 1000])
    
    violation = constraint.check(forecasts)
    assert violation > 0
    
    # Enforce
    adjusted = constraint.enforce(forecasts)
    violation_after = constraint.check(adjusted)
    assert violation_after < 1e-6


def test_non_negativity_constraint():
    """Test non-negativity constraint."""
    constraint = NonNegativityConstraint(
        nodes=['Area_1'],
        severity=ConstraintSeverity.HARD
    )
    
    forecasts = {
        'Area_1': np.array([100, -50, 200])
    }
    
    violation = constraint.check(forecasts)
    assert violation == 50
    
    adjusted = constraint.enforce(forecasts)
    assert np.all(adjusted['Area_1'] >= 0)


def test_constraint_enforcer():
    """Test constraint enforcer."""
    constraints = [
        AggregationConstraint(
            parent_node='National',
            child_nodes=['SE', 'S'],
            aggregation_weights={'SE': 1.0, 'S': 1.0},
            severity=ConstraintSeverity.HARD
        )
    ]
    
    enforcer = ConstraintEnforcer(
        constraints=constraints,
        strategy=EnforcementStrategy.ITERATIVE
    )
    
    forecasts = {
        'National': np.array([900]),
        'SE': np.array([600]),
        'S': np.array([400])
    }
    
    adjusted = enforcer.enforce(forecasts)
    assert adjusted['National'][0] == 1000  # Should be sum of children
```

---

## 📝 Technical Notes

### Enforcement Methods
- Iterative projection for hard constraints
- Optimization for soft constraints
- Hybrid approach for complex scenarios

### Performance
- Target overhead <5%
- Vectorized operations
- Early convergence detection

---

## 🔗 Dependencies

**Depends On:**
- PC-048-06A: Hierarchy Definition System
- PC-051-06A: Hierarchy Validation Framework

**Blocks:**
- PC-053-06A: Loss Calculation System
- Epic-06B: Advanced Reconciliation Methods

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Hard constraint enforcement working
- [ ] Soft constraint enforcement implemented
- [ ] All constraint types functional
- [ ] Violation detection accurate
- [ ] Multiple enforcement strategies
- [ ] Performance overhead <5%
- [ ] Unit tests >85% coverage
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**Previous:** [PC-051-06A: Hierarchy Validation Framework](PC-051-06A-hierarchy-validator.md)  
**Next:** [PC-053-06A: Loss Calculation System](PC-053-06A-loss-calculator.md)
