# PC-051-06A: Hierarchy Validation Framework

**Ticket ID:** PC-051-06A  
**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**User Story:** US-4  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement comprehensive validation framework for hierarchy structure and forecast coherence. Validates hierarchy topology (cycles, completeness, orphans), runtime forecast validation, aggregation constraint checking, and detailed error reporting. Ensures hierarchy integrity before reconciliation and provides diagnostics for constraint violations.

**As a** data quality engineer  
**I want** comprehensive hierarchy and forecast validation  
**So that** I can ensure data quality and identify issues before reconciliation

---

## ✅ Acceptance Criteria

- [ ] Hierarchy structure validation (cycles, completeness)
- [ ] Forecast coherence validation
- [ ] Aggregation constraint checking
- [ ] Detailed validation error reporting
- [ ] Performance overhead <10%
- [ ] Configurable validation levels (strict/lenient)
- [ ] Integration with reconciliation pipeline
- [ ] Validation reports with actionable insights
- [ ] Support for partial hierarchy validation
- [ ] Historical validation tracking

---

## 🔧 Implementation Tasks

### 1. Create Validation Module Structure
- [ ] Create `src/models/reconciliation/validation/__init__.py`
- [ ] Create `src/models/reconciliation/validation/hierarchy_validator.py`
- [ ] Create `src/models/reconciliation/validation/forecast_validator.py`
- [ ] Create validation result dataclasses
- [ ] Add module docstrings

### 2. Implement ValidationResult Dataclass
- [ ] Create `ValidationResult` dataclass
- [ ] Add is_valid boolean attribute
- [ ] Add errors list attribute
- [ ] Add warnings list attribute
- [ ] Add validation_time attribute
- [ ] Add validation_level attribute

### 3. Implement ValidationError Dataclass
- [ ] Create `ValidationError` dataclass
- [ ] Add error_type attribute
- [ ] Add severity attribute (ERROR/WARNING/INFO)
- [ ] Add message attribute
- [ ] Add context dict attribute
- [ ] Add suggested_fix attribute

### 4. Implement HierarchyValidator Class
- [ ] Create `HierarchyValidator` class
- [ ] Add validation level configuration
- [ ] Add strict/lenient modes
- [ ] Initialize with tolerance parameters
- [ ] Add logging

### 5. Implement Cycle Detection
- [ ] Create `_detect_cycles()` method
- [ ] Use depth-first search
- [ ] Detect circular dependencies
- [ ] Report cycle paths
- [ ] Mark as critical error

### 6. Implement Completeness Validation
- [ ] Create `_validate_completeness()` method
- [ ] Check all nodes have required attributes
- [ ] Verify aggregation weights sum to 1.0
- [ ] Check parent-child consistency
- [ ] Validate node types

### 7. Implement Orphan Detection
- [ ] Create `_detect_orphans()` method
- [ ] Find nodes without parents (except root)
- [ ] Find nodes without children (leaf validation)
- [ ] Report unreachable nodes
- [ ] Mark as warning or error

### 8. Implement Aggregation Matrix Validation
- [ ] Create `_validate_aggregation_matrix()` method
- [ ] Check matrix dimensions
- [ ] Verify row sums (should be n_bottom)
- [ ] Check for zero columns
- [ ] Validate sparsity pattern

### 9. Implement Forecast Coherence Validation
- [ ] Create `ForecastValidator` class
- [ ] Check forecast dimensions
- [ ] Validate data types
- [ ] Check for NaN/Inf values
- [ ] Verify time alignment

### 10. Implement Constraint Checking
- [ ] Create `_check_aggregation_constraints()` method
- [ ] Verify Σ(children) = parent
- [ ] Calculate violation magnitudes
- [ ] Compute relative errors
- [ ] Return constraint report

### 11. Implement Multi-Level Validation
- [ ] Create `validate_hierarchy()` main method
- [ ] Run structure validation
- [ ] Run topology validation
- [ ] Run consistency validation
- [ ] Aggregate all results
- [ ] Return ValidationResult

### 12. Implement Forecast Validation Pipeline
- [ ] Create `validate_forecasts()` method
- [ ] Check forecast availability
- [ ] Validate forecast ranges
- [ ] Check temporal consistency
- [ ] Validate against hierarchy
- [ ] Return validation result

### 13. Implement Error Reporting
- [ ] Create `_generate_report()` method
- [ ] Format validation errors
- [ ] Group by severity
- [ ] Add suggested fixes
- [ ] Generate summary statistics
- [ ] Return formatted report

### 14. Implement Validation Levels
- [ ] Support STRICT mode (all errors fail)
- [ ] Support LENIENT mode (warnings only)
- [ ] Support DIAGNOSTIC mode (log all)
- [ ] Configure via validation_level
- [ ] Document level behavior

### 15. Implement Tolerance Configuration
- [ ] Create `ValidationConfig` dataclass
- [ ] Add constraint_tolerance parameter
- [ ] Add missing_data_tolerance parameter
- [ ] Add time_alignment_tolerance parameter
- [ ] Load from config file

### 16. Handle Edge Cases
- [ ] Handle empty hierarchy
- [ ] Handle single-node hierarchy
- [ ] Handle missing forecast data
- [ ] Handle partial hierarchies
- [ ] Handle dynamic hierarchies

### 17. Write Comprehensive Tests
- [ ] Create `tests/models/reconciliation/validation/test_hierarchy_validator.py`
- [ ] Test cycle detection
- [ ] Test completeness validation
- [ ] Test orphan detection
- [ ] Test constraint checking
- [ ] Test validation levels
- [ ] Test edge cases

### 18. Create Validation Documentation
- [ ] Create `docs/validation_framework.md`
- [ ] Document validation rules
- [ ] Provide examples
- [ ] Document error codes
- [ ] Show suggested fixes

---

## 💻 Implementation Details

### HierarchyValidator Implementation

```python
"""Hierarchy validation framework."""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import pandas as pd
import numpy as np
import time

from src.models.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "STRICT"      # All errors cause validation failure
    LENIENT = "LENIENT"    # Only critical errors cause failure
    DIAGNOSTIC = "DIAGNOSTIC"  # All issues logged, no failure


@dataclass
class ValidationError:
    """Validation error or warning."""
    
    error_type: str
    severity: ValidationSeverity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation process."""
    
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    validation_time: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.STRICT
    
    def get_errors_by_severity(self, severity: ValidationSeverity) -> List[ValidationError]:
        """Get errors by severity level."""
        all_issues = self.errors + self.warnings + self.info
        return [e for e in all_issues if e.severity == severity]
    
    def has_critical_errors(self) -> bool:
        """Check if critical errors present."""
        return len(self.errors) > 0
    
    def summary(self) -> str:
        """Generate validation summary."""
        return (
            f"Validation {'PASSED' if self.is_valid else 'FAILED'}\n"
            f"Errors: {len(self.errors)}, "
            f"Warnings: {len(self.warnings)}, "
            f"Info: {len(self.info)}\n"
            f"Time: {self.validation_time:.3f}s"
        )


@dataclass
class ValidationConfig:
    """Configuration for validation."""
    
    validation_level: ValidationLevel = ValidationLevel.STRICT
    constraint_tolerance: float = 1e-6
    missing_data_tolerance: float = 0.1  # 10% missing allowed
    enable_cycle_detection: bool = True
    enable_orphan_detection: bool = True
    enable_constraint_check: bool = True


class HierarchyValidator:
    """
    Validates hierarchy structure and forecast coherence.
    
    Performs comprehensive validation including:
        - Topology validation (cycles, connectivity)
        - Structural validation (completeness, consistency)
        - Constraint validation (aggregation rules)
        - Forecast validation (coherence, quality)
    
    Validation Levels:
        - STRICT: All errors cause failure
        - LENIENT: Only critical errors cause failure
        - DIAGNOSTIC: All issues logged, no failure
    
    Example:
        >>> validator = HierarchyValidator(config=ValidationConfig(
        ...     validation_level=ValidationLevel.STRICT,
        ...     constraint_tolerance=1e-6
        ... ))
        >>> result = validator.validate_hierarchy(hierarchy)
        >>> if not result.is_valid:
        ...     print(result.summary())
        ...     for error in result.errors:
        ...         print(f"- {error.message}")
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize hierarchy validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
    
    def validate_hierarchy(
        self,
        hierarchy: HierarchyDefinition
    ) -> ValidationResult:
        """
        Validate hierarchy structure.
        
        Args:
            hierarchy: Hierarchy to validate
        
        Returns:
            ValidationResult with errors and warnings
        """
        start_time = time.time()
        logger.info("Starting hierarchy validation")
        
        errors = []
        warnings = []
        info = []
        
        # Topology validation
        if self.config.enable_cycle_detection:
            cycle_errors = self._detect_cycles(hierarchy)
            errors.extend(cycle_errors)
        
        if self.config.enable_orphan_detection:
            orphan_errors = self._detect_orphans(hierarchy)
            warnings.extend(orphan_errors)
        
        # Structural validation
        completeness_errors = self._validate_completeness(hierarchy)
        errors.extend(completeness_errors)
        
        # Aggregation matrix validation
        if hierarchy.aggregation_matrix is not None:
            matrix_errors = self._validate_aggregation_matrix(hierarchy)
            errors.extend(matrix_errors)
        
        # Determine if validation passes
        is_valid = self._determine_validity(errors, warnings)
        
        validation_time = time.time() - start_time
        logger.info(f"Validation complete in {validation_time:.3f}s")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            validation_time=validation_time,
            validation_level=self.config.validation_level
        )
    
    def _detect_cycles(self, hierarchy: HierarchyDefinition) -> List[ValidationError]:
        """
        Detect cycles in hierarchy graph.
        
        Args:
            hierarchy: Hierarchy to check
        
        Returns:
            List of cycle errors
        """
        errors = []
        visited = set()
        rec_stack = set()
        
        def dfs(node_name: str, path: List[str]) -> Optional[List[str]]:
            """DFS to detect cycles."""
            visited.add(node_name)
            rec_stack.add(node_name)
            
            node = hierarchy.nodes.get(node_name)
            if node:
                for child_name in node.children:
                    if child_name not in visited:
                        cycle_path = dfs(child_name, path + [child_name])
                        if cycle_path:
                            return cycle_path
                    elif child_name in rec_stack:
                        return path + [child_name]
            
            rec_stack.remove(node_name)
            return None
        
        # Check from each node
        for node_name in hierarchy.nodes:
            if node_name not in visited:
                cycle_path = dfs(node_name, [node_name])
                if cycle_path:
                    errors.append(ValidationError(
                        error_type='CYCLE_DETECTED',
                        severity=ValidationSeverity.ERROR,
                        message=f"Cycle detected in hierarchy: {' -> '.join(cycle_path)}",
                        context={'cycle_path': cycle_path},
                        suggested_fix="Remove circular dependency from hierarchy definition"
                    ))
        
        return errors
    
    def _detect_orphans(self, hierarchy: HierarchyDefinition) -> List[ValidationError]:
        """
        Detect orphan nodes (no parent except root).
        
        Args:
            hierarchy: Hierarchy to check
        
        Returns:
            List of orphan warnings
        """
        warnings = []
        
        # Find root nodes (level 0)
        root_nodes = [name for name, node in hierarchy.nodes.items() 
                     if node.level == 0]
        
        # Check for nodes without parents (except roots)
        for node_name, node in hierarchy.nodes.items():
            if node.level > 0 and node.parent is None:
                warnings.append(ValidationError(
                    error_type='ORPHAN_NODE',
                    severity=ValidationSeverity.WARNING,
                    message=f"Node {node_name} has no parent",
                    context={'node': node_name, 'level': node.level},
                    suggested_fix="Assign parent node or move to root level"
                ))
        
        # Check for unreachable nodes
        reachable = set()
        def mark_reachable(node_name: str):
            reachable.add(node_name)
            node = hierarchy.nodes.get(node_name)
            if node:
                for child in node.children:
                    if child not in reachable:
                        mark_reachable(child)
        
        for root in root_nodes:
            mark_reachable(root)
        
        unreachable = set(hierarchy.nodes.keys()) - reachable
        for node_name in unreachable:
            warnings.append(ValidationError(
                error_type='UNREACHABLE_NODE',
                severity=ValidationSeverity.WARNING,
                message=f"Node {node_name} is unreachable from root",
                context={'node': node_name},
                suggested_fix="Connect node to hierarchy or remove"
            ))
        
        return warnings
    
    def _validate_completeness(
        self,
        hierarchy: HierarchyDefinition
    ) -> List[ValidationError]:
        """
        Validate hierarchy completeness.
        
        Args:
            hierarchy: Hierarchy to check
        
        Returns:
            List of completeness errors
        """
        errors = []
        
        for node_name, node in hierarchy.nodes.items():
            # Check required attributes
            if not node.name:
                errors.append(ValidationError(
                    error_type='MISSING_NAME',
                    severity=ValidationSeverity.ERROR,
                    message=f"Node {node_name} missing name attribute",
                    context={'node': node_name}
                ))
            
            # Check aggregation weights for aggregate nodes
            if node.node_type == 'aggregate' and node.children:
                total_weight = sum(node.aggregation_weights.get(child, 1.0) 
                                 for child in node.children)
                
                # Weights should sum to number of children (each defaults to 1.0)
                expected_sum = len(node.children)
                if abs(total_weight - expected_sum) > 1e-6:
                    errors.append(ValidationError(
                        error_type='INVALID_WEIGHTS',
                        severity=ValidationSeverity.ERROR,
                        message=f"Aggregation weights for {node_name} sum to {total_weight}, expected {expected_sum}",
                        context={'node': node_name, 'total_weight': total_weight},
                        suggested_fix="Adjust aggregation weights to sum correctly"
                    ))
        
        return errors
    
    def _validate_aggregation_matrix(
        self,
        hierarchy: HierarchyDefinition
    ) -> List[ValidationError]:
        """
        Validate aggregation matrix S.
        
        Args:
            hierarchy: Hierarchy with aggregation matrix
        
        Returns:
            List of matrix errors
        """
        errors = []
        S = hierarchy.aggregation_matrix
        
        # Check dimensions
        n_total = len(hierarchy.nodes)
        n_bottom = len(hierarchy.get_bottom_level_nodes())
        
        if S.shape[1] != n_bottom:
            errors.append(ValidationError(
                error_type='MATRIX_DIMENSION',
                severity=ValidationSeverity.ERROR,
                message=f"Aggregation matrix has {S.shape[1]} columns, expected {n_bottom}",
                context={'actual': S.shape[1], 'expected': n_bottom}
            ))
        
        # Check for zero columns (bottom series not used)
        zero_cols = np.where(S.sum(axis=0) == 0)[0]
        if len(zero_cols) > 0:
            errors.append(ValidationError(
                error_type='ZERO_COLUMN',
                severity=ValidationSeverity.WARNING,
                message=f"Aggregation matrix has {len(zero_cols)} zero columns",
                context={'zero_columns': zero_cols.tolist()},
                suggested_fix="Check that all bottom-level series are included"
            ))
        
        return errors
    
    def _determine_validity(
        self,
        errors: List[ValidationError],
        warnings: List[ValidationError]
    ) -> bool:
        """
        Determine if validation passes based on level.
        
        Args:
            errors: Critical errors
            warnings: Warnings
        
        Returns:
            True if validation passes
        """
        if self.config.validation_level == ValidationLevel.DIAGNOSTIC:
            return True
        
        if self.config.validation_level == ValidationLevel.LENIENT:
            # Only fail on ERROR severity
            return len(errors) == 0
        
        # STRICT: fail on any errors
        return len(errors) == 0


class ForecastValidator:
    """
    Validates forecast data quality and coherence.
    
    Checks:
        - Data availability and completeness
        - Numeric validity (no NaN/Inf)
        - Temporal consistency
        - Aggregation constraint satisfaction
    
    Example:
        >>> validator = ForecastValidator(tolerance=1e-6)
        >>> result = validator.validate_forecasts(forecasts, hierarchy)
        >>> print(f"Constraint violations: {len(result.errors)}")
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize forecast validator.
        
        Args:
            tolerance: Tolerance for constraint satisfaction
        """
        self.tolerance = tolerance
    
    def validate_forecasts(
        self,
        forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition
    ) -> ValidationResult:
        """
        Validate forecast data.
        
        Args:
            forecasts: Forecast dictionary
            hierarchy: Hierarchy definition
        
        Returns:
            ValidationResult
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        # Check availability
        missing_nodes = set(hierarchy.nodes.keys()) - set(forecasts.keys())
        if missing_nodes:
            warnings.append(ValidationError(
                error_type='MISSING_FORECASTS',
                severity=ValidationSeverity.WARNING,
                message=f"Missing forecasts for {len(missing_nodes)} nodes",
                context={'missing_nodes': list(missing_nodes)}
            ))
        
        # Check numeric validity
        for node_name, forecast in forecasts.items():
            if np.any(np.isnan(forecast)):
                errors.append(ValidationError(
                    error_type='NAN_VALUES',
                    severity=ValidationSeverity.ERROR,
                    message=f"NaN values in forecast for {node_name}",
                    context={'node': node_name}
                ))
            
            if np.any(np.isinf(forecast)):
                errors.append(ValidationError(
                    error_type='INF_VALUES',
                    severity=ValidationSeverity.ERROR,
                    message=f"Infinite values in forecast for {node_name}",
                    context={'node': node_name}
                ))
        
        # Check aggregation constraints
        constraint_errors = self._check_aggregation_constraints(forecasts, hierarchy)
        warnings.extend(constraint_errors)
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validation_time=validation_time
        )
    
    def _check_aggregation_constraints(
        self,
        forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition
    ) -> List[ValidationError]:
        """Check if aggregation constraints are satisfied."""
        warnings = []
        
        for node_name, node in hierarchy.nodes.items():
            if node.node_type != 'aggregate':
                continue
            
            if node_name not in forecasts:
                continue
            
            # Calculate sum of children
            expected = np.zeros_like(forecasts[node_name])
            for child_name in node.children:
                if child_name in forecasts:
                    weight = node.aggregation_weights.get(child_name, 1.0)
                    expected += weight * forecasts[child_name]
            
            actual = forecasts[node_name]
            abs_error = np.abs(actual - expected)
            
            if np.any(abs_error > self.tolerance):
                max_error = np.max(abs_error)
                warnings.append(ValidationError(
                    error_type='CONSTRAINT_VIOLATION',
                    severity=ValidationSeverity.WARNING,
                    message=f"Aggregation constraint violated for {node_name}, max error: {max_error:.2e}",
                    context={'node': node_name, 'max_error': float(max_error)},
                    suggested_fix="Apply hierarchical reconciliation"
                ))
        
        return warnings
```

---

## 🧪 Testing & Validation

```python
"""Tests for hierarchy validator."""
import pytest
import numpy as np

from src.models.reconciliation.validation.hierarchy_validator import (
    HierarchyValidator, ForecastValidator, ValidationLevel, ValidationConfig
)


def test_cycle_detection(hierarchy_with_cycle):
    """Test cycle detection."""
    validator = HierarchyValidator()
    result = validator.validate_hierarchy(hierarchy_with_cycle)
    
    assert not result.is_valid
    assert any('CYCLE' in e.error_type for e in result.errors)


def test_orphan_detection(hierarchy_with_orphan):
    """Test orphan node detection."""
    validator = HierarchyValidator()
    result = validator.validate_hierarchy(hierarchy_with_orphan)
    
    assert any('ORPHAN' in w.error_type for w in result.warnings)


def test_validation_levels(sample_hierarchy):
    """Test different validation levels."""
    # STRICT mode
    validator_strict = HierarchyValidator(
        config=ValidationConfig(validation_level=ValidationLevel.STRICT)
    )
    
    # DIAGNOSTIC mode
    validator_diag = HierarchyValidator(
        config=ValidationConfig(validation_level=ValidationLevel.DIAGNOSTIC)
    )
    
    # DIAGNOSTIC should always pass
    result_diag = validator_diag.validate_hierarchy(sample_hierarchy)
    assert result_diag.is_valid


def test_forecast_validation(sample_forecasts, sample_hierarchy):
    """Test forecast validation."""
    validator = ForecastValidator(tolerance=1e-6)
    result = validator.validate_forecasts(sample_forecasts, sample_hierarchy)
    
    assert isinstance(result, ValidationResult)
```

---

## 📝 Technical Notes

### Validation Strategy
- Multi-level validation (structure, topology, data)
- Configurable strictness levels
- Actionable error messages

### Performance
- Target overhead <10%
- Lazy validation where possible
- Cacheable validation results

---

## 🔗 Dependencies

**Depends On:**
- PC-048-06A: Hierarchy Definition System
- PC-049-06A: Base Reconciler Interface

**Blocks:**
- PC-052-06A: Constraint Enforcer
- All reconciliation operations

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Cycle detection working
- [ ] Orphan detection implemented
- [ ] Constraint checking functional
- [ ] Validation levels implemented
- [ ] Error reporting comprehensive
- [ ] Performance overhead <10%
- [ ] Unit tests >85% coverage
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**Previous:** [PC-050-06A: MinT Reconciler Implementation](PC-050-06A-mint-reconciler.md)  
**Next:** [PC-052-06A: Constraint Enforcement System](PC-052-06A-constraint-enforcer.md)
