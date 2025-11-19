# PC-049-06A: Base Reconciliation Interface

**Ticket ID:** PC-049-06A  
**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**User Story:** US-2  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `BaseReconciler` abstract base class that defines the standardized interface for all hierarchical reconciliation methods. Provides contract for reconciliation operations, supports point forecasts and prediction intervals, handles missing predictions, and includes comprehensive diagnostics and metadata generation.

**As a** ML engineer  
**I want** a standardized interface for hierarchical reconciliation methods  
**So that** I can easily integrate multiple reconciliation approaches

---

## ✅ Acceptance Criteria

- [ ] `BaseReconciler` interface defines reconciliation contract
- [ ] Supports both point forecasts and prediction intervals
- [ ] Handles missing predictions gracefully
- [ ] Provides reconciliation diagnostics and metadata
- [ ] Extensible design for new reconciliation methods
- [ ] Abstract `reconcile()` method
- [ ] Input validation framework
- [ ] Error handling and logging
- [ ] Performance monitoring
- [ ] ReconciliationResult dataclass

---

## 🔧 Implementation Tasks

### 1. Create Reconciliation Module Structure
- [ ] Create `src/models/reconciliation/__init__.py`
- [ ] Create `src/models/reconciliation/base_reconciler.py`
- [ ] Import ABC and abstractmethod
- [ ] Import dataclasses for results
- [ ] Add module docstrings

### 2. Implement ReconciliationResult Dataclass
- [ ] Create `ReconciliationResult` dataclass
- [ ] Add reconciled_forecasts attribute (Dict or DataFrame)
- [ ] Add original_forecasts attribute
- [ ] Add reconciliation_metadata attribute
- [ ] Add constraint_violations attribute
- [ ] Add processing_time attribute

### 3. Implement ReconciliationMetadata Dataclass
- [ ] Create `ReconciliationMetadata` dataclass
- [ ] Add timestamp attribute
- [ ] Add reconciliation_method attribute
- [ ] Add hierarchy_structure attribute
- [ ] Add covariance_method attribute
- [ ] Add performance_metrics dict

### 4. Implement BaseReconciler Abstract Class
- [ ] Create `BaseReconciler` abstract class
- [ ] Define `__init__()` with config parameter
- [ ] Add fitted flag and timestamp
- [ ] Add hierarchy reference
- [ ] Add configuration storage

### 5. Implement Abstract reconcile() Method
- [ ] Define `@abstractmethod reconcile()`
- [ ] Parameters: base_forecasts, hierarchy, covariance_matrix
- [ ] Return type: ReconciliationResult
- [ ] Add comprehensive docstring
- [ ] Define expected behavior

### 6. Implement Forecast Validation
- [ ] Create `_validate_forecasts()` method
- [ ] Check forecast dictionary structure
- [ ] Validate array shapes and dimensions
- [ ] Check for NaN and infinite values
- [ ] Verify alignment with hierarchy nodes
- [ ] Return validation status

### 7. Implement Missing Data Handling
- [ ] Create `_handle_missing_forecasts()` method
- [ ] Identify missing forecast series
- [ ] Strategy: exclude or impute
- [ ] Log missing data warnings
- [ ] Update hierarchy for reconciliation
- [ ] Return cleaned forecasts dict

### 8. Implement Metadata Generation
- [ ] Create `_create_metadata()` method
- [ ] Capture reconciliation timestamp
- [ ] Store reconciliation method name
- [ ] Record hierarchy structure used
- [ ] Calculate processing time
- [ ] Return ReconciliationMetadata

### 9. Implement Constraint Validation
- [ ] Create `_validate_constraints()` method
- [ ] Check aggregation constraints satisfied
- [ ] Calculate constraint violations
- [ ] Compute violation metrics (RMSE, max abs error)
- [ ] Return constraint violation report

### 10. Implement Performance Monitoring
- [ ] Create `_monitor_performance()` method
- [ ] Track reconciliation time
- [ ] Monitor memory usage
- [ ] Log performance metrics
- [ ] Store in metadata

### 11. Implement Error Handling
- [ ] Create `ReconciliationError` exception class
- [ ] Handle matrix singularity errors
- [ ] Handle dimension mismatch errors
- [ ] Handle missing data errors
- [ ] Provide informative error messages

### 12. Implement Diagnostic Output
- [ ] Create `get_diagnostics()` method
- [ ] Return reconciliation statistics
- [ ] Calculate forecast coherence metrics
- [ ] Measure constraint satisfaction
- [ ] Return diagnostic dict

### 13. Implement Forecast Format Conversion
- [ ] Create `_convert_to_array()` helper
- [ ] Convert DataFrames to arrays
- [ ] Handle dict of arrays
- [ ] Preserve node ordering
- [ ] Return numpy array

### 14. Implement Result Formatting
- [ ] Create `_format_result()` method
- [ ] Convert reconciled arrays to original format
- [ ] Restore DataFrame structure if needed
- [ ] Add metadata and diagnostics
- [ ] Return ReconciliationResult

### 15. Implement is_fitted Property
- [ ] Create `@property is_fitted`
- [ ] Check if reconciler has been fitted
- [ ] Used for fit-then-predict pattern
- [ ] Return boolean

### 16. Handle Edge Cases
- [ ] Handle empty forecast dict
- [ ] Handle single-series hierarchy
- [ ] Handle all-NaN forecasts
- [ ] Handle incompatible dimensions
- [ ] Validate before reconciliation

### 17. Write Comprehensive Tests
- [ ] Create `tests/models/reconciliation/test_base_reconciler.py`
- [ ] Test abstract interface enforcement
- [ ] Test validation methods
- [ ] Test error handling
- [ ] Test metadata generation
- [ ] Create mock reconciler for testing

### 18. Create Usage Documentation
- [ ] Create `docs/reconciliation_interface.md`
- [ ] Document BaseReconciler contract
- [ ] Show implementation example
- [ ] Document expected inputs/outputs
- [ ] Provide best practices

---

## 💻 Implementation Details

### BaseReconciler Implementation

```python
"""Base interface for hierarchical reconciliation methods."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import time

from src.models.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReconciliationMetadata:
    """Metadata for reconciliation results."""
    
    timestamp: pd.Timestamp
    reconciliation_method: str
    hierarchy_structure: str
    covariance_method: Optional[str] = None
    processing_time_seconds: float = 0.0
    n_series: int = 0
    n_bottom_series: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintViolation:
    """Information about constraint violations."""
    
    node_name: str
    expected_value: float
    actual_value: float
    absolute_error: float
    relative_error: float
    constraint_type: str  # 'aggregation', 'energy_balance'


@dataclass
class ReconciliationResult:
    """
    Result of hierarchical reconciliation.
    
    Attributes:
        reconciled_forecasts: Reconciled forecast values (Dict or DataFrame)
        original_forecasts: Original base forecasts before reconciliation
        metadata: Reconciliation metadata and diagnostics
        constraint_violations: List of any constraint violations
        coherence_improved: Whether forecast coherence improved
    """
    
    reconciled_forecasts: Dict[str, np.ndarray]
    original_forecasts: Dict[str, np.ndarray]
    metadata: ReconciliationMetadata
    constraint_violations: List[ConstraintViolation] = field(default_factory=list)
    coherence_improved: bool = True


class ReconciliationError(Exception):
    """Exception raised during reconciliation process."""
    pass


class BaseReconciler(ABC):
    """
    Base interface for hierarchical forecast reconciliation.
    
    All reconciliation methods must inherit from this class and implement
    the abstract `reconcile()` method.
    
    Reconciliation ensures forecast coherence:
        - Subsystem = Σ(Areas within subsystem)
        - National = Σ(All subsystems)
        - Energy balance maintained
    
    Standard Workflow:
        1. Validate input forecasts
        2. Handle missing data
        3. Apply reconciliation algorithm
        4. Validate constraints
        5. Return ReconciliationResult
    
    Example:
        >>> class MyReconciler(BaseReconciler):
        ...     def reconcile(self, base_forecasts, hierarchy, covariance_matrix=None):
        ...         # Implementation here
        ...         pass
        >>> 
        >>> reconciler = MyReconciler(config={'method': 'custom'})
        >>> result = reconciler.reconcile(forecasts, hierarchy)
        >>> print(f"Reconciled {result.metadata.n_series} series")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base reconciler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._fitted = False
        self._fit_timestamp: Optional[pd.Timestamp] = None
        self.hierarchy: Optional[HierarchyDefinition] = None
    
    @property
    def name(self) -> str:
        """Return reconciler name."""
        return self.__class__.__name__
    
    @property
    def is_fitted(self) -> bool:
        """Check if reconciler has been fitted."""
        return self._fitted
    
    @abstractmethod
    def reconcile(
        self,
        base_forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition,
        covariance_matrix: Optional[np.ndarray] = None
    ) -> ReconciliationResult:
        """
        Reconcile base forecasts to satisfy hierarchical constraints.
        
        Args:
            base_forecasts: Dictionary mapping node names to forecast arrays
            hierarchy: Hierarchy definition with structure and relationships
            covariance_matrix: Optional forecast error covariance matrix
        
        Returns:
            ReconciliationResult with reconciled forecasts and metadata
        
        Raises:
            ReconciliationError: If reconciliation fails
        """
        pass
    
    def _validate_forecasts(
        self,
        forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition
    ) -> List[str]:
        """
        Validate forecast inputs.
        
        Args:
            forecasts: Forecast dictionary
            hierarchy: Hierarchy definition
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check if forecasts is empty
        if not forecasts:
            errors.append("Forecasts dictionary is empty")
            return errors
        
        # Check for missing required nodes
        hierarchy_nodes = set(hierarchy.nodes.keys())
        forecast_nodes = set(forecasts.keys())
        
        missing_nodes = hierarchy_nodes - forecast_nodes
        if missing_nodes:
            logger.warning(f"Missing forecasts for nodes: {missing_nodes}")
        
        # Validate array shapes
        first_shape = None
        for node_name, forecast_array in forecasts.items():
            if not isinstance(forecast_array, np.ndarray):
                errors.append(f"Forecast for {node_name} is not numpy array")
                continue
            
            if first_shape is None:
                first_shape = forecast_array.shape
            elif forecast_array.shape != first_shape:
                errors.append(
                    f"Shape mismatch for {node_name}: "
                    f"expected {first_shape}, got {forecast_array.shape}"
                )
            
            # Check for NaN or infinite values
            if np.any(np.isnan(forecast_array)):
                errors.append(f"NaN values found in {node_name}")
            if np.any(np.isinf(forecast_array)):
                errors.append(f"Infinite values found in {node_name}")
        
        return errors
    
    def _handle_missing_forecasts(
        self,
        forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition
    ) -> Dict[str, np.ndarray]:
        """
        Handle missing forecasts.
        
        Args:
            forecasts: Forecast dictionary
            hierarchy: Hierarchy definition
        
        Returns:
            Cleaned forecast dictionary
        """
        # For now, just return forecasts as-is
        # Future: implement imputation strategies
        
        missing_count = len(hierarchy.nodes) - len(forecasts)
        if missing_count > 0:
            logger.warning(f"Missing {missing_count} forecast series")
        
        return forecasts
    
    def _create_metadata(
        self,
        method_name: str,
        hierarchy: HierarchyDefinition,
        processing_time: float,
        **kwargs
    ) -> ReconciliationMetadata:
        """
        Create reconciliation metadata.
        
        Args:
            method_name: Name of reconciliation method
            hierarchy: Hierarchy used
            processing_time: Time taken for reconciliation
            **kwargs: Additional metadata fields
        
        Returns:
            ReconciliationMetadata instance
        """
        return ReconciliationMetadata(
            timestamp=pd.Timestamp.now(),
            reconciliation_method=method_name,
            hierarchy_structure=f"{len(hierarchy.nodes)} nodes",
            processing_time_seconds=processing_time,
            n_series=len(hierarchy.nodes),
            n_bottom_series=len(hierarchy.get_bottom_level_nodes()),
            configuration=self.config.copy(),
            **kwargs
        )
    
    def _validate_constraints(
        self,
        reconciled_forecasts: Dict[str, np.ndarray],
        hierarchy: HierarchyDefinition,
        tolerance: float = 1e-6
    ) -> List[ConstraintViolation]:
        """
        Validate aggregation constraints.
        
        Args:
            reconciled_forecasts: Reconciled forecasts
            hierarchy: Hierarchy definition
            tolerance: Tolerance for constraint satisfaction
        
        Returns:
            List of constraint violations
        """
        violations = []
        
        # Check aggregation constraints
        for node_name, node in hierarchy.nodes.items():
            if node.node_type != 'aggregate':
                continue
            
            if node_name not in reconciled_forecasts:
                continue
            
            # Calculate expected value (sum of children)
            expected = np.zeros_like(reconciled_forecasts[node_name])
            for child_name in node.children:
                if child_name in reconciled_forecasts:
                    weight = node.aggregation_weights.get(child_name, 1.0)
                    expected += weight * reconciled_forecasts[child_name]
            
            actual = reconciled_forecasts[node_name]
            abs_error = np.abs(actual - expected)
            
            # Check each time point
            for t in range(len(actual)):
                if abs_error[t] > tolerance:
                    rel_error = abs_error[t] / (abs(expected[t]) + 1e-10)
                    
                    violation = ConstraintViolation(
                        node_name=node_name,
                        expected_value=float(expected[t]),
                        actual_value=float(actual[t]),
                        absolute_error=float(abs_error[t]),
                        relative_error=float(rel_error),
                        constraint_type='aggregation'
                    )
                    violations.append(violation)
        
        if violations:
            logger.warning(f"Found {len(violations)} constraint violations")
        
        return violations
    
    def get_diagnostics(
        self,
        result: ReconciliationResult
    ) -> Dict[str, Any]:
        """
        Get diagnostic information about reconciliation.
        
        Args:
            result: Reconciliation result
        
        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            'method': result.metadata.reconciliation_method,
            'processing_time': result.metadata.processing_time_seconds,
            'n_series': result.metadata.n_series,
            'n_violations': len(result.constraint_violations),
            'coherence_improved': result.coherence_improved
        }
        
        if result.constraint_violations:
            diagnostics['max_violation'] = max(
                v.absolute_error for v in result.constraint_violations
            )
            diagnostics['mean_violation'] = np.mean([
                v.absolute_error for v in result.constraint_violations
            ])
        
        return diagnostics
```

---

## 🧪 Testing & Validation

```python
"""Tests for base reconciler interface."""
import pytest
import numpy as np

from src.models.reconciliation.base_reconciler import (
    BaseReconciler, ReconciliationResult, ReconciliationError
)
from src.models.reconciliation.hierarchy import HierarchyDefinition


class MockReconciler(BaseReconciler):
    """Mock reconciler for testing."""
    
    def reconcile(self, base_forecasts, hierarchy, covariance_matrix=None):
        """Simply return original forecasts."""
        import time
        start_time = time.time()
        
        # Simulate reconciliation
        reconciled = base_forecasts.copy()
        
        processing_time = time.time() - start_time
        metadata = self._create_metadata(
            method_name='mock',
            hierarchy=hierarchy,
            processing_time=processing_time
        )
        
        return ReconciliationResult(
            reconciled_forecasts=reconciled,
            original_forecasts=base_forecasts,
            metadata=metadata
        )


def test_abstract_interface():
    """Test that BaseReconciler is abstract."""
    with pytest.raises(TypeError):
        BaseReconciler()


def test_mock_reconciler(sample_hierarchy_yaml):
    """Test mock reconciler implementation."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    forecasts = {
        'BR_National': np.array([1000, 1100, 1200]),
        'SE_Subsystem': np.array([600, 650, 700]),
        'Area_1': np.array([300, 325, 350])
    }
    
    reconciler = MockReconciler()
    result = reconciler.reconcile(forecasts, hierarchy)
    
    assert isinstance(result, ReconciliationResult)
    assert len(result.reconciled_forecasts) == 3


def test_forecast_validation(sample_hierarchy_yaml):
    """Test forecast validation."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    reconciler = MockReconciler()
    
    # Valid forecasts
    forecasts = {
        'Area_1': np.array([100, 200, 300])
    }
    errors = reconciler._validate_forecasts(forecasts, hierarchy)
    assert len(errors) == 0
    
    # Invalid forecasts (NaN)
    forecasts_nan = {
        'Area_1': np.array([100, np.nan, 300])
    }
    errors = reconciler._validate_forecasts(forecasts_nan, hierarchy)
    assert len(errors) > 0


def test_metadata_generation(sample_hierarchy_yaml):
    """Test metadata creation."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    reconciler = MockReconciler()
    
    metadata = reconciler._create_metadata(
        method_name='test',
        hierarchy=hierarchy,
        processing_time=1.5
    )
    
    assert metadata.reconciliation_method == 'test'
    assert metadata.processing_time_seconds == 1.5
    assert metadata.n_series > 0
```

---

## 📝 Technical Notes

### Interface Design
- Abstract base class enforces contract
- Extensible for new methods
- Standardized input/output

### Error Handling
- Comprehensive validation
- Informative error messages
- Graceful degradation

---

## 🔗 Dependencies

**Depends On:**
- PC-048-06A: Hierarchy Definition System

**Blocks:**
- PC-050-06A: MinT Reconciler Implementation
- All reconciliation methods

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] BaseReconciler abstract class implemented
- [ ] ReconciliationResult dataclass complete
- [ ] Validation methods working
- [ ] Error handling comprehensive
- [ ] Metadata generation functional
- [ ] Abstract interface enforced
- [ ] Unit tests >85% coverage
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**Previous:** [PC-048-06A: Hierarchy Definition System](PC-048-06A-hierarchy-definition.md)  
**Next:** [PC-050-06A: MinT Reconciler Implementation](PC-050-06A-mint-reconciler.md)
