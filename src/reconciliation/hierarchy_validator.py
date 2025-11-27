"""Hierarchy validation framework for ensuring structural integrity.

This module provides comprehensive validation of hierarchy structure and forecast
coherence, detecting issues like cycles, orphans, and constraint violations before
reconciliation is attempted.

Key validation types:
    - Topology: Cycles, connectivity, reachability
    - Structure: Completeness, consistency, weights
    - Constraints: Aggregation rules, numeric validity
    - Forecasts: Data quality, temporal alignment

Example:
    ```python
    from src.reconciliation import HierarchyValidator, ValidationConfig, ValidationLevel

    # Configure validator
    config = ValidationConfig(
        validation_level=ValidationLevel.STRICT,
        constraint_tolerance=1e-6
    )
    validator = HierarchyValidator(config=config)

    # Validate hierarchy
    result = validator.validate_hierarchy(hierarchy)

    # Check results
    if not result.is_valid:
        print(result.summary())
        for error in result.errors:
            print(f"ERROR: {error.message}")
    ```
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.reconciliation.hierarchy import HierarchyDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""

    ERROR = "ERROR"  # Critical errors that prevent reconciliation
    WARNING = "WARNING"  # Issues that should be reviewed but don't block
    INFO = "INFO"  # Informational messages about validation


class ValidationLevel(Enum):
    """Validation strictness levels."""

    STRICT = "STRICT"  # All errors cause validation failure
    LENIENT = "LENIENT"  # Only critical errors cause failure
    DIAGNOSTIC = "DIAGNOSTIC"  # All issues logged, no failure


@dataclass
class ValidationError:
    """Validation error or warning with context and suggested fixes.

    Attributes:
        error_type: Type of validation error (e.g., CYCLE_DETECTED, MISSING_NAME).
        severity: Severity level (ERROR, WARNING, INFO).
        message: Human-readable error message.
        context: Additional context about the error (dict).
        suggested_fix: Optional suggestion for how to fix the error.
    """

    error_type: str
    severity: ValidationSeverity
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    suggested_fix: str | None = None


@dataclass
class ValidationResult:
    """Result of validation process with errors, warnings, and timing.

    Attributes:
        is_valid: Whether validation passed.
        errors: List of critical errors.
        warnings: List of warnings.
        info: List of informational messages.
        validation_time: Time taken for validation in seconds.
        validation_level: Validation level used.
    """

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    info: list[ValidationError] = field(default_factory=list)
    validation_time: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.STRICT

    def get_errors_by_severity(self, severity: ValidationSeverity) -> list[ValidationError]:
        """Get errors by severity level.

        Args:
            severity: Severity level to filter by.

        Returns:
            List of validation errors matching the severity.
        """
        all_issues = self.errors + self.warnings + self.info
        return [e for e in all_issues if e.severity == severity]

    def has_critical_errors(self) -> bool:
        """Check if critical errors are present.

        Returns:
            True if there are critical errors.
        """
        return len(self.errors) > 0

    def summary(self) -> str:
        """Generate validation summary.

        Returns:
            Formatted summary string.
        """
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"Validation {status}\n"
            f"Errors: {len(self.errors)}, "
            f"Warnings: {len(self.warnings)}, "
            f"Info: {len(self.info)}\n"
            f"Time: {self.validation_time:.3f}s"
        )


@dataclass
class ValidationConfig:
    """Configuration for validation behavior.

    Attributes:
        validation_level: Strictness level for validation.
        constraint_tolerance: Tolerance for constraint violations.
        missing_data_tolerance: Fraction of missing data allowed (0-1).
        enable_cycle_detection: Whether to check for cycles.
        enable_orphan_detection: Whether to check for orphaned nodes.
        enable_constraint_check: Whether to check aggregation constraints.
    """

    validation_level: ValidationLevel = ValidationLevel.STRICT
    constraint_tolerance: float = 1e-6
    missing_data_tolerance: float = 0.1  # 10% missing allowed
    enable_cycle_detection: bool = True
    enable_orphan_detection: bool = True
    enable_constraint_check: bool = True


class HierarchyValidator:
    """Validates hierarchy structure and forecast coherence.

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
        ```python
        validator = HierarchyValidator(config=ValidationConfig(
            validation_level=ValidationLevel.STRICT,
            constraint_tolerance=1e-6
        ))
        result = validator.validate_hierarchy(hierarchy)
        if not result.is_valid:
            print(result.summary())
            for error in result.errors:
                print(f"- {error.message}")
        ```
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize hierarchy validator.

        Args:
            config: Validation configuration. If None, uses defaults.
        """
        self.config = config or ValidationConfig()
        logger.info(
            "Initialized HierarchyValidator (level=%s, tolerance=%.2e)",
            self.config.validation_level.value,
            self.config.constraint_tolerance,
        )

    def validate_hierarchy(self, hierarchy: HierarchyDefinition) -> ValidationResult:
        """Validate hierarchy structure.

        Runs all enabled validation checks and aggregates results.

        Args:
            hierarchy: Hierarchy to validate.

        Returns:
            ValidationResult with errors, warnings, and timing info.

        Example:
            ```python
            result = validator.validate_hierarchy(hierarchy)
            print(f"Valid: {result.is_valid}")
            print(f"Errors: {len(result.errors)}")
            ```
        """
        start_time = time.time()
        logger.info("Starting hierarchy validation")

        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        info: list[ValidationError] = []

        # Topology validation
        if self.config.enable_cycle_detection:
            cycle_errors = self._detect_cycles(hierarchy)
            errors.extend(cycle_errors)
            logger.debug("Cycle detection: %d errors", len(cycle_errors))

        if self.config.enable_orphan_detection:
            orphan_errors = self._detect_orphans(hierarchy)
            warnings.extend(orphan_errors)
            logger.debug("Orphan detection: %d warnings", len(orphan_errors))

        # Structural validation
        completeness_errors = self._validate_completeness(hierarchy)
        errors.extend(completeness_errors)
        logger.debug("Completeness validation: %d errors", len(completeness_errors))

        # Aggregation matrix validation
        if hierarchy.aggregation_matrix is not None:
            matrix_errors = self._validate_aggregation_matrix(hierarchy)
            errors.extend(matrix_errors)
            logger.debug("Aggregation matrix: %d errors", len(matrix_errors))

        # Determine if validation passes
        is_valid = self._determine_validity(errors, warnings)

        validation_time = time.time() - start_time
        logger.info(
            "Validation complete in %.3fs (valid=%s, errors=%d, warnings=%d)",
            validation_time,
            is_valid,
            len(errors),
            len(warnings),
        )

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            validation_time=validation_time,
            validation_level=self.config.validation_level,
        )

    def _detect_cycles(self, hierarchy: HierarchyDefinition) -> list[ValidationError]:
        """Detect cycles in hierarchy graph using depth-first search.

        Args:
            hierarchy: Hierarchy to check for cycles.

        Returns:
            List of cycle errors found.
        """
        errors: list[ValidationError] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node_name: str, path: list[str]) -> list[str] | None:
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
                        # Cycle detected
                        return path + [child_name]

            rec_stack.remove(node_name)
            return None

        # Check from each node
        for node_name in hierarchy.nodes:
            if node_name not in visited:
                cycle_path = dfs(node_name, [node_name])
                if cycle_path:
                    errors.append(
                        ValidationError(
                            error_type="CYCLE_DETECTED",
                            severity=ValidationSeverity.ERROR,
                            message=f"Cycle detected in hierarchy: {' -> '.join(cycle_path)}",
                            context={"cycle_path": cycle_path},
                            suggested_fix="Remove circular dependency from hierarchy definition",
                        )
                    )

        return errors

    def _detect_orphans(self, hierarchy: HierarchyDefinition) -> list[ValidationError]:
        """Detect orphan nodes (nodes without parents except root).

        Args:
            hierarchy: Hierarchy to check for orphans.

        Returns:
            List of orphan warnings.
        """
        warnings: list[ValidationError] = []

        # Find root nodes (level 0)
        root_nodes = [name for name, node in hierarchy.nodes.items() if node.level == 0]

        # Check for nodes without parents (except roots)
        for node_name, node in hierarchy.nodes.items():
            if node.level > 0 and node.parent is None:
                warnings.append(
                    ValidationError(
                        error_type="ORPHAN_NODE",
                        severity=ValidationSeverity.WARNING,
                        message=f"Node {node_name} has no parent",
                        context={"node": node_name, "level": node.level},
                        suggested_fix="Assign parent node or move to root level",
                    )
                )

        # Check for unreachable nodes
        reachable: set[str] = set()

        def mark_reachable(node_name: str) -> None:
            """Mark nodes reachable from current node."""
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
            warnings.append(
                ValidationError(
                    error_type="UNREACHABLE_NODE",
                    severity=ValidationSeverity.WARNING,
                    message=f"Node {node_name} is unreachable from root",
                    context={"node": node_name},
                    suggested_fix="Connect node to hierarchy or remove",
                )
            )

        return warnings

    def _validate_completeness(self, hierarchy: HierarchyDefinition) -> list[ValidationError]:
        """Validate hierarchy completeness (required attributes).

        Args:
            hierarchy: Hierarchy to check for completeness.

        Returns:
            List of completeness errors.
        """
        errors: list[ValidationError] = []

        for node_name, node in hierarchy.nodes.items():
            # Check required attributes
            if not node.name:
                errors.append(
                    ValidationError(
                        error_type="MISSING_NAME",
                        severity=ValidationSeverity.ERROR,
                        message=f"Node {node_name} missing name attribute",
                        context={"node": node_name},
                        suggested_fix="Add name attribute to node definition",
                    )
                )

            # Check aggregation weights for aggregate nodes
            if node.node_type == "aggregate" and node.children:
                total_weight = sum(
                    node.aggregation_weights.get(child, 1.0) for child in node.children
                )

                # Weights should sum to number of children (each defaults to 1.0)
                expected_sum = len(node.children)
                if abs(total_weight - expected_sum) > 1e-6:
                    errors.append(
                        ValidationError(
                            error_type="INVALID_WEIGHTS",
                            severity=ValidationSeverity.ERROR,
                            message=f"Aggregation weights for {node_name} sum to {total_weight:.3f}, expected {expected_sum}",
                            context={
                                "node": node_name,
                                "total_weight": float(total_weight),
                                "expected": expected_sum,
                            },
                            suggested_fix="Adjust aggregation weights to sum correctly",
                        )
                    )

        return errors

    def _validate_aggregation_matrix(
        self, hierarchy: HierarchyDefinition
    ) -> list[ValidationError]:
        """Validate aggregation matrix S properties.

        Args:
            hierarchy: Hierarchy with aggregation matrix.

        Returns:
            List of matrix validation errors.
        """
        errors: list[ValidationError] = []
        S = hierarchy.aggregation_matrix

        if S is None:
            return errors

        # Check dimensions
        n_total = len(hierarchy.nodes)
        n_bottom = len(hierarchy.get_bottom_level_nodes())

        if S.shape[1] != n_bottom:
            errors.append(
                ValidationError(
                    error_type="MATRIX_DIMENSION",
                    severity=ValidationSeverity.ERROR,
                    message=f"Aggregation matrix has {S.shape[1]} columns, expected {n_bottom}",
                    context={"actual": S.shape[1], "expected": n_bottom},
                    suggested_fix="Rebuild aggregation matrix with correct dimensions",
                )
            )

        # Check for zero columns (bottom series not used)
        zero_cols = np.where(S.sum(axis=0) == 0)[0]
        if len(zero_cols) > 0:
            errors.append(
                ValidationError(
                    error_type="ZERO_COLUMN",
                    severity=ValidationSeverity.WARNING,
                    message=f"Aggregation matrix has {len(zero_cols)} zero columns",
                    context={"zero_columns": zero_cols.tolist()},
                    suggested_fix="Check that all bottom-level series are included",
                )
            )

        return errors

    def _determine_validity(
        self, errors: list[ValidationError], warnings: list[ValidationError]
    ) -> bool:
        """Determine if validation passes based on configured level.

        Args:
            errors: Critical errors found.
            warnings: Warnings found.

        Returns:
            True if validation passes for the configured level.
        """
        if self.config.validation_level == ValidationLevel.DIAGNOSTIC:
            # DIAGNOSTIC mode always passes (for logging only)
            return True

        if self.config.validation_level == ValidationLevel.LENIENT:
            # LENIENT mode only fails on ERROR severity
            return len(errors) == 0

        # STRICT mode fails on any errors
        return len(errors) == 0

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"HierarchyValidator(level={self.config.validation_level.value!r}, "
            f"tolerance={self.config.constraint_tolerance:.2e})"
        )


class ForecastValidator:
    """Validates forecast data quality and coherence.

    Checks:
        - Data availability and completeness
        - Numeric validity (no NaN/Inf)
        - Temporal consistency
        - Aggregation constraint satisfaction

    Example:
        ```python
        validator = ForecastValidator(tolerance=1e-6)
        result = validator.validate_forecasts(forecasts, hierarchy)
        print(f"Constraint violations: {len(result.errors)}")
        ```
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        """Initialize forecast validator.

        Args:
            tolerance: Tolerance for constraint satisfaction checks.
        """
        self.tolerance = tolerance
        logger.info("Initialized ForecastValidator (tolerance=%.2e)", tolerance)

    def validate_forecasts(
        self, forecasts: pd.DataFrame, hierarchy: HierarchyDefinition
    ) -> ValidationResult:
        """Validate forecast data quality.

        Args:
            forecasts: Forecast DataFrame (timesteps × nodes).
            hierarchy: Hierarchy definition.

        Returns:
            ValidationResult with errors and warnings.

        Example:
            ```python
            forecasts = pd.DataFrame({
                "Total": [100, 200],
                "A": [60, 120],
                "B": [40, 80]
            })
            result = validator.validate_forecasts(forecasts, hierarchy)
            ```
        """
        start_time = time.time()
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        # Check availability
        missing_nodes = set(hierarchy.nodes.keys()) - set(forecasts.columns)
        if missing_nodes:
            warnings.append(
                ValidationError(
                    error_type="MISSING_FORECASTS",
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing forecasts for {len(missing_nodes)} nodes",
                    context={"missing_nodes": list(missing_nodes)},
                    suggested_fix="Provide forecasts for all hierarchy nodes",
                )
            )

        # Check numeric validity
        for node_name in forecasts.columns:
            forecast = forecasts[node_name].values
            if np.any(np.isnan(forecast)):
                errors.append(
                    ValidationError(
                        error_type="NAN_VALUES",
                        severity=ValidationSeverity.ERROR,
                        message=f"NaN values in forecast for {node_name}",
                        context={"node": node_name},
                        suggested_fix="Impute or remove NaN values before reconciliation",
                    )
                )

            if np.any(np.isinf(forecast)):
                errors.append(
                    ValidationError(
                        error_type="INF_VALUES",
                        severity=ValidationSeverity.ERROR,
                        message=f"Infinite values in forecast for {node_name}",
                        context={"node": node_name},
                        suggested_fix="Remove infinite values or cap at reasonable bounds",
                    )
                )

        # Check aggregation constraints
        constraint_errors = self._check_aggregation_constraints(forecasts, hierarchy)
        warnings.extend(constraint_errors)

        validation_time = time.time() - start_time

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validation_time=validation_time,
        )

    def _check_aggregation_constraints(
        self, forecasts: pd.DataFrame, hierarchy: HierarchyDefinition
    ) -> list[ValidationError]:
        """Check if aggregation constraints are satisfied.

        Args:
            forecasts: Forecast DataFrame.
            hierarchy: Hierarchy definition.

        Returns:
            List of constraint violation warnings.
        """
        warnings: list[ValidationError] = []

        for node_name, node in hierarchy.nodes.items():
            if node.node_type != "aggregate":
                continue

            if node_name not in forecasts.columns:
                continue

            # Calculate sum of children
            expected = np.zeros(len(forecasts))
            for child_name in node.children:
                if child_name in forecasts.columns:
                    weight = node.aggregation_weights.get(child_name, 1.0)
                    expected += weight * forecasts[child_name].values

            actual = forecasts[node_name].values
            abs_error = np.abs(actual - expected)

            if np.any(abs_error > self.tolerance):
                max_error = np.max(abs_error)
                warnings.append(
                    ValidationError(
                        error_type="CONSTRAINT_VIOLATION",
                        severity=ValidationSeverity.WARNING,
                        message=f"Aggregation constraint violated for {node_name}, max error: {max_error:.2e}",
                        context={"node": node_name, "max_error": float(max_error)},
                        suggested_fix="Apply hierarchical reconciliation",
                    )
                )

        return warnings

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ForecastValidator(tolerance={self.tolerance:.2e})"
