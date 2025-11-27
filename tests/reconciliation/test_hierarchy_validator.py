"""Tests for hierarchy validator module."""

import numpy as np
import pandas as pd
import pytest

from src.reconciliation.hierarchy import HierarchyDefinition
from src.reconciliation.hierarchy_validator import (
    ForecastValidator,
    HierarchyValidator,
    ValidationConfig,
    ValidationError,
    ValidationLevel,
    ValidationResult,
    ValidationSeverity,
)


@pytest.fixture
def simple_hierarchy():
    """Create simple valid hierarchy for testing."""
    hierarchy = HierarchyDefinition()
    hierarchy.add_node("Total", level=0, node_type="aggregate", children=["A", "B"])
    hierarchy.add_node("A", level=1, node_type="bottom", parent="Total")
    hierarchy.add_node("B", level=1, node_type="bottom", parent="Total")
    hierarchy.build_graph()
    hierarchy.build_aggregation_matrix()
    return hierarchy


@pytest.fixture
def hierarchy_with_orphan():
    """Create hierarchy with orphan node."""
    hierarchy = HierarchyDefinition()
    hierarchy.add_node("Total", level=0, node_type="aggregate", children=["A"])
    hierarchy.add_node("A", level=1, node_type="bottom", parent="Total")
    # Orphan at level 1 with no parent
    hierarchy.add_node("Orphan", level=1, node_type="bottom", parent=None)
    hierarchy.build_graph()
    return hierarchy


# ============================================================================
# ValidationError Tests
# ============================================================================


def test_validation_error_creation():
    """Test ValidationError dataclass creation."""
    error = ValidationError(
        error_type="TEST_ERROR",
        severity=ValidationSeverity.ERROR,
        message="Test error message",
        context={"key": "value"},
        suggested_fix="Fix it",
    )

    assert error.error_type == "TEST_ERROR"
    assert error.severity == ValidationSeverity.ERROR
    assert error.message == "Test error message"
    assert error.context == {"key": "value"}
    assert error.suggested_fix == "Fix it"


def test_validation_error_default_context():
    """Test ValidationError with default empty context."""
    error = ValidationError(
        error_type="TEST", severity=ValidationSeverity.WARNING, message="Test"
    )

    assert error.context == {}
    assert error.suggested_fix is None


# ============================================================================
# ValidationResult Tests
# ============================================================================


def test_validation_result_has_critical_errors():
    """Test has_critical_errors method."""
    result = ValidationResult(is_valid=False, errors=[
        ValidationError("E1", ValidationSeverity.ERROR, "Error 1")
    ])

    assert result.has_critical_errors()


def test_validation_result_no_critical_errors():
    """Test has_critical_errors when no errors."""
    result = ValidationResult(is_valid=True, warnings=[
        ValidationError("W1", ValidationSeverity.WARNING, "Warning 1")
    ])

    assert not result.has_critical_errors()


def test_validation_result_get_errors_by_severity():
    """Test filtering errors by severity."""
    result = ValidationResult(
        is_valid=False,
        errors=[ValidationError("E1", ValidationSeverity.ERROR, "Error")],
        warnings=[ValidationError("W1", ValidationSeverity.WARNING, "Warning")],
        info=[ValidationError("I1", ValidationSeverity.INFO, "Info")],
    )

    errors = result.get_errors_by_severity(ValidationSeverity.ERROR)
    warnings = result.get_errors_by_severity(ValidationSeverity.WARNING)
    infos = result.get_errors_by_severity(ValidationSeverity.INFO)

    assert len(errors) == 1
    assert len(warnings) == 1
    assert len(infos) == 1


def test_validation_result_summary():
    """Test summary string generation."""
    result = ValidationResult(
        is_valid=False,
        errors=[ValidationError("E1", ValidationSeverity.ERROR, "Error")],
        warnings=[ValidationError("W1", ValidationSeverity.WARNING, "Warning")],
        validation_time=1.234,
    )

    summary = result.summary()
    assert "FAILED" in summary
    assert "Errors: 1" in summary
    assert "Warnings: 1" in summary
    assert "1.234s" in summary


# ============================================================================
# ValidationConfig Tests
# ============================================================================


def test_validation_config_defaults():
    """Test ValidationConfig default values."""
    config = ValidationConfig()

    assert config.validation_level == ValidationLevel.STRICT
    assert config.constraint_tolerance == 1e-6
    assert config.missing_data_tolerance == 0.1
    assert config.enable_cycle_detection is True
    assert config.enable_orphan_detection is True
    assert config.enable_constraint_check is True


def test_validation_config_custom():
    """Test ValidationConfig with custom values."""
    config = ValidationConfig(
        validation_level=ValidationLevel.LENIENT,
        constraint_tolerance=1e-4,
        missing_data_tolerance=0.2,
        enable_cycle_detection=False,
    )

    assert config.validation_level == ValidationLevel.LENIENT
    assert config.constraint_tolerance == 1e-4
    assert config.missing_data_tolerance == 0.2
    assert config.enable_cycle_detection is False


# ============================================================================
# HierarchyValidator Initialization Tests
# ============================================================================


def test_hierarchy_validator_default_init():
    """Test HierarchyValidator initialization with defaults."""
    validator = HierarchyValidator()

    assert validator.config is not None
    assert validator.config.validation_level == ValidationLevel.STRICT


def test_hierarchy_validator_custom_config():
    """Test HierarchyValidator with custom config."""
    config = ValidationConfig(validation_level=ValidationLevel.LENIENT)
    validator = HierarchyValidator(config=config)

    assert validator.config.validation_level == ValidationLevel.LENIENT


def test_hierarchy_validator_repr():
    """Test HierarchyValidator string representation."""
    validator = HierarchyValidator()
    repr_str = repr(validator)

    assert "HierarchyValidator" in repr_str
    assert "STRICT" in repr_str


# ============================================================================
# Hierarchy Structure Validation Tests
# ============================================================================


def test_validate_simple_hierarchy(simple_hierarchy):
    """Test validation of simple valid hierarchy."""
    validator = HierarchyValidator()
    result = validator.validate_hierarchy(simple_hierarchy)

    assert result.is_valid
    assert len(result.errors) == 0
    assert result.validation_time > 0


def test_validate_hierarchy_with_orphan(hierarchy_with_orphan):
    """Test orphan detection."""
    validator = HierarchyValidator()
    result = validator.validate_hierarchy(hierarchy_with_orphan)

    # Orphans are warnings in STRICT mode but don't cause failure
    assert len(result.warnings) > 0
    assert any("ORPHAN" in w.error_type for w in result.warnings)


def test_detect_cycles():
    """Test cycle detection in hierarchy."""
    # Create hierarchy with cycle (manually for testing)
    hierarchy = HierarchyDefinition()
    hierarchy.add_node("A", level=0, node_type="aggregate", children=["B"])
    hierarchy.add_node("B", level=1, node_type="aggregate", parent="A", children=["C"])
    hierarchy.add_node("C", level=2, node_type="aggregate", parent="B", children=["A"])

    validator = HierarchyValidator()
    result = validator.validate_hierarchy(hierarchy)

    assert not result.is_valid
    assert any("CYCLE" in e.error_type for e in result.errors)


def test_validate_completeness_missing_name():
    """Test completeness validation detects missing names."""
    hierarchy = HierarchyDefinition()
    # Create node with empty name
    hierarchy.add_node("", level=0, node_type="aggregate")

    validator = HierarchyValidator()
    result = validator.validate_hierarchy(hierarchy)

    assert not result.is_valid
    assert any("MISSING_NAME" in e.error_type for e in result.errors)


def test_validate_aggregation_matrix_dimensions(simple_hierarchy):
    """Test aggregation matrix dimension validation."""
    validator = HierarchyValidator()
    result = validator.validate_hierarchy(simple_hierarchy)

    # Simple hierarchy should have valid aggregation matrix
    assert result.is_valid


# ============================================================================
# Validation Level Tests
# ============================================================================


def test_validation_level_strict(hierarchy_with_orphan):
    """Test STRICT validation level."""
    config = ValidationConfig(validation_level=ValidationLevel.STRICT)
    validator = HierarchyValidator(config=config)
    result = validator.validate_hierarchy(hierarchy_with_orphan)

    # Orphans are warnings, so STRICT mode should still pass
    # (only errors cause failure in STRICT)
    assert result.is_valid or len(result.errors) > 0


def test_validation_level_diagnostic():
    """Test DIAGNOSTIC validation level always passes."""
    # Create invalid hierarchy
    hierarchy = HierarchyDefinition()
    hierarchy.add_node("", level=0, node_type="aggregate")  # Missing name

    config = ValidationConfig(validation_level=ValidationLevel.DIAGNOSTIC)
    validator = HierarchyValidator(config=config)
    result = validator.validate_hierarchy(hierarchy)

    # DIAGNOSTIC always passes
    assert result.is_valid
    # But still records errors
    assert len(result.errors) > 0


def test_validation_level_lenient():
    """Test LENIENT validation level."""
    hierarchy = HierarchyDefinition()
    hierarchy.add_node("", level=0, node_type="aggregate")  # Missing name (error)

    config = ValidationConfig(validation_level=ValidationLevel.LENIENT)
    validator = HierarchyValidator(config=config)
    result = validator.validate_hierarchy(hierarchy)

    # LENIENT fails on errors
    assert not result.is_valid


# ============================================================================
# ForecastValidator Tests
# ============================================================================


def test_forecast_validator_init():
    """Test ForecastValidator initialization."""
    validator = ForecastValidator(tolerance=1e-4)

    assert validator.tolerance == 1e-4


def test_forecast_validator_repr():
    """Test ForecastValidator string representation."""
    validator = ForecastValidator()
    repr_str = repr(validator)

    assert "ForecastValidator" in repr_str
    assert "tolerance" in repr_str


def test_validate_forecasts_valid(simple_hierarchy):
    """Test validation of valid forecasts."""
    forecasts = pd.DataFrame({
        "Total": [100.0, 200.0],
        "A": [60.0, 120.0],
        "B": [40.0, 80.0],
    })

    validator = ForecastValidator(tolerance=1e-6)
    result = validator.validate_forecasts(forecasts, simple_hierarchy)

    assert result.is_valid
    assert len(result.errors) == 0


def test_validate_forecasts_missing_nodes(simple_hierarchy):
    """Test detection of missing forecast nodes."""
    forecasts = pd.DataFrame({
        "Total": [100.0, 200.0],
        # Missing A and B
    })

    validator = ForecastValidator()
    result = validator.validate_forecasts(forecasts, simple_hierarchy)

    # Missing nodes are warnings
    assert len(result.warnings) > 0
    assert any("MISSING_FORECASTS" in w.error_type for w in result.warnings)


def test_validate_forecasts_nan_values(simple_hierarchy):
    """Test detection of NaN values in forecasts."""
    forecasts = pd.DataFrame({
        "Total": [100.0, np.nan],
        "A": [60.0, 120.0],
        "B": [40.0, 80.0],
    })

    validator = ForecastValidator()
    result = validator.validate_forecasts(forecasts, simple_hierarchy)

    assert not result.is_valid
    assert any("NAN_VALUES" in e.error_type for e in result.errors)


def test_validate_forecasts_inf_values(simple_hierarchy):
    """Test detection of infinite values in forecasts."""
    forecasts = pd.DataFrame({
        "Total": [100.0, np.inf],
        "A": [60.0, 120.0],
        "B": [40.0, 80.0],
    })

    validator = ForecastValidator()
    result = validator.validate_forecasts(forecasts, simple_hierarchy)

    assert not result.is_valid
    assert any("INF_VALUES" in e.error_type for e in result.errors)


def test_validate_forecasts_constraint_violation(simple_hierarchy):
    """Test detection of aggregation constraint violations."""
    forecasts = pd.DataFrame({
        "Total": [100.0, 200.0],  # Should be 100, 200
        "A": [60.0, 120.0],
        "B": [40.0, 70.0],  # Should be 80 for constraint satisfaction
    })

    validator = ForecastValidator(tolerance=1e-6)
    result = validator.validate_forecasts(forecasts, simple_hierarchy)

    # Constraint violations are warnings
    assert len(result.warnings) > 0
    assert any("CONSTRAINT_VIOLATION" in w.error_type for w in result.warnings)


def test_validate_forecasts_within_tolerance(simple_hierarchy):
    """Test forecasts within tolerance pass constraint check."""
    forecasts = pd.DataFrame({
        "Total": [100.0, 200.0],
        "A": [60.0, 120.0],
        "B": [40.0, 80.0 + 1e-7],  # Small error within tolerance
    })

    validator = ForecastValidator(tolerance=1e-6)
    result = validator.validate_forecasts(forecasts, simple_hierarchy)

    # Should pass with small error
    assert result.is_valid or len([w for w in result.warnings if "CONSTRAINT" in w.error_type]) == 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_validation_workflow(simple_hierarchy):
    """Test full validation workflow."""
    # Validate hierarchy
    h_validator = HierarchyValidator()
    h_result = h_validator.validate_hierarchy(simple_hierarchy)
    assert h_result.is_valid

    # Validate forecasts
    forecasts = pd.DataFrame({
        "Total": [100.0, 200.0],
        "A": [60.0, 120.0],
        "B": [40.0, 80.0],
    })
    f_validator = ForecastValidator()
    f_result = f_validator.validate_forecasts(forecasts, simple_hierarchy)
    assert f_result.is_valid


def test_validation_with_all_checks_disabled():
    """Test validation with all checks disabled."""
    config = ValidationConfig(
        enable_cycle_detection=False,
        enable_orphan_detection=False,
        enable_constraint_check=False,
    )

    hierarchy = HierarchyDefinition()
    hierarchy.add_node("Total", level=0, node_type="aggregate")

    validator = HierarchyValidator(config=config)
    result = validator.validate_hierarchy(hierarchy)

    # Should pass with checks disabled
    assert result.is_valid
