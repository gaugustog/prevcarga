"""Tests for constraint enforcer."""
import numpy as np
import pytest

from src.reconciliation.constraints.constraint_enforcer import (
    AggregationConstraint,
    ConstraintEnforcer,
    ConstraintSeverity,
    EnergyBalanceConstraint,
    EnforcementStrategy,
    NonNegativityConstraint,
)


# ============================================================================
# AggregationConstraint Tests
# ============================================================================


def test_aggregation_constraint_check():
    """Test aggregation constraint violation checking."""
    constraint = AggregationConstraint(
        parent_node='National',
        child_nodes=['SE', 'S'],
        aggregation_weights={'SE': 1.0, 'S': 1.0},
        severity=ConstraintSeverity.HARD,
        tolerance=1e-6,
    )

    # Satisfies constraint
    forecasts = {
        'National': np.array([1000, 1100]),
        'SE': np.array([600, 650]),
        'S': np.array([400, 450]),
    }

    violation = constraint.check(forecasts)
    assert violation < 1e-6

    # Violates constraint
    forecasts['National'] = np.array([900, 1000])

    violation = constraint.check(forecasts)
    assert violation > 0
    assert violation == pytest.approx(100, abs=1e-6)


def test_aggregation_constraint_enforce():
    """Test aggregation constraint enforcement."""
    constraint = AggregationConstraint(
        parent_node='National',
        child_nodes=['SE', 'S'],
        aggregation_weights={'SE': 1.0, 'S': 1.0},
        severity=ConstraintSeverity.HARD,
    )

    forecasts = {
        'National': np.array([900, 1000]),  # Wrong
        'SE': np.array([600, 650]),
        'S': np.array([400, 450]),
    }

    adjusted = constraint.enforce(forecasts)

    # Parent should be adjusted to sum of children
    assert adjusted['National'][0] == 1000
    assert adjusted['National'][1] == 1100

    # Children unchanged
    assert np.array_equal(adjusted['SE'], forecasts['SE'])
    assert np.array_equal(adjusted['S'], forecasts['S'])


def test_aggregation_constraint_with_weights():
    """Test aggregation constraint with custom weights."""
    constraint = AggregationConstraint(
        parent_node='Total',
        child_nodes=['A', 'B'],
        aggregation_weights={'A': 2.0, 'B': 0.5},
        severity=ConstraintSeverity.HARD,
    )

    forecasts = {
        'Total': np.array([500]),
        'A': np.array([100]),  # 100 * 2.0 = 200
        'B': np.array([200]),  # 200 * 0.5 = 100
    }  # Should be 300

    adjusted = constraint.enforce(forecasts)
    assert adjusted['Total'][0] == 300


def test_aggregation_constraint_missing_parent():
    """Test aggregation constraint with missing parent node."""
    constraint = AggregationConstraint(
        parent_node='National',
        child_nodes=['SE', 'S'],
        aggregation_weights={'SE': 1.0, 'S': 1.0},
        severity=ConstraintSeverity.HARD,
    )

    forecasts = {
        'SE': np.array([600, 650]),
        'S': np.array([400, 450]),
    }

    violation = constraint.check(forecasts)
    assert violation == 0.0

    adjusted = constraint.enforce(forecasts)
    assert 'National' not in adjusted


def test_aggregation_constraint_missing_children():
    """Test aggregation constraint with missing child nodes."""
    constraint = AggregationConstraint(
        parent_node='National',
        child_nodes=['SE', 'S', 'NE'],
        aggregation_weights={'SE': 1.0, 'S': 1.0, 'NE': 1.0},
        severity=ConstraintSeverity.HARD,
    )

    # Only SE and S present
    forecasts = {
        'National': np.array([1000]),
        'SE': np.array([600]),
        'S': np.array([400]),
    }

    adjusted = constraint.enforce(forecasts)
    assert adjusted['National'][0] == 1000  # Sum of available children


# ============================================================================
# NonNegativityConstraint Tests
# ============================================================================


def test_non_negativity_constraint_check():
    """Test non-negativity constraint checking."""
    constraint = NonNegativityConstraint(
        nodes=['Area_1'], severity=ConstraintSeverity.HARD
    )

    # No violations
    forecasts = {'Area_1': np.array([100, 0, 200])}
    violation = constraint.check(forecasts)
    assert violation == 0.0

    # With violations
    forecasts = {'Area_1': np.array([100, -50, 200])}
    violation = constraint.check(forecasts)
    assert violation == 50


def test_non_negativity_constraint_enforce():
    """Test non-negativity constraint enforcement."""
    constraint = NonNegativityConstraint(
        nodes=['Area_1'], severity=ConstraintSeverity.HARD
    )

    forecasts = {'Area_1': np.array([100, -50, 200, -30])}

    adjusted = constraint.enforce(forecasts)
    assert np.all(adjusted['Area_1'] >= 0)
    assert adjusted['Area_1'][0] == 100
    assert adjusted['Area_1'][1] == 0
    assert adjusted['Area_1'][2] == 200
    assert adjusted['Area_1'][3] == 0


def test_non_negativity_with_floor():
    """Test non-negativity constraint with custom floor value."""
    constraint = NonNegativityConstraint(
        nodes=['Area_1'], floor_value=10.0, severity=ConstraintSeverity.HARD
    )

    forecasts = {'Area_1': np.array([100, 5, 20])}

    adjusted = constraint.enforce(forecasts)
    assert np.all(adjusted['Area_1'] >= 10.0)
    assert adjusted['Area_1'][0] == 100
    assert adjusted['Area_1'][1] == 10.0
    assert adjusted['Area_1'][2] == 20


def test_non_negativity_multiple_nodes():
    """Test non-negativity constraint on multiple nodes."""
    constraint = NonNegativityConstraint(
        nodes=['A', 'B', 'C'], severity=ConstraintSeverity.HARD
    )

    forecasts = {'A': np.array([100, -10]), 'B': np.array([50, 30]), 'C': np.array([-5, 20])}

    adjusted = constraint.enforce(forecasts)
    assert np.all(adjusted['A'] >= 0)
    assert np.all(adjusted['B'] >= 0)
    assert np.all(adjusted['C'] >= 0)


# ============================================================================
# EnergyBalanceConstraint Tests
# ============================================================================


def test_energy_balance_constraint_check():
    """Test energy balance constraint checking."""
    constraint = EnergyBalanceConstraint(
        generation_nodes=['Gen'],
        consumption_nodes=['Load'],
        loss_nodes=['Losses'],
        severity=ConstraintSeverity.HARD,
    )

    # Balanced
    forecasts = {
        'Gen': np.array([1000, 1100]),
        'Load': np.array([950, 1040]),
        'Losses': np.array([50, 60]),
    }

    violation = constraint.check(forecasts)
    assert violation < 1e-6

    # Imbalanced
    forecasts['Losses'] = np.array([30, 40])  # Too low

    violation = constraint.check(forecasts)
    assert violation > 0


def test_energy_balance_constraint_enforce():
    """Test energy balance constraint enforcement."""
    constraint = EnergyBalanceConstraint(
        generation_nodes=['Gen'],
        consumption_nodes=['Load'],
        loss_nodes=['Losses'],
        severity=ConstraintSeverity.HARD,
    )

    forecasts = {
        'Gen': np.array([1000]),
        'Load': np.array([950]),
        'Losses': np.array([30]),  # Should be 50
    }

    adjusted = constraint.enforce(forecasts)

    # Check balance
    gen = adjusted['Gen'][0]
    load = adjusted['Load'][0]
    losses = adjusted['Losses'][0]

    assert gen == pytest.approx(load + losses, abs=1e-6)


def test_energy_balance_multiple_nodes():
    """Test energy balance with multiple generation/consumption nodes."""
    constraint = EnergyBalanceConstraint(
        generation_nodes=['Gen1', 'Gen2'],
        consumption_nodes=['Load1', 'Load2'],
        loss_nodes=['Loss1', 'Loss2'],
        severity=ConstraintSeverity.HARD,
    )

    forecasts = {
        'Gen1': np.array([600]),
        'Gen2': np.array([400]),
        'Load1': np.array([500]),
        'Load2': np.array([450]),
        'Loss1': np.array([30]),
        'Loss2': np.array([20]),
    }

    adjusted = constraint.enforce(forecasts)

    # Total balance
    total_gen = adjusted['Gen1'][0] + adjusted['Gen2'][0]
    total_load = adjusted['Load1'][0] + adjusted['Load2'][0]
    total_loss = adjusted['Loss1'][0] + adjusted['Loss2'][0]

    assert total_gen == pytest.approx(total_load + total_loss, abs=1e-6)


# ============================================================================
# ConstraintEnforcer Tests
# ============================================================================


def test_constraint_enforcer_single_constraint():
    """Test constraint enforcer with single constraint."""
    constraints = [
        AggregationConstraint(
            parent_node='National',
            child_nodes=['SE', 'S'],
            aggregation_weights={'SE': 1.0, 'S': 1.0},
            severity=ConstraintSeverity.HARD,
        )
    ]

    enforcer = ConstraintEnforcer(
        constraints=constraints, strategy=EnforcementStrategy.ITERATIVE
    )

    forecasts = {
        'National': np.array([900]),  # Wrong
        'SE': np.array([600]),
        'S': np.array([400]),
    }

    adjusted = enforcer.enforce(forecasts)
    assert adjusted['National'][0] == 1000  # Should be sum of children


def test_constraint_enforcer_multiple_constraints():
    """Test constraint enforcer with multiple constraints."""
    constraints = [
        AggregationConstraint(
            parent_node='National',
            child_nodes=['SE', 'S'],
            aggregation_weights={'SE': 1.0, 'S': 1.0},
            severity=ConstraintSeverity.HARD,
        ),
        NonNegativityConstraint(nodes=['SE', 'S'], severity=ConstraintSeverity.HARD),
    ]

    enforcer = ConstraintEnforcer(
        constraints=constraints, strategy=EnforcementStrategy.ITERATIVE
    )

    forecasts = {
        'National': np.array([900]),
        'SE': np.array([600]),
        'S': np.array([-100]),  # Negative
    }

    adjusted = enforcer.enforce(forecasts)

    # Non-negativity enforced
    assert adjusted['S'][0] >= 0

    # Aggregation enforced (parent = SE + max(S, 0))
    expected_national = adjusted['SE'][0] + adjusted['S'][0]
    assert adjusted['National'][0] == pytest.approx(expected_national, abs=1e-6)


def test_constraint_enforcer_convergence():
    """Test constraint enforcer convergence detection."""
    constraints = [
        AggregationConstraint(
            parent_node='Total',
            child_nodes=['A'],
            aggregation_weights={'A': 1.0},
            severity=ConstraintSeverity.HARD,
        )
    ]

    enforcer = ConstraintEnforcer(
        constraints=constraints,
        strategy=EnforcementStrategy.ITERATIVE,
        max_iterations=10,
        convergence_tolerance=1e-8,
    )

    forecasts = {'Total': np.array([100]), 'A': np.array([100])}

    adjusted = enforcer.enforce(forecasts)
    # Should converge immediately since already satisfied
    assert adjusted['Total'][0] == 100


def test_constraint_enforcer_check_violations():
    """Test constraint violation checking."""
    constraints = [
        AggregationConstraint(
            parent_node='National',
            child_nodes=['SE', 'S'],
            aggregation_weights={'SE': 1.0, 'S': 1.0},
            severity=ConstraintSeverity.HARD,
            tolerance=1e-6,
        ),
        NonNegativityConstraint(
            nodes=['SE'], severity=ConstraintSeverity.HARD, tolerance=1e-6
        ),
    ]

    enforcer = ConstraintEnforcer(constraints=constraints)

    # Violates both constraints
    forecasts = {
        'National': np.array([900]),  # Wrong sum
        'SE': np.array([-10]),  # Negative
        'S': np.array([400]),
    }

    report = enforcer.check_violations(forecasts)

    assert report.total_violations == 2
    assert report.hard_constraint_violations == 2
    assert report.soft_constraint_violations == 0
    assert report.max_violation > 0


def test_constraint_enforcer_no_violations():
    """Test violation checking with no violations."""
    constraints = [
        NonNegativityConstraint(nodes=['A'], severity=ConstraintSeverity.HARD)
    ]

    enforcer = ConstraintEnforcer(constraints=constraints)

    forecasts = {'A': np.array([100, 200, 300])}

    report = enforcer.check_violations(forecasts)

    assert report.total_violations == 0
    assert report.max_violation == 0.0
    assert report.mean_violation == 0.0


def test_constraint_enforcer_strategies():
    """Test different enforcement strategies."""
    constraints = [
        AggregationConstraint(
            parent_node='Total',
            child_nodes=['A', 'B'],
            aggregation_weights={'A': 1.0, 'B': 1.0},
            severity=ConstraintSeverity.HARD,
        )
    ]

    forecasts = {'Total': np.array([100]), 'A': np.array([60]), 'B': np.array([50])}

    # ITERATIVE
    enforcer_iter = ConstraintEnforcer(
        constraints=constraints, strategy=EnforcementStrategy.ITERATIVE
    )
    adjusted_iter = enforcer_iter.enforce(forecasts)
    assert adjusted_iter['Total'][0] == 110

    # PROJECTION (falls back to iterative)
    enforcer_proj = ConstraintEnforcer(
        constraints=constraints, strategy=EnforcementStrategy.PROJECTION
    )
    adjusted_proj = enforcer_proj.enforce(forecasts)
    assert adjusted_proj['Total'][0] == 110

    # OPTIMIZATION (falls back to iterative)
    enforcer_opt = ConstraintEnforcer(
        constraints=constraints, strategy=EnforcementStrategy.OPTIMIZATION
    )
    adjusted_opt = enforcer_opt.enforce(forecasts)
    assert adjusted_opt['Total'][0] == 110

    # HYBRID
    enforcer_hybrid = ConstraintEnforcer(
        constraints=constraints, strategy=EnforcementStrategy.HYBRID
    )
    adjusted_hybrid = enforcer_hybrid.enforce(forecasts)
    assert adjusted_hybrid['Total'][0] == 110


def test_constraint_enforcer_max_iterations():
    """Test constraint enforcer respects max_iterations."""
    # Create circular constraints that can't converge
    constraints = [
        AggregationConstraint(
            parent_node='A',
            child_nodes=['B'],
            aggregation_weights={'B': 1.0},
            severity=ConstraintSeverity.HARD,
        )
    ]

    enforcer = ConstraintEnforcer(
        constraints=constraints, strategy=EnforcementStrategy.ITERATIVE, max_iterations=5
    )

    forecasts = {'A': np.array([100]), 'B': np.array([200])}

    # Should stop after 5 iterations (will converge in 1 though)
    adjusted = enforcer.enforce(forecasts)
    assert adjusted['A'][0] == 200


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_empty_forecasts():
    """Test with empty forecast dictionary."""
    constraint = AggregationConstraint(
        parent_node='A',
        child_nodes=['B'],
        aggregation_weights={'B': 1.0},
        severity=ConstraintSeverity.HARD,
    )

    forecasts = {}
    violation = constraint.check(forecasts)
    assert violation == 0.0

    adjusted = constraint.enforce(forecasts)
    assert adjusted == {}


def test_constraint_with_soft_severity():
    """Test constraints with soft severity."""
    constraint = AggregationConstraint(
        parent_node='Total',
        child_nodes=['A'],
        aggregation_weights={'A': 1.0},
        severity=ConstraintSeverity.SOFT,  # Soft
        penalty_weight=0.5,
    )

    forecasts = {'Total': np.array([100]), 'A': np.array([110])}

    # Soft constraints still detected as violations
    violation = constraint.check(forecasts)
    assert violation > 0

    # But enforcer only applies hard constraints by default
    enforcer = ConstraintEnforcer(constraints=[constraint])
    adjusted = enforcer.enforce(forecasts)
    # Soft constraints not enforced in iterative strategy
    assert adjusted['Total'][0] == 100  # Unchanged


def test_hierarchical_constraint_chain():
    """Test hierarchical chain of aggregation constraints."""
    constraints = [
        # Level 1: National = SE + S
        AggregationConstraint(
            parent_node='National',
            child_nodes=['SE', 'S'],
            aggregation_weights={'SE': 1.0, 'S': 1.0},
            severity=ConstraintSeverity.HARD,
        ),
        # Level 2: SE = Area1 + Area2
        AggregationConstraint(
            parent_node='SE',
            child_nodes=['Area1', 'Area2'],
            aggregation_weights={'Area1': 1.0, 'Area2': 1.0},
            severity=ConstraintSeverity.HARD,
        ),
    ]

    enforcer = ConstraintEnforcer(constraints=constraints)

    forecasts = {
        'National': np.array([500]),  # Wrong
        'SE': np.array([250]),  # Wrong
        'S': np.array([200]),
        'Area1': np.array([150]),
        'Area2': np.array([100]),
    }

    adjusted = enforcer.enforce(forecasts)

    # Check both levels
    assert adjusted['SE'][0] == 250  # Area1 + Area2
    assert adjusted['National'][0] == 450  # SE + S


def test_constraint_violation_report_format():
    """Test constraint violation report structure."""
    constraints = [
        AggregationConstraint(
            parent_node='A',
            child_nodes=['B'],
            aggregation_weights={'B': 1.0},
            severity=ConstraintSeverity.HARD,
            tolerance=1e-6,
        )
    ]

    enforcer = ConstraintEnforcer(constraints=constraints)

    forecasts = {'A': np.array([100]), 'B': np.array([200])}

    report = enforcer.check_violations(forecasts)

    # Check report structure
    assert hasattr(report, 'violations')
    assert hasattr(report, 'total_violations')
    assert hasattr(report, 'max_violation')
    assert hasattr(report, 'mean_violation')
    assert hasattr(report, 'hard_constraint_violations')
    assert hasattr(report, 'soft_constraint_violations')

    # Check violation details
    assert len(report.violations) == 1
    viol = report.violations[0]
    assert 'constraint_type' in viol
    assert 'severity' in viol
    assert 'magnitude' in viol
    assert 'tolerance' in viol
