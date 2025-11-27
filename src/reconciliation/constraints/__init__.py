"""Constraint enforcement system for hierarchical reconciliation.

This module provides constraint enforcement capabilities for hierarchical
forecasting reconciliation. Supports hard constraints (must be satisfied
exactly), soft constraints (penalized violations), and multiple enforcement
strategies.

Main Classes:
    ConstraintEnforcer: Main constraint enforcement orchestrator
    AggregationConstraint: Enforce hierarchical aggregation rules
    EnergyBalanceConstraint: Enforce energy balance equations
    NonNegativityConstraint: Enforce non-negativity constraints

Example:
    >>> from src.reconciliation.constraints import (
    ...     ConstraintEnforcer, AggregationConstraint,
    ...     ConstraintSeverity, EnforcementStrategy
    ... )
    >>>
    >>> # Define aggregation constraint
    >>> constraint = AggregationConstraint(
    ...     parent_node='National',
    ...     child_nodes=['SE', 'S', 'NE', 'N'],
    ...     aggregation_weights={'SE': 1.0, 'S': 1.0, 'NE': 1.0, 'N': 1.0},
    ...     severity=ConstraintSeverity.HARD
    ... )
    >>>
    >>> # Enforce constraints
    >>> enforcer = ConstraintEnforcer(
    ...     constraints=[constraint],
    ...     strategy=EnforcementStrategy.ITERATIVE
    ... )
    >>> adjusted_forecasts = enforcer.enforce(forecasts)
"""

from src.reconciliation.constraints.constraint_enforcer import (
    AggregationConstraint,
    Constraint,
    ConstraintEnforcer,
    ConstraintSeverity,
    ConstraintViolationReport,
    EnergyBalanceConstraint,
    EnforcementStrategy,
    NonNegativityConstraint,
)

__all__ = [
    'AggregationConstraint',
    'Constraint',
    'ConstraintEnforcer',
    'ConstraintSeverity',
    'ConstraintViolationReport',
    'EnergyBalanceConstraint',
    'EnforcementStrategy',
    'NonNegativityConstraint',
]
