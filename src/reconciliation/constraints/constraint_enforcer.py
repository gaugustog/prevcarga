"""Constraint enforcement system for hierarchical reconciliation."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConstraintSeverity(Enum):
    """Constraint severity levels."""

    HARD = "HARD"  # Must be satisfied exactly (within tolerance)
    SOFT = "SOFT"  # Violations penalized but allowed


class EnforcementStrategy(Enum):
    """Constraint enforcement strategies."""

    PROJECTION = "PROJECTION"  # Geometric projection
    OPTIMIZATION = "OPTIMIZATION"  # Constrained optimization
    ITERATIVE = "ITERATIVE"  # Successive adjustment
    HYBRID = "HYBRID"  # Combination of methods


@dataclass
class Constraint:
    """Base constraint class.

    Attributes:
        constraint_type: Type of constraint
        severity: Severity level (HARD or SOFT)
        tolerance: Allowed violation tolerance
        penalty_weight: Weight for soft constraint penalties
    """

    constraint_type: str
    severity: ConstraintSeverity
    tolerance: float = 1e-6
    penalty_weight: float = 1.0

    def check(self, forecasts: Dict[str, np.ndarray]) -> float:
        """Check constraint violation.

        Args:
            forecasts: Dictionary mapping node names to forecast arrays

        Returns:
            Violation magnitude (0 if satisfied)
        """
        raise NotImplementedError

    def enforce(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Enforce constraint on forecasts.

        Args:
            forecasts: Dictionary mapping node names to forecast arrays

        Returns:
            Adjusted forecasts satisfying constraint
        """
        raise NotImplementedError


@dataclass
class AggregationConstraint(Constraint):
    """Aggregation constraint: parent = Σ(weighted children).

    Ensures that parent node equals the sum of child nodes with optional
    weighting. This is the fundamental hierarchical coherence constraint.

    Example:
        National = SE_Subsystem + S_Subsystem + NE_Subsystem + N_Subsystem

    Attributes:
        parent_node: Parent node name
        child_nodes: List of child node names
        aggregation_weights: Optional weights for each child (default 1.0)
        severity: Constraint severity
        tolerance: Allowed violation tolerance
        penalty_weight: Weight for soft constraint penalties
    """

    parent_node: str = ""
    child_nodes: List[str] = field(default_factory=list)
    aggregation_weights: Dict[str, float] = field(default_factory=dict)
    constraint_type: str = field(default="AGGREGATION", init=False)

    def check(self, forecasts: Dict[str, np.ndarray]) -> float:
        """Check aggregation constraint.

        Args:
            forecasts: Dictionary mapping node names to forecast arrays

        Returns:
            Maximum absolute violation across all time periods
        """
        if self.parent_node not in forecasts:
            return 0.0

        parent_forecast = forecasts[self.parent_node]

        # Calculate expected value (sum of children)
        expected = np.zeros_like(parent_forecast, dtype=np.float64)
        for child in self.child_nodes:
            if child in forecasts:
                weight = self.aggregation_weights.get(child, 1.0)
                expected += weight * forecasts[child]

        # Calculate violation
        violation = np.abs(parent_forecast - expected)
        return float(np.max(violation))

    def enforce(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Enforce aggregation constraint by adjusting parent.

        Args:
            forecasts: Dictionary mapping node names to forecast arrays

        Returns:
            Adjusted forecasts with parent equal to sum of children
        """
        adjusted = forecasts.copy()

        if self.parent_node not in adjusted:
            return adjusted

        # Calculate correct parent value
        correct_parent = np.zeros_like(adjusted[self.parent_node], dtype=np.float64)
        for child in self.child_nodes:
            if child in adjusted:
                weight = self.aggregation_weights.get(child, 1.0)
                correct_parent += weight * adjusted[child]

        # Adjust parent (ensure dtype matches original)
        adjusted[self.parent_node] = correct_parent.astype(
            forecasts[self.parent_node].dtype
        )

        return adjusted


@dataclass
class EnergyBalanceConstraint(Constraint):
    """Energy balance constraint: Generation = Consumption + Losses.

    Ensures energy conservation across the system by enforcing that total
    generation equals total consumption plus transmission losses.

    Example:
        Total_Generation = Total_Consumption + Transmission_Losses

    Attributes:
        generation_nodes: List of generation node names
        consumption_nodes: List of consumption node names
        loss_nodes: List of loss node names
        severity: Constraint severity
        tolerance: Allowed violation tolerance
        penalty_weight: Weight for soft constraint penalties
    """

    generation_nodes: List[str] = field(default_factory=list)
    consumption_nodes: List[str] = field(default_factory=list)
    loss_nodes: List[str] = field(default_factory=list)
    constraint_type: str = field(default="ENERGY_BALANCE", init=False)

    def check(self, forecasts: Dict[str, np.ndarray]) -> float:
        """Check energy balance constraint.

        Args:
            forecasts: Dictionary mapping node names to forecast arrays

        Returns:
            Maximum absolute balance violation across all time periods
        """
        if not forecasts:
            return 0.0

        # Get array length from first forecast
        n_periods = len(next(iter(forecasts.values())))

        # Sum generation
        generation = np.zeros(n_periods)
        for node in self.generation_nodes:
            if node in forecasts:
                generation += forecasts[node]

        # Sum consumption
        consumption = np.zeros(n_periods)
        for node in self.consumption_nodes:
            if node in forecasts:
                consumption += forecasts[node]

        # Sum losses
        losses = np.zeros(n_periods)
        for node in self.loss_nodes:
            if node in forecasts:
                losses += forecasts[node]

        # Check balance
        violation = np.abs(generation - (consumption + losses))
        return float(np.max(violation))

    def enforce(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Enforce energy balance by adjusting losses.

        Adjusts loss node forecasts proportionally to ensure energy balance
        is satisfied.

        Args:
            forecasts: Dictionary mapping node names to forecast arrays

        Returns:
            Adjusted forecasts satisfying energy balance
        """
        adjusted = forecasts.copy()

        if not forecasts:
            return adjusted

        # Get array length from first forecast
        n_periods = len(next(iter(forecasts.values())))

        # Calculate total generation
        generation = np.zeros(n_periods)
        for node in self.generation_nodes:
            if node in adjusted:
                generation += adjusted[node]

        # Calculate total consumption
        consumption = np.zeros(n_periods)
        for node in self.consumption_nodes:
            if node in adjusted:
                consumption += adjusted[node]

        # Calculate required losses
        required_losses = generation - consumption

        # Distribute losses proportionally
        if self.loss_nodes and all(node in adjusted for node in self.loss_nodes):
            total_current_losses = sum(adjusted[node] for node in self.loss_nodes)

            for node in self.loss_nodes:
                if np.any(total_current_losses > 0):
                    proportion = adjusted[node] / np.maximum(
                        total_current_losses, 1e-10
                    )
                else:
                    proportion = 1.0 / len(self.loss_nodes)

                adjusted[node] = proportion * required_losses

        return adjusted


@dataclass
class NonNegativityConstraint(Constraint):
    """Non-negativity constraint: forecasts ≥ floor_value.

    Enforces that forecasts for specified nodes remain above a floor value
    (typically 0 for load forecasts).

    Attributes:
        nodes: List of node names to constrain
        floor_value: Minimum allowed value (default 0.0)
        severity: Constraint severity
        tolerance: Allowed violation tolerance
        penalty_weight: Weight for soft constraint penalties
    """

    nodes: List[str] = field(default_factory=list)
    floor_value: float = 0.0
    constraint_type: str = field(default="NON_NEGATIVITY", init=False)

    def check(self, forecasts: Dict[str, np.ndarray]) -> float:
        """Check non-negativity constraint.

        Args:
            forecasts: Dictionary mapping node names to forecast arrays

        Returns:
            Maximum violation magnitude (how far below floor)
        """
        max_violation = 0.0

        for node in self.nodes:
            if node in forecasts:
                violations = self.floor_value - forecasts[node]
                violations = violations[violations > 0]  # Only negative values
                if len(violations) > 0:
                    max_violation = max(max_violation, float(np.max(violations)))

        return max_violation

    def enforce(self, forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Enforce non-negativity by clipping.

        Args:
            forecasts: Dictionary mapping node names to forecast arrays

        Returns:
            Adjusted forecasts with values >= floor_value
        """
        adjusted = forecasts.copy()

        for node in self.nodes:
            if node in adjusted:
                adjusted[node] = np.maximum(adjusted[node], self.floor_value)

        return adjusted


@dataclass
class ConstraintViolationReport:
    """Report of constraint violations.

    Attributes:
        violations: List of violation dictionaries with details
        total_violations: Total number of violated constraints
        max_violation: Maximum violation magnitude
        mean_violation: Mean violation magnitude
        hard_constraint_violations: Number of hard constraint violations
        soft_constraint_violations: Number of soft constraint violations
    """

    violations: List[Dict[str, Any]]
    total_violations: int
    max_violation: float
    mean_violation: float
    hard_constraint_violations: int
    soft_constraint_violations: int


class ConstraintEnforcer:
    """Enforces constraints on hierarchical forecasts.

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

    Attributes:
        constraints: List of constraints to enforce
        strategy: Enforcement strategy to use
        max_iterations: Maximum iterations for iterative methods
        convergence_tolerance: Convergence criterion for iterative methods
    """

    def __init__(
        self,
        constraints: List[Constraint],
        strategy: EnforcementStrategy = EnforcementStrategy.ITERATIVE,
        max_iterations: int = 100,
        convergence_tolerance: float = 1e-8,
    ):
        """Initialize constraint enforcer.

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
        """Enforce all constraints on forecasts.

        Args:
            forecasts: Forecast dictionary mapping node names to arrays

        Returns:
            Constrained forecasts

        Raises:
            ValueError: If unknown enforcement strategy specified
        """
        logger.info(
            f"Enforcing {len(self.constraints)} constraints "
            f"using {self.strategy.value}"
        )

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
        self, forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Enforce constraints iteratively.

        Applies each hard constraint in sequence until convergence or max
        iterations reached.

        Args:
            forecasts: Forecast dictionary

        Returns:
            Constrained forecasts
        """
        adjusted = {k: v.copy() for k, v in forecasts.items()}

        for iteration in range(self.max_iterations):
            prev_adjusted = {k: v.copy() for k, v in adjusted.items()}

            # Apply each hard constraint
            for constraint in self.constraints:
                if constraint.severity == ConstraintSeverity.HARD:
                    adjusted = constraint.enforce(adjusted)

            # Check convergence
            max_change = max(
                np.max(np.abs(adjusted[k] - prev_adjusted[k])) for k in adjusted.keys()
            )

            if max_change < self.convergence_tolerance:
                logger.info(f"Converged in {iteration+1} iterations")
                break
        else:
            logger.warning(
                f"Did not converge after {self.max_iterations} iterations "
                f"(max change: {max_change:.2e})"
            )

        return adjusted

    def _enforce_projection(
        self, forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Enforce constraints via projection.

        Projects forecasts onto constraint set using geometric projection.

        Args:
            forecasts: Forecast dictionary

        Returns:
            Projected forecasts
        """
        # Simplified projection method - falls back to iterative
        return self._enforce_iterative(forecasts)

    def _enforce_optimization(
        self, forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Enforce constraints via optimization.

        Solves constrained optimization problem to find forecasts that
        minimize deviation from original while satisfying constraints.

        Args:
            forecasts: Forecast dictionary

        Returns:
            Optimal forecasts
        """
        # This would use scipy.optimize.minimize with constraints
        # For now, fallback to iterative
        logger.warning("Optimization method not fully implemented, using iterative")
        return self._enforce_iterative(forecasts)

    def _enforce_hybrid(
        self, forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Enforce constraints using hybrid approach.

        Combines multiple enforcement methods: hard constraints via iterative
        projection, soft constraints via optimization.

        Args:
            forecasts: Forecast dictionary

        Returns:
            Constrained forecasts
        """
        # Hard constraints via iterative
        adjusted = self._enforce_iterative(forecasts)

        # Soft constraints via optimization (if needed)
        # Not yet implemented

        return adjusted

    def check_violations(
        self, forecasts: Dict[str, np.ndarray]
    ) -> ConstraintViolationReport:
        """Check constraint violations.

        Evaluates all constraints and reports violations.

        Args:
            forecasts: Forecasts to check

        Returns:
            Violation report with detailed statistics
        """
        violations = []
        hard_violations = 0
        soft_violations = 0

        for constraint in self.constraints:
            violation_magnitude = constraint.check(forecasts)

            if violation_magnitude > constraint.tolerance:
                violations.append(
                    {
                        "constraint_type": constraint.constraint_type,
                        "severity": constraint.severity.value,
                        "magnitude": float(violation_magnitude),
                        "tolerance": constraint.tolerance,
                    }
                )

                if constraint.severity == ConstraintSeverity.HARD:
                    hard_violations += 1
                else:
                    soft_violations += 1

        if violations:
            max_viol = max(v["magnitude"] for v in violations)
            mean_viol = np.mean([v["magnitude"] for v in violations])
        else:
            max_viol = 0.0
            mean_viol = 0.0

        return ConstraintViolationReport(
            violations=violations,
            total_violations=len(violations),
            max_violation=max_viol,
            mean_violation=mean_viol,
            hard_constraint_violations=hard_violations,
            soft_constraint_violations=soft_violations,
        )
