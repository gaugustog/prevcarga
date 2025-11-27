"""Transmission loss calculation for hierarchical reconciliation.

This module calculates and validates transmission losses in the Brazilian
electric grid system. Typical losses range from 5-15% of generation.

Key Features:
- Multiple loss calculation models (percentage, absolute, historical, adaptive)
- Loss validation within expected ranges
- Energy balance constraint checking
- Loss distribution to subsystems
- Integration with hierarchical reconciliation

Loss Equation:
    Losses = Generation - Consumption
    Loss % = Losses / Generation * 100

Example:
    ```python
    from src.reconciliation.losses import LossCalculator, LossConfig, LossModel

    calculator = LossCalculator(config=LossConfig(
        loss_model=LossModel.PERCENTAGE,
        default_loss_percentage=0.08
    ))

    forecasts = {
        'Generation': np.array([10000, 11000, 12000]),
        'Consumption': np.array([9200, 10120, 11040])
    }

    result = calculator.calculate_losses(forecasts)
    print(f"Losses: {result.losses['Total_Losses']}")
    print(f"Loss %: {result.loss_percentages['Total_Losses']:.1f}%")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.reconciliation.hierarchy import HierarchyDefinition

# ruff: noqa: PLR2004, PLR0912

logger = get_logger(__name__)


class LossModel(Enum):
    """Loss calculation models.

    Attributes:
        PERCENTAGE: Losses as fixed percentage of generation
        ABSOLUTE: Fixed loss values (calculated from generation - consumption)
        HISTORICAL: Based on historical loss patterns
        ADAPTIVE: Adapts loss percentage based on load conditions
    """

    PERCENTAGE = "PERCENTAGE"
    ABSOLUTE = "ABSOLUTE"
    HISTORICAL = "HISTORICAL"
    ADAPTIVE = "ADAPTIVE"


@dataclass
class LossConfig:
    """Configuration for loss calculation.

    Attributes:
        loss_model: Model type for loss calculation.
        default_loss_percentage: Default loss as fraction of generation (0.08 = 8%).
        min_loss_percentage: Minimum allowed loss percentage.
        max_loss_percentage: Maximum allowed loss percentage (Brazilian grid typically <15%).
        enable_validation: Whether to validate calculated losses.
        adaptive_scale_factor: Scale factor for adaptive model adjustment.
        historical_window: Window size for historical pattern analysis.
    """

    loss_model: LossModel = LossModel.PERCENTAGE
    default_loss_percentage: float = 0.08  # 8% typical for Brazil
    min_loss_percentage: float = 0.0
    max_loss_percentage: float = 0.15  # 15% maximum
    enable_validation: bool = True
    adaptive_scale_factor: float = 0.001
    historical_window: int = 168  # 1 week of hourly data


@dataclass
class LossCalculationResult:
    """Result of loss calculation.

    Attributes:
        losses: Dictionary mapping loss names to loss arrays.
        loss_percentages: Dictionary mapping loss names to average percentage values.
        energy_balance_satisfied: Whether energy balance constraint is satisfied.
        validation_status: Detailed validation results.
        model_used: The loss model that was used for calculation.
    """

    losses: dict[str, np.ndarray]
    loss_percentages: dict[str, float]
    energy_balance_satisfied: bool
    validation_status: dict[str, Any]
    model_used: LossModel = LossModel.PERCENTAGE


@dataclass
class LossStatistics:
    """Statistics from loss analysis.

    Attributes:
        mean_loss_percentage: Mean loss as percentage of generation.
        std_loss_percentage: Standard deviation of loss percentage.
        min_loss_percentage: Minimum observed loss percentage.
        max_loss_percentage: Maximum observed loss percentage.
        temporal_patterns: Detected temporal patterns in losses.
        load_correlation: Correlation between load and loss percentage.
    """

    mean_loss_percentage: float = 0.0
    std_loss_percentage: float = 0.0
    min_loss_percentage: float = 0.0
    max_loss_percentage: float = 0.0
    temporal_patterns: dict[str, float] = field(default_factory=dict)
    load_correlation: float = 0.0


class LossCalculator:
    """Calculates and validates transmission losses.

    Transmission losses represent energy lost during transmission from
    generation to consumption. In the Brazilian system, typical losses
    range from 5-15% of generation.

    Loss Equation:
        Losses = Generation - Consumption
        Loss % = Losses / Generation * 100

    Models:
        - PERCENTAGE: Fixed percentage of generation
        - ABSOLUTE: Calculated from generation - consumption
        - HISTORICAL: Based on historical patterns
        - ADAPTIVE: Adapts based on load conditions

    Validation:
        - Losses should be non-negative
        - Loss percentage should be 0-15%
        - Energy balance: Generation = Consumption + Losses

    Integration with Reconciliation:
        - Losses are reconciled along with other forecasts
        - Energy balance constraints ensure consistency
        - Loss nodes participate in hierarchical aggregation

    Attributes:
        config: Loss calculation configuration.

    Example:
        >>> calculator = LossCalculator(config=LossConfig(
        ...     loss_model=LossModel.PERCENTAGE,
        ...     default_loss_percentage=0.08
        ... ))
        >>>
        >>> forecasts = {
        ...     'Generation': np.array([10000, 11000, 12000]),
        ...     'Consumption': np.array([9200, 10120, 11040])
        ... }
        >>>
        >>> result = calculator.calculate_losses(forecasts)
        >>> print(f"Losses: {result.losses['Total_Losses']}")
        >>> print(f"Loss %: {result.loss_percentages['Total_Losses']:.1f}%")
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        """Initialize loss calculator.

        Args:
            config: Loss calculation configuration. If None, uses defaults.
        """
        self.config = config or LossConfig()
        self._historical_losses: list[float] = []
        self._adaptive_model: LinearRegression | None = None

        logger.debug(
            "Initialized LossCalculator with model=%s, default_percentage=%.1f%%",
            self.config.loss_model.value,
            self.config.default_loss_percentage * 100,
        )

    def calculate_losses(
        self,
        forecasts: dict[str, np.ndarray | pd.Series],
        generation_nodes: list[str] | None = None,
        consumption_nodes: list[str] | None = None,
    ) -> LossCalculationResult:
        """Calculate transmission losses.

        Calculates losses using the configured model and validates results
        against expected ranges.

        Args:
            forecasts: Dictionary mapping node names to forecast arrays.
                Expected to contain generation and consumption nodes.
            generation_nodes: Optional list of generation node names.
                If None, auto-detects nodes containing 'generation'.
            consumption_nodes: Optional list of consumption node names.
                If None, auto-detects nodes containing 'consumption'.

        Returns:
            LossCalculationResult with calculated losses, percentages,
            and validation status.

        Raises:
            ValueError: If no generation or consumption nodes found.
        """
        logger.info("Calculating losses using %s model", self.config.loss_model.value)

        # Convert pandas Series to numpy arrays
        forecasts_np = self._to_numpy(forecasts)

        # Identify generation and consumption nodes
        if generation_nodes is None:
            generation_nodes = [k for k in forecasts_np if "generation" in k.lower()]
        if consumption_nodes is None:
            consumption_nodes = [k for k in forecasts_np if "consumption" in k.lower()]

        if not generation_nodes and not consumption_nodes:
            # Try to infer from 'generation' and 'consumption' keys directly
            for key in forecasts_np:
                if key.lower() == "generation":
                    generation_nodes = [key]
                elif key.lower() == "consumption":
                    consumption_nodes = [key]

        # Calculate total generation and consumption
        total_generation = self._sum_nodes(forecasts_np, generation_nodes)
        total_consumption = self._sum_nodes(forecasts_np, consumption_nodes)

        # Calculate losses based on model
        if self.config.loss_model == LossModel.PERCENTAGE:
            losses = self._calculate_percentage_losses(total_generation)
        elif self.config.loss_model == LossModel.ABSOLUTE:
            losses = self._calculate_absolute_losses(total_generation, total_consumption)
        elif self.config.loss_model == LossModel.HISTORICAL:
            losses = self._calculate_historical_losses(total_generation)
        elif self.config.loss_model == LossModel.ADAPTIVE:
            losses = self._calculate_adaptive_losses(total_generation, total_consumption)
        else:
            # Default to absolute calculation
            losses = self._calculate_absolute_losses(total_generation, total_consumption)

        # Calculate loss percentages
        loss_percentages = self._calculate_loss_percentages(
            {"Total_Losses": losses}, total_generation
        )

        # Validate losses if enabled
        if self.config.enable_validation:
            validation_status = self._validate_losses(losses, total_generation, total_consumption)
        else:
            validation_status = {"valid": True, "warnings": [], "errors": []}

        # Check energy balance
        energy_balance_satisfied = self._check_energy_balance(
            total_generation, total_consumption, losses
        )

        return LossCalculationResult(
            losses={"Total_Losses": losses},
            loss_percentages=loss_percentages,
            energy_balance_satisfied=energy_balance_satisfied,
            validation_status=validation_status,
            model_used=self.config.loss_model,
        )

    def _to_numpy(
        self, forecasts: dict[str, np.ndarray | pd.Series]
    ) -> dict[str, np.ndarray]:
        """Convert forecast values to numpy arrays.

        Args:
            forecasts: Dictionary of forecasts.

        Returns:
            Dictionary with all values as numpy arrays.
        """
        result = {}
        for key, value in forecasts.items():
            if isinstance(value, pd.Series):
                result[key] = value.to_numpy()
            else:
                result[key] = np.asarray(value)
        return result

    def _sum_nodes(
        self,
        forecasts: dict[str, np.ndarray],
        node_names: list[str],
    ) -> np.ndarray:
        """Sum forecasts for specified nodes.

        Args:
            forecasts: Dictionary of forecast arrays.
            node_names: List of node names to sum.

        Returns:
            Sum of forecasts for specified nodes.
        """
        if not node_names:
            # Get length from first available forecast
            first_value = next(iter(forecasts.values()), None)
            if first_value is not None:
                return np.zeros(len(first_value))
            return np.zeros(1)

        # Initialize with first node
        first_node = node_names[0]
        if first_node not in forecasts:
            return np.zeros(1)

        total = np.zeros(len(forecasts[first_node]))

        for node_name in node_names:
            if node_name in forecasts:
                total = total + forecasts[node_name]

        return total

    def _calculate_percentage_losses(self, generation: np.ndarray) -> np.ndarray:
        """Calculate losses as percentage of generation.

        Args:
            generation: Total generation array.

        Returns:
            Losses array calculated as percentage of generation.
        """
        return generation * self.config.default_loss_percentage

    def _calculate_absolute_losses(
        self,
        generation: np.ndarray,
        consumption: np.ndarray,
    ) -> np.ndarray:
        """Calculate losses as difference between generation and consumption.

        Args:
            generation: Total generation array.
            consumption: Total consumption array.

        Returns:
            Losses calculated as generation - consumption.
        """
        losses = generation - consumption
        # Ensure non-negative
        return np.maximum(losses, 0.0)

    def _calculate_historical_losses(self, generation: np.ndarray) -> np.ndarray:
        """Calculate losses based on historical patterns.

        Uses stored historical loss percentages to calculate expected losses.
        Falls back to percentage model if insufficient history.

        Args:
            generation: Total generation array.

        Returns:
            Losses based on historical patterns.
        """
        if len(self._historical_losses) < self.config.historical_window:
            logger.warning(
                "Insufficient historical data (%d < %d), using percentage model",
                len(self._historical_losses),
                self.config.historical_window,
            )
            return self._calculate_percentage_losses(generation)

        # Use recent average loss percentage
        recent_losses = self._historical_losses[-self.config.historical_window :]
        avg_loss_pct = np.mean(recent_losses)

        # Clip to valid range
        avg_loss_pct = np.clip(
            avg_loss_pct,
            self.config.min_loss_percentage,
            self.config.max_loss_percentage,
        )

        return generation * avg_loss_pct

    def _calculate_adaptive_losses(
        self,
        generation: np.ndarray,
        consumption: np.ndarray,
    ) -> np.ndarray:
        """Calculate losses adaptively based on load level.

        Higher loads typically result in slightly higher loss percentages
        due to increased transmission line heating.

        Args:
            generation: Total generation array.
            consumption: Total consumption array.

        Returns:
            Adaptively calculated losses.
        """
        base_percentage = self.config.default_loss_percentage

        # Increase loss percentage with load
        # Higher loads → higher losses due to I²R heating
        avg_load = float(np.mean(consumption))
        reference_load = 10000.0  # Reference load in MW

        load_factor = (avg_load / reference_load) * self.config.adaptive_scale_factor

        adjusted_percentage = np.clip(
            base_percentage + load_factor,
            self.config.min_loss_percentage,
            self.config.max_loss_percentage,
        )

        return generation * adjusted_percentage

    def _calculate_loss_percentages(
        self,
        losses: dict[str, np.ndarray],
        generation: np.ndarray,
    ) -> dict[str, float]:
        """Calculate loss percentages.

        Args:
            losses: Dictionary of loss arrays.
            generation: Total generation array.

        Returns:
            Dictionary mapping loss names to percentage values.
        """
        percentages = {}

        for loss_name, loss_values in losses.items():
            avg_loss = float(np.mean(loss_values))
            avg_generation = float(np.mean(generation))

            percentage = (avg_loss / avg_generation * 100) if avg_generation > 0 else 0.0
            percentages[loss_name] = percentage

        return percentages

    def _validate_losses(
        self,
        losses: np.ndarray,
        generation: np.ndarray,
        consumption: np.ndarray,
    ) -> dict[str, Any]:
        """Validate calculated losses.

        Checks for:
        - Non-negative losses
        - Loss percentage within expected range
        - Energy balance consistency

        Args:
            losses: Calculated losses array.
            generation: Total generation array.
            consumption: Total consumption array.

        Returns:
            Validation status dictionary with 'valid', 'warnings', 'errors' keys.
        """
        status: dict[str, Any] = {"valid": True, "warnings": [], "errors": []}

        # Check non-negativity
        if np.any(losses < 0):
            negative_count = int(np.sum(losses < 0))
            status["errors"].append(
                f"Negative losses detected in {negative_count} time periods"
            )
            status["valid"] = False

        # Check loss percentage range
        avg_generation = float(np.mean(generation))
        avg_loss_pct = (
            float(np.mean(losses)) / avg_generation * 100 if avg_generation > 0 else 0.0
        )

        if avg_loss_pct < self.config.min_loss_percentage * 100:
            status["warnings"].append(
                f"Loss percentage ({avg_loss_pct:.1f}%) below minimum "
                f"({self.config.min_loss_percentage * 100:.1f}%)"
            )

        if avg_loss_pct > self.config.max_loss_percentage * 100:
            status["errors"].append(
                f"Loss percentage ({avg_loss_pct:.1f}%) exceeds maximum "
                f"({self.config.max_loss_percentage * 100:.1f}%)"
            )
            status["valid"] = False

        # Check energy balance
        balance_error = np.abs(generation - (consumption + losses))
        max_balance_error = float(np.max(balance_error))

        if max_balance_error > 1.0:  # More than 1 MW error
            status["warnings"].append(
                f"Energy balance error: {max_balance_error:.2f} MW"
            )

        return status

    def _check_energy_balance(
        self,
        generation: np.ndarray,
        consumption: np.ndarray,
        losses: np.ndarray,
        tolerance: float = 1e-6,
    ) -> bool:
        """Check if energy balance is satisfied.

        Verifies: Generation = Consumption + Losses

        Args:
            generation: Total generation array.
            consumption: Total consumption array.
            losses: Calculated losses array.
            tolerance: Maximum allowed balance error.

        Returns:
            True if energy balance is satisfied within tolerance.
        """
        balance = generation - (consumption + losses)
        return bool(np.all(np.abs(balance) < tolerance))

    def distribute_losses(
        self,
        total_losses: np.ndarray,
        subsystem_loads: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Distribute total losses to subsystems proportionally.

        Distributes losses in proportion to each subsystem's load share.

        Args:
            total_losses: Total system losses array.
            subsystem_loads: Dictionary mapping subsystem names to load arrays.

        Returns:
            Dictionary mapping subsystem loss names to distributed loss arrays.
        """
        if not subsystem_loads:
            return {}

        # Calculate total load
        total_load = sum(subsystem_loads.values())

        # Avoid division by zero
        total_load_safe = np.where(total_load > 0, total_load, 1.0)

        # Distribute proportionally to load
        distributed_losses = {}
        for subsystem_name, load in subsystem_loads.items():
            proportion = load / total_load_safe
            loss_name = f"{subsystem_name}_Losses"
            distributed_losses[loss_name] = proportion * total_losses

        return distributed_losses

    def adjust_for_reconciliation(
        self,
        forecasts: dict[str, np.ndarray],
        target_loss_percentage: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Adjust forecasts to achieve target loss percentage.

        Adjusts loss nodes proportionally to achieve the target loss
        percentage while maintaining forecast structure.

        Args:
            forecasts: Forecast dictionary with generation, consumption, losses.
            target_loss_percentage: Target loss as fraction (e.g., 0.08 for 8%).
                If None, uses config default.

        Returns:
            Adjusted forecast dictionary.
        """
        if target_loss_percentage is None:
            target_loss_percentage = self.config.default_loss_percentage

        adjusted = dict(forecasts)

        # Identify keys
        generation_keys = [k for k in forecasts if "generation" in k.lower()]
        loss_keys = [k for k in forecasts if "loss" in k.lower()]

        if not generation_keys or not loss_keys:
            logger.warning("Cannot adjust: missing generation or loss keys")
            return adjusted

        # Calculate target losses
        total_generation = self._sum_nodes(forecasts, generation_keys)
        target_losses = total_generation * target_loss_percentage

        # Get current total losses
        current_total_losses = self._sum_nodes(forecasts, loss_keys)

        # Adjust each loss node proportionally
        for loss_key in loss_keys:
            current = forecasts[loss_key]
            # Safe division
            proportion = np.where(
                current_total_losses > 0,
                current / current_total_losses,
                1.0 / len(loss_keys),
            )
            adjusted[loss_key] = proportion * target_losses

        return adjusted

    def update_historical(self, loss_percentage: float) -> None:
        """Update historical loss data.

        Args:
            loss_percentage: Observed loss percentage to add to history.
        """
        self._historical_losses.append(loss_percentage)

        # Keep only recent history
        max_history = self.config.historical_window * 2
        if len(self._historical_losses) > max_history:
            self._historical_losses = self._historical_losses[-max_history:]

    def fit_adaptive_model(
        self,
        historical_loads: np.ndarray,
        historical_loss_percentages: np.ndarray,
    ) -> None:
        """Fit adaptive loss model using historical data.

        Trains a linear regression model relating load to loss percentage.

        Args:
            historical_loads: Historical load values.
            historical_loss_percentages: Corresponding loss percentages.
        """
        if len(historical_loads) < 10:
            logger.warning("Insufficient data to fit adaptive model")
            return

        self._adaptive_model = LinearRegression()
        x_train = historical_loads.reshape(-1, 1)
        self._adaptive_model.fit(x_train, historical_loss_percentages)

        logger.info(
            "Fitted adaptive model: coefficient=%.6f, intercept=%.4f",
            self._adaptive_model.coef_[0],
            self._adaptive_model.intercept_,
        )

    def analyze_historical_losses(
        self,
        generation: np.ndarray,
        consumption: np.ndarray,
        timestamps: pd.DatetimeIndex | None = None,
    ) -> LossStatistics:
        """Analyze historical loss patterns.

        Computes statistics on historical losses and detects temporal patterns.

        Args:
            generation: Historical generation data.
            consumption: Historical consumption data.
            timestamps: Optional timestamps for temporal pattern analysis.

        Returns:
            LossStatistics with computed metrics.
        """
        # Calculate loss percentages
        losses = generation - consumption
        loss_pct = np.where(generation > 0, (losses / generation) * 100, 0)

        # Basic statistics
        stats = LossStatistics(
            mean_loss_percentage=float(np.mean(loss_pct)),
            std_loss_percentage=float(np.std(loss_pct)),
            min_loss_percentage=float(np.min(loss_pct)),
            max_loss_percentage=float(np.max(loss_pct)),
        )

        # Correlation with load
        if len(consumption) > 2:
            valid_mask = (consumption > 0) & np.isfinite(loss_pct)
            if np.sum(valid_mask) > 2:
                corr = np.corrcoef(consumption[valid_mask], loss_pct[valid_mask])
                stats.load_correlation = float(corr[0, 1])

        # Temporal patterns (if timestamps provided)
        if timestamps is not None and len(timestamps) == len(loss_pct):
            df = pd.DataFrame({"loss_pct": loss_pct}, index=timestamps)

            # Hourly pattern
            if len(df) >= 24:
                hourly = df.groupby(df.index.hour)["loss_pct"].mean()
                stats.temporal_patterns["hourly_range"] = float(
                    hourly.max() - hourly.min()
                )

            # Weekly pattern
            if len(df) >= 168:
                daily = df.groupby(df.index.dayofweek)["loss_pct"].mean()
                stats.temporal_patterns["weekly_range"] = float(
                    daily.max() - daily.min()
                )

        # Update internal history
        for pct in loss_pct:
            if np.isfinite(pct):
                self._historical_losses.append(float(pct / 100))  # Store as fraction

        return stats

    def generate_loss_report(
        self,
        result: LossCalculationResult,
        _forecasts: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Generate comprehensive loss report.

        Args:
            result: Loss calculation result.
            forecasts: Original forecasts.

        Returns:
            Dictionary containing loss report data.
        """
        report = {
            "model_used": result.model_used.value,
            "loss_percentages": result.loss_percentages,
            "energy_balance_satisfied": result.energy_balance_satisfied,
            "validation": result.validation_status,
        }

        # Add loss statistics
        losses = result.losses.get("Total_Losses", np.array([]))
        if len(losses) > 0:
            report["loss_statistics"] = {
                "mean_mw": float(np.mean(losses)),
                "min_mw": float(np.min(losses)),
                "max_mw": float(np.max(losses)),
                "std_mw": float(np.std(losses)),
            }

        # Compare to historical if available
        if self._historical_losses:
            avg_historical = np.mean(self._historical_losses) * 100
            current_pct = result.loss_percentages.get("Total_Losses", 0)
            report["historical_comparison"] = {
                "historical_avg_pct": avg_historical,
                "current_pct": current_pct,
                "deviation_pct": current_pct - avg_historical,
            }

        return report


class LossIntegrator:
    """Integrates loss calculation with hierarchical reconciliation.

    Ensures that:
    - Loss nodes are properly included in hierarchy
    - Energy balance constraints are enforced
    - Losses are reconciled coherently with other forecasts

    Attributes:
        loss_calculator: The LossCalculator instance to use.

    Example:
        >>> integrator = LossIntegrator(loss_calculator)
        >>> forecasts_with_losses = integrator.integrate(
        ...     forecasts, hierarchy
        ... )
    """

    def __init__(self, loss_calculator: LossCalculator) -> None:
        """Initialize loss integrator.

        Args:
            loss_calculator: LossCalculator instance for loss calculations.
        """
        self.loss_calculator = loss_calculator

    def integrate(
        self,
        forecasts: dict[str, np.ndarray],
        _hierarchy: HierarchyDefinition | None = None,
        generation_nodes: list[str] | None = None,
        consumption_nodes: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Integrate loss calculations with forecasts.

        Calculates losses and adds them to the forecast dictionary,
        adjusting if needed to satisfy energy balance.

        Args:
            forecasts: Base forecasts dictionary.
            hierarchy: Optional hierarchy definition for validation.
            generation_nodes: Optional list of generation node names.
            consumption_nodes: Optional list of consumption node names.

        Returns:
            Forecasts with integrated losses.
        """
        logger.info("Integrating loss calculations with forecasts")

        # Calculate losses
        loss_result = self.loss_calculator.calculate_losses(
            forecasts,
            generation_nodes=generation_nodes,
            consumption_nodes=consumption_nodes,
        )

        # Add losses to forecasts
        integrated = dict(forecasts)
        integrated.update(loss_result.losses)

        # Validate energy balance
        if not loss_result.energy_balance_satisfied:
            logger.warning("Energy balance not satisfied, adjusting losses...")
            integrated = self.loss_calculator.adjust_for_reconciliation(integrated)

        # Log validation status
        if loss_result.validation_status.get("errors"):
            for error in loss_result.validation_status["errors"]:
                logger.error("Loss validation error: %s", error)

        if loss_result.validation_status.get("warnings"):
            for warning in loss_result.validation_status["warnings"]:
                logger.warning("Loss validation warning: %s", warning)

        return integrated

    def distribute_to_subsystems(
        self,
        forecasts: dict[str, np.ndarray],
        subsystem_keys: list[str],
        total_loss_key: str = "Total_Losses",
    ) -> dict[str, np.ndarray]:
        """Distribute total losses to subsystems.

        Args:
            forecasts: Forecasts including total losses.
            subsystem_keys: List of subsystem keys in forecasts.
            total_loss_key: Key for total losses in forecasts.

        Returns:
            Forecasts with distributed subsystem losses.
        """
        if total_loss_key not in forecasts:
            logger.warning("Total losses not found in forecasts")
            return forecasts

        total_losses = forecasts[total_loss_key]

        # Extract subsystem loads
        subsystem_loads = {k: forecasts[k] for k in subsystem_keys if k in forecasts}

        if not subsystem_loads:
            logger.warning("No subsystem data found for loss distribution")
            return forecasts

        # Distribute losses
        distributed = self.loss_calculator.distribute_losses(total_losses, subsystem_loads)

        # Add to forecasts
        result = dict(forecasts)
        result.update(distributed)

        return result

    def validate_hierarchy_losses(
        self,
        forecasts: dict[str, np.ndarray],
        hierarchy: HierarchyDefinition,
    ) -> dict[str, Any]:
        """Validate that losses in hierarchy are coherent.

        Args:
            forecasts: Forecasts with losses.
            hierarchy: Hierarchy definition.

        Returns:
            Validation result dictionary.
        """
        validation_result: dict[str, Any] = {
            "valid": True,
            "issues": [],
        }

        # Find loss nodes in hierarchy
        loss_nodes = [n for n in hierarchy.get_all_node_names() if "loss" in n.lower()]

        if not loss_nodes:
            validation_result["issues"].append("No loss nodes found in hierarchy")
            return validation_result

        # Check that all loss nodes have forecasts
        missing = [n for n in loss_nodes if n not in forecasts]
        if missing:
            validation_result["issues"].append(
                f"Missing forecasts for loss nodes: {missing}"
            )
            validation_result["valid"] = False

        # Check non-negative
        for node in loss_nodes:
            if node in forecasts and np.any(forecasts[node] < 0):
                validation_result["issues"].append(
                    f"Negative losses in node {node}"
                )
                validation_result["valid"] = False

        return validation_result
