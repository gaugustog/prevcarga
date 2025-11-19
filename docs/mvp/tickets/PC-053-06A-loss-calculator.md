# PC-053-06A: Loss Calculation System

**Ticket ID:** PC-053-06A  
**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**User Story:** US-6  
**Story Points:** 3  
**Priority:** Medium  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement transmission loss calculation and integration with hierarchical reconciliation. Calculates losses as the difference between generation and consumption (Losses = Generation - Consumption), supports percentage-based and absolute loss models, validates losses within expected ranges (0-15% typical for Brazilian system), and integrates with energy balance constraints for coherent reconciliation.

**As a** power system analyst  
**I want** accurate transmission loss calculations integrated with forecasts  
**So that** energy balance constraints are satisfied and losses are realistic

---

## ✅ Acceptance Criteria

- [ ] Loss calculation: Losses = Generation - Consumption
- [ ] Percentage-based loss model support
- [ ] Absolute loss model support
- [ ] Loss validation (0-15% range)
- [ ] Integration with hierarchy reconciliation
- [ ] Historical loss pattern analysis
- [ ] Loss forecasting for future periods
- [ ] Energy balance constraint integration
- [ ] Performance overhead <2%
- [ ] Configurable loss bounds

---

## 🔧 Implementation Tasks

### 1. Create Loss Calculation Module
- [ ] Create `src/models/reconciliation/losses/__init__.py`
- [ ] Create `src/models/reconciliation/losses/loss_calculator.py`
- [ ] Import numpy for calculations
- [ ] Add module docstrings
- [ ] Define loss model types

### 2. Implement LossModel Enum
- [ ] Create `LossModel` enum
- [ ] PERCENTAGE: losses as % of generation
- [ ] ABSOLUTE: fixed loss values
- [ ] HISTORICAL: based on historical patterns
- [ ] ADAPTIVE: adaptive based on load

### 3. Implement LossCalculator Class
- [ ] Create `LossCalculator` base class
- [ ] Add loss_model attribute
- [ ] Add expected_loss_range attribute
- [ ] Add validation parameters
- [ ] Initialize with configuration

### 4. Implement Loss Calculation Method
- [ ] Create `calculate_losses()` method
- [ ] Calculate: losses = generation - consumption
- [ ] Support multiple loss series
- [ ] Handle time series data
- [ ] Return loss forecasts

### 5. Implement Percentage Loss Model
- [ ] Create `PercentageLossCalculator` class
- [ ] Calculate losses as % of generation
- [ ] Default percentage range: 5-15%
- [ ] Configurable percentage parameter
- [ ] Apply: losses = generation * loss_percentage

### 6. Implement Absolute Loss Model
- [ ] Create `AbsoluteLossCalculator` class
- [ ] Calculate fixed loss values
- [ ] Support time-varying losses
- [ ] Validate against bounds
- [ ] Apply: losses = fixed_value

### 7. Implement Historical Loss Pattern Analysis
- [ ] Create `HistoricalLossAnalyzer` class
- [ ] Analyze historical loss patterns
- [ ] Calculate typical loss percentages
- [ ] Identify temporal patterns
- [ ] Return loss statistics

### 8. Implement Adaptive Loss Forecasting
- [ ] Create `AdaptiveLossForecaster` class
- [ ] Predict losses based on load level
- [ ] Use historical loss-load relationship
- [ ] Apply regression model
- [ ] Return loss forecasts

### 9. Implement Loss Validation
- [ ] Create `_validate_losses()` method
- [ ] Check losses within expected range (0-15%)
- [ ] Validate non-negative losses
- [ ] Check energy balance
- [ ] Return validation status

### 10. Implement Integration with Reconciliation
- [ ] Create `integrate_with_reconciliation()` method
- [ ] Add loss nodes to hierarchy
- [ ] Create energy balance constraints
- [ ] Ensure losses updated during reconciliation
- [ ] Return integrated forecasts

### 11. Implement Loss Adjustment During Reconciliation
- [ ] Create `adjust_losses()` method
- [ ] Adjust losses to maintain energy balance
- [ ] Preserve loss percentage if possible
- [ ] Apply constraints
- [ ] Return adjusted losses

### 12. Implement Loss Distribution
- [ ] Create `distribute_losses()` method
- [ ] Distribute total losses to subsystems
- [ ] Proportional to load or generation
- [ ] Support custom distribution weights
- [ ] Return distributed losses

### 13. Implement Loss Reporting
- [ ] Create `generate_loss_report()` method
- [ ] Calculate loss statistics
- [ ] Report loss percentages
- [ ] Compare to historical averages
- [ ] Return loss report

### 14. Handle Edge Cases
- [ ] Handle generation < consumption (invalid)
- [ ] Handle negative losses (adjust)
- [ ] Handle extreme loss percentages
- [ ] Handle missing generation/consumption data
- [ ] Graceful error handling

### 15. Write Comprehensive Tests
- [ ] Create `tests/models/reconciliation/losses/test_loss_calculator.py`
- [ ] Test percentage loss model
- [ ] Test absolute loss model
- [ ] Test loss validation
- [ ] Test integration with reconciliation
- [ ] Test edge cases

### 16. Create Loss Calculation Documentation
- [ ] Create `docs/loss_calculation.md`
- [ ] Document loss models
- [ ] Explain integration with reconciliation
- [ ] Provide examples
- [ ] Document configuration

---

## 💻 Implementation Details

### LossCalculator Implementation

```python
"""Transmission loss calculation for hierarchical reconciliation."""
from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LossModel(Enum):
    """Loss calculation models."""
    PERCENTAGE = "PERCENTAGE"      # Losses as % of generation
    ABSOLUTE = "ABSOLUTE"          # Fixed loss values
    HISTORICAL = "HISTORICAL"      # Based on historical patterns
    ADAPTIVE = "ADAPTIVE"          # Adaptive based on load level


@dataclass
class LossCalculationResult:
    """Result of loss calculation."""
    
    losses: Dict[str, np.ndarray]
    loss_percentages: Dict[str, float]
    energy_balance_satisfied: bool
    validation_status: Dict[str, Any]


@dataclass
class LossConfig:
    """Configuration for loss calculation."""
    
    loss_model: LossModel = LossModel.PERCENTAGE
    default_loss_percentage: float = 0.08  # 8% typical
    min_loss_percentage: float = 0.0
    max_loss_percentage: float = 0.15  # 15% maximum
    enable_validation: bool = True


class LossCalculator:
    """
    Calculates and validates transmission losses.
    
    Transmission losses represent energy lost during transmission
    from generation to consumption. In the Brazilian system, typical
    losses range from 5-15% of generation.
    
    Loss Equation:
        Losses = Generation - Consumption
        Loss % = Losses / Generation * 100
    
    Models:
        - PERCENTAGE: Fixed percentage of generation
        - ABSOLUTE: Fixed loss values
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
    
    def __init__(self, config: Optional[LossConfig] = None):
        """
        Initialize loss calculator.
        
        Args:
            config: Loss calculation configuration
        """
        self.config = config or LossConfig()
    
    def calculate_losses(
        self,
        forecasts: Dict[str, np.ndarray],
        generation_nodes: Optional[List[str]] = None,
        consumption_nodes: Optional[List[str]] = None
    ) -> LossCalculationResult:
        """
        Calculate transmission losses.
        
        Args:
            forecasts: Forecast dictionary
            generation_nodes: List of generation node names
            consumption_nodes: List of consumption node names
        
        Returns:
            LossCalculationResult with calculated losses
        """
        logger.info(f"Calculating losses using {self.config.loss_model.value} model")
        
        # Identify generation and consumption
        if generation_nodes is None:
            generation_nodes = [k for k in forecasts.keys() if 'generation' in k.lower()]
        if consumption_nodes is None:
            consumption_nodes = [k for k in forecasts.keys() if 'consumption' in k.lower()]
        
        # Calculate total generation and consumption
        total_generation = self._sum_nodes(forecasts, generation_nodes)
        total_consumption = self._sum_nodes(forecasts, consumption_nodes)
        
        # Calculate losses based on model
        if self.config.loss_model == LossModel.PERCENTAGE:
            losses = self._calculate_percentage_losses(total_generation)
        elif self.config.loss_model == LossModel.ABSOLUTE:
            losses = total_generation - total_consumption
        elif self.config.loss_model == LossModel.HISTORICAL:
            losses = self._calculate_historical_losses(total_generation)
        elif self.config.loss_model == LossModel.ADAPTIVE:
            losses = self._calculate_adaptive_losses(total_generation, total_consumption)
        else:
            losses = total_generation - total_consumption
        
        # Calculate loss percentages
        loss_percentages = self._calculate_loss_percentages({
            'Total_Losses': losses
        }, total_generation)
        
        # Validate losses
        validation_status = self._validate_losses(
            losses, total_generation, total_consumption
        )
        
        # Check energy balance
        energy_balance_satisfied = self._check_energy_balance(
            total_generation, total_consumption, losses
        )
        
        return LossCalculationResult(
            losses={'Total_Losses': losses},
            loss_percentages=loss_percentages,
            energy_balance_satisfied=energy_balance_satisfied,
            validation_status=validation_status
        )
    
    def _calculate_percentage_losses(self, generation: np.ndarray) -> np.ndarray:
        """Calculate losses as percentage of generation."""
        return generation * self.config.default_loss_percentage
    
    def _calculate_historical_losses(self, generation: np.ndarray) -> np.ndarray:
        """Calculate losses based on historical patterns."""
        # Simplified: use default percentage
        # In practice, would use historical data
        logger.warning("Historical loss model not fully implemented, using default percentage")
        return self._calculate_percentage_losses(generation)
    
    def _calculate_adaptive_losses(
        self,
        generation: np.ndarray,
        consumption: np.ndarray
    ) -> np.ndarray:
        """Calculate losses adaptively based on load."""
        # Simplified: losses scale with load level
        # Higher loads → slightly higher loss percentages
        base_percentage = self.config.default_loss_percentage
        
        # Increase loss percentage by 0.1% per 1000 MW increase
        avg_load = np.mean(consumption)
        load_factor = (avg_load / 10000) * 0.001  # Scale factor
        
        adjusted_percentage = np.clip(
            base_percentage + load_factor,
            self.config.min_loss_percentage,
            self.config.max_loss_percentage
        )
        
        return generation * adjusted_percentage
    
    def _sum_nodes(
        self,
        forecasts: Dict[str, np.ndarray],
        node_names: List[str]
    ) -> np.ndarray:
        """Sum forecasts for specified nodes."""
        if not node_names:
            return np.zeros(len(next(iter(forecasts.values()))))
        
        total = np.zeros(len(forecasts[node_names[0]]))
        for node_name in node_names:
            if node_name in forecasts:
                total += forecasts[node_name]
        
        return total
    
    def _calculate_loss_percentages(
        self,
        losses: Dict[str, np.ndarray],
        generation: np.ndarray
    ) -> Dict[str, float]:
        """Calculate loss percentages."""
        percentages = {}
        
        for loss_name, loss_values in losses.items():
            avg_loss = np.mean(loss_values)
            avg_generation = np.mean(generation)
            
            if avg_generation > 0:
                percentage = (avg_loss / avg_generation) * 100
            else:
                percentage = 0.0
            
            percentages[loss_name] = percentage
        
        return percentages
    
    def _validate_losses(
        self,
        losses: np.ndarray,
        generation: np.ndarray,
        consumption: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate calculated losses.
        
        Returns:
            Validation status dictionary
        """
        status = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check non-negativity
        if np.any(losses < 0):
            status['errors'].append("Negative losses detected")
            status['valid'] = False
        
        # Check loss percentage range
        avg_loss_pct = (np.mean(losses) / np.mean(generation)) * 100
        
        if avg_loss_pct < self.config.min_loss_percentage * 100:
            status['warnings'].append(
                f"Loss percentage ({avg_loss_pct:.1f}%) below minimum "
                f"({self.config.min_loss_percentage * 100:.1f}%)"
            )
        
        if avg_loss_pct > self.config.max_loss_percentage * 100:
            status['errors'].append(
                f"Loss percentage ({avg_loss_pct:.1f}%) exceeds maximum "
                f"({self.config.max_loss_percentage * 100:.1f}%)"
            )
            status['valid'] = False
        
        # Check energy balance
        balance_error = np.abs(generation - (consumption + losses))
        max_balance_error = np.max(balance_error)
        
        if max_balance_error > 1e-6:
            status['warnings'].append(
                f"Energy balance error: {max_balance_error:.2e}"
            )
        
        return status
    
    def _check_energy_balance(
        self,
        generation: np.ndarray,
        consumption: np.ndarray,
        losses: np.ndarray,
        tolerance: float = 1e-6
    ) -> bool:
        """Check if energy balance is satisfied."""
        balance = generation - (consumption + losses)
        return np.all(np.abs(balance) < tolerance)
    
    def distribute_losses(
        self,
        total_losses: np.ndarray,
        subsystem_loads: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Distribute total losses to subsystems proportionally.
        
        Args:
            total_losses: Total system losses
            subsystem_loads: Load for each subsystem
        
        Returns:
            Dictionary of losses per subsystem
        """
        # Calculate total load
        total_load = sum(subsystem_loads.values())
        
        # Distribute proportionally to load
        distributed_losses = {}
        for subsystem_name, load in subsystem_loads.items():
            proportion = load / total_load
            distributed_losses[f"{subsystem_name}_Losses"] = proportion * total_losses
        
        return distributed_losses
    
    def adjust_for_reconciliation(
        self,
        forecasts: Dict[str, np.ndarray],
        target_loss_percentage: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Adjust forecasts to achieve target loss percentage.
        
        Args:
            forecasts: Forecast dictionary with generation, consumption, losses
            target_loss_percentage: Target loss % (if None, use config default)
        
        Returns:
            Adjusted forecast dictionary
        """
        if target_loss_percentage is None:
            target_loss_percentage = self.config.default_loss_percentage
        
        adjusted = forecasts.copy()
        
        # Identify generation and consumption
        generation_keys = [k for k in forecasts.keys() if 'generation' in k.lower()]
        consumption_keys = [k for k in forecasts.keys() if 'consumption' in k.lower()]
        loss_keys = [k for k in forecasts.keys() if 'loss' in k.lower()]
        
        if generation_keys and loss_keys:
            total_generation = self._sum_nodes(forecasts, generation_keys)
            target_losses = total_generation * target_loss_percentage
            
            # Adjust loss nodes proportionally
            current_total_losses = self._sum_nodes(forecasts, loss_keys)
            
            for loss_key in loss_keys:
                if current_total_losses.sum() > 0:
                    proportion = forecasts[loss_key] / current_total_losses
                else:
                    proportion = 1.0 / len(loss_keys)
                
                adjusted[loss_key] = proportion * target_losses
        
        return adjusted


class LossIntegrator:
    """
    Integrates loss calculation with hierarchical reconciliation.
    
    Ensures that:
        - Loss nodes are included in hierarchy
        - Energy balance constraints are enforced
        - Losses are reconciled coherently with other forecasts
    
    Example:
        >>> integrator = LossIntegrator(loss_calculator)
        >>> forecasts_with_losses = integrator.integrate(
        ...     forecasts, hierarchy
        ... )
    """
    
    def __init__(self, loss_calculator: LossCalculator):
        """
        Initialize loss integrator.
        
        Args:
            loss_calculator: LossCalculator instance
        """
        self.loss_calculator = loss_calculator
    
    def integrate(
        self,
        forecasts: Dict[str, np.ndarray],
        hierarchy: 'HierarchyDefinition'
    ) -> Dict[str, np.ndarray]:
        """
        Integrate loss calculations with forecasts.
        
        Args:
            forecasts: Base forecasts
            hierarchy: Hierarchy definition
        
        Returns:
            Forecasts with integrated losses
        """
        logger.info("Integrating loss calculations with hierarchy")
        
        # Calculate losses
        loss_result = self.loss_calculator.calculate_losses(forecasts)
        
        # Add losses to forecasts
        integrated = forecasts.copy()
        integrated.update(loss_result.losses)
        
        # Validate energy balance
        if not loss_result.energy_balance_satisfied:
            logger.warning("Energy balance not satisfied, adjusting...")
            integrated = self.loss_calculator.adjust_for_reconciliation(integrated)
        
        return integrated
```

---

## 🧪 Testing & Validation

```python
"""Tests for loss calculator."""
import pytest
import numpy as np

from src.models.reconciliation.losses.loss_calculator import (
    LossCalculator, LossModel, LossConfig
)


def test_percentage_loss_calculation():
    """Test percentage-based loss calculation."""
    config = LossConfig(
        loss_model=LossModel.PERCENTAGE,
        default_loss_percentage=0.08
    )
    calculator = LossCalculator(config)
    
    forecasts = {
        'Generation': np.array([10000, 11000, 12000]),
        'Consumption': np.array([9200, 10120, 11040])
    }
    
    result = calculator.calculate_losses(forecasts)
    
    # Check losses are 8% of generation
    expected_losses = np.array([800, 880, 960])
    np.testing.assert_array_almost_equal(
        result.losses['Total_Losses'],
        expected_losses
    )


def test_loss_validation():
    """Test loss validation."""
    config = LossConfig(max_loss_percentage=0.15)
    calculator = LossCalculator(config)
    
    # Valid losses (8%)
    generation = np.array([10000])
    consumption = np.array([9200])
    losses = np.array([800])
    
    status = calculator._validate_losses(losses, generation, consumption)
    assert status['valid']
    
    # Invalid losses (20%)
    losses_invalid = np.array([2000])
    status_invalid = calculator._validate_losses(
        losses_invalid, generation, consumption
    )
    assert not status_invalid['valid']


def test_energy_balance():
    """Test energy balance checking."""
    calculator = LossCalculator()
    
    generation = np.array([10000, 11000])
    consumption = np.array([9200, 10120])
    losses = np.array([800, 880])
    
    balance_ok = calculator._check_energy_balance(
        generation, consumption, losses
    )
    assert balance_ok


def test_loss_distribution():
    """Test loss distribution to subsystems."""
    calculator = LossCalculator()
    
    total_losses = np.array([800, 880])
    subsystem_loads = {
        'SE': np.array([4600, 5060]),
        'S': np.array([3680, 4048]),
        'NE': np.array([920, 1012])
    }
    
    distributed = calculator.distribute_losses(total_losses, subsystem_loads)
    
    # Check sum equals total
    sum_distributed = sum(distributed.values())
    np.testing.assert_array_almost_equal(sum_distributed, total_losses)
```

---

## 📝 Technical Notes

### Loss Models
- Percentage model: simple and predictable
- Adaptive model: responds to load conditions
- Historical model: uses past patterns

### Integration
- Losses participate in hierarchical aggregation
- Energy balance constraints ensure consistency
- Validation prevents unrealistic loss values

---

## 🔗 Dependencies

**Depends On:**
- PC-048-06A: Hierarchy Definition System
- PC-052-06A: Constraint Enforcement System

**Blocks:**
- Epic-06B: Advanced Reconciliation Methods
- Complete Epic-06A

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Multiple loss models implemented
- [ ] Loss validation functional
- [ ] Integration with reconciliation working
- [ ] Energy balance maintained
- [ ] Loss distribution implemented
- [ ] Performance overhead <2%
- [ ] Unit tests >85% coverage
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**Previous:** [PC-052-06A: Constraint Enforcement System](PC-052-06A-constraint-enforcer.md)  
**Next:** Epic-06B Advanced Reconciliation Methods
