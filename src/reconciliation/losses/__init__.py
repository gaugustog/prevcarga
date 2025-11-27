"""Transmission loss calculation for hierarchical reconciliation.

This module provides loss calculation and validation for the Brazilian
electric grid. Supports multiple loss models and integrates with the
hierarchical reconciliation framework.

Key Components:
- LossModel: Enumeration of loss calculation models
- LossConfig: Configuration for loss calculations
- LossCalculationResult: Container for calculated losses
- LossCalculator: Main loss calculation engine
- LossIntegrator: Integrates losses with hierarchy reconciliation

Example:
    ```python
    from src.reconciliation.losses import (
        LossCalculator,
        LossConfig,
        LossModel,
    )
    import numpy as np

    # Create calculator with percentage model
    config = LossConfig(
        loss_model=LossModel.PERCENTAGE,
        default_loss_percentage=0.08  # 8% typical for Brazil
    )
    calculator = LossCalculator(config)

    # Calculate losses
    forecasts = {
        'Generation': np.array([10000, 11000, 12000]),
        'Consumption': np.array([9200, 10120, 11040])
    }

    result = calculator.calculate_losses(forecasts)
    print(f"Losses: {result.losses['Total_Losses']}")
    print(f"Loss %: {result.loss_percentages['Total_Losses']:.1f}%")
    ```
"""

from src.reconciliation.losses.loss_calculator import (
    LossCalculationResult,
    LossCalculator,
    LossConfig,
    LossIntegrator,
    LossModel,
)

__all__ = [
    "LossCalculationResult",
    "LossCalculator",
    "LossConfig",
    "LossIntegrator",
    "LossModel",
]
