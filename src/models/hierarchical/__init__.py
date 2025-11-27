"""Hierarchical forecasting models.

This package provides the infrastructure for two-stage hierarchical forecasting
models in the PrevCarga system. Hierarchical models decompose load forecasting
into:
1. Demand mean forecasting (daily average load)
2. Profile forecasting (semi-hourly patterns)

The final semi-hourly load is computed by combining these components:
    load = demand_mean * profile_ratio

Example:
    ```python
    from src.models.hierarchical import BaseHierarchicalModel, ProfileCombiner

    # Create a concrete hierarchical model by subclassing
    class MyHierarchicalModel(BaseHierarchicalModel):
        # Implement required methods and properties
        ...

    # Use the model
    model = MyHierarchicalModel()
    model.fit(X_train, y_train, config)
    predictions = model.predict(X_test)

    # Get decomposition for analysis
    decomp = model.get_decomposition(X_test)
    print(decomp["demand_mean"])  # Daily average predictions
    print(decomp["profiles"])      # Semi-hourly profile ratios
    print(decomp["combined_load"]) # Final semi-hourly load
    ```
"""

from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.models.hierarchical.holt_winters import HoltWintersModel
from src.models.hierarchical.orchestration import HierarchicalTrainer
from src.models.hierarchical.profile_combiner import ProfileCombiner
from src.models.hierarchical.regdin_svm import RegDinSVMModel
from src.models.hierarchical.validation import HierarchicalValidator

__all__ = [
    "BaseHierarchicalModel",
    "HierarchicalTrainer",
    "HierarchicalValidator",
    "HoltWintersModel",
    "ProfileCombiner",
    "RegDinSVMModel",
]
