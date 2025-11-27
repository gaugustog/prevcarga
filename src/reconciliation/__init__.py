"""Hierarchical reconciliation methods for electric load forecasting.

This module provides the foundational infrastructure for hierarchical forecast
reconciliation in the PrevCarga system. It supports the Brazilian electric grid
hierarchy with 4 subsystems (SE, S, NE, N), 17 areas, 4 loss components, and
1 national level.

Key Components:
- HierarchyDefinition: Loads and manages hierarchy structure from YAML
- HierarchyNode: Represents a single node in the hierarchy
- BaseReconciler: Abstract base class for reconciliation methods
- ReconciliationResult: Container for reconciled forecasts and metadata
- ReconciliationConfig: Configuration for reconciliation methods
- Aggregation matrix construction for reconciliation

Example:
    ```python
    from src.reconciliation import (
        HierarchyDefinition,
        BaseReconciler,
        ReconciliationConfig,
        ReconciliationResult,
    )
    import pandas as pd

    # Load hierarchy from configuration
    hierarchy = HierarchyDefinition("config/hierarchy_brazil.yaml")

    # Validate structure
    errors = hierarchy.validate_structure()
    if not errors:
        print("Hierarchy valid!")

    # Get aggregation matrix for reconciliation
    S = hierarchy.aggregation_matrix
    print(f"Aggregation matrix shape: {S.shape}")

    # Configure reconciliation
    config = ReconciliationConfig(
        method="mint",
        non_negative=True,
        check_coherence=True
    )

    # Create reconciler (subclass implements reconcile method)
    reconciler = MintReconciler(config=config)

    # Reconcile forecasts
    result = reconciler.reconcile(base_forecasts, hierarchy)
    print(f"Coherence error: {result.metadata['coherence_after']:.6e}")
    ```
"""

from src.reconciliation.base_reconciler import BaseReconciler
from src.reconciliation.data_structures import ReconciliationConfig, ReconciliationResult
from src.reconciliation.hierarchy import HierarchyDefinition, HierarchyNode
from src.reconciliation.hierarchy_validator import (
    ForecastValidator,
    HierarchyValidator,
    ValidationConfig,
    ValidationLevel,
    ValidationResult,
    ValidationSeverity,
)
from src.reconciliation.losses import (
    LossCalculationResult,
    LossCalculator,
    LossConfig,
    LossIntegrator,
    LossModel,
)
from src.reconciliation.mint_reconciler import MintReconciler
from src.reconciliation.ols_reconciler import (
    FactorCovarianceEstimator,
    LedoitWolfEstimator,
    OLSCovarianceMethod,
    OLSReconciler,
    SampleCovarianceEstimator,
)
from src.reconciliation.shrinkage_reconciler import ShrinkageReconciler, ShrinkageTarget
from src.reconciliation.method_comparator import (
    ComparatorConfig,
    ComparisonReport,
    CVFoldResult,
    PairwiseTest,
    ReconciliationComparator,
)
from src.reconciliation.quality_analyzer import (
    QualityConfig,
    QualityMetrics,
    QualityReport,
    ReconciliationQualityAnalyzer,
    StatisticalTests,
)
from src.reconciliation.reconciler_selector import (
    MethodPerformance,
    ReconcilerSelector,
    SelectionResult,
    SelectorConfig,
)
from src.reconciliation.wls_reconciler import (
    VarianceWeightCalculator,
    WeightDiagnostics,
    WeightingScheme,
    WLSConfig,
    WLSReconciler,
)

__all__ = [
    "HierarchyDefinition",
    "HierarchyNode",
    "BaseReconciler",
    "ReconciliationConfig",
    "ReconciliationResult",
    "MintReconciler",
    "ShrinkageReconciler",
    "ShrinkageTarget",
    "HierarchyValidator",
    "ForecastValidator",
    "ValidationConfig",
    "ValidationLevel",
    "ValidationResult",
    "ValidationSeverity",
    "LossModel",
    "LossConfig",
    "LossCalculationResult",
    "LossCalculator",
    "LossIntegrator",
    "OLSReconciler",
    "OLSCovarianceMethod",
    "SampleCovarianceEstimator",
    "LedoitWolfEstimator",
    "FactorCovarianceEstimator",
    "VarianceWeightCalculator",
    "WeightDiagnostics",
    "WeightingScheme",
    "WLSConfig",
    "WLSReconciler",
    "MethodPerformance",
    "ReconcilerSelector",
    "SelectionResult",
    "SelectorConfig",
    "QualityConfig",
    "QualityMetrics",
    "QualityReport",
    "ReconciliationQualityAnalyzer",
    "StatisticalTests",
    "ComparatorConfig",
    "ComparisonReport",
    "CVFoldResult",
    "PairwiseTest",
    "ReconciliationComparator",
]
