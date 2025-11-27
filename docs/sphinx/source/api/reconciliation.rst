Reconciliation Module
=====================

The reconciliation module provides hierarchical forecast reconciliation methods
to ensure forecasts are coherent across different aggregation levels.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

Hierarchical reconciliation ensures that forecasts at different levels of the
hierarchy (areas, subsystems, national) are mathematically consistent:

.. code-block:: text

   National (SIN) = Sum of all Subsystems
   Subsystem = Sum of Areas + Losses

Supported reconciliation methods:

- **Bottom-Up**: Simple aggregation from lowest level
- **OLS**: Ordinary Least Squares reconciliation
- **WLS**: Weighted Least Squares (variance-weighted)
- **MinT**: Minimum Trace reconciliation (optimal)

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   from src.reconciliation import (
       MinTReconciler,
       HierarchyMatrix,
       HierarchyNode,
   )

   # Define hierarchy
   hierarchy = HierarchyMatrix()
   hierarchy.add_node(HierarchyNode("SIN", children=["SECO", "S", "NE", "N"]))
   hierarchy.add_node(HierarchyNode("SECO", children=["SECO_RJ", "SECO_SP"]))

   # Create reconciler
   reconciler = MinTReconciler(method="ols")
   reconciler.fit(hierarchy, historical_errors)

   # Reconcile forecasts
   reconciled = reconciler.reconcile(base_forecasts)

Hierarchy Definition
--------------------

HierarchyMatrix
^^^^^^^^^^^^^^^

.. autoclass:: src.reconciliation.hierarchy.HierarchyMatrix
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

.. automodule:: src.reconciliation.data_structures
   :members:
   :undoc-members:
   :show-inheritance:

Base Reconciler
---------------

.. autoclass:: src.reconciliation.base_reconciler.BaseReconciler
   :members:
   :undoc-members:
   :show-inheritance:

Reconciliation Methods
----------------------

MinT Reconciler
^^^^^^^^^^^^^^^

The MinT (Minimum Trace) reconciler provides optimal reconciliation
by minimizing the trace of the covariance matrix.

.. autoclass:: src.reconciliation.mint_reconciler.MinTReconciler
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.reconciliation import MinTReconciler

   # Create reconciler with OLS covariance estimation
   reconciler = MinTReconciler(
       method="ols",
       nonneg=True  # Ensure non-negative reconciled values
   )

   # Fit on historical data
   reconciler.fit(hierarchy, historical_errors)

   # Reconcile new forecasts
   reconciled_forecasts = reconciler.reconcile(base_forecasts)

OLS Reconciler
^^^^^^^^^^^^^^

Ordinary Least Squares reconciliation.

.. autoclass:: src.reconciliation.ols_reconciler.OLSReconciler
   :members:
   :undoc-members:
   :show-inheritance:

WLS Reconciler
^^^^^^^^^^^^^^

Weighted Least Squares reconciliation with variance-based weights.

.. autoclass:: src.reconciliation.wls_reconciler.WLSReconciler
   :members:
   :undoc-members:
   :show-inheritance:

Shrinkage Reconciler
^^^^^^^^^^^^^^^^^^^^

Reconciler with shrinkage estimation for covariance matrix.

.. autoclass:: src.reconciliation.shrinkage_reconciler.ShrinkageReconciler
   :members:
   :undoc-members:
   :show-inheritance:

Hierarchy Validation
--------------------

.. autoclass:: src.reconciliation.hierarchy_validator.HierarchyValidator
   :members:
   :undoc-members:
   :show-inheritance:

Reconciler Selection
--------------------

.. autoclass:: src.reconciliation.reconciler_selector.ReconcilerSelector
   :members:
   :undoc-members:
   :show-inheritance:

Method Comparison
-----------------

.. autoclass:: src.reconciliation.method_comparator.MethodComparator
   :members:
   :undoc-members:
   :show-inheritance:

Quality Analysis
----------------

.. autoclass:: src.reconciliation.quality_analyzer.QualityAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Constraints
-----------

.. automodule:: src.reconciliation.constraints
   :members:
   :undoc-members:
   :show-inheritance:

Loss Functions
--------------

.. automodule:: src.reconciliation.losses
   :members:
   :undoc-members:
   :show-inheritance:

Usage Patterns
--------------

Brazilian Power Grid Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.reconciliation import HierarchyMatrix, HierarchyNode

   # Define Brazilian SIN hierarchy
   hierarchy = HierarchyMatrix()

   # National level
   hierarchy.add_node(HierarchyNode(
       name="SIN",
       children=["SECO", "S", "NE", "N"]
   ))

   # Subsystem level
   hierarchy.add_node(HierarchyNode(
       name="SECO",
       children=["SECO_RJ", "SECO_SP", "SECO_MG", "SECO_ES"]
   ))

   hierarchy.add_node(HierarchyNode(
       name="S",
       children=["S_PR", "S_SC", "S_RS"]
   ))

   # Add more subsystems...

   # Validate hierarchy
   validator = HierarchyValidator()
   is_valid = validator.validate(hierarchy)

Comparing Reconciliation Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.reconciliation import (
       MethodComparator,
       MinTReconciler,
       OLSReconciler,
       WLSReconciler,
   )

   # Create reconcilers
   reconcilers = {
       "mint_ols": MinTReconciler(method="ols"),
       "mint_wls": MinTReconciler(method="wls"),
       "ols": OLSReconciler(),
       "wls": WLSReconciler(),
   }

   # Compare methods
   comparator = MethodComparator(reconcilers)
   results = comparator.compare(
       hierarchy=hierarchy,
       base_forecasts=forecasts,
       actuals=actuals,
       metrics=["mape", "rmse", "coherence_error"]
   )

   # Print comparison
   print(results.summary())
