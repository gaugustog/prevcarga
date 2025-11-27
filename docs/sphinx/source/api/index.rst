API Reference
=============

Complete API reference for PrevCarga programmatic usage.

This documentation is automatically generated from docstrings in the source code.
All public APIs are documented with type hints and examples.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   features
   models
   reconciliation
   evaluation

.. toctree::
   :maxdepth: 2
   :caption: Infrastructure

   orchestration
   storage
   cli
   validation

.. contents:: Quick Navigation
   :local:
   :depth: 1

Overview
--------

PrevCarga provides a comprehensive Python API for electric load forecasting:

- **Features**: Plugin-based feature engineering with temporal, calendar, and signal processing features
- **Models**: Machine learning models including LightGBM, Random Forest, and hierarchical approaches
- **Reconciliation**: Hierarchical forecast reconciliation (MinT, OLS, WLS)
- **Orchestration**: Workflow management for training and prediction
- **Storage**: Storage backends for local and S3 data persistence

Quick Start
-----------

Feature Engineering
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.features import (
       PluginRegistry,
       FeaturePipeline,
       TemporalFeaturesPlugin,
       LagFeaturesPlugin,
   )

   # Create feature pipeline
   pipeline = FeaturePipeline()
   pipeline.add_plugin(TemporalFeaturesPlugin())
   pipeline.add_plugin(LagFeaturesPlugin(lags=[1, 24, 48, 168]))

   # Transform data
   features = pipeline.transform(data)

Model Training
^^^^^^^^^^^^^^

.. code-block:: python

   from src.models import LGBMModel, LGBMConfig

   # Create model with configuration
   config = LGBMConfig(
       n_estimators=1000,
       learning_rate=0.05,
       max_depth=10
   )
   model = LGBMModel(config)

   # Train model
   model.fit(X_train, y_train)

   # Make predictions
   predictions = model.predict(X_test)

Hierarchical Reconciliation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.reconciliation import MinTReconciler, HierarchyMatrix

   # Define hierarchy
   hierarchy = HierarchyMatrix.from_dict({
       "SIN": ["SECO", "S", "NE", "N"]
   })

   # Create reconciler
   reconciler = MinTReconciler(method="ols")
   reconciler.fit(hierarchy)

   # Reconcile forecasts
   reconciled = reconciler.reconcile(base_forecasts)

Workflow Orchestration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.orchestration import TrainingWorkflow, PredictionWorkflow

   # Training workflow
   training = TrainingWorkflow(config)
   result = training.run(
       models=["lgbm"],
       areas=["SECO_RJ"],
       start_date="2023-01-01",
       end_date="2023-12-31"
   )

   # Prediction workflow
   prediction = PredictionWorkflow(config)
   forecasts = prediction.run(
       models=["lgbm"],
       areas=["SECO_RJ"],
       date="2024-01-15"
   )

Module Index
------------

Core Modules
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :doc:`features`
     - Plugin-based feature engineering framework
   * - :doc:`models`
     - Forecasting models (LightGBM, Random Forest, etc.)
   * - :doc:`reconciliation`
     - Hierarchical forecast reconciliation methods
   * - :doc:`evaluation`
     - Model evaluation and comparison tools

Infrastructure Modules
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :doc:`orchestration`
     - Workflow management and parallel execution
   * - :doc:`storage`
     - Storage backends (local, S3)
   * - :doc:`cli`
     - Command-line interface
   * - :doc:`validation`
     - Data and model validation frameworks

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
