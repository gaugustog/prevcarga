Tutorials
=========

This section contains step-by-step tutorials for common tasks with PrevCarga.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   getting-started
   training-workflow
   prediction-workflow

.. contents:: Quick Examples
   :local:
   :depth: 2

Overview
--------

PrevCarga provides comprehensive tutorials to help you master the system:

- **Getting Started**: Installation, configuration, and first prediction
- **Training Workflow**: Model training, cross-validation, and hyperparameter tuning
- **Prediction Workflow**: Generating forecasts, ensemble methods, and reconciliation

Start with the :doc:`getting-started` tutorial if you're new to PrevCarga.

Quick Examples
--------------

Your First Prediction
^^^^^^^^^^^^^^^^^^^^^

This quick example walks you through making your first load prediction with PrevCarga.

.. code-block:: python

   import pandas as pd
   from src.models.end_to_end import LightGBMModel
   from src.features.pipeline import FeaturePipeline
   from src.features.plugins import TemporalFeaturePlugin, LagFeaturePlugin

   # 1. Load your data
   data = pd.read_csv("load_data.csv", parse_dates=["timestamp"])

   # 2. Create feature pipeline
   pipeline = FeaturePipeline()
   pipeline.add_plugin(TemporalFeaturePlugin())
   pipeline.add_plugin(LagFeaturePlugin(lags=[24, 48, 168]))

   # 3. Generate features
   features = pipeline.transform(data)

   # 4. Split data
   train = features[features["timestamp"] < "2024-01-01"]
   test = features[features["timestamp"] >= "2024-01-01"]

   # 5. Train model
   model = LightGBMModel()
   model.fit(train.drop(columns=["load"]), train["load"])

   # 6. Make predictions
   predictions = model.predict(test.drop(columns=["load"]))

Feature Engineering
-------------------

Using Feature Plugins
^^^^^^^^^^^^^^^^^^^^^

PrevCarga uses a plugin architecture for feature engineering:

.. code-block:: python

   from src.features.pipeline import FeaturePipeline
   from src.features.plugins import (
       TemporalFeaturePlugin,
       CalendarFeaturePlugin,
       LagFeaturePlugin,
       CyclicalEncodingPlugin,
   )

   # Create pipeline
   pipeline = FeaturePipeline()

   # Add temporal features (hour, day, month, etc.)
   pipeline.add_plugin(TemporalFeaturePlugin())

   # Add Brazilian calendar features (holidays, bridge days)
   pipeline.add_plugin(CalendarFeaturePlugin(country="BR"))

   # Add lag features
   pipeline.add_plugin(LagFeaturePlugin(
       lags=[1, 24, 48, 168],
       rolling_windows=[24, 168],
   ))

   # Add cyclical encoding
   pipeline.add_plugin(CyclicalEncodingPlugin(
       columns=["hour", "day_of_week", "month"]
   ))

   # Transform data
   features = pipeline.transform(data)

Hierarchical Reconciliation
---------------------------

Reconciling Forecasts
^^^^^^^^^^^^^^^^^^^^^

Apply hierarchical reconciliation to ensure forecast coherence:

.. code-block:: python

   from src.reconciliation import MinTReconciler, HierarchyMatrix

   # Define hierarchy
   hierarchy = HierarchyMatrix.from_config({
       "total": ["SECO", "S", "NE", "N"],
   })

   # Create reconciler
   reconciler = MinTReconciler(
       method="ols",
       covariance_method="shrink",
   )

   # Fit reconciler
   reconciler.fit(historical_forecasts, hierarchy)

   # Reconcile new forecasts
   reconciled = reconciler.reconcile(base_forecasts)

Model Comparison
----------------

Comparing Models
^^^^^^^^^^^^^^^^

Compare multiple models using the evaluation framework:

.. code-block:: python

   from src.evaluation import ModelComparator
   from src.models.end_to_end import LightGBMModel, RandomForestModel
   from src.models.demand_mean import HoltWintersModel

   # Define models
   models = {
       "lightgbm": LightGBMModel(),
       "random_forest": RandomForestModel(),
       "holt_winters": HoltWintersModel(),
   }

   # Create comparator
   comparator = ModelComparator(models)

   # Run comparison
   results = comparator.compare(
       X_train, y_train,
       X_test, y_test,
       metrics=["mape", "rmse", "mae"],
   )

   # View results
   print(results.summary())
   results.plot_comparison()

Backtesting
-----------

Walk-Forward Validation
^^^^^^^^^^^^^^^^^^^^^^^

Perform comprehensive backtesting:

.. code-block:: python

   from src.validation import ComprehensiveBacktester, BacktestConfig

   # Configure backtest
   config = BacktestConfig(
       backtest_year=2024,
       initial_window=365,
       step_size=7,
       horizons=[1, 2, 3, 4, 5, 6, 7, 8],
   )

   # Create backtester
   backtester = ComprehensiveBacktester(config)

   # Run backtest
   results = await backtester.execute_yearly_backtest(model)

   # Analyze results
   print(f"Average MAPE: {results.average_mape:.2f}%")
   results.plot_performance_over_time()
