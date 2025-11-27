Orchestration Module
====================

The orchestration module provides workflow management and parallel execution
for training, prediction, and backtesting operations.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The orchestration module handles:

- **Training Workflows**: Coordinate model training across areas and models
- **Prediction Workflows**: Generate forecasts with optional reconciliation
- **Backtesting Workflows**: Run historical backtests with walk-forward validation
- **Parallel Execution**: Distribute work across multiple processes

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   from src.orchestration import (
       TrainingWorkflow,
       PredictionWorkflow,
       BacktestingWorkflow,
   )

   # Training workflow
   training = TrainingWorkflow(config)
   training.run(
       models=["lgbm", "random_forest"],
       areas=["SECO_RJ", "SECO_SP"],
       start_date="2023-01-01",
       end_date="2023-12-31",
       parallel=4
   )

   # Prediction workflow
   prediction = PredictionWorkflow(config)
   forecasts = prediction.run(
       models=["lgbm"],
       areas=["SECO_RJ"],
       date="2024-01-15",
       reconcile=True
   )

Training Workflow
-----------------

.. automodule:: src.orchestration.training_workflow
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.orchestration import TrainingWorkflow

   # Create workflow
   workflow = TrainingWorkflow(config)

   # Train single model
   result = workflow.train_model(
       model="lgbm",
       area="SECO_RJ",
       start_date="2023-01-01",
       end_date="2023-12-31"
   )

   print(f"Training MAPE: {result.metrics['mape']:.2f}%")
   print(f"Model saved to: {result.model_path}")

   # Batch training
   results = workflow.run(
       models=["lgbm", "random_forest"],
       areas=["SECO_RJ", "SECO_SP", "S_PR"],
       start_date="2023-01-01",
       end_date="2023-12-31",
       parallel=4
   )

Prediction Workflow
-------------------

.. automodule:: src.orchestration.prediction_workflow
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.orchestration import PredictionWorkflow

   # Create workflow
   workflow = PredictionWorkflow(config)

   # Single prediction
   result = workflow.predict(
       model="lgbm",
       area="SECO_RJ",
       date="2024-01-15"
   )

   print(f"Predictions: {len(result.predictions)} half-hours")

   # Batch prediction with reconciliation
   results = workflow.run(
       models=["lgbm", "random_forest"],
       areas="all",
       date="2024-01-15",
       reconcile=True,
       parallel=4
   )

Backtesting Workflow
--------------------

.. automodule:: src.orchestration.backtesting_workflow
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.orchestration import BacktestingWorkflow

   # Create workflow
   workflow = BacktestingWorkflow(config)

   # Run backtest
   results = workflow.run(
       model="lgbm",
       area="SECO_RJ",
       start_date="2024-01-01",
       end_date="2024-06-30",
       retrain_interval=30,  # days
       initial_training_days=365
   )

   # Analyze results
   for period, metrics in results.items():
       print(f"{period}: MAPE={metrics['mape']:.2f}%")

Configuration Manager
---------------------

.. automodule:: src.orchestration.config_manager
   :members:
   :undoc-members:
   :show-inheritance:

Parallel Executor
-----------------

.. automodule:: src.orchestration.parallel_executor
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.orchestration import ParallelExecutor

   # Create executor
   executor = ParallelExecutor(n_workers=4)

   # Define tasks
   tasks = [
       {"model": "lgbm", "area": "SECO_RJ"},
       {"model": "lgbm", "area": "SECO_SP"},
       {"model": "rf", "area": "SECO_RJ"},
       {"model": "rf", "area": "SECO_SP"},
   ]

   # Execute in parallel
   results = executor.map(train_model, tasks)

Structured Logger
-----------------

.. automodule:: src.orchestration.structured_logger
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.orchestration import StructuredLogger

   # Create logger
   logger = StructuredLogger(
       name="training",
       output_format="json"
   )

   # Log events
   logger.info("Training started", model="lgbm", area="SECO_RJ")
   logger.info("Epoch complete", epoch=100, loss=0.023)
   logger.info("Training complete", duration_seconds=1234)

Usage Patterns
--------------

Complete Training Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.orchestration import (
       TrainingWorkflow,
       PredictionWorkflow,
       ConfigManager,
   )

   # Load configuration
   config_manager = ConfigManager()
   config = config_manager.load("config/config.yaml")

   # 1. Train models
   training = TrainingWorkflow(config)
   training_results = training.run(
       models=["lgbm", "random_forest"],
       areas="all",
       start_date="2023-01-01",
       end_date="2023-12-31",
       parallel=8
   )

   # 2. Generate predictions
   prediction = PredictionWorkflow(config)
   predictions = prediction.run(
       models=["lgbm", "random_forest"],
       areas="all",
       date="2024-01-15",
       reconcile=True
   )

   # 3. Save results
   predictions.save("predictions/2024-01-15.parquet")

Scheduled Operations
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.orchestration import (
       TrainingWorkflow,
       PredictionWorkflow,
   )
   import schedule
   import time

   def daily_prediction():
       workflow = PredictionWorkflow(config)
       workflow.run(
           models="all",
           areas="all",
           date="today",
           reconcile=True
       )

   def weekly_retrain():
       workflow = TrainingWorkflow(config)
       workflow.run(
           models="all",
           areas="all",
           retrain=True
       )

   # Schedule jobs
   schedule.every().day.at("00:30").do(daily_prediction)
   schedule.every().sunday.at("02:00").do(weekly_retrain)

   # Run scheduler
   while True:
       schedule.run_pending()
       time.sleep(60)
