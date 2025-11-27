Evaluation Module
=================

The evaluation module provides tools for model evaluation, comparison,
and performance analysis.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The evaluation module supports:

- **Metrics Calculation**: MAPE, MAE, RMSE, and custom metrics
- **Model Comparison**: Compare multiple models side-by-side
- **Drift Detection**: Monitor model performance degradation
- **Report Generation**: Generate comprehensive evaluation reports

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   from src.evaluation import (
       ModelComparator,
       MetricsCalculator,
       ReportGenerator,
   )

   # Calculate metrics
   calculator = MetricsCalculator()
   metrics = calculator.calculate(predictions, actuals)
   print(f"MAPE: {metrics['mape']:.2f}%")

   # Compare models
   comparator = ModelComparator()
   comparison = comparator.compare(
       models={"lgbm": lgbm_preds, "rf": rf_preds},
       actuals=actuals
   )

Metrics
-------

MetricsCalculator
^^^^^^^^^^^^^^^^^

.. automodule:: src.evaluation.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Available metrics:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Description
     - Formula
   * - MAPE
     - Mean Absolute Percentage Error
     - ``mean(abs((actual - pred) / actual)) * 100``
   * - MAE
     - Mean Absolute Error
     - ``mean(abs(actual - pred))``
   * - RMSE
     - Root Mean Square Error
     - ``sqrt(mean((actual - pred)^2))``
   * - Bias
     - Systematic error
     - ``mean(pred - actual)``

Model Comparison
----------------

ModelComparator
^^^^^^^^^^^^^^^

.. autoclass:: src.evaluation.model_comparator.ModelComparator
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.evaluation import ModelComparator

   # Create comparator
   comparator = ModelComparator()

   # Compare multiple models
   results = comparator.compare(
       models={
           "lgbm": lgbm_predictions,
           "random_forest": rf_predictions,
           "regdin_svm": svm_predictions,
       },
       actuals=actual_values,
       metrics=["mape", "mae", "rmse"]
   )

   # Print comparison table
   print(results.to_dataframe())

   # Get best model
   best = results.get_best_model(metric="mape")
   print(f"Best model: {best}")

Drift Detection
---------------

.. automodule:: src.evaluation.drift
   :members:
   :undoc-members:
   :show-inheritance:

Drift detection monitors model performance over time:

.. code-block:: python

   from src.evaluation.drift import DriftDetector

   # Create detector
   detector = DriftDetector(
       baseline_window=30,  # days
       threshold=0.1  # 10% performance degradation
   )

   # Check for drift
   drift_detected = detector.check(
       predictions=recent_predictions,
       actuals=recent_actuals,
       baseline_metrics=baseline_metrics
   )

   if drift_detected:
       print("Model drift detected! Consider retraining.")

Horizon Analysis
----------------

.. automodule:: src.evaluation.horizon
   :members:
   :undoc-members:
   :show-inheritance:

Analyze performance by forecast horizon:

.. code-block:: python

   from src.evaluation.horizon import HorizonAnalyzer

   analyzer = HorizonAnalyzer()
   results = analyzer.analyze(
       predictions=predictions,
       actuals=actuals,
       horizons=[0, 1, 2, 3, 4, 5, 6, 7, 8]
   )

   # Performance typically degrades with horizon
   for horizon, metrics in results.items():
       print(f"D+{horizon}: MAPE={metrics['mape']:.2f}%")

Time Period Analysis
--------------------

.. automodule:: src.evaluation.time_period
   :members:
   :undoc-members:
   :show-inheritance:

Analyze performance by time periods:

.. code-block:: python

   from src.evaluation.time_period import TimePeriodAnalyzer

   analyzer = TimePeriodAnalyzer()

   # By hour of day
   hourly = analyzer.by_hour(predictions, actuals)

   # By day of week
   daily = analyzer.by_day_of_week(predictions, actuals)

   # By peak/off-peak
   peak = analyzer.by_period(
       predictions, actuals,
       periods={"peak": (18, 21), "off_peak": (0, 6)}
   )

Percentile Analysis
-------------------

.. automodule:: src.evaluation.percentile
   :members:
   :undoc-members:
   :show-inheritance:

Analyze error distribution:

.. code-block:: python

   from src.evaluation.percentile import PercentileAnalyzer

   analyzer = PercentileAnalyzer()
   results = analyzer.analyze(
       predictions=predictions,
       actuals=actuals,
       percentiles=[5, 25, 50, 75, 95]
   )

   print(f"Median error: {results['p50']:.2f}")
   print(f"95th percentile: {results['p95']:.2f}")

Report Generation
-----------------

.. automodule:: src.evaluation.report_generator
   :members:
   :undoc-members:
   :show-inheritance:

Generate comprehensive evaluation reports:

.. code-block:: python

   from src.evaluation.report_generator import ReportGenerator

   generator = ReportGenerator()

   # Generate HTML report
   report = generator.generate(
       predictions=predictions,
       actuals=actuals,
       model_name="lgbm",
       area="SECO_RJ",
       period="2024-01"
   )

   # Save report
   report.save_html("evaluation_report.html")
   report.save_json("evaluation_report.json")

Usage Patterns
--------------

Complete Evaluation Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.evaluation import (
       MetricsCalculator,
       ModelComparator,
       HorizonAnalyzer,
       TimePeriodAnalyzer,
       ReportGenerator,
   )

   # 1. Calculate basic metrics
   calculator = MetricsCalculator()
   metrics = calculator.calculate(predictions, actuals)

   # 2. Analyze by horizon
   horizon_analyzer = HorizonAnalyzer()
   horizon_results = horizon_analyzer.analyze(predictions, actuals)

   # 3. Analyze by time period
   period_analyzer = TimePeriodAnalyzer()
   period_results = period_analyzer.by_hour(predictions, actuals)

   # 4. Generate report
   generator = ReportGenerator()
   report = generator.generate(
       predictions=predictions,
       actuals=actuals,
       additional_metrics={
           "horizon_analysis": horizon_results,
           "period_analysis": period_results,
       }
   )

   report.save_html("comprehensive_evaluation.html")

Backtesting Integration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.evaluation import MetricsCalculator
   from src.validation import ComprehensiveBacktester

   # Run backtest
   backtester = ComprehensiveBacktester(config)
   backtest_results = backtester.execute(model)

   # Evaluate backtest results
   calculator = MetricsCalculator()
   for period, predictions in backtest_results.items():
       metrics = calculator.calculate(predictions, actuals[period])
       print(f"{period}: MAPE={metrics['mape']:.2f}%")
