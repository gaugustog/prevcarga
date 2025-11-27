Validation Module
=================

The validation module provides comprehensive frameworks for data validation,
model validation, and production readiness assessment.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The validation module supports:

- **Data Validation**: Validate input data quality and completeness
- **Model Validation**: Validate model outputs and predictions
- **Backtesting**: Walk-forward validation with retraining
- **Production Readiness**: Assess system readiness for deployment
- **Epic Acceptance**: Validate acceptance criteria across epics

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   from src.validation import (
       DataValidator,
       ComprehensiveBacktester,
       ProductionReadinessValidator,
   )

   # Data validation
   validator = DataValidator()
   result = validator.validate(data)
   if not result.is_valid:
       print(f"Validation failed: {result.errors}")

   # Production readiness
   readiness = ProductionReadinessValidator()
   report = readiness.validate(system_config)
   print(f"Production ready: {report.is_ready}")

Module Reference
----------------

.. automodule:: src.validation
   :members:
   :undoc-members:
   :show-inheritance:

Backtesting
-----------

ComprehensiveBacktester
^^^^^^^^^^^^^^^^^^^^^^^

Walk-forward backtesting with configurable retraining.

.. autoclass:: src.validation.backtester.ComprehensiveBacktester
   :members:
   :undoc-members:
   :show-inheritance:

BacktestConfig
^^^^^^^^^^^^^^

.. autoclass:: src.validation.backtester.BacktestConfig
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.validation import ComprehensiveBacktester, BacktestConfig

   # Configure backtest
   config = BacktestConfig(
       backtest_year=2024,
       initial_window=365,  # days of initial training data
       step_size=7,  # forecast every 7 days
       horizons=[0, 1, 2, 3, 4, 5, 6, 7, 8],
       retrain_interval=30  # retrain every 30 days
   )

   # Run backtest
   backtester = ComprehensiveBacktester(config)
   results = backtester.execute(model, data)

   # Analyze results
   print(f"Average MAPE: {results.average_mape:.2f}%")
   print(f"Best horizon: D+{results.best_horizon}")

Robustness Testing
------------------

RobustnessTester
^^^^^^^^^^^^^^^^

Test model robustness under various conditions.

.. autoclass:: src.validation.robustness_tester.RobustnessTester
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.validation import RobustnessTester

   tester = RobustnessTester()

   # Test with missing data
   missing_result = tester.test_missing_data(
       model=model,
       data=test_data,
       missing_rates=[0.01, 0.05, 0.10]
   )

   # Test with outliers
   outlier_result = tester.test_outliers(
       model=model,
       data=test_data,
       outlier_rates=[0.01, 0.05]
   )

   # Test with distribution shift
   shift_result = tester.test_distribution_shift(
       model=model,
       data=test_data,
       shift_magnitude=[0.1, 0.2, 0.3]
   )

Production Readiness
--------------------

ProductionReadinessValidator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validate system readiness for production deployment.

.. autoclass:: src.validation.production_readiness.ProductionReadinessValidator
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.validation import ProductionReadinessValidator

   validator = ProductionReadinessValidator()

   # Check all components
   report = validator.validate(
       config=system_config,
       checks=[
           "storage_connectivity",
           "model_availability",
           "data_freshness",
           "memory_requirements",
           "api_health",
       ]
   )

   # Review results
   print(f"Production ready: {report.is_ready}")
   for check in report.checks:
       status = "PASS" if check.passed else "FAIL"
       print(f"  {check.name}: {status}")

Epic Acceptance
---------------

EpicAcceptanceCriteriaValidator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validate acceptance criteria across project epics.

.. autoclass:: src.validation.epic_acceptance_validator.EpicAcceptanceCriteriaValidator
   :members:
   :undoc-members:
   :show-inheritance:

Final Report Generator
----------------------

FinalValidationReportGenerator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate comprehensive validation reports.

.. autoclass:: src.validation.final_report_generator.FinalValidationReportGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.validation import FinalValidationReportGenerator

   generator = FinalValidationReportGenerator()

   # Generate comprehensive report
   report = generator.generate_final_validation_report(
       epic_validation=epic_results,
       production_validation=prod_results,
       quality_validation=quality_results,
       security_validation=security_results,
       robustness_validation=robustness_results,
       config=report_config
   )

   # Export report
   generator.export_report_html(report, "final_report.html")
   generator.export_report_json(report, "final_report.json")

Data Structures
---------------

ValidationResult
^^^^^^^^^^^^^^^^

.. autoclass:: src.validation.data_structures.ValidationResult
   :members:
   :undoc-members:
   :show-inheritance:

ValidationReport
^^^^^^^^^^^^^^^^

.. autoclass:: src.validation.data_structures.ValidationReport
   :members:
   :undoc-members:
   :show-inheritance:

Usage Patterns
--------------

Complete Validation Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.validation import (
       ComprehensiveBacktester,
       RobustnessTester,
       ProductionReadinessValidator,
       FinalValidationReportGenerator,
   )

   # 1. Backtesting
   backtest_config = BacktestConfig(backtest_year=2024)
   backtester = ComprehensiveBacktester(backtest_config)
   backtest_results = backtester.execute(model, data)

   # 2. Robustness testing
   robustness_tester = RobustnessTester()
   robustness_results = robustness_tester.test_all(model, data)

   # 3. Production readiness
   prod_validator = ProductionReadinessValidator()
   prod_results = prod_validator.validate(config)

   # 4. Generate final report
   report_generator = FinalValidationReportGenerator()
   final_report = report_generator.generate_final_validation_report(
       backtest_validation=backtest_results,
       robustness_validation=robustness_results,
       production_validation=prod_results,
   )

   # 5. Export and review
   report_generator.export_report_html(final_report, "validation_report.html")

   # Check approval status
   if final_report.executive_summary.approval_status == "APPROVED":
       print("System validated for production!")
   else:
       print("Issues found:")
       for rec in final_report.recommendations:
           print(f"  - {rec}")

Continuous Validation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.validation import ProductionReadinessValidator
   import schedule

   def daily_validation():
       validator = ProductionReadinessValidator()
       report = validator.validate(config)

       if not report.is_ready:
           send_alert(f"Validation failed: {report.issues}")

   # Schedule daily validation
   schedule.every().day.at("06:00").do(daily_validation)
