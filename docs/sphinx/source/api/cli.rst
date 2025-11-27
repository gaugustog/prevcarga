CLI Module
==========

The CLI module provides the command-line interface for PrevCarga operations.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

PrevCarga provides a comprehensive CLI built with Click:

- **Training Commands**: Train models for specific areas and date ranges
- **Prediction Commands**: Generate forecasts
- **Evaluation Commands**: Evaluate model performance
- **Data Commands**: Data management and validation
- **Configuration Commands**: View and manage configuration

Quick Reference
---------------

.. code-block:: bash

   # General help
   prevcarga --help

   # Training
   prevcarga train --areas SECO_RJ --models lgbm --start-date 2023-01-01

   # Prediction
   prevcarga predict --date 2024-01-15 --areas all --models all

   # Evaluation
   prevcarga evaluate --start-date 2024-01-01 --end-date 2024-01-31

   # Status
   prevcarga status

CLI Module
----------

.. automodule:: src.cli
   :members:
   :undoc-members:
   :show-inheritance:

Commands
--------

Core Commands
^^^^^^^^^^^^^

.. automodule:: src.cli.commands
   :members:
   :undoc-members:
   :show-inheritance:

Interactive Mode
^^^^^^^^^^^^^^^^

.. automodule:: src.cli.interactive
   :members:
   :undoc-members:
   :show-inheritance:

Validators
^^^^^^^^^^

.. automodule:: src.cli.validators
   :members:
   :undoc-members:
   :show-inheritance:

Progress Display
^^^^^^^^^^^^^^^^

.. automodule:: src.cli.progress
   :members:
   :undoc-members:
   :show-inheritance:

Command Reference
-----------------

prevcarga train
^^^^^^^^^^^^^^^

Train forecasting models.

.. code-block:: bash

   prevcarga train [OPTIONS]

   Options:
     --config PATH           Configuration file path
     --storage-backend TEXT  Storage backend (local, s3)
     --areas TEXT           Target areas (or 'all')
     --models TEXT          Model types (lgbm, rf, all)
     --start-date TEXT      Training start date (YYYY-MM-DD)
     --end-date TEXT        Training end date (YYYY-MM-DD)
     --parallel INTEGER     Number of parallel workers
     --force               Force retrain existing models
     --dry-run             Preview without executing
     --help                 Show help message

Examples:

.. code-block:: bash

   # Train LightGBM for one area
   prevcarga train \
       --areas SECO_RJ \
       --models lgbm \
       --start-date 2023-01-01 \
       --end-date 2023-12-31

   # Train all models for all areas
   prevcarga train \
       --areas all \
       --models all \
       --start-date 2023-01-01 \
       --end-date 2023-12-31 \
       --parallel 8

prevcarga predict
^^^^^^^^^^^^^^^^^

Generate load forecasts.

.. code-block:: bash

   prevcarga predict [OPTIONS]

   Options:
     --config PATH           Configuration file path
     --date TEXT             Prediction date (YYYY-MM-DD)
     --areas TEXT            Target areas (or 'all')
     --models TEXT           Model types (or 'all')
     --mode TEXT             Prediction mode (batch, intraday)
     --reconcile            Apply hierarchical reconciliation
     --output-path PATH      Output directory
     --output-format TEXT    Output format (parquet, csv, json)
     --help                  Show help message

Examples:

.. code-block:: bash

   # Single day prediction
   prevcarga predict \
       --date 2024-01-15 \
       --areas SECO_RJ \
       --models lgbm

   # All areas with reconciliation
   prevcarga predict \
       --date 2024-01-15 \
       --areas all \
       --models all \
       --reconcile

   # Intraday mode
   prevcarga predict \
       --date 2024-01-15 \
       --mode intraday \
       --areas SECO_RJ

prevcarga evaluate
^^^^^^^^^^^^^^^^^^

Evaluate model performance.

.. code-block:: bash

   prevcarga evaluate [OPTIONS]

   Options:
     --config PATH           Configuration file path
     --start-date TEXT       Evaluation start date
     --end-date TEXT         Evaluation end date
     --areas TEXT            Target areas
     --models TEXT           Model types
     --metrics TEXT          Metrics to calculate
     --output PATH           Output file path
     --report PATH           HTML report path
     --help                  Show help message

Examples:

.. code-block:: bash

   # Evaluate LightGBM
   prevcarga evaluate \
       --start-date 2024-01-01 \
       --end-date 2024-01-31 \
       --areas SECO_RJ \
       --models lgbm \
       --metrics mape,mae,rmse

   # Generate HTML report
   prevcarga evaluate \
       --start-date 2024-01-01 \
       --end-date 2024-01-31 \
       --areas all \
       --models all \
       --report evaluation_report.html

prevcarga backtest
^^^^^^^^^^^^^^^^^^

Run historical backtesting.

.. code-block:: bash

   prevcarga backtest [OPTIONS]

   Options:
     --config PATH           Configuration file path
     --start-date TEXT       Backtest start date
     --end-date TEXT         Backtest end date
     --areas TEXT            Target areas
     --models TEXT           Model types
     --retrain-interval INT  Days between retraining
     --report PATH           Output report path
     --help                  Show help message

Examples:

.. code-block:: bash

   # Monthly retraining backtest
   prevcarga backtest \
       --start-date 2024-01-01 \
       --end-date 2024-06-30 \
       --areas SECO_RJ \
       --models lgbm \
       --retrain-interval 30 \
       --report backtest_report.html

prevcarga status
^^^^^^^^^^^^^^^^

Show system status.

.. code-block:: bash

   prevcarga status [OPTIONS]

   Options:
     --config PATH           Configuration file path
     --verbose              Show detailed status
     --help                  Show help message

Output:

.. code-block:: text

   PrevCarga System Status
   =======================
   Version: 1.0.0
   Python: 3.11.7

   Storage:
     Backend: s3
     Bucket: prevcarga-data
     Access: Connected

   Models Available:
     lgbm: 17 areas
     random_forest: 17 areas

   System Health: OK

prevcarga config
^^^^^^^^^^^^^^^^

Configuration management.

.. code-block:: bash

   # Show current configuration
   prevcarga config show

   # Validate configuration
   prevcarga config validate --config config.yaml

Exit Codes
----------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Code
     - Description
   * - 0
     - Success
   * - 1
     - General error
   * - 2
     - Invalid arguments
   * - 3
     - Configuration error
   * - 4
     - Data not found
   * - 5
     - Model not found

Extending the CLI
-----------------

Add custom commands:

.. code-block:: python

   import click
   from src.cli.core import cli

   @cli.command()
   @click.option("--custom-option", help="Custom option")
   def my_command(custom_option):
       """My custom command."""
       click.echo(f"Running with: {custom_option}")

   # Register command
   if __name__ == "__main__":
       cli()
