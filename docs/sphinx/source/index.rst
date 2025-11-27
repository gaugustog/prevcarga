PrevCarga Unified Forecasting System
=====================================

Welcome to PrevCarga's documentation!

PrevCarga is a unified electric load forecasting system that provides accurate
short-term and medium-term load predictions for the Brazilian Interconnected
Power System (SIN). The system uses advanced machine learning models and
hierarchical forecasting strategies to deliver reliable predictions across
multiple forecast horizons (D+0 through D+8).

.. note::
   This documentation covers version 1.0.0 (MVP Release) of PrevCarga.

Features
--------

- **Multi-Model Forecasting**: LightGBM, Random Forest, Holt-Winters, and ARIMA models
- **Hierarchical Reconciliation**: MinT, WLS, OLS, and Shrinkage reconciliation methods
- **Feature Engineering**: Comprehensive temporal, calendar, and lag feature plugins
- **Performance Optimization**: Caching, parallel execution, and feature selection
- **Robust Validation**: Backtesting, cross-validation, and model comparison

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/prevcarga.git
   cd prevcarga

   # Install with uv (recommended)
   uv sync

   # Or with pip
   pip install -e .

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from src.models import LightGBMModel
   from src.features import FeaturePipeline
   from src.reconciliation import MinTReconciler

   # Create and train a model
   model = LightGBMModel()
   model.fit(X_train, y_train)

   # Generate predictions
   predictions = model.predict(X_test)

   # Apply hierarchical reconciliation
   reconciler = MinTReconciler()
   reconciled = reconciler.reconcile(predictions, hierarchy)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   guides/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Project Information
-------------------

- **Version**: 1.0.0 (MVP)
- **License**: Proprietary
- **Source Code**: Internal Repository
- **Issue Tracker**: Internal JIRA

Support
-------

For support and questions:

- Check the :doc:`guides/index` for common issues
- Review the :doc:`api/index` for technical details
- Contact the development team for urgent issues
