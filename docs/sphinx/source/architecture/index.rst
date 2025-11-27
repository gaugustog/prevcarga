Architecture
============

This section describes the architecture and design patterns used in PrevCarga.

.. contents:: Table of Contents
   :local:
   :depth: 2

System Overview
---------------

PrevCarga is designed as a modular system with clear separation of concerns:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                         CLI Interface                           │
   └─────────────────────────────────────────────────────────────────┘
                                    │
   ┌─────────────────────────────────────────────────────────────────┐
   │                      Orchestration Layer                        │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
   │  │  Training   │  │  Prediction │  │  Backtesting│             │
   │  │  Workflow   │  │  Workflow   │  │  Workflow   │             │
   │  └─────────────┘  └─────────────┘  └─────────────┘             │
   └─────────────────────────────────────────────────────────────────┘
                                    │
   ┌─────────────────────────────────────────────────────────────────┐
   │                        Core Components                          │
   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
   │  │  Data    │  │ Features │  │  Models  │  │ Reconcil │       │
   │  │  Module  │  │  Module  │  │  Module  │  │  Module  │       │
   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
   └─────────────────────────────────────────────────────────────────┘
                                    │
   ┌─────────────────────────────────────────────────────────────────┐
   │                       Infrastructure                            │
   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
   │  │ Storage  │  │ Logging  │  │  Cache   │  │Validation│       │
   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
   └─────────────────────────────────────────────────────────────────┘

Design Patterns
---------------

Plugin Pattern
^^^^^^^^^^^^^^

Used extensively in the features module for extensibility:

.. code-block:: python

   from abc import ABC, abstractmethod

   class FeaturePlugin(ABC):
       """Base class for feature plugins."""

       @property
       @abstractmethod
       def name(self) -> str:
           """Plugin name."""
           pass

       @abstractmethod
       def transform(self, df: pd.DataFrame) -> pd.DataFrame:
           """Transform input data."""
           pass

Registry Pattern
^^^^^^^^^^^^^^^^

Used for automatic discovery and registration of components:

.. code-block:: python

   class ModelRegistry:
       """Registry for model classes."""

       _registry: dict[str, type] = {}

       @classmethod
       def register(cls, name: str):
           def decorator(model_class):
               cls._registry[name] = model_class
               return model_class
           return decorator

       @classmethod
       def get(cls, name: str) -> type:
           return cls._registry[name]

Factory Pattern
^^^^^^^^^^^^^^^

Used for creating storage backends and other configurable components:

.. code-block:: python

   class StorageFactory:
       """Factory for creating storage backends."""

       @staticmethod
       def create(config: StorageConfig) -> StorageBackend:
           if config.backend == "local":
               return LocalStorageBackend(config)
           elif config.backend == "s3":
               return S3StorageBackend(config)
           else:
               raise ValueError(f"Unknown backend: {config.backend}")

Module Structure
----------------

Data Module
^^^^^^^^^^^

Responsible for data loading, validation, and preprocessing:

- **Loaders**: Load data from various sources (S3, local files)
- **Validators**: Validate data schemas and quality
- **Preprocessors**: Handle imputation, resampling, and normalization
- **Filters**: Filter and select data subsets

Features Module
^^^^^^^^^^^^^^^

Provides extensible feature engineering:

- **Plugins**: Individual feature generators (temporal, calendar, lag)
- **Pipeline**: Orchestrates plugin execution
- **Registry**: Manages plugin discovery
- **Cache**: Caches computed features

Models Module
^^^^^^^^^^^^^

Contains forecasting models:

- **Base**: Abstract base classes and interfaces
- **End-to-End**: ML models (LightGBM, Random Forest)
- **Demand Mean**: Statistical models (Holt-Winters, ARIMA)
- **Hierarchical**: Hierarchical model implementations
- **Combination**: Model combination strategies

Reconciliation Module
^^^^^^^^^^^^^^^^^^^^^

Implements hierarchical reconciliation:

- **Base**: Abstract reconciler interface
- **MinT**: Minimum Trace reconciler
- **OLS**: Ordinary Least Squares reconciler
- **WLS**: Weighted Least Squares reconciler
- **Shrinkage**: Shrinkage estimator reconciler

Extension Points
----------------

Adding New Features
^^^^^^^^^^^^^^^^^^^

1. Create a new plugin class inheriting from ``FeaturePlugin``
2. Implement ``name`` property and ``transform`` method
3. Register with ``@FeatureRegistry.register()``
4. Add to pipeline configuration

Adding New Models
^^^^^^^^^^^^^^^^^

1. Create a new model class inheriting from ``BaseModel``
2. Implement ``fit``, ``predict``, and serialization methods
3. Register with ``@ModelRegistry.register()``
4. Create configuration class if needed

Adding New Reconcilers
^^^^^^^^^^^^^^^^^^^^^^

1. Create a new reconciler inheriting from ``BaseReconciler``
2. Implement ``fit`` and ``reconcile`` methods
3. Register with reconciler registry
4. Add tests and documentation

Performance Considerations
--------------------------

Caching
^^^^^^^

PrevCarga uses multi-level caching:

- **Feature Cache**: Caches computed features
- **Model Cache**: Caches trained models
- **Prediction Cache**: Caches recent predictions

Parallel Execution
^^^^^^^^^^^^^^^^^^

Supports parallel execution for:

- Feature computation across multiple series
- Model training for multiple horizons
- Backtesting across time windows

Memory Management
^^^^^^^^^^^^^^^^^

- Uses chunked processing for large datasets
- Implements lazy loading where possible
- Provides configuration for memory limits
