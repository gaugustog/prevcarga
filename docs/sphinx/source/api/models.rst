Models Module
=============

The models module provides machine learning models for electric load forecasting,
including end-to-end models and hierarchical approaches.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

PrevCarga supports multiple model architectures:

- **End-to-End Models**: Direct prediction from features to load values (LightGBM, Random Forest)
- **Hierarchical Models**: Decompose forecasting into daily mean and hourly profile
- **Model Registry**: Central management of model classes and instances

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   from src.models import LGBMModel, LGBMConfig, ModelRegistry

   # Create model with configuration
   config = LGBMConfig(
       n_estimators=1000,
       learning_rate=0.05,
       max_depth=10,
       num_leaves=31
   )
   model = LGBMModel(config)

   # Train model
   model.fit(X_train, y_train, X_val, y_val)

   # Make predictions
   predictions = model.predict(X_test)

   # Save model
   model.save("models/lgbm_SECO_RJ.pkl")

Base Classes
------------

BaseModel
^^^^^^^^^

.. autoclass:: src.models.base.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:

ModelMetadata
^^^^^^^^^^^^^

.. autoclass:: src.models.base.ModelMetadata
   :members:
   :undoc-members:
   :show-inheritance:

SemanticVersion
^^^^^^^^^^^^^^^

.. autoclass:: src.models.base.SemanticVersion
   :members:
   :undoc-members:
   :show-inheritance:

Model Registry
--------------

ModelRegistry
^^^^^^^^^^^^^

.. autoclass:: src.models.base.ModelRegistry
   :members:
   :undoc-members:
   :show-inheritance:

Registry Functions
^^^^^^^^^^^^^^^^^^

.. autofunction:: src.models.base.register_model_class

.. autofunction:: src.models.base.get_model_class

End-to-End Models
-----------------

LGBMModel
^^^^^^^^^

The LightGBM-based model for direct load forecasting.

.. autoclass:: src.models.end_to_end.LGBMModel
   :members:
   :undoc-members:
   :show-inheritance:

Example usage:

.. code-block:: python

   from src.models.end_to_end import LGBMModel
   from src.models.config import LGBMConfig

   # Configure model
   config = LGBMConfig(
       n_estimators=1000,
       learning_rate=0.05,
       max_depth=10,
       num_leaves=31,
       min_child_samples=20,
       subsample=0.8,
       colsample_bytree=0.8,
       early_stopping_rounds=50
   )

   # Create and train
   model = LGBMModel(config)
   model.fit(X_train, y_train, X_val, y_val)

   # Predict
   predictions = model.predict(X_test)

   # Get feature importance
   importance = model.feature_importance()

Model Configuration
-------------------

LGBMConfig
^^^^^^^^^^

.. autoclass:: src.models.config.LGBMConfig
   :members:
   :undoc-members:
   :show-inheritance:

Configuration example:

.. code-block:: python

   from src.models.config import LGBMConfig

   config = LGBMConfig(
       # Core parameters
       n_estimators=1000,
       learning_rate=0.05,
       max_depth=10,
       num_leaves=31,

       # Regularization
       min_child_samples=20,
       subsample=0.8,
       colsample_bytree=0.8,
       reg_alpha=0.0,
       reg_lambda=0.0,

       # Training
       early_stopping_rounds=50,
       verbose=-1
   )

Model Serialization
-------------------

.. automodule:: src.models.serialization
   :members:
   :undoc-members:
   :show-inheritance:

Hierarchical Models
-------------------

The hierarchical models decompose forecasting into two stages:

1. **Daily Mean (DM)**: Predict the average daily load
2. **Profile**: Predict the hourly distribution within the day

.. note::
   Hierarchical models are useful when the daily pattern is stable
   but the daily total varies.

Demand Mean Models
^^^^^^^^^^^^^^^^^^

.. automodule:: src.models.demand_mean
   :members:
   :undoc-members:
   :show-inheritance:

Profile Models
^^^^^^^^^^^^^^

.. automodule:: src.models.profile
   :members:
   :undoc-members:
   :show-inheritance:

Model Combination
-----------------

.. automodule:: src.models.combination
   :members:
   :undoc-members:
   :show-inheritance:

Usage Patterns
--------------

Training with Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.model_selection import TimeSeriesSplit
   from src.models import LGBMModel, LGBMConfig

   # Time series cross-validation
   tscv = TimeSeriesSplit(n_splits=5)

   results = []
   for train_idx, val_idx in tscv.split(X):
       X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
       y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

       model = LGBMModel(LGBMConfig())
       model.fit(X_train, y_train, X_val, y_val)

       predictions = model.predict(X_val)
       mape = np.mean(np.abs((y_val - predictions) / y_val)) * 100
       results.append(mape)

   print(f"Average MAPE: {np.mean(results):.2f}%")

Model Persistence
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.models import LGBMModel

   # Save model
   model.save("models/lgbm_SECO_RJ.pkl")

   # Load model
   loaded_model = LGBMModel.load("models/lgbm_SECO_RJ.pkl")

   # Verify
   predictions = loaded_model.predict(X_test)

Feature Importance Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get feature importance
   importance = model.feature_importance()

   # Sort by importance
   sorted_importance = dict(
       sorted(importance.items(), key=lambda x: x[1], reverse=True)
   )

   # Print top 10 features
   for feature, score in list(sorted_importance.items())[:10]:
       print(f"{feature}: {score:.4f}")
