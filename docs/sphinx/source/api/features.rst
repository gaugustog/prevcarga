Features Module
===============

The features module provides a plugin-based architecture for feature engineering
in the PrevCarga electric load forecasting system.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The feature engineering system is built around a plugin architecture that allows:

- **Modular Features**: Each feature type is implemented as a separate plugin
- **Composable Pipelines**: Plugins can be combined into pipelines
- **Validated Outputs**: Feature outputs are validated for consistency
- **Extensibility**: Custom plugins can be easily added

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   from src.features import (
       FeaturePipeline,
       TemporalFeaturesPlugin,
       LagFeaturesPlugin,
       CyclicalEncodingPlugin,
   )

   # Create pipeline
   pipeline = FeaturePipeline()
   pipeline.add_plugin(TemporalFeaturesPlugin())
   pipeline.add_plugin(LagFeaturesPlugin(lags=[1, 24, 48, 168]))
   pipeline.add_plugin(CyclicalEncodingPlugin(columns=["hour", "day_of_week"]))

   # Transform data
   features = pipeline.transform(input_data)
   print(f"Generated {len(pipeline.get_feature_names())} features")

Base Classes
------------

BaseFeaturePlugin
^^^^^^^^^^^^^^^^^

.. autoclass:: src.features.base.plugin.BaseFeaturePlugin
   :members:
   :undoc-members:
   :show-inheritance:

PluginConfig
^^^^^^^^^^^^

.. autoclass:: src.features.base.config.PluginConfig
   :members:
   :undoc-members:
   :show-inheritance:

Feature Pipeline
----------------

FeaturePipeline
^^^^^^^^^^^^^^^

.. autoclass:: src.features.pipeline.FeaturePipeline
   :members:
   :undoc-members:
   :show-inheritance:

PipelineConfig
^^^^^^^^^^^^^^

.. autoclass:: src.features.pipeline.PipelineConfig
   :members:
   :undoc-members:
   :show-inheritance:

PipelineResult
^^^^^^^^^^^^^^

.. autoclass:: src.features.pipeline.PipelineResult
   :members:
   :undoc-members:
   :show-inheritance:

Plugin Registry
---------------

PluginRegistry
^^^^^^^^^^^^^^

.. autoclass:: src.features.registry.PluginRegistry
   :members:
   :undoc-members:
   :show-inheritance:

Registry Functions
^^^^^^^^^^^^^^^^^^

.. autofunction:: src.features.registry.register_plugin

.. autofunction:: src.features.registry.get_plugin

Built-in Plugins
----------------

Temporal Features
^^^^^^^^^^^^^^^^^

.. autoclass:: src.features.plugins.temporal.TemporalFeaturesPlugin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.features.plugins.temporal.TemporalFeaturesConfig
   :members:
   :undoc-members:
   :show-inheritance:

Calendar Features
^^^^^^^^^^^^^^^^^

.. autoclass:: src.features.plugins.calendar.CalendarFeaturesPlugin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.features.plugins.calendar.CalendarFeaturesConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.features.plugins.calendar.BrazilianHolidays
   :members:
   :undoc-members:
   :show-inheritance:

Lag Features
^^^^^^^^^^^^

.. autoclass:: src.features.plugins.lag.LagFeaturesPlugin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.features.plugins.lag.LagFeaturesConfig
   :members:
   :undoc-members:
   :show-inheritance:

Cyclical Encoding
^^^^^^^^^^^^^^^^^

.. autoclass:: src.features.plugins.cyclical.CyclicalEncodingPlugin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.features.plugins.cyclical.CyclicalEncodingConfig
   :members:
   :undoc-members:
   :show-inheritance:

LOESS Smoothing
^^^^^^^^^^^^^^^

.. autoclass:: src.features.plugins.smoothing.LoessSmoothingPlugin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.features.plugins.smoothing.LoessSmoothingConfig
   :members:
   :undoc-members:
   :show-inheritance:

Wavelet Transform
^^^^^^^^^^^^^^^^^

.. autoclass:: src.features.plugins.wavelet.WaveletTransformPlugin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.features.plugins.wavelet.WaveletTransformConfig
   :members:
   :undoc-members:
   :show-inheritance:

BLF Strategy
^^^^^^^^^^^^

.. autoclass:: src.features.plugins.blf.BLFStrategyPlugin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.features.plugins.blf.BLFStrategyConfig
   :members:
   :undoc-members:
   :show-inheritance:

Validation
----------

FeatureValidator
^^^^^^^^^^^^^^^^

.. autoclass:: src.features.base.validator.FeatureValidator
   :members:
   :undoc-members:
   :show-inheritance:

ValidationResult
^^^^^^^^^^^^^^^^

.. autoclass:: src.features.base.validator.ValidationResult
   :members:
   :undoc-members:
   :show-inheritance:

ValidationReport
^^^^^^^^^^^^^^^^

.. autoclass:: src.features.base.validator.ValidationReport
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: src.features.utils.merge_feature_dataframes

.. autofunction:: src.features.utils.check_feature_names

.. autofunction:: src.features.utils.get_feature_statistics

.. autofunction:: src.features.utils.validate_dataframe_index

Exceptions
----------

.. autoexception:: src.features.registry.PluginRegistryError

.. autoexception:: src.features.registry.PluginNotFoundError

.. autoexception:: src.features.registry.DuplicatePluginError

.. autoexception:: src.features.registry.InvalidPluginNameError

.. autoexception:: src.features.registry.InvalidVersionError

.. autoexception:: src.features.pipeline.PipelineError

.. autoexception:: src.features.pipeline.PipelineConfigError

.. autoexception:: src.features.pipeline.PipelineExecutionError

.. autoexception:: src.features.pipeline.FeatureCollisionError

.. autoexception:: src.features.utils.FeatureUtilsError

.. autoexception:: src.features.utils.FeatureNameCollisionError

.. autoexception:: src.features.utils.InvalidIndexError
