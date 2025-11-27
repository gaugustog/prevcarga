User Guides
===========

This section contains comprehensive guides for using PrevCarga effectively.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   installation
   quick-start
   usage
   troubleshooting

.. contents:: Quick Links
   :local:
   :depth: 1

Overview
--------

PrevCarga provides extensive documentation to help you get started and become productive quickly:

- **Installation Guide**: Complete setup instructions for development, Docker, and production environments
- **Quick Start Guide**: Get up and running with PrevCarga in 5 minutes
- **Usage Guide**: Complete CLI and configuration reference
- **Troubleshooting Guide**: Solutions for common issues and problems

Getting Started
---------------

New to PrevCarga? Follow these steps:

1. **Install PrevCarga**: Follow the :doc:`installation` guide for your environment
2. **Run Your First Prediction**: Use the :doc:`quick-start` guide to generate forecasts
3. **Explore Features**: Check the :doc:`../api/index` for available features
4. **Solve Issues**: Refer to :doc:`troubleshooting` if you encounter problems

Key Topics
----------

Installation and Setup
^^^^^^^^^^^^^^^^^^^^^^

- System requirements and prerequisites - :doc:`installation`
- Development installation with uv - :doc:`installation`
- Docker containerized deployment - :doc:`installation`
- AWS S3 backend configuration - :doc:`installation`

Quick Start Examples
^^^^^^^^^^^^^^^^^^^^

- Basic prediction workflow - :doc:`quick-start`
- Model training - :doc:`quick-start`
- Backtesting - :doc:`quick-start`
- Python API usage - :doc:`quick-start`

Troubleshooting Topics
^^^^^^^^^^^^^^^^^^^^^^

- Installation issues - :doc:`troubleshooting`
- AWS and S3 problems - :doc:`troubleshooting`
- Runtime and performance - :doc:`troubleshooting`
- Docker deployment - :doc:`troubleshooting`

Environment Variables
---------------------

PrevCarga uses environment variables for configuration:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``AWS_ACCESS_KEY_ID``
     - AWS access key for S3 storage
   * - ``AWS_SECRET_ACCESS_KEY``
     - AWS secret key for S3 storage
   * - ``AWS_DEFAULT_REGION``
     - Default AWS region (e.g., us-east-1)
   * - ``AWS_PROFILE``
     - AWS named profile to use
   * - ``PREVCARGA_LOG_LEVEL``
     - Logging level (DEBUG, INFO, WARNING, ERROR)
   * - ``PREVCARGA_STORAGE_BACKEND``
     - Storage backend type (local, s3)

Configuration Files
-------------------

Configuration can be provided via YAML files:

.. code-block:: yaml

   # config/config.yaml
   storage:
     backend: local
     local:
       base_path: ./data

   logging:
     level: INFO
     format: json

   features:
     plugins:
       - temporal
       - calendar
       - lag

   models:
     default: lgbm
     enabled:
       - lgbm
       - random_forest

Getting Help
------------

- Check the :doc:`../api/index` for API documentation
- Review :doc:`../tutorials/index` for step-by-step tutorials
- Visit `GitHub Issues <https://github.com/gaugustog/prevcarga/issues>`_ to report bugs
- Ask questions in `GitHub Discussions <https://github.com/gaugustog/prevcarga/discussions>`_
