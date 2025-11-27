Storage Module
==============

The storage module provides storage backends for data persistence,
supporting both local filesystem and AWS S3 storage.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The storage module implements:

- **Storage Factory**: Create storage backends from configuration
- **Local Backend**: File-based storage for development
- **S3 Backend**: AWS S3 storage for production
- **Abstract Interface**: Common interface for all backends

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   from src.storage import StorageFactory

   # Create local storage
   local_storage = StorageFactory.create(
       backend="local",
       base_path="./data"
   )

   # Create S3 storage
   s3_storage = StorageFactory.create(
       backend="s3",
       bucket="prevcarga-data",
       region="us-east-1"
   )

   # Use storage (same interface)
   data = storage.read("raw/SECO_RJ/2024-01-15.parquet")
   storage.write("predictions/2024-01-15.parquet", predictions)

Storage Factory
---------------

.. automodule:: src.storage
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from src.storage import StorageFactory

   # From configuration dictionary
   storage = StorageFactory.create(
       backend="s3",
       bucket="prevcarga-data",
       region="us-east-1",
       prefix="v1/"
   )

   # From YAML configuration
   storage = StorageFactory.from_config("config/storage.yaml")

Storage Interface
-----------------

All storage backends implement a common interface:

.. code-block:: python

   class StorageBackend:
       def read(self, path: str) -> bytes:
           """Read data from storage."""
           ...

       def write(self, path: str, data: bytes) -> None:
           """Write data to storage."""
           ...

       def exists(self, path: str) -> bool:
           """Check if path exists."""
           ...

       def list(self, prefix: str) -> List[str]:
           """List objects with prefix."""
           ...

       def delete(self, path: str) -> None:
           """Delete object at path."""
           ...

Local Storage Backend
---------------------

File-based storage for development and testing.

Example:

.. code-block:: python

   from src.storage import StorageFactory

   # Create local storage
   storage = StorageFactory.create(
       backend="local",
       base_path="./data"
   )

   # Read data
   data = storage.read("raw/SECO_RJ/2024-01-15.parquet")

   # Write data
   storage.write("predictions/forecast.parquet", predictions)

   # Check existence
   if storage.exists("models/lgbm/SECO_RJ.pkl"):
       model = storage.read("models/lgbm/SECO_RJ.pkl")

   # List files
   files = storage.list("raw/SECO_RJ/")
   print(f"Found {len(files)} files")

S3 Storage Backend
------------------

AWS S3 storage for production deployments.

Example:

.. code-block:: python

   from src.storage import StorageFactory

   # Create S3 storage
   storage = StorageFactory.create(
       backend="s3",
       bucket="prevcarga-data-production",
       region="us-east-1",
       prefix="prevcarga/v1/"
   )

   # Read data (same interface as local)
   data = storage.read("raw/SECO_RJ/2024-01-15.parquet")

   # Write predictions
   storage.write(
       "predictions/2024-01-15/SECO_RJ.parquet",
       predictions
   )

   # List models
   models = storage.list("models/lgbm/")

Configuration
-------------

Storage Configuration (YAML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # config/storage.yaml

   # Local storage (development)
   storage:
     backend: local
     local:
       base_path: ./data

   # S3 storage (production)
   storage:
     backend: s3
     s3:
       bucket: prevcarga-data-production
       region: us-east-1
       prefix: prevcarga/v1/

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``AWS_ACCESS_KEY_ID``
     - AWS access key for S3
   * - ``AWS_SECRET_ACCESS_KEY``
     - AWS secret key for S3
   * - ``AWS_DEFAULT_REGION``
     - Default AWS region
   * - ``AWS_PROFILE``
     - Named AWS profile
   * - ``PREVCARGA_STORAGE_BACKEND``
     - Storage backend type

Usage Patterns
--------------

Switching Between Backends
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import os
   from src.storage import StorageFactory

   # Use environment variable to switch
   backend = os.getenv("PREVCARGA_STORAGE_BACKEND", "local")

   if backend == "local":
       storage = StorageFactory.create(
           backend="local",
           base_path="./data"
       )
   else:
       storage = StorageFactory.create(
           backend="s3",
           bucket=os.getenv("PREVCARGA_S3_BUCKET"),
           region=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
       )

   # Code works with both backends
   data = storage.read("raw/SECO_RJ/2024-01-15.parquet")

Data Organization
^^^^^^^^^^^^^^^^^

Recommended directory structure:

.. code-block:: text

   data/
   ├── raw/                  # Raw input data
   │   ├── SECO_RJ/
   │   │   ├── 2024-01-01.parquet
   │   │   └── 2024-01-02.parquet
   │   └── SECO_SP/
   │       └── ...
   ├── features/             # Engineered features
   │   └── SECO_RJ/
   │       └── 2024-01-01.parquet
   ├── models/               # Trained models
   │   ├── lgbm/
   │   │   └── SECO_RJ.pkl
   │   └── rf/
   │       └── SECO_RJ.pkl
   └── predictions/          # Generated predictions
       └── 2024-01-15/
           └── SECO_RJ.parquet

Caching
^^^^^^^

.. code-block:: python

   from src.storage import StorageFactory
   from functools import lru_cache

   storage = StorageFactory.create(backend="s3", bucket="...")

   @lru_cache(maxsize=100)
   def load_cached(path: str):
       """Cache frequently accessed data."""
       return storage.read(path)

   # First call fetches from S3
   data = load_cached("models/lgbm/SECO_RJ.pkl")

   # Subsequent calls use cache
   data = load_cached("models/lgbm/SECO_RJ.pkl")
