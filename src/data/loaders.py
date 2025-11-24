"""Data loaders for loading raw load data from storage backends.

This module provides the DataLoader class for loading electric load data
from various storage backends (S3, local filesystem) with support for
parallel loading and standardized output formatting.

Example:
    ```python
    from datetime import date
    from src.data.loaders import DataLoader
    from src.storage import StorageFactory

    # Using default backend from config
    loader = DataLoader()
    df = loader.load_carga(
        areas=["SP", "RJ"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )

    # Using custom backend
    backend = StorageFactory.from_path("s3://my-bucket/data/")
    loader = DataLoader(storage_backend=backend, max_workers=8)
    df = loader.load_carga(areas=["SP"], start_date=date(2024, 1, 1), end_date=date(2024, 1, 7))
    ```
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd

from src.data.utils import DateRangeGenerator
from src.storage import ObjectNotFoundError, StorageBackend, StorageFactory
from src.utils import get_logger

logger = get_logger(__name__)

# Standardized output column names
OUTPUT_COLUMNS = ["timestamp", "cod_area", "carga_mwh"]


class DataLoader:
    """Loader for raw electric load data from storage backends.

    This class provides methods for loading electric load data from storage,
    with support for parallel loading of multiple files and standardized
    output formatting.

    Attributes:
        storage_backend: The storage backend used for data retrieval.
        max_workers: Maximum number of parallel workers for file loading.

    Example:
        ```python
        from datetime import date
        from src.data.loaders import DataLoader

        # Initialize with default config
        loader = DataLoader()

        # Load data for multiple areas
        df = loader.load_carga(
            areas=["SP", "RJ", "MG"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        print(df.columns)  # ['timestamp', 'cod_area', 'carga_mwh']
        ```
    """

    def __init__(
        self,
        storage_backend: StorageBackend | None = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize the DataLoader.

        Args:
            storage_backend: Storage backend for data retrieval. If None,
                creates a backend using StorageFactory.from_config().
            max_workers: Maximum number of parallel workers for file loading.
                Defaults to 4.

        Example:
            ```python
            # Use default backend from config
            loader = DataLoader()

            # Use custom backend with more workers
            from src.storage import StorageFactory
            backend = StorageFactory.from_path("s3://bucket/")
            loader = DataLoader(storage_backend=backend, max_workers=8)
            ```
        """
        if storage_backend is None:
            logger.debug("Creating storage backend from config")
            self.storage_backend = StorageFactory.from_config()
        else:
            self.storage_backend = storage_backend

        self.max_workers = max_workers
        logger.info(
            "DataLoader initialized with %s backend, max_workers=%d",
            self.storage_backend.backend_type,
            self.max_workers,
        )

    def load_carga(
        self,
        areas: list[str],
        start_date: date,
        end_date: date,
        validate: bool = False,
    ) -> pd.DataFrame:
        """Load electric load (carga) data for specified areas and date range.

        Loads parquet files from storage for each area and date combination,
        combining them into a single DataFrame with standardized columns.
        Missing files are handled gracefully by logging a warning and
        continuing with available data.

        Args:
            areas: List of area codes to load (e.g., ["SP", "RJ", "MG"]).
            start_date: Start date of the data range (inclusive).
            end_date: End date of the data range (inclusive).
            validate: Whether to validate loaded data. Defaults to False
                (validation not yet implemented).

        Returns:
            DataFrame with columns [timestamp, cod_area, carga_mwh] containing
            the loaded data from all available files. Returns empty DataFrame
            with the correct columns if no files are found.

        Example:
            ```python
            loader = DataLoader()
            df = loader.load_carga(
                areas=["SP", "RJ"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 7),
            )

            # Access data
            sp_data = df[df["cod_area"] == "SP"]
            print(f"Loaded {len(df)} records")
            ```
        """
        logger.info(
            "Loading carga data for areas=%s from %s to %s",
            areas,
            start_date,
            end_date,
        )

        # Generate file paths
        paths = self._generate_paths("carga", areas, start_date, end_date)
        logger.debug("Generated %d file paths to load", len(paths))

        if not paths:
            logger.warning("No paths generated for the given parameters")
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        # Load files in parallel
        dataframes: list[pd.DataFrame] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._load_single_file, path): path for path in paths
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    df = future.result()
                    if df is not None:
                        dataframes.append(df)
                except Exception as e:
                    logger.error("Unexpected error loading %s: %s", path, e)

        if not dataframes:
            logger.warning("No data files were successfully loaded")
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        # Combine all dataframes
        result = pd.concat(dataframes, ignore_index=True)
        logger.info(
            "Loaded %d records from %d files",
            len(result),
            len(dataframes),
        )

        # Validate if requested (placeholder for future implementation)
        if validate:
            logger.warning("Validation requested but not yet implemented")

        return result

    def _load_single_file(self, path: str) -> pd.DataFrame | None:
        """Load a single parquet file from storage.

        Args:
            path: Path to the parquet file in storage.

        Returns:
            DataFrame with the file contents, or None if the file
            doesn't exist or cannot be loaded.
        """
        try:
            logger.debug("Loading file: %s", path)
            return self.storage_backend.get_parquet(path)
        except ObjectNotFoundError:
            logger.warning("File not found: %s", path)
            return None
        except Exception as e:
            logger.error("Error loading file %s: %s", path, e)
            return None

    def _generate_paths(
        self,
        prefix: str,
        areas: list[str],
        start_date: date,
        end_date: date,
    ) -> list[str]:
        """Generate storage paths for the given areas and date range.

        Generates paths following the pattern:
        raw_data/{year}/{month:02d}/{prefix}_{AREA}_{YYYYMMDD}.parquet

        Args:
            prefix: File prefix (e.g., "carga").
            areas: List of area codes.
            start_date: Start date of the range.
            end_date: End date of the range.

        Returns:
            List of storage paths to load.

        Example:
            ```python
            paths = loader._generate_paths(
                prefix="carga",
                areas=["SP"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 2),
            )
            # Returns:
            # [
            #     "raw_data/2024/01/carga_SP_20240101.parquet",
            #     "raw_data/2024/01/carga_SP_20240102.parquet",
            # ]
            ```
        """
        paths: list[str] = []

        date_range = DateRangeGenerator(start_date, end_date)

        for current_date in date_range:
            year = current_date.year
            month = current_date.month
            date_str = current_date.strftime("%Y%m%d")

            for area in areas:
                path = f"raw_data/{year}/{month:02d}/{prefix}_{area}_{date_str}.parquet"
                paths.append(path)

        return paths
