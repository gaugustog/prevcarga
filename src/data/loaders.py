"""Data loaders for loading raw load data from storage backends.

This module provides the DataLoader class for loading electric load data
from various storage backends (S3, local filesystem) with support for
parallel loading and standardized output formatting. It also includes
MultiFormatLoader for handling CSV and Parquet files, and DataJoiner
for combining data from multiple sources.

Example:
    ```python
    from datetime import date
    from src.data.loaders import DataLoader, MultiFormatLoader, DataJoiner
    from src.storage import StorageFactory

    # Using default backend from config
    loader = DataLoader()
    df = loader.load_carga(
        areas=["SP", "RJ"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )

    # Load temperature data
    df_temp = loader.load_temperatura_d1(
        areas=["SP"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 7),
    )

    # Load holidays
    df_holidays = loader.load_feriados(years=[2024])

    # Join data sources
    df_joined = DataJoiner.join_load_and_temperature(df, df_temp)

    # Using custom backend
    backend = StorageFactory.from_path("s3://my-bucket/data/")
    loader = DataLoader(storage_backend=backend, max_workers=8)
    df = loader.load_carga(areas=["SP"], start_date=date(2024, 1, 1), end_date=date(2024, 1, 7))
    ```
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Literal

import pandas as pd

from src.data.utils import DateRangeGenerator
from src.storage import ObjectNotFoundError, StorageBackend, StorageFactory
from src.utils import get_logger

logger = get_logger(__name__)

# Standardized output column names
OUTPUT_COLUMNS = ["timestamp", "cod_area", "carga_mwh"]

# Temperature output columns
TEMPERATURE_OUTPUT_COLUMNS = ["timestamp", "cod_area", "temp_celsius", "source"]

# Heat index output columns
HEAT_INDEX_OUTPUT_COLUMNS = ["timestamp", "cod_area", "heat_index"]

# Holiday output columns
HOLIDAY_OUTPUT_COLUMNS = ["date", "holiday_name", "holiday_type", "id_tipodiaespecial"]


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
            future_to_path = {executor.submit(self._load_single_file, path): path for path in paths}

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

    def load_temperatura_d1(
        self,
        areas: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Load D+1 temperature forecast data for specified areas and date range.

        Loads parquet files from storage containing day-ahead temperature
        forecasts for each area and date combination.

        Args:
            areas: List of area codes to load (e.g., ["SP", "RJ", "MG"]).
            start_date: Start date of the data range (inclusive).
            end_date: End date of the data range (inclusive).

        Returns:
            DataFrame with columns [timestamp, cod_area, temp_celsius, source]
            containing the loaded temperature data. Returns empty DataFrame
            with the correct columns if no files are found.

        Example:
            ```python
            loader = DataLoader()
            df = loader.load_temperatura_d1(
                areas=["SP", "RJ"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 7),
            )
            ```
        """
        return self._load_temperatura(areas, start_date, end_date, horizon=1)

    def load_temperatura_d2(
        self,
        areas: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Load D+2 temperature forecast data for specified areas and date range.

        Loads parquet files from storage containing two-day-ahead temperature
        forecasts for each area and date combination.

        Args:
            areas: List of area codes to load (e.g., ["SP", "RJ", "MG"]).
            start_date: Start date of the data range (inclusive).
            end_date: End date of the data range (inclusive).

        Returns:
            DataFrame with columns [timestamp, cod_area, temp_celsius, source]
            containing the loaded temperature data. Returns empty DataFrame
            with the correct columns if no files are found.

        Example:
            ```python
            loader = DataLoader()
            df = loader.load_temperatura_d2(
                areas=["SP", "RJ"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 7),
            )
            ```
        """
        return self._load_temperatura(areas, start_date, end_date, horizon=2)

    def _load_temperatura(
        self,
        areas: list[str],
        start_date: date,
        end_date: date,
        horizon: int,
    ) -> pd.DataFrame:
        """Load temperature forecast data for a specific horizon.

        Internal method that handles loading temperature data for D+1 or D+2
        forecasts with standardized column formatting.

        Args:
            areas: List of area codes to load.
            start_date: Start date of the data range (inclusive).
            end_date: End date of the data range (inclusive).
            horizon: Forecast horizon (1 for D+1, 2 for D+2).

        Returns:
            DataFrame with standardized temperature columns.
        """
        logger.info(
            "Loading temperatura_d%d data for areas=%s from %s to %s",
            horizon,
            areas,
            start_date,
            end_date,
        )

        # Generate file paths
        paths = self._generate_temperatura_paths(areas, start_date, end_date, horizon)
        logger.debug("Generated %d temperature file paths to load", len(paths))

        if not paths:
            logger.warning("No paths generated for the given parameters")
            return pd.DataFrame(columns=TEMPERATURE_OUTPUT_COLUMNS)

        # Load files in parallel
        dataframes: list[pd.DataFrame] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(self._load_single_file, path): path for path in paths}

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    df = future.result()
                    if df is not None:
                        # Standardize columns and add source
                        df = self._standardize_temperatura_columns(df, horizon)
                        dataframes.append(df)
                except Exception as e:
                    logger.error("Unexpected error loading %s: %s", path, e)

        if not dataframes:
            logger.warning("No temperature data files were successfully loaded")
            return pd.DataFrame(columns=TEMPERATURE_OUTPUT_COLUMNS)

        # Combine all dataframes
        result = pd.concat(dataframes, ignore_index=True)
        logger.info(
            "Loaded %d temperature records from %d files",
            len(result),
            len(dataframes),
        )

        return result

    def _generate_temperatura_paths(
        self,
        areas: list[str],
        start_date: date,
        end_date: date,
        horizon: int,
    ) -> list[str]:
        """Generate storage paths for temperature data.

        Generates paths following the pattern:
        raw_data/{year}/{month:02d}/temperatura_d{horizon}_{AREA}_{YYYYMMDD}.parquet

        Args:
            areas: List of area codes.
            start_date: Start date of the range.
            end_date: End date of the range.
            horizon: Forecast horizon (1 or 2).

        Returns:
            List of storage paths to load.
        """
        paths: list[str] = []

        date_range = DateRangeGenerator(start_date, end_date)

        for current_date in date_range:
            year = current_date.year
            month = current_date.month
            date_str = current_date.strftime("%Y%m%d")

            for area in areas:
                path = (
                    f"raw_data/{year}/{month:02d}/"
                    f"temperatura_d{horizon}_{area}_{date_str}.parquet"
                )
                paths.append(path)

        return paths

    def _standardize_temperatura_columns(
        self,
        df: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """Standardize temperature DataFrame columns.

        Maps various column name formats to the standard output format
        and adds source information.

        Args:
            df: Raw temperature DataFrame.
            horizon: Forecast horizon for source labeling.

        Returns:
            DataFrame with standardized columns.
        """
        result = df.copy()

        # Map common column name variations
        column_mapping = {
            "data_hora": "timestamp",
            "datahora": "timestamp",
            "dt_timestamp": "timestamp",
            "area": "cod_area",
            "codigo_area": "cod_area",
            "temperatura": "temp_celsius",
            "temperatura_celsius": "temp_celsius",
            "temp": "temp_celsius",
        }

        for old_col, new_col in column_mapping.items():
            if old_col in result.columns and new_col not in result.columns:
                result = result.rename(columns={old_col: new_col})

        # Add source column
        source_value = f"D+{horizon}"
        result["source"] = source_value

        # Ensure all required columns exist
        for col in TEMPERATURE_OUTPUT_COLUMNS:
            if col not in result.columns:
                logger.warning("Missing column %s in temperature data", col)

        # Return only the standard columns that exist
        available_cols = [c for c in TEMPERATURE_OUTPUT_COLUMNS if c in result.columns]
        return result[available_cols]

    def load_heat_index(
        self,
        areas: list[str],
        file_format: Literal["csv", "parquet"] = "csv",
    ) -> pd.DataFrame:
        """Load heat index data from heatindex files.

        Loads heat index data files for the specified areas. Heat index files
        are stored per-area without date partitioning.

        Args:
            areas: List of area codes to load (e.g., ["SP", "RJ"]).
            file_format: File format to load ("csv" or "parquet"). Defaults to "csv".

        Returns:
            DataFrame with columns [timestamp, cod_area, heat_index]
            containing the heat index data. Returns empty DataFrame
            with the correct columns if no files are found.

        Example:
            ```python
            loader = DataLoader()
            df = loader.load_heat_index(areas=["SP", "RJ"], file_format="csv")
            ```
        """
        logger.info(
            "Loading heat index data for areas=%s, format=%s",
            areas,
            file_format,
        )

        if not areas:
            logger.warning("No areas specified for heat index loading")
            return pd.DataFrame(columns=HEAT_INDEX_OUTPUT_COLUMNS)

        # Create multi-format loader for handling different file formats
        multi_loader = MultiFormatLoader(self.storage_backend)

        dataframes: list[pd.DataFrame] = []

        for area in areas:
            path = f"raw_data/heatindex_{area}.{file_format}"
            try:
                df = multi_loader.load(path, file_format=file_format)
                if df is not None and not df.empty:
                    df = self._standardize_heat_index_columns(df, area)
                    dataframes.append(df)
            except Exception as e:
                logger.warning("Failed to load heat index for area %s: %s", area, e)

        if not dataframes:
            logger.warning("No heat index data files were successfully loaded")
            return pd.DataFrame(columns=HEAT_INDEX_OUTPUT_COLUMNS)

        result = pd.concat(dataframes, ignore_index=True)
        logger.info("Loaded %d heat index records", len(result))

        return result

    def load_temperatura_ecmwf(
        self,
        areas: list[str],
        file_format: Literal["csv", "parquet"] = "csv",
    ) -> pd.DataFrame:
        """Load ECMWF temperature data from heatindex files.

        Loads ECMWF (European Centre for Medium-Range Weather Forecasts)
        temperature data from the heatindex file set.

        Args:
            areas: List of area codes to load (e.g., ["SP", "RJ"]).
            file_format: File format to load ("csv" or "parquet"). Defaults to "csv".

        Returns:
            DataFrame with columns [timestamp, cod_area, temp_celsius, source]
            containing the ECMWF temperature data. Returns empty DataFrame
            with the correct columns if no files are found.

        Example:
            ```python
            loader = DataLoader()
            df = loader.load_temperatura_ecmwf(areas=["SP", "RJ"], file_format="csv")
            ```
        """
        logger.info(
            "Loading ECMWF temperature data for areas=%s, format=%s",
            areas,
            file_format,
        )

        if not areas:
            logger.warning("No areas specified for ECMWF temperature loading")
            return pd.DataFrame(columns=TEMPERATURE_OUTPUT_COLUMNS)

        # Create multi-format loader for handling different file formats
        multi_loader = MultiFormatLoader(self.storage_backend)

        dataframes: list[pd.DataFrame] = []

        for area in areas:
            path = f"raw_data/heatindex_{area}.{file_format}"
            try:
                df = multi_loader.load(path, file_format=file_format)
                if df is not None and not df.empty:
                    df = self._standardize_ecmwf_columns(df, area)
                    dataframes.append(df)
            except Exception as e:
                logger.warning("Failed to load ECMWF temperature for area %s: %s", area, e)

        if not dataframes:
            logger.warning("No ECMWF temperature data files were successfully loaded")
            return pd.DataFrame(columns=TEMPERATURE_OUTPUT_COLUMNS)

        result = pd.concat(dataframes, ignore_index=True)
        logger.info("Loaded %d ECMWF temperature records", len(result))

        return result

    def _standardize_heat_index_columns(
        self,
        df: pd.DataFrame,
        area: str,
    ) -> pd.DataFrame:
        """Standardize heat index DataFrame columns.

        Args:
            df: Raw heat index DataFrame.
            area: Area code to add if not present.

        Returns:
            DataFrame with standardized columns.
        """
        result = df.copy()

        # Map common column name variations
        column_mapping = {
            "data_hora": "timestamp",
            "datahora": "timestamp",
            "dt_timestamp": "timestamp",
            "area": "cod_area",
            "codigo_area": "cod_area",
            "indice_calor": "heat_index",
            "hi": "heat_index",
        }

        for old_col, new_col in column_mapping.items():
            if old_col in result.columns and new_col not in result.columns:
                result = result.rename(columns={old_col: new_col})

        # Add cod_area if not present
        if "cod_area" not in result.columns:
            result["cod_area"] = area

        # Return only the standard columns that exist
        available_cols = [c for c in HEAT_INDEX_OUTPUT_COLUMNS if c in result.columns]
        return result[available_cols]

    def _standardize_ecmwf_columns(
        self,
        df: pd.DataFrame,
        area: str,
    ) -> pd.DataFrame:
        """Standardize ECMWF temperature DataFrame columns.

        Args:
            df: Raw ECMWF temperature DataFrame.
            area: Area code to add if not present.

        Returns:
            DataFrame with standardized temperature columns.
        """
        result = df.copy()

        # Map common column name variations
        column_mapping = {
            "data_hora": "timestamp",
            "datahora": "timestamp",
            "dt_timestamp": "timestamp",
            "area": "cod_area",
            "codigo_area": "cod_area",
            "temperatura_ecmwf": "temp_celsius",
            "temp_ecmwf": "temp_celsius",
            "ecmwf": "temp_celsius",
            "temperatura": "temp_celsius",
        }

        for old_col, new_col in column_mapping.items():
            if old_col in result.columns and new_col not in result.columns:
                result = result.rename(columns={old_col: new_col})

        # Add cod_area if not present
        if "cod_area" not in result.columns:
            result["cod_area"] = area

        # Add source column
        result["source"] = "ECMWF"

        # Return only the standard columns that exist
        available_cols = [c for c in TEMPERATURE_OUTPUT_COLUMNS if c in result.columns]
        return result[available_cols]

    def load_feriados(
        self,
        years: list[int],
    ) -> pd.DataFrame:
        """Load holiday calendar data for specified years.

        Loads holiday (feriado) data from parquet files containing
        holiday information for load forecasting adjustments.

        Args:
            years: List of years to load holiday data for (e.g., [2023, 2024]).

        Returns:
            DataFrame with columns [date, holiday_name, holiday_type, id_tipodiaespecial]
            containing the holiday data. Returns empty DataFrame
            with the correct columns if no files are found.

        Example:
            ```python
            loader = DataLoader()
            df = loader.load_feriados(years=[2023, 2024])
            ```
        """
        logger.info("Loading feriados data for years=%s", years)

        if not years:
            logger.warning("No years specified for holiday loading")
            return pd.DataFrame(columns=HOLIDAY_OUTPUT_COLUMNS)

        dataframes: list[pd.DataFrame] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_year = {
                executor.submit(self._load_single_file, f"raw_data/feriados_{year}.parquet"): year
                for year in years
            }

            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    df = future.result()
                    if df is not None:
                        df = self._standardize_feriados_columns(df)
                        dataframes.append(df)
                except Exception as e:
                    logger.error("Unexpected error loading feriados for %d: %s", year, e)

        if not dataframes:
            logger.warning("No holiday data files were successfully loaded")
            return pd.DataFrame(columns=HOLIDAY_OUTPUT_COLUMNS)

        result = pd.concat(dataframes, ignore_index=True)
        logger.info("Loaded %d holiday records", len(result))

        return result

    def _standardize_feriados_columns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Standardize holiday DataFrame columns.

        Args:
            df: Raw holiday DataFrame.

        Returns:
            DataFrame with standardized columns.
        """
        result = df.copy()

        # Map common column name variations
        column_mapping = {
            "data": "date",
            "dt_feriado": "date",
            "data_feriado": "date",
            "nome": "holiday_name",
            "nome_feriado": "holiday_name",
            "descricao": "holiday_name",
            "tipo": "holiday_type",
            "tipo_feriado": "holiday_type",
            "id_tipo": "id_tipodiaespecial",
            "tipo_dia_especial": "id_tipodiaespecial",
        }

        for old_col, new_col in column_mapping.items():
            if old_col in result.columns and new_col not in result.columns:
                result = result.rename(columns={old_col: new_col})

        # Return only the standard columns that exist
        available_cols = [c for c in HOLIDAY_OUTPUT_COLUMNS if c in result.columns]
        return result[available_cols]


class MultiFormatLoader:
    """Loader for files in multiple formats (CSV, Parquet).

    Provides a unified interface for loading data files in various formats,
    with support for auto-detecting format from file extensions and handling
    different encodings for CSV files.

    Attributes:
        storage_backend: The storage backend used for data retrieval.

    Example:
        ```python
        from src.storage import StorageFactory
        from src.data.loaders import MultiFormatLoader

        backend = StorageFactory.from_path("s3://my-bucket/")
        loader = MultiFormatLoader(backend)

        # Load with explicit format
        df = loader.load("data/file.csv", file_format="csv")

        # Auto-detect format from extension
        df = loader.load("data/file.parquet")
        ```
    """

    # Supported file formats
    SUPPORTED_FORMATS = ("csv", "parquet")

    # Default encodings to try for CSV files
    CSV_ENCODINGS = ("utf-8", "latin-1", "iso-8859-1", "cp1252")

    def __init__(self, storage_backend: StorageBackend) -> None:
        """Initialize the MultiFormatLoader.

        Args:
            storage_backend: Storage backend for data retrieval.
        """
        self.storage_backend = storage_backend
        logger.debug(
            "MultiFormatLoader initialized with %s backend",
            storage_backend.backend_type,
        )

    def load(
        self,
        path: str,
        file_format: Literal["csv", "parquet"] | None = None,
        encoding: str | None = None,
    ) -> pd.DataFrame | None:
        """Load a file from storage in the specified or auto-detected format.

        Args:
            path: Path to the file in storage.
            file_format: File format ("csv" or "parquet"). If None, auto-detects
                from file extension.
            encoding: Character encoding for CSV files. If None, tries multiple
                encodings (UTF-8, Latin-1, etc.).

        Returns:
            DataFrame with the file contents, or None if the file
            doesn't exist or cannot be loaded.

        Raises:
            ValueError: If file format is not supported or cannot be detected.

        Example:
            ```python
            loader = MultiFormatLoader(backend)

            # Explicit format
            df = loader.load("data/file.csv", file_format="csv")

            # Auto-detect with specific encoding
            df = loader.load("data/file.csv", encoding="latin-1")

            # Parquet file
            df = loader.load("data/file.parquet")
            ```
        """
        # Auto-detect format if not specified
        if file_format is None:
            file_format = self._detect_format(path)

        if file_format not in self.SUPPORTED_FORMATS:
            msg = (
                f"Unsupported file format: {file_format}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
            raise ValueError(msg)

        logger.debug("Loading file %s with format %s", path, file_format)

        try:
            if file_format == "parquet":
                return self._load_parquet(path)
            # csv format
            return self._load_csv(path, encoding)
        except ObjectNotFoundError:
            logger.warning("File not found: %s", path)
            return None
        except Exception as e:
            logger.error("Error loading file %s: %s", path, e)
            return None

    def _detect_format(self, path: str) -> Literal["csv", "parquet"]:
        """Detect file format from path extension.

        Args:
            path: File path to analyze.

        Returns:
            Detected file format.

        Raises:
            ValueError: If format cannot be detected.
        """
        path_lower = path.lower()

        if path_lower.endswith(".parquet"):
            return "parquet"
        if path_lower.endswith(".csv"):
            return "csv"
        msg = (
            f"Cannot detect file format from path: {path}. " f"Supported extensions: .csv, .parquet"
        )
        raise ValueError(msg)

    def _load_parquet(self, path: str) -> pd.DataFrame:
        """Load a Parquet file from storage.

        Args:
            path: Path to the Parquet file.

        Returns:
            DataFrame with the file contents.
        """
        return self.storage_backend.get_parquet(path)

    def _load_csv(
        self,
        path: str,
        encoding: str | None = None,
    ) -> pd.DataFrame:
        """Load a CSV file from storage with encoding handling.

        Tries multiple encodings if not specified, to handle files
        with different character encodings.

        Args:
            path: Path to the CSV file.
            encoding: Specific encoding to use, or None to auto-detect.

        Returns:
            DataFrame with the file contents.

        Raises:
            ValueError: If file cannot be decoded with any encoding.
        """
        data = self.storage_backend.get_object(path)

        if encoding is not None:
            # Use specified encoding
            return pd.read_csv(io.BytesIO(data), encoding=encoding)

        # Try multiple encodings
        last_error: Exception | None = None
        for enc in self.CSV_ENCODINGS:
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except UnicodeDecodeError as e:
                last_error = e
                logger.debug("Encoding %s failed for %s: %s", enc, path, e)
                continue

        # If all encodings fail, raise the last error
        msg = f"Failed to decode CSV file {path} with any encoding"
        raise ValueError(msg) from last_error


class DataJoiner:
    """Utility class for joining data from multiple sources.

    Provides static methods for combining load data with auxiliary
    data sources such as temperature and holiday information.

    Example:
        ```python
        from src.data.loaders import DataLoader, DataJoiner

        loader = DataLoader()
        df_load = loader.load_carga(areas=["SP"], start_date=date(2024, 1, 1), end_date=date(2024, 1, 7))
        df_temp = loader.load_temperatura_d1(areas=["SP"], start_date=date(2024, 1, 1), end_date=date(2024, 1, 7))
        df_holidays = loader.load_feriados(years=[2024])

        # Join load with temperature
        df_joined = DataJoiner.join_load_and_temperature(df_load, df_temp)

        # Join with holidays
        df_with_holidays = DataJoiner.join_with_holidays(df_joined, df_holidays)

        # Or join all at once
        df_full = DataJoiner.join_all_sources(df_load, [df_temp], None, df_holidays)
        ```
    """

    @staticmethod
    def join_load_and_temperature(
        df_load: pd.DataFrame,
        df_temp: pd.DataFrame,
        how: Literal["left", "right", "inner", "outer"] = "left",
    ) -> pd.DataFrame:
        """Join load data with temperature data.

        Joins DataFrames on timestamp and cod_area columns using the
        specified join method.

        Args:
            df_load: Load DataFrame with columns [timestamp, cod_area, ...].
            df_temp: Temperature DataFrame with columns [timestamp, cod_area, temp_celsius, source].
            how: Join method ("left", "right", "inner", "outer"). Defaults to "left".

        Returns:
            Joined DataFrame containing columns from both DataFrames.

        Example:
            ```python
            df_joined = DataJoiner.join_load_and_temperature(df_load, df_temp, how="left")
            ```
        """
        if df_load.empty:
            logger.warning("Load DataFrame is empty, returning empty result")
            return df_load.copy()

        if df_temp.empty:
            logger.warning("Temperature DataFrame is empty, returning load data only")
            return df_load.copy()

        # Ensure timestamp columns are datetime type
        df_load = df_load.copy()
        df_temp = df_temp.copy()

        if "timestamp" in df_load.columns:
            df_load["timestamp"] = pd.to_datetime(df_load["timestamp"])
        if "timestamp" in df_temp.columns:
            df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])

        # Perform the join
        result = df_load.merge(
            df_temp,
            on=["timestamp", "cod_area"],
            how=how,
            suffixes=("", "_temp"),
        )

        logger.info(
            "Joined load (%d rows) with temperature (%d rows) -> %d rows",
            len(df_load),
            len(df_temp),
            len(result),
        )

        return result

    @staticmethod
    def join_with_holidays(
        df: pd.DataFrame,
        df_holidays: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Join a DataFrame with holiday information.

        Extracts the date from the timestamp column and joins with
        holiday data to add holiday information to each record.

        Args:
            df: DataFrame with a timestamp column.
            df_holidays: Holiday DataFrame with columns [date, holiday_name, ...].
            timestamp_col: Name of the timestamp column in df. Defaults to "timestamp".

        Returns:
            DataFrame with holiday columns added (holiday_name, holiday_type, etc.).
            Non-holiday dates will have NaN values for holiday columns.

        Example:
            ```python
            df_with_holidays = DataJoiner.join_with_holidays(
                df,
                df_holidays,
                timestamp_col="timestamp",
            )
            ```
        """
        if df.empty:
            logger.warning("Input DataFrame is empty, returning empty result")
            return df.copy()

        if df_holidays.empty:
            logger.warning("Holiday DataFrame is empty, returning input data only")
            return df.copy()

        df = df.copy()
        df_holidays = df_holidays.copy()

        # Ensure timestamp column is datetime and extract date
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df["_join_date"] = df[timestamp_col].dt.date

        # Ensure holiday date column is date type
        if "date" in df_holidays.columns:
            df_holidays["date"] = pd.to_datetime(df_holidays["date"]).dt.date

        # Perform the join
        result = df.merge(
            df_holidays,
            left_on="_join_date",
            right_on="date",
            how="left",
            suffixes=("", "_holiday"),
        )

        # Remove the temporary join column
        result = result.drop(columns=["_join_date"], errors="ignore")

        logger.info(
            "Joined data (%d rows) with holidays (%d holidays) -> %d rows",
            len(df),
            len(df_holidays),
            len(result),
        )

        return result

    @staticmethod
    def join_all_sources(
        df_load: pd.DataFrame,
        df_temps: list[pd.DataFrame] | None = None,
        df_heat: pd.DataFrame | None = None,
        df_holidays: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Join load data with all available auxiliary data sources.

        Convenience method that combines load data with multiple temperature
        sources, heat index, and holiday information in a single call.

        Args:
            df_load: Load DataFrame with columns [timestamp, cod_area, carga_mwh].
            df_temps: Optional list of temperature DataFrames to join.
            df_heat: Optional heat index DataFrame to join.
            df_holidays: Optional holiday DataFrame to join.

        Returns:
            DataFrame with all data sources joined together.

        Example:
            ```python
            df_full = DataJoiner.join_all_sources(
                df_load=df_load,
                df_temps=[df_temp_d1, df_temp_d2],
                df_heat=df_heat_index,
                df_holidays=df_holidays,
            )
            ```
        """
        if df_load.empty:
            logger.warning("Load DataFrame is empty, returning empty result")
            return df_load.copy()

        result = df_load.copy()

        # Ensure timestamp is datetime
        if "timestamp" in result.columns:
            result["timestamp"] = pd.to_datetime(result["timestamp"])

        # Join temperature DataFrames
        if df_temps:
            for i, df_temp in enumerate(df_temps):
                if df_temp is not None and not df_temp.empty:
                    # Create unique suffixes for each temperature source
                    df_temp_copy = df_temp.copy()
                    if "timestamp" in df_temp_copy.columns:
                        df_temp_copy["timestamp"] = pd.to_datetime(df_temp_copy["timestamp"])

                    # Get the source value if available for suffix
                    source_suffix = ""
                    if "source" in df_temp_copy.columns:
                        unique_sources = df_temp_copy["source"].unique()
                        if len(unique_sources) == 1:
                            source_suffix = f"_{unique_sources[0].replace('+', 'plus')}"

                    result = result.merge(
                        df_temp_copy,
                        on=["timestamp", "cod_area"],
                        how="left",
                        suffixes=("", source_suffix if source_suffix else f"_temp{i}"),
                    )

        # Join heat index
        if df_heat is not None and not df_heat.empty:
            df_heat_copy = df_heat.copy()
            if "timestamp" in df_heat_copy.columns:
                df_heat_copy["timestamp"] = pd.to_datetime(df_heat_copy["timestamp"])

            result = result.merge(
                df_heat_copy,
                on=["timestamp", "cod_area"],
                how="left",
                suffixes=("", "_heat"),
            )

        # Join holidays
        if df_holidays is not None and not df_holidays.empty:
            result = DataJoiner.join_with_holidays(result, df_holidays)

        logger.info(
            "Joined all sources: %d rows, %d columns",
            len(result),
            len(result.columns),
        )

        return result
